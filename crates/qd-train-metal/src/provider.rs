//! qd-train's [`StepProvider`] for ojas-qwen35's [`Qwen35Step`]: a Qwen3.5 text tower on
//! canonical tessl's Metal kernels.
//!
//! The adapter is thin on purpose. ojas-qwen35 already does the step, the bank, the norm and
//! AdamW, and refuses bad input on the host before any GPU work. This file only translates
//! between the two contracts, and refuses where they differ:
//!
//! * **The bank.** qd-train passes [`BankMode::Overwrite`] on a step's first sequence and
//!   [`BankMode::Add`] after it. ojas overwrites an empty bank and adds to a holding one. So
//!   `Overwrite` on a bank that still holds sequences (a step abandoned part-way) discards them
//!   first, and `Add` on an empty bank is refused: it would mean the loop lost count.
//! * **The outside loss.** The span head's positions are the job's `hidden_at`, which qd-train
//!   already requires to be distinct (ojas's `ExternalGrad` does too). The final-norm rows go to
//!   the callback, and its gradient goes back as a host f32 tensor charged to the provider's
//!   budget.
//! * **The optimizer.** ojas-qwen35's `Qwen35Step::adamw_step` takes an `OptimizerPlan` of
//!   per-entry learning-rate scales and weight decays and runs tessl's `adamw_step_scaled`
//!   (entry `i` learns at `lr * lr_scale[i]`), so the layer-wise split of F's recipe is the
//!   caller's own `lr_scale` vector, passed through whole (board-sweep ruling 2h, 2026-10-12;
//!   it retires ruling 6's refusal). The scales and the weight decays are the caller's
//!   per-entry vectors. They are grouped by value into the plan's rules ([`build_plan`]), with
//!   no rule of this crate's choosing, and ojas-qwen35 refuses a scale that is negative or not
//!   finite. tessl takes them as f32: f32(0.01) is 0.00999999977648258, 2.2e-8
//!   relative below 0.01. The clip coefficient is f32 too in ojas-qwen35's `AdamWHyper`, and
//!   torch's `clip_grad_norm_` also forms it in the gradients' dtype. Both roundings are named
//!   in [`StepProvider::describe`]'s text, so the ledger row carries them.
//! * **Every entry steps on every step.** tessl's AdamW updates every tower tensor with one
//!   model-wide count. That is torch's per-parameter AdamW only while each tower tensor has a
//!   gradient on every step, which the torch reference shows for rung (b): every tower entry
//!   took 20 AdamW steps of 20 (`tiny-published/manifest.json` `per_parameter`), the three
//!   span-only batches included. That ojas-qwen35 also writes a gradient into every entry on a
//!   sequence with no letter rows is [inferred] from the tower's forward path, not measured: it
//!   is checked when rung (b)'s parity tests run. The span head, which a letter-only batch never
//!   reaches, is a host entry and steps on its own count.
//! * **State.** ojas-qwen35 writes a state directory that must not exist, while qd-train hands
//!   over an empty directory. The state goes into its [`STATE_DIR`] child.
//!
//! **Read-back on the 2B.** `read_parameters` and `save_state` stage a whole f32 table on the
//! device. On the 2B, after a backward, that faulted (`MTL4CommandQueueErrorTimeout`): the
//! allocation reached 52.44 GB against a recommended working set of 51.54 GB (AUDIT
//! `ojas-qwen35-gpu-diag-*`, main 991dd67).
//! - The ojas-qwen35 source this crate builds against now synchronizes before staging. It also
//!   refuses a staging that would pass the working set.
//! - `GAP-OJAS-QWEN35-2B-READ-GRADIENTS-METAL-FAULT-2026-10-01` is resolved: the full 2B test
//!   passed in the lead-cleared window.
//! - This lane has not run a 2B read-back through this adapter.
//! - A refusal at the read-back fails the run loudly after training, before any row is written.

use std::fs;
use std::path::{Path, PathBuf};

use ojas_core::{Budget, Tensor};
use ojas_qwen35::{
    AdamWHyper as OjasHyper, BankState, ExternalGrad, GroupSpec, LrRule, Numerics, OptimizerPlan,
    Precision, Qwen35Error, Qwen35Step, Qwen35TextConfig, Select, Sequence, Snapshot, TensorSpec,
    WdRule, Which, tower_tensors,
};
use qd_train::ledger::git_commit;
use qd_train::step::{
    AdamWHyper, BankMode, HiddenGrad, ParamSpec, SequenceJob, StepError, StepProvider,
    check_per_entry,
};

use crate::args::Operands;

/// The canonical tessl this crate builds against: the path qd-metal and qd-export name
/// (`../../../tessl` from their manifests), which ojas-qwen35's `../../tessl` also reaches.
pub const TESSL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tessl");
/// The ojas workspace ojas-qwen35 is built from.
pub const OJAS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../ojas");
/// The child of a checkpoint's provider directory that holds ojas-qwen35's state.
pub const STATE_DIR: &str = "qwen35-state";
/// Host bytes beyond one f32 parameter table: hidden rows and their gradients, per sequence.
pub const HOST_MARGIN_BYTES: u64 = 1 << 30;

/// ojas-qwen35's name for the operand precision.
pub fn numerics(operands: Operands) -> Numerics {
    match operands {
        Operands::ExactF32 => Numerics::ExactF32,
        Operands::Bf16 => Numerics::Bf16Operands,
    }
}

/// ojas-qwen35's error as qd-train's: a refused argument stays a refusal, a missing capability
/// stays unsupported, and everything the device or tessl did is the backend's.
pub fn step_error(e: Qwen35Error) -> StepError {
    match e {
        Qwen35Error::Invalid { .. } | Qwen35Error::Config(_) => StepError::Contract(e.to_string()),
        Qwen35Error::Unsupported { .. } => StepError::Unsupported(e.to_string()),
        Qwen35Error::Io(_) => StepError::Io(e.to_string()),
        Qwen35Error::Ojas(_) | Qwen35Error::Tessl { .. } | Qwen35Error::Poisoned(_) => {
            StepError::Backend(e.to_string())
        }
    }
}

/// The host budget a provider for `cfg` needs: one whole f32 parameter table (what
/// `read_table` hands out, 7.5 GB on the 2B) plus [`HOST_MARGIN_BYTES`]. A budget sized only
/// for hidden rows would train and then refuse the export.
pub fn host_budget_bytes(cfg: &Qwen35TextConfig) -> Result<u64, StepError> {
    tower_tensors(cfg)
        .iter()
        .try_fold(HOST_MARGIN_BYTES, |acc, t| {
            u64::try_from(t.numel())
                .ok()
                .and_then(|n| n.checked_mul(4))
                .and_then(|b| acc.checked_add(b))
        })
        .ok_or_else(|| StepError::Contract("the parameter table's byte count overflows u64".into()))
}

/// What ran, for the ledger: the provider, both source trees' commits at run time, the
/// kernels, and the two f32 roundings the adapter makes.
pub fn description(operands: Operands, tessl_commit: &str, ojas_commit: &str) -> String {
    format!(
        "ojas-qwen35 over tessl {tessl_commit} (ojas {ojas_commit}; commits read at run time): \
         {} GEMM operands with f32 accumulation, f32 weights, gradients and AdamW state; one unpadded \
         sequence per forward; attention tessl::attn_train (attn_train_forward/attn_train_backward, \
         causal GQA, head_dim 256, log-sum-exp); GDN tessl::gdn_train (rule published); AdamW \
         tessl::qwen35_adamw (adamw_step_scaled: per-entry lr_scale, per-entry weight decay as f32: \
         f32(0.01) = 0.00999999977648258); clip coefficient applied as f32",
        operands.as_str()
    )
}

/// One Qwen3.5 tower open for training. Not `Send`: neither is tessl's runtime.
pub struct Qwen35Provider {
    step: Qwen35Step,
    params: Vec<ParamSpec>,
    hidden: usize,
    vocab: usize,
    budget: Budget,
    /// The plan for the last `(lr_scale, weight decay)` vectors, keyed by their bits.
    plan: Option<(Vec<(u64, u32)>, OptimizerPlan)>,
    description: String,
}

impl Qwen35Provider {
    /// Load `snapshot` onto the GPU through [`Qwen35Step::open`] at [`Precision::F32`] (f32
    /// weights, gradients and AdamW moments: the optimizer recipe this crate's rows are labelled
    /// for; ojas-qwen35's bf16 tower is a different recipe and is not offered), with a host
    /// budget of
    /// [`host_budget_bytes`].
    pub fn open(snapshot: &Snapshot, operands: Operands) -> Result<Self, StepError> {
        let budget = Budget::new(host_budget_bytes(&snapshot.config)?);
        let step = Qwen35Step::open(snapshot, numerics(operands), Precision::F32, budget.clone())
            .map_err(step_error)?;
        let params = step
            .parameter_table()
            .iter()
            .map(|t| ParamSpec::new(t.name.clone(), &t.shape))
            .collect();
        let hidden = step.config().hidden as usize;
        let vocab = step.config().vocab as usize;
        let description = description(
            operands,
            &git_commit(Path::new(TESSL_DIR)),
            &git_commit(Path::new(OJAS_DIR)),
        );
        Ok(Self {
            step,
            params,
            hidden,
            vocab,
            budget,
            plan: None,
            description,
        })
    }

    /// The provider underneath, for a test that drives it directly beside the adapter.
    pub fn qwen35(&self) -> &Qwen35Step {
        &self.step
    }

    fn ensure_plan(&mut self, lr_scale: &[f64], weight_decay: &[f64]) -> Result<(), StepError> {
        let key = plan_key(lr_scale, weight_decay)?;
        if self.plan.as_ref().is_some_and(|(k, _)| *k == key) {
            return Ok(());
        }
        let plan = build_plan(self.step.parameter_table(), lr_scale, weight_decay)?;
        self.plan = Some((key, plan));
        Ok(())
    }
}

/// The cache key of a plan: each entry's `lr_scale` bits and its weight decay's f32 bits.
fn plan_key(lr_scale: &[f64], weight_decay: &[f64]) -> Result<Vec<(u64, u32)>, StepError> {
    lr_scale
        .iter()
        .zip(weight_decay)
        .enumerate()
        .map(|(i, (&s, &w))| {
            let w = w as f32;
            if !s.is_finite() || s < 0.0 {
                return Err(StepError::Contract(format!(
                    "entry {i}: lr_scale {s} must be finite and >= 0"
                )));
            }
            if !w.is_finite() {
                return Err(StepError::Contract(format!(
                    "entry {i}: weight decay {w} is not finite as f32"
                )));
            }
            Ok((s.to_bits(), w.to_bits()))
        })
        .collect()
}

/// ojas-qwen35's [`OptimizerPlan`] over `table` for the caller's per-entry vectors: one
/// learning-rate rule per distinct `lr_scale` and one weight-decay rule per distinct f32 weight
/// decay, each selecting exactly the entries that carry it. The plan's vectors are checked
/// equal, entry by entry and bit for bit, to the caller's (the weight decays as f32), so the
/// plan cannot differ from the optimizer table the loop derived.
pub fn build_plan(
    table: &[TensorSpec],
    lr_scale: &[f64],
    weight_decay: &[f64],
) -> Result<OptimizerPlan, StepError> {
    check_per_entry(table.len(), lr_scale, weight_decay)?;
    let key = plan_key(lr_scale, weight_decay)?;
    let mut lr_groups: Vec<(u64, Vec<Select>)> = Vec::new();
    let mut wd_groups: Vec<(u32, Vec<Select>)> = Vec::new();
    for (t, (scale_bits, wd_bits)) in table.iter().zip(&key) {
        let name = Select::Name(t.name.clone());
        match lr_groups.iter_mut().find(|(b, _)| b == scale_bits) {
            Some((_, names)) => names.push(name.clone()),
            None => lr_groups.push((*scale_bits, vec![name.clone()])),
        }
        match wd_groups.iter_mut().find(|(b, _)| b == wd_bits) {
            Some((_, names)) => names.push(name),
            None => wd_groups.push((*wd_bits, vec![name])),
        }
    }
    let spec = GroupSpec {
        lr: lr_groups
            .into_iter()
            .map(|(bits, names)| LrRule {
                label: format!(
                    "lr_scale={:?} (f64 bits {bits:#018x})",
                    f64::from_bits(bits)
                ),
                select: Select::AnyOf(names),
                lr_scale: f64::from_bits(bits),
            })
            .collect(),
        weight_decay: wd_groups
            .into_iter()
            .map(|(bits, names)| WdRule {
                label: format!(
                    "weight_decay={:?} (f32 bits {bits:#010x})",
                    f32::from_bits(bits)
                ),
                select: Select::AnyOf(names),
                weight_decay: f32::from_bits(bits),
            })
            .collect(),
    };
    let plan = OptimizerPlan::build(table, &spec).map_err(step_error)?;
    let same = plan
        .lr_scale()
        .iter()
        .map(|s| s.to_bits())
        .eq(key.iter().map(|k| k.0))
        && plan
            .weight_decay()
            .iter()
            .map(|w| w.to_bits())
            .eq(key.iter().map(|k| k.1));
    if !same {
        return Err(StepError::Backend(
            "ojas-qwen35's plan does not carry the recipe's per-entry learning-rate scales and weight decays".into(),
        ));
    }
    Ok(plan)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), StepError> {
    let io = |e: std::io::Error| StepError::Io(format!("{}: {e}", dir.display()));
    for entry in fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        let kind = entry.file_type().map_err(io)?;
        if kind.is_dir() {
            walk(&entry.path(), out)?;
        } else if kind.is_file() {
            out.push(entry.path());
        } else {
            return Err(StepError::Io(format!(
                "{} is neither a file nor a directory in a written state",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

impl StepProvider for Qwen35Provider {
    fn parameters(&self) -> &[ParamSpec] {
        &self.params
    }

    fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        job.validate(self.vocab)?;
        if !job.hidden_at.is_empty() && external.is_none() {
            return Err(StepError::Contract(
                "the job reads hidden states but no gradient callback was given".into(),
            ));
        }
        match (mode, self.step.bank_state()) {
            (BankMode::Overwrite, BankState::Empty) | (BankMode::Add, BankState::Holds(_)) => {}
            (BankMode::Overwrite, _) => self.step.discard_gradients(),
            (BankMode::Add, BankState::Empty) => {
                return Err(StepError::Contract(
                    "Add on an empty bank: a step's first sequence passes Overwrite".into(),
                ));
            }
            (BankMode::Add, BankState::Invalid(why)) => {
                return Err(StepError::Backend(format!(
                    "the gradient bank is invalid: {why}"
                )));
            }
        }
        let pending = self
            .step
            .forward(&Sequence {
                ids: job.tokens,
                letter_rows: job.rows.positions,
                letter_targets: job.rows.targets,
                letter_scale: job.rows.scale,
                span_positions: job.hidden_at,
            })
            .map_err(step_error)?;
        let sum = pending.letter_ce_sum();
        let dh = match external {
            Some(grad) if !job.hidden_at.is_empty() => {
                let hidden = pending
                    .hidden()
                    .to_f32_vec()
                    .map_err(|e| StepError::Backend(e.to_string()))?;
                let dh = grad(job.hidden_at, &hidden)?;
                if dh.len() != hidden.len() {
                    return Err(StepError::Contract(format!(
                        "the hidden-state gradient has {} values for {} rows of {}",
                        dh.len(),
                        job.hidden_at.len(),
                        self.hidden
                    )));
                }
                Some(
                    Tensor::from_f32(&dh, &[job.hidden_at.len(), self.hidden], &self.budget)
                        .map_err(|e| StepError::Backend(e.to_string()))?,
                )
            }
            _ => None,
        };
        let ext = dh.as_ref().map(|t| ExternalGrad {
            positions: job.hidden_at,
            dh: t,
        });
        self.step.backward(pending, ext).map_err(step_error)?;
        Ok(sum)
    }

    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        self.step.grad_sq_norm().map_err(step_error)
    }

    fn supports_lr_scale(&self) -> bool {
        true
    }

    fn adamw_step(
        &mut self,
        hyper: &AdamWHyper,
        step: u64,
        lr_scale: &[f64],
        weight_decay: &[f64],
    ) -> Result<(), StepError> {
        hyper.validate()?;
        check_per_entry(self.params.len(), lr_scale, weight_decay)?;
        let have = self.step.step_count();
        if have.checked_add(1) != Some(step) {
            return Err(StepError::Contract(format!(
                "asked for AdamW step {step} but the provider has taken {have}: a bias correction at \
                 the wrong t is a different optimizer"
            )));
        }
        let grad_scale = hyper.grad_scale as f32;
        if !(grad_scale.is_finite() && grad_scale > 0.0) {
            return Err(StepError::Contract(format!(
                "grad_scale {} is {grad_scale} as f32",
                hyper.grad_scale
            )));
        }
        self.ensure_plan(lr_scale, weight_decay)?;
        let plan = match &self.plan {
            Some((_, p)) => p,
            None => {
                return Err(StepError::Backend(
                    "no optimizer plan after building one".into(),
                ));
            }
        };
        let next = self
            .step
            .adamw_step(
                &OjasHyper {
                    lr: hyper.lr,
                    beta1: hyper.beta1,
                    beta2: hyper.beta2,
                    eps: hyper.eps,
                    grad_scale,
                },
                plan,
            )
            .map_err(step_error)?;
        if next != step {
            return Err(StepError::Backend(format!(
                "ojas-qwen35 reports step {next} after step {step}"
            )));
        }
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.step.step_count()
    }

    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        let table = self
            .step
            .read_table(Which::Parameters)
            .map_err(step_error)?;
        if table.len() != self.params.len() {
            return Err(StepError::Backend(format!(
                "read {} tensors for {} parameters",
                table.len(),
                self.params.len()
            )));
        }
        let mut out = Vec::with_capacity(table.len());
        // One host copy at a time: each ojas tensor is dropped once its values are copied out.
        for (t, spec) in table.into_iter().zip(&self.params) {
            if t.name != spec.name || t.tensor.shape() != spec.shape.as_slice() {
                return Err(StepError::Backend(format!(
                    "read {} {:?} where the table lists {} {:?}",
                    t.name,
                    t.tensor.shape(),
                    spec.name,
                    spec.shape
                )));
            }
            out.push(
                t.tensor
                    .to_f32_vec()
                    .map_err(|e| StepError::Backend(e.to_string()))?,
            );
        }
        Ok(out)
    }

    fn describe(&self) -> String {
        self.description.clone()
    }

    fn save_state(&mut self, dir: &Path) -> Result<Vec<PathBuf>, StepError> {
        let target = dir.join(STATE_DIR);
        self.step.save_state(&target).map_err(step_error)?;
        let mut files = Vec::new();
        walk(&target, &mut files)?;
        files.sort();
        Ok(files)
    }

    fn load_state(&mut self, dir: &Path) -> Result<(), StepError> {
        self.step
            .load_state(&dir.join(STATE_DIR))
            .map_err(step_error)
    }
}
