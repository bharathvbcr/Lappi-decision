//! One run, end to end. Everything that can be refused on the host is refused before the GPU
//! is opened:
//!
//! 1. the output directory is new or empty;
//! 2. the train shard set opens through the held-out door, with no escape hatch, and is the
//!    `train` split;
//! 3. the epoch plan (`reader.plan(batch_tokens, DataConfig().seed, epoch 0)`, the order
//!    real_ft's `plan_all` takes) has at least `steps` batches;
//! 4. the snapshot's config and weights header pass ojas-qwen35's checks, its directory name
//!    can be a `backbone_snapshot`, and qd-export's `Layout` reads its config;
//! 5. the shard set's remap is the identity over the snapshot's vocabulary ([`crate::remap`]);
//! 6. the span head's initial weights load at the tower's width.
//!
//! Then the ft row's id is minted, the provider opens, and qd-train's loop runs the first
//! `steps` batches with `grad_accum` 1 and `LrSchedule::real_ft(lr, steps)`. The trained
//! masters are read back once and exported twice: the scored bf16 file and the f32 masters.
//! Last, one `quick` row goes to the `mac-ojas` ledger. A run that fails at any point writes no
//! row, so it is rerun rather than remembered.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use ojas_qwen35::Snapshot;
use qd_export::layout::Layout;
use qd_train::export::{self, ExportSummary, ManifestInfo, NamedTensor, Precision, METAL_DEVICE, METAL_TRAINER};
use qd_train::files::OsFiles;
use qd_train::ft_data::{load_span_head_pinned, HostSpanHead, RealBatch};
use qd_train::held_out::DataConfig;
use qd_train::ledger::{self, Environment, FtRecipe, FtRow, Protocol, RunFacts, Status, WallClockSource};
use qd_train::objective::LetterSpanObjective;
use qd_train::recipe::{self, OptimizerRecipe};
use qd_train::run_control::{MonotonicClock, WallClockCap};
use qd_train::schedule::LrSchedule;
use qd_train::shards::{ShardOpen, ShardReader};
use qd_train::step::{ParamSpec, StepProvider};
use qd_train::trainer::{self, CheckpointPolicy, HostParams, Hooks, Progress, Termination, TrainConfig, TrainError};
use qd_train::tristate::TriState;

use crate::args::TrainArgs;
use crate::provider::Qwen35Provider;
use crate::remap::require_identity_remap;

/// What a finished run wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub row_id: String,
    pub termination: Termination,
    pub optimizer_steps: u64,
    pub export: ExportSummary,
    pub masters: ExportSummary,
}

#[derive(Debug)]
pub enum RunError {
    Refused(String),
    Failed(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Refused(m) => write!(f, "refused: {m}"),
            RunError::Failed(m) => write!(f, "failed: {m}"),
        }
    }
}

impl std::error::Error for RunError {}

fn refused<T>(m: impl Into<String>) -> Result<T, RunError> {
    Err(RunError::Refused(m.into()))
}

fn failed<E: std::fmt::Display>(what: &'static str) -> impl FnOnce(E) -> RunError {
    move |e| RunError::Failed(format!("{what}: {e}"))
}

/// The single-group arm's quick reason. Every row this binary writes is quick (rule 8).
fn quick_reason(steps: u64, epoch_batches: usize) -> String {
    format!(
        "Metal/tessl trainer (ojas-qwen35), 1 seed, {steps} of {epoch_batches} epoch batches, \
         single-group optimizer: tessl has no per-entry lr_scale, so F's layer-wise split is not \
         applied and this is not rung (d)"
    )
}

fn check_out_dir(out: &Path) -> Result<(), RunError> {
    match fs::read_dir(out) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                return refused(format!("{} is not empty; a run writes into a new directory", out.display()));
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(out).map_err(|e| RunError::Failed(format!("{}: {e}", out.display())))
        }
        Err(e) => Err(RunError::Failed(format!("{}: {e}", out.display()))),
    }
}

/// Run `a` to completion. See the module docs for the order of checks.
pub fn run(a: &TrainArgs) -> Result<RunSummary, RunError> {
    // ---- host-side refusals, before the GPU ------------------------------------------------
    check_out_dir(&a.out)?;
    let config = DataConfig::default();
    let data_seed = config.seed();
    let mut opts = ShardOpen::new(config, a.data_root.clone(), a.qd_data.clone());
    opts.expect_rev = a.expect_rev.clone();
    opts.files = Arc::new(OsFiles);
    let reader = ShardReader::open(&a.shards, &a.manifest, &opts).map_err(|e| RunError::Refused(format!("the shard door: {e}")))?;
    let header = reader.header().clone();
    if header.split != "train" {
        return refused(format!("{} is the {:?} split; training reads only train", a.shards.display(), header.split));
    }
    let plans = reader.plan(a.batch_tokens, data_seed, 0).map_err(failed("the epoch plan"))?;
    let n = usize::try_from(a.steps).map_err(|e| RunError::Refused(format!("--steps: {e}")))?;
    if plans.len() < n {
        return refused(format!(
            "the epoch has {} batches at batch_tokens {}; {} steps of one batch each do not fit",
            plans.len(),
            a.batch_tokens,
            a.steps
        ));
    }
    let width = plans[..n].iter().map(|p| p.width as u64).max().unwrap_or(0);

    let snapshot = Snapshot::from_dir(&a.snapshot).map_err(|e| RunError::Refused(format!("the snapshot: {e}")))?;
    let backbone_snapshot = a
        .snapshot
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .ok_or_else(|| RunError::Refused(format!("{} has no directory name to record", a.snapshot.display())))?
        .to_owned();
    let cfg_text = fs::read_to_string(&snapshot.config_path).map_err(failed("config.json"))?;
    let mut layout = Layout::from_config_json(&cfg_text).map_err(|e| RunError::Refused(format!("qd-export's layout: {e}")))?;
    let vocab = require_identity_remap(&reader, &OsFiles, u64::from(snapshot.config.vocab)).map_err(RunError::Refused)?;
    layout.vocab = usize::try_from(vocab).map_err(|e| RunError::Refused(e.to_string()))?;
    let hidden = snapshot.config.hidden as usize;
    if layout.hidden != hidden {
        return refused(format!("qd-export reads hidden {} but ojas-qwen35 reads {hidden}", layout.hidden));
    }
    let (head, head_sha, head_content) = load_span_head_pinned(&a.head_init, hidden, &a.head_init_pin)
        .map_err(|e| RunError::Refused(format!("the head init: {e}")))?;
    let head_file = a
        .head_init
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| RunError::Refused(format!("{} has no file name", a.head_init.display())))?
        .to_owned();
    let schedule = LrSchedule::real_ft(a.lr, a.steps).map_err(failed("the schedule"))?;
    let train_cfg = TrainConfig {
        schedule,
        cap: WallClockCap::new(a.cap_s).map_err(failed("the cap"))?,
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: a.lr,
            beta2: recipe::DEFAULT_BETA2,
            lower_layers: None,
        },
        max_grad_norm: recipe::MAX_GRAD_NORM,
        epoch: 0,
        seed: a.seed,
        checkpoint: a.checkpoint_every.map(|every| CheckpointPolicy {
            dir: a.out.join("checkpoints"),
            every,
        }),
        eta: a.eta,
    };
    let row_id = ledger::uuid4().map_err(failed("the row id"))?;

    // ---- the GPU --------------------------------------------------------------------------
    let mut provider = Qwen35Provider::open(&snapshot, a.operands).map_err(failed("Qwen35Provider::open"))?;
    let provider_text = provider.describe();
    let mut objective: LetterSpanObjective<RealBatch, HostSpanHead> =
        LetterSpanObjective::new(HostSpanHead::new(head).map_err(failed("the head"))?, a.span_weight)
            .map_err(failed("the objective"))?;
    let batches = reader
        .batches(a.batch_tokens, data_seed, 0)
        .map_err(failed("the batches"))?
        .take(n)
        .map(|b| b.map_err(|e| TrainError::Io(e.to_string())).and_then(RealBatch::new));
    let mut on_progress = |p: &Progress| {
        eprintln!(
            "step {}/{} lr {:e} loss {:.6} norm {:.4} clip {:.4} elapsed {:.1}s",
            p.record.optimizer_step + 1,
            p.total_steps,
            p.record.lr,
            p.record.loss,
            p.record.grad_norm,
            p.record.clip_coefficient,
            p.elapsed_s
        );
    };
    let mut hooks = Hooks {
        on_progress: Some(&mut on_progress),
    };
    let clock = MonotonicClock::new();
    let result = trainer::train(&mut provider, &mut objective, batches, &train_cfg, &clock, None, &mut hooks)
        .map_err(failed("training"))?;

    // ---- the export -----------------------------------------------------------------------
    let values = provider.read_parameters().map_err(failed("reading the trained masters"))?;
    let specs: Vec<ParamSpec> = provider.parameters().to_vec();
    let tower: Vec<NamedTensor<'_>> = specs
        .iter()
        .zip(&values)
        .map(|(spec, v)| NamedTensor {
            spec: spec.clone(),
            values: v,
        })
        .collect();
    let head = objective.head();
    let head_tensors: Vec<NamedTensor<'_>> = head
        .entries()
        .into_iter()
        .zip(head.values())
        .map(|(spec, values)| NamedTensor { spec, values })
        .collect();
    let recipe = FtRecipe {
        trainer: METAL_TRAINER.into(),
        tag: export::EXPORT_TAG.into(),
        device: METAL_DEVICE.into(),
        lr: a.lr,
        passes: 1,
        batches: a.steps,
        width,
        span_weight: a.span_weight,
        shard_hash: header.shard_hash().map_err(failed("shard_hash"))?,
        backbone_snapshot,
        backbone_vocab: vocab,
        backbone_params: snapshot.config.parameter_count(),
        attn_implementation: "sdpa".into(),
        optimizer_recipe: "master".into(),
        wall_clock_cap_s: Some(a.cap_s),
        batch_tokens: Some(a.batch_tokens),
        no_memorise: true,
        beta2: None,
        lower_layers: None,
        provider: provider_text.clone(),
        operands: a.operands.as_str().into(),
        optimizer_groups: "single".into(),
        train_attention_mask: "padding".into(),
        deterministic: false,
        extra: Default::default(),
    };
    let protocol = Protocol::for_recipe(&recipe, &header.data_snapshot_hash, &header.tokenizer_hash, a.seed)
        .map_err(failed("the protocol"))?;
    let info = ManifestInfo {
        trainer: METAL_TRAINER.into(),
        device: METAL_DEVICE.into(),
        seed: a.seed,
        optimizer_step: result.optimizer_steps,
        ft_row_id: row_id.clone(),
        vocab_size: vocab,
        span_weight: a.span_weight,
        schedule: schedule.to_json().map_err(failed("the schedule's JSON"))?,
        provider: provider_text.clone(),
        operands: a.operands.as_str().into(),
        recipe_hash: protocol.recipe_hash.clone(),
        loss_log_digest: result.loss_log_digest().map_err(failed("the loss log digest"))?,
        consumed_digest: result.consumed_digest.clone(),
        head_init_digest: Some(head_sha.clone()),
    };
    let scored = export::export(&a.out, &tower, &head_tensors, Some(&layout), &info, Precision::Bf16)
        .map_err(failed("the bf16 export"))?;
    let masters = export::export(&a.out, &tower, &head_tensors, Some(&layout), &info, Precision::F32Masters)
        .map_err(failed("the f32 masters export"))?;

    // ---- the row --------------------------------------------------------------------------
    let facts = RunFacts {
        cap_s: a.cap_s,
        provider: &provider_text,
        head_init: Some((&head_file, &head_sha)),
        eta: a.eta,
    };
    let mut metrics = ledger::ft_metrics(&result, &facts).map_err(failed("the metrics"))?;
    metrics.insert(
        "deterministic_kernels".into(),
        TriState::not_run("one run; repeat-run equality was not measured by this run"),
    );
    // Amendment 2 (iv): the digest a torch arm records for the head it built from seed 0, under
    // the same name, so gate 0 compares the two before anything else.
    metrics.insert(
        "train.span_head_init_digest".into(),
        TriState::ran(true, head_content.as_str()).with_detail(format!(
            "run_control._sidecar_digest over qd-tensor-ref-v1 of {head_file}'s span_head.* (float32), \
             recomputed at load and equal to the pinned value; file sha256 {head_sha}"
        )),
    );
    metrics.insert(
        "train.vocab_remap".into(),
        TriState::ran(true, "identity").with_detail(format!(
            "{} re-read through the door and every id maps to itself over {vocab} ids, the snapshot's \
             vocabulary: the tower is trained as loaded",
            reader.root().join("remap.npz").display()
        )),
    );
    let row = FtRow {
        protocol,
        recipe,
        status: Status::Completed,
        quick_reason: quick_reason(a.steps, plans.len()),
        code_commit: ledger::git_commit(&a.repo),
        env: Environment::rust_metal(ledger::hostname()),
        metrics,
        wall_clock_s: result.wall_clock_s,
        wall_clock_source: WallClockSource::Recorder,
        notes: format!(
            "qd-train-metal. Batches: the first {} of reader.batches(batch_tokens={}, seed={data_seed}, \
             epoch=0), real_ft's plan_all order, one per optimizer step. Schedule real_ft(lr, {}): warmup \
             max(1, steps/20), min lr/10. Optimizer: AdamW betas (0.9, 0.999), eps 1e-8, weight decay \
             0.01 on every tower and head entry (tessl applies it as f32(0.01) = 0.00999999977648258, \
             2.2e-8 relative), clip 1.0 over tower and head together; one learning rate for every \
             entry. Exports: {} (scored, bf16 tower) and {} (f32 masters, never scored).",
            a.steps,
            a.batch_tokens,
            a.steps,
            scored.weights.display(),
            masters.weights.display()
        ),
    };
    let written_at = ledger::isoformat_utc(SystemTime::now()).map_err(failed("written_at"))?;
    ledger::append(&a.ledger, &row, row_id.clone(), written_at).map_err(failed("the ledger append"))?;
    Ok(RunSummary {
        row_id,
        termination: result.termination,
        optimizer_steps: result.optimizer_steps,
        export: scored,
        masters,
    })
}
