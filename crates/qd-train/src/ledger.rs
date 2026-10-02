//! The Rust trainer's `ft` ledger row, in the bytes `python/qd_train/ledger.py` writes and reads.
//!
//! **Where it may write.** Only `ledger/mac-ojas-*.jsonl` (`HANDOFF/ojas-training-2026-10-01.md`,
//! invariants): never a campaign ledger, never a box path. And every row is `quick=True` with a
//! non-empty `quick_reason` (rule 8): the type has no way to say otherwise.
//!
//! **Other run kinds.** [`Row`] is the same row for any non-`build` kind (a `throughput`
//! benchmark, a `smoke` check), and [`append_line`] the same chained append without the
//! trainer's path rule; `qd-metal` writes its `ledger/mac-qd-metal-*.jsonl` rows through both and
//! states its own path rule. [`FtRow`] is a `Row` of kind `ft`.
//!
//! **What Python requires of the row** (`LedgerRow.__post_init__`, `from_json`,
//! `Ledger.verify_chain`): a known `run_kind` and `status`; `quick_reason` exactly when `quick`;
//! `wall_clock_source` in `{caller, recorder}`; finite non-negative `wall_clock_s` and
//! `cost_usd`; `protocol_hash` equal to sha256 of the canonical protocol; no `n/a:build` in a
//! training row's protocol; every tri-state well formed; `env` with its six keys; and
//! `prev_row_hash` equal to sha256 of the previous line's bytes. The line is
//! `json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, written with
//! `O_APPEND`, under an exclusive `flock` (the one `Ledger.append` takes), and fsynced.
//!
//! **What the scorer pairs a Metal export with** (`tools/real_ft_run.py` at main 1651fdf,
//! `_metal_export_weights`): `_ft_row` needs `run_kind == "ft"` and `status == "completed"`;
//! `_ft_row_mismatches` compares `recipe.tag == "epoch"`, `recipe.device == "metal"`,
//! `protocol.seed`, `recipe.shard_hash` and `recipe.backbone_snapshot` (the snapshot
//! directory's name); `_metal_row_problems` needs `recipe.trainer == "qd-train-metal"`,
//! `recipe.{attn_implementation, lr, span_weight, optimizer_recipe}` (what `_scoring_step`
//! builds the torch step from, so `attn_implementation` is a torch kernel name -- `sdpa`),
//! `quick` a bool with a reason, `metrics["train.optimizer_steps"]["value"]` an int and
//! `metrics["train.termination"]["value"]` a str. [`FtRecipe::to_json`] and [`ft_metrics`]
//! carry every one of those. `trained_by` is not an ft-row key: the scorer writes it on the
//! rows that score the export, from the manifest's sha256. The `recipe_hash` is sha256 over
//! `json.dumps(recipe, sort_keys=True, separators=(",", ":"))`, the bytes
//! `real_ft_run._protocol` hashes, so Python recomputes the stated hash from the stored recipe.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::pyjson::{dumps, float, obj, PyJsonError, CANONICAL, CANONICAL_ASCII};
use crate::run_control::hex;
use crate::trainer::{EtaRule, TrainResult};
use crate::tristate::{TriState, TriStateError};

/// `ledger.REQUIRED_GATES`, filled `not_run` when a run did not evaluate them.
pub const REQUIRED_GATES: [&str; 5] = [
    "paired_margin_vs_linear",
    "ood_abstain",
    "needle_hunk_recall",
    "permutation_consistency",
    "ece",
];
/// `ledger.REQUIRED_CONTROLS`.
pub const REQUIRED_CONTROLS: [&str; 4] = ["shuffled_label", "privileged_hunk", "degenerate_head", "transfer_gate"];
/// `ledger.NOT_APPLICABLE`: a build row's marker, refused in a training row.
pub const NOT_APPLICABLE: &str = "n/a:build";

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LedgerError {
    #[error("refused: {0}")]
    Refused(String),
    #[error(transparent)]
    Json(#[from] PyJsonError),
    #[error(transparent)]
    TriState(#[from] TriStateError),
    #[error("io: {0}")]
    Io(String),
}

fn refuse<T>(m: impl Into<String>) -> Result<T, LedgerError> {
    Err(LedgerError::Refused(m.into()))
}

/// The string a `Ran` record measured, if that is what it carries: how a row's
/// `train.termination` is read back.
fn termination_value(t: &TriState) -> Option<&str> {
    match t {
        TriState::Ran { value: Value::String(s), .. } => Some(s),
        _ => None,
    }
}

/// A map of named tri-states as the row stores it, each written by [`TriState::to_json`].
fn tristates(m: &BTreeMap<String, TriState>) -> Result<Value, LedgerError> {
    Ok(Value::Object(
        m.iter()
            .map(|(k, v)| Ok((k.clone(), v.to_json(k)?)))
            .collect::<Result<_, LedgerError>>()?,
    ))
}

/// The ft recipe as the row stores (and hashes) it. Keys follow `real_ft_run._train`'s recipe
/// and `_recipe_pieces`, plus the keys only a Metal run has (`provider`, `operands`,
/// `optimizer_groups`), so its rows hash apart from every PyTorch row.
#[derive(Debug, Clone, PartialEq)]
pub struct FtRecipe {
    /// The trainer that wrote the row: [`crate::export::METAL_TRAINER`] for a Metal run, which
    /// `real_ft_run._metal_row_problems` requires of the row a Metal export is scored against.
    pub trainer: String,
    pub tag: String,
    pub device: String,
    pub lr: f64,
    pub passes: u64,
    pub batches: u64,
    pub width: u64,
    pub span_weight: f64,
    pub shard_hash: String,
    pub backbone_snapshot: String,
    pub backbone_vocab: u64,
    pub backbone_params: u64,
    pub attn_implementation: String,
    pub optimizer_recipe: String,
    /// Recorded only when it is not the 30-minute default (`_recipe_pieces`).
    pub wall_clock_cap_s: Option<f64>,
    pub batch_tokens: Option<u64>,
    pub no_memorise: bool,
    pub beta2: Option<f64>,
    pub lower_layers: Option<(u64, f64)>,
    /// What computed the step: e.g. `"ojas-qwen35 over tessl"`.
    pub provider: String,
    /// `"exact_f32"` or `"bf16"` GEMM operands.
    pub operands: String,
    /// `"single"` while tessl has no per-entry lr (the only arm that runs before it lands).
    pub optimizer_groups: String,
    pub train_attention_mask: String,
    pub deterministic: bool,
    pub extra: BTreeMap<String, Value>,
}

impl FtRecipe {
    pub fn to_json(&self) -> Result<Value, LedgerError> {
        if self.backbone_snapshot.contains('/') || self.backbone_snapshot.contains('\\') {
            return refuse(format!(
                "backbone_snapshot is {:?}, a path rather than a revision; it feeds recipe_hash and \
                 an absolute path differs between machines",
                self.backbone_snapshot
            ));
        }
        for (name, v) in [
            ("trainer", &self.trainer),
            ("tag", &self.tag),
            ("device", &self.device),
            ("attn_implementation", &self.attn_implementation),
            ("optimizer_recipe", &self.optimizer_recipe),
        ] {
            if v.trim().is_empty() {
                return refuse(format!(
                    "recipe.{name} is empty: the scorer builds its step from it and pairs rows on it"
                ));
            }
        }
        let mut pairs: Vec<(String, Value)> = vec![
            ("tool".into(), Value::from("crates/qd-train-metal")),
            ("trainer".into(), Value::from(self.trainer.as_str())),
            ("tag".into(), Value::from(self.tag.as_str())),
            ("device".into(), Value::from(self.device.as_str())),
            ("lr".into(), float(self.lr)?),
            ("passes".into(), Value::from(self.passes)),
            ("batches".into(), Value::from(self.batches)),
            ("width".into(), Value::from(self.width)),
            ("span_weight".into(), float(self.span_weight)?),
            ("deterministic".into(), Value::Bool(self.deterministic)),
            ("shard_hash".into(), Value::from(self.shard_hash.as_str())),
            ("backbone_snapshot".into(), Value::from(self.backbone_snapshot.as_str())),
            ("backbone_vocab".into(), Value::from(self.backbone_vocab)),
            ("backbone_params".into(), Value::from(self.backbone_params)),
            ("attn_implementation".into(), Value::from(self.attn_implementation.as_str())),
            ("optimizer_recipe".into(), Value::from(self.optimizer_recipe.as_str())),
            ("provider".into(), Value::from(self.provider.as_str())),
            ("operands".into(), Value::from(self.operands.as_str())),
            ("optimizer_groups".into(), Value::from(self.optimizer_groups.as_str())),
        ];
        if self.train_attention_mask != "padding" {
            pairs.push(("train_attention_mask".into(), Value::from(self.train_attention_mask.as_str())));
        }
        if let Some(c) = self.wall_clock_cap_s
            && c != crate::run_control::DEFAULT_CAP_S
        {
            pairs.push(("wall_clock_cap_s".into(), float(c)?));
        }
        if let Some(b) = self.batch_tokens {
            pairs.push(("batch_tokens".into(), Value::from(b)));
        }
        if self.no_memorise {
            pairs.push(("no_memorise".into(), Value::Bool(true)));
        }
        if let Some(b2) = self.beta2
            && b2 != crate::recipe::DEFAULT_BETA2
        {
            pairs.push(("beta2".into(), float(b2)?));
        }
        if let Some((n, s)) = self.lower_layers {
            pairs.push(("lower_layers_n".into(), Value::from(n)));
            pairs.push(("lower_lr_scale".into(), float(s)?));
        }
        for (k, v) in &self.extra {
            pairs.push((k.clone(), v.clone()));
        }
        Ok(obj(pairs)?)
    }
}

/// `real_ft_run._protocol` / `ledger.Protocol`.
#[derive(Debug, Clone, PartialEq)]
pub struct Protocol {
    pub data_snapshot_hash: String,
    pub tokenizer_hash: String,
    pub backbone_commit: String,
    pub recipe_hash: String,
    pub seed: u64,
}

impl Protocol {
    /// The protocol of a run of `recipe` at `seed`: `backbone_commit` is
    /// `"<snapshot>:vocab<vocab>"` (`_backbone_commit`), `recipe_hash` is sha256 over the
    /// recipe's `json.dumps(sort_keys=True, separators=(",", ":"))`.
    pub fn for_recipe(recipe: &FtRecipe, data_snapshot_hash: &str, tokenizer_hash: &str, seed: u64) -> Result<Self, LedgerError> {
        let body = dumps(&recipe.to_json()?, CANONICAL_ASCII)?;
        Ok(Self {
            data_snapshot_hash: data_snapshot_hash.to_string(),
            tokenizer_hash: tokenizer_hash.to_string(),
            backbone_commit: format!("{}:vocab{}", recipe.backbone_snapshot, recipe.backbone_vocab),
            recipe_hash: hex(&Sha256::digest(body.as_bytes())),
            seed,
        })
    }

    pub fn to_json(&self) -> Result<Value, LedgerError> {
        for (name, v) in [
            ("data_snapshot_hash", &self.data_snapshot_hash),
            ("tokenizer_hash", &self.tokenizer_hash),
            ("backbone_commit", &self.backbone_commit),
            ("recipe_hash", &self.recipe_hash),
        ] {
            if v.trim().is_empty() {
                return refuse(format!("Protocol.{name} must be a non-empty string"));
            }
            if v == NOT_APPLICABLE {
                return refuse(format!(
                    "Protocol.{name} carries the build marker {NOT_APPLICABLE:?}; only a build row may"
                ));
            }
        }
        Ok(obj([
            ("data_snapshot_hash", Value::from(self.data_snapshot_hash.as_str())),
            ("tokenizer_hash", Value::from(self.tokenizer_hash.as_str())),
            ("backbone_commit", Value::from(self.backbone_commit.as_str())),
            ("recipe_hash", Value::from(self.recipe_hash.as_str())),
            ("seed", Value::from(self.seed)),
        ])?)
    }

    /// `Protocol.hash`: sha256 over the canonical protocol (`ensure_ascii=False`).
    pub fn hash(&self) -> Result<String, LedgerError> {
        Ok(hex(&Sha256::digest(dumps(&self.to_json()?, CANONICAL)?.as_bytes())))
    }
}

/// How the run ended, as the row records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Completed,
    Killed,
    Failed,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Completed => "completed",
            Status::Killed => "killed",
            Status::Failed => "failed",
        }
    }
}

/// Who measured `wall_clock_s` (`WallClockSource`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallClockSource {
    Caller,
    Recorder,
}

/// `ledger.Environment`. A Rust run has no torch and no transformers; it says so.
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    pub torch: String,
    pub transformers_sha: String,
    pub device: String,
    pub host: String,
}

impl Environment {
    /// The Rust trainer on this host's Metal device.
    pub fn rust_metal(host: String) -> Self {
        Self {
            torch: "n/a: Rust trainer, no torch in the process".into(),
            transformers_sha: "n/a: Rust trainer, no transformers in the process".into(),
            device: "metal".into(),
            host,
        }
    }

    /// `Environment.to_json`. A Rust process probes neither CUDA library, so both are `not_run`.
    pub fn to_json(&self) -> Result<Value, LedgerError> {
        let probe = |what: &str, field: &str| TriState::not_run(format!("{what} dry run not executed")).to_json(field);
        Ok(obj([
            ("torch", Value::from(self.torch.as_str())),
            ("transformers_sha", Value::from(self.transformers_sha.as_str())),
            ("device", Value::from(self.device.as_str())),
            ("host", Value::from(self.host.as_str())),
            ("fla_present", probe("fla", "env.fla_present")?),
            ("causal_conv1d_present", probe("causal-conv1d", "env.causal_conv1d_present")?),
        ])?)
    }
}

/// An `ft` row before the ledger gives it a place in the chain.
#[derive(Debug, Clone, PartialEq)]
pub struct FtRow {
    pub protocol: Protocol,
    pub recipe: FtRecipe,
    pub status: Status,
    pub quick_reason: String,
    pub code_commit: String,
    pub env: Environment,
    pub metrics: BTreeMap<String, TriState>,
    pub wall_clock_s: f64,
    pub wall_clock_source: WallClockSource,
    pub notes: String,
}

/// The fields the ledger assigns at append time; injected in tests.
#[derive(Debug, Clone, PartialEq)]
pub struct Stamp {
    pub row_id: String,
    pub written_at: String,
    pub prev_row_hash: Option<String>,
}

/// `ledger.RunKind` less `build`: the kinds a [`Row`] may carry. A build row's protocol carries
/// the [`NOT_APPLICABLE`] marker, which [`Protocol::to_json`] refuses; Python's
/// `Protocol.for_build` / `qd_train.ledger record` is that row's writer.
pub const RUN_KINDS: [&str; 12] = [
    "teacher",
    "lr_probe",
    "cpt",
    "prune_heal",
    "ft",
    "ablation",
    "eval",
    "calibration",
    "smoke",
    "throughput",
    "resume",
    "scale",
];

/// A row of any [`RUN_KINDS`] kind before the ledger gives it a place in the chain. [`FtRow`] is
/// this for `ft` plus the trainer's own rules; a benchmark or a check (`throughput`, `smoke`)
/// fills it directly. Every row is `quick` (rule 8): the type has no way to say otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub run_kind: String,
    pub protocol: Protocol,
    pub status: Status,
    pub quick_reason: String,
    pub code_commit: String,
    pub env: Environment,
    pub metrics: BTreeMap<String, TriState>,
    pub noul_rate: TriState,
    pub wall_clock_s: f64,
    pub wall_clock_source: WallClockSource,
    pub notes: String,
    /// The settings `protocol.recipe_hash` was taken of: sha256 over this object's
    /// `json.dumps(sort_keys=True, separators=(",", ":"))`, checked when the line is built.
    pub recipe: Value,
}

impl Row {
    /// The row's JSON with `stamp`'s fields: the exact line `Ledger.append` writes, newline
    /// excluded. Refuses what `LedgerRow.__post_init__` refuses, and a recipe that does not
    /// reproduce `protocol.recipe_hash`.
    pub fn line(&self, stamp: &Stamp) -> Result<String, LedgerError> {
        if !RUN_KINDS.contains(&self.run_kind.as_str()) {
            return refuse(format!("run_kind {:?} is not one of {RUN_KINDS:?}", self.run_kind));
        }
        if self.quick_reason.trim().is_empty() {
            return refuse("quick=True requires quick_reason; every row this crate writes is quick (rule 8)");
        }
        if !(self.wall_clock_s.is_finite() && self.wall_clock_s >= 0.0) {
            return refuse(format!("wall_clock_s {} is not a measured duration", self.wall_clock_s));
        }
        if !self.recipe.as_object().is_some_and(|o| !o.is_empty()) {
            return refuse(
                "recipe is not a non-empty object: a row with a recipe_hash always had settings, and \
                 `LedgerRow` refuses an empty one",
            );
        }
        let recipe_hash = hex(&Sha256::digest(dumps(&self.recipe, CANONICAL_ASCII)?.as_bytes()));
        if recipe_hash != self.protocol.recipe_hash {
            return refuse(format!(
                "protocol.recipe_hash {} is not the hash of the stored recipe ({recipe_hash})",
                self.protocol.recipe_hash
            ));
        }
        let gates: BTreeMap<String, TriState> = REQUIRED_GATES
            .iter()
            .map(|g| (g.to_string(), TriState::not_run(format!("gate '{g}' was never evaluated by this run"))))
            .collect();
        let controls: BTreeMap<String, TriState> = REQUIRED_CONTROLS
            .iter()
            .map(|c| (c.to_string(), TriState::not_run(format!("control '{c}' was never evaluated by this run"))))
            .collect();
        let row = obj([
            ("row_id", Value::from(stamp.row_id.as_str())),
            ("written_at", Value::from(stamp.written_at.as_str())),
            (
                "prev_row_hash",
                stamp.prev_row_hash.clone().map_or(Value::Null, Value::from),
            ),
            ("protocol_hash", Value::from(self.protocol.hash()?)),
            ("protocol", self.protocol.to_json()?),
            ("run_kind", Value::from(self.run_kind.as_str())),
            ("status", Value::from(self.status.as_str())),
            ("quick", Value::Bool(true)),
            ("quick_reason", Value::from(self.quick_reason.as_str())),
            ("code_commit", Value::from(self.code_commit.as_str())),
            ("env", self.env.to_json()?),
            ("metrics", tristates(&self.metrics)?),
            ("noul_rate", self.noul_rate.to_json("noul_rate")?),
            ("controls", tristates(&controls)?),
            ("gates", tristates(&gates)?),
            ("wall_clock_s", float(self.wall_clock_s)?),
            (
                "wall_clock_source",
                Value::from(match self.wall_clock_source {
                    WallClockSource::Caller => "caller",
                    WallClockSource::Recorder => "recorder",
                }),
            ),
            ("cost_usd", float(0.0)?),
            ("notes", Value::from(self.notes.as_str())),
            ("recipe", self.recipe.clone()),
        ])?;
        Ok(dumps(&row, CANONICAL)?)
    }
}

impl FtRow {
    /// The row's JSON with `stamp`'s fields: the exact line `Ledger.append` writes, newline
    /// excluded. A [`Row`] of kind `ft`, with the trainer's own rule on top: a schedule that did
    /// not run to its end is quick for that reason too.
    pub fn line(&self, stamp: &Stamp) -> Result<String, LedgerError> {
        if self.quick_reason.trim().is_empty() {
            return refuse("quick=True requires quick_reason; every ojas row is quick (rule 8)");
        }
        // `_quick_if_truncated`: a schedule that did not run to its end is a truncated one.
        let mut quick_reason = self.quick_reason.clone();
        if let Some(term) = self.metrics.get("train.termination") {
            let said = termination_value(term).unwrap_or("not_run");
            if said != "steps_exhausted" {
                quick_reason = format!(
                    "{quick_reason}; train.termination is '{said}', not 'steps_exhausted': the schedule \
                     did not run to its end, which rule 8 calls a truncated schedule"
                );
            }
        }
        Row {
            run_kind: "ft".to_string(),
            protocol: self.protocol.clone(),
            status: self.status,
            quick_reason,
            code_commit: self.code_commit.clone(),
            env: self.env.clone(),
            metrics: self.metrics.clone(),
            noul_rate: TriState::not_run("noul rate not computed by this run"),
            wall_clock_s: self.wall_clock_s,
            wall_clock_source: self.wall_clock_source,
            notes: self.notes.clone(),
            recipe: self.recipe.to_json()?,
        }
        .line(stamp)
    }
}

/// What a run's metrics say beyond its [`TrainResult`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunFacts<'a> {
    /// The wall-clock cap the run was given.
    pub cap_s: f64,
    /// What computed the step (`train.path`).
    pub provider: &'a str,
    /// The span head's initial weights, when they were read from a file: `(file name, sha256 of
    /// its bytes)`. Rung (d)'s arms compare this digest before anything else.
    pub head_init: Option<(&'a str, &'a str)>,
    /// The early-stop rule the run was given, if any.
    pub eta: Option<EtaRule>,
}

/// The training metrics `trainer._train_loop` records, from a [`TrainResult`], plus what a
/// Rust run adds: the consumed-batch digest, the provider that ran, the head-init digest and
/// the ETA projection. A run that did not end by running out of work says so with
/// `train.termination` `passed=false` (a capped run, and an `eta_rule` stop).
pub fn ft_metrics(result: &TrainResult, facts: &RunFacts<'_>) -> Result<BTreeMap<String, TriState>, LedgerError> {
    let mut m = BTreeMap::new();
    let steps = result.optimizer_steps;
    let cap_s = facts.cap_s;
    m.insert(
        "train.termination".to_string(),
        TriState::ran(result.termination.finished(), result.termination.as_str())
            .with_detail(format!("ft stopped after {steps} optimizer step(s)")),
    );
    m.insert(
        "train.head_init_digest".to_string(),
        match facts.head_init {
            Some((file, sha)) => {
                if !(sha.len() == 64 && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))) {
                    return refuse(format!("head-init digest {sha:?} is not a lowercase hex sha256"));
                }
                TriState::ran(true, sha).with_detail(format!("sha256 of {file}, the span head's initial weights"))
            }
            None => TriState::not_run("the span head was not initialised from a file, so there is no init digest to compare"),
        },
    );
    m.insert(
        "train.eta_projected_s".to_string(),
        match (facts.eta, result.eta_projected_s) {
            (Some(rule), Some(p)) => TriState::ran(!rule.stops(p, cap_s), float(p)?).with_detail(format!(
                "projected at optimizer step {} from this process's elapsed time; the rule stops a run \
                 projected past {:.1} s (the {cap_s:.1} s cap less {:.1} s)",
                rule.at_step,
                cap_s - rule.margin_s,
                rule.margin_s
            )),
            (Some(rule), None) => TriState::not_run(format!(
                "the ETA rule at step {} was not evaluated by this process (it ended first, or resumed past it)",
                rule.at_step
            )),
            (None, _) => TriState::not_run("this run had no ETA rule"),
        },
    );
    m.insert("train.optimizer_steps".to_string(), TriState::ran(true, steps));
    m.insert("train.micro_batches".to_string(), TriState::ran(true, result.micro_batches));
    let c = &result.counts;
    m.insert(
        "train.supervised_tokens".to_string(),
        TriState::ran(true, c.supervised_tokens).with_coverage(c.supervised_tokens, c.total_positions),
    );
    m.insert("train.span_rows".to_string(), TriState::ran(true, c.span_rows));
    let frac = if c.total_positions > 0 {
        c.padded_positions as f64 / c.total_positions as f64
    } else {
        0.0
    };
    m.insert(
        "train.padding_fraction".to_string(),
        TriState::ran(true, float(frac)?).with_coverage(c.padded_positions, c.total_positions),
    );
    m.insert(
        "train.final_loss".to_string(),
        match result.loss_log.last() {
            Some(p) => TriState::ran(true, float(p.loss)?),
            None => TriState::ran(false, Value::Null).with_detail("no optimizer step completed"),
        },
    );
    m.insert("train.loss_log_digest".to_string(), TriState::ran(true, result.loss_log_digest()?));
    m.insert(
        "train.consumed_digest".to_string(),
        TriState::ran(true, result.consumed_digest.as_str())
            .with_detail(format!("ConsumedPrefix over the {} batch(es) this run consumed", result.consumed_n)),
    );
    m.insert(
        "train.projected_usd_at_cap".to_string(),
        TriState::ran(true, float(0.0)?).with_detail(format!(
            "local-metal: the Mac's own GPU, not billed, capped at {:.2} h -> $0.00 at the cap -- no \
             approval required",
            cap_s / 3600.0
        )),
    );
    m.insert(
        "train.path".to_string(),
        TriState::ran(true, facts.provider)
            .with_detail("what this row ran on; compare rows only where this agrees or says why not"),
    );
    Ok(m)
}

/// The rows this crate may write go to `ledger/mac-ojas-*.jsonl` and nowhere else.
pub fn check_ledger_path(path: &Path) -> Result<(), LedgerError> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parent = path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("");
    if !(name.starts_with("mac-ojas-") && name.ends_with(".jsonl") && name.len() > "mac-ojas-.jsonl".len()) {
        return refuse(format!(
            "{}: the Rust trainer writes only ledger/mac-ojas-*.jsonl, never a campaign ledger",
            path.display()
        ));
    }
    if parent != "ledger" {
        return refuse(format!("{}: the file must sit in a directory named `ledger`", path.display()));
    }
    Ok(())
}

/// The lines of a ledger file as `Ledger.raw_lines` splits them.
fn raw_lines(bytes: &[u8]) -> Vec<&[u8]> {
    bytes
        .split(|b| *b == b'\n')
        .filter(|l| !l.iter().all(|b| b.is_ascii_whitespace()))
        .collect()
}

/// `Ledger.append` for the Rust trainer's `ft` row: only into `ledger/mac-ojas-*.jsonl`
/// ([`check_ledger_path`]), then [`append_line`].
pub fn append(path: &Path, row: &FtRow, row_id: String, written_at: String) -> Result<Stamp, LedgerError> {
    check_ledger_path(path)?;
    append_line(path, row_id, written_at, |stamp| row.line(stamp))
}

/// `Ledger.append`: take the write lock, refuse a duplicate `row_id`, chain to the last line,
/// append the line `line(stamp)` builds with `O_APPEND`, fsync. Returns the stamp the row was
/// written with. Which files a writer may append to is the writer's rule: [`append`] holds the
/// trainer to `mac-ojas-*`; another caller states its own before calling this.
pub fn append_line(
    path: &Path,
    row_id: String,
    written_at: String,
    line: impl FnOnce(&Stamp) -> Result<String, LedgerError>,
) -> Result<Stamp, LedgerError> {
    let ioe = |e: std::io::Error| LedgerError::Io(format!("{}: {e}", path.display()));
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(ioe)?;
    }
    let mut f = fs::File::options().create(true).append(true).read(true).open(path).map_err(ioe)?;
    f.lock().map_err(ioe)?;
    let result = (|| {
        let mut bytes = Vec::new();
        fs::File::open(path).and_then(|mut r| r.read_to_end(&mut bytes)).map_err(ioe)?;
        let lines = raw_lines(&bytes);
        let needle = format!("\"row_id\":\"{row_id}\"");
        if lines.iter().any(|l| l.windows(needle.len()).any(|w| w == needle.as_bytes())) {
            return refuse(format!("row_id {row_id} already present: the ledger is append-only"));
        }
        let stamp = Stamp {
            row_id: row_id.clone(),
            written_at: written_at.clone(),
            prev_row_hash: lines.last().map(|l| hex(&Sha256::digest(l))),
        };
        let mut out = line(&stamp)?.into_bytes();
        out.push(b'\n');
        f.write_all(&out).map_err(ioe)?;
        f.sync_all().map_err(ioe)?;
        Ok(stamp)
    })();
    let unlock = f.unlock().map_err(ioe);
    let stamp = result?;
    unlock?;
    Ok(stamp)
}

/// A version-4 UUID from the OS's random source, as `uuid.uuid4()` formats it.
pub fn uuid4() -> Result<String, LedgerError> {
    let mut b = [0u8; 16];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| LedgerError::Io(format!("/dev/urandom: {e}")))?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h = hex(&b);
    Ok(format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32]))
}

/// `datetime.now(UTC).isoformat()` for `t`: microseconds when non-zero, `+00:00`.
pub fn isoformat_utc(t: SystemTime) -> Result<String, LedgerError> {
    let d = t
        .duration_since(UNIX_EPOCH)
        .map_err(|e| LedgerError::Refused(format!("a time before 1970: {e}")))?;
    let secs = i64::try_from(d.as_secs()).map_err(|e| LedgerError::Refused(e.to_string()))?;
    let micros = d.subsec_micros();
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let frac = if micros == 0 { String::new() } else { format!(".{micros:06}") };
    Ok(format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}{frac}+00:00"))
}

/// `_git_commit`: HEAD, with `-dirty` when the tree has any change (an unknown(...) string when
/// git cannot answer, never a guess).
pub fn git_commit(repo: &Path) -> String {
    let run = |args: &[&str]| -> Result<String, String> {
        let out = Command::new("git").args(args).current_dir(repo).output().map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    match (run(&["rev-parse", "HEAD"]), run(&["status", "--porcelain"])) {
        (Ok(head), Ok(dirty)) => {
            if dirty.is_empty() {
                head
            } else {
                format!("{head}-dirty")
            }
        }
        (Err(e), _) | (_, Err(e)) => format!("unknown({e})"),
    }
}

/// This host's name (`socket.gethostname()`), or an `unknown(...)` string.
pub fn hostname() -> String {
    match Command::new("hostname").output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Ok(o) => format!("unknown(hostname exited {})", o.status),
        Err(e) => format!("unknown({e})"),
    }
}

/// Where a row for `lane` written on `date` goes: `<repo>/ledger/mac-ojas-<lane>-<date>.jsonl`.
pub fn ojas_ledger_path(repo: &Path, lane: &str, date: &str) -> Result<PathBuf, LedgerError> {
    let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-');
    if !ok(lane) || !ok(date) {
        return refuse(format!("lane {lane:?} and date {date:?} must be [A-Za-z0-9-]+"));
    }
    let p = repo.join("ledger").join(format!("mac-ojas-{lane}-{date}.jsonl"));
    check_ledger_path(&p)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_mac_ojas_ledgers_are_writable() {
        assert!(check_ledger_path(Path::new("/r/ledger/mac-ojas-rung-b-2026-10-01.jsonl")).is_ok());
        assert!(check_ledger_path(Path::new("/r/ledger/gh200-p4-v4-2026-10-01.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/ledger/runs.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/elsewhere/mac-ojas-x.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/ledger/mac-ojas-.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/ledger/mac-ojas-x.json")).is_err());
        assert!(ojas_ledger_path(Path::new("/r"), "rung-b", "2026-10-01").is_ok());
        assert!(ojas_ledger_path(Path::new("/r"), "../x", "2026-10-01").is_err());
    }

    #[test]
    fn isoformat_matches_pythons_layout() {
        let t = UNIX_EPOCH + std::time::Duration::new(1_790_000_000, 123_456_000);
        assert_eq!(isoformat_utc(t).unwrap(), "2026-09-21T14:13:20.123456+00:00");
        let whole = UNIX_EPOCH + std::time::Duration::new(1_790_000_000, 0);
        assert_eq!(isoformat_utc(whole).unwrap(), "2026-09-21T14:13:20+00:00", "no fraction at 0 us");
        assert_eq!(isoformat_utc(UNIX_EPOCH + std::time::Duration::new(951_782_400, 0)).unwrap(), "2000-02-29T00:00:00+00:00");
    }

    #[test]
    fn uuid4_has_the_version_and_variant_bits() {
        let u = uuid4().unwrap();
        assert_eq!(u.len(), 36);
        assert_eq!(&u[14..15], "4");
        assert!(matches!(&u[19..20], "8" | "9" | "a" | "b"));
        assert_ne!(u, uuid4().unwrap());
    }
}
