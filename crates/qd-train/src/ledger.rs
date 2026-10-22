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

use crate::pyjson::{CANONICAL, CANONICAL_ASCII, PyJsonError, dumps, float, obj};
use crate::run_control::hex;
use crate::step::ParamSpec;
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
pub const REQUIRED_CONTROLS: [&str; 4] = [
    "shuffled_label",
    "privileged_hunk",
    "degenerate_head",
    "transfer_gate",
];
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
        TriState::Ran {
            value: Value::String(s),
            ..
        } => Some(s),
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

/// `recipe.optimizer_recipe` of a Metal full fine-tune: AdamW over f32 parameters, with no
/// master/bf16 split. It is not `"master"` (torch's `MasterWeightAdamW` over bf16 parameters),
/// so a Mac row is never pooled with a v5 row (Fable's board-sweep ruling 2h). LoRA-2B's label is
/// `"lora"` (D6); both are in Python's `OPTIMIZER_RECIPES`.
pub const METAL_FULL_FT_OPTIMIZER_RECIPE: &str = "f32";

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
            (
                "backbone_snapshot".into(),
                Value::from(self.backbone_snapshot.as_str()),
            ),
            ("backbone_vocab".into(), Value::from(self.backbone_vocab)),
            ("backbone_params".into(), Value::from(self.backbone_params)),
            (
                "attn_implementation".into(),
                Value::from(self.attn_implementation.as_str()),
            ),
            (
                "optimizer_recipe".into(),
                Value::from(self.optimizer_recipe.as_str()),
            ),
            ("provider".into(), Value::from(self.provider.as_str())),
            ("operands".into(), Value::from(self.operands.as_str())),
            (
                "optimizer_groups".into(),
                Value::from(self.optimizer_groups.as_str()),
            ),
        ];
        if self.train_attention_mask != "padding" {
            pairs.push((
                "train_attention_mask".into(),
                Value::from(self.train_attention_mask.as_str()),
            ));
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

/// The [`FtRecipe::extra`] keys a run on a shard set of `prompt_format` adds: `prompt_format`
/// when it is not 1, nothing when it is (`real_ft_run._recipe_pieces`' rule), so every v4 ft
/// row's recipe -- and its `recipe_hash` -- is what it was. `real_ft_run.py --score-checkpoint`
/// refuses a row whose recipe format (absent = 1) is not the scorer's `PROMPT_FORMAT`, so a
/// format-2 run that omitted the key could not be scored.
pub fn prompt_format_extra(prompt_format: u64) -> BTreeMap<String, Value> {
    let mut extra = BTreeMap::new();
    if prompt_format != 1 {
        extra.insert("prompt_format".to_owned(), Value::from(prompt_format));
    }
    extra
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
    pub fn for_recipe(
        recipe: &FtRecipe,
        data_snapshot_hash: &str,
        tokenizer_hash: &str,
        seed: u64,
    ) -> Result<Self, LedgerError> {
        let body = dumps(&recipe.to_json()?, CANONICAL_ASCII)?;
        Ok(Self {
            data_snapshot_hash: data_snapshot_hash.to_string(),
            tokenizer_hash: tokenizer_hash.to_string(),
            backbone_commit: format!(
                "{}:vocab{}",
                recipe.backbone_snapshot, recipe.backbone_vocab
            ),
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
            (
                "data_snapshot_hash",
                Value::from(self.data_snapshot_hash.as_str()),
            ),
            ("tokenizer_hash", Value::from(self.tokenizer_hash.as_str())),
            (
                "backbone_commit",
                Value::from(self.backbone_commit.as_str()),
            ),
            ("recipe_hash", Value::from(self.recipe_hash.as_str())),
            ("seed", Value::from(self.seed)),
        ])?)
    }

    /// `Protocol.hash`: sha256 over the canonical protocol (`ensure_ascii=False`).
    pub fn hash(&self) -> Result<String, LedgerError> {
        Ok(hex(&Sha256::digest(
            dumps(&self.to_json()?, CANONICAL)?.as_bytes(),
        )))
    }
}

/// How the run ended, as the row records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The one non-terminal status (`ledger.Status`): a run's lifecycle marker, written by
    /// Python's `RunRecorder.__enter__`. A reader skips it by default, as `Ledger.rows()` does;
    /// this crate never writes one ([`Row::line`] refuses it).
    Started,
    Completed,
    Killed,
    Failed,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Started => "started",
            Status::Completed => "completed",
            Status::Killed => "killed",
            Status::Failed => "failed",
        }
    }

    /// The status a row line names; an unknown one is refused, never mapped to a default.
    pub fn parse(s: &str) -> Result<Self, LedgerError> {
        match s {
            "started" => Ok(Status::Started),
            "completed" => Ok(Status::Completed),
            "killed" => Ok(Status::Killed),
            "failed" => Ok(Status::Failed),
            other => refuse(format!(
                "status {other:?} is not one of started, completed, killed, failed"
            )),
        }
    }
}

/// The two optional row fields Python's `LedgerRow` carries beyond the core row. Absent means
/// "not recorded" and the key is omitted from the line, so a row recording neither is the line
/// it always was. Neither is a `Protocol` field: that would move every protocol hash.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RowExtras {
    /// `LedgerRow.val_without_train`: the data build's block naming families with val rows and
    /// no train rows (Fable Q14 P2). An object; its inner shape belongs to its producer.
    pub val_without_train: Option<Value>,
    /// `LedgerRow.train_manifest_sha256`: sha256 of the train manifest file's bytes, lowercase hex.
    pub train_manifest_sha256: Option<String>,
}

impl RowExtras {
    fn validate(&self) -> Result<(), LedgerError> {
        match &self.val_without_train {
            None => {}
            Some(Value::Object(o)) => {
                if o.get("families").is_some_and(|f| !f.is_object()) {
                    return refuse(
                        "val_without_train.families is not a mapping of family to val rows",
                    );
                }
            }
            Some(_) => return refuse("val_without_train is not an object"),
        }
        if let Some(d) = &self.train_manifest_sha256
            && !is_hex64(d)
        {
            return refuse(format!(
                "train_manifest_sha256 {d:?} is not a lowercase hex sha256"
            ));
        }
        Ok(())
    }

    /// Read the two keys off a parsed row line, refusing what [`Self::validate`] refuses. An
    /// absent key and a `null` are the same: not recorded.
    fn from_line(v: &Value) -> Result<Self, LedgerError> {
        let extras = Self {
            val_without_train: v.get("val_without_train").filter(|x| !x.is_null()).cloned(),
            train_manifest_sha256: match v.get("train_manifest_sha256") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(_) => return refuse("train_manifest_sha256 is not a string"),
            },
        };
        extras.validate()?;
        Ok(extras)
    }
}

/// One ledger line as a reader sees it: which row, how it ended, the optional fields it
/// records, and the hash the next line chains to.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerEntry {
    pub row_id: String,
    pub status: Status,
    pub extras: RowExtras,
    /// sha256 of the raw line's bytes (`prev_row_hash` of the line after it).
    pub line_sha256: String,
}

/// The most a reader will load: one ledger file, so a corrupt or runaway file is refused rather
/// than allocated.
pub const MAX_LEDGER_BYTES: u64 = 256 * 1024 * 1024;

/// `Ledger.rows(include_started=...)` for the fields a Rust reader needs. A missing file is an
/// empty ledger (`raw_lines` returns `[]`). `started` rows are skipped unless `include_started`,
/// but every raw line is parsed and hashed either way, so a malformed line is refused in both
/// modes and the chain a writer extends is the one over all lines.
pub fn read_entries(path: &Path, include_started: bool) -> Result<Vec<LedgerEntry>, LedgerError> {
    let ioe = |e: std::io::Error| LedgerError::Io(format!("{}: {e}", path.display()));
    let len = match fs::metadata(path) {
        Ok(m) => m.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(ioe(e)),
    };
    if len > MAX_LEDGER_BYTES {
        return refuse(format!(
            "{}: {len} bytes exceeds the {MAX_LEDGER_BYTES}-byte reader bound",
            path.display()
        ));
    }
    let bytes = fs::read(path).map_err(ioe)?;
    let mut out = Vec::new();
    for (i, line) in raw_lines(&bytes).into_iter().enumerate() {
        let bad = |why: String| {
            LedgerError::Refused(format!("{}: line {}: {why}", path.display(), i + 1))
        };
        let v: Value = serde_json::from_slice(line).map_err(|e| bad(e.to_string()))?;
        let field = |k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| bad(format!("no string field `{k}`")))
        };
        let status = Status::parse(&field("status")?).map_err(|e| bad(e.to_string()))?;
        let extras = RowExtras::from_line(&v).map_err(|e| bad(e.to_string()))?;
        if include_started || status != Status::Started {
            out.push(LedgerEntry {
                row_id: field("row_id")?,
                status,
                extras,
                line_sha256: hex(&Sha256::digest(line)),
            });
        }
    }
    Ok(out)
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
        let probe = |what: &str, field: &str| {
            TriState::not_run(format!("{what} dry run not executed")).to_json(field)
        };
        Ok(obj([
            ("torch", Value::from(self.torch.as_str())),
            (
                "transformers_sha",
                Value::from(self.transformers_sha.as_str()),
            ),
            ("device", Value::from(self.device.as_str())),
            ("host", Value::from(self.host.as_str())),
            ("fla_present", probe("fla", "env.fla_present")?),
            (
                "causal_conv1d_present",
                probe("causal-conv1d", "env.causal_conv1d_present")?,
            ),
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
        self.line_with(stamp, &RowExtras::default())
    }

    /// [`Self::line`] with the optional fields `extras` records, written under their Python
    /// names after the core row and sorted into place by the canonical dump.
    pub fn line_with(&self, stamp: &Stamp, extras: &RowExtras) -> Result<String, LedgerError> {
        extras.validate()?;
        if self.status == Status::Started {
            return refuse(
                "status `started` is the lifecycle row Python's RunRecorder.__enter__ writes, \
                 with its six lifecycle metrics; the Rust trainer writes only rows that end a run",
            );
        }
        if !RUN_KINDS.contains(&self.run_kind.as_str()) {
            return refuse(format!(
                "run_kind {:?} is not one of {RUN_KINDS:?}",
                self.run_kind
            ));
        }
        if self.quick_reason.trim().is_empty() {
            return refuse(
                "quick=True requires quick_reason; every row this crate writes is quick (rule 8)",
            );
        }
        if !(self.wall_clock_s.is_finite() && self.wall_clock_s >= 0.0) {
            return refuse(format!(
                "wall_clock_s {} is not a measured duration",
                self.wall_clock_s
            ));
        }
        if !self.recipe.as_object().is_some_and(|o| !o.is_empty()) {
            return refuse(
                "recipe is not a non-empty object: a row with a recipe_hash always had settings, and \
                 `LedgerRow` refuses an empty one",
            );
        }
        let recipe_hash = hex(&Sha256::digest(
            dumps(&self.recipe, CANONICAL_ASCII)?.as_bytes(),
        ));
        if recipe_hash != self.protocol.recipe_hash {
            return refuse(format!(
                "protocol.recipe_hash {} is not the hash of the stored recipe ({recipe_hash})",
                self.protocol.recipe_hash
            ));
        }
        check_optimizer_label(&self.recipe, self.status, &self.metrics, extras)?;
        let gates: BTreeMap<String, TriState> = REQUIRED_GATES
            .iter()
            .map(|g| {
                (
                    g.to_string(),
                    TriState::not_run(format!("gate '{g}' was never evaluated by this run")),
                )
            })
            .collect();
        let controls: BTreeMap<String, TriState> = REQUIRED_CONTROLS
            .iter()
            .map(|c| {
                (
                    c.to_string(),
                    TriState::not_run(format!("control '{c}' was never evaluated by this run")),
                )
            })
            .collect();
        let mut fields: Vec<(&str, Value)> = vec![
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
        ];
        if let Some(b) = &extras.val_without_train {
            fields.push(("val_without_train", b.clone()));
        }
        if let Some(d) = &extras.train_manifest_sha256 {
            fields.push(("train_manifest_sha256", Value::from(d.as_str())));
        }
        let row = obj(fields)?;
        Ok(dumps(&row, CANONICAL)?)
    }
}

impl FtRow {
    /// The row's JSON with `stamp`'s fields: the exact line `Ledger.append` writes, newline
    /// excluded. A [`Row`] of kind `ft`, with the trainer's own rule on top: a schedule that did
    /// not run to its end is quick for that reason too.
    pub fn line(&self, stamp: &Stamp) -> Result<String, LedgerError> {
        self.line_with(stamp, &RowExtras::default())
    }

    /// [`Self::line`] with the optional fields `extras` records ([`Row::line_with`]).
    pub fn line_with(&self, stamp: &Stamp, extras: &RowExtras) -> Result<String, LedgerError> {
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
        .line_with(stamp, extras)
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
pub fn ft_metrics(
    result: &TrainResult,
    facts: &RunFacts<'_>,
) -> Result<BTreeMap<String, TriState>, LedgerError> {
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
    m.insert(
        "train.optimizer_steps".to_string(),
        TriState::ran(true, steps),
    );
    m.insert(
        "train.micro_batches".to_string(),
        TriState::ran(true, result.micro_batches),
    );
    let c = &result.counts;
    m.insert(
        "train.supervised_tokens".to_string(),
        TriState::ran(true, c.supervised_tokens)
            .with_coverage(c.supervised_tokens, c.total_positions),
    );
    m.insert(
        "train.span_rows".to_string(),
        TriState::ran(true, c.span_rows),
    );
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
    m.insert(
        "train.loss_log_digest".to_string(),
        TriState::ran(true, result.loss_log_digest()?),
    );
    m.insert(
        "train.consumed_digest".to_string(),
        TriState::ran(true, result.consumed_digest.as_str()).with_detail(format!(
            "ConsumedPrefix over the {} batch(es) this run consumed",
            result.consumed_n
        )),
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
        TriState::ran(true, facts.provider).with_detail(
            "what this row ran on; compare rows only where this agrees or says why not",
        ),
    );
    Ok(m)
}

/// The rows this crate may write go to `ledger/mac-ojas-*.jsonl` and nowhere else.
pub fn check_ledger_path(path: &Path) -> Result<(), LedgerError> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if !(name.starts_with("mac-ojas-")
        && name.ends_with(".jsonl")
        && name.len() > "mac-ojas-.jsonl".len())
    {
        return refuse(format!(
            "{}: the Rust trainer writes only ledger/mac-ojas-*.jsonl, never a campaign ledger",
            path.display()
        ));
    }
    if parent != "ledger" {
        return refuse(format!(
            "{}: the file must sit in a directory named `ledger`",
            path.display()
        ));
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
pub fn append(
    path: &Path,
    row: &FtRow,
    row_id: String,
    written_at: String,
) -> Result<Stamp, LedgerError> {
    append_with(path, row, &RowExtras::default(), row_id, written_at)
}

/// [`append`] recording the optional fields `extras` carries ([`FtRow::line_with`]). With
/// `RowExtras::default()` it writes the line [`append`] always wrote.
pub fn append_with(
    path: &Path,
    row: &FtRow,
    extras: &RowExtras,
    row_id: String,
    written_at: String,
) -> Result<Stamp, LedgerError> {
    check_ledger_path(path)?;
    append_line(path, row_id, written_at, |stamp| {
        row.line_with(stamp, extras)
    })
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
    let mut f = fs::File::options()
        .create(true)
        .append(true)
        .read(true)
        .open(path)
        .map_err(ioe)?;
    f.lock().map_err(ioe)?;
    let result = (|| {
        let mut bytes = Vec::new();
        fs::File::open(path)
            .and_then(|mut r| r.read_to_end(&mut bytes))
            .map_err(ioe)?;
        let lines = raw_lines(&bytes);
        let needle = format!("\"row_id\":\"{row_id}\"");
        if lines
            .iter()
            .any(|l| l.windows(needle.len()).any(|w| w == needle.as_bytes()))
        {
            return refuse(format!(
                "row_id {row_id} already present: the ledger is append-only"
            ));
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
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    ))
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
    let frac = if micros == 0 {
        String::new()
    } else {
        format!(".{micros:06}")
    };
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}{frac}+00:00"
    ))
}

/// `_git_commit`: HEAD, with `-dirty` when the tree has any change (an unknown(...) string when
/// git cannot answer, never a guess).
pub fn git_commit(repo: &Path) -> String {
    let run = |args: &[&str]| -> Result<String, String> {
        let out = Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .map_err(|e| e.to_string())?;
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
        return refuse(format!(
            "lane {lane:?} and date {date:?} must be [A-Za-z0-9-]+"
        ));
    }
    let p = repo
        .join("ledger")
        .join(format!("mac-ojas-{lane}-{date}.jsonl"));
    check_ledger_path(&p)?;
    Ok(p)
}

/// `recipe.optimizer_recipe` of a LoRA-2B run (D6): AdamW over f32 adapters, D17 and the span
/// head over a frozen base. It is in Python's `real_ft_run.OPTIMIZER_RECIPES` so a ledger reader
/// never pools it with `master`, `bf16`, `kahan` or `f32`.
pub const LORA_OPTIMIZER_RECIPE: &str = "lora";

/// Why a LoRA row is quick (rule 8), named rather than left as a bare `quick=true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraQuickReason {
    /// The 0.8B or tiny smoke: a mechanics check, never a result.
    Smoke,
    /// The schedule was cut short of the registered full recipe.
    TruncatedSchedule,
    /// `seeds` seeds were run; promotion needs three on the full schedule.
    FewerThanThreeSeeds { seeds: u32 },
}

impl LoraQuickReason {
    fn text(&self) -> Result<String, LedgerError> {
        Ok(match self {
            Self::Smoke => "smoke run, not a result".to_owned(),
            Self::TruncatedSchedule => "truncated schedule, not the full recipe".to_owned(),
            Self::FewerThanThreeSeeds { seeds } if *seeds < 3 => {
                format!("fewer than 3 seeds ({seeds})")
            }
            Self::FewerThanThreeSeeds { seeds } => {
                return refuse(format!(
                    "{seeds} seeds is not fewer than 3: that is not a reason this row is quick"
                ));
            }
        })
    }
}

/// The init identities `campaign/lora-2b-preregistered.json` `init.record_in_every_row` names
/// that a caller supplies. The D17 digest is computed from the host entry's shape, the LoRA-A
/// digest is recomputed by `attach_lora`'s rule and compared, and the base snapshot and the
/// train shard hash are the row's own recipe's (`backbone_snapshot`, `shard_hash`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraInit<'a> {
    /// `ft_data::span_head_content_digest` of the head the run started from.
    pub span_head_content_digest: &'a str,
    /// The digest of the A matrices the run started from, claimed; refused unless the rule at
    /// the row's seed gives it ([`crate::lora_init::verify_lora_a_digest`]).
    pub lora_a_digest: &'a str,
    /// sha256 of the base model's safetensors file.
    pub base_safetensors_sha256: &'a str,
    /// The pre-registration's `dtype_map_sha256` of the base's tensor dtypes.
    pub dtype_map_sha256: &'a str,
    /// `corpus.plan_seed`: the seed the batch order was drawn at.
    pub plan_seed: u64,
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The metrics [`lora_ft_row`] adds whose value is a lowercase hex sha256 / digest string. The
/// other two it adds are `corpus.plan_seed` (an unsigned integer) and `lora.param_groups`.
const LORA_ROW_HEX_METRICS: [&str; 5] = [
    "train.span_head_init_digest",
    "lora.d17_init_digest",
    "lora.a_init_digest",
    "lora.base_safetensors_sha256",
    "lora.dtype_map_sha256",
];

/// Every `recipe.optimizer_recipe` label a row may carry: Python's `OPTIMIZER_RECIPES`
/// (`tools/real_ft_run.py`) and `python/qd_train/ledger.py`'s `OPTIMIZER_RECIPE_LABELS`. A
/// label outside the set is refused whatever its spelling, so no near-spelling of `lora` is a
/// row that no gate recognises as one.
pub const OPTIMIZER_RECIPE_LABELS: [&str; 5] = ["bf16", "master", "kahan", "f32", "lora"];

/// `lora.param_groups` as [`lora_ft_row`] writes it: `name xN at lr_scale S`, comma-separated.
fn is_param_group_summary(s: &str) -> bool {
    !s.is_empty()
        && s.split(", ").all(|part| {
            let Some((head, scale)) = part.rsplit_once(" at lr_scale ") else {
                return false;
            };
            let Some((name, count)) = head.rsplit_once(" x") else {
                return false;
            };
            !name.is_empty()
                && count.bytes().all(|b| b.is_ascii_digit())
                && !count.starts_with('0')
                && count.parse::<usize>().is_ok()
                && scale.parse::<f64>().is_ok_and(|v| v.is_finite() && v >= 0.0)
        })
}

/// The gate every ft line passes ([`Row::line_with`]): `recipe.optimizer_recipe`, when a recipe
/// states one, is one of [`OPTIMIZER_RECIPE_LABELS`] exactly, and a row labelled
/// [`LORA_OPTIMIZER_RECIPE`] carries what [`lora_ft_row`] writes: the seven metrics as passed
/// `ran` records (digests lowercase hex sha256, `corpus.plan_seed` an unsigned integer,
/// `lora.param_groups` a group summary) and the train manifest's sha256, which the release
/// export requires. A `not_run`, null, failed or malformed metric is not evidence. A row
/// labelled otherwise is not looked at beyond the label, so its line is what it always was.
fn check_optimizer_label(
    recipe: &Value,
    status: Status,
    metrics: &BTreeMap<String, TriState>,
    extras: &RowExtras,
) -> Result<(), LedgerError> {
    let Some(label) = recipe.get("optimizer_recipe") else {
        return Ok(());
    };
    let Some(label) = label.as_str().filter(|l| OPTIMIZER_RECIPE_LABELS.contains(l)) else {
        return refuse(format!(
            "recipe.optimizer_recipe {label} is not one of {OPTIMIZER_RECIPE_LABELS:?}"
        ));
    };
    // A failed or killed row records that a run did not finish: it is no LoRA result and its
    // failure must stay writable even when the run died before its digests existed.
    if label != LORA_OPTIMIZER_RECIPE || status != Status::Completed {
        return Ok(());
    }
    let value_of = |name: &str| -> Result<&Value, LedgerError> {
        match metrics.get(name) {
            None => refuse(format!(
                "a row labelled {LORA_OPTIMIZER_RECIPE:?} lacks metric {name}: only lora_ft_row \
                 builds a lora row"
            )),
            Some(TriState::NotRun { reason }) => refuse(format!(
                "a row labelled {LORA_OPTIMIZER_RECIPE:?} has metric {name} not_run ({reason}): \
                 that is not evidence"
            )),
            Some(TriState::Ran { passed: false, .. }) => refuse(format!(
                "a row labelled {LORA_OPTIMIZER_RECIPE:?} has metric {name} failed"
            )),
            Some(TriState::Ran { value, .. }) => Ok(value),
        }
    };
    for name in LORA_ROW_HEX_METRICS {
        if !value_of(name)?.as_str().is_some_and(is_hex64) {
            return refuse(format!(
                "metric {name} of a lora row is not a lowercase hex sha256 string"
            ));
        }
    }
    if value_of("corpus.plan_seed")?.as_u64().is_none() {
        return refuse("metric corpus.plan_seed of a lora row is not an unsigned integer");
    }
    if !value_of("lora.param_groups")?
        .as_str()
        .is_some_and(is_param_group_summary)
    {
        return refuse(
            "metric lora.param_groups of a lora row is not a `name xN at lr_scale S` summary",
        );
    }
    if extras.train_manifest_sha256.is_none() {
        return refuse(format!(
            "a row labelled {LORA_OPTIMIZER_RECIPE:?} records no train_manifest_sha256: the \
             release export refuses it"
        ));
    }
    Ok(())
}

/// sha256 (over file contents) of every `tools/span_refit_score.py` that emits the span null
/// (`campaign/span-refit-mac-preregistered.json` amendment 2): Python's
/// `SPAN_REFIT_SPAN_NULL_SCORER_SHAS`. A completed `span-refit-*` row whose recipe pins one of
/// these as `script_sha256["span_refit_score.py"]` carries [`SPAN_REFIT_METRICS`].
pub const SPAN_REFIT_SPAN_NULL_SCORER_SHAS: [&str; 1] =
    ["8dc59bd4df5f14e4bd106982f3d40119bf33380268e54c87520b166d1f77b2ed"];
const SPAN_REFIT_TAG_PREFIX: &str = "span-refit-";
const SPAN_REFIT_VALIDITY_GATE: &str = "span_refit.validity_gate";
/// The three metrics `span_beats_null` reads (`qd_train.span_null`).
pub const SPAN_REFIT_METRICS: [&str; 3] = [
    "val_span_start_top1_over_chance",
    "val_span_end_top1_over_chance",
    "val_span_pointing_added_line_chance",
];

/// Python's `LedgerRow.check_span_refit_span_null`: a completed span-refit row recorded under a
/// scorer that emits the span null holds the three metrics as `ran` records over one pointing-row
/// count. Not looked at: any other status, tag or scorer sha (it predates the null), and arm (a)'s
/// row after its validity gate failed, which records no span reading.
fn check_span_refit_span_null(
    recipe: &Value,
    status: Status,
    metrics: &BTreeMap<String, TriState>,
    gates: &BTreeMap<String, TriState>,
) -> Result<(), LedgerError> {
    let Some(tag) = recipe
        .get("tag")
        .and_then(Value::as_str)
        .filter(|t| t.starts_with(SPAN_REFIT_TAG_PREFIX))
    else {
        return Ok(());
    };
    if status != Status::Completed {
        return Ok(());
    }
    let pinned = recipe
        .get("script_sha256")
        .and_then(|s| s.get("span_refit_score.py"))
        .and_then(Value::as_str);
    if !pinned.is_some_and(|s| SPAN_REFIT_SPAN_NULL_SCORER_SHAS.contains(&s)) {
        return Ok(());
    }
    if matches!(
        gates.get(SPAN_REFIT_VALIDITY_GATE),
        Some(TriState::Ran { passed: false, .. })
    ) {
        return Ok(());
    }
    let mut totals = std::collections::BTreeSet::new();
    for name in SPAN_REFIT_METRICS {
        match metrics.get(name) {
            Some(TriState::Ran { coverage, .. }) => {
                totals.insert(coverage.map(|c| c.n_total));
            }
            _ => {
                return refuse(format!(
                    "a completed {tag} row recorded under a span-null scorer lacks {name} as a \
                     'ran' metric: the scorer emits it (amendment 2)"
                ));
            }
        }
    }
    if totals.len() != 1 || totals.contains(&None) {
        return refuse(format!(
            "a completed {tag} row's span top-1 and null do not state one pointing-row count"
        ));
    }
    Ok(())
}

/// The reading side of [`check_optimizer_label`]: the verdict Rust's gate gives a ledger line
/// that already exists as bytes, the way Python's `LedgerRow.from_json` plus
/// `check_optimizer_label` give it. It reads only what the gate reads (`status`, `recipe`,
/// `metrics` as tri-states, `train_manifest_sha256`), so `crates/qd-train/tests/lora_gate_parity.rs`
/// and `python/tests/test_lora_gate_parity.py` can hand both gates the same bytes.
pub fn check_row_line(line: &str) -> Result<(), LedgerError> {
    let v: Value = serde_json::from_str(line)
        .map_err(|e| LedgerError::Refused(format!("the line is not JSON: {e}")))?;
    let status = match v.get("status").and_then(Value::as_str) {
        Some("started") => Status::Started,
        Some("completed") => Status::Completed,
        Some("killed") => Status::Killed,
        Some("failed") => Status::Failed,
        other => return refuse(format!("status {other:?} is not a ledger status")),
    };
    let extras = RowExtras::from_line(&v)?;
    let mut metrics = BTreeMap::new();
    match v.get("metrics") {
        Some(Value::Object(m)) => {
            for (name, raw) in m {
                metrics.insert(
                    name.clone(),
                    crate::tristate::parse_tristate(raw, &format!("metrics.{name}"))?,
                );
            }
        }
        _ => return refuse("metrics is not an object"),
    }
    let mut gates = BTreeMap::new();
    if let Some(Value::Object(g)) = v.get("gates") {
        for (name, raw) in g {
            gates.insert(
                name.clone(),
                crate::tristate::parse_tristate(raw, &format!("gates.{name}"))?,
            );
        }
    }
    let recipe = v.get("recipe").cloned().unwrap_or(Value::Null);
    check_optimizer_label(&recipe, status, &metrics, &extras)?;
    check_span_refit_span_null(&recipe, status, &metrics, &gates)
}

/// Turn an assembled `ft` row into a `lora` row: the checks a LoRA run owes, then the digests
/// in its metrics. `row.recipe.optimizer_recipe` must already be [`LORA_OPTIMIZER_RECIPE`] and
/// name no lower-layer split; `entries` and `host` are the plan the run trained
/// ([`crate::recipe::lora_param_groups`] refuses a plan the pre-registration does not allow, so a
/// row cannot be written for one); the row's seed is `row.protocol.seed`, the seed
/// `attach_lora` is recomputed at. Every row is quick, and `reasons` must name why.
///
/// Metrics added (none may already be present): `train.span_head_init_digest` (Python's name),
/// `lora.d17_init_digest`, `lora.a_init_digest`, `lora.base_safetensors_sha256`,
/// `lora.dtype_map_sha256`, `corpus.plan_seed` (Python's name) and `lora.param_groups`. They are
/// metrics, not recipe keys, so the recipe hash is the recipe's alone.
pub fn lora_ft_row(
    mut row: FtRow,
    init: &LoraInit<'_>,
    entries: &[ParamSpec],
    host: &[ParamSpec],
    reasons: &[LoraQuickReason],
) -> Result<FtRow, LedgerError> {
    if row.recipe.optimizer_recipe != LORA_OPTIMIZER_RECIPE {
        return refuse(format!(
            "recipe.optimizer_recipe is {:?}, not {LORA_OPTIMIZER_RECIPE:?}: this writer labels a \
             LoRA run and nothing else",
            row.recipe.optimizer_recipe
        ));
    }
    if row.recipe.lower_layers.is_some() {
        return refuse("a lora row names no lower-layer split: D6 dropped the 0.1x split for LoRA");
    }
    if reasons.is_empty() {
        return refuse(
            "a lora row is quick (rule 8) and says why: smoke, truncated schedule, or fewer than \
             3 seeds",
        );
    }
    for (what, v) in [
        (
            "recipe.shard_hash (the train shard hash)",
            row.recipe.shard_hash.as_str(),
        ),
        (
            "span-head init content digest",
            init.span_head_content_digest,
        ),
        ("LoRA-A digest", init.lora_a_digest),
        ("base safetensors sha256", init.base_safetensors_sha256),
        ("dtype_map_sha256", init.dtype_map_sha256),
    ] {
        if !is_hex64(v) {
            return refuse(format!("{what} {v:?} is not a lowercase hex sha256"));
        }
    }
    let groups = crate::recipe::lora_param_groups(entries, host)
        .map_err(|e| LedgerError::Refused(format!("the lora plan: {e}")))?;
    let (rank, modules) = crate::recipe::lora_adapter_layout(entries)
        .map_err(|e| LedgerError::Refused(format!("the lora plan: {e}")))?;
    let a_digest = crate::lora_init::verify_lora_a_digest(
        row.protocol.seed,
        rank,
        &modules,
        init.lora_a_digest,
    )
    .map_err(|e| LedgerError::Refused(e.to_string()))?;
    let d17 = host
        .iter()
        .find(|h| h.name == crate::d17::D17_ENTRY)
        .ok_or_else(|| {
            LedgerError::Refused(
                "the host holds no D17 entry, so its init digest cannot be recorded".into(),
            )
        })?;
    let &[d17_rows, d17_hidden] = d17.shape.as_slice() else {
        return refuse(format!("D17 has shape {:?}, not [17, hidden]", d17.shape));
    };
    if d17_rows != 17 {
        return refuse(format!(
            "D17 has {d17_rows} rows, the registered readout has 17"
        ));
    }
    let d17_digest = crate::lora_init::d17_zero_init_digest(d17_rows, d17_hidden)
        .map_err(|e| LedgerError::Refused(e.to_string()))?;
    let reason_text = reasons
        .iter()
        .map(LoraQuickReason::text)
        .collect::<Result<Vec<_>, _>>()?
        .join("; ");
    row.quick_reason = if row.quick_reason.trim().is_empty() {
        format!("lora: {reason_text}")
    } else {
        format!("lora: {reason_text}; {}", row.quick_reason)
    };
    let summary = groups
        .iter()
        .map(|g| format!("{} x{} at lr_scale {}", g.name, g.params.len(), g.lr_scale))
        .collect::<Vec<_>>()
        .join(", ");
    let additions = [
        (
            "train.span_head_init_digest",
            TriState::ran(true, init.span_head_content_digest).with_detail(
                "run_control._sidecar_digest over the qd-tensor-ref-v1 digests of the span head \
                 the run started from",
            ),
        ),
        (
            "lora.d17_init_digest",
            TriState::ran(true, d17_digest).with_detail(format!(
                "TensorRef.digest of a float32 zero tensor [{d17_rows}, {d17_hidden}], computed \
                 from the host entry {}",
                crate::d17::D17_ENTRY
            )),
        ),
        (
            "lora.a_init_digest",
            TriState::ran(true, a_digest).with_detail(format!(
                "recomputed by attach_lora's rule (one CPU mt19937 at seed {}, kaiming_uniform_ \
                 a=sqrt(5), {} module(s) in canonical draw order (layer index, then D6's suffix table), rank {rank}) and equal to the \
                 digest the run recorded",
                row.protocol.seed,
                modules.len()
            )),
        ),
        (
            "lora.base_safetensors_sha256",
            TriState::ran(true, init.base_safetensors_sha256).with_detail(format!(
                "sha256 of the base's safetensors at snapshot {}",
                row.recipe.backbone_snapshot
            )),
        ),
        (
            "lora.dtype_map_sha256",
            TriState::ran(true, init.dtype_map_sha256)
                .with_detail("the pre-registration's dtype_map_sha256 formula over the base"),
        ),
        (
            "corpus.plan_seed",
            TriState::ran(true, init.plan_seed)
                .with_detail("the seed the plan was drawn at, as real_ft_run records it"),
        ),
        ("lora.param_groups", TriState::ran(true, summary)),
    ];
    for (name, state) in additions {
        if row.metrics.insert(name.to_owned(), state).is_some() {
            return refuse(format!(
                "metric {name} is already on the row; a lora digest is never overwritten"
            ));
        }
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_metal_full_ft_row_is_never_labelled_master() {
        assert_eq!(METAL_FULL_FT_OPTIMIZER_RECIPE, "f32");
        assert_ne!(METAL_FULL_FT_OPTIMIZER_RECIPE, "master");
    }

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
        assert_eq!(
            isoformat_utc(t).unwrap(),
            "2026-09-21T14:13:20.123456+00:00"
        );
        let whole = UNIX_EPOCH + std::time::Duration::new(1_790_000_000, 0);
        assert_eq!(
            isoformat_utc(whole).unwrap(),
            "2026-09-21T14:13:20+00:00",
            "no fraction at 0 us"
        );
        assert_eq!(
            isoformat_utc(UNIX_EPOCH + std::time::Duration::new(951_782_400, 0)).unwrap(),
            "2000-02-29T00:00:00+00:00"
        );
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
