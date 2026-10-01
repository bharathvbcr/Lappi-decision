//! The Rust trainer's `ft` ledger row, in the bytes `python/qd_train/ledger.py` writes and reads.
//!
//! **Where it may write.** Only `ledger/mac-ojas-*.jsonl` (`HANDOFF/ojas-training-2026-10-01.md`,
//! invariants): never a campaign ledger, never a box path. And every row is `quick=True` with a
//! non-empty `quick_reason` (rule 8): the type has no way to say otherwise.
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
//! **What the scorer pairs on** (`tools/real_ft_run.py`): `_ft_row` needs `run_kind == "ft"`
//! and `status == "completed"`; `_ft_row_mismatches` compares `recipe.tag == "epoch"`,
//! `recipe.device`, `protocol.seed`, `recipe.shard_hash` and `recipe.backbone_snapshot`;
//! `_seed_weights` reads `metrics["train.optimizer_steps"]["value"]` as an integer;
//! `_scoring_step` reads `recipe.attn_implementation`, `recipe.lr`, `recipe.span_weight` and
//! `recipe.optimizer_recipe`; `ScoredModel` reads `metrics["train.termination"]["value"]`.
//! [`FtRecipe::to_json`] and [`ft_metrics`] carry every one of those. The `recipe_hash` is
//! sha256 over `json.dumps(recipe, sort_keys=True, separators=(",", ":"))`, the bytes
//! `real_ft_run._protocol` hashes, so Python recomputes the stated hash from the stored recipe.
//! (`GAP-LTRAINER-METAL-ARTIFACT-CONTRACT-PENDING-2026-10-01`: L-oracle's ft-row contract and
//! L-scorer's Metal reader are not written yet; this is L-trainer's reading of the code.)

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::pyjson::{Json, PyJsonError};
use crate::run_control::hex;
use crate::trainer::{Termination, TrainResult};

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
    #[error("io: {0}")]
    Io(String),
}

fn refuse<T>(m: impl Into<String>) -> Result<T, LedgerError> {
    Err(LedgerError::Refused(m.into()))
}

/// A check that ran (with whether it passed) or did not (with why). `tristate.py`'s two cases.
#[derive(Debug, Clone, PartialEq)]
pub enum TriState {
    Ran {
        passed: bool,
        value: Option<Json>,
        /// `(n, n_total)`, carried together or not at all.
        coverage: Option<(u64, u64)>,
        detail: String,
    },
    NotRun {
        reason: String,
    },
}

impl TriState {
    pub fn ran(passed: bool, value: Json) -> Self {
        TriState::Ran {
            passed,
            value: Some(value),
            coverage: None,
            detail: String::new(),
        }
    }

    pub fn with_detail(self, d: impl Into<String>) -> Self {
        match self {
            TriState::Ran {
                passed, value, coverage, ..
            } => TriState::Ran {
                passed,
                value,
                coverage,
                detail: d.into(),
            },
            other => other,
        }
    }

    pub fn with_coverage(self, n: u64, n_total: u64) -> Self {
        match self {
            TriState::Ran { passed, value, detail, .. } => TriState::Ran {
                passed,
                value,
                coverage: Some((n, n_total)),
                detail,
            },
            other => other,
        }
    }

    pub fn not_run(reason: impl Into<String>) -> Self {
        TriState::NotRun { reason: reason.into() }
    }

    /// `Ran.to_json` / `NotRun.to_json`, with `Ran.__post_init__`'s and `NotRun`'s checks.
    pub fn to_json(&self) -> Result<Json, LedgerError> {
        match self {
            TriState::NotRun { reason } => {
                if reason.trim().is_empty() {
                    return refuse("NotRun requires a non-empty reason");
                }
                Ok(Json::obj([("state", Json::str("not_run")), ("reason", Json::str(reason.clone()))])?)
            }
            TriState::Ran {
                passed,
                value,
                coverage,
                detail,
            } => {
                let mut pairs = vec![("state", Json::str("ran")), ("passed", Json::Bool(*passed))];
                if let Some(v) = value
                    && *v != Json::Null
                {
                    pairs.push(("value", v.clone()));
                }
                if let Some((n, total)) = coverage {
                    if n > total {
                        return refuse(format!("examined {n} of {total}: n exceeds n_total"));
                    }
                    pairs.push(("n", int(*n)?));
                    pairs.push(("n_total", int(*total)?));
                }
                if !detail.is_empty() {
                    pairs.push(("detail", Json::str(detail.clone())));
                }
                Ok(Json::obj(pairs)?)
            }
        }
    }

    fn termination_value(&self) -> Option<&str> {
        match self {
            TriState::Ran {
                value: Some(Json::Str(s)), ..
            } => Some(s),
            _ => None,
        }
    }
}

fn int(x: u64) -> Result<Json, LedgerError> {
    Ok(Json::Int(i64::try_from(x).map_err(|e| LedgerError::Refused(format!("{x}: {e}")))?))
}

/// The ft recipe as the row stores (and hashes) it. Keys follow `real_ft_run._train`'s recipe
/// and `_recipe_pieces`, plus the keys only a Metal run has (`provider`, `operands`,
/// `optimizer_groups`), so its rows hash apart from every PyTorch row.
#[derive(Debug, Clone, PartialEq)]
pub struct FtRecipe {
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
    pub extra: BTreeMap<String, Json>,
}

impl FtRecipe {
    pub fn to_json(&self) -> Result<Json, LedgerError> {
        if self.backbone_snapshot.contains('/') || self.backbone_snapshot.contains('\\') {
            return refuse(format!(
                "backbone_snapshot is {:?}, a path rather than a revision; it feeds recipe_hash and \
                 an absolute path differs between machines",
                self.backbone_snapshot
            ));
        }
        let mut m: BTreeMap<String, Json> = BTreeMap::new();
        let mut put = |k: &str, v: Json| -> Result<(), LedgerError> {
            if m.insert(k.to_string(), v).is_some() {
                return refuse(format!("recipe key {k} given twice"));
            }
            Ok(())
        };
        put("tool", Json::str("crates/qd-train-metal"))?;
        put("tag", Json::str(self.tag.clone()))?;
        put("device", Json::str(self.device.clone()))?;
        put("lr", Json::Float(self.lr))?;
        put("passes", int(self.passes)?)?;
        put("batches", int(self.batches)?)?;
        put("width", int(self.width)?)?;
        put("span_weight", Json::Float(self.span_weight))?;
        put("deterministic", Json::Bool(self.deterministic))?;
        put("shard_hash", Json::str(self.shard_hash.clone()))?;
        put("backbone_snapshot", Json::str(self.backbone_snapshot.clone()))?;
        put("backbone_vocab", int(self.backbone_vocab)?)?;
        put("backbone_params", int(self.backbone_params)?)?;
        put("attn_implementation", Json::str(self.attn_implementation.clone()))?;
        put("optimizer_recipe", Json::str(self.optimizer_recipe.clone()))?;
        put("provider", Json::str(self.provider.clone()))?;
        put("operands", Json::str(self.operands.clone()))?;
        put("optimizer_groups", Json::str(self.optimizer_groups.clone()))?;
        if self.train_attention_mask != "padding" {
            put("train_attention_mask", Json::str(self.train_attention_mask.clone()))?;
        }
        if let Some(c) = self.wall_clock_cap_s
            && c != crate::run_control::DEFAULT_CAP_S
        {
            put("wall_clock_cap_s", Json::Float(c))?;
        }
        if let Some(b) = self.batch_tokens {
            put("batch_tokens", int(b)?)?;
        }
        if self.no_memorise {
            put("no_memorise", Json::Bool(true))?;
        }
        if let Some(b2) = self.beta2
            && b2 != crate::recipe::DEFAULT_BETA2
        {
            put("beta2", Json::Float(b2))?;
        }
        if let Some((n, s)) = self.lower_layers {
            put("lower_layers_n", int(n)?)?;
            put("lower_lr_scale", Json::Float(s))?;
        }
        for (k, v) in &self.extra {
            put(k, v.clone())?;
        }
        Ok(Json::Obj(m))
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
        let body = recipe.to_json()?.dumps(true)?;
        Ok(Self {
            data_snapshot_hash: data_snapshot_hash.to_string(),
            tokenizer_hash: tokenizer_hash.to_string(),
            backbone_commit: format!("{}:vocab{}", recipe.backbone_snapshot, recipe.backbone_vocab),
            recipe_hash: hex(&Sha256::digest(body.as_bytes())),
            seed,
        })
    }

    pub fn to_json(&self) -> Result<Json, LedgerError> {
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
        Ok(Json::obj([
            ("data_snapshot_hash", Json::str(self.data_snapshot_hash.clone())),
            ("tokenizer_hash", Json::str(self.tokenizer_hash.clone())),
            ("backbone_commit", Json::str(self.backbone_commit.clone())),
            ("recipe_hash", Json::str(self.recipe_hash.clone())),
            ("seed", int(self.seed)?),
        ])?)
    }

    /// `Protocol.hash`: sha256 over the canonical protocol (`ensure_ascii=False`).
    pub fn hash(&self) -> Result<String, LedgerError> {
        Ok(hex(&Sha256::digest(self.to_json()?.dumps(false)?.as_bytes())))
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

    fn to_json(&self) -> Result<Json, LedgerError> {
        let probe = |what: &str| TriState::not_run(format!("{what} dry run not executed")).to_json();
        Ok(Json::obj([
            ("torch", Json::str(self.torch.clone())),
            ("transformers_sha", Json::str(self.transformers_sha.clone())),
            ("device", Json::str(self.device.clone())),
            ("host", Json::str(self.host.clone())),
            ("fla_present", probe("fla")?),
            ("causal_conv1d_present", probe("causal-conv1d")?),
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

impl FtRow {
    /// The row's JSON with `stamp`'s fields: the exact line `Ledger.append` writes, newline
    /// excluded.
    pub fn line(&self, stamp: &Stamp) -> Result<String, LedgerError> {
        if self.quick_reason.trim().is_empty() {
            return refuse("quick=True requires quick_reason; every ojas row is quick (rule 8)");
        }
        if !(self.wall_clock_s.is_finite() && self.wall_clock_s >= 0.0) {
            return refuse(format!("wall_clock_s {} is not a measured duration", self.wall_clock_s));
        }
        let recipe_hash = hex(&Sha256::digest(self.recipe.to_json()?.dumps(true)?.as_bytes()));
        if recipe_hash != self.protocol.recipe_hash {
            return refuse(format!(
                "protocol.recipe_hash {} is not the hash of the stored recipe ({recipe_hash})",
                self.protocol.recipe_hash
            ));
        }
        // `_quick_if_truncated`: a schedule that did not run to its end is a truncated one.
        let mut quick_reason = self.quick_reason.clone();
        if let Some(term) = self.metrics.get("train.termination") {
            let said = term.termination_value().unwrap_or("not_run");
            if said != "steps_exhausted" {
                quick_reason = format!(
                    "{quick_reason}; train.termination is '{said}', not 'steps_exhausted': the schedule \
                     did not run to its end, which rule 8 calls a truncated schedule"
                );
            }
        }
        let tri = |m: &BTreeMap<String, TriState>| -> Result<Json, LedgerError> {
            Ok(Json::Obj(
                m.iter()
                    .map(|(k, v)| Ok((k.clone(), v.to_json()?)))
                    .collect::<Result<_, LedgerError>>()?,
            ))
        };
        let gates: BTreeMap<String, TriState> = REQUIRED_GATES
            .iter()
            .map(|g| (g.to_string(), TriState::not_run(format!("gate '{g}' was never evaluated by this run"))))
            .collect();
        let controls: BTreeMap<String, TriState> = REQUIRED_CONTROLS
            .iter()
            .map(|c| (c.to_string(), TriState::not_run(format!("control '{c}' was never evaluated by this run"))))
            .collect();
        let row = Json::obj([
            ("row_id", Json::str(stamp.row_id.clone())),
            ("written_at", Json::str(stamp.written_at.clone())),
            (
                "prev_row_hash",
                stamp.prev_row_hash.clone().map_or(Json::Null, Json::Str),
            ),
            ("protocol_hash", Json::Str(self.protocol.hash()?)),
            ("protocol", self.protocol.to_json()?),
            ("run_kind", Json::str("ft")),
            ("status", Json::str(self.status.as_str())),
            ("quick", Json::Bool(true)),
            ("quick_reason", Json::Str(quick_reason)),
            ("code_commit", Json::str(self.code_commit.clone())),
            ("env", self.env.to_json()?),
            ("metrics", tri(&self.metrics)?),
            ("noul_rate", TriState::not_run("noul rate not computed by this run").to_json()?),
            ("controls", tri(&controls)?),
            ("gates", tri(&gates)?),
            ("wall_clock_s", Json::Float(self.wall_clock_s)),
            (
                "wall_clock_source",
                Json::str(match self.wall_clock_source {
                    WallClockSource::Caller => "caller",
                    WallClockSource::Recorder => "recorder",
                }),
            ),
            ("cost_usd", Json::Float(0.0)),
            ("notes", Json::str(self.notes.clone())),
            ("recipe", self.recipe.to_json()?),
        ])?;
        Ok(row.dumps(false)?)
    }
}

/// The training metrics `trainer._train_loop` records, from a [`TrainResult`], plus the two a
/// Rust run adds: the consumed-batch digest and the provider that ran.
pub fn ft_metrics(result: &TrainResult, cap_s: f64, provider: &str) -> Result<BTreeMap<String, TriState>, LedgerError> {
    let mut m = BTreeMap::new();
    let steps = result.optimizer_steps;
    m.insert(
        "train.termination".to_string(),
        TriState::ran(result.termination != Termination::WallClockCap, Json::str(result.termination.as_str()))
            .with_detail(format!("ft stopped after {steps} optimizer step(s)")),
    );
    m.insert("train.optimizer_steps".to_string(), TriState::ran(true, int(steps)?));
    m.insert("train.micro_batches".to_string(), TriState::ran(true, int(result.micro_batches)?));
    let c = &result.counts;
    m.insert(
        "train.supervised_tokens".to_string(),
        TriState::ran(true, int(c.supervised_tokens)?).with_coverage(c.supervised_tokens, c.total_positions),
    );
    m.insert("train.span_rows".to_string(), TriState::ran(true, int(c.span_rows)?));
    let frac = if c.total_positions > 0 {
        c.padded_positions as f64 / c.total_positions as f64
    } else {
        0.0
    };
    m.insert(
        "train.padding_fraction".to_string(),
        TriState::ran(true, Json::Float(frac)).with_coverage(c.padded_positions, c.total_positions),
    );
    m.insert(
        "train.final_loss".to_string(),
        match result.loss_log.last() {
            Some(p) => TriState::ran(true, Json::Float(p.loss)),
            None => TriState::Ran {
                passed: false,
                value: None,
                coverage: None,
                detail: "no optimizer step completed".into(),
            },
        },
    );
    m.insert("train.loss_log_digest".to_string(), TriState::ran(true, Json::Str(result.loss_log_digest()?)));
    m.insert(
        "train.consumed_digest".to_string(),
        TriState::ran(true, Json::Str(result.consumed_digest.clone()))
            .with_detail(format!("ConsumedPrefix over the {} batch(es) this run consumed", result.consumed_n)),
    );
    m.insert(
        "train.projected_usd_at_cap".to_string(),
        TriState::ran(true, Json::Float(0.0)).with_detail(format!(
            "local-metal: the Mac's own GPU, not billed, capped at {:.2} h -> $0.00 at the cap -- no \
             approval required",
            cap_s / 3600.0
        )),
    );
    m.insert(
        "train.path".to_string(),
        TriState::ran(true, Json::str(provider.to_string()))
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

/// `Ledger.append`: take the write lock, refuse a duplicate `row_id`, chain to the last line,
/// append one line with `O_APPEND`, fsync. Returns the stamp the row was written with.
pub fn append(path: &Path, row: &FtRow, row_id: String, written_at: String) -> Result<Stamp, LedgerError> {
    check_ledger_path(path)?;
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
        let mut line = row.line(&stamp)?.into_bytes();
        line.push(b'\n');
        f.write_all(&line).map_err(ioe)?;
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

    #[test]
    fn a_tristate_is_checked_as_python_checks_it() {
        assert!(TriState::not_run("  ").to_json().is_err());
        assert!(TriState::ran(true, Json::Int(1)).with_coverage(3, 2).to_json().is_err());
        assert_eq!(
            TriState::ran(true, Json::Int(3)).with_coverage(3, 4).to_json().unwrap().dumps(false).unwrap(),
            "{\"n\":3,\"n_total\":4,\"passed\":true,\"state\":\"ran\",\"value\":3}"
        );
    }
}
