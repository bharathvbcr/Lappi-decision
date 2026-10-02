//! The v4 shard reader: a port of `python/qd_train/shards.py::ShardReader` and the parts of
//! `python/qd_train/artifacts.py` it is bound by (`ShardHeader`, `assert_shard_trainable`,
//! `RemapTable.read`, `Batch`, `bucket_for`, `padding_waste`), plus the consumed-prefix digest
//! of `run_control.ConsumedPrefix` / `trainer._fold`.
//!
//! # The door comes first, and it is the only way in
//!
//! [`ShardReader::open`] is the reader's only constructor, and before it opens any file in
//! the shard set it runs the rule-3 door ([`crate::held_out`]) in this order:
//!
//! 1. layers 1-2 on the shard directory's resolved path -- **before any file in it is read**
//!    (Python reads `header.json` first and checks the path after);
//! 2. the full four-layer door on the manifest the set was built from: layers 1-2 on its
//!    path, then its contents' hash, split, families and status;
//! 3. `header.json`, re-hashed, and rule 3 at the shard boundary: a `heldout` split, a split
//!    that is not trainable, a `report_only` set, or a header whose `data_snapshot_hash` or
//!    split is not the admitted manifest's is refused.
//!
//! Only then are `offsets.npy`, `supervision.npz`, `sequence_index.json` and the remap read;
//! `tokens.u32` is only ever *measured* at open, and its bytes are read by
//! [`ShardReader::sequence`] after the reader exists.
//!
//! # Deliberate differences from Python, each stricter
//!
//! * **The manifest is required.** Python's `ShardReader` checks the header's split and the
//!   path but never opens the manifest, so layers 3-4 (the manifest's own split, the family
//!   holdout) did not apply to a shard set at all. Here the set must name its manifest, the
//!   manifest must pass the door, and the header must pin that manifest's hash.
//! * **`report_only` is refused outright.** Python admits a report-only val set to any reader
//!   and leaves refusal to the gate paths; this reader exists to train.
//! * **The code fingerprint is checked against an explicit `qd_data_dir`.** Python compares
//!   the header's fingerprint with the `qd_data` it has loaded; a Rust process has none
//!   loaded, so the caller names the sources, and the check is required, not optional.
//! * **Every sequence's `row_id` must be in the manifest**, every slot kind must be a known
//!   non-LM kind at open (Python finds both only per batch or not at all), a missing
//!   `shard_hash` in `header.json` is refused (Python skips the comparison), candidate
//!   offsets must be non-decreasing, and every token id must be below `vocab_size` when read.

use std::collections::{BTreeMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::files::{OsFiles, RangedRead, ShardFiles};
use crate::held_out::{
    DataConfig, DoorError, HeldOutLayer, HeldOutViolation, SPLITS, TrainingManifest,
    assert_path_not_held_out, hex, open_training_manifest,
};
use crate::np_random::default_rng;
use crate::npy::NpyArray;
use crate::npz::StoredZip;
use crate::pyjson::{self, SORTED_DEFAULT};
use crate::tristate::{Coverage, TriState, parse_tristate};

/// `shards.HEADER_NAME`.
pub const HEADER_NAME: &str = "header.json";
/// `shards.TOKENS_NAME`.
pub const TOKENS_NAME: &str = "tokens.u32";
/// `shards.OFFSETS_NAME`.
pub const OFFSETS_NAME: &str = "offsets.npy";
/// `shards.SUPERVISION_NAME`.
pub const SUPERVISION_NAME: &str = "supervision.npz";
/// `shards.COVERAGE_NAME`.
pub const COVERAGE_NAME: &str = "coverage.json";
/// `shards.SPAN_CHECK_NAME`.
pub const SPAN_CHECK_NAME: &str = "span_check.json";
/// `shards.SEQUENCE_INDEX_NAME`.
pub const SEQUENCE_INDEX_NAME: &str = "sequence_index.json";
/// `shards.REMAP_NAME`: `remap.npz` + `remap.json`.
pub const REMAP_NAME: &str = "remap";
/// `artifacts.SHARD_FORMAT`.
pub const SHARD_FORMAT: &str = "qd-shard-v1";
/// `artifacts.REMAP_FORMAT`.
pub const REMAP_FORMAT: &str = "qd-remap-v1";
/// `artifacts.TOKEN_DTYPE`.
pub const TOKEN_DTYPE: &str = "uint32";
/// `shards.SEQUENCE_INDEX_FORMAT`.
pub const SEQUENCE_INDEX_FORMAT: &str = "qd-sequence-index/1";
/// `shards.GOLD_ROLE`.
pub const GOLD_ROLE: &str = "gold";
/// `qd_data.general.REPLAY_ONLY`.
pub const REPLAY_ONLY: &str = "replay_only";
/// `shards.PAD_ID`.
pub const PAD_ID: i32 = 0;
/// `shards.MAX_ROWS_PER_BATCH`.
pub const MAX_ROWS_PER_BATCH: usize = 4096;
/// `shards.MAX_POSITIONS_PER_BATCH`.
pub const MAX_POSITIONS_PER_BATCH: u64 = 1 << 20;
/// `artifacts.MAX_PADDING_WASTE`. Read-only under rule 2.
pub const MAX_PADDING_WASTE: f64 = 0.15;
/// `artifacts.SLOT_LM`: the CPT kind, refused inside an FT batch.
pub const SLOT_LM: u8 = 0;
/// `artifacts.SLOT_CHOICE`.
pub const SLOT_CHOICE: u8 = 1;
/// `artifacts.SLOT_SCORE`.
pub const SLOT_SCORE: u8 = 2;
/// `artifacts.SLOT_SPAN`.
pub const SLOT_SPAN: u8 = 3;
/// `artifacts.NO_SPAN`: `span_target` for a row that is not a span slot.
pub const NO_SPAN: i32 = -1;
/// `artifacts.SPAN_ABSTAIN`: `span_target` for a span row whose gold is abstain.
pub const SPAN_ABSTAIN: i32 = -2;
/// `artifacts._MIN_ROW_TOKENS`.
pub const MIN_ROW_TOKENS: u64 = 2;
/// `run_control.ConsumedPrefix.DOMAIN`.
pub const CONSUMED_PREFIX_DOMAIN: &[u8] = b"qd-consumed-prefix-v1";
/// The fourth entropy word of the batch-interleaving shuffle in `ShardReader._plan`.
const ORDER_TAG: u64 = 0xB17C;
/// `artifacts.SPAN_COLLAPSE_POLICIES` plus the empty "written before the field" value.
const SPAN_COLLAPSE_VALUES: [&str; 3] = ["", "refuse-any", "refuse-gold"];

const MAX_SMALL_JSON_BYTES: u64 = 4 << 20;
const MAX_SUPERVISION_BYTES: u64 = 8 << 30;
const MAX_SEQUENCE_INDEX_BYTES: u64 = 4 << 30;
const MAX_REMAP_BYTES: u64 = 256 << 20;
/// Ceiling on `qd_data` source files fingerprinted.
const MAX_FINGERPRINT_FILES: usize = 4096;

/// Everything that can stop the shard reader.
#[derive(Debug, Error)]
pub enum ShardError {
    /// The door refused (rule 3), or the manifest could not be verified.
    #[error(transparent)]
    Door(#[from] DoorError),
    /// The shard set violates the format (`ShardContractViolation`).
    #[error("{path}: {detail}")]
    Contract {
        /// The file or directory at fault.
        path: PathBuf,
        /// What is wrong.
        detail: String,
    },
    /// A file could not be read.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The error.
        source: io::Error,
    },
    /// An epoch plan was asked for with arguments `_plan` refuses (`ValueError`).
    #[error("{0}")]
    Plan(String),
}

impl ShardError {
    /// The rule-3 refusal, if that is what this is.
    pub fn held_out(&self) -> Option<&HeldOutViolation> {
        match self {
            ShardError::Door(DoorError::HeldOut(v)) => Some(v),
            _ => None,
        }
    }
}

fn contract(path: &Path, detail: impl Into<String>) -> ShardError {
    ShardError::Contract {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

fn sha256_parts(parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
        h.update([0u8]);
    }
    hex(&h.finalize())
}

// --- the header ---------------------------------------------------------------------------------

/// `artifacts.ShardHeader`: what a shard set is and what it was built from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardHeader {
    /// `train`, `val` or `heldout`.
    pub split: String,
    /// The manifest this set was built from (`Manifest.snapshot_hash()`).
    pub data_snapshot_hash: String,
    /// The tokenizer the remap renumbers.
    pub tokenizer_hash: String,
    /// The remap the ids were written under.
    pub remap_hash: String,
    /// Rows of the remapped vocabulary.
    pub vocab_size: u64,
    /// Sequences in the set.
    pub n_sequences: u64,
    /// Tokens in `tokens.u32`.
    pub total_tokens: u64,
    /// The longest sequence.
    pub max_seq_len: u64,
    /// Bucket widths, strictly increasing.
    pub buckets: Vec<u64>,
    /// `{module: sha256}` over the `qd_data` that wrote the set; empty for old sets.
    pub code_fingerprint: BTreeMap<String, String>,
    /// The revision the corpus was read at; empty for old sets.
    pub corpus_rev: String,
    /// sha256 of `sequence_index.json`; empty for old sets.
    pub sequence_index_hash: String,
    /// `""`, `refuse-any` or `refuse-gold`.
    pub span_collapse_policy: String,
    /// A report-only val set.
    pub report_only: bool,
    /// sha256 of the `exclusions.txt` whose identity keys left this train set
    /// (`qd_train.exclusions`); empty when none was applied, which is every set written before
    /// the field existed.
    pub exclusions_sha256: String,
    /// The v5 contrast rows this train set carries (`qd_train.contrast`); `None` for every set
    /// written before the field existed.
    pub contrast_rows: Option<ContrastRows>,
    /// The prompt layout the sequences were rendered in (`qd_data.render.PROMPT_FORMAT`); 1 for
    /// every set written before the field existed, which is format 1.
    pub prompt_format: u64,
}

/// `artifacts.ContrastRows`: how many contrast rows a train set carries, a sha256 over their row
/// ids and content hashes in write order, and the draw order's seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContrastRows {
    /// Rows added; positive.
    pub count: u64,
    /// Lower-case sha256.
    pub sha256: String,
    /// The draw order's seed.
    pub seed: u64,
}

impl ContrastRows {
    /// `ContrastRows.from_json` + `__post_init__`: exactly `{count, sha256, seed}`, a positive
    /// integer count, a lower-case sha256 and a non-negative integer seed. A JSON boolean is not
    /// an integer here, as it is not in Python's check.
    fn from_json(raw: &Value) -> Result<Self, String> {
        let obj = raw
            .as_object()
            .filter(|o| o.len() == 3 && ["count", "sha256", "seed"].iter().all(|k| o.contains_key(*k)))
            .ok_or(format!(
                "contrast_rows is {raw}; a header states it as {{count, sha256, seed}}"
            ))?;
        let count = obj["count"].as_u64().filter(|n| *n > 0).ok_or(format!(
            "contrast_rows.count {} is not a positive int: a set that carries no contrast row \
             names none",
            obj["count"]
        ))?;
        let sha256 = obj["sha256"]
            .as_str()
            .filter(|s| is_lower_hex64(s))
            .ok_or(format!(
                "contrast_rows.sha256 {} is not a lower-case sha256",
                obj["sha256"]
            ))?
            .to_owned();
        let seed = obj["seed"].as_u64().ok_or(format!(
            "contrast_rows.seed {} is not a non-negative int",
            obj["seed"]
        ))?;
        Ok(Self { count, sha256, seed })
    }

    /// `ContrastRows.to_json()`.
    fn to_json(&self) -> Value {
        json!({"count": self.count, "sha256": self.sha256, "seed": self.seed})
    }
}

fn is_lower_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl ShardHeader {
    /// `ShardHeader.from_json` + `__post_init__`, and the `shard_hash` comparison. A header
    /// without `shard_hash` is refused (Python skips the comparison when the key is absent).
    pub fn from_json(raw: &Value) -> Result<Self, String> {
        let obj = raw.as_object().ok_or("header must be a JSON object")?;
        if let Some(v) = obj.get("report_only")
            && !v.is_boolean()
        {
            return Err(format!(
                "report_only is {v}; a header states it as a JSON boolean"
            ));
        }
        let packed = match obj.get("packed") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => false,
            Some(Value::Number(n)) => n.as_f64() != Some(0.0),
            Some(Value::String(s)) => !s.is_empty(),
            Some(_) => true,
        };
        if packed {
            return Err(
                "packed=True is refused. docs/plan-corrections.md SAFETY-2: packing is \
                        unsupported for the hybrid -- a GDN layer carries recurrent state along \
                        the sequence, so two examples in one row bleed state across the boundary \
                        and no attention mask expresses a reset."
                    .to_owned(),
            );
        }
        let format = opt_str(obj, "format")?.unwrap_or_else(|| SHARD_FORMAT.to_owned());
        if format != SHARD_FORMAT {
            return Err(format!("format {format:?}, expected {SHARD_FORMAT:?}"));
        }
        let dtype = opt_str(obj, "dtype")?.unwrap_or_else(|| TOKEN_DTYPE.to_owned());
        if dtype != TOKEN_DTYPE {
            return Err(format!(
                "dtype {dtype:?}: token ids are stored as {TOKEN_DTYPE}. uint16 cannot hold a \
                 pre-remap id from a 248,320-token vocabulary"
            ));
        }
        let mut code_fingerprint = BTreeMap::new();
        match obj.get("code_fingerprint") {
            None | Some(Value::Null) => {}
            Some(Value::Object(m)) => {
                for (k, v) in m {
                    let v = v
                        .as_str()
                        .ok_or(format!("code_fingerprint[{k:?}] is not a string"))?;
                    code_fingerprint.insert(k.clone(), v.to_owned());
                }
            }
            Some(other) => return Err(format!("code_fingerprint is {other}, not an object")),
        }
        let buckets = match obj.get("buckets") {
            Some(Value::Array(items)) => items
                .iter()
                .map(|b| {
                    b.as_u64()
                        .ok_or(format!("bucket {b} is not a non-negative integer"))
                })
                .collect::<Result<Vec<u64>, String>>()?,
            _ => return Err("buckets is missing or not a list".to_owned()),
        };
        let header = Self {
            split: req_str(obj, "split")?,
            data_snapshot_hash: req_str(obj, "data_snapshot_hash")?,
            tokenizer_hash: req_str(obj, "tokenizer_hash")?,
            remap_hash: req_str(obj, "remap_hash")?,
            vocab_size: req_u64(obj, "vocab_size")?,
            n_sequences: req_u64(obj, "n_sequences")?,
            total_tokens: req_u64(obj, "total_tokens")?,
            max_seq_len: req_u64(obj, "max_seq_len")?,
            buckets,
            code_fingerprint,
            corpus_rev: opt_str(obj, "corpus_rev")?.unwrap_or_default(),
            sequence_index_hash: opt_str(obj, "sequence_index_hash")?.unwrap_or_default(),
            span_collapse_policy: opt_str(obj, "span_collapse_policy")?.unwrap_or_default(),
            report_only: obj.get("report_only") == Some(&Value::Bool(true)),
            exclusions_sha256: opt_str(obj, "exclusions_sha256")?.unwrap_or_default(),
            contrast_rows: obj
                .get("contrast_rows")
                .map(ContrastRows::from_json)
                .transpose()?,
            // Absent is format 1. Not coerced: a string, a float or a boolean does not state a
            // format, exactly as `ShardHeader.from_json` refuses them.
            prompt_format: match obj.get("prompt_format") {
                None => 1,
                Some(v) => v.as_u64().filter(|n| *n >= 1).ok_or(format!(
                    "prompt_format is {v}; a header states it as a positive JSON integer"
                ))?,
            },
        };
        header.validate()?;
        let recorded = obj.get("shard_hash").and_then(Value::as_str).ok_or(
            "header.json carries no shard_hash, so nothing says it is the header that was written",
        )?;
        let derived = header.shard_hash().map_err(|e| e.to_string())?;
        if recorded != derived {
            return Err(format!(
                "shard_hash on disk is {recorded:?} but the header hashes to {derived:?}; the \
                 artifact was modified after it was written"
            ));
        }
        Ok(header)
    }

    /// `ShardHeader.__post_init__`.
    fn validate(&self) -> Result<(), String> {
        if !SPLITS.contains(&self.split.as_str()) {
            return Err(format!("split {:?} not one of {SPLITS:?}", self.split));
        }
        for (name, v) in [
            ("data_snapshot_hash", &self.data_snapshot_hash),
            ("tokenizer_hash", &self.tokenizer_hash),
            ("remap_hash", &self.remap_hash),
        ] {
            if v.is_empty() {
                return Err(format!(
                    "{name} is required. Without it the trainer cannot tell whether these shards \
                     were built from the data, tokenizer and remap it is about to use"
                ));
            }
        }
        for (name, v) in [
            ("vocab_size", self.vocab_size),
            ("n_sequences", self.n_sequences),
            ("total_tokens", self.total_tokens),
            ("max_seq_len", self.max_seq_len),
        ] {
            if v == 0 {
                return Err(format!("{name} must be positive, got 0"));
            }
        }
        if self.buckets.is_empty() {
            return Err("at least one bucket boundary is required".to_owned());
        }
        if self.buckets.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!(
                "buckets must be strictly increasing and unique, got {:?}",
                self.buckets
            ));
        }
        if self.buckets[self.buckets.len() - 1] < self.max_seq_len {
            return Err(format!(
                "largest bucket {} is below max_seq_len {}: a sequence that fits no bucket would \
                 be dropped or truncated silently",
                self.buckets[self.buckets.len() - 1],
                self.max_seq_len
            ));
        }
        if self.total_tokens < self.n_sequences {
            return Err(format!(
                "total_tokens {} < n_sequences {}: at least one sequence is empty",
                self.total_tokens, self.n_sequences
            ));
        }
        if !SPAN_COLLAPSE_VALUES.contains(&self.span_collapse_policy.as_str()) {
            return Err(format!(
                "span_collapse_policy {:?} is not refuse-any or refuse-gold",
                self.span_collapse_policy
            ));
        }
        if self.report_only && self.split != "val" {
            return Err(format!(
                "report_only on split {:?}: only a val set may be report-only",
                self.split
            ));
        }
        if !self.exclusions_sha256.is_empty() && !is_lower_hex64(&self.exclusions_sha256) {
            return Err(format!(
                "exclusions_sha256 {:?} is not a lower-case sha256",
                self.exclusions_sha256
            ));
        }
        if !self.exclusions_sha256.is_empty() && self.split != "train" {
            return Err(format!(
                "exclusions_sha256 on split {:?}: the exclusion removes train rows only, so only \
                 a train set (gold or replay) names it",
                self.split
            ));
        }
        if self.contrast_rows.is_some() && self.split != "train" {
            return Err(format!(
                "contrast_rows on split {:?}: contrast rows are train rows only",
                self.split
            ));
        }
        if self.prompt_format == 0 {
            return Err("prompt_format 0 is not a positive int".to_owned());
        }
        Ok(())
    }

    /// `ShardHeader.shard_hash()`: sha256 over the format, the four identities, the
    /// dimensions as `json.dumps(sort_keys=True)`, and each optional pin that is set.
    pub fn shard_hash(&self) -> Result<String, pyjson::PyJsonError> {
        let dims = json!({
            "vocab_size": self.vocab_size,
            "n_sequences": self.n_sequences,
            "total_tokens": self.total_tokens,
            "max_seq_len": self.max_seq_len,
            "buckets": self.buckets,
        });
        let dims_text = pyjson::dumps(&dims, SORTED_DEFAULT)?;
        let mut parts: Vec<Vec<u8>> = vec![
            SHARD_FORMAT.as_bytes().to_vec(),
            self.split.as_bytes().to_vec(),
            self.data_snapshot_hash.as_bytes().to_vec(),
            self.tokenizer_hash.as_bytes().to_vec(),
            self.remap_hash.as_bytes().to_vec(),
            dims_text.into_bytes(),
        ];
        if !self.code_fingerprint.is_empty() {
            let fp: Map<String, Value> = self
                .code_fingerprint
                .iter()
                .map(|(k, v)| (k.clone(), Value::from(v.clone())))
                .collect();
            parts.push(pyjson::dumps(&Value::Object(fp), SORTED_DEFAULT)?.into_bytes());
        }
        if !self.corpus_rev.is_empty() {
            parts.push(self.corpus_rev.as_bytes().to_vec());
        }
        if !self.sequence_index_hash.is_empty() {
            parts.push(format!("sequence_index_hash:{}", self.sequence_index_hash).into_bytes());
        }
        if !matches!(self.span_collapse_policy.as_str(), "" | "refuse-any") {
            parts.push(format!("span_collapse_policy:{}", self.span_collapse_policy).into_bytes());
        }
        if self.report_only {
            parts.push(b"report_only:true".to_vec());
        }
        // The v5 pins, in `ShardHeader.shard_hash`'s order, each tagged and each contributing
        // nothing when absent (format 1, for `prompt_format`), so every header written before
        // them still verifies.
        if !self.exclusions_sha256.is_empty() {
            parts.push(format!("exclusions_sha256:{}", self.exclusions_sha256).into_bytes());
        }
        if let Some(rows) = &self.contrast_rows {
            let text = pyjson::dumps(&rows.to_json(), SORTED_DEFAULT)?;
            parts.push(format!("contrast_rows:{text}").into_bytes());
        }
        if self.prompt_format != 1 {
            parts.push(format!("prompt_format:{}", self.prompt_format).into_bytes());
        }
        let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
        Ok(sha256_parts(&refs))
    }

    /// `ShardHeader.require_gate_population`: refuse a set that is not a gate population.
    pub fn require_gate_population(&self, at: &str) -> Result<(), String> {
        if self.report_only || !matches!(self.span_collapse_policy.as_str(), "" | "refuse-any") {
            return Err(format!(
                "{at}: this shard set is not a gate population (report_only={}, \
                 span_collapse_policy={}). A gate scores the population it was measured on.",
                self.report_only,
                if self.span_collapse_policy.is_empty() {
                    "refuse-any"
                } else {
                    &self.span_collapse_policy
                }
            ));
        }
        Ok(())
    }
}

fn req_str(obj: &Map<String, Value>, key: &str) -> Result<String, String> {
    obj.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or(format!("{key} is missing or not a string"))
}

fn opt_str(obj: &Map<String, Value>, key: &str) -> Result<Option<String>, String> {
    match obj.get(key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(format!("{key} is {other}, not a string")),
    }
}

fn req_u64(obj: &Map<String, Value>, key: &str) -> Result<u64, String> {
    obj.get(key)
        .and_then(Value::as_u64)
        .ok_or(format!("{key} is missing or not a non-negative integer"))
}

// --- rule 3 at the shard boundary, and the provenance checks -----------------------------------

/// `qd_data.fingerprint.code_fingerprint(directory)`: `{name: sha256(bytes)}` over `*.py`.
pub fn code_fingerprint(directory: &Path) -> Result<BTreeMap<String, String>, ShardError> {
    let io_err = |source| ShardError::Io {
        path: directory.to_path_buf(),
        source,
    };
    let mut names: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(io_err)? {
        let path = entry.map_err(io_err)?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("py") && path.is_file() {
            names.push(path);
            if names.len() > MAX_FINGERPRINT_FILES {
                return Err(contract(
                    directory,
                    "more source files than the fingerprint bound",
                ));
            }
        }
    }
    names.sort();
    let mut out = BTreeMap::new();
    for path in names {
        let bytes = std::fs::read(&path).map_err(|source| ShardError::Io {
            path: path.clone(),
            source,
        })?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.insert(name, hex(&Sha256::digest(&bytes)));
    }
    if out.is_empty() {
        return Err(contract(
            directory,
            "no .py sources found; refusing an empty fingerprint, which would be \
             indistinguishable from a shard set created before fingerprints existed",
        ));
    }
    Ok(out)
}

/// `qd_data.fingerprint.describe_drift`: empty when nothing drifted.
pub fn describe_drift(
    recorded: &BTreeMap<String, String>,
    current: &BTreeMap<String, String>,
) -> String {
    let changed: Vec<&str> = recorded
        .iter()
        .filter(|(k, v)| current.get(*k).is_some_and(|c| c != *v))
        .map(|(k, _)| k.as_str())
        .collect();
    let added: Vec<&str> = current
        .keys()
        .filter(|k| !recorded.contains_key(*k))
        .map(String::as_str)
        .collect();
    let removed: Vec<&str> = recorded
        .keys()
        .filter(|k| !current.contains_key(*k))
        .map(String::as_str)
        .collect();
    let mut parts = Vec::new();
    if !changed.is_empty() {
        parts.push(format!("changed: {}", changed.join(", ")));
    }
    if !added.is_empty() {
        parts.push(format!("added since: {}", added.join(", ")));
    }
    if !removed.is_empty() {
        parts.push(format!("removed since: {}", removed.join(", ")));
    }
    parts.join("; ")
}

/// How [`ShardReader::open`] is to open a shard set.
#[derive(Clone)]
pub struct ShardOpen {
    /// The data config whose holdout the door enforces.
    pub config: DataConfig,
    /// Anchor for relative held-out roots: the run's out directory (`real_ft_run --out`).
    pub repo_root: PathBuf,
    /// The `qd_data` sources the header's `code_fingerprint` is compared with
    /// (`python/qd_data` in this repository). Required: an optional source would let a caller
    /// omit it and skip the check.
    pub qd_data_dir: PathBuf,
    /// The revision labels are reconstructed at; `None` records the rev check as not run.
    pub expect_rev: Option<String>,
    /// Read a set whose `corpus_rev` differs from `expect_rev`, deliberately.
    pub allow_rev_mismatch: bool,
    /// Read a set whose `qd_data` fingerprint has drifted, deliberately.
    pub allow_stale_code: bool,
    /// Admit a manifest whose snapshot status is `not_run`, deliberately.
    pub allow_not_run_snapshot: bool,
    /// Where file contents come from.
    pub files: Arc<dyn ShardFiles>,
}

impl ShardOpen {
    /// Defaults: no flags, no expected revision, the real filesystem.
    pub fn new(config: DataConfig, repo_root: PathBuf, qd_data_dir: PathBuf) -> Self {
        Self {
            config,
            repo_root,
            qd_data_dir,
            expect_rev: None,
            allow_rev_mismatch: false,
            allow_stale_code: false,
            allow_not_run_snapshot: false,
            files: Arc::new(OsFiles),
        }
    }
}

/// `assert_shard_trainable` (minus the path check, which [`ShardReader::open`] runs first)
/// plus this reader's additions: no `report_only`, and the header bound to its manifest.
fn shard_trainable(
    header: &ShardHeader,
    root: &Path,
    manifest: &TrainingManifest,
    opts: &ShardOpen,
) -> Result<Vec<(String, TriState)>, ShardError> {
    let code_check = if header.code_fingerprint.is_empty() {
        TriState::not_run(format!(
            "{}: this shard set's header carries no code_fingerprint, so it was written before \
             the generating code was pinned",
            root.display()
        ))
    } else {
        let current = code_fingerprint(&opts.qd_data_dir)?;
        let drift = describe_drift(&header.code_fingerprint, &current);
        if drift.is_empty() {
            TriState::Ran {
                passed: true,
                value: Value::from(current.len()),
                coverage: None,
                detail: format!(
                    "{} qd_data module(s) at {} hash as they did when this shard set was written",
                    current.len(),
                    opts.qd_data_dir.display()
                ),
            }
        } else {
            let detail = format!(
                "{}: qd_data at {} has changed since this shard set was written -- {drift}. \
                 Regenerate the shard set, or pass allow_stale_code to read it deliberately.",
                root.display(),
                opts.qd_data_dir.display()
            );
            if !opts.allow_stale_code {
                return Err(contract(root, detail));
            }
            TriState::Ran {
                passed: false,
                value: Value::from(current.len()),
                coverage: None,
                detail,
            }
        }
    };
    let rev_check = match (&header.corpus_rev, opts.expect_rev.as_deref()) {
        (rev, _) if rev.is_empty() => TriState::not_run(format!(
            "{}: this shard set's header carries no corpus_rev",
            root.display()
        )),
        (rev, None | Some("")) => TriState::not_run(format!(
            "{}: the header pins corpus_rev={rev}, and this caller named no revision to check it \
             against",
            root.display()
        )),
        (rev, Some(want)) if rev == want => TriState::passed(
            rev.clone(),
            "built from the revision this run reads its labels at",
        ),
        (rev, Some(want)) => {
            let detail = format!(
                "{}: this shard set was built from {rev} and this run reads its labels from \
                 {want}. Regenerate the set at this revision, or read it deliberately with \
                 allow_rev_mismatch.",
                root.display()
            );
            if !opts.allow_rev_mismatch {
                return Err(contract(root, detail));
            }
            TriState::Ran {
                passed: false,
                value: Value::from(rev.clone()),
                coverage: None,
                detail,
            }
        }
    };
    if header.split == "heldout" {
        return Err(HeldOutViolation {
            layer: HeldOutLayer::ShardSplit,
            expected: "a shard set whose split is train or val".to_owned(),
            actual: header.split.clone(),
            detail: format!(
                "{}: this shard set's split is 'heldout'. Rule 3 -- held-out data is never read \
                 by a training process; reading these shards would bypass the manifest door by \
                 indirection.",
                root.display()
            ),
        }
        .into_shard());
    }
    if header.split != "train" && header.split != "val" {
        return Err(HeldOutViolation {
            layer: HeldOutLayer::ShardSplit,
            expected: "a shard set whose split is train or val".to_owned(),
            actual: header.split.clone(),
            detail: format!(
                "{}: split {:?} is not a trainable split",
                root.display(),
                header.split
            ),
        }
        .into_shard());
    }
    if header.report_only {
        return Err(contract(
            root,
            "this shard set is report_only: outside every gate population and read only by its \
             own scorer. A training reader refuses it.",
        ));
    }
    if header.split != manifest.split || header.data_snapshot_hash != manifest.data_snapshot_hash {
        return Err(contract(
            root,
            format!(
                "this shard set was built from a {} snapshot {} but the manifest the door \
                 admitted, {}, is a {} snapshot {}. A shard set is read only beside the \
                 manifest it was written from.",
                header.split,
                header.data_snapshot_hash,
                manifest.path.display(),
                manifest.split,
                manifest.data_snapshot_hash
            ),
        ));
    }
    let shard_hash = header
        .shard_hash()
        .map_err(|e| contract(root, e.to_string()))?;
    Ok(vec![
        ("shard_rev_matches".to_owned(), rev_check),
        (
            "shard_path_not_held_out".to_owned(),
            TriState::passed(root.display().to_string(), ""),
        ),
        (
            "shard_split_trainable".to_owned(),
            TriState::passed(header.split.clone(), ""),
        ),
        (
            "shard_provenance_pinned".to_owned(),
            TriState::passed(
                shard_hash,
                format!(
                    "data_snapshot_hash={} tokenizer_hash={} remap_hash={}",
                    header.data_snapshot_hash, header.tokenizer_hash, header.remap_hash
                ),
            ),
        ),
        (
            "shard_not_packed".to_owned(),
            TriState::passed(false, "SAFETY-2"),
        ),
        (
            "held_out_families_configured".to_owned(),
            TriState::passed(
                opts.config.held_out_families().len(),
                format!("{:?}", opts.config.held_out_families()),
            ),
        ),
        ("shard_code_current".to_owned(), code_check),
        (
            "shard_not_report_only".to_owned(),
            TriState::passed(false, ""),
        ),
        (
            "shard_bound_to_manifest".to_owned(),
            TriState::passed(
                manifest.data_snapshot_hash.clone(),
                format!(
                    "header pins the admitted manifest {}",
                    manifest.path.display()
                ),
            ),
        ),
    ])
}

impl HeldOutViolation {
    fn into_shard(self) -> ShardError {
        ShardError::Door(DoorError::HeldOut(self))
    }
}

// --- the sequence index ---------------------------------------------------------------------

/// One `(row_id, slot_name)` that produced no sequence (`shards.SlotExclusion`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotExclusion {
    /// The row.
    pub row_id: String,
    /// The slot.
    pub slot_name: String,
    /// `row` or `slot`.
    pub scope: String,
    /// The refusal's class name.
    pub refusal: String,
    /// Its message.
    pub detail: String,
}

/// The parsed `sequence_index.json` (`shards.SequenceIndex`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceIndex {
    /// Rows offered to the writer.
    pub rows_in: u64,
    /// `gold` or `replay_only`.
    pub role: String,
    /// `(row_id, slot_name)` per sequence, in write order.
    pub sequences: Vec<(String, String)>,
    /// The slot kind per sequence.
    pub slot_kinds: Vec<i64>,
    /// What was excluded and why.
    pub excluded: Vec<SlotExclusion>,
}

impl SequenceIndex {
    fn from_json(raw: &Value, at: &Path) -> Result<Self, ShardError> {
        let bad = |d: String| contract(at, format!("malformed sequence index: {d}"));
        let obj = raw
            .as_object()
            .ok_or_else(|| bad("not an object".to_owned()))?;
        if obj.get("format").and_then(Value::as_str) != Some(SEQUENCE_INDEX_FORMAT) {
            return Err(contract(
                at,
                format!(
                    "format {:?}, expected {SEQUENCE_INDEX_FORMAT:?}",
                    obj.get("format")
                ),
            ));
        }
        let field = |o: &Map<String, Value>, k: &str| -> Result<String, ShardError> {
            o.get(k)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| bad(k.to_owned()))
        };
        let mut sequences = Vec::new();
        let mut slot_kinds = Vec::new();
        for s in obj
            .get("sequences")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("sequences".into()))?
        {
            let s = s
                .as_object()
                .ok_or_else(|| bad("a sequence entry".into()))?;
            sequences.push((field(s, "row_id")?, field(s, "slot_name")?));
            slot_kinds.push(
                s.get("slot_kind")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| bad("slot_kind".into()))?,
            );
        }
        let mut excluded = Vec::new();
        for e in obj
            .get("excluded")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("excluded".into()))?
        {
            let e = e
                .as_object()
                .ok_or_else(|| bad("an excluded entry".into()))?;
            excluded.push(SlotExclusion {
                row_id: field(e, "row_id")?,
                slot_name: field(e, "slot_name")?,
                scope: field(e, "scope")?,
                refusal: field(e, "refusal")?,
                detail: field(e, "detail")?,
            });
        }
        let rows_in = obj
            .get("rows_in")
            .and_then(Value::as_u64)
            .ok_or_else(|| bad("rows_in".into()))?;
        let role = field(obj, "role")?;
        let bad_scope: Vec<&str> = excluded
            .iter()
            .map(|e| e.scope.as_str())
            .filter(|s| *s != "row" && *s != "slot")
            .collect();
        if !bad_scope.is_empty() {
            return Err(contract(
                at,
                format!("unknown exclusion scope(s) {bad_scope:?}"),
            ));
        }
        let mut keys: HashSet<(&str, &str)> = HashSet::new();
        for k in sequences
            .iter()
            .map(|(r, s)| (r.as_str(), s.as_str()))
            .chain(
                excluded
                    .iter()
                    .map(|e| (e.row_id.as_str(), e.slot_name.as_str())),
            )
        {
            if !keys.insert(k) {
                return Err(contract(
                    at,
                    format!(
                        "(row_id, slot_name) {k:?} appears more than once across the written and \
                         excluded lists, so a label paired by it would be ambiguous"
                    ),
                ));
            }
        }
        if role != GOLD_ROLE && role != REPLAY_ONLY {
            return Err(contract(
                at,
                format!("role {role:?} is not {GOLD_ROLE:?} or {REPLAY_ONLY:?}"),
            ));
        }
        Ok(Self {
            rows_in,
            role,
            sequences,
            slot_kinds,
            excluded,
        })
    }
}

// --- the reader -----------------------------------------------------------------------------

/// One batch's membership, before any token is read (`shards._Plan`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Bucket index.
    pub bucket: usize,
    /// Bucket width: every row is padded to it.
    pub width: usize,
    /// The sequences, in row order.
    pub rows: Vec<usize>,
}

/// A shard set a training process may read. Construct with [`ShardReader::open`] only.
pub struct ShardReader {
    root: PathBuf,
    header: ShardHeader,
    manifest_hash: String,
    checks: Vec<(String, TriState)>,
    offsets: Vec<u64>,
    lengths: Vec<u64>,
    tokens: Box<dyn RangedRead>,
    slot_kind: Vec<u8>,
    target_index: Vec<i32>,
    span_target: Vec<[i32; 2]>,
    candidate_offsets: Vec<u64>,
    candidate_positions: Vec<i32>,
    sequence_index: Option<SequenceIndex>,
    slot_coverage: TriState,
    coverage: TriState,
    span_check: TriState,
}

impl std::fmt::Debug for ShardReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardReader")
            .field("root", &self.root)
            .field("split", &self.header.split)
            .field("n_sequences", &self.header.n_sequences)
            .finish_non_exhaustive()
    }
}

impl ShardReader {
    /// Open the shard set at `root`, built from the manifest at `manifest`, through the door.
    ///
    /// The only constructor. See the module docs for the order of the checks; every Python
    /// `ShardReader.__init__` refusal is ported, and none of them reads a token byte.
    pub fn open(root: &Path, manifest: &Path, opts: &ShardOpen) -> Result<Self, ShardError> {
        // Rule 3, before any file in the set is opened.
        assert_path_not_held_out(root, &opts.config, &opts.repo_root)?;
        let files = opts.files.as_ref();
        let admitted = open_training_manifest(
            manifest,
            &opts.config,
            &opts.repo_root,
            opts.allow_not_run_snapshot,
            files,
        )?;

        let header_path = root.join(HEADER_NAME);
        let raw = read_json(files, &header_path, MAX_SMALL_JSON_BYTES)?;
        let header = ShardHeader::from_json(&raw).map_err(|d| contract(&header_path, d))?;
        let mut checks: Vec<(String, TriState)> = admitted
            .checks
            .iter()
            .map(|(k, v)| (format!("manifest_{k}"), v.clone()))
            .collect();
        checks.extend(shard_trainable(&header, root, &admitted, opts)?);

        let offsets_path = root.join(OFFSETS_NAME);
        let max_offsets = header
            .n_sequences
            .saturating_add(1)
            .saturating_mul(8)
            .saturating_add(1 << 16);
        let offsets_raw = NpyArray::parse(&read_file(files, &offsets_path, max_offsets)?)
            .map_err(|e| contract(&offsets_path, e.to_string()))?;
        if offsets_raw.shape.len() != 1 {
            return Err(contract(
                &offsets_path,
                format!(
                    "offsets must be 1-D int64, got {}-D",
                    offsets_raw.shape.len()
                ),
            ));
        }
        let offsets_i = offsets_raw
            .exact_i64()
            .map_err(|e| contract(&offsets_path, format!("offsets must be 1-D int64: {e}")))?;
        let n = usize::try_from(header.n_sequences)
            .map_err(|_| contract(root, "n_sequences exceeds usize"))?;
        if offsets_i.len() != n + 1 {
            return Err(contract(
                &offsets_path,
                format!(
                    "{} offsets describe {} sequences but the header declares {n}",
                    offsets_i.len(),
                    offsets_i.len().saturating_sub(1)
                ),
            ));
        }
        if offsets_i[0] != 0 {
            return Err(contract(&offsets_path, "offsets must start at 0"));
        }
        if u64::try_from(offsets_i[n]).ok() != Some(header.total_tokens) {
            return Err(contract(
                &offsets_path,
                format!(
                    "offsets end at {} but the header declares {} tokens",
                    offsets_i[n], header.total_tokens
                ),
            ));
        }
        let mut offsets = Vec::with_capacity(n + 1);
        let mut lengths = Vec::with_capacity(n);
        for (i, &o) in offsets_i.iter().enumerate() {
            let o = u64::try_from(o).map_err(|_| contract(&offsets_path, "a negative offset"))?;
            if i > 0 {
                let prev = offsets[i - 1];
                if o <= prev {
                    return Err(contract(
                        &offsets_path,
                        "at least one sequence is empty or the offsets are not increasing",
                    ));
                }
                lengths.push(o - prev);
            }
            offsets.push(o);
        }
        let longest = lengths.iter().copied().max().unwrap_or(0);
        if !lengths.is_empty() && longest != header.max_seq_len {
            return Err(contract(
                &offsets_path,
                format!(
                    "the longest sequence is {longest} tokens but the header declares max_seq_len={}",
                    header.max_seq_len
                ),
            ));
        }

        // Measured, never read: the token bytes are read only after the reader exists.
        let tokens_path = root.join(TOKENS_NAME);
        let expected = header
            .total_tokens
            .checked_mul(4)
            .ok_or_else(|| contract(&tokens_path, "total_tokens overflows"))?;
        let actual = files.len(&tokens_path).map_err(|source| ShardError::Io {
            path: tokens_path.clone(),
            source,
        })?;
        if actual != expected {
            return Err(contract(
                &tokens_path,
                format!(
                    "{actual} bytes on disk, but the header describes {} {TOKEN_DTYPE} tokens ({expected} bytes). \
                     The shard set is truncated or was written by a different format.",
                    header.total_tokens
                ),
            ));
        }

        let sup = load_supervision(files, root, n, &lengths)?;
        let (sequence_index, slot_coverage) =
            load_sequence_index(files, root, &header, &sup.slot_kind, &admitted)?;
        let coverage = side_tristate(
            files,
            &root.join(COVERAGE_NAME),
            "how much of the manifest reached this shard set was not recorded. Absent coverage is unknown coverage, not full coverage.",
        )?;
        let span_check = side_tristate(
            files,
            &root.join(SPAN_CHECK_NAME),
            "whether this set's span mapping was ever checked against decoded text was not recorded",
        )?;
        checks.push((
            "shard_remap_matches_header".to_owned(),
            load_remap(files, root, &header)?,
        ));

        let tokens = files
            .open_ranged(&tokens_path)
            .map_err(|source| ShardError::Io {
                path: tokens_path.clone(),
                source,
            })?;
        Ok(Self {
            root: root.to_path_buf(),
            manifest_hash: admitted.data_snapshot_hash,
            header,
            checks,
            offsets,
            lengths,
            tokens,
            slot_kind: sup.slot_kind,
            target_index: sup.target_index,
            span_target: sup.span_target,
            candidate_offsets: sup.candidate_offsets,
            candidate_positions: sup.candidate_positions,
            sequence_index,
            slot_coverage,
            coverage,
            span_check,
        })
    }

    /// The shard directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The verified header.
    pub fn header(&self) -> &ShardHeader {
        &self.header
    }

    /// The admitted manifest's `data_snapshot_hash`.
    pub fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    /// Every check the door and the reader recorded, by name.
    pub fn checks(&self) -> &[(String, TriState)] {
        &self.checks
    }

    /// The check named `name`.
    pub fn check(&self, name: &str) -> Option<&TriState> {
        self.checks.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// Sequences in the set.
    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    /// Whether the set holds no sequences (never true for a valid header).
    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }

    /// Real tokens per sequence.
    pub fn lengths(&self) -> &[u64] {
        &self.lengths
    }

    /// Supervision kind of sequence `i`.
    pub fn slot_kind(&self, i: usize) -> u8 {
        self.slot_kind[i]
    }

    /// Position whose next token is sequence `i`'s gold letter.
    pub fn target_index(&self, i: usize) -> i32 {
        self.target_index[i]
    }

    /// Gold span `(start, end)` of sequence `i`, or `(NO_SPAN, NO_SPAN)`.
    pub fn span_target(&self, i: usize) -> [i32; 2] {
        self.span_target[i]
    }

    /// The parsed sequence index, when the header pins one.
    pub fn sequence_index(&self) -> Option<&SequenceIndex> {
        self.sequence_index.as_ref()
    }

    /// Sequences written against `(row, slot)` pairs offered.
    pub fn slot_coverage(&self) -> &TriState {
        &self.slot_coverage
    }

    /// `coverage.json`, or `NotRun` when absent.
    pub fn coverage(&self) -> &TriState {
        &self.coverage
    }

    /// `span_check.json`, or `NotRun` when absent.
    pub fn span_check(&self) -> &TriState {
        &self.span_check
    }

    /// Sequence `i`'s line-start candidates (empty for a non-span sequence).
    pub fn candidates(&self, i: usize) -> Result<&[i32], ShardError> {
        if i >= self.len() {
            return Err(ShardError::Plan(format!(
                "sequence {i} out of range for {} sequences",
                self.len()
            )));
        }
        let lo = self.candidate_offsets[i] as usize;
        let hi = self.candidate_offsets[i + 1] as usize;
        Ok(&self.candidate_positions[lo..hi])
    }

    /// Sequence `i` as post-remap ids, read from `tokens.u32` by position. An id at or above
    /// `vocab_size` is refused (it would fault the embedding lookup).
    pub fn sequence(&self, i: usize) -> Result<Vec<i32>, ShardError> {
        if i >= self.len() {
            return Err(ShardError::Plan(format!(
                "sequence {i} out of range for {} sequences",
                self.len()
            )));
        }
        let (start, end) = (self.offsets[i], self.offsets[i + 1]);
        let n = usize::try_from(end - start)
            .map_err(|_| contract(&self.root, "sequence length exceeds usize"))?;
        let mut buf = vec![0u8; n * 4];
        let path = self.root.join(TOKENS_NAME);
        self.tokens
            .read_exact_at(start * 4, &mut buf)
            .map_err(|source| ShardError::Io {
                path: path.clone(),
                source,
            })?;
        buf.as_chunks::<4>()
            .0
            .iter()
            .map(|c| {
                let id = u32::from_le_bytes(*c);
                if u64::from(id) >= self.header.vocab_size {
                    return Err(contract(
                        &path,
                        format!(
                            "sequence {i} holds token id {id}, outside vocab_size {}",
                            self.header.vocab_size
                        ),
                    ));
                }
                i32::try_from(id)
                    .map_err(|_| contract(&path, format!("token id {id} exceeds int32")))
            })
            .collect()
    }

    /// `padding_waste` over one epoch: `(padded - real, padded)` and the S4 gate's verdict.
    pub fn padding_waste(&self) -> Result<PaddingWaste, ShardError> {
        padding_waste(&self.lengths, &self.header.buckets)
    }

    /// `ShardReader._plan`: every batch of the epoch as membership only. Pure in its arguments.
    pub fn plan(&self, batch_tokens: u64, seed: u64, epoch: u64) -> Result<Vec<Plan>, ShardError> {
        plan_epoch(
            &self.lengths,
            &self.header.buckets,
            batch_tokens,
            seed,
            epoch,
        )
    }

    /// `assemble_batch` for one plan entry at epoch position `index`.
    pub fn batch(&self, plan: &Plan, index: u64) -> Result<Batch, ShardError> {
        let mut sequences = Vec::with_capacity(plan.rows.len());
        let mut candidates = Vec::with_capacity(plan.rows.len());
        for &i in &plan.rows {
            sequences.push(self.sequence(i)?);
            candidates.push(self.candidates(i)?.to_vec());
        }
        let kinds: Vec<u8> = plan.rows.iter().map(|&i| self.slot_kind[i]).collect();
        let target: Vec<i32> = plan.rows.iter().map(|&i| self.target_index[i]).collect();
        let spans: Vec<[i32; 2]> = plan.rows.iter().map(|&i| self.span_target[i]).collect();
        assemble_batch(
            &sequences,
            &plan.rows,
            kinds,
            target,
            spans,
            &candidates,
            plan.width,
            plan.bucket,
            index,
        )
        .map_err(|d| contract(&self.root, format!("batch {index}: {d}")))
    }

    /// `ShardReader.batches(batch_tokens, seed, epoch)`: one epoch, same order every time.
    pub fn batches(
        &self,
        batch_tokens: u64,
        seed: u64,
        epoch: u64,
    ) -> Result<Batches<'_>, ShardError> {
        Ok(Batches {
            reader: self,
            plans: self.plan(batch_tokens, seed, epoch)?,
            next: 0,
        })
    }
}

/// The epoch iterator [`ShardReader::batches`] returns.
pub struct Batches<'a> {
    reader: &'a ShardReader,
    plans: Vec<Plan>,
    next: usize,
}

impl Batches<'_> {
    /// Batches in the epoch.
    pub fn n_batches(&self) -> usize {
        self.plans.len()
    }
}

impl Iterator for Batches<'_> {
    type Item = Result<Batch, ShardError>;

    fn next(&mut self) -> Option<Self::Item> {
        let plan = self.plans.get(self.next)?;
        let index = self.next as u64;
        self.next += 1;
        Some(self.reader.batch(plan, index))
    }
}

fn read_file(files: &dyn ShardFiles, path: &Path, max: u64) -> Result<Vec<u8>, ShardError> {
    files.read(path, max).map_err(|source| ShardError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_json(files: &dyn ShardFiles, path: &Path, max: u64) -> Result<Value, ShardError> {
    let bytes = read_file(files, path, max)?;
    serde_json::from_slice(&bytes).map_err(|e| contract(path, format!("not JSON: {e}")))
}

fn side_tristate(
    files: &dyn ShardFiles,
    path: &Path,
    absent: &str,
) -> Result<TriState, ShardError> {
    if !files.is_file(path) {
        return Ok(TriState::not_run(format!(
            "{} is absent, so {absent}",
            path.display()
        )));
    }
    let raw = read_json(files, path, MAX_SMALL_JSON_BYTES)?;
    parse_tristate(&raw, &path.display().to_string()).map_err(|e| contract(path, e.to_string()))
}

struct Supervision {
    slot_kind: Vec<u8>,
    target_index: Vec<i32>,
    span_target: Vec<[i32; 2]>,
    candidate_offsets: Vec<u64>,
    candidate_positions: Vec<i32>,
}

/// `ShardReader._load_supervision`: every refusal it makes, plus a slot-kind check at open.
fn load_supervision(
    files: &dyn ShardFiles,
    root: &Path,
    n: usize,
    lengths: &[u64],
) -> Result<Supervision, ShardError> {
    let path = root.join(SUPERVISION_NAME);
    if !files.is_file(&path) {
        return Err(contract(
            &path,
            "absent. This shard set's sequences are prompt-plus-answer rows, so their supervision \
             is not derivable from the tokens alone; yielding CPT batches instead would silently \
             train a different objective.",
        ));
    }
    let zip = StoredZip::parse(&read_file(files, &path, MAX_SUPERVISION_BYTES)?)
        .map_err(|e| contract(&path, e.to_string()))?;
    let get = |key: &str| zip.array(key).map_err(|e| contract(&path, e.to_string()));
    let (kinds_a, target_a, spans_a, offs_a, pos_a) = (
        get("slot_kind")?,
        get("target_index")?,
        get("span_target")?,
        get("candidate_offsets")?,
        get("candidate_positions")?,
    );
    if kinds_a.shape != [n] || target_a.shape != [n] || spans_a.shape != [n, 2] {
        return Err(contract(
            &path,
            format!(
                "supervision arrays are {:?}, {:?}, {:?}; expected ({n},), ({n},) and ({n}, 2)",
                kinds_a.shape, target_a.shape, spans_a.shape
            ),
        ));
    }
    if offs_a.shape != [n + 1] {
        return Err(contract(
            &path,
            format!(
                "candidate_offsets is {:?}, expected ({},)",
                offs_a.shape,
                n + 1
            ),
        ));
    }
    if pos_a.shape.len() != 1 {
        return Err(contract(
            &path,
            format!("candidate_positions is {:?}, expected 1-D", pos_a.shape),
        ));
    }
    let cast_err = |e: crate::npy::NpyError| contract(&path, e.to_string());
    let slot_kind: Vec<u8> = kinds_a.cast("slot_kind").map_err(cast_err)?;
    let target_index: Vec<i32> = target_a.cast("target_index").map_err(cast_err)?;
    let flat_spans: Vec<i32> = spans_a.cast("span_target").map_err(cast_err)?;
    let span_target: Vec<[i32; 2]> = flat_spans.as_chunks::<2>().0.to_vec();
    let offs_i: Vec<i64> = offs_a.cast("candidate_offsets").map_err(cast_err)?;
    let candidate_positions: Vec<i32> = pos_a.cast("candidate_positions").map_err(cast_err)?;
    if offs_i[0] != 0 || usize::try_from(offs_i[n]).ok() != Some(candidate_positions.len()) {
        return Err(contract(
            &path,
            format!(
                "candidate_offsets run 0..{} but there are {} candidate positions",
                offs_i[n],
                candidate_positions.len()
            ),
        ));
    }
    if offs_i.windows(2).any(|w| w[1] < w[0]) {
        return Err(contract(
            &path,
            "candidate_offsets decrease somewhere: the ragged candidate lists overlap",
        ));
    }
    let candidate_offsets: Vec<u64> = offs_i.iter().map(|&o| o as u64).collect();

    for (i, &k) in slot_kind.iter().enumerate() {
        if k == SLOT_LM || k > SLOT_SPAN {
            return Err(contract(
                &path,
                format!(
                    "sequence {i} has slot kind {k}; an FT shard set carries {SLOT_CHOICE}, \
                     {SLOT_SCORE} or {SLOT_SPAN} (SLOT_LM is the CPT kind and has no gold letter)"
                ),
            ));
        }
    }
    for (i, (&t, &len)) in target_index.iter().zip(lengths).enumerate() {
        if t < 0 || (t as u64) + 1 >= len {
            return Err(contract(
                &path,
                format!(
                    "target_index[{i}]={t} is not in [0, {len}-1); the supervised position's next \
                     token is the gold, so the last position supervises nothing"
                ),
            ));
        }
    }
    for i in 0..n {
        let is_span = slot_kind[i] == SLOT_SPAN;
        let [s, e] = span_target[i];
        if !is_span && (s != NO_SPAN || e != NO_SPAN) {
            return Err(contract(
                &path,
                format!(
                    "a non-span sequence ({i}) carries a span position; non-span rows must be {NO_SPAN}"
                ),
            ));
        }
        if !is_span {
            continue;
        }
        if (s == SPAN_ABSTAIN) != (e == SPAN_ABSTAIN) {
            return Err(contract(
                &path,
                format!(
                    "span_target[{i}] abstains in one position only; a span row abstains in both or neither"
                ),
            ));
        }
        let pointing = s != SPAN_ABSTAIN;
        if pointing && e >= 0 && (e as u64) >= lengths[i] {
            return Err(contract(
                &path,
                format!(
                    "span_target[{i}] ends at {e} but that sequence holds {} tokens: the span points past it",
                    lengths[i]
                ),
            ));
        }
        let positions =
            &candidate_positions[candidate_offsets[i] as usize..candidate_offsets[i + 1] as usize];
        if positions.is_empty() {
            return Err(contract(
                &path,
                format!(
                    "sequence {i} is SLOT_SPAN but has no line-start candidates; an empty set is a mapping failure"
                ),
            ));
        }
        let (lo, hi) = (
            positions.iter().min().copied().unwrap_or(0),
            positions.iter().max().copied().unwrap_or(0),
        );
        if lo < 0 || (hi as u64) >= lengths[i] {
            return Err(contract(
                &path,
                format!(
                    "sequence {i} has a line-start candidate outside its {} real tokens; the pointer head could answer with padding",
                    lengths[i]
                ),
            ));
        }
        if pointing {
            for (name, pos) in [("start", s), ("end", e)] {
                if !positions.contains(&pos) {
                    return Err(contract(
                        &path,
                        format!(
                            "span_target[{i}] {name}={pos} is not one of that sequence's line-start candidates"
                        ),
                    ));
                }
            }
        }
    }
    Ok(Supervision {
        slot_kind,
        target_index,
        span_target,
        candidate_offsets,
        candidate_positions,
    })
}

/// `ShardReader._load_sequence_index`, plus every sequence's `row_id` checked against the
/// admitted manifest.
fn load_sequence_index(
    files: &dyn ShardFiles,
    root: &Path,
    header: &ShardHeader,
    slot_kind: &[u8],
    manifest: &TrainingManifest,
) -> Result<(Option<SequenceIndex>, TriState), ShardError> {
    let pinned = &header.sequence_index_hash;
    let path = root.join(SEQUENCE_INDEX_NAME);
    if pinned.is_empty() {
        return Ok((
            None,
            TriState::not_run(format!(
                "{} pins no sequence_index_hash, so which row and slot each sequence is was not recorded by the writer",
                root.join(HEADER_NAME).display()
            )),
        ));
    }
    if !files.is_file(&path) {
        return Err(contract(
            &path,
            format!("absent but the header pins sequence_index_hash={pinned}"),
        ));
    }
    let raw = read_file(files, &path, MAX_SEQUENCE_INDEX_BYTES)?;
    let found = hex(&Sha256::digest(&raw));
    if &found != pinned {
        return Err(contract(
            &path,
            format!(
                "hashes to {found:?} but the header pins {pinned:?}; modified after it was written, or belongs to another set"
            ),
        ));
    }
    let value: Value =
        serde_json::from_slice(&raw).map_err(|e| contract(&path, format!("not JSON: {e}")))?;
    drop(raw);
    let index = SequenceIndex::from_json(&value, &path)?;
    let n = slot_kind.len();
    if index.sequences.len() != n {
        return Err(contract(
            &path,
            format!(
                "{} indexed sequence(s) but the header declares {n}",
                index.sequences.len()
            ),
        ));
    }
    if let Some(i) = (0..n).find(|&i| index.slot_kinds[i] != i64::from(slot_kind[i])) {
        return Err(contract(
            &path,
            format!(
                "sequence {i} is indexed as slot kind {} but supervision.npz holds {}; the index and the shards describe different orders",
                index.slot_kinds[i], slot_kind[i]
            ),
        ));
    }
    if let Some((i, (row, _))) = index
        .sequences
        .iter()
        .enumerate()
        .find(|(_, (row, _))| !manifest.row_ids.contains(row))
    {
        return Err(contract(
            &path,
            format!(
                "sequence {i} is row {row:?}, which the admitted manifest {} does not contain. A shard \
                 sequence is trained only if the door admitted its row.",
                manifest.path.display()
            ),
        ));
    }
    let excluded = index.excluded.len() as u64;
    let n64 = n as u64;
    let slot = index.excluded.iter().filter(|e| e.scope == "slot").count();
    let row = index.excluded.iter().filter(|e| e.scope == "row").count();
    let coverage = TriState::Ran {
        passed: excluded == 0,
        value: Value::from(n64),
        coverage: Some(Coverage {
            n: n64,
            n_total: n64 + excluded,
        }),
        detail: format!(
            "{n64} of {} (row, slot) sequence(s) written; excluded {slot} slot(s) on their own account and {row} slot(s) of rows refused whole, over {} row(s) offered",
            n64 + excluded,
            index.rows_in
        ),
    };
    Ok((Some(index), coverage))
}

/// `ShardReader._load_remap` + `RemapTable.read`: `NotRun` when absent, refused when wrong.
fn load_remap(
    files: &dyn ShardFiles,
    root: &Path,
    header: &ShardHeader,
) -> Result<TriState, ShardError> {
    let npz = root.join(format!("{REMAP_NAME}.npz"));
    let meta_path = root.join(format!("{REMAP_NAME}.json"));
    if !(files.is_file(&npz) && files.is_file(&meta_path)) {
        return Ok(TriState::not_run(format!(
            "{}.npz/.json is absent, so the vocabulary this set's ids were renumbered under is not available and the header's remap_hash names no artifact here",
            root.join(REMAP_NAME).display()
        )));
    }
    let meta = read_json(files, &meta_path, MAX_SMALL_JSON_BYTES)?;
    if meta.get("format").and_then(Value::as_str) != Some(REMAP_FORMAT) {
        return Err(contract(
            &meta_path,
            format!(
                "format is {:?}, expected {REMAP_FORMAT:?}",
                meta.get("format")
            ),
        ));
    }
    let tokenizer_hash = meta
        .get("tokenizer_hash")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    if tokenizer_hash.is_empty() {
        return Err(contract(
            &meta_path,
            "tokenizer_hash is required: a remap is only meaningful against the tokenizer whose ids it renumbers",
        ));
    }
    let special: Vec<i64> = match meta.get("special_ids") {
        None => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_i64()
                    .ok_or_else(|| contract(&meta_path, "a special id is not an integer"))
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(contract(&meta_path, "special_ids is not a list")),
    };
    let zip = StoredZip::parse(&read_file(files, &npz, MAX_REMAP_BYTES)?)
        .map_err(|e| contract(&npz, e.to_string()))?;
    let table = |key: &str| -> Result<Vec<i32>, ShardError> {
        let a = zip.array(key).map_err(|e| contract(&npz, e.to_string()))?;
        if a.shape.len() != 1 {
            return Err(contract(&npz, "remap tables must be 1-D"));
        }
        a.cast::<i32>(key)
            .map_err(|e| contract(&npz, e.to_string()))
    };
    let old_to_new = table("old_to_new")?;
    let new_to_old = table("new_to_old")?;
    let kept: Vec<usize> = old_to_new
        .iter()
        .enumerate()
        .filter(|(_, v)| **v >= 0)
        .map(|(i, _)| i)
        .collect();
    if kept.len() != new_to_old.len() {
        return Err(contract(
            &npz,
            format!(
                "{} kept entries in old_to_new but new_to_old has {}",
                kept.len(),
                new_to_old.len()
            ),
        ));
    }
    if new_to_old.is_empty() {
        return Err(contract(
            &npz,
            "a remap that keeps no tokens is not a remap",
        ));
    }
    for (new, &old) in new_to_old.iter().enumerate() {
        let back = usize::try_from(old)
            .ok()
            .and_then(|o| old_to_new.get(o))
            .copied();
        if back != i32::try_from(new).ok() {
            return Err(contract(
                &npz,
                format!(
                    "tables are not inverse: new id {new} maps to old id {old}, which maps back to {back:?}"
                ),
            ));
        }
    }
    for &old in &kept {
        let new = old_to_new[old] as usize;
        if new_to_old.get(new).copied() != i32::try_from(old).ok() {
            return Err(contract(
                &npz,
                "tables are not inverse in the old -> new direction",
            ));
        }
    }
    let dropped: Vec<i64> = special
        .iter()
        .copied()
        .filter(|&t| {
            usize::try_from(t)
                .ok()
                .and_then(|t| old_to_new.get(t))
                .is_none_or(|v| *v < 0)
        })
        .collect();
    if !dropped.is_empty() {
        return Err(contract(
            &npz,
            format!("the remap dropped special token id(s) {dropped:?}"),
        ));
    }
    let mut new_to_old_bytes = Vec::with_capacity(new_to_old.len() * 4);
    for v in &new_to_old {
        new_to_old_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let source_vocab = old_to_new.len().to_string();
    let found = sha256_parts(&[
        REMAP_FORMAT.as_bytes(),
        tokenizer_hash.as_bytes(),
        source_vocab.as_bytes(),
        &new_to_old_bytes,
    ]);
    if meta.get("remap_hash").and_then(Value::as_str) != Some(found.as_str()) {
        return Err(contract(
            &meta_path,
            format!(
                "remap_hash on disk is {:?} but the tables hash to {found:?}. The artifact was modified after it was written.",
                meta.get("remap_hash")
            ),
        ));
    }
    if found != header.remap_hash {
        return Err(contract(
            &npz,
            format!(
                "the remap beside these shards hashes to {found:?} but the header pins {:?}. Every id in this set would mean a different token.",
                header.remap_hash
            ),
        ));
    }
    if new_to_old.len() as u64 != header.vocab_size {
        return Err(contract(
            &npz,
            format!(
                "the remap keeps {} tokens but the header declares vocab_size={}",
                new_to_old.len(),
                header.vocab_size
            ),
        ));
    }
    Ok(TriState::Ran {
        passed: true,
        value: Value::from(found),
        coverage: Some(Coverage {
            n: new_to_old.len() as u64,
            n_total: old_to_new.len() as u64,
        }),
        detail: format!(
            "{} re-read and re-hashed; matches the header. tokenizer_hash={tokenizer_hash}",
            npz.display()
        ),
    })
}

// --- buckets, padding, the epoch plan ---------------------------------------------------------

/// `artifacts.bucket_for`: the index of the smallest bucket that fits `length`.
pub fn bucket_for(length: u64, buckets: &[u64]) -> Result<usize, String> {
    if length == 0 {
        return Err("length must be positive, got 0".to_owned());
    }
    buckets.iter().position(|&b| length <= b).ok_or_else(|| {
        format!(
            "length {length} exceeds the largest bucket {:?}; a sequence that fits no bucket must be refused at write time, not truncated at read time",
            buckets.last()
        )
    })
}

/// `artifacts.padding_waste` over one epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct PaddingWaste {
    /// Real tokens.
    pub real: u64,
    /// Positions paid for: every sequence padded to its bucket's width.
    pub padded: u64,
    /// Sequences.
    pub n: u64,
}

impl PaddingWaste {
    /// `(padded - real) / padded`.
    pub fn waste(&self) -> f64 {
        (self.padded - self.real) as f64 / self.padded as f64
    }

    /// The S4 gate: `Ran(passed = waste <= MAX_PADDING_WASTE)`.
    pub fn tristate(&self) -> TriState {
        TriState::Ran {
            passed: self.waste() <= MAX_PADDING_WASTE,
            value: Value::from(self.waste()),
            coverage: Some(Coverage {
                n: self.n,
                n_total: self.n,
            }),
            detail: format!(
                "{} padding of {} positions over {} sequences",
                self.padded - self.real,
                self.padded,
                self.n
            ),
        }
    }
}

/// `artifacts.padding_waste`. Refused (not 0%) over zero sequences.
pub fn padding_waste(lengths: &[u64], buckets: &[u64]) -> Result<PaddingWaste, ShardError> {
    if lengths.is_empty() {
        return Err(ShardError::Plan(
            "no sequences: padding waste over an empty shard set is 0/0".to_owned(),
        ));
    }
    let mut real = 0u64;
    let mut padded = 0u64;
    for &n in lengths {
        real += n;
        padded += buckets[bucket_for(n, buckets).map_err(ShardError::Plan)?];
    }
    Ok(PaddingWaste {
        real,
        padded,
        n: lengths.len() as u64,
    })
}

/// `ShardReader._plan`: two independent shuffles per epoch -- one over each bucket's members,
/// one interleaving the resulting batches -- each from its own `SeedSequence`.
pub fn plan_epoch(
    lengths: &[u64],
    buckets: &[u64],
    batch_tokens: u64,
    seed: u64,
    epoch: u64,
) -> Result<Vec<Plan>, ShardError> {
    if batch_tokens == 0 {
        return Err(ShardError::Plan(
            "batch_tokens must be positive, got 0".to_owned(),
        ));
    }
    if batch_tokens > MAX_POSITIONS_PER_BATCH {
        return Err(ShardError::Plan(format!(
            "batch_tokens={batch_tokens} exceeds MAX_POSITIONS_PER_BATCH={MAX_POSITIONS_PER_BATCH}. A batch's memory cost is rows x width, and batch_tokens is the only ceiling on that product."
        )));
    }
    let mut members: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    for (i, &n) in lengths.iter().enumerate() {
        members
            .entry(bucket_for(n, buckets).map_err(ShardError::Plan)?)
            .or_default()
            .push(i as u64);
    }
    if let Some((&b, m)) = members.iter().find(|(b, _)| buckets[**b] > batch_tokens) {
        return Err(ShardError::Plan(format!(
            "batch_tokens={batch_tokens} is below bucket width {}, which holds {} sequence(s). A single sequence from that bucket already exceeds the budget, so no batch can honour it.",
            buckets[b],
            m.len()
        )));
    }
    let mut plans = Vec::new();
    for (&b, m) in &members {
        let width = buckets[b];
        let order = default_rng(&[seed, epoch, batch_tokens, b as u64]).permutation(m);
        let per_batch = usize::try_from(batch_tokens / width)
            .unwrap_or(usize::MAX)
            .min(MAX_ROWS_PER_BATCH);
        for chunk in order.chunks(per_batch) {
            plans.push(Plan {
                bucket: b,
                width: usize::try_from(width)
                    .map_err(|_| ShardError::Plan("bucket width exceeds usize".to_owned()))?,
                rows: chunk.iter().map(|&i| i as usize).collect(),
            });
        }
    }
    let shuffled = default_rng(&[seed, epoch, batch_tokens, ORDER_TAG])
        .permutation(&(0..plans.len()).collect::<Vec<_>>());
    Ok(shuffled.into_iter().map(|i| plans[i].clone()).collect())
}

// --- the batch ------------------------------------------------------------------------------

/// One padded FT batch (`artifacts.Batch` with the supervision channel always present).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    /// Position in the epoch.
    pub index: u64,
    /// Bucket index.
    pub bucket: usize,
    /// Padded width; every row is this wide.
    pub width: usize,
    /// Which shard sequences the rows are (not part of Python's `Batch`, not hashed).
    pub sequences: Vec<usize>,
    /// `int32[B, width]`, row-major, padded with [`PAD_ID`].
    pub tokens: Vec<i32>,
    /// Real tokens per row (`int64`).
    pub lengths: Vec<i64>,
    /// Supervision kind per row (`uint8`).
    pub slot_kind: Vec<u8>,
    /// Position whose next token is the gold letter (`int32`).
    pub target_index: Vec<i32>,
    /// `int32[B, 2]`, present exactly when some row is `SLOT_SPAN`.
    pub span_target: Option<Vec<[i32; 2]>>,
    /// `bool[B, width]`, present exactly when some row is `SLOT_SPAN`.
    pub line_starts: Option<Vec<bool>>,
}

impl Batch {
    /// Rows.
    pub fn rows(&self) -> usize {
        self.lengths.len()
    }

    /// `tokens[r, p]`.
    pub fn token(&self, r: usize, p: usize) -> i32 {
        self.tokens[r * self.width + p]
    }

    /// `line_starts[r, p]` (false when the batch has no span row).
    pub fn is_line_start(&self, r: usize, p: usize) -> bool {
        self.line_starts
            .as_ref()
            .is_some_and(|m| m[r * self.width + p])
    }

    /// `Batch.__post_init__`, `_check_supervision` and `_check_line_starts`, for an FT batch.
    pub fn validate(&self) -> Result<(), String> {
        let b = self.rows();
        if b == 0 {
            return Err("an empty batch is not a batch".to_owned());
        }
        if self.tokens.len() != b * self.width
            || self.slot_kind.len() != b
            || self.target_index.len() != b
        {
            return Err(
                "tokens, slot_kind and target_index must all describe the same rows".to_owned(),
            );
        }
        if self.width < 2 {
            return Err(format!(
                "a batch padded to width {} has no next-token pair",
                self.width
            ));
        }
        for (r, &n) in self.lengths.iter().enumerate() {
            if n < MIN_ROW_TOKENS as i64 {
                return Err(format!(
                    "row {r} carries {n} real token(s); every row needs at least {MIN_ROW_TOKENS}"
                ));
            }
            if n as usize > self.width {
                return Err(format!(
                    "length {n} exceeds the padded width {}: a row claims more tokens than it has",
                    self.width
                ));
            }
        }
        for (r, &k) in self.slot_kind.iter().enumerate() {
            if k > SLOT_SPAN {
                return Err(format!("unknown slot kind {k} in row {r}"));
            }
            if k == SLOT_LM {
                return Err(format!(
                    "row {r} carries SLOT_LM in a batch that carries the slot channel. GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED"
                ));
            }
            let t = self.target_index[r];
            if t < 0 || i64::from(t) >= self.lengths[r] - 1 {
                return Err(format!(
                    "target_index[{r}]={t} is not in [0, lengths[{r}]-1) with lengths[{r}]={}",
                    self.lengths[r]
                ));
            }
        }
        let is_span: Vec<bool> = self.slot_kind.iter().map(|&k| k == SLOT_SPAN).collect();
        if !is_span.iter().any(|&s| s) {
            if self.line_starts.is_some() {
                return Err("line_starts without any SLOT_SPAN row".to_owned());
            }
            if self
                .span_target
                .as_ref()
                .is_some_and(|s| s.iter().any(|p| *p != [NO_SPAN, NO_SPAN]))
            {
                return Err("span_target carries positions but no row is SLOT_SPAN".to_owned());
            }
            return Ok(());
        }
        let spans = self
            .span_target
            .as_ref()
            .ok_or("SLOT_SPAN rows but span_target is absent")?;
        if spans.len() != b {
            return Err(format!("span_target must be [{b}, 2]"));
        }
        let mask = self
            .line_starts
            .as_ref()
            .ok_or("SLOT_SPAN rows but line_starts is absent")?;
        if mask.len() != b * self.width {
            return Err(format!("line_starts must be [{b}, {}]", self.width));
        }
        for r in 0..b {
            let [s, e] = spans[r];
            if !is_span[r] {
                if [s, e] != [NO_SPAN, NO_SPAN] {
                    return Err(format!("a non-span row ({r}) carries a span position"));
                }
                continue;
            }
            if (s == SPAN_ABSTAIN) != (e == SPAN_ABSTAIN) {
                return Err(format!(
                    "span_target[{r}] is ({s}, {e}): a span row abstains in both positions or neither"
                ));
            }
            let row = &mask[r * self.width..(r + 1) * self.width];
            let n = self.lengths[r] as usize;
            if row[n..].iter().any(|&m| m) {
                return Err(format!(
                    "line_starts[{r}] marks a candidate at or past position {n}, which is padding"
                ));
            }
            if !row[..n].iter().any(|&m| m) {
                return Err(format!(
                    "line_starts[{r}] marks no candidate in {n} real token(s)"
                ));
            }
            if s == SPAN_ABSTAIN {
                continue;
            }
            if s < 0 || e < 0 {
                return Err(format!(
                    "a SLOT_SPAN row must carry real start/end positions, or {SPAN_ABSTAIN} in both"
                ));
            }
            if s > e {
                return Err(format!(
                    "span_target[{r}] is ({s}, {e}): start is after end"
                ));
            }
            if i64::from(e) >= self.lengths[r] {
                return Err(format!(
                    "span_target[{r}] ends at {e} but that row holds {} tokens",
                    self.lengths[r]
                ));
            }
            for (name, pos) in [("start", s), ("end", e)] {
                if !row[pos as usize] {
                    return Err(format!(
                        "span_target[{r}] {name}={pos} is not a line-start token"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// `shards.assemble_batch`: one padded batch, one row per sequence. `span_target` and
/// `line_starts` are carried only when a span row is present, and the line-start mask marks
/// every row's candidates, as Python's does.
#[allow(clippy::too_many_arguments)]
pub fn assemble_batch(
    sequences: &[Vec<i32>],
    rows: &[usize],
    kinds: Vec<u8>,
    target_index: Vec<i32>,
    spans: Vec<[i32; 2]>,
    candidates: &[Vec<i32>],
    width: usize,
    bucket: usize,
    index: u64,
) -> Result<Batch, String> {
    let n = sequences.len();
    let mut tokens = vec![PAD_ID; n * width];
    let mut lengths = Vec::with_capacity(n);
    for (r, seq) in sequences.iter().enumerate() {
        if seq.len() > width {
            return Err(format!(
                "sequence of {} tokens in a bucket {width} wide",
                seq.len()
            ));
        }
        tokens[r * width..r * width + seq.len()].copy_from_slice(seq);
        lengths.push(seq.len() as i64);
    }
    let has_span = kinds.contains(&SLOT_SPAN);
    let line_starts = if has_span {
        let mut mask = vec![false; n * width];
        for (r, cands) in candidates.iter().enumerate() {
            for &p in cands {
                let p = usize::try_from(p)
                    .ok()
                    .filter(|p| *p < width)
                    .ok_or(format!("candidate {p} outside width {width}"))?;
                mask[r * width + p] = true;
            }
        }
        Some(mask)
    } else {
        None
    };
    let batch = Batch {
        index,
        bucket,
        width,
        sequences: rows.to_vec(),
        tokens,
        lengths,
        slot_kind: kinds,
        target_index,
        span_target: if has_span { Some(spans) } else { None },
        line_starts,
    };
    batch.validate()?;
    Ok(batch)
}

// --- the consumed digest ----------------------------------------------------------------------

/// `run_control.ConsumedPrefix`: a running sha256 over the batches consumed, in order.
#[derive(Clone)]
pub struct ConsumedPrefix {
    hasher: Sha256,
    n: u64,
}

impl Default for ConsumedPrefix {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsumedPrefix {
    /// An empty prefix, seeded with the domain string.
    pub fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CONSUMED_PREFIX_DOMAIN);
        Self { hasher, n: 0 }
    }

    /// `ConsumedPrefix.fold(*parts)`: every part length-prefixed, the count first.
    pub fn fold(&mut self, parts: &[&[u8]]) {
        let mut head = vec![0u8];
        head.extend_from_slice(&(parts.len() as u32).to_be_bytes());
        self.hasher.update(&head);
        for p in parts {
            self.hasher.update((p.len() as u64).to_be_bytes());
            self.hasher.update(p);
        }
        self.n += 1;
    }

    /// `trainer._fold`: index, bucket, tokens, lengths and the four supervision channels
    /// (an absent channel folds as zero bytes).
    pub fn fold_batch(&mut self, batch: &Batch) {
        let tokens: Vec<u8> = batch.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let lengths: Vec<u8> = batch.lengths.iter().flat_map(|t| t.to_le_bytes()).collect();
        let target: Vec<u8> = batch
            .target_index
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        let spans: Vec<u8> = batch
            .span_target
            .as_ref()
            .map(|s| {
                s.iter()
                    .flat_map(|[a, b]| a.to_le_bytes().into_iter().chain(b.to_le_bytes()))
                    .collect()
            })
            .unwrap_or_default();
        let mask: Vec<u8> = batch
            .line_starts
            .as_ref()
            .map(|m| m.iter().map(|&b| u8::from(b)).collect())
            .unwrap_or_default();
        self.fold(&[
            &batch.index.to_be_bytes(),
            &(batch.bucket as u64).to_be_bytes(),
            &tokens,
            &lengths,
            &batch.slot_kind,
            &target,
            &spans,
            &mask,
        ]);
    }

    /// Batches folded.
    pub fn n_folded(&self) -> u64 {
        self.n
    }

    /// The digest so far.
    pub fn hexdigest(&self) -> String {
        hex(&self.hasher.clone().finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_for_takes_the_smallest_fitting_bucket_and_refuses_overflow() {
        assert_eq!(bucket_for(5, &[4, 8, 16]).unwrap(), 1);
        assert_eq!(bucket_for(8, &[4, 8, 16]).unwrap(), 1);
        assert!(bucket_for(17, &[4, 8, 16]).is_err());
        assert!(bucket_for(0, &[4]).is_err());
    }

    #[test]
    fn the_plan_refuses_a_budget_below_a_bucket_and_a_mistyped_budget() {
        assert!(plan_epoch(&[3, 9], &[4, 16], 8, 0, 0).is_err());
        assert!(plan_epoch(&[3], &[4], MAX_POSITIONS_PER_BATCH + 1, 0, 0).is_err());
        assert!(plan_epoch(&[3], &[4], 0, 0, 0).is_err());
        let plans = plan_epoch(&[3, 3, 3, 9], &[4, 16], 16, 1, 2).unwrap();
        let mut seen: Vec<usize> = plans.iter().flat_map(|p| p.rows.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3], "every sequence once per epoch");
    }

    #[test]
    fn an_empty_prefix_has_a_real_digest() {
        let empty = ConsumedPrefix::new().hexdigest();
        assert_eq!(empty.len(), 64);
        let mut one = ConsumedPrefix::new();
        one.fold(&[b"ab"]);
        let mut two = ConsumedPrefix::new();
        two.fold(&[b"a", b"b"]);
        assert_ne!(
            one.hexdigest(),
            two.hexdigest(),
            "parts are length-prefixed"
        );
    }
}
