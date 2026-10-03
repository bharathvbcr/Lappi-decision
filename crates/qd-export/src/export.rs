//! The export: refuse everything that is wrong before writing anything, then stage the release
//! in a sibling directory and rename it into place only once every file is written and checked.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use qd_runtime::calibration::CalibrationTable;
use qd_runtime::hex;

use crate::bf16::{check_finite_f32, to_bf16, BadValue, FloatSource};
use crate::layout::{Layout, SPAN_PREFIX, TEXT_PREFIX, TOWER_PREFIX};
use crate::refusal::{Refusal, RefusalKind, Result};
use crate::safetensors::{sha256_file, Dtype, PlannedTensor, SafeTensorsFile, TensorInfo, Writer, CHUNK_BYTES};

/// The release's file names and manifest format. Owned by the reader, `qd_runtime::release`, so
/// the writer and the runtime cannot spell a file two ways. The first three are what the loader
/// opens (`qd-metal/src/config.rs:88`, `model.rs:896`, `backend.rs:331`); no serving loader reads
/// the span head yet (`GAP-J7-EXPORT-SPAN-HEAD-UNSERVED`).
pub use qd_runtime::release::{
    CALIBRATION_FILE, CONFIG_FILE, MANIFEST_FILE, MANIFEST_FORMAT, MAX_SMALL_FILE_BYTES,
    SPAN_HEAD_FILE, TOKENIZER_FILE, TRAINED_FAMILIES_KEY, WEIGHTS_FILE,
};
use qd_runtime::release::validate_trained_families;

/// Every tokenizer file copied from the base snapshot. `tokenizer.json` is the one the loader
/// reads and hashes into the backend's identity (`qd-metal/src/tokenizer.rs:28-37`); the other
/// three are what `transformers`' tokenizer loads beside it. All four are in the Qwen3.5-2B-Base
/// snapshot `b1485b2f`, and a snapshot missing one is not that snapshot.
pub const TOKENIZER_FILES: [&str; 4] = [TOKENIZER_FILE, "tokenizer_config.json", "vocab.json", "merges.txt"];

/// A source tensor dropped from the release on purpose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedExtra {
    pub name: String,
    pub reason: String,
}

impl AllowedExtra {
    /// `NAME=REASON`, both non-empty.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let (name, reason) = s
            .split_once('=')
            .ok_or_else(|| format!("{s:?} is not NAME=REASON"))?;
        let (name, reason) = (name.trim(), reason.trim());
        if name.is_empty() || reason.is_empty() {
            return Err(format!("{s:?}: both the tensor name and the reason must be non-empty"));
        }
        Ok(Self {
            name: name.to_string(),
            reason: reason.to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExportRequest {
    /// The averaged checkpoint (`tools/ckpt_average.py --out`).
    pub source: PathBuf,
    /// Its manifest; `<source>.manifest.json` when `None` (`tools/ckpt_average.py:188`).
    pub source_manifest: Option<PathBuf>,
    /// The Qwen3.5-2B-Base snapshot directory: `config.json` and the tokenizer files.
    pub base_snapshot: PathBuf,
    /// Expected SHA-256 of the base snapshot's `tokenizer.json`, hex.
    pub tokenizer_sha256: String,
    /// The vocabulary the operator expects: the config, the embedding and the source manifest
    /// must all say this.
    pub expect_vocab_size: usize,
    pub calibration: Option<PathBuf>,
    /// The train split's data manifest the tower was trained from
    /// (`python/qd_data/manifest.py`, `split: "train"`). The release records its family set as
    /// `trained_families`, and the runtime refuses every other task (Fable's pipeline ruling,
    /// item 4). Required: a release that cannot say what it trained admits nothing.
    pub train_manifest: PathBuf,
    pub allow_extra: Vec<AllowedExtra>,
    /// The release directory; must not exist.
    pub out: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSummary {
    pub out: PathBuf,
    pub weight_hash: String,
    pub tokenizer_hash: String,
    pub calibration_hash: Option<String>,
    /// The `trained_families` the manifest records: sorted, unique.
    pub trained_families: Vec<String>,
    /// File name -> SHA-256 hex, for every file except the manifest itself.
    pub files: BTreeMap<String, String>,
    /// Tower tensors rounded from F32/F16, and copied from BF16.
    pub rounded: usize,
    pub copied: usize,
    pub dropped: Vec<AllowedExtra>,
}

pub(crate) fn refuse(kind: RefusalKind, msg: impl Into<String>) -> Refusal {
    Refusal::new(kind, msg)
}

/// Read a small file whole, bounded.
pub(crate) fn read_small(path: &Path, kind: RefusalKind) -> Result<Vec<u8>> {
    let meta = std::fs::metadata(path).map_err(|e| refuse(kind, format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(refuse(kind, format!("{} is not a file", path.display())));
    }
    if meta.len() > MAX_SMALL_FILE_BYTES {
        return Err(refuse(kind, format!("{}: {} bytes exceeds {MAX_SMALL_FILE_BYTES}", path.display(), meta.len())));
    }
    std::fs::read(path).map_err(|e| refuse(kind, format!("{}: {e}", path.display())))
}

pub(crate) fn len64(bytes: &[u8]) -> Result<u64> {
    u64::try_from(bytes.len()).map_err(|e| refuse(RefusalKind::Io, format!("byte count: {e}")))
}

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub(crate) fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// What the source manifest must say.
struct SourceManifest {
    path: PathBuf,
    sha256: String,
    safetensors_sha256: String,
    n_tensors: usize,
    vocab_size: usize,
    ft_row_ids: Vec<String>,
    tool: Option<String>,
}

fn read_source_manifest(path: &Path) -> Result<SourceManifest> {
    let k = RefusalKind::SourceManifest;
    let bytes = read_small(path, k)?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| refuse(k, format!("{}: not JSON: {e}", path.display())))?;
    let field = |key: &str| v.get(key).ok_or_else(|| refuse(k, format!("{}: no {key}", path.display())));
    let st_sha = field("safetensors_sha256")?
        .as_str()
        .filter(|s| is_hex64(s))
        .ok_or_else(|| refuse(k, format!("{}: safetensors_sha256 is not 64 hex digits", path.display())))?
        .to_ascii_lowercase();
    let count = |key: &str| -> Result<usize> {
        field(key)?
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| refuse(k, format!("{}: {key} is not a non-negative integer", path.display())))
    };
    let n_tensors = count("n_tensors")?;
    let vocab_size = count("vocab_size")?;
    let ids = field("ft_row_ids")?
        .as_array()
        .ok_or_else(|| refuse(k, format!("{}: ft_row_ids is not an array", path.display())))?;
    let ft_row_ids = ids
        .iter()
        .map(|id| {
            id.as_str()
                .filter(|s| !s.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| refuse(k, format!("{}: ft_row_ids holds {id}, not a non-empty string", path.display())))
        })
        .collect::<Result<Vec<_>>>()?;
    if ft_row_ids.is_empty() {
        return Err(refuse(
            k,
            format!(
                "{}: ft_row_ids is empty. A release names the ledger rows its weights came from; \
                 rerun tools/ckpt_average.py with --ft-row-ids",
                path.display()
            ),
        ));
    }
    Ok(SourceManifest {
        path: path.to_path_buf(),
        sha256: hex(&sha256(&bytes)),
        safetensors_sha256: st_sha,
        n_tensors,
        vocab_size,
        ft_row_ids,
        tool: v.get("tool").and_then(Value::as_str).map(str::to_string),
    })
}

/// `qd_data.manifest.MANIFEST_FORMAT_VERSION`.
const TRAIN_MANIFEST_FORMAT_VERSION: i64 = 1;
/// The split whose families the weights were trained on (`qd_data.config.SPLITS`).
const TRAIN_SPLIT: &str = "train";
/// Bytes of a manifest string a refusal echoes back.
const ECHO_BYTES: usize = 128;

/// `s` quoted, cut to [`ECHO_BYTES`] on a character boundary.
fn echo(s: &str) -> String {
    if s.len() <= ECHO_BYTES {
        return format!("{s:?}");
    }
    let mut end = ECHO_BYTES;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{:?} (first {end} of {} bytes)", &s[..end], s.len())
}

/// What the train manifest says the tower was trained on.
struct TrainManifest {
    path: PathBuf,
    sha256: String,
    /// As the file records it. Not re-derived here: the canonical-JSON port that re-derives it
    /// is `qd-train`'s held-out door, and `qd-train` depends on this crate.
    data_snapshot_hash: String,
    n_rows: usize,
    rows_by_family: BTreeMap<String, u64>,
}

impl TrainManifest {
    /// The family set, sorted and unique: the keys of `rows_by_family`.
    fn families(&self) -> Vec<String> {
        self.rows_by_family.keys().cloned().collect()
    }
}

/// Read the train split's data manifest (`Manifest.to_json()`, `python/qd_data/manifest.py`)
/// for the families its entries hold.
///
/// Refused unless it is format 1, declares `split: "train"`, records a 64-hex
/// `data_snapshot_hash`, `held_out_families` and an `n_rows` equal to its entry count, and every
/// entry is a train row with a non-blank `family_id` outside the held-out families (CLAUDE.md
/// rule 3: a train manifest naming one is not a manifest anything was trained from). The family
/// set must also be one the runtime's reader accepts ([`validate_trained_families`]).
///
/// The file is read whole within [`MAX_SMALL_FILE_BYTES`] and parsed as one JSON document. The
/// v4 train manifest is 152,784,427 bytes (`data/pool/train.json`), which fits; the export runs
/// on the box, where its parse is not the memory that matters.
fn read_train_manifest(path: &Path) -> Result<TrainManifest> {
    let k = RefusalKind::TrainManifest;
    let bad = |detail: String| refuse(k, format!("{}: {detail}", path.display()));
    let bytes = read_small(path, k)?;
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| bad(format!("not JSON: {e}")))?;
    let obj = doc
        .as_object()
        .ok_or_else(|| bad("not a JSON object; a data manifest is one".to_string()))?;
    let version = obj.get("manifest_format_version").and_then(Value::as_i64);
    if version != Some(TRAIN_MANIFEST_FORMAT_VERSION) {
        return Err(bad(format!(
            "manifest_format_version is {}, not {TRAIN_MANIFEST_FORMAT_VERSION}; this reader does \
             not interpret another format",
            version.map_or_else(|| "absent or not an integer".to_string(), |v| v.to_string())
        )));
    }
    let split = obj.get("split").and_then(Value::as_str);
    if split != Some(TRAIN_SPLIT) {
        return Err(bad(format!(
            "split is {}, not {TRAIN_SPLIT:?}. A release records the families its weights were \
             trained on, which are the train split's; a val or held-out manifest names others",
            split.map_or_else(|| "absent or not a string".to_string(), echo)
        )));
    }
    let data_snapshot_hash = obj
        .get("data_snapshot_hash")
        .and_then(Value::as_str)
        .filter(|s| is_hex64(s))
        .map(str::to_string)
        .ok_or_else(|| bad("data_snapshot_hash is not 64 hex digits".to_string()))?;
    let held_out = obj
        .get("held_out_families")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            bad("held_out_families is not an array; without it a held-out family among the \
                 entries could not be refused"
                .to_string())
        })?
        .iter()
        .map(|family| {
            family
                .as_str()
                .ok_or_else(|| bad("held_out_families holds a value that is not a string".to_string()))
        })
        .collect::<Result<BTreeSet<&str>>>()?;
    let entries = obj
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("entries is not an array".to_string()))?;
    if entries.is_empty() {
        return Err(bad(
            "no rows: entries is empty, so it names no family anything was trained on".to_string(),
        ));
    }
    let n_rows = len64_of(entries.len())?;
    if obj.get("n_rows").and_then(Value::as_u64) != Some(n_rows) {
        return Err(bad(format!(
            "n_rows is {}, but entries holds {n_rows}; the file is truncated or was edited",
            obj.get("n_rows")
                .map_or_else(|| "absent".to_string(), |n| echo(&n.to_string()))
        )));
    }
    let mut rows_by_family: BTreeMap<String, u64> = BTreeMap::new();
    for (i, entry) in entries.iter().enumerate() {
        let row_id = entry
            .get("row_id")
            .and_then(Value::as_str)
            .map_or_else(|| "absent".to_string(), echo);
        let entry_split = entry.get("split").and_then(Value::as_str);
        if entry_split != Some(TRAIN_SPLIT) {
            return Err(bad(format!(
                "entry {i} (row_id {row_id}) is split {}, in a manifest for split {TRAIN_SPLIT:?}",
                entry_split.map_or_else(|| "absent or not a string".to_string(), echo)
            )));
        }
        let family = entry
            .get("family_id")
            .and_then(Value::as_str)
            .filter(|family| !qd_runtime::is_blank(family))
            .ok_or_else(|| {
                bad(format!(
                    "entry {i} (row_id {row_id}) has no family_id, or a blank one"
                ))
            })?;
        if held_out.contains(family) {
            return Err(bad(format!(
                "entry {i} (row_id {row_id}) names held-out family {}; the held-out families \
                 {held_out:?} are never trained (CLAUDE.md rule 3), so no release was trained \
                 from this manifest",
                echo(family)
            )));
        }
        *rows_by_family.entry(family.to_string()).or_default() += 1;
    }
    let train = TrainManifest {
        path: path.to_path_buf(),
        sha256: hex(&sha256(&bytes)),
        data_snapshot_hash,
        n_rows: entries.len(),
        rows_by_family,
    };
    validate_trained_families(&train.families()).map_err(|detail| {
        bad(format!(
            "its families are not a list a release can record, and the runtime would refuse the \
             release: {detail}"
        ))
    })?;
    Ok(train)
}

/// A count as `u64`, refused rather than truncated.
fn len64_of(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|e| refuse(RefusalKind::Io, format!("count {n}: {e}")))
}

/// Structural equality in which numbers compare by value (`1` == `1.0`), so a table written
/// with an integer literal is not refused for its spelling.
fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64() && x.as_f64().is_some(),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_json(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(key, p)| y.get(key).is_some_and(|q| same_json(p, q)))
        }
        _ => a == b,
    }
}

/// The calibration table: parsed as the runtime's own type, validated, and refused unless it
/// reads back as the same document. serde ignores fields a struct does not name, so without the
/// read-back a table carrying a field the runtime drops would pass through looking honoured.
pub(crate) fn read_calibration(path: &Path) -> Result<(Vec<u8>, CalibrationTable)> {
    let k = RefusalKind::Calibration;
    let bytes = read_small(path, k)?;
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| refuse(k, format!("{}: not JSON: {e}", path.display())))?;
    let table: CalibrationTable = serde_json::from_value(doc.clone())
        .map_err(|e| refuse(k, format!("{}: not a CalibrationTable: {e}", path.display())))?;
    table
        .validate()
        .map_err(|e| refuse(k, format!("{}: {e}", path.display())))?;
    let back = serde_json::to_value(&table).map_err(|e| refuse(k, format!("{}: {e}", path.display())))?;
    if !same_json(&doc, &back) {
        return Err(refuse(
            k,
            format!(
                "{}: the file does not read back as itself through qd_runtime's CalibrationTable \
                 (a field the runtime does not read, or one it fills in); the release would carry \
                 a table whose bytes say more than the runtime uses",
                path.display()
            ),
        ));
    }
    Ok((bytes, table))
}

/// One planned tower tensor and where it comes from.
struct TowerTensor {
    short: String,
    source: TensorInfo,
    kind: FloatSource,
}

/// Everything decided before a byte is written.
struct Plan {
    layout: Layout,
    config_bytes: Vec<u8>,
    manifest: SourceManifest,
    source_sha256: String,
    tower: Vec<TowerTensor>,
    head: Vec<(String, TensorInfo)>,
    dropped: Vec<AllowedExtra>,
    dtype_counts: BTreeMap<&'static str, usize>,
    tokenizer: Vec<(&'static str, Vec<u8>)>,
    calibration: Option<(PathBuf, Vec<u8>, CalibrationTable)>,
    train: TrainManifest,
}

fn plan(req: &ExportRequest, src: &SafeTensorsFile) -> Result<Plan> {
    // Config, and the operator's vocabulary.
    let config_path = req.base_snapshot.join(CONFIG_FILE);
    let config_bytes = read_small(&config_path, RefusalKind::Config)?;
    let config_text = std::str::from_utf8(&config_bytes)
        .map_err(|e| refuse(RefusalKind::Config, format!("{}: {e}", config_path.display())))?;
    let layout = Layout::from_config_json(config_text)?;
    if layout.vocab != req.expect_vocab_size {
        return Err(refuse(
            RefusalKind::VocabMismatch,
            format!(
                "{} says vocab_size {}, but --expect-vocab-size is {}",
                config_path.display(),
                layout.vocab,
                req.expect_vocab_size
            ),
        ));
    }

    // Tokenizer files, pinned.
    let pin = req.tokenizer_sha256.to_ascii_lowercase();
    if !is_hex64(&pin) {
        return Err(refuse(RefusalKind::Tokenizer, format!("--tokenizer-sha256 {:?} is not 64 hex digits", req.tokenizer_sha256)));
    }
    let mut tokenizer = Vec::with_capacity(TOKENIZER_FILES.len());
    for name in TOKENIZER_FILES {
        let bytes = read_small(&req.base_snapshot.join(name), RefusalKind::Tokenizer)?;
        if name == TOKENIZER_FILE {
            let got = hex(&sha256(&bytes));
            if got != pin {
                return Err(refuse(
                    RefusalKind::Tokenizer,
                    format!(
                        "{}: sha256 {got}, pinned {pin}. This is not the tokenizer the release is \
                         being cut for",
                        req.base_snapshot.join(name).display()
                    ),
                ));
            }
        }
        tokenizer.push((name, bytes));
    }

    let calibration = match &req.calibration {
        Some(p) => {
            let (bytes, table) = read_calibration(p)?;
            Some((p.clone(), bytes, table))
        }
        None => None,
    };

    // What the tower was trained on.
    let train = read_train_manifest(&req.train_manifest)?;

    // The source manifest, against the source's header.
    let manifest_path = req
        .source_manifest
        .clone()
        .unwrap_or_else(|| {
            let mut name = req.source.file_name().map(std::ffi::OsString::from).unwrap_or_default();
            name.push(".manifest.json");
            req.source.with_file_name(name)
        });
    let manifest = read_source_manifest(&manifest_path)?;
    if manifest.n_tensors != src.tensors().len() {
        return Err(refuse(
            RefusalKind::SourceManifest,
            format!(
                "{} counts {} tensors, {} holds {}",
                manifest.path.display(),
                manifest.n_tensors,
                src.path().display(),
                src.tensors().len()
            ),
        ));
    }
    if manifest.vocab_size != layout.vocab {
        return Err(refuse(
            RefusalKind::VocabMismatch,
            format!(
                "{} says vocab_size {}, the config {}",
                manifest.path.display(),
                manifest.vocab_size,
                layout.vocab
            ),
        ));
    }

    // Classify every source tensor.
    let want_text = layout.text_tensors()?;
    let want_head = layout.span_head_tensors();
    let mut tower: BTreeMap<String, &TensorInfo> = BTreeMap::new();
    let mut head: BTreeMap<String, &TensorInfo> = BTreeMap::new();
    let mut extras: Vec<&str> = Vec::new();
    for (name, info) in src.tensors() {
        if let Some(short) = name.strip_prefix(TOWER_PREFIX)
            && want_text.contains_key(short)
        {
            tower.insert(short.to_string(), info);
        } else if let Some(short) = name.strip_prefix(SPAN_PREFIX)
            && want_head.contains_key(short)
        {
            head.insert(short.to_string(), info);
        } else {
            extras.push(name);
        }
    }

    let mut missing: Vec<String> = want_text
        .keys()
        .filter(|k| !tower.contains_key(*k))
        .map(|k| format!("{TOWER_PREFIX}{k}"))
        .collect();
    missing.extend(want_head.keys().filter(|k| !head.contains_key(*k)).map(|k| format!("{SPAN_PREFIX}{k}")));
    if !missing.is_empty() {
        return Err(refuse(
            RefusalKind::MissingTensor,
            format!("{} of the tensors the release needs are not in the source: {missing:?}", missing.len()),
        ));
    }

    // Extras: each must be named in the allowlist with a reason, and every allowlist entry must
    // name an extra -- a stale entry is a reason nobody re-read.
    let mut allowed: BTreeMap<&str, &AllowedExtra> = BTreeMap::new();
    for a in &req.allow_extra {
        if allowed.insert(a.name.as_str(), a).is_some() {
            return Err(refuse(RefusalKind::ExtraTensor, format!("--allow-extra names {} twice", a.name)));
        }
    }
    let unexplained: Vec<&str> = extras.iter().copied().filter(|n| !allowed.contains_key(n)).collect();
    if !unexplained.is_empty() {
        return Err(refuse(
            RefusalKind::ExtraTensor,
            format!(
                "the source holds {} tensor(s) the release would not carry: {unexplained:?}. The \
                 loader reads exactly the model.language_model.* set; pass --allow-extra \
                 NAME=REASON to drop one on purpose",
                unexplained.len()
            ),
        ));
    }
    let extra_set: BTreeSet<&str> = extras.iter().copied().collect();
    let stale: Vec<&str> = allowed.keys().copied().filter(|n| !extra_set.contains(n)).collect();
    if !stale.is_empty() {
        return Err(refuse(
            RefusalKind::ExtraTensor,
            format!("--allow-extra names {stale:?}, which are not extra tensors in the source"),
        ));
    }
    let dropped: Vec<AllowedExtra> = extras.iter().map(|n| allowed[n].clone()).collect();

    // The vocabulary the embedding actually has, named as such rather than as a shape.
    let embed = tower["embed_tokens.weight"];
    if embed.shape.first() != Some(&layout.vocab) {
        return Err(refuse(
            RefusalKind::VocabMismatch,
            format!(
                "{TOWER_PREFIX}embed_tokens.weight is {:?}; the config's vocabulary is {} rows. A \
                 remapped tower is not servable under the base config",
                embed.shape, layout.vocab
            ),
        ));
    }

    let mut bad_shapes = Vec::new();
    for (short, info) in &tower {
        if info.shape != want_text[short] {
            bad_shapes.push(format!("{TOWER_PREFIX}{short}: {:?}, config implies {:?}", info.shape, want_text[short]));
        }
    }
    for (short, info) in &head {
        if info.shape != want_head[short] {
            bad_shapes.push(format!("{SPAN_PREFIX}{short}: {:?}, config implies {:?}", info.shape, want_head[short]));
        }
    }
    if !bad_shapes.is_empty() {
        return Err(refuse(RefusalKind::ShapeMismatch, bad_shapes.join("; ")));
    }

    let mut bad_dtypes = Vec::new();
    let mut tower_out = Vec::with_capacity(tower.len());
    let mut dtype_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (short, info) in &tower {
        match FloatSource::from_dtype(info.dtype) {
            Some(kind) => {
                *dtype_counts.entry(info.dtype.as_str()).or_default() += 1;
                tower_out.push(TowerTensor {
                    short: short.clone(),
                    source: (*info).clone(),
                    kind,
                });
            }
            None => bad_dtypes.push(format!(
                "{TOWER_PREFIX}{short} is {}: not a float dtype a tower is trained in (BF16, F16, \
                 F32), so a cast to bf16 would change what it means, not only its precision",
                info.dtype.as_str()
            )),
        }
    }
    for (short, info) in &head {
        if info.dtype != Dtype::F32 {
            bad_dtypes.push(format!(
                "{SPAN_PREFIX}{short} is {}: the span head is built float32 on purpose \
                 (python/qd_train/backbone.py:825-833) and is released as it was trained",
                info.dtype.as_str()
            ));
        }
    }
    if !bad_dtypes.is_empty() {
        return Err(refuse(RefusalKind::Dtype, bad_dtypes.join("; ")));
    }

    // Last, because it reads the whole file: the manifest must describe these bytes.
    let source_sha256 = hex(&src.sha256()?);
    if source_sha256 != manifest.safetensors_sha256 {
        return Err(refuse(
            RefusalKind::SourceManifest,
            format!(
                "{} has sha256 {source_sha256}, but {} describes {}",
                src.path().display(),
                manifest.path.display(),
                manifest.safetensors_sha256
            ),
        ));
    }

    Ok(Plan {
        layout,
        config_bytes,
        manifest,
        source_sha256,
        tower: tower_out,
        head: head.into_iter().map(|(k, v)| (k, v.clone())).collect(),
        dropped,
        dtype_counts,
        tokenizer,
        calibration,
        train,
    })
}

fn non_finite(name: &str, b: BadValue) -> Refusal {
    refuse(
        RefusalKind::NonFinite,
        format!(
            "{name}[{}] is {}: {}",
            b.index,
            b.value,
            if b.value.is_finite() {
                "finite in the source but past bf16's largest value, so it would round to infinity"
            } else {
                "not a finite number"
            }
        ),
    )
}

/// Write `bytes` to a new file, fsync it, then read it back and require the same SHA-256.
pub(crate) fn write_verified(path: &Path, bytes: &[u8], kind: RefusalKind) -> Result<String> {
    let want = hex(&sha256(bytes));
    let mut f = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| Refusal::io(path, e))?;
    f.write_all(bytes).map_err(|e| Refusal::io(path, e))?;
    f.sync_all().map_err(|e| Refusal::io(path, e))?;
    let got = hex(&sha256_file(path)?);
    if got != want {
        return Err(refuse(kind, format!("{}: wrote sha256 {want}, read back {got}", path.display())));
    }
    Ok(got)
}

/// qd-metal's weight hash (`weights.rs:262-272`): SHA-256 over `name \0 sha256(bytes)` for every
/// tensor the loader reads, in name order. `Model::load` reports it as the backend identity's
/// `weight_hash` (`backend.rs:360`), which a caller pins through `HashExpectation`.
pub fn weight_hash(digests: &[(String, [u8; 32])]) -> String {
    let mut sorted: Vec<&(String, [u8; 32])> = digests.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut buf = Vec::with_capacity(sorted.len() * 96);
    for (name, d) in sorted {
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(d);
    }
    hex(&sha256(&buf))
}

fn write_release(req: &ExportRequest, src: &SafeTensorsFile, plan: &Plan, dir: &Path) -> Result<ExportSummary> {
    let mut files: BTreeMap<String, Value> = BTreeMap::new();
    let mut shas: BTreeMap<String, String> = BTreeMap::new();
    let mut record = |name: &str, sha: String, bytes: u64| {
        files.insert(name.to_string(), json!({"sha256": sha, "bytes": bytes}));
        shas.insert(name.to_string(), sha);
    };

    // model.safetensors: every tower tensor as BF16 under the loader's name.
    let by_short: BTreeMap<&str, &TowerTensor> = plan.tower.iter().map(|t| (t.short.as_str(), t)).collect();
    let weights_path = dir.join(WEIGHTS_FILE);
    let mut w = Writer::create(
        &weights_path,
        plan.tower
            .iter()
            .map(|t| PlannedTensor {
                name: format!("{TEXT_PREFIX}{}", t.short),
                dtype: Dtype::Bf16,
                shape: t.source.shape.clone(),
            })
            .collect(),
    )?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut converted = Vec::with_capacity(CHUNK_BYTES);
    let (mut rounded, mut copied) = (0usize, 0usize);
    for planned in w.order() {
        let short = planned
            .name
            .strip_prefix(TEXT_PREFIX)
            .ok_or_else(|| refuse(RefusalKind::Io, format!("writer planned {} outside {TEXT_PREFIX}", planned.name)))?;
        let t = by_short
            .get(short)
            .ok_or_else(|| refuse(RefusalKind::Io, format!("writer planned {} with no source", planned.name)))?;
        let src_name = format!("{TOWER_PREFIX}{short}");
        w.begin(&planned.name)?;
        let size = u64::try_from(t.kind.size()).map_err(|e| Refusal::io(&weights_path, e))?;
        let total = t.source.nbytes();
        let mut at = 0u64;
        while at < total {
            let n = usize::try_from((total - at).min(u64::try_from(CHUNK_BYTES).map_err(|e| Refusal::io(&weights_path, e))?))
                .map_err(|e| Refusal::io(&weights_path, e))?;
            src.read_tensor_range(&t.source, at, &mut buf[..n])?;
            converted.clear();
            to_bf16(t.kind, &buf[..n], at / size, &mut converted).map_err(|b| non_finite(&src_name, b))?;
            w.write(&converted)?;
            at += u64::try_from(n).map_err(|e| Refusal::io(&weights_path, e))?;
        }
        w.end()?;
        match t.kind {
            FloatSource::Bf16 => copied += 1,
            FloatSource::F16 | FloatSource::F32 => rounded += 1,
        }
    }
    let weights = w.finish()?;
    let weight_hash = weight_hash(&weights.tensor_digests);
    record(WEIGHTS_FILE, hex(&weights.sha256), weights.bytes);

    // span_head.safetensors: F32, bit for bit.
    let head_path = dir.join(SPAN_HEAD_FILE);
    let mut hw = Writer::create(
        &head_path,
        plan.head
            .iter()
            .map(|(short, info)| PlannedTensor {
                name: short.clone(),
                dtype: Dtype::F32,
                shape: info.shape.clone(),
            })
            .collect(),
    )?;
    let head_by: BTreeMap<&str, &TensorInfo> = plan.head.iter().map(|(k, v)| (k.as_str(), v)).collect();
    for planned in hw.order() {
        let info = head_by
            .get(planned.name.as_str())
            .ok_or_else(|| refuse(RefusalKind::Io, format!("writer planned {} with no source", planned.name)))?;
        let src_name = format!("{SPAN_PREFIX}{}", planned.name);
        hw.begin(&planned.name)?;
        let total = info.nbytes();
        let mut at = 0u64;
        while at < total {
            let n = usize::try_from((total - at).min(u64::try_from(CHUNK_BYTES).map_err(|e| Refusal::io(&head_path, e))?))
                .map_err(|e| Refusal::io(&head_path, e))?;
            src.read_tensor_range(info, at, &mut buf[..n])?;
            check_finite_f32(&buf[..n], at / 4).map_err(|b| non_finite(&src_name, b))?;
            hw.write(&buf[..n])?;
            at += u64::try_from(n).map_err(|e| Refusal::io(&head_path, e))?;
        }
        hw.end()?;
    }
    let head = hw.finish()?;
    record(SPAN_HEAD_FILE, hex(&head.sha256), head.bytes);

    // config.json and the tokenizer files, byte for byte.
    let config_sha = write_verified(&dir.join(CONFIG_FILE), &plan.config_bytes, RefusalKind::Config)?;
    record(CONFIG_FILE, config_sha.clone(), len64(&plan.config_bytes)?);
    let mut tokenizer_hash = String::new();
    for (name, bytes) in &plan.tokenizer {
        let sha = write_verified(&dir.join(name), bytes, RefusalKind::Tokenizer)?;
        if *name == TOKENIZER_FILE {
            tokenizer_hash = sha.clone();
        }
        record(name, sha, len64(bytes)?);
    }

    let calibration = match &plan.calibration {
        Some((path, bytes, table)) => {
            let sha = write_verified(&dir.join(CALIBRATION_FILE), bytes, RefusalKind::Calibration)?;
            record(CALIBRATION_FILE, sha.clone(), len64(bytes)?);
            Some(json!({
                "file": CALIBRATION_FILE,
                "source": path.display().to_string(),
                "file_sha256": sha,
                "name": table.name,
                "table_hash": table.hash(),
            }))
        }
        None => None,
    };
    let calibration_hash = plan.calibration.as_ref().map(|(_, _, t)| t.hash());
    let trained_families = plan.train.families();

    let mut manifest = json!({
        "format": MANIFEST_FORMAT,
        "tool": "qd-export",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "loader": {
            "crate": "qd-metal",
            "config": CONFIG_FILE,
            "weights": WEIGHTS_FILE,
            "tensor_prefix": TEXT_PREFIX,
            "tokenizer": TOKENIZER_FILE,
            "span_head": format!("{SPAN_HEAD_FILE} (no serving loader reads it yet: GAP-J7-EXPORT-SPAN-HEAD-UNSERVED)"),
        },
        "source": {
            "safetensors": req.source.display().to_string(),
            "safetensors_sha256": plan.source_sha256,
            "manifest": plan.manifest.path.display().to_string(),
            "manifest_sha256": plan.manifest.sha256,
            "manifest_tool": plan.manifest.tool,
            "ft_row_ids": plan.manifest.ft_row_ids,
            "n_tensors": src.tensors().len(),
            "tower_dtypes": plan.dtype_counts,
        },
        "base_snapshot": {
            "path": req.base_snapshot.display().to_string(),
            "config_sha256": config_sha,
            "tokenizer_json_sha256_pinned": req.tokenizer_sha256.to_ascii_lowercase(),
        },
        "conversion": {
            "tower": "every model.language_model.* tensor is BF16: F32 rounded to nearest with ties \
                      to even, F16 widened exactly then rounded once, BF16 copied bit for bit; a \
                      NaN, an infinity, or a value rounding to infinity is refused",
            "tower_rounded": rounded,
            "tower_copied": copied,
            "span_head": "F32, copied bit for bit",
            "vocab_size": plan.layout.vocab,
            "hidden_size": plan.layout.hidden,
            "num_hidden_layers": plan.layout.layer_kinds.len(),
        },
        "dropped_extras": plan.dropped.iter().map(|d| json!({"name": d.name, "reason": d.reason})).collect::<Vec<_>>(),
        "calibration": calibration,
        // What goes together: this tower, served under this config.json, with this tokenizer.
        // `qd_runtime::release::Release::open` refuses a config.json whose sha256 is not
        // `config_sha256`, and `Release::check_backend` a loaded tower whose hash is not
        // `weight_hash`.
        "expected_identity": {
            "weight_hash": weight_hash,
            "config_sha256": config_sha,
            "tokenizer_hash": tokenizer_hash,
            "calibration_hash": calibration_hash,
            "not_computed": {
                "head_hash": "needs the tokenizer's 17 letter ids (qd-metal/src/backend.rs:345-356); \
                              this binary links no tokenizer",
                "label_set_hash": "needs the tokenizer's 17 letter ids (qd-metal/src/backend.rs:354-355)",
            },
        },
        "files": files,
        // Where `trained_families` came from. The runtime reads only `trained_families`.
        "train_manifest": {
            "path": plan.train.path.display().to_string(),
            "sha256": plan.train.sha256,
            "split": TRAIN_SPLIT,
            "n_rows": plan.train.n_rows,
            "rows_by_family": plan.train.rows_by_family,
            "data_snapshot_hash_as_recorded": plan.train.data_snapshot_hash,
            "rule": "trained_families is the sorted, unique family_id set of this manifest's \
                     entries. data_snapshot_hash is as the file records it, not re-derived here \
                     (qd-train's held-out door re-derives it)",
        },
    });
    // The task families the tower was trained on: `qd_runtime::admission` refuses every other
    // task as `task_not_trained` (Fable's pipeline ruling, item 4).
    manifest[TRAINED_FAMILIES_KEY] = Value::from(trained_families.clone());
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| refuse(RefusalKind::Io, e.to_string()))? + "\n";
    write_verified(&dir.join(MANIFEST_FILE), text.as_bytes(), RefusalKind::Io)?;

    Ok(ExportSummary {
        out: req.out.clone(),
        weight_hash,
        tokenizer_hash,
        calibration_hash,
        trained_families,
        files: shas,
        rounded,
        copied,
        dropped: plan.dropped.clone(),
    })
}

/// `<parent>/.<name>.qd-export-partial-<pid>`: beside the output, so the final rename does not
/// cross a filesystem.
fn staging_dir(out: &Path) -> Result<PathBuf> {
    let name = out
        .file_name()
        .ok_or_else(|| refuse(RefusalKind::OutputExists, format!("{} has no final component", out.display())))?;
    let mut staged = std::ffi::OsString::from(".");
    staged.push(name);
    staged.push(format!(".qd-export-partial-{}", std::process::id()));
    Ok(out.with_file_name(staged))
}

/// A new output directory and the sibling directory it is staged in. Shared by every writer in
/// this crate, so a release and an ensemble are published the same way.
pub(crate) struct Staging {
    out: PathBuf,
    parent: PathBuf,
    dir: PathBuf,
}

impl Staging {
    /// Refuse unless `out` does not exist, its parent is a directory, and its staging directory
    /// is free. Nothing is created.
    pub(crate) fn plan(out: &Path) -> Result<Self> {
        if std::fs::symlink_metadata(out).is_ok() {
            return Err(refuse(
                RefusalKind::OutputExists,
                format!("{} already exists; a release is never overwritten", out.display()),
            ));
        }
        let parent = out
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err(refuse(RefusalKind::Io, format!("{} is not a directory", parent.display())));
        }
        let dir = staging_dir(out)?;
        if std::fs::symlink_metadata(&dir).is_ok() {
            return Err(refuse(
                RefusalKind::OutputExists,
                format!("{} exists; another export may be running", dir.display()),
            ));
        }
        Ok(Self {
            out: out.to_path_buf(),
            parent: parent.to_path_buf(),
            dir,
        })
    }

    /// Create the staging directory, run `write` into it, and rename it to `out` only if
    /// `write` succeeded and `out` is still free. On any failure the staging directory is
    /// removed, and the refusal says so if it could not be.
    pub(crate) fn publish<T>(&self, write: impl FnOnce(&Path) -> Result<T>) -> Result<T> {
        std::fs::create_dir(&self.dir).map_err(|e| Refusal::io(&self.dir, e))?;
        let written = write(&self.dir).and_then(|summary| {
            if std::fs::symlink_metadata(&self.out).is_ok() {
                return Err(refuse(
                    RefusalKind::OutputExists,
                    format!("{} appeared while the release was being written", self.out.display()),
                ));
            }
            std::fs::rename(&self.dir, &self.out).map_err(|e| Refusal::io(&self.out, e))?;
            File::open(&self.parent)
                .and_then(|d| d.sync_all())
                .map_err(|e| Refusal::io(&self.parent, e))?;
            Ok(summary)
        });
        match written {
            Ok(summary) => Ok(summary),
            Err(mut err) => {
                if std::fs::symlink_metadata(&self.dir).is_ok()
                    && let Err(cleanup) = std::fs::remove_dir_all(&self.dir)
                {
                    err.detail.push_str(&format!(
                        "; and the staging directory {} could not be removed: {cleanup}",
                        self.dir.display()
                    ));
                }
                Err(err)
            }
        }
    }
}

/// Export `req.source` to `req.out`. Nothing is written unless every check passes; a failure
/// after staging began removes the staging directory and says so if it cannot.
pub fn export(req: &ExportRequest) -> Result<ExportSummary> {
    let staging = Staging::plan(&req.out)?;
    let src = SafeTensorsFile::open(&req.source)?;
    let plan = plan(req, &src)?;
    staging.publish(|dir| write_release(req, &src, &plan, dir))
}
