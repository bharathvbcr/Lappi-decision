//! The rule-3 door: held-out data and the two task-holdout families are never read by a
//! training process. **Removing or weakening any check in this module is a refused change**
//! (CLAUDE.md rule 3).
//!
//! A port of `python/qd_train/data_access.py` (`open_training_data` and its four layers) with
//! the manifest re-verification of `python/qd_data/manifest.py`. Each layer alone has a way to
//! be wrong, so all four run:
//!
//! 1. **Configured roots.** A path at or under any [`DataConfig::held_out_roots`] entry is
//!    refused, compared on *resolved absolute* paths ([`resolve`], a port of
//!    `Path.expanduser().resolve(strict=False)`), so `train/../heldout/x` and a symlink into
//!    the held-out tree are both caught. Relative roots are taken against `repo_root`.
//! 2. **Path markers.** Any path *segment* equal (case-folded) to `heldout` / `held_out` /
//!    `held-out` is refused wherever it sits. Segment-exact: `withheld_outputs` is not a
//!    holdout.
//! 3. **The manifest's own split.** A manifest moved out of the held-out tree still declares
//!    `split: "heldout"`, and only [`TRAINING_SPLITS`] are admitted.
//! 4. **The family holdout.** A manifest that says `train` is still refused if any entry
//!    names a held-out task family, or if the holdout it records differs from the config's.
//!
//! The manifest's `data_snapshot_hash` is re-derived from its contents (Python's
//! `canonical_json`, ported in [`crate::pyjson`]) and a mismatch is refused as tampering, and
//! a snapshot whose own pipeline status is `not_run` or `ran, passed=false` is refused the way
//! `open_training_data` refuses it.
//!
//! 5. **The content guard** ([`assert_no_caller_records`]). Layers 1-4 are by path or by what a
//!    manifest says, so a copy of a caller-record file placed anywhere passes them (the store's
//!    `heldout` segment is the whole path mechanism). This layer reads bytes: any line whose
//!    parsed `"record"` starts with `lappi.caller_record` refuses the whole file, whatever it is
//!    called and wherever it lives. It is bounded ([`GuardLimits`]) and fails closed: bytes it
//!    cannot scan (gzip, zstd, UTF-16) or could not finish scanning are refused as
//!    [`HeldOutLayer::GuardIncomplete`], never passed as clean. It narrows admission only.
//!
//!    **What calls it.** In this crate only [`open_training_manifest`], on the manifest bytes it
//!    reads (a shard set's `manifest.json` is never a row file, so that call is a backstop, not
//!    the data path). This crate reads token shards, not JSONL: pool and source rows reach shards
//!    through Python, where the same rule is `python/qd_data/caller_records.py`, called at every
//!    row-file open on that path, and through `qd-prep`, where it is
//!    `crates/qd-prep/src/heldout.rs`. The three implementations run one oracle,
//!    `fixtures/caller/guard-cases.json`. A Rust reader of JSONL rows added here must call
//!    [`assert_jsonl_file_not_caller_records`].
//!
//! The refusal is [`HeldOutViolation`]: a typed error that names its layer, never a value a
//! caller could mistake for an empty dataset.

use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::files::ShardFiles;
use crate::pyjson::{self, CANONICAL};
use crate::tristate::{TriState, TriStateError, parse_tristate};

/// `qd_data.config.SPLITS`.
pub const SPLITS: [&str; 3] = ["train", "val", "heldout"];
/// `qd_data.split.TRAINING_SPLITS`: the splits a training process may open.
pub const TRAINING_SPLITS: [&str; 2] = ["train", "val"];
/// `qd_data.config.N_HELD_OUT_FAMILIES`: the plan fixes the holdout at two families.
pub const N_HELD_OUT_FAMILIES: usize = 2;
/// `qd_data.config.DEFAULT_HELD_OUT_FAMILIES`.
pub const DEFAULT_HELD_OUT_FAMILIES: [&str; N_HELD_OUT_FAMILIES] =
    ["code.language_id", "qa.answerability"];
/// `DataConfig.held_out_roots`' default, relative to the run's root.
pub const DEFAULT_HELD_OUT_ROOT: &str = "data/heldout";
/// `DataConfig.held_out_path_markers`' default.
pub const DEFAULT_HELD_OUT_PATH_MARKERS: [&str; 3] = ["heldout", "held_out", "held-out"];
/// `DataConfig.seed`'s default.
pub const DEFAULT_SEED: u64 = 20_260_919;
/// `qd_data.manifest.MANIFEST_FORMAT_VERSION`.
pub const MANIFEST_FORMAT_VERSION: i64 = 1;
/// Ceiling on a manifest read. The v4 train manifest is 152,784,427 bytes; this bound is a
/// backstop against a mis-pointed path, not a statement about corpus size.
pub const MAX_MANIFEST_BYTES: u64 = 2 << 30;
/// Ceiling on path components processed while resolving one path (symlink expansion
/// included), so a pathological link chain cannot loop.
const MAX_RESOLVE_STEPS: usize = 4096;

/// Which layer of the door refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeldOutLayer {
    /// The configuration would disable a layer (no roots, no markers): the check disabled,
    /// not the check passing.
    CheckDisabled,
    /// Layer 1: at or under a configured held-out root.
    ConfiguredRoot,
    /// Layer 2: a path segment is a held-out marker.
    PathMarker,
    /// Layer 3: the manifest declares a split a training process may not open.
    ManifestSplit,
    /// Layer 4: an entry names a held-out task family.
    HeldOutFamily,
    /// Layer 4: the manifest records a different holdout than the config names.
    RecordedHoldout,
    /// The snapshot's own pipeline status is `not_run`.
    SnapshotNotRun,
    /// The snapshot's own pipeline ran and failed.
    SnapshotFailed,
    /// Rule 3 at the shard boundary: the shard header declares `split: "heldout"` or a split
    /// that is not trainable.
    ShardSplit,
    /// Layer 5: a line of the file is a `lappi.caller_record` (an app's record of what it saw
    /// and chose, which is never training data).
    CallerRecord,
    /// Layer 5: the content guard could not scan the whole file within its budget. A check
    /// that could not finish is a refusal, not a pass.
    GuardIncomplete,
}

/// A held-out refusal. Raised, never returned as data.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("rule 3 ({layer:?}): expected {expected}, got {actual}. {detail}")]
pub struct HeldOutViolation {
    /// Which layer refused.
    pub layer: HeldOutLayer,
    /// What a trainable input would have been.
    pub expected: String,
    /// What was found.
    pub actual: String,
    /// Why, and the rule.
    pub detail: String,
}

/// A `DataConfig` the door will not run under.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("data config refused: {0}")]
pub struct ConfigError(pub String);

/// Everything that can stop [`open_training_manifest`].
#[derive(Debug, Error)]
pub enum DoorError {
    /// A rule-3 refusal.
    #[error(transparent)]
    HeldOut(#[from] HeldOutViolation),
    /// The manifest is not a manifest this door can verify.
    #[error("{path}: {detail}")]
    Manifest {
        /// The manifest.
        path: PathBuf,
        /// What is wrong.
        detail: String,
    },
    /// The path could not be resolved or read.
    #[error("{path}: {source}")]
    Io {
        /// The path.
        path: PathBuf,
        /// The error.
        source: io::Error,
    },
}

impl From<TriStateError> for DoorError {
    fn from(e: TriStateError) -> Self {
        DoorError::Manifest {
            path: PathBuf::from(e.field.clone()),
            detail: e.detail,
        }
    }
}

/// The data config fields the door reads. Python's `qd_data.config.DataConfig`, restricted to
/// what rule 3 needs, and validated at construction.
///
/// **Not ported:** Python's check that each held-out family is a registered, otherwise
/// admitted task family (`qd_data.sources.TASK_FAMILIES` / `admitted_task_families`). That
/// needs the source registry and licence config, which this crate does not mirror; the
/// default families are asserted equal to Python's `DataConfig()` by the oracle fixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataConfig {
    held_out_families: Vec<String>,
    held_out_roots: Vec<PathBuf>,
    held_out_path_markers: Vec<String>,
    seed: u64,
}

impl DataConfig {
    /// A config, refused if any layer would be disabled by it.
    ///
    /// Stricter than Python in two places, both fail-closed: an empty marker list is refused
    /// (Python accepts one and layer 2 then matches nothing), and markers must be ASCII so
    /// that case folding them cannot depend on Unicode tables this crate does not carry.
    pub fn new(
        held_out_families: Vec<String>,
        held_out_roots: Vec<PathBuf>,
        held_out_path_markers: Vec<String>,
        seed: u64,
    ) -> Result<Self, ConfigError> {
        if held_out_families.len() != N_HELD_OUT_FAMILIES {
            return Err(ConfigError(format!(
                "the plan fixes the holdout at {N_HELD_OUT_FAMILIES} task families; got {}: \
                 {held_out_families:?}",
                held_out_families.len()
            )));
        }
        let unique: HashSet<&String> = held_out_families.iter().collect();
        if unique.len() != held_out_families.len() {
            return Err(ConfigError(format!(
                "held_out_families contains a duplicate: {held_out_families:?}"
            )));
        }
        if held_out_families.iter().any(|f| f.trim().is_empty()) {
            return Err(ConfigError("a held-out family id is empty".to_owned()));
        }
        if held_out_roots.is_empty() {
            return Err(ConfigError(
                "held_out_roots is empty: the training-time path check would then have nothing \
                 to refuse, which is the check silently disabled"
                    .to_owned(),
            ));
        }
        if held_out_roots.iter().any(|r| r.as_os_str().is_empty()) {
            return Err(ConfigError("a held-out root is the empty path".to_owned()));
        }
        if held_out_path_markers.is_empty() {
            return Err(ConfigError(
                "held_out_path_markers is empty: layer 2 would match nothing, which is the \
                 check disabled, not the check passing"
                    .to_owned(),
            ));
        }
        if let Some(bad) = held_out_path_markers
            .iter()
            .find(|m| m.is_empty() || !m.is_ascii() || m.contains('/'))
        {
            return Err(ConfigError(format!(
                "held-out path marker {bad:?} must be a non-empty ASCII path segment"
            )));
        }
        Ok(Self {
            held_out_families,
            held_out_roots,
            held_out_path_markers,
            seed,
        })
    }

    /// The two held-out task families.
    pub fn held_out_families(&self) -> &[String] {
        &self.held_out_families
    }

    /// Roots a training process must never read; relative ones are taken against the run root.
    pub fn held_out_roots(&self) -> &[PathBuf] {
        &self.held_out_roots
    }

    /// Path segments that mark held-out data wherever they sit.
    pub fn held_out_path_markers(&self) -> &[String] {
        &self.held_out_path_markers
    }

    /// The data seed (`DataConfig.seed`), which `real_ft_run` uses as the batch-order seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether `family_id` is held out.
    pub fn is_held_out_family(&self, family_id: &str) -> bool {
        self.held_out_families.iter().any(|f| f == family_id)
    }
}

impl Default for DataConfig {
    /// Python's `DataConfig()` defaults, pinned against the oracle's dump of them.
    fn default() -> Self {
        Self {
            held_out_families: DEFAULT_HELD_OUT_FAMILIES
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            held_out_roots: vec![PathBuf::from(DEFAULT_HELD_OUT_ROOT)],
            held_out_path_markers: DEFAULT_HELD_OUT_PATH_MARKERS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            seed: DEFAULT_SEED,
        }
    }
}

// --- layer 5: the content guard --------------------------------------------------------------

/// The `"record"` value every caller-record line carries (`qd_runtime::caller_record::RECORD`,
/// `docs/caller-contract.md`). The guard matches the prefix, so a later version is refused too.
pub const CALLER_RECORD_PREFIX: &str = "lappi.caller_record";
/// A line longer than this is skipped unparsed. The contract writes no record over 64 KiB, so
/// an over-long line is not one; 1 MiB leaves a wide margin.
pub const MAX_GUARD_LINE_BYTES: usize = 1 << 20;

/// The content guard's budget: bytes per line parsed, and bytes per file scanned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardLimits {
    /// A longer line is not parsed (see [`MAX_GUARD_LINE_BYTES`]).
    pub line_bytes: usize,
    /// A file longer than this is refused as unscanned ([`HeldOutLayer::GuardIncomplete`]).
    pub total_bytes: u64,
}

impl Default for GuardLimits {
    fn default() -> Self {
        Self {
            line_bytes: MAX_GUARD_LINE_BYTES,
            total_bytes: MAX_MANIFEST_BYTES,
        }
    }
}

/// The end (exclusive) of the JSON string token that starts at `bytes[start] == b'"'`. The text
/// is valid JSON, so an unescaped quote ends the token.
fn string_token_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    bytes.len()
}

/// The JSON string token `token` (quotes included) as text.
fn decode_string(token: &[u8]) -> Option<String> {
    serde_json::from_slice::<String>(token).ok()
}

/// Whether `json`, which is valid JSON, has a TOP-LEVEL `"record"` key (any copy of a
/// duplicated key) holding a string that starts with the prefix: in a top-level object, or in an
/// object that is a direct element of a top-level array. Nothing nested deeper is read, and a
/// string value that quotes the format is never a key. A parse into a map would keep only the
/// last copy of a duplicated key, hence the token walk.
fn top_level_record(json: &[u8]) -> bool {
    let mut stack: Vec<u8> = Vec::new();
    let mut expect_key = false;
    let mut want_value = false;
    let mut i = 0;
    while i < json.len() {
        let c = json[i];
        if matches!(c, b' ' | b'\t' | b'\r' | b'\n') {
            i += 1;
            continue;
        }
        let is_colon = c == b':';
        let mut end = i + 1;
        match c {
            b'{' => {
                stack.push(b'{');
                expect_key = true;
            }
            b'[' => {
                stack.push(b'[');
                expect_key = false;
            }
            b'}' | b']' => {
                stack.pop();
                expect_key = false;
            }
            b',' => expect_key = stack.last() == Some(&b'{'),
            b':' => expect_key = false,
            b'"' => {
                end = string_token_end(json, i);
                let token = &json[i..end];
                let at_target = matches!(stack.as_slice(), [b'{'] | [b'[', b'{']);
                if want_value {
                    if decode_string(token).is_some_and(|r| r.starts_with(CALLER_RECORD_PREFIX)) {
                        return true;
                    }
                } else if expect_key
                    && at_target
                    && decode_string(token).as_deref() == Some("record")
                {
                    want_value = true;
                    expect_key = false;
                    i = end;
                    continue;
                }
                expect_key = false;
            }
            _ => {
                // A number or literal: scan to its end.
                while end < json.len()
                    && !matches!(json[end], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
                {
                    end += 1;
                }
                expect_key = false;
            }
        }
        if want_value && !is_colon {
            want_value = false;
        }
        i = end;
    }
    false
}

/// A line that does not parse (a BOM, damage, invalid UTF-8) is a record only when it starts,
/// in column 0 after an optional UTF-8 BOM, with `{` or `[` and carries the key and the prefix
/// side by side: `"record"`, optional JSON whitespace, `:`, optional whitespace, `"lappi.caller_record`.
/// A line that merely contains both strings (source code, prose) is not one.
fn damaged_record_line(line: &[u8]) -> bool {
    const BOM: &[u8] = &[0xef, 0xbb, 0xbf];
    let head = line.strip_prefix(BOM).unwrap_or(line);
    if !matches!(head.first(), Some(b'{' | b'[')) {
        return false;
    }
    let ws = |b: &u8| matches!(b, b' ' | b'\t' | b'\r' | b'\n');
    let key = b"\"record\"";
    let prefix = CALLER_RECORD_PREFIX.as_bytes();
    line.windows(key.len()).enumerate().any(|(i, w)| {
        if w != key {
            return false;
        }
        let rest = &line[i + key.len()..];
        let rest = &rest[rest.iter().take_while(|b| ws(b)).count()..];
        let Some(rest) = rest.strip_prefix(b":") else {
            return false;
        };
        let rest = &rest[rest.iter().take_while(|b| ws(b)).count()..];
        rest.strip_prefix(b"\"")
            .is_some_and(|r| r.starts_with(prefix))
    })
}

/// One line (terminator already removed) is a caller record when it is JSONL whose top-level
/// `"record"` key (any copy of a duplicated key) starts with the prefix, or a JSON array holding
/// such an object; a line that parses and is not that is clean, however much text it quotes. A
/// line that does not parse is read by [`damaged_record_line`]. The budget is on the content, so
/// a record of exactly `line_bytes` is read.
fn is_caller_record_line(line: &[u8], line_bytes: usize) -> bool {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line.len() > line_bytes {
        return false;
    }
    match serde_json::from_slice::<serde_json::Value>(line) {
        Ok(_) => top_level_record(line),
        Err(_) => damaged_record_line(line),
    }
}

/// Finds a `str.splitlines()` separator the guard does not split on, in one line (a mirror of
/// `qd-prep`'s `heldout::BreakScan`, which this crate cannot depend on). A raw control
/// character (`\x0b`, `\x0c`, `\x1c`..`\x1e`) is refused anywhere: it is not legal inside a JSON
/// string either. U+0085, U+2028 and U+2029 are refused outside a string only: inside a string
/// value they are legal JSON and legitimate data, and the readers split on `\n` alone. A
/// backslash escapes the byte after it, in or out of a string.
#[derive(Default)]
struct BreakScan {
    in_string: bool,
    escaped: bool,
    tail: [u8; 2],
    seen: usize,
}

impl BreakScan {
    /// The separator `chunk` holds that the guard must refuse, if any.
    fn feed(&mut self, chunk: &[u8]) -> Option<&'static str> {
        for &b in chunk {
            if matches!(b, 0x0b | 0x0c | 0x1c..=0x1e) {
                return Some("a raw control character (U+000B, U+000C or U+001C..U+001E)");
            }
            if !self.in_string {
                let [p2, p1] = self.tail;
                if b == 0x85 && self.seen >= 1 && p1 == 0xC2 {
                    return Some("U+0085");
                }
                if matches!(b, 0xA8 | 0xA9) && self.seen >= 2 && p2 == 0xE2 && p1 == 0x80 {
                    return Some(if b == 0xA8 { "U+2028" } else { "U+2029" });
                }
            }
            if self.escaped {
                self.escaped = false;
            } else if b == b'\\' {
                self.escaped = true;
            } else if b == b'"' {
                self.in_string = !self.in_string;
            }
            self.tail = [self.tail[1], b];
            self.seen = self.seen.saturating_add(1);
        }
        None
    }
}

/// Layer 5 on bytes already read: refuse the whole file if any line is a caller record, or if
/// the file is longer than `limits.total_bytes` and the scanned prefix held none.
pub fn assert_no_caller_records(
    bytes: &[u8],
    label: &str,
    limits: GuardLimits,
) -> Result<(), HeldOutViolation> {
    if let Some(name) = unscannable_magic(bytes) {
        return Err(HeldOutViolation {
            layer: HeldOutLayer::GuardIncomplete,
            expected: "bytes the caller-record guard can read as lines of text".to_owned(),
            actual: format!("{label}: {name} bytes"),
            detail: "bytes that cannot be scanned for caller records are refused, not passed as \
                     clean (CLAUDE.md rule 3)."
                .to_owned(),
        });
    }
    let scanned_len =
        usize::try_from(limits.total_bytes).map_or(bytes.len(), |n| n.min(bytes.len()));
    let scanned = &bytes[..scanned_len];
    let last = scanned.iter().filter(|b| **b == b'\n').count();
    for (i, line) in scanned.split(|b| *b == b'\n').enumerate() {
        // The line without its CRLF's '\r' (a '\r' on the last line has no '\n' after it, so it
        // is a bare one); any other '\r' is a separator the text-mode readers honour and this
        // scan does not, so a record could sit behind it.
        let content = if i == last {
            line
        } else {
            line.strip_suffix(b"\r").unwrap_or(line)
        };
        if content.contains(&b'\r') {
            return Err(HeldOutViolation {
                layer: HeldOutLayer::GuardIncomplete,
                expected: "lines the caller-record guard can split exactly as the readers do"
                    .to_owned(),
                actual: format!("{label}: line {} holds a bare carriage return", i + 1),
                detail:
                    "the readers split on a bare '\\r' (text mode) or do not ('\\n' only), so a \
                         record could sit behind it; refused, not passed as clean (CLAUDE.md \
                         rule 3: a check that could not run is not a check that passed)."
                        .to_owned(),
            });
        }
        if let Some(what) = BreakScan::default().feed(line) {
            return Err(HeldOutViolation {
                layer: HeldOutLayer::GuardIncomplete,
                expected: "lines the caller-record guard can split exactly as the readers do"
                    .to_owned(),
                actual: format!("{label}: line {} holds {what}", i + 1),
                detail: "str.splitlines() splits on it and the guard does not, so a reader that \
                         did could read a record behind it; refused, not passed as clean \
                         (CLAUDE.md rule 3: a check that could not run is not a check that \
                         passed)."
                    .to_owned(),
            });
        }
        if is_caller_record_line(line, limits.line_bytes) {
            return Err(HeldOutViolation {
                layer: HeldOutLayer::CallerRecord,
                expected: "no line whose \"record\" starts with lappi.caller_record".to_owned(),
                actual: format!("{label}: line {} is a caller record", i + 1),
                detail: "caller records are never training data wherever the file lives \
                         (CLAUDE.md rule 3; docs/caller-contract.md). The whole file is \
                         refused. Removing this check is a refused change."
                    .to_owned(),
            });
        }
    }
    if scanned_len < bytes.len() {
        return Err(guard_incomplete(label, limits));
    }
    Ok(())
}

/// Compressed or UTF-16 bytes hide a record from a line scan.
fn unscannable_magic(bytes: &[u8]) -> Option<&'static str> {
    const MAGIC: [(&[u8], &str); 4] = [
        (&[0x1f, 0x8b], "gzip"),
        (&[0x28, 0xb5, 0x2f, 0xfd], "zstd"),
        (&[0xff, 0xfe], "UTF-16 (little-endian BOM)"),
        (&[0xfe, 0xff], "UTF-16 (big-endian BOM)"),
    ];
    MAGIC
        .iter()
        .find(|(m, _)| bytes.starts_with(m))
        .map(|(_, name)| *name)
}

fn guard_incomplete(label: &str, limits: GuardLimits) -> HeldOutViolation {
    HeldOutViolation {
        layer: HeldOutLayer::GuardIncomplete,
        expected: format!(
            "a file the caller-record guard can scan within {} bytes",
            limits.total_bytes
        ),
        actual: format!("{label}: longer than the budget"),
        detail: "a file the content guard could not finish scanning is refused, not passed as \
                 clean (CLAUDE.md: a check that could not run is not a check that passed)."
            .to_owned(),
    }
}

/// Layer 5 for a file on disk: read through `files` (bounded by `limits.total_bytes`) and scan.
pub fn assert_jsonl_file_not_caller_records(
    path: &Path,
    files: &dyn ShardFiles,
    limits: GuardLimits,
) -> Result<(), DoorError> {
    let label = path.display().to_string();
    // Decide by stat, never by open: opening a FIFO for reading blocks with no writer. A path
    // that is missing or a directory has no bytes and is left to the read, which names it.
    if let Ok(meta) = std::fs::metadata(path)
        && !meta.is_dir()
        && !meta.is_file()
    {
        return Err(HeldOutViolation {
            layer: HeldOutLayer::GuardIncomplete,
            expected: "a regular file the caller-record guard can scan".to_owned(),
            actual: format!("{label}: not a regular file"),
            detail: "a FIFO, device or socket has no stable bytes to scan, so it is refused, \
                     not passed as clean (CLAUDE.md rule 3: a check that could not run is not \
                     a check that passed)."
                .to_owned(),
        }
        .into());
    }
    let bytes = match files.read(path, limits.total_bytes) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::InvalidData => {
            return Err(guard_incomplete(&label, limits).into());
        }
        Err(source) => {
            return Err(DoorError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    assert_no_caller_records(&bytes, &label, limits).map_err(DoorError::from)
}

// --- layers 1 and 2: the path -----------------------------------------------------------------

/// `Path(path).expanduser().resolve(strict=False)`: absolute, symlinks resolved where the
/// path exists, and the not-yet-existing remainder appended as written.
///
/// A port of CPython's `posixpath.realpath(strict=False)`: components are consumed left to
/// right; `..` pops the resolved prefix; an existing symlink is replaced by its target's
/// components (an absolute target resets to `/`); a component that cannot be `lstat`ed is
/// appended unresolved; a symlink loop leaves the looping link unresolved. `strict=False`
/// matters: the check must fire on the *intent* to read a held-out path, not only on a path
/// that exists.
pub fn resolve(path: &Path) -> io::Result<PathBuf> {
    let expanded = expand_user(path)?;
    let mut current = if expanded.is_absolute() {
        PathBuf::from("/")
    } else {
        std::env::current_dir()?
    };
    // Unprocessed parts, last-in-first-out. `Resolved(link)` marks the point where a symlink's
    // target has been fully consumed, so its resolution can be cached (and loops detected).
    enum Part {
        Name(OsString),
        Resolved(PathBuf),
    }
    let mut rest: Vec<Part> = Vec::new();
    push_components(&mut rest, &expanded, Part::Name);
    let mut seen: HashMap<PathBuf, Option<PathBuf>> = HashMap::new();
    let mut steps = 0usize;
    while let Some(part) = rest.pop() {
        steps += 1;
        if steps > MAX_RESOLVE_STEPS {
            return Err(io::Error::other(format!(
                "resolving {} took more than {MAX_RESOLVE_STEPS} steps",
                path.display()
            )));
        }
        let name = match part {
            Part::Resolved(link) => {
                seen.insert(link, Some(current.clone()));
                continue;
            }
            Part::Name(name) => name,
        };
        if name.is_empty() || name == OsStr::new(".") {
            continue;
        }
        if name == OsStr::new("..") {
            current.pop();
            continue;
        }
        let candidate = current.join(&name);
        let is_link = match std::fs::symlink_metadata(&candidate) {
            Ok(meta) => meta.file_type().is_symlink(),
            Err(_) => false,
        };
        if !is_link {
            current = candidate;
            continue;
        }
        if let Some(cached) = seen.get(&candidate) {
            // A resolved link is reused; an unresolved one is a loop, left as written.
            current = cached.clone().unwrap_or(candidate);
            continue;
        }
        let target = match std::fs::read_link(&candidate) {
            Ok(t) => t,
            Err(_) => {
                current = candidate;
                continue;
            }
        };
        if target.is_absolute() {
            current = PathBuf::from("/");
        }
        seen.insert(candidate.clone(), None);
        rest.push(Part::Resolved(candidate));
        push_components(&mut rest, &target, Part::Name);
    }
    Ok(current)
}

/// Push `path`'s components onto a LIFO stack so they pop in order. Root and prefix
/// components are dropped (the caller handles absoluteness); `.` is dropped; `..` is kept.
fn push_components<T>(stack: &mut Vec<T>, path: &Path, make: impl Fn(OsString) -> T) {
    let parts: Vec<OsString> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(n) => Some(n.to_os_string()),
            Component::ParentDir => Some(OsString::from("..")),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => None,
        })
        .collect();
    for p in parts.into_iter().rev() {
        stack.push(make(p));
    }
}

/// `Path.expanduser`: a leading `~` is `$HOME`. `~user` is refused rather than looked up; no
/// path in this repository's runs is spelled that way, and a guess would be a silent wrong root.
fn expand_user(path: &Path) -> io::Result<PathBuf> {
    let mut comps = path.components();
    match comps.next() {
        Some(Component::Normal(first)) if first == OsStr::new("~") => {
            let home = std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .ok_or_else(|| {
                    io::Error::other("Could not determine home directory (HOME is unset)")
                })?;
            Ok(PathBuf::from(home).join(comps.as_path()))
        }
        Some(Component::Normal(first)) if first.to_string_lossy().starts_with('~') => {
            Err(io::Error::other(format!(
                "{}: '~user' expansion is not supported; spell the path out",
                path.display()
            )))
        }
        _ => Ok(path.to_path_buf()),
    }
}

/// Python's `str.casefold(segment) == marker` for an ASCII `marker`.
///
/// Folding the marker is plain ASCII lowercasing. Folding the segment must catch the
/// non-ASCII code points whose full case folding is ASCII -- KELVIN SIGN to `k`, LONG S to
/// `s`, sharp s to `ss`, the Latin `ff`/`fi`/`fl`/`ffi`/`ffl`/`st` ligatures -- because Python
/// folds those to ASCII and so would match a marker spelled with them. Every other non-ASCII
/// code point folds to something non-ASCII and cannot equal an ASCII marker. [I] This list
/// is from the Unicode CaseFolding table as recalled, not read; the default markers contain
/// none of k, s, f.
fn casefold_equals_ascii(segment: &str, marker: &str) -> bool {
    let mut folded = String::with_capacity(segment.len());
    for ch in segment.chars() {
        match ch {
            c if c.is_ascii() => folded.push(c.to_ascii_lowercase()),
            '\u{212a}' => folded.push('k'),
            '\u{17f}' => folded.push('s'),
            '\u{df}' | '\u{1e9e}' => folded.push_str("ss"),
            '\u{fb00}' => folded.push_str("ff"),
            '\u{fb01}' => folded.push_str("fi"),
            '\u{fb02}' => folded.push_str("fl"),
            '\u{fb03}' => folded.push_str("ffi"),
            '\u{fb04}' => folded.push_str("ffl"),
            '\u{fb05}' | '\u{fb06}' => folded.push_str("st"),
            _ => return false,
        }
    }
    folded == marker.to_ascii_lowercase()
}

/// Layers 1 and 2 (`data_access.assert_path_not_held_out`). `repo_root` anchors relative
/// held-out roots; it is required because a check that silently skipped a relative root
/// would be no check at all.
pub fn assert_path_not_held_out(
    path: &Path,
    config: &DataConfig,
    repo_root: &Path,
) -> Result<(), DoorError> {
    let io_err = |p: &Path| {
        let p = p.to_path_buf();
        move |source| DoorError::Io { path: p, source }
    };
    let resolved = resolve(path).map_err(io_err(path))?;
    let root_abs = resolve(repo_root).map_err(io_err(repo_root))?;
    // Both forms are checked: a symlink can hide a marker the given path spells out
    // (`heldout/ -> clean/`), and the given path can hide the target resolution finds.
    let given = lexical_absolute(path).map_err(io_err(path))?;
    // The path exactly as spelled (user-expanded, nothing normalised): `x/heldout/../train`
    // names a marker segment that `lexical_absolute` pops away. Python checks the same third
    // form (`Path(path).expanduser().parts`), so the two doors agree on every spelling
    // (`python/tests/data/held_out_given_path_cases.json`).
    let raw = expand_user(path).map_err(io_err(path))?;
    let candidates = [&resolved, &given];
    let marker_candidates = [&resolved, &given, &raw];

    for root in config.held_out_roots() {
        let joined = if root.is_absolute() {
            root.clone()
        } else {
            root_abs.join(root)
        };
        let absolute_root = resolve(&joined).map_err(io_err(&joined))?;
        for candidate in candidates {
            if candidate.starts_with(&absolute_root) {
                return Err(HeldOutViolation {
                    layer: HeldOutLayer::ConfiguredRoot,
                    expected: format!(
                        "a path outside the held-out root {}",
                        absolute_root.display()
                    ),
                    actual: candidate.display().to_string(),
                    detail: "CLAUDE.md rule 3: held-out data is never read by a training \
                             process. Removing this check is a refused change."
                        .to_owned(),
                }
                .into());
            }
        }
    }

    for candidate in marker_candidates {
        for component in candidate.components() {
            let Component::Normal(segment) = component else {
                continue;
            };
            let segment = segment.to_string_lossy();
            if let Some(marker) = config
                .held_out_path_markers()
                .iter()
                .find(|m| casefold_equals_ascii(&segment, m))
            {
                return Err(HeldOutViolation {
                    layer: HeldOutLayer::PathMarker,
                    expected: format!("no path segment in {:?}", config.held_out_path_markers()),
                    actual: candidate.display().to_string(),
                    detail: format!(
                        "path segment {segment:?} matches marker {marker:?}: it marks this as \
                         held-out data wherever it sits. CLAUDE.md rule 3."
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// `os.path.abspath(Path(path).expanduser())`: absolute and `..`-normalised lexically, following
/// no symlink. The given path as the caller spelled it, for the half of rule 3 that resolution
/// erases.
fn lexical_absolute(path: &Path) -> io::Result<PathBuf> {
    let expanded = expand_user(path)?;
    let mut out = if expanded.is_absolute() {
        PathBuf::from("/")
    } else {
        std::env::current_dir()?
    };
    for component in expanded.components() {
        match component {
            Component::Normal(n) => out.push(n),
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    Ok(out)
}

// --- layers 3 and 4: the manifest -------------------------------------------------------------

#[derive(Deserialize)]
struct RawEntry {
    row_id: String,
    content_hash: String,
    split: String,
    source_id: String,
    #[allow(dead_code)]
    host: String,
    family_id: String,
    #[allow(dead_code)]
    repo_key: String,
    #[allow(dead_code)]
    identity_key: String,
    licence_id: String,
    #[serde(default)]
    #[allow(dead_code)]
    obligations: Vec<String>,
}

#[derive(Deserialize)]
struct RawManifest {
    #[serde(default)]
    manifest_format_version: Value,
    split: String,
    #[serde(default)]
    data_snapshot_hash: Value,
    #[serde(default)]
    entries: Vec<RawEntry>,
    #[serde(default)]
    config_fingerprint: Map<String, Value>,
    #[serde(default)]
    admitted_source_ids: Vec<String>,
    #[serde(default)]
    refused_sources: HashMap<String, Vec<String>>,
    #[serde(default)]
    held_out_families: Vec<String>,
    #[serde(default)]
    status: Value,
}

/// A manifest the door admitted, and the checks that cleared it (`TrainingData`).
#[derive(Clone, Debug)]
pub struct TrainingManifest {
    /// Where it was read from, as given.
    pub path: PathBuf,
    /// Its declared split (one of [`TRAINING_SPLITS`]).
    pub split: String,
    /// The re-derived `data_snapshot_hash`, equal to the one recorded in the file.
    pub data_snapshot_hash: String,
    /// Every row id the manifest admits.
    pub row_ids: HashSet<String>,
    /// The checks, by name, as `open_training_data` records them.
    pub checks: Vec<(String, TriState)>,
}

impl TrainingManifest {
    /// Rows in the manifest.
    pub fn n_rows(&self) -> usize {
        self.row_ids.len()
    }
}

/// `Manifest.hashed_body()` re-derived from the parsed file, then `canonical_json` + sha256.
fn snapshot_hash(raw: &RawManifest) -> Result<String, pyjson::PyJsonError> {
    let mut admitted = raw.admitted_source_ids.clone();
    admitted.sort();
    let mut refused = Map::new();
    let mut refused_keys: Vec<&String> = raw.refused_sources.keys().collect();
    refused_keys.sort();
    for k in refused_keys {
        let mut reasons = raw.refused_sources[k].clone();
        reasons.sort();
        refused.insert(k.clone(), Value::from(reasons));
    }
    let mut held = raw.held_out_families.clone();
    held.sort();
    let mut bodies: Vec<[&str; 5]> = raw
        .entries
        .iter()
        .map(|e| {
            [
                &e.content_hash,
                &e.split,
                &e.source_id,
                &e.family_id,
                &e.licence_id,
            ]
        })
        .map(|[c, s, src, f, l]| [c.as_str(), s.as_str(), src.as_str(), f.as_str(), l.as_str()])
        .collect();
    // sorted(..., key=(content_hash, family_id, source_id)); stable, as Python's sort is.
    bodies.sort_by(|a, b| (a[0], a[3], a[2]).cmp(&(b[0], b[3], b[2])));
    let entries: Vec<Value> = bodies
        .into_iter()
        .map(|[c, s, src, f, l]| {
            serde_json::json!({
                "content_hash": c, "split": s, "source_id": src, "family_id": f, "licence_id": l,
            })
        })
        .collect();
    let body = serde_json::json!({
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "split": raw.split,
        "config_fingerprint": Value::Object(raw.config_fingerprint.clone()),
        "admitted_source_ids": admitted,
        "refused_sources": Value::Object(refused),
        "held_out_families": held,
        "entries": entries,
    });
    let text = pyjson::dumps(&body, CANONICAL)?;
    Ok(hex(&Sha256::digest(text.as_bytes())))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Every refusal `Manifest.read` + `Manifest.__post_init__` make, then layers 3 and 4.
fn verify_manifest(
    raw: &RawManifest,
    path: &Path,
    config: &DataConfig,
) -> Result<String, DoorError> {
    let bad = |detail: String| DoorError::Manifest {
        path: path.to_path_buf(),
        detail,
    };
    if raw.manifest_format_version.as_i64() != Some(MANIFEST_FORMAT_VERSION) {
        return Err(bad(format!(
            "manifest_format_version {} != {MANIFEST_FORMAT_VERSION}. Guessing \
             forward-compatibly is how a field changes meaning silently.",
            raw.manifest_format_version
        )));
    }
    if !SPLITS.contains(&raw.split.as_str()) {
        return Err(bad(format!(
            "unknown split {:?}; known: {SPLITS:?}",
            raw.split
        )));
    }
    let wrong: Vec<&str> = raw
        .entries
        .iter()
        .filter(|e| e.split != raw.split)
        .map(|e| e.row_id.as_str())
        .collect();
    if !wrong.is_empty() {
        return Err(bad(format!(
            "manifest for split {:?} carries {} entries from another split, first five: {:?}. \
             A manifest that misreports its own split is how a held-out shard gets opened by a \
             trainer.",
            raw.split,
            wrong.len(),
            &wrong[..wrong.len().min(5)]
        )));
    }
    let mut ids = HashSet::with_capacity(raw.entries.len());
    if raw.entries.iter().any(|e| !ids.insert(e.row_id.as_str())) {
        return Err(bad(format!(
            "manifest for split {:?} has duplicate row ids",
            raw.split
        )));
    }
    let derived = snapshot_hash(raw).map_err(|e| bad(e.to_string()))?;
    if raw.data_snapshot_hash.as_str() != Some(derived.as_str()) {
        return Err(bad(format!(
            "recorded data_snapshot_hash {} does not match the hash re-derived from the file's \
             contents ({derived:?}). The file has been edited, or was written by a different \
             manifest format.",
            raw.data_snapshot_hash
        )));
    }

    // Layer 3: the file's own declaration.
    if !TRAINING_SPLITS.contains(&raw.split.as_str()) {
        return Err(HeldOutViolation {
            layer: HeldOutLayer::ManifestSplit,
            expected: format!("a manifest whose split is one of {TRAINING_SPLITS:?}"),
            actual: raw.split.clone(),
            detail: format!(
                "{}: the manifest declares itself split {:?}. Moving or renaming the file does \
                 not change what it says it is. CLAUDE.md rule 3.",
                path.display(),
                raw.split
            ),
        }
        .into());
    }
    // Layer 4: its families.
    let mut offending: Vec<&str> = raw
        .entries
        .iter()
        .filter(|e| config.is_held_out_family(&e.family_id))
        .map(|e| e.family_id.as_str())
        .collect();
    offending.sort_unstable();
    offending.dedup();
    if !offending.is_empty() {
        return Err(HeldOutViolation {
            layer: HeldOutLayer::HeldOutFamily,
            expected: format!(
                "no entry from a held-out task family {:?}",
                sorted(config.held_out_families())
            ),
            actual: format!("{offending:?}"),
            detail: format!(
                "{}: a manifest declaring split {:?} carries {} held-out task family \
                 identifier(s). The shard was built with a different holdout than this config \
                 names. CLAUDE.md rule 3.",
                path.display(),
                raw.split,
                offending.len()
            ),
        }
        .into());
    }
    let recorded = sorted(&raw.held_out_families);
    let expected = sorted(config.held_out_families());
    if !recorded.is_empty() && recorded != expected {
        return Err(HeldOutViolation {
            layer: HeldOutLayer::RecordedHoldout,
            expected: format!("a shard built with held-out families {expected:?}"),
            actual: format!("{recorded:?}"),
            detail: format!(
                "{}: the shard records a different holdout than this run configures. Rule 2: an \
                 agent may report that a gate failed; it may not shrink a held-out set to make \
                 one pass.",
                path.display()
            ),
        }
        .into());
    }
    Ok(derived)
}

fn sorted(items: &[String]) -> Vec<String> {
    let mut v = items.to_vec();
    v.sort();
    v
}

/// `data_access.open_training_data`: the one way a training process opens a manifest.
///
/// Layers 1-2 run on the path **before the file is opened**; layers 3-4 and the snapshot
/// status run on its contents. `allow_not_run_snapshot` admits a snapshot whose own status is
/// `not_run`, deliberately; nothing admits one that ran and failed.
pub fn open_training_manifest(
    path: &Path,
    config: &DataConfig,
    repo_root: &Path,
    allow_not_run_snapshot: bool,
    files: &dyn ShardFiles,
) -> Result<TrainingManifest, DoorError> {
    assert_path_not_held_out(path, config, repo_root)?;
    let bytes = files
        .read(path, MAX_MANIFEST_BYTES)
        .map_err(|source| DoorError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    assert_no_caller_records(&bytes, &path.display().to_string(), GuardLimits::default())?;
    let raw: RawManifest = serde_json::from_slice(&bytes).map_err(|e| DoorError::Manifest {
        path: path.to_path_buf(),
        detail: format!("not a manifest this door can read: {e}"),
    })?;
    drop(bytes);
    let data_snapshot_hash = verify_manifest(&raw, path, config)?;
    let status = parse_tristate(&raw.status, &format!("{}:status", path.display()))?;
    match &status {
        TriState::NotRun { reason } if !allow_not_run_snapshot => {
            return Err(HeldOutViolation {
                layer: HeldOutLayer::SnapshotNotRun,
                expected: "a data snapshot whose pipeline status is 'ran'".to_owned(),
                actual: "not_run".to_owned(),
                detail: format!(
                    "{}: {reason}. Training on an unverified snapshot and then reporting a \
                     clean gate is the exact failure the tri-state exists to prevent. Pass \
                     allow_not_run_snapshot to proceed deliberately; the reason stays in the \
                     ledger row.",
                    path.display()
                ),
            }
            .into());
        }
        TriState::Ran {
            passed: false,
            detail,
            ..
        } => {
            let mut why = format!(
                "{}: the snapshot's own pipeline ran and failed",
                path.display()
            );
            if !detail.is_empty() {
                why.push_str(&format!(" -- {detail}"));
            }
            return Err(HeldOutViolation {
                layer: HeldOutLayer::SnapshotFailed,
                expected: "a data snapshot whose pipeline status is 'ran' and passed".to_owned(),
                actual: "ran, passed=false".to_owned(),
                detail: format!(
                    "{why}. Rebuild the snapshot or fix the gate it failed; there is no flag \
                     that admits it, and allow_not_run_snapshot does not."
                ),
            }
            .into());
        }
        _ => {}
    }
    let n = raw.entries.len() as u64;
    let row_ids: HashSet<String> = raw.entries.into_iter().map(|e| e.row_id).collect();
    Ok(TrainingManifest {
        path: path.to_path_buf(),
        split: raw.split.clone(),
        data_snapshot_hash: data_snapshot_hash.clone(),
        row_ids,
        checks: vec![
            (
                "path_not_held_out".to_owned(),
                TriState::passed(
                    path.display().to_string(),
                    "no configured held-out root and no held-out path marker matched",
                ),
            ),
            (
                "manifest_split_trainable".to_owned(),
                TriState::passed(raw.split, ""),
            ),
            (
                "no_held_out_families".to_owned(),
                TriState::Ran {
                    passed: true,
                    value: Value::Null,
                    coverage: Some(crate::tristate::Coverage { n, n_total: n }),
                    detail: format!(
                        "held-out families: {:?}",
                        sorted(config.held_out_families())
                    ),
                },
            ),
            ("snapshot_status".to_owned(), status),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_are_segment_exact_and_case_folded() {
        assert!(casefold_equals_ascii("HeldOut", "heldout"));
        assert!(casefold_equals_ascii("HELD-OUT", "held-out"));
        assert!(!casefold_equals_ascii("withheld_outputs", "held_out"));
        assert!(!casefold_equals_ascii("heldout2", "heldout"));
        assert!(casefold_equals_ascii("\u{fb06}x", "stx"));
    }

    #[test]
    fn config_refuses_a_disabled_layer() {
        let fams = || vec!["a".to_owned(), "b".to_owned()];
        assert!(DataConfig::new(fams(), vec![], vec!["heldout".into()], 1).is_err());
        assert!(DataConfig::new(fams(), vec!["r".into()], vec![], 1).is_err());
        assert!(DataConfig::new(vec!["a".into()], vec!["r".into()], vec!["m".into()], 1).is_err());
        assert!(
            DataConfig::new(
                vec!["a".into(), "a".into()],
                vec!["r".into()],
                vec!["m".into()],
                1
            )
            .is_err()
        );
        assert!(DataConfig::new(fams(), vec!["r".into()], vec!["h\u{e9}ld".into()], 1).is_err());
        assert!(DataConfig::new(fams(), vec!["r".into()], vec!["m".into()], 1).is_ok());
    }

    /// The content door's inputs (`qd_prep::posting_door::DEFAULT_SIGNATURES_PATH`: Vani's held-out
    /// posting keys and MinHash signatures) sit under the held-out root, so the training process's
    /// path door refuses them whichever way the path is spelled. `qd-train` does not depend on
    /// `qd-prep`, so the path is spelled out here and the two are held together by this test and
    /// `qd-prep`'s own marker test.
    #[test]
    fn the_vani_posting_signatures_are_behind_the_path_door() {
        let repo = Path::new("/definitely-missing-qd-repo");
        let config = DataConfig::default();
        for spelling in [
            "data/heldout/vani-postings/signatures.json",
            "/definitely-missing-qd-repo/data/heldout/vani-postings/signatures.json",
            "/definitely-missing-qd-repo/data/HeldOut/vani-postings/signatures.json",
            "/definitely-missing-qd-repo/other/heldout/../heldout/vani-postings/signatures.json",
        ] {
            let got = assert_path_not_held_out(Path::new(spelling), &config, repo);
            assert!(
                matches!(got, Err(DoorError::HeldOut(_))),
                "{spelling}: the training door must refuse it, got {got:?}"
            );
        }
    }

    #[test]
    fn resolve_normalises_dotdot_and_keeps_a_missing_tail() {
        let base = resolve(Path::new("/")).unwrap();
        assert_eq!(base, PathBuf::from("/"));
        let p = resolve(Path::new("/definitely-missing-qd/a/../b/./c")).unwrap();
        assert_eq!(p, PathBuf::from("/definitely-missing-qd/b/c"));
        assert_eq!(resolve(Path::new("/..")).unwrap(), PathBuf::from("/"));
    }
}
