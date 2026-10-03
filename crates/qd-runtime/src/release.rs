//! The release directory `qd-export` writes, opened and bound before anything serves from it.
//!
//! `crates/qd-export/src/export.rs` is the writer. This module is the one reader, and it owns the
//! file names and the manifest format both sides use: `qd-export` imports them from here, so the
//! writer and the reader cannot spell a file two ways.
//!
//! # What is bound, and where it is checked
//!
//! `release_manifest.json`'s `expected_identity` is the release's one statement of what goes
//! together: the tower (`weight_hash`), the `config.json` it is served under (`config_sha256`),
//! the tokenizer (`tokenizer_hash`) and the calibration table (`calibration_hash`). Two halves
//! check it:
//!
//! * [`Release::open`] reads `config.json` and refuses unless its sha256 is the bound one **and**
//!   the one `files` records. A same-shaped config from another revision (a different
//!   `rope_theta`, say) beside the same tower is refused here, before a backend reads it.
//!   It reads `calibration.json` the same way, parses it as the runtime's own
//!   [`CalibrationTable`], and refuses unless the table's [`CalibrationTable::hash`] is the one
//!   the manifest binds. A release with no table is refused: there is no fallback to
//!   [`CalibrationTable::reference`], whose numbers were fitted on nothing.
//! * [`Release::check_backend`] refuses a backend whose reported `weight_hash`, `tokenizer_hash`
//!   or `calibration_hash` is not the bound one. The weight hash is computed by the loader from
//!   the tensors it actually read, so this is the tower half of the binding.
//!
//! Nothing here parses `config.json` as a model config: which layouts a kernel runs is the
//! backend's question. The runtime binds bytes.
//!
//! # What the release was trained on
//!
//! `trained_families` is the sorted, unique list of task family ids in the train split's
//! manifest the tower was trained from (`qd-export --train-manifest` writes it). A request's
//! `task` is its family id (`python/qd_data/mixture.py::_request`, `task=family_id`), so
//! [`crate::admission`] refuses a task outside the list as
//! [`crate::refusal::Refusal::TaskNotTrained`] (Fable's pipeline ruling, item 4). A manifest
//! without the field (or with `null`) opens, and admits no task: a release that cannot say what
//! it trained cannot admit anything. A recorded list that is empty, unsorted, repeated, over
//! [`MAX_TRAINED_FAMILIES`], or holds an id no request could name is refused at open
//! ([`validate_trained_families`]), so an empty `available` in a refusal always means "not
//! recorded". An [`Ensemble`]'s members must record the same list, or record none alike.
//!
//! # An ensemble of towers
//!
//! [`Ensemble::open`] reads `ensemble_manifest.json` (`qd-ensemble.v1`, written by
//! `qd-export-ensemble`): N member directories, each a full release opened as a [`Tower`] (so
//! each member's `config.json` is bound to its own tower), each checked against the manifest's
//! record of it (`release_manifest.json` and `model.safetensors` sha256, weight, config and
//! tokenizer hashes), and all N required to agree on config, tokenizer, trained width and
//! trained families and to be N different towers. The ensemble's calibration table is read exactly as a release's. One
//! failing member refuses the whole ensemble; N-1 towers are never served as N.
//!
//! # What this does not close
//!
//! The backend opens `config.json` by path after [`Release::open`] hashed it. A file swapped
//! between the two reads is not caught here; closing that needs the backend to take the verified
//! bytes, which is a change to the backend's constructor.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::backend::BackendIdentity;
use crate::calibration::CalibrationTable;
use crate::render::RenderCaps;

/// The manifest's file name in a release directory.
pub const MANIFEST_FILE: &str = "release_manifest.json";
/// The manifest's `format`. A manifest that says anything else is refused, not interpreted.
pub const MANIFEST_FORMAT: &str = "qd-release.v1";
/// The model config the backend reads (`qd-metal/src/config.rs`).
pub const CONFIG_FILE: &str = "config.json";
/// The tower's tensors (`qd-metal/src/model.rs`).
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// The tokenizer the backend hashes into its identity (`qd-metal/src/tokenizer.rs`).
pub const TOKENIZER_FILE: &str = "tokenizer.json";
/// The span pointer head, F32 (`GAP-J7-EXPORT-SPAN-HEAD-UNSERVED`).
pub const SPAN_HEAD_FILE: &str = "span_head.safetensors";
/// The calibration table, as `CalibrationTable` JSON.
pub const CALIBRATION_FILE: &str = "calibration.json";

/// Bound on every release file read whole: the manifest, the config, the tokenizer files and the
/// calibration table. The real `tokenizer.json` is 12,807,196 bytes (`b1485b2f`, measured
/// 2026-10-01). One constant for the writer and the reader, so a file the exporter accepted is
/// never one the runtime refuses for its size.
pub const MAX_SMALL_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// The manifest key naming the task families the tower was trained on. One spelling for the
/// writer (`qd-export`) and the reader.
pub const TRAINED_FAMILIES_KEY: &str = "trained_families";
/// Bound on a recorded `trained_families`. The plan's data has about a dozen families; every
/// `task_not_trained` refusal echoes the whole list, so the list is bounded like any payload.
pub const MAX_TRAINED_FAMILIES: usize = 1024;

/// Refuse a `trained_families` list a release must never carry, with what is wrong.
///
/// The reader runs it on every manifest that records the field and the writer on every list it
/// writes, so a release the exporter wrote is never one the runtime refuses for its families. A
/// family id is held to what a request's `task` can be: non-blank, no surrounding whitespace
/// (a task is matched byte for byte), and within [`RenderCaps::DEFAULT`]'s task cap. The list is
/// strictly ascending, which is sorted and unique in one check.
pub fn validate_trained_families(families: &[String]) -> Result<(), String> {
    let cap = RenderCaps::DEFAULT.max_task_bytes;
    if families.len() > MAX_TRAINED_FAMILIES {
        return Err(format!(
            "{TRAINED_FAMILIES_KEY} lists {} families, over the bound of {MAX_TRAINED_FAMILIES}; \
             every task_not_trained refusal names them all",
            families.len()
        ));
    }
    if families.is_empty() {
        return Err(format!(
            "{TRAINED_FAMILIES_KEY} is empty: a release trained on no task family is not a \
             release. A release that cannot say what it trained omits the field, and admits no \
             task"
        ));
    }
    for (i, family) in families.iter().enumerate() {
        if crate::is_blank(family) {
            return Err(format!("{TRAINED_FAMILIES_KEY}[{i}] is blank"));
        }
        if family.len() > cap {
            return Err(format!(
                "{TRAINED_FAMILIES_KEY}[{i}] is {} bytes, over the {cap}-byte task cap; no \
                 request could name it",
                family.len()
            ));
        }
        if family.trim() != family {
            return Err(format!(
                "{TRAINED_FAMILIES_KEY}[{i}] {family:?} has leading or trailing whitespace; a \
                 request's task is matched byte for byte"
            ));
        }
    }
    if let Some(i) = families.windows(2).position(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "{TRAINED_FAMILIES_KEY} is not in strictly ascending order at index {} ({:?} then \
             {:?}); the writer writes it sorted and unique",
            i + 1,
            families[i],
            families[i + 1]
        ));
    }
    Ok(())
}

/// Why a release was refused at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseRefusalKind {
    /// `release_manifest.json` is missing, unreadable, over the bound, not JSON, of another
    /// format, or missing a field this reader requires.
    Manifest,
    /// `config.json` is not the config the manifest binds to the tower.
    ConfigMismatch,
    /// The release carries no calibration table.
    CalibrationMissing,
    /// `calibration.json` is not a usable `CalibrationTable`.
    CalibrationInvalid,
    /// `calibration.json`'s bytes or its table hash are not the ones the manifest binds.
    CalibrationMismatch,
    /// The backend reports a weight, tokenizer or calibration hash the manifest does not bind.
    IdentityMismatch,
    /// An ensemble member is not the tower the ensemble manifest names.
    MemberMismatch,
    /// Ensemble members disagree on config, tokenizer or trained width, or two are one tower.
    MembersDisagree,
    /// A release file could not be read.
    Io,
}

impl ReleaseRefusalKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ReleaseRefusalKind::Manifest => "manifest",
            ReleaseRefusalKind::ConfigMismatch => "config_mismatch",
            ReleaseRefusalKind::CalibrationMissing => "calibration_missing",
            ReleaseRefusalKind::CalibrationInvalid => "calibration_invalid",
            ReleaseRefusalKind::CalibrationMismatch => "calibration_mismatch",
            ReleaseRefusalKind::IdentityMismatch => "identity_mismatch",
            ReleaseRefusalKind::MemberMismatch => "member_mismatch",
            ReleaseRefusalKind::MembersDisagree => "members_disagree",
            ReleaseRefusalKind::Io => "io",
        }
    }
}

/// A release that will not be served, with the check that failed and the values it compared.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("release refused ({}): {detail}", kind.as_str())]
pub struct ReleaseRefusal {
    pub kind: ReleaseRefusalKind,
    pub detail: String,
}

impl ReleaseRefusal {
    fn new(kind: ReleaseRefusalKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

/// One tower's release directory: its manifest parsed and its `config.json` the bound one.
///
/// This is the half of a [`Release`] that does not involve calibration, and what an
/// [`Ensemble`] member is: the ensemble's table is the ensemble's, so a member carries none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tower {
    dir: PathBuf,
    manifest_sha256: String,
    weights_sha256: String,
    config_sha256: String,
    weight_hash: String,
    tokenizer_hash: String,
    /// `None` when the manifest does not record what the tower was trained on.
    trained_families: Option<Vec<String>>,
}

/// A release directory whose tower opened and whose calibration table hashes to the bound
/// table hash.
#[derive(Debug, Clone, PartialEq)]
pub struct Release {
    tower: Tower,
    calibration: CalibrationTable,
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Read a release file whole, bounded. A file that is absent, not a regular file, or over
/// [`MAX_SMALL_FILE_BYTES`] is refused with `kind`.
fn read_bounded(path: &Path, kind: ReleaseRefusalKind) -> Result<Vec<u8>, ReleaseRefusal> {
    let meta = std::fs::metadata(path)
        .map_err(|e| ReleaseRefusal::new(kind, format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(ReleaseRefusal::new(
            kind,
            format!("{} is not a regular file", path.display()),
        ));
    }
    if meta.len() > MAX_SMALL_FILE_BYTES {
        return Err(ReleaseRefusal::new(
            kind,
            format!(
                "{}: {} bytes exceeds the {MAX_SMALL_FILE_BYTES}-byte bound",
                path.display(),
                meta.len()
            ),
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| {
        ReleaseRefusal::new(ReleaseRefusalKind::Io, format!("{}: {e}", path.display()))
    })?;
    // The bound is on what was read, not only on what `metadata` said: a file that grew between
    // the two calls is not let through on the strength of its earlier size.
    let read = u64::try_from(bytes.len()).map_err(|e| {
        ReleaseRefusal::new(
            ReleaseRefusalKind::Io,
            format!("{}: byte count: {e}", path.display()),
        )
    })?;
    if read > MAX_SMALL_FILE_BYTES {
        return Err(ReleaseRefusal::new(
            kind,
            format!(
                "{}: read {read} bytes, over the {MAX_SMALL_FILE_BYTES}-byte bound",
                path.display()
            ),
        ));
    }
    Ok(bytes)
}

/// The parsed manifest, read through the fields this reader requires and nothing else.
struct Manifest {
    path: PathBuf,
    doc: Value,
}

impl Manifest {
    fn refuse(&self, detail: impl std::fmt::Display) -> ReleaseRefusal {
        ReleaseRefusal::new(
            ReleaseRefusalKind::Manifest,
            format!("{}: {detail}", self.path.display()),
        )
    }

    /// `doc[a][b]...` as a lowercase 64-hex-digit string.
    fn hex_at(&self, keys: &[&str]) -> Result<String, ReleaseRefusal> {
        let mut at = &self.doc;
        for key in keys {
            at = at
                .get(key)
                .ok_or_else(|| self.refuse(format!("no {}", keys.join("."))))?;
        }
        at.as_str()
            .filter(|s| is_hex64(s))
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| self.refuse(format!("{} is {at}, not 64 hex digits", keys.join("."))))
    }

    /// The sha256 `files` records for `name`.
    fn file_sha256(&self, name: &str) -> Result<String, ReleaseRefusal> {
        self.hex_at(&["files", name, "sha256"])
    }

    /// `trained_families`: `None` when absent or `null`, else a list
    /// [`validate_trained_families`] accepts.
    fn trained_families(&self) -> Result<Option<Vec<String>>, ReleaseRefusal> {
        let value = match self.doc.get(TRAINED_FAMILIES_KEY) {
            None | Some(Value::Null) => return Ok(None),
            Some(value) => value,
        };
        let items = value.as_array().ok_or_else(|| {
            self.refuse(format!(
                "{TRAINED_FAMILIES_KEY} is {}, not an array of family ids",
                json_kind(value)
            ))
        })?;
        let families = items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                item.as_str().map(str::to_string).ok_or_else(|| {
                    self.refuse(format!(
                        "{TRAINED_FAMILIES_KEY}[{i}] is {}, not a string",
                        json_kind(item)
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_trained_families(&families).map_err(|detail| self.refuse(detail))?;
        Ok(Some(families))
    }
}

/// A JSON value's kind, for a refusal that must not echo the value itself.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Read `calibration.json` and refuse unless it is the table the manifest binds.
///
/// Three hashes, checked separately because they answer different questions: the file's bytes
/// against `files` and `calibration.file_sha256` (is this the file that was exported?), and the
/// parsed table's [`CalibrationTable::hash`] against `calibration.table_hash` and
/// `expected_identity.calibration_hash` (is this the table a caller's `expect.calibration_hash`
/// pins?). The bytes' sha256 and the table hash are **not** required to be equal: `qd-export`
/// accepts a table spelled `1` where the runtime writes `1.0`, so the two can differ for one table.
fn load_calibration(dir: &Path, manifest: &Manifest) -> Result<CalibrationTable, ReleaseRefusal> {
    if manifest.doc.get("calibration").is_none_or(Value::is_null) {
        return Err(ReleaseRefusal::new(
            ReleaseRefusalKind::CalibrationMissing,
            format!(
                "{}: calibration is absent: the release was exported without --calibration. The \
                 runtime does not fall back to the reference table, which was fitted on nothing",
                manifest.path.display()
            ),
        ));
    }
    match manifest
        .doc
        .pointer("/calibration/file")
        .and_then(Value::as_str)
    {
        Some(CALIBRATION_FILE) => {}
        other => {
            return Err(manifest.refuse(format!(
                "calibration.file is {other:?}, not {CALIBRATION_FILE:?}"
            )));
        }
    }
    let recorded_file = manifest.file_sha256(CALIBRATION_FILE)?;
    let block_file = manifest.hex_at(&["calibration", "file_sha256"])?;
    if recorded_file != block_file {
        return Err(manifest.refuse(format!(
            "files.{CALIBRATION_FILE}.sha256 is {recorded_file} but calibration.file_sha256 is \
             {block_file}; the manifest disagrees with itself"
        )));
    }
    let bound_table = manifest.hex_at(&["expected_identity", "calibration_hash"])?;
    let block_table = manifest.hex_at(&["calibration", "table_hash"])?;
    if bound_table != block_table {
        return Err(manifest.refuse(format!(
            "expected_identity.calibration_hash is {bound_table} but calibration.table_hash is \
             {block_table}; the manifest disagrees with itself"
        )));
    }

    let path = dir.join(CALIBRATION_FILE);
    let bytes = read_bounded(&path, ReleaseRefusalKind::CalibrationMissing)?;
    let file_sha = crate::hex(&crate::sha256(&bytes));
    if file_sha != recorded_file {
        return Err(ReleaseRefusal::new(
            ReleaseRefusalKind::CalibrationMismatch,
            format!(
                "{} has sha256 {file_sha}; the manifest records {recorded_file}. This is not the \
                 table that was exported",
                path.display()
            ),
        ));
    }
    let table: CalibrationTable = serde_json::from_slice(&bytes).map_err(|e| {
        ReleaseRefusal::new(
            ReleaseRefusalKind::CalibrationInvalid,
            format!("{}: not a CalibrationTable: {e}", path.display()),
        )
    })?;
    table.validate().map_err(|e| {
        ReleaseRefusal::new(
            ReleaseRefusalKind::CalibrationInvalid,
            format!("{}: {e}", path.display()),
        )
    })?;
    let table_hash = table.hash();
    if table_hash != bound_table {
        return Err(ReleaseRefusal::new(
            ReleaseRefusalKind::CalibrationMismatch,
            format!(
                "{} parses to a table with hash {table_hash}; the manifest binds {bound_table}, \
                 which is what a caller's expect.calibration_hash pins",
                path.display()
            ),
        ));
    }
    Ok(table)
}

/// Read `path` as a manifest of `format`: bounded, JSON, and of exactly that format.
fn read_manifest(path: PathBuf, format: &str) -> Result<(Manifest, Vec<u8>), ReleaseRefusal> {
    let bytes = read_bounded(&path, ReleaseRefusalKind::Manifest)?;
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| {
        ReleaseRefusal::new(
            ReleaseRefusalKind::Manifest,
            format!("{}: not JSON: {e}", path.display()),
        )
    })?;
    let manifest = Manifest { path, doc };
    match manifest.doc.get("format").and_then(Value::as_str) {
        Some(found) if found == format => Ok((manifest, bytes)),
        other => Err(manifest.refuse(format!(
            "format is {other:?}, not {format:?}; this reader does not interpret another format"
        ))),
    }
}

/// [`Tower::open`], keeping the parsed manifest for a caller that reads more of it.
fn open_tower(dir: &Path) -> Result<(Tower, Manifest), ReleaseRefusal> {
    let (manifest, bytes) = read_manifest(dir.join(MANIFEST_FILE), MANIFEST_FORMAT)?;

    let weight_hash = manifest.hex_at(&["expected_identity", "weight_hash"])?;
    let tokenizer_hash = manifest.hex_at(&["expected_identity", "tokenizer_hash"])?;
    let weights_sha256 = manifest.file_sha256(WEIGHTS_FILE)?;
    let bound_config = manifest
        .hex_at(&["expected_identity", "config_sha256"])
        .map_err(|e| {
            ReleaseRefusal::new(
                e.kind,
                format!(
                    "{}. A release whose manifest does not bind config.json to the tower \
                     predates the binding; re-export it",
                    e.detail
                ),
            )
        })?;
    let trained_families = manifest.trained_families()?;
    let recorded_config = manifest.file_sha256(CONFIG_FILE)?;
    if recorded_config != bound_config {
        return Err(manifest.refuse(format!(
            "files.{CONFIG_FILE}.sha256 is {recorded_config} but expected_identity.config_sha256 \
             is {bound_config}; the manifest disagrees with itself"
        )));
    }

    let config_path = dir.join(CONFIG_FILE);
    let config = read_bounded(&config_path, ReleaseRefusalKind::ConfigMismatch)?;
    let config_sha256 = crate::hex(&crate::sha256(&config));
    if config_sha256 != bound_config {
        return Err(ReleaseRefusal::new(
            ReleaseRefusalKind::ConfigMismatch,
            format!(
                "{} has sha256 {config_sha256}, but the manifest binds the tower (weight_hash \
                 {weight_hash}) to config sha256 {bound_config}. A config the tower was not \
                 exported with is refused, however its shapes look",
                config_path.display()
            ),
        ));
    }

    let tower = Tower {
        dir: dir.to_path_buf(),
        manifest_sha256: crate::hex(&crate::sha256(&bytes)),
        weights_sha256,
        config_sha256,
        weight_hash,
        tokenizer_hash,
        trained_families,
    };
    Ok((tower, manifest))
}

fn identity_mismatch(
    identity: &BackendIdentity,
    what: &str,
    reported: &str,
    dir: &Path,
    bound: &str,
) -> ReleaseRefusal {
    ReleaseRefusal::new(
        ReleaseRefusalKind::IdentityMismatch,
        format!(
            "backend `{}` reports {what} {reported}; release {} binds {bound}",
            identity.name,
            dir.display()
        ),
    )
}

impl Tower {
    /// Open `dir`, parse its manifest, and refuse unless `config.json` is the config the
    /// manifest binds to the tower.
    pub fn open(dir: &Path) -> Result<Self, ReleaseRefusal> {
        open_tower(dir).map(|(tower, _)| tower)
    }

    /// Refuse a backend that did not load the tower and tokenizer this release binds.
    ///
    /// Run on the identity the backend reports **after** it loaded from [`Tower::dir`]: the
    /// weight hash is the loader's, computed from the tensors it read.
    pub fn check_backend(&self, identity: &BackendIdentity) -> Result<(), ReleaseRefusal> {
        for (what, bound, reported) in [
            ("weight_hash", &self.weight_hash, &identity.weight_hash),
            (
                "tokenizer_hash",
                &self.tokenizer_hash,
                &identity.tokenizer_hash,
            ),
        ] {
            if bound != reported {
                return Err(identity_mismatch(
                    identity, what, reported, &self.dir, bound,
                ));
            }
        }
        Ok(())
    }

    /// The release directory; the backend loads from here.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// SHA-256 of `release_manifest.json` as read.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// The sha256 the manifest records for `model.safetensors`. Recorded, not re-hashed here:
    /// the bytes the backend loaded are bound through [`Tower::check_backend`]'s weight hash.
    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    /// SHA-256 of `config.json`, equal to the bound one.
    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    /// The tower's weight hash the manifest binds.
    pub fn weight_hash(&self) -> &str {
        &self.weight_hash
    }

    /// The tokenizer hash the manifest binds.
    pub fn tokenizer_hash(&self) -> &str {
        &self.tokenizer_hash
    }

    /// The task families the tower was trained on, sorted and unique; `None` when the manifest
    /// does not record them.
    pub fn trained_families(&self) -> Option<&[String]> {
        self.trained_families.as_deref()
    }
}

/// A recorded family list, or that none is recorded, for a refusal that compares two.
pub fn describe_trained_families(families: Option<&[String]>) -> String {
    match families {
        Some(families) => format!("{families:?}"),
        None => "none recorded".to_string(),
    }
}

impl Release {
    /// Open `dir`, parse its manifest, and refuse unless `config.json` is the config the
    /// manifest binds to the tower and `calibration.json` is the table it binds.
    pub fn open(dir: &Path) -> Result<Self, ReleaseRefusal> {
        let (tower, manifest) = open_tower(dir)?;
        let calibration = load_calibration(dir, &manifest)?;
        Ok(Self { tower, calibration })
    }

    /// Refuse a backend that did not load the tower and tokenizer this release binds, or that
    /// declares a calibration table other than the release's.
    ///
    /// Run on the identity the backend reports **after** it loaded from [`Release::dir`]: the
    /// weight hash is the loader's, computed from the tensors it read.
    pub fn check_backend(&self, identity: &BackendIdentity) -> Result<(), ReleaseRefusal> {
        self.tower.check_backend(identity)?;
        let calibration_hash = self.calibration.hash();
        if identity.calibration_hash != calibration_hash {
            return Err(identity_mismatch(
                identity,
                "calibration_hash",
                &identity.calibration_hash,
                &self.tower.dir,
                &calibration_hash,
            ));
        }
        Ok(())
    }

    /// The release's tower.
    pub fn tower(&self) -> &Tower {
        &self.tower
    }

    /// The release directory; the backend loads from here.
    pub fn dir(&self) -> &Path {
        self.tower.dir()
    }

    /// SHA-256 of `config.json`, equal to the bound one.
    pub fn config_sha256(&self) -> &str {
        self.tower.config_sha256()
    }

    /// The tower's weight hash the manifest binds.
    pub fn weight_hash(&self) -> &str {
        self.tower.weight_hash()
    }

    /// The tokenizer hash the manifest binds.
    pub fn tokenizer_hash(&self) -> &str {
        self.tower.tokenizer_hash()
    }

    /// The calibration table, verified against the manifest's hashes.
    pub fn calibration(&self) -> &CalibrationTable {
        &self.calibration
    }

    /// The task families the tower was trained on; `None` when the manifest does not record
    /// them, and then the release admits no task.
    pub fn trained_families(&self) -> Option<&[String]> {
        self.tower.trained_families()
    }
}

// -- the ensemble ---------------------------------------------------------------------------------

/// The ensemble manifest's file name, in the ensemble directory beside the member directories.
pub const ENSEMBLE_MANIFEST_FILE: &str = "ensemble_manifest.json";
/// The ensemble manifest's `format`.
pub const ENSEMBLE_FORMAT: &str = "qd-ensemble.v1";
/// The ensemble's decode, stated in the manifest so a reader that decodes otherwise can refuse.
/// It is what [`crate::ensemble::EnsembleBackend`] does.
pub const ENSEMBLE_DECODE: &str = "mean over members, in manifest order, of the log-softmax over \
                                    the slot's rows; then the runtime's calibration and abstention \
                                    rule on the mean";

/// An ensemble directory: N member towers, each a full release directory that opened as a
/// [`Tower`] and is the tower the ensemble manifest names, plus the ensemble's own calibration
/// table. Every check runs before any member serves; one failing member refuses the whole load.
#[derive(Debug, Clone, PartialEq)]
pub struct Ensemble {
    dir: PathBuf,
    towers: Vec<Tower>,
    trained_width: u64,
    weight_hash: String,
    calibration: CalibrationTable,
}

/// A member's directory: one plain path component, so the manifest cannot point outside the
/// ensemble directory.
fn member_dir_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0'])
}

impl Ensemble {
    /// Open `dir`: the ensemble manifest, every member tower, and the ensemble's table.
    pub fn open(dir: &Path) -> Result<Self, ReleaseRefusal> {
        let (manifest, _) = read_manifest(dir.join(ENSEMBLE_MANIFEST_FILE), ENSEMBLE_FORMAT)?;
        match manifest.doc.get("decode").and_then(Value::as_str) {
            Some(ENSEMBLE_DECODE) => {}
            other => {
                return Err(manifest.refuse(format!(
                    "decode is {other:?}; this runtime decodes an ensemble as {ENSEMBLE_DECODE:?} \
                     and does not serve one recorded for another decode"
                )));
            }
        }
        let members = manifest
            .doc
            .get("members")
            .and_then(Value::as_array)
            .ok_or_else(|| manifest.refuse("members is not an array"))?;
        if !(2..=crate::ensemble::MAX_MEMBERS).contains(&members.len()) {
            return Err(manifest.refuse(format!(
                "{} members; an ensemble has 2 to {}",
                members.len(),
                crate::ensemble::MAX_MEMBERS
            )));
        }

        let mut towers = Vec::with_capacity(members.len());
        let mut widths = Vec::with_capacity(members.len());
        for (i, member) in members.iter().enumerate() {
            let field = |key: &str| {
                member
                    .get(key)
                    .ok_or_else(|| manifest.refuse(format!("members[{i}] has no {key}")))
            };
            let name = field("dir")?
                .as_str()
                .filter(|name| member_dir_name(name))
                .ok_or_else(|| {
                    manifest.refuse(format!(
                        "members[{i}].dir is not one plain path component inside the ensemble \
                         directory"
                    ))
                })?;
            let tower = Tower::open(&dir.join(name)).map_err(|e| {
                ReleaseRefusal::new(
                    e.kind,
                    format!(
                        "ensemble member {i} (`{name}`): {}. One member that does not open \
                         refuses the ensemble; it is never served with the rest",
                        e.detail
                    ),
                )
            })?;
            for (key, opened) in [
                ("release_manifest_sha256", tower.manifest_sha256()),
                ("weights_sha256", tower.weights_sha256()),
                ("weight_hash", tower.weight_hash()),
                ("config_sha256", tower.config_sha256()),
                ("tokenizer_hash", tower.tokenizer_hash()),
            ] {
                let named = field(key)?
                    .as_str()
                    .filter(|s| is_hex64(s))
                    .map(str::to_ascii_lowercase)
                    .ok_or_else(|| {
                        manifest.refuse(format!("members[{i}].{key} is not 64 hex digits"))
                    })?;
                if named != opened {
                    return Err(ReleaseRefusal::new(
                        ReleaseRefusalKind::MemberMismatch,
                        format!(
                            "ensemble member {i} (`{name}`) has {key} {opened}; the ensemble \
                             manifest names {named}. It is not the tower the ensemble was \
                             written with"
                        ),
                    ));
                }
            }
            let width = field("trained_width")?
                .as_u64()
                .filter(|w| *w > 0)
                .ok_or_else(|| {
                    manifest.refuse(format!(
                        "members[{i}].trained_width is not a positive integer; a member whose \
                         trained width is unknown cannot be checked against the others"
                    ))
                })?;
            towers.push(tower);
            widths.push(width);
        }

        let first = &towers[0];
        for (i, tower) in towers.iter().enumerate().skip(1) {
            for (what, mine, theirs) in [
                (
                    "config_sha256",
                    tower.config_sha256(),
                    first.config_sha256(),
                ),
                (
                    "tokenizer_hash",
                    tower.tokenizer_hash(),
                    first.tokenizer_hash(),
                ),
            ] {
                if mine != theirs {
                    return Err(ReleaseRefusal::new(
                        ReleaseRefusalKind::MembersDisagree,
                        format!(
                            "ensemble member {i} has {what} {mine}, member 0 has {theirs}. \
                             Towers served under different configs or tokenizers do not \
                             answer the same question"
                        ),
                    ));
                }
            }
            if widths[i] != widths[0] {
                return Err(ReleaseRefusal::new(
                    ReleaseRefusalKind::MembersDisagree,
                    format!(
                        "ensemble member {i} was trained at width {}, member 0 at {}",
                        widths[i], widths[0]
                    ),
                ));
            }
            if tower.trained_families() != first.trained_families() {
                return Err(ReleaseRefusal::new(
                    ReleaseRefusalKind::MembersDisagree,
                    format!(
                        "ensemble member {i} has trained_families {}, member 0 has {}. The mean \
                         of towers trained on different task families answers no family set any \
                         one of them was trained on",
                        describe_trained_families(tower.trained_families()),
                        describe_trained_families(first.trained_families())
                    ),
                ));
            }
            if let Some(j) = towers[..i]
                .iter()
                .position(|other| other.weight_hash() == tower.weight_hash())
            {
                return Err(ReleaseRefusal::new(
                    ReleaseRefusalKind::MembersDisagree,
                    format!(
                        "ensemble members {j} and {i} are one tower (weight_hash {}); averaging \
                         it twice would weight one seed double while the manifest counts two",
                        tower.weight_hash()
                    ),
                ));
            }
        }

        let member_hashes: Vec<&str> = towers.iter().map(Tower::weight_hash).collect();
        let weight_hash = crate::ensemble::ensemble_weight_hash(&member_hashes);
        for (key, value) in [
            ("weight_hash", weight_hash.as_str()),
            ("tokenizer_hash", first.tokenizer_hash()),
            ("config_sha256", first.config_sha256()),
        ] {
            let bound = manifest.hex_at(&["expected_identity", key])?;
            if bound != value {
                return Err(manifest.refuse(format!(
                    "expected_identity.{key} is {bound}, but the members give {value}; the \
                     manifest disagrees with itself"
                )));
            }
        }
        match manifest.doc.pointer("/expected_identity/trained_width") {
            Some(bound) if bound.as_u64() == Some(widths[0]) => {}
            other => {
                return Err(manifest.refuse(format!(
                    "expected_identity.trained_width is {other:?}, the members' is {}",
                    widths[0]
                )));
            }
        }

        let calibration = load_calibration(dir, &manifest)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            towers,
            trained_width: widths[0],
            weight_hash,
            calibration,
        })
    }

    /// The ensemble directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The members, in manifest order: the order their log-probabilities are averaged in.
    pub fn towers(&self) -> &[Tower] {
        &self.towers
    }

    /// The width every member was trained at, as the writer's operator stated and cited it.
    pub fn trained_width(&self) -> u64 {
        self.trained_width
    }

    /// The ensemble's weight hash: [`crate::ensemble::ensemble_weight_hash`] over the members'.
    pub fn weight_hash(&self) -> &str {
        &self.weight_hash
    }

    /// The ensemble's calibration table, verified against the manifest's hashes.
    pub fn calibration(&self) -> &CalibrationTable {
        &self.calibration
    }

    /// The task families every member was trained on ([`Ensemble::open`] refuses members that
    /// disagree); `None` when the members do not record them, and then the ensemble admits no
    /// task.
    pub fn trained_families(&self) -> Option<&[String]> {
        self.towers.first().and_then(Tower::trained_families)
    }
}
