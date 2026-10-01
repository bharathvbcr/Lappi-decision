//! The release directory `qd-export` writes, opened and bound before anything serves from it.
//!
//! `crates/qd-export/src/export.rs` is the writer. This module is the one reader, and it owns the
//! file names and the manifest format both sides use: `qd-export` imports them from here, so the
//! writer and the reader cannot spell a file two ways.
//!
//! # What is bound, and where it is checked
//!
//! `release_manifest.json`'s `expected_identity` is the release's one statement of what goes
//! together: the tower (`weight_hash`), the `config.json` it is served under (`config_sha256`) and
//! the tokenizer (`tokenizer_hash`). Two halves check it:
//!
//! * [`Release::open`] reads `config.json` and refuses unless its sha256 is the bound one **and**
//!   the one `files` records. A same-shaped config from another revision (a different
//!   `rope_theta`, say) beside the same tower is refused here, before a backend reads it.
//! * [`Release::check_backend`] refuses a backend whose reported `weight_hash` or
//!   `tokenizer_hash` is not the bound one. The weight hash is computed by the loader from the
//!   tensors it actually read, so this is the tower half of the binding.
//!
//! Nothing here parses `config.json` as a model config: which layouts a kernel runs is the
//! backend's question. The runtime binds bytes.
//!
//! # What this does not close
//!
//! The backend opens `config.json` by path after [`Release::open`] hashed it. A file swapped
//! between the two reads is not caught here; closing that needs the backend to take the verified
//! bytes, which is a change to the backend's constructor.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::backend::BackendIdentity;

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

/// Why a release was refused at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseRefusalKind {
    /// `release_manifest.json` is missing, unreadable, over the bound, not JSON, of another
    /// format, or missing a field this reader requires.
    Manifest,
    /// `config.json` is not the config the manifest binds to the tower.
    ConfigMismatch,
    /// The backend reports a weight or tokenizer hash the manifest does not bind.
    IdentityMismatch,
    /// A release file could not be read.
    Io,
}

impl ReleaseRefusalKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ReleaseRefusalKind::Manifest => "manifest",
            ReleaseRefusalKind::ConfigMismatch => "config_mismatch",
            ReleaseRefusalKind::IdentityMismatch => "identity_mismatch",
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

/// A release directory whose manifest parsed and whose `config.json` is the bound one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    dir: PathBuf,
    config_sha256: String,
    weight_hash: String,
    tokenizer_hash: String,
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
}

impl Release {
    /// Open `dir`, parse its manifest, and refuse unless `config.json` is the config the
    /// manifest binds to the tower.
    pub fn open(dir: &Path) -> Result<Self, ReleaseRefusal> {
        let path = dir.join(MANIFEST_FILE);
        let bytes = read_bounded(&path, ReleaseRefusalKind::Manifest)?;
        let doc: Value = serde_json::from_slice(&bytes).map_err(|e| {
            ReleaseRefusal::new(
                ReleaseRefusalKind::Manifest,
                format!("{}: not JSON: {e}", path.display()),
            )
        })?;
        let manifest = Manifest { path, doc };
        match manifest.doc.get("format").and_then(Value::as_str) {
            Some(MANIFEST_FORMAT) => {}
            other => {
                return Err(manifest.refuse(format!(
                    "format is {other:?}, not {MANIFEST_FORMAT:?}; this reader does not \
                     interpret another format"
                )));
            }
        }

        let weight_hash = manifest.hex_at(&["expected_identity", "weight_hash"])?;
        let tokenizer_hash = manifest.hex_at(&["expected_identity", "tokenizer_hash"])?;
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

        Ok(Self {
            dir: dir.to_path_buf(),
            config_sha256,
            weight_hash,
            tokenizer_hash,
        })
    }

    /// Refuse a backend that did not load the tower and tokenizer this release binds.
    ///
    /// Run on the identity the backend reports **after** it loaded from [`Release::dir`]: the
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
                return Err(ReleaseRefusal::new(
                    ReleaseRefusalKind::IdentityMismatch,
                    format!(
                        "backend `{}` reports {what} {reported}; release {} binds {bound}",
                        identity.name,
                        self.dir.display()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The release directory; the backend loads from here.
    pub fn dir(&self) -> &Path {
        &self.dir
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
}
