//! Every way an export ends without a release. Each is a refusal with a kind a test can match
//! and a detail a human can act on; nothing is coerced into a release.

/// Why an export was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    /// The output directory (or this run's staging directory) already exists.
    OutputExists,
    /// The base snapshot's `config.json` does not describe a layout the loader reads.
    Config,
    /// The vocabulary disagrees between the operator, the config, the embedding and the source
    /// manifest.
    VocabMismatch,
    /// The source safetensors file is malformed.
    SourceFormat,
    /// The source manifest is missing, malformed, or does not describe the source file.
    SourceManifest,
    /// A tensor the loader needs is not in the source.
    MissingTensor,
    /// The source holds a tensor the release would not carry, and no allowlist entry gives a
    /// reason to drop it.
    ExtraTensor,
    /// A tensor's shape disagrees with the shape the config implies.
    ShapeMismatch,
    /// A tensor's dtype is not one this export converts without changing more than precision.
    Dtype,
    /// A value is NaN or infinite, or a finite value would round to infinity in bf16.
    NonFinite,
    /// A tokenizer file is missing, does not match its pin, or did not copy intact.
    Tokenizer,
    /// The calibration table is not one the runtime would read back unchanged.
    Calibration,
    /// The filesystem failed.
    Io,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{kind:?}: {detail}")]
pub struct Refusal {
    pub kind: RefusalKind,
    pub detail: String,
}

impl Refusal {
    pub fn new(kind: RefusalKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    pub fn io(path: &std::path::Path, err: impl std::fmt::Display) -> Self {
        Self::new(RefusalKind::Io, format!("{}: {err}", path.display()))
    }
}

pub type Result<T> = std::result::Result<T, Refusal>;
