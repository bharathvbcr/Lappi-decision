//! One error type for the crate, tagged by the stage that failed.
//!
//! tessl reports failures as `String`; they are wrapped here with the stage that called it, so a
//! message reaching `qd-runtime` says whether the model files, the tokenizer, the host-side input
//! or a GPU dispatch was at fault.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// `config.json` is missing, malformed, or describes a model these kernels do not implement.
    Config(String),
    /// The safetensors file is malformed or a tensor is missing / the wrong shape or dtype.
    Weights(String),
    /// `tokenizer.json` failed to load, or a letter is not a single token.
    Tokenizer(String),
    /// The caller's input is outside what the backend serves (empty, too long, bad token id).
    Input(String),
    /// A tessl call or a host mapping failed.
    Gpu(String),
    /// The snapshot store: an unknown, evicted or foreign snapshot.
    State(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::Config(s) => write!(f, "model config: {s}"),
            MetalError::Weights(s) => write!(f, "weights: {s}"),
            MetalError::Tokenizer(s) => write!(f, "tokenizer: {s}"),
            MetalError::Input(s) => write!(f, "input: {s}"),
            MetalError::Gpu(s) => write!(f, "gpu: {s}"),
            MetalError::State(s) => write!(f, "state: {s}"),
        }
    }
}

impl std::error::Error for MetalError {}

pub type Result<T> = std::result::Result<T, MetalError>;

/// Wrap a tessl `Result<_, String>` with the operation that produced it.
pub(crate) trait GpuContext<T> {
    fn gpu(self, what: &str) -> Result<T>;
}

impl<T> GpuContext<T> for std::result::Result<T, String> {
    fn gpu(self, what: &str) -> Result<T> {
        self.map_err(|e| MetalError::Gpu(format!("{what}: {e}")))
    }
}
