//! `qd-metal-serve`: `qd serve` with the Metal backend behind it.
//!
//! Fable's Mac ruling, item 5 (`GAP-QD-SERVE-WIRES-NO-METAL-BACKEND-2026-10-02`): `qd serve`
//! wires no model backend (`crates/qd-runtime/src/main.rs`), and qd-runtime cannot depend on
//! qd-metal because tessl links Metal. So the product path is assembled here, from qd-runtime's
//! public pieces and nothing else:
//!
//! ```text
//! Release::open(dir)                     manifest, config and calibration bound, at startup
//!   -> MetalBackend::start(snapshot = dir, calibration_hash = the release's table)
//!   -> Runtime::from_release             refuses a backend that is not the release's tower
//!   -> Service::with_factory             the warm slot, idle eviction, poison and rebuild
//!   -> Server::bind / run                the socket, one line in, one line out
//! ```
//!
//! The release opens once, before the socket binds: a directory that is not a release is refused
//! at startup and nothing listens. The backend is started by the service's factory on the first
//! request and again after an idle eviction or a poison, exactly as `qd serve` builds its
//! runtime. [`release_factory`] takes the backend starter as a parameter so the wiring is tested
//! on the CPU with the reference backend (`tests/serve_cpu.rs`); the binary passes
//! [`metal_starter`].
//!
//! # What this inherits from qd-runtime as it is on main
//!
//! `Service` (main, 276a475) runs the factory while it holds its lifecycle lock, so `status` and
//! `ping` wait for a multi-second Metal cold start. Another session's uncommitted
//! `service.rs` change moves the build outside the lock; nothing here depends on either.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use qd_runtime::backend::DecisionBackend;
use qd_runtime::refusal::BackendError;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::release::{Release, ReleaseRefusal};
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::service::RuntimeFactory;

use crate::backend::{MetalBackend, MetalConfig};

/// Starts a backend that loaded from a release directory.
pub type BackendStarter =
    Arc<dyn Fn(&Release) -> Result<Arc<dyn DecisionBackend>, BackendError> + Send + Sync>;

/// Open the release the server will answer from. Refused before anything binds.
pub fn open_release(dir: &Path) -> Result<Release, ReleaseRefusal> {
    Release::open(dir)
}

/// The service's factory: start a backend over `release` and bind it to the release with
/// [`Runtime::from_release`], which refuses a backend whose weight, tokenizer or calibration hash
/// is not the release's. Called on every cold start.
pub fn release_factory(
    release: Release,
    registry: HeadRegistry,
    caps: RenderCaps,
    start: BackendStarter,
) -> RuntimeFactory {
    Arc::new(move || {
        let backend = start(&release)?;
        Runtime::from_release(&release, backend, registry.clone(), caps)
    })
}

/// The Metal backend's bounds. Every one is stated by the caller ([`MetalConfig`]'s rule); the
/// binary's flags carry the values it uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalBounds {
    pub queue_capacity: usize,
    pub max_entries: usize,
    pub max_tokens: usize,
    pub job_timeout: Duration,
}

/// The [`MetalConfig`] for `release`: its directory is the snapshot the loader reads
/// (`config.json`, `model.safetensors`, `tokenizer.json`), and the calibration hash the backend
/// reports is the hash of the table [`Release::open`] verified.
pub fn metal_config(release: &Release, bounds: MetalBounds) -> MetalConfig {
    MetalConfig {
        snapshot: Some(release.dir().to_path_buf()),
        calibration_hash: release.calibration().hash(),
        queue_capacity: bounds.queue_capacity,
        max_entries: bounds.max_entries,
        max_tokens: bounds.max_tokens,
        job_timeout: bounds.job_timeout,
    }
}

/// Start [`MetalBackend`] over the release. A load failure is `unavailable`, naming the release:
/// the model was never asked.
pub fn metal_starter(bounds: MetalBounds) -> BackendStarter {
    Arc::new(move |release: &Release| {
        let backend = MetalBackend::start(metal_config(release, bounds)).map_err(|e| {
            BackendError::Unavailable {
                detail: format!(
                    "the qd-metal backend did not start over release {}: {e}",
                    release.dir().display()
                ),
            }
        })?;
        Ok(Arc::new(backend) as Arc<dyn DecisionBackend>)
    })
}
