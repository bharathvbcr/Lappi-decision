//! `qd-metal-serve` — `qd serve` answering from a release through the Metal backend.
//!
//! ```text
//! cargo run --release -p qd-metal --bin qd-metal-serve -- --release <qd-export release dir>
//! ```
//!
//! Same socket, same JSON-lines protocol, same lifecycle as `qd serve`: every line goes through
//! `qd_runtime::Service::handle_line`. The difference is the runtime it builds: a
//! `MetalBackend` loaded from `--release`, bound to that release by `Runtime::from_release`. The
//! wiring is `qd_metal::serve` ([`qd_metal::serve::release_factory`]).
//!
//! **This loads the model on the GPU** (3.8 GB of bf16 weights) on the first request and again
//! after every idle eviction. The release is opened and checked at startup; a directory that is
//! not a release is refused and nothing binds.
//!
//! # `--idle-timeout-ms`
//!
//! The default is qd-runtime's (`ServiceConfig::default`, ten minutes, `docs/hardening.md` §6).
//! A resident 3.8 GB model is a different trade-off from the 1.1 GB that figure was set for; a
//! longer default is the human's decision (Fable's ruling, item 5), so it is only a flag here.
//!
//! # Exit codes
//!
//! 0 after a clean shutdown; 1 when the release is refused, the socket cannot bind, or the
//! server fails.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;

use qd_metal::serve::{metal_starter, open_release, release_factory, MetalBounds};
use qd_runtime::registry::HeadRegistry;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::RuntimeConfig;
use qd_runtime::serve::{ServeOptions, Server};
use qd_runtime::service::{Service, ServiceConfig};

/// qd-runtime's idle eviction, in milliseconds: the flag's default is read from it, not copied.
fn default_idle_timeout_ms() -> u64 {
    u64::try_from(ServiceConfig::default().idle_timeout.as_millis()).unwrap_or(u64::MAX)
}

fn default_request_timeout_ms() -> u64 {
    u64::try_from(ServiceConfig::default().request_timeout.as_millis()).unwrap_or(u64::MAX)
}

fn default_max_in_flight() -> usize {
    ServiceConfig::default().max_in_flight
}

#[derive(Parser, Debug)]
#[command(
    name = "qd-metal-serve",
    about = "qd serve with the Metal backend: one warm runtime over a qd-export release, behind a Unix socket"
)]
pub struct Cli {
    /// The release directory `qd-export` wrote (release_manifest.json, config.json,
    /// model.safetensors, tokenizer.json, calibration.json).
    #[arg(long, value_name = "DIR")]
    pub release: PathBuf,

    /// Socket to listen on. Defaults to $HOME/Library/Caches/qd/qd.sock, as `qd serve`.
    #[arg(long, value_name = "PATH")]
    pub socket: Option<PathBuf>,

    /// Drop the warm runtime (and the model) after this long with no request. The default is
    /// qd-runtime's; a longer one for a resident 3.8 GB model is the human's decision.
    #[arg(long, value_name = "MS", default_value_t = default_idle_timeout_ms())]
    pub idle_timeout_ms: u64,

    /// Wall clock for one request, end to end.
    #[arg(long, value_name = "MS", default_value_t = default_request_timeout_ms())]
    pub request_timeout_ms: u64,

    /// Concurrent in-flight requests. Beyond this the runtime replies `overloaded`.
    #[arg(long, value_name = "N", default_value_t = default_max_in_flight())]
    pub max_in_flight: usize,

    /// Concurrent connections. Beyond this a connection is told `overloaded` and closed.
    #[arg(long, value_name = "N", default_value_t = 64)]
    pub max_connections: usize,

    /// Backend jobs that may wait for the GPU thread. Beyond this a job is refused `overloaded`.
    #[arg(long, value_name = "N", default_value_t = 8)]
    pub queue_capacity: usize,

    /// Prefills and snapshots the backend holds at once (LRU beyond this).
    #[arg(long, value_name = "N", default_value_t = 8)]
    pub max_entries: usize,

    /// Longest prefix + continuation the backend serves, in tokens.
    #[arg(long, value_name = "TOKENS", default_value_t = 16_384)]
    pub max_tokens: usize,

    /// How long one backend job (a prefill or a decode) may take before it is `deadline_exceeded`.
    #[arg(long, value_name = "MS", default_value_t = 30_000)]
    pub job_timeout_ms: u64,
}

fn default_socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join("Library").join("Caches").join("qd").join("qd.sock"))
}

fn run(cli: Cli) -> Result<(), String> {
    let release = open_release(&cli.release).map_err(|e| format!("{e}; nothing was served"))?;
    let socket = cli
        .socket
        .clone()
        .or_else(default_socket_path)
        .ok_or("no --socket given and $HOME is unset, so there is no default path to listen on")?;
    let bounds = MetalBounds {
        queue_capacity: cli.queue_capacity,
        max_entries: cli.max_entries,
        max_tokens: cli.max_tokens,
        job_timeout: Duration::from_millis(cli.job_timeout_ms),
    };
    let caps = RenderCaps::DEFAULT;
    let factory = release_factory(release.clone(), HeadRegistry::new(), caps, metal_starter(bounds));
    let cfg = ServiceConfig {
        runtime: RuntimeConfig {
            caps,
            enable_reference_backend: false,
        },
        idle_timeout: Duration::from_millis(cli.idle_timeout_ms),
        request_timeout: Duration::from_millis(cli.request_timeout_ms),
        max_in_flight: cli.max_in_flight.max(1),
    };
    let service = Arc::new(Service::with_factory(cfg, factory));
    let mut options = ServeOptions::new(socket);
    options.max_connections = cli.max_connections.max(1);
    let server = Server::bind(Arc::clone(&service), options)
        .map_err(|e| format!("could not bind the socket: {e}"))?;
    eprintln!(
        "qd-metal-serve: listening on {} (release {}, weight_hash {}, calibration {}; idle \
         eviction {} ms, request timeout {} ms; the model loads on the first request)",
        server.socket_path().display(),
        release.dir().display(),
        release.weight_hash(),
        release.calibration().hash(),
        cli.idle_timeout_ms,
        cli.request_timeout_ms,
    );
    match release.trained_families() {
        Some(families) => eprintln!(
            "qd-metal-serve: admits the tasks the release was trained on: {}",
            families.join(", ")
        ),
        None => eprintln!(
            "qd-metal-serve: the release does not record trained_families, so every request is \
             refused as task_not_trained; re-export it with qd-export --train-manifest"
        ),
    }
    server.run().map_err(|e| format!("serve failed: {e}"))
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("qd-metal-serve: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_idle_timeout_default_is_qd_runtimes_not_a_new_one() {
        let cli = Cli::try_parse_from(["qd-metal-serve", "--release", "/r"]).unwrap();
        assert_eq!(cli.idle_timeout_ms, 600_000);
        assert_eq!(Duration::from_millis(cli.idle_timeout_ms), ServiceConfig::default().idle_timeout);
        assert_eq!(Duration::from_millis(cli.request_timeout_ms), ServiceConfig::default().request_timeout);
        assert_eq!(cli.max_in_flight, ServiceConfig::default().max_in_flight);
        assert_eq!((cli.queue_capacity, cli.max_entries, cli.max_tokens, cli.job_timeout_ms), (8, 8, 16_384, 30_000));
    }

    #[test]
    fn the_release_is_required() {
        assert!(Cli::try_parse_from(["qd-metal-serve"]).is_err());
        let cli = Cli::try_parse_from(["qd-metal-serve", "--release", "/r", "--idle-timeout-ms", "3600000"]).unwrap();
        assert_eq!(cli.idle_timeout_ms, 3_600_000);
    }

    #[test]
    fn a_directory_that_is_not_a_release_binds_nothing() {
        let dir = std::env::temp_dir().join(format!("qdm-serve-notrelease-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket = dir.join("s.sock");
        let cli = Cli::try_parse_from([
            "qd-metal-serve",
            "--release",
            dir.to_str().unwrap(),
            "--socket",
            socket.to_str().unwrap(),
        ])
        .unwrap();
        let e = run(cli).unwrap_err();
        assert!(e.contains("release refused (manifest)"), "{e}");
        assert!(!socket.exists(), "a refused release must not leave a socket behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
