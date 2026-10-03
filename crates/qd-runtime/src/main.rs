//! `qd` — the two serving modes.
//!
//! ```text
//! qd serve    one warm runtime behind a Unix socket, JSON lines. The launchd agent.
//! qd oneshot  one request on stdin, one reply on stdout, exit.
//! ```
//!
//! Both modes route every line through [`qd_runtime::Service::handle_line`]. That is the whole of
//! the shared contract, and `tests/two_modes_one_contract.rs` asserts it by driving identical bytes
//! through both and comparing them byte for byte.
//!
//! # Exit codes
//!
//! `qd oneshot` distinguishes the three outcomes, because a caller scripting it should not have to
//! parse JSON to find out whether it got an answer:
//!
//! | Code | Meaning |
//! | --- | --- |
//! | 0 | an answer, or a control-op reply |
//! | 2 | a **refusal** — the request was not answerable as posed |
//! | 3 | a **backend error** — the model was never asked |
//! | 1 | this process could not run: bad arguments, no socket, an I/O failure |
//!
//! 2 and 3 are deliberately different. A refusal is a bug in the request; a backend error is a bug
//! in the machine. Collapsing them would put a hash mismatch and a missing model in the same bucket.

use std::io::{self, Write};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};

use qd_runtime::oneshot::{ask, read_one_line, write_reply};
use qd_runtime::runtime::RuntimeConfig;
use qd_runtime::serve::{ServeOptions, Server};
use qd_runtime::service::{Service, ServiceConfig};
use qd_runtime::wire::MAX_PAYLOAD_BYTES;

#[derive(Parser, Debug)]
#[command(
    name = "qd",
    version,
    about = "The universal typed-decision interface: qd serve and qd oneshot",
    long_about = "Both modes answer the same JSON-lines protocol through one code path.\n\n\
                  No model backend ships in this build: the Metal kernels (K1-K7) start after the \
                  shipping gate. Without --reference-backend every request is answered with a \
                  typed `unavailable` error, which is the honest reply and is never a `noul`."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the agent: one warm runtime behind a Unix socket.
    Serve(ServeArgs),
    /// Answer one request from stdin and exit.
    Oneshot(OneshotArgs),
}

#[derive(Args, Debug, Clone)]
struct RuntimeArgs {
    /// Enable the deterministic reference backend.
    ///
    /// It is NOT a model. Its scores are a hash of the prompt, it names itself in every answer, and
    /// every answer it produces is flagged `degraded`. It exists so the plumbing can be tested
    /// before a kernel exists.
    #[arg(long)]
    reference_backend: bool,

    /// Wall clock for one request, end to end.
    #[arg(long, value_name = "MS", default_value_t = NonZeroU64::new(30_000).unwrap())]
    request_timeout_ms: NonZeroU64,

    /// Concurrent in-flight requests. Beyond this the runtime replies `overloaded`.
    #[arg(long, value_name = "N", default_value_t = NonZeroUsize::new(8).unwrap())]
    max_in_flight: NonZeroUsize,
}

#[derive(Args, Debug)]
struct ServeArgs {
    /// Socket to listen on. Defaults to $HOME/Library/Caches/qd/qd.sock.
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,

    /// Drop the warm runtime after this long with no request.
    ///
    /// `docs/hardening.md` §6: an agent that never evicts keeps 1.1 GB wired forever; one that
    /// evicts too eagerly makes every DevType call cold. Ten minutes is the plan's figure.
    #[arg(long, value_name = "MS", default_value_t = NonZeroU64::new(600_000).unwrap())]
    idle_timeout_ms: NonZeroU64,

    /// Concurrent connections. Beyond this a connection is told `overloaded` and closed.
    #[arg(long, value_name = "N", default_value_t = NonZeroUsize::new(64).unwrap())]
    max_connections: NonZeroUsize,

    #[command(flatten)]
    runtime: RuntimeArgs,
}

#[derive(Args, Debug)]
struct OneshotArgs {
    /// Try this socket first and fall back to answering in-process.
    ///
    /// This is dcverify's path: a verify run must work with nothing else running, so the fallback
    /// is not optional. Omit the flag to skip the socket entirely.
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,

    /// Timeout for the socket attempt before falling back.
    #[arg(long, value_name = "MS", default_value_t = NonZeroU64::new(5_000).unwrap())]
    socket_timeout_ms: NonZeroU64,

    #[command(flatten)]
    runtime: RuntimeArgs,
}

impl RuntimeArgs {
    fn service_config(&self, idle_timeout_ms: u64) -> ServiceConfig {
        ServiceConfig {
            runtime: RuntimeConfig {
                caps: qd_runtime::render::RenderCaps::DEFAULT,
                enable_reference_backend: self.reference_backend,
            },
            idle_timeout: Duration::from_millis(idle_timeout_ms),
            request_timeout: Duration::from_millis(self.request_timeout_ms.get()),
            max_in_flight: self.max_in_flight.get(),
        }
    }
}

fn default_socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join("Library")
            .join("Caches")
            .join("qd")
            .join("qd.sock"),
    )
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("qd: {error}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    match Cli::parse().command {
        Command::Serve(args) => run_serve(args),
        Command::Oneshot(args) => run_oneshot(args),
    }
}

fn run_serve(args: ServeArgs) -> Result<ExitCode, String> {
    let socket = args
        .socket
        .clone()
        .or_else(default_socket_path)
        .ok_or_else(|| {
            "no --socket given and $HOME is unset, so there is no default path to listen on"
                .to_string()
        })?;

    let service = Arc::new(Service::new(
        args.runtime.service_config(args.idle_timeout_ms.get()),
    ));
    let mut options = ServeOptions::new(socket);
    options.max_connections = args.max_connections.get();

    let server = Server::bind(Arc::clone(&service), options)
        .map_err(|e| format!("could not bind the socket: {e}"))?;
    eprintln!(
        "qd serve: listening on {} (idle eviction {} ms, request timeout {} ms, backend {})",
        server.socket_path().display(),
        args.idle_timeout_ms,
        args.runtime.request_timeout_ms,
        if args.runtime.reference_backend {
            "reference-deterministic (NOT a model; every answer is degraded)"
        } else {
            "none (every request answers `unavailable`)"
        }
    );
    server.run().map_err(|e| format!("serve failed: {e}"))?;
    Ok(ExitCode::SUCCESS)
}

fn run_oneshot(args: OneshotArgs) -> Result<ExitCode, String> {
    let line = read_one_line(io::stdin().lock(), MAX_PAYLOAD_BYTES)
        .map_err(|e| format!("could not read the request from stdin: {e}"))?;
    if line.is_empty() {
        return Err("stdin was empty: `qd oneshot` answers exactly one request".to_string());
    }

    // Built whether or not the socket is tried: it is the fallback, and building it is cheap
    // because a `Service` does not build a runtime until a request needs one.
    let service = Service::new(args.runtime.service_config(u64::MAX / 2));
    let (reply, served_by) = ask(
        &service,
        args.socket.as_deref(),
        &line,
        Duration::from_millis(args.socket_timeout_ms.get()),
    );

    let mut stdout = io::stdout().lock();
    write_reply(&mut stdout, &reply).map_err(|e| format!("could not write the reply: {e}"))?;
    let _ = stdout.flush();
    eprintln!("qd oneshot: served by {}", served_by.as_str());

    Ok(exit_code_for(&reply))
}

/// Read the reply's `status` tag. Every reply this build emits carries one; a reply without one did
/// not come from this protocol, and 1 ("this process could not run") is the honest code for that.
fn exit_code_for(reply: &[u8]) -> ExitCode {
    let status = serde_json::from_slice::<serde_json::Value>(reply)
        .ok()
        .and_then(|v| v.get("status").and_then(|s| s.as_str()).map(str::to_string));
    match status.as_deref() {
        Some("ok") | Some("status") | Some("ack") => ExitCode::SUCCESS,
        Some("refused") => ExitCode::from(2),
        Some("error") => ExitCode::from(3),
        _ => ExitCode::from(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fail-first (audit 2026-10-03, M4): a zero bound used to parse. `--max-in-flight 0` and
    /// `--max-connections 0` were coerced to 1, the timeouts were taken as zero (every request
    /// expired at once, or the runtime was evicted after every request), and
    /// `--socket-timeout-ms 0` made every socket attempt fail and fall back to the sidecar.
    #[test]
    fn a_zero_bound_is_refused_at_parse() {
        let cases: [(&str, &str); 6] = [
            ("serve", "--idle-timeout-ms"),
            ("serve", "--max-connections"),
            ("serve", "--request-timeout-ms"),
            ("serve", "--max-in-flight"),
            ("oneshot", "--socket-timeout-ms"),
            ("oneshot", "--request-timeout-ms"),
        ];
        for (command, flag) in cases {
            assert!(
                Cli::try_parse_from(["qd", command, flag, "0"]).is_err(),
                "qd {command} {flag} 0 was accepted"
            );
            assert!(
                Cli::try_parse_from(["qd", command, flag, "1"]).is_ok(),
                "qd {command} {flag} 1 was refused"
            );
        }
    }

    #[test]
    fn the_defaults_are_unchanged() {
        let Command::Serve(serve) = Cli::try_parse_from(["qd", "serve"]).unwrap().command else {
            panic!("`qd serve` parsed as another command");
        };
        assert_eq!(
            (
                serve.idle_timeout_ms.get(),
                serve.max_connections.get(),
                serve.runtime.request_timeout_ms.get(),
                serve.runtime.max_in_flight.get(),
            ),
            (600_000, 64, 30_000, 8)
        );
        let Command::Oneshot(oneshot) = Cli::try_parse_from(["qd", "oneshot"]).unwrap().command
        else {
            panic!("`qd oneshot` parsed as another command");
        };
        assert_eq!(oneshot.socket_timeout_ms.get(), 5_000);
    }
}
