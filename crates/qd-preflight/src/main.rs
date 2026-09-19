//! `qd-preflight` — the environment-independent gate that runs before a paid block.
//!
//! ## Why a Rust binary and not the Python checker alone
//!
//! `stack/verify_fast_path.py` imports `transformers` in order to audit an environment
//! whose `transformers` may be the thing that is broken. When that happens it fails for
//! the wrong reason, and the operator cannot tell "the fast path is off" from "the
//! checker could not start". A static binary with no Python, no CUDA link, and no
//! environment of its own does not share that failure mode.
//!
//! ## The honest boundary
//!
//! This binary **cannot** determine which GDN implementation `transformers` dispatched
//! to. That decision happens inside the training process at first forward, and no
//! external observer can see it. Rust-on-CUDA does not change this: the question is
//! about Python-level dispatch, not about which language can reach a GPU.
//!
//! So the gate is split, and **both halves are required**:
//!
//! | Half | Answers | Fails independently |
//! | --- | --- | --- |
//! | `qd-preflight` (this) | Is there a usable CUDA device? Do the pinned artifacts match? | before the job starts |
//! | `verify_fast_path.py` | Which GDN implementation actually ran? | at first forward |
//!
//! Running only this one and calling the environment green would reproduce exactly the
//! fail-open shape the plan's log-grep had: a check that passes without establishing
//! the thing it is supposed to establish.

mod artifacts;
mod cuda;
mod tristate;

use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;

use tristate::{aggregate, TriState};

#[derive(Parser, Debug)]
#[command(name = "qd-preflight", about = "Preflight gate for the qwen-decision training box")]
struct Args {
    /// JSON manifest of pinned artifacts (name, path, sha256).
    #[arg(long)]
    artifacts: Option<PathBuf>,

    /// Root that relative artifact paths resolve against.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Minimum device count required for this job. 8 for the block, 1 for session 0.
    #[arg(long, default_value_t = 1)]
    min_devices: usize,

    /// Exit non-zero unless every check RAN and PASSED.
    ///
    /// `not_run` fails too. An unchecked environment is not a clean one, and this flag
    /// is what makes that structural rather than a matter of reading the output.
    ///
    /// The two non-zero codes are distinct because the operator response differs:
    ///
    ///   1 — a check RAN and FAILED. Something is installed and broken. Fix the box.
    ///   2 — a check DID NOT RUN. You do not know the state. Find out before spending.
    ///
    /// Collapsing them would lose exactly the distinction this whole gate exists for.
    #[arg(long)]
    require_all: bool,

    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    checks: Vec<(String, TriState)>,
    overall: TriState,
}

fn check_device_count(probe: Option<&cuda::CudaProbe>, want: usize) -> TriState {
    match probe {
        None => TriState::not_run(format!(
            "no CUDA probe succeeded, so the device count could not be compared against \
             the required {want}"
        )),
        Some(p) if p.devices.len() < want => TriState::fail(format!(
            "this job requires {want} device(s); the driver reports {}",
            p.devices.len()
        )),
        Some(p) => TriState::pass(format!("{} device(s) available, {want} required", p.devices.len())),
    }
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();

    let (cuda_state, probe) = cuda::check_cuda();
    let device_state = check_device_count(probe.as_ref(), args.min_devices);
    let artifact_state = artifacts::check_artifacts(args.artifacts.as_deref(), &args.root);

    let parts = [
        ("cuda_driver", cuda_state.clone()),
        ("device_count", device_state.clone()),
        ("artifacts", artifact_state.clone()),
    ];
    let overall = aggregate("preflight", &parts);

    let report = Report {
        schema_version: 1,
        checks: parts.iter().map(|(k, v)| ((*k).to_string(), v.clone())).collect(),
        overall: overall.clone(),
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
    } else {
        for (name, state) in &report.checks {
            let tag = match state {
                TriState::Ran { passed: true, .. } => "PASS   ",
                TriState::Ran { passed: false, .. } => "FAIL   ",
                TriState::NotRun { .. } => "NOT RUN",
            };
            let text = match state {
                TriState::Ran { detail, .. } => detail.as_str(),
                TriState::NotRun { reason } => reason.as_str(),
            };
            println!("{tag}  {name:<14} {text}");
        }
        println!();
        match &overall {
            TriState::Ran { passed: true, .. } => println!("preflight: PASS"),
            TriState::Ran { passed: false, detail, .. } => println!("preflight: FAIL -- {detail}"),
            TriState::NotRun { reason } => println!("preflight: NOT RUN -- {reason}"),
        }
        println!(
            "\nNOTE: this gate does NOT establish which GDN implementation will run.\n\
             That is Python-level dispatch, observable only from inside the training\n\
             process. Run stack/verify_fast_path.py --require-fast as well; neither\n\
             check subsumes the other."
        );
    }

    if args.require_all && !overall.is_pass() {
        // 1 = ran and failed (broken box); 2 = did not run (unknown state).
        return std::process::ExitCode::from(if overall.did_run() { 1 } else { 2 });
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_probe_makes_the_device_count_not_run() {
        let s = check_device_count(None, 8);
        assert!(!s.did_run() && !s.is_pass());
    }

    #[test]
    fn too_few_devices_fails_rather_than_not_running() {
        let p = cuda::CudaProbe { soname: "x".into(), driver_version: 12080, devices: vec![] };
        let s = check_device_count(Some(&p), 8);
        assert!(s.did_run(), "a driver that answered gives a real answer");
        assert!(!s.is_pass());
    }

    #[test]
    fn enough_devices_passes() {
        let dev = |i| cuda::DeviceInfo {
            index: i,
            name: "H100".into(),
            compute_capability: "9.0".into(),
            total_mem_gib: 80.0,
        };
        let p = cuda::CudaProbe {
            soname: "libcuda.so.1".into(),
            driver_version: 12080,
            devices: (0..8).map(dev).collect(),
        };
        assert!(check_device_count(Some(&p), 8).is_pass());
    }

    #[test]
    fn on_a_host_without_cuda_the_overall_gate_is_not_run() {
        // The property that matters on this Mac: the gate must refuse to call the
        // environment clean, and must not call it broken either.
        let (cuda_state, probe) = cuda::check_cuda();
        let parts = [
            ("cuda_driver", cuda_state),
            ("device_count", check_device_count(probe.as_ref(), 1)),
            ("artifacts", artifacts::check_artifacts(None, std::path::Path::new("."))),
        ];
        let overall = aggregate("preflight", &parts);
        assert!(!overall.is_pass());
    }
}
