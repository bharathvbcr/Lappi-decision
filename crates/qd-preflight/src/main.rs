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
//! | `qd-preflight` (this) | Is there a usable CUDA device, is it big enough, do the pinned artifacts match? | before the job starts |
//! | `verify_fast_path.py` | Which GDN implementation actually ran? | at first forward |
//!
//! ## "Big enough" was added on 2026-09-20, and why it was missing
//!
//! `cuda.rs` has always read each device's `total_mem_gib` (`:174`). It spent it on a
//! display string (`:217`) and compared it to nothing, so this gate counted devices and
//! never sized them — a 24 GiB A10 passed a gate for a job whose static state alone needs
//! 40 GiB, and the first thing to notice was the allocator, minutes into a billed run.
//! `--min-memory-gib` is the comparison. The requirement itself is arithmetic and lives in
//! `python/qd_train/memory.py`; `python tools/memory_budget.py` prints the number to pass
//! here. Splitting it that way keeps this binary free of Python and of the model.
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

    /// Memory this job needs on EVERY device, in GiB.
    ///
    /// Counting devices is not checking them. Before this flag existed the probe read
    /// `total_mem_gib` off each card (`cuda.rs:174`), spent it on a display string
    /// (`cuda.rs:217`) and compared it to nothing — so a 24 GiB A10 passed a gate for a
    /// job whose static state alone needs 40 GiB, and the first thing that noticed was
    /// the allocator, on a box already being billed.
    ///
    /// There is no default. An absent requirement reports `not_run`, never `pass`: a
    /// budget nobody stated has not been checked, and `--require-all` exits 2 on it,
    /// which is the "you do not know the state" code and the right one.
    ///
    /// `python tools/memory_budget.py` computes the number to pass here.
    #[arg(long)]
    min_memory_gib: Option<f64>,

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

/// Does every device hold at least `want` GiB?
///
/// The smallest card decides, not the average and not the largest: a job is placed on one
/// device, and a box whose cards differ is a box where "it fits" depends on which card the
/// scheduler picked. Reporting the minimum turns that into a property of the box.
///
/// `want` of `None` is `not_run`. That is the whole point of the check — a memory budget
/// nobody supplied is not a memory budget that passed, and the state this crate's own
/// docstring warns about ("a check that passes without establishing the thing it is
/// supposed to establish") is exactly what returning `pass` here would reintroduce.
fn check_device_memory(probe: Option<&cuda::CudaProbe>, want: Option<f64>) -> TriState {
    let Some(want) = want else {
        return TriState::not_run(
            "no --min-memory-gib was given, so no device's memory was compared against \
             anything. Counting devices does not check them; pass the requirement (see \
             `python tools/memory_budget.py`) to turn this into a real check."
                .to_string(),
        );
    };
    if !want.is_finite() || want <= 0.0 {
        return TriState::fail(format!(
            "--min-memory-gib must be a positive, finite number of GiB; got {want}"
        ));
    }
    match probe {
        None => TriState::not_run(format!(
            "no CUDA probe succeeded, so no device's memory could be compared against the \
             required {want:.1} GiB"
        )),
        Some(p) if p.devices.is_empty() => TriState::fail(format!(
            "the driver reports zero devices, so nothing can hold the required {want:.1} GiB"
        )),
        Some(p) => {
            let mut short: Vec<String> = Vec::with_capacity(p.devices.len());
            let mut smallest = f64::INFINITY;
            for d in &p.devices {
                smallest = smallest.min(d.total_mem_gib);
                if d.total_mem_gib < want {
                    short.push(format!(
                        "device {} ({}) has {:.1} GiB, {:.1} GiB short",
                        d.index,
                        d.name,
                        d.total_mem_gib,
                        want - d.total_mem_gib
                    ));
                }
            }
            if short.is_empty() {
                TriState::pass(format!(
                    "every one of {} device(s) holds at least {:.1} GiB; smallest is {:.1} GiB",
                    p.devices.len(),
                    want,
                    smallest
                ))
            } else {
                TriState::fail(format!(
                    "{} of {} device(s) cannot hold this job's {:.1} GiB: {}. Refused here \
                     rather than by the allocator once the box is being billed.",
                    short.len(),
                    p.devices.len(),
                    want,
                    short.join("; ")
                ))
            }
        }
    }
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();

    let (cuda_state, probe) = cuda::check_cuda();
    let device_state = check_device_count(probe.as_ref(), args.min_devices);
    let memory_state = check_device_memory(probe.as_ref(), args.min_memory_gib);
    let artifact_state = artifacts::check_artifacts(args.artifacts.as_deref(), &args.root);

    let parts = [
        ("cuda_driver", cuda_state.clone()),
        ("device_count", device_state.clone()),
        ("device_memory", memory_state.clone()),
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

    fn probe_of(n: usize, gib: f64, name: &str) -> cuda::CudaProbe {
        let dev = |i: usize| cuda::DeviceInfo {
            index: i as i32,
            name: name.into(),
            compute_capability: "8.6".into(),
            total_mem_gib: gib,
        };
        cuda::CudaProbe {
            soname: "libcuda.so.1".into(),
            driver_version: 12080,
            devices: (0..n).map(dev).collect(),
        }
    }

    #[test]
    fn a_card_too_small_for_the_job_is_refused_before_the_money_is_spent() {
        // One A10 at 24 GiB against a job whose static state alone needs 40 GiB.
        let p = probe_of(1, 24.0, "NVIDIA A10");
        let s = check_device_memory(Some(&p), Some(40.0));
        assert!(s.did_run(), "a driver that answered gives a real answer, never not_run");
        assert!(
            !s.is_pass(),
            "24.0 GiB cannot hold a job that needs 40.0 GiB, and a gate that says it can \
             is the gate this check exists to replace"
        );
    }

    #[test]
    fn the_whole_gate_refuses_a_box_whose_cards_are_too_small() {
        let p = probe_of(1, 24.0, "NVIDIA A10");
        let parts = [
            ("cuda_driver", TriState::pass("driver ok")),
            ("device_count", check_device_count(Some(&p), 1)),
            ("device_memory", check_device_memory(Some(&p), Some(40.0))),
            ("artifacts", TriState::pass("pinned")),
        ];
        let overall = aggregate("preflight", &parts);
        assert!(overall.did_run(), "every input ran, so the aggregate must too");
        assert!(
            !overall.is_pass(),
            "device_count alone passes this box -- that is exactly why counting devices \
             is not checking them"
        );
        assert!(
            check_device_count(Some(&p), 1).is_pass(),
            "the device-count check is unchanged and still passes; the refusal is the new \
             check's, not a regression in the old one"
        );
    }

    #[test]
    fn a_card_large_enough_passes_and_names_the_smallest() {
        let p = probe_of(1, 96.0, "NVIDIA GH200");
        let s = check_device_memory(Some(&p), Some(40.0));
        assert!(s.is_pass());
        match s {
            TriState::Ran { detail, .. } => assert!(
                detail.contains("96.0"),
                "the passing detail must name the smallest card, got {detail:?}"
            ),
            TriState::NotRun { .. } => panic!("expected Ran"),
        }
    }

    #[test]
    fn the_smallest_card_decides_not_the_largest() {
        // A box with one big card and one small one cannot run a job that needs the big
        // one, because nothing here chooses which card the job lands on.
        let mut p = probe_of(1, 80.0, "NVIDIA H100");
        p.devices.push(cuda::DeviceInfo {
            index: 1,
            name: "NVIDIA A10".into(),
            compute_capability: "8.6".into(),
            total_mem_gib: 24.0,
        });
        let s = check_device_memory(Some(&p), Some(40.0));
        assert!(s.did_run() && !s.is_pass(), "the 24 GiB card must fail the box");
    }

    #[test]
    fn an_absent_memory_requirement_does_not_run_rather_than_passing() {
        let p = probe_of(8, 80.0, "NVIDIA H100");
        let s = check_device_memory(Some(&p), None);
        assert!(!s.did_run(), "an unstated budget was not checked");
        assert!(!s.is_pass(), "and must never read as one");
        // And it blocks the aggregate, so `--require-all` exits 2 (unknown), not 0.
        let parts = [
            ("cuda_driver", TriState::pass("driver ok")),
            ("device_count", check_device_count(Some(&p), 8)),
            ("device_memory", s),
            ("artifacts", TriState::pass("pinned")),
        ];
        let overall = aggregate("preflight", &parts);
        assert!(!overall.did_run() && !overall.is_pass());
    }

    #[test]
    fn a_nonsense_requirement_fails_rather_than_being_ignored() {
        let p = probe_of(1, 80.0, "NVIDIA H100");
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let s = check_device_memory(Some(&p), Some(bad));
            assert!(s.did_run() && !s.is_pass(), "{bad} must be refused, not treated as 0");
        }
    }

    #[test]
    fn zero_devices_fails_the_memory_check_rather_than_passing_vacuously() {
        // `all()` over an empty list is true; this must not be.
        let p = cuda::CudaProbe {
            soname: "libcuda.so.1".into(),
            driver_version: 12080,
            devices: vec![],
        };
        let s = check_device_memory(Some(&p), Some(40.0));
        assert!(s.did_run() && !s.is_pass());
    }

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
