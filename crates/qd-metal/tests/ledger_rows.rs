//! The rows qd-metal writes are rows Python reads: built from synthetic results (no GPU), appended
//! through `qd_metal::ledger::write_row` to a scratch `ledger/mac-qd-metal-*.jsonl`, then
//! verified by `python/qd_train/ledger.py` itself (`Ledger.verify_chain`, which re-validates every
//! row through `LedgerRow.from_json`) and against the check `test_ledger_provenance.py` holds
//! every non-`build` row to (a non-empty `metrics.code_that_ran`).
//!
//! The Python half needs an interpreter that can import `qd_train.ledger` (standard library only):
//! `QD_PYTHON`, else `<repo>/.venv/bin/python`, else `python3`. With none it **fails**. It used to
//! print SKIPPED and return, which libtest records as `ok`: in a worktree without `.venv` and
//! without `QD_PYTHON` the suite read "4 passed" with the Python half never run (observed
//! 2026-10-02 on the c1fa18f binary). `crates/qd-preflight/tests/schema_compat.rs` retired the same
//! pattern for the same reason (CLAUDE.md: a check that could not run must never report the same
//! result as a check that ran and passed).

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use qd_metal::decision::{self, Arm, ArmResult, DecisionArgs, DecisionPrompt, RowTarget, RunContext, Sample, TResult};
use qd_metal::ledger::{self, Provenance};
use qd_metal::model::EmbedPath;
use qd_metal::parity_row::{self, ModelFacts, ParityRun};
use serde_json::{json, Value};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Every interpreter call here is short; one that runs past this is killed and fails the test.
const PYTHON_TIMEOUT: Duration = Duration::from_secs(60);

fn python_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("QD_PYTHON") {
        out.push(PathBuf::from(p));
    }
    out.push(repo().join(".venv/bin/python"));
    out.push(PathBuf::from("python3"));
    out
}

/// Run `cmd` with piped output, killing it past `limit`. The scripts here print a few hundred
/// bytes, well under a pipe buffer, so reading after exit cannot deadlock.
fn output_bounded(cmd: &mut Command, limit: Duration) -> io::Result<Output> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let start = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            return child.wait_with_output();
        }
        if start.elapsed() > limit {
            child.kill()?;
            child.wait()?;
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("interpreter did not exit within {limit:?}"),
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn python_command(py: &Path) -> Command {
    let mut cmd = Command::new(py);
    cmd.env("PYTHONPATH", repo().join("python"));
    cmd
}

/// True when `p` runs and can import the module under test.
fn can_import_ledger(p: &Path) -> bool {
    let mut cmd = python_command(p);
    cmd.arg("-c").arg("import qd_train.ledger");
    matches!(output_bounded(&mut cmd, PYTHON_TIMEOUT), Ok(out) if out.status.success())
}

/// The interpreter the Python half must use, or a panic naming the remedy. Takes the candidate
/// list so the fail-closed path is itself tested against a list known to hold nothing usable.
fn require_python_from(candidates: &[PathBuf]) -> PathBuf {
    match candidates.iter().find(|p| can_import_ledger(p)) {
        Some(p) => p.clone(),
        None => panic!(
            "no interpreter among {candidates:?} can import qd_train.ledger, so Python did NOT \
             read these rows. This fails rather than skipping: a skip is recorded as `ok`. \
             Remedy: point QD_PYTHON at a Python >= 3.11."
        ),
    }
}

fn python() -> PathBuf {
    require_python_from(&python_candidates())
}

/// Run `script` with `path` as `sys.argv[1]`, bounded; a spawn error or a timeout fails the test.
fn python_out(py: &Path, script: &str, path: &Path) -> Output {
    let mut cmd = python_command(py);
    cmd.arg("-c").arg(script).arg(path);
    output_bounded(&mut cmd, PYTHON_TIMEOUT).unwrap_or_else(|e| panic!("could not run {}: {e}", py.display()))
}

fn scratch_ledger(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("qdm-ledger-{tag}-{}", std::process::id()))
        .join("ledger");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("mac-qd-metal-test.jsonl")
}

/// Python's verdict on the file: `verify_chain`, then every row's `code_that_ran`.
fn python_verifies(path: &Path) -> Value {
    let py = python();
    let script = "import json, sys\n\
from qd_train.ledger import Ledger\n\
led = Ledger(sys.argv[1])\n\
led.verify_chain()\n\
rows = led.rows()\n\
out = [{'run_kind': r.run_kind, 'status': r.status, 'quick': r.quick,\n\
        'code_that_ran': bool(json.loads(json.dumps(r.metrics['code_that_ran'].to_json())).get('value'))} for r in rows]\n\
print(json.dumps(out))\n";
    let out = python_out(&py, script, path);
    assert!(
        out.status.success(),
        "Python refused the rows ({}):\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("python printed JSON")
}

fn sample(total: f64, logits: &[f32]) -> Sample {
    Sample {
        total_ms: total,
        prefill_ms: total * 0.6,
        decode_ms: [total * 0.1, total * 0.1],
        digest_ms: [total * 0.1, total * 0.05, total * 0.05],
        counts: tessl::infer_trace::Snapshot {
            dispatches: 900,
            barriers: 800,
            commits: 6,
            cold_allocs: 3,
            sync_wait_us: 1200,
            residency_flushes: 0,
        },
        logits: logits.to_vec(),
        state_digest: [7; 32],
        state_bytes: 32_000_000,
    }
}

fn decision_row(provenance: Provenance, split_logits: bool) -> qd_train::ledger::Row {
    decision_row_for_targets(provenance, split_logits, vec![512])
}

/// One T (512) always runs; `ts` is what the recipe says should have.
fn decision_row_for_targets(provenance: Provenance, split_logits: bool, ts: Vec<usize>) -> qd_train::ledger::Row {
    let args = DecisionArgs {
        ts,
        k: 4,
        iters: 3,
        warmup: 1,
        arms: vec![Arm::Embed(EmbedPath::Host), Arm::Embed(EmbedPath::Device)],
        row: RowTarget::None,
        snapshot: None,
    };
    let base = [0.5f32, -1.0, 2.0, 0.25, -3.0, 0.5, -1.0, 2.0, 0.25, -3.0];
    let mut other = base;
    if split_logits {
        other[3] = f32::from_bits(other[3].to_bits() + 1);
    }
    let t = TResult {
        prompt: DecisionPrompt {
            target: 512,
            context_lines: 30,
            prefix: (0..500).collect(),
            passes: [vec![1, 2, 3], vec![1, 3, 2]],
            rows: 5,
        },
        arms: vec![
            ArmResult { arm: args.arms[0], samples: vec![sample(200.0, &base), sample(190.0, &base), sample(210.0, &base)] },
            ArmResult { arm: args.arms[1], samples: vec![sample(180.0, &other), sample(185.0, &other), sample(175.0, &other)] },
        ],
    };
    let ctx = RunContext {
        device: "Apple M5 Pro".into(),
        snapshot: PathBuf::from("/hf/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
        vocab: 248_320,
        weight_hash: "c".repeat(64),
        tokenizer_hash: "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927".into(),
        load_s: 3.5,
        wall_clock_s: 42.0,
        tessl_after: provenance.tessl.clone(),
        provenance,
    };
    decision::build_row(&args, &[t], &ctx).expect("the decision row builds")
}

fn parity_object(nan: bool) -> String {
    let th = json!({
        "tight_layer_rel_l2_vs_bf16in": 2e-2, "loose_layer_rel_l2_vs_fp32": 5e-2,
        "tight_logprob_abs_vs_bf16in": 0.10, "loose_logprob_abs_vs_fp32": 0.25,
        "argmax_min_agreement": 0.95, "argmax_disagree_max_ref_margin": 0.10,
        "snapshot_path_logprob_abs": 0.05,
    });
    let prompts: Vec<Value> = (0..20)
        .map(|i| json!({"id": format!("p{i}"), "tokens": 300, "prefix_tokens": 280,
            "letter_logprob_maxabs_bf16in": 0.01, "letter_logprob_maxabs_fp32": 0.03,
            "argmax_metal": 2, "argmax_fp32": 2, "ref_margin_fp32": 1.5,
            "snapshot_path_maxabs": if nan && i == 3 { Value::Null } else { json!(0.001) },
            "layers": []}))
        .collect();
    json!({"kind": "qd-metal-parity", "device": "Apple M5 Pro", "weight_hash": "c".repeat(64),
        "tokenizer_hash": "fe000e3e", "fixtures_manifest_sha256": "d".repeat(64),
        "reference": {"torch": "2.12.1", "device": "cpu"}, "n_prompts": 20, "thresholds": th,
        "argmax_agreement": 1.0, "gate": "pass", "failures": [],
        "per_layer_worst": (0..24).map(|l| json!({"layer": l, "rel_l2_bf16in": 0.004, "rel_l2_fp32": 0.006, "max_abs_fp32": 0.5, "ref_max_abs": 40.0})).collect::<Vec<_>>(),
        "prompts": prompts, "row_id": "qdm-parity-0123456789abcdef"})
    .to_string()
}

#[test]
fn decision_and_parity_rows_are_rows_python_verifies() {
    let prov = Provenance::of(None).expect("provenance of this test binary");
    let path = scratch_ledger("rows");
    let d = decision_row(prov.clone(), false);
    assert!(d.metrics["decision.t512.arms_bit_identical"].is_pass());
    let s1 = ledger::write_row(&path, &d).unwrap();
    assert!(s1.prev_row_hash.is_none());

    // A gate that said pass with a NaN in it: the row records the gate as failed.
    let run = ParityRun {
        raw_line: Some(parity_object(true)),
        exit_code: 0,
        stdout: "gate: PASS\n".into(),
        stderr: String::new(),
        wall_clock_s: 61.0,
        argv: vec!["--fixtures".into(), "/f".into()],
    };
    let model = ModelFacts { snapshot_name: "b1485b2fa6dfa1287294f269f5fb618e03d52d7c".into(), vocab: 248_320 };
    let p = parity_row::build_row(&run, &prov, &model).unwrap();
    assert!(p.metrics["parity.gate"].is_fail(), "a NaN must fail the gate whatever it said");
    assert!(p.metrics["parity.non_finite_values"].is_fail());
    assert!(p.metrics["parity.worst_snapshot_path"].is_fail());
    let s2 = ledger::write_row(&path, &p).unwrap();
    assert!(s2.prev_row_hash.is_some());

    // The same gate without the NaN passes.
    let clean = ParityRun { raw_line: Some(parity_object(false)), ..run.clone() };
    let p2 = parity_row::build_row(&clean, &prov, &model).unwrap();
    assert!(p2.metrics["parity.gate"].is_pass());
    // And a gate that could not run is a failed row with the gate not_run, never a pass.
    let not_run = ParityRun { raw_line: None, exit_code: 2, stderr: "gate NOT RUN: no fixtures".into(), ..run };
    let p3 = parity_row::build_row(&not_run, &prov, &model).unwrap();
    assert!(!p3.metrics["parity.gate"].is_pass() && !p3.metrics["parity.gate"].is_fail());
    assert_eq!(p3.status, qd_train::ledger::Status::Failed);
    ledger::write_row(&path, &p2).unwrap();
    ledger::write_row(&path, &p3).unwrap();

    let v = python_verifies(&path);
    let rows = v.as_array().expect("python printed a list of rows");
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0]["run_kind"], "throughput");
    assert_eq!(rows[1]["run_kind"], "smoke");
    assert_eq!(rows[3]["status"], "failed");
    for r in rows {
        assert_eq!(r["quick"], true);
        assert_eq!(r["code_that_ran"], true, "test_ledger_provenance requires it: {r}");
    }
    std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap()).unwrap();
}

/// The negative control for the Python half: an edited row must be refused, or the check above
/// proves nothing.
#[test]
fn python_refuses_a_row_edited_after_it_was_written() {
    let py = python();
    let prov = Provenance::of(None).unwrap();
    let path = scratch_ledger("tamper");
    ledger::write_row(&path, &decision_row(prov, false)).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let edited = text.replacen("\"seed\":0", "\"seed\":1", 1);
    assert_ne!(edited, text, "the edit must change the row");
    std::fs::write(&path, edited).unwrap();
    let out = python_out(
        &py,
        "import sys\nfrom qd_train.ledger import Ledger\nLedger(sys.argv[1]).verify_chain()\n",
        &path,
    );
    assert!(!out.status.success(), "Python accepted a row whose protocol no longer hashes to its protocol_hash");
    assert!(String::from_utf8_lossy(&out.stderr).contains("protocol_hash"), "{}", String::from_utf8_lossy(&out.stderr));
    std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap()).unwrap();
}

/// With no usable interpreter the Python half fails the suite; it never returns as a pass.
#[test]
#[should_panic(expected = "can import qd_train.ledger")]
fn no_usable_python_fails_rather_than_skips() {
    let missing = std::env::temp_dir().join(format!("qdm-no-python-{}", std::process::id()));
    assert!(!missing.exists());
    require_python_from(&[missing]);
}

#[test]
fn arms_that_differ_by_one_bit_are_recorded_as_not_identical() {
    let prov = Provenance::of(None).unwrap();
    let d = decision_row(prov, true);
    let t = &d.metrics["decision.t512.arms_bit_identical"];
    assert!(t.is_fail(), "{t:?}");
    // Fail-first (audit 2026-10-03): until then a row whose arms disagreed still said completed.
    assert_eq!(d.status, qd_train::ledger::Status::Failed, "a non-identical A/B row read as completed");
}

#[test]
fn a_complete_identical_run_is_the_only_completed_row() {
    let prov = Provenance::of(None).unwrap();
    let d = decision_row(prov, false);
    assert_eq!(d.status, qd_train::ledger::Status::Completed);
    assert!(d.metrics["t_coverage"].is_pass(), "{:?}", d.metrics["t_coverage"]);
}

/// Fail-first (audit 2026-10-03): `bench.rs` stops early when tessl moves, and a row built from
/// the Ts that did run used to say completed whenever tessl read unchanged at the end.
#[test]
fn a_run_cut_short_is_a_capped_sample_not_a_completed_row() {
    let prov = Provenance::of(None).unwrap();
    let d = decision_row_for_targets(prov, false, vec![512, 2048]);
    assert_eq!(d.status, qd_train::ledger::Status::Failed, "1 of 2 Ts read as a completed row");
    let c = &d.metrics["t_coverage"];
    assert!(c.is_fail(), "{c:?}");
    assert!(d.notes.contains("Only 1 of 2 Ts ran"), "{}", d.notes);
}

#[test]
fn a_row_never_lands_outside_mac_qd_metal() {
    let prov = Provenance::of(None).unwrap();
    let d = decision_row(prov, false);
    let dir = std::env::temp_dir().join(format!("qdm-ledger-refuse-{}", std::process::id())).join("ledger");
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["runs.jsonl", "mac-ojas-x.jsonl", "gh200-x.jsonl"] {
        assert!(ledger::write_row(&dir.join(name), &d).is_err(), "{name} was written");
        assert!(!dir.join(name).exists());
    }
    std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
}
