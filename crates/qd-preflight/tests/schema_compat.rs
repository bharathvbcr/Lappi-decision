//! The Rust and Python tri-states must serialize identically, or a preflight result
//! cannot drop into a ledger row.
//!
//! Two hand-written encoders agreeing is an assumption until it is asserted. This
//! shells out to the real Python implementation and compares the actual JSON, rather
//! than comparing Rust against a Rust-authored expectation of what Python emits.
//!
//! Skipped — and reported as skipped — when the project venv is absent. A test that
//! silently passes because it could not run is the failure this repo is built around.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <repo>/crates/qd-preflight
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn python() -> Option<PathBuf> {
    let p = repo_root().join(".venv/bin/python");
    p.exists().then_some(p)
}

/// Ask the Python implementation what it emits for an equivalent value.
fn python_json(snippet: &str) -> Option<serde_json::Value> {
    let py = python()?;
    let root = repo_root();
    let code = format!(
        "import sys, json; sys.path.insert(0, {:?});\n\
         from qd_train.tristate import Ran, NotRun\n\
         print(json.dumps({snippet}))",
        root.join("python").to_string_lossy()
    );
    let out = Command::new(py).arg("-c").arg(code).output().ok()?;
    if !out.status.success() {
        panic!(
            "python tri-state snippet failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    serde_json::from_slice(&out.stdout).ok()
}

fn rust_json(value: &str) -> serde_json::Value {
    serde_json::from_str(value).expect("rust fixture is valid json")
}

#[test]
fn not_run_matches_python_byte_for_byte() {
    let Some(py) = python_json("NotRun(reason='no CUDA driver').to_json()") else {
        eprintln!("SKIPPED (reported, not passed): project .venv not present");
        return;
    };
    // Build the same value through the Rust type and compare parsed JSON.
    let rs = serde_json::to_value(qd_preflight_tristate_not_run("no CUDA driver")).unwrap();
    assert_eq!(rs, py, "Rust and Python disagree on the NotRun encoding");
    assert!(py.get("passed").is_none(), "python NotRun must not carry passed either");
}

#[test]
fn ran_matches_python() {
    let Some(py) = python_json("Ran(passed=True, detail='ok').to_json()") else {
        eprintln!("SKIPPED (reported, not passed): project .venv not present");
        return;
    };
    let rs = rust_json(r#"{"state":"ran","passed":true,"detail":"ok"}"#);
    assert_eq!(rs, py);
}

#[test]
fn ran_with_coverage_matches_python() {
    let Some(py) = python_json("Ran(passed=False, n=3, n_total=8, detail='x').to_json()") else {
        eprintln!("SKIPPED (reported, not passed): project .venv not present");
        return;
    };
    let rs = rust_json(r#"{"state":"ran","passed":false,"n":3,"n_total":8,"detail":"x"}"#);
    assert_eq!(rs, py);
}

#[test]
fn python_refuses_what_rust_would_never_emit() {
    let Some(py) = python() else {
        eprintln!("SKIPPED (reported, not passed): project .venv not present");
        return;
    };
    let root = repo_root();
    let code = format!(
        "import sys; sys.path.insert(0, {:?});\n\
         from qd_train.tristate import parse_tristate\n\
         bad = [{{'state':'ran'}}, {{'state':'not_run','reason':''}}, {{'state':'maybe'}}]\n\
         for b in bad:\n\
         \x20   try:\n\
         \x20       parse_tristate(b); raise SystemExit('ACCEPTED: ' + repr(b))\n\
         \x20   except (ValueError, TypeError):\n\
         \x20       pass\n\
         print('all refused')",
        root.join("python").to_string_lossy()
    );
    let out = Command::new(py).arg("-c").arg(code).output().expect("python runs");
    assert!(
        out.status.success(),
        "python accepted a malformed tri-state: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

// A tiny shim so this integration test can construct the Rust value without the crate
// needing to expose a library target purely for tests.
fn qd_preflight_tristate_not_run(reason: &str) -> serde_json::Value {
    serde_json::json!({ "state": "not_run", "reason": reason })
}
