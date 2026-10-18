//! `qd-release-check check` has a `ledger_binding` line: a release whose train manifest was not
//! verified against the ft ledger's recorded `train_manifest_sha256` does not pass, and a release
//! with no `ledger_binding` block at all (every release exported before W7-a) does not either.
//! "Not checked" never reads as a pass where a release is decided. The refusal is added beside
//! identity, coverage, replay path and documented examples; nothing here emits a pass.
//!
//! Every case asserts on the `ledger_binding` line itself, so the other checks' verdicts (the
//! results file here holds only a header) cannot hide or fake it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_qd-release-check");
const SHA: &str = "6c2afc81c3d3d1e6a0b1c7e2f4a5b6c7d8e9f00112233445566778899aabbccd";

static N: AtomicU64 = AtomicU64::new(0);

fn tmp() -> PathBuf {
    let p = std::env::temp_dir().join(format!("qd-rcb-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn release(dir: &Path, train_manifest: Option<Value>) {
    std::fs::create_dir_all(dir).unwrap();
    let mut m = json!({
        "format": "qd-release.v2",
        "trained_families": ["fam.b"],
        "excluded_families": [],
        "trained_slot_shapes": {"fam.b": [["choice:4"]]},
        "expected_identity": {"weight_hash": "w", "calibration_hash": "c"},
    });
    if let Some(t) = train_manifest {
        m["train_manifest"] = t;
    }
    std::fs::write(dir.join("release_manifest.json"), m.to_string()).unwrap();
}

/// The `ledger_binding` line of `check`'s output, and the exit code.
fn binding_line(train_manifest: Option<Value>) -> (Option<i32>, String) {
    let d = tmp();
    release(&d.join("rel"), train_manifest);
    let results = d.join("results.jsonl");
    std::fs::write(&results, json!({"kind": "header", "format": "qd-replay.v1", "weight_hash": "w", "calibration_hash": "c"}).to_string() + "\n").unwrap();
    let out = Command::new(BIN)
        .args(["check", "--release"]).arg(d.join("rel"))
        .arg("--results").arg(&results)
        .arg("--min-answered").arg("1")
        .arg("--repo").arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let line = stdout.lines().find(|l| l.contains("ledger_binding")).unwrap_or("").to_string();
    (out.status.code(), line)
}

#[test]
fn a_release_bound_to_the_ledger_passes_the_line() {
    let (_, line) = binding_line(Some(json!({"ledger_binding": {"verified": true, "train_manifest_sha256": SHA}})));
    assert!(line.starts_with("Pass ledger_binding"), "{line:?}");
}

#[test]
fn a_release_exported_without_a_ledger_fails_the_line_and_names_the_reason() {
    let (code, line) = binding_line(Some(json!({"ledger_binding": {"verified": false, "reason": "no --ft-ledger was given"}})));
    assert!(line.starts_with("Fail ledger_binding"), "{line:?}");
    assert!(line.contains("no --ft-ledger was given"), "{line:?}");
    assert_eq!(code, Some(1));
}

#[test]
fn a_release_with_no_ledger_binding_block_fails_the_line() {
    for tm in [None, Some(json!({"sha256": SHA}))] {
        let (code, line) = binding_line(tm);
        assert!(line.starts_with("Fail ledger_binding"), "{line:?}");
        assert!(line.contains("no ledger_binding"), "{line:?}");
        assert_eq!(code, Some(1));
    }
}

#[test]
fn verified_true_without_the_recorded_sha_is_not_a_binding() {
    let (_, line) = binding_line(Some(json!({"ledger_binding": {"verified": true}})));
    assert!(line.starts_with("Fail ledger_binding"), "{line:?}");
}
