//! `qd-export --ft-ledger` binds against a ledger row written by the REAL producer, the Python
//! `RunRecorder` with `train_manifest_sha256` set (`python/qd_train/ledger.py`), not a row this
//! suite wrote by hand. `train_manifest_binding.rs` pins the refusals with hand-built rows; this
//! file proves the field name and shape the exporter reads are the ones the writer emits.
//!
//! The recorder is run through `QD_PYTHON`, the repository's Python. Without it the tests fail
//! loudly: a binding that could not be checked against the real writer is not a passing one.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use qd_export::export;
use serde_json::{Value, json};

const WRITE_ROW: &str = r#"
import sys
repo, ledger, sha = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo + "/python")
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder
proto = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                 backbone_commit="b" * 40, recipe_hash="r" * 64, seed=0)
rec = RunRecorder(Ledger(ledger), protocol=proto, run_kind="ft", repo=repo,
                  env=Environment.detect(device="cpu"), wall_clock_s=1.0, cost=None,
                  train_manifest_sha256=(sha or None))
with rec:
    pass
print(rec.row.row_id)
"#;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Run the real recorder: append a completed ft row recording `sha` (none when empty) to `ledger`
/// and return its row id.
fn write_row(ledger: &Path, sha: &str) -> String {
    let python = std::env::var("QD_PYTHON")
        .expect("QD_PYTHON must name the repo's Python: the real writer was not run");
    let out = Command::new(python)
        .arg("-c")
        .arg(WRITE_ROW)
        .arg(repo())
        .arg(ledger)
        .arg(sha)
        .output()
        .expect("python runs");
    assert!(
        out.status.success(),
        "the recorder failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn train_manifest(dir: &Path) -> (PathBuf, String) {
    let path = common::write_train_manifest(&dir.join("train.json"), &common::TRAINED_FAMILIES);
    let sha = qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(&path).unwrap()));
    (path, sha)
}

fn request_for(
    row_id: &str,
    train: &Path,
    ledger: &Path,
) -> (common::Fixture, qd_export::ExportRequest) {
    let fx = common::build(&common::standard_tensors(), &common::tiny_config(), |m| {
        m["ft_row_ids"] = json!([row_id])
    });
    let mut req = fx.request();
    req.train_manifest = train.to_path_buf();
    req.ft_ledger = Some(ledger.to_path_buf());
    (fx, req)
}

#[test]
fn a_row_the_real_recorder_wrote_binds_the_train_manifest() {
    let dir = common::TempDir::new("real-producer");
    let (train, sha) = train_manifest(&dir.0);
    let ledger = dir.0.join("ledger.jsonl");
    let row_id = write_row(&ledger, &sha);
    let on_disk: Value = serde_json::from_str(
        std::fs::read_to_string(&ledger)
            .unwrap()
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        on_disk["train_manifest_sha256"],
        json!(sha),
        "the writer's own field name and value"
    );

    let (fx, req) = request_for(&row_id, &train, &ledger);
    export(&req).expect("the exporter binds against the producer's row");
    let m: Value =
        serde_json::from_slice(&std::fs::read(fx.out.join("release_manifest.json")).unwrap())
            .unwrap();
    let b = &m["train_manifest"]["ledger_binding"];
    assert_eq!(b["verified"], json!(true));
    assert_eq!(b["train_manifest_sha256"], json!(sha));
    assert_eq!(b["ft_row_ids"], json!([row_id]));
}

#[test]
fn a_real_row_that_recorded_another_manifest_is_refused() {
    let dir = common::TempDir::new("real-producer-other");
    let (train, _) = train_manifest(&dir.0);
    let ledger = dir.0.join("ledger.jsonl");
    let row_id = write_row(&ledger, &"0".repeat(64));
    let (fx, req) = request_for(&row_id, &train, &ledger);
    let err = export(&req).expect_err("another manifest's sha");
    assert!(
        err.detail.contains(&row_id) && err.detail.contains("train_manifest_sha256"),
        "{}",
        err.detail
    );
    assert!(!fx.out.exists());
}

#[test]
fn a_real_row_that_recorded_no_manifest_is_refused_as_not_recorded() {
    let dir = common::TempDir::new("real-producer-none");
    let (train, _) = train_manifest(&dir.0);
    let ledger = dir.0.join("ledger.jsonl");
    let row_id = write_row(&ledger, "");
    let (fx, req) = request_for(&row_id, &train, &ledger);
    let err = export(&req).expect_err("no recorded sha");
    assert!(
        err.detail.contains("no train_manifest_sha256"),
        "{}",
        err.detail
    );
    assert!(!fx.out.exists());
}
