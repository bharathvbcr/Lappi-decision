//! The ft row's `train_manifest_sha256`: sha256 of the train manifest file's bytes, hashed before
//! and after the read and refused when the two differ (Python's `open_training_data`). CPU only.
//! The row `run()` itself writes is `gpu_tiny.rs`'s `gpu_run_completed_writes_exactly_one_quick_row`;
//! here the functions `run()` builds the row's extras and its read from.

#![cfg(target_os = "macos")]

mod run_v64;

use qd_export::safetensors::sha256_file;
use qd_train::export::METAL_TRAINER;
use qd_train::ledger::{self, Environment, FtRecipe, FtRow, Protocol, Status, WallClockSource};
use qd_train_metal::run::{RunError, read_with_manifest_sha256, row_extras};
use run_v64::{args, fixture, scratch};

fn hex(d: [u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn the_digest_is_the_sha256_of_the_manifest_files_bytes_and_the_read_runs_once() {
    let manifest = fixture().join("data/pool/train.json");
    let mut reads = 0;
    let (value, sha) = read_with_manifest_sha256(&manifest, || {
        reads += 1;
        Ok(7)
    })
    .unwrap();
    assert_eq!((value, reads), (7, 1));
    assert_eq!(sha, hex(sha256_file(&manifest).unwrap()));
    assert_eq!(sha.len(), 64);
}

#[test]
fn a_manifest_that_changes_during_the_read_is_refused() {
    let root = scratch("manifest-changes");
    let manifest = root.join("train.json");
    std::fs::write(&manifest, b"{\"a\":1}").unwrap();
    let err = read_with_manifest_sha256(&manifest, || {
        std::fs::write(&manifest, b"{\"a\":2}").unwrap();
        Ok(())
    })
    .unwrap_err();
    match err {
        RunError::Refused(m) => assert!(m.contains("changed between the hash and the read"), "{m}"),
        other => panic!("wanted Refused, got {other}"),
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_missing_manifest_is_refused_before_the_read_runs() {
    let root = scratch("manifest-missing");
    let err = read_with_manifest_sha256(&root.join("nope.json"), || -> Result<(), RunError> {
        panic!("the read must not run on a manifest that cannot be hashed")
    })
    .unwrap_err();
    assert!(matches!(err, RunError::Refused(_)), "{err}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// The row `run()` appends carries the field: the writer `run()` calls, on a minimal ft row.
#[test]
fn the_appended_row_carries_the_digest_and_equals_the_manifest_hash() {
    let root = scratch("row");
    let a = args(&root, 1);
    let (_, sha) = read_with_manifest_sha256(&a.manifest, || Ok(())).unwrap();
    let recipe = FtRecipe {
        trainer: METAL_TRAINER.into(),
        tag: "epoch".into(),
        device: "metal".into(),
        lr: 1e-3,
        passes: 1,
        batches: 1,
        width: 64,
        span_weight: 1.0,
        shard_hash: "s".repeat(64),
        backbone_snapshot: "qwen35_train".into(),
        backbone_vocab: 64,
        backbone_params: 1,
        attn_implementation: "sdpa".into(),
        optimizer_recipe: "f32".into(),
        wall_clock_cap_s: None,
        batch_tokens: None,
        no_memorise: true,
        beta2: None,
        lower_layers: None,
        provider: "test".into(),
        operands: "exact_f32".into(),
        optimizer_groups: "single".into(),
        train_attention_mask: "padding".into(),
        deterministic: false,
        extra: Default::default(),
    };
    let row = FtRow {
        protocol: Protocol::for_recipe(&recipe, &"d".repeat(64), &"t".repeat(64), 0).unwrap(),
        recipe,
        status: Status::Completed,
        quick_reason: "test".to_owned(),
        code_commit: "0".repeat(40),
        env: Environment::rust_metal("h".to_owned()),
        metrics: Default::default(),
        wall_clock_s: 1.0,
        wall_clock_source: WallClockSource::Recorder,
        notes: String::new(),
    };
    qd_train_metal::run::append_row(
        &a.ledger,
        &row,
        &row_extras(&sha),
        "row-sha-test".to_owned(),
        ledger::isoformat_utc(std::time::SystemTime::now()).unwrap(),
    )
    .unwrap();
    let line = std::fs::read_to_string(&a.ledger).unwrap();
    let v: serde_json::Value = serde_json::from_str(line.lines().next().unwrap()).unwrap();
    assert_eq!(
        v["train_manifest_sha256"].as_str(),
        Some(hex(sha256_file(&a.manifest).unwrap()).as_str())
    );
    std::fs::remove_dir_all(&root).unwrap();
}
