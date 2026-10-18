//! `ledger::append` is `append_with` over empty extras: the two write the same line, and extras
//! add exactly their field.

use qd_train::export::METAL_TRAINER;
use qd_train::ledger::{
    self, Environment, FtRecipe, FtRow, Protocol, RowExtras, Status, WallClockSource,
};

fn row() -> FtRow {
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
    FtRow {
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
    }
}

fn written(name: &str, extras: Option<&RowExtras>) -> String {
    let dir = std::env::temp_dir().join(format!("qd-append-with-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("ledger").join("mac-ojas-w32.jsonl");
    let id = "00000000-0000-4000-8000-0000000000a1".to_owned();
    let at = "2026-10-18T10:00:00+00:00".to_owned();
    match extras {
        None => ledger::append(&path, &row(), id, at).unwrap(),
        Some(x) => ledger::append_with(&path, &row(), x, id, at).unwrap(),
    };
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    text
}

#[test]
fn append_and_append_with_empty_extras_write_identical_lines() {
    assert_eq!(
        written("plain", None),
        written("empty", Some(&RowExtras::default()))
    );
}

#[test]
fn append_with_extras_adds_the_field_and_keeps_the_path_rule() {
    let sha = "ab".repeat(32);
    let x = RowExtras {
        train_manifest_sha256: Some(sha.clone()),
        ..RowExtras::default()
    };
    let v: serde_json::Value = serde_json::from_str(written("extras", Some(&x)).trim()).unwrap();
    assert_eq!(v["train_manifest_sha256"].as_str(), Some(sha.as_str()));
    let bad = std::env::temp_dir().join("not-a-ledger.jsonl");
    assert!(ledger::append_with(&bad, &row(), &x, "id".into(), "t".into()).is_err());
}
