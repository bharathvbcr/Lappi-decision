//! Adversarial tests for `check_lora_label` (ft-57fc7484 L5, cc60581). Each test states the
//! behaviour a "a row labelled lora carries the evidence" gate must have; a failing test is a
//! break. A lora-labelled row here is built by assignment, never by `lora_ft_row`.

use std::path::PathBuf;

use qd_train::export::METAL_TRAINER;
use qd_train::ledger::{
    self, Environment, FtRecipe, FtRow, Protocol, RowExtras, Status, WallClockSource,
};
use qd_train::tristate::TriState;

const SEVEN: [&str; 7] = [
    "train.span_head_init_digest",
    "lora.d17_init_digest",
    "lora.a_init_digest",
    "lora.base_safetensors_sha256",
    "lora.dtype_map_sha256",
    "corpus.plan_seed",
    "lora.param_groups",
];

fn row(label: &str) -> FtRow {
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
        optimizer_recipe: label.into(),
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
        quick_reason: "adversarial".to_owned(),
        code_commit: "0".repeat(40),
        env: Environment::rust_metal("h".to_owned()),
        metrics: Default::default(),
        wall_clock_s: 1.0,
        wall_clock_source: WallClockSource::Recorder,
        notes: String::new(),
    }
}

fn extras() -> RowExtras {
    RowExtras {
        train_manifest_sha256: Some("c".repeat(64)),
        ..RowExtras::default()
    }
}

fn ledger_path(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qd-adv-lora-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d.join("ledger").join("mac-ojas-adv.jsonl")
}

const ID: &str = "00000000-0000-4000-8000-0000000000b1";
const AT: &str = "2026-10-21T10:00:00+00:00";

/// The presence check accepts any value: a lora row whose seven "digests" are empty strings,
/// non-hex, wrong length, failed (`passed=false`) or null is appended.
#[test]
fn adv_garbage_metric_values_are_refused() {
    let garbage: Vec<(&str, TriState)> = vec![
        ("empty", TriState::ran(true, "")),
        ("non_hex", TriState::ran(true, "zz".repeat(32))),
        ("short", TriState::ran(true, "ab")),
        ("failed", TriState::ran(false, "ab".repeat(32))),
        ("null", TriState::ran(true, serde_json::Value::Null)),
        ("number", TriState::ran(true, 7)),
    ];
    let mut accepted = Vec::new();
    for (name, v) in garbage {
        let mut r = row("lora");
        for m in SEVEN {
            r.metrics.insert(m.to_owned(), v.clone());
        }
        let p = ledger_path(name);
        if ledger::append_with(&p, &r, &extras(), ID.into(), AT.into()).is_ok() {
            accepted.push(name);
        }
    }
    assert!(
        accepted.is_empty(),
        "lora rows with garbage values for all seven metrics were appended: {accepted:?}"
    );
}

/// A NotRun record counts as "present".
#[test]
fn adv_not_run_metrics_are_refused() {
    let mut r = row("lora");
    for m in SEVEN {
        r.metrics
            .insert(m.to_owned(), TriState::not_run("never measured"));
    }
    let p = ledger_path("notrun");
    assert!(
        ledger::append_with(&p, &r, &extras(), ID.into(), AT.into()).is_err(),
        "a lora row whose seven metrics are NotRun was appended"
    );
}

/// `append_line` is `pub` and `FtRow::line_with` is `pub`: the gate lives only in `append_with`.
#[test]
fn adv_append_line_bypass_cannot_write_a_bare_lora_row() {
    let r = row("lora");
    let x = extras();
    let p = ledger_path("bypass");
    let res = ledger::append_line(&p, ID.into(), AT.into(), |s| r.line_with(s, &x));
    assert!(
        res.is_err(),
        "append_line wrote a lora-labelled row with no metrics: the gate is only in append_with"
    );
}

/// The gate keys on exact equality with "lora"; a reader that lowercases/trims sees a lora row.
#[test]
fn adv_label_spelling_variants_are_refused_or_gated() {
    let mut passed = Vec::new();
    for label in ["LORA", "Lora", " lora", "lora ", "lora\n", "l\u{43e}ra"] {
        let r = row(label);
        let p = ledger_path(&format!("v{}", passed.len()));
        if ledger::append_with(&p, &r, &extras(), ID.into(), AT.into()).is_ok() {
            passed.push(label);
        }
    }
    assert!(
        passed.is_empty(),
        "bare rows labelled with a near-spelling of lora were appended unchecked: {passed:?}"
    );
}

/// A non-hex manifest sha on a lora row (all seven metrics present) must still be refused.
#[test]
fn adv_non_hex_manifest_sha_is_refused() {
    let mut r = row("lora");
    for m in SEVEN {
        r.metrics
            .insert(m.to_owned(), TriState::ran(true, "ab".repeat(32)));
    }
    let x = RowExtras {
        train_manifest_sha256: Some("not-a-sha".into()),
        ..RowExtras::default()
    };
    let p = ledger_path("badmanifest");
    assert!(ledger::append_with(&p, &r, &x, ID.into(), AT.into()).is_err());
}
