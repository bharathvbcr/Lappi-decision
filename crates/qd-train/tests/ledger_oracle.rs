//! The Rust `ft` row is Python's row, byte for byte: `tools/qd_train_oracle_trainer.py` wrote
//! two rows with `Ledger.append` (re-read by `LedgerRow.from_json` and chain-verified there);
//! this writes the same two rows through `qd_train::ledger` and compares the lines.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use qd_train::ledger::{self, Environment, FtRecipe, FtRow, Protocol, Stamp, Status, TriState, WallClockSource};
use qd_train::pyjson::Json;
use qd_train::run_control::{LossLog, LossPoint};
use qd_train::trainer::{BatchCounts, Termination, TrainResult};

fn recipe() -> FtRecipe {
    FtRecipe {
        tag: "epoch".into(),
        device: "metal".into(),
        lr: 1e-5,
        passes: 1,
        batches: 3,
        width: 1625,
        span_weight: 1.0,
        shard_hash: "d773b87666e1b042279271ab0f891246b7268d4ce0cad2c3e677bb415c147e1a".into(),
        backbone_snapshot: "b1485b2fa6dfa1287294f269f5fb618e03d52d7c".into(),
        backbone_vocab: 248_320,
        backbone_params: 1_881_825_088,
        attn_implementation: "tessl".into(),
        optimizer_recipe: "master".into(),
        wall_clock_cap_s: Some(21_600.0),
        batch_tokens: Some(35_403),
        no_memorise: true,
        beta2: None,
        lower_layers: None,
        provider: "ojas-qwen35 over tessl".into(),
        operands: "bf16".into(),
        optimizer_groups: "single".into(),
        train_attention_mask: "padding".into(),
        deterministic: true,
        extra: BTreeMap::new(),
    }
}

fn result(o: &serde_json::Value, termination: Termination) -> TrainResult {
    let mut log = LossLog::new();
    for p in o["ledger"]["loss_points"].as_array().unwrap() {
        let p = p.as_array().unwrap();
        log.append(LossPoint {
            optimizer_step: p[0].as_u64().unwrap(),
            epoch: p[1].as_u64().unwrap(),
            batch_index: p[2].as_u64().unwrap(),
            loss: common::fhex(&p[3]),
        })
        .unwrap();
    }
    TrainResult {
        termination,
        optimizer_steps: 3,
        micro_batches: 3,
        counts: BatchCounts {
            supervised_tokens: 7,
            span_rows: 2,
            padded_positions: 1200,
            total_positions: 4875,
        },
        loss_log: log,
        steps: Vec::new(),
        channel_log: Vec::new(),
        consumed_digest: "ab".repeat(32),
        consumed_n: 3,
        next_index: 3,
        wall_clock_s: 1234.5,
        final_checkpoint: None,
    }
}

fn row(o: &serde_json::Value, termination: Termination) -> FtRow {
    let r = recipe();
    let protocol = Protocol::for_recipe(
        &r,
        "e9e55ba7b9975872cf1680e71a05f68d5e34fba38aed16987309ea26a7abf225",
        "7fbd94d096a01bca55f22c1852ed490a6c19759560577b1ffb68e207932308a9",
        0,
    )
    .unwrap();
    let mut metrics = ledger::ft_metrics(&result(o, termination), 21_600.0, "ojas-qwen35 over tessl").unwrap();
    metrics.insert(
        "deterministic_kernels".into(),
        TriState::not_run("one run; repeat-run equality was not measured"),
    );
    FtRow {
        protocol,
        recipe: r,
        status: Status::Completed,
        quick_reason: "Metal/tessl trainer, 1 seed, oracle fixture".into(),
        code_commit: format!("593b568{}", "0".repeat(33)),
        env: Environment::rust_metal("oracle-host".into()),
        metrics,
        wall_clock_s: 1234.5,
        wall_clock_source: WallClockSource::Recorder,
        notes: "crates/qd-train-metal rung (d) oracle row".into(),
    }
}

#[test]
fn the_recipe_and_protocol_hash_as_python_hashes_them() {
    let o = common::oracle();
    let r = row(&o, Termination::StepsExhausted);
    assert_eq!(r.protocol.recipe_hash, o["ledger"]["recipe_hash"].as_str().unwrap());
    assert_eq!(r.protocol.hash().unwrap(), o["ledger"]["protocol_hash"].as_str().unwrap());
}

#[test]
fn rows_are_pythons_lines_and_chain_as_python_chains_them() {
    let o = common::oracle();
    let lines: Vec<&str> = o["ledger"]["lines"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
    let first = row(&o, Termination::StepsExhausted)
        .line(&Stamp {
            row_id: "00000000-0000-4000-8000-000000000000".into(),
            written_at: "2026-10-01T18:02:09.195699+00:00".into(),
            prev_row_hash: None,
        })
        .unwrap();
    assert_eq!(first, lines[0]);
    // The second row ended on the cap: `_quick_if_truncated`'s reason is appended, and its
    // predecessor is the first line's sha256.
    let prev = {
        use sha2::Digest;
        let d = sha2::Sha256::digest(lines[0].as_bytes());
        d.iter().map(|b| format!("{b:02x}")).collect::<String>()
    };
    let second = row(&o, Termination::WallClockCap)
        .line(&Stamp {
            row_id: "00000000-0000-4000-8000-000000000001".into(),
            written_at: "2026-10-01T18:02:09.195699+00:00".into(),
            prev_row_hash: Some(prev),
        })
        .unwrap();
    assert_eq!(second, lines[1]);
}

#[test]
fn append_chains_refuses_duplicates_and_writes_only_mac_ojas_ledgers() {
    let o = common::oracle();
    let dir = std::env::temp_dir().join(format!("qd-train-ledger-{}", std::process::id()));
    let path: PathBuf = dir.join("ledger").join("mac-ojas-test-2026-10-01.jsonl");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    let r = row(&o, Termination::StepsExhausted);
    let s1 = ledger::append(&path, &r, "00000000-0000-4000-8000-000000000000".into(), "2026-10-01T18:02:09.195699+00:00".into()).unwrap();
    assert_eq!(s1.prev_row_hash, None);
    let s2 = ledger::append(&path, &r, "00000000-0000-4000-8000-000000000009".into(), "2026-10-01T18:02:09.195699+00:00".into()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let first_line = text.lines().next().unwrap();
    assert_eq!(first_line, o["ledger"]["lines"][0].as_str().unwrap());
    let want = {
        use sha2::Digest;
        sha2::Sha256::digest(first_line.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>()
    };
    assert_eq!(s2.prev_row_hash.as_deref(), Some(want.as_str()));
    assert!(
        ledger::append(&path, &r, "00000000-0000-4000-8000-000000000009".into(), "x".into()).is_err(),
        "a repeated row_id"
    );
    let campaign = dir.join("ledger").join("gh200-p4-v4-2026-10-01.jsonl");
    assert!(ledger::append(&campaign, &r, ledger::uuid4().unwrap(), "x".into()).is_err());
    assert!(!campaign.exists(), "a refused path is not even created");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_row_whose_recipe_hash_is_not_its_recipes_is_refused() {
    let o = common::oracle();
    let mut r = row(&o, Termination::StepsExhausted);
    r.recipe.lr = 2e-5;
    let stamp = Stamp {
        row_id: "x".into(),
        written_at: "x".into(),
        prev_row_hash: None,
    };
    assert!(r.line(&stamp).is_err());
    let mut q = row(&o, Termination::StepsExhausted);
    q.quick_reason = " ".into();
    assert!(q.line(&stamp).is_err(), "every row is quick, and quick needs a reason");
    let mut b = row(&o, Termination::StepsExhausted);
    b.protocol.tokenizer_hash = ledger::NOT_APPLICABLE.into();
    assert!(b.line(&stamp).is_err(), "the build marker in a training row");
    let mut p = row(&o, Termination::StepsExhausted);
    p.recipe.backbone_snapshot = "/Users/bharath/snap".into();
    assert!(p.recipe.to_json().is_err(), "a path is not a revision");
    // The scorer's pairing keys are all present in the stored recipe.
    let Json::Obj(m) = row(&o, Termination::StepsExhausted).recipe.to_json().unwrap() else { panic!() };
    for k in ["tag", "device", "shard_hash", "backbone_snapshot", "attn_implementation", "lr", "span_weight", "optimizer_recipe"] {
        assert!(m.contains_key(k), "recipe lacks {k}");
    }
}
