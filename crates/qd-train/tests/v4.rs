//! Opt-in: the real phase-4 v4 shard sets, opened through the door, checked against the
//! counts ledger row `d96409bd` states (`ledger/mac-phase4-v4-shards-2026-10-01.jsonl`).
//!
//! Ignored by default because it needs the local out directory (1.2 GB of tokens and a
//! 152 MB manifest). Run it with the directory named -- an `--ignored` run without it fails
//! rather than passing vacuously:
//!
//! ```text
//! QD_TRAIN_V4_OUT=/Users/bharath/qd-campaign/phase4-v4-2026-10-01 \
//!     cargo test -p qd-train --release --test v4 -- --ignored --nocapture
//! ```
//!
//! The expected numbers are read out of the ledger row, not restated here, so this test
//! checks the row rather than a copy of it. The code fingerprint is checked against this
//! repository's `python/qd_data` with no `allow_stale_code`: a later `qd_data` change that
//! would relabel the set fails here, loudly, as Python's reader would.

mod common;

use std::path::{Path, PathBuf};

use common::{CountingFiles, qd_data_dir, repo_root};
use qd_train::held_out::{DataConfig, DoorError, HeldOutLayer, open_training_manifest};
use qd_train::shards::{ConsumedPrefix, ShardOpen, ShardReader};
use qd_train::supervision::{ft_supervision, plan_span_batch};
use qd_train::tristate::TriState;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ROW_ID: &str = "d96409bd-4890-4d7f-9e7c-ca4cafbdb9a8";
const LEDGER: &str = "ledger/mac-phase4-v4-shards-2026-10-01.jsonl";
/// Python's answers for the same plan, from
/// `tools/qd_train_oracle_shards.py --v4-out <dir>` (2026-10-01, numpy 2.5.0): the full
/// epoch-0 membership hash at `batch_tokens = 2 * max_seq_len`, `DataConfig().seed`, and the
/// consumed digest (`trainer._fold`) over that epoch's first 8 batches.
const PY_V4_BATCHES: usize = 22_713;
const PY_V4_MEMBERSHIP: &str = "20026a67b56deb4bba707ba0fcaceca91c5a7c8b3512b8db23c04153053d0534";
const PY_V4_CONSUMED_8: &str = "26c94be40ebb3bd6969347ae05302b056a4eca9bc40c9a08c6589b8b25fca81d";

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("QD_TRAIN_V4_OUT")
        .map(PathBuf::from)
        .expect(
            "QD_TRAIN_V4_OUT is unset: this test reads the local v4 out directory and does not \
         pass without it",
        );
    assert!(
        dir.join("shards/train/header.json").is_file(),
        "{} holds no v4 shard set",
        dir.display()
    );
    dir
}

fn ledger_row() -> Value {
    let text = std::fs::read_to_string(repo_root().join(LEDGER)).expect("the v4 ledger file");
    text.lines()
        .map(|l| serde_json::from_str::<Value>(l).expect("ledger line"))
        .find(|r| r["row_id"] == ROW_ID)
        .unwrap_or_else(|| panic!("row {ROW_ID} not in {LEDGER}"))
}

fn open(out: &Path, split: &str, rev: &str) -> ShardReader {
    let mut opts = ShardOpen::new(DataConfig::default(), out.to_path_buf(), qd_data_dir());
    opts.expect_rev = Some(rev.to_owned());
    ShardReader::open(
        &out.join("shards").join(split),
        &out.join("data").join("pool").join(format!("{split}.json")),
        &opts,
    )
    .unwrap_or_else(|e| panic!("v4 {split} through the door: {e}"))
}

fn leading_number(text: &str) -> u64 {
    text.chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("no leading number in {text:?}"))
}

fn number_after(text: &str, key: &str) -> u64 {
    leading_number(
        text.split(key)
            .nth(1)
            .unwrap_or_else(|| panic!("{key} not in {text:?}")),
    )
}

fn assert_pass(reader: &ShardReader, name: &str) {
    let check = reader
        .check(name)
        .unwrap_or_else(|| panic!("{name} not recorded"));
    assert!(check.is_pass(), "{name}: {check:?}");
}

#[test]
#[ignore = "opt-in: set QD_TRAIN_V4_OUT to the local phase4-v4 out dir and pass --ignored"]
fn v4_train_and_val_open_through_the_door_with_the_ledger_rows_counts() {
    let out = out_dir();
    let row = ledger_row();
    let rev = row["recipe"]["rev"].as_str().expect("recipe.rev");
    let m = &row["metrics"];

    let train = open(&out, "train", rev);
    for name in [
        "manifest_path_not_held_out",
        "manifest_manifest_split_trainable",
        "manifest_no_held_out_families",
        "manifest_snapshot_status",
        "shard_rev_matches",
        "shard_code_current",
        "shard_split_trainable",
        "shard_bound_to_manifest",
        "shard_remap_matches_header",
    ] {
        assert_pass(&train, name);
    }
    let h = train.header();
    let sequences = &m["shard_sequences"];
    assert_eq!(
        train.len() as u64,
        sequences["n"].as_u64().expect("n"),
        "shard_sequences.n"
    );
    assert_eq!(
        h.total_tokens,
        number_after(sequences["detail"].as_str().expect("d"), "total_tokens=")
    );
    assert_eq!(
        h.max_seq_len,
        m["shard_max_seq_len"]["value"].as_u64().expect("max")
    );
    let buckets: Vec<u64> = serde_json::from_str(
        m["shard_max_seq_len"]["detail"]
            .as_str()
            .expect("d")
            .trim_start_matches("buckets="),
    )
    .expect("bucket list");
    assert_eq!(h.buckets, buckets);
    let slots = &m["shard_slots_written"];
    match train.slot_coverage() {
        TriState::Ran {
            passed,
            coverage: Some(c),
            ..
        } => {
            assert_eq!(
                c.n,
                slots["n"].as_u64().expect("n"),
                "shard_slots_written.n"
            );
            assert_eq!(
                c.n_total,
                slots["n_total"].as_u64().expect("n_total"),
                "shard_slots_written.n_total"
            );
            assert_eq!(*passed, slots["passed"].as_bool().expect("passed"));
        }
        other => panic!("slot coverage: {other:?}"),
    }
    let index = train.sequence_index().expect("v4 pins its sequence index");
    assert_eq!(
        index.rows_in,
        m["shard_rows_written"]["n"].as_u64().expect("rows"),
        "shard_rows_written.n"
    );
    let waste = train.padding_waste().expect("waste");
    let gate = row["gates"]["padding_waste"]["detail"]
        .as_str()
        .expect("gate detail");
    assert_eq!(waste.padded - waste.real, leading_number(gate), "{gate}");
    assert_eq!(waste.padded, number_after(gate, "padding of "), "{gate}");
    assert_eq!(
        waste.tristate().is_pass(),
        row["gates"]["padding_waste"]["passed"]
            .as_bool()
            .expect("p")
    );

    // A real epoch plan, and the first batches read, assembled and supervised.
    let seed = DataConfig::default().seed();
    let batch_tokens = 2 * h.max_seq_len;
    let plans = train.plan(batch_tokens, seed, 0).expect("plan");
    let mut membership = Sha256::new();
    let mut planned = 0usize;
    for p in &plans {
        membership.update((p.bucket as u64).to_le_bytes());
        for r in &p.rows {
            membership.update((*r as u64).to_le_bytes());
        }
        planned += p.rows.len();
    }
    assert_eq!(planned, train.len(), "every sequence once per epoch");
    let mut prefix = ConsumedPrefix::new();
    for (index, plan) in plans.iter().take(8).enumerate() {
        let batch = train.batch(plan, index as u64).expect("batch");
        let sup = ft_supervision(&batch).expect("supervision");
        if let Some(span) = &sup.span {
            plan_span_batch(span).expect("span plan");
        }
        prefix.fold_batch(&batch);
    }
    let membership: String = membership
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "v4 train: {} sequences, {} batches at batch_tokens={batch_tokens} seed={seed} epoch=0; \
         membership sha256 {membership}; consumed digest over the first 8 batches {}",
        train.len(),
        plans.len(),
        prefix.hexdigest()
    );
    assert_eq!(
        plans.len(),
        PY_V4_BATCHES,
        "epoch-0 batch count differs from Python's"
    );
    assert_eq!(
        membership, PY_V4_MEMBERSHIP,
        "epoch-0 plan membership differs from Python's"
    );
    assert_eq!(
        prefix.hexdigest(),
        PY_V4_CONSUMED_8,
        "first-8-batch consumed digest differs"
    );

    let val = open(&out, "val", rev);
    assert_pass(&val, "shard_rev_matches");
    assert_pass(&val, "shard_code_current");
    let vs = &m["val_shard_slots_written"];
    match val.slot_coverage() {
        TriState::Ran {
            coverage: Some(c), ..
        } => {
            assert_eq!(
                c.n,
                vs["n"].as_u64().expect("n"),
                "val_shard_slots_written.n"
            );
            assert_eq!(
                c.n_total,
                vs["n_total"].as_u64().expect("n_total"),
                "val_shard_slots_written.n_total"
            );
        }
        other => panic!("val slot coverage: {other:?}"),
    }
    assert_eq!(
        val.sequence_index().expect("index").rows_in,
        m["val_shard_coverage"]["n"].as_u64().expect("rows"),
        "val_shard_coverage.n"
    );
}

#[test]
#[ignore = "opt-in: set QD_TRAIN_V4_OUT to the local phase4-v4 out dir and pass --ignored"]
fn v4_held_out_manifest_is_refused_before_a_byte_of_it_is_read() {
    let out = out_dir();
    let files = CountingFiles::new();
    let err = open_training_manifest(
        &out.join("data/heldout/heldout.json"),
        &DataConfig::default(),
        &out,
        false,
        files.as_ref(),
    )
    .expect_err("the v4 held-out manifest must be refused");
    assert_eq!(
        files.total(),
        0,
        "bytes of the real held-out manifest were read"
    );
    assert!(
        matches!(err, DoorError::HeldOut(ref v) if v.layer == HeldOutLayer::ConfiguredRoot),
        "{err}"
    );
    // And the real train set under a held-out root is refused before its 1.2 GB are touched.
    let files = CountingFiles::new();
    let mut opts = ShardOpen::new(DataConfig::default(), out.clone(), qd_data_dir());
    opts.config = DataConfig::new(
        DataConfig::default().held_out_families().to_vec(),
        vec![PathBuf::from("shards")],
        DataConfig::default().held_out_path_markers().to_vec(),
        DataConfig::default().seed(),
    )
    .expect("config");
    opts.files = files.clone();
    let err = ShardReader::open(
        &out.join("shards/train"),
        &out.join("data/pool/train.json"),
        &opts,
    )
    .expect_err("a configured root over the shards refuses them");
    assert_eq!(files.total(), 0);
    assert_eq!(
        err.held_out().map(|v| v.layer),
        Some(HeldOutLayer::ConfiguredRoot)
    );
}
