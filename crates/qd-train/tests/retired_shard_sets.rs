//! The retired v5r shard sets (contaminated, LoRA-2B amendment 3) cannot be opened from Rust.
//!
//! One registry, two readers: `AUDIT/x5-v5r-retire-2026-10-22/retirement.json` is the list, and
//! `python/tests/test_retired_shard_sets.py` holds Python's constants equal to it exactly as this
//! file holds Rust's. A hand-edit of either constant fails its own test.

use std::collections::BTreeSet;
use std::path::Path;

use qd_train::shards::{RETIRED_DATA_SNAPSHOTS, RETIRED_SHARD_SETS, ShardHeader};
use serde_json::{Value, json};

/// A valid synthetic v5 val header, so only the field a test changes can refuse it. Its
/// `shard_hash` is the one both Python's `ShardHeader.from_json` and this crate derive for these
/// fields (checked against both).
const HEADER: &str = r#"{"buckets": [1024, 2048], "code_fingerprint": {"mixture.py": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd", "render.py": "abababababababababababababababababababababababababababababababab"}, "corpus_rev": "92e1c74", "created_at": "2026-10-02T12:00:00+00:00", "data_snapshot_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "uint32", "format": "qd-shard-v1", "max_seq_len": 2048, "n_sequences": 3, "packed": false, "prompt_format": 2, "remap_hash": "9999999999999999999999999999999999999999999999999999999999999999", "report_only": true, "sequence_index_hash": "5555555555555555555555555555555555555555555555555555555555555555", "shard_hash": "989de8e6bdc1d95c736b449beb71c1dc9cff4135763e9a580bff24e999781c73", "split": "val", "tokenizer_hash": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927", "total_tokens": 4097, "vocab_size": 248320}"#;

fn registry() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../AUDIT/x5-v5r-retire-2026-10-22/retirement.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("the registry is JSON")
}

fn column(registry: &Value, key: &str) -> BTreeSet<String> {
    registry["sets"]
        .as_array()
        .expect("sets is a list")
        .iter()
        .map(|s| s[key].as_str().expect("a string").to_owned())
        .collect()
}

fn header() -> Value {
    serde_json::from_str(HEADER).expect("header JSON")
}

#[test]
fn the_rust_constants_are_the_registry() {
    let reg = registry();
    let shard: BTreeSet<String> = RETIRED_SHARD_SETS.iter().map(|s| (*s).to_owned()).collect();
    let snap: BTreeSet<String> = RETIRED_DATA_SNAPSHOTS.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(shard, column(&reg, "shard_hash"));
    assert_eq!(snap, column(&reg, "data_snapshot_hash"));
    assert_eq!(RETIRED_SHARD_SETS.len(), 3);
}

#[test]
fn the_unretired_header_opens() {
    ShardHeader::from_json(&header()).expect("the synthetic header verifies");
}

#[test]
fn a_retired_shard_hash_is_refused_as_retired() {
    for hash in RETIRED_SHARD_SETS {
        let mut raw = header();
        raw["shard_hash"] = json!(hash);
        let err = ShardHeader::from_json(&raw).expect_err("refused");
        assert!(err.contains("retired: contaminated"), "{err}");
    }
}

#[test]
fn a_retired_data_snapshot_is_refused_whatever_the_shard_hash() {
    for snapshot in RETIRED_DATA_SNAPSHOTS {
        let mut raw = header();
        raw["data_snapshot_hash"] = json!(snapshot);
        // The recorded shard_hash no longer matches either; it is still refused as retired.
        let err = ShardHeader::from_json(&raw).expect_err("refused");
        assert!(err.contains("retired: contaminated"), "{err}");
    }
}
