//! The Rust shard reader hashes and validates a `header.json` exactly as
//! `python/qd_train/artifacts.py::ShardHeader` does, including the three optional pins v5 added:
//! `exclusions_sha256`, `contrast_rows` and `prompt_format`, each hashed only when present (when
//! not 1, for `prompt_format`), so every v4 header still verifies.
//!
//! The v5 headers below were written by Python, not by this crate, so a drift on either side
//! fails here. They are `ShardHeader(...).to_json()` from `python/qd_train/artifacts.py` at the
//! commit that added `prompt_format`, printed with `json.dumps(..., sort_keys=True)`; the field
//! values are synthetic. The v4 header is the tracked `shards-tiny` fixture's, written before any
//! of the three existed.
//!
//! Before this change the reader hashed none of the three, so it refused every v5 header as
//! "modified after it was written".

mod common;

use qd_train::shards::ShardHeader;
use serde_json::{Value, json};

/// `split=train`, `span_collapse_policy=refuse-gold`, `exclusions_sha256`, `contrast_rows` and
/// `prompt_format=2`: every optional pin a v5 train set can carry.
const V5_TRAIN: &str = r#"{"buckets": [1024, 2048], "code_fingerprint": {"mixture.py": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd", "render.py": "abababababababababababababababababababababababababababababababab"}, "contrast_rows": {"count": 2000, "seed": 20260919, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}, "corpus_rev": "92e1c74", "created_at": "2026-10-02T12:00:00+00:00", "data_snapshot_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "uint32", "exclusions_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "format": "qd-shard-v1", "max_seq_len": 2048, "n_sequences": 3, "packed": false, "prompt_format": 2, "remap_hash": "9999999999999999999999999999999999999999999999999999999999999999", "sequence_index_hash": "5555555555555555555555555555555555555555555555555555555555555555", "shard_hash": "d816dbfe587ddde55f0ddc471a4dee232be6bdfa86c836b2afe3048033b6e974", "span_collapse_policy": "refuse-gold", "split": "train", "tokenizer_hash": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927", "total_tokens": 4097, "vocab_size": 248320}"#;

/// A report-only val set of format 2: `prompt_format` without the train-only pins.
const V5_VAL_REPORT_ONLY: &str = r#"{"buckets": [1024, 2048], "code_fingerprint": {"mixture.py": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd", "render.py": "abababababababababababababababababababababababababababababababab"}, "corpus_rev": "92e1c74", "created_at": "2026-10-02T12:00:00+00:00", "data_snapshot_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "uint32", "format": "qd-shard-v1", "max_seq_len": 2048, "n_sequences": 3, "packed": false, "prompt_format": 2, "remap_hash": "9999999999999999999999999999999999999999999999999999999999999999", "report_only": true, "sequence_index_hash": "5555555555555555555555555555555555555555555555555555555555555555", "shard_hash": "6ec84e5f67a901100ed7c779d9f6906bf12cbcd971492511ee0333c3ff5113cd", "split": "val", "tokenizer_hash": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927", "total_tokens": 4097, "vocab_size": 248320}"#;

/// `exclusions_sha256` alone (format 1, no contrast rows).
const V5_EXCLUSIONS_ONLY: &str = r#"{"buckets": [1024, 2048], "code_fingerprint": {"mixture.py": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd", "render.py": "abababababababababababababababababababababababababababababababab"}, "corpus_rev": "92e1c74", "created_at": "2026-10-02T12:00:00+00:00", "data_snapshot_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "uint32", "exclusions_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "format": "qd-shard-v1", "max_seq_len": 2048, "n_sequences": 3, "packed": false, "remap_hash": "9999999999999999999999999999999999999999999999999999999999999999", "sequence_index_hash": "5555555555555555555555555555555555555555555555555555555555555555", "shard_hash": "aae799287ea6e23878bcf284fb85afa6bed3da2ae9f506650f2ed2b5435ec186", "split": "train", "tokenizer_hash": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927", "total_tokens": 4097, "vocab_size": 248320}"#;

/// `contrast_rows` alone (format 1, no exclusion list).
const V5_CONTRAST_ONLY: &str = r#"{"buckets": [1024, 2048], "code_fingerprint": {"mixture.py": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd", "render.py": "abababababababababababababababababababababababababababababababab"}, "contrast_rows": {"count": 7, "seed": 0, "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}, "corpus_rev": "92e1c74", "created_at": "2026-10-02T12:00:00+00:00", "data_snapshot_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "uint32", "format": "qd-shard-v1", "max_seq_len": 2048, "n_sequences": 3, "packed": false, "remap_hash": "9999999999999999999999999999999999999999999999999999999999999999", "sequence_index_hash": "5555555555555555555555555555555555555555555555555555555555555555", "shard_hash": "caeae7c8816438156fd5e7fe6e18ba0dce5940e38379a601ab07639295d1abe1", "split": "train", "tokenizer_hash": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927", "total_tokens": 4097, "vocab_size": 248320}"#;

fn parse(text: &str) -> Value {
    serde_json::from_str(text).expect("a Python-written header is JSON")
}

/// `from_json` accepts `raw` and re-derives the shard_hash Python recorded.
#[track_caller]
fn verifies(raw: &Value) -> ShardHeader {
    let header = ShardHeader::from_json(raw).unwrap_or_else(|e| panic!("refused: {e}"));
    assert_eq!(
        Value::from(header.shard_hash().expect("hashes")),
        raw["shard_hash"],
        "the Rust hash of a Python-written header"
    );
    header
}

/// `from_json` refuses `raw`, and the refusal names `needle`.
#[track_caller]
fn refused(raw: &Value, needle: &str) {
    match ShardHeader::from_json(raw) {
        Err(e) => assert!(e.contains(needle), "{needle:?} not in: {e}"),
        Ok(_) => panic!("a header that must be refused was accepted: {raw}"),
    }
}

#[test]
fn every_python_written_v5_header_verifies_with_the_hash_python_recorded() {
    for text in [V5_TRAIN, V5_VAL_REPORT_ONLY, V5_EXCLUSIONS_ONLY, V5_CONTRAST_ONLY] {
        verifies(&parse(text));
    }
}

#[test]
fn the_v4_fixture_header_still_verifies_as_it_was_written() {
    let path = common::fixture().join("shards").join("train").join("header.json");
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for key in ["prompt_format", "exclusions_sha256", "contrast_rows"] {
        assert!(raw.get(key).is_none(), "the v4 fixture predates {key}");
    }
    verifies(&raw);
}

/// One edit to a parsed header: what it does, and the edit.
type Edit = (&'static str, fn(&mut Value));

/// Each v5 pin is covered by the hash: changed or removed, the header is refused as modified.
#[test]
fn each_v5_pin_is_covered_by_the_hash() {
    // Not vacuous: unedited, the header verifies, so each refusal below is the edit's.
    verifies(&parse(V5_TRAIN));
    let edits: [Edit; 7] = [
        ("prompt_format 2 -> 3", |v| v["prompt_format"] = json!(3)),
        ("prompt_format removed", |v| {
            v.as_object_mut().unwrap().remove("prompt_format");
        }),
        ("prompt_format 2 -> 1", |v| v["prompt_format"] = json!(1)),
        ("exclusions changed", |v| v["exclusions_sha256"] = json!("f".repeat(64))),
        ("exclusions removed", |v| {
            v.as_object_mut().unwrap().remove("exclusions_sha256");
        }),
        ("contrast count changed", |v| v["contrast_rows"]["count"] = json!(1999)),
        ("contrast removed", |v| {
            v.as_object_mut().unwrap().remove("contrast_rows");
        }),
    ];
    for (what, edit) in edits {
        let mut raw = parse(V5_TRAIN);
        edit(&mut raw);
        match ShardHeader::from_json(&raw) {
            Err(e) => assert!(e.contains("modified after it was written"), "{what}: {e}"),
            Ok(_) => panic!("{what}: an edited header verified"),
        }
    }
}

/// `prompt_format: 1` stated explicitly hashes as absent: format 1 is what every header without
/// the field means.
#[test]
fn an_explicit_prompt_format_of_1_hashes_as_absent() {
    let mut raw = parse(V5_EXCLUSIONS_ONLY);
    raw["prompt_format"] = json!(1);
    verifies(&raw);
}

/// The same refusals `ShardHeader.__post_init__` / `from_json` / `ContrastRows` make, checked
/// before the hash: a malformed pin is refused for what it is, not as a hash mismatch.
#[test]
fn a_malformed_v5_pin_is_refused_as_python_refuses_it() {
    for bad in [json!(0), json!(-1), json!(true), json!("2"), json!(2.0), json!(null)] {
        let mut raw = parse(V5_TRAIN);
        raw["prompt_format"] = bad;
        refused(&raw, "prompt_format");
    }

    let mut raw = parse(V5_TRAIN);
    raw["exclusions_sha256"] = json!("E".repeat(64));
    refused(&raw, "not a lower-case sha256");
    let mut raw = parse(V5_VAL_REPORT_ONLY);
    raw["exclusions_sha256"] = json!("e".repeat(64));
    refused(&raw, "removes train rows only");

    for (field, bad, needle) in [
        ("count", json!(0), "not a positive int"),
        ("count", json!(true), "not a positive int"),
        ("count", json!(3.0), "not a positive int"),
        ("sha256", json!("A".repeat(64)), "not a lower-case sha256"),
        ("seed", json!(-1), "not a non-negative int"),
        ("seed", json!(false), "not a non-negative int"),
    ] {
        let mut raw = parse(V5_TRAIN);
        raw["contrast_rows"][field] = bad;
        refused(&raw, needle);
    }
    let mut raw = parse(V5_TRAIN);
    raw["contrast_rows"]["extra"] = json!(1);
    refused(&raw, "{count, sha256, seed}");
    let mut raw = parse(V5_TRAIN);
    raw["contrast_rows"] = json!([2000]);
    refused(&raw, "{count, sha256, seed}");
    let mut raw = parse(V5_VAL_REPORT_ONLY);
    raw["contrast_rows"] = json!({"count": 7, "sha256": "b".repeat(64), "seed": 0});
    refused(&raw, "train rows only");
}
