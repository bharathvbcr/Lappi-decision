//! The reader's code-fingerprint rule follows the header's `code_fingerprint_version`, as
//! `python/qd_train/artifacts.py::assert_shard_trainable` does: rule 1 (absent) hashes every
//! `qd_data/*.py`; rule 2 hashes only the modules `qd_data/fingerprint.py::SHARD_CONTENT_MODULES`
//! lists. The Rust reader reads that tuple from the `qd_data` it is given, so there is no second
//! list to drift.
//!
//! The mini `qd_data` below (`a.py`, `b.py`, an unlisted `c.py`, a `fingerprint.py` listing
//! `a.py` and `b.py`) matches the digests in the header constants, which were written by
//! `ShardHeader(...).to_json()` from the Python writer with `code_fingerprint_version=2` and with
//! rule 1. Before this change the reader hashed every `*.py` for every header, so a v2 header was
//! refused as "added since: c.py" (and its recorded version was never read).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::{copy_dir, fixture, fixture_open, qd_data_dir, scratch};
use qd_train::files::OsFiles;
use qd_train::shards::{
    ShardHeader, ShardReader, code_fingerprint, shard_code_fingerprint, shard_content_modules,
};
use serde_json::{Value, json};

const FP_PY: &[u8] =
    b"\"\"\"mini.\"\"\"\n\nSHARD_CONTENT_MODULES: Final[tuple[str, ...]] = (\n    \"a.py\",\n    \"b.py\",\n)\n";
const A_PY: &[u8] = b"a = 1\n";
const B_PY: &[u8] = b"b = 2\n";
const C_PY: &[u8] = b"c = 3\n";

/// Python-written, `code_fingerprint_version=2`: `{a.py, b.py}`.
const PY_V2: &str = r#"{"buckets": [8], "code_fingerprint": {"a.py": "cb78bd8a17f7b751fe0d4663366dcbc257204033ef7ddd64b1f2969573b5b2e2", "b.py": "17dd98c94951360cbd9b278de6650598290a9f9121367458640c94bfbbd2c037"}, "code_fingerprint_version": 2, "corpus_rev": "92e1c74", "created_at": "2026-10-18T01:47:22.825896+00:00", "data_snapshot_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "uint32", "format": "qd-shard-v1", "max_seq_len": 8, "n_sequences": 3, "packed": false, "remap_hash": "9999999999999999999999999999999999999999999999999999999999999999", "sequence_index_hash": "", "shard_hash": "91cbf91505257a45d1f1745e39a3c4424c5602567a8c00765a00d18cd8dbdacf", "split": "train", "tokenizer_hash": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "total_tokens": 10, "vocab_size": 64}"#;

/// Python-written, rule 1 (no version field): every module of the mini package.
const PY_V1: &str = r#"{"buckets": [8], "code_fingerprint": {"a.py": "cb78bd8a17f7b751fe0d4663366dcbc257204033ef7ddd64b1f2969573b5b2e2", "b.py": "17dd98c94951360cbd9b278de6650598290a9f9121367458640c94bfbbd2c037", "c.py": "1b0c816fe818ab7f866b923a450b5afee2622b53f387fe9277e38ac265d38303", "fingerprint.py": "ee224bb129330207fe8539a09b1703ca1a33c7729f11c90628900b245fd3639f"}, "corpus_rev": "92e1c74", "created_at": "2026-10-18T01:47:22.826231+00:00", "data_snapshot_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "dtype": "uint32", "format": "qd-shard-v1", "max_seq_len": 8, "n_sequences": 3, "packed": false, "remap_hash": "9999999999999999999999999999999999999999999999999999999999999999", "sequence_index_hash": "", "shard_hash": "a855f1adf50feca716ac8050eb5de0be14355ba0b94502cff2f569d28f55357b", "split": "train", "tokenizer_hash": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "total_tokens": 10, "vocab_size": 64}"#;

fn mini_qd_data(tag: &str) -> PathBuf {
    let dir = scratch(tag).join("qd_data");
    std::fs::create_dir_all(&dir).expect("mkdir");
    for (name, bytes) in [
        ("fingerprint.py", FP_PY),
        ("a.py", A_PY),
        ("b.py", B_PY),
        ("c.py", C_PY),
    ] {
        std::fs::write(dir.join(name), bytes).expect("write module");
    }
    dir
}

fn parse(text: &str) -> Value {
    serde_json::from_str(text).expect("JSON")
}

#[test]
fn a_python_written_v2_header_verifies_and_a_v1_header_still_does() {
    for text in [PY_V2, PY_V1] {
        let raw = parse(text);
        let header = ShardHeader::from_json(&raw).unwrap_or_else(|e| panic!("refused: {e}"));
        assert_eq!(Value::from(header.shard_hash().unwrap()), raw["shard_hash"]);
    }
    assert_eq!(
        ShardHeader::from_json(&parse(PY_V2))
            .unwrap()
            .code_fingerprint_version,
        2
    );
    assert_eq!(
        ShardHeader::from_json(&parse(PY_V1))
            .unwrap()
            .code_fingerprint_version,
        1,
        "absent is rule 1"
    );
}

#[test]
fn an_unknown_or_mistyped_version_is_refused_and_the_version_is_covered_by_the_hash() {
    for bad in [json!(3), json!(0), json!("2"), json!(2.0), json!(true)] {
        let mut raw = parse(PY_V2);
        raw["code_fingerprint_version"] = bad;
        let err = ShardHeader::from_json(&raw).expect_err("refused");
        assert!(err.contains("code_fingerprint_version"), "{err}");
    }
    let mut raw = parse(PY_V2);
    raw.as_object_mut()
        .unwrap()
        .remove("code_fingerprint_version");
    let err = ShardHeader::from_json(&raw).expect_err("edited out");
    assert!(err.contains("modified after it was written"), "{err}");
}

#[test]
fn the_module_list_is_read_from_the_package_it_describes() {
    let dir = mini_qd_data("list");
    assert_eq!(shard_content_modules(&dir).unwrap(), ["a.py", "b.py"]);
    let fp = shard_code_fingerprint(&dir).unwrap();
    assert_eq!(fp.keys().collect::<Vec<_>>(), ["a.py", "b.py"]);
    assert_eq!(
        Value::from(fp["a.py"].clone()),
        parse(PY_V2)["code_fingerprint"]["a.py"],
        "the digest Python recorded"
    );
    std::fs::remove_file(dir.join("b.py")).unwrap();
    let err = shard_code_fingerprint(&dir).expect_err("a listed module is missing");
    assert!(err.to_string().contains("b.py"), "{err}");
    std::fs::write(dir.join("fingerprint.py"), b"x = 1\n").unwrap();
    assert!(shard_content_modules(&dir).is_err(), "no tuple, no guess");
}

/// The repository's own list: every entry exists in `python/qd_data`, and it is a strict subset
/// of what rule 1 hashes (the reason rule 2 exists).
#[test]
fn the_real_qd_data_list_parses_and_names_files_that_exist() {
    let dir = qd_data_dir();
    let names = shard_content_modules(&dir).expect("the repository's fingerprint.py parses");
    assert!(names.len() >= 10, "{names:?}");
    let all = code_fingerprint(&dir).unwrap();
    assert!(names.iter().all(|n| all.contains_key(n)), "{names:?}");
    assert!(names.len() < all.len(), "rule 2 is narrower than rule 1");
    assert_eq!(shard_code_fingerprint(&dir).unwrap().len(), names.len());
}

/// The fixture's train set with `header.json` rewritten to `fingerprint` under `version`
/// (rule 1 leaves the field out, as the Python writer does).
fn set_with_fingerprint(tag: &str, version: u64, fingerprint: &Value) -> PathBuf {
    let shards = scratch(tag).join("train");
    copy_dir(&fixture().join("shards").join("train"), &shards);
    let path = shards.join("header.json");
    let mut raw: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    // The reader's own `shard_hash` re-derives the hash for the edited fields; the cross-language
    // agreement of that hash is `a_python_written_v2_header_verifies_and_a_v1_header_still_does`.
    let mut header = ShardHeader::from_json(&raw).expect("the fixture header");
    header.code_fingerprint = serde_json::from_value(fingerprint.clone()).unwrap();
    header.code_fingerprint_version = version;
    raw["code_fingerprint"] = fingerprint.clone();
    if version != 1 {
        raw["code_fingerprint_version"] = json!(version);
    }
    raw["shard_hash"] = json!(header.shard_hash().unwrap());
    std::fs::write(&path, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();
    shards
}

fn open(shards: &Path, qd_data: &Path) -> Result<ShardReader, qd_train::shards::ShardError> {
    let mut opts = fixture_open(Arc::new(OsFiles));
    opts.qd_data_dir = qd_data.to_path_buf();
    opts.allow_stale_code = false;
    ShardReader::open(
        shards,
        &fixture().join("data").join("pool").join("train.json"),
        &opts,
    )
}

fn fp_of(header: &str) -> Value {
    parse(header)["code_fingerprint"].clone()
}

#[test]
fn a_v2_set_opens_under_rule_2_with_an_unlisted_module_present_and_edited() {
    let qd = mini_qd_data("v2-open");
    let shards = set_with_fingerprint("v2-open-set", 2, &fp_of(PY_V2));
    let reader = open(&shards, &qd).unwrap_or_else(|e| panic!("a v2 set is refused: {e}"));
    assert!(reader.check("shard_code_current").unwrap().is_pass());
    // c.py is not in the list: editing it cannot drift a v2 set.
    std::fs::write(qd.join("c.py"), b"c = 'edited'\n").unwrap();
    std::fs::write(qd.join("d.py"), b"d = 4\n").unwrap();
    open(&shards, &qd).unwrap_or_else(|e| panic!("an unlisted edit refused a v2 set: {e}"));
}

#[test]
fn a_stale_v2_set_is_refused_and_the_refusal_names_the_module() {
    let qd = mini_qd_data("v2-stale");
    let shards = set_with_fingerprint("v2-stale-set", 2, &fp_of(PY_V2));
    std::fs::write(qd.join("b.py"), b"b = 'edited'\n").unwrap();
    let err = open(&shards, &qd).expect_err("a stale v2 set is refused");
    assert!(err.to_string().contains("changed: b.py"), "{err}");
    assert!(!err.to_string().contains("c.py"), "{err}");
}

#[test]
fn a_v1_set_still_uses_rule_1_every_module() {
    let qd = mini_qd_data("v1");
    let shards = set_with_fingerprint("v1-set", 1, &fp_of(PY_V1));
    open(&shards, &qd).unwrap_or_else(|e| panic!("a current v1 set is refused: {e}"));
    std::fs::write(qd.join("c.py"), b"c = 'edited'\n").unwrap();
    let err = open(&shards, &qd).expect_err("rule 1 sees the unlisted module");
    assert!(err.to_string().contains("changed: c.py"), "{err}");
    std::fs::write(qd.join("c.py"), C_PY).unwrap();
    std::fs::write(qd.join("d.py"), b"d = 4\n").unwrap();
    let err = open(&shards, &qd).expect_err("rule 1 sees a new module");
    assert!(err.to_string().contains("added since: d.py"), "{err}");
}
