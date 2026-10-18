//! What ties the train manifest `qd-export` reads to the weights it exports.
//!
//! * `slot_shape_digest`: a versioned digest over every entry's `slot_shape`, re-derived on read.
//!   `slot_shape` decides what a release admits and sits outside `data_snapshot_hash`, so an edit
//!   to one entry that leaves the digest alone is refused (defence in depth).
//! * `--ft-ledger`: when given, the train manifest's file sha256 must equal the
//!   `train_manifest_sha256` row field of every ft row the release is built from. A differing
//!   value is refused, and an ABSENT field is refused too: absent means not recorded, which is
//!   not the same as matching. This depends on the W2-1 lane adding that optional field to the
//!   ft LedgerRow; the fixture rows here carry it under that name.
//! * No `--ft-ledger`: the release says the binding was not verified, in words a reader cannot
//!   mistake for a pass.

mod common;

use std::path::{Path, PathBuf};

use qd_export::export;
use serde_json::{Value, json};

const DIGEST_VECTOR: &str = "ac5cb778fa420b231bf726ea7bf8e99d7d8e8319c29090c075fa4bc2aa6bfa39";

/// The digest, spelled out as `python/qd_data/manifest.py::slot_shape_digest` defines it.
fn digest(entries: &[Value]) -> String {
    let mut rows: Vec<&Value> = entries.iter().collect();
    rows.sort_by_key(|e| e["row_id"].as_str().unwrap().to_string());
    let body: String = rows
        .iter()
        .map(|e| {
            let keys: Vec<&str> = e["slot_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|k| k.as_str().unwrap())
                .collect();
            format!(
                "{}\t{}\t{}\n",
                e["row_id"].as_str().unwrap(),
                e["content_hash"].as_str().unwrap(),
                keys.join(",")
            )
        })
        .collect();
    qd_runtime::hex(&qd_runtime::sha256(body.as_bytes()))
}

fn entries() -> Vec<Value> {
    ["code.defect_class", "devcouncil.verdict"]
        .iter()
        .enumerate()
        .map(|(i, family)| {
            json!({
                "row_id": format!("train-row-{i}"), "content_hash": format!("{i:064x}"), "split": "train",
                "source_id": "test/source", "host": "test", "family_id": family, "slot_shape": ["choice:5"],
                "repo_key": format!("repo-{i}"), "identity_key": format!("identity-{i}"),
                "licence_id": "MIT", "obligations": [],
            })
        })
        .collect()
}

fn manifest_doc(entries: &[Value], digest_value: Option<&str>) -> Value {
    let mut doc = json!({
        "manifest_format_version": 1, "split": "train", "data_snapshot_hash": "ab".repeat(32),
        "n_rows": entries.len(), "held_out_families": ["code.language_id", "qa.answerability"],
        "entries": entries,
    });
    if let Some(d) = digest_value {
        doc["slot_shape_digest"] = json!({"version": 1, "sha256": d});
    }
    doc
}

fn write_manifest(path: &Path, doc: &Value) -> String {
    let bytes = serde_json::to_vec_pretty(doc).unwrap();
    std::fs::write(path, &bytes).unwrap();
    qd_runtime::hex(&qd_runtime::sha256(&bytes))
}

fn good_manifest(path: &Path) -> String {
    let e = entries();
    write_manifest(path, &manifest_doc(&e, Some(&digest(&e))))
}

fn ledger(dir: &Path, rows: &[Value]) -> PathBuf {
    let path = dir.join("ft-ledger.jsonl");
    let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(&path, text).unwrap();
    path
}

fn ft_row(id: &str, seed: u64, sha: Option<&str>) -> Value {
    let mut row =
        json!({"row_id": id, "run_kind": "ft", "status": "completed", "protocol": {"seed": seed}});
    if let Some(sha) = sha {
        row["train_manifest_sha256"] = json!(sha);
    }
    row
}

fn rows_with(sha: &str) -> Vec<Value> {
    common::FT_ROW_IDS
        .iter()
        .enumerate()
        .map(|(i, id)| ft_row(id, i as u64, Some(sha)))
        .collect()
}

fn refused(req: &qd_export::ExportRequest, needle: &str) {
    let err = export(req).expect_err("this export must be refused");
    assert!(
        err.detail.contains(needle),
        "{needle:?} not in {}",
        err.detail
    );
    assert!(!req.out.exists(), "a refused export left a release");
}

fn release_manifest(out: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(out.join("release_manifest.json")).unwrap()).unwrap()
}

#[test]
fn the_digest_definition_has_the_pinned_value() {
    let e =
        json!({"row_id": "r1", "content_hash": "0".repeat(64), "slot_shape": ["choice:5", "span"]});
    assert_eq!(digest(&[e]), DIGEST_VECTOR);
}

#[test]
fn a_manifest_whose_digest_matches_its_entries_is_read() {
    let fx = common::standard();
    good_manifest(&fx.train_manifest);
    export(&fx.request()).expect("a digest-bound manifest exports");
}

#[test]
fn one_edited_slot_shape_with_the_old_digest_is_refused() {
    let fx = common::standard();
    let mut e = entries();
    let d = digest(&e);
    e[0]["slot_shape"] = json!(["choice:16"]);
    write_manifest(&fx.train_manifest, &manifest_doc(&e, Some(&d)));
    refused(&fx.request(), "slot_shape_digest");
}

#[test]
fn a_manifest_with_no_digest_or_an_unknown_digest_version_is_refused() {
    let fx = common::standard();
    let e = entries();
    write_manifest(&fx.train_manifest, &manifest_doc(&e, None));
    refused(&fx.request(), "slot_shape_digest");
    let mut doc = manifest_doc(&e, Some(&digest(&e)));
    doc["slot_shape_digest"]["version"] = json!(2);
    write_manifest(&fx.train_manifest, &doc);
    refused(&fx.request(), "slot_shape_digest");
}

#[test]
fn a_ledger_whose_rows_record_this_manifests_sha_binds_it_and_the_release_says_so() {
    let fx = common::standard();
    let sha = good_manifest(&fx.train_manifest);
    let mut req = fx.request();
    req.ft_ledger = Some(ledger(&fx.dir.0, &rows_with(&sha)));
    export(&req).expect("bound");
    let m = release_manifest(&fx.out);
    assert_eq!(
        m["train_manifest"]["ledger_binding"]["verified"],
        json!(true)
    );
    assert_eq!(
        m["train_manifest"]["ledger_binding"]["train_manifest_sha256"],
        json!(sha)
    );
}

#[test]
fn a_train_manifest_edited_after_the_ledger_recorded_it_is_refused_even_with_a_fresh_digest() {
    let fx = common::standard();
    let sha = good_manifest(&fx.train_manifest);
    let mut e = entries();
    e[0]["slot_shape"] = json!(["choice:16"]);
    write_manifest(&fx.train_manifest, &manifest_doc(&e, Some(&digest(&e))));
    let mut req = fx.request();
    req.ft_ledger = Some(ledger(&fx.dir.0, &rows_with(&sha)));
    refused(&req, "train_manifest_sha256");
}

#[test]
fn one_row_recording_another_sha_is_refused_and_named() {
    let fx = common::standard();
    let sha = good_manifest(&fx.train_manifest);
    let mut rows = rows_with(&sha);
    rows[1]["train_manifest_sha256"] = json!("0".repeat(64));
    let mut req = fx.request();
    req.ft_ledger = Some(ledger(&fx.dir.0, &rows));
    refused(&req, common::FT_ROW_IDS[1]);
}

#[test]
fn a_row_with_no_recorded_sha_is_refused_not_skipped() {
    let fx = common::standard();
    let sha = good_manifest(&fx.train_manifest);
    let mut rows = rows_with(&sha);
    rows[2] = ft_row(common::FT_ROW_IDS[2], 2, None);
    let mut req = fx.request();
    req.ft_ledger = Some(ledger(&fx.dir.0, &rows));
    refused(&req, "not recorded");
}

#[test]
fn a_row_the_source_manifest_names_but_the_ledger_lacks_is_refused() {
    let fx = common::standard();
    let sha = good_manifest(&fx.train_manifest);
    let mut rows = rows_with(&sha);
    rows.pop();
    let mut req = fx.request();
    req.ft_ledger = Some(ledger(&fx.dir.0, &rows));
    refused(&req, common::FT_ROW_IDS[2]);
}

#[test]
fn with_no_ledger_the_release_says_the_binding_was_not_verified() {
    let fx = common::standard();
    good_manifest(&fx.train_manifest);
    export(&fx.request()).expect("exports");
    let b = &release_manifest(&fx.out)["train_manifest"]["ledger_binding"];
    assert_eq!(b["verified"], json!(false));
    assert!(b["reason"].as_str().unwrap().contains("--ft-ledger"), "{b}");
}
