//! The train manifest owns the request shapes each family trained on
//! (GAP-QD-EXPORT-NO-TRACKED-PER-FAMILY-SLOT-SHAPE-REGISTRY-2026-10-17): every entry carries the
//! `slot_shape` of its request, `qd-export` reads the per-family set from the manifest it already
//! binds, and a release whose train manifest records none is refused. A hand-passed list is gone,
//! not cross-checked: a second owner could only agree or be refused.
//!
//! Driven through the binary, as the box runs it.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::calibration_fit::ECE_THRESHOLD;
use qd_runtime::schema::SlotKind;
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-export");

fn entry() -> CalibrationEntry {
    CalibrationEntry {
        temperature: 1.1,
        conformal_quantile: 0.3,
        noul_margin: 0.1,
    }
}

fn write_json(path: &Path, value: &Value) -> PathBuf {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    path.to_path_buf()
}

fn ran(value: f64) -> Value {
    json!({"state": "ran", "bins": 15, "n": 500, "populated_bins": 9, "threshold": ECE_THRESHOLD,
           "value": value, "passed": value <= ECE_THRESHOLD})
}

/// A train manifest in `Manifest.to_json()`'s shape. `rows` is `(family, slot_shape)`; a `None`
/// shape writes the entry without the key, as a manifest built before the shapes were recorded.
fn train_manifest(path: &Path, rows: &[(&str, Option<&[&str]>)]) -> PathBuf {
    let entries: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(i, (family, shape))| {
            let mut e = json!({
                "row_id": format!("train-row-{i}"), "content_hash": format!("{i:064x}"), "split": "train",
                "source_id": "test/source", "host": "test", "family_id": family,
                "repo_key": format!("repo-{i}"), "identity_key": format!("identity-{i}"),
                "licence_id": "MIT", "obligations": [],
            });
            if let Some(shape) = shape {
                e["slot_shape"] = json!(shape);
            }
            e
        })
        .collect();
    let mut doc = json!({
        "manifest_format_version": 1, "split": "train", "data_snapshot_hash": "ab".repeat(32),
        "n_rows": entries.len(), "held_out_families": ["code.language_id", "qa.answerability"],
        "entries": entries,
    });
    // The digest is over the shapes as written; an entry with none has nothing to bind, and the
    // refusal that case expects comes before the digest is read.
    if rows
        .iter()
        .all(|(_, shape)| shape.is_some_and(|s| !s.is_empty()))
    {
        doc["slot_shape_digest"] = common::slot_shape_digest(doc["entries"].as_array().unwrap());
    }
    write_json(path, &doc)
}

struct Case {
    fx: common::Fixture,
    table: PathBuf,
    report: PathBuf,
}

/// A table with choice:5, choice:16 and span, and a report in which `bad` (if any) failed.
fn case(bad: Option<&str>) -> Case {
    let fx = common::standard();
    let table = CalibrationTable::new("shapes-under-test")
        .with_letters(SlotKind::Choice, 5, entry())
        .with_letters(SlotKind::Choice, 16, entry())
        .with_span(entry());
    let table_path = write_json(
        &fx.dir.0.join("table.json"),
        &serde_json::to_value(table).unwrap(),
    );
    let sha = qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(&table_path).unwrap()));
    let ece = |k: &str| {
        if bad == Some(k) {
            ran(0.079)
        } else {
            ran(0.01)
        }
    };
    let report = write_json(
        &fx.dir.0.join("report.json"),
        &json!({
            "ece": {"bins": 15, "min_samples": 100, "threshold": ECE_THRESHOLD},
            "entries": {
                "choice:5": {"scored": {"ece_after": ece("choice:5")}},
                "choice:16": {"scored": {"ece_after": ece("choice:16")}},
            },
            "span": {"fitted": true, "scored": {"ece_after": ece("span")}},
            "tables": [{"file": "table.json", "sha256": sha}],
        }),
    );
    Case {
        fx,
        table: table_path,
        report,
    }
}

fn run(c: &Case, train: &Path, extra: &[&str]) -> Output {
    Command::new(BIN)
        .arg("--source")
        .arg(&c.fx.source)
        .arg("--base-snapshot")
        .arg(&c.fx.snapshot)
        .arg("--tokenizer-sha256")
        .arg(common::tokenizer_sha256(&c.fx.snapshot))
        .arg("--expect-vocab-size")
        .arg(common::VOCAB.to_string())
        .arg("--train-manifest")
        .arg(train)
        .arg("--out")
        .arg(&c.fx.out)
        .args(extra)
        .output()
        .expect("the binary runs")
}

fn calibrated(c: &Case, train: &Path) -> Output {
    run(
        c,
        train,
        &[
            "--calibration",
            c.table.to_str().unwrap(),
            "--calibration-report",
            c.report.to_str().unwrap(),
        ],
    )
}

fn refused(c: &Case, out: &Output, needle: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("refused"), "{stderr}");
    assert!(stderr.contains(needle), "{needle:?} not in {stderr}");
    assert!(!c.fx.out.exists(), "a refused export left a release");
    assert!(c.fx.leftovers().is_empty(), "left {:?}", c.fx.leftovers());
}

const DEFECT: [&str; 2] = ["choice:5", "span"];
const VERDICT: [&str; 1] = ["choice:5"];

fn standard_rows() -> Vec<(&'static str, Option<&'static [&'static str]>)> {
    vec![
        ("code.defect_class", Some(&DEFECT[..])),
        ("devcouncil.verdict", Some(&VERDICT[..])),
    ]
}

fn release_manifest(c: &Case) -> Value {
    serde_json::from_slice(&std::fs::read(c.fx.out.join("release_manifest.json")).unwrap()).unwrap()
}

#[test]
fn the_release_records_the_shapes_its_train_manifest_holds_with_no_flag() {
    let c = case(None);
    let train = train_manifest(&c.fx.dir.0.join("t.json"), &standard_rows());
    let out = calibrated(&c, &train);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert_eq!(
        release_manifest(&c)["trained_slot_shapes"],
        json!({"code.defect_class": [["choice:5", "span"]], "devcouncil.verdict": [["choice:5"]]})
    );
}

#[test]
fn a_family_trained_on_two_shapes_has_both_recorded_and_both_gated() {
    let c = case(Some("choice:16"));
    let rows: Vec<(&str, Option<&[&str]>)> = vec![
        ("code.defect_class", Some(&DEFECT[..])),
        ("devcouncil.verdict", Some(&VERDICT[..])),
        ("devcouncil.verdict", Some(&["choice:16"][..])),
    ];
    let train = train_manifest(&c.fx.dir.0.join("t.json"), &rows);
    let out = calibrated(&c, &train);
    refused(&c, &out, "devcouncil.verdict/choice:16");
}

#[test]
fn a_train_manifest_that_records_no_shapes_is_refused() {
    let c = case(None);
    let rows: Vec<(&str, Option<&[&str]>)> =
        vec![("code.defect_class", None), ("devcouncil.verdict", None)];
    let train = train_manifest(&c.fx.dir.0.join("t.json"), &rows);
    let out = run(
        &c,
        &train,
        &["--uncalibrated-reason", "trial export; not served"],
    );
    refused(&c, &out, "slot_shape");
}

#[test]
fn one_entry_without_a_shape_refuses_the_whole_manifest() {
    let c = case(None);
    let rows: Vec<(&str, Option<&[&str]>)> = vec![
        ("code.defect_class", Some(&DEFECT[..])),
        ("devcouncil.verdict", None),
    ];
    let train = train_manifest(&c.fx.dir.0.join("t.json"), &rows);
    let out = calibrated(&c, &train);
    refused(&c, &out, "slot_shape");
}

#[test]
fn a_malformed_shape_is_refused() {
    let c = case(None);
    for bad in [&["choice:"][..], &["span", "tree"][..], &[][..]] {
        let rows: Vec<(&str, Option<&[&str]>)> = vec![
            ("code.defect_class", Some(bad)),
            ("devcouncil.verdict", Some(&VERDICT[..])),
        ];
        let train = train_manifest(&c.fx.dir.0.join("t.json"), &rows);
        let out = calibrated(&c, &train);
        refused(&c, &out, "slot_shape");
    }
}

#[test]
fn a_trained_shape_the_table_has_no_entry_for_is_refused_by_name() {
    let c = case(None);
    let rows: Vec<(&str, Option<&[&str]>)> = vec![
        ("code.defect_class", Some(&["choice:7", "span"][..])),
        ("devcouncil.verdict", Some(&VERDICT[..])),
    ];
    let train = train_manifest(&c.fx.dir.0.join("t.json"), &rows);
    let out = calibrated(&c, &train);
    refused(&c, &out, "choice:7");
}

#[test]
fn the_hand_passed_shapes_flag_no_longer_exists() {
    let c = case(None);
    let train = train_manifest(&c.fx.dir.0.join("t.json"), &standard_rows());
    let shapes = write_json(
        &c.fx.dir.0.join("shapes.json"),
        &json!({"code.defect_class": [["choice:5"]]}),
    );
    let out = Command::new(BIN)
        .arg("--source")
        .arg(&c.fx.source)
        .arg("--base-snapshot")
        .arg(&c.fx.snapshot)
        .arg("--tokenizer-sha256")
        .arg(common::tokenizer_sha256(&c.fx.snapshot))
        .arg("--expect-vocab-size")
        .arg(common::VOCAB.to_string())
        .arg("--train-manifest")
        .arg(&train)
        .arg("--calibration")
        .arg(&c.table)
        .arg("--calibration-report")
        .arg(&c.report)
        .arg("--trained-slot-shapes")
        .arg(&shapes)
        .arg("--out")
        .arg(&c.fx.out)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_ne!(out.status.code(), Some(0), "{stderr}");
    assert!(stderr.contains("unexpected argument"), "{stderr}");
    assert!(!c.fx.out.exists());
}
