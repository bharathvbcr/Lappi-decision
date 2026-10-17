//! `qd-export` refuses to write a release unless every (trained family, slot shape as trained) has
//! a calibration entry and the fit's own recorded check for that entry ran and passed
//! (GAP-PREVIEW-DOES-NOT-SERVE-CODE-DEFECT-CLASS-2026-10-06,
//! GAP-16-OPTION-CALIBRATION-FAILS-AND-HAS-NO-ID-2026-10-06). The only way past is an explicit,
//! recorded per-family exclusion.
//!
//! Driven through the binary, as the box runs it, so the tests state the operator's command line.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::calibration_fit::ECE_THRESHOLD;
use qd_runtime::schema::SlotKind;
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_qd-export");

fn entry() -> CalibrationEntry {
    CalibrationEntry { temperature: 1.1, conformal_quantile: 0.3, noul_margin: 0.1 }
}

/// choice:5 and choice:16, plus a span entry when `with_span`.
fn table(with_span: bool) -> CalibrationTable {
    let t = CalibrationTable::new("gate-under-test")
        .with_letters(SlotKind::Choice, 5, entry())
        .with_letters(SlotKind::Choice, 16, entry());
    if with_span { t.with_span(entry()) } else { t }
}

fn write_json(path: &Path, value: &Value) -> PathBuf {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    path.to_path_buf()
}

fn ran(value: f64) -> Value {
    json!({"state": "ran", "bins": 15, "n": 500, "populated_bins": 9, "threshold": ECE_THRESHOLD,
           "value": value, "passed": value <= ECE_THRESHOLD})
}

fn not_run() -> Value {
    json!({"state": "not_run", "reason": "ECE needs at least 100 examples to be meaningful; got 20"})
}

/// The shape of `qd-calib-fit`'s report.json, restricted to what the gate reads.
fn report(table_sha: &str, ece_by_key: &[(&str, Value)], span_fitted: bool, threshold: f64) -> Value {
    let entries: serde_json::Map<String, Value> = ece_by_key
        .iter()
        .map(|(k, ece)| ((*k).to_string(), json!({"kind": "choice", "n": 500, "scored": {"ece_after": ece}})))
        .collect();
    json!({
        "ece": {"bins": 15, "min_samples": 100, "threshold": threshold},
        "entries": entries,
        "span": if span_fitted { json!({"fitted": true, "ece_after": ran(0.02)}) } else { json!({"fitted": false}) },
        "tables": [{"file": "table.json", "sha256": table_sha}],
    })
}

struct Case {
    fx: common::Fixture,
    table: PathBuf,
    table_sha: String,
}

fn case(with_span: bool) -> Case {
    let fx = common::standard();
    let path = write_json(&fx.dir.0.join("table.json"), &serde_json::to_value(table(with_span)).unwrap());
    let table_sha = qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(&path).unwrap()));
    Case { fx, table: path, table_sha }
}

fn shapes(c: &Case, doc: &Value) -> PathBuf {
    write_json(&c.fx.dir.0.join("shapes.json"), doc)
}

fn default_shapes() -> Value {
    json!({"code.defect_class": [["choice:5", "span"]], "devcouncil.verdict": [["choice:5"]]})
}

fn run(c: &Case, report: Option<&Value>, shapes: Option<&Value>, extra: &[&str]) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.arg("--source").arg(&c.fx.source)
        .arg("--base-snapshot").arg(&c.fx.snapshot)
        .arg("--tokenizer-sha256").arg(common::tokenizer_sha256(&c.fx.snapshot))
        .arg("--expect-vocab-size").arg(common::VOCAB.to_string())
        .arg("--train-manifest").arg(&c.fx.train_manifest)
        .arg("--calibration").arg(&c.table)
        .arg("--out").arg(&c.fx.out);
    if let Some(r) = report {
        cmd.arg("--calibration-report").arg(write_json(&c.fx.dir.0.join("report.json"), r));
    }
    if let Some(s) = shapes {
        cmd.arg("--trained-slot-shapes").arg(shapes_path(c, s));
    }
    cmd.args(extra);
    cmd.output().expect("the binary runs")
}

fn shapes_path(c: &Case, doc: &Value) -> PathBuf {
    shapes(c, doc)
}

fn refused(c: &Case, out: &Output, needle: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("refused"), "{stderr}");
    assert!(stderr.contains(needle), "{needle:?} not in {stderr}");
    assert!(!c.fx.out.exists(), "a refused export left a release");
    assert!(c.fx.leftovers().is_empty(), "left {:?}", c.fx.leftovers());
}

fn all_passing(c: &Case) -> Value {
    report(&c.table_sha, &[("choice:5", ran(0.01)), ("choice:16", ran(0.02))], true, ECE_THRESHOLD)
}

#[test]
fn a_table_with_no_fit_report_is_refused() {
    let c = case(true);
    let out = run(&c, None, Some(&default_shapes()), &[]);
    refused(&c, &out, "--calibration-report");
}

#[test]
fn a_table_with_no_declared_slot_shapes_is_refused() {
    let c = case(true);
    let out = run(&c, Some(&all_passing(&c)), None, &[]);
    refused(&c, &out, "--trained-slot-shapes");
}

#[test]
fn a_failing_recorded_check_is_refused_and_names_the_entry() {
    let c = case(true);
    let r = report(&c.table_sha, &[("choice:5", ran(0.01)), ("choice:16", ran(0.079))], true, ECE_THRESHOLD);
    let shapes = json!({"code.defect_class": [["choice:16", "span"]], "devcouncil.verdict": [["choice:5"]]});
    let out = run(&c, Some(&r), Some(&shapes), &[]);
    refused(&c, &out, "choice:16");
    assert!(String::from_utf8_lossy(&out.stderr).contains("passed=false"));
}

#[test]
fn a_check_that_could_not_run_is_not_a_check_that_passed() {
    let c = case(true);
    let r = report(&c.table_sha, &[("choice:5", ran(0.01)), ("choice:16", not_run())], true, ECE_THRESHOLD);
    let shapes = json!({"code.defect_class": [["choice:16", "span"]], "devcouncil.verdict": [["choice:5"]]});
    let out = run(&c, Some(&r), Some(&shapes), &[]);
    refused(&c, &out, "not_run");
}

#[test]
fn a_family_whose_trained_span_slot_has_no_entry_is_refused() {
    let c = case(false);
    let r = report(&c.table_sha, &[("choice:5", ran(0.01)), ("choice:16", ran(0.02))], false, ECE_THRESHOLD);
    let out = run(&c, Some(&r), Some(&default_shapes()), &[]);
    refused(&c, &out, "code.defect_class");
    assert!(String::from_utf8_lossy(&out.stderr).contains("span"));
}

#[test]
fn a_shape_the_table_has_no_entry_for_is_refused() {
    let c = case(true);
    let shapes = json!({"code.defect_class": [["choice:7", "span"]], "devcouncil.verdict": [["choice:5"]]});
    let out = run(&c, Some(&all_passing(&c)), Some(&shapes), &[]);
    refused(&c, &out, "choice:7");
}

#[test]
fn a_trained_family_with_no_declared_shapes_is_refused() {
    let c = case(true);
    let shapes = json!({"devcouncil.verdict": [["choice:5"]]});
    let out = run(&c, Some(&all_passing(&c)), Some(&shapes), &[]);
    refused(&c, &out, "code.defect_class");
}

#[test]
fn a_report_for_another_table_is_refused() {
    let c = case(true);
    let r = report(&"0".repeat(64), &[("choice:5", ran(0.01)), ("choice:16", ran(0.02))], true, ECE_THRESHOLD);
    let out = run(&c, Some(&r), Some(&default_shapes()), &[]);
    refused(&c, &out, "sha256");
}

#[test]
fn a_report_fitted_under_another_ece_threshold_is_refused() {
    let c = case(true);
    let r = report(&c.table_sha, &[("choice:5", ran(0.01)), ("choice:16", ran(0.02))], true, 0.5);
    let out = run(&c, Some(&r), Some(&default_shapes()), &[]);
    refused(&c, &out, "threshold");
}

#[test]
fn a_calibrated_table_whose_checks_all_passed_is_exported_and_the_gate_is_recorded() {
    let c = case(true);
    let out = run(&c, Some(&all_passing(&c)), Some(&default_shapes()), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    let m: Value = serde_json::from_slice(&std::fs::read(c.fx.out.join("release_manifest.json")).unwrap()).unwrap();
    assert_eq!(m["trained_families"], json!(["code.defect_class", "devcouncil.verdict"]));
    assert_eq!(m["trained_slot_shapes"], default_shapes());
    assert_eq!(m["excluded_families"], json!([]));
    assert_eq!(m["calibration"]["gate"]["ece_threshold"], json!(ECE_THRESHOLD));
    assert_eq!(m["calibration"]["gate"]["table_sha256"], json!(c.table_sha));
}

#[test]
fn an_explicit_exclusion_is_the_way_past_and_is_recorded() {
    let c = case(false);
    let r = report(&c.table_sha, &[("choice:5", ran(0.01)), ("choice:16", ran(0.02))], false, ECE_THRESHOLD);
    let reason = "span is not fittable from verdicts (GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS)";
    let out = run(&c, Some(&r), Some(&default_shapes()), &["--exclude-family", &format!("code.defect_class={reason}")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    let m: Value = serde_json::from_slice(&std::fs::read(c.fx.out.join("release_manifest.json")).unwrap()).unwrap();
    assert_eq!(m["trained_families"], json!(["devcouncil.verdict"]));
    assert_eq!(m["excluded_families"], json!([{"family": "code.defect_class", "reason": reason}]));
    assert_eq!(m["trained_slot_shapes"], json!({"devcouncil.verdict": [["choice:5"]]}));
}

#[test]
fn an_exclusion_without_a_reason_or_for_an_unknown_family_is_refused() {
    let c = case(false);
    let r = report(&c.table_sha, &[("choice:5", ran(0.01)), ("choice:16", ran(0.02))], false, ECE_THRESHOLD);
    let out = run(&c, Some(&r), Some(&default_shapes()), &["--exclude-family", "code.defect_class="]);
    assert_ne!(out.status.code(), Some(0));
    assert!(!c.fx.out.exists());
    let out = run(&c, Some(&r), Some(&default_shapes()), &["--exclude-family", "no.such.family=because"]);
    refused(&c, &out, "no.such.family");
}

#[test]
fn a_release_without_a_table_needs_a_recorded_reason() {
    let c = case(true);
    let mut cmd = Command::new(BIN);
    cmd.arg("--source").arg(&c.fx.source)
        .arg("--base-snapshot").arg(&c.fx.snapshot)
        .arg("--tokenizer-sha256").arg(common::tokenizer_sha256(&c.fx.snapshot))
        .arg("--expect-vocab-size").arg(common::VOCAB.to_string())
        .arg("--train-manifest").arg(&c.fx.train_manifest)
        .arg("--out").arg(&c.fx.out);
    let out = cmd.output().unwrap();
    refused(&c, &out, "--uncalibrated-reason");
    cmd.arg("--uncalibrated-reason").arg("trial export for a tower round trip; not served");
    let out = cmd.output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let m: Value = serde_json::from_slice(&std::fs::read(c.fx.out.join("release_manifest.json")).unwrap()).unwrap();
    assert_eq!(m["calibration"], Value::Null);
    assert_eq!(m["calibration_absent_reason"], json!("trial export for a tower round trip; not served"));
}
