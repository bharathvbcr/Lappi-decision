//! The ensemble's own table goes through the calibration gate the single-table path uses
//! (GAP-QD-EXPORT-ENSEMBLE-TABLE-NOT-GATED-2026-10-17): the fit's report must be bound to the
//! table by sha256, and every (family, shape) the members record as trained must have an entry
//! whose recorded check ran and passed. The shapes are the members' own, read from their release
//! manifests, which got them from their train manifests.
//!
//! Driven through the `qd-export-ensemble` binary, as the box runs it.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use qd_export::export;
use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::calibration_fit::ECE_THRESHOLD;
use qd_runtime::schema::SlotKind;
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_qd-export-ensemble");
const V5_PROMPT_FORMAT: u64 = 2;

fn entry() -> CalibrationEntry {
    CalibrationEntry { temperature: 1.21, conformal_quantile: 0.09, noul_margin: 0.02 }
}

fn write_json(path: &Path, value: &Value) -> PathBuf {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    path.to_path_buf()
}

fn ran(value: f64) -> Value {
    json!({"state": "ran", "bins": 15, "n": 500, "populated_bins": 9, "threshold": ECE_THRESHOLD,
           "value": value, "passed": value <= ECE_THRESHOLD})
}

fn train_manifest(path: &Path, rows: &[(&str, &[&str])]) -> PathBuf {
    let entries: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(i, (family, shape))| {
            json!({
                "row_id": format!("train-row-{i}"), "content_hash": format!("{i:064x}"), "split": "train",
                "source_id": "test/source", "host": "test", "family_id": family, "slot_shape": shape,
                "repo_key": format!("repo-{i}"), "identity_key": format!("identity-{i}"),
                "licence_id": "MIT", "obligations": [],
            })
        })
        .collect();
    write_json(
        path,
        &json!({
            "manifest_format_version": 1, "split": "train", "data_snapshot_hash": "ab".repeat(32),
            "n_rows": entries.len(), "held_out_families": ["code.language_id", "qa.answerability"],
            "entries": entries,
        }),
    )
}

const DEFECT: [&str; 2] = ["choice:5", "span"];
const VERDICT: [&str; 1] = ["choice:5"];

/// One member release exported from a train manifest holding `rows`.
fn member(seed: u64, rows: &[(&str, &[&str])]) -> common::Fixture {
    let mut tensors = common::standard_tensors();
    let (name, shape) = tensors
        .iter()
        .find(|(name, _)| name.starts_with("tower.") && name.contains("layernorm"))
        .map(|(name, t)| (name.clone(), t.shape.clone()))
        .expect("the standard fixture has a layernorm");
    tensors.insert(name, common::bf16_tensor(&shape, 7000 + seed));
    let fx = common::build(&tensors, &common::tiny_config(), |m| m["prompt_format"] = json!(V5_PROMPT_FORMAT));
    let mut req = fx.request();
    req.train_manifest = train_manifest(&fx.dir.0.join(format!("train-{seed}.json")), rows);
    export(&req).expect("a member release exports");
    fx
}

fn standard_rows() -> Vec<(&'static str, &'static [&'static str])> {
    vec![("code.defect_class", &DEFECT[..]), ("devcouncil.verdict", &VERDICT[..])]
}

struct Case {
    members: Vec<common::Fixture>,
    dir: common::TempDir,
    table: PathBuf,
    table_sha: String,
}

fn case(rows_per_member: &[Vec<(&str, &[&str])>]) -> Case {
    let members = rows_per_member.iter().enumerate().map(|(i, rows)| member(i as u64, rows)).collect();
    let dir = common::TempDir::new("ens-gate");
    let table = CalibrationTable::new("ensemble-gate").with_letters(SlotKind::Choice, 5, entry()).with_span(entry());
    let table_path = write_json(&dir.0.join("table.json"), &serde_json::to_value(table).unwrap());
    let table_sha = qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(&table_path).unwrap()));
    Case { members, dir, table: table_path, table_sha }
}

fn report(c: &Case, choice5: Value, sha: &str) -> PathBuf {
    write_json(
        &c.dir.0.join("report.json"),
        &json!({
            "ece": {"bins": 15, "min_samples": 100, "threshold": ECE_THRESHOLD},
            "entries": {"choice:5": {"scored": {"ece_after": choice5}}},
            "span": {"fitted": true, "scored": {"ece_after": ran(0.02)}},
            "tables": [{"file": "table.json", "sha256": sha}],
        }),
    )
}

fn out_dir(c: &Case) -> PathBuf {
    c.dir.0.join("ensemble")
}

fn run(c: &Case, report: Option<&Path>) -> Output {
    let mut cmd = Command::new(BIN);
    for m in &c.members {
        cmd.arg("--member").arg(&m.out).arg("--trained-width").arg("1105");
    }
    cmd.arg("--trained-width-source").arg("test: widths")
        .arg("--calibration").arg(&c.table)
        .arg("--out").arg(out_dir(c));
    if let Some(r) = report {
        cmd.arg("--calibration-report").arg(r);
    }
    cmd.output().expect("the binary runs")
}

fn refused(c: &Case, out: &Output, needle: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains(needle), "{needle:?} not in {stderr}");
    assert!(!out_dir(c).exists(), "a refused ensemble left a directory");
}

fn three() -> Vec<Vec<(&'static str, &'static [&'static str])>> {
    vec![standard_rows(), standard_rows(), standard_rows()]
}

#[test]
fn an_ensemble_table_whose_checks_passed_is_written_and_the_gate_is_recorded() {
    let c = case(&three());
    let r = report(&c, ran(0.01), &c.table_sha);
    let out = run(&c, Some(&r));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    let m: Value = serde_json::from_slice(&std::fs::read(out_dir(&c).join("ensemble_manifest.json")).unwrap()).unwrap();
    assert_eq!(m["calibration"]["gate"]["table_sha256"], json!(c.table_sha));
    assert_eq!(m["calibration"]["gate"]["ece_threshold"], json!(ECE_THRESHOLD));
}

#[test]
fn an_ensemble_with_no_fit_report_is_refused() {
    let c = case(&three());
    let out = run(&c, None);
    refused(&c, &out, "--calibration-report");
}

#[test]
fn a_failing_recorded_check_on_the_ensemble_table_is_refused() {
    let c = case(&three());
    let r = report(&c, ran(0.079), &c.table_sha);
    let out = run(&c, Some(&r));
    refused(&c, &out, "choice:5");
    assert!(String::from_utf8_lossy(&out.stderr).contains("passed=false"));
}

#[test]
fn a_check_that_could_not_run_on_the_ensemble_table_is_refused() {
    let c = case(&three());
    let r = report(&c, json!({"state": "not_run", "reason": "fewer than 100 examples"}), &c.table_sha);
    let out = run(&c, Some(&r));
    refused(&c, &out, "not_run");
}

#[test]
fn a_report_for_another_table_is_refused() {
    let c = case(&three());
    let r = report(&c, ran(0.01), &"0".repeat(64));
    let out = run(&c, Some(&r));
    refused(&c, &out, "sha256");
}

#[test]
fn members_that_trained_on_different_shapes_are_refused() {
    let other: Vec<(&str, &[&str])> = vec![("code.defect_class", &["choice:5"][..]), ("devcouncil.verdict", &VERDICT[..])];
    let c = case(&[standard_rows(), other, standard_rows()]);
    let r = report(&c, ran(0.01), &c.table_sha);
    let out = run(&c, Some(&r));
    refused(&c, &out, "trained_slot_shapes");
}
