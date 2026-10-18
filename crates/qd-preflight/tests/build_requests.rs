//! `qd-release-check build-requests`: the tracked builder of the replay's request file
//! (GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17). It turns a release's recorded
//! (admitted family, trained shape) set and the val rows' wire requests into the requests
//! `replay --requests` sends, deterministically, and says which pairs it could not fill.
//! `--per-shape` is required and has no default, as `check --min-answered` has none
//! (GAP-QD-RELEASE-CHECK-MIN-ANSWERED-HAS-NO-RECORD-2026-10-17).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use qd_preflight::release_check::request_shape;
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_qd-release-check");

static N: AtomicU64 = AtomicU64::new(0);

fn tmp() -> PathBuf {
    let p = std::env::temp_dir().join(format!("qd-br-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn release(dir: &Path, shapes: Option<Value>) {
    let mut m = json!({
        "format": "qd-release.v2",
        "trained_families": ["code.defect_class", "fam.b"],
        "excluded_families": [],
        "expected_identity": {"weight_hash": "w", "calibration_hash": "c"},
    });
    if let Some(s) = shapes {
        m["trained_slot_shapes"] = s;
    }
    std::fs::write(dir.join("release_manifest.json"), m.to_string()).unwrap();
}

fn shapes() -> Value {
    json!({"code.defect_class": [["choice:5", "span"]], "fam.b": [["choice:4"], ["choice:6"]]})
}

fn choice(options: usize) -> Value {
    let opts: Vec<String> = (0..options).map(|i| format!("o{i}")).collect();
    json!({"name": "x", "type": "choice", "options": opts})
}

fn req(task: &str, id: usize, slots: Vec<Value>) -> Value {
    json!({"schema_version": 1, "task": task, "context_b64": "eA==", "context_len": 1, "question": "Q?",
           "slots": slots, "route": "generic", "example_id": format!("val:{task}:{id}"), "metadata": {}})
}

fn defect(id: usize) -> Value {
    req("code.defect_class", id, vec![choice(4), json!({"name": "s", "type": "span"})])
}

/// 3 defect rows, 4 of fam.b at choice:4, 1 at choice:6; and rows the release does not admit: a
/// family it does not train, and fam.b at a shape it did not train on.
fn val_rows() -> Vec<Value> {
    let mut v: Vec<Value> = (0..3).map(defect).collect();
    v.extend((0..4).map(|i| req("fam.b", i, vec![choice(3)])));
    v.push(req("fam.b", 10, vec![choice(5)]));
    v.push(req("fam.b", 20, vec![choice(7)]));
    v.extend((0..5).map(|i| req("not.admitted", i, vec![choice(3)])));
    v
}

fn write_lines(path: &Path, rows: &[Value]) {
    let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(path, text).unwrap();
}

fn run(release_dir: &Path, val: &Path, out: &Path, per_shape: &str) -> Output {
    Command::new(BIN)
        .args(["build-requests", "--release"]).arg(release_dir)
        .arg("--val-requests").arg(val)
        .arg("--per-shape").arg(per_shape)
        .arg("--out").arg(out)
        .output()
        .unwrap()
}

fn lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[test]
fn it_emits_per_shape_requests_for_exactly_the_admitted_pairs() {
    let d = tmp();
    release(&d, Some(shapes()));
    // Every admitted pair has at least two val rows here (fam.b at choice:6 gets a second), so
    // asking for two fills them all: a full run exits 0.
    let mut rows = val_rows();
    rows.push(req("fam.b", 11, vec![choice(5)]));
    write_lines(&d.join("val.jsonl"), &rows);
    let out = run(&d, &d.join("val.jsonl"), &d.join("req.jsonl"), "2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}{}", String::from_utf8_lossy(&out.stdout));
    let got = lines(&d.join("req.jsonl"));
    let mut count = std::collections::BTreeMap::<(String, Vec<String>), usize>::new();
    for r in &got {
        let key = (r["task"].as_str().unwrap().to_string(), request_shape(r).unwrap());
        *count.entry(key).or_default() += 1;
    }
    let want: std::collections::BTreeMap<(String, Vec<String>), usize> = [
        (("code.defect_class", vec!["choice:5", "span"]), 2),
        (("fam.b", vec!["choice:4"]), 2),
        (("fam.b", vec!["choice:6"]), 2),
    ]
    .into_iter()
    .map(|((f, s), n)| ((f.to_string(), s.into_iter().map(String::from).collect()), n))
    .collect();
    assert_eq!(count, want, "{got:?}");
}

#[test]
fn a_pair_with_fewer_rows_than_asked_is_reported_and_fails() {
    let d = tmp();
    release(&d, Some(shapes()));
    write_lines(&d.join("val.jsonl"), &val_rows());
    let out = run(&d, &d.join("val.jsonl"), &d.join("req.jsonl"), "2");
    // fam.b at choice:6 has a single val row, so two cannot be built.
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("fam.b") && text.contains("choice:6") && text.contains("short"), "{text}");
    let got = lines(&d.join("req.jsonl"));
    assert_eq!(got.len(), 2 + 2 + 1, "what exists is still written: {got:?}");
}

#[test]
fn nothing_outside_the_admitted_pairs_is_emitted() {
    let d = tmp();
    release(&d, Some(shapes()));
    write_lines(&d.join("val.jsonl"), &val_rows());
    run(&d, &d.join("val.jsonl"), &d.join("req.jsonl"), "10");
    let admitted = [
        ("code.defect_class".to_string(), vec!["choice:5".to_string(), "span".to_string()]),
        ("fam.b".to_string(), vec!["choice:4".to_string()]),
        ("fam.b".to_string(), vec!["choice:6".to_string()]),
    ];
    let got = lines(&d.join("req.jsonl"));
    assert!(!got.is_empty());
    for r in got {
        let key = (r["task"].as_str().unwrap().to_string(), request_shape(&r).unwrap());
        assert!(admitted.contains(&key), "{key:?} is not an admitted (family, shape)");
    }
}

#[test]
fn the_output_does_not_depend_on_the_input_order_and_repeats_byte_for_byte() {
    let d = tmp();
    release(&d, Some(shapes()));
    let rows = val_rows();
    write_lines(&d.join("a.jsonl"), &rows);
    let mut rev = rows.clone();
    rev.reverse();
    write_lines(&d.join("b.jsonl"), &rev);
    run(&d, &d.join("a.jsonl"), &d.join("ra.jsonl"), "2");
    run(&d, &d.join("b.jsonl"), &d.join("rb.jsonl"), "2");
    run(&d, &d.join("a.jsonl"), &d.join("ra2.jsonl"), "2");
    let a = std::fs::read(d.join("ra.jsonl")).unwrap();
    assert!(!a.is_empty());
    assert_eq!(a, std::fs::read(d.join("rb.jsonl")).unwrap());
    assert_eq!(a, std::fs::read(d.join("ra2.jsonl")).unwrap());
}

#[test]
fn duplicate_requests_count_once() {
    let d = tmp();
    release(&d, Some(shapes()));
    let mut rows = val_rows();
    rows.extend((0..3).map(|_| req("fam.b", 0, vec![choice(3)])));
    write_lines(&d.join("val.jsonl"), &rows);
    run(&d, &d.join("val.jsonl"), &d.join("req.jsonl"), "4");
    let got = lines(&d.join("req.jsonl"));
    let mut ids: Vec<String> = got.iter().map(|r| r.to_string()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), got.len(), "a request was emitted twice");
}

#[test]
fn a_release_that_records_no_shapes_is_not_run_and_writes_nothing() {
    let d = tmp();
    release(&d, None);
    write_lines(&d.join("val.jsonl"), &val_rows());
    let out = run(&d, &d.join("val.jsonl"), &d.join("req.jsonl"), "2");
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("trained_slot_shapes"));
    assert!(!d.join("req.jsonl").exists());
}

#[test]
fn per_shape_is_required_and_zero_is_refused() {
    let d = tmp();
    release(&d, Some(shapes()));
    write_lines(&d.join("val.jsonl"), &val_rows());
    let none = Command::new(BIN)
        .args(["build-requests", "--release"]).arg(&d)
        .arg("--val-requests").arg(d.join("val.jsonl"))
        .arg("--out").arg(d.join("req.jsonl"))
        .output()
        .unwrap();
    assert_eq!(none.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&none.stderr).contains("--per-shape"));
    let zero = run(&d, &d.join("val.jsonl"), &d.join("req.jsonl"), "0");
    assert_eq!(zero.status.code(), Some(3), "{}", String::from_utf8_lossy(&zero.stderr));
    assert!(!d.join("req.jsonl").exists());
}
