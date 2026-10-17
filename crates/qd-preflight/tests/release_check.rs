//! `qd-release-check`, driven as a binary, over a release directory and replay results built in a
//! temp dir. The documents and wire fixtures are the repository's own, so the examples replayed
//! are the ones the contract documents.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use qd_preflight::release_check::{load_examples, request_shape, DocExample, RESULTS_FORMAT};
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_qd-release-check");
const WEIGHT: &str = "6f7b9ba73bca3a0b2e4ec9b2c0fc2aa6e7459f7e3286b26386633a222667dfcb";
const CALIB: &str = "7e56b34e378907adccc7cfc0a086507d81c762ed6627694a0a59c3c977aeb677";

static N: AtomicU64 = AtomicU64::new(0);

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("qd-rc-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn b64(s: &str) -> String {
    // Only used for contexts of 1..=3 bytes' worth of groups; the runtime is not in the loop.
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in s.as_bytes().chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for (i, sh) in [18, 12, 6, 0].iter().enumerate() {
            out.push(if i <= c.len() { T[((n >> sh) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

fn release(dir: &Path, shapes: Option<Value>, weight: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let mut m = json!({
        "format": "qd-release.v2",
        "trained_families": ["code.defect_class", "fam.b"],
        "excluded_families": [],
        "expected_identity": {"weight_hash": weight, "calibration_hash": CALIB},
    });
    if let Some(s) = shapes {
        m["trained_slot_shapes"] = s;
    }
    std::fs::write(dir.join("release_manifest.json"), m.to_string()).unwrap();
}

fn good_shapes() -> Value {
    json!({"code.defect_class": [["choice:5", "span"]], "fam.b": [["choice:4"]]})
}

fn ok_reply() -> Value {
    json!({"status": "ok", "schema_version": 1, "backend": "qd-metal/test", "degraded": false,
           "slots": {"x": {"value": "a", "conformal_set": ["a"], "score": 0.5, "noul": false, "degraded": false}}})
}

fn refused_reply(kind: &str) -> Value {
    json!({"status": "refused", "schema_version": 1, "refusal": {"kind": kind}, "message": "m"})
}

fn fam_b(i: usize) -> Value {
    let ctx = format!("ctx{i}");
    json!({"schema_version": 1, "task": "fam.b", "context_b64": b64(&ctx), "context_len": ctx.len(),
           "question": "Q?", "slots": [{"name": "x", "type": "choice", "options": ["a", "b", "c"]}],
           "route": "generic", "example_id": format!("t:{i}"), "metadata": {}})
}

fn replay_line(n: usize, request: &Value, served_by: &str, reply: Value) -> Value {
    json!({"kind": "replay", "id": format!("r{n}"), "source": "test", "request": request,
           "exit_code": 0, "served_by": served_by, "reply": reply, "seconds": 0.1})
}

fn header(weight: &str) -> Value {
    json!({"kind": "header", "format": RESULTS_FORMAT, "weight_hash": weight, "calibration_hash": CALIB})
}

/// Every documented example answered, plus `n_b` distinct answered requests of fam.b.
fn good_results(n_b: usize) -> Vec<Value> {
    let mut lines = vec![header(WEIGHT)];
    let examples = load_examples(&repo()).unwrap();
    let mut k = 0;
    for ex in &examples {
        let req = match ex {
            DocExample::Full { request, .. } => request.clone(),
            DocExample::Row { task, question, slots, .. } => json!({
                "schema_version": 1, "task": task, "context_b64": "eA==", "context_len": 1,
                "question": question, "slots": slots, "route": "generic",
                "example_id": "t", "metadata": {}}),
        };
        k += 1;
        // The caller contract's rows name tasks this fixture release does not admit.
        let reply = if matches!(req["task"].as_str(), Some("code.defect_class" | "fam.b")) { ok_reply() } else { refused_reply("task_not_trained") };
        lines.push(replay_line(k, &req, "socket", reply));
    }
    for i in 0..n_b {
        k += 1;
        lines.push(replay_line(k, &fam_b(i), "socket", ok_reply()));
    }
    lines
}

fn write_results(dir: &Path, lines: &[Value]) -> PathBuf {
    let p = dir.join("results.jsonl");
    std::fs::write(&p, lines.iter().map(|l| l.to_string() + "\n").collect::<String>()).unwrap();
    p
}

fn check(rel: &Path, results: &Path, n: usize, extra: &[&str]) -> (Option<i32>, String) {
    let out: Output = Command::new(BIN)
        .args(["check", "--release"]).arg(rel)
        .arg("--results").arg(results)
        .arg("--min-answered").arg(n.to_string())
        .arg("--repo").arg(repo())
        .args(extra)
        .output()
        .unwrap();
    (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned())
}

fn last_line(s: &str) -> &str {
    s.trim_end().lines().last().unwrap_or("")
}

#[test]
fn a_release_whose_every_family_shape_and_example_was_answered_through_the_socket_passes() {
    let d = tmp("pass");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let r = write_results(&d, &good_results(3));
    let (code, out) = check(&d.join("rel"), &r, 1, &[]);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(last_line(&out), "RESULT: PASS");
}

#[test]
fn a_family_that_was_only_ever_refused_is_not_served() {
    let d = tmp("refused-only");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let mut lines = good_results(0);
    for i in 0..4 {
        lines.push(replay_line(900 + i, &fam_b(i), "socket", refused_reply("calibration_entry_missing")));
    }
    let r = write_results(&d, &lines);
    let (code, out) = check(&d.join("rel"), &r, 1, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("fam.b") && out.contains("calibration_entry_missing"), "{out}");
    assert_eq!(last_line(&out), "RESULT: FAIL");
}

#[test]
fn n_counts_distinct_requests_not_repeats() {
    let d = tmp("distinct");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let mut lines = good_results(0);
    for i in 0..5 {
        lines.push(replay_line(900 + i, &fam_b(0), "socket", ok_reply()));
    }
    let r = write_results(&d, &lines);
    let (code, out) = check(&d.join("rel"), &r, 2, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("fam.b") && out.contains("1 answered"), "{out}");
}

#[test]
fn a_replay_with_the_wrong_weight_hash_fails() {
    let d = tmp("hash");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let mut lines = good_results(2);
    lines[0] = header(&"0".repeat(64));
    let r = write_results(&d, &lines);
    let (code, out) = check(&d.join("rel"), &r, 1, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("weight_hash"), "{out}");
}

#[test]
fn a_replay_that_fell_back_in_process_is_not_the_product_path() {
    let d = tmp("sidecar");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let mut lines = good_results(2);
    for l in lines.iter_mut().skip(1) {
        l["served_by"] = json!("sidecar");
    }
    let r = write_results(&d, &lines);
    let (code, out) = check(&d.join("rel"), &r, 1, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("replay_path"), "{out}");
}

#[test]
fn a_documented_example_that_was_never_replayed_fails() {
    let d = tmp("example");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let mut lines = good_results(2);
    // Drop the schema-api request example.
    lines.retain(|l| l["request"]["example_id"] != json!("qdm:code.defect_class:doc-example"));
    let r = write_results(&d, &lines);
    let (code, out) = check(&d.join("rel"), &r, 0, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("not replayed") && out.contains("docs/schema-api.md"), "{out}");
}

#[test]
fn a_documented_example_refused_with_an_undocumented_kind_fails() {
    let d = tmp("undocumented");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let mut lines = good_results(2);
    for l in lines.iter_mut() {
        if l["request"]["example_id"] == json!("qdm:code.defect_class:doc-example") {
            l["reply"] = refused_reply("made_up_kind");
        }
    }
    let r = write_results(&d, &lines);
    let (code, out) = check(&d.join("rel"), &r, 0, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("made_up_kind"), "{out}");
}

#[test]
fn a_release_that_records_no_admitted_shapes_cannot_pass() {
    let d = tmp("noshapes");
    release(&d.join("rel"), None, WEIGHT);
    let r = write_results(&d, &good_results(3));
    let (code, out) = check(&d.join("rel"), &r, 1, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("trained_slot_shapes"), "{out}");
}

#[test]
fn an_unreadable_input_is_not_run_never_a_pass() {
    let d = tmp("notrun");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let (code, out) = check(&d.join("rel"), &d.join("missing.jsonl"), 1, &[]);
    assert_eq!(code, Some(3), "{out}");
    assert_eq!(last_line(&out), "RESULT: NOT_RUN");
}

#[test]
fn the_request_shape_counts_the_reserved_noul_row() {
    assert_eq!(request_shape(&fam_b(0)), Some(vec!["choice:4".to_string()]));
}

#[test]
fn headline_figures_are_over_the_admitted_families_with_numerators_and_denominators() {
    let d = tmp("metrics");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let r = write_results(&d, &good_results(3));
    // fam.b: 2 of 3 correct; code.defect_class: 1 of 2; an unadmitted family: 3 of 3.
    let rows = [
        ("fam.b", true), ("fam.b", true), ("fam.b", false),
        ("code.defect_class", true), ("code.defect_class", false),
        ("other.family", true), ("other.family", true), ("other.family", true),
    ];
    let verdicts: String = rows.iter().enumerate().map(|(i, (f, c))| {
        json!({"family_id": f, "kind": "choice", "correct": c, "row_id": format!("row{i}"), "slot_name": "x"}).to_string() + "\n"
    }).collect();
    let calib: String = rows.iter().enumerate().map(|(i, (_, c))| {
        // Row 2 abstains; every other row answers.
        json!({"row_id": format!("row{i}"), "slot_name": "x", "scored": true, "correct": c,
               "abstained_noul_row_or_margin": i == 2}).to_string() + "\n"
    }).collect();
    let v = d.join("verdicts.jsonl");
    let c = d.join("calib-rows.jsonl");
    std::fs::write(&v, verdicts).unwrap();
    std::fs::write(&c, calib).unwrap();
    let (code, out) = check(&d.join("rel"), &r, 1, &["--verdicts", v.to_str().unwrap(), "--calib-rows", c.to_str().unwrap()]);
    assert_eq!(code, Some(0), "{out}");
    assert!(out.contains("headline over admitted_population (2 families)"), "{out}");
    assert!(out.contains("choice_top1: 0.6000 (3/5)"), "{out}");
    assert!(out.contains("coverage: 0.8000 (4/5)"), "{out}");
    assert!(out.contains("precision_on_answered: 0.7500 (3/4)"), "{out}");
    assert!(out.contains("excluded rows (families not admitted): 3"), "{out}");
    assert!(out.contains("NOT_THE_RELEASES_all_rows_choice_top1: 0.7500 (6/8)"), "{out}");
}

/// A stand-in for `qd`: answers a status op with the release's hashes and every other request with
/// an ok reply, and says on stderr that the socket served it, as `qd oneshot` does.
fn fake_qd(dir: &Path, served_by: &str) -> PathBuf {
    let p = dir.join("fake-qd.sh");
    let script = format!(
        "#!/bin/sh\nIN=$(cat)\ncase \"$IN\" in\n  *'\"op\"'*) echo '{{\"status\":\"status\",\"backend\":{{\"weight_hash\":\"{WEIGHT}\",\"calibration_hash\":\"{CALIB}\"}}}}' ;;\n  *) echo '{}' ;;\nesac\necho 'qd oneshot: served by {served_by}' 1>&2\n",
        ok_reply()
    );
    std::fs::write(&p, script).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn replay(d: &Path, qd: &Path, requests: &Path) -> (Option<i32>, PathBuf) {
    let out = d.join("replayed.jsonl");
    let o = Command::new(BIN)
        .args(["replay", "--release"]).arg(d.join("rel"))
        .arg("--socket").arg(d.join("q.sock"))
        .arg("--out").arg(&out)
        .arg("--requests").arg(requests)
        .arg("--qd").arg(qd)
        .arg("--repo").arg(repo())
        .args(["--socket-timeout-ms", "2000"])
        .output()
        .unwrap();
    (o.status.code(), out)
}

#[test]
fn one_replay_command_produces_a_results_file_the_check_accepts() {
    let d = tmp("replay");
    release(&d.join("rel"), Some(json!({"code.defect_class": [["choice:5", "span"]], "fam.b": [["choice:4"]]})), WEIGHT);
    let reqs = d.join("requests.jsonl");
    std::fs::write(&reqs, (0..3).map(|i| fam_b(i).to_string() + "\n").collect::<String>()).unwrap();
    let qd = fake_qd(&d, "socket");
    let (code, results) = replay(&d, &qd, &reqs);
    assert_eq!(code, Some(0));
    let (code, out) = check(&d.join("rel"), &results, 3, &[]);
    // fam.b has 3 distinct answered requests and the documented defect example is answered once.
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("code.defect_class") && out.contains("1 answered"), "{out}");
    let (code, out) = check(&d.join("rel"), &results, 1, &[]);
    assert_eq!(code, Some(0), "{out}");
}

#[test]
fn a_replay_through_a_qd_that_fell_back_in_process_is_recorded_as_such_and_fails_the_check() {
    let d = tmp("replay-sidecar");
    release(&d.join("rel"), Some(good_shapes()), WEIGHT);
    let reqs = d.join("requests.jsonl");
    std::fs::write(&reqs, fam_b(0).to_string() + "\n").unwrap();
    let qd = fake_qd(&d, "sidecar");
    let (code, results) = replay(&d, &qd, &reqs);
    assert_eq!(code, Some(0));
    let (code, out) = check(&d.join("rel"), &results, 1, &[]);
    assert_eq!(code, Some(1), "{out}");
    assert!(out.contains("replay_path"), "{out}");
}
