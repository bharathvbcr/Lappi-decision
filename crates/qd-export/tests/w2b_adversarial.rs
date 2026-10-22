//! Adversarial tests against lane W2b (reviewer branch adv/ft-c5ea3e71-w2b): the calibration
//! gate's switch to `CalibrationTable::lookup` and `verify_release_train_manifest`.
//!
//! Each test states the invariant it expects. A failing test is a reported break.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::mpsc;
use std::time::Duration;

use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::calibration_fit::ECE_THRESHOLD;
use qd_runtime::schema::SlotKind;
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-export");

// --- calibration gate through the binary --------------------------------------------------------

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

struct Case {
    fx: common::Fixture,
    table: PathBuf,
    table_sha: String,
}

/// A table whose letters are exactly `letters` (raw key spellings, as a hand-edited file could
/// carry them), plus a span entry.
fn case(letters: &[&str]) -> Case {
    let fx = common::standard();
    let mut doc = serde_json::to_value(
        CalibrationTable::new("adv")
            .with_letters(SlotKind::Choice, 5, entry())
            .with_span(entry()),
    )
    .unwrap();
    let map = doc["letters"].as_object_mut().unwrap();
    map.clear();
    for key in letters {
        map.insert((*key).to_string(), serde_json::to_value(entry()).unwrap());
    }
    let path = write_json(&fx.dir.0.join("table.json"), &doc);
    let table_sha = qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(&path).unwrap()));
    Case {
        fx,
        table: path,
        table_sha,
    }
}

fn report(c: &Case, keys: &[&str]) -> Value {
    let entries: serde_json::Map<String, Value> = keys
        .iter()
        .map(|k| {
            (
                (*k).to_string(),
                json!({"kind": "choice", "n": 500, "scored": {"ece_after": ran(0.01)}}),
            )
        })
        .collect();
    json!({
        "ece": {"bins": 15, "min_samples": 100, "threshold": ECE_THRESHOLD},
        "entries": entries,
        "span": {"fitted": true, "scored": {"ece_after": ran(0.02)}},
        "tables": [{"file": "table.json", "sha256": c.table_sha}],
    })
}

fn train_manifest(c: &Case, shapes: &Value) -> PathBuf {
    let mut entries: Vec<Value> = Vec::new();
    for (family, list) in shapes.as_object().unwrap() {
        for shape in list.as_array().unwrap() {
            let i = entries.len();
            entries.push(json!({
                "row_id": format!("train-row-{i}"), "content_hash": format!("{i:064x}"),
                "split": "train", "source_id": "test/source", "host": "test",
                "family_id": family, "slot_shape": shape, "repo_key": format!("repo-{i}"),
                "identity_key": format!("identity-{i}"), "licence_id": "MIT", "obligations": [],
            }));
        }
    }
    let mut m = json!({
        "manifest_format_version": 1, "split": "train", "data_snapshot_hash": "ab".repeat(32),
        "n_rows": entries.len(), "held_out_families": ["code.language_id", "qa.answerability"],
        "entries": entries,
    });
    m["slot_shape_digest"] = common::slot_shape_digest(m["entries"].as_array().unwrap());
    write_json(&c.fx.dir.0.join("train-shapes.json"), &m)
}

fn run(c: &Case, report: &Value, shapes: &Value, extra: &[&str]) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.arg("--source")
        .arg(&c.fx.source)
        .arg("--base-snapshot")
        .arg(&c.fx.snapshot)
        .arg("--tokenizer-sha256")
        .arg(common::tokenizer_sha256(&c.fx.snapshot))
        .arg("--expect-vocab-size")
        .arg(common::VOCAB.to_string())
        .arg("--train-manifest")
        .arg(train_manifest(c, shapes))
        .arg("--calibration")
        .arg(&c.table)
        .arg("--calibration-report")
        .arg(write_json(&c.fx.dir.0.join("report.json"), report))
        .arg("--out")
        .arg(&c.fx.out);
    cmd.args(extra);
    cmd.output().expect("the binary runs")
}

fn shapes_with(key: &str) -> Value {
    json!({"code.defect_class": [[key, "span"]], "devcouncil.verdict": [[key]]})
}

/// Every spelling the runtime would never build is refused, whatever the table holds.
#[test]
fn ambiguous_shape_keys_are_all_refused() {
    for key in [
        "choice:+5",
        "choice: 5",
        "choice:5 ",
        "choice:5\n",
        "choice:\u{0665}",
        "choice:\u{ff15}",
        "choice:99999999999999999999999999",
        "choice:-5",
        "choice:",
        "choice:5:5",
        "Choice:5",
        "CHOICE:5",
        "ranking:5",
        "span:0",
        "span:",
        "choice:005",
        "score:+5",
    ] {
        // The table carries the very same spelling, and the report passes it: only the key's
        // spelling can refuse this.
        let c = case(&[key, "choice:5"]);
        let out = run(&c, &report(&c, &[key, "choice:5"]), &shapes_with(key), &[]);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{key:?} was not refused: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !c.fx.out.exists(),
            "{key:?}: a refused export left a release"
        );
    }
}

/// Positive control: a canonical `score:N` shape the table holds is accepted (the runtime looks
/// `score` up in `letters` too).
#[test]
fn a_canonical_score_shape_with_an_entry_is_accepted() {
    let c = case(&["score:5", "choice:5"]);
    let out = run(
        &c,
        &report(&c, &["score:5", "choice:5"]),
        &json!({"code.defect_class": [["choice:5", "span"]], "devcouncil.verdict": [["score:5"]]}),
        &[],
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// An excluded family's shapes are not gated: a spelling the runtime would miss in an EXCLUDED
/// family must not block the export of the others (and the exclusion is recorded).
#[test]
fn exclude_family_still_exempts_a_family_with_an_unlookable_shape() {
    let c = case(&["choice:5"]);
    let shapes =
        json!({"code.defect_class": [["choice:05", "span"]], "devcouncil.verdict": [["choice:5"]]});
    let out = run(
        &c,
        &report(&c, &["choice:5"]),
        &shapes,
        &[
            "--exclude-family",
            "code.defect_class=span and 05 shape are not served",
        ],
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// --- verify_release_train_manifest --------------------------------------------------------------

fn sha(bytes: &[u8]) -> String {
    qd_runtime::hex(&qd_runtime::sha256(bytes))
}

/// A hand-written release directory whose manifest block binds `manifest_path`'s bytes `body`.
fn release(dir: &Path, manifest_path: &Path, body: &[u8], ids: &[&str]) -> PathBuf {
    let rel = dir.join("release");
    std::fs::create_dir_all(&rel).unwrap();
    let digest = sha(body);
    let doc = json!({"train_manifest": {
        "path": manifest_path.display().to_string(), "sha256": digest,
        "ledger_binding": {"verified": true, "train_manifest_sha256": digest, "ft_row_ids": ids},
    }});
    write_json(&rel.join("release_manifest.json"), &doc);
    rel
}

fn rewrite_hashes(rel: &Path, new_sha: &str) {
    let p = rel.join("release_manifest.json");
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    doc["train_manifest"]["sha256"] = json!(new_sha);
    doc["train_manifest"]["ledger_binding"]["train_manifest_sha256"] = json!(new_sha);
    write_json(&p, &doc);
}

fn ft_ledger(dir: &Path, rows: &[Value]) -> PathBuf {
    let path = dir.join("ft-ledger.jsonl");
    std::fs::write(
        &path,
        rows.iter().map(|r| format!("{r}\n")).collect::<String>(),
    )
    .unwrap();
    path
}

fn ft_row(id: &str, train_sha: &str) -> Value {
    json!({"row_id": id, "run_kind": "ft", "status": "completed",
           "train_manifest_sha256": train_sha})
}

/// An attacker who swaps the manifest and rewrites BOTH release hashes defeats the check when
/// no ft-ledger is given, and the caller gets `Ok(sha)`, the same as a check that ran against
/// an anchor. A check that could not anchor must not report what a passing check reports.
#[test]
fn a_swapped_manifest_with_rewritten_release_hashes_is_not_ok_without_a_ledger() {
    let dir = common::TempDir::new("adv-swap");
    let manifest = dir.0.join("train.json");
    std::fs::write(&manifest, b"{\"original\":true}").unwrap();
    let rel = release(&dir.0, &manifest, b"{\"original\":true}", &["ft-a"]);
    let evil = b"{\"original\":false}";
    std::fs::write(&manifest, evil).unwrap();
    rewrite_hashes(&rel, &sha(evil));
    let verdict = qd_export::verify_release_train_manifest(&rel, None, None);
    assert!(
        verdict.is_err(),
        "swap with rewritten hashes passed with no ledger: {verdict:?}"
    );
}

/// The same swap with the ledger as anchor is refused (the ledger rows still name the original).
#[test]
fn a_swapped_manifest_with_rewritten_release_hashes_is_refused_with_a_ledger() {
    let dir = common::TempDir::new("adv-swap-ledger");
    let manifest = dir.0.join("train.json");
    let original: &[u8] = b"{\"original\":true}";
    std::fs::write(&manifest, original).unwrap();
    let rel = release(&dir.0, &manifest, original, &["ft-a"]);
    let evil: &[u8] = b"{\"original\":false}";
    std::fs::write(&manifest, evil).unwrap();
    rewrite_hashes(&rel, &sha(evil));
    let ledger = ft_ledger(&dir.0, &[ft_row("ft-a", &sha(original))]);
    assert!(qd_export::verify_release_train_manifest(&rel, Some(&ledger), None).is_err());
}

/// A manifest path that is not a regular file must be refused from its type, as `read_bounded`
/// does for the export. `/dev/zero` reports length 0, so the length bound passes and the hash
/// would read forever.
#[test]
fn a_non_regular_manifest_path_is_refused_not_hashed_forever() {
    let dir = common::TempDir::new("adv-devzero");
    let manifest = dir.0.join("train.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let rel = release(&dir.0, &manifest, b"{}", &["ft-a"]);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let verdict =
            qd_export::verify_release_train_manifest(&rel, None, Some(Path::new("/dev/zero")));
        let _sent = tx.send(verdict.is_err());
    });
    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(refused) => assert!(refused, "/dev/zero verified"),
        Err(_) => panic!("hashing /dev/zero did not finish in 10 s: no regular-file check"),
    }
}

#[test]
fn a_directory_as_the_manifest_is_refused() {
    let dir = common::TempDir::new("adv-dir");
    let manifest = dir.0.join("train.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let rel = release(&dir.0, &manifest, b"{}", &["ft-a"]);
    assert!(qd_export::verify_release_train_manifest(&rel, None, Some(&dir.0)).is_err());
}

#[test]
fn a_ledger_path_that_does_not_exist_is_refused() {
    let dir = common::TempDir::new("adv-noledger");
    let manifest = dir.0.join("train.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let rel = release(&dir.0, &manifest, b"{}", &["ft-a"]);
    let missing = dir.0.join("nope.jsonl");
    assert!(qd_export::verify_release_train_manifest(&rel, Some(&missing), None).is_err());
}

#[test]
fn malformed_ft_row_shas_are_refused() {
    let dir = common::TempDir::new("adv-badsha");
    let manifest = dir.0.join("train.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let rel = release(&dir.0, &manifest, b"{}", &["ft-a"]);
    let good = sha(b"{}");
    for bad in [
        format!("sha256:{good}"),
        format!(" {good}"),
        format!("{good}\n"),
        good[..63].to_string(),
        String::new(),
    ] {
        let ledger = ft_ledger(&dir.0, &[ft_row("ft-a", &bad)]);
        assert!(
            qd_export::verify_release_train_manifest(&rel, Some(&ledger), None).is_err(),
            "{bad:?} accepted"
        );
    }
}

/// Upper-case hex of the right digest names the same bytes; either answer is defensible, so this
/// pins today's behaviour (accepted) rather than asserting a preference.
#[test]
fn an_upper_case_ft_row_sha_is_accepted_today() {
    let dir = common::TempDir::new("adv-upper");
    let manifest = dir.0.join("train.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let rel = release(&dir.0, &manifest, b"{}", &["ft-a"]);
    let ledger = ft_ledger(&dir.0, &[ft_row("ft-a", &sha(b"{}").to_ascii_uppercase())]);
    assert!(qd_export::verify_release_train_manifest(&rel, Some(&ledger), None).is_ok());
}

/// The ledger row the release names must be an ft row that completed: a row that is not
/// `run_kind: ft` or whose status is `failed` carrying the right sha anchors nothing.
#[test]
fn a_failed_or_non_ft_ledger_row_is_not_an_anchor() {
    let dir = common::TempDir::new("adv-kind");
    let manifest = dir.0.join("train.json");
    std::fs::write(&manifest, b"{}").unwrap();
    let rel = release(&dir.0, &manifest, b"{}", &["ft-a"]);
    let d = sha(b"{}");
    for row in [
        json!({"row_id": "ft-a", "run_kind": "eval", "status": "completed", "train_manifest_sha256": d}),
        json!({"row_id": "ft-a", "run_kind": "ft", "status": "failed", "train_manifest_sha256": d}),
    ] {
        let ledger = ft_ledger(&dir.0, &[row.clone()]);
        assert!(
            qd_export::verify_release_train_manifest(&rel, Some(&ledger), None).is_err(),
            "{row} anchored the release"
        );
    }
}

/// A 200 MB manifest: hashing is streamed. Run under `/usr/bin/time -l` to read the peak RSS;
/// the test itself asserts only the verdict and prints the wall time.
#[test]
#[ignore = "writes 200 MB; run explicitly under /usr/bin/time -l"]
fn a_200_mb_manifest_verifies() {
    use std::io::Write;
    let dir = common::TempDir::new("adv-200mb");
    let manifest = dir.0.join("train.json");
    let mut f = std::fs::File::create(&manifest).unwrap();
    let chunk = vec![b' '; 1 << 20];
    // Hashed incrementally here too, so the peak RSS belongs to the function under test.
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    sha2::Digest::update(&mut hasher, b"{}");
    f.write_all(b"{}").unwrap();
    for _ in 0..200 {
        f.write_all(&chunk).unwrap();
        sha2::Digest::update(&mut hasher, &chunk);
    }
    drop(f);
    let digest = qd_runtime::hex(&sha2::Digest::finalize(hasher));
    let rel = dir.0.join("release");
    std::fs::create_dir_all(&rel).unwrap();
    write_json(
        &rel.join("release_manifest.json"),
        &json!({"train_manifest": {"path": manifest.display().to_string(), "sha256": digest,
            "ledger_binding": {"verified": true, "train_manifest_sha256": digest, "ft_row_ids": ["x"]}}}),
    );
    let t = std::time::Instant::now();
    // The verifier refuses to run without an anchor, so the ledger the release names is given;
    // the timing and RSS being measured are the hash's.
    let ledger = ft_ledger(&dir.0, &[ft_row("x", &digest)]);
    let got = qd_export::verify_release_train_manifest(&rel, Some(&ledger), None).unwrap();
    eprintln!("200 MB verified in {:?}", t.elapsed());
    assert_eq!(got, digest);
}
