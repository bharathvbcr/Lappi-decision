//! v2 plan section 6, gates 4 and 5, driven through the two binaries a person runs.
//!
//! * `qd-release-check check` has an `adapter_visibility` line: a public release trained on a
//!   BY-SA, research-only or `vani-agreed` row does not pass, and a manifest that never said
//!   `private` is public. BY-SA stays admissible for a private adapter (v1's pools read it).
//! * `qd-train-data` has a `mac_only` tier (owner ruling 12): `vani-agreed` rows in a pool whose
//!   run is cloud, or does not say it is on the Mac, are a hit.
//!
//! Every release case asserts on the `adapter_visibility` line itself, so the other checks' verdicts
//! (the results file holds only a header) cannot hide or fake it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use sha2::Digest;

const RELEASE_BIN: &str = env!("CARGO_BIN_EXE_qd-release-check");
const TRAIN_DATA_BIN: &str = env!("CARGO_BIN_EXE_qd-train-data");

static N: AtomicU64 = AtomicU64::new(0);

fn tmp() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "qd-pp-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn entry(id: &str, source: &str, licence: &str) -> Value {
    json!({"row_id": id, "source_id": source, "licence_id": licence})
}

/// A release whose train manifest is `train` (as bytes), recorded with `recorded_sha` when given
/// (the file's own sha256 otherwise) and `release_visibility` in the release manifest when given.
fn release(
    dir: &Path,
    train: Option<&str>,
    recorded_sha: Option<&str>,
    release_visibility: Option<&str>,
) -> PathBuf {
    let rel = dir.join("rel");
    std::fs::create_dir_all(&rel).unwrap();
    let train_path = rel.join("train-manifest.json");
    let mut tm = json!({"ledger_binding": {"verified": true, "train_manifest_sha256": "ab".repeat(32)}});
    if let Some(bytes) = train {
        std::fs::write(&train_path, bytes).unwrap();
        tm["path"] = json!(train_path.display().to_string());
        tm["sha256"] = json!(
            recorded_sha.map_or_else(
                || {
                    sha2::Sha256::digest(bytes.as_bytes())
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                },
                str::to_owned,
            )
        );
    } else {
        tm["path"] = json!(rel.join("absent.json").display().to_string());
        tm["sha256"] = json!("00".repeat(32));
    }
    let mut m = json!({
        "format": "qd-release.v2",
        "trained_families": ["fam.b"],
        "excluded_families": [],
        "trained_slot_shapes": {"fam.b": [["choice:4"]]},
        "expected_identity": {"weight_hash": "w", "calibration_hash": "c"},
        "train_manifest": tm,
    });
    if let Some(v) = release_visibility {
        m["adapter_visibility"] = json!(v);
    }
    std::fs::write(rel.join("release_manifest.json"), m.to_string()).unwrap();
    rel
}

/// The `adapter_visibility` line of `qd-release-check check`, and the exit code.
fn visibility_line(rel: &Path, dir: &Path) -> (Option<i32>, String) {
    let results = dir.join("results.jsonl");
    std::fs::write(
        &results,
        json!({"kind": "header", "format": "qd-replay.v1", "weight_hash": "w", "calibration_hash": "c"}).to_string() + "\n",
    )
    .unwrap();
    let out = Command::new(RELEASE_BIN)
        .args(["check", "--release"])
        .arg(rel)
        .arg("--results")
        .arg(&results)
        .args(["--min-answered", "1", "--repo"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let line = stdout
        .lines()
        .find(|l| l.contains("adapter_visibility"))
        .unwrap_or("")
        .to_owned();
    (out.status.code(), line)
}

fn manifest(visibility: Option<&str>, entries: Vec<Value>) -> String {
    let mut m = json!({"entries": entries});
    if let Some(v) = visibility {
        m["adapter_visibility"] = json!(v);
    }
    m.to_string()
}

fn line_for(train: &str, release_visibility: Option<&str>) -> String {
    let d = tmp();
    let rel = release(&d, Some(train), None, release_visibility);
    visibility_line(&rel, &d).1
}

#[test]
fn a_public_release_trained_on_a_by_sa_row_is_refused() {
    let train = manifest(Some("public"), vec![entry("r0", "vitaminc", "cc-by-sa-3.0"), entry("r1", "s", "mit")]);
    let line = line_for(&train, None);
    assert!(line.starts_with("Fail adapter_visibility"), "{line:?}");
    assert!(line.contains("r0"), "the refusal names the row: {line:?}");
}

#[test]
fn a_manifest_that_never_says_private_is_public_and_is_refused() {
    let train = manifest(None, vec![entry("r0", "squad", "cc-by-sa-4.0")]);
    let line = line_for(&train, None);
    assert!(line.starts_with("Fail adapter_visibility"), "{line:?}");
    assert!(line.contains("absent"), "the refusal says why: {line:?}");
}

#[test]
fn a_private_release_may_carry_by_sa_research_only_and_vani_agreed_rows() {
    let train = manifest(
        Some("private"),
        vec![
            entry("r0", "vitaminc", "cc-by-sa-3.0"),
            entry("r1", "mind2web", "cc-by-4.0-research-only"),
            entry("r2", "vani-agreed", "synthetic-by-rule"),
        ],
    );
    let line = line_for(&train, Some("private"));
    assert!(line.starts_with("Pass adapter_visibility"), "{line:?}");
}

#[test]
fn a_research_only_row_and_a_vani_agreed_row_each_refuse_a_public_release() {
    for (source, licence) in [("mind2web", "CC BY 4.0 research only"), ("vani-agreed", "mit"), ("Vani-Agreed/seniority", "mit")] {
        let train = manifest(Some("public"), vec![entry("r0", source, licence)]);
        let line = line_for(&train, None);
        assert!(line.starts_with("Fail adapter_visibility"), "{source} {licence}: {line:?}");
    }
}

#[test]
fn a_public_release_with_no_restricted_row_passes() {
    let train = manifest(Some("public"), vec![entry("r0", "s", "mit"), entry("r1", "t", "cc-by-4.0")]);
    let line = line_for(&train, None);
    assert!(line.starts_with("Pass adapter_visibility"), "{line:?}");
}

#[test]
fn a_release_manifest_that_says_public_overrides_a_private_train_manifest() {
    let train = manifest(Some("private"), vec![entry("r0", "vitaminc", "cc-by-sa-3.0")]);
    let line = line_for(&train, Some("public"));
    assert!(line.starts_with("Fail adapter_visibility"), "{line:?}");
}

#[test]
fn an_unrecognised_visibility_is_refused_not_guessed() {
    for bad in ["Private", "internal", ""] {
        let train = manifest(Some(bad), vec![entry("r0", "s", "mit")]);
        let line = line_for(&train, None);
        assert!(line.starts_with("Fail adapter_visibility"), "{bad:?}: {line:?}");
    }
}

#[test]
fn a_by_sa_licence_only_the_histogram_counts_still_refuses_a_public_release() {
    let mut m: Value = serde_json::from_str(&manifest(Some("public"), vec![entry("r0", "s", "mit")])).unwrap();
    m["licence_histogram"] = json!({"mit": 1, "cc-by-sa-4.0": 3});
    let line = line_for(&m.to_string(), None);
    assert!(line.starts_with("Fail adapter_visibility"), "{line:?}");
}

#[test]
fn a_train_manifest_that_is_not_the_one_the_release_records_is_refused() {
    let d = tmp();
    let train = manifest(Some("private"), vec![entry("r0", "s", "mit")]);
    let rel = release(&d, Some(&train), Some(&"11".repeat(32)), None);
    let line = visibility_line(&rel, &d).1;
    assert!(line.starts_with("Fail adapter_visibility"), "{line:?}");
    assert!(line.contains("not the manifest these weights came from"), "{line:?}");
}

#[test]
fn an_unreadable_train_manifest_is_not_run_never_a_pass() {
    let d = tmp();
    let rel = release(&d, None, None, None);
    let (_, line) = visibility_line(&rel, &d);
    assert!(line.starts_with("NotRun adapter_visibility"), "{line:?}");
}

#[test]
fn a_manifest_with_no_entries_or_an_unreadable_entry_is_refused() {
    for train in [
        json!({"adapter_visibility": "private", "entries": []}).to_string(),
        json!({"adapter_visibility": "private"}).to_string(),
        json!({"adapter_visibility": "private", "entries": [{"row_id": "r0"}]}).to_string(),
    ] {
        let line = line_for(&train, None);
        assert!(line.starts_with("Fail adapter_visibility"), "{train}: {line:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// qd-train-data: the mac_only tier

/// An examples.jsonl of one `vani-agreed` row and one synthetic row, and a target no row holds.
fn train_set(dir: &Path) -> (PathBuf, PathBuf) {
    let rows = [
        json!({"id": "v0", "source_id": "vani-agreed", "family_id": "career.jd_seniority",
               "context": "a posting about a senior platform role", "options": ["senior", "mid"]}),
        json!({"id": "s0", "source_id": "lappi/synth-career", "family_id": "career.jd_seniority",
               "context": "a synthetic posting about an intern role", "options": ["senior", "mid"]}),
    ];
    let examples = dir.join("examples.jsonl");
    std::fs::write(
        &examples,
        rows.iter().map(Value::to_string).collect::<Vec<_>>().join("\n") + "\n",
    )
    .unwrap();
    let targets = dir.join("targets.jsonl");
    std::fs::write(
        &targets,
        json!({"id": "t0", "text": "lighthouse keepers counting migrating cranes over a frozen northern estuary at dawn"}).to_string() + "\n",
    )
    .unwrap();
    (examples, targets)
}

fn mac_only_line(extra: &[&str]) -> String {
    let d = tmp();
    let (examples, targets) = train_set(&d);
    let out = Command::new(TRAIN_DATA_BIN)
        .arg("--train")
        .arg(&examples)
        .arg("--target")
        .arg(format!("t={}", targets.display()))
        .args(extra)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    stdout
        .lines()
        .find(|l| l.starts_with("tier mac_only"))
        .unwrap_or_else(|| panic!("no mac_only tier in: {stdout}"))
        .to_owned()
}

#[test]
fn vani_agreed_rows_in_a_cloud_run_are_a_hit() {
    let line = mac_only_line(&["--run-target", "cloud"]);
    assert!(line.starts_with("tier mac_only: FAILED hits=1"), "{line:?}");
    assert!(line.contains("v0"), "names the row: {line:?}");
}

#[test]
fn vani_agreed_rows_in_a_run_that_does_not_say_where_it_trains_are_a_hit() {
    let line = mac_only_line(&[]);
    assert!(line.starts_with("tier mac_only: FAILED hits=1"), "{line:?}");
}

#[test]
fn vani_agreed_rows_in_a_run_stated_to_be_on_the_mac_pass_this_tier() {
    let line = mac_only_line(&["--run-target", "mac"]);
    assert!(line.starts_with("tier mac_only: PASS hits=0"), "{line:?}");
}

#[test]
fn an_unknown_run_target_is_a_usage_error() {
    let d = tmp();
    let (examples, targets) = train_set(&d);
    let out = Command::new(TRAIN_DATA_BIN)
        .arg("--train")
        .arg(&examples)
        .arg("--target")
        .arg(format!("t={}", targets.display()))
        .args(["--run-target", "laptop"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}
