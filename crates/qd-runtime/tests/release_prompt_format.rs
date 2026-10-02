//! A release binds the prompt format its tower was trained on, and the runtime refuses any other.
//!
//! `campaign/v5-preregistered.DRAFT.json` `format.refusal_both_ways.serving`: the manifest format
//! moved from `qd-release.v1` to `qd-release.v2`, and `expected_identity.prompt_format` is required
//! and must equal `render::PROMPT_FORMAT`. So:
//!
//! * a v5 runtime refuses every v4 release -- it says `qd-release.v1` and carries no
//!   `prompt_format` -- by its format;
//! * a v4-era runtime refuses a v5 release by its format (`qd-release.v2` is not `qd-release.v1`);
//! * a `qd-release.v2` manifest whose tower was trained on another format (a v4 checkpoint
//!   exported after the change is stamped 1 by `qd-export`), or that states none, is refused by
//!   its `prompt_format`, naming both numbers.
//!
//! The release directories are synthetic and hand-written, so this file pins the wire contract
//! (the strings and numbers a manifest carries) rather than the constants that produce them.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use qd_runtime::calibration::CalibrationTable;
use qd_runtime::release::{Release, ReleaseRefusal, Tower, MANIFEST_FORMAT};
use serde_json::{json, Value};

static N: AtomicUsize = AtomicUsize::new(0);

fn sha(bytes: &[u8]) -> String {
    qd_runtime::hex(&qd_runtime::sha256(bytes))
}

/// A release directory, removed on drop.
struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        // A leftover temp dir only costs disk; the test already has its answer.
        let _ignored = std::fs::remove_dir_all(&self.0);
    }
}

/// A complete format-2 release whose manifest `edit` may then change.
fn release(edit: impl FnOnce(&mut Value)) -> Dir {
    let dir = std::env::temp_dir().join(format!(
        "qd-release-fmt-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    let config = br#"{"architectures":["Qwen3_5ForConditionalGeneration"]}"#;
    let table = CalibrationTable::reference();
    let calibration = serde_json::to_vec(&table).unwrap();
    let weights = b"not read by this test";
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("calibration.json"), &calibration).unwrap();
    std::fs::write(dir.join("model.safetensors"), weights).unwrap();
    let mut manifest = json!({
        "format": "qd-release.v2",
        "expected_identity": {
            "weight_hash": "a".repeat(64),
            "tokenizer_hash": "b".repeat(64),
            "config_sha256": sha(config),
            "calibration_hash": table.hash(),
            "prompt_format": 2,
        },
        "files": {
            "config.json": {"sha256": sha(config)},
            "model.safetensors": {"sha256": sha(weights)},
            "calibration.json": {"sha256": sha(&calibration)},
        },
        "calibration": {
            "file": "calibration.json",
            "file_sha256": sha(&calibration),
            "table_hash": table.hash(),
        },
    });
    edit(&mut manifest);
    std::fs::write(
        dir.join("release_manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    Dir(dir)
}

/// Both readers refuse `dir` with `kind`, and the detail names every one of `needles`.
fn refused(dir: &Path, kind: &str, needles: &[&str]) -> ReleaseRefusal {
    let tower = Tower::open(dir).expect_err("the tower must be refused");
    let err = Release::open(dir).expect_err("the release must be refused");
    assert_eq!(tower, err, "Tower::open and Release::open refuse a manifest alike");
    assert_eq!(err.kind.as_str(), kind, "{err}");
    for needle in needles {
        assert!(err.detail.contains(needle), "{needle:?} missing from: {err}");
    }
    err
}

#[test]
fn the_manifest_format_is_qd_release_v2() {
    assert_eq!(MANIFEST_FORMAT, "qd-release.v2");
}

#[test]
fn a_format_2_release_for_a_format_2_runtime_opens() {
    let dir = release(|_| {});
    let release = Release::open(&dir.0).expect("a format-2 release opens under a format-2 runtime");
    assert_eq!(release.weight_hash(), "a".repeat(64));
    Tower::open(&dir.0).expect("and so does its tower");
}

/// F's export, as `qd-export` wrote it before the change: `qd-release.v1`, no `prompt_format`.
#[test]
fn a_v4_release_is_refused_by_its_format() {
    let dir = release(|m| {
        m["format"] = json!("qd-release.v1");
        let identity = m["expected_identity"].as_object_mut().unwrap();
        identity.remove("prompt_format");
    });
    refused(&dir.0, "manifest", &["qd-release.v1", "qd-release.v2"]);
}

#[test]
fn a_v2_release_that_states_no_prompt_format_is_refused() {
    let dir = release(|m| {
        m["expected_identity"]
            .as_object_mut()
            .unwrap()
            .remove("prompt_format");
    });
    refused(&dir.0, "prompt_format", &["expected_identity.prompt_format"]);
}

/// A v4 checkpoint exported after the change is stamped 1 by `qd-export` (its recipe states no
/// format), so it is refused rather than served format-2 prompts it never saw.
#[test]
fn a_v2_release_of_another_prompt_format_is_refused_naming_both() {
    for other in [1u64, 3] {
        let dir = release(|m| m["expected_identity"]["prompt_format"] = json!(other));
        let err = refused(&dir.0, "prompt_format", &["prompt_format"]);
        assert!(
            err.detail.contains(&format!("prompt_format {other}")) && err.detail.contains('2'),
            "the refusal names the release's format and the runtime's: {err}"
        );
    }
}

#[test]
fn a_prompt_format_that_is_not_a_positive_integer_is_refused() {
    for bad in [json!("2"), json!(2.0), json!(true), json!(null), json!(-2), json!(0), json!([2])] {
        let dir = release(|m| m["expected_identity"]["prompt_format"] = bad.clone());
        refused(&dir.0, "prompt_format", &["expected_identity.prompt_format"]);
    }
}
