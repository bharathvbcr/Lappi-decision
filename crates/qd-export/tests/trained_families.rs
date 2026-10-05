//! `qd-export --train-manifest`: the release records the task families its weights were trained
//! on, read from the train split's manifest (`python/qd_data/manifest.py`), so the runtime can
//! refuse a task the release never trained (Fable's pipeline ruling, item 4).
//!
//! Driven through the binary, as the box runs it (`campaign/post-f-queue/box_q_j7p_cpu.sh`).

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use qd_runtime::calibration::CalibrationTable;
use qd_runtime::release::{MANIFEST_FILE, Release};
use qd_runtime::render::PROMPT_FORMAT;
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-export");

/// `qd_data.config.DEFAULT_HELD_OUT_FAMILIES`.
const HELD_OUT: [&str; 2] = ["code.language_id", "qa.answerability"];
const SNAPSHOT_HASH: &str = "5f1c0c6e0b8a4f0e9d7c3b2a1908f7e6d5c4b3a29180f7e6d5c4b3a291807f6e";

fn entry(i: usize, split: &str, family: &str) -> Value {
    json!({
        "row_id": format!("row-{i}"),
        "content_hash": format!("{i:064x}"),
        "split": split,
        "source_id": "test/source",
        "host": "test",
        "family_id": family,
        "repo_key": format!("repo-{i}"),
        "identity_key": format!("id-{i}"),
        "licence_id": "MIT",
        "obligations": [],
    })
}

/// A data manifest in `Manifest.to_json()`'s shape, one entry per `(split, family)`.
fn manifest_json(split: &str, rows: &[(&str, &str)]) -> Value {
    json!({
        "manifest_format_version": 1,
        "split": split,
        "data_snapshot_hash": SNAPSHOT_HASH,
        "n_rows": rows.len(),
        "held_out_families": HELD_OUT,
        "entries": rows
            .iter()
            .enumerate()
            .map(|(i, (s, f))| entry(i, s, f))
            .collect::<Vec<_>>(),
    })
}

fn write_json(path: &Path, value: &Value) -> PathBuf {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    path.to_path_buf()
}

fn table(dir: &Path) -> PathBuf {
    write_json(
        &dir.join("table.json"),
        &serde_json::to_value(CalibrationTable::reference()).unwrap(),
    )
}

/// Run the binary over the standard fixture, with `--train-manifest` when given.
fn run(fx: &common::Fixture, train_manifest: Option<&Path>) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.arg("--source")
        .arg(&fx.source)
        .arg("--base-snapshot")
        .arg(&fx.snapshot)
        .arg("--tokenizer-sha256")
        .arg(common::tokenizer_sha256(&fx.snapshot))
        .arg("--expect-vocab-size")
        .arg(common::VOCAB.to_string())
        .arg("--calibration")
        .arg(table(&fx.dir.0))
        .arg("--out")
        .arg(&fx.out);
    if let Some(path) = train_manifest {
        cmd.arg("--train-manifest").arg(path);
    }
    cmd.output().expect("the binary runs")
}

/// The standard fixture of a checkpoint trained on the prompt format this runtime serves, so the
/// release it exports is one `Release::open` accepts (`release_binding.rs` owns the refusal of
/// every other format).
fn served_format_fixture() -> common::Fixture {
    common::build(&common::standard_tensors(), &common::tiny_config(), |m| {
        m["prompt_format"] = json!(PROMPT_FORMAT);
    })
}

fn release_manifest(out: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(out.join(MANIFEST_FILE)).unwrap()).unwrap()
}

#[test]
fn the_release_records_the_families_its_train_manifest_holds() {
    let fx = served_format_fixture();
    let rows = [
        ("train", "code.defect_class"),
        ("train", "intent.in_scope"),
        ("train", "code.defect_class"),
        ("train", "code.change_scope"),
    ];
    let path = write_json(&fx.dir.0.join("train.json"), &manifest_json("train", &rows));
    let done = run(&fx, Some(&path));
    let stdout = String::from_utf8_lossy(&done.stdout);
    let stderr = String::from_utf8_lossy(&done.stderr);
    assert_eq!(done.status.code(), Some(0), "{stdout}{stderr}");

    let manifest = release_manifest(&fx.out);
    // Sorted and unique: the reader refuses anything else.
    assert_eq!(
        manifest["trained_families"],
        json!(["code.change_scope", "code.defect_class", "intent.in_scope"])
    );
    let block = &manifest["train_manifest"];
    let sha = qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(&path).unwrap()));
    assert_eq!(block["sha256"], json!(sha));
    assert_eq!(block["path"], json!(path.display().to_string()));
    assert_eq!(block["split"], json!("train"));
    assert_eq!(block["n_rows"], json!(4));
    assert_eq!(
        block["rows_by_family"],
        json!({"code.change_scope": 1, "code.defect_class": 2, "intent.in_scope": 1})
    );
    assert_eq!(block["data_snapshot_hash_as_recorded"], json!(SNAPSHOT_HASH));
    assert!(
        stdout.contains("trained_families: code.change_scope, code.defect_class, intent.in_scope"),
        "{stdout}"
    );
    Release::open(&fx.out).expect("the runtime's reader opens the release");
}

/// Write `doc` as JSON followed by JSON whitespace until the file is `len` bytes long: a train
/// manifest of that size which parses to `doc`.
fn write_padded_json(path: &Path, doc: &Value, len: u64) -> PathBuf {
    use std::io::Write;
    let body = serde_json::to_vec(doc).unwrap();
    let mut left = len.checked_sub(body.len() as u64).expect("len holds the document");
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&body).unwrap();
    let chunk = vec![b' '; 1 << 20];
    while left > 0 {
        let n = left.min(chunk.len() as u64);
        file.write_all(&chunk[..n as usize]).unwrap();
        left -= n;
    }
    file.sync_all().unwrap();
    assert_eq!(std::fs::metadata(path).unwrap().len(), len);
    path.to_path_buf()
}

/// v5's train manifest (`phase4-v5r-2026-10-03/data/pool/train.json`, 270,406,786 bytes) is
/// larger than the 256 MiB bound on the files the runtime reads. The train manifest is an
/// export-time input only, so it has a bound of its own and a manifest past the small-file
/// bound is read.
#[test]
fn a_train_manifest_past_the_small_file_bound_is_read() {
    let fx = served_format_fixture();
    let doc = manifest_json("train", &[("train", "code.defect_class")]);
    let len = qd_runtime::release::MAX_SMALL_FILE_BYTES + 1;
    let path = write_padded_json(&fx.dir.0.join("train.json"), &doc, len);
    let done = run(&fx, Some(&path));
    let stderr = String::from_utf8_lossy(&done.stderr);
    assert_eq!(done.status.code(), Some(0), "{stderr}");
    assert_eq!(release_manifest(&fx.out)["trained_families"], json!(["code.defect_class"]));
    std::fs::remove_file(&path).unwrap();
}

/// The train manifest's own bound still holds: a file past it is refused from its length alone,
/// before a byte is read (the file is sparse), and the refusal names that bound.
#[test]
fn a_train_manifest_past_its_own_bound_is_refused() {
    let fx = common::standard();
    let path = fx.dir.0.join("huge.json");
    let file = std::fs::File::create(&path).unwrap();
    file.set_len(qd_export::export::MAX_TRAIN_MANIFEST_BYTES + 1).unwrap();
    drop(file);
    let done = run(&fx, Some(&path));
    let stderr = String::from_utf8_lossy(&done.stderr);
    assert_eq!(done.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("qd-export: refused: TrainManifest"), "{stderr}");
    let bound = qd_export::export::MAX_TRAIN_MANIFEST_BYTES.to_string();
    assert!(stderr.contains(&bound), "{bound} not in {stderr}");
    assert!(!fx.out.exists(), "a refused export left {} behind", fx.out.display());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn an_export_that_cannot_say_what_was_trained_is_refused() {
    let fx = common::standard();
    let done = run(&fx, None);
    let stderr = String::from_utf8_lossy(&done.stderr);
    assert_eq!(done.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--train-manifest"), "{stderr}");
    assert!(!fx.out.exists(), "a refused export left {} behind", fx.out.display());
}

#[test]
fn a_train_manifest_that_does_not_describe_trained_rows_is_refused() {
    let heldout_rows = [("heldout", "code.language_id")];
    let mut versioned = manifest_json("train", &[("train", "code.defect_class")]);
    versioned["manifest_format_version"] = json!(2);
    let mut unhashed = manifest_json("train", &[("train", "code.defect_class")]);
    unhashed["data_snapshot_hash"] = json!("not a hash");
    let cases: Vec<(&str, Value, &str)> = vec![
        ("heldout", manifest_json("heldout", &heldout_rows), "split"),
        ("val", manifest_json("val", &[("val", "code.defect_class")]), "split"),
        (
            "mixed-split",
            manifest_json("train", &[("train", "code.defect_class"), ("val", "intent.in_scope")]),
            "row-1",
        ),
        (
            "held-out-family",
            manifest_json("train", &[("train", "code.defect_class"), ("train", "qa.answerability")]),
            "qa.answerability",
        ),
        ("no-rows", manifest_json("train", &[]), "no rows"),
        ("blank-family", manifest_json("train", &[("train", "  ")]), "family_id"),
        ("version", versioned, "manifest_format_version"),
        ("unhashed", unhashed, "data_snapshot_hash"),
        ("not-an-object", json!(["train"]), "object"),
    ];
    for (tag, doc, needle) in cases {
        let fx = common::standard();
        let path = write_json(&fx.dir.0.join(format!("{tag}.json")), &doc);
        let done = run(&fx, Some(&path));
        let stderr = String::from_utf8_lossy(&done.stderr);
        assert_eq!(done.status.code(), Some(2), "{tag}: {stderr}");
        assert!(stderr.contains("qd-export: refused: TrainManifest"), "{tag}: {stderr}");
        assert!(stderr.contains(needle), "{tag}: {needle:?} not in {stderr}");
        assert!(!fx.out.exists(), "{tag}: a refused export left {} behind", fx.out.display());
    }
}
