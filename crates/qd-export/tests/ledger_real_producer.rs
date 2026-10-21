//! `qd-export --ft-ledger` binds against a ledger row written by the REAL producer, the Python
//! `RunRecorder` with `train_manifest_sha256` set (`python/qd_train/ledger.py`), not a row this
//! suite wrote by hand. `train_manifest_binding.rs` pins the refusals with hand-built rows; this
//! file proves the field name and shape the exporter reads are the ones the writer emits.
//!
//! The recorder is run through the repository's Python: `QD_PYTHON`, else this checkout's
//! `.venv`, else the main checkout's (the linked-worktree case). With none the tests fail loudly:
//! a binding that could not be checked against the real writer is not a passing one.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use qd_export::export;
use serde_json::{Value, json};

const WRITE_ROW: &str = r#"
import sys
repo, ledger, sha = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo + "/python")
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder
proto = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                 backbone_commit="b" * 40, recipe_hash="r" * 64, seed=0)
rec = RunRecorder(Ledger(ledger), protocol=proto, run_kind="ft", repo=repo,
                  env=Environment.detect(device="cpu"), wall_clock_s=1.0, cost=None,
                  train_manifest_sha256=(sha or None))
with rec:
    pass
print(rec.row.row_id)
"#;

// BEGIN qd-python-resolver
// The same text sits in qd-prep/tests/convert_licence_parity.rs, qd-metal/tests/ledger_rows.rs,
// qd-preflight/tests/schema_compat.rs and qd-export/tests/ledger_real_producer.rs;
// python/tests/test_qd_python_resolver_pinned.py fails if any differ, or if a crate source that
// reads QD_PYTHON lacks it. There is no shared test-support crate to own it, and a new workspace
// member was not authorised, so the text is pinned instead.

/// The interpreters to try, in order: `QD_PYTHON`; this checkout's `.venv/bin/python`; the main
/// checkout's `.venv/bin/python` when this is a linked worktree (a worktree has no `.venv` of its
/// own); then `python3`. The caller fails loudly, naming `QD_PYTHON`, when none is usable.
fn python_candidates() -> Vec<PathBuf> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    resolve_python_candidates(std::env::var_os("QD_PYTHON"), &repo)
}

fn resolve_python_candidates(qd_python: Option<std::ffi::OsString>, repo: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = qd_python {
        out.push(PathBuf::from(p));
    }
    out.push(repo.join(".venv/bin/python"));
    if let Some(main) = main_checkout(repo) {
        out.push(main.join(".venv/bin/python"));
    }
    out.push(PathBuf::from("python3"));
    out
}

/// The main checkout's root, from `git rev-parse --git-common-dir`. `None` when `repo` is not in
/// a git checkout, git cannot run, or the common dir is not named `.git` (a bare repository or a
/// `--separate-git-dir` checkout has no main working tree to find); the candidate list then
/// simply lacks that entry. `GIT_DIR` and `GIT_WORK_TREE` are removed from the call so an
/// inherited value cannot point it at another repository.
fn main_checkout(repo: &Path) -> Option<PathBuf> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let common = PathBuf::from(text.strip_suffix('\n').unwrap_or(&text));
    if common.file_name()? != ".git" {
        return None;
    }
    common.parent().map(Path::to_path_buf)
}

#[test]
fn a_linked_worktree_resolves_the_main_checkouts_venv() {
    let root = std::env::temp_dir().join(format!("qd-python-resolver-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let main = root.join("main");
    let linked = root.join("linked");
    std::fs::create_dir_all(main.join(".venv/bin")).unwrap();
    std::fs::write(main.join(".venv/bin/python"), "").unwrap();
    let git = |dir: &Path, args: &[&str]| {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&main, &["init", "-q"]);
    git(&main, &["commit", "-q", "--allow-empty", "-m", "x"]);
    git(&main, &["worktree", "add", "-q", linked.to_str().unwrap()]);
    let main_py = main.join(".venv/bin/python").canonicalize().unwrap();

    let got = resolve_python_candidates(None, &linked);
    assert!(
        got.iter()
            .any(|p| p.canonicalize().ok().as_deref() == Some(main_py.as_path())),
        "a linked worktree without a .venv must try the main checkout's: {got:?}"
    );
    let order: Vec<_> = got
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    assert!(
        order[0].ends_with("linked/.venv/bin/python"),
        "own .venv first: {order:?}"
    );
    assert_eq!(order.last().map(String::as_str), Some("python3"));

    let with_env = resolve_python_candidates(Some("/x/py".into()), &linked);
    assert_eq!(
        with_env[0],
        PathBuf::from("/x/py"),
        "QD_PYTHON outranks every checkout"
    );

    // A checkout whose git dir is elsewhere has no `.git` common dir to take the main tree from.
    let separate = root.join("separate");
    git(
        &root,
        &[
            "init",
            "-q",
            "--separate-git-dir",
            root.join("gitstore").to_str().unwrap(),
            separate.to_str().unwrap(),
        ],
    );
    assert_eq!(main_checkout(&separate), None);
    std::fs::remove_dir_all(&root).unwrap();
}
// END qd-python-resolver

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// The first interpreter that can import the real recorder (`qd_train.ledger`, standard library
/// only), or a panic naming `QD_PYTHON`: a binding that could not be checked against the real
/// writer is not a passing one, and this never skips.
fn python() -> PathBuf {
    let candidates = python_candidates();
    for p in &candidates {
        let ok = Command::new(p)
            .env("PYTHONPATH", repo().join("python"))
            .arg("-c")
            .arg("import qd_train.ledger")
            .stdin(Stdio::null())
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return p.clone();
        }
    }
    panic!(
        "no interpreter among {candidates:?} can import qd_train.ledger, so the real writer was \
         NOT run. This fails rather than skipping. Remedy: point QD_PYTHON at a Python >= 3.11."
    )
}

/// Run the real recorder: append a completed ft row recording `sha` (none when empty) to `ledger`
/// and return its row id.
fn write_row(ledger: &Path, sha: &str) -> String {
    let out = Command::new(python())
        .arg("-c")
        .arg(WRITE_ROW)
        .arg(repo())
        .arg(ledger)
        .arg(sha)
        .output()
        .expect("python runs");
    assert!(
        out.status.success(),
        "the recorder failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn train_manifest(dir: &Path) -> (PathBuf, String) {
    let path = common::write_train_manifest(&dir.join("train.json"), &common::TRAINED_FAMILIES);
    let sha = qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(&path).unwrap()));
    (path, sha)
}

fn request_for(
    row_id: &str,
    train: &Path,
    ledger: &Path,
) -> (common::Fixture, qd_export::ExportRequest) {
    let fx = common::build(&common::standard_tensors(), &common::tiny_config(), |m| {
        m["ft_row_ids"] = json!([row_id])
    });
    let mut req = fx.request();
    req.train_manifest = train.to_path_buf();
    req.ft_ledger = Some(ledger.to_path_buf());
    (fx, req)
}

#[test]
fn a_row_the_real_recorder_wrote_binds_the_train_manifest() {
    let dir = common::TempDir::new("real-producer");
    let (train, sha) = train_manifest(&dir.0);
    let ledger = dir.0.join("ledger.jsonl");
    let row_id = write_row(&ledger, &sha);
    let on_disk: Value = serde_json::from_str(
        std::fs::read_to_string(&ledger)
            .unwrap()
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        on_disk["train_manifest_sha256"],
        json!(sha),
        "the writer's own field name and value"
    );

    let (fx, req) = request_for(&row_id, &train, &ledger);
    export(&req).expect("the exporter binds against the producer's row");
    let m: Value =
        serde_json::from_slice(&std::fs::read(fx.out.join("release_manifest.json")).unwrap())
            .unwrap();
    let b = &m["train_manifest"]["ledger_binding"];
    assert_eq!(b["verified"], json!(true));
    assert_eq!(b["train_manifest_sha256"], json!(sha));
    assert_eq!(b["ft_row_ids"], json!([row_id]));
}

#[test]
fn a_real_row_that_recorded_another_manifest_is_refused() {
    let dir = common::TempDir::new("real-producer-other");
    let (train, _) = train_manifest(&dir.0);
    let ledger = dir.0.join("ledger.jsonl");
    let row_id = write_row(&ledger, &"0".repeat(64));
    let (fx, req) = request_for(&row_id, &train, &ledger);
    let err = export(&req).expect_err("another manifest's sha");
    assert!(
        err.detail.contains(&row_id) && err.detail.contains("train_manifest_sha256"),
        "{}",
        err.detail
    );
    assert!(!fx.out.exists());
}

#[test]
fn a_real_row_that_recorded_no_manifest_is_refused_as_not_recorded() {
    let dir = common::TempDir::new("real-producer-none");
    let (train, _) = train_manifest(&dir.0);
    let ledger = dir.0.join("ledger.jsonl");
    let row_id = write_row(&ledger, "");
    let (fx, req) = request_for(&row_id, &train, &ledger);
    let err = export(&req).expect_err("no recorded sha");
    assert!(
        err.detail.contains("no train_manifest_sha256"),
        "{}",
        err.detail
    );
    assert!(!fx.out.exists());
}
