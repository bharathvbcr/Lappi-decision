//! `qd_prep::convert_licence` against its oracle, `python/qd_data/licences.py`: the same table
//! (ids and tiers), the same aliases, and the same `classify` answer for every licence string
//! the conversion meets -- SWE-rebench's `license_name` values, GitHub's licence keys from the
//! CodeSearchNet lookup, the dataset-level ids and adversarial spellings.
//!
//! It also holds [`convert_licence::PENDING`] honest: every pending id must be one Python does
//! NOT admit. The day `qd_data.licences` registers one, this fails, and the id moves from the
//! pending list to the mirrored table.
//!
//! The interpreter: `QD_PYTHON`, else `<repo>/.venv/bin/python`, else `python3`; it needs only
//! the standard library and `python/qd_data`. With none that imports `qd_data.licences` the test
//! **fails** rather than skipping: a skip is recorded as `ok` (the pattern
//! `crates/qd-metal/tests/ledger_rows.rs` retired).

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use qd_prep::convert_licence::{self, Tier};
use serde_json::{Value, json};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

const PYTHON_TIMEOUT: Duration = Duration::from_secs(60);

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

fn output_bounded(cmd: &mut Command, limit: Duration) -> io::Result<Output> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let start = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            return child.wait_with_output();
        }
        if start.elapsed() > limit {
            child.kill()?;
            child.wait()?;
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "interpreter timed out",
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn python_command(py: &Path) -> Command {
    let mut cmd = Command::new(py);
    cmd.env("PYTHONPATH", repo().join("python"));
    cmd
}

fn python() -> PathBuf {
    let candidates = python_candidates();
    for p in &candidates {
        let mut cmd = python_command(p);
        cmd.arg("-c").arg("import qd_data.licences");
        if matches!(output_bounded(&mut cmd, PYTHON_TIMEOUT), Ok(o) if o.status.success()) {
            return p.clone();
        }
    }
    panic!(
        "no interpreter among {candidates:?} can import qd_data.licences, so the oracle did NOT \
         run. This fails rather than skipping. Remedy: point QD_PYTHON at a Python >= 3.11."
    )
}

const ORACLE: &str = "import json, sys\n\
from qd_data.licences import LICENCE_POLICY, _ALIASES, classify\n\
raws = json.load(open(sys.argv[1]))\n\
import qd_data.licences as mod\n\
print(json.dumps({\n\
  'read': raws,\n\
  'module': mod.__file__,\n\
  'policy': {k: p.tier.value for k, p in LICENCE_POLICY.items()},\n\
  'aliases': dict(_ALIASES),\n\
  'classified': [[classify(r).licence_id, classify(r).tier.value] for r in raws],\n\
}))\n";

/// Every licence string the conversion meets, plus spellings that test the normaliser.
const OBSERVED: [&str; 74] = [
    // nebius/SWE-rebench `license_name` values (filtered and test files, 2026-10-06).
    "MIT License",
    "Apache License 2.0",
    "BSD 3-Clause \"New\" or \"Revised\" License",
    "BSD 2-Clause \"Simplified\" License",
    "BSD License",
    "BSD",
    "Zope Public License 2.1",
    "BSD-3-Clause",
    "BSD 3-Clause",
    "New BSD License",
    "The Unlicense",
    "BSD 3-Clause License",
    "ISC License",
    "Apache License 2.0 or MIT License",
    "MIT/Apache-2.0 Dual License",
    "BSD-2-Clause-Patent",
    "3-Clause BSD license",
    "Creative Commons Zero v1.0 Universal",
    "MIT/X Consortium license",
    "BSD-3",
    "Apache 2.0 or BSD3",
    "BSD 4-Clause \"Original\" or \"Old\" License",
    "3-clause BSD License",
    "BSD 2-clause license",
    "BSD-2-Clause",
    "Modified BSD License",
    "MIT",
    "CC0 1.0 Universal",
    "BSD-2 License",
    "BSD License (3 Clause)",
    "PostgreSQL License",
    "Apache License 2.0 and MIT License",
    "MIT No Attribution",
    "Mozilla Public License 2.0",
    "MIT license",
    "Expat License",
    "Academic Free License 3.0",
    "BSD-3 Clause",
    "Revised BSD License",
    "MIT-CMU License",
    // GitHub licence keys from the CodeSearchNet lookup (csn-licences-2026-10-06.record.json).
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "gpl-3.0",
    "bsd-2-clause",
    "gpl-2.0",
    "lgpl-3.0",
    "other",
    "isc",
    "mpl-2.0",
    "agpl-3.0",
    "lgpl-2.1",
    "wtfpl",
    "cc0-1.0",
    "epl-1.0",
    "zlib",
    "artistic-2.0",
    "bsd-3-clause-clear",
    "bsl-1.0",
    "unlicense",
    "eupl-1.1",
    "0bsd",
    "epl-2.0",
    "afl-3.0",
    "cc-by-sa-4.0",
    // Dataset-level ids and adversarial spellings.
    "cc-by-4.0",
    "  Apache   2.0 ",
    "CC-BY-NC-SA-3.0",
    "public domain",
    "nc",
    "unknown",
    "",
    "Public\tDomain",
    "APACHE-2",
];

/// The input file for one oracle call. Unique per call: the tests run on parallel threads of one
/// process, and a shared path let one test's Python read the other's list (2026-10-06, 17:48Z:
/// the pending test read "MIT License" first and reported `odc-by-1.0` as "allow").
fn input_path(tag: &str) -> PathBuf {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("qd-licence-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("raws-{tag}-{n}.json"))
}

fn oracle(tag: &str, raws: &[String]) -> Value {
    let input = input_path(tag);
    std::fs::write(&input, serde_json::to_vec(&json!(raws)).unwrap()).unwrap();
    let py = python();
    let mut cmd = python_command(&py);
    cmd.arg("-c").arg(ORACLE).arg(&input);
    let out = output_bounded(&mut cmd, PYTHON_TIMEOUT)
        .unwrap_or_else(|e| panic!("could not run {}: {e}", py.display()));
    assert!(
        out.status.success(),
        "the oracle failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).expect("the oracle printed JSON");
    // Python's answers are about the list it read; it must be the list this call sent.
    assert_eq!(
        v["read"],
        json!(raws),
        "the oracle read another call's input"
    );
    // The oracle is the qd_data of the tree under test (merged main, at merge).
    let module = v["module"].as_str().expect("the module path");
    let want = repo().join("python/qd_data/licences.py");
    assert_eq!(
        std::fs::canonicalize(module).unwrap(),
        std::fs::canonicalize(&want).unwrap(),
        "the oracle imported {module}, not this tree's {}",
        want.display()
    );
    v
}

#[test]
fn concurrent_oracle_calls_never_share_an_input() {
    let a = input_path("x");
    let b = input_path("x");
    assert_ne!(a, b);
    let raws = |s: &str| vec![s.to_owned()];
    std::thread::scope(|s| {
        let one = s.spawn(|| oracle("one", &raws("MIT License")));
        let two = s.spawn(|| oracle("two", &raws("odc-by-1.0")));
        assert_eq!(one.join().unwrap()["classified"][0][1], "allow");
        assert_eq!(two.join().unwrap()["read"], json!(["odc-by-1.0"]));
    });
}

#[test]
fn the_table_aliases_and_every_classification_match_python() {
    let mut raws: Vec<String> = OBSERVED.iter().map(|s| (*s).to_owned()).collect();
    raws.extend(
        convert_licence::POLICY
            .iter()
            .map(|(id, _)| (*id).to_owned()),
    );
    raws.extend(
        convert_licence::ALIASES
            .iter()
            .map(|(k, _)| (*k).to_owned()),
    );
    raws.extend(
        convert_licence::PENDING
            .iter()
            .map(|(id, _)| (*id).to_owned()),
    );
    let py = oracle("table", &raws);

    let rust_policy: serde_json::Map<String, Value> = convert_licence::POLICY
        .iter()
        .map(|(id, t)| ((*id).to_owned(), json!(t.as_str())))
        .collect();
    assert_eq!(
        py["policy"],
        Value::Object(rust_policy),
        "LICENCE_POLICY differs from the mirror"
    );
    let rust_aliases: serde_json::Map<String, Value> = convert_licence::ALIASES
        .iter()
        .map(|(k, v)| ((*k).to_owned(), json!(v)))
        .collect();
    assert_eq!(
        py["aliases"],
        Value::Object(rust_aliases),
        "_ALIASES differs from the mirror"
    );

    let classified = py["classified"].as_array().unwrap();
    assert_eq!(classified.len(), raws.len());
    let mut diffs = Vec::new();
    for (raw, want) in raws.iter().zip(classified) {
        let (id, tier) = convert_licence::classify(raw);
        let got = json!([id, tier.as_str()]);
        if &got != want {
            diffs.push(format!("{raw:?}: rust {got}, python {want}"));
        }
    }
    assert!(diffs.is_empty(), "classify differs:\n{}", diffs.join("\n"));
}

#[test]
fn every_pending_id_is_one_python_does_not_admit() {
    let raws: Vec<String> = convert_licence::PENDING
        .iter()
        .map(|(id, _)| (*id).to_owned())
        .collect();
    let py = oracle("pending", &raws);
    for (raw, c) in raws.iter().zip(py["classified"].as_array().unwrap()) {
        assert_eq!(
            c[1],
            json!(Tier::NeedsHumanCall.as_str()),
            "{raw} is now {} in qd_data.licences: move it from convert_licence::PENDING to the \
             mirrored table",
            c[1]
        );
    }
}
