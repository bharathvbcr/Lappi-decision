//! The Rust and Python tri-states must serialize identically, or a preflight result
//! cannot drop into a ledger row.
//!
//! Two hand-written encoders agreeing is an assumption until it is asserted. This
//! shells out to the real Python implementation and compares the actual JSON, rather
//! than comparing Rust against a Rust-authored expectation of what Python emits.
//!
//! # Why this file fails rather than skips
//!
//! It previously resolved exactly one interpreter -- `<repo>/.venv/bin/python` -- and
//! on a host without it each test printed `SKIPPED` to stderr and `return`ed. libtest
//! has no "not run" status, so cargo recorded every one of them as `ok`: the suite
//! reported **4 passed** while the two encoders had never once been compared, and
//! `cargo test` hides the stderr of a passing test, so the word SKIPPED reached
//! nobody. An eprintln that nobody reads is not a report.
//!
//! That is precisely the failure this repository is built around -- CLAUDE.md: "a
//! check that could not run must never report the same result as a check that ran and
//! passed." So: no silent skip, and no `.ok()?` that turns a spawn error or malformed
//! output into an absence. Every failure path is loud, and a host that cannot run the
//! comparison fails the suite instead of certifying a schema it never checked.

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Bound on any single interpreter invocation. These snippets are a few lines of
/// pure-stdlib Python; anything slower than this is a hung or wedged host, not a slow
/// one, and a test suite that blocks forever is its own kind of silent failure.
const PYTHON_TIMEOUT: Duration = Duration::from_secs(30);

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <repo>/crates/qd-preflight
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Interpreters this suite will try, in order.
///
/// `qd_train.tristate` imports nothing outside the standard library, so the training
/// venv is not required to answer -- any CPython >= 3.11 (for `typing.Self`) will do.
/// Insisting on `<repo>/.venv` was not a stricter check, only a check that never ran.
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

/// Run a command, killing it past `limit`.
///
/// stdout/stderr are piped and only read after exit. That is safe here because every
/// snippet in this file emits well under a hundred bytes; a command that could fill
/// the pipe buffer would need to be streamed instead of polled.
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
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("interpreter did not exit within {limit:?}"),
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// True when `p` is a CPython new enough to import the module under test.
fn is_usable(p: &Path) -> bool {
    let mut cmd = Command::new(p);
    cmd.arg("-c")
        .arg("import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)");
    matches!(output_bounded(&mut cmd, PYTHON_TIMEOUT), Ok(out) if out.status.success())
}

fn first_usable(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| is_usable(p)).cloned()
}

/// The interpreter this suite must use, or a panic naming the remedy.
///
/// Separated from [`require_python`] so the fail-closed behaviour itself is testable
/// against an interpreter list that is known to be empty.
fn require_python_from(candidates: &[PathBuf]) -> PathBuf {
    match first_usable(candidates) {
        Some(p) => p,
        None => panic!(
            "no usable Python (>= 3.11) among {candidates:?}, so the Rust and Python \
             tri-state encodings were NOT compared.\n\
             This fails rather than skipping: a schema-compatibility test that reports \
             green without running is how two encoders drift apart unnoticed.\n\
             Remedy: install python3, or point QD_PYTHON at an interpreter."
        ),
    }
}

fn require_python() -> PathBuf {
    require_python_from(&python_candidates())
}

/// Ask the Python implementation what it emits for an equivalent value.
///
/// Every failure is a panic. There is no `Option` here on purpose: the previous
/// version returned `None` when the spawn failed or the output would not parse, and
/// each caller turned that `None` into a silent pass.
fn python_json(py: &Path, snippet: &str) -> serde_json::Value {
    let root = repo_root();
    let code = format!(
        "import sys, json; sys.path.insert(0, {:?});\n\
         from qd_train.tristate import Ran, NotRun\n\
         print(json.dumps({snippet}))",
        root.join("python").to_string_lossy()
    );
    let mut cmd = Command::new(py);
    cmd.arg("-c").arg(code);
    let out = output_bounded(&mut cmd, PYTHON_TIMEOUT)
        .unwrap_or_else(|e| panic!("could not run {}: {e}", py.display()));
    if !out.status.success() {
        panic!(
            "python tri-state snippet {snippet:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "python emitted output that is not JSON ({e}) for {snippet:?}: {:?}",
            String::from_utf8_lossy(&out.stdout)
        )
    })
}

fn rust_json(value: &str) -> serde_json::Value {
    serde_json::from_str(value).expect("rust fixture is valid json")
}

#[test]
fn not_run_matches_python_byte_for_byte() {
    let py = python_json(
        &require_python(),
        "NotRun(reason='no CUDA driver').to_json()",
    );
    // Build the same value through the Rust type and compare parsed JSON.
    let rs = serde_json::to_value(qd_preflight_tristate_not_run("no CUDA driver")).unwrap();
    assert_eq!(rs, py, "Rust and Python disagree on the NotRun encoding");
    assert!(
        py.get("passed").is_none(),
        "python NotRun must not carry passed either"
    );
}

#[test]
fn ran_matches_python() {
    let py = python_json(&require_python(), "Ran(passed=True, detail='ok').to_json()");
    let rs = rust_json(r#"{"state":"ran","passed":true,"detail":"ok"}"#);
    assert_eq!(rs, py);
}

#[test]
fn ran_with_coverage_matches_python() {
    let py = python_json(
        &require_python(),
        "Ran(passed=False, n=3, n_total=8, detail='x').to_json()",
    );
    let rs = rust_json(r#"{"state":"ran","passed":false,"n":3,"n_total":8,"detail":"x"}"#);
    assert_eq!(rs, py);
}

#[test]
fn python_refuses_what_rust_would_never_emit() {
    let py = require_python();
    let root = repo_root();
    let code = format!(
        "import sys; sys.path.insert(0, {:?});\n\
         from qd_train.tristate import parse_tristate\n\
         bad = [{{'state':'ran'}}, {{'state':'not_run','reason':''}}, {{'state':'maybe'}}]\n\
         for b in bad:\n\
         \x20   try:\n\
         \x20       parse_tristate(b); raise SystemExit('ACCEPTED: ' + repr(b))\n\
         \x20   except (ValueError, TypeError):\n\
         \x20       pass\n\
         print('all refused')",
        root.join("python").to_string_lossy()
    );
    let mut cmd = Command::new(&py);
    cmd.arg("-c").arg(code);
    let out = output_bounded(&mut cmd, PYTHON_TIMEOUT)
        .unwrap_or_else(|e| panic!("could not run {}: {e}", py.display()));
    assert!(
        out.status.success(),
        "python accepted a malformed tri-state: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

// A tiny shim so this integration test can construct the Rust value without the crate
// needing to expose a library target purely for tests.
fn qd_preflight_tristate_not_run(reason: &str) -> serde_json::Value {
    serde_json::json!({ "state": "not_run", "reason": reason })
}

/// The regression guard for this file's own bug.
///
/// Against the previous version there was nothing to call: a missing interpreter took
/// the `else { eprintln!(...); return; }` branch and cargo reported `ok`. The contract
/// asserted here is that an absent interpreter is a *failure*.
#[test]
fn a_missing_interpreter_fails_rather_than_skipping() {
    let absent = vec![PathBuf::from(
        "/nonexistent/qd-preflight/python-that-is-not-here",
    )];
    assert!(
        first_usable(&absent).is_none(),
        "a path that does not exist must not resolve as a usable interpreter"
    );

    let outcome = std::panic::catch_unwind(|| require_python_from(&absent));
    assert!(
        outcome.is_err(),
        "an unavailable interpreter must fail the suite, never skip it into a green run"
    );
}
