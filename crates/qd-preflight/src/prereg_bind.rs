//! `qd-prereg-bind`: bind a run to the pre-registration it claims.
//!
//! A pre-registration pins what the run will read (`data.train_shard.shard_hash`) and is only
//! worth anything if it was committed, unmodified, before the run. This gate refuses, with the
//! repo's tri-state (0 PASS, 1 FAILED, 3 NOT_RUN), when
//!
//! 1. the shard set the run will read does not carry the hash the registration pins;
//! 2. the registration file, or any decision record it cites, is untracked or differs from
//!    `HEAD` in the working tree;
//! 3. the commit that last touched the registration is not an ancestor of the run's git `HEAD`.
//!
//! **What the shard check compares.** The registration's `shard_hash` is the `shard_hash` that
//! `ShardHeader.shard_hash()` (Python) / `ShardHeader::shard_hash` (`qd-train/src/shards.rs`)
//! derives and writes into the set's `header.json`; the reader re-derives it on load and refuses
//! a header whose recorded hash differs. This gate compares the RECORDED field only: it does not
//! re-derive, because the derivation lives in `qd-train` (a second copy here would drift) and
//! `qd-preflight` does not depend on it. A `header.json` edited in place with its hash left
//! unchanged therefore passes this check and is refused by the shard reader at load.
//!
//! A check that could not run (no header, no git, a cited record that does not exist) is
//! `NOT_RUN`, never a pass.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

#[path = "tristate.rs"]
pub mod tristate;

use tristate::{aggregate, TriState};

/// Directories whose files a registration cites as decision records.
const CITED_ROOTS: [&str; 5] = ["AUDIT/", "HANDOFF/", "campaign/", "docs/", "ledger/"];
const CITED_SUFFIXES: [&str; 3] = [".md", ".jsonl", ".json"];

/// What the run is about to use.
pub struct BindInputs {
    /// The registration JSON, inside `repo`.
    pub prereg: PathBuf,
    /// The git checkout the registration and its decision records live in.
    pub repo: PathBuf,
    /// The train shard set directory (holds `header.json`).
    pub shard_dir: PathBuf,
    /// The git commit the run will execute at; `None` is the checkout's `HEAD`.
    pub head: Option<String>,
}

fn is_path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '/')
}

/// The repo-relative decision-record paths a registration's text cites, sorted and unique.
/// A token containing a glob (`ledger/*.jsonl`) names no file and is skipped.
pub fn cited_paths(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < chars.len() {
        let boundary = i == 0 || !is_path_char(chars[i - 1]);
        if boundary {
            for root in CITED_ROOTS {
                let r: Vec<char> = root.chars().collect();
                if chars[i..].starts_with(&r) {
                    let mut j = i;
                    while j < chars.len() && is_path_char(chars[j]) {
                        j += 1;
                    }
                    let globbed = chars.get(j) == Some(&'*');
                    let token: String = chars[i..j].iter().collect();
                    let token = token.trim_end_matches('.').to_owned();
                    if !globbed && CITED_SUFFIXES.iter().any(|s| token.ends_with(s)) {
                        found.insert(token);
                    }
                    i = j.max(i + 1) - 1;
                    break;
                }
            }
        }
        i += 1;
    }
    found.into_iter().collect()
}

fn git(repo: &Path, args: &[&str]) -> Result<(i32, String), String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    let code = out.status.code().ok_or_else(|| format!("git {} was killed", args.join(" ")))?;
    Ok((code, String::from_utf8_lossy(&out.stdout).trim().to_owned()))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: not JSON: {e}", path.display()))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Check 1: the shard set's recorded `shard_hash` is the one the registration pins.
pub fn check_shard_hash(prereg: &Value, shard_dir: &Path) -> TriState {
    let Some(pinned) = prereg.pointer("/data/train_shard/shard_hash").and_then(Value::as_str)
    else {
        return TriState::not_run(
            "the registration has no data.train_shard.shard_hash, so there is nothing to bind to",
        );
    };
    if !is_sha256(pinned) {
        return TriState::not_run(format!("the registration's shard_hash {pinned:?} is not a sha256"));
    }
    let header = match read_json(&shard_dir.join("header.json")) {
        Ok(v) => v,
        Err(e) => return TriState::not_run(format!("the shard set's header could not be read: {e}")),
    };
    let Some(actual) = header.get("shard_hash").and_then(Value::as_str) else {
        return TriState::not_run("the shard set's header.json records no shard_hash");
    };
    if actual == pinned {
        TriState::pass(format!("shard_hash {pinned} (recorded header hash; the shard reader re-derives it)"))
    } else {
        TriState::fail(format!(
            "the shard set at {} carries shard_hash {actual}, but the registration pins {pinned}; \
             a run on it is not a run of this registration",
            shard_dir.display()
        ))
    }
}

/// Check 2: every path is tracked and identical to `HEAD` in the working tree and the index.
pub fn check_committed_clean(repo: &Path, label: &str, paths: &[String]) -> TriState {
    if paths.is_empty() {
        return TriState::not_run(format!("{label}: no paths to check"));
    }
    let mut problems = Vec::new();
    let mut missing = Vec::new();
    for p in paths {
        if !repo.join(p).is_file() {
            missing.push(p.clone());
            continue;
        }
        let tracked = match git(repo, &["ls-files", "--error-unmatch", "--", p]) {
            Ok((0, _)) => true,
            Ok(_) => false,
            Err(e) => return TriState::not_run(format!("{label}: {e}")),
        };
        if !tracked {
            problems.push(format!("{p} is untracked (never committed)"));
            continue;
        }
        match git(repo, &["status", "--porcelain=v1", "--", p]) {
            Ok((0, out)) if out.is_empty() => {}
            Ok((0, out)) => problems.push(format!("{p} differs from HEAD ({})", out.trim())),
            Ok((code, _)) => return TriState::not_run(format!("{label}: git status exited {code}")),
            Err(e) => return TriState::not_run(format!("{label}: {e}")),
        }
    }
    if !problems.is_empty() {
        return TriState::fail(format!("{label}: {}", problems.join("; ")));
    }
    if !missing.is_empty() {
        return TriState::not_run(format!(
            "{label}: cited but not present in the checkout, so not verifiable: {}",
            missing.join(", ")
        ));
    }
    TriState::pass(format!("{label}: {} file(s) committed and unmodified", paths.len()))
}

/// Check 3: the commit that last touched the registration is an ancestor of the run's `HEAD`.
pub fn check_ancestor(repo: &Path, prereg_rel: &str, head: Option<&str>) -> TriState {
    let commit = match git(repo, &["log", "-1", "--format=%H", "HEAD", "--", prereg_rel]) {
        Ok((0, out)) if !out.is_empty() => out,
        Ok((0, _)) => {
            return TriState::fail(format!("{prereg_rel} has no commit on HEAD: it was never committed"));
        }
        Ok((code, _)) => return TriState::not_run(format!("git log exited {code}")),
        Err(e) => return TriState::not_run(e),
    };
    let target = head.unwrap_or("HEAD");
    let resolved = match git(repo, &["rev-parse", "--verify", &format!("{target}^{{commit}}")]) {
        Ok((0, out)) => out,
        Ok(_) => return TriState::not_run(format!("the run's head {target:?} is not a commit here")),
        Err(e) => return TriState::not_run(e),
    };
    match git(repo, &["merge-base", "--is-ancestor", &commit, &resolved]) {
        Ok((0, _)) => TriState::pass(format!("registration commit {commit} is an ancestor of {resolved}")),
        Ok((1, _)) => TriState::fail(format!(
            "registration commit {commit} is not an ancestor of the run's head {resolved}: the \
             run is not built on the registration"
        )),
        Ok((code, _)) => TriState::not_run(format!("git merge-base exited {code}")),
        Err(e) => TriState::not_run(e),
    }
}

/// Every check, and the aggregate. Always three named checks, so an aggregate is never empty.
pub fn bind(inputs: &BindInputs) -> (Vec<(String, TriState)>, TriState) {
    let rel = inputs
        .prereg
        .strip_prefix(&inputs.repo)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| inputs.prereg.clone());
    let rel_s = rel.to_string_lossy().into_owned();
    let (value, text) = match fs::read_to_string(&inputs.prereg) {
        Ok(t) => (serde_json::from_str::<Value>(&t).map_err(|e| format!("not JSON: {e}")), Some(t)),
        Err(e) => (Err(format!("{}: {e}", inputs.prereg.display())), None),
    };
    let unreadable = |what: &str| {
        TriState::not_run(format!(
            "{what}: the registration could not be read ({})",
            value.as_ref().err().cloned().unwrap_or_default()
        ))
    };
    let shard = match &value {
        Ok(v) => check_shard_hash(v, &inputs.shard_dir),
        Err(_) => unreadable("shard_hash"),
    };
    let mut files = vec![rel_s.clone()];
    if let Some(t) = &text {
        files.extend(cited_paths(t).into_iter().filter(|p| *p != rel_s));
    }
    let clean = check_committed_clean(&inputs.repo, "registration and cited decision records", &files);
    let ancestor = check_ancestor(&inputs.repo, &rel_s, inputs.head.as_deref());
    let checks = vec![
        ("shard_hash".to_owned(), shard),
        ("committed_clean".to_owned(), clean),
        ("ancestor".to_owned(), ancestor),
    ];
    let parts: Vec<(&str, TriState)> = checks.iter().map(|(n, t)| (n.as_str(), t.clone())).collect();
    let overall = aggregate("qd-prereg-bind", &parts);
    (checks, overall)
}

/// 0 PASS, 1 FAILED, 3 NOT_RUN.
pub fn exit_code(t: &TriState) -> u8 {
    match t {
        TriState::Ran { passed: true, .. } => 0,
        TriState::Ran { passed: false, .. } => 1,
        TriState::NotRun { .. } => 3,
    }
}

fn describe(t: &TriState) -> (&'static str, String) {
    match t {
        TriState::Ran { passed: true, detail, .. } => ("PASS", detail.clone()),
        TriState::Ran { passed: false, detail, .. } => ("FAILED", detail.clone()),
        TriState::NotRun { reason } => ("NOT_RUN", reason.clone()),
    }
}

/// The command line: returns the report (ending in the `RESULT:` line) and the exit code.
/// A bad command line is exit 2 with no `RESULT:` line, since nothing was checked.
pub fn run(args: &[String]) -> (String, u8) {
    let usage = "usage: qd-prereg-bind --prereg FILE --shard-dir DIR [--repo DIR] [--head SHA]";
    let (mut prereg, mut shard, mut repo, mut head) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            return (format!("qd-prereg-bind: {flag} needs a value\n{usage}\n"), 2);
        };
        match flag.as_str() {
            "--prereg" => prereg = Some(PathBuf::from(value)),
            "--shard-dir" => shard = Some(PathBuf::from(value)),
            "--repo" => repo = Some(PathBuf::from(value)),
            "--head" => head = Some(value.clone()),
            other => return (format!("qd-prereg-bind: unknown argument {other:?}\n{usage}\n"), 2),
        }
    }
    let (Some(prereg), Some(shard_dir)) = (prereg, shard) else {
        return (format!("qd-prereg-bind: --prereg and --shard-dir are required\n{usage}\n"), 2);
    };
    let repo = repo.unwrap_or_else(|| PathBuf::from("."));
    let prereg = if prereg.is_absolute() { prereg } else { repo.join(prereg) };
    let (checks, overall) = bind(&BindInputs { prereg, repo, shard_dir, head });
    let mut out = String::new();
    for (name, t) in &checks {
        let (label, detail) = describe(t);
        out.push_str(&format!("{label:8} {name}: {detail}\n"));
    }
    let (label, _) = describe(&overall);
    out.push_str(&format!("RESULT: {label}\n"));
    (out, exit_code(&overall))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    const HASH: &str = "108082545a66ec3e0b008f8ad7a8425625cd9bc323630f8d0cd186f40e00cdf4";

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            static N: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "qd-prereg-bind-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::SeqCst)
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn sh(repo: &Path, args: &[&str]) -> String {
        let (code, out) = git(repo, args).unwrap();
        assert_eq!(code, 0, "git {args:?}");
        out
    }

    fn write(repo: &Path, rel: &str, body: &str) {
        let p = repo.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    fn commit_all(repo: &Path, msg: &str) -> String {
        sh(repo, &["add", "."]);
        sh(repo, &["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg]);
        sh(repo, &["rev-parse", "HEAD"])
    }

    fn prereg_json() -> String {
        format!(
            r#"{{"data": {{"train_shard": {{"shard_hash": "{HASH}"}}}}, "design_by": "AUDIT/d/ruling.md and campaign/other.json; ledger/*.jsonl"}}"#
        )
    }

    /// A repo with the registration and the one decision record it cites, committed.
    fn fixture() -> (Scratch, Scratch, String) {
        let repo = Scratch::new();
        sh(repo.path(), &["init", "-q"]);
        write(repo.path(), "campaign/p.json", &prereg_json());
        write(repo.path(), "AUDIT/d/ruling.md", "ruling");
        write(repo.path(), "campaign/other.json", "{}");
        let c = commit_all(repo.path(), "register");
        let shards = Scratch::new();
        write(shards.path(), "header.json", &format!(r#"{{"shard_hash": "{HASH}"}}"#));
        (repo, shards, c)
    }

    fn args(repo: &Scratch, shards: &Scratch, extra: &[&str]) -> Vec<String> {
        let mut a: Vec<String> = [
            "--prereg", "campaign/p.json", "--shard-dir", shards.path().to_str().unwrap(),
            "--repo", repo.path().to_str().unwrap(),
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        a.extend(extra.iter().map(|s| (*s).to_owned()));
        a
    }

    #[test]
    fn cited_paths_finds_records_and_skips_globs_and_sentence_dots() {
        let text = "see AUDIT/a/b.md. Also campaign/v5.json, docs/x.json and ledger/*.jsonl, \
                    xAUDIT/no.md and HANDOFF/h-2026.md";
        assert_eq!(
            cited_paths(text),
            vec!["AUDIT/a/b.md", "HANDOFF/h-2026.md", "campaign/v5.json", "docs/x.json"]
        );
    }

    #[test]
    fn a_committed_clean_registration_on_the_pinned_shards_passes() {
        let (repo, shards, _) = fixture();
        let (out, code) = run(&args(&repo, &shards, &[]));
        assert_eq!(code, 0, "{out}");
        assert!(out.ends_with("RESULT: PASS\n"), "{out}");
    }

    #[test]
    fn a_shard_set_with_another_hash_fails() {
        let (repo, shards, _) = fixture();
        write(shards.path(), "header.json", &format!(r#"{{"shard_hash": "{}"}}"#, "0".repeat(64)));
        let (out, code) = run(&args(&repo, &shards, &[]));
        assert_eq!(code, 1, "{out}");
        // The label column is `{label:8} `, so FAILED (6) is followed by two pad spaces and the
        // separator; NOT_RUN (7) by one and the separator, as the test below spells it.
        assert!(out.contains("FAILED   shard_hash") && out.ends_with("RESULT: FAILED\n"), "{out}");
    }

    #[test]
    fn a_shard_set_with_no_readable_header_is_not_run_not_a_pass() {
        let (repo, shards, _) = fixture();
        fs::remove_file(shards.path().join("header.json")).unwrap();
        let (out, code) = run(&args(&repo, &shards, &[]));
        assert_eq!(code, 3, "{out}");
        assert!(out.contains("NOT_RUN  shard_hash") && out.ends_with("RESULT: NOT_RUN\n"), "{out}");
    }

    #[test]
    fn a_modified_registration_fails() {
        let (repo, shards, _) = fixture();
        write(repo.path(), "campaign/p.json", &prereg_json().replace("design_by", "design_by2"));
        let (out, code) = run(&args(&repo, &shards, &[]));
        assert_eq!(code, 1, "{out}");
        assert!(out.contains("campaign/p.json differs from HEAD"), "{out}");
    }

    #[test]
    fn a_modified_cited_decision_record_fails() {
        let (repo, shards, _) = fixture();
        write(repo.path(), "AUDIT/d/ruling.md", "ruling, edited after the fact");
        let (out, code) = run(&args(&repo, &shards, &[]));
        assert_eq!(code, 1, "{out}");
        assert!(out.contains("AUDIT/d/ruling.md differs from HEAD"), "{out}");
    }

    #[test]
    fn an_uncommitted_registration_fails() {
        let (repo, shards, _) = fixture();
        write(repo.path(), "campaign/new.json", &prereg_json());
        let mut a = args(&repo, &shards, &[]);
        a[1] = "campaign/new.json".to_owned();
        let (out, code) = run(&a);
        assert_eq!(code, 1, "{out}");
        assert!(out.contains("campaign/new.json is untracked"), "{out}");
        assert!(out.contains("never committed"), "{out}");
    }

    #[test]
    fn a_cited_record_missing_from_the_checkout_is_not_run() {
        let (repo, shards, _) = fixture();
        sh(repo.path(), &["rm", "-q", "AUDIT/d/ruling.md"]);
        commit_all(repo.path(), "drop a record");
        let (out, code) = run(&args(&repo, &shards, &[]));
        assert_eq!(code, 3, "{out}");
        assert!(out.contains("not verifiable: AUDIT/d/ruling.md"), "{out}");
    }

    #[test]
    fn a_run_head_older_than_the_registration_commit_fails() {
        let (repo, shards, registered) = fixture();
        // The run's head is the registration commit itself: an ancestor of itself, so it binds.
        let (out, code) = run(&args(&repo, &shards, &["--head", &registered]));
        assert_eq!(code, 0, "{out}");
        // The registration is then re-committed (a different commit touches the file).
        write(repo.path(), "campaign/p.json", &format!("{} ", prereg_json()));
        let re = commit_all(repo.path(), "re-register");
        // A run whose head is the earlier commit predates it: not built on this registration.
        let (out, code) = run(&args(&repo, &shards, &["--head", &registered]));
        assert_eq!(code, 1, "the run's head {registered} predates {re}\n{out}");
        assert!(out.contains("is not an ancestor"), "{out}");
        let (out, code) = run(&args(&repo, &shards, &["--head", &re]));
        assert_eq!(code, 0, "{out}");
    }

    #[test]
    fn a_head_that_is_not_a_commit_is_not_run() {
        let (repo, shards, _) = fixture();
        let (out, code) = run(&args(&repo, &shards, &["--head", "no-such-ref"]));
        assert_eq!(code, 3, "{out}");
    }

    #[test]
    fn a_bad_command_line_is_exit_2_with_no_result_line() {
        let (out, code) = run(&["--prereg".to_owned()]);
        assert_eq!(code, 2);
        assert!(!out.contains("RESULT:"));
        assert_eq!(run(&[]).1, 2);
    }
}
