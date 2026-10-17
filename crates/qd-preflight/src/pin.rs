//! The sibling pin: does the checkout beside this repo sit at the commit `upstream.lock.json`
//! names?
//!
//! Lappi builds against `ojas` by path (`crates/qd-train-metal/Cargo.toml`), so what it builds
//! against is whatever commit that checkout happens to be on. The lock file is the claim, this
//! module is the check. The commit is read with plain file reads of the sibling's `.git`
//! (`HEAD`, a loose ref, `packed-refs`, a `gitdir:` file, `commondir`), never by running `git`
//! in it: a gate that shells out to a tool in a tree it does not own is a gate that can move
//! that tree's state or hang on its lock.
//!
//! Three outcomes, never two (the repo's tri-state): [`PinOutcome::Match`] (exit 0),
//! [`PinOutcome::Mismatch`] (exit 1, names both commits) and [`PinOutcome::NotRun`] (exit 3,
//! the sibling or the lock could not be read). A sibling that is absent is `NotRun`, never a
//! pass.
//!
//! What this does not establish: that the sibling's working tree is clean. It compares HEAD.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The exit code of a check that could not run (the Makefile's `NOT_RUN`).
pub const EXIT_NOT_RUN: u8 = 3;

/// The lock's entry for one sibling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockEntry {
    pub revision: String,
    pub url: Option<String>,
}

/// The result of comparing a sibling's HEAD against its lock entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinOutcome {
    Match { sha: String },
    Mismatch { locked: String, actual: String },
    NotRun { reason: String },
}

impl PinOutcome {
    /// 0 for a match, 1 for a mismatch, 3 for a check that could not run.
    pub fn exit_code(&self) -> u8 {
        match self {
            PinOutcome::Match { .. } => 0,
            PinOutcome::Mismatch { .. } => 1,
            PinOutcome::NotRun { .. } => EXIT_NOT_RUN,
        }
    }

    /// The one line the gate prints.
    pub fn line(&self, key: &str) -> String {
        match self {
            PinOutcome::Match { sha } => format!("pin PASS: {key} is at the locked commit {sha}"),
            PinOutcome::Mismatch { locked, actual } => format!(
                "pin FAILED: {key} is at {actual} but upstream.lock.json pins {locked}. \
                 Check the sibling out at the locked commit, or bump the lock in a reviewed commit."
            ),
            PinOutcome::NotRun { reason } => format!("pin NotRun: {key}: {reason}"),
        }
    }
}

fn is_hex_sha(s: &str) -> bool {
    matches!(s.len(), 40 | 64) && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Read `lock_path` (`{"schema_version": 1, "<key>": {"url": ..., "revision": <sha>}}`) and
/// return `key`'s entry.
pub fn read_lock(lock_path: &Path, key: &str) -> Result<LockEntry, String> {
    let text = fs::read_to_string(lock_path)
        .map_err(|e| format!("cannot read the lock {}: {e}", lock_path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| format!("the lock {} is not JSON: {e}", lock_path.display()))?;
    match v.get("schema_version").and_then(Value::as_u64) {
        Some(1) => {}
        other => {
            return Err(format!(
                "the lock {} has schema_version {other:?}; this reader knows 1",
                lock_path.display()
            ));
        }
    }
    let entry = v
        .get(key)
        .ok_or_else(|| format!("the lock {} has no entry {key:?}", lock_path.display()))?;
    let revision = entry
        .get("revision")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("the lock's {key:?} entry has no string \"revision\""))?;
    if !is_hex_sha(revision) {
        return Err(format!(
            "the lock's {key:?} revision {revision:?} is not a full lowercase hex commit id"
        ));
    }
    Ok(LockEntry {
        revision: revision.to_owned(),
        url: entry.get("url").and_then(Value::as_str).map(str::to_owned),
    })
}

fn read_trimmed(path: &Path) -> Result<String, String> {
    fs::read_to_string(path)
        .map(|s| s.trim().to_owned())
        .map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// Locate the git directory of `dir`: `dir/.git` when it is a directory, or the target of a
/// `gitdir:` file (relative paths resolve against `dir`). Symlinked checkouts need nothing
/// special: the filesystem follows the link.
fn git_dir(dir: &Path) -> Result<PathBuf, String> {
    let dot_git = dir.join(".git");
    let meta = fs::metadata(&dot_git).map_err(|e| {
        format!(
            "{} is not a git checkout ({}: {e})",
            dir.display(),
            dot_git.display()
        )
    })?;
    if meta.is_dir() {
        return Ok(dot_git);
    }
    let text = read_trimmed(&dot_git)?;
    let target = text
        .strip_prefix("gitdir:")
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| format!("{} is a file without a \"gitdir:\" line", dot_git.display()))?;
    let target = Path::new(target);
    Ok(if target.is_absolute() {
        target.to_path_buf()
    } else {
        dir.join(target)
    })
}

/// A ref name read out of `HEAD` is data from the sibling's tree: refuse anything that could
/// leave `.git`.
fn safe_ref(name: &str) -> Result<&str, String> {
    let bad = name.is_empty()
        || name.starts_with('/')
        || name.contains('\\')
        || name
            .split('/')
            .any(|c| c.is_empty() || c == "." || c == "..");
    if bad {
        return Err(format!("HEAD names an unsafe ref {name:?}"));
    }
    Ok(name)
}

/// The commit `dir`'s HEAD points at, by plain file reads.
pub fn resolve_head(dir: &Path) -> Result<String, String> {
    let gitdir = git_dir(dir)?;
    // A linked worktree keeps HEAD in its own gitdir and refs in the common one.
    let common = match fs::read_to_string(gitdir.join("commondir")) {
        Ok(s) => {
            let p = Path::new(s.trim());
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                gitdir.join(p)
            }
        }
        Err(_) => gitdir.clone(),
    };
    let head = read_trimmed(&gitdir.join("HEAD"))?;
    let Some(refname) = head.strip_prefix("ref:").map(str::trim) else {
        return if is_hex_sha(&head) {
            Ok(head)
        } else {
            Err(format!(
                "{}/HEAD is neither a ref nor a commit id: {head:?}",
                gitdir.display()
            ))
        };
    };
    let refname = safe_ref(refname)?;
    for base in [&gitdir, &common] {
        if let Ok(s) = fs::read_to_string(base.join(refname)) {
            let sha = s.trim();
            return if is_hex_sha(sha) {
                Ok(sha.to_owned())
            } else {
                Err(format!("ref {refname} holds {sha:?}, not a commit id"))
            };
        }
    }
    let packed = fs::read_to_string(common.join("packed-refs")).map_err(|_| {
        format!("ref {refname} has no loose file and there is no packed-refs: an unborn branch")
    })?;
    for line in packed.lines() {
        if line.starts_with('#') || line.starts_with('^') {
            continue;
        }
        if let Some((sha, name)) = line.split_once(' ')
            && name.trim() == refname
        {
            return if is_hex_sha(sha) {
                Ok(sha.to_owned())
            } else {
                Err(format!(
                    "packed-refs holds {sha:?} for {refname}, not a commit id"
                ))
            };
        }
    }
    Err(format!(
        "ref {refname} is in neither the loose refs nor packed-refs"
    ))
}

/// Compare `sibling`'s HEAD with `key`'s entry in `lock_path`.
pub fn check_pin(lock_path: &Path, key: &str, sibling: &Path) -> PinOutcome {
    let locked = match read_lock(lock_path, key) {
        Ok(e) => e.revision,
        Err(reason) => return PinOutcome::NotRun { reason },
    };
    match resolve_head(sibling) {
        Ok(actual) if actual == locked => PinOutcome::Match { sha: actual },
        Ok(actual) => PinOutcome::Mismatch { locked, actual },
        Err(reason) => PinOutcome::NotRun { reason },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    const A: &str = "bc498c4375c2e4753d784e6e9fc267a8d18b038a";
    const B: &str = "0123456789abcdef0123456789abcdef01234567";

    fn scratch(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "qd-pin-{}-{tag}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn write(p: &Path, s: &str) {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, s).unwrap();
    }

    fn lock(dir: &Path, sha: &str) -> PathBuf {
        let p = dir.join("upstream.lock.json");
        write(
            &p,
            &format!(r#"{{"schema_version":1,"ojas":{{"url":"u","revision":"{sha}"}}}}"#),
        );
        p
    }

    #[test]
    fn a_loose_ref_resolves() {
        let d = scratch("loose");
        write(&d.join(".git/HEAD"), "ref: refs/heads/main\n");
        write(&d.join(".git/refs/heads/main"), &format!("{A}\n"));
        assert_eq!(resolve_head(&d).unwrap(), A);
    }

    #[test]
    fn a_packed_ref_resolves_when_there_is_no_loose_file() {
        let d = scratch("packed");
        write(&d.join(".git/HEAD"), "ref: refs/heads/main\n");
        write(
            &d.join(".git/packed-refs"),
            &format!("# pack-refs with: peeled\n{B} refs/heads/other\n^{A}\n{A} refs/heads/main\n"),
        );
        assert_eq!(resolve_head(&d).unwrap(), A);
    }

    #[test]
    fn a_detached_head_resolves() {
        let d = scratch("detached");
        write(&d.join(".git/HEAD"), &format!("{A}\n"));
        assert_eq!(resolve_head(&d).unwrap(), A);
    }

    #[test]
    fn a_gitdir_file_with_commondir_resolves_a_linked_worktree() {
        let d = scratch("gitdir");
        let common = d.join("main/.git");
        let wt_git = common.join("worktrees/wt");
        write(&common.join("refs/heads/feature"), &format!("{A}\n"));
        write(&wt_git.join("HEAD"), "ref: refs/heads/feature\n");
        write(&wt_git.join("commondir"), "../..\n");
        let wt = d.join("wt");
        write(&wt.join(".git"), &format!("gitdir: {}\n", wt_git.display()));
        assert_eq!(resolve_head(&wt).unwrap(), A);
        // A relative gitdir resolves against the checkout, not the process directory.
        let rel = d.join("rel");
        write(&rel.join(".git"), "gitdir: ../main/.git/worktrees/wt\n");
        assert_eq!(resolve_head(&rel).unwrap(), A);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_checkout_resolves() {
        let d = scratch("symlink");
        write(&d.join("real/.git/HEAD"), &format!("{A}\n"));
        std::os::unix::fs::symlink(d.join("real"), d.join("link")).unwrap();
        assert_eq!(resolve_head(&d.join("link")).unwrap(), A);
    }

    #[test]
    fn a_ref_that_escapes_the_git_dir_is_refused() {
        let d = scratch("escape");
        write(&d.join(".git/HEAD"), "ref: ../../secret\n");
        assert!(resolve_head(&d).unwrap_err().contains("unsafe ref"));
    }

    #[test]
    fn an_unborn_or_malformed_head_is_an_error_not_a_sha() {
        let d = scratch("unborn");
        write(&d.join(".git/HEAD"), "ref: refs/heads/main\n");
        assert!(resolve_head(&d).is_err());
        write(&d.join(".git/refs/heads/main"), "not-a-sha\n");
        assert!(resolve_head(&d).is_err());
    }

    #[test]
    fn a_matching_head_passes_with_exit_0() {
        let d = scratch("match");
        write(&d.join("sib/.git/HEAD"), &format!("{A}\n"));
        let out = check_pin(&lock(&d, A), "ojas", &d.join("sib"));
        assert_eq!(out, PinOutcome::Match { sha: A.to_owned() });
        assert_eq!(out.exit_code(), 0);
    }

    #[test]
    fn a_different_head_fails_with_exit_1_naming_both_shas() {
        let d = scratch("mismatch");
        write(&d.join("sib/.git/HEAD"), &format!("{B}\n"));
        let out = check_pin(&lock(&d, A), "ojas", &d.join("sib"));
        assert_eq!(out.exit_code(), 1);
        let line = out.line("ojas");
        assert!(line.contains(A) && line.contains(B), "{line}");
    }

    #[test]
    fn an_absent_sibling_is_not_run_with_exit_3_never_a_pass() {
        let d = scratch("absent");
        let out = check_pin(&lock(&d, A), "ojas", &d.join("no-such-dir"));
        assert!(matches!(out, PinOutcome::NotRun { .. }), "{out:?}");
        assert_eq!(out.exit_code(), 3);
        let empty = d.join("empty");
        fs::create_dir_all(&empty).unwrap();
        assert_eq!(check_pin(&lock(&d, A), "ojas", &empty).exit_code(), 3);
    }

    #[test]
    fn a_missing_or_malformed_lock_is_not_run() {
        let d = scratch("badlock");
        write(&d.join("sib/.git/HEAD"), &format!("{A}\n"));
        assert_eq!(
            check_pin(&d.join("none.json"), "ojas", &d.join("sib")).exit_code(),
            3
        );
        for body in [
            "not json",
            r#"{"schema_version":2,"ojas":{"revision":"bc498c4375c2e4753d784e6e9fc267a8d18b038a"}}"#,
            r#"{"schema_version":1}"#,
            r#"{"schema_version":1,"ojas":{"revision":"bc498c4"}}"#,
        ] {
            let p = d.join("l.json");
            write(&p, body);
            assert_eq!(
                check_pin(&p, "ojas", &d.join("sib")).exit_code(),
                3,
                "{body}"
            );
        }
    }
}
