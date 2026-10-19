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
//! [`check_pin`] compares HEAD alone. [`check_pin_full`], which `qd-pin` runs, also refuses a
//! sibling whose tracked files differ from HEAD (HEAD then does not pin what builds) and a pin
//! with no delta report; that part shells out to `git status` with optional locks off.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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

/// Pins that predate the delta-report convention (`AUDIT/ojas-pin-<sha7>-<date>/delta-report.md`,
/// first written for 5b67cdd). Each entry is a full commit id and the Lappi commit that set it in
/// `upstream.lock.json`. The list is closed: a new pin needs a delta report, not an entry here.
pub const DELTA_REPORT_ALLOWANCE: &[(&str, &str)] = &[
    ("bc498c4375c2e4753d784e6e9fc267a8d18b038a", "7362473"),
    ("132b46ea63e87868f468c016e02fe99acf2bb164", "21fb7ad"),
];

/// How long `git status` in the sibling may run before it is killed and the check is `NotRun`.
const GIT_STATUS_TIMEOUT: Duration = Duration::from_secs(120);
/// Upper bound on the bytes of `git status` output read; more is a sibling in no state to pin.
const GIT_STATUS_MAX_BYTES: u64 = 4 * 1024 * 1024;
/// How many dirty paths a failure line names.
const NAMED_FILES: usize = 10;

/// What the full pin gate decided. [`PinOutcome`] is the HEAD comparison alone; the two extra
/// failures are checks that HEAD cannot answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinGate {
    /// HEAD matches, the tracked files are clean and the pin has a delta report (or an allowance).
    Pass { sha: String, evidence: String },
    /// The HEAD comparison alone failed or could not run.
    Head(PinOutcome),
    /// HEAD matches but tracked files differ from it: HEAD does not pin what builds.
    Dirty { sha: String, files: Vec<String> },
    /// HEAD matches and is clean, but no delta report covers the pinned commit.
    NoDeltaReport { sha: String },
    /// HEAD matches but git could not say whether the tree is clean.
    DirtyUnknown { sha: String, reason: String },
}

impl PinGate {
    pub fn exit_code(&self) -> u8 {
        match self {
            PinGate::Pass { .. } => 0,
            PinGate::Head(o) => o.exit_code(),
            PinGate::Dirty { .. } | PinGate::NoDeltaReport { .. } => 1,
            PinGate::DirtyUnknown { .. } => EXIT_NOT_RUN,
        }
    }

    pub fn line(&self, key: &str) -> String {
        match self {
            PinGate::Pass { sha, evidence } => {
                format!(
                    "pin PASS: {key} is at the locked commit {sha}, tracked files clean, {evidence}"
                )
            }
            PinGate::Head(o) => o.line(key),
            PinGate::Dirty { sha, files } => {
                let shown: Vec<&str> = files.iter().take(NAMED_FILES).map(String::as_str).collect();
                let more = files.len().saturating_sub(NAMED_FILES);
                format!(
                    "pin FAILED: {key} is at {sha} but {} tracked file(s) differ from it, so the pin \
                     does not describe what builds: {}{}. Commit or revert them in the sibling.",
                    files.len(),
                    shown.join(", "),
                    if more > 0 {
                        format!(" (+{more} more)")
                    } else {
                        String::new()
                    }
                )
            }
            PinGate::NoDeltaReport { sha } => format!(
                "pin FAILED: upstream.lock.json pins {key} {sha} and no AUDIT/{key}-pin-{}-<date>/\
                 delta-report.md covers it. Write the delta report in the bump commit.",
                &sha[..7]
            ),
            PinGate::DirtyUnknown { sha, reason } => format!(
                "pin NotRun: {key} is at {sha} but whether its tracked files are clean is unknown: {reason}"
            ),
        }
    }
}

/// The delta report covering `sha`: a file `<audit_dir>/<key>-pin-<prefix>-<date>/delta-report.md`
/// where `<prefix>` is at least 7 hex characters and a prefix of `sha`.
pub fn find_delta_report(
    audit_dir: &Path,
    key: &str,
    sha: &str,
) -> Result<Option<PathBuf>, String> {
    let entries = fs::read_dir(audit_dir).map_err(|e| {
        format!(
            "cannot list the audit directory {}: {e}",
            audit_dir.display()
        )
    })?;
    let head = format!("{key}-pin-");
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot list {}: {e}", audit_dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(&head) else {
            continue;
        };
        let prefix = rest.split('-').next().unwrap_or("");
        if prefix.len() >= 7 && is_hex_prefix(prefix) && sha.starts_with(prefix) {
            let report = entry.path().join("delta-report.md");
            if report.is_file() {
                return Ok(Some(report));
            }
        }
    }
    Ok(None)
}

fn is_hex_prefix(s: &str) -> bool {
    s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Tracked files of `dir` that differ from HEAD, via `git status` with the optional index lock
/// off so it writes nothing in a tree this repo does not own. Untracked files are ignored: they
/// are not in HEAD and `cargo` only follows what the manifests name. Bounded in time and output;
/// the child is the one this call started and the only one it kills.
pub fn dirty_tracked_files(dir: &Path) -> Result<Vec<String>, String> {
    use std::io::Read;
    use std::process::{Command, Stdio};

    let mut child = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(dir)
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=no"])
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run git: {e}"))?;
    let stdout = child.stdout.take().ok_or("git stdout was not captured")?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.take(GIT_STATUS_MAX_BYTES + 1).read_to_end(&mut buf);
        buf
    });
    let deadline = Instant::now() + GIT_STATUS_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("git status in {} timed out", dir.display()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(format!("waiting for git failed: {e}")),
        }
    };
    let out = reader
        .join()
        .map_err(|_| "git reader thread panicked".to_owned())?;
    if !status.success() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        return Err(format!(
            "git status in {} exited {status}: {}",
            dir.display(),
            err.trim()
        ));
    }
    if out.len() as u64 > GIT_STATUS_MAX_BYTES {
        return Err(format!(
            "git status in {} printed over {GIT_STATUS_MAX_BYTES} bytes",
            dir.display()
        ));
    }
    // `-z`: entries are "XY path\0"; a rename or copy is followed by one extra "orig\0" field.
    let text = String::from_utf8_lossy(&out);
    let mut files = Vec::new();
    let mut fields = text.split('\0').filter(|f| !f.is_empty());
    while let Some(field) = fields.next() {
        let (xy, path) = field.split_at(field.len().min(3));
        files.push(path.to_owned());
        if xy.starts_with(['R', 'C'])
            || xy
                .as_bytes()
                .get(1)
                .is_some_and(|b| matches!(b, b'R' | b'C'))
        {
            fields.next();
        }
    }
    Ok(files)
}

/// The whole gate: HEAD against the lock, then a clean tracked tree, then a delta report (or a
/// listed allowance) for the pinned commit.
pub fn check_pin_full(lock_path: &Path, key: &str, sibling: &Path, audit_dir: &Path) -> PinGate {
    let sha = match check_pin(lock_path, key, sibling) {
        PinOutcome::Match { sha } => sha,
        other => return PinGate::Head(other),
    };
    match dirty_tracked_files(sibling) {
        Ok(files) if !files.is_empty() => return PinGate::Dirty { sha, files },
        Ok(_) => {}
        Err(reason) => return PinGate::DirtyUnknown { sha, reason },
    }
    if let Some((_, set_by)) = DELTA_REPORT_ALLOWANCE.iter().find(|(s, _)| *s == sha) {
        return PinGate::Pass {
            sha,
            evidence: format!("pin predates delta reports (allowance, set by commit {set_by})"),
        };
    }
    match find_delta_report(audit_dir, key, &sha) {
        Ok(Some(report)) => PinGate::Pass {
            evidence: format!("delta report {}", report.display()),
            sha,
        },
        Ok(None) => PinGate::NoDeltaReport { sha },
        Err(reason) => PinGate::Head(PinOutcome::NotRun { reason }),
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

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
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
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    /// A real one-commit repo with a tracked file; returns its HEAD.
    fn real_repo(dir: &Path) -> String {
        fs::create_dir_all(dir).unwrap();
        git(dir, &["init", "-q"]);
        write(&dir.join("lib.rs"), "fn a() {}\n");
        git(dir, &["add", "lib.rs"]);
        git(dir, &["commit", "-q", "-m", "one"]);
        git(dir, &["rev-parse", "HEAD"])
    }

    fn audit_with_report(d: &Path, sha: &str) -> PathBuf {
        let audit = d.join("AUDIT");
        write(
            &audit.join(format!("ojas-pin-{}-2026-10-17/delta-report.md", &sha[..7])),
            "delta\n",
        );
        audit
    }

    #[test]
    fn a_clean_pinned_sibling_with_a_delta_report_passes() {
        let d = scratch("full-pass");
        let sha = real_repo(&d.join("sib"));
        let out = check_pin_full(
            &lock(&d, &sha),
            "ojas",
            &d.join("sib"),
            &audit_with_report(&d, &sha),
        );
        assert_eq!(out.exit_code(), 0, "{}", out.line("ojas"));
        assert!(matches!(out, PinGate::Pass { .. }));
    }

    #[test]
    fn a_dirty_tracked_file_fails_and_names_it_but_an_untracked_one_does_not() {
        let d = scratch("full-dirty");
        let sha = real_repo(&d.join("sib"));
        let audit = audit_with_report(&d, &sha);
        write(&d.join("sib/untracked.txt"), "x");
        assert_eq!(
            check_pin_full(&lock(&d, &sha), "ojas", &d.join("sib"), &audit).exit_code(),
            0
        );
        write(&d.join("sib/lib.rs"), "fn changed() {}\n");
        let out = check_pin_full(&lock(&d, &sha), "ojas", &d.join("sib"), &audit);
        assert_eq!(out.exit_code(), 1);
        let line = out.line("ojas");
        assert!(
            line.contains("lib.rs") && line.contains("pin FAILED"),
            "{line}"
        );
        assert!(!line.contains("untracked.txt"), "{line}");
    }

    #[test]
    fn a_pin_with_no_delta_report_fails() {
        let d = scratch("full-nodelta");
        let sha = real_repo(&d.join("sib"));
        let audit = d.join("AUDIT");
        fs::create_dir_all(&audit).unwrap();
        // A report for a different commit, and a prefix shorter than 7, cover nothing.
        write(
            &audit.join("ojas-pin-0123456-2026-10-17/delta-report.md"),
            "x",
        );
        write(
            &audit.join(format!("ojas-pin-{}-2026-10-17/delta-report.md", &sha[..6])),
            "x",
        );
        let out = check_pin_full(&lock(&d, &sha), "ojas", &d.join("sib"), &audit);
        assert!(matches!(out, PinGate::NoDeltaReport { .. }), "{out:?}");
        assert_eq!(out.exit_code(), 1);
    }

    #[test]
    fn a_sibling_git_cannot_read_is_not_run_never_a_pass() {
        let d = scratch("full-nogit");
        write(&d.join("sib/.git/HEAD"), &format!("{A}\n"));
        let out = check_pin_full(
            &lock(&d, A),
            "ojas",
            &d.join("sib"),
            &audit_with_report(&d, A),
        );
        assert!(matches!(out, PinGate::DirtyUnknown { .. }), "{out:?}");
        assert_eq!(out.exit_code(), 3);
    }

    #[test]
    fn the_allowance_is_closed_and_cites_a_commit_for_every_entry() {
        assert_eq!(DELTA_REPORT_ALLOWANCE.len(), 2);
        for (sha, set_by) in DELTA_REPORT_ALLOWANCE {
            assert!(is_hex_sha(sha) && sha.len() == 40, "{sha}");
            assert!(set_by.len() == 7 && is_hex_prefix(set_by), "{set_by}");
        }
    }
}
