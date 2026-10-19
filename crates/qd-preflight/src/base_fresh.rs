//! Is this checkout built on a fresh base?
//!
//! A worktree cut from a stale ref compiles, tests and commits without complaint while missing
//! every commit that landed since (GAP-AGENT-WORKTREE-ISOLATION-CUTS-FROM-STALE-ORIGIN-MAIN-2026-10-17:
//! three lanes were built on a base 221 commits behind). The check is one question: is the target
//! branch an ancestor of HEAD? If not, HEAD lacks commits the branch has.
//!
//! Three outcomes (the repo's tri-state): [`BaseOutcome::Fresh`] exit 0, [`BaseOutcome::Stale`]
//! exit 1 (counts the missing commits), [`BaseOutcome::NotRun`] exit 3 (git could not answer:
//! not a repository, the target branch does not resolve, git hung). A check that could not run is
//! never a pass.
//!
//! This asks git, which is the owner of ancestry; it reads nothing by hand. Every git call is
//! bounded in time and output, and the child killed on timeout is the one this module started.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Exit code of a check that could not run (the Makefile's `NOT_RUN`).
pub const EXIT_NOT_RUN: u8 = 3;

const GIT_TIMEOUT: Duration = Duration::from_secs(60);
const GIT_MAX_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseOutcome {
    /// `into` is an ancestor of (or equal to) HEAD.
    Fresh {
        head: String,
        into: String,
        into_sha: String,
    },
    /// `into` has `behind` commits HEAD lacks.
    Stale {
        head: String,
        into: String,
        into_sha: String,
        behind: u64,
    },
    NotRun {
        reason: String,
    },
}

impl BaseOutcome {
    pub fn exit_code(&self) -> u8 {
        match self {
            BaseOutcome::Fresh { .. } => 0,
            BaseOutcome::Stale { .. } => 1,
            BaseOutcome::NotRun { .. } => EXIT_NOT_RUN,
        }
    }

    pub fn line(&self) -> String {
        match self {
            BaseOutcome::Fresh {
                head,
                into,
                into_sha,
            } => format!("base-fresh PASS: HEAD {head} contains {into} ({into_sha})"),
            BaseOutcome::Stale {
                head,
                into,
                into_sha,
                behind,
            } => format!(
                "base-fresh FAILED: HEAD {head} is {behind} commit(s) behind {into} ({into_sha}); \
                 it was cut from a stale base. Rebuild the branch on {into} before running anything."
            ),
            BaseOutcome::NotRun { reason } => format!("base-fresh NotRun: {reason}"),
        }
    }

    pub fn result(&self) -> &'static str {
        match self {
            BaseOutcome::Fresh { .. } => "RESULT: PASS",
            BaseOutcome::Stale { .. } => "RESULT: FAILED",
            BaseOutcome::NotRun { .. } => "RESULT: NOT_RUN",
        }
    }
}

/// Run `git -C repo args...`; `Ok((exit code, stdout))` however git exited, `Err` when git could
/// not be run, was killed, was signalled or printed too much.
fn git(repo: &Path, args: &[&str]) -> Result<(i32, String), String> {
    let mut child = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(repo)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot run git: {e}"))?;
    let stdout = child.stdout.take().ok_or("git stdout was not captured")?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.take(GIT_MAX_BYTES + 1).read_to_end(&mut buf);
        buf
    });
    let deadline = Instant::now() + GIT_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "git {} timed out after {GIT_TIMEOUT:?}",
                    args.join(" ")
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => return Err(format!("waiting for git failed: {e}")),
        }
    };
    let out = reader
        .join()
        .map_err(|_| "git reader thread panicked".to_owned())?;
    if out.len() as u64 > GIT_MAX_BYTES {
        return Err(format!(
            "git {} printed over {GIT_MAX_BYTES} bytes",
            args.join(" ")
        ));
    }
    let code = status
        .code()
        .ok_or_else(|| format!("git {} was killed by a signal", args.join(" ")))?;
    Ok((code, String::from_utf8_lossy(&out).trim().to_owned()))
}

fn resolve(repo: &Path, rev: &str) -> Result<String, String> {
    let spec = format!("{rev}^{{commit}}");
    match git(
        repo,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            &spec,
        ],
    )? {
        (0, sha) if !sha.is_empty() => Ok(sha),
        (code, _) => Err(format!(
            "{rev:?} does not resolve to a commit in {} (git exit {code})",
            repo.display()
        )),
    }
}

fn short(sha: &str) -> String {
    sha.chars().take(12).collect()
}

/// Is `into` an ancestor of HEAD in `repo`?
pub fn check_base(repo: &Path, into: &str) -> BaseOutcome {
    if into.is_empty() || into.starts_with('-') {
        return BaseOutcome::NotRun {
            reason: format!("{into:?} is not a usable branch name"),
        };
    }
    let run = || -> Result<BaseOutcome, String> {
        let head = resolve(repo, "HEAD")?;
        let into_sha = resolve(repo, into)?;
        let (head, into_s) = (short(&head), short(&into_sha));
        match git(repo, &["merge-base", "--is-ancestor", &into_sha, "HEAD"])?.0 {
            0 => Ok(BaseOutcome::Fresh {
                head,
                into: into.to_owned(),
                into_sha: into_s,
            }),
            1 => {
                let range = format!("HEAD..{into_sha}");
                let (code, n) = git(repo, &["rev-list", "--count", &range])?;
                let behind = n.parse::<u64>().ok().filter(|_| code == 0).ok_or_else(|| {
                    format!("git rev-list --count {range} gave {n:?} (exit {code})")
                })?;
                Ok(BaseOutcome::Stale {
                    head,
                    into: into.to_owned(),
                    into_sha: into_s,
                    behind,
                })
            }
            code => Err(format!(
                "git merge-base --is-ancestor exited {code}; it answers only 0 or 1"
            )),
        }
    };
    run().unwrap_or_else(|reason| BaseOutcome::NotRun { reason })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn scratch(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "qd-base-fresh-{}-{tag}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn g(dir: &Path, args: &[&str]) -> String {
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
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    fn commit(dir: &Path, n: u32) {
        fs::write(dir.join("f.txt"), n.to_string()).unwrap();
        g(dir, &["add", "f.txt"]);
        g(dir, &["commit", "-q", "-m", &format!("c{n}")]);
    }

    /// A repo whose `main` has `ahead` commits past the commit `topic` was cut from.
    fn repo_with_stale_topic(tag: &str, ahead: u32) -> PathBuf {
        let d = scratch(tag);
        g(&d, &["init", "-q", "-b", "main"]);
        commit(&d, 0);
        g(&d, &["switch", "-q", "-c", "topic"]);
        g(&d, &["switch", "-q", "main"]);
        for n in 1..=ahead {
            commit(&d, n);
        }
        g(&d, &["switch", "-q", "topic"]);
        d
    }

    #[test]
    fn a_branch_cut_from_an_old_main_fails_and_counts_the_missing_commits() {
        let d = repo_with_stale_topic("stale", 5);
        let out = check_base(&d, "main");
        assert!(
            matches!(out, BaseOutcome::Stale { behind: 5, .. }),
            "{out:?}"
        );
        assert_eq!(out.exit_code(), 1);
        assert!(
            out.line().contains("5 commit(s) behind main"),
            "{}",
            out.line()
        );
        assert_eq!(out.result(), "RESULT: FAILED");
    }

    #[test]
    fn a_branch_that_contains_main_passes_even_with_its_own_commits() {
        let d = repo_with_stale_topic("fresh", 3);
        g(&d, &["merge", "-q", "--no-edit", "main"]);
        commit(&d, 99);
        let out = check_base(&d, "main");
        assert!(matches!(out, BaseOutcome::Fresh { .. }), "{out:?}");
        assert_eq!(out.exit_code(), 0);
    }

    #[test]
    fn git_that_cannot_answer_is_not_run_never_a_pass() {
        let not_a_repo = scratch("norepo");
        assert_eq!(check_base(&not_a_repo, "main").exit_code(), 3);
        let d = repo_with_stale_topic("nobranch", 1);
        let out = check_base(&d, "no-such-branch");
        assert!(matches!(out, BaseOutcome::NotRun { .. }), "{out:?}");
        assert_eq!(out.result(), "RESULT: NOT_RUN");
        for bad in ["", "--output=x", "-h"] {
            assert_eq!(check_base(&d, bad).exit_code(), 3, "{bad:?}");
        }
    }
}
