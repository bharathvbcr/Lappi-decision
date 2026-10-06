//! The git CLI, run read-only, bounded in time and output.
//!
//! History is read through the `git` binary, never a git library: a library crate is a new
//! dependency and needs the human's yes (`campaign/v6-caller-families.DRAFT.json`,
//! `build_order` step 2). Every call:
//!
//! - runs with the repository as its working directory, with `GIT_CONFIG_NOSYSTEM=1`,
//!   `GIT_CONFIG_GLOBAL=/dev/null` and `GIT_OPTIONAL_LOCKS=0`, so neither the host's git
//!   configuration (an external diff driver, a pager, a signature display) nor an index
//!   refresh can change what is read or write to the repository;
//! - is killed after [`TIMEOUT`];
//! - is refused when its stdout passes the caller's byte bound, rather than being cut short:
//!   a truncated `git log` is a history with commits missing, and reads as complete.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// One git call's wall-clock bound.
pub const TIMEOUT: Duration = Duration::from_secs(600);
/// Stderr kept for an error message.
const MAX_STDERR_BYTES: u64 = 64 * 1024;
const POLL: Duration = Duration::from_millis(5);

/// Why a git call did not answer.
#[derive(Debug)]
pub enum GitError {
    /// The binary could not be started at all: every later call would fail the same way.
    Spawn(String),
    /// git ran and exited non-zero, or was killed at the timeout.
    Failed(String),
    /// git answered with more than the caller's bound.
    TooLarge(String),
}

impl std::fmt::Display for GitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitError::Spawn(s) | GitError::Failed(s) | GitError::TooLarge(s) => f.write_str(s),
        }
    }
}

/// Run `git <args>` in `repo`; stdout's bytes, at most `max_stdout` of them.
pub fn run(repo: &Path, args: &[&str], max_stdout: u64) -> Result<Vec<u8>, GitError> {
    let what = || format!("git {} (in {})", args.join(" "), repo.display());
    let mut child = Command::new("git")
        .args(args)
        .current_dir(repo)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| GitError::Spawn(format!("{}: {e}", what())))?;
    let (Some(mut out), Some(mut err)) = (child.stdout.take(), child.stderr.take()) else {
        let _ = child.kill();
        return Err(GitError::Spawn(format!(
            "{}: no stdout/stderr pipe",
            what()
        )));
    };
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let read = (&mut out).take(max_stdout + 1).read_to_end(&mut buf);
        // Keep draining past the bound so git is not blocked on a full pipe; what is past the
        // bound is discarded, and the length check below refuses the call.
        let drained = std::io::copy(&mut out, &mut std::io::sink());
        read.and(drained).map(|_| buf)
    });
    let err_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let read = (&mut err).take(MAX_STDERR_BYTES).read_to_end(&mut buf);
        let drained = std::io::copy(&mut err, &mut std::io::sink());
        read.and(drained).map(|_| buf)
    });
    let deadline = Instant::now() + TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(GitError::Failed(format!(
                    "{}: killed after {} s",
                    what(),
                    TIMEOUT.as_secs()
                )));
            }
            Ok(None) => std::thread::sleep(POLL),
            Err(e) => return Err(GitError::Failed(format!("{}: {e}", what()))),
        }
    };
    let stdout = reader
        .join()
        .map_err(|_| GitError::Failed(format!("{}: stdout reader panicked", what())))?
        .map_err(|e| GitError::Failed(format!("{}: reading stdout: {e}", what())))?;
    let stderr = err_reader
        .join()
        .map_err(|_| GitError::Failed(format!("{}: stderr reader panicked", what())))?
        .map_err(|e| GitError::Failed(format!("{}: reading stderr: {e}", what())))?;
    if !status.success() {
        return Err(GitError::Failed(format!(
            "{}: {status}: {}",
            what(),
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    if stdout.len() as u64 > max_stdout {
        return Err(GitError::TooLarge(format!(
            "{}: more than {max_stdout} bytes of output; refusing rather than reading a prefix",
            what()
        )));
    }
    Ok(stdout)
}

/// The commit `rev` names in `repo`, as a full 40-hex sha (`tools/repo_git.py::resolve_rev`'s
/// rule: a symbolic name is not a pin, and a bad revision is an error, not an echo).
pub fn resolve_commit(repo: &Path, rev: &str) -> Result<String, GitError> {
    let spec = format!("{rev}^{{commit}}");
    let out = run(repo, &["rev-parse", "--verify", "--quiet", &spec], 4096)?;
    let sha = String::from_utf8_lossy(&out).trim().to_string();
    if is_full_sha(&sha) {
        Ok(sha)
    } else {
        Err(GitError::Failed(format!(
            "git rev-parse in {} resolved {rev:?} to {sha:?}, which is not a commit sha",
            repo.display()
        )))
    }
}

/// A full lowercase 40-hex SHA-1 object name.
pub fn is_full_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_sha_is_forty_lowercase_hex_and_nothing_else() {
        assert!(is_full_sha(&"a".repeat(40)));
        assert!(!is_full_sha(&"a".repeat(39)));
        assert!(!is_full_sha(&"A".repeat(40)));
        assert!(!is_full_sha("HEAD"));
    }

    #[test]
    fn a_failing_command_is_an_error_carrying_its_stderr() {
        let dir = std::env::temp_dir();
        match run(&dir, &["definitely-not-a-git-subcommand"], 1024) {
            Err(GitError::Failed(msg)) => assert!(msg.contains("definitely-not-a-git-subcommand")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }
}
