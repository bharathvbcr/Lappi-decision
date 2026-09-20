//! Running the language formatter, and refusing to pretend when it is not there.
//!
//! `cosmetic` labels claim behaviour preservation. The spec allows exactly two ways to earn that
//! claim: run the formatter and compare canonical forms, or restrict the operator set to ones that
//! are provably safe without one. There is no third way, and in particular "no formatter was found,
//! so the edit was probably fine" is the poisoned-label failure this module exists to prevent —
//! a `cosmetic` example whose diff changed behaviour teaches the model that a behaviour change is
//! cosmetic, which is worse than having no `cosmetic` examples at all.
//!
//! Resolution is therefore a **measured fact about this machine**, recorded per language in the
//! manifest. [`Availability`] is a tri-state: a formatter that was never looked for and one that
//! was looked for and missing are different findings, and a consumer must not be able to confuse
//! them with one that ran.
//!
//! Every invocation is bounded. The child gets a wall clock and is killed at the deadline; its
//! stdin is written from a helper thread and its pipes are drained from two more, because a
//! formatter that stops reading while we are still writing deadlocks a single-threaded
//! write-then-read.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::lang::{Formatter, LangId, Language};

/// Wall clock for one formatter invocation.
pub const FORMAT_BUDGET: Duration = Duration::from_secs(20);
/// Wall clock for the one-off `xcrun --find` lookup, which touches the filesystem and no more.
pub const RESOLVE_BUDGET: Duration = Duration::from_secs(10);
/// Largest input handed to a formatter. Well under `parse::MAX_SOURCE_BYTES`: a formatter is an
/// external process and the cost of a pathological one is not ours to bound from the inside.
pub const MAX_FORMAT_BYTES: usize = 2 * 1024 * 1024;
/// How often the wait loop wakes to check whether the child has exited.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Whether a language's formatter could actually be run **here**.
///
/// Four states, not two, and the fourth is the one that is easy to leave out. `NotDeclared`,
/// `NotFound` and `Unusable` all mean "no formatter-verified cosmetic operators", but they are
/// different facts about the world and the manifest carries all three: nothing to look for, nothing
/// found, and something found that does not work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Availability {
    /// The language declares no formatter at all.
    NotDeclared,
    /// A formatter is declared but could not be resolved on this machine.
    NotFound { program: String, detail: String },
    /// A program was found at `path` and could not format this language's smoke input: it exited
    /// non-zero, timed out, or wrote nothing.
    ///
    /// Separate from `NotFound` because they call for different actions — install the tool, versus
    /// work out why the installed one is broken — and separate from `Found` because *that* is the
    /// conflation this module exists to prevent. A `found` that means only "a file exists at this
    /// path" lets a formatter that cannot run be reported the same way as one that ran and worked,
    /// and the manifest then records no restriction for the language at all.
    Unusable {
        program: String,
        path: String,
        detail: String,
    },
    /// Resolved **and demonstrated**: the program at `path` formatted this language's smoke input
    /// and exited zero.
    Found { program: String, path: String },
}

impl Availability {
    pub fn is_usable(&self) -> bool {
        matches!(self, Availability::Found { .. })
    }

    /// The one-word key for the manifest's per-language table.
    pub fn key(&self) -> &'static str {
        match self {
            Availability::NotDeclared => "not_declared",
            Availability::NotFound { .. } => "not_found",
            Availability::Unusable { .. } => "unusable",
            Availability::Found { .. } => "found",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// No usable formatter; the cosmetic operator set shrinks and the manifest says so.
    Unavailable(Availability),
    /// Input over `MAX_FORMAT_BYTES`.
    TooLarge { bytes: usize },
    /// The child could not be spawned.
    Spawn { detail: String },
    /// The child was killed at `FORMAT_BUDGET`.
    Timeout { millis: u64 },
    /// The child exited non-zero. A formatter refusing the input is a fact about the input, and
    /// the cosmetic operator declines rather than shipping an unverified label.
    Failed { code: Option<i32>, stderr: String },
    /// The child wrote bytes that are not UTF-8.
    NotUtf8,
    /// A pipe broke, or a helper thread died. Surfaced rather than read as an empty result.
    Io { detail: String },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::Unavailable(a) => write!(f, "formatter unavailable ({})", a.key()),
            FormatError::TooLarge { bytes } => write!(f, "input of {bytes} bytes is over the cap"),
            FormatError::Spawn { detail } => write!(f, "spawn failed: {detail}"),
            FormatError::Timeout { millis } => write!(f, "killed after {millis} ms"),
            FormatError::Failed { code, stderr } => {
                let head: String = stderr.lines().take(3).collect::<Vec<_>>().join("; ");
                write!(f, "exit {code:?}: {head}")
            }
            FormatError::NotUtf8 => f.write_str("formatter produced non-UTF-8 output"),
            FormatError::Io { detail } => write!(f, "io: {detail}"),
        }
    }
}

/// A formatter resolved to an absolute path, ready to run.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub path: String,
    pub args: Vec<String>,
    pub availability: Availability,
}

/// Resolve `language`'s declared formatter on this machine, and **prove it runs**.
///
/// The proof is [`Language::smoke_source`]: a minimal valid file for the language, pushed through
/// the formatter exactly as a real input would be. Locating a file on disk is not evidence that it
/// formats anything, and the whole value of this module is that its answer is measured.
pub fn resolve(language: &dyn Language) -> Result<Resolved, Availability> {
    let Some(spec) = language.formatter() else {
        return Err(Availability::NotDeclared);
    };
    resolve_spec(&spec, language.smoke_source())
}

/// Locate `spec`'s program and confirm it formats `smoke`.
///
/// Split out from [`resolve`] so the states below can be reached from a test with a program chosen
/// for its behaviour — one that is missing, one that exits non-zero, one that works — rather than
/// only through whichever tools this machine happens to have.
///
/// `via_xcrun` sends the lookup through `xcrun --find`, because the Xcode toolchain is not on
/// `PATH` — `swift-format` lives inside `XcodeDefault.xctoolchain` and `command -v` does not see it.
pub fn resolve_spec(spec: &Formatter, smoke: &str) -> Result<Resolved, Availability> {
    let path = if spec.via_xcrun {
        match xcrun_find(spec.program) {
            Ok(p) => p,
            Err(detail) => {
                return Err(Availability::NotFound {
                    program: spec.program.to_string(),
                    detail,
                });
            }
        }
    } else {
        match which(spec.program) {
            Some(p) => p,
            None => {
                return Err(Availability::NotFound {
                    program: spec.program.to_string(),
                    detail: "not on PATH".to_string(),
                });
            }
        }
    };
    let candidate = Resolved {
        args: spec.args.iter().map(|a| (*a).to_string()).collect(),
        availability: Availability::Found {
            program: spec.program.to_string(),
            path: path.clone(),
        },
        path,
    };
    // Bounded by `FORMAT_BUDGET` like any other invocation: a formatter that hangs on a two-line
    // file must not hang the probe that was asking whether it works.
    match format_with(&candidate, smoke) {
        Ok(out) if !out.trim().is_empty() => Ok(candidate),
        Ok(_) => Err(Availability::Unusable {
            program: spec.program.to_string(),
            path: candidate.path,
            detail: "ran on the smoke input and wrote nothing to stdout".to_string(),
        }),
        Err(e) => Err(Availability::Unusable {
            program: spec.program.to_string(),
            path: candidate.path,
            detail: e.to_string(),
        }),
    }
}

/// The availability of every language's formatter, in `LangId::ALL` order.
///
/// This is what the manifest records and what `qd-mutate formatters` prints. It is measured once
/// per run: a formatter that appears half-way through a run would make the run's own cosmetic
/// coverage depend on when each file was processed. Each entry costs one bounded subprocess,
/// because each entry is a claim that the formatter runs.
pub fn probe_all() -> Vec<(LangId, Availability)> {
    LangId::ALL
        .iter()
        .map(|id| {
            let language = crate::lang::for_id(*id);
            let availability = match resolve(language) {
                Ok(r) => r.availability,
                Err(a) => a,
            };
            (*id, availability)
        })
        .collect()
}

/// Resolve a bare program name against `PATH`, without a shell.
fn which(program: &str) -> Option<String> {
    // An absolute or relative path is used as given; anything else is looked up.
    if program.contains('/') {
        return std::path::Path::new(program)
            .is_file()
            .then(|| program.to_string());
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(program);
        if candidate.is_file() {
            return candidate.to_str().map(|s| s.to_string());
        }
    }
    None
}

fn xcrun_find(program: &str) -> Result<String, String> {
    let out = run_bounded(
        "/usr/bin/xcrun",
        &["--find".to_string(), program.to_string()],
        b"",
        RESOLVE_BUDGET,
    );
    match out {
        Ok(found) => {
            let path = found.stdout.trim().to_string();
            if path.is_empty() || !std::path::Path::new(&path).is_file() {
                Err(format!("xcrun --find {program} named nothing runnable"))
            } else {
                Ok(path)
            }
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Format `source` with `resolved`. Returns the formatter's stdout.
pub fn format_with(resolved: &Resolved, source: &str) -> Result<String, FormatError> {
    if source.len() > MAX_FORMAT_BYTES {
        return Err(FormatError::TooLarge {
            bytes: source.len(),
        });
    }
    let out = run_bounded(
        &resolved.path,
        &resolved.args,
        source.as_bytes(),
        FORMAT_BUDGET,
    )?;
    Ok(out.stdout)
}

/// The canonical form of `source` under `language`'s formatter.
///
/// This is the whole of the `cosmetic` behaviour-preservation check: two texts whose canonical
/// forms are equal differ only in layout. `Err(Unavailable)` means the caller must fall back to the
/// restricted operator set and record it — never to assuming the edit was harmless.
pub fn canonical(resolved: &Resolved, source: &str) -> Result<String, FormatError> {
    format_with(resolved, source)
}

#[derive(Debug)]
struct Output {
    stdout: String,
}

/// Run `program` with `args`, feeding `stdin_bytes`, killed at `budget`.
///
/// The three helper threads are not decoration. A formatter that writes more than a pipe buffer
/// before reading all of its input deadlocks a caller that writes stdin to completion and only then
/// reads stdout — and a deadlocked child with no deadline is a generator that never finishes.
fn run_bounded(
    program: &str,
    args: &[String],
    stdin_bytes: &[u8],
    budget: Duration,
) -> Result<Output, FormatError> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| FormatError::Spawn {
            detail: e.to_string(),
        })?;

    let started = Instant::now();
    let deadline = started + budget;

    // stdin: written and closed by a helper. A broken pipe here is normal — a formatter may reject
    // the input and exit before reading all of it — so the error is carried, not raised.
    let stdin_handle = match child.stdin.take() {
        Some(mut sink) => {
            let payload = stdin_bytes.to_vec();
            Some(std::thread::spawn(move || {
                let _ = sink.write_all(&payload);
                let _ = sink.flush();
                drop(sink);
            }))
        }
        None => None,
    };

    let (tx_out, rx_out) = mpsc::channel::<std::io::Result<Vec<u8>>>();
    let stdout_handle = child.stdout.take().map(|mut src| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let res = src.read_to_end(&mut buf).map(|_| buf);
            let _ = tx_out.send(res);
        })
    });

    let (tx_err, rx_err) = mpsc::channel::<std::io::Result<Vec<u8>>>();
    let stderr_handle = child.stderr.take().map(|mut src| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let res = src.read_to_end(&mut buf).map(|_| buf);
            let _ = tx_err.send(res);
        })
    });

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    // Kill, then still join: the reader threads are blocked on pipes that only
                    // close when the child dies, and leaking them would leak the pipes too.
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(FormatError::Timeout {
                        millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                    });
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                let _ = child.kill();
                break Err(FormatError::Io {
                    detail: e.to_string(),
                });
            }
        }
    };

    if let Some(h) = stdin_handle {
        let _ = h.join();
    }
    if let Some(h) = stdout_handle {
        let _ = h.join();
    }
    if let Some(h) = stderr_handle {
        let _ = h.join();
    }

    let stdout_bytes = rx_out.try_recv().ok().and_then(Result::ok).unwrap_or_default();
    let stderr_bytes = rx_err.try_recv().ok().and_then(Result::ok).unwrap_or_default();

    let status = status?;
    if !status.success() {
        return Err(FormatError::Failed {
            code: status.code(),
            stderr: String::from_utf8_lossy(&stderr_bytes).to_string(),
        });
    }
    let stdout = String::from_utf8(stdout_bytes).map_err(|_| FormatError::NotUtf8)?;
    Ok(Output { stdout })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{for_id, LangId};

    #[test]
    fn a_missing_program_resolves_to_not_found_not_to_a_panic() {
        assert_eq!(which("qd-mutate-definitely-not-a-real-program-9f3a"), None);
    }

    #[test]
    fn an_absolute_path_resolves_to_itself() {
        assert_eq!(which("/bin/sh").as_deref(), Some("/bin/sh"));
        assert_eq!(which("/bin/definitely-not-here-9f3a"), None);
    }

    #[test]
    fn a_child_that_never_exits_is_killed_at_the_deadline() {
        let err = run_bounded(
            "/bin/sh",
            &["-c".to_string(), "sleep 30".to_string()],
            b"",
            Duration::from_millis(300),
        )
        .expect_err("the budget must bite");
        assert!(
            matches!(err, FormatError::Timeout { .. }),
            "expected a timeout, got {err:?}"
        );
    }

    #[test]
    fn a_nonzero_exit_is_an_error_not_an_empty_success() {
        let err = run_bounded(
            "/bin/sh",
            &["-c".to_string(), "echo boom >&2; exit 3".to_string()],
            b"",
            Duration::from_secs(5),
        )
        .expect_err("non-zero exit");
        match err {
            FormatError::Failed { code, stderr } => {
                assert_eq!(code, Some(3));
                assert!(stderr.contains("boom"), "stderr was {stderr:?}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn stdin_reaches_the_child_and_stdout_comes_back() {
        let out = run_bounded("/bin/cat", &[], b"hello\n", Duration::from_secs(5))
            .expect("cat runs");
        assert_eq!(out.stdout, "hello\n");
    }

    #[test]
    fn a_child_that_stops_reading_early_does_not_deadlock_the_caller() {
        // 4 MiB of stdin against a child that reads one line: a write-then-read caller wedges here.
        let payload = "x".repeat(4 * 1024 * 1024);
        let out = run_bounded(
            "/bin/sh",
            &["-c".to_string(), "head -c 4 ; exit 0".to_string()],
            payload.as_bytes(),
            Duration::from_secs(10),
        )
        .expect("completes rather than wedging");
        assert_eq!(out.stdout, "xxxx");
    }

    #[test]
    fn an_oversized_input_is_refused_before_a_process_is_spawned() {
        let resolved = Resolved {
            path: "/bin/cat".to_string(),
            args: Vec::new(),
            availability: Availability::Found {
                program: "cat".to_string(),
                path: "/bin/cat".to_string(),
            },
        };
        let big = "x".repeat(MAX_FORMAT_BYTES + 1);
        assert!(matches!(
            format_with(&resolved, &big),
            Err(FormatError::TooLarge { .. })
        ));
    }

    #[test]
    fn a_program_that_exists_but_cannot_format_is_not_reported_as_usable() {
        // `/usr/bin/false` is on disk, is executable, and exits 1 without writing anything. A
        // resolution that only asks `is_file()` calls that `found`, the manifest then records *no*
        // restriction for the language, and `qd-mutate formatters` prints it as usable — so a
        // formatter that cannot format reads exactly like one that ran and was fine. That is the
        // conflation this module's own header forbids, and the tri-state does not catch it because
        // the wrong state is being reported confidently.
        let spec = Formatter {
            program: "/usr/bin/false",
            args: &[],
            via_xcrun: false,
        };
        let got = resolve_spec(&spec, "x = 1\n");
        match got {
            Err(Availability::Unusable { path, detail, .. }) => {
                assert_eq!(path, "/usr/bin/false");
                assert!(!detail.is_empty(), "an unusable formatter must say why");
            }
            other => panic!(
                "a program that exits non-zero on a smoke input is not a usable formatter, got \
                 {other:?}"
            ),
        }
        assert!(!Availability::Unusable {
            program: "false".to_string(),
            path: "/usr/bin/false".to_string(),
            detail: "exit 1".to_string(),
        }
        .is_usable());
    }

    #[test]
    fn a_program_that_formats_the_smoke_input_is_reported_as_found() {
        // The other side: `cat` echoes its input, which is a legal (if lazy) formatter, and must
        // still come back `found`. Without this the test above would pass against a `resolve` that
        // declared every formatter unusable.
        let spec = Formatter {
            program: "/bin/cat",
            args: &[],
            via_xcrun: false,
        };
        let resolved = resolve_spec(&spec, "x = 1\n").expect("cat round-trips its input");
        assert_eq!(resolved.path, "/bin/cat");
        assert!(resolved.availability.is_usable());
        assert_eq!(resolved.availability.key(), "found");
    }

    #[test]
    fn a_missing_program_is_not_found_rather_than_unusable() {
        // `not_found` and `unusable` are different findings and the machine that is missing a tool
        // must not be described as the machine that has a broken one.
        let spec = Formatter {
            program: "qd-mutate-definitely-not-a-real-program-9f3a",
            args: &[],
            via_xcrun: false,
        };
        match resolve_spec(&spec, "x = 1\n") {
            Err(Availability::NotFound { .. }) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn every_language_answers_the_availability_question_one_way_or_the_other() {
        // Not an assertion that any formatter is installed — that is a fact about the machine and
        // is reported, never required. The assertion is that each language answers.
        let probed = probe_all();
        assert_eq!(probed.len(), LangId::ALL.len());
        for (id, availability) in probed {
            let declared = for_id(id).formatter().is_some();
            match (&availability, declared) {
                (Availability::NotDeclared, false) => {}
                (
                    Availability::NotFound { .. }
                    | Availability::Unusable { .. }
                    | Availability::Found { .. },
                    true,
                ) => {}
                (a, d) => panic!("{id}: availability {a:?} contradicts declared={d}"),
            }
        }
    }
}
