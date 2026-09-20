//! Every tracked file in this repository is text. No tracked file carries a NUL byte.
//!
//! # Why this gate exists
//!
//! `crates/qd-runtime/src/wire.rs` carried a literal NUL byte at offset 3858 for four commits. A
//! doc comment meant to *describe* the six-character JSON escape for a NUL and typed the character
//! instead. It compiled — `U+0000` is valid UTF-8 — and all 332 Rust tests passed, so nothing in
//! the suite noticed. What it silently cost, each measured rather than supposed:
//!
//! * **DevMap refused the whole file.** `devmap_status` reported it under
//!   `coverage_gaps.parse_failed`: *"source contains a NUL byte at offset 3858 and is not text;
//!   refused before parsing"*. Every symbol declared in `wire.rs` — `MAX_PAYLOAD_BYTES`,
//!   `parse_slot`, `PayloadBudget`, `known_keys` — was absent from the code graph, and DevMap
//!   answers "nothing" for a symbol it never indexed. That is indistinguishable from "does not
//!   exist", which is the shape of wrong answer this repository keeps designing against.
//! * **ripgrep skipped it.** `rg -l MAX_PAYLOAD_BYTES crates/` listed six files and not the one
//!   that *defines* the constant. `CLAUDE.md` makes `rg` the default search tool, so every
//!   repo-wide search for anything declared here came back short without saying so.
//! * **git treated it as binary.** `git diff --numstat` reported `-  -`; the file had never had a
//!   line-level diff and could not be reviewed by one.
//! * **`file(1)` reported `data`**, not text.
//!
//! One invisible byte removed a file from the code graph, the search tool and the diff at once,
//! and every one of those tools reported the resulting silence exactly as it reports a genuine
//! absence. `CLAUDE.md`: *a check that could not run must never report the same result as a check
//! that ran and passed.* That is the defect class, and this test is the gate on the class rather
//! than on the one byte — `GAP-RT-SOURCE-NUL-BYTE-UNGATED`.
//!
//! # Why the rule is "no tracked file at all" and has no allowlist
//!
//! Measured at the time of writing: all 243 tracked files are text, and the repository tracks no
//! binary of any kind. Large generated artefacts — the DevMap index, checkpoints, `.safetensors`,
//! `.npy.gz` — are all gitignored on purpose (`.gitignore`), because they are derived state. So an
//! allowlist would have no entries and would only be a hole for the next NUL to arrive through.
//! If this repository ever legitimately needs to track a binary, this test fails and names it, and
//! that is a decision a human should make deliberately rather than one a skipped check makes
//! silently.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `crates/qd-runtime/tests/..` -> the repository root.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .expect("the crate lives two directories below the repository root")
}

/// Largest file this gate will read whole.
///
/// Bounded because an unbounded read is an unbounded read. It is not a sampling cap: a file past
/// this size is a **failure** that names the file, never a file that was partly scanned and
/// reported clean. The largest tracked file today is `uv.lock` at roughly 310 KiB.
const MAX_SCAN_BYTES: u64 = 8 * 1024 * 1024;

/// Every path `git` currently tracks, as raw bytes (paths are not guaranteed UTF-8).
///
/// Any failure to enumerate panics. This gate cannot pass on an empty list: "git told us nothing"
/// and "nothing is wrong" are different facts and must not share an outcome.
fn tracked_paths(root: &Path) -> Vec<PathBuf> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .unwrap_or_else(|e| {
            panic!("could not run `git ls-files` in {}: {e}", root.display())
        });
    assert!(
        out.status.success(),
        "`git ls-files` failed ({}) in {}:\n{}",
        out.status,
        root.display(),
        String::from_utf8_lossy(&out.stderr)
    );

    let paths: Vec<PathBuf> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| {
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStrExt;
                PathBuf::from(std::ffi::OsStr::from_bytes(s))
            }
            #[cfg(not(unix))]
            {
                PathBuf::from(String::from_utf8_lossy(s).into_owned())
            }
        })
        .collect();

    assert!(
        !paths.is_empty(),
        "`git ls-files` returned no tracked files in {}. This gate scans what git tracks, so an \
         empty list means the enumeration failed, not that the repository is clean.",
        root.display()
    );
    paths
}

#[test]
fn no_tracked_source_file_contains_a_nul_byte() {
    let root = repo_root();
    let tracked = tracked_paths(&root);

    let mut scanned = 0usize;
    // Tracked but absent from the worktree: possible mid-rebase or with a sparse checkout. Counted
    // and reported, never folded into the scanned total.
    let mut absent: Vec<PathBuf> = Vec::new();
    let mut too_large: Vec<(PathBuf, u64)> = Vec::new();
    let mut offenders: Vec<String> = Vec::new();

    for rel in &tracked {
        let abs = root.join(rel);
        let meta = match std::fs::metadata(&abs) {
            Ok(m) => m,
            Err(_) => {
                absent.push(rel.clone());
                continue;
            }
        };
        if !meta.is_file() {
            absent.push(rel.clone());
            continue;
        }
        if meta.len() > MAX_SCAN_BYTES {
            too_large.push((rel.clone(), meta.len()));
            continue;
        }

        let bytes = std::fs::read(&abs)
            .unwrap_or_else(|e| panic!("could not read tracked file {}: {e}", abs.display()));
        scanned += 1;

        if let Some(offset) = bytes.iter().position(|b| *b == 0) {
            let line = bytes[..offset].iter().filter(|b| **b == b'\n').count() + 1;
            let count = bytes.iter().filter(|b| **b == 0).count();
            offenders.push(format!(
                "  {} — {count} NUL byte(s), first at offset {offset} (line {line})",
                rel.display()
            ));
        }
    }

    // Carry both numbers. A capped sample is never presented as complete coverage.
    println!(
        "scanned {scanned} of {} tracked files ({} absent from the worktree, {} over the {MAX_SCAN_BYTES}-byte scan cap)",
        tracked.len(),
        absent.len(),
        too_large.len()
    );

    assert!(
        too_large.is_empty(),
        "these tracked files are larger than this gate will read, so they were NOT checked for NUL \
         bytes. A file this gate cannot scan must not be reported as a file that passed it. Either \
         the file does not belong in git or {MAX_SCAN_BYTES} is the wrong bound — decide, do not \
         let it through:\n{}",
        too_large
            .iter()
            .map(|(p, n)| format!("  {} — {n} bytes", p.display()))
            .collect::<Vec<_>>()
            .join("\n")
    );

    assert!(
        offenders.is_empty(),
        "a tracked file contains a NUL byte, which makes it binary to DevMap, ripgrep, git diff \
         and file(1) all at once — each of which then reports its absence exactly as it reports a \
         genuine absence. If you meant to write an escape in a comment or a string, write the \
         characters (`\\0`, `\\u0000`), not the byte. GAP-RT-SOURCE-NUL-BYTE-UNGATED:\n{}",
        offenders.join("\n")
    );

    // The scan is only as good as its reach: if git names files this gate never opened, say so.
    assert_eq!(
        scanned + absent.len() + too_large.len(),
        tracked.len(),
        "every tracked path must be accounted for as scanned, absent or over-cap"
    );
}
