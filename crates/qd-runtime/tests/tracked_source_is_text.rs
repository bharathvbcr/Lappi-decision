//! Every tracked file in this repository is text, except binary test fixtures whose bytes are
//! pinned by sha256 in a tracked manifest. No other tracked file carries a NUL byte.
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
//! # The rule, as the human decided it on 2026-10-02: "Allow pinned fixtures only"
//!
//! Until 2026-10-02 the rule was "no tracked file at all, no allowlist" (its reasoning is kept
//! below). Then the qd-train lanes' commits of 2026-10-01 (L-head `adf6aea` first, then L-data
//! `3400063` and L-oracle `75ca2fb`, `fec0d69`, `f818333`, `fd5fe80`) started tracking 281 binary
//! test fixtures, all under `crates/qd-train/tests/fixtures/` (`.npy`, `.npz`, `.safetensors`,
//! `.u32`). Once merged, this gate failed on main and named every one of them, which is what it
//! was written to do (`GAP-QD-RUNTIME-NUL-BYTE-TEST-FAILS-ON-QD-TRAIN-BINARY-FIXTURES-2026-10-02`). The
//! decision it asked a human for was made: **allow pinned fixtures only**. A tracked file that
//! contains a NUL byte passes only if all three of these hold, and each is checked separately so a
//! failure names every condition it broke:
//!
//! 1. **Directory.** Its path is `crates/<crate>/tests/fixtures/<anything>`: four components
//!    exactly `crates`, one crate name, `tests`, `fixtures`, and then at least the file name.
//!    That `crates/<crate>/tests/fixtures` directory is the file's *fixture root*.
//! 2. **Extension.** Its extension is exactly one of [`FIXTURE_EXTENSIONS`]: `npy`, `npz`,
//!    `safetensors`, `u32` (lower case, compared as written).
//! 3. **Pin.** The sha256 of its bytes, as 64 lower-case hex digits, is recorded by an accepted
//!    manifest (next section) **under the same fixture root**. A manifest under another crate's
//!    root, or anywhere outside a fixture root, does not pin it.
//!
//! Every other tracked file must be NUL-free, as before.
//!
//! # Why the rule is a pin and not a skip
//!
//! A skip — by extension, by directory, or by a `.gitattributes` `binary` line — would answer "is
//! this file binary?", and every tool here already answers that. The reason this gate exists is
//! that a binary file cannot be reviewed line by line, so a change to it reaches main unseen. A
//! skip keeps that hole open and makes it permanent. A pin closes it. The bytes cannot be read in a
//! diff, but they can be checked whole: change one byte and the sha256 changes, the pin stops
//! matching, and this gate fails and names the file. So does the consumer's own check in qd-train
//! (`crates/qd-train/tests/common/pins.rs`, run by `tests/fixture_pins.rs` and before each
//! fixture set is used). A new binary fails until someone records its digest, so tracking a binary
//! stays the deliberate act the original rule asked for, and it is visible as a text diff to a
//! manifest. The directory and extension conditions keep the allowance away from source: a NUL in
//! a `.rs` file, or a `.npy` outside `tests/fixtures`, fails whatever its pin.
//!
//! # Accepted manifests
//!
//! A manifest is a tracked file under a fixture root that contains no NUL byte and is one of the
//! two forms below. A manifest that contains a NUL is itself binary. It pins nothing, and it is
//! classified like any other binary, so it fails, since neither manifest name is a fixture
//! extension.
//!
//! * **`SHA256SUMS`** (the file name exactly): coreutils `sha256sum` / `shasum -a 256` output.
//!   The grammar is strict. The file is UTF-8, has no `\r`, and its lines are separated by `\n`
//!   (a final newline is optional). It has at least one line and no blank line. Each line is
//!   `<digest><sep><path>`:
//!   - `<digest>` is 64 lower-case hex digits;
//!   - `<sep>` is two spaces (text mode) or a space and `*` (binary mode);
//!   - `<path>` is non-empty and relative. It has no `.` or `..` component and does not start
//!     with `/`. The line does not start with `\`, which is coreutils' escaped-name form and is
//!     not accepted.
//!
//!   The digest field is the pin. A line outside this grammar makes the whole manifest
//!   **unreadable**, which fails the gate. It is reported before any binary is classified, so it
//!   is not mistaken for a run of "unpinned" files.
//! * **`*.json`** (extension `json`): any JSON document. A pin is any JSON string at any depth,
//!   an object key or a string value, that is exactly 64 lower-case hex digits. A longer string
//!   that contains such a run is not a pin. A `.json` under a fixture root that does not parse is
//!   **unreadable** and fails the gate, because the gate cannot tell what it was meant to pin.
//!
//! The gate asks one question of a manifest: does it record this digest? The opposite question,
//! whether each path a `SHA256SUMS` lists hashes to its digest, belongs to the consumer:
//! `crates/qd-train/tests/common/pins.rs` parses the same grammar strictly. The two parsers are
//! separate on purpose, because sharing one would mean a cross-crate `#[path]` include, which this
//! repository has no precedent for. They still cannot disagree quietly: a line either side refuses
//! fails that side's test.
//!
//! # Pinning a fixture set, and re-pinning after a regeneration
//!
//! The `SHA256SUMS` files are written by `shasum`, not by code in this repository. That keeps the
//! digests a second implementation (Perl's `Digest::SHA`) independent of the Rust that checks
//! them, and it means no test can bless whatever bytes happen to be on disk. Run this from the
//! fixture set's directory (for example `crates/qd-train/tests/fixtures/span-head`), after the
//! regenerated files are tracked:
//!
//! ```text
//! git ls-files -z -- . ':!:SHA256SUMS' | xargs -0 shasum -a 256 > SHA256SUMS
//! ```
//!
//! and check it with `shasum -a 256 -c SHA256SUMS`. Both oracles refuse or remove the pin on a
//! regeneration, and that is intended. `tools/qd_train_oracle_span_head.py` refuses an `--out`
//! directory that holds anything but its cases, so delete `span-head/SHA256SUMS` first.
//! `tools/qd_train_oracle_shards.py --replace` deletes `shards-tiny/` along with its pin. Either
//! way the gate fails (unpinned) until the command above has been run, and nothing keeps a stale
//! pin alive.
//!
//! # The rule before 2026-10-02, as it was written
//!
//! > Measured at the time of writing: all 243 tracked files are text, and the repository tracks no
//! > binary of any kind. Large generated artefacts — the DevMap index, checkpoints,
//! > `.safetensors`, `.npy.gz` — are all gitignored on purpose (`.gitignore`), because they are
//! > derived state. So an allowlist would have no entries and would only be a hole for the next
//! > NUL to arrive through. If this repository ever legitimately needs to track a binary, this test
//! > fails and names it, and that is a decision a human should make deliberately rather than one a
//! > skipped check makes silently.
//!
//! The rule is still not an allowlist. No file is let through by name. A binary is let through
//! only by having its exact bytes on record, under its own fixture root.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

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
/// reported clean. The largest tracked files today are two of
/// `crates/qd-train/tests/fixtures/tiny-published/*/final.safetensors`, at 1,491,616 bytes each.
const MAX_SCAN_BYTES: u64 = 8 * 1024 * 1024;

/// The only extensions a binary fixture may have (condition 2 in the module docs).
const FIXTURE_EXTENSIONS: [&str; 4] = ["npy", "npz", "safetensors", "u32"];

/// The coreutils manifest's file name.
const SHA256SUMS: &str = "SHA256SUMS";

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
        .unwrap_or_else(|e| panic!("could not run `git ls-files` in {}: {e}", root.display()));
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

/// Lower-case hex sha256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Exactly 64 lower-case hex digits: the only spelling of a digest either manifest form accepts.
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// `crates/<crate>/tests/fixtures` when `rel` lies beneath one (condition 1), else `None`.
///
/// Only plain components count: a path with `.`, `..` or a root is never under a fixture root.
fn fixture_root(rel: &Path) -> Option<PathBuf> {
    let parts: Vec<&OsStr> = rel
        .components()
        .map(|c| match c {
            Component::Normal(s) => Some(s),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    match parts.as_slice() {
        [crates, krate, tests, fixtures, _, ..]
            if *crates == "crates" && *tests == "tests" && *fixtures == "fixtures" =>
        {
            Some([crates, krate, tests, fixtures].iter().collect())
        }
        _ => None,
    }
}

/// Which manifest form `rel` is, by name alone (the content is judged by [`PinIndex::offer`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManifestForm {
    Sha256Sums,
    Json,
}

fn manifest_form(rel: &Path) -> Option<ManifestForm> {
    if rel.file_name() == Some(OsStr::new(SHA256SUMS)) {
        Some(ManifestForm::Sha256Sums)
    } else if rel.extension() == Some(OsStr::new("json")) {
        Some(ManifestForm::Json)
    } else {
        None
    }
}

/// The digests a `SHA256SUMS` records, or why it is not one (the grammar in the module docs).
fn parse_sha256sums(bytes: &[u8]) -> Result<Vec<String>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("not UTF-8: {e}"))?;
    if text.contains('\r') {
        return Err("contains `\\r`; lines are separated by `\\n` alone".into());
    }
    let body = text.strip_suffix('\n').unwrap_or(text);
    if body.is_empty() {
        return Err("has no lines, so it pins nothing".into());
    }
    let mut digests = Vec::new();
    for (i, line) in body.split('\n').enumerate() {
        let n = i + 1;
        if line.is_empty() {
            return Err(format!("line {n} is blank"));
        }
        if line.starts_with('\\') {
            return Err(format!(
                "line {n} is coreutils' escaped-name form (leading `\\`), which is not accepted"
            ));
        }
        let (digest, rest) = line
            .split_at_checked(64)
            .ok_or_else(|| format!("line {n} is shorter than a 64-digit digest: {line:?}"))?;
        if !is_sha256_hex(digest) {
            return Err(format!(
                "line {n}: {digest:?} is not 64 lower-case hex digits"
            ));
        }
        let path = rest
            .strip_prefix("  ")
            .or_else(|| rest.strip_prefix(" *"))
            .ok_or_else(|| {
                format!("line {n}: the digest is not followed by two spaces or ` *`: {line:?}")
            })?;
        if path.is_empty() {
            return Err(format!("line {n} names no path"));
        }
        let p = Path::new(path);
        if p.is_absolute() || !p.components().all(|c| matches!(c, Component::Normal(_))) {
            return Err(format!(
                "line {n}: path {path:?} must be relative with no `.` or `..` component"
            ));
        }
        digests.push(digest.to_owned());
    }
    Ok(digests)
}

/// Every JSON string, key or value, at any depth, that is exactly a digest.
fn json_digests(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) if is_sha256_hex(s) => out.push(s.clone()),
        serde_json::Value::Array(items) => items.iter().for_each(|v| json_digests(v, out)),
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if is_sha256_hex(k) {
                    out.push(k.clone());
                }
                json_digests(v, out);
            }
        }
        _ => {}
    }
}

/// What the tracked manifests record: digest -> every manifest that records it.
#[derive(Debug, Default)]
struct PinIndex {
    by_digest: BTreeMap<String, BTreeSet<PathBuf>>,
    sha256sums_read: usize,
    json_read: usize,
    /// Manifest candidates that could not be read, each with why. Non-empty fails the gate.
    unreadable: Vec<String>,
}

impl PinIndex {
    /// Offer one tracked file. Anything but a NUL-free manifest under a fixture root is ignored:
    /// a manifest elsewhere is not a manifest, and a manifest with a NUL is a binary.
    fn offer(&mut self, rel: &Path, bytes: &[u8]) {
        if fixture_root(rel).is_none() || bytes.contains(&0) {
            return;
        }
        let digests = match manifest_form(rel) {
            None => return,
            Some(ManifestForm::Sha256Sums) => match parse_sha256sums(bytes) {
                Ok(d) => {
                    self.sha256sums_read += 1;
                    d
                }
                Err(why) => {
                    self.unreadable.push(format!("  {} — {why}", rel.display()));
                    return;
                }
            },
            Some(ManifestForm::Json) => match serde_json::from_slice::<serde_json::Value>(bytes) {
                Ok(v) => {
                    self.json_read += 1;
                    let mut d = Vec::new();
                    json_digests(&v, &mut d);
                    d
                }
                Err(e) => {
                    self.unreadable
                        .push(format!("  {} — does not parse as JSON: {e}", rel.display()));
                    return;
                }
            },
        };
        for d in digests {
            self.by_digest
                .entry(d)
                .or_default()
                .insert(rel.to_path_buf());
        }
    }

    fn pin_count(&self) -> usize {
        self.by_digest.values().map(BTreeSet::len).sum()
    }
}

/// One condition a NUL-bearing file failed (the module docs' three, in order).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Failed {
    /// Not under `crates/<crate>/tests/fixtures/`.
    Directory,
    /// The extension, if any, is not one of [`FIXTURE_EXTENSIONS`].
    Extension(Option<String>),
    /// No manifest under the file's own fixture root records its digest. `elsewhere` names the
    /// manifests under *other* fixture roots that do, so a misplaced pin is visible as one.
    Unpinned { elsewhere: Vec<PathBuf> },
}

impl std::fmt::Display for Failed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failed::Directory => write!(f, "not under crates/<crate>/tests/fixtures/"),
            Failed::Extension(None) => {
                write!(
                    f,
                    "has no extension; a fixture's is one of {FIXTURE_EXTENSIONS:?}"
                )
            }
            Failed::Extension(Some(ext)) => {
                write!(f, "extension `.{ext}` is not one of {FIXTURE_EXTENSIONS:?}")
            }
            Failed::Unpinned { elsewhere } if elsewhere.is_empty() => {
                write!(f, "no manifest under its fixture root records its sha256")
            }
            Failed::Unpinned { elsewhere } => write!(
                f,
                "no manifest under its fixture root records its sha256; only these do, and each \
                 pins files under its own fixture root, not this path: {}",
                elsewhere
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// The classification, pure: a tracked file that contains a NUL byte, by its repository-relative
/// path and the lower-case hex sha256 of its bytes. `Ok` is the manifest that pins it under its
/// own fixture root; `Err` is every condition it failed, never only the first.
fn classify_binary(rel: &Path, sha256: &str, pins: &PinIndex) -> Result<PathBuf, Vec<Failed>> {
    let root = fixture_root(rel);
    let mut failed = Vec::new();
    if root.is_none() {
        failed.push(Failed::Directory);
    }
    let ext = rel.extension().map(|e| e.to_string_lossy().into_owned());
    if !ext
        .as_deref()
        .is_some_and(|e| FIXTURE_EXTENSIONS.contains(&e))
    {
        failed.push(Failed::Extension(ext));
    }
    let recorders = pins.by_digest.get(sha256);
    let mine = recorders.and_then(|ms| {
        ms.iter()
            .find(|m| root.is_some() && fixture_root(m) == root)
            .cloned()
    });
    if mine.is_none() {
        let elsewhere = recorders
            .map(|ms| ms.iter().cloned().collect())
            .unwrap_or_default();
        failed.push(Failed::Unpinned { elsewhere });
    }
    match mine {
        Some(manifest) if failed.is_empty() => Ok(manifest),
        _ => Err(failed),
    }
}

/// A tracked file that contains a NUL byte.
struct Binary {
    rel: PathBuf,
    sha256: String,
    nul_count: usize,
    first_offset: usize,
    first_line: usize,
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
    let mut binaries: Vec<Binary> = Vec::new();
    let mut pins = PinIndex::default();

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
        pins.offer(rel, &bytes);

        if let Some(offset) = bytes.iter().position(|b| *b == 0) {
            binaries.push(Binary {
                rel: rel.clone(),
                sha256: sha256_hex(&bytes),
                nul_count: bytes.iter().filter(|b| **b == 0).count(),
                first_offset: offset,
                first_line: bytes[..offset].iter().filter(|b| **b == b'\n').count() + 1,
            });
        }
    }

    let mut allowed = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    for b in &binaries {
        match classify_binary(&b.rel, &b.sha256, &pins) {
            Ok(_) => allowed += 1,
            Err(failed) => offenders.push(format!(
                "  {} — {} NUL byte(s), first at offset {} (line {}), sha256 {}:\n{}",
                b.rel.display(),
                b.nul_count,
                b.first_offset,
                b.first_line,
                b.sha256,
                failed
                    .iter()
                    .map(|f| format!("      fails: {f}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )),
        }
    }

    // Carry both numbers. A capped sample is never presented as complete coverage.
    println!(
        "scanned {scanned} of {} tracked files ({} absent from the worktree, {} over the \
         {MAX_SCAN_BYTES}-byte scan cap)",
        tracked.len(),
        absent.len(),
        too_large.len()
    );
    println!(
        "{} contain a NUL byte: {allowed} binary-allowed (pinned fixtures), {} offenders",
        binaries.len(),
        offenders.len()
    );
    println!(
        "manifests read under fixture roots: {} SHA256SUMS, {} JSON, {} unreadable; {} pins",
        pins.sha256sums_read,
        pins.json_read,
        pins.unreadable.len(),
        pins.pin_count()
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

    // Before the offenders: an unreadable manifest would otherwise surface as a run of "unpinned"
    // fixtures and send the reader after the wrong thing.
    assert!(
        pins.unreadable.is_empty(),
        "these manifests under a fixture root could not be read, so this gate cannot tell what \
         they pin. A manifest it cannot read must not be reported as one that pinned nothing. The \
         accepted forms are in this file's module docs:\n{}",
        pins.unreadable.join("\n")
    );

    assert!(
        offenders.is_empty(),
        "a tracked file contains a NUL byte and is not a pinned fixture. A NUL makes a file binary \
         to DevMap, ripgrep, git diff and file(1) all at once — each of which then reports its \
         absence exactly as it reports a genuine absence. If you meant to write an escape in a \
         comment or a string, write the characters (`\\0`, `\\u0000`), not the byte. If it is a \
         binary test fixture, it must be a .npy/.npz/.safetensors/.u32 under \
         crates/<crate>/tests/fixtures/ and its sha256 must be in a manifest under that root \
         (see the module docs for the forms and the one-line `shasum` command). \
         GAP-RT-SOURCE-NUL-BYTE-UNGATED:\n{}",
        offenders.join("\n")
    );

    // The scan is only as good as its reach: if git names files this gate never opened, say so.
    assert_eq!(
        scanned + absent.len() + too_large.len(),
        tracked.len(),
        "every tracked path must be accounted for as scanned, absent or over-cap"
    );
    assert_eq!(
        allowed + offenders.len(),
        binaries.len(),
        "every file with a NUL byte must be accounted for as binary-allowed or an offender"
    );
}

// -------------------------------------------------------------------------------------------------
// The classification and the manifest grammar, on in-memory files.
// -------------------------------------------------------------------------------------------------

/// An index over `(path, bytes)` files, as the gate builds one.
fn index(files: &[(&str, &[u8])]) -> PinIndex {
    let mut pins = PinIndex::default();
    for (rel, bytes) in files {
        pins.offer(Path::new(rel), bytes);
    }
    pins
}

/// A fixture's bytes: a NUL-bearing npy header and a payload.
const NPY: &[u8] = b"\x93NUMPY\x01\x00v\x00{'descr': '<f4'}\x00\x00\x80\x3f";

fn sums_line(bytes: &[u8], path: &str) -> String {
    format!("{}  {path}\n", sha256_hex(bytes))
}

#[test]
fn a_fixture_pinned_by_a_sha256sums_under_its_own_root_is_allowed() {
    let sums = sums_line(NPY, "case/a.npy");
    let pins = index(&[(
        "crates/qd-train/tests/fixtures/set/SHA256SUMS",
        sums.as_bytes(),
    )]);
    for (rel, ext) in [
        ("crates/qd-train/tests/fixtures/set/case/a.npy", "npy"),
        // Directly under the root counts: the root's only requirement is the file name after it.
        ("crates/qd-train/tests/fixtures/a.npz", "npz"),
        (
            "crates/qd-train/tests/fixtures/x/y/z/w.safetensors",
            "safetensors",
        ),
        ("crates/qd-train/tests/fixtures/set/tokens.u32", "u32"),
    ] {
        assert_eq!(
            classify_binary(Path::new(rel), &sha256_hex(NPY), &pins),
            Ok(PathBuf::from(
                "crates/qd-train/tests/fixtures/set/SHA256SUMS"
            )),
            "{rel} (.{ext})"
        );
    }
    // Binary mode (` *`) is the same pin.
    let binary_mode = format!("{} *a.npy\n", sha256_hex(NPY));
    let pins = index(&[(
        "crates/qd-train/tests/fixtures/SHA256SUMS",
        binary_mode.as_bytes(),
    )]);
    assert!(
        classify_binary(
            Path::new("crates/qd-train/tests/fixtures/a.npy"),
            &sha256_hex(NPY),
            &pins
        )
        .is_ok()
    );
}

#[test]
fn a_fixture_pinned_by_a_json_string_under_its_own_root_is_allowed() {
    let manifest = format!(
        r#"{{"files": {{"a.npy": {{"bytes": 22, "sha256": "{}"}}}}}}"#,
        sha256_hex(NPY)
    );
    let pins = index(&[(
        "crates/qd-train/tests/fixtures/set/manifest.json",
        manifest.as_bytes(),
    )]);
    assert_eq!(
        classify_binary(
            Path::new("crates/qd-train/tests/fixtures/set/a.npy"),
            &sha256_hex(NPY),
            &pins
        ),
        Ok(PathBuf::from(
            "crates/qd-train/tests/fixtures/set/manifest.json"
        ))
    );
    // An object key counts too, at any depth.
    let keyed = format!(r#"[{{"by_digest": {{"{}": "a.npy"}}}}]"#, sha256_hex(NPY));
    let pins = index(&[("crates/qd-train/tests/fixtures/k.json", keyed.as_bytes())]);
    assert!(
        classify_binary(
            Path::new("crates/qd-train/tests/fixtures/set/a.npy"),
            &sha256_hex(NPY),
            &pins
        )
        .is_ok()
    );
}

#[test]
fn a_binary_outside_tests_fixtures_fails_on_its_directory_whatever_its_pin() {
    let sums = sums_line(NPY, "a.npy");
    let pins = index(&[("crates/qd-train/tests/fixtures/SHA256SUMS", sums.as_bytes())]);
    for rel in [
        "crates/qd-train/src/a.npy",
        "crates/qd-train/tests/a.npy",
        "crates/qd-train/fixtures/a.npy",
        "crates/qd-train/tests/fixture/a.npy",
        "crates/a/b/tests/fixtures/a.npy",
        "tests/fixtures/a.npy",
        "tools/a.npy",
        "a.npy",
        // The fixture root itself as a file name: nothing comes after `fixtures`.
        "crates/qd-train/tests/fixtures",
    ] {
        let got = classify_binary(Path::new(rel), &sha256_hex(NPY), &pins);
        let failed = got.expect_err(rel);
        assert_eq!(failed[0], Failed::Directory, "{rel}: {failed:?}");
        // No fixture root, so no manifest can be its own: the pin fails too, and says the pin
        // it found is under another root.
        assert!(
            failed.contains(&Failed::Unpinned {
                elsewhere: vec![PathBuf::from("crates/qd-train/tests/fixtures/SHA256SUMS")]
            }),
            "{rel}: {failed:?}"
        );
    }
}

#[test]
fn a_pinned_binary_under_fixtures_with_another_extension_fails_on_its_extension_only() {
    let sums = sums_line(NPY, "a");
    let pins = index(&[("crates/qd-train/tests/fixtures/SHA256SUMS", sums.as_bytes())]);
    for (rel, ext) in [
        ("crates/qd-train/tests/fixtures/a.rs", Some("rs")),
        ("crates/qd-train/tests/fixtures/a.bin", Some("bin")),
        ("crates/qd-train/tests/fixtures/a.NPY", Some("NPY")),
        ("crates/qd-train/tests/fixtures/a.npy.gz", Some("gz")),
        ("crates/qd-train/tests/fixtures/a.json", Some("json")),
        ("crates/qd-train/tests/fixtures/a", None),
    ] {
        assert_eq!(
            classify_binary(Path::new(rel), &sha256_hex(NPY), &pins),
            Err(vec![Failed::Extension(ext.map(str::to_owned))]),
            "{rel}"
        );
    }
}

#[test]
fn an_unpinned_fixture_fails_on_its_pin_only() {
    // A manifest under the root that records a different digest; and one that records this one
    // as a substring of a longer string, in upper case, and as a number-like value, none of which
    // is a pin.
    let other = sums_line(b"other bytes", "a.npy");
    let digest = sha256_hex(NPY);
    let near_misses = format!(
        r#"{{"longer": "{digest}0", "prefixed": "sha256:{digest}", "upper": "{}"}}"#,
        digest.to_uppercase()
    );
    let pins = index(&[
        (
            "crates/qd-train/tests/fixtures/SHA256SUMS",
            other.as_bytes(),
        ),
        (
            "crates/qd-train/tests/fixtures/near.json",
            near_misses.as_bytes(),
        ),
    ]);
    assert_eq!(
        classify_binary(
            Path::new("crates/qd-train/tests/fixtures/a.npy"),
            &digest,
            &pins
        ),
        Err(vec![Failed::Unpinned { elsewhere: vec![] }])
    );
    // And with no manifests at all.
    assert_eq!(
        classify_binary(
            Path::new("crates/qd-train/tests/fixtures/a.npy"),
            &digest,
            &PinIndex::default()
        ),
        Err(vec![Failed::Unpinned { elsewhere: vec![] }])
    );
}

#[test]
fn a_pin_in_a_manifest_outside_the_files_own_fixture_root_does_not_pin_it() {
    let sums = sums_line(NPY, "a.npy");
    let json = format!(r#"{{"sha256": "{}"}}"#, sha256_hex(NPY));
    // Under another crate's fixture root: indexed, but it pins only that root's files.
    let pins = index(&[(
        "crates/qd-export/tests/fixtures/SHA256SUMS",
        sums.as_bytes(),
    )]);
    assert_eq!(
        classify_binary(
            Path::new("crates/qd-train/tests/fixtures/a.npy"),
            &sha256_hex(NPY),
            &pins
        ),
        Err(vec![Failed::Unpinned {
            elsewhere: vec![PathBuf::from("crates/qd-export/tests/fixtures/SHA256SUMS")]
        }])
    );
    // Outside any fixture root (repository root, a crate's tests/, a sibling of fixtures/): not a
    // manifest at all, so not even indexed.
    for rel in [
        "SHA256SUMS",
        "crates/qd-train/tests/SHA256SUMS",
        "crates/qd-train/SHA256SUMS",
        "crates/qd-train/tests/manifest.json",
        "AUDIT/pins.json",
    ] {
        let bytes = if rel.ends_with(".json") {
            json.as_bytes()
        } else {
            sums.as_bytes()
        };
        let pins = index(&[(rel, bytes)]);
        assert_eq!(pins.pin_count(), 0, "{rel} must not be indexed");
        assert_eq!(
            classify_binary(
                Path::new("crates/qd-train/tests/fixtures/a.npy"),
                &sha256_hex(NPY),
                &pins
            ),
            Err(vec![Failed::Unpinned { elsewhere: vec![] }]),
            "{rel}"
        );
    }
}

#[test]
fn a_manifest_that_is_itself_binary_pins_nothing_and_fails_as_a_binary() {
    // A well-formed line, then a NUL: the manifest is binary, so its line is not a pin.
    let mut sums = sums_line(NPY, "a.npy").into_bytes();
    sums.extend_from_slice(b"\x00");
    let manifest = "crates/qd-train/tests/fixtures/SHA256SUMS";
    let pins = index(&[(manifest, &sums)]);
    assert_eq!(pins.pin_count(), 0);
    assert!(
        pins.unreadable.is_empty(),
        "a binary is not an unreadable manifest; it is a binary"
    );
    assert_eq!(
        classify_binary(
            Path::new("crates/qd-train/tests/fixtures/a.npy"),
            &sha256_hex(NPY),
            &pins
        ),
        Err(vec![Failed::Unpinned { elsewhere: vec![] }])
    );
    // The manifest itself is then classified like any binary, and it cannot pass: neither
    // manifest name carries a fixture extension.
    let failed = classify_binary(Path::new(manifest), &sha256_hex(&sums), &pins).unwrap_err();
    assert!(failed.contains(&Failed::Extension(None)), "{failed:?}");
    let mut json = format!(r#"{{"sha256": "{}"}}"#, sha256_hex(NPY)).into_bytes();
    json.push(0);
    let pins = index(&[("crates/qd-train/tests/fixtures/m.json", &json)]);
    assert_eq!(pins.pin_count(), 0);
    let failed = classify_binary(
        Path::new("crates/qd-train/tests/fixtures/m.json"),
        &sha256_hex(&json),
        &pins,
    )
    .unwrap_err();
    assert!(
        failed.contains(&Failed::Extension(Some("json".into()))),
        "{failed:?}"
    );
}

#[test]
fn a_sha256sums_outside_the_grammar_is_unreadable_not_empty() {
    let d = sha256_hex(NPY);
    for (why, body) in [
        ("one space", format!("{d} a.npy\n")),
        ("tab", format!("{d}\ta.npy\n")),
        ("upper case", format!("{}  a.npy\n", d.to_uppercase())),
        ("short digest", format!("{}  a.npy\n", &d[..63])),
        ("no path", format!("{d}  \n")),
        ("parent path", format!("{d}  ../a.npy\n")),
        ("dot path", format!("{d}  ./a.npy\n")),
        ("absolute path", format!("{d}  /a.npy\n")),
        ("escaped name", format!("\\{d}  a\\nb.npy\n")),
        ("crlf", format!("{d}  a.npy\r\n")),
        ("blank line", format!("{d}  a.npy\n\n{d}  b.npy\n")),
        ("empty", String::new()),
        ("only a newline", "\n".to_owned()),
        ("a bsd-style line", format!("SHA256 (a.npy) = {d}\n")),
    ] {
        let pins = index(&[("crates/qd-train/tests/fixtures/SHA256SUMS", body.as_bytes())]);
        assert_eq!(pins.unreadable.len(), 1, "{why}: {body:?} must be refused");
        assert_eq!(
            pins.pin_count(),
            0,
            "{why}: a refused manifest pins nothing"
        );
        assert_eq!(pins.sha256sums_read, 0, "{why}");
    }
    // A line that is fine after a bad one does not rescue the manifest.
    let pins = index(&[(
        "crates/qd-train/tests/fixtures/SHA256SUMS",
        format!("{d}  a.npy\n{d} b.npy\n").as_bytes(),
    )]);
    assert_eq!((pins.unreadable.len(), pins.pin_count()), (1, 0));
    // Not UTF-8.
    let pins = index(&[("crates/qd-train/tests/fixtures/SHA256SUMS", b"\xff\xfe")]);
    assert_eq!(pins.unreadable.len(), 1);
    // And the accepted forms, for contrast: no final newline, several lines, both modes.
    let pins = index(&[(
        "crates/qd-train/tests/fixtures/SHA256SUMS",
        format!("{d}  a.npy\n{}  sub/b.npy\n{d} *c.npy", sha256_hex(b"b")).as_bytes(),
    )]);
    assert!(pins.unreadable.is_empty(), "{:?}", pins.unreadable);
    assert_eq!((pins.sha256sums_read, pins.pin_count()), (1, 2));
}

#[test]
fn a_json_manifest_that_does_not_parse_is_unreadable_not_empty() {
    let pins = index(&[("crates/qd-train/tests/fixtures/m.json", br#"{"sha256": "#)]);
    assert_eq!(
        (pins.unreadable.len(), pins.json_read, pins.pin_count()),
        (1, 0, 0)
    );
    // A JSON file that parses and records no digest is read, and pins nothing: fine.
    let pins = index(&[(
        "crates/qd-train/tests/fixtures/m.json",
        br#"{"a": [1, "b"]}"#,
    )]);
    assert_eq!(
        (pins.unreadable.len(), pins.json_read, pins.pin_count()),
        (0, 1, 0)
    );
}

#[test]
fn every_condition_a_binary_breaks_is_named_not_only_the_first() {
    // A binary `.rs` in `src/` with no pin anywhere: all three.
    assert_eq!(
        classify_binary(
            Path::new("crates/qd-runtime/src/wire.rs"),
            &sha256_hex(NPY),
            &PinIndex::default()
        ),
        Err(vec![
            Failed::Directory,
            Failed::Extension(Some("rs".into())),
            Failed::Unpinned { elsewhere: vec![] },
        ])
    );
}

#[test]
fn the_fixture_root_is_exactly_crates_crate_tests_fixtures() {
    assert_eq!(
        fixture_root(Path::new(
            "crates/qd-train/tests/fixtures/span-head/a/b.npy"
        )),
        Some(PathBuf::from("crates/qd-train/tests/fixtures"))
    );
    for rel in [
        "crates/qd-train/tests/fixtures",
        "crates/qd-train/tests/fixtures/../../src/a.npy",
        "./crates/qd-train/tests/fixtures/a.npy",
        "/crates/qd-train/tests/fixtures/a.npy",
        "Crates/qd-train/tests/fixtures/a.npy",
        "crates/qd-train/Tests/fixtures/a.npy",
    ] {
        assert_eq!(fixture_root(Path::new(rel)), None, "{rel}");
    }
}
