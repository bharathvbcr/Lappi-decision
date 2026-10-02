//! qd-metal's ledger rows: `ledger/mac-qd-metal-*.jsonl`, in the schema `docs/ledger-schema.md`
//! states and `python/qd_train/ledger.py` reads.
//!
//! The row and the chained append are qd-train's ([`qd_train::ledger::Row`],
//! [`qd_train::ledger::append_line`]): the lock, the duplicate-id refusal, the chain and the fsync
//! have one owner. This module adds what is qd-metal's:
//!
//! * **where** its rows may go ([`check_ledger_path`]): `ledger/mac-qd-metal-*.jsonl` only;
//! * **what ran** ([`Provenance`]): the sha256 of the executable that produced the numbers (it
//!   links qd-metal, qd-runtime and tessl, so it pins all three), this checkout's commit, and
//!   canonical tessl's `HEAD` plus the sha256 of every file `git status --porcelain` lists there.
//!   tessl is shared with other sessions and changes under a running lane; a row that named only
//!   its `HEAD` would not say which kernels ran.
//!
//! `metrics.code_that_ran` carries the executable's sha256. `python/tests/test_ledger_provenance.py`
//! requires a non-empty one on every non-`build` row, and for a Rust binary the executable is the
//! closure of what ran.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use qd_train::ledger::{self as qledger, Row, Stamp};
use qd_train::tristate::TriState;

use crate::error::{MetalError, Result};

/// The file-name prefix every qd-metal ledger carries.
pub const LEDGER_PREFIX: &str = "mac-qd-metal-";

fn ledger_err(e: qledger::LedgerError) -> MetalError {
    MetalError::Ledger(e.to_string())
}

/// qd-metal writes only `ledger/mac-qd-metal-*.jsonl`: never a campaign ledger, never the
/// trainer's `mac-ojas-*`, never `runs.jsonl`.
pub fn check_ledger_path(path: &Path) -> Result<()> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if !(name.starts_with(LEDGER_PREFIX)
        && name.ends_with(".jsonl")
        && name.len() > LEDGER_PREFIX.len() + ".jsonl".len())
    {
        return Err(MetalError::Ledger(format!(
            "{}: qd-metal writes only ledger/{LEDGER_PREFIX}*.jsonl",
            path.display()
        )));
    }
    if parent != "ledger" {
        return Err(MetalError::Ledger(format!(
            "{}: the file must sit in a directory named `ledger`",
            path.display()
        )));
    }
    Ok(())
}

/// Append `row` to `path` with a fresh uuid4 and the current time; returns the stamp written.
pub fn write_row(path: &Path, row: &Row) -> Result<Stamp> {
    check_ledger_path(path)?;
    let row_id = qledger::uuid4().map_err(ledger_err)?;
    let written_at = qledger::isoformat_utc(SystemTime::now()).map_err(ledger_err)?;
    qledger::append_line(path, row_id, written_at, |stamp| row.line(stamp)).map_err(ledger_err)
}

/// The checkout this binary was built from (`crates/qd-metal/../..`).
pub fn build_repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The tessl checkout qd-metal's path dependency resolves to (`crates/qd-metal/../../../tessl`).
pub fn tessl_dir() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("tessl");
    std::fs::canonicalize(&p).unwrap_or(p)
}

fn git(dir: &Path, args: &[&str]) -> std::result::Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "git {} exited {}: {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A git checkout's state as far as it changes what compiles: `HEAD`, and every path
/// `git status --porcelain` lists with the sha256 of its bytes.
///
/// An untracked directory (porcelain lists it once, `?? dir/`) is hashed as a tree: sha256 over
/// its files' relative paths and sha256s, in order. Except a cargo build directory (a top-level
/// name starting with `target`): build output is not source, it changes whenever another session
/// builds, and hashing it would make every A/B look like it ran across a tessl change. It is
/// listed as [`BUILD_OUTPUT`] and not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeState {
    pub dir: PathBuf,
    pub head: String,
    /// `"XY path" -> sha256 | "tree:<sha256>" | "deleted" |` [`BUILD_OUTPUT`], sorted.
    pub dirty: BTreeMap<String, String>,
    /// Unix seconds of the newest change this state reflects: `HEAD`'s commit time or a dirty
    /// file's mtime, whichever is later. Compared with the executable's mtime, it says whether
    /// the tree read now could be the tree the executable was built from.
    pub latest_change_s: u64,
}

/// What an untracked cargo build directory is recorded as, instead of a hash.
pub const BUILD_OUTPUT: &str = "build-output: not hashed";

/// Bounds on hashing one untracked directory. Past either, the entry says so instead of a hash.
const TREE_MAX_FILES: usize = 20_000;
const TREE_MAX_BYTES: u64 = 1 << 30;

fn mtime_s(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

/// A top-level untracked directory named like a cargo target dir (`target`, `target-redteam`).
fn is_build_output(path: &str) -> bool {
    let first = path.trim_end_matches('/').split('/').next().unwrap_or("");
    first.starts_with("target")
}

/// sha256 over `relpath \t sha256 \n` for every file under `root`, sorted; symlinks by their
/// target. Returns `(entry, newest mtime)`, or an "over the bound" entry.
fn hash_tree(root: &Path) -> Result<(String, u64)> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let entries = std::fs::read_dir(&d).map_err(|e| MetalError::Ledger(format!("{}: {e}", d.display())))?;
        for e in entries {
            let e = e.map_err(|e| MetalError::Ledger(format!("{}: {e}", d.display())))?;
            let ft = e.file_type().map_err(|err| MetalError::Ledger(format!("{}: {err}", e.path().display())))?;
            if ft.is_dir() {
                stack.push(e.path());
            } else {
                files.push(e.path());
            }
            if files.len() > TREE_MAX_FILES {
                return Ok((format!("tree over the hashing bound ({TREE_MAX_FILES} files)"), 0));
            }
        }
    }
    files.sort();
    let (mut acc, mut bytes, mut newest) = (String::new(), 0u64, 0u64);
    for f in &files {
        let rel = f.strip_prefix(root).unwrap_or(f).display().to_string();
        let meta = std::fs::symlink_metadata(f).map_err(|e| MetalError::Ledger(format!("{}: {e}", f.display())))?;
        newest = newest.max(mtime_s(f).unwrap_or(0));
        let sha = if meta.file_type().is_symlink() {
            let to = std::fs::read_link(f).map_err(|e| MetalError::Ledger(format!("{}: {e}", f.display())))?;
            format!("symlink:{}", to.display())
        } else {
            bytes += meta.len();
            if bytes > TREE_MAX_BYTES {
                return Ok((format!("tree over the hashing bound ({TREE_MAX_BYTES} bytes)"), newest));
            }
            let b = std::fs::read(f).map_err(|e| MetalError::Ledger(format!("{}: {e}", f.display())))?;
            qd_runtime::hex(&qd_runtime::sha256(&b))
        };
        acc.push_str(&format!("{rel}\t{sha}\n"));
    }
    Ok((format!("tree:{}", qd_runtime::hex(&qd_runtime::sha256(acc.as_bytes()))), newest))
}

impl TreeState {
    /// Read-only: `git rev-parse HEAD` and `git status --porcelain`, then the listed files' bytes.
    pub fn read(dir: &Path) -> Result<Self> {
        let head = git(dir, &["rev-parse", "HEAD"])
            .map_err(MetalError::Ledger)?
            .trim()
            .to_string();
        let status = git(dir, &["status", "--porcelain", "--untracked-files=normal"])
            .map_err(MetalError::Ledger)?;
        let commit_s: u64 = git(dir, &["log", "-1", "--format=%ct", "HEAD"])
            .map_err(MetalError::Ledger)?
            .trim()
            .parse()
            .map_err(|e| MetalError::Ledger(format!("{}: HEAD commit time: {e}", dir.display())))?;
        let mut latest_change_s = commit_s;
        let mut dirty = BTreeMap::new();
        for line in status.lines().filter(|l| l.len() > 3) {
            // `XY path` or `XY old -> new` for a rename; the new path is what compiles.
            let path = line[3..].rsplit(" -> ").next().unwrap_or(&line[3..]).trim_matches('"');
            let full = dir.join(path);
            let what = if full.is_dir() {
                if is_build_output(path) {
                    BUILD_OUTPUT.to_string()
                } else {
                    let (entry, newest) = hash_tree(&full)?;
                    latest_change_s = latest_change_s.max(newest);
                    entry
                }
            } else {
                if let Some(t) = mtime_s(&full) {
                    latest_change_s = latest_change_s.max(t);
                }
                match std::fs::read(&full) {
                    Ok(bytes) => qd_runtime::hex(&qd_runtime::sha256(&bytes)),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => "deleted".to_string(),
                    Err(e) => {
                        return Err(MetalError::Ledger(format!("{}: {e}", full.display())));
                    }
                }
            };
            dirty.insert(format!("{} {path}", &line[..2]), what);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            head,
            dirty,
            latest_change_s,
        })
    }

    /// sha256 over `HEAD` and every dirty entry, in order: equal iff the two states are.
    pub fn digest(&self) -> String {
        let mut acc = format!("{}\n", self.head);
        for (k, v) in &self.dirty {
            acc.push_str(&format!("{k}\t{v}\n"));
        }
        qd_runtime::hex(&qd_runtime::sha256(acc.as_bytes()))
    }

    /// One line a row can carry: `HEAD` and the dirty count. The entries themselves are
    /// [`TreeState::files_json`].
    pub fn describe(&self) -> String {
        let skipped = self.dirty.values().filter(|v| v.as_str() == BUILD_OUTPUT).count();
        format!(
            "{} HEAD {}; {} dirty path(s), {skipped} of them build output (not hashed)",
            self.dir.display(),
            self.head,
            self.dirty.len(),
        )
    }

    /// Every dirty entry with its full sha256 (or what stands for one), as a JSON object.
    pub fn files_json(&self) -> serde_json::Value {
        serde_json::Value::Object(
            self.dirty
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::from(v.as_str())))
                .collect(),
        )
    }
}

/// What produced a row's numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// sha256 of the executable's bytes.
    pub exe_sha256: String,
    pub exe_path: PathBuf,
    /// The executable's mtime, Unix seconds: when it was linked.
    pub exe_built_s: u64,
    /// `qd_train::ledger::git_commit` of the checkout this binary was built from.
    pub code_commit: String,
    pub tessl: TreeState,
    pub host: String,
}

impl Provenance {
    /// The provenance of `exe` (the running binary when `None`).
    pub fn of(exe: Option<&Path>) -> Result<Self> {
        let exe_path = match exe {
            Some(p) => p.to_path_buf(),
            None => std::env::current_exe()
                .map_err(|e| MetalError::Ledger(format!("current_exe: {e}")))?,
        };
        let bytes = std::fs::read(&exe_path)
            .map_err(|e| MetalError::Ledger(format!("{}: {e}", exe_path.display())))?;
        let exe_built_s = mtime_s(&exe_path)
            .ok_or_else(|| MetalError::Ledger(format!("{}: no mtime", exe_path.display())))?;
        Ok(Self {
            exe_sha256: qd_runtime::hex(&qd_runtime::sha256(&bytes)),
            exe_built_s,
            exe_path,
            code_commit: qledger::git_commit(&build_repo()),
            tessl: TreeState::read(&tessl_dir())?,
            host: qledger::hostname(),
        })
    }

    /// `metrics.code_that_ran`, `metrics.tessl_state` and `metrics.lappi_commit`.
    pub fn metrics(&self) -> BTreeMap<String, TriState> {
        let mut m = BTreeMap::new();
        m.insert(
            "code_that_ran".to_string(),
            TriState::passed(
                self.exe_sha256.as_str(),
                format!(
                    "sha256 of {}, the executable that produced these numbers; it links qd-metal, \
                     qd-runtime and tessl, so it pins all three",
                    self.exe_path.display()
                ),
            ),
        );
        m.insert(
            "tessl_state".to_string(),
            TriState::passed(self.tessl.digest(), self.tessl.describe()),
        );
        m.insert(
            "tessl_dirty_files".to_string(),
            TriState::passed(
                self.tessl.files_json(),
                "every path `git status --porcelain` lists in tessl, with the sha256 of its bytes \
                 (an untracked directory as a tree hash; cargo build output not hashed)",
            ),
        );
        // The state above is read when the binary runs. If tessl changed after the binary was
        // linked, the binary holds older kernels than the state names: the row says so.
        let predates = self.tessl.latest_change_s <= self.exe_built_s;
        m.insert(
            "tessl_state_predates_exe".to_string(),
            TriState::ran(predates, self.tessl.latest_change_s).with_detail(format!(
                "newest tessl change (HEAD commit time or a dirty file's mtime) at {} s; the \
                 executable was linked at {} s. Failed means tessl moved after the build, so \
                 tessl_state may not describe the kernels that ran",
                self.tessl.latest_change_s, self.exe_built_s
            )),
        );
        m.insert(
            "lappi_commit".to_string(),
            TriState::passed(
                self.code_commit.as_str(),
                "git HEAD of the checkout the executable was built from; `-dirty` when any path \
                 differs from it",
            ),
        );
        m
    }
}

/// `"<snapshot dir name>:vocab<vocab>"`, the `backbone_commit` the Python rows write.
pub fn backbone_commit(snapshot: &Path, vocab: usize) -> Result<String> {
    let name = snapshot
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| MetalError::Ledger(format!("{}: no directory name", snapshot.display())))?;
    Ok(format!("{name}:vocab{vocab}"))
}

/// sha256 over `json.dumps(recipe, sort_keys=True, separators=(",", ":"))`: the hash
/// [`Row::line`] checks `protocol.recipe_hash` against.
pub fn recipe_hash(recipe: &serde_json::Value) -> Result<String> {
    let body = qd_train::pyjson::dumps(recipe, qd_train::pyjson::CANONICAL_ASCII)
        .map_err(|e| MetalError::Ledger(e.to_string()))?;
    Ok(qd_runtime::hex(&qd_runtime::sha256(body.as_bytes())))
}

/// A JSON float, refused when not finite (a row never carries NaN as a measurement).
pub fn float(x: f64) -> Result<serde_json::Value> {
    qd_train::pyjson::float(x).map_err(|e| MetalError::Ledger(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_mac_qd_metal_ledgers_are_writable() {
        assert!(check_ledger_path(Path::new("/r/ledger/mac-qd-metal-2026-10-02.jsonl")).is_ok());
        assert!(check_ledger_path(Path::new("/r/ledger/mac-ojas-rung-b-2026-10-01.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/ledger/runs.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/ledger/gh200-p4-v4-2026-10-01.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/elsewhere/mac-qd-metal-x.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/ledger/mac-qd-metal-.jsonl")).is_err());
        assert!(check_ledger_path(Path::new("/r/ledger/mac-qd-metal-x.json")).is_err());
    }

    #[test]
    fn a_tree_digest_moves_with_head_and_with_each_dirty_file() {
        let base = TreeState {
            dir: PathBuf::from("/t"),
            head: "a".repeat(40),
            dirty: BTreeMap::from([(" M kernels/x.metal".to_string(), "1".repeat(64))]),
            latest_change_s: 1_790_000_000,
        };
        let mut other_head = base.clone();
        other_head.head = "b".repeat(40);
        let mut other_bytes = base.clone();
        other_bytes
            .dirty
            .insert(" M kernels/x.metal".to_string(), "2".repeat(64));
        let mut clean = base.clone();
        clean.dirty.clear();
        let d = base.digest();
        assert_eq!(d, base.clone().digest());
        assert_ne!(d, other_head.digest());
        assert_ne!(d, other_bytes.digest());
        assert_ne!(d, clean.digest());
        assert!(base.describe().contains("1 dirty path(s), 0 of them build output"));
        assert_eq!(base.files_json()[" M kernels/x.metal"], "1".repeat(64));
    }

    #[test]
    fn build_output_is_named_not_hashed_and_a_source_tree_is_hashed_by_content() {
        assert!(is_build_output("target-redteam/"));
        assert!(is_build_output("target/release/x"));
        assert!(!is_build_output("tests/common/"));
        assert!(!is_build_output("src/target.rs"));
        let root = std::env::temp_dir().join(format!("qdm-tree-{}", std::process::id()));
        let _fresh = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("a/b/x.rs"), b"one").unwrap();
        std::fs::write(root.join("y.rs"), b"two").unwrap();
        let (h1, _) = hash_tree(&root).unwrap();
        assert!(h1.starts_with("tree:"), "{h1}");
        assert_eq!(hash_tree(&root).unwrap().0, h1, "deterministic");
        std::fs::write(root.join("a/b/x.rs"), b"onE").unwrap();
        assert_ne!(hash_tree(&root).unwrap().0, h1, "a changed byte moves the tree hash");
        std::fs::write(root.join("a/b/x.rs"), b"one").unwrap();
        std::fs::rename(root.join("y.rs"), root.join("z.rs")).unwrap();
        assert_ne!(hash_tree(&root).unwrap().0, h1, "a renamed file moves the tree hash");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn backbone_commit_is_the_python_spelling() {
        assert_eq!(
            backbone_commit(Path::new("/x/snapshots/b1485b2f"), 248_320).unwrap(),
            "b1485b2f:vocab248320"
        );
    }

    #[test]
    fn the_running_tessl_checkout_is_readable() {
        // Not a GPU test: reads git state only. A tessl dir that is not a git checkout is a
        // provenance failure the bench must refuse on, so this is asserted rather than skipped.
        let s = TreeState::read(&tessl_dir()).expect("tessl state");
        println!("{} (digest {})", s.describe(), s.digest());
        assert_eq!(s.head.len(), 40, "{}", s.head);
        assert_eq!(s.digest().len(), 64);
    }
}
