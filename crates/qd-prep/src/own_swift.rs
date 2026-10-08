//! `qd-prep own-swift`: the Swift files of the human's own TRAIN repositories, as the pool
//! `qd-mutate generate` reads.
//!
//! R3 of `AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md`: corpus-v3 holds no Swift
//! file, all 39 v5 needle misses are Swift or TypeScript, and the admitted own train repositories
//! hold Swift. This module only writes the pool; `qd-mutate generate --language swift` mutates it.
//!
//! **Which repositories.** Exactly the `split: train` rows of a `qd-prep own-repos` manifest,
//! each read at the `head` the manifest pinned. The split is never re-decided here:
//! [`natural_bugs::read_manifest`] refuses a manifest whose rows disagree with the holdout rule
//! or that admits a repository the human excluded or struck, and every row read is checked
//! against [`own_repos::split_of`] once more before its tree is listed. A held-out repository
//! (GitPulse among them) is never listed, let alone read (CLAUDE.md rule 3), and neither is a
//! repository whose path carries a held-out marker ([`refuse_training_input`]).
//!
//! **Which files.** Every blob in `git ls-tree -r <head>` that [`language_from_path`] calls
//! Swift, minus, each a counted reason, first reason wins:
//! - `inside_nested_repository`: the path lies under another repository the manifest names
//!   (admitted or excluded) nested inside this one, so it is that repository's file, read under
//!   that repository's verdict or not at all (`scholarlm/wisdev-arc` is held out inside the
//!   train repository `scholarlm`);
//! - `struck_path`: under a struck directory of the manifest's `rules.struck_paths`;
//! - `not_a_regular_file`: a symlink (mode 120000) or a submodule;
//! - `file_too_large` (over [`MAX_FILE_BYTES`]), `binary` (a NUL byte), `not_utf8`.
//!
//! The working tree is never read: the pinned commit is, so the pool is a function of the
//! manifest and the repositories' object stores, and uncommitted work never enters it.
//!
//! **The record** is `qd-mutate`'s `pool::PoolRecord`: `id` (`own-swift:<slug>:<path>`),
//! `repo` (the manifest's lowercased `owner/name` slug), `path` and `source`. `hunks` and
//! `prior_source` are absent, as for any pool walked off a tree with no diff attached
//! (`crates/qd-mutate/src/pool.rs`). What that means downstream is a property of the pool, so the
//! manifest says it: `generate` stamps every example `hunk_constrained: false`, its `clean`
//! examples carry an empty diff (`qd_train.mutate_adapter.refuse_leaky_diff_corpus` refuses such
//! a corpus in diff mode), and `qd-mutate compose` has no pristine rendering of any record.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use qd_lang::{LangId, language_from_path};
use serde_json::{Value, json};

use crate::gitcli::{self, is_full_sha};
use crate::heldout::refuse_training_input;
use crate::natural_bugs;
use crate::own_repos::{self, split_of};
use crate::sha256::sha256_hex;

pub const SCHEMA: &str = "qd-own-swift-pool/v1";
/// The language this pool holds.
pub const LANGUAGE: LangId = LangId::Swift;
/// Largest file emitted, natural-bugs' bound.
pub const MAX_FILE_BYTES: u64 = natural_bugs::MAX_FILE_BYTES;
/// `ls-tree -r` output one repository may produce.
const MAX_TREE_BYTES: u64 = 256 << 20;
/// Records one run may emit, and their total bytes: a code root past either is mis-pointed.
const MAX_RECORDS: usize = 200_000;
const MAX_POOL_BYTES: usize = 1 << 30;
/// The manifest is read whole.
const MAX_MANIFEST_BYTES: u64 = 16 << 20;

/// One train repository as the manifest pinned it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainRepo {
    pub repo: String,
    pub slug: String,
    pub head: String,
}

/// What this command takes from an own-repos manifest.
#[derive(Clone, Debug)]
pub struct View {
    pub code_root: PathBuf,
    pub train: Vec<TrainRepo>,
    /// Held-out repositories: named in the output manifest, never read.
    pub heldout: Vec<String>,
    /// Every repository path the manifest names, admitted or excluded: the nesting check.
    pub all_repos: Vec<String>,
    pub struck: Vec<(String, String)>,
}

/// The manifest's train repositories, after [`natural_bugs::read_manifest`]'s checks.
pub fn read_view(doc: &Value) -> Result<View, String> {
    let checked = natural_bugs::read_manifest(doc)?;
    let admitted = doc["admitted"]
        .as_array()
        .ok_or("the manifest has no admitted list")?;
    let mut train = Vec::new();
    let mut all_repos = Vec::new();
    for a in admitted {
        let field = |k: &str| {
            a[k].as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("an admitted row has no {k}: {a}"))
        };
        let (repo, slug, split, head) = (
            field("repo")?,
            field("slug")?,
            field("split")?,
            field("head")?,
        );
        all_repos.push(repo.clone());
        match split.as_str() {
            "train" => train.push(TrainRepo { repo, slug, head }),
            "heldout" => {}
            other => {
                return Err(format!(
                    "{repo}: split {other:?} is neither train nor heldout"
                ));
            }
        }
    }
    let excluded = doc["excluded"]
        .as_object()
        .ok_or("the manifest has no excluded map")?;
    all_repos.extend(excluded.keys().cloned());
    all_repos.sort();
    all_repos.dedup();
    train.sort_by(|a, b| a.repo.cmp(&b.repo));
    Ok(View {
        code_root: checked.code_root,
        train,
        heldout: checked.held.into_iter().map(|h| h.repo).collect(),
        all_repos,
        struck: checked.struck,
    })
}

/// One `ls-tree -r -l -z` entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeEntry {
    pub mode: String,
    pub kind: String,
    pub object: String,
    /// `None` for a tree or commit entry, which `-l` prints as `-`.
    pub size: Option<u64>,
    pub path: String,
}

/// Parse `git ls-tree -r -l -z` output: `<mode> SP <type> SP <object> SP+ <size> TAB <path> NUL`.
pub fn parse_ls_tree(out: &[u8]) -> Result<Vec<TreeEntry>, String> {
    let mut entries = Vec::new();
    for raw in out.split(|&b| b == 0) {
        if raw.is_empty() {
            continue;
        }
        let text = std::str::from_utf8(raw)
            .map_err(|e| format!("ls-tree entry is not UTF-8 ({e}): {raw:?}"))?;
        let (meta, path) = text
            .split_once('\t')
            .ok_or_else(|| format!("ls-tree entry has no tab: {text:?}"))?;
        let mut f = meta.split_ascii_whitespace();
        let (Some(mode), Some(kind), Some(object), Some(size), None) =
            (f.next(), f.next(), f.next(), f.next(), f.next())
        else {
            return Err(format!(
                "ls-tree entry is not mode/type/object/size: {text:?}"
            ));
        };
        let size = match size {
            "-" => None,
            s => Some(
                s.parse::<u64>()
                    .map_err(|e| format!("ls-tree size {s:?}: {e}"))?,
            ),
        };
        if path.is_empty() {
            return Err(format!("ls-tree entry has an empty path: {text:?}"));
        }
        entries.push(TreeEntry {
            mode: mode.to_string(),
            kind: kind.to_string(),
            object: object.to_string(),
            size,
            path: path.to_string(),
        });
    }
    Ok(entries)
}

/// The nested repository (as a prefix inside `repo`) that holds `path`, if any.
fn nested_owner<'a>(all_repos: &'a [String], repo: &str, path: &str) -> Option<&'a str> {
    all_repos
        .iter()
        .filter_map(|other| {
            other
                .strip_prefix(repo)?
                .strip_prefix('/')
                .map(|rel| (other, rel))
        })
        .find(|(_, rel)| path.starts_with(&format!("{rel}/")))
        .map(|(other, _)| other.as_str())
}

/// Reads a repository's tree and blobs. The git CLI in production; a map in unit tests.
pub trait Reader {
    fn ls_tree(&self, repo: &Path, head: &str) -> Result<Vec<TreeEntry>, String>;
    fn blob(&self, repo: &Path, object: &str) -> Result<Vec<u8>, String>;
}

/// [`Reader`] over the git CLI ([`gitcli::run`]: bounded output, timeout, no host config).
pub struct GitReader;

impl Reader for GitReader {
    fn ls_tree(&self, repo: &Path, head: &str) -> Result<Vec<TreeEntry>, String> {
        let out = gitcli::run(
            repo,
            &["ls-tree", "-r", "-l", "-z", "--full-tree", head],
            MAX_TREE_BYTES,
        )
        .map_err(|e| e.to_string())?;
        parse_ls_tree(&out)
    }

    fn blob(&self, repo: &Path, object: &str) -> Result<Vec<u8>, String> {
        gitcli::run(repo, &["cat-file", "blob", object], MAX_FILE_BYTES).map_err(|e| e.to_string())
    }
}

/// One repository's records (`(id, line)`) and counts.
#[derive(Debug, Default)]
pub struct RepoOut {
    pub records: Vec<(String, String)>,
    pub tree_blobs: u64,
    pub swift_paths: u64,
    pub refusals: BTreeMap<String, u64>,
}

fn bump(m: &mut BTreeMap<String, u64>, k: &str) {
    *m.entry(k.to_string()).or_default() += 1;
}

/// One train repository's Swift files at its pinned head, as pool record lines.
pub fn emit_repo(
    t: &TrainRepo,
    repo_path: &Path,
    view: &View,
    reader: &dyn Reader,
) -> Result<RepoOut, String> {
    if split_of(&t.slug) != "train" {
        return Err(format!(
            "{}: {} is held out by the rule; a training pool never reads it (CLAUDE.md rule 3)",
            t.repo, t.slug
        ));
    }
    if !is_full_sha(&t.head) {
        return Err(format!("{}: head {:?} is not a commit sha", t.repo, t.head));
    }
    refuse_training_input(repo_path, "repository")?;
    let mut out = RepoOut::default();
    let mut entries = reader.ls_tree(repo_path, &t.head)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    for e in entries {
        if e.kind == "blob" {
            out.tree_blobs += 1;
        }
        if language_from_path(&e.path) != Some(LANGUAGE) {
            continue;
        }
        out.swift_paths += 1;
        if nested_owner(&view.all_repos, &t.repo, &e.path).is_some() {
            bump(&mut out.refusals, "inside_nested_repository");
            continue;
        }
        if own_repos::struck_path(&view.struck, &t.repo, &e.path).is_some() {
            bump(&mut out.refusals, "struck_path");
            continue;
        }
        if e.kind != "blob" || !(e.mode == "100644" || e.mode == "100755") {
            bump(&mut out.refusals, "not_a_regular_file");
            continue;
        }
        match e.size {
            Some(n) if n > MAX_FILE_BYTES => {
                bump(&mut out.refusals, "file_too_large");
                continue;
            }
            Some(_) => {}
            None => return Err(format!("{}:{}: a blob with no size", t.repo, e.path)),
        }
        let bytes = reader.blob(repo_path, &e.object)?;
        if bytes.contains(&0) {
            bump(&mut out.refusals, "binary");
            continue;
        }
        let Ok(source) = String::from_utf8(bytes) else {
            bump(&mut out.refusals, "not_utf8");
            continue;
        };
        let id = format!("own-swift:{}:{}", t.slug, e.path);
        let line = serde_json::to_string(&json!({
            "id": id, "repo": t.slug, "path": e.path, "source": source,
        }))
        .map_err(|e| e.to_string())?;
        out.records.push((id, line));
    }
    Ok(out)
}

/// The pool's JSONL bytes and its manifest, for `manifest_bytes` read from `manifest_path`.
pub fn build(
    manifest_path: &Path,
    manifest_bytes: &[u8],
    reader: &dyn Reader,
) -> Result<(Vec<u8>, Value), String> {
    let doc: Value = serde_json::from_slice(manifest_bytes)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let view = read_view(&doc)?;
    if view.train.is_empty() {
        return Err("the manifest admits no train repository; there is nothing to emit".into());
    }
    let mut jsonl: Vec<u8> = Vec::new();
    let mut per_repo = BTreeMap::new();
    let mut refusals: BTreeMap<String, u64> = BTreeMap::new();
    let (mut records, mut swift_paths, mut tree_blobs) = (0usize, 0u64, 0u64);
    let mut seen_ids = std::collections::BTreeSet::new();
    for t in &view.train {
        let path = view.code_root.join(&t.repo);
        let r = emit_repo(t, &path, &view, reader)?;
        for (k, v) in &r.refusals {
            *refusals.entry(k.clone()).or_default() += v;
        }
        swift_paths += r.swift_paths;
        tree_blobs += r.tree_blobs;
        per_repo.insert(
            t.repo.clone(),
            json!({
                "slug": t.slug, "head": t.head, "tree_blobs": r.tree_blobs,
                "swift_paths": r.swift_paths, "emitted": r.records.len(), "refusals": r.refusals,
            }),
        );
        for (id, line) in r.records {
            if !seen_ids.insert(id.clone()) {
                return Err(format!("duplicate record id {id}"));
            }
            records += 1;
            if records > MAX_RECORDS {
                return Err(format!("more than {MAX_RECORDS} records; refusing"));
            }
            jsonl.extend_from_slice(line.as_bytes());
            jsonl.push(b'\n');
            if jsonl.len() > MAX_POOL_BYTES {
                return Err(format!("the pool passes {MAX_POOL_BYTES} bytes; refusing"));
            }
        }
    }
    let manifest = json!({
        "schema": SCHEMA,
        "tool": "qd-prep own-swift",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "ruling": "AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md R3",
        "own_repos_manifest": {
            "path": manifest_path.display().to_string(),
            "sha256": sha256_hex(manifest_bytes),
        },
        "code_root": view.code_root.display().to_string(),
        "language": LANGUAGE.as_str(),
        "rules": {
            "repositories": "the manifest's split: train rows, each re-checked against own_repos::split_of; held-out rows are never read",
            "files": "blobs in `git ls-tree -r <pinned head>` that qd_lang::language_from_path calls swift; the working tree is never read",
            "refusals_first_wins": ["inside_nested_repository", "struck_path", "not_a_regular_file", "file_too_large", "binary", "not_utf8"],
            "max_file_bytes": MAX_FILE_BYTES,
        },
        "heldout_repos_not_read": view.heldout,
        "per_repo": per_repo,
        "totals": {
            "train_repos": view.train.len(), "tree_blobs": tree_blobs,
            "swift_paths": swift_paths, "records": records, "refusals": refusals,
        },
        "record_shape": "qd-mutate pool::PoolRecord {id, repo, path, source}; hunks and prior_source absent (a pool walked off a pinned tree has no diff attached)",
        "downstream": [
            "qd-mutate generate stamps every example hunk_constrained: false",
            "a clean example from this pool carries an empty diff, which qd_train.mutate_adapter.refuse_leaky_diff_corpus refuses in diff mode; --clean-permille 0 emits none",
            "qd-mutate compose has no pristine rendering of a record without prior_source (compose::render_pristine), so every record is file_no_pristine_change there",
        ],
        "pool": {"file": "pool.jsonl", "sha256": sha256_hex(&jsonl), "records": records, "bytes": jsonl.len()},
    });
    Ok((jsonl, manifest))
}

/// `qd-prep own-swift --manifest M --out-dir DIR`: `DIR/{pool.jsonl, manifest.json}`, whole or
/// not at all.
pub fn run(manifest_path: &Path, out_dir: &Path) -> Result<String, String> {
    let partial = crate::files::partial_path(out_dir)?;
    for p in [out_dir, partial.as_path()] {
        if p.exists() {
            return Err(format!("{} exists; refusing to overwrite it", p.display()));
        }
    }
    let manifest_bytes =
        crate::files::read_bounded(manifest_path, MAX_MANIFEST_BYTES, "--manifest")?;
    let (jsonl, manifest) = build(manifest_path, &manifest_bytes, &GitReader)?;
    let mut manifest_out = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    manifest_out.push(b'\n');
    crate::files::write_new_dir(out_dir, |partial| {
        crate::files::write_synced_new(&partial.join("pool.jsonl"), &jsonl)?;
        crate::files::write_synced_new(&partial.join("manifest.json"), &manifest_out)
    })?;
    Ok(format!(
        "qd-prep own-swift: {} train repo(s), {} swift path(s), {} record(s) sha256 {} -> {}",
        manifest["totals"]["train_repos"],
        manifest["totals"]["swift_paths"],
        manifest["totals"]["records"],
        manifest["pool"]["sha256"].as_str().unwrap_or_default(),
        out_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ls_tree_entries_parse_with_sizes_and_tabs_in_no_field_but_the_path() {
        let out = b"100644 blob 0123456789012345678901234567890123456789     120\tA/b c.swift\0\
                    160000 commit 1111111111111111111111111111111111111111       -\tsub\0";
        let e = parse_ls_tree(out).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].path, "A/b c.swift");
        assert_eq!(e[0].size, Some(120));
        assert_eq!(e[1].kind, "commit");
        assert_eq!(e[1].size, None);
        assert!(
            parse_ls_tree(b"100644 blob abc\tx\0").is_err(),
            "a missing size is refused"
        );
        assert!(
            parse_ls_tree(b"100644 blob abc 1 x\0").is_err(),
            "no tab is refused"
        );
    }

    #[test]
    fn a_nested_repository_owns_its_prefix_and_only_inside_its_parent() {
        let all = vec!["scholarlm".to_string(), "scholarlm/wisdev-arc".to_string()];
        assert_eq!(
            nested_owner(&all, "scholarlm", "wisdev-arc/A.swift"),
            Some("scholarlm/wisdev-arc")
        );
        assert_eq!(
            nested_owner(&all, "scholarlm", "wisdev-arcade/A.swift"),
            None
        );
        assert_eq!(nested_owner(&all, "scholarlm/wisdev-arc", "A.swift"), None);
        assert_eq!(nested_owner(&all, "scholar", "lm/wisdev-arc/A.swift"), None);
    }
}
