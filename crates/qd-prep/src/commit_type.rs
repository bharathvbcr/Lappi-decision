//! `qd-prep commit-type`: `gitpulse.commit_type` rows from the human's own TRAIN repositories'
//! commit history (R5 of `AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md`).
//!
//! "Synthetic only" governs caller records; a repository's own commit history is the human's own
//! data, admitted on 2026-09-28. This reads **no caller record** (rule 3): its only inputs are
//! an own-repos manifest and the git object stores it names.
//!
//! **Which repositories.** Exactly the `split: train` rows of a `qd-prep own-repos` manifest, at
//! the heads it pinned ([`own_swift::read_view`] re-checks the manifest against the holdout
//! rule). The held-out repositories are never listed. GitPulse is the held-out eval of this
//! family, and the one repository that must never be a producer of it: a manifest that admits it
//! as `train` is refused whole ([`REFUSED_PRODUCERS`]), not skipped.
//!
//! **The row** is the request GitPulse sends (`docs/caller-contract.md` section 4,
//! `GitPulse/src-tauri/src/lappi/request.rs`):
//! - question and slot: the family's description in `qd_data.sources.DECISION_FAMILIES` and
//!   [`SLOT`]; options exactly [`KNOWN_TYPES`], in order (`commit_brief.rs` `KNOWN_TYPES`);
//! - context: the commit's unified diff **as bytes**, with nothing in front of it
//!   ([`decisions::BARE_CONTEXT_FAMILIES`]). The app sends the staged diff; the commit's
//!   equivalent is `git -c core.quotepath=false diff --no-color --no-ext-diff -U3 <parent> <commit>` (the empty
//!   tree for a root commit): GitPulse's own argv is `diff --cached --no-color --no-ext-diff -U3`
//!   under `core.quotepath=false` (`ai/mod.rs` `read_staged_patch`, `engine/git_cli.rs`, at 211cd903);
//! - gold: the `type` of the commit subject's conventional prefix `type(scope)!: `, when it is
//!   one of [`KNOWN_TYPES`]. A subject with another prefix, or none, is counted and dropped.
//!   The message is never in the context, so the label is not readable from it.
//!
//! **Drops, each counted and none silent:** `no_conventional_prefix`, `type_not_in_known_types`,
//! `struck_path`, `inside_nested_repository` and `heldout_repository_content` (a commit touching a
//! struck path, a nested repository's path, or a blob of a held-out repository), `secret_like` (a
//! credential-shaped string in the diff, by `qd_runtime::caller_record::secret_like_prefix`),
//! `empty_diff`, `context_over_bound` (a diff past [`decisions::MAX_CONTEXT_BYTES`], the
//! 131,072-byte bound), `context_not_utf8`, `unassigned_code_point`, `repo_cap` and
//! `exact_duplicate`. `context_not_utf8` departs from the ruling's "invalid UTF-8 is not a drop
//! rule": a pool row is a JSON string, which cannot hold bytes that are not UTF-8, and lossy
//! conversion would train a byte string the app never sends. It is the departure, counted.
//!
//! **Caps.** At most [`PER_REPO_CAP`] rows per repository, drawn in a seeded order among the rows
//! that passed every other rule (DevCouncil and scholarlm are 67% of the supply). Val is a seeded
//! 5% of commits, within the train repositories; `group_key` is the commit. Amendment 2 is met by
//! reporting every count split by agent-authored (a `Co-Authored-By` trailer) and human-authored.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Value, json};

use crate::decisions::{self, Candidate, Gold};
use crate::gitcli::{self, GitError};
use crate::natural_bugs;
use crate::own_repos;
use crate::own_swift::{self, TrainRepo, View};
use crate::pool;
use crate::sha256::sha256_hex;

pub const FAMILY: &str = "gitpulse.commit_type";
pub const SCHEMA: &str = "qd-commit-type-pool/v1";
/// `docs/caller-contract.md` section 4: the family's description, which is the request's question.
pub const QUESTION: &str = "Which conventional commit type describes this staged change?";
pub const SLOT: &str = "commit_type";
/// `GitPulse/src-tauri/src/ai/commit_brief.rs` `KNOWN_TYPES`: the options, in this order.
pub const KNOWN_TYPES: [&str; 11] = [
    "feat", "fix", "refactor", "docs", "test", "chore", "perf", "build", "ci", "style", "revert",
];
/// The pool's source (`qd_data.sources`): the human's own repositories' commit history.
pub const SOURCE: &str = "lappi/own-commits";
/// The human's grant (`qd_data.licences`).
pub const LICENCE: &str = "owner-granted";
/// R5: DevCouncil and scholarlm are 67% of the supply.
pub const PER_REPO_CAP: usize = 400;
/// R5: "Val is 5%, within the train repos."
pub const VAL_SHARE: f64 = 0.05;
/// Repositories that are never a producer of this family: its held-out eval. A repository is
/// refused by ANY of: its manifest slug, its repository directory name, its origin remote's
/// `owner/name` (all compared lowercased), or one of its root commits ([`REFUSED_ROOT_COMMITS`]):
/// an edited slug or a renamed directory does not change what a repository is.
pub const REFUSED_PRODUCERS: [&str; 1] = ["bharathvbcr/gitpulse"];
/// GitPulse's root commit(s) (`git rev-list --max-parents=0`, read 2026-10-12).
pub const REFUSED_ROOT_COMMITS: [&str; 1] = ["5a633d91b505b44e23bb0ed761605a0f383205d1"];
/// `git log` output one repository may produce.
const MAX_LOG_BYTES: u64 = 256 << 20;
/// A manifest is read whole.
const MAX_MANIFEST_BYTES: u64 = 16 << 20;
/// The empty tree: what a root commit is diffed against.
const EMPTY_TREE: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
/// Rows one run may emit before the loader would refuse the pool.
const MAX_ROWS: usize = 50_000;

/// What a commit subject's prefix says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prefix {
    /// An index into [`KNOWN_TYPES`].
    Known(usize),
    /// A well-formed `type:` prefix whose type is not one of the eleven.
    Other(String),
    /// No conventional prefix.
    None,
}

/// The conventional prefix of `message`'s first line: `type`, an optional `(scope)`, an optional
/// `!`, then `: `. `type` is lowercase ASCII letters only.
pub fn prefix_of(message: &str) -> Prefix {
    let subject = message.lines().next().unwrap_or("");
    let ty_len = subject
        .bytes()
        .take_while(|b| b.is_ascii_lowercase())
        .count();
    if ty_len == 0 {
        return Prefix::None;
    }
    let (ty, mut rest) = subject.split_at(ty_len);
    if let Some(after) = rest.strip_prefix('(') {
        match after.find(')') {
            Some(close) if close > 0 && !after[..close].contains(['\n', '(']) => {
                rest = &after[close + 1..];
            }
            _ => return Prefix::None,
        }
    }
    let rest = rest.strip_prefix('!').unwrap_or(rest);
    if !rest.starts_with(": ") {
        return Prefix::None;
    }
    match KNOWN_TYPES.iter().position(|k| *k == ty) {
        Some(i) => Prefix::Known(i),
        None => Prefix::Other(ty.to_owned()),
    }
}

/// Whether the message carries a `Co-Authored-By:` trailer line (agent-authored, by the
/// convention of the human's own repositories).
pub fn is_agent_authored(message: &str) -> bool {
    message
        .lines()
        .any(|l| l.trim_start().to_ascii_lowercase().starts_with("co-authored-by:"))
}

/// The commits of `t` reachable from its pinned head, merges excluded:
/// `(sha, first parent or none, message)`.
fn commits(root: &Path, t: &TrainRepo) -> Result<Vec<(String, Option<String>, String)>, String> {
    let repo = root.join(&t.repo);
    if !gitcli::is_full_sha(&t.head) {
        return Err(format!("{}: head {:?} is not a full sha", t.repo, t.head));
    }
    let out = gitcli::run(
        &repo,
        &[
            "log",
            "--no-merges",
            "-z",
            "--format=%H%x1f%P%x1f%B",
            &t.head,
        ],
        MAX_LOG_BYTES,
    )
    .map_err(|e| format!("{}: {e}", t.repo))?;
    let mut rows = Vec::new();
    for rec in out.split(|b| *b == 0).filter(|r| !r.is_empty()) {
        let text = String::from_utf8_lossy(rec);
        let mut parts = text.trim_start_matches('\n').splitn(3, '\u{1f}');
        let (Some(sha), Some(parents), Some(message)) = (parts.next(), parts.next(), parts.next())
        else {
            return Err(format!("{}: a git log record has no three fields", t.repo));
        };
        if !gitcli::is_full_sha(sha) {
            return Err(format!("{}: git log gave {sha:?}, not a sha", t.repo));
        }
        let parent = parents.split_whitespace().next().map(str::to_owned);
        rows.push((sha.to_owned(), parent, message.to_owned()));
    }
    Ok(rows)
}

/// A commit's unified diff as bytes. `Ok(None)` when it passes the context bound.
fn diff_bytes(
    root: &Path,
    t: &TrainRepo,
    sha: &str,
    parent: Option<&str>,
) -> Result<Option<Vec<u8>>, String> {
    let repo = root.join(&t.repo);
    let base = parent.unwrap_or(EMPTY_TREE);
    match gitcli::run(
        &repo,
        &[
            "-c",
            "core.quotepath=false",
            "diff",
            "--no-color",
            "--no-ext-diff",
            "-U3",
            base,
            sha,
        ],
        decisions::MAX_CONTEXT_BYTES as u64,
    ) {
        Ok(b) => Ok(Some(b)),
        Err(GitError::TooLarge(_)) => Ok(None),
        Err(e) => Err(format!("{}: {e}", t.repo)),
    }
}

/// Why `t` may not produce rows, by any of the identities of [`REFUSED_PRODUCERS`].
fn refuse_producer(view: &View, t: &TrainRepo) -> Result<(), String> {
    let refused = |why: String| {
        Err(format!(
            "{} ({}) is {why}, and it is the held-out eval of {FAMILY}: refused as a producer, \
             whole run",
            t.repo, t.slug
        ))
    };
    let name = t.repo.rsplit('/').next().unwrap_or(&t.repo).to_lowercase();
    if REFUSED_PRODUCERS.contains(&t.slug.to_lowercase().as_str())
        || REFUSED_PRODUCERS
            .iter()
            .any(|p| p.rsplit('/').next() == Some(name.as_str()))
    {
        return refused("named as GitPulse by its slug or directory".to_owned());
    }
    let repo = view.code_root.join(&t.repo);
    let origin = gitcli::run(&repo, &["config", "--get", "remote.origin.url"], 4096)
        .map_err(|e| format!("{}: no readable origin remote: {e}", t.repo))?;
    let origin = String::from_utf8_lossy(&origin);
    let Some(origin_slug) = crate::own_repos::slug(origin.trim()) else {
        return Err(format!("{}: origin {:?} is not a github owner/name", t.repo, origin.trim()));
    };
    if REFUSED_PRODUCERS.contains(&origin_slug.as_str()) {
        return refused(format!("GitPulse by its origin remote {origin_slug}"));
    }
    if origin_slug != t.slug.to_lowercase() {
        return Err(format!(
            "{}: the manifest's slug {} is not the repository's origin {origin_slug}: refused",
            t.repo, t.slug
        ));
    }
    let roots = gitcli::run(&repo, &["rev-list", "--max-parents=0", &t.head], 1 << 20)
        .map_err(|e| format!("{}: {e}", t.repo))?;
    for root in String::from_utf8_lossy(&roots).split_whitespace() {
        if REFUSED_ROOT_COMMITS.contains(&root) {
            return refused(format!("GitPulse by its root commit {root}"));
        }
    }
    Ok(())
}

/// The blob ids of the held-out repositories nested inside the code root, at their pinned heads:
/// object ids only, never content. A commit of a train repository whose diff touches one of them
/// carries a held-out repository's file whatever path it sits under.
pub fn heldout_blob_ids(
    code_root: &Path,
    held: &[natural_bugs::HeldOutRepo],
) -> Result<BTreeSet<String>, String> {
    let mut ids = BTreeSet::new();
    for h in held {
        let out = gitcli::run(
            &code_root.join(&h.repo),
            &["ls-tree", "-r", "--full-tree", &h.head],
            256 << 20,
        )
        .map_err(|e| format!("{}: {e}", h.repo))?;
        for line in String::from_utf8_lossy(&out).lines() {
            let mut f = line.split_whitespace();
            if let (Some(_), Some("blob"), Some(id)) = (f.next(), f.next(), f.next()) {
                ids.insert(id.to_owned());
            }
        }
    }
    Ok(ids)
}

/// A changed file of a commit: its path and its old and new blob ids.
struct Changed {
    path: String,
    blobs: [String; 2],
}

fn changed_files(
    root: &Path,
    t: &TrainRepo,
    parent: Option<&str>,
    sha: &str,
) -> Result<Vec<Changed>, String> {
    let out = gitcli::run(
        &root.join(&t.repo),
        &[
            "diff", "--raw", "-z", "--no-abbrev", "--no-renames", "--no-ext-diff",
            parent.unwrap_or(EMPTY_TREE), sha,
        ],
        64 << 20,
    )
    .map_err(|e| format!("{}: {e}", t.repo))?;
    let mut parts = out.split(|b| *b == 0).filter(|p| !p.is_empty());
    let mut files = Vec::new();
    while let Some(meta) = parts.next() {
        let meta = String::from_utf8_lossy(meta);
        let f: Vec<&str> = meta.trim_start_matches(':').split_whitespace().collect();
        let (Some(path), true) = (parts.next(), f.len() >= 5) else {
            return Err(format!("{}: a diff --raw record is malformed: {meta:?}", t.repo));
        };
        files.push(Changed {
            path: String::from_utf8_lossy(path).into_owned(),
            blobs: [f[2].to_owned(), f[3].to_owned()],
        });
    }
    Ok(files)
}

/// Counts of one repository's commits by outcome, and by who authored them.
#[derive(Default)]
struct Tally {
    commits: usize,
    kept: usize,
    drops: BTreeMap<&'static str, usize>,
    kept_agent: usize,
    kept_human: usize,
    drops_agent: usize,
    drops_human: usize,
    other_types: BTreeMap<String, usize>,
}

impl Tally {
    fn drop(&mut self, reason: &'static str, agent: bool) {
        *self.drops.entry(reason).or_default() += 1;
        if agent {
            self.drops_agent += 1;
        } else {
            self.drops_human += 1;
        }
    }
}

/// The candidates `view` yields at the ruled per-repository cap, and the report of how.
pub fn candidates(
    view: &View,
    seed: u64,
    heldout_blobs: &BTreeSet<String>,
) -> Result<(Vec<Candidate>, Value), String> {
    candidates_capped(view, seed, PER_REPO_CAP, heldout_blobs)
}

/// [`candidates`] with the per-repository cap named (tests use a small one).
pub fn candidates_capped(
    view: &View,
    seed: u64,
    per_repo_cap: usize,
    heldout_blobs: &BTreeSet<String>,
) -> Result<(Vec<Candidate>, Value), String> {
    for t in &view.train {
        refuse_producer(view, t)?;
    }
    let mut out: Vec<Candidate> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut per_repo = serde_json::Map::new();
    let mut by_type: BTreeMap<&str, usize> = BTreeMap::new();
    for t in &view.train {
        let mut tally = Tally::default();
        let mut eligible: Vec<(String, Option<String>, String, usize, bool)> = Vec::new();
        for (sha, parent, message) in commits(&view.code_root, t)? {
            tally.commits += 1;
            let agent = is_agent_authored(&message);
            match prefix_of(&message) {
                Prefix::None => tally.drop("no_conventional_prefix", agent),
                Prefix::Other(ty) => {
                    *tally.other_types.entry(ty).or_default() += 1;
                    tally.drop("type_not_in_known_types", agent);
                }
                Prefix::Known(i) => eligible.push((sha, parent, message, i, agent)),
            }
        }
        // Seeded order: the cap keeps the first PER_REPO_CAP that pass every other rule.
        eligible.sort_by_key(|(sha, ..)| decisions::keyed(seed, &["commit-type", &t.slug, sha]));
        let mut taken = 0usize;
        for (sha, parent, _message, gold, agent) in eligible {
            if taken >= per_repo_cap {
                tally.drop("repo_cap", agent);
                continue;
            }
            // The human's struck paths, nested repositories (admitted, excluded or held out) and
            // held-out blobs, by every file the commit touches: one such file refuses the commit.
            let mut refusal: Option<&'static str> = None;
            for c in changed_files(&view.code_root, t, parent.as_deref(), &sha)? {
                if own_repos::struck_path(&view.struck, &t.repo, &c.path).is_some() {
                    refusal = Some("struck_path");
                } else if own_swift::nested_owner(&view.all_repos, &t.repo, &c.path).is_some() {
                    refusal = Some("inside_nested_repository");
                } else if c.blobs.iter().any(|b| heldout_blobs.contains(b)) {
                    refusal = Some("heldout_repository_content");
                }
                if refusal.is_some() {
                    break;
                }
            }
            if let Some(reason) = refusal {
                tally.drop(reason, agent);
                continue;
            }
            let Some(bytes) = diff_bytes(&view.code_root, t, &sha, parent.as_deref())? else {
                tally.drop("context_over_bound", agent);
                continue;
            };
            let Ok(context) = String::from_utf8(bytes) else {
                tally.drop("context_not_utf8", agent);
                continue;
            };
            if context.trim().is_empty() {
                tally.drop("empty_diff", agent);
                continue;
            }
            if qd_runtime::caller_record::secret_like_prefix(&context).is_some() {
                tally.drop("secret_like", agent);
                continue;
            }
            if crate::pyunicode::check_assigned(&context).is_err() {
                tally.drop("unassigned_code_point", agent);
                continue;
            }
            let digest = sha256_hex(format!("{gold}\u{1f}{context}").as_bytes());
            if !seen.insert(digest) {
                tally.drop("exact_duplicate", agent);
                continue;
            }
            let split = if decisions::unit_draw(seed, &["commit-type-val", &t.slug, &sha])
                < VAL_SHARE
            {
                "val"
            } else {
                "train"
            };
            let c = Candidate {
                id: format!("commit-type:{}:{}", t.slug, &sha[..12]),
                source_id: SOURCE,
                family_id: FAMILY.to_owned(),
                stratum: format!("{FAMILY}/{}", KNOWN_TYPES[gold]),
                group_key: format!("{}@{sha}", t.slug),
                licence: LICENCE.to_owned(),
                context,
                question: String::new(),
                slot_name: SLOT.to_owned(),
                options: KNOWN_TYPES.iter().map(|k| (*k).to_owned()).collect(),
                gold: Gold::Option(gold),
                label_basis: "by_construction",
                split,
            };
            if let Some(reason) = decisions::structural_refusal(&c) {
                return Err(format!("{}: a row built here is refused ({reason}): {}", t.repo, c.id));
            }
            taken += 1;
            tally.kept += 1;
            if agent {
                tally.kept_agent += 1;
            } else {
                tally.kept_human += 1;
            }
            *by_type.entry(KNOWN_TYPES[gold]).or_default() += 1;
            out.push(c);
            if out.len() > MAX_ROWS {
                return Err(format!("more than {MAX_ROWS} rows; refusing"));
            }
        }
        per_repo.insert(
            t.repo.clone(),
            json!({
                "slug": t.slug, "head": t.head, "commits": tally.commits, "kept": tally.kept,
                "kept_agent_authored": tally.kept_agent, "kept_human_authored": tally.kept_human,
                "dropped": tally.drops, "dropped_agent_authored": tally.drops_agent,
                "dropped_human_authored": tally.drops_human,
                "prefixes_outside_known_types": tally.other_types,
            }),
        );
    }
    let report = json!({
        "per_repo": per_repo,
        "kept_by_type": by_type,
        "per_repo_cap": per_repo_cap,
        "val_share": VAL_SHARE,
        "heldout_repos_not_read": view.heldout,
        "refused_producers": REFUSED_PRODUCERS,
        "heldout_blob_ids_checked": heldout_blobs.len(),
        "secret_screen": "qd_runtime::caller_record::secret_like_prefix over every diff",
    });
    Ok((out, report))
}

/// The pool's examples bytes and manifest, for `manifest_bytes` read from `manifest_path`.
pub fn build(
    manifest_path: &Path,
    manifest_bytes: &[u8],
    seed: u64,
) -> Result<(Vec<u8>, Value), String> {
    let doc: Value = serde_json::from_slice(manifest_bytes)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let view = own_swift::read_view(&doc)?;
    let held = natural_bugs::read_manifest(&doc)?.held;
    let blobs = heldout_blob_ids(&view.code_root, &held)?;
    build_from_view(&view, manifest_path, manifest_bytes, seed, &blobs)
}

/// [`build`] over a view already read from the manifest at `manifest_path`.
pub fn build_from_view(
    view: &View,
    manifest_path: &Path,
    manifest_bytes: &[u8],
    seed: u64,
    heldout_blobs: &BTreeSet<String>,
) -> Result<(Vec<u8>, Value), String> {
    if view.train.is_empty() {
        return Err("the manifest admits no train repository; there is nothing to emit".into());
    }
    let (rows, report) = candidates(view, seed, heldout_blobs)?;
    if rows.is_empty() {
        return Err("no commit passed the rules: a pool of nothing is refused".into());
    }
    let refs: Vec<&Candidate> = rows.iter().collect();
    let ex = pool::examples(&refs)?;
    let manifest = json!({
        "schema": decisions::MANIFEST_SCHEMA, "mode": "build", "tool": "qd-prep commit-type",
        "tool_version": env!("CARGO_PKG_VERSION"), "family": FAMILY, "source_id": SOURCE,
        "licence": LICENCE, "seed": seed,
        "ruling": "AUDIT/v6-rulings-2026-10-08/fable-v6-data-design-ruling.md R5; AUDIT/board-sweep-2026-10-12/fable-board-sweep-ruling.md 2i",
        "own_repos_manifest": {
            "path": manifest_path.display().to_string(),
            "sha256": sha256_hex(manifest_bytes),
        },
        "inputs": {"own_repos_manifest": sha256_hex(manifest_bytes)},
        "request": {"question": QUESTION, "slot": SLOT, "options": KNOWN_TYPES,
                    "context": "the commit's unified diff bytes, nothing in front of it"},
        "privacy": "own commit history only; no caller record, mailbox or held-out repository is read",
        "selected": ex.selected,
        "noul_gold_selected": ex.noul,
        "gold_position": ex.gold_position,
        "val_without_train": ex.val_without_train,
        "generator": report,
        "decontamination": {
            "state": "not_run",
            "reason": "no target set exists for this family: its held-out eval is the held-out \
                       repositories' commit history, which this command never reads (rule 3). \
                       Exact-content duplicates are dropped (`exact_duplicate`).",
        },
        "allocation": pool::allocation(false, "candidates: no cap table has drawn from these \
                 rows; the v6 allocation over every pool decides what enters a mixture, and the \
                 loader refuses this pool until then"),
        "examples": ex.rows, "examples_sha256": ex.sha256,
    });
    Ok((ex.bytes, manifest))
}

/// `qd-prep commit-type --manifest M --out-dir DIR`: `DIR/{examples.jsonl, manifest.json}`, whole
/// or not at all.
pub fn run(manifest_path: &Path, out_dir: &Path, seed: u64) -> Result<String, String> {
    let partial = crate::files::partial_path(out_dir)?;
    for p in [out_dir, partial.as_path()] {
        if p.exists() {
            return Err(format!("{} exists; refusing to overwrite it", p.display()));
        }
    }
    crate::heldout::refuse_training_input(manifest_path, "--manifest")?;
    let bytes = crate::files::read_bounded(manifest_path, MAX_MANIFEST_BYTES, "--manifest")?;
    let (examples, manifest) = build(manifest_path, &bytes, seed)?;
    let mut manifest_out = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    manifest_out.push(b'\n');
    crate::files::write_new_dir(out_dir, |p| {
        crate::files::write_synced_new(&p.join("examples.jsonl"), &examples)?;
        crate::files::write_synced_new(&p.join("manifest.json"), &manifest_out)
    })?;
    Ok(format!(
        "qd-prep commit-type: {} rows sha256 {} -> {}",
        manifest["examples"],
        manifest["examples_sha256"].as_str().unwrap_or_default(),
        out_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefix_is_a_lowercase_type_an_optional_scope_and_bang_then_colon_space() {
        assert_eq!(prefix_of("feat: add x"), Prefix::Known(0));
        assert_eq!(prefix_of("fix(core)!: y\n\nbody"), Prefix::Known(1));
        assert_eq!(prefix_of("revert: z"), Prefix::Known(10));
        assert_eq!(prefix_of("wip: z"), Prefix::Other("wip".into()));
        for none in ["Feat: x", "feat:x", "feat (a): x", "feat(: x", "feat(): x", ": x", "", "feat"] {
            assert_eq!(prefix_of(none), Prefix::None, "{none:?}");
        }
    }

    #[test]
    fn the_options_are_the_apps_in_the_apps_order() {
        assert_eq!(KNOWN_TYPES.len(), 11);
        assert_eq!(&KNOWN_TYPES[..3], ["feat", "fix", "refactor"]);
        assert_eq!(KNOWN_TYPES[10], "revert");
    }

    #[test]
    fn a_trailer_marks_a_commit_agent_authored() {
        assert!(is_agent_authored("fix: x\n\nCo-Authored-By: A <a@b>"));
        assert!(is_agent_authored("fix: x\n\nco-authored-by: A <a@b>"));
        assert!(!is_agent_authored("fix: x\n\nbody mentions Co-Authored-By inline"));
    }
}
