//! `qd-prep natural-bugs`: real single-statement bug-fix commits from the human's **held-out**
//! repositories, as a held-out evaluation set that nothing trains on.
//!
//! The natural-bug transfer gate v0.1 lacked (`AUDIT/training-audit-2026-10-06.md` §4.7; v5's
//! defect skill did not carry over to model-written bugs). The output is a neutral JSONL, not
//! shard rows: rendering it into rows is a later, separate step.
//!
//! **Inputs, all required, none defaulted:**
//! - `--manifest`: a `qd-prep own-repos` manifest. Only its `split: heldout` repositories are
//!   read, at the `head` it pinned. A miner cannot run before a manifest exists, which makes
//!   "manifest before any commit is read" structural rather than procedural.
//! - `--cutoff YYYY-MM-DD` and `--cutoff-basis TEXT`: commits authored (`%at`, UTC) before
//!   00:00:00Z of that date are counted and never emitted. There is no default: the base
//!   model's data cutoff is unknown, and a defaulted date would be a guess presented as a rule.
//!   The basis text is copied into the report so a probe value cannot later read as the cutoff.
//! - `--v5-files`: own-prose-v1's `files.jsonl`. A commit touching any file it lists is
//!   excluded (`holdout_rule.v5_exposure.rule`, option ii).
//! - `--out-dir`: must have a path segment that is a held-out marker (`heldout`, `held_out`,
//!   `held-out`, `crates/qd-train/src/held_out.rs::DEFAULT_HELD_OUT_PATH_MARKERS`), so
//!   `qd-train`'s path check refuses the set to any training process (CLAUDE.md rule 3).
//!
//! **Single-statement fix, v1 (line-based)**, first failing check wins, each a counted reason:
//! 1. one parent (`root_commit`, `merge_commit` otherwise);
//! 2. fix-like: the message (subject and body) holds a word in [`FIX_WORDS`], where a word is a
//!    maximal run of ASCII letters and digits compared case-insensitively (`not_fix_like`);
//! 3. no touched path (old or new) is in `--v5-files` (`v5_own_prose_file`) or in a struck
//!    directory (`struck_path`);
//! 4. exactly one changed file (`empty_diff`, `multi_file`), modified in place, not renamed or
//!    copied (`rename_or_copy`, `not_a_modification`), a regular file both sides with the same
//!    mode (`not_a_regular_file`, `mode_change`), in a Lappi-admitted language
//!    (`qd_lang::language_from_path`; `language_not_admitted`);
//! 5. both blobs at most [`MAX_FILE_BYTES`], no NUL byte, valid UTF-8 (`file_too_large`,
//!    `binary`, `not_utf8`);
//! 6. `git diff --unified=0` of the two blobs has exactly one hunk with exactly one removed and
//!    one added line (`multi_hunk`, `not_one_line_each`);
//! 7. the two lines differ by more than whitespace (`whitespace_only`), neither is blank
//!    (`blank_line_side`), and neither is a whole-line comment (`comment_only` when both are,
//!    `commented_out_or_in` when one is). A trailing comment on a code line is **not** detected
//!    in v1.
//!
//! That is the SStuB shape (ManySStuBs4J's README: "single statement fixes", ignoring
//! "stylistic differences such as spaces or empty as well as differences in comments"),
//! approximated by lines rather than statements. A tree-sitter statement-level v2 is a
//! proposal only; `qd-prep` links no grammars.
//!
//! **The label is the span.** A natural bug has no mutation class, so every row carries
//! `mutation_class: null`: the class slot is unknown, not "none".

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use qd_lang::{DEFECT_CLASS_POOL_LANGUAGES, LangId, language_from_path};
use serde_json::{Value, json};

use crate::gitcli::{self, GitError, is_full_sha};
use crate::heldout::check_held_out_path;
use crate::own_repos::{self, civil_from_days, days_from_civil, split_of, utc_now};
use crate::sha256::sha256_hex;

pub const SCHEMA: &str = "qd-natural-bugs/v1";
/// A commit is fix-like when its message holds one of these words. The README of ManySStuBs4J
/// names "bug, fix, fault etc." without the full list; these are those three stems and their
/// inflections. "error" and "issue" are not in it: in these repositories they name features
/// ("add error handling") as often as fixes.
pub const FIX_WORDS: [&str; 11] = [
    "bug", "bugfix", "bugfixes", "bugs", "fault", "faults", "fix", "fixed", "fixes", "fixing",
    "hotfix",
];
/// A `Co-Authored-By:` trailer naming one of these marks a commit written with an agent.
pub const AGENT_NAMES: [&str; 10] = [
    "aider",
    "anthropic",
    "chatgpt",
    "claude",
    "codex",
    "copilot",
    "cursor",
    "devin",
    "gemini",
    "openai",
];
/// The set's target size (`AUDIT/training-audit-2026-10-06.md` §4.7). Reported, never enforced
/// by loosening a filter.
pub const TARGET: (usize, usize) = (100, 300);
pub const MAX_FILE_BYTES: u64 = 1 << 20;
const MAX_LOG_BYTES: u64 = 1 << 30;
const MAX_COMMITS_PER_REPO: usize = 500_000;
const MAX_DIFF_TREE_BYTES: u64 = 64 << 20;
const MAX_DIFF_BYTES: u64 = 8 << 20;
const MAX_INPUT_BYTES: u64 = 64 << 20;
const LOG_FORMAT: &str = "--format=%H%x00%P%x00%at%x00%aI%x00%B%x1e";

/// `YYYY-MM-DD` -> seconds since the epoch at 00:00:00Z of that day. Strict: ten characters,
/// a real calendar date.
pub fn parse_cutoff(s: &str) -> Result<i64, String> {
    let b = s.as_bytes();
    let shape = b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b.iter()
            .enumerate()
            .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit());
    if !shape {
        return Err(format!("--cutoff {s:?} is not YYYY-MM-DD"));
    }
    let num = |r: std::ops::Range<usize>| s[r].parse::<u32>().map_err(|e| e.to_string());
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    if !(1..=12).contains(&m) || d == 0 {
        return Err(format!("--cutoff {s:?} is not a calendar date"));
    }
    let days = days_from_civil(i64::from(y), m, d);
    if civil_from_days(days) != (i64::from(y), m, d) {
        return Err(format!("--cutoff {s:?} is not a calendar date"));
    }
    Ok(days * 86_400)
}

/// The [`FIX_WORDS`] a message holds, sorted and distinct.
pub fn fix_words(message: &str) -> Vec<&'static str> {
    let mut found = BTreeSet::new();
    for word in message.split(|c: char| !c.is_ascii_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let lower = word.to_ascii_lowercase();
        if let Some(w) = FIX_WORDS.iter().find(|w| **w == lower) {
            found.insert(*w);
        }
    }
    found.into_iter().collect()
}

/// The [`AGENT_NAMES`] named by the message's `Co-Authored-By:` trailers.
pub fn agent_trailers(message: &str) -> Vec<&'static str> {
    let mut found = BTreeSet::new();
    for line in message.lines() {
        let lower = line.trim().to_ascii_lowercase();
        if !lower.starts_with("co-authored-by:") {
            continue;
        }
        for name in AGENT_NAMES {
            if lower.contains(name) {
                found.insert(name);
            }
        }
    }
    found.into_iter().collect()
}

/// A whole-line comment in `lang`, judged on the trimmed line.
pub fn is_comment_line(lang: LangId, line: &str) -> bool {
    let t = line.trim();
    match lang {
        LangId::Python => t.starts_with('#'),
        LangId::Rust | LangId::Go | LangId::TypeScript | LangId::Swift => {
            t.starts_with("//")
                || t.starts_with("/*")
                || t.starts_with("* ")
                || t == "*"
                || t.starts_with("*/")
        }
    }
}

/// Whether the content's last line (under the runtime's line rule: a trailing `\n` opens no
/// line) is whitespace only. Composed rows ending so merge that line with the delimiter's
/// newline under BPE (`HANDOFF/needle-span-mac-2026-10-06.md` §2 note 2).
pub fn last_line_is_whitespace_only(content: &str) -> bool {
    let body = content.strip_suffix('\n').unwrap_or(content);
    if content.is_empty() {
        return false;
    }
    body.rsplit('\n')
        .next()
        .is_some_and(|l| l.chars().all(char::is_whitespace))
}

/// One `git diff-tree --raw -z` entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeEntry {
    pub old_mode: String,
    pub new_mode: String,
    pub old_sha: String,
    pub new_sha: String,
    pub status: char,
    pub old_path: String,
    pub path: String,
}

/// Parse `git diff-tree -r -z --raw -M --no-commit-id` output.
pub fn parse_diff_tree(out: &[u8]) -> Result<Vec<TreeEntry>, String> {
    let mut fields = out.split(|b| *b == 0).peekable();
    let mut entries = Vec::new();
    while let Some(meta) = fields.next() {
        if meta.is_empty() && fields.peek().is_none() {
            break;
        }
        let meta = String::from_utf8_lossy(meta);
        let meta = meta
            .strip_prefix(':')
            .ok_or_else(|| format!("diff-tree entry {meta:?} does not start with ':'"))?;
        let parts: Vec<&str> = meta.split_whitespace().collect();
        let [old_mode, new_mode, old_sha, new_sha, status] = parts.as_slice() else {
            return Err(format!("diff-tree entry {meta:?} is not five fields"));
        };
        let status = status
            .chars()
            .next()
            .ok_or_else(|| format!("diff-tree entry {meta:?} has no status"))?;
        let mut next_path = || {
            fields
                .next()
                .map(|p| String::from_utf8_lossy(p).into_owned())
                .ok_or_else(|| format!("diff-tree entry {meta:?} has no path"))
        };
        let first = next_path()?;
        let (old_path, path) = if matches!(status, 'R' | 'C') {
            (first, next_path()?)
        } else {
            (first.clone(), first)
        };
        entries.push(TreeEntry {
            old_mode: old_mode.to_string(),
            new_mode: new_mode.to_string(),
            old_sha: old_sha.to_string(),
            new_sha: new_sha.to_string(),
            status,
            old_path,
            path,
        });
    }
    Ok(entries)
}

/// What `git diff --unified=0` of two blobs says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineDiff {
    pub hunks: usize,
    pub removed: Vec<String>,
    pub added: Vec<String>,
    /// The first hunk's old and new start lines (1-based).
    pub old_start: Option<u64>,
    pub new_start: Option<u64>,
}

/// Parse a `--unified=0` diff. Everything before the first `@@` is header.
pub fn parse_unified0(text: &str) -> Result<LineDiff, String> {
    let mut d = LineDiff {
        hunks: 0,
        removed: Vec::new(),
        added: Vec::new(),
        old_start: None,
        new_start: None,
    };
    for line in text.split('\n') {
        if line.starts_with("@@ ") {
            d.hunks += 1;
            if d.hunks == 1 {
                let (o, n) = hunk_starts(line)?;
                d.old_start = Some(o);
                d.new_start = Some(n);
            }
        } else if d.hunks == 0 || line.starts_with('\\') {
            continue;
        } else if let Some(r) = line.strip_prefix('-') {
            d.removed.push(r.to_string());
        } else if let Some(a) = line.strip_prefix('+') {
            d.added.push(a.to_string());
        }
    }
    Ok(d)
}

/// `@@ -a[,b] +c[,d] @@ ...` -> (a, c).
fn hunk_starts(header: &str) -> Result<(u64, u64), String> {
    let bad = || format!("hunk header {header:?} is not '@@ -a[,b] +c[,d] @@'");
    let mut it = header.split(' ');
    let (Some("@@"), Some(old), Some(new)) = (it.next(), it.next(), it.next()) else {
        return Err(bad());
    };
    let start = |s: &str, sign: char| -> Result<u64, String> {
        let s = s.strip_prefix(sign).ok_or_else(bad)?;
        s.split(',').next().unwrap_or(s).parse().map_err(|_| bad())
    };
    Ok((start(old, '-')?, start(new, '+')?))
}

/// One parsed commit from `git log`.
#[derive(Clone, Debug)]
pub struct LogCommit {
    pub sha: String,
    pub parents: Vec<String>,
    pub author_time: i64,
    pub author_date: String,
    pub message: String,
}

/// Parse `git log` output in [`LOG_FORMAT`].
pub fn parse_log(out: &[u8]) -> Result<Vec<LogCommit>, String> {
    let mut commits = Vec::new();
    for record in out.split(|b| *b == 0x1e) {
        let record = match record.iter().position(|b| *b != b'\n') {
            Some(i) => &record[i..],
            None => continue,
        };
        let mut f = record.splitn(5, |b| *b == 0);
        let (Some(sha), Some(parents), Some(at), Some(ai), Some(body)) =
            (f.next(), f.next(), f.next(), f.next(), f.next())
        else {
            return Err(format!(
                "a git log record has fewer than five fields: {:?}",
                String::from_utf8_lossy(&record[..record.len().min(120)])
            ));
        };
        let sha = String::from_utf8_lossy(sha).into_owned();
        if !is_full_sha(&sha) {
            return Err(format!("git log gave {sha:?} as a commit sha"));
        }
        let parents = String::from_utf8_lossy(parents)
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let author_time = String::from_utf8_lossy(at)
            .trim()
            .parse::<i64>()
            .map_err(|e| format!("{sha}: author time: {e}"))?;
        commits.push(LogCommit {
            sha,
            parents,
            author_time,
            author_date: String::from_utf8_lossy(ai).into_owned(),
            message: String::from_utf8_lossy(body).into_owned(),
        });
        if commits.len() > MAX_COMMITS_PER_REPO {
            return Err(format!(
                "more than {MAX_COMMITS_PER_REPO} commits in one repository"
            ));
        }
    }
    Ok(commits)
}

/// A selected fix, before it becomes a row.
#[derive(Clone, Debug)]
pub struct Fix {
    pub path: String,
    pub language: LangId,
    pub pre_blob: String,
    pub post_blob: String,
    pub pre_content: String,
    pub post_content: String,
    pub line_pre: u64,
    pub line_post: u64,
    pub removed: String,
    pub added: String,
}

/// Where one commit landed.
pub enum Outcome {
    Excluded(&'static str),
    Selected(Box<Fix>),
}

/// The repository-side reads a classification needs, so the rules can be tested without git.
pub trait Reader {
    fn diff_tree(&self, parent: &str, sha: &str) -> Result<Vec<TreeEntry>, String>;
    fn blob_size(&self, sha: &str) -> Result<u64, String>;
    fn blob(&self, sha: &str) -> Result<Vec<u8>, String>;
    fn diff_blobs(&self, old: &str, new: &str) -> Result<String, String>;
}

/// The git CLI in one repository.
pub struct GitReader<'a> {
    pub repo: &'a Path,
}

fn git_err(e: GitError) -> String {
    e.to_string()
}

impl Reader for GitReader<'_> {
    fn diff_tree(&self, parent: &str, sha: &str) -> Result<Vec<TreeEntry>, String> {
        let out = gitcli::run(
            self.repo,
            &[
                "diff-tree",
                "-r",
                "-z",
                "--raw",
                "-M",
                "--no-commit-id",
                parent,
                sha,
            ],
            MAX_DIFF_TREE_BYTES,
        )
        .map_err(git_err)?;
        parse_diff_tree(&out)
    }

    fn blob_size(&self, sha: &str) -> Result<u64, String> {
        let out = gitcli::run(self.repo, &["cat-file", "-s", sha], 64).map_err(git_err)?;
        String::from_utf8_lossy(&out)
            .trim()
            .parse()
            .map_err(|e| format!("cat-file -s {sha}: {e}"))
    }

    fn blob(&self, sha: &str) -> Result<Vec<u8>, String> {
        gitcli::run(self.repo, &["cat-file", "blob", sha], MAX_FILE_BYTES).map_err(git_err)
    }

    fn diff_blobs(&self, old: &str, new: &str) -> Result<String, String> {
        let out = gitcli::run(
            self.repo,
            &[
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--unified=0",
                "--diff-algorithm=myers",
                "--full-index",
                old,
                new,
            ],
            MAX_DIFF_BYTES,
        )
        .map_err(git_err)?;
        String::from_utf8(out).map_err(|e| format!("diff of {old}..{new} is not UTF-8: {e}"))
    }
}

/// The per-repository context of a classification.
pub struct RepoRules<'a> {
    pub repo: &'a str,
    pub v5_files: &'a BTreeSet<(String, String)>,
    pub struck_paths: &'a [(String, String)],
}

/// Classify one commit. `Err` is a fault in this tool or the repository (a diff that does not
/// match the blobs it was taken from), never a commit that merely fails a rule.
pub fn classify(
    commit: &LogCommit,
    rules: &RepoRules<'_>,
    reader: &dyn Reader,
) -> Result<Outcome, String> {
    use Outcome::Excluded;
    let parent = match commit.parents.as_slice() {
        [] => return Ok(Excluded("root_commit")),
        [p] => p.as_str(),
        _ => return Ok(Excluded("merge_commit")),
    };
    if fix_words(&commit.message).is_empty() {
        return Ok(Excluded("not_fix_like"));
    }
    let entries = reader.diff_tree(parent, &commit.sha)?;
    let touches = |p: &str| {
        rules
            .v5_files
            .contains(&(rules.repo.to_string(), p.to_string()))
    };
    if entries
        .iter()
        .any(|e| touches(&e.old_path) || touches(&e.path))
    {
        return Ok(Excluded("v5_own_prose_file"));
    }
    let struck = |p: &str| own_repos::struck_path(rules.struck_paths, rules.repo, p).is_some();
    if entries
        .iter()
        .any(|e| struck(&e.old_path) || struck(&e.path))
    {
        return Ok(Excluded("struck_path"));
    }
    let e = match entries.as_slice() {
        [] => return Ok(Excluded("empty_diff")),
        [e] => e,
        _ => return Ok(Excluded("multi_file")),
    };
    if matches!(e.status, 'R' | 'C') {
        return Ok(Excluded("rename_or_copy"));
    }
    if e.status != 'M' {
        return Ok(Excluded("not_a_modification"));
    }
    let regular = |m: &str| m == "100644" || m == "100755";
    if !regular(&e.old_mode) || !regular(&e.new_mode) {
        return Ok(Excluded("not_a_regular_file"));
    }
    if e.old_mode != e.new_mode {
        return Ok(Excluded("mode_change"));
    }
    let Some(language) = language_from_path(&e.path) else {
        return Ok(Excluded("language_not_admitted"));
    };
    if reader.blob_size(&e.old_sha)? > MAX_FILE_BYTES
        || reader.blob_size(&e.new_sha)? > MAX_FILE_BYTES
    {
        return Ok(Excluded("file_too_large"));
    }
    let (pre, post) = (reader.blob(&e.old_sha)?, reader.blob(&e.new_sha)?);
    if pre.contains(&0) || post.contains(&0) {
        return Ok(Excluded("binary"));
    }
    let (Ok(pre), Ok(post)) = (String::from_utf8(pre), String::from_utf8(post)) else {
        return Ok(Excluded("not_utf8"));
    };
    let diff = parse_unified0(&reader.diff_blobs(&e.old_sha, &e.new_sha)?)?;
    if diff.hunks == 0 {
        return Ok(Excluded("empty_diff"));
    }
    if diff.hunks > 1 {
        return Ok(Excluded("multi_hunk"));
    }
    let ([removed], [added]) = (diff.removed.as_slice(), diff.added.as_slice()) else {
        return Ok(Excluded("not_one_line_each"));
    };
    let (Some(line_pre), Some(line_post)) = (diff.old_start, diff.new_start) else {
        return Err(format!("{}: a hunk with no start line", commit.sha));
    };
    let at = |content: &str, n: u64| -> Option<String> {
        let idx = usize::try_from(n).ok()?.checked_sub(1)?;
        content.split('\n').nth(idx).map(str::to_string)
    };
    if at(&pre, line_pre).as_deref() != Some(removed.as_str())
        || at(&post, line_post).as_deref() != Some(added.as_str())
    {
        return Err(format!(
            "{} {}: the diff's line {line_pre}/{line_post} is not the blobs' line there; \
             refusing rather than emitting a span that points at the wrong line",
            commit.sha, e.path
        ));
    }
    let squash = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    if squash(removed.as_str()) == squash(added.as_str()) {
        return Ok(Excluded("whitespace_only"));
    }
    if removed.trim().is_empty() || added.trim().is_empty() {
        return Ok(Excluded("blank_line_side"));
    }
    match (
        is_comment_line(language, removed),
        is_comment_line(language, added),
    ) {
        (true, true) => return Ok(Excluded("comment_only")),
        (true, false) | (false, true) => return Ok(Excluded("commented_out_or_in")),
        (false, false) => {}
    }
    Ok(Outcome::Selected(Box::new(Fix {
        path: e.path.clone(),
        language,
        pre_blob: e.old_sha.clone(),
        post_blob: e.new_sha.clone(),
        pre_content: pre,
        post_content: post,
        line_pre,
        line_post,
        removed: removed.clone(),
        added: added.clone(),
    })))
}

/// One held-out repository as the manifest pinned it.
#[derive(Clone, Debug)]
pub struct HeldOutRepo {
    pub repo: String,
    pub slug: String,
    pub head: String,
}

/// What the miner takes from an own-repos manifest.
#[derive(Clone, Debug)]
pub struct ManifestView {
    pub code_root: PathBuf,
    pub held: Vec<HeldOutRepo>,
    pub struck: Vec<(String, String)>,
}

/// The manifest's held-out repositories, its struck paths and code root, after checking that
/// each held-out row is one the rule would hold out.
pub fn read_manifest(doc: &Value) -> Result<ManifestView, String> {
    if doc["schema"] != own_repos::SCHEMA {
        return Err(format!(
            "the manifest's schema is {}, not {}",
            doc["schema"],
            own_repos::SCHEMA
        ));
    }
    let code_root = doc["code_root"]
        .as_str()
        .ok_or("the manifest has no code_root")?;
    let admitted = doc["admitted"]
        .as_array()
        .ok_or("the manifest has no admitted list")?;
    let mut held = Vec::new();
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
        if split_of(&slug) != split {
            return Err(format!(
                "{repo}: the manifest says {split}, but the rule puts {slug} in {}; refusing \
                 a manifest that disagrees with the rule",
                split_of(&slug)
            ));
        }
        if !is_full_sha(&head) {
            return Err(format!("{repo}: head {head:?} is not a commit sha"));
        }
        if own_repos::EXCLUDED_BY_THE_HUMAN.contains(&repo.as_str())
            || own_repos::STRUCK_REPOS.contains(&repo.as_str())
        {
            return Err(format!(
                "{repo} is admitted, but the human excluded or struck it"
            ));
        }
        if split == "heldout" {
            held.push(HeldOutRepo { repo, slug, head });
        }
    }
    let struck = doc["rules"]["struck_paths"]
        .as_array()
        .ok_or("the manifest has no rules.struck_paths")?
        .iter()
        .map(|s| match (s["repo"].as_str(), s["prefix"].as_str()) {
            (Some(r), Some(p)) => Ok((r.to_string(), p.to_string())),
            _ => Err(format!("a struck path {s} is not {{repo, prefix}}")),
        })
        .collect::<Result<Vec<_>, String>>()?;
    let pre: BTreeSet<(String, String)> = own_repos::STRUCK_PATHS
        .iter()
        .map(|(r, p)| (r.to_string(), p.to_string()))
        .collect();
    if !pre.iter().all(|s| struck.contains(s)) {
        return Err("the manifest's struck_paths omit one the human struck".to_string());
    }
    Ok(ManifestView {
        code_root: PathBuf::from(code_root),
        held,
        struck,
    })
}

/// own-prose-v1's `files.jsonl`: `(repo, path)` pairs.
pub fn read_v5_files(bytes: &[u8]) -> Result<BTreeSet<(String, String)>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("--v5-files: {e}"))?;
    let mut out = BTreeSet::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(line).map_err(|e| format!("--v5-files line {}: {e}", i + 1))?;
        match (v["repo"].as_str(), v["path"].as_str()) {
            (Some(r), Some(p)) => {
                out.insert((r.to_string(), p.to_string()));
            }
            _ => return Err(format!("--v5-files line {} has no repo/path", i + 1)),
        }
    }
    if out.is_empty() {
        return Err("--v5-files lists no file; an empty exclusion list excludes nothing".into());
    }
    Ok(out)
}

/// The command's inputs.
pub struct Args<'a> {
    pub manifest: &'a Path,
    pub cutoff: &'a str,
    pub cutoff_basis: &'a str,
    pub v5_files: &'a Path,
    pub out_dir: &'a Path,
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, String> {
    crate::files::read_bounded(path, MAX_INPUT_BYTES, "input")
}

#[derive(Default)]
struct Counts {
    scanned: usize,
    root_commits: usize,
    merge_commits: usize,
    after_cutoff: usize,
    fix_like_all: usize,
    fix_like_after_cutoff: usize,
    fix_like_after_cutoff_agent_trailer: usize,
    selected: usize,
    single_statement_before_cutoff: usize,
    reasons_after_cutoff: BTreeMap<&'static str, usize>,
    reasons_before_cutoff: BTreeMap<&'static str, usize>,
}

impl Counts {
    fn to_json(&self) -> Value {
        json!({
            "commits_scanned": self.scanned,
            "root_commits": self.root_commits,
            "merge_commits": self.merge_commits,
            "non_merge_after_cutoff": self.after_cutoff,
            "fix_like_all_dates": self.fix_like_all,
            "fix_like_after_cutoff": self.fix_like_after_cutoff,
            "fix_like_after_cutoff_with_agent_trailer": self.fix_like_after_cutoff_agent_trailer,
            "single_statement_fixes_after_cutoff": self.selected,
            "single_statement_fixes_before_cutoff_not_emitted": self.single_statement_before_cutoff,
            "fix_like_after_cutoff_excluded_by_reason": self.reasons_after_cutoff,
            "fix_like_before_cutoff_excluded_by_reason": self.reasons_before_cutoff,
        })
    }

    fn add(&mut self, o: &Counts) {
        self.scanned += o.scanned;
        self.root_commits += o.root_commits;
        self.merge_commits += o.merge_commits;
        self.after_cutoff += o.after_cutoff;
        self.fix_like_all += o.fix_like_all;
        self.fix_like_after_cutoff += o.fix_like_after_cutoff;
        self.fix_like_after_cutoff_agent_trailer += o.fix_like_after_cutoff_agent_trailer;
        self.selected += o.selected;
        self.single_statement_before_cutoff += o.single_statement_before_cutoff;
        for (k, v) in &o.reasons_after_cutoff {
            *self.reasons_after_cutoff.entry(*k).or_default() += v;
        }
        for (k, v) in &o.reasons_before_cutoff {
            *self.reasons_before_cutoff.entry(*k).or_default() += v;
        }
    }
}

/// One repository's selected rows (each with its author time) and counts.
struct Mined {
    rows: Vec<(i64, Value)>,
    counts: Counts,
}

/// Mine one repository: its rows and its counts.
fn mine_repo(
    held: &HeldOutRepo,
    repo_path: &Path,
    cutoff: i64,
    v5_files: &BTreeSet<(String, String)>,
    struck: &[(String, String)],
) -> Result<Mined, String> {
    let reader = GitReader { repo: repo_path };
    let out = gitcli::run(
        repo_path,
        &[
            "-c",
            "log.showSignature=false",
            "log",
            "--no-color",
            "--no-decorate",
            LOG_FORMAT,
            held.head.as_str(),
            "--",
        ],
        MAX_LOG_BYTES,
    )
    .map_err(git_err)?;
    let commits = parse_log(&out)?;
    let rules = RepoRules {
        repo: &held.repo,
        v5_files,
        struck_paths: struck,
    };
    let mut c = Counts {
        scanned: commits.len(),
        ..Counts::default()
    };
    let mut rows = Vec::new();
    for commit in &commits {
        let after = commit.author_time >= cutoff;
        match commit.parents.len() {
            0 => c.root_commits += 1,
            1 => {}
            _ => c.merge_commits += 1,
        }
        if commit.parents.len() == 1 && after {
            c.after_cutoff += 1;
        }
        let fix_like = commit.parents.len() == 1 && !fix_words(&commit.message).is_empty();
        if !fix_like {
            continue;
        }
        c.fix_like_all += 1;
        let agents = agent_trailers(&commit.message);
        if after {
            c.fix_like_after_cutoff += 1;
            if !agents.is_empty() {
                c.fix_like_after_cutoff_agent_trailer += 1;
            }
        }
        let outcome = classify(commit, &rules, &reader)?;
        match (outcome, after) {
            (Outcome::Excluded(r), true) => *c.reasons_after_cutoff.entry(r).or_default() += 1,
            (Outcome::Excluded(r), false) => *c.reasons_before_cutoff.entry(r).or_default() += 1,
            // Read to be counted, never emitted: the date rule is what keeps it out.
            (Outcome::Selected(_), false) => c.single_statement_before_cutoff += 1,
            (Outcome::Selected(fix), true) => {
                c.selected += 1;
                let subject = commit.message.lines().next().unwrap_or("").to_string();
                let row = json!({
                    "id": format!("{}@{}", held.slug, commit.sha),
                    "repo": held.repo,
                    "repo_slug": held.slug,
                    "commit": commit.sha,
                    "parent": commit.parents[0],
                    "author_date": commit.author_date,
                    "author_time": commit.author_time,
                    "path": fix.path,
                    "language": fix.language.as_str(),
                    "language_in_defect_class_pool": DEFECT_CLASS_POOL_LANGUAGES.contains(&fix.language),
                    "pre_blob": fix.pre_blob,
                    "post_blob": fix.post_blob,
                    "pre_content": fix.pre_content,
                    "post_content": fix.post_content,
                    "changed_line_pre": fix.line_pre,
                    "changed_line_post": fix.line_post,
                    "removed_text": fix.removed,
                    "added_text": fix.added,
                    "subject": subject,
                    "fix_words": fix_words(&commit.message),
                    "agent_coauthor_trailer": !agents.is_empty(),
                    "agent_trailer_names": agents,
                    "pre_last_line_whitespace_only": last_line_is_whitespace_only(&fix.pre_content),
                    "label": {"kind": "span", "pre_line": fix.line_pre, "post_line": fix.line_post},
                    "mutation_class": Value::Null,
                    "mutation_class_note": "[U] a natural bug has no mutation class; the span is the label",
                });
                rows.push((commit.author_time, row));
            }
        }
    }
    Ok(Mined { rows, counts: c })
}

/// `qd-prep natural-bugs`.
pub fn run(args: &Args<'_>) -> Result<String, String> {
    let cutoff = parse_cutoff(args.cutoff)?;
    if args.cutoff_basis.trim().is_empty() {
        return Err("--cutoff-basis is empty; say where the cutoff date comes from".into());
    }
    check_held_out_path(args.out_dir)?;
    let partial = crate::files::partial_path(args.out_dir)?;
    for p in [args.out_dir, partial.as_path()] {
        if p.exists() {
            return Err(format!("{} exists; refusing to overwrite it", p.display()));
        }
    }
    let manifest_bytes = read_bounded(args.manifest)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("{}: {e}", args.manifest.display()))?;
    let ManifestView {
        code_root,
        held,
        struck,
    } = read_manifest(&manifest)?;
    if held.is_empty() {
        return Err("the manifest holds out no repository; there is nothing to mine".into());
    }
    let v5_bytes = read_bounded(args.v5_files)?;
    let v5_files = read_v5_files(&v5_bytes)?;

    let mut all_rows: Vec<(String, i64, String, Value)> = Vec::new();
    let mut per_repo = BTreeMap::new();
    let mut total = Counts::default();
    for h in &held {
        let path = code_root.join(&h.repo);
        let Mined { rows, counts } = mine_repo(h, &path, cutoff, &v5_files, &struck)?;
        total.add(&counts);
        per_repo.insert(h.repo.clone(), counts.to_json());
        for (t, row) in rows {
            let sha = row["commit"].as_str().unwrap_or_default().to_string();
            all_rows.push((h.repo.clone(), t, sha, row));
        }
    }
    all_rows.sort_by(|a, b| (&a.0, a.1, &a.2).cmp(&(&b.0, b.1, &b.2)));

    let mut jsonl = Vec::new();
    let mut by_language: BTreeMap<String, usize> = BTreeMap::new();
    let (mut with_agent, mut ws_last) = (0usize, 0usize);
    for (_, _, _, row) in &all_rows {
        *by_language
            .entry(row["language"].as_str().unwrap_or("?").to_string())
            .or_default() += 1;
        with_agent += usize::from(row["agent_coauthor_trailer"].as_bool() == Some(true));
        ws_last += usize::from(row["pre_last_line_whitespace_only"].as_bool() == Some(true));
        serde_json::to_writer(&mut jsonl, row).map_err(|e| e.to_string())?;
        jsonl.push(b'\n');
    }
    let n = all_rows.len();
    let met = (TARGET.0..=TARGET.1).contains(&n);
    let report = json!({
        "schema": SCHEMA,
        "tool": "qd-prep natural-bugs",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "written_at": utc_now(),
        "never_train": "held-out evaluation set (CLAUDE.md rule 3); its path carries a held-out marker",
        "manifest": {"path": args.manifest.display().to_string(), "sha256": sha256_hex(&manifest_bytes)},
        "heldout_repos": held.iter().map(|h| json!({"repo": h.repo, "slug": h.slug, "head": h.head})).collect::<Vec<_>>(),
        "cutoff": {
            "flag": args.cutoff,
            "epoch_utc": cutoff,
            "basis": args.cutoff_basis,
            "rule": "a commit is after the cutoff iff its author time (%at) >= 00:00:00Z of the flag's date",
        },
        "v5_files": {
            "path": args.v5_files.display().to_string(),
            "sha256": sha256_hex(&v5_bytes),
            "files": v5_files.len(),
        },
        "rules": {
            "fix_like": format!("the commit message holds a word (maximal ASCII alphanumeric run, case-insensitive) in {FIX_WORDS:?}"),
            "single_statement": "one parent; one file modified in place (no rename/copy, regular file, same mode); Lappi-admitted language (qd_lang::language_from_path); <= 1 MiB, no NUL, UTF-8; git diff --unified=0 --diff-algorithm=myers of the two blobs has one hunk, one removed and one added line; not whitespace-only, no blank side, neither side a whole-line comment (trailing comments not detected)",
            "agent_trailer": format!("a Co-Authored-By: line naming one of {AGENT_NAMES:?}"),
            "struck_paths": struck.iter().map(|(r, p)| format!("{r}:{p}")).collect::<Vec<_>>(),
            "first_reason_wins": true,
        },
        "per_repo": per_repo,
        "totals": total.to_json(),
        "selected": {
            "rows": n,
            "by_language": by_language,
            "with_agent_coauthor_trailer": with_agent,
            "pre_last_line_whitespace_only": ws_last,
        },
        "target": {"min": TARGET.0, "max": TARGET.1, "met": met,
                   "note": "reported, not enforced: a short set is reported short; no filter is loosened to fill it, and a long one is not subsampled here"},
        "output": {"file": "natural-bugs.jsonl", "sha256": sha256_hex(&jsonl), "rows": n},
        "open": [
            "mutation_class is [U] for every row: a natural bug has no mutation class, so the span is the label",
            "single-statement is line-based in v1; a tree-sitter statement-level check is a proposal only",
            "no file-mode span renderer exists; whoever renders rows must handle a whitespace-only last line (HANDOFF/needle-span-mac-2026-10-06.md §2 note 2)",
        ],
    });
    let mut report_bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    report_bytes.push(b'\n');

    // This subcommand makes the held-out directory it writes under; the write itself is whole
    // or not at all.
    if let Some(parent) = args.out_dir.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    crate::files::write_new_dir(args.out_dir, |partial| {
        // A symlinked parent could land the files somewhere unmarked; check where they go.
        let real = fs::canonicalize(partial).map_err(|e| format!("{}: {e}", partial.display()))?;
        check_held_out_path(&real)?;
        crate::files::write_synced_new(&partial.join("natural-bugs.jsonl"), &jsonl)?;
        crate::files::write_synced_new(&partial.join("report.json"), &report_bytes)
    })?;
    Ok(format!(
        "qd-prep natural-bugs: {} held-out repo(s), {} commits scanned, {} fix-like after the \
         cutoff, {n} single-statement fixes (target {}-{}: {}) -> {}",
        held.len(),
        total.scanned,
        total.fix_like_after_cutoff,
        TARGET.0,
        TARGET.1,
        if met {
            "met"
        } else if n < TARGET.0 {
            "SHORT"
        } else {
            "OVER"
        },
        args.out_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cutoff_is_a_strict_calendar_date() {
        assert_eq!(parse_cutoff("2026-02-01"), Ok(1_769_904_000));
        assert_eq!(parse_cutoff("1970-01-01"), Ok(0));
        for bad in [
            "",
            "2026-2-01",
            "2026-02-30",
            "2026-13-01",
            "2026-00-10",
            "2025-02-29",
            "2026/02/01",
            "20260201",
            "2026-02-01T00:00",
        ] {
            assert!(parse_cutoff(bad).is_err(), "{bad:?} accepted");
        }
        assert!(parse_cutoff("2024-02-29").is_ok(), "a leap day");
    }

    #[test]
    fn fix_words_are_whole_words_in_any_case() {
        assert_eq!(fix_words("Fix off-by-one in parser"), ["fix"]);
        assert_eq!(fix_words("fix(parser): bugfix for #12"), ["bugfix", "fix"]);
        assert_eq!(fix_words("HOTFIX"), ["hotfix"]);
        assert!(fix_words("add prefix and fixture; debug output").is_empty());
        assert!(
            fix_words("handle error issue").is_empty(),
            "error/issue are not fix words"
        );
    }

    #[test]
    fn agent_trailers_are_co_authored_by_lines_only() {
        let m = "fix x\n\nmentions claude in the body\n\nCo-Authored-By: Claude Opus <noreply@anthropic.com>\n";
        assert_eq!(agent_trailers(m), ["anthropic", "claude"]);
        assert!(agent_trailers("fix x\n\nwritten with claude\n").is_empty());
        assert!(agent_trailers("fix\n\nCo-authored-by: Jane <j@example.com>\n").is_empty());
    }

    #[test]
    fn comment_lines_per_language() {
        assert!(is_comment_line(LangId::Python, "   # note"));
        assert!(!is_comment_line(LangId::Python, "x = 1  # note"));
        assert!(is_comment_line(LangId::Rust, "// note"));
        assert!(is_comment_line(LangId::Go, " * continuation"));
        assert!(
            !is_comment_line(LangId::Rust, "*ptr = 1;"),
            "a deref is code"
        );
        assert!(!is_comment_line(LangId::TypeScript, "let a = b / c;"));
    }

    #[test]
    fn a_whitespace_only_last_line_is_detected_under_the_runtime_line_rule() {
        assert!(last_line_is_whitespace_only("a\n    \n"));
        assert!(last_line_is_whitespace_only("a\n\n"));
        assert!(!last_line_is_whitespace_only("a\nb\n"));
        assert!(!last_line_is_whitespace_only("a\nb"));
        assert!(!last_line_is_whitespace_only(""));
    }

    #[test]
    fn diff_tree_entries_parse_with_renames_taking_two_paths() {
        let z = |s: &str| s.replace('|', "\0");
        let m = ":100644 100644 aaaa bbbb M|src/a.rs|";
        let r = ":100644 100644 aaaa bbbb R087|old.rs|new.rs|:100644 100644 cccc dddd M|b.py|";
        let e = parse_diff_tree(z(m).as_bytes()).unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!((e[0].status, e[0].path.as_str()), ('M', "src/a.rs"));
        let e = parse_diff_tree(z(r).as_bytes()).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(
            (e[0].old_path.as_str(), e[0].path.as_str()),
            ("old.rs", "new.rs")
        );
        assert_eq!(e[1].path, "b.py");
        assert!(parse_diff_tree(b"").unwrap().is_empty());
        assert!(parse_diff_tree(b"garbage\0").is_err());
    }

    #[test]
    fn a_unified0_diff_counts_hunks_and_lines_after_its_header() {
        let one = "diff --git a/x b/x\nindex 1..2\n--- a/x\n+++ b/x\n@@ -3 +3 @@\n-    return a - 1\n+    return a + 1\n";
        let d = parse_unified0(one).unwrap();
        assert_eq!((d.hunks, d.old_start, d.new_start), (1, Some(3), Some(3)));
        assert_eq!(d.removed, ["    return a - 1"]);
        assert_eq!(d.added, ["    return a + 1"]);
        let two = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n@@ -9,2 +9,0 @@\n--c\n-d\n";
        let d = parse_unified0(two).unwrap();
        assert_eq!(d.hunks, 2);
        assert_eq!(
            d.removed,
            ["a", "-c", "d"],
            "a removed line starting '-' is content"
        );
        let eof =
            "@@ -2 +2 @@\n-x\n\\ No newline at end of file\n+y\n\\ No newline at end of file\n";
        let d = parse_unified0(eof).unwrap();
        assert_eq!((d.removed.len(), d.added.len()), (1, 1));
    }

    #[test]
    fn a_log_record_parses_and_a_short_one_is_refused() {
        let sha = "a".repeat(40);
        let p = "b".repeat(40);
        let rec = format!(
            "{sha}\0{p}\0{}\02026-03-01T10:00:00+00:00\0fix x\n\nbody\n\x1e\n",
            1_772_359_200
        );
        let c = parse_log(rec.as_bytes()).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].parents.len(), c[0].author_time), (1, 1_772_359_200));
        assert!(c[0].message.starts_with("fix x"));
        assert!(parse_log(format!("{sha}\0only\x1e").as_bytes()).is_err());
        assert!(parse_log(b"\n").unwrap().is_empty());
    }

    /// A fake repository for the rules, one commit at a time.
    struct Fake {
        entries: Vec<TreeEntry>,
        pre: Vec<u8>,
        post: Vec<u8>,
        diff: String,
    }

    impl Reader for Fake {
        fn diff_tree(&self, _: &str, _: &str) -> Result<Vec<TreeEntry>, String> {
            Ok(self.entries.clone())
        }
        fn blob_size(&self, sha: &str) -> Result<u64, String> {
            let n = if sha == "old" {
                self.pre.len()
            } else {
                self.post.len()
            };
            Ok(n as u64)
        }
        fn blob(&self, sha: &str) -> Result<Vec<u8>, String> {
            Ok(if sha == "old" {
                self.pre.clone()
            } else {
                self.post.clone()
            })
        }
        fn diff_blobs(&self, _: &str, _: &str) -> Result<String, String> {
            Ok(self.diff.clone())
        }
    }

    fn entry(status: char, path: &str) -> TreeEntry {
        TreeEntry {
            old_mode: "100644".into(),
            new_mode: "100644".into(),
            old_sha: "old".into(),
            new_sha: "new".into(),
            status,
            old_path: path.into(),
            path: path.into(),
        }
    }

    fn commit(msg: &str, parents: usize) -> LogCommit {
        LogCommit {
            sha: "c".repeat(40),
            parents: (0..parents).map(|i| format!("{i:040}")).collect(),
            author_time: 0,
            author_date: String::new(),
            message: msg.into(),
        }
    }

    fn outcome(c: &LogCommit, f: &Fake) -> &'static str {
        let v5: BTreeSet<(String, String)> = [("r".to_string(), "README.py".to_string())]
            .into_iter()
            .collect();
        let struck = [("r".to_string(), "docs/business/".to_string())];
        let rules = RepoRules {
            repo: "r",
            v5_files: &v5,
            struck_paths: &struck,
        };
        match classify(c, &rules, f).unwrap() {
            Outcome::Excluded(r) => r,
            Outcome::Selected(_) => "selected",
        }
    }

    fn one_line(path: &str, old: &str, new: &str) -> Fake {
        Fake {
            entries: vec![entry('M', path)],
            pre: format!("def f(a):\n{old}\n").into_bytes(),
            post: format!("def f(a):\n{new}\n").into_bytes(),
            diff: format!("--- a\n+++ b\n@@ -2 +2 @@\n-{old}\n+{new}\n"),
        }
    }

    #[test]
    fn the_single_line_filter_keeps_one_real_change_and_names_every_refusal() {
        let fix = commit("fix off by one", 1);
        let good = one_line("a.py", "    return a - 1", "    return a + 1");
        assert_eq!(outcome(&fix, &good), "selected");
        assert_eq!(outcome(&commit("tweak", 1), &good), "not_fix_like");
        assert_eq!(outcome(&commit("fix", 0), &good), "root_commit");
        assert_eq!(outcome(&commit("fix", 2), &good), "merge_commit");
        let mut two_files = one_line("a.py", "x", "y");
        two_files.entries.push(entry('M', "b.py"));
        assert_eq!(outcome(&fix, &two_files), "multi_file");
        let mut empty = one_line("a.py", "x", "y");
        empty.entries.clear();
        assert_eq!(outcome(&fix, &empty), "empty_diff");
        let mut renamed = one_line("a.py", "x", "y");
        renamed.entries[0].status = 'R';
        assert_eq!(outcome(&fix, &renamed), "rename_or_copy");
        let mut added = one_line("a.py", "x", "y");
        added.entries[0].status = 'A';
        assert_eq!(outcome(&fix, &added), "not_a_modification");
        let mut link = one_line("a.py", "x", "y");
        link.entries[0].new_mode = "120000".into();
        assert_eq!(outcome(&fix, &link), "not_a_regular_file");
        let mut exec = one_line("a.py", "x", "y");
        exec.entries[0].new_mode = "100755".into();
        assert_eq!(outcome(&fix, &exec), "mode_change");
        assert_eq!(
            outcome(&fix, &one_line("a.txt", "x", "y")),
            "language_not_admitted"
        );
        assert_eq!(
            outcome(&fix, &one_line("README.py", "x", "y")),
            "v5_own_prose_file"
        );
        assert_eq!(
            outcome(&fix, &one_line("docs/business/p.py", "x", "y")),
            "struck_path"
        );
        let mut binary = one_line("a.py", "x", "y");
        binary.post.push(0);
        assert_eq!(outcome(&fix, &binary), "binary");
        let mut latin1 = one_line("a.py", "x", "y");
        latin1.post = vec![b'x', 0xe9, b'\n'];
        assert_eq!(outcome(&fix, &latin1), "not_utf8");
        let mut two_hunks = one_line("a.py", "x", "y");
        two_hunks.diff.push_str("@@ -9 +9 @@\n-p\n+q\n");
        assert_eq!(outcome(&fix, &two_hunks), "multi_hunk");
        let mut insert = one_line("a.py", "x", "y");
        insert.diff = "@@ -1,0 +2 @@\n+y\n".into();
        assert_eq!(outcome(&fix, &insert), "not_one_line_each");
        let mut nohunk = one_line("a.py", "x", "y");
        nohunk.diff = String::new();
        assert_eq!(outcome(&fix, &nohunk), "empty_diff");
        assert_eq!(
            outcome(&fix, &one_line("a.py", "x = 1", "x =  1")),
            "whitespace_only"
        );
        assert_eq!(
            outcome(&fix, &one_line("a.py", "x = 1", "   ")),
            "blank_line_side"
        );
        assert_eq!(
            outcome(&fix, &one_line("a.py", "# a", "# b")),
            "comment_only"
        );
        assert_eq!(
            outcome(&fix, &one_line("a.py", "x = 1", "# x = 1")),
            "commented_out_or_in"
        );
    }

    #[test]
    fn a_diff_that_does_not_match_its_blobs_is_an_error_not_a_row() {
        let mut lie = one_line("a.py", "x", "y");
        lie.diff = "@@ -1 +1 @@\n-x\n+y\n".into();
        let v5 = BTreeSet::new();
        let rules = RepoRules {
            repo: "r",
            v5_files: &v5,
            struck_paths: &[],
        };
        assert!(classify(&commit("fix", 1), &rules, &lie).is_err());
    }
}
