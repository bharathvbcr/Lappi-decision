//! v5 prose-noul route (i): Markdown prose the human authored in their own repositories.
//!
//! `campaign/v5-preregistered.DRAFT.json` `data.sources[3]` and Fable's v5 review section 2.9:
//! **own-repo prose is text the user authored or had written in their own repositories, not text
//! hosted there.** This module applies that rule and writes prose *units* -- `(repo, path, form,
//! text)` -- never rows: `qd_data.defect_class.prose_defect_row`, the one "prose text -> noul row"
//! builder the contrast route also uses, turns each unit into a row at load time.
//!
//! 1. **Repo admission.** A repository of the inventory (`AUDIT/v5-plan-2026-10-02/
//!    own_repo_inventory.json`) is admitted when its `.git/config` -- re-read here, not taken
//!    from the inventory -- names github owner `bharathvbcr` in a remote and no remote names any
//!    other owner (a second owner is a fork: `devtools/DevPrism`'s `delibae/claude-prism`). A
//!    repo with no remote is out (ownership is not readable), and so is every `--exclude-repo`
//!    (the human's tick: "All but Lappi-decision"). `--expect-admitted` refuses a run whose count
//!    differs from the ticked list's.
//! 2. **Tree pruning.** Nested git trees (any other inventory repo, any directory holding
//!    `.git`), hidden and build directories, `vendor/ external/ third_party/ node_modules/
//!    site-packages/`, any directory below the root holding its own LICENSE/COPYING/NOTICE, any
//!    file with a Copyright, (c)-sign or SPDX line that names no owner token, any file opening
//!    with YAML front matter (model cards among them), and any Markdown byte-identical to a file
//!    in a repository the inventory gives another owner.
//! 3. **Content.** Markdown only. Fenced and indented code, tables, HTML, headings, list items,
//!    block quotes, link-only lines and trailers are dropped; a paragraph is a run of the plain
//!    lines left, joined with single spaces, of at least 25 words, at most `MAX_PARAGRAPH_CHARS`
//!    characters, at least 90% ASCII, with no invisible-format codepoint. A question is a
//!    sentence ending in `?`, 6-60 words, from any kept line (headings and list items included,
//!    their markers trimmed), under the same ASCII and codepoint rules. Each distinct text is one
//!    unit.
//! 4. **Selection.** Every distinct question first, then paragraphs round-robin over repos in a
//!    keyed order, to `--target` units, at most `--per-repo-cap` per repo. Fewer than `--floor`
//!    refuses the run.
//! 5. **Record.** `files.jsonl` lists every file read (repo, path, sha256, words), so the human
//!    can strike any; `units.jsonl` the units; `manifest.json` pins both by sha256.
//!
//! Commit bodies are not read: git runs only in the lane's own worktree
//! (GAP-L-V5PLAN-OWN-REPO-COMMIT-BODIES-NOT-MEASURED-2026-10-02).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use qd_mutate::manifest::sha256_hex;

use crate::hunk::{has_char_in, order_key, short_hash};

pub const SCHEMA: &str = "qd-own-prose/v1";
/// `qd_data.sources.OWN_REPOS_SOURCE_ID`; the loader refuses a manifest naming another.
pub const SOURCE_ID: &str = "bharathvbcr/own-repositories";
/// The licence the human's 2026-09-28 approval gives this source (`qd_data.licences`).
pub const LICENCE: &str = "owner-granted";
pub const OWNER: &str = "bharathvbcr";
pub const FORM_QUESTION: &str = "question";
pub const FORM_PARAGRAPH: &str = "paragraph";
const SOURCE: &str = "own-prose";

pub const MIN_PARAGRAPH_WORDS: usize = 25;
pub const QUESTION_WORDS: (usize, usize) = (6, 60);
pub const MIN_ASCII_PERMILLE: usize = 900;
/// A bound, not a rule from the DRAFT: v3b's prose band tops out at 1,400 characters, and a
/// paragraph several times that is a document, not a paragraph.
pub const MAX_PARAGRAPH_CHARS: usize = 2_000;
const MAX_FILE_BYTES: u64 = 2_000_000;
const MAX_FILES: usize = 100_000;

/// The owner's name as it appears in a copyright line of their own text.
const OWNER_TOKENS: [&str; 3] = ["bharath", "vaddaram", "bharathvbcr"];

/// The inventory script's excluded directories, plus the vendored-tree names section 2.9 names.
const SKIP_DIRS: [&str; 33] = [
    ".git",
    "node_modules",
    "vendor",
    ".venv",
    "venv",
    "env",
    "target",
    "build",
    "dist",
    "external",
    "third_party",
    "third-party",
    "site-packages",
    "__pycache__",
    ".next",
    "Pods",
    "DerivedData",
    ".build",
    ".gradle",
    ".idea",
    ".cache",
    "checkpoints",
    "worktrees",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "coverage",
    "out",
    "bin",
    "obj",
    "vendored",
    "deps",
];
const LICENCE_FILE_PREFIXES: [&str; 4] = ["LICENSE", "LICENCE", "COPYING", "NOTICE"];
const TRAILERS: [&str; 9] = [
    "co-authored-by:",
    "signed-off-by:",
    "reviewed-by:",
    "acked-by:",
    "change-id:",
    "refs:",
    "fixes:",
    "closes:",
    "see-also:",
];

#[derive(Deserialize)]
struct Inventory {
    code_root: String,
    repos: Vec<InventoryRepo>,
}

#[derive(Deserialize)]
struct InventoryRepo {
    repo: String,
    owner: String,
}

#[derive(Serialize)]
pub struct AdmittedRepo {
    repo: String,
    remotes: Vec<String>,
}

#[derive(Serialize)]
struct FileRecord<'a> {
    repo: &'a str,
    path: &'a str,
    sha256: &'a str,
    words: usize,
}

#[derive(Serialize, Clone)]
struct Unit {
    id: String,
    repo: String,
    path: String,
    file_sha256: String,
    form: &'static str,
    text: String,
    words: usize,
}

#[derive(Serialize)]
struct Pinned {
    file: &'static str,
    sha256: String,
    count: usize,
}

#[derive(Serialize)]
struct Manifest {
    schema: &'static str,
    tool: &'static str,
    tool_version: &'static str,
    seed: u64,
    source_id: &'static str,
    licence: &'static str,
    code_root: String,
    inventory: BTreeMap<&'static str, String>,
    allowlist_sha256: String,
    invisible_format_ranges: Vec<(u32, u32)>,
    admitted_repos: Vec<AdmittedRepo>,
    excluded_repos: BTreeMap<String, String>,
    rules: serde_json::Value,
    files: Pinned,
    units: Pinned,
    target: usize,
    floor: usize,
    per_repo_cap: usize,
    totals: serde_json::Value,
    skipped: BTreeMap<String, u64>,
    commit_bodies: &'static str,
}

pub struct Args<'a> {
    pub inventory: &'a Path,
    pub allowlist: &'a Path,
    pub exclude_repos: &'a [PathBuf],
    pub expect_admitted: usize,
    pub target: usize,
    pub floor: usize,
    pub per_repo_cap: usize,
    pub seed: u64,
    pub out: &'a Path,
}

/// What a word is: `[A-Za-z][A-Za-z'\-]+`, `qd_data.defect_class.OWN_PROSE_WORD`, scanned the way
/// a leftmost-first regex finds it.
pub fn words(text: &str) -> usize {
    let b = text.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_ascii_alphabetic() || b[j] == b'\'' || b[j] == b'-') {
                j += 1;
            }
            if j - i >= 2 {
                n += 1;
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    n
}

/// Characters that are ASCII, per thousand characters (`qd_data.defect_class
/// .own_prose_ascii_ratio`, in integer permille so the comparison cannot round two ways).
pub fn ascii_permille(text: &str) -> usize {
    let total = text.chars().count();
    if total == 0 {
        return 0;
    }
    text.chars().filter(char::is_ascii).count() * 1000 / total
}

/// The github owners a `.git/config`'s remote URLs name, and every URL. A `.git` file (a linked
/// worktree or submodule) is followed to its common directory's config.
pub fn remotes(repo: &Path) -> Result<Vec<String>> {
    let git = repo.join(".git");
    let config = if git.is_dir() {
        git.join("config")
    } else if git.is_file() {
        let line = fs::read_to_string(&git)?;
        let gitdir = line
            .trim()
            .strip_prefix("gitdir:")
            .map(str::trim)
            .unwrap_or_default();
        let dir = repo.join(gitdir);
        let common = dir.join("commondir");
        let base = if common.exists() {
            dir.join(fs::read_to_string(common)?.trim())
        } else {
            dir
        };
        base.join("config")
    } else {
        return Ok(Vec::new());
    };
    if !config.exists() {
        return Ok(Vec::new());
    }
    let text =
        fs::read_to_string(&config).with_context(|| format!("reading {}", config.display()))?;
    Ok(text
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            let (k, v) = t.split_once('=')?;
            (k.trim() == "url").then(|| v.trim().to_string())
        })
        .collect())
}

/// The github owner a remote URL names, or `None` for a URL that is not github's.
pub fn github_owner(url: &str) -> Option<&str> {
    let rest = url.split_once("github.com").map(|(_, r)| r)?;
    let rest = rest.strip_prefix(':').or_else(|| rest.strip_prefix('/'))?;
    let owner = rest.split('/').next()?;
    (!owner.is_empty()).then_some(owner)
}

/// Section 2.9 (1): admitted iff a remote names `OWNER` and no remote names anything else.
pub fn admission(urls: &[String]) -> std::result::Result<(), String> {
    if urls.is_empty() {
        return Err("no_remote".to_string());
    }
    let mut own = false;
    for url in urls {
        match github_owner(url) {
            Some(o) if o.eq_ignore_ascii_case(OWNER) => own = true,
            Some(o) => return Err(format!("fork_or_other_owner:{o}")),
            None => return Err("non_github_remote".to_string()),
        }
    }
    if own {
        Ok(())
    } else {
        Err("no_owner_remote".to_string())
    }
}

/// A line that carries a licence or copyright claim, and whether it names the owner.
fn foreign_claim(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let claim = lower.contains("copyright")
        || line.contains('\u{a9}')
        || lower.contains("spdx-license-identifier");
    claim && !OWNER_TOKENS.iter().any(|t| lower.contains(t))
}

fn is_link_only(t: &str) -> bool {
    let mut rest = t.trim();
    if rest.starts_with("http://") || rest.starts_with("https://") {
        return !rest.contains(' ');
    }
    let mut any = false;
    while !rest.is_empty() {
        let s = rest.strip_prefix('!').unwrap_or(rest);
        let Some(s) = s.strip_prefix('[') else {
            return false;
        };
        let Some(close) = s.find("](") else {
            return false;
        };
        let after = &s[close + 2..];
        let Some(end) = after.find(')') else {
            return false;
        };
        rest = after[end + 1..].trim_start();
        any = true;
    }
    any
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A plain prose line: it may join a paragraph.
    Plain,
    /// A heading, list item or block quote: not a paragraph line, but a question may sit in it.
    Marked,
    /// Code, a table, HTML, a link-only line or a trailer: nothing is read from it.
    Dropped,
}

fn classify(line: &str) -> Kind {
    if line.starts_with("    ") || line.starts_with('\t') {
        return Kind::Dropped;
    }
    let t = line.trim();
    let lower = t.to_ascii_lowercase();
    if t.starts_with('|')
        || t.starts_with('<')
        || is_link_only(t)
        || TRAILERS.iter().any(|p| lower.starts_with(p))
        || (t.contains('<') && t.contains('>') && t.contains("</"))
    {
        return Kind::Dropped;
    }
    let numbered = t
        .split_once(". ")
        .is_some_and(|(n, _)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
    if t.starts_with('#')
        || t.starts_with('>')
        || t.starts_with("- ")
        || t.starts_with("* ")
        || t.starts_with("+ ")
        || numbered
        || t.starts_with("---")
        || t.starts_with("===")
    {
        return Kind::Marked;
    }
    Kind::Plain
}

/// The paragraphs and question sentences of one Markdown text, before the per-unit rules.
pub fn extract(text: &str) -> (Vec<String>, Vec<String>) {
    let mut paragraphs = Vec::new();
    let mut questions = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    let mut fence: Option<&str> = None;
    // A paragraph's questions are read from its joined lines, so a question wrapped across two
    // source lines is one sentence rather than two fragments.
    let flush =
        |block: &mut Vec<&str>, paragraphs: &mut Vec<String>, questions: &mut Vec<String>| {
            if !block.is_empty() {
                let joined = block.iter().map(|l| l.trim()).collect::<Vec<_>>().join(" ");
                questions.extend(question_sentences(&joined));
                paragraphs.push(joined);
                block.clear();
            }
        };
    for line in text.lines() {
        let t = line.trim_start();
        let opener = if t.starts_with("```") {
            Some("```")
        } else if t.starts_with("~~~") {
            Some("~~~")
        } else {
            None
        };
        if let Some(f) = fence {
            if opener == Some(f) {
                fence = None;
            }
            continue;
        }
        if let Some(f) = opener {
            flush(&mut block, &mut paragraphs, &mut questions);
            fence = Some(f);
            continue;
        }
        if line.trim().is_empty() {
            flush(&mut block, &mut paragraphs, &mut questions);
            continue;
        }
        let kind = classify(line);
        if kind == Kind::Plain {
            block.push(line);
        } else {
            flush(&mut block, &mut paragraphs, &mut questions);
            if kind == Kind::Marked {
                questions.extend(question_sentences(line));
            }
        }
    }
    flush(&mut block, &mut paragraphs, &mut questions);
    (paragraphs, questions)
}

/// Whether a `.`, `!` or `?` followed by `next` ends a sentence there. A mark followed by a
/// letter, digit or `=` is inside a token (`/synthesis?q=1`, `v1.2`, `e.g.x`), not an end.
fn ends_sentence(next: Option<char>) -> bool {
    next.is_none_or(|c| c.is_whitespace() || ")]\"'*_`".contains(c))
}

/// Every sentence of a text that ends in `?`, its outer markers trimmed. Whether it is a
/// question unit is [`question_shape`]'s and [`unit_ok`]'s call, so a refusal is counted.
fn question_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if !matches!(c, '.' | '!' | '?') || !ends_sentence(chars.peek().map(|&(_, n)| n)) {
            continue;
        }
        if c == '?' {
            let q = text[start..=i].trim_matches(|c: char| " -*#>`[_".contains(c));
            if !q.is_empty() {
                out.push(q.to_string());
            }
        }
        start = i + c.len_utf8();
    }
    out
}

/// Marks that say a question sentence is a fragment of markup or code rather than prose: inline
/// code, emphasis cut by the sentence split, a table cell, a link, a brace. `qd_data
/// .defect_class.OWN_PROSE_QUESTION_MARKUP` is the same list.
pub const QUESTION_MARKUP: [&str; 7] = ["`", "*", "|", "](", "http", "{", "}"];

/// A question unit starts a sentence (an ASCII capital), carries none of [`QUESTION_MARKUP`],
/// and closes every double quote and parenthesis it opens: a `?` inside a quotation ends the
/// split, not the sentence. The reason codes are `qd_data.defect_class._own_prose_violation`'s.
pub fn question_shape(text: &str) -> std::result::Result<(), &'static str> {
    if !text.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return Err("question_not_a_sentence_start");
    }
    if QUESTION_MARKUP.iter().any(|m| text.contains(m)) {
        return Err("question_has_markup");
    }
    let count = |c: char| text.chars().filter(|&x| x == c).count();
    if count('"') % 2 == 1 || count('(') != count(')') {
        return Err("question_unbalanced");
    }
    Ok(())
}

fn unit_ok(
    text: &str,
    form: &str,
    ranges: &[(u32, u32)],
) -> std::result::Result<usize, &'static str> {
    if has_char_in(text, ranges) {
        return Err("invisible_format_character");
    }
    if ascii_permille(text) < MIN_ASCII_PERMILLE {
        return Err("ascii_ratio_below_min");
    }
    let n = words(text);
    if form == FORM_QUESTION {
        if !(QUESTION_WORDS.0..=QUESTION_WORDS.1).contains(&n) {
            return Err("question_words_out_of_band");
        }
        question_shape(text)?;
    } else {
        if n < MIN_PARAGRAPH_WORDS {
            return Err("paragraph_under_min_words");
        }
        if text.chars().count() > MAX_PARAGRAPH_CHARS {
            return Err("paragraph_over_max_chars");
        }
    }
    Ok(n)
}

fn skip_dir(name: &str) -> bool {
    name.starts_with('.') || SKIP_DIRS.contains(&name)
}

fn is_markdown(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("md" | "markdown")
    )
}

fn has_licence_file(dir: &Path) -> Result<bool> {
    for e in fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let name = e?.file_name();
        let upper = name.to_string_lossy().to_ascii_uppercase();
        if LICENCE_FILE_PREFIXES.iter().any(|p| upper.starts_with(p)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Every Markdown file under `root`, pruned by section 2.9 (2)'s directory rules.
fn markdown_files(
    root: &Path,
    nested: &BTreeSet<PathBuf>,
    skipped: &mut BTreeMap<String, u64>,
) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<_> = fs::read_dir(&dir)
            .with_context(|| format!("listing {}", dir.display()))?
            .collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let path = e.path();
            let ft = e.file_type()?;
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                let name = e.file_name().to_string_lossy().to_string();
                if skip_dir(&name) {
                    continue;
                }
                if nested.contains(&path) || path.join(".git").exists() {
                    *skipped.entry("dir:nested_git_tree".into()).or_insert(0) += 1;
                    continue;
                }
                if has_licence_file(&path)? {
                    *skipped.entry("dir:own_licence_file".into()).or_insert(0) += 1;
                    continue;
                }
                stack.push(path);
            } else if ft.is_file() && is_markdown(&path) {
                if e.metadata()?.len() > MAX_FILE_BYTES {
                    *skipped.entry("file:over_max_bytes".into()).or_insert(0) += 1;
                    continue;
                }
                out.push(path);
                if out.len() > MAX_FILES {
                    bail!(
                        "{} holds more than {MAX_FILES} Markdown files",
                        root.display()
                    );
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

#[derive(Deserialize)]
struct RangesOnly {
    invisible_format_ranges: Vec<(u32, u32)>,
}

struct Candidate {
    repo: String,
    path: String,
    file_sha256: String,
    text: String,
}

pub fn run(args: &Args<'_>) -> Result<()> {
    let units_path = args.out.join("units.jsonl");
    let files_path = args.out.join("files.jsonl");
    let manifest_path = args.out.join("manifest.json");
    if units_path.exists() || files_path.exists() || manifest_path.exists() {
        bail!(
            "{} already holds a corpus; a corpus is written once",
            args.out.display()
        );
    }
    if args.floor == 0 || args.floor > args.target || args.per_repo_cap == 0 {
        bail!("need 0 < --floor <= --target and --per-repo-cap > 0");
    }
    let inventory_bytes = fs::read(args.inventory)
        .with_context(|| format!("reading {}", args.inventory.display()))?;
    let inventory: Inventory =
        serde_json::from_slice(&inventory_bytes).context("parsing the inventory")?;
    let allowlist_bytes = fs::read(args.allowlist)
        .with_context(|| format!("reading {}", args.allowlist.display()))?;
    let ranges = serde_json::from_slice::<RangesOnly>(&allowlist_bytes)
        .context("the allowlist carries no invisible_format_ranges")?
        .invisible_format_ranges;
    if ranges.is_empty() || ranges.iter().any(|(lo, hi)| lo > hi) {
        bail!("the allowlist's invisible-format ranges are empty or backwards");
    }
    let code_root = PathBuf::from(&inventory.code_root);
    let all_repos: BTreeSet<PathBuf> = inventory
        .repos
        .iter()
        .map(|r| PathBuf::from(&r.repo))
        .collect();
    let excluded: BTreeSet<&PathBuf> = args.exclude_repos.iter().collect();

    // (1) admission; the "other" repos' Markdown hashes for (2)'s byte-identity rule.
    let mut admitted: Vec<(PathBuf, String, Vec<String>)> = Vec::new();
    let mut excluded_repos: BTreeMap<String, String> = BTreeMap::new();
    let mut other_md: BTreeSet<String> = BTreeSet::new();
    let mut skipped: BTreeMap<String, u64> = BTreeMap::new();
    for r in &inventory.repos {
        let path = PathBuf::from(&r.repo);
        let rel = path
            .strip_prefix(&code_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let nested: BTreeSet<PathBuf> = all_repos
            .iter()
            .filter(|p| **p != path && p.starts_with(&path))
            .cloned()
            .collect();
        if r.owner.starts_with("other") {
            let mut sink = BTreeMap::new();
            for f in markdown_files(&path, &nested, &mut sink)? {
                other_md.insert(sha256_hex(&fs::read(&f)?));
            }
        }
        if excluded.contains(&path) {
            excluded_repos.insert(rel, "excluded_by_the_human_tick".into());
            continue;
        }
        if r.owner != "own" {
            excluded_repos.insert(rel, format!("inventory_owner:{}", r.owner));
            continue;
        }
        let urls = remotes(&path)?;
        match admission(&urls) {
            Ok(()) => admitted.push((path, rel, urls)),
            Err(why) => {
                excluded_repos.insert(rel, why);
            }
        }
    }
    if admitted.len() != args.expect_admitted {
        let names: Vec<&str> = admitted.iter().map(|(_, r, _)| r.as_str()).collect();
        bail!(
            "{} repositories are admitted, {} were ticked: {names:?}; excluded {excluded_repos:?}",
            admitted.len(),
            args.expect_admitted
        );
    }

    // (2) and (3): walk, filter files, extract and filter units.
    let mut file_records: Vec<(String, String, String, usize)> = Vec::new();
    let mut questions: BTreeMap<String, Candidate> = BTreeMap::new();
    let mut paragraphs: BTreeMap<String, Candidate> = BTreeMap::new();
    for (path, rel, _) in &admitted {
        let nested: BTreeSet<PathBuf> = all_repos
            .iter()
            .filter(|p| *p != path && p.starts_with(path))
            .cloned()
            .collect();
        for f in markdown_files(path, &nested, &mut skipped)? {
            let bytes = fs::read(&f).with_context(|| format!("reading {}", f.display()))?;
            let Ok(text) = std::str::from_utf8(&bytes) else {
                *skipped.entry("file:not_utf8".into()).or_insert(0) += 1;
                continue;
            };
            let sha = sha256_hex(&bytes);
            let reason = if text.starts_with("---\n") || text.starts_with("---\r\n") {
                Some("file:yaml_front_matter")
            } else if text.lines().any(foreign_claim) {
                Some("file:foreign_copyright_or_spdx")
            } else if other_md.contains(&sha) {
                Some("file:identical_to_another_owners_file")
            } else {
                None
            };
            if let Some(reason) = reason {
                *skipped.entry(reason.into()).or_insert(0) += 1;
                continue;
            }
            let file_rel = f
                .strip_prefix(path)
                .unwrap_or(&f)
                .to_string_lossy()
                .to_string();
            file_records.push((rel.clone(), file_rel.clone(), sha.clone(), words(text)));
            let (paras, qs) = extract(text);
            for (form, texts, pool) in [
                (FORM_PARAGRAPH, paras, &mut paragraphs),
                (FORM_QUESTION, qs, &mut questions),
            ] {
                for t in texts {
                    match unit_ok(&t, form, &ranges) {
                        Ok(_) => {
                            pool.entry(t.clone()).or_insert(Candidate {
                                repo: rel.clone(),
                                path: file_rel.clone(),
                                file_sha256: sha.clone(),
                                text: t,
                            });
                        }
                        Err(why) => *skipped.entry(format!("{form}:{why}")).or_insert(0) += 1,
                    }
                }
            }
        }
    }
    // A paragraph that is also a question sentence is one text: the question keeps it.
    paragraphs.retain(|t, _| !questions.contains_key(t));

    // (4) selection: questions, then paragraphs round-robin, per-repo cap.
    let key = |c: &Candidate, form: &str| {
        order_key(
            args.seed,
            SOURCE,
            &format!("{form}\0{}\0{}\0{}", c.repo, c.path, c.text),
        )
    };
    let mut per_repo: BTreeMap<String, usize> = BTreeMap::new();
    let mut chosen: Vec<(Candidate, &'static str)> = Vec::new();
    let mut qs: Vec<Candidate> = questions.into_values().collect();
    qs.sort_by_key(|c| key(c, FORM_QUESTION));
    for c in qs {
        if chosen.len() == args.target {
            break;
        }
        let n = per_repo.entry(c.repo.clone()).or_insert(0);
        if *n >= args.per_repo_cap {
            *skipped
                .entry("question:over_per_repo_cap".into())
                .or_insert(0) += 1;
            continue;
        }
        *n += 1;
        chosen.push((c, FORM_QUESTION));
    }
    let mut by_repo: BTreeMap<String, Vec<Candidate>> = BTreeMap::new();
    for c in paragraphs.into_values() {
        by_repo.entry(c.repo.clone()).or_default().push(c);
    }
    let mut queues: Vec<(String, std::vec::IntoIter<Candidate>)> = by_repo
        .into_iter()
        .map(|(repo, mut v)| {
            v.sort_by_key(|c| key(c, FORM_PARAGRAPH));
            (repo, v.into_iter())
        })
        .collect();
    queues.sort_by_key(|(repo, _)| order_key(args.seed, SOURCE, repo));
    let mut progressed = true;
    while chosen.len() < args.target && progressed {
        progressed = false;
        for (repo, queue) in &mut queues {
            if chosen.len() == args.target {
                break;
            }
            let n = per_repo.entry(repo.clone()).or_insert(0);
            if *n >= args.per_repo_cap {
                continue;
            }
            if let Some(c) = queue.next() {
                *n += 1;
                chosen.push((c, FORM_PARAGRAPH));
                progressed = true;
            }
        }
    }
    if chosen.len() < args.floor {
        bail!(
            "{} units, under the floor of {}: the supply binds (skipped {skipped:?})",
            chosen.len(),
            args.floor
        );
    }

    // (5) record.
    let mut units: Vec<Unit> = chosen
        .into_iter()
        .map(|(c, form)| Unit {
            id: short_hash(&[&c.repo, &c.path, form, &c.text]),
            words: words(&c.text),
            repo: c.repo,
            path: c.path,
            file_sha256: c.file_sha256,
            form,
            text: c.text,
        })
        .collect();
    units.sort_by(|a, b| (&a.repo, &a.path, a.form, &a.id).cmp(&(&b.repo, &b.path, b.form, &b.id)));
    let ids: BTreeSet<&str> = units.iter().map(|u| u.id.as_str()).collect();
    if ids.len() != units.len() {
        bail!("{} units, {} distinct ids", units.len(), ids.len());
    }
    file_records.sort();
    let mut files_body = String::new();
    for (repo, path, sha, w) in &file_records {
        files_body.push_str(&serde_json::to_string(&FileRecord {
            repo,
            path,
            sha256: sha,
            words: *w,
        })?);
        files_body.push('\n');
    }
    let mut units_body = String::new();
    let mut totals: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut by_form: BTreeMap<&str, usize> = BTreeMap::new();
    for u in &units {
        units_body.push_str(&serde_json::to_string(u)?);
        units_body.push('\n');
        *totals
            .entry(u.repo.clone())
            .or_default()
            .entry(u.form)
            .or_insert(0) += 1;
        *by_form.entry(u.form).or_insert(0) += 1;
    }
    let manifest = Manifest {
        schema: SCHEMA,
        tool: "qd-noul-rows own-prose",
        tool_version: qd_mutate::TOOL_VERSION,
        seed: args.seed,
        source_id: SOURCE_ID,
        licence: LICENCE,
        code_root: inventory.code_root.clone(),
        inventory: BTreeMap::from([
            ("path", args.inventory.display().to_string()),
            ("sha256", sha256_hex(&inventory_bytes)),
        ]),
        allowlist_sha256: sha256_hex(&allowlist_bytes),
        invisible_format_ranges: ranges,
        admitted_repos: admitted
            .into_iter()
            .map(|(_, repo, remotes)| AdmittedRepo { repo, remotes })
            .collect(),
        excluded_repos,
        rules: serde_json::json!({
            "provenance": "Fable's v5 review section 2.9: own-repo prose is text the user authored or had written in their own repositories, not text hosted there",
            "admission": "a remote naming github owner bharathvbcr and no remote naming another owner; no-remote repos out; --exclude-repo out",
            "pruning": "nested git trees; hidden and build directories; vendor/ external/ third_party/ node_modules/ site-packages/; a directory below the root with its own LICENSE/LICENCE/COPYING/NOTICE; a file with a Copyright, (c)-sign or SPDX-License-Identifier line naming no owner token (bharath, vaddaram, bharathvbcr); a file opening with YAML front matter; a Markdown file byte-identical to one in a repo of another owner",
            "content": "Markdown (.md, .markdown) only; fenced and indented code, tables, HTML, headings, list items, block quotes, link-only lines and trailers dropped from paragraphs; questions also read from headings, list items and block quotes",
            "paragraph": {"min_words": MIN_PARAGRAPH_WORDS, "max_chars": MAX_PARAGRAPH_CHARS},
            "question": {
                "words": [QUESTION_WORDS.0, QUESTION_WORDS.1],
                "ends_with": "?",
                "starts_with": "an ASCII capital",
                "refused_markup": QUESTION_MARKUP,
                "sentences": "read from a paragraph's joined lines, or from one heading, list item or block quote; a . ! or ? ends a sentence only before whitespace, a closing mark or the end"
            },
            "word": "[A-Za-z][A-Za-z'\\-]+",
            "min_ascii_permille": MIN_ASCII_PERMILLE,
            "selection": "every distinct question in keyed order, then paragraphs round-robin over repos in keyed order, each repo at most per_repo_cap units",
            "other_owner_markdown_files_hashed": other_md.len(),
        }),
        files: Pinned {
            file: "files.jsonl",
            sha256: sha256_hex(files_body.as_bytes()),
            count: file_records.len(),
        },
        units: Pinned {
            file: "units.jsonl",
            sha256: sha256_hex(units_body.as_bytes()),
            count: units.len(),
        },
        target: args.target,
        floor: args.floor,
        per_repo_cap: args.per_repo_cap,
        totals: serde_json::json!({"units": units.len(), "by_form": by_form, "by_repo": totals}),
        skipped,
        commit_bodies: "not read: git runs only in the lane's worktree (GAP-L-V5PLAN-OWN-REPO-COMMIT-BODIES-NOT-MEASURED-2026-10-02)",
    };
    fs::create_dir_all(args.out).with_context(|| format!("creating {}", args.out.display()))?;
    crate::write_atomic(&files_path, files_body.as_bytes())?;
    crate::write_atomic(&units_path, units_body.as_bytes())?;
    let mut text = serde_json::to_string_pretty(&manifest)?;
    text.push('\n');
    crate::write_atomic(&manifest_path, text.as_bytes())?;
    println!(
        "own-prose: {} units {by_form:?} from {} files of {} repos -> {}",
        units.len(),
        file_records.len(),
        manifest.admitted_repos.len(),
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_match_the_python_regex_on_its_edge_cases() {
        // [A-Za-z][A-Za-z'\-]+ : one letter alone is no word; apostrophes and hyphens continue.
        assert_eq!(words("a I x y"), 0);
        assert_eq!(words("it's a well-known fact"), 3);
        // Python's re.findall gives ['caf', 'na', 've']: a non-ASCII letter splits a word.
        assert_eq!(words("caf\u{e9} na\u{ef}ve"), 3);
        assert_eq!(words("--ab-- 9to5 x2y z'"), 3);
    }

    #[test]
    fn admission_is_the_owner_and_no_one_else() {
        let u = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert!(admission(&u(&["https://github.com/bharathvbcr/Chronicle.git"])).is_ok());
        assert!(admission(&u(&["git@github.com:bharathvbcr/x.git"])).is_ok());
        assert_eq!(
            admission(&u(&[
                "https://github.com/delibae/claude-prism.git",
                "https://github.com/bharathvbcr/DevPrism.git"
            ])),
            Err("fork_or_other_owner:delibae".to_string())
        );
        assert_eq!(admission(&[]), Err("no_remote".to_string()));
        assert_eq!(
            admission(&u(&["https://gitlab.com/bharathvbcr/x"])),
            Err("non_github_remote".to_string())
        );
    }

    #[test]
    fn extraction_keeps_prose_and_drops_code_tables_html_links_and_trailers() {
        let md = "# Why does it retry the upload before it gives up on the server?\n\n\
                  This tool keeps a ledger of every run\nso a reader can compare them.\n\n\
                  ```\nwhy is this code a question at all here?\n```\n\n\
                  | a | b |\n|---|---|\n\n<div>html</div>\n\n[link](http://x)\n\n    indented code line\n\n\
                  Signed-off-by: someone\n\n- a list item asks: what happens when two runs collide?\n";
        // A `\` continuation strips the next line's leading blanks, so the indented line is
        // written after an explicit `\n` on the line above it.
        assert!(md.contains("\n    indented code line\n"));
        let (paras, qs) = extract(md);
        assert_eq!(
            paras,
            vec!["This tool keeps a ledger of every run so a reader can compare them."]
        );
        // The list item's sentence is extracted here and refused later by question_shape.
        assert_eq!(
            qs,
            vec![
                "Why does it retry the upload before it gives up on the server?".to_string(),
                "a list item asks: what happens when two runs collide?".to_string(),
            ]
        );
    }

    #[test]
    fn a_foreign_claim_is_one_that_names_no_owner_token() {
        assert!(foreign_claim(
            "Copyright (c) 2020 The Rust Project Developers"
        ));
        assert!(foreign_claim("SPDX-License-Identifier: MIT"));
        assert!(foreign_claim("\u{a9} Some Corp"));
        assert!(!foreign_claim(
            "Copyright (c) 2026 Bharath Chandra Vaddaram"
        ));
        assert!(!foreign_claim("This is ordinary prose."));
    }

    #[test]
    fn unit_rules_bound_words_ascii_and_length() {
        let r = [(0x200B, 0x200F)];
        let para = "word ".repeat(30);
        assert!(unit_ok(&para, FORM_PARAGRAPH, &r).is_ok());
        assert_eq!(
            unit_ok("too few words here", FORM_PARAGRAPH, &r),
            Err("paragraph_under_min_words")
        );
        assert_eq!(
            unit_ok(&"word ".repeat(500), FORM_PARAGRAPH, &r),
            Err("paragraph_over_max_chars")
        );
        assert_eq!(
            unit_ok(&format!("{para}\u{200B}"), FORM_PARAGRAPH, &r),
            Err("invisible_format_character")
        );
        assert_eq!(
            unit_ok(&"\u{e9}t\u{e9} ".repeat(30), FORM_PARAGRAPH, &r),
            Err("ascii_ratio_below_min")
        );
        assert_eq!(
            unit_ok("is it?", FORM_QUESTION, &r),
            Err("question_words_out_of_band")
        );
        assert!(
            unit_ok(
                "Why does the cache drop entries when it is full?",
                FORM_QUESTION,
                &r
            )
            .is_ok()
        );
    }

    #[test]
    fn a_question_is_a_whole_sentence_and_not_a_fragment_of_markup() {
        let r = [(0x200B, 0x200F)];
        // A `?` inside a token is not a sentence end; a `.` before a space is.
        assert_eq!(
            question_sentences(
                "Call /synthesis?q=1 from v1.2 first. Does the retry loop give up after a failure?"
            ),
            vec!["Does the retry loop give up after a failure?".to_string()]
        );
        // A question wrapped across two source lines is read whole from its paragraph.
        let (_, qs) = extract("Can a sparse network learn\ncompetitively without any backprop?\n");
        assert_eq!(
            qs,
            vec!["Can a sparse network learn competitively without any backprop?".to_string()]
        );
        for (text, why) in [
            (
                "trained accuracy under tempered or hard winner evaluation?",
                "question_not_a_sentence_start",
            ),
            (
                "It asks: **how much more did the attention model lose?",
                "question_has_markup",
            ),
            (
                "Which gate moved the `POST /synthesis` route to Go?",
                "question_has_markup",
            ),
            (
                "Did the page at https://example.org say why it failed?",
                "question_has_markup",
            ),
            (
                "WorkspaceSearchParams { query, filters, mode, quality, sources?",
                "question_has_markup",
            ),
            (
                "Users ask questions like \"What should I work on today?",
                "question_unbalanced",
            ),
            (
                "Which run (the second or the third failed?",
                "question_unbalanced",
            ),
        ] {
            assert_eq!(unit_ok(text, FORM_QUESTION, &r), Err(why), "{text}");
        }
    }
}
