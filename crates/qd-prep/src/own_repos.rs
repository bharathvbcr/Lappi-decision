//! `qd-prep own-repos`: the human's own repositories, admitted and split once, written down
//! before any commit is read.
//!
//! The canonical own-repo enumerator for v6 (`HANDOFF/data-clean-v6-plan-2026-10-06.md` §5,
//! `campaign/v6-caller-families.DRAFT.json` `build_order` step 1). The natural-bug miner reads its
//! manifest, and the caller-families lane is meant to read it too. The manifest file is
//! therefore the proof of "manifest before any commit": a miner cannot run without one, and
//! this module runs exactly one git command per admitted repository (`rev-parse` of `HEAD`).
//!
//! **The rule** is `holdout_rule` of the DRAFT, applied as written:
//!
//! 1. **Discovery.** Every directory under `--code-root`, at most [`MAX_DEPTH`] levels down, that
//!    holds a `.git` entry. Hidden directories, `node_modules` and any `worktrees` directory are
//!    not descended into (v5's `own_repo_inventory.py::find_repos`, which globbed one to four
//!    levels for `.git`). A nested repository is a repository of its own.
//! 2. **Exclusion,** first reason wins, each written to the manifest:
//!    - `excluded_by_the_human_tick`: `research/Lappi-decision` (the human's "All but
//!      Lappi-decision");
//!    - `struck_by_the_human`: `web/Lappi-BDay`, `web/WhimsicalLove`, `web/bharathvbcr`
//!      (`AUDIT/v5-plan-2026-10-02/human-answer-own-prose-strike.md`);
//!    - `linked_worktree_or_submodule`: `.git` is a file, so the directory is a checkout of a
//!      repository that lives elsewhere, not a repository of its own;
//!    - `no_remote`: no `[remote]` section names a URL; ownership is not readable;
//!    - `upstream_remote:<url>`: a remote is named `upstream` (the DRAFT's "repos with an
//!      upstream remote are out"), whatever owner it names;
//!    - [`admission`]'s reasons: `fork_or_other_owner:<owner>`, `non_github_remote`,
//!      `no_owner_remote`;
//!    - `no_origin_remote`, `origin_not_a_github_slug`: the split reads `origin`, so a repository
//!      without one cannot be split and is out rather than guessed;
//!    - `no_head_commit:<why>`: an admitted repository whose `HEAD` names no commit.
//! 3. **Struck paths** (`scholarlm:docs/business/`, `research/BINN:writing/`) are recorded in
//!    the manifest; every reader applies them. They strike files, not repositories.
//! 4. **Split.** A repository is held out iff
//!    `int(sha256(lowercase('owner/name' from its origin URL)).hexdigest()[:8], 16) % 5 == 0`
//!    ([`holdout_bucket`]). `owner/name` is the URL's path after `github.com`, with a trailing
//!    `/` and then a trailing `.git` removed; the DRAFT does not spell that out, and the GitHub
//!    slug is its plain reading.
//!
//! **Ported, not shared.** [`github_owner`] and [`admission`] are
//! `crates/qd-mutate/src/noul_rows/own_prose.rs:283-308` (v5's own-prose admission), copied
//! because `qd-prep` cannot link `qd-mutate`'s tree-sitter grammars. The copy's test carries
//! that module's `admission_is_the_owner_and_no_one_else` cases verbatim as its parity anchor.
//! Two deliberate differences from v5, both recorded per repository in the manifest's
//! `v5_rule_comparison`:
//! - v5 read every `url =` line in `.git/config` (a submodule's URL included); this reads the
//!   URLs of `[remote "..."]` sections only, because the rule is about remotes;
//! - v5 had no `upstream` rule beyond the owner check.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::gitcli::{self, GitError};
use crate::sha256::sha256_hex;

pub const SCHEMA: &str = "qd-own-repos/v1";
/// The github owner whose repositories are the human's own.
pub const OWNER: &str = "bharathvbcr";
/// v5's `find_repos` globbed `*/.git` to `*/*/*/*/.git`: repositories one to four levels down.
pub const MAX_DEPTH: usize = 4;
/// One repository in [`HOLDOUT_MODULUS`] is held out.
pub const HOLDOUT_MODULUS: u32 = 5;
/// The human's tick: "All but Lappi-decision".
pub const EXCLUDED_BY_THE_HUMAN: [&str; 1] = ["research/Lappi-decision"];
/// The human's strike, whole repositories.
pub const STRUCK_REPOS: [&str; 3] = ["web/Lappi-BDay", "web/WhimsicalLove", "web/bharathvbcr"];
/// The human's strike, repo-relative directories (`prefix` ends in `/`).
pub const STRUCK_PATHS: [(&str, &str); 2] = [
    ("scholarlm", "docs/business/"),
    ("research/BINN", "writing/"),
];
/// Directory names never descended into, besides hidden ones.
const SKIP_DIRS: [&str; 2] = ["node_modules", "worktrees"];
/// Bounds on the walk: a code root with more is mis-pointed.
const MAX_DIRS_VISITED: usize = 2_000_000;
const MAX_REPOS: usize = 10_000;
const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// The pre-committed rule's inputs. [`Rules::pre_committed`] is the only value the CLI uses;
/// tests build fixtures under the same names.
#[derive(Clone, Debug)]
pub struct Rules {
    pub code_root: PathBuf,
    pub excluded_by_the_human: Vec<String>,
    pub struck_repos: Vec<String>,
    pub struck_paths: Vec<(String, String)>,
    pub max_depth: usize,
}

impl Rules {
    /// `holdout_rule` of `campaign/v6-caller-families.DRAFT.json`, under `code_root`.
    pub fn pre_committed(code_root: PathBuf) -> Self {
        Rules {
            code_root,
            excluded_by_the_human: EXCLUDED_BY_THE_HUMAN
                .iter()
                .map(|s| s.to_string())
                .collect(),
            struck_repos: STRUCK_REPOS.iter().map(|s| s.to_string()).collect(),
            struck_paths: STRUCK_PATHS
                .iter()
                .map(|(r, p)| (r.to_string(), p.to_string()))
                .collect(),
            max_depth: MAX_DEPTH,
        }
    }
}

/// A `[remote "name"]` section's URL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Remote {
    pub name: String,
    pub url: String,
}

/// What `.git/config` says: the remotes' URLs by name, and every `url =` line anywhere (v5's
/// reading, kept for the comparison).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConfigUrls {
    pub remotes: Vec<Remote>,
    pub all_urls: Vec<String>,
}

/// Parse a git config file's remote URLs. Section headers are `[remote "name"]`; a key is
/// compared case-insensitively, as git does; anything that is not a `url` key is ignored.
pub fn parse_config(text: &str) -> ConfigUrls {
    let mut out = ConfigUrls::default();
    let mut section: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            section = remote_section(line);
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        if !k.trim().eq_ignore_ascii_case("url") {
            continue;
        }
        let url = v.trim().to_string();
        out.all_urls.push(url.clone());
        if let Some(name) = &section {
            out.remotes.push(Remote {
                name: name.clone(),
                url,
            });
        }
    }
    out
}

/// `[remote "origin"]` -> `Some("origin")`; any other header -> `None`.
fn remote_section(line: &str) -> Option<String> {
    let inner = line.strip_prefix('[')?.strip_suffix(']')?.trim();
    let (kind, rest) = inner.split_once(char::is_whitespace)?;
    if !kind.eq_ignore_ascii_case("remote") {
        return None;
    }
    let name = rest.trim().strip_prefix('"')?.strip_suffix('"')?;
    (!name.is_empty()).then(|| name.to_string())
}

/// The github owner a remote URL names, or `None` for a URL that is not github's.
/// Ported from `own_prose.rs::github_owner`.
pub fn github_owner(url: &str) -> Option<&str> {
    let rest = url.split_once("github.com").map(|(_, r)| r)?;
    let rest = rest.strip_prefix(':').or_else(|| rest.strip_prefix('/'))?;
    let owner = rest.split('/').next()?;
    (!owner.is_empty()).then_some(owner)
}

/// Admitted iff a remote names [`OWNER`] and no remote names anything else.
/// Ported from `own_prose.rs::admission`; the error strings are that function's.
pub fn admission(urls: &[String]) -> Result<(), String> {
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

/// `owner/name`, lowercased, from a github URL: the path after `github.com`, a trailing `/`
/// and then a trailing `.git` removed. `None` unless that is exactly two non-empty segments.
pub fn slug(url: &str) -> Option<String> {
    let rest = url.split_once("github.com").map(|(_, r)| r)?;
    let rest = rest.strip_prefix(':').or_else(|| rest.strip_prefix('/'))?;
    let rest = rest.trim_end_matches('/');
    let rest = rest.strip_suffix(".git").unwrap_or(rest);
    let mut parts = rest.split('/');
    let (owner, name) = (parts.next()?, parts.next()?);
    if owner.is_empty() || name.is_empty() || parts.next().is_some() {
        return None;
    }
    Some(format!("{owner}/{name}").to_lowercase())
}

/// `int(sha256(slug).hexdigest()[:8], 16) % 5`. Held out iff 0.
pub fn holdout_bucket(slug: &str) -> u32 {
    let hex = sha256_hex(slug.as_bytes());
    // Eight hex digits always fit a u32, and sha256_hex always returns 64 of them.
    let first = u32::from_str_radix(&hex[..8], 16).unwrap_or(u32::MAX);
    first % HOLDOUT_MODULUS
}

/// `"heldout"` or `"train"`: the repository's split under the rule. Train repositories feed
/// train and val; the family that builds from them splits further.
pub fn split_of(slug: &str) -> &'static str {
    if holdout_bucket(slug) == 0 {
        "heldout"
    } else {
        "train"
    }
}

/// Is `path` (repo-relative) inside one of the struck directories of `repo`?
pub fn struck_path(struck: &[(String, String)], repo: &str, path: &str) -> Option<String> {
    struck
        .iter()
        .find(|(r, prefix)| r == repo && path.starts_with(prefix.as_str()))
        .map(|(r, prefix)| format!("{r}:{prefix}"))
}

/// Resolves a repository's `HEAD` to a commit sha.
pub type HeadFn = dyn Fn(&Path) -> Result<String, GitError>;

/// What the walk found.
#[derive(Clone, Debug, Default)]
pub struct Discovered {
    /// Every repository, sorted.
    pub repos: Vec<PathBuf>,
    /// Directories that could not be read, each with its error, sorted.
    pub unreadable: Vec<String>,
}

/// Every repository under `root`, at most `max_depth` levels down, sorted; and the directories
/// that could not be read (reported, not fatal: an unreadable directory holds nothing this
/// walk could admit, and the manifest says which they were).
pub fn find_repos(root: &Path, max_depth: usize) -> Result<Discovered, String> {
    let mut repos = Vec::new();
    let mut unreadable = Vec::new();
    let mut stack: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];
    let mut visited = 0usize;
    while let Some((dir, depth)) = stack.pop() {
        visited += 1;
        if visited > MAX_DIRS_VISITED {
            return Err(format!(
                "walked more than {MAX_DIRS_VISITED} directories under {}; refusing",
                root.display()
            ));
        }
        if depth >= 1 && fs::symlink_metadata(dir.join(".git")).is_ok() {
            repos.push(dir.clone());
            if repos.len() > MAX_REPOS {
                return Err(format!(
                    "more than {MAX_REPOS} repositories under {}",
                    root.display()
                ));
            }
        }
        if depth == max_depth {
            continue;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                unreadable.push(format!("{}: {e}", dir.display()));
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    unreadable.push(format!("{}: {e}", dir.display()));
                    continue;
                }
            };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            // symlink_metadata: a symlinked directory is not followed, so a link cannot make
            // one repository appear twice or lead the walk outside the root.
            match fs::symlink_metadata(entry.path()) {
                Ok(m) if m.is_dir() => stack.push((entry.path(), depth + 1)),
                Ok(_) => {}
                Err(e) => unreadable.push(format!("{}: {e}", entry.path().display())),
            }
        }
    }
    repos.sort();
    unreadable.sort();
    Ok(Discovered { repos, unreadable })
}

/// `repo` relative to `root`, `/`-separated.
fn rel_path(root: &Path, repo: &Path) -> Result<String, String> {
    let r = repo
        .strip_prefix(root)
        .map_err(|_| format!("{} is not under {}", repo.display(), root.display()))?;
    let parts: Vec<String> = r
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    Ok(parts.join("/"))
}

/// One repository's verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Admitted {
        origin_url: String,
        slug: String,
        bucket: u32,
        head: String,
    },
    Excluded(String),
}

/// The verdict for one discovered repository. `head` resolves `HEAD`; it is called only for
/// a repository every other rule admits.
pub fn judge(
    rules: &Rules,
    rel: &str,
    repo: &Path,
    head: &HeadFn,
) -> Result<(Verdict, ConfigUrls), String> {
    let urls = read_config(repo)?;
    if rules.excluded_by_the_human.iter().any(|r| r == rel) {
        return Ok((Verdict::Excluded("excluded_by_the_human_tick".into()), urls));
    }
    if rules.struck_repos.iter().any(|r| r == rel) {
        return Ok((Verdict::Excluded("struck_by_the_human".into()), urls));
    }
    if repo.join(".git").is_file() {
        return Ok((
            Verdict::Excluded("linked_worktree_or_submodule".into()),
            urls,
        ));
    }
    if urls.remotes.is_empty() {
        return Ok((Verdict::Excluded("no_remote".into()), urls));
    }
    if let Some(up) = urls.remotes.iter().find(|r| r.name == "upstream") {
        return Ok((
            Verdict::Excluded(format!("upstream_remote:{}", up.url)),
            urls,
        ));
    }
    let remote_urls: Vec<String> = urls.remotes.iter().map(|r| r.url.clone()).collect();
    if let Err(reason) = admission(&remote_urls) {
        return Ok((Verdict::Excluded(reason), urls));
    }
    let Some(origin) = urls.remotes.iter().find(|r| r.name == "origin") else {
        return Ok((Verdict::Excluded("no_origin_remote".into()), urls));
    };
    let Some(s) = slug(&origin.url) else {
        return Ok((Verdict::Excluded("origin_not_a_github_slug".into()), urls));
    };
    let head = match head(repo) {
        Ok(sha) => sha,
        Err(GitError::Spawn(e)) => return Err(e),
        Err(e) => return Ok((Verdict::Excluded(format!("no_head_commit:{e}")), urls)),
    };
    let bucket = holdout_bucket(&s);
    Ok((
        Verdict::Admitted {
            origin_url: origin.url.clone(),
            slug: s,
            bucket,
            head,
        },
        urls,
    ))
}

/// The remotes of `repo`'s own `.git/config`. A `.git` file (linked worktree or submodule) is
/// followed to its common directory's config, as v5's reader did, so the comparison with v5
/// can be made for it too; [`judge`] excludes it before its remotes matter.
fn read_config(repo: &Path) -> Result<ConfigUrls, String> {
    let git = repo.join(".git");
    let config = if git.is_dir() {
        git.join("config")
    } else if git.is_file() {
        let line = read_bounded(&git)?;
        let Some(gitdir) = line.trim().strip_prefix("gitdir:").map(str::trim) else {
            return Ok(ConfigUrls::default());
        };
        let dir = repo.join(gitdir);
        let common = dir.join("commondir");
        let base = if common.is_file() {
            dir.join(read_bounded(&common)?.trim())
        } else {
            dir
        };
        base.join("config")
    } else {
        return Ok(ConfigUrls::default());
    };
    if !config.is_file() {
        return Ok(ConfigUrls::default());
    }
    Ok(parse_config(&read_bounded(&config)?))
}

fn read_bounded(path: &Path) -> Result<String, String> {
    let len = fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if len > MAX_CONFIG_BYTES {
        return Err(format!(
            "{}: {len} bytes; a git config past {MAX_CONFIG_BYTES} is refused",
            path.display()
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Enumerate, judge and split every repository under `rules.code_root`; the manifest.
pub fn build_manifest(rules: &Rules, head: &HeadFn, written_at: &str) -> Result<Value, String> {
    let root = &rules.code_root;
    if !root.is_absolute() {
        return Err(format!("--code-root {} must be absolute", root.display()));
    }
    let Discovered { repos, unreadable } = find_repos(root, rules.max_depth)?;
    let mut admitted = Vec::new();
    let mut excluded = BTreeMap::new();
    let mut comparison = Vec::new();
    let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
    let mut seen_slugs: BTreeMap<String, String> = BTreeMap::new();
    for repo in &repos {
        let rel = rel_path(root, repo)?;
        let (verdict, urls) = judge(rules, &rel, repo, head)?;
        let v5 = admission(&urls.all_urls);
        let ours = matches!(verdict, Verdict::Admitted { .. });
        let human_choice = rules.excluded_by_the_human.contains(&rel)
            || rules.struck_repos.contains(&rel)
            || repo.join(".git").is_file();
        if v5.is_ok() != ours && !human_choice {
            let v5_rule = v5.clone().err().unwrap_or_else(|| "admitted".into());
            let this_rule = match &verdict {
                Verdict::Admitted { .. } => "admitted".to_string(),
                Verdict::Excluded(r) => r.clone(),
            };
            comparison.push(json!({"repo": rel, "v5_rule": v5_rule, "this_rule": this_rule}));
        }
        let remotes: Vec<Value> = urls
            .remotes
            .iter()
            .map(|r| json!({"name": r.name, "url": r.url}))
            .collect();
        match verdict {
            Verdict::Admitted {
                origin_url,
                slug,
                bucket,
                head,
            } => {
                if let Some(other) = seen_slugs.insert(slug.clone(), rel.clone()) {
                    return Err(format!(
                        "{rel} and {other} share the origin slug {slug}: two checkouts of one \
                         repository would put one history in the set twice. Refusing; the \
                         human decides which one is the repository."
                    ));
                }
                let split = split_of(&slug);
                admitted.push(json!({
                    "repo": rel,
                    "remotes": remotes,
                    "origin_url": origin_url,
                    "slug": slug,
                    "split_bucket": bucket,
                    "split": split,
                    "head": head,
                }));
            }
            Verdict::Excluded(reason) => {
                let key = reason.split(':').next().unwrap_or(&reason).to_string();
                *by_reason.entry(key).or_default() += 1;
                excluded.insert(rel, json!({"reason": reason, "remotes": remotes}));
            }
        }
    }
    let heldout: Vec<&str> = admitted
        .iter()
        .filter(|a| a["split"] == "heldout")
        .filter_map(|a| a["repo"].as_str())
        .collect();
    let struck_paths: Vec<Value> = rules
        .struck_paths
        .iter()
        .map(|(r, p)| json!({"repo": r, "prefix": p}))
        .collect();
    let struck_not_found: BTreeSet<&str> = rules
        .struck_repos
        .iter()
        .chain(rules.excluded_by_the_human.iter())
        .map(String::as_str)
        .filter(|r| !excluded.contains_key(*r))
        .collect();
    Ok(json!({
        "schema": SCHEMA,
        "tool": "qd-prep own-repos",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "written_at": written_at,
        "commits_read": "none: this manifest is written before any commit is read; the only git \
                         command run is rev-parse of HEAD in each admitted repository",
        "code_root": root.display().to_string(),
        "rules": {
            "source": "campaign/v6-caller-families.DRAFT.json holdout_rule",
            "owner": OWNER,
            "discovery": format!(
                "directories 1..={} levels under code_root holding a .git entry; hidden \
                 directories, node_modules and worktrees not descended into; symlinks not \
                 followed",
                rules.max_depth
            ),
            "admission": "a [remote] URL names github owner bharathvbcr and no [remote] URL \
                          names another owner or a non-github host (own_prose.rs::admission, \
                          ported); no remote, a remote named upstream, a .git file (linked \
                          worktree or submodule) and no origin remote are out",
            "excluded_by_the_human_tick": rules.excluded_by_the_human,
            "struck_repos": rules.struck_repos,
            "struck_paths": struck_paths,
            "split": "heldout iff int(sha256(slug).hexdigest()[:8], 16) % 5 == 0; slug = \
                      lowercase owner/name of the origin URL after github.com, trailing / then \
                      .git removed",
            "first_reason_wins": true,
        },
        "admitted": admitted,
        "excluded": excluded,
        "unreadable_dirs": unreadable,
        "v5_rule_comparison": {
            "note": "repos (other than the human's exclusions and .git files) whose verdict \
                     differs between v5's reading (every url = line in .git/config, owner rule \
                     only) and this one",
            "differences": comparison,
        },
        "totals": {
            "discovered": repos.len(),
            "admitted": admitted.len(),
            "heldout": heldout.len(),
            "train": admitted.len() - heldout.len(),
            "excluded": excluded.len(),
            "excluded_by_reason": by_reason,
            "heldout_repos": heldout,
            "human_exclusions_not_found": struck_not_found,
        },
    }))
}

/// UTC now as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Days since 1970-01-01 -> (year, month, day), proleptic Gregorian (H. Hinnant's algorithm).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

/// (year, month, day) -> days since 1970-01-01 (the inverse of [`civil_from_days`]).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = i64::from(m);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Write `bytes` to `path` whole or not at all; refused if `path` or its `.partial` exists.
pub fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            path.display()
        ));
    }
    let partial = path.with_extension("partial");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    fs::rename(partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// `qd-prep own-repos --code-root ROOT --out FILE`.
pub fn run(code_root: &Path, out: &Path) -> Result<String, String> {
    if out.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            out.display()
        ));
    }
    let rules = Rules::pre_committed(code_root.to_path_buf());
    let head = |repo: &Path| gitcli::resolve_commit(repo, "HEAD");
    let manifest = build_manifest(&rules, &head, &utc_now())?;
    let mut bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    write_new(out, &bytes)?;
    let t = &manifest["totals"];
    Ok(format!(
        "qd-prep own-repos: {} discovered, {} admitted ({} heldout, {} train), {} excluded -> {} \
         (sha256 {})",
        t["discovered"],
        t["admitted"],
        t["heldout"],
        t["train"],
        t["excluded"],
        out.display(),
        sha256_hex(&bytes)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `own_prose.rs::admission_is_the_owner_and_no_one_else`, verbatim: the port's parity
    /// anchor with v5's admission.
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
    fn the_split_hash_matches_values_computed_by_hand() {
        // `printf 'bharathvbcr/gusset' | shasum -a 256` -> dd5f2ef1aad0...; 0xdd5f2ef1 =
        // 3714002673; 3714002673 % 5 = 3.
        assert_eq!(
            &sha256_hex(b"bharathvbcr/gusset")[..8],
            "dd5f2ef1",
            "the digest itself"
        );
        assert_eq!(holdout_bucket("bharathvbcr/gusset"), 3_714_002_673 % 5);
        assert_eq!(split_of("bharathvbcr/gusset"), "train");
        // Python: int(sha256(b"bharathvbcr/gitpulse").hexdigest()[:8], 16) = 0x0b262446 =
        // 187049030; % 5 = 0.
        assert_eq!(&sha256_hex(b"bharathvbcr/gitpulse")[..8], "0b262446");
        assert_eq!(holdout_bucket("bharathvbcr/gitpulse"), 0);
        assert_eq!(split_of("bharathvbcr/gitpulse"), "heldout");
    }

    #[test]
    fn a_slug_is_the_lowercased_owner_and_name_without_dot_git() {
        let s = slug;
        assert_eq!(
            s("https://github.com/bharathvbcr/GitPulse.git").as_deref(),
            Some("bharathvbcr/gitpulse")
        );
        assert_eq!(
            s("git@github.com:bharathvbcr/GitPulse.git").as_deref(),
            Some("bharathvbcr/gitpulse")
        );
        assert_eq!(
            s("https://github.com/bharathvbcr/AcademiaTrack").as_deref(),
            Some("bharathvbcr/academiatrack")
        );
        assert_eq!(
            s("https://github.com/bharathvbcr/x.git/").as_deref(),
            Some("bharathvbcr/x")
        );
        assert_eq!(s("https://github.com/bharathvbcr"), None, "no name");
        assert_eq!(s("https://github.com/a/b/c"), None, "three segments");
        assert_eq!(s("https://gitlab.com/bharathvbcr/x"), None);
    }

    #[test]
    fn a_config_binds_urls_to_remote_sections_and_keeps_every_url_for_v5() {
        let cfg = "[core]\n\tbare = false\n[remote \"origin\"]\n\turl = https://github.com/bharathvbcr/a.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n[remote \"upstream\"]\n\tURL = https://github.com/other/a.git\n[submodule \"lib\"]\n\turl = https://github.com/someone/lib.git\n";
        let c = parse_config(cfg);
        assert_eq!(
            c.remotes,
            [
                Remote {
                    name: "origin".into(),
                    url: "https://github.com/bharathvbcr/a.git".into()
                },
                Remote {
                    name: "upstream".into(),
                    url: "https://github.com/other/a.git".into()
                },
            ]
        );
        assert_eq!(c.all_urls.len(), 3, "v5 read the submodule's url too");
    }

    #[test]
    fn a_struck_path_is_a_prefix_inside_its_own_repo_only() {
        let s = [("scholarlm".to_string(), "docs/business/".to_string())];
        assert!(struck_path(&s, "scholarlm", "docs/business/plan.py").is_some());
        assert!(struck_path(&s, "scholarlm", "docs/businessplan.py").is_none());
        assert!(struck_path(&s, "other", "docs/business/plan.py").is_none());
    }

    #[test]
    fn civil_dates_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2026, 2, 1), 20_485);
        for z in [-1, 0, 59, 60, 11_016, 20_485, 20_734] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(civil_from_days(20_485), (2026, 2, 1));
    }
}
