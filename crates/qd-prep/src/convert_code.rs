//! Code: the CodeSearchNet pool (jevgrep-shaped "does function F do what query Q describes")
//! and SWE-rebench's row filter (a filtered view, not a pool: its family needs the repository
//! tree at `base_commit`, which is not on this machine -- parked).
//!
//! **CodeSearchNet** (`python` and `go` train; Lappi-admitted languages only). The dataset
//! carries no per-example licence, so each repository's licence comes from the cache
//! `csn_licences.py` wrote (GitHub's detected licence key, today, on the default branch -- not
//! the licence at the 2019 commit; pinned by sha256 in the config). Default-deny: a repository
//! missing from the cache, not found, with no detected licence, or with a key outside Python's
//! allowlist is refused and counted by reason. The upstream `_licenses.pkl` is never read.
//!
//! - **Query** = the first paragraph of the function's docstring, whitespace-collapsed (Go's
//!   `//` markers removed). The function's own short name is replaced by "this function" where
//!   it appears as a word: Go docs open with it, a lexical shortcut to the positive (counted).
//! - **Function text** = the code with the docstring removed: CodeSearchNet's Python
//!   `whole_func_string` contains its docstring (1,000 of 1,000 sampled 2026-10-06), so the
//!   positive would contain the query verbatim. A function whose docstring's opening still
//!   appears in the stripped code is refused (`docstring_not_stripped`).
//! - **Rows** per repository: a seeded reservoir of `max_functions_per_repo` functions; the first
//!   `max_queries_per_repo` by seeded order each give a positive (its own function) and
//!   `negatives_per_query` hard negatives (other functions of the same repository, seeded,
//!   never one with the same query or the same code). A repository with one function has no
//!   negative and is refused (`no_negatives`). Caveat: a same-repo negative may also do what
//!   the query says (overloads, wrappers); that noise is unmeasured.
//! - `group_key` = the repository, so the split is repository-disjoint by construction.
//! - **Pretraining exposure**: CodeSearchNet (2019, public GitHub) is very likely in the base
//!   model's pretraining data; the manifest says so. Its test/validation splits are targets.
//!
//! **SWE-rebench** row filter: `license_name` through the licence mirror (so a display name
//! Python has no alias for is refused until `qd_data.licences` registers it), and every
//! `instance_id` found in SWE-bench (dev/test/train), SWE-bench_Lite (dev/test) or Loc-Bench
//! dropped. Writes `rows.jsonl` (`{"licence", "row"}`) and `counts.json`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Value, json};

use crate::convert::{self, Acc, Config, PoolFacts, Row, Tally, Views};
use crate::convert_licence::{self, Verdict};
use crate::decisions::{self, Gold};
use crate::sha256::sha256_hex;

pub const CSN: &str = "code-search-net/code_search_net";
pub const CSN_FAMILY: &str = "csn.func_match";
pub const CSN_LANGS: [&str; 2] = ["python", "go"];
pub const CSN_QUESTION: &str = "Does this function do what the query describes?";
/// The longest query kept, in bytes; a longer first paragraph is refused, not cut.
pub const MAX_QUERY_BYTES: usize = 1_000;
pub const MIN_QUERY_WORDS: usize = 3;
/// Repositories in the licence cache.
pub const MAX_CACHE_REPOS: usize = 100_000;

/// A repository's licence as the cache states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepoLicence {
    Found(String),
    NotFound,
    NoLicence,
    LookupError,
}

/// Read the licence cache, checked against its pin.
pub fn read_cache(
    path: &Path,
    pin: &str,
) -> Result<(BTreeMap<String, RepoLicence>, decisions::Lines), String> {
    let mut out = BTreeMap::new();
    let got = decisions::for_lines(path, decisions::Role::Reference, |n, r| {
        let what = format!("licence cache line {n}");
        let repo = decisions::get_str(&r, "repo", &what)?.to_owned();
        let lic = match decisions::get_str(&r, "status", &what)? {
            "found" => RepoLicence::Found(
                r.get("key")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
            ),
            "not_found" => RepoLicence::NotFound,
            "no_licence" => RepoLicence::NoLicence,
            "error" => RepoLicence::LookupError,
            s => return Err(format!("{what}: status {s:?}")),
        };
        if out.insert(repo, lic).is_some() {
            return Err(format!("{what}: a repository twice"));
        }
        if out.len() > MAX_CACHE_REPOS {
            return Err(format!(
                "licence cache: more than {MAX_CACHE_REPOS} repositories"
            ));
        }
        Ok(())
    })?;
    decisions::check_pin(
        path,
        &got.sha256,
        pin,
        "the config (pools.csn.licence_cache_sha256)",
    )?;
    Ok((out, got))
}

/// The licence gate for a repository: its key through the mirror, or the cache's reason.
pub fn repo_licence(cache: &BTreeMap<String, RepoLicence>, repo: &str) -> Result<String, String> {
    match cache.get(repo) {
        None => Err("licence_not_in_cache".into()),
        Some(RepoLicence::NotFound) => Err("licence_repo_not_found".into()),
        Some(RepoLicence::NoLicence) => Err("licence_none_detected".into()),
        Some(RepoLicence::LookupError) => Err("licence_lookup_error".into()),
        Some(RepoLicence::Found(key)) => match convert_licence::gate(Some(key.as_str()))? {
            Verdict::Admitted(id) => Ok(id),
            Verdict::Pending(id) => Err(format!("licence_pending_not_for_code:{id}")),
        },
    }
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The first paragraph of a docstring, with Go's `//` markers removed, collapsed.
pub fn first_paragraph(doc: &str) -> String {
    let mut lines = Vec::new();
    for l in doc.lines() {
        let l = l.trim();
        let l = l.strip_prefix("//").map(str::trim).unwrap_or(l);
        if l.is_empty() {
            if !lines.is_empty() {
                break;
            }
            continue;
        }
        lines.push(l);
    }
    collapse(&lines.join(" "))
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `text` with every whole-word `name` replaced by `with`, and how many were replaced.
pub fn mask_word(text: &str, name: &str, with: &str) -> (String, usize) {
    if name.is_empty() {
        return (text.to_owned(), 0);
    }
    let mut out = String::with_capacity(text.len());
    let mut n = 0;
    let mut rest = text;
    while let Some(at) = rest.find(name) {
        let before = rest[..at].chars().next_back();
        let after = rest[at + name.len()..].chars().next();
        out.push_str(&rest[..at]);
        if before.is_none_or(|c| !is_ident(c)) && after.is_none_or(|c| !is_ident(c)) {
            out.push_str(with);
            n += 1;
        } else {
            out.push_str(name);
        }
        rest = &rest[at + name.len()..];
    }
    out.push_str(rest);
    (out, n)
}

/// Whether a function's short name is masked in its query: only an identifier-shaped name (an
/// underscore, a digit or an inner capital: `DeleteThing`, `get_area`) or one the query opens
/// with (Go's convention). A plain word (`area`, `run`) is the query's English, not a leak.
pub fn maskable(short: &str, query: &str) -> bool {
    let identifier = short.contains('_')
        || short.chars().any(|c| c.is_ascii_digit())
        || short.chars().skip(1).any(|c| c.is_ascii_uppercase());
    let opens = query
        .split(|c: char| !is_ident(c))
        .next()
        .is_some_and(|w| w == short);
    !short.is_empty() && (identifier || opens)
}

/// A Python function with its docstring removed: the first triple-quoted string that opens its
/// own line (after an optional `r`/`u`/`b` prefix), lines and all. `None` when there is none.
pub fn strip_py_docstring(code: &str) -> Option<String> {
    let open = ["\"\"\"", "'''"]
        .iter()
        .filter_map(|q| code.find(q).map(|i| (i, *q)))
        .min()?;
    let (at, q) = open;
    let line_start = code[..at].rfind('\n').map_or(0, |i| i + 1);
    let lead = code[line_start..at].trim();
    if !lead.is_empty()
        && !lead
            .chars()
            .all(|c| matches!(c, 'r' | 'u' | 'b' | 'R' | 'U' | 'B'))
    {
        return None;
    }
    let close = at + 3 + code[at + 3..].find(q)? + 3;
    let line_end = code[close..]
        .find('\n')
        .map_or(code.len(), |i| close + i + 1);
    if !code[close..line_end].trim().is_empty() {
        return None;
    }
    Some(format!("{}{}", &code[..line_start], &code[line_end..]))
}

/// One CodeSearchNet function, admitted.
#[derive(Clone, Debug)]
pub struct Func {
    pub key: [u8; 32],
    pub id: String,
    pub lang: &'static str,
    pub path: String,
    pub query: String,
    pub code: String,
    pub licence: String,
}

/// What one row of a CSN view is: an admitted function, or the reason it is refused.
pub fn csn_func(
    seed: u64,
    lang: &'static str,
    cache: &BTreeMap<String, RepoLicence>,
    r: &Value,
) -> Result<(String, Func, usize), String> {
    let repo = convert::opt_str(r, "repository_name").ok_or("malformed")?;
    let licence = repo_licence(cache, repo)?;
    let url = convert::opt_str(r, "func_code_url").ok_or("malformed")?;
    let path = convert::opt_str(r, "func_path_in_repository").ok_or("malformed")?;
    let name = convert::opt_str(r, "func_name").ok_or("malformed")?;
    let whole = convert::opt_str(r, "whole_func_string").ok_or("malformed")?;
    let doc = convert::opt_str(r, "func_documentation_string").ok_or("malformed")?;
    let para = first_paragraph(doc);
    let short = name.rsplit('.').next().unwrap_or(name);
    let (query, masked) = if maskable(short, &para) {
        mask_word(&para, short, "this function")
    } else {
        (para.clone(), 0)
    };
    if query.split_whitespace().count() < MIN_QUERY_WORDS {
        return Err("query_too_short".into());
    }
    if query.len() > MAX_QUERY_BYTES {
        return Err("query_too_long".into());
    }
    let code = match lang {
        "python" => strip_py_docstring(whole).ok_or("docstring_not_stripped")?,
        _ => whole.to_owned(),
    };
    let probe: String = para.chars().take(40).collect();
    if collapse(&code).contains(&probe) {
        return Err("docstring_not_stripped".into());
    }
    let id = sha256_hex(url.as_bytes())[..16].to_owned();
    Ok((
        repo.to_owned(),
        Func {
            key: decisions::keyed(seed, &["csn-function", &id]),
            id,
            lang,
            path: path.to_owned(),
            query,
            code,
            licence,
        },
        masked,
    ))
}

/// The CSN caps.
#[derive(Clone, Copy, Debug)]
pub struct CsnCaps {
    pub max_functions_per_repo: usize,
    pub max_queries_per_repo: usize,
    pub negatives_per_query: usize,
}

fn csn_row(repo: &str, q: &Func, f: &Func, positive: bool) -> Row {
    Row {
        id: if positive {
            format!("csn:{}:{}:pos", q.lang, q.id)
        } else {
            format!("csn:{}:{}:neg:{}", q.lang, q.id, f.id)
        },
        source_id: CSN,
        family_id: CSN_FAMILY,
        stratum: format!("{CSN_FAMILY}/{}", q.lang),
        group_key: repo.to_owned(),
        licence: Some(q.licence.clone()),
        context: format!("Query: {}\n\nFunction ({}):\n{}", q.query, f.path, f.code),
        question: CSN_QUESTION.to_owned(),
        slot_name: "matches",
        options: crate::convert_text::YES_NO.map(str::to_owned).to_vec(),
        gold: Gold::Option(if positive { 0 } else { 1 }),
        label_basis: if positive {
            "own_function"
        } else {
            "same_repo_other_function"
        },
    }
}

/// Every repository's rows: positives and same-repo hard negatives. Returns the number of
/// functions refused for want of a negative.
pub fn csn_rows(
    acc: &mut Acc<'_>,
    caps: CsnCaps,
    repos: BTreeMap<String, Vec<Func>>,
) -> Result<usize, String> {
    let seed = acc.cfg.seed;
    let mut no_negatives = 0;
    for (repo, mut funcs) in repos {
        funcs.sort_by_key(|f| f.key);
        if funcs.len() < 2 {
            no_negatives += funcs.len();
            continue;
        }
        for q in funcs.iter().take(caps.max_queries_per_repo) {
            acc.offer(CSN, CSN_FAMILY, Ok(csn_row(&repo, q, q, true)))?;
            let mut others: Vec<&Func> = funcs
                .iter()
                .filter(|f| f.id != q.id && f.query != q.query && f.code != q.code)
                .collect();
            others.sort_by_key(|f| decisions::keyed(seed, &["csn-negative", &q.id, &f.id]));
            for f in others.into_iter().take(caps.negatives_per_query) {
                acc.offer(CSN, CSN_FAMILY, Ok(csn_row(&repo, q, f, false)))?;
            }
        }
    }
    Ok(no_negatives)
}

pub fn build_csn(
    cfg: &Config,
    views: &Views,
    cache_path: &Path,
    mut digests: BTreeMap<String, String>,
    threads: usize,
) -> Result<decisions::Built, String> {
    let p = cfg.pool("csn")?;
    let caps = CsnCaps {
        max_functions_per_repo: convert::count(p, "max_functions_per_repo", "pools.csn", 10_000)?,
        max_queries_per_repo: convert::count(p, "max_queries_per_repo", "pools.csn", 10_000)?,
        negatives_per_query: convert::count(p, "negatives_per_query", "pools.csn", 15)?,
    };
    if caps.max_queries_per_repo > caps.max_functions_per_repo {
        return Err("pools.csn: max_queries_per_repo exceeds max_functions_per_repo".into());
    }
    let pin = decisions::get_str(p, "licence_cache_sha256", "pools.csn")?;
    let (cache, cache_sha) = read_cache(cache_path, pin)?;
    decisions::record_input(&mut digests, "csn_licence_cache".to_owned(), cache_sha);
    let mut functions = Tally::default();
    let mut masked = 0usize;
    let mut repos: BTreeMap<String, Vec<Func>> = BTreeMap::new();
    let mut reservoir_dropped = 0usize;
    for lang in CSN_LANGS {
        let file = format!("{lang}/train-00000-of-00001.parquet");
        let got = convert::read_view(views.train_rows(CSN, &file)?, |_, r| {
            functions.offered += 1;
            match csn_func(cfg.seed, lang, &cache, &r) {
                Err(reason) => functions.refuse(&reason),
                Ok((repo, f, m)) => {
                    functions.kept += 1;
                    masked += usize::from(m > 0);
                    let v = repos.entry(format!("{lang}:{repo}")).or_default();
                    v.push(f);
                    // A bounded reservoir: the seeded-smallest keys survive, the rest go now.
                    if v.len() >= 2 * caps.max_functions_per_repo {
                        v.sort_by_key(|f| f.key);
                        reservoir_dropped += v.len() - caps.max_functions_per_repo;
                        v.truncate(caps.max_functions_per_repo);
                    }
                }
            }
            if functions.offered > convert::MAX_OFFERED {
                return Err(format!("more than {} functions", convert::MAX_OFFERED));
            }
            Ok(())
        })?;
        decisions::record_input(&mut digests, format!("{CSN}/{file}"), got);
    }
    for v in repos.values_mut() {
        if v.len() > caps.max_functions_per_repo {
            v.sort_by_key(|f| f.key);
            reservoir_dropped += v.len() - caps.max_functions_per_repo;
            v.truncate(caps.max_functions_per_repo);
        }
    }
    // The group key is the repository without the language prefix; a repository's python and
    // go functions (none expected) would share a group.
    let mut by_repo: BTreeMap<String, Vec<Func>> = BTreeMap::new();
    let n_repos = repos.len();
    for (k, v) in repos {
        let repo = k.split_once(':').map_or(k.as_str(), |(_, r)| r).to_owned();
        by_repo.entry(repo).or_default().extend(v);
    }
    let mut acc = Acc::new(cfg);
    let no_negatives = csn_rows(&mut acc, caps, by_repo)?;
    let mut refs: Vec<(String, String)> = Vec::new();
    for lang in CSN_LANGS {
        for split in ["test", "validation"] {
            refs.push((
                format!("csn-{lang}-{split}"),
                format!("{lang}/{split}-00000-of-00001.parquet"),
            ));
        }
    }
    let r: Vec<(&str, &str, &str)> = refs
        .iter()
        .map(|(n, f)| (n.as_str(), CSN, f.as_str()))
        .collect();
    let (targets, td) = views.target_texts(&r)?;
    digests.extend(td);
    let facts = PoolFacts {
        name: "csn",
        licence_notes: BTreeMap::from([(
            CSN,
            "per row: the repository's licence as GitHub detects it \
            today on the default branch (csn_licences.py, 2026-10-06), not at the 2019 commit; \
            default-deny for not found, none detected, lookup error or a key outside \
            qd_data.licences' allowlist. Attribution: Husain et al., CodeSearchNet (2019); each \
            function's own repository and licence apply"
                .to_owned(),
        )]),
        caps: json!({
            "max_functions_per_repo": caps.max_functions_per_repo,
            "max_queries_per_repo": caps.max_queries_per_repo,
            "negatives_per_query": caps.negatives_per_query,
            "functions": functions.json(),
            "functions_dropped_by_reservoir": reservoir_dropped,
            "functions_without_negatives": no_negatives,
            "repositories_with_admitted_functions": n_repos,
            "queries_with_own_name_masked": masked,
        }),
        id_checks: json!({"state": "not_applicable",
            "reason": "CodeSearchNet's train/test/validation split is by repository upstream; text \
                       containment against test and validation covers forks and copies"}),
        notes: json!({
            "pretraining_exposure": "CodeSearchNet (2019, public GitHub code) is very likely in the \
                base model's pretraining data; a score on this family may measure recall, not search",
            "docstring": "Python docstrings removed from the function text; a function whose \
                docstring's opening survives is refused (docstring_not_stripped)",
            "name_leak": "Go docs open with the function's name; the short name is masked in the \
                query as 'this function' (count above). The positive's code still contains it",
            "negatives": "same-repository functions; one may also do what the query describes \
                (label noise unmeasured)",
        }),
    };
    convert::build(acc, targets, digests, facts, threads)
}

pub const SWE_REBENCH: &str = "nebius/SWE-rebench";
pub const SWE_FILES: [&str; 3] = [
    "data/filtered-00000-of-00001.parquet",
    "data/test-00000-of-00002.parquet",
    "data/test-00001-of-00002.parquet",
];
pub const SWE_TARGETS: [(&str, &str); 6] = [
    ("princeton-nlp/SWE-bench", "data/dev-00000-of-00001.parquet"),
    (
        "princeton-nlp/SWE-bench",
        "data/test-00000-of-00001.parquet",
    ),
    (
        "princeton-nlp/SWE-bench",
        "data/train-00000-of-00001.parquet",
    ),
    (
        "princeton-nlp/SWE-bench_Lite",
        "data/dev-00000-of-00001.parquet",
    ),
    (
        "princeton-nlp/SWE-bench_Lite",
        "data/test-00000-of-00001.parquet",
    ),
    ("czlll/Loc-Bench_V1", "data/test-00000-of-00001.parquet"),
];
/// Rows written to the filtered view.
pub const MAX_SWE_ROWS: usize = 200_000;

/// One SWE-rebench row's verdict: its admitted licence id, or the refusal.
pub fn swe_verdict(
    r: &Value,
    excluded: &BTreeSet<String>,
    seen: &mut BTreeSet<String>,
) -> Result<String, String> {
    let id = convert::opt_str(r, "instance_id").ok_or("malformed")?;
    if excluded.contains(id) {
        return Err("instance_in_swe_bench_lite_or_locbench".into());
    }
    if !seen.insert(id.to_owned()) {
        return Err("duplicate_instance_id".into());
    }
    match convert_licence::gate(convert::opt_str(r, "license_name"))? {
        Verdict::Admitted(l) => Ok(l),
        Verdict::Pending(l) => Err(format!("licence_pending_not_for_code:{l}")),
    }
}

/// SWE-rebench's row filter: `DIR/{rows.jsonl, counts.json}`.
pub fn swe_rebench_filter(
    _cfg: &Config,
    views: &Views,
    mut digests: BTreeMap<String, String>,
    out_dir: &Path,
) -> Result<String, String> {
    let mut excluded = BTreeSet::new();
    let mut per_target = serde_json::Map::new();
    for (dataset, file) in SWE_TARGETS {
        let before = excluded.len();
        let got = views.target_rows(dataset, file)?.read(|_, r| {
            if let Some(id) = convert::opt_str(&r, "instance_id") {
                excluded.insert(id.to_owned());
            }
            Ok(())
        })?;
        decisions::record_input(&mut digests, format!("target-rows/{dataset}/{file}"), got);
        per_target.insert(format!("{dataset}/{file}"), json!(excluded.len() - before));
    }
    let mut rows = Vec::new();
    let mut tally = Tally::default();
    let mut names: BTreeMap<String, (String, usize)> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for file in SWE_FILES {
        let got = convert::read_view(views.train_rows(SWE_REBENCH, file)?, |_, r| {
            tally.offered += 1;
            let name = convert::opt_str(&r, "license_name")
                .unwrap_or("<null>")
                .to_owned();
            let tier = match convert::opt_str(&r, "license_name") {
                Some(n) => convert_licence::classify(n).1.as_str().to_owned(),
                None => "unstated".to_owned(),
            };
            names.entry(name).or_insert((tier, 0)).1 += 1;
            match swe_verdict(&r, &excluded, &mut seen) {
                Err(reason) => tally.refuse(&reason),
                Ok(l) => {
                    if tally.kept >= MAX_SWE_ROWS {
                        return Err(format!("more than {MAX_SWE_ROWS} SWE-rebench rows kept"));
                    }
                    tally.kept += 1;
                    *tally.licence_admitted.entry(l.clone()).or_default() += 1;
                    serde_json::to_writer(&mut rows, &json!({"licence": l, "row": r}))
                        .map_err(|e| e.to_string())?;
                    rows.push(b'\n');
                }
            }
            Ok(())
        })?;
        decisions::record_input(&mut digests, format!("{SWE_REBENCH}/{file}"), got);
    }
    let counts = json!({
        "schema": "qd-convert-swe-rebench-filter/v1", "inputs": digests,
        "not_viewed": views.not_viewed,
        "rows": tally.json(), "rows_sha256": sha256_hex(&rows),
        "excluded_instance_ids_by_target": per_target, "excluded_instance_ids": excluded.len(),
        "license_name_values": names.iter().map(|(n, (t, c))| (n.clone(), json!({"tier": t, "rows": c})))
            .collect::<serde_json::Map<_, _>>(),
        "parked": "a filtered view, not a pool: the localisation family needs each repository's \
                   tree at base_commit, which is not on this machine",
        "licence_rule": "license_name through the Rust mirror of qd_data.licences: display names \
                         Python has no alias for (\"MIT License\" has one; \"Apache License 2.0\", \
                         \"BSD 3-Clause ...\" do not) are refused until a human registers them",
    });
    let mut cb = serde_json::to_vec_pretty(&counts).map_err(|e| e.to_string())?;
    cb.push(b'\n');
    convert::write_dir(
        out_dir,
        &[("rows.jsonl", &rows[..]), ("counts.json", &cb[..])],
    )?;
    Ok(format!(
        "qd-prep convert swe-rebench-filter: {} of {} rows kept -> {}",
        tally.kept,
        tally.offered,
        out_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> BTreeMap<String, RepoLicence> {
        BTreeMap::from([
            ("ok/repo".to_owned(), RepoLicence::Found("mit".into())),
            ("gpl/repo".to_owned(), RepoLicence::Found("gpl-3.0".into())),
            ("gone/repo".to_owned(), RepoLicence::NotFound),
            ("bare/repo".to_owned(), RepoLicence::NoLicence),
        ])
    }

    #[test]
    fn repository_licences_default_deny() {
        let c = cache();
        assert_eq!(repo_licence(&c, "ok/repo").unwrap(), "mit");
        assert_eq!(
            repo_licence(&c, "gpl/repo").unwrap_err(),
            "licence_needs_human_call:gpl-3.0"
        );
        assert_eq!(
            repo_licence(&c, "gone/repo").unwrap_err(),
            "licence_repo_not_found"
        );
        assert_eq!(
            repo_licence(&c, "bare/repo").unwrap_err(),
            "licence_none_detected"
        );
        assert_eq!(
            repo_licence(&c, "unseen/repo").unwrap_err(),
            "licence_not_in_cache"
        );
    }

    fn py_row(repo: &str, name: &str, whole: &str, doc: &str) -> Value {
        json!({"repository_name": repo, "func_path_in_repository": "pkg/mod.py", "func_name": name,
               "whole_func_string": whole, "language": "python", "func_code_string": whole,
               "func_code_tokens": ["def"], "func_documentation_string": doc,
               "func_documentation_tokens": ["x"], "split_name": "train",
               "func_code_url": format!("https://github.com/{repo}/blob/abc/pkg/mod.py#L{}", whole.len())})
    }

    #[test]
    fn the_python_docstring_is_stripped_and_a_survivor_is_refused() {
        let whole = "def area(self, w, h):\n        \"\"\"\n        Compute the area of a rectangle from width and height.\n\n        :return: area\n        \"\"\"\n        return w * h";
        let doc = "Compute the area of a rectangle from width and height.\n\n        :return: area";
        let (_, f, _) = csn_func(
            1,
            "python",
            &cache(),
            &py_row("ok/repo", "Shape.area", whole, doc),
        )
        .unwrap();
        assert!(!f.code.contains("Compute the area"), "{}", f.code);
        assert!(f.code.contains("return w * h"));
        // "area" is the method's name and a plain word: not masked.
        assert_eq!(
            f.query,
            "Compute the area of a rectangle from width and height."
        );
        assert!(maskable("get_area", "Return it."));
        assert!(maskable("run", "run the job."));
        assert!(!maskable("run", "Start the job and run it."));
        // A single-quoted docstring is not removed by the rule: refused, never kept with a leak.
        let single = "def area(w, h):\n    \"Compute the area of a rectangle from width and height.\"\n    return w * h";
        assert_eq!(
            csn_func(
                1,
                "python",
                &cache(),
                &py_row(
                    "ok/repo",
                    "area",
                    single,
                    "Compute the area of a rectangle from width and height."
                )
            )
            .unwrap_err(),
            "docstring_not_stripped"
        );
        assert_eq!(
            csn_func(
                1,
                "python",
                &cache(),
                &py_row("gpl/repo", "area", whole, doc)
            )
            .unwrap_err(),
            "licence_needs_human_call:gpl-3.0"
        );
    }

    #[test]
    fn go_docs_lose_their_markers_and_the_function_s_own_name() {
        let doc = "// DeleteThing removes the thing named by the request.\n// More detail here.";
        assert_eq!(
            first_paragraph(doc),
            "DeleteThing removes the thing named by the request. More detail here."
        );
        let (q, n) = mask_word(
            "DeleteThing removes; DeleteThings stays; (DeleteThing)",
            "DeleteThing",
            "this function",
        );
        assert_eq!(
            q,
            "this function removes; DeleteThings stays; (this function)"
        );
        assert_eq!(n, 2);
        let r = json!({"repository_name": "ok/repo", "func_path_in_repository": "a.go", "func_name": "Client.DeleteThing",
            "whole_func_string": "func (c *Client) DeleteThing(id string) error {\n\treturn c.del(id)\n}",
            "func_documentation_string": doc, "func_code_url": "https://github.com/ok/repo/blob/x/a.go#L1-L3"});
        let (_, f, m) = csn_func(1, "go", &cache(), &r).unwrap();
        assert!(f.query.starts_with("this function removes"));
        assert_eq!(m, 1);
    }

    fn func(id: &str, query: &str) -> Func {
        Func {
            key: decisions::keyed(1, &[id]),
            id: id.into(),
            lang: "python",
            path: "p.py".into(),
            query: query.into(),
            code: format!("def {id}(): pass"),
            licence: "mit".into(),
        }
    }

    #[test]
    fn negatives_come_from_the_same_repo_and_the_split_is_by_repo() {
        let cfg = crate::convert::tests::cfg();
        let mut acc = Acc::new(&cfg);
        let caps = CsnCaps {
            max_functions_per_repo: 10,
            max_queries_per_repo: 2,
            negatives_per_query: 1,
        };
        let repos = BTreeMap::from([
            (
                "a/one".to_owned(),
                vec![
                    func("f1", "q one a"),
                    func("f2", "q two a"),
                    func("f3", "q three a"),
                ],
            ),
            ("b/two".to_owned(), vec![func("g1", "only one here")]),
        ]);
        let none = csn_rows(&mut acc, caps, repos).unwrap();
        assert_eq!(none, 1);
        assert_eq!(acc.rows.len(), 4);
        assert!(acc.rows.iter().all(|c| c.group_key == "a/one"));
        let splits: BTreeSet<&str> = acc.rows.iter().map(|c| c.split).collect();
        assert_eq!(splits.len(), 1);
        assert_eq!(
            acc.rows
                .iter()
                .filter(|c| crate::convert::gold_class(c) == "yes")
                .count(),
            2
        );
    }

    #[test]
    fn swe_rebench_rows_in_an_eval_or_without_an_admitted_licence_are_refused() {
        let excluded = BTreeSet::from(["astropy__astropy-1".to_owned()]);
        let mut seen = BTreeSet::new();
        let row = |id: &str, lic: Value| json!({"instance_id": id, "license_name": lic});
        assert_eq!(
            swe_verdict(
                &row("astropy__astropy-1", json!("MIT License")),
                &excluded,
                &mut seen
            )
            .unwrap_err(),
            "instance_in_swe_bench_lite_or_locbench"
        );
        assert_eq!(
            swe_verdict(&row("a__b-2", json!("MIT License")), &excluded, &mut seen).unwrap(),
            "mit"
        );
        assert_eq!(
            swe_verdict(&row("a__b-2", json!("MIT License")), &excluded, &mut seen).unwrap_err(),
            "duplicate_instance_id"
        );
        assert_eq!(
            swe_verdict(&row("a__b-3", Value::Null), &excluded, &mut seen).unwrap_err(),
            "licence_unstated"
        );
        assert!(
            swe_verdict(&row("a__b-4", json!("BSD License")), &excluded, &mut seen)
                .unwrap_err()
                .starts_with("licence_needs_human_call")
        );
    }
}
