//! G6: unseen-language noul rows from real code (`campaign/v5-preregistered.DRAFT.json`
//! `data.sources[6]`).
//!
//! v3b's unseen-language rows are 834 instantiations of 42 templates in six languages; F scored
//! 20/0/9 of 60 on the OOD suite's unseen-language cases, and Fable read the repeats as
//! memorisation of a narrow feature. G6 adds **real** commits from bigcode/commitpackft in eight
//! languages no suite and no other source holds (clojure, erlang, fortran, julia, ocaml, perl, r,
//! tcl; `qd_data.defect_class.G6_LANGUAGES`), downloaded on the human's yes and sha-pinned
//! (`AUDIT/v5-plan-2026-10-02/g6-commitpackft-download.md`).
//!
//! **The shape is the corpus's.** Each row is the commit's own change, `old_contents` ->
//! `new_contents`, rendered by [`qd_mutate::diffspan::unified_multi`] at context 3 -- the renderer
//! and context `commitpackft-corpus-v3` was generated with (its manifest's `diff.renderer
//! multi_hunk, context 3`) -- so neither the hunk count nor the header form tells a G6 row from a
//! defect row. A band on hunks and characters keeps the length distribution near the corpus's.
//!
//! **Which rows may be read is Python's.** The licence filter (`admit_licence`, the per-row
//! filter every corpus-v3 row passes) and the split (`assign_repo` on the primary repo) are
//! computed by `qd_data.defect_class.noul_v5_allowlist`, never here; this binary reads the
//! allowlist, checks every input file against the sha256 it pins, and renders only the lines it
//! lists. The loader (`qd_data.defect_class._load_g6`) re-derives both for every row.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use qd_mutate::diffspan::unified_multi;
use qd_mutate::manifest::sha256_hex;

use crate::allowlist::Split;
use crate::hunk::{has_char_in, order_key, short_hash};

pub const SCHEMA: &str = "qd-noul-g6/v1";
/// The noul source a G6 row teaches: the OOD suite's `unseen-language` category.
pub const SOURCE: &str = "unseen-language";
pub const ROUTE: &str = "commitpackft-g6";
pub const FORM: &str = "commit";
/// `qd_data.defect_class.G6_LANGUAGES`; the allowlist must name exactly these.
pub const LANGUAGES: [&str; 8] = [
    "clojure", "erlang", "fortran", "julia", "ocaml", "perl", "r", "tcl",
];
/// `commitpackft-corpus-v3`'s manifest: `diff.renderer multi_hunk`, `context 3`.
pub const CONTEXT: usize = 3;
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_LINES: usize = 100_000;

#[derive(Deserialize)]
struct G6File {
    file: String,
    sha256: String,
    rows: u64,
}

#[derive(Deserialize)]
struct G6Allow {
    root: String,
    files: BTreeMap<String, G6File>,
    /// Per language: `[line number, primary repo, licence]` of every admitted row.
    rows: BTreeMap<String, Vec<(u64, String, String)>>,
    counts: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct V5Allowlist {
    schema: String,
    split: Split,
    invisible_format_ranges: Vec<(u32, u32)>,
    g6: G6Allow,
}

#[derive(Deserialize)]
struct CommitRow {
    commit: String,
    new_file: String,
    old_contents: String,
    new_contents: String,
    license: String,
    repos: String,
}

#[derive(Serialize, Clone)]
struct Row {
    id: String,
    class: &'static str,
    noul_source: &'static str,
    noul_route: &'static str,
    noul_form: &'static str,
    repo: String,
    path: String,
    language: String,
    diff: String,
    licence: String,
    commit: String,
    seed: u64,
    tool_version: &'static str,
}

pub struct Args<'a> {
    pub allowlist: &'a Path,
    pub per_language_cap: usize,
    pub max_per_repo: usize,
    pub min_chars: usize,
    pub max_chars: usize,
    pub max_hunks: usize,
    pub seed: u64,
    pub out: &'a Path,
}

/// Why a candidate row was not rendered, or `Ok(diff)`.
pub fn render(
    old: &str,
    new: &str,
    path: &str,
    band: (usize, usize, usize),
    ranges: &[(u32, u32)],
) -> std::result::Result<String, &'static str> {
    let (min_chars, max_chars, max_hunks) = band;
    if path.contains(['\n', '\r'])
        || Path::new(path)
            .file_name()
            .is_some_and(|n| n == "notes.txt")
    {
        return Err("path_refused");
    }
    if old.trim().is_empty() || new.trim().is_empty() {
        return Err("added_or_deleted_file");
    }
    let diff = unified_multi(old, new, CONTEXT).map_err(|_| "diff_refused")?;
    if diff.is_empty() {
        return Err("no_change");
    }
    let hunks = diff.lines().filter(|l| l.starts_with("@@ -")).count();
    if hunks == 0 || hunks > max_hunks {
        return Err("hunks_out_of_band");
    }
    let chars = diff.chars().count();
    if chars < min_chars {
        return Err("chars_under_min");
    }
    if chars > max_chars {
        return Err("chars_over_max");
    }
    if has_char_in(&diff, ranges) || has_char_in(path, ranges) {
        return Err("invisible_format_character");
    }
    Ok(diff)
}

fn bump(map: &mut BTreeMap<String, u64>, reason: &str) {
    *map.entry(reason.to_string()).or_insert(0) += 1;
}

pub fn run(args: &Args<'_>) -> Result<()> {
    let examples_path = args.out.join("examples.jsonl");
    let manifest_path = args.out.join("manifest.json");
    if examples_path.exists() || manifest_path.exists() {
        bail!(
            "{} already holds a corpus; a corpus is written once",
            args.out.display()
        );
    }
    if args.per_language_cap == 0
        || args.max_per_repo == 0
        || args.max_hunks == 0
        || args.min_chars > args.max_chars
    {
        bail!("caps must be positive and --min-chars <= --max-chars");
    }
    let allow_bytes = fs::read(args.allowlist)
        .with_context(|| format!("reading {}", args.allowlist.display()))?;
    let allow: V5Allowlist =
        serde_json::from_slice(&allow_bytes).context("parsing the v5 allowlist")?;
    if allow.schema != "qd-noul-v5-allowlist/v1" {
        bail!("allowlist schema {:?}", allow.schema);
    }
    let langs: BTreeSet<&str> = allow.g6.files.keys().map(String::as_str).collect();
    if langs != LANGUAGES.iter().copied().collect::<BTreeSet<_>>() {
        bail!("the allowlist names languages {langs:?}, not G6's {LANGUAGES:?}");
    }
    let ranges = &allow.invisible_format_ranges;
    let band = (args.min_chars, args.max_chars, args.max_hunks);
    let root = PathBuf::from(&allow.g6.root);

    let mut rows: Vec<Row> = Vec::new();
    let mut funnel: BTreeMap<String, BTreeMap<&'static str, u64>> = BTreeMap::new();
    let mut skipped: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for lang in LANGUAGES {
        let pin = &allow.g6.files[lang];
        let path = root.join(&pin.file);
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let sha = sha256_hex(&bytes);
        if sha != pin.sha256 {
            bail!(
                "{} hashes to {sha}; the allowlist pins {}",
                path.display(),
                pin.sha256
            );
        }
        let admitted: BTreeMap<u64, (&str, &str)> = allow.g6.rows[lang]
            .iter()
            .map(|(n, repo, lic)| (*n, (repo.as_str(), lic.as_str())))
            .collect();
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("{} is not UTF-8", path.display()))?;
        let mut candidates: Vec<([u8; 32], Row)> = Vec::new();
        let skips = skipped.entry(lang.to_string()).or_default();
        let mut n_lines = 0u64;
        for (i, line) in text.lines().enumerate() {
            let lineno = i as u64 + 1;
            n_lines = lineno;
            if lineno as usize > MAX_LINES || line.len() > MAX_LINE_BYTES {
                bail!("{}:{lineno} is past the file bounds", path.display());
            }
            let Some((repo, licence)) = admitted.get(&lineno) else {
                continue;
            };
            let row: CommitRow = serde_json::from_str(line)
                .with_context(|| format!("{}:{lineno}", path.display()))?;
            let primary = row.repos.split(',').next().unwrap_or("").trim();
            if primary != *repo || row.license.trim().to_ascii_lowercase() != *licence {
                bail!(
                    "{}:{lineno}: the allowlist says ({repo}, {licence}), the row ({primary}, {})",
                    path.display(),
                    row.license
                );
            }
            match render(
                &row.old_contents,
                &row.new_contents,
                &row.new_file,
                band,
                ranges,
            ) {
                Ok(diff) => {
                    let key = order_key(
                        args.seed,
                        ROUTE,
                        &format!("{lang}\0{}\0{}", row.commit, row.new_file),
                    );
                    candidates.push((
                        key,
                        Row {
                            id: format!("noul:{ROUTE}:{}", short_hash(&[ROUTE, repo, &diff])),
                            class: "noul",
                            noul_source: SOURCE,
                            noul_route: ROUTE,
                            noul_form: FORM,
                            repo: repo.to_string(),
                            path: row.new_file,
                            language: lang.to_string(),
                            diff,
                            licence: licence.to_string(),
                            commit: row.commit,
                            seed: args.seed,
                            tool_version: qd_mutate::TOOL_VERSION,
                        },
                    ));
                }
                Err(reason) => bump(skips, reason),
            }
        }
        if n_lines != pin.rows {
            bail!(
                "{} holds {n_lines} rows, the allowlist read {}",
                path.display(),
                pin.rows
            );
        }
        let rendered = candidates.len() as u64;
        candidates.sort_by_key(|c| c.0);
        let mut per_repo: BTreeMap<String, usize> = BTreeMap::new();
        let mut seen_diffs: BTreeSet<String> = BTreeSet::new();
        let mut selected = 0u64;
        for (_, row) in candidates {
            if selected as usize == args.per_language_cap {
                bump(skips, "over_per_language_cap");
                continue;
            }
            if !seen_diffs.insert(row.diff.clone()) {
                bump(skips, "duplicate_diff");
                continue;
            }
            let n = per_repo.entry(row.repo.clone()).or_insert(0);
            if *n >= args.max_per_repo {
                bump(skips, "over_max_per_repo");
                continue;
            }
            *n += 1;
            selected += 1;
            rows.push(row);
        }
        funnel.insert(
            lang.to_string(),
            BTreeMap::from([
                ("admitted_by_allowlist", admitted.len() as u64),
                ("rendered_in_band", rendered),
                ("selected", selected),
            ]),
        );
    }

    let ids: BTreeSet<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    if ids.len() != rows.len() {
        bail!("{} rows, {} distinct ids", rows.len(), ids.len());
    }
    let mut body = String::new();
    let mut by_language: BTreeMap<&str, u64> = BTreeMap::new();
    let mut by_licence: BTreeMap<&str, u64> = BTreeMap::new();
    let mut repos: BTreeSet<&str> = BTreeSet::new();
    for r in &rows {
        body.push_str(&serde_json::to_string(r)?);
        body.push('\n');
        *by_language.entry(&r.language).or_insert(0) += 1;
        *by_licence.entry(&r.licence).or_insert(0) += 1;
        repos.insert(&r.repo);
    }
    let examples_sha256 = sha256_hex(body.as_bytes());
    let manifest = serde_json::json!({
        "schema": SCHEMA,
        "tool": "qd-noul-rows g6",
        "tool_version": qd_mutate::TOOL_VERSION,
        "seed": args.seed,
        "split": allow.split,
        "allowlist_sha256": sha256_hex(&allow_bytes),
        "root": allow.g6.root,
        "files": allow.g6.files.iter().map(|(k, v)| (k.clone(), serde_json::json!({"file": v.file, "sha256": v.sha256, "rows": v.rows}))).collect::<BTreeMap<_, _>>(),
        "licence": {
            "licence": null,
            "basis": "per row: bigcode/commitpackft's license field, admitted by qd_data.licences.admit_licence under DataConfig().licence -- the per-row filter every commitpackft-corpus-v3 row passes; dataset card licence mit",
        },
        "per_language_cap": args.per_language_cap,
        "max_per_repo": args.max_per_repo,
        "diff_band": {
            "renderer": "qd_mutate::diffspan::unified_multi (commitpackft-corpus-v3's multi_hunk renderer)",
            "context": CONTEXT,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "max_hunks": args.max_hunks,
        },
        "examples_sha256": examples_sha256,
        "allowlist_counts": allow.g6.counts,
        "funnel": funnel,
        "skipped": skipped,
        "totals": {"examples": rows.len(), "by_language": by_language, "by_licence": by_licence, "repos": repos.len()},
    });
    fs::create_dir_all(args.out).with_context(|| format!("creating {}", args.out.display()))?;
    crate::write_atomic(&examples_path, body.as_bytes())?;
    let mut text = serde_json::to_string_pretty(&manifest)?;
    text.push('\n');
    crate::write_atomic(&manifest_path, text.as_bytes())?;
    println!(
        "g6: {} rows {by_language:?} from {} repos -> {}  examples sha256 {examples_sha256}",
        rows.len(),
        repos.len(),
        examples_path.display()
    );
    println!("g6 funnel: {}", serde_json::to_string(&manifest["funnel"])?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAND: (usize, usize, usize) = (10, 4000, 5);

    #[test]
    fn a_real_change_renders_as_the_corpus_renders_it() {
        let old = "(ns a)\n(defn f [x]\n  (inc x))\n";
        let new = "(ns a)\n(defn f [x]\n  (dec x))\n";
        let diff = render(old, new, "src/a.clj", BAND, &[]).unwrap();
        assert_eq!(diff, unified_multi(old, new, CONTEXT).unwrap());
        assert!(diff.starts_with("@@ -"));
        assert!(diff.contains("\n-  (inc x))\n+  (dec x))\n"), "{diff}");
    }

    #[test]
    fn every_reason_a_commit_has_no_row_is_named() {
        let (o, n) = ("a\nb\n", "a\nc\n");
        assert_eq!(render(o, n, "notes.txt", BAND, &[]), Err("path_refused"));
        assert_eq!(render(o, n, "a\nb", BAND, &[]), Err("path_refused"));
        assert_eq!(
            render("", n, "x.pl", BAND, &[]),
            Err("added_or_deleted_file")
        );
        assert_eq!(
            render(o, "  \n", "x.pl", BAND, &[]),
            Err("added_or_deleted_file")
        );
        assert_eq!(render(o, o, "x.pl", BAND, &[]), Err("no_change"));
        assert_eq!(
            render(o, n, "x.pl", (1000, 4000, 5), &[]),
            Err("chars_under_min")
        );
        assert_eq!(render(o, n, "x.pl", (1, 5, 5), &[]), Err("chars_over_max"));
        let far_old: String = (0..40).map(|i| format!("l{i}\n")).collect();
        let far_new = far_old.replace("l1\n", "x1\n").replace("l30\n", "x30\n");
        assert_eq!(
            render(&far_old, &far_new, "x.pl", (1, 4000, 1), &[]),
            Err("hunks_out_of_band")
        );
        assert_eq!(
            render(o, "a\n\u{200B}\n", "x.pl", BAND, &[(0x200B, 0x200B)]),
            Err("invisible_format_character")
        );
    }
}
