//! `qd-noul-rows` — `code.defect_class` rows whose gold is `noul`.
//!
//! The decision model answers a defect question over a diff. Asked the same question over
//! something that is not a diff of its kind -- English prose, code scrambled past reading, code in
//! a language it never saw -- the right answer is `noul`, and the OOD gate measured that the model
//! does not give it (prose 31/60, scrambled 1/60, unseen language 0/60 on the 2026-09-30 probe).
//! CLINC's out-of-scope rows teach `noul` on intent questions and it did not transfer to
//! code-shaped inputs. This binary writes the rows that teach it on the defect question itself.
//!
//! Three sources in equal shares, each disjoint from the OOD suite's (`qd_train.ood`) by source:
//! [`prose`] (SQuAD paragraphs), [`scramble`] (pool files, shuffled), [`templates`] (six
//! languages neither the pool nor the suite holds). Every row is one hunk in the corpus's exact
//! shape ([`hunk`]), so the label is not readable off the shape.
//!
//! Two subcommands:
//!
//! * `units` — the template split units, for the allowlist step. The split is the canonical
//!   Python one (see [`allowlist`]); this binary never computes it.
//! * `generate` — read the allowlist, check every input against the sha256 it records, write
//!   `examples.jsonl` and `manifest.json`. Byte-identical for the same inputs and seed.
//!
//! `generate --forms v2 --corpus <dir>` writes defect-noul-v2 (the v3.1 re-form, Fable F2,
//! 2026-10-01): the same share and thirds, with each third's FORM widened toward what the OOD
//! gate measured did not transfer -- prose half paragraphs, half short questions
//! ([`questions`]); scrambled half line-shuffled, half token-shuffled diffs of the corpus
//! itself ([`corpus`]); unseen-language as v1. `--forms v1`, the default, writes v1 byte for
//! byte (pinned by `tests/noul_rows.rs`).
//!
//! Exit codes: `0` on success, `2` on any refusal, with the reason on stderr.

mod allowlist;
mod corpus;
mod hunk;
mod prose;
mod questions;
mod scramble;
mod templates;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;

use qd_mutate::manifest::sha256_hex;
use qd_mutate::pool;

use allowlist::Allowlist;

/// The class every row carries. `qd_data.defect_class.NOUL_CLASS` on the Python side.
const NOUL_CLASS: &str = "noul";
/// A share large enough to be a mistake rather than a corpus.
const MAX_PER_SOURCE: usize = 100_000;

#[derive(Parser, Debug)]
#[command(name = "qd-noul-rows", version, about = "code.defect_class rows whose gold is noul")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print the template catalogue's split units and its sha256, as JSON.
    Units,
    /// Write examples.jsonl and manifest.json from an allowlist and the inputs it names.
    Generate(GenerateArgs),
}

#[derive(clap::Args, Debug)]
struct GenerateArgs {
    /// The allowlist `tools/noul_allowlist.py` wrote.
    #[arg(long)]
    allowlist: PathBuf,
    /// SQuAD v2 train JSONL (the general fetch record's `rajpurkar/squad_v2` train file).
    #[arg(long)]
    squad: PathBuf,
    /// The commitpackft pool JSONL (`data/pool/commitpackft-pool-v2.jsonl`).
    #[arg(long)]
    pool: PathBuf,
    /// Output directory. Refused if it already holds a corpus.
    #[arg(long)]
    out: PathBuf,
    /// Rows per source; the corpus holds three times this. Default 834. Not with
    /// `--form-counts`, which sets every form's count itself.
    #[arg(long)]
    per_source: Option<usize>,
    /// `--forms v2` only: every form's row count, instead of equal thirds in exact halves, as
    /// `paragraph=N,question=N,lines=N,tokens=N,template=N` (all five, each at least 1).
    /// Recorded in the manifest as `form_counts`.
    #[arg(long)]
    form_counts: Option<String>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Which row forms to write. `v1` (default) is R2's defect-noul-v1, byte for byte.
    #[arg(long, value_enum, default_value_t = Forms::V1)]
    forms: Forms,
    /// `--forms v2` only: the `code.defect_class` corpus directory (`examples.jsonl` and
    /// `manifest.json`) whose diffs the scrambled rows are made from. Its manifest must name the
    /// allowlist's pool.
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// `--forms v2` only: a JSON object recorded verbatim as the manifest's `preregistered`
    /// (the bars this corpus is judged by, fixed before any run reads it).
    #[arg(long)]
    preregistered: Option<PathBuf>,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Forms {
    /// Prose paragraphs, scrambled pool windows, unseen-language templates.
    V1,
    /// Prose half paragraphs / half short questions; scrambled half line-shuffled / half
    /// token-shuffled corpus diffs; unseen-language templates.
    V2,
}

#[derive(Serialize)]
struct Row<'a> {
    id: String,
    class: &'static str,
    noul_source: &'static str,
    /// The form within the source (`--forms v2` only; absent in v1, whose bytes are pinned).
    #[serde(skip_serializing_if = "Option::is_none")]
    noul_form: Option<&'static str>,
    /// The split unit: `squad-title:<title>`, the pool repo, or `noul-template/<lang>/<name>`.
    repo: String,
    path: String,
    language: &'a str,
    diff: String,
    pool_id: Option<String>,
    squad_title: Option<String>,
    template: Option<String>,
    licence: String,
    seed: u64,
    tool_version: &'static str,
}

#[derive(Serialize)]
struct InputFile {
    file: String,
    sha256: String,
}

#[derive(Serialize)]
struct PoolMeta {
    /// File name only: the loader re-anchors it under `data/pool/`, as it does the main
    /// corpus's pool, so the manifest reads the same in every checkout.
    path: String,
    sha256: String,
    records: u64,
}

#[derive(Serialize)]
struct Totals {
    examples: usize,
    by_source: BTreeMap<&'static str, usize>,
    by_language: BTreeMap<String, usize>,
    by_licence: BTreeMap<String, usize>,
    /// Distinct split units per source: titles, repos, template units.
    units_by_source: BTreeMap<&'static str, usize>,
    /// Rows per form (`--forms v2` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    by_form: Option<BTreeMap<&'static str, usize>>,
}

#[derive(Serialize)]
struct LicenceNote {
    licence: Option<String>,
    basis: String,
}

#[derive(Serialize)]
struct Manifest {
    tool: &'static str,
    tool_version: &'static str,
    seed: u64,
    seed_derivation: &'static str,
    /// Absent when `--form-counts` set the forms one by one; `form_counts` says how many.
    #[serde(skip_serializing_if = "Option::is_none")]
    per_source: Option<usize>,
    /// The split the allowlist was computed under. The loader refuses a run at another seed.
    split: allowlist::Split,
    allowlist_sha256: String,
    squad: InputFile,
    pool: PoolMeta,
    template_catalogue_sha256: String,
    examples_sha256: String,
    totals: Totals,
    licences: BTreeMap<&'static str, LicenceNote>,
    /// What the allowlist step excluded, by source and reason (held-out titles, val repos, ...).
    excluded_by_allowlist: BTreeMap<&'static str, BTreeMap<String, u64>>,
    /// Candidates this run read and could not use, by source and reason.
    skipped: BTreeMap<&'static str, BTreeMap<String, u64>>,
    /// `v2` when written with `--forms v2`; absent for v1, whose manifest bytes are pinned.
    #[serde(skip_serializing_if = "Option::is_none")]
    forms: Option<&'static str>,
    /// The corpus the v2 scrambled rows were made from, as checked.
    #[serde(skip_serializing_if = "Option::is_none")]
    corpus: Option<corpus::CorpusMeta>,
    /// The bars this corpus is judged by, recorded before any run reads it.
    #[serde(skip_serializing_if = "Option::is_none")]
    preregistered: Option<serde_json::Value>,
    /// The per-form counts `--form-counts` asked for (absent for the equal-thirds layouts,
    /// whose bytes are pinned). `totals.by_form` is what was written; the two are equal.
    #[serde(skip_serializing_if = "Option::is_none")]
    form_counts: Option<BTreeMap<&'static str, usize>>,
}

/// How many rows each form gets. v1 has no forms; it uses `paragraph`, `lines` and `template`
/// as its three sources' counts and leaves `question` and `tokens` at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mix {
    paragraph: usize,
    question: usize,
    lines: usize,
    tokens: usize,
    template: usize,
}

impl Mix {
    /// v1: every source `n` rows. v2: equal thirds of `n`, prose and scrambled in exact halves.
    fn thirds(n: usize, v2: bool) -> Mix {
        if v2 {
            Mix { paragraph: n / 2, question: n - n / 2, lines: n / 2, tokens: n - n / 2,
                  template: n }
        } else {
            Mix { paragraph: n, question: 0, lines: n, tokens: 0, template: n }
        }
    }

    fn by_form(self) -> BTreeMap<&'static str, usize> {
        BTreeMap::from([
            (FORM_PARAGRAPH, self.paragraph),
            (questions::FORM, self.question),
            (corpus::FORM_LINES, self.lines),
            (corpus::FORM_TOKENS, self.tokens),
            (FORM_TEMPLATE, self.template),
        ])
    }

    /// `paragraph=N,question=N,lines=N,tokens=N,template=N`: all five forms, once each, in any
    /// order, each count in `1..=MAX_PER_SOURCE`.
    fn parse(spec: &str) -> Result<Mix> {
        let forms = [FORM_PARAGRAPH, questions::FORM, corpus::FORM_LINES, corpus::FORM_TOKENS,
                     FORM_TEMPLATE];
        let mut got: BTreeMap<&'static str, usize> = BTreeMap::new();
        for part in spec.split(',') {
            let (name, count) = part
                .split_once('=')
                .with_context(|| format!("--form-counts: {part:?} is not form=count"))?;
            let name = name.trim();
            let Some(form) = forms.iter().copied().find(|f| *f == name) else {
                bail!("--form-counts: {name:?} is not a form; the forms are {forms:?}");
            };
            let count: usize = count.trim().parse().with_context(|| {
                format!("--form-counts: {form}={count:?} is not a count")
            })?;
            if count == 0 || count > MAX_PER_SOURCE {
                bail!("--form-counts: {form} must be at least 1 and at most {MAX_PER_SOURCE}, \
                       got {count}");
            }
            if got.insert(form, count).is_some() {
                bail!("--form-counts names {form} twice");
            }
        }
        let missing: Vec<&str> = forms.iter().copied().filter(|f| !got.contains_key(f)).collect();
        if !missing.is_empty() {
            bail!("--form-counts must name every form; missing {missing:?}");
        }
        Ok(Mix {
            paragraph: got[FORM_PARAGRAPH],
            question: got[questions::FORM],
            lines: got[corpus::FORM_LINES],
            tokens: got[corpus::FORM_TOKENS],
            template: got[FORM_TEMPLATE],
        })
    }
}

fn file_name(path: &Path) -> Result<String> {
    Ok(path
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("{} has no file name", path.display()))?
        .to_string())
}

/// Read `path` and refuse it unless it hashes to what the allowlist recorded.
fn read_pinned(path: &Path, expected_name: &str, expected_sha: &str) -> Result<Vec<u8>> {
    let name = file_name(path)?;
    if name != expected_name {
        bail!("{} is not the {expected_name} the allowlist was computed from", path.display());
    }
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let sha = sha256_hex(&bytes);
    if sha != expected_sha {
        bail!(
            "{} hashes to {sha}, but the allowlist was computed from {expected_sha}: re-run the \
             allowlist step against these bytes",
            path.display()
        );
    }
    Ok(bytes)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))
}

fn row_id(source: &str, unit: &str, diff: &str) -> String {
    format!("noul:{source}:{}", hunk::short_hash(&[source, unit, diff]))
}

/// The paragraph half of v2's prose (v1's paragraphs, drawn the same way).
const FORM_PARAGRAPH: &str = "paragraph";
/// v2's unseen-language rows, which are v1's.
const FORM_TEMPLATE: &str = "template";

/// v1's scrambled rows: pool windows, tokens and lines shuffled, one row per repo per language.
fn scrambled_from_pool(
    rows: &mut Vec<Row<'_>>,
    records: &[pool::PoolRecord],
    allow: &Allowlist,
    n: usize,
    seed: u64,
) -> Result<BTreeMap<String, u64>> {
    let ranges = &allow.invisible_format_ranges;
    let by_language = scramble::candidates(records, &allow.pool.files, seed);
    let mut scramble_skips: BTreeMap<String, u64> = BTreeMap::new();
    for (k, lang) in scramble::LANGUAGES.iter().enumerate() {
        let quota = n / scramble::LANGUAGES.len() + usize::from(k < n % scramble::LANGUAGES.len());
        let mut used_repos: BTreeSet<&str> = BTreeSet::new();
        let mut got = 0usize;
        for record in by_language.get(lang).map(Vec::as_slice).unwrap_or_default() {
            if got == quota {
                break;
            }
            // One row per repo: the share is spread over repos, not drawn from a few.
            if used_repos.contains(record.repo.as_str()) {
                continue;
            }
            match scramble::render(record, seed, ranges) {
                Ok(diff) => {
                    used_repos.insert(record.repo.as_str());
                    let licence = allow.pool.files[&record.id].clone();
                    rows.push(Row {
                        id: row_id(scramble::SOURCE, &record.repo, &diff),
                        class: NOUL_CLASS,
                        noul_source: scramble::SOURCE,
                        noul_form: None,
                        repo: record.repo.clone(),
                        path: record.path.clone(),
                        language: lang.as_str(),
                        diff,
                        pool_id: Some(record.id.clone()),
                        squad_title: None,
                        template: None,
                        licence,
                        seed,
                        tool_version: qd_mutate::TOOL_VERSION,
                    });
                    got += 1;
                }
                Err(skip) => *scramble_skips.entry(skip.as_str().to_string()).or_insert(0) += 1,
            }
        }
        if got < quota {
            bail!("{}: {got} scrambled rows, {quota} were asked for", lang.as_str());
        }
    }
    Ok(scramble_skips)
}

/// v2's scrambled rows: the corpus's own diffs, `n_lines` line-shuffled and `n_tokens`
/// token-shuffled, one row per repo across both forms, an equal quota per language within each
/// form.
fn scrambled_from_corpus(
    rows: &mut Vec<Row<'_>>,
    corpus: &corpus::Corpus,
    records: &[pool::PoolRecord],
    allow: &Allowlist,
    (n_lines, n_tokens): (usize, usize),
    seed: u64,
) -> Result<BTreeMap<String, u64>> {
    // The loader re-checks every row's repo against its pool id; a corpus row that disagrees
    // with the pool is a corpus of unknown provenance, so refuse it here rather than write it.
    let pool_repo: BTreeMap<&str, &str> =
        records.iter().map(|r| (r.id.as_str(), r.repo.as_str())).collect();
    for e in &corpus.examples {
        match pool_repo.get(e.pool_id.as_str()) {
            Some(repo) if *repo == e.repo => {}
            other => bail!(
                "corpus row {} names pool id {} in repo {}, but the pool has {:?}",
                e.id,
                e.pool_id,
                e.repo,
                other
            ),
        }
    }
    let ranges = &allow.invisible_format_ranges;
    let by_language = corpus::candidates(&corpus.examples, seed);
    let mut skips: BTreeMap<String, u64> = BTreeMap::new();
    let mut used_repos: BTreeSet<&str> = BTreeSet::new();
    let langs = scramble::LANGUAGES.len();
    for (form, total) in [(corpus::FORM_LINES, n_lines), (corpus::FORM_TOKENS, n_tokens)] {
        for (k, lang) in scramble::LANGUAGES.iter().enumerate() {
            let quota = total / langs + usize::from(k < total % langs);
            let mut got = 0usize;
            for e in by_language.get(lang).map(Vec::as_slice).unwrap_or_default() {
                if got == quota {
                    break;
                }
                if used_repos.contains(e.repo.as_str()) {
                    continue;
                }
                match corpus::render(e, form, seed, ranges) {
                    Ok(diff) => {
                        used_repos.insert(e.repo.as_str());
                        rows.push(Row {
                            id: row_id(scramble::SOURCE, &e.repo, &diff),
                            class: NOUL_CLASS,
                            noul_source: scramble::SOURCE,
                            noul_form: Some(form),
                            repo: e.repo.clone(),
                            path: e.path.clone(),
                            language: lang.as_str(),
                            diff,
                            pool_id: Some(e.pool_id.clone()),
                            squad_title: None,
                            template: None,
                            licence: allow.pool.files[&e.pool_id].clone(),
                            seed,
                            tool_version: qd_mutate::TOOL_VERSION,
                        });
                        got += 1;
                    }
                    Err(skip) => *skips.entry(format!("{form}:{}", skip.as_str())).or_insert(0) += 1,
                }
            }
            if got < quota {
                bail!("{} {form}: {got} scrambled rows, {quota} were asked for", lang.as_str());
            }
        }
    }
    Ok(skips)
}

fn generate(args: &GenerateArgs) -> Result<()> {
    let v2 = args.forms == Forms::V2;
    let (mix, per_source, form_counts) = match (&args.form_counts, args.per_source) {
        (Some(_), Some(_)) => bail!(
            "--per-source and --form-counts both size the corpus; give one (--form-counts sets \
             every form's count)"
        ),
        (Some(_), None) if !v2 => bail!("--form-counts is a --forms v2 input; v1 has no forms"),
        (Some(spec), None) => {
            let mix = Mix::parse(spec)?;
            (mix, None, Some(mix.by_form()))
        }
        (None, n) => {
            let n = n.unwrap_or(834);
            if n == 0 || n > MAX_PER_SOURCE {
                bail!("--per-source must be in 1..={MAX_PER_SOURCE}, got {n}");
            }
            (Mix::thirds(n, v2), Some(n), None)
        }
    };
    let examples_path = args.out.join("examples.jsonl");
    let manifest_path = args.out.join("manifest.json");
    if examples_path.exists() || manifest_path.exists() {
        bail!(
            "{} already holds a corpus; a corpus is written once, into a new directory",
            args.out.display()
        );
    }
    let allowlist_bytes = fs::read(&args.allowlist)
        .with_context(|| format!("reading {}", args.allowlist.display()))?;
    let allow: Allowlist = serde_json::from_slice(&allowlist_bytes)
        .with_context(|| format!("{} is not an allowlist", args.allowlist.display()))?;
    allow.check().map_err(anyhow::Error::msg)?;
    let catalogue = templates::catalogue_sha256();
    if catalogue != allow.templates.catalogue_sha256 {
        bail!(
            "the template catalogue is {catalogue}, the allowlist split {}: the catalogue \
             changed after `qd-noul-rows units` was run; list the units and make the allowlist \
             again",
            allow.templates.catalogue_sha256
        );
    }
    let ranges = &allow.invisible_format_ranges;
    let seed = args.seed;
    if !v2 && (args.corpus.is_some() || args.preregistered.is_some()) {
        bail!("--corpus and --preregistered are --forms v2 inputs; --forms v1 reads neither");
    }
    let corpus = match (&args.corpus, v2) {
        (Some(dir), true) => Some(
            corpus::read(dir, &allow.pool.files, &allow.pool.sha256)
                .map_err(anyhow::Error::msg)?,
        ),
        (None, true) => bail!(
            "--forms v2 needs --corpus: its scrambled rows are made from the corpus's own diffs"
        ),
        _ => None,
    };
    let preregistered = match &args.preregistered {
        Some(path) => {
            let bytes =
                fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("{} is not JSON", path.display()))?;
            if !value.is_object() {
                bail!("{} must hold a JSON object", path.display());
            }
            Some(value)
        }
        None => None,
    };

    let squad_bytes = read_pinned(&args.squad, &allow.squad.file, &allow.squad.sha256)?;
    let pool_bytes = read_pinned(&args.pool, &allow.pool.file, &allow.pool.sha256)?;

    let total = mix.by_form().values().sum::<usize>();
    let mut rows: Vec<Row> = Vec::with_capacity(total);
    let mut skipped: BTreeMap<&'static str, BTreeMap<String, u64>> = BTreeMap::new();

    // (a) prose: v1 all paragraphs; v2 paragraphs (the same draw) and short questions.
    let (paragraphs, mut prose_skips) =
        prose::read_paragraphs(&squad_bytes[..], &allow.squad.titles, ranges)
            .map_err(anyhow::Error::msg)?;
    if v2 {
        let (groups, read_skips) =
            questions::read(&squad_bytes[..], &allow.squad.titles, ranges)
                .map_err(anyhow::Error::msg)?;
        let (picked, render_skips) =
            questions::select(&groups, mix.question, seed).map_err(anyhow::Error::msg)?;
        for (reason, count) in read_skips.into_iter().chain(render_skips) {
            *prose_skips.entry(reason).or_insert(0) += count;
        }
        for (g, path, diff) in picked {
            let unit = allow.squad.titles[&g.title].clone();
            rows.push(Row {
                id: row_id(prose::SOURCE, &unit, &diff),
                class: NOUL_CLASS,
                noul_source: prose::SOURCE,
                noul_form: Some(questions::FORM),
                repo: unit,
                path,
                language: prose::LANGUAGE,
                diff,
                pool_id: None,
                squad_title: Some(g.title.clone()),
                template: None,
                licence: allow.squad.licence.clone(),
                seed,
                tool_version: qd_mutate::TOOL_VERSION,
            });
        }
    }
    skipped.insert(prose::SOURCE, prose_skips);
    for p in prose::select(&paragraphs, mix.paragraph, seed).map_err(anyhow::Error::msg)? {
        let (path, diff) = prose::render(p, seed);
        let unit = allow.squad.titles[&p.title].clone();
        rows.push(Row {
            id: row_id(prose::SOURCE, &unit, &diff),
            class: NOUL_CLASS,
            noul_source: prose::SOURCE,
            noul_form: v2.then_some(FORM_PARAGRAPH),
            repo: unit,
            path,
            language: prose::LANGUAGE,
            diff,
            pool_id: None,
            squad_title: Some(p.title.clone()),
            template: None,
            licence: allow.squad.licence.clone(),
            seed,
            tool_version: qd_mutate::TOOL_VERSION,
        });
    }

    // (b) scrambled pool files
    let records = pool::read_jsonl(&pool_bytes[..]).context("reading the pool")?;
    if records.len() as u64 != allow.pool.records {
        bail!(
            "the pool holds {} records, the allowlist says {}",
            records.len(),
            allow.pool.records
        );
    }
    let scramble_skips = match &corpus {
        Some(corpus) => scrambled_from_corpus(
            &mut rows, corpus, &records, &allow, (mix.lines, mix.tokens), seed,
        )?,
        None => scrambled_from_pool(&mut rows, &records, &allow, mix.lines, seed)?,
    };
    skipped.insert(scramble::SOURCE, scramble_skips);

    // (c) unseen-language templates
    let allowed_units: BTreeSet<String> = allow.templates.units.iter().cloned().collect();
    let (template_rows, template_skips) =
        templates::rows(&allowed_units, mix.template, seed).map_err(anyhow::Error::msg)?;
    skipped.insert(templates::SOURCE, template_skips);
    for t in template_rows {
        rows.push(Row {
            id: row_id(templates::SOURCE, &t.unit, &t.diff),
            class: NOUL_CLASS,
            noul_source: templates::SOURCE,
            noul_form: v2.then_some(FORM_TEMPLATE),
            repo: t.unit,
            path: t.path,
            language: t.language,
            diff: t.diff,
            pool_id: None,
            squad_title: None,
            template: Some(t.template),
            licence: allow.templates.licence.clone(),
            seed,
            tool_version: qd_mutate::TOOL_VERSION,
        });
    }

    // Distinct ids and distinct contexts: two rows rendering one prompt would be dropped by the
    // mixture's consistency pass, unevenly by source.
    let ids: BTreeSet<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let diffs: BTreeSet<&str> = rows.iter().map(|r| r.diff.as_str()).collect();
    if ids.len() != rows.len() || diffs.len() != rows.len() {
        bail!(
            "{} rows, {} distinct ids, {} distinct diffs",
            rows.len(),
            ids.len(),
            diffs.len()
        );
    }

    let mut body = String::new();
    let mut totals = Totals {
        examples: rows.len(),
        by_source: BTreeMap::new(),
        by_language: BTreeMap::new(),
        by_licence: BTreeMap::new(),
        units_by_source: BTreeMap::new(),
        by_form: None,
    };
    let mut units: BTreeMap<&'static str, BTreeSet<&str>> = BTreeMap::new();
    let mut by_form: BTreeMap<&'static str, usize> = BTreeMap::new();
    for row in &rows {
        body.push_str(&serde_json::to_string(row)?);
        body.push('\n');
        *totals.by_source.entry(row.noul_source).or_insert(0) += 1;
        *totals.by_language.entry(row.language.to_string()).or_insert(0) += 1;
        *totals.by_licence.entry(row.licence.clone()).or_insert(0) += 1;
        units.entry(row.noul_source).or_default().insert(row.repo.as_str());
        if let Some(form) = row.noul_form {
            *by_form.entry(form).or_insert(0) += 1;
        }
    }
    totals.units_by_source = units.into_iter().map(|(k, v)| (k, v.len())).collect();
    if v2 {
        if by_form.values().sum::<usize>() != rows.len() {
            bail!("a v2 row carries no form");
        }
        if let Some(asked) = &form_counts
            && *asked != by_form
        {
            bail!("--form-counts asked for {asked:?}, but {by_form:?} were written");
        }
        totals.by_form = Some(by_form);
    }
    let examples_sha256 = sha256_hex(body.as_bytes());

    let manifest = Manifest {
        tool: "qd-noul-rows",
        tool_version: qd_mutate::TOOL_VERSION,
        seed,
        seed_derivation: "ChaCha20Rng::from_seed(sha256(seed.to_le_bytes() || 0 || source || 0 \
                          || key)); sample order sha256((seed+1).to_le_bytes() || 0 || source \
                          || 0 || key)",
        per_source,
        split: allow.split.clone(),
        allowlist_sha256: sha256_hex(&allowlist_bytes),
        squad: InputFile { file: allow.squad.file.clone(), sha256: allow.squad.sha256.clone() },
        pool: PoolMeta {
            path: allow.pool.file.clone(),
            sha256: allow.pool.sha256.clone(),
            records: allow.pool.records,
        },
        template_catalogue_sha256: catalogue,
        examples_sha256: examples_sha256.clone(),
        totals,
        licences: BTreeMap::from([
            (prose::SOURCE, LicenceNote {
                licence: Some(allow.squad.licence.clone()),
                basis: "rajpurkar/squad_v2's registered licence (qd_data.sources); \
                        share-alike, carried per row"
                    .to_string(),
            }),
            (scramble::SOURCE, LicenceNote {
                licence: None,
                basis: if v2 {
                    "per row: the corpus row's pool_id -> commitpackft pool -> \
                     bigcode/commitpackft download (commit, new_file), as the main defect \
                     corpus is joined"
                } else {
                    "per row: pool_id -> commitpackft pool -> bigcode/commitpackft download \
                     (commit, new_file), as the main defect corpus is joined"
                }
                .to_string(),
            }),
            (templates::SOURCE, LicenceNote {
                licence: Some(allow.templates.licence.clone()),
                basis: allow.templates.licence_basis.clone(),
            }),
        ]),
        excluded_by_allowlist: BTreeMap::from([
            (prose::SOURCE, allow.squad.excluded.clone()),
            (scramble::SOURCE, allow.pool.excluded.clone()),
            (templates::SOURCE, allow.templates.excluded.clone()),
        ]),
        skipped,
        forms: v2.then_some("v2"),
        corpus: corpus.map(|c| c.meta),
        preregistered,
        form_counts,
    };

    fs::create_dir_all(&args.out).with_context(|| format!("creating {}", args.out.display()))?;
    write_atomic(&examples_path, body.as_bytes())?;
    let mut manifest_json = serde_json::to_string_pretty(&manifest)?;
    manifest_json.push('\n');
    write_atomic(&manifest_path, manifest_json.as_bytes())?;
    println!(
        "{} rows ({}) -> {}  examples sha256 {examples_sha256}",
        rows.len(),
        match per_source {
            Some(n) => format!("{n} per source"),
            None => format!("form counts {:?}", mix.by_form()),
        },
        examples_path.display()
    );
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Units => serde_json::to_string_pretty(&serde_json::json!({
            "catalogue_sha256": templates::catalogue_sha256(),
            "units": templates::units(),
        }))
        .map(|s| println!("{s}"))
        .map_err(anyhow::Error::from),
        Command::Generate(args) => generate(args),
    };
    if let Err(e) = result {
        eprintln!("qd-noul-rows: {e:#}");
        std::process::exit(2);
    }
}
