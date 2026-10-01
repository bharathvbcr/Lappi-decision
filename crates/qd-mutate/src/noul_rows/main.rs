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
//! Exit codes: `0` on success, `2` on any refusal, with the reason on stderr.

mod allowlist;
mod hunk;
mod prose;
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
    /// Rows per source; the corpus holds three times this.
    #[arg(long, default_value_t = 834)]
    per_source: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Serialize)]
struct Row<'a> {
    id: String,
    class: &'static str,
    noul_source: &'static str,
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
    per_source: usize,
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

fn generate(args: &GenerateArgs) -> Result<()> {
    if args.per_source == 0 || args.per_source > MAX_PER_SOURCE {
        bail!("--per-source must be in 1..={MAX_PER_SOURCE}, got {}", args.per_source);
    }
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
    let n = args.per_source;

    let squad_bytes = read_pinned(&args.squad, &allow.squad.file, &allow.squad.sha256)?;
    let pool_bytes = read_pinned(&args.pool, &allow.pool.file, &allow.pool.sha256)?;

    let mut rows: Vec<Row> = Vec::with_capacity(3 * n);
    let mut skipped: BTreeMap<&'static str, BTreeMap<String, u64>> = BTreeMap::new();

    // (a) prose
    let (paragraphs, prose_skips) =
        prose::read_paragraphs(&squad_bytes[..], &allow.squad.titles, ranges)
            .map_err(anyhow::Error::msg)?;
    skipped.insert(prose::SOURCE, prose_skips);
    for p in prose::select(&paragraphs, n, seed).map_err(anyhow::Error::msg)? {
        let (path, diff) = prose::render(p, seed);
        let unit = allow.squad.titles[&p.title].clone();
        rows.push(Row {
            id: row_id(prose::SOURCE, &unit, &diff),
            class: NOUL_CLASS,
            noul_source: prose::SOURCE,
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
    let by_language = scramble::candidates(&records, &allow.pool.files, seed);
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
    skipped.insert(scramble::SOURCE, scramble_skips);

    // (c) unseen-language templates
    let allowed_units: BTreeSet<String> = allow.templates.units.iter().cloned().collect();
    let (template_rows, template_skips) =
        templates::rows(&allowed_units, n, seed).map_err(anyhow::Error::msg)?;
    skipped.insert(templates::SOURCE, template_skips);
    for t in template_rows {
        rows.push(Row {
            id: row_id(templates::SOURCE, &t.unit, &t.diff),
            class: NOUL_CLASS,
            noul_source: templates::SOURCE,
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
    };
    let mut units: BTreeMap<&'static str, BTreeSet<&str>> = BTreeMap::new();
    for row in &rows {
        body.push_str(&serde_json::to_string(row)?);
        body.push('\n');
        *totals.by_source.entry(row.noul_source).or_insert(0) += 1;
        *totals.by_language.entry(row.language.to_string()).or_insert(0) += 1;
        *totals.by_licence.entry(row.licence.clone()).or_insert(0) += 1;
        units.entry(row.noul_source).or_default().insert(row.repo.as_str());
    }
    totals.units_by_source = units.into_iter().map(|(k, v)| (k, v.len())).collect();
    let examples_sha256 = sha256_hex(body.as_bytes());

    let manifest = Manifest {
        tool: "qd-noul-rows",
        tool_version: qd_mutate::TOOL_VERSION,
        seed,
        seed_derivation: "ChaCha20Rng::from_seed(sha256(seed.to_le_bytes() || 0 || source || 0 \
                          || key)); sample order sha256((seed+1).to_le_bytes() || 0 || source \
                          || 0 || key)",
        per_source: n,
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
                basis: "per row: pool_id -> commitpackft pool -> bigcode/commitpackft download \
                        (commit, new_file), as the main defect corpus is joined"
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
    };

    fs::create_dir_all(&args.out).with_context(|| format!("creating {}", args.out.display()))?;
    write_atomic(&examples_path, body.as_bytes())?;
    let mut manifest_json = serde_json::to_string_pretty(&manifest)?;
    manifest_json.push('\n');
    write_atomic(&manifest_path, manifest_json.as_bytes())?;
    println!(
        "{} rows ({} per source) -> {}  examples sha256 {examples_sha256}",
        rows.len(),
        n,
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
