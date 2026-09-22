//! `qd-mutate` — the command line over [`qd_mutate`].
//!
//! Three subcommands, and each one answers a question the lane actually asks:
//!
//! * `generate` — the pipeline. Reads a JSONL pool, writes a JSONL of labelled examples and a
//!   manifest describing exactly how they came about.
//! * `formatters` — what this machine can actually verify. `cosmetic` labels are only as good as
//!   the formatter that proved them, and on a machine with no `prettier` the TypeScript cosmetic
//!   set is smaller. That is a fact to print before a run, not to discover in a manifest after one.
//! * `one` — mutate a single file and print the result. For looking at what the generator does
//!   without building a pool first.
//!
//! Exit codes are meaningful: `0` for a run that completed, `2` for a usage or I/O failure. A run
//! that completed with zero examples is still `0` — it did what it was asked and the manifest says
//! what it found. A run that could not read its pool is not.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use qd_mutate::generate::{Generator, Options};
use qd_mutate::lang::LangId;
use qd_mutate::manifest::{sha256_hex, PoolReport};
use qd_mutate::pool::{self, PoolRecord};

#[derive(Parser, Debug)]
#[command(
    name = "qd-mutate",
    version,
    about = "Mutation engine: labelled diffs with exact line spans",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Mutate a JSONL pool into labelled examples plus a manifest.
    Generate(GenerateArgs),
    /// Report which language formatters resolve on this machine.
    Formatters,
    /// Mutate one source file and print the examples as JSONL.
    One(OneArgs),
}

#[derive(clap::Args, Debug)]
struct GenerateArgs {
    /// JSONL pool. One record per line: id, repo, path, source, and optionally hunks.
    #[arg(long)]
    pool: PathBuf,
    /// Where to write the examples, as JSONL.
    #[arg(long)]
    out: PathBuf,
    /// Where to write the manifest, as JSON.
    #[arg(long)]
    manifest: PathBuf,
    /// The seed every choice in the run comes from. Recorded in the manifest; the same seed over
    /// the same pool produces byte-identical output.
    #[arg(long)]
    seed: u64,
    /// Records in a thousand emitted `clean` rather than mutated.
    #[arg(long, default_value_t = 200, value_parser = clap::value_parser!(u32).range(0..=1000))]
    clean_permille: u32,
    /// Restrict to these languages. Repeatable; the default is all five.
    #[arg(long, value_parser = parse_language)]
    language: Vec<LangId>,
    /// Stop after this many examples.
    #[arg(long)]
    limit: Option<usize>,
    /// Examples one file may contribute.
    #[arg(long, default_value_t = qd_mutate::generate::MAX_EXAMPLES_PER_FILE)]
    max_per_file: usize,
    /// Print the coverage table to stderr when the run finishes.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    report: bool,
}

#[derive(clap::Args, Debug)]
struct OneArgs {
    /// The source file to mutate.
    #[arg(long)]
    path: PathBuf,
    /// Override the language the extension implies.
    #[arg(long, value_parser = parse_language)]
    language: Option<LangId>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Examples to emit from this file.
    #[arg(long, default_value_t = 8)]
    limit: usize,
}

fn parse_language(raw: &str) -> Result<LangId, String> {
    LangId::parse(raw).ok_or_else(|| {
        let known: Vec<&str> = LangId::ALL.iter().map(|l| l.as_str()).collect();
        format!("unknown language {raw:?}; known: {}", known.join(", "))
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Generate(args) => generate(args),
        Command::Formatters => formatters(),
        Command::One(args) => one(args),
    }
}

fn generate(args: GenerateArgs) -> Result<()> {
    let bytes = std::fs::read(&args.pool)
        .with_context(|| format!("reading the pool at {}", args.pool.display()))?;
    let records = pool::read_jsonl(bytes.as_slice())
        .with_context(|| format!("parsing the pool at {}", args.pool.display()))?;

    let report = PoolReport {
        path: args.pool.display().to_string(),
        sha256: sha256_hex(&bytes),
        records: records.len() as u64,
        by_language: pool::language_census(&records),
    };

    let generator = Generator::new(Options {
        seed: args.seed,
        clean_permille: args.clean_permille,
        languages: args.language,
        limit: args.limit,
        max_examples_per_file: args.max_per_file,
    });
    let mut run = generator.run(&records, report);

    let (jsonl, digest) = run.to_jsonl().context("serialising the examples")?;
    run.manifest.examples_sha256 = digest.clone();
    write_atomically(&args.out, jsonl.as_bytes())
        .with_context(|| format!("writing {}", args.out.display()))?;

    let manifest_json = serde_json::to_string_pretty(&run.manifest)
        .context("serialising the manifest")?;
    write_atomically(&args.manifest, manifest_json.as_bytes())
        .with_context(|| format!("writing {}", args.manifest.display()))?;

    if args.report {
        eprint!("{}", run.manifest.coverage_table());
    }
    // The two numbers a caller most needs, on stdout, where a script can read them.
    println!("examples {} sha256 {}", run.examples.len(), digest);
    // Not a failure — a run that found nothing still ran — but it must not pass unremarked.
    if run.examples.is_empty() {
        eprintln!(
            "warning: no examples were emitted; see the refusal histogram in {}",
            args.manifest.display()
        );
    }
    if run.manifest.totals.span_disagreements > 0 {
        eprintln!(
            "warning: {} example(s) dropped on span disagreement — the mutator's bookkeeping and \
             the textual differ have drifted apart",
            run.manifest.totals.span_disagreements
        );
    }
    Ok(())
}

fn formatters() -> Result<()> {
    for (id, availability) in qd_mutate::fmt::probe_all() {
        let language = qd_mutate::lang::for_id(id);
        let declared = language
            .formatter()
            .map(|f| {
                let via = if f.via_xcrun { " (via xcrun)" } else { "" };
                format!("{}{via}", f.program)
            })
            .unwrap_or_else(|| "none declared".to_string());
        let resolved = match &availability {
            qd_mutate::fmt::Availability::Found { path, .. } => path.clone(),
            qd_mutate::fmt::Availability::NotFound { detail, .. } => format!("NOT FOUND: {detail}"),
            qd_mutate::fmt::Availability::Unusable { path, detail, .. } => {
                format!("UNUSABLE at {path}: {detail}")
            }
            qd_mutate::fmt::Availability::NotDeclared => "-".to_string(),
        };
        let mut restricted: Vec<&str> = if availability.is_usable() {
            Vec::new()
        } else {
            qd_mutate::ops::OpId::ALL
                .into_iter()
                .filter(|o| o.requires_formatter())
                .map(|o| o.as_str())
                .collect()
        };
        // The language axis belongs here too. This command is the one a human runs to ask "what
        // can this machine verify", and answering only about formatters would let a reader take
        // `cosmetic.reorder_imports`'s absence from the list as "available" when it is refused for
        // the language on every machine.
        let language_restriction = language.import_order_is_semantic();
        if language_restriction.is_some() {
            restricted.push(qd_mutate::ops::OpId::CosmeticReorderImports.as_str());
        }
        println!(
            "{:<11} {:<24} {:<10} {}",
            id.as_str(),
            declared,
            availability.key(),
            resolved
        );
        if !restricted.is_empty() {
            println!("{:<11} restricted: {}", "", restricted.join(", "));
        }
        if let Some(why) = language_restriction {
            println!("{:<11}   cosmetic.reorder_imports: {why}", "");
        }
    }
    Ok(())
}

fn one(args: OneArgs) -> Result<()> {
    let source = std::fs::read_to_string(&args.path)
        .with_context(|| format!("reading {}", args.path.display()))?;
    let path = args.path.display().to_string();
    let language = match args.language.or_else(|| pool::language_from_path(&path)) {
        Some(l) => l,
        None => bail!(
            "cannot tell what language {path} is; pass --language (one of {})",
            LangId::ALL
                .iter()
                .map(|l| l.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };

    let record = PoolRecord {
        id: path.clone(),
        repo: "local".to_string(),
        path: path.clone(),
        language: Some(language),
        source,
        hunks: None,
        // A single file handed to `inspect` has no prior version, exactly as it has no hunks.
        prior_source: None,
    };
    let generator = Generator::new(Options {
        seed: args.seed,
        // Never clean: the point of this subcommand is to look at a mutation.
        clean_permille: 0,
        languages: vec![language],
        limit: Some(args.limit),
        max_examples_per_file: args.limit,
    });
    let run = generator.run(
        std::slice::from_ref(&record),
        PoolReport {
            path,
            sha256: String::new(),
            records: 1,
            by_language: std::collections::BTreeMap::new(),
        },
    );
    let (jsonl, _) = run.to_jsonl().context("serialising the examples")?;
    print!("{jsonl}");
    eprint!("{}", run.manifest.coverage_table());
    Ok(())
}

/// Write via a sibling temporary and rename.
///
/// A half-written examples file that a later stage reads as complete is a data snapshot whose hash
/// nobody can reproduce. The rename is atomic within a filesystem, and the temporary is a sibling
/// so it is on the same one.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".qd-mutate.partial");
    let temporary = PathBuf::from(temporary);
    {
        let mut file = std::fs::File::create(&temporary)
            .with_context(|| format!("creating {}", temporary.display()))?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path)
        .with_context(|| format!("renaming {} into place", temporary.display()))?;
    Ok(())
}
