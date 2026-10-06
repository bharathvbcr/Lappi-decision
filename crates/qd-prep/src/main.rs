//! `qd-prep` -- data-preparation loops the Python pipeline hands to Rust.
//!
//! Every subcommand reads one request file and writes one reply file, atomically: the reply
//! appears whole or not at all. Any refusal exits non-zero with the reason on stderr and writes
//! nothing.
//!
//! - `qd-prep minhash --input IN --output OUT`: a `QDPMHIN1` request (see `qd_prep::wire`) ->
//!   `QDPMHOK1` signatures.
//! - `qd-prep ngrams --input IN --output OUT`: a `QDPNGIN1` request (see `qd_prep::linwire`) ->
//!   the `QDPNGOK1` hashed n-gram rows, as `CharNGramHasher.transform` builds them.
//! - `qd-prep linfit --input IN --output OUT`: a `QDPLFIN1` request -> the `QDPLFOK1` fit, as
//!   `LinearBaseline.fit` makes it, with the evaluation rows' logits.
//! - `qd-prep lsh --input IN --output OUT`: a `QDPLSIN1` request (see `qd_prep::lsh`) -> the
//!   `QDPLSOK1` banded-LSH candidate pairs, as `qd_data.minhash.candidate_pairs` finds them.
//! - `qd-prep containment --input IN --out-dir DIR`: a `QDPCTIN1` request (see
//!   `qd_prep::containment`) -> `DIR/{pairs.tsv, exclusions.txt, attestation.json}`, the
//!   complete word n-gram containment pair list `qd_train.replay.decontaminate` defines.
//! - `qd-prep decisions --config C --fetch-record R --decider-dir D --target NAME=FILE ...
//!   --out-dir DIR [--survey]`: the v5 general-decision pool (see `qd_prep::decisions`) ->
//!   `DIR/{examples.jsonl, manifest.json, containment/}`, or `texts.jsonl` with `--survey`.
//! - `qd-prep spancheck --input IN --output OUT`: a `QDPSCIN1` request (see
//!   `qd_prep::spancheck`) -> `QDPSCOK1`, each span sequence's line-start candidates, gold
//!   positions or the refusal, as `qd_train.shards._span_token_positions` finds them before it
//!   decodes.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qd_prep::{containment, decisions, linwire, lsh, spancheck, wire};

/// Threads are bounded whatever the host reports.
const MAX_THREADS: usize = 256;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Data-preparation loops ported out of the Python pipeline"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

/// The arguments every subcommand takes.
#[derive(clap::Args, Debug)]
struct Io {
    /// The request file.
    #[arg(long)]
    input: PathBuf,
    /// Where to write the reply; refused if it exists.
    #[arg(long)]
    output: PathBuf,
    /// Worker threads; default every core the host reports, at most 256. The result does not
    /// depend on it.
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// MinHash signatures of shingle sets, as qd_data.minhash.MinHasher.signature computes them.
    Minhash(Io),
    /// Hashed char n-gram rows, as qd_train.baseline.CharNGramHasher.transform builds them.
    Ngrams(Io),
    /// The FT linear control's fit, as qd_train.baseline.LinearBaseline.fit makes it.
    Linfit(Io),
    /// Banded-LSH candidate pairs, as qd_data.minhash.candidate_pairs finds them.
    Lsh(Io),
    /// Word n-gram containment with the complete pair list, as qd_train.replay.decontaminate
    /// defines it; writes pairs.tsv, exclusions.txt and attestation.json into --out-dir.
    Containment(DirIo),
    /// The v5 general-decision pool: six typed-decision sources, decontaminated against the
    /// external targets, split by group and capped by one table.
    Decisions(DecisionsIo),
    /// The decision pool's cap refresh between two surveys: every all-admitted stratum's cap
    /// becomes its new train availability, the rest are kept (`decisions::refresh_train_caps`).
    DecisionCaps(DecisionCapsIo),
    /// Span sequences' line starts and gold projected onto token positions, as
    /// qd_train.shards._span_token_positions does before its decode check.
    Spancheck(Io),
}

/// `qd-prep decision-caps`' inputs.
#[derive(clap::Args, Debug)]
struct DecisionCapsIo {
    /// The allocation config whose `train_caps` are refreshed (read, never written).
    #[arg(long)]
    config: PathBuf,
    /// The `--survey` manifest the current caps were set from.
    #[arg(long)]
    before: PathBuf,
    /// The `--survey` manifest of the new inputs.
    #[arg(long)]
    after: PathBuf,
    /// Where to write `{"train_caps", "moved", inputs' sha256}`; refused if it exists.
    #[arg(long)]
    output: PathBuf,
}

/// `qd-prep decision-caps`: the config and two surveys in, the refreshed caps out.
fn run_decision_caps(io: &DecisionCapsIo) -> Result<String, String> {
    if io.output.exists() {
        return Err(format!("{} exists; refusing to overwrite it", io.output.display()));
    }
    let read = |p: &PathBuf| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
    let (config, before, after) = (read(&io.config)?, read(&io.before)?, read(&io.after)?);
    let cfg = decisions::Config::parse(&config)?;
    let (caps, moved) = decisions::refresh_train_caps(
        &cfg.train_caps,
        &decisions::survey_train_availability(&before)?,
        &decisions::survey_train_availability(&after)?,
    )?;
    let sha = qd_prep::sha256::sha256_hex;
    let out = serde_json::json!({
        "schema": "qd-decision-caps/v1",
        "config_sha256": sha(&config), "before_sha256": sha(&before), "after_sha256": sha(&after),
        "train_caps": caps,
        "moved": moved.iter().map(|(s, a, b)| serde_json::json!({"stratum": s, "from": a, "to": b}))
            .collect::<Vec<_>>(),
    });
    let mut bytes = serde_json::to_vec_pretty(&out).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    std::fs::write(&io.output, bytes).map_err(|e| format!("{}: {e}", io.output.display()))?;
    Ok(format!("{} cap(s) moved -> {}", moved.len(), io.output.display()))
}

/// `qd-prep decisions`' inputs: the policy (config) and the pinned data it names.
#[derive(clap::Args, Debug)]
struct DecisionsIo {
    /// The allocation config (`data/decisions/*.json`, schema qd-decisions-config/v1).
    #[arg(long)]
    config: PathBuf,
    /// The fetch record naming every downloaded source file and its sha256.
    #[arg(long)]
    fetch_record: PathBuf,
    /// decider's `teacher_data` directory.
    #[arg(long)]
    decider_dir: PathBuf,
    /// A decontamination target set, `NAME=FILE` of `{"id", "text"}` JSONL; repeat per set.
    #[arg(long = "target", value_parser = parse_target)]
    targets: Vec<(String, PathBuf)>,
    /// The directory to create; refused if it, or DIR.partial, exists.
    #[arg(long)]
    out_dir: PathBuf,
    /// Write every surviving candidate's text for token counting instead of the pool.
    #[arg(long)]
    survey: bool,
    /// Worker threads for the containment scan; default every core, at most 256.
    #[arg(long)]
    threads: Option<usize>,
}

fn parse_target(s: &str) -> Result<(String, PathBuf), String> {
    match s.split_once('=') {
        Some((name, path)) if !name.is_empty() && !path.is_empty() => {
            Ok((name.to_owned(), PathBuf::from(path)))
        }
        _ => Err(format!("{s:?} is not NAME=FILE")),
    }
}

/// `qd-prep decisions`: the sources in, the pool directory out.
fn run_decisions(io: &DecisionsIo) -> Result<String, String> {
    if io.out_dir.exists() {
        return Err(format!("{} exists; refusing to overwrite it", io.out_dir.display()));
    }
    let threads = match io.threads {
        Some(0) => return Err("--threads 0 would do nothing".to_string()),
        Some(n) => n,
        None => std::thread::available_parallelism().map(usize::from).unwrap_or(1),
    }
    .min(MAX_THREADS);
    let inputs = decisions::Inputs {
        config: io.config.clone(),
        fetch_record: io.fetch_record.clone(),
        decider_dir: io.decider_dir.clone(),
        targets: io.targets.clone(),
        survey: io.survey,
    };
    let built = decisions::run(&inputs, threads)?;
    decisions::write_out(&io.out_dir, &built)?;
    Ok(format!("{} on {threads} thread(s) -> {}", built.summary, io.out_dir.display()))
}

/// The arguments of a subcommand whose answer is a directory of files.
#[derive(clap::Args, Debug)]
struct DirIo {
    /// The request file.
    #[arg(long)]
    input: PathBuf,
    /// The directory to create; refused if it, or DIR.partial, exists.
    #[arg(long)]
    out_dir: PathBuf,
    /// Worker threads; default every core the host reports, at most 256. The result does not
    /// depend on it.
    #[arg(long)]
    threads: Option<usize>,
}

/// `qd-prep containment`: the request in, the directory out.
fn run_containment(io: &DirIo) -> Result<String, String> {
    if io.out_dir.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            io.out_dir.display()
        ));
    }
    let (buf, threads) = read_input(&io.input, io.threads, containment::MAX_INPUT_BYTES)?;
    let written = containment::run(&buf, threads)?;
    containment::write_dir(&io.out_dir, &written)?;
    Ok(format!(
        "{} on {threads} thread(s) -> {}",
        written.summary,
        io.out_dir.display()
    ))
}

/// The request's bytes, refused past `max_input_bytes`, and the thread count to use.
fn read_input(
    input: &Path,
    threads: Option<usize>,
    max_input_bytes: u64,
) -> Result<(Vec<u8>, usize), String> {
    let size = std::fs::metadata(input)
        .map_err(|e| format!("{}: {e}", input.display()))?
        .len();
    if size > max_input_bytes {
        return Err(format!(
            "{}: {size} bytes; the bound is {max_input_bytes}",
            input.display(),
        ));
    }
    let buf = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let threads = match threads {
        Some(0) => return Err("--threads 0 would do nothing".to_string()),
        Some(n) => n,
        None => std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    }
    .min(MAX_THREADS);
    Ok((buf, threads))
}

/// Read `io.input`, hand it to `work` with the thread count, and write what it returns to
/// `io.output` atomically. `work` returns the reply's bytes and a summary line.
fn run(
    io: &Io,
    max_input_bytes: u64,
    work: impl FnOnce(&[u8], usize) -> Result<(Vec<u8>, String), String>,
) -> Result<String, String> {
    let (input, output) = (&io.input, &io.output);
    if output.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            output.display()
        ));
    }
    let (buf, threads) = read_input(input, io.threads, max_input_bytes)?;
    let (bytes, line) = work(&buf, threads)?;
    write_atomically(output, &bytes)?;
    Ok(format!("{line} -> {}", output.display()))
}

/// Written beside the destination and renamed over it, so a reader never sees a prefix.
fn write_atomically(output: &Path, bytes: &[u8]) -> Result<(), String> {
    let partial = output.with_extension("partial");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    file.sync_all()
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, output).map_err(|e| format!("{}: {e}", output.display()))
}

fn minhash(buf: &[u8], threads: usize) -> Result<(Vec<u8>, String), String> {
    let request = wire::parse(buf)?;
    let signatures = request.sign(threads)?;
    let bytes = wire::encode_output(request.family.num_perm(), &signatures)?;
    Ok((
        bytes,
        format!(
            "qd-prep minhash: {} documents x {} permutations on {threads} thread(s)",
            request.docs.len(),
            request.family.num_perm(),
        ),
    ))
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match &args.command {
        // Each request format has its own bound: a MinHash request carries shingle sets, a
        // linear-control request the documents themselves (see linwire::MAX_INPUT_BYTES).
        Command::Minhash(io) => run(io, wire::MAX_INPUT_BYTES, minhash),
        Command::Ngrams(io) => run(io, linwire::MAX_INPUT_BYTES, linwire::run_ngrams),
        Command::Linfit(io) => run(io, linwire::MAX_INPUT_BYTES, linwire::run_linfit),
        Command::Lsh(io) => run(io, lsh::MAX_INPUT_BYTES, lsh::run_lsh),
        Command::Containment(io) => run_containment(io),
        Command::Decisions(io) => run_decisions(io),
        Command::DecisionCaps(io) => run_decision_caps(io),
        Command::Spancheck(io) => run(io, spancheck::MAX_INPUT_BYTES, spancheck::run_spancheck),
    };
    match result {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("qd-prep: {e}");
            ExitCode::FAILURE
        }
    }
}
