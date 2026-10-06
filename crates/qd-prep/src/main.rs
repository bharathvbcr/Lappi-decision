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
//! - `qd-prep dedupe --input IN --output OUT`: a `QDPDDIN1` request (see `qd_prep::dedupe`) ->
//!   `QDPDDOK1`, the near-duplicate clusters and survivors `qd_data.dedupe.dedupe` finds.
//! - `qd-prep own-repos --code-root ROOT --out FILE`: the human's own repositories, admitted
//!   and split by the v6 holdout rule (see `qd_prep::own_repos`), written before any commit is
//!   read.
//! - `qd-prep natural-bugs --manifest M --cutoff YYYY-MM-DD --cutoff-basis TEXT --v5-files F
//!   --out-dir DIR`: single-statement fix commits from the manifest's held-out repositories
//!   (see `qd_prep::natural_bugs`) -> `DIR/{natural-bugs.jsonl, report.json}`; `DIR` must carry
//!   a held-out path marker.
//! - `qd-prep spancheck --input IN --output OUT`: a `QDPSCIN1` request (see
//!   `qd_prep::spancheck`) -> `QDPSCOK1`, each span sequence's line-start candidates, gold
//!   positions or the refusal, as `qd_train.shards._span_token_positions` finds them before it
//!   decodes.
//! - `qd-prep synth --config C [--target NAME=FILE ...] --out-dir DIR`: decision rows
//!   synthesised by rule for the email sorter, Jarvis or tool selection (see `qd_prep::synth`) ->
//!   `DIR/{examples.jsonl, manifest.json, containment/}`, the pool shape `decisions` writes.
//! - `qd-prep convert --config C --view-record VR --pool NAME [--licence-cache F] --out-dir DIR`:
//!   one downloaded v6 dataset as a decision pool (see `qd_prep::convert`), the same pool shape
//!   `decisions` writes; `swe-rebench-filter` and `injections` write a view and a corpus.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qd_prep::{containment, decisions, dedupe, linwire, lsh, spancheck, wire};
use qd_prep::{natural_bugs, own_repos};

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
    /// Near-duplicate clusters and survivors, as qd_data.dedupe.dedupe's MinHash path finds them.
    Dedupe(Io),
    /// The human's own repositories under a code root, admitted and split by the v6 holdout
    /// rule; the manifest every own-repo family reads before it reads a commit.
    OwnRepos(OwnReposIo),
    /// Single-statement bug-fix commits from an own-repos manifest's held-out repositories:
    /// a held-out evaluation set, never training data.
    NaturalBugs(NaturalBugsIo),
    /// Span sequences' line starts and gold projected onto token positions, as
    /// qd_train.shards._span_token_positions does before its decode check.
    Spancheck(Io),
    /// Decision rows synthesised by rule for callers with no usable real data (the email
    /// sorter, Jarvis, tool selection), split by template, leak-checked and decontaminated
    /// (`qd_prep::synth`).
    Synth(SynthIo),
    /// A downloaded v6 dataset made into a decision pool, licence-filtered and decontaminated
    /// (`qd_prep::convert`), or SWE-rebench's filtered view, or the injection-text corpus.
    Convert(ConvertIo),
}

/// `qd-prep synth`'s inputs.
#[derive(clap::Args, Debug)]
struct SynthIo {
    /// The generator config (`data/synth/*.json`, schema qd-synth-config/v1).
    #[arg(long)]
    config: PathBuf,
    /// A decontamination target set, `NAME=FILE` of `{"id", "text"}` JSONL; repeat per set.
    #[arg(long = "target", value_parser = parse_target)]
    targets: Vec<(String, PathBuf)>,
    /// The directory to create; refused if it, or DIR.partial, exists.
    #[arg(long)]
    out_dir: PathBuf,
    /// Worker threads for the containment scan and the leak probe; default every core, at
    /// most 256.
    #[arg(long)]
    threads: Option<usize>,
    /// Write the config's held-out templates (`heldout_templates_per_class`) to --out-dir as
    /// an evaluation set and a target file, instead of the pool. --out-dir must carry a
    /// held-out path marker; no --target is read.
    #[arg(long)]
    emit_heldout: bool,
}

/// `qd-prep convert`'s inputs.
#[derive(clap::Args, Debug)]
struct ConvertIo {
    /// The conversion config (`data/convert/*.json`, schema qd-convert-config/v1).
    #[arg(long)]
    config: PathBuf,
    /// The view record `view_v6.py` wrote beside the data (schema qd-view-record/v1).
    #[arg(long)]
    view_record: PathBuf,
    /// One of: tools, mnli, scirepeval, csn, swe-rebench-filter, injections.
    #[arg(long)]
    pool: String,
    /// CodeSearchNet's repository licence cache (`csn_licences.py`); `--pool csn` only.
    #[arg(long)]
    licence_cache: Option<PathBuf>,
    /// The directory to create; refused if it, or DIR.partial, exists.
    #[arg(long)]
    out_dir: PathBuf,
    /// Worker threads for the containment scan; default every core, at most 256.
    #[arg(long)]
    threads: Option<usize>,
}

/// `--threads`, bounded: an explicit 0 is refused, the default is every core, at most 256.
fn bounded_threads(threads: Option<usize>) -> Result<usize, String> {
    Ok(match threads {
        Some(0) => return Err("--threads 0 would do nothing".to_string()),
        Some(n) => n,
        None => std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    }
    .min(MAX_THREADS))
}

/// `qd-prep synth`: the config in, the pool directory out.
fn run_synth(io: &SynthIo) -> Result<String, String> {
    if io.emit_heldout {
        if !io.targets.is_empty() {
            return Err("--emit-heldout writes a target set; it reads no --target".into());
        }
        return qd_prep::synth::emit_heldout(&io.config, &io.out_dir);
    }
    let threads = bounded_threads(io.threads)?;
    let inputs = qd_prep::synth::Inputs {
        config: io.config.clone(),
        targets: io.targets.clone(),
    };
    qd_prep::synth::run(&inputs, &io.out_dir, threads)
}

/// `qd-prep convert`: the config, the view record and a pool name in, the directory out.
fn run_convert(io: &ConvertIo) -> Result<String, String> {
    let threads = bounded_threads(io.threads)?;
    let inputs = qd_prep::convert::Inputs {
        config: io.config.clone(),
        view_record: io.view_record.clone(),
        pool: io.pool.clone(),
        licence_cache: io.licence_cache.clone(),
    };
    qd_prep::convert::run(&inputs, &io.out_dir, threads)
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
    let read =
        |p: &PathBuf| qd_prep::files::read_bounded(p, qd_prep::files::MAX_RECORD_BYTES, "input");
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
    qd_prep::files::write_new_file(&io.output, &bytes)?;
    Ok(format!("{} cap(s) moved -> {}", moved.len(), io.output.display()))
}

/// `qd-prep own-repos`' inputs.
#[derive(clap::Args, Debug)]
struct OwnReposIo {
    /// The directory the human's repositories live under (absolute).
    #[arg(long)]
    code_root: PathBuf,
    /// Where to write the manifest; refused if it exists.
    #[arg(long)]
    out: PathBuf,
}

/// `qd-prep natural-bugs`' inputs. Every one is required; none has a default.
#[derive(clap::Args, Debug)]
struct NaturalBugsIo {
    /// A `qd-prep own-repos` manifest.
    #[arg(long)]
    manifest: PathBuf,
    /// YYYY-MM-DD: commits authored before 00:00:00Z of this date are not emitted. No default:
    /// the base model's data cutoff is unknown, and a default would be a guess.
    #[arg(long)]
    cutoff: String,
    /// Where the --cutoff date comes from, copied into the report (for example, "yield probe:
    /// the base model's release month, not its cutoff").
    #[arg(long)]
    cutoff_basis: String,
    /// own-prose-v1's files.jsonl: any commit touching a file it lists is excluded.
    #[arg(long)]
    v5_files: PathBuf,
    /// The directory to create; it must carry a held-out path marker, and it (or DIR.partial)
    /// must not exist.
    #[arg(long)]
    out_dir: PathBuf,
}

fn run_natural_bugs(io: &NaturalBugsIo) -> Result<String, String> {
    natural_bugs::run(&natural_bugs::Args {
        manifest: &io.manifest,
        cutoff: &io.cutoff,
        cutoff_basis: &io.cutoff_basis,
        v5_files: &io.v5_files,
        out_dir: &io.out_dir,
    })
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
    let threads = bounded_threads(io.threads)?;
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
    let buf = qd_prep::files::read_bounded(input, max_input_bytes, "--input")?;
    Ok((buf, bounded_threads(threads)?))
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
    // Written beside the destination and renamed over it, so a reader never sees a prefix.
    qd_prep::files::write_new_file(output, &bytes)?;
    Ok(format!("{line} -> {}", output.display()))
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
        Command::Dedupe(io) => run(io, dedupe::MAX_INPUT_BYTES, dedupe::run_dedupe),
        Command::OwnRepos(io) => own_repos::run(&io.code_root, &io.out),
        Command::NaturalBugs(io) => run_natural_bugs(io),
        Command::Spancheck(io) => run(io, spancheck::MAX_INPUT_BYTES, spancheck::run_spancheck),
        Command::Synth(io) => run_synth(io),
        Command::Convert(io) => run_convert(io),
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
