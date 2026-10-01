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

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qd_prep::{linwire, wire};

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
    let threads = match io.threads {
        Some(0) => return Err("--threads 0 would do nothing".to_string()),
        Some(n) => n,
        None => std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    }
    .min(MAX_THREADS);
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
