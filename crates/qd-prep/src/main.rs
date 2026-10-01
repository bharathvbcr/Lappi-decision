//! `qd-prep` -- data-preparation loops the Python pipeline hands to Rust.
//!
//! `qd-prep minhash --input IN --output OUT` reads a `QDPMHIN1` file (see `qd_prep::wire`),
//! signs every document and writes a `QDPMHOK1` file, atomically: the output appears whole or
//! not at all. Any refusal exits non-zero with the reason on stderr and writes nothing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qd_prep::wire;

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

#[derive(Subcommand, Debug)]
enum Command {
    /// MinHash signatures of shingle sets, as qd_data.minhash.MinHasher.signature computes them.
    Minhash {
        /// A QDPMHIN1 request.
        #[arg(long)]
        input: PathBuf,
        /// Where to write the QDPMHOK1 reply; refused if it exists.
        #[arg(long)]
        output: PathBuf,
        /// Signing threads; default every core the host reports, at most 256.
        #[arg(long)]
        threads: Option<usize>,
    },
}

fn minhash(input: &Path, output: &Path, threads: Option<usize>) -> Result<String, String> {
    if output.exists() {
        return Err(format!(
            "{} exists; refusing to overwrite it",
            output.display()
        ));
    }
    let size = std::fs::metadata(input)
        .map_err(|e| format!("{}: {e}", input.display()))?
        .len();
    if size > wire::MAX_INPUT_BYTES {
        return Err(format!(
            "{}: {size} bytes; the bound is {}",
            input.display(),
            wire::MAX_INPUT_BYTES
        ));
    }
    let buf = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let request = wire::parse(&buf)?;
    let threads = match threads {
        Some(0) => return Err("--threads 0 would sign nothing".to_string()),
        Some(n) => n,
        None => std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    }
    .min(MAX_THREADS);
    let signatures = request.sign(threads)?;
    let bytes = wire::encode_output(request.family.num_perm(), &signatures)?;
    // Written beside the destination and renamed over it, so a reader never sees a prefix.
    let partial = output.with_extension("partial");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    file.write_all(&bytes)
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    file.sync_all()
        .map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, output).map_err(|e| format!("{}: {e}", output.display()))?;
    Ok(format!(
        "qd-prep minhash: {} documents x {} permutations on {threads} thread(s) -> {}",
        request.docs.len(),
        request.family.num_perm(),
        output.display()
    ))
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = match &args.command {
        Command::Minhash {
            input,
            output,
            threads,
        } => minhash(input, output, *threads),
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
