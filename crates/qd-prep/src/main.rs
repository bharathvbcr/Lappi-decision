//! `qd-prep`: the native half of `tools/real_tokenizer_pipeline.py`'s MinHash adapter.
//!
//! ```text
//! qd-prep version
//! qd-prep minhash --input REQ --output SIGS [--threads N]
//! ```
//!
//! `version` prints one JSON object the adapter checks before it sends a byte; `minhash`
//! reads a request in the layout `qd_prep::wire` documents and writes every set's signature.
//! Any failure exits non-zero with the reason on stderr and leaves no output file.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use qd_prep::minhash::{MAX_THREADS, MinHasher, sign_all};
use qd_prep::wire::{
    MINHASH_PROTOCOL, encode_output, parse_request, read_request_file, write_atomically,
};

#[derive(Parser)]
#[command(name = "qd-prep", about = "Native inner loops for the shard pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the binary's name, version and wire protocols as one JSON object.
    Version,
    /// Sign every shingle set in a request, exactly as qd_data.minhash.MinHasher does.
    Minhash {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Worker threads. The output does not depend on it.
        #[arg(long)]
        threads: Option<usize>,
    },
}

fn main() -> std::process::ExitCode {
    match run(Cli::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("qd-prep: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Command::Version => {
            let v = serde_json::json!({
                "name": env!("CARGO_PKG_NAME"),
                "version": env!("CARGO_PKG_VERSION"),
                "minhash_protocol": MINHASH_PROTOCOL,
            });
            println!("{v}");
            Ok(())
        }
        Command::Minhash {
            input,
            output,
            threads,
        } => {
            let threads = match threads {
                Some(n) => n,
                None => std::thread::available_parallelism()
                    .map(|n| n.get().min(MAX_THREADS))
                    .unwrap_or(1),
            };
            let bytes = read_request_file(&input)?;
            let req = parse_request(&bytes)
                .with_context(|| format!("{}: malformed request", input.display()))?;
            let hasher = MinHasher::new(req.num_perm, req.seed)?;
            let sigs = sign_all(&hasher, &req.sets, threads)?;
            let out = encode_output(req.num_perm, req.sets.len(), &sigs)?;
            write_atomically(&output, &out)
        }
    }
}
