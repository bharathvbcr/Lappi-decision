//! `qd-export-ensemble` — N released towers and the ensemble's calibration table -> one ensemble
//! directory the runtime serves as one backend.
//!
//! ```text
//! qd-export-ensemble --member REL0 --trained-width 1105 \
//!                    --member REL1 --trained-width 1105 \
//!                    --member REL2 --trained-width 1105 \
//!                    --trained-width-source "ledger rows ..." \
//!                    --calibration ENSEMBLE_TABLE.json --out ENSEMBLE
//! ```
//!
//! `--member` and `--trained-width` pair by position. Exit codes: 0 written; 2 refused (nothing
//! is left behind); 1 the filesystem failed.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use qd_export::{EnsembleRequest, RefusalKind, export_ensemble};

#[derive(Parser, Debug)]
#[command(
    name = "qd-export-ensemble",
    version,
    about = "Write an N-tower ensemble directory from released towers and the ensemble's table"
)]
struct Cli {
    /// A member release directory (`qd-export --out`). Repeat once per tower, in the order
    /// their log-probabilities are averaged.
    #[arg(long = "member", value_name = "DIR", required = true)]
    members: Vec<PathBuf>,

    /// The member's trained width in tokens, paired with `--member` by position.
    #[arg(long = "trained-width", value_name = "TOKENS", required = true)]
    trained_widths: Vec<u64>,

    /// Where the widths come from (ledger row ids, a handoff); recorded in the manifest.
    #[arg(long, value_name = "TEXT")]
    trained_width_source: String,

    /// The ensemble's calibration table, fitted on the ensemble's mean decode.
    #[arg(long, value_name = "PATH")]
    calibration: PathBuf,

    /// The ensemble directory to create. It must not exist.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let req = EnsembleRequest {
        members: cli.members,
        trained_widths: cli.trained_widths,
        trained_width_source: cli.trained_width_source,
        calibration: cli.calibration,
        out: cli.out,
    };
    match export_ensemble(&req) {
        Ok(s) => {
            println!("ensemble: {}", s.out.display());
            for (i, hash) in s.member_weight_hashes.iter().enumerate() {
                println!("member {i} weight_hash: {hash}");
            }
            println!("weight_hash: {}", s.weight_hash);
            println!("calibration_hash: {}", s.calibration_hash);
            println!("trained_width: {}", s.trained_width);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("qd-export-ensemble: refused: {e}");
            if e.kind == RefusalKind::Io {
                ExitCode::from(1)
            } else {
                ExitCode::from(2)
            }
        }
    }
}
