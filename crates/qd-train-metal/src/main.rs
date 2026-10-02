//! `qd-train-metal`: one fine-tune run on this Mac's GPU. See the library docs for what it runs
//! and what it refuses; `--help` prints the flags.
//!
//! Exit codes: 0 the run finished and its row was written; 2 the command line was refused or
//! malformed (nothing was read or opened); 1 the run was refused or failed after that (no row
//! was written).

use std::process::ExitCode;

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match qd_train_metal::args::parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("qd-train-metal: {e}");
            return ExitCode::from(2);
        }
    };
    match qd_train_metal::run::run(&parsed) {
        Ok(s) => {
            println!(
                "qd-train-metal: {} after {} optimizer steps; row {} in {}; export {} (sha256 {}); masters {}",
                s.termination.as_str(),
                s.optimizer_steps,
                s.row_id,
                parsed.ledger.display(),
                s.export.weights.display(),
                s.export.safetensors_sha256,
                s.masters.weights.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("qd-train-metal: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = qd_train_metal::args::parse(&args) {
        eprintln!("qd-train-metal: {e}");
        return ExitCode::from(2);
    }
    eprintln!("qd-train-metal: refused: the step runs on tessl's Metal kernels, which build only on macOS");
    ExitCode::from(1)
}
