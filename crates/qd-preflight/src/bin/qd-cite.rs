//! `qd-cite`: every measured number in a new handoff or audit cites a ledger row, a gap record or
//! an evidence path. See `qd_preflight::cite` for the rule.
//!
//! ```text
//! qd-cite [--root DIR] [--since YYYY-MM-DD] [FILE...]
//! ```
//!
//! With no FILE, every `HANDOFF/*.md` and `AUDIT/**/*.md` under the root. Exit 0 PASS, 1 FAIL,
//! 3 NOT_RUN (the ledger or gaps.jsonl could not be read: no id could be resolved). The last line
//! is `RESULT: PASS|FAIL|NOT_RUN`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use qd_preflight::cite::{run, scope_files, Reason, DEFAULT_SINCE};

#[derive(Parser, Debug)]
#[command(name = "qd-cite", version)]
struct Cli {
    /// The repository root: `ledger/`, `gaps.jsonl`, `HANDOFF/`, `AUDIT/`.
    #[arg(long, default_value = ".")]
    root: PathBuf,
    /// Examine files dated on or after this day. Earlier files are historical and not flagged.
    #[arg(long, default_value = DEFAULT_SINCE)]
    since: String,
    files: Vec<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let files = if cli.files.is_empty() {
        match scope_files(&cli.root) {
            Ok(f) => f,
            Err(e) => return not_run(&format!("{e:#}")),
        }
    } else {
        cli.files.clone()
    };
    let report = match run(&cli.root, &files, &cli.since) {
        Ok(r) => r,
        Err(e) => return not_run(&format!("{e:#}")),
    };
    for f in &report.findings {
        let what = match &f.reason {
            Reason::Uncited(n) => format!("measured number {n} with no ledger row, gap id or [V: path] in its paragraph"),
            Reason::Unresolvable(why) => why.clone(),
        };
        println!("{}:{}: {what}\n    {}", f.file.display(), f.line, f.text);
    }
    println!(
        "files in scope (dated >= {}): {}; before the cutoff: {}; undated: {}; findings: {}",
        cli.since,
        report.in_scope,
        report.skipped_before_cutoff,
        report.skipped_undated,
        report.findings.len()
    );
    if report.findings.is_empty() {
        println!("RESULT: PASS");
        ExitCode::SUCCESS
    } else {
        println!("RESULT: FAIL");
        ExitCode::from(1)
    }
}

fn not_run(why: &str) -> ExitCode {
    eprintln!("qd-cite: {why}");
    println!("RESULT: NOT_RUN");
    ExitCode::from(3)
}
