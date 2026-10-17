//! `qd-release-check`: is a release shown to serve what it admits?
//!
//! ```text
//! qd-release-check replay --release DIR --socket SOCK --out RESULTS.jsonl [--requests FILE]... [--qd BIN]
//! qd-release-check check  --release DIR --results RESULTS.jsonl --min-answered N
//!                         [--verdicts V.jsonl [--calib-rows R.jsonl] [--calibration-report REPORT.json]]
//! ```
//!
//! `replay` is one command: it sends the documented examples and every request in the given JSONL
//! files through `qd oneshot --socket` (the product path), asks the warm build for its identity
//! the same way, and writes the results file. `check` reads it. Exit codes: 0 PASS, 1 FAIL, 3
//! NOT_RUN; the last line is `RESULT: PASS|FAIL|NOT_RUN`. Clap's own usage errors exit 2.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use qd_preflight::release_check::{
    documented_refusals, exit_code, load_examples, load_results, metrics, overall, replay,
    result_line, run_checks, Inputs, ReleaseView, ReplayConfig, Verdict,
};

#[derive(Parser, Debug)]
#[command(name = "qd-release-check", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Replay the documented examples and request files through `qd oneshot --socket`.
    Replay {
        #[arg(long)]
        release: PathBuf,
        /// The socket a `qd-metal-serve` (or `qd serve`) bound to this release listens on.
        #[arg(long)]
        socket: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// JSONL, one wire request per line. Repeatable.
        #[arg(long = "requests")]
        requests: Vec<PathBuf>,
        /// The `qd` binary.
        #[arg(long, default_value = "qd")]
        qd: PathBuf,
        /// The repository root: `docs/` and `fixtures/wire/`.
        #[arg(long, default_value = ".")]
        repo: PathBuf,
        #[arg(long, default_value_t = 20_000)]
        socket_timeout_ms: u64,
        /// Refuse to replay more than this many requests.
        #[arg(long, default_value_t = 100_000)]
        max_requests: usize,
    },
    /// Check a release against a replay results file.
    Check {
        #[arg(long)]
        release: PathBuf,
        #[arg(long)]
        results: PathBuf,
        /// Distinct answered requests required per (admitted family, trained request shape). No
        /// default: no record states one.
        #[arg(long)]
        min_answered: usize,
        #[arg(long, default_value = ".")]
        repo: PathBuf,
        /// Eval verdict rows (`family_id`, `kind`, `correct`, `row_id`): the headline figures.
        #[arg(long)]
        verdicts: Option<PathBuf>,
        /// qd-calib-fit's rows.jsonl, for coverage and precision on answered.
        #[arg(long)]
        calib_rows: Option<PathBuf>,
        /// qd-calib-fit's report.json, for the precision target.
        #[arg(long)]
        calibration_report: Option<PathBuf>,
        #[arg(long)]
        eval_row_id: Option<String>,
        /// Also write the full report as JSON.
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            // An input that could not be read is NOT_RUN, never a pass and never a failure of the release.
            eprintln!("qd-release-check: {e:#}");
            println!("{}", result_line(Verdict::NotRun));
            ExitCode::from(exit_code(Verdict::NotRun))
        }
    }
}

fn run() -> Result<ExitCode> {
    match Cli::parse().command {
        Command::Replay { release, socket, out, requests, qd, repo, socket_timeout_ms, max_requests } => {
            let cfg = ReplayConfig {
                qd,
                socket,
                release,
                repo,
                request_files: requests,
                socket_timeout: Duration::from_millis(socket_timeout_ms.max(1)),
                max_requests,
            };
            let text = replay(&cfg)?;
            std::fs::write(&out, &text).with_context(|| format!("writing {}", out.display()))?;
            println!("wrote {} ({} lines)", out.display(), text.lines().count());
            Ok(ExitCode::SUCCESS)
        }
        Command::Check {
            release, results, min_answered, repo, verdicts, calib_rows, calibration_report, eval_row_id, json_out,
        } => {
            let view = ReleaseView::load(&release)?;
            let res = load_results(&results)?;
            let examples = load_examples(&repo)?;
            let refusals = documented_refusals(&repo)?;
            let inp = Inputs { release: &view, results: &res, examples: &examples, refusals: &refusals, min_answered };
            let (checks, cov) = run_checks(&inp);

            println!("release: {}", view.dir.display());
            println!("admitted families: {}; excluded (recorded): {}", view.trained_families.len(), view.excluded_families.len());
            for (f, why) in &view.excluded_families {
                println!("  excluded {f}: {why}");
            }
            for c in &checks {
                println!("{:?} {}: {}", c.verdict, c.name, c.detail);
            }
            if let Some(cov) = &cov {
                for c in cov {
                    println!(
                        "  coverage {} {:?}: answered {}, refused {} {:?}",
                        c.family, c.shape, c.answered, c.refused, c.refusal_kinds
                    );
                }
            }

            let admitted: BTreeSet<String> = view.trained_families.iter().cloned().collect();
            let shown: BTreeSet<String> = cov
                .as_ref()
                .map(|cov| {
                    admitted
                        .iter()
                        .filter(|f| {
                            let mine: Vec<_> = cov.iter().filter(|c| &c.family == *f).collect();
                            !mine.is_empty() && mine.iter().all(|c| c.answered >= min_answered)
                        })
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            let target = match &calibration_report {
                Some(p) => {
                    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
                    v.get("target_precision").and_then(serde_json::Value::as_f64)
                }
                None => None,
            };
            let m = match &verdicts {
                Some(v) => match metrics(&admitted, &shown, v, calib_rows.as_deref(), target, eval_row_id.as_deref()) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        println!("NotRun headline metrics: {e:#}");
                        None
                    }
                },
                None => {
                    println!("NotRun headline metrics: no --verdicts; no release headline figure is computed, and an all-rows figure is not the release's");
                    None
                }
            };
            if let Some(m) = &m {
                let show = |name: &str, f: &qd_preflight::release_check::Fraction| {
                    let pct = if f.denominator == 0 { "n/a".to_string() } else { format!("{:.4}", f.numerator as f64 / f.denominator as f64) };
                    println!("  {name}: {pct} ({}/{})", f.numerator, f.denominator);
                };
                for (label, p) in [("admitted_population", &m.admitted_population), ("shown_served_population", &m.shown_served_population)] {
                    println!("headline over {label} ({} families):", p.families.len());
                    show("choice_top1", &p.choice_top1);
                    show("coverage", &p.coverage);
                    show("precision_on_answered", &p.precision_on_answered);
                    println!("  verdict rows with no calibration row: {}", p.rows_without_calibration_row);
                }
                println!("  target_precision: {:?}", m.target_precision);
                println!("  excluded rows (families not admitted): {} {:?}", m.excluded_rows, m.excluded_families);
                show("NOT_THE_RELEASES_all_rows_choice_top1", &m.not_the_releases_all_rows_choice_top1);
            }

            let verdict = overall(&checks);
            if let Some(p) = json_out {
                let doc = serde_json::json!({"checks": checks, "coverage": cov, "metrics": m, "min_answered": min_answered});
                std::fs::write(&p, serde_json::to_vec_pretty(&doc)?)?;
            }
            println!("{}", result_line(verdict));
            Ok(ExitCode::from(exit_code(verdict)))
        }
    }
}

