//! `qd-metal-record-parity` — run the parity gate and write its result as a ledger row.
//!
//! ```text
//! qd-metal-record-parity --parity-bin target/release/qd-metal-parity \
//!     --ledger ledger/mac-qd-metal-<date>.jsonl --raw-out <scratch>/parity-raw.jsonl \
//!     -- --fixtures <torch reference dir> [--snapshot <dir>]
//! ```
//!
//! Runs `<parity-bin> <args after --> --ledger <raw-out>`, passes its output through, times it,
//! and appends one `smoke` row built by [`qd_metal::parity_row`] from the object the gate wrote.
//! `--raw-out` must not exist: the row cites exactly the object this run wrote. **This runs on
//! the GPU** (through the parity binary).
//!
//! Exit: the gate's own code (0 pass, 1 a threshold failed, 2 it could not run) once the row is
//! written, except that a pass with a non-finite value exits 1; 2 when the row could not be
//! written. A gate run whose row is missing did not happen.

use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::time::Instant;

use qd_metal::ledger::{self, Provenance};
use qd_metal::parity_row::{self, ModelFacts, ParityRun};
use qd_metal::{MetalError, Result};

struct Args {
    parity_bin: PathBuf,
    ledger: PathBuf,
    raw_out: PathBuf,
    passthrough: Vec<String>,
}

fn parse(argv: &[String]) -> Result<Args> {
    let (ours, theirs) = match argv.iter().position(|a| a == "--") {
        Some(i) => (&argv[..i], argv[i + 1..].to_vec()),
        None => (argv, Vec::new()),
    };
    let (mut bin, mut led, mut raw) = (None, None, None);
    let mut it = ours.iter();
    while let Some(a) = it.next() {
        let v = it
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| MetalError::Input(format!("{a} needs a value")))?;
        match a.as_str() {
            "--parity-bin" => bin = Some(v),
            "--ledger" => led = Some(v),
            "--raw-out" => raw = Some(v),
            other => return Err(MetalError::Input(format!("unknown argument {other:?}"))),
        }
    }
    if theirs.iter().any(|a| a == "--ledger") {
        return Err(MetalError::Input(
            "the gate's --ledger is --raw-out here; it never writes into ledger/ itself".into(),
        ));
    }
    if !theirs.iter().any(|a| a == "--fixtures") {
        return Err(MetalError::Input("pass the gate's --fixtures <dir> after --".into()));
    }
    let args = Args {
        parity_bin: bin.ok_or_else(|| MetalError::Input("--parity-bin is required".into()))?,
        ledger: led.ok_or_else(|| MetalError::Input("--ledger is required".into()))?,
        raw_out: raw.ok_or_else(|| MetalError::Input("--raw-out is required".into()))?,
        passthrough: theirs,
    };
    ledger::check_ledger_path(&args.ledger)?;
    if args.raw_out.exists() {
        return Err(MetalError::Input(format!(
            "{} exists; the row must cite exactly the object this run writes",
            args.raw_out.display()
        )));
    }
    Ok(args)
}

/// The value after `flag` in `argv`.
fn flag_value<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
    argv.iter().position(|a| a == flag).and_then(|i| argv.get(i + 1)).map(String::as_str)
}

fn run(argv: &[String]) -> Result<i32> {
    let args = parse(argv)?;
    // What the row will say about the model, resolved before the GPU is touched.
    let snapshot = qd_metal::config::resolve_snapshot(flag_value(&args.passthrough, "--snapshot").map(std::path::Path::new))?;
    let cfg = qd_metal::config::ModelConfig::load(&snapshot)?;
    let model = ModelFacts {
        snapshot_name: snapshot
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string(),
        vocab: cfg.vocab,
    };
    let prov = Provenance::of(Some(&args.parity_bin))?;
    println!("tessl: {}", prov.tessl.describe());

    let mut cmd_args = args.passthrough.clone();
    cmd_args.push("--ledger".into());
    cmd_args.push(args.raw_out.to_string_lossy().into_owned());
    let started = Instant::now();
    let out = Command::new(&args.parity_bin)
        .args(&cmd_args)
        .output()
        .map_err(|e| MetalError::Input(format!("{}: {e}", args.parity_bin.display())))?;
    let wall_clock_s = started.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    print!("{stdout}");
    eprint!("{stderr}");
    let exit_code = out.status.code().unwrap_or(-1);

    let raw_line = match std::fs::read_to_string(&args.raw_out) {
        Ok(text) => {
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            match lines.as_slice() {
                [one] => Some((*one).to_string()),
                other => {
                    return Err(MetalError::Ledger(format!(
                        "{} holds {} objects; one run writes one",
                        args.raw_out.display(),
                        other.len()
                    )));
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(MetalError::Ledger(format!("{}: {e}", args.raw_out.display()))),
    };
    let run = ParityRun {
        raw_line,
        exit_code,
        stdout,
        stderr,
        wall_clock_s,
        argv: cmd_args,
    };
    let row = parity_row::build_row(&run, &prov, &model)?;
    let gate_pass = row.metrics.get("parity.gate").is_some_and(|t| t.is_pass());
    let stamp = ledger::write_row(&args.ledger, &row)?;
    println!(
        "ledger row {} appended to {} (parity.gate {})",
        stamp.row_id,
        args.ledger.display(),
        if gate_pass { "pass" } else { "NOT pass" }
    );
    Ok(match exit_code {
        0 if gate_pass => 0,
        0 => 1,
        1 => 1,
        _ => 2,
    })
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(2)),
        Err(e) => {
            eprintln!("qd-metal-record-parity: NO ROW WRITTEN: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn the_gate_never_writes_into_ledger_itself() {
        let base = ["--parity-bin", "/b", "--ledger", "/r/ledger/mac-qd-metal-x.jsonl", "--raw-out", "/nonexistent/raw.jsonl"];
        let mut ok = s(&base);
        ok.extend(s(&["--", "--fixtures", "/f"]));
        let a = parse(&ok).unwrap();
        assert_eq!(a.passthrough, s(&["--fixtures", "/f"]));
        let mut bad = s(&base);
        bad.extend(s(&["--", "--fixtures", "/f", "--ledger", "/r/ledger/mac-qd-metal-x.jsonl"]));
        assert!(parse(&bad).is_err());
        let mut campaign = s(&["--parity-bin", "/b", "--ledger", "/r/ledger/runs.jsonl", "--raw-out", "/n/raw.jsonl"]);
        campaign.extend(s(&["--", "--fixtures", "/f"]));
        assert!(parse(&campaign).is_err());
        assert!(parse(&s(&base)).is_err(), "no --fixtures after --");
    }
}
