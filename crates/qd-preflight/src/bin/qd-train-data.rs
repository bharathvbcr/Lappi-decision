//! `qd-train-data`: is any benchmark target in this train set? Exit 0 PASS (every tier ran, none
//! found one), 1 FAILED (a hit; counts and sample ids are printed), 3 NOT_RUN (an input could not
//! be read, a positive control failed, or a tier could not run), 2 a bad command line. The last
//! line is `RESULT: ...`, as `make gates` prints.
//!
//! ```text
//! qd-train-data --train SHARD_DIR_OR_EXAMPLES.jsonl --target NAME=FILE [--target ...]
//!               [--examples EXAMPLES.jsonl] [--manifest TRAIN.json]
//!               [--forbid-family FAMILY ...] [--n 8] [--threshold 0.5] [--threads N]
//! ```
//!
//! See `qd_preflight::train_data` for the four tiers and what each needs.

use std::path::PathBuf;
use std::process::ExitCode;

use qd_preflight::train_data::{
    DEFAULT_N, DEFAULT_THRESHOLD, FORBIDDEN_FAMILIES, Inputs, check, parse_target, summary_lines,
};

fn main() -> ExitCode {
    let inputs = match parse(std::env::args().skip(1)) {
        Ok(i) => i,
        Err(msg) => return usage(&msg),
    };
    println!(
        "qd-train-data: train={} targets={:?} n={} threshold={}",
        inputs.train.display(),
        inputs.targets.iter().map(|(n, p)| format!("{n}={}", p.display())).collect::<Vec<_>>(),
        inputs.n,
        inputs.threshold
    );
    let verdict = check(&inputs);
    for line in summary_lines(&verdict) {
        println!("{line}");
    }
    println!("JSON: {}", verdict.to_json());
    println!("{}", verdict.result_line());
    ExitCode::from(verdict.exit_code())
}

fn parse(args: impl Iterator<Item = String>) -> Result<Inputs, String> {
    let mut train: Option<PathBuf> = None;
    let mut inputs = Inputs {
        train: PathBuf::new(),
        examples: None,
        manifest: None,
        targets: Vec::new(),
        forbid_families: FORBIDDEN_FAMILIES.iter().map(|s| (*s).to_owned()).collect(),
        n: DEFAULT_N,
        threshold: DEFAULT_THRESHOLD,
        threads: std::thread::available_parallelism().map_or(1, |p| p.get().min(16)),
    };
    let mut extra_families: Vec<String> = Vec::new();
    let mut it = args;
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--train" => train = Some(PathBuf::from(value)),
            "--examples" => inputs.examples = Some(PathBuf::from(value)),
            "--manifest" => inputs.manifest = Some(PathBuf::from(value)),
            "--target" => inputs.targets.push(parse_target(&value)?),
            "--forbid-family" => extra_families.push(value),
            "--n" => inputs.n = value.parse().map_err(|e| format!("--n {value:?}: {e}"))?,
            "--threshold" => {
                inputs.threshold = value.parse().map_err(|e| format!("--threshold {value:?}: {e}"))?;
            }
            "--threads" => {
                inputs.threads = value.parse().map_err(|e| format!("--threads {value:?}: {e}"))?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    inputs.train = train.ok_or("--train is required")?;
    if inputs.targets.is_empty() {
        return Err("at least one --target NAME=FILE is required".into());
    }
    if !(1..=64).contains(&inputs.n) {
        return Err(format!("--n {} is outside 1..=64", inputs.n));
    }
    if !(inputs.threshold > 0.0 && inputs.threshold <= 1.0) {
        return Err(format!("--threshold {} is outside (0, 1]", inputs.threshold));
    }
    if inputs.threads == 0 || inputs.threads > 256 {
        return Err(format!("--threads {} is outside 1..=256", inputs.threads));
    }
    for f in extra_families {
        if !inputs.forbid_families.contains(&f) {
            inputs.forbid_families.push(f);
        }
    }
    Ok(inputs)
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("qd-train-data: {msg}");
    eprintln!(
        "usage: qd-train-data --train SHARD_DIR|EXAMPLES.jsonl --target NAME=FILE [--target ...] \
         [--examples FILE] [--manifest FILE] [--forbid-family F] [--n 8] [--threshold 0.5] [--threads N]"
    );
    ExitCode::from(2)
}
