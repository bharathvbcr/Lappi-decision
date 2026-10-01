//! `qd-margin-probe` — would the runtime's calibrated-margin half move `ood_abstain`?
//!
//! A diagnostic, not a gate (Fable decision C, approved 2026-09-30). `ood_abstain` is scored
//! with the margin half of the runtime's abstain rule NOT applied, because no calibration table
//! exists (`GAP-RT-CALIBRATION-NOT-FITTED`). This reads the verdict files a scoring run wrote —
//! the val set's (`--verdicts-out`, which carries each choice row's `row_logits`) and the OOD
//! suite's (`--suite-verdicts-out`, `row_logits_1` per case) — and computes every margin with
//! [`qd_runtime::calibration::calibrate`], the function `answer.rs` abstains on, at
//! temperature 1. It then fits a `noul_margin` on one keyed-hash half of the val choice rows,
//! with the same sweep as `python/qd_train/calibration_fit.py::fit_noul_margin`, and reports
//! what that margin would add to the abstentions on the other half and on each OOD category.
//!
//! The fitted margin is reported, never installed: choosing a `noul_margin` is a runtime
//! threshold, which is the human's (CLAUDE.md rule 2). Temperature is fixed at 1.0 because no
//! temperature was fitted; a margin fitted at T=1 is on the wrong scale for a table that later
//! ships a fitted temperature, and the report says so.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use qd_runtime::calibration::{CalibrationEntry, calibrate};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Every failure is a message: this binary links no error crate (qd-runtime adds none).
type Result<T> = std::result::Result<T, String>;

macro_rules! ensure {
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            return Err(format!($($arg)+));
        }
    };
}

/// A verdict file larger than this is not one a scoring run wrote (the val file is ~1.3 MB
/// without logits, a few MB with them); refused rather than read into memory.
const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;
/// The quantiles reported for every margin population.
const QUANTILES: [f64; 7] = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0];

#[derive(Parser, Debug)]
#[command(about = "Fit-and-report diagnostic for the runtime's calibrated-margin abstain half")]
struct Args {
    /// `--verdicts-out` JSONL of one eval row (choice rows must carry `row_logits`).
    #[arg(long)]
    val: PathBuf,
    /// `--suite-verdicts-out` JSONL of the same eval row (OOD lines carry `row_logits_1`).
    #[arg(long)]
    suite: PathBuf,
    /// Key for the half split of the val choice rows: half A iff
    /// `sha256(key || 0x00 || row_id)[0]` is even. Recorded in the report.
    #[arg(long)]
    split_key: String,
    /// Retained-set precision the fitted margin must reach on half A.
    #[arg(long, default_value_t = 0.95)]
    target_precision: f64,
    /// Report path; refused if it exists.
    #[arg(long)]
    out: PathBuf,
}

/// One scored item: its margin at T=1, and what the margin is judged against.
#[derive(Debug, Clone)]
struct Item {
    margin: f64,
    correct: bool,
}

fn read_lines(path: &Path) -> Result<Vec<Value>> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    ensure!(
        meta.len() <= MAX_INPUT_BYTES,
        "{}: {} bytes is not a verdict file (cap {MAX_INPUT_BYTES})",
        path.display(),
        meta.len()
    );
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader = BufReader::new(file.take(MAX_INPUT_BYTES));
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|e| format!("{} line {}: not JSON: {e}", path.display(), i + 1))?;
        out.push(value);
    }
    ensure!(!out.is_empty(), "{}: no verdict lines", path.display());
    Ok(out)
}

fn logits_of(line: &Value, key: &str, what: &str) -> Result<Vec<f32>> {
    let values = line.get(key).and_then(Value::as_array).ok_or_else(|| {
        format!("{what}: no `{key}` -- written by a scoring run older than 444bedc")
    })?;
    values
        .iter()
        .map(|v| {
            v.as_f64()
                .map(|x| x as f32)
                .filter(|x| x.is_finite())
                .ok_or_else(|| format!("{what}: `{key}` holds a non-finite or non-number value"))
        })
        .collect()
}

/// The runtime's margin for these logits at temperature 1: `calibrate`, nothing re-derived.
fn margin(logits: &[f32], what: &str) -> Result<f64> {
    let entry = CalibrationEntry {
        temperature: 1.0,
        conformal_quantile: 1.0,
        noul_margin: 0.0,
    };
    calibrate(what, logits, &entry)
        .map(|c| c.margin)
        .map_err(|e| format!("{what}: {e}"))
}

fn in_half_a(key: &str, row_id: &str) -> bool {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    h.update([0u8]);
    h.update(row_id.as_bytes());
    h.finalize()[0] % 2 == 0
}

/// `fit_noul_margin`, exactly: sweep the distinct observed margins ascending and return the
/// first whose retained set (`margin >= thr`) reaches `target`; if none does, the largest
/// observed margin. Pinned to the Python reference by `test_margin_probe_parity.py`.
fn fit_noul_margin(items: &[Item], target: f64) -> Result<f64> {
    ensure!(!items.is_empty(), "no items to fit a margin on");
    ensure!(
        target > 0.0 && target <= 1.0,
        "target precision must be in (0, 1], got {target}"
    );
    let mut sorted: Vec<f64> = items.iter().map(|i| i.margin).collect();
    ensure!(
        sorted.iter().all(|m| m.is_finite()),
        "a margin is non-finite"
    );
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    for thr in &sorted {
        let (kept, right) = items
            .iter()
            .filter(|i| i.margin >= *thr)
            .fold((0usize, 0usize), |(k, r), i| {
                (k + 1, r + usize::from(i.correct))
            });
        if kept > 0 && (right as f64) / (kept as f64) >= target {
            return Ok(*thr);
        }
    }
    Ok(*sorted.last().expect("non-empty"))
}

/// Nearest-rank quantiles over the margins, keyed `p00`..`p100`.
fn quantiles(margins: &[f64]) -> Value {
    let mut sorted = margins.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut out = serde_json::Map::new();
    for q in QUANTILES {
        let label = format!("p{:02}", (q * 100.0).round() as u32);
        let value = if sorted.is_empty() {
            Value::Null
        } else {
            let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
            json!(sorted[rank - 1])
        };
        out.insert(label, value);
    }
    Value::Object(out)
}

fn population(margins: &[f64], below: usize, rule_or_margin: Option<usize>) -> Value {
    json!({
        "n": margins.len(),
        "margin_quantiles": quantiles(margins),
        "below_fitted_margin": below,
        "abstained_rule_or_margin": rule_or_margin,
    })
}

fn run(args: &Args) -> Result<Value> {
    ensure!(
        !args.out.exists(),
        "--out {} already exists",
        args.out.display()
    );
    ensure!(!args.split_key.is_empty(), "--split-key must not be empty");

    let mut eval_rows: BTreeMap<String, usize> = BTreeMap::new();
    let (mut half_a, mut half_b) = (Vec::new(), Vec::new());
    for line in read_lines(&args.val)? {
        if let Some(id) = line.get("eval_row_id").and_then(Value::as_str) {
            *eval_rows.entry(id.to_string()).or_default() += 1;
        }
        if line.get("kind").and_then(Value::as_str) != Some("choice") {
            continue;
        }
        if line.get("expected_abstain").and_then(Value::as_bool) == Some(true) {
            continue; // a gold-noul row is not in-distribution evidence (choice_rule_abstentions)
        }
        let row_id = line
            .get("row_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "a val choice line has no row_id".to_string())?;
        let correct = line
            .get("correct")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("val row {row_id}: no boolean `correct`"))?;
        let m = margin(&logits_of(&line, "row_logits", row_id)?, row_id)?;
        let item = Item { margin: m, correct };
        if in_half_a(&args.split_key, row_id) {
            half_a.push(item)
        } else {
            half_b.push(item)
        }
    }
    ensure!(
        eval_rows.len() == 1,
        "--val must hold one eval row's verdicts, found {eval_rows:?}"
    );
    let eval_row_id = eval_rows.keys().next().expect("one").clone();
    ensure!(
        !half_a.is_empty() && !half_b.is_empty(),
        "a val half is empty"
    );

    let thr = fit_noul_margin(&half_a, args.target_precision)?;

    let mut categories: BTreeMap<String, (Vec<f64>, usize, usize, usize)> = BTreeMap::new();
    for line in read_lines(&args.suite)? {
        if line.get("suite").and_then(Value::as_str) != Some("ood") {
            continue;
        }
        if line.get("eval_row_id").and_then(Value::as_str) != Some(eval_row_id.as_str()) {
            return Err(format!(
                "the suite file names another eval row than --val's {eval_row_id}"
            ));
        }
        let case = line
            .get("case_id")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let category = line
            .get("category")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("OOD case {case}: no category"))?
            .to_string();
        let rule = line
            .get("abstained")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("OOD case {case}: no boolean `abstained`"))?;
        let m = margin(&logits_of(&line, "row_logits_1", &case)?, &case)?;
        let entry = categories.entry(category).or_default();
        entry.0.push(m);
        entry.1 += usize::from(m < thr);
        entry.2 += usize::from(rule || m < thr);
        entry.3 += usize::from(rule);
    }
    ensure!(!categories.is_empty(), "--suite holds no OOD line");

    let below = |items: &[Item]| items.iter().filter(|i| i.margin < thr).count();
    let margins = |items: &[Item]| items.iter().map(|i| i.margin).collect::<Vec<_>>();
    let retained_precision = |items: &[Item]| {
        let kept: Vec<&Item> = items.iter().filter(|i| i.margin >= thr).collect();
        if kept.is_empty() {
            Value::Null
        } else {
            json!(kept.iter().filter(|i| i.correct).count() as f64 / kept.len() as f64)
        }
    };
    let mut ood = serde_json::Map::new();
    let (mut ood_n, mut ood_rule, mut ood_both) = (0usize, 0usize, 0usize);
    for (name, (ms, below_n, both, rule)) in &categories {
        ood_n += ms.len();
        ood_rule += rule;
        ood_both += both;
        let mut p = population(ms, *below_n, Some(*both));
        p["abstained_rule_only"] = json!(rule);
        ood.insert(name.clone(), p);
    }
    Ok(json!({
        "tool": "qd-margin-probe",
        "eval_row_id": eval_row_id,
        "val": args.val.display().to_string(),
        "suite": args.suite.display().to_string(),
        "split_key": args.split_key,
        "temperature": 1.0,
        "target_precision": args.target_precision,
        "fitted_noul_margin": thr,
        "fitted_on": "val choice rows in half A (gold-noul rows excluded)",
        "val_half_a": {
            "population": population(&margins(&half_a), below(&half_a), None),
            "retained_precision": retained_precision(&half_a),
        },
        "val_half_b": {
            "population": population(&margins(&half_b), below(&half_b), None),
            "retained_precision": retained_precision(&half_b),
        },
        "ood": ood,
        "ood_total": {"n": ood_n, "abstained_rule_only": ood_rule, "abstained_rule_or_margin": ood_both},
        "caveats": [
            "temperature fixed at 1.0: no temperature was fitted, so this noul_margin is on the T=1 scale",
            "a diagnostic: the fitted margin is reported, not installed; selecting one is the human's (rule 2)",
            "the gate's in-distribution bound reads every val choice row; this reports half B only, so it is not the gate",
        ],
    }))
}

fn write_report(path: &Path, report: &Value) -> Result<()> {
    let body = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("--out {}: {e}", path.display()))?;
    file.write_all(body.as_bytes())
        .and_then(|()| file.write_all(b"\n"))
        .map_err(|e| format!("--out {}: {e}", path.display()))
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    match run(&args).and_then(|report| write_report(&args.out, &report).map(|()| report)) {
        Ok(report) => {
            println!("{report}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("qd-margin-probe: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(pairs: &[(f64, bool)]) -> Vec<Item> {
        pairs
            .iter()
            .map(|(m, c)| Item {
                margin: *m,
                correct: *c,
            })
            .collect()
    }

    #[test]
    fn fit_returns_the_lowest_margin_reaching_the_target() {
        let it = items(&[(0.1, false), (0.2, false), (0.5, true), (0.9, true)]);
        assert_eq!(fit_noul_margin(&it, 0.95).unwrap(), 0.5);
        assert_eq!(fit_noul_margin(&it, 0.5).unwrap(), 0.1);
    }

    #[test]
    fn fit_returns_the_largest_margin_when_nothing_reaches_the_target() {
        let it = items(&[(0.3, false), (0.7, false)]);
        assert_eq!(fit_noul_margin(&it, 0.95).unwrap(), 0.7);
    }

    #[test]
    fn fit_refuses_an_empty_set_and_a_bad_target() {
        assert!(fit_noul_margin(&[], 0.95).is_err());
        assert!(fit_noul_margin(&items(&[(0.1, true)]), 0.0).is_err());
        assert!(fit_noul_margin(&items(&[(0.1, true)]), 1.5).is_err());
    }

    #[test]
    fn the_margin_is_the_runtimes() {
        // softmax([0, ln 3]) = [0.25, 0.75]: margin 0.5.
        let m = margin(&[0.0, 3f32.ln()], "t").unwrap();
        assert!((m - 0.5).abs() < 1e-6, "{m}");
    }

    #[test]
    fn the_split_is_keyed_and_deterministic() {
        let ids: Vec<String> = (0..200).map(|i| format!("r{i}")).collect();
        let a: Vec<bool> = ids.iter().map(|r| in_half_a("k1", r)).collect();
        assert_eq!(
            a,
            ids.iter().map(|r| in_half_a("k1", r)).collect::<Vec<_>>()
        );
        assert_ne!(
            a,
            ids.iter().map(|r| in_half_a("k2", r)).collect::<Vec<_>>()
        );
        let n = a.iter().filter(|x| **x).count();
        assert!((60..140).contains(&n), "{n} of 200 in half A");
    }

    #[test]
    fn quantiles_are_nearest_rank() {
        let q = quantiles(&[0.4, 0.1, 0.3, 0.2]);
        assert_eq!(q["p00"], json!(0.1));
        assert_eq!(q["p50"], json!(0.2));
        assert_eq!(q["p100"], json!(0.4));
    }

    #[test]
    fn a_line_without_logits_is_refused_not_skipped() {
        let line = json!({"row_id": "r"});
        let err = logits_of(&line, "row_logits", "r").unwrap_err().to_string();
        assert!(err.contains("older than 444bedc"), "{err}");
    }
}
