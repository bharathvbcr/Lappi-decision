//! `qd-gate-report` — the numbers a human needs to rule on the open gate questions, re-reported
//! on CPU from a scoring run's verdict files. **Report-only.**
//!
//! Fable round G (2026-10-01). Reads `tools/real_ft_run.py`'s `--verdicts-out` files (one eval
//! row each) and, optionally, its `--suite-verdicts-out` files, and reports:
//!
//! * **G1** — every gate and control's value **per val family beside its pooled value**. The
//!   pooled value is the eval row's recorded verdict, verbatim, labelled as the gate's
//!   population; a family's value carries no pass/fail, because whether a bar applies per family
//!   is the human's ruling (`GAP-GATES-POOL-THE-GENERAL-FAMILIES-INTO-DEFECT-CONTRACTS`). A gate a
//!   verdict file cannot score per family is `not_run` per family, with the reason — never absent.
//! * **G2** — ECE over every letter row pooled across slot shapes, and per language with a
//!   `none` bucket for the rows that carry no language
//!   (`GAP-ECE-GATE-IS-NOT-RUN-ON-THE-FULL-MIXTURE`). The `ece` gate is not touched.
//! * **G3** — for every two- and four-option slot shape, overall and per family: mean entropy,
//!   accuracy, the majority-class rate, and each decode row's predicted marginal against its gold
//!   marginal with the deviation in standard errors (`GAP-DEGENERATE-HEAD-FAILS-ON-BINARY-SLOTS-TOO`).
//!   The `degenerate_head` control is not touched.
//!
//! It also prints the promotion population exactly as the human decisions record
//! (`docs/promotion-decisions.json`) states it, with the record's sha256, and the questions that
//! record still holds open.
//!
//! # Report-only
//!
//! Nothing here is a gate, moves a threshold, or changes a population (CLAUDE.md rule 2). The
//! entropy floor and class-share cap are `eval_harness.py`'s, stated beside each value and applied
//! to nothing; the ECE bar is `ece_gate`'s, stated the same way. No report value says `passed`.
//!
//! # Proof it read the right files
//!
//! Each verdict file names one eval row, which must be in an `--eval-ledger` exactly once, at the
//! file's seed. Every number that row also records is recomputed here and compared: the decoded
//! line count, `val_top1.*`, `permutation_consistency` and its per-family split, the
//! in-distribution abstentions pooled and per family, every `ece.{shape}`, `ece.lang.*`,
//! `ece.family.*` and `degenerate_head.{shape}`, and — given suite verdicts — each needle depth
//! bucket, the needle gate's worst bucket, each OOD category and the OOD gate's count. Any
//! disagreement refuses the whole report, and so does a report that could check nothing: a
//! cross-check that did not run must not read as one that passed.
//!
//! # Numbers
//!
//! Logits are read as f64, as `letter_distributions` reads them (qd-calib-fit reads f32, the
//! runtime's view; on bf16-derived logits the two are the same values). Softmax, entropy and every
//! mean use numpy's pairwise summation ([`pairwise_sum`]) in file order, and the ECE is
//! [`ece_state`], so the recomputed values equal the Python ones bit for bit on this host
//! (`python/tests/test_gate_report_parity.py`). The ledger comparison allows 1e-12 absolute.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use qd_runtime::calibration_fit::{ECE_BINS, ECE_THRESHOLD, EceState, ece_state, pairwise_sum};
use qd_runtime::schema::{
    MAX_BINS, MAX_OPTIONS, MIN_BINS, MIN_OPTIONS, RESERVED_NOUL_ROWS, SlotKind,
};
use serde_json::{Map, Value, json};
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

/// A verdict file larger than this is not one a scoring run wrote (J4's are ~8 MB).
const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;
/// A ledger larger than this is not one this repo writes.
const MAX_LEDGER_BYTES: u64 = 256 * 1024 * 1024;
/// The decisions record is a page of JSON (`ledger.py`'s `MAX_DECISIONS_BYTES`).
const MAX_DECISIONS_BYTES: u64 = 1 << 20;
/// More files than this of one kind is not one report.
const MAX_FILES: usize = 16;
/// `eval_harness.DEFAULT_ENTROPY_FLOOR` (nats). Read-only (rule 2): stated, never applied here.
const ENTROPY_FLOOR: f64 = 0.15;
/// `eval_harness.DEFAULT_MAX_CLASS_SHARE`. Read-only: stated, never applied here.
const MAX_CLASS_SHARE: f64 = 0.95;
/// `eval_harness._entropy`'s clip.
const ENTROPY_CLIP: f64 = 1e-12;
/// How far a recomputed value may sit from the eval row's before the report refuses.
const VALUE_TOLERANCE: f64 = 1e-12;
/// G3's slot shapes: two and four options.
const SLOT_OPTIONS: [usize; 2] = [2, 4];
/// What the reporter says about every per-family value.
const PER_FAMILY_NOTE: &str = "report-only: no pass/fail, because whether a gate's bar applies \
     per family is the human's ruling (GAP-GATES-POOL-THE-GENERAL-FAMILIES-INTO-DEFECT-CONTRACTS)";
/// `real_ft_run.NO_FAMILY_REASON`, for the same case.
const NO_FAMILY_REASON: &str = "carry no family_id: _decode writes the label's family onto \
     every verdict, so these were not produced by it and cannot be grouped by family";

#[derive(Parser, Debug)]
#[command(
    about = "Report-only gate metrics per family, pooled ECE and slot diagnostics from \
                   --verdicts-out files"
)]
struct Args {
    /// A `--verdicts-out` JSONL file: one eval row's val verdicts. Repeat for more rows.
    #[arg(long = "verdicts", required = true)]
    verdicts: Vec<PathBuf>,
    /// A `--suite-verdicts-out` JSONL file; its lines are matched to rows by `eval_row_id`.
    #[arg(long = "suite-verdicts")]
    suite_verdicts: Vec<PathBuf>,
    /// A ledger holding the eval rows the verdicts belong to. Repeatable; required, because the
    /// cross-check against the row is the proof the report read that row's files.
    #[arg(long = "eval-ledger", required = true)]
    eval_ledger: Vec<PathBuf>,
    /// The human decisions record (`docs/promotion-decisions.json`).
    #[arg(long)]
    decisions: PathBuf,
    /// Write the JSON report here; refused if the path exists. The text report goes to stdout.
    #[arg(long = "out-json")]
    out_json: Option<PathBuf>,
}

/// The permuted second pass `annotate_second_pass` writes onto an asked choice row.
#[derive(Debug, Clone)]
struct Second {
    perm: Vec<usize>,
    top_permuted: usize,
    agreed: bool,
}

/// One letter verdict line, checked, with its distribution computed once.
#[derive(Debug, Clone)]
struct Letter {
    row_id: String,
    family: Option<String>,
    language: Option<String>,
    kind: SlotKind,
    rows: usize,
    noul_row: usize,
    gold: usize,
    top: usize,
    correct: bool,
    expected_abstain: bool,
    second: Option<Second>,
    probs: Vec<f64>,
    argmax: usize,
    confidence: f64,
    entropy: f64,
}

impl Letter {
    /// `{kind}.k{options}`: `letter_distributions`' key, one entry of the calibration table.
    fn shape(&self) -> String {
        format!("{}.k{}", self.kind.as_str(), self.rows - RESERVED_NOUL_ROWS)
    }
}

/// One span verdict line: only what accuracy reads.
#[derive(Debug, Clone)]
struct SpanLine {
    family: Option<String>,
    correct: bool,
}

/// One verdict file: one eval row's val verdicts.
#[derive(Debug)]
struct VerdictFile {
    path: String,
    sha256: String,
    eval_row_id: String,
    seed: i64,
    lines: usize,
    letters: Vec<Letter>,
    spans: Vec<SpanLine>,
}

/// The eval row a verdict file belongs to, as its ledger holds it.
#[derive(Debug)]
struct EvalRow {
    ledger: String,
    row: Map<String, Value>,
}

/// One suite verdict line's fields the cross-check and the report read.
#[derive(Debug, Clone)]
enum SuiteLine {
    Needle { bucket: String, hit: bool },
    Ood { category: String, abstained: bool },
}

/// Every suite file's identity, and every suite line by the eval row it names.
type SuiteRead = (Vec<Value>, BTreeMap<String, Vec<SuiteLine>>);

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Read a whole file under a byte cap, and hash exactly the bytes that were read.
fn read_bounded(path: &Path, cap: u64) -> Result<(Vec<u8>, String)> {
    let shown = path.display();
    let meta = std::fs::metadata(path).map_err(|e| format!("{shown}: {e}"))?;
    ensure!(
        meta.len() <= cap,
        "{shown}: {} bytes is over the {cap}-byte cap for this input",
        meta.len()
    );
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    std::fs::File::open(path)
        .and_then(|f| f.take(cap + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("{shown}: {e}"))?;
    ensure!(
        bytes.len() as u64 <= cap,
        "{shown}: grew past {cap} bytes while it was read"
    );
    let sha = hex(&Sha256::digest(&bytes));
    Ok((bytes, sha))
}

/// Each non-blank line of a JSONL file as a JSON object, with where it came from.
fn json_lines(bytes: &[u8], shown: &str) -> Result<Vec<(String, Map<String, Value>)>> {
    let mut out = Vec::new();
    for (i, raw) in bytes.split(|b| *b == b'\n').enumerate() {
        if raw.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let at = format!("{shown} line {}", i + 1);
        let line: Value = serde_json::from_slice(raw).map_err(|e| {
            format!("{at}: not JSON ({e}); a NaN or Infinity token is refused, not read")
        })?;
        match line {
            Value::Object(map) => out.push((at, map)),
            _ => return Err(format!("{at}: not a JSON object")),
        }
    }
    Ok(out)
}

fn str_field<'a>(line: &'a Map<String, Value>, key: &str, at: &str) -> Result<&'a str> {
    match line.get(key).and_then(Value::as_str) {
        Some(s) if !s.is_empty() => Ok(s),
        _ => Err(format!("{at}: no non-empty string `{key}`")),
    }
}

fn index_field(line: &Map<String, Value>, key: &str, at: &str) -> Result<usize> {
    line.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("{at}: `{key}` is not a non-negative integer"))
}

fn bool_field(line: &Map<String, Value>, key: &str, at: &str) -> Result<bool> {
    line.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{at}: no boolean `{key}`"))
}

/// An optional label: absent, null and "" are all "none"; any other non-string is refused.
fn label_field(line: &Map<String, Value>, key: &str, at: &str) -> Result<Option<String>> {
    match line.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(format!("{at}: `{key}` is {other}, not a string or null")),
    }
}

/// The legal decode-row counts of a letter slot (qd-calib-fit's `legal_rows`).
fn legal_rows(kind: SlotKind) -> (usize, usize) {
    match kind {
        SlotKind::Choice => (
            MIN_OPTIONS + RESERVED_NOUL_ROWS,
            MAX_OPTIONS + RESERVED_NOUL_ROWS,
        ),
        _ => (
            MIN_BINS as usize + RESERVED_NOUL_ROWS,
            MAX_BINS as usize + RESERVED_NOUL_ROWS,
        ),
    }
}

/// `letter_distributions`' softmax, then the top row (numpy's first-max `argmax`), its
/// probability, and `eval_harness._entropy`. Every sum is numpy's pairwise sum.
fn distribution(logits: &[f64]) -> (Vec<f64>, usize, f64, f64) {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
    let total = pairwise_sum(&z);
    let probs: Vec<f64> = z.iter().map(|v| v / total).collect();
    let mut argmax = 0usize;
    for (i, p) in probs.iter().enumerate() {
        if *p > probs[argmax] {
            argmax = i;
        }
    }
    let terms: Vec<f64> = probs
        .iter()
        .map(|p| {
            let c = p.clamp(ENTROPY_CLIP, 1.0);
            c * c.ln()
        })
        .collect();
    let entropy = -pairwise_sum(&terms);
    let confidence = probs[argmax];
    (probs, argmax, confidence, entropy)
}

/// The second pass on a letter line: all three fields or none.
fn second_pass(line: &Map<String, Value>, rows: usize, at: &str) -> Result<Option<Second>> {
    let present = ["perm", "top_permuted", "permutation_agreed"]
        .iter()
        .filter(|k| line.contains_key(**k))
        .count();
    if present == 0 {
        return Ok(None);
    }
    ensure!(
        present == 3,
        "{at}: carries {present} of perm, top_permuted and permutation_agreed; \
         annotate_second_pass writes all three or none"
    );
    let options = rows - RESERVED_NOUL_ROWS;
    let perm: Vec<usize> = line
        .get("perm")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{at}: `perm` is not an array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| format!("{at}: `perm` holds a non-index"))
        })
        .collect::<Result<_>>()?;
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    ensure!(
        sorted == (0..options).collect::<Vec<_>>(),
        "{at}: perm {perm:?} is not a permutation of the {options} options"
    );
    let top_permuted = index_field(line, "top_permuted", at)?;
    ensure!(
        top_permuted < rows,
        "{at}: top_permuted {top_permuted} is outside {rows} rows"
    );
    Ok(Some(Second {
        perm,
        top_permuted,
        agreed: bool_field(line, "permutation_agreed", at)?,
    }))
}

fn letter_line(line: &Map<String, Value>, kind: SlotKind, at: &str) -> Result<Letter> {
    let rows = index_field(line, "rows", at)?;
    let (lo, hi) = legal_rows(kind);
    ensure!(
        (lo..=hi).contains(&rows),
        "{at}: a {} slot with {rows} decode rows is outside the schema's {lo}..={hi}",
        kind.as_str()
    );
    let noul_row = index_field(line, "noul_row", at)?;
    ensure!(
        noul_row == rows - RESERVED_NOUL_ROWS,
        "{at}: noul_row {noul_row} is not the last of {rows} rows"
    );
    let gold = index_field(line, "gold_row", at)?;
    ensure!(gold < rows, "{at}: gold_row {gold} is outside {rows} rows");
    let top = index_field(line, "top", at)?;
    ensure!(top < rows, "{at}: top {top} is outside {rows} rows");
    let values = line
        .get("row_logits")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            format!("{at}: no `row_logits`: written by a scoring run older than 444bedc")
        })?;
    ensure!(
        values.len() == rows,
        "{at}: {} logits for {rows} decode rows",
        values.len()
    );
    let logits: Vec<f64> = values
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| format!("{at}: `row_logits` holds a non-finite or non-number value"))
        })
        .collect::<Result<_>>()?;
    let (probs, argmax, confidence, entropy) = distribution(&logits);
    Ok(Letter {
        row_id: str_field(line, "row_id", at)?.to_string(),
        family: label_field(line, "family_id", at)?,
        language: label_field(line, "language", at)?,
        kind,
        rows,
        noul_row,
        gold,
        top,
        correct: bool_field(line, "correct", at)?,
        expected_abstain: bool_field(line, "expected_abstain", at)?,
        second: second_pass(line, rows, at)?,
        probs,
        argmax,
        confidence,
        entropy,
    })
}

fn read_verdicts(path: &Path) -> Result<VerdictFile> {
    let shown = path.display().to_string();
    let (bytes, sha256) = read_bounded(path, MAX_INPUT_BYTES)?;
    let mut file = VerdictFile {
        path: shown.clone(),
        sha256,
        eval_row_id: String::new(),
        seed: 0,
        lines: 0,
        letters: Vec::new(),
        spans: Vec::new(),
    };
    let mut seeds: BTreeSet<i64> = BTreeSet::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for (at, line) in json_lines(&bytes, &shown)? {
        let eval_row_id = str_field(&line, "eval_row_id", &at)?;
        if file.eval_row_id.is_empty() {
            file.eval_row_id = eval_row_id.to_string();
        }
        ensure!(
            file.eval_row_id == eval_row_id,
            "{at}: eval row {eval_row_id}, but the file began with {}; one file is one eval \
             row's verdicts",
            file.eval_row_id
        );
        seeds.insert(
            line.get("seed")
                .and_then(Value::as_i64)
                .ok_or_else(|| format!("{at}: no integer `seed`"))?,
        );
        let row_id = str_field(&line, "row_id", &at)?;
        let slot_name = str_field(&line, "slot_name", &at)?;
        ensure!(
            seen.insert((row_id.to_string(), slot_name.to_string())),
            "{at}: row {row_id} slot {slot_name} appears twice"
        );
        match str_field(&line, "kind", &at)? {
            "choice" => file
                .letters
                .push(letter_line(&line, SlotKind::Choice, &at)?),
            "score" => file.letters.push(letter_line(&line, SlotKind::Score, &at)?),
            "span" => file.spans.push(SpanLine {
                family: label_field(&line, "family_id", &at)?,
                correct: bool_field(&line, "correct", &at)?,
            }),
            other => return Err(format!("{at}: unknown kind {other:?}")),
        }
        file.lines += 1;
    }
    ensure!(file.lines > 0, "{shown}: no verdict lines");
    ensure!(
        seeds.len() == 1,
        "{shown}: lines carry seeds {seeds:?}; one eval row is one seed"
    );
    file.seed = *seeds.iter().next().unwrap_or(&0);
    let with_second = file.letters.iter().filter(|l| l.second.is_some()).count();
    let choice = file
        .letters
        .iter()
        .filter(|l| matches!(l.kind, SlotKind::Choice))
        .count();
    ensure!(
        with_second == 0 || with_second == choice,
        "{shown}: {with_second} of {choice} choice rows carry a second pass; every choice row \
         has two or more options and so a derangement, so a row without one was decoded in the \
         first pass and not the second"
    );
    Ok(file)
}

/// Every row of every ledger whose `row_id` is `wanted`, refused unless exactly one.
fn find_eval_row(ledgers: &[(String, Vec<u8>)], wanted: &str) -> Result<EvalRow> {
    let mut found = Vec::new();
    for (shown, bytes) in ledgers {
        for (_, row) in json_lines(bytes, shown)? {
            if row.get("row_id").and_then(Value::as_str) == Some(wanted) {
                found.push(EvalRow {
                    ledger: shown.clone(),
                    row,
                });
            }
        }
    }
    ensure!(
        found.len() == 1,
        "eval row {wanted} appears {} time(s) in the --eval-ledger files, not once",
        found.len()
    );
    let row = found.remove(0);
    ensure!(
        row.row.get("run_kind").and_then(Value::as_str) == Some("eval"),
        "row {wanted} in {} is not an eval row",
        row.ledger
    );
    Ok(row)
}

fn read_suite(paths: &[PathBuf]) -> Result<SuiteRead> {
    let mut files = Vec::new();
    let mut by_row: BTreeMap<String, Vec<SuiteLine>> = BTreeMap::new();
    for path in paths {
        let shown = path.display().to_string();
        let (bytes, sha256) = read_bounded(path, MAX_INPUT_BYTES)?;
        let mut lines = 0usize;
        for (at, line) in json_lines(&bytes, &shown)? {
            let eval_row_id = str_field(&line, "eval_row_id", &at)?.to_string();
            let parsed = match str_field(&line, "gate", &at)? {
                "needle_hunk_recall" => SuiteLine::Needle {
                    bucket: str_field(&line, "depth_bucket", &at)?.to_string(),
                    hit: bool_field(&line, "hit", &at)?,
                },
                "ood_abstain" => SuiteLine::Ood {
                    category: str_field(&line, "category", &at)?.to_string(),
                    abstained: bool_field(&line, "abstained", &at)?,
                },
                other => return Err(format!("{at}: unknown suite gate {other:?}")),
            };
            by_row.entry(eval_row_id).or_default().push(parsed);
            lines += 1;
        }
        files.push(json!({"path": shown, "sha256": sha256, "lines": lines}));
    }
    Ok((files, by_row))
}

/// The human decisions record: the population as stated, and the questions still open.
fn read_decisions(path: &Path) -> Result<(Value, Vec<Value>)> {
    let shown = path.display().to_string();
    let (bytes, sha256) = read_bounded(path, MAX_DECISIONS_BYTES)?;
    let record: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{shown}: not JSON ({e})"))?;
    ensure!(
        record.get("schema").and_then(Value::as_str) == Some("qd.promotion-decisions.v1"),
        "{shown}: schema is not qd.promotion-decisions.v1"
    );
    let decisions = record
        .get("decisions")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{shown}: no `decisions` object"))?;
    let population = decisions
        .get("promotion_population")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{shown}: no `decisions.promotion_population`"))?;
    let mut stated = Map::new();
    for key in ["value", "families", "status", "gap", "source"] {
        let v = population
            .get(key)
            .ok_or_else(|| format!("{shown}: promotion_population has no `{key}`"))?;
        stated.insert(key.to_string(), v.clone());
    }
    for key in ["decided_by", "decided_on", "decision_ref"] {
        stated.insert(
            key.to_string(),
            population.get(key).cloned().unwrap_or(Value::Null),
        );
    }
    stated.insert("record".into(), json!(shown));
    stated.insert("record_sha256".into(), json!(sha256));
    let open = decisions
        .iter()
        .filter(|(_, d)| d.get("status").and_then(Value::as_str) == Some("open"))
        .map(|(name, d)| json!({"name": name, "gap": d.get("gap").cloned().unwrap_or(Value::Null)}))
        .collect();
    Ok((Value::Object(stated), open))
}

fn ran(value: Value, n: usize, n_total: usize, detail: impl Into<String>) -> Value {
    json!({"state": "ran", "value": value, "n": n, "n_total": n_total, "detail": detail.into()})
}

fn not_run(reason: impl Into<String>) -> Value {
    json!({"state": "not_run", "reason": reason.into()})
}

fn share(k: usize, n: usize, detail: impl Into<String>) -> Value {
    ran(json!(k as f64 / n as f64), k, n, detail)
}

fn mean(values: &[f64]) -> f64 {
    pairwise_sum(values) / values.len() as f64
}

/// An ECE as a report value: the number, how many bins carried mass, and the bar beside it.
fn ece_value(state: &EceState, what: &str) -> Value {
    match state {
        EceState::Ran {
            value,
            n,
            populated_bins,
            ..
        } => ran(
            json!(value),
            *n,
            *n,
            format!(
                "ECE {value:.4} over {n} rows, {populated_bins} of {ECE_BINS} bins carried mass \
                 (the ece gate's bar, {ECE_THRESHOLD}, stated not applied); {what}"
            ),
        ),
        EceState::NotRun { reason } => not_run(reason.clone()),
    }
}

fn ece_of(rows: &[&Letter]) -> Result<EceState> {
    let confidence: Vec<f64> = rows.iter().map(|l| l.confidence).collect();
    let correct: Vec<bool> = rows.iter().map(|l| l.argmax == l.gold).collect();
    ece_state(&confidence, &correct)
}

/// The shapes present in `rows`, in file order of first appearance.
fn shapes_of(rows: &[&Letter]) -> Vec<String> {
    let mut seen = Vec::new();
    for l in rows {
        let s = l.shape();
        if !seen.contains(&s) {
            seen.push(s);
        }
    }
    seen
}

/// Mean entropy and the top predicted class's share: `degenerate_head_check`'s two numbers.
fn head_numbers(rows: &[&Letter]) -> (f64, f64, usize) {
    let entropies: Vec<f64> = rows.iter().map(|l| l.entropy).collect();
    let width = rows[0].rows;
    let mut counts = vec![0usize; width];
    for l in rows {
        counts[l.argmax] += 1;
    }
    let (top_class, top) =
        counts.iter().enumerate().fold(
            (0usize, 0usize),
            |best, (c, k)| if *k > best.1 { (c, *k) } else { best },
        );
    (mean(&entropies), top as f64 / rows.len() as f64, top_class)
}

fn head_value(rows: &[&Letter]) -> Value {
    if rows.is_empty() {
        return not_run("no letter row of this shape");
    }
    let (entropy, top_share, top_class) = head_numbers(rows);
    let mut v = ran(
        json!(entropy),
        rows.len(),
        rows.len(),
        format!(
            "mean predictive entropy {entropy:.4} nats (floor {ENTROPY_FLOOR}, stated not \
             applied); top predicted row {top_class} takes {top_share:.3} (cap {MAX_CLASS_SHARE}, \
             stated not applied)"
        ),
    );
    v["top_predicted_share"] = json!(top_share);
    v
}

/// The runtime's choice rule, minus the calibrated margin (`choice_rule_abstentions`), keyed by
/// `row_id` with the same last-write-wins as the Python dict.
fn in_distribution(letters: &[Letter]) -> BTreeMap<String, (bool, Option<String>)> {
    let mut out: BTreeMap<String, (bool, Option<String>)> = BTreeMap::new();
    for l in letters {
        if !matches!(l.kind, SlotKind::Choice) || l.expected_abstain {
            continue;
        }
        let mut abstained = l.top == l.noul_row;
        if let Some(s) = &l.second {
            abstained =
                abstained || s.top_permuted == l.noul_row || s.perm[s.top_permuted] != l.top;
        }
        out.insert(l.row_id.clone(), (abstained, l.family.clone()));
    }
    out
}

/// What was compared with the eval row, and what could not be.
struct Checks<'a> {
    row: &'a EvalRow,
    checked: Vec<String>,
    not_checked: Vec<Value>,
}

impl<'a> Checks<'a> {
    fn new(row: &'a EvalRow) -> Self {
        Self {
            row,
            checked: Vec::new(),
            not_checked: Vec::new(),
        }
    }

    fn recorded(&self, section: &str, name: &str) -> Option<&'a Map<String, Value>> {
        self.row
            .row
            .get(section)
            .and_then(Value::as_object)
            .and_then(|m| m.get(name))
            .and_then(Value::as_object)
    }

    /// The recorded state when it ran; otherwise note why it was not compared.
    fn ran_state(&mut self, section: &str, name: &str) -> Option<&'a Map<String, Value>> {
        match self.recorded(section, name) {
            None => {
                self.not_checked.push(
                    json!({"name": format!("{section}.{name}"), "why": "absent on the eval row"}),
                );
                None
            }
            Some(r) if r.get("state").and_then(Value::as_str) != Some("ran") => {
                self.not_checked.push(json!({
                    "name": format!("{section}.{name}"),
                    "why": format!("recorded not_run: {}", r.get("reason").and_then(Value::as_str).unwrap_or("?")),
                }));
                None
            }
            Some(r) => Some(r),
        }
    }

    fn counts(&mut self, section: &str, name: &str, n: usize, n_total: usize) -> Result<()> {
        if let Some(r) = self.ran_state(section, name) {
            let rec = (
                r.get("n").and_then(Value::as_u64),
                r.get("n_total").and_then(Value::as_u64),
            );
            ensure!(
                rec == (Some(n as u64), Some(n_total as u64)),
                "{section}.{name}: the eval row records {rec:?}, the verdicts give {n}/{n_total}; \
                 these are not that row's verdicts"
            );
            self.checked.push(format!("{section}.{name}"));
        }
        Ok(())
    }

    fn value(
        &mut self,
        section: &str,
        name: &str,
        value: f64,
        n_total: Option<usize>,
    ) -> Result<()> {
        if let Some(r) = self.ran_state(section, name) {
            let rec = r.get("value").and_then(Value::as_f64);
            ensure!(
                rec.is_some_and(|x| (x - value).abs() <= VALUE_TOLERANCE),
                "{section}.{name}: the eval row records {rec:?}, the verdicts give {value}; these \
                 are not that row's verdicts"
            );
            if let Some(total) = n_total {
                let stated = r.get("n_total").and_then(Value::as_u64);
                ensure!(
                    stated == Some(total as u64),
                    "{section}.{name}: the eval row records n_total {stated:?}, the verdicts hold \
                     {total}"
                );
            }
            self.checked.push(format!("{section}.{name}"));
        }
        Ok(())
    }

    fn ece(&mut self, name: &str, state: &EceState) -> Result<()> {
        match state {
            EceState::Ran { value, n, .. } => self.value("metrics", name, *value, Some(*n)),
            EceState::NotRun { reason } => {
                if let Some(r) = self.ran_state("metrics", name) {
                    return Err(format!(
                        "metrics.{name}: the eval row records it ran ({:?}), the verdicts give \
                         not_run ({reason})",
                        r.get("value")
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Family ids of every line, sorted; `None` when any line has none (fail closed, as Python).
fn families(file: &VerdictFile) -> (Vec<String>, usize) {
    let mut set = BTreeSet::new();
    let mut missing = 0usize;
    for f in file
        .letters
        .iter()
        .map(|l| &l.family)
        .chain(file.spans.iter().map(|s| &s.family))
    {
        match f {
            Some(f) => {
                set.insert(f.clone());
            }
            None => missing += 1,
        }
    }
    (set.into_iter().collect(), missing)
}

/// Per family, or one `not_run` naming why there is no per-family breakdown at all.
fn per_family(
    fams: &[String],
    missing: usize,
    total: usize,
    mut each: impl FnMut(&str) -> Value,
) -> Value {
    if missing > 0 {
        return json!({"family": not_run(format!("{missing} of {total} verdict lines {NO_FAMILY_REASON}"))});
    }
    Value::Object(fams.iter().map(|f| (f.clone(), each(f))).collect())
}

fn pooled(population: &str, recorded: Option<&Map<String, Value>>, recomputed: Value) -> Value {
    json!({
        "population": population,
        "recorded_on_the_eval_row": recorded.map_or(Value::Null, |r| Value::Object(r.clone())),
        "recomputed_from_the_verdicts": recomputed,
    })
}

/// G3: one slot shape's diagnostics over `rows`.
fn slot_stats(rows: &[&Letter]) -> Value {
    if rows.is_empty() {
        return not_run("no row of this shape");
    }
    let n = rows.len();
    let width = rows[0].rows;
    let noul = rows[0].noul_row;
    let (entropy, top_share, top_class) = head_numbers(rows);
    let mut gold = vec![0usize; width];
    let mut pred = vec![0usize; width];
    for l in rows {
        gold[l.gold] += 1;
        pred[l.argmax] += 1;
    }
    let argmax_right = rows.iter().filter(|l| l.argmax == l.gold).count();
    let decoded_right = rows.iter().filter(|l| l.correct).count();
    let (majority_class, majority) = gold.iter().enumerate().fold(
        (0usize, 0usize),
        |best, (c, k)| if *k > best.1 { (c, *k) } else { best },
    );
    let mut classes = Vec::new();
    for c in 0..width {
        let g = gold[c] as f64 / n as f64;
        let column: Vec<f64> = rows.iter().map(|l| l.probs[c]).collect();
        let mean_p = mean(&column);
        let p_hat = pred[c] as f64 / n as f64;
        let deviation = |observed: f64| -> Value {
            if gold[c] == 0 || gold[c] == n {
                return not_run(format!(
                    "the gold share of this row is {g}: its binomial standard error \
                     sqrt(g(1-g)/n) is 0, so a deviation in standard errors is undefined"
                ));
            }
            let sigma = (g * (1.0 - g) / n as f64).sqrt();
            ran(
                json!((observed - g) / sigma),
                n,
                n,
                format!("({observed:.4} - {g:.4}) / sigma, sigma = sqrt(g(1-g)/n) = {sigma:.5}"),
            )
        };
        classes.push(json!({
            "row": c,
            "label": if c == noul { "noul".to_string() } else { format!("option {c}") },
            "gold": {"count": gold[c], "share": g},
            "predicted": {"count": pred[c], "share": p_hat, "deviation_sigma": deviation(p_hat)},
            "mean_probability": {"value": mean_p, "deviation_sigma": deviation(mean_p)},
        }));
    }
    json!({
        "n": n,
        "mean_entropy": entropy,
        "entropy_floor_stated": ENTROPY_FLOOR,
        "accuracy": share(argmax_right, n, "top row of the distribution equals the gold row"),
        "decoded_correct": share(decoded_right, n, "the verdict's own `correct`, as answer.rs decoded it"),
        "majority_class_rate": share(majority, n, format!("gold row {majority_class} is the most common gold")),
        "top_predicted_share": top_share,
        "top_predicted_row": top_class,
        "max_class_share_stated": MAX_CLASS_SHARE,
        "classes": classes,
    })
}

#[allow(clippy::too_many_lines)]
fn report_row(file: &VerdictFile, row: &EvalRow, suite: Option<&Vec<SuiteLine>>) -> Result<Value> {
    let mut checks = Checks::new(row);
    let seed = row
        .row
        .get("protocol")
        .and_then(|p| p.get("seed"))
        .and_then(Value::as_i64);
    ensure!(
        seed == Some(file.seed),
        "{}: verdict lines are seed {}, eval row {} is seed {seed:?}",
        file.path,
        file.seed,
        file.eval_row_id
    );
    // Its n_total also counts rows that were not decoded and so wrote no verdict line.
    checks.value("metrics", "val_rows_decoded", file.lines as f64, None)?;
    let (fams, missing) = families(file);
    let total_lines = file.lines;
    let letters: Vec<&Letter> = file.letters.iter().collect();
    let choice: Vec<&Letter> = letters
        .iter()
        .copied()
        .filter(|l| matches!(l.kind, SlotKind::Choice))
        .collect();

    // -- accuracy ------------------------------------------------------------------------------
    let mut acc_pooled = Map::new();
    for kind in ["choice", "score", "span"] {
        let (k, n) = if kind == "span" {
            (
                file.spans.iter().filter(|s| s.correct).count(),
                file.spans.len(),
            )
        } else {
            let rows: Vec<&&Letter> = letters.iter().filter(|l| l.kind.as_str() == kind).collect();
            (rows.iter().filter(|l| l.correct).count(), rows.len())
        };
        if n == 0 {
            acc_pooled.insert(
                kind.into(),
                not_run(format!("the verdicts hold no {kind} row")),
            );
            continue;
        }
        checks.counts("metrics", &format!("val_top1.{kind}"), k, n)?;
        acc_pooled.insert(
            kind.into(),
            share(
                k,
                n,
                format!("{kind} rows decoded to the gold (val_top1.{kind})"),
            ),
        );
    }
    let acc_family = per_family(&fams, missing, total_lines, |f| {
        let mut by = Map::new();
        for kind in ["choice", "score", "span"] {
            let (k, n) = if kind == "span" {
                let rows: Vec<&SpanLine> = file
                    .spans
                    .iter()
                    .filter(|s| s.family.as_deref() == Some(f))
                    .collect();
                (rows.iter().filter(|s| s.correct).count(), rows.len())
            } else {
                let rows: Vec<&&Letter> = letters
                    .iter()
                    .filter(|l| l.kind.as_str() == kind && l.family.as_deref() == Some(f))
                    .collect();
                (rows.iter().filter(|l| l.correct).count(), rows.len())
            };
            if n > 0 {
                by.insert(kind.into(), share(k, n, PER_FAMILY_NOTE));
            }
        }
        Value::Object(by)
    });

    // -- permutation_consistency -----------------------------------------------------------------
    let asked: Vec<&&Letter> = choice.iter().filter(|l| l.second.is_some()).collect();
    let agree = asked
        .iter()
        .filter(|l| l.second.as_ref().is_some_and(|s| s.agreed))
        .count();
    let perm_recomputed = if asked.is_empty() {
        not_run("these verdicts carry no second pass (perm / top_permuted / permutation_agreed)")
    } else {
        checks.counts("gates", "permutation_consistency", agree, asked.len())?;
        share(
            agree,
            asked.len(),
            "asked choice rows that agreed with themselves across the derangement",
        )
    };
    let mut fam_checks: Vec<(String, usize, usize)> = Vec::new();
    let perm_family = per_family(&fams, missing, total_lines, |f| {
        let rows: Vec<&&&Letter> = asked
            .iter()
            .filter(|l| l.family.as_deref() == Some(f))
            .collect();
        let in_family = choice
            .iter()
            .filter(|l| l.family.as_deref() == Some(f))
            .count();
        if asked.is_empty() {
            return not_run("these verdicts carry no second pass");
        }
        if in_family == 0 {
            return not_run("no choice row in this family: the gate reads choice rows only");
        }
        let k = rows
            .iter()
            .filter(|l| l.second.as_ref().is_some_and(|s| s.agreed))
            .count();
        fam_checks.push((f.to_string(), k, rows.len()));
        share(k, rows.len(), PER_FAMILY_NOTE)
    });
    for (f, k, n) in fam_checks.drain(..) {
        checks.counts(
            "metrics",
            &format!("permutation_consistency.family.{f}"),
            k,
            n,
        )?;
    }

    // -- ood_abstain: the in-distribution half from the val verdicts, the OOD half from the suite
    let indist = in_distribution(&file.letters);
    let in_k = indist.values().filter(|(a, _)| *a).count();
    let indist_recomputed = if asked.is_empty() {
        not_run(
            "the runtime rule's permuted half needs the second pass, which these verdicts do not \
             carry",
        )
    } else {
        checks.counts("metrics", "ood_abstain.in_distribution", in_k, indist.len())?;
        share(
            in_k,
            indist.len(),
            "val choice rows, gold not noul, the runtime rule would abstain on",
        )
    };
    let ood_family = per_family(&fams, missing, total_lines, |f| {
        let flags: Vec<bool> = indist
            .values()
            .filter(|(_, fam)| fam.as_deref() == Some(f))
            .map(|(a, _)| *a)
            .collect();
        let half = if asked.is_empty() {
            not_run("these verdicts carry no second pass")
        } else if flags.is_empty() {
            not_run("no choice row of this family is in the in-distribution bound")
        } else {
            let k = flags.iter().filter(|a| **a).count();
            fam_checks.push((f.to_string(), k, flags.len()));
            share(k, flags.len(), PER_FAMILY_NOTE)
        };
        json!({
            "in_distribution": half,
            "ood_suite": not_run("the OOD suite's cases are generated, carry a category and no val family"),
        })
    });
    for (f, k, n) in fam_checks.drain(..) {
        checks.counts(
            "metrics",
            &format!("ood_abstain.in_distribution.family.{f}"),
            k,
            n,
        )?;
    }
    let mut ood_suite = Map::new();
    let mut needle_suite = Map::new();
    if let Some(lines) = suite {
        let mut needle: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        let mut ood: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for l in lines {
            match l {
                SuiteLine::Needle { bucket, hit } => {
                    let e = needle.entry(bucket).or_default();
                    e.0 += usize::from(*hit);
                    e.1 += 1;
                }
                SuiteLine::Ood {
                    category,
                    abstained,
                } => {
                    let e = ood.entry(category).or_default();
                    e.0 += usize::from(*abstained);
                    e.1 += 1;
                }
            }
        }
        for (bucket, (k, n)) in &needle {
            checks.counts(
                "metrics",
                &format!("needle_hunk_recall.depth.{bucket}"),
                *k,
                *n,
            )?;
            needle_suite.insert(
                (*bucket).to_string(),
                share(*k, *n, "needle recall in this depth bucket"),
            );
        }
        if let Some((bucket, (k, n))) = needle
            .iter()
            .min_by(|a, b| (a.1.0 as f64 / a.1.1 as f64).total_cmp(&(b.1.0 as f64 / b.1.1 as f64)))
        {
            let total: usize = needle.values().map(|v| v.1).sum();
            checks.value(
                "gates",
                "needle_hunk_recall",
                *k as f64 / *n as f64,
                Some(total),
            )?;
            needle_suite.insert("worst_bucket".into(), json!(bucket));
        }
        let (mut ok, mut on) = (0usize, 0usize);
        for (category, (k, n)) in &ood {
            checks.counts("metrics", &format!("ood_abstain.{category}"), *k, *n)?;
            ood_suite.insert(
                (*category).to_string(),
                share(*k, *n, "OOD cases the runtime rule abstained on"),
            );
            ok += k;
            on += n;
        }
        if on > 0 {
            checks.counts("gates", "ood_abstain", ok, on)?;
        }
    }
    let suite_note = if suite.is_some() {
        "recomputed from the suite verdicts"
    } else {
        "no --suite-verdicts line names this eval row; not recomputed"
    };

    // -- ece and degenerate_head: the gate's inputs, then the per-family split ---------------------
    let mut by_shape: BTreeMap<String, Vec<&Letter>> = BTreeMap::new();
    for l in &letters {
        by_shape.entry(l.shape()).or_default().push(l);
    }
    let mut ece_inputs = Map::new();
    let mut head_inputs = Map::new();
    for (shape, rows) in &by_shape {
        let state = ece_of(rows)?;
        checks.ece(&format!("ece.{shape}"), &state)?;
        ece_inputs.insert(
            format!("ece.{shape}"),
            ece_value(&state, "one slot shape: a gate input"),
        );
        let head = head_value(rows);
        if let Some(entropy) = head["value"].as_f64() {
            checks.value(
                "metrics",
                &format!("degenerate_head.{shape}"),
                entropy,
                Some(rows.len()),
            )?;
        }
        head_inputs.insert(format!("degenerate_head.{shape}"), head);
    }
    let mut by_language: BTreeMap<String, Vec<&Letter>> = BTreeMap::new();
    for l in &letters {
        by_language
            .entry(l.language.clone().unwrap_or_else(|| "none".to_string()))
            .or_default()
            .push(l);
    }
    for (language, rows) in &by_language {
        if language == "none" {
            continue;
        }
        // The gate's per-language input refuses a language spanning slot shapes; compare only
        // where it ran, which is the one-shape case.
        if shapes_of(rows).len() == 1 {
            checks.ece(&format!("ece.lang.{language}"), &ece_of(rows)?)?;
        }
    }
    let mut ece_checks: Vec<(String, EceState)> = Vec::new();
    let ece_family = per_family(&fams, missing, total_lines, |f| {
        let mut out = Map::new();
        for (shape, rows) in &by_shape {
            let mine: Vec<&Letter> = rows
                .iter()
                .copied()
                .filter(|l| l.family.as_deref() == Some(f))
                .collect();
            if mine.is_empty() {
                continue;
            }
            match ece_of(&mine) {
                Ok(state) => {
                    out.insert(shape.clone(), ece_value(&state, PER_FAMILY_NOTE));
                    ece_checks.push((format!("ece.family.{f}.{shape}"), state));
                }
                Err(e) => {
                    out.insert(shape.clone(), not_run(e));
                }
            }
        }
        if out.is_empty() {
            return not_run("no letter row in this family: ECE reads letter distributions only");
        }
        Value::Object(out)
    });
    for (name, state) in ece_checks.drain(..) {
        checks.ece(&name, &state)?;
    }
    let mut head_checks: Vec<(String, f64, usize)> = Vec::new();
    let head_family = per_family(&fams, missing, total_lines, |f| {
        let mut out = Map::new();
        for (shape, rows) in &by_shape {
            let mine: Vec<&Letter> = rows
                .iter()
                .copied()
                .filter(|l| l.family.as_deref() == Some(f))
                .collect();
            if !mine.is_empty() {
                let mut v = head_value(&mine);
                if let Some(entropy) = v["value"].as_f64() {
                    head_checks.push((
                        format!("degenerate_head.family.{f}.{shape}"),
                        entropy,
                        mine.len(),
                    ));
                }
                v["detail"] = json!(format!(
                    "{}; {PER_FAMILY_NOTE}",
                    v["detail"].as_str().unwrap_or("")
                ));
                out.insert(shape.clone(), v);
            }
        }
        if out.is_empty() {
            return not_run(
                "no letter row in this family: the control reads letter distributions only",
            );
        }
        Value::Object(out)
    });
    // The eval row's own report-only copies (real_ft_run.family_heads), where it records them.
    for (name, entropy, n) in head_checks.drain(..) {
        checks.value("metrics", &name, entropy, Some(n))?;
    }

    // -- G2 ------------------------------------------------------------------------------------
    let pooled_ece = ece_of(&letters)?;
    // ... and real_ft_run.report_eces' copies of the pooled and per-language ECE.
    checks.ece("ece.report.pooled", &pooled_ece)?;
    let mut lang_report = Map::new();
    for (language, rows) in &by_language {
        let state = ece_of(rows)?;
        checks.ece(&format!("ece.report.lang.{language}"), &state)?;
        lang_report.insert(
            language.clone(),
            ece_value(
                &state,
                &format!(
                    "pooled across slot shapes {:?} by top-1 confidence; report-only, not the gate",
                    shapes_of(rows)
                ),
            ),
        );
    }

    // -- G3 ------------------------------------------------------------------------------------
    let mut slots = Map::new();
    for (shape, rows) in &by_shape {
        let options = rows[0].rows - RESERVED_NOUL_ROWS;
        if !SLOT_OPTIONS.contains(&options) {
            continue;
        }
        // Only the families that ask this shape: a family with no row here has no slot to report.
        let fam_stats = if missing > 0 {
            json!({"family": not_run(format!("{missing} of {total_lines} verdict lines {NO_FAMILY_REASON}"))})
        } else {
            let mut m = Map::new();
            for f in &fams {
                let mine: Vec<&Letter> = rows
                    .iter()
                    .copied()
                    .filter(|l| l.family.as_deref() == Some(f))
                    .collect();
                if !mine.is_empty() {
                    m.insert(f.clone(), slot_stats(&mine));
                }
            }
            Value::Object(m)
        };
        slots.insert(
            shape.clone(),
            json!({"all": slot_stats(rows), "per_family": fam_stats}),
        );
    }

    ensure!(
        !checks.checked.is_empty(),
        "{}: no number on eval row {} could be compared with its verdicts, so nothing shows these \
         are that row's verdicts; refused rather than reported",
        file.path,
        file.eval_row_id
    );

    let gate = |name: &str| checks.recorded("gates", name);
    let control = |name: &str| checks.recorded("controls", name);
    let supplement = |what: &str| {
        per_family(&fams, missing, total_lines, |_| {
            not_run(format!(
                "measured on a supplement row by {what}, from its own data; a verdict file holds \
                 no per-family input for it"
            ))
        })
    };
    let undefined = |gap: &str| {
        per_family(&fams, missing, total_lines, |_| {
            not_run(format!("not specified: {gap}"))
        })
    };
    let gates = json!({
        "permutation_consistency": {
            "pooled": pooled("every val family's choice rows, pooled (the gate as built)", gate("permutation_consistency"), perm_recomputed),
            "per_family": perm_family,
        },
        "ood_abstain": {
            "pooled": pooled(
                "OOD suite cases, and every val family's choice rows whose gold is not noul (the gate as built)",
                gate("ood_abstain"),
                json!({"in_distribution": indist_recomputed, "ood_suite": Value::Object(ood_suite), "note": suite_note}),
            ),
            "per_family": ood_family,
        },
        "needle_hunk_recall": {
            "pooled": pooled("the needle suite's cases", gate("needle_hunk_recall"), json!({"by_depth": Value::Object(needle_suite), "note": suite_note})),
            "per_family": per_family(&fams, missing, total_lines, |_| not_run("the needle suite's cases are generated haystacks, not val rows, and carry no val family")),
        },
        "ece": {
            "pooled": pooled("every letter row, per slot shape and per language (the gate as built)", gate("ece"), Value::Object(ece_inputs)),
            "per_family": ece_family,
        },
        "paired_margin_vs_linear": {
            "pooled": pooled("the linear control's paired rows", gate("paired_margin_vs_linear"), not_run("a supplement row's gate; not in a verdict file")),
            "per_family": supplement("tools/ft_linear_control.py"),
        },
    });
    let controls = json!({
        "degenerate_head": {
            "pooled": pooled("every letter row, per slot shape (the control as built)", control("degenerate_head"), Value::Object(head_inputs)),
            "per_family": head_family,
        },
        "shuffled_label": {
            "pooled": pooled("the shuffled-label model's val rows", control("shuffled_label"), not_run("a supplement row's control; not in a verdict file")),
            "per_family": supplement("real_ft_run.py --shuffled-label"),
        },
        "privileged_hunk": {
            "pooled": pooled("not defined", control("privileged_hunk"), not_run("not specified")),
            "per_family": undefined("GAP-PRIVILEGED-HUNK-IS-NEARLY-VACUOUS-ON-COMMITPACKFT"),
        },
        "transfer_gate": {
            "pooled": pooled("not defined", control("transfer_gate"), not_run("not specified")),
            "per_family": undefined("GAP-TRANSFER-GATE-IS-NAMED-NOT-SPECIFIED-AND-HAS-NO-DATA"),
        },
    });
    let letter_n = letters.len();
    Ok(json!({
        "eval_row_id": file.eval_row_id,
        "seed": file.seed,
        "ledger": row.ledger,
        "verdicts": {"path": file.path, "sha256": file.sha256, "lines": file.lines,
                     "letter_rows": letter_n, "span_rows": file.spans.len()},
        "families": fams,
        "cross_check": {"checked": checks.checked.len(), "names": checks.checked,
                        "not_checked": checks.not_checked},
        "g1": {
            "note": format!("pooled = the gate's population and its recorded verdict; per family is {PER_FAMILY_NOTE}"),
            "gates": gates,
            "controls": controls,
            "accuracy": {"pooled": Value::Object(acc_pooled), "per_family": acc_family},
        },
        "g2": {
            "note": "report-only: the ece gate is unchanged and still reads per shape and per language",
            "pooled_all_letter_rows": ece_value(&pooled_ece, &format!("every letter row, pooled across slot shapes {:?} by top-1 confidence; report-only, not the gate", shapes_of(&letters))),
            "by_language": Value::Object(lang_report),
        },
        "g3": {
            "note": "report-only: the degenerate_head control is unchanged; sigma is the binomial standard error under the gold marginal",
            "slots": Value::Object(slots),
        },
    }))
}

fn fmt_tri(v: &Value) -> String {
    match v.get("state").and_then(Value::as_str) {
        Some("ran") => {
            let passed = match v.get("passed").and_then(Value::as_bool) {
                Some(true) => " passed",
                Some(false) => " FAILED",
                None => "",
            };
            let coverage = match (
                v.get("n").and_then(Value::as_u64),
                v.get("n_total").and_then(Value::as_u64),
            ) {
                (Some(n), Some(t)) => format!(" ({n}/{t})"),
                _ => String::new(),
            };
            match v.get("value") {
                Some(Value::Number(x)) => {
                    format!("{:.4}{coverage}{passed}", x.as_f64().unwrap_or(f64::NAN))
                }
                Some(Value::Null) | None => format!(
                    "ran{passed}{coverage}: {}",
                    v.get("detail").and_then(Value::as_str).unwrap_or("")
                ),
                Some(other) => format!("{other}{coverage}{passed}"),
            }
        }
        Some("not_run") => format!(
            "not_run: {}",
            v.get("reason").and_then(Value::as_str).unwrap_or("?")
        ),
        _ => "absent".to_string(),
    }
}

fn render_text(report: &Value) -> String {
    let mut out = String::new();
    let pop = &report["promotion_population"];
    out.push_str(&format!(
        "promotion_population: {} [families: {}] -- status {} ({}); source: {}\n  record {} sha256 {}\n",
        pop["value"].as_str().unwrap_or("?"),
        pop["families"],
        pop["status"].as_str().unwrap_or("?"),
        pop["gap"].as_str().unwrap_or("?"),
        pop["source"].as_str().unwrap_or("?"),
        pop["record"].as_str().unwrap_or("?"),
        pop["record_sha256"].as_str().unwrap_or("?"),
    ));
    let open: Vec<String> = report["open_human_decisions"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|d| {
                    format!(
                        "{} ({})",
                        d["name"].as_str().unwrap_or("?"),
                        d["gap"].as_str().unwrap_or("?")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    out.push_str(&format!("open human decisions: {}\n", open.join("; ")));
    for row in report["eval_rows"].as_array().into_iter().flatten() {
        let v = &row["verdicts"];
        out.push_str(&format!(
            "\n=== eval row {} (seed {}) ===\nverdicts {} sha256 {}: {} lines ({} letter, {} span); ledger {}\n",
            row["eval_row_id"].as_str().unwrap_or("?"),
            row["seed"],
            v["path"].as_str().unwrap_or("?"),
            v["sha256"].as_str().unwrap_or("?"),
            v["lines"], v["letter_rows"], v["span_rows"],
            row["ledger"].as_str().unwrap_or("?"),
        ));
        let cc = &row["cross_check"];
        out.push_str(&format!(
            "cross-check: {} recorded numbers recomputed from the verdicts and equal; {} not compared\n",
            cc["checked"],
            cc["not_checked"].as_array().map_or(0, Vec::len),
        ));
        out.push_str(&format!(
            "\nG1 -- {}\n",
            row["g1"]["note"].as_str().unwrap_or("")
        ));
        for section in ["gates", "controls"] {
            for (name, body) in row["g1"][section].as_object().into_iter().flatten() {
                let pooled = &body["pooled"];
                out.push_str(&format!(
                    "  {name}\n    POOLED [{}] recorded: {}\n",
                    pooled["population"].as_str().unwrap_or("?"),
                    fmt_tri(&pooled["recorded_on_the_eval_row"]),
                ));
                render_recomputed(&mut out, &pooled["recomputed_from_the_verdicts"], "      ");
                render_families(&mut out, &body["per_family"]);
            }
        }
        out.push_str("  accuracy (val_top1)\n");
        for (kind, val) in row["g1"]["accuracy"]["pooled"]
            .as_object()
            .into_iter()
            .flatten()
        {
            out.push_str(&format!("    POOLED {kind:<6} {}\n", fmt_tri(val)));
        }
        for (fam, val) in row["g1"]["accuracy"]["per_family"]
            .as_object()
            .into_iter()
            .flatten()
        {
            render_family(&mut out, fam, val);
        }
        out.push_str(&format!(
            "\nG2 -- {}\n",
            row["g2"]["note"].as_str().unwrap_or("")
        ));
        out.push_str(&format!(
            "    pooled, every letter row      {}\n",
            fmt_tri(&row["g2"]["pooled_all_letter_rows"])
        ));
        for (lang, val) in row["g2"]["by_language"].as_object().into_iter().flatten() {
            out.push_str(&format!("    ece.lang.{lang:<20} {}\n", fmt_tri(val)));
        }
        out.push_str(&format!(
            "\nG3 -- {}\n",
            row["g3"]["note"].as_str().unwrap_or("")
        ));
        for (shape, body) in row["g3"]["slots"].as_object().into_iter().flatten() {
            render_slot(&mut out, &format!("{shape} all families"), &body["all"]);
            for (fam, stats) in body["per_family"].as_object().into_iter().flatten() {
                render_slot(&mut out, &format!("{shape} {fam}"), stats);
            }
        }
    }
    out
}

fn render_recomputed(out: &mut String, v: &Value, indent: &str) {
    if v.get("state").is_some() {
        out.push_str(&format!("{indent}recomputed: {}\n", fmt_tri(v)));
        return;
    }
    for (k, val) in v.as_object().into_iter().flatten() {
        if val.get("state").is_some() {
            out.push_str(&format!("{indent}recomputed {k}: {}\n", fmt_tri(val)));
        } else if let Some(s) = val.as_str() {
            out.push_str(&format!("{indent}{k}: {s}\n"));
        } else {
            render_recomputed(out, val, &format!("{indent}  {k} "));
        }
    }
}

/// Every family's line; one line when every family is `not_run` for the same reason.
fn render_families(out: &mut String, per_family: &Value) {
    let Some(map) = per_family.as_object() else {
        return;
    };
    let reasons: BTreeSet<Option<&str>> = map
        .values()
        .map(|v| match v.get("state").and_then(Value::as_str) {
            Some("not_run") => v.get("reason").and_then(Value::as_str),
            _ => None,
        })
        .collect();
    if map.len() > 1
        && reasons.len() == 1
        && let Some(Some(reason)) = reasons.iter().next()
    {
        out.push_str(&format!(
            "    every family ({})  not_run: {reason}\n",
            map.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
        return;
    }
    for (fam, val) in map {
        render_family(out, fam, val);
    }
}

fn render_family(out: &mut String, fam: &str, val: &Value) {
    if val.get("state").is_some() {
        out.push_str(&format!("    family {fam:<30} {}\n", fmt_tri(val)));
        return;
    }
    for (k, inner) in val.as_object().into_iter().flatten() {
        out.push_str(&format!(
            "    family {fam:<30} {k:<16} {}\n",
            fmt_tri(inner)
        ));
    }
}

fn render_slot(out: &mut String, label: &str, s: &Value) {
    if s.get("state").is_some() {
        out.push_str(&format!("  {label}: {}\n", fmt_tri(s)));
        return;
    }
    out.push_str(&format!(
        "  {label} (n={}): mean entropy {:.4} nats [floor {} stated]; accuracy {}; decoded correct {}; \
         majority-class rate {}; top predicted row {} at {:.4} [cap {} stated]\n",
        s["n"],
        s["mean_entropy"].as_f64().unwrap_or(f64::NAN),
        ENTROPY_FLOOR,
        fmt_tri(&s["accuracy"]),
        fmt_tri(&s["decoded_correct"]),
        fmt_tri(&s["majority_class_rate"]),
        s["top_predicted_row"],
        s["top_predicted_share"].as_f64().unwrap_or(f64::NAN),
        MAX_CLASS_SHARE,
    ));
    out.push_str("      row        gold share  predicted share (sigma)        mean prob (sigma)\n");
    for c in s["classes"].as_array().into_iter().flatten() {
        let sigma = |v: &Value| match v.get("state").and_then(Value::as_str) {
            Some("ran") => format!("{:+.2}", v["value"].as_f64().unwrap_or(f64::NAN)),
            _ => "undefined".to_string(),
        };
        out.push_str(&format!(
            "      {:<10} {:.4}      {:.4} ({:>9})           {:.4} ({:>9})\n",
            c["label"].as_str().unwrap_or("?"),
            c["gold"]["share"].as_f64().unwrap_or(f64::NAN),
            c["predicted"]["share"].as_f64().unwrap_or(f64::NAN),
            sigma(&c["predicted"]["deviation_sigma"]),
            c["mean_probability"]["value"].as_f64().unwrap_or(f64::NAN),
            sigma(&c["mean_probability"]["deviation_sigma"]),
        ));
    }
}

fn run(args: &Args) -> Result<Value> {
    ensure!(
        args.verdicts.len() <= MAX_FILES
            && args.suite_verdicts.len() <= MAX_FILES
            && args.eval_ledger.len() <= MAX_FILES,
        "at most {MAX_FILES} files of each kind"
    );
    if let Some(out) = &args.out_json {
        ensure!(!out.exists(), "--out-json {} already exists", out.display());
    }
    let (population, open) = read_decisions(&args.decisions)?;
    let mut ledgers = Vec::new();
    for path in &args.eval_ledger {
        let (bytes, _) = read_bounded(path, MAX_LEDGER_BYTES)?;
        ledgers.push((path.display().to_string(), bytes));
    }
    let (suite_files, suite) = read_suite(&args.suite_verdicts)?;
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for path in &args.verdicts {
        let file = read_verdicts(path)?;
        ensure!(
            seen.insert(file.eval_row_id.clone()),
            "{}: eval row {} was already given by another --verdicts file",
            file.path,
            file.eval_row_id
        );
        let row = find_eval_row(&ledgers, &file.eval_row_id)?;
        rows.push(report_row(&file, &row, suite.get(&file.eval_row_id))?);
    }
    Ok(json!({
        "tool": "qd-gate-report",
        "report_only": "nothing here is a gate, moves a threshold or changes a population (CLAUDE.md rule 2)",
        "promotion_population": population,
        "open_human_decisions": open,
        "suite_verdicts": suite_files,
        "eval_rows": rows,
    }))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let result = run(&args).and_then(|report| {
        if let Some(path) = &args.out_json {
            let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
            write_new(path, &bytes)?;
        }
        Ok(report)
    });
    match result {
        Ok(report) => {
            print!("{}", render_text(&report));
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("qd-gate-report: refused: {e}");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letter(logits: &[f64], gold: usize, family: &str) -> Letter {
        let (probs, argmax, confidence, entropy) = distribution(logits);
        Letter {
            row_id: format!("r{gold}{}", logits.len()),
            family: Some(family.to_string()),
            language: None,
            kind: SlotKind::Choice,
            rows: logits.len(),
            noul_row: logits.len() - 1,
            gold,
            top: argmax,
            correct: argmax == gold,
            expected_abstain: false,
            second: None,
            probs,
            argmax,
            confidence,
            entropy,
        }
    }

    #[test]
    fn the_softmax_takes_numpys_first_max_and_entropy_is_clipped() {
        let (p, argmax, conf, h) = distribution(&[2.0, 2.0, 0.0]);
        assert_eq!(
            argmax, 0,
            "numpy's argmax takes the first of two equal maxima"
        );
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert_eq!(conf, p[0]);
        assert!(h > 0.0);
        // A one-hot distribution is clipped at 1e-12, not log(0).
        let (_, _, _, h) = distribution(&[0.0, -1e4, -1e4]);
        assert!(h.is_finite() && (0.0..1e-9).contains(&h), "{h}");
    }

    #[test]
    fn a_deviation_is_undefined_where_the_gold_share_is_zero_or_one() {
        let rows: Vec<Letter> = (0..10).map(|_| letter(&[3.0, 0.0, -5.0], 0, "f")).collect();
        let refs: Vec<&Letter> = rows.iter().collect();
        let s = slot_stats(&refs);
        let classes = s["classes"].as_array().unwrap();
        assert_eq!(
            classes[0]["predicted"]["deviation_sigma"]["state"],
            "not_run"
        );
        assert_eq!(classes[2]["label"], "noul");
        assert_eq!(s["majority_class_rate"]["value"], 1.0);
        assert!(
            s.get("passed").is_none(),
            "a report value never says passed"
        );
    }

    #[test]
    fn a_deviation_is_in_standard_errors_of_the_gold_marginal() {
        // Gold is row 0 on 5 of 10 rows; the head predicts row 0 on all 10.
        let mut rows: Vec<Letter> = (0..5).map(|_| letter(&[3.0, 0.0, -5.0], 0, "f")).collect();
        rows.extend((0..5).map(|_| letter(&[3.0, 0.0, -5.0], 1, "f")));
        let refs: Vec<&Letter> = rows.iter().collect();
        let s = slot_stats(&refs);
        let z = s["classes"][0]["predicted"]["deviation_sigma"]["value"]
            .as_f64()
            .unwrap();
        let sigma = (0.5f64 * 0.5 / 10.0).sqrt();
        assert!((z - 0.5 / sigma).abs() < 1e-12, "{z}");
        assert_eq!(s["accuracy"]["n"], 5);
    }

    #[test]
    fn the_in_distribution_rule_is_the_runtimes_and_skips_gold_noul_rows() {
        let mut a = letter(&[3.0, 0.0, -5.0], 0, "f");
        a.second = Some(Second {
            perm: vec![1, 0],
            top_permuted: 1,
            agreed: true,
        });
        let mut b = letter(&[3.0, 0.0, -5.0], 0, "f");
        b.row_id = "b".into();
        b.second = Some(Second {
            perm: vec![1, 0],
            top_permuted: 0,
            agreed: false,
        });
        let mut c = letter(&[3.0, 0.0, -5.0], 2, "f");
        c.row_id = "c".into();
        c.expected_abstain = true;
        let mut d = letter(&[-5.0, 0.0, 3.0], 0, "f");
        d.row_id = "d".into();
        d.second = Some(Second {
            perm: vec![1, 0],
            top_permuted: 2,
            agreed: true,
        });
        let got = in_distribution(&[a.clone(), b, c, d]);
        assert_eq!(got.len(), 3, "the gold-noul row is outside the bound");
        assert!(
            !got[&a.row_id].0,
            "perm[top2] maps back to top1: no abstention"
        );
        assert!(
            got["b"].0,
            "the permuted winner maps to another option: abstain"
        );
        assert!(got["d"].0, "top is noul: abstain");
    }

    #[test]
    fn a_partial_second_pass_is_refused() {
        let mut line = Map::new();
        line.insert("perm".into(), json!([1, 0]));
        assert!(
            second_pass(&line, 3, "x")
                .unwrap_err()
                .contains("1 of perm")
        );
        line.insert("top_permuted".into(), json!(0));
        line.insert("permutation_agreed".into(), json!(true));
        assert!(second_pass(&line, 3, "x").unwrap().is_some());
        line.insert("perm".into(), json!([0, 0]));
        assert!(
            second_pass(&line, 3, "x")
                .unwrap_err()
                .contains("not a permutation")
        );
    }
}
