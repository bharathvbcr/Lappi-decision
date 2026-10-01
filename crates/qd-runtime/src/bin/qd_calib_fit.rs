//! `qd-calib-fit` — fit the runtime's calibration table from scoring runs' verdict files.
//!
//! Phase 6 piece (ii), `GAP-RT-CALIBRATION-NOT-FITTED`. Reads one or more `--verdicts-out` JSONL
//! files (`tools/real_ft_run.py`), groups the letter rows by the table's key `{kind}:{rows}`, and
//! per entry fits a temperature, a `noul_margin` and a conformal cutoff with
//! [`qd_runtime::calibration_fit`] — the Rust owner of `python/qd_train/calibration_fit.py`,
//! bit-exact against it (`python/tests/test_calib_fit_parity.py`). Every probability, margin and
//! set is read through [`qd_runtime::calibration::calibrate`], the function `answer.rs` serves
//! with. The table is built as a [`CalibrationTable`] — the runtime's own type — and written as
//! `serde_json::to_vec` of it, so the file's sha256 **is** `CalibrationTable::hash()`.
//!
//! # Population is an input
//!
//! `--population all` fits each entry on every row and scores it on the same rows (in-sample).
//! `--population two-fold --split-key K` splits rows by `sha256(K || 0x00 || row_id)[0]` parity
//! (the `qd-margin-probe` construction, so one key gives both tools the same halves), fits each
//! fold, and scores every row with the parameters of the fold it was **not** fitted on. It writes
//! both fold tables and no pooled one. Which table ships is the human's call (CLAUDE.md rule 2);
//! this tool writes and reports, it installs nothing.
//!
//! # What it does not fit
//!
//! The span entry. A span verdict line carries the two pointer heads' winners and no
//! distribution (`_decode` writes `row_logits` for letter rows only), so there is nothing to fit
//! it on. The table's `span` is `null`, and a span request against it refuses with
//! `calibration_entry_missing` — `GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::calibration_fit::{
    DEFAULT_ALPHA, DEFAULT_TARGET_PRECISION, ECE_BINS, ECE_MIN_SAMPLES, ECE_THRESHOLD, EceState,
    EntryFit, LetterRows, RowScore, TEMPERATURE_HI, TEMPERATURE_ITERS, TEMPERATURE_LO, ece_state,
    fit_entry, score_rows, temperature_only,
};
use qd_runtime::schema::{
    MAX_BINS, MAX_OPTIONS, MIN_BINS, MIN_OPTIONS, RESERVED_NOUL_ROWS, SlotKind,
};
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

/// A verdict file larger than this is not one a scoring run wrote (J1's val file, ~22k lines
/// with logits, is tens of MB); refused rather than read into memory.
const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;
/// More verdict files than this in one fit is not a fit of one model.
const MAX_INPUTS: usize = 64;
/// A table name is what a hash pins; it is a name, not a document.
const MAX_NAME_BYTES: usize = 200;
/// The gap the span caveat cites.
const SPAN_GAP: &str = "GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS";

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Population {
    /// Fit and score every row of an entry together.
    All,
    /// Fit each keyed half; score every row with the other half's parameters.
    TwoFold,
}

impl Population {
    fn as_str(self) -> &'static str {
        match self {
            Population::All => "all",
            Population::TwoFold => "two-fold",
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Fit the runtime's calibration table from --verdicts-out files")]
struct Args {
    /// A `--verdicts-out` JSONL file of one eval row; repeat for more. Letter lines must carry
    /// `row_logits`.
    #[arg(long = "verdicts", required = true)]
    verdicts: Vec<PathBuf>,
    /// Which rows each entry is fitted on: `all`, or `two-fold` (needs `--split-key`).
    #[arg(long, value_enum)]
    population: Population,
    /// Key of the two-fold split: fold A iff `sha256(key || 0x00 || row_id)[0]` is even.
    #[arg(long)]
    split_key: Option<String>,
    /// The table's name (fold tables get `.fold-a` / `.fold-b`). Hashed with the table.
    #[arg(long)]
    name: String,
    /// Split-conformal miscoverage: the fitted sets target `1 - alpha` coverage.
    #[arg(long, default_value_t = DEFAULT_ALPHA)]
    alpha: f64,
    /// Retained-set precision the fitted `noul_margin` must reach.
    #[arg(long, default_value_t = DEFAULT_TARGET_PRECISION)]
    target_precision: f64,
    /// Output directory, created by this run; refused if it exists. Written whole or not at all.
    #[arg(long)]
    out_dir: PathBuf,
}

/// One letter verdict line, checked.
#[derive(Debug, Clone)]
struct LetterLine {
    eval_row_id: String,
    row_id: String,
    slot_name: String,
    kind: SlotKind,
    rows: usize,
    gold: usize,
    verdict_top: usize,
    logits: Vec<f32>,
}

/// One verdict file's identity and counts, as the eval row it came from states them.
#[derive(Debug, Clone)]
struct InputSummary {
    path: String,
    sha256: String,
    eval_row_id: String,
    lines: usize,
    /// kind -> (rows, rows whose verdict says `correct`)
    by_kind: BTreeMap<String, (usize, usize)>,
    seeds: BTreeSet<i64>,
}

/// The rows of one table entry, in file order, with what identifies each.
#[derive(Debug, Clone)]
struct Group {
    kind: SlotKind,
    rows: LetterRows,
    lines: Vec<LetterLine>,
}

impl Group {
    fn key(&self) -> String {
        format!("{}:{}", self.kind.as_str(), self.rows.width())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Fold {
    A,
    B,
}

impl Fold {
    fn as_str(self) -> &'static str {
        match self {
            Fold::A => "a",
            Fold::B => "b",
        }
    }

    fn other(self) -> Fold {
        match self {
            Fold::A => Fold::B,
            Fold::B => Fold::A,
        }
    }
}

/// `qd-margin-probe`'s split: fold A iff `sha256(key || 0x00 || row_id)[0]` is even. By `row_id`
/// alone, so a row's slots, and the same row in several files, always share a fold.
fn fold_of(key: &str, row_id: &str) -> Fold {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    h.update([0u8]);
    h.update(row_id.as_bytes());
    if h.finalize()[0] % 2 == 0 {
        Fold::A
    } else {
        Fold::B
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn str_field<'a>(line: &'a Value, key: &str, at: &str) -> Result<&'a str> {
    match line.get(key).and_then(Value::as_str) {
        Some(s) if !s.is_empty() => Ok(s),
        _ => Err(format!("{at}: no non-empty string `{key}`")),
    }
}

fn index_field(line: &Value, key: &str, at: &str) -> Result<usize> {
    line.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("{at}: `{key}` is not a non-negative integer"))
}

/// The legal decode-row counts of a letter slot: its named options or bins plus `noul`.
fn legal_rows(kind: SlotKind) -> (usize, usize) {
    match kind {
        SlotKind::Choice => (
            MIN_OPTIONS + RESERVED_NOUL_ROWS,
            MAX_OPTIONS + RESERVED_NOUL_ROWS,
        ),
        // `MIN_BINS`/`MAX_BINS` are `u32` in schema.rs; both are small.
        _ => (
            MIN_BINS as usize + RESERVED_NOUL_ROWS,
            MAX_BINS as usize + RESERVED_NOUL_ROWS,
        ),
    }
}

/// A letter line, refused on anything the runtime could not have produced.
fn letter_line(line: &Value, kind: SlotKind, at: &str) -> Result<LetterLine> {
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
        "{at}: noul_row {noul_row} is not the last of {rows} rows, which is where the runtime \
         puts it"
    );
    let gold = index_field(line, "gold_row", at)?;
    ensure!(gold < rows, "{at}: gold_row {gold} is outside {rows} rows");
    let verdict_top = index_field(line, "top", at)?;
    ensure!(
        verdict_top < rows,
        "{at}: top {verdict_top} is outside {rows} rows"
    );
    let values = line
        .get("row_logits")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            format!("{at}: no `row_logits` -- written by a scoring run older than 444bedc")
        })?;
    ensure!(
        values.len() == rows,
        "{at}: {} logits for {rows} decode rows",
        values.len()
    );
    let mut logits = Vec::with_capacity(rows);
    for v in values {
        // `calibrate` takes f32. A value that is not a number, or overflows f32, is refused.
        let x = v
            .as_f64()
            .map(|x| x as f32)
            .filter(|x| x.is_finite())
            .ok_or_else(|| format!("{at}: `row_logits` holds a non-finite or non-number value"))?;
        logits.push(x);
    }
    Ok(LetterLine {
        eval_row_id: str_field(line, "eval_row_id", at)?.to_string(),
        row_id: str_field(line, "row_id", at)?.to_string(),
        slot_name: str_field(line, "slot_name", at)?.to_string(),
        kind,
        rows,
        gold,
        verdict_top,
        logits,
    })
}

/// Read every input: hash the bytes once and parse those same bytes.
fn read_inputs(paths: &[PathBuf]) -> Result<(Vec<InputSummary>, Vec<LetterLine>, usize)> {
    ensure!(!paths.is_empty(), "no --verdicts file");
    ensure!(
        paths.len() <= MAX_INPUTS,
        "{} --verdicts files; at most {MAX_INPUTS}",
        paths.len()
    );
    let mut summaries = Vec::new();
    let mut letters = Vec::new();
    let mut span_rows = 0usize;
    let mut seen: BTreeSet<(String, String, String)> = BTreeSet::new();
    for path in paths {
        let shown = path.display().to_string();
        let meta = std::fs::metadata(path).map_err(|e| format!("{shown}: {e}"))?;
        ensure!(
            meta.len() <= MAX_INPUT_BYTES,
            "{shown}: {} bytes is not a verdict file (cap {MAX_INPUT_BYTES})",
            meta.len()
        );
        let mut bytes = Vec::with_capacity(meta.len() as usize);
        std::fs::File::open(path)
            .and_then(|f| f.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes))
            .map_err(|e| format!("{shown}: {e}"))?;
        ensure!(
            bytes.len() as u64 <= MAX_INPUT_BYTES,
            "{shown}: grew past {MAX_INPUT_BYTES} bytes while it was read"
        );
        let sha256 = hex(&Sha256::digest(&bytes));
        let mut summary = InputSummary {
            path: shown.clone(),
            sha256,
            eval_row_id: String::new(),
            lines: 0,
            by_kind: BTreeMap::new(),
            seeds: BTreeSet::new(),
        };
        for (i, raw) in bytes.split(|b| *b == b'\n').enumerate() {
            let at = format!("{shown} line {}", i + 1);
            if raw.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let line: Value = serde_json::from_slice(raw).map_err(|e| {
                format!(
                    "{at}: not JSON ({e}); a NaN or Infinity token is a non-finite logit, which \
                     is refused"
                )
            })?;
            ensure!(line.is_object(), "{at}: not a JSON object");
            let eval_row_id = str_field(&line, "eval_row_id", &at)?;
            if summary.eval_row_id.is_empty() {
                summary.eval_row_id = eval_row_id.to_string();
            }
            ensure!(
                summary.eval_row_id == eval_row_id,
                "{at}: eval row {eval_row_id}, but the file began with {}; one file is one eval \
                 row's verdicts",
                summary.eval_row_id
            );
            let row_id = str_field(&line, "row_id", &at)?;
            let slot_name = str_field(&line, "slot_name", &at)?;
            let kind_name = str_field(&line, "kind", &at)?;
            let correct = line
                .get("correct")
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("{at}: no boolean `correct`"))?;
            if let Some(seed) = line.get("seed").and_then(Value::as_i64) {
                summary.seeds.insert(seed);
            }
            ensure!(
                seen.insert((
                    eval_row_id.to_string(),
                    row_id.to_string(),
                    slot_name.to_string()
                )),
                "{at}: row {row_id} slot {slot_name} of eval row {eval_row_id} appears twice"
            );
            let kind = match kind_name {
                "choice" => Some(SlotKind::Choice),
                "score" => Some(SlotKind::Score),
                "span" => None,
                other => return Err(format!("{at}: unknown kind {other:?}")),
            };
            match kind {
                Some(kind) => letters.push(letter_line(&line, kind, &at)?),
                None => span_rows += 1,
            }
            let counts = summary.by_kind.entry(kind_name.to_string()).or_default();
            counts.0 += 1;
            counts.1 += usize::from(correct);
            summary.lines += 1;
        }
        ensure!(summary.lines > 0, "{shown}: no verdict lines");
        summaries.push(summary);
    }
    Ok((summaries, letters, span_rows))
}

/// Group letter lines by table key, keeping file order inside each group.
fn group(letters: Vec<LetterLine>) -> Result<Vec<Group>> {
    let mut groups: BTreeMap<(&'static str, usize), Group> = BTreeMap::new();
    for line in letters {
        let key = (line.kind.as_str(), line.rows);
        let group = match groups.entry(key) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => e.insert(Group {
                kind: line.kind,
                rows: LetterRows::new(line.rows)?,
                lines: Vec::new(),
            }),
        };
        group
            .rows
            .push(&line.logits, line.gold)
            .map_err(|e| format!("row {} slot {}: {e}", line.row_id, line.slot_name))?;
        group.lines.push(line);
    }
    Ok(groups.into_values().collect())
}

fn ece_json(state: &EceState) -> Value {
    match state {
        EceState::Ran {
            value,
            n,
            populated_bins,
            passed,
        } => json!({
            "state": "ran", "value": value, "n": n, "populated_bins": populated_bins,
            "passed": passed, "threshold": ECE_THRESHOLD, "bins": ECE_BINS,
        }),
        EceState::NotRun { reason } => json!({"state": "not_run", "reason": reason}),
    }
}

fn fit_json(fit: &EntryFit) -> Value {
    json!({
        "n": fit.n,
        "temperature": fit.temperature.temperature,
        "temperature_at_lower_bound": fit.temperature.at_lower_bound,
        "temperature_at_upper_bound": fit.temperature.at_upper_bound,
        "nonconformity_quantile": fit.nonconformity_quantile,
        "conformal_quantile": fit.conformal_quantile,
        "noul_margin": fit.noul_margin,
    })
}

/// What scoring rows with fitted parameters measured: the abstain rule's margin half and noul-row
/// half, the conformal set, and ECE. In group order, so ECE's bin means sum in that order.
fn summarise(
    scored: &[Option<(RowScore, CalibrationEntry)>],
    noul_row: usize,
    how: &str,
) -> Result<Value> {
    let rows: Vec<&(RowScore, CalibrationEntry)> = scored.iter().flatten().collect();
    let n = rows.len();
    let confidence: Vec<f64> = rows.iter().map(|(s, _)| s.confidence).collect();
    let correct: Vec<bool> = rows.iter().map(|(s, _)| s.correct).collect();
    let by_margin = rows
        .iter()
        .filter(|(s, e)| s.margin < e.noul_margin)
        .count();
    let by_rule_or_margin = rows
        .iter()
        .filter(|(s, e)| s.top == noul_row || s.margin < e.noul_margin)
        .count();
    let retained: Vec<&&(RowScore, CalibrationEntry)> = rows
        .iter()
        .filter(|(s, e)| s.margin >= e.noul_margin)
        .collect();
    let retained_correct = retained.iter().filter(|(s, _)| s.correct).count();
    let covered = rows.iter().filter(|(s, _)| s.gold_in_set).count();
    let set_sizes: usize = rows.iter().map(|(s, _)| s.set_size).sum();
    let ratio = |num: usize, den: usize| -> Value {
        if den == 0 {
            Value::Null
        } else {
            json!(num as f64 / den as f64)
        }
    };
    let ece = if n == 0 {
        json!({"state": "not_run", "reason": "no row of this entry could be scored"})
    } else {
        ece_json(&ece_state(&confidence, &correct)?)
    };
    Ok(json!({
        "how": how,
        "n_scored": n,
        "n_not_scored": scored.len() - n,
        "ece_after": ece,
        "abstained_by_margin": by_margin,
        "abstained_noul_row_or_margin": by_rule_or_margin,
        "retained": retained.len(),
        "retained_correct": retained_correct,
        "retained_precision": ratio(retained_correct, retained.len()),
        "covered": covered,
        "conformal_coverage": ratio(covered, n),
        "mean_set_size": ratio(set_sizes, n),
    }))
}

fn row_line(
    line: &LetterLine,
    key: &str,
    fold: Option<Fold>,
    params: Option<&str>,
    scored: Option<&(RowScore, CalibrationEntry)>,
) -> Value {
    let base = json!({
        "eval_row_id": line.eval_row_id, "row_id": line.row_id, "slot_name": line.slot_name,
        "key": key, "fold": fold.map(Fold::as_str), "params": params, "gold_row": line.gold,
    });
    let Value::Object(mut out) = base else {
        unreachable!("json! of an object literal is an object")
    };
    let fields = match scored {
        Some((s, e)) => json!({
            "scored": true, "top": s.top, "correct": s.correct, "p_gold": s.p_gold,
            "confidence": s.confidence, "margin": s.margin,
            "abstained_by_margin": s.margin < e.noul_margin,
            "abstained_noul_row_or_margin": s.top == line.rows - RESERVED_NOUL_ROWS
                || s.margin < e.noul_margin,
            "gold_in_set": s.gold_in_set, "set_size": s.set_size,
        }),
        None => json!({"scored": false}),
    };
    if let Value::Object(more) = fields {
        out.extend(more);
    }
    Value::Object(out)
}

/// A table as its file bytes: `serde_json::to_vec`, so `sha256(bytes) == table.hash()`. Read
/// back through the runtime's own deserializer before it is trusted.
fn table_bytes(table: &CalibrationTable) -> Result<Vec<u8>> {
    table.validate().map_err(|e| {
        format!(
            "the fitted table {} is not one the runtime would load: {e}",
            table.name
        )
    })?;
    let bytes = serde_json::to_vec(table).map_err(|e| e.to_string())?;
    let back: CalibrationTable = serde_json::from_slice(&bytes)
        .map_err(|e| format!("the written table does not read back: {e}"))?;
    if &back != table {
        return Err(format!(
            "the table {} read back differently from how it was written: {}",
            table.name,
            first_difference(table, &back)
        ));
    }
    ensure!(
        hex(&Sha256::digest(&bytes)) == table.hash(),
        "the file bytes of {} do not hash to CalibrationTable::hash()",
        table.name
    );
    Ok(bytes)
}

/// The first number that did not survive the trip through JSON, with both bit patterns.
fn first_difference(written: &CalibrationTable, read: &CalibrationTable) -> String {
    let (a, b) = (json!(written), json!(read));
    let (Some(la), Some(lb)) = (a["letters"].as_object(), b["letters"].as_object()) else {
        return "the letters maps differ in shape".to_string();
    };
    for (key, ea) in la {
        for field in ["temperature", "conformal_quantile", "noul_margin"] {
            let (x, y) = (
                ea[field].as_f64(),
                lb.get(key).and_then(|e| e[field].as_f64()),
            );
            if let (Some(x), Some(y)) = (x, y)
                && x.to_bits() != y.to_bits()
            {
                return format!(
                    "{key}.{field} written {x:?} ({:#018x}) read {y:?} ({:#018x})",
                    x.to_bits(),
                    y.to_bits()
                );
            }
        }
    }
    "the entries differ outside their numbers".to_string()
}

/// Everything one run writes, computed before anything is written.
struct Outputs {
    report: Value,
    tables: Vec<(String, Vec<u8>)>,
    rows: Vec<u8>,
}

fn run(args: &Args) -> Result<Outputs> {
    ensure!(
        !args.out_dir.exists(),
        "--out-dir {} already exists",
        args.out_dir.display()
    );
    staging_site(&args.out_dir)?;
    ensure!(
        !args.name.trim().is_empty() && args.name.len() <= MAX_NAME_BYTES,
        "--name must be a non-empty name of at most {MAX_NAME_BYTES} bytes: it is what a hash pins"
    );
    ensure!(
        args.alpha > 0.0 && args.alpha < 1.0,
        "--alpha must be in (0, 1), got {}",
        args.alpha
    );
    ensure!(
        args.target_precision > 0.0 && args.target_precision <= 1.0,
        "--target-precision must be in (0, 1], got {}",
        args.target_precision
    );
    let split_key = match (args.population, &args.split_key) {
        (Population::TwoFold, Some(k)) if !k.is_empty() => Some(k.as_str()),
        (Population::TwoFold, _) => {
            return Err("--population two-fold needs a non-empty --split-key".to_string());
        }
        (Population::All, None) => None,
        (Population::All, Some(_)) => {
            return Err("--split-key means nothing under --population all; refused".to_string());
        }
    };

    let (inputs, letters, span_rows) = read_inputs(&args.verdicts)?;
    let letter_rows = letters.len();
    ensure!(
        letter_rows > 0,
        "the inputs hold {span_rows} span rows and no letter row: there is nothing to fit"
    );
    let groups = group(letters)?;

    let mut entries = serde_json::Map::new();
    let mut row_lines: Vec<u8> = Vec::new();
    let mut table_all = CalibrationTable::new(args.name.clone());
    let mut table_a = CalibrationTable::new(format!("{}.fold-a", args.name));
    let mut table_b = CalibrationTable::new(format!("{}.fold-b", args.name));
    let (mut fold_rows_a, mut fold_rows_b) = (0usize, 0usize);
    let mut fitted_keys: BTreeMap<&str, Vec<String>> = BTreeMap::new();

    for g in &groups {
        let key = g.key();
        let width = g.rows.width();
        let noul_row = width - RESERVED_NOUL_ROWS;
        let before = score_rows(&g.rows, &temperature_only(1.0))?;
        let ece_before = ece_state(
            &before.iter().map(|s| s.confidence).collect::<Vec<_>>(),
            &before.iter().map(|s| s.correct).collect::<Vec<_>>(),
        )?;
        let disagreements = g
            .lines
            .iter()
            .zip(&before)
            .filter(|(line, s)| line.verdict_top != s.top)
            .count();
        let gold_noul = g.lines.iter().filter(|l| l.gold == noul_row).count();
        let mut entry = json!({
            "kind": g.kind.as_str(),
            "rows": width,
            "options": width - RESERVED_NOUL_ROWS,
            "n": g.rows.len(),
            "gold_noul": gold_noul,
            "verdict_top_disagreements": disagreements,
            "ece_before": ece_json(&ece_before),
        });

        let mut scored: Vec<Option<(RowScore, CalibrationEntry)>> = vec![None; g.rows.len()];
        match split_key {
            None => {
                let fit = fit_entry(&g.rows, args.alpha, args.target_precision)
                    .map_err(|e| format!("{key}: {e}"))?;
                let e = fit.entry();
                table_all = table_all.with_letters(g.kind, width, e);
                fitted_keys.entry("all").or_default().push(key.clone());
                for (i, s) in score_rows(&g.rows, &e)?.into_iter().enumerate() {
                    scored[i] = Some((s, e));
                }
                entry["fit"] = fit_json(&fit);
                entry["scored"] = summarise(&scored, noul_row, "in_sample")?;
                for (i, line) in g.lines.iter().enumerate() {
                    let v = row_line(line, &key, None, Some("all"), scored[i].as_ref());
                    row_lines.extend(serde_json::to_vec(&v).map_err(|e| e.to_string())?);
                    row_lines.push(b'\n');
                }
            }
            Some(split) => {
                let folds: Vec<Fold> = g.lines.iter().map(|l| fold_of(split, &l.row_id)).collect();
                let mut fits: BTreeMap<&str, Option<EntryFit>> = BTreeMap::new();
                let mut folds_json = serde_json::Map::new();
                for fold in [Fold::A, Fold::B] {
                    let idx: Vec<usize> = (0..folds.len()).filter(|i| folds[*i] == fold).collect();
                    match fold {
                        Fold::A => fold_rows_a += idx.len(),
                        Fold::B => fold_rows_b += idx.len(),
                    }
                    let fit = if idx.is_empty() {
                        folds_json.insert(
                            fold.as_str().to_string(),
                            json!({"n": 0, "fitted": false,
                                   "reason": format!("fold {} holds no row of {key}", fold.as_str())}),
                        );
                        None
                    } else {
                        let fit =
                            fit_entry(&g.rows.subset(&idx)?, args.alpha, args.target_precision)
                                .map_err(|e| format!("{key} fold {}: {e}", fold.as_str()))?;
                        let mut body = fit_json(&fit);
                        body["fitted"] = json!(true);
                        folds_json.insert(fold.as_str().to_string(), body);
                        let (table, name) = match fold {
                            Fold::A => (&mut table_a, "fold-a"),
                            Fold::B => (&mut table_b, "fold-b"),
                        };
                        *table = std::mem::replace(table, CalibrationTable::new("")).with_letters(
                            g.kind,
                            width,
                            fit.entry(),
                        );
                        fitted_keys.entry(name).or_default().push(key.clone());
                        Some(fit)
                    };
                    fits.insert(fold.as_str(), fit);
                }
                // Every row scored with the parameters of the fold it was not fitted on.
                for fold in [Fold::A, Fold::B] {
                    let Some(Some(other)) = fits.get(fold.other().as_str()) else {
                        continue;
                    };
                    let e = other.entry();
                    let idx: Vec<usize> = (0..folds.len()).filter(|i| folds[*i] == fold).collect();
                    for (k, s) in score_rows(&g.rows.subset(&idx)?, &e)?
                        .into_iter()
                        .enumerate()
                    {
                        scored[idx[k]] = Some((s, e));
                    }
                }
                entry["folds"] = Value::Object(folds_json);
                entry["scored"] = summarise(&scored, noul_row, "out_of_fold")?;
                for (i, line) in g.lines.iter().enumerate() {
                    let params = scored[i].as_ref().map(|_| {
                        if folds[i] == Fold::A {
                            "fold-b"
                        } else {
                            "fold-a"
                        }
                    });
                    let v = row_line(line, &key, Some(folds[i]), params, scored[i].as_ref());
                    row_lines.extend(serde_json::to_vec(&v).map_err(|e| e.to_string())?);
                    row_lines.push(b'\n');
                }
            }
        }
        entries.insert(key, entry);
    }

    let tables: Vec<(String, CalibrationTable, &str)> = match split_key {
        None => vec![("table.json".to_string(), table_all, "all")],
        Some(_) => {
            ensure!(
                fold_rows_a > 0 && fold_rows_b > 0,
                "fold A holds {fold_rows_a} letter rows and fold B {fold_rows_b}: a two-fold fit \
                 needs rows in both"
            );
            vec![
                ("table.fold-a.json".to_string(), table_a, "fold-a"),
                ("table.fold-b.json".to_string(), table_b, "fold-b"),
            ]
        }
    };
    let mut written = Vec::new();
    let mut tables_json = Vec::new();
    for (file, table, fitted_on) in &tables {
        let bytes = table_bytes(table)?;
        tables_json.push(json!({
            "file": file, "name": table.name, "sha256": table.hash(), "fitted_on": fitted_on,
            "entries": fitted_keys.get(fitted_on).cloned().unwrap_or_default(),
            "span": "null: not fitted",
        }));
        written.push((file.clone(), bytes));
    }

    let mut eval_rows: BTreeMap<String, Value> = BTreeMap::new();
    for input in &inputs {
        let slot = eval_rows
            .entry(input.eval_row_id.clone())
            .or_insert_with(|| json!({"lines": 0, "by_kind": {}, "seeds": [], "files": []}));
        slot["lines"] = json!(slot["lines"].as_u64().unwrap_or(0) + input.lines as u64);
        for (kind, (n, right)) in &input.by_kind {
            let prev = &slot["by_kind"][kind];
            let n0 = prev["n"].as_u64().unwrap_or(0);
            let r0 = prev["verdict_correct"].as_u64().unwrap_or(0);
            slot["by_kind"][kind] =
                json!({"n": n0 + *n as u64, "verdict_correct": r0 + *right as u64});
        }
        let mut seeds: BTreeSet<i64> = slot["seeds"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_default();
        seeds.extend(input.seeds.iter().copied());
        slot["seeds"] = json!(seeds);
        if let Some(files) = slot["files"].as_array_mut() {
            files.push(json!(input.path));
        }
    }

    let kinds_present: BTreeSet<&str> = groups.iter().map(|g| g.kind.as_str()).collect();
    let mut caveats = vec![format!(
        "{SPAN_GAP}: span verdict lines carry no pointer distribution, so the span entry is not \
         fitted; every table here has span null, and a span request against it refuses with \
         calibration_entry_missing. {span_rows} span rows were read and not fitted"
    )];
    for kind in ["choice", "score"] {
        if !kinds_present.contains(kind) {
            caveats.push(format!(
                "no {kind} row in the inputs: the table has no {kind} entry, and a {kind} request \
                 refuses with calibration_entry_missing"
            ));
        }
    }
    caveats.push(
        "a table is written and reported, not installed: which population ships is the human's \
         call (CLAUDE.md rule 2)"
            .to_string(),
    );
    caveats.push(match args.population {
        Population::All => "population all scores each entry on the rows it was fitted on: \
                            ece_after, coverage and abstentions are in-sample and optimistic"
            .to_string(),
        Population::TwoFold => "population two-fold scores every row with the other fold's \
                                parameters; rows whose other fold fitted nothing for their \
                                entry are counted in n_not_scored, never in the rates"
            .to_string(),
    });
    caveats.push(
        "abstentions model answer.rs's noul-row and margin halves; the choice slot's permuted \
         second pass is not modelled"
            .to_string(),
    );

    let report = json!({
        "tool": "qd-calib-fit",
        "name": args.name,
        "population": args.population.as_str(),
        "split_key": split_key,
        "split_rule": split_key.map(|_| "fold a iff sha256(split_key || 0x00 || row_id)[0] is even"),
        "alpha": args.alpha,
        "target_precision": args.target_precision,
        "temperature_search": {
            "lo": TEMPERATURE_LO, "hi": TEMPERATURE_HI, "iters": TEMPERATURE_ITERS,
            "method": "ternary search on log T minimising mean NLL",
        },
        "ece": {"bins": ECE_BINS, "min_samples": ECE_MIN_SAMPLES, "threshold": ECE_THRESHOLD},
        "logits_read_as": "f32, calibrate()'s input type",
        "inputs": inputs.iter().map(|i| json!({
            "path": i.path, "sha256": i.sha256, "eval_row_id": i.eval_row_id, "lines": i.lines,
            "by_kind": i.by_kind.iter().map(|(k, (n, r))| (k.clone(), json!({
                "n": n, "verdict_correct": r,
            }))).collect::<serde_json::Map<_, _>>(),
            "seeds": i.seeds,
        })).collect::<Vec<_>>(),
        "eval_rows": eval_rows,
        "population_counts": {
            "letter_rows": letter_rows,
            "span_rows_not_fitted": span_rows,
            "entries": groups.len(),
            "fold_a_rows": split_key.map(|_| fold_rows_a),
            "fold_b_rows": split_key.map(|_| fold_rows_b),
        },
        "span": {
            "rows": span_rows, "fitted": false, "gap": SPAN_GAP,
            "reason": "span verdict lines carry the pointer heads' winners, not their \
                       distributions; there is nothing to fit the span entry on",
        },
        "entries": entries,
        "tables": tables_json,
        "rows_file": "rows.jsonl",
        "caveats": caveats,
    });
    Ok(Outputs {
        report,
        tables: written,
        rows: row_lines,
    })
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

/// Write into a sibling staging directory and rename it into place, so `--out-dir` appears whole
/// or not at all.
/// Where the staging directory goes: beside `--out-dir`, in a parent that must already exist.
/// Asked before any input is read, so a run that could not land its outputs fails in
/// milliseconds rather than after the fit.
fn staging_site(out_dir: &Path) -> Result<(PathBuf, String)> {
    let name = out_dir
        .file_name()
        .ok_or_else(|| format!("--out-dir {} has no final component", out_dir.display()))?;
    let parent = match out_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    ensure!(
        parent.is_dir(),
        "--out-dir {}: its parent {} does not exist; create it first",
        out_dir.display(),
        parent.display()
    );
    Ok((parent, name.to_string_lossy().into_owned()))
}

fn write_outputs(out_dir: &Path, outputs: &Outputs) -> Result<()> {
    let (parent, name) = staging_site(out_dir)?;
    let staging = parent.join(format!(".{name}.partial-{}", std::process::id()));
    std::fs::create_dir(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    let report = serde_json::to_string_pretty(&outputs.report).map_err(|e| e.to_string())?;
    write_new(
        &staging.join("report.json"),
        format!("{report}\n").as_bytes(),
    )?;
    for (file, bytes) in &outputs.tables {
        write_new(&staging.join(file), bytes)?;
    }
    write_new(&staging.join("rows.jsonl"), &outputs.rows)?;
    ensure!(
        !out_dir.exists(),
        "--out-dir {} appeared while this run wrote; the outputs are left in {}",
        out_dir.display(),
        staging.display()
    );
    std::fs::rename(&staging, out_dir).map_err(|e| {
        format!(
            "renaming {} to {}: {e}; the outputs are left in the former",
            staging.display(),
            out_dir.display()
        )
    })
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    match run(&args).and_then(|out| write_outputs(&args.out_dir, &out).map(|()| out)) {
        Ok(out) => {
            let summary = json!({
                "out_dir": args.out_dir.display().to_string(),
                "tables": out.report["tables"],
                "population_counts": out.report["population_counts"],
            });
            println!("{summary}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("qd-calib-fit: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_is_the_margin_probes_and_keyed() {
        // `sha256("k" || 0x00 || "r0")`, computed independently of `fold_of`.
        let mut h = Sha256::new();
        h.update(b"k\x00r0");
        let expected = if h.finalize()[0] % 2 == 0 {
            Fold::A
        } else {
            Fold::B
        };
        assert_eq!(fold_of("k", "r0"), expected);
        let ids: Vec<String> = (0..400).map(|i| format!("r{i}")).collect();
        let a: Vec<Fold> = ids.iter().map(|r| fold_of("k1", r)).collect();
        let b: Vec<Fold> = ids.iter().map(|r| fold_of("k2", r)).collect();
        assert_ne!(a, b);
        let n = a.iter().filter(|f| **f == Fold::A).count();
        assert!((140..260).contains(&n), "{n} of 400 in fold A");
    }

    #[test]
    fn a_letter_line_outside_the_runtime_shape_is_refused() {
        let ok = json!({"eval_row_id": "E", "row_id": "r", "slot_name": "s", "kind": "choice",
                        "correct": true, "rows": 3, "noul_row": 2, "gold_row": 0, "top": 0,
                        "row_logits": [1.0, 0.0, -1.0]});
        assert!(letter_line(&ok, SlotKind::Choice, "t").is_ok());
        for (key, bad) in [
            ("rows", json!(2)),
            ("rows", json!(18)),
            ("noul_row", json!(0)),
            ("gold_row", json!(3)),
            ("top", json!(-1)),
            ("row_logits", json!([1.0, 0.0])),
            ("row_logits", json!([1.0, 0.0, 1e39])),
            ("row_logits", json!([1.0, "x", 0.0])),
        ] {
            let mut line = ok.clone();
            line[key] = bad.clone();
            assert!(
                letter_line(&line, SlotKind::Choice, "t").is_err(),
                "{key} = {bad} was accepted"
            );
        }
        let mut no_logits = ok.clone();
        no_logits.as_object_mut().unwrap().remove("row_logits");
        let err = letter_line(&no_logits, SlotKind::Choice, "t").unwrap_err();
        assert!(err.contains("older than 444bedc"), "{err}");
    }
}
