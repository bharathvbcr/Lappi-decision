//! `qd-post-f-rules` — the three decisions Fable pre-registered for the GPU queue after run F,
//! read off ledger rows, plus the two row look-ups that queue's box scripts need to pin the ids
//! they grep out of F's logs to the ledger.
//!
//! The rules are `campaign/f-j7prime-preregistered.json` (committed at 4fd08cf; Fable's reading
//! of avg-np's (b) and (c), `avg_np_qualifies_for_f_j7prime.avg_np_reading`, at 54512e6) and,
//! for the OOD falsifier, `campaign/f-v4-preregistered.json` (3a1796d). The scripts that call
//! this are `campaign/post-f-queue/*.sh`.
//!
//! Decisions (each writes its JSON to `--out` and stderr, and one word to stdout):
//!
//! * `avgnp` — (i) does J4's norm-preserving average qualify for F's J7'? Iff all of
//!   (a) `ood_abstain` total >= 34/180, (b) 8K needle worst bucket at least 1d93b3ee's and no
//!   1K/2K/4K length-control worst bucket below 0b86fae3's, (c) `val_top1.choice` and
//!   `val_top1.span` at least J4's seed minima (f3612f73, 2f5fe57a). Printed as 0.279, 0.895 /
//!   0.767 / 0.426, 0.765 and 0.884; the rows' counts govern (below). Words: `qualifies`,
//!   `fails`.
//! * `seeds34` — (ii) do F's three 8K needle worst buckets spread by more than 0.30? Words:
//!   `fires`, `quiet`.
//! * `j6f` — (iii) does J6(f) run right after J5'? Iff F's 8K worst bucket is < 0.95 on >= 2 of
//!   3 seeds, or any F seed has prose <= 9/60 or scrambled <= 8/60. Words: `fires`, `quiet`.
//!
//! Look-ups (ids on stdout, JSON on stderr): `ft-rows` checks a set of ft rows can be averaged
//! or ensembled (completed, not quick, tag `epoch`, the seed claimed, one recipe and one data
//! snapshot) and prints their ids in order; `eval-row` prints the one completed eval row of a
//! tag scored from a given ft row.
//!
//! # Fail closed
//!
//! A missing row, a row present twice, a metric that is absent or not `ran`, a malformed ledger
//! line, a count whose recorded value disagrees with its own `n / n_total`, a needle gate whose
//! value is not the minimum of its own depth buckets, or a cited reference row that does not
//! round to the decimal the pre-registration printed for it: every one refuses. A refusal exits
//! 3, prints `refused` on stdout and writes the reason into the JSON. Nothing defaults.
//!
//! # Exact arithmetic
//!
//! Every comparison is made on the integer counts a row records (`n`, `n_total`), never on the
//! float `value`, so a threshold sitting exactly on a bucket's fraction cannot flip on rounding.
//! The float is read only to cross-check it against its own counts.
//!
//! # The cited row's count governs (b) and (c)
//!
//! (b) and (c) print their thresholds as three-decimal numbers *and* name the rows they were read
//! from. The decimals are roundings of those rows' fractions and do not sit on them (0.279,
//! 0.895 and 0.767 above 17/61, 51/57 and 46/60; 0.765, 0.426 and 0.884 below 8405/10985, 26/61
//! and 6399/7238). Fable's reading (54512e6, `avg_np_reading`) settles which governs: the cited
//! row's exact count. Each part passes iff
//! `candidate.n * cited.n_total >= cited.n * candidate.n_total`, by integer cross-multiplication
//! on the candidate's own worst-bucket (or val) fraction, never the float; a tie passes. (a) is
//! the integer 34/180, one reading. The decimal comparison is still computed and written to the
//! JSON as `literal.passes`, report-only: it decides nothing. A cited row that no longer rounds
//! to its printed decimal still refuses, since then the record and its row disagree. With the
//! suites' fixed sizes the two readings differ on exactly four candidate values (17/61 at 8K,
//! 51/57 at 1K and 46/60 at 2K pass; 8404/10985 choice fails); a test enumerates every value.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, String>;

macro_rules! ensure {
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            return Err(format!($($arg)+));
        }
    };
}

const TOOL: &str = "qd-post-f-rules";
const PREREG: &str = "campaign/f-j7prime-preregistered.json (54512e6)";
const PREREG_F: &str = "campaign/f-v4-preregistered.json (3a1796d)";
/// A ledger larger than this is not one this repo writes (`qd-gate-report`'s cap).
const MAX_LEDGER_BYTES: u64 = 256 * 1024 * 1024;
/// How far a recorded float may sit from its own counts before the row is refused.
const VALUE_TOLERANCE: f64 = 1e-12;
/// clap exits 2 on a usage error; a refusal is its own code.
const EXIT_REFUSED: u8 = 3;
/// More ft rows than this is not one J7' (a plan holds at most 8 kinds).
const MAX_FT_ROWS: usize = 8;

/// A threshold as the pre-registration printed it: `num / den`, and the text itself.
#[derive(Clone, Copy, Debug)]
struct Dec {
    num: u64,
    den: u64,
    text: &'static str,
}

// --- (i): campaign/f-j7prime-preregistered.json, avg_np_qualifies_for_f_j7prime -------------
/// (a) "ood_abstain total >= 34/180 (the fp32 seed minimum, J4 seed1's diagnostic row 6fcba23d)".
const OOD_MIN: u64 = 34;
const OOD_TOTAL: u64 = 180;
const REF_OOD_SEED_MIN: &str = "6fcba23d-210c-44fe-b9b3-9ea8e76c53e6";
/// (b) "8K needle worst bucket >= 0.279".
const NEEDLE_8K_MIN: Dec = Dec {
    num: 279,
    den: 1000,
    text: "0.279",
};
/// (b) "no 1K/2K/4K length-control worst bucket below the average's 0.895 / 0.767 / 0.426".
const CONTROL_MIN: [(u32, Dec); 3] = [
    (
        1024,
        Dec {
            num: 895,
            den: 1000,
            text: "0.895",
        },
    ),
    (
        2048,
        Dec {
            num: 767,
            den: 1000,
            text: "0.767",
        },
    ),
    (
        4096,
        Dec {
            num: 426,
            den: 1000,
            text: "0.426",
        },
    ),
];
/// (b)'s cited rows: the J7g average's gate row and its needle length control.
const REF_AVG_GATE: &str = "1d93b3ee-28fc-4345-a02c-f757bcdb8466";
const REF_AVG_CONTROL: &str = "0b86fae3-f6d7-4bac-b887-176052502b67";
/// (c) "val_top1.choice >= 0.765 and val_top1.span >= 0.884 (J4 seed minima, f3612f73 / 2f5fe57a)".
const CHOICE_MIN: Dec = Dec {
    num: 765,
    den: 1000,
    text: "0.765",
};
const SPAN_MIN: Dec = Dec {
    num: 884,
    den: 1000,
    text: "0.884",
};
const REF_CHOICE_MIN: &str = "f3612f73-df9f-43a3-adf0-1493f7c659da";
const REF_SPAN_MIN: &str = "2f5fe57a-189f-4e00-a8e7-8824466c5ea1";
/// The tags `real_ft_run.AVERAGED_NP_TAG` gives avg-np's rows.
const AVGNP_GATE_TAG: &str = "avg-np-score-val";
const AVGNP_CONTROL_TAG: &str = "avg-np-needle-length-control";

// --- (ii): seeds_3_4 --------------------------------------------------------------------------
/// "if F's three 8K needle worst buckets spread by more than 0.30".
const SPREAD_MAX: Dec = Dec {
    num: 30,
    den: 100,
    text: "0.30",
};

// --- (iii): j6f_position ----------------------------------------------------------------------
/// "F's 8K worst bucket < 0.95 on >= 2 of 3 seeds".
const NEEDLE_FLOOR: Dec = Dec {
    num: 95,
    den: 100,
    text: "0.95",
};
const NEEDLE_FLOOR_SEEDS: usize = 2;
/// "any F seed has prose <= 9/60 or scrambled <= 8/60" (f-v4's OOD falsifier: J4's envelope).
const PROSE_MAX: u64 = 9;
const SCRAMBLED_MAX: u64 = 8;
const OOD_CATEGORY_TOTAL: u64 = 60;

/// F's three seeds, which (ii) and (iii) are about.
const F_SEEDS: [i64; 3] = [0, 1, 2];
/// The needle suite's depth buckets (`needle.depth_bucket_label`), in depth order.
const DEPTH_BUCKETS: [&str; 5] = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"];
/// The in-training score row's tag, whose metrics carry `ft_run_row_id`.
const EVAL_TAG: &str = "epoch-score-val";

#[derive(Parser, Debug)]
#[command(
    name = "qd-post-f-rules",
    about = "The post-F queue's pre-registered decisions, read off ledger rows; refuses rather \
             than defaults"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// (i) Does J4's avg-np qualify for F's J7'? Prints `qualifies` or `fails`.
    Avgnp {
        /// J7's ledger: the avg-np gate row, its needle control, and reference rows
        /// 1d93b3ee, 0b86fae3 and 6fcba23d.
        #[arg(long)]
        j7_ledger: PathBuf,
        /// J4's ledger: reference rows f3612f73 and 2f5fe57a.
        #[arg(long)]
        j4_ledger: PathBuf,
        /// Where the decision's JSON is written; refused if it exists.
        #[arg(long)]
        out: PathBuf,
    },
    /// (ii) Do F's three 8K worst buckets spread by more than 0.30? Prints `fires` or `quiet`.
    Seeds34 {
        /// F's ledger.
        #[arg(long)]
        f_ledger: PathBuf,
        /// `SEED=FT_ROW_ID`, once for each of seeds 0, 1 and 2 (ids from F's logs).
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
        #[arg(long)]
        out: PathBuf,
    },
    /// (iii) Does J6(f) run right after J5'? Prints `fires` or `quiet`.
    J6f {
        #[arg(long)]
        f_ledger: PathBuf,
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Check ft rows can be averaged or ensembled together; print their ids in order.
    FtRows {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
    },
    /// Print the one completed eval row of `--tag` scored from the given ft row.
    EvalRow {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long = "ft-row", value_parser = parse_ft_row)]
        ft_row: (i64, String),
        #[arg(long, default_value = EVAL_TAG)]
        tag: String,
    },
}

/// `SEED=UUID`: a seed in 0..=99 and a full 8-4-4-4-12 lower-case hex row id.
fn parse_ft_row(text: &str) -> Result<(i64, String)> {
    let (seed, id) = text
        .split_once('=')
        .ok_or_else(|| format!("{text:?} is not SEED=FT_ROW_ID"))?;
    let seed: i64 = seed
        .parse()
        .map_err(|_| format!("{text:?}: seed {seed:?} is not an integer"))?;
    ensure!(
        (0..=99).contains(&seed),
        "{text:?}: seed {seed} is outside 0..=99"
    );
    ensure!(
        is_row_id(id),
        "{text:?}: {id:?} is not a full row id (8-4-4-4-12 lower-case hex)"
    );
    Ok((seed, id.to_string()))
}

fn is_row_id(id: &str) -> bool {
    let parts: Vec<&str> = id.split('-').collect();
    parts.iter().map(|p| p.len()).collect::<Vec<_>>() == [8, 4, 4, 4, 12]
        && parts.iter().all(|p| {
            p.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
}

// --- ledgers ----------------------------------------------------------------------------------

struct Row {
    at: String,
    map: Map<String, Value>,
}

struct Ledger {
    shown: String,
    rows: Vec<Row>,
}

/// Every ledger read, with its sha256, so the JSON says exactly what was decided on.
#[derive(Default)]
struct Inputs(Vec<Value>);

impl Inputs {
    fn read(&mut self, path: &Path) -> Result<Ledger> {
        let shown = path.display().to_string();
        let meta = std::fs::metadata(path).map_err(|e| format!("{shown}: {e}"))?;
        ensure!(
            meta.len() <= MAX_LEDGER_BYTES,
            "{shown}: {} bytes is over the {MAX_LEDGER_BYTES}-byte cap for a ledger",
            meta.len()
        );
        let mut bytes = Vec::with_capacity(meta.len() as usize);
        std::fs::File::open(path)
            .and_then(|f| f.take(MAX_LEDGER_BYTES + 1).read_to_end(&mut bytes))
            .map_err(|e| format!("{shown}: {e}"))?;
        ensure!(
            bytes.len() as u64 <= MAX_LEDGER_BYTES,
            "{shown}: grew past {MAX_LEDGER_BYTES} bytes while it was read"
        );
        let rows = parse_rows(&bytes, &shown)?;
        self.0.push(json!({
            "path": shown,
            "sha256": Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>(),
            "bytes": bytes.len(),
            "rows": rows.len(),
        }));
        Ok(Ledger { shown, rows })
    }
}

/// Every line of a ledger as a row. The only line skipped is the empty one after the final
/// newline; any other line that is not a JSON object with a full `row_id` refuses the ledger,
/// a half-written last line included.
fn parse_rows(bytes: &[u8], shown: &str) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    let lines: Vec<&[u8]> = bytes.split(|b| *b == b'\n').collect();
    let last = lines.len() - 1;
    for (i, raw) in lines.into_iter().enumerate() {
        let at = format!("{shown} line {}", i + 1);
        if raw.is_empty() && i == last {
            continue;
        }
        let value: Value = serde_json::from_slice(raw)
            .map_err(|e| format!("{at}: malformed ledger line ({e})"))?;
        let Value::Object(map) = value else {
            return Err(format!("{at}: malformed ledger line (not a JSON object)"));
        };
        match map.get("row_id").and_then(Value::as_str) {
            Some(id) if is_row_id(id) => {}
            _ => return Err(format!("{at}: malformed ledger line (no full `row_id`)")),
        }
        rows.push(Row { at, map });
    }
    Ok(rows)
}

impl Row {
    fn id(&self) -> &str {
        self.map["row_id"].as_str().unwrap_or_default()
    }
    fn get(&self, path: &[&str]) -> Option<&Value> {
        let mut cur = self.map.get(path[0])?;
        for key in &path[1..] {
            cur = cur.get(key)?;
        }
        Some(cur)
    }
    fn str_at(&self, path: &[&str]) -> Option<&str> {
        self.get(path).and_then(Value::as_str)
    }
    fn completed(&self) -> bool {
        self.str_at(&["status"]) == Some("completed")
    }
    fn tag(&self) -> Option<&str> {
        self.str_at(&["recipe", "tag"])
    }
    fn seed(&self) -> Option<i64> {
        self.get(&["protocol", "seed"]).and_then(Value::as_i64)
    }
    /// A gate or metric that ran, refused if absent or in any other state.
    fn ran(&self, section: &str, name: &str) -> Result<&Map<String, Value>> {
        let entry = self
            .get(&[section, name])
            .and_then(Value::as_object)
            .ok_or_else(|| format!("row {} ({}): no {section}.{name}", self.id(), self.at))?;
        match entry.get("state").and_then(Value::as_str) {
            Some("ran") => Ok(entry),
            state => Err(format!(
                "row {} ({}): {section}.{name} is {} ({}), not ran",
                self.id(),
                self.at,
                state.unwrap_or("stateless"),
                entry
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("no reason recorded")
            )),
        }
    }
    /// A count `n / n_total` that ran, with its recorded value checked against it.
    fn count(&self, section: &str, name: &str) -> Result<Frac> {
        let entry = self.ran(section, name)?;
        let frac = counts(entry, &format!("row {} {section}.{name}", self.id()))?;
        let value = recorded_value(entry, &format!("row {} {section}.{name}", self.id()))?;
        ensure!(
            close(value, frac.f64()),
            "row {} {section}.{name}: value {value} is not its own n/n_total {}/{}",
            self.id(),
            frac.k,
            frac.n
        );
        Ok(frac)
    }
}

fn counts(entry: &Map<String, Value>, what: &str) -> Result<Frac> {
    let n = entry.get("n").and_then(Value::as_u64);
    let total = entry.get("n_total").and_then(Value::as_u64);
    match (n, total) {
        (Some(k), Some(n)) => Frac::new(k, n, what),
        _ => Err(format!("{what}: no integer n and n_total")),
    }
}

/// Whether a recorded float is the value of its own counts. A NaN is never close.
fn close(recorded: f64, exact: f64) -> bool {
    (recorded - exact).abs() <= VALUE_TOLERANCE
}

fn recorded_value(entry: &Map<String, Value>, what: &str) -> Result<f64> {
    entry
        .get("value")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("{what}: no finite value"))
}

impl Ledger {
    fn by_id(&self, id: &str) -> Result<&Row> {
        let found: Vec<&Row> = self.rows.iter().filter(|r| r.id() == id).collect();
        ensure!(
            found.len() == 1,
            "row {id} appears {} time(s) in {}, not once",
            found.len(),
            self.shown
        );
        Ok(found[0])
    }
    /// The one completed row of `run_kind` and `tag` that `pred` accepts.
    fn one<'a>(
        &'a self,
        run_kind: &str,
        tag: &str,
        what: &str,
        pred: impl Fn(&Row) -> bool,
    ) -> Result<&'a Row> {
        let found: Vec<&Row> = self
            .rows
            .iter()
            .filter(|r| {
                r.completed()
                    && r.str_at(&["run_kind"]) == Some(run_kind)
                    && r.tag() == Some(tag)
                    && pred(r)
            })
            .collect();
        match found.len() {
            1 => Ok(found[0]),
            0 => Err(format!(
                "missing row: no completed {run_kind} row tagged {tag} for {what} in {}",
                self.shown
            )),
            n => Err(format!(
                "{n} completed {run_kind} rows tagged {tag} for {what} in {} ({}); which one \
                 decides is not this tool's call",
                self.shown,
                found.iter().map(|r| r.id()).collect::<Vec<_>>().join(", ")
            )),
        }
    }
}

// --- exact fractions ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Frac {
    k: u64,
    n: u64,
}

impl Frac {
    fn new(k: u64, n: u64, what: &str) -> Result<Frac> {
        ensure!(
            n > 0 && k <= n,
            "{what}: {k}/{n} is not a fraction of a non-empty set"
        );
        Ok(Frac { k, n })
    }
    fn f64(self) -> f64 {
        self.k as f64 / self.n as f64
    }
    fn ge(self, o: Frac) -> bool {
        u128::from(self.k) * u128::from(o.n) >= u128::from(o.k) * u128::from(self.n)
    }
    fn lt(self, o: Frac) -> bool {
        !self.ge(o)
    }
    fn ge_dec(self, d: Dec) -> bool {
        u128::from(self.k) * u128::from(d.den) >= u128::from(d.num) * u128::from(self.n)
    }
    fn lt_dec(self, d: Dec) -> bool {
        !self.ge_dec(d)
    }
    /// Whether this fraction, printed to `d`'s number of decimals, is `d`:
    /// |k/n - num/den| <= 1/(2 den).
    fn rounds_to(self, d: Dec) -> bool {
        let lhs = (2 * i128::from(d.den) * i128::from(self.k)
            - 2 * i128::from(d.num) * i128::from(self.n))
        .abs();
        lhs <= i128::from(self.n)
    }
    fn json(self) -> Value {
        json!({"n": self.k, "n_total": self.n, "value": self.f64()})
    }
}

/// `max - min > d`, exactly.
fn spread_exceeds(max: Frac, min: Frac, d: Dec) -> bool {
    let (a, b, c, e) = (
        i128::from(max.k),
        i128::from(max.n),
        i128::from(min.k),
        i128::from(min.n),
    );
    i128::from(d.den) * (a * e - c * b) > i128::from(d.num) * b * e
}

// --- the needle --------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Worst {
    bucket: &'static str,
    frac: Frac,
}

impl Worst {
    fn json(self) -> Value {
        json!({"bucket": self.bucket, "n": self.frac.k, "n_total": self.frac.n, "value": self.frac.f64()})
    }
}

/// The worst depth bucket under `prefix`, cross-checked against the gate (or control) value the
/// row recorded for it. A bucket missing, an extra bucket, or a disagreement refuses.
fn needle_worst(row: &Row, section: &str, name: &str, prefix: &str) -> Result<Worst> {
    let gate = row.ran(section, name)?;
    let value = recorded_value(gate, &format!("row {} {section}.{name}", row.id()))?;
    let metrics = row
        .get(&["metrics"])
        .and_then(Value::as_object)
        .ok_or_else(|| format!("row {}: no metrics", row.id()))?;
    let extra: Vec<&String> = metrics
        .keys()
        .filter(|k| {
            k.strip_prefix(prefix)
                .is_some_and(|label| !DEPTH_BUCKETS.contains(&label))
        })
        .collect();
    ensure!(
        extra.is_empty(),
        "row {}: depth buckets {extra:?} are not the suite's {DEPTH_BUCKETS:?}",
        row.id()
    );
    let mut worst: Option<Worst> = None;
    for label in DEPTH_BUCKETS {
        let frac = row.count("metrics", &format!("{prefix}{label}"))?;
        if worst.is_none_or(|w| frac.lt(w.frac)) {
            worst = Some(Worst {
                bucket: label,
                frac,
            });
        }
    }
    let worst = worst.ok_or_else(|| format!("row {}: no depth bucket", row.id()))?;
    ensure!(
        close(value, worst.frac.f64()),
        "row {} {section}.{name}: value {value} is not its worst depth bucket {} at {}/{}",
        row.id(),
        worst.bucket,
        worst.frac.k,
        worst.frac.n
    );
    Ok(worst)
}

fn needle_8k(row: &Row) -> Result<Worst> {
    needle_worst(
        row,
        "gates",
        "needle_hunk_recall",
        "needle_hunk_recall.depth.",
    )
}

fn needle_control(row: &Row, length: u32) -> Result<Worst> {
    needle_worst(
        row,
        "metrics",
        &format!("needle_hunk_recall.control.{length}"),
        &format!("needle_hunk_recall.control.{length}.depth."),
    )
}

// --- (i) ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Pass,
    Fail,
}

impl Verdict {
    fn of(pass: bool) -> Verdict {
        if pass { Verdict::Pass } else { Verdict::Fail }
    }
    fn word(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
        }
    }
    /// A clause of several parts passes iff every part passes.
    fn all(parts: &[Verdict]) -> Verdict {
        Verdict::of(parts.iter().all(|&p| p == Verdict::Pass))
    }
}

/// `candidate >= cited`, exactly (the avg-np reading, 54512e6): the cited row's count is the
/// threshold and a tie passes. The printed decimal is compared too and reported, deciding
/// nothing; the cited fraction must still round to it, or the record and its row disagree.
fn at_least(
    what: &str,
    candidate: Frac,
    literal: Dec,
    cited: Frac,
    cited_row: &str,
) -> Result<(Verdict, Value)> {
    ensure!(
        cited.rounds_to(literal),
        "{what}: the cited row {cited_row} records {}/{} = {:.6}, which does not print as the \
         pre-registered {}; the record and its row disagree",
        cited.k,
        cited.n,
        cited.f64(),
        literal.text
    );
    let by_row = candidate.ge(cited);
    let verdict = Verdict::of(by_row);
    Ok((
        verdict,
        json!({
            "what": what,
            "verdict": verdict.word(),
            "candidate": candidate.json(),
            "cited_row": {"row_id": cited_row, "threshold": cited.json(), "passes": by_row},
            "literal": {
                "threshold": literal.text,
                "passes": candidate.ge_dec(literal),
                "role": "report-only: the printed rendering of the cited row; it decides nothing",
            },
        }),
    ))
}

fn rule_avgnp(inputs: &mut Inputs, j7: &Path, j4: &Path) -> Result<(String, Value)> {
    let j7 = inputs.read(j7)?;
    let j4 = inputs.read(j4)?;
    let ref_gate = j7.by_id(REF_AVG_GATE)?;
    let ref_control = j7.by_id(REF_AVG_CONTROL)?;
    let ref_ood = j7.by_id(REF_OOD_SEED_MIN)?;
    let ref_choice = j4.by_id(REF_CHOICE_MIN)?;
    let ref_span = j4.by_id(REF_SPAN_MIN)?;

    // The candidate is J4's avg-np: the same three ft rows as the average it is set against,
    // scored in the average's dtype.
    let inputs_of = |r: &Row| -> Option<String> {
        r.str_at(&["metrics", "ft_run_row_ids", "value"])
            .map(str::to_string)
    };
    let ft_ids = inputs_of(ref_gate)
        .ok_or_else(|| format!("row {REF_AVG_GATE}: no metrics.ft_run_row_ids"))?;
    let what = format!("J4's avg-np (ft rows {ft_ids})");
    let gate = j7.one("eval", AVGNP_GATE_TAG, &what, |r| {
        inputs_of(r).as_deref() == Some(ft_ids.as_str())
    })?;
    let control = j7.one("eval", AVGNP_CONTROL_TAG, &what, |r| {
        inputs_of(r).as_deref() == Some(ft_ids.as_str())
    })?;
    let dtype = ref_gate.str_at(&["recipe", "score_dtype"]);
    ensure!(
        gate.str_at(&["recipe", "score_dtype"]) == dtype,
        "row {}: score_dtype {:?}, not the average's {:?} (the rule is scored exactly as {REF_AVG_GATE})",
        gate.id(),
        gate.str_at(&["recipe", "score_dtype"]),
        dtype
    );

    // (a): an integer count, one reading. The cited seed minimum must be the stated count.
    let cited_ood = ref_ood.count("metrics", "ood_diagnostic.all")?;
    ensure!(
        cited_ood
            == Frac {
                k: OOD_MIN,
                n: OOD_TOTAL
            },
        "(a): the cited row {REF_OOD_SEED_MIN} records {}/{}, not the pre-registered \
         {OOD_MIN}/{OOD_TOTAL}",
        cited_ood.k,
        cited_ood.n
    );
    let ood = gate.count("gates", "ood_abstain")?;
    ensure!(
        ood.n == OOD_TOTAL,
        "row {}: gates.ood_abstain is over {} cases, not the rule's {OOD_TOTAL}",
        gate.id(),
        ood.n
    );
    let mut by_category = BTreeMap::new();
    for cat in ["prose", "scrambled", "unseen-language"] {
        by_category.insert(cat, gate.count("metrics", &format!("ood_abstain.{cat}"))?);
    }
    let summed: u64 = by_category.values().map(|f| f.k).sum();
    ensure!(
        summed == ood.k,
        "row {}: gates.ood_abstain counts {} but its categories sum to {summed}",
        gate.id(),
        ood.k
    );
    let a = Verdict::of(ood.k >= OOD_MIN);
    let a_json = json!({
        "verdict": a.word(),
        "rule": format!("ood_abstain total >= {OOD_MIN}/{OOD_TOTAL}"),
        "candidate": ood.json(),
        "by_category": by_category.iter().map(|(c, f)| (c.to_string(), f.json())).collect::<Map<_, _>>(),
        "cited_row": {"row_id": REF_OOD_SEED_MIN, "metric": "ood_diagnostic.all", "threshold": cited_ood.json()},
    });

    // (b): the 8K gate's worst bucket, and each length control's.
    let mut b_parts = vec![at_least(
        "8K needle worst bucket",
        needle_8k(gate)?.frac,
        NEEDLE_8K_MIN,
        needle_8k(ref_gate)?.frac,
        REF_AVG_GATE,
    )?];
    for (length, literal) in CONTROL_MIN {
        b_parts.push(at_least(
            &format!("{length}-token length-control worst bucket"),
            needle_control(control, length)?.frac,
            literal,
            needle_control(ref_control, length)?.frac,
            REF_AVG_CONTROL,
        )?);
    }

    // (c): val top-1, choice and span.
    let c_parts = vec![
        at_least(
            "val_top1.choice",
            gate.count("metrics", "val_top1.choice")?,
            CHOICE_MIN,
            ref_choice.count("metrics", "val_top1.choice")?,
            REF_CHOICE_MIN,
        )?,
        at_least(
            "val_top1.span",
            gate.count("metrics", "val_top1.span")?,
            SPAN_MIN,
            ref_span.count("metrics", "val_top1.span")?,
            REF_SPAN_MIN,
        )?,
    ];
    let (b, b_parts): (Vec<Verdict>, Vec<Value>) = b_parts.into_iter().unzip();
    let (c, c_parts): (Vec<Verdict>, Vec<Value>) = c_parts.into_iter().unzip();
    let (b, c) = (Verdict::all(&b), Verdict::all(&c));

    let body = json!({
        "rows": {"avg_np_gate": gate.id(), "avg_np_needle_control": control.id()},
        "clauses": {
            "a": a_json,
            "b": {"verdict": b.word(), "parts": b_parts},
            "c": {"verdict": c.word(), "parts": c_parts},
        },
        "logic": "qualifies iff (a), (b) and (c) all pass",
        "reading": "(b), (c): each part passes iff candidate.n * cited.n_total >= cited.n * \
                    candidate.n_total (the cited row's exact count governs; a tie passes; \
                    the printed decimal is report-only), avg_np_reading at 54512e6",
    });
    match Verdict::all(&[a, b, c]) {
        Verdict::Pass => Ok(("qualifies".into(), body)),
        Verdict::Fail => Ok(("fails".into(), body)),
    }
}

// --- (ii) and (iii) ----------------------------------------------------------------------------

struct FSeed {
    seed: i64,
    ft_row: String,
    ft_quick: Option<bool>,
    eval_row: String,
    worst_8k: Worst,
    prose: Frac,
    scrambled: Frac,
}

impl FSeed {
    fn json(&self) -> Value {
        json!({
            "seed": self.seed,
            "ft_row": self.ft_row,
            "ft_row_quick": self.ft_quick,
            "eval_row": self.eval_row,
            "needle_8k_worst": self.worst_8k.json(),
            "ood_abstain.prose": self.prose.json(),
            "ood_abstain.scrambled": self.scrambled.json(),
        })
    }
}

fn ft_row<'a>(ledger: &'a Ledger, seed: i64, id: &str) -> Result<&'a Row> {
    let row = ledger.by_id(id)?;
    ensure!(
        row.str_at(&["run_kind"]) == Some("ft") && row.completed(),
        "row {id} in {} is not a completed ft row (run_kind {:?}, status {:?})",
        ledger.shown,
        row.str_at(&["run_kind"]),
        row.str_at(&["status"])
    );
    ensure!(
        row.seed() == Some(seed),
        "ft row {id} is seed {:?}, not the {seed} it was given as",
        row.seed()
    );
    Ok(row)
}

fn eval_row_of<'a>(ledger: &'a Ledger, seed: i64, ft: &str, tag: &str) -> Result<&'a Row> {
    ledger.one(
        "eval",
        tag,
        &format!("seed {seed} scored from ft row {ft}"),
        |r| r.seed() == Some(seed) && r.str_at(&["metrics", "ft_run_row_id", "value"]) == Some(ft),
    )
}

/// F's three seeds' rows, pinned to the ft row ids from F's logs.
fn f_seeds(inputs: &mut Inputs, f_ledger: &Path, ft_rows: &[(i64, String)]) -> Result<Vec<FSeed>> {
    let seeds: Vec<i64> = ft_rows.iter().map(|(s, _)| *s).collect();
    ensure!(
        seeds == F_SEEDS,
        "these rules are about F's seeds {F_SEEDS:?}, given in that order; got {seeds:?}"
    );
    let ledger = inputs.read(f_ledger)?;
    let mut out = Vec::new();
    for (seed, ft) in ft_rows {
        let ft_row = ft_row(&ledger, *seed, ft)?;
        let eval = eval_row_of(&ledger, *seed, ft, EVAL_TAG)?;
        let prose = eval.count("metrics", "ood_abstain.prose")?;
        let scrambled = eval.count("metrics", "ood_abstain.scrambled")?;
        for (name, f) in [("prose", prose), ("scrambled", scrambled)] {
            ensure!(
                f.n == OOD_CATEGORY_TOTAL,
                "row {}: ood_abstain.{name} is over {} cases, not the rule's {OOD_CATEGORY_TOTAL}",
                eval.id(),
                f.n
            );
        }
        out.push(FSeed {
            seed: *seed,
            ft_row: ft.clone(),
            ft_quick: ft_row.get(&["quick"]).and_then(Value::as_bool),
            eval_row: eval.id().to_string(),
            worst_8k: needle_8k(eval)?,
            prose,
            scrambled,
        });
    }
    Ok(out)
}

fn rule_seeds34(
    inputs: &mut Inputs,
    f_ledger: &Path,
    ft_rows: &[(i64, String)],
) -> Result<(String, Value)> {
    let seeds = f_seeds(inputs, f_ledger, ft_rows)?;
    let mut max = seeds[0].worst_8k.frac;
    let mut min = max;
    for s in &seeds[1..] {
        if max.lt(s.worst_8k.frac) {
            max = s.worst_8k.frac;
        }
        if s.worst_8k.frac.lt(min) {
            min = s.worst_8k.frac;
        }
    }
    let fires = spread_exceeds(max, min, SPREAD_MAX);
    let body = json!({
        "rule": format!("seeds 3 and 4 run iff F's three 8K needle worst buckets spread by more than {}", SPREAD_MAX.text),
        "seeds": seeds.iter().map(FSeed::json).collect::<Vec<_>>(),
        "max": max.json(),
        "min": min.json(),
        "spread": max.f64() - min.f64(),
        "spread_exceeds": fires,
        "then": if fires {
            "seeds 3 and 4 run (cap 32,400 s each, with --needle-control 1024,2048,4096) before J7'; J7' averages and ensembles five; a three-seed average is not scored as a candidate"
        } else {
            "no seeds 3-4; J7' averages and ensembles F's three seeds"
        },
    });
    Ok((if fires { "fires" } else { "quiet" }.into(), body))
}

fn rule_j6f(
    inputs: &mut Inputs,
    f_ledger: &Path,
    ft_rows: &[(i64, String)],
) -> Result<(String, Value)> {
    let seeds = f_seeds(inputs, f_ledger, ft_rows)?;
    let below: Vec<i64> = seeds
        .iter()
        .filter(|s| s.worst_8k.frac.lt_dec(NEEDLE_FLOOR))
        .map(|s| s.seed)
        .collect();
    let prose: Vec<i64> = seeds
        .iter()
        .filter(|s| s.prose.k <= PROSE_MAX)
        .map(|s| s.seed)
        .collect();
    let scrambled: Vec<i64> = seeds
        .iter()
        .filter(|s| s.scrambled.k <= SCRAMBLED_MAX)
        .map(|s| s.seed)
        .collect();
    let by_needle = below.len() >= NEEDLE_FLOOR_SEEDS;
    let fires = by_needle || !prose.is_empty() || !scrambled.is_empty();
    let body = json!({
        "rule": format!(
            "J6(f) runs right after J5' iff F's 8K worst bucket < {} on >= {NEEDLE_FLOOR_SEEDS} of 3 seeds, or any F seed has prose <= {PROSE_MAX}/{OOD_CATEGORY_TOTAL} or scrambled <= {SCRAMBLED_MAX}/{OOD_CATEGORY_TOTAL} ({PREREG_F}'s OOD falsifier); otherwise after the no-mask outcome run, before J6(d)",
            NEEDLE_FLOOR.text
        ),
        "seeds": seeds.iter().map(FSeed::json).collect::<Vec<_>>(),
        "needle_below_floor_seeds": below,
        "needle_condition": by_needle,
        "prose_falsifier_seeds": prose,
        "scrambled_falsifier_seeds": scrambled,
        "then": if fires { "J6(f) runs right after J5'" } else { "J6(f) runs after the no-mask outcome run (item 9), before J6(d)" },
    });
    Ok((if fires { "fires" } else { "quiet" }.into(), body))
}

// --- look-ups ----------------------------------------------------------------------------------

fn lookup_ft_rows(
    inputs: &mut Inputs,
    ledger: &Path,
    ft_rows: &[(i64, String)],
) -> Result<(String, Value)> {
    ensure!(
        (1..=MAX_FT_ROWS).contains(&ft_rows.len()),
        "{} ft rows; between 1 and {MAX_FT_ROWS}",
        ft_rows.len()
    );
    let distinct_seeds: BTreeSet<i64> = ft_rows.iter().map(|(s, _)| *s).collect();
    let distinct_ids: BTreeSet<&str> = ft_rows.iter().map(|(_, id)| id.as_str()).collect();
    ensure!(
        distinct_seeds.len() == ft_rows.len() && distinct_ids.len() == ft_rows.len(),
        "a seed or an ft row is named twice"
    );
    let ledger = inputs.read(ledger)?;
    let mut shared: Option<(String, String)> = None;
    let mut rows = Vec::new();
    for (seed, id) in ft_rows {
        let row = ft_row(&ledger, *seed, id)?;
        ensure!(
            row.get(&["quick"]) == Some(&Value::Bool(false)),
            "ft row {id} does not say quick: false; an average or ensemble of it promotes \
             nothing (rule 8) and its scoring refuses it"
        );
        ensure!(
            row.tag() == Some("epoch") && row.get(&["recipe", "shuffled_label"]).is_none(),
            "ft row {id} is not an epoch arm of the recipe (tag {:?}, shuffled_label {})",
            row.tag(),
            row.get(&["recipe", "shuffled_label"]).is_some()
        );
        let recipe = row
            .str_at(&["protocol", "recipe_hash"])
            .ok_or_else(|| format!("ft row {id}: no protocol.recipe_hash"))?;
        let data = row
            .str_at(&["protocol", "data_snapshot_hash"])
            .ok_or_else(|| format!("ft row {id}: no protocol.data_snapshot_hash"))?;
        match &shared {
            None => shared = Some((recipe.to_string(), data.to_string())),
            Some((r, d)) => ensure!(
                r == recipe && d == data,
                "ft row {id} has recipe {recipe} / data {data}, not the first row's {r} / {d}: \
                 not one configuration"
            ),
        }
        rows.push(json!({"seed": seed, "ft_row": id}));
    }
    let (recipe, data) = shared.unwrap_or_default();
    let ids: Vec<&str> = ft_rows.iter().map(|(_, id)| id.as_str()).collect();
    Ok((
        ids.join(" "),
        json!({"rows": rows, "recipe_hash": recipe, "data_snapshot_hash": data}),
    ))
}

fn lookup_eval_row(
    inputs: &mut Inputs,
    ledger: &Path,
    ft: &(i64, String),
    tag: &str,
) -> Result<(String, Value)> {
    let ledger = inputs.read(ledger)?;
    ft_row(&ledger, ft.0, &ft.1)?;
    let row = eval_row_of(&ledger, ft.0, &ft.1, tag)?;
    Ok((
        row.id().to_string(),
        json!({"seed": ft.0, "ft_row": ft.1, "tag": tag, "eval_row": row.id()}),
    ))
}

// --- driver ------------------------------------------------------------------------------------

struct Outcome {
    word: String,
    json: Value,
    refused: bool,
}

fn rule_name(cmd: &Cmd) -> &'static str {
    match cmd {
        Cmd::Avgnp { .. } => "avgnp_qualifies_for_f_j7prime",
        Cmd::Seeds34 { .. } => "seeds_3_4",
        Cmd::J6f { .. } => "j6f_position",
        Cmd::FtRows { .. } => "ft_rows",
        Cmd::EvalRow { .. } => "eval_row",
    }
}

fn run(cmd: &Cmd) -> Outcome {
    let mut inputs = Inputs::default();
    let result = match cmd {
        Cmd::Avgnp {
            j7_ledger,
            j4_ledger,
            ..
        } => rule_avgnp(&mut inputs, j7_ledger, j4_ledger),
        Cmd::Seeds34 {
            f_ledger, ft_rows, ..
        } => rule_seeds34(&mut inputs, f_ledger, ft_rows),
        Cmd::J6f {
            f_ledger, ft_rows, ..
        } => rule_j6f(&mut inputs, f_ledger, ft_rows),
        Cmd::FtRows { ledger, ft_rows } => lookup_ft_rows(&mut inputs, ledger, ft_rows),
        Cmd::EvalRow {
            ledger,
            ft_row,
            tag,
        } => lookup_eval_row(&mut inputs, ledger, ft_row, tag),
    };
    let mut json = json!({
        "tool": TOOL,
        "rule": rule_name(cmd),
        "preregistration": PREREG,
        "inputs": inputs.0,
    });
    let (word, refused) = match result {
        Ok((word, body)) => {
            json["decision"] = Value::String(word.clone());
            json["detail"] = body;
            (word, false)
        }
        Err(reason) => {
            json["decision"] = Value::String("refused".into());
            json["refused"] = Value::String(reason);
            ("refused".to_string(), true)
        }
    };
    Outcome {
        word,
        json,
        refused,
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("--out {}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| format!("--out {}: {e}", path.display()))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let out_path = match &cli.cmd {
        Cmd::Avgnp { out, .. } | Cmd::Seeds34 { out, .. } | Cmd::J6f { out, .. } => Some(out),
        Cmd::FtRows { .. } | Cmd::EvalRow { .. } => None,
    };
    if let Some(path) = out_path
        && path.exists()
    {
        eprintln!("{TOOL}: refused: --out {} already exists", path.display());
        println!("refused");
        return ExitCode::from(EXIT_REFUSED);
    }
    let mut outcome = run(&cli.cmd);
    let mut text = serde_json::to_string_pretty(&outcome.json).unwrap_or_else(|e| {
        outcome.refused = true;
        format!("{{\"tool\": \"{TOOL}\", \"refused\": \"the JSON did not serialise: {e}\"}}")
    });
    text.push('\n');
    if let Some(path) = out_path
        && let Err(e) = write_new(path, text.as_bytes())
    {
        eprintln!("{text}{TOOL}: refused: {e}");
        println!("refused");
        return ExitCode::from(EXIT_REFUSED);
    }
    eprint!("{text}");
    if outcome.refused {
        eprintln!(
            "{TOOL}: refused: {}",
            outcome.json["refused"].as_str().unwrap_or("see above")
        );
        println!("refused");
        return ExitCode::from(EXIT_REFUSED);
    }
    println!("{}", outcome.word);
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    const J4: &str = "ledger/gh200-p4-v3-2026-10-01.jsonl";
    const J7: &str = "ledger/gh200-p6-j7-avg-2026-10-01.jsonl";
    const J4_FT: [&str; 3] = [
        "80b19a41-ad0a-4af7-b1d9-2de675d5199b",
        "9b108fe2-3cf1-41b7-bef0-7dc0dbb7fc45",
        "8b600511-887e-4ebc-97d0-7794127c8d0b",
    ];

    fn repo(rel: &str) -> PathBuf {
        Path::new(REPO).join(rel)
    }

    fn real_rows(rel: &str) -> Vec<Value> {
        std::fs::read_to_string(repo(rel))
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn real_row(rel: &str, prefix: &str) -> Value {
        real_rows(rel)
            .into_iter()
            .find(|r| r["row_id"].as_str().unwrap().starts_with(prefix))
            .unwrap()
    }

    /// A ledger file under the system temp dir, removed on drop.
    struct Temp(PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    static N: AtomicUsize = AtomicUsize::new(0);
    fn temp_path(stem: &str) -> Temp {
        let n = N.fetch_add(1, Ordering::SeqCst);
        Temp(
            std::env::temp_dir().join(format!("qd-post-f-rules-{}-{n}-{stem}", std::process::id())),
        )
    }
    fn temp_ledger(rows: &[Value]) -> Temp {
        let t = temp_path("ledger.jsonl");
        let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
        std::fs::write(&t.0, text).unwrap();
        t
    }

    fn ft_args(ids: &[&str]) -> Vec<(i64, String)> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| (i as i64, id.to_string()))
            .collect()
    }

    fn outcome(cmd: Cmd) -> Outcome {
        run(&cmd)
    }

    fn with_id(mut row: Value, id: &str) -> Value {
        row["row_id"] = json!(id);
        row
    }

    fn set_count(row: &mut Value, section: &str, name: &str, k: u64, n: u64) {
        let entry = &mut row[section][name];
        assert!(entry.is_object(), "{section}.{name} absent");
        entry["n"] = json!(k);
        entry["n_total"] = json!(n);
        entry["value"] = json!(k as f64 / n as f64);
    }

    /// Set one depth bucket and keep the gate (or control) value its worst bucket.
    fn set_bucket(
        row: &mut Value,
        section: &str,
        name: &str,
        prefix: &str,
        label: &str,
        k: u64,
        n: u64,
    ) {
        set_count(row, "metrics", &format!("{prefix}{label}"), k, n);
        let worst = DEPTH_BUCKETS
            .iter()
            .map(|l| {
                let e = &row["metrics"][format!("{prefix}{l}")];
                (e["n"].as_u64().unwrap(), e["n_total"].as_u64().unwrap())
            })
            .min_by(|a, b| {
                (u128::from(a.0) * u128::from(b.1)).cmp(&(u128::from(b.0) * u128::from(a.1)))
            })
            .unwrap();
        row[section][name]["value"] = json!(worst.0 as f64 / worst.1 as f64);
    }

    const GATE_ID: &str = "aaaaaaaa-0000-4000-8000-000000000001";
    const CONTROL_ID: &str = "aaaaaaaa-0000-4000-8000-000000000002";

    /// J7's ledger plus J4's average gate row and its needle control, copied under new ids and
    /// retagged as avg-np's rows, after `edit` has been applied to them.
    fn avgnp_ledger(edit: impl Fn(&mut Value, &mut Value)) -> Temp {
        let mut rows = real_rows(J7);
        let mut gate = with_id(real_row(J7, "1d93b3ee"), GATE_ID);
        gate["recipe"]["tag"] = json!(AVGNP_GATE_TAG);
        let mut control = with_id(real_row(J7, "0b86fae3"), CONTROL_ID);
        control["recipe"]["tag"] = json!(AVGNP_CONTROL_TAG);
        edit(&mut gate, &mut control);
        rows.push(gate);
        rows.push(control);
        temp_ledger(&rows)
    }

    fn avgnp(ledger: &Temp) -> Outcome {
        outcome(Cmd::Avgnp {
            j7_ledger: ledger.0.clone(),
            j4_ledger: repo(J4),
            out: PathBuf::from("/unused"),
        })
    }

    fn clause<'a>(o: &'a Outcome, c: &str) -> &'a str {
        o.json["detail"]["clauses"][c]["verdict"].as_str().unwrap()
    }

    /// Make the avg-np candidate pass (a) (34 OOD abstentions, all prose) and (b) (every worst
    /// bucket one hit above the average's), leaving (c) as J7g's average had it.
    fn passing(gate: &mut Value, control: &mut Value) {
        set_count(gate, "gates", "ood_abstain", 34, 180);
        set_count(gate, "metrics", "ood_abstain.prose", 34, 60);
        set_count(gate, "metrics", "ood_abstain.scrambled", 0, 60);
        set_count(gate, "metrics", "ood_abstain.unseen-language", 0, 60);
        set_bucket(
            gate,
            "gates",
            "needle_hunk_recall",
            "needle_hunk_recall.depth.",
            "80-100%",
            18,
            61,
        );
        set_bucket(
            control,
            "metrics",
            "needle_hunk_recall.control.1024",
            "needle_hunk_recall.control.1024.depth.",
            "0-20%",
            52,
            57,
        );
        set_bucket(
            control,
            "metrics",
            "needle_hunk_recall.control.2048",
            "needle_hunk_recall.control.2048.depth.",
            "80-100%",
            47,
            60,
        );
        set_bucket(
            control,
            "metrics",
            "needle_hunk_recall.control.4096",
            "needle_hunk_recall.control.4096.depth.",
            "80-100%",
            27,
            61,
        );
    }

    // --- the pre-registration -------------------------------------------------------------

    #[test]
    fn every_constant_is_the_preregistrations_own_text() {
        let prereg =
            std::fs::read_to_string(repo("campaign/f-j7prime-preregistered.json")).unwrap();
        let f_prereg = std::fs::read_to_string(repo("campaign/f-v4-preregistered.json")).unwrap();
        let [(_, c1), (_, c2), (_, c4)] = CONTROL_MIN;
        for phrase in [
            format!(
                "ood_abstain total >= {OOD_MIN}/{OOD_TOTAL} (the fp32 seed minimum, J4 seed1's diagnostic row {})",
                &REF_OOD_SEED_MIN[..8]
            ),
            format!("8K needle worst bucket >= {}", NEEDLE_8K_MIN.text),
            format!(
                "below the average's {} / {} / {} ({}, {})",
                c1.text,
                c2.text,
                c4.text,
                &REF_AVG_GATE[..8],
                &REF_AVG_CONTROL[..8]
            ),
            format!(
                "val_top1.choice >= {} and val_top1.span >= {} (J4 seed minima, {} / {})",
                CHOICE_MIN.text,
                SPAN_MIN.text,
                &REF_CHOICE_MIN[..8],
                &REF_SPAN_MIN[..8]
            ),
            format!(
                "8K needle worst buckets spread by more than {}",
                SPREAD_MAX.text
            ),
            format!(
                "8K worst bucket < {} on >= {NEEDLE_FLOOR_SEEDS} of 3 seeds, or any F seed has prose <= {PROSE_MAX}/{OOD_CATEGORY_TOTAL} or scrambled <= {SCRAMBLED_MAX}/{OOD_CATEGORY_TOTAL}",
                NEEDLE_FLOOR.text
            ),
        ] {
            assert!(
                prereg.contains(&phrase),
                "not in the pre-registration: {phrase}"
            );
        }
        let falsifier = format!(
            "(<= {PROSE_MAX}/{OOD_CATEGORY_TOTAL} prose, <= {SCRAMBLED_MAX}/{OOD_CATEGORY_TOTAL} scrambled)"
        );
        assert!(
            f_prereg.contains(&falsifier),
            "not in F's pre-registration: {falsifier}"
        );
    }

    #[test]
    fn the_reading_is_the_preregistrations_own_text() {
        let prereg =
            std::fs::read_to_string(repo("campaign/f-j7prime-preregistered.json")).unwrap();
        let reading = &serde_json::from_str::<Value>(&prereg).unwrap()["avg_np_qualifies_for_f_j7prime"]
            ["avg_np_reading"]["rule"];
        let reading = reading
            .as_str()
            .expect("avg_np_reading.rule in the pre-registration");
        let ge = "candidate.n * cited.n_total >= cited.n * candidate.n_total";
        for phrase in [
            "the threshold is the cited row's exact count, not the printed decimal",
            ge,
            "a tie passes",
            "(a) is unchanged",
            "report-only",
            "the checker's refusal when a cited row does not round to its decimal stays",
        ] {
            assert!(reading.contains(phrase), "not in avg_np_reading: {phrase}");
        }
        // The JSON every avg-np decision writes names the same comparison.
        let o = avgnp(&avgnp_ledger(passing));
        assert!(o.json["detail"]["reading"].as_str().unwrap().contains(ge));
        assert!(PREREG.contains("54512e6"));
    }

    #[test]
    fn every_decimal_is_the_fraction_its_text_prints() {
        let [(_, c1), (_, c2), (_, c4)] = CONTROL_MIN;
        for d in [
            NEEDLE_8K_MIN,
            c1,
            c2,
            c4,
            CHOICE_MIN,
            SPAN_MIN,
            SPREAD_MAX,
            NEEDLE_FLOOR,
        ] {
            let (whole, frac) = d.text.split_once('.').unwrap();
            assert_eq!(whole, "0", "{}", d.text);
            assert_eq!(d.den, 10u64.pow(frac.len() as u32), "{}", d.text);
            assert_eq!(d.num, frac.parse::<u64>().unwrap(), "{}", d.text);
        }
        assert_eq!(CONTROL_MIN.map(|(l, _)| l), [1024, 2048, 4096]);
    }

    // --- the J4 rows the brief names ------------------------------------------------------

    #[test]
    fn j4s_average_scored_as_avgnp_fails_a_with_b_and_c_passing() {
        let ledger = avgnp_ledger(|_, _| {});
        let o = avgnp(&ledger);
        assert!(!o.refused, "{}", o.json);
        assert_eq!(o.word, "fails");
        assert_eq!(clause(&o, "a"), "fail");
        assert_eq!(o.json["detail"]["clauses"]["a"]["candidate"]["n"], 0);
        // The average ties itself on every part of (b): 17/61 >= 17/61 by its row, which
        // governs, though it is below the printed 0.279 (reported, deciding nothing).
        assert_eq!(clause(&o, "b"), "pass");
        assert_eq!(clause(&o, "c"), "pass");
        let parts = o.json["detail"]["clauses"]["b"]["parts"]
            .as_array()
            .unwrap();
        assert_eq!(parts.len(), 4);
        assert!(parts.iter().all(|p| p["verdict"] == "pass"), "{parts:?}");
        assert_eq!(parts[0]["candidate"]["n"], 17);
        assert_eq!(parts[0]["candidate"]["n_total"], 61);
        assert_eq!(parts[0]["cited_row"]["passes"], true);
        assert_eq!(parts[0]["literal"]["passes"], false);
    }

    #[test]
    fn j4s_seeds_spread_by_0_75_so_seeds_3_and_4_fire() {
        let o = outcome(Cmd::Seeds34 {
            f_ledger: repo(J4),
            ft_rows: ft_args(&J4_FT),
            out: PathBuf::from("/unused"),
        });
        assert!(!o.refused, "{}", o.json);
        assert_eq!(o.word, "fires");
        let d = &o.json["detail"];
        assert_eq!(
            (d["max"]["n"].as_u64(), d["max"]["n_total"].as_u64()),
            (Some(45), Some(60))
        );
        assert_eq!(d["min"]["n"], 0);
        assert!((d["spread"].as_f64().unwrap() - 0.75).abs() < 1e-12);
        let evals: Vec<&str> = d["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["eval_row"].as_str().unwrap())
            .collect();
        assert_eq!(
            evals,
            [
                REF_SPAN_MIN,
                REF_CHOICE_MIN,
                "02cf5ff4-a344-4a0c-afad-6e31d7e5924d"
            ]
        );
    }

    #[test]
    fn j4_fires_j6f_on_the_needle_and_on_both_falsifiers() {
        let o = outcome(Cmd::J6f {
            f_ledger: repo(J4),
            ft_rows: ft_args(&J4_FT),
            out: PathBuf::from("/unused"),
        });
        assert!(!o.refused, "{}", o.json);
        assert_eq!(o.word, "fires");
        let d = &o.json["detail"];
        assert_eq!(d["needle_below_floor_seeds"], json!([0, 1, 2]));
        assert_eq!(d["needle_condition"], true);
        // prose 3 / 9 / 2 and scrambled 3 / 4 / 8 (2f5fe57a, f3612f73, 02cf5ff4).
        assert_eq!(d["prose_falsifier_seeds"], json!([0, 1, 2]));
        assert_eq!(d["scrambled_falsifier_seeds"], json!([0, 1, 2]));
    }

    #[test]
    fn j4s_ft_rows_average_and_their_eval_rows_resolve() {
        let o = outcome(Cmd::FtRows {
            ledger: repo(J4),
            ft_rows: ft_args(&J4_FT),
        });
        assert!(!o.refused, "{}", o.json);
        assert_eq!(o.word, J4_FT.join(" "));
        let o = outcome(Cmd::EvalRow {
            ledger: repo(J4),
            ft_row: (1, J4_FT[1].to_string()),
            tag: EVAL_TAG.to_string(),
        });
        assert_eq!(o.word, REF_CHOICE_MIN);
    }

    // --- boundaries: these pin every comparison's direction and every threshold -------------

    fn f_eval(
        id: &str,
        seed: i64,
        ft: &str,
        worst: (u64, u64),
        prose: u64,
        scrambled: u64,
    ) -> Value {
        let mut metrics = Map::new();
        for (i, label) in DEPTH_BUCKETS.iter().enumerate() {
            let (k, n) = if i == 4 { worst } else { (60, 60) };
            metrics.insert(
                format!("needle_hunk_recall.depth.{label}"),
                json!({"state": "ran", "n": k, "n_total": n, "value": k as f64 / n as f64}),
            );
        }
        for (cat, k) in [("prose", prose), ("scrambled", scrambled)] {
            metrics.insert(
                format!("ood_abstain.{cat}"),
                json!({"state": "ran", "n": k, "n_total": 60, "value": k as f64 / 60.0}),
            );
        }
        metrics.insert("ft_run_row_id".into(), json!({"state": "ran", "value": ft}));
        json!({
            "row_id": id, "run_kind": "eval", "status": "completed", "quick": false,
            "protocol": {"seed": seed}, "recipe": {"tag": EVAL_TAG},
            "gates": {"needle_hunk_recall": {"state": "ran", "n": 300, "n_total": 300, "value": worst.0 as f64 / worst.1 as f64}},
            "metrics": metrics,
        })
    }

    fn f_ft(id: &str, seed: i64) -> Value {
        json!({
            "row_id": id, "run_kind": "ft", "status": "completed", "quick": false,
            "protocol": {"seed": seed, "recipe_hash": "r", "data_snapshot_hash": "d"},
            "recipe": {"tag": "epoch"},
        })
    }

    const FT: [&str; 3] = [
        "f0000000-0000-4000-8000-000000000000",
        "f1000000-0000-4000-8000-000000000000",
        "f2000000-0000-4000-8000-000000000000",
    ];
    const EV: [&str; 3] = [
        "e0000000-0000-4000-8000-000000000000",
        "e1000000-0000-4000-8000-000000000000",
        "e2000000-0000-4000-8000-000000000000",
    ];

    /// F's ledger with these (worst, prose, scrambled) per seed.
    fn f_ledger(seeds: [((u64, u64), u64, u64); 3]) -> Temp {
        let mut rows = Vec::new();
        for (s, (worst, prose, scrambled)) in seeds.into_iter().enumerate() {
            rows.push(f_ft(FT[s], s as i64));
            rows.push(f_eval(EV[s], s as i64, FT[s], worst, prose, scrambled));
        }
        temp_ledger(&rows)
    }

    fn f_rule(ledger: &Temp, j6f: bool) -> Outcome {
        let (f_ledger, ft_rows, out) = (ledger.0.clone(), ft_args(&FT), PathBuf::from("/unused"));
        outcome(if j6f {
            Cmd::J6f {
                f_ledger,
                ft_rows,
                out,
            }
        } else {
            Cmd::Seeds34 {
                f_ledger,
                ft_rows,
                out,
            }
        })
    }

    #[test]
    fn a_spread_of_exactly_0_30_is_quiet_and_one_hit_more_fires() {
        let at = f_ledger([((54, 60), 30, 30), ((50, 60), 30, 30), ((36, 60), 30, 30)]);
        assert_eq!(f_rule(&at, false).word, "quiet");
        let over = f_ledger([((54, 60), 30, 30), ((50, 60), 30, 30), ((35, 60), 30, 30)]);
        assert_eq!(f_rule(&over, false).word, "fires");
        // Unequal denominators: 0.95 - 0.65 = 0.30 exactly (57/60, 13/20) is not more than 0.30.
        let mixed = f_ledger([((57, 60), 30, 30), ((13, 20), 30, 30), ((57, 60), 30, 30)]);
        assert_eq!(f_rule(&mixed, false).word, "quiet");
    }

    #[test]
    fn j6f_counts_a_seed_below_0_95_never_one_at_it_and_needs_two() {
        // 57/60 = 0.95 exactly is not below the floor; OOD far outside J4's envelope.
        let none = f_ledger([((57, 60), 30, 30), ((57, 60), 30, 30), ((57, 60), 30, 30)]);
        assert_eq!(f_rule(&none, true).word, "quiet");
        let one = f_ledger([((56, 60), 30, 30), ((57, 60), 30, 30), ((60, 60), 30, 30)]);
        assert_eq!(f_rule(&one, true).word, "quiet");
        let two = f_ledger([((56, 60), 30, 30), ((56, 60), 30, 30), ((60, 60), 30, 30)]);
        assert_eq!(f_rule(&two, true).word, "fires");
    }

    #[test]
    fn j6f_fires_on_prose_at_9_and_scrambled_at_8_and_not_one_above() {
        let ok = [((60, 60), 30, 30), ((60, 60), 30, 30)];
        let prose_9 = f_ledger([((60, 60), 9, 30), ok[0], ok[1]]);
        assert_eq!(f_rule(&prose_9, true).word, "fires");
        let prose_10 = f_ledger([((60, 60), 10, 30), ok[0], ok[1]]);
        assert_eq!(f_rule(&prose_10, true).word, "quiet");
        let scrambled_8 = f_ledger([ok[0], ok[1], ((60, 60), 30, 8)]);
        assert_eq!(f_rule(&scrambled_8, true).word, "fires");
        let scrambled_9 = f_ledger([ok[0], ok[1], ((60, 60), 30, 9)]);
        assert_eq!(f_rule(&scrambled_9, true).word, "quiet");
    }

    #[test]
    fn avgnp_qualifies_one_hit_above_the_average_everywhere() {
        let ledger = avgnp_ledger(passing);
        let o = avgnp(&ledger);
        assert!(!o.refused, "{}", o.json);
        assert_eq!(o.word, "qualifies", "{}", o.json);
    }

    #[test]
    fn avgnp_needs_34_ood_abstentions_not_33() {
        let ledger = avgnp_ledger(|g, c| {
            passing(g, c);
            set_count(g, "gates", "ood_abstain", 33, 180);
            set_count(g, "metrics", "ood_abstain.prose", 33, 60);
        });
        let o = avgnp(&ledger);
        assert_eq!((o.word.as_str(), clause(&o, "a")), ("fails", "fail"));
    }

    #[test]
    fn a_tie_with_the_cited_row_passes_where_the_decimal_would_fail_it() {
        // (b): 17/61 at 8K, 51/57 at 1K and 46/60 at 2K each tie the cited row and sit below
        // the printed 0.279 / 0.895 / 0.767: the row governs, so each qualifies.
        let gate_8k = |g: &mut Value, k: u64| {
            set_bucket(
                g,
                "gates",
                "needle_hunk_recall",
                "needle_hunk_recall.depth.",
                "80-100%",
                k,
                61,
            )
        };
        let control = |c: &mut Value, length: u32, label: &str, k: u64, n: u64| {
            let name = format!("needle_hunk_recall.control.{length}");
            set_bucket(c, "metrics", &name, &format!("{name}.depth."), label, k, n);
        };
        for (what, ledger) in [
            (
                "8K 17/61",
                avgnp_ledger(|g, c| {
                    passing(g, c);
                    gate_8k(g, 17);
                }),
            ),
            (
                "1K 51/57",
                avgnp_ledger(|g, c| {
                    passing(g, c);
                    control(c, 1024, "0-20%", 51, 57);
                }),
            ),
            (
                "2K 46/60",
                avgnp_ledger(|g, c| {
                    passing(g, c);
                    control(c, 2048, "80-100%", 46, 60);
                }),
            ),
        ] {
            let o = avgnp(&ledger);
            assert!(!o.refused, "{what}: {}", o.json);
            assert_eq!(
                (o.word.as_str(), clause(&o, "b")),
                ("qualifies", "pass"),
                "{what}"
            );
        }
        // One hit below the cited row fails (both renderings agree there).
        let ledger = avgnp_ledger(|g, c| {
            passing(g, c);
            gate_8k(g, 16);
        });
        assert_eq!(avgnp(&ledger).word, "fails");
    }

    #[test]
    fn a_choice_between_the_decimal_and_the_seed_minimum_fails_c() {
        // (c): 8404/10985 = 0.76505 clears the printed 0.765 but is below f3612f73's
        // 8405/10985: the row governs, so it fails. 8405 ties it and passes.
        let ledger = avgnp_ledger(|g, c| {
            passing(g, c);
            set_count(g, "metrics", "val_top1.choice", 8404, 10985);
        });
        let o = avgnp(&ledger);
        assert!(!o.refused, "{}", o.json);
        assert_eq!((o.word.as_str(), clause(&o, "c")), ("fails", "fail"));
        let ledger = avgnp_ledger(|g, c| {
            passing(g, c);
            set_count(g, "metrics", "val_top1.choice", 8405, 10985);
        });
        assert_eq!(avgnp(&ledger).word, "qualifies");
        // Span: 6398/7238 is below the row (6399) and below 0.884 (0.884 x 7238 = 6398.39), so
        // no span value separates the readings; 6399 ties the row and passes.
        let ledger = avgnp_ledger(|g, c| {
            passing(g, c);
            set_count(g, "metrics", "val_top1.span", 6398, 7238);
        });
        assert_eq!(avgnp(&ledger).word, "fails");
        let ledger = avgnp_ledger(|g, c| {
            passing(g, c);
            set_count(g, "metrics", "val_top1.span", 6399, 7238);
        });
        assert_eq!(avgnp(&ledger).word, "qualifies");
    }

    /// Fable's `what_changes` (54512e6), checked by enumeration: over every candidate count the
    /// suites can produce (8K buckets of 59/61/59/60/61, 1K 57/61/59/64/59, 2K 58/61/60/61/60,
    /// 4K 59/59/61/60/61; val 10985 and 7238), the verdict is the row reading, and it differs
    /// from the decimal reading on exactly four values.
    #[test]
    fn the_verdict_is_the_rows_and_differs_from_the_decimals_on_exactly_four_values() {
        let [(_, c1), (_, c2), (_, c4)] = CONTROL_MIN;
        let clauses: [(&str, Dec, Frac, &[u64]); 6] = [
            ("8K", NEEDLE_8K_MIN, Frac { k: 17, n: 61 }, &[59, 61, 60]),
            ("1K", c1, Frac { k: 51, n: 57 }, &[57, 61, 59, 64]),
            ("2K", c2, Frac { k: 46, n: 60 }, &[58, 61, 60]),
            ("4K", c4, Frac { k: 26, n: 61 }, &[59, 61, 60]),
            ("choice", CHOICE_MIN, Frac { k: 8405, n: 10985 }, &[10985]),
            ("span", SPAN_MIN, Frac { k: 6399, n: 7238 }, &[7238]),
        ];
        let mut differ = Vec::new();
        for (what, literal, cited, sizes) in clauses {
            for &n in sizes {
                for k in 0..=n {
                    let candidate = Frac { k, n };
                    let (verdict, json) = at_least(what, candidate, literal, cited, "row").unwrap();
                    let by_row =
                        u128::from(k) * u128::from(cited.n) >= u128::from(cited.k) * u128::from(n);
                    assert_eq!(verdict, Verdict::of(by_row), "{what} {k}/{n}");
                    assert_eq!(json["verdict"], verdict.word());
                    if json["literal"]["passes"] != by_row {
                        differ.push(format!("{what} {k}/{n} {}", verdict.word()));
                    }
                }
            }
        }
        assert_eq!(
            differ,
            [
                "8K 17/61 pass",
                "1K 51/57 pass",
                "2K 46/60 pass",
                "choice 8404/10985 fail"
            ]
        );
    }

    #[test]
    fn a_length_control_below_the_average_at_any_length_fails_b() {
        for (length, label, k, n) in [
            (1024, "0-20%", 50, 57),
            (2048, "80-100%", 45, 60),
            (4096, "80-100%", 25, 61),
        ] {
            let ledger = avgnp_ledger(|g, c| {
                passing(g, c);
                let name = format!("needle_hunk_recall.control.{length}");
                set_bucket(c, "metrics", &name, &format!("{name}.depth."), label, k, n);
            });
            let o = avgnp(&ledger);
            assert_eq!(
                (o.word.as_str(), clause(&o, "b")),
                ("fails", "fail"),
                "{length}"
            );
        }
    }

    // --- refusals -------------------------------------------------------------------------

    #[test]
    fn a_not_run_metric_refuses() {
        let mut rows = Vec::new();
        for s in 0..3 {
            rows.push(f_ft(FT[s], s as i64));
            let mut e = f_eval(EV[s], s as i64, FT[s], (60, 60), 30, 30);
            if s == 1 {
                e["metrics"]["ood_abstain.prose"] =
                    json!({"state": "not_run", "reason": "no suite"});
            }
            rows.push(e);
        }
        let ledger = temp_ledger(&rows);
        let o = f_rule(&ledger, true);
        assert!(o.refused);
        assert!(
            o.json["refused"]
                .as_str()
                .unwrap()
                .contains("not_run (no suite)"),
            "{}",
            o.json
        );
    }

    #[test]
    fn a_missing_or_doubled_eval_row_refuses() {
        let mut rows = vec![f_ft(FT[0], 0), f_ft(FT[1], 1), f_ft(FT[2], 2)];
        rows.push(f_eval(EV[0], 0, FT[0], (60, 60), 30, 30));
        rows.push(f_eval(EV[1], 1, FT[1], (60, 60), 30, 30));
        let missing = temp_ledger(&rows);
        let o = f_rule(&missing, false);
        assert!(
            o.refused
                && o.json["refused"]
                    .as_str()
                    .unwrap()
                    .starts_with("missing row"),
            "{}",
            o.json
        );
        rows.push(f_eval(EV[2], 2, FT[2], (60, 60), 30, 30));
        rows.push(f_eval(
            "e3000000-0000-4000-8000-000000000000",
            2,
            FT[2],
            (50, 60),
            30,
            30,
        ));
        let doubled = temp_ledger(&rows);
        assert!(f_rule(&doubled, false).refused);
        // A killed duplicate is not a candidate.
        let last = rows.len() - 1;
        rows[last]["status"] = json!("killed");
        let killed = temp_ledger(&rows);
        assert_eq!(f_rule(&killed, false).word, "quiet");
    }

    #[test]
    fn a_malformed_line_refuses_even_a_half_written_last_one() {
        let ledger = f_ledger([((60, 60), 30, 30), ((60, 60), 30, 30), ((60, 60), 30, 30)]);
        let mut text = std::fs::read_to_string(&ledger.0).unwrap();
        text.push_str("{\"row_id\": \"e9");
        std::fs::write(&ledger.0, text).unwrap();
        let o = f_rule(&ledger, false);
        assert!(
            o.refused
                && o.json["refused"]
                    .as_str()
                    .unwrap()
                    .contains("malformed ledger line")
        );
        let blank = f_ledger([((60, 60), 30, 30), ((60, 60), 30, 30), ((60, 60), 30, 30)]);
        let text = std::fs::read_to_string(&blank.0)
            .unwrap()
            .replacen('\n', "\n\n", 1);
        std::fs::write(&blank.0, text).unwrap();
        assert!(f_rule(&blank, false).refused);
    }

    #[test]
    fn a_gate_value_that_is_not_its_worst_bucket_refuses() {
        let mut rows = Vec::new();
        for s in 0..3 {
            rows.push(f_ft(FT[s], s as i64));
            let mut e = f_eval(EV[s], s as i64, FT[s], (40, 60), 30, 30);
            if s == 2 {
                e["gates"]["needle_hunk_recall"]["value"] = json!(0.95);
            }
            rows.push(e);
        }
        let ledger = temp_ledger(&rows);
        let o = f_rule(&ledger, false);
        assert!(
            o.refused
                && o.json["refused"]
                    .as_str()
                    .unwrap()
                    .contains("not its worst depth bucket")
        );
    }

    #[test]
    fn a_category_not_out_of_60_refuses() {
        let mut rows = Vec::new();
        for s in 0..3 {
            rows.push(f_ft(FT[s], s as i64));
            let mut e = f_eval(EV[s], s as i64, FT[s], (60, 60), 30, 30);
            if s == 0 {
                e["metrics"]["ood_abstain.scrambled"] =
                    json!({"state": "ran", "n": 8, "n_total": 59, "value": 8.0 / 59.0});
            }
            rows.push(e);
        }
        let ledger = temp_ledger(&rows);
        assert!(f_rule(&ledger, true).refused);
    }

    #[test]
    fn seeds_other_than_fs_three_refuse() {
        let ledger = f_ledger([((60, 60), 30, 30), ((60, 60), 30, 30), ((60, 60), 30, 30)]);
        let o = outcome(Cmd::Seeds34 {
            f_ledger: ledger.0.clone(),
            ft_rows: vec![(0, FT[0].into()), (1, FT[1].into())],
            out: PathBuf::from("/unused"),
        });
        assert!(o.refused);
        let o = outcome(Cmd::J6f {
            f_ledger: ledger.0.clone(),
            ft_rows: vec![(1, FT[0].into()), (0, FT[1].into()), (2, FT[2].into())],
            out: PathBuf::from("/unused"),
        });
        assert!(o.refused);
    }

    #[test]
    fn ft_rows_refuse_a_quick_row_another_recipe_or_a_wrong_seed() {
        let mut rows = vec![f_ft(FT[0], 0), f_ft(FT[1], 1), f_ft(FT[2], 2)];
        let ok = temp_ledger(&rows);
        let o = outcome(Cmd::FtRows {
            ledger: ok.0.clone(),
            ft_rows: ft_args(&FT),
        });
        assert_eq!(o.word, FT.join(" "));
        let o = outcome(Cmd::FtRows {
            ledger: ok.0.clone(),
            ft_rows: vec![(1, FT[0].into())],
        });
        assert!(o.refused);
        rows[1]["quick"] = json!(true);
        let quick = temp_ledger(&rows);
        assert!(
            outcome(Cmd::FtRows {
                ledger: quick.0.clone(),
                ft_rows: ft_args(&FT)
            })
            .refused
        );
        rows[1]["quick"] = json!(false);
        rows[2]["protocol"]["recipe_hash"] = json!("other");
        let other = temp_ledger(&rows);
        assert!(
            outcome(Cmd::FtRows {
                ledger: other.0.clone(),
                ft_rows: ft_args(&FT)
            })
            .refused
        );
    }

    #[test]
    fn a_cited_row_that_no_longer_prints_as_its_decimal_refuses() {
        // Drift J4's choice minimum: f3612f73 at 8300/10985 = 0.756 no longer prints as 0.765.
        let mut rows = real_rows(J4);
        for r in &mut rows {
            if r["row_id"] == REF_CHOICE_MIN {
                set_count(r, "metrics", "val_top1.choice", 8300, 10985);
            }
        }
        let j4 = temp_ledger(&rows);
        let j7 = avgnp_ledger(passing);
        let o = outcome(Cmd::Avgnp {
            j7_ledger: j7.0.clone(),
            j4_ledger: j4.0.clone(),
            out: PathBuf::from("/unused"),
        });
        assert!(
            o.refused
                && o.json["refused"]
                    .as_str()
                    .unwrap()
                    .contains("record and its row disagree"),
            "{}",
            o.json
        );
    }

    #[test]
    fn a_missing_avgnp_control_row_refuses_even_when_a_fails() {
        let mut rows = real_rows(J7);
        let mut gate = with_id(real_row(J7, "1d93b3ee"), GATE_ID);
        gate["recipe"]["tag"] = json!(AVGNP_GATE_TAG);
        rows.push(gate);
        let ledger = temp_ledger(&rows);
        let o = avgnp(&ledger);
        assert!(
            o.refused
                && o.json["refused"]
                    .as_str()
                    .unwrap()
                    .starts_with("missing row"),
            "{}",
            o.json
        );
    }

    #[test]
    fn ft_row_arguments_are_full_ids() {
        assert!(parse_ft_row(&format!("0={}", J4_FT[0])).is_ok());
        assert!(parse_ft_row("0=80b19a41").is_err());
        assert!(parse_ft_row(&format!("x={}", J4_FT[0])).is_err());
        assert!(parse_ft_row(&format!("0={}", J4_FT[0].to_uppercase())).is_err());
        assert!(parse_ft_row(&format!("100={}", J4_FT[0])).is_err());
    }
}
