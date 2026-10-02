//! `qd-post-f-rules` — the four decisions Fable pre-registered for the GPU queue after run F,
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
//! * `successor` — (iv) does F' (fsucc) run, and on which v4 arm's recipe?
//!   (`campaign/f-successor-preregistered.json`, ba1cedb, merged at bcb3c72, R5 struck at
//!   e9cff78: Fable's idle-GPU ruling.) The J6(b) win definition (`f-v4-preregistered.json:13`) with F's seeds 0-2 as the
//!   envelope, applied to J6(f) and J6(d)-v4 separately: an arm wins iff every target lands
//!   outside the envelope in the right direction by more than the envelope's range and no
//!   must-not-lose metric lands outside it in the wrong direction. Words: `fires:j6f`,
//!   `fires:j6dv4`, `quiet`; both arms winning is `refused` (the human decides), as is any row
//!   or metric that cannot be read, and (R9_room, 56d74cf) any target the envelope leaves no
//!   room to clear (`2*max - min >= U`, `2*min - max <= L`), decided from the envelope before
//!   any arm row is read and listed as `cannot_clear`. The paired margin is compared as the
//!   exact fraction `m / n_total` it is (a mean of per-row differences); ECE, which has no
//!   count, on its f64.
//! * `j6a` — (v) does J6(a), +replay on v4, win against F's envelope?
//!   (`campaign/j6a-preregistered.json`, d66f3b0 / 74aea95: Fable's J6(a) ruling, Q4.) The
//!   successor's envelope, comparison and R9_room on J6(a)'s targets (MMLU / CSQA permutation
//!   consistency, higher; in-distribution abstention, lower) and guards, under J6(a)'s own
//!   identity and declared data delta, never R8's. The pre-registration is read at run time: a
//!   structured field that disagrees with this code refuses, and the six values it pins by
//!   amendment (top-level `amendments`) must be present and repeated on the command line.
//!   Words: `wins`, `quiet`, `refused`.
//! * `j6g` — (vi) does J6(g), F plus `--option-permutation-seed 20260919` on v4, win against F's
//!   envelope? (`campaign/j6g-preregistered.json`, dec48d8: Fable's 2026-10-02 optimize ruling,
//!   Q1(b) and Q3.) J6(a)'s targets and guards, with the successor's envelope, comparison and
//!   R9_room, under J6(g)'s own identity: F's data snapshot, and F seed 0's ft recipe (973cd4e3)
//!   plus exactly `option_permutation_seed = 20260919`; any other difference refuses. The
//!   pre-registration is read at run time and must agree with this code, the identity it states
//!   included. Words: `wins`, `quiet`, `refused`.
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
const PREREG_SUCC: &str = "campaign/f-successor-preregistered.json (ba1cedb, merged at bcb3c72; \
                           R5 struck at e9cff78; R9_room at 56d74cf)";
/// The file as registered; the text a decision applied is the one whose sha256 it records.
const PREREG_J6A: &str = "campaign/j6a-preregistered.json (d66f3b0, 74aea95)";
/// A ledger larger than this is not one this repo writes (`qd-gate-report`'s cap).
const MAX_LEDGER_BYTES: u64 = 256 * 1024 * 1024;
/// A pre-registration read at run time is a few KiB; anything past this is not one.
const MAX_PREREG_BYTES: u64 = 1024 * 1024;
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
    /// (iv) Does F' run, and on which v4 arm's recipe? Prints `fires:j6f`, `fires:j6dv4`,
    /// `quiet`, or `refused` (both arms win, anything could not be read, or F's envelope
    /// leaves a target no room to clear: R9_room, listed as `cannot_clear`).
    Successor {
        /// F's ledger: the envelope's rows.
        #[arg(long)]
        f_ledger: PathBuf,
        /// `SEED=FT_ROW_ID` for F's seeds 0, 1 and 2 exactly, in order (the envelope pin).
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
        /// The v4 ablation ledger: both arms' ft and eval rows, J6(f)'s needle-control row and
        /// J6(d)-v4's letter-control row (R5 as struck: J6(d)-v4 reads no length control).
        #[arg(long)]
        arm_ledger: PathBuf,
        /// `0=FT_ROW_ID` of J6(f)'s run (from /home/ubuntu/j6f-v4/train.log).
        #[arg(long, value_parser = parse_ft_row)]
        j6f_ft_row: (i64, String),
        /// `0=FT_ROW_ID` of J6(d)-v4's run (from /home/ubuntu/j6d-v4/train.log).
        #[arg(long, value_parser = parse_ft_row)]
        j6dv4_ft_row: (i64, String),
        #[arg(long)]
        out: PathBuf,
    },
    /// (v) Does J6(a) (+replay) win against F's envelope? Prints `wins`, `quiet` or `refused`
    /// (anything could not be read, an identity or data-delta check failed, the amendments are
    /// pending or disagree with these flags, or a target has no room: R9_room).
    J6a {
        /// campaign/j6a-preregistered.json as committed; read at run time and its sha256
        /// recorded. Its structured fields must agree with this checker, and its top-level
        /// `amendments` must carry the six pins below.
        #[arg(long)]
        preregistration: PathBuf,
        /// F's ledger: the envelope's ft and epoch-score-val rows.
        #[arg(long)]
        f_ledger: PathBuf,
        /// `SEED=FT_ROW_ID` for F's seeds 0, 1 and 2 exactly, in order (the envelope pin).
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
        /// J6(a)'s ledger (arm.ledger): its ft and epoch-score-val rows (R2).
        #[arg(long)]
        arm_ledger: PathBuf,
        /// `0=FT_ROW_ID` of J6(a)'s run.
        #[arg(long, value_parser = parse_ft_row)]
        j6a_ft_row: (i64, String),
        /// Build 2's train-manifest hash; must equal amendments.data_snapshot_hash.
        #[arg(long)]
        data_snapshot_hash: String,
        /// Build 2's train shard hash; must equal amendments.shard_hash.
        #[arg(long)]
        shard_hash: String,
        /// Build 2's replay shard hash; must equal amendments.replay_shard_hash.
        #[arg(long)]
        replay_shard_hash: String,
        /// Decontam 2's attestation sha256; must equal amendments.replay_attestation_sha256.
        #[arg(long)]
        replay_attestation_sha256: String,
        /// h, the distinct replay rows in the full hit list; must equal amendments.h.
        #[arg(long = "h")]
        h: u64,
        /// The full hit list's sha256; must equal amendments.hit_list_sha256.
        #[arg(long)]
        hit_list_sha256: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// (vi) Does J6(g) (F + train-time option permutation) win against F's envelope? Prints
    /// `wins`, `quiet` or `refused` (anything could not be read, an identity, data or
    /// comparability check failed, the pre-registration disagrees with this checker, or a target
    /// has no room: R9_room).
    J6g {
        /// campaign/j6g-preregistered.json as committed; read at run time and its sha256
        /// recorded. Its structured fields and stated identity must agree with this checker.
        #[arg(long)]
        preregistration: PathBuf,
        /// F's ledger: the envelope's ft and epoch-score-val rows.
        #[arg(long)]
        f_ledger: PathBuf,
        /// `SEED=FT_ROW_ID` for F's seeds 0, 1 and 2 exactly, in order (the envelope pin); seed
        /// 0's must be 973cd4e3-e0d2-4ff8-8588-b761cb842b75.
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
        /// J6(g)'s ledger (arm.ledger): its ft and epoch-score-val rows (R2).
        #[arg(long)]
        arm_ledger: PathBuf,
        /// `0=FT_ROW_ID` of J6(g)'s run (from /home/ubuntu/j6g-v4/train.log).
        #[arg(long, value_parser = parse_ft_row)]
        j6g_ft_row: (i64, String),
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
        let (shown, bytes) = read_capped(path, MAX_LEDGER_BYTES, "a ledger")?;
        let rows = parse_rows(&bytes, &shown)?;
        self.0.push(json!({
            "path": shown,
            "sha256": sha256_hex(&bytes),
            "bytes": bytes.len(),
            "rows": rows.len(),
        }));
        Ok(Ledger { shown, rows })
    }
    /// A pre-registration read at run time: its JSON object, with its sha256 recorded, so the
    /// decision names exactly which text it applied.
    fn read_preregistration(&mut self, path: &Path) -> Result<(Map<String, Value>, String)> {
        let (shown, bytes) = read_capped(path, MAX_PREREG_BYTES, "a pre-registration")?;
        let sha256 = sha256_hex(&bytes);
        let parsed: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{shown}: not JSON: {e}"))?;
        let Value::Object(map) = parsed else {
            return Err(format!("{shown}: not a JSON object"));
        };
        self.0.push(json!({
            "path": shown,
            "role": "preregistration",
            "sha256": sha256,
            "bytes": bytes.len(),
        }));
        Ok((map, sha256))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A file's bytes, refused if it is (or grows while read to be) over `cap`.
fn read_capped(path: &Path, cap: u64, what: &str) -> Result<(String, Vec<u8>)> {
    let shown = path.display().to_string();
    let meta = std::fs::metadata(path).map_err(|e| format!("{shown}: {e}"))?;
    ensure!(
        meta.len() <= cap,
        "{shown}: {} bytes is over the {cap}-byte cap for {what}",
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
    Ok((shown, bytes))
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

// --- (iv): successor, campaign/f-successor-preregistered.json ----------------------------------
//
// The J6(b) win definition (f-v4-preregistered.json:13) with F's three seeds as the envelope,
// applied to each v4 arm separately. Each of the lane's readings R1-R8 (the pre-registration's
// `readings`) lives in exactly one place below, named where it is, so an amendment touches one
// item: R1 `Arm::must_not_lose`, R2 `BASE_MUST_NOT_LOSE`, R3 `targets_win`, R4 and R5 (as
// amended at e9cff78) the `guards` of `ARM_J6DV4`, R6 `PERM_DC` / `ID_ABSTAIN_DC`, R7 `clears`
// / `loses`, R8 `arm_identity`, R9_room (56d74cf) `room_bound` / `no_room`, applied in
// `decide_arms` before any arm row is read and refused as (c) in `rule_successor`.

/// The length-control row's tag (`real_ft_run --needle-control`).
const CONTROL_TAG: &str = "epoch-needle-length-control";
/// The letter control's per-family paired margin (`ft_linear_control.py`); J6(d)-v4's target.
const MARGIN_KEY: &str = "paired_margin_vs_linear.choice.code.defect_class";
/// The arms are one run each, seed 0, F seed 0's data order (box_q_j6f.sh, box_q_j6dv4.sh).
const ARM_SEED: i64 = 0;
/// No suite this repo scores has a billion cases; a larger denominator refuses, which keeps
/// every three-way product below far inside i128.
const MAX_DENOMINATOR: u64 = 1_000_000_000;
/// The lead's pin, recorded in every decision's JSON.
const ENVELOPE_PIN: &str = "F seeds 0, 1 and 2 only (the pre-registered F); seeds added by \
                            post-F rule (ii) are not part of the envelope";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Higher,
    Lower,
}

/// Where a metric's value is read, and in which form it is compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// `n / n_total` on the eval row, cross-checked against its value.
    Count(&'static str),
    /// A recorded float on the eval row that has no count (ECE).
    Float(&'static str),
    /// The 8K gate's worst depth bucket on the eval row (`needle_worst`).
    Needle8k,
    /// A length control's worst depth bucket on the needle-control row.
    Control(u32),
    /// The letter-control row's signed paired margin, recovered as `m / n_total`.
    Margin,
}

#[derive(Clone, Copy, Debug)]
struct Metric {
    name: &'static str,
    source: Source,
    dir: Dir,
}

const fn count_metric(name: &'static str, dir: Dir) -> Metric {
    Metric {
        name,
        source: Source::Count(name),
        dir,
    }
}

const fn control_metric(name: &'static str, length: u32) -> Metric {
    Metric {
        name,
        source: Source::Control(length),
        dir: Dir::Higher,
    }
}

/// R2: "defect_class choice/span" is val_top1.choice / val_top1.span; the eval rows carry no
/// per-family top-1 key.
const VAL_CHOICE: Metric = count_metric("val_top1.choice", Dir::Higher);
const VAL_SPAN: Metric = count_metric("val_top1.span", Dir::Higher);
/// R6: "permutation" is the defect_class family's.
const PERM_DC: Metric = count_metric(
    "permutation_consistency.family.code.defect_class",
    Dir::Higher,
);
/// R6: "in-distribution abstention" is the defect_class family's; lower is better.
const ID_ABSTAIN_DC: Metric = count_metric(
    "ood_abstain.in_distribution.family.code.defect_class",
    Dir::Lower,
);
const ECE_DC_KEY: &str = "ece.family.code.defect_class.choice.k4";
const ECE_DC: Metric = Metric {
    name: ECE_DC_KEY,
    source: Source::Float(ECE_DC_KEY),
    dir: Dir::Lower,
};
const NEEDLE_8K_WORST: Metric = Metric {
    name: "needle_8k_worst_bucket",
    source: Source::Needle8k,
    dir: Dir::Higher,
};
const CONTROL_1K: Metric = control_metric("needle_hunk_recall.control.1024", 1024);
const CONTROL_2K: Metric = control_metric("needle_hunk_recall.control.2048", 2048);
const CONTROL_4K: Metric = control_metric("needle_hunk_recall.control.4096", 4096);
const PROSE: Metric = count_metric("ood_abstain.prose", Dir::Higher);
const SCRAMBLED: Metric = count_metric("ood_abstain.scrambled", Dir::Higher);
const UNSEEN: Metric = count_metric("ood_abstain.unseen-language", Dir::Higher);
const MARGIN_DC: Metric = Metric {
    name: MARGIN_KEY,
    source: Source::Margin,
    dir: Dir::Higher,
};

/// The win definition's "gate-bearing metric of the promotion population (defect_class
/// choice/span, defect_class permutation, defect_class ECE, needle)".
const BASE_MUST_NOT_LOSE: [Metric; 5] = [VAL_CHOICE, VAL_SPAN, PERM_DC, ECE_DC, NEEDLE_8K_WORST];

/// One v4 ablation arm: the recipe delta that identifies its ft row (R8), the metrics its flag
/// was meant to move, and Fable's per-arm must-not-lose list.
struct Arm {
    name: &'static str,
    /// The word printed when this arm alone wins.
    word: &'static str,
    /// Its ft recipe is the envelope's seed-0 ft recipe with exactly these keys changed:
    /// `None` = absent, `Some(v)` = the number `v`.
    delta: &'static [(&'static str, Option<f64>)],
    targets: &'static [Metric],
    guards: &'static [Metric],
}

impl Arm {
    /// R1: the base list, then the arm's own, each metric once.
    fn must_not_lose(&self) -> Vec<Metric> {
        let mut out: Vec<Metric> = Vec::new();
        for m in BASE_MUST_NOT_LOSE.iter().chain(self.guards) {
            if !out.iter().any(|o| o.name == m.name) {
                out.push(*m);
            }
        }
        out
    }
    fn reads(&self, pred: impl Fn(Source) -> bool) -> bool {
        self.targets
            .iter()
            .chain(self.must_not_lose().iter())
            .any(|m| pred(m.source))
    }
    fn reads_control(&self) -> bool {
        self.reads(|s| matches!(s, Source::Control(_)))
    }
    fn reads_margin(&self) -> bool {
        self.reads(|s| s == Source::Margin)
    }
}

/// J6(f): no layer-wise LR. Target: the 8K worst bucket alone. Must not lose: the length
/// controls and prose / scrambled OOD.
const ARM_J6F: Arm = Arm {
    name: "j6f",
    word: "fires:j6f",
    delta: &[("lower_layers_n", None), ("lower_lr_scale", None)],
    targets: &[NEEDLE_8K_WORST],
    guards: &[CONTROL_1K, CONTROL_2K, CONTROL_4K, PROSE, SCRAMBLED],
};

/// J6(d)-v4: --lr 3e-5 --beta2 0.95. Targets: the defect_class paired margin and val top-1
/// choice / span. Must not lose: OOD (R4: all three categories), needle, permutation (both in
/// the base list) and in-distribution abstention. R5 as amended (Fable, e9cff78): "needle" is
/// the gate-bearing 8K worst bucket only, so this arm neither guards on nor reads the 1K/2K/4K
/// length controls, and its needle-control row is not required.
const ARM_J6DV4: Arm = Arm {
    name: "j6dv4",
    word: "fires:j6dv4",
    delta: &[("lr", Some(3e-5)), ("beta2", Some(0.95))],
    targets: &[MARGIN_DC, VAL_CHOICE, VAL_SPAN],
    guards: &[PROSE, SCRAMBLED, UNSEEN, ID_ABSTAIN_DC],
};

/// A signed exact fraction `num / den`, `0 < den <= MAX_DENOMINATOR`, `|num| <= den`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rat {
    num: i64,
    den: i64,
}

impl Rat {
    fn new(num: i64, den: u64, what: &str) -> Result<Rat> {
        ensure!(
            den > 0 && den <= MAX_DENOMINATOR,
            "{what}: denominator {den} is outside 1..={MAX_DENOMINATOR}"
        );
        let den = i64::try_from(den).map_err(|_| format!("{what}: denominator {den} overflows"))?;
        ensure!(
            num.abs() <= den,
            "{what}: {num}/{den} is not a fraction in [-1, 1]"
        );
        Ok(Rat { num, den })
    }
    fn of_count(f: Frac, what: &str) -> Result<Rat> {
        let num = i64::try_from(f.k).map_err(|_| format!("{what}: count {} overflows", f.k))?;
        Rat::new(num, f.n, what)
    }
    fn f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
    fn parts(self) -> (i128, i128) {
        (i128::from(self.num), i128::from(self.den))
    }
}

/// A metric's value in the form it is compared in.
#[derive(Clone, Copy, Debug)]
enum Val {
    Exact(Rat),
    Float(f64),
}

impl Val {
    fn json(self) -> Value {
        match self {
            Val::Exact(r) => json!({"n": r.num, "n_total": r.den, "value": r.f64()}),
            Val::Float(v) => json!({"value": v}),
        }
    }
}

fn mixed(a: Val, b: Val) -> String {
    format!("cannot compare {a:?} with {b:?}: one metric read in two forms")
}

fn cmp_val(a: Val, b: Val) -> Result<std::cmp::Ordering> {
    match (a, b) {
        (Val::Exact(x), Val::Exact(y)) => {
            let ((xn, xd), (yn, yd)) = (x.parts(), y.parts());
            Ok((xn * yd).cmp(&(yn * xd)))
        }
        (Val::Float(x), Val::Float(y)) => x
            .partial_cmp(&y)
            .ok_or_else(|| format!("{x} and {y} do not compare")),
        _ => Err(mixed(a, b)),
    }
}

/// R7, targets: the candidate lands outside the envelope in the right direction by MORE than
/// the envelope's own range. Higher: `c - max > max - min`; lower: `min - c > max - min`. A
/// candidate at exactly max + range (min - range) does not clear. Exact for counts and
/// margins: with c = a/b, min = e/f, max = g/h, higher clears iff a*f*h + e*b*h > 2*g*b*f and
/// lower iff 2*e*b*h > a*f*h + g*b*f.
fn clears(c: Val, min: Val, max: Val, dir: Dir) -> Result<bool> {
    match (c, min, max) {
        (Val::Exact(c), Val::Exact(lo), Val::Exact(hi)) => {
            let ((a, b), (e, f), (g, h)) = (c.parts(), lo.parts(), hi.parts());
            Ok(match dir {
                Dir::Higher => a * f * h + e * b * h > 2 * g * b * f,
                Dir::Lower => 2 * e * b * h > a * f * h + g * b * f,
            })
        }
        (Val::Float(c), Val::Float(lo), Val::Float(hi)) => Ok(match dir {
            Dir::Higher => c - hi > hi - lo,
            Dir::Lower => lo - c > hi - lo,
        }),
        _ => Err(mixed(c, max)),
    }
}

/// R7, must-not-lose: the candidate lands outside the envelope in the wrong direction,
/// strictly. Higher: `c < min`; lower: `c > max`. Equal to the bound is inside.
fn loses(c: Val, min: Val, max: Val, dir: Dir) -> Result<bool> {
    Ok(match dir {
        Dir::Higher => cmp_val(c, min)? == std::cmp::Ordering::Less,
        Dir::Lower => cmp_val(c, max)? == std::cmp::Ordering::Greater,
    })
}

/// R3: an arm's targets win only if every one of them clears.
fn targets_win(cleared: &[bool]) -> bool {
    !cleared.is_empty() && cleared.iter().all(|&c| c)
}

/// The bound a target's metric cannot pass (R9_room): `U` above a higher-better target, `L`
/// below a lower-better one.
#[derive(Clone, Copy, Debug)]
struct Bound {
    name: &'static str,
    value: Rat,
}

impl Bound {
    fn json(self) -> Value {
        json!({
            "name": self.name,
            "n": self.value.num,
            "n_total": self.value.den,
            "value": self.value.f64(),
        })
    }
}

/// R9_room's bounds (56d74cf): U = 1 for a higher-better count (a worst depth bucket, val
/// top-1) and for the paired margin; L = 0 for a lower-better count. Nothing else is
/// pre-registered, so a float target or a lower-better margin refuses rather than defaults.
fn room_bound(m: Metric) -> Result<Bound> {
    match (m.source, m.dir) {
        (
            Source::Count(_) | Source::Needle8k | Source::Control(_) | Source::Margin,
            Dir::Higher,
        ) => Ok(Bound {
            name: "U",
            value: Rat { num: 1, den: 1 },
        }),
        (Source::Count(_) | Source::Needle8k | Source::Control(_), Dir::Lower) => Ok(Bound {
            name: "L",
            value: Rat { num: 0, den: 1 },
        }),
        (Source::Margin, Dir::Lower) | (Source::Float(_), _) => Err(format!(
            "target {}: R9_room pre-registers no bound for a {:?} {:?} target (U = 1 for \
             higher-better counts and the paired margin, L = 0 for lower-better counts)",
            m.name, m.source, m.dir
        )),
    }
}

/// R9_room: the envelope leaves a target no room, so no candidate can clear it. Higher-better
/// with bound U: `2*max - min >= U`; lower-better with bound L: `2*min - max <= L`. Exact: with
/// min = e/f, max = g/h and the bound u/v, higher iff `(2*g*f - e*h)*v >= u*f*h` and lower iff
/// `(2*e*h - g*f)*v <= u*f*h`, which for counts (U = 1, L = 0) are the pre-registration's
/// `2*g*f - e*h >= f*h` and `2*e*h <= g*f`. It reads the envelope alone; `clears`, the
/// comparison, is unchanged.
fn no_room(min: Val, max: Val, bound: Rat, dir: Dir) -> Result<bool> {
    match (min, max) {
        (Val::Exact(lo), Val::Exact(hi)) => {
            let ((e, f), (g, h), (u, v)) = (lo.parts(), hi.parts(), bound.parts());
            Ok(match dir {
                Dir::Higher => (2 * g * f - e * h) * v >= u * f * h,
                Dir::Lower => (2 * e * h - g * f) * v <= u * f * h,
            })
        }
        _ => Err(format!(
            "R9_room is pre-registered on exact fractions, not on {min:?} / {max:?}"
        )),
    }
}

fn text(v: Val) -> String {
    match v {
        Val::Exact(r) if r.den == 1 => r.num.to_string(),
        Val::Exact(r) => format!("{}/{}", r.num, r.den),
        Val::Float(x) => x.to_string(),
    }
}

/// R9_room for one target of one arm, decided from F's envelope alone.
struct Room {
    arm: &'static str,
    metric: Metric,
    /// The envelope's min and max, the bound, and whether they leave no room; or why the
    /// envelope's value could not be read, which is itself a refusal (outcomes.refused (b)).
    decided: Result<(Val, Val, Bound, bool)>,
}

impl Room {
    fn of(arm: &Arm, m: Metric, envelope: &[SeedRows]) -> Room {
        let decide = || -> Result<(Val, Val, Bound, bool)> {
            let bound = room_bound(m)?;
            let (_, min, max) = envelope_range(m, envelope)?;
            Ok((min, max, bound, no_room(min, max, bound.value, m.dir)?))
        };
        Room {
            arm: arm.name,
            metric: m,
            decided: decide(),
        }
    }
    fn json(&self) -> Value {
        let mut j = json!({"arm": self.arm, "target": self.metric.name});
        match &self.decided {
            Ok((min, max, bound, blocked)) => {
                j["direction"] = json!(dir_word(self.metric.dir));
                j["min"] = min.json();
                j["max"] = max.json();
                j["bound"] = bound.json();
                j["room"] = json!(!blocked);
            }
            Err(e) => {
                j["room"] = Value::Null;
                j["not_decided"] = json!(e);
            }
        }
        j
    }
    /// The pre-registration's `{arm, target, max, min, bound}`, for a target with no room.
    fn cannot_clear(&self) -> Option<Value> {
        match &self.decided {
            Ok((min, max, bound, true)) => Some(json!({
                "arm": self.arm,
                "target": self.metric.name,
                "max": max.json(),
                "min": min.json(),
                "bound": bound.json(),
            })),
            _ => None,
        }
    }
    /// Why this target refuses the decision, if it does: no room, or unreadable, each under the
    /// label its pre-registration gives it.
    fn refusal(&self, labels: Labels) -> Option<String> {
        match &self.decided {
            Ok((min, max, bound, true)) => {
                let (form, beyond) = match self.metric.dir {
                    Dir::Higher => ("2*max - min >=", "above max + range"),
                    Dir::Lower => ("2*min - max <=", "below min - range"),
                };
                Some(format!(
                    "{} {} target {} cannot clear: F's envelope leaves it no room (R9_room): \
                     {}-better with max {} and min {} has {form} {} = {}, so no candidate can \
                     land {beyond}; a target that could not be examined is not one that was \
                     examined and missed",
                    labels.no_room,
                    self.arm,
                    self.metric.name,
                    dir_word(self.metric.dir),
                    text(*max),
                    text(*min),
                    bound.name,
                    text(Val::Exact(bound.value)),
                ))
            }
            Ok(_) => None,
            Err(e) => Some(format!("{} {e}", labels.unreadable)),
        }
    }
}

/// How a pre-registration labels its refusals: a required row or value that could not be read
/// (or an identity, data-delta or comparability check that failed), and a target its envelope
/// leaves no room (R9_room). The successor's outcomes.refused calls them (b) and (c); J6(a)'s,
/// (a) and (b).
#[derive(Clone, Copy, Debug)]
struct Labels {
    unreadable: &'static str,
    no_room: &'static str,
}

const SUCC_LABELS: Labels = Labels {
    unreadable: "(b)",
    no_room: "(c)",
};

fn dir_word(dir: Dir) -> &'static str {
    match dir {
        Dir::Higher => "higher",
        Dir::Lower => "lower",
    }
}

/// The signed paired margin `mean(model_correct - control_correct)` over `n_total` rows
/// (eval_harness.paired_margin_test), recovered as the exact fraction `m / n_total`. A value
/// that is not `m / n_total` for an integer m refuses.
fn paired_margin(row: &Row) -> Result<Rat> {
    let what = format!("row {} metrics.{MARGIN_KEY}", row.id());
    let entry = row.ran("metrics", MARGIN_KEY)?;
    let rows = entry
        .get("n_total")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{what}: no integer n_total"))?;
    ensure!(
        entry.get("n").and_then(Value::as_u64) == Some(rows),
        "{what}: n is not n_total {rows} (a paired margin is over every row)"
    );
    let value = recorded_value(entry, &what)?;
    ensure!(
        (-1.0..=1.0).contains(&value),
        "{what}: {value} is not a mean of per-row differences in [-1, 1]"
    );
    ensure!(
        rows > 0 && rows <= MAX_DENOMINATOR,
        "{what}: n_total {rows} is outside 1..={MAX_DENOMINATOR}"
    );
    let scaled = (value * rows as f64).round();
    // |scaled| <= rows <= 1e9, so the conversion below is exact.
    let m = scaled as i64;
    let exact = Rat::new(m, rows, &what)?;
    ensure!(
        close(value, exact.f64()),
        "{what}: value {value} is not m/{rows} for any integer m (nearest {m}/{rows}); a \
         paired margin over {rows} rows is"
    );
    Ok(exact)
}

/// One seed's rows: its ft row, its eval row, and the needle-control and letter-control rows
/// when an arm reads them.
struct SeedRows<'a> {
    seed: i64,
    ft: &'a Row,
    eval: &'a Row,
    control: Option<&'a Row>,
    letter: Option<&'a Row>,
}

impl SeedRows<'_> {
    fn json(&self) -> Value {
        json!({
            "seed": self.seed,
            "ft_row": self.ft.id(),
            "eval_row": self.eval.id(),
            "needle_control_row": self.control.map(Row::id),
            "letter_control_row": self.letter.map(Row::id),
        })
    }
    fn read(&self, m: Metric) -> Result<Val> {
        let what = format!("row {} {}", self.eval.id(), m.name);
        Ok(match m.source {
            Source::Count(key) => {
                Val::Exact(Rat::of_count(self.eval.count("metrics", key)?, &what)?)
            }
            Source::Float(key) => {
                let entry = self.eval.ran("metrics", key)?;
                Val::Float(recorded_value(entry, &what)?)
            }
            Source::Needle8k => Val::Exact(Rat::of_count(needle_8k(self.eval)?.frac, &what)?),
            Source::Control(length) => {
                let row = self
                    .control
                    .ok_or_else(|| format!("{what}: no needle-control row was looked up"))?;
                Val::Exact(Rat::of_count(needle_control(row, length)?.frac, &what)?)
            }
            Source::Margin => {
                let row = self
                    .letter
                    .ok_or_else(|| format!("{what}: no letter-control row was looked up"))?;
                Val::Exact(paired_margin(row)?)
            }
        })
    }
}

/// The one completed letter-control row of an eval row: it names the eval row in
/// `metrics.scored_eval_row_id` and carries the per-family margin. An older letter-control
/// row without per-family margins (J4's d597ee7d) is not one; the option control records
/// `linear_option_control.scored_eval_row_id` and is never one.
fn letter_control_row<'a>(ledger: &'a Ledger, eval: &Row) -> Result<&'a Row> {
    let found: Vec<&Row> = ledger
        .rows
        .iter()
        .filter(|r| {
            r.completed()
                && r.str_at(&["run_kind"]) == Some("eval")
                && r.str_at(&["metrics", "scored_eval_row_id", "value"]) == Some(eval.id())
                && r.get(&["metrics", MARGIN_KEY]).is_some()
        })
        .collect();
    let row = match found.len() {
        1 => found[0],
        0 => {
            return Err(format!(
                "missing row: no completed letter-control row carrying {MARGIN_KEY} for eval \
                 row {} in {}",
                eval.id(),
                ledger.shown
            ));
        }
        n => {
            return Err(format!(
                "{n} completed letter-control rows carry {MARGIN_KEY} for eval row {} in {} \
                 ({}); which one decides is not this tool's call",
                eval.id(),
                ledger.shown,
                found.iter().map(|r| r.id()).collect::<Vec<_>>().join(", ")
            ));
        }
    };
    ensure!(
        row.str_at(&["recipe", "eval_row_id"]) == Some(eval.id()),
        "letter-control row {}: recipe.eval_row_id {:?} is not its scored eval row {}",
        row.id(),
        row.str_at(&["recipe", "eval_row_id"]),
        eval.id()
    );
    Ok(row)
}

fn seed_rows<'a>(
    ledger: &'a Ledger,
    seed: i64,
    ft: &str,
    control: bool,
    letter: bool,
) -> Result<SeedRows<'a>> {
    let ft_row = ft_row(ledger, seed, ft)?;
    let eval = eval_row_of(ledger, seed, ft, EVAL_TAG)?;
    let control = if control {
        Some(eval_row_of(ledger, seed, ft, CONTROL_TAG)?)
    } else {
        None
    };
    let letter = if letter {
        Some(letter_control_row(ledger, eval)?)
    } else {
        None
    };
    Ok(SeedRows {
        seed,
        ft: ft_row,
        eval,
        control,
        letter,
    })
}

/// `row.recipe` agrees with `reference.recipe` on every key in `keys`, each of which the
/// reference records (an absent key on both sides would otherwise pass unexamined).
fn same_recipe_keys(row: &Row, reference: &Row, keys: &[&str], what: &str) -> Result<()> {
    for key in keys {
        let want = reference.get(&["recipe", key]).ok_or_else(|| {
            format!(
                "{what}: reference row {} has no recipe.{key}",
                reference.id()
            )
        })?;
        ensure!(
            row.get(&["recipe", key]) == Some(want),
            "{what}: row {} recipe.{key} is {}, not reference row {}'s {want}: not comparable",
            row.id(),
            row.get(&["recipe", key])
                .map_or("absent".to_string(), Value::to_string),
            reference.id()
        );
    }
    Ok(())
}

fn recipe_of<'a>(row: &'a Row, what: &str) -> Result<&'a Map<String, Value>> {
    row.get(&["recipe"])
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{what}: row {} has no recipe object", row.id()))
}

/// Every row an arm or envelope seed is read from is comparable with the envelope's seed 0:
/// the same val set and needle / OOD suites, the same length-control suite, and the same
/// letter control (its recipe but for its own eval row id).
fn comparable(rows: &SeedRows, reference: &SeedRows, what: &str) -> Result<()> {
    same_recipe_keys(
        rows.eval,
        reference.eval,
        &["val_shard_hash", "needle", "ood"],
        what,
    )?;
    if let (Some(c), Some(r)) = (rows.control, reference.control) {
        same_recipe_keys(c, r, &["needle_control"], what)?;
    }
    if let (Some(l), Some(r)) = (rows.letter, reference.letter) {
        let (mut a, mut b) = (recipe_of(l, what)?.clone(), recipe_of(r, what)?.clone());
        a.remove("eval_row_id");
        b.remove("eval_row_id");
        ensure!(
            a == b,
            "{what}: letter-control row {}'s recipe is not reference row {}'s (but for \
             eval_row_id): not the same control",
            l.id(),
            r.id()
        );
    }
    Ok(())
}

/// R8: the arm's ft row is the declared ablation of the envelope's recipe: the same data
/// snapshot, and the same recipe but for exactly the arm's delta. Swapped or wrong ft ids
/// refuse rather than name the wrong recipe. Returns the delta it checked, for the JSON.
fn arm_identity(arm: &Arm, ft: &Row, reference: &Row) -> Result<Value> {
    let what = format!("arm {} ft row {}", arm.name, ft.id());
    let data = ft.str_at(&["protocol", "data_snapshot_hash"]);
    ensure!(
        data.is_some() && data == reference.str_at(&["protocol", "data_snapshot_hash"]),
        "{what}: data snapshot {data:?} is not the envelope's {:?}",
        reference.str_at(&["protocol", "data_snapshot_hash"])
    );
    let (got, base) = (recipe_of(ft, &what)?, recipe_of(reference, &what)?);
    for (key, want) in arm.delta {
        match want {
            None => ensure!(
                !got.contains_key(*key),
                "{what}: recipe.{key} is {}, but the arm drops it",
                got[*key]
            ),
            Some(v) => ensure!(
                got.get(*key).and_then(Value::as_f64) == Some(*v),
                "{what}: recipe.{key} is {}, not the arm's {v}",
                got.get(*key).map_or("absent".to_string(), Value::to_string)
            ),
        }
    }
    let changed: BTreeSet<&str> = arm.delta.iter().map(|(k, _)| *k).collect();
    let keys: BTreeSet<&String> = got.keys().chain(base.keys()).collect();
    for key in keys {
        if changed.contains(key.as_str()) {
            continue;
        }
        ensure!(
            got.get(key) == base.get(key),
            "{what}: recipe.{key} is {}, not the envelope's {}; the arm differs from F only by \
             {:?}",
            got.get(key).map_or("absent".to_string(), Value::to_string),
            base.get(key).map_or("absent".to_string(), Value::to_string),
            changed
        );
    }
    Ok(Value::Object(
        arm.delta
            .iter()
            .map(|(k, v)| (k.to_string(), v.map_or(Value::Null, |x| json!(x))))
            .collect(),
    ))
}

/// How an arm's ft row is identified against the envelope's seed-0 ft row: `arm_identity`
/// (R8) for the successor's arms, J6(a)'s own (`replay_identity`) for j6a. It returns the JSON
/// record of what it checked.
type Identity<'a> = &'a dyn Fn(&Arm, &Row, &Row) -> Result<Value>;

/// A metric's value on each envelope seed, and their minimum and maximum.
fn envelope_range(m: Metric, envelope: &[SeedRows]) -> Result<(Vec<Val>, Val, Val)> {
    let values: Vec<Val> = envelope.iter().map(|s| s.read(m)).collect::<Result<_>>()?;
    let first = *values
        .first()
        .ok_or_else(|| format!("{}: an empty envelope has no range", m.name))?;
    let (mut min, mut max) = (first, first);
    for &v in &values[1..] {
        if cmp_val(v, min)? == std::cmp::Ordering::Less {
            min = v;
        }
        if cmp_val(v, max)? == std::cmp::Ordering::Greater {
            max = v;
        }
    }
    Ok((values, min, max))
}

/// One metric against the envelope: a target (does it clear?) or a guard (does it lose?).
fn judge(
    m: Metric,
    envelope: &[SeedRows],
    candidate: &SeedRows,
    target: bool,
) -> Result<(bool, Value)> {
    let (values, min, max) = envelope_range(m, envelope)?;
    let c = candidate.read(m)?;
    let verdict = if target {
        clears(c, min, max, m.dir)?
    } else {
        loses(c, min, max, m.dir)?
    };
    let mut j = json!({
        "metric": m.name,
        "direction": dir_word(m.dir),
        "envelope": envelope
            .iter()
            .zip(&values)
            .map(|(s, v)| json!({"seed": s.seed, "value": v.json()}))
            .collect::<Vec<_>>(),
        "min": min.json(),
        "max": max.json(),
        "candidate": c.json(),
    });
    j[if target { "clears" } else { "loses" }] = Value::Bool(verdict);
    Ok((verdict, j))
}

/// What `decide_arms` decided: R9_room for every target of every arm, from the envelope before
/// any arm row was read; then the arms judged, or the refusal met while reading their rows.
/// The second is carried as data, not returned as an error, so an arm row that cannot be read
/// never hides a target that has no room.
struct Decided {
    envelope: Value,
    room: Vec<Room>,
    /// Each arm's verdict and the JSON of every comparison, keyed by arm name.
    judged: Result<(Vec<bool>, Value)>,
}

/// F's envelope resolved and checked (an error here means there is no envelope, so no room was
/// decided), R9_room decided for every target from it alone, then each arm judged against it.
/// Any row or metric that cannot be read refuses the whole decision.
fn decide_arms(
    inputs: &mut Inputs,
    envelope_ledger: &Path,
    envelope_ft: &[(i64, String)],
    arm_ledger: &Path,
    arms: &[(&Arm, &(i64, String))],
    identity: Identity,
) -> Result<Decided> {
    let seeds: Vec<i64> = envelope_ft.iter().map(|(s, _)| *s).collect();
    ensure!(
        seeds == F_SEEDS,
        "the envelope is {ENVELOPE_PIN}, given as seeds {F_SEEDS:?} in that order; got {seeds:?}"
    );
    let env = inputs.read(envelope_ledger)?;
    one_configuration(&env, envelope_ft)?;
    let control = arms.iter().any(|(a, _)| a.reads_control());
    let letter = arms.iter().any(|(a, _)| a.reads_margin());
    let envelope: Vec<SeedRows> = envelope_ft
        .iter()
        .map(|(seed, ft)| seed_rows(&env, *seed, ft, control, letter))
        .collect::<Result<_>>()?;
    let reference = &envelope[0];
    for s in &envelope[1..] {
        comparable(s, reference, &format!("envelope seed {}", s.seed))?;
    }
    // R9_room: "decided from the envelope alone, before the arm's rows are read".
    let room: Vec<Room> = arms
        .iter()
        .flat_map(|(arm, _)| arm.targets.iter().map(|m| Room::of(arm, *m, &envelope)))
        .collect();
    let judged = judge_arms(inputs, arm_ledger, &envelope, arms, identity);
    Ok(Decided {
        envelope: json!({
            "pin": ENVELOPE_PIN,
            "seeds": envelope.iter().map(SeedRows::json).collect::<Vec<_>>(),
        }),
        room,
        judged,
    })
}

/// Each arm's rows read from `arm_ledger` and judged against the envelope: whether it wins,
/// and the JSON of every comparison.
fn judge_arms(
    inputs: &mut Inputs,
    arm_ledger: &Path,
    envelope: &[SeedRows],
    arms: &[(&Arm, &(i64, String))],
    identity: Identity,
) -> Result<(Vec<bool>, Value)> {
    let arm_rows = inputs.read(arm_ledger)?;
    let reference = envelope
        .first()
        .ok_or_else(|| "an empty envelope".to_string())?;
    let mut wins = Vec::new();
    let mut detail = Map::new();
    for (arm, (seed, ft)) in arms {
        ensure!(
            *seed == ARM_SEED,
            "arm {}: seed {seed}, but each arm is one seed-{ARM_SEED} run",
            arm.name
        );
        let rows = seed_rows(
            &arm_rows,
            *seed,
            ft,
            arm.reads_control(),
            arm.reads_margin(),
        )?;
        let delta = identity(arm, rows.ft, reference.ft)?;
        comparable(&rows, reference, &format!("arm {}", arm.name))?;
        let mut cleared = Vec::new();
        let mut targets = Vec::new();
        for m in arm.targets {
            let (c, j) = judge(*m, envelope, &rows, true)?;
            cleared.push(c);
            targets.push(j);
        }
        let mut lost = Vec::new();
        let mut guards = Vec::new();
        for m in arm.must_not_lose() {
            let (l, j) = judge(m, envelope, &rows, false)?;
            if l {
                lost.push(m.name);
            }
            guards.push(j);
        }
        let win = targets_win(&cleared) && lost.is_empty();
        wins.push(win);
        detail.insert(
            arm.name.to_string(),
            json!({
                "wins": win,
                "rows": rows.json(),
                "recipe_delta": delta,
                "targets_all_clear": targets_win(&cleared),
                "targets": targets,
                "must_not_lose_lost": lost,
                "must_not_lose": guards,
            }),
        );
    }
    Ok((wins, Value::Object(detail)))
}

/// A decision's JSON body, every refusal met so far in the order it was met (each once), and
/// the arms' verdicts when they were judged.
struct Settled {
    because: Vec<String>,
    body: Value,
    wins: Option<Vec<bool>>,
}

impl Settled {
    fn refuse(&mut self, reason: String) {
        if !self.because.contains(&reason) {
            self.because.push(reason);
        }
    }
    /// The word and what follows it, or `refused` with every reason listed: any refusal wins
    /// over any word, so a target that could not be examined never reads as one that was.
    fn finish(mut self, word: Option<(&str, &str)>, refused_then: &str) -> Result<(String, Value)> {
        if !self.because.is_empty() {
            self.body["refused_because"] = json!(self.because);
            self.body["then"] = json!(refused_then);
            return Ok(("refused".to_string(), self.body));
        }
        let (word, then) =
            word.ok_or_else(|| "the decision has neither a word nor a refusal".to_string())?;
        self.body["then"] = json!(then);
        Ok((word.to_string(), self.body))
    }
}

/// The refusal when F's envelope does not resolve: there is no envelope, so no room was decided.
fn no_envelope(labels: Labels, e: &str) -> String {
    format!(
        "{} {e}; room not decided: F's envelope did not resolve, so no target was examined for \
         room (R9_room) and no arm was judged",
        labels.unreadable
    )
}

/// The decided envelope, room and arms as a decision's body, with the refusals they carry in
/// the order they were met: room first (no room, or an envelope value that could not be read),
/// both decided from the envelope before any arm row was read; then the arm phase's.
fn settle(decided: Decided, labels: Labels, rule: &str) -> Settled {
    let mut s = Settled {
        because: Vec::new(),
        body: json!({
            "rule": rule,
            "envelope": decided.envelope,
            "room": decided.room.iter().map(Room::json).collect::<Vec<_>>(),
            "cannot_clear": decided.room.iter().filter_map(Room::cannot_clear).collect::<Vec<_>>(),
        }),
        wins: None,
    };
    for reason in decided.room.iter().filter_map(|r| r.refusal(labels)) {
        s.refuse(reason);
    }
    match decided.judged {
        Err(e) => s.refuse(format!("{} {e}", labels.unreadable)),
        Ok((wins, arms)) => {
            s.body["arms"] = arms;
            s.wins = Some(wins);
        }
    }
    s
}

/// The successor decision. Every applicable refusal is listed in `refused_because`, in the
/// order it was met: (c) a target with no room and (b) an envelope value that could not be
/// read, both decided from the envelope first; then (b) an arm row or metric that could not be
/// read; then (a) both arms winning. Any of them refuses, so a target that could not be
/// examined never reads as quiet or as the other arm firing.
fn rule_successor(
    inputs: &mut Inputs,
    f_ledger: &Path,
    ft_rows: &[(i64, String)],
    arm_ledger: &Path,
    j6f: &(i64, String),
    j6dv4: &(i64, String),
) -> Result<(String, Value)> {
    let decided = decide_arms(
        inputs,
        f_ledger,
        ft_rows,
        arm_ledger,
        &[(&ARM_J6F, j6f), (&ARM_J6DV4, j6dv4)],
        &arm_identity,
    )
    .map_err(|e| no_envelope(SUCC_LABELS, &e))?;
    let mut s = settle(
        decided,
        SUCC_LABELS,
        "an arm wins iff every metric its flag was meant to move lands outside F's three-seed \
         envelope in the right direction by more than F's own seed range, and no must-not-lose \
         metric lands outside the envelope in the wrong direction (f-v4-preregistered.json:13); \
         F' runs the one winning arm's recipe. A target the envelope leaves no room to clear \
         refuses the decision (R9_room).",
    );
    let word = match s.wins.as_deref() {
        None => None,
        Some(&[true, false]) => Some((
            ARM_J6F.word,
            "F' runs J6(f)'s recipe: F's argv without --lower-layers-n 8 --lower-layers-lr-scale 0.1, seeds 0 1 2",
        )),
        Some(&[false, true]) => Some((
            ARM_J6DV4.word,
            "F' runs J6(d)-v4's recipe: F's argv with --lr 3e-5 --beta2 0.95 in place of --lr 1e-5, seeds 0 1 2",
        )),
        Some(&[false, false]) => Some((
            "quiet",
            "no F'; re-plan from J7''s avg / ens3 / avg-np rows (Fable Q2)",
        )),
        Some(&[true, true]) => {
            s.refuse(
                "(a) both arms win, on different metrics (J6(f) on the 8K needle, J6(d)-v4 on \
                 the margin and val top-1): the human decides; two one-seed wins are never \
                 combined into an untested recipe"
                    .to_string(),
            );
            None
        }
        Some(other) => return Err(format!("{} arm verdicts for two arms", other.len())),
    };
    s.finish(word, "no F' until the human decides")
}

// --- (v): j6a, campaign/j6a-preregistered.json -------------------------------------------------
//
// J6(a) (+replay on v4, seed 0, quick) read against F's seeds 0-2 with the successor's envelope,
// comparison and R9_room (`decide_arms`, `settle`), under its own identity: J6(a) trains on
// build 2, so R8's snapshot equality (`arm_identity`) cannot hold for it and is not loosened;
// `replay_identity` checks J6(a)'s declared data delta instead. The pre-registration is read at
// run time: its structured fields must agree with this code (a disagreement refuses), and the
// six values it pins by amendment are read from its top-level `amendments`.

/// The pre-registration's own words for the outcome.
const J6A_WORDS: [&str; 3] = ["wins", "quiet", "refused"];
/// J6(a)'s outcomes.refused calls a row / identity / data-delta failure (a), no room (b).
const J6A_LABELS: Labels = Labels {
    unreadable: "(a)",
    no_room: "(b)",
};
/// arm.identity.ft_row: J6(a)'s ft row's code commit, a502670, as F's 973cd4e3 records it.
const J6A_CODE_COMMIT: &str = "a5026707b6e3e57c253be32003ba32420f2d78e2";
/// arm.replay_flags (Fable Q2): the unit weight, every 6th micro-batch, base to model.
const REPLAY_WEIGHT: f64 = 1.0;
const REPLAY_EVERY: i64 = 6;
const REPLAY_DIRECTION: &str = "base_to_model";
/// The five keys a502670 records for a replay run (tools/real_ft_run.py:359-363).
const REPLAY_KEYS: [&str; 5] = [
    "replay_shard_hash",
    "replay_attestation_sha256",
    "replay_weight",
    "replay_every",
    "replay_direction",
];
/// R1_derived_recipe_keys: derived by the batch planner from the train set; recorded and
/// reported, not compared.
const DERIVED_RECIPE_KEYS: [&str; 2] = ["batches", "width"];
/// declared_data_delta: the replay set's 3,568 row_ids, of which h are excluded, so h <= 3,568.
const REPLAY_SET_ROWS: u64 = 3568;
/// The six values `amendments` must pin, in the pre-registration's order.
const AMENDMENT_KEYS: [&str; 6] = [
    "data_snapshot_hash",
    "shard_hash",
    "replay_shard_hash",
    "replay_attestation_sha256",
    "h",
    "hit_list_sha256",
];

const PERM_KNOWLEDGE: Metric = count_metric(
    "permutation_consistency.family.knowledge.multiple_choice",
    Dir::Higher,
);
const PERM_COMMONSENSE: Metric = count_metric(
    "permutation_consistency.family.commonsense.multiple_choice",
    Dir::Higher,
);
const ID_ABSTAIN_KNOWLEDGE: Metric = count_metric(
    "ood_abstain.in_distribution.family.knowledge.multiple_choice",
    Dir::Lower,
);
const ID_ABSTAIN_COMMONSENSE: Metric = count_metric(
    "ood_abstain.in_distribution.family.commonsense.multiple_choice",
    Dir::Lower,
);

/// J6(a): +replay. Targets: MMLU / CSQA permutation consistency (higher) and in-distribution
/// abstention (lower). Guards: the successor's base list, then defect_class in-distribution
/// abstention and the three OOD categories. It reads no needle-control or letter-control row.
/// Its identity is `replay_identity`, not a delta (`delta` is empty, so `arm_identity` would
/// refuse its rows).
const ARM_J6A: Arm = Arm {
    name: "j6a",
    word: "wins",
    delta: &[],
    targets: &[
        PERM_KNOWLEDGE,
        PERM_COMMONSENSE,
        ID_ABSTAIN_KNOWLEDGE,
        ID_ABSTAIN_COMMONSENSE,
    ],
    guards: &[ID_ABSTAIN_DC, PROSE, SCRAMBLED, UNSEEN],
};

/// The six values J6(a)'s pre-registration pins by amendment after build 2 and decontam 2.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pins {
    data_snapshot_hash: String,
    shard_hash: String,
    replay_shard_hash: String,
    replay_attestation_sha256: String,
    h: u64,
    hit_list_sha256: String,
}

impl Pins {
    fn json(&self) -> Value {
        json!({
            "data_snapshot_hash": self.data_snapshot_hash,
            "shard_hash": self.shard_hash,
            "replay_shard_hash": self.replay_shard_hash,
            "replay_attestation_sha256": self.replay_attestation_sha256,
            "h": self.h,
            "hit_list_sha256": self.hit_list_sha256,
        })
    }
    /// Each hash a sha256 in lower-case hex, and h at most the replay set's 3,568 rows.
    fn well_formed(&self, whose: &str) -> Result<()> {
        for (key, v) in [
            ("data_snapshot_hash", &self.data_snapshot_hash),
            ("shard_hash", &self.shard_hash),
            ("replay_shard_hash", &self.replay_shard_hash),
            ("replay_attestation_sha256", &self.replay_attestation_sha256),
            ("hit_list_sha256", &self.hit_list_sha256),
        ] {
            ensure!(
                v.len() == 64 && v.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                "{whose} {key} {v:?} is not a sha256 in lower-case hex"
            );
        }
        ensure!(
            self.h <= REPLAY_SET_ROWS,
            "{whose} h {} is more than the replay set's {REPLAY_SET_ROWS} rows",
            self.h
        );
        Ok(())
    }
}

/// The pins the pre-registration's top-level `amendments` carries: exactly the six keys, each
/// well-formed. Absent is pending; any other shape refuses.
fn amendments_of(p: &Map<String, Value>) -> Result<Pins> {
    let form = "an object with exactly data_snapshot_hash, shard_hash, replay_shard_hash, \
                replay_attestation_sha256 and hit_list_sha256 (each a sha256 in lower-case hex) \
                and h (an integer, 0..=3568)";
    let Some(a) = p.get("amendments") else {
        return Err(format!(
            "the pre-registration's amendments are pending (no top-level \"amendments\"; \
             amendments_pending: {}); the checker refuses until they are filled as {form}",
            p.get("amendments_pending")
                .map_or("absent".to_string(), Value::to_string)
        ));
    };
    let a = a
        .as_object()
        .ok_or_else(|| format!("the pre-registration's amendments are {a}, not {form}"))?;
    let extra: Vec<&String> = a
        .keys()
        .filter(|k| !AMENDMENT_KEYS.contains(&k.as_str()))
        .collect();
    ensure!(
        extra.is_empty(),
        "the pre-registration's amendments carry {extra:?}, which are not among the six; \
         amendments is {form}"
    );
    let hash = |key: &str| -> Result<String> {
        a.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                format!(
                    "the pre-registration's amendments have no string {key}; amendments is {form}"
                )
            })
    };
    let pins = Pins {
        data_snapshot_hash: hash("data_snapshot_hash")?,
        shard_hash: hash("shard_hash")?,
        replay_shard_hash: hash("replay_shard_hash")?,
        replay_attestation_sha256: hash("replay_attestation_sha256")?,
        h: a.get("h").and_then(Value::as_u64).ok_or_else(|| {
            format!("the pre-registration's amendments have no integer h; amendments is {form}")
        })?,
        hit_list_sha256: hash("hit_list_sha256")?,
    };
    pins.well_formed("the pre-registration's amendments:")?;
    Ok(pins)
}

/// The pins the pre-registration fixes, which the command line must repeat exactly. h and the
/// hit list's sha256 are checked against the file only: no ledger row records them.
fn pinned(p: &Map<String, Value>, cli: &Pins) -> Result<Pins> {
    let filed = amendments_of(p)?;
    cli.well_formed("the command line's")?;
    let (f, c) = (filed.json(), cli.json());
    for key in AMENDMENT_KEYS {
        ensure!(
            f[key] == c[key],
            "--{} {} is not the pre-registration's pinned {}",
            key.replace('_', "-"),
            c[key],
            f[key]
        );
    }
    Ok(filed)
}

/// A metric as the pre-registration lists it: name, direction and form.
fn listed(m: Metric) -> (&'static str, &'static str, &'static str) {
    let form = match m.source {
        Source::Count(_) => "count",
        Source::Float(_) => "f64",
        Source::Needle8k => "needle_worst_bucket",
        Source::Control(_) => "control",
        Source::Margin => "paired_margin",
    };
    (m.name, dir_word(m.dir), form)
}

/// The pre-registration's outcome words and its arm's targets and guards (name, direction and
/// form, in order) agree with this code's `words` and `checked`. Returns the arm object, for the
/// fields a rule checks on its own. Shared by j6a and j6g.
fn arm_agrees<'p>(
    p: &'p Map<String, Value>,
    want_words: &[&str],
    checked: &Arm,
) -> Result<&'p Map<String, Value>> {
    let words: Vec<&str> = p
        .get("outcomes")
        .and_then(|o| o.get("words"))
        .and_then(Value::as_array)
        .map(|w| w.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    ensure!(
        words == want_words,
        "the pre-registration's outcomes.words are {words:?}, not this checker's {want_words:?}"
    );
    let arm = p
        .get("arm")
        .and_then(Value::as_object)
        .ok_or("the pre-registration has no arm object")?;
    for (list, want) in [
        ("targets", checked.targets.to_vec()),
        ("guards", checked.must_not_lose()),
    ] {
        let got: Vec<(&str, &str, &str)> = arm
            .get(list)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("the pre-registration has no arm.{list} list"))?
            .iter()
            .map(|m| {
                let form = m["form"].as_str().unwrap_or("");
                // "needle_worst_bucket (campaign/f-successor-preregistered.json ...)" names
                // its form, then where it is defined.
                let form = form.split(' ').next().unwrap_or("");
                (
                    m["name"].as_str().unwrap_or(""),
                    m["direction"].as_str().unwrap_or(""),
                    form,
                )
            })
            .collect();
        let want: Vec<(&str, &str, &str)> = want.into_iter().map(listed).collect();
        ensure!(
            got == want,
            "the pre-registration's arm.{list} are {got:?}, not this checker's {want:?}"
        );
    }
    Ok(arm)
}

/// The pre-registration's structured fields agree with this code: its words, its targets and
/// guards (name, direction and form, in order) and its replay flags. A disagreement refuses: the
/// file is what binds, and a checker applying anything else applies a rule nobody registered.
fn j6a_agrees(p: &Map<String, Value>) -> Result<()> {
    let arm = arm_agrees(p, &J6A_WORDS, &ARM_J6A)?;
    let flags = arm.get("replay_flags").unwrap_or(&Value::Null);
    ensure!(
        flags["replay_weight"].as_f64() == Some(REPLAY_WEIGHT)
            && flags["replay_every"].as_i64() == Some(REPLAY_EVERY)
            && flags["replay_direction"].as_str() == Some(REPLAY_DIRECTION),
        "the pre-registration's arm.replay_flags (weight {}, every {}, direction {}) are not \
         this checker's ({REPLAY_WEIGHT}, {REPLAY_EVERY}, {REPLAY_DIRECTION})",
        flags["replay_weight"],
        flags["replay_every"],
        flags["replay_direction"]
    );
    Ok(())
}

/// J6(a)'s own identity (arm.identity.ft_row and .recipe), not R8's. The ft row is quick (one
/// seed, by construction), at a502670, on build 2's snapshot, which is pinned and is not F's;
/// its recipe is F seed 0's except shard_hash (pinned), exactly the five replay keys (pinned or
/// registered values, none of them on F's recipe), and R1's derived keys, which are recorded
/// and not compared. Returns what it checked, for the JSON.
fn replay_identity(arm: &Arm, ft: &Row, reference: &Row, pins: &Pins) -> Result<Value> {
    let what = format!("arm {} ft row {}", arm.name, ft.id());
    ensure!(
        ft.get(&["quick"]) == Some(&Value::Bool(true)),
        "{what}: quick is {}, but J6(a) is one seed and quick by construction",
        ft.get(&["quick"])
            .map_or("absent".to_string(), Value::to_string)
    );
    ensure!(
        ft.str_at(&["code_commit"]) == Some(J6A_CODE_COMMIT),
        "{what}: code_commit {:?} is not {J6A_CODE_COMMIT} (a502670, as F's 973cd4e3)",
        ft.str_at(&["code_commit"])
    );
    let data = ft.str_at(&["protocol", "data_snapshot_hash"]);
    let f_data = reference.str_at(&["protocol", "data_snapshot_hash"]);
    ensure!(
        data == Some(pins.data_snapshot_hash.as_str()),
        "{what}: data snapshot {data:?} is not build 2's pinned {}",
        pins.data_snapshot_hash
    );
    ensure!(
        f_data.is_some() && data != f_data,
        "{what}: data snapshot {data:?} is F's ({f_data:?}): J6(a) trains on build 2, whose train \
         set is F's minus the replay set (declared_data_delta)"
    );
    let (got, base) = (recipe_of(ft, &what)?, recipe_of(reference, &what)?);
    ensure!(
        got.get("tag").and_then(Value::as_str) == Some("epoch"),
        "{what}: recipe.tag is {}, not epoch",
        got.get("tag")
            .map_or("absent".to_string(), Value::to_string)
    );
    ensure!(
        got.get("shard_hash").and_then(Value::as_str) == Some(pins.shard_hash.as_str()),
        "{what}: recipe.shard_hash is {}, not build 2's pinned {}",
        got.get("shard_hash")
            .map_or("absent".to_string(), Value::to_string),
        pins.shard_hash
    );
    for key in REPLAY_KEYS {
        ensure!(
            !base.contains_key(key),
            "envelope ft row {}: recipe.{key} is {}, but F ran no replay",
            reference.id(),
            base[key]
        );
        let v = got.get(key);
        let ok = match key {
            "replay_shard_hash" => v.and_then(Value::as_str) == Some(&pins.replay_shard_hash),
            "replay_attestation_sha256" => {
                v.and_then(Value::as_str) == Some(&pins.replay_attestation_sha256)
            }
            "replay_weight" => v.and_then(Value::as_f64) == Some(REPLAY_WEIGHT),
            "replay_every" => v.and_then(Value::as_i64) == Some(REPLAY_EVERY),
            "replay_direction" => v.and_then(Value::as_str) == Some(REPLAY_DIRECTION),
            _ => false,
        };
        ensure!(
            ok,
            "{what}: recipe.{key} is {}, not the registered or pinned value",
            v.map_or("absent".to_string(), Value::to_string)
        );
    }
    let declared: BTreeSet<&str> = REPLAY_KEYS
        .iter()
        .chain(&DERIVED_RECIPE_KEYS)
        .chain(&["shard_hash"])
        .copied()
        .collect();
    let keys: BTreeSet<&String> = got.keys().chain(base.keys()).collect();
    for key in keys {
        if declared.contains(key.as_str()) {
            continue;
        }
        ensure!(
            got.get(key) == base.get(key),
            "{what}: recipe.{key} is {}, not F seed 0's {}; J6(a) differs from F's recipe only \
             by {declared:?} (arm.identity.recipe)",
            got.get(key).map_or("absent".to_string(), Value::to_string),
            base.get(key).map_or("absent".to_string(), Value::to_string)
        );
    }
    Ok(json!({
        "identity": "J6(a)'s own (campaign/j6a-preregistered.json arm.identity), not R8",
        "quick": true,
        "code_commit": J6A_CODE_COMMIT,
        "data_snapshot_hash": data,
        "f_data_snapshot_hash": f_data,
        "shard_hash": pins.shard_hash,
        "f_shard_hash": base.get("shard_hash"),
        "replay": REPLAY_KEYS
            .iter()
            .map(|k| (k.to_string(), got.get(*k).cloned().unwrap_or(Value::Null)))
            .collect::<Map<_, _>>(),
        "derived_recorded_not_compared": DERIVED_RECIPE_KEYS
            .iter()
            .map(|k| (k.to_string(), json!({"j6a": got.get(*k), "f": base.get(*k)})))
            .collect::<Map<_, _>>(),
    }))
}

/// The J6(a) decision: `wins`, `quiet` or `refused`, every refusal listed in the order it was
/// met. The pre-registration is read and checked first (a disagreement refuses before any
/// ledger is read); pins that are pending or disagree with the command line refuse as (a), and
/// room is still decided from the envelope and reported.
fn rule_j6a(
    inputs: &mut Inputs,
    preregistration: &Path,
    f_ledger: &Path,
    ft_rows: &[(i64, String)],
    arm_ledger: &Path,
    j6a: &(i64, String),
    cli: &Pins,
) -> Result<(String, Value)> {
    let (prereg, prereg_sha256) = inputs.read_preregistration(preregistration)?;
    j6a_agrees(&prereg).map_err(|e| format!("{} {e}", J6A_LABELS.unreadable))?;
    let pins = pinned(&prereg, cli);
    let identity = |arm: &Arm, ft: &Row, reference: &Row| -> Result<Value> {
        let pins = pins.as_ref().map_err(Clone::clone)?;
        replay_identity(arm, ft, reference, pins)
    };
    let decided = decide_arms(
        inputs,
        f_ledger,
        ft_rows,
        arm_ledger,
        &[(&ARM_J6A, j6a)],
        &identity,
    )
    .map_err(|e| match &pins {
        // No envelope, so nothing else is decided, but pins that are pending or disagree are
        // a reason of their own and are listed beside it.
        Err(p) => format!(
            "{}; {} {p}",
            no_envelope(J6A_LABELS, &e),
            J6A_LABELS.unreadable
        ),
        Ok(_) => no_envelope(J6A_LABELS, &e),
    })?;
    let mut s = settle(
        decided,
        J6A_LABELS,
        "J6(a) wins iff every target (MMLU / CSQA permutation consistency, higher; MMLU / CSQA \
         in-distribution abstention, lower) lands outside F's three-seed envelope in the right \
         direction by more than F's own seed range, and no guard lands outside it in the wrong \
         direction (campaign/j6a-preregistered.json, f-v4-preregistered.json:13). A target the \
         envelope leaves no room to clear refuses the decision (R9_room). 'wins' feeds the next \
         re-plan only.",
    );
    if let Err(e) = &pins {
        s.refuse(format!("{} {e}", J6A_LABELS.unreadable));
    }
    s.body["preregistration_sha256"] = json!(prereg_sha256);
    s.body["amendments"] = match &pins {
        Ok(p) => json!({
            "pinned": p.json(),
            "h_and_hit_list_checked_against": "the pre-registration only; no ledger row records them",
        }),
        Err(e) => json!({"not_pinned": e}),
    };
    let word = match s.wins.as_deref() {
        None => None,
        Some(&[true]) => Some((
            ARM_J6A.word,
            "J6(a) wins: this feeds the next re-plan only and starts nothing (one quick seed; \
             rule 8)",
        )),
        Some(&[false]) => Some((
            "quiet",
            "J6(a) does not win: a target did not clear or a guard lost; the next re-plan reads \
             the rows",
        )),
        Some(other) => return Err(format!("{} arm verdicts for one arm", other.len())),
    };
    s.finish(word, "no reading; the human decides from the rows")
}

// --- (vi): j6g, campaign/j6g-preregistered.json ------------------------------------------------
//
// J6(g) (F plus --option-permutation-seed 20260919 on v4, seed 0, quick) read against F's seeds
// 0-2 with the successor's envelope, comparison and R9_room (`decide_arms`, `settle`), on J6(a)'s
// targets and guards, under its own identity (`option_identity`): F's data snapshot (R8's
// equality holds, since J6(g) changes no data), and F seed 0's ft recipe plus exactly one key.
// The pre-registration is read at run time: its structured fields and the identity it states
// must agree with this code, and a disagreement refuses (R1: an amendment that admits more
// recipe keys refuses here until the checker follows it).

/// The file as registered; the text a decision applied is the one whose sha256 it records.
const PREREG_J6G: &str = "campaign/j6g-preregistered.json (dec48d8)";
/// The pre-registration's own words for the outcome.
const J6G_WORDS: [&str; 3] = ["wins", "quiet", "refused"];
/// J6(g)'s outcomes.refused calls a row / identity / data / comparability failure (a), no room (b).
const J6G_LABELS: Labels = Labels {
    unreadable: "(a)",
    no_room: "(b)",
};
/// arm.identity.ft_row: code_commit a502670, as F's 973cd4e3 (the commit J6(a) also runs at).
const J6G_CODE_COMMIT: &str = J6A_CODE_COMMIT;
/// arm.identity.recipe: "except exactly one added key: option_permutation_seed = 20260919".
const OPTION_PERMUTATION_KEY: &str = "option_permutation_seed";
const OPTION_PERMUTATION_SEED: i64 = 20_260_919;
/// arm.identity.recipe: "Equal to F seed 0's ft recipe (973cd4e3)". The envelope's seed-0 ft row
/// must be this row, so a wrong F ledger cannot supply the recipe J6(g) is compared with.
const J6G_F_SEED0_FT: &str = "973cd4e3-e0d2-4ff8-8588-b761cb842b75";
/// arm.identity.ft_row: "protocol.data_snapshot_hash equal to F's ea3215c4...".
const J6G_F_DATA_SNAPSHOT_HASH: &str =
    "ea3215c4f36d57f74d291fb94c3fa8724fa5a14a303ea7572aa0dafb2a0933a1";
/// arm.identity.eval_row: "recipe.val_shard_hash = ef06ab99... (F's)".
const J6G_F_VAL_SHARD_HASH: &str =
    "ef06ab99eef107dd608424c80117b81b3986ce5ea827d2d4f47f990d960b1bc5";
/// arm.identity.recipe: "batches (9683) and width (7936)", F seed 0's; R1 compares them like
/// every other key, since J6(g) changes no data.
const J6G_F_BATCHES: i64 = 9683;
const J6G_F_WIDTH: i64 = 7936;

/// J6(g): F plus train-time option permutation. J6(a)'s four targets (MMLU / CSQA permutation
/// consistency, higher; in-distribution abstention, lower) and its guards: the successor's base
/// list, then defect_class in-distribution abstention and the three OOD categories. It reads no
/// needle-control or letter-control row. Its identity is `option_identity`, not a delta.
const ARM_J6G: Arm = Arm {
    name: "j6g",
    word: "wins",
    delta: &[],
    targets: &[
        PERM_KNOWLEDGE,
        PERM_COMMONSENSE,
        ID_ABSTAIN_KNOWLEDGE,
        ID_ABSTAIN_COMMONSENSE,
    ],
    guards: &[ID_ABSTAIN_DC, PROSE, SCRAMBLED, UNSEEN],
};

/// The identity this checker applies, as the pre-registration states it: (arm.identity key, a
/// phrase its text must hold). An amended identity changes the text and refuses until the
/// checker follows it.
fn j6g_identity_text() -> [(&'static str, String); 7] {
    [
        ("ft_row", format!("code_commit {J6G_CODE_COMMIT}")),
        ("ft_row", "quick true".to_string()),
        (
            "ft_row",
            format!("protocol.data_snapshot_hash equal to F's {J6G_F_DATA_SNAPSHOT_HASH}"),
        ),
        (
            "recipe",
            "Equal to F seed 0's ft recipe (973cd4e3), including shard_hash, batches (9683) and \
             width (7936)"
                .to_string(),
        ),
        (
            "recipe",
            format!(
                "except exactly one added key: {OPTION_PERMUTATION_KEY} = \
                 {OPTION_PERMUTATION_SEED}. Any other difference refuses."
            ),
        ),
        ("eval_row", "tag epoch-score-val".to_string()),
        (
            "eval_row",
            format!("recipe.val_shard_hash = {J6G_F_VAL_SHARD_HASH}"),
        ),
    ]
}

/// The pre-registration's structured fields agree with this code: its words, its targets and
/// guards (name, direction and form, in order), the option permutation's seed, and the identity
/// it states (`j6g_identity_text`). A disagreement refuses before any ledger is read.
fn j6g_agrees(p: &Map<String, Value>) -> Result<()> {
    let arm = arm_agrees(p, &J6G_WORDS, &ARM_J6G)?;
    let seed = arm.get("option_permutation").and_then(|o| o.get("seed"));
    ensure!(
        seed.and_then(Value::as_i64) == Some(OPTION_PERMUTATION_SEED),
        "the pre-registration's arm.option_permutation.seed is {}, not this checker's \
         {OPTION_PERMUTATION_SEED}",
        seed.map_or("absent".to_string(), Value::to_string)
    );
    let identity = arm
        .get("identity")
        .and_then(Value::as_object)
        .ok_or("the pre-registration has no arm.identity object")?;
    for (key, phrase) in j6g_identity_text() {
        let text = identity.get(key).and_then(Value::as_str).unwrap_or("");
        ensure!(
            text.contains(&phrase),
            "the pre-registration's arm.identity.{key} does not say {phrase:?}, which is the \
             identity this checker applies"
        );
    }
    Ok(())
}

/// J6(g)'s own identity (arm.identity.ft_row and .recipe): the ft row is quick (one seed, by
/// construction), at a502670, on F's data snapshot, and its recipe is F seed 0's (973cd4e3) plus
/// exactly `option_permutation_seed = 20260919`; F's own recipe must not carry that key. Every
/// protocol key but recipe_hash (a hash of the recipe, which differs by that key) equals F seed
/// 0's: any data difference refuses (declared_data_delta). Every failure is listed, not only the
/// first, so F's own row read as J6(g) names the missing key beside `quick`. Returns what it
/// checked, for the JSON.
fn option_identity(arm: &Arm, ft: &Row, reference: &Row) -> Result<Value> {
    let what = format!("arm {} ft row {}", arm.name, ft.id());
    let shown = |v: Option<&Value>| v.map_or("absent".to_string(), Value::to_string);
    let mut wrong: Vec<String> = Vec::new();
    if reference.id() != J6G_F_SEED0_FT {
        wrong.push(format!(
            "the envelope's seed-0 ft row is {}, not F seed 0's {J6G_F_SEED0_FT}, whose recipe \
             arm.identity.recipe names",
            reference.id()
        ));
    }
    if ft.get(&["quick"]) != Some(&Value::Bool(true)) {
        wrong.push(format!(
            "quick is {}, but J6(g) is one seed and quick by construction",
            shown(ft.get(&["quick"]))
        ));
    }
    if ft.str_at(&["code_commit"]) != Some(J6G_CODE_COMMIT) {
        wrong.push(format!(
            "code_commit {:?} is not {J6G_CODE_COMMIT} (a502670, as F's 973cd4e3)",
            ft.str_at(&["code_commit"])
        ));
    }
    let data = ft.str_at(&["protocol", "data_snapshot_hash"]);
    if data != Some(J6G_F_DATA_SNAPSHOT_HASH) {
        wrong.push(format!(
            "data snapshot {data:?} is not F's {J6G_F_DATA_SNAPSHOT_HASH}: J6(g) trains on F's \
             data (declared_data_delta: none)"
        ));
    }
    match (
        ft.get(&["protocol"]).and_then(Value::as_object),
        reference.get(&["protocol"]).and_then(Value::as_object),
    ) {
        (Some(got), Some(base)) => {
            let keys: BTreeSet<&String> = got.keys().chain(base.keys()).collect();
            for key in keys {
                if key != "recipe_hash" && got.get(key) != base.get(key) {
                    wrong.push(format!(
                        "protocol.{key} is {}, not F seed 0's {}",
                        shown(got.get(key)),
                        shown(base.get(key))
                    ));
                }
            }
        }
        _ => wrong.push("it or F seed 0's ft row has no protocol object".to_string()),
    }
    let (got, base) = (recipe_of(ft, &what)?, recipe_of(reference, &what)?);
    if let Some(v) = base.get(OPTION_PERMUTATION_KEY) {
        wrong.push(format!(
            "envelope ft row {}: recipe.{OPTION_PERMUTATION_KEY} is {v}, but F trained without \
             option permutation",
            reference.id()
        ));
    }
    match got.get(OPTION_PERMUTATION_KEY) {
        Some(v) if v.as_i64() == Some(OPTION_PERMUTATION_SEED) => {}
        v => wrong.push(format!(
            "recipe.{OPTION_PERMUTATION_KEY} is {}, not the pre-registered \
             {OPTION_PERMUTATION_SEED}: J6(g)'s one added key (arm.identity.recipe)",
            shown(v)
        )),
    }
    for (key, want) in [("batches", J6G_F_BATCHES), ("width", J6G_F_WIDTH)] {
        if base.get(key).and_then(Value::as_i64) != Some(want) {
            wrong.push(format!(
                "envelope ft row {}: recipe.{key} is {}, not F's {want} (arm.identity.recipe)",
                reference.id(),
                shown(base.get(key))
            ));
        }
    }
    let keys: BTreeSet<&String> = got.keys().chain(base.keys()).collect();
    for key in keys {
        if key != OPTION_PERMUTATION_KEY && got.get(key) != base.get(key) {
            wrong.push(format!(
                "recipe.{key} is {}, not F seed 0's {}; J6(g) differs from F's recipe only by \
                 {OPTION_PERMUTATION_KEY} (arm.identity.recipe)",
                shown(got.get(key)),
                shown(base.get(key))
            ));
        }
    }
    ensure!(wrong.is_empty(), "{what}: {}", wrong.join("; "));
    Ok(json!({
        "identity": "J6(g)'s own (campaign/j6g-preregistered.json arm.identity): F's data, and F \
                     seed 0's ft recipe plus exactly one key",
        "f_seed0_ft_row": reference.id(),
        "quick": true,
        "code_commit": J6G_CODE_COMMIT,
        "data_snapshot_hash": data,
        "added": {OPTION_PERMUTATION_KEY: OPTION_PERMUTATION_SEED},
        "compared_equal": got
            .keys()
            .filter(|k| k.as_str() != OPTION_PERMUTATION_KEY)
            .collect::<Vec<_>>(),
    }))
}

/// The J6(g) decision: `wins`, `quiet` or `refused`, every refusal listed in the order it was
/// met. The pre-registration is read and checked first (a disagreement refuses before any
/// ledger is read); then room from F's envelope, then J6(g)'s rows.
fn rule_j6g(
    inputs: &mut Inputs,
    preregistration: &Path,
    f_ledger: &Path,
    ft_rows: &[(i64, String)],
    arm_ledger: &Path,
    j6g: &(i64, String),
) -> Result<(String, Value)> {
    let (prereg, prereg_sha256) = inputs.read_preregistration(preregistration)?;
    j6g_agrees(&prereg).map_err(|e| format!("{} {e}", J6G_LABELS.unreadable))?;
    let decided = decide_arms(
        inputs,
        f_ledger,
        ft_rows,
        arm_ledger,
        &[(&ARM_J6G, j6g)],
        &option_identity,
    )
    .map_err(|e| no_envelope(J6G_LABELS, &e))?;
    let mut s = settle(
        decided,
        J6G_LABELS,
        "J6(g) wins iff every target (MMLU / CSQA permutation consistency, higher; MMLU / CSQA \
         in-distribution abstention, lower) lands outside F's three-seed envelope in the right \
         direction by more than F's own seed range, and no guard lands outside it in the wrong \
         direction (campaign/j6g-preregistered.json, as campaign/j6a-preregistered.json and \
         f-v4-preregistered.json:13). A target the envelope leaves no room to clear refuses the \
         decision (R9_room). 'wins' feeds the next re-plan only.",
    );
    s.body["preregistration_sha256"] = json!(prereg_sha256);
    let word = match s.wins.as_deref() {
        None => None,
        Some(&[true]) => Some((
            ARM_J6G.word,
            "J6(g) wins: this feeds the next re-plan only and starts nothing (one quick seed; \
             rule 8)",
        )),
        Some(&[false]) => Some((
            "quiet",
            "J6(g) does not win: a target did not clear or a guard lost; the next re-plan reads \
             the rows",
        )),
        Some(other) => return Err(format!("{} arm verdicts for one arm", other.len())),
    };
    s.finish(word, "no reading; the human decides from the rows")
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
    let (_, recipe, data) = one_configuration(&ledger, ft_rows)?;
    let rows: Vec<Value> = ft_rows
        .iter()
        .map(|(seed, id)| json!({"seed": seed, "ft_row": id}))
        .collect();
    let ids: Vec<&str> = ft_rows.iter().map(|(_, id)| id.as_str()).collect();
    Ok((
        ids.join(" "),
        json!({"rows": rows, "recipe_hash": recipe, "data_snapshot_hash": data}),
    ))
}

/// The ft rows, in order, checked as one configuration that can be averaged, ensembled or
/// used as an envelope: each completed, `quick: false`, an `epoch` arm with no shuffle, the
/// seed it was given as, and all sharing one recipe hash and one data snapshot (returned).
fn one_configuration<'a>(
    ledger: &'a Ledger,
    ft_rows: &[(i64, String)],
) -> Result<(Vec<&'a Row>, String, String)> {
    let mut shared: Option<(String, String)> = None;
    let mut rows = Vec::new();
    for (seed, id) in ft_rows {
        let row = ft_row(ledger, *seed, id)?;
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
        rows.push(row);
    }
    let (recipe, data) = shared.ok_or_else(|| "no ft row was given".to_string())?;
    Ok((rows, recipe, data))
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
        Cmd::Successor { .. } => "f_successor",
        Cmd::J6a { .. } => "j6a_replay",
        Cmd::J6g { .. } => "j6g_option_permutation",
        Cmd::FtRows { .. } => "ft_rows",
        Cmd::EvalRow { .. } => "eval_row",
    }
}

fn preregistration(cmd: &Cmd) -> &'static str {
    match cmd {
        Cmd::Successor { .. } => PREREG_SUCC,
        Cmd::J6a { .. } => PREREG_J6A,
        Cmd::J6g { .. } => PREREG_J6G,
        _ => PREREG,
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
        Cmd::Successor {
            f_ledger,
            ft_rows,
            arm_ledger,
            j6f_ft_row,
            j6dv4_ft_row,
            ..
        } => rule_successor(
            &mut inputs,
            f_ledger,
            ft_rows,
            arm_ledger,
            j6f_ft_row,
            j6dv4_ft_row,
        ),
        Cmd::J6a {
            preregistration,
            f_ledger,
            ft_rows,
            arm_ledger,
            j6a_ft_row,
            data_snapshot_hash,
            shard_hash,
            replay_shard_hash,
            replay_attestation_sha256,
            h,
            hit_list_sha256,
            ..
        } => rule_j6a(
            &mut inputs,
            preregistration,
            f_ledger,
            ft_rows,
            arm_ledger,
            j6a_ft_row,
            &Pins {
                data_snapshot_hash: data_snapshot_hash.clone(),
                shard_hash: shard_hash.clone(),
                replay_shard_hash: replay_shard_hash.clone(),
                replay_attestation_sha256: replay_attestation_sha256.clone(),
                h: *h,
                hit_list_sha256: hit_list_sha256.clone(),
            },
        ),
        Cmd::J6g {
            preregistration,
            f_ledger,
            ft_rows,
            arm_ledger,
            j6g_ft_row,
            ..
        } => rule_j6g(
            &mut inputs,
            preregistration,
            f_ledger,
            ft_rows,
            arm_ledger,
            j6g_ft_row,
        ),
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
        "preregistration": preregistration(cmd),
        "inputs": inputs.0,
    });
    let (word, refused) = match result {
        // A rule that refuses by its own terms (successor: both arms win, a target without
        // room, or an arm row that could not be read) keeps its detail, lists every reason in
        // `refused_because`, and exits as every refusal does.
        Ok((word, body)) if word == "refused" => {
            let reason = match body.get("refused_because") {
                Some(Value::Array(reasons)) if !reasons.is_empty() => reasons
                    .iter()
                    .map(|r| r.as_str().map_or_else(|| r.to_string(), str::to_string))
                    .collect::<Vec<_>>()
                    .join("; "),
                _ => "the rule refused without recording why; see detail".to_string(),
            };
            json["decision"] = Value::String(word.clone());
            json["refused"] = Value::String(reason);
            json["detail"] = body;
            (word, true)
        }
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
        Cmd::Avgnp { out, .. }
        | Cmd::Seeds34 { out, .. }
        | Cmd::J6f { out, .. }
        | Cmd::Successor { out, .. }
        | Cmd::J6a { out, .. }
        | Cmd::J6g { out, .. } => Some(out),
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

    // --- (iv) successor: campaign/f-successor-preregistered.json -----------------------------

    const SUCC_PREREG: &str = "campaign/f-successor-preregistered.json";
    const J6B: &str = "ledger/gh200-j6-ablations-2026-10-01.jsonl";
    const J6B_FT: &str = "3fa33161-0bdb-42fd-b8b1-509c16e32a27";
    /// F's ledger as it stood at 2026-10-02 01:13 UTC (sha256 abb48449...): seed 0's rows only.
    const F_SNAPSHOT: &str =
        "AUDIT/idle-gpu-queue-2026-10-02/f-ledger-seed0-snapshot-2026-10-02T0113Z.jsonl";
    const F_SEED0_FT: &str = "973cd4e3-e0d2-4ff8-8588-b761cb842b75";

    /// The replay arm: J6(b) against J4, with the targets its flag was meant to move
    /// (`j6b_evidence.met`) and the base must-not-lose list only.
    const ARM_J6B: Arm = Arm {
        name: "j6b",
        word: "fires",
        delta: &[("lower_layers_n", Some(8.0)), ("lower_lr_scale", Some(0.1))],
        targets: &[PROSE, SCRAMBLED],
        guards: &[],
    };

    fn prereg_succ() -> Value {
        serde_json::from_str(&std::fs::read_to_string(repo(SUCC_PREREG)).unwrap()).unwrap()
    }

    fn names(ms: &[Metric]) -> Vec<&'static str> {
        ms.iter().map(|m| m.name).collect()
    }

    #[test]
    fn the_successor_rule_is_the_preregistrations_own_text() {
        let p = prereg_succ();
        let words: Vec<&str> = p["outcomes"]["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect();
        assert_eq!(words, [ARM_J6F.word, ARM_J6DV4.word, "quiet", "refused"]);
        let base: Vec<&str> = p["base_must_not_lose"]["metrics"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        assert_eq!(base, names(&BASE_MUST_NOT_LOSE));
        for (arm, key) in [(&ARM_J6F, "j6f"), (&ARM_J6DV4, "j6dv4")] {
            let a = &p["arms"][key];
            let targets: Vec<&str> = a["targets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["name"].as_str().unwrap())
                .collect();
            assert_eq!(targets, names(arm.targets), "{key}");
            let text = a["must_not_lose"].as_str().unwrap();
            assert!(text.starts_with("base_must_not_lose, plus: "), "{key}");
            for m in arm.guards {
                let phrase = match m.source {
                    Source::Control(_) => {
                        "needle_hunk_recall.control.1024 / .2048 / .4096 worst depth bucket"
                    }
                    _ => m.name,
                };
                assert!(text.contains(phrase), "{key}: {phrase} not in {text}");
            }
            assert_eq!(a["reads_margin"], arm.reads_margin(), "{key}");
        }
        // Each arm's must-not-lose names exactly the guards above, no more: count the metric
        // names the text lists against the arm's own list (controls are one phrase).
        let j6f_text = p["arms"]["j6f"]["must_not_lose"].as_str().unwrap();
        assert!(!j6f_text.contains("unseen-language") && !j6f_text.contains("in_distribution"));
        // R5 struck (e9cff78): J6(d)-v4 lists no length control and requires no control row.
        let j6dv4 = &p["arms"]["j6dv4"];
        assert!(
            !j6dv4["must_not_lose"]
                .as_str()
                .unwrap()
                .contains("needle_hunk_recall.control")
        );
        assert!(
            !j6dv4["rows"]
                .as_str()
                .unwrap()
                .contains("needle-length-control")
        );
        assert!(
            p["arms"]["j6f"]["rows"]
                .as_str()
                .unwrap()
                .contains("epoch-needle-length-control")
        );
        assert!(
            p["readings"]["R5_j6dv4_needle"]
                .as_str()
                .unwrap()
                .starts_with("struck by Fable 2026-10-02")
        );
        assert!(PREREG_SUCC.contains("e9cff78"));
        assert!(
            p["arms"]["j6f"]["identity"]
                .as_str()
                .unwrap()
                .contains("lower_layers_n and lower_lr_scale are absent")
        );
        assert_eq!(
            ARM_J6F.delta,
            &[("lower_layers_n", None), ("lower_lr_scale", None)]
        );
        assert!(
            p["arms"]["j6dv4"]["identity"]
                .as_str()
                .unwrap()
                .contains("except lr = 3e-5 and beta2 = 0.95")
        );
        assert_eq!(
            ARM_J6DV4.delta,
            &[("lr", Some(3e-5)), ("beta2", Some(0.95))]
        );
        assert!(
            p["envelope"]["pin"]
                .as_str()
                .unwrap()
                .contains("The envelope is F seeds 0, 1 and 2 only")
        );
        assert_eq!(F_SEEDS, [0, 1, 2]);
        let c = &p["comparison"];
        assert!(c["target_clears"].as_str().unwrap().contains(
            "higher-better: candidate - max > max - min; lower-better: min - candidate > max - min"
        ));
        assert!(
            c["guard_loses"]
                .as_str()
                .unwrap()
                .contains("higher-better: candidate < min; lower-better: candidate > max")
        );
        assert!(c["counts"].as_str().unwrap().contains(
            "clears (higher-better) iff a*f*h + e*b*h > 2*g*b*f; clears (lower-better) iff \
             2*e*b*h > a*f*h + g*b*f"
        ));
        assert!(c["paired_margin"].as_str().unwrap().contains("873/2304"));
        // R9_room (56d74cf): the Room clause, its (c) refusal and its reading, as no_room and
        // room_bound implement them.
        let target_clears = c["target_clears"].as_str().unwrap();
        for phrase in [
            "Higher-better, bound U (1 for counts and the paired margin): if 2*max - min >= U \
             (counts: 2*g*f - e*h >= f*h), no candidate can clear.",
            "Lower-better, bound L (0 for counts): if 2*min - max <= L (counts: 2*e*h <= g*f), \
             likewise.",
            "Decided from the envelope alone, before the arm's rows are read.",
            "refuses the decision (outcomes.refused (c))",
        ] {
            assert!(target_clears.contains(phrase), "{phrase}");
        }
        let refused = p["outcomes"]["refused"].as_str().unwrap();
        assert!(refused.contains(
            "The JSON carries cannot_clear: [{arm, target, max, min, bound}] and refused_because \
             names them. Never quiet and never fires:<other arm> in case (c)"
        ));
        assert!(
            p["readings"]["R9_room"]
                .as_str()
                .unwrap()
                .contains("2*max - min >= 1 = spread >= 1 - max")
        );
        assert!(PREREG_SUCC.contains("56d74cf"));
        for arm in [&ARM_J6F, &ARM_J6DV4] {
            for m in arm.targets {
                let b = room_bound(*m).unwrap();
                assert_eq!(
                    (b.name, b.value),
                    ("U", Rat { num: 1, den: 1 }),
                    "{}",
                    m.name
                );
            }
        }
        assert!(
            p["readings"]["R3_all_targets_clear"]
                .as_str()
                .unwrap()
                .contains("every one of its targets clears")
        );
        assert_eq!(p["arms"]["j6dv4"]["targets"][0]["name"], MARGIN_KEY);
        assert!(PREREG_SUCC.contains("ba1cedb") && PREREG_SUCC.contains(SUCC_PREREG));
    }

    /// Fable's real-row check: the definition, run on J6(b)'s and J4's committed rows, says
    /// what the record says (`met: yes`).
    #[test]
    fn j6b_against_j4_fires_on_the_real_rows() {
        let mut inputs = Inputs::default();
        let decided = decide_arms(
            &mut inputs,
            &repo(J4),
            &ft_args(&J4_FT),
            &repo(J6B),
            &[(&ARM_J6B, &(0, J6B_FT.to_string()))],
            &arm_identity,
        )
        .unwrap();
        // R9_room on the real rows: prose 2*9 - 2 = 16 < 60 and scrambled 2*8 - 3 = 13 < 60
        // leave J6(b)'s targets room, so the replay is decided by the comparison as before.
        let room: Vec<(&str, bool)> = decided
            .room
            .iter()
            .map(|r| (r.metric.name, r.decided.as_ref().unwrap().3))
            .collect();
        assert_eq!(
            room,
            [
                ("ood_abstain.prose", false),
                ("ood_abstain.scrambled", false)
            ]
        );
        let (wins, arms) = decided.judged.unwrap();
        assert_eq!(wins, [true], "{arms}");
        let d = json!({"arms": arms, "envelope": decided.envelope});
        let arm = &d["arms"]["j6b"];
        assert!(
            arm["rows"]["eval_row"]
                .as_str()
                .unwrap()
                .starts_with("f4958492")
        );
        assert_eq!(arm["rows"]["ft_row"], J6B_FT);
        let t = &arm["targets"];
        // prose 44 > 9 + (9 - 2); scrambled 51 > 8 + (8 - 3).
        assert_eq!(
            (
                t[0]["candidate"]["n"].as_i64(),
                t[0]["max"]["n"].as_i64(),
                t[0]["min"]["n"].as_i64()
            ),
            (Some(44), Some(9), Some(2))
        );
        assert_eq!(
            (
                t[1]["candidate"]["n"].as_i64(),
                t[1]["max"]["n"].as_i64(),
                t[1]["min"]["n"].as_i64()
            ),
            (Some(51), Some(8), Some(3))
        );
        assert!(t.as_array().unwrap().iter().all(|x| x["clears"] == true));
        let g = arm["must_not_lose"].as_array().unwrap();
        assert_eq!(g.len(), 5);
        assert!(g.iter().all(|x| x["loses"] == false), "{g:?}");
        // The 8K needle: J6(b)'s worst bucket 0/59 ties J4's minimum 0/59: inside, not lost.
        let needle = g
            .iter()
            .find(|x| x["metric"] == "needle_8k_worst_bucket")
            .unwrap();
        assert_eq!(
            (
                needle["candidate"]["n"].as_i64(),
                needle["min"]["n"].as_i64()
            ),
            (Some(0), Some(0))
        );
        let ece = g.iter().find(|x| x["metric"] == ECE_DC_KEY).unwrap();
        assert!(
            ece["candidate"]["value"].as_f64().unwrap() < ece["max"]["value"].as_f64().unwrap()
        );
        // The envelope is J4's three eval rows, pinned through their ft rows.
        let evals: Vec<&str> = d["envelope"]["seeds"]
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
        // Both ledgers were read and hashed; the J6(b) one is the box's (4a32bc0a...).
        assert_eq!(inputs.0.len(), 2);
        assert!(
            inputs.0[1]["sha256"]
                .as_str()
                .unwrap()
                .starts_with("4a32bc0a")
        );
    }

    #[test]
    fn j6b_against_j4_without_its_flags_is_refused_by_the_identity_check() {
        // The same rows, declared as J6(f)'s delta (lower-layers flags absent): J6(b)'s ft row
        // carries them, so it is not that arm and the decision refuses rather than judges.
        let arm = Arm {
            delta: &[("lower_layers_n", None), ("lower_lr_scale", None)],
            ..ARM_J6B
        };
        let err = decide_arms(
            &mut Inputs::default(),
            &repo(J4),
            &ft_args(&J4_FT),
            &repo(J6B),
            &[(&arm, &(0, J6B_FT.to_string()))],
            &arm_identity,
        )
        .unwrap()
        .judged
        .unwrap_err();
        assert!(
            err.contains("recipe.lower_layers_n is 8, but the arm drops it"),
            "{err}"
        );
    }

    /// Every key the rule reads is on F seed 0's real rows (f4feac15, a4f244f0, c89b89a1),
    /// read through the checker's own readers, with the values the pre-registration cites.
    #[test]
    fn every_key_the_rule_reads_is_on_f_seed_0s_real_rows() {
        let ledger = Inputs::default().read(&repo(F_SNAPSHOT)).unwrap();
        let rows = seed_rows(&ledger, 0, F_SEED0_FT, true, true).unwrap();
        assert!(rows.eval.id().starts_with("f4feac15"));
        assert!(rows.control.unwrap().id().starts_with("a4f244f0"));
        assert!(rows.letter.unwrap().id().starts_with("c89b89a1"));
        let exact = |m: Metric| match rows.read(m).unwrap() {
            Val::Exact(r) => (r.num, r.den),
            v => panic!("{} read as {v:?}", m.name),
        };
        assert_eq!(exact(NEEDLE_8K_WORST), (40, 61));
        for m in [CONTROL_1K, CONTROL_2K, CONTROL_4K] {
            let (k, n) = exact(m);
            assert_eq!(k, n, "{}", m.name);
        }
        assert_eq!(exact(PROSE), (54, 60));
        assert_eq!(exact(SCRAMBLED), (58, 60));
        assert_eq!(exact(UNSEEN), (20, 60));
        assert_eq!(exact(PERM_DC), (2291, 2304));
        assert_eq!(exact(ID_ABSTAIN_DC), (13, 2304));
        assert_eq!(exact(VAL_CHOICE), (9130, 10985));
        assert_eq!(exact(VAL_SPAN), (6529, 7238));
        assert_eq!(exact(MARGIN_DC), (873, 2304));
        match rows.read(ECE_DC).unwrap() {
            Val::Float(v) => assert!((v - 0.020_707_060_589_094_58).abs() < 1e-15),
            v => panic!("ECE read as {v:?}"),
        }
        // Every metric either arm reads is among those just read.
        for arm in [&ARM_J6F, &ARM_J6DV4] {
            for m in arm.targets.iter().chain(arm.must_not_lose().iter()) {
                assert!(rows.read(*m).is_ok(), "{}", m.name);
            }
        }
        // The option-control row 4e8ec654 is in the ledger and is not selected.
        assert!(ledger.rows.iter().any(|r| r.id().starts_with("4e8ec654")));
    }

    // Synthetic ledgers: F's three seeds (ft, eval, needle control, letter control each) and the
    // two arms, every metric set from a profile.

    #[derive(Clone, Copy, Debug)]
    struct Profile {
        /// The 8K worst bucket: (depth bucket index, hits); the other buckets are full.
        needle: (usize, u64),
        /// 1K/2K/4K hits in the 80-100% bucket, out of 60; the other buckets are full.
        controls: [u64; 3],
        prose: u64,
        scrambled: u64,
        unseen: u64,
        choice: u64,
        span: u64,
        perm: u64,
        id_abstain: u64,
        ece: f64,
        /// Numerator of the paired margin over 2304 rows.
        margin: i64,
        /// J6(a)'s targets: MMLU / CSQA permutation consistency (of 1485 / 1197) and
        /// in-distribution abstention (of 1485 / 1197).
        perm_know: u64,
        perm_csqa: u64,
        id_know: u64,
        id_csqa: u64,
    }

    const NEEDLE_SIZES: [u64; 5] = [59, 61, 59, 60, 61];

    /// F's envelope: 8K worst 40/45/50 of 61, controls 60/60, prose 54/50/52, scrambled
    /// 58/55/57, unseen 20/25/30, choice 9130/9100/9160, span 6529/6500/6550, permutation
    /// 2291/2285/2295, in-distribution abstention 13/15/11, ECE 0.0207/0.0190/0.0220, margin
    /// 873/860/880 of 2304.
    const F_ENV: [Profile; 3] = [
        Profile {
            needle: (4, 40),
            controls: [60, 60, 60],
            prose: 54,
            scrambled: 58,
            unseen: 20,
            choice: 9130,
            span: 6529,
            perm: 2291,
            id_abstain: 13,
            ece: 0.0207,
            margin: 873,
            perm_know: 1168,
            perm_csqa: 1053,
            id_know: 317,
            id_csqa: 144,
        },
        Profile {
            needle: (4, 45),
            controls: [60, 60, 60],
            prose: 50,
            scrambled: 55,
            unseen: 25,
            choice: 9100,
            span: 6500,
            perm: 2285,
            id_abstain: 15,
            ece: 0.0190,
            margin: 860,
            perm_know: 1150,
            perm_csqa: 1040,
            id_know: 330,
            id_csqa: 150,
        },
        Profile {
            needle: (4, 50),
            controls: [60, 60, 60],
            prose: 52,
            scrambled: 57,
            unseen: 30,
            choice: 9160,
            span: 6550,
            perm: 2295,
            id_abstain: 11,
            ece: 0.0220,
            margin: 880,
            perm_know: 1180,
            perm_csqa: 1060,
            id_know: 300,
            id_csqa: 140,
        },
    ];
    /// Inside the envelope on every metric.
    const NEUTRAL: Profile = F_ENV[0];
    /// J6(f) clearing its one target: 61/61 at 8K (threshold: more than 60/61).
    const J6F_WINS: Profile = Profile {
        needle: (4, 61),
        ..NEUTRAL
    };
    /// J6(d)-v4 clearing all three targets: margin 901 > 900, choice 9221 > 9220, span 6601 > 6600.
    const J6DV4_WINS: Profile = Profile {
        margin: 901,
        choice: 9221,
        span: 6601,
        ..NEUTRAL
    };

    fn rid(kind: u64, n: u64) -> String {
        format!("{kind:08x}-0000-4000-8000-{n:012x}")
    }

    fn ran(k: u64, n: u64) -> Value {
        json!({"state": "ran", "n": k, "n_total": n, "value": k as f64 / n as f64})
    }

    fn s_ft(
        id: &str,
        seed: i64,
        recipe_hash: &str,
        edit: impl Fn(&mut Map<String, Value>),
    ) -> Value {
        let mut recipe = json!({
            "tag": "epoch", "tool": "tools/real_ft_run.py", "optimizer_recipe": "master",
            "lr": 1e-5, "lower_layers_n": 8, "lower_lr_scale": 0.1,
            "checkpoint_skip_layers": 6, "batch_tokens": 35403, "no_memorise": true,
            "shard_hash": "5f".repeat(32), "batches": 9683, "width": 7936,
        });
        edit(recipe.as_object_mut().unwrap());
        json!({
            "row_id": id, "run_kind": "ft", "status": "completed", "quick": false,
            "code_commit": "a5026707b6e3e57c253be32003ba32420f2d78e2",
            "protocol": {"seed": seed, "recipe_hash": recipe_hash, "data_snapshot_hash": "d"},
            "recipe": recipe,
        })
    }

    fn s_eval(id: &str, seed: i64, ft: &str, p: &Profile) -> Value {
        let mut m = Map::new();
        let mut worst = (u64::MAX, 1u64);
        for (i, label) in DEPTH_BUCKETS.iter().enumerate() {
            let n = NEEDLE_SIZES[i];
            let k = if i == p.needle.0 { p.needle.1 } else { n };
            if u128::from(k) * u128::from(worst.1) < u128::from(worst.0) * u128::from(n) {
                worst = (k, n);
            }
            m.insert(format!("needle_hunk_recall.depth.{label}"), ran(k, n));
        }
        for (key, k, n) in [
            ("ood_abstain.prose", p.prose, 60),
            ("ood_abstain.scrambled", p.scrambled, 60),
            ("ood_abstain.unseen-language", p.unseen, 60),
            ("val_top1.choice", p.choice, 10985),
            ("val_top1.span", p.span, 7238),
            (PERM_DC.name, p.perm, 2304),
            (ID_ABSTAIN_DC.name, p.id_abstain, 2304),
            (PERM_KNOWLEDGE.name, p.perm_know, 1485),
            (PERM_COMMONSENSE.name, p.perm_csqa, 1197),
            (ID_ABSTAIN_KNOWLEDGE.name, p.id_know, 1485),
            (ID_ABSTAIN_COMMONSENSE.name, p.id_csqa, 1197),
        ] {
            m.insert(key.into(), ran(k, n));
        }
        m.insert(
            ECE_DC_KEY.into(),
            json!({"state": "ran", "n": 2304, "n_total": 2304, "value": p.ece}),
        );
        m.insert("ft_run_row_id".into(), json!({"state": "ran", "value": ft}));
        json!({
            "row_id": id, "run_kind": "eval", "status": "completed", "quick": false,
            "protocol": {"seed": seed},
            "recipe": {"tag": EVAL_TAG, "val_shard_hash": "v", "needle": {"cases": 300}, "ood": {"cases_per_category": 60}},
            "gates": {"needle_hunk_recall": {"state": "ran", "n": 300, "n_total": 300, "value": worst.0 as f64 / worst.1 as f64}},
            "metrics": m,
        })
    }

    fn s_control(id: &str, seed: i64, ft: &str, p: &Profile) -> Value {
        let mut m = Map::new();
        for (length, k) in [1024u32, 2048, 4096].into_iter().zip(p.controls) {
            for (i, label) in DEPTH_BUCKETS.iter().enumerate() {
                let hits = if i == 4 { k } else { 60 };
                m.insert(
                    format!("needle_hunk_recall.control.{length}.depth.{label}"),
                    ran(hits, 60),
                );
            }
            m.insert(
                format!("needle_hunk_recall.control.{length}"),
                json!({"state": "ran", "n": 300, "n_total": 300, "value": k as f64 / 60.0}),
            );
        }
        m.insert("ft_run_row_id".into(), json!({"state": "ran", "value": ft}));
        json!({
            "row_id": id, "run_kind": "eval", "status": "completed", "quick": true,
            "protocol": {"seed": seed},
            "recipe": {"tag": CONTROL_TAG, "needle_control": {"target_tokens": [1024, 2048, 4096]}},
            "metrics": m,
        })
    }

    fn s_letter(id: &str, seed: i64, eval: &str, margin: i64) -> Value {
        json!({
            "row_id": id, "run_kind": "eval", "status": "completed", "quick": false,
            "protocol": {"seed": seed},
            "recipe": {"tool": "tools/ft_linear_control.py", "control_engine_sha256": "c", "eval_row_id": eval},
            "metrics": {
                "scored_eval_row_id": {"state": "ran", "value": eval},
                MARGIN_KEY: {"state": "ran", "n": 2304, "n_total": 2304, "value": margin as f64 / 2304.0},
            },
        })
    }

    /// A run's four rows; `kind` keeps ids distinct per run.
    fn s_run(
        rows: &mut Vec<Value>,
        kind: u64,
        seed: i64,
        recipe_hash: &str,
        p: &Profile,
        recipe: impl Fn(&mut Map<String, Value>),
    ) -> String {
        let (ft, ev) = (rid(kind, 1), rid(kind, 2));
        rows.push(s_ft(&ft, seed, recipe_hash, recipe));
        rows.push(s_eval(&ev, seed, &ft, p));
        rows.push(s_control(&rid(kind, 3), seed, &ft, p));
        rows.push(s_letter(&rid(kind, 4), seed, &ev, p.margin));
        ft
    }

    fn j6f_recipe(r: &mut Map<String, Value>) {
        r.remove("lower_layers_n");
        r.remove("lower_lr_scale");
    }
    fn j6dv4_recipe(r: &mut Map<String, Value>) {
        r.insert("lr".into(), json!(3e-5));
        r.insert("beta2".into(), json!(0.95));
    }

    struct Succ {
        f: Temp,
        arms: Temp,
        f_ft: Vec<(i64, String)>,
        j6f: (i64, String),
        j6dv4: (i64, String),
    }

    /// F's ledger and the arm ledger, with `edit` applied to their rows (F's, then the arms').
    fn succ_fixture(
        env: [Profile; 3],
        j6f: Profile,
        j6dv4: Profile,
        edit: impl Fn(&mut Vec<Value>, &mut Vec<Value>),
    ) -> Succ {
        let (mut f_rows, mut a_rows) = (Vec::new(), Vec::new());
        let f_ft: Vec<(i64, String)> = env
            .iter()
            .enumerate()
            .map(|(s, p)| {
                (
                    s as i64,
                    s_run(&mut f_rows, 0xf0 + s as u64, s as i64, "r", p, |_| {}),
                )
            })
            .collect();
        let j6f_ft = s_run(&mut a_rows, 0xa6f, 0, "rf", &j6f, j6f_recipe);
        let j6dv4_ft = s_run(&mut a_rows, 0xa6d, 0, "rd", &j6dv4, j6dv4_recipe);
        edit(&mut f_rows, &mut a_rows);
        Succ {
            f: temp_ledger(&f_rows),
            arms: temp_ledger(&a_rows),
            f_ft,
            j6f: (0, j6f_ft),
            j6dv4: (0, j6dv4_ft),
        }
    }

    fn succ_of(fx: &Succ) -> Outcome {
        outcome(Cmd::Successor {
            f_ledger: fx.f.0.clone(),
            ft_rows: fx.f_ft.clone(),
            arm_ledger: fx.arms.0.clone(),
            j6f_ft_row: fx.j6f.clone(),
            j6dv4_ft_row: fx.j6dv4.clone(),
            out: PathBuf::from("/unused"),
        })
    }

    fn succ(env: [Profile; 3], j6f: Profile, j6dv4: Profile) -> Outcome {
        succ_of(&succ_fixture(env, j6f, j6dv4, |_, _| {}))
    }

    fn no_edit(_: &mut Vec<Value>, _: &mut Vec<Value>) {}

    fn row_mut<'a>(rows: &'a mut [Value], id: &str) -> &'a mut Value {
        rows.iter_mut().find(|r| r["row_id"] == id).unwrap()
    }

    fn refused_with(o: &Outcome, phrase: &str) {
        assert!(o.refused, "not refused: {}", o.json);
        assert_eq!(o.word, "refused");
        let reason = o.json["refused"].as_str().unwrap();
        assert!(reason.contains(phrase), "{reason}");
    }

    #[test]
    fn quiet_when_both_arms_sit_inside_the_envelope() {
        let o = succ(F_ENV, NEUTRAL, NEUTRAL);
        assert!(!o.refused, "{}", o.json);
        assert_eq!(o.word, "quiet");
        let d = &o.json["detail"];
        assert_eq!(d["arms"]["j6f"]["wins"], false);
        assert_eq!(d["arms"]["j6dv4"]["wins"], false);
        assert!(
            d["envelope"]["pin"]
                .as_str()
                .unwrap()
                .contains("seeds added by post-F rule (ii) are not part")
        );
        assert_eq!(o.json["preregistration"], PREREG_SUCC);
    }

    #[test]
    fn j6f_alone_fires_and_j6dv4_alone_fires() {
        let o = succ(F_ENV, J6F_WINS, NEUTRAL);
        assert_eq!(
            (o.word.as_str(), o.refused),
            ("fires:j6f", false),
            "{}",
            o.json
        );
        let o = succ(F_ENV, NEUTRAL, J6DV4_WINS);
        assert_eq!(
            (o.word.as_str(), o.refused),
            ("fires:j6dv4", false),
            "{}",
            o.json
        );
        // Each arm judges only its own targets: J6(d)-v4's winning profile does not move J6(f).
        let o = succ(F_ENV, J6DV4_WINS, NEUTRAL);
        assert_eq!(o.word, "quiet", "{}", o.json);
    }

    #[test]
    fn both_arms_winning_refuses_and_keeps_both_verdicts() {
        let o = succ(F_ENV, J6F_WINS, J6DV4_WINS);
        refused_with(&o, "the human decides");
        assert_eq!(o.json["decision"], "refused");
        let d = &o.json["detail"];
        assert_eq!(
            (
                d["arms"]["j6f"]["wins"].as_bool(),
                d["arms"]["j6dv4"]["wins"].as_bool()
            ),
            (Some(true), Some(true))
        );
    }

    #[test]
    fn a_target_at_exactly_max_plus_range_does_not_fire_and_one_hit_more_does() {
        // 8K: envelope 40..50 of 61, range 10/61; 60/61 is exactly max + range.
        let tie = Profile {
            needle: (4, 60),
            ..NEUTRAL
        };
        assert_eq!(succ(F_ENV, tie, NEUTRAL).word, "quiet");
        assert_eq!(succ(F_ENV, J6F_WINS, NEUTRAL).word, "fires:j6f");
        // Margin: envelope 860..880 of 2304, range 20; 900 ties, 901 clears.
        let tie = Profile {
            margin: 900,
            ..J6DV4_WINS
        };
        assert_eq!(succ(F_ENV, NEUTRAL, tie).word, "quiet");
        // Choice 9160 + 60 = 9220 ties; span 6550 + 50 = 6600 ties.
        let tie = Profile {
            choice: 9220,
            ..J6DV4_WINS
        };
        assert_eq!(succ(F_ENV, NEUTRAL, tie).word, "quiet");
        let tie = Profile {
            span: 6600,
            ..J6DV4_WINS
        };
        assert_eq!(succ(F_ENV, NEUTRAL, tie).word, "quiet");
    }

    #[test]
    fn j6dv4_needs_every_target_to_clear() {
        // R3: two of three clearing is not a win.
        for p in [
            Profile {
                margin: 873,
                ..J6DV4_WINS
            },
            Profile {
                choice: 9130,
                ..J6DV4_WINS
            },
            Profile {
                span: 6529,
                ..J6DV4_WINS
            },
        ] {
            let o = succ(F_ENV, NEUTRAL, p);
            assert_eq!(o.word, "quiet", "{p:?}");
            assert_eq!(
                o.json["detail"]["arms"]["j6dv4"]["targets_all_clear"],
                false
            );
        }
    }

    #[test]
    fn a_guard_at_its_bound_holds_and_one_count_past_it_loses() {
        // Higher-better guards: J6(f) prose at F's minimum 50 fires; 49 loses.
        assert_eq!(
            succ(
                F_ENV,
                Profile {
                    prose: 50,
                    ..J6F_WINS
                },
                NEUTRAL
            )
            .word,
            "fires:j6f"
        );
        let o = succ(
            F_ENV,
            Profile {
                prose: 49,
                ..J6F_WINS
            },
            NEUTRAL,
        );
        assert_eq!(o.word, "quiet");
        assert_eq!(
            o.json["detail"]["arms"]["j6f"]["must_not_lose_lost"],
            json!(["ood_abstain.prose"])
        );
        // Lower-better guards: ECE at F's maximum 0.0220 holds; above it loses.
        assert_eq!(
            succ(
                F_ENV,
                Profile {
                    ece: 0.0220,
                    ..J6F_WINS
                },
                NEUTRAL
            )
            .word,
            "fires:j6f"
        );
        assert_eq!(
            succ(
                F_ENV,
                Profile {
                    ece: 0.0221,
                    ..J6F_WINS
                },
                NEUTRAL
            )
            .word,
            "quiet"
        );
        // J6(d)-v4's in-distribution abstention: 15 (F's max) holds, 16 loses.
        assert_eq!(
            succ(
                F_ENV,
                NEUTRAL,
                Profile {
                    id_abstain: 15,
                    ..J6DV4_WINS
                }
            )
            .word,
            "fires:j6dv4"
        );
        assert_eq!(
            succ(
                F_ENV,
                NEUTRAL,
                Profile {
                    id_abstain: 16,
                    ..J6DV4_WINS
                }
            )
            .word,
            "quiet"
        );
        // Base list: a val top-1 below F's minimum blocks J6(f) too (R1, R2).
        assert_eq!(
            succ(
                F_ENV,
                Profile {
                    span: 6499,
                    ..J6F_WINS
                },
                NEUTRAL
            )
            .word,
            "quiet"
        );
    }

    #[test]
    fn a_length_control_miss_blocks_j6f_and_not_j6dv4() {
        // J6(f)'s guard is Fable's explicit "the length controls": one hit short of F's 60/60
        // at any length loses.
        for controls in [[59, 60, 60], [60, 59, 60], [60, 60, 59]] {
            let o = succ(
                F_ENV,
                Profile {
                    controls,
                    ..J6F_WINS
                },
                NEUTRAL,
            );
            assert_eq!(o.word, "quiet", "{controls:?}");
            assert_eq!(o.json["detail"]["arms"]["j6f"]["targets_all_clear"], true);
        }
        // R5 as struck (e9cff78): J6(d)-v4's "needle" is the 8K worst bucket only, so a control
        // miss does not block it -- not at one length, not at all three.
        for controls in [[59, 60, 60], [60, 60, 59], [0, 0, 0]] {
            let o = succ(
                F_ENV,
                NEUTRAL,
                Profile {
                    controls,
                    ..J6DV4_WINS
                },
            );
            assert_eq!(o.word, "fires:j6dv4", "{controls:?}: {}", o.json);
            let guards = o.json["detail"]["arms"]["j6dv4"]["must_not_lose"]
                .as_array()
                .unwrap();
            assert!(
                guards
                    .iter()
                    .all(|g| !g["metric"].as_str().unwrap().contains(".control."))
            );
        }
        // ...while its 8K worst bucket, in the base list, still guards it.
        let o = succ(
            F_ENV,
            NEUTRAL,
            Profile {
                needle: (4, 39),
                ..J6DV4_WINS
            },
        );
        assert_eq!(o.word, "quiet");
        assert_eq!(
            o.json["detail"]["arms"]["j6dv4"]["must_not_lose_lost"],
            json!(["needle_8k_worst_bucket"])
        );
    }

    #[test]
    fn j6dv4_needs_no_needle_control_row_and_j6f_does() {
        let fx = succ_fixture(F_ENV, NEUTRAL, J6DV4_WINS, |_, a| {
            let ctl = rid(0xa6d, 3);
            a.retain(|r| r["row_id"] != ctl.as_str());
        });
        let o = succ_of(&fx);
        assert_eq!(o.word, "fires:j6dv4", "{}", o.json);
        assert_eq!(
            o.json["detail"]["arms"]["j6dv4"]["rows"]["needle_control_row"],
            Value::Null
        );
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |_, a| {
            let ctl = rid(0xa6f, 3);
            a.retain(|r| r["row_id"] != ctl.as_str());
        });
        refused_with(
            &succ_of(&fx),
            "missing row: no completed eval row tagged epoch-needle-length-control",
        );
        assert!(ARM_J6F.reads_control() && !ARM_J6DV4.reads_control());
    }

    /// The (arm, target) pairs a decision's `cannot_clear` names (R9_room, outcomes.refused (c)).
    fn cannot_clear(o: &Outcome) -> Vec<(String, String)> {
        o.json["detail"]["cannot_clear"]
            .as_array()
            .unwrap_or_else(|| panic!("no detail.cannot_clear: {}", o.json))
            .iter()
            .map(|c| {
                (
                    c["arm"].as_str().unwrap().to_string(),
                    c["target"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    }

    /// Refused under outcomes.refused (c): exit 3's word, never quiet and never fires:<arm>,
    /// with `cannot_clear` naming exactly `blocked` and `refused_because` naming each of them.
    fn refused_cannot_clear(o: &Outcome, blocked: &[(&str, &str)]) {
        refused_with(o, "(c)");
        let want: Vec<(String, String)> = blocked
            .iter()
            .map(|(a, t)| (a.to_string(), t.to_string()))
            .collect();
        assert_eq!(cannot_clear(o), want, "{}", o.json);
        let because: Vec<&str> = o.json["detail"]["refused_because"]
            .as_array()
            .unwrap_or_else(|| panic!("no detail.refused_because list: {}", o.json))
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        for (arm, target) in blocked {
            assert!(
                because
                    .iter()
                    .any(|r| r.starts_with("(c)") && r.contains(arm) && r.contains(target)),
                "{arm} / {target} not named in {because:?}"
            );
        }
    }

    /// An F envelope that is F_ENV with the 8K worst buckets replaced.
    fn needle_env(hits: [u64; 3]) -> [Profile; 3] {
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip(hits) {
            p.needle = (4, k);
        }
        env
    }

    /// GAP-SUCC-J6F-CANNOT-FIRE-WHEN-2MAX-MINUS-MIN-REACHES-1-2026-10-02, fixed by R9_room
    /// (Fable 2026-10-02, AUDIT/idle-gpu-queue-2026-10-02/fable-j6f-room.md): a worst bucket is
    /// at most 1 and must exceed max + range = 2*max - min strictly, so when F's 8K envelope has
    /// 2*max - min >= 1 no J6(f) run can clear. That target cannot be examined, so the decision
    /// refuses (outcomes.refused (c)) with cannot_clear naming it: never quiet, and never
    /// fires:j6dv4, even when J6(d)-v4 wins on its own.
    #[test]
    fn j6f_cannot_fire_when_twice_max_minus_min_reaches_1() {
        // 8K worst 40, 20, 45 of 61: spread 25/61 = 0.41 > 0.30 (rule (ii) fires); 2*45 - 20 =
        // 70 > 61, so even 61/61 does not clear.
        let wide = needle_env([40, 20, 45]);
        let o = succ(wide, J6F_WINS, NEUTRAL);
        refused_cannot_clear(&o, &[("j6f", "needle_8k_worst_bucket")]);
        // The comparison is unchanged and still recorded: 61/61 does not clear.
        assert_eq!(
            o.json["detail"]["arms"]["j6f"]["targets"][0]["clears"],
            false
        );
        // J6(d)-v4 wins on its own, and the decision still is not fires:j6dv4.
        let o = succ(wide, J6F_WINS, J6DV4_WINS);
        refused_cannot_clear(&o, &[("j6f", "needle_8k_worst_bucket")]);
        assert_eq!(o.json["detail"]["arms"]["j6dv4"]["wins"], true);
        // Exactly at 1: 2*40 - 19 = 61, so 61/61 ties max + range and does not clear.
        let at_one = needle_env([40, 19, 40]);
        refused_cannot_clear(
            &succ(at_one, J6F_WINS, NEUTRAL),
            &[("j6f", "needle_8k_worst_bucket")],
        );
        // A spread over 0.30 with a low maximum still leaves room: 2*40 - 21 = 59 < 61, and a
        // spread of 19/61 = 0.31 does not by itself rule J6(f) out.
        let room = needle_env([40, 21, 40]);
        let o = succ(room, J6F_WINS, NEUTRAL);
        assert_eq!(o.word, "fires:j6f", "{}", o.json);
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
    }

    /// The lead's counterexample to "the block is rule (ii)'s spread > 0.30": 8K worst 46/61
    /// and 31/61 with a third seed inside, spread 15/61 = 0.246 (rule (ii) silent), and
    /// 2*46 - 31 = 61, so 61/61 ties max + range. Before R9_room this printed quiet, the word
    /// for "examined and did not clear".
    #[test]
    fn the_leads_46_31_envelope_refuses_where_it_read_quiet() {
        let env = needle_env([46, 31, 40]);
        let o = succ(env, J6F_WINS, NEUTRAL);
        refused_cannot_clear(&o, &[("j6f", "needle_8k_worst_bucket")]);
        refused_with(&o, "max 46/61 and min 31/61 has 2*max - min >= U = 1,");
        let c = &o.json["detail"]["cannot_clear"][0];
        assert_eq!(
            (
                c["max"]["n"].as_i64(),
                c["max"]["n_total"].as_i64(),
                c["min"]["n"].as_i64(),
                c["min"]["n_total"].as_i64(),
            ),
            (Some(46), Some(61), Some(31), Some(61)),
            "{c}"
        );
        assert_eq!(
            (c["bound"]["name"].as_str(), c["bound"]["value"].as_f64()),
            (Some("U"), Some(1.0)),
            "{c}"
        );
        // The prereg's five keys, no more.
        let mut keys: Vec<&String> = c.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["arm", "bound", "max", "min", "target"]);
        // One hit more at the minimum, 32/61: 2*46 - 32 = 60 < 61, so 61/61 clears and fires.
        let o = succ(needle_env([46, 32, 40]), J6F_WINS, NEUTRAL);
        assert_eq!(o.word, "fires:j6f", "{}", o.json);
    }

    /// R9_room: an F seed at 61/61 leaves J6(f) no room at any spread, spread 0 included
    /// (max = 1, and 2 - min >= 1 for every min <= 1).
    #[test]
    fn an_f_seed_at_61_of_61_blocks_j6f_at_any_spread() {
        for env in [needle_env([61, 61, 61]), needle_env([61, 45, 50])] {
            let o = succ(env, J6F_WINS, NEUTRAL);
            refused_cannot_clear(&o, &[("j6f", "needle_8k_worst_bucket")]);
            assert_eq!(
                o.json["detail"]["cannot_clear"][0]["max"]["value"].as_f64(),
                Some(1.0)
            );
        }
    }

    /// R9_room covers J6(d)-v4's targets too: val_top1.span 7000 / 6762 / 6900 of 7238 has
    /// 2*7000 - 6762 = 7238, so even 7238/7238 ties. J6(f) winning on its own does not make the
    /// decision fires:j6f.
    #[test]
    fn j6dv4_span_near_its_ceiling_refuses_and_never_fires_j6f() {
        let spans = |s: [u64; 3]| {
            let mut env = F_ENV;
            for (p, k) in env.iter_mut().zip(s) {
                p.span = k;
            }
            env
        };
        let inside = |p: Profile| Profile { span: 6900, ..p };
        let o = succ(
            spans([7000, 6762, 6900]),
            inside(J6F_WINS),
            inside(J6DV4_WINS),
        );
        refused_cannot_clear(&o, &[("j6dv4", "val_top1.span")]);
        assert_eq!(o.json["detail"]["arms"]["j6f"]["wins"], true, "{}", o.json);
        // One count of room (6763: 2*7000 - 6763 = 7237 < 7238): decided as before.
        let o = succ(spans([7000, 6763, 6900]), inside(J6F_WINS), inside(NEUTRAL));
        assert_eq!(o.word, "fires:j6f", "{}", o.json);
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
    }

    /// R9_room on the paired margin (bound U = 1), on signed fractions: margins 1102, -100,
    /// 500 of 2304 have 2*1102 - (-100) = 2304, so even +2304/2304 ties.
    #[test]
    fn a_paired_margin_envelope_with_no_room_refuses_on_signed_fractions() {
        let margins = |m: [i64; 3]| {
            let mut env = F_ENV;
            for (p, k) in env.iter_mut().zip(m) {
                p.margin = k;
            }
            env
        };
        let o = succ(margins([1102, -100, 500]), NEUTRAL, J6DV4_WINS);
        refused_cannot_clear(&o, &[("j6dv4", MARGIN_KEY)]);
        let c = &o.json["detail"]["cannot_clear"][0];
        assert_eq!(
            (c["min"]["n"].as_i64(), c["max"]["n"].as_i64()),
            (Some(-100), Some(1102))
        );
        // min -99: 2*1102 + 99 = 2303 < 2304 leaves one count of room, and +2304 clears it.
        let o = succ(
            margins([1102, -99, 500]),
            NEUTRAL,
            Profile {
                margin: 2304,
                ..J6DV4_WINS
            },
        );
        assert_eq!(o.word, "fires:j6dv4", "{}", o.json);
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
    }

    /// Room is decided from the envelope before any arm row is read, so an arm row that is
    /// missing (outcomes.refused (b)) does not hide a target with no room (c): both are listed.
    #[test]
    fn a_missing_arm_row_and_a_target_without_room_are_both_listed() {
        let fx = succ_fixture(needle_env([46, 31, 40]), J6F_WINS, NEUTRAL, |_, a| {
            let letter = rid(0xa6d, 4);
            a.retain(|r| r["row_id"] != letter.as_str());
        });
        let o = succ_of(&fx);
        refused_cannot_clear(&o, &[("j6f", "needle_8k_worst_bucket")]);
        refused_with(&o, "missing row: no completed letter-control row");
        let because = o.json["detail"]["refused_because"].as_array().unwrap();
        assert_eq!(because.len(), 2, "{because:?}");
        assert!(because[1].as_str().unwrap().starts_with("(b)"));
    }

    /// Every target of both arms gets a room verdict and no guard does; with room everywhere
    /// cannot_clear is the empty list, which is a decided "none", not an absent check.
    #[test]
    fn room_is_decided_for_every_target_and_for_no_guard() {
        let o = succ(F_ENV, NEUTRAL, NEUTRAL);
        assert_eq!(o.word, "quiet", "{}", o.json);
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
        let room: Vec<(&str, &str, bool)> = o.json["detail"]["room"]
            .as_array()
            .unwrap_or_else(|| panic!("no detail.room: {}", o.json))
            .iter()
            .map(|r| {
                (
                    r["arm"].as_str().unwrap(),
                    r["target"].as_str().unwrap(),
                    r["room"].as_bool().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            room,
            [
                ("j6f", "needle_8k_worst_bucket", true),
                ("j6dv4", MARGIN_KEY, true),
                ("j6dv4", "val_top1.choice", true),
                ("j6dv4", "val_top1.span", true),
            ]
        );
    }

    /// R9_room's two forms on the helper, both directions, at and one step off the boundary,
    /// with unequal denominators and signed margins.
    #[test]
    fn no_room_is_the_preregistered_count_form_in_both_directions() {
        let r = |num: i64, den: u64| Val::Exact(Rat::new(num, den, "t").unwrap());
        let (u, l) = (Rat { num: 1, den: 1 }, Rat { num: 0, den: 1 });
        // Higher-better, U = 1: 2*max - min >= 1.
        assert!(no_room(r(31, 61), r(46, 61), u, Dir::Higher).unwrap());
        assert!(!no_room(r(32, 61), r(46, 61), u, Dir::Higher).unwrap());
        assert!(no_room(r(61, 61), r(61, 61), u, Dir::Higher).unwrap());
        // 2*(3/4) - 1/2 = 1 exactly, across denominators; 2*(3/4) - 3/5 = 0.9 leaves room.
        assert!(no_room(r(1, 2), r(3, 4), u, Dir::Higher).unwrap());
        assert!(!no_room(r(3, 5), r(3, 4), u, Dir::Higher).unwrap());
        // Signed margins: 2*1102 + 100 = 2304 blocks, 2*1102 + 99 = 2303 does not.
        assert!(no_room(r(-100, 2304), r(1102, 2304), u, Dir::Higher).unwrap());
        assert!(!no_room(r(-99, 2304), r(1102, 2304), u, Dir::Higher).unwrap());
        // Lower-better, L = 0: 2*min - max <= 0.
        assert!(no_room(r(1, 10), r(2, 10), l, Dir::Lower).unwrap());
        assert!(!no_room(r(2, 10), r(3, 10), l, Dir::Lower).unwrap());
        assert!(no_room(r(0, 2304), r(0, 2304), l, Dir::Lower).unwrap());
        assert!(!no_room(r(1, 2304), r(1, 2304), l, Dir::Lower).unwrap());
        // 2*(1/6) - 1/3 = 0 exactly; 2*(1/6) - 2/7 = 1/21 > 0 leaves room.
        assert!(no_room(r(1, 6), r(1, 3), l, Dir::Lower).unwrap());
        assert!(!no_room(r(1, 6), r(2, 7), l, Dir::Lower).unwrap());
        // Direction matters: 1/10..2/10 has room upward (2*2/10 - 1/10 < 1) and none downward
        // (2*1/10 - 2/10 = 0); 6/10..10/10 is the reverse (2 - 6/10 >= 1; 12/10 - 1 > 0).
        assert!(!no_room(r(1, 10), r(2, 10), u, Dir::Higher).unwrap());
        assert!(no_room(r(1, 10), r(2, 10), l, Dir::Lower).unwrap());
        assert!(no_room(r(6, 10), r(10, 10), u, Dir::Higher).unwrap());
        assert!(!no_room(r(6, 10), r(10, 10), l, Dir::Lower).unwrap());
        // A float envelope has no pre-registered room.
        assert!(no_room(Val::Float(0.01), Val::Float(0.02), l, Dir::Lower).is_err());
    }

    /// Two independent signals agree: no room iff the bound itself (the best any candidate can
    /// score) does not clear by the unchanged comparison. Every envelope with denominators up
    /// to 7, both directions; signed numerators in the higher direction (the paired margin).
    #[test]
    fn no_room_iff_the_bound_itself_does_not_clear() {
        let mut fracs = Vec::new();
        for den in 1..=7u64 {
            for num in -(den as i64)..=den as i64 {
                fracs.push(Rat::new(num, den, "t").unwrap());
            }
        }
        let mut checked = 0;
        for &lo in &fracs {
            for &hi in &fracs {
                let (lo, hi) = (Val::Exact(lo), Val::Exact(hi));
                if cmp_val(lo, hi).unwrap() == std::cmp::Ordering::Greater {
                    continue;
                }
                let u = room_bound(VAL_SPAN).unwrap().value;
                assert_eq!(
                    no_room(lo, hi, u, Dir::Higher).unwrap(),
                    !clears(Val::Exact(u), lo, hi, Dir::Higher).unwrap(),
                    "{lo:?} {hi:?} higher"
                );
                let nonneg = |v: Val| matches!(v, Val::Exact(r) if r.num >= 0);
                if nonneg(lo) && nonneg(hi) {
                    let l = room_bound(ID_ABSTAIN_DC).unwrap().value;
                    assert_eq!(
                        no_room(lo, hi, l, Dir::Lower).unwrap(),
                        !clears(Val::Exact(l), lo, hi, Dir::Lower).unwrap(),
                        "{lo:?} {hi:?} lower"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 1000, "{checked}");
    }

    /// Bounds are pre-registered for counts and the higher-better margin only; anything else
    /// refuses rather than defaults.
    #[test]
    fn room_bounds_are_the_preregistered_ones_and_nothing_else() {
        assert_eq!(room_bound(NEEDLE_8K_WORST).unwrap().name, "U");
        assert_eq!(room_bound(MARGIN_DC).unwrap().name, "U");
        assert_eq!(room_bound(CONTROL_1K).unwrap().name, "U");
        let l = room_bound(ID_ABSTAIN_DC).unwrap();
        assert_eq!((l.name, l.value), ("L", Rat { num: 0, den: 1 }));
        let err = room_bound(ECE_DC).unwrap_err();
        assert!(err.contains("pre-registers no bound"), "{err}");
        let lower_margin = Metric {
            dir: Dir::Lower,
            ..MARGIN_DC
        };
        assert!(room_bound(lower_margin).is_err());
    }

    /// An envelope that does not resolve decides no room: the refusal says so, and there is no
    /// cannot_clear to be misread as "decided, none blocked".
    #[test]
    fn an_envelope_that_does_not_resolve_decides_no_room() {
        let fx = succ_fixture(needle_env([46, 31, 40]), J6F_WINS, NEUTRAL, |f, _| {
            let ft = rid(0xf1, 1);
            row_mut(f, &ft)["quick"] = json!(true);
        });
        let o = succ_of(&fx);
        refused_with(&o, "does not say quick: false");
        refused_with(&o, "room not decided");
        assert!(o.json["detail"]["cannot_clear"].is_null(), "{}", o.json);
    }

    #[test]
    fn unseen_language_guards_j6dv4_and_not_j6f() {
        // R4: J6(d)-v4's OOD is all three categories; J6(f)'s is prose and scrambled only.
        let low = Profile {
            unseen: 0,
            ..J6F_WINS
        };
        assert_eq!(succ(F_ENV, low, NEUTRAL).word, "fires:j6f");
        let low = Profile {
            unseen: 19,
            ..J6DV4_WINS
        };
        assert_eq!(succ(F_ENV, NEUTRAL, low).word, "quiet");
        assert_eq!(
            succ(
                F_ENV,
                NEUTRAL,
                Profile {
                    unseen: 20,
                    ..J6DV4_WINS
                }
            )
            .word,
            "fires:j6dv4"
        );
    }

    #[test]
    fn exact_forms_with_unequal_denominators_and_signed_margins() {
        let r = |num: i64, den: u64| Val::Exact(Rat::new(num, den, "t").unwrap());
        // 7/10 + 1/2 = 2 * 3/5 exactly: a tie across three denominators does not clear.
        assert!(!clears(r(7, 10), r(1, 2), r(3, 5), Dir::Higher).unwrap());
        assert!(clears(r(43, 61), r(30, 60), r(36, 60), Dir::Higher).unwrap());
        assert!(!clears(r(42, 61), r(30, 60), r(36, 60), Dir::Higher).unwrap());
        // Lower-better mirror: 2 * 3/5 = 7/10 + 1/2 ties; 1/2 - 1/10 = 0.4 < 0.5 range... clears
        // needs min - c > max - min.
        assert!(!clears(r(1, 2), r(3, 5), r(7, 10), Dir::Lower).unwrap());
        assert!(clears(r(49, 100), r(3, 5), r(7, 10), Dir::Lower).unwrap());
        // Signed margins: envelope -10..10 of 2304, range 20; 30 ties, 31 clears.
        assert!(!clears(r(30, 2304), r(-10, 2304), r(10, 2304), Dir::Higher).unwrap());
        assert!(clears(r(31, 2304), r(-10, 2304), r(10, 2304), Dir::Higher).unwrap());
        let env = [
            Profile {
                margin: -10,
                ..NEUTRAL
            },
            Profile {
                margin: 0,
                ..NEUTRAL
            },
            Profile {
                margin: 10,
                ..NEUTRAL
            },
        ];
        assert_eq!(
            succ(
                env,
                NEUTRAL,
                Profile {
                    margin: 30,
                    ..J6DV4_WINS
                }
            )
            .word,
            "quiet"
        );
        assert_eq!(
            succ(
                env,
                NEUTRAL,
                Profile {
                    margin: 31,
                    ..J6DV4_WINS
                }
            )
            .word,
            "fires:j6dv4"
        );
        // Guards: equal to the bound is inside, in either direction.
        assert!(!loses(r(1, 2), r(30, 60), r(36, 60), Dir::Higher).unwrap());
        assert!(loses(r(29, 60), r(1, 2), r(3, 5), Dir::Higher).unwrap());
        assert!(!loses(r(36, 60), r(1, 2), r(3, 5), Dir::Lower).unwrap());
        assert!(loses(r(37, 60), r(1, 2), r(3, 5), Dir::Lower).unwrap());
        // Two forms never compare.
        assert!(cmp_val(r(1, 2), Val::Float(0.5)).is_err());
    }

    #[test]
    fn a_missing_metric_refuses_even_when_an_arm_would_fire() {
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let ev = rid(0xf1, 2);
            row_mut(f, &ev)["metrics"]
                .as_object_mut()
                .unwrap()
                .remove(ECE_DC_KEY);
        });
        refused_with(
            &succ_of(&fx),
            "no metrics.ece.family.code.defect_class.choice.k4",
        );
    }

    #[test]
    fn a_not_run_metric_or_a_row_that_did_not_complete_refuses() {
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |_, a| {
            let ev = rid(0xa6d, 2);
            row_mut(a, &ev)["metrics"][PERM_DC.name] =
                json!({"state": "not_run", "reason": "no suite"});
        });
        refused_with(&succ_of(&fx), "is not_run (no suite), not ran");
        let fx = succ_fixture(F_ENV, NEUTRAL, J6DV4_WINS, |_, a| {
            let ev = rid(0xa6f, 2);
            row_mut(a, &ev)["status"] = json!("failed");
        });
        refused_with(
            &succ_of(&fx),
            "missing row: no completed eval row tagged epoch-score-val",
        );
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let ctl = rid(0xf2, 3);
            row_mut(f, &ctl)["status"] = json!("killed");
        });
        refused_with(
            &succ_of(&fx),
            "missing row: no completed eval row tagged epoch-needle-length-control",
        );
    }

    #[test]
    fn a_missing_arm_letter_control_row_refuses_even_when_j6f_alone_would_fire() {
        // box_q_j6ctl.sh has not written J6(d)-v4's control: a both-win cannot be ruled out.
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |_, a| {
            let letter = rid(0xa6d, 4);
            a.retain(|r| r["row_id"] != letter.as_str());
        });
        refused_with(
            &succ_of(&fx),
            "missing row: no completed letter-control row",
        );
    }

    #[test]
    fn letter_control_rows_are_selected_by_the_per_family_key() {
        // An older control row of the same eval without the per-family key (J4's d597ee7d) is
        // not a candidate; a second one with it refuses.
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let ev = rid(0xf0, 2);
            let mut old = s_letter(&rid(0xb0, 9), 0, &ev, 0);
            old["metrics"].as_object_mut().unwrap().remove(MARGIN_KEY);
            f.push(old);
        });
        assert_eq!(succ_of(&fx).word, "fires:j6f");
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let ev = rid(0xf0, 2);
            f.push(s_letter(&rid(0xb0, 9), 0, &ev, 873));
        });
        refused_with(&succ_of(&fx), "2 completed letter-control rows");
        // The option control (linear_option_control.scored_eval_row_id) is never selected.
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let ev = rid(0xf0, 2);
            let mut opt = s_letter(&rid(0xb0, 9), 0, &ev, 873);
            let m = opt["metrics"].as_object_mut().unwrap();
            let v = m.remove("scored_eval_row_id").unwrap();
            m.insert("linear_option_control.scored_eval_row_id".into(), v);
            f.push(opt);
        });
        assert_eq!(succ_of(&fx).word, "fires:j6f");
    }

    #[test]
    fn a_margin_off_its_integer_grid_refuses() {
        let fx = succ_fixture(F_ENV, NEUTRAL, J6DV4_WINS, |_, a| {
            let letter = rid(0xa6d, 4);
            row_mut(a, &letter)["metrics"][MARGIN_KEY]["value"] = json!(0.391);
        });
        refused_with(&succ_of(&fx), "is not m/2304 for any integer m");
    }

    #[test]
    fn the_envelope_is_fs_seeds_0_1_2_only() {
        // Seeds 3 and 4 in F's ledger (rule (ii)) change nothing: they are never read.
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let low = Profile {
                needle: (4, 0),
                prose: 0,
                ..NEUTRAL
            };
            s_run(f, 0xf3, 3, "r", &low, |_| {});
            s_run(f, 0xf4, 4, "r", &low, |_| {});
        });
        let o = succ_of(&fx);
        assert_eq!(o.word, "fires:j6f", "{}", o.json);
        let seeds: Vec<i64> = o.json["detail"]["envelope"]["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["seed"].as_i64().unwrap())
            .collect();
        assert_eq!(seeds, [0, 1, 2]);
        // Naming any other seed set refuses.
        for ft_rows in [
            vec![fx.f_ft[0].clone(), fx.f_ft[1].clone()],
            vec![
                fx.f_ft[0].clone(),
                fx.f_ft[1].clone(),
                fx.f_ft[2].clone(),
                (3, rid(0xf3, 1)),
            ],
            vec![fx.f_ft[1].clone(), fx.f_ft[0].clone(), fx.f_ft[2].clone()],
            vec![fx.f_ft[0].clone(), fx.f_ft[1].clone(), (3, rid(0xf3, 1))],
        ] {
            let o = outcome(Cmd::Successor {
                f_ledger: fx.f.0.clone(),
                ft_rows,
                arm_ledger: fx.arms.0.clone(),
                j6f_ft_row: fx.j6f.clone(),
                j6dv4_ft_row: fx.j6dv4.clone(),
                out: PathBuf::from("/unused"),
            });
            refused_with(&o, "the envelope is F seeds 0, 1 and 2 only");
        }
    }

    #[test]
    fn swapped_or_wrong_arm_rows_refuse() {
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, no_edit);
        let swapped = outcome(Cmd::Successor {
            f_ledger: fx.f.0.clone(),
            ft_rows: fx.f_ft.clone(),
            arm_ledger: fx.arms.0.clone(),
            j6f_ft_row: fx.j6dv4.clone(),
            j6dv4_ft_row: fx.j6f.clone(),
            out: PathBuf::from("/unused"),
        });
        refused_with(&swapped, "but the arm drops it");
        // An arm that changed anything beyond its delta is not that arm.
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |_, a| {
            let ft = rid(0xa6f, 1);
            row_mut(a, &ft)["recipe"]["batch_tokens"] = json!(16384);
        });
        refused_with(
            &succ_of(&fx),
            "recipe.batch_tokens is 16384, not the envelope's 35403",
        );
        // An arm on another data snapshot, or scored on another val set, is not comparable.
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |_, a| {
            let ft = rid(0xa6f, 1);
            row_mut(a, &ft)["protocol"]["data_snapshot_hash"] = json!("other");
        });
        refused_with(&succ_of(&fx), "is not the envelope's");
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |_, a| {
            let ev = rid(0xa6f, 2);
            row_mut(a, &ev)["recipe"]["val_shard_hash"] = json!("w");
        });
        refused_with(&succ_of(&fx), "recipe.val_shard_hash");
        // A seed other than 0 for an arm refuses.
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, no_edit);
        let o = outcome(Cmd::Successor {
            f_ledger: fx.f.0.clone(),
            ft_rows: fx.f_ft.clone(),
            arm_ledger: fx.arms.0.clone(),
            j6f_ft_row: (1, fx.j6f.1.clone()),
            j6dv4_ft_row: fx.j6dv4.clone(),
            out: PathBuf::from("/unused"),
        });
        refused_with(&o, "each arm is one seed-0 run");
    }

    /// J6(a) gets its own identity (campaign/j6a-preregistered.json, Fable Q4); arm_identity's
    /// snapshot equality for J6(f) and J6(d)-v4 is not loosened. A J6(f) or J6(d)-v4 ft row
    /// shaped like J6(a)'s (build 2's snapshot, the five replay keys) still refuses through the
    /// successor, on the snapshot, even when its recipe otherwise matches its delta.
    #[test]
    fn the_successor_still_refuses_an_arm_on_another_snapshot() {
        for arm in [0xa6f, 0xa6d] {
            let fx = succ_fixture(F_ENV, J6F_WINS, J6DV4_WINS, |_, a| {
                let ft = rid(arm, 1);
                let row = row_mut(a, &ft);
                row["protocol"]["data_snapshot_hash"] = json!("b2".repeat(32));
                let r = row["recipe"].as_object_mut().unwrap();
                r.insert("replay_shard_hash".into(), json!("7e".repeat(32)));
                r.insert("replay_attestation_sha256".into(), json!("a7".repeat(32)));
                r.insert("replay_weight".into(), json!(1.0));
                r.insert("replay_every".into(), json!(6));
                r.insert("replay_direction".into(), json!("base_to_model"));
            });
            let o = succ_of(&fx);
            refused_with(&o, "is not the envelope's");
            refused_with(&o, &format!("ft row {}: data snapshot", rid(arm, 1)));
        }
        // The snapshot alone is enough.
        let fx = succ_fixture(F_ENV, NEUTRAL, J6DV4_WINS, |_, a| {
            let ft = rid(0xa6d, 1);
            row_mut(a, &ft)["protocol"]["data_snapshot_hash"] = json!("b2".repeat(32));
        });
        refused_with(&succ_of(&fx), "is not the envelope's");
    }

    #[test]
    fn an_envelope_seed_that_is_quick_or_off_recipe_refuses() {
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let ft = rid(0xf1, 1);
            row_mut(f, &ft)["quick"] = json!(true);
        });
        refused_with(&succ_of(&fx), "does not say quick: false");
        let fx = succ_fixture(F_ENV, J6F_WINS, NEUTRAL, |f, _| {
            let ft = rid(0xf2, 1);
            row_mut(f, &ft)["protocol"]["recipe_hash"] = json!("other");
        });
        refused_with(&succ_of(&fx), "not one configuration");
    }

    // --- j6a: campaign/j6a-preregistered.json ------------------------------------------------
    //
    // F's envelope from the same synthetic F ledger; J6(a)'s ft and eval rows in their own
    // ledger; the pre-registration is the committed file plus a filled top-level `amendments`.

    const J6A_PREREG: &str = "campaign/j6a-preregistered.json";
    /// F's ledger as pulled at 2c52b7a (seed 0's five rows; sha256 abb48449...).
    const F_LEDGER_V4: &str = "ledger/gh200-p4-v4-2026-10-01.jsonl";
    /// F's data snapshot, which J6(a)'s must differ from (arm.identity.ft_row).
    const F_DATA_SNAPSHOT_HASH: &str =
        "ea3215c4f36d57f74d291fb94c3fa8724fa5a14a303ea7572aa0dafb2a0933a1";
    /// F's val shard (arm.identity.eval_row).
    const F_VAL_SHARD_HASH: &str =
        "ef06ab99eef107dd608424c80117b81b3986ce5ea827d2d4f47f990d960b1bc5";

    fn prereg_j6a() -> Value {
        serde_json::from_str(&std::fs::read_to_string(repo(J6A_PREREG)).unwrap()).unwrap()
    }

    fn j6a_pins() -> Pins {
        Pins {
            data_snapshot_hash: "b2".repeat(32),
            shard_hash: "5b".repeat(32),
            replay_shard_hash: "7e".repeat(32),
            replay_attestation_sha256: "a7".repeat(32),
            h: 185,
            hit_list_sha256: "41".repeat(32),
        }
    }

    /// J6(a)'s recipe as a502670 would record it: F's, with build 2's shard hash, the five
    /// replay keys, and the batch planner's own batches and width (R1: not compared).
    fn j6a_recipe(r: &mut Map<String, Value>) {
        let p = j6a_pins();
        r.insert("shard_hash".into(), json!(p.shard_hash));
        r.insert("replay_shard_hash".into(), json!(p.replay_shard_hash));
        r.insert(
            "replay_attestation_sha256".into(),
            json!(p.replay_attestation_sha256),
        );
        r.insert("replay_weight".into(), json!(1.0));
        r.insert("replay_every".into(), json!(6));
        r.insert("replay_direction".into(), json!("base_to_model"));
        r.insert("batches".into(), json!(9214));
        r.insert("width".into(), json!(7680));
    }

    struct J6a {
        f: Temp,
        arm: Temp,
        prereg: Temp,
        f_ft: Vec<(i64, String)>,
        ft: (i64, String),
        cli: Pins,
    }

    /// `edit` gets F's rows, J6(a)'s rows and the pre-registration, in that order.
    fn j6a_fixture(
        env: [Profile; 3],
        arm: Profile,
        edit: impl Fn(&mut Vec<Value>, &mut Vec<Value>, &mut Value),
    ) -> J6a {
        let mut f_rows = Vec::new();
        let f_ft: Vec<(i64, String)> = env
            .iter()
            .enumerate()
            .map(|(s, p)| {
                (
                    s as i64,
                    s_run(&mut f_rows, 0xf0 + s as u64, s as i64, "r", p, |_| {}),
                )
            })
            .collect();
        let ft = rid(0xa6a, 1);
        let mut a_rows = vec![
            s_ft(&ft, 0, "ra", j6a_recipe),
            s_eval(&rid(0xa6a, 2), 0, &ft, &arm),
        ];
        a_rows[0]["quick"] = json!(true);
        a_rows[0]["protocol"]["data_snapshot_hash"] = json!(j6a_pins().data_snapshot_hash);
        let mut prereg = prereg_j6a();
        prereg["amendments"] = j6a_pins().json();
        edit(&mut f_rows, &mut a_rows, &mut prereg);
        let p = temp_path("j6a-preregistered.json");
        std::fs::write(&p.0, serde_json::to_vec_pretty(&prereg).unwrap()).unwrap();
        J6a {
            f: temp_ledger(&f_rows),
            arm: temp_ledger(&a_rows),
            prereg: p,
            f_ft,
            ft: (0, ft),
            cli: j6a_pins(),
        }
    }

    fn j6a_of(fx: &J6a) -> Outcome {
        let c = &fx.cli;
        outcome(Cmd::J6a {
            preregistration: fx.prereg.0.clone(),
            f_ledger: fx.f.0.clone(),
            ft_rows: fx.f_ft.clone(),
            arm_ledger: fx.arm.0.clone(),
            j6a_ft_row: fx.ft.clone(),
            data_snapshot_hash: c.data_snapshot_hash.clone(),
            shard_hash: c.shard_hash.clone(),
            replay_shard_hash: c.replay_shard_hash.clone(),
            replay_attestation_sha256: c.replay_attestation_sha256.clone(),
            h: c.h,
            hit_list_sha256: c.hit_list_sha256.clone(),
            out: PathBuf::from("/unused"),
        })
    }

    fn j6a(env: [Profile; 3], arm: Profile) -> Outcome {
        j6a_of(&j6a_fixture(env, arm, |_, _, _| {}))
    }

    /// J6(a) clearing all four targets against F_ENV: permutation 1211 > 2*1180 - 1150 = 1210
    /// and 1081 > 2*1060 - 1040 = 1080; in-distribution abstention 269 < 2*300 - 330 = 270 and
    /// 129 < 2*140 - 150 = 130.
    const J6A_WINS: Profile = Profile {
        perm_know: 1211,
        perm_csqa: 1081,
        id_know: 269,
        id_csqa: 129,
        ..NEUTRAL
    };

    fn arm_j6a(o: &Outcome) -> &Value {
        &o.json["detail"]["arms"]["j6a"]
    }

    #[test]
    fn j6a_wins_or_is_quiet_in_its_own_words_and_records_what_it_applied() {
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, _, _| {});
        let o = j6a_of(&fx);
        assert_eq!((o.word.as_str(), o.refused), ("wins", false), "{}", o.json);
        assert_eq!(o.json["rule"], "j6a_replay");
        assert_eq!(o.json["preregistration"], PREREG_J6A);
        let a = arm_j6a(&o);
        assert_eq!(a["wins"], true);
        assert_eq!(a["targets"].as_array().unwrap().len(), 4);
        assert_eq!(a["must_not_lose"].as_array().unwrap().len(), 9);
        // The identity is J6(a)'s own, and R1's derived keys are recorded, not compared.
        let id = &a["recipe_delta"];
        assert!(id["identity"].as_str().unwrap().contains("not R8"), "{id}");
        assert_eq!(
            id["derived_recorded_not_compared"]["batches"],
            json!({"j6a": 9214, "f": 9683})
        );
        assert_eq!(id["replay"]["replay_every"], 6);
        // The pre-registration's sha256 is in the JSON twice: as an input and in the detail.
        let sha = sha256_hex(&std::fs::read(&fx.prereg.0).unwrap());
        assert_eq!(o.json["detail"]["preregistration_sha256"], sha);
        assert_eq!(o.json["inputs"][0]["role"], "preregistration");
        assert_eq!(o.json["inputs"][0]["sha256"], sha);
        assert_eq!(o.json["detail"]["amendments"]["pinned"], j6a_pins().json());
        assert!(
            o.json["detail"]["amendments"]["h_and_hit_list_checked_against"]
                .as_str()
                .unwrap()
                .contains("no ledger row records them")
        );
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
        // Inside the envelope everywhere: quiet.
        let o = j6a(F_ENV, NEUTRAL);
        assert_eq!((o.word.as_str(), o.refused), ("quiet", false), "{}", o.json);
        assert_eq!(arm_j6a(&o)["targets_all_clear"], false);
    }

    /// R7 and R3 on each of the four targets: exactly max + range (min - range) does not clear,
    /// one count past it does, and one target short of clearing is quiet.
    #[test]
    fn every_j6a_target_ties_at_its_threshold_and_clears_one_count_past_it() {
        for (tie, metric) in [
            (
                Profile {
                    perm_know: 1210,
                    ..J6A_WINS
                },
                PERM_KNOWLEDGE.name,
            ),
            (
                Profile {
                    perm_csqa: 1080,
                    ..J6A_WINS
                },
                PERM_COMMONSENSE.name,
            ),
            (
                Profile {
                    id_know: 270,
                    ..J6A_WINS
                },
                ID_ABSTAIN_KNOWLEDGE.name,
            ),
            (
                Profile {
                    id_csqa: 130,
                    ..J6A_WINS
                },
                ID_ABSTAIN_COMMONSENSE.name,
            ),
        ] {
            let o = j6a(F_ENV, tie);
            assert_eq!(o.word, "quiet", "{metric}: {}", o.json);
            let t = arm_j6a(&o)["targets"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["metric"] == metric)
                .unwrap()
                .clone();
            assert_eq!(t["clears"], false, "{metric}");
        }
        assert_eq!(j6a(F_ENV, J6A_WINS).word, "wins");
        // The lower-better targets clear downward only: a rise in abstention never clears.
        let o = j6a(
            F_ENV,
            Profile {
                id_know: 1485,
                ..J6A_WINS
            },
        );
        assert_eq!(o.word, "quiet");
    }

    type GuardEdit = fn(Profile) -> Profile;

    /// J6(a)'s and J6(g)'s nine guards against F_ENV: (name, at F's bound, one count or a 1e-4
    /// ECE step past it, a move the right way).
    fn guard_cases() -> [(&'static str, GuardEdit, GuardEdit, GuardEdit); 9] {
        [
            (
                "val_top1.choice",
                |p| Profile { choice: 9100, ..p },
                |p| Profile { choice: 9099, ..p },
                |p| Profile { choice: 10985, ..p },
            ),
            (
                "val_top1.span",
                |p| Profile { span: 6500, ..p },
                |p| Profile { span: 6499, ..p },
                |p| Profile { span: 7238, ..p },
            ),
            (
                PERM_DC.name,
                |p| Profile { perm: 2285, ..p },
                |p| Profile { perm: 2284, ..p },
                |p| Profile { perm: 2304, ..p },
            ),
            (
                ECE_DC_KEY,
                |p| Profile { ece: 0.0220, ..p },
                |p| Profile { ece: 0.0221, ..p },
                |p| Profile { ece: 0.0, ..p },
            ),
            (
                "needle_8k_worst_bucket",
                |p| Profile {
                    needle: (4, 40),
                    ..p
                },
                |p| Profile {
                    needle: (4, 39),
                    ..p
                },
                |p| Profile {
                    needle: (4, 61),
                    ..p
                },
            ),
            (
                ID_ABSTAIN_DC.name,
                |p| Profile {
                    id_abstain: 15,
                    ..p
                },
                |p| Profile {
                    id_abstain: 16,
                    ..p
                },
                |p| Profile { id_abstain: 0, ..p },
            ),
            (
                "ood_abstain.prose",
                |p| Profile { prose: 50, ..p },
                |p| Profile { prose: 49, ..p },
                |p| Profile { prose: 60, ..p },
            ),
            (
                "ood_abstain.scrambled",
                |p| Profile { scrambled: 55, ..p },
                |p| Profile { scrambled: 54, ..p },
                |p| Profile { scrambled: 60, ..p },
            ),
            (
                "ood_abstain.unseen-language",
                |p| Profile { unseen: 20, ..p },
                |p| Profile { unseen: 19, ..p },
                |p| Profile { unseen: 60, ..p },
            ),
        ]
    }

    /// Each of the nine guards, in its own direction: at F's bound it holds (wins), one count
    /// (or a 1e-4 ECE step) past it loses (quiet, naming it), and a move the right way holds.
    #[test]
    fn every_j6a_guard_holds_at_its_bound_and_loses_one_past_it() {
        for (name, at_bound, past, right_way) in guard_cases() {
            let o = j6a(F_ENV, at_bound(J6A_WINS));
            assert_eq!(o.word, "wins", "{name} at its bound: {}", o.json);
            let o = j6a(F_ENV, past(J6A_WINS));
            assert_eq!(o.word, "quiet", "{name} one past: {}", o.json);
            assert_eq!(arm_j6a(&o)["must_not_lose_lost"], json!([name]), "{name}");
            let o = j6a(F_ENV, right_way(J6A_WINS));
            assert_eq!(o.word, "wins", "{name} the right way: {}", o.json);
        }
    }

    /// The declared data delta: J6(a)'s snapshot is build 2's pinned one and is not F's.
    #[test]
    fn j6a_on_fs_snapshot_or_off_its_pin_refuses() {
        // Everything says build 2 is F's snapshot: the rows, the file and the flags agree, and
        // the checker still refuses, because J6(a)'s train set is F's minus the replay set.
        let mut fx = j6a_fixture(F_ENV, J6A_WINS, |f, a, p| {
            for s in 0..3 {
                row_mut(f, &rid(0xf0 + s, 1))["protocol"]["data_snapshot_hash"] =
                    json!(F_DATA_SNAPSHOT_HASH);
            }
            a[0]["protocol"]["data_snapshot_hash"] = json!(F_DATA_SNAPSHOT_HASH);
            p["amendments"]["data_snapshot_hash"] = json!(F_DATA_SNAPSHOT_HASH);
        });
        fx.cli.data_snapshot_hash = F_DATA_SNAPSHOT_HASH.to_string();
        let o = j6a_of(&fx);
        refused_with(&o, "(a) ");
        refused_with(&o, "is F's");
        // J6(a)'s row on a snapshot other than the pinned one.
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
            a[0]["protocol"]["data_snapshot_hash"] = json!("c3".repeat(32));
        });
        refused_with(&j6a_of(&fx), "is not build 2's pinned");
    }

    #[test]
    fn a_j6a_recipe_delta_outside_the_declared_keys_refuses() {
        type RecipeEdit = fn(&mut Map<String, Value>);
        let cases: [(RecipeEdit, &str); 11] = [
            (
                |r| {
                    r.insert("lr".into(), json!(3e-5));
                },
                "recipe.lr is",
            ),
            (
                |r| {
                    r.insert("beta2".into(), json!(0.95));
                },
                "recipe.beta2 is 0.95, not F seed 0's absent",
            ),
            (
                |r| {
                    r.remove("lower_layers_n");
                },
                "recipe.lower_layers_n is absent",
            ),
            (
                |r| {
                    r.remove("replay_every");
                },
                "recipe.replay_every is absent",
            ),
            (
                |r| {
                    r.insert("replay_weight".into(), json!(0.5));
                },
                "recipe.replay_weight is 0.5",
            ),
            (
                |r| {
                    r.insert("replay_every".into(), json!(7));
                },
                "recipe.replay_every is 7",
            ),
            (
                |r| {
                    r.insert("replay_direction".into(), json!("model_to_base"));
                },
                "recipe.replay_direction is",
            ),
            (
                |r| {
                    r.insert("replay_shard_hash".into(), json!("00".repeat(32)));
                },
                "recipe.replay_shard_hash is",
            ),
            (
                |r| {
                    r.insert("replay_attestation_sha256".into(), json!("00".repeat(32)));
                },
                "recipe.replay_attestation_sha256 is",
            ),
            (
                |r| {
                    r.insert("shard_hash".into(), json!("5f".repeat(32)));
                },
                "recipe.shard_hash is",
            ),
            (
                |r| {
                    r.insert("tag".into(), json!("epoch-x"));
                },
                "recipe.tag is",
            ),
        ];
        for (edit, phrase) in cases {
            let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
                edit(a[0]["recipe"].as_object_mut().unwrap());
            });
            let o = j6a_of(&fx);
            refused_with(&o, phrase);
            refused_with(&o, "(a) ");
        }
        // F's own recipe carrying a replay key is not F.
        let fx = j6a_fixture(F_ENV, J6A_WINS, |f, _, _| {
            row_mut(f, &rid(0xf0, 1))["recipe"]["replay_weight"] = json!(1.0);
        });
        refused_with(&j6a_of(&fx), "but F ran no replay");
    }

    #[test]
    fn j6a_must_be_quick_and_at_a502670() {
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
            a[0]["quick"] = json!(false);
        });
        refused_with(&j6a_of(&fx), "quick is false");
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
            a[0].as_object_mut().unwrap().remove("quick");
        });
        refused_with(&j6a_of(&fx), "quick is absent");
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
            a[0]["code_commit"] = json!("881ab304".repeat(5));
        });
        refused_with(&j6a_of(&fx), "is not a5026707");
    }

    /// While the file has no `amendments`, the checker refuses, even on a run that would win,
    /// and still decides room from the envelope and reports it.
    #[test]
    fn j6a_refuses_while_amendments_are_pending() {
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, _, p| {
            p.as_object_mut().unwrap().remove("amendments");
        });
        let o = j6a_of(&fx);
        refused_with(&o, "(a) the pre-registration's amendments are pending");
        refused_with(&o, "amendments_pending: \"The lead fills these");
        refused_with(&o, "an object with exactly data_snapshot_hash");
        assert_eq!(o.json["detail"]["room"].as_array().unwrap().len(), 4);
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
        assert!(o.json["detail"]["amendments"]["not_pinned"].is_string());
        // Pending and a missing J6(a) eval row: both are listed, not only the first one met.
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, p| {
            p.as_object_mut().unwrap().remove("amendments");
            a.truncate(1);
        });
        let o = j6a_of(&fx);
        refused_with(
            &o,
            "(a) missing row: no completed eval row tagged epoch-score-val",
        );
        refused_with(&o, "(a) the pre-registration's amendments are pending");
        // Pending and an envelope that does not resolve (today's state: F seeds 1 and 2 have no
        // rows): both are listed, though no room was decided.
        let fx = j6a_fixture(F_ENV, J6A_WINS, |f, _, p| {
            p.as_object_mut().unwrap().remove("amendments");
            f.retain(|r| r["protocol"]["seed"] == 0);
        });
        let o = j6a_of(&fx);
        refused_with(&o, "room not decided");
        refused_with(&o, "(a) the pre-registration's amendments are pending");
        // The committed file is in one of its two legitimate states, and this test needs no
        // edit when the lead amends it: pending (no `amendments`: refused as pending), or
        // amended (well-formed pins, build 2's snapshot not F's).
        let committed = prereg_j6a();
        let committed = committed.as_object().unwrap();
        match committed.get("amendments") {
            None => {
                let err = amendments_of(committed).unwrap_err();
                assert!(err.contains("amendments are pending"), "{err}");
            }
            Some(_) => {
                let pins = amendments_of(committed).unwrap();
                assert_ne!(pins.data_snapshot_hash, F_DATA_SNAPSHOT_HASH);
            }
        }
    }

    #[test]
    fn malformed_amendments_or_flags_that_disagree_with_them_refuse() {
        type PreregEdit = fn(&mut Value);
        let cases: [(PreregEdit, &str); 7] = [
            (
                |p| p["amendments"]["note"] = json!("filled 2026-10-02"),
                "carry [\"note\"], which are not among the six",
            ),
            (
                |p| {
                    p["amendments"].as_object_mut().unwrap().remove("h");
                },
                "have no integer h",
            ),
            (
                |p| {
                    p["amendments"]
                        .as_object_mut()
                        .unwrap()
                        .remove("hit_list_sha256");
                },
                "have no string hit_list_sha256",
            ),
            (
                |p| p["amendments"]["shard_hash"] = json!("5B".repeat(32)),
                "shard_hash \"5B5B",
            ),
            (
                |p| p["amendments"]["replay_shard_hash"] = json!("7e".repeat(31)),
                "is not a sha256 in lower-case hex",
            ),
            (
                |p| p["amendments"]["h"] = json!(3569),
                "h 3569 is more than the replay set's 3568 rows",
            ),
            (
                |p| p["amendments"] = json!("filled"),
                "amendments are \"filled\", not an object",
            ),
        ];
        for (edit, phrase) in cases {
            let fx = j6a_fixture(F_ENV, J6A_WINS, |_, _, p| edit(p));
            let o = j6a_of(&fx);
            refused_with(&o, phrase);
            refused_with(&o, "(a) ");
        }
        // Each flag must repeat the file's pin exactly.
        type CliEdit = fn(&mut Pins);
        let flags: [(CliEdit, &str); 6] = [
            (
                |c| c.data_snapshot_hash = "c3".repeat(32),
                "--data-snapshot-hash",
            ),
            (|c| c.shard_hash = "c3".repeat(32), "--shard-hash"),
            (
                |c| c.replay_shard_hash = "c3".repeat(32),
                "--replay-shard-hash",
            ),
            (
                |c| c.replay_attestation_sha256 = "c3".repeat(32),
                "--replay-attestation-sha256",
            ),
            (
                |c| c.h = 184,
                "--h 184 is not the pre-registration's pinned 185",
            ),
            (|c| c.hit_list_sha256 = "c3".repeat(32), "--hit-list-sha256"),
        ];
        for (edit, phrase) in flags {
            let mut fx = j6a_fixture(F_ENV, J6A_WINS, |_, _, _| {});
            edit(&mut fx.cli);
            refused_with(&j6a_of(&fx), phrase);
        }
        let mut fx = j6a_fixture(F_ENV, J6A_WINS, |_, _, _| {});
        fx.cli.shard_hash = "not-a-hash".into();
        refused_with(&j6a_of(&fx), "the command line's shard_hash");
    }

    /// R9_room on J6(a)'s targets, both directions, with its own label (b); a missing arm row
    /// beside it is listed too, as (a).
    #[test]
    fn a_ceilinged_j6a_target_refuses_under_r9_room() {
        // Higher-better at the ceiling: 2*1485 - 1300 = 1670 >= 1485.
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip([1485, 1300, 1400]) {
            p.perm_know = k;
        }
        let o = j6a(env, J6A_WINS);
        refused_with(
            &o,
            "(b) j6a target permutation_consistency.family.knowledge",
        );
        refused_with(&o, "2*max - min >= U = 1");
        assert_eq!(
            cannot_clear(&o),
            [("j6a".to_string(), PERM_KNOWLEDGE.name.to_string())]
        );
        // Lower-better at the floor: 2*0 - 20 <= 0.
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip([0, 20, 10]) {
            p.id_know = k;
        }
        let o = j6a(env, J6A_WINS);
        refused_with(&o, "2*min - max <= L = 0");
        assert_eq!(
            cannot_clear(&o),
            [("j6a".to_string(), ID_ABSTAIN_KNOWLEDGE.name.to_string())]
        );
        // With room (11..20: 2*11 - 20 = 2 > 0) the comparison decides: 1 < 2 clears.
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip([11, 20, 15]) {
            p.id_know = k;
        }
        let o = j6a(
            env,
            Profile {
                id_know: 1,
                ..J6A_WINS
            },
        );
        assert_eq!(o.word, "wins", "{}", o.json);
        // No room and a missing J6(a) eval row: both listed, room first.
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip([1485, 1300, 1400]) {
            p.perm_know = k;
        }
        let fx = j6a_fixture(env, J6A_WINS, |_, a, _| {
            a.truncate(1);
        });
        let o = j6a_of(&fx);
        let because: Vec<&str> = o.json["detail"]["refused_because"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        assert_eq!(because.len(), 2, "{because:?}");
        assert!(because[0].starts_with("(b) j6a target"));
        assert!(because[1].starts_with("(a) missing row"));
    }

    /// The file binds: a structured field that disagrees with the checker refuses before any
    /// ledger is read.
    #[test]
    fn a_preregistration_that_disagrees_with_the_checker_refuses() {
        type PreregEdit = fn(&mut Value);
        let cases: [(PreregEdit, &str); 7] = [
            (
                |p| p["outcomes"]["words"] = json!(["wins", "quiet"]),
                "outcomes.words are",
            ),
            (
                |p| {
                    p["arm"]["targets"].as_array_mut().unwrap().pop();
                },
                "arm.targets are",
            ),
            (
                |p| p["arm"]["targets"][2]["direction"] = json!("higher"),
                "arm.targets are",
            ),
            (
                |p| p["arm"]["guards"][3]["form"] = json!("count"),
                "arm.guards are",
            ),
            (
                |p| p["arm"]["replay_flags"]["replay_weight"] = json!(0.5),
                "arm.replay_flags",
            ),
            (
                |p| p["arm"]["replay_flags"]["replay_every"] = json!(7),
                "arm.replay_flags",
            ),
            (
                |p| p["arm"]["replay_flags"]["replay_direction"] = json!("model_to_base"),
                "arm.replay_flags",
            ),
        ];
        for (edit, phrase) in cases {
            let fx = j6a_fixture(F_ENV, J6A_WINS, |_, _, p| edit(p));
            let o = j6a_of(&fx);
            refused_with(&o, phrase);
            refused_with(&o, "(a) the pre-registration's");
            // Only the pre-registration was read.
            assert_eq!(o.json["inputs"].as_array().unwrap().len(), 1, "{}", o.json);
        }
    }

    #[test]
    fn j6a_eval_row_is_one_comparable_epoch_score_val_row() {
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
            a[1]["recipe"]["val_shard_hash"] = json!("w");
        });
        refused_with(&j6a_of(&fx), "recipe.val_shard_hash");
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
            a[1]["recipe"]["ood"]["cases_per_category"] = json!(30);
        });
        refused_with(&j6a_of(&fx), "recipe.ood");
        // R2: a second completed epoch-score-val row for the ft row refuses.
        let fx = j6a_fixture(F_ENV, J6A_WINS, |_, a, _| {
            let second = with_id(a[1].clone(), &rid(0xa6a, 3));
            a.push(second);
        });
        refused_with(&j6a_of(&fx), "(a) ");
        refused_with(&j6a_of(&fx), "epoch-score-val");
        // J6(a)'s rows are read from its own ledger only.
        let fx = j6a_fixture(F_ENV, J6A_WINS, |f, a, _| {
            f.append(a);
        });
        refused_with(&j6a_of(&fx), "(a) ");
    }

    /// "No needle-control or letter-control row is required": an envelope without them decides.
    #[test]
    fn j6a_needs_no_control_or_letter_row() {
        assert!(!ARM_J6A.reads_control() && !ARM_J6A.reads_margin());
        let fx = j6a_fixture(F_ENV, J6A_WINS, |f, _, _| {
            f.retain(|r| {
                r["recipe"]["tag"] != json!(CONTROL_TAG) && r["metrics"].get(MARGIN_KEY).is_none()
            });
        });
        let o = j6a_of(&fx);
        assert_eq!(o.word, "wins", "{}", o.json);
        assert_eq!(
            o.json["detail"]["envelope"]["seeds"][0]["letter_control_row"],
            Value::Null
        );
    }

    /// The checker's constants are the pre-registration's text, and its literals are what F
    /// seed 0's committed rows record.
    #[test]
    fn the_j6a_rule_is_the_preregistrations_own_text() {
        let p = prereg_j6a();
        let map = p.as_object().unwrap();
        j6a_agrees(map).unwrap();
        let id = &p["arm"]["identity"];
        let ft_text = id["ft_row"].as_str().unwrap();
        assert!(ft_text.contains(&format!("code_commit {J6A_CODE_COMMIT}")));
        assert!(ft_text.contains("quick true"));
        assert!(ft_text.contains(&format!("must differ from F's {F_DATA_SNAPSHOT_HASH}")));
        let recipe_text = id["recipe"].as_str().unwrap();
        assert!(recipe_text.contains(
            "replay_shard_hash, replay_attestation_sha256, replay_weight = 1.0, replay_every = 6, \
             replay_direction = 'base_to_model'"
        ));
        assert!(recipe_text.contains("shard_hash, which must equal build 2's train shard hash"));
        assert!(
            id["eval_row"]
                .as_str()
                .unwrap()
                .contains(&format!("recipe.val_shard_hash = {F_VAL_SHARD_HASH}"))
        );
        let r1 = p["readings"]["R1_derived_recipe_keys"].as_str().unwrap();
        for k in DERIVED_RECIPE_KEYS {
            assert!(r1.contains(&format!("recipe.{k}")), "{k}");
        }
        assert!(
            p["arm"]["declared_data_delta"]
                .as_str()
                .unwrap()
                .contains("exactly the 3,568 row_ids")
        );
        assert_eq!(REPLAY_SET_ROWS, 3568);
        let pending = p["amendments_pending"].as_str().unwrap();
        for phrase in [
            "train-manifest hash (data_snapshot_hash)",
            "its train shard hash",
            "the replay shard hash",
            "attestation sha256",
            "and h (with the full hit list's sha256)",
            "Until they are filled, the checker refuses",
        ] {
            assert!(pending.contains(phrase), "{phrase}");
        }
        assert!(
            p["envelope"]
                .as_str()
                .unwrap()
                .contains("F seeds 0, 1 and 2 only, never seeds 3-4")
        );
        assert!(
            p["outcomes"]["refused"]
                .as_str()
                .unwrap()
                .contains("(b) a target has no room under R9_room (cannot_clear)")
        );
        assert_eq!((J6A_LABELS.unreadable, J6A_LABELS.no_room), ("(a)", "(b)"));
        // The literals against F seed 0's committed rows (973cd4e3, f4feac15).
        let ledger = Inputs::default().read(&repo(F_LEDGER_V4)).unwrap();
        let rows = seed_rows(&ledger, 0, F_SEED0_FT, false, false).unwrap();
        assert_eq!(rows.ft.str_at(&["code_commit"]), Some(J6A_CODE_COMMIT));
        assert_eq!(
            rows.ft.str_at(&["protocol", "data_snapshot_hash"]),
            Some(F_DATA_SNAPSHOT_HASH)
        );
        assert_eq!(
            rows.eval.str_at(&["recipe", "val_shard_hash"]),
            Some(F_VAL_SHARD_HASH)
        );
        for k in REPLAY_KEYS {
            assert!(rows.ft.get(&["recipe", k]).is_none(), "{k}");
        }
        for k in DERIVED_RECIPE_KEYS.iter().chain(&["shard_hash", "tag"]) {
            assert!(rows.ft.get(&["recipe", k]).is_some(), "{k}");
        }
        // Every target and guard reads on f4feac15 with the value the file cites (f_seed0).
        for list in ["targets", "guards"] {
            for m in p["arm"][list].as_array().unwrap() {
                let name = m["name"].as_str().unwrap();
                let metric = ARM_J6A
                    .targets
                    .iter()
                    .copied()
                    .chain(ARM_J6A.must_not_lose())
                    .find(|x| x.name == name)
                    .unwrap();
                let shown = match rows.read(metric).unwrap() {
                    Val::Exact(r) => format!("{}/{}", r.num, r.den),
                    Val::Float(x) => x.to_string(),
                };
                assert_eq!(shown, m["f_seed0"].as_str().unwrap(), "{name}");
            }
        }
    }

    // --- j6g: campaign/j6g-preregistered.json ------------------------------------------------
    //
    // F's envelope from the same synthetic F ledger, with seed 0's ft row renamed to 973cd4e3
    // (arm.identity.recipe names it) and every row on F's data snapshot and val shard; J6(g)'s
    // ft and eval rows in their own ledger; the pre-registration is the committed file.

    const J6G_PREREG: &str = "campaign/j6g-preregistered.json";

    fn prereg_j6g() -> Value {
        serde_json::from_str(&std::fs::read_to_string(repo(J6G_PREREG)).unwrap()).unwrap()
    }

    /// J6(g)'s recipe as a502670 records it with the flag on: F's, plus the one key
    /// (tools/real_ft_run.py:356-357 at a502670).
    fn j6g_recipe(r: &mut Map<String, Value>) {
        r.insert(
            OPTION_PERMUTATION_KEY.into(),
            json!(OPTION_PERMUTATION_SEED),
        );
    }

    struct J6g {
        f: Temp,
        arm: Temp,
        prereg: Temp,
        f_ft: Vec<(i64, String)>,
        ft: (i64, String),
    }

    /// `edit` gets F's rows, J6(g)'s rows (ft, then eval) and the pre-registration, in that order.
    fn j6g_fixture(
        env: [Profile; 3],
        arm: Profile,
        edit: impl Fn(&mut Vec<Value>, &mut Vec<Value>, &mut Value),
    ) -> J6g {
        let mut f_rows = Vec::new();
        let mut f_ft: Vec<(i64, String)> = env
            .iter()
            .enumerate()
            .map(|(s, p)| {
                (
                    s as i64,
                    s_run(&mut f_rows, 0xf0 + s as u64, s as i64, "r", p, |_| {}),
                )
            })
            .collect();
        let seed0 = f_ft[0].1.clone();
        for r in &mut f_rows {
            if r["row_id"] == seed0.as_str() {
                r["row_id"] = json!(F_SEED0_FT);
            }
            if r["metrics"]["ft_run_row_id"]["value"] == seed0.as_str() {
                r["metrics"]["ft_run_row_id"]["value"] = json!(F_SEED0_FT);
            }
        }
        f_ft[0].1 = F_SEED0_FT.to_string();
        let ft = rid(0xa6b, 1);
        let mut a_rows = vec![
            s_ft(&ft, 0, "rg", j6g_recipe),
            s_eval(&rid(0xa6b, 2), 0, &ft, &arm),
        ];
        a_rows[0]["quick"] = json!(true);
        for r in f_rows.iter_mut().chain(a_rows.iter_mut()) {
            if r["run_kind"] == "ft" {
                r["protocol"]["data_snapshot_hash"] = json!(F_DATA_SNAPSHOT_HASH);
            }
            if r["recipe"]["tag"] == EVAL_TAG {
                r["recipe"]["val_shard_hash"] = json!(F_VAL_SHARD_HASH);
            }
        }
        let mut prereg = prereg_j6g();
        edit(&mut f_rows, &mut a_rows, &mut prereg);
        let p = temp_path("j6g-preregistered.json");
        std::fs::write(&p.0, serde_json::to_vec_pretty(&prereg).unwrap()).unwrap();
        J6g {
            f: temp_ledger(&f_rows),
            arm: temp_ledger(&a_rows),
            prereg: p,
            f_ft,
            ft: (0, ft),
        }
    }

    fn j6g_of(fx: &J6g) -> Outcome {
        outcome(Cmd::J6g {
            preregistration: fx.prereg.0.clone(),
            f_ledger: fx.f.0.clone(),
            ft_rows: fx.f_ft.clone(),
            arm_ledger: fx.arm.0.clone(),
            j6g_ft_row: fx.ft.clone(),
            out: PathBuf::from("/unused"),
        })
    }

    fn j6g(env: [Profile; 3], arm: Profile) -> Outcome {
        j6g_of(&j6g_fixture(env, arm, |_, _, _| {}))
    }

    fn j6g_edited(edit: impl Fn(&mut Vec<Value>, &mut Vec<Value>, &mut Value)) -> Outcome {
        j6g_of(&j6g_fixture(F_ENV, J6G_WINS, edit))
    }

    /// J6(g) has J6(a)'s targets, so the same profile clears all four against F_ENV.
    const J6G_WINS: Profile = J6A_WINS;

    fn arm_j6g(o: &Outcome) -> &Value {
        &o.json["detail"]["arms"]["j6g"]
    }

    #[test]
    fn j6g_wins_or_is_quiet_in_its_own_words_and_records_what_it_applied() {
        let fx = j6g_fixture(F_ENV, J6G_WINS, |_, _, _| {});
        let o = j6g_of(&fx);
        assert_eq!((o.word.as_str(), o.refused), ("wins", false), "{}", o.json);
        assert_eq!(o.json["rule"], "j6g_option_permutation");
        assert_eq!(o.json["preregistration"], PREREG_J6G);
        let a = arm_j6g(&o);
        assert_eq!(a["wins"], true);
        assert_eq!(a["targets"].as_array().unwrap().len(), 4);
        assert_eq!(a["must_not_lose"].as_array().unwrap().len(), 9);
        let id = &a["recipe_delta"];
        assert_eq!(id["added"], json!({"option_permutation_seed": 20260919}));
        assert_eq!(id["f_seed0_ft_row"], F_SEED0_FT);
        assert_eq!(id["data_snapshot_hash"], F_DATA_SNAPSHOT_HASH);
        assert!(
            id["identity"]
                .as_str()
                .unwrap()
                .contains("plus exactly one key")
        );
        let compared: Vec<&str> = id["compared_equal"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k.as_str().unwrap())
            .collect();
        for k in ["batches", "width", "shard_hash", "lower_layers_n", "tag"] {
            assert!(compared.contains(&k), "{k}: {compared:?}");
        }
        let sha = sha256_hex(&std::fs::read(&fx.prereg.0).unwrap());
        assert_eq!(o.json["detail"]["preregistration_sha256"], sha);
        assert_eq!(o.json["inputs"][0]["role"], "preregistration");
        assert_eq!(o.json["inputs"][0]["sha256"], sha);
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
        assert_eq!(o.json["detail"]["room"].as_array().unwrap().len(), 4);
        // Inside the envelope everywhere: quiet, which is a word, not a refusal.
        let o = j6g(F_ENV, NEUTRAL);
        assert_eq!((o.word.as_str(), o.refused), ("quiet", false), "{}", o.json);
        assert_eq!(arm_j6g(&o)["targets_all_clear"], false);
        assert_eq!(arm_j6g(&o)["must_not_lose_lost"], json!([]));
    }

    /// R7 and R3 on each target: exactly max + range (min - range) does not clear, one count past
    /// it does, and one target short of clearing is quiet even with the other three clearing.
    #[test]
    fn every_j6g_target_ties_at_its_threshold_and_clears_one_count_past_it() {
        for (tie, metric) in [
            (
                Profile {
                    perm_know: 1210,
                    ..J6G_WINS
                },
                PERM_KNOWLEDGE.name,
            ),
            (
                Profile {
                    perm_csqa: 1080,
                    ..J6G_WINS
                },
                PERM_COMMONSENSE.name,
            ),
            (
                Profile {
                    id_know: 270,
                    ..J6G_WINS
                },
                ID_ABSTAIN_KNOWLEDGE.name,
            ),
            (
                Profile {
                    id_csqa: 130,
                    ..J6G_WINS
                },
                ID_ABSTAIN_COMMONSENSE.name,
            ),
        ] {
            let o = j6g(F_ENV, tie);
            assert_eq!(
                (o.word.as_str(), o.refused),
                ("quiet", false),
                "{metric}: {}",
                o.json
            );
            let targets = arm_j6g(&o)["targets"].as_array().unwrap().clone();
            for t in &targets {
                assert_eq!(t["clears"], t["metric"] != metric, "{metric}: {t}");
            }
        }
        // One past each tie clears: J6G_WINS is every tie plus one count.
        assert_eq!(j6g(F_ENV, J6G_WINS).word, "wins");
        // A lower-better target moving up never clears; a higher-better one moving down never.
        let o = j6g(
            F_ENV,
            Profile {
                id_csqa: 1197,
                ..J6G_WINS
            },
        );
        assert_eq!(o.word, "quiet");
        let o = j6g(
            F_ENV,
            Profile {
                perm_csqa: 0,
                ..J6G_WINS
            },
        );
        assert_eq!(o.word, "quiet");
    }

    /// A guard that loses makes a run whose targets all clear quiet, naming it; at F's bound it
    /// holds. All nine guards, each in its own direction.
    #[test]
    fn every_j6g_guard_holds_at_its_bound_and_loses_one_past_it() {
        for (name, at_bound, past, right_way) in guard_cases() {
            let o = j6g(F_ENV, at_bound(J6G_WINS));
            assert_eq!(o.word, "wins", "{name} at its bound: {}", o.json);
            let o = j6g(F_ENV, past(J6G_WINS));
            assert_eq!(
                (o.word.as_str(), o.refused),
                ("quiet", false),
                "{name}: {}",
                o.json
            );
            assert_eq!(arm_j6g(&o)["must_not_lose_lost"], json!([name]), "{name}");
            assert_eq!(arm_j6g(&o)["targets_all_clear"], true, "{name}");
            let o = j6g(F_ENV, right_way(J6G_WINS));
            assert_eq!(o.word, "wins", "{name} the right way: {}", o.json);
        }
    }

    /// R9_room on J6(g)'s targets, both directions, with its own label (b), decided from the
    /// envelope alone: a candidate that would otherwise win refuses, and a missing arm row
    /// beside it is listed too, as (a).
    #[test]
    fn a_ceilinged_j6g_target_refuses_under_r9_room() {
        // Higher-better at the ceiling: 2*1485 - 1300 = 1670 >= 1485.
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip([1485, 1300, 1400]) {
            p.perm_know = k;
        }
        let o = j6g(env, J6G_WINS);
        refused_with(
            &o,
            "(b) j6g target permutation_consistency.family.knowledge",
        );
        refused_with(&o, "2*max - min >= U = 1");
        assert_eq!(
            cannot_clear(&o),
            [("j6g".to_string(), PERM_KNOWLEDGE.name.to_string())]
        );
        // Lower-better at the floor: 2*0 - 20 <= 0.
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip([0, 20, 10]) {
            p.id_csqa = k;
        }
        let o = j6g(env, J6G_WINS);
        refused_with(&o, "2*min - max <= L = 0");
        assert_eq!(
            cannot_clear(&o),
            [("j6g".to_string(), ID_ABSTAIN_COMMONSENSE.name.to_string())]
        );
        // No room and a missing J6(g) eval row: both listed, room first.
        let mut env = F_ENV;
        for (p, k) in env.iter_mut().zip([1485, 1300, 1400]) {
            p.perm_know = k;
        }
        let o = j6g_of(&j6g_fixture(env, J6G_WINS, |_, a, _| a.truncate(1)));
        let because: Vec<&str> = o.json["detail"]["refused_because"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        assert_eq!(because.len(), 2, "{because:?}");
        assert!(because[0].starts_with("(b) j6g target"), "{because:?}");
        assert!(because[1].starts_with("(a) missing row"), "{because:?}");
    }

    /// arm.identity.recipe: F seed 0's recipe plus exactly option_permutation_seed = 20260919.
    /// A missing key, a wrong value, an extra key or any other changed key refuses as (a); so do
    /// another data snapshot, another code commit and a run that is not quick.
    #[test]
    fn each_j6g_identity_difference_refuses() {
        type RowsEdit = fn(&mut Vec<Value>, &mut Vec<Value>);
        let cases: [(RowsEdit, &str); 17] = [
            (
                |_, a| {
                    a[0]["recipe"]
                        .as_object_mut()
                        .unwrap()
                        .remove(OPTION_PERMUTATION_KEY);
                },
                "recipe.option_permutation_seed is absent, not the pre-registered 20260919",
            ),
            (
                |_, a| a[0]["recipe"][OPTION_PERMUTATION_KEY] = json!(20260920),
                "recipe.option_permutation_seed is 20260920, not the pre-registered 20260919",
            ),
            (
                |_, a| a[0]["recipe"][OPTION_PERMUTATION_KEY] = json!(20260919.0),
                "recipe.option_permutation_seed is 20260919.0",
            ),
            (
                |_, a| a[0]["recipe"][OPTION_PERMUTATION_KEY] = json!("20260919"),
                "recipe.option_permutation_seed is \"20260919\"",
            ),
            (
                |_, a| a[0]["recipe"]["beta2"] = json!(0.95),
                "recipe.beta2 is 0.95, not F seed 0's absent",
            ),
            (
                |_, a| a[0]["recipe"]["replay_weight"] = json!(1.0),
                "recipe.replay_weight is 1.0, not F seed 0's absent",
            ),
            (
                |_, a| a[0]["recipe"]["lr"] = json!(3e-5),
                "recipe.lr is 0.00003, not F seed 0's 0.00001",
            ),
            (
                |_, a| {
                    a[0]["recipe"]
                        .as_object_mut()
                        .unwrap()
                        .remove("lower_layers_n");
                },
                "recipe.lower_layers_n is absent, not F seed 0's 8",
            ),
            (
                |_, a| a[0]["recipe"]["batches"] = json!(9682),
                "recipe.batches is 9682, not F seed 0's 9683",
            ),
            (
                |_, a| a[0]["recipe"]["width"] = json!(8192),
                "recipe.width is 8192, not F seed 0's 7936",
            ),
            (
                |_, a| a[0]["recipe"]["shard_hash"] = json!("5b".repeat(32)),
                "recipe.shard_hash is",
            ),
            (
                |_, a| a[0]["protocol"]["data_snapshot_hash"] = json!("c3".repeat(32)),
                "is not F's ea3215c4",
            ),
            (
                |_, a| a[0]["protocol"]["tokenizer_hash"] = json!("t"),
                "protocol.tokenizer_hash is \"t\", not F seed 0's absent",
            ),
            (
                |_, a| a[0]["code_commit"] = json!("881ab304".repeat(5)),
                "code_commit Some(\"881ab304881ab304881ab304881ab304881ab304\") is not a5026707",
            ),
            (|_, a| a[0]["quick"] = json!(false), "quick is false"),
            (
                |_, a| {
                    a[0].as_object_mut().unwrap().remove("quick");
                },
                "quick is absent",
            ),
            (
                |f, _| row_mut(f, F_SEED0_FT)["recipe"][OPTION_PERMUTATION_KEY] = json!(1),
                "but F trained without option permutation",
            ),
        ];
        for (edit, phrase) in cases {
            let o = j6g_edited(|f, a, _| edit(f, a));
            refused_with(&o, phrase);
            refused_with(&o, "(a) arm j6g ft row ");
            assert!(arm_j6g(&o).is_null(), "{phrase}: judged though refused");
        }
        // An F ledger whose seed-0 recipe is not the pre-registration's 9,683 x 7,936 refuses,
        // even when J6(g)'s row matches it key for key.
        let o = j6g_edited(|f, a, _| {
            row_mut(f, F_SEED0_FT)["recipe"]["batches"] = json!(9682);
            a[0]["recipe"]["batches"] = json!(9682);
        });
        refused_with(
            &o,
            "envelope ft row 973cd4e3-e0d2-4ff8-8588-b761cb842b75: recipe.batches is 9682, not \
             F's 9683",
        );
        assert_eq!(o.json["refused"].as_str().unwrap().matches("; ").count(), 0);
        // F's recipe carrying the very key and value J6(g) adds is still not F: refused, and
        // that is the one failure.
        let o = j6g_edited(|f, _, _| {
            row_mut(f, F_SEED0_FT)["recipe"][OPTION_PERMUTATION_KEY] =
                json!(OPTION_PERMUTATION_SEED);
        });
        refused_with(&o, "but F trained without option permutation");
        assert_eq!(o.json["refused"].as_str().unwrap().matches("; ").count(), 0);
    }

    /// The recipe J6(g) is compared with is F seed 0's 973cd4e3 and no other row; and every
    /// identity failure is listed, not only the first.
    #[test]
    fn j6g_is_compared_with_fs_seed_0_row_and_lists_every_identity_failure() {
        let row = |v: Value| Row {
            at: "test".into(),
            map: v.as_object().unwrap().clone(),
        };
        let mut f = s_ft(&rid(0xf0, 1), 0, "r", |_| {});
        f["protocol"]["data_snapshot_hash"] = json!(F_DATA_SNAPSHOT_HASH);
        let mut g = s_ft(&rid(0xa6b, 1), 0, "rg", j6g_recipe);
        g["quick"] = json!(true);
        g["protocol"]["data_snapshot_hash"] = json!(F_DATA_SNAPSHOT_HASH);
        let err = option_identity(&ARM_J6G, &row(g.clone()), &row(f.clone())).unwrap_err();
        assert!(
            err.contains("the envelope's seed-0 ft row is 000000f0-0000-4000-8000-000000000001, not F seed 0's 973cd4e3"),
            "{err}"
        );
        f = with_id(f, F_SEED0_FT);
        assert!(option_identity(&ARM_J6G, &row(g.clone()), &row(f.clone())).is_ok());
        // Not quick, another commit and no key: three failures in one message.
        g["quick"] = json!(false);
        g["code_commit"] = json!("0".repeat(40));
        g["recipe"]
            .as_object_mut()
            .unwrap()
            .remove(OPTION_PERMUTATION_KEY);
        let err = option_identity(&ARM_J6G, &row(g), &row(f)).unwrap_err();
        for phrase in [
            "quick is false",
            "code_commit Some(\"0000000000000000000000000000000000000000\") is not",
            "recipe.option_permutation_seed is absent",
        ] {
            assert!(err.contains(phrase), "{phrase}: {err}");
        }
        assert_eq!(err.matches("; ").count(), 2, "{err}");
    }

    /// A required row missing, doubled or not completed, a metric that did not run or is absent,
    /// and a count off its own value: each refuses as (a), never quiet.
    #[test]
    fn a_missing_doubled_or_incomplete_j6g_row_refuses() {
        type AllEdit = fn(&mut Vec<Value>, &mut Vec<Value>);
        let cases: [(AllEdit, &str); 12] = [
            (
                |_, a| a.truncate(1),
                "(a) missing row: no completed eval row tagged epoch-score-val",
            ),
            (
                |_, a| drop(a.remove(0)),
                "(a) row 00000a6b-0000-4000-8000-000000000001 appears 0 time(s)",
            ),
            (
                |_, a| {
                    let again = a[0].clone();
                    a.push(again);
                },
                "appears 2 time(s)",
            ),
            (
                |_, a| {
                    let second = with_id(a[1].clone(), &rid(0xa6b, 3));
                    a.push(second);
                },
                "2 completed eval rows tagged epoch-score-val",
            ),
            (
                |_, a| a[0]["status"] = json!("running"),
                "is not a completed ft row",
            ),
            (
                |_, a| a[1]["status"] = json!("failed"),
                "(a) missing row: no completed eval row tagged epoch-score-val",
            ),
            (
                |_, a| a[1]["metrics"][PERM_KNOWLEDGE.name]["state"] = json!("not_run"),
                "permutation_consistency.family.knowledge.multiple_choice is not_run",
            ),
            (
                |_, a| {
                    a[1]["metrics"]
                        .as_object_mut()
                        .unwrap()
                        .remove(ID_ABSTAIN_COMMONSENSE.name);
                },
                "no metrics.ood_abstain.in_distribution.family.commonsense.multiple_choice",
            ),
            (
                |_, a| a[1]["metrics"][PERM_COMMONSENSE.name]["value"] = json!(0.5),
                "is not its own n/n_total",
            ),
            (
                |_, a| a[1]["metrics"]["ft_run_row_id"]["value"] = json!(rid(0xa6b, 9)),
                "(a) missing row: no completed eval row tagged epoch-score-val",
            ),
            // J6(g)'s rows are read from its own ledger only (R2).
            (
                |f, a| f.append(a),
                "(a) row 00000a6b-0000-4000-8000-000000000001 appears 0 time(s)",
            ),
            // An envelope seed missing: there is no envelope, so no room is decided.
            (
                |f, _| f.retain(|r| r["protocol"]["seed"] != 2),
                "room not decided",
            ),
        ];
        for (edit, phrase) in cases {
            let o = j6g_edited(|f, a, _| edit(f, a));
            refused_with(&o, phrase);
            assert_ne!(o.word, "quiet");
        }
    }

    /// The eval row is one comparable epoch-score-val row (arm.identity.eval_row): F's val shard
    /// and suites. The eval row's mirrored option_permutation_seed (a502670 copies the ported
    /// pieces into score rows, tools/real_ft_run.py:2605) is not a difference.
    #[test]
    fn j6g_eval_row_is_one_comparable_epoch_score_val_row() {
        let o = j6g_edited(|_, a, _| a[1]["recipe"]["val_shard_hash"] = json!("w"));
        refused_with(&o, "recipe.val_shard_hash");
        let o = j6g_edited(|_, a, _| a[1]["recipe"]["ood"]["cases_per_category"] = json!(30));
        refused_with(&o, "recipe.ood");
        let o = j6g_edited(|_, a, _| a[1]["recipe"]["needle"]["cases"] = json!(299));
        refused_with(&o, "recipe.needle");
        let o = j6g_edited(|_, a, _| {
            a[1]["recipe"][OPTION_PERMUTATION_KEY] = json!(OPTION_PERMUTATION_SEED);
        });
        assert_eq!(o.word, "wins", "{}", o.json);
        // No needle-control or letter-control row is required.
        assert!(!ARM_J6G.reads_control() && !ARM_J6G.reads_margin());
        let o = j6g_edited(|f, _, _| {
            f.retain(|r| {
                r["recipe"]["tag"] != json!(CONTROL_TAG) && r["metrics"].get(MARGIN_KEY).is_none()
            });
        });
        assert_eq!(o.word, "wins", "{}", o.json);
    }

    /// The file binds: a structured field or a stated identity that disagrees with the checker
    /// refuses before any ledger is read (R1: an amendment that admits more keys refuses here).
    #[test]
    fn a_preregistration_that_disagrees_with_the_j6g_checker_refuses() {
        type PreregEdit = fn(&mut Value);
        let cases: [(PreregEdit, &str); 11] = [
            (
                |p| p["outcomes"]["words"] = json!(["wins", "quiet"]),
                "outcomes.words are",
            ),
            (
                |p| {
                    p["arm"]["targets"].as_array_mut().unwrap().pop();
                },
                "arm.targets are",
            ),
            (
                |p| p["arm"]["targets"][3]["direction"] = json!("higher"),
                "arm.targets are",
            ),
            (
                |p| p["arm"]["guards"][4]["form"] = json!("count"),
                "arm.guards are",
            ),
            (
                |p| {
                    p["arm"]["guards"].as_array_mut().unwrap().swap(0, 1);
                },
                "arm.guards are",
            ),
            (
                |p| p["arm"]["option_permutation"]["seed"] = json!(20260920),
                "arm.option_permutation.seed is 20260920",
            ),
            (
                |p| {
                    p["arm"]["option_permutation"]
                        .as_object_mut()
                        .unwrap()
                        .remove("seed");
                },
                "arm.option_permutation.seed is absent",
            ),
            (
                |p| {
                    p["arm"]["identity"]["recipe"] = json!(
                        "Equal to F seed 0's ft recipe (973cd4e3), including shard_hash, batches \
                         (9683) and width (7936), except exactly two added keys: \
                         option_permutation_seed = 20260919 and tokenizer_json. Any other \
                         difference refuses."
                    )
                },
                "arm.identity.recipe does not say",
            ),
            (
                |p| {
                    p["arm"]["identity"]["ft_row"] = json!(
                        "completed; protocol.seed 0; recipe.tag 'epoch'; code_commit \
                         0264732; quick true"
                    )
                },
                "arm.identity.ft_row does not say",
            ),
            (
                |p| p["arm"]["identity"]["eval_row"] = json!("tag epoch-score-val"),
                "arm.identity.eval_row does not say",
            ),
            (
                |p| {
                    p["arm"].as_object_mut().unwrap().remove("identity");
                },
                "has no arm.identity object",
            ),
        ];
        for (edit, phrase) in cases {
            let o = j6g_edited(|_, _, p| edit(p));
            refused_with(&o, phrase);
            refused_with(&o, "(a) the pre-registration");
            assert_eq!(o.json["inputs"].as_array().unwrap().len(), 1, "{}", o.json);
        }
    }

    /// F seed 0's own committed rows (973cd4e3, f4feac15), presented as a J6(g) arm, refuse on
    /// identity: F trained without option permutation and is not quick. The envelope is F seed
    /// 0's real rows plus seeds 1 and 2 cloned from them (new ids, seed changed), because the
    /// committed ledger holds seed 0 only; the reference row stays the real 973cd4e3.
    #[test]
    fn f_seed_0s_own_rows_presented_as_j6g_refuse_on_identity() {
        let rows = real_rows(F_LEDGER_V4);
        let ft = rows
            .iter()
            .find(|r| r["row_id"] == F_SEED0_FT)
            .unwrap()
            .clone();
        let ev = rows
            .iter()
            .find(|r| {
                r["recipe"]["tag"] == EVAL_TAG
                    && r["metrics"]["ft_run_row_id"]["value"] == F_SEED0_FT
            })
            .unwrap()
            .clone();
        assert!(ev["row_id"].as_str().unwrap().starts_with("f4feac15"));
        let mut env = vec![ft.clone(), ev.clone()];
        let mut f_ft = vec![(0, F_SEED0_FT.to_string())];
        for s in 1..3u64 {
            let (ft_id, ev_id) = (rid(0xf5, s), rid(0xf6, s));
            let mut a = with_id(ft.clone(), &ft_id);
            a["protocol"]["seed"] = json!(s);
            let mut b = with_id(ev.clone(), &ev_id);
            b["protocol"]["seed"] = json!(s);
            b["metrics"]["ft_run_row_id"]["value"] = json!(ft_id);
            env.push(a);
            env.push(b);
            f_ft.push((s as i64, ft_id));
        }
        let (f, arm) = (temp_ledger(&env), temp_ledger(&[ft, ev]));
        let o = outcome(Cmd::J6g {
            preregistration: repo(J6G_PREREG),
            f_ledger: f.0.clone(),
            ft_rows: f_ft,
            arm_ledger: arm.0.clone(),
            j6g_ft_row: (0, F_SEED0_FT.to_string()),
            out: PathBuf::from("/unused"),
        });
        refused_with(
            &o,
            "(a) arm j6g ft row 973cd4e3-e0d2-4ff8-8588-b761cb842b75: ",
        );
        refused_with(
            &o,
            "recipe.option_permutation_seed is absent, not the pre-registered 20260919",
        );
        refused_with(&o, "quick is false");
        let reason = o.json["refused"].as_str().unwrap();
        // Only those two: F's own data, commit, protocol and every other recipe key agree.
        assert_eq!(reason.matches("; ").count(), 1, "{reason}");
        // Room was decided from the (real seed-0) envelope before the arm was read.
        assert_eq!(o.json["detail"]["room"].as_array().unwrap().len(), 4);
        assert_eq!(o.json["detail"]["cannot_clear"], json!([]));
    }

    /// The checker's constants are the pre-registration's text, and its literals are what F seed
    /// 0's committed rows record.
    #[test]
    fn the_j6g_rule_is_the_preregistrations_own_text() {
        let p = prereg_j6g();
        j6g_agrees(p.as_object().unwrap()).unwrap();
        assert_eq!(J6G_F_SEED0_FT, F_SEED0_FT);
        assert_eq!(J6G_F_DATA_SNAPSHOT_HASH, F_DATA_SNAPSHOT_HASH);
        assert_eq!(J6G_F_VAL_SHARD_HASH, F_VAL_SHARD_HASH);
        assert_eq!(
            p["arm"]["option_permutation"]["seed"],
            OPTION_PERMUTATION_SEED
        );
        assert_eq!(
            p["arm"]["ledger"],
            "/home/ubuntu/ledger/gh200-j6g-v4-2026-10-02.jsonl"
        );
        assert!(
            p["arm"]["declared_data_delta"]
                .as_str()
                .unwrap()
                .starts_with("None.")
        );
        let refused = p["outcomes"]["refused"].as_str().unwrap();
        assert!(refused.contains("(a) A required row is missing, doubled or not completed"));
        assert!(refused.contains("(b) A target has no room under R9_room (cannot_clear)"));
        assert_eq!((J6G_LABELS.unreadable, J6G_LABELS.no_room), ("(a)", "(b)"));
        assert!(
            p["envelope"]
                .as_str()
                .unwrap()
                .contains("F seeds 0, 1 and 2 only, never seeds 3-4")
        );
        let r1 = p["readings"]["R1_recipe_keys"].as_str().unwrap();
        assert!(r1.contains("recipe.batches and recipe.width are compared to F's (9683, 7936)"));
        assert!(
            p["readings"]["R2_ledger"]
                .as_str()
                .unwrap()
                .contains("refuses")
        );
        // The literals against F seed 0's committed rows (973cd4e3, f4feac15).
        let ledger = Inputs::default().read(&repo(F_LEDGER_V4)).unwrap();
        let rows = seed_rows(&ledger, 0, F_SEED0_FT, false, false).unwrap();
        assert_eq!(rows.ft.str_at(&["code_commit"]), Some(J6G_CODE_COMMIT));
        assert_eq!(
            rows.ft.str_at(&["protocol", "data_snapshot_hash"]),
            Some(J6G_F_DATA_SNAPSHOT_HASH)
        );
        assert_eq!(
            rows.ft.get(&["recipe", "batches"]).and_then(Value::as_i64),
            Some(J6G_F_BATCHES)
        );
        assert_eq!(
            rows.ft.get(&["recipe", "width"]).and_then(Value::as_i64),
            Some(J6G_F_WIDTH)
        );
        assert!(rows.ft.get(&["recipe", OPTION_PERMUTATION_KEY]).is_none());
        assert_eq!(
            rows.eval.str_at(&["recipe", "val_shard_hash"]),
            Some(J6G_F_VAL_SHARD_HASH)
        );
        // Every target and guard reads on f4feac15 with the value the file cites (f_seed0).
        for list in ["targets", "guards"] {
            for m in p["arm"][list].as_array().unwrap() {
                let name = m["name"].as_str().unwrap();
                let metric = ARM_J6G
                    .targets
                    .iter()
                    .copied()
                    .chain(ARM_J6G.must_not_lose())
                    .find(|x| x.name == name)
                    .unwrap();
                let shown = match rows.read(metric).unwrap() {
                    Val::Exact(r) => format!("{}/{}", r.num, r.den),
                    Val::Float(x) => x.to_string(),
                };
                assert_eq!(shown, m["f_seed0"].as_str().unwrap(), "{name}");
            }
        }
    }
}
