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
//!   `fires`, `quiet`. Its identity checks name no F row, hash or commit, so it reads any
//!   run's ledger. With `--preregistration` (v5's, `seeds.seeds_3_4`) it reads that file's
//!   threshold, which must be this rule's 0.30, adds its seeds' identity (`one_configuration`:
//!   completed, quick false, one recipe and data snapshot) and records the file's words and
//!   sha256; without it F's decision is byte for byte what it was.
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
//! * `v5-pause` — (vii) R9 (`campaign/v5-preregistered.json` `readings.R9_pause_after_seed_0`):
//!   after v5 seed 0's epoch-score-val row, do seeds 1-2 start? `continue` iff `val_top1.span`
//!   and the 8K worst bucket meet the count forms R9 writes (`5*n >= 4*n_total`,
//!   `2*n >= n_total`), else `pause`. A spending rule, not a gate.
//! * `v5-noulw` — (viii) the noul-weight arm (`arm_noul_weight`) against v5 seeds 0-2. With
//!   `--room`, its launch condition from v5's rows alone: `room` iff some target is held on
//!   fewer than all of v5's seeds, else `no_room`. Without it, the arm's three seeds read under
//!   `seed_holds` targets (a seed holds iff `2*n >= n_total`, the F2 bar of
//!   `campaign/v4-noul-v3b-preregistered.json`), worst-seed strict-envelope guards and F3's
//!   absolute guard. Words: `wins`, `quiet`, `refused`.
//! * `tierb` — (ix) the Tier-B outcome rule (`recipe.tierb_outcome_rule` of the v5
//!   pre-registration, as Fable amended it on 2026-10-03, before any outcome row existed:
//!   AUDIT/tierb-outcome-2026-10-03/fable-tierb-outcome-ruling.md), which decides v5's
//!   conditionals C2a (`--candidate nomask`) and C2b (`--candidate fused`). One candidate run's
//!   training-run score row and fp32 all-gates re-score against phase-3 seeds 0-2's envelope,
//!   its linear control's CI against eeda5db4's, and its gate flags against 58fd1532's. Words:
//!   `pass`, `fail`, `refused`. Decided before the DRAFT is renamed, so unlike the other v5
//!   rules it reads the DRAFT too and records which it read (`draft`).
//!
//! The v5 rules read their thresholds, seed sets, added recipe keys, targets and guards from the
//! pre-registration at run time; a file whose words disagree with what this code applies, or
//! that still carries the DRAFT's `draft` key, refuses before any ledger is read.
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

use clap::{Parser, Subcommand, ValueEnum};
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
        /// F's ledger (or v5's, with --preregistration).
        #[arg(long)]
        f_ledger: PathBuf,
        /// `SEED=FT_ROW_ID`, once for each of seeds 0, 1 and 2 (ids from F's logs).
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
        /// campaign/v5-preregistered.json, for v5's seeds 3-4: its seeds.seeds_3_4 threshold
        /// must be this rule's, and its seeds.v5 identity (quick false, one configuration) is
        /// applied. Omitted, the rule is F's, unchanged.
        #[arg(long)]
        preregistration: Option<PathBuf>,
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
    /// (vii) R9: after v5 seed 0's epoch-score-val row, do seeds 1-2 start? Prints `continue`,
    /// `pause` (a spending rule, not a gate) or `refused`.
    V5Pause {
        /// campaign/v5-preregistered.json as committed; read at run time, its sha256 recorded.
        /// The DRAFT (top-level `draft`) refuses.
        #[arg(long)]
        preregistration: PathBuf,
        /// v5's ledger (/home/ubuntu/ledger/gh200-v5-<date>.jsonl).
        #[arg(long)]
        ledger: PathBuf,
        /// `SEED=FT_ROW_ID` of the seed R9 names (0).
        #[arg(long = "ft-row", value_parser = parse_ft_row)]
        ft_row: (i64, String),
        #[arg(long)]
        out: PathBuf,
    },
    /// (viii) The noul-weight arm against v5 seeds 0-2. With --room: prints `room` or `no_room`
    /// from v5's rows alone (the arm's launch condition). Without: `wins`, `quiet` or `refused`.
    V5Noulw {
        /// campaign/v5-preregistered.json as committed; read at run time, its sha256 recorded.
        #[arg(long)]
        preregistration: PathBuf,
        /// campaign/v4-noul-v3b-preregistered.json: the F2 bar seed_holds cites and the F3 bound
        /// the absolute guard cites; read at run time, its sha256 recorded.
        #[arg(long)]
        noul_preregistration: PathBuf,
        /// v5's ledger: the envelope's ft and epoch-score-val rows.
        #[arg(long)]
        v5_ledger: PathBuf,
        /// `SEED=FT_ROW_ID` for v5's seeds 0, 1 and 2 exactly, in order (never seeds 3-4).
        #[arg(long = "ft-row", required = true, value_parser = parse_ft_row)]
        ft_rows: Vec<(i64, String)>,
        /// Decide the launch condition only; no arm row is read.
        #[arg(long)]
        room: bool,
        /// The arm's ledger (/home/ubuntu/ledger/gh200-v5-noulw-<date>.jsonl).
        #[arg(long, required_unless_present = "room", conflicts_with = "room")]
        arm_ledger: Option<PathBuf>,
        /// `SEED=FT_ROW_ID` for the arm's seeds 0, 1 and 2 exactly, in order.
        #[arg(
            long = "arm-ft-row",
            value_parser = parse_ft_row,
            required_unless_present = "room",
            conflicts_with = "room"
        )]
        arm_ft_rows: Vec<(i64, String)>,
        #[arg(long)]
        out: PathBuf,
    },
    /// (ix) The Tier-B outcome rule for one candidate (v5's C2a / C2b). Prints `pass`, `fail` or
    /// `refused`.
    Tierb {
        /// Which candidate's outcome run: `nomask` (C2a) or `fused` (C2b).
        #[arg(long, value_enum)]
        candidate: Candidate,
        /// The candidate ledger: its ft row, both score rows and its linear-control row.
        #[arg(long)]
        ledger: PathBuf,
        /// ledger/gh200-seed0-weights-2026-09-30.jsonl: the envelope rows, 7f2c11db and eeda5db4.
        #[arg(long)]
        phase3_ledger: PathBuf,
        /// ledger/gh200-allgates-2026-09-30.jsonl: 58fd1532.
        #[arg(long)]
        allgates_ledger: PathBuf,
        /// The v5 pre-registration, the DRAFT or the renamed file; read at run time, its sha256
        /// and whether it is the DRAFT recorded.
        #[arg(long)]
        preregistration: PathBuf,
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

/// Whether an eval row was scored from ft row `ft` of seed `seed`: the link every score row
/// records (`metrics.ft_run_row_id`).
fn scored_from(r: &Row, seed: i64, ft: &str) -> bool {
    r.seed() == Some(seed) && r.str_at(&["metrics", "ft_run_row_id", "value"]) == Some(ft)
}

fn eval_row_of<'a>(ledger: &'a Ledger, seed: i64, ft: &str, tag: &str) -> Result<&'a Row> {
    ledger.one(
        "eval",
        tag,
        &format!("seed {seed} scored from ft row {ft}"),
        |r| scored_from(r, seed, ft),
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
    seeds_in(&ledger, ft_rows)
}

/// Each seed's ft row and its one completed epoch-score-val row in `ledger`, with the 8K worst
/// bucket and the two OOD counts (ii) and (iii) read. Nothing here names a run: the ft ids
/// given pin the rows, so it reads F's ledger or v5's alike.
fn seeds_in(ledger: &Ledger, ft_rows: &[(i64, String)]) -> Result<Vec<FSeed>> {
    let mut out = Vec::new();
    for (seed, ft) in ft_rows {
        let ft_row = ft_row(ledger, *seed, ft)?;
        let eval = eval_row_of(ledger, *seed, ft, EVAL_TAG)?;
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

/// (ii). Without a pre-registration it is F's rule exactly as it was. With one (v5's), the file's
/// `seeds.seeds_3_4` threshold must be this rule's 0.30, its `seeds.v5` seed set and identity
/// (quick false, one recipe and data snapshot: `one_configuration`) are applied, and the decision
/// records the file's words and sha256 instead of F's.
fn rule_seeds34(
    inputs: &mut Inputs,
    f_ledger: &Path,
    ft_rows: &[(i64, String)],
    preregistration: Option<&Path>,
) -> Result<(String, Value)> {
    let Some(path) = preregistration else {
        return Ok(seeds34_decision(&f_seeds(inputs, f_ledger, ft_rows)?, None));
    };
    let (p, sha256) = inputs.read_preregistration(path)?;
    let reading = seeds34_reading(&p)?;
    let seeds: Vec<i64> = ft_rows.iter().map(|(s, _)| *s).collect();
    ensure!(
        seeds == reading.seeds,
        "seeds.v5 names seeds {:?}, given in that order; got {seeds:?}",
        reading.seeds
    );
    let ledger = inputs.read(f_ledger)?;
    one_configuration(&ledger, ft_rows)?;
    let seeds = seeds_in(&ledger, ft_rows)?;
    Ok(seeds34_decision(&seeds, Some((&reading, sha256.as_str()))))
}

/// The spread rule on the seeds' 8K worst buckets, and its JSON: F's words, or the
/// pre-registration's when one was read.
fn seeds34_decision(seeds: &[FSeed], v5: Option<(&Seeds34Reading, &str)>) -> (String, Value) {
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
    let mut body = json!({
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
    if let Some((reading, sha256)) = v5 {
        body["rule"] = json!(format!(
            "seeds 3 and 4 run iff the three 8K needle worst buckets spread by more than {} \
             (seeds.seeds_3_4, read from the pre-registration: {:?})",
            SPREAD_MAX.text, reading.text
        ));
        body["then"] = json!(if fires {
            reading.fires.as_str()
        } else {
            "no seeds 3-4 (seeds.seeds_3_4)"
        });
        body["preregistration_sha256"] = json!(sha256);
        body["not_checked"] = json!(NOT_CHECKED_SUITE);
    }
    (if fires { "fires" } else { "quiet" }.into(), body)
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
    control_row_of(
        ledger,
        eval,
        "letter-control",
        &format!("carrying {MARGIN_KEY}"),
        |r| r.get(&["metrics", MARGIN_KEY]).is_some(),
    )
}

/// The one completed control row of `kind` that `which` accepts among those that name `eval` in
/// `metrics.scored_eval_row_id`; its `recipe.eval_row_id` must name the same row. `what`
/// describes `which` in the refusal. Shared by the letter control (`letter_control_row`) and
/// tierb's linear control.
fn control_row_of<'a>(
    ledger: &'a Ledger,
    eval: &Row,
    kind: &str,
    what: &str,
    which: impl Fn(&Row) -> bool,
) -> Result<&'a Row> {
    let found: Vec<&Row> = ledger
        .rows
        .iter()
        .filter(|r| {
            r.completed()
                && r.str_at(&["run_kind"]) == Some("eval")
                && r.str_at(&["metrics", "scored_eval_row_id", "value"]) == Some(eval.id())
                && which(r)
        })
        .collect();
    let row = match found.len() {
        1 => found[0],
        0 => {
            return Err(format!(
                "missing row: no completed {kind} row {what} for eval row {} in {}",
                eval.id(),
                ledger.shown
            ));
        }
        n => {
            return Err(format!(
                "{n} completed {kind} rows {what} for eval row {} in {} ({}); which one decides \
                 is not this tool's call",
                eval.id(),
                ledger.shown,
                found.iter().map(|r| r.id()).collect::<Vec<_>>().join(", ")
            ));
        }
    };
    ensure!(
        row.str_at(&["recipe", "eval_row_id"]) == Some(eval.id()),
        "{kind} row {}: recipe.eval_row_id {:?} is not its scored eval row {}",
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

/// The eval-row recipe keys `comparable` holds equal to the envelope's seed 0: the val set and
/// the needle and OOD suites.
const COMPARABLE_EVAL_KEYS: [&str; 3] = ["val_shard_hash", "needle", "ood"];

/// Every row an arm or envelope seed is read from is comparable with the envelope's seed 0:
/// the same val set and needle / OOD suites, the same length-control suite, and the same
/// letter control (its recipe but for its own eval row id).
fn comparable(rows: &SeedRows, reference: &SeedRows, what: &str) -> Result<()> {
    same_recipe_keys(rows.eval, reference.eval, &COMPARABLE_EVAL_KEYS, what)?;
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
    let (min, max) = min_max(m.name, &values)?;
    Ok((values, min, max))
}

/// The minimum and maximum of an envelope's values of metric `name`, compared exactly
/// (`cmp_val`). Shared by `envelope_range` (envelope seeds read through `SeedRows`) and tierb
/// (envelope rows pinned by eval row id).
fn min_max(name: &str, values: &[Val]) -> Result<(Val, Val)> {
    let first = *values
        .first()
        .ok_or_else(|| format!("{name}: an empty envelope has no range"))?;
    let (mut min, mut max) = (first, first);
    for &v in &values[1..] {
        if cmp_val(v, min)? == std::cmp::Ordering::Less {
            min = v;
        }
        if cmp_val(v, max)? == std::cmp::Ordering::Greater {
            max = v;
        }
    }
    Ok((min, max))
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
        let got = listed_in(arm, "arm", list)?;
        let want: Vec<(&str, &str, &str)> = want.into_iter().map(listed).collect();
        ensure!(
            got == want,
            "the pre-registration's arm.{list} are {got:?}, not this checker's {want:?}"
        );
    }
    Ok(arm)
}

/// A pre-registration's metric list `{label}.{list}`, each entry as (name, direction, form). A
/// form names itself first and may go on: "needle_worst_bucket (campaign/f-successor-...)",
/// "count, plus absolute_guard", "count (pooled)". Shared by j6a, j6g and v5-noulw.
fn listed_in<'p>(
    arm: &'p Map<String, Value>,
    label: &str,
    list: &str,
) -> Result<Vec<(&'p str, &'p str, &'p str)>> {
    Ok(arm
        .get(list)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("the pre-registration has no {label}.{list} list"))?
        .iter()
        .map(|m| {
            let form = m["form"].as_str().unwrap_or("");
            let form = form.split(' ').next().unwrap_or("").trim_end_matches(',');
            (
                m["name"].as_str().unwrap_or(""),
                m["direction"].as_str().unwrap_or(""),
                form,
            )
        })
        .collect())
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

// --- (vii), (viii): v5, campaign/v5-preregistered.json -----------------------------------------
//
// R9's spending rule after v5 seed 0 (`v5-pause`), the noul-weight arm's launch condition
// (`v5-noulw --room`) and its reading (`v5-noulw`), and v5's use of (ii) (`seeds34
// --preregistration`). Each reads the binding file at run time. What it reads out of the file:
// thresholds (the count forms R9, seed_holds and the absolute guard write, cross-checked against
// the F2 / F3 texts of campaign/v4-noul-v3b-preregistered.json they cite), seed sets, the arm's
// added recipe keys and their values, and its target and guard lists. What it checks the file
// says: how each metric is read and which direction is better, and the rule's words (`says`).
// Either kind of disagreement refuses before a ledger is read. The arithmetic is the
// successor's: counts on the integers (`Frac`, `Rat`), the envelope range (`envelope_range`),
// the strict guard (`loses`) and the row look-ups (`one_configuration`, `seed_rows`,
// `comparable`).

/// The file as it binds; the text a decision applied is the one whose sha256 it records.
const PREREG_V5: &str = "campaign/v5-preregistered.json (the DRAFT renamed on main, binds_iff; \
                         read at run time, its sha256 recorded)";
/// arm_noul_weight.outcomes.words.
const V5NW_WORDS: [&str; 3] = ["wins", "quiet", "refused"];
/// arm_noul_weight.launch_condition's two words.
const ROOM_WORDS: [&str; 2] = ["room", "no_room"];
/// readings.R9_pause_after_seed_0's two words.
const PAUSE_WORDS: [&str; 2] = ["continue", "pause"];
/// R9 names the 8K gate's worst depth bucket in words, not by its listed name.
const R9_NEEDLE_PHRASE: &str = "8K needle worst bucket";
/// identity.ft_rows' pairing check follows the metric it names with these words (Fable's
/// seed-order ruling, section 2).
const PAIRED_PHRASE: &str = " equal to v5's ft row of the same seed";
/// Pooled in-distribution abstention: every val choice row, CLINC's included (a v5 guard).
const ID_ABSTAIN_POOLED: Metric = count_metric("ood_abstain.in_distribution", Dir::Lower);
/// Every metric a v5 rule reads off an epoch-score-val row, under the name the pre-registration
/// lists it by. A listed name outside this set refuses: the checker reads nothing it was not
/// written to read.
const V5_READABLE: [Metric; 10] = [
    VAL_CHOICE,
    VAL_SPAN,
    PERM_DC,
    ECE_DC,
    NEEDLE_8K_WORST,
    PROSE,
    SCRAMBLED,
    UNSEEN,
    ID_ABSTAIN_DC,
    ID_ABSTAIN_POOLED,
];
/// What no v5 rule can check from a row (GAP-V5-RULES-REBUILT-SUITE-NOT-ON-THE-ROW-2026-10-02).
const NOT_CHECKED_SUITE: &str = "that the eval rows were scored on the rebuilt 8K needle suite: \
     recipe.needle carries no suite digest and needle_suite_tokens records the median, so no row \
     field tells the rebuilt suite from F's; recipe.needle is recorded for the reader \
     (GAP-V5-RULES-REBUILT-SUITE-NOT-ON-THE-ROW-2026-10-02)";

/// The string at `path` in a pre-registration.
fn prereg_text<'p>(p: &'p Map<String, Value>, path: &[&str]) -> Result<&'p str> {
    let mut cur = p.get(path[0]);
    for key in &path[1..] {
        cur = cur.and_then(|v| v.get(key));
    }
    cur.and_then(Value::as_str)
        .ok_or_else(|| format!("the pre-registration has no string {}", path.join(".")))
}

/// The pre-registration's text at `path` says `phrase`, which is the rule this checker applies.
/// An amended text changes the words and refuses until the checker follows it.
fn says(p: &Map<String, Value>, path: &[&str], phrase: &str) -> Result<()> {
    ensure!(
        prereg_text(p, path)?.contains(phrase),
        "the pre-registration's {} does not say {phrase:?}, which is the rule this checker \
         applies",
        path.join(".")
    );
    Ok(())
}

/// The DRAFT's own first words are that no checker, waiter or lane may read it as a rule; it
/// binds only once renamed to campaign/v5-preregistered.json on main (binds_iff). A file that
/// still carries its `draft` key is the DRAFT.
fn not_a_draft(p: &Map<String, Value>) -> Result<()> {
    ensure!(
        !p.contains_key("draft"),
        "the pre-registration still carries the DRAFT's top-level \"draft\" key, which says no \
         checker may read it as a rule; it binds only once renamed to \
         campaign/v5-preregistered.json on main without that key (binds_iff)"
    );
    Ok(())
}

fn count_word(n: usize) -> Option<&'static str> {
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ]
    .get(n)
    .copied()
}

/// `[0, 1, 2]` as the pre-registration writes it: "0, 1 and 2".
fn and_list(seeds: &[i64]) -> String {
    match seeds {
        [] => String::new(),
        [one] => one.to_string(),
        [head @ .., last] => format!(
            "{} and {last}",
            head.iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// A cursor over pre-registration prose, for the few forms a v5 rule reads out of its text.
struct Words<'a>(&'a str);

impl<'a> Words<'a> {
    /// The text after the one occurrence of `anchor`; none, or more than one, refuses.
    fn after(text: &'a str, anchor: &str, what: &str) -> Result<Words<'a>> {
        let hits = text.matches(anchor).count();
        ensure!(
            hits == 1,
            "{what}: {anchor:?} occurs {hits} time(s) in {text:?}, not once"
        );
        let at = text.find(anchor).map_or(text.len(), |i| i + anchor.len());
        Ok(Words(&text[at..]))
    }
    fn shown(&self) -> String {
        self.0.chars().take(48).collect()
    }
    /// `token` next, after any spaces; consumed if so.
    fn eat(&mut self, token: &str) -> bool {
        match self.0.trim_start().strip_prefix(token) {
            Some(rest) => {
                self.0 = rest;
                true
            }
            None => false,
        }
    }
    fn int(&mut self) -> Option<u64> {
        let s = self.0.trim_start();
        let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        let v = s[..end].parse().ok()?;
        self.0 = &s[end..];
        Some(v)
    }
    /// A decimal as written ("0.30", "4.0", "12"): `num / den` with den a power of ten, and the
    /// text. A trailing full stop is not a decimal point.
    fn decimal(&mut self) -> Option<(u64, u64, &'a str)> {
        let s = self.0.trim_start();
        let int_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        if int_end == 0 {
            return None;
        }
        let mut end = int_end;
        if s[end..].starts_with('.') && s[end + 1..].starts_with(|c: char| c.is_ascii_digit()) {
            let frac = &s[end + 1..];
            end += 1 + frac
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(frac.len());
        }
        let shown = &s[..end];
        let num: u64 = shown.replace('.', "").parse().ok()?;
        let places = u32::try_from(end - int_end).ok()?.saturating_sub(1);
        let den = 10u64.checked_pow(places)?;
        self.0 = &s[end..];
        Some((num, den, shown))
    }
    /// The text up to `end`, consumed with it.
    fn until(&mut self, end: &str) -> Option<&'a str> {
        let at = self.0.find(end)?;
        let head = &self.0[..at];
        self.0 = &self.0[at + end.len()..];
        Some(head)
    }
    /// A key as a pre-registration writes one: `[A-Za-z0-9_.-]+`.
    fn name(&mut self) -> Option<&'a str> {
        let s = self.0.trim_start();
        let end = s
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')))
            .unwrap_or(s.len());
        // A sentence's full stop is not part of the name before it.
        let end = s[..end].trim_end_matches('.').len();
        if end == 0 {
            return None;
        }
        self.0 = &s[end..];
        Some(&s[..end])
    }
    /// `K/N`, as a bar or an example is printed.
    fn ratio(&mut self) -> Option<(u64, u64)> {
        let k = self.int()?;
        if !self.eat("/") {
            return None;
        }
        Some((k, self.int()?))
    }
}

/// `a*n OP b*n_total`, as a pre-registration writes a count form (spacing free, a factor of 1
/// unwritten): a count `n / n_total` meets it iff `a*n OP b*n_total`, on the integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CountForm {
    a: u64,
    at_least: bool,
    b: u64,
}

impl CountForm {
    fn read(w: &mut Words, what: &str) -> Result<CountForm> {
        let shown = w.shown();
        let bad = || {
            format!(
                "{what}: {shown:?} is not a count form `a*n >= b*n_total` or `a*n <= b*n_total`"
            )
        };
        let factor = |w: &mut Words| -> Result<u64> {
            let save = w.0;
            match w.int() {
                None => Ok(1),
                Some(k) if w.eat("*") => Ok(k),
                Some(_) => {
                    w.0 = save;
                    Err(bad())
                }
            }
        };
        let a = factor(&mut *w)?;
        ensure!(w.eat("n"), "{}", bad());
        let at_least = if w.eat(">=") {
            true
        } else if w.eat("<=") {
            false
        } else {
            return Err(bad());
        };
        let b = factor(&mut *w)?;
        ensure!(
            w.eat("n_total")
                && !w
                    .0
                    .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_'),
            "{}",
            bad()
        );
        ensure!(
            a > 0 && b > 0,
            "{what}: a factor of 0 in {shown:?} is not a bound"
        );
        Ok(CountForm { a, at_least, b })
    }
    fn meets(self, f: Frac) -> bool {
        let (l, r) = (
            u128::from(self.a) * u128::from(f.k),
            u128::from(self.b) * u128::from(f.n),
        );
        if self.at_least { l >= r } else { l <= r }
    }
    /// The form is exactly the bound `n / n_total OP num / den`: `b * den == a * num`.
    fn is_bound(self, num: u64, den: u64) -> bool {
        u128::from(self.b) * u128::from(den) == u128::from(self.a) * u128::from(num)
    }
    fn text(self) -> String {
        format!(
            "{}*n {} {}*n_total",
            self.a,
            if self.at_least { ">=" } else { "<=" },
            self.b
        )
    }
}

/// `seeds.v5`'s seed list ("0, 1, 2; one ft row and one completed epoch-score-val eval row
/// each, quick false ..."), with the identity it states for each seed's rows.
fn v5_seeds(p: &Map<String, Value>) -> Result<Vec<i64>> {
    let path = ["seeds", "v5"];
    let text = prereg_text(p, &path)?;
    let head = text.split(';').next().unwrap_or("");
    let seeds: Vec<i64> = head
        .split(',')
        .map(|s| {
            s.trim().parse::<i64>().map_err(|_| {
                format!("the pre-registration's seeds.v5 {text:?} does not open with its seed list")
            })
        })
        .collect::<Result<_>>()?;
    let distinct: BTreeSet<i64> = seeds.iter().copied().collect();
    ensure!(
        !seeds.is_empty() && distinct.len() == seeds.len(),
        "the pre-registration's seeds.v5 names {seeds:?}: no seed, or one twice"
    );
    says(
        p,
        &path,
        &format!("one ft row and one completed {EVAL_TAG} eval row each, quick false"),
    )?;
    Ok(seeds)
}

/// The metric a v5 rule lists as `name`, read the way this checker reads it, in the direction
/// the list gives; a name it cannot read, or a direction it does not read it in, refuses.
fn v5_metric(name: &str, direction: &str, what: &str) -> Result<Metric> {
    let m = V5_READABLE
        .iter()
        .find(|m| m.name == name)
        .copied()
        .ok_or_else(|| {
            format!(
                "{what}: {name:?} is not a metric this checker reads off an {EVAL_TAG} row ({:?})",
                V5_READABLE.iter().map(|m| m.name).collect::<Vec<_>>()
            )
        })?;
    ensure!(
        dir_word(m.dir) == direction,
        "{what}: {name} is listed {direction:?}, but this checker reads it {}-better",
        dir_word(m.dir)
    );
    Ok(m)
}

// --- (ii) on v5's seeds -------------------------------------------------------------------------

/// What `seeds34 --preregistration` read from `seeds.seeds_3_4` and `seeds.v5`.
struct Seeds34Reading {
    text: String,
    /// The file's own words for what runs when the rule fires.
    fires: String,
    seeds: Vec<i64>,
}

/// `seeds.seeds_3_4` names this rule and its threshold, which must be (ii)'s 0.30: the spread
/// arithmetic is (ii)'s, so a different threshold refuses rather than run under the wrong one.
fn seeds34_reading(p: &Map<String, Value>) -> Result<Seeds34Reading> {
    not_a_draft(p)?;
    let path = ["seeds", "seeds_3_4"];
    let what = "seeds.seeds_3_4";
    let text = prereg_text(p, &path)?;
    let (num, den, shown) = Words::after(text, "spread by more than ", what)?
        .decimal()
        .ok_or_else(|| format!("{what}: no decimal after \"spread by more than\" in {text:?}"))?;
    ensure!(
        u128::from(num) * u128::from(SPREAD_MAX.den)
            == u128::from(SPREAD_MAX.num) * u128::from(den),
        "{what}: the spread threshold {shown} is not this rule's {}; (ii) applies only its own",
        SPREAD_MAX.text
    );
    says(p, &path, "checked by qd-post-f-rules seeds34")?;
    let mut w = Words::after(text, "seeds 3 and 4 run", what)?;
    let fires = format!(
        "seeds 3 and 4 run{}",
        w.until(";")
            .ok_or_else(|| format!("{what}: no ';' after \"seeds 3 and 4 run\""))?
    );
    Ok(Seeds34Reading {
        text: text.to_string(),
        fires,
        seeds: v5_seeds(p)?,
    })
}

// --- (vii): R9 ----------------------------------------------------------------------------------

/// readings.R9_pause_after_seed_0, as this checker applies it.
struct PauseRule {
    text: String,
    seed: i64,
    tag: String,
    span: CountForm,
    needle: CountForm,
    seeds: Vec<i64>,
}

fn pause_rule(p: &Map<String, Value>) -> Result<PauseRule> {
    not_a_draft(p)?;
    let path = ["readings", "R9_pause_after_seed_0"];
    let what = "readings.R9_pause_after_seed_0";
    let text = prereg_text(p, &path)?;
    let mut w = Words::after(text, "after v5 seed ", what)?;
    let seed = w
        .int()
        .and_then(|s| i64::try_from(s).ok())
        .ok_or_else(|| format!("{what}: no seed after \"after v5 seed\""))?;
    ensure!(
        w.eat("'s"),
        "{what}: \"after v5 seed {seed}\" is not followed by 's"
    );
    let tag = w
        .until(" row")
        .ok_or_else(|| format!("{what}: no row named after \"after v5 seed {seed}'s\""))?
        .trim()
        .to_string();
    ensure!(
        tag == EVAL_TAG,
        "{what}: R9 reads the {tag:?} row; this checker reads val_top1.span and the 8K gate off \
         the {EVAL_TAG} row"
    );
    says(
        p,
        &path,
        &format!("qd-post-f-rules v5-pause prints {} iff", PAUSE_WORDS[0]),
    )?;
    says(p, &path, &format!("otherwise {}", PAUSE_WORDS[1]))?;
    let span = CountForm::read(
        &mut Words::after(text, &format!("{} has ", VAL_SPAN.name), what)?,
        what,
    )?;
    let needle = CountForm::read(
        &mut Words::after(text, &format!("{R9_NEEDLE_PHRASE} has "), what)?,
        what,
    )?;
    ensure!(
        span.at_least && needle.at_least,
        "{what}: {} and {} are not both floors; R9 continues on counts at least their bounds",
        span.text(),
        needle.text()
    );
    let seeds = v5_seeds(p)?;
    ensure!(
        seeds.contains(&seed),
        "{what}: seed {seed} is not one of seeds.v5's {seeds:?}"
    );
    Ok(PauseRule {
        text: text.to_string(),
        seed,
        tag,
        span,
        needle,
        seeds,
    })
}

/// R9: after v5 seed 0's epoch-score-val row, `continue` iff val_top1.span and the 8K worst
/// bucket each meet R9's count form, else `pause`. A spending rule, not a gate.
fn rule_v5_pause(
    inputs: &mut Inputs,
    preregistration: &Path,
    ledger: &Path,
    ft: &(i64, String),
) -> Result<(String, Value)> {
    let (p, sha256) = inputs.read_preregistration(preregistration)?;
    let rule = pause_rule(&p)?;
    ensure!(
        ft.0 == rule.seed,
        "R9 reads v5 seed {}'s rows; --ft-row names seed {}",
        rule.seed,
        ft.0
    );
    let ledger = inputs.read(ledger)?;
    let (rows, recipe_hash, data) = one_configuration(&ledger, std::slice::from_ref(ft))?;
    let eval = eval_row_of(&ledger, ft.0, &ft.1, &rule.tag)?;
    let span = eval.count("metrics", VAL_SPAN.name)?;
    let worst = needle_8k(eval)?;
    let (span_ok, needle_ok) = (rule.span.meets(span), rule.needle.meets(worst.frac));
    let rest: Vec<i64> = rule
        .seeds
        .iter()
        .copied()
        .filter(|s| *s != rule.seed)
        .collect();
    let go = span_ok && needle_ok;
    let body = json!({
        "rule": rule.text,
        "kind": "a spending rule, not a gate (R9)",
        "preregistration_sha256": sha256,
        "seed": ft.0,
        "ft_row": ft.1,
        "eval_row": eval.id(),
        "code_commit": rows.first().and_then(|r| r.str_at(&["code_commit"])),
        "recipe_hash": recipe_hash,
        "data_snapshot_hash": data,
        "val_top1.span": {"count": span.json(), "form": rule.span.text(), "meets": span_ok},
        "needle_8k_worst": {"worst": worst.json(), "form": rule.needle.text(), "meets": needle_ok},
        "suite": {
            "recipe.needle": eval.get(&["recipe", "needle"]),
            "needle_suite_tokens.value": eval.get(&["metrics", "needle_suite_tokens", "value"]),
        },
        "not_checked": NOT_CHECKED_SUITE,
        "then": if go {
            format!("v5 seeds {} start", and_list(&rest))
        } else {
            format!(
                "v5 seeds {} wait until the human pins V5_CONTINUE (R9; a spending rule, not a gate)",
                and_list(&rest)
            )
        },
    });
    Ok((PAUSE_WORDS[usize::from(!go)].to_string(), body))
}

// --- (viii): the noul-weight arm ----------------------------------------------------------------

/// arm_noul_weight, and the F2 / F3 texts it cites, as this checker applies them.
struct NoulwRule {
    seeds: Vec<i64>,
    /// identity.ft_rows: the arm's ft rows' recipe.tag.
    ft_tag: String,
    /// identity.ft_rows: the keys the arm's recipe adds to v5's same-seed recipe, and their values.
    added: Vec<(String, Value)>,
    /// identity.ft_rows' pairing check: the ft-row metric (`corpus.plan_order_digest`) the arm's
    /// and v5's same-seed ft rows must both record, equal.
    paired: String,
    /// seed_holds: a seed holds a target iff its count meets this form...
    holds: CountForm,
    /// ...which is the F2 bar, over exactly this many cases.
    bar: Frac,
    /// comparison.room: a target has room iff v5 holds it on at most this many seeds.
    room_at_most: usize,
    targets: Vec<Metric>,
    guards: Vec<Metric>,
    /// comparison.absolute_guard: on every arm seed this metric meets this form (F3).
    absolute: (Metric, CountForm),
    /// The comparison texts applied, for the record.
    applied: Value,
}

/// `except exactly two added keys, noul_weight = 4.0 and noul_weight_scope =
/// 'code.defect_class'.`: each key with its value, a number or a quoted string.
fn added_keys(text: &str, what: &str) -> Result<Vec<(String, Value)>> {
    let mut w = Words::after(text, "except exactly ", what)?;
    let word = w
        .until(" added key")
        .ok_or_else(|| format!("{what}: no \"added key\" after \"except exactly\""))?
        .trim();
    let n = (0..10)
        .find(|&i| count_word(i) == Some(word))
        .ok_or_else(|| format!("{what}: {word:?} is not a count of added keys"))?;
    w.eat("s");
    ensure!(
        w.eat(",") || w.eat(":"),
        "{what}: the added keys are not listed after \"{word} added key(s)\""
    );
    let mut out: Vec<(String, Value)> = Vec::new();
    loop {
        let key = w
            .name()
            .ok_or_else(|| format!("{what}: no key at {:?}", w.shown()))?;
        ensure!(w.eat("="), "{what}: {key} is not followed by '='");
        let value = if w.eat("'") {
            json!(
                w.until("'")
                    .ok_or_else(|| format!("{what}: {key}'s quoted value is not closed"))?
            )
        } else {
            let at = w.shown();
            let (num, den, _) = w.decimal().ok_or_else(|| {
                format!("{what}: {key}'s value at {at:?} is not a number or a quoted string")
            })?;
            json!(num as f64 / den as f64)
        };
        ensure!(
            !out.iter().any(|(k, _)| k == key),
            "{what}: {key} is added twice"
        );
        out.push((key.to_string(), value));
        if !(w.eat("and") || w.eat(",")) {
            break;
        }
    }
    ensure!(
        out.len() == n,
        "{what}: \"{word} added key(s)\" lists {} ({:?})",
        out.len(),
        out.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
    );
    Ok(out)
}

/// The arm's block read and checked against what this checker applies; F2 and F3 read from
/// campaign/v4-noul-v3b-preregistered.json (`noul`) and checked against the forms that cite them.
fn noulw_rule(p: &Map<String, Value>, noul: &Map<String, Value>) -> Result<NoulwRule> {
    not_a_draft(p)?;
    let arm = p
        .get("arm_noul_weight")
        .and_then(Value::as_object)
        .ok_or("the pre-registration has no arm_noul_weight object")?;
    let a = |rest: &[&'static str]| -> Vec<&'static str> {
        std::iter::once("arm_noul_weight")
            .chain(rest.iter().copied())
            .collect()
    };
    let seeds = v5_seeds(p)?;
    let n = seeds.len();
    let n_word = count_word(n).ok_or_else(|| format!("seeds.v5 names {n} seeds"))?;
    let seed_list = and_list(&seeds);

    // Words and the launch condition.
    let words: Vec<&str> = arm
        .get("outcomes")
        .and_then(|o| o.get("words"))
        .and_then(Value::as_array)
        .map(|w| w.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    ensure!(
        words == V5NW_WORDS,
        "the pre-registration's arm_noul_weight.outcomes.words are {words:?}, not this checker's \
         {V5NW_WORDS:?}"
    );
    says(
        p,
        &a(&["outcomes", "wins"]),
        "every target with room clears and no guard (including the absolute guard and any \
         no-room target read as a guard) loses",
    )?;
    says(p, &a(&["outcomes", "refused"]), "or no target has room")?;
    says(
        p,
        &a(&["launch_condition"]),
        &format!(
            "qd-post-f-rules v5-noulw --room prints {} or {} from v5 seeds",
            ROOM_WORDS[0], ROOM_WORDS[1]
        ),
    )?;

    // The envelope and the rows.
    says(p, &a(&["envelope"]), &format!("v5 seeds {seed_list} only"))?;
    says(
        p,
        &a(&["arm_rows"]),
        "from the arm ledger only; a second eval row for an arm ft row refuses",
    )?;

    // Identity: the ft rows.
    let id_path = a(&["identity", "ft_rows"]);
    let id_text = prereg_text(p, &id_path)?;
    let id_seeds = seeds
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    for phrase in [
        "completed".to_string(),
        format!("protocol.seed {id_seeds} once each"),
        "quick false".to_string(),
        "code_commit equal to v5's".to_string(),
        "protocol.data_snapshot_hash and recipe.shard_hash equal to v5's".to_string(),
        "recipe equal to v5's ft recipe of the same seed".to_string(),
        "Any other difference refuses".to_string(),
    ] {
        says(p, &id_path, &phrase)?;
    }
    let ft_tag = Words::after(id_text, "recipe.tag '", "arm_noul_weight.identity.ft_rows")?
        .until("'")
        .ok_or("arm_noul_weight.identity.ft_rows: recipe.tag's quote is not closed")?
        .to_string();
    let added = added_keys(id_text, "arm_noul_weight.identity.ft_rows")?;
    let w_value = arm
        .get("w")
        .and_then(|w| w.get("value"))
        .and_then(Value::as_f64)
        .ok_or("the pre-registration has no number arm_noul_weight.w.value")?;
    ensure!(
        added
            .iter()
            .any(|(k, v)| k == "noul_weight" && v.as_f64() == Some(w_value)),
        "arm_noul_weight.identity.ft_rows adds {added:?}, which does not set noul_weight to \
         arm_noul_weight.w.value {w_value}"
    );
    // The pairing check (Fable's seed-order ruling, section 2): the ft-row metric named just
    // before PAIRED_PHRASE is equal on the arm's and v5's same-seed ft rows.
    let at = id_text.find(PAIRED_PHRASE).ok_or_else(|| {
        format!(
            "arm_noul_weight.identity.ft_rows does not say \"<metric>{PAIRED_PHRASE}\": the \
             pairing check this checker applies"
        )
    })?;
    ensure!(
        id_text.matches(PAIRED_PHRASE).count() == 1,
        "arm_noul_weight.identity.ft_rows names more than one paired metric"
    );
    let head = &id_text[..at];
    let start = head
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let paired = head[start..].to_string();
    ensure!(
        !paired.is_empty() && !paired.starts_with('.') && !paired.ends_with('.'),
        "arm_noul_weight.identity.ft_rows: no metric named before {PAIRED_PHRASE:?}"
    );
    says(
        p,
        &id_path,
        &format!("{paired}{PAIRED_PHRASE}, absent on either side refuses"),
    )?;

    // Identity: the eval rows. Their tag, and the recipe keys held equal to v5 seed 0's, which
    // must be `comparable`'s.
    let ev_path = a(&["identity", "eval_rows"]);
    let ev_text = prereg_text(p, &ev_path)?;
    let tag = Words::after(ev_text, "tag ", "arm_noul_weight.identity.eval_rows")?
        .until(";")
        .ok_or("arm_noul_weight.identity.eval_rows: no ';' after its tag")?
        .trim();
    ensure!(
        tag == EVAL_TAG,
        "arm_noul_weight.identity.eval_rows reads the {tag:?} row; this checker reads every \
         target and guard off the {EVAL_TAG} row"
    );
    says(p, &ev_path, "metrics.ft_run_row_id = the arm ft row")?;
    says(
        p,
        &ev_path,
        &format!("equal to v5 seed {}'s eval row", seeds[0]),
    )?;
    let keys: Vec<&str> = ev_text
        .match_indices("recipe.")
        .filter_map(|(i, _)| Words(&ev_text[i + "recipe.".len()..]).name())
        .collect();
    ensure!(
        keys == COMPARABLE_EVAL_KEYS,
        "arm_noul_weight.identity.eval_rows holds recipe keys {keys:?} equal to v5 seed {}'s; \
         this checker holds {COMPARABLE_EVAL_KEYS:?}",
        seeds[0]
    );

    // seed_holds, and the F2 bar it cites.
    let tf_path = a(&["comparison", "targets_form"]);
    let tf_text = prereg_text(p, &tf_path)?;
    ensure!(
        tf_text.starts_with("seed_holds."),
        "arm_noul_weight.comparison.targets_form is not seed_holds"
    );
    let mut w = Words::after(
        tf_text,
        "its count is at least half: ",
        "arm_noul_weight.comparison.targets_form",
    )?;
    let holds = CountForm::read(&mut w, "arm_noul_weight.comparison.targets_form")?;
    ensure!(
        w.eat("(>="),
        "arm_noul_weight.comparison.targets_form: {} is not followed by its printed bar (>= K/N",
        holds.text()
    );
    let printed = w
        .ratio()
        .ok_or("arm_noul_weight.comparison.targets_form: the printed bar is not K/N")?;
    says(
        p,
        &tf_path,
        "the F2 bar of campaign/v4-noul-v3b-preregistered.json",
    )?;
    says(
        p,
        &tf_path,
        &format!(
            "A target clears iff all {n_word} arm seeds hold AND the arm holds on strictly more \
             seeds than v5's envelope does"
        ),
    )?;
    let f2 = prereg_text(noul, &["ood_abstain_by_category", "bar"])?;
    let mut f2w = Words(f2);
    let f2_bar = (if f2w.eat(">=") { f2w.ratio() } else { None }).ok_or_else(|| {
        format!("campaign/v4-noul-v3b-preregistered.json's F2 bar {f2:?} is not \">= K/N\"")
    })?;
    ensure!(
        printed == f2_bar,
        "seed_holds prints the F2 bar as {}/{}, but F2 is {f2:?}",
        printed.0,
        printed.1
    );
    let bar = Frac::new(f2_bar.0, f2_bar.1, "the F2 bar")?;
    ensure!(
        holds.at_least
            && holds.is_bound(bar.k, bar.n)
            && holds.meets(bar)
            && (bar.k == 0
                || !holds.meets(Frac {
                    k: bar.k - 1,
                    n: bar.n
                })),
        "seed_holds' {} is not the F2 bar {}/{} (at least {} of {} cases)",
        holds.text(),
        bar.k,
        bar.n,
        bar.k,
        bar.n
    );

    // Room.
    let room_text = prereg_text(p, &a(&["comparison", "room"]))?;
    let mut w = Words::after(
        room_text,
        "a target has room iff v5 holds on at most ",
        "arm_noul_weight.comparison.room",
    )?;
    let at_most = w.int();
    let of = if w.eat("of its") { w.int() } else { None };
    let (room_at_most, of) = match (at_most, of) {
        (Some(k), Some(s)) if w.eat("seeds") => (k, s),
        _ => {
            return Err(
                "arm_noul_weight.comparison.room: not \"at most K of its N seeds\"".to_string(),
            );
        }
    };
    let room_at_most =
        usize::try_from(room_at_most).map_err(|_| "room bound overflows".to_string())?;
    ensure!(
        usize::try_from(of).ok() == Some(n),
        "arm_noul_weight.comparison.room counts seeds of {of}, not seeds.v5's {n}"
    );
    says(
        p,
        &a(&["comparison", "room"]),
        &format!(
            "If exactly one target has no room it is read as a guard instead (the arm must hold \
             on {n} of {n} seeds too)"
        ),
    )?;
    says(
        p,
        &a(&["comparison", "room"]),
        "if neither has room the arm does not launch",
    )?;

    // Guards' form: the strict envelope with the arm's worst seed.
    for phrase in [
        "strict envelope",
        "with the arm's worst seed as the candidate",
        "higher-better loses iff the arm's minimum < v5's minimum",
        "lower-better loses iff the arm's maximum > v5's maximum",
        "ties never lose",
        "Counts on the integers n / n_total by i128 cross-multiplication",
        "ECE on the recorded f64",
    ] {
        says(p, &a(&["comparison", "guards_form"]), phrase)?;
    }

    // Targets and guards, as listed.
    let mut targets = Vec::new();
    for (name, dir, form) in listed_in(arm, "arm_noul_weight", "targets")? {
        let m = v5_metric(name, dir, "arm_noul_weight.targets")?;
        ensure!(
            form == "seed_holds" && matches!(m.source, Source::Count(_)),
            "arm_noul_weight.targets: {name} is read as {form:?}; this checker reads targets as \
             seed_holds on counts"
        );
        let category = name.strip_prefix("ood_abstain.").unwrap_or(name);
        ensure!(
            f2.contains(category),
            "arm_noul_weight.targets: {name}'s category is not among the F2 bar's {f2:?}"
        );
        targets.push(m);
    }
    ensure!(!targets.is_empty(), "arm_noul_weight.targets is empty");
    let mut guards = Vec::new();
    for (name, dir, form) in listed_in(arm, "arm_noul_weight", "guards")? {
        let m = v5_metric(name, dir, "arm_noul_weight.guards")?;
        let want = listed(m).2;
        ensure!(
            form == want,
            "arm_noul_weight.guards: {name} is read as {form:?}; this checker reads it as {want:?}"
        );
        ensure!(
            !targets.iter().any(|t| t.name == m.name)
                && !guards.iter().any(|g: &Metric| g.name == m.name),
            "arm_noul_weight lists {name} twice"
        );
        guards.push(m);
    }

    // The absolute guard, and the F3 bound it cites.
    let ag_path = a(&["comparison", "absolute_guard"]);
    let ag_text = prereg_text(p, &ag_path)?;
    let what = "arm_noul_weight.comparison.absolute_guard";
    let mut w = Words::after(ag_text, "on every arm seed ", what)?;
    let ag_name = w
        .name()
        .ok_or_else(|| format!("{what}: no metric after \"on every arm seed\""))?;
    ensure!(w.eat("has"), "{what}: {ag_name} is not followed by \"has\"");
    let ag_form = CountForm::read(&mut w, what)?;
    ensure!(
        w.eat("(<="),
        "{what}: {} is not followed by its printed example (<= K/N",
        ag_form.text()
    );
    let (ex_k, ex_n) = w
        .ratio()
        .ok_or_else(|| format!("{what}: the printed example is not K/N"))?;
    let example = Frac::new(ex_k, ex_n, what)?;
    ensure!(
        !ag_form.at_least
            && ag_form.meets(example)
            && (ex_k >= ex_n
                || !ag_form.meets(Frac {
                    k: ex_k + 1,
                    n: ex_n
                })),
        "{what}: {} does not make {ex_k}/{ex_n} the largest count it allows",
        ag_form.text()
    );
    says(
        p,
        &ag_path,
        "campaign/v4-noul-v3b-preregistered.json defect_class_in_distribution_abstention",
    )?;
    says(p, &ag_path, "one seed over it loses")?;
    let raw_guards = arm
        .get("guards")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let ag = guards
        .iter()
        .copied()
        .find(|g| g.name == ag_name)
        .ok_or_else(|| format!("{what}: {ag_name} is not among arm_noul_weight.guards"))?;
    ensure!(
        raw_guards
            .iter()
            .any(|g| g["name"].as_str() == Some(ag_name)
                && g["form"]
                    .as_str()
                    .is_some_and(|f| f.contains("absolute_guard"))),
        "{what}: arm_noul_weight.guards does not mark {ag_name} with its absolute_guard"
    );
    let f3 = prereg_text(noul, &["defect_class_in_distribution_abstention", "bound"])?;
    let mut f3w = Words(f3);
    let pct = (if f3w.eat("<=") {
        f3w.int().filter(|_| f3w.eat("%"))
    } else {
        None
    })
    .ok_or_else(|| {
        format!("campaign/v4-noul-v3b-preregistered.json's F3 bound {f3:?} is not \"<= P%\"")
    })?;
    ensure!(
        ag_form.is_bound(pct, 100),
        "{what}: {} is not F3's {f3:?}",
        ag_form.text()
    );

    let applied = json!({
        "targets_form": tf_text,
        "room": room_text,
        "guards_form": prereg_text(p, &a(&["comparison", "guards_form"]))?,
        "absolute_guard": ag_text,
        "identity": {"ft_rows": id_text, "eval_rows": ev_text},
        "f2_bar": f2,
        "f3_bound": f3,
    });
    Ok(NoulwRule {
        seeds,
        ft_tag,
        added,
        paired,
        holds,
        bar,
        room_at_most,
        targets,
        guards,
        absolute: (ag, ag_form),
        applied,
    })
}

/// A target's count on each seed, and whether the seed holds it (seed_holds).
struct Held {
    metric: Metric,
    per_seed: Vec<(i64, Frac, bool)>,
}

impl Held {
    fn of(rule: &NoulwRule, metric: Metric, rows: &[SeedRows]) -> Result<Held> {
        let Source::Count(key) = metric.source else {
            return Err(format!("{}: seed_holds is read on counts", metric.name));
        };
        let per_seed = rows
            .iter()
            .map(|s| {
                let f = s.eval.count("metrics", key)?;
                ensure!(
                    f.n == rule.bar.n,
                    "row {} {key} is over {} cases, not the {} the F2 bar is stated over",
                    s.eval.id(),
                    f.n,
                    rule.bar.n
                );
                Ok((s.seed, f, rule.holds.meets(f)))
            })
            .collect::<Result<_>>()?;
        Ok(Held { metric, per_seed })
    }
    fn holding(&self) -> usize {
        self.per_seed.iter().filter(|(_, _, h)| *h).count()
    }
    fn json(&self) -> Value {
        json!(
            self.per_seed
                .iter()
                .map(|(s, f, h)| json!({"seed": s, "count": f.json(), "holds": h}))
                .collect::<Vec<_>>()
        )
    }
}

/// The arm's ft row is v5's same-seed ft row plus exactly the added keys (identity.ft_rows):
/// quick false, its tag, v5's code commit and data snapshot, every recipe key but the added
/// ones equal, and the paired metric (v5's batch order for that seed) recorded on both rows and
/// equal. Every failure is listed, not only the first.
fn noulw_identity(rule: &NoulwRule, ft: &Row, v5: &Row) -> Result<Value> {
    let what = format!("arm ft row {} (seed {:?})", ft.id(), ft.seed());
    let shown = |v: Option<&Value>| v.map_or("absent".to_string(), Value::to_string);
    let mut wrong: Vec<String> = Vec::new();
    if ft.get(&["quick"]) != Some(&Value::Bool(false)) {
        wrong.push(format!("quick is {}, not false", shown(ft.get(&["quick"]))));
    }
    if ft.tag() != Some(rule.ft_tag.as_str()) {
        wrong.push(format!(
            "recipe.tag is {:?}, not {:?}",
            ft.tag(),
            rule.ft_tag
        ));
    }
    for (path, name) in [
        (&["code_commit"][..], "code_commit"),
        (
            &["protocol", "data_snapshot_hash"][..],
            "protocol.data_snapshot_hash",
        ),
        (&["recipe", "shard_hash"][..], "recipe.shard_hash"),
    ] {
        let (got, want) = (ft.str_at(path), v5.str_at(path));
        if want.is_none() || got != want {
            wrong.push(format!(
                "{name} is {got:?}, not v5 seed {:?}'s {want:?}",
                v5.seed()
            ));
        }
    }
    let (got, base) = (recipe_of(ft, &what)?, recipe_of(v5, &what)?);
    for (key, want) in &rule.added {
        if let Some(v) = base.get(key) {
            wrong.push(format!(
                "v5's ft row {}: recipe.{key} is {v}, but the arm adds it",
                v5.id()
            ));
        }
        let ok = match (got.get(key), want) {
            (Some(g), Value::Number(_)) => g.as_f64().is_some() && g.as_f64() == want.as_f64(),
            (Some(g), _) => g == want,
            (None, _) => false,
        };
        if !ok {
            wrong.push(format!(
                "recipe.{key} is {}, not the added {want}",
                shown(got.get(key))
            ));
        }
    }
    let keys: BTreeSet<&String> = got.keys().chain(base.keys()).collect();
    for key in keys {
        if !rule.added.iter().any(|(k, _)| k == key) && got.get(key) != base.get(key) {
            wrong.push(format!(
                "recipe.{key} is {}, not v5's {} (the arm differs from v5 only by {:?})",
                shown(got.get(key)),
                shown(base.get(key)),
                rule.added
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>()
            ));
        }
    }
    // The pairing check: both rows record the paired metric, as a non-empty string, equal.
    let paired = |row: &Row| -> Result<String> {
        row.ran("metrics", &rule.paired)?
            .get("value")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .ok_or_else(|| {
                format!(
                    "row {} ({}): metrics.{} records no string value",
                    row.id(),
                    row.at,
                    rule.paired
                )
            })
    };
    let order = match (paired(ft), paired(v5)) {
        (Ok(a), Ok(b)) if a == b => Some(a),
        (Ok(a), Ok(b)) => {
            wrong.push(format!(
                "metrics.{} is {a}, not v5 seed {:?}'s {b}: not v5's batch order (the pairing \
                 check)",
                rule.paired,
                v5.seed()
            ));
            None
        }
        (a, b) => {
            for e in [a.err(), b.err()].into_iter().flatten() {
                wrong.push(format!(
                    "{e} (the pairing check: absent on either side refuses)"
                ));
            }
            None
        }
    };
    ensure!(wrong.is_empty(), "{what}: {}", wrong.join("; "));
    Ok(json!({
        "seed": ft.seed(),
        "arm_ft_row": ft.id(),
        "v5_ft_row": v5.id(),
        "added": rule.added.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<Map<_, _>>(),
        "paired": {rule.paired.as_str(): order},
    }))
}

/// The arm's rows read and judged: each target with room clears iff every arm seed holds it and
/// the arm holds it on more seeds than v5; a target without room is a guard the arm holds on
/// every seed; every listed guard compares the arm's worst seed with v5's envelope (strict);
/// the absolute guard holds on every arm seed. Returns whether it wins, and the record.
fn judge_noulw(
    inputs: &mut Inputs,
    rule: &NoulwRule,
    envelope: &[SeedRows],
    room: &[(Held, bool)],
    arm_ledger: &Path,
    arm_ft: &[(i64, String)],
) -> Result<(bool, Value)> {
    let seeds: Vec<i64> = arm_ft.iter().map(|(s, _)| *s).collect();
    ensure!(
        seeds == rule.seeds,
        "the arm's ft rows are seeds {:?} (identity.ft_rows), given in that order; got {seeds:?}",
        rule.seeds
    );
    let ledger = inputs.read(arm_ledger)?;
    let reference = envelope
        .first()
        .ok_or_else(|| "an empty envelope".to_string())?;
    let mut arm = Vec::new();
    let mut identity = Vec::new();
    for ((seed, ft), v5) in arm_ft.iter().zip(envelope) {
        let rows = seed_rows(&ledger, *seed, ft, false, false)?;
        identity.push(noulw_identity(rule, rows.ft, v5.ft)?);
        comparable(&rows, reference, &format!("arm seed {seed}"))?;
        arm.push(rows);
    }
    // The arm's three ft rows are one configuration, as v5's are: one recipe hash and data
    // snapshot (batch_order = "seed" is a constant, so per-seed orders keep one recipe hash).
    // The per-seed identity above does not compare protocol.recipe_hash; this does.
    let (_, arm_recipe_hash, _) = one_configuration(&ledger, arm_ft)?;
    let n = rule.seeds.len();
    let mut lost: Vec<String> = Vec::new();
    let mut cleared = true;
    let mut targets = Vec::new();
    for (v5_held, has_room) in room {
        let held = Held::of(rule, v5_held.metric, &arm)?;
        let (arm_n, v5_n) = (held.holding(), v5_held.holding());
        let mut j = json!({
            "target": v5_held.metric.name,
            "room": has_room,
            "v5_holds": v5_n,
            "arm_holds": arm_n,
            "of": n,
            "arm_per_seed": held.json(),
        });
        if *has_room {
            let clears = arm_n == n && arm_n > v5_n;
            cleared &= clears;
            j["clears"] = json!(clears);
        } else {
            let loses = arm_n < n;
            if loses {
                lost.push(format!(
                    "{} (no room; read as a guard)",
                    v5_held.metric.name
                ));
            }
            j["read_as"] = json!("guard: the arm must hold on every seed");
            j["loses"] = json!(loses);
        }
        targets.push(j);
    }
    let mut guards = Vec::new();
    for m in &rule.guards {
        let (v5_values, v5_min, v5_max) = envelope_range(*m, envelope)?;
        let (arm_values, arm_min, arm_max) = envelope_range(*m, &arm)?;
        let candidate = match m.dir {
            Dir::Higher => arm_min,
            Dir::Lower => arm_max,
        };
        let l = loses(candidate, v5_min, v5_max, m.dir)?;
        if l {
            lost.push(m.name.to_string());
        }
        guards.push(json!({
            "metric": m.name,
            "direction": dir_word(m.dir),
            "v5": v5_values.iter().zip(envelope).map(|(v, s)| json!({"seed": s.seed, "value": v.json()})).collect::<Vec<_>>(),
            "v5_min": v5_min.json(),
            "v5_max": v5_max.json(),
            "arm": arm_values.iter().zip(&arm).map(|(v, s)| json!({"seed": s.seed, "value": v.json()})).collect::<Vec<_>>(),
            "candidate": candidate.json(),
            "loses": l,
        }));
    }
    let (am, form) = rule.absolute;
    let Source::Count(key) = am.source else {
        return Err(format!(
            "{}: the absolute guard is read on a count",
            am.name
        ));
    };
    let per_seed: Vec<(i64, Frac, bool)> = arm
        .iter()
        .map(|s| {
            let f = s.eval.count("metrics", key)?;
            Ok((s.seed, f, form.meets(f)))
        })
        .collect::<Result<_>>()?;
    let over: Vec<i64> = per_seed
        .iter()
        .filter(|(_, _, ok)| !ok)
        .map(|(s, _, _)| *s)
        .collect();
    if !over.is_empty() {
        lost.push(format!(
            "{} absolute guard ({}) on seeds {over:?}",
            am.name,
            form.text()
        ));
    }
    let win = cleared && lost.is_empty();
    Ok((
        win,
        json!({
            "wins": win,
            "rows": arm.iter().map(SeedRows::json).collect::<Vec<_>>(),
            "recipe_hash": arm_recipe_hash,
            "identity": identity,
            "targets_with_room_all_clear": cleared,
            "targets": targets,
            "guards": guards,
            "absolute_guard": {
                "metric": am.name,
                "form": form.text(),
                "per_seed": per_seed.iter().map(|(s, f, ok)| json!({"seed": s, "count": f.json(), "holds": ok})).collect::<Vec<_>>(),
                "loses": !over.is_empty(),
            },
            "lost": lost,
        }),
    ))
}

/// (viii). Room is decided from v5's envelope alone, before the arm's ledger is opened. With
/// `arm` absent (`--room`) the word is `room` iff some target has room, else `no_room`. With it,
/// `wins`, `quiet` or `refused`; no target with room refuses, as does anything unreadable.
fn rule_v5_noulw(
    inputs: &mut Inputs,
    preregistration: &Path,
    noul_preregistration: &Path,
    v5_ledger: &Path,
    ft_rows: &[(i64, String)],
    arm: Option<(&Path, &[(i64, String)])>,
) -> Result<(String, Value)> {
    let (p, sha256) = inputs.read_preregistration(preregistration)?;
    let (noul, noul_sha256) = inputs.read_preregistration(noul_preregistration)?;
    let rule = noulw_rule(&p, &noul)?;
    let seeds: Vec<i64> = ft_rows.iter().map(|(s, _)| *s).collect();
    ensure!(
        seeds == rule.seeds,
        "the envelope is v5 seeds {:?} only (arm_noul_weight.envelope), given in that order; got \
         {seeds:?}",
        rule.seeds
    );
    let env = inputs.read(v5_ledger)?;
    one_configuration(&env, ft_rows)?;
    let envelope: Vec<SeedRows> = ft_rows
        .iter()
        .map(|(seed, ft)| seed_rows(&env, *seed, ft, false, false))
        .collect::<Result<_>>()?;
    let reference = &envelope[0];
    for s in &envelope[1..] {
        comparable(s, reference, &format!("v5 envelope seed {}", s.seed))?;
    }
    // Room: "decided from the envelope alone, before the arm's rows are read".
    let room: Vec<(Held, bool)> = rule
        .targets
        .iter()
        .map(|m| {
            let held = Held::of(&rule, *m, &envelope)?;
            let has_room = held.holding() <= rule.room_at_most;
            Ok((held, has_room))
        })
        .collect::<Result<_>>()?;
    let any_room = room.iter().any(|(_, r)| *r);
    let body = json!({
        "rule": rule.applied,
        "preregistration_sha256": sha256,
        "noul_preregistration_sha256": noul_sha256,
        "envelope": {
            "pin": format!("v5 seeds {} only (never seeds 3-4, never F)", and_list(&rule.seeds)),
            "seeds": envelope.iter().map(SeedRows::json).collect::<Vec<_>>(),
        },
        "seed_holds": {"form": rule.holds.text(), "f2_bar": rule.bar.json()},
        "room": room.iter().map(|(h, r)| json!({
            "target": h.metric.name,
            "v5_holds": h.holding(),
            "of": rule.seeds.len(),
            "room_iff_at_most": rule.room_at_most,
            "room": r,
            "v5_per_seed": h.json(),
        })).collect::<Vec<_>>(),
        "not_checked": NOT_CHECKED_SUITE,
    });
    let Some((arm_ledger, arm_ft)) = arm else {
        let mut body = body;
        body["then"] = json!(if any_room {
            "the noul-weight arm launches in its slot (arm_noul_weight.launch_condition)"
        } else {
            "the noul-weight arm does not launch unless the human pins V5NW_HUMAN_YES (R7); \
             J5' takes the slot"
        });
        return Ok((ROOM_WORDS[usize::from(!any_room)].to_string(), body));
    };
    let mut s = Settled {
        because: Vec::new(),
        body,
        wins: None,
    };
    if !any_room {
        s.refuse(format!(
            "no target has room: v5 holds every target on more than {} of its {} seeds, so none \
             can clear; the arm does not launch (launch_condition) and rows that exist anyway \
             read refused",
            rule.room_at_most,
            rule.seeds.len()
        ));
    }
    match judge_noulw(inputs, &rule, &envelope, &room, arm_ledger, arm_ft) {
        Ok((win, detail)) => {
            s.body["arm"] = detail;
            s.wins = Some(vec![win]);
        }
        Err(e) => s.refuse(e),
    }
    let word = match s.wins.as_deref() {
        None => None,
        Some(&[true]) => Some((
            V5NW_WORDS[0],
            "the noul-weight arm wins: this feeds the next re-plan only and promotes nothing",
        )),
        Some(&[false]) => Some((
            V5NW_WORDS[1],
            "the noul-weight arm does not win: a target with room did not clear or a guard lost; \
             the next re-plan reads the rows",
        )),
        Some(other) => return Err(format!("{} verdicts for one arm", other.len())),
    };
    s.finish(word, "no reading; the human decides from the rows")
}

// --- (ix): tierb, the v5 pre-registration's recipe.tierb_outcome_rule ----------------------------
//
// The Tier-B outcome rule for one candidate run (no-mask, C2a; fused AdamW, C2b), as Fable amended
// it on 2026-10-03 before any outcome row existed. The envelope reuses the successor's exact
// comparison (`min_max`, `clears`), the row look-ups are the file's (`Ledger::one`,
// `one_configuration`, `scored_from`, `control_row_of`, `same_recipe_keys`), and the
// pre-registration is read at run time and must agree with this code (`tierb_agrees`).

/// The file as it binds; the text a decision applied is the one whose sha256 it records.
const PREREG_TIERB: &str = "campaign/v5-preregistered.json recipe.tierb_outcome_rule and \
                            recipe.conditionals C2a / C2b (Fable's amendment of 2026-10-03; the \
                            DRAFT or the renamed file, read at run time, its sha256 and draft \
                            flag recorded)";
/// outcome_rule.checker, .words, .on_word and .rule, as C2a and C2b write them.
const TIERB_CHECKER: &str = "qd-post-f-rules tierb";
const TIERB_WORDS: [&str; 3] = ["pass", "fail", "refused"];
const TIERB_ON_WORD: &str = "pass";
const TIERB_RULE: &str = "recipe.tierb_outcome_rule";
/// envelope.rows: phase-3 seeds 0, 1 and 2's epoch-score-val rows, in the phase-3 ledger.
const TIERB_ENVELOPE: [&str; 3] = [
    "6d170b3c-5676-446b-b6be-5bb17d6d7aa0",
    "60f29b07-c7d1-41a9-a8e5-af96227206a6",
    "5c19c0e8-0e8d-4aa9-b9c6-73febce8a57d",
];
/// envelope.metrics.
const TIERB_METRICS: [Metric; 2] = [VAL_CHOICE, VAL_SPAN];
/// rows.identity: phase-3 seed 0's ft row, whose recipe the candidate's must agree with.
const TIERB_FT: &str = "7f2c11db-3eb8-4361-9620-6b164ec37f1e";
/// clauses.2: phase-3 seed 0's linear control, CI [+0.1359, +0.1651].
const TIERB_CONTROL: &str = "eeda5db4-0e99-415c-9a9d-5bd0efc8b71b";
const TIERB_CONTROL_CI: (i64, i64) = (1359, 1651);
/// clauses.3: phase-3 seed 0's fp32 all-gates re-score of 7f2c11db.
const TIERB_ALLGATES: &str = "58fd1532-3239-41eb-a506-33ba10079425";
/// 58fd1532's gates as committed: `Some(passed)` for a gate that ran, `None` for one that did
/// not. A reference row that says otherwise is not the row the rule names.
const TIERB_ALLGATES_GATES: [(&str, Option<bool>); 5] = [
    ("ece", Some(true)),
    ("needle_hunk_recall", Some(false)),
    ("ood_abstain", Some(false)),
    ("paired_margin_vs_linear", None),
    ("permutation_consistency", Some(true)),
];
const TIERB_SHARD_HASH: &str = "d773b87666e1b042279271ab0f891246b7268d4ce0cad2c3e677bb415c147e1a";
const TIERB_VAL_SHARD_HASH: &str =
    "105513b98887a24391339f8a19ac9dfef07645abd258a22482087b0699066a72";
const TIERB_BACKBONE: &str = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c";
/// The outcome run is one seed-0 run (tools/perf_tierb_outcome.sh `--seeds 0`).
const TIERB_SEED: i64 = 0;
/// rows.candidate: the linear control is the row this tool wrote.
const LINEAR_CONTROL_TOOL: &str = "tools/ft_linear_control.py";
const LINEAR_GATE: &str = "paired_margin_vs_linear";
/// paired_margin_test's default bootstrap count, which every control row so far records.
const CI_N_BOOT: u64 = 10_000;
/// The straddling suffix paired_margin_test appends when lo <= 0 and hi >= 0.
const CI_INCLUDES_ZERO: &str = " -- CI includes zero, so this is not a win";
/// envelope.literal: the design doc's printed decimals, report-only.
const TIERB_LITERAL: [(&str, [Dec; 2], &str); 2] = [
    (
        "val_top1.choice",
        [
            Dec {
                num: 997,
                den: 1000,
                text: "0.997",
            },
            Dec {
                num: 998,
                den: 1000,
                text: "0.998",
            },
        ],
        "99.7-99.8%",
    ),
    (
        "val_top1.span",
        [
            Dec {
                num: 998,
                den: 1000,
                text: "0.998",
            },
            Dec {
                num: 999,
                den: 1000,
                text: "0.999",
            },
        ],
        "99.8-99.9%",
    ),
];

/// rows.identity: the recipe keys the candidate's ft row must share with 7f2c11db, the value
/// 7f2c11db itself must carry for each, and that value as rows.identity prints it.
fn tierb_identity() -> [(&'static str, Value, &'static str); 8] {
    [
        ("shard_hash", json!(TIERB_SHARD_HASH), TIERB_SHARD_HASH),
        ("batch_tokens", json!(16384), "16384"),
        ("optimizer_recipe", json!("master"), "master"),
        ("lr", json!(1e-05), "1e-05"),
        ("no_memorise", json!(true), "true"),
        ("passes", json!(1), "1"),
        ("tag", json!("epoch"), "epoch"),
        ("backbone_snapshot", json!(TIERB_BACKBONE), TIERB_BACKBONE),
    ]
}

/// The candidate a tierb decision is about: each is screened alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Candidate {
    /// `--train-attention-mask none` (C2a): recipe.train_attention_mask = "none"
    /// (1cf8e6d tools/real_ft_run.py:360).
    Nomask,
    /// `--fused-adamw` (C2b): recipe.optimizer_fused = true (e19dcf6 tools/real_ft_run.py:336-337).
    Fused,
}

impl Candidate {
    fn name(self) -> &'static str {
        match self {
            Candidate::Nomask => "nomask",
            Candidate::Fused => "fused",
        }
    }
    /// Its own recipe key and value: outcome_rule.candidate_recipe.
    fn key(self) -> (&'static str, Value) {
        match self {
            Candidate::Nomask => ("train_attention_mask", json!("none")),
            Candidate::Fused => ("optimizer_fused", json!(true)),
        }
    }
    fn recipe(self) -> Value {
        let (key, value) = self.key();
        json!({ key: value })
    }
    /// The other candidate's key, and the one value of it this candidate's run may record.
    /// `real_ft_run.py` writes `optimizer_fused` only when fused and `train_attention_mask` only
    /// when it is not "padding" (1cf8e6d:357-360), so a masked row has no such key; "padding"
    /// written out is still the masked path. `optimizer_fused` has no value a no-mask run carries.
    fn other(self) -> (&'static str, Option<Value>) {
        match self {
            Candidate::Nomask => ("optimizer_fused", None),
            Candidate::Fused => ("train_attention_mask", Some(json!("padding"))),
        }
    }
}

/// A signed number printed by Python's `{:+.4f}`, in ten-thousandths. The printed sign is kept
/// apart, since "-0.0000" (a negative number that rounds to zero) is not "+0.0000".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fixed4 {
    negative: bool,
    ten_thousandths: i64,
}

impl Fixed4 {
    /// `[+-]D+.DDDD`, with no more than nine integer digits; anything else is not this form.
    fn parse(s: &str) -> Option<Fixed4> {
        let negative = match s.as_bytes().first()? {
            b'+' => false,
            b'-' => true,
            _ => return None,
        };
        let (int, frac) = s[1..].split_once('.')?;
        let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
        if !digits(int) || int.len() > 9 || frac.len() != 4 || !digits(frac) {
            return None;
        }
        let magnitude = int.parse::<i64>().ok()? * 10_000 + frac.parse::<i64>().ok()?;
        Some(Fixed4 {
            negative,
            ten_thousandths: if negative { -magnitude } else { magnitude },
        })
    }
    fn magnitude(self) -> String {
        let m = self.ten_thousandths.abs();
        format!("{}.{:04}", m / 10_000, m % 10_000)
    }
    /// As `{:+.4f}` printed it.
    fn text(self) -> String {
        format!(
            "{}{}",
            if self.negative { '-' } else { '+' },
            self.magnitude()
        )
    }
    /// `{-x:.4f}` of the number this prints: no plus sign, and a minus iff this one had none.
    fn negated_plain(self) -> String {
        format!(
            "{}{}",
            if self.negative { "" } else { "-" },
            self.magnitude()
        )
    }
}

/// A paired-margin gate's 95% CI, read from its `detail` string: the only place a control row
/// records it (GAP-LINEAR-CONTROL-CI-ONLY-IN-DETAIL-STRING-2026-10-03).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Ci {
    point: Fixed4,
    lo: Fixed4,
    hi: Fixed4,
    n_boot: u64,
    passed: bool,
}

impl Ci {
    fn json(&self) -> Value {
        json!({
            "point": self.point.ten_thousandths,
            "lo": self.lo.ten_thousandths,
            "hi": self.hi.ten_thousandths,
            "unit": "1e-4 (the printed four decimals as integers)",
            "printed": format!("{} [{}, {}]", self.point.text(), self.lo.text(), self.hi.text()),
            "n_boot": self.n_boot,
            "passed": self.passed,
        })
    }
    /// Clause 2: the two intervals share a point, ends included.
    fn overlaps(&self, other: &Ci) -> bool {
        self.lo.ten_thousandths <= other.hi.ten_thousandths
            && other.lo.ten_thousandths <= self.hi.ten_thousandths
    }
}

/// The form `qd_train.eval_harness.paired_margin_test` writes (884b658
/// python/qd_train/eval_harness.py:282-287), for refusals.
const CI_FORM: &str = "\"paired margin {point:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] over {n_boot} \
                       bootstrap resamples\" then \"\" iff lo > 0 (passed), else \" -- CI \
                       includes zero, so this is not a win\" (hi >= 0) or \" -- the whole \
                       interval is below zero, so this is not an inconclusive result: the \
                       baseline beats the model by {-point:.4f} and the comparison separates \
                       them\" (hi < 0)";

/// The detail string paired_margin_test writes for these numbers, byte for byte. The suffix
/// follows Python's own branches: `passed` (lo > 0), else hi < 0, which is the printed sign of
/// hi, else the straddling sentence.
fn ci_detail(point: Fixed4, lo: Fixed4, hi: Fixed4, n_boot: u64, passed: bool) -> String {
    let suffix = if passed {
        String::new()
    } else if hi.negative {
        format!(
            " -- the whole interval is below zero, so this is not an inconclusive result: the \
             baseline beats the model by {} and the comparison separates them",
            point.negated_plain()
        )
    } else {
        CI_INCLUDES_ZERO.to_string()
    };
    format!(
        "paired margin {}, 95% CI [{}, {}] over {n_boot} bootstrap resamples{suffix}",
        point.text(),
        lo.text(),
        hi.text()
    )
}

/// The point, bounds and bootstrap count a paired-margin detail prints, read loosely; `paired_ci`
/// then re-renders the whole string and requires it byte for byte.
fn ci_numbers(detail: &str) -> Option<(Fixed4, Fixed4, Fixed4, u64)> {
    let rest = detail.strip_prefix("paired margin ")?;
    let (point, rest) = rest.split_once(", 95% CI [")?;
    let (lo, rest) = rest.split_once(", ")?;
    let (hi, rest) = rest.split_once("] over ")?;
    let (n_boot, _) = rest.split_once(" bootstrap resamples")?;
    Some((
        Fixed4::parse(point)?,
        Fixed4::parse(lo)?,
        Fixed4::parse(hi)?,
        n_boot.parse::<u64>().ok()?,
    ))
}

/// `gates.paired_margin_vs_linear` of a control row, its CI parsed from `detail`. Refused: a gate
/// that did not run; a detail that is not, byte for byte, what paired_margin_test writes for the
/// numbers it prints (re-rendered and compared, so every deviation refuses); a bootstrap count
/// other than 10000; lo above hi; a `passed` flag that is not `lo > 0` on the printed lo; a
/// printed point that is not the gate's `value` to four places. At a rounding edge (lo printed
/// "+0.0000" with passed true) the printed text cannot show what the float decided, and that
/// refuses too.
fn paired_ci(row: &Row) -> Result<Ci> {
    let what = format!("row {} gates.{LINEAR_GATE}", row.id());
    let gate = row.ran("gates", LINEAR_GATE)?;
    let detail = gate
        .get("detail")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{what}: no detail string, which is where the CI is recorded"))?;
    let value = recorded_value(gate, &what)?;
    let passed = gate
        .get("passed")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{what}: no boolean passed"))?;
    let off_form = |why: String| {
        format!(
            "{what}: detail {detail:?} is not in the form paired_margin_test writes ({why}); the \
             form is {CI_FORM}"
        )
    };
    let (point, lo, hi, n_boot) = ci_numbers(detail).ok_or_else(|| {
        off_form(
            "its point, bounds and bootstrap count do not parse as {:+.4f} and an integer".into(),
        )
    })?;
    ensure!(
        n_boot == CI_N_BOOT,
        "{what}: {n_boot} bootstrap resamples, not the {CI_N_BOOT} every control row records"
    );
    ensure!(
        lo.ten_thousandths <= hi.ten_thousandths,
        "{what}: lo {} is above hi {}",
        lo.text(),
        hi.text()
    );
    ensure!(
        passed == (lo.ten_thousandths > 0),
        "{what}: passed is {passed} but the printed lo is {}; paired_margin_test passes iff lo > 0",
        lo.text()
    );
    let expected = ci_detail(point, lo, hi, n_boot, passed);
    if detail != expected {
        return Err(off_form(format!(
            "for passed {passed} and hi {} the suffix and text are {expected:?}",
            hi.text()
        )));
    }
    let shown = format!("{value:+.4}");
    ensure!(
        shown == point.text(),
        "{what}: the printed point {} is not the gate's value {value} to four places ({shown})",
        point.text()
    );
    Ok(Ci {
        point,
        lo,
        hi,
        n_boot,
        passed,
    })
}

/// The pre-registration's own words this checker applies, as (path under
/// `recipe.tierb_outcome_rule`, a phrase that text must hold). Each pinned id and hash must appear
/// in the field that names it, and each reading in the words the amendment wrote; an amended
/// text changes them and refuses until the checker follows it.
fn tierb_rule_text() -> Vec<(&'static [&'static str], String)> {
    const INSIDE: &[&str] = &["envelope", "inside"];
    const CANDIDATE: &[&str] = &["rows", "candidate"];
    const IDENTITY: &[&str] = &["rows", "identity"];
    const CLAUSE_1: &[&str] = &["clauses", "1"];
    const CLAUSE_2: &[&str] = &["clauses", "2"];
    const CLAUSE_3: &[&str] = &["clauses", "3"];
    const WORD: &[&str] = &["word"];
    let fixed = |t: i64| format!("+{}.{:04}", t / 10_000, t % 10_000);
    let mut out: Vec<(&'static [&'static str], String)> = vec![
        (INSIDE, "2*min - max <= c <= 2*max - min".into()),
        (
            INSIDE,
            "compared on the rows' integer counts by cross-multiplication".into(),
        ),
        (INSIDE, "A range of 0 refuses".into()),
        (
            CANDIDATE,
            "exactly one completed, non-quick ft row tagged epoch at seed 0".into(),
        ),
        (
            CANDIDATE,
            "eval row scored from it without score_dtype".into(),
        ),
        (
            CANDIDATE,
            "with score_dtype fp32 and the needle and OOD suites".into(),
        ),
        (
            CANDIDATE,
            format!("{LINEAR_CONTROL_TOOL} row whose recipe.eval_row_id is the first eval row"),
        ),
        (IDENTITY, TIERB_FT.into()),
        (IDENTITY, "not the other candidate's key".into()),
        (IDENTITY, format!("val_shard_hash {TIERB_VAL_SHARD_HASH}")),
        (
            CLAUSE_1,
            "val_top1.choice and val_top1.span each inside the envelope".into(),
        ),
        (
            CLAUSE_2,
            format!(
                "{TIERB_CONTROL}'s [{}, {}]",
                fixed(TIERB_CONTROL_CI.0),
                fixed(TIERB_CONTROL_CI.1)
            ),
        ),
        (CLAUSE_2, "lo <= other hi and other lo <= hi".into()),
        (
            CLAUSE_2,
            "a passed flag that disagrees with lo > 0 refuses".into(),
        ),
        (CLAUSE_3, TIERB_ALLGATES.into()),
        (
            CLAUSE_3,
            "is ran with the same passed in the candidate's all-gates row".into(),
        ),
        (
            CLAUSE_3,
            "a gate ran there and not_run in the candidate refuses".into(),
        ),
        (WORD, "pass iff clauses 1, 2 and 3 all hold".into()),
        (WORD, "Never pass by absence.".into()),
    ];
    for (key, _, printed) in tierb_identity() {
        out.push((IDENTITY, format!("{key} {printed}")));
    }
    out
}

/// The pre-registration agrees with this code: the one conditional whose outcome_rule names this
/// candidate carries this checker, its words, its on-word, this rule and this candidate's recipe
/// key; `recipe.tierb_outcome_rule` lists this envelope's rows and metrics exactly; and its prose
/// says every phrase of `tierb_rule_text`. Its `draft` key is not checked: the decision is taken
/// before the rename, and the JSON records which file it read.
fn tierb_agrees(p: &Map<String, Value>, candidate: Candidate) -> Result<()> {
    let conditionals = p
        .get("recipe")
        .and_then(|r| r.get("conditionals"))
        .and_then(Value::as_array)
        .ok_or("the pre-registration has no recipe.conditionals list")?;
    let found: Vec<&Map<String, Value>> = conditionals
        .iter()
        .filter_map(|c| c.get("outcome_rule").and_then(Value::as_object))
        .filter(|o| o.get("candidate").and_then(Value::as_str) == Some(candidate.name()))
        .collect();
    ensure!(
        found.len() == 1,
        "the pre-registration has {} recipe.conditionals whose outcome_rule.candidate is {:?}, \
         not one",
        found.len(),
        candidate.name()
    );
    let rule = found[0];
    for (key, want) in [
        ("checker", json!(TIERB_CHECKER)),
        ("words", json!(TIERB_WORDS)),
        ("on_word", json!(TIERB_ON_WORD)),
        ("rule", json!(TIERB_RULE)),
        ("candidate_recipe", candidate.recipe()),
    ] {
        ensure!(
            rule.get(key) == Some(&want),
            "the pre-registration's {} outcome_rule.{key} is {}, not this checker's {want}",
            candidate.name(),
            rule.get(key).map_or("absent".to_string(), Value::to_string)
        );
    }
    let outcome = p
        .get("recipe")
        .and_then(|r| r.get("tierb_outcome_rule"))
        .and_then(Value::as_object)
        .ok_or(
            "the pre-registration has no recipe.tierb_outcome_rule object, the rule C2a / C2b name",
        )?;
    let envelope = outcome.get("envelope");
    let rows = envelope.and_then(|e| e.get("rows"));
    ensure!(
        rows == Some(&json!(TIERB_ENVELOPE)),
        "the pre-registration's recipe.tierb_outcome_rule.envelope.rows are {}, not this \
         checker's {TIERB_ENVELOPE:?}",
        rows.map_or("absent".to_string(), Value::to_string)
    );
    let metrics = envelope.and_then(|e| e.get("metrics"));
    let want: Vec<&str> = TIERB_METRICS.iter().map(|m| m.name).collect();
    ensure!(
        metrics == Some(&json!(want)),
        "the pre-registration's recipe.tierb_outcome_rule.envelope.metrics are {}, not this \
         checker's {want:?}",
        metrics.map_or("absent".to_string(), Value::to_string)
    );
    for (tail, phrase) in tierb_rule_text() {
        let path: Vec<&str> = ["recipe", "tierb_outcome_rule"]
            .iter()
            .chain(tail)
            .copied()
            .collect();
        says(p, &path, &phrase)?;
    }
    Ok(())
}

/// A count metric of a row, exactly.
fn count_val(row: &Row, m: Metric) -> Result<(Frac, Rat)> {
    let f = row.count("metrics", m.name)?;
    Ok((
        f,
        Rat::of_count(f, &format!("row {} {}", row.id(), m.name))?,
    ))
}

/// `ka * a - kb * b` as a fraction, for the JSON: over the shared denominator when the two have
/// one (every val count here), else over their product.
fn combined(ka: i128, a: Rat, kb: i128, b: Rat) -> Result<Value> {
    let ((an, ad), (bn, bd)) = (a.parts(), b.parts());
    let (n, d) = if ad == bd {
        (ka * an - kb * bn, ad)
    } else {
        (ka * an * bd - kb * bn * ad, ad * bd)
    };
    // Denominators are at most MAX_DENOMINATOR (1e9), so both fit an i64; refuse if not.
    let fit = |v: i128| i64::try_from(v).map_err(|_| format!("{v} does not fit the JSON's i64"));
    Ok(json!({"n": fit(n)?, "n_total": fit(d)?}))
}

/// "Inside the envelope": neither above max nor below min by more than the range, the exact
/// complement of `clears` in both directions.
fn inside(c: Val, min: Val, max: Val) -> Result<bool> {
    Ok(!clears(c, min, max, Dir::Higher)? && !clears(c, min, max, Dir::Lower)?)
}

/// The envelope of one metric: its three rows' values, min and max (a range of 0 refuses).
struct TierbEnvelope {
    metric: Metric,
    min: Val,
    max: Val,
    json: Value,
}

impl TierbEnvelope {
    fn of(metric: Metric, rows: &[&Row]) -> Result<TierbEnvelope> {
        let mut values = Vec::new();
        let mut shown = Vec::new();
        for r in rows {
            let (f, rat) = count_val(r, metric)?;
            values.push(Val::Exact(rat));
            shown.push(json!({"row": r.id(), "n": f.k, "n_total": f.n, "value": f.f64()}));
        }
        let (min, max) = min_max(metric.name, &values)?;
        ensure!(
            cmp_val(min, max)? == std::cmp::Ordering::Less,
            "{}: the envelope's range is 0 ({} on every row); a range of 0 refuses \
             (envelope.inside), with no floor invented",
            metric.name,
            text(min)
        );
        let (Val::Exact(lo), Val::Exact(hi)) = (min, max) else {
            return Err(format!("{}: the envelope is not exact counts", metric.name));
        };
        let json = json!({
            "values": shown,
            "min": min.json(),
            "max": max.json(),
            "range": combined(1, hi, 1, lo)?,
            "inside_from": combined(2, lo, 1, hi)?,
            "inside_to": combined(2, hi, 1, lo)?,
        });
        Ok(TierbEnvelope {
            metric,
            min,
            max,
            json,
        })
    }
    /// One candidate row's value against this envelope, with the literal decimals beside it.
    fn judge(&self, row: &Row) -> Result<(bool, Value)> {
        let (f, rat) = count_val(row, self.metric)?;
        let c = Val::Exact(rat);
        let holds = inside(c, self.min, self.max)?;
        let (bands, band_text) = TIERB_LITERAL
            .iter()
            .find(|(name, _, _)| *name == self.metric.name)
            .map(|(_, bands, text)| (bands, *text))
            .ok_or_else(|| format!("{}: no literal band", self.metric.name))?;
        Ok((
            holds,
            json!({
                "row": row.id(),
                "candidate": c.json(),
                "inside": holds,
                "literal": {
                    "band": band_text,
                    "inside": bands.iter().any(|d| f.rounds_to(*d)),
                    "role": "report-only: the design doc's printed decimals (the value printed \
                             to one decimal percent lies in the band); it decides nothing",
                },
            }),
        ))
    }
}

/// The reference rows the rule names, each checked to be the row it names: the envelope rows,
/// 7f2c11db (its identity keys at the pinned values), eeda5db4's CI (the pinned interval) and
/// 58fd1532's gates (the pinned flags).
struct TierbReference<'a> {
    envelope: Vec<&'a Row>,
    ft: &'a Row,
    ci: Ci,
    gates: Vec<(&'static str, bool)>,
}

fn tierb_reference<'a>(phase3: &'a Ledger, allgates: &'a Ledger) -> Result<TierbReference<'a>> {
    let mut envelope = Vec::new();
    for id in TIERB_ENVELOPE {
        let r = phase3.by_id(id)?;
        ensure!(
            r.completed()
                && r.str_at(&["run_kind"]) == Some("eval")
                && r.tag() == Some(EVAL_TAG)
                && r.get(&["recipe", "score_dtype"]).is_none(),
            "envelope row {id} in {} is not a completed {EVAL_TAG} row scored in its training run \
             (run_kind {:?}, status {:?}, tag {:?}, score_dtype {:?})",
            phase3.shown,
            r.str_at(&["run_kind"]),
            r.str_at(&["status"]),
            r.tag(),
            r.get(&["recipe", "score_dtype"])
        );
        ensure!(
            r.str_at(&["recipe", "val_shard_hash"]) == Some(TIERB_VAL_SHARD_HASH),
            "envelope row {id}: recipe.val_shard_hash {:?} is not the pinned {TIERB_VAL_SHARD_HASH}",
            r.str_at(&["recipe", "val_shard_hash"])
        );
        envelope.push(r);
    }
    let ft = ft_row(phase3, TIERB_SEED, TIERB_FT)?;
    for (key, want, _) in tierb_identity() {
        ensure!(
            ft.get(&["recipe", key]) == Some(&want),
            "reference ft row {TIERB_FT}: recipe.{key} is {}, not the pinned {want} \
             (rows.identity)",
            ft.get(&["recipe", key])
                .map_or("absent".to_string(), Value::to_string)
        );
    }
    let control = phase3.by_id(TIERB_CONTROL)?;
    ensure!(
        control.completed()
            && control.str_at(&["recipe", "tool"]) == Some(LINEAR_CONTROL_TOOL)
            && control.str_at(&["recipe", "eval_row_id"]) == Some(TIERB_ENVELOPE[0]),
        "reference control row {TIERB_CONTROL} is not a completed {LINEAR_CONTROL_TOOL} row of \
         {} (status {:?}, tool {:?}, eval_row_id {:?})",
        TIERB_ENVELOPE[0],
        control.str_at(&["status"]),
        control.str_at(&["recipe", "tool"]),
        control.str_at(&["recipe", "eval_row_id"])
    );
    let ci = paired_ci(control)?;
    ensure!(
        (ci.lo.ten_thousandths, ci.hi.ten_thousandths) == TIERB_CONTROL_CI,
        "reference control row {TIERB_CONTROL}: its CI [{}, {}] is not the pinned [+0.1359, \
         +0.1651]; the record and its row disagree",
        ci.lo.text(),
        ci.hi.text()
    );
    let all = allgates.by_id(TIERB_ALLGATES)?;
    ensure!(
        all.completed()
            && all.str_at(&["run_kind"]) == Some("eval")
            && all.tag() == Some(EVAL_TAG)
            && scored_from(all, TIERB_SEED, TIERB_FT)
            && all.str_at(&["recipe", "score_dtype"]) == Some("fp32")
            && all.str_at(&["recipe", "val_shard_hash"]) == Some(TIERB_VAL_SHARD_HASH),
        "reference all-gates row {TIERB_ALLGATES} in {} is not a completed fp32 {EVAL_TAG} \
         re-score of {TIERB_FT} on val_shard_hash {TIERB_VAL_SHARD_HASH}",
        allgates.shown
    );
    let row_gates = all
        .get(&["gates"])
        .and_then(Value::as_object)
        .ok_or_else(|| format!("reference all-gates row {TIERB_ALLGATES}: no gates"))?;
    let named: BTreeSet<&str> = TIERB_ALLGATES_GATES.iter().map(|(g, _)| *g).collect();
    let recorded: BTreeSet<&str> = row_gates.keys().map(String::as_str).collect();
    ensure!(
        named == recorded,
        "reference all-gates row {TIERB_ALLGATES}: gates {recorded:?}, not the pinned {named:?}"
    );
    let mut gates = Vec::new();
    for (name, pinned) in TIERB_ALLGATES_GATES {
        let g = &row_gates[name];
        let state = g.get("state").and_then(Value::as_str);
        let got = match state {
            Some("ran") => Some(g.get("passed").and_then(Value::as_bool).ok_or_else(|| {
                format!("reference all-gates row {TIERB_ALLGATES}: gates.{name} ran with no passed")
            })?),
            _ => None,
        };
        ensure!(
            got == pinned,
            "reference all-gates row {TIERB_ALLGATES}: gates.{name} is {state:?} with passed \
             {got:?}, not the pinned {pinned:?} (ran with that passed, or not ran)"
        );
        if let Some(passed) = got {
            gates.push((name, passed));
        }
    }
    Ok(TierbReference {
        envelope,
        ft,
        ci,
        gates,
    })
}

/// (ix) The Tier-B outcome rule for one candidate: `recipe.tierb_outcome_rule` of the v5
/// pre-registration, applied as written there and as amended by Fable on 2026-10-03
/// (AUDIT/tierb-outcome-2026-10-03/fable-tierb-outcome-ruling.md). In the file's own words:
///
/// * envelope.inside: "a value is inside iff it lands neither above the envelope's max nor below
///   its min by MORE than the envelope's range (max - min): 2*min - max <= c <= 2*max - min, the
///   exact complement of the successor's clears test in both directions, compared on the rows'
///   integer counts by cross-multiplication, never the float. On these rows: choice 2325-2328 of
///   2332 (range 1), span 2046-2052 of 2053 (range 2), computed at run time. A range of 0
///   refuses."
/// * envelope.literal: "the design doc's printed decimals (choice 99.7-99.8%, span 99.8-99.9%)
///   are written to the JSON beside each value, report-only; they decide nothing"
/// * clause 1: "the training run's eval row: val_top1.choice and val_top1.span each inside the
///   envelope"
/// * clause 2: "the control row's gates.paired_margin_vs_linear 95% CI, parsed from its detail
///   string (the only place the CI is recorded; GAP-LINEAR-CONTROL-CI-ONLY-IN-DETAIL-STRING-
///   2026-10-03) in exactly the form qd_train.eval_harness.paired_margin_test writes, intersects
///   eeda5db4-0e99-415c-9a9d-5bd0efc8b71b's [+0.1359, +0.1651] (lo <= other hi and other lo <=
///   hi, on the printed four-decimal bounds as integers); a detail that does not parse exactly, a
///   printed point that disagrees with the gate's value, or a passed flag that disagrees with lo
///   > 0 refuses"
/// * clause 3: "the all-gates row: val_top1.choice and val_top1.span each inside the same
///   envelope, and every gate whose state is ran in 58fd1532-3239-41eb-a506-33ba10079425
///   (ledger/gh200-allgates-2026-09-30.jsonl; phase-3 seed 0's fp32 all-gates re-score of the
///   masked path) is ran with the same passed in the candidate's all-gates row; a gate ran there
///   and not_run in the candidate refuses. Flags only: no value comparison on needle, OOD or
///   permutation beyond the envelope."
/// * word: "pass iff clauses 1, 2 and 3 all hold; fail iff every input was read and some clause
///   does not hold; refused otherwise (exit 3). Never pass by absence."
///
/// The rows (rows.candidate, rows.identity): exactly one completed ft row tagged epoch at seed 0
/// (`Ledger::one`), not quick (`one_configuration`); its completed epoch-score-val rows
/// (`scored_from`) are exactly one without score_dtype (the training run's) and one with
/// score_dtype fp32 carrying the needle and OOD suites (the all-gates re-score), and any other
/// dtype refuses; exactly one completed tools/ft_linear_control.py row of the first
/// (`control_row_of`). The ft recipe agrees with 7f2c11db's on the eight identity keys
/// (`same_recipe_keys`; 7f2c11db's own values are checked against the pinned ones), carries the
/// candidate's key and value and not the other candidate's (`Candidate::other`); both eval rows
/// carry the pinned val_shard_hash.
///
/// How the pre-registration is checked (`tierb_agrees`): its structured fields (the conditional's
/// checker, words, on-word, rule and candidate_recipe; envelope.rows and envelope.metrics) must
/// equal this code's; its prose fields must each contain the phrases of `tierb_rule_text`, which
/// are every pinned id and hash in the field that names it, each identity key with its value, and
/// the readings above in the amendment's words. Containment, not equality, so a typo fixed
/// elsewhere in a sentence does not refuse, while any change to a pinned id, hash, bound or reading
/// does. The DRAFT is accepted (this decision is taken before the rename) and `draft` records
/// which file was read.
///
/// The control CI's printed bounds decide clause 2 as integers in ten-thousandths. At a rounding
/// edge where the printed text cannot show what the float decided (lo "+0.0000" with passed true),
/// `paired_ci` refuses rather than guess. Clause 1 is a low-power instrument at a 99.7% ceiling
/// (the pre-registration's `power`): the control CI and the gate flags carry the test.
fn rule_tierb(
    inputs: &mut Inputs,
    candidate: Candidate,
    ledger: &Path,
    phase3: &Path,
    allgates: &Path,
    preregistration: &Path,
) -> Result<(String, Value)> {
    let (p, prereg_sha256) = inputs.read_preregistration(preregistration)?;
    let draft = p.contains_key("draft");
    tierb_agrees(&p, candidate)?;
    let phase3 = inputs.read(phase3)?;
    let allgates = inputs.read(allgates)?;
    let ledger = inputs.read(ledger)?;
    let reference = tierb_reference(&phase3, &allgates)?;

    // The candidate's ft row, its identity and its own key.
    let ft = ledger.one(
        "ft",
        "epoch",
        &format!(
            "seed {TIERB_SEED} (the {} candidate's training run)",
            candidate.name()
        ),
        |r| r.seed() == Some(TIERB_SEED),
    )?;
    let ft_id = ft.id().to_string();
    one_configuration(&ledger, &[(TIERB_SEED, ft_id.clone())])?;
    let what = format!("the {} candidate's ft row {ft_id}", candidate.name());
    let identity: Vec<(&str, Value)> = tierb_identity()
        .into_iter()
        .map(|(k, v, _)| (k, v))
        .collect();
    let keys: Vec<&str> = identity.iter().map(|(k, _)| *k).collect();
    same_recipe_keys(ft, reference.ft, &keys, &what)?;
    let recipe = recipe_of(ft, &what)?;
    let (key, value) = candidate.key();
    ensure!(
        recipe.get(key) == Some(&value),
        "{what}: recipe.{key} is {}, not the {} candidate's {value} (its candidate_recipe)",
        recipe
            .get(key)
            .map_or("absent".to_string(), Value::to_string),
        candidate.name()
    );
    let (other, allowed) = candidate.other();
    if let Some(v) = recipe.get(other) {
        ensure!(
            Some(v) == allowed.as_ref(),
            "{what}: recipe.{other} is {v}, the other candidate's key; each candidate is screened \
             alone (rows.identity)"
        );
    }

    // Its two score rows, told apart by score_dtype, and its linear control.
    let scored = |r: &Row| {
        r.completed()
            && r.str_at(&["run_kind"]) == Some("eval")
            && r.tag() == Some(EVAL_TAG)
            && scored_from(r, TIERB_SEED, &ft_id)
    };
    for r in ledger.rows.iter().filter(|r| scored(r)) {
        let dtype = r.get(&["recipe", "score_dtype"]);
        ensure!(
            dtype.is_none() || dtype == Some(&json!("fp32")),
            "row {} scored from ft row {ft_id} has score_dtype {}: neither the training run's \
             --score-val (no score_dtype) nor the all-gates re-score (fp32); which one decides is \
             not this tool's call",
            r.id(),
            dtype.map_or("absent".to_string(), Value::to_string)
        );
    }
    let train_eval = ledger.one(
        "eval",
        EVAL_TAG,
        &format!(
            "seed {TIERB_SEED} scored from ft row {ft_id} without score_dtype (the training \
             run's --score-val)"
        ),
        |r| scored_from(r, TIERB_SEED, &ft_id) && r.get(&["recipe", "score_dtype"]).is_none(),
    )?;
    let all_eval = ledger.one(
        "eval",
        EVAL_TAG,
        &format!(
            "seed {TIERB_SEED} scored from ft row {ft_id} with score_dtype fp32 (the all-gates \
             re-score)"
        ),
        |r| {
            scored_from(r, TIERB_SEED, &ft_id)
                && r.str_at(&["recipe", "score_dtype"]) == Some("fp32")
        },
    )?;
    for suite in ["needle", "ood"] {
        ensure!(
            all_eval
                .get(&["recipe", suite])
                .is_some_and(Value::is_object),
            "row {}: the fp32 re-score of ft row {ft_id} has no recipe.{suite} suite, so it is not \
             the all-gates re-score (rows.candidate)",
            all_eval.id()
        );
    }
    for r in [train_eval, all_eval] {
        same_recipe_keys(
            r,
            reference.envelope[0],
            &["val_shard_hash"],
            "rows.identity: the candidate's eval rows are scored on phase-3's val set",
        )?;
    }
    let control = control_row_of(
        &ledger,
        train_eval,
        "linear-control",
        &format!("written by {LINEAR_CONTROL_TOOL}"),
        |r| r.str_at(&["recipe", "tool"]) == Some(LINEAR_CONTROL_TOOL),
    )?;
    let ci = paired_ci(control)?;

    // Every input is read; the clauses.
    let envelope: Vec<TierbEnvelope> = TIERB_METRICS
        .iter()
        .map(|m| TierbEnvelope::of(*m, &reference.envelope))
        .collect::<Result<_>>()?;
    let mut c1 = Map::new();
    let mut c3 = Map::new();
    let (mut c1_holds, mut c3_holds) = (true, true);
    for e in &envelope {
        let (holds, j) = e.judge(train_eval)?;
        c1_holds &= holds;
        c1.insert(e.metric.name.to_string(), j);
        let (holds, j) = e.judge(all_eval)?;
        c3_holds &= holds;
        c3.insert(e.metric.name.to_string(), j);
    }
    c1.insert("holds".into(), json!(c1_holds));
    let c2_holds = ci.overlaps(&reference.ci);
    let mut gates = Vec::new();
    for (name, want) in &reference.gates {
        let g = all_eval.ran("gates", name).map_err(|e| {
            format!(
                "clause 3: gate {name} ran in {TIERB_ALLGATES} but not in the candidate's \
                 all-gates row {}, which refuses: {e}",
                all_eval.id()
            )
        })?;
        let got = g.get("passed").and_then(Value::as_bool).ok_or_else(|| {
            format!(
                "clause 3: row {} gates.{name} ran with no boolean passed",
                all_eval.id()
            )
        })?;
        c3_holds &= got == *want;
        gates.push(json!({"gate": name, "reference_passed": want, "candidate_passed": got, "same": got == *want}));
    }
    c3.insert("gates".into(), json!(gates));
    c3.insert("holds".into(), json!(c3_holds));

    let pass = c1_holds && c2_holds && c3_holds;
    let body = json!({
        "candidate": candidate.name(),
        "candidate_recipe": candidate.recipe(),
        "preregistration": {"sha256": prereg_sha256, "draft": draft},
        "rule": "pass iff clauses 1, 2 and 3 all hold; fail iff every input was read and some \
                 clause does not hold; refused otherwise (recipe.tierb_outcome_rule.word)",
        "rows": {
            "ft": ft_id,
            "eval": train_eval.id(),
            "allgates": all_eval.id(),
            "control": control.id(),
            "reference": {
                "envelope": TIERB_ENVELOPE,
                "identity_ft": TIERB_FT,
                "control": TIERB_CONTROL,
                "allgates": TIERB_ALLGATES,
            },
        },
        "identity": identity.into_iter().map(|(k, v)| (k.to_string(), v)).collect::<Map<_, _>>(),
        "envelope": {
            "rows": TIERB_ENVELOPE,
            "ledger": phase3.shown,
            "metrics": envelope
                .iter()
                .map(|e| (e.metric.name.to_string(), e.json.clone()))
                .collect::<Map<_, _>>(),
            "inside": "2*min - max <= c <= 2*max - min: neither clears(c, min, max, higher) nor \
                       clears(c, min, max, lower), on the integer counts",
        },
        "clauses": {
            "1": Value::Object(c1),
            "2": {
                "holds": c2_holds,
                "candidate_ci": ci.json(),
                "reference_ci": reference.ci.json(),
                "rule": "candidate lo <= reference hi and reference lo <= candidate hi, in \
                         ten-thousandths",
            },
            "3": Value::Object(c3),
        },
        "power": "clause 1 is a low-power instrument at a 99.7% ceiling: the amendment makes it \
                  satisfiable, not sharp; the control CI and the gate flags carry the test",
        "then": if pass {
            "pass: the candidate's flag is on for v5 as its conditional says (C2a also needs both \
             P2 shapes' latest verdict pass)"
        } else {
            "fail: the candidate's flag stays off for v5"
        },
    });
    let word = if pass { TIERB_WORDS[0] } else { TIERB_WORDS[1] };
    Ok((word.to_string(), body))
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
        Cmd::V5Pause { .. } => "v5_pause_after_seed_0",
        Cmd::V5Noulw { room: true, .. } => "v5_noul_weight_room",
        Cmd::V5Noulw { room: false, .. } => "v5_noul_weight",
        Cmd::Tierb { .. } => "tierb_outcome",
        Cmd::FtRows { .. } => "ft_rows",
        Cmd::EvalRow { .. } => "eval_row",
    }
}

fn preregistration(cmd: &Cmd) -> &'static str {
    match cmd {
        Cmd::Successor { .. } => PREREG_SUCC,
        Cmd::J6a { .. } => PREREG_J6A,
        Cmd::J6g { .. } => PREREG_J6G,
        Cmd::Seeds34 {
            preregistration: Some(_),
            ..
        }
        | Cmd::V5Pause { .. }
        | Cmd::V5Noulw { .. } => PREREG_V5,
        Cmd::Tierb { .. } => PREREG_TIERB,
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
            f_ledger,
            ft_rows,
            preregistration,
            ..
        } => rule_seeds34(&mut inputs, f_ledger, ft_rows, preregistration.as_deref()),
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
        Cmd::V5Pause {
            preregistration,
            ledger,
            ft_row,
            ..
        } => rule_v5_pause(&mut inputs, preregistration, ledger, ft_row),
        Cmd::V5Noulw {
            preregistration,
            noul_preregistration,
            v5_ledger,
            ft_rows,
            room,
            arm_ledger,
            arm_ft_rows,
            ..
        } => match (*room, arm_ledger) {
            (true, _) => rule_v5_noulw(
                &mut inputs,
                preregistration,
                noul_preregistration,
                v5_ledger,
                ft_rows,
                None,
            ),
            (false, Some(arm)) => rule_v5_noulw(
                &mut inputs,
                preregistration,
                noul_preregistration,
                v5_ledger,
                ft_rows,
                Some((arm, arm_ft_rows)),
            ),
            (false, None) => Err("v5-noulw reads the arm's rows from --arm-ledger, or with \
                                  --room reads none; neither was given"
                .to_string()),
        },
        Cmd::Tierb {
            candidate,
            ledger,
            phase3_ledger,
            allgates_ledger,
            preregistration,
            ..
        } => rule_tierb(
            &mut inputs,
            *candidate,
            ledger,
            phase3_ledger,
            allgates_ledger,
            preregistration,
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
        | Cmd::J6g { out, .. }
        | Cmd::V5Pause { out, .. }
        | Cmd::V5Noulw { out, .. }
        | Cmd::Tierb { out, .. } => Some(out),
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
            preregistration: None,
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
                preregistration: None,
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
            preregistration: None,
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

    // --- (ix): tierb's CI parser ------------------------------------------------------------

    const TIERB_P3: &str = "ledger/gh200-seed0-weights-2026-09-30.jsonl";
    const EEDA_DETAIL: &str =
        "paired margin +0.1501, 95% CI [+0.1359, +0.1651] over 10000 bootstrap resamples";

    fn as_row(v: Value) -> Row {
        let Value::Object(map) = v else {
            panic!("not an object")
        };
        Row {
            at: "test".into(),
            map,
        }
    }

    fn with_gate(detail: &str, value: f64, passed: bool) -> Row {
        let mut v = real_row(TIERB_P3, "eeda5db4");
        let g = &mut v["gates"][LINEAR_GATE];
        g["detail"] = json!(detail);
        g["value"] = json!(value);
        g["passed"] = json!(passed);
        as_row(v)
    }

    #[test]
    fn eeda5db4s_exact_detail_parses_to_its_printed_integers() {
        let row = as_row(real_row(TIERB_P3, "eeda5db4"));
        assert_eq!(
            row.str_at(&["gates", LINEAR_GATE, "detail"]),
            Some(EEDA_DETAIL)
        );
        let ci = paired_ci(&row).unwrap();
        assert_eq!(ci.point.ten_thousandths, 1501);
        assert_eq!(
            (ci.lo.ten_thousandths, ci.hi.ten_thousandths),
            TIERB_CONTROL_CI
        );
        assert_eq!(ci.n_boot, CI_N_BOOT);
        assert!(ci.passed);
        assert!(ci.overlaps(&ci));
    }

    #[test]
    fn each_suffix_form_renders_byte_for_byte_and_round_trips() {
        let f = |s: &str| Fixed4::parse(s).unwrap();
        // passed: no suffix.
        assert_eq!(
            ci_detail(f("+0.1501"), f("+0.1359"), f("+0.1651"), 10_000, true),
            EEDA_DETAIL
        );
        // Straddling zero, and wholly below it: the harness's two sentences.
        let straddles = ci_detail(f("+0.0010"), f("-0.0050"), f("+0.0070"), 10_000, false);
        assert!(straddles.ends_with(CI_INCLUDES_ZERO), "{straddles}");
        let below = ci_detail(f("-0.0100"), f("-0.0200"), f("-0.0050"), 10_000, false);
        assert!(
            below.ends_with(
                "the baseline beats the model by 0.0100 and the comparison separates them"
            ),
            "{below}"
        );
        for (detail, value) in [(straddles, 0.001), (below, -0.01)] {
            let ci = paired_ci(&with_gate(&detail, value, false)).unwrap();
            assert!(!ci.passed);
        }
        // "-0.0000" keeps its sign: Python prints it for a negative hi that rounds to zero, and
        // takes the below-zero branch for it.
        let z = f("-0.0000");
        assert!(z.negative && z.ten_thousandths == 0 && z.text() == "-0.0000");
        assert_eq!(f("+0.0000").negated_plain(), "-0.0000");
        assert_eq!(f("-0.0100").negated_plain(), "0.0100");
    }

    #[test]
    fn a_number_off_the_four_decimal_form_does_not_parse() {
        for bad in [
            "0.1501", "+.1501", "+0.150", "+0.15010", "+1e-4", "+0.15a1", "", "+",
        ] {
            assert_eq!(Fixed4::parse(bad), None, "{bad}");
        }
        assert_eq!(
            Fixed4::parse("+1.0000").map(|x| x.ten_thousandths),
            Some(10_000)
        );
    }

    #[test]
    fn a_ci_at_a_rounding_edge_or_off_its_flag_refuses() {
        // lo printed +0.0000 with passed true: the text cannot show lo > 0.
        let edge =
            "paired margin +0.0100, 95% CI [+0.0000, +0.0200] over 10000 bootstrap resamples";
        let e = paired_ci(&with_gate(edge, 0.01, true)).unwrap_err();
        assert!(e.contains("passes iff lo > 0"), "{e}");
        // A trailing space is a byte off the form.
        let e = paired_ci(&with_gate(
            &format!("{EEDA_DETAIL} "),
            0.150_085_763_293_310_47,
            true,
        ))
        .unwrap_err();
        assert!(
            e.contains("not in the form paired_margin_test writes"),
            "{e}"
        );
    }

    #[test]
    fn inside_is_the_complement_of_clears_at_both_ends() {
        let v = |k: i64| Val::Exact(Rat { num: k, den: 2332 });
        let (min, max) = (v(2326), v(2327));
        let got: Vec<i64> = (2320..=2332)
            .filter(|&k| inside(v(k), min, max).unwrap())
            .collect();
        assert_eq!(got, vec![2325, 2326, 2327, 2328]);
    }
}
