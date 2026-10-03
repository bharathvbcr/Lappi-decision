//! `qd-post-f-rules`' v5 readings, run the way the box runs them: the binary, the one word it
//! prints, its exit code and its `--out` JSON.
//!
//! * `v5-pause` — R9 (`readings.R9_pause_after_seed_0`).
//! * `v5-noulw --room` and `v5-noulw` — the noul-weight arm (`arm_noul_weight`).
//! * `seeds34` and `eval-row` on a v5-shaped ledger, and `seeds34` on F's real rows pinned to the
//!   decision the unmodified binary wrote (AUDIT/v5-rules-2026-10-02/characterization-*).
//!
//! The pre-registration is the real `campaign/v5-preregistered.DRAFT.json` with its top-level
//! `draft` key removed, which is what the rename to `campaign/v5-preregistered.json` leaves; the
//! DRAFT itself must refuse. F2 and F3 are the real `campaign/v4-noul-v3b-preregistered.json`.
//! Ledger rows are F's real rows (`ledger/gh200-p4-v4-2026-10-01.jsonl`: ft 973cd4e3, eval
//! f4feac15) re-identified as v5's, with the counts each test sets.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const BIN: &str = env!("CARGO_BIN_EXE_qd-post-f-rules");
const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const F_LEDGER: &str = "ledger/gh200-p4-v4-2026-10-01.jsonl";
const F_FT: [&str; 3] = [
    "973cd4e3-e0d2-4ff8-8588-b761cb842b75",
    "95fa4854-6146-4824-b7ad-22bb54744ff1",
    "32990e1a-0c61-49fa-9217-800eb845f27e",
];
const DRAFT: &str = "campaign/v5-preregistered.DRAFT.json";
const NOUL: &str = "campaign/v4-noul-v3b-preregistered.json";
const EXIT_REFUSED: i32 = 3;
/// The needle suite's depth buckets and F seed 0's bucket sizes (f4feac15).
const DEPTH: [&str; 5] = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"];
const SIZES: [u64; 5] = [59, 61, 59, 60, 61];
const V5_COMMIT: &str = "5555555555555555555555555555555555555555";

fn repo(rel: &str) -> PathBuf {
    Path::new(REPO).join(rel)
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A directory under the system temp dir, removed on drop.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

static N: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> Scratch {
    let n = N.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("qd-post-f-rules-v5-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

impl Scratch {
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn json(&self, name: &str, value: &Value) -> PathBuf {
        let path = self.path(name);
        std::fs::write(&path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        path
    }
    fn ledger(&self, name: &str, rows: &[Value]) -> PathBuf {
        let path = self.path(name);
        let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
        std::fs::write(&path, text).unwrap();
        path
    }
}

struct Run {
    word: String,
    code: i32,
    json: Value,
    stderr: String,
}

impl Run {
    fn refused_with(&self, phrase: &str) {
        assert_eq!(self.word, "refused", "{}", self.stderr);
        assert_eq!(self.code, EXIT_REFUSED, "{}", self.stderr);
        let reason = self.json["refused"].as_str().unwrap_or_default();
        assert!(reason.contains(phrase), "{phrase:?} not in {reason}");
    }
    fn said(&self, word: &str) {
        assert_eq!(self.word, word, "{}", self.stderr);
        assert_eq!(self.code, 0, "{}", self.stderr);
    }
}

/// The binary with `args` from `cwd`, its JSON written to a new file in `s`.
fn run_in(s: &Scratch, cwd: &Path, args: &[String]) -> Run {
    let out = s.path(&format!("out-{}.json", N.fetch_add(1, Ordering::SeqCst)));
    let o = Command::new(BIN)
        .args(args)
        .arg("--out")
        .arg(&out)
        .current_dir(cwd)
        .output()
        .unwrap();
    Run {
        word: String::from_utf8(o.stdout).unwrap().trim_end().to_string(),
        code: o.status.code().unwrap_or(-1),
        json: std::fs::read(&out)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Value::Null),
        stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn run(s: &Scratch, args: &[String]) -> Run {
    run_in(s, &s.0, args)
}

fn args(parts: &[&dyn AsRef<std::ffi::OsStr>]) -> Vec<String> {
    parts
        .iter()
        .map(|p| p.as_ref().to_str().unwrap().to_string())
        .collect()
}

fn ft_args(flag: &str, ids: &[String]) -> Vec<String> {
    ids.iter()
        .enumerate()
        .flat_map(|(s, id)| [flag.to_string(), format!("{s}={id}")])
        .collect()
}

// --- rows ----------------------------------------------------------------------------------------

fn f_row(prefix: &str) -> Value {
    std::fs::read_to_string(repo(F_LEDGER))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|r| r["row_id"].as_str().unwrap().starts_with(prefix))
        .unwrap()
}

fn rid(kind: u64, n: u64) -> String {
    format!("{kind:08x}-0000-4000-8000-{n:012x}")
}

/// One seed's counts. Needle: (depth bucket index, hits); the other buckets are full.
#[derive(Clone, Copy, Debug)]
struct P {
    span: (u64, u64),
    choice: (u64, u64),
    needle: (usize, u64),
    prose: u64,
    unseen: u64,
    scrambled: u64,
    perm: u64,
    ece: f64,
    id_dc: (u64, u64),
    id_pooled: (u64, u64),
}

/// F seed 0's counts but prose and unseen-language, which each test sets.
const BASE: P = P {
    span: (6529, 7238),
    choice: (9130, 10985),
    needle: (4, 40),
    prose: 54,
    unseen: 20,
    scrambled: 58,
    perm: 2291,
    ece: 0.020_7,
    id_dc: (13, 2304),
    id_pooled: (843, 10985),
};

fn set(row: &mut Value, key: &str, (k, n): (u64, u64)) {
    let e = &mut row["metrics"][key];
    assert!(e.is_object(), "{key} is not on F's eval row");
    e["n"] = json!(k);
    e["n_total"] = json!(n);
    e["value"] = json!(k as f64 / n as f64);
}

fn v5_data() -> String {
    "d5".repeat(32)
}

fn v5_shard() -> String {
    "5d".repeat(32)
}

/// The batch order v5 seed `seed` trains on (`--batch-order seed`): its plan's digest.
fn plan_order_digest(seed: i64) -> String {
    format!("{:02x}", 0xb0 + seed).repeat(32)
}

const PAIRED: &str = "corpus.plan_order_digest";

/// A v5 ft row (`arm` adds the noul-weight arm's two keys). Every v5 run trains with
/// `--batch-order seed` (Fable's seed-order ruling): the recipe key is the constant string, and
/// the ft row records seed `seed`'s plan seed and order digest.
fn ft(id: &str, seed: i64, arm: bool) -> Value {
    let mut r = f_row("973cd4e3");
    r["row_id"] = json!(id);
    r["code_commit"] = json!(V5_COMMIT);
    r["protocol"]["seed"] = json!(seed);
    r["protocol"]["data_snapshot_hash"] = json!(v5_data());
    r["protocol"]["recipe_hash"] = json!(if arm { "a5" } else { "55" }.repeat(32));
    r["recipe"]["shard_hash"] = json!(v5_shard());
    r["recipe"]["min_lr"] = json!(0.0);
    r["recipe"]["batch_order"] = json!("seed");
    r["metrics"]["corpus.plan_seed"] = json!({"state": "ran", "value": seed});
    r["metrics"][PAIRED] = json!({"state": "ran", "value": plan_order_digest(seed)});
    if arm {
        r["recipe"]["noul_weight"] = json!(4.0);
        r["recipe"]["noul_weight_scope"] = json!("code.defect_class");
    }
    r
}

/// A v5 epoch-score-val row of `seed` scored from ft row `ft`, with `p`'s counts.
fn eval(id: &str, seed: i64, ft: &str, p: &P) -> Value {
    let mut r = f_row("f4feac15");
    r["row_id"] = json!(id);
    r["code_commit"] = json!(V5_COMMIT);
    r["protocol"]["seed"] = json!(seed);
    r["protocol"]["data_snapshot_hash"] = json!(v5_data());
    r["recipe"]["shard_hash"] = json!(v5_shard());
    r["metrics"]["ft_run_row_id"]["value"] = json!(ft);
    set(&mut r, "val_top1.span", p.span);
    set(&mut r, "val_top1.choice", p.choice);
    set(&mut r, "ood_abstain.prose", (p.prose, 60));
    set(&mut r, "ood_abstain.unseen-language", (p.unseen, 60));
    set(&mut r, "ood_abstain.scrambled", (p.scrambled, 60));
    set(
        &mut r,
        "permutation_consistency.family.code.defect_class",
        (p.perm, 2304),
    );
    set(
        &mut r,
        "ood_abstain.in_distribution.family.code.defect_class",
        p.id_dc,
    );
    set(&mut r, "ood_abstain.in_distribution", p.id_pooled);
    r["metrics"]["ece.family.code.defect_class.choice.k4"]["value"] = json!(p.ece);
    let mut worst = (u64::MAX, 1u64);
    for (i, label) in DEPTH.iter().enumerate() {
        let n = SIZES[i];
        let k = if i == p.needle.0 { p.needle.1 } else { n };
        if u128::from(k) * u128::from(worst.1) < u128::from(worst.0) * u128::from(n) {
            worst = (k, n);
        }
        set(&mut r, &format!("needle_hunk_recall.depth.{label}"), (k, n));
    }
    r["gates"]["needle_hunk_recall"]["value"] = json!(worst.0 as f64 / worst.1 as f64);
    r
}

/// Three seeds' ft and eval rows; `kind` keeps ids distinct per run. Returns the rows and the ft
/// ids in seed order.
fn seeds(kind: u64, arm: bool, profiles: [P; 3]) -> (Vec<Value>, Vec<String>) {
    let mut rows = Vec::new();
    let mut ids = Vec::new();
    for (s, p) in profiles.iter().enumerate() {
        let (f, e) = (rid(kind, 2 * s as u64 + 1), rid(kind, 2 * s as u64 + 2));
        rows.push(ft(&f, s as i64, arm));
        rows.push(eval(&e, s as i64, &f, p));
        ids.push(f);
    }
    (rows, ids)
}

fn row_mut<'a>(rows: &'a mut [Value], id: &str) -> &'a mut Value {
    rows.iter_mut().find(|r| r["row_id"] == id).unwrap()
}

/// The eval row scored from ft row `ft`.
fn eval_of<'a>(rows: &'a mut [Value], ft: &str) -> &'a mut Value {
    rows.iter_mut()
        .find(|r| r["metrics"]["ft_run_row_id"]["value"] == ft)
        .unwrap()
}

// --- the pre-registration ------------------------------------------------------------------------

/// The real DRAFT as the rename leaves it: no `draft` key; then `edit`.
fn prereg(s: &Scratch, edit: impl FnOnce(&mut Map<String, Value>)) -> PathBuf {
    let mut p: Map<String, Value> =
        serde_json::from_str(&std::fs::read_to_string(repo(DRAFT)).unwrap()).unwrap();
    assert!(p.remove("draft").is_some(), "the DRAFT has no draft key");
    edit(&mut p);
    s.json(
        &format!("prereg-{}.json", N.fetch_add(1, Ordering::SeqCst)),
        &Value::Object(p),
    )
}

fn bound(s: &Scratch) -> PathBuf {
    prereg(s, |_| {})
}

/// Replace `from` (which must be there) with `to` in the string at `path`.
fn retext(p: &mut Map<String, Value>, path: &[&str], from: &str, to: &str) {
    let mut cur = p.get_mut(path[0]).unwrap();
    for k in &path[1..] {
        cur = cur.get_mut(k).unwrap();
    }
    let text = cur.as_str().unwrap();
    assert!(text.contains(from), "{from:?} not in {text}");
    *cur = json!(text.replace(from, to));
}

fn noul(s: &Scratch, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let mut q: Value = serde_json::from_str(&std::fs::read_to_string(repo(NOUL)).unwrap()).unwrap();
    edit(&mut q);
    s.json(
        &format!("noul-{}.json", N.fetch_add(1, Ordering::SeqCst)),
        &q,
    )
}

// --- characterization: F's seeds34 and eval-row, pinned before any change ----------------------

/// Captured from the unmodified binary (source sha256 739355ba..., main a3c09de) on F's real
/// rows, cwd = the repository and the ledger given relatively, so the JSON names the same path:
/// AUDIT/v5-rules-2026-10-02/characterization-seeds34-f-pre-change.json.
const F_SEEDS34_JSON_SHA256: &str =
    "399c4e9205586205252b1dbec575dd0f3c2f17d9787572b3c96faaf7b499aafa";
/// The same binary's `eval-row` stderr (its JSON) for F seed 1:
/// AUDIT/v5-rules-2026-10-02/characterization-eval-row-f-pre-change.stderr.
const F_EVAL_ROW_STDERR_SHA256: &str =
    "a6fbd508b941495a57debe3f35bd7b7cfa4782f66264214efe2294032e83e6bb";
/// Both captures record F's ledger as it stood then: its first 15 rows, 195,429 bytes. The ledger
/// is append-only and has grown since (F seed 3's rows, f320a33), so the captures are replayed on
/// that prefix; a rewrite of any of those rows fails `f_ledger_as_captured` instead of the pins.
const F_LEDGER_CAPTURED_ROWS: usize = 15;
const F_LEDGER_CAPTURED_SHA256: &str =
    "14ec1d56b7c71ec0713f134ccc317aaf035391e543b93f2e2f398053f4163edf";

/// A stand-in for the repository holding only F's ledger, cut to the rows the captures saw and
/// at the same relative path, so the binary (which reads nothing else for `seeds34` without
/// `--preregistration`, or for `eval-row`) names the same path and records the same bytes.
fn f_ledger_as_captured() -> Scratch {
    let text = std::fs::read(repo(F_LEDGER)).unwrap();
    let end = text
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == b'\n')
        .nth(F_LEDGER_CAPTURED_ROWS - 1)
        .map(|(i, _)| i + 1)
        .unwrap_or_else(|| panic!("{F_LEDGER} has fewer than {F_LEDGER_CAPTURED_ROWS} rows"));
    assert_eq!(
        sha256(&text[..end]),
        F_LEDGER_CAPTURED_SHA256,
        "{F_LEDGER}'s first {F_LEDGER_CAPTURED_ROWS} rows are not the ones captured: the ledger \
         is append-only"
    );
    let s = scratch();
    let path = s.path(F_LEDGER);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &text[..end]).unwrap();
    s
}

#[test]
fn seeds34_on_fs_real_rows_is_the_unmodified_binarys_decision_byte_for_byte() {
    let root = f_ledger_as_captured();
    let s = scratch();
    let out = s.path("s34.json");
    let mut a = args(&[&"seeds34", &"--f-ledger", &F_LEDGER]);
    a.extend(ft_args("--ft-row", &F_FT.map(String::from)));
    a.extend(args(&[&"--out", &out]));
    let o = Command::new(BIN)
        .args(&a)
        .current_dir(&root.0)
        .output()
        .unwrap();
    assert_eq!(String::from_utf8(o.stdout).unwrap(), "fires\n");
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(sha256(&std::fs::read(&out).unwrap()), F_SEEDS34_JSON_SHA256);
}

#[test]
fn eval_row_on_fs_real_rows_is_the_unmodified_binarys_answer_byte_for_byte() {
    let root = f_ledger_as_captured();
    let o = Command::new(BIN)
        .args(["eval-row", "--ledger", F_LEDGER, "--ft-row"])
        .arg(format!("1={}", F_FT[1]))
        .current_dir(&root.0)
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8(o.stdout).unwrap(),
        "aeca8d69-4733-4593-92de-2073021ed684\n"
    );
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(sha256(&o.stderr), F_EVAL_ROW_STDERR_SHA256);
}

// --- seeds34 and eval-row on v5's ledger ----------------------------------------------------------

/// v5's three seeds with these 8K worst buckets in the 60-80% bucket (of 60).
fn v5_needles(hits: [u64; 3]) -> [P; 3] {
    hits.map(|h| P {
        needle: (3, h),
        ..BASE
    })
}

fn seeds34(s: &Scratch, ledger: &Path, ids: &[String], prereg: Option<&Path>) -> Run {
    let mut a = args(&[&"seeds34", &"--f-ledger", &ledger]);
    a.extend(ft_args("--ft-row", ids));
    if let Some(p) = prereg {
        a.extend(args(&[&"--preregistration", &p]));
    }
    run(s, &a)
}

#[test]
fn seeds34_reads_v5s_ledger_at_the_0_30_boundary_with_and_without_v5s_preregistration() {
    let s = scratch();
    let p = bound(&s);
    // 54/60 - 36/60 = 0.30 exactly: not more than 0.30. One hit fewer: more.
    for (low, word) in [(36, "quiet"), (35, "fires")] {
        let (rows, ids) = seeds(0x5, false, v5_needles([54, 50, low]));
        let ledger = s.ledger(&format!("gh200-v5-2026-10-04-{low}.jsonl"), &rows);
        let f_form = seeds34(&s, &ledger, &ids, None);
        f_form.said(word);
        assert_eq!(
            f_form.json["preregistration"],
            "campaign/f-j7prime-preregistered.json (54512e6)"
        );
        let v5_form = seeds34(&s, &ledger, &ids, Some(&p));
        v5_form.said(word);
        let d = &v5_form.json["detail"];
        assert!(
            v5_form.json["preregistration"]
                .as_str()
                .unwrap()
                .starts_with("campaign/v5-preregistered.json")
        );
        assert_eq!(
            d["preregistration_sha256"],
            sha256(&std::fs::read(&p).unwrap())
        );
        assert_eq!(d["min"]["n"], low);
        assert_eq!(
            d["then"],
            if word == "fires" {
                "seeds 3 and 4 run in v5's form before J5'"
            } else {
                "no seeds 3-4 (seeds.seeds_3_4)"
            }
        );
        // Both forms read the same rows the same way.
        assert_eq!(f_form.json["detail"]["seeds"], d["seeds"]);
    }
}

#[test]
fn seeds34_under_v5s_preregistration_refuses_a_quick_or_shuffled_seed_fs_form_never_checked() {
    let s = scratch();
    let p = bound(&s);
    let edits: [fn(&mut Value); 2] = [
        |r| r["quick"] = json!(true),
        |r| r["recipe"]["shuffled_label"] = json!("e1000000-0000-4000-8000-000000000000"),
    ];
    for edit in edits {
        let (mut rows, ids) = seeds(0x5, false, v5_needles([54, 50, 40]));
        edit(row_mut(&mut rows, &ids[1]));
        let ledger = s.ledger("gh200-v5.jsonl", &rows);
        // F's form never checked quick or shuffled_label (its seeds were pinned by F's logs).
        seeds34(&s, &ledger, &ids, None).said("quiet");
        let v5_form = seeds34(&s, &ledger, &ids, Some(&p));
        v5_form.refused_with(&ids[1]);
    }
}

#[test]
fn seeds34_refuses_the_draft_and_a_threshold_that_is_not_0_30() {
    let s = scratch();
    let (rows, ids) = seeds(0x5, false, v5_needles([54, 50, 40]));
    let ledger = s.ledger("gh200-v5.jsonl", &rows);
    seeds34(&s, &ledger, &ids, Some(&repo(DRAFT))).refused_with("\"draft\" key");
    let p = prereg(&s, |p| {
        retext(
            p,
            &["seeds", "seeds_3_4"],
            "more than 0.30",
            "more than 0.25",
        );
    });
    seeds34(&s, &ledger, &ids, Some(&p)).refused_with("0.25 is not this rule's 0.30");
    // Seeds 1, 2, 3 are not seeds.v5's 0, 1, 2.
    let mut a = args(&[
        &"seeds34",
        &"--f-ledger",
        &ledger,
        &"--preregistration",
        &bound(&s),
    ]);
    for (i, id) in ids.iter().enumerate() {
        a.extend([String::from("--ft-row"), format!("{}={id}", i + 1)]);
    }
    run(&s, &a).refused_with("seeds.v5 names seeds [0, 1, 2]");
}

#[test]
fn eval_row_finds_a_v5_seeds_epoch_score_val_row() {
    let s = scratch();
    let (rows, ids) = seeds(0x5, false, v5_needles([54, 50, 40]));
    let ledger = s.ledger("gh200-v5-2026-10-04.jsonl", &rows);
    let o = Command::new(BIN)
        .args(["eval-row", "--ledger"])
        .arg(&ledger)
        .arg("--ft-row")
        .arg(format!("1={}", ids[1]))
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(String::from_utf8(o.stdout).unwrap().trim(), rid(0x5, 4));
}

// --- v5-pause (R9) --------------------------------------------------------------------------------

fn pause(s: &Scratch, prereg: &Path, ledger: &Path, ft: &str) -> Run {
    run(
        s,
        &args(&[
            &"v5-pause",
            &"--preregistration",
            &prereg,
            &"--ledger",
            &ledger,
            &"--ft-row",
            &format!("0={ft}"),
        ]),
    )
}

/// v5 seed 0's ft and eval rows with `p`'s counts.
fn seed0(s: &Scratch, p: P) -> (PathBuf, String) {
    let (rows, ids) = seeds(0x50, false, [p, BASE, BASE]);
    (s.ledger("gh200-v5-2026-10-04.jsonl", &rows), ids[0].clone())
}

#[test]
fn r9_continues_at_exactly_5n_eq_4n_total_and_pauses_one_count_below() {
    let s = scratch();
    let p = bound(&s);
    // 5 * 5792 = 4 * 7240.
    for (span, word) in [(5792, "continue"), (5791, "pause")] {
        let (ledger, ft0) = seed0(
            &s,
            P {
                span: (span, 7240),
                needle: (3, 30),
                ..BASE
            },
        );
        let o = pause(&s, &p, &ledger, &ft0);
        o.said(word);
        let d = &o.json["detail"];
        assert_eq!(d["val_top1.span"]["form"], "5*n >= 4*n_total");
        assert_eq!(d["val_top1.span"]["meets"], word == "continue");
        assert_eq!(d["needle_8k_worst"]["meets"], true);
        assert_eq!(d["kind"], "a spending rule, not a gate (R9)");
    }
}

#[test]
fn r9_continues_at_exactly_2n_eq_n_total_and_pauses_one_hit_below_in_either_bucket_size() {
    let s = scratch();
    let p = bound(&s);
    // 60-80% holds 60 cases, 80-100% holds 61: 2 * 30 = 60; 2 * 31 >= 61 > 2 * 30.
    for (needle, word) in [
        ((3, 30), "continue"),
        ((3, 29), "pause"),
        ((4, 31), "continue"),
        ((4, 30), "pause"),
    ] {
        let (ledger, ft0) = seed0(
            &s,
            P {
                span: (5792, 7240),
                needle,
                ..BASE
            },
        );
        let o = pause(&s, &p, &ledger, &ft0);
        o.said(word);
        assert_eq!(
            o.json["detail"]["needle_8k_worst"]["form"],
            "2*n >= 1*n_total"
        );
        assert_eq!(o.json["detail"]["val_top1.span"]["meets"], true);
    }
}

#[test]
fn r9s_thresholds_are_read_from_the_preregistration_not_from_this_code() {
    let s = scratch();
    let (ledger, ft0) = seed0(
        &s,
        P {
            span: (5792, 7240),
            needle: (3, 30),
            ..BASE
        },
    );
    pause(&s, &bound(&s), &ledger, &ft0).said("continue");
    let stricter = prereg(&s, |p| {
        retext(
            p,
            &["readings", "R9_pause_after_seed_0"],
            "5*n >= 4*n_total",
            "9*n >= 8*n_total",
        );
    });
    let o = pause(&s, &stricter, &ledger, &ft0);
    o.said("pause");
    assert_eq!(
        o.json["detail"]["val_top1.span"]["form"],
        "9*n >= 8*n_total"
    );
}

#[test]
fn r9_refuses_a_preregistration_whose_words_it_does_not_apply() {
    let s = scratch();
    let (ledger, ft0) = seed0(
        &s,
        P {
            span: (5792, 7240),
            needle: (3, 30),
            ..BASE
        },
    );
    pause(&s, &repo(DRAFT), &ledger, &ft0).refused_with("\"draft\" key");
    let r9 = ["readings", "R9_pause_after_seed_0"];
    for (from, to, phrase) in [
        ("5*n >= 4*n_total", "at least 0.80", "is not a count form"),
        (
            "5*n >= 4*n_total",
            "5*n <= 4*n_total",
            "are not both floors",
        ),
        (
            "seed 0's epoch-score-val row",
            "seed 0's epoch-needle-length-control row",
            "R9 reads the",
        ),
        ("prints continue iff", "prints go iff", "does not say"),
        ("otherwise pause", "otherwise hold", "does not say"),
        (
            "after v5 seed 0's",
            "after v5 seed 7's",
            "seed 7 is not one of seeds.v5's",
        ),
        (
            "8K needle worst bucket has",
            "8K needle bucket has",
            "occurs 0 time(s)",
        ),
    ] {
        let p = prereg(&s, |p| retext(p, &r9, from, to));
        let o = pause(&s, &p, &ledger, &ft0);
        o.refused_with(phrase);
        assert!(
            o.json["inputs"].as_array().unwrap().len() == 1,
            "a ledger was read: {}",
            o.json
        );
    }
}

#[test]
fn r9_refuses_missing_doubled_unfinished_or_malformed_rows_and_never_prints_a_word() {
    let s = scratch();
    let p = bound(&s);
    let good = P {
        span: (5792, 7240),
        needle: (3, 30),
        ..BASE
    };
    let (rows, ids) = seeds(0x50, false, [good, BASE, BASE]);
    let ft0 = ids[0].clone();
    let ev0 = rid(0x50, 2);
    type Edit = fn(&mut Vec<Value>, &str, &str);
    let cases: [(Edit, &str); 8] = [
        (|r, _, ev| r.retain(|x| x["row_id"] != ev), "missing row"),
        (
            |r, _, ev| {
                let mut twin = row_mut(r, ev).clone();
                twin["row_id"] = json!(rid(0x51, 1));
                r.push(twin);
            },
            "2 completed eval rows",
        ),
        (
            |r, _, ev| row_mut(r, ev)["status"] = json!("failed"),
            "missing row",
        ),
        (
            |r, _, ev| {
                row_mut(r, ev)["metrics"]["val_top1.span"] =
                    json!({"state": "not_run", "reason": "test"});
            },
            "val_top1.span is not_run",
        ),
        (
            |r, _, ev| row_mut(r, ev)["metrics"]["val_top1.span"]["value"] = json!(0.9),
            "is not its own n/n_total",
        ),
        (
            |r, ft, _| row_mut(r, ft)["quick"] = json!(true),
            "does not say quick: false",
        ),
        (
            |r, ft, _| row_mut(r, ft)["protocol"]["seed"] = json!(1),
            "is seed Some(1)",
        ),
        (
            |r, _, ev| {
                row_mut(r, ev)["gates"]["needle_hunk_recall"]["value"] = json!(0.9);
            },
            "is not its worst depth bucket",
        ),
    ];
    for (edit, phrase) in cases {
        let mut r = rows.clone();
        edit(&mut r, &ft0, &ev0);
        let ledger = s.ledger("gh200-v5.jsonl", &r);
        pause(&s, &p, &ledger, &ft0).refused_with(phrase);
    }
    // A half-written last line.
    let ledger = s.ledger("gh200-v5.jsonl", &rows);
    let mut text = std::fs::read_to_string(&ledger).unwrap();
    text.push_str("{\"row_id\": \"e");
    std::fs::write(&ledger, text).unwrap();
    pause(&s, &p, &ledger, &ft0).refused_with("malformed ledger line");
    // R9 is about seed 0; a seed-1 row id is refused before the ledger is read.
    let ledger = s.ledger("gh200-v5.jsonl", &rows);
    let o = run(
        &s,
        &args(&[
            &"v5-pause",
            &"--preregistration",
            &p,
            &"--ledger",
            &ledger,
            &"--ft-row",
            &format!("1={}", ids[1]),
        ]),
    );
    o.refused_with("R9 reads v5 seed 0's rows; --ft-row names seed 1");
    // An --out that exists is never overwritten.
    let out = s.path("exists.json");
    std::fs::write(&out, "x").unwrap();
    let o = Command::new(BIN)
        .args(args(&[
            &"v5-pause",
            &"--preregistration",
            &p,
            &"--ledger",
            &ledger,
            &"--ft-row",
            &format!("0={ft0}"),
            &"--out",
            &out,
        ]))
        .output()
        .unwrap();
    assert_eq!(String::from_utf8(o.stdout).unwrap(), "refused\n");
    assert_eq!(o.status.code(), Some(EXIT_REFUSED));
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "x");
}

// --- v5-noulw --room -------------------------------------------------------------------------------

/// v5 seeds 0-2 with these prose and unseen-language counts (of 60); the rest `BASE`.
fn ood(prose: [u64; 3], unseen: [u64; 3]) -> [P; 3] {
    [0, 1, 2].map(|i| P {
        prose: prose[i],
        unseen: unseen[i],
        ..BASE
    })
}

fn room(s: &Scratch, prereg: &Path, noul: &Path, ledger: &Path, ids: &[String]) -> Run {
    let mut a = args(&[
        &"v5-noulw",
        &"--room",
        &"--preregistration",
        &prereg,
        &"--noul-preregistration",
        &noul,
        &"--v5-ledger",
        &ledger,
    ]);
    a.extend(ft_args("--ft-row", ids));
    run(s, &a)
}

#[test]
fn room_iff_v5_holds_a_target_on_at_most_two_seeds_at_the_30_of_60_bar() {
    let s = scratch();
    let (p, q) = (bound(&s), repo(NOUL));
    for (prose, unseen, word) in [
        // 30/60 holds (2 * 30 >= 60): both targets held 3/3, no room.
        ([30, 30, 30], [30, 31, 60], "no_room"),
        // 29/60 does not hold: unseen-language held 2/3, room.
        ([30, 30, 30], [30, 29, 60], "room"),
        ([0, 0, 0], [60, 60, 60], "room"),
    ] {
        let (rows, ids) = seeds(0x5, false, ood(prose, unseen));
        let ledger = s.ledger("gh200-v5.jsonl", &rows);
        let o = room(&s, &p, &q, &ledger, &ids);
        o.said(word);
        let r = &o.json["detail"]["room"];
        assert_eq!(r[0]["target"], "ood_abstain.prose");
        assert_eq!(r[1]["target"], "ood_abstain.unseen-language");
        assert_eq!(o.json["detail"]["seed_holds"]["form"], "2*n >= 1*n_total");
        assert_eq!(o.json["detail"]["seed_holds"]["f2_bar"]["n"], 30);
        // --room reads the pre-registrations and v5's ledger, nothing else.
        assert_eq!(o.json["inputs"].as_array().unwrap().len(), 3);
    }
}

#[test]
fn room_refuses_unreadable_v5_rows_and_never_prints_room() {
    let s = scratch();
    let (p, q) = (bound(&s), repo(NOUL));
    let (rows, ids) = seeds(0x5, false, ood([0, 0, 0], [0, 0, 0]));
    let mut missing = rows.clone();
    missing.retain(|r| r["row_id"] != rid(0x5, 4));
    room(&s, &p, &q, &s.ledger("a.jsonl", &missing), &ids).refused_with("missing row");
    let mut over_61 = rows.clone();
    set(eval_of(&mut over_61, &ids[2]), "ood_abstain.prose", (0, 61));
    room(&s, &p, &q, &s.ledger("b.jsonl", &over_61), &ids)
        .refused_with("over 61 cases, not the 60 the F2 bar is stated over");
    let mut not_run = rows.clone();
    eval_of(&mut not_run, &ids[0])["metrics"]["ood_abstain.unseen-language"] =
        json!({"state": "not_run", "reason": "test"});
    room(&s, &p, &q, &s.ledger("c.jsonl", &not_run), &ids).refused_with("not ran");
    let mut quick = rows.clone();
    row_mut(&mut quick, &ids[1])["quick"] = json!(true);
    room(&s, &p, &q, &s.ledger("d.jsonl", &quick), &ids).refused_with("quick: false");
    // Seeds 3-4 are never part of the envelope.
    let ledger = s.ledger("e.jsonl", &rows);
    let mut a = args(&[
        &"v5-noulw",
        &"--room",
        &"--preregistration",
        &p,
        &"--noul-preregistration",
        &q,
        &"--v5-ledger",
        &ledger,
    ]);
    for (seed, id) in [(0, &ids[0]), (1, &ids[1]), (3, &ids[2])] {
        a.extend([String::from("--ft-row"), format!("{seed}={id}")]);
    }
    run(&s, &a).refused_with("the envelope is v5 seeds [0, 1, 2] only");
    // v5's seeds out of order refuse too: the arm's seed s is paired with v5's by position, so
    // the order is the same-seed alignment (each row's protocol.seed is checked against its own).
    let mut b = a[..8].to_vec();
    for (seed, id) in [(1, &ids[1]), (0, &ids[0]), (2, &ids[2])] {
        b.extend([String::from("--ft-row"), format!("{seed}={id}")]);
    }
    run(&s, &b).refused_with("given in that order; got [1, 0, 2]");
    room(&s, &repo(DRAFT), &q, &ledger, &ids).refused_with("\"draft\" key");
}

// --- v5-noulw ----------------------------------------------------------------------------------

struct Arm {
    v5: PathBuf,
    v5_ids: Vec<String>,
    arm: PathBuf,
    arm_ids: Vec<String>,
}

/// v5's ledger and the arm's, with `edit` applied to the arm's rows.
fn arm_fixture(
    s: &Scratch,
    v5: [P; 3],
    arm: [P; 3],
    edit: impl FnOnce(&mut Vec<Value>, &[String]),
) -> Arm {
    arm_fixture_both(s, v5, arm, |_, _| {}, edit)
}

/// v5's ledger and the arm's, with `edit_v5` applied to v5's rows and `edit` to the arm's.
fn arm_fixture_both(
    s: &Scratch,
    v5: [P; 3],
    arm: [P; 3],
    edit_v5: impl FnOnce(&mut Vec<Value>, &[String]),
    edit: impl FnOnce(&mut Vec<Value>, &[String]),
) -> Arm {
    let (mut v5_rows, v5_ids) = seeds(0x5, false, v5);
    let (mut arm_rows, arm_ids) = seeds(0xa, true, arm);
    edit_v5(&mut v5_rows, &v5_ids);
    edit(&mut arm_rows, &arm_ids);
    Arm {
        v5: s.ledger(
            &format!("gh200-v5-{}.jsonl", N.fetch_add(1, Ordering::SeqCst)),
            &v5_rows,
        ),
        v5_ids,
        arm: s.ledger(
            &format!("gh200-v5-noulw-{}.jsonl", N.fetch_add(1, Ordering::SeqCst)),
            &arm_rows,
        ),
        arm_ids,
    }
}

fn noulw_with(s: &Scratch, prereg: &Path, noul: &Path, fx: &Arm) -> Run {
    let mut a = args(&[
        &"v5-noulw",
        &"--preregistration",
        &prereg,
        &"--noul-preregistration",
        &noul,
        &"--v5-ledger",
        &fx.v5,
        &"--arm-ledger",
        &fx.arm,
    ]);
    a.extend(ft_args("--ft-row", &fx.v5_ids));
    a.extend(ft_args("--arm-ft-row", &fx.arm_ids));
    run(s, &a)
}

fn noulw(s: &Scratch, fx: &Arm) -> Run {
    noulw_with(s, &bound(s), &repo(NOUL), fx)
}

/// v5: prose held on 2 of 3 seeds and unseen-language on none, so both targets have room.
const V5_ROOM: ([u64; 3], [u64; 3]) = ([50, 20, 45], [20, 25, 10]);
/// The arm holding both on every seed.
const ARM_HOLDS: ([u64; 3], [u64; 3]) = ([50, 50, 50], [30, 31, 40]);

fn v5_room() -> [P; 3] {
    ood(V5_ROOM.0, V5_ROOM.1)
}

fn arm_holds() -> [P; 3] {
    ood(ARM_HOLDS.0, ARM_HOLDS.1)
}

/// `profiles` with one field of one seed changed.
fn with(mut profiles: [P; 3], seed: usize, edit: impl FnOnce(&mut P)) -> [P; 3] {
    edit(&mut profiles[seed]);
    profiles
}

#[test]
fn an_arm_holding_every_target_with_room_on_every_seed_and_losing_no_guard_wins() {
    let s = scratch();
    let o = noulw(&s, &arm_fixture(&s, v5_room(), arm_holds(), |_, _| {}));
    o.said("wins");
    let d = &o.json["detail"];
    assert_eq!(d["arm"]["targets"][0]["v5_holds"], 2);
    assert_eq!(d["arm"]["targets"][0]["arm_holds"], 3);
    assert_eq!(d["arm"]["targets"][0]["clears"], true);
    assert_eq!(d["arm"]["targets"][1]["clears"], true);
    // Every listed guard was judged, in the pre-registration's order.
    let guards: Vec<&str> = d["arm"]["guards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["metric"].as_str().unwrap())
        .collect();
    assert_eq!(
        guards,
        [
            "val_top1.choice",
            "val_top1.span",
            "permutation_consistency.family.code.defect_class",
            "ece.family.code.defect_class.choice.k4",
            "needle_8k_worst_bucket",
            "ood_abstain.scrambled",
            "ood_abstain.in_distribution.family.code.defect_class",
            "ood_abstain.in_distribution",
        ]
    );
    assert_eq!(d["arm"]["absolute_guard"]["form"], "50*n <= 1*n_total");
    assert_eq!(
        d["arm"]["identity"][0]["added"],
        json!({"noul_weight": 4.0, "noul_weight_scope": "code.defect_class"})
    );
    // Seed for seed, the arm trained on v5's batch order.
    for s_ in 0..3 {
        assert_eq!(
            d["arm"]["identity"][s_]["paired"][PAIRED],
            plan_order_digest(s_ as i64)
        );
    }
    assert_eq!(d["arm"]["recipe_hash"], "a5".repeat(32));
}

#[test]
fn a_target_clears_only_when_every_arm_seed_holds_even_above_v5s_count() {
    let s = scratch();
    // v5 holds prose on 1 of 3; the arm on 2 of 3: more than v5, but not every seed.
    let v5 = ood([50, 20, 20], V5_ROOM.1);
    let arm = with(arm_holds(), 2, |p| p.prose = 29);
    let o = noulw(&s, &arm_fixture(&s, v5, arm, |_, _| {}));
    o.said("quiet");
    let t = &o.json["detail"]["arm"]["targets"][0];
    assert_eq!(
        (t["v5_holds"].as_u64(), t["arm_holds"].as_u64()),
        (Some(1), Some(2))
    );
    assert_eq!(t["clears"], false);
}

#[test]
fn an_arm_holding_on_as_many_seeds_as_v5_does_not_clear() {
    let s = scratch();
    // v5 holds prose on 2 of 3; so does the arm (29/60 on seed 1).
    let arm = with(arm_holds(), 1, |p| p.prose = 29);
    let o = noulw(&s, &arm_fixture(&s, v5_room(), arm, |_, _| {}));
    o.said("quiet");
    let t = &o.json["detail"]["arm"]["targets"][0];
    assert_eq!(
        (t["v5_holds"].as_u64(), t["arm_holds"].as_u64()),
        (Some(2), Some(2))
    );
    assert_eq!(t["clears"], false);
    // At 30/60 it holds, and the arm wins.
    let arm = with(arm_holds(), 1, |p| p.prose = 30);
    noulw(&s, &arm_fixture(&s, v5_room(), arm, |_, _| {})).said("wins");
}

#[test]
fn a_target_v5_holds_on_every_seed_is_read_as_a_guard_the_arm_must_hold_everywhere() {
    let s = scratch();
    // v5 holds prose 3/3 (no room) and unseen-language 0/3 (room).
    let v5 = ood([50, 30, 45], V5_ROOM.1);
    noulw(&s, &arm_fixture(&s, v5, arm_holds(), |_, _| {})).said("wins");
    let arm = with(arm_holds(), 2, |p| p.prose = 29);
    let o = noulw(&s, &arm_fixture(&s, v5, arm, |_, _| {}));
    o.said("quiet");
    let t = &o.json["detail"]["arm"]["targets"][0];
    assert_eq!(t["room"], false);
    assert_eq!(t["loses"], true);
    assert!(t.get("clears").is_none());
}

#[test]
fn with_room_at_three_of_three_an_arm_holding_as_many_seeds_as_v5_does_not_clear() {
    let s = scratch();
    // "at most 3 of its 3 seeds": prose, which v5 holds on 3 of 3, has room. The arm holds it on
    // every seed too, but not on more seeds than v5, so it does not clear ("more" is strict).
    let three = prereg(&s, |p| {
        retext(
            p,
            &["arm_noul_weight", "comparison", "room"],
            "at most 2 of its 3 seeds",
            "at most 3 of its 3 seeds",
        );
    });
    let v5 = ood([50, 30, 45], V5_ROOM.1);
    let fx = arm_fixture(&s, v5, arm_holds(), |_, _| {});
    let o = noulw_with(&s, &three, &repo(NOUL), &fx);
    o.said("quiet");
    let t = &o.json["detail"]["arm"]["targets"][0];
    assert_eq!(t["room"], true);
    assert_eq!(
        (t["v5_holds"].as_u64(), t["arm_holds"].as_u64()),
        (Some(3), Some(3))
    );
    assert_eq!(t["clears"], false);
    // Under the bound text (at most 2) the same rows make prose a guard the arm holds: wins.
    noulw(&s, &fx).said("wins");
}

#[test]
fn no_target_with_room_is_no_room_for_the_waiter_and_refused_for_rows_that_exist_anyway() {
    let s = scratch();
    let v5 = ood([50, 30, 45], [30, 31, 40]);
    let fx = arm_fixture(&s, v5, arm_holds(), |_, _| {});
    room(&s, &bound(&s), &repo(NOUL), &fx.v5, &fx.v5_ids).said("no_room");
    let o = noulw(&s, &fx);
    o.refused_with("no target has room");
    // The arm's rows were still read and judged, for the record.
    assert_eq!(o.json["detail"]["arm"]["wins"], true);
}

#[test]
fn each_guard_holds_at_v5s_bound_and_loses_one_count_past_it_on_the_arms_worst_seed() {
    let s = scratch();
    // v5's envelope per guard, the arm at the bound (wins) and one past it (quiet).
    type Edit = fn(&mut P, bool);
    let cases: [(&str, [P; 3], Edit); 8] = [
        (
            "val_top1.choice",
            [
                BASE,
                P {
                    choice: (9100, 10985),
                    ..BASE
                },
                BASE,
            ],
            |p, past| p.choice = (if past { 9099 } else { 9100 }, 10985),
        ),
        (
            "val_top1.span",
            [
                BASE,
                P {
                    span: (6500, 7238),
                    ..BASE
                },
                BASE,
            ],
            |p, past| p.span = (if past { 6499 } else { 6500 }, 7238),
        ),
        (
            "permutation_consistency.family.code.defect_class",
            [BASE, P { perm: 2285, ..BASE }, BASE],
            |p, past| p.perm = if past { 2284 } else { 2285 },
        ),
        (
            "ece.family.code.defect_class.choice.k4",
            [BASE, P { ece: 0.022, ..BASE }, BASE],
            |p, past| p.ece = if past { 0.022_000_1 } else { 0.022 },
        ),
        (
            "needle_8k_worst_bucket",
            [
                BASE,
                P {
                    needle: (4, 35),
                    ..BASE
                },
                BASE,
            ],
            |p, past| p.needle = (4, if past { 34 } else { 35 }),
        ),
        (
            "ood_abstain.scrambled",
            [
                BASE,
                P {
                    scrambled: 50,
                    ..BASE
                },
                BASE,
            ],
            |p, past| p.scrambled = if past { 49 } else { 50 },
        ),
        (
            "ood_abstain.in_distribution.family.code.defect_class",
            [
                BASE,
                P {
                    id_dc: (20, 2304),
                    ..BASE
                },
                BASE,
            ],
            |p, past| p.id_dc = (if past { 21 } else { 20 }, 2304),
        ),
        (
            "ood_abstain.in_distribution",
            [
                BASE,
                P {
                    id_pooled: (860, 10985),
                    ..BASE
                },
                BASE,
            ],
            |p, past| p.id_pooled = (if past { 861 } else { 860 }, 10985),
        ),
    ];
    for (name, v5, edit) in cases {
        let v5 = [0, 1, 2].map(|i| P {
            prose: V5_ROOM.0[i],
            unseen: V5_ROOM.1[i],
            ..v5[i]
        });
        for past in [false, true] {
            // The edit lands on the arm's seed 2: its worst seed for that guard.
            let arm = with(arm_holds(), 2, |p| edit(p, past));
            let o = noulw(&s, &arm_fixture(&s, v5, arm, |_, _| {}));
            o.said(if past { "quiet" } else { "wins" });
            let g = o.json["detail"]["arm"]["guards"]
                .as_array()
                .unwrap()
                .iter()
                .find(|g| g["metric"] == name)
                .unwrap()
                .clone();
            assert_eq!(g["loses"], past, "{name}: {g}");
        }
    }
}

#[test]
fn the_absolute_guard_allows_50n_eq_n_total_and_loses_one_count_past_it_inside_v5s_envelope() {
    let s = scratch();
    // v5's defect_class in-distribution abstention 40 / 50 / 45 of 2304: every arm count below
    // is inside the envelope, so only the absolute guard can lose. 46/2304 and 47/2304 are the
    // printed example's boundary; at 2300 slots, 50 * 46 = 2300 exactly.
    let id_dc = [40, 50, 45];
    let v5 = [0, 1, 2].map(|i| P {
        prose: V5_ROOM.0[i],
        unseen: V5_ROOM.1[i],
        id_dc: (id_dc[i], 2304),
        ..BASE
    });
    for (worst, word) in [
        ((46, 2304), "wins"),
        ((47, 2304), "quiet"),
        ((46, 2300), "wins"),
        ((47, 2300), "quiet"),
    ] {
        let arm = [0, 1, 2].map(|i| P {
            prose: ARM_HOLDS.0[i],
            unseen: ARM_HOLDS.1[i],
            id_dc: if i == 1 { worst } else { (40, 2304) },
            ..BASE
        });
        let o = noulw(&s, &arm_fixture(&s, v5, arm, |_, _| {}));
        o.said(word);
        let a = &o.json["detail"]["arm"];
        assert_eq!(a["absolute_guard"]["loses"], word == "quiet");
        let envelope_guard = a["guards"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["metric"] == "ood_abstain.in_distribution.family.code.defect_class")
            .unwrap();
        assert_eq!(envelope_guard["loses"], false);
    }
}

#[test]
fn an_arm_that_is_not_v5_plus_exactly_the_two_added_keys_refuses() {
    let s = scratch();
    type Edit = fn(&mut Value);
    let cases: [(Edit, &str); 8] = [
        (
            |r| r["recipe"]["noul_weight"] = json!(3.0),
            "recipe.noul_weight is 3.0, not the added 4.0",
        ),
        (
            |r| {
                r["recipe"]
                    .as_object_mut()
                    .unwrap()
                    .remove("noul_weight_scope");
            },
            "recipe.noul_weight_scope is absent",
        ),
        (
            |r| r["recipe"]["noul_weight_scope"] = json!("all"),
            "recipe.noul_weight_scope is \"all\"",
        ),
        (
            |r| r["recipe"]["lr"] = json!(3e-5),
            "recipe.lr is 0.00003, not v5's 0.00001",
        ),
        (
            |r| r["recipe"]["beta2"] = json!(0.95),
            "recipe.beta2 is 0.95, not v5's absent",
        ),
        (
            |r| r["code_commit"] = json!("a5026707b6e3e57c253be32003ba32420f2d78e2"),
            "code_commit is",
        ),
        (
            |r| r["protocol"]["data_snapshot_hash"] = json!("ea".repeat(32)),
            "protocol.data_snapshot_hash is",
        ),
        (|r| r["quick"] = json!(true), "quick is true, not false"),
    ];
    for (edit, phrase) in cases {
        let fx = arm_fixture(&s, v5_room(), arm_holds(), |rows, ids| {
            edit(row_mut(rows, &ids[1]))
        });
        noulw(&s, &fx).refused_with(phrase);
    }
    // v5's own rows presented as the arm's: the two keys are missing.
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |_, _| {});
    let mut a = args(&[
        &"v5-noulw",
        &"--preregistration",
        &bound(&s),
        &"--noul-preregistration",
        &repo(NOUL),
        &"--v5-ledger",
        &fx.v5,
        &"--arm-ledger",
        &fx.v5,
    ]);
    a.extend(ft_args("--ft-row", &fx.v5_ids));
    a.extend(ft_args("--arm-ft-row", &fx.v5_ids));
    run(&s, &a).refused_with("recipe.noul_weight is absent");
}

#[test]
fn a_missing_doubled_unfinished_or_incomparable_arm_row_refuses() {
    let s = scratch();
    type Edit = fn(&mut Vec<Value>, &[String]);
    let cases: [(Edit, &str); 6] = [
        (
            |r, ids| r.retain(|x| x["metrics"]["ft_run_row_id"]["value"] != ids[0].as_str()),
            "missing row",
        ),
        (
            |r, ids| {
                let mut twin = eval_of(r, &ids[2]).clone();
                twin["row_id"] = json!(rid(0xb, 1));
                r.push(twin);
            },
            "2 completed eval rows",
        ),
        (
            |r, ids| row_mut(r, &ids[1])["status"] = json!("failed"),
            "is not a completed ft row",
        ),
        (
            |r, ids| {
                eval_of(r, &ids[1])["metrics"]["ood_abstain.prose"] =
                    json!({"state": "not_run", "reason": "test"});
            },
            "ood_abstain.prose is not_run",
        ),
        (
            |r, ids| eval_of(r, &ids[1])["recipe"]["val_shard_hash"] = json!("ff".repeat(32)),
            "recipe.val_shard_hash",
        ),
        (
            |r, ids| row_mut(r, &ids[2])["protocol"]["seed"] = json!(1),
            "is seed Some(1)",
        ),
    ];
    for (edit, phrase) in cases {
        let fx = arm_fixture(&s, v5_room(), arm_holds(), edit);
        noulw(&s, &fx).refused_with(phrase);
    }
}

#[test]
fn the_pairing_check_refuses_an_arm_seed_not_on_v5s_batch_order_for_that_seed() {
    let s = scratch();
    // Arm seed 1 trained on v5 seed 2's order: every other key agrees, and it refuses.
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |r, ids| {
        row_mut(r, &ids[1])["metrics"][PAIRED]["value"] = json!(plan_order_digest(2));
    });
    noulw(&s, &fx).refused_with(&format!(
        "arm ft row {} (seed Some(1)): metrics.{PAIRED} is {}, not v5 seed Some(1)'s {}: not \
         v5's batch order (the pairing check)",
        fx.arm_ids[1],
        plan_order_digest(2),
        plan_order_digest(1)
    ));
}

#[test]
fn the_arms_seeds_out_of_order_refuse_so_arm_seed_s_meets_v5_seed_s() {
    let s = scratch();
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |_, _| {});
    let mut a = args(&[
        &"v5-noulw",
        &"--preregistration",
        &bound(&s),
        &"--noul-preregistration",
        &repo(NOUL),
        &"--v5-ledger",
        &fx.v5,
        &"--arm-ledger",
        &fx.arm,
    ]);
    a.extend(ft_args("--ft-row", &fx.v5_ids));
    for seed in [1, 0, 2] {
        a.extend([
            String::from("--arm-ft-row"),
            format!("{seed}={}", fx.arm_ids[seed]),
        ]);
    }
    run(&s, &a).refused_with(
        "the arm's ft rows are seeds [0, 1, 2] (identity.ft_rows), given in that order; got \
         [1, 0, 2]",
    );
}

#[test]
fn the_pairing_check_refuses_when_the_order_digest_is_absent_or_unread_on_either_side() {
    let s = scratch();
    type Edit = fn(&mut Vec<Value>, &[String]);
    let none: Edit = |_, _| {};
    let drop_seed_2: Edit = |r, ids| {
        row_mut(r, &ids[2])["metrics"]
            .as_object_mut()
            .unwrap()
            .remove(PAIRED);
    };
    let not_run: Edit = |r, ids| {
        row_mut(r, &ids[0])["metrics"][PAIRED] = json!({"state": "not_run", "reason": "test"});
    };
    let empty: Edit = |r, ids| row_mut(r, &ids[0])["metrics"][PAIRED]["value"] = json!("");
    let number: Edit = |r, ids| row_mut(r, &ids[0])["metrics"][PAIRED]["value"] = json!(176);
    // (v5's edit, the arm's edit, the row the reason names, the phrase)
    let cases: [(Edit, Edit, bool, String); 5] = [
        (
            none,
            drop_seed_2,
            true,
            format!("no metrics.{PAIRED} (the pairing check: absent on either side refuses)"),
        ),
        (
            drop_seed_2,
            none,
            false,
            format!("no metrics.{PAIRED} (the pairing check: absent on either side refuses)"),
        ),
        (
            none,
            not_run,
            true,
            format!("metrics.{PAIRED} is not_run (test), not ran (the pairing check"),
        ),
        (
            none,
            empty,
            true,
            format!("metrics.{PAIRED} records no string value (the pairing check"),
        ),
        (
            number,
            none,
            false,
            format!("metrics.{PAIRED} records no string value (the pairing check"),
        ),
    ];
    for (edit_v5, edit, on_arm, phrase) in cases {
        let fx = arm_fixture_both(&s, v5_room(), arm_holds(), edit_v5, edit);
        let o = noulw(&s, &fx);
        o.refused_with(&phrase);
        // The reason names v5's row iff v5's side lacks it (the arm's row id is always named).
        let reason = o.json["refused"].as_str().unwrap();
        let names_v5 = fx
            .v5_ids
            .iter()
            .any(|id| reason.contains(&format!("row {id} (")));
        assert_eq!(names_v5, !on_arm, "{reason}");
    }
}

#[test]
fn the_paired_metric_is_the_one_the_preregistration_names() {
    let s = scratch();
    // Name train.consumed_digest instead: the rows carry corpus.plan_order_digest only, so the
    // check reads the named metric, finds it on neither side, and refuses.
    let consumed = prereg(&s, |p| {
        retext(
            p,
            &["arm_noul_weight", "identity", "ft_rows"],
            "corpus.plan_order_digest equal to",
            "train.consumed_digest equal to",
        );
    });
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |_, _| {});
    noulw_with(&s, &consumed, &repo(NOUL), &fx)
        .refused_with("no metrics.train.consumed_digest (the pairing check");
    // With it recorded and equal seed for seed, the arm wins on that metric.
    let digest = |r: &mut Vec<Value>, ids: &[String]| {
        for (seed, id) in ids.iter().enumerate() {
            row_mut(r, id)["metrics"]["train.consumed_digest"] =
                json!({"state": "ran", "value": format!("c{seed}").repeat(32)});
        }
    };
    let fx = arm_fixture_both(&s, v5_room(), arm_holds(), digest, digest);
    let o = noulw_with(&s, &consumed, &repo(NOUL), &fx);
    o.said("wins");
    assert_eq!(
        o.json["detail"]["arm"]["identity"][2]["paired"]["train.consumed_digest"],
        "c2".repeat(32)
    );
}

#[test]
fn the_arms_three_ft_rows_must_be_one_configuration_like_v5s() {
    let s = scratch();
    // Arm seed 1's recipe hash differs; its recipe, which the per-seed identity compares key by
    // key, does not. one_configuration (the envelope's own check) refuses it.
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |r, ids| {
        row_mut(r, &ids[1])["protocol"]["recipe_hash"] = json!("a6".repeat(32));
    });
    noulw(&s, &fx).refused_with(&format!(
        "ft row {} has recipe {} / data {}, not the first row's {} / {}: not one configuration",
        fx.arm_ids[1],
        "a6".repeat(32),
        v5_data(),
        "a5".repeat(32),
        v5_data()
    ));
}

#[test]
fn a_preregistration_that_disagrees_with_the_checker_refuses_before_any_ledger_is_read() {
    let s = scratch();
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |_, _| {});
    noulw_with(&s, &repo(DRAFT), &repo(NOUL), &fx).refused_with("\"draft\" key");
    type Edit = fn(&mut Map<String, Value>);
    let cases: [(Edit, &str); 13] = [
        (
            |p| {
                p["arm_noul_weight"]["guards"].as_array_mut().unwrap().push(
                    json!({"name": "val_top1.score", "direction": "higher", "form": "count"}),
                );
            },
            "\"val_top1.score\" is not a metric this checker reads",
        ),
        (
            |p| p["arm_noul_weight"]["guards"][1]["direction"] = json!("lower"),
            "val_top1.span is listed \"lower\"",
        ),
        (
            |p| p["arm_noul_weight"]["guards"][3]["form"] = json!("count"),
            "is read as \"count\"; this checker reads it as \"f64\"",
        ),
        (
            |p| p["arm_noul_weight"]["targets"][0]["form"] = json!("count"),
            "this checker reads targets as seed_holds",
        ),
        (
            |p| p["arm_noul_weight"]["w"]["value"] = json!(5),
            "does not set noul_weight",
        ),
        (
            |p| p["arm_noul_weight"]["outcomes"]["words"] = json!(["wins", "quiet"]),
            "outcomes.words",
        ),
        (
            |p| {
                retext(
                    p,
                    &["arm_noul_weight", "identity", "ft_rows"],
                    "two added keys",
                    "three added keys",
                );
            },
            "\"three added key(s)\" lists 2",
        ),
        (
            |p| {
                retext(
                    p,
                    &["arm_noul_weight", "comparison", "targets_form"],
                    "2 * n >= n_total",
                    "3 * n >= n_total",
                );
            },
            "is not the F2 bar 30/60",
        ),
        (
            |p| {
                retext(
                    p,
                    &["arm_noul_weight", "identity", "eval_rows"],
                    "recipe.needle and recipe.ood",
                    "recipe.needle, recipe.ood and recipe.score_dtype",
                );
            },
            "holds recipe keys",
        ),
        (
            |p| {
                retext(
                    p,
                    &["arm_noul_weight", "comparison", "absolute_guard"],
                    "(<= 46/2304",
                    "(<= 47/2304",
                );
            },
            "does not make 47/2304 the largest count it allows",
        ),
        (
            |p| {
                retext(
                    p,
                    &["arm_noul_weight", "comparison", "guards_form"],
                    "ties never lose",
                    "ties lose",
                );
            },
            "does not say \"ties never lose\"",
        ),
        (
            |p| {
                retext(
                    p,
                    &["arm_noul_weight", "identity", "ft_rows"],
                    "corpus.plan_order_digest equal to v5's ft row of the same seed",
                    "corpus.plan_order_digest recorded",
                );
            },
            "does not say \"<metric> equal to v5's ft row of the same seed\"",
        ),
        (
            |p| {
                retext(
                    p,
                    &["arm_noul_weight", "identity", "ft_rows"],
                    "absent on either side refuses",
                    "absent on the arm's side refuses",
                );
            },
            "corpus.plan_order_digest equal to v5's ft row of the same seed, absent on either \
             side refuses",
        ),
    ];
    for (edit, phrase) in cases {
        let o = noulw_with(&s, &prereg(&s, edit), &repo(NOUL), &fx);
        o.refused_with(phrase);
        let ledgers = o.json["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|i| i.get("role").is_none())
            .count();
        assert_eq!(ledgers, 0, "a ledger was read: {}", o.json);
    }
}

#[test]
fn f2_and_f3_are_read_from_the_noul_preregistration_and_must_agree_with_v5s_forms() {
    let s = scratch();
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |_, _| {});
    let p = bound(&s);
    let bar = noul(&s, |q| {
        q["ood_abstain_by_category"]["bar"] =
            json!(">= 31/60 on every seed (0, 1, 2), each of prose, scrambled, unseen-language")
    });
    noulw_with(&s, &p, &bar, &fx).refused_with("seed_holds prints the F2 bar as 30/60");
    let bound3 = noul(&s, |q| {
        q["defect_class_in_distribution_abstention"]["bound"] = json!("<= 3% on every seed")
    });
    noulw_with(&s, &p, &bound3, &fx).refused_with("is not F3's");
    let unnamed = noul(&s, |q| {
        q["ood_abstain_by_category"]["bar"] =
            json!(">= 30/60 on every seed (0, 1, 2), each of prose and scrambled")
    });
    noulw_with(&s, &p, &unnamed, &fx)
        .refused_with("unseen-language's category is not among the F2 bar's");
}

#[test]
fn room_and_the_added_keys_are_read_from_the_preregistration_not_from_this_code() {
    let s = scratch();
    // "at most 1 of its 3 seeds": prose, held by v5 on 2, has no room and is read as a guard,
    // which the arm holds on 3 of 3; the decision rests on unseen-language and the arm wins.
    let one = prereg(&s, |p| {
        retext(
            p,
            &["arm_noul_weight", "comparison", "room"],
            "at most 2 of its 3 seeds",
            "at most 1 of its 3 seeds",
        );
    });
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |_, _| {});
    let o = noulw_with(&s, &one, &repo(NOUL), &fx);
    o.said("wins");
    assert_eq!(o.json["detail"]["room"][0]["room"], false);
    assert_eq!(o.json["detail"]["room"][1]["room"], true);
    // A weight of 5 written in both places is applied: the arm's recorded 4.0 then refuses.
    let five = prereg(&s, |p| {
        p["arm_noul_weight"]["w"]["value"] = json!(5);
        retext(
            p,
            &["arm_noul_weight", "identity", "ft_rows"],
            "noul_weight = 4.0",
            "noul_weight = 5.0",
        );
    });
    noulw_with(&s, &five, &repo(NOUL), &fx)
        .refused_with("recipe.noul_weight is 4.0, not the added 5.0");
}

#[test]
fn the_cli_refuses_an_arm_ledger_with_room_and_no_arm_ledger_without_it() {
    let s = scratch();
    let fx = arm_fixture(&s, v5_room(), arm_holds(), |_, _| {});
    let mut a = args(&[
        &"v5-noulw",
        &"--preregistration",
        &bound(&s),
        &"--noul-preregistration",
        &repo(NOUL),
        &"--v5-ledger",
        &fx.v5,
    ]);
    a.extend(ft_args("--ft-row", &fx.v5_ids));
    // Neither --room nor --arm-ledger: a usage error (exit 2), and no word on stdout.
    let o = run(&s, &a);
    assert_eq!((o.word.as_str(), o.code), ("", 2), "{}", o.stderr);
    assert!(
        o.stderr.contains("required arguments were not provided")
            && o.stderr.contains("--arm-ledger"),
        "{}",
        o.stderr
    );
    a.extend(args(&[&"--room", &"--arm-ledger", &fx.arm]));
    let o = run(&s, &a);
    assert_eq!((o.word.as_str(), o.code), ("", 2), "{}", o.stderr);
    assert!(
        o.stderr
            .contains("'--room' cannot be used with '--arm-ledger"),
        "{}",
        o.stderr
    );
}
