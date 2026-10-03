//! `qd-post-f-rules tierb`: the Tier-B outcome rule (`recipe.tierb_outcome_rule` of
//! `campaign/v5-preregistered.json`, as amended by Fable on 2026-10-03), run the way the box
//! runs it: the binary, the one word it prints, its exit code and its `--out` JSON.
//!
//! Every candidate ledger here is built from phase-3 seed 0's real committed rows: ft 7f2c11db,
//! its training-run score row 6d170b3c, its linear control eeda5db4
//! (`ledger/gh200-seed0-weights-2026-09-30.jsonl`) and its fp32 all-gates re-score 58fd1532
//! (`ledger/gh200-allgates-2026-09-30.jsonl`), with the candidate's recipe key injected into the
//! ft recipe and whatever edit each test makes. The phase-3 and all-gates ledgers are the real
//! files unless a test says otherwise; the pre-registration is the real one (the DRAFT as the
//! freeze renamed it, AUDIT/finalize-2026-10-03/apply_v5_freeze.py) unless a test edits a copy.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-post-f-rules");
const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const PHASE3: &str = "ledger/gh200-seed0-weights-2026-09-30.jsonl";
const ALLGATES: &str = "ledger/gh200-allgates-2026-09-30.jsonl";
const PREREG: &str = "campaign/v5-preregistered.json";
const EXIT_REFUSED: i32 = 3;

const FT: &str = "7f2c11db-3eb8-4361-9620-6b164ec37f1e";
const EVAL: &str = "6d170b3c-5676-446b-b6be-5bb17d6d7aa0";
const CONTROL: &str = "eeda5db4-0e99-415c-9a9d-5bd0efc8b71b";
const ALL: &str = "58fd1532-3239-41eb-a506-33ba10079425";
const ENVELOPE: [&str; 3] = [
    EVAL,
    "60f29b07-c7d1-41a9-a8e5-af96227206a6",
    "5c19c0e8-0e8d-4aa9-b9c6-73febce8a57d",
];
/// eeda5db4's detail string, byte for byte.
const REF_DETAIL: &str =
    "paired margin +0.1501, 95% CI [+0.1359, +0.1651] over 10000 bootstrap resamples";
const INCLUDES_ZERO: &str = " -- CI includes zero, so this is not a win";
/// What a detail that is a byte off the harness form refuses with.
const FORM: &str = "not in the form paired_margin_test writes";
/// What a detail whose suffix is not the one its flag and bounds call for refuses with.
const SUFFIX: &str = "the suffix and text are";

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

fn real_row(rel: &str, id: &str) -> Value {
    real_rows(rel)
        .into_iter()
        .find(|r| r["row_id"] == json!(id))
        .unwrap()
}

fn rid(n: u64) -> String {
    format!("7e57b000-0000-4000-8000-{n:012x}")
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
    let dir =
        std::env::temp_dir().join(format!("qd-post-f-rules-tierb-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

impl Scratch {
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
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
    fn clause(&self, n: &str) -> bool {
        self.json["detail"]["clauses"][n]["holds"]
            .as_bool()
            .unwrap_or_else(|| panic!("no clause {n} in {}", self.json))
    }
}

/// One tierb invocation's inputs: the candidate's four rows (and any extra), and the reference
/// ledgers and pre-registration when a test replaces the real ones.
struct Fx {
    candidate: &'static str,
    ft: Option<Value>,
    eval: Option<Value>,
    control: Option<Value>,
    allgates: Option<Value>,
    extra: Vec<Value>,
    phase3: Option<Vec<Value>>,
    allgates_ledger: Option<Vec<Value>>,
    prereg: Option<Value>,
}

/// Phase-3 seed 0's real rows as a candidate's. The one edit: the ft recipe carries the
/// candidate's own key (`real_ft_run.py` writes `train_attention_mask: "none"` for nomask,
/// 1cf8e6d:360, and `optimizer_fused: true` for fused, e19dcf6:336-337). Everything else is
/// already linked as the outcome script's rows are: 6d170b3c and 58fd1532 name 7f2c11db in
/// `metrics.ft_run_row_id` (58fd1532 with `score_dtype` fp32 and the needle and OOD suites,
/// 6d170b3c without `score_dtype`), and eeda5db4 is `tools/ft_linear_control.py` with
/// `recipe.eval_row_id` 6d170b3c.
fn fx(candidate: &'static str) -> Fx {
    let mut ft = real_row(PHASE3, FT);
    match candidate {
        "nomask" => ft["recipe"]["train_attention_mask"] = json!("none"),
        "fused" => ft["recipe"]["optimizer_fused"] = json!(true),
        other => panic!("no candidate {other}"),
    }
    Fx {
        candidate,
        ft: Some(ft),
        eval: Some(real_row(PHASE3, EVAL)),
        control: Some(real_row(PHASE3, CONTROL)),
        allgates: Some(real_row(ALLGATES, ALL)),
        extra: Vec::new(),
        phase3: None,
        allgates_ledger: None,
        prereg: None,
    }
}

fn real_prereg() -> Value {
    serde_json::from_slice(&std::fs::read(repo(PREREG)).unwrap()).unwrap()
}

fn go(f: &Fx) -> Run {
    let s = scratch();
    let rows: Vec<Value> = [&f.ft, &f.eval, &f.control, &f.allgates]
        .into_iter()
        .flatten()
        .cloned()
        .chain(f.extra.iter().cloned())
        .collect();
    let ledger = s.ledger("candidate.jsonl", &rows);
    let phase3 = match &f.phase3 {
        Some(rows) => s.ledger("phase3.jsonl", rows),
        None => repo(PHASE3),
    };
    let allgates = match &f.allgates_ledger {
        Some(rows) => s.ledger("allgates.jsonl", rows),
        None => repo(ALLGATES),
    };
    let prereg = match &f.prereg {
        Some(p) => {
            let path = s.path("prereg.json");
            std::fs::write(&path, serde_json::to_vec_pretty(p).unwrap()).unwrap();
            path
        }
        None => repo(PREREG),
    };
    let out = s.path("out.json");
    let o = Command::new(BIN)
        .arg("tierb")
        .arg("--candidate")
        .arg(f.candidate)
        .arg("--ledger")
        .arg(&ledger)
        .arg("--phase3-ledger")
        .arg(&phase3)
        .arg("--allgates-ledger")
        .arg(&allgates)
        .arg("--preregistration")
        .arg(&prereg)
        .arg("--out")
        .arg(&out)
        .current_dir(&s.0)
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

fn set_count(row: &mut Value, key: &str, k: u64) {
    let e = &mut row["metrics"][key];
    assert!(e.is_object(), "{key} is not on the row");
    let n = e["n_total"].as_u64().unwrap();
    e["n"] = json!(k);
    e["value"] = json!(k as f64 / n as f64);
}

/// The control row's gate as `paired_margin_test` would write it: detail, value, passed.
fn set_gate(control: &mut Value, detail: &str, value: f64, passed: bool) {
    let g = &mut control["gates"]["paired_margin_vs_linear"];
    g["detail"] = json!(detail);
    g["value"] = json!(value);
    g["passed"] = json!(passed);
}

fn ci(point: &str, lo: &str, hi: &str) -> String {
    format!("paired margin {point}, 95% CI [{lo}, {hi}] over 10000 bootstrap resamples")
}

fn conditional<'a>(p: &'a mut Value, id: &str) -> &'a mut Value {
    p["recipe"]["conditionals"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == json!(id))
        .unwrap()
}

// --- the real rows ------------------------------------------------------------------------------

/// The smoke: phase-3 seed 0's own rows, read as a candidate, pass every clause against the real
/// phase-3 and all-gates ledgers and the real pre-registration. This is what proves the checker
/// reads the real row formats (counts, the control's detail string, the fp32 row's gates).
#[test]
fn phase3_seed0s_real_rows_pass_as_either_candidate_against_the_real_preregistration() {
    for candidate in ["nomask", "fused"] {
        let r = go(&fx(candidate));
        r.said("pass");
        let j = &r.json;
        assert_eq!(j["rule"], json!("tierb_outcome"), "{j}");
        assert_eq!(j["decision"], json!("pass"));
        let d = &j["detail"];
        assert_eq!(d["candidate"], json!(candidate));
        assert_eq!(d["preregistration"]["draft"], json!(false), "{d}");
        assert_eq!(
            d["preregistration"]["sha256"].as_str().map(str::len),
            Some(64),
            "{d}"
        );
        assert_eq!(d["rows"]["ft"], json!(FT));
        assert_eq!(d["rows"]["eval"], json!(EVAL));
        assert_eq!(d["rows"]["control"], json!(CONTROL));
        assert_eq!(d["rows"]["allgates"], json!(ALL));
        let env = &d["envelope"];
        assert_eq!(env["rows"], json!(ENVELOPE));
        let choice = &env["metrics"]["val_top1.choice"];
        assert_eq!(choice["min"]["n"], json!(2326), "{choice}");
        assert_eq!(choice["max"]["n"], json!(2327));
        assert_eq!(choice["range"]["n"], json!(1));
        assert_eq!(choice["inside_from"]["n"], json!(2325));
        assert_eq!(choice["inside_to"]["n"], json!(2328));
        let span = &env["metrics"]["val_top1.span"];
        assert_eq!(span["min"]["n"], json!(2048), "{span}");
        assert_eq!(span["max"]["n"], json!(2050));
        assert_eq!(span["range"]["n"], json!(2));
        assert_eq!(span["inside_from"]["n"], json!(2046));
        assert_eq!(span["inside_to"]["n"], json!(2052));
        let c1 = &d["clauses"]["1"];
        assert_eq!(c1["val_top1.choice"]["candidate"]["n"], json!(2327), "{c1}");
        assert_eq!(c1["val_top1.choice"]["inside"], json!(true));
        assert_eq!(c1["val_top1.choice"]["literal"]["inside"], json!(true));
        assert_eq!(c1["val_top1.span"]["candidate"]["n"], json!(2050));
        assert_eq!(c1["val_top1.span"]["literal"]["inside"], json!(true));
        let c2 = &d["clauses"]["2"];
        assert_eq!(c2["candidate_ci"]["lo"], json!(1359), "{c2}");
        assert_eq!(c2["candidate_ci"]["hi"], json!(1651));
        assert_eq!(c2["reference_ci"]["lo"], json!(1359));
        assert_eq!(c2["reference_ci"]["hi"], json!(1651));
        assert_eq!(c2["reference_ci"]["point"], json!(1501));
        assert_eq!(d["clauses"]["3"]["holds"], json!(true));
        let inputs = j["inputs"].as_array().unwrap();
        assert_eq!(inputs.len(), 4, "{j}");
        assert!(
            inputs
                .iter()
                .all(|i| i["sha256"].as_str().is_some_and(|s| s.len() == 64))
        );
    }
}

#[test]
fn the_draft_and_the_renamed_file_both_bind_and_say_which_was_read() {
    // The DRAFT's form: the renamed file with its top-level `draft` key put back.
    let mut p = real_prereg();
    p.as_object_mut()
        .unwrap()
        .insert("draft".to_string(), json!("NOT BINDING."));
    let mut f = fx("nomask");
    f.prereg = Some(p);
    let r = go(&f);
    r.said("pass");
    assert_eq!(r.json["detail"]["preregistration"]["draft"], json!(true));
}

// --- clause 1, the envelope ---------------------------------------------------------------------

#[test]
fn a_synthetic_candidate_inside_every_clause_passes() {
    let mut f = fx("nomask");
    set_count(f.eval.as_mut().unwrap(), "val_top1.choice", 2328);
    set_count(f.eval.as_mut().unwrap(), "val_top1.span", 2046);
    set_count(f.allgates.as_mut().unwrap(), "val_top1.choice", 2325);
    set_count(f.allgates.as_mut().unwrap(), "val_top1.span", 2052);
    set_gate(
        f.control.as_mut().unwrap(),
        &ci("+0.1550", "+0.1400", "+0.1700"),
        0.155,
        true,
    );
    let r = go(&f);
    r.said("pass");
    assert!(r.clause("1") && r.clause("2") && r.clause("3"));
}

#[test]
fn the_envelope_is_inside_exactly_at_2min_minus_max_and_2max_minus_min() {
    for (key, inside, outside) in [
        ("val_top1.choice", [2325, 2328], [2324, 2329]),
        ("val_top1.span", [2046, 2052], [2045, 2053]),
    ] {
        for k in inside {
            let mut f = fx("nomask");
            set_count(f.eval.as_mut().unwrap(), key, k);
            go(&f).said("pass");
        }
        for k in outside {
            let mut f = fx("nomask");
            set_count(f.eval.as_mut().unwrap(), key, k);
            let r = go(&f);
            r.said("fail");
            assert!(
                !r.clause("1") && r.clause("2") && r.clause("3"),
                "{key} {k}"
            );
            assert_eq!(
                r.json["detail"]["clauses"]["1"][key]["inside"],
                json!(false)
            );
        }
    }
}

#[test]
fn a_range_of_zero_refuses() {
    let mut rows = real_rows(PHASE3);
    for r in rows.iter_mut() {
        if ENVELOPE.contains(&r["row_id"].as_str().unwrap()) {
            set_count(r, "val_top1.choice", 2326);
        }
    }
    let mut f = fx("nomask");
    f.phase3 = Some(rows);
    go(&f).refused_with("range of 0");
}

// --- clause 2, the control CI -------------------------------------------------------------------

#[test]
fn a_control_ci_that_misses_the_reference_fails_and_a_touching_one_overlaps() {
    // Touching from above: candidate lo == reference hi.
    let mut f = fx("nomask");
    set_gate(
        f.control.as_mut().unwrap(),
        &ci("+0.1680", "+0.1651", "+0.1700"),
        0.168,
        true,
    );
    go(&f).said("pass");
    // Touching from below: candidate hi == reference lo.
    let mut f = fx("nomask");
    set_gate(
        f.control.as_mut().unwrap(),
        &ci("+0.1330", "+0.1300", "+0.1359"),
        0.133,
        true,
    );
    go(&f).said("pass");
    // One ten-thousandth past either end: no overlap, fail.
    for (point, lo, hi, value) in [
        ("+0.1680", "+0.1652", "+0.1700", 0.168),
        ("+0.1330", "+0.1300", "+0.1358", 0.133),
    ] {
        let mut f = fx("nomask");
        set_gate(f.control.as_mut().unwrap(), &ci(point, lo, hi), value, true);
        let r = go(&f);
        r.said("fail");
        assert!(r.clause("1") && !r.clause("2") && r.clause("3"));
    }
}

#[test]
fn the_two_not_a_win_forms_parse_and_fail() {
    let below = format!(
        "{} -- the whole interval is below zero, so this is not an inconclusive result: the \
         baseline beats the model by 0.0100 and the comparison separates them",
        ci("-0.0100", "-0.0200", "-0.0050")
    );
    let straddles = format!("{}{INCLUDES_ZERO}", ci("+0.0010", "-0.0050", "+0.0070"));
    for (detail, value) in [(below, -0.01), (straddles, 0.001)] {
        let mut f = fx("nomask");
        set_gate(f.control.as_mut().unwrap(), &detail, value, false);
        let r = go(&f);
        r.said("fail");
        assert!(!r.clause("2"), "{detail}");
    }
}

#[test]
fn a_control_detail_off_the_harness_form_refuses() {
    let below_wrong_amount = format!(
        "{} -- the whole interval is below zero, so this is not an inconclusive result: the \
         baseline beats the model by 0.0101 and the comparison separates them",
        ci("-0.0100", "-0.0200", "-0.0050")
    );
    let cases: Vec<(String, f64, bool, &str)> = vec![
        // One byte changed: a trailing full stop, a missing space, three decimals.
        (
            format!("{REF_DETAIL}."),
            0.150_085_763_293_310_47,
            true,
            FORM,
        ),
        (
            REF_DETAIL.replace("+0.1359, +0.1651", "+0.1359,+0.1651"),
            0.150_085_763_293_310_47,
            true,
            FORM,
        ),
        (
            REF_DETAIL.replace("+0.1359", "+0.136"),
            0.150_085_763_293_310_47,
            true,
            FORM,
        ),
        // The suffix disagrees with the passed flag.
        (
            format!("{REF_DETAIL}{INCLUDES_ZERO}"),
            0.150_085_763_293_310_47,
            true,
            SUFFIX,
        ),
        (
            REF_DETAIL.to_string(),
            0.150_085_763_293_310_47,
            false,
            "passes iff lo > 0",
        ),
        // The straddling suffix on an interval wholly below zero.
        (
            format!("{}{INCLUDES_ZERO}", ci("-0.0100", "-0.0200", "-0.0050")),
            -0.01,
            false,
            SUFFIX,
        ),
        (below_wrong_amount, -0.01, false, SUFFIX),
        // The printed point is not the gate's value to four places.
        (
            REF_DETAIL.to_string(),
            0.16,
            true,
            "is not the gate's value",
        ),
        // Another bootstrap count.
        (
            REF_DETAIL.replace("10000", "1000"),
            0.150_085_763_293_310_47,
            true,
            "bootstrap resamples, not the 10000",
        ),
        // lo above hi.
        (
            ci("+0.1501", "+0.1651", "+0.1359"),
            0.150_085_763_293_310_47,
            true,
            "is above hi",
        ),
    ];
    for (detail, value, passed, phrase) in cases {
        let mut f = fx("nomask");
        set_gate(f.control.as_mut().unwrap(), &detail, value, passed);
        let r = go(&f);
        assert_eq!(r.word, "refused", "{detail:?}: {}", r.stderr);
        r.refused_with(phrase);
    }
}

#[test]
fn a_control_gate_that_did_not_run_refuses() {
    let mut f = fx("nomask");
    f.control.as_mut().unwrap()["gates"]["paired_margin_vs_linear"] =
        json!({"state": "not_run", "reason": "not evaluated"});
    go(&f).refused_with("gates.paired_margin_vs_linear is not_run");
}

// --- clause 3, the all-gates row ----------------------------------------------------------------

#[test]
fn the_allgates_row_outside_the_envelope_or_with_another_flag_fails() {
    let mut f = fx("nomask");
    set_count(f.allgates.as_mut().unwrap(), "val_top1.span", 2045);
    let r = go(&f);
    r.said("fail");
    assert!(r.clause("1") && r.clause("2") && !r.clause("3"));

    for gate in ["needle_hunk_recall", "ece"] {
        let mut f = fx("nomask");
        let g = &mut f.allgates.as_mut().unwrap()["gates"][gate];
        let flipped = !g["passed"].as_bool().unwrap();
        g["passed"] = json!(flipped);
        let r = go(&f);
        r.said("fail");
        assert!(r.clause("1") && r.clause("2") && !r.clause("3"), "{gate}");
    }
}

#[test]
fn a_gate_ran_in_the_reference_but_not_in_the_candidate_refuses() {
    for state in [Some("not_run"), None] {
        let mut f = fx("nomask");
        let gates = f.allgates.as_mut().unwrap()["gates"]
            .as_object_mut()
            .unwrap();
        match state {
            Some(s) => {
                gates.insert(
                    "ece".to_string(),
                    json!({"state": s, "reason": "not evaluated"}),
                );
            }
            None => {
                gates.remove("ece");
            }
        }
        go(&f).refused_with("gate ece ran in 58fd1532");
    }
}

// --- the candidate's rows -----------------------------------------------------------------------

#[test]
fn each_candidate_row_missing_refuses() {
    for (which, phrase) in [
        ("ft", "no completed ft row tagged epoch"),
        ("eval", "without score_dtype"),
        ("control", "linear-control"),
        ("allgates", "score_dtype fp32"),
    ] {
        let mut f = fx("nomask");
        match which {
            "ft" => f.ft = None,
            "eval" => f.eval = None,
            "control" => f.control = None,
            _ => f.allgates = None,
        }
        let r = go(&f);
        r.refused_with(phrase);
        assert!(
            r.json["refused"].as_str().unwrap().contains("missing row"),
            "{which}: {}",
            r.json["refused"]
        );
    }
}

#[test]
fn each_candidate_row_present_twice_refuses() {
    for (which, phrase) in [
        ("ft", "2 completed ft rows tagged epoch"),
        ("eval", "without score_dtype"),
        ("control", "2 completed linear-control rows"),
        ("allgates", "score_dtype fp32"),
    ] {
        let mut f = fx("nomask");
        let row = match which {
            "ft" => f.ft.clone(),
            "eval" => f.eval.clone(),
            "control" => f.control.clone(),
            _ => f.allgates.clone(),
        };
        let mut twin = row.unwrap();
        twin["row_id"] = json!(rid(1));
        f.extra.push(twin);
        go(&f).refused_with(phrase);
    }
}

#[test]
fn eval_rows_that_cannot_be_told_apart_by_score_dtype_refuse() {
    // The all-gates re-score without its dtype: two rows read as the training run's.
    let mut f = fx("nomask");
    f.allgates.as_mut().unwrap()["recipe"]
        .as_object_mut()
        .unwrap()
        .remove("score_dtype");
    go(&f).refused_with("without score_dtype");
    // A third score row of the same ft row in another dtype.
    let mut f = fx("nomask");
    let mut third = f.allgates.clone().unwrap();
    third["row_id"] = json!(rid(2));
    third["recipe"]["score_dtype"] = json!("bf16");
    f.extra.push(third);
    go(&f).refused_with("has score_dtype \"bf16\"");
    // The fp32 re-score without the needle suite is not the all-gates row.
    let mut f = fx("nomask");
    f.allgates.as_mut().unwrap()["recipe"]
        .as_object_mut()
        .unwrap()
        .remove("needle");
    go(&f).refused_with("no recipe.needle suite");
}

#[test]
fn a_quick_candidate_ft_row_refuses() {
    let mut f = fx("nomask");
    f.ft.as_mut().unwrap()["quick"] = json!(true);
    go(&f).refused_with("does not say quick: false");
}

#[test]
fn an_identity_key_off_phase3_seed0_refuses() {
    for (key, other) in [
        ("shard_hash", json!("00".repeat(32))),
        ("batch_tokens", json!(8192)),
        ("optimizer_recipe", json!("bf16")),
        ("lr", json!(3e-5)),
        ("no_memorise", json!(false)),
        ("passes", json!(2)),
        ("backbone_snapshot", json!("11".repeat(20))),
    ] {
        let mut f = fx("nomask");
        f.ft.as_mut().unwrap()["recipe"][key] = other;
        go(&f).refused_with(&format!("recipe.{key} is"));
        let mut f = fx("nomask");
        f.ft.as_mut().unwrap()["recipe"]
            .as_object_mut()
            .unwrap()
            .remove(key);
        go(&f).refused_with(&format!("recipe.{key} is"));
    }
    // tag: a candidate ft row tagged otherwise is not an epoch row at all.
    let mut f = fx("nomask");
    f.ft.as_mut().unwrap()["recipe"]["tag"] = json!("epoch-shuffled-label");
    go(&f).refused_with("missing row");
}

#[test]
fn an_eval_row_on_another_val_set_refuses() {
    for which in ["eval", "allgates"] {
        let mut f = fx("nomask");
        let row = if which == "eval" {
            f.eval.as_mut()
        } else {
            f.allgates.as_mut()
        };
        row.unwrap()["recipe"]["val_shard_hash"] = json!("22".repeat(32));
        go(&f).refused_with("recipe.val_shard_hash is");
    }
}

#[test]
fn the_candidates_own_key_is_required_and_the_others_refused() {
    // nomask without its key, or with the masked value.
    let mut f = fx("nomask");
    f.ft.as_mut().unwrap()["recipe"]
        .as_object_mut()
        .unwrap()
        .remove("train_attention_mask");
    go(&f).refused_with("recipe.train_attention_mask is");
    let mut f = fx("nomask");
    f.ft.as_mut().unwrap()["recipe"]["train_attention_mask"] = json!("padding");
    go(&f).refused_with("recipe.train_attention_mask is");
    // nomask carrying fused's key: each candidate is screened alone.
    let mut f = fx("nomask");
    f.ft.as_mut().unwrap()["recipe"]["optimizer_fused"] = json!(true);
    go(&f).refused_with("recipe.optimizer_fused is");
    // fused without its key.
    let mut f = fx("fused");
    f.ft.as_mut().unwrap()["recipe"]
        .as_object_mut()
        .unwrap()
        .remove("optimizer_fused");
    go(&f).refused_with("recipe.optimizer_fused is");
    // fused carrying nomask's key.
    let mut f = fx("fused");
    f.ft.as_mut().unwrap()["recipe"]["train_attention_mask"] = json!("none");
    go(&f).refused_with("recipe.train_attention_mask is");
    // fused with the masked path named explicitly is still masked.
    let mut f = fx("fused");
    f.ft.as_mut().unwrap()["recipe"]["train_attention_mask"] = json!("padding");
    go(&f).said("pass");
}

#[test]
fn a_reference_row_that_disagrees_with_the_pinned_record_refuses() {
    // 58fd1532's gate flags are pinned: a different all-gates ledger is not the reference.
    let mut rows = real_rows(ALLGATES);
    for r in rows.iter_mut() {
        if r["row_id"] == json!(ALL) {
            r["gates"]["ood_abstain"]["passed"] = json!(true);
        }
    }
    let mut f = fx("nomask");
    f.allgates_ledger = Some(rows);
    go(&f).refused_with(&format!("reference all-gates row {ALL}: gates.ood_abstain"));
    // eeda5db4's CI is pinned.
    let mut rows = real_rows(PHASE3);
    for r in rows.iter_mut() {
        if r["row_id"] == json!(CONTROL) {
            set_gate(
                r,
                &ci("+0.1501", "+0.1360", "+0.1651"),
                0.150_085_763_293_310_47,
                true,
            );
        }
    }
    let mut f = fx("nomask");
    f.phase3 = Some(rows);
    go(&f).refused_with(&format!("reference control row {CONTROL}: its CI"));
}

// --- the pre-registration -----------------------------------------------------------------------

#[test]
fn a_preregistration_that_disagrees_with_the_checker_refuses() {
    type Edit = fn(&mut Value);
    let edits: [(Edit, &str); 9] = [
        (
            |p| {
                conditional(p, "C2a")["outcome_rule"]["candidate_recipe"] =
                    json!({"train_attention_mask": "padding"})
            },
            "outcome_rule.candidate_recipe is",
        ),
        (
            |p| conditional(p, "C2a")["outcome_rule"]["checker"] = json!("qd-post-f-rules v5"),
            "outcome_rule.checker is",
        ),
        (
            |p| conditional(p, "C2a")["outcome_rule"]["words"] = json!(["pass", "fail"]),
            "outcome_rule.words is",
        ),
        (
            |p| conditional(p, "C2a")["outcome_rule"]["on_word"] = json!("fail"),
            "outcome_rule.on_word is",
        ),
        (
            |p| {
                conditional(p, "C2a")["outcome_rule"]["rule"] = json!("recipe.tierb_rule");
            },
            "outcome_rule.rule is",
        ),
        (
            |p| {
                p["recipe"]["tierb_outcome_rule"]["envelope"]["rows"][1] =
                    json!("9c27097d-1313-4f61-9ee1-0985a2dbf687")
            },
            "envelope.rows are",
        ),
        (
            |p| {
                p["recipe"]["tierb_outcome_rule"]["envelope"]["metrics"] =
                    json!(["val_top1.choice"])
            },
            "envelope.metrics are",
        ),
        (
            |p| {
                p["recipe"]
                    .as_object_mut()
                    .unwrap()
                    .remove("tierb_outcome_rule");
            },
            "has no recipe.tierb_outcome_rule object",
        ),
        (
            |p| {
                let c = &mut p["recipe"]["tierb_outcome_rule"]["clauses"]["2"];
                let text = c.as_str().unwrap().replace("+0.1651", "+0.1652");
                *c = json!(text);
            },
            "recipe.tierb_outcome_rule.clauses.2 does not say",
        ),
    ];
    for (edit, phrase) in edits {
        let mut p = real_prereg();
        edit(&mut p);
        let mut f = fx("nomask");
        f.prereg = Some(p);
        go(&f).refused_with(phrase);
    }
    // The other conditional's edit does not touch this candidate's decision, and does refuse
    // the other one.
    let mut p = real_prereg();
    conditional(&mut p, "C2b")["outcome_rule"]["candidate_recipe"] =
        json!({"optimizer_fused": false});
    let mut f = fx("nomask");
    f.prereg = Some(p.clone());
    go(&f).said("pass");
    let mut f = fx("fused");
    f.prereg = Some(p);
    go(&f).refused_with("fused outcome_rule.candidate_recipe is");
}

#[test]
fn an_identity_or_clause_text_without_its_pinned_ids_refuses() {
    for (path, id) in [
        ("identity", FT),
        (
            "identity",
            "d773b87666e1b042279271ab0f891246b7268d4ce0cad2c3e677bb415c147e1a",
        ),
        (
            "identity",
            "105513b98887a24391339f8a19ac9dfef07645abd258a22482087b0699066a72",
        ),
        ("identity", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
        ("3", ALL),
    ] {
        let mut p = real_prereg();
        let rule = &mut p["recipe"]["tierb_outcome_rule"];
        let field = if path == "identity" {
            &mut rule["rows"]["identity"]
        } else {
            &mut rule["clauses"][path]
        };
        let text = field.as_str().unwrap().replace(id, "an id");
        *field = json!(text);
        let mut f = fx("nomask");
        f.prereg = Some(p);
        let r = go(&f);
        let field = if path == "identity" {
            "rows.identity".to_string()
        } else {
            format!("clauses.{path}")
        };
        r.refused_with(&format!("recipe.tierb_outcome_rule.{field} does not say"));
        r.refused_with(id);
    }
}
