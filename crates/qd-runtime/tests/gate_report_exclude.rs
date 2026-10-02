//! `qd-gate-report --exclude-rows FILE`: the re-score of an eval row's val numbers with the val
//! rows that some train row overlaps (E_val) held out, and with E_val alone.
//!
//! Fable's ruling Q6 (`AUDIT/idle-gpu-queue-2026-10-02/fable-j6a-replay.md`). The 09-29 MMLU/CSQA
//! exclusion was never wired into a build, so F gold-trains rows whose 8-gram containment against
//! its own val is at least 0.5. Report-only: nothing here moves a gate (CLAUDE.md rule 2).
//!
//! FILE holds one `row_id#slot_name` key per line, the key `tools/replay_decontam.py`'s
//! `row_texts` gives a val target. Each key is matched against a verdict line's
//! `row_id + "#" + slot_name`.
//!
//! # The fixture
//!
//! [`build`] writes a scoring run deterministically (SplitMix64, as `calib_fit_roundtrip.rs`
//! does). Four val families:
//! * `knowledge.multiple_choice`: four options, with every 8th row planted.
//! * `commonsense.multiple_choice`: five options, with every 8th row planted.
//! * `code.defect_class`: four options, never planted, so an unaffected family.
//! * `qa.answer_span`: spans, with every 10th row planted.
//!
//! A *planted* row is the shape a memorised val twin takes, made as clean as possible:
//! * top-1 is the gold, by a margin no noise can close;
//! * it agrees with itself across the derangement and never abstains;
//! * its gold is spread over the options, so its marginal has a standard error.
//!
//! The other rows of the knowledge family over-predict the last option. So the full numbers sit
//! above the excluded ones in a known direction, and the excluded set's last-option deviation is
//! positive and computable here from the logits alone.
//!
//! # Byte identity without the flag
//!
//! `tests/fixtures/gate_report_exclude/no_flag.report.json` and `no_flag.stdout.txt` were written
//! by `qd-gate-report` built at `74aea95`, before `--exclude-rows` existed. That was a debug build
//! on aarch64-apple-darwin, binary sha256
//! `ca6dc0aa3a1bcd0cb1603579f56149014d05f9ed7302e99f4fd74faec9686b10`. The run is
//! [`without_the_flag_the_output_is_the_pre_change_binarys_byte_for_byte`]: [`build`]'s inputs,
//! relative paths, and the working directory set to the scratch directory. The goldens are that
//! run's outputs, copied from the scratch directory where the test keeps them. Any change to
//! [`build`] invalidates them, so this test fails until they are re-made the same way at a
//! pre-change commit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const BIN: &str = env!("CARGO_BIN_EXE_qd-gate-report");
const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gate_report_exclude"
);
const KNOWLEDGE: &str = "knowledge.multiple_choice";
const COMMONSENSE: &str = "commonsense.multiple_choice";
const DEFECT: &str = "code.defect_class";
const SPAN: &str = "qa.answer_span";
/// The families with planted rows, which the report must give three views each.
const AFFECTED: [&str; 3] = [KNOWLEDGE, COMMONSENSE, SPAN];

/// A fresh directory under the target dir's tmp, unique to this test and this run.
fn scratch(test: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after 1970")
        .as_nanos();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "gate-report-exclude-{test}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// SplitMix64: deterministic numbers without a crate.
struct Mix(u64);
impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A logit offset on a 1/16 grid in [-1.5, 1.5): bf16-like, and exact in binary.
    fn noise(&mut self) -> f64 {
        ((self.unit() * 48.0).floor() - 24.0) / 16.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `(family, slot, decode rows incl. noul, rows, plant every Nth, languages)`; `0` decode rows
/// means a span family.
type Spec = (
    &'static str,
    &'static str,
    usize,
    usize,
    usize,
    &'static [Option<&'static str>],
);
const SPECS: [Spec; 4] = [
    (KNOWLEDGE, "answer", 5, 200, 8, &[None]),
    (COMMONSENSE, "answer", 6, 160, 8, &[None]),
    (
        DEFECT,
        "defect_class",
        5,
        150,
        0,
        &[Some("python"), Some("go")],
    ),
    (SPAN, "evidence", 0, 40, 10, &[None]),
];

fn row_id(family: &str, i: usize) -> String {
    match family {
        KNOWLEDGE => format!("mmlu:{family}:validation:d{i:04x}:{i}"),
        COMMONSENSE => format!("csqa:{family}:q{i:04x}"),
        DEFECT => format!("defect:{i}"),
        _ => format!("squad:{i}"),
    }
}

/// The first maximum, as numpy's and the decode's argmax take it.
fn argmax(z: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in z.iter().enumerate() {
        if *v > z[best] {
            best = i;
        }
    }
    best
}

/// One scoring run's verdict lines and the keys of its planted rows, leaving out `drop`.
fn verdict_lines(seed: i64, eval_row: &str, drop: Option<&str>) -> (Vec<Value>, Vec<String>) {
    let mut mix = Mix(20_261_002 + seed as u64);
    let mut lines = Vec::new();
    let mut planted = Vec::new();
    for (family, slot, rows, count, every, languages) in SPECS {
        for i in 0..count {
            let id = row_id(family, i);
            let key = format!("{id}#{slot}");
            let plant = every > 0 && i % every == 0;
            let line = if rows == 0 {
                json!({
                    "eval_row_id": eval_row, "seed": seed, "kind": "span", "row_id": id,
                    "slot_name": slot, "family_id": family, "rows": 30, "noul_row": 29,
                    "gold_row": null, "top": [3, 4], "expected_abstain": false,
                    "correct": plant || mix.unit() < 0.6,
                })
            } else {
                let options = rows - 1;
                let noul = options;
                let gold = if plant {
                    (i / every) % options
                } else if family == DEFECT && mix.unit() < 0.05 {
                    noul
                } else {
                    mix.below(options)
                };
                let mut z: Vec<f64> = (0..rows).map(|_| 20.0 + mix.noise()).collect();
                if plant {
                    z[gold] += 6.0;
                    z[noul] -= 3.0;
                } else if gold == noul {
                    z[noul] += 3.0;
                } else {
                    if mix.unit() < 0.55 {
                        z[gold] += 2.0;
                    }
                    if family == KNOWLEDGE {
                        z[options - 1] += 1.0;
                    }
                    if mix.unit() < 0.06 {
                        z[noul] += 4.0;
                    } else {
                        z[noul] -= 2.0;
                    }
                }
                let top = argmax(&z);
                assert!(!plant || top == gold, "a planted row decodes to its gold");
                // perm[j] is the option first shown at j; a rotation is a derangement.
                let perm: Vec<usize> = (0..options).map(|j| (j + 1) % options).collect();
                let (top2, agreed) = if top == noul {
                    if mix.unit() < 0.5 {
                        (noul, true)
                    } else {
                        (0, false)
                    }
                } else if plant || mix.unit() < 0.8 {
                    ((top + options - 1) % options, true)
                } else {
                    (top, false)
                };
                json!({
                    "eval_row_id": eval_row, "seed": seed, "kind": "choice", "row_id": id,
                    "slot_name": slot, "family_id": family,
                    "language": languages[i % languages.len()], "rows": rows,
                    "noul_row": noul, "gold_row": gold, "top": top, "row_logits": z,
                    "correct": top == gold, "expected_abstain": gold == noul,
                    "perm": perm, "top_permuted": top2, "permutation_agreed": agreed,
                })
            };
            if drop == Some(key.as_str()) {
                continue;
            }
            if plant {
                planted.push(key);
            }
            lines.push(line);
        }
    }
    (lines, planted)
}

fn ran(k: usize, n: usize) -> Value {
    json!({"state": "ran", "passed": true, "value": k as f64 / n as f64, "n": k, "n_total": n})
}

/// The eval row these verdicts belong to: the counts the cross-check compares, computed here.
fn eval_row(eval_row: &str, seed: i64, lines: &[Value]) -> Value {
    let of = |kind: &str| -> (usize, usize) {
        let rows: Vec<&Value> = lines.iter().filter(|l| l["kind"] == kind).collect();
        let k = rows.iter().filter(|l| l["correct"] == true).count();
        (k, rows.len())
    };
    let (ck, cn) = of("choice");
    let (sk, sn) = of("span");
    let asked = lines.iter().filter(|l| l.get("perm").is_some()).count();
    let agree = lines
        .iter()
        .filter(|l| l["permutation_agreed"] == true)
        .count();
    json!({
        "row_id": eval_row, "run_kind": "eval", "status": "completed",
        "protocol": {"seed": seed}, "recipe": {"tag": "epoch-score-val"},
        "metrics": {
            "val_rows_decoded": {"state": "ran", "passed": true, "value": lines.len() as f64,
                                 "n": lines.len(), "n_total": lines.len()},
            "val_top1.choice": ran(ck, cn),
            "val_top1.span": ran(sk, sn),
        },
        "gates": {"permutation_consistency": ran(agree, asked)},
    })
}

fn jsonl(lines: &[Value]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

struct Fixture {
    dir: PathBuf,
    lines: Vec<Value>,
    planted: Vec<String>,
}

/// Seed 7's verdicts, its suite verdicts, the eval ledger and a decisions record, written into a
/// fresh scratch directory under fixed names. `second_drops` adds seed 8's verdicts without that
/// one row, and its eval row.
fn build(test: &str, second_drops: Option<&str>) -> Fixture {
    let dir = scratch(test);
    let (lines, planted) = verdict_lines(7, "fixture-eval-row-seed-7", None);
    let mut ledger = vec![eval_row("fixture-eval-row-seed-7", 7, &lines)];
    if let Some(drop) = second_drops {
        let (second, _) = verdict_lines(8, "fixture-eval-row-seed-8", Some(drop));
        ledger.push(eval_row("fixture-eval-row-seed-8", 8, &second));
        std::fs::write(dir.join("verdicts-s8.jsonl"), jsonl(&second)).expect("write");
    }
    let mut suite = Vec::new();
    for (i, bucket) in ["0-4k", "4k-8k"].iter().enumerate() {
        for j in 0..6 {
            suite.push(
                json!({"eval_row_id": "fixture-eval-row-seed-7", "gate": "needle_hunk_recall",
                              "depth_bucket": bucket, "hit": j > i}),
            );
        }
    }
    for (i, category) in ["prose", "scrambled"].iter().enumerate() {
        for j in 0..6 {
            suite.push(
                json!({"eval_row_id": "fixture-eval-row-seed-7", "gate": "ood_abstain",
                              "category": category, "abstained": j >= i}),
            );
        }
    }
    let decisions = json!({
        "schema": "qd.promotion-decisions.v1",
        "decisions": {
            "promotion_population": {
                "value": "fixture population", "families": [DEFECT], "status": "open",
                "gap": "fixture-gap-population", "source": "this test's decisions record",
            },
            "ece_population": {"status": "open", "gap": "fixture-gap-ece"},
        },
    });
    std::fs::write(dir.join("verdicts.jsonl"), jsonl(&lines)).expect("write");
    std::fs::write(dir.join("suite.jsonl"), jsonl(&suite)).expect("write");
    std::fs::write(dir.join("eval.jsonl"), jsonl(&ledger)).expect("write");
    std::fs::write(dir.join("decisions.json"), format!("{decisions}\n")).expect("write");
    Fixture {
        dir,
        lines,
        planted,
    }
}

const BASE: [&str; 10] = [
    "--verdicts",
    "verdicts.jsonl",
    "--suite-verdicts",
    "suite.jsonl",
    "--eval-ledger",
    "eval.jsonl",
    "--decisions",
    "decisions.json",
    "--out-json",
    "report.json",
];

fn run(dir: &Path, extra: &[&str]) -> Output {
    Command::new(BIN)
        .current_dir(dir)
        .args(BASE)
        .args(extra)
        .output()
        .expect("run qd-gate-report")
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "qd-gate-report refused: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn refused(dir: &Path, out: &Output) -> String {
    assert_eq!(
        out.status.code(),
        Some(2),
        "expected a refusal; stdout {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        !dir.join("report.json").exists(),
        "a refused report writes nothing"
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn report(dir: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(dir.join("report.json")).expect("report.json"))
        .expect("report is JSON")
}

fn write_keys(dir: &Path, name: &str, keys: &[String]) -> Vec<u8> {
    let bytes: Vec<u8> = keys
        .iter()
        .flat_map(|k| format!("{k}\n").into_bytes())
        .collect();
    std::fs::write(dir.join(name), &bytes).expect("write keys");
    bytes
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn golden(name: &str) -> Vec<u8> {
    let path = Path::new(FIXTURES).join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("golden {}: {e}", path.display()))
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number"))
}

fn key_of(line: &Value) -> String {
    format!(
        "{}#{}",
        line["row_id"].as_str().expect("row_id"),
        line["slot_name"].as_str().expect("slot_name")
    )
}

#[test]
fn without_the_flag_the_output_is_the_pre_change_binarys_byte_for_byte() {
    let fx = build("golden", None);
    let out = run(&fx.dir, &[]);
    ok(&out);
    std::fs::write(fx.dir.join("stdout.txt"), &out.stdout).expect("keep stdout");
    let json = std::fs::read(fx.dir.join("report.json")).expect("report.json");
    assert!(
        json == golden("no_flag.report.json"),
        "the JSON report is not the pre-change binary's; this run's is kept at {}",
        fx.dir.join("report.json").display()
    );
    assert!(
        out.stdout == golden("no_flag.stdout.txt"),
        "the text report is not the pre-change binary's; this run's is kept at {}",
        fx.dir.join("stdout.txt").display()
    );
}

#[test]
fn the_fixture_plants_what_its_doc_says() {
    let fx = build("planted-shape", None);
    let planted: BTreeSet<&String> = fx.planted.iter().collect();
    let by_family: BTreeMap<&str, usize> = fx
        .lines
        .iter()
        .filter(|l| planted.contains(&key_of(l)))
        .fold(BTreeMap::new(), |mut m, l| {
            *m.entry(l["family_id"].as_str().expect("family"))
                .or_default() += 1;
            m
        });
    assert_eq!(
        by_family,
        BTreeMap::from([(KNOWLEDGE, 25), (COMMONSENSE, 20), (SPAN, 4)])
    );
    for l in fx.lines.iter().filter(|l| planted.contains(&key_of(l))) {
        assert_eq!(l["correct"], true, "{l}");
        if l["kind"] == "choice" {
            assert_eq!(l["permutation_agreed"], true, "{l}");
            assert_ne!(l["top"], l["noul_row"], "{l}");
        }
    }
}

#[test]
fn planted_all_correct_rows_flatter_the_full_view_in_a_known_direction() {
    let fx = build("direction", None);
    let bytes = write_keys(&fx.dir, "e_val.txt", &fx.planted);
    let out = run(&fx.dir, &["--exclude-rows", "e_val.txt"]);
    ok(&out);
    let report = report(&fx.dir);
    let stated = &report["exclude_rows"];
    assert_eq!(stated["sha256"], sha256(&bytes));
    assert_eq!(stated["path"], "e_val.txt");
    assert_eq!(stated["keys"], fx.planted.len());
    let ex = &report["eval_rows"][0]["excluded_rows"];
    assert_eq!(ex["keys_matched"], fx.planted.len());
    assert_eq!(ex["absent"]["count"], 0);
    let families: Vec<&str> = ex["per_family"]
        .as_object()
        .expect("per_family")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        families,
        [COMMONSENSE, KNOWLEDGE, SPAN],
        "only the families FILE names a row of"
    );
    let mut scopes: Vec<(&str, &Value)> = AFFECTED
        .iter()
        .map(|fam| (*fam, &ex["per_family"][*fam]))
        .collect();
    scopes.push(("pooled", &ex["pooled"]));
    for (scope, views) in scopes {
        let (full, excl, only) = (&views["full"], &views["excluded"], &views["only"]);
        assert_eq!(
            f(&full["n"]),
            f(&excl["n"]) + f(&only["n"]),
            "{scope}: the views split the rows"
        );
        for kind in full["accuracy"].as_object().expect("accuracy").keys() {
            let (a, e, o) = (
                &full["accuracy"][kind],
                &excl["accuracy"][kind],
                &only["accuracy"][kind],
            );
            assert_eq!(o["value"], 1.0, "{scope} {kind}: E_val is all correct");
            assert!(
                f(&e["value"]) < f(&a["value"]),
                "{scope} {kind}: excluding the planted rows lowers top-1"
            );
            assert_eq!(f(&a["n_total"]), f(&e["n_total"]) + f(&o["n_total"]));
        }
        if scope == SPAN {
            continue;
        }
        let perm = |v: &Value| f(&v["permutation_consistency"]["value"]);
        assert_eq!(perm(only), 1.0, "{scope}");
        assert!(perm(excl) < perm(full), "{scope}");
        let abstain = |v: &Value| f(&v["ood_abstain_in_distribution"]["value"]);
        assert_eq!(abstain(only), 0.0, "{scope}");
        assert!(abstain(excl) > abstain(full), "{scope}");
    }
}

#[test]
fn the_full_view_is_the_reports_own_per_family_and_pooled_numbers() {
    let fx = build("one-owner", None);
    write_keys(&fx.dir, "e_val.txt", &fx.planted);
    ok(&run(&fx.dir, &["--exclude-rows", "e_val.txt"]));
    let report = report(&fx.dir);
    let row = &report["eval_rows"][0];
    let g1 = &row["g1"];
    let ex = &row["excluded_rows"];
    let same = |a: &Value, b: &Value, what: &str| {
        for k in ["state", "value", "n", "n_total"] {
            assert_eq!(a.get(k), b.get(k), "{what}: {k}");
        }
    };
    for fam in AFFECTED {
        let full = &ex["per_family"][fam]["full"];
        for (kind, v) in full["accuracy"].as_object().expect("accuracy") {
            same(
                v,
                &g1["accuracy"]["per_family"][fam][kind],
                &format!("{fam} accuracy {kind}"),
            );
        }
        if fam == SPAN {
            continue;
        }
        same(
            &full["permutation_consistency"],
            &g1["gates"]["permutation_consistency"]["per_family"][fam],
            fam,
        );
        same(
            &full["ood_abstain_in_distribution"],
            &g1["gates"]["ood_abstain"]["per_family"][fam]["in_distribution"],
            fam,
        );
        for (shape, v) in full["ece"].as_object().expect("ece") {
            same(v, &g1["gates"]["ece"]["per_family"][fam][shape], fam);
        }
        for (shape, v) in full["slots"].as_object().expect("slots") {
            if let Some(g3) = row["g3"]["slots"][shape]["per_family"].get(fam) {
                assert_eq!(v, g3, "{fam} {shape}: the slot diagnostics are G3's");
            }
        }
    }
    let pooled = &ex["pooled"]["full"];
    let recomputed = |gate: &str| &g1["gates"][gate]["pooled"]["recomputed_from_the_verdicts"];
    same(
        &pooled["permutation_consistency"],
        recomputed("permutation_consistency"),
        "pooled permutation",
    );
    same(
        &pooled["ood_abstain_in_distribution"],
        &recomputed("ood_abstain")["in_distribution"],
        "pooled in-distribution",
    );
    for (shape, v) in pooled["ece"].as_object().expect("ece") {
        same(v, &recomputed("ece")[format!("ece.{shape}")], "pooled ece");
    }
    for (kind, v) in pooled["accuracy"].as_object().expect("accuracy") {
        same(v, &g1["accuracy"]["pooled"][kind], "pooled accuracy");
    }
}

#[test]
fn the_last_option_deviation_is_computed_on_the_excluded_set() {
    let fx = build("last-option", None);
    write_keys(&fx.dir, "e_val.txt", &fx.planted);
    ok(&run(&fx.dir, &["--exclude-rows", "e_val.txt"]));
    let report = report(&fx.dir);
    let ex = &report["eval_rows"][0]["excluded_rows"];
    let planted: BTreeSet<&String> = fx.planted.iter().collect();
    for (fam, shape, width) in [
        (KNOWLEDGE, "choice.k4", 5usize),
        (COMMONSENSE, "choice.k5", 6),
    ] {
        let last = width - 2;
        // The oracle: the excluded rows' argmax and gold marginals, from the logits alone.
        let rows: Vec<&Value> = fx
            .lines
            .iter()
            .filter(|l| l["family_id"] == fam && !planted.contains(&key_of(l)))
            .collect();
        let n = rows.len() as f64;
        let mut gold = 0usize;
        let mut pred = 0usize;
        let mut mean_p = 0.0f64;
        for l in &rows {
            let z: Vec<f64> = l["row_logits"]
                .as_array()
                .expect("logits")
                .iter()
                .map(f)
                .collect();
            gold += usize::from(l["gold_row"] == last);
            pred += usize::from(argmax(&z) == last);
            let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
            mean_p += e[last] / e.iter().sum::<f64>();
        }
        let g = gold as f64 / n;
        let sigma = (g * (1.0 - g) / n).sqrt();
        let want = (pred as f64 / n - g) / sigma;
        let got = &ex["per_family"][fam]["excluded"]["last_option"][shape];
        assert_eq!(
            got["row"], last,
            "{fam}: the last option is the row before noul"
        );
        assert_eq!(f(&got["predicted"]["n"]), n);
        assert!(
            (f(&got["predicted"]["value"]) - want).abs() < 1e-12,
            "{fam}: {} vs {want}",
            got["predicted"]["value"]
        );
        let want_mean = (mean_p / n - g) / sigma;
        assert!(
            (f(&got["mean_probability"]["value"]) - want_mean).abs() < 1e-9,
            "{fam}: {} vs {want_mean}",
            got["mean_probability"]["value"]
        );
        if fam == KNOWLEDGE {
            assert!(
                want > 2.0,
                "the knowledge rows over-predict the last option: {want}"
            );
        }
        // E_val decodes every row to its gold, so its predicted marginal IS its gold marginal.
        let only = &ex["per_family"][fam]["only"]["last_option"][shape]["predicted"];
        assert_eq!(only["value"], 0.0, "{fam}: {only}");
    }
}

#[test]
fn a_key_absent_from_the_verdicts_is_refused_and_named() {
    let fx = build("absent", None);
    let typo = "mmlu:knowledge.multiple_choice:validation:nope:0#answer".to_string();
    let mut keys = fx.planted.clone();
    keys.push(typo.clone());
    write_keys(&fx.dir, "e_val.txt", &keys);
    let err = refused(&fx.dir, &run(&fx.dir, &["--exclude-rows", "e_val.txt"]));
    assert!(err.contains(&typo), "{err}");
    assert!(err.contains("1 of"), "{err}");
}

#[test]
fn an_absent_key_in_one_verdict_file_of_two_is_refused_for_that_file() {
    let fx0 = build("per-file", None);
    let dropped = fx0.planted[3].clone();
    let fx = build("per-file-two", Some(&dropped));
    write_keys(&fx.dir, "e_val.txt", &fx.planted);
    let two = [
        "--verdicts",
        "verdicts-s8.jsonl",
        "--exclude-rows",
        "e_val.txt",
    ];
    let err = refused(&fx.dir, &run(&fx.dir, &two));
    assert!(err.contains(&dropped), "{err}");
    assert!(err.contains("verdicts-s8.jsonl"), "{err}");
}

#[test]
fn allowed_absent_keys_are_recorded_with_their_count_list_and_sha256() {
    let fx0 = build("allow-absent-src", None);
    let dropped = fx0.planted[3].clone();
    let fx = build("allow-absent", Some(&dropped));
    write_keys(&fx.dir, "e_val.txt", &fx.planted);
    let args = [
        "--verdicts",
        "verdicts-s8.jsonl",
        "--exclude-rows",
        "e_val.txt",
        "--allow-absent-exclude-rows",
    ];
    let out = run(&fx.dir, &args);
    ok(&out);
    let report = report(&fx.dir);
    assert_eq!(report["exclude_rows"]["allow_absent"], true);
    let (s7, s8) = (
        &report["eval_rows"][0]["excluded_rows"],
        &report["eval_rows"][1]["excluded_rows"],
    );
    assert_eq!(s7["absent"]["count"], 0);
    assert_eq!(s7["keys_matched"], fx.planted.len());
    assert_eq!(s8["absent"]["count"], 1);
    assert_eq!(s8["absent"]["keys"], json!([dropped]));
    assert_eq!(
        s8["absent"]["sha256"],
        sha256(format!("{dropped}\n").as_bytes())
    );
    assert_eq!(s8["keys_matched"], fx.planted.len() - 1);
    assert_eq!(s8["pooled"]["only"]["n"], fx.planted.len() - 1);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains(&dropped),
        "the text report names the absent key"
    );
}

#[test]
fn allowing_absent_keys_still_refuses_a_file_none_of_whose_keys_match() {
    let fx = build("none-match", None);
    let keys = vec![
        "heldout:row:1#answer".to_string(),
        "typo#answer".to_string(),
    ];
    write_keys(&fx.dir, "e_val.txt", &keys);
    let err = refused(
        &fx.dir,
        &run(
            &fx.dir,
            &["--exclude-rows", "e_val.txt", "--allow-absent-exclude-rows"],
        ),
    );
    assert!(err.contains("none of"), "{err}");
}

#[test]
fn an_empty_file_is_refused_unless_explicitly_allowed() {
    let fx = build("empty", None);
    std::fs::write(fx.dir.join("e_val.txt"), b"").expect("write");
    let err = refused(&fx.dir, &run(&fx.dir, &["--exclude-rows", "e_val.txt"]));
    assert!(err.contains("--allow-empty-exclude-rows"), "{err}");
    let out = run(
        &fx.dir,
        &["--exclude-rows", "e_val.txt", "--allow-empty-exclude-rows"],
    );
    ok(&out);
    let report = report(&fx.dir);
    assert_eq!(report["exclude_rows"]["keys"], 0);
    assert_eq!(report["exclude_rows"]["sha256"], sha256(b""));
    let ex = &report["eval_rows"][0]["excluded_rows"];
    assert_eq!(ex["per_family"], json!({}), "no family is affected");
    assert_eq!(ex["pooled"]["only"]["n"], 0);
    assert_eq!(ex["pooled"]["full"], ex["pooled"]["excluded"]);
}

#[test]
fn a_malformed_file_is_refused() {
    let fx = build("malformed", None);
    let first = fx.planted[0].clone();
    for (name, body, says) in [
        ("dup", format!("{first}\n{first}\n"), "twice".to_string()),
        ("space", format!(" {first}\n"), "whitespace".to_string()),
        ("crlf", format!("{first}\r\n"), "whitespace".to_string()),
        ("blank", format!("{first}\n\n"), "blank".to_string()),
        ("nohash", "no-slot-here\n".to_string(), "#".to_string()),
    ] {
        std::fs::write(fx.dir.join("e_val.txt"), body).expect("write");
        let err = refused(&fx.dir, &run(&fx.dir, &["--exclude-rows", "e_val.txt"]));
        assert!(err.contains(&says), "{name}: {err}");
    }
}

#[test]
fn the_flag_adds_its_sections_and_changes_nothing_else() {
    let fx = build("additive", None);
    write_keys(&fx.dir, "e_val.txt", &fx.planted);
    let out = run(&fx.dir, &["--exclude-rows", "e_val.txt"]);
    ok(&out);
    let mut with: Value = report(&fx.dir);
    let without: Value =
        serde_json::from_slice(&golden("no_flag.report.json")).expect("golden is JSON");
    assert!(with["exclude_rows"].is_object());
    with.as_object_mut().expect("report").remove("exclude_rows");
    for row in with["eval_rows"]
        .as_array_mut()
        .expect("eval_rows")
        .iter_mut()
    {
        assert!(row["excluded_rows"].is_object());
        row.as_object_mut().expect("row").remove("excluded_rows");
    }
    assert_eq!(with, without);
    let gold = golden("no_flag.stdout.txt");
    assert!(
        out.stdout.starts_with(&gold),
        "the text report is the plain one plus a section"
    );
    let tail = String::from_utf8_lossy(&out.stdout[gold.len()..]).into_owned();
    assert!(tail.contains("EXCLUDED ROWS"), "{tail}");
    assert!(
        tail.contains(&sha256(
            &std::fs::read(fx.dir.join("e_val.txt")).expect("read")
        )),
        "{tail}"
    );
}
