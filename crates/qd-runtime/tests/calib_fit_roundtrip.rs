//! What `qd-calib-fit` writes is what the runtime serves.
//!
//! The fitter writes `serde_json::to_vec` of a [`CalibrationTable`]. These tests read that file
//! back the only way the runtime can take a table — `serde_json` into the type, `validate`, then
//! [`Runtime::with_backend`] — and answer requests through it. There is no serve-time file reader
//! yet (`qd serve` builds the reference table in code); the day one exists it is this
//! deserialization plus a path, and these are the properties it inherits.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use qd_runtime::calibration::{CalibrationEntry, CalibrationTable, calibrate};
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{CallerReading, DecisionRequest, Response, SlotKind};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_qd-calib-fit");

/// A fresh directory under the target dir's tmp, unique to this test and this run.
fn scratch(test: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after 1970")
        .as_nanos();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("calib-fit-{test}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// SplitMix64: deterministic logits without a crate.
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
}

/// One verdict file in `_verdict_lines`' shape: `choice` rows at 5 decode rows (four options),
/// `score` rows at 6 (five bins), and span rows, which carry no logits.
fn write_verdicts(path: &Path, eval_row: &str) {
    let mut mix = Mix(11);
    let mut out = String::new();
    for i in 0..300 {
        for (kind, rows, slot) in [("choice", 5usize, "defect_class"), ("score", 6, "severity")] {
            let gold = (mix.next() % rows as u64) as usize;
            let mut logits: Vec<f32> = (0..rows).map(|_| (mix.unit() * 4.0) as f32).collect();
            let winner = if mix.unit() < 0.8 {
                gold
            } else {
                (gold + 1) % rows
            };
            logits[winner] += 3.0;
            let top = (0..rows).fold(0, |best, j| if logits[j] > logits[best] { j } else { best });
            let line = json!({
                "eval_row_id": eval_row, "seed": 0, "row_id": format!("r{i}"), "kind": kind,
                "slot_name": slot, "correct": top == gold, "top": top, "gold_row": gold,
                "rows": rows, "noul_row": rows - 1, "language": "python",
                "row_logits": logits.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            });
            out.push_str(&line.to_string());
            out.push('\n');
        }
        let line = json!({
            "eval_row_id": eval_row, "seed": 0, "row_id": format!("r{i}"), "kind": "span",
            "slot_name": "defect_span", "correct": true, "top": [1, 2], "gold_row": null,
            "rows": 20, "noul_row": 19, "expected_abstain": false,
        });
        out.push_str(&line.to_string());
        out.push('\n');
    }
    std::fs::write(path, out).expect("write verdicts");
}

fn fit(dir: &Path, extra: &[&str]) -> Value {
    let verdicts = dir.join("verdicts.jsonl");
    write_verdicts(&verdicts, "E-roundtrip");
    let out = dir.join("fit");
    let done = Command::new(BIN)
        .arg("--verdicts")
        .arg(&verdicts)
        .args(["--name", "roundtrip-v1", "--out-dir"])
        .arg(&out)
        .args(extra)
        .output()
        .expect("run qd-calib-fit");
    assert!(
        done.status.success(),
        "qd-calib-fit failed: {}",
        String::from_utf8_lossy(&done.stderr)
    );
    let report = std::fs::read(out.join("report.json")).expect("report");
    serde_json::from_slice(&report).expect("report is JSON")
}

fn request(slots: Value) -> DecisionRequest {
    let context = b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";
    let value = json!({
        "schema_version": 1, "task": "devcouncil.verdict",
        "context_b64": qd_runtime::b64::encode(context), "context_len": context.len(),
        "question": "Does this diff implement what the commit message claims?",
        "slots": slots, "route": "generic",
    });
    match parse_line(&serde_json::to_vec(&value).expect("serializes")) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted: {other:?}"),
    }
}

fn refusal_kind(response: &Response) -> Option<&'static str> {
    match response {
        Response::Refused(envelope) => Some(envelope.refusal.kind()),
        _ => None,
    }
}

/// The value `qd-calib-fit` fitted on `margin-2026-09-30/verdicts-s0.jsonl` (`choice:5`'s
/// conformal cutoff). serde_json's default float parser read its shortest representation back
/// one ulp high (…b9d1 for …b9d0) before `float_roundtrip` was enabled for this crate, so a
/// table loaded from the fitter's file was not the table that was fitted, and its `hash()` was
/// not the file's sha256.
#[test]
fn a_fitted_number_survives_the_trip_through_json_bit_for_bit() {
    let x = f64::from_bits(0x3fef_f2fa_0476_b9d0);
    let entry = CalibrationEntry {
        temperature: 1.0,
        conformal_quantile: x,
        noul_margin: 0.0,
    };
    let table = CalibrationTable::new("ulp").with_letters(SlotKind::Choice, 5, entry);
    let bytes = serde_json::to_vec(&table).expect("serializes");
    let back: CalibrationTable = serde_json::from_slice(&bytes).expect("parses");
    let read = back
        .lookup("s", SlotKind::Choice, 5)
        .expect("entry")
        .conformal_quantile;
    assert_eq!(read.to_bits(), x.to_bits(), "wrote {x:?}, read {read:?}");
    assert_eq!(back.hash(), table.hash());
}

#[test]
fn the_fitted_file_is_the_table_the_runtime_answers_with() {
    let dir = scratch("all");
    let report = fit(&dir, &["--population", "all"]);
    let bytes = std::fs::read(dir.join("fit").join("table.json")).expect("table");

    // Hash-bound: the file's sha256 is CalibrationTable::hash() of what it holds, and the report
    // states that same hash.
    let table: CalibrationTable = serde_json::from_slice(&bytes).expect("the runtime's type");
    table.validate().expect("validates");
    let file_sha = qd_runtime::hex(&qd_runtime::sha256(&bytes));
    assert_eq!(table.hash(), file_sha);
    assert_eq!(report["tables"][0]["sha256"], json!(file_sha));
    assert_eq!(serde_json::to_vec(&table).expect("re-serializes"), bytes);

    // Every fitted entry is the one the report states, bit for bit.
    for key in ["choice:5", "score:6"] {
        let (kind, rows) = if key == "choice:5" {
            (SlotKind::Choice, 5)
        } else {
            (SlotKind::Score, 6)
        };
        let entry = table.lookup("slot", kind, rows).expect("fitted");
        let fit = &report["entries"][key]["fit"];
        assert_eq!(entry.temperature, fit["temperature"].as_f64().unwrap());
        assert_eq!(entry.noul_margin, fit["noul_margin"].as_f64().unwrap());
        assert_eq!(
            entry.conformal_quantile,
            fit["conformal_quantile"].as_f64().unwrap()
        );
        assert_eq!(
            entry.conformal_quantile,
            1.0 - fit["nonconformity_quantile"].as_f64().unwrap()
        );
        assert!(entry.temperature != 1.0, "{key} was not fitted");
        calibrate("slot", &vec![0.5f32; rows], entry).expect("calibrates");
    }
    // A shape never fitted, and span, refuse rather than default.
    assert!(table.lookup("slot", SlotKind::Choice, 3).is_err());
    assert!(table.lookup("slot", SlotKind::Span, 20).is_err());

    // Through the runtime: the fitted shape answers, an unfitted one and span refuse.
    let runtime = Runtime::with_backend(
        Arc::new(ReferenceBackend::new(true)),
        table.clone(),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("the runtime accepts the fitted table");
    assert_eq!(runtime.calibration().hash(), file_sha);
    let four = request(json!([{"name": "verdict", "type": "choice",
                                "options": ["stub", "logic", "cosmetic", "clean"]}]));
    let answered = runtime.answer(&four, None);
    assert_ne!(
        answered.caller_reading(),
        CallerReading::RequestRefused,
        "{answered:?}"
    );
    let bins = request(json!([{"name": "severity", "type": "score", "bins": 5}]));
    assert_eq!(refusal_kind(&runtime.answer(&bins, None)), None);
    let two = request(json!([{"name": "verdict", "type": "choice", "options": ["a", "b"]}]));
    assert_eq!(
        refusal_kind(&runtime.answer(&two, None)),
        Some("calibration_entry_missing")
    );
    let span = request(json!([{"name": "evidence", "type": "span"}]));
    assert_eq!(
        refusal_kind(&runtime.answer(&span, None)),
        Some("calibration_entry_missing")
    );
}

#[test]
fn two_fold_writes_both_fold_tables_and_no_pooled_one() {
    let dir = scratch("two-fold");
    let report = fit(&dir, &["--population", "two-fold", "--split-key", "k"]);
    let out = dir.join("fit");
    assert!(!out.join("table.json").exists());
    for (i, (file, name)) in [
        ("table.fold-a.json", "roundtrip-v1.fold-a"),
        ("table.fold-b.json", "roundtrip-v1.fold-b"),
    ]
    .into_iter()
    .enumerate()
    {
        let bytes = std::fs::read(out.join(file)).expect("fold table");
        let table: CalibrationTable = serde_json::from_slice(&bytes).expect("parses");
        table.validate().expect("validates");
        assert_eq!(table.name, name);
        assert_eq!(table.hash(), qd_runtime::hex(&qd_runtime::sha256(&bytes)));
        assert_eq!(report["tables"][i]["sha256"], json!(table.hash()));
        Runtime::with_backend(
            Arc::new(ReferenceBackend::new(true)),
            table,
            HeadRegistry::new(),
            RenderCaps::DEFAULT,
        )
        .expect("the runtime accepts each fold table");
    }
    let scored = &report["entries"]["choice:5"]["scored"];
    assert_eq!(scored["how"], json!("out_of_fold"));
    assert_eq!(scored["n_scored"], json!(300));
    let counts = &report["population_counts"];
    assert_eq!(
        counts["fold_a_rows"].as_u64().unwrap() + counts["fold_b_rows"].as_u64().unwrap(),
        600
    );
}

#[test]
fn an_existing_out_dir_and_a_meaningless_split_key_are_refused() {
    let dir = scratch("refusals");
    let verdicts = dir.join("verdicts.jsonl");
    write_verdicts(&verdicts, "E");
    let exists = dir.join("exists");
    std::fs::create_dir(&exists).expect("mkdir");
    for (args, needle) in [
        (
            vec!["--population", "all", "--out-dir", exists.to_str().unwrap()],
            "already exists",
        ),
        (
            vec!["--population", "all", "--split-key", "k", "--out-dir", "x"],
            "means nothing",
        ),
        (
            vec!["--population", "two-fold", "--out-dir", "y"],
            "needs a non-empty --split-key",
        ),
    ] {
        let done = Command::new(BIN)
            .arg("--verdicts")
            .arg(&verdicts)
            .args(["--name", "n"])
            .args(&args)
            .current_dir(&dir)
            .output()
            .expect("run");
        let err = String::from_utf8_lossy(&done.stderr);
        assert!(
            !done.status.success() && err.contains(needle),
            "{args:?}: {err}"
        );
    }
    assert!(!dir.join("x").exists() && !dir.join("y").exists());
}

/// An `--out-dir` whose parent does not exist is refused before any input is read. Before
/// this check the run fitted everything and then failed creating its staging directory --
/// inside `calib_fit_row.py`'s recorder block, so a fresh box would have written a `failed`
/// calibration row for a fit that was never going to land.
#[test]
fn an_out_dir_without_a_parent_is_refused_before_the_inputs_are_read() {
    let dir = scratch("no-parent");
    let done = Command::new(BIN)
        .args([
            "--verdicts",
            "never-read.jsonl",
            "--population",
            "all",
            "--name",
            "n",
        ])
        .args(["--out-dir", "missing/fit"])
        .current_dir(&dir)
        .output()
        .expect("run");
    let err = String::from_utf8_lossy(&done.stderr);
    assert!(
        !done.status.success() && err.contains("parent") && !err.contains("never-read"),
        "{err}"
    );
    assert!(!dir.join("missing").exists());
}
