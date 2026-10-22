//! Adversarial (X2 review): ~70 mutated span-refit ledger lines. Each case's `python` field is
//! the verdict Python's `Ledger.verify_chain` gave the bytes when the fixture was written
//! (`python/tests/adv_x2_gen.py <out.json>`): `accept`, `refuse` (the span-null refusal) or
//! `other:<exception>` (Python raised something else). Rust's `check_row_line` must agree: Ok
//! exactly when Python accepted.

mod common;

use qd_train::ledger::check_row_line;
use serde_json::Value;

#[test]
fn rust_agrees_with_python_on_every_mutated_line() {
    let path = common::crate_dir().join("tests/fixtures/adv-x2-span-refit-cases.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut diffs = Vec::new();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let py = case["python"].as_str().unwrap();
        let rust_ok = check_row_line(case["line"].as_str().unwrap()).is_ok();
        if rust_ok != (py == "accept") {
            diffs.push(format!(
                "{name}: python={py} rust={}",
                if rust_ok { "accept" } else { "refuse" }
            ));
        }
    }
    assert!(diffs.is_empty(), "Rust and Python disagree:\n{}", diffs.join("\n"));
}

/// X2 noted that `read_entries` does not apply the check. A reader that loads a ledger holding a
/// row Python's `verify_chain` refuses should not hand it back as a valid entry.
#[test]
fn read_entries_refuses_a_line_check_row_line_refuses() {
    let path = common::crate_dir().join("tests/fixtures/span-refit-gate-cases.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let line = cases
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "new_sha_missing_the_metrics")
        .unwrap()["line"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(check_row_line(&line).is_err());
    let out = std::env::temp_dir().join(format!("adv-x2-{}.jsonl", std::process::id()));
    std::fs::write(&out, format!("{line}\n")).unwrap();
    let got = qd_train::ledger::read_entries(&out, true);
    std::fs::remove_file(&out).unwrap();
    assert!(got.is_err(), "read_entries returned a row check_row_line refuses");
}
