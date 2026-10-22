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
