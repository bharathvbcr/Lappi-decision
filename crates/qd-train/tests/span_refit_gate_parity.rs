//! Rust's half of the span-refit span-null check's cross-language parity: the same row bytes
//! (`tests/fixtures/span-refit-gate-cases.json`, written by `tools/gen_span_refit_gate_cases.py`)
//! go to `qd_train::ledger::check_row_line` here and to `Ledger.append` / `Ledger.verify_chain`
//! in `python/tests/test_span_refit_gate_parity.py`; both must give each case's `expect`.

mod common;

use qd_train::ledger::{SPAN_REFIT_SPAN_NULL_SCORER_SHAS, check_row_line};
use serde_json::Value;

fn cases() -> Vec<Value> {
    let path = common::crate_dir().join("tests/fixtures/span-refit-gate-cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str::<Value>(&text)
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn rust_gives_every_fixture_line_its_expected_verdict() {
    let cases = cases();
    assert!(cases.len() >= 11, "the fixture lost cases: {}", cases.len());
    let (mut accepted, mut refused) = (0, 0);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let verdict = check_row_line(case["line"].as_str().unwrap());
        match case["expect"].as_str().unwrap() {
            "accept" => {
                verdict.unwrap_or_else(|e| panic!("{name}: Rust refused what Python accepts: {e}"));
                accepted += 1;
            }
            "refuse" => {
                assert!(verdict.is_err(), "{name}: Rust accepted what Python refuses");
                refused += 1;
            }
            other => panic!("{name}: expect {other:?}"),
        }
    }
    assert!(accepted >= 5 && refused >= 6, "{accepted} accept, {refused} refuse");
}

#[test]
fn the_named_cases_have_the_expected_verdicts() {
    let by_name = |n: &str| -> Value {
        cases()
            .into_iter()
            .find(|c| c["name"] == n)
            .unwrap_or_else(|| panic!("no case {n}"))
    };
    for (name, want) in [
        ("new_sha_valid", "accept"),
        ("new_sha_missing_the_metrics", "refuse"),
        ("new_sha_counts_disagree", "refuse"),
        ("arm_a_after_validity_gate_failed", "accept"),
        ("old_sha_without_metrics", "accept"),
    ] {
        assert_eq!(by_name(name)["expect"], want, "{name}");
    }
}

#[test]
fn the_fixtures_new_sha_is_the_one_rust_names() {
    let line = cases()
        .into_iter()
        .find(|c| c["name"] == "new_sha_valid")
        .unwrap()["line"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(SPAN_REFIT_SPAN_NULL_SCORER_SHAS.iter().all(|s| line.contains(s)));
}
