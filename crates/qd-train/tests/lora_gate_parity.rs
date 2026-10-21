//! Rust's half of the `lora` label gate's cross-language parity: the same row bytes
//! (`tests/fixtures/lora-gate-cases.json`, written by `tools/gen_lora_gate_cases.py`) go to
//! `qd_train::ledger::check_row_line` here and to `Ledger.append` / `Ledger.verify_chain` in
//! `python/tests/test_lora_gate_parity.py`; both must give each case's `expect`. A case the two
//! gates would split on is a failing test on whichever side moved.

mod common;

use qd_train::ledger::check_row_line;
use serde_json::Value;

fn cases() -> Vec<Value> {
    let path = common::crate_dir().join("tests/fixtures/lora-gate-cases.json");
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
    assert!(cases.len() >= 10, "the fixture lost cases: {}", cases.len());
    let mut accepted = 0;
    let mut refused = 0;
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let line = case["line"].as_str().unwrap();
        let verdict = check_row_line(line);
        match case["expect"].as_str().unwrap() {
            "accept" => {
                verdict.unwrap_or_else(|e| panic!("{name}: Rust refused what Python accepts: {e}"));
                accepted += 1;
            }
            "refuse" => {
                assert!(
                    verdict.is_err(),
                    "{name}: Rust accepted what Python refuses"
                );
                refused += 1;
            }
            other => panic!("{name}: expect {other:?}"),
        }
    }
    assert!(accepted >= 3 && refused >= 8, "{accepted} accept, {refused} refuse");
}

#[test]
fn the_three_named_cases_are_present_with_the_expected_verdicts() {
    let by_name = |n: &str| -> Value {
        cases()
            .into_iter()
            .find(|c| c["name"] == n)
            .unwrap_or_else(|| panic!("no case {n}"))
    };
    assert_eq!(by_name("valid_rust_named")["expect"], "accept");
    assert_eq!(by_name("init_named")["expect"], "refuse");
    assert_eq!(by_name("mixed_rust_and_init_names")["expect"], "refuse");
}
