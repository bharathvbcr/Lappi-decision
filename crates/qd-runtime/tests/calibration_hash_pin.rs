//! `expect.calibration_hash` pins the table the runtime calibrates with.
//!
//! `GAP-RT-CALIBRATION-HASH-PIN-BINDS-THE-BACKEND-NOT-THE-LOADED-TABLE`: `check_hashes` compared
//! the pin with the hash the backend *declares*, not with the table the runtime holds, and nothing
//! made the two agree. A runtime serving a fitted table answered a request pinning the reference
//! table's hash and refused one pinning the fitted table's. These are that measurement, kept.

use std::sync::Arc;

use qd_runtime::backend::DecisionBackend;
use qd_runtime::calibration::{CalibrationEntry, CalibrationTable};
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::refusal::{HashKind, Refusal};
use qd_runtime::registry::HeadRegistry;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{DecisionRequest, Response, SlotKind};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::json;

/// A table that is not the reference one: only `choice:5`, with fitted-looking numbers.
fn fitted() -> CalibrationTable {
    CalibrationTable::new("fitted-under-test").with_letters(
        SlotKind::Choice,
        5,
        CalibrationEntry {
            temperature: 1.37,
            conformal_quantile: 0.11,
            noul_margin: 0.03,
        },
    )
}

fn runtime_serving(table: CalibrationTable) -> Runtime {
    Runtime::with_backend(
        Arc::new(ReferenceBackend::new(true)),
        table,
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("assembles")
}

fn pinning(calibration_hash: &str) -> DecisionRequest {
    let context = b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";
    let value = json!({
        "schema_version": 1, "task": "devcouncil.verdict",
        "context_b64": qd_runtime::b64::encode(context), "context_len": context.len(),
        "question": "Does this diff implement what the commit message claims?",
        "slots": [{"name": "verdict", "type": "choice",
                   "options": ["stub", "logic", "cosmetic", "clean"]}],
        "route": "generic",
        "expect": {"calibration_hash": calibration_hash},
    });
    match parse_line(&serde_json::to_vec(&value).expect("serializes")) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted: {other:?}"),
    }
}

#[test]
fn the_pin_of_the_table_being_served_is_answered() {
    let table = fitted();
    let runtime = runtime_serving(table.clone());
    // The two hashes differ, or this test could not tell the two quantities apart.
    assert_ne!(table.hash(), runtime.identity().calibration_hash);
    match runtime.answer(&pinning(&table.hash()), None) {
        Response::Ok(_) => {}
        other => panic!("a pin of the table that is calibrating must be answered, got {other:?}"),
    }
}

#[test]
fn the_pin_of_a_table_that_is_not_being_served_is_refused() {
    let table = fitted();
    let runtime = runtime_serving(table.clone());
    let declared = ReferenceBackend::new(true)
        .identity()
        .calibration_hash
        .clone();
    assert_eq!(declared, CalibrationTable::reference().hash());
    match runtime.answer(&pinning(&declared), None) {
        Response::Refused(envelope) => assert_eq!(
            envelope.refusal,
            Refusal::HashMismatch {
                which: HashKind::Calibration,
                expected: table.hash(),
                actual: declared,
            }
        ),
        other => panic!(
            "a pin of the reference table, while a fitted one calibrates, was not refused: \
             {other:?}"
        ),
    }
}
