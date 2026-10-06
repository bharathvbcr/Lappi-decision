//! `caller_record::validate_line` against `fixtures/caller/valid-records.jsonl`, and against every
//! way a calling app could get a record wrong.
//!
//! Each adversarial case mutates one field of a known-valid record, so a case that passes for the
//! wrong reason (the record was invalid somewhere else) cannot hide: the unmutated record is
//! asserted valid first.

use std::path::Path;

use qd_runtime::caller_record::{
    check_store_path, default_store_root, validate_line, RecordError, RecordKind,
    MAX_RECORD_BYTES, READINGS, STORE_RELATIVE_TO_HOME,
};
use serde_json::{json, Value};

fn fixture_lines() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/caller/valid-records.jsonl");
    let text = std::fs::read_to_string(&path).expect("the caller fixture exists");
    let lines: Vec<String> = text.lines().map(str::to_owned).collect();
    assert!(lines.len() >= 5, "the fixture carries at least five records, got {}", lines.len());
    lines
}

fn decision() -> Value {
    serde_json::from_str(&fixture_lines()[0]).expect("fixture line 0 parses")
}

fn check(v: &Value) -> Result<qd_runtime::caller_record::RecordSummary, RecordError> {
    validate_line(serde_json::to_string(v).expect("serializes").as_bytes())
}

fn refused(v: &Value) -> RecordError {
    match check(v) {
        Ok(s) => panic!("expected a refusal, the record validated as {s:?}: {v}"),
        Err(e) => e,
    }
}

#[test]
fn every_fixture_record_is_valid_and_outcomes_pair_with_decisions() {
    let lines = fixture_lines();
    let summaries: Vec<_> = lines
        .iter()
        .map(|l| validate_line(l.as_bytes()).unwrap_or_else(|e| panic!("{e}: {l}")))
        .collect();
    let outcome = summaries.iter().find(|s| s.kind == RecordKind::Outcome).expect("an outcome");
    assert!(
        summaries
            .iter()
            .any(|s| s.kind == RecordKind::Decision && s.record_id == outcome.record_id),
        "the fixture's outcome pairs with its decision"
    );
    // Every reading but one that needs no fixture is exercised by a real line.
    for reading in ["request_refused", "not_asked", "model_answered", "unavailable"] {
        assert!(summaries.iter().any(|s| s.reading.as_deref() == Some(reading)), "{reading}");
    }
}

#[test]
fn admission_has_exactly_one_legal_value() {
    for got in ["admitted", "train", "eval", "", "NOT_ADMITTED"] {
        let mut v = decision();
        v["admission"] = json!(got);
        assert!(
            matches!(refused(&v), RecordError::Inconsistent { ref path, .. } if path == "$.admission"),
            "{got:?}"
        );
    }
}

#[test]
fn unknown_and_missing_keys_are_refused_at_every_level() {
    let mut v = decision();
    v["consent"] = json!("yes");
    assert!(matches!(refused(&v), RecordError::UnknownKey { .. }));

    let mut v = decision();
    v["lappi"]["confidence"] = json!(0.9);
    assert!(matches!(refused(&v), RecordError::UnknownKey { ref path, .. } if path == "$.lappi"));

    let mut v = decision();
    v["redaction"]["extra"] = json!(1);
    assert!(matches!(refused(&v), RecordError::UnknownKey { ref path, .. } if path == "$.redaction"));

    for key in ["record_id", "facts", "app_choice", "lappi", "redaction", "created_at"] {
        let mut v = decision();
        v.as_object_mut().unwrap().remove(key);
        assert!(matches!(refused(&v), RecordError::MissingKey { .. }), "{key}");
    }
}

#[test]
fn an_outcome_cannot_carry_decision_fields_or_skip_observed() {
    let lines = fixture_lines();
    let outcome: Value = serde_json::from_str(&lines[1]).unwrap();
    check(&outcome).expect("fixture outcome is valid");
    let mut v = outcome.clone();
    v["lappi"] = decision()["lappi"].clone();
    assert!(matches!(refused(&v), RecordError::UnknownKey { .. }));
    let mut v = outcome;
    v.as_object_mut().unwrap().remove("observed");
    assert!(matches!(refused(&v), RecordError::MissingKey { .. }));
}

#[test]
fn identifiers_and_timestamps_are_closed_shapes() {
    let bad_ids = [
        "0F3C9A7E5B2D4C18A9E6F1B0C7D2E4A5",
        "0f3c9a7e5b2d4c18a9e6f1b0c7d2e4a",
        "0f3c9a7e5b2d4c18a9e6f1b0c7d2e4a5z",
        "g0000000000000000000000000000000",
        "",
    ];
    for id in bad_ids {
        let mut v = decision();
        v["record_id"] = json!(id);
        assert!(matches!(refused(&v), RecordError::Invalid { .. }), "{id:?}");
    }
    for app in ["GitPulse", "1app", "", "a b", "x".repeat(33).as_str()] {
        let mut v = decision();
        v["app"] = json!(app);
        assert!(matches!(refused(&v), RecordError::Invalid { .. }), "{app:?}");
    }
    for point in ["Gitpulse.type", "gitpulse..type", ".x", "x.", "a.1b"] {
        let mut v = decision();
        v["decision_point"] = json!(point);
        assert!(matches!(refused(&v), RecordError::Invalid { .. }), "{point:?}");
    }
    let bad_times = [
        "2026-10-06 15:04:05Z",
        "2026-10-06T15:04:05+00:00",
        "2026-13-06T15:04:05Z",
        "2026-10-06T24:04:05Z",
        "2026-10-06T15:04:05.Z",
        "2026-10-06T15:04:05",
        "",
    ];
    for t in bad_times {
        let mut v = decision();
        v["created_at"] = json!(t);
        assert!(matches!(refused(&v), RecordError::Invalid { .. }), "{t:?}");
    }
    let mut v = decision();
    v["record_version"] = json!(2);
    assert!(matches!(refused(&v), RecordError::Invalid { .. }));
    v["record_version"] = json!("1");
    assert!(matches!(refused(&v), RecordError::Invalid { .. }));
}

#[test]
fn the_lappi_block_must_be_internally_consistent() {
    // asked=false but a reading that implies a request.
    let mut v = decision();
    v["lappi"]["asked"] = json!(false);
    assert!(matches!(refused(&v), RecordError::Inconsistent { .. }));

    // A refusal with no kind.
    let mut v = decision();
    v["lappi"]["kind"] = Value::Null;
    assert!(matches!(refused(&v), RecordError::Inconsistent { .. }));

    // A refusal that also carries slots.
    let mut v = decision();
    v["lappi"]["slots"] = json!({});
    assert!(matches!(refused(&v), RecordError::Inconsistent { .. }));

    // An answer with no backend.
    let mut v = decision();
    v["lappi"]["reading"] = json!("model_answered");
    v["lappi"]["kind"] = Value::Null;
    v["lappi"]["slots"] = json!({"commit_type": {}});
    assert!(matches!(refused(&v), RecordError::Inconsistent { .. }));
    v["lappi"]["backend"] = json!("reference-deterministic-v1");
    check(&v).expect("an answer with slots and a backend is consistent");

    // A sent request that does not name its task.
    let mut v = decision();
    v["lappi"]["task"] = Value::Null;
    assert!(matches!(refused(&v), RecordError::Inconsistent { .. }));

    // Not asked, but carrying a reply field.
    let mut v = decision();
    v["lappi"] = json!({"asked": false, "task": null, "reading": "not_asked", "kind": "x",
                        "backend": null, "slots": null, "latency_ms": null});
    assert!(matches!(refused(&v), RecordError::Inconsistent { .. }));

    // A reading outside the closed set.
    let mut v = decision();
    v["lappi"]["reading"] = json!("timeout");
    assert!(matches!(refused(&v), RecordError::Invalid { .. }));
    assert_eq!(READINGS.len(), 6);

    let mut v = decision();
    v["lappi"]["latency_ms"] = json!(-1);
    assert!(matches!(refused(&v), RecordError::Invalid { .. }));
}

#[test]
fn a_secret_shaped_string_anywhere_refuses_the_line() {
    let secrets = [
        "ghp_0123456789abcdefABCDEF0123456789abcd",
        "token=sk-ant-api03-abcdefghijklmnop",
        "AKIAIOSFODNN7EXAMPLE",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "xoxb-123456789012-abcdefghijkl",
    ];
    for secret in secrets {
        let mut v = decision();
        v["facts"]["note"] = json!(secret);
        assert!(matches!(refused(&v), RecordError::SecretLike { .. }), "{secret}");

        // In an array element, and as an object key.
        let mut v = decision();
        v["app_choice"]["list"] = json!(["ok", secret]);
        assert!(matches!(refused(&v), RecordError::SecretLike { .. }), "{secret} in array");
        let mut v = decision();
        v["facts"][secret] = json!(1);
        assert!(matches!(refused(&v), RecordError::SecretLike { .. }), "{secret} as key");
    }
    // Not credentials: a prefix inside a word, or one with no body.
    for benign in ["MASKAKIA0123456789ABCD", "ghp_", "the sk- prefix", "AKIA"] {
        let mut v = decision();
        v["facts"]["note"] = json!(benign);
        check(&v).unwrap_or_else(|e| panic!("{benign:?} is not a credential: {e}"));
    }
}

#[test]
fn bytes_that_are_not_one_json_object_are_refused() {
    assert!(matches!(validate_line(b"\xff\xfe"), Err(RecordError::NotUtf8)));
    assert!(matches!(validate_line(b"[1,2]"), Err(RecordError::NotJsonObject(_))));
    assert!(matches!(validate_line(b""), Err(RecordError::NotJsonObject(_))));
    assert!(matches!(validate_line(b"{\"a\":1} {\"b\":2}"), Err(RecordError::NotJsonObject(_))));
    let mut v = decision();
    v["facts"]["pad"] = json!("x".repeat(MAX_RECORD_BYTES));
    assert!(matches!(refused(&v), RecordError::TooLarge { .. }));
}

#[test]
fn the_store_is_under_a_held_out_segment_and_nothing_else_is() {
    assert!(STORE_RELATIVE_TO_HOME.split('/').any(|s| s == "heldout"));
    check_store_path(Path::new("/Users/x/Library/Application Support/Lappi/heldout/caller-records/gitpulse/2026-10-06.jsonl"))
        .expect("the documented store is held out");
    check_store_path(Path::new("/tmp/HeldOut/a.jsonl")).expect("markers are case-folded");
    for bad in [
        "/Users/x/Code/research/Lappi-decision/data/pool/a.jsonl",
        "/tmp/withheld_outputs/a.jsonl",
        "relative/a.jsonl",
    ] {
        assert!(
            matches!(check_store_path(Path::new(bad)), Err(RecordError::NotHeldOutPath { .. })),
            "{bad}"
        );
    }
    if let Some(root) = default_store_root() {
        assert!(root.ends_with(STORE_RELATIVE_TO_HOME));
        check_store_path(&root.join("devtype/2026-10-06.jsonl")).expect("the default root is held out");
    }
}
