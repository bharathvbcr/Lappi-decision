//! The **answer** side of `docs/schema-api.md`, asserted rather than exemplified.
//!
//! The contract specifies the request side in detail and the answer side with one example object.
//! Four audit questions went unanswered because of it, and this file is where three of them are
//! settled in code:
//!
//! * `GAP-RT-SPEC-ANSWER-ENVELOPE-FIELDS` — the envelope carries `status`, `backend` and
//!   `degraded` beyond the `schema_version` and `slots` the example shows;
//! * the field-presence rule the contract's own "what this document still does not specify"
//!   section names — **`value` is absent if and only if `noul`** — which was two independent
//!   fields with nothing asserting agreement;
//! * `GAP-RT-SPEC-SPAN-DECODE`'s companion hazard: the reserved abstain row is the **last** row of
//!   every slot kind, never the first.
//!
//! # Why the biconditional is a bug and not a nicety
//!
//! `value: Option<SlotValue>` and `noul: bool` are one fact spelled twice. This repo has now
//! produced that exact shape three times — `GAP-RT-WIRE-CONTEXT-ENCODING`,
//! `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`, and the label-set hash — and each time both suites
//! were green because neither side asserted the agreement. A parser that trusts `noul` and a
//! parser that trusts `value.is_none()` disagree on `{"value": "stub", "noul": true}`, and nothing
//! before this file rejected it.

use qd_runtime::schema::{AnswerEnvelope, Response, SlotAnswer, SlotValue};
use serde_json::json;

mod common;

// -- 1. the field-presence rule -------------------------------------------------------------------

/// A well-formed answered slot, as a JSON object the fields can be edited on.
fn answered_slot() -> serde_json::Value {
    json!({
        "value": "stub",
        "conformal_set": ["stub", "logic"],
        "score": 0.71,
        "noul": false,
        "degraded": false
    })
}

/// A well-formed abstention.
fn abstained_slot() -> serde_json::Value {
    json!({
        "value": null,
        "conformal_set": null,
        "score": 0.01,
        "noul": true,
        "degraded": false
    })
}

#[test]
fn a_slot_answer_carrying_both_a_value_and_an_abstention_is_refused() {
    let mut slot = answered_slot();
    slot["noul"] = json!(true);
    let err = serde_json::from_value::<SlotAnswer>(slot.clone()).unwrap_err();
    assert!(
        err.to_string().contains("noul"),
        "the error must name the field that contradicted the value, got: {err}"
    );

    // And the same object refused inside a whole envelope, which is the shape a caller actually
    // receives: a per-field guard that the envelope walks around is not a guard.
    let envelope = json!({
        "schema_version": 1,
        "backend": "reference-deterministic-v1",
        "degraded": true,
        "slots": {"verdict": slot}
    });
    assert!(
        serde_json::from_value::<AnswerEnvelope>(envelope).is_err(),
        "an envelope carrying a contradictory slot must be refused as a whole"
    );
}

#[test]
fn a_slot_answer_carrying_neither_a_value_nor_an_abstention_is_refused() {
    let mut slot = abstained_slot();
    slot["noul"] = json!(false);
    let err = serde_json::from_value::<SlotAnswer>(slot).unwrap_err();
    assert!(
        err.to_string().contains("noul"),
        "the error must name the field that contradicted the absent value, got: {err}"
    );
}

#[test]
fn an_unknown_field_on_the_answer_side_is_refused_the_way_the_request_side_refuses_one() {
    // `docs/schema-api.md`'s own "still does not specify" list names this: *"whether unknown
    // fields are refused or ignored — the two lanes chose differently"*. The request side refuses
    // (`wire::known_keys`); before this test the answer side silently ignored, so the two
    // directions of one protocol had opposite rules.
    let mut slot = answered_slot();
    slot["surprise"] = json!(1);
    assert!(
        serde_json::from_value::<SlotAnswer>(slot).is_err(),
        "an unknown key on a slot answer must be refused, not dropped"
    );

    let envelope = json!({
        "schema_version": 1,
        "backend": "reference-deterministic-v1",
        "degraded": false,
        "slots": {"verdict": answered_slot()},
        "surprise": 1
    });
    assert!(
        serde_json::from_value::<AnswerEnvelope>(envelope).is_err(),
        "an unknown key on the answer envelope must be refused, not dropped"
    );
}

#[test]
fn both_well_formed_shapes_still_parse() {
    // The guard must refuse exactly the two contradictory shapes and nothing else, or it has
    // broken the format it was meant to pin.
    let answered = serde_json::from_value::<SlotAnswer>(answered_slot())
        .expect("an answered slot is still a legal slot answer");
    assert_eq!(answered.value, Some(SlotValue::Choice("stub".to_string())));
    assert!(!answered.noul);

    let abstained = serde_json::from_value::<SlotAnswer>(abstained_slot())
        .expect("an abstention is still a legal slot answer");
    assert_eq!(abstained.value, None);
    assert!(abstained.noul);

    // `value` omitted entirely is the same statement as `value: null`.
    let omitted = json!({"conformal_set": null, "score": 0.01, "noul": true, "degraded": false});
    assert_eq!(
        serde_json::from_value::<SlotAnswer>(omitted).expect("an omitted value is an absent value"),
        abstained
    );
}

// -- 2. the envelope's own fields -----------------------------------------------------------------

#[test]
fn the_answer_envelope_carries_exactly_the_five_documented_keys() {
    // `GAP-RT-SPEC-ANSWER-ENVELOPE-FIELDS`. The contract's example showed `schema_version` and
    // `slots`; a caller written from it would never read `backend`, which is the field that says
    // an answer came from `reference-deterministic-v1` rather than from a model.
    let service = common::reference_service();
    let reply = service.handle_line(&common::line(&common::sample_request()));
    let value: serde_json::Value = serde_json::from_slice(&reply).expect("the reply is JSON");

    let keys: Vec<&str> = value
        .as_object()
        .expect("the reply is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        vec!["backend", "degraded", "schema_version", "slots", "status"],
        "the answer envelope's key set is part of the contract; update docs/schema-api.md in the \
         same change"
    );
    assert_eq!(value["status"], json!("ok"));
    assert_eq!(value["backend"], json!("reference-deterministic-v1"));

    let slot = &value["slots"]["verdict"];
    let slot_keys: Vec<&str> = slot
        .as_object()
        .expect("a slot answer is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        slot_keys,
        vec!["conformal_set", "degraded", "noul", "score", "value"],
        "a slot answer's key set is part of the contract"
    );
}

#[test]
fn every_answer_this_runtime_emits_satisfies_the_biconditional() {
    // The guard above is on the wire. This one is on the producer: whatever the reference backend
    // scores, no slot may reach a caller with the two fields disagreeing — which is only
    // checkable because the strict parser above now exists to check it with.
    let service = common::reference_service();
    let reply = service.handle_line(&common::line(&common::sample_request()));
    let value: serde_json::Value = serde_json::from_slice(&reply).expect("the reply is JSON");
    assert_eq!(value["status"], json!("ok"), "reply was: {value}");

    // Read as a `Response`, not as a bare `AnswerEnvelope`: `status` belongs to the enum, and the
    // envelope now refuses unknown keys. That is itself part of the contract this file pins.
    let Response::Ok(envelope) = serde_json::from_value::<Response>(value.clone())
        .expect("the runtime's own answer must survive its own strict parser")
    else {
        panic!("expected an answer, got {value}");
    };
    assert_eq!(envelope.slots.len(), 3, "one answer per requested slot");
    for (name, slot) in &envelope.slots {
        assert_eq!(
            slot.value.is_none(),
            slot.noul,
            "slot `{name}`: `value` must be absent exactly when `noul` is true"
        );
    }
}

#[test]
fn the_envelope_degraded_flag_is_mirrored_onto_every_slot() {
    // The contract describes `degraded` as a per-slot field, but its stated meaning — "the runtime
    // answered from a rebuilt-after-poison `GpuRuntime`" — is a property of the runtime. The
    // implementation resolves that by mirroring: the envelope's flag and every slot's flag are the
    // same bit. A caller may therefore read either; what it may not do is expect them to differ.
    let service = common::reference_service();
    let reply = service.handle_line(&common::line(&common::sample_request()));
    let Response::Ok(envelope) =
        serde_json::from_slice::<Response>(&reply).expect("the reply parses as an answer")
    else {
        panic!("expected an answer, got {}", String::from_utf8_lossy(&reply));
    };
    assert!(
        envelope.degraded,
        "the reference backend is not a model, so every answer it produces is degraded"
    );
    for (name, slot) in &envelope.slots {
        assert_eq!(
            slot.degraded, envelope.degraded,
            "slot `{name}`'s degraded flag must mirror the envelope's"
        );
    }
}
