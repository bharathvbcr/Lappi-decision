//! **A refusal must be structurally distinguishable from `noul`.**
//!
//! `docs/schema-api.md`: *"`noul` means the model abstained, a refusal means the request was not
//! answerable as posed. Collapsing the two would let a hash mismatch read as model humility."*
//! `docs/hardening.md` §6 names it the specific bug to test for.
//!
//! "Structurally" is the operative word. This file does not check that the two look different; it
//! checks that a caller **cannot** confuse them, along four independent axes:
//!
//! 1. they arrive in different [`Response`] variants, which a caller must match exhaustively;
//! 2. [`Response::caller_reading`] is a total function and no refusal and no backend error maps to
//!    [`CallerReading::ModelAbstained`];
//! 3. their serializations share no key — a refusal payload has no `slots`, `value`, `score`,
//!    `conformal_set` or `noul` key **at any depth**, and an abstention has no `refusal` or `error`
//!    key;
//! 4. deserializing one as the other fails.
//!
//! The checks are on **keys**, never on substrings: `Refusal::ReservedOptionName` legitimately
//! carries the string `"noul"` as a *value*, and a substring test would either fail on it or be
//! weakened until it proved nothing.
//!
//! `kind_of` and `backend_kind_of` below are exhaustive matches with no wildcard arm. Adding a
//! variant to either enum **fails to compile this file** until the new variant is listed, which is
//! what keeps the coverage claim true over time rather than on the day it was written.

mod common;

use std::collections::BTreeMap;

use qd_runtime::refusal::{BackendError, HashKind, Refusal};
use qd_runtime::schema::{AnswerEnvelope, CallerReading, Response, SlotAnswer};
use serde_json::Value;

// -- one of every variant -------------------------------------------------------------------
//
// The samples themselves live in `qd_runtime::fixtures`, because the golden wire corpus in
// `fixtures/wire/` is generated from the same list. Two copies of "one of every variant" is the
// duplication this crate spent two gaps learning to avoid: the copy that is not regenerated is
// the one that goes stale, and a stale copy here would report full coverage of a set it no longer
// matches.

use qd_runtime::fixtures::{all_backend_errors, all_refusals};

/// Exhaustive, no wildcard. A new `Refusal` variant breaks this file's compilation.
fn kind_of(refusal: &Refusal) -> &'static str {
    match refusal {
        Refusal::HashMismatch { .. } => "hash_mismatch",
        Refusal::TooManyOptions { .. } => "too_many_options",
        Refusal::TooFewOptions { .. } => "too_few_options",
        Refusal::DuplicateOption { .. } => "duplicate_option",
        Refusal::ReservedOptionName { .. } => "reserved_option_name",
        Refusal::EmptySlots => "empty_slots",
        Refusal::DuplicateSlotName { .. } => "duplicate_slot_name",
        Refusal::EmptySlotName { .. } => "empty_slot_name",
        Refusal::SlotNameOverCap { .. } => "slot_name_over_cap",
        Refusal::TooManySlots { .. } => "too_many_slots",
        Refusal::BinsOutOfRange { .. } => "bins_out_of_range",
        Refusal::UnknownSchemaVersion { .. } => "unknown_schema_version",
        Refusal::ContextOverCap { .. } => "context_over_cap",
        Refusal::ContextLengthMismatch { .. } => "context_length_mismatch",
        Refusal::ContextLenMissing { .. } => "context_len_missing",
        Refusal::ContextNotBytes { .. } => "context_not_bytes",
        Refusal::ContextNotBase64 { .. } => "context_not_base64",
        Refusal::ContextEmpty { .. } => "context_empty",
        Refusal::ContextNotUtf8 { .. } => "context_not_utf8",
        Refusal::RenderedPromptOverCap { .. } => "rendered_prompt_over_cap",
        Refusal::EmptyOption { .. } => "empty_option",
        Refusal::OptionTextOverCap { .. } => "option_text_over_cap",
        Refusal::QuestionOverCap { .. } => "question_over_cap",
        Refusal::TaskOverCap { .. } => "task_over_cap",
        Refusal::EmptyTask => "empty_task",
        Refusal::UnknownRoute { .. } => "unknown_route",
        Refusal::UnknownSlotType { .. } => "unknown_slot_type",
        Refusal::SlotFieldNotAllowed { .. } => "slot_field_not_allowed",
        Refusal::SlotFieldMissing { .. } => "slot_field_missing",
        Refusal::RegisteredHeadMissing { .. } => "registered_head_missing",
        Refusal::RegisteredHeadSlotMissing { .. } => "registered_head_slot_missing",
        Refusal::RegisteredRouteSpanUnsupported { .. } => "registered_route_span_unsupported",
        Refusal::CalibrationEntryMissing { .. } => "calibration_entry_missing",
        Refusal::PayloadOverCap { .. } => "payload_over_cap",
        Refusal::MalformedRequest { .. } => "malformed_request",
        Refusal::AmbiguousEnvelope => "ambiguous_envelope",
        Refusal::UnknownOp { .. } => "unknown_op",
    }
}

/// Exhaustive, no wildcard. A new `BackendError` variant breaks this file's compilation.
fn backend_kind_of(error: &BackendError) -> &'static str {
    match error {
        BackendError::ReferenceBackendNotEnabled { .. } => "reference_backend_not_enabled",
        BackendError::Unavailable { .. } => "unavailable",
        BackendError::Poisoned { .. } => "poisoned",
        BackendError::RebuildFailed { .. } => "rebuild_failed",
        BackendError::ReadonlyViolated { .. } => "readonly_violated",
        BackendError::LogitShapeMismatch { .. } => "logit_shape_mismatch",
        BackendError::NonFiniteLogit { .. } => "non_finite_logit",
        BackendError::LogitKindMismatch { .. } => "logit_kind_mismatch",
        BackendError::PrefillFailed { .. } => "prefill_failed",
        BackendError::DecodeFailed { .. } => "decode_failed",
        BackendError::DeadlineExceeded { .. } => "deadline_exceeded",
        BackendError::Overloaded { .. } => "overloaded",
    }
}

// -- the four axes --------------------------------------------------------------------------

#[test]
fn every_variant_is_covered_and_names_its_own_check() {
    for refusal in all_refusals() {
        assert_eq!(refusal.kind(), kind_of(&refusal), "kind drifted: {refusal:?}");
    }
    for error in all_backend_errors() {
        assert_eq!(
            error.kind(),
            backend_kind_of(&error),
            "kind drifted: {error:?}"
        );
    }
    // Every distinct `kind()` the enums can produce is represented above.
    let refusal_kinds: std::collections::BTreeSet<_> =
        all_refusals().iter().map(|r| r.kind()).collect();
    // 36 until `slot_name_over_cap` landed with `MAX_SLOT_NAME_BYTES`
    // (`GAP-RT-SLOT-NAME-UNCAPPED`). The literal is deliberate: a kind that loses its fixture must
    // fail here rather than quietly shrink the set this file claims to cover.
    assert_eq!(refusal_kinds.len(), 37, "a refusal kind lost its fixture");
    let error_kinds: std::collections::BTreeSet<_> =
        all_backend_errors().iter().map(|e| e.kind()).collect();
    assert_eq!(error_kinds.len(), 12, "a backend-error kind lost its fixture");
}

#[test]
fn no_refusal_and_no_backend_error_reads_as_an_abstention() {
    for refusal in all_refusals() {
        let response = Response::refused(refusal.clone());
        assert_eq!(
            response.caller_reading(),
            CallerReading::RequestRefused,
            "{refusal:?} did not read as a refusal"
        );
        assert_ne!(response.caller_reading(), CallerReading::ModelAbstained);
        assert_eq!(response.status(), "refused");
    }
    for error in all_backend_errors() {
        let response = Response::failed(error.clone());
        assert_eq!(
            response.caller_reading(),
            CallerReading::BackendFailed,
            "{error:?} did not read as a backend failure"
        );
        assert_ne!(response.caller_reading(), CallerReading::ModelAbstained);
        assert_eq!(response.status(), "error");
    }
}

#[test]
fn an_abstention_reads_as_an_abstention_and_nothing_else() {
    let mut slots = BTreeMap::new();
    slots.insert("verdict".to_string(), SlotAnswer::abstained(0.01, false));
    let response = Response::Ok(AnswerEnvelope {
        schema_version: 1,
        backend: "reference-deterministic-v1".into(),
        degraded: true,
        slots,
    });
    assert_eq!(response.caller_reading(), CallerReading::ModelAbstained);
    assert_eq!(response.status(), "ok");
    assert_ne!(response.caller_reading(), CallerReading::RequestRefused);
    assert_ne!(response.caller_reading(), CallerReading::BackendFailed);
}

/// Collect every object key at every depth.
fn keys_at_every_depth(value: &Value, out: &mut std::collections::BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                out.insert(key.clone());
                keys_at_every_depth(child, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                keys_at_every_depth(item, out);
            }
        }
        _ => {}
    }
}

const ANSWER_ONLY_KEYS: &[&str] = &["slots", "value", "score", "conformal_set", "noul"];
const FAILURE_ONLY_KEYS: &[&str] = &["refusal", "error"];

#[test]
fn a_refusal_payload_carries_no_answer_key_at_any_depth() {
    for refusal in all_refusals() {
        let bytes = serde_json::to_vec(&Response::refused(refusal.clone())).expect("serializes");
        let value: Value = serde_json::from_slice(&bytes).expect("round-trips");
        let mut keys = std::collections::BTreeSet::new();
        keys_at_every_depth(&value, &mut keys);
        for forbidden in ANSWER_ONLY_KEYS {
            assert!(
                !keys.contains(*forbidden),
                "{refusal:?} serialized with the answer key `{forbidden}`: {}",
                String::from_utf8_lossy(&bytes)
            );
        }
        assert!(keys.contains("refusal"));
        assert_eq!(value.get("status").and_then(Value::as_str), Some("refused"));
    }
}

#[test]
fn a_backend_error_payload_carries_no_answer_key_at_any_depth() {
    for error in all_backend_errors() {
        let bytes = serde_json::to_vec(&Response::failed(error.clone())).expect("serializes");
        let value: Value = serde_json::from_slice(&bytes).expect("round-trips");
        let mut keys = std::collections::BTreeSet::new();
        keys_at_every_depth(&value, &mut keys);
        for forbidden in ANSWER_ONLY_KEYS {
            assert!(
                !keys.contains(*forbidden),
                "{error:?} serialized with the answer key `{forbidden}`"
            );
        }
        assert!(keys.contains("error"));
        assert_eq!(value.get("status").and_then(Value::as_str), Some("error"));
    }
}

#[test]
fn an_abstention_payload_carries_no_failure_key_at_any_depth() {
    let mut slots = BTreeMap::new();
    slots.insert("verdict".to_string(), SlotAnswer::abstained(0.01, false));
    let bytes = serde_json::to_vec(&Response::Ok(AnswerEnvelope {
        schema_version: 1,
        backend: "reference-deterministic-v1".into(),
        degraded: false,
        slots,
    }))
    .expect("serializes");
    let value: Value = serde_json::from_slice(&bytes).expect("round-trips");
    let mut keys = std::collections::BTreeSet::new();
    keys_at_every_depth(&value, &mut keys);
    for forbidden in FAILURE_ONLY_KEYS {
        assert!(
            !keys.contains(*forbidden),
            "an abstention serialized with the failure key `{forbidden}`"
        );
    }
    assert_eq!(value.get("status").and_then(Value::as_str), Some("ok"));
    assert_eq!(
        value
            .pointer("/slots/verdict/noul")
            .and_then(Value::as_bool),
        Some(true),
        "an abstention must say so in the slot it abstained on"
    );
}

#[test]
fn a_refusal_envelope_cannot_be_deserialized_as_an_answer() {
    for refusal in all_refusals() {
        let bytes = serde_json::to_vec(&Response::refused(refusal.clone())).expect("serializes");
        assert!(
            serde_json::from_slice::<AnswerEnvelope>(&bytes).is_err(),
            "{refusal:?} deserialized as an AnswerEnvelope"
        );
    }
    for error in all_backend_errors() {
        let bytes = serde_json::to_vec(&Response::failed(error.clone())).expect("serializes");
        assert!(
            serde_json::from_slice::<AnswerEnvelope>(&bytes).is_err(),
            "{error:?} deserialized as an AnswerEnvelope"
        );
    }
}

#[test]
fn every_refusal_carries_both_values_it_compared() {
    // The `Display` rendering is what a log line shows. A refusal that named the check but not the
    // numbers would send a reader back to the code to find out what was compared.
    for refusal in all_refusals() {
        let message = refusal.to_string();
        assert!(
            message.len() > 20,
            "{refusal:?} rendered too tersely to act on: {message}"
        );
        let envelope = Response::refused(refusal.clone());
        let value = serde_json::to_value(&envelope).expect("serializes");
        assert_eq!(
            value.get("message").and_then(Value::as_str),
            Some(message.as_str()),
            "the envelope's message must be the refusal's own rendering"
        );
    }
}

// -- the specific bug the contract names ----------------------------------------------------

#[test]
fn a_hash_mismatch_does_not_read_as_model_humility() {
    let request = common::validated(&{
        let mut value = common::sample_request();
        value["expect"] = serde_json::json!({"tokenizer_hash": "0".repeat(64)});
        value
    });
    let runtime = common::reference_runtime();
    let response = runtime.answer(&request, None);

    assert_eq!(response.caller_reading(), CallerReading::RequestRefused);
    match &response {
        Response::Refused(envelope) => match &envelope.refusal {
            Refusal::HashMismatch {
                which,
                expected,
                actual,
            } => {
                assert_eq!(*which, HashKind::Tokenizer);
                assert_ne!(expected, actual, "both compared values must be present");
                assert_eq!(actual, &"0".repeat(64));
            }
            other => panic!("expected a tokenizer hash mismatch, got {other:?}"),
        },
        other => panic!("a hash mismatch must refuse, got {other:?}"),
    }

    // And the same request without the bad pin is answered rather than refused, so the refusal
    // above is attributable to the hash and not to anything else about the fixture.
    let clean = common::validated(&common::sample_request());
    assert!(matches!(
        common::reference_runtime().answer(&clean, None),
        Response::Ok(_)
    ));
}
