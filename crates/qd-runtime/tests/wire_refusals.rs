//! Every refusal in `docs/schema-api.md`'s table, driven from the wire.
//!
//! `docs/hardening.md` §6: *"Every refusal in `docs/schema-api.md` has a test asserting a typed
//! error, and asserting it is distinguishable from `noul`."* The second half lives in
//! `refusal_is_not_noul.rs`, which covers every variant of the enum; this file covers the *path* —
//! that a caller sending a particular malformed request gets that particular typed refusal, rather
//! than a generic parse error or, worse, an answer.
//!
//! Every case here also asserts the reply's `status` is `refused`, so none of them can pass by
//! accidentally producing a backend error instead.

mod common;

use std::sync::Arc;

use qd_runtime::backend::DecisionBackend;
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::reference::{ReferenceBackend, REFERENCE_FEATURE_DIM};
use qd_runtime::refusal::Refusal;
use qd_runtime::registry::{HeadMatrix, HeadRegistry, RegisteredHead};
use qd_runtime::render::{RenderCaps, ESCAPE_WORST_CASE_GROWTH};
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{CallerReading, Response, MAX_OPTIONS, MAX_SLOT_NAME_BYTES};
use qd_runtime::service::Service;
use qd_runtime::wire::{
    parse_line, payload_budget, JSON_ESCAPE_WORST_CASE_GROWTH, MAX_PAYLOAD_BYTES, MAX_SLOTS,
};
use serde_json::{json, Value};

/// Send a wire value through the service and assert the refusal it earns.
#[track_caller]
fn assert_refused(what: &str, value: Value, expected_kind: &str) {
    let service = common::reference_service();
    let reply = service.handle_line(&common::line(&value));
    assert_eq!(
        common::status_of(&reply),
        "refused",
        "{what}: expected a refusal, got {}",
        String::from_utf8_lossy(&reply)
    );
    assert_eq!(
        common::refusal_kind(&reply),
        expected_kind,
        "{what}: wrong refusal, reply was {}",
        String::from_utf8_lossy(&reply)
    );
}

/// The **typed** refusal a wire value earns, for cases that assert on its fields and not only on
/// its kind. Goes through the same service entry point as [`assert_refused`], so a case using this
/// still proves the reply is a refusal envelope rather than an answer.
#[track_caller]
fn refuse(value: Value) -> Refusal {
    let service = common::reference_service();
    let reply = service.handle_line(&common::line(&value));
    match serde_json::from_slice::<Response>(&reply) {
        Ok(Response::Refused(envelope)) => envelope.refusal,
        other => panic!(
            "expected a refusal envelope, got {other:?} from {}",
            String::from_utf8_lossy(&reply)
        ),
    }
}

#[track_caller]
fn assert_refused_bytes(what: &str, payload: &[u8], expected_kind: &str) {
    let service = common::reference_service();
    let reply = service.handle_line(payload);
    assert_eq!(common::status_of(&reply), "refused", "{what}");
    assert_eq!(
        common::refusal_kind(&reply),
        expected_kind,
        "{what}: reply was {}",
        String::from_utf8_lossy(&reply)
    );
}

fn with_slots(slots: Value) -> Value {
    common::request_with_slots(common::SAMPLE_CONTEXT.as_bytes(), slots)
}

fn one_choice() -> Value {
    json!([{"name": "verdict", "type": "choice", "options": ["stub", "clean"]}])
}

// -- the contract's own table -------------------------------------------------------------------

#[test]
fn schema_version_unknown() {
    let mut value = common::sample_request();
    value["schema_version"] = json!(2);
    assert_refused("a future version", value, "unknown_schema_version");

    let mut value = common::sample_request();
    value["schema_version"] = json!("1");
    assert_refused("a stringly version", value, "unknown_schema_version");
}

#[test]
fn options_over_sixteen() {
    let options: Vec<String> = (0..17).map(|i| format!("option-{i}")).collect();
    assert_refused(
        "17 options",
        with_slots(json!([{"name": "verdict", "type": "choice", "options": options}])),
        "too_many_options",
    );
}

#[test]
fn options_below_two() {
    assert_refused(
        "one option",
        with_slots(json!([{"name": "verdict", "type": "choice", "options": ["stub"]}])),
        "too_few_options",
    );
    assert_refused(
        "no options",
        with_slots(json!([{"name": "verdict", "type": "choice", "options": []}])),
        "too_few_options",
    );
}

#[test]
fn context_over_the_cap() {
    let big = vec![b'x'; RenderCaps::DEFAULT.max_context_bytes + 1];
    assert_refused(
        "an oversized context",
        common::request_with_slots(&big, one_choice()),
        "context_over_cap",
    );
}

#[test]
fn slots_empty_or_duplicated() {
    assert_refused("no slots", with_slots(json!([])), "empty_slots");
    assert_refused(
        "two slots with one name",
        with_slots(json!([
            {"name": "verdict", "type": "choice", "options": ["stub", "clean"]},
            {"name": "verdict", "type": "span"}
        ])),
        "duplicate_slot_name",
    );
    let many: Vec<Value> = (0..33)
        .map(|i| json!({"name": format!("slot-{i}"), "type": "span"}))
        .collect();
    assert_refused("33 slots", with_slots(json!(many)), "too_many_slots");
}

#[test]
fn bins_out_of_range() {
    for bins in [0, 1, 17, 1000] {
        assert_refused(
            &format!("bins = {bins}"),
            with_slots(json!([{"name": "severity", "type": "score", "bins": bins}])),
            "bins_out_of_range",
        );
    }
}

// -- `context_b64` + `context_len`: exactly one form -----------------------------------------------

#[test]
fn the_retired_byte_array_context_is_refused_rather_than_read() {
    // The form this crate accepted until 2026-09-19. A lenient read of it is what would make the
    // two encodings coexist, and "which one wins" is how a context silently becomes a different
    // context. `GAP-RT-WIRE-CONTEXT-ENCODING`.
    let mut value = common::sample_request();
    if let Some(object) = value.as_object_mut() {
        object.remove("context_b64");
        object.insert("context".to_string(), json!([104, 105, 33]));
        object.insert("context_len".to_string(), json!(3));
    }
    let refusal = refuse(value);
    // The same identifier `python/qd_data/errors.py::ContextNotBytesRefusal.check` carries, so a
    // caller branching on it gets one answer from both lanes.
    assert_eq!(refusal.kind(), "context_not_bytes");
    let message = refusal.to_string();
    assert!(
        message.contains("context_b64") && message.contains("context_len"),
        "the refusal must name the fields that replaced it: {message}"
    );
}

#[test]
fn a_context_that_is_not_a_json_string_is_refused_rather_than_coerced() {
    for (label, bad) in [
        ("an integer array", json!([104, 105, 33])),
        ("a number", json!(17)),
        ("an object", json!({"bytes": []})),
        ("null", json!(null)),
    ] {
        let mut value = common::sample_request();
        value["context_b64"] = bad;
        assert_refused(label, value, "context_not_bytes");
    }

    // Absent entirely is the same check, and names itself.
    let mut value = common::sample_request();
    if let Some(object) = value.as_object_mut() {
        object.remove("context_b64");
    }
    let refusal = refuse(value);
    assert_eq!(refusal.kind(), "context_not_bytes");
    assert!(refusal.to_string().contains("absent"), "{refusal}");
}

#[test]
fn base64_that_is_not_canonical_is_refused() {
    for (label, blob, len) in [
        ("a character outside the alphabet", "Zm9v*g==", 4),
        ("whitespace inside the blob", "Zm9 v", 4),
        ("the URL-safe alphabet", "Zm9-", 3),
        ("missing padding", "Zm9vYg", 4),
        ("non-zero trailing bits", "Zh==", 1),
    ] {
        let mut value = common::sample_request();
        value["context_b64"] = json!(blob);
        value["context_len"] = json!(len);
        assert_refused(label, value, "context_not_base64");
    }
}

#[test]
fn a_declared_length_that_disagrees_is_refused_and_names_both_numbers() {
    let mut value = common::sample_request();
    value["context_len"] = json!(999);
    let refusal = refuse(value);
    assert_eq!(refusal.kind(), "context_length_mismatch");
    match refusal {
        Refusal::ContextLengthMismatch { declared, actual } => {
            assert_eq!(declared, 999);
            assert_eq!(actual, common::SAMPLE_CONTEXT.len());
            // The signature this field exists to catch: a payload truncated to a length that still
            // base64-decodes cleanly.
            let message = Refusal::ContextLengthMismatch { declared, actual }.to_string();
            assert!(message.contains("999"), "{message}");
            assert!(message.contains(&actual.to_string()), "{message}");
        }
        other => panic!("expected a length mismatch, got {other:?}"),
    }
}

#[test]
fn a_truncated_payload_that_still_decodes_is_caught_by_the_length() {
    // Why `context_len` is kept although base64 makes it derivable. Chop whole quanta off the
    // encoded blob: what is left is still valid base64 and decodes to a shorter context, which
    // nothing but the declared length would notice.
    let full = qd_runtime::b64::encode(common::SAMPLE_CONTEXT.as_bytes());
    let truncated = &full[..full.len() - 4];
    let mut value = common::sample_request();
    value["context_b64"] = json!(truncated);
    // `context_len` still claims the original size.
    assert_refused(
        "a payload truncated by one quantum",
        value,
        "context_length_mismatch",
    );
}

#[test]
fn a_missing_length_is_refused() {
    let mut value = common::sample_request();
    if let Some(object) = value.as_object_mut() {
        object.remove("context_len");
    }
    let refusal = refuse(value);
    assert_eq!(refusal.kind(), "context_len_missing");
    assert!(
        refusal.to_string().contains("no default"),
        "a missing length must not read as zero: {refusal}"
    );
}

#[test]
fn a_length_that_is_not_a_non_negative_integer_is_refused() {
    // Absent and "present but unusable" are one condition under one identifier — no declared
    // length arrived that the check could use — matching `ContextLenMissingRefusal` on the Python
    // side. The `found` field is what keeps them apart in the message.
    for bad in [json!(-1), json!(1.5), json!("12"), json!(null), json!([3])] {
        let mut value = common::sample_request();
        value["context_len"] = bad.clone();
        let refusal = refuse(value);
        assert_eq!(refusal.kind(), "context_len_missing", "for {bad}");
        assert!(
            !refusal.to_string().contains("absent"),
            "a present-but-wrong length must not report itself absent: {refusal}"
        );
    }
}

#[test]
fn an_empty_context_carries_an_empty_blob_and_a_zero_length() {
    // The empty context is a legal *encoding* — it is refused downstream for having nothing to
    // point at, not by the codec. The distinction matters: `context_empty` is a property of the
    // request, `context_not_base64` would be a property of the transport.
    let mut value = common::sample_request();
    value["context_b64"] = json!("");
    value["context_len"] = json!(0);
    assert_refused("the empty context", value, "context_empty");
}

#[test]
fn an_empty_or_whitespace_context_is_refused() {
    assert_refused(
        "an all-whitespace context",
        common::request_with_slots(b"   \n\t\n", one_choice()),
        "context_empty",
    );
}

#[test]
fn a_context_that_is_not_utf8_is_refused() {
    assert_refused(
        "a lone 0xFF",
        common::request_with_slots(&[b'a', 0xff, b'b'], one_choice()),
        "context_not_utf8",
    );
}

// -- caps that refuse rather than truncate ---------------------------------------------------------

#[test]
fn an_oversized_question_task_or_option_is_refused() {
    let mut value = with_slots(one_choice());
    value["question"] = json!("q".repeat(RenderCaps::DEFAULT.max_question_bytes + 1));
    assert_refused("a long question", value, "question_over_cap");

    let mut value = with_slots(one_choice());
    value["task"] = json!("t".repeat(RenderCaps::DEFAULT.max_task_bytes + 1));
    assert_refused("a long task", value, "task_over_cap");

    let long = "o".repeat(RenderCaps::DEFAULT.max_option_bytes + 1);
    assert_refused(
        "a long option",
        with_slots(json!([{"name": "verdict", "type": "choice", "options": [long, "clean"]}])),
        "option_text_over_cap",
    );
}

#[test]
fn a_maximal_nul_dense_context_is_no_longer_refused_by_arithmetic() {
    // The regression `GAP-RT-ESCAPE-GROWTH-FLOOR` named. NUL is not whitespace and not a line
    // break, but it renders as the six bytes `\u0000`, so a context at exactly
    // `max_context_bytes` escapes to 6x its size. While this lane still computed the floor at
    // 2x, `max_rendered_bytes` was 393216 and such a context was refused by
    // `rendered_prompt_over_cap` -- a legal request rejected by arithmetic, and rejected by
    // only one of the two lanes, since the Python side had already moved to 6x / 802816.
    //
    // It must now be answered. This test fails against the pre-fix constants.
    let dense = vec![0u8; RenderCaps::DEFAULT.max_context_bytes];
    let request = common::request_with_slots(&dense, one_choice());
    let service = common::reference_service();
    let reply = service.handle_line(&common::line(&request));
    assert_eq!(
        common::status_of(&reply),
        "ok",
        "a context inside max_context_bytes must not be refused by the rendered cap: {}",
        String::from_utf8_lossy(&reply[..reply.len().min(400)])
    );
}

#[test]
fn the_rendered_cap_still_refuses_what_actually_exceeds_it() {
    // Raising `max_rendered_bytes` to clear the escape floor must not make the cap unreachable: a
    // check that can no longer fire is not a check, and a cap nothing can reach is how "bounded"
    // quietly becomes "unbounded".
    //
    // The floor is `6 * max_context_bytes + 8192`, so it covers a worst-case context plus 8192
    // bytes of scaffolding — and *not* the options, which have their own 512-byte cap and are
    // escaped by the same 6x rule. A maximal context alongside 16 options that are each inside the
    // option cap but expand 6x is comfortably over the total.
    let dense_context = vec![0u8; RenderCaps::DEFAULT.max_context_bytes];
    // 511 vertical tabs, each rendering as the six bytes \u000b, behind one ordinary character
    // so the option is not all-whitespace (U+000B is `White_Space`, and an option that trims to
    // nothing is `empty_option`, which would be a different test).
    let expands: String =
        std::iter::once('x').chain(std::iter::repeat_n('\u{000b}', 511)).collect();
    assert_eq!(expands.len(), RenderCaps::DEFAULT.max_option_bytes);
    let options: Vec<String> = (0..16).map(|k| format!("{k:x}{}", &expands[1..])).collect();
    let mut value = common::request_with_slots(
        &dense_context,
        json!([{"name": "verdict", "type": "choice", "options": options}]),
    );
    // Keep the payload honest: this is a legal request in every other respect.
    value["task"] = json!("devcouncil.verdict");
    assert_refused(
        "a maximal context plus options that expand 6x",
        value,
        "rendered_prompt_over_cap",
    );
}

#[test]
fn the_caps_configuration_itself_is_refused_below_the_escape_floor() {
    // The floor is `ESCAPE_WORST_CASE_GROWTH * max_context_bytes + 8192`, and it is a
    // transcription of `RenderCaps.__post_init__` in `python/qd_data/render.py`. Pinning both
    // sides of the boundary is what stops the factor drifting back to 2 unnoticed.
    assert_eq!(ESCAPE_WORST_CASE_GROWTH, 6);
    assert_eq!(RenderCaps::DEFAULT.max_rendered_bytes, 802_816);
    RenderCaps::DEFAULT
        .validate()
        .expect("the shipping defaults clear their own floor");

    let floor = ESCAPE_WORST_CASE_GROWTH * 1024 + 8192;
    let at_floor = RenderCaps {
        max_context_bytes: 1024,
        max_rendered_bytes: floor,
        ..RenderCaps::DEFAULT
    };
    at_floor.validate().expect("exactly at the floor is legal");

    let below = RenderCaps {
        max_rendered_bytes: floor - 1,
        ..at_floor
    };
    // One byte below, and every 2x-era configuration, is refused.
    assert!(below.validate().is_err(), "one byte below the floor must be refused");
    let two_x = RenderCaps {
        max_context_bytes: 131_072,
        max_rendered_bytes: 393_216,
        ..RenderCaps::DEFAULT
    };
    assert!(
        two_x.validate().is_err(),
        "the pre-fix defaults must not validate under a 6x floor"
    );
}

#[test]
fn an_empty_task_slot_name_or_option_is_refused() {
    let mut value = with_slots(one_choice());
    value["task"] = json!("   ");
    assert_refused("a blank task", value, "empty_task");

    assert_refused(
        "a blank slot name",
        with_slots(json!([{"name": "  ", "type": "span"}])),
        "empty_slot_name",
    );
    assert_refused(
        "a blank option",
        with_slots(json!([{"name": "verdict", "type": "choice", "options": ["  ", "clean"]}])),
        "empty_option",
    );
}

#[test]
fn a_duplicate_or_reserved_option_is_refused() {
    assert_refused(
        "two identical options",
        with_slots(json!([{"name": "verdict", "type": "choice",
                           "options": ["stub", "clean", "stub"]}])),
        "duplicate_option",
    );
    for spelling in ["noul", "NOUL", " noul "] {
        assert_refused(
            &format!("an option literally named {spelling:?}"),
            with_slots(
                json!([{"name": "verdict", "type": "choice", "options": [spelling, "clean"]}]),
            ),
            "reserved_option_name",
        );
    }
}

// -- envelope and slot shape -------------------------------------------------------------------

#[test]
fn an_unknown_route_or_slot_type_is_refused() {
    let mut value = with_slots(one_choice());
    value["route"] = json!("turbo");
    assert_refused("an unknown route", value, "unknown_route");

    assert_refused(
        "an unknown slot type",
        with_slots(json!([{"name": "verdict", "type": "vibe"}])),
        "unknown_slot_type",
    );
}

#[test]
fn a_slot_carrying_the_wrong_field_is_refused() {
    assert_refused(
        "a span with bins",
        with_slots(json!([{"name": "evidence", "type": "span", "bins": 5}])),
        "slot_field_not_allowed",
    );
    assert_refused(
        "a score with options",
        with_slots(json!([{"name": "severity", "type": "score", "options": ["a", "b"]}])),
        "slot_field_not_allowed",
    );
    assert_refused(
        "a choice with no options",
        with_slots(json!([{"name": "verdict", "type": "choice"}])),
        "slot_field_missing",
    );
    assert_refused(
        "a score with no bins",
        with_slots(json!([{"name": "severity", "type": "score"}])),
        "slot_field_missing",
    );
}

#[test]
fn an_unknown_field_is_refused_rather_than_ignored() {
    let mut value = with_slots(one_choice());
    value["contextt"] = json!([1, 2, 3]);
    assert_refused("a misspelled field", value, "malformed_request");
}

#[test]
fn malformed_ambiguous_and_unknown_lines_are_refused() {
    assert_refused_bytes("not JSON", b"{nope", "malformed_request");
    assert_refused_bytes("a JSON array", b"[1,2,3]", "malformed_request");
    assert_refused_bytes(
        "both op and schema_version",
        br#"{"op":"status","schema_version":1}"#,
        "ambiguous_envelope",
    );
    assert_refused_bytes("an unknown op", br#"{"op":"restart"}"#, "unknown_op");
}

#[test]
fn a_line_over_the_payload_cap_is_refused() {
    let mut payload = Vec::with_capacity(qd_runtime::wire::MAX_PAYLOAD_BYTES + 64);
    payload.extend_from_slice(br#"{"schema_version":1,"task":""#);
    payload.resize(qd_runtime::wire::MAX_PAYLOAD_BYTES + 32, b'x');
    payload.extend_from_slice(br#""}"#);
    assert_refused_bytes("an oversized line", &payload, "payload_over_cap");
}

// -- hash binding --------------------------------------------------------------------------------

#[test]
fn every_pinned_hash_is_checked_and_names_which_one_failed() {
    let bogus = "0".repeat(64);
    for (field, expected_which) in [
        ("tokenizer_hash", "tokenizer"),
        ("weight_hash", "weights"),
        ("head_hash", "head"),
        ("label_set_hash", "label_set"),
        ("calibration_hash", "calibration"),
    ] {
        let mut value = common::sample_request();
        value["expect"] = json!({ field: bogus });
        let service = common::reference_service();
        let reply = service.handle_line(&common::line(&value));
        assert_eq!(common::status_of(&reply), "refused", "{field}");
        assert_eq!(common::refusal_kind(&reply), "hash_mismatch", "{field}");
        let parsed: Value = serde_json::from_slice(&reply).expect("parses");
        assert_eq!(
            parsed.pointer("/refusal/which").and_then(Value::as_str),
            Some(expected_which),
            "the refusal must name which hash failed"
        );
        assert_eq!(
            parsed.pointer("/refusal/actual").and_then(Value::as_str),
            Some(bogus.as_str()),
            "the refusal must carry the value the caller pinned"
        );
        assert!(
            parsed
                .pointer("/refusal/expected")
                .and_then(Value::as_str)
                .is_some_and(|s| s != bogus),
            "the refusal must carry the build's own value too"
        );
    }
}

#[test]
fn a_matching_pin_is_answered() {
    let identity = ReferenceBackend::new(true).identity().clone();
    let mut value = common::sample_request();
    value["expect"] = json!({
        "tokenizer_hash": identity.tokenizer_hash,
        "weight_hash": identity.weight_hash,
        "head_hash": identity.head_hash,
        "label_set_hash": identity.label_set_hash,
        "calibration_hash": identity.calibration_hash,
    });
    let service = common::reference_service();
    let reply = service.handle_line(&common::line(&value));
    assert_eq!(
        common::status_of(&reply),
        "ok",
        "a correct pin must be answered, or the mismatch test above proves nothing: {}",
        String::from_utf8_lossy(&reply)
    );
}

// -- the registered route ------------------------------------------------------------------------

fn registered_runtime(bind_to_backbone: bool, include_verdict: bool) -> Runtime {
    let backend = ReferenceBackend::new(true);
    let backbone = backend.identity().weight_hash.clone();
    let mut slots = std::collections::BTreeMap::new();
    if include_verdict {
        slots.insert(
            "verdict".to_string(),
            HeadMatrix {
                // Two named options plus the reserved noul row.
                rows: vec![vec![0.01f32; REFERENCE_FEATURE_DIM]; 3],
            },
        );
    } else {
        slots.insert(
            "severity".to_string(),
            HeadMatrix {
                rows: vec![vec![0.01f32; REFERENCE_FEATURE_DIM]; 6],
            },
        );
    }
    let mut registry = HeadRegistry::new();
    registry.insert(RegisteredHead {
        task: "devcouncil.verdict".to_string(),
        head_hash: "head-hash-under-test".to_string(),
        backbone_hash: if bind_to_backbone {
            backbone
        } else {
            "a-different-backbone".to_string()
        },
        feature_dim: REFERENCE_FEATURE_DIM,
        slots,
    });
    Runtime::with_backend(
        Arc::new(backend),
        CalibrationTable::reference(),
        registry,
        RenderCaps::DEFAULT,
    )
    .expect("assembles")
}

#[track_caller]
fn assert_runtime_refuses(runtime: &Runtime, value: Value, expected_kind: &str) {
    let request = common::validated(&value);
    let response = runtime.answer(&request, None);
    assert_eq!(response.caller_reading(), CallerReading::RequestRefused);
    match response {
        Response::Refused(envelope) => assert_eq!(envelope.refusal.kind(), expected_kind),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

fn registered_request(slots: Value) -> Value {
    let mut value = with_slots(slots);
    value["route"] = json!("registered");
    value
}

#[test]
fn the_registered_route_refuses_a_task_with_no_head() {
    let runtime = common::reference_runtime();
    assert_runtime_refuses(
        &runtime,
        registered_request(one_choice()),
        "registered_head_missing",
    );
}

#[test]
fn a_head_not_bound_to_the_loaded_backbone_is_refused() {
    let runtime = registered_runtime(false, true);
    let request = common::validated(&registered_request(one_choice()));
    match runtime.answer(&request, None) {
        Response::Refused(envelope) => {
            assert_eq!(envelope.refusal.kind(), "hash_mismatch");
            let value = serde_json::to_value(&envelope.refusal).expect("serializes");
            assert_eq!(
                value.get("which").and_then(Value::as_str),
                Some("head_backbone_binding")
            );
        }
        other => panic!("a head fitted on other weights must be refused, got {other:?}"),
    }
}

#[test]
fn a_head_with_no_rows_for_a_slot_is_refused() {
    let runtime = registered_runtime(true, false);
    assert_runtime_refuses(
        &runtime,
        registered_request(one_choice()),
        "registered_head_slot_missing",
    );
}

#[test]
fn the_registered_route_refuses_a_span_slot() {
    let runtime = registered_runtime(true, true);
    assert_runtime_refuses(
        &runtime,
        registered_request(json!([{"name": "evidence", "type": "span"}])),
        "registered_route_span_unsupported",
    );
}

#[test]
fn a_bound_head_answers_and_pins_its_own_hash() {
    let runtime = registered_runtime(true, true);
    let mut value = registered_request(one_choice());
    value["expect"] = json!({"head_hash": "head-hash-under-test"});
    let request = common::validated(&value);
    match runtime.answer(&request, None) {
        Response::Ok(envelope) => {
            assert!(envelope.slots.contains_key("verdict"));
            assert!(
                envelope.degraded,
                "the reference backend is not a model, so every answer is degraded"
            );
        }
        other => panic!("a bound head must answer, got {other:?}"),
    }

    // And a wrong pin on the *head file* is refused, which is what makes the pin above meaningful.
    let mut value = registered_request(one_choice());
    value["expect"] = json!({"head_hash": "some-other-head"});
    assert_runtime_refuses(&runtime, value, "hash_mismatch");
}

// -- calibration ---------------------------------------------------------------------------------

#[test]
fn a_shape_the_calibration_table_was_never_fitted_for_is_refused() {
    let runtime = Runtime::with_backend(
        Arc::new(ReferenceBackend::new(true)),
        // A table with nothing in it: every slot shape is missing.
        CalibrationTable::new("empty-under-test"),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("assembles");
    assert_runtime_refuses(
        &runtime,
        with_slots(one_choice()),
        "calibration_entry_missing",
    );
}

// -- the shipping default --------------------------------------------------------------------------

#[test]
fn with_no_backend_a_valid_request_is_an_error_not_an_abstention() {
    let service: Service = common::backendless_service();
    let reply = service.handle_line(&common::line(&common::sample_request()));
    assert_eq!(common::status_of(&reply), "error");
    let value: Value = serde_json::from_slice(&reply).expect("parses");
    assert_eq!(
        value.pointer("/error/kind").and_then(Value::as_str),
        Some("unavailable")
    );
    // No answer keys anywhere: an absent model cannot read as an abstaining one.
    let text = String::from_utf8_lossy(&reply);
    assert!(!text.contains("\"noul\""), "{text}");
    assert!(!text.contains("\"slots\""), "{text}");
}

// -- the payload cap, and what it does not cover ----------------------------------------------------

#[test]
fn the_payload_cap_does_not_cover_a_request_at_every_other_limit_at_once() {
    // `GAP-RT-PAYLOAD-CAP-VS-MAXIMAL-SLOTS`, settled as arithmetic rather than as a remark.
    //
    // The question was "is MAX_PAYLOAD_BYTES still derived correctly now that context is base64?"
    // The answer is no, and it never was: the derivation covers the context, the question, the
    // task and the envelope, and omits the slot list — which on its own is larger than all of
    // them together. This asserts both halves of that, so the day either cap moves the numbers
    // are re-stated by a failing test rather than by nobody.
    let budget = payload_budget(&RenderCaps::DEFAULT);
    // Printed, not asserted: the digits belong in `cargo test -- --nocapture` output, where they
    // are regenerated, and not in a document where they would go stale the way the comment this
    // test replaces did.
    println!(
        "payload budget vs MAX_PAYLOAD_BYTES={MAX_PAYLOAD_BYTES}: context_b64={} question={} \
         task={} slot_list={} envelope={} total={} without_slot_list={} uncapped={:?}",
        budget.context_b64,
        budget.question,
        budget.task,
        budget.slot_list,
        budget.envelope,
        budget.total(),
        budget.without_slot_list(),
        budget.uncapped,
    );

    assert!(
        budget.without_slot_list() <= MAX_PAYLOAD_BYTES,
        "the terms the cap's comment was derived from must fit inside it: {} > {}",
        budget.without_slot_list(),
        MAX_PAYLOAD_BYTES
    );
    assert!(
        !budget.fits(),
        "a request legal at every documented limit at once is expected NOT to fit \
         ({} <= {}). If it now does, MAX_PAYLOAD_BYTES or RenderCaps changed and \
         docs/schema-api.md's statement of this must change with it",
        budget.total(),
        MAX_PAYLOAD_BYTES
    );
    assert!(
        budget.slot_list > budget.without_slot_list(),
        "the slot list is the term that overflows the cap; if that stops being true the \
         explanation in docs/schema-api.md is wrong. slot_list = {}, everything else = {}",
        budget.slot_list,
        budget.without_slot_list()
    );
}

#[test]
fn the_payload_budget_counts_every_term_so_the_total_is_the_worst_case() {
    // This test used to assert the opposite. It read:
    //
    //     assert!(budget.uncapped.contains(&"slots[].name"));
    //     assert!(parse_line(&…4096-byte name…).is_ok());
    //
    // because a slot name had no cap on any side, which made `total()` a lower bound on the worst
    // case rather than the worst case. `GAP-RT-SLOT-NAME-UNCAPPED` is now closed by
    // `MAX_SLOT_NAME_BYTES`, so both halves invert.
    let budget = payload_budget(&RenderCaps::DEFAULT);
    assert!(
        budget.uncapped.is_empty(),
        "every request term is now capped, so nothing belongs in `uncapped` and `total()` is the \
         worst case rather than a lower bound on it. Still uncounted: {:?}",
        budget.uncapped
    );

    // The name term is actually in the arithmetic, not merely declared to be.
    //
    // An earlier version of this assertion read `budget.slot_list > names_term` and was vacuous:
    // the options term alone is 1_575_936 bytes against a 49_152-byte names term, so deleting the
    // names term from `payload_budget` left it passing. Mutation testing caught that. The check
    // below is per-slot and additive, so dropping either term fails it.
    assert_eq!(
        budget.slot_list % MAX_SLOTS,
        0,
        "the slot-list term is {MAX_SLOTS} identical worst-case slots; slot_list = {}",
        budget.slot_list
    );
    let per_slot = budget.slot_list / MAX_SLOTS;
    let name_bytes = MAX_SLOT_NAME_BYTES * JSON_ESCAPE_WORST_CASE_GROWTH;
    let option_bytes =
        MAX_OPTIONS * RenderCaps::DEFAULT.max_option_bytes * JSON_ESCAPE_WORST_CASE_GROWTH;
    assert!(
        per_slot >= name_bytes + option_bytes,
        "a worst-case slot weighs at least its name ({name_bytes}) plus its options \
         ({option_bytes}) = {}, but the budget allows {per_slot} per slot. A term is missing from \
         `payload_budget`.",
        name_bytes + option_bytes
    );
}

#[test]
fn a_slot_name_is_capped_in_bytes_and_the_boundary_is_exact() {
    // At the cap, accepted. One byte over, refused. An off-by-one here is the difference between
    // a cap and a suggestion.
    let at_cap = "n".repeat(MAX_SLOT_NAME_BYTES);
    let value = with_slots(json!([
        {"name": at_cap, "type": "choice", "options": ["stub", "clean"]}
    ]));
    assert!(
        parse_line(&serde_json::to_vec(&value).expect("serializes")).is_ok(),
        "a slot name of exactly MAX_SLOT_NAME_BYTES ({MAX_SLOT_NAME_BYTES}) is legal; the cap is \
         an inclusive maximum, not an exclusive one"
    );

    assert_refused(
        "a slot name one byte over the cap",
        with_slots(json!([
            {"name": "n".repeat(MAX_SLOT_NAME_BYTES + 1), "type": "choice",
             "options": ["stub", "clean"]}
        ])),
        "slot_name_over_cap",
    );

    // The specific regression `GAP-RT-SLOT-NAME-UNCAPPED` recorded: this parsed cleanly before.
    assert_refused(
        "the 4096-byte slot name the gap record measured",
        with_slots(json!([
            {"name": "n".repeat(4096), "type": "choice", "options": ["stub", "clean"]}
        ])),
        "slot_name_over_cap",
    );

    // Bytes, not characters. 'é' is two bytes in UTF-8, so 128 of them is 256 bytes — at the cap —
    // and 129 is 258, over it. A cap measured in `chars()` would accept both and a cap measured in
    // UTF-16 units would accept both; only a byte count refuses the second, and the byte count is
    // what `MAX_PAYLOAD_BYTES` is denominated in.
    let two_byte_char_at_cap = "é".repeat(MAX_SLOT_NAME_BYTES / 2);
    assert_eq!(two_byte_char_at_cap.len(), MAX_SLOT_NAME_BYTES);
    assert!(
        parse_line(
            &serde_json::to_vec(&with_slots(json!([
                {"name": two_byte_char_at_cap, "type": "span"}
            ])))
            .expect("serializes")
        )
        .is_ok(),
        "128 two-byte characters is exactly 256 bytes and must be accepted"
    );
    assert_refused(
        "129 two-byte characters — 258 bytes, over the cap, though only 129 chars",
        with_slots(json!([{"name": "é".repeat(MAX_SLOT_NAME_BYTES / 2 + 1), "type": "span"}])),
        "slot_name_over_cap",
    );
}

#[test]
fn the_over_cap_refusal_names_the_numbers_and_not_the_name() {
    // An oversized, caller-controlled string must not be echoed into the refusal: a request would
    // then choose the size of its own error message. The refusal carries counts and an index.
    let name = "x".repeat(4096);
    let line = serde_json::to_vec(&with_slots(json!([
        {"name": name, "type": "span"}
    ])))
    .expect("serializes");
    let refusal = parse_line(&line).expect_err("an over-cap slot name is refused");
    assert_eq!(refusal.kind(), "slot_name_over_cap");

    let rendered = refusal.to_string();
    assert!(
        rendered.contains("4096") && rendered.contains(&MAX_SLOT_NAME_BYTES.to_string()),
        "the refusal states both the actual size and the cap: {rendered}"
    );
    assert!(
        !rendered.contains(&name),
        "the refusal must not echo the oversized name back; it was {} bytes and the message is {} \
         bytes",
        name.len(),
        rendered.len()
    );
    assert!(
        rendered.len() < 512,
        "an over-cap refusal message must be bounded regardless of the input that caused it, got \
         {} bytes",
        rendered.len()
    );
}
