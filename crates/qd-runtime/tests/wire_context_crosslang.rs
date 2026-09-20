//! **The standing cross-lane contract check.** The real Python encoder, into the real Rust decoder.
//!
//! `docs/schema-api.md`: *"A cross-language test asserts the Python encoder and the Rust decoder
//! agree on the same bytes... Two sides agreeing is an assumption until it is asserted — which is
//! precisely how this divergence survived in the first place."*
//!
//! # Why a per-lane suite cannot replace this file
//!
//! Two independent cross-lane incompatibilities were found in this repo on the same day, both
//! between suites that were **fully green on each side**:
//!
//! * `GAP-RT-WIRE-CONTEXT-ENCODING` — `qd_data` emitted `context_b64`; `qd-runtime` read `context`
//!   as a JSON array of byte values and refused a string in that position. No caller could satisfy
//!   both, so the system did not work end to end.
//! * `GAP-MUTATE-SPAN-WIRE-NAMES` — `qd-mutate` emitted spans as `{"start","end"}` while
//!   `qd-runtime`'s [`SpanValue`] declares `{start_line,end_line}` with `deny_unknown_fields`.
//!
//! Two is a pattern, and the pattern has a cause: **a suite that exercises one lane's encoder
//! against that same lane's decoder cannot observe the other lane at all.** Each side was
//! internally consistent the entire time it was incompatible. So this file runs *both*
//! implementations, and it covers the whole shared surface rather than only the field that broke
//! most recently.
//!
//! # What it asserts
//!
//! 1. Every request `python/qd_data/schema.py::Request.to_wire()` produces decodes, through
//!    [`parse_line`], to the bytes and fields Python meant — over contexts chosen for what they
//!    break: non-UTF-8, empty, NUL, the exotic line separators, NFC vs NFD, every byte value.
//! 2. Rust re-encoding that context reproduces Python's `context_b64` **character for character**,
//!    so the agreement is bidirectional and not merely "both are some base64".
//! 3. All three [`SlotSpec`] kinds, both routes, the option and bin bounds, and the
//!    accepted-and-ignored `example_id` / `metadata`.
//! 4. [`SpanValue`] is `{start_line, end_line}` on the wire and refuses `{start, end}`.
//! 5. [`HashExpectation`]'s five optional fields, and the absent case defaulting to empty.
//! 6. The malformed payloads both lanes must refuse, **and the identifier each one uses**, because
//!    the identifier is what a caller branches on.
//!
//! Where a shared type exists on only one side, the asymmetry is asserted and carries a gap id.
//! It is not papered over, and it is not invented in the missing lane by this test.
//!
//! # When the venv is absent
//!
//! Reported as skipped, never as passed. `CLAUDE.md`: *a check that could not run must never
//! report the same result as a check that ran and passed.*

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use qd_runtime::context::Context;
use qd_runtime::render::{RenderCaps, ESCAPE_WORST_CASE_GROWTH};
use qd_runtime::schema::{
    HashExpectation, Route, SlotSpec, SpanValue, MAX_BINS, MAX_OPTIONS, MIN_BINS,
};
use qd_runtime::wire::{parse_line, Incoming};
use serde_json::Value;

// -- driving the other lane ----------------------------------------------------------------------

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <repo>/crates/qd-runtime
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn python() -> Option<PathBuf> {
    let p = repo_root().join(".venv/bin/python");
    p.exists().then_some(p)
}

/// Run the real Python encoder once and cache what it emitted.
///
/// `None` only ever means "the project venv is absent". Every other failure — a non-zero exit, a
/// traceback, output that is not JSON — panics, because those are the Python lane telling us
/// something and a test that swallowed them would be the silent pass this file exists to prevent.
fn document() -> Option<&'static Value> {
    static DOC: OnceLock<Option<Value>> = OnceLock::new();
    DOC.get_or_init(|| {
        let py = python()?;
        let script = repo_root()
            .join("crates/qd-runtime/tests/crosslang_fixtures.py");
        assert!(
            script.exists(),
            "the Python half of this test is missing: {}",
            script.display()
        );
        let out = Command::new(&py)
            .arg(&script)
            .output()
            .unwrap_or_else(|e| panic!("could not run {}: {e}", py.display()));
        assert!(
            out.status.success(),
            "the Python fixture emitter failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        let parsed: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
            panic!(
                "the Python fixture emitter did not produce JSON: {e}\nstdout was:\n{}",
                String::from_utf8_lossy(&out.stdout)
            )
        });
        Some(parsed)
    })
    .as_ref()
}

/// The skip notice. Printed, never silent.
macro_rules! doc_or_skip {
    () => {
        match document() {
            Some(d) => d,
            None => {
                eprintln!(
                    "SKIPPED (reported, not passed): the project .venv is absent, so the Python \
                     encoder could not be run. This assertion did NOT pass; it did not run."
                );
                return;
            }
        }
    };
}

fn array<'a>(doc: &'a Value, key: &str) -> &'a Vec<Value> {
    doc.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("the Python document has no `{key}` array"))
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("fixture field `{key}` is missing or not a string: {value}"))
}

fn count(value: &Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("fixture field `{key}` is missing or not an integer: {value}"))
}

/// Parse the fixture's wire object the way the service does.
fn decode(fixture: &Value) -> qd_runtime::DecisionRequest {
    let wire = fixture
        .get("wire")
        .unwrap_or_else(|| panic!("fixture has no `wire`: {fixture}"));
    let line = serde_json::to_vec(wire).expect("the Python wire object re-serializes");
    match parse_line(&line) {
        Ok(Incoming::Request(request)) => *request,
        Ok(Incoming::Op(op)) => panic!(
            "`{}`: Python emitted a request, Rust read the control op {op:?}",
            text(fixture, "label")
        ),
        Err(refusal) => panic!(
            "`{}`: Rust refused a request the Python encoder produced: {refusal}\nwire was: {wire}",
            text(fixture, "label")
        ),
    }
}

fn hex(bytes: &[u8]) -> String {
    qd_runtime::hex(bytes)
}

// -- 1. the context: bytes in, the same bytes out --------------------------------------------------

#[test]
fn every_python_encoded_context_decodes_to_the_bytes_python_meant() {
    let doc = doc_or_skip!();
    let requests = array(doc, "requests");
    assert!(
        requests.len() >= 20,
        "the fixture set shrank to {} — this is the whole cross-lane surface, not a sample",
        requests.len()
    );

    let mut seen_non_utf8 = false;
    let mut seen_empty = false;
    let mut seen_nul = false;
    let mut seen_separators = false;

    for fixture in requests {
        let label = text(fixture, "label");
        let request = decode(fixture);
        let expected_hex = text(fixture, "context_hex");
        assert_eq!(
            hex(request.context.as_bytes()),
            expected_hex,
            "`{label}`: the decoded context is not the bytes Python encoded"
        );
        assert_eq!(
            request.context.len() as u64,
            count(fixture, "context_len"),
            "`{label}`: length disagreement"
        );

        // Bidirectional: Rust's encoder must reproduce Python's exact characters. Equal bytes out
        // of two decoders would still leave two encoders free to disagree.
        let python_blob = text(
            fixture
                .get("wire")
                .unwrap_or_else(|| panic!("`{label}`: no wire")),
            "context_b64",
        );
        assert_eq!(
            request.context.to_b64(),
            python_blob,
            "`{label}`: Rust re-encodes this context differently from Python"
        );

        seen_non_utf8 |= std::str::from_utf8(request.context.as_bytes()).is_err();
        seen_empty |= request.context.is_empty();
        seen_nul |= request.context.as_bytes().contains(&0u8);
        seen_separators |= request.context.as_bytes().windows(3).any(|w| w == [0xe2, 0x80, 0xa8])
            || request.context.as_bytes().contains(&0x0b);
    }

    // The cases this test exists for must actually be present, or the sweep above proves less than
    // it appears to. A fixture list that quietly lost its non-UTF-8 entry would still pass every
    // assertion in the loop.
    assert!(seen_non_utf8, "no non-UTF-8 context in the fixture set");
    assert!(seen_empty, "no empty context in the fixture set");
    assert!(seen_nul, "no NUL-carrying context in the fixture set");
    assert!(
        seen_separators,
        "no exotic line separator in the fixture set"
    );
}

#[test]
fn a_python_encoded_context_survives_the_type_that_carries_it() {
    // `Context`'s own serde form is the same two fields, so a `Context` nested anywhere carries
    // the encoding the envelope does. A second serialization of this type is how the divergence
    // this file documents would come back.
    let doc = doc_or_skip!();
    for fixture in array(doc, "requests") {
        let label = text(fixture, "label");
        let request = decode(fixture);
        let value = serde_json::to_value(&request.context).expect("a context serializes");
        let object = value.as_object().expect("an object");
        assert_eq!(
            object.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["context_b64", "context_len"],
            "`{label}`: Context's own wire form drifted from the envelope's"
        );
        let round: Context = serde_json::from_value(value).expect("round-trips");
        assert_eq!(
            round.as_bytes(),
            request.context.as_bytes(),
            "`{label}`: Context did not round-trip through its own serde form"
        );
    }
}

// -- 2. the rest of the envelope --------------------------------------------------------------

#[test]
fn the_envelope_fields_cross_intact() {
    let doc = doc_or_skip!();
    for fixture in array(doc, "requests") {
        let label = text(fixture, "label");
        let request = decode(fixture);
        assert_eq!(
            u64::from(request.schema_version),
            count(fixture, "schema_version"),
            "`{label}`: schema_version"
        );
        assert_eq!(request.task, text(fixture, "task"), "`{label}`: task");
        assert_eq!(
            request.question,
            text(fixture, "question"),
            "`{label}`: question"
        );
        assert_eq!(
            request.route.as_str(),
            text(fixture, "route"),
            "`{label}`: route"
        );
        // Python emits no `expect`, and absent must mean "pinned nothing" rather than "pinned
        // empty strings". `expect` is optional by design; see `HashExpectation`.
        assert!(
            request.expect.is_empty(),
            "`{label}`: a request with no `expect` must default to pinning nothing"
        );
        assert_eq!(request.expect, HashExpectation::default());
    }
}

#[test]
fn both_routes_are_reachable_from_the_python_encoder() {
    let doc = doc_or_skip!();
    let routes: Vec<Route> = array(doc, "requests")
        .iter()
        .map(|f| decode(f).route)
        .collect();
    assert!(
        routes.contains(&Route::Generic),
        "no `generic` request in the fixture set"
    );
    assert!(
        routes.contains(&Route::Registered),
        "no `registered` request in the fixture set — the route field would be untested"
    );
}

#[test]
fn every_slot_kind_crosses_intact_with_its_payload() {
    let doc = doc_or_skip!();
    let mut kinds_seen: BTreeMap<&str, usize> = BTreeMap::new();

    for fixture in array(doc, "requests") {
        let label = text(fixture, "label");
        let request = decode(fixture);
        let truth = array(fixture, "slots");
        assert_eq!(
            request.slots.len(),
            truth.len(),
            "`{label}`: slot count disagreement"
        );
        for (slot, expected) in request.slots.iter().zip(truth) {
            let kind = text(expected, "kind");
            *kinds_seen.entry(match kind {
                "choice" => "choice",
                "score" => "score",
                "span" => "span",
                other => panic!("`{label}`: Python named an unknown slot kind `{other}`"),
            })
            .or_default() += 1;
            assert_eq!(slot.name(), text(expected, "name"), "`{label}`: slot name");
            assert_eq!(
                slot.kind().as_str(),
                kind,
                "`{label}`: slot `{}` kind",
                slot.name()
            );
            match slot {
                SlotSpec::Choice { options, .. } => {
                    let want: Vec<&str> = array(expected, "options")
                        .iter()
                        .map(|o| o.as_str().expect("option is a string"))
                        .collect();
                    // Order matters: a permuted option list is a different prompt and therefore a
                    // different label set. `qd_data.schema.label_set_hash` hashes in order too.
                    assert_eq!(options, &want, "`{label}`: option list or its order");
                }
                SlotSpec::Score { bins, .. } => {
                    assert_eq!(
                        u64::from(*bins),
                        count(expected, "bins"),
                        "`{label}`: bins"
                    );
                }
                SlotSpec::Span { .. } => {
                    assert!(
                        expected.get("options").is_none() && expected.get("bins").is_none(),
                        "`{label}`: a span slot carries neither options nor bins"
                    );
                }
            }
        }
    }

    for kind in ["choice", "score", "span"] {
        assert!(
            kinds_seen.contains_key(kind),
            "no `{kind}` slot in the fixture set, so that slot kind is untested"
        );
    }
}

#[test]
fn the_option_and_bin_bounds_are_the_same_number_on_both_sides() {
    let doc = doc_or_skip!();
    let inv = doc
        .get("inventory")
        .unwrap_or_else(|| panic!("no `inventory` in the Python document"));
    // `GAP-SCHEMA-NOUL-LETTER-BUDGET` was settled as 16 named options plus one reserved `noul`
    // row. If either lane ever flips it, the other must fail rather than silently render a
    // 17th letter nobody decodes.
    assert_eq!(
        count(inv, "max_choice_options"),
        MAX_OPTIONS as u64,
        "the two lanes disagree on the named-option cap"
    );
    assert_eq!(count(inv, "min_score_bins"), u64::from(MIN_BINS));
    assert_eq!(count(inv, "max_score_bins"), u64::from(MAX_BINS));

    // And the encoder really does reach the caps, so the numbers above are not merely equal
    // constants that nothing exercises.
    let widest = array(doc, "requests")
        .iter()
        .map(decode)
        .filter_map(|r| {
            r.slots
                .iter()
                .filter_map(|s| match s {
                    SlotSpec::Choice { options, .. } => Some(options.len()),
                    _ => None,
                })
                .max()
        })
        .max()
        .expect("some choice slot exists");
    assert_eq!(widest, MAX_OPTIONS, "no fixture exercises the option cap");
}

#[test]
fn example_id_and_metadata_are_accepted_and_ignored() {
    // `wire.rs::REQUEST_KEYS` accepts these deliberately: they steer the *training* option
    // shuffle only, the serving path renders with no shuffle, and rejecting them would make a
    // request the training lane emits unservable. "Ignored" is structural — `DecisionRequest`
    // has no field for either — so the assertion is that such a request is accepted at all.
    let doc = doc_or_skip!();
    let carrying: Vec<&Value> = array(doc, "requests")
        .iter()
        .filter(|f| {
            let wire = f.get("wire").expect("wire");
            wire.get("example_id").and_then(Value::as_str).is_some_and(|s| !s.is_empty())
                || wire
                    .get("metadata")
                    .and_then(Value::as_object)
                    .is_some_and(|m| !m.is_empty())
        })
        .collect();
    assert!(
        !carrying.is_empty(),
        "no fixture carries example_id/metadata, so the accepted-and-ignored path is untested"
    );
    for fixture in carrying {
        let label = text(fixture, "label");
        // Decoding is the whole assertion: it must not refuse.
        let request = decode(fixture);
        assert_eq!(request.task, text(fixture, "task"), "`{label}`");
    }
}

// -- 3. SpanValue: the shape that already broke once --------------------------------------------

#[test]
fn span_value_is_start_line_and_end_line_and_refuses_anything_else() {
    // `GAP-MUTATE-SPAN-WIRE-NAMES`: `qd-mutate` emitted `{"start","end"}` against this type's
    // `deny_unknown_fields`, so every span row would have hard-failed on all three consuming
    // surfaces. The field names are pinned here as literals rather than derived, because a
    // derivation would rename with the struct and assert nothing.
    let span = SpanValue {
        start_line: 41,
        end_line: 47,
    };
    let value = serde_json::to_value(span).expect("a span serializes");
    let object = value.as_object().expect("an object");
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, vec!["end_line", "start_line"]);
    assert_eq!(object["start_line"], serde_json::json!(41));
    assert_eq!(object["end_line"], serde_json::json!(47));

    // The spelling that broke.
    let short: Result<SpanValue, _> = serde_json::from_value(serde_json::json!({
        "start": 41, "end": 47
    }));
    assert!(short.is_err(), "`{{start,end}}` must not deserialize");
    // A correct span with one extra field is also refused: `deny_unknown_fields` is what makes
    // the first assertion a contract rather than a preference.
    let extra: Result<SpanValue, _> = serde_json::from_value(serde_json::json!({
        "start_line": 41, "end_line": 47, "start": 41
    }));
    assert!(extra.is_err(), "an unknown field must not be ignored");
}

#[test]
fn the_three_span_spellings_stay_distinct_across_the_two_lanes() {
    // The asymmetry, asserted rather than assumed. `qd_data` has no `{start_line, end_line}`
    // producer at all: the one place this lane writes a line span down is
    // `qd_data.rows.GoldAnswer`, whose `value` is a two-element array — a *training label*, not a
    // runtime answer, and never sent to this crate.
    //
    // It is left alone rather than unified: they answer different questions, and forcing one
    // shape on both would be over-unification. Recorded as `GAP-XLANG-SPAN-THREE-SPELLINGS` so
    // that anyone who does route gold answers into `SlotValue::Span` finds the mapping named.
    // **Update 2026-09-19.** Python *has* grown a `SpanValue` — `qd_wire.answer.SpanValue`, the
    // answer-side parser's. So there are now three spellings, not two, and the gap's name is
    // literally right. The assertion below therefore changed direction: it no longer claims
    // Python has none, it claims the two Python spellings stay **distinct**, because that is the
    // thing that would actually break. `qd_wire.SpanValue` mirrors this crate's
    // `{start_line, end_line}`; `qd_data.rows.GoldAnswer.value` is a two-element array and is a
    // *training label* that never crosses this wire.
    let doc = doc_or_skip!();
    let inv = doc.get("inventory").expect("inventory");
    let answer_types = inv
        .get("answer_side_types")
        .and_then(Value::as_object)
        .expect("answer_side_types");
    assert_eq!(
        answer_types.get("SpanValue"),
        Some(&Value::Bool(true)),
        "qd_wire lost its SpanValue. The answer-side parser is the half that makes the \
         Rust -> Python direction assertable at all; without it the corpus in fixtures/wire/ has \
         no second reader"
    );

    let shapes = doc.get("type_shapes").expect("type_shapes");
    let gold = shapes.get("gold_answer_span").expect("gold_answer_span");
    assert!(
        gold.get("value").and_then(Value::as_array).is_some(),
        "GoldAnswer's span stopped being an array; the mapping in \
         GAP-XLANG-SPAN-THREE-SPELLINGS is stale: {gold}"
    );
}

// -- 4. HashExpectation --------------------------------------------------------------------------

#[test]
fn hash_expectation_carries_five_optional_fields_and_refuses_the_rest() {
    let all = serde_json::json!({
        "tokenizer_hash": "a".repeat(64),
        "weight_hash": "b".repeat(64),
        "head_hash": "c".repeat(64),
        "label_set_hash": "d".repeat(64),
        "calibration_hash": "e".repeat(64),
    });
    let parsed: HashExpectation = serde_json::from_value(all.clone()).expect("all five parse");
    assert!(!parsed.is_empty());
    assert_eq!(parsed.tokenizer_hash.as_deref(), Some("a".repeat(64).as_str()));
    assert_eq!(parsed.weight_hash.as_deref(), Some("b".repeat(64).as_str()));
    assert_eq!(parsed.head_hash.as_deref(), Some("c".repeat(64).as_str()));
    assert_eq!(parsed.label_set_hash.as_deref(), Some("d".repeat(64).as_str()));
    assert_eq!(
        parsed.calibration_hash.as_deref(),
        Some("e".repeat(64).as_str())
    );
    assert_eq!(serde_json::to_value(&parsed).expect("serializes"), all);

    // Empty and absent are the same thing, and neither pins anything.
    let empty: HashExpectation = serde_json::from_value(serde_json::json!({})).expect("empty");
    assert!(empty.is_empty());
    assert_eq!(empty, HashExpectation::default());
    assert_eq!(
        serde_json::to_value(&empty).expect("serializes"),
        serde_json::json!({}),
        "an empty expectation must not spell itself as five nulls"
    );

    // A sixth field is refused, not ignored: a caller that misspells `weight_hash` would
    // otherwise be told nothing and pin nothing.
    let sixth: Result<HashExpectation, _> = serde_json::from_value(serde_json::json!({
        "tokenizer_hash": "a", "weights_hash": "b"
    }));
    assert!(sixth.is_err(), "an unknown hash field must be refused");
}

#[test]
fn python_has_no_hash_expectation_and_its_label_set_hash_is_a_different_quantity() {
    // Two asymmetries, both asserted rather than described:
    //
    // 1. `qd_data` has no `HashExpectation` type and `Request.to_wire()` emits no `expect`, so
    //    nothing in the training lane can pin a hash. That is consistent — `expect` is optional
    //    and absent means "pinned nothing" — but it means the five fields have no Python
    //    producer. `GAP-XLANG-NO-PY-HASH-EXPECTATION`.
    //
    // 2. `qd_data.schema.label_set_hash(slots)` hashes the *request's slot list*, while
    //    `HashExpectation.label_set_hash` is compared against the **backend identity's**
    //    `label_set_hash`, a property of the loaded build (`runtime.rs::check_hashes`). Same
    //    name, different quantity: a caller that pinned the Python value would earn a
    //    `hash_mismatch` on every request. `GAP-XLANG-LABEL-SET-HASH-TWO-MEANINGS`.
    let doc = doc_or_skip!();
    let inv = doc.get("inventory").expect("inventory");
    assert_eq!(
        inv.get("answer_side_types")
            .and_then(|a| a.get("HashExpectation")),
        Some(&Value::Bool(false)),
        "Python grew a HashExpectation: cross-assert it instead of skipping it"
    );
    assert_eq!(
        inv.get("request_to_wire_emits_expect"),
        Some(&Value::Bool(false)),
        "Python's to_wire() started emitting `expect`: assert its contents here"
    );

    // The Python label-set hash is a well-formed hex digest, and it is *per request slots* — two
    // different slot lists give two different values. That is what makes it not the build
    // property `expect.label_set_hash` is compared against.
    let hashes = doc
        .get("label_set_hashes")
        .and_then(Value::as_object)
        .expect("label_set_hashes");
    assert!(hashes.len() > 1);
    let distinct: std::collections::BTreeSet<&str> =
        hashes.values().filter_map(Value::as_str).collect();
    assert!(
        distinct.len() > 1,
        "every fixture hashed to one label set, so this assertion proves nothing"
    );
    for (label, value) in hashes {
        let digest = value.as_str().unwrap_or_else(|| panic!("`{label}`"));
        assert_eq!(digest.len(), 64, "`{label}`: not a sha256 hex digest");
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()), "`{label}`");
    }
}

// -- 5. the answer side: Rust -> Python ----------------------------------------------------------

#[test]
fn the_answer_side_now_has_a_python_parser_and_a_corpus_to_feed_it() {
    // **This test used to assert the opposite**, and the flip is the point of it.
    //
    // It read: `SlotAnswer`, `AnswerEnvelope`, `RefusalEnvelope` and `ErrorEnvelope` exist only in
    // this crate, `python/` parses none of them, so the Rust -> Python direction **cannot be
    // asserted** — there is no second implementation to disagree with. `GAP-XLANG-NO-PY-ANSWER-
    // PARSER`. Its closing line was "the day Python grows one, this test demands the other half".
    //
    // Python grew one. `python/qd_wire/` is an independent parser for all four types, and this
    // lane generated `fixtures/wire/` for it to read. Both halves now exist, so the assertion
    // inverts: it fails if **either** half disappears.
    //
    // Two things had to be fixed together to get here, and the second is the interesting one:
    //
    // 1. the probe feeding this test searched a hardcoded tuple of four modules and `qd_wire` was
    //    not among them, so it reported all four types absent while a parser for them sat three
    //    directories away — a check that ran, looked in the wrong place, and came back green.
    //    That is the same shape as the divergences this file exists to catch. The XLANG-PY lane
    //    found it, could not fix it (`crates/` is not theirs), and pinned it as
    //    `GAP-XLANG-RS-GUARD-BLIND-TO-NEW-PACKAGE`. Fixed in `crosslang_fixtures.py`;
    // 2. with the probe fixed, this test failed — correctly, and exactly as written.
    let doc = doc_or_skip!();
    let inv = doc.get("inventory").expect("inventory");
    let types = inv
        .get("answer_side_types")
        .and_then(Value::as_object)
        .expect("answer_side_types");
    let searched = inv
        .get("modules_searched")
        .and_then(Value::as_array)
        .expect("modules_searched");
    assert!(
        searched.len() >= 7,
        "the probe stopped looking in enough places to mean anything: {searched:?}"
    );
    assert!(
        searched.iter().any(|m| m.as_str() == Some("qd_wire")),
        "the probe no longer searches `qd_wire`, so this test would report the answer-side \
         parser absent without looking for it — the blind spot that was \
         GAP-XLANG-RS-GUARD-BLIND-TO-NEW-PACKAGE. Searched: {searched:?}"
    );
    for name in [
        "SlotAnswer",
        "AnswerEnvelope",
        "RefusalEnvelope",
        "ErrorEnvelope",
    ] {
        assert_eq!(
            types.get(name),
            Some(&Value::Bool(true)),
            "python/ lost its `{name}`. The Rust -> Python direction stops being assertable the \
             moment there is no second implementation to disagree with, and fixtures/wire/ then \
             has no reader"
        );
    }

    // The other half of the pair: a parser with nothing to parse asserts nothing. The corpus this
    // lane generates is what the Python side reads, so its absence is a failure here too — the
    // two artefacts are one seam and neither is useful alone.
    let corpus = repo_root().join(qd_runtime::fixtures::CORPUS_DIR);
    assert!(
        corpus.join(qd_runtime::fixtures::MANIFEST).is_file(),
        "{} is missing. Regenerate with `QD_WRITE_FIXTURES=1 cargo test -p qd-runtime --test \
         wire_fixtures`; python/tests/test_wire_golden_corpus.py reports NotRun without it",
        corpus.join(qd_runtime::fixtures::MANIFEST).display()
    );
}

// -- 6. the malformed payloads, and who calls them what --------------------------------------------

/// Where the two lanes' refusal vocabularies genuinely differ today.
///
/// `python/qd_data/errors.py` names its checks after the *property* (`options_len`,
/// `slot_names_unique`) and `crates/qd-runtime/src/refusal.rs` after the *failure*
/// (`too_many_options`, `duplicate_slot_name`). Neither is wrong; they are simply two
/// vocabularies, and a caller that branches on the identifier gets different strings from the two
/// lanes. Unifying them is a rename across both suites and is not this change.
///
/// Pinned as a table so the divergence is visible and cannot widen unnoticed.
/// `GAP-XLANG-REFUSAL-VOCABULARIES`.
const KNOWN_VOCABULARY_DIFFERENCES: &[(&str, &str)] = &[
    // (python check, rust kind)
    ("schema_version", "unknown_schema_version"),
    ("slots_non_empty", "empty_slots"),
    ("slot_type_known", "unknown_slot_type"),
    ("options_len", "too_many_options"),
    ("slot_names_unique", "duplicate_slot_name"),
    ("bins_range", "bins_out_of_range"),
];

#[test]
fn both_lanes_refuse_the_same_payloads_and_name_the_context_checks_identically() {
    let doc = doc_or_skip!();
    let negatives = array(doc, "negatives");
    assert!(
        negatives.len() >= 20,
        "the negative set shrank to {}",
        negatives.len()
    );

    let mut context_cases = 0usize;
    let mut mapped: Vec<(&str, String)> = Vec::new();
    let mut python_accepted: Vec<&str> = Vec::new();
    // Every context identifier this change made shared must actually be exercised, or "they
    // agree" is a statement about an empty set.
    let mut context_kinds: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for case in negatives {
        let label = text(case, "label");
        let wire = case.get("wire").expect("wire");
        let line = serde_json::to_vec(wire).expect("re-serializes");
        let rust_kind = match parse_line(&line) {
            Err(refusal) => refusal.kind().to_string(),
            Ok(_) => panic!("`{label}`: Rust accepted a payload that must be refused: {wire}"),
        };
        let python_check = case.get("python_check").and_then(Value::as_str);
        // The context wire form is where the two lanes were deliberately made to name the same
        // check, so those cases must *agree*, not merely appear in a mapping table.
        let is_context_case = label.starts_with("context") || label.starts_with("retired context");

        match python_check {
            Some(check) if check == rust_kind => {
                if is_context_case {
                    context_cases += 1;
                    context_kinds.insert(rust_kind);
                }
            }
            Some(check) => {
                assert!(
                    !is_context_case,
                    "`{label}`: the context wire form must name one check in both lanes, but \
                     Python says `{check}` and Rust says `{rust_kind}`"
                );
                assert!(
                    KNOWN_VOCABULARY_DIFFERENCES.contains(&(check, rust_kind.as_str())),
                    "`{label}`: the lanes disagree on the refusal identifier — Python says \
                     `{check}`, Rust says `{rust_kind}` — and that pair is not in \
                     KNOWN_VOCABULARY_DIFFERENCES. Either make them agree, or add the pair with \
                     a gap id. Do not leave it unstated."
                );
                mapped.push((check, rust_kind));
            }
            None => {
                assert!(
                    !is_context_case,
                    "`{label}`: Python accepted a malformed context that Rust refused as \
                     `{rust_kind}`"
                );
                python_accepted.push(label);
            }
        }
    }

    assert!(
        context_cases >= 15,
        "only {context_cases} context negatives agreed across the lanes"
    );
    assert_eq!(
        context_kinds.iter().map(String::as_str).collect::<Vec<_>>(),
        vec![
            "context_len_missing",
            "context_length_mismatch",
            "context_not_base64",
            "context_not_bytes",
        ],
        "all four shared context identifiers must be exercised by a real payload"
    );
    assert_eq!(
        mapped.len(),
        KNOWN_VOCABULARY_DIFFERENCES.len(),
        "the known-differences table and what the lanes actually do have drifted apart: {mapped:?}"
    );

    // Python accepting what Rust refuses is a real asymmetry, not a blank. Rust refuses unknown
    // top-level fields; `qd_data.schema.Request.from_wire` reads through `dict.get` and ignores
    // them, so a caller that misspells a field is told by one lane and not the other.
    // `GAP-XLANG-UNKNOWN-FIELD-LENIENCY`.
    assert_eq!(
        python_accepted,
        vec!["unknown top-level field"],
        "the set of payloads Python accepts and Rust refuses changed; it is pinned because each \
         entry is a way for a caller to satisfy one lane and not the other"
    );
}

#[test]
fn a_python_encoded_payload_truncated_in_transit_is_caught_by_the_length() {
    // The reason `context_len` is carried although base64 makes it derivable, asserted across the
    // lanes: take what Python actually emitted, drop whole quanta, and the result is still valid
    // base64 that decodes to a prefix of the real context. Nothing but the declared length
    // notices.
    let doc = doc_or_skip!();
    let mut checked = 0usize;
    for fixture in array(doc, "requests") {
        let label = text(fixture, "label");
        let wire = fixture.get("wire").expect("wire");
        let blob = text(wire, "context_b64");
        if blob.len() < 8 {
            continue; // nothing to chop
        }
        let mut damaged = wire.clone();
        damaged["context_b64"] = Value::String(blob[..blob.len() - 4].to_string());
        let line = serde_json::to_vec(&damaged).expect("re-serializes");
        match parse_line(&line) {
            Err(refusal) => assert_eq!(
                refusal.kind(),
                "context_length_mismatch",
                "`{label}`: a truncated payload must be caught by the length, not by the codec"
            ),
            Ok(_) => panic!("`{label}`: a truncated payload was accepted"),
        }
        checked += 1;
    }
    assert!(checked >= 10, "only {checked} fixtures were long enough");
}

#[test]
fn the_escape_growth_bound_and_every_byte_cap_are_the_same_numbers_on_both_sides() {
    // **The pin `render.rs` claimed already existed.** Its comment read: "`ESCAPE_WORST_CASE_GROWTH`
    // in `python/qd_data/render.py` is the other half of the pair, and `tests/render_contract.rs`
    // pins them together." That was false, and in the strongest way — `render_contract.rs` compares
    // frozen golden *prompt bytes* captured from one Python run on 2026-09-19, reads no Python file
    // at test time, and its sample contains no escapable character, so it cannot observe this
    // constant even in principle. Its `the_default_caps_are_self_consistent` only calls Rust's own
    // `RenderCaps::DEFAULT.validate()`, which says nothing about Python.
    //
    // That matters because this exact pair has diverged before: `ESCAPE_WORST_CASE_GROWTH` was 6 in
    // Python and 2 in Rust, and for that window the two lanes refused *different* requests while
    // both suites were green (`GAP-RT-ESCAPE-GROWTH-FLOOR`). The numbers agree again today; what was
    // missing was anything that would say so tomorrow.
    let doc = doc_or_skip!();
    let inv = doc.get("inventory").expect("inventory");
    let caps = inv
        .get("render_caps")
        .and_then(Value::as_object)
        .expect("render_caps");
    let rust = RenderCaps::DEFAULT;

    assert_eq!(
        count(&Value::Object(caps.clone()), "escape_worst_case_growth"),
        ESCAPE_WORST_CASE_GROWTH as u64,
        "the escape-growth bound differs between the lanes. This is GAP-RT-ESCAPE-GROWTH-FLOOR \
         recurring: the lane with the smaller number refuses legal requests by arithmetic alone"
    );

    for (key, rust_value) in [
        ("max_context_bytes", rust.max_context_bytes),
        ("max_question_bytes", rust.max_question_bytes),
        ("max_option_bytes", rust.max_option_bytes),
        ("max_task_bytes", rust.max_task_bytes),
        ("max_rendered_bytes", rust.max_rendered_bytes),
    ] {
        assert_eq!(
            count(&Value::Object(caps.clone()), key),
            rust_value as u64,
            "RenderCaps.{key} differs between the lanes, so the two refuse different requests"
        );
    }

    // The *formula*, not just the constants. Python reported its own accept/reject verdict on two
    // configurations either side of its floor; Rust must agree on the same two. A lane that
    // computed the floor differently would pass the equality checks above and fail here.
    let floor = count(&Value::Object(caps.clone()), "probe_floor") as usize;
    assert_eq!(
        caps.get("accepts_at_floor"),
        Some(&Value::Bool(true)),
        "Python rejects its own floor, so the probe points are wrong and this test proves nothing"
    );
    assert_eq!(
        caps.get("accepts_below_floor"),
        Some(&Value::Bool(false)),
        "Python accepts a rendered cap below its own floor; the configuration check is not firing"
    );

    let at_floor = RenderCaps {
        max_rendered_bytes: floor,
        ..rust
    };
    let below_floor = RenderCaps {
        max_rendered_bytes: floor - 1,
        ..rust
    };
    assert!(
        at_floor.validate().is_ok(),
        "Python accepts max_rendered_bytes = {floor} and Rust refuses it: the two lanes compute \
         the escaped-size floor differently, so a caps config legal in one is illegal in the other"
    );
    assert!(
        below_floor.validate().is_err(),
        "Python refuses max_rendered_bytes = {} and Rust accepts it: Rust would then admit a \
         configuration under which a legal context is refused by arithmetic alone",
        floor - 1
    );
}
