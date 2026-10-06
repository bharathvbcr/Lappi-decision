//! The five steps of `docs/schema-api.md`'s answering procedure, with step 4 as the centrepiece.
//!
//! > For `choice`: run a **second pass with the options permuted** and require agreement.
//! > Disagreement is `noul`. This turns letter-position bias from a silent in-domain win into an
//! > abstention.
//!
//! A test that runs the second pass and observes agreement proves nothing, because a backend that
//! reads the *content* agrees trivially. So this file drives the procedure with two backends that
//! differ in exactly one respect:
//!
//! * [`Pick::AlwaysFirstLetter`] — a caricature of letter-position bias. It answers "A" whatever
//!   the options are, so the permuted pass names a different option and the slot **must** abstain.
//! * [`Pick::LowestOptionText`] — content-addressed. It names the same option whatever order they
//!   are in, so the slot **must** answer.
//!
//! Both are honest about what they are: they parse the option lines out of the rendered suffix,
//! which is the only information a model has.

mod common;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use qd_runtime::answer::{AbstainRule, abstain_rule, span_rows};
use qd_runtime::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, QueryKind, SlotQuery,
    StateBuffer, StateSnapshot,
};
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::context::Context;
use qd_runtime::reference::{REFERENCE_FEATURE_DIM, ReferenceBackend};
use qd_runtime::refusal::BackendError;
use qd_runtime::registry::{HeadMatrix, HeadRegistry, RegisteredHead};
use qd_runtime::render::{RenderCaps, second_pass_permutation};
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{ConformalSet, Response, Route, SlotAnswer, SlotKind, SlotValue};
use serde_json::json;

#[derive(Clone, Copy)]
enum Pick {
    /// Row 0, always. Letter-position bias in its purest form.
    AlwaysFirstLetter,
    /// The row whose option text sorts first. Invariant under permutation.
    LowestOptionText,
    /// The reserved abstain row.
    AlwaysNoul,
    /// A span whose end precedes its start.
    SpanBackwards,
    /// A span from line 1 to line 2.
    SpanForward,
}

struct Scripted {
    identity: BackendIdentity,
    inner: ReferenceBackend,
    pick: Pick,
}

impl Scripted {
    fn new(pick: Pick) -> Self {
        let inner = ReferenceBackend::new(true);
        let mut identity = inner.identity().clone();
        identity.name = "test-scripted".to_string();
        // Declared a model so `degraded` is not set for an unrelated reason, which keeps the
        // assertions in this file about the procedure rather than about the backend.
        identity.is_model = true;
        Self {
            identity,
            inner,
            pick,
        }
    }

    fn runtime(pick: Pick) -> Runtime {
        Runtime::with_backend(
            Arc::new(Self::new(pick)),
            CalibrationTable::reference(),
            HeadRegistry::new(),
            RenderCaps::DEFAULT,
        )
        .expect("assembles")
    }
}

/// Read the option lines back out of a rendered suffix, which is all a model can see.
fn options_in_prompt(suffix: &str) -> Vec<String> {
    let Some(start) = suffix.find("<|qd_options_begin|>\n") else {
        return Vec::new();
    };
    let Some(end) = suffix.find("<|qd_options_end|>") else {
        return Vec::new();
    };
    suffix[start + "<|qd_options_begin|>\n".len()..end]
        .lines()
        .map(|line| {
            line.split_once(". ")
                .map(|(_letter, text)| text)
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

fn one_hot(rows: usize, winner: usize, kind: QueryKind) -> Logits {
    let values = (0..rows)
        .map(|row| if row == winner { 8.0f32 } else { -8.0f32 })
        .collect();
    Logits { kind, values }
}

impl DecisionBackend for Scripted {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        self.inner.prefill(prefix)
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        self.inner.snapshot(handle)
    }

    fn decode_slot(
        &self,
        _snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        _mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        let noul_row = query.rows - 1;
        let winner = match (self.pick, query.kind) {
            (Pick::AlwaysNoul, _) => noul_row,
            (Pick::SpanBackwards, QueryKind::PointerStart) => 2.min(noul_row.saturating_sub(1)),
            (Pick::SpanBackwards, QueryKind::PointerEnd) => 0,
            (Pick::SpanForward, QueryKind::PointerStart) => 0,
            (Pick::SpanForward, QueryKind::PointerEnd) => 1.min(noul_row.saturating_sub(1)),
            (_, QueryKind::PointerStart) => 0,
            (_, QueryKind::PointerEnd) => noul_row.saturating_sub(1),
            (Pick::AlwaysFirstLetter, QueryKind::Letters) => 0,
            // The span scripts say nothing about letters; answer the first row, which is what the
            // tests that mix a span with other slots expect.
            (Pick::SpanBackwards | Pick::SpanForward, QueryKind::Letters) => 0,
            (Pick::LowestOptionText, QueryKind::Letters) => {
                let options = options_in_prompt(query.suffix);
                // The last rendered line is the reserved `noul` row; never pick it here.
                options
                    .iter()
                    .take(noul_row)
                    .enumerate()
                    .min_by(|a, b| a.1.cmp(b.1))
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            }
        };
        Ok(one_hot(query.rows, winner, query.kind))
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.inner.pooled_features(snapshot)
    }
}

fn choice_request(options: &[&str]) -> serde_json::Value {
    common::request_with_slots(
        common::SAMPLE_CONTEXT.as_bytes(),
        json!([{"name": "verdict", "type": "choice", "options": options}]),
    )
}

fn answer_of(runtime: &Runtime, value: &serde_json::Value, slot: &str) -> SlotAnswer {
    let request = common::validated(value);
    match runtime.answer(&request, None) {
        Response::Ok(envelope) => envelope
            .slots
            .get(slot)
            .cloned()
            .unwrap_or_else(|| panic!("no slot `{slot}` in the answer")),
        other => panic!("expected an answer, got {other:?}"),
    }
}

// -- step 4 -----------------------------------------------------------------------------------

#[test]
fn letter_position_bias_becomes_an_abstention() {
    let runtime = Scripted::runtime(Pick::AlwaysFirstLetter);
    let answer = answer_of(
        &runtime,
        &choice_request(&["stub", "logic", "cosmetic", "clean"]),
        "verdict",
    );
    assert!(
        answer.noul,
        "a backend that always answers the first letter must be caught by the permuted pass, got \
         {answer:?}"
    );
    assert!(answer.value.is_none(), "an abstention carries no value");
    assert!(answer.conformal_set.is_none());
}

#[test]
fn a_content_addressed_answer_survives_the_permuted_pass() {
    let runtime = Scripted::runtime(Pick::LowestOptionText);
    let answer = answer_of(
        &runtime,
        &choice_request(&["stub", "logic", "cosmetic", "clean"]),
        "verdict",
    );
    assert!(
        !answer.noul,
        "a backend that reads the options must be allowed to answer, got {answer:?}"
    );
    // "clean" sorts first of the four.
    assert_eq!(answer.value, Some(SlotValue::Choice("clean".to_string())));
    assert!(
        answer.score > 0.0,
        "the score is a margin, and it is positive here"
    );
}

/// The property `letter_position_bias_becomes_an_abstention` depends on, stated directly.
///
/// A merely non-identity permutation is not enough: if the winning row happens to be a fixed point,
/// a position-biased backend agrees with itself and the abstention silently does not fire. Every
/// option must move, for every seed, at every `n`.
#[test]
fn the_second_pass_permutation_is_a_derangement() {
    for n in 2..=16usize {
        for slot in ["verdict", "severity", "a", "zzz"] {
            for seed in 0u8..32 {
                let digest = [seed; 32];
                let perm = second_pass_permutation(&digest, slot, n);
                assert_eq!(perm.len(), n);
                let mut sorted = perm.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "not a permutation");
                for (row, option) in perm.iter().enumerate() {
                    assert_ne!(
                        row, *option,
                        "n={n} slot={slot} seed={seed}: option {option} did not move, so a \
                         position-biased backend would agree with itself on it"
                    );
                }
            }
        }
    }
    // n < 2 cannot be deranged, and cannot reach here either: `wire::validate` refuses a choice
    // slot with fewer than two options.
    assert_eq!(second_pass_permutation(&[0u8; 32], "x", 1), vec![0]);
}

#[test]
fn the_second_pass_permutation_is_reproducible_for_a_given_request() {
    let digest = [3u8; 32];
    assert_eq!(
        second_pass_permutation(&digest, "verdict", 8),
        second_pass_permutation(&digest, "verdict", 8)
    );
    assert_ne!(
        second_pass_permutation(&digest, "verdict", 8),
        second_pass_permutation(&[4u8; 32], "verdict", 8),
        "a different request must draw a different permutation"
    );
}

#[test]
fn an_abstaining_backend_abstains_rather_than_returning_a_letter() {
    let runtime = Scripted::runtime(Pick::AlwaysNoul);
    let answer = answer_of(&runtime, &choice_request(&["stub", "clean"]), "verdict");
    assert!(answer.noul);
    assert!(answer.value.is_none());
    // And the response still reads as an answer, not as a refusal: the model *was* asked.
    let request = common::validated(&choice_request(&["stub", "clean"]));
    assert_eq!(
        runtime.answer(&request, None).caller_reading(),
        qd_runtime::schema::CallerReading::ModelAbstained
    );
}

// -- score slots are ordinal and are never permuted ----------------------------------------------

#[test]
fn a_score_slot_is_answered_without_a_second_pass() {
    // The same position-biased backend that cannot answer a `choice` answers a `score`, because the
    // bins are ordered and permuting them would destroy the ordering the loss depends on.
    let runtime = Scripted::runtime(Pick::AlwaysFirstLetter);
    let value = common::request_with_slots(
        common::SAMPLE_CONTEXT.as_bytes(),
        json!([{"name": "severity", "type": "score", "bins": 5}]),
    );
    let answer = answer_of(&runtime, &value, "severity");
    assert!(
        !answer.noul,
        "a score slot has no permuted pass: {answer:?}"
    );
    assert_eq!(
        answer.value,
        Some(SlotValue::Score(1)),
        "row 0 is bin 1, and the bins are rendered in order"
    );
    match answer.conformal_set {
        Some(ConformalSet::Scores(set)) => {
            assert!(set.contains(&1));
            assert!(
                !set.is_empty() && set.iter().all(|bin| (1..=5).contains(bin)),
                "the conformal set holds bins, never the reserved abstain row: {set:?}"
            );
        }
        other => panic!("a score slot answers with a score conformal set, got {other:?}"),
    }
}

// -- span slots -----------------------------------------------------------------------------------

#[test]
fn the_pointer_head_ranges_over_the_contexts_line_starts_plus_the_abstain_row() {
    for (text, lines) in [
        ("a\n", 1usize),
        ("a\nb\n", 2),
        ("a\nb\nc", 3),
        ("a\nb\nc\n", 3),
        ("no newline at all", 1),
    ] {
        let context = Context::from_bytes(text.as_bytes().to_vec());
        assert_eq!(context.line_count(), lines, "line count of {text:?}");
        assert_eq!(span_rows(&context), lines + 1, "span rows of {text:?}");
    }
}

#[test]
fn a_span_is_answered_as_one_based_inclusive_lines() {
    let runtime = Scripted::runtime(Pick::SpanForward);
    let value = common::request_with_slots(
        b"alpha\nbeta\ngamma\n",
        json!([{"name": "evidence", "type": "span"}]),
    );
    let answer = answer_of(&runtime, &value, "evidence");
    assert!(!answer.noul, "{answer:?}");
    assert_eq!(
        answer.value,
        Some(SlotValue::Span(qd_runtime::schema::SpanValue {
            start_line: 1,
            end_line: 2
        }))
    );
    assert!(
        answer.conformal_set.is_none(),
        "the contract's example has `conformal_set: null` for a span"
    );
}

#[test]
fn a_span_that_runs_backwards_abstains_rather_than_being_reordered() {
    let runtime = Scripted::runtime(Pick::SpanBackwards);
    let value = common::request_with_slots(
        b"alpha\nbeta\ngamma\ndelta\n",
        json!([{"name": "evidence", "type": "span"}]),
    );
    let answer = answer_of(&runtime, &value, "evidence");
    assert!(
        answer.noul,
        "an end before a start is not a low-confidence span, it is not a span: {answer:?}"
    );
}

// -- the whole envelope ----------------------------------------------------------------------------

#[test]
fn every_requested_slot_appears_in_the_answer_exactly_once() {
    let runtime = Scripted::runtime(Pick::LowestOptionText);
    let request = common::validated(&common::sample_request());
    match runtime.answer(&request, None) {
        Response::Ok(envelope) => {
            let names: Vec<&String> = envelope.slots.keys().collect();
            assert_eq!(names, vec!["evidence", "severity", "verdict"]);
            assert_eq!(envelope.schema_version, 1);
            assert_eq!(envelope.backend, "test-scripted");
        }
        other => panic!("expected an answer, got {other:?}"),
    }
}

#[test]
fn the_same_request_answers_identically_twice() {
    let runtime = common::reference_runtime();
    let request = common::validated(&common::sample_request());
    let first = serde_json::to_string(&runtime.answer(&request, None)).expect("serializes");
    let second = serde_json::to_string(&runtime.answer(&request, None)).expect("serializes");
    assert_eq!(first, second);
}

#[test]
fn the_conformal_set_holds_values_and_never_the_abstain_row() {
    let runtime = Scripted::runtime(Pick::LowestOptionText);
    let answer = answer_of(
        &runtime,
        &choice_request(&["stub", "logic", "cosmetic", "clean"]),
        "verdict",
    );
    match answer.conformal_set {
        Some(ConformalSet::Choices(set)) => {
            assert!(set.contains(&"clean".to_string()), "{set:?}");
            assert!(
                !set.iter().any(|v| v == "noul"),
                "the abstain row is reported through the `noul` flag, never as a value: {set:?}"
            );
            assert!(
                set.iter()
                    .all(|v| ["stub", "logic", "cosmetic", "clean"].contains(&v.as_str())),
                "{set:?}"
            );
        }
        other => panic!("a choice answers with a choice conformal set, got {other:?}"),
    }
}

/// The contract's own example, answered end to end.
#[test]
fn the_contracts_example_request_is_answerable() {
    let runtime = Scripted::runtime(Pick::LowestOptionText);
    let request = common::validated(&common::sample_request());
    let Response::Ok(envelope) = runtime.answer(&request, None) else {
        panic!("the contract's own example must be answerable");
    };
    let mut kinds: BTreeMap<&str, bool> = BTreeMap::new();
    for (name, slot) in &envelope.slots {
        kinds.insert(name.as_str(), slot.noul);
        assert!(
            slot.score.is_finite(),
            "slot `{name}` reported a non-finite score"
        );
        assert!(
            (0.0..=1.0).contains(&slot.score),
            "a margin on calibrated probabilities is in [0, 1], got {}",
            slot.score
        );
    }
    assert_eq!(kinds.len(), 3);
}

// -- how many decodes, over which rows, on which route ----------------------------------------------
//
// `GAP-RT-SPEC-SPAN-DECODE` and `GAP-RT-SPEC-REGISTERED-ROUTE`. The contract says every slot is
// "a 1-token query" and that a span answers `{start_line, end_line}`; those cannot both be
// literally true for a two-ended answer. And it says "same abstain rule" for a route that has no
// option text to permute. Both are settled by counting what the runtime actually asks the backend
// for, rather than by a sentence.

/// Records every call, so "how many decodes" is a measured number and not a reading of the source.
struct Counting {
    identity: BackendIdentity,
    inner: ReferenceBackend,
    decodes: Mutex<Vec<(String, QueryKind, usize, String)>>,
    pooled: AtomicUsize,
}

impl Counting {
    fn new() -> Self {
        let inner = ReferenceBackend::new(true);
        let mut identity = inner.identity().clone();
        identity.name = "test-counting".to_string();
        identity.is_model = true;
        Self {
            identity,
            inner,
            decodes: Mutex::new(Vec::new()),
            pooled: AtomicUsize::new(0),
        }
    }

    /// `(slot name, query kind, rows, suffix)` for every decode, in order.
    fn decodes(&self) -> Vec<(String, QueryKind, usize, String)> {
        self.decodes.lock().expect("not poisoned").clone()
    }
}

impl DecisionBackend for Counting {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        self.inner.prefill(prefix)
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        self.inner.snapshot(handle)
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        self.decodes.lock().expect("not poisoned").push((
            query.slot_name.to_string(),
            query.kind,
            query.rows,
            query.suffix.to_string(),
        ));
        self.inner.decode_slot(snapshot, query, mode)
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.pooled.fetch_add(1, Ordering::SeqCst);
        self.inner.pooled_features(snapshot)
    }
}

fn counting_runtime(registry: HeadRegistry) -> (Runtime, Arc<Counting>) {
    let backend = Arc::new(Counting::new());
    let runtime = Runtime::with_backend(
        backend.clone(),
        CalibrationTable::reference(),
        registry,
        RenderCaps::DEFAULT,
    )
    .expect("assembles");
    (runtime, backend)
}

#[test]
fn a_span_slot_takes_exactly_two_pointer_decodes_from_one_snapshot() {
    // Four lines of context, so the pointer head's row count is a number the test can name.
    let context = b"alpha\nbeta\ngamma\ndelta\n";
    let expected_rows = span_rows(&Context::from_bytes(context.to_vec()));
    assert_eq!(
        expected_rows, 5,
        "four line starts plus the reserved noul row"
    );

    let (runtime, backend) = counting_runtime(HeadRegistry::new());
    let value = common::request_with_slots(context, json!([{"name": "evidence", "type": "span"}]));
    let request = common::validated(&value);
    let Response::Ok(_) = runtime.answer(&request, None) else {
        panic!("a span slot must be answerable on the generic route");
    };

    let decodes = backend.decodes();
    assert_eq!(
        decodes.len(),
        2,
        "a span is two 1-token pointer reads, not one decode read twice and not three: {decodes:?}"
    );
    assert_eq!(decodes[0].1, QueryKind::PointerStart);
    assert_eq!(decodes[1].1, QueryKind::PointerEnd);
    for (name, kind, rows, _) in &decodes {
        assert_eq!(name, "evidence");
        assert_eq!(
            *rows, expected_rows,
            "the {kind:?} decode must range over this context's line starts plus the abstain row"
        );
    }
    assert_eq!(
        decodes[0].3, decodes[1].3,
        "both ends are read from the same rendered suffix and the same snapshot; a differing \
         suffix would mean the two ends saw different prompts"
    );
}

#[test]
fn a_choice_slot_is_one_batched_decode_covering_both_passes() {
    let backend = Arc::new(BatchProbe::new());
    let runtime = Runtime::with_backend(
        backend.clone(),
        CalibrationTable::reference(),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("assembles");
    let request = common::validated(&choice_request(&["stub", "logic", "cosmetic", "clean"]));
    let Response::Ok(_) = runtime.answer(&request, None) else {
        panic!("a choice slot must be answerable");
    };
    assert_eq!(
        backend.batched.load(Ordering::SeqCst),
        1,
        "both passes are one decode_slots call"
    );
    assert_eq!(
        backend.single.load(Ordering::SeqCst),
        0,
        "the batch path must not also call decode_slot"
    );
    let seen = backend
        .seen
        .lock()
        .expect("batch probe")
        .clone()
        .expect("decode_slots ran");
    assert_eq!(
        seen.mode,
        DecodeMode::ReadOnly,
        "write-back must not be the batched call"
    );
    assert_eq!(seen.kinds, vec![QueryKind::Letters, QueryKind::Letters]);
    assert_eq!(seen.rows.len(), 2);
    assert_eq!(
        seen.rows[0], seen.rows[1],
        "the two passes share one row count"
    );
    assert_eq!(seen.rows[0], 5, "four options plus the reserved noul row");
    assert_ne!(
        seen.suffixes[0], seen.suffixes[1],
        "the second pass must be a different suffix, or the batch is the first pass twice"
    );
}

#[test]
fn decode_slots_refuses_an_empty_query_list() {
    let backend = ReferenceBackend::new(true);
    let mut snapshot = StateSnapshot {
        backend: "reference".to_string(),
        prompt_digest: [0; 32],
        token_count: 0,
        state: StateBuffer::from_bytes(vec![1]),
    };
    let err = backend
        .decode_slots(&mut snapshot, &[], DecodeMode::ReadOnly)
        .expect_err("no queries");
    let BackendError::DecodeFailed { detail } = err else {
        panic!("expected DecodeFailed, got {err:?}");
    };
    assert!(detail.contains("no queries"), "{detail}");
}

#[derive(Clone)]
struct SeenBatch {
    mode: DecodeMode,
    kinds: Vec<QueryKind>,
    rows: Vec<usize>,
    suffixes: Vec<String>,
}

struct BatchProbe {
    identity: BackendIdentity,
    inner: ReferenceBackend,
    batched: AtomicUsize,
    single: AtomicUsize,
    seen: Mutex<Option<SeenBatch>>,
}

impl BatchProbe {
    fn new() -> Self {
        let inner = ReferenceBackend::new(true);
        let mut identity = inner.identity().clone();
        identity.name = "test-batch-probe".to_string();
        identity.is_model = true;
        Self {
            identity,
            inner,
            batched: AtomicUsize::new(0),
            single: AtomicUsize::new(0),
            seen: Mutex::new(None),
        }
    }
}

impl DecisionBackend for BatchProbe {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        self.inner.prefill(prefix)
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        self.inner.snapshot(handle)
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        self.single.fetch_add(1, Ordering::SeqCst);
        self.inner.decode_slot(snapshot, query, mode)
    }

    fn decode_slots(
        &self,
        snapshot: &mut StateSnapshot,
        queries: &[SlotQuery<'_>],
        mode: DecodeMode,
    ) -> Result<Vec<Logits>, BackendError> {
        self.batched.fetch_add(1, Ordering::SeqCst);
        *self.seen.lock().expect("batch probe") = Some(SeenBatch {
            mode,
            kinds: queries.iter().map(|q| q.kind).collect(),
            rows: queries.iter().map(|q| q.rows).collect(),
            suffixes: queries.iter().map(|q| q.suffix.to_string()).collect(),
        });
        let mut out = Vec::with_capacity(queries.len());
        for query in queries {
            out.push(self.inner.decode_slot(snapshot, query, mode)?);
        }
        Ok(out)
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.inner.pooled_features(snapshot)
    }
}

#[test]
fn a_choice_slot_takes_two_decodes_with_different_suffixes_and_a_score_slot_takes_one() {
    let (runtime, backend) = counting_runtime(HeadRegistry::new());
    let request = common::validated(&choice_request(&["stub", "logic", "cosmetic", "clean"]));
    let Response::Ok(_) = runtime.answer(&request, None) else {
        panic!("a choice slot must be answerable");
    };
    let decodes = backend.decodes();
    assert_eq!(decodes.len(), 2, "step 4's second pass: {decodes:?}");
    assert_ne!(
        decodes[0].3, decodes[1].3,
        "the second pass must re-render the options in a different order, or it is the first pass \
         run twice"
    );
    assert_eq!(decodes[0].2, 5, "four options plus the reserved noul row");
    assert_eq!(decodes[1].2, 5);

    let (runtime, backend) = counting_runtime(HeadRegistry::new());
    let value = common::request_with_slots(
        common::SAMPLE_CONTEXT.as_bytes(),
        json!([{"name": "severity", "type": "score", "bins": 5}]),
    );
    let request = common::validated(&value);
    let Response::Ok(_) = runtime.answer(&request, None) else {
        panic!("a score slot must be answerable");
    };
    let decodes = backend.decodes();
    assert_eq!(
        decodes.len(),
        1,
        "ordinal bins are never permuted: the loss is cumulative and permuting destroys the \
         ordering it depends on. {decodes:?}"
    );
    assert_eq!(decodes[0].2, 6, "five bins plus the reserved noul row");
}

#[test]
fn the_registered_route_has_no_second_pass_to_run_and_the_rule_table_says_so() {
    // `GAP-RT-SPEC-REGISTERED-ROUTE`, half one. "Same abstain rule" cannot be literally true: the
    // permutation half needs option text in the prompt, and the registered route has none. This
    // measures the consequence — the registered route issues **no decode at all**, so there is no
    // second pass that could fire — rather than asserting the sentence.
    let backend = ReferenceBackend::new(true);
    let backbone = backend.identity().weight_hash.clone();
    let mut slots = BTreeMap::new();
    slots.insert(
        "verdict".to_string(),
        HeadMatrix {
            rows: vec![vec![0.01f32; REFERENCE_FEATURE_DIM]; 5],
        },
    );
    let mut registry = HeadRegistry::new();
    registry.insert(RegisteredHead {
        task: "devcouncil.verdict".to_string(),
        head_hash: "head-under-test".to_string(),
        backbone_hash: backbone,
        feature_dim: REFERENCE_FEATURE_DIM,
        slots,
    });

    let (runtime, counting) = counting_runtime(registry);
    let mut value = choice_request(&["stub", "logic", "cosmetic", "clean"]);
    value["route"] = json!("registered");
    let request = common::validated(&value);
    let Response::Ok(_) = runtime.answer(&request, None) else {
        panic!("a bound head must answer");
    };
    assert!(
        counting.decodes().is_empty(),
        "the registered route answers from one GEMV on pooled features; any decode here would \
         mean it had fallen through to the generic route: {:?}",
        counting.decodes()
    );
    assert_eq!(
        counting.pooled.load(Ordering::SeqCst),
        1,
        "one pooled-feature read per request, from the same snapshot as the generic route's"
    );

    // And the table that states this is the same one the runtime consults for the refusal.
    assert_eq!(
        abstain_rule(Route::Generic, SlotKind::Choice),
        Some(AbstainRule::MarginAndPermutedSecondPass)
    );
    assert_eq!(
        abstain_rule(Route::Registered, SlotKind::Choice),
        Some(AbstainRule::MarginOnly),
        "the registered route runs the margin half of the abstain rule and nothing else"
    );
    assert!(
        !abstain_rule(Route::Registered, SlotKind::Choice)
            .expect("served")
            .runs_permuted_second_pass()
    );
}

#[test]
fn the_abstain_rule_table_is_total_and_names_exactly_one_unserved_pair() {
    let mut unserved = Vec::new();
    for route in [Route::Generic, Route::Registered] {
        for kind in [SlotKind::Choice, SlotKind::Score, SlotKind::Span] {
            match abstain_rule(route, kind) {
                // Every served pair runs the margin half; that much of "same abstain rule" does
                // hold, and naming the variant here is what makes a silent re-tabling fail.
                Some(AbstainRule::MarginOnly)
                | Some(AbstainRule::MarginAndPermutedSecondPass)
                | Some(AbstainRule::MarginAndSpanOrdering) => {}
                None => unserved.push((route, kind)),
            }
        }
    }
    assert_eq!(
        unserved,
        vec![(Route::Registered, SlotKind::Span)],
        "a registered-route span is the one pair this runtime does not serve; adding another is a \
         contract change, not an implementation detail"
    );
}

// -- the reserved abstain row is LAST, for every slot kind ------------------------------------------

/// Answers a fixed row for every query, whatever the kind.
struct PicksRow {
    identity: BackendIdentity,
    inner: ReferenceBackend,
    /// `None` means "the last row", which is where the reserved `noul` row is claimed to be.
    row: Option<usize>,
}

impl PicksRow {
    fn runtime(row: Option<usize>) -> Runtime {
        let inner = ReferenceBackend::new(true);
        let mut identity = inner.identity().clone();
        identity.name = "test-picks-row".to_string();
        identity.is_model = true;
        Runtime::with_backend(
            Arc::new(Self {
                identity,
                inner,
                row,
            }),
            CalibrationTable::reference(),
            HeadRegistry::new(),
            RenderCaps::DEFAULT,
        )
        .expect("assembles")
    }
}

impl DecisionBackend for PicksRow {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        self.inner.prefill(prefix)
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        self.inner.snapshot(handle)
    }

    fn decode_slot(
        &self,
        _snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        _mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        let winner = self.row.unwrap_or(query.rows - 1);
        Ok(one_hot(query.rows, winner, query.kind))
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.inner.pooled_features(snapshot)
    }
}

#[test]
fn the_reserved_abstain_row_is_the_last_row_of_every_slot_kind() {
    // The hazard this guards: if a head were trained with the abstain column **first** and served
    // by a runtime reading it **last**, every "no evidence" answer would land on the context's
    // final line, or on option A, and no loss curve would show it. `answer.rs` computes
    // `noul_row = rows - RESERVED_NOUL_ROWS` for all three kinds, and `python/qd_train/heads.py`
    // puts the abstain column at index `n_candidates`. This asserts the runtime half of that
    // agreement behaviourally, for every kind at once.
    let runtime = PicksRow::runtime(None);
    let request = common::validated(&common::sample_request());
    let Response::Ok(envelope) = runtime.answer(&request, None) else {
        panic!("expected an answer");
    };
    for (name, slot) in &envelope.slots {
        assert!(
            slot.noul,
            "slot `{name}`: a backend that wins the LAST row must read as an abstention, because \
             the last row is the reserved noul row: {slot:?}"
        );
        assert_eq!(slot.value, None, "slot `{name}`");
    }
}

#[test]
fn row_zero_is_a_named_answer_and_never_the_abstention() {
    // The control for the test above. Without it, a runtime that abstained on *everything* would
    // pass that one for the wrong reason.
    let runtime = PicksRow::runtime(Some(0));
    let value = common::request_with_slots(
        b"alpha\nbeta\ngamma\n",
        json!([
            {"name": "verdict", "type": "choice", "options": ["stub", "logic", "cosmetic", "clean"]},
            {"name": "severity", "type": "score", "bins": 5},
            {"name": "evidence", "type": "span"}
        ]),
    );
    let request = common::validated(&value);
    let Response::Ok(envelope) = runtime.answer(&request, None) else {
        panic!("expected an answer");
    };

    // `verdict` abstains for a *different* reason — row 0 is letter-position bias, and step 4's
    // permuted pass catches it — which is itself the contract working, so it is asserted as that
    // and not as an abstain-row fact.
    let verdict = &envelope.slots["verdict"];
    assert!(
        verdict.noul,
        "row 0 on both passes names two different options, so step 4 must abstain: {verdict:?}"
    );

    let severity = &envelope.slots["severity"];
    assert!(
        !severity.noul,
        "row 0 of a score slot is bin 1, not the abstention: {severity:?}"
    );
    assert_eq!(severity.value, Some(SlotValue::Score(1)));

    let evidence = &envelope.slots["evidence"];
    assert!(
        !evidence.noul,
        "row 0 of a pointer head is line 1, not the abstention: {evidence:?}"
    );
    assert_eq!(
        evidence.value,
        Some(SlotValue::Span(qd_runtime::schema::SpanValue {
            start_line: 1,
            end_line: 1
        })),
        "row 0 for both ends is the one-line span on line 1"
    );
}
