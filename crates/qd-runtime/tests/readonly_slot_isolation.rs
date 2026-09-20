//! **`readonly` decode leaves the state buffer hash-identical.**
//!
//! `docs/hardening.md` §6: *"This is the slot-isolation guarantee; if it fails, slot 2's answer
//! depends on slot 1's, and the whole snapshot design is void."*
//!
//! A test that only asserts "the hash did not change" passes against a backend that never touches
//! the state, which is the easiest way to write a green test that proves nothing. So this file
//! asserts three things together, and the middle one is what makes the other two mean something:
//!
//! 1. a read-only decode leaves the hash identical;
//! 2. a **write-back** decode changes it — so the buffer is genuinely reachable and the hash is
//!    genuinely sensitive;
//! 3. a backend that writes back while the caller asked for read-only is **caught**, with both
//!    hashes in the error.
//!
//! Then the property itself, at the level a caller cares about: slot 2's logits are the same
//! whether or not slot 1 was answered first.

mod common;

use common::{Behaviour, Wrapped};
use qd_runtime::answer::readonly_decode;
use qd_runtime::backend::{
    DecisionBackend, DecodeMode, QueryKind, SlotIsolationCheck, SlotQuery,
};
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::refusal::BackendError;
use qd_runtime::render::{render, RenderCaps};
use qd_runtime::schema::{CallerReading, Response};
use serde_json::json;

fn letters_query<'a>(name: &'a str, suffix: &'a str, rows: usize) -> SlotQuery<'a> {
    SlotQuery {
        slot_name: name,
        suffix,
        rows,
        kind: QueryKind::Letters,
    }
}

#[test]
fn a_readonly_decode_leaves_the_state_buffer_hash_identical() {
    let backend = ReferenceBackend::new(true);
    let handle = backend.prefill("prefix").expect("prefills");
    let mut snapshot = backend.snapshot(&handle).expect("snapshots");

    let before = snapshot.state_digest();
    let query = letters_query("verdict", "suffix", 5);
    backend
        .decode_slot(&mut snapshot, &query, DecodeMode::ReadOnly)
        .expect("decodes");
    assert_eq!(
        snapshot.state_digest(),
        before,
        "a read-only decode moved the state buffer"
    );
}

/// Without this, the assertion above is satisfied by a backend that does nothing at all.
#[test]
fn a_writeback_decode_does_change_the_state_buffer_hash() {
    let backend = ReferenceBackend::new(true);
    let handle = backend.prefill("prefix").expect("prefills");
    let mut snapshot = backend.snapshot(&handle).expect("snapshots");

    let before = snapshot.state_digest();
    let query = letters_query("verdict", "suffix", 5);
    backend
        .decode_slot(&mut snapshot, &query, DecodeMode::WriteBack)
        .expect("decodes");
    assert_ne!(
        snapshot.state_digest(),
        before,
        "the write-back path does not touch the state, so the read-only test is vacuous"
    );
}

#[test]
fn slot_two_does_not_depend_on_slot_one() {
    let backend = ReferenceBackend::new(true);
    let handle = backend.prefill("shared prefix").expect("prefills");

    let first = letters_query("verdict", "<|qd_slot|>verdict\n", 5);
    let second = letters_query("severity", "<|qd_slot|>severity\n", 6);

    // Answer slot 1, then slot 2, from one snapshot.
    let mut snapshot = backend.snapshot(&handle).expect("snapshots");
    let (_, check_a) = readonly_decode(&backend, &mut snapshot, &first, 0).expect("slot 1 decodes");
    let (after_first, check_b) =
        readonly_decode(&backend, &mut snapshot, &second, 1).expect("slot 2 decodes");

    // Answer slot 2 alone, from a fresh snapshot of the same prefill.
    let mut fresh = backend.snapshot(&handle).expect("snapshots");
    let (alone, _) = readonly_decode(&backend, &mut fresh, &second, 0).expect("slot 2 decodes");

    assert_eq!(
        after_first.values, alone.values,
        "slot 2's logits changed because slot 1 was answered first"
    );
    assert!(check_a.ran() && check_b.ran(), "both checks must have run");
}

#[test]
fn the_check_records_the_hash_it_compared() {
    let backend = ReferenceBackend::new(true);
    let handle = backend.prefill("prefix").expect("prefills");
    let mut snapshot = backend.snapshot(&handle).expect("snapshots");
    let expected = snapshot.state_digest_hex();

    let query = letters_query("verdict", "suffix", 5);
    let (_, check) = readonly_decode(&backend, &mut snapshot, &query, 0).expect("decodes");
    match check {
        SlotIsolationCheck::Ran { state_hash } => assert_eq!(state_hash, expected),
        SlotIsolationCheck::NotRun { reason } => {
            panic!("the reference backend's state is host-visible, so the check must run: {reason}")
        }
    }
}

#[test]
fn a_backend_that_writes_back_under_readonly_is_caught() {
    let backend = Wrapped::new(Behaviour::LeaksUnderReadonly);
    let handle = backend.prefill("prefix").expect("prefills");
    let mut snapshot = backend.snapshot(&handle).expect("snapshots");
    let before = snapshot.state_digest_hex();

    let query = letters_query("verdict", "suffix", 5);
    match readonly_decode(&backend, &mut snapshot, &query, 3) {
        Err(BackendError::ReadonlyViolated {
            slot_index,
            before: reported_before,
            after,
        }) => {
            assert_eq!(slot_index, 3, "the error must name the slot");
            assert_eq!(reported_before, before);
            assert_ne!(reported_before, after, "both hashes must be carried");
        }
        other => panic!("a leaking backend must be caught, got {other:?}"),
    }
}

#[test]
fn a_leaking_backend_fails_the_whole_request_rather_than_answering() {
    let runtime = Wrapped::runtime(Behaviour::LeaksUnderReadonly);
    let request = common::validated(&common::sample_request());
    let response = runtime.answer(&request, None);

    assert_eq!(response.caller_reading(), CallerReading::BackendFailed);
    match &response {
        Response::Error(envelope) => {
            assert_eq!(envelope.error.kind(), "readonly_violated");
        }
        other => panic!("expected a backend failure, got {other:?}"),
    }
}

// -- when the check cannot run --------------------------------------------------------------

#[test]
fn an_unverifiable_state_degrades_the_answer_rather_than_passing_silently() {
    let request = common::validated(&common::sample_request());

    // Control: a backend that claims to be a model and exposes its state answers clean.
    let control = Wrapped::runtime(Behaviour::ClaimsToBeAModel);
    let clean = control.answer(&request, None);
    let clean_degraded = match &clean {
        Response::Ok(envelope) => envelope.degraded,
        other => panic!("the control must answer, got {other:?}"),
    };
    assert!(
        !clean_degraded,
        "the control must be clean, or `degraded` below proves nothing"
    );

    // The case: identical in every way except that the state cannot be hashed.
    let opaque = Wrapped::runtime(Behaviour::OpaqueState);
    match opaque.answer(&request, None) {
        Response::Ok(envelope) => {
            assert!(
                envelope.degraded,
                "a slot-isolation check that could not run must not read as one that passed"
            );
            assert!(
                envelope.slots.values().all(|slot| slot.degraded),
                "every slot of a degraded answer carries the flag"
            );
        }
        other => panic!("an opaque-state backend still answers, got {other:?}"),
    }
}

#[test]
fn a_not_run_check_says_why() {
    let backend = Wrapped::new(Behaviour::OpaqueState);
    let handle = backend.prefill("prefix").expect("prefills");
    let mut snapshot = backend.snapshot(&handle).expect("snapshots");
    let query = letters_query("verdict", "suffix", 5);
    match readonly_decode(&backend, &mut snapshot, &query, 0).expect("decodes") {
        (_, SlotIsolationCheck::NotRun { reason }) => {
            assert!(
                reason.contains("state_host_visible"),
                "the reason must name the capability that is missing: {reason}"
            );
        }
        (_, SlotIsolationCheck::Ran { .. }) => {
            panic!("a backend with no host-visible state cannot have run the check")
        }
    }
}

// -- docs/hardening.md §6: recycled scratch --------------------------------------------------

#[test]
fn a_second_request_does_not_read_the_first_requests_leftovers() {
    let runtime = common::reference_runtime();
    let first_request = common::validated(&common::request_with_slots(
        b"fn one() { todo!() }\n",
        json!([{"name": "verdict", "type": "choice", "options": ["stub", "clean"]}]),
    ));
    let target = common::validated(&common::sample_request());

    let alone = runtime.answer(&target, None);
    let _ = runtime.answer(&first_request, None);
    let after_another = runtime.answer(&target, None);

    assert_eq!(
        serde_json::to_string(&alone).expect("serializes"),
        serde_json::to_string(&after_another).expect("serializes"),
        "the same request answered differently once another request had run"
    );
}

#[test]
fn the_prefill_happens_once_per_request_whatever_the_slot_count() {
    // The snapshot is the prefix cache: the prefix is identical across slots, so every slot decodes
    // against one prefill. Asserted through the renderer, which is where the split lives.
    let request = common::validated(&common::sample_request());
    let prompt = render(&request, &RenderCaps::DEFAULT).expect("renders");
    assert_eq!(prompt.slots.len(), 3);
    for full in prompt.prompts() {
        assert!(
            full.starts_with(&prompt.prefix),
            "every slot's prompt must be the one shared prefix plus its own suffix"
        );
    }
}
