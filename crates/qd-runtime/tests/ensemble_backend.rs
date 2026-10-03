//! `EnsembleBackend`: N members, one decode — the mean of the members' per-row log-softmax —
//! and every way it refuses rather than serve fewer members than it names.

use std::sync::Arc;

use qd_runtime::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, QueryKind, SlotQuery,
    StateBuffer, StateSnapshot,
};
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::ensemble::{EnsembleBackend, MAX_MEMBERS, ensemble_weight_hash};
use qd_runtime::reference::ReferenceBackend;
use qd_runtime::refusal::BackendError;
use qd_runtime::registry::HeadRegistry;
use qd_runtime::render::RenderCaps;
use qd_runtime::runtime::Runtime;
use qd_runtime::schema::{DecisionRequest, Response};
use qd_runtime::wire::{Incoming, parse_line};
use serde_json::json;

#[derive(Clone, Copy)]
enum Fault {
    None,
    Fails,
    NonFinite,
    WritesState,
    /// Claims host-visible state (the reference identity does) but prefills and snapshots none.
    EmptyState,
}

/// The reference backend's prefill and snapshot, with chosen logits and a chosen identity.
struct Scripted {
    inner: ReferenceBackend,
    identity: BackendIdentity,
    logits: Vec<f32>,
    fault: Fault,
}

fn identity(weight: &str) -> BackendIdentity {
    let mut id = ReferenceBackend::new(true).identity().clone();
    id.weight_hash = qd_runtime::hex(&qd_runtime::sha256(weight.as_bytes()));
    id
}

fn member(weight: &str, logits: &[f32]) -> Scripted {
    Scripted {
        inner: ReferenceBackend::new(true),
        identity: identity(weight),
        logits: logits.to_vec(),
        fault: Fault::None,
    }
}

impl DecisionBackend for Scripted {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        let mut handle = self.inner.prefill(prefix)?;
        if let Fault::EmptyState = self.fault {
            handle.state = StateBuffer::opaque();
        }
        Ok(handle)
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        let mut snapshot = self.inner.snapshot(handle)?;
        if let Fault::EmptyState = self.fault {
            snapshot.state = StateBuffer::opaque();
        }
        Ok(snapshot)
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        _mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        let mut values: Vec<f32> = (0..query.rows)
            .map(|r| self.logits[r % self.logits.len()])
            .collect();
        match self.fault {
            Fault::None => {}
            Fault::Fails => {
                return Err(BackendError::DecodeFailed {
                    detail: "scripted failure".to_string(),
                });
            }
            Fault::NonFinite => values[1] = f32::NAN,
            Fault::WritesState => snapshot.state.as_mut_bytes().push(0xAB),
            Fault::EmptyState => {}
        }
        Ok(Logits {
            kind: query.kind,
            values,
        })
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.inner.pooled_features(snapshot)
    }
}

const A: [f32; 5] = [2.0, 0.5, -1.0, 0.0, -3.0];
const B: [f32; 5] = [0.1, 1.9, -0.4, 0.7, -2.2];
const C: [f32; 5] = [-1.0, 0.0, 3.5, 0.2, -0.5];

fn three() -> Vec<Arc<dyn DecisionBackend>> {
    vec![
        Arc::new(member("seed-0", &A)),
        Arc::new(member("seed-1", &B)),
        Arc::new(member("seed-2", &C)),
    ]
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let max = v
        .iter()
        .map(|x| f64::from(*x))
        .fold(f64::NEG_INFINITY, f64::max);
    let lse = max
        + v.iter()
            .map(|x| (f64::from(*x) - max).exp())
            .sum::<f64>()
            .ln();
    v.iter().map(|x| f64::from(*x) - lse).collect()
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn decode(backend: &EnsembleBackend, rows: usize) -> Result<Logits, BackendError> {
    let handle = backend.prefill("a prefix")?;
    let mut snapshot = backend.snapshot(&handle)?;
    let query = SlotQuery {
        slot_name: "class",
        suffix: " suffix",
        rows,
        kind: QueryKind::Letters,
    };
    backend.decode_slot(&mut snapshot, &query, DecodeMode::ReadOnly)
}

#[test]
fn the_ensemble_returns_the_mean_of_the_members_row_log_softmax() {
    let ensemble = EnsembleBackend::new(three()).expect("three agreeing members");
    let got = decode(&ensemble, 5).expect("decodes");
    assert_eq!(got.kind, QueryKind::Letters);
    let (a, b, c) = (log_softmax(&A), log_softmax(&B), log_softmax(&C));
    for (row, value) in got.values.iter().enumerate() {
        let want = (a[row] + b[row] + c[row]) / 3.0;
        assert!(
            (f64::from(*value) - want).abs() < 1e-6,
            "row {row}: {value} vs {want}"
        );
    }
}

#[test]
fn the_mean_log_softmax_and_the_mean_logit_give_one_distribution() {
    // Each member's log-normaliser is one constant across its rows, so the mean of the
    // log-softmaxes differs from the mean of the raw logits by a constant, and softmax over the
    // rows forgets it. A gate row that averages either computes the runtime's distribution.
    let ensemble = EnsembleBackend::new(three()).expect("three agreeing members");
    let got: Vec<f64> = decode(&ensemble, 5)
        .expect("decodes")
        .values
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    let mean_logits: Vec<f64> = (0..5)
        .map(|r| (f64::from(A[r]) + f64::from(B[r]) + f64::from(C[r])) / 3.0)
        .collect();
    for (p, q) in softmax(&got).iter().zip(softmax(&mean_logits)) {
        assert!((p - q).abs() < 1e-6, "{p} vs {q}");
    }
}

#[test]
fn the_identity_is_the_members_combined() {
    let ensemble = EnsembleBackend::new(three()).expect("builds");
    let id = ensemble.identity();
    let members = [identity("seed-0"), identity("seed-1"), identity("seed-2")];
    let hashes: Vec<&str> = members.iter().map(|m| m.weight_hash.as_str()).collect();
    assert_eq!(id.weight_hash, ensemble_weight_hash(&hashes));
    assert!(!hashes.contains(&id.weight_hash.as_str()));
    assert_eq!(id.tokenizer_hash, members[0].tokenizer_hash);
    assert_eq!(id.label_set_hash, members[0].label_set_hash);
    assert_eq!(id.calibration_hash, members[0].calibration_hash);
    assert!(id.name.starts_with("ensemble/3["), "{}", id.name);
    // Order is part of the identity: the mean is the same, the record of what was averaged is not.
    let reversed: Vec<&str> = hashes.iter().rev().copied().collect();
    assert_ne!(ensemble_weight_hash(&reversed), id.weight_hash);
}

#[track_caller]
fn refused(members: Vec<Arc<dyn DecisionBackend>>, needle: &str) {
    match EnsembleBackend::new(members) {
        Err(BackendError::Unavailable { detail }) => {
            assert!(detail.contains(needle), "{needle:?} not in {detail}")
        }
        Err(other) => panic!("expected unavailable, got {other:?}"),
        Ok(_) => panic!("an ensemble that cannot be averaged honestly was built"),
    }
}

#[test]
fn a_partial_or_oversized_ensemble_is_never_built() {
    refused(vec![Arc::new(member("seed-0", &A))], "2 to");
    let many: Vec<Arc<dyn DecisionBackend>> = (0..=MAX_MEMBERS)
        .map(|i| Arc::new(member(&format!("seed-{i}"), &A)) as Arc<dyn DecisionBackend>)
        .collect();
    refused(many, "2 to");
}

#[test]
fn members_that_disagree_or_repeat_are_refused() {
    for (field, edit) in [
        ("tokenizer_hash", 0usize),
        ("label_set_hash", 1),
        ("calibration_hash", 2),
    ] {
        let mut odd = member("seed-1", &B);
        match edit {
            0 => odd.identity.tokenizer_hash = "t".repeat(64),
            1 => odd.identity.label_set_hash = "l".repeat(64),
            _ => odd.identity.calibration_hash = CalibrationTable::new("other").hash(),
        }
        refused(
            vec![
                Arc::new(member("seed-0", &A)),
                Arc::new(odd),
                Arc::new(member("seed-2", &C)),
            ],
            field,
        );
    }
    refused(
        vec![
            Arc::new(member("seed-0", &A)),
            Arc::new(member("seed-1", &B)),
            Arc::new(member("seed-0", &C)),
        ],
        "one tower averaged twice",
    );
}

#[test]
fn one_failing_member_fails_the_decode_and_names_itself() {
    let mut failing = member("seed-1", &B);
    failing.fault = Fault::Fails;
    let ensemble = EnsembleBackend::new(vec![
        Arc::new(member("seed-0", &A)),
        Arc::new(failing),
        Arc::new(member("seed-2", &C)),
    ])
    .expect("builds");
    match decode(&ensemble, 5) {
        Err(BackendError::DecodeFailed { detail }) => {
            assert!(detail.contains("ensemble member 1"), "{detail}");
            assert!(detail.contains("scripted failure"), "{detail}");
        }
        other => panic!("the mean of two members was returned for three: {other:?}"),
    }
}

#[test]
fn a_member_with_a_non_finite_logit_fails_the_decode_and_names_itself() {
    let mut nan = member("seed-2", &C);
    nan.fault = Fault::NonFinite;
    let ensemble = EnsembleBackend::new(vec![
        Arc::new(member("seed-0", &A)),
        Arc::new(member("seed-1", &B)),
        Arc::new(nan),
    ])
    .expect("builds");
    match decode(&ensemble, 5) {
        Err(BackendError::DecodeFailed { detail }) => {
            assert!(detail.contains("ensemble member 2"), "{detail}");
            assert!(detail.contains("non-finite"), "{detail}");
        }
        other => panic!("a NaN was averaged away: {other:?}"),
    }
}

#[test]
fn a_corrupt_ensemble_state_is_refused() {
    let ensemble = EnsembleBackend::new(three()).expect("builds");
    let handle = ensemble.prefill("a prefix").expect("prefills");
    let mut snapshot = ensemble.snapshot(&handle).expect("snapshots");
    let query = SlotQuery {
        slot_name: "class",
        suffix: " suffix",
        rows: 5,
        kind: QueryKind::Letters,
    };
    let mut truncated = snapshot.clone();
    let n = truncated.state.len();
    truncated.state.as_mut_bytes().truncate(n - 1);
    assert!(matches!(
        ensemble.decode_slot(&mut truncated, &query, DecodeMode::ReadOnly),
        Err(BackendError::DecodeFailed { .. })
    ));
    snapshot.state.as_mut_bytes().push(0);
    match ensemble.decode_slot(&mut snapshot, &query, DecodeMode::ReadOnly) {
        Err(BackendError::DecodeFailed { detail }) => {
            assert!(detail.contains("follow the last member"), "{detail}")
        }
        other => panic!("a trailing byte was accepted: {other:?}"),
    }
}

fn request() -> DecisionRequest {
    let context = b"fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n";
    let value = json!({
        "schema_version": 1, "task": "devcouncil.verdict",
        "context_b64": qd_runtime::b64::encode(context), "context_len": context.len(),
        "question": "Does this diff implement what the commit message claims?",
        "slots": [{"name": "verdict", "type": "choice",
                   "options": ["stub", "logic", "cosmetic", "clean"]}],
        "route": "generic",
    });
    match parse_line(&serde_json::to_vec(&value).expect("serializes")) {
        Ok(Incoming::Request(request)) => *request,
        other => panic!("the test request was not accepted: {other:?}"),
    }
}

fn runtime(members: Vec<Arc<dyn DecisionBackend>>) -> Runtime {
    Runtime::with_backend(
        Arc::new(EnsembleBackend::new(members).expect("builds")),
        CalibrationTable::reference(),
        HeadRegistry::new(),
        RenderCaps::DEFAULT,
    )
    .expect("assembles")
}

#[test]
fn the_runtime_answers_through_an_ensemble_and_its_readonly_check_covers_every_member() {
    match runtime(three()).answer(&request(), None) {
        Response::Ok(_) => {}
        other => panic!("an ensemble of agreeing members must answer: {other:?}"),
    }
    let mut writer = member("seed-1", &B);
    writer.fault = Fault::WritesState;
    let members: Vec<Arc<dyn DecisionBackend>> = vec![
        Arc::new(member("seed-0", &A)),
        Arc::new(writer),
        Arc::new(member("seed-2", &C)),
    ];
    match runtime(members).answer(&request(), None) {
        Response::Error(envelope) => {
            assert_eq!(
                envelope.error.kind(),
                "readonly_violated",
                "{:?}",
                envelope.error
            )
        }
        other => {
            panic!("a member that wrote its state under a read-only decode was missed: {other:?}")
        }
    }
}

/// Fail-first (Fable ruling 2, section 3): a member that claims host-visible state but hands none
/// was framed into a non-empty ensemble state, so the slot-isolation check recorded `Ran` for an
/// answer it compared nothing of. The ensemble now refuses that member before any decode.
#[test]
fn a_member_claiming_visible_state_but_handing_none_is_refused() {
    let members = || -> Vec<Arc<dyn DecisionBackend>> {
        let mut empty = member("seed-1", &B);
        empty.fault = Fault::EmptyState;
        assert!(
            empty.identity.state_host_visible,
            "the case: the member claims visible state"
        );
        vec![
            Arc::new(member("seed-0", &A)),
            Arc::new(empty),
            Arc::new(member("seed-2", &C)),
        ]
    };
    let backend = EnsembleBackend::new(members()).expect("builds");
    match backend.prefill("a prefix") {
        Err(BackendError::PrefillFailed { detail }) => {
            assert!(
                detail.contains("member 1") && detail.contains("handed none"),
                "{detail}"
            );
        }
        other => panic!("an empty member state was framed and accepted: {other:?}"),
    }
    match runtime(members()).answer(&request(), None) {
        Response::Ok(envelope) => panic!(
            "answered (degraded = {}) over a member whose state nothing compared",
            envelope.degraded
        ),
        Response::Error(_) => {}
        other => panic!("expected the backend error, got {other:?}"),
    }
}
