//! The model backend seam.
//!
//! **There is no Metal backend in this crate and none may be written here.** The plan puts K1-K7
//! after the shipping gate; this module is the interface those kernels will implement, defined now
//! so the rest of the runtime is written against the real operations rather than against a guess.
//!
//! The four operations are exactly `docs/schema-api.md`'s answering procedure:
//!
//! 1. [`DecisionBackend::prefill`] — *"prefill the context once"*;
//! 2. [`DecisionBackend::snapshot`] — *"snapshot the recurrent GDN state and the attention KV"*;
//! 3. [`DecisionBackend::decode_slot`] — *"answer each slot as a 1-token query from the snapshot;
//!    never write back"*, with [`DecodeMode`] as the `readonly` flag on the decode kernel (K2);
//! 4. [`DecisionBackend::pooled_features`] — the registered route's *"single GEMV on pooled
//!    features"*.
//!
//! # Why `decode_slot` takes `&mut StateSnapshot`
//!
//! A read-only operation that cannot write is not a test of anything. `decode_slot` is handed a
//! mutable snapshot precisely so that a backend *is able* to violate slot isolation, and the
//! runtime then catches it: [`crate::answer`] hashes the state buffer either side of every decode
//! and raises [`crate::refusal::BackendError::ReadonlyViolated`] if it moved. If the parameter were
//! `&StateSnapshot`, the guarantee would be enforced by the borrow checker for in-process backends
//! and by nothing at all for a backend holding GPU buffers behind a handle — which is every
//! backend this seam exists for.
//!
//! # When the check cannot run
//!
//! A backend whose state lives in device memory may not be able to expose it to the host cheaply.
//! [`BackendIdentity::state_host_visible`] says so, and the runtime then reports the check
//! **not run** and marks the answer `degraded`. `CLAUDE.md`: *a check that could not run must never
//! report the same result as a check that ran and passed.*

use serde::{Deserialize, Serialize};

use crate::refusal::BackendError;

/// What a backend is, and what can be checked about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendIdentity {
    /// The backend's own name, reported in every answer. A reference backend says so here.
    pub name: String,
    pub tokenizer_hash: String,
    pub weight_hash: String,
    pub head_hash: String,
    pub label_set_hash: String,
    pub calibration_hash: String,
    /// False for anything that is not a trained model.
    ///
    /// The runtime marks every answer from a non-model backend `degraded`, so a caller that reads
    /// the flag it asked for cannot mistake a reference answer for a model's.
    pub is_model: bool,
    /// Whether the state buffer can be hashed by the host.
    ///
    /// False is not a failure — it is the honest answer for a backend holding device memory. It
    /// changes what the runtime may claim: with `false`, the slot-isolation check is reported
    /// [`SlotIsolationCheck::NotRun`] and the answer is degraded.
    pub state_host_visible: bool,
}

/// The opaque state a backend carries between prefill and decode.
///
/// The bytes are the backend's own layout — the recurrent GDN state and the attention KV, however
/// it chooses to mirror them for the host. The runtime never interprets them; it only hashes them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StateBuffer {
    bytes: Vec<u8>,
}

impl StateBuffer {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// An empty buffer, for a backend whose state is not host-visible.
    pub fn opaque() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_bytes(&mut self) -> &mut Vec<u8> {
        &mut self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn digest(&self) -> [u8; 32] {
        crate::sha256(&self.bytes)
    }

    pub fn digest_hex(&self) -> String {
        crate::hex(&self.digest())
    }
}

/// The result of prefilling the shared prefix once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillHandle {
    pub backend: String,
    pub prompt_digest: [u8; 32],
    /// How many tokens the backend consumed. Reported, never used as a cap by the runtime — the
    /// caps are on bytes, which is what the caller controls.
    pub token_count: usize,
    pub state: StateBuffer,
}

/// The snapshot every slot decodes from. `docs/schema-api.md`: *"the snapshot is the prefix
/// cache."*
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSnapshot {
    pub backend: String,
    pub prompt_digest: [u8; 32],
    pub token_count: usize,
    pub state: StateBuffer,
}

impl StateSnapshot {
    pub fn state_digest(&self) -> [u8; 32] {
        self.state.digest()
    }

    pub fn state_digest_hex(&self) -> String {
        self.state.digest_hex()
    }
}

/// The `readonly` flag on the decode kernel (K2).
///
/// [`DecodeMode::ReadOnly`] is the slot-isolation mechanism — *"one flag instead of a mask
/// kernel"*. [`DecodeMode::WriteBack`] exists so the guarantee is falsifiable: a backend whose
/// write-back path does not move the state hash has not implemented write-back, and the read-only
/// test against it would pass for the wrong reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeMode {
    ReadOnly,
    WriteBack,
}

/// Which head a decode reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryKind {
    /// The letter-token slice of `lm_head`: `choice` and `score`.
    Letters,
    /// The pointer head over line-start tokens, start of a `span`.
    PointerStart,
    /// The pointer head over line-start tokens, end of a `span`.
    PointerEnd,
}

impl QueryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            QueryKind::Letters => "letters",
            QueryKind::PointerStart => "pointer_start",
            QueryKind::PointerEnd => "pointer_end",
        }
    }
}

/// One slot's 1-token query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotQuery<'a> {
    pub slot_name: &'a str,
    /// The rendered suffix for this slot. The prefix is already in the snapshot.
    pub suffix: &'a str,
    /// How many rows the backend must return. The runtime refuses any other count.
    pub rows: usize,
    pub kind: QueryKind,
}

/// Raw scores for one query, one per row, in row order.
#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    pub kind: QueryKind,
    pub values: Vec<f32>,
}

/// Whether the slot-isolation guarantee was checked, and what it found.
///
/// Never collapses to a boolean. `CLAUDE.md`: *a check that could not run must never report the
/// same result as a check that ran and passed.*
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SlotIsolationCheck {
    /// The state buffer was hashed either side of the decode and was unchanged.
    Ran { state_hash: String },
    /// The state is not host-visible, so nothing could be compared.
    NotRun { reason: String },
}

impl SlotIsolationCheck {
    pub fn ran(&self) -> bool {
        matches!(self, SlotIsolationCheck::Ran { .. })
    }
}

/// The operations the real backend will implement.
///
/// `Send + Sync` because one warm runtime serves concurrent requests from the socket: the decode
/// path must not hold the service lock, or one slow request wedges the agent.
pub trait DecisionBackend: Send + Sync {
    fn identity(&self) -> &BackendIdentity;

    /// Prefill the shared prefix. Called **once** per request, whatever the slot count and whatever
    /// the route.
    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError>;

    /// Snapshot the recurrent state and the attention KV.
    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError>;

    /// Answer one slot as a 1-token query from the snapshot.
    ///
    /// Under [`DecodeMode::ReadOnly`] the implementation must leave `snapshot` byte-identical. The
    /// runtime verifies it whenever the state is host-visible.
    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> Result<Logits, BackendError>;

    /// Answer several queries from one snapshot.
    ///
    /// The default runs [`decode_slot`](DecisionBackend::decode_slot) once per query, so a backend
    /// that has not overridden this keeps today's two-pass behaviour. A backend that can run the
    /// queries as one forward (same length, read-only) overrides it. The runtime hashes the
    /// snapshot once around the whole call, and that one comparison covers every query in it.
    fn decode_slots(
        &self,
        snapshot: &mut StateSnapshot,
        queries: &[SlotQuery<'_>],
        mode: DecodeMode,
    ) -> Result<Vec<Logits>, BackendError> {
        if queries.is_empty() {
            return Err(BackendError::DecodeFailed {
                detail: "decode_slots was given no queries".into(),
            });
        }
        let mut out = Vec::with_capacity(queries.len());
        for query in queries {
            out.push(self.decode_slot(snapshot, query, mode)?);
        }
        Ok(out)
    }

    /// Pooled features for the registered route's single GEMV. Same prefill, same snapshot.
    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError>;
}

/// Check a backend's logits before anything reads them as an answer.
///
/// A wrong row count, a NaN or an infinity is a backend failure, not a low-confidence answer: every
/// one of them would otherwise reach the caller as a letter.
pub fn validate_logits(
    logits: &Logits,
    query: &SlotQuery<'_>,
) -> Result<(), BackendError> {
    if logits.kind != query.kind {
        return Err(BackendError::LogitKindMismatch {
            slot: query.slot_name.to_string(),
            expected: query.kind.as_str().to_string(),
        });
    }
    if logits.values.len() != query.rows {
        return Err(BackendError::LogitShapeMismatch {
            slot: query.slot_name.to_string(),
            expected: query.rows,
            actual: logits.values.len(),
        });
    }
    for (row, v) in logits.values.iter().enumerate() {
        if !v.is_finite() {
            return Err(BackendError::NonFiniteLogit {
                slot: query.slot_name.to_string(),
                row,
            });
        }
    }
    Ok(())
}
