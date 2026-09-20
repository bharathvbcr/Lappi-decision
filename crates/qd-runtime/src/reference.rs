//! A deterministic reference backend. **It is not a model.**
//!
//! It exists so the plumbing — rendering, snapshotting, slot isolation, calibration, the permuted
//! second pass, the socket, the lifecycle — can be tested before a single Metal kernel exists. Its
//! scores are a hash of the prompt. They carry no information about the question, and any letter it
//! returns is arbitrary.
//!
//! Three properties keep that from ever being mistaken for an answer:
//!
//! 1. **It refuses to serve unless explicitly enabled.** Constructed disabled, every operation
//!    returns [`BackendError::ReferenceBackendNotEnabled`]. There is no default that turns it on.
//! 2. **It names itself in every answer.** [`REFERENCE_BACKEND_NAME`] appears in
//!    [`crate::schema::AnswerEnvelope::backend`].
//! 3. **`is_model` is false**, so [`crate::runtime::Runtime`] marks every slot it produces
//!    `degraded`. A caller that reads the flag it asked for cannot read a reference answer as a
//!    model's.
//!
//! What it is *faithful* about is structure, and that is the point: it prefills once, exposes a
//! host-visible state buffer, leaves that buffer byte-identical under
//! [`DecodeMode::ReadOnly`], and genuinely mutates it under [`DecodeMode::WriteBack`]. That last
//! one is what makes `tests/readonly_slot_isolation.rs` a test rather than a tautology.

use crate::backend::{
    BackendIdentity, DecisionBackend, DecodeMode, Logits, PrefillHandle, SlotQuery, StateBuffer,
    StateSnapshot,
};
use crate::refusal::BackendError;

/// The name that appears in every answer this backend produces.
pub const REFERENCE_BACKEND_NAME: &str = "reference-deterministic-v1";

/// Bytes of state the reference backend carries. Nothing depends on the size; it is large enough
/// that a write-back is visibly a partial update rather than a whole-buffer replacement.
const STATE_BYTES: usize = 512;

/// Deterministic, hash-driven, and honest about it.
pub struct ReferenceBackend {
    enabled: bool,
    identity: BackendIdentity,
}

impl ReferenceBackend {
    /// Build one. `enabled` is not defaulted anywhere: the caller states it.
    pub fn new(enabled: bool) -> Self {
        let identity = BackendIdentity {
            name: REFERENCE_BACKEND_NAME.to_string(),
            // These are hashes *of the reference backend's own definition*, not of any model. They
            // are stable across processes so a caller can pin them in `expect` while testing the
            // hash-binding path itself.
            tokenizer_hash: crate::hex(&crate::sha256(b"qd-reference.tokenizer.v1")),
            weight_hash: crate::hex(&crate::sha256(b"qd-reference.weights.v1")),
            head_hash: crate::hex(&crate::sha256(b"qd-reference.head.v1")),
            label_set_hash: crate::hex(&crate::sha256(b"qd-reference.label-set.v1")),
            calibration_hash: crate::calibration::CalibrationTable::reference().hash(),
            is_model: false,
            state_host_visible: true,
        };
        Self { enabled, identity }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn guard(&self) -> Result<(), BackendError> {
        if self.enabled {
            Ok(())
        } else {
            Err(BackendError::ReferenceBackendNotEnabled {
                name: REFERENCE_BACKEND_NAME.to_string(),
            })
        }
    }
}

/// Expand a seed into `n` deterministic bytes by counter-mode SHA-256.
fn expand(seed: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut counter: u64 = 0;
    while out.len() < n {
        let mut block_input = Vec::with_capacity(seed.len() + 8);
        block_input.extend_from_slice(seed);
        block_input.extend_from_slice(&counter.to_be_bytes());
        out.extend_from_slice(&crate::sha256(&block_input));
        counter += 1;
    }
    out.truncate(n);
    out
}

/// Map four bytes to a finite score in `[-8, 8)`.
fn score_from(bytes: &[u8]) -> f32 {
    let mut four = [0u8; 4];
    four.copy_from_slice(&bytes[..4]);
    let raw = u32::from_be_bytes(four) as f64 / (u32::MAX as f64 + 1.0);
    ((raw * 16.0) - 8.0) as f32
}

impl DecisionBackend for ReferenceBackend {
    fn identity(&self) -> &BackendIdentity {
        &self.identity
    }

    fn prefill(&self, prefix: &str) -> Result<PrefillHandle, BackendError> {
        self.guard()?;
        let prompt_digest = crate::sha256(prefix.as_bytes());
        let mut seed = Vec::with_capacity(32 + 16);
        seed.extend_from_slice(b"qd-reference.prefill.v1");
        seed.extend_from_slice(&prompt_digest);
        Ok(PrefillHandle {
            backend: REFERENCE_BACKEND_NAME.to_string(),
            prompt_digest,
            // Not a tokenizer. A declared, reproducible count over bytes, so a caller can see the
            // field move with the prompt without it pretending to be a token count.
            token_count: prefix.len().div_ceil(4),
            state: StateBuffer::from_bytes(expand(&seed, STATE_BYTES)),
        })
    }

    fn snapshot(&self, handle: &PrefillHandle) -> Result<StateSnapshot, BackendError> {
        self.guard()?;
        Ok(StateSnapshot {
            backend: handle.backend.clone(),
            prompt_digest: handle.prompt_digest,
            token_count: handle.token_count,
            state: handle.state.clone(),
        })
    }

    fn decode_slot(
        &self,
        snapshot: &mut StateSnapshot,
        query: &SlotQuery<'_>,
        mode: DecodeMode,
    ) -> Result<Logits, BackendError> {
        self.guard()?;
        if query.rows == 0 {
            return Err(BackendError::DecodeFailed {
                detail: format!("slot `{}` asked for zero rows", query.slot_name),
            });
        }

        if mode == DecodeMode::WriteBack {
            // A real write-back folds the query into the recurrent state. This one does the same
            // shape of thing: it XORs a query-derived stream over the buffer in place, so the state
            // hash moves and the read-only test has something to fail against.
            let overlay = expand(
                &[
                    b"qd-reference.writeback.v1".as_slice(),
                    query.slot_name.as_bytes(),
                    query.suffix.as_bytes(),
                ]
                .concat(),
                snapshot.state.len(),
            );
            for (b, o) in snapshot.state.as_mut_bytes().iter_mut().zip(overlay) {
                *b ^= o;
            }
        }

        // Read-only: derived from the snapshot's bytes and the query, writing nothing.
        let seed = [
            b"qd-reference.decode.v1".as_slice(),
            snapshot.state.as_bytes(),
            query.kind.as_str().as_bytes(),
            query.slot_name.as_bytes(),
            query.suffix.as_bytes(),
        ]
        .concat();
        let stream = expand(&seed, query.rows * 4);
        let values = (0..query.rows)
            .map(|row| score_from(&stream[row * 4..row * 4 + 4]))
            .collect();
        Ok(Logits {
            kind: query.kind,
            values,
        })
    }

    fn pooled_features(&self, snapshot: &StateSnapshot) -> Result<Vec<f32>, BackendError> {
        self.guard()?;
        let seed = [
            b"qd-reference.pooled.v1".as_slice(),
            snapshot.state.as_bytes(),
        ]
        .concat();
        let stream = expand(&seed, REFERENCE_FEATURE_DIM * 4);
        Ok((0..REFERENCE_FEATURE_DIM)
            .map(|i| score_from(&stream[i * 4..i * 4 + 4]))
            .collect())
    }
}

/// Pooled-feature width of the reference backend. A registered head fitted for it declares this
/// `feature_dim`; any other width is a shape mismatch, not a quietly-truncated GEMV.
pub const REFERENCE_FEATURE_DIM: usize = 64;
