//! `qd-runtime` — the universal typed-decision interface.
//!
//! `docs/schema-api.md` is the contract; this crate is its executable form. The shape of the crate
//! follows the contract's own sections:
//!
//! | Module | Contract section |
//! | --- | --- |
//! | [`schema`] | Request, slot types, Answer |
//! | [`context`] | "`context` crosses the FFI boundary as **bytes and a length**" |
//! | [`b64`] | the codec behind `context_b64` — RFC 4648, canonical only |
//! | [`refusal`] | "Refusals — fail closed, never truncate" |
//! | [`wire`] | the JSON envelope, and the only way to build a validated request |
//! | [`render`] | `docs/hardening.md` §3 — the context is adversarial by construction |
//! | [`backend`] | the four operations a model backend must provide |
//! | [`reference`] | a deterministic reference backend that is **not** a model and says so |
//! | [`calibration`] | split-conformal sets on calibrated scores — margin, never entropy |
//! | [`registry`] | the registered route's head files, hash-bound to the backbone |
//! | [`fixtures`] | the golden wire corpus in `fixtures/wire/` — the executable answer-side seam |
//! | [`answer`] | "Answering procedure": prefill once, snapshot, readonly slot queries |
//! | [`runtime`] | hash binding, poison/rebuild, degraded |
//! | [`service`] | **the one contract both modes share** |
//! | [`serve`] / [`oneshot`] | `qd serve` and `qd oneshot` |
//!
//! # The two properties this crate exists to keep
//!
//! **A refusal is not a `noul`.** A refusal means the request was not answerable as posed; `noul`
//! means the model was asked and abstained. They are different types, they reach the caller in
//! different [`schema::Response`] variants, and their serializations share no field name. See
//! `tests/refusal_is_not_noul.rs`.
//!
//! **A slot's answer does not depend on the previous slot's.** Every slot decodes from the same
//! snapshot with [`backend::DecodeMode::ReadOnly`], and the runtime hashes the state buffer either
//! side of every decode. A backend that writes back is caught by
//! [`refusal::BackendError::ReadonlyViolated`] rather than quietly producing correlated answers.
//! Where a backend cannot expose its state to the host, the check is reported **not run** and the
//! answer is `degraded` — `CLAUDE.md`: *a check that could not run must never report the same
//! result as a check that ran and passed*.
//!
//! # What is deliberately absent
//!
//! There is **no Metal backend and no stubbed stand-in for one**. Kernel work (K1-K7) starts after
//! the shipping gate. [`backend::DecisionBackend`] is the seam it will implement; until then the
//! only backend that exists is [`reference::ReferenceBackend`], which refuses to serve unless it is
//! explicitly enabled and marks every answer it does produce `degraded`. An absent backend is
//! [`refusal::BackendError::Unavailable`], a typed error — never a plausible letter.

pub mod answer;
pub mod b64;
pub mod backend;
pub mod calibration;
pub mod context;
pub mod fixtures;
pub mod oneshot;
pub mod reference;
pub mod refusal;
pub mod registry;
pub mod render;
pub mod runtime;
pub mod schema;
pub mod serve;
pub mod service;
pub mod wire;

pub use context::Context;
pub use refusal::{BackendError, QdError, Refusal};
pub use schema::{AnswerEnvelope, CallerReading, DecisionRequest, Response, SlotAnswer, SlotSpec};
pub use service::{Service, ServiceConfig};

/// Version stamped into `status` responses, so a caller can tell which build answered.
pub const RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Lowercase hex of a byte string. Used for every hash this crate prints or compares.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0x0f) as usize] as char);
    }
    out
}

/// SHA-256, as a convenience over the one hash this crate uses everywhere.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// A counter-mode PRNG over SHA-256.
///
/// Used for exactly one thing: the option permutation of the contract's **second pass**. It is
/// seeded from [`schema::DecisionRequest::digest`], so the second pass is reproducible for a given
/// request rather than run-to-run random — a caller that sees a `noul` from permutation
/// disagreement can replay the exact pair of passes that disagreed.
///
/// This is deliberately **not** the same stream as `python/qd_data/render.py`'s
/// `DeterministicRng` (keyed blake2b). That one shuffles options per *training example*; this one
/// permutes them per *serving request*. They answer different questions and are never compared, so
/// matching their streams would be a coincidence maintained by hand. What must match between the
/// two lanes is the rendered prompt, and that is asserted in `tests/render_contract.rs`.
pub struct CounterRng {
    key: [u8; 32],
    counter: u64,
}

impl CounterRng {
    pub fn new(domain: &str, seed: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"qd-runtime.CounterRng.v1\x00");
        h.update(domain.as_bytes());
        h.update([0u8]);
        h.update(seed);
        Self {
            key: h.finalize().into(),
            counter: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.key);
        h.update(self.counter.to_be_bytes());
        self.counter = self.counter.wrapping_add(1);
        let block: [u8; 32] = h.finalize().into();
        let mut eight = [0u8; 8];
        eight.copy_from_slice(&block[..8]);
        u64::from_be_bytes(eight)
    }

    /// Uniform in `[0, n)` by rejection sampling — no modulo bias.
    ///
    /// Returns 0 for `n <= 1`, which is the only sensible value and keeps the function total; a
    /// permutation of 0 or 1 elements never calls it with a meaningful range.
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let n64 = n as u64;
        let limit = u64::MAX - (u64::MAX % n64);
        loop {
            let v = self.next_u64();
            if v < limit {
                return (v % n64) as usize;
            }
        }
    }

    /// Fisher-Yates over `0..n`. Uniform over all `n!` permutations, fixed points included.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        let mut i = n;
        while i > 1 {
            i -= 1;
            let j = self.below(i + 1);
            idx.swap(i, j);
        }
        idx
    }

    /// Sattolo's algorithm: uniform over the `(n-1)!` **cyclic** permutations of `0..n`.
    ///
    /// The difference from [`Self::permutation`] is one character — `below(i)` instead of
    /// `below(i + 1)` — and it is the difference between a permutation that may have fixed points
    /// and one that provably has none. A single cycle of length `n >= 2` moves every element, so
    /// this is a derangement.
    ///
    /// That is what the contract's second pass needs. See
    /// [`crate::render::second_pass_permutation`].
    pub fn cyclic_permutation(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        if n < 2 {
            return idx;
        }
        let mut i = n;
        while i > 1 {
            i -= 1;
            let j = self.below(i);
            idx.swap(i, j);
        }
        idx
    }
}
