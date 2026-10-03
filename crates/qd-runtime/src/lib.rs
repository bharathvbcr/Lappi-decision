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
//! | [`admission`] | a `code.defect_class` context not in the trained diff shape or pool language |
//! | [`wire`] | the JSON envelope, and the only way to build a validated request |
//! | [`render`] | `docs/hardening.md` §3 — the context is adversarial by construction |
//! | [`backend`] | the four operations a model backend must provide |
//! | [`reference`] | a deterministic reference backend that is **not** a model and says so |
//! | [`ensemble`] | N towers behind one backend: per-row letter log-probabilities averaged |
//! | [`calibration`] | split-conformal sets on calibrated scores — margin, never entropy |
//! | [`calibration_fit`] | fitting that table from verdicts: temperature, `noul_margin`, `1 - q̂` |
//! | [`registry`] | the registered route's head files, hash-bound to the backbone |
//! | [`release`] | the release directory `qd-export` writes, opened and hash-bound at load |
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

pub mod admission;
pub mod answer;
pub mod b64;
pub mod backend;
pub mod calibration;
pub mod calibration_fit;
pub mod context;
pub mod ensemble;
pub mod fixtures;
pub mod oneshot;
pub mod reference;
pub mod refusal;
pub mod registry;
pub mod release;
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

/// Characters CPython's `str.isspace()` accepts that Unicode's `White_Space` property does not:
/// bidirectional class B (paragraph separator) for U+001C..U+001E, and S (segment separator) for
/// U+001F.
const ISSPACE_BEYOND_WHITE_SPACE: &[char] = &['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'];

/// Whether one character is whitespace **for this wire contract**.
///
/// # Why this is not `char::is_whitespace`
///
/// Every caller-supplied string on this wire is asked "is it blank" somewhere: the task
/// ([`wire::validate`]), an option and its reserved-label comparison ([`wire::parse_slot`]), a slot
/// name ([`schema::slot_name_fault`]) and the context ([`context::Context::non_whitespace_len`]).
/// The Python lane asks all four with `str.strip()`, and `str.strip()` strips exactly what
/// `str.isspace()` accepts. Rust's `char::is_whitespace()` is the Unicode `White_Space` property:
/// 25 codepoints against CPython's 29. The four in [`ISSPACE_BEYOND_WHITE_SPACE`] are the whole
/// difference, and until 2026-09-19 all four Rust sites asked the narrower question.
///
/// What that cost, measured rather than imagined:
///
/// * a context of nothing but U+001C rendered here and was `context_empty` there — a confident
///   letter produced from a context with no content, which is the one thing that rule prevents;
/// * an option of `"\u{1c}"` rendered here and was `empty_option` there;
/// * an option of `"\u{1c}noul"` rendered here as an ordinary option and was
///   `reserved_option_name` there — a seventeenth option whose text *is* the abstain label, which
///   `docs/hardening.md` §3 forbids by name;
/// * a slot name of `"\u{1c}"` rendered here and was refused there.
///
/// Four sites, one rule, so it is one function. `tests/wire_context_crosslang.rs` reads the real
/// `str.isspace()` and the real `str.strip()` at test time and compares — the whole codepoint set,
/// and the blank/strips-to-`noul` answer for a table of adversarial strings — so the two lanes
/// cannot drift apart again by a comment going stale, which is how this one was found.
pub fn is_wire_whitespace(c: char) -> bool {
    c.is_whitespace() || ISSPACE_BEYOND_WHITE_SPACE.contains(&c)
}

/// [`str::trim`] under [`is_wire_whitespace`] — the exact counterpart of Python's `str.strip()`.
///
/// Use this instead of `str::trim` anywhere the result is compared against the Python lane's, which
/// on this wire is everywhere.
pub fn wire_trim(s: &str) -> &str {
    s.trim_matches(is_wire_whitespace)
}

/// Whether a caller-supplied string carries nothing. The counterpart of `not s.strip()`.
pub fn is_blank(s: &str) -> bool {
    wire_trim(s).is_empty()
}

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

/// The most threads [`sha256_slices_parallel`] ever runs on, whatever the machine offers. Eight
/// is the bound in the lead's lane grant for this change (2026-10-03, `min(available_parallelism,
/// 8)`), set after the two kernel panics of 2026-10-02: a hash fan-out is bounded like every other
/// one. More threads would buy little, because each buffer is one sequential SHA-256 and the
/// largest buffer on the slowest core bounds a round: at 8K the state is 12 K/V buffers of
/// ~16.8 MB, two rounds on 8 threads; at task length it is 18 buffers of ~1 MB either way.
pub const SHA256_PARALLEL_MAX_WORKERS: std::num::NonZeroUsize =
    std::num::NonZeroUsize::new(8).unwrap();

/// How many threads (the calling one included) [`sha256_slices_parallel`] uses for `slices`
/// slices on a machine offering `available`: the caller's request, never above
/// [`SHA256_PARALLEL_MAX_WORKERS`], the machine or the work. 1 (or 0 for no slices) is serial.
pub fn sha256_worker_count(
    max_workers: std::num::NonZeroUsize,
    available: usize,
    slices: usize,
) -> usize {
    max_workers
        .get()
        .min(SHA256_PARALLEL_MAX_WORKERS.get())
        .min(available)
        .min(slices)
}

/// [`sha256`] of every slice, in input order, computed on up to `max_workers` threads.
///
/// [`SHA256_PARALLEL_MAX_WORKERS`] is a ceiling no caller can raise: `max_workers` is a request
/// that can only lower it ([`sha256_worker_count`]).
///
/// `out[i] == sha256(slices[i])` bit for bit: each slice is one ordinary SHA-256 and only which
/// thread computes it changes, so a record built from these hashes is the record a serial loop
/// builds. The calling thread hashes too, beside `sha256_worker_count(..) - 1` helpers in a
/// `std::thread::scope`; all of them take
/// slices largest-first from one shared cursor, so a long slice starts first rather than last.
/// A helper the OS refuses to start costs speed, never coverage: the cursor still hands every
/// slice to some thread. A panic in a helper is re-raised on the calling thread with its
/// original payload. No slices is no hashes.
pub fn sha256_slices_parallel(
    slices: &[&[u8]],
    max_workers: std::num::NonZeroUsize,
) -> Vec<[u8; 32]> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let workers = sha256_worker_count(max_workers, available, slices.len());
    if workers <= 1 {
        return slices.iter().map(|s| sha256(s)).collect();
    }
    let mut order: Vec<usize> = (0..slices.len()).collect();
    // Stable: equal lengths keep input order, so the schedule depends on the lengths alone.
    order.sort_by_key(|&i| std::cmp::Reverse(slices[i].len()));
    let cursor = AtomicUsize::new(0);
    // Captures only shared references, so it is `Copy`: each helper gets its own copy.
    let work = || {
        let mut done = Vec::new();
        loop {
            let k = cursor.fetch_add(1, Ordering::Relaxed);
            let Some(&i) = order.get(k) else { break };
            done.push((i, sha256(slices[i])));
        }
        done
    };
    let parts: Vec<Vec<(usize, [u8; 32])>> = std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..workers)
            .filter_map(|n| {
                std::thread::Builder::new()
                    .name(format!("qd-sha256-{n}"))
                    .spawn_scoped(scope, work)
                    .ok()
            })
            .collect();
        let mut parts = vec![work()];
        for h in helpers {
            parts.push(
                h.join()
                    .unwrap_or_else(|payload| std::panic::resume_unwind(payload)),
            );
        }
        parts
    });
    let mut out: Vec<Option<[u8; 32]>> = vec![None; slices.len()];
    for (i, h) in parts.into_iter().flatten() {
        assert!(out[i].replace(h).is_none(), "slice {i} was hashed twice");
    }
    out.into_iter()
        .enumerate()
        .map(|(i, h)| h.unwrap_or_else(|| panic!("slice {i} was never hashed")))
        .collect()
}

#[cfg(test)]
mod sha256_parallel_tests {
    //! Characterization of [`sha256_slices_parallel`] against the serial [`sha256`]. Every test
    //! here passes on both sides of the parallel digest by construction; the fail-first artifact
    //! for that change is its A/B ledger row (`digest_ms`), not these.

    use std::num::NonZeroUsize;

    use super::{
        CounterRng, SHA256_PARALLEL_MAX_WORKERS, sha256, sha256_slices_parallel,
        sha256_worker_count,
    };

    /// The ceiling holds against any request, machine or batch: no argument raises it above 8.
    #[test]
    fn no_request_raises_the_thread_ceiling() {
        let ceiling = SHA256_PARALLEL_MAX_WORKERS.get();
        assert_eq!(ceiling, 8);
        for request in [1, 2, 7, 8, 9, 18, 1000, usize::MAX] {
            for available in [1, 6, 8, 18, 64, usize::MAX] {
                for slices in [0, 1, 7, 48, 10_000] {
                    let n = sha256_worker_count(nz(request), available, slices);
                    assert!(n <= ceiling, "{request}/{available}/{slices} -> {n}");
                    assert!(n <= request && n <= available && n <= slices, "{n}");
                }
            }
        }
        assert_eq!(sha256_worker_count(nz(usize::MAX), 18, 48), 8);
        assert_eq!(sha256_worker_count(nz(3), 18, 48), 3);
    }

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    fn serial(slices: &[&[u8]]) -> Vec<[u8; 32]> {
        slices.iter().map(|s| sha256(s)).collect()
    }

    /// Deterministic bytes: the slice contents differ, so a hash landing on the wrong index shows.
    /// One `CounterRng` draw seeds each slice and a splitmix64 stream fills it, eight bytes a step:
    /// a SHA-256 per byte would make the fixtures, not the code under test, the slow part.
    fn buffers(lens: &[usize], seed: &[u8]) -> Vec<Vec<u8>> {
        let mut rng = CounterRng::new("sha256_parallel_tests", seed);
        lens.iter()
            .map(|&n| {
                let mut state = rng.next_u64();
                let mut out = Vec::with_capacity(n + 8);
                while out.len() < n {
                    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    out.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
                }
                out.truncate(n);
                out
            })
            .collect()
    }

    fn check(lens: &[usize], workers: usize) {
        let owned = buffers(lens, format!("{lens:?}/{workers}").as_bytes());
        let slices: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        assert_eq!(
            sha256_slices_parallel(&slices, nz(workers)),
            serial(&slices),
            "lens {lens:?}, {workers} workers"
        );
    }

    #[test]
    fn a_known_answer_survives_the_fan_out() {
        // FIPS 180-2 "abc", at every position of a batch large enough to fan out.
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let slices: Vec<&[u8]> = vec![b"abc"; 9];
        for h in sha256_slices_parallel(&slices, SHA256_PARALLEL_MAX_WORKERS) {
            assert_eq!(super::hex(&h), abc);
        }
    }

    #[test]
    fn no_slices_is_no_hashes() {
        assert!(sha256_slices_parallel(&[], SHA256_PARALLEL_MAX_WORKERS).is_empty());
        assert!(sha256_slices_parallel(&[], nz(1)).is_empty());
    }

    #[test]
    fn edge_lengths_match_the_serial_hash() {
        check(&[0], 8);
        check(&[1], 8);
        check(&[0, 0, 0, 0], 8);
        check(&[1, 0, 64, 55, 56, 63, 65, 1], 8); // SHA-256 padding boundaries
        check(&[0, 1 << 20, 0, 3], 8);
    }

    #[test]
    fn every_worker_count_matches_the_serial_hash() {
        // The prefix state's shape: 18 conv, 18 GDN, 6 K, 6 V buffers of unequal size.
        let mut lens = vec![24_576; 18];
        lens.extend(vec![262_144; 18]);
        lens.extend(vec![
            200_000, 1, 0, 199_999, 100, 7, 200_000, 3, 0, 9, 4096, 5,
        ]);
        for workers in [1, 2, 3, 7, 8, 9, 48, 49, 1000, usize::MAX] {
            check(&lens, workers);
        }
    }

    #[test]
    fn more_workers_than_slices_and_duplicate_contents() {
        let same = vec![7u8; 1000];
        let slices: Vec<&[u8]> = vec![&same, &same, &same];
        assert_eq!(sha256_slices_parallel(&slices, nz(64)), serial(&slices));
        check(&[5, 5], 64);
    }

    #[test]
    fn randomized_shapes_match_the_serial_hash() {
        let mut rng = CounterRng::new("sha256_parallel_tests.shapes", b"v1");
        // ~0.5 MB a round on average: the shapes, not the volume, are what this varies.
        for round in 0..200u64 {
            let n = (rng.next_u64() % 49) as usize;
            let lens: Vec<usize> = (0..n)
                .map(|_| match rng.next_u64() % 4 {
                    0 => 0,
                    1 => (rng.next_u64() % 130) as usize,
                    2 => (rng.next_u64() % 20_000) as usize,
                    _ => (rng.next_u64() % 150_000) as usize,
                })
                .collect();
            let workers = 1 + (rng.next_u64() % 12) as usize;
            let owned = buffers(&lens, &round.to_le_bytes());
            let slices: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            assert_eq!(
                sha256_slices_parallel(&slices, nz(workers)),
                serial(&slices),
                "round {round}: lens {lens:?}, {workers} workers"
            );
        }
    }

    #[test]
    fn concurrent_callers_each_get_their_own_answers() {
        // Several callers fanning out at once: no shared state between calls.
        let lens: Vec<usize> = (0..40).map(|i| (i * 7919) % 100_000).collect();
        std::thread::scope(|scope| {
            for t in 0..3u8 {
                let lens = lens.clone();
                scope.spawn(move || {
                    for r in 0..4u8 {
                        let owned = buffers(&lens, &[t, r]);
                        let slices: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
                        assert_eq!(
                            sha256_slices_parallel(&slices, SHA256_PARALLEL_MAX_WORKERS),
                            serial(&slices)
                        );
                    }
                });
            }
        });
    }
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
