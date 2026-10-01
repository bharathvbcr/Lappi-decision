//! Native inner loops for the shard pipeline (`tools/real_tokenizer_pipeline.py`).
//!
//! Measured before it was written (cProfile, phase-3 inputs, 2026-09-30): of 626 s profiled,
//! 268 s were `qd_data.minhash.MinHasher.signature` -- 100,018 calls, once per content unit in
//! `qd_data.dedupe.dedupe` and once per surviving row in `qd_data.split`'s cross-split
//! re-derivation. That is pure-Python big-integer arithmetic, and it is what this crate takes.
//!
//! The Python implementation is the reference oracle the parity tests import
//! (`python/tests/test_qd_prep_parity.py`) and is not edited: every `qd_data/*.py` is hashed
//! into each shard header's `code_fingerprint`.

pub mod blake2b;
pub mod minhash;
pub mod wire;
