//! CPU-bound data-preparation loops ported out of Python, behind one binary (`qd-prep`).
//!
//! Python stays the reference for everything here. A port is admitted only with a parity test
//! that compares its output, byte for byte, with the Python function it replaces, on real
//! corpus rows and on adversarial inputs; the Python caller refuses to proceed when the binary
//! is absent, unrunnable, or returns anything but exactly the shape it asked for.
//!
//! - [`minhash`]: `qd_data.minhash.MinHasher.signature`, which was 75% of the
//!   `tools/real_ft_run.py --score-checkpoint` prelude (two `ft_splits` rebuilds, each signing
//!   every content unit for dedupe and every row again for the split's cross-check).
//! - [`ngram`] and [`linfit`]: `qd_train.baseline.CharNGramHasher.transform` and
//!   `LinearBaseline.fit`, the FT linear control's featurisation and fit, which ran 2.3 h+ on
//!   ~26 cores for one eval row of the full mixture on the GH200 (2026-10-01) without finishing.
//!   [`pairwise`] is numpy's summation order, which the fit reproduces.
//! - [`lsh`]: `qd_data.minhash.candidate_pairs`, the banded-LSH bucketing that dedupe and the
//!   split's cross-check run over those signatures: 24.6 s of the 78 s rebuild on the Mac
//!   (one unprofiled run at load ~40).
//! - [`containment`]: `qd_train.replay.decontaminate`'s word 8-gram containment with the
//!   complete pair list, the v5 decontamination scan (the Python gold-side scan of 2026-10-02
//!   peaked at 17.4 GB). [`pyunicode`] is Python's `\w` and `str.lower()`, from tables read
//!   off CPython ([`pyunicode_tables`]); [`sha256`] the digests its attestation names.
//! - [`team`]: threads started once and reused for every phase of the fit's loop, which
//!   started them twice an iteration (~3.4 ms an iteration at 52 threads on the H100 box).
//! - [`spancheck`]: `qd_train.shards._span_token_positions` up to its decode check -- a span
//!   slot's line starts and gold projected onto token positions, every refusal the reference
//!   makes before it decodes, and the runs its decode check walks. The per-slot Python of shard
//!   writing's stage 6 (`AUDIT/perf-pipeline-shards-2026-09-30.md`); decoding stays in Python.
//! - [`synth`], [`synth_email`], [`synth_jarvis`] and [`synth_tools`]: decision rows synthesised
//!   by rule for callers with no licensable or shareable data (the human's "Synthetic only"
//!   ruling of 2026-10-06). Not a port: there is no Python reference, so its tests pin the
//!   rules directly. [`pool`] is the seam every v6 pool producer writes through.

pub mod blake2b;
pub mod containment;
pub mod decisions;
pub mod linfit;
pub mod linwire;
pub mod lsh;
pub mod minhash;
pub mod ngram;
pub mod pairwise;
pub mod pool;
pub mod pyunicode;
#[rustfmt::skip]
mod pyunicode_tables;
pub mod sha256;
pub mod spancheck;
pub mod synth;
pub mod synth_email;
pub mod synth_jarvis;
pub mod synth_tools;
pub mod team;
pub mod wire;
