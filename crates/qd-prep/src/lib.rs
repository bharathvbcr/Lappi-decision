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

pub mod blake2b;
pub mod minhash;
pub mod wire;
