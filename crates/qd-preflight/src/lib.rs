//! The part of `qd-preflight` other crates call: the sibling-pin check ([`pin`]).
//!
//! The binary (`src/main.rs`) owns the CUDA preflight; the pin check lives here so that the
//! `qd-pin` gate and `qd-train-metal`'s refusal resolve a sibling's commit through one
//! implementation rather than two that could drift.

pub mod pin;
pub mod release_check;
pub mod cite;
