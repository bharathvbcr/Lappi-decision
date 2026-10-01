//! `qd-mutate` — the mutation engine.
//!
//! Takes a clean function from a code pool and emits a labeled diff. Labels are exact **because the
//! generator wrote them**; every design choice in this crate protects that property.
//!
//! The load-bearing invariant is the **span guarantee**: every emitted diff records the mutated line
//! range, and two independent derivations must agree or the example is dropped.
//!
//! * derivation A — [`span::span_from_byte_range`], fed by the mutator's own byte bookkeeping
//!   ([`edit::EditSet::apply`]);
//! * derivation B — [`diffspan::span_from_text_diff`], which sees only the before and after text and
//!   reads the changed range off a line diff.
//!
//! They share the *convention* (1-based, inclusive both ends, counted over the post-mutation file)
//! and nothing else. An off-by-one in either is invisible in class accuracy and would systematically
//! teach the pointer head to point one line off on 40% of the training mixture, so agreement is
//! checked on every example, never sampled.

pub mod compose;
pub mod diffspan;
pub mod edit;
pub mod fmt;
pub mod generate;
pub mod lang;
pub mod manifest;
pub mod normalize;
pub mod ops;
pub mod parse;
pub mod pool;
pub mod span;

pub use lang::{LangId, Language};
pub use span::LineSpan;

/// Version stamped into every manifest, so a data snapshot names the code that produced it.
pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");
