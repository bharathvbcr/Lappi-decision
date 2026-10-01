//! `qd-train`: Lappi's Rust fine-tune path, without PyTorch.
//!
//! Plan: `AUDIT/ojas-training-2026-10-01/fable-advice.md` (Fable, 2026-10-01). Kernels are
//! canonical tessl's (CLAUDE.md rule 6); this crate holds what is Lappi-specific: reading the
//! v4 shards behind the held-out door (CLAUDE.md rule 3), the supervision port, and the span
//! head. Every number it produces is checked against the PyTorch trainer it replaces, on the
//! parity ladder in the plan (rungs a-d); rows it writes are `quick` (rule 8).

pub mod files;
pub mod held_out;
pub mod np_random;
pub mod npy;
pub mod npz;
pub mod pyjson;
pub mod shards;
pub mod supervision;
pub mod tristate;
