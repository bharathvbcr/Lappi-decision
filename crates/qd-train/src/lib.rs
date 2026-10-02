//! `qd-train`: Lappi's Rust fine-tune path, without PyTorch.
//!
//! Plan: `AUDIT/ojas-training-2026-10-01/fable-advice.md` (Fable, 2026-10-01). Kernels are
//! canonical tessl's (CLAUDE.md rule 6); this crate holds what is Lappi-specific: reading the
//! v4 shards behind the held-out door (CLAUDE.md rule 3), the supervision port, and the span
//! head. Every number it produces is checked against the PyTorch trainer it replaces, on the
//! parity ladder in the plan (rungs a-d); rows it writes are `quick` (rule 8).
//!
//! The trainer half (L-trainer): [`step`] is the model-generic step-provider trait the loop
//! drives, [`trainer`] the loop, [`objective`] Lappi's letter + span objective, [`schedule`] the
//! learning-rate schedule, [`recipe`] the optimizer recipe, [`adamw`] the host-side AdamW,
//! [`run_control`] the cap and the digests, [`ledger`] the `ft` row, [`export`] the trained
//! weights, and [`mock`] a tiny CPU model that exercises the loop with no GPU. [`ft_data`] puts
//! the data half's [`shards::Batch`] and the head lane's [`span_head::SpanHead`] behind the
//! objective's traits. [`pyjson`] (the bytes Python hashes) and [`tristate`] are shared by both
//! halves. None of it depends on tessl, so the crate builds and tests on Linux too.

pub mod adamw;
pub mod export;
pub mod files;
pub mod ft_data;
pub mod held_out;
pub mod ledger;
pub mod mock;
pub mod np_random;
pub mod npy;
pub mod npz;
pub mod objective;
pub mod pyjson;
pub mod recipe;
pub mod run_control;
pub mod schedule;
pub mod shards;
pub mod span_head;
pub mod step;
pub mod supervision;
pub mod trainer;
pub mod tristate;
