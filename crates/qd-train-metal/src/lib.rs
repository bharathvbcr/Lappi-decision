//! `qd-train-metal`: Lappi's fine-tune on Apple silicon, without PyTorch.
//!
//! `qd-train` owns the loop, the objective, the schedule, the ledger row and the export, and
//! none of it knows a GPU. This crate adds the one thing that does: [`provider`], qd-train's
//! [`qd_train::step::StepProvider`] for ojas-qwen35's `Qwen35Step` (a Qwen3.5 text tower on
//! canonical tessl's Metal kernels). ojas never depends on Lappi; the adapter lives here.
//!
//! [`run`] wires one run end to end: the train shard set through the held-out door, the
//! identity-remap check ([`remap`]), the span head from its initial-weights file, the loop, the
//! scorer's Metal export (bf16 and the f32 masters) and a `ledger/mac-ojas-*.jsonl` row, always
//! `quick`. [`args`] is the command line, and refuses what this trainer cannot honour before
//! anything is read or opened.
//!
//! **What runs where.** [`args`] and [`remap`] are portable and tested on any host by
//! `cargo test`. [`provider`] and [`run`] are macOS-only, because tessl is. Every test that
//! opens a Metal runtime is `#[ignore]` and named `gpu_*` (`tests/gpu_tiny.rs`); `cargo test`
//! never touches the GPU.
//!
//! **What it cannot run yet.** F's recipe trains decoder layers 0-7 at 0.1x the learning rate
//! (`--lower-layers-n 8 --lower-layers-lr-scale 0.1`). tessl's `Qwen35Model::adamw_step` takes
//! one learning rate for every tensor, so a per-entry scale is refused at the command line, by
//! qd-train's loop and by ojas-qwen35's plan check -- never folded into one rate or split into
//! two updates. Until tessl's per-entry `lr_scale` lands, a run of this binary is the
//! single-group optimizer, recorded as such (`optimizer_groups: "single"`), and is not rung (d).

pub mod args;
pub mod remap;

#[cfg(target_os = "macos")]
pub mod provider;
#[cfg(target_os = "macos")]
pub mod run;
