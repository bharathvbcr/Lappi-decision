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
//! **The layer-wise split.** F's recipe trains decoder layers 0-7 at 0.1x the learning rate
//! (`--lower-layers-n 8 --lower-layers-lr-scale 0.1`). ojas-qwen35's `adamw_step` takes a plan
//! of per-entry learning-rate scales (tessl's `adamw_step_scaled`), so the provider passes the
//! loop's own `lr_scale` vector through whole ([`provider::build_plan`]); the run is recorded
//! `optimizer_groups: "two"` with `lower_layers_n` and `lower_lr_scale` in the recipe. Without
//! the flags it is the single-group optimizer (`"single"`), which is not rung (d). The GPU
//! parity tests of the split (`tests/gpu_rung_b.rs`) are `#[ignore]` and have not been run
//! against this path.

pub mod args;
pub mod pin;
pub mod remap;

#[cfg(target_os = "macos")]
pub mod provider;
#[cfg(target_os = "macos")]
pub mod run;
