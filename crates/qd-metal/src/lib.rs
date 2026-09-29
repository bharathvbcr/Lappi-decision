//! `qd-metal` — Qwen3.5-2B decision inference on Apple silicon through tessl's Metal kernels.
//!
//! Plan: `docs/train-plan-2026-09-28.md`, Lane B Phase 2. This crate implements
//! [`qd_runtime::backend::DecisionBackend`]; every kernel it dispatches lives in canonical tessl
//! (`~/Code/research/tessl`, CLAUDE.md rule 6), which this crate uses and never edits.
//!
//! | Module | Holds |
//! | --- | --- |
//! | [`config`] | `config.json` → shapes, refused where the kernels do not implement them |
//! | [`safetensors`] | the checkpoint reader (hand-parsed header, positioned reads) |
//! | [`tokenizer`] | `tokenizer.json` via `tokenizers`, and the 17 answer-letter ids |
//! | [`weights`] | checkpoint tensors → the packed layouts the kernels read (CPU) |
//! | [`model`] | the GPU model: upload, prefill, snapshot-continuation decode, answer scoring |
//! | [`backend`] | [`backend::MetalBackend`]: the `DecisionBackend`, on one GPU-owning thread |
//!
//! Tests that need the model snapshot or the GPU are `#[ignore]`d and named `snapshot_*` /
//! `gpu_*`: `cargo test` never touches the GPU, and an ignored test is reported as ignored, not
//! as passed.

pub mod backend;
pub mod config;
pub mod error;
pub mod model;
pub mod safetensors;
pub mod tokenizer;
pub mod weights;

pub use error::{MetalError, Result};
