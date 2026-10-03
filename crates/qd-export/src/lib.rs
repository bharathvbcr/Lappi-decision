//! `qd-export` — an averaged checkpoint to the release directory the serving loader reads.
//!
//! Plan: phase 6, piece (iii). Input is what `tools/ckpt_average.py` writes: one safetensors file
//! of `tower.*` (the Qwen3.5-2B text tower as `Qwen3_5TextModel.state_dict()` names it,
//! `python/qd_train/backbone.py:1000-1054`) and `span_head.*` tensors, plus
//! `<file>.manifest.json`. Output is the layout **qd-metal's loader** reads, which is a
//! Hugging Face snapshot's text half:
//!
//! | File | Loader | Holds |
//! | --- | --- | --- |
//! | `config.json` | `config.rs:87-92` | the base snapshot's, byte for byte |
//! | `model.safetensors` | `model.rs:870-901` | the 320 `model.language_model.*` tensors, BF16 |
//! | `tokenizer.json` | `backend.rs:331`, `tokenizer.rs:28-37` | the base snapshot's, pinned |
//! | `tokenizer_config.json`, `vocab.json`, `merges.txt` | (transformers) | the base snapshot's |
//! | `span_head.safetensors` | none yet | the span pointer head, F32 |
//! | `calibration.json` | `qd_runtime::release` | optional here, validated; a release without it is refused at load |
//! | `release_manifest.json` | `qd_runtime::release` | every file's sha256, the source, and `expected_identity`: the tower's `weight_hash` bound to `config.json`'s sha256 and the tokenizer's; `trained_families`, the train manifest's family set, which admission checks every request's task against |
//!
//! | Module | Holds |
//! | --- | --- |
//! | [`safetensors`] | reader and streaming writer, by hand |
//! | [`bf16`] | the one cast: round to nearest, ties to even |
//! | [`layout`] | `config.json` -> the tensor set and shapes the loader reads |
//! | [`export`] | the refusals, the staging directory, the manifest |
//! | [`ensemble`] | N released towers and the ensemble's table -> one `qd-ensemble.v1` directory |

pub mod bf16;
pub mod ensemble;
pub mod export;
pub mod layout;
pub mod refusal;
pub mod safetensors;

pub use ensemble::{EnsembleRequest, EnsembleSummary, export_ensemble};
pub use export::{export, AllowedExtra, ExportRequest, ExportSummary};
pub use refusal::{Refusal, RefusalKind};
