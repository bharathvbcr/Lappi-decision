//! The trained model, as the scorer and the release exporter read a checkpoint: `tower.*` in
//! bf16 and `span_head.*` in f32 in one safetensors file, and `<file>.manifest.json` beside it.
//!
//! **The cast.** The provider's masters are f32; the tower is rounded to bf16 once, to nearest
//! with ties to even, by qd-export's [`qd_export::bf16::f32_to_bf16_rne`] (the rule torch's
//! `.to(torch.bfloat16)` and tessl use). A value that is not finite, or that rounds to
//! infinity, is refused: a weight that is not a number is a broken checkpoint. The span head
//! stays f32, as `QwenDecisionStep` keeps it (`backbone.py:994-1002`), and is checked finite.
//!
//! **The file.** Written by qd-export's streaming [`qd_export::safetensors::Writer`] -- the one
//! the release path already uses, not a second writer -- which refuses an existing path, writes
//! tensors in name order and fsyncs. Names are `tower.<transformers name below the text tower>`
//! and `span_head.<SpanPointerHead.state_dict() name>`, the averaged checkpoint's convention
//! (`qd_export::layout::{TOWER_PREFIX, SPAN_PREFIX}`). When a [`Layout`] is given, the tower's
//! and head's names and shapes must be exactly the ones it derives from `config.json` -- the set
//! qd-metal's loader reads -- so a missing, extra or misshapen tensor is refused here rather
//! than at load time.
//!
//! **The manifest.** Modelled on `tools/ckpt_average.py::write`'s keys where they apply
//! (`optimizer_step`, `schedule`, `inputs`, `ft_row_ids`, `safetensors_sha256`, `n_tensors`,
//! `tensor_sources`, `resumable`). Its `from` is `"metal-masters"`, which is neither of
//! `ckpt_average.SOURCES` (`tower`, `masters`), so the scorer cannot mistake a Metal artifact
//! for an average. The contract for scoring it is L-scorer's change (human ask 7, approved) and
//! L-oracle's spec, pending (`GAP-LTRAINER-METAL-ARTIFACT-CONTRACT-PENDING-2026-10-01`).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use qd_export::bf16::f32_to_bf16_rne;
use qd_export::layout::{Layout, SPAN_PREFIX, TOWER_PREFIX};
use qd_export::safetensors::{Dtype, PlannedTensor, Writer};

use serde_json::Value;

use crate::pyjson::{dumps, obj, PyJsonError, CANONICAL};
use crate::run_control::hex;
use crate::step::ParamSpec;

/// The manifest's `from`: an export of one run's f32 masters, made on Metal.
pub const FROM_METAL_MASTERS: &str = "metal-masters";
/// `ckpt_average.MANIFEST_SUFFIX`.
pub const MANIFEST_SUFFIX: &str = ".manifest.json";

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ExportError {
    #[error("refused: {0}")]
    Refused(String),
    #[error("non-finite: {0}")]
    NonFinite(String),
    #[error(transparent)]
    Writer(#[from] qd_export::Refusal),
    #[error(transparent)]
    Json(#[from] PyJsonError),
    #[error("io: {0}")]
    Io(String),
}

/// One tensor to export: its entry and its f32 values, row-major in the entry's shape.
pub struct NamedTensor<'a> {
    pub spec: ParamSpec,
    pub values: &'a [f32],
}

/// What the manifest says about the run the weights came from.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestInfo {
    pub optimizer_step: u64,
    pub seed: u64,
    pub schedule: Value,
    /// The ft row that trained these weights; `None` only when no row was written (a test).
    pub ft_row_id: Option<String>,
    pub provider: String,
    pub operands: String,
    pub recipe_hash: String,
    pub loss_log_digest: String,
    pub consumed_digest: String,
}

/// What [`export`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportSummary {
    pub weights: PathBuf,
    pub manifest: PathBuf,
    pub safetensors_sha256: String,
    pub n_tensors: usize,
}

/// `<out>.manifest.json`.
pub fn manifest_path(out: &Path) -> PathBuf {
    let mut name = out.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(MANIFEST_SUFFIX);
    out.with_file_name(name)
}

fn check_against(prefix: &str, given: &[NamedTensor<'_>], want: &BTreeMap<String, Vec<usize>>) -> Result<(), ExportError> {
    let have: BTreeMap<&str, &Vec<usize>> = given.iter().map(|t| (t.spec.name.as_str(), &t.spec.shape)).collect();
    if have.len() != given.len() {
        return Err(ExportError::Refused(format!("a {prefix}* tensor is named twice")));
    }
    let missing: Vec<&String> = want.keys().filter(|k| !have.contains_key(k.as_str())).collect();
    let extra: Vec<&&str> = have.keys().filter(|k| !want.contains_key(**k)).collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(ExportError::Refused(format!(
            "the {prefix}* tensors are not the set config.json implies: missing {missing:?}, extra {extra:?}"
        )));
    }
    for (name, shape) in &have {
        if want[*name] != **shape {
            return Err(ExportError::Refused(format!(
                "{prefix}{name} has shape {shape:?}; config.json implies {:?}",
                want[*name]
            )));
        }
    }
    Ok(())
}

/// Write the weights to `out` and the manifest beside it. Refuses an existing `out` or
/// manifest, a tensor whose value count disagrees with its shape, a non-finite value, and --
/// when `layout` is given -- any tensor set but the one the loader reads.
pub fn export(
    out: &Path,
    tower: &[NamedTensor<'_>],
    head: &[NamedTensor<'_>],
    layout: Option<&Layout>,
    info: &ManifestInfo,
) -> Result<ExportSummary, ExportError> {
    let manifest = manifest_path(out);
    for target in [out, manifest.as_path()] {
        if target.exists() {
            return Err(ExportError::Refused(format!("{} already exists; refusing to overwrite it", target.display())));
        }
    }
    if let Some(l) = layout {
        check_against(TOWER_PREFIX, tower, &l.text_tensors()?)?;
        check_against(SPAN_PREFIX, head, &l.span_head_tensors())?;
    }
    let mut plan = Vec::with_capacity(tower.len() + head.len());
    let mut sources: BTreeMap<String, Value> = BTreeMap::new();
    let mut by_name: BTreeMap<String, (&NamedTensor<'_>, bool)> = BTreeMap::new();
    for (tensors, prefix, bf16) in [(tower, TOWER_PREFIX, true), (head, SPAN_PREFIX, false)] {
        for t in tensors {
            let n = t.spec.numel().map_err(|e| ExportError::Refused(e.to_string()))?;
            if n != t.values.len() {
                return Err(ExportError::Refused(format!(
                    "{}: {} values for shape {:?}",
                    t.spec.name,
                    t.values.len(),
                    t.spec.shape
                )));
            }
            let name = format!("{prefix}{}", t.spec.name);
            plan.push(PlannedTensor {
                name: name.clone(),
                dtype: if bf16 { Dtype::Bf16 } else { Dtype::F32 },
                shape: t.spec.shape.clone(),
            });
            sources.insert(
                name.clone(),
                Value::from(if bf16 { "f32 master, rounded to bf16 (RNE)" } else { "f32 master, as is" }),
            );
            if by_name.insert(name.clone(), (t, bf16)).is_some() {
                return Err(ExportError::Refused(format!("{name} is planned twice")));
            }
        }
    }
    // Every value is checked before the file is created, so a refusal leaves nothing behind.
    for (name, (t, bf16)) in &by_name {
        for (i, &x) in t.values.iter().enumerate() {
            let ok = if *bf16 { f32_to_bf16_rne(x).is_some() } else { x.is_finite() };
            if !ok {
                return Err(ExportError::NonFinite(format!(
                    "{name}[{i}] = {x} is not a finite {}",
                    if *bf16 { "bf16" } else { "f32" }
                )));
            }
        }
    }
    let mut w = Writer::create(out, plan)?;
    let order = w.order();
    for p in &order {
        let (t, bf16) = by_name[&p.name];
        w.begin(&p.name)?;
        let mut buf = Vec::with_capacity(t.values.len() * if bf16 { 2 } else { 4 });
        for &x in t.values {
            if bf16 {
                let b = f32_to_bf16_rne(x).ok_or_else(|| ExportError::NonFinite(format!("{} changed while written", p.name)))?;
                buf.extend_from_slice(&b.to_le_bytes());
            } else {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        w.write(&buf)?;
        w.end()?;
    }
    let written = w.finish()?;
    let sha = hex(&written.sha256);
    let body = obj([
        ("tool", Value::from("crates/qd-train (export.rs)")),
        ("from", Value::from(FROM_METAL_MASTERS)),
        (
            "method",
            Value::from("one run's f32 master weights at its last optimizer step; tower.* rounded once to bf16 (RNE), span_head.* kept f32"),
        ),
        ("resumable", Value::Bool(false)),
        (
            "why_not_resumable",
            Value::from("weights only: the optimizer moments are in the trainer's checkpoint, not here"),
        ),
        ("n_inputs", Value::from(1u64)),
        ("optimizer_step", Value::from(info.optimizer_step)),
        ("schedule", info.schedule.clone()),
        (
            "inputs",
            Value::Array(vec![obj([
                ("seed", Value::from(info.seed)),
                ("provider", Value::from(info.provider.as_str())),
                ("operands", Value::from(info.operands.as_str())),
                ("recipe_hash", Value::from(info.recipe_hash.as_str())),
                ("loss_log_digest", Value::from(info.loss_log_digest.as_str())),
                ("consumed_digest", Value::from(info.consumed_digest.as_str())),
            ])?]),
        ),
        (
            "ft_row_ids",
            Value::Array(info.ft_row_id.iter().map(|r| Value::from(r.as_str())).collect()),
        ),
        ("safetensors_sha256", Value::from(sha.as_str())),
        ("n_tensors", Value::from(by_name.len())),
        ("tensor_sources", obj(sources)?),
    ])?;
    let text = dumps(&body, CANONICAL)? + "\n";
    let tmp = manifest.with_file_name(format!(
        ".{}.partial",
        manifest.file_name().and_then(|n| n.to_str()).unwrap_or("manifest")
    ));
    let ioe = |p: &Path, e: std::io::Error| ExportError::Io(format!("{}: {e}", p.display()));
    {
        let mut f = fs::File::options().write(true).create_new(true).open(&tmp).map_err(|e| ioe(&tmp, e))?;
        f.write_all(text.as_bytes()).map_err(|e| ioe(&tmp, e))?;
        f.sync_all().map_err(|e| ioe(&tmp, e))?;
    }
    if manifest.exists() {
        return Err(ExportError::Refused(format!("{} appeared while the weights were written", manifest.display())));
    }
    fs::rename(&tmp, &manifest).map_err(|e| ioe(&manifest, e))?;
    Ok(ExportSummary {
        weights: out.to_path_buf(),
        manifest,
        safetensors_sha256: sha,
        n_tensors: by_name.len(),
    })
}
