//! The trained model as the campaign scorer reads a Metal export: `tower.*` and `span_head.*`
//! in one safetensors file named `epoch-seed<N>-<device>.safetensors`, and
//! `<file>.manifest.json` beside it.
//!
//! **The contract** is the scorer's, read from `tools/real_ft_run.py` at main 1651fdf
//! (`_is_metal_export`, `_metal_export_seed`, `_read_metal_manifest`, `_metal_export_weights`)
//! and `ckpt_average.read_average`, not guessed:
//!
//! * the name is `epoch-seed<N>-metal.safetensors` -- `stem.rsplit("-", 2)` must give exactly
//!   `epoch`, `seed<N>` and the device, so the device is `[a-z0-9]+` and carries no `-`;
//! * the manifest carries every one of `METAL_MANIFEST_FIELDS` with its JSON type: `from`
//!   ([`EXPORT_SOURCE`]), `trainer` ([`METAL_TRAINER`]), `device` ([`METAL_DEVICE`]), `seed`,
//!   `optimizer_step`, `ft_row_id`, `vocab_size` (ints, never bools), `span_weight` (a number),
//!   `safetensors_sha256` (lowercase hex) and `n_tensors`. Further keys are provenance the scorer
//!   does not read, and are kept;
//! * the file holds only `tower.*` and `span_head.*`, `n_tensors` of them, hashing to
//!   `safetensors_sha256`; `vocab_size` is the tower's own row count (`load_weights` refuses a
//!   different remap), so it must equal `embed_tokens.weight`'s rows.
//!
//! A device other than `metal` writes a file the scorer refuses by name and by manifest, which
//! is what a CPU test's export should get.
//!
//! **The cast.** The provider's masters are f32. The scored export rounds the tower to bf16
//! once, to nearest with ties to even, by qd-export's [`qd_export::bf16::f32_to_bf16_rne`] (the
//! rule torch's `.to(torch.bfloat16)` and tessl use); a value that is not finite, or rounds to
//! infinity, is refused. The span head stays f32, as `QwenDecisionStep` keeps it. The masters
//! export ([`Precision::F32Masters`], rung (d)'s criterion 4) keeps the tower f32 under a name
//! and a `from` the scorer refuses, so it can never be scored in place of the bf16 export.
//!
//! **The file.** Written by qd-export's streaming [`qd_export::safetensors::Writer`] -- the one
//! the release path already uses -- which refuses an existing path, writes tensors in name order
//! and fsyncs. When a [`Layout`] is given, the tower's and head's names and shapes must be
//! exactly the ones it derives from `config.json` (with its `vocab` set to the run's remapped
//! vocabulary), so a missing, extra or misshapen tensor is refused here, not at load time.
//! Every value is checked before the file is created, so a refusal leaves nothing behind.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use qd_export::bf16::f32_to_bf16_rne;
use qd_export::layout::{Layout, SPAN_PREFIX, TOWER_PREFIX};
use qd_export::safetensors::{Dtype, PlannedTensor, Writer};
use serde_json::Value;

use crate::pyjson::{dumps, float, obj, PyJsonError, CANONICAL};
use crate::run_control::hex;
use crate::step::ParamSpec;

/// `real_ft_run.METAL_EXPORT_SOURCE`: the scored export's manifest `from`.
pub const EXPORT_SOURCE: &str = "qd-train-export";
/// The masters export's `from`: neither the scored export's nor an average's, so the scorer
/// refuses it.
pub const MASTERS_SOURCE: &str = "qd-train-export-f32-masters";
/// `real_ft_run.METAL_TRAINER`: the manifest's and the ft row's `trainer`.
pub const METAL_TRAINER: &str = "qd-train-metal";
/// `real_ft_run.METAL_DEVICE`.
pub const METAL_DEVICE: &str = "metal";
/// The only arm the scorer scores (`_metal_export_seed`).
pub const EXPORT_TAG: &str = "epoch";
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

fn refuse<T>(m: impl Into<String>) -> Result<T, ExportError> {
    Err(ExportError::Refused(m.into()))
}

/// Which artifact to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// The scored export: tower bf16 (RNE), head f32, `from` [`EXPORT_SOURCE`].
    Bf16,
    /// The f32 masters as they are, for the master-delta comparison: `from` [`MASTERS_SOURCE`],
    /// named `...-f32-masters.safetensors`.
    F32Masters,
}

/// One tensor to export: its entry and its f32 values, row-major in the entry's shape.
pub struct NamedTensor<'a> {
    pub spec: ParamSpec,
    pub values: &'a [f32],
}

/// What the manifest says about the run the weights came from.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestInfo {
    /// [`METAL_TRAINER`] for the Metal trainer.
    pub trainer: String,
    /// [`METAL_DEVICE`] for the Metal trainer; `[a-z0-9]+`.
    pub device: String,
    pub seed: u64,
    pub optimizer_step: u64,
    /// The pre-minted id of the ft row that will describe this run (a canonical UUID).
    pub ft_row_id: String,
    /// The tower's vocabulary (its `embed_tokens.weight` rows).
    pub vocab_size: u64,
    pub span_weight: f64,
    pub schedule: Value,
    pub provider: String,
    pub operands: String,
    pub recipe_hash: String,
    pub loss_log_digest: String,
    pub consumed_digest: String,
    /// sha256 of the span head's initial-weights file, when it was read from one.
    pub head_init_digest: Option<String>,
}

/// What [`export`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportSummary {
    pub weights: PathBuf,
    pub manifest: PathBuf,
    pub safetensors_sha256: String,
    /// sha256 of the manifest's bytes: what the scorer records as `trained_by.manifest_sha256`.
    pub manifest_sha256: String,
    pub n_tensors: usize,
}

/// `<out>.manifest.json`.
pub fn manifest_path(out: &Path) -> PathBuf {
    let mut name = out.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(MANIFEST_SUFFIX);
    out.with_file_name(name)
}

/// `epoch-seed<seed>-<device>.safetensors` (or `...-f32-masters.safetensors`). The device is
/// `[a-z0-9]+`: the scorer splits the stem on its last two `-`.
pub fn export_file_name(seed: u64, device: &str, precision: Precision) -> Result<String, ExportError> {
    if device.is_empty() || !device.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()) {
        return refuse(format!(
            "device {device:?} must be [a-z0-9]+: the scorer reads the name as <tag>-seed<N>-<device>"
        ));
    }
    Ok(match precision {
        Precision::Bf16 => format!("{EXPORT_TAG}-seed{seed}-{device}.safetensors"),
        Precision::F32Masters => format!("{EXPORT_TAG}-seed{seed}-{device}-f32-masters.safetensors"),
    })
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// `uuid.uuid4()`'s text: 8-4-4-4-12 lowercase hex.
fn is_canonical_uuid(s: &str) -> bool {
    let groups: Vec<&str> = s.split('-').collect();
    groups.len() == 5
        && groups.iter().zip([8, 4, 4, 4, 12]).all(|(g, n)| g.len() == n && g.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
}

impl ManifestInfo {
    fn validate(&self) -> Result<(), ExportError> {
        if self.trainer.trim().is_empty() || self.trainer.contains(char::is_whitespace) {
            return refuse(format!("trainer {:?} must be a non-empty name", self.trainer));
        }
        if !is_canonical_uuid(&self.ft_row_id) {
            return refuse(format!(
                "ft_row_id {:?} is not a canonical UUID: the scorer pairs the export with the ft row by it",
                self.ft_row_id
            ));
        }
        if self.vocab_size == 0 {
            return refuse("vocab_size 0");
        }
        if !(self.span_weight.is_finite() && self.span_weight > 0.0) {
            return refuse(format!("span_weight {} must be positive and finite", self.span_weight));
        }
        for (name, v) in [
            ("recipe_hash", Some(&self.recipe_hash)),
            ("loss_log_digest", Some(&self.loss_log_digest)),
            ("consumed_digest", Some(&self.consumed_digest)),
            ("head_init_digest", self.head_init_digest.as_ref()),
        ] {
            if let Some(v) = v
                && !is_sha256_hex(v)
            {
                return refuse(format!("{name} {v:?} is not a lowercase hex sha256"));
            }
        }
        Ok(())
    }
}

fn check_against(prefix: &str, given: &[NamedTensor<'_>], want: &BTreeMap<String, Vec<usize>>) -> Result<(), ExportError> {
    let have: BTreeMap<&str, &Vec<usize>> = given.iter().map(|t| (t.spec.name.as_str(), &t.spec.shape)).collect();
    if have.len() != given.len() {
        return refuse(format!("a {prefix}* tensor is named twice"));
    }
    let missing: Vec<&String> = want.keys().filter(|k| !have.contains_key(k.as_str())).collect();
    let extra: Vec<&&str> = have.keys().filter(|k| !want.contains_key(**k)).collect();
    if !missing.is_empty() || !extra.is_empty() {
        return refuse(format!(
            "the {prefix}* tensors are not the set config.json implies: missing {missing:?}, extra {extra:?}"
        ));
    }
    for (name, shape) in &have {
        if want[*name] != **shape {
            return refuse(format!("{prefix}{name} has shape {shape:?}; config.json implies {:?}", want[*name]));
        }
    }
    Ok(())
}

/// Write the weights to `<dir>/<export_file_name>` and the manifest beside it; returns what was
/// written. Refuses an existing file or manifest, a malformed [`ManifestInfo`], a `vocab_size`
/// that is not the tower's, a tensor whose value count disagrees with its shape, a non-finite
/// value, and -- when `layout` is given -- any tensor set but the one the loader reads.
pub fn export(
    dir: &Path,
    tower: &[NamedTensor<'_>],
    head: &[NamedTensor<'_>],
    layout: Option<&Layout>,
    info: &ManifestInfo,
    precision: Precision,
) -> Result<ExportSummary, ExportError> {
    info.validate()?;
    let out = dir.join(export_file_name(info.seed, &info.device, precision)?);
    let manifest = manifest_path(&out);
    for target in [out.as_path(), manifest.as_path()] {
        if target.exists() {
            return refuse(format!("{} already exists; refusing to overwrite it", target.display()));
        }
    }
    if let Some(l) = layout {
        if l.vocab as u64 != info.vocab_size {
            return refuse(format!(
                "vocab_size {} but the layout's vocabulary is {}: set Layout::vocab to the run's remapped vocabulary",
                info.vocab_size, l.vocab
            ));
        }
        check_against(TOWER_PREFIX, tower, &l.text_tensors()?)?;
        check_against(SPAN_PREFIX, head, &l.span_head_tensors())?;
    }
    if let Some(e) = tower.iter().find(|t| t.spec.name == "embed_tokens.weight")
        && e.spec.shape.first().map(|&r| r as u64) != Some(info.vocab_size)
    {
        return refuse(format!(
            "vocab_size {} but embed_tokens.weight is {:?}: load_weights refuses a different remap",
            info.vocab_size, e.spec.shape
        ));
    }
    let tower_bf16 = precision == Precision::Bf16;
    let mut plan = Vec::with_capacity(tower.len() + head.len());
    let mut sources: BTreeMap<String, Value> = BTreeMap::new();
    let mut by_name: BTreeMap<String, (&NamedTensor<'_>, bool)> = BTreeMap::new();
    for (tensors, prefix, bf16) in [(tower, TOWER_PREFIX, tower_bf16), (head, SPAN_PREFIX, false)] {
        for t in tensors {
            let n = t.spec.numel().map_err(|e| ExportError::Refused(e.to_string()))?;
            if n != t.values.len() {
                return refuse(format!("{}: {} values for shape {:?}", t.spec.name, t.values.len(), t.spec.shape));
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
                return refuse(format!("{name} is planned twice"));
            }
        }
    }
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
    let mut w = Writer::create(&out, plan)?;
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
    let (source, method) = match precision {
        Precision::Bf16 => (
            EXPORT_SOURCE,
            "one run's f32 master weights at its last optimizer step; tower.* rounded once to bf16 (RNE), span_head.* kept f32",
        ),
        Precision::F32Masters => (
            MASTERS_SOURCE,
            "one run's f32 master weights at its last optimizer step, as they are (for the master-delta comparison; never scored)",
        ),
    };
    let body = obj([
        ("from", Value::from(source)),
        ("trainer", Value::from(info.trainer.as_str())),
        ("device", Value::from(info.device.as_str())),
        ("seed", Value::from(info.seed)),
        ("optimizer_step", Value::from(info.optimizer_step)),
        ("ft_row_id", Value::from(info.ft_row_id.as_str())),
        ("vocab_size", Value::from(info.vocab_size)),
        ("span_weight", float(info.span_weight)?),
        ("safetensors_sha256", Value::from(sha.as_str())),
        ("n_tensors", Value::from(by_name.len())),
        ("tool", Value::from("crates/qd-train (export.rs)")),
        ("method", Value::from(method)),
        ("resumable", Value::Bool(false)),
        (
            "why_not_resumable",
            Value::from("weights only: the optimizer moments are in the trainer's checkpoint, not here"),
        ),
        ("schedule", info.schedule.clone()),
        ("provider", Value::from(info.provider.as_str())),
        ("operands", Value::from(info.operands.as_str())),
        ("recipe_hash", Value::from(info.recipe_hash.as_str())),
        ("loss_log_digest", Value::from(info.loss_log_digest.as_str())),
        ("consumed_digest", Value::from(info.consumed_digest.as_str())),
        ("head_init_digest", info.head_init_digest.as_deref().map_or(Value::Null, Value::from)),
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
        return refuse(format!("{} appeared while the weights were written", manifest.display()));
    }
    fs::rename(&tmp, &manifest).map_err(|e| ioe(&manifest, e))?;
    Ok(ExportSummary {
        weights: out,
        manifest,
        safetensors_sha256: sha,
        manifest_sha256: hex(&<sha2::Sha256 as sha2::Digest>::digest(text.as_bytes())),
        n_tensors: by_name.len(),
    })
}
