//! A synthetic averaged checkpoint, its manifest and a base snapshot, built in a temp dir.
//!
//! The tower's config is tiny but satisfies qd-metal's kernel constraints (`head_dim` 256, GDN
//! key dim 128, value dim a multiple of 32), so the release it exports can be read back through
//! the serving loader's own CPU path. Tensors are spread over BF16, F16 and F32 so every cast
//! is exercised, and the F32 ones open with exact rounding ties.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use qd_export::safetensors::{Dtype, PlannedTensor, Writer};
use qd_export::ExportRequest;
use serde_json::{json, Value};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A directory removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("qd-export-it-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // A leftover temp dir only costs disk; the test already has its answer.
        let _ignored = std::fs::remove_dir_all(&self.0);
    }
}

pub const HIDDEN: usize = 64;
pub const INTER: usize = 128;
pub const VOCAB: usize = 300;

/// The tiny config, in the conditional-generation shape the real snapshot has.
pub fn tiny_config() -> Value {
    json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": true,
        "text_config": {
            "attention_bias": false, "attn_output_gate": true, "head_dim": 256,
            "hidden_act": "silu", "hidden_size": HIDDEN, "intermediate_size": INTER,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 1,
            "linear_num_value_heads": 1, "linear_value_head_dim": 32, "mlp_only_layers": [],
            "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 1,
            "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "vocab_size": VOCAB,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                                "partial_rotary_factor": 0.25}
        }
    })
}

/// A minimal `tokenizers` WordLevel tokenizer whose 17 answer letters are single tokens.
pub fn tiny_tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".into(), json!(0));
    for (i, c) in ('A'..='Z').enumerate() {
        vocab.insert(c.to_string(), json!(i + 1));
    }
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

/// One source tensor: its dtype, shape and raw little-endian bytes.
#[derive(Clone)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` values in [-2, 2), deterministic in `seed`.
pub fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            // A uniform mantissa in [1, 2), scaled to [-2, 2): every low bit is populated, so
            // the F32 -> BF16 rounding is exercised on all of them.
            let top = u32::try_from(splitmix(&mut s) >> 32).unwrap();
            f32::from_bits(0x3F80_0000 | (top & 0x007F_FFFF)) * 4.0 - 6.0
        })
        .collect()
}

/// Exact rounding ties and boundary values, written over the start of every F32 tensor.
pub const F32_EDGES: [u32; 6] = [0x3F80_8000, 0x3F81_8000, 0xBF80_8000, 0x0000_8000, 0x0001_8000, 0x3FFF_FFFF];

pub fn f32_tensor(shape: &[usize], seed: u64) -> Tensor {
    let n = shape.iter().product();
    let mut v = values(n, seed);
    for (slot, bits) in v.iter_mut().zip(F32_EDGES) {
        *slot = f32::from_bits(bits);
    }
    Tensor {
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

pub fn bf16_tensor(shape: &[usize], seed: u64) -> Tensor {
    let n = shape.iter().product();
    Tensor {
        dtype: Dtype::Bf16,
        shape: shape.to_vec(),
        bytes: values(n, seed)
            .iter()
            .flat_map(|x| tessl::tensor::f32_to_bf16_bits(*x).to_le_bytes())
            .collect(),
    }
}

pub fn f16_tensor(shape: &[usize], seed: u64) -> Tensor {
    let n = shape.iter().product();
    Tensor {
        dtype: Dtype::F16,
        shape: shape.to_vec(),
        bytes: values(n, seed)
            .iter()
            .flat_map(|x| tessl::tensor::f32_to_f16_bits(*x).to_le_bytes())
            .collect(),
    }
}

/// The standard source: every tower tensor the tiny config implies plus the span head.
pub fn standard_tensors() -> BTreeMap<String, Tensor> {
    let layout = qd_export::layout::Layout::from_config_json(&tiny_config().to_string()).unwrap();
    let mut out = BTreeMap::new();
    for (seed, (short, shape)) in (1u64..).zip(layout.text_tensors().unwrap()) {
        let t = if short.contains("mlp.") {
            f16_tensor(&shape, seed)
        } else if short.contains("layernorm") || short.contains("self_attn") || short.ends_with("dt_bias") {
            bf16_tensor(&shape, seed)
        } else {
            f32_tensor(&shape, seed)
        };
        out.insert(format!("tower.{short}"), t);
    }
    for (seed, (short, shape)) in (1000u64..).zip(layout.span_head_tensors()) {
        out.insert(format!("span_head.{short}"), f32_tensor(&shape, seed));
    }
    out
}

/// A fixture on disk.
pub struct Fixture {
    pub dir: TempDir,
    pub source: PathBuf,
    pub manifest: PathBuf,
    pub snapshot: PathBuf,
    /// The train split's data manifest, holding [`TRAINED_FAMILIES`].
    pub train_manifest: PathBuf,
    pub out: PathBuf,
}

pub const FT_ROW_IDS: [&str; 3] = ["row-seed0", "row-seed1", "row-seed2"];

/// The families the fixture's train manifest holds, sorted. `devcouncil.verdict` is the task the
/// serving tests ask; `code.defect_class` is a family training builds.
pub const TRAINED_FAMILIES: [&str; 2] = ["code.defect_class", "devcouncil.verdict"];

/// A train-split data manifest in `Manifest.to_json()`'s shape (`python/qd_data/manifest.py`),
/// one row per family, written to `path`.
pub fn write_train_manifest(path: &Path, families: &[&str]) -> PathBuf {
    let entries: Vec<Value> = families
        .iter()
        .enumerate()
        .map(|(i, family)| {
            json!({
                "row_id": format!("train-row-{i}"),
                "content_hash": format!("{i:064x}"),
                "split": "train",
                "source_id": "test/source",
                "host": "test",
                "family_id": family,
                "repo_key": format!("repo-{i}"),
                "identity_key": format!("identity-{i}"),
                "licence_id": "MIT",
                "obligations": [],
            })
        })
        .collect();
    let manifest = json!({
        "manifest_format_version": 1,
        "split": "train",
        "data_snapshot_hash": "ab".repeat(32),
        "n_rows": entries.len(),
        "held_out_families": ["code.language_id", "qa.answerability"],
        "entries": entries,
    });
    std::fs::write(path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    path.to_path_buf()
}

pub fn tokenizer_sha256(snapshot: &Path) -> String {
    qd_runtime::hex(&qd_runtime::sha256(&std::fs::read(snapshot.join("tokenizer.json")).unwrap()))
}

/// Write `tensors` as the averaged checkpoint, its manifest (edited by `manifest_edit`), and a
/// base snapshot with `config`.
pub fn build(tensors: &BTreeMap<String, Tensor>, config: &Value, manifest_edit: impl FnOnce(&mut Value)) -> Fixture {
    let dir = TempDir::new("fx");
    let source = dir.0.join("avg.safetensors");
    let plan = tensors
        .iter()
        .map(|(name, t)| PlannedTensor {
            name: name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
        })
        .collect();
    let mut w = Writer::create(&source, plan).unwrap();
    for planned in w.order() {
        w.begin(&planned.name).unwrap();
        w.write(&tensors[&planned.name].bytes).unwrap();
        w.end().unwrap();
    }
    let written = w.finish().unwrap();

    let mut manifest = json!({
        "tool": "tools/ckpt_average.py",
        "resumable": false,
        "n_inputs": 3,
        "vocab_size": config["text_config"]["vocab_size"],
        "ft_row_ids": FT_ROW_IDS,
        "safetensors_sha256": qd_runtime::hex(&written.sha256),
        "n_tensors": tensors.len(),
    });
    manifest_edit(&mut manifest);
    let manifest_path = dir.0.join("avg.safetensors.manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    let snapshot = dir.0.join("snapshot");
    std::fs::create_dir(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.json"), serde_json::to_string_pretty(config).unwrap()).unwrap();
    std::fs::write(snapshot.join("tokenizer.json"), tiny_tokenizer_json()).unwrap();
    std::fs::write(snapshot.join("tokenizer_config.json"), r#"{"model_max_length": 262144}"#).unwrap();
    std::fs::write(snapshot.join("vocab.json"), r#"{"A": 1}"#).unwrap();
    std::fs::write(snapshot.join("merges.txt"), "#version: 0.2\n").unwrap();

    let train_manifest = write_train_manifest(&dir.0.join("train.json"), &TRAINED_FAMILIES);
    let out = dir.0.join("release");
    Fixture {
        dir,
        source,
        manifest: manifest_path,
        snapshot,
        train_manifest,
        out,
    }
}

pub fn standard() -> Fixture {
    build(&standard_tensors(), &tiny_config(), |_| {})
}

impl Fixture {
    pub fn request(&self) -> ExportRequest {
        ExportRequest {
            source: self.source.clone(),
            source_manifest: None,
            base_snapshot: self.snapshot.clone(),
            tokenizer_sha256: tokenizer_sha256(&self.snapshot),
            expect_vocab_size: VOCAB,
            calibration: None,
            train_manifest: self.train_manifest.clone(),
            allow_extra: Vec::new(),
            out: self.out.clone(),
        }
    }

    /// Staging directories an export left beside the release
    /// (`.release.qd-export-partial-<pid>`).
    pub fn leftovers(&self) -> Vec<String> {
        std::fs::read_dir(&self.dir.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("qd-export-partial"))
            .collect()
    }
}
