//! `config.json` → the shapes this crate runs, validated against what tessl's kernels implement.
//!
//! Every number the forward pass uses comes from here, never from a constant: the model is
//! described by its checkpoint, and a checkpoint describing something the kernels cannot run is
//! refused at load with the field named, not discovered as wrong logits.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{MetalError, Result};

/// Which mixer a decoder layer uses (`text_config.layer_types`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// `linear_attention`: the gated delta net.
    Gdn,
    /// `full_attention`: gated softmax attention with partial RoPE.
    Attention,
}

/// The text tower's shapes. Built only by [`ModelConfig::from_json`], which validates them.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub layer_kinds: Vec<LayerKind>,
    pub rms_eps: f32,
    // GDN
    pub gdn_k_heads: u32,
    pub gdn_v_heads: u32,
    pub gdn_k_dim: u32,
    pub gdn_v_dim: u32,
    pub conv_kernel: u32,
    // attention
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub rope_theta: f32,
}

fn field<'a>(obj: &'a Value, key: &str) -> Result<&'a Value> {
    obj.get(key)
        .ok_or_else(|| MetalError::Config(format!("text_config.{key} is missing")))
}

fn usize_field(obj: &Value, key: &str) -> Result<usize> {
    let v = field(obj, key)?
        .as_u64()
        .ok_or_else(|| MetalError::Config(format!("text_config.{key} is not a non-negative integer")))?;
    let v = usize::try_from(v)
        .map_err(|_| MetalError::Config(format!("text_config.{key} = {v} overflows usize")))?;
    if v == 0 {
        return Err(MetalError::Config(format!("text_config.{key} is zero")));
    }
    Ok(v)
}

fn u32_field(obj: &Value, key: &str) -> Result<u32> {
    let v = usize_field(obj, key)?;
    u32::try_from(v).map_err(|_| MetalError::Config(format!("text_config.{key} = {v} exceeds u32")))
}

fn f64_field(obj: &Value, key: &str) -> Result<f64> {
    let v = field(obj, key)?
        .as_f64()
        .ok_or_else(|| MetalError::Config(format!("text_config.{key} is not a number")))?;
    if !v.is_finite() {
        return Err(MetalError::Config(format!("text_config.{key} is not finite")));
    }
    Ok(v)
}

fn require_eq<T: PartialEq + std::fmt::Debug>(key: &str, got: T, want: T, why: &str) -> Result<()> {
    if got != want {
        return Err(MetalError::Config(format!(
            "text_config.{key} is {got:?}; this backend implements only {want:?} ({why})"
        )));
    }
    Ok(())
}

impl ModelConfig {
    /// Read `<snapshot>/config.json`.
    pub fn load(snapshot: &Path) -> Result<Self> {
        let path = snapshot.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| MetalError::Config(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Parse and validate. The top-level object may be the conditional-generation config (with a
    /// `text_config`) or a text config itself.
    pub fn from_json(text: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(text)
            .map_err(|e| MetalError::Config(format!("config.json is not JSON: {e}")))?;
        let tc = root.get("text_config").unwrap_or(&root);
        if !tc.is_object() {
            return Err(MetalError::Config("text_config is not an object".into()));
        }

        let hidden = usize_field(tc, "hidden_size")?;
        let intermediate = usize_field(tc, "intermediate_size")?;
        let vocab = usize_field(tc, "vocab_size")?;
        let n_layers = usize_field(tc, "num_hidden_layers")?;
        let types = field(tc, "layer_types")?
            .as_array()
            .ok_or_else(|| MetalError::Config("text_config.layer_types is not an array".into()))?;
        if types.len() != n_layers {
            return Err(MetalError::Config(format!(
                "text_config.layer_types has {} entries for {n_layers} layers",
                types.len()
            )));
        }
        let layer_kinds = types
            .iter()
            .enumerate()
            .map(|(i, t)| match t.as_str() {
                Some("linear_attention") => Ok(LayerKind::Gdn),
                Some("full_attention") => Ok(LayerKind::Attention),
                other => Err(MetalError::Config(format!(
                    "text_config.layer_types[{i}] is {other:?}, not linear_attention/full_attention"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;

        let eps = f64_field(tc, "rms_norm_eps")?;
        if eps <= 0.0 {
            return Err(MetalError::Config("text_config.rms_norm_eps must be positive".into()));
        }

        // Semantics the kernels hard-code. Each is a property of the forward pass, so a
        // checkpoint that differs is a different model, not a different size.
        require_eq(
            "hidden_act",
            field(tc, "hidden_act")?.as_str(),
            Some("silu"),
            "the MLP is nn::mlp_silu",
        )?;
        require_eq(
            "attn_output_gate",
            field(tc, "attn_output_gate")?.as_bool(),
            Some(true),
            "the attention output is qwen35::attn_output_gate",
        )?;
        require_eq(
            "attention_bias",
            field(tc, "attention_bias")?.as_bool(),
            Some(false),
            "the fused projection has no bias",
        )?;
        require_eq(
            "tie_word_embeddings",
            root.get("tie_word_embeddings")
                .or_else(|| tc.get("tie_word_embeddings"))
                .and_then(Value::as_bool),
            Some(true),
            "the answer rows are scored against embed_tokens",
        )?;
        let mlp_only = tc
            .get("mlp_only_layers")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        require_eq("mlp_only_layers (length)", mlp_only, 0, "every layer has a mixer")?;

        let gdn_k_dim = u32_field(tc, "linear_key_head_dim")?;
        require_eq(
            "linear_key_head_dim",
            gdn_k_dim,
            tessl::qwen35::GDN_KEY_DIM,
            "the GDN kernels are compiled for this key dim",
        )?;
        let gdn_k_heads = u32_field(tc, "linear_num_key_heads")?;
        let gdn_v_heads = u32_field(tc, "linear_num_value_heads")?;
        let gdn_v_dim = u32_field(tc, "linear_value_head_dim")?;
        let conv_kernel = u32_field(tc, "linear_conv_kernel_dim")?;
        // GdnProjLayout::new is the canonical validation of the GDN shape.
        tessl::qwen35::GdnProjLayout::new(gdn_k_heads, gdn_v_heads, gdn_v_dim)
            .map_err(|e| MetalError::Config(format!("GDN shape: {e}")))?;
        if !(2..=8).contains(&conv_kernel) {
            return Err(MetalError::Config(format!(
                "text_config.linear_conv_kernel_dim = {conv_kernel}; conv1d_silu supports 2..=8"
            )));
        }

        let q_heads = u32_field(tc, "num_attention_heads")?;
        let kv_heads = u32_field(tc, "num_key_value_heads")?;
        let head_dim = u32_field(tc, "head_dim")?;
        if q_heads % kv_heads != 0 {
            return Err(MetalError::Config(format!(
                "num_attention_heads {q_heads} is not a multiple of num_key_value_heads {kv_heads}"
            )));
        }
        tessl::qwen35::AttnProjLayout::new(q_heads, kv_heads, head_dim)
            .map_err(|e| MetalError::Config(format!("attention shape: {e}")))?;
        require_eq(
            "head_dim",
            head_dim,
            256,
            "the prefill and decode attention kernels used here are the h256 variants",
        )?;

        let rope = field(tc, "rope_parameters")?;
        require_eq(
            "rope_parameters.rope_type",
            rope.get("rope_type").and_then(Value::as_str),
            Some("default"),
            "plain (text-only mRoPE) rotary",
        )?;
        let theta = f64_field(rope, "rope_theta")?;
        let factor = f64_field(rope, "partial_rotary_factor")?;
        let rotary = f64::from(head_dim) * factor;
        if theta <= 0.0 || rotary.fract() != 0.0 || rotary <= 0.0 || rotary > f64::from(head_dim) {
            return Err(MetalError::Config(format!(
                "rope: theta {theta}, head_dim {head_dim} x partial_rotary_factor {factor} = \
                 {rotary} is not a positive whole number of dims"
            )));
        }
        // `rotary` is a whole number in (0, head_dim], so it fits u32.
        let rotary_dim = rotary as u32;
        if !rotary_dim.is_multiple_of(2) {
            return Err(MetalError::Config(format!("rotary dim {rotary_dim} is odd")));
        }

        Ok(Self {
            hidden,
            intermediate,
            vocab,
            layer_kinds,
            rms_eps: eps as f32,
            gdn_k_heads,
            gdn_v_heads,
            gdn_k_dim,
            gdn_v_dim,
            conv_kernel,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta: theta as f32,
        })
    }

    pub fn n_layers(&self) -> usize {
        self.layer_kinds.len()
    }

    pub fn n_gdn(&self) -> usize {
        self.layer_kinds.iter().filter(|k| **k == LayerKind::Gdn).count()
    }

    pub fn n_attention(&self) -> usize {
        self.n_layers() - self.n_gdn()
    }

    pub fn gdn_layout(&self) -> Result<tessl::qwen35::GdnProjLayout> {
        tessl::qwen35::GdnProjLayout::new(self.gdn_k_heads, self.gdn_v_heads, self.gdn_v_dim)
            .map_err(MetalError::Config)
    }

    pub fn attn_layout(&self) -> Result<tessl::qwen35::AttnProjLayout> {
        tessl::qwen35::AttnProjLayout::new(self.q_heads, self.kv_heads, self.head_dim)
            .map_err(MetalError::Config)
    }
}

/// Where the Qwen3.5-2B-Base snapshot lives: `explicit` if given, else the one directory under
/// the Hugging Face cache. More than one snapshot is refused rather than picked.
pub fn resolve_snapshot(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.join("config.json").is_file() {
            return Err(MetalError::Config(format!("{}: no config.json", p.display())));
        }
        return Ok(p.to_path_buf());
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| MetalError::Config("HOME is unset; pass the snapshot directory".into()))?;
    let root = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots");
    let entries = std::fs::read_dir(&root)
        .map_err(|e| MetalError::Config(format!("{}: {e}", root.display())))?;
    let mut dirs = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| MetalError::Config(format!("{}: {e}", root.display())))?;
        if entry.path().join("config.json").is_file() {
            dirs.push(entry.path());
        }
    }
    match dirs.len() {
        1 => Ok(dirs.remove(0)),
        0 => Err(MetalError::Config(format!("{}: no snapshot with a config.json", root.display()))),
        n => Err(MetalError::Config(format!(
            "{}: {n} snapshots; pass the one to load explicitly",
            root.display()
        ))),
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    /// The real 2B text_config, verbatim in the fields this module reads.
    pub(crate) const QWEN35_2B: &str = r#"{
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "tie_word_embeddings": true,
      "text_config": {
        "attention_bias": false, "attn_output_gate": true, "head_dim": 256,
        "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 6144,
        "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention",
          "linear_attention","linear_attention","linear_attention","full_attention",
          "linear_attention","linear_attention","linear_attention","full_attention",
          "linear_attention","linear_attention","linear_attention","full_attention",
          "linear_attention","linear_attention","linear_attention","full_attention",
          "linear_attention","linear_attention","linear_attention","full_attention"],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 16, "linear_value_head_dim": 128, "mlp_only_layers": [],
        "num_attention_heads": 8, "num_hidden_layers": 24, "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "vocab_size": 248320,
        "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11,11,10],
          "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}
      }
    }"#;
}

#[cfg(test)]
mod tests {
    use super::tests_support::QWEN35_2B;
    use super::*;

    fn with(edit: impl FnOnce(&mut Value)) -> String {
        let mut v: Value = serde_json::from_str(QWEN35_2B).unwrap();
        edit(&mut v);
        v.to_string()
    }

    #[test]
    fn parses_the_2b_config() {
        let c = ModelConfig::from_json(QWEN35_2B).unwrap();
        assert_eq!((c.hidden, c.intermediate, c.vocab), (2048, 6144, 248_320));
        assert_eq!((c.n_gdn(), c.n_attention()), (18, 6));
        assert_eq!(c.layer_kinds[3], LayerKind::Attention);
        assert_eq!(c.rotary_dim, 64);
        assert_eq!((c.q_heads, c.kv_heads, c.head_dim), (8, 2, 256));
        assert_eq!((c.gdn_k_heads, c.gdn_v_heads, c.gdn_v_dim), (16, 16, 128));
        assert_eq!(c.rope_theta, 1e7);
        assert_eq!(c.rms_eps, 1e-6);
    }

    #[test]
    fn refuses_what_the_kernels_do_not_implement() {
        let cases: Vec<(String, &str)> = vec![
            (with(|v| v["text_config"]["hidden_act"] = "gelu".into()), "hidden_act"),
            (with(|v| v["text_config"]["linear_key_head_dim"] = 64.into()), "linear_key_head_dim"),
            (with(|v| v["text_config"]["linear_value_head_dim"] = 48.into()), "GDN shape"),
            (with(|v| v["text_config"]["head_dim"] = 128.into()), "head_dim"),
            (with(|v| v["text_config"]["num_hidden_layers"] = 23.into()), "layer_types has 24"),
            (with(|v| v["text_config"]["layer_types"][0] = "sliding".into()), "layer_types[0]"),
            (with(|v| v["text_config"]["attn_output_gate"] = false.into()), "attn_output_gate"),
            (with(|v| v["tie_word_embeddings"] = false.into()), "tie_word_embeddings"),
            (
                with(|v| v["text_config"]["rope_parameters"]["partial_rotary_factor"] = 0.3.into()),
                "partial_rotary_factor",
            ),
            (
                with(|v| v["text_config"]["rope_parameters"]["rope_type"] = "yarn".into()),
                "rope_type",
            ),
            (with(|v| v["text_config"]["mlp_only_layers"] = serde_json::json!([3])), "mlp_only"),
            (with(|v| v["text_config"]["rms_norm_eps"] = 0.0.into()), "rms_norm_eps"),
            (
                with(|v| {
                    v["text_config"].as_object_mut().unwrap().remove("vocab_size");
                }),
                "vocab_size is missing",
            ),
        ];
        for (json, needle) in cases {
            let err = ModelConfig::from_json(&json).unwrap_err().to_string();
            assert!(err.contains(needle), "expected {needle:?} in {err:?}");
        }
    }
}
