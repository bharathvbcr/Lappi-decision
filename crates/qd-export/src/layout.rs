//! `config.json` -> the exact tensor set, and shapes, the serving loader reads.
//!
//! The loader is qd-metal. It opens `<dir>/config.json` (`config.rs:87-92`), then
//! `<dir>/model.safetensors` (`model.rs:870-901`), and requires the `model.language_model.*`
//! tensors to be exactly the set the forward pass reads (`weights.rs:199-260`), each at the shape
//! `prepare_layer` and `Model::load` expect (`weights.rs:132-197`, `model.rs:328-345`). This
//! module derives that set and those shapes from the config, by the same arithmetic. It checks
//! only what decides the **layout**:
//!
//! * the sizes and `layer_types` (which mixer, hence which tensors, each layer has);
//! * `tie_word_embeddings` true (no `lm_head` tensor; the loader reads the top-level flag first,
//!   then `text_config`'s, `config.rs:154-161`, and so does this);
//! * `attention_bias` false (no bias tensors), `attn_output_gate` true (`q_proj` is `2 x` wide),
//!   and no `mlp_only_layers` (every layer has a mixer).
//!
//! It does **not** re-check qd-metal's kernel-only constraints (`head_dim == 256`,
//! `linear_key_head_dim == 128`, the GDN and rope rules, `config.rs:169-226`). Those are tessl's,
//! not the layout's. A release's `config.json` is the base snapshot's, copied byte for byte, and
//! the real Qwen3.5-2B-Base config passes them (`config.rs:339-349`); the round-trip test runs
//! qd-metal's own `ModelConfig` over the export to keep it so. Re-implementing them here would be
//! a second copy of tessl's limits that could drift from the first.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::refusal::{Refusal, RefusalKind, Result};

/// The loader's prefix for the text tower (`qd-metal/src/weights.rs:26`; the real checkpoint's
/// name, `python/qd_train/backbone.py:139`).
pub const TEXT_PREFIX: &str = "model.language_model.";
/// The averaged checkpoint's prefix for the tower (`tools/ckpt_average.py:56`, `:168`).
pub const TOWER_PREFIX: &str = "tower.";
/// The averaged checkpoint's prefix for the span pointer head.
pub const SPAN_PREFIX: &str = "span_head.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// `linear_attention`: the gated delta net.
    Gdn,
    /// `full_attention`.
    Attention,
}

/// The shapes the config implies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub layer_kinds: Vec<LayerKind>,
    pub gdn_k_heads: usize,
    pub gdn_v_heads: usize,
    pub gdn_k_dim: usize,
    pub gdn_v_dim: usize,
    pub conv_kernel: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

fn config_err(msg: impl Into<String>) -> Refusal {
    Refusal::new(RefusalKind::Config, msg)
}

fn positive(tc: &Value, key: &str) -> Result<usize> {
    let v = tc
        .get(key)
        .ok_or_else(|| config_err(format!("text_config.{key} is missing")))?
        .as_u64()
        .ok_or_else(|| config_err(format!("text_config.{key} is not a non-negative integer")))?;
    let v = usize::try_from(v).map_err(|_| config_err(format!("text_config.{key} = {v} overflows usize")))?;
    if v == 0 {
        return Err(config_err(format!("text_config.{key} is zero")));
    }
    Ok(v)
}

fn require_bool(v: Option<&Value>, key: &str, want: bool, why: &str) -> Result<()> {
    match v.and_then(Value::as_bool) {
        Some(b) if b == want => Ok(()),
        other => Err(config_err(format!(
            "{key} is {other:?}; the loader's layout needs {want} ({why})"
        ))),
    }
}

fn mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| config_err(format!("{what} overflows usize")))
}

impl Layout {
    /// Parse and validate. The top level may be the conditional-generation config (with a
    /// `text_config`) or a text config itself, as qd-metal accepts (`config.rs:94-102`).
    pub fn from_config_json(text: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(text).map_err(|e| config_err(format!("config.json is not JSON: {e}")))?;
        let tc = root.get("text_config").unwrap_or(&root);
        if !tc.is_object() {
            return Err(config_err("text_config is not an object"));
        }
        let hidden = positive(tc, "hidden_size")?;
        let intermediate = positive(tc, "intermediate_size")?;
        let vocab = positive(tc, "vocab_size")?;
        let n_layers = positive(tc, "num_hidden_layers")?;
        let types = tc
            .get("layer_types")
            .ok_or_else(|| config_err("text_config.layer_types is missing"))?
            .as_array()
            .ok_or_else(|| config_err("text_config.layer_types is not an array"))?;
        if types.len() != n_layers {
            return Err(config_err(format!(
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
                other => Err(config_err(format!(
                    "text_config.layer_types[{i}] is {other:?}, not linear_attention/full_attention"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;

        require_bool(
            root.get("tie_word_embeddings").or_else(|| tc.get("tie_word_embeddings")),
            "tie_word_embeddings",
            true,
            "the head is embed_tokens; there is no lm_head tensor to write",
        )?;
        require_bool(tc.get("attention_bias"), "text_config.attention_bias", false, "no bias tensors")?;
        require_bool(
            tc.get("attn_output_gate"),
            "text_config.attn_output_gate",
            true,
            "q_proj carries the output gate, 2 x num_attention_heads x head_dim rows",
        )?;
        let mlp_only = tc.get("mlp_only_layers").map(|v| v.as_array().map(Vec::len));
        match mlp_only {
            None | Some(Some(0)) => {}
            Some(other) => {
                return Err(config_err(format!(
                    "text_config.mlp_only_layers is {other:?} entries; every layer the loader reads has a mixer"
                )));
            }
        }

        let n_gdn = layer_kinds.iter().filter(|k| **k == LayerKind::Gdn).count();
        let n_attn = layer_kinds.len() - n_gdn;
        // Mixer fields are required only when some layer uses that mixer: the loader reads them
        // for every config, but a layout field nothing is shaped by cannot be wrong.
        let (gdn_k_heads, gdn_v_heads, gdn_k_dim, gdn_v_dim, conv_kernel) = if n_gdn > 0 {
            (
                positive(tc, "linear_num_key_heads")?,
                positive(tc, "linear_num_value_heads")?,
                positive(tc, "linear_key_head_dim")?,
                positive(tc, "linear_value_head_dim")?,
                positive(tc, "linear_conv_kernel_dim")?,
            )
        } else {
            (0, 0, 0, 0, 0)
        };
        let (q_heads, kv_heads, head_dim) = if n_attn > 0 {
            (
                positive(tc, "num_attention_heads")?,
                positive(tc, "num_key_value_heads")?,
                positive(tc, "head_dim")?,
            )
        } else {
            (0, 0, 0)
        };
        let layout = Self {
            hidden,
            intermediate,
            vocab,
            layer_kinds,
            gdn_k_heads,
            gdn_v_heads,
            gdn_k_dim,
            gdn_v_dim,
            conv_kernel,
            q_heads,
            kv_heads,
            head_dim,
        };
        // Every shape is computed once here so an overflow is a config refusal, not a panic.
        layout.text_tensors()?;
        Ok(layout)
    }

    /// Every text tensor the loader reads, **without** [`TEXT_PREFIX`], with its shape.
    ///
    /// The arithmetic is `weights.rs:143-186` and `model.rs:328-341`: a GDN layer's conv runs
    /// over `q | k | v` (`2 * key + value` channels, tessl `GdnProjLayout::conv_dim`), and an
    /// attention layer's `q_proj` holds a query and a gate per head.
    pub fn text_tensors(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        let (h, i) = (self.hidden, self.intermediate);
        let mut out = BTreeMap::new();
        out.insert("embed_tokens.weight".to_string(), vec![self.vocab, h]);
        out.insert("norm.weight".to_string(), vec![h]);
        for (n, kind) in self.layer_kinds.iter().enumerate() {
            let p = format!("layers.{n}.");
            out.insert(format!("{p}input_layernorm.weight"), vec![h]);
            out.insert(format!("{p}post_attention_layernorm.weight"), vec![h]);
            out.insert(format!("{p}mlp.gate_proj.weight"), vec![i, h]);
            out.insert(format!("{p}mlp.up_proj.weight"), vec![i, h]);
            out.insert(format!("{p}mlp.down_proj.weight"), vec![h, i]);
            match kind {
                LayerKind::Gdn => {
                    let key = mul(self.gdn_k_heads, self.gdn_k_dim, "GDN key dim")?;
                    let value = mul(self.gdn_v_heads, self.gdn_v_dim, "GDN value dim")?;
                    let conv = mul(2, key, "GDN conv dim")?
                        .checked_add(value)
                        .ok_or_else(|| config_err("GDN conv dim overflows usize"))?;
                    let vh = self.gdn_v_heads;
                    let m = format!("{p}linear_attn.");
                    out.insert(format!("{m}in_proj_qkv.weight"), vec![conv, h]);
                    out.insert(format!("{m}in_proj_z.weight"), vec![value, h]);
                    out.insert(format!("{m}in_proj_b.weight"), vec![vh, h]);
                    out.insert(format!("{m}in_proj_a.weight"), vec![vh, h]);
                    out.insert(format!("{m}out_proj.weight"), vec![h, value]);
                    out.insert(format!("{m}conv1d.weight"), vec![conv, 1, self.conv_kernel]);
                    out.insert(format!("{m}A_log"), vec![vh]);
                    out.insert(format!("{m}dt_bias"), vec![vh]);
                    out.insert(format!("{m}norm.weight"), vec![self.gdn_v_dim]);
                }
                LayerKind::Attention => {
                    let q = mul(self.q_heads, self.head_dim, "attention width")?;
                    let kv = mul(self.kv_heads, self.head_dim, "attention kv width")?;
                    let m = format!("{p}self_attn.");
                    out.insert(format!("{m}q_proj.weight"), vec![mul(2, q, "q_proj rows")?, h]);
                    out.insert(format!("{m}k_proj.weight"), vec![kv, h]);
                    out.insert(format!("{m}v_proj.weight"), vec![kv, h]);
                    out.insert(format!("{m}o_proj.weight"), vec![h, q]);
                    out.insert(format!("{m}q_norm.weight"), vec![self.head_dim]);
                    out.insert(format!("{m}k_norm.weight"), vec![self.head_dim]);
                }
            }
        }
        Ok(out)
    }

    /// The span pointer head's tensors, as `SpanPointerHead.state_dict()` names them
    /// (`python/qd_train/heads.py:304-314`): two bias-free `[H, H]` projections and two `[H]`
    /// abstain vectors.
    pub fn span_head_tensors(&self) -> BTreeMap<String, Vec<usize>> {
        let h = self.hidden;
        BTreeMap::from([
            ("abstain_end".to_string(), vec![h]),
            ("abstain_start".to_string(), vec![h]),
            ("end_proj.weight".to_string(), vec![h, h]),
            ("start_proj.weight".to_string(), vec![h, h]),
        ])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The real Qwen3.5-2B-Base `config.json`, byte for byte (snapshot `b1485b2f`, sha256
    /// `ed1c1723...`).
    pub(crate) const QWEN35_2B: &str = include_str!("../tests/fixtures/qwen35_2b_base_config.json");

    fn with(edit: impl FnOnce(&mut Value)) -> String {
        let mut v: Value = serde_json::from_str(QWEN35_2B).unwrap();
        edit(&mut v);
        v.to_string()
    }

    #[test]
    fn the_real_2b_config_is_320_tensors_at_the_248320_row_vocab() {
        let l = Layout::from_config_json(QWEN35_2B).unwrap();
        assert_eq!((l.hidden, l.intermediate, l.vocab), (2048, 6144, 248_320));
        let t = l.text_tensors().unwrap();
        // 2 + 24 * 5 + 18 * 9 + 6 * 6, the count qd-metal and qd_train.backbone both measured.
        assert_eq!(t.len(), 320);
        assert_eq!(t["embed_tokens.weight"], [248_320, 2048]);
        // 16 key heads x 128 x 2 + 16 value heads x 128.
        assert_eq!(t["layers.0.linear_attn.in_proj_qkv.weight"], [6144, 2048]);
        assert_eq!(t["layers.0.linear_attn.conv1d.weight"], [6144, 1, 4]);
        assert_eq!(t["layers.3.self_attn.q_proj.weight"], [4096, 2048]);
        assert_eq!(t["layers.3.self_attn.k_proj.weight"], [512, 2048]);
        assert_eq!(t["layers.3.self_attn.o_proj.weight"], [2048, 2048]);
        assert_eq!(l.span_head_tensors()["start_proj.weight"], [2048, 2048]);
    }

    #[test]
    fn refuses_configs_whose_layout_the_loader_does_not_read() {
        let cases: Vec<(String, &str)> = vec![
            (with(|v| v["tie_word_embeddings"] = false.into()), "tie_word_embeddings"),
            (
                with(|v| {
                    v.as_object_mut().unwrap().remove("tie_word_embeddings");
                    v["text_config"]["tie_word_embeddings"] = false.into();
                }),
                "tie_word_embeddings",
            ),
            (with(|v| v["text_config"]["attention_bias"] = true.into()), "attention_bias"),
            (with(|v| v["text_config"]["attn_output_gate"] = false.into()), "attn_output_gate"),
            (with(|v| v["text_config"]["mlp_only_layers"] = serde_json::json!([3])), "mlp_only_layers"),
            (with(|v| v["text_config"]["num_hidden_layers"] = 23.into()), "layer_types has 24"),
            (with(|v| v["text_config"]["layer_types"][0] = "sliding".into()), "layer_types[0]"),
            (with(|v| v["text_config"]["hidden_size"] = 0.into()), "hidden_size is zero"),
            (
                with(|v| {
                    v["text_config"].as_object_mut().unwrap().remove("vocab_size");
                }),
                "vocab_size is missing",
            ),
            ("[]".to_string(), "not an object"),
            ("{".to_string(), "not JSON"),
        ];
        for (json, needle) in cases {
            let err = Layout::from_config_json(&json).unwrap_err();
            assert_eq!(err.kind, RefusalKind::Config);
            assert!(err.detail.contains(needle), "expected {needle:?} in {err:?}");
        }
    }
}
