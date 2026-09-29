//! Checkpoint tensors → the host-side layouts tessl's kernels read. CPU only; the upload is in
//! [`crate::model`].
//!
//! What each weight becomes, from `transformers/models/qwen3_5/modeling_qwen3_5.py` (5.12.1):
//!
//! | checkpoint | becomes | why |
//! | --- | --- | --- |
//! | `input_layernorm`, `post_attention_layernorm` | f32 `1 + w` | `Qwen3_5RMSNorm` is zero-centred (`* (1.0 + weight)`, l.749); `nn::rms_norm_bf16` multiplies by its weight as given |
//! | `self_attn.q_norm`, `k_norm` | f32 `w` | `attn_qk_norm_rope` applies `1 + w` itself |
//! | `linear_attn.norm` | f32 `w` | `Qwen3_5RMSNormGated` multiplies by `weight` plainly (l.199) |
//! | final `norm` | f32 `w` | `score_answer_rows` is called with `w_offset = 1.0` |
//! | `A_log` (F32), `dt_bias` (BF16) | f32 | the GDN kernels read f32 parameters |
//! | `conv1d.weight` `[C, 1, K]` | f32 `[C, K]` | `conv1d_silu`'s weight layout |
//! | every `nn.Linear` `[out, in]` | bf16 `[in, out]` | tessl's GEMM is `x @ W`; `pack_linear_weights_bf16` is the transpose |
//! | `in_proj_{qkv,z,b,a}` | one bf16 `[in, width]` | `GdnProjLayout`: `[q | k | v | z | b | a]` |
//! | `q_proj, k_proj, v_proj` | one bf16 `[in, width]` | `AttnProjLayout`: `[q+gate | k | v]`; `q_proj`'s rows are already `[D q, D gate]` per head (`view(..., -1, head_dim * 2)` then `chunk`, l.683) |

use tessl::qwen35::{pack_linear_weights_bf16, AttnProjLayout, GdnProjLayout};

use crate::config::{LayerKind, ModelConfig};
use crate::error::{MetalError, Result};
use crate::safetensors::{Dtype, SafeTensors};

/// Prefix of the text tower's tensors in the conditional-generation checkpoint. The `visual.` and
/// `mtp.` tensors are never read.
pub const TEXT_PREFIX: &str = "model.language_model.";

/// One tensor's identity for the weight hash: name, then SHA-256 of its raw bytes.
pub type TensorDigest = (String, [u8; 32]);

pub enum HostMixer {
    Gdn {
        /// `[hidden, layout.width()]`
        w_in: Vec<u16>,
        /// `[value_dim, hidden]`
        w_out: Vec<u16>,
        /// `[conv_dim, kernel]`
        conv_w: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm_w: Vec<f32>,
    },
    Attn {
        /// `[hidden, layout.width()]`
        w_in: Vec<u16>,
        /// `[q_heads * head_dim, hidden]`
        w_out: Vec<u16>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

pub struct HostLayer {
    pub index: usize,
    /// `1 + w`
    pub in_norm: Vec<f32>,
    /// `1 + w`
    pub post_norm: Vec<f32>,
    pub mixer: HostMixer,
    /// `[hidden, intermediate]`
    pub w_gate: Vec<u16>,
    /// `[hidden, intermediate]`
    pub w_up: Vec<u16>,
    /// `[intermediate, hidden]`
    pub w_down: Vec<u16>,
    pub digests: Vec<TensorDigest>,
}

/// Reads tensors and records each one's digest.
struct Reader<'a> {
    st: &'a SafeTensors,
    digests: Vec<TensorDigest>,
}

impl Reader<'_> {
    fn raw(&mut self, name: &str, shape: &[usize], dtypes: &[Dtype]) -> Result<(Dtype, Vec<u8>)> {
        let full = format!("{TEXT_PREFIX}{name}");
        let info = self.st.expect(&full, shape, dtypes)?.clone();
        let mut bytes = vec![0u8; info.nbytes()];
        self.st.read_into(&full, &mut bytes)?;
        self.digests.push((full, qd_runtime::sha256(&bytes)));
        Ok((info.dtype, bytes))
    }

    fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let (_, bytes) = self.raw(name, shape, &[Dtype::Bf16])?;
        Ok(crate::safetensors::u16_words(&bytes))
    }

    fn f32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (dtype, bytes) = self.raw(name, shape, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?;
        Ok(crate::safetensors::widen(dtype, &bytes))
    }
}

/// `1 + w`: the zero-centred RMSNorm's weight in the form a plain multiply expects.
pub fn offset_norm(w: &[f32]) -> Vec<f32> {
    w.iter().map(|v| 1.0 + v).collect()
}

/// `[out, in]` → `[in, out]`.
pub fn transpose_linear(w: &[u16], out_features: usize, in_features: usize) -> Result<Vec<u16>> {
    pack_linear_weights_bf16(&[w], &[out_features], in_features).map_err(MetalError::Weights)
}

/// `[in_proj_qkv, in_proj_z, in_proj_b, in_proj_a]` → one `[hidden, width]` operand whose columns
/// are laid out as `layout` says.
pub fn pack_gdn_in(
    layout: &GdnProjLayout,
    hidden: usize,
    qkv: &[u16],
    z: &[u16],
    b: &[u16],
    a: &[u16],
) -> Result<Vec<u16>> {
    pack_linear_weights_bf16(&[qkv, z, b, a], &layout.part_widths(), hidden)
        .map_err(MetalError::Weights)
}

/// `[q_proj, k_proj, v_proj]` → one `[hidden, width]` operand laid out as `layout` says.
pub fn pack_attn_in(
    layout: &AttnProjLayout,
    hidden: usize,
    q: &[u16],
    k: &[u16],
    v: &[u16],
) -> Result<Vec<u16>> {
    pack_linear_weights_bf16(&[q, k, v], &layout.part_widths(), hidden).map_err(MetalError::Weights)
}

/// Read and lay out one decoder layer.
pub fn prepare_layer(st: &SafeTensors, cfg: &ModelConfig, index: usize) -> Result<HostLayer> {
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    let mut r = Reader {
        st,
        digests: Vec::new(),
    };
    let p = format!("layers.{index}.");
    let kind = *cfg
        .layer_kinds
        .get(index)
        .ok_or_else(|| MetalError::Weights(format!("layer {index} is past the config's layers")))?;
    let in_norm = offset_norm(&r.f32(&format!("{p}input_layernorm.weight"), &[h])?);
    let post_norm = offset_norm(&r.f32(&format!("{p}post_attention_layernorm.weight"), &[h])?);
    let mixer = match kind {
        LayerKind::Gdn => {
            let l = cfg.gdn_layout()?;
            let [conv_dim, value_dim, vh, _] = l.part_widths();
            let m = format!("{p}linear_attn.");
            let qkv = r.bf16(&format!("{m}in_proj_qkv.weight"), &[conv_dim, h])?;
            let z = r.bf16(&format!("{m}in_proj_z.weight"), &[value_dim, h])?;
            let b = r.bf16(&format!("{m}in_proj_b.weight"), &[vh, h])?;
            let a = r.bf16(&format!("{m}in_proj_a.weight"), &[vh, h])?;
            let out = r.bf16(&format!("{m}out_proj.weight"), &[h, value_dim])?;
            let kw = cfg.conv_kernel as usize;
            HostMixer::Gdn {
                w_in: pack_gdn_in(&l, h, &qkv, &z, &b, &a)?,
                w_out: transpose_linear(&out, h, value_dim)?,
                // [C, 1, K] row-major is [C, K] row-major.
                conv_w: r.f32(&format!("{m}conv1d.weight"), &[conv_dim, 1, kw])?,
                a_log: r.f32(&format!("{m}A_log"), &[vh])?,
                dt_bias: r.f32(&format!("{m}dt_bias"), &[vh])?,
                norm_w: r.f32(&format!("{m}norm.weight"), &[cfg.gdn_v_dim as usize])?,
            }
        }
        LayerKind::Attention => {
            let l = cfg.attn_layout()?;
            let [qg, kd, vd] = l.part_widths();
            let m = format!("{p}self_attn.");
            let q = r.bf16(&format!("{m}q_proj.weight"), &[qg, h])?;
            let k = r.bf16(&format!("{m}k_proj.weight"), &[kd, h])?;
            let v = r.bf16(&format!("{m}v_proj.weight"), &[vd, h])?;
            let attn_out = (cfg.q_heads * cfg.head_dim) as usize;
            let o = r.bf16(&format!("{m}o_proj.weight"), &[h, attn_out])?;
            let hd = cfg.head_dim as usize;
            HostMixer::Attn {
                w_in: pack_attn_in(&l, h, &q, &k, &v)?,
                w_out: transpose_linear(&o, h, attn_out)?,
                q_norm: r.f32(&format!("{m}q_norm.weight"), &[hd])?,
                k_norm: r.f32(&format!("{m}k_norm.weight"), &[hd])?,
            }
        }
    };
    let gate = r.bf16(&format!("{p}mlp.gate_proj.weight"), &[inter, h])?;
    let up = r.bf16(&format!("{p}mlp.up_proj.weight"), &[inter, h])?;
    let down = r.bf16(&format!("{p}mlp.down_proj.weight"), &[h, inter])?;
    Ok(HostLayer {
        index,
        in_norm,
        post_norm,
        mixer,
        w_gate: transpose_linear(&gate, inter, h)?,
        w_up: transpose_linear(&up, inter, h)?,
        w_down: transpose_linear(&down, h, inter)?,
        digests: r.digests,
    })
}

/// Every text tensor the forward pass reads, so a checkpoint with extra or missing text tensors
/// is refused rather than half-loaded. Returns the expected names.
pub fn expected_text_tensors(cfg: &ModelConfig) -> Vec<String> {
    let mut names = vec![
        format!("{TEXT_PREFIX}embed_tokens.weight"),
        format!("{TEXT_PREFIX}norm.weight"),
    ];
    for (i, kind) in cfg.layer_kinds.iter().enumerate() {
        let p = format!("{TEXT_PREFIX}layers.{i}.");
        for n in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("{p}{n}"));
        }
        let mixer: &[&str] = match kind {
            LayerKind::Gdn => &[
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.out_proj.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.norm.weight",
            ],
            LayerKind::Attention => &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ],
        };
        names.extend(mixer.iter().map(|n| format!("{p}{n}")));
    }
    names.sort();
    names
}

/// The checkpoint's text tensors must be exactly [`expected_text_tensors`].
pub fn check_text_tensor_set(st: &SafeTensors, cfg: &ModelConfig) -> Result<()> {
    let want = expected_text_tensors(cfg);
    let have: Vec<String> = st
        .names()
        .filter(|n| n.starts_with(TEXT_PREFIX))
        .map(str::to_string)
        .collect();
    let missing: Vec<&String> = want.iter().filter(|n| !have.contains(n)).collect();
    let extra: Vec<&String> = have.iter().filter(|n| !want.contains(n)).collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(MetalError::Weights(format!(
            "text tensors differ from what the forward pass reads: missing {missing:?}, unread {extra:?}"
        )));
    }
    Ok(())
}

/// The weight hash: SHA-256 over `name \0 sha256(bytes)` for every tensor read, in name order.
pub fn weight_hash(digests: &mut [TensorDigest]) -> String {
    digests.sort_by(|a, b| a.0.cmp(&b.0));
    let mut buf = Vec::with_capacity(digests.len() * 96);
    for (name, d) in digests.iter() {
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(d);
    }
    qd_runtime::hex(&qd_runtime::sha256(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u16> {
        tessl::tensor::f32_slice_to_bf16(v)
    }

    #[test]
    fn transpose_puts_out_features_on_columns() {
        // [out=2, in=3]
        let w = bits(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = transpose_linear(&w, 2, 3).unwrap();
        assert_eq!(t, bits(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
        assert!(transpose_linear(&w, 3, 3).is_err());
    }

    /// Row `k` of the packed operand holds input feature `k` of every projection, at the column
    /// the layout names: the property every kernel reading `Cols` windows relies on.
    #[test]
    fn gdn_packing_agrees_with_the_layout_getters() {
        let l = GdnProjLayout::new(1, 1, 32).unwrap();
        let hidden = 3;
        let [c, v, b, a] = l.part_widths();
        // Raw 16-bit tags, not floats: bf16 cannot hold distinct integers this large.
        let val = |part: usize, row: usize, k: usize| u16::try_from((part << 12) | (row << 2) | k).unwrap();
        let mk = |part: usize, rows: usize| {
            (0..rows * hidden).map(|i| val(part, i / hidden, i % hidden)).collect::<Vec<u16>>()
        };
        let packed = pack_gdn_in(&l, hidden, &mk(0, c), &mk(1, v), &mk(2, b), &mk(3, a)).unwrap();
        let w = l.width() as usize;
        assert_eq!(packed.len(), hidden * w);
        let at = |k: usize, col: u32| packed[k * w + col as usize];
        for k in 0..hidden {
            assert_eq!(at(k, 0), val(0, 0, k), "q");
            assert_eq!(at(k, l.key_dim()), val(0, l.key_dim() as usize, k), "k");
            assert_eq!(at(k, 2 * l.key_dim()), val(0, 2 * l.key_dim() as usize, k), "v");
            assert_eq!(at(k, l.z_off() + 5), val(1, 5, k), "z");
            assert_eq!(at(k, l.b_off()), val(2, 0, k), "b");
            assert_eq!(at(k, l.a_off()), val(3, 0, k), "a");
        }
        assert!(pack_gdn_in(&l, hidden, &mk(0, c), &mk(1, v), &mk(3, a), &mk(2, b)[..1]).is_err());
    }

    #[test]
    fn attn_packing_agrees_with_the_layout_getters() {
        let l = AttnProjLayout::new(2, 1, 4).unwrap();
        let hidden = 2;
        let [qg, kd, vd] = l.part_widths();
        let val = |part: usize, row: usize, k: usize| u16::try_from((part << 12) | (row << 2) | k).unwrap();
        let mk = |part: usize, rows: usize| {
            (0..rows * hidden).map(|i| val(part, i / hidden, i % hidden)).collect::<Vec<u16>>()
        };
        let packed = pack_attn_in(&l, hidden, &mk(0, qg), &mk(1, kd), &mk(2, vd)).unwrap();
        let w = l.width() as usize;
        let at = |k: usize, col: u32| packed[k * w + col as usize];
        for k in 0..hidden {
            // head 1's gate is q_proj row 1*2*4 + 4 = 12.
            assert_eq!(at(k, 12), val(0, 12, k));
            assert_eq!(at(k, l.k_off() + 3), val(1, 3, k));
            assert_eq!(at(k, l.v_off()), val(2, 0, k));
        }
    }

    #[test]
    fn norms_are_offset_by_one() {
        assert_eq!(offset_norm(&[0.0, -0.5, 2.0]), [1.0, 0.5, 3.0]);
    }

    #[test]
    fn weight_hash_is_order_independent_and_name_sensitive() {
        let mut a = vec![("x".to_string(), [1u8; 32]), ("y".to_string(), [2u8; 32])];
        let mut b = vec![("y".to_string(), [2u8; 32]), ("x".to_string(), [1u8; 32])];
        assert_eq!(weight_hash(&mut a), weight_hash(&mut b));
        let mut c = vec![("x".to_string(), [2u8; 32]), ("y".to_string(), [1u8; 32])];
        assert_ne!(weight_hash(&mut a), weight_hash(&mut c));
    }

    #[test]
    fn the_2b_reads_exactly_its_320_text_tensors() {
        let cfg = ModelConfig::from_json(crate::config::tests_support::QWEN35_2B).unwrap();
        // 2 + 24 * 5 + 18 * 9 + 6 * 6 = 320, the count qd_train.backbone measured.
        assert_eq!(expected_text_tensors(&cfg).len(), 320);
    }
}
