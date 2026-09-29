//! CPU checks against the real Qwen3.5-2B-Base snapshot: the header, the tensor set, the packing
//! shapes and the tokenizer. No GPU. `#[ignore]`d because they need the 4.5 GB snapshot in the
//! Hugging Face cache, which a checkout elsewhere does not have:
//!
//! ```text
//! cargo test --release -p qd-metal --test snapshot_cpu -- --ignored
//! ```

use std::time::Instant;

use qd_metal::config::{LayerKind, ModelConfig};
use qd_metal::safetensors::{Dtype, SafeTensors};
use qd_metal::tokenizer::QwenTokenizer;
use qd_metal::weights::{self, HostMixer};

fn snapshot() -> std::path::PathBuf {
    qd_metal::config::resolve_snapshot(None).expect("the Qwen3.5-2B-Base snapshot")
}

#[test]
#[ignore = "needs the model snapshot"]
fn snapshot_header_and_tensor_set_match_the_config() {
    let snap = snapshot();
    let cfg = ModelConfig::load(&snap).unwrap();
    assert_eq!((cfg.n_gdn(), cfg.n_attention(), cfg.rotary_dim), (18, 6, 64));
    let st = SafeTensors::open(&qd_metal::model::weights_file(&snap).unwrap()).unwrap();
    weights::check_text_tensor_set(&st, &cfg).unwrap();
    // The dtypes the loader converts: A_log and the gated norm are F32, dt_bias is BF16.
    let p = weights::TEXT_PREFIX;
    assert_eq!(st.info(&format!("{p}layers.0.linear_attn.A_log")).unwrap().dtype, Dtype::F32);
    assert_eq!(st.info(&format!("{p}layers.0.linear_attn.norm.weight")).unwrap().dtype, Dtype::F32);
    assert_eq!(st.info(&format!("{p}layers.0.linear_attn.dt_bias")).unwrap().dtype, Dtype::Bf16);
    assert_eq!(
        st.info(&format!("{p}embed_tokens.weight")).unwrap().shape,
        [cfg.vocab, cfg.hidden]
    );
}

#[test]
#[ignore = "needs the model snapshot"]
fn snapshot_layers_pack_to_the_kernel_layouts() {
    let snap = snapshot();
    let cfg = ModelConfig::load(&snap).unwrap();
    let st = SafeTensors::open(&qd_metal::model::weights_file(&snap).unwrap()).unwrap();
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    for (i, want) in [(0usize, LayerKind::Gdn), (3, LayerKind::Attention)] {
        let t0 = Instant::now();
        let l = weights::prepare_layer(&st, &cfg, i).unwrap();
        println!("layer {i} prepared in {:.2} s", t0.elapsed().as_secs_f64());
        assert_eq!(l.w_gate.len(), h * inter);
        assert_eq!(l.w_down.len(), inter * h);
        assert!(l.in_norm.iter().all(|w| w.is_finite()));
        match (&l.mixer, want) {
            (HostMixer::Gdn { w_in, w_out, conv_w, a_log, dt_bias, norm_w }, LayerKind::Gdn) => {
                let lay = cfg.gdn_layout().unwrap();
                assert_eq!(w_in.len(), h * lay.width() as usize);
                assert_eq!(w_out.len(), lay.value_dim() as usize * h);
                assert_eq!(conv_w.len(), lay.conv_dim() as usize * cfg.conv_kernel as usize);
                assert_eq!((a_log.len(), dt_bias.len(), norm_w.len()), (16, 16, 128));
                // Values read with the Python `safetensors` package from this snapshot
                // (2026-09-29): layer 0 A_log[0] = -0.25976563 (F32), dt_bias[3] = 9.5 (BF16).
                // A swapped pair or a misread dtype shows here.
                assert_eq!(a_log[0], -0.259_765_63_f32);
                assert_eq!(a_log[8], -5.906_25_f32);
                assert_eq!(dt_bias[3], 9.5);
                assert_eq!(dt_bias[8], -12.3125);
            }
            (HostMixer::Attn { w_in, w_out, q_norm, k_norm }, LayerKind::Attention) => {
                let lay = cfg.attn_layout().unwrap();
                assert_eq!(w_in.len(), h * lay.width() as usize);
                assert_eq!(w_out.len(), (cfg.q_heads * cfg.head_dim) as usize * h);
                assert_eq!((q_norm.len(), k_norm.len()), (256, 256));
            }
            _ => panic!("layer {i} has the wrong mixer"),
        }
    }
}

#[test]
#[ignore = "needs the model snapshot"]
fn snapshot_tokenizer_letters_are_single_tokens() {
    let tok = QwenTokenizer::load(&snapshot().join("tokenizer.json")).unwrap();
    // Measured with the Python tokenizers 0.22.2 on this snapshot, 2026-09-29:
    // A..P = 32..47, Z = 57.
    let mut want: Vec<u32> = (32..48).collect();
    want.push(57);
    assert_eq!(tok.letter_ids()[..], want[..]);
    // The answer marker is not a special token in the base tokenizer: several ordinary tokens.
    assert!(tok.encode("<|qd_answer|>").unwrap().len() > 1);
}
