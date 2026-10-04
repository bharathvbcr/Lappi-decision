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

/// `qd-metal-bench --decision`'s prompts: every T is met from below, the context is as long as
/// fits, both passes tokenize after the prefix, and the permuted pass differs. CPU only.
#[test]
#[ignore = "needs the model snapshot"]
fn snapshot_decision_prompts_fill_each_t_from_below() {
    let tok = QwenTokenizer::load(&snapshot().join("tokenizer.json")).unwrap();
    for t in qd_metal::decision::DEFAULT_T {
        let p = qd_metal::decision::build_prompt(&tok, t, 4).unwrap();
        assert!(p.prefix.len() <= t, "T={t}: prefix {} tokens", p.prefix.len());
        // One more source line must not fit, or the context is shorter than it needs to be.
        assert!(p.prefix.len() + 64 > t, "T={t}: prefix {} tokens is far below the target", p.prefix.len());
        assert_eq!(p.rows, 5);
        assert_ne!(p.passes[0], p.passes[1]);
        assert!(p.passes.iter().all(|s| !s.is_empty() && s.len() < 64), "{:?}", p.passes.iter().map(Vec::len).collect::<Vec<_>>());
        println!("T={t}: prefix {} tokens ({} context lines), passes {}+{} tokens", p.prefix.len(), p.context_lines, p.passes[0].len(), p.passes[1].len());
    }
    assert!(qd_metal::decision::build_prompt(&tok, 16, 4).is_err(), "a T below the fixed text is refused");
}

/// The frozen context feeds the ids rows 147da0cc and 28505f4c fed: their `data_snapshot_hash`
/// and context lines, at their five Ts. This is what makes a later row comparable to them; it
/// failed while the context was read from the live model.rs. The ids depend on the tokenizer, so
/// it must be the one those rows recorded; on another the test refuses rather than mismatch.
#[test]
#[ignore = "needs the model snapshot"]
fn snapshot_decision_inputs_reproduce_rows_1_and_2() {
    const ROWS_TOKENIZER: &str = "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927";
    const ROWS_INPUTS: &str = "3d38c835cfe2c4b8704d9208e16f5a4597bcd3cc0db2509ae2e1a80467c3b16f";
    let tok = QwenTokenizer::load(&snapshot().join("tokenizer.json")).unwrap();
    assert_eq!(tok.hash(), ROWS_TOKENIZER, "not rows 1-2's tokenizer");
    let ts = [131, 409, 770, 2048, 8192];
    let prompts: Vec<_> = ts
        .iter()
        .map(|&t| qd_metal::decision::build_prompt(&tok, t, 4).unwrap())
        .collect();
    let lines: Vec<usize> = prompts.iter().map(|p| p.context_lines).collect();
    let prefix: Vec<usize> = prompts.iter().map(|p| p.prefix.len()).collect();
    println!("context lines {lines:?}, prefix tokens {prefix:?}");
    assert_eq!(lines, [3, 18, 38, 145, 690]);
    assert_eq!(prefix, [115, 399, 762, 2033, 8185]);
    assert_eq!(qd_metal::decision::inputs_digest(&prompts), ROWS_INPUTS);
}
