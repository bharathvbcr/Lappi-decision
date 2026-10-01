//! The export: f32 masters -> `tower.*` bf16 (RNE) + `span_head.*` f32 + a manifest, read back
//! through qd-export's own safetensors reader, and every refusal.

use std::collections::BTreeMap;
use std::path::PathBuf;

use qd_export::bf16::f32_to_bf16_rne;
use qd_export::layout::{LayerKind, Layout};
use qd_export::safetensors::{Dtype, SafeTensorsFile};
use qd_train::export::{export, manifest_path, ExportError, ManifestInfo, NamedTensor, FROM_METAL_MASTERS};
use qd_train::pyjson::{float, obj};
use qd_train::step::ParamSpec;

fn layout() -> Layout {
    Layout {
        hidden: 4,
        intermediate: 6,
        vocab: 8,
        layer_kinds: vec![LayerKind::Gdn, LayerKind::Attention],
        gdn_k_heads: 1,
        gdn_v_heads: 1,
        gdn_k_dim: 2,
        gdn_v_dim: 2,
        conv_kernel: 4,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 2,
    }
}

fn values(map: &BTreeMap<String, Vec<usize>>, salt: f32) -> Vec<(ParamSpec, Vec<f32>)> {
    map.iter()
        .enumerate()
        .map(|(k, (name, shape))| {
            let n: usize = shape.iter().product();
            // Values with low bits set, so the bf16 rounding is exercised, plus one exact tie.
            let v = (0..n)
                .map(|i| ((k * 31 + i) as f32 * 0.123_456_7 + salt).sin() * 3.0 + f32::from_bits(0x3F80_8000) - 1.0)
                .collect();
            (ParamSpec::new(name.clone(), shape), v)
        })
        .collect()
}

fn named(v: &[(ParamSpec, Vec<f32>)]) -> Vec<NamedTensor<'_>> {
    v.iter()
        .map(|(s, x)| NamedTensor {
            spec: s.clone(),
            values: x,
        })
        .collect()
}

fn info() -> ManifestInfo {
    ManifestInfo {
        optimizer_step: 20,
        seed: 0,
        schedule: obj([("peak_lr", float(1e-5).unwrap())]).unwrap(),
        ft_row_id: Some("00000000-0000-4000-8000-000000000000".into()),
        provider: "toy".into(),
        operands: "exact_f32".into(),
        recipe_hash: "ab".repeat(32),
        loss_log_digest: "cd".repeat(32),
        consumed_digest: "ef".repeat(32),
    }
}

fn out(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qd-train-export-{tag}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d.join("weights.safetensors")
}

#[test]
fn the_export_is_bf16_tower_f32_head_and_a_manifest_that_names_it() {
    let l = layout();
    let tower = values(&l.text_tensors().unwrap(), 0.0);
    let head = values(&l.span_head_tensors(), 1.0);
    let path = out("ok");
    let s = export(&path, &named(&tower), &named(&head), Some(&l), &info()).unwrap();
    assert_eq!(s.n_tensors, tower.len() + head.len());

    let f = SafeTensorsFile::open(&path).unwrap();
    assert_eq!(qd_train_hex(&f.sha256().unwrap()), s.safetensors_sha256);
    for (spec, v) in &tower {
        let info = &f.tensors()[&format!("tower.{}", spec.name)];
        assert_eq!(info.dtype, Dtype::Bf16);
        assert_eq!(info.shape, spec.shape);
        let mut bytes = vec![0u8; v.len() * 2];
        f.read_tensor_range(info, 0, &mut bytes).unwrap();
        let got: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
        let want: Vec<u16> = v.iter().map(|x| f32_to_bf16_rne(*x).unwrap()).collect();
        assert_eq!(got, want, "{}", spec.name);
    }
    for (spec, v) in &head {
        let info = &f.tensors()[&format!("span_head.{}", spec.name)];
        assert_eq!(info.dtype, Dtype::F32);
        let mut bytes = vec![0u8; v.len() * 4];
        f.read_tensor_range(info, 0, &mut bytes).unwrap();
        let got: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        assert_eq!(&got, v, "{}", spec.name);
    }
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(manifest_path(&path)).unwrap()).unwrap();
    assert_eq!(m["from"], FROM_METAL_MASTERS);
    assert_ne!(m["from"], "tower", "never mistaken for an average");
    assert_ne!(m["from"], "masters");
    assert_eq!(m["resumable"], false);
    assert_eq!(m["optimizer_step"], 20);
    assert_eq!(m["n_tensors"], s.n_tensors);
    assert_eq!(m["safetensors_sha256"], s.safetensors_sha256);
    assert_eq!(m["ft_row_ids"][0], "00000000-0000-4000-8000-000000000000");
    assert_eq!(m["tensor_sources"].as_object().unwrap().len(), s.n_tensors);

    // A second export onto the same path is refused, and leaves the first intact.
    let again = export(&path, &named(&tower), &named(&head), Some(&l), &info());
    assert!(matches!(again, Err(ExportError::Refused(_))));
    assert_eq!(qd_train_hex(&SafeTensorsFile::open(&path).unwrap().sha256().unwrap()), s.safetensors_sha256);
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn a_tensor_set_the_loader_would_not_read_is_refused_and_nothing_is_written() {
    let l = layout();
    let tower = values(&l.text_tensors().unwrap(), 0.0);
    let head = values(&l.span_head_tensors(), 1.0);

    let path = out("missing");
    let short = &tower[1..];
    assert!(matches!(export(&path, &named(short), &named(&head), Some(&l), &info()), Err(ExportError::Refused(m)) if m.contains("missing")));
    assert!(!path.exists());

    let mut extra = tower.clone();
    extra.push((ParamSpec::new("lm_head.weight", &[8, 4]), vec![0.0; 32]));
    assert!(matches!(export(&path, &named(&extra), &named(&head), Some(&l), &info()), Err(ExportError::Refused(m)) if m.contains("extra")));

    let mut bent = tower.clone();
    bent[0].0.shape = vec![bent[0].1.len()];
    assert!(matches!(export(&path, &named(&bent), &named(&head), Some(&l), &info()), Err(ExportError::Refused(_))));

    let mut miscount = tower.clone();
    miscount[0].1.pop();
    assert!(export(&path, &named(&miscount), &named(&head), None, &info()).is_err());

    let mut nan = tower.clone();
    nan[3].1[1] = f32::NAN;
    assert!(matches!(export(&path, &named(&nan), &named(&head), Some(&l), &info()), Err(ExportError::NonFinite(_))));
    let mut huge = tower.clone();
    huge[3].1[0] = f32::MAX;
    assert!(matches!(export(&path, &named(&huge), &named(&head), Some(&l), &info()), Err(ExportError::NonFinite(_))), "rounds to inf in bf16");
    let mut bad_head = head.clone();
    bad_head[0].1[0] = f32::INFINITY;
    assert!(matches!(export(&path, &named(&tower), &named(&bad_head), Some(&l), &info()), Err(ExportError::NonFinite(_))));
    assert!(!path.exists() && !manifest_path(&path).exists(), "no refusal leaves a file behind");
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn the_real_2b_layout_names_320_tower_tensors() {
    let cfg = include_str!("../../qd-export/tests/fixtures/qwen35_2b_base_config.json");
    let l = Layout::from_config_json(cfg).unwrap();
    assert_eq!(l.text_tensors().unwrap().len(), 320);
    assert_eq!(l.span_head_tensors().len(), 4);
}

fn qd_train_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
