//! The export: f32 masters -> `tower.*` bf16 (RNE) + `span_head.*` f32 + a manifest in the
//! scorer's Metal-export contract (`real_ft_run._read_metal_manifest`), read back through
//! qd-export's own safetensors reader, and every refusal.

use std::collections::BTreeMap;
use std::path::PathBuf;

use qd_export::bf16::f32_to_bf16_rne;
use qd_export::layout::{LayerKind, Layout};
use qd_export::safetensors::{Dtype, SafeTensorsFile};
use qd_train::export::{
    export, export_file_name, manifest_path, ExportError, ManifestInfo, NamedTensor, Precision, EXPORT_SOURCE,
    MASTERS_SOURCE, METAL_DEVICE, METAL_TRAINER,
};
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

const ROW: &str = "3e7a1000-aaaa-4bbb-8ccc-00000000000a";

fn info() -> ManifestInfo {
    ManifestInfo {
        trainer: METAL_TRAINER.into(),
        device: METAL_DEVICE.into(),
        seed: 0,
        optimizer_step: 20,
        ft_row_id: ROW.into(),
        vocab_size: 8,
        span_weight: 1.0,
        schedule: obj([("peak_lr", float(1e-5).unwrap())]).unwrap(),
        provider: "toy".into(),
        operands: "exact_f32".into(),
        recipe_hash: "ab".repeat(32),
        loss_log_digest: "cd".repeat(32),
        consumed_digest: "ef".repeat(32),
        head_init_digest: Some("01".repeat(32)),
        prompt_format: 1,
    }
}

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qd-train-export-{tag}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn the_export_is_the_scorers_metal_artifact_bf16_tower_f32_head_and_its_manifest() {
    let l = layout();
    let tower = values(&l.text_tensors().unwrap(), 0.0);
    let head = values(&l.span_head_tensors(), 1.0);
    let d = dir("ok");
    let s = export(&d, &named(&tower), &named(&head), Some(&l), &info(), Precision::Bf16).unwrap();
    assert_eq!(s.weights, d.join("epoch-seed0-metal.safetensors"), "the name the scorer routes by");
    assert_eq!(s.n_tensors, tower.len() + head.len());

    let f = SafeTensorsFile::open(&s.weights).unwrap();
    assert_eq!(sha256_hex(&std::fs::read(&s.weights).unwrap()), s.safetensors_sha256);
    assert_eq!(f.tensors().len(), s.n_tensors, "only tower.* and span_head.*");
    for (spec, v) in &tower {
        let t = &f.tensors()[&format!("tower.{}", spec.name)];
        assert_eq!(t.dtype, Dtype::Bf16);
        assert_eq!(t.shape, spec.shape);
        let mut bytes = vec![0u8; v.len() * 2];
        f.read_tensor_range(t, 0, &mut bytes).unwrap();
        let got: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
        let want: Vec<u16> = v.iter().map(|x| f32_to_bf16_rne(*x).unwrap()).collect();
        assert_eq!(got, want, "{}", spec.name);
    }
    for (spec, v) in &head {
        let t = &f.tensors()[&format!("span_head.{}", spec.name)];
        assert_eq!(t.dtype, Dtype::F32);
        let mut bytes = vec![0u8; v.len() * 4];
        f.read_tensor_range(t, 0, &mut bytes).unwrap();
        let got: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        assert_eq!(&got, v, "{}", spec.name);
    }

    // `METAL_MANIFEST_FIELDS`, each of its JSON type (ints are ints, never bools).
    let raw = std::fs::read(manifest_path(&s.weights)).unwrap();
    assert_eq!(sha256_hex(&raw), s.manifest_sha256, "what the scorer records as trained_by.manifest_sha256");
    let m: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(m["from"], EXPORT_SOURCE);
    assert_eq!(m["from"], "qd-train-export", "the scorer's literal");
    assert_eq!(m["trainer"], "qd-train-metal");
    assert_eq!(m["device"], "metal");
    for k in ["seed", "optimizer_step", "vocab_size", "n_tensors"] {
        assert!(m[k].is_u64(), "{k} is an int: {}", m[k]);
    }
    assert_eq!(m["seed"], 0);
    assert_eq!(m["optimizer_step"], 20);
    assert_eq!(m["vocab_size"], 8);
    assert_eq!(m["n_tensors"], s.n_tensors);
    assert!(m["span_weight"].is_f64() && m["span_weight"] == 1.0);
    assert_eq!(m["ft_row_id"], ROW);
    assert_eq!(m["safetensors_sha256"], s.safetensors_sha256);
    assert_eq!(m["head_init_digest"], "01".repeat(32));
    assert_eq!(m["tensor_sources"].as_object().unwrap().len(), s.n_tensors);

    // A second export onto the same name is refused, and leaves the first intact.
    let again = export(&d, &named(&tower), &named(&head), Some(&l), &info(), Precision::Bf16);
    assert!(matches!(again, Err(ExportError::Refused(_))));
    assert_eq!(sha256_hex(&std::fs::read(&s.weights).unwrap()), s.safetensors_sha256);

    // The masters export sits beside it, f32 throughout, under a name and `from` the scorer refuses.
    let m2 = export(&d, &named(&tower), &named(&head), Some(&l), &info(), Precision::F32Masters).unwrap();
    assert_eq!(m2.weights, d.join("epoch-seed0-metal-f32-masters.safetensors"));
    let f2 = SafeTensorsFile::open(&m2.weights).unwrap();
    assert!(f2.tensors().values().all(|t| t.dtype == Dtype::F32));
    let mm: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest_path(&m2.weights)).unwrap()).unwrap();
    assert_eq!(mm["from"], MASTERS_SOURCE);
    assert_ne!(mm["from"], EXPORT_SOURCE);
    std::fs::remove_dir_all(&d).unwrap();
}

/// The keys the manifest carried before `prompt_format` existed: what a format-1 export must
/// still write, key for key, so its bytes and `manifest_sha256` do not move.
const V4_MANIFEST_KEYS: [&str; 22] = [
    "consumed_digest", "device", "from", "ft_row_id", "head_init_digest", "loss_log_digest", "method",
    "n_tensors", "operands", "optimizer_step", "provider", "recipe_hash", "resumable", "safetensors_sha256",
    "schedule", "seed", "span_weight", "tensor_sources", "tool", "trainer", "vocab_size", "why_not_resumable",
];

/// qd-export stamps the release's `expected_identity.prompt_format` from this manifest's
/// top-level `prompt_format` and reads an absent key as 1. So a format-2 run's export must say 2
/// (as a JSON integer), and a format-1 run's must stay what it was.
#[test]
fn the_manifest_states_a_prompt_format_other_than_1_and_a_format_1_manifest_is_unchanged() {
    let l = layout();
    let tower = values(&l.text_tensors().unwrap(), 0.0);
    let head = values(&l.span_head_tensors(), 1.0);
    let read = |p: &std::path::Path| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(manifest_path(p)).unwrap()).unwrap()
    };

    let d = dir("prompt-format-2");
    let v5 = ManifestInfo { prompt_format: 2, ..info() };
    for precision in [Precision::Bf16, Precision::F32Masters] {
        let s = export(&d, &named(&tower), &named(&head), Some(&l), &v5, precision).unwrap();
        let m = read(&s.weights);
        assert!(m["prompt_format"].is_u64() && m["prompt_format"] == 2, "{precision:?}: {}", m["prompt_format"]);
    }
    std::fs::remove_dir_all(&d).unwrap();

    let d = dir("prompt-format-1");
    let s = export(&d, &named(&tower), &named(&head), Some(&l), &info(), Precision::Bf16).unwrap();
    let m = read(&s.weights);
    let keys: Vec<&str> = m.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, V4_MANIFEST_KEYS, "format 1 writes no prompt_format and nothing else new");
    std::fs::remove_dir_all(&d).unwrap();

    let d = dir("prompt-format-0");
    let r = export(&d, &named(&tower), &named(&head), Some(&l), &ManifestInfo { prompt_format: 0, ..info() }, Precision::Bf16);
    assert!(matches!(r, Err(ExportError::Refused(ref m)) if m.contains("prompt_format")), "{r:?}");
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "the refusal leaves nothing behind");
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn the_name_is_the_one_the_scorer_splits() {
    assert_eq!(export_file_name(2, "metal", Precision::Bf16).unwrap(), "epoch-seed2-metal.safetensors");
    assert!(export_file_name(0, "metal-x", Precision::Bf16).is_err(), "a '-' in the device breaks rsplit('-', 2)");
    assert!(export_file_name(0, "", Precision::Bf16).is_err());
    assert!(export_file_name(0, "Metal", Precision::Bf16).is_err());
    assert_eq!(export_file_name(0, "cpu", Precision::Bf16).unwrap(), "epoch-seed0-cpu.safetensors");
}

#[test]
fn a_manifest_the_scorer_would_refuse_is_refused_here_and_nothing_is_written() {
    let l = layout();
    let tower = values(&l.text_tensors().unwrap(), 0.0);
    let head = values(&l.span_head_tensors(), 1.0);
    let d = dir("bad-info");
    let cases: Vec<(&str, ManifestInfo)> = vec![
        ("ft_row_id not a uuid", ManifestInfo { ft_row_id: "3e7a1000".into(), ..info() }),
        ("vocab_size not the tower's", ManifestInfo { vocab_size: 9, ..info() }),
        ("span_weight 0", ManifestInfo { span_weight: 0.0, ..info() }),
        ("span_weight NaN", ManifestInfo { span_weight: f64::NAN, ..info() }),
        ("empty trainer", ManifestInfo { trainer: " ".into(), ..info() }),
        ("upper-case digest", ManifestInfo { consumed_digest: "EF".repeat(32), ..info() }),
        ("short head digest", ManifestInfo { head_init_digest: Some("01".into()), ..info() }),
        ("device with a dash", ManifestInfo { device: "me-tal".into(), ..info() }),
    ];
    for (what, bad) in cases {
        let r = export(&d, &named(&tower), &named(&head), Some(&l), &bad, Precision::Bf16);
        assert!(matches!(r, Err(ExportError::Refused(_))), "{what}: {r:?}");
    }
    // Without a layout the tower's own embed_tokens still fixes the vocabulary.
    let r = export(&d, &named(&tower), &named(&head), None, &ManifestInfo { vocab_size: 9, ..info() }, Precision::Bf16);
    assert!(matches!(r, Err(ExportError::Refused(m)) if m.contains("embed_tokens")));
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "no refusal leaves a file behind");
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_tensor_set_the_loader_would_not_read_is_refused_and_nothing_is_written() {
    let l = layout();
    let tower = values(&l.text_tensors().unwrap(), 0.0);
    let head = values(&l.span_head_tensors(), 1.0);
    let d = dir("missing");
    let go = |t: &[(ParamSpec, Vec<f32>)], h: &[(ParamSpec, Vec<f32>)], lay: Option<&Layout>| {
        export(&d, &named(t), &named(h), lay, &info(), Precision::Bf16)
    };

    assert!(matches!(go(&tower[1..], &head, Some(&l)), Err(ExportError::Refused(m)) if m.contains("missing")));

    let mut extra = tower.clone();
    extra.push((ParamSpec::new("lm_head.weight", &[8, 4]), vec![0.0; 32]));
    assert!(matches!(go(&extra, &head, Some(&l)), Err(ExportError::Refused(m)) if m.contains("extra")));

    let mut bent = tower.clone();
    let k = bent
        .iter()
        .position(|(s, _)| s.name != "embed_tokens.weight" && s.shape.len() >= 2)
        .unwrap();
    bent[k].0.shape = vec![bent[k].1.len()];
    assert!(matches!(go(&bent, &head, Some(&l)), Err(ExportError::Refused(_))));

    let mut miscount = tower.clone();
    miscount[k].1.pop();
    assert!(go(&miscount, &head, None).is_err());

    let mut nan = tower.clone();
    nan[3].1[1] = f32::NAN;
    assert!(matches!(go(&nan, &head, Some(&l)), Err(ExportError::NonFinite(_))));
    let mut huge = tower.clone();
    huge[3].1[0] = f32::MAX;
    assert!(matches!(go(&huge, &head, Some(&l)), Err(ExportError::NonFinite(_))), "rounds to inf in bf16");
    let mut bad_head = head.clone();
    bad_head[0].1[0] = f32::INFINITY;
    assert!(matches!(go(&tower, &bad_head, Some(&l)), Err(ExportError::NonFinite(_))));
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "no refusal leaves a file behind");
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn the_real_2b_layout_names_320_tower_tensors() {
    let cfg = include_str!("../../qd-export/tests/fixtures/qwen35_2b_base_config.json");
    let l = Layout::from_config_json(cfg).unwrap();
    assert_eq!(l.text_tensors().unwrap().len(), 320);
    assert_eq!(l.span_head_tensors().len(), 4);
}

/// Writes one Metal export where the Python scorer's own reader can check it:
/// `QD_TRAIN_EXPORT_FOR_SCORER=<dir> cargo test -p qd-train --test export -- --ignored`, then
/// `tools/qd_train_oracle_trainer.py --verify-export <dir>/epoch-seed0-metal.safetensors`. CPU
/// only; ignored by default because it needs that directory and the Python half to mean anything.
#[test]
#[ignore = "writes an export for the Python scorer's reader; see the doc comment"]
fn export_for_the_python_scorer() {
    let d = PathBuf::from(std::env::var("QD_TRAIN_EXPORT_FOR_SCORER").expect("QD_TRAIN_EXPORT_FOR_SCORER=<dir>"));
    std::fs::create_dir_all(&d).unwrap();
    let l = layout();
    let tower = values(&l.text_tensors().unwrap(), 0.0);
    let head = values(&l.span_head_tensors(), 1.0);
    let s = export(&d, &named(&tower), &named(&head), Some(&l), &info(), Precision::Bf16).unwrap();
    println!("{}", s.weights.display());
}
