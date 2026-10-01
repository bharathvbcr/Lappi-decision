//! Every refusal, each from the standard fixture with one thing wrong. Each test also asserts
//! that nothing was left behind: no release, and no staging directory.

mod common;

use std::collections::BTreeMap;

use common::{build, standard_tensors, tiny_config, Fixture, Tensor};
use qd_export::safetensors::Dtype;
use qd_export::{export, AllowedExtra, ExportRequest, Refusal, RefusalKind};
use serde_json::Value;

fn refused(fx: &Fixture, req: &ExportRequest, kind: RefusalKind, needle: &str) -> Refusal {
    let err = export(req).expect_err("this export must be refused");
    assert_eq!(err.kind, kind, "{err}");
    assert!(err.detail.contains(needle), "{needle:?} not in {err}");
    assert!(!fx.out.exists(), "a refused export left a release");
    assert!(fx.leftovers().is_empty(), "a refused export left {:?}", fx.leftovers());
    err
}

fn with_tensors(edit: impl FnOnce(&mut BTreeMap<String, Tensor>)) -> Fixture {
    let mut t = standard_tensors();
    edit(&mut t);
    build(&t, &tiny_config(), |_| {})
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn an_existing_output_directory_is_never_written_into() {
    let fx = common::standard();
    std::fs::create_dir(&fx.out).unwrap();
    let err = export(&fx.request()).unwrap_err();
    assert_eq!(err.kind, RefusalKind::OutputExists);
    assert_eq!(std::fs::read_dir(&fx.out).unwrap().count(), 0, "the existing directory was touched");
}

#[test]
fn the_vocabulary_must_agree_everywhere() {
    // The operator's expectation against the config.
    let fx = common::standard();
    let mut req = fx.request();
    req.expect_vocab_size = 248_320;
    refused(&fx, &req, RefusalKind::VocabMismatch, "--expect-vocab-size is 248320");

    // The source manifest against the config.
    let fx = build(&standard_tensors(), &tiny_config(), |m| m["vocab_size"] = 13_787.into());
    refused(&fx, &fx.request(), RefusalKind::VocabMismatch, "says vocab_size 13787");

    // A remapped embedding against the config: refused as a vocabulary, not served short.
    let fx = with_tensors(|t| {
        t.insert("tower.embed_tokens.weight".into(), common::f32_tensor(&[200, common::HIDDEN], 7));
    });
    refused(&fx, &fx.request(), RefusalKind::VocabMismatch, "remapped tower");
}

#[test]
fn a_tensor_the_loader_needs_must_be_there() {
    let fx = with_tensors(|t| {
        t.remove("tower.layers.1.self_attn.k_norm.weight");
        t.remove("span_head.abstain_end");
    });
    let err = refused(&fx, &fx.request(), RefusalKind::MissingTensor, "tower.layers.1.self_attn.k_norm.weight");
    assert!(err.detail.contains("span_head.abstain_end"), "every missing tensor is named: {err}");
}

#[test]
fn an_extra_tensor_is_refused_unless_dropped_with_a_reason() {
    let extra = |t: &mut BTreeMap<String, Tensor>| {
        t.insert("tower.rotary_emb.inv_freq".into(), common::f32_tensor(&[32], 9));
        t.insert("optimizer.step".into(), common::f32_tensor(&[1], 10));
    };
    let fx = with_tensors(extra);
    let err = refused(&fx, &fx.request(), RefusalKind::ExtraTensor, "tower.rotary_emb.inv_freq");
    assert!(err.detail.contains("optimizer.step"));

    // One allowlisted, one not: still refused, naming the one without a reason.
    let mut req = fx.request();
    req.allow_extra = vec![AllowedExtra::parse("tower.rotary_emb.inv_freq=recomputed from config").unwrap()];
    let err = refused(&fx, &req, RefusalKind::ExtraTensor, "optimizer.step");
    assert!(!err.detail.contains("inv_freq"), "{err}");

    // A stale allowlist entry is refused: a reason nobody re-read.
    let mut req = fx.request();
    req.allow_extra = vec![
        AllowedExtra::parse("tower.rotary_emb.inv_freq=recomputed from config").unwrap(),
        AllowedExtra::parse("optimizer.step=not weights").unwrap(),
        AllowedExtra::parse("tower.gone=was here once").unwrap(),
    ];
    refused(&fx, &req, RefusalKind::ExtraTensor, "tower.gone");

    // Both explained: dropped from the release and recorded with their reasons.
    let mut req = fx.request();
    req.allow_extra = vec![
        AllowedExtra::parse("tower.rotary_emb.inv_freq=recomputed from config").unwrap(),
        AllowedExtra::parse("optimizer.step=not weights").unwrap(),
    ];
    let summary = export(&req).unwrap();
    assert_eq!(summary.dropped.len(), 2);
    let manifest: Value = serde_json::from_slice(&std::fs::read(fx.out.join("release_manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["dropped_extras"][0]["name"], "optimizer.step");
    assert_eq!(manifest["dropped_extras"][1]["reason"], "recomputed from config");
    let st = qd_export::safetensors::SafeTensorsFile::open(&fx.out.join("model.safetensors")).unwrap();
    assert!(!st.tensors().keys().any(|k| k.contains("inv_freq") || k.contains("optimizer")));

    assert!(AllowedExtra::parse("tower.x=").is_err());
    assert!(AllowedExtra::parse("tower.x").is_err());
}

#[test]
fn a_shape_the_config_does_not_imply_is_refused() {
    let fx = with_tensors(|t| {
        t.insert("tower.layers.0.mlp.up_proj.weight".into(), common::f16_tensor(&[common::INTER, common::HIDDEN + 1], 3));
        t.insert("span_head.start_proj.weight".into(), common::f32_tensor(&[common::HIDDEN, 2], 4));
    });
    let err = refused(&fx, &fx.request(), RefusalKind::ShapeMismatch, "tower.layers.0.mlp.up_proj.weight");
    assert!(err.detail.contains("span_head.start_proj.weight"));
}

#[test]
fn a_dtype_that_is_not_a_trained_float_is_refused() {
    let fx = with_tensors(|t| {
        let n = t["tower.layers.0.linear_attn.A_log"].shape.clone();
        t.insert(
            "tower.layers.0.linear_attn.A_log".into(),
            Tensor { dtype: Dtype::F64, shape: n.clone(), bytes: vec![0u8; 8 * n.iter().product::<usize>()] },
        );
        let e = t["tower.norm.weight"].shape.clone();
        t.insert(
            "tower.norm.weight".into(),
            Tensor { dtype: Dtype::I32, shape: e.clone(), bytes: vec![0u8; 4 * e.iter().product::<usize>()] },
        );
    });
    let err = refused(&fx, &fx.request(), RefusalKind::Dtype, "tower.layers.0.linear_attn.A_log is F64");
    assert!(err.detail.contains("tower.norm.weight is I32"));

    // The span head is released at the precision it was trained in, so a cast one is refused.
    let fx = with_tensors(|t| {
        let s = t["span_head.abstain_start"].shape.clone();
        t.insert("span_head.abstain_start".into(), common::bf16_tensor(&s, 5));
    });
    refused(&fx, &fx.request(), RefusalKind::Dtype, "span_head.abstain_start is BF16");
}

#[test]
fn a_value_that_is_not_a_finite_bf16_is_refused_and_staging_is_cleaned_up() {
    // A NaN deep in an F32 tensor: found while streaming, after staging began.
    let fx = with_tensors(|t| {
        let shape = t["tower.embed_tokens.weight"].shape.clone();
        let mut v = common::values(shape.iter().product(), 1);
        v[4321] = f32::NAN;
        t.insert("tower.embed_tokens.weight".into(), Tensor { dtype: Dtype::F32, shape, bytes: f32_bytes(&v) });
    });
    refused(&fx, &fx.request(), RefusalKind::NonFinite, "tower.embed_tokens.weight[4321]");

    // A finite value that rounds to infinity in bf16.
    let fx = with_tensors(|t| {
        let shape = t["tower.layers.0.linear_attn.A_log"].shape.clone();
        let mut v = common::values(shape.iter().product(), 2);
        v[0] = f32::MAX;
        t.insert("tower.layers.0.linear_attn.A_log".into(), Tensor { dtype: Dtype::F32, shape, bytes: f32_bytes(&v) });
    });
    refused(&fx, &fx.request(), RefusalKind::NonFinite, "would round to infinity");

    // An infinity in the span head, which is copied rather than cast.
    let fx = with_tensors(|t| {
        let shape = t["span_head.end_proj.weight"].shape.clone();
        let mut v = common::values(shape.iter().product(), 3);
        v[7] = f32::NEG_INFINITY;
        t.insert("span_head.end_proj.weight".into(), Tensor { dtype: Dtype::F32, shape, bytes: f32_bytes(&v) });
    });
    refused(&fx, &fx.request(), RefusalKind::NonFinite, "span_head.end_proj.weight[7]");
}

#[test]
fn the_source_manifest_must_describe_the_source() {
    let fx = build(&standard_tensors(), &tiny_config(), |m| {
        m["safetensors_sha256"] = "00".repeat(32).into();
    });
    refused(&fx, &fx.request(), RefusalKind::SourceManifest, "describes 0000");

    let fx = build(&standard_tensors(), &tiny_config(), |m| m["n_tensors"] = 3.into());
    refused(&fx, &fx.request(), RefusalKind::SourceManifest, "counts 3 tensors");

    let fx = build(&standard_tensors(), &tiny_config(), |m| m["ft_row_ids"] = serde_json::json!([]));
    refused(&fx, &fx.request(), RefusalKind::SourceManifest, "--ft-row-ids");

    let fx = build(&standard_tensors(), &tiny_config(), |m| {
        m.as_object_mut().unwrap().remove("safetensors_sha256");
    });
    refused(&fx, &fx.request(), RefusalKind::SourceManifest, "no safetensors_sha256");

    let fx = common::standard();
    std::fs::remove_file(&fx.manifest).unwrap();
    refused(&fx, &fx.request(), RefusalKind::SourceManifest, "avg.safetensors.manifest.json");
}

#[test]
fn the_tokenizer_must_be_the_pinned_one_and_complete() {
    let fx = common::standard();
    let mut req = fx.request();
    req.tokenizer_sha256 = "ab".repeat(32);
    refused(&fx, &req, RefusalKind::Tokenizer, "pinned abab");

    let mut req = fx.request();
    req.tokenizer_sha256 = "not-hex".into();
    refused(&fx, &req, RefusalKind::Tokenizer, "not 64 hex digits");

    std::fs::remove_file(fx.snapshot.join("merges.txt")).unwrap();
    refused(&fx, &fx.request(), RefusalKind::Tokenizer, "merges.txt");
}

#[test]
fn the_calibration_table_must_be_one_the_runtime_reads_back_unchanged() {
    let fx = common::standard();
    let table = qd_runtime::calibration::CalibrationTable::reference();
    let mut doc = serde_json::to_value(&table).unwrap();

    let bad = fx.dir.0.join("bad.json");
    let mut invalid = doc.clone();
    invalid["span"]["temperature"] = 0.0.into();
    std::fs::write(&bad, invalid.to_string()).unwrap();
    let mut req = fx.request();
    req.calibration = Some(bad.clone());
    refused(&fx, &req, RefusalKind::Calibration, "temperature");

    // A field the runtime does not read: serde would drop it silently.
    doc["fitted_on"] = "val rows 0-999".into();
    std::fs::write(&bad, doc.to_string()).unwrap();
    refused(&fx, &req, RefusalKind::Calibration, "does not read back as itself");

    std::fs::write(&bad, "{\"name\": 3}").unwrap();
    refused(&fx, &req, RefusalKind::Calibration, "not a CalibrationTable");

    // An integer literal for a float is the same table, not a different one.
    let mut ints = serde_json::to_value(&table).unwrap();
    ints["span"]["temperature"] = 1.into();
    std::fs::write(&bad, ints.to_string()).unwrap();
    let summary = export(&req).expect("1 and 1.0 are one temperature");
    assert_eq!(summary.calibration_hash, Some(table.hash()));
}

#[test]
fn a_config_whose_layout_the_loader_does_not_read_is_refused() {
    let mut cfg = tiny_config();
    cfg["tie_word_embeddings"] = false.into();
    let fx = build(&standard_tensors(), &cfg, |_| {});
    refused(&fx, &fx.request(), RefusalKind::Config, "tie_word_embeddings");
}
