//! The release, read back by the **serving loader's own code**, not by this crate's.
//!
//! Everything here that touches the export goes through qd-metal: its safetensors reader, its
//! `ModelConfig`, `weights_file`, `check_text_tensor_set`, `prepare_layer` for every layer (the
//! CPU half of `Model::load`; the other half uploads to the GPU and is not run), its weight hash
//! and its tokenizer. The expected bf16 bits come from tessl's `f32_to_bf16_bits`, not from
//! `qd_export::bf16`. So a defect has to be made identically twice -- here and in the code the
//! release is served by -- to pass.
//!
//! No GPU: `Model::load` itself needs a Metal device and is not called (CLAUDE.md rule 5).

mod common;

use std::collections::BTreeMap;

use qd_export::bf16::{f16_bits_to_f32, f32_to_bf16_rne};
use qd_export::layout::Layout;
use qd_export::{export, RefusalKind};
use qd_metal::config::ModelConfig;
use qd_metal::safetensors::{widen, Dtype as MetalDtype, SafeTensors};
use qd_metal::tokenizer::QwenTokenizer;
use qd_metal::weights::{self, TensorDigest, TEXT_PREFIX};
use qd_runtime::calibration::CalibrationTable;
use qd_runtime::{hex, sha256};
use serde_json::Value;

/// The bf16 bits the release must hold for a source tensor, computed by tessl / qd-metal.
fn oracle_bf16(t: &common::Tensor) -> Vec<u16> {
    let words = |b: &[u8]| -> Vec<u16> { b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect() };
    match t.dtype {
        qd_export::safetensors::Dtype::Bf16 => words(&t.bytes),
        qd_export::safetensors::Dtype::F16 => widen(MetalDtype::F16, &t.bytes)
            .into_iter()
            .map(tessl::tensor::f32_to_bf16_bits)
            .collect(),
        qd_export::safetensors::Dtype::F32 => widen(MetalDtype::F32, &t.bytes)
            .into_iter()
            .map(tessl::tensor::f32_to_bf16_bits)
            .collect(),
        other => panic!("fixture holds a {other:?} tower tensor"),
    }
}

#[test]
fn the_release_loads_through_qd_metal_and_holds_the_bf16_cast_of_every_tensor() {
    let tensors = common::standard_tensors();
    let fx = common::build(&tensors, &common::tiny_config(), |_| {});
    let table = CalibrationTable::reference();
    let table_path = fx.dir.0.join("table.json");
    std::fs::write(&table_path, serde_json::to_vec_pretty(&table).unwrap()).unwrap();
    let mut req = fx.request();
    req.calibration = Some(table_path.clone());
    let summary = export(&req).expect("the standard fixture exports");
    assert!(fx.leftovers().is_empty(), "staging left behind: {:?}", fx.leftovers());

    // The loader's entry points, in Model::load's order.
    let cfg = ModelConfig::load(&fx.out).expect("qd-metal's ModelConfig accepts the release's config.json");
    let wf = qd_metal::model::weights_file(&fx.out).unwrap();
    assert_eq!(wf, fx.out.join("model.safetensors"));
    let st = SafeTensors::open(&wf).expect("qd-metal's reader opens model.safetensors");
    weights::check_text_tensor_set(&st, &cfg).expect("exactly the text tensors the forward pass reads");
    let names: Vec<String> = st.names().map(str::to_string).collect();
    assert_eq!(names, weights::expected_text_tensors(&cfg), "and nothing outside the prefix either");

    // Both readers agree on every header entry.
    let mine = qd_export::safetensors::SafeTensorsFile::open(&wf).unwrap();
    assert_eq!(mine.tensors().len(), names.len());
    for (name, info) in mine.tensors() {
        let theirs = st.info(name).unwrap();
        assert_eq!(theirs.dtype, MetalDtype::Bf16, "{name}");
        assert_eq!((&theirs.shape, theirs.start, theirs.end), (&info.shape, info.start, info.end), "{name}");
        assert_eq!(info.dtype.as_str(), "BF16");
    }

    // Every tensor, bit for bit, against tessl's cast of the source.
    let mut rounded = 0;
    for name in &names {
        let short = name.strip_prefix(TEXT_PREFIX).unwrap();
        let src = &tensors[&format!("tower.{short}")];
        let got = st.bf16(name, &src.shape).unwrap();
        assert_eq!(got, oracle_bf16(src), "{name}");
        if src.dtype != qd_export::safetensors::Dtype::Bf16 {
            rounded += 1;
        }
    }
    assert_eq!((summary.rounded, summary.copied), (rounded, names.len() - rounded));
    // The ties at the head of every F32 tensor went to even, as tessl says they should.
    let embed = st.bf16(&format!("{TEXT_PREFIX}embed_tokens.weight"), &[common::VOCAB, common::HIDDEN]).unwrap();
    assert_eq!(&embed[..6], &[0x3F80, 0x3F82, 0xBF80, 0x0000, 0x0002, 0x4000]);

    // The CPU half of Model::load (model.rs:323-429): embed and final norm digests, then
    // prepare_layer for every layer -- which also checks each tensor's shape and dtype against
    // what the kernels read -- then the weight hash the backend reports.
    let h = cfg.hidden;
    let embed_name = format!("{TEXT_PREFIX}embed_tokens.weight");
    let info = st.expect(&embed_name, &[cfg.vocab, h], &[MetalDtype::Bf16]).unwrap().clone();
    let mut bytes = vec![0u8; info.nbytes()];
    st.read_into(&embed_name, &mut bytes).unwrap();
    let mut digests: Vec<TensorDigest> = vec![(embed_name, sha256(&bytes))];
    let norm_name = format!("{TEXT_PREFIX}norm.weight");
    let info = st.expect(&norm_name, &[h], &[MetalDtype::Bf16, MetalDtype::F32]).unwrap().clone();
    let mut bytes = vec![0u8; info.nbytes()];
    st.read_into(&norm_name, &mut bytes).unwrap();
    digests.push((norm_name, sha256(&bytes)));
    for i in 0..cfg.n_layers() {
        let layer = weights::prepare_layer(&st, &cfg, i).unwrap_or_else(|e| panic!("layer {i}: {e}"));
        digests.extend(layer.digests);
    }
    assert_eq!(digests.len(), names.len(), "Model::load reads every tensor once");
    let loader_hash = weights::weight_hash(&mut digests);
    assert_eq!(summary.weight_hash, loader_hash, "the manifest's weight_hash is the one qd-metal will report");

    // Tokenizer: the loader's hash is the pinned file's.
    let tok = QwenTokenizer::load(&fx.out.join("tokenizer.json")).unwrap();
    assert_eq!(tok.hash(), summary.tokenizer_hash);
    assert_eq!(tok.hash(), common::tokenizer_sha256(&fx.snapshot));

    // Span head: F32, bit for bit, read by qd-metal's reader.
    let head = SafeTensors::open(&fx.out.join("span_head.safetensors")).unwrap();
    let layout = Layout::from_config_json(&common::tiny_config().to_string()).unwrap();
    for (short, shape) in layout.span_head_tensors() {
        let info = head.expect(&short, &shape, &[MetalDtype::F32]).unwrap().clone();
        let mut got = vec![0u8; info.nbytes()];
        head.read_into(&short, &mut got).unwrap();
        assert_eq!(got, tensors[&format!("span_head.{short}")].bytes, "{short}");
    }

    // The manifest: every file's sha256 is the file's, and the identity is the loader's.
    let manifest: Value = serde_json::from_slice(&std::fs::read(fx.out.join("release_manifest.json")).unwrap()).unwrap();
    let files = manifest["files"].as_object().unwrap();
    let on_disk: BTreeMap<String, String> = std::fs::read_dir(&fx.out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n != "release_manifest.json")
        .map(|n| (n.clone(), hex(&sha256(&std::fs::read(fx.out.join(&n)).unwrap()))))
        .collect();
    assert_eq!(files.len(), on_disk.len());
    for (name, sha) in &on_disk {
        assert_eq!(files[name]["sha256"], sha.as_str(), "{name}");
    }
    assert_eq!(on_disk, summary.files);
    let id = &manifest["expected_identity"];
    assert_eq!(id["weight_hash"], loader_hash.as_str());
    assert_eq!(id["tokenizer_hash"], tok.hash());
    assert_eq!(id["calibration_hash"], table.hash().as_str());
    assert!(id["not_computed"]["head_hash"].is_string(), "named as not computed, not omitted");
    assert_eq!(manifest["calibration"]["table_hash"], table.hash().as_str());
    let src = &manifest["source"];
    assert_eq!(src["ft_row_ids"], serde_json::json!(common::FT_ROW_IDS));
    assert_eq!(src["manifest_sha256"], hex(&sha256(&std::fs::read(&fx.manifest).unwrap())).as_str());
    assert_eq!(src["safetensors_sha256"], hex(&sha256(&std::fs::read(&fx.source).unwrap())).as_str());

    // Passed through byte for byte.
    assert_eq!(std::fs::read(fx.out.join("config.json")).unwrap(), std::fs::read(fx.snapshot.join("config.json")).unwrap());
    assert_eq!(std::fs::read(fx.out.join("calibration.json")).unwrap(), std::fs::read(&table_path).unwrap());

    // And the release is never overwritten.
    assert_eq!(export(&req).unwrap_err().kind, RefusalKind::OutputExists);
}

/// The real thing at real size, on CPU: the Qwen3.5-2B-Base snapshot's own 320 text tensors,
/// renamed `tower.*` (as `ckpt_average` names a tower) beside a zero span head, exported at the
/// 248,320-row vocabulary and read back through qd-metal's CPU load path. The base stores the
/// 18 GDN layers' `A_log` and `linear_attn.norm.weight` as F32 (measured from its header,
/// 2026-10-01), so those 36 tensors are rounded; the other 284 are BF16 in the base and must
/// come back bit-identical to it.
///
/// `#[ignore]`d: it needs the 4.5 GB snapshot and writes ~8.5 GB to the temp dir. Run with
/// `cargo test --release -p qd-export --test roundtrip -- --ignored`.
#[test]
#[ignore = "needs the model snapshot and ~9 GB of temp disk"]
fn snapshot_the_base_weights_export_at_248320_and_load_through_qd_metal() {
    use qd_export::safetensors::{Dtype, PlannedTensor, SafeTensorsFile, Writer, CHUNK_BYTES};

    let snap = qd_metal::config::resolve_snapshot(None).expect("the Qwen3.5-2B-Base snapshot");
    let base = SafeTensorsFile::open(&qd_metal::model::weights_file(&snap).unwrap()).unwrap();
    let layout = Layout::from_config_json(&std::fs::read_to_string(snap.join("config.json")).unwrap()).unwrap();
    let dir = common::TempDir::new("real");

    // The source: the base's text tensors as tower.*, plus a zero F32 span head.
    let source = dir.0.join("avg.safetensors");
    let mut plan: Vec<PlannedTensor> = base
        .tensors()
        .iter()
        .filter_map(|(n, t)| {
            n.strip_prefix(TEXT_PREFIX).map(|short| PlannedTensor {
                name: format!("tower.{short}"),
                dtype: t.dtype,
                shape: t.shape.clone(),
            })
        })
        .collect();
    assert_eq!(plan.len(), 320);
    for (short, shape) in layout.span_head_tensors() {
        plan.push(PlannedTensor { name: format!("span_head.{short}"), dtype: Dtype::F32, shape });
    }
    let n_tensors = plan.len();
    let mut w = Writer::create(&source, plan).unwrap();
    let mut buf = vec![0u8; CHUNK_BYTES];
    for planned in w.order() {
        w.begin(&planned.name).unwrap();
        if let Some(short) = planned.name.strip_prefix("tower.") {
            let info = &base.tensors()[&format!("{TEXT_PREFIX}{short}")];
            let mut at = 0u64;
            while at < info.nbytes() {
                let n = usize::try_from((info.nbytes() - at).min(CHUNK_BYTES as u64)).unwrap();
                base.read_tensor_range(info, at, &mut buf[..n]).unwrap();
                w.write(&buf[..n]).unwrap();
                at += n as u64;
            }
        } else {
            let n: usize = planned.shape.iter().product::<usize>() * 4;
            w.write(&vec![0u8; n]).unwrap();
        }
        w.end().unwrap();
    }
    let written = w.finish().unwrap();
    let manifest = serde_json::json!({
        "tool": "qd-export roundtrip test (the base weights, renamed)",
        "vocab_size": 248_320,
        "ft_row_ids": ["none: the base weights, not a trained tower"],
        "safetensors_sha256": hex(&written.sha256),
        "n_tensors": n_tensors,
    });
    std::fs::write(dir.0.join("avg.safetensors.manifest.json"), manifest.to_string()).unwrap();

    let out = dir.0.join("release");
    let summary = export(&qd_export::ExportRequest {
        source: source.clone(),
        source_manifest: None,
        base_snapshot: snap.clone(),
        tokenizer_sha256: "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927".into(),
        expect_vocab_size: 248_320,
        calibration: None,
        allow_extra: Vec::new(),
        out: out.clone(),
    })
    .expect("the base weights export");
    assert_eq!(
        (summary.rounded, summary.copied),
        (36, 284),
        "the GDN A_log and gated-norm weights are the base's only F32 text tensors"
    );
    // The release does not need the source any more; free the disk before the read-back.
    std::fs::remove_file(&source).unwrap();

    let cfg = ModelConfig::load(&out).unwrap();
    assert_eq!(cfg.vocab, 248_320);
    let st = SafeTensors::open(&qd_metal::model::weights_file(&out).unwrap()).unwrap();
    weights::check_text_tensor_set(&st, &cfg).unwrap();
    let base_m = SafeTensors::open(&qd_metal::model::weights_file(&snap).unwrap()).unwrap();

    let digest_of = |s: &SafeTensors, name: &str| {
        let info = s.info(name).unwrap().clone();
        let mut b = vec![0u8; info.nbytes()];
        s.read_into(name, &mut b).unwrap();
        (info.dtype, sha256(&b))
    };
    let embed = format!("{TEXT_PREFIX}embed_tokens.weight");
    st.expect(&embed, &[248_320, 2048], &[MetalDtype::Bf16]).unwrap();
    let norm = format!("{TEXT_PREFIX}norm.weight");
    let mut digests: Vec<TensorDigest> = vec![(embed.clone(), digest_of(&st, &embed).1), (norm.clone(), digest_of(&st, &norm).1)];
    for i in 0..cfg.n_layers() {
        digests.extend(weights::prepare_layer(&st, &cfg, i).unwrap_or_else(|e| panic!("layer {i}: {e}")).digests);
    }
    assert_eq!(digests.len(), 320);
    let mut identical = 0;
    for (name, d) in &digests {
        let (base_dtype, base_d) = digest_of(&base_m, name);
        if base_dtype == MetalDtype::Bf16 {
            assert_eq!(*d, base_d, "{name}: a BF16 base tensor must come back bit for bit");
            identical += 1;
        } else {
            assert!(
                name.ends_with("linear_attn.A_log") || name.ends_with("linear_attn.norm.weight"),
                "{name} is {base_dtype:?} in the base"
            );
            // Rounded from F32: it must be the bf16 cast of the base's value, by tessl's rule.
            let info = base_m.info(name).unwrap().clone();
            let mut raw = vec![0u8; info.nbytes()];
            base_m.read_into(name, &mut raw).unwrap();
            let cast: Vec<u8> = widen(MetalDtype::F32, &raw)
                .into_iter()
                .flat_map(|x| tessl::tensor::f32_to_bf16_bits(x).to_le_bytes())
                .collect();
            assert_eq!(*d, sha256(&cast), "{name}: not the bf16 cast of the base's F32");
        }
    }
    assert_eq!(identical, 284);
    assert_eq!(weights::weight_hash(&mut digests), summary.weight_hash);

    let tok = QwenTokenizer::load(&out.join("tokenizer.json")).unwrap();
    assert_eq!(tok.hash(), "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927");
    assert_eq!(std::fs::read(out.join("config.json")).unwrap(), std::fs::read(snap.join("config.json")).unwrap());
    eprintln!(
        "real export: weight_hash {}; model.safetensors {} bytes",
        summary.weight_hash,
        std::fs::metadata(out.join("model.safetensors")).unwrap().len()
    );
}

/// The layout this crate derives for the real 2B config is the set qd-metal reads, name for
/// name.
#[test]
fn the_real_2b_layout_is_the_set_qd_metal_reads() {
    let real = include_str!("fixtures/qwen35_2b_base_config.json");
    let cfg = ModelConfig::from_json(real).expect("qd-metal accepts the real config");
    let layout = Layout::from_config_json(real).unwrap();
    let mine: Vec<String> = layout
        .text_tensors()
        .unwrap()
        .keys()
        .map(|k| format!("{TEXT_PREFIX}{k}"))
        .collect();
    assert_eq!(mine, weights::expected_text_tensors(&cfg));
    assert_eq!((layout.vocab, cfg.vocab), (248_320, 248_320));
}

/// Round to nearest-even agrees with tessl's `f32_to_bf16_bits` wherever the result is a finite
/// bf16, and is refused exactly where tessl's is not: every upper half with the three low halves
/// that decide rounding (just below, exactly at, just above the tie), plus a strided sweep.
#[test]
fn the_bf16_cast_is_tessls_on_every_finite_result() {
    let mut checked = 0u64;
    let mut check = |bits: u32| {
        let x = f32::from_bits(bits);
        let theirs = tessl::tensor::f32_to_bf16_bits(x);
        match f32_to_bf16_rne(x) {
            Some(mine) => assert_eq!(mine, theirs, "{bits:#010x}"),
            None => assert!(
                !x.is_finite() || theirs & 0x7FFF == 0x7F80,
                "refused {bits:#010x}, which tessl rounds to the finite {theirs:#06x}"
            ),
        }
        checked += 1;
    };
    for hi in 0..=u32::from(u16::MAX) {
        for lo in [0x0000, 0x7FFF, 0x8000, 0x8001, 0xFFFF] {
            check((hi << 16) | lo);
        }
    }
    let mut bits = 0u32;
    while let Some(next) = bits.checked_add(65_521) {
        check(bits);
        bits = next;
    }
    assert_eq!(checked, 65_536 * 5 + 65_551);
}

/// F16 widening agrees with tessl's on all 65,536 halves (NaNs compared as NaN).
#[test]
fn f16_widening_is_tessls_on_every_half() {
    for h in 0..=u16::MAX {
        let mine = f16_bits_to_f32(h);
        let theirs = widen(MetalDtype::F16, &h.to_le_bytes())[0];
        if theirs.is_nan() {
            assert!(mine.is_nan(), "{h:#06x}");
        } else {
            assert_eq!(mine.to_bits(), theirs.to_bits(), "{h:#06x}");
        }
    }
}
