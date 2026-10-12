//! D17: the 17-row readout delta (`AUDIT/lora-2b-2026-10-09/fable-ft-decisions.md` D2), exported
//! beside the tower as `d17.safetensors` and served as `embed[answer ids] + D17`.
//!
//! The source carries it as `d17.delta` (`qd_train.lora.AnswerRowDelta.delta`, a `[17, H]` F32
//! parameter; `tools/ckpt_average.py` averages it as the `d17` tree). The release carries it F32,
//! bit for bit, as `delta` in `d17.safetensors`, recorded in `files` and in a `d17` manifest block.
//! A source without it exports exactly as before: no file, and `"d17": null`.

mod common;

use common::{build, f32_tensor, standard_tensors, tiny_config, Fixture, Tensor, HIDDEN};
use qd_export::safetensors::{Dtype, SafeTensorsFile};
use qd_export::{export, RefusalKind};
use serde_json::Value;

const D17_FILE: &str = "d17.safetensors";

fn fixture_with(name: &str, t: Tensor) -> Fixture {
    let mut tensors = standard_tensors();
    tensors.insert(name.to_string(), t);
    build(&tensors, &tiny_config(), |_| {})
}

fn manifest(fx: &Fixture) -> Value {
    serde_json::from_slice(&std::fs::read(fx.out.join("release_manifest.json")).unwrap()).unwrap()
}

#[test]
fn a_source_with_d17_exports_it_f32_bit_for_bit_and_records_it() {
    let delta = f32_tensor(&[17, HIDDEN], 77);
    let fx = fixture_with("d17.delta", delta.clone());
    let summary = export(&fx.request()).expect("a source carrying d17.delta exports");

    let file = SafeTensorsFile::open(&fx.out.join(D17_FILE)).expect("the release holds d17.safetensors");
    assert_eq!(file.tensors().len(), 1, "one tensor, `delta`");
    let info = &file.tensors()["delta"];
    assert_eq!((info.dtype, info.shape.clone()), (Dtype::F32, vec![17, HIDDEN]));
    let mut got = vec![0u8; delta.bytes.len()];
    file.read_tensor_range(info, 0, &mut got).unwrap();
    assert_eq!(got, delta.bytes, "F32, bit for bit");

    let m = manifest(&fx);
    let sha = summary.files.get(D17_FILE).expect("the summary lists the file");
    assert_eq!(m["files"][D17_FILE]["sha256"].as_str(), Some(sha.as_str()));
    assert_eq!(m["d17"]["file"], D17_FILE);
    assert_eq!(m["d17"]["rows"], 17);
    assert_eq!(m["d17"]["hidden_size"], HIDDEN);
    assert_eq!(m["d17"]["dtype"], "F32");
    assert!(
        m["d17"]["serving"].as_str().unwrap().contains("embed[answer ids] + D17"),
        "the manifest states the readout: {}",
        m["d17"]["serving"]
    );
}

#[test]
fn a_source_without_d17_exports_as_before() {
    let fx = build(&standard_tensors(), &tiny_config(), |_| {});
    export(&fx.request()).unwrap();
    assert!(!fx.out.join(D17_FILE).exists());
    let m = manifest(&fx);
    assert!(m["d17"].is_null(), "no D17, no block: {}", m["d17"]);
    assert!(m["files"].get(D17_FILE).is_none());
}

#[test]
fn a_d17_that_is_not_17_rows_by_hidden_f32_and_finite_is_refused() {
    let cases: [(&str, Tensor, RefusalKind, &str); 5] = [
        ("16 rows", f32_tensor(&[16, HIDDEN], 1), RefusalKind::ShapeMismatch, "d17.delta"),
        ("wrong width", f32_tensor(&[17, HIDDEN + 1], 2), RefusalKind::ShapeMismatch, "d17.delta"),
        ("bf16", common::bf16_tensor(&[17, HIDDEN], 3), RefusalKind::Dtype, "d17.delta is BF16"),
        ("f16", common::f16_tensor(&[17, HIDDEN], 4), RefusalKind::Dtype, "d17.delta is F16"),
        (
            "nan",
            {
                let mut t = f32_tensor(&[17, HIDDEN], 5);
                t.bytes[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
                t
            },
            RefusalKind::NonFinite,
            "d17.delta[10]",
        ),
    ];
    for (what, tensor, kind, needle) in cases {
        let fx = fixture_with("d17.delta", tensor);
        let err = export(&fx.request()).expect_err(what);
        assert_eq!(err.kind, kind, "{what}: {err}");
        assert!(err.detail.contains(needle), "{what}: expected {needle:?} in {err}");
        assert!(!fx.out.exists(), "{what}: nothing is published");
        assert!(fx.leftovers().is_empty(), "{what}: staging left behind");
    }
}

#[test]
fn a_d17_tree_with_any_other_tensor_is_still_an_unexplained_extra() {
    let fx = fixture_with("d17.other", f32_tensor(&[17, HIDDEN], 6));
    let err = export(&fx.request()).expect_err("only d17.delta is the D17 tensor");
    assert_eq!(err.kind, RefusalKind::ExtraTensor, "{err}");
    assert!(err.detail.contains("d17.other"), "{err}");
}
