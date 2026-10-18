//! Shared by the span tests: the fixture directory, JSON and bit-pattern readers, and the tiny
//! fixture checkpoint exported into a release by `qd_export` itself.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use qd_export::safetensors::{Dtype, PlannedTensor, Writer};
use qd_export::{ExportRequest, export};
use qd_metal::backend::{MetalBackend, MetalConfig};
use qd_runtime::backend::{DecisionBackend, DecodeMode, Logits, QueryKind, SlotQuery};
use serde_json::Value;

/// The tiny tower's hidden size (`tools/gen_span_fixture.py`).
pub const HIDDEN: usize = 64;

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/span")
}

pub fn read_json(path: &Path) -> Value {
    serde_json::from_slice(
        &std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

pub fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|b| f32::from_bits(u32::try_from(b.as_i64().unwrap() & 0xFFFF_FFFF).unwrap()))
        .collect()
}

pub fn bf16s(v: &Value) -> Vec<u16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|b| u16::try_from(b.as_i64().unwrap()).unwrap())
        .collect()
}

pub struct Temp(pub PathBuf);
impl Temp {
    pub fn new(tag: &str) -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("qd-metal-span-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        // A leftover temp dir only costs disk.
        let _ignored = std::fs::remove_dir_all(&self.0);
    }
}

/// Write `tensors` (F32) to `path` as a safetensors file.
pub fn write_f32(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    // The writer never overwrites: a test that rewrites a file removes it first.
    let _absent_is_fine = std::fs::remove_file(path);
    let plan = tensors
        .iter()
        .map(|(n, shape, _)| PlannedTensor {
            name: (*n).to_string(),
            dtype: Dtype::F32,
            shape: shape.clone(),
        })
        .collect();
    let mut w = Writer::create(path, plan).unwrap();
    for planned in w.order() {
        let (_, _, values) = tensors.iter().find(|(n, _, _)| *n == planned.name).unwrap();
        w.begin(&planned.name).unwrap();
        w.write(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        w.end().unwrap();
    }
    w.finish().unwrap();
}

pub fn tiny_release(tag: &str, edit_source_manifest: impl FnOnce(&mut Value)) -> (Temp, PathBuf) {
    let tiny = fixtures().join("tiny");
    let dir = Temp::new(tag);
    let manifest_path = dir.0.join("source.safetensors.manifest.json");
    let mut manifest = read_json(&tiny.join("source.safetensors.manifest.json"));
    edit_source_manifest(&mut manifest);
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let out = dir.0.join("release");
    let snapshot = tiny.join("snapshot");
    let tokenizer_sha = qd_runtime::hex(&qd_runtime::sha256(
        &std::fs::read(snapshot.join("tokenizer.json")).unwrap(),
    ));
    export(&ExportRequest {
        source: tiny.join("source.safetensors"),
        source_manifest: Some(manifest_path),
        base_snapshot: snapshot,
        tokenizer_sha256: tokenizer_sha,
        expect_vocab_size: usize::try_from(manifest["vocab_size"].as_u64().unwrap()).unwrap(),
        calibration: None,
        calibration_report: None,
        exclude_families: Vec::new(),
        uncalibrated_reason: Some("tiny fixture: not served with a table".to_string()),
        train_manifest: tiny.join("train.json"),
        allow_extra: Vec::new(),
        averaged_head_preregistration: None,
        ft_ledger: None,
        out: out.clone(),
    })
    .expect("the tiny fixture exports");
    (dir, out)
}

pub fn edit_release_manifest(release: &Path, edit: impl FnOnce(&mut Value)) {
    let path = release.join("release_manifest.json");
    let mut m = read_json(&path);
    edit(&mut m);
    std::fs::write(path, serde_json::to_string_pretty(&m).unwrap()).unwrap();
}

// ---- the GPU tests' shared reference and backend helpers ----

/// Largest |rust - python| over the largest |python| score. Measured on this fixture; the
/// pre-norm convention misses it by more than 20x (see the convention test).
pub const SCORE_REL: f64 = 0.04;

pub struct Reference(pub Value);

impl Reference {
    pub fn load() -> Self {
        Self(read_json(&fixtures().join("tiny/reference.json")))
    }
    pub fn str(&self, key: &str) -> &str {
        self.0[key].as_str().unwrap()
    }
    pub fn floats(&self, key: &str) -> Vec<f32> {
        f32s(&self.0[key])
    }
    pub fn usize(&self, key: &str) -> usize {
        usize::try_from(self.0[key].as_u64().unwrap()).unwrap()
    }
    pub fn rows(&self, key: &str) -> Vec<u32> {
        self.0[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
            .collect()
    }
}

pub fn rel_diff(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .fold(0f64, |m, v| m.max(f64::from(v.abs())))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0f64, |m, (g, w)| m.max(f64::from((g - w).abs())))
        / scale
}

pub fn start(release: &std::path::Path) -> MetalBackend {
    MetalBackend::start(MetalConfig {
        snapshot: Some(release.to_path_buf()),
        calibration_hash: "fixture".into(),
        queue_capacity: 4,
        max_entries: 8,
        max_tokens: 1024,
        job_timeout: Duration::from_secs(120),
    })
    .expect("the tiny release starts a backend")
}

pub fn decode(
    backend: &MetalBackend,
    refc: &Reference,
    suffix_key: &str,
    kind: QueryKind,
    rows: usize,
) -> Result<Logits, qd_runtime::BackendError> {
    let handle = backend.prefill(refc.str("prefix"))?;
    let mut snapshot = backend.snapshot(&handle)?;
    let query = SlotQuery {
        slot_name: "slot",
        suffix: refc.str(suffix_key),
        rows,
        kind,
    };
    backend.decode_slot(&mut snapshot, &query, DecodeMode::ReadOnly)
}

/// The campaign file's recorded digest in a release manifest (`campaign/v6-averaging-preregistered.json`).
pub fn known_preregistration() -> Value {
    serde_json::json!({
        "file": "v6-averaging-preregistered.json",
        "sha256": "73ce1cbac9652b0fda5d170c029c8e147d5810c8fe08740439ddb9b7f9c0ceed",
        "name": "v6-averaging",
        "row": "primary",
    })
}

/// Shared-init evidence as the export records it after reading the ft ledger.
pub fn shared_init() -> Value {
    serde_json::json!({
        "shared": true, "content_digest": "cd".repeat(32), "seeds": [0, 1, 2],
        "ft_row_ids": ["row-seed0", "row-seed1", "row-seed2"],
    })
}
