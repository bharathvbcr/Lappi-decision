//! L-oracle's rung (b) fixture (`crates/qd-train/tests/fixtures/tiny-published`, written by
//! `tools/qd_train_oracle_tiny.py`), read for the Rust trainer.
//!
//! The batches are rebuilt as qd-train's own [`Batch`] (padded with `PAD_ID`, the
//! supervision arrays and the line-start mask the shard reader would produce), so the run goes
//! through the same [`RealBatch`] glue as a v4 run: `ft_supervision` and `plan_span_batch` decide
//! every letter target and span row, and `tests/rung_b_data.rs` checks their answers against the
//! oracle's `plan.json`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use qd_export::safetensors::{Dtype, SafeTensorsFile};
use qd_train::npy::NpyArray;
use qd_train::npz::StoredZip;
use qd_train::shards::{Batch, PAD_ID, SLOT_SPAN};
use qd_train::span_head::SpanHead;
use serde_json::Value;

pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../qd-train/tests/fixtures/tiny-published")
}

pub fn json(rel: &str) -> Value {
    let p = dir().join(rel);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

pub struct Sequences {
    pub tokens: Vec<u32>,
    pub offsets: Vec<i64>,
    pub slot_kind: Vec<u8>,
    pub target_index: Vec<i32>,
    pub span_target: Vec<[i32; 2]>,
    pub candidate_offsets: Vec<i64>,
    pub candidate_positions: Vec<i32>,
}

impl Sequences {
    pub fn load() -> Self {
        let raw = std::fs::read(dir().join("batches/tokens.u32")).unwrap();
        let tokens: Vec<u32> = raw.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        let offsets = NpyArray::parse(&std::fs::read(dir().join("batches/offsets.npy")).unwrap())
            .unwrap()
            .cast::<i64>("offsets")
            .unwrap();
        let zip = StoredZip::parse(&std::fs::read(dir().join("batches/supervision.npz")).unwrap()).unwrap();
        let st: Vec<i32> = zip.array("span_target").unwrap().cast::<i32>("span_target").unwrap();
        let s = Self {
            tokens,
            offsets,
            slot_kind: zip.array("slot_kind").unwrap().cast::<u8>("slot_kind").unwrap(),
            target_index: zip.array("target_index").unwrap().cast::<i32>("target_index").unwrap(),
            span_target: st.as_chunks::<2>().0.to_vec(),
            candidate_offsets: zip.array("candidate_offsets").unwrap().cast::<i64>("candidate_offsets").unwrap(),
            candidate_positions: zip.array("candidate_positions").unwrap().cast::<i32>("candidate_positions").unwrap(),
        };
        let n = s.offsets.len() - 1;
        assert_eq!(*s.offsets.last().unwrap() as usize, s.tokens.len());
        assert_eq!((s.slot_kind.len(), s.target_index.len(), s.span_target.len(), s.candidate_offsets.len()), (n, n, n, n + 1));
        s
    }

    pub fn sequence(&self, i: usize) -> &[u32] {
        &self.tokens[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    pub fn candidates(&self, i: usize) -> &[i32] {
        &self.candidate_positions[self.candidate_offsets[i] as usize..self.candidate_offsets[i + 1] as usize]
    }

    /// `plan.json`'s batch `b` as the shard reader's [`Batch`].
    pub fn batch(&self, b: &Value) -> Batch {
        let rows: Vec<usize> = b["rows"].as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize).collect();
        let width = b["width"].as_u64().unwrap() as usize;
        let lengths: Vec<i64> = b["lengths"].as_array().unwrap().iter().map(|l| l.as_i64().unwrap()).collect();
        let any_span = rows.iter().any(|&s| self.slot_kind[s] == SLOT_SPAN);
        let mut tokens = Vec::with_capacity(rows.len() * width);
        let mut line_starts = vec![false; rows.len() * width];
        for (r, &s) in rows.iter().enumerate() {
            let seq = self.sequence(s);
            assert_eq!(seq.len() as i64, lengths[r], "plan length of sequence {s}");
            tokens.extend(seq.iter().map(|&t| t as i32));
            tokens.extend(std::iter::repeat_n(PAD_ID, width - seq.len()));
            for &p in self.candidates(s) {
                line_starts[r * width + p as usize] = true;
            }
        }
        Batch {
            index: b["index"].as_u64().unwrap(),
            bucket: b["bucket"].as_u64().unwrap() as usize,
            width,
            sequences: rows.clone(),
            tokens,
            lengths,
            slot_kind: rows.iter().map(|&s| self.slot_kind[s]).collect(),
            target_index: rows.iter().map(|&s| self.target_index[s]).collect(),
            span_target: any_span.then(|| rows.iter().map(|&s| self.span_target[s]).collect()),
            line_starts: any_span.then_some(line_starts),
        }
    }

    pub fn batches(&self) -> Vec<Batch> {
        json("batches/plan.json")["batches"].as_array().unwrap().iter().map(|b| self.batch(b)).collect()
    }
}

/// Every tensor of a safetensors file as f32 (bf16 widened exactly), by name.
pub fn tensors(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
    let file = SafeTensorsFile::open(path).unwrap();
    file.tensors()
        .iter()
        .map(|(name, info)| {
            let n: usize = info.shape.iter().product();
            let values = match info.dtype {
                Dtype::F32 => {
                    let mut b = vec![0u8; n * 4];
                    file.read_tensor_range(info, 0, &mut b).unwrap();
                    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
                }
                Dtype::Bf16 => {
                    let mut b = vec![0u8; n * 2];
                    file.read_tensor_range(info, 0, &mut b).unwrap();
                    b.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect()
                }
                other => panic!("{name}: {other:?}"),
            };
            (name.clone(), (info.shape.clone(), values))
        })
        .collect()
}

/// The span head `init.safetensors` carries under `span_head.*` (f32).
pub fn init_head() -> SpanHead {
    let t = tensors(&dir().join("init.safetensors"));
    let get = |n: &str| t[&format!("span_head.{n}")].1.clone();
    let h = t["span_head.abstain_start"].0[0];
    SpanHead::new(h, get("start_proj.weight"), get("end_proj.weight"), get("abstain_start"), get("abstain_end")).unwrap()
}
