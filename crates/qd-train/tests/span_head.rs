//! Parity of `qd_train::span_head` with the PyTorch span head the campaign trains, and the
//! pin of its row layout to the Python plan and to `qd-runtime`'s serving convention.
//!
//! The fixtures under `tests/fixtures/span-head/` are written by
//! `tools/qd_train_oracle_span_head.py`, which runs `python/qd_train/heads.py`'s
//! `SpanPointerHead` over a plan from `plan_span_batch`, on the CPU, in float32 (the head as
//! the campaign runs it) and in float64 (the exact value, the gate's reference). See that
//! script's docstring for what each file holds and why the `[H, H]` tensors are generated or
//! encoded rather than stored.
//!
//! # The gate (ojas plan Q3 rung a, Amendment 1)
//!
//! The tolerance was written here before the first run of the implementation: "span head
//! <= 1e-6 rel (f32, host)". Amendment 1 (`AUDIT/ojas-training-2026-10-01/fable-advice.md`
//! Q3; ruling in `fable-span-head-reference.md`) fixed its **reference** without moving the
//! number. The reference is the oracle's float64 arm (`ref64_*`). For every tensor of every
//! case, `max |rust - ref64| <= 1e-6 * max |ref64|`; for the loss, `|rust - ref64| <= 1e-6 *
//! |ref64|`. All 9 cases are gated and none is dropped.
//!
//! As first written, the test also gated against torch float32 at the same 1e-6 ("gate A").
//! At `H = 2048`, torch float32's own `d_hidden` rounding is 9-11 ulps of max, the size of the
//! gate, so that comparison measured whether two roundings happened to align. It failed once,
//! at `adf6aea`; ledger row 6d6ca078 records that run and stands. The float32 arm is now a
//! report-only pin. Each case prints rust-vs-fp32 and fp32-vs-f64 per tensor, and asserts one
//! fixture-integrity fact: each manifest `torch_f32_vs_f64` equals the value recomputed here
//! from the two dumps, exactly (see [`assert_manifest_matches_dumps`]).
//!
//! There is deliberately no gate derived from torch's own float32 error. Any such gate (for
//! example rust-vs-fp32 <= 1e-6 + fp32-vs-f64) is implied by the float64 gate through the
//! triangle inequality, so it cannot fail. A check that cannot fail is not a gate.
//! Nothing here may be loosened to make a run pass (CLAUDE.md rule 2).
//!
//! # The npy reader
//!
//! Private to this test on purpose. Lane L-data owns the crate's npy reader
//! (`crates/qd-train/src/shards.rs`, Fable's Q5) and it was not merged when this was written;
//! when it lands, this one is the duplicate to delete.

mod common;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use qd_train::span_head::{
    HiddenGrad, PointerScores, RESERVED_NOUL_ROWS, SpanGold, SpanHead, SpanRowInput, SpanRowPlan,
    SpanStep, span_head_rows,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The parity tolerance, relative to each tensor's largest magnitude. See the module docs.
const PARITY_TOL: f64 = 1e-6;

/// Every case the oracle writes, each checked by the test of the same name below. The fixture
/// directory must hold exactly these: a case the oracle wrote and nothing checks would read
/// as covered.
const CASES: &[&str] = &[
    "random-h16",
    "random-h48",
    "random-h96",
    "one-candidate",
    "gold-abstain-zero-init",
    "duplicate-positions",
    "many-candidates",
    "h2048-point",
    "h2048-abstain-duplicate",
];

// ---------------------------------------------------------------------------------------------
// npy, the subset the oracle writes: version 1-3, C order, <f4 / <f8 / <i8 / |b1.
// ---------------------------------------------------------------------------------------------

enum NpyData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

struct Npy {
    shape: Vec<usize>,
    data: NpyData,
}

fn header_field<'a>(header: &'a str, key: &str, path: &Path) -> &'a str {
    let tag = format!("'{key}':");
    let at = header
        .find(&tag)
        .unwrap_or_else(|| panic!("{}: npy header has no {key}: {header}", path.display()));
    header[at + tag.len()..].trim_start()
}

fn read_npy(path: &Path) -> Npy {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "{}: not npy",
        path.display()
    );
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => panic!(
            "{}: npy version {v} is not one the oracle writes",
            path.display()
        ),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len])
        .unwrap_or_else(|e| panic!("{}: header is not utf-8: {e}", path.display()));
    let descr_rest = header_field(header, "descr", path);
    let descr = descr_rest
        .strip_prefix('\'')
        .and_then(|r| r.split('\'').next())
        .unwrap_or_else(|| panic!("{}: unreadable descr in {header}", path.display()));
    assert!(
        header_field(header, "fortran_order", path).starts_with("False"),
        "{}: Fortran order is not read here",
        path.display()
    );
    let shape_rest = header_field(header, "shape", path);
    let open = shape_rest
        .strip_prefix('(')
        .unwrap_or_else(|| panic!("{}: unreadable shape in {header}", path.display()));
    let inner = &open[..open.find(')').expect("shape tuple closes")];
    let shape: Vec<usize> = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|e| panic!("{}: shape {s}: {e}", path.display()))
        })
        .collect();
    let count: usize = shape.iter().product();
    let payload = &bytes[start + header_len..];
    let item = match descr {
        "<f4" => 4,
        "<f8" | "<i8" => 8,
        "|b1" => 1,
        other => panic!(
            "{}: dtype {other} is not one the oracle writes",
            path.display()
        ),
    };
    assert_eq!(
        payload.len(),
        count * item,
        "{}: payload is {} bytes, shape {shape:?} of {descr} needs {}",
        path.display(),
        payload.len(),
        count * item
    );
    let data = match descr {
        // The payload length was checked against the shape above, so no chunk is left over.
        "<f4" => NpyData::F32(
            payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        ),
        "<f8" => NpyData::F64(
            payload
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect(),
        ),
        "<i8" => NpyData::I64(
            payload
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect(),
        ),
        _ => NpyData::Bool(
            payload
                .iter()
                .map(|&b| match b {
                    0 => false,
                    1 => true,
                    other => panic!("{}: bool byte {other}", path.display()),
                })
                .collect(),
        ),
    };
    Npy { shape, data }
}

impl Npy {
    fn f32(self, what: &str) -> (Vec<usize>, Vec<f32>) {
        match self.data {
            NpyData::F32(v) => (self.shape, v),
            _ => panic!("{what}: expected <f4"),
        }
    }
    fn f64(self, what: &str) -> (Vec<usize>, Vec<f64>) {
        match self.data {
            NpyData::F64(v) => (self.shape, v),
            _ => panic!("{what}: expected <f8"),
        }
    }
    fn i64(self, what: &str) -> (Vec<usize>, Vec<i64>) {
        match self.data {
            NpyData::I64(v) => (self.shape, v),
            _ => panic!("{what}: expected <i8"),
        }
    }
    fn bool(self, what: &str) -> (Vec<usize>, Vec<bool>) {
        match self.data {
            NpyData::Bool(v) => (self.shape, v),
            _ => panic!("{what}: expected |b1"),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The oracle's weight generator, re-implemented (tools/qd_train_oracle_span_head.py).
// ---------------------------------------------------------------------------------------------

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn generated_weight(seed: u64, stream: u64, width: usize, shift: i32) -> Vec<f32> {
    let scale = 2f64.powi(-(23 + shift));
    (0..(width * width) as u64)
        .map(|index| {
            let u = (splitmix64((seed << 40) ^ (stream << 32) ^ index) >> 40) as i64 - (1 << 23);
            (u as f64 * scale) as f32
        })
        .collect()
}

fn sha256_hex(bytes: impl Iterator<Item = u8>) -> String {
    let mut hasher = Sha256::new();
    let buf: Vec<u8> = bytes.collect();
    hasher.update(&buf);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn sha256_f32(values: &[f32]) -> String {
    sha256_hex(values.iter().flat_map(|v| v.to_le_bytes()))
}

fn sha256_f64(values: &[f64]) -> String {
    sha256_hex(values.iter().flat_map(|v| v.to_le_bytes()))
}

// ---------------------------------------------------------------------------------------------
// One case, loaded.
// ---------------------------------------------------------------------------------------------

/// `tests/fixtures/span-head`, once every file under it matches its `SHA256SUMS` (checked once
/// per test binary, before any case is read).
fn fixture_root() -> PathBuf {
    common::pins::verified(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/span-head"),
        common::pins::Form::Sha256Sums,
    )
}

struct Case {
    name: String,
    dir: PathBuf,
    manifest: Value,
    /// The manifest as written, for the numbers that must be read without `serde_json`'s
    /// float parse (see [`Case::manifest_f32_vs_f64`]).
    manifest_text: String,
    width: usize,
    n_spans: usize,
    seq_len: usize,
    span_weight: f32,
}

impl Case {
    fn load(name: &str) -> Self {
        let dir = fixture_root().join(name);
        let text = fs::read_to_string(dir.join("manifest.json"))
            .unwrap_or_else(|e| panic!("{name}: manifest.json: {e}"));
        let manifest: Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: manifest: {e}"));
        let field = |key: &str| {
            manifest[key]
                .as_u64()
                .unwrap_or_else(|| panic!("{name}: manifest.{key} is not an integer"))
                as usize
        };
        let width = field("hidden_size");
        let n_spans = field("n_spans");
        let seq_len = field("seq_len");
        let span_weight = manifest["span_weight"]
            .as_f64()
            .unwrap_or_else(|| panic!("{name}: manifest.span_weight"))
            as f32;
        assert_eq!(
            manifest["reduction"], "mean",
            "{name}: the trainer's reduction is mean"
        );
        assert_eq!(
            manifest["dtype"], "float32",
            "{name}: the oracle's head arm is float32 (its float64 arm is ref64_*)"
        );
        Case {
            name: name.to_owned(),
            dir,
            manifest,
            manifest_text: text,
            width,
            n_spans,
            seq_len,
            span_weight,
        }
    }

    fn npy(&self, file: &str) -> Npy {
        read_npy(&self.dir.join(format!("{file}.npy")))
    }

    fn f32(&self, file: &str, shape: &[usize]) -> Vec<f32> {
        let (got, v) = self.npy(file).f32(file);
        assert_eq!(got, shape, "{}: {file} shape", self.name);
        v
    }

    fn f64(&self, file: &str, shape: &[usize]) -> Vec<f64> {
        let (got, v) = self.npy(file).f64(file);
        assert_eq!(got, shape, "{}: {file} shape", self.name);
        v
    }

    fn i64(&self, file: &str) -> (Vec<usize>, Vec<i64>) {
        self.npy(file).i64(file)
    }

    /// `torch_f32_vs_f64`, in [`TENSORS`] order, parsed from the manifest's own text with
    /// `str::parse::<f64>`, which rounds correctly. `serde_json` parses floats correctly only
    /// with its `float_roundtrip` feature. Whether that feature is on depends on which other
    /// workspace crates share the build, and an exact comparison cannot depend on that.
    fn manifest_f32_vs_f64(&self) -> [f64; 8] {
        let open = "\"torch_f32_vs_f64\": {";
        let at = self
            .manifest_text
            .find(open)
            .unwrap_or_else(|| panic!("{}: manifest has no torch_f32_vs_f64", self.name));
        let body = &self.manifest_text[at + open.len()..];
        let body = &body[..body.find('}').expect("torch_f32_vs_f64 closes")];
        let mut out = [0f64; 8];
        for (slot, tensor) in out.iter_mut().zip(TENSORS) {
            let key = format!("\"{tensor}\":");
            let from = body
                .find(&key)
                .unwrap_or_else(|| panic!("{}: torch_f32_vs_f64 has no {tensor}", self.name));
            let rest = body[from + key.len()..].trim_start();
            let end = rest.find([',', '\n']).unwrap_or(rest.len());
            let text = rest[..end].trim();
            *slot = text.parse().unwrap_or_else(|e| {
                panic!("{}: torch_f32_vs_f64.{tensor} = {text:?}: {e}", self.name)
            });
            // The structured parse must name the same number to within its own rounding; a
            // disagreement beyond that is a malformed manifest, not a float-parsing nuance.
            let structured = self.manifest["torch_f32_vs_f64"][tensor]
                .as_f64()
                .unwrap_or_else(|| panic!("{}: torch_f32_vs_f64.{tensor}", self.name));
            assert!(
                (structured - *slot).abs() <= slot.abs() * 4.0 * f64::EPSILON,
                "{}: torch_f32_vs_f64.{tensor}: text {text} vs parsed {structured:e}",
                self.name
            );
        }
        out
    }

    fn sha(&self, key: &str) -> String {
        self.manifest["sha256"][key]
            .as_str()
            .unwrap_or_else(|| panic!("{}: manifest.sha256.{key}", self.name))
            .to_owned()
    }

    /// The two projections torch ran with, regenerated and checked against the digest of the
    /// bytes torch used: a generator that drifted fails here, as wrong input.
    fn projections(&self) -> (Vec<f32>, Vec<f32>) {
        let g = &self.manifest["weight_generator"];
        let int = |key: &str| g[key].as_u64().unwrap_or_else(|| panic!("generator.{key}"));
        let shift = i32::try_from(int("shift")).expect("shift fits i32");
        let start = generated_weight(int("seed"), int("stream_start_proj"), self.width, shift);
        let end = generated_weight(int("seed"), int("stream_end_proj"), self.width, shift);
        assert_eq!(
            sha256_f32(&start),
            self.sha("start_proj_weight"),
            "{}: start_proj",
            self.name
        );
        assert_eq!(
            sha256_f32(&end),
            self.sha("end_proj_weight"),
            "{}: end_proj",
            self.name
        );
        (start, end)
    }

    /// One row of the plan exactly as `plan_span_batch` laid it out, read back.
    fn python_plan(&self) -> Vec<PythonRow> {
        let (shape, cand) = self.i64("candidate_pos");
        let (k, max_cand) = (shape[0], shape[1]);
        assert_eq!(k, self.n_spans, "{}: candidate_pos rows", self.name);
        let (vshape, valid) = self.npy("candidate_valid").bool("candidate_valid");
        assert_eq!(vshape, shape, "{}: candidate_valid shape", self.name);
        let col = |file: &str| {
            let (s, v) = self.i64(file);
            assert_eq!(s, [k], "{}: {file} shape", self.name);
            v
        };
        let n = col("n_candidates");
        let query = col("query_index");
        let gold_start = col("gold_start");
        let gold_end = col("gold_end");
        let runtime_rows = col("runtime_rows");
        let gold_start_token = col("gold_start_token");
        let gold_end_token = col("gold_end_token");
        let (ashape, abstaining) = self.npy("abstaining").bool("abstaining");
        assert_eq!(ashape, [k], "{}: abstaining shape", self.name);
        (0..k)
            .map(|r| {
                let nr = usize::try_from(n[r]).expect("n_candidates >= 0");
                PythonRow {
                    candidates: cand[r * max_cand..r * max_cand + nr]
                        .iter()
                        .map(|&p| u32::try_from(p).expect("position fits u32"))
                        .collect(),
                    valid: valid[r * max_cand..(r + 1) * max_cand].to_vec(),
                    max_cand,
                    query: u32::try_from(query[r]).expect("query fits u32"),
                    gold_start: usize::try_from(gold_start[r]).expect("gold >= 0"),
                    gold_end: usize::try_from(gold_end[r]).expect("gold >= 0"),
                    runtime_rows: usize::try_from(runtime_rows[r]).expect("rows >= 0"),
                    gold_start_token: gold_start_token[r],
                    gold_end_token: gold_end_token[r],
                    abstaining: abstaining[r],
                }
            })
            .collect()
    }
}

struct PythonRow {
    candidates: Vec<u32>,
    valid: Vec<bool>,
    max_cand: usize,
    query: u32,
    gold_start: usize,
    gold_end: usize,
    runtime_rows: usize,
    gold_start_token: i64,
    gold_end_token: i64,
    abstaining: bool,
}

impl PythonRow {
    /// The Rust plan for this row, from the head-row gold Python chose.
    fn rust_plan(&self) -> SpanRowPlan {
        let gold = if self.abstaining {
            SpanGold::Abstain
        } else {
            SpanGold::Lines {
                start: self.gold_start,
                end: self.gold_end,
            }
        };
        SpanRowPlan::new(self.query, self.candidates.clone(), gold)
            .unwrap_or_else(|e| panic!("the Rust plan refused Python's row: {e}"))
    }
}

/// `hidden[k]`'s rows at `plan.positions()`, as tessl's `PendingStep::hidden` would hand them.
fn gather(hidden: &[f32], k: usize, seq_len: usize, width: usize, plan: &SpanRowPlan) -> Vec<f32> {
    let mut out = Vec::with_capacity(plan.positions().len() * width);
    for &p in plan.positions() {
        let at = (k * seq_len + p as usize) * width;
        out.extend_from_slice(&hidden[at..at + width]);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Comparison.
// ---------------------------------------------------------------------------------------------

/// `max |rust - reference| / max |reference|` over the finite entries, or infinity when the two
/// disagree on which entries are finite (a padded column that became selectable, or a dead
/// one). A reference whose max is zero demands an exact zero.
fn rel_to_max(rust: &[f64], reference: &[f64]) -> f64 {
    assert_eq!(
        rust.len(),
        reference.len(),
        "compared tensors differ in length"
    );
    let mut scale = 0f64;
    let mut diff = 0f64;
    for (&r, &t) in rust.iter().zip(reference) {
        if r.is_finite() != t.is_finite() {
            return f64::INFINITY;
        }
        if !t.is_finite() {
            if r != t {
                return f64::INFINITY;
            }
            continue;
        }
        scale = scale.max(t.abs());
        diff = diff.max((r - t).abs());
    }
    if scale > 0.0 {
        diff / scale
    } else if diff == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// The gate: false for anything over [`PARITY_TOL`], and for NaN.
fn within_tolerance(rel: f64) -> bool {
    rel <= PARITY_TOL
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// The outputs of one run, laid out in the oracle's dense shapes.
struct Outputs {
    scores_start: Vec<f64>,
    scores_end: Vec<f64>,
    loss: Vec<f64>,
    d_hidden: Vec<f64>,
    d_start_proj: Vec<f64>,
    d_end_proj: Vec<f64>,
    d_abstain_start: Vec<f64>,
    d_abstain_end: Vec<f64>,
}

const TENSORS: [&str; 8] = [
    "scores_start",
    "scores_end",
    "loss",
    "d_hidden",
    "d_start_proj",
    "d_end_proj",
    "d_abstain_start",
    "d_abstain_end",
];

impl Outputs {
    fn get(&self, name: &str) -> &[f64] {
        match name {
            "scores_start" => &self.scores_start,
            "scores_end" => &self.scores_end,
            "loss" => &self.loss,
            "d_hidden" => &self.d_hidden,
            "d_start_proj" => &self.d_start_proj,
            "d_end_proj" => &self.d_end_proj,
            "d_abstain_start" => &self.d_abstain_start,
            "d_abstain_end" => &self.d_abstain_end,
            other => panic!("no tensor {other}"),
        }
    }
}

/// One torch arm's outputs from the fixture: `prefix` is `""` (float32) or `"ref64_"`.
fn torch_outputs(case: &Case, prefix: &str, plan: &[PythonRow], hidden: &[f32]) -> Outputs {
    let (k, l, h) = (case.n_spans, case.seq_len, case.width);
    let cols = plan[0].max_cand + RESERVED_NOUL_ROWS;
    let wide = prefix == "ref64_";
    let load = |name: &str, shape: &[usize]| -> Vec<f64> {
        let file = format!("{prefix}{name}");
        if wide {
            case.f64(&file, shape)
        } else {
            widen(&case.f32(&file, shape))
        }
    };
    let dw = |which: &str| -> Vec<f64> {
        match case.manifest["dw_encoding"].as_str() {
            Some("npy") => load(&format!("d_{which}_proj"), &[h, h]),
            Some("outer_k1") => {
                // Lossless only for K = 1, where torch's dW is one rounded product per
                // element; the oracle asserted that, and the digest below re-proves it here.
                assert_eq!(k, 1, "{}: outer_k1 needs K = 1", case.name);
                let dq = load(&format!("dq_{which}"), &[1, h]);
                let query = plan[0].query as usize;
                let hq = &hidden[query * h..(query + 1) * h];
                let key = format!("{prefix}d_{which}_proj");
                if wide {
                    let w: Vec<f64> = dq
                        .iter()
                        .flat_map(|&d| hq.iter().map(move |&x| d * f64::from(x)))
                        .collect();
                    assert_eq!(sha256_f64(&w), case.sha(&key), "{}: {key}", case.name);
                    w
                } else {
                    let w: Vec<f32> = dq
                        .iter()
                        .flat_map(|&d| hq.iter().map(move |&x| d as f32 * x))
                        .collect();
                    assert_eq!(sha256_f32(&w), case.sha(&key), "{}: {key}", case.name);
                    widen(&w)
                }
            }
            other => panic!("{}: unknown dw_encoding {other:?}", case.name),
        }
    };
    Outputs {
        scores_start: load("scores_start", &[k, cols]),
        scores_end: load("scores_end", &[k, cols]),
        // `np.ascontiguousarray` lifts torch's 0-d loss to shape (1,) on its way to disk.
        loss: load("loss", &[1]),
        d_hidden: load("d_hidden", &[k, l, h]),
        d_start_proj: dw("start"),
        d_end_proj: dw("end"),
        d_abstain_start: load("d_abstain_start", &[h]),
        d_abstain_end: load("d_abstain_end", &[h]),
    }
}

/// The Rust step's outputs in the same dense shapes: scores padded with `-inf` past each row's
/// abstention, `d_hidden` scattered from its distinct positions into `[K, L, H]`.
fn rust_outputs(
    case: &Case,
    plans: &[SpanRowPlan],
    max_cand: usize,
    step: &SpanStep,
    grads: &SpanHead,
) -> Outputs {
    let (l, h) = (case.seq_len, case.width);
    let cols = max_cand + RESERVED_NOUL_ROWS;
    let pad = |pick: fn(&PointerScores) -> &[f32]| -> Vec<f64> {
        let mut out = vec![f64::NEG_INFINITY; plans.len() * cols];
        for (k, scores) in step.scores.iter().enumerate() {
            let row = pick(scores);
            assert!(
                row.len() <= cols,
                "{}: row {k} has {} scores",
                case.name,
                row.len()
            );
            for (j, &s) in row.iter().enumerate() {
                out[k * cols + j] = f64::from(s);
            }
        }
        out
    };
    let mut d_hidden = vec![0f64; plans.len() * l * h];
    assert_eq!(
        step.d_hidden.len(),
        plans.len(),
        "{}: one hidden gradient per row",
        case.name
    );
    for (k, HiddenGrad { positions, rows }) in step.d_hidden.iter().enumerate() {
        assert_eq!(
            rows.len(),
            positions.len() * h,
            "{}: row {k} gradient shape",
            case.name
        );
        // tessl's train_backward_into takes each position once; a repeat here would also be
        // summed by a `+=` scatter and hide exactly the pre-sum this crate owes it.
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "{}: row {k} gradient positions are not distinct and ascending: {positions:?}",
            case.name
        );
        for (slot, &p) in positions.iter().enumerate() {
            let at = (k * l + p as usize) * h;
            for (dst, &g) in d_hidden[at..at + h]
                .iter_mut()
                .zip(&rows[slot * h..(slot + 1) * h])
            {
                *dst = f64::from(g);
            }
        }
    }
    let named: Vec<(&str, &[f32])> = grads.named().to_vec();
    let grad = |name: &str| -> Vec<f64> {
        widen(
            named
                .iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| panic!("no gradient named {name}"))
                .1,
        )
    };
    Outputs {
        scores_start: pad(|s| &s.start),
        scores_end: pad(|s| &s.end),
        loss: vec![f64::from(step.loss)],
        d_hidden,
        d_start_proj: grad("start_proj.weight"),
        d_end_proj: grad("end_proj.weight"),
        d_abstain_start: grad("abstain_start"),
        d_abstain_end: grad("abstain_end"),
    }
}

fn check_case(name: &str) {
    let case = Case::load(name);
    let (k, l, h) = (case.n_spans, case.seq_len, case.width);
    let (w_start, w_end) = case.projections();
    let hidden = case.f32("hidden", &[k, l, h]);
    let abstain_start = case.f32("abstain_start", &[h]);
    let abstain_end = case.f32("abstain_end", &[h]);
    let python = case.python_plan();
    let plans: Vec<SpanRowPlan> = python.iter().map(PythonRow::rust_plan).collect();
    let gathered: Vec<Vec<f32>> = plans
        .iter()
        .enumerate()
        .map(|(r, p)| gather(&hidden, r, l, h, p))
        .collect();
    let batch: Vec<SpanRowInput<'_>> = plans
        .iter()
        .zip(&gathered)
        .map(|(plan, hidden)| SpanRowInput { plan, hidden })
        .collect();

    let head = SpanHead::new(h, w_start, w_end, abstain_start, abstain_end)
        .unwrap_or_else(|e| panic!("{name}: head: {e}"));
    let mut grads = SpanHead::zeros(h).unwrap_or_else(|e| panic!("{name}: grads: {e}"));
    let step = head
        .loss_and_backward(&batch, case.span_weight, &mut grads)
        .unwrap_or_else(|e| panic!("{name}: loss_and_backward: {e}"));

    // `scores` must be the same function the loss used.
    for (r, input) in batch.iter().enumerate() {
        let alone = head
            .scores(input.plan, input.hidden)
            .unwrap_or_else(|e| panic!("{name}: scores: {e}"));
        assert_eq!(
            alone, step.scores[r],
            "{name}: row {r}: scores() differs from the step's"
        );
    }

    let rust = rust_outputs(&case, &plans, python[0].max_cand, &step, &grads);
    let torch32 = torch_outputs(&case, "", &python, &hidden);
    let torch64 = torch_outputs(&case, "ref64_", &python, &hidden);

    // Fixture integrity first: a float32 arm regenerated without its float64 arm (or the
    // reverse), or a hand-edited manifest, fails here as a fixture defect rather than as parity.
    let own = assert_manifest_matches_dumps(&case, &torch32, &torch64);

    let mut failures = Vec::new();
    println!("{name}: H={h} K={k} L={l} span_weight={}", case.span_weight);
    println!(
        "  {:<16} {:>12} {:>12} {:>12}",
        "tensor", "rust-f64", "rust-torch32", "torch32-f64"
    );
    for (tensor, own) in TENSORS.into_iter().zip(own) {
        // The gate (Amendment 1): against the exact value.
        let gate = rel_to_max(rust.get(tensor), torch64.get(tensor));
        // Report-only: the campaign's dtype, and torch's own distance from the exact value.
        let fp32 = rel_to_max(rust.get(tensor), torch32.get(tensor));
        println!("  {tensor:<16} {gate:>12.3e} {fp32:>12.3e} {own:>12.3e}");
        if !within_tolerance(gate) {
            failures.push(format!(
                "gate (rust vs torch float64) {tensor}: {gate:.3e} > {PARITY_TOL:e}"
            ));
        }
    }
    assert!(failures.is_empty(), "{name}: {}", failures.join("; "));
}

/// The manifest's `torch_f32_vs_f64`, recomputed from the `.npy` dumps and required to be
/// **exactly** equal, tensor by tensor. Returns the values, in [`TENSORS`] order.
///
/// Exact, not toleranced, because both sides are the same IEEE-754 binary64 computation on
/// bit-identical inputs. The oracle computes `max|x32 - x64| / max|x64|` over the finite
/// entries in float64. The subtraction is correctly rounded, `abs` and `max` are exact and
/// independent of order, and the division is one correctly rounded operation. That is
/// [`rel_to_max`] whenever the float64 tensor is not all zero, which no case's is. The inputs
/// are the dumps, and the encoded `dW` is reconstructed under a sha256 proof. The manifest
/// holds Python's shortest round-trip `repr`, and this reads that number's text with Rust's
/// correctly rounded parser rather than through `serde_json`, whose default float parse is not
/// guaranteed to round-trip. Any tolerance here would only hide an edit smaller than itself.
fn assert_manifest_matches_dumps(case: &Case, torch32: &Outputs, torch64: &Outputs) -> [f64; 8] {
    let recorded = case.manifest_f32_vs_f64();
    let mut out = [0f64; 8];
    for ((slot, tensor), stated) in out.iter_mut().zip(TENSORS).zip(recorded) {
        let recomputed = rel_to_max(torch32.get(tensor), torch64.get(tensor));
        assert!(
            recomputed.to_bits() == stated.to_bits(),
            "{}: manifest torch_f32_vs_f64.{tensor} is {stated:e} but the two dumps give \
             {recomputed:e}: one arm was regenerated without the other, or the manifest was \
             edited",
            case.name
        );
        *slot = recomputed;
    }
    out
}

macro_rules! parity {
    ($($test:ident => $case:literal),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                check_case($case);
            }
        )*

        /// The cases with a test above, which must be exactly [`CASES`].
        const TESTED: &[&str] = &[$($case),*];
    };
}

parity! {
    parity_random_h16 => "random-h16",
    parity_random_h48 => "random-h48",
    parity_random_h96 => "random-h96",
    parity_one_candidate => "one-candidate",
    parity_gold_abstain_zero_init => "gold-abstain-zero-init",
    parity_duplicate_positions => "duplicate-positions",
    parity_many_candidates => "many-candidates",
    parity_h2048_point => "h2048-point",
    parity_h2048_abstain_duplicate => "h2048-abstain-duplicate",
}

#[test]
fn the_fixture_set_is_exactly_the_cases_tested() {
    let on_disk: BTreeSet<String> = fs::read_dir(fixture_root())
        .expect("fixture root exists")
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    let listed: BTreeSet<String> = CASES.iter().map(|s| (*s).to_owned()).collect();
    let tested: BTreeSet<String> = TESTED.iter().map(|s| (*s).to_owned()).collect();
    // The cases, and beside them the set's pin and nothing else (the human's 2026-10-02 decision:
    // a tracked binary fixture is sha256-pinned; `crates/qd-runtime/tests/tracked_source_is_text.rs`).
    let mut expected = listed.clone();
    expected.insert(common::pins::SHA256SUMS.to_owned());
    assert_eq!(on_disk, expected, "fixture root entries vs CASES and SHA256SUMS");
    assert_eq!(tested, listed, "parity tests vs CASES");
    // H = 2048 is the real width; a fixture set without it says nothing about it.
    assert!(
        CASES.iter().any(|c| Case::load(c).width == 2048),
        "no case at the real hidden width"
    );
}

// ---------------------------------------------------------------------------------------------
// The layout: Python's plan, qd-runtime's serving convention, and this head agree.
// ---------------------------------------------------------------------------------------------

fn runtime_source(file: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../qd-runtime/src")
        .join(file);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The text of the `{ ... }` block that opens at the first `{` at or after `from`.
fn block_after(source: &str, from: usize) -> &str {
    let open = from + source[from..].find('{').expect("a block opens");
    let mut depth = 0usize;
    for (i, ch) in source[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open..=open + i];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced block");
}

#[test]
fn reserved_noul_rows_is_qd_runtime_schemas() {
    let schema = runtime_source("schema.rs");
    let decl = "pub const RESERVED_NOUL_ROWS: usize = ";
    let at = schema
        .find(decl)
        .expect("qd-runtime/src/schema.rs declares RESERVED_NOUL_ROWS");
    let rest = &schema[at + decl.len()..];
    let value: usize = rest[..rest.find(';').expect("declaration ends")]
        .trim()
        .parse()
        .expect("an integer literal");
    assert_eq!(
        value, RESERVED_NOUL_ROWS,
        "qd-runtime says {value}, span_head says {RESERVED_NOUL_ROWS}: a head trained with one \
         and served with the other moves the abstention"
    );
}

#[test]
fn span_rows_is_one_per_line_plus_the_reserved_row() {
    let answer = runtime_source("answer.rs");
    let at = answer
        .find("pub fn span_rows(context: &Context) -> usize")
        .expect("answer.rs defines span_rows");
    let body = block_after(&answer, at);
    assert!(
        body.contains("context.line_count() + RESERVED_NOUL_ROWS"),
        "answer.rs::span_rows no longer reads line_count + RESERVED_NOUL_ROWS: {body}"
    );
    for n in [1usize, 2, 7, 300] {
        assert_eq!(span_head_rows(n).expect("n >= 1"), n + RESERVED_NOUL_ROWS);
        let plan = SpanRowPlan::new(10_000, (0..n as u32).collect(), SpanGold::Abstain)
            .expect("a valid plan");
        assert_eq!(plan.runtime_rows(), n + RESERVED_NOUL_ROWS);
    }
    assert!(
        span_head_rows(0).is_err(),
        "an empty candidate set is a mapping failure"
    );
}

#[test]
fn the_abstention_is_the_last_row_and_lines_ascend_as_answer_rs_reads_them() {
    let answer = runtime_source("answer.rs");
    // The arm with a block body (`span_rows`'s one-line arm in `answer` comes first), read
    // from its opening brace so the block is the body, not the pattern's `{ .. }`.
    let pattern = "SlotSpec::Span { .. } => {";
    let at = answer
        .find(pattern)
        .expect("answer.rs has a span arm with a body");
    let arm = block_after(&answer, at + pattern.len() - 1);
    for needle in [
        "let noul_row = plan.rows - RESERVED_NOUL_ROWS;",
        "start.top == noul_row || end.top == noul_row",
        "start_line: start.top + 1,",
        "end_line: end.top + 1,",
    ] {
        assert!(
            arm.contains(needle),
            "answer.rs's span arm no longer reads `{needle}`"
        );
    }

    // This head, on a hand-made case: row j is candidate j (ascending, line j + 1), the last
    // row is the abstention, and nothing follows it. Identity projections make each score a
    // plain dot product that can be computed here independently.
    let h = 3;
    let identity: Vec<f32> = (0..h * h)
        .map(|i| if i % (h + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let abstain = vec![0.5, -1.0, 2.0];
    let head = SpanHead::new(
        h,
        identity.clone(),
        identity,
        abstain.clone(),
        abstain.clone(),
    )
    .expect("a valid head");
    let plan = SpanRowPlan::new(7, vec![1, 4, 6], SpanGold::Abstain).expect("a valid plan");
    assert_eq!(plan.positions(), [1, 4, 6, 7]);
    // Rows of hidden at positions 1, 4, 6 (candidates) and 7 (query).
    let hidden = [
        1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0,
    ];
    let scores = head.scores(&plan, &hidden).expect("scores");
    assert_eq!(scores.start, [2.0, 3.0, 5.0, 0.5 * 2.0 - 3.0 + 2.0 * 5.0]);
    assert_eq!(scores.end, scores.start);
    assert_eq!(scores.start.len(), plan.runtime_rows());
    assert_eq!(plan.noul_row(), plan.runtime_rows() - RESERVED_NOUL_ROWS);
    assert_eq!(plan.noul_row(), plan.n_candidates());
    assert_eq!(
        (plan.gold_start(), plan.gold_end()),
        (3, 3),
        "abstain golds the last row"
    );
}

#[test]
fn every_fixture_plan_is_the_python_plan_in_this_heads_layout() {
    for name in CASES {
        let case = Case::load(name);
        for (r, row) in case.python_plan().iter().enumerate() {
            let n = row.candidates.len();
            assert!(n >= 1, "{name} row {r}: no candidate");
            assert_eq!(
                row.runtime_rows,
                n + RESERVED_NOUL_ROWS,
                "{name} row {r}: runtime_rows"
            );
            assert!(
                row.valid.iter().enumerate().all(|(j, &v)| v == (j < n)),
                "{name} row {r}: candidate_valid is not exactly the first {n} columns"
            );
            assert!(
                row.candidates.windows(2).all(|w| w[0] < w[1]),
                "{name} row {r}: Python's candidates are not ascending"
            );
            if row.abstaining {
                assert_eq!(
                    (row.gold_start, row.gold_end),
                    (n, n),
                    "{name} row {r}: abstain row"
                );
                assert_eq!((row.gold_start_token, row.gold_end_token), (-2, -2));
            } else {
                assert!(
                    row.gold_start <= row.gold_end && row.gold_end < n,
                    "{name} row {r}"
                );
                // The ordinal names the line whose start token is the gold.
                assert_eq!(
                    i64::from(row.candidates[row.gold_start]),
                    row.gold_start_token
                );
                assert_eq!(i64::from(row.candidates[row.gold_end]), row.gold_end_token);
            }
            let plan = row.rust_plan();
            assert_eq!(plan.n_candidates(), n);
            assert_eq!(plan.runtime_rows(), row.runtime_rows, "{name} row {r}");
            assert_eq!(plan.noul_row(), n, "{name} row {r}");
            assert_eq!(plan.is_abstaining(), row.abstaining, "{name} row {r}");
            assert_eq!(
                (plan.gold_start(), plan.gold_end()),
                (row.gold_start, row.gold_end),
                "{name} row {r}: gold rows"
            );
            assert_eq!(plan.candidates(), row.candidates.as_slice());
            assert_eq!(plan.query_index(), row.query);
            // tessl's distinct-positions contract: every position the head reads, once each.
            let mut want: Vec<u32> = row.candidates.clone();
            want.push(row.query);
            want.sort_unstable();
            want.dedup();
            assert_eq!(
                plan.positions(),
                want.as_slice(),
                "{name} row {r}: positions"
            );
        }
    }
}
