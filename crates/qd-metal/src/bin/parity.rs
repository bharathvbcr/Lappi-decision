//! Parity gate: qd-metal on the real 2B weights against the torch CPU fp32 reference that
//! `tools/dump_torch_reference.py` wrote.
//!
//! ```text
//! cargo run --release -p qd-metal --bin qd-metal-parity -- --fixtures <dir> [--ledger <file>]
//! cargo run --release -p qd-metal --bin qd-metal-parity -- --fixtures <dir> --tokenizer-only
//! ```
//!
//! `--tokenizer-only` runs the CPU half (Rust tokenizer ids == the Python ids for every prompt
//! and letter) and stops before the GPU is opened.
//!
//! **This runs on the GPU.** It exits 0 only if every threshold below holds, 1 if a threshold
//! failed, 2 if the gate could not run. The thresholds were written into this file before the
//! first run and are not tuned to its results (CLAUDE.md rule 2).
//!
//! Two references, one model: `fp32` is transformers in fp32 (what the gate names); `bf16in` is
//! the same with every `nn.Linear` input rounded to bf16, which is tessl's GEMM numerics. A
//! difference from `bf16in` is wiring; a difference from `fp32` is also rounding. The tight
//! bounds are against `bf16in`, the loose ones against `fp32`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use qd_metal::model::Model;
use qd_metal::tokenizer::QwenTokenizer;
use qd_metal::{MetalError, Result};
use serde_json::{json, Value};

// ---- thresholds, fixed before the first run ----------------------------------------------------
/// Per layer, per prompt: `||metal - bf16in|| / ||bf16in||` over the dumped rows.
const TIGHT_LAYER_REL_L2: f64 = 2e-2;
/// Per layer, per prompt: `||metal - fp32|| / ||fp32||`.
const LOOSE_LAYER_REL_L2: f64 = 5e-2;
/// Per prompt: max |Δ log-softmax| over the 17 letter rows, vs bf16in.
const TIGHT_LOGPROB_ABS: f64 = 0.10;
/// Per prompt: max |Δ log-softmax| over the 17 letter rows, vs fp32.
const LOOSE_LOGPROB_ABS: f64 = 0.25;
/// Fraction of prompts whose 17-row argmax matches fp32's.
const ARGMAX_MIN_AGREEMENT: f64 = 0.95;
/// A prompt whose argmax disagrees with fp32 must have fp32's own top-2 margin below this (nats).
const ARGMAX_DISAGREE_MAX_REF_MARGIN: f64 = 0.10;
/// Product path (prefill the prefix, continue the suffix from the snapshot) vs one pass over the
/// whole prompt on the same backend: max |Δ log-softmax|.
const DECODE_PATH_LOGPROB_ABS: f64 = 0.05;
/// Fewer prompts than this and the gate does not run.
const MIN_PROMPTS: usize = 20;

struct Args {
    fixtures: PathBuf,
    snapshot: Option<PathBuf>,
    ledger: Option<PathBuf>,
    /// Stop after the CPU tokenizer check; nothing touches the GPU.
    tokenizer_only: bool,
}

fn parse_args() -> std::result::Result<Args, String> {
    let mut fixtures = None;
    let mut snapshot = None;
    let mut ledger = None;
    let mut tokenizer_only = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--fixtures" => fixtures = Some(PathBuf::from(val()?)),
            "--snapshot" => snapshot = Some(PathBuf::from(val()?)),
            "--ledger" => ledger = Some(PathBuf::from(val()?)),
            "--tokenizer-only" => tokenizer_only = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        fixtures: fixtures.ok_or("--fixtures <dir> is required")?,
        snapshot,
        ledger,
        tokenizer_only,
    })
}

fn f32_list(v: &Value, key: &str) -> Result<Vec<f32>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| MetalError::Input(format!("manifest: no {key}")))?
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| MetalError::Input(format!("{key}: not a number"))))
        .collect()
}

fn u32_list(v: &Value, key: &str) -> Result<Vec<u32>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| MetalError::Input(format!("manifest: no {key}")))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|u| u32::try_from(u).ok())
                .ok_or_else(|| MetalError::Input(format!("{key}: not a u32")))
        })
        .collect()
}

fn str_field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| MetalError::Input(format!("manifest: no {key}")))
}

fn read_hidden(path: &Path, layers: usize, rows: usize, hidden: usize) -> Result<Vec<f32>> {
    let a = tessl::npy::read_npy(path).map_err(|e| MetalError::Input(format!("{}: {e}", path.display())))?;
    if a.shape != [layers, rows, hidden] {
        return Err(MetalError::Input(format!(
            "{}: shape {:?}, expected [{layers}, {rows}, {hidden}]",
            path.display(),
            a.shape
        )));
    }
    Ok(a.f32_slice().map_err(MetalError::Input)?.to_vec())
}

/// (||a - b|| / ||b||, max |a - b|, max |b|)
fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut d2, mut b2, mut dmax, mut bmax) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        d2 += d * d;
        b2 += f64::from(y) * f64::from(y);
        dmax = dmax.max(d.abs());
        bmax = bmax.max(f64::from(y).abs());
    }
    ((d2 / b2.max(f64::MIN_POSITIVE)).sqrt(), dmax, bmax)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn top2_margin(v: &[f32]) -> f64 {
    let mut s: Vec<f64> = v.iter().map(|&x| f64::from(x)).collect();
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] - s.get(1).copied().unwrap_or(f64::NEG_INFINITY)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
        .fold(0.0, f64::max)
}

#[derive(Default, Clone, Copy)]
struct LayerWorst {
    rel_bf16in: f64,
    rel_fp32: f64,
    abs_fp32: f64,
    ref_max: f64,
}

fn run(args: &Args) -> Result<bool> {
    let manifest_path = args.fixtures.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .map_err(|e| MetalError::Input(format!("{}: {e}", manifest_path.display())))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| MetalError::Input(format!("manifest: {e}")))?;
    let prompts = manifest
        .get("prompts")
        .and_then(Value::as_array)
        .ok_or_else(|| MetalError::Input("manifest: no prompts".into()))?;
    if prompts.len() < MIN_PROMPTS {
        return Err(MetalError::Input(format!(
            "{} prompts; the gate needs at least {MIN_PROMPTS}",
            prompts.len()
        )));
    }
    let letter_ids_ref = u32_list(&manifest, "letter_ids")?;

    let snapshot = qd_metal::config::resolve_snapshot(args.snapshot.as_deref())?;
    let tok = QwenTokenizer::load(&snapshot.join("tokenizer.json"))?;
    // CPU half of the gate: the Rust tokenizer must reproduce the Python ids exactly.
    if tok.letter_ids()[..] != letter_ids_ref[..] {
        return Err(MetalError::Tokenizer(format!(
            "letter ids {:?} differ from the reference's {letter_ids_ref:?}",
            tok.letter_ids()
        )));
    }
    let mut tokenizer_mismatch = Vec::new();
    for p in prompts {
        let text = format!("{}{}", str_field(p, "prefix")?, str_field(p, "suffix")?);
        if tok.encode(&text)? != u32_list(p, "ids")? {
            tokenizer_mismatch.push(str_field(p, "id")?.to_string());
        }
    }
    println!(
        "tokenizer: {} of {} prompts tokenize identically to the Python reference; letter ids {:?}",
        prompts.len() - tokenizer_mismatch.len(),
        prompts.len(),
        tok.letter_ids()
    );
    if args.tokenizer_only {
        if !tokenizer_mismatch.is_empty() {
            println!("tokenizer ids differ from Python on {tokenizer_mismatch:?}");
        }
        println!("--tokenizer-only: the GPU half of the gate is NOT RUN");
        return Ok(tokenizer_mismatch.is_empty());
    }

    let rt = tessl::GpuRuntime::new().map_err(MetalError::Gpu)?;
    rt.set_async_encode(true).map_err(MetalError::Gpu)?;
    println!("device: {}", rt.device_name());
    let t0 = Instant::now();
    let model = Model::load(&rt, &snapshot)?;
    println!("weights loaded in {:.1} s; weight hash {}", t0.elapsed().as_secs_f64(), model.weight_hash());
    let (n_layers, hidden) = (model.config().n_layers(), model.config().hidden);
    let letters = tok.letter_ids().to_vec();

    let mut worst = vec![LayerWorst::default(); n_layers];
    let mut prompt_rows = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut agree = 0usize;
    for (pi, p) in prompts.iter().enumerate() {
        let id = str_field(p, "id")?;
        let ids = u32_list(p, "ids")?;
        let keep: Vec<usize> = u32_list(p, "rows")?.into_iter().map(|r| r as usize).collect();
        let ref_fp32 = read_hidden(&args.fixtures.join(format!("hidden_{pi:02}_fp32.npy")), n_layers, keep.len(), hidden)?;
        let ref_bf = read_hidden(&args.fixtures.join(format!("hidden_{pi:02}_bf16in.npy")), n_layers, keep.len(), hidden)?;

        let mut got: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let mut observe = |_layer: usize, resid: &[f32]| -> Result<()> {
            let mut rows = Vec::with_capacity(keep.len() * hidden);
            for &r in &keep {
                rows.extend_from_slice(&resid[r * hidden..(r + 1) * hidden]);
            }
            got.push(rows);
            Ok(())
        };
        let t = u32::try_from(ids.len()).map_err(|_| MetalError::Input("prompt too long".into()))?;
        let out = model.run(&ids, 1, t, None, false, Some(&mut observe))?;
        let whole = model.score(&out, &[t - 1], &letters)?;
        drop(out);

        let mut layer_rows = Vec::new();
        let per = keep.len() * hidden;
        for l in 0..n_layers {
            let (rb, _, _) = compare(&got[l], &ref_bf[l * per..(l + 1) * per]);
            let (rf, af, mf) = compare(&got[l], &ref_fp32[l * per..(l + 1) * per]);
            let w = &mut worst[l];
            w.rel_bf16in = w.rel_bf16in.max(rb);
            w.rel_fp32 = w.rel_fp32.max(rf);
            w.abs_fp32 = w.abs_fp32.max(af);
            w.ref_max = w.ref_max.max(mf);
            if rb > TIGHT_LAYER_REL_L2 {
                failures.push(format!("{id} layer {l}: rel L2 vs bf16in {rb:.3e} > {TIGHT_LAYER_REL_L2:e}"));
            }
            if rf > LOOSE_LAYER_REL_L2 {
                failures.push(format!("{id} layer {l}: rel L2 vs fp32 {rf:.3e} > {LOOSE_LAYER_REL_L2:e}"));
            }
            layer_rows.push(json!({"layer": l, "rel_l2_bf16in": rb, "rel_l2_fp32": rf, "max_abs_fp32": af, "ref_max_abs": mf}));
        }

        let lp_fp32 = f32_list(p, "logprobs_fp32")?;
        let lp_bf = f32_list(p, "logprobs_bf16in")?;
        let d_bf = max_abs_diff(&whole.logprobs, &lp_bf);
        let d_fp = max_abs_diff(&whole.logprobs, &lp_fp32);
        if d_bf > TIGHT_LOGPROB_ABS {
            failures.push(format!("{id}: letter logprob max |d| vs bf16in {d_bf:.4} > {TIGHT_LOGPROB_ABS}"));
        }
        if d_fp > LOOSE_LOGPROB_ABS {
            failures.push(format!("{id}: letter logprob max |d| vs fp32 {d_fp:.4} > {LOOSE_LOGPROB_ABS}"));
        }
        let (am, ar) = (argmax(&whole.logprobs), argmax(&lp_fp32));
        let margin = top2_margin(&lp_fp32);
        if am == ar {
            agree += 1;
        } else if margin >= ARGMAX_DISAGREE_MAX_REF_MARGIN {
            failures.push(format!(
                "{id}: argmax {am} vs fp32 {ar} with a reference margin of {margin:.3} nats"
            ));
        }

        // The product path: prefill the prefix, continue the suffix from its snapshot.
        let prefix_tokens = p
            .get("prefix_tokens")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| MetalError::Input("manifest: no prefix_tokens".into()))?;
        let (_, state) = model.prefill(&ids[..prefix_tokens])?;
        let suffix = &ids[prefix_tokens..];
        let s = u32::try_from(suffix.len()).map_err(|_| MetalError::Input("suffix too long".into()))?;
        let cont = model.run(suffix, 1, s, Some(&state), false, None)?;
        let split = model.score(&cont, &[s - 1], &letters)?;
        let d_path = max_abs_diff(&split.logprobs, &whole.logprobs);
        if d_path > DECODE_PATH_LOGPROB_ABS {
            failures.push(format!("{id}: snapshot path vs whole prompt max |d| {d_path:.4} > {DECODE_PATH_LOGPROB_ABS}"));
        }
        let worst_bf = layer_rows
            .iter()
            .filter_map(|r| r["rel_l2_bf16in"].as_f64())
            .fold(0.0, f64::max);
        println!(
            "prompt {pi:2} {id:>28} T={:5} (prefix {prefix_tokens:5}): worst layer rel L2 vs bf16in {worst_bf:.2e}; \
             letters max|d| vs bf16in {d_bf:.4} vs fp32 {d_fp:.4}; argmax {am} ref {ar} (ref margin {margin:.2}); \
             snapshot path max|d| {d_path:.4}",
            ids.len()
        );
        prompt_rows.push(json!({
            "id": id, "tokens": ids.len(), "prefix_tokens": prefix_tokens,
            "letter_logprob_maxabs_bf16in": d_bf, "letter_logprob_maxabs_fp32": d_fp,
            "argmax_metal": am, "argmax_fp32": ar, "ref_margin_fp32": margin,
            "snapshot_path_maxabs": d_path, "layers": layer_rows,
        }));
    }

    let agreement = agree as f64 / prompts.len() as f64;
    if agreement < ARGMAX_MIN_AGREEMENT {
        failures.push(format!("argmax agreement {agree}/{} < {ARGMAX_MIN_AGREEMENT}", prompts.len()));
    }
    if !tokenizer_mismatch.is_empty() {
        failures.push(format!("tokenizer ids differ from Python on {tokenizer_mismatch:?}"));
    }

    println!("\nper layer, worst over {} prompts", prompts.len());
    println!("{:>5} {:>14} {:>14} {:>14} {:>14}", "layer", "relL2 bf16in", "relL2 fp32", "max|d| fp32", "max|ref|");
    for (l, w) in worst.iter().enumerate() {
        println!(
            "{l:>5} {:>14.3e} {:>14.3e} {:>14.3e} {:>14.3e}",
            w.rel_bf16in, w.rel_fp32, w.abs_fp32, w.ref_max
        );
    }
    println!(
        "\nargmax agreement with fp32: {agree}/{} ({:.1}%)",
        prompts.len(),
        100.0 * agreement
    );
    let pass = failures.is_empty();
    println!("gate: {}", if pass { "PASS" } else { "FAIL" });
    for f in &failures {
        println!("  {f}");
    }

    if let Some(ledger) = &args.ledger {
        let thresholds = BTreeMap::from([
            ("tight_layer_rel_l2_vs_bf16in", TIGHT_LAYER_REL_L2),
            ("loose_layer_rel_l2_vs_fp32", LOOSE_LAYER_REL_L2),
            ("tight_logprob_abs_vs_bf16in", TIGHT_LOGPROB_ABS),
            ("loose_logprob_abs_vs_fp32", LOOSE_LOGPROB_ABS),
            ("argmax_min_agreement", ARGMAX_MIN_AGREEMENT),
            ("argmax_disagree_max_ref_margin", ARGMAX_DISAGREE_MAX_REF_MARGIN),
            ("snapshot_path_logprob_abs", DECODE_PATH_LOGPROB_ABS),
        ]);
        let mut row = json!({
            "kind": "qd-metal-parity",
            "device": rt.device_name(),
            "weight_hash": model.weight_hash(),
            "tokenizer_hash": tok.hash(),
            "fixtures_manifest_sha256": qd_runtime::hex(&qd_runtime::sha256(&manifest_bytes)),
            "reference": {"torch": manifest.get("torch"), "device": manifest.get("device")},
            "n_prompts": prompts.len(),
            "thresholds": thresholds,
            "argmax_agreement": agreement,
            "gate": if pass { "pass" } else { "fail" },
            "failures": failures,
            "per_layer_worst": worst.iter().enumerate().map(|(l, w)| json!({
                "layer": l, "rel_l2_bf16in": w.rel_bf16in, "rel_l2_fp32": w.rel_fp32,
                "max_abs_fp32": w.abs_fp32, "ref_max_abs": w.ref_max})).collect::<Vec<_>>(),
            "prompts": prompt_rows,
        });
        let id = qd_runtime::hex(&qd_runtime::sha256(row.to_string().as_bytes()));
        row["row_id"] = json!(format!("qdm-parity-{}", &id[..16]));
        append_line(ledger, &row.to_string())?;
        println!("ledger row {} appended to {}", row["row_id"], ledger.display());
    }
    Ok(pass)
}

/// One `write` of one line with `O_APPEND`, then `fsync` (CLAUDE.md: never read-modify-write).
fn append_line(path: &Path, line: &str) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| MetalError::Input(format!("{}: {e}", path.display())))?;
    let mut buf = line.as_bytes().to_vec();
    buf.push(b'\n');
    f.write_all(&buf)
        .and_then(|()| f.sync_all())
        .map_err(|e| MetalError::Input(format!("{}: {e}", path.display())))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("qd-metal-parity: {e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("qd-metal-parity: gate NOT RUN: {e}");
            ExitCode::from(2)
        }
    }
}
