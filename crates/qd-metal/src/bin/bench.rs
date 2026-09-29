//! Prefill throughput of qd-metal on the real 2B weights.
//!
//! ```text
//! cargo run --release -p qd-metal --bin qd-metal-bench                 # T = 1024, 2048, 8192
//! cargo run --release -p qd-metal --bin qd-metal-bench -- 1024 4096    # chosen T
//! ```
//!
//! **This runs on the GPU.** What is timed is what `MetalBackend::prefill` + one answer does:
//! host embedding gather, all layers with the prefix state kept (conv, GDN, KV), final norm and
//! the 17 answer rows at the last position only — no full-vocabulary LM head. Wall clock from
//! the first host write to the scores being read back, median of `ITERS` after `WARMUP`.
//! A run whose scores are not finite is refused before anything is timed.
//!
//! The token ids are this file's own source, tokenized and repeated to length: throughput does
//! not depend on which tokens, and the bench then needs nothing outside the crate and snapshot.

use std::process::ExitCode;
use std::time::Instant;

use qd_metal::model::Model;
use qd_metal::tokenizer::QwenTokenizer;
use qd_metal::{MetalError, Result};

const WARMUP: usize = 2;
const ITERS: usize = 7;
const DEFAULT_T: [usize; 3] = [1024, 2048, 8192];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn run() -> Result<()> {
    let mut ts = Vec::new();
    for a in std::env::args().skip(1) {
        let t: usize = a
            .parse()
            .map_err(|_| MetalError::Input(format!("expected a token count, got {a:?}")))?;
        if t == 0 {
            return Err(MetalError::Input("T must be positive".into()));
        }
        ts.push(t);
    }
    if ts.is_empty() {
        ts = DEFAULT_T.to_vec();
    }
    let snapshot = qd_metal::config::resolve_snapshot(None)?;
    let tok = QwenTokenizer::load(&snapshot.join("tokenizer.json"))?;
    let base = tok.encode(include_str!("../model.rs"))?;
    if base.is_empty() {
        return Err(MetalError::Input("no tokens to repeat".into()));
    }

    let rt = tessl::GpuRuntime::new().map_err(MetalError::Gpu)?;
    rt.set_async_encode(true).map_err(MetalError::Gpu)?;
    println!("device: {}", rt.device_name());
    let t0 = Instant::now();
    let model = Model::load(&rt, &snapshot)?;
    println!("weights loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let letters = tok.letter_ids().to_vec();

    println!(
        "{:>6} {:>12} {:>12} {:>10} {:>10} {:>12}",
        "T", "median ms", "min ms", "tok/s", "launches", "state MB"
    );
    for &t in &ts {
        let ids: Vec<u32> = base.iter().copied().cycle().take(t).collect();
        let last = u32::try_from(t - 1).map_err(|_| MetalError::Input("T too large".into()))?;
        let once = || -> Result<(f64, usize)> {
            let start = Instant::now();
            let (out, state) = model.prefill(&ids)?;
            let scores = model.score(&out, &[last], &letters)?;
            let ms = start.elapsed().as_secs_f64() * 1e3;
            if let Some(bad) = scores.logits.iter().find(|x| !x.is_finite()) {
                return Err(MetalError::Gpu(format!("T={t}: non-finite answer logit {bad}")));
            }
            Ok((ms, state.nbytes()))
        };
        rt.take_dispatch_count();
        let (_, state_bytes) = once()?;
        let launches = rt.take_dispatch_count();
        for _ in 1..WARMUP {
            once()?;
        }
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            samples.push(once()?.0);
        }
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let med = median(samples);
        println!(
            "{t:>6} {med:>12.2} {min:>12.2} {:>10.0} {launches:>10} {:>12.1}",
            t as f64 / med * 1e3,
            state_bytes as f64 / 1e6
        );
    }
    println!(
        "timed: embed gather + {} layers (state kept) + final norm + 17 answer rows at the last \
         position; median of {ITERS} after {WARMUP} warm-up",
        model.config().n_layers()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("qd-metal-bench: NOT RUN: {e}");
            ExitCode::from(2)
        }
    }
}
