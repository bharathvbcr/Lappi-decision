//! Prefill throughput of qd-metal on the real 2B weights, and the latency of one decision.
//!
//! ```text
//! cargo run --release -p qd-metal --bin qd-metal-bench                 # T = 1024, 2048, 8192
//! cargo run --release -p qd-metal --bin qd-metal-bench -- 1024 4096    # chosen T
//! cargo run --release -p qd-metal --bin qd-metal-bench -- --decision T=512,2048,8192 k=4 \
//!     --ledger ledger/mac-qd-metal-<date>.jsonl                        # writes a quick row
//! ```
//!
//! `--decision` is [`qd_metal::decision`]: prefill + 2 read-only passes + 3 state digests +
//! readback, interleaved over `--arms` (default and only arm `product`; an A/B adds an arm per
//! flag it varies), min-of-7 and median, `tessl::infer_trace` counts, one `throughput` row.
//! Exit 0 when the row says completed and every sample agreed bit for bit, 1 when it records a
//! failed check, 2 when it could not run.
//!
//! The rest of this header is the prefill-only mode, which prints and writes no row.
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

/// `--decision`: returns whether the row records completed with every arm bit-identical.
fn run_decision(argv: &[String]) -> Result<bool> {
    use qd_metal::decision::{self, RowTarget};
    use qd_metal::ledger::{Provenance, TreeState};

    let started = Instant::now();
    let args = decision::parse_args(argv)?;
    let snapshot = qd_metal::config::resolve_snapshot(args.snapshot.as_deref())?;
    let tok = QwenTokenizer::load(&snapshot.join("tokenizer.json"))?;
    let answers = tok.answer_ids(args.k + 1)?;
    // Everything that can be refused on the host is refused before the GPU opens.
    let prompts = args
        .ts
        .iter()
        .map(|&t| decision::build_prompt(&tok, t, args.k))
        .collect::<Result<Vec<_>>>()?;
    let provenance = Provenance::of(None)?;
    println!("tessl: {}", provenance.tessl.describe());

    let rt = tessl::GpuRuntime::new().map_err(MetalError::Gpu)?;
    rt.set_async_encode(true).map_err(MetalError::Gpu)?;
    tessl::infer_trace::set_enabled(true);
    println!("device: {}", rt.device_name());
    let t0 = Instant::now();
    let model = Model::load(&rt, &snapshot)?;
    let load_s = t0.elapsed().as_secs_f64();
    println!("weights loaded in {load_s:.1} s; weight hash {}", model.weight_hash());

    let mut results = Vec::with_capacity(prompts.len());
    let mut tessl_after = provenance.tessl.clone();
    for p in &prompts {
        let before = TreeState::read(&provenance.tessl.dir)?;
        let r = decision::run_t(&model, p, &answers, &args.arms, args.warmup, args.iters)?;
        tessl_after = TreeState::read(&provenance.tessl.dir)?;
        let moved = before != provenance.tessl || tessl_after != before;
        println!(
            "T={:>5} prefix {:>5} tok, passes {}+{} tok",
            p.target,
            p.prefix.len(),
            p.passes[0].len(),
            p.passes[1].len()
        );
        for a in &r.arms {
            let total: Vec<f64> = a.samples.iter().map(|s| s.total_ms).collect();
            let pick = |f: &dyn Fn(&decision::Sample) -> f64| {
                decision::median(&a.samples.iter().map(f).collect::<Vec<_>>())
            };
            let first = a.samples.first();
            println!(
                "  {:<14} total min {:8.2} ms  median {:8.2} ms | median prefill {:8.2}  decode {:7.2}  digest {:7.2} | dispatches {:>6} commits {:>4} sync-wait {:>8} us",
                a.arm.name(),
                decision::min(&total),
                decision::median(&total),
                pick(&|s: &decision::Sample| s.prefill_ms),
                pick(&|s: &decision::Sample| s.decode_ms[0] + s.decode_ms[1]),
                pick(&|s: &decision::Sample| s.digest_ms.iter().sum()),
                first.map_or(0, |s| s.counts.dispatches),
                first.map_or(0, |s| s.counts.commits),
                first.map_or(0, |s| s.counts.sync_wait_us),
            );
        }
        let (same, n_same, n) = decision::bit_identical(&r);
        println!("  arms bit-identical: {same} ({n_same}/{n} samples)");
        results.push(r);
        if moved {
            println!("  tessl changed during this A/B: it is discarded; the row records status failed");
            break;
        }
    }

    let ctx = decision::RunContext {
        device: rt.device_name(),
        snapshot: snapshot.clone(),
        vocab: model.config().vocab,
        weight_hash: model.weight_hash().to_string(),
        tokenizer_hash: tok.hash().to_string(),
        load_s,
        wall_clock_s: started.elapsed().as_secs_f64(),
        provenance,
        tessl_after,
    };
    let row = decision::build_row(&args, &results, &ctx)?;
    // `build_row` owns "completed" (tessl held, every recipe T ran, arms bit-identical); the exit
    // code reads it rather than restating the rule.
    let ok = row.status == qd_train::ledger::Status::Completed;
    match &args.row {
        RowTarget::Ledger(path) => {
            let stamp = qd_metal::ledger::write_row(path, &row)?;
            println!("ledger row {} appended to {}", stamp.row_id, path.display());
        }
        RowTarget::None => println!("--no-ledger: NO ROW WRITTEN; these numbers are not citable"),
    }
    Ok(ok)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("--decision") {
        return match run_decision(&argv[1..]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::from(1),
            Err(e) => {
                eprintln!("qd-metal-bench --decision: NOT RUN: {e}");
                ExitCode::from(2)
            }
        };
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("qd-metal-bench: NOT RUN: {e}");
            ExitCode::from(2)
        }
    }
}
