//! Per-iteration cost of `qd_prep::linfit::fit` against its thread count, on a synthetic problem
//! shaped like a v5 control task (hashed char n-grams: 65,536 columns, ~1,800 nonzeros a row).
//!
//! The v5 OFF-arm control (attempt 1, 2026-10-04, seed 2's eval row 2af3c80d) spent 5.4 ms an
//! iteration on openjev.game (971 rows, 1.73M nonzeros) and 6.6 ms on arc.science (3,178 rows,
//! 3.78M nonzeros) at 52 threads: a floor that the arithmetic does not explain. This measures
//! where the floor comes from on the machine it runs on.
//!
//! The tolerance is 0, so every fit runs exactly `--iters` iterations, and the grid holds one
//! L2, so a call is two fits (the grid point and the refit): `2 * iters` iterations. The rounds
//! are interleaved -- each round runs every thread count once, starting one later than the last
//! round -- and the minimum over rounds is reported, with the median beside it. Every call's
//! result is checked equal to the first thread count's: the fit does not depend on the thread
//! count, and a benchmark that let it would be measuring a different fit.
//!
//! cargo run --release -p qd-prep --example linfit_bench -- --rows 971 --nnz-per-row 1782 \
//!     --iters 200 --rounds 5 --threads 1,2,4,8,18

use std::time::Instant;

use qd_prep::linfit::{Fitted, Hyper, Selection, fit};
use qd_prep::ngram::Csr;

struct Args {
    rows: usize,
    nnz_per_row: usize,
    cols: usize,
    classes: usize,
    iters: u32,
    rounds: usize,
    threads: Vec<usize>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        rows: 971,
        nnz_per_row: 1782,
        cols: 65_536,
        classes: 4,
        iters: 200,
        rounds: 5,
        threads: vec![1, 2, 4, 8],
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let int = |v: &str| v.parse::<usize>().map_err(|e| format!("{flag} {v}: {e}"));
        match flag.as_str() {
            "--rows" => a.rows = int(&value)?,
            "--nnz-per-row" => a.nnz_per_row = int(&value)?,
            "--cols" => a.cols = int(&value)?,
            "--classes" => a.classes = int(&value)?,
            "--iters" => a.iters = value.parse().map_err(|e| format!("--iters {value}: {e}"))?,
            "--rounds" => a.rounds = int(&value)?,
            "--threads" => {
                a.threads = value.split(',').map(int).collect::<Result<_, _>>()?;
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if a.rows < 8 || a.cols == 0 || a.classes < 2 || a.rounds == 0 || a.threads.is_empty() {
        return Err("need --rows >= 8, --cols > 0, --classes >= 2, --rounds > 0, --threads".into());
    }
    if a.nnz_per_row == 0 || a.nnz_per_row > a.cols {
        return Err(format!("--nnz-per-row must be in 1..={}", a.cols));
    }
    if a.threads.contains(&0) {
        return Err("a thread count of 0".into());
    }
    Ok(a)
}

/// SplitMix64: a fixed, dependency-free stream, so every run builds the same problem.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Rows of distinct hashed columns, L2-normalised like the n-gram hasher's rows, with a label
/// carried by a few class-specific columns so the fit has something to learn.
fn problem(a: &Args) -> (Csr, Vec<u32>, Vec<f64>) {
    let mut s = 0x5EED_u64;
    let mut indptr = vec![0usize];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    let mut y = Vec::with_capacity(a.rows);
    for r in 0..a.rows {
        let label = r % a.classes;
        y.push(label as u32);
        let mut cols: Vec<u32> = (0..a.nnz_per_row)
            .map(|_| (mix(&mut s) % a.cols as u64) as u32)
            .collect();
        cols.extend((0..4).map(|j| ((label * 4 + j) % a.cols) as u32));
        cols.sort_unstable();
        cols.dedup();
        let norm = (cols.len() as f64).sqrt();
        for c in cols {
            indices.push(c);
            data.push(1.0 / norm);
        }
        indptr.push(indices.len());
    }
    let w0 = (0..a.cols * a.classes)
        .map(|i| ((i as f64) * 0.618).sin() * 0.01)
        .collect();
    let x = Csr {
        n_cols: a.cols,
        indptr,
        indices,
        data,
    };
    (x, y, w0)
}

fn main() {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("linfit_bench: {e}");
            std::process::exit(64);
        }
    };
    let (x, y, w0) = problem(&a);
    let n_val = (a.rows / 5).max(1);
    let order: Vec<usize> = {
        let mut s = 0xC0FFEE_u64;
        let mut o: Vec<usize> = (0..a.rows).collect();
        for i in (1..o.len()).rev() {
            o.swap(i, (mix(&mut s) % (i as u64 + 1)) as usize);
        }
        o
    };
    let hyper = Hyper {
        n_classes: a.classes,
        max_iter: a.iters,
        tol: 0.0,
        lr: 0.05,
        l2_grid: vec![1e-4],
    };
    let iterations = 2.0 * f64::from(a.iters);
    println!(
        "problem: {} rows x {} cols, {} nonzeros, {} classes; {} iterations a call; {} rounds",
        a.rows,
        a.cols,
        x.indices.len(),
        a.classes,
        iterations,
        a.rounds
    );

    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); a.threads.len()];
    let mut reference: Option<Fitted> = None;
    for round in 0..a.rounds {
        for step in 0..a.threads.len() {
            let slot = (round + step) % a.threads.len();
            let threads = a.threads[slot];
            let start = Instant::now();
            let got = fit(
                &x,
                &y,
                &order,
                n_val,
                &w0,
                &x,
                &hyper,
                Selection::Top1,
                threads,
            )
            .unwrap_or_else(|e| {
                eprintln!("linfit_bench: fit refused: {e}");
                std::process::exit(1);
            });
            let ms = start.elapsed().as_secs_f64() * 1e3 / iterations;
            match &reference {
                None => reference = Some(got),
                Some(first) if *first != got => {
                    eprintln!("linfit_bench: {threads} threads fitted a different result");
                    std::process::exit(1);
                }
                Some(_) => {}
            }
            samples[slot].push(ms);
        }
    }
    for (threads, ms) in a.threads.iter().zip(&mut samples) {
        ms.sort_by(f64::total_cmp);
        println!(
            "threads {threads:>3}: min {:.3} ms/iteration, median {:.3} (n={})",
            ms[0],
            ms[ms.len() / 2],
            ms.len()
        );
    }
}
