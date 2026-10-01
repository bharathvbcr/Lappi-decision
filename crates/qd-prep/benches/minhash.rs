//! `cargo bench -p qd-prep --bench minhash`: min-of-N wall time for signing a corpus-shaped
//! request, on one thread and on every core.
//!
//! The shape is the phase-3 `code.defect_class` corpus as measured on 2026-09-30: ~50,000
//! content units, a mean of 47 five-token shingles each, 128 permutations. The Python side of
//! the A/B -- the reference `MinHasher` against this binary on real rows, interleaved -- is
//! `python/tests/test_qd_prep_parity.py::test_benchmark_native_against_the_reference`.

use std::time::Instant;

use qd_prep::minhash::{MinHasher, ShingleSet, sign_all};

const UNITS: usize = 50_000;
const REPEATS: usize = 5;

fn main() {
    let owned: Vec<Vec<Vec<u8>>> = (0..UNITS)
        .map(|u| {
            let n = 20 + (u * 7919) % 55; // 20..75, mean ~47
            (0..n)
                .map(|j| format!("tok{u} a{j} b{} c d", j * 3).into_bytes())
                .collect()
        })
        .collect();
    let sets: Vec<ShingleSet<'_>> = owned
        .iter()
        .map(|s| s.iter().map(Vec::as_slice).collect())
        .collect();
    let hasher = MinHasher::new(128, "20260919").expect("hasher");
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut reference: Option<Vec<u64>> = None;
    for threads in [1, cores] {
        let mut best = f64::INFINITY;
        for _ in 0..REPEATS {
            let t = Instant::now();
            let sigs = sign_all(&hasher, &sets, threads).expect("sign");
            best = best.min(t.elapsed().as_secs_f64());
            match &reference {
                None => reference = Some(sigs),
                Some(r) => assert_eq!(r, &sigs, "threads={threads} changed the output"),
            }
        }
        println!(
            "qd-prep minhash: {UNITS} sets, 128 perms, threads={threads}: min of {REPEATS} = \
             {best:.3} s ({:.1} us/set)",
            best * 1e6 / UNITS as f64
        );
    }
}
