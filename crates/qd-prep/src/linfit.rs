//! `qd_train.baseline.LinearBaseline.fit`, evaluated in Rust.
//!
//! The FT linear control is multinomial logistic regression on hashed char n-grams, trained by
//! Adam with an L2 penalty selected on a validation slice of the training data and refitted on
//! all of it (`LinearBaseline.fit` / `_train_once`). In Python that fit, with the hashing, was
//! the measured CPU bottleneck of `tools/ft_linear_control.py`.
//!
//! The Python stays the reference and owns everything that is a *choice*: the features, the
//! classes, the seed's permutation and initial weights (both drawn by numpy and handed in), the
//! L2 grid, the tolerance, the learning rate and the iteration budget. This module owns only the
//! arithmetic, and it is written to be the reference's arithmetic **operation for operation**
//! as the reference runs it on its sparse operand (`CSR.matmul` / `CSR.rmatmul`, which is what
//! `CSR.as_operand` keeps above its byte budget):
//!
//! - `X @ W`: `np.bincount(rows, weights=data * W[indices, c])` adds each row's products in
//!   column order from zero -- here, one thread per row, the same products in the same order;
//! - `X.T @ D`: `np.bincount(indices, weights=data * D[rows, c])` adds each column's products in
//!   row order -- here, a transposed copy whose entries are in row order, one thread per column;
//! - the softmax's row sums, `sum(W * W)`, `sum(gW * gW)`, `sum(gb * gb)` and the loss's
//!   `sum(Y * log P)` are numpy pairwise sums ([`crate::pairwise`]); `diff.sum(axis=0)` is a
//!   sequential sum down each column;
//! - every elementwise step keeps the reference's operand order -- `(m*b1) + ((1-b1)*g)`,
//!   `(lr*mhat) / (sqrt(vhat)+eps)`, `rmatmul(diff) + (2*l2)*W` -- and nothing is fused into a
//!   multiply-add; `exp`, `ln` and `powf` are the platform libm's, which numpy's float64 `exp`,
//!   `log` and Python's `**` also call (measured on the Mac, see the handoff).
//!
//! So each output element is produced by exactly one thread, in a fixed order: the result does
//! not depend on the thread count or on scheduling, and on the Mac it is the reference's sparse
//! path bit for bit (`python/tests/test_qd_prep_linear_parity.py`). Against the reference's
//! *dense* operand (a BLAS GEMM, used below `--dense-budget-gb`) it differs by rounding, as the
//! reference's own two operands already differ (`test_the_fit_is_unchanged_by_densification`).
//!
//! The same fit serves the per-option control (`qd_train.option_control.OptionScorer.fit`): a
//! two-class `_train_once`, with only the grid's accuracy counted differently -- by the best
//! option of each validation row ([`Selection::BestOption`]) rather than per example.

use std::collections::HashSet;
use std::sync::{Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::ngram::Csr;
use crate::pairwise::pairwise_sum;
use crate::team::{Job, SharedF64s, with_team};

/// `_train_once`'s Adam constants, fixed there and here.
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
/// `np.clip(P, 1e-12, None)` inside the loss.
const P_FLOOR: f64 = 1e-12;
/// `qd_train.baseline.LR_CONSTANT_ITERS` and `LR_HALVING_PERIOD`, fixed there and here: the
/// step is `lr` for this many iterations -- so every fit that converges within the old
/// 6,000-iteration budget is unchanged bit for bit -- and halves every `LR_HALVING_PERIOD`
/// after. Adam at a constant step can settle into a limit cycle that never meets the
/// tolerance (F seed 0's intent.domain control, c89b89a1); the cycle's amplitude scales with
/// the step.
pub const LR_CONSTANT_ITERS: u32 = 6_000;
pub const LR_HALVING_PERIOD: u32 = 500;

/// `qd_train.baseline.step_size(lr, step)`: `lr * 0.5 ** n` with `n` halvings. `0.5^n` is a
/// power of two, exact down to the subnormals and zero past them, as Python's float `**` is,
/// so the product is the reference's bit for bit.
pub fn step_size(lr: f64, step: u32) -> f64 {
    if step <= LR_CONSTANT_ITERS {
        return lr;
    }
    let halvings = (step - LR_CONSTANT_ITERS - 1) / LR_HALVING_PERIOD + 1;
    // At most MAX_ITER / LR_HALVING_PERIOD = 20,000 halvings, well inside i32.
    lr * 0.5f64.powi(halvings as i32)
}
/// Work items per thread, so threads that finish early take more: the Mac's efficiency cores
/// are slower than its performance cores and n-gram rows differ in length.
const CHUNKS_PER_THREAD: usize = 8;

/// What the reference's `LinearBaseline.__init__` holds, for one fit.
#[derive(Clone, Debug)]
pub struct Hyper {
    pub n_classes: usize,
    pub max_iter: u32,
    pub tol: f64,
    pub lr: f64,
    pub l2_grid: Vec<f64>,
}

/// One `_train_once`: `(W, b, converged, it, grad_norm, history)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Trained {
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub converged: bool,
    pub iterations: u32,
    pub grad_norm: f64,
    pub history: Vec<f64>,
}

/// One L2 value's fit on the training slice, scored on the validation slice.
#[derive(Clone, Debug, PartialEq)]
pub struct GridPoint {
    pub l2: f64,
    pub val_correct: u64,
    pub val_total: u64,
    pub converged: bool,
    pub iterations: u32,
    pub grad_norm: f64,
}

/// How a grid point's validation accuracy is counted, which is what selects the L2.
#[derive(Clone, Copy, Debug)]
pub enum Selection<'g> {
    /// `LinearBaseline.fit`: an example is correct when `argmax(z)` is its label.
    Top1,
    /// `OptionScorer.fit`, for a two-class scorer over each row's shown options. `groups[i]` is
    /// training example `i`'s row. The validation slice is whole rows, each contiguous in
    /// `order[..n_val]`; a row is correct when its first example of greatest log-odds
    /// `z[1] - z[0]` is its one positive, and the accuracy is over rows.
    BestOption { groups: &'g [u32] },
}

/// `LinearBaseline.fit`, plus the logits of the documents the caller will score.
#[derive(Clone, Debug, PartialEq)]
pub struct Fitted {
    pub grid: Vec<GridPoint>,
    pub selected: usize,
    pub refit: Trained,
    pub eval_logits: Vec<f64>,
}

/// Rows of a CSR in a given order: the reference's `X.select(idx)` without the copy.
#[derive(Clone, Copy)]
struct Rows<'a> {
    csr: &'a Csr,
    order: Option<&'a [usize]>,
}

impl<'a> Rows<'a> {
    fn all(csr: &'a Csr) -> Self {
        Self { csr, order: None }
    }

    fn len(&self) -> usize {
        self.order.map_or(self.csr.n_rows(), <[usize]>::len)
    }

    #[inline(always)]
    fn row(&self, i: usize) -> (&'a [u32], &'a [f64]) {
        self.csr.row(self.order.map_or(i, |o| o[i]))
    }
}

/// `X.T` with each column's entries in row order: `rmatmul`'s accumulation order.
struct Csc {
    col_ptr: Vec<usize>,
    rows: Vec<u32>,
    vals: Vec<f64>,
}

fn transpose(rows: Rows<'_>, n_cols: usize) -> Csc {
    let mut col_ptr = vec![0usize; n_cols + 1];
    for i in 0..rows.len() {
        for &c in rows.row(i).0 {
            col_ptr[c as usize + 1] += 1;
        }
    }
    for c in 0..n_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let nnz = col_ptr[n_cols];
    let mut next = col_ptr[..n_cols].to_vec();
    let mut out_rows = vec![0u32; nnz];
    let mut vals = vec![0f64; nnz];
    for i in 0..rows.len() {
        let (cols, data) = rows.row(i);
        for (&c, &v) in cols.iter().zip(data) {
            let at = next[c as usize];
            // Rows are bounded below u32::MAX by the wire's refusal, so the cast is exact.
            out_rows[at] = i as u32;
            vals[at] = v;
            next[c as usize] = at + 1;
        }
    }
    Csc {
        col_ptr,
        rows: out_rows,
        vals,
    }
}

/// Boundaries `0 = b[0] < ... < b[m] = n` splitting `n` items into about `chunks` runs of equal
/// total `weight`. Never an empty run.
fn balanced(n: usize, chunks: usize, weight: impl Fn(usize) -> usize) -> Vec<usize> {
    let total: usize = (0..n).map(&weight).sum();
    let chunks = chunks.clamp(1, n.max(1));
    let target = total.div_ceil(chunks).max(1);
    let mut bounds = vec![0usize];
    let mut acc = 0usize;
    for i in 0..n {
        acc += weight(i);
        if acc >= target && i + 1 < n {
            bounds.push(i + 1);
            acc = 0;
        }
    }
    if n > 0 {
        bounds.push(n);
    }
    bounds
}

/// `slice` cut at `bounds[i] * stride`.
fn cut<'s, T>(mut slice: &'s mut [T], bounds: &[usize], stride: usize) -> Vec<&'s mut [T]> {
    let mut out = Vec::with_capacity(bounds.len().saturating_sub(1));
    for w in bounds.windows(2) {
        let (head, tail) = slice.split_at_mut((w[1] - w[0]) * stride);
        out.push(head);
        slice = tail;
    }
    out
}

/// Run `work` over `items` on up to `threads` threads, each taking the next item in turn, while
/// the calling thread runs `meanwhile`. Each item owns disjoint output, so which thread takes
/// which item cannot change any result.
fn parallel<T: Send, R>(
    items: Vec<T>,
    threads: usize,
    work: impl Fn(T) + Sync,
    meanwhile: impl FnOnce() -> R,
) -> R {
    let workers = threads.clamp(1, items.len().max(1));
    #[cfg(test)]
    crate::team::STARTED.with(|s| s.set(s.get() + workers));
    let queue = Mutex::new(items.into_iter());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let item = queue.lock().unwrap_or_else(|p| p.into_inner()).next();
                    match item {
                        Some(t) => work(t),
                        None => break,
                    }
                }
            });
        }
        meanwhile()
    })
}

/// `X @ W + b` for one row into `z` (`k` values): the products in the row's column order,
/// summed from zero, then the bias added -- `CSR.matmul(W) + b`. `w_row(at)` yields `W`'s `k`
/// values from flat index `at` and `b(c)` the bias, so the fit's shared cells and a scoring
/// pass's plain slices take the same arithmetic.
#[inline(always)]
fn logits_row<R: IntoIterator<Item = f64>>(
    cols: &[u32],
    vals: &[f64],
    w_row: impl Fn(usize) -> R,
    b: impl Fn(usize) -> f64,
    z: &mut [f64],
) {
    let k = z.len();
    z.fill(0.0);
    for (&c, &v) in cols.iter().zip(vals) {
        for (zc, wc) in z.iter_mut().zip(w_row(c as usize * k)) {
            *zc += v * wc;
        }
    }
    for (j, zc) in z.iter_mut().enumerate() {
        *zc += b(j);
    }
}

/// The validation slice's rows as runs `[start, end)` of `order[..n_val]`, refusing a slice the
/// reference's row carve could not have drawn: a row split into two runs, a row on both sides of
/// the carve, or a row without exactly one positive.
fn validation_rows(
    groups: &[u32],
    y: &[u32],
    order: &[usize],
    n_val: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let (val, tr) = order.split_at(n_val);
    let mut runs = Vec::new();
    let mut seen = HashSet::new();
    let mut start = 0;
    while start < val.len() {
        let g = groups[val[start]];
        let mut end = start + 1;
        while end < val.len() && groups[val[end]] == g {
            end += 1;
        }
        if !seen.insert(g) {
            return Err(format!(
                "option row {g} is split into two runs of the validation slice"
            ));
        }
        runs.push((start, end));
        start = end;
    }
    if let Some(&i) = tr.iter().find(|&&i| seen.contains(&groups[i])) {
        return Err(format!(
            "option row {} has options on both sides of the validation carve",
            groups[i]
        ));
    }
    for &(start, end) in &runs {
        let positives = val[start..end].iter().filter(|&&i| y[i] == 1).count();
        if positives != 1 {
            return Err(format!(
                "validation row {} has {positives} positive option(s); a row has one gold",
                groups[val[start]]
            ));
        }
    }
    Ok(runs)
}

/// Rows of `runs` whose first option of greatest `z[1] - z[0]` is labelled 1 (numpy's `argmax`
/// tie rule over the log-odds), for two-class logits `z` laid out `[z0, z1]` per example.
fn best_option_correct(z: &[f64], y_val: &[u32], runs: &[(usize, usize)]) -> u64 {
    runs.iter()
        .filter(|&&(start, end)| {
            let score = |i: usize| z[2 * i + 1] - z[2 * i];
            let mut best = start;
            for i in start + 1..end {
                if score(i) > score(best) {
                    best = i;
                }
            }
            y_val[best] == 1
        })
        .count() as u64
}

/// `argmax` with numpy's tie rule: the first of equal maxima.
fn argmax(z: &[f64]) -> usize {
    let mut best = 0;
    for (c, v) in z.iter().enumerate().skip(1) {
        if *v > z[best] {
            best = c;
        }
    }
    best
}

/// `X @ W + b` for every row, `k` per row.
fn logits(rows: Rows<'_>, w: &[f64], b: &[f64], threads: usize) -> Vec<f64> {
    let k = b.len();
    let n = rows.len();
    let mut out = vec![0f64; n * k];
    let bounds = balanced(n, threads * CHUNKS_PER_THREAD, |i| rows.row(i).0.len() + k);
    let items: Vec<(usize, &mut [f64])> = bounds
        .iter()
        .copied()
        .zip(cut(&mut out, &bounds, k))
        .collect();
    parallel(
        items,
        threads,
        |(start, chunk): (usize, &mut [f64])| {
            for (j, z) in chunk.chunks_exact_mut(k).enumerate() {
                let (cols, vals) = rows.row(start + j);
                logits_row(cols, vals, |at| w[at..at + k].iter().copied(), |c| b[c], z);
            }
        },
        || (),
    );
    out
}

/// `_train_once(X, y, n_classes, l2)` with `W` initialised to `w0`.
///
/// Each iteration is two phases on one [`crate::team`] team, started once per call: phase A
/// over runs of rows, phase B over runs of columns, with the leader's reductions between and
/// beside them. Every element is still written by exactly one item in a fixed order, so the
/// result does not depend on the thread count or on which thread runs which item.
///
/// The two loops over the nonzeros read plain slices, so the compiler can vectorize their
/// `k`-wide multiply-add as the threads-per-phase loop did: phase A reads `W` from `w_now`,
/// phase B reads `diff` from `diff_now`, each a snapshot the leader copies between phases
/// (`d * k` and `n * k` values an iteration). Everything else the phases share is
/// [`SharedF64s`] cells. Phase B writes the Adam step's `W` into `w_next`, and the leader copies
/// it into `w_now` only when the gradient missed the tolerance, so a converged fit returns the
/// `W` whose gradient was measured, as the reference's `break` does; `m` and `v` are updated in
/// place, since a converged fit discards them.
fn train_once(
    rows: Rows<'_>,
    n_cols: usize,
    y: &[u32],
    w0: &[f64],
    hyper: &Hyper,
    l2: f64,
    threads: usize,
) -> Trained {
    let k = hyper.n_classes;
    let n = rows.len();
    let n_f = n as f64;
    let csc = transpose(rows, n_cols);
    let dk = n_cols * k;

    let w_now = RwLock::new(w0.to_vec());
    let w_next = SharedF64s::zeros(dk);
    let mw = SharedF64s::zeros(dk);
    let vw = SharedF64s::zeros(dk);
    let b = SharedF64s::zeros(k);
    let gw = SharedF64s::zeros(dk);
    let diff = SharedF64s::zeros(n * k);
    let diff_now = RwLock::new(vec![0f64; n * k]);
    let log_terms = SharedF64s::zeros(n * k);
    // This iteration's step and bias corrections, written by the leader before phase B.
    let step = SharedF64s::zeros(3);
    let (mut mb, mut vb) = (vec![0f64; k], vec![0f64; k]);
    let mut history = Vec::new();
    let (mut converged, mut grad_norm, mut it) = (false, f64::INFINITY, 0u32);

    let row_bounds = balanced(n, threads * CHUNKS_PER_THREAD, |i| rows.row(i).0.len() + k);
    let col_bounds = balanced(n_cols, threads * CHUNKS_PER_THREAD, |c| {
        csc.col_ptr[c + 1] - csc.col_ptr[c] + k
    });
    let two_l2 = 2.0 * l2;
    let (one_minus_b1, one_minus_b2) = (1.0 - BETA1, 1.0 - BETA2);

    // Phase A, one item per run of rows: P = softmax(X @ W + b), the loss's
    // Y * log(clip(P)), and diff = (P - Y) / n.
    let phase_a = |item: usize| {
        let w_guard = read(&w_now);
        // A plain slice, bound once per item: indexed through the guard, the hot loop reloads
        // the Vec's pointer and length for every nonzero (its stores to `z` may alias them, as
        // far as the compiler can tell), ~10% of phase A on the H100 box (box_ab8, box_ab9).
        let w: &[f64] = &w_guard;
        let mut z = vec![0f64; k];
        let mut e = vec![0f64; k];
        // `r` also indexes `rows`, `y`, `log_terms` and `diff` (as `r * k + c`): an iterator over
        // one of them, as the lint suggests, would hide the shared index.
        #[allow(clippy::needless_range_loop)]
        for r in row_bounds[item]..row_bounds[item + 1] {
            let (cols, vals) = rows.row(r);
            logits_row(
                cols,
                vals,
                |at| w[at..at + k].iter().copied(),
                |c| b.get(c),
                &mut z,
            );
            // _softmax: z - z.max(axis=1), exp, / its pairwise row sum.
            let mut m = z[0];
            for v in &z[1..] {
                if *v > m {
                    m = *v;
                }
            }
            for (ec, zc) in e.iter_mut().zip(&z) {
                *ec = (zc - m).exp();
            }
            let s = pairwise_sum(k, &|c| e[c]);
            let label = y[r] as usize;
            // `c` is compared with `label` and indexes `log_terms` and `diff` as `r * k + c`.
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                let p = e[c] / s;
                let yc = if c == label { 1.0 } else { 0.0 };
                let clipped = if p < P_FLOOR { P_FLOOR } else { p };
                log_terms.set(r * k + c, yc * clipped.ln());
                diff.set(r * k + c, (p - yc) / n_f);
            }
        }
    };
    // Phase B, one item per run of columns: gW = X.T @ diff + (2*l2)*W, and the Adam step.
    let phase_b = |item: usize| {
        let (w_guard, d_guard) = (read(&w_now), read(&diff_now));
        // Plain slices, bound once per item, as in phase A.
        let (w, d): (&[f64], &[f64]) = (&w_guard, &d_guard);
        let (lr, bias_correction1, bias_correction2) = (step.get(0), step.get(1), step.get(2));
        let mut g = vec![0f64; k];
        for col in col_bounds[item]..col_bounds[item + 1] {
            g.fill(0.0);
            for e in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                let r = csc.rows[e] as usize;
                let v = csc.vals[e];
                for (gc, dc) in g.iter_mut().zip(&d[r * k..r * k + k]) {
                    *gc += v * dc;
                }
            }
            for (c, &gc) in g.iter().enumerate() {
                let i = col * k + c;
                let grad = gc + two_l2 * w[i];
                gw.set(i, grad);
                let m1 = mw.get(i) * BETA1 + one_minus_b1 * grad;
                let v1 = vw.get(i) * BETA2 + one_minus_b2 * (grad * grad);
                let mhat = m1 / bias_correction1;
                let vhat = v1 / bias_correction2;
                mw.set(i, m1);
                vw.set(i, v1);
                w_next.set(i, w[i] - (lr * mhat) / (vhat.sqrt() + EPS));
            }
        }
    };
    let jobs: [Job<'_>; 2] = [&phase_a, &phase_b];
    let (row_items, col_items) = (row_bounds.len() - 1, col_bounds.len() - 1);

    with_team(threads, &jobs, |team| {
        for t in 1..=hyper.max_iter {
            it = t;
            // Meanwhile: the loss's sum(W * W).
            let sum_ww = team.run(0, row_items, || {
                let w = read(&w_now);
                pairwise_sum(dk, &|i| w[i] * w[i])
            });

            diff.copy_to(&mut write(&diff_now));
            let bias_correction1 = 1.0 - BETA1.powf(f64::from(t));
            let bias_correction2 = 1.0 - BETA2.powf(f64::from(t));
            let lr = step_size(hyper.lr, t);
            step.set(0, lr);
            step.set(1, bias_correction1);
            step.set(2, bias_correction2);
            // Meanwhile: the loss, and gb = diff.sum(axis=0) down each column.
            let (loss, gb) = team.run(1, col_items, || {
                let sum_log = pairwise_sum(n * k, &|i| log_terms.get(i));
                let loss = (-sum_log) / n_f + l2 * sum_ww;
                let d = read(&diff_now);
                let mut gb = vec![0f64; k];
                for row in d.chunks_exact(k) {
                    for (s, dc) in gb.iter_mut().zip(row) {
                        *s += dc;
                    }
                }
                (loss, gb)
            });
            history.push(loss);

            grad_norm = (pairwise_sum(dk, &|i| {
                let g = gw.get(i);
                g * g
            }) + pairwise_sum(k, &|c| gb[c] * gb[c]))
            .sqrt();
            if grad_norm < hyper.tol {
                converged = true;
                break;
            }
            w_next.copy_to(&mut write(&w_now));
            for c in 0..k {
                let grad = gb[c];
                mb[c] = mb[c] * BETA1 + one_minus_b1 * grad;
                vb[c] = vb[c] * BETA2 + one_minus_b2 * (grad * grad);
                let mhat = mb[c] / bias_correction1;
                let vhat = vb[c] / bias_correction2;
                b.set(c, b.get(c) - (lr * mhat) / (vhat.sqrt() + EPS));
            }
        }
    });
    Trained {
        w: w_now.into_inner().unwrap_or_else(PoisonError::into_inner),
        b: b.to_vec(),
        converged,
        iterations: it,
        grad_norm,
        history,
    }
}

/// A read guard on a snapshot no writer holds during a phase (poisoning is not a state here:
/// a panic anywhere in the fit panics the fit).
fn read(lock: &RwLock<Vec<f64>>) -> RwLockReadGuard<'_, Vec<f64>> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// The leader's write guard on a snapshot, taken only between phases.
fn write(lock: &RwLock<Vec<f64>>) -> RwLockWriteGuard<'_, Vec<f64>> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// `LinearBaseline.fit(docs, labels)` on `x` (every training row, in input order) and the
/// logits of `eval_x` under the refitted weights.
///
/// `order` is the reference's `default_rng(seed).permutation(n)` and `n_val` its
/// `max(1, int(n * val_frac))`: validation is `order[:n_val]`, training `order[n_val:]`, and
/// both keep that order, as `CSR.select` does. `w0` is `_train_once`'s initial `W`, the same for
/// every fit because the reference reseeds per call. `selection` says how each grid point's
/// validation accuracy is counted; [`Selection::BestOption`] counts rows, so its grid points'
/// `val_total` is the number of validation rows.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    x: &Csr,
    y: &[u32],
    order: &[usize],
    n_val: usize,
    w0: &[f64],
    eval_x: &Csr,
    hyper: &Hyper,
    selection: Selection<'_>,
    threads: usize,
) -> Result<Fitted, String> {
    let n = x.n_rows();
    let k = hyper.n_classes;
    let d = x.n_cols;
    if y.len() != n || order.len() != n {
        return Err(format!(
            "{n} rows, {} labels, {} permutation entries",
            y.len(),
            order.len()
        ));
    }
    if n < 4 {
        return Err(format!(
            "need at least 4 examples to fit and validate, got {n}"
        ));
    }
    if n_val == 0 || n_val >= n {
        return Err(format!(
            "validation slice of {n_val} rows out of {n}: the reference requires 1..{n}"
        ));
    }
    if w0.len() != d * k {
        return Err(format!(
            "initial weights hold {} values, not {d} x {k}",
            w0.len()
        ));
    }
    if eval_x.n_cols != d {
        return Err(format!(
            "evaluation rows have {} columns, training rows {d}",
            eval_x.n_cols
        ));
    }
    if hyper.l2_grid.is_empty() {
        return Err("an empty L2 grid selects nothing".to_string());
    }
    // Checked before anything is trained: a malformed carve is refused, not discovered late.
    let rows = match selection {
        Selection::Top1 => None,
        Selection::BestOption { groups } => {
            if k != 2 {
                return Err(format!(
                    "the best-option rule scores a two-class scorer's log-odds; {k} classes"
                ));
            }
            if groups.len() != n {
                return Err(format!("{n} rows, {} option-row ids", groups.len()));
            }
            Some(validation_rows(groups, y, order, n_val)?)
        }
    };
    let (val_idx, tr_idx) = order.split_at(n_val);
    let (y_val, y_tr): (Vec<u32>, Vec<u32>) = (
        val_idx.iter().map(|&i| y[i]).collect(),
        tr_idx.iter().map(|&i| y[i]).collect(),
    );
    let tr = Rows {
        csr: x,
        order: Some(tr_idx),
    };
    let val = Rows {
        csr: x,
        order: Some(val_idx),
    };

    let mut grid = Vec::with_capacity(hyper.l2_grid.len());
    // (accuracy, index): the reference keeps the first L2 whose validation accuracy is
    // strictly greater than every earlier one.
    let mut best: Option<(f64, usize)> = None;
    for (g, &l2) in hyper.l2_grid.iter().enumerate() {
        let trained = train_once(tr, d, &y_tr, w0, hyper, l2, threads);
        let z = logits(val, &trained.w, &trained.b, threads);
        let (correct, total) = match &rows {
            None => (
                z.chunks_exact(k)
                    .zip(&y_val)
                    .filter(|(row, label)| argmax(row) == **label as usize)
                    .count() as u64,
                n_val as u64,
            ),
            Some(runs) => (best_option_correct(&z, &y_val, runs), runs.len() as u64),
        };
        let acc = correct as f64 / total as f64;
        if best.is_none_or(|(b, _)| acc > b) {
            best = Some((acc, g));
        }
        grid.push(GridPoint {
            l2,
            val_correct: correct,
            val_total: total,
            converged: trained.converged,
            iterations: trained.iterations,
            grad_norm: trained.grad_norm,
        });
    }
    let selected = best.map_or(0, |(_, g)| g);
    let refit = train_once(
        Rows::all(x),
        d,
        y,
        w0,
        hyper,
        hyper.l2_grid[selected],
        threads,
    );
    if let Some(i) = refit.w.iter().chain(&refit.b).position(|v| !v.is_finite()) {
        return Err(format!(
            "the refitted weights hold a non-finite value at {i}; refusing to score with them"
        ));
    }
    let eval_logits = logits(Rows::all(eval_x), &refit.w, &refit.b, threads);
    Ok(Fitted {
        grid,
        selected,
        refit,
        eval_logits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOP1: Selection<'static> = Selection::Top1;

    #[test]
    fn the_step_is_constant_through_the_old_budget_then_halves_every_period() {
        let lr = 0.05;
        for step in [1, 2, 5_999, 6_000] {
            assert_eq!(step_size(lr, step).to_bits(), lr.to_bits(), "step {step}");
        }
        assert_eq!(step_size(lr, 6_001), lr / 2.0);
        assert_eq!(step_size(lr, 6_500), lr / 2.0);
        assert_eq!(step_size(lr, 6_501), lr / 4.0);
        assert_eq!(step_size(lr, 7_000), lr / 4.0);
        assert_eq!(step_size(lr, 12_000), lr / 4096.0);
        // Powers of two: exact, and zero once 0.5^n underflows, as Python's `0.5 ** n` is.
        assert_eq!(step_size(1.0, 6_000 + 500 * 1_074), f64::from_bits(1));
        assert_eq!(step_size(1.0, 6_000 + 500 * 1_075), 0.0);
    }

    /// A small problem with structure: class = (row % k), with features that mostly say so.
    fn problem(n: usize, d: usize, k: usize) -> (Csr, Vec<u32>, Vec<f64>) {
        let mut indptr = vec![0usize];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        let mut y = Vec::new();
        for r in 0..n {
            let label = r % k;
            y.push(label as u32);
            let mut cols: Vec<u32> = (0..5)
                .map(|j| ((r * 7 + j * 13 + label * 31) % d) as u32)
                .collect();
            cols.push((label % d) as u32);
            cols.sort_unstable();
            cols.dedup();
            let norm = (cols.len() as f64).sqrt();
            for c in cols {
                indices.push(c);
                data.push(1.0 / norm);
            }
            indptr.push(indices.len());
        }
        let w0 = (0..d * k)
            .map(|i| ((i as f64) * 0.618).sin() * 0.01)
            .collect();
        (
            Csr {
                n_cols: d,
                indptr,
                indices,
                data,
            },
            y,
            w0,
        )
    }

    fn hyper(k: usize, max_iter: u32) -> Hyper {
        Hyper {
            n_classes: k,
            max_iter,
            tol: 1e-4,
            lr: 0.05,
            l2_grid: vec![1e-4, 1e-3, 1e-2, 1e-1],
        }
    }

    fn order(n: usize) -> Vec<usize> {
        (0..n).map(|i| (i * 37 + 11) % n).collect()
    }

    #[test]
    fn the_fit_does_not_depend_on_the_thread_count() {
        for k in [2usize, 3, 9] {
            let (x, y, w0) = problem(97, 64, k);
            let ord = order(97);
            let one = fit(&x, &y, &ord, 19, &w0, &x, &hyper(k, 150), TOP1, 1).expect("fits");
            for threads in [2, 3, 5, 18, 64] {
                let many =
                    fit(&x, &y, &ord, 19, &w0, &x, &hyper(k, 150), TOP1, threads).expect("fits");
                assert_eq!(many, one, "k={k}, {threads} threads");
            }
        }
    }

    /// The threads a fit starts must not grow with its iterations: on the H100 box each thread
    /// start cost ~32 us, and starting them for each of an iteration's two phases was ~3.4 ms an
    /// iteration at 52 threads (2026-10-04, `examples/linfit_bench.rs`).
    #[test]
    fn a_fit_starts_its_threads_per_training_run_not_per_iteration() {
        let (x, y, w0) = problem(97, 64, 3);
        let ord = order(97);
        let started = |max_iter: u32| {
            let mut h = hyper(3, max_iter);
            // Never met, so every training run takes exactly max_iter iterations.
            h.tol = 0.0;
            let before = crate::team::STARTED.with(std::cell::Cell::get);
            let got = fit(&x, &y, &ord, 19, &w0, &x, &h, TOP1, 4).expect("fits");
            assert_eq!(got.refit.iterations, max_iter);
            crate::team::STARTED.with(std::cell::Cell::get) - before
        };
        let (short, long) = (started(10), started(40));
        assert_eq!(
            long, short,
            "40 iterations started {long} threads, 10 started {short}"
        );
    }

    #[test]
    fn no_iterations_is_unconverged_at_infinity_like_the_reference() {
        let (x, y, w0) = problem(20, 32, 2);
        let got = fit(&x, &y, &order(20), 4, &w0, &x, &hyper(2, 0), TOP1, 4).expect("fits");
        assert!(!got.refit.converged);
        assert_eq!(got.refit.iterations, 0);
        assert!(got.refit.grad_norm.is_infinite());
        assert!(got.refit.history.is_empty());
        assert_eq!(got.refit.w, w0, "no step was taken");
    }

    #[test]
    fn a_converged_fit_returns_the_weights_whose_gradient_met_the_tolerance() {
        let (x, y, w0) = problem(60, 32, 3);
        let mut h = hyper(3, 5000);
        h.tol = 5e-2;
        let got = fit(&x, &y, &order(60), 12, &w0, &x, &h, TOP1, 3).expect("fits");
        assert!(got.refit.converged);
        assert!(got.refit.grad_norm < h.tol);
        assert_eq!(got.refit.history.len(), got.refit.iterations as usize);
        // One more iteration's budget changes nothing about a fit that stopped early.
        h.max_iter = got.refit.iterations;
        let again = fit(&x, &y, &order(60), 12, &w0, &x, &h, TOP1, 3).expect("fits");
        assert_eq!(again.refit, got.refit);
    }

    #[test]
    fn selection_keeps_the_first_strictly_best_l2() {
        let (x, y, w0) = problem(80, 48, 2);
        let got = fit(&x, &y, &order(80), 16, &w0, &x, &hyper(2, 200), TOP1, 2).expect("fits");
        let best = got
            .grid
            .iter()
            .map(|g| g.val_correct)
            .max()
            .expect("non-empty");
        let first = got
            .grid
            .iter()
            .position(|g| g.val_correct == best)
            .expect("present");
        assert_eq!(got.selected, first);
    }

    #[test]
    fn malformed_requests_are_refused() {
        let (x, y, w0) = problem(20, 32, 2);
        let h = hyper(2, 10);
        assert!(
            fit(&x, &y, &order(20), 0, &w0, &x, &h, TOP1, 1).is_err(),
            "no validation rows"
        );
        assert!(
            fit(&x, &y, &order(20), 20, &w0, &x, &h, TOP1, 1).is_err(),
            "no training rows"
        );
        assert!(
            fit(&x, &y[..19], &order(20), 4, &w0, &x, &h, TOP1, 1).is_err(),
            "labels short"
        );
        assert!(
            fit(&x, &y, &order(20), 4, &w0[1..], &x, &h, TOP1, 1).is_err(),
            "w0 short"
        );
        let mut empty_grid = h.clone();
        empty_grid.l2_grid.clear();
        assert!(fit(&x, &y, &order(20), 4, &w0, &x, &empty_grid, TOP1, 1).is_err());
    }

    #[test]
    fn transposition_keeps_each_columns_entries_in_row_order() {
        let (x, _, _) = problem(30, 16, 3);
        let ord: Vec<usize> = (0..30).rev().collect();
        let rows = Rows {
            csr: &x,
            order: Some(&ord),
        };
        let csc = transpose(rows, 16);
        for c in 0..16 {
            let span = &csc.rows[csc.col_ptr[c]..csc.col_ptr[c + 1]];
            assert!(span.windows(2).all(|p| p[0] < p[1]), "column {c}");
            for (&r, &v) in span
                .iter()
                .zip(&csc.vals[csc.col_ptr[c]..csc.col_ptr[c + 1]])
            {
                let (cols, vals) = rows.row(r as usize);
                let at = cols
                    .iter()
                    .position(|&cc| cc as usize == c)
                    .expect("present");
                assert_eq!(vals[at].to_bits(), v.to_bits());
            }
        }
    }

    /// `n_rows` rows of `per_row` options, row-major: option `j` of row `r` is the positive when
    /// `j == r % per_row`, and its features mostly say whether it is.
    fn option_problem(n_rows: usize, per_row: usize, d: usize) -> (Csr, Vec<u32>, Vec<u32>) {
        let mut indptr = vec![0usize];
        let (mut indices, mut data, mut y, mut groups) = (vec![], vec![], vec![], vec![]);
        for r in 0..n_rows {
            for j in 0..per_row {
                let positive = j == r % per_row;
                y.push(u32::from(positive));
                groups.push(r as u32);
                let mut cols: Vec<u32> = (0..4)
                    .map(|t| ((r * 5 + j * 11 + t * 3) % d) as u32)
                    .collect();
                // A noisy marker of the positive: present on most positives, few negatives.
                if positive != (r % 7 == 0) {
                    cols.push(0);
                }
                cols.sort_unstable();
                cols.dedup();
                let norm = (cols.len() as f64).sqrt();
                for c in cols {
                    indices.push(c);
                    data.push(1.0 / norm);
                }
                indptr.push(indices.len());
            }
        }
        let csr = Csr {
            n_cols: d,
            indptr,
            indices,
            data,
        };
        (csr, y, groups)
    }

    /// `OptionScorer.carve`'s shape: whole rows, validation rows first, each row's options in
    /// order. Returns the example order and `n_val` in examples.
    fn row_carve(n_rows: usize, per_row: usize, n_val_rows: usize) -> (Vec<usize>, usize) {
        let rows: Vec<usize> = (0..n_rows).map(|i| (i * 13 + 5) % n_rows).collect();
        let order = rows
            .iter()
            .flat_map(|&r| (r * per_row)..(r * per_row + per_row))
            .collect();
        (order, n_val_rows * per_row)
    }

    fn w0(d: usize, k: usize) -> Vec<f64> {
        (0..d * k)
            .map(|i| ((i as f64) * 0.618).sin() * 0.01)
            .collect()
    }

    #[test]
    fn best_option_counts_validation_rows_by_their_best_option() {
        let (x, y, groups) = option_problem(40, 3, 32);
        let (ord, n_val) = row_carve(40, 3, 8);
        let w = w0(32, 2);
        let h = hyper(2, 300);
        let sel = Selection::BestOption { groups: &groups };
        let got = fit(&x, &y, &ord, n_val, &w, &x, &h, sel, 2).expect("fits");
        let (val, tr) = ord.split_at(n_val);
        let y_tr: Vec<u32> = tr.iter().map(|&i| y[i]).collect();
        let mut best: Option<(f64, usize)> = None;
        for (g, point) in got.grid.iter().enumerate() {
            assert_eq!(point.val_total, 8, "rows, not the {n_val} option examples");
            // Recounted independently: retrain, then per row the option of greatest log-odds.
            let trained = train_once(
                Rows {
                    csr: &x,
                    order: Some(tr),
                },
                32,
                &y_tr,
                &w,
                &h,
                point.l2,
                1,
            );
            let z = logits(
                Rows {
                    csr: &x,
                    order: Some(val),
                },
                &trained.w,
                &trained.b,
                1,
            );
            let correct = val
                .chunks(3)
                .enumerate()
                .filter(|(r, opts)| {
                    let odds: Vec<f64> = (0..3)
                        .map(|j| z[2 * (r * 3 + j) + 1] - z[2 * (r * 3 + j)])
                        .collect();
                    let top = odds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let first = odds.iter().position(|&o| o == top).expect("a max");
                    y[opts[first]] == 1
                })
                .count() as u64;
            assert_eq!(point.val_correct, correct, "grid point {g}");
            let acc = correct as f64 / 8.0;
            if best.is_none_or(|(b, _)| acc > b) {
                best = Some((acc, g));
            }
        }
        assert_eq!(got.selected, best.expect("a grid").1);
        assert!(
            got.grid.iter().any(|p| p.val_correct > 8 / 3),
            "{:?}",
            got.grid
        );
    }

    #[test]
    fn best_option_fit_does_not_depend_on_the_thread_count() {
        let (x, y, groups) = option_problem(31, 4, 48);
        let (ord, n_val) = row_carve(31, 4, 6);
        let w = w0(48, 2);
        let sel = Selection::BestOption { groups: &groups };
        let one = fit(&x, &y, &ord, n_val, &w, &x, &hyper(2, 120), sel, 1).expect("fits");
        for threads in [2, 3, 7, 64] {
            let many =
                fit(&x, &y, &ord, n_val, &w, &x, &hyper(2, 120), sel, threads).expect("fits");
            assert_eq!(many, one, "{threads} threads");
        }
    }

    #[test]
    fn a_validation_slice_the_row_carve_could_not_draw_is_refused() {
        let (x, y, groups) = option_problem(10, 3, 16);
        let (ord, n_val) = row_carve(10, 3, 2);
        let w = w0(16, 2);
        let h = hyper(2, 5);
        let sel = Selection::BestOption { groups: &groups };
        assert!(fit(&x, &y, &ord, n_val, &w, &x, &h, sel, 1).is_ok());

        let err = |ord: &[usize], n_val: usize, y: &[u32], groups: &[u32], k: usize| {
            let sel = Selection::BestOption { groups };
            let w = w0(16, k);
            fit(&x, y, ord, n_val, &w, &x, &hyper(k, 5), sel, 1).expect_err("refused")
        };
        // A row whose options straddle the carve.
        assert!(err(&ord, n_val - 1, &y, &groups, 2).contains("both sides"));
        // A row split into two runs of the validation slice.
        let mut split = ord.clone();
        split.swap(1, 4);
        assert!(err(&split, n_val, &y, &groups, 2).contains("two runs"));
        // A validation row with two positives, and one with none.
        let first = ord[0] / 3;
        let mut two = y.clone();
        for j in 0..3 {
            two[first * 3 + j] = u32::from(j < 2);
        }
        assert!(err(&ord, n_val, &two, &groups, 2).contains("2 positive"));
        let mut none = y.clone();
        for j in 0..3 {
            none[first * 3 + j] = 0;
        }
        assert!(err(&ord, n_val, &none, &groups, 2).contains("0 positive"));
        // Not a binary scorer; ids for a different number of rows.
        assert!(err(&ord, n_val, &y, &groups, 3).contains("two-class"));
        assert!(err(&ord, n_val, &y, &groups[1..], 2).contains("option-row ids"));
    }
}
