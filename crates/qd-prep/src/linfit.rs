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

use std::sync::Mutex;

use crate::ngram::Csr;
use crate::pairwise::pairwise_sum;

/// `_train_once`'s Adam constants, fixed there and here.
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
/// `np.clip(P, 1e-12, None)` inside the loss.
const P_FLOOR: f64 = 1e-12;
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
/// summed from zero, then the bias added -- `CSR.matmul(W) + b`.
#[inline(always)]
fn logits_row(cols: &[u32], vals: &[f64], w: &[f64], b: &[f64], z: &mut [f64]) {
    let k = z.len();
    z.fill(0.0);
    for (&c, &v) in cols.iter().zip(vals) {
        let wr = &w[c as usize * k..c as usize * k + k];
        for (zc, wc) in z.iter_mut().zip(wr) {
            *zc += v * wc;
        }
    }
    for (zc, bc) in z.iter_mut().zip(b) {
        *zc += bc;
    }
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
                logits_row(cols, vals, w, b, z);
            }
        },
        || (),
    );
    out
}

/// Phase A's unit of work: rows `start..`, and their slices of `diff` and the loss terms.
struct RowRun<'s> {
    start: usize,
    diff: &'s mut [f64],
    log_terms: &'s mut [f64],
}

/// Phase B's unit of work: columns `start..`, their gradient rows and their next `W`, `m`, `v`.
struct ColumnRun<'s> {
    start: usize,
    grad: &'s mut [f64],
    w: &'s mut [f64],
    m: &'s mut [f64],
    v: &'s mut [f64],
}

/// `_train_once(X, y, n_classes, l2)` with `W` initialised to `w0`.
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

    let mut w = w0.to_vec();
    let mut b = vec![0f64; k];
    let (mut mw, mut vw) = (vec![0f64; dk], vec![0f64; dk]);
    let (mut mb, mut vb) = (vec![0f64; k], vec![0f64; k]);
    // Phase B writes the gradient and, speculatively, the Adam step's next W, m and v; they
    // are swapped in only when the gradient did not meet the tolerance, so a converged fit
    // returns the W whose gradient was measured, as the reference's `break` does.
    let mut gw = vec![0f64; dk];
    let (mut w_next, mut mw_next, mut vw_next) = (vec![0f64; dk], vec![0f64; dk], vec![0f64; dk]);
    let mut diff = vec![0f64; n * k];
    let mut log_terms = vec![0f64; n * k];
    let mut history = Vec::new();
    let (mut converged, mut grad_norm, mut it) = (false, f64::INFINITY, 0u32);

    let row_bounds = balanced(n, threads * CHUNKS_PER_THREAD, |i| rows.row(i).0.len() + k);
    let col_bounds = balanced(n_cols, threads * CHUNKS_PER_THREAD, |c| {
        csc.col_ptr[c + 1] - csc.col_ptr[c] + k
    });
    let two_l2 = 2.0 * l2;
    let (one_minus_b1, one_minus_b2) = (1.0 - BETA1, 1.0 - BETA2);

    for step in 1..=hyper.max_iter {
        it = step;
        // Phase A, one thread per row: P = softmax(X @ W + b), the loss's Y * log(clip(P)),
        // and diff = (P - Y) / n. Meanwhile: the loss's sum(W * W).
        let sum_ww = {
            let (w, b) = (&w, &b);
            let items: Vec<RowRun<'_>> = row_bounds
                .iter()
                .copied()
                .zip(cut(&mut diff, &row_bounds, k))
                .zip(cut(&mut log_terms, &row_bounds, k))
                .map(|((start, diff), log_terms)| RowRun {
                    start,
                    diff,
                    log_terms,
                })
                .collect();
            parallel(
                items,
                threads,
                |run: RowRun<'_>| {
                    let mut z = vec![0f64; k];
                    let mut e = vec![0f64; k];
                    for (j, (d_row, l_row)) in run
                        .diff
                        .chunks_exact_mut(k)
                        .zip(run.log_terms.chunks_exact_mut(k))
                        .enumerate()
                    {
                        let r = run.start + j;
                        let (cols, vals) = rows.row(r);
                        logits_row(cols, vals, w, b, &mut z);
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
                        for c in 0..k {
                            let p = e[c] / s;
                            let yc = if c == label { 1.0 } else { 0.0 };
                            let clipped = if p < P_FLOOR { P_FLOOR } else { p };
                            l_row[c] = yc * clipped.ln();
                            d_row[c] = (p - yc) / n_f;
                        }
                    }
                },
                || pairwise_sum(dk, &|i| w[i] * w[i]),
            )
        };

        // Phase B, one thread per column: gW = X.T @ diff + (2*l2)*W, and the Adam step
        // computed ahead. Meanwhile: the loss, and gb = diff.sum(axis=0) down each column.
        let bias_correction1 = 1.0 - BETA1.powf(f64::from(step));
        let bias_correction2 = 1.0 - BETA2.powf(f64::from(step));
        let (loss, gb) = {
            let (w_ref, mw_ref, vw_ref, diff_ref) = (&w, &mw, &vw, &diff);
            let csc = &csc;
            let items: Vec<ColumnRun<'_>> = col_bounds
                .iter()
                .copied()
                .zip(cut(&mut gw, &col_bounds, k))
                .zip(cut(&mut w_next, &col_bounds, k))
                .zip(cut(&mut mw_next, &col_bounds, k))
                .zip(cut(&mut vw_next, &col_bounds, k))
                .map(|((((start, grad), w), m), v)| ColumnRun {
                    start,
                    grad,
                    w,
                    m,
                    v,
                })
                .collect();
            parallel(
                items,
                threads,
                |run: ColumnRun<'_>| {
                    let mut g = vec![0f64; k];
                    for (j, (((g_row, w_row), m_row), v_row)) in run
                        .grad
                        .chunks_exact_mut(k)
                        .zip(run.w.chunks_exact_mut(k))
                        .zip(run.m.chunks_exact_mut(k))
                        .zip(run.v.chunks_exact_mut(k))
                        .enumerate()
                    {
                        let col = run.start + j;
                        g.fill(0.0);
                        for e in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                            let r = csc.rows[e] as usize;
                            let v = csc.vals[e];
                            for (gc, dc) in g.iter_mut().zip(&diff_ref[r * k..r * k + k]) {
                                *gc += v * dc;
                            }
                        }
                        for c in 0..k {
                            let i = col * k + c;
                            let grad = g[c] + two_l2 * w_ref[i];
                            g_row[c] = grad;
                            let m1 = mw_ref[i] * BETA1 + one_minus_b1 * grad;
                            let v1 = vw_ref[i] * BETA2 + one_minus_b2 * (grad * grad);
                            let mhat = m1 / bias_correction1;
                            let vhat = v1 / bias_correction2;
                            m_row[c] = m1;
                            v_row[c] = v1;
                            w_row[c] = w_ref[i] - (hyper.lr * mhat) / (vhat.sqrt() + EPS);
                        }
                    }
                },
                || {
                    let sum_log = pairwise_sum(n * k, &|i| log_terms[i]);
                    let loss = (-sum_log) / n_f + l2 * sum_ww;
                    let mut gb = vec![0f64; k];
                    for row in diff.chunks_exact(k) {
                        for (s, d) in gb.iter_mut().zip(row) {
                            *s += d;
                        }
                    }
                    (loss, gb)
                },
            )
        };
        history.push(loss);

        grad_norm =
            (pairwise_sum(dk, &|i| gw[i] * gw[i]) + pairwise_sum(k, &|c| gb[c] * gb[c])).sqrt();
        if grad_norm < hyper.tol {
            converged = true;
            break;
        }
        std::mem::swap(&mut w, &mut w_next);
        std::mem::swap(&mut mw, &mut mw_next);
        std::mem::swap(&mut vw, &mut vw_next);
        for c in 0..k {
            let grad = gb[c];
            mb[c] = mb[c] * BETA1 + one_minus_b1 * grad;
            vb[c] = vb[c] * BETA2 + one_minus_b2 * (grad * grad);
            let mhat = mb[c] / bias_correction1;
            let vhat = vb[c] / bias_correction2;
            b[c] -= (hyper.lr * mhat) / (vhat.sqrt() + EPS);
        }
    }
    Trained {
        w,
        b,
        converged,
        iterations: it,
        grad_norm,
        history,
    }
}

/// `LinearBaseline.fit(docs, labels)` on `x` (every training row, in input order) and the
/// logits of `eval_x` under the refitted weights.
///
/// `order` is the reference's `default_rng(seed).permutation(n)` and `n_val` its
/// `max(1, int(n * val_frac))`: validation is `order[:n_val]`, training `order[n_val:]`, and
/// both keep that order, as `CSR.select` does. `w0` is `_train_once`'s initial `W`, the same for
/// every fit because the reference reseeds per call.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    x: &Csr,
    y: &[u32],
    order: &[usize],
    n_val: usize,
    w0: &[f64],
    eval_x: &Csr,
    hyper: &Hyper,
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
        let correct = z
            .chunks_exact(k)
            .zip(&y_val)
            .filter(|(row, label)| argmax(row) == **label as usize)
            .count() as u64;
        let acc = correct as f64 / n_val as f64;
        if best.is_none_or(|(b, _)| acc > b) {
            best = Some((acc, g));
        }
        grid.push(GridPoint {
            l2,
            val_correct: correct,
            val_total: n_val as u64,
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
            let one = fit(&x, &y, &ord, 19, &w0, &x, &hyper(k, 150), 1).expect("fits");
            for threads in [2, 3, 5, 18, 64] {
                let many = fit(&x, &y, &ord, 19, &w0, &x, &hyper(k, 150), threads).expect("fits");
                assert_eq!(many, one, "k={k}, {threads} threads");
            }
        }
    }

    #[test]
    fn no_iterations_is_unconverged_at_infinity_like_the_reference() {
        let (x, y, w0) = problem(20, 32, 2);
        let got = fit(&x, &y, &order(20), 4, &w0, &x, &hyper(2, 0), 4).expect("fits");
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
        let got = fit(&x, &y, &order(60), 12, &w0, &x, &h, 3).expect("fits");
        assert!(got.refit.converged);
        assert!(got.refit.grad_norm < h.tol);
        assert_eq!(got.refit.history.len(), got.refit.iterations as usize);
        // One more iteration's budget changes nothing about a fit that stopped early.
        h.max_iter = got.refit.iterations;
        let again = fit(&x, &y, &order(60), 12, &w0, &x, &h, 3).expect("fits");
        assert_eq!(again.refit, got.refit);
    }

    #[test]
    fn selection_keeps_the_first_strictly_best_l2() {
        let (x, y, w0) = problem(80, 48, 2);
        let got = fit(&x, &y, &order(80), 16, &w0, &x, &hyper(2, 200), 2).expect("fits");
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
            fit(&x, &y, &order(20), 0, &w0, &x, &h, 1).is_err(),
            "no validation rows"
        );
        assert!(
            fit(&x, &y, &order(20), 20, &w0, &x, &h, 1).is_err(),
            "no training rows"
        );
        assert!(
            fit(&x, &y[..19], &order(20), 4, &w0, &x, &h, 1).is_err(),
            "labels short"
        );
        assert!(
            fit(&x, &y, &order(20), 4, &w0[1..], &x, &h, 1).is_err(),
            "w0 short"
        );
        let mut empty_grid = h.clone();
        empty_grid.l2_grid.clear();
        assert!(fit(&x, &y, &order(20), 4, &w0, &x, &empty_grid, 1).is_err());
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
}
