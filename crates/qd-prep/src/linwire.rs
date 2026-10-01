//! The file formats between `tools/linear_control_native.py` and the `ngrams` / `linfit`
//! subcommands.
//!
//! Little-endian throughout, every count checked against the bytes present, and a file with
//! bytes left over refused -- as [`crate::wire`] does for MinHash, and for the same reason: a
//! reader that stopped early would fit a prefix of the corpus and say nothing.
//!
//! `ngrams` request (`QDPNGIN1`) and reply (`QDPNGOK1`):
//!
//! ```text
//! magic 8 | n_min u32 | n_max u32 | dim u32 | DOCS
//! magic 8 | n_rows u64 | n_cols u32 | nnz u64 | indptr (n_rows+1) x u64
//!         | indices nnz x u32 | data nnz x f64
//! ```
//!
//! `linfit` request (`QDPLFIN1`):
//!
//! ```text
//! magic 8 | features u8 (1: n-gram documents, 2: CSR rows, 3: option pairs)
//!         | features 1, 3: n_min u32, n_max u32, dim u32 / features 2: n_cols u32
//!         | n_classes u32 | max_iter u32 | tol f64 | lr f64 | n_grid u32, n_grid x f64
//!         | training ROWS | labels n x u32 | features 3 only: option rows n x u32
//!         | n_val u64 | order n x u64 | w0 (d*k) x f64
//!         | evaluation ROWS
//! ```
//!
//! where ROWS is DOCS for features 1, CSR for features 2 and PAIRS for features 3:
//!
//! ```text
//! DOCS:  n u64 | byte lengths n x u64 | the documents' UTF-8 bytes, concatenated
//! CSR:   n u64 | indptr (n+1) x u64 | indices indptr[n] x u32 | data indptr[n] x f64
//! PAIRS: DOCS (each example's pair text) | DOCS (each example's option text), the same n
//! ```
//!
//! Features 3 is the per-option control (`qd_train.option_control`): each PAIRS example is
//! hashed as two n-gram rows, stacked into `d = 2 * dim` columns (the pair text's in `[0, dim)`,
//! the option's in `[dim, 2 * dim)`); `n_classes` must be 2; the option rows name each training
//! example's row; and the grid is selected by [`linfit::Selection::BestOption`]. Features 1 and
//! 2 are unchanged by its existence, byte for byte in both directions.
//!
//! `linfit` reply (`QDPLFOK1`):
//!
//! ```text
//! magic 8 | n_classes u32 | n_cols u32 | n_grid u32
//!         | per grid value: l2 f64, val_correct u64, val_total u64, converged u8,
//!           iterations u32, grad_norm f64
//!         | selected u32 | converged u8 | iterations u32 | grad_norm f64
//!         | n_history u32, n_history x f64 | w (d*k) x f64 | b k x f64
//!         | n_eval u64 | logits (n_eval*k) x f64
//! ```

use std::time::Instant;

use crate::linfit::{self, Fitted, Hyper, Selection};
use crate::ngram::{self, Csr, NGramSpec};
use crate::wire::Cursor;

/// A request past this is refused before it is read. Not [`crate::wire::MAX_INPUT_BYTES`]
/// (4 GiB, sized for MinHash's shingle sets): a linear-control request carries a task's
/// documents themselves, and on the full mixture the `knowledge.multiple_choice` task alone
/// is 5,422,458,422 bytes (J1's eval row 09ff303f on the GH200, 2026-10-01). The shared bound
/// refused it after six tasks had fitted, which lost the whole control. 32 GiB is six times
/// that request and a small fraction of the box's 525 GB.
pub const MAX_INPUT_BYTES: u64 = 32 << 30;

pub const NGRAMS_INPUT_MAGIC: &[u8; 8] = b"QDPNGIN1";
pub const NGRAMS_OUTPUT_MAGIC: &[u8; 8] = b"QDPNGOK1";
pub const LINFIT_INPUT_MAGIC: &[u8; 8] = b"QDPLFIN1";
pub const LINFIT_OUTPUT_MAGIC: &[u8; 8] = b"QDPLFOK1";

/// Rows per block; also keeps every row index below `u32::MAX` for the transposed copy.
pub const MAX_ROWS: u64 = 1 << 24;
/// More classes than this is not a label space a char-n-gram control is fitted on.
pub const MAX_CLASSES: u32 = 1 << 16;
/// `d * k` weights at most: 4 GiB of float64.
pub const MAX_WEIGHTS: u64 = 1 << 29;
/// More L2 values than this is not the reference's grid (it has four).
pub const MAX_GRID: u32 = 64;
/// An iteration budget past this is a mistake, not a schedule (the control uses 6,000).
pub const MAX_ITER: u32 = 10_000_000;
/// Columns of a CSR block: as wide as the widest n-gram space.
pub const MAX_COLS: u32 = ngram::MAX_DIM;

impl<'b> Cursor<'b> {
    fn u8(&mut self, what: &str) -> Result<u8, String> {
        let r = self.take(1, what)?;
        Ok(self.buf[r.start])
    }

    fn f64(&mut self, what: &str) -> Result<f64, String> {
        Ok(f64::from_bits(self.u64(what)?))
    }

    /// `n` little-endian values of `width` bytes, the byte count checked before anything is
    /// allocated, so a forged count cannot ask for more memory than the file holds.
    fn array<T>(
        &mut self,
        n: u64,
        width: usize,
        what: &str,
        decode: impl Fn(&[u8]) -> T,
    ) -> Result<Vec<T>, String> {
        let bytes = usize::try_from(n)
            .ok()
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| format!("{what}: {n} values overflow"))?;
        let r = self.take(bytes, what)?;
        Ok(self.buf[r].chunks_exact(width).map(decode).collect())
    }

    fn u32s(&mut self, n: u64, what: &str) -> Result<Vec<u32>, String> {
        self.array(n, 4, what, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64s_checked(&mut self, n: u64, what: &str) -> Result<Vec<u64>, String> {
        self.array(n, 8, what, |b| {
            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        })
    }

    fn f64s(&mut self, n: u64, what: &str) -> Result<Vec<f64>, String> {
        Ok(self
            .u64s_checked(n, what)?
            .into_iter()
            .map(f64::from_bits)
            .collect())
    }

    fn row_count(&mut self, what: &str) -> Result<u64, String> {
        let n = self.u64(what)?;
        if n > MAX_ROWS {
            return Err(format!("{what}: {n} rows; the bound is {MAX_ROWS}"));
        }
        Ok(n)
    }

    /// DOCS: the documents as `&str`, each checked to be UTF-8.
    fn docs(&mut self, what: &str) -> Result<Vec<&'b str>, String> {
        let n = self.row_count(what)?;
        let lengths = self.u64s_checked(n, what)?;
        let total = lengths
            .iter()
            .try_fold(0u64, |acc, l| acc.checked_add(*l))
            .ok_or_else(|| format!("{what}: document lengths overflow"))?;
        let all = self.take(
            usize::try_from(total).map_err(|e| format!("{what}: {e}"))?,
            what,
        )?;
        let buf: &'b [u8] = self.buf;
        let mut at = all.start;
        let mut out = Vec::with_capacity(lengths.len());
        for (i, len) in lengths.into_iter().enumerate() {
            // Each length is at most `total`, which fit a usize above.
            let end = at + len as usize;
            let doc = std::str::from_utf8(&buf[at..end])
                .map_err(|e| format!("{what}: document {i} is not UTF-8 ({e})"))?;
            out.push(doc);
            at = end;
        }
        Ok(out)
    }

    /// CSR: validated as [`Csr::validate`] validates it.
    fn csr(&mut self, n_cols: usize, what: &str) -> Result<Csr, String> {
        let n = self.row_count(what)?;
        let indptr = self
            .u64s_checked(n + 1, what)?
            .into_iter()
            .map(|v| usize::try_from(v).map_err(|e| format!("{what}: {e}")))
            .collect::<Result<Vec<usize>, String>>()?;
        let nnz = *indptr.last().unwrap_or(&0) as u64;
        let indices = self.u32s(nnz, what)?;
        let data = self.f64s(nnz, what)?;
        let csr = Csr {
            n_cols,
            indptr,
            indices,
            data,
        };
        csr.validate().map_err(|e| format!("{what}: {e}"))?;
        Ok(csr)
    }
}

fn check_size(buf: &[u8]) -> Result<(), String> {
    if buf.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; the bound is {MAX_INPUT_BYTES}",
            buf.len()
        ));
    }
    Ok(())
}

fn expect_magic(c: &mut Cursor<'_>, magic: &[u8; 8]) -> Result<(), String> {
    let r = c.take(8, "the magic")?;
    if &c.buf[r] != magic {
        return Err(format!(
            "input does not start with {:?}",
            String::from_utf8_lossy(magic)
        ));
    }
    Ok(())
}

fn expect_end(c: &Cursor<'_>) -> Result<(), String> {
    if c.at != c.buf.len() {
        return Err(format!(
            "{} bytes follow the request; refusing a file that is not exactly what its \
             header describes",
            c.buf.len() - c.at
        ));
    }
    Ok(())
}

fn spec(c: &mut Cursor<'_>) -> Result<NGramSpec, String> {
    let n_min = c.u32("n_min")?;
    let n_max = c.u32("n_max")?;
    let dim = c.u32("dim")?;
    NGramSpec::new(n_min, n_max, dim)
}

/// `ngrams`: hash a `QDPNGIN1` request and return the `QDPNGOK1` reply and a summary line.
pub fn run_ngrams(buf: &[u8], threads: usize) -> Result<(Vec<u8>, String), String> {
    check_size(buf)?;
    let mut c = Cursor::new(buf);
    expect_magic(&mut c, NGRAMS_INPUT_MAGIC)?;
    let spec = spec(&mut c)?;
    let docs = c.docs("documents")?;
    expect_end(&c)?;
    let started = Instant::now();
    let csr = ngram::transform(spec, &docs, threads)?;
    let secs = started.elapsed().as_secs_f64();
    let mut out = Vec::with_capacity(28 + csr.indptr.len() * 8 + csr.nnz() * 12);
    out.extend_from_slice(NGRAMS_OUTPUT_MAGIC);
    out.extend_from_slice(&(csr.n_rows() as u64).to_le_bytes());
    out.extend_from_slice(&spec.dim.to_le_bytes());
    out.extend_from_slice(&(csr.nnz() as u64).to_le_bytes());
    for v in &csr.indptr {
        out.extend_from_slice(&(*v as u64).to_le_bytes());
    }
    for v in &csr.indices {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &csr.data {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok((
        out,
        format!(
            "qd-prep ngrams: {} documents -> {} nonzeros in {secs:.3} s on {threads} thread(s)",
            docs.len(),
            csr.nnz()
        ),
    ))
}

/// How a `linfit` request lays out its features.
#[derive(Clone, Copy)]
enum Layout {
    Docs(NGramSpec),
    Rows,
    Pairs(NGramSpec),
}

/// One side (training or evaluation) of a `linfit` request, as read.
enum Side<'b> {
    Docs(Vec<&'b str>),
    Rows(Csr),
    /// Each example's pair text and option text.
    Pairs(Vec<&'b str>, Vec<&'b str>),
}

impl<'b> Side<'b> {
    fn read(c: &mut Cursor<'b>, layout: Layout, n_cols: usize, what: &str) -> Result<Self, String> {
        Ok(match layout {
            Layout::Docs(_) => Side::Docs(c.docs(&format!("{what} documents"))?),
            Layout::Rows => Side::Rows(c.csr(n_cols, &format!("{what} rows"))?),
            Layout::Pairs(_) => {
                let pairs = c.docs(&format!("{what} pair texts"))?;
                let options = c.docs(&format!("{what} option texts"))?;
                if pairs.len() != options.len() {
                    return Err(format!(
                        "{} {what} pair texts and {} option texts; each example has one of each",
                        pairs.len(),
                        options.len()
                    ));
                }
                Side::Pairs(pairs, options)
            }
        })
    }

    fn len(&self) -> usize {
        match self {
            Side::Docs(d) => d.len(),
            Side::Rows(r) => r.n_rows(),
            Side::Pairs(p, _) => p.len(),
        }
    }

    /// The side's feature rows: hashed, as read, or hashed twice and stacked.
    fn features(self, layout: Layout, threads: usize) -> Result<Csr, String> {
        match (self, layout) {
            (Side::Docs(d), Layout::Docs(spec)) => ngram::transform(spec, &d, threads),
            (Side::Rows(r), Layout::Rows) => Ok(r),
            (Side::Pairs(p, o), Layout::Pairs(spec)) => Ok(hstack(
                &ngram::transform(spec, &p, threads)?,
                &ngram::transform(spec, &o, threads)?,
            )),
            _ => Err("a side was read under another layout".to_string()),
        }
    }
}

/// `[left | right]`, as `qd_train.option_control.hstack` stacks them: each row's `left` entries,
/// then its `right` entries at columns shifted by `left`'s width. Both are column-sorted, so the
/// result is.
fn hstack(left: &Csr, right: &Csr) -> Csr {
    let width = left.n_cols as u32;
    let mut indptr = Vec::with_capacity(left.n_rows() + 1);
    indptr.push(0usize);
    let mut indices = Vec::with_capacity(left.nnz() + right.nnz());
    let mut data = Vec::with_capacity(left.nnz() + right.nnz());
    for r in 0..left.n_rows() {
        let (lc, lv) = left.row(r);
        let (rc, rv) = right.row(r);
        indices.extend_from_slice(lc);
        indices.extend(rc.iter().map(|c| c + width));
        data.extend_from_slice(lv);
        data.extend_from_slice(rv);
        indptr.push(indices.len());
    }
    Csr {
        n_cols: left.n_cols + right.n_cols,
        indptr,
        indices,
        data,
    }
}

/// `linfit`: fit a `QDPLFIN1` request and return the `QDPLFOK1` reply and a summary line.
pub fn run_linfit(buf: &[u8], threads: usize) -> Result<(Vec<u8>, String), String> {
    check_size(buf)?;
    let mut c = Cursor::new(buf);
    expect_magic(&mut c, LINFIT_INPUT_MAGIC)?;
    let kind = c.u8("the features tag")?;
    let (layout, n_cols) = match kind {
        1 => {
            let s = spec(&mut c)?;
            (Layout::Docs(s), s.dim as usize)
        }
        2 => {
            let n_cols = c.u32("n_cols")?;
            if !(1..=MAX_COLS).contains(&n_cols) {
                return Err(format!("n_cols {n_cols} is outside 1..={MAX_COLS}"));
            }
            (Layout::Rows, n_cols as usize)
        }
        3 => {
            let s = spec(&mut c)?;
            let n_cols = 2 * u64::from(s.dim);
            if n_cols > u64::from(MAX_COLS) {
                return Err(format!(
                    "two blocks of {} columns are {n_cols}; the bound is {MAX_COLS}",
                    s.dim
                ));
            }
            (Layout::Pairs(s), n_cols as usize)
        }
        other => {
            return Err(format!(
                "features tag {other} is not 1 (documents), 2 (rows) or 3 (option pairs)"
            ));
        }
    };
    let n_classes = c.u32("n_classes")?;
    if !(2..=MAX_CLASSES).contains(&n_classes) {
        return Err(format!(
            "n_classes {n_classes} is outside 2..={MAX_CLASSES}; the reference needs two"
        ));
    }
    if matches!(layout, Layout::Pairs(_)) && n_classes != 2 {
        return Err(format!(
            "an option-pairs request is a binary scorer's; n_classes is {n_classes}"
        ));
    }
    let k = n_classes as usize;
    if (n_cols as u64) * u64::from(n_classes) > MAX_WEIGHTS {
        return Err(format!(
            "{n_cols} x {n_classes} weights; the bound is {MAX_WEIGHTS}"
        ));
    }
    let max_iter = c.u32("max_iter")?;
    if max_iter > MAX_ITER {
        return Err(format!("max_iter {max_iter}; the bound is {MAX_ITER}"));
    }
    let tol = c.f64("tol")?;
    let lr = c.f64("lr")?;
    if !(tol.is_finite() && tol >= 0.0) || !(lr.is_finite() && lr > 0.0) {
        return Err(format!(
            "tol {tol} must be finite and >= 0, lr {lr} finite and > 0"
        ));
    }
    let n_grid = c.u32("n_grid")?;
    if !(1..=MAX_GRID).contains(&n_grid) {
        return Err(format!("{n_grid} L2 values; the grid holds 1..={MAX_GRID}"));
    }
    let l2_grid = c.f64s(u64::from(n_grid), "the L2 grid")?;
    if let Some(v) = l2_grid.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
        return Err(format!("L2 value {v} is not finite and >= 0"));
    }

    let train = Side::read(&mut c, layout, n_cols, "training")?;
    let n = train.len();
    let labels = c.u32s(n as u64, "labels")?;
    if let Some(l) = labels.iter().find(|l| **l >= n_classes) {
        return Err(format!("label {l} is outside 0..{n_classes}"));
    }
    let mut seen = vec![false; k];
    for l in &labels {
        seen[*l as usize] = true;
    }
    if let Some(missing) = seen.iter().position(|s| !s) {
        return Err(format!(
            "class {missing} has no training row; the reference's classes are the labels present"
        ));
    }
    let groups = match layout {
        Layout::Pairs(_) => Some(c.u32s(n as u64, "the option rows")?),
        Layout::Docs(_) | Layout::Rows => None,
    };
    let n_val = c.u64("n_val")?;
    let order = c
        .u64s_checked(n as u64, "the permutation")?
        .into_iter()
        .map(|v| usize::try_from(v).map_err(|e| e.to_string()))
        .collect::<Result<Vec<usize>, String>>()?;
    let mut hit = vec![false; n];
    for &i in &order {
        if i >= n || std::mem::replace(&mut hit[i], true) {
            return Err(format!(
                "the order is not a permutation of 0..{n}: {i} is out of range or repeated"
            ));
        }
    }
    let w0 = c.f64s((n_cols * k) as u64, "the initial weights")?;
    if let Some(i) = w0.iter().position(|v| !v.is_finite()) {
        return Err(format!("initial weight {i} is not finite"));
    }
    let eval = Side::read(&mut c, layout, n_cols, "evaluation")?;
    expect_end(&c)?;

    let hyper = Hyper {
        n_classes: k,
        max_iter,
        tol,
        lr,
        l2_grid,
    };
    let started = Instant::now();
    let train = train.features(layout, threads)?;
    let eval = eval.features(layout, threads)?;
    let hashed_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let n_val = usize::try_from(n_val).map_err(|e| e.to_string())?;
    let selection = match &groups {
        Some(groups) => Selection::BestOption { groups },
        None => Selection::Top1,
    };
    let fitted = linfit::fit(
        &train, &labels, &order, n_val, &w0, &eval, &hyper, selection, threads,
    )?;
    let fit_s = started.elapsed().as_secs_f64();
    let mut summary = summary(&fitted, &train, eval.n_rows(), hashed_s, fit_s, threads);
    if groups.is_some() {
        summary.push_str("; the grid counts validation ROWS, by each row's best option");
    }
    Ok((encode_linfit(&fitted, n_cols, k), summary))
}

fn summary(
    f: &Fitted,
    train: &Csr,
    n_eval: usize,
    hashed_s: f64,
    fit_s: f64,
    threads: usize,
) -> String {
    let grid: Vec<String> = f
        .grid
        .iter()
        .map(|g| {
            format!(
                "l2={} {}/{} {} in {}",
                g.l2,
                g.val_correct,
                g.val_total,
                if g.converged {
                    "converged"
                } else {
                    "UNCONVERGED"
                },
                g.iterations
            )
        })
        .collect();
    format!(
        "qd-prep linfit: {} rows x {} cols, {} nonzeros, {} eval rows; features {hashed_s:.2} s, \
         fit {fit_s:.2} s on {threads} thread(s); grid [{}]; refit l2={} {} in {} iterations \
         (grad norm {:.6e})",
        train.n_rows(),
        train.n_cols,
        train.nnz(),
        n_eval,
        grid.join("; "),
        f.grid[f.selected].l2,
        if f.refit.converged {
            "converged"
        } else {
            "UNCONVERGED"
        },
        f.refit.iterations,
        f.refit.grad_norm,
    )
}

fn push_f64s(out: &mut Vec<u8>, values: &[f64]) {
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

/// The `QDPLFOK1` bytes for a fit. Counts are bounded by the request's own bounds, so the
/// narrowing casts below cannot truncate.
pub fn encode_linfit(f: &Fitted, n_cols: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(LINFIT_OUTPUT_MAGIC);
    out.extend_from_slice(&(k as u32).to_le_bytes());
    out.extend_from_slice(&(n_cols as u32).to_le_bytes());
    out.extend_from_slice(&(f.grid.len() as u32).to_le_bytes());
    for g in &f.grid {
        out.extend_from_slice(&g.l2.to_bits().to_le_bytes());
        out.extend_from_slice(&g.val_correct.to_le_bytes());
        out.extend_from_slice(&g.val_total.to_le_bytes());
        out.push(u8::from(g.converged));
        out.extend_from_slice(&g.iterations.to_le_bytes());
        out.extend_from_slice(&g.grad_norm.to_bits().to_le_bytes());
    }
    out.extend_from_slice(&(f.selected as u32).to_le_bytes());
    out.push(u8::from(f.refit.converged));
    out.extend_from_slice(&f.refit.iterations.to_le_bytes());
    out.extend_from_slice(&f.refit.grad_norm.to_bits().to_le_bytes());
    out.extend_from_slice(&(f.refit.history.len() as u32).to_le_bytes());
    push_f64s(&mut out, &f.refit.history);
    push_f64s(&mut out, &f.refit.w);
    push_f64s(&mut out, &f.refit.b);
    out.extend_from_slice(&((f.eval_logits.len() / k) as u64).to_le_bytes());
    push_f64s(&mut out, &f.eval_logits);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // A runtime test rather than a `const` assertion on purpose: it is the check that fails
    // against the code that shared the MinHash bound, which a compile error would not report.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn the_bound_admits_the_measured_mixture_request() {
        // J1's knowledge.multiple_choice request on the GH200 (2026-10-01): refused at the
        // MinHash bound, which this module used to share, after six tasks had fitted.
        const MEASURED_MMLU_REQUEST: u64 = 5_422_458_422;
        assert!(MAX_INPUT_BYTES >= MEASURED_MMLU_REQUEST);
        assert!(MAX_INPUT_BYTES > crate::wire::MAX_INPUT_BYTES);
    }

    fn docs_block(docs: &[&str]) -> Vec<u8> {
        let mut v = (docs.len() as u64).to_le_bytes().to_vec();
        for d in docs {
            v.extend_from_slice(&(d.len() as u64).to_le_bytes());
        }
        for d in docs {
            v.extend_from_slice(d.as_bytes());
        }
        v
    }

    fn linfit_request(docs: &[&str], labels: &[u32], order: &[u64], n_val: u64) -> Vec<u8> {
        let mut v = LINFIT_INPUT_MAGIC.to_vec();
        v.push(1);
        for x in [3u32, 5, 16] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&50u32.to_le_bytes());
        v.extend_from_slice(&1e-4f64.to_bits().to_le_bytes());
        v.extend_from_slice(&0.05f64.to_bits().to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        for l2 in [1e-3f64, 1e-1] {
            v.extend_from_slice(&l2.to_bits().to_le_bytes());
        }
        v.extend(docs_block(docs));
        for l in labels {
            v.extend_from_slice(&l.to_le_bytes());
        }
        v.extend_from_slice(&n_val.to_le_bytes());
        for o in order {
            v.extend_from_slice(&o.to_le_bytes());
        }
        for i in 0..32 {
            v.extend_from_slice(&(f64::from(i) * 1e-3).to_bits().to_le_bytes());
        }
        v.extend(docs_block(&docs[..2]));
        v
    }

    const DOCS: [&str; 6] = [
        "alpha beta",
        "gamma delta",
        "alpha gamma",
        "beta beta",
        "delta",
        "x",
    ];

    #[test]
    fn a_well_formed_request_fits_and_replies_with_its_own_shape() {
        let req = linfit_request(&DOCS, &[0, 1, 0, 1, 1, 0], &[5, 4, 3, 2, 1, 0], 1);
        let (reply, line) = run_linfit(&req, 3).expect("fits");
        assert_eq!(&reply[..8], LINFIT_OUTPUT_MAGIC);
        assert!(line.contains("2 eval rows"), "{line}");
        assert_eq!(run_linfit(&req, 1).expect("fits").0, reply, "thread count");
    }

    #[test]
    fn a_request_that_is_not_exactly_its_header_is_refused() {
        let good = linfit_request(&DOCS, &[0, 1, 0, 1, 1, 0], &[5, 4, 3, 2, 1, 0], 1);
        for cut in [0, 7, 8, 9, 30, good.len() - 1] {
            assert!(run_linfit(&good[..cut], 1).is_err(), "truncated at {cut}");
        }
        let mut longer = good.clone();
        longer.push(0);
        assert!(run_linfit(&longer, 1).is_err(), "a trailing byte");
        let mut magic = good.clone();
        magic[0] = b'X';
        assert!(run_linfit(&magic, 1).is_err(), "magic");
    }

    #[test]
    fn labels_permutations_and_splits_the_reference_could_not_produce_are_refused() {
        let ok = [0u32, 1, 0, 1, 1, 0];
        let perm = [5u64, 4, 3, 2, 1, 0];
        assert!(run_linfit(&linfit_request(&DOCS, &[0, 1, 0, 1, 2, 0], &perm, 1), 1).is_err());
        assert!(
            run_linfit(&linfit_request(&DOCS, &[0; 6], &perm, 1), 1).is_err(),
            "one class"
        );
        assert!(run_linfit(&linfit_request(&DOCS, &ok, &[5, 4, 3, 2, 1, 1], 1), 1).is_err());
        assert!(run_linfit(&linfit_request(&DOCS, &ok, &[6, 4, 3, 2, 1, 0], 1), 1).is_err());
        assert!(
            run_linfit(&linfit_request(&DOCS, &ok, &perm, 0), 1).is_err(),
            "n_val 0"
        );
        assert!(
            run_linfit(&linfit_request(&DOCS, &ok, &perm, 6), 1).is_err(),
            "n_val n"
        );
    }

    /// A features-2 request over six two-entry rows, k = 3.
    fn rows_request() -> Vec<u8> {
        fn csr(rows: &[[u32; 2]]) -> Vec<u8> {
            let mut v = (rows.len() as u64).to_le_bytes().to_vec();
            for p in 0..=rows.len() as u64 {
                v.extend_from_slice(&(2 * p).to_le_bytes());
            }
            for r in rows {
                for c in r {
                    v.extend_from_slice(&c.to_le_bytes());
                }
            }
            for _ in rows {
                for x in [0.6f64, 0.8] {
                    v.extend_from_slice(&x.to_bits().to_le_bytes());
                }
            }
            v
        }
        let mut v = LINFIT_INPUT_MAGIC.to_vec();
        v.push(2);
        v.extend_from_slice(&8u32.to_le_bytes());
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&1e-4f64.to_bits().to_le_bytes());
        v.extend_from_slice(&0.05f64.to_bits().to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        for l2 in [1e-3f64, 1e-1] {
            v.extend_from_slice(&l2.to_bits().to_le_bytes());
        }
        v.extend(csr(&[[0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [0, 5]]));
        for l in [0u32, 1, 2, 0, 1, 2] {
            v.extend_from_slice(&l.to_le_bytes());
        }
        v.extend_from_slice(&2u64.to_le_bytes());
        for o in [5u64, 4, 3, 2, 1, 0] {
            v.extend_from_slice(&o.to_le_bytes());
        }
        for i in 0..24 {
            v.extend_from_slice(&(f64::from(i) * 1e-3).to_bits().to_le_bytes());
        }
        v.extend(csr(&[[1, 2], [6, 7]]));
        v
    }

    fn blake2b_256_hex(bytes: &[u8]) -> String {
        let digest = crate::blake2b::Keyed::new(&[], 32)
            .expect("unkeyed BLAKE2b-256")
            .digest(bytes);
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn documents_and_rows_requests_reply_exactly_as_before_option_pairs_existed() {
        // The digests of the replies the binary built at 88373b8 (before features 3) wrote for
        // these two requests, on 2 threads: hashlib.blake2b(reply, digest_size=32).
        let tag1 = linfit_request(&DOCS, &[0, 1, 0, 1, 1, 0], &[5, 4, 3, 2, 1, 0], 1);
        let (reply, _) = run_linfit(&tag1, 2).expect("fits");
        assert_eq!(
            blake2b_256_hex(&reply),
            "2e19eaef34b741ea34967bf1edc6f44a119b8b497a7f52eb64326f048a6937d4"
        );
        let (reply, _) = run_linfit(&rows_request(), 2).expect("fits");
        assert_eq!(
            blake2b_256_hex(&reply),
            "c5d1e88b6d76dd9f689cfe2793b333555ce387f58072277773cb7261b2db0e66"
        );
    }

    /// Six rows of three options; row `r`'s gold is option `r % 3`, and its option text says so.
    struct OptionRequest {
        pairs: Vec<String>,
        options: Vec<String>,
        labels: Vec<u32>,
        groups: Vec<u32>,
        order: Vec<u64>,
        n_val: u64,
        n_classes: u32,
    }

    impl OptionRequest {
        fn new() -> Self {
            let (mut pairs, mut options, mut labels, mut groups) = (vec![], vec![], vec![], vec![]);
            for r in 0..6u32 {
                for j in 0..3u32 {
                    let gold = j == r % 3;
                    let option = format!("{} {j}", if gold { "right" } else { "wrong" });
                    pairs.push(format!("question {r}\n{option}"));
                    options.push(option);
                    labels.push(u32::from(gold));
                    groups.push(r);
                }
            }
            // Rows 4 and 1 are the validation slice, then the rest; options in shown order.
            let rows = [4u64, 1, 0, 2, 3, 5];
            let order = rows.iter().flat_map(|&r| 3 * r..3 * r + 3).collect();
            Self {
                pairs,
                options,
                labels,
                groups,
                order,
                n_val: 6,
                n_classes: 2,
            }
        }

        fn bytes(&self) -> Vec<u8> {
            fn refs(v: &[String]) -> Vec<&str> {
                v.iter().map(String::as_str).collect()
            }
            let (pairs, options) = (refs(&self.pairs), refs(&self.options));
            let mut v = LINFIT_INPUT_MAGIC.to_vec();
            v.push(3);
            for x in [3u32, 5, 16, self.n_classes, 60] {
                v.extend_from_slice(&x.to_le_bytes());
            }
            v.extend_from_slice(&1e-4f64.to_bits().to_le_bytes());
            v.extend_from_slice(&0.05f64.to_bits().to_le_bytes());
            v.extend_from_slice(&2u32.to_le_bytes());
            for l2 in [1e-3f64, 1e-1] {
                v.extend_from_slice(&l2.to_bits().to_le_bytes());
            }
            v.extend(docs_block(&pairs));
            v.extend(docs_block(&options));
            for l in &self.labels {
                v.extend_from_slice(&l.to_le_bytes());
            }
            for g in &self.groups {
                v.extend_from_slice(&g.to_le_bytes());
            }
            v.extend_from_slice(&self.n_val.to_le_bytes());
            for o in &self.order {
                v.extend_from_slice(&o.to_le_bytes());
            }
            for i in 0..32 * self.n_classes {
                v.extend_from_slice(&(f64::from(i) * 1e-3).to_bits().to_le_bytes());
            }
            v.extend(docs_block(&pairs[..4]));
            v.extend(docs_block(&options[..4]));
            v
        }
    }

    #[test]
    fn an_option_pairs_request_is_two_stacked_blocks_selected_by_row() {
        let req = OptionRequest::new().bytes();
        let (reply, line) = run_linfit(&req, 3).expect("fits");
        assert_eq!(&reply[..8], LINFIT_OUTPUT_MAGIC);
        let u32_at = |at: usize| u32::from_le_bytes(reply[at..at + 4].try_into().expect("4"));
        let u64_at = |at: usize| u64::from_le_bytes(reply[at..at + 8].try_into().expect("8"));
        assert_eq!(
            (u32_at(8), u32_at(12), u32_at(16)),
            (2, 32, 2),
            "k, 2 x dim, grid"
        );
        // The first grid point: l2 f64, val_correct u64, val_total u64 -- two validation rows.
        assert_eq!(
            u64_at(20 + 16),
            2,
            "val_total counts rows, not the six options"
        );
        assert!(line.contains("4 eval rows"), "{line}");
        assert!(line.contains("validation ROWS"), "{line}");
        assert!(line.contains("18 rows x 32 cols"), "{line}");
        assert_eq!(run_linfit(&req, 1).expect("fits").0, reply, "thread count");
    }

    #[test]
    fn option_pairs_requests_the_reference_could_not_produce_are_refused() {
        let refused = |r: &OptionRequest| run_linfit(&r.bytes(), 1).expect_err("refused");
        let mut three = OptionRequest::new();
        three.n_classes = 3;
        assert!(
            refused(&three).contains("binary scorer"),
            "{}",
            refused(&three)
        );
        let mut straddle = OptionRequest::new();
        straddle.n_val = 5;
        assert!(refused(&straddle).contains("both sides"));
        let mut two_gold = OptionRequest::new();
        two_gold.labels[12] = 1;
        two_gold.labels[13] = 1;
        assert!(refused(&two_gold).contains("positive"));
        let mut short = OptionRequest::new();
        short.options.pop();
        assert!(refused(&short).contains("option texts"));
        let good = OptionRequest::new().bytes();
        for cut in [9, good.len() / 2, good.len() - 1] {
            assert!(run_linfit(&good[..cut], 1).is_err(), "truncated at {cut}");
        }
        let mut longer = good;
        longer.push(0);
        assert!(run_linfit(&longer, 1).is_err(), "a trailing byte");
    }

    #[test]
    fn a_document_that_is_not_utf8_is_refused() {
        let mut v = NGRAMS_INPUT_MAGIC.to_vec();
        for x in [3u32, 5, 16] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.extend_from_slice(&1u64.to_le_bytes());
        v.extend_from_slice(&2u64.to_le_bytes());
        v.extend_from_slice(&[0xED, 0xA0]);
        assert!(run_ngrams(&v, 1).expect_err("refused").contains("UTF-8"));
    }

    #[test]
    fn csr_rows_out_of_order_or_out_of_range_are_refused() {
        let mut c_ok = Vec::new();
        c_ok.extend_from_slice(&1u64.to_le_bytes());
        for p in [0u64, 2] {
            c_ok.extend_from_slice(&p.to_le_bytes());
        }
        for col in [0u32, 3] {
            c_ok.extend_from_slice(&col.to_le_bytes());
        }
        for v in [0.5f64, 0.25] {
            c_ok.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        assert!(Cursor::new(&c_ok).csr(4, "rows").is_ok());
        assert!(Cursor::new(&c_ok).csr(3, "rows").is_err(), "column 3 of 3");
        let mut swapped = c_ok.clone();
        swapped[24..28].copy_from_slice(&3u32.to_le_bytes());
        swapped[28..32].copy_from_slice(&0u32.to_le_bytes());
        assert!(
            Cursor::new(&swapped).csr(4, "rows").is_err(),
            "descending columns"
        );
        let mut nan = c_ok.clone();
        nan[32..40].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());
        assert!(Cursor::new(&nan).csr(4, "rows").is_err(), "NaN");
    }
}
