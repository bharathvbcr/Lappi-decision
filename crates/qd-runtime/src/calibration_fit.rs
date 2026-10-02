//! Fitting the [`CalibrationTable`](crate::calibration::CalibrationTable) this crate serves.
//!
//! The Rust owner of what `python/qd_train/calibration_fit.py` defines — `fit_temperature`,
//! `fit_noul_margin`, `expected_calibration_error` / `ece_gate`, and the bridge
//! `probability_cutoff_from_nonconformity` — together with
//! `qd_train.eval_harness.split_conformal_threshold`, whose nonconformity quantile that bridge
//! converts. `qd-calib-fit` (`src/bin/qd_calib_fit.rs`) is the only caller; the Python module
//! is now the reference oracle `python/tests/test_calib_fit_parity.py` pins this against.
//!
//! # Bit-exact, and what that takes
//!
//! The parity target is equality, not a tolerance. Three numpy behaviours are reproduced
//! exactly, each measured on numpy 2.5.0 / macOS arm64 before it was written down here:
//!
//! * **Summation.** `np.add.reduce` over float64 is numpy's *pairwise* sum (blocks of 8,
//!   unrolled, split in halves above 128 elements), initialised at 0.0, over the whole axis — no
//!   8192-element buffer chunking for a contiguous array. `ndarray.mean` is that sum divided by
//!   the count. A sequential sum differs in the last bits, and the ternary search below compares
//!   two nearly equal NLLs 400 times, so a last-bit difference moves the fitted temperature.
//!   [`pairwise_sum`] is the port.
//! * **`exp` / `log`.** `np.exp` and `np.log` on float64 return what the C library's `exp` and
//!   `log` return on this host (no SIMD float64 path on NEON). Rust's `f64::exp` / `f64::ln` call
//!   the same library on macOS. On another libm (glibc on the aarch64 box) both sides are
//!   deterministic but may differ from macOS in the last bit; parity was established on macOS.
//! * **The quantile.** `np.quantile(..., method="higher")` reads `sorted[ceil((n - 1) * q)]`,
//!   with `q = min(1, ceil((n + 1)(1 - alpha)) / n)` computed in float64 exactly as Python does.
//!
//! # Three quantities, two of them called a threshold
//!
//! `GAP-CALIB-CONFORMAL-SCALE-TWO-MEANINGS`: [`split_conformal_nonconformity`] returns `q̂` on
//! the **nonconformity** scale (`1 - p(true class)`), small for a good model.
//! [`CalibrationEntry::conformal_quantile`] is a **probability** cutoff (`p >= cutoff` enters the
//! set), `1 - q̂`. The margin `p_top - p_second` is the third quantity and the basis of
//! `noul_margin`. [`fit_entry`] takes the conversion through
//! [`probability_cutoff_from_nonconformity`] and nowhere else, so no caller holding `q̂` can
//! write it into the table directly.
//!
//! # What the fit reads
//!
//! Logits are `f32`, because [`calibrate`] — the function `answer.rs` serves through — takes
//! `&[f32]`. A verdict file's `row_logits` are bf16 widened to fp32, so this is the identity on
//! real data; on anything else it is the runtime's view of the number, not the file's.

use crate::calibration::{CalibrationEntry, calibrate};

/// `fit_temperature`'s lower search bound (`calibration_fit.py`, `lo`).
pub const TEMPERATURE_LO: f64 = 0.05;
/// `fit_temperature`'s upper search bound (`hi`).
pub const TEMPERATURE_HI: f64 = 20.0;
/// `fit_temperature`'s ternary-search iterations (`iters`).
pub const TEMPERATURE_ITERS: usize = 200;
/// `ece_gate`'s bin count. Read-only (CLAUDE.md rule 2): reported against, never moved.
pub const ECE_BINS: usize = 15;
/// `ece_gate`'s sample floor, below which ECE is not run rather than reported.
pub const ECE_MIN_SAMPLES: usize = 100;
/// `ece_gate`'s bar.
pub const ECE_THRESHOLD: f64 = 0.05;
/// `split_conformal_threshold`'s default `alpha`: target coverage 0.90.
pub const DEFAULT_ALPHA: f64 = 0.1;
/// `fit_noul_margin`'s default retained-set precision.
pub const DEFAULT_TARGET_PRECISION: f64 = 0.95;

/// numpy's `PW_BLOCKSIZE`.
const PAIRWISE_BLOCK: usize = 128;

type Result<T> = std::result::Result<T, String>;

macro_rules! ensure {
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            return Err(format!($($arg)+));
        }
    };
}

/// numpy's float64 pairwise summation, exactly (`DOUBLE_pairwise_sum` in `loops_utils.h.src`).
///
/// Below 8 elements a sequential sum from 0.0; up to 128, eight interleaved accumulators
/// combined as `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))` and then the remainder; above that, split
/// at the largest multiple of 8 not above half and recurse.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for x in a {
            res += *x;
        }
        res
    } else if n <= PAIRWISE_BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// The letter rows of one table entry: one `(kind, rows)` shape, so every row has `width` logits.
#[derive(Debug, Clone, PartialEq)]
pub struct LetterRows {
    width: usize,
    logits: Vec<f32>,
    gold: Vec<usize>,
}

impl LetterRows {
    /// An empty set of rows `width` logits wide. A slot with fewer than two rows is not a question.
    pub fn new(width: usize) -> Result<Self> {
        ensure!(
            width >= 2,
            "a letter row needs at least 2 decode rows, got {width}"
        );
        Ok(Self {
            width,
            logits: Vec::new(),
            gold: Vec::new(),
        })
    }

    /// Append one row. Refuses a wrong width, a non-finite logit and a gold row outside the slot.
    pub fn push(&mut self, logits: &[f32], gold: usize) -> Result<()> {
        ensure!(
            logits.len() == self.width,
            "{} logits for a {}-row slot",
            logits.len(),
            self.width
        );
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "a logit is non-finite; a fit over it would fit nothing the runtime serves"
        );
        ensure!(
            gold < self.width,
            "gold row {gold} is outside a {}-row slot",
            self.width
        );
        self.logits.extend_from_slice(logits);
        self.gold.push(gold);
        Ok(())
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn len(&self) -> usize {
        self.gold.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gold.is_empty()
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.logits[i * self.width..(i + 1) * self.width]
    }

    pub fn gold(&self, i: usize) -> usize {
        self.gold[i]
    }

    /// The rows at `indices`, in the order given — order is part of the fit, because the NLL and
    /// every mean are pairwise sums.
    pub fn subset(&self, indices: &[usize]) -> Result<Self> {
        let mut out = Self::new(self.width)?;
        for &i in indices {
            ensure!(
                i < self.len(),
                "row index {i} is outside {} rows",
                self.len()
            );
            out.logits.extend_from_slice(self.row(i));
            out.gold.push(self.gold[i]);
        }
        Ok(out)
    }
}

/// The fitted temperature, and whether it is an interior optimum at all.
///
/// The search always returns a number; these say when that number is not a minimum the data
/// located. Measured on the NLL itself rather than on where the search ended, because the NLL
/// of an entry that is never wrong underflows to exactly 0 below some temperature and the
/// search then wanders off the bound across a flat plateau — it does not stay at
/// [`TEMPERATURE_LO`] (a never-wrong two-row entry ends at T = 0.0817).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureFit {
    pub temperature: f64,
    /// The NLL at [`TEMPERATURE_LO`] is no worse than at the fit: the minimum is at or below the
    /// lower bound. Every entry whose rows are all answered right lands here — sharpening a
    /// model that is never wrong always lowers its NLL.
    pub at_lower_bound: bool,
    /// The mirror case at [`TEMPERATURE_HI`]: an entry that is never right.
    pub at_upper_bound: bool,
}

/// Reused buffers for [`nll`], so 400 evaluations allocate once.
struct NllScratch {
    shifted: Vec<f64>,
    exps: Vec<f64>,
    per_row: Vec<f64>,
}

/// `fit_temperature`'s `nll(t)`, operation for operation:
/// `s = z / t; s = s - s.max(axis=1); -(s[rows, y] - log(exp(s).sum(axis=1))).mean()`.
fn nll(z: &[f64], width: usize, gold: &[usize], t: f64, s: &mut NllScratch) -> f64 {
    for (i, row) in z.chunks_exact(width).enumerate() {
        // `np.maximum.reduce` keeps the earlier element on a tie, which only matters for the
        // sign of a zero; `d = s - max` then equals for both signs, and exp(+-0) is 1. Every
        // logit is finite (`LetterRows::push`), so `>` is the whole comparison.
        let mut max = row[0] / t;
        for (shifted, z) in s.shifted.iter_mut().zip(row) {
            let v = z / t;
            *shifted = v;
            if v > max {
                max = v;
            }
        }
        for (shifted, e) in s.shifted.iter_mut().zip(s.exps.iter_mut()) {
            *shifted -= max;
            *e = shifted.exp();
        }
        let log_sum = pairwise_sum(&s.exps[..width]).ln();
        s.per_row[i] = s.shifted[gold[i]] - log_sum;
    }
    -(pairwise_sum(&s.per_row) / s.per_row.len() as f64)
}

/// The temperature minimising held-out NLL: ternary search on `log T` over
/// `[TEMPERATURE_LO, TEMPERATURE_HI]`, [`TEMPERATURE_ITERS`] iterations — `fit_temperature`.
pub fn fit_temperature(rows: &LetterRows) -> Result<TemperatureFit> {
    ensure!(
        !rows.is_empty(),
        "logits must be 2-D and non-empty, got 0 rows"
    );
    let width = rows.width;
    let z: Vec<f64> = rows.logits.iter().map(|v| f64::from(*v)).collect();
    let mut scratch = NllScratch {
        shifted: vec![0.0; width],
        exps: vec![0.0; width],
        per_row: vec![0.0; rows.len()],
    };
    let (lo, hi) = (TEMPERATURE_LO.ln(), TEMPERATURE_HI.ln());
    let (mut a, mut b) = (lo, hi);
    for _ in 0..TEMPERATURE_ITERS {
        let m1 = a + (b - a) / 3.0;
        let m2 = b - (b - a) / 3.0;
        if nll(&z, width, &rows.gold, m1.exp(), &mut scratch)
            < nll(&z, width, &rows.gold, m2.exp(), &mut scratch)
        {
            b = m2;
        } else {
            a = m1;
        }
    }
    let temperature = ((a + b) / 2.0).exp();
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "the temperature search produced {temperature}"
    );
    let at_fit = nll(&z, width, &rows.gold, temperature, &mut scratch);
    Ok(TemperatureFit {
        temperature,
        at_lower_bound: nll(&z, width, &rows.gold, TEMPERATURE_LO, &mut scratch) <= at_fit,
        at_upper_bound: nll(&z, width, &rows.gold, TEMPERATURE_HI, &mut scratch) <= at_fit,
    })
}

/// One row read through [`calibrate`] at one entry: the runtime's own view of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowScore {
    /// The winning row, first maximum (what `answer.rs` serves and `np.argmax` returns).
    pub top: usize,
    /// `p_top - p_second`: the reported `score`, compared against `noul_margin`.
    pub margin: f64,
    /// `p_top`: what ECE bins.
    pub confidence: f64,
    /// `p(gold)`: what the nonconformity score `1 - p(gold)` is made of.
    pub p_gold: f64,
    pub correct: bool,
    /// Rows in the conformal set (`p >= conformal_quantile`, plus the top row).
    pub set_size: usize,
    pub gold_in_set: bool,
}

/// Every row through [`calibrate`] at `entry`.
pub fn score_rows(rows: &LetterRows, entry: &CalibrationEntry) -> Result<Vec<RowScore>> {
    let mut out = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        let gold = rows.gold(i);
        let cal = calibrate("calibration-fit", rows.row(i), entry).map_err(|e| e.to_string())?;
        out.push(RowScore {
            top: cal.top,
            margin: cal.margin,
            confidence: cal.probs[cal.top],
            p_gold: cal.probs[gold],
            correct: cal.top == gold,
            set_size: cal.set.len(),
            gold_in_set: cal.set.contains(&gold),
        });
    }
    Ok(out)
}

/// The entry that reads rows at `temperature` and nothing else: a set holding only the top row,
/// no margin abstention. What the fit scores through before the other two parameters exist.
pub fn temperature_only(temperature: f64) -> CalibrationEntry {
    CalibrationEntry {
        temperature,
        conformal_quantile: 1.0,
        noul_margin: 0.0,
    }
}

/// `fit_noul_margin`: the smallest observed margin whose retained set (`margin >= thr`) reaches
/// `target` precision; if none does, the largest observed margin.
///
/// The Python sweeps `np.unique(margins)` and recomputes the retained set each time, which is
/// quadratic; this sorts once and walks suffix counts. The counts are integers and the ratio is
/// the same IEEE division, so the returned threshold is the same number.
pub fn fit_noul_margin(margins: &[f64], correct: &[bool], target: f64) -> Result<f64> {
    ensure!(
        !margins.is_empty(),
        "margins must be 1-D and non-empty, got 0"
    );
    ensure!(
        margins.len() == correct.len(),
        "margins/correct length mismatch: {} vs {}",
        margins.len(),
        correct.len()
    );
    ensure!(
        margins.iter().all(|m| m.is_finite()),
        "margins contain a non-finite value"
    );
    ensure!(
        target > 0.0 && target <= 1.0,
        "target_precision must be in (0, 1], got {target}"
    );
    let n = margins.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| margins[*a].total_cmp(&margins[*b]));
    let total_right = correct.iter().filter(|c| **c).count();
    let mut right_below = 0usize;
    let mut i = 0usize;
    while i < n {
        let thr = margins[order[i]];
        let kept = n - i;
        let right = total_right - right_below;
        if (right as f64) / (kept as f64) >= target {
            return Ok(thr);
        }
        while i < n && margins[order[i]] == thr {
            right_below += usize::from(correct[order[i]]);
            i += 1;
        }
    }
    Ok(margins[order[n - 1]])
}

/// `split_conformal_threshold`: `q̂` on the **nonconformity** scale, from each row's `p(gold)`.
///
/// Not a probability cutoff. [`probability_cutoff_from_nonconformity`] is the only way it may
/// reach a [`CalibrationEntry`].
pub fn split_conformal_nonconformity(p_gold: &[f64], alpha: f64) -> Result<f64> {
    ensure!(
        !p_gold.is_empty(),
        "calibration probs must be 2-D non-empty, got 0 rows"
    );
    ensure!(
        p_gold.iter().all(|p| p.is_finite()),
        "a calibrated probability is non-finite"
    );
    ensure!(
        alpha > 0.0 && alpha < 1.0,
        "alpha must be in (0, 1), got {alpha}"
    );
    let mut scores: Vec<f64> = p_gold.iter().map(|p| 1.0 - p).collect();
    scores.sort_by(f64::total_cmp);
    let n = scores.len();
    // Finite-sample correction, in Python's float order: `(n + 1) * (1 - alpha)`, `ceil`, then
    // int / int true division (correctly rounded, as IEEE division of the two exact values is).
    let q = (((n + 1) as f64 * (1.0 - alpha)).ceil() / n as f64).min(1.0);
    // `method="higher"`: `sorted[ceil((n - 1) * q)]`.
    let index = ((n - 1) as f64 * q).ceil() as usize;
    ensure!(index < n, "quantile index {index} is outside {n} scores");
    Ok(scores[index])
}

/// `probability_cutoff_from_nonconformity`: `1 - q̂`, the number `conformal_quantile` means.
pub fn probability_cutoff_from_nonconformity(q_hat: f64) -> Result<f64> {
    ensure!(
        q_hat.is_finite(),
        "nonconformity quantile must be a finite number, got {q_hat}"
    );
    ensure!(
        (0.0..=1.0).contains(&q_hat),
        "nonconformity quantile must be in [0, 1], got {q_hat}"
    );
    Ok(1.0 - q_hat)
}

/// `expected_calibration_error`: equal-width binned ECE over the top-class confidence, and the
/// number of bins that carried mass.
pub fn expected_calibration_error(
    confidence: &[f64],
    correct: &[bool],
    n_bins: usize,
) -> Result<(f64, usize)> {
    ensure!(
        !confidence.is_empty(),
        "probs must be 2-D and non-empty, got 0 rows"
    );
    ensure!(
        confidence.len() == correct.len(),
        "confidence/correct length mismatch: {} vs {}",
        confidence.len(),
        correct.len()
    );
    ensure!(
        confidence.iter().all(|c| c.is_finite()),
        "probs contain a non-finite value"
    );
    ensure!(n_bins >= 1, "n_bins must be >= 1, got {n_bins}");
    // `np.linspace(0, 1, n_bins + 1)`: `i * (1 / n_bins) + 0.0`; the interior edges are 1..n_bins.
    let step = 1.0 / n_bins as f64;
    let interior: Vec<f64> = (1..n_bins).map(|i| i as f64 * step + 0.0).collect();
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    let mut right = vec![0usize; n_bins];
    for (c, ok) in confidence.iter().zip(correct) {
        // `np.digitize(conf, interior, right=True)` is `searchsorted(side="left")`: the count of
        // edges strictly below. Right-closed, so a confidence of exactly 1.0 lands in the last bin.
        let bin = interior.partition_point(|edge| *edge < *c).min(n_bins - 1);
        members[bin].push(*c);
        right[bin] += usize::from(*ok);
    }
    let total = confidence.len() as f64;
    let mut ece = 0.0;
    let mut populated = 0usize;
    for (bin, values) in members.iter().enumerate() {
        let count = values.len();
        if count == 0 {
            continue;
        }
        populated += 1;
        let accuracy = right[bin] as f64 / count as f64;
        let mean_confidence = pairwise_sum(values) / count as f64;
        ece += (count as f64 / total) * (accuracy - mean_confidence).abs();
    }
    Ok((ece, populated))
}

/// What `ece_gate` returns, as data.
#[derive(Debug, Clone, PartialEq)]
pub enum EceState {
    Ran {
        value: f64,
        n: usize,
        populated_bins: usize,
        /// `value <= ECE_THRESHOLD`. Reported, not gated on, by every caller in this crate.
        passed: bool,
    },
    NotRun {
        reason: String,
    },
}

/// `ece_gate` at its defaults: not run below [`ECE_MIN_SAMPLES`], otherwise the ECE against
/// [`ECE_THRESHOLD`] over [`ECE_BINS`] bins.
pub fn ece_state(confidence: &[f64], correct: &[bool]) -> Result<EceState> {
    let n = confidence.len();
    if n < ECE_MIN_SAMPLES {
        return Ok(EceState::NotRun {
            reason: format!(
                "ECE needs at least {ECE_MIN_SAMPLES} examples to be meaningful; got {n}. A \
                 binned calibration error over a small sample measures the binning."
            ),
        });
    }
    let (value, populated_bins) = expected_calibration_error(confidence, correct, ECE_BINS)?;
    Ok(EceState::Ran {
        value,
        n,
        populated_bins,
        passed: value <= ECE_THRESHOLD,
    })
}

/// One fitted entry and the numbers it was made from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntryFit {
    pub n: usize,
    pub temperature: TemperatureFit,
    /// `q̂`, nonconformity scale. Reported; never written into the table.
    pub nonconformity_quantile: f64,
    /// `1 - q̂`: what the table's `conformal_quantile` holds.
    pub conformal_quantile: f64,
    pub noul_margin: f64,
}

impl EntryFit {
    /// The runtime's entry. Ranges are checked where the fit is produced rather than where the
    /// table is loaded (`CalibrationEntryFit.__post_init__` / `CalibrationEntry::validate`).
    pub fn entry(&self) -> CalibrationEntry {
        CalibrationEntry {
            temperature: self.temperature.temperature,
            conformal_quantile: self.conformal_quantile,
            noul_margin: self.noul_margin,
        }
    }
}

/// Fit one entry: the temperature on the rows' NLL, then — on the rows read through
/// [`calibrate`] at that temperature — `noul_margin` on the margins at `target_precision`, and
/// the conformal cutoff as `1 - q̂` at `alpha`.
pub fn fit_entry(rows: &LetterRows, alpha: f64, target_precision: f64) -> Result<EntryFit> {
    let temperature = fit_temperature(rows)?;
    let scored = score_rows(rows, &temperature_only(temperature.temperature))?;
    let margins: Vec<f64> = scored.iter().map(|s| s.margin).collect();
    let correct: Vec<bool> = scored.iter().map(|s| s.correct).collect();
    let p_gold: Vec<f64> = scored.iter().map(|s| s.p_gold).collect();
    let noul_margin = fit_noul_margin(&margins, &correct, target_precision)?;
    let nonconformity_quantile = split_conformal_nonconformity(&p_gold, alpha)?;
    let conformal_quantile = probability_cutoff_from_nonconformity(nonconformity_quantile)?;
    ensure!(
        temperature.temperature.is_finite() && temperature.temperature > 0.0,
        "temperature must be finite and > 0, got {}",
        temperature.temperature
    );
    for (name, v) in [
        ("conformal_quantile", conformal_quantile),
        ("noul_margin", noul_margin),
    ] {
        ensure!(
            v.is_finite() && (0.0..=1.0).contains(&v),
            "{name} must be finite and in [0, 1], got {v}"
        );
    }
    Ok(EntryFit {
        n: rows.len(),
        temperature,
        nonconformity_quantile,
        conformal_quantile,
        noul_margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(width: usize, data: &[(&[f32], usize)]) -> LetterRows {
        let mut out = LetterRows::new(width).unwrap();
        for (logits, gold) in data {
            out.push(logits, *gold).unwrap();
        }
        out
    }

    /// A deterministic generator, so the tests need no crate: SplitMix64.
    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Rows whose gold wins with probability `skill`, logits scaled by `sharpness`.
    fn synthetic(n: usize, width: usize, skill: f64, sharpness: f32, seed: u64) -> LetterRows {
        let mut mix = Mix(seed);
        let mut out = LetterRows::new(width).unwrap();
        for _ in 0..n {
            let gold = (mix.next() % width as u64) as usize;
            let mut logits: Vec<f32> = (0..width).map(|_| mix.unit() as f32).collect();
            let winner = if mix.unit() < skill {
                gold
            } else {
                (gold + 1) % width
            };
            logits[winner] += 1.5;
            for v in &mut logits {
                *v *= sharpness;
            }
            out.push(&logits, gold).unwrap();
        }
        out
    }

    #[test]
    fn pairwise_sum_is_sequential_below_eight_and_blocked_above() {
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
        // 1e16 + 1 + ... loses the ones sequentially; the 8-way blocks keep some.
        let mut a = vec![1e16];
        a.extend(std::iter::repeat_n(1.0, 15));
        let sequential = a.iter().fold(0.0, |s, x| s + x);
        assert_ne!(pairwise_sum(&a), sequential);
        // r0 = 1e16 + 1 (rounds to 1e16); r1..r7 = 2 each; combined then no remainder.
        assert_eq!(pairwise_sum(&a), 1e16 + 14.0);
    }

    #[test]
    fn pairwise_sum_splits_above_the_block_at_a_multiple_of_eight() {
        let a: Vec<f64> = (0..300).map(|i| 1.0 / (1.0 + i as f64)).collect();
        // 300 / 2 = 150, minus 150 % 8 = 144.
        assert_eq!(
            pairwise_sum(&a),
            pairwise_sum(&a[..144]) + pairwise_sum(&a[144..])
        );
    }

    #[test]
    fn fit_temperature_recovers_a_known_distortion() {
        let base = synthetic(4000, 5, 0.7, 1.0, 1);
        let t1 = fit_temperature(&base).unwrap().temperature;
        let mut scaled = LetterRows::new(5).unwrap();
        for i in 0..base.len() {
            let row: Vec<f32> = base.row(i).iter().map(|v| v * 3.0).collect();
            scaled.push(&row, base.gold(i)).unwrap();
        }
        let t3 = fit_temperature(&scaled).unwrap().temperature;
        assert!((t3 / t1 - 3.0).abs() < 0.01, "t1 {t1} t3 {t3}");
    }

    #[test]
    fn an_entry_never_wrong_or_never_right_has_no_interior_temperature() {
        let always = rows(3, &[(&[5.0, 0.0, 0.0], 0), (&[0.0, 4.0, 1.0], 1)]);
        let fit = fit_temperature(&always).unwrap();
        assert!(fit.at_lower_bound && !fit.at_upper_bound, "{fit:?}");
        // The search does not stay at the bound: the NLL underflows to 0 and goes flat.
        assert!(fit.temperature > TEMPERATURE_LO, "{fit:?}");
        let never = rows(3, &[(&[5.0, 0.0, 0.0], 1), (&[0.0, 4.0, 1.0], 2)]);
        let fit = fit_temperature(&never).unwrap();
        assert!(fit.at_upper_bound && !fit.at_lower_bound, "{fit:?}");
        let mixed = synthetic(500, 4, 0.7, 2.0, 3);
        let fit = fit_temperature(&mixed).unwrap();
        assert!(!fit.at_lower_bound && !fit.at_upper_bound, "{fit:?}");
    }

    #[test]
    fn non_finite_logits_and_bad_gold_rows_are_refused() {
        let mut r = LetterRows::new(3).unwrap();
        assert!(r.push(&[0.0, f32::NAN, 1.0], 0).is_err());
        assert!(r.push(&[0.0, f32::INFINITY, 1.0], 0).is_err());
        assert!(r.push(&[0.0, 1.0, 2.0], 3).is_err());
        assert!(r.push(&[0.0, 1.0], 0).is_err());
        assert!(r.is_empty());
        assert!(fit_temperature(&r).is_err());
        assert!(LetterRows::new(1).is_err());
    }

    #[test]
    fn noul_margin_is_the_lowest_margin_reaching_the_target_or_the_largest() {
        let m = [0.1, 0.2, 0.5, 0.9];
        let c = [false, false, true, true];
        assert_eq!(fit_noul_margin(&m, &c, 0.95).unwrap(), 0.5);
        assert_eq!(fit_noul_margin(&m, &c, 0.5).unwrap(), 0.1);
        assert_eq!(
            fit_noul_margin(&[0.3, 0.7], &[false, false], 0.95).unwrap(),
            0.7
        );
        // Ties are one threshold: both 0.2s are kept or dropped together.
        let m = [0.2, 0.2, 0.4];
        let c = [true, false, true];
        assert_eq!(fit_noul_margin(&m, &c, 0.6).unwrap(), 0.2);
        assert_eq!(fit_noul_margin(&m, &c, 0.7).unwrap(), 0.4);
        assert!(fit_noul_margin(&[], &[], 0.95).is_err());
        assert!(fit_noul_margin(&[0.1], &[true], 0.0).is_err());
        assert!(fit_noul_margin(&[f64::NAN], &[true], 0.9).is_err());
    }

    /// `GAP-CALIB-CONFORMAL-SCALE-TWO-MEANINGS`, in both regimes. At `q̂ = 0.5` the two scales
    /// coincide and writing `q̂` as the cutoff has no effect, so one fixture is not enough: a weak
    /// model (`q̂ > 0.5`) and a strong one (`q̂ < 0.5`) are both pinned, each far enough from 0.5
    /// that the two numbers cannot be confused, and in both the served set covers at least
    /// `1 - alpha` of the rows it was fitted on. For the weak model the un-converted value is
    /// also served, and collapses coverage — the failure the gap record measured.
    #[test]
    fn the_table_holds_the_probability_cutoff_not_the_nonconformity_quantile() {
        let alpha = 0.1;
        for (skill, sharpness, weak) in [(0.45, 2.0f32, true), (0.97, 3.0f32, false)] {
            let r = synthetic(2000, 5, skill, sharpness, 7);
            let fit = fit_entry(&r, alpha, DEFAULT_TARGET_PRECISION).unwrap();
            assert_eq!(fit.conformal_quantile, 1.0 - fit.nonconformity_quantile);
            assert_eq!(
                fit.nonconformity_quantile > 0.5,
                weak,
                "q_hat {} for skill {skill}",
                fit.nonconformity_quantile
            );
            assert!(
                (fit.conformal_quantile - fit.nonconformity_quantile).abs() > 0.2,
                "q_hat {} sits too near 0.5 to tell the scales apart",
                fit.nonconformity_quantile
            );
            let served = score_rows(&r, &fit.entry()).unwrap();
            let covered = served.iter().filter(|s| s.gold_in_set).count() as f64 / r.len() as f64;
            assert!(
                covered >= 1.0 - alpha,
                "coverage {covered} at skill {skill}"
            );
            if weak {
                let wrong = CalibrationEntry {
                    conformal_quantile: fit.nonconformity_quantile,
                    ..fit.entry()
                };
                let wrong_served = score_rows(&r, &wrong).unwrap();
                let wrong_cov =
                    wrong_served.iter().filter(|s| s.gold_in_set).count() as f64 / r.len() as f64;
                assert!(
                    wrong_cov < 1.0 - alpha,
                    "weak model, wrong cutoff covers {wrong_cov}"
                );
            }
        }
    }

    #[test]
    fn the_conformal_index_is_numpys_higher_method() {
        // n = 9, alpha = 0.1: q = min(1, ceil(10 * 0.9) / 9) = 1, index ceil(8 * 1) = 8.
        let p: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();
        assert_eq!(split_conformal_nonconformity(&p, 0.1).unwrap(), 1.0 - 0.1);
        // n = 20, alpha = 0.25: q = ceil(21 * 0.75) / 20 = 16 / 20 = 0.8, index ceil(19 * 0.8) = 16.
        let p: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let mut scores: Vec<f64> = p.iter().map(|x| 1.0 - x).collect();
        scores.sort_by(f64::total_cmp);
        assert_eq!(split_conformal_nonconformity(&p, 0.25).unwrap(), scores[16]);
        assert!(split_conformal_nonconformity(&p, 0.0).is_err());
        assert!(split_conformal_nonconformity(&[], 0.1).is_err());
        assert!(probability_cutoff_from_nonconformity(1.5).is_err());
        assert!(probability_cutoff_from_nonconformity(f64::NAN).is_err());
        assert_eq!(probability_cutoff_from_nonconformity(0.25).unwrap(), 0.75);
    }

    #[test]
    fn ece_is_not_run_below_the_floor_and_puts_confidence_one_in_the_last_bin() {
        assert!(matches!(
            ece_state(&[0.9; 99], &[true; 99]).unwrap(),
            EceState::NotRun { .. }
        ));
        let (ece, populated) = expected_calibration_error(&[1.0, 1.0], &[true, true], 15).unwrap();
        assert_eq!((ece, populated), (0.0, 1));
        let (ece, populated) =
            expected_calibration_error(&[0.95, 0.55], &[false, true], 15).unwrap();
        assert_eq!(populated, 2);
        assert!((ece - (0.5 * 0.95 + 0.5 * 0.45)).abs() < 1e-15, "{ece}");
        match ece_state(&[0.9; 100], &[true; 100]).unwrap() {
            EceState::Ran {
                value, n, passed, ..
            } => {
                assert_eq!(n, 100);
                assert!((value - 0.1).abs() < 1e-12 && !passed, "{value}");
            }
            other => panic!("{other:?}"),
        }
    }
}
