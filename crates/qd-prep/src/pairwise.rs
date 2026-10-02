//! numpy's float64 summation order, reproduced so a sum here is the sum the oracle computes.
//!
//! `np.sum` over a contiguous float64 array, and `a.sum(axis=1)` over each row of a C-ordered
//! one, both go through numpy's `pairwise_sum` (`numpy/_core/src/umath/loops_utils.h.src`):
//!
//! - fewer than 8 values: `res = 0.0`, then `res += a[i]` in order;
//! - up to `PW_BLOCKSIZE` (128): eight running accumulators seeded with `a[0..8]`, advanced
//!   eight at a time, combined as `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`, then the remainder
//!   added in order;
//! - more: split at `n2 = n/2 - (n/2 % 8)` and add the two halves' sums.
//!
//! Measured, not recalled: on this Mac's numpy 2.5.0 the probe in the handoff
//! (`HANDOFF/perf-linear-control-rust-2026-10-01.md`) compared this rule with `np.sum` on 303
//! lengths up to 655,360 and with `a.sum(axis=1)` for every row width 1..19, all bit for bit;
//! `a.sum(axis=0)` is NOT this rule (it accumulates down each column in row order), and the
//! training loop sums those columns sequentially for that reason.

/// numpy's block size for the unrolled base case.
const PW_BLOCKSIZE: usize = 128;

/// `np.sum` of `n` values, where value `i` is `at(i)`, in numpy's pairwise order.
///
/// A closure rather than a slice so a sum of products (`sum(W * W)`) needs no temporary; the
/// product is formed exactly as numpy forms the temporary's element, so the value is the same.
pub fn pairwise_sum<F: Fn(usize) -> f64>(n: usize, at: &F) -> f64 {
    sum_range(0, n, at)
}

fn sum_range<F: Fn(usize) -> f64>(start: usize, n: usize, at: &F) -> f64 {
    if n < 8 {
        let mut res = 0.0;
        for i in 0..n {
            res += at(start + i);
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        for (j, slot) in r.iter_mut().enumerate() {
            *slot = at(start + j);
        }
        let whole = n - (n % 8);
        let mut i = 8;
        while i < whole {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += at(start + i + j);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += at(start + i);
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        sum_range(start, n2, at) + sum_range(start + n2, n - n2, at)
    }
}

/// `pairwise_sum` over a slice.
pub fn pairwise_sum_slice(a: &[f64]) -> f64 {
    pairwise_sum(a.len(), &|i| a[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values whose sum depends on the order they are added in: a large value next to many
    /// small ones loses them when added sequentially and keeps some of them pairwise.
    fn order_sensitive(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match i % 5 {
                0 => 1.0e16,
                1 => 1.0,
                2 => -1.0e16,
                3 => 0.3,
                _ => 3.0,
            })
            .collect()
    }

    fn sequential_from_zero(a: &[f64]) -> f64 {
        let mut res = 0.0;
        for v in a {
            res += v;
        }
        res
    }

    #[test]
    fn short_arrays_are_summed_in_order_from_zero() {
        for n in 0..8 {
            let a = order_sensitive(n);
            assert_eq!(
                pairwise_sum_slice(&a).to_bits(),
                sequential_from_zero(&a).to_bits()
            );
        }
    }

    #[test]
    fn the_eight_accumulator_block_is_not_the_sequential_sum() {
        // The values are chosen so the two orders disagree; if they agreed this test would
        // not distinguish the rule numpy uses from the naive one.
        let a = order_sensitive(40);
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        for chunk in a[8..40].chunks(8) {
            for (slot, v) in r.iter_mut().zip(chunk) {
                *slot += v;
            }
        }
        let want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        assert_eq!(pairwise_sum_slice(&a).to_bits(), want.to_bits());
        assert_ne!(want.to_bits(), sequential_from_zero(&a).to_bits());
    }

    #[test]
    fn long_arrays_split_at_a_multiple_of_eight() {
        let a = order_sensitive(300);
        // 300 / 2 = 150, 150 - 150 % 8 = 144.
        let want = pairwise_sum_slice(&a[..144]) + pairwise_sum_slice(&a[144..]);
        assert_eq!(pairwise_sum_slice(&a).to_bits(), want.to_bits());
    }

    #[test]
    fn the_closure_form_sums_products_without_a_temporary() {
        let a: Vec<f64> = (0..1000).map(|i| (i as f64).sin() * 1.0e3).collect();
        let squares: Vec<f64> = a.iter().map(|v| v * v).collect();
        assert_eq!(
            pairwise_sum(a.len(), &|i| a[i] * a[i]).to_bits(),
            pairwise_sum_slice(&squares).to_bits()
        );
    }
}
