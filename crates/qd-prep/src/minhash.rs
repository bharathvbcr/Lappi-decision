//! MinHash signatures, bit-identical to `qd_data.minhash.MinHasher.signature`.
//!
//! The Python class is the reference oracle and stays where it is: its source is part of
//! the `code_fingerprint` every shard header pins, so it is not edited. What this module
//! reproduces, line for line against `python/qd_data/minhash.py`:
//!
//! * the key: `blake2b(f"qd_data.minhash.v1|{seed}".encode(), digest_size=32)`, unkeyed;
//! * permutation `i`: `blob = blake2b(i.to_bytes(8, "big"), key=key, digest_size=32)`,
//!   `a = int(blob[:16], big) % (p - 1) + 1`, `b = int(blob[16:], big) % p`, `p = 2**61 - 1`;
//! * a shingle's base hash: `int(blake2b(item, key=key, digest_size=8), big)` -- a full
//!   64-bit value, **not** reduced into the field first;
//! * the signature: `min((a*h + b) % p for h in bases)` per permutation.
//!
//! `a < 2**61`, `h < 2**64` and `b < 2**61`, so `a*h + b < 2**126` fits a `u128` exactly and
//! the reduction is exact integer arithmetic -- the same integers Python's big ints produce.
//!
//! The seed enters only through its decimal spelling, which the caller passes as text
//! (`str(seed)` on the Python side). That keeps any Python `int` representable and makes
//! the key material the same bytes by construction rather than by a formatting agreement.

use crate::blake2b::{blake2b, blake2b_into};

/// 2**61 - 1, `qd_data.minhash.MERSENNE_PRIME`.
pub const MERSENNE_PRIME: u64 = (1u64 << 61) - 1;

/// Bound on the permutation count. The configured value is 128; anything near this bound is
/// a corrupted request, not a configuration.
pub const MAX_NUM_PERM: usize = 4096;

/// `x mod (2**61 - 1)` for `x < 2**126`, without a 128-bit division.
///
/// `x = hi * 2**61 + lo` and `2**61 = 1 (mod p)`, so `x = hi + lo (mod p)`. Two folds bring
/// any value below `2**126` under `p + 33`, and one conditional subtraction finishes it.
#[inline(always)]
pub fn mod_mersenne61(x: u128) -> u64 {
    let p = MERSENNE_PRIME as u128;
    let s = (x & p) + (x >> 61);
    let s = (s & p) + (s >> 61);
    let s = s as u64;
    if s >= MERSENNE_PRIME {
        s - MERSENNE_PRIME
    } else {
        s
    }
}

/// The permutation family for one `(num_perm, seed)`.
#[derive(Clone, Debug)]
pub struct MinHasher {
    key: [u8; 32],
    a: Vec<u64>,
    b: Vec<u64>,
}

impl MinHasher {
    /// `seed_decimal` is the seed exactly as Python spells it (`str(seed)`).
    pub fn new(num_perm: usize, seed_decimal: &str) -> anyhow::Result<Self> {
        if num_perm == 0 || num_perm > MAX_NUM_PERM {
            anyhow::bail!("num_perm must be in 1..={MAX_NUM_PERM}, got {num_perm}");
        }
        let digits = seed_decimal.strip_prefix('-').unwrap_or(seed_decimal);
        if digits.is_empty()
            || !digits.bytes().all(|c| c.is_ascii_digit())
            || (digits.len() > 1 && digits.starts_with('0'))
            || seed_decimal == "-0"
        {
            anyhow::bail!(
                "seed {seed_decimal:?} is not the decimal spelling of an int; the key material \
                 would not be the bytes the Python reference hashes"
            );
        }
        let material = format!("qd_data.minhash.v1|{seed_decimal}");
        let key: [u8; 32] = blake2b(b"", material.as_bytes());
        let p = MERSENNE_PRIME as u128;
        let mut a = Vec::with_capacity(num_perm);
        let mut b = Vec::with_capacity(num_perm);
        for i in 0..num_perm as u64 {
            let blob: [u8; 32] = blake2b(&key, &i.to_be_bytes());
            let mut hi = [0u8; 16];
            let mut lo = [0u8; 16];
            hi.copy_from_slice(&blob[..16]);
            lo.copy_from_slice(&blob[16..]);
            // Both results are below 2**61, so the narrowing is exact.
            a.push((u128::from_be_bytes(hi) % (p - 1) + 1) as u64);
            b.push((u128::from_be_bytes(lo) % p) as u64);
        }
        Ok(Self { key, a, b })
    }

    pub fn num_perm(&self) -> usize {
        self.a.len()
    }

    /// `MinHasher.base_hash`: the shingle's 8-byte keyed digest, read big-endian.
    #[inline]
    pub fn base_hash(&self, item: &[u8]) -> u64 {
        let mut out = [0u8; 8];
        blake2b_into(&mut out, &self.key, item);
        u64::from_be_bytes(out)
    }

    /// The signature of one shingle set, written into `out` (`num_perm` values).
    ///
    /// Refuses an empty set, as the reference does: it has no minimum, and inventing one would
    /// make every empty document identical to every other.
    pub fn signature_into<'a, I>(&self, items: I, out: &mut [u64]) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        if out.len() != self.a.len() {
            anyhow::bail!(
                "signature buffer holds {} values for {} permutations",
                out.len(),
                self.a.len()
            );
        }
        out.fill(u64::MAX);
        let mut any = false;
        for item in items {
            any = true;
            let h = self.base_hash(item) as u128;
            for ((slot, &a), &b) in out.iter_mut().zip(&self.a).zip(&self.b) {
                let v = mod_mersenne61(a as u128 * h + b as u128);
                if v < *slot {
                    *slot = v;
                }
            }
        }
        if !any {
            anyhow::bail!(
                "cannot sign an empty shingle set: every empty document would hash identically \
                 and be deduped against every other empty document"
            );
        }
        Ok(())
    }

    pub fn signature<'a, I>(&self, items: I) -> anyhow::Result<Vec<u64>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut out = vec![0u64; self.a.len()];
        self.signature_into(items, &mut out)?;
        Ok(out)
    }
}

/// One shingle set: borrowed items, in whatever order the caller holds them. The minimum is
/// order-independent, so the order a Python `frozenset` iterates in does not matter.
pub type ShingleSet<'a> = Vec<&'a [u8]>;

/// Bound on worker threads, whatever the caller asks for.
pub const MAX_THREADS: usize = 256;

/// Every set's signature, row-major (`sets.len() * num_perm` values), on up to `threads`
/// threads.
///
/// Deterministic in the thread count: the sets are cut into contiguous index ranges, each
/// thread fills its own disjoint slice of the output, and no value depends on any other set.
/// One failing set fails the whole call, naming the first failing index.
pub fn sign_all(
    hasher: &MinHasher,
    sets: &[ShingleSet<'_>],
    threads: usize,
) -> anyhow::Result<Vec<u64>> {
    if threads == 0 || threads > MAX_THREADS {
        anyhow::bail!("threads must be in 1..={MAX_THREADS}, got {threads}");
    }
    let k = hasher.num_perm();
    let mut out = vec![0u64; sets.len() * k];
    if sets.is_empty() {
        return Ok(out);
    }
    let per = sets.len().div_ceil(threads.min(sets.len()));
    let failures: Vec<anyhow::Result<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(per * k)
            .zip(sets.chunks(per))
            .enumerate()
            .map(|(chunk, (dst, src))| {
                scope.spawn(move || -> anyhow::Result<()> {
                    for (j, (set, sig)) in src.iter().zip(dst.chunks_mut(k)).enumerate() {
                        hasher
                            .signature_into(set.iter().copied(), sig)
                            .map_err(|e| anyhow::anyhow!("set {}: {e}", chunk * per + j))?;
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!("a signing thread panicked")),
            })
            .collect()
    });
    for r in failures {
        r?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_reduction_is_exact() {
        // Edges, then a deterministic spread across the whole input range (no RNG crate).
        let p = MERSENNE_PRIME as u128;
        let mut cases: Vec<u128> = vec![
            0,
            1,
            p - 1,
            p,
            p + 1,
            2 * p,
            2 * p - 1,
            (1u128 << 61),
            (1u128 << 64) - 1,
            (1u128 << 122),
            (1u128 << 125) + (1u128 << 61),
            (1u128 << 126) - 1,
            (p - 1) * ((1u128 << 64) - 1) + (p - 1),
        ];
        let mut x: u128 = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834;
        for _ in 0..200_000 {
            x = x
                .wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645)
                .wrapping_add(1);
            cases.push(x >> 2); // < 2**126
        }
        for c in cases {
            assert_eq!(mod_mersenne61(c) as u128, c % p, "x = {c}");
        }
    }

    #[test]
    fn empty_set_is_refused() {
        let h = MinHasher::new(8, "1").unwrap();
        assert!(h.signature(std::iter::empty()).is_err());
    }

    #[test]
    fn seed_spelling_is_checked() {
        for bad in ["", "-", "-0", "01", "1.0", "+1", " 1", "1e3"] {
            assert!(MinHasher::new(8, bad).is_err(), "{bad:?} was accepted");
        }
        for good in ["0", "7", "-3", "20260919", "123456789012345678901234567890"] {
            assert!(MinHasher::new(8, good).is_ok(), "{good:?} was refused");
        }
    }

    #[test]
    fn thread_count_does_not_change_the_output() {
        let h = MinHasher::new(128, "20260919").unwrap();
        let owned: Vec<Vec<Vec<u8>>> = (0..997)
            .map(|i| {
                (0..(i % 37) + 1)
                    .map(|j| format!("tok{i} {j}").into_bytes())
                    .collect()
            })
            .collect();
        let sets: Vec<ShingleSet<'_>> = owned
            .iter()
            .map(|s| s.iter().map(Vec::as_slice).collect())
            .collect();
        let one = sign_all(&h, &sets, 1).unwrap();
        for threads in [2, 3, 7, 16, 64, 256] {
            assert_eq!(
                sign_all(&h, &sets, threads).unwrap(),
                one,
                "threads={threads}"
            );
        }
        assert!(sign_all(&h, &sets, 0).is_err());
        assert!(sign_all(&h, &sets, MAX_THREADS + 1).is_err());
    }

    #[test]
    fn a_failing_set_is_named_whatever_the_thread_count() {
        let h = MinHasher::new(4, "1").unwrap();
        let x: &[u8] = b"x";
        let sets: Vec<ShingleSet<'_>> = vec![vec![x], vec![x], vec![], vec![x]];
        for threads in [1, 2, 4] {
            let err = sign_all(&h, &sets, threads).unwrap_err().to_string();
            assert!(err.contains("set 2"), "threads={threads}: {err}");
        }
    }
}
