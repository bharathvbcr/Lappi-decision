//! `qd_data.minhash.MinHasher.signature`, evaluated in Rust.
//!
//! The Python is the reference and stays the reference: it owns the permutation family (`a`,
//! `b` and the blake2b key are derived there and handed in), the shingling and every decision
//! made with a signature. What moves here is only the inner loop it spent 75% of the
//! `--score-checkpoint` prelude in (cProfile, 2026-09-30): for every shingle `s` and every
//! permutation `i`, `(a[i] * h(s) + b[i]) mod p`, minimised over `s`, where
//! `h(s) = int.from_bytes(blake2b(s, key=key, digest_size=8).digest(), "big")` and
//! `p = 2**61 - 1`. Python does it in arbitrary-precision integers; here the product is a
//! `u128` (`a < 2**61`, `h < 2**64`, so `a*h + b < 2**126`) reduced exactly.

use crate::blake2b::Keyed;

/// 2**61 - 1, `qd_data.minhash.MERSENNE_PRIME`.
pub const MERSENNE_PRIME: u64 = (1 << 61) - 1;
/// `MinHasher.base_hash` reads an eight-byte digest.
const BASE_HASH_BYTES: usize = 8;

/// `x mod (2**61 - 1)` for any `u128`, exactly.
///
/// `2**61 = 1 (mod p)`, so folding the high bits onto the low ones preserves the residue.
/// After two folds the value is below `2**61 + 128 < 2p`, and one conditional subtraction
/// finishes it.
#[inline(always)]
pub fn mod_mersenne(x: u128) -> u64 {
    let p = u128::from(MERSENNE_PRIME);
    let r = (x & p) + (x >> 61);
    let r = (r & p) + (r >> 61);
    // r < 2**61 + 128, so it fits a u64 and needs at most one subtraction.
    let r = r as u64;
    if r >= MERSENNE_PRIME {
        r - MERSENNE_PRIME
    } else {
        r
    }
}

/// A MinHash permutation family: the `MinHasher` it came from, as numbers.
#[derive(Clone, Debug)]
pub struct Family {
    a: Vec<u64>,
    b: Vec<u64>,
    hasher: Keyed,
}

impl Family {
    /// Refuses a family `MinHasher` could not have produced: lengths that differ, an empty
    /// family, `a` outside `[1, p)` or `b` outside `[0, p)`, or a key blake2b refuses.
    pub fn new(key: &[u8], a: Vec<u64>, b: Vec<u64>) -> Result<Self, String> {
        if a.is_empty() || a.len() != b.len() {
            return Err(format!(
                "a permutation family needs as many a as b values, at least one: {} and {}",
                a.len(),
                b.len()
            ));
        }
        if let Some((i, v)) = a
            .iter()
            .enumerate()
            .find(|(_, v)| **v == 0 || **v >= MERSENNE_PRIME)
        {
            return Err(format!("a[{i}] = {v} is outside [1, 2**61 - 1)"));
        }
        if let Some((i, v)) = b.iter().enumerate().find(|(_, v)| **v >= MERSENNE_PRIME) {
            return Err(format!("b[{i}] = {v} is outside [0, 2**61 - 1)"));
        }
        let hasher = Keyed::new(key, BASE_HASH_BYTES)?;
        Ok(Self { a, b, hasher })
    }

    /// How many permutations, which is the signature's length.
    pub fn num_perm(&self) -> usize {
        self.a.len()
    }

    /// `MinHasher.signature` of one shingle set, into `out` (`num_perm` values).
    ///
    /// An empty set is refused, as the reference refuses it: it has no minimum, and inventing
    /// one would make every empty document a duplicate of every other.
    pub fn signature<'s>(
        &self,
        shingles: impl IntoIterator<Item = &'s [u8]>,
        out: &mut [u64],
    ) -> Result<(), String> {
        if out.len() != self.num_perm() {
            return Err(format!(
                "signature buffer holds {} values for {} permutations",
                out.len(),
                self.num_perm()
            ));
        }
        // Every residue is below p < u64::MAX, so this is beaten by the first shingle.
        out.fill(u64::MAX);
        let mut seen = false;
        for shingle in shingles {
            seen = true;
            let h = u128::from(self.hasher.u64_be(shingle)?);
            for ((slot, a), b) in out.iter_mut().zip(&self.a).zip(&self.b) {
                let v = mod_mersenne(u128::from(*a) * h + u128::from(*b));
                if v < *slot {
                    *slot = v;
                }
            }
        }
        if !seen {
            return Err("cannot sign an empty shingle set".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic stream (SplitMix64), so the reduction is tested on many values
    /// without a randomness crate.
    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[test]
    fn mersenne_reduction_is_exact_on_edges_and_a_million_values() {
        let p = u128::from(MERSENNE_PRIME);
        let edges = [
            0,
            1,
            p - 1,
            p,
            p + 1,
            2 * p - 1,
            2 * p,
            2 * p + 1,
            p * p,
            u128::MAX,
        ];
        for x in edges {
            assert_eq!(u128::from(mod_mersenne(x)), x % p, "x = {x}");
        }
        let mut s = 7u64;
        for _ in 0..1_000_000 {
            let x = (u128::from(splitmix(&mut s)) << 64) | u128::from(splitmix(&mut s));
            assert_eq!(u128::from(mod_mersenne(x)), x % p);
            // The shape the signature actually reduces: a < p, h < 2**64, b < p.
            let a = u128::from(splitmix(&mut s)) % (p - 1) + 1;
            let b = u128::from(splitmix(&mut s)) % p;
            let y = a * u128::from(splitmix(&mut s)) + b;
            assert_eq!(u128::from(mod_mersenne(y)), y % p);
        }
    }

    #[test]
    fn a_family_minhasher_could_not_produce_is_refused() {
        let key = [1u8; 32];
        assert!(Family::new(&key, vec![], vec![]).is_err());
        assert!(Family::new(&key, vec![1, 2], vec![0]).is_err());
        assert!(
            Family::new(&key, vec![0], vec![0]).is_err(),
            "a = 0 collapses the permutation"
        );
        assert!(Family::new(&key, vec![MERSENNE_PRIME], vec![0]).is_err());
        assert!(Family::new(&key, vec![1], vec![MERSENNE_PRIME]).is_err());
        assert!(Family::new(&[0u8; 65], vec![1], vec![0]).is_err());
        assert!(Family::new(&key, vec![MERSENNE_PRIME - 1], vec![MERSENNE_PRIME - 1]).is_ok());
    }

    #[test]
    fn the_signature_is_the_per_permutation_minimum_and_ignores_order() {
        let key = [3u8; 32];
        let family = Family::new(&key, vec![1, 5, MERSENNE_PRIME - 1], vec![0, 7, 11])
            .expect("valid family");
        let hasher = Keyed::new(&key, 8).expect("valid key");
        let shingles: [&[u8]; 3] = [b"a b c d e", b"b c d e f", "ü ñ".as_bytes()];
        let mut got = vec![0u64; 3];
        family.signature(shingles, &mut got).expect("signs");
        let mut reversed = vec![0u64; 3];
        family
            .signature(shingles.iter().rev().copied(), &mut reversed)
            .expect("signs");
        assert_eq!(
            got, reversed,
            "a set has no order, so neither may its signature"
        );
        let p = u128::from(MERSENNE_PRIME);
        for (i, (a, b)) in [(1u128, 0u128), (5, 7), (p - 1, 11)]
            .into_iter()
            .enumerate()
        {
            let want = shingles
                .iter()
                .map(|s| (a * u128::from(hasher.u64_be(s).expect("8 bytes")) + b) % p)
                .min()
                .expect("non-empty");
            assert_eq!(u128::from(got[i]), want);
        }
    }

    #[test]
    fn an_empty_set_and_a_wrong_buffer_are_refused() {
        let family = Family::new(&[9u8; 32], vec![1, 2], vec![3, 4]).expect("valid family");
        let mut out = vec![0u64; 2];
        assert!(family.signature(std::iter::empty(), &mut out).is_err());
        let mut short = vec![0u64; 1];
        assert!(family.signature([&b"x"[..]], &mut short).is_err());
    }
}
