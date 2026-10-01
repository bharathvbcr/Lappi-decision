//! NumPy's `SeedSequence`, `PCG64` and `Generator.permutation`, bit for bit.
//!
//! `ShardReader._plan` orders an epoch with
//! `np.random.default_rng(np.random.SeedSequence([seed, epoch, batch_tokens, b])).permutation(..)`,
//! so a batch order byte-identical to Python's needs these three exactly, not a statistically
//! equivalent RNG. Ported from numpy's sources as they stand in 2.x (the oracle fixture
//! records the numpy version it was generated with, and `tests/parity.rs` checks the
//! intermediate states -- the pool, `generate_state`, raw draws and a permutation -- before it
//! checks any batch order, so a mismatch is located to the layer that broke):
//!
//! * `SeedSequence` (`bit_generator.pyx`): entropy integers become little-endian `uint32`
//!   words, hash-mixed into a 4-word pool; `generate_state` hashes the pool out again.
//! * `PCG64` (`pcg64.h`): a 128-bit LCG seeded from `generate_state(4, uint64)` as
//!   `(initstate = w0:w1, initseq = w2:w3)`, output XSL-RR; `next_uint32` hands out the low
//!   half of a 64-bit draw and buffers the high half.
//! * `Generator.shuffle` (`_generator.pyx` `_shuffle_raw`): Fisher-Yates from the top, each
//!   index drawn by `random_interval` -- masked rejection sampling, 32-bit draws for bounds
//!   that fit in 32 bits.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// `PCG_DEFAULT_MULTIPLIER_128`.
const PCG_MULT: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `np.random.SeedSequence(entropy)` for a sequence of non-negative integers, no spawn key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    /// `SeedSequence([e0, e1, ...])`. Each integer contributes its `uint32` words, least
    /// significant first, and zero contributes one zero word (`_int_to_uint32_array`).
    pub fn new(entropy: &[u64]) -> Self {
        let mut words: Vec<u32> = Vec::with_capacity(entropy.len() * 2);
        for &e in entropy {
            let (lo, hi) = (e as u32, (e >> 32) as u32);
            words.push(lo);
            if hi != 0 {
                words.push(hi);
            }
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for &w in words.iter().skip(POOL_SIZE) {
            for slot in pool.iter_mut() {
                let h = hashmix(w, &mut hash_const);
                *slot = mix(*slot, h);
            }
        }
        Self { pool }
    }

    /// The mixed entropy pool (`SeedSequence.pool`).
    pub fn pool(&self) -> [u32; POOL_SIZE] {
        self.pool
    }

    /// `generate_state(n_words, np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut v = self.pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect()
    }

    /// `generate_state(n, np.uint64)`: `2n` words viewed little-endian as `uint64`.
    pub fn generate_state_u64(&self, n: usize) -> Vec<u64> {
        let words = self.generate_state_u32(n * 2);
        words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
            .collect()
    }
}

/// `np.random.PCG64` with numpy's 32-bit buffering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    /// `PCG64(seed_sequence)`.
    pub fn new(seed: &SeedSequence) -> Self {
        let v = seed.generate_state_u64(4);
        let initstate = (u128::from(v[0]) << 64) | u128::from(v[1]);
        let initseq = (u128::from(v[2]) << 64) | u128::from(v[3]);
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// One 64-bit draw (`random_raw`).
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        let value = ((s >> 64) as u64) ^ (s as u64);
        value.rotate_right((s >> 122) as u32)
    }

    /// `pcg64_next32`: the low half of a fresh draw, the high half buffered for next time.
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }

    /// `random_interval(max)`: uniform in `[0, max]` by masked rejection.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= 0xffff_ffff {
            loop {
                let v = u64::from(self.next_u32()) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `Generator.shuffle` on a 1-D array.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            xs.swap(i, j);
        }
    }

    /// `Generator.permutation(array)`: a shuffled copy.
    pub fn permutation<T: Clone>(&mut self, xs: &[T]) -> Vec<T> {
        let mut out = xs.to_vec();
        self.shuffle(&mut out);
        out
    }
}

/// `np.random.default_rng(np.random.SeedSequence(entropy))`.
pub fn default_rng(entropy: &[u64]) -> Pcg64 {
    Pcg64::new(&SeedSequence::new(entropy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_permutation_is_a_permutation() {
        let mut rng = default_rng(&[1, 2, 3, 4]);
        let mut p = rng.permutation(&(0..100u32).collect::<Vec<_>>());
        p.sort_unstable();
        assert_eq!(p, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn next_u32_hands_out_both_halves_of_one_draw() {
        let mut a = default_rng(&[9]);
        let mut b = default_rng(&[9]);
        let whole = a.next_u64();
        assert_eq!(b.next_u32(), whole as u32);
        assert_eq!(b.next_u32(), (whole >> 32) as u32);
    }
}
