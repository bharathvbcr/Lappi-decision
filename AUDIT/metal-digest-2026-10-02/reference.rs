//! Reference for the proposed device-side prefix-state digest (`README.md` beside this file).
//!
//! Std-only, so it compiles and tests on its own:
//!
//! ```text
//! rustc --edition 2021 --test -O AUDIT/metal-digest-2026-10-02/reference.rs -o <scratch>/digest-ref
//! <scratch>/digest-ref
//! ```
//!
//! The kernel in `state_digest.metal` must produce [`buffer_lanes`] bit for bit; this file is
//! the oracle a GPU test compares it against, and the home of the property the ruling names: a
//! one-bit flip anywhere in any buffer changes the digest.
//!
//! # The construction
//!
//! A buffer is `n` little-endian `u32` words (every state buffer is f32). Each word `w` at index
//! `i` contributes, to each of 8 lanes `l`, the term
//!
//! ```text
//! t_l(i, w) = fmix32(w ^ key_l(i)),      key_l(i) = fmix32(i ^ SEED[l]) ^ SEED[(l + 1) % 8]
//! ```
//!
//! and lane `l` is the XOR of its terms. `fmix32` (murmur3's finalizer) is a bijection on `u32`,
//! and `w -> w ^ key_l(i)` is one too, so for a fixed index `t_l(i, ·)` is injective: changing the
//! word at one index changes exactly one term in every lane, and so changes every lane. A one-bit
//! flip changes exactly one word, so it always changes the digest. XOR is associative and
//! commutative, so the result does not depend on how a GPU splits or orders the reduction: the
//! digest is deterministic by construction, not by a fixed reduction tree. The index enters every
//! term, so moving a value to another position changes the digest too.
//!
//! What it does **not** give: collision resistance against an adversary. Several words changed
//! at once can cancel in a lane only if their term differences XOR to zero in all 8 lanes at once
//! (256 bits); for the accidental write a read-only decode might make that is negligible, for a
//! chosen one it is not. The runtime's check (`qd_runtime::answer::readonly_decode`) is the first
//! kind. 32-bit lanes because Apple GPUs multiply 32-bit integers natively and emulate 64-bit.
//!
//! The 32-byte record per buffer then goes, with its tag and byte length and the token count,
//! through SHA-256 on the host (a few hundred bytes), exactly where `PrefixState::digest` hashes
//! per-buffer SHA-256s today.

pub const LANES: usize = 8;

/// The lanes' seeds: the first 32 bits of the fractional parts of the square roots of the first
/// eight primes (SHA-256's initial hash values), so nobody chose them.
pub const SEED: [u32; LANES] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
];

/// murmur3's 32-bit finalizer: a bijection on `u32` (each step is invertible).
#[inline]
pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

#[inline]
pub fn key(lane: usize, index: u32) -> u32 {
    fmix32(index ^ SEED[lane]) ^ SEED[(lane + 1) % LANES]
}

/// The 8 lanes of one buffer's digest. Refuses a byte length that is not whole words, and more
/// than `u32::MAX` words (the index must be unique per word).
pub fn buffer_lanes(bytes: &[u8]) -> Result<[u32; LANES], String> {
    if bytes.len() % 4 != 0 {
        return Err(format!("{} bytes is not whole 32-bit words", bytes.len()));
    }
    if bytes.len() / 4 > u32::MAX as usize {
        return Err("more than 2^32 words: the word index would repeat".into());
    }
    let mut lanes = [0u32; LANES];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        for (l, lane) in lanes.iter_mut().enumerate() {
            *lane ^= fmix32(w ^ key(l, i as u32));
        }
    }
    Ok(lanes)
}

/// The same digest folded in an arbitrary split and order, as a GPU would: chunks of `chunk`
/// words reduced separately, then combined in reverse.
pub fn buffer_lanes_split(bytes: &[u8], chunk: usize) -> [u32; LANES] {
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut partials = Vec::new();
    for (ci, ch) in words.chunks(chunk.max(1)).enumerate() {
        let mut p = [0u32; LANES];
        for (j, &w) in ch.iter().enumerate() {
            let i = (ci * chunk.max(1) + j) as u32;
            for (l, lane) in p.iter_mut().enumerate() {
                *lane ^= fmix32(w ^ key(l, i));
            }
        }
        partials.push(p);
    }
    let mut out = [0u32; LANES];
    for p in partials.iter().rev() {
        for l in 0..LANES {
            out[l] ^= p[l];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random buffer (xorshift), so the test needs no crate.
    fn buffer(words: usize, seed: u32) -> Vec<u8> {
        let mut s = seed.max(1);
        let mut out = Vec::with_capacity(words * 4);
        for _ in 0..words {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn fmix32_is_a_bijection_on_a_sampled_domain_and_by_construction() {
        // Exhaustive over 2^20 inputs: no two collide.
        let mut seen = std::collections::HashSet::with_capacity(1 << 20);
        for x in 0u32..(1 << 20) {
            assert!(seen.insert(fmix32(x.wrapping_mul(0x9e37_79b9))), "collision at {x}");
        }
    }

    #[test]
    fn every_one_bit_flip_in_a_buffer_changes_every_lane() {
        let b = buffer(257, 7);
        let base = buffer_lanes(&b).unwrap();
        for byte in 0..b.len() {
            for bit in 0..8 {
                let mut f = b.clone();
                f[byte] ^= 1 << bit;
                let d = buffer_lanes(&f).unwrap();
                for l in 0..LANES {
                    assert_ne!(d[l], base[l], "flip at byte {byte} bit {bit} left lane {l} unchanged");
                }
            }
        }
    }

    #[test]
    fn a_flip_in_the_last_word_of_a_large_buffer_and_special_floats_are_seen() {
        let mut b = buffer(1 << 20, 3);
        let base = buffer_lanes(&b).unwrap();
        let last = b.len() - 1;
        b[last] ^= 0x80;
        assert_ne!(buffer_lanes(&b).unwrap(), base);
        // +0.0 vs -0.0, and two NaN payloads: bit patterns a float sum would not tell apart.
        let z = |x: f32| x.to_bits().to_le_bytes().to_vec();
        assert_ne!(buffer_lanes(&z(0.0)).unwrap(), buffer_lanes(&z(-0.0)).unwrap());
        assert_ne!(
            buffer_lanes(&f32::from_bits(0x7fc0_0001).to_bits().to_le_bytes()).unwrap(),
            buffer_lanes(&f32::from_bits(0x7fc0_0002).to_bits().to_le_bytes()).unwrap()
        );
    }

    #[test]
    fn moving_a_value_changes_the_digest_and_the_split_does_not() {
        let b = buffer(1000, 11);
        let base = buffer_lanes(&b).unwrap();
        let mut swapped = b.clone();
        swapped.swap(0, 4);
        swapped.swap(1, 5);
        swapped.swap(2, 6);
        swapped.swap(3, 7);
        assert_ne!(buffer_lanes(&swapped).unwrap(), base, "swapping words 0 and 1 must be seen");
        for chunk in [1, 3, 32, 256, 999, 1000, 4096] {
            assert_eq!(buffer_lanes_split(&b, chunk), base, "chunk {chunk}");
        }
    }

    #[test]
    fn a_length_that_is_not_whole_words_is_refused() {
        assert!(buffer_lanes(&[1, 2, 3]).is_err());
        assert_eq!(buffer_lanes(&[]).unwrap(), [0; LANES]);
    }
}
