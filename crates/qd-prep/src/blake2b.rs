//! BLAKE2b (RFC 7693), sequential mode, with an optional key and a 1..=64-byte digest.
//!
//! Written out here because the workspace carries no BLAKE2 crate and new dependencies need a
//! human yes. It is the hash `qd_data.minhash` builds its permutation family and shingle hashes
//! from (`hashlib.blake2b(data, key=..., digest_size=...)`), so it must agree with CPython's
//! `hashlib` bit for bit: the parameter block is the default one hashlib uses -- fanout 1,
//! depth 1, no salt, no personalisation -- and nothing else is configurable.
//!
//! Agreement is not assumed: `tests/blake2b_vectors.rs` holds digests printed by `hashlib` for
//! empty, one-block, block-boundary and multi-block inputs, keyed and unkeyed, at digest sizes
//! 8, 16, 32 and 64, and `python/tests/test_qd_prep_parity.py` compares whole MinHash signatures
//! against the Python reference.

const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

const BLOCK: usize = 128;

/// The largest key and digest BLAKE2b defines.
pub const MAX_KEY: usize = 64;
pub const MAX_DIGEST: usize = 64;

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; BLOCK], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, word) in m.iter_mut().enumerate() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&block[i * 8..i * 8 + 8]);
        *word = u64::from_le_bytes(bytes);
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in &SIGMA {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// `hashlib.blake2b(data, key=key, digest_size=out.len()).digest()`, written into `out`.
///
/// Panics on a key over 64 bytes or a digest outside 1..=64: both are programming errors in
/// this crate (every caller passes a constant), never data-dependent.
pub fn blake2b_into(out: &mut [u8], key: &[u8], data: &[u8]) {
    let nn = out.len();
    let kk = key.len();
    assert!(
        (1..=MAX_DIGEST).contains(&nn),
        "BLAKE2b digest size {nn} outside 1..=64"
    );
    assert!(kk <= MAX_KEY, "BLAKE2b key of {kk} bytes exceeds 64");

    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ ((kk as u64) << 8) ^ (nn as u64);

    // The key, when present, is the first block of the message, zero-padded.
    let mut t: u128 = 0;
    let mut block = [0u8; BLOCK];
    let mut rest = data;
    if kk > 0 {
        block[..kk].copy_from_slice(key);
        if rest.is_empty() {
            t += BLOCK as u128;
            compress(&mut h, &block, t, true);
            write_digest(&h, out);
            return;
        }
        t += BLOCK as u128;
        compress(&mut h, &block, t, false);
    }
    if rest.is_empty() {
        // Unkeyed empty message: one all-zero block, counter zero, final.
        compress(&mut h, &[0u8; BLOCK], 0, true);
        write_digest(&h, out);
        return;
    }
    while rest.len() > BLOCK {
        block.copy_from_slice(&rest[..BLOCK]);
        t += BLOCK as u128;
        compress(&mut h, &block, t, false);
        rest = &rest[BLOCK..];
    }
    let mut last = [0u8; BLOCK];
    last[..rest.len()].copy_from_slice(rest);
    t += rest.len() as u128;
    compress(&mut h, &last, t, true);
    write_digest(&h, out);
}

fn write_digest(h: &[u64; 8], out: &mut [u8]) {
    let mut full = [0u8; 64];
    for (i, word) in h.iter().enumerate() {
        full[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    let n = out.len();
    out.copy_from_slice(&full[..n]);
}

/// The digest as a fixed-size array.
pub fn blake2b<const N: usize>(key: &[u8], data: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    blake2b_into(&mut out, key, data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn rfc7693_abc() {
        // RFC 7693 appendix A: BLAKE2b-512("abc").
        assert_eq!(
            hex(&blake2b::<64>(b"", b"abc")),
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
             7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
        );
    }
}
