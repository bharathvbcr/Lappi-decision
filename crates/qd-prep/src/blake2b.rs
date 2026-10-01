//! BLAKE2b (RFC 7693), keyed, any digest length 1..=64 -- what Python's `hashlib.blake2b`
//! computes with `key=` and `digest_size=`.
//!
//! Written out because no blake2 crate is in `[workspace.dependencies]` and the repo rule is
//! no new dependency without asking. It is checked against `hashlib` digests in the tests
//! below (every block-boundary case, keyed and unkeyed, the empty message with a key) and,
//! through the MinHash signatures it feeds, against `qd_data.minhash` in
//! `python/tests/test_qd_prep_minhash_parity.py`.

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
/// The largest key RFC 7693 defines for BLAKE2b.
pub const MAX_KEY: usize = 64;
/// The largest digest RFC 7693 defines for BLAKE2b.
pub const MAX_OUT: usize = 64;

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
    // The low and high words of the byte counter. Truncation is the point: the counter is
    // 128 bits and is split across two 64-bit words.
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

/// A BLAKE2b instance for one (key, digest length), with the key block already absorbed, so
/// that hashing many short messages under one key costs each message only its own blocks.
#[derive(Clone, Debug)]
pub struct Keyed {
    /// `IV[0]` is XORed with this parameter word: fanout 1, depth 1, key length, digest length.
    param: u64,
    /// State after the key block -- valid only for a NON-empty message, because with a key and
    /// no data RFC 7693 makes the key block itself the final block.
    h_after_key: [u64; 8],
    /// The zero-padded key block, `None` when the key is empty.
    key_block: Option<[u8; BLOCK]>,
    out_len: usize,
}

impl Keyed {
    /// Refuses a key longer than 64 bytes or a digest length outside 1..=64, as `hashlib` does.
    pub fn new(key: &[u8], out_len: usize) -> Result<Self, String> {
        if key.len() > MAX_KEY {
            return Err(format!(
                "blake2b key is {} bytes; at most {MAX_KEY}",
                key.len()
            ));
        }
        if !(1..=MAX_OUT).contains(&out_len) {
            return Err(format!(
                "blake2b digest length {out_len} is outside 1..={MAX_OUT}"
            ));
        }
        let param = 0x0101_0000 ^ ((key.len() as u64) << 8) ^ out_len as u64;
        let mut h = IV;
        h[0] ^= param;
        let key_block = if key.is_empty() {
            None
        } else {
            let mut block = [0u8; BLOCK];
            block[..key.len()].copy_from_slice(key);
            compress(&mut h, &block, BLOCK as u128, false);
            Some(block)
        };
        Ok(Self {
            param,
            h_after_key: h,
            key_block,
            out_len,
        })
    }

    /// The full 64-byte output state of `msg`; the digest is its first `out_len` bytes.
    fn state(&self, msg: &[u8]) -> [u64; 8] {
        if msg.is_empty() {
            let mut h = IV;
            h[0] ^= self.param;
            match &self.key_block {
                Some(block) => compress(&mut h, block, BLOCK as u128, true),
                None => compress(&mut h, &[0u8; BLOCK], 0, true),
            }
            return h;
        }
        let mut h = self.h_after_key;
        let mut t: u128 = if self.key_block.is_some() {
            BLOCK as u128
        } else {
            0
        };
        let mut rest = msg;
        // Every block but the last is non-final, including a last block that is exactly full.
        while rest.len() > BLOCK {
            let (head, tail) = rest.split_at(BLOCK);
            let mut block = [0u8; BLOCK];
            block.copy_from_slice(head);
            t += BLOCK as u128;
            compress(&mut h, &block, t, false);
            rest = tail;
        }
        let mut block = [0u8; BLOCK];
        block[..rest.len()].copy_from_slice(rest);
        t += rest.len() as u128;
        compress(&mut h, &block, t, true);
        h
    }

    /// The digest of `msg`, `out_len` bytes.
    pub fn digest(&self, msg: &[u8]) -> Vec<u8> {
        let h = self.state(msg);
        let mut out = Vec::with_capacity(MAX_OUT);
        for word in h {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.truncate(self.out_len);
        out
    }

    /// An 8-byte digest read big-endian: `int.from_bytes(blake2b(msg, key=k,
    /// digest_size=8).digest(), "big")`, which is `qd_data.minhash.MinHasher.base_hash`.
    /// Refuses an instance built for any other digest length, whose bytes would differ.
    pub fn u64_be(&self, msg: &[u8]) -> Result<u64, String> {
        if self.out_len != 8 {
            return Err(format!(
                "u64_be needs an 8-byte digest, this one is {}",
                self.out_len
            ));
        }
        // The first digest byte is the low byte of h[0]; big-endian over the eight digest
        // bytes is therefore h[0] byte-swapped.
        Ok(self.state(msg)[0].swap_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn key32() -> Vec<u8> {
        (0u8..32).collect()
    }

    /// Every vector below was produced by CPython's `hashlib.blake2b(msg, key=key,
    /// digest_size=n).hexdigest()` (3.14.7) on 2026-09-30; none is restated from the RFC except
    /// `abc`, which both agree on.
    #[test]
    fn matches_hashlib_across_block_boundaries_keys_and_lengths() {
        let a = |n: usize| vec![b'a'; n];
        let cases: Vec<(Vec<u8>, Vec<u8>, usize, &str)> = vec![
            (
                vec![],
                vec![],
                64,
                "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
            ),
            (
                b"abc".to_vec(),
                vec![],
                64,
                "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
            ),
            (vec![], key32(), 8, "b6fbe0459eb623da"),
            (
                vec![],
                key32(),
                32,
                "4e51e7a913fc80137da52880fecca175bf81e117d5c68126dc2774033517ea0d",
            ),
            (a(127), key32(), 8, "8a22bbd2b403c0a9"),
            (a(128), key32(), 8, "fbb0c21a1139534f"),
            (a(129), key32(), 8, "131d0b654ab88775"),
            (
                a(255),
                vec![],
                32,
                "177ec7b22a982dd81ec80e0f8fd488bb347952a0876fed488191b6dede62df81",
            ),
            (
                a(256),
                vec![],
                32,
                "eae4d3a7627549b383179dc18049964f91a6fed14c9f3fb26705eda3eeda5558",
            ),
            (
                a(257),
                key32(),
                64,
                "920024fd6e8d9971b48ab27be0ea7504bda6c51d5a6ef5e7ee17226bee540dd38d81702426166d186faf532b03362041f58181fe770644fda5bef1557d9de646",
            ),
            (
                "ünïcødé x".as_bytes().to_vec(),
                key32(),
                8,
                "04e9a2259c460f1a",
            ),
            (vec![0u8; 128], vec![], 8, "2ca5d3f9f96a9545"),
            (
                (0..3).flat_map(|_| 0u8..=255).collect(),
                (0u8..64).collect(),
                64,
                "81380d61d438c42eb39629bf670f5e5eda128d8c62eb28668899496fefe844a0631a10cd97654adbe9322bc7582445b557dd14a6aa794b91a74c6ca664e0f877",
            ),
        ];
        for (msg, key, n, want) in cases {
            let got = Keyed::new(&key, n).expect("valid parameters").digest(&msg);
            assert_eq!(
                hex(&got),
                want,
                "msg len {} key len {} out {n}",
                msg.len(),
                key.len()
            );
        }
    }

    #[test]
    fn u64_be_is_the_digest_read_big_endian() {
        let k = Keyed::new(&key32(), 8).expect("valid parameters");
        for msg in [&b""[..], b"x", &[b'a'; 129][..], "ünïcødé x".as_bytes()] {
            let d = k.digest(msg);
            let mut be = [0u8; 8];
            be.copy_from_slice(&d);
            assert_eq!(k.u64_be(msg), Ok(u64::from_be_bytes(be)));
        }
        let wide = Keyed::new(&key32(), 16).expect("valid parameters");
        assert!(
            wide.u64_be(b"x").is_err(),
            "a 16-byte digest is not base_hash's"
        );
    }

    #[test]
    fn refuses_what_hashlib_refuses() {
        assert!(Keyed::new(&[0u8; 65], 8).is_err());
        assert!(Keyed::new(&[], 0).is_err());
        assert!(Keyed::new(&[], 65).is_err());
        assert!(Keyed::new(&[0u8; 64], 64).is_ok());
    }
}
