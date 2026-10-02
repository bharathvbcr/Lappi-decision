//! SHA-256 (FIPS 180-4) -- what Python's `hashlib.sha256` computes.
//!
//! Written out for the same reason as [`crate::blake2b`]: `sha2` is in the workspace, but
//! qd-prep takes nothing beyond clap (it is cross-built for the aarch64 box, and every
//! dependency is one more thing to cross-compile), and the repo rule is no new dependency
//! without asking. `qd-prep containment` needs it for the digests its attestation names
//! (`pairs_sha256`, `exclusions_sha256`, each target set's `digest`), which Python readers
//! recompute with `hashlib`. Checked against `hashlib` digests below.

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const BLOCK: usize = 64;

fn compress(h: &mut [u32; 8], block: &[u8; BLOCK]) {
    let mut w = [0u32; 64];
    for (i, word) in w.iter_mut().take(16).enumerate() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&block[i * 4..i * 4 + 4]);
        *word = u32::from_be_bytes(bytes);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (state, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
        *state = state.wrapping_add(v);
    }
}

/// An incremental SHA-256, fed in pieces: `hashlib.sha256()` with `update` and `hexdigest`.
#[derive(Clone, Debug)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; BLOCK],
    filled: usize,
    /// Message length in bytes so far. FIPS 180-4 bounds it below 2^64 bits.
    len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            h: H0,
            buf: [0u8; BLOCK],
            filled: 0,
            len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.filled > 0 {
            let take = (BLOCK - self.filled).min(data.len());
            self.buf[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled == BLOCK {
                let block = self.buf;
                compress(&mut self.h, &block);
                self.filled = 0;
            }
        }
        while data.len() >= BLOCK {
            let mut block = [0u8; BLOCK];
            block.copy_from_slice(&data[..BLOCK]);
            compress(&mut self.h, &block);
            data = &data[BLOCK..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.filled = data.len();
        }
    }

    /// The 32-byte digest. Consumes the hasher, as a finished SHA-256 cannot take more input.
    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        let rem = (self.filled + 1) % BLOCK;
        let zeros = if rem <= BLOCK - 8 {
            BLOCK - 8 - rem
        } else {
            2 * BLOCK - 8 - rem
        };
        pad.extend(std::iter::repeat_n(0u8, zeros));
        pad.extend_from_slice(&bits.to_be_bytes());
        // `update` would count the padding into the length; the length is already fixed.
        let len = self.len;
        self.update(&pad);
        self.len = len;
        debug_assert_eq!(self.filled, 0, "padding ends on a block boundary");
        let mut out = [0u8; 32];
        for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.h) {
            *chunk = word.to_be_bytes();
        }
        out
    }

    /// `hexdigest()`: lower-case hex of the digest.
    pub fn hex(self) -> String {
        hex(&self.finish())
    }
}

/// Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[usize::from(b >> 4)] as char);
        s.push(DIGITS[usize::from(b & 0x0f)] as char);
    }
    s
}

/// `hashlib.sha256(data).hexdigest()`.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every digest below was produced by CPython 3.13.14's `hashlib.sha256(msg).hexdigest()`
    /// on 2026-10-02 (the empty message and `abc` are also FIPS 180-4's own examples).
    #[test]
    fn matches_hashlib_across_block_and_padding_boundaries() {
        let a = |n: usize| vec![b'a'; n];
        let cases: Vec<(Vec<u8>, &str)> = vec![
            (
                vec![],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc".to_vec(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".to_vec(),
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                a(55),
                "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                a(56),
                "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                a(63),
                "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",
            ),
            (
                a(64),
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
            (
                a(65),
                "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0",
            ),
            (
                a(1000),
                "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
            ),
            (
                "ünïcødé x".as_bytes().to_vec(),
                "c0aee1d4251b819ffa55f71cb48adac6293de63445c33ca03eb66fd6912d0086",
            ),
        ];
        for (msg, want) in cases {
            assert_eq!(sha256_hex(&msg), want, "message of {} bytes", msg.len());
        }
    }

    #[test]
    fn feeding_in_pieces_is_feeding_at_once() {
        let msg: Vec<u8> = (0..2000u32).map(|i| (i * 7 % 251) as u8).collect();
        let whole = sha256_hex(&msg);
        for cut in [0usize, 1, 55, 63, 64, 65, 127, 128, 999, 2000] {
            let mut h = Sha256::new();
            h.update(&msg[..cut]);
            h.update(&msg[cut..]);
            assert_eq!(h.hex(), whole, "cut at {cut}");
        }
        let mut byte_by_byte = Sha256::new();
        for b in &msg {
            byte_by_byte.update(std::slice::from_ref(b));
        }
        assert_eq!(byte_by_byte.hex(), whole);
    }
}
