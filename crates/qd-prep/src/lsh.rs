//! Banded LSH candidate pairs: `qd_data.minhash.candidate_pairs`, pair for pair.
//!
//! The reference buckets every key of every band by `blake2b(b"|".join(v.to_bytes(8, "big")
//! for v in sig[lo:hi]), digest_size=16)`, walks the buckets of a band in the order their
//! first member arrived, sorts each bucket's keys, and adds every pair `(members[i],
//! members[j])`, `i < j`, to one set -- returning as soon as the set holds more than
//! `max_pairs`, with `truncated` true. This module does exactly that, in that order, so even a
//! truncated answer is the reference's set. Only the hashing runs on several threads (each
//! key's digest is independent of every other's); the bucket walk is sequential.
//!
//! It was 24.6 of the 78 s of the phase-4 `ft_splits` rebuild on the Mac (2026-10-01, one
//! unprofiled run at load ~40): a per-key, per-band Python loop of `int.to_bytes`,
//! `bytes.join` and `hashlib`, run twice per rebuild (`dedupe`'s content units, `split`'s
//! rows), and a `tools/real_ft_run.py` prelude with `--ood` rebuilds twice (the corpus and the
//! OOD record).
//!
//! Request (`QDPLSIN1`), little-endian throughout:
//!
//! ```text
//! magic 8 | bands u32 | rows u32 | max_pairs u64 | n_keys u64
//!         | n_keys x u32 key lengths | the keys' bytes, concatenated
//!         | signatures n_keys x (bands * rows) u64, key-major (only the banded prefix)
//! ```
//!
//! Reply (`QDPLSOK1`): `magic 8 | truncated u8 | n_pairs u64 | n_pairs x (lo u32, hi u32)`, key
//! indices in the order the reference added the pairs, `key[lo] < key[hi]` bytewise.
//!
//! Keys are opaque bytes, ordered bytewise. The caller encodes each `str` key with UTF-8 and
//! `surrogatepass`, which writes every code point (a lone surrogate included) in UTF-8's bit
//! layout, so bytewise order is Python's code-point order for `str`. A bands or rows of zero
//! is the reference's too: no bands bands nothing, and zero rows hashes the empty chunk, so
//! every key shares every band's one bucket.

use std::collections::{HashMap, HashSet};

use crate::blake2b::Keyed;
use crate::wire::Cursor;

pub const INPUT_MAGIC: &[u8; 8] = b"QDPLSIN1";
pub const OUTPUT_MAGIC: &[u8; 8] = b"QDPLSOK1";
/// The phase-4 rebuild's largest call is ~280 thousand keys of 128 values, ~290 MB; J1's is
/// about twice that. 4 GiB is the MinHash request's bound, and as far past either.
pub const MAX_INPUT_BYTES: u64 = 4 << 30;
/// Keys per call. Reply indices are u32.
pub const MAX_KEYS: u64 = 1 << 31;
/// `choose_bands` picks `bands * rows <= num_perm`, and `DataConfig` signs 128 permutations.
/// Bands and rows are each bounded by it as well, so zero rows cannot ask for 2^32 bands.
pub const MAX_BANDED_VALUES: u64 = 4096;
/// `qd_data.dedupe.DEFAULT_MAX_CANDIDATE_PAIRS` is five million; a bound far past it still
/// keeps the reply (8 bytes a pair) under a few GiB.
pub const MAX_PAIRS: u64 = 1 << 28;

/// A parsed request: the banding, the bound, each key's bytes and the banded signatures.
#[derive(Debug)]
pub struct Request<'b> {
    pub bands: usize,
    pub rows: usize,
    pub max_pairs: u64,
    pub keys: Vec<&'b [u8]>,
    /// `keys.len() x (bands * rows)` values, key-major.
    pub values: Vec<u64>,
}

/// Parse and validate a request. Nothing is hashed until all of it has been read.
pub fn parse(buf: &[u8]) -> Result<Request<'_>, String> {
    if buf.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; the bound is {MAX_INPUT_BYTES}",
            buf.len()
        ));
    }
    let mut c = Cursor::new(buf);
    let magic = c.take(8, "the magic")?;
    if &buf[magic] != INPUT_MAGIC {
        return Err(format!(
            "input does not start with {:?}",
            String::from_utf8_lossy(INPUT_MAGIC)
        ));
    }
    let bands = c.u32("bands")? as u64;
    let rows = c.u32("rows")? as u64;
    if bands > MAX_BANDED_VALUES || rows > MAX_BANDED_VALUES || bands * rows > MAX_BANDED_VALUES {
        return Err(format!(
            "{bands} bands of {rows} rows: each, and their product, must be at most \
             {MAX_BANDED_VALUES}"
        ));
    }
    let max_pairs = c.u64("max_pairs")?;
    if max_pairs > MAX_PAIRS {
        return Err(format!("max_pairs {max_pairs}; the bound is {MAX_PAIRS}"));
    }
    let n_keys = c.u64("n_keys")?;
    if n_keys > MAX_KEYS {
        return Err(format!("{n_keys} keys; the bound is {MAX_KEYS}"));
    }
    // The lengths are taken before anything is allocated for them, so a header claiming more
    // keys than the file holds is refused by the cursor, not by the allocator.
    let lengths = c.take(n_keys as usize * 4, "key lengths")?;
    let mut keys = Vec::with_capacity(n_keys as usize);
    for len in buf[lengths].as_chunks::<4>().0 {
        let range = c.take(u32::from_le_bytes(*len) as usize, "key bytes")?;
        keys.push(&buf[range]);
    }
    let width = (bands * rows) as usize;
    let n_values = keys
        .len()
        .checked_mul(width)
        .ok_or("the signature count overflows")?;
    let raw = c.take(
        n_values
            .checked_mul(8)
            .ok_or("the signature bytes overflow")?,
        "signatures",
    )?;
    if c.at != buf.len() {
        return Err(format!(
            "{} bytes follow the last of {n_keys} signatures; refusing a file that is not \
             exactly what its header describes",
            buf.len() - c.at
        ));
    }
    let values = buf[raw]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| u64::from_le_bytes(*v))
        .collect();
    Ok(Request {
        bands: bands as usize,
        rows: rows as usize,
        max_pairs,
        keys,
        values,
    })
}

impl Request<'_> {
    /// Every key's 16-byte digest for `band`, in key order, on up to `threads` threads.
    fn band_digests(&self, band: usize, threads: usize, hasher: &Keyed) -> Vec<[u8; 16]> {
        let width = self.bands * self.rows;
        let (lo, hi) = (band * self.rows, (band + 1) * self.rows);
        let mut out = vec![[0u8; 16]; self.keys.len()];
        if out.is_empty() {
            return out;
        }
        let threads = threads.clamp(1, out.len());
        let per = out.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, slots) in out.chunks_mut(per).enumerate() {
                scope.spawn(move || {
                    // `b"|".join(v.to_bytes(8, "big") for v in sig[lo:hi])`.
                    let mut chunk = Vec::with_capacity(self.rows * 9);
                    for (offset, slot) in slots.iter_mut().enumerate() {
                        let key = chunk_index * per + offset;
                        let sig = &self.values[key * width..(key + 1) * width];
                        chunk.clear();
                        for (j, v) in sig[lo..hi].iter().enumerate() {
                            if j > 0 {
                                chunk.push(b'|');
                            }
                            chunk.extend_from_slice(&v.to_be_bytes());
                        }
                        slot.copy_from_slice(&hasher.digest(&chunk));
                    }
                });
            }
        });
        out
    }

    /// The reference's pairs, in the order it added them, and whether it stopped at
    /// `max_pairs`. The result does not depend on `threads`.
    pub fn candidate_pairs(&self, threads: usize) -> Result<(Vec<(u32, u32)>, bool), String> {
        let hasher = Keyed::new(&[], 16)?;
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut order: Vec<(u32, u32)> = Vec::new();
        for band in 0..self.bands {
            let digests = self.band_digests(band, threads, &hasher);
            // Buckets in the order their first member arrived: a Python dict's iteration.
            let mut slot_of: HashMap<[u8; 16], usize> = HashMap::new();
            let mut buckets: Vec<Vec<u32>> = Vec::new();
            for (key, digest) in digests.iter().enumerate() {
                let slot = *slot_of.entry(*digest).or_insert_with(|| {
                    buckets.push(Vec::new());
                    buckets.len() - 1
                });
                buckets[slot].push(key as u32);
            }
            for mut members in buckets {
                if members.len() < 2 {
                    continue;
                }
                members.sort_by(|a, b| self.keys[*a as usize].cmp(self.keys[*b as usize]));
                for i in 0..members.len() {
                    for j in i + 1..members.len() {
                        let pair = (members[i], members[j]);
                        if seen.insert(pair) {
                            order.push(pair);
                            if order.len() as u64 > self.max_pairs {
                                return Ok((order, true));
                            }
                        }
                    }
                }
            }
        }
        Ok((order, false))
    }
}

/// The reply's bytes.
pub fn encode_output(pairs: &[(u32, u32)], truncated: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + 1 + 8 + pairs.len() * 8);
    out.extend_from_slice(OUTPUT_MAGIC);
    out.push(u8::from(truncated));
    out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
    for (lo, hi) in pairs {
        out.extend_from_slice(&lo.to_le_bytes());
        out.extend_from_slice(&hi.to_le_bytes());
    }
    out
}

/// `qd-prep lsh`: request bytes in, reply bytes and a summary line out.
pub fn run_lsh(buf: &[u8], threads: usize) -> Result<(Vec<u8>, String), String> {
    let request = parse(buf)?;
    let (pairs, truncated) = request.candidate_pairs(threads)?;
    Ok((
        encode_output(&pairs, truncated),
        format!(
            "qd-prep lsh: {} keys x {} bands of {} rows -> {} pairs{} on {threads} thread(s)",
            request.keys.len(),
            request.bands,
            request.rows,
            pairs.len(),
            if truncated { " (truncated)" } else { "" },
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(bands: u32, rows: u32, max_pairs: u64, keys: &[&str], sigs: &[Vec<u64>]) -> Vec<u8> {
        let mut out = INPUT_MAGIC.to_vec();
        out.extend_from_slice(&bands.to_le_bytes());
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&max_pairs.to_le_bytes());
        out.extend_from_slice(&(keys.len() as u64).to_le_bytes());
        for k in keys {
            out.extend_from_slice(&(k.len() as u32).to_le_bytes());
        }
        for k in keys {
            out.extend_from_slice(k.as_bytes());
        }
        for s in sigs {
            for v in s {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    fn pairs(buf: &[u8], threads: usize) -> (Vec<(u32, u32)>, bool) {
        parse(buf).unwrap().candidate_pairs(threads).unwrap()
    }

    #[test]
    fn keys_sharing_a_band_pair_up_in_key_order() {
        // b and a agree on band 0 only; c on band 1 with a. Pairs come out in band order,
        // each with the bytewise-smaller key first.
        let keys = ["b", "a", "c"];
        let sigs = [vec![1, 2, 3, 4], vec![1, 2, 9, 9], vec![7, 7, 9, 9]];
        let buf = request(2, 2, 100, &keys, &sigs);
        assert_eq!(pairs(&buf, 1), (vec![(1, 0), (1, 2)], false));
        assert_eq!(pairs(&buf, 3), (vec![(1, 0), (1, 2)], false));
    }

    #[test]
    fn a_pair_seen_in_two_bands_counts_once_and_the_bound_truncates() {
        let keys = ["k0", "k1", "k2"];
        let sigs = [vec![5, 5], vec![5, 5], vec![5, 5]];
        let all = request(2, 1, 100, &keys, &sigs);
        assert_eq!(pairs(&all, 2), (vec![(0, 1), (0, 2), (1, 2)], false));
        // The pair that crosses the bound is kept, as the reference's set.add precedes its check.
        assert_eq!(
            pairs(&request(2, 1, 1, &keys, &sigs), 1),
            (vec![(0, 1), (0, 2)], true)
        );
        assert_eq!(
            pairs(&request(2, 1, 0, &keys, &sigs), 1),
            (vec![(0, 1)], true)
        );
        assert!(!pairs(&request(2, 1, 3, &keys, &sigs), 1).1);
    }

    #[test]
    fn no_keys_no_pairs() {
        assert_eq!(pairs(&request(4, 2, 10, &[], &[]), 8), (vec![], false));
    }

    #[test]
    fn a_band_digest_is_pythons_blake2b_of_the_big_endian_chunk() {
        // hashlib.blake2b(b"|".join(v.to_bytes(8, "big") for v in sig[lo:hi]), digest_size=16)
        let hex = |d: &[u8; 16]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let hasher = Keyed::new(&[], 16).unwrap();
        let two = request(2, 1, 0, &["a", "b"], &[vec![1, 7], vec![u64::MAX, 7]]);
        let r = parse(&two).unwrap();
        assert_eq!(
            hex(&r.band_digests(1, 2, &hasher)[0]),
            "5ab07ffcb868c737f9ce09bfb5044208"
        );
        let wide = request(1, 2, 0, &["a"], &[vec![1, u64::MAX]]);
        let w = parse(&wide).unwrap();
        assert_eq!(
            hex(&w.band_digests(0, 1, &hasher)[0]),
            "584b6c118233cd4c469d6e0251d3569b"
        );
        let empty = request(1, 0, 0, &["a"], &[vec![]]);
        let e = parse(&empty).unwrap();
        assert_eq!(
            hex(&e.band_digests(0, 1, &hasher)[0]),
            "cae66941d9efbd404e4d88758ea67670"
        );
    }

    #[test]
    fn zero_rows_buckets_every_key_together_and_zero_bands_bands_nothing() {
        let keys = ["c", "a", "b"];
        let none: Vec<Vec<u64>> = vec![vec![], vec![], vec![]];
        assert_eq!(
            pairs(&request(2, 0, 10, &keys, &none), 2),
            (vec![(1, 2), (1, 0), (2, 0)], false)
        );
        assert_eq!(pairs(&request(0, 3, 10, &keys, &none), 2), (vec![], false));
    }

    #[test]
    fn keys_order_bytewise_whatever_their_encoding() {
        // U+00E9 (C3 A9) sorts after z (7A), as its code point does; a lone surrogate arrives
        // from Python's surrogatepass as ED A0 80 and sorts after both, as U+D800 does.
        let keys = ["\u{e9}", "z", "\u{e9}!"];
        let mut buf = request(1, 1, 10, &keys, &[vec![3], vec![3], vec![3]]);
        // Rewrite the third key ("\u{e9}!", three bytes) as the surrogate's three bytes.
        let third = buf.len() - 3 * 8 - 3;
        buf[third..third + 3].copy_from_slice(&[0xED, 0xA0, 0x80]);
        assert_eq!(pairs(&buf, 1), (vec![(1, 0), (1, 2), (0, 2)], false));
    }

    #[test]
    fn malformed_requests_are_refused() {
        let good = request(1, 2, 3, &["a"], &[vec![1, 2]]);
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(parse(&bad_magic).unwrap_err().contains("does not start"));
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(parse(&trailing).unwrap_err().contains("follow the last"));
        assert!(
            parse(&good[..good.len() - 1])
                .unwrap_err()
                .contains("signatures")
        );
        let many_bands = request(MAX_BANDED_VALUES as u32 + 1, 0, 3, &["a"], &[vec![]]);
        assert!(parse(&many_bands).unwrap_err().contains("must be at most"));
        let wide = request(65, 64, 3, &[], &[]);
        assert!(parse(&wide).unwrap_err().contains("must be at most"));
        let mut claims_more = request(1, 2, 3, &["a"], &[vec![1, 2]]);
        let n_keys_at = 8 + 4 + 4 + 8;
        claims_more[n_keys_at..n_keys_at + 8].copy_from_slice(&MAX_KEYS.to_le_bytes());
        assert!(parse(&claims_more).unwrap_err().contains("key lengths"));
        let too_many = request(1, 1, MAX_PAIRS + 1, &["a"], &[vec![1]]);
        assert!(parse(&too_many).unwrap_err().contains("max_pairs"));
    }

    #[test]
    fn the_reply_is_magic_flag_count_and_pairs() {
        let bytes = encode_output(&[(0, 2)], true);
        assert_eq!(&bytes[..8], OUTPUT_MAGIC);
        assert_eq!(bytes[8], 1);
        assert_eq!(u64::from_le_bytes(bytes[9..17].try_into().unwrap()), 1);
        assert_eq!(&bytes[17..], &[0, 0, 0, 0, 2, 0, 0, 0]);
    }
}
