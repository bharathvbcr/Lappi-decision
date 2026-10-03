//! The file format between `tools/real_tokenizer_pipeline.py::native_minhash` and this binary.
//!
//! Little-endian throughout. Every count is checked against the bytes actually present, and
//! a file with bytes left over is refused: a reader that stops early would sign a prefix of
//! the corpus and say nothing.
//!
//! Input (`QDPMHIN1`):
//!
//! ```text
//! magic     8 bytes   b"QDPMHIN1"
//! num_perm  u32
//! key_len   u32, then key_len bytes      (MinHasher._key)
//! a         num_perm x u64               (MinHasher._a)
//! b         num_perm x u64               (MinHasher._b)
//! n_docs    u64
//! per doc:  n u32, then n x u32 shingle byte lengths, then the n shingles' bytes concatenated
//! ```
//!
//! Output (`QDPMHOK1`): `magic`, `num_perm u32`, `n_docs u64`, then `n_docs x num_perm` u64,
//! document-major, in input order.

use std::ops::Range;

use crate::minhash::Family;

pub const INPUT_MAGIC: &[u8; 8] = b"QDPMHIN1";
pub const OUTPUT_MAGIC: &[u8; 8] = b"QDPMHOK1";
/// More permutations than this is not a configuration `DataConfig` would accept (it uses 128).
pub const MAX_NUM_PERM: u32 = 4096;
/// An input past this is not a corpus this tool was built for. It was 4 GiB, sized to the
/// phase-3 corpus (~60 MB); the v5 containment scan's request was 4,534,193,280 bytes
/// (2026-10-03: the shingles travel as bytes, ~5x the text). 16 GiB is ~3.8x that. It stays
/// under linwire's 32 GiB, which its own test requires (`the_bound_admits_the_measured_mixture_
/// request`: linwire keeps a bound of its own, above this one, since J1's refusal at this one).
pub const MAX_INPUT_BYTES: u64 = 16 << 30;
/// Documents per call; the phase-3 corpus has ~50 thousand distinct shingle sets.
pub const MAX_DOCS: u64 = 1 << 24;

/// One document's shingles inside the input buffer.
#[derive(Clone, Debug)]
pub struct Doc {
    /// Byte range of the `n` u32 lengths.
    lengths: Range<usize>,
    /// Byte range of the concatenated shingle bytes.
    bytes: Range<usize>,
}

/// A parsed input file: the family and where each document's shingles are.
#[derive(Debug)]
pub struct Request<'b> {
    pub family: Family,
    pub docs: Vec<Doc>,
    buf: &'b [u8],
}

/// A bounds-checked reader over a request buffer; shared with [`crate::linwire`].
pub(crate) struct Cursor<'b> {
    pub(crate) buf: &'b [u8],
    pub(crate) at: usize,
}

impl<'b> Cursor<'b> {
    pub(crate) fn new(buf: &'b [u8]) -> Self {
        Self { buf, at: 0 }
    }

    pub(crate) fn take(&mut self, n: usize, what: &str) -> Result<Range<usize>, String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|end| *end <= self.buf.len())
            .ok_or_else(|| {
                format!(
                    "input ends inside {what}: {n} bytes needed at offset {}, {} present",
                    self.at,
                    self.buf.len() - self.at
                )
            })?;
        let range = self.at..end;
        self.at = end;
        Ok(range)
    }

    pub(crate) fn u32(&mut self, what: &str) -> Result<u32, String> {
        let r = self.take(4, what)?;
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.buf[r]);
        Ok(u32::from_le_bytes(bytes))
    }

    pub(crate) fn u64(&mut self, what: &str) -> Result<u64, String> {
        let r = self.take(8, what)?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.buf[r]);
        Ok(u64::from_le_bytes(bytes))
    }

    fn u64s(&mut self, n: usize, what: &str) -> Result<Vec<u64>, String> {
        (0..n).map(|_| self.u64(what)).collect()
    }
}

/// The little-endian u32s of a byte range whose length is a multiple of four (`parse` takes
/// exactly `4 * n` bytes for `n` lengths, so there is never a remainder to drop).
fn le_u32s(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|l| u32::from_le_bytes(*l))
}

/// Parse and validate an input buffer. Nothing is signed until all of it has been read.
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
    let num_perm = c.u32("num_perm")?;
    if !(1..=MAX_NUM_PERM).contains(&num_perm) {
        return Err(format!("num_perm {num_perm} is outside 1..={MAX_NUM_PERM}"));
    }
    let num_perm = num_perm as usize;
    let key_len = c.u32("key_len")? as usize;
    let key = c.take(key_len, "the key")?;
    let a = c.u64s(num_perm, "a")?;
    let b = c.u64s(num_perm, "b")?;
    let family = Family::new(&buf[key], a, b)?;
    let n_docs = c.u64("n_docs")?;
    if n_docs > MAX_DOCS {
        return Err(format!("{n_docs} documents; the bound is {MAX_DOCS}"));
    }
    let mut docs = Vec::with_capacity(n_docs as usize);
    for d in 0..n_docs {
        let n = c.u32("a shingle count")? as usize;
        if n == 0 {
            return Err(format!(
                "document {d} has no shingles; an empty set cannot be signed"
            ));
        }
        let lengths = c.take(
            n.checked_mul(4).ok_or("shingle count overflows")?,
            "lengths",
        )?;
        let total = le_u32s(&buf[lengths.clone()])
            .try_fold(0usize, |acc, l| acc.checked_add(l as usize))
            .ok_or_else(|| format!("document {d}'s shingle lengths overflow"))?;
        let bytes = c.take(total, "shingle bytes")?;
        docs.push(Doc { lengths, bytes });
    }
    if c.at != buf.len() {
        return Err(format!(
            "{} bytes follow the last of {n_docs} documents; refusing a file that is not \
             exactly what its header describes",
            buf.len() - c.at
        ));
    }
    Ok(Request { family, docs, buf })
}

impl Request<'_> {
    /// The shingles of one document, in input order.
    fn shingles(&self, doc: &Doc) -> impl Iterator<Item = &[u8]> {
        let mut at = doc.bytes.start;
        le_u32s(&self.buf[doc.lengths.clone()]).map(move |l| {
            let len = l as usize;
            let s = &self.buf[at..at + len];
            at += len;
            s
        })
    }

    /// Every document's signature, document-major, on up to `threads` threads. The result does
    /// not depend on `threads`: each document is signed independently into its own slots.
    pub fn sign(&self, threads: usize) -> Result<Vec<u64>, String> {
        let k = self.family.num_perm();
        let mut out = vec![0u64; self.docs.len() * k];
        if self.docs.is_empty() {
            return Ok(out);
        }
        let threads = threads.clamp(1, self.docs.len());
        let per = self.docs.len().div_ceil(threads);
        let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .docs
                .chunks(per)
                .zip(out.chunks_mut(per * k))
                .map(|(docs, slots)| {
                    scope.spawn(move || {
                        for (doc, sig) in docs.iter().zip(slots.chunks_exact_mut(k)) {
                            self.family.signature(self.shingles(doc), sig)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("a signing thread panicked".to_string()))
                })
                .collect()
        });
        results.into_iter().collect::<Result<Vec<()>, String>>()?;
        Ok(out)
    }
}

/// The output file's bytes for `signatures` (document-major, `num_perm` per document).
pub fn encode_output(num_perm: usize, signatures: &[u64]) -> Result<Vec<u8>, String> {
    if num_perm == 0 || !signatures.len().is_multiple_of(num_perm) {
        return Err(format!(
            "{} values are not whole {num_perm}-value signatures",
            signatures.len()
        ));
    }
    let n_docs = (signatures.len() / num_perm) as u64;
    let mut out = Vec::with_capacity(8 + 4 + 8 + signatures.len() * 8);
    out.extend_from_slice(OUTPUT_MAGIC);
    out.extend_from_slice(
        &u32::try_from(num_perm)
            .map_err(|e| e.to_string())?
            .to_le_bytes(),
    );
    out.extend_from_slice(&n_docs.to_le_bytes());
    for v in signatures {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(num_perm: u32, key: &[u8], docs: &[&[&[u8]]]) -> Vec<u8> {
        let mut v = INPUT_MAGIC.to_vec();
        v.extend_from_slice(&num_perm.to_le_bytes());
        v.extend_from_slice(&(key.len() as u32).to_le_bytes());
        v.extend_from_slice(key);
        for i in 0..num_perm {
            v.extend_from_slice(&u64::from(i + 1).to_le_bytes());
        }
        for i in 0..num_perm {
            v.extend_from_slice(&u64::from(i * 3).to_le_bytes());
        }
        v.extend_from_slice(&(docs.len() as u64).to_le_bytes());
        for doc in docs {
            v.extend_from_slice(&(doc.len() as u32).to_le_bytes());
            for s in *doc {
                v.extend_from_slice(&(s.len() as u32).to_le_bytes());
            }
            for s in *doc {
                v.extend_from_slice(s);
            }
        }
        v
    }

    #[test]
    fn signing_does_not_depend_on_the_thread_count() {
        let docs: Vec<Vec<Vec<u8>>> = (0..37)
            .map(|d| {
                (0..=d % 5)
                    .map(|s| format!("doc{d} shingle{s} ü").into_bytes())
                    .collect()
            })
            .collect();
        let refs: Vec<Vec<&[u8]>> = docs
            .iter()
            .map(|d| d.iter().map(Vec::as_slice).collect())
            .collect();
        let refs: Vec<&[&[u8]]> = refs.iter().map(Vec::as_slice).collect();
        let buf = input(16, &[5u8; 32], &refs);
        let req = parse(&buf).expect("parses");
        let one = req.sign(1).expect("signs");
        for threads in [2, 3, 8, 64, 1000] {
            assert_eq!(req.sign(threads).expect("signs"), one, "{threads} threads");
        }
        // Each document is its own set: signing it alone gives the same slots.
        for (d, doc) in refs.iter().enumerate() {
            let alone = parse(&input(16, &[5u8; 32], &[doc]))
                .expect("parses")
                .sign(1)
                .expect("signs");
            assert_eq!(alone[..], one[d * 16..(d + 1) * 16]);
        }
    }

    #[test]
    fn a_file_that_is_not_exactly_its_header_is_refused() {
        let docs: [&[&[u8]]; 2] = [&[b"a b"], &[b"c", b"d"]];
        let good = input(4, &[1u8; 32], &docs);
        assert!(parse(&good).is_ok());
        for cut in [0, 7, 8, 12, good.len() - 1] {
            assert!(
                parse(&good[..cut]).is_err(),
                "truncated at {cut} of {}",
                good.len()
            );
        }
        let mut longer = good.clone();
        longer.push(0);
        assert!(parse(&longer).is_err(), "a trailing byte");
        let mut magic = good.clone();
        magic[0] = b'X';
        assert!(parse(&magic).is_err());
        assert!(parse(&input(0, &[1u8; 32], &docs)).is_err(), "num_perm 0");
        let empty_doc: [&[&[u8]]; 1] = [&[]];
        assert!(
            parse(&input(4, &[1u8; 32], &empty_doc)).is_err(),
            "an empty shingle set"
        );
    }

    #[test]
    fn the_output_is_its_header_then_the_values() {
        let bytes = encode_output(2, &[1, 2, 3, 4]).expect("encodes");
        assert_eq!(&bytes[..8], OUTPUT_MAGIC);
        assert_eq!(bytes.len(), 8 + 4 + 8 + 4 * 8);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")),
            2
        );
        assert_eq!(
            u64::from_le_bytes(bytes[12..20].try_into().expect("8 bytes")),
            2
        );
        assert!(encode_output(3, &[1, 2, 3, 4]).is_err());
    }
}
