//! The file contract between `tools/real_tokenizer_pipeline.py` and `qd-prep minhash`.
//!
//! Little-endian throughout. Input:
//!
//! ```text
//! magic     8 bytes   b"QDMHIN01"
//! num_perm  u32
//! seed_len  u32       then seed_len bytes: the seed's decimal spelling, str(seed)
//! n_sets    u64
//! per set:  n_items u32, then n_items lengths (u32), then the items' bytes back to back
//! ```
//!
//! Output, written to a temporary name and renamed into place so a reader never sees half
//! of it:
//!
//! ```text
//! magic     8 bytes   b"QDMHOUT1"
//! num_perm  u32
//! reserved  u32       zero, so the signatures start 8-byte aligned
//! n_sets    u64
//! sigs      n_sets * num_perm values, u64, row-major in input order
//! ```
//!
//! Every count is checked against the bytes actually present before it is used, and bytes
//! left over after the last set refuse the file: a request that does not parse exactly is
//! not a request this binary understands.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, bail};

use crate::minhash::{MAX_NUM_PERM, ShingleSet};

pub const INPUT_MAGIC: &[u8; 8] = b"QDMHIN01";
pub const OUTPUT_MAGIC: &[u8; 8] = b"QDMHOUT1";

/// The protocol this binary speaks; the Python adapter refuses any other.
pub const MINHASH_PROTOCOL: u32 = 1;

/// Bound on a request file. The phase-4 corpus is a few hundred MB of shingles; a file
/// past this is a corrupted or runaway request, refused before it is read.
pub const MAX_INPUT_BYTES: u64 = 16 << 30;

/// Bound on the seed's spelling.
pub const MAX_SEED_LEN: usize = 4096;

/// A parsed request. The shingle sets borrow from the bytes they were read from.
#[derive(Debug)]
pub struct Request<'a> {
    pub num_perm: usize,
    pub seed: &'a str,
    pub sets: Vec<ShingleSet<'a>>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize, what: &str) -> anyhow::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .with_context(|| {
                format!(
                    "{what}: needs {n} byte(s) at offset {}, the request holds {}",
                    self.at,
                    self.buf.len()
                )
            })?;
        let out = &self.buf[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self, what: &str) -> anyhow::Result<u32> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self, what: &str) -> anyhow::Result<u64> {
        let b = self.take(8, what)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
}

fn parse_set<'a>(c: &mut Cursor<'a>) -> anyhow::Result<ShingleSet<'a>> {
    let n_items = c.u32("set size")? as usize;
    let lens_raw = c.take(
        n_items.checked_mul(4).context("set size overflows")?,
        "item lengths",
    )?;
    let (lens, rest) = lens_raw.as_chunks::<4>();
    debug_assert!(rest.is_empty(), "lens_raw is n_items * 4 bytes");
    let mut items: ShingleSet<'a> = Vec::with_capacity(n_items);
    for len in lens {
        items.push(c.take(u32::from_le_bytes(*len) as usize, "item bytes")?);
    }
    Ok(items)
}

pub fn parse_request(buf: &[u8]) -> anyhow::Result<Request<'_>> {
    let mut c = Cursor { buf, at: 0 };
    if c.take(8, "magic")? != INPUT_MAGIC {
        bail!("not a qd-prep minhash request: the magic is not {INPUT_MAGIC:?}");
    }
    let num_perm = c.u32("num_perm")? as usize;
    if num_perm == 0 || num_perm > MAX_NUM_PERM {
        bail!("num_perm must be in 1..={MAX_NUM_PERM}, got {num_perm}");
    }
    let seed_len = c.u32("seed length")? as usize;
    if seed_len > MAX_SEED_LEN {
        bail!("the seed's spelling is {seed_len} bytes, over the {MAX_SEED_LEN}-byte bound");
    }
    let seed = std::str::from_utf8(c.take(seed_len, "seed")?).context("seed is not UTF-8")?;
    let n_sets = c.u64("n_sets")?;
    // Each set costs at least its 4-byte count, so a count the file cannot hold is refused
    // before anything is allocated for it.
    let remaining = (buf.len() - c.at) as u64;
    if n_sets > remaining / 4 {
        bail!("the request claims {n_sets} sets but holds {remaining} more byte(s)");
    }
    let mut sets = Vec::with_capacity(n_sets as usize);
    for s in 0..n_sets {
        sets.push(parse_set(&mut c).with_context(|| format!("set {s}"))?);
    }
    if c.at != buf.len() {
        bail!(
            "{} byte(s) follow the last set; the request does not parse exactly",
            buf.len() - c.at
        );
    }
    Ok(Request {
        num_perm,
        seed,
        sets,
    })
}

pub fn read_request_file(path: &Path) -> anyhow::Result<Vec<u8>> {
    let size = std::fs::metadata(path)
        .with_context(|| format!("{}: cannot stat the request", path.display()))?
        .len();
    if size > MAX_INPUT_BYTES {
        bail!(
            "{}: {size} bytes, over the {MAX_INPUT_BYTES}-byte bound",
            path.display()
        );
    }
    std::fs::read(path).with_context(|| format!("{}: cannot read the request", path.display()))
}

/// The output bytes for `n_sets` signatures of `num_perm` values each.
pub fn encode_output(num_perm: usize, n_sets: usize, sigs: &[u64]) -> anyhow::Result<Vec<u8>> {
    if sigs.len() != num_perm * n_sets {
        bail!(
            "{} signature values for {n_sets} sets of {num_perm}",
            sigs.len()
        );
    }
    let mut out = Vec::with_capacity(24 + sigs.len() * 8);
    out.extend_from_slice(OUTPUT_MAGIC);
    out.extend_from_slice(&(num_perm as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(n_sets as u64).to_le_bytes());
    for v in sigs {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// Write-then-rename in the destination's directory, fsynced.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let name = path
        .file_name()
        .with_context(|| format!("{}: not a file path", path.display()))?;
    let tmp = path.with_file_name(format!(".{}.tmp", name.to_string_lossy()));
    {
        let mut fh = std::fs::File::create(&tmp)
            .with_context(|| format!("{}: cannot create", tmp.display()))?;
        fh.write_all(bytes)
            .with_context(|| format!("{}: write failed", tmp.display()))?;
        fh.sync_all()
            .with_context(|| format!("{}: fsync failed", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("{} -> {}: rename failed", tmp.display(), path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(num_perm: u32, seed: &str, sets: &[&[&[u8]]]) -> Vec<u8> {
        let mut b = INPUT_MAGIC.to_vec();
        b.extend_from_slice(&num_perm.to_le_bytes());
        b.extend_from_slice(&(seed.len() as u32).to_le_bytes());
        b.extend_from_slice(seed.as_bytes());
        b.extend_from_slice(&(sets.len() as u64).to_le_bytes());
        for set in sets {
            b.extend_from_slice(&(set.len() as u32).to_le_bytes());
            for item in *set {
                b.extend_from_slice(&(item.len() as u32).to_le_bytes());
            }
            for item in *set {
                b.extend_from_slice(item);
            }
        }
        b
    }

    #[test]
    fn a_request_round_trips() {
        let bytes = request(8, "-12", &[&[b"a b", b""], &[b"\xff\x00"]]);
        let r = parse_request(&bytes).unwrap();
        assert_eq!((r.num_perm, r.seed), (8, "-12"));
        assert_eq!(
            r.sets,
            vec![vec![&b"a b"[..], &b""[..]], vec![&b"\xff\x00"[..]]]
        );
    }

    #[test]
    fn malformed_requests_are_refused() {
        let good = request(8, "1", &[&[b"abc"]]);
        assert!(parse_request(&good).is_ok());
        // Every strict prefix is short somewhere.
        for cut in 0..good.len() {
            assert!(
                parse_request(&good[..cut]).is_err(),
                "prefix of {cut} bytes parsed"
            );
        }
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(parse_request(&trailing).is_err());
        let mut magic = good.clone();
        magic[0] = b'X';
        assert!(parse_request(&magic).is_err());
        assert!(parse_request(&request(0, "1", &[])).is_err());
        assert!(parse_request(&request(MAX_NUM_PERM as u32 + 1, "1", &[])).is_err());
        // A set count the file cannot hold is refused before allocation.
        let mut huge = request(8, "1", &[]);
        let at = huge.len() - 8;
        huge[at..].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse_request(&huge).is_err());
    }

    #[test]
    fn output_layout() {
        let out = encode_output(2, 2, &[1, 2, 3, u64::MAX]).unwrap();
        assert_eq!(&out[..8], OUTPUT_MAGIC);
        assert_eq!(out.len(), 24 + 32);
        assert_eq!(&out[48..], &u64::MAX.to_le_bytes());
        assert!(encode_output(2, 2, &[1, 2, 3]).is_err());
    }
}
