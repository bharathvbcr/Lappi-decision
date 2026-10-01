//! `qd_train.baseline.CharNGramHasher.transform`, evaluated in Rust.
//!
//! The reference hashes every character n-gram, `n_min..=n_max` code points long, of every
//! document: FNV-1a 64-bit over the gram's UTF-8 bytes, masked to `dim` columns, counted per
//! document, L2-normalised, and stored with the columns ascending. Python does it one gram at a
//! time with an interpreted byte loop -- the measured CPU bottleneck of the FT linear control
//! together with the fit (`crates/qd-prep/src/linfit.rs`).
//!
//! **Why the result is bit-identical, not merely close.** The counts are integers, accumulated
//! in float64 by the reference and exactly in `u32` here. The reference's norm is
//! `sqrt(np.sum(vals * vals))`: every square and every partial sum is an integer, so while the
//! total is at most 2**53 the float64 sum is exact in any order, and `sqrt` and the division
//! are correctly rounded in both languages. Above 2**53 the reference's answer would depend on
//! numpy's summation order, so such a document is refused rather than approximated -- it would
//! take tens of millions of characters, against rendered prompts of a few kilobytes. The order
//! in which grams are counted (the reference loops `n` outermost, this loops start positions)
//! cannot change an integer count, and both emit the columns sorted.
//!
//! A Python `str` is a sequence of code points and a Rust `str` a sequence of scalar values;
//! they coincide for every string that encodes to UTF-8, and the adapter encodes strictly, so a
//! lone surrogate is a refusal there rather than a silent difference here.

use std::sync::Mutex;

/// FNV-1a 64-bit offset basis, as `CharNGramHasher._hash` starts.
pub const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
/// FNV-1a 64-bit prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
/// The largest sum of squared counts whose float64 sum is exact in any order.
const EXACT_SUM_BOUND: u128 = 1 << 53;
/// Longest gram the hasher accepts; `CharNGramHasher` defaults to 3..=5.
pub const MAX_ORDER: u32 = 64;
/// Widest feature space accepted (the reference's default is 2**16).
pub const MAX_DIM: u32 = 1 << 24;

/// The hasher's parameters, validated as `CharNGramHasher.__init__` validates them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NGramSpec {
    pub n_min: u32,
    pub n_max: u32,
    pub dim: u32,
}

impl NGramSpec {
    pub fn new(n_min: u32, n_max: u32, dim: u32) -> Result<Self, String> {
        if n_min < 1 || n_max < n_min || n_max > MAX_ORDER {
            return Err(format!(
                "require 1 <= n_min <= n_max <= {MAX_ORDER}, got {n_min}, {n_max}"
            ));
        }
        if dim < 16 || !dim.is_power_of_two() || dim > MAX_DIM {
            return Err(format!(
                "dim must be a power of two in 16..={MAX_DIM}, got {dim}"
            ));
        }
        Ok(Self { n_min, n_max, dim })
    }
}

/// FNV-1a over `bytes`, continuing from `h`.
#[inline(always)]
pub fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
    }
    h
}

/// Rows of a sparse matrix, as `qd_train.baseline.CSR` holds them: row `i` is
/// `data[indptr[i]..indptr[i+1]]` at columns `indices[...]`, columns strictly ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct Csr {
    pub n_cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub data: Vec<f64>,
}

impl Csr {
    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Row `r`'s columns and values.
    #[inline(always)]
    pub fn row(&self, r: usize) -> (&[u32], &[f64]) {
        let (a, b) = (self.indptr[r], self.indptr[r + 1]);
        (&self.indices[a..b], &self.data[a..b])
    }

    /// Refuse anything `CharNGramHasher` or `ContextLengthFeatures` could not have produced:
    /// a malformed `indptr`, a column out of range or out of order (a repeated column would be
    /// summed by one of the reference's two operands and overwritten by the other), or a value
    /// that is not finite.
    pub fn validate(&self) -> Result<(), String> {
        if self.indptr.first() != Some(&0) || self.indptr.last() != Some(&self.indices.len()) {
            return Err(format!(
                "indptr must run from 0 to nnz {}, got {:?}..{:?}",
                self.indices.len(),
                self.indptr.first(),
                self.indptr.last()
            ));
        }
        if self.indices.len() != self.data.len() {
            return Err(format!(
                "{} column indices for {} values",
                self.indices.len(),
                self.data.len()
            ));
        }
        for (r, w) in self.indptr.windows(2).enumerate() {
            if w[1] < w[0] {
                return Err(format!("indptr decreases at row {r}"));
            }
            let cols = &self.indices[w[0]..w[1]];
            if let Some(c) = cols.iter().find(|c| **c as usize >= self.n_cols) {
                return Err(format!(
                    "row {r} has column {c}, outside 0..{}",
                    self.n_cols
                ));
            }
            if cols.windows(2).any(|p| p[1] <= p[0]) {
                return Err(format!(
                    "row {r}'s columns are not strictly ascending; a repeated column has no \
                     single meaning between the reference's dense and sparse operands"
                ));
            }
        }
        if let Some(i) = self.data.iter().position(|v| !v.is_finite()) {
            return Err(format!("value {i} is {}, not finite", self.data[i]));
        }
        Ok(())
    }
}

/// One document's sorted columns and normalised values, appended to `cols`/`vals`.
/// `counts` is a zeroed `dim`-length scratch array and is left zeroed.
fn hash_doc(
    spec: NGramSpec,
    doc: &str,
    counts: &mut [u32],
    touched: &mut Vec<u32>,
    bounds: &mut Vec<usize>,
    cols: &mut Vec<u32>,
    vals: &mut Vec<f64>,
) -> Result<(), String> {
    let mask = u64::from(spec.dim - 1);
    bounds.clear();
    bounds.extend(doc.char_indices().map(|(at, _)| at));
    let chars = bounds.len();
    bounds.push(doc.len());
    let bytes = doc.as_bytes();
    let (n_min, n_max) = (spec.n_min as usize, spec.n_max as usize);
    touched.clear();
    for j in 0..chars {
        let mut h = FNV_OFFSET;
        // Grams starting at j, one code point longer each step; FNV-1a streams, so the hash of
        // a gram is the hash of its prefix continued over its last code point.
        for len in 1..=n_max.min(chars - j) {
            h = fnv1a(h, &bytes[bounds[j + len - 1]..bounds[j + len]]);
            if len >= n_min {
                let col = (h & mask) as usize;
                if counts[col] == 0 {
                    touched.push(col as u32);
                }
                counts[col] = counts[col]
                    .checked_add(1)
                    .ok_or("a column count overflowed u32")?;
            }
        }
    }
    touched.sort_unstable();
    let mut sumsq: u128 = 0;
    for &c in touched.iter() {
        let n = u128::from(counts[c as usize]);
        sumsq += n * n;
    }
    if sumsq > EXACT_SUM_BOUND {
        return Err(format!(
            "a {}-byte document's squared counts sum to {sumsq}, above 2**53, where the \
             reference's float64 norm depends on numpy's summation order",
            doc.len()
        ));
    }
    // The reference: norm = float(np.sqrt(np.sum(vals * vals))); vals / norm when norm > 0.
    // A document with any gram has norm >= 1.
    let norm = (sumsq as f64).sqrt();
    for &c in touched.iter() {
        cols.push(c);
        vals.push(f64::from(counts[c as usize]) / norm);
        counts[c as usize] = 0;
    }
    Ok(())
}

/// `CharNGramHasher(n_min, n_max, dim).transform(docs)`, on up to `threads` threads. The
/// result does not depend on `threads`: each document is hashed alone and the rows are joined
/// in input order.
pub fn transform(spec: NGramSpec, docs: &[&str], threads: usize) -> Result<Csr, String> {
    let n = docs.len();
    let threads = threads.clamp(1, n.max(1));
    // More chunks than threads, taken in turn: documents differ in length and the Mac's cores
    // differ in speed, so a fixed split would leave fast threads idle.
    let chunk = n.div_ceil(threads * 8).max(1);
    let starts: Vec<usize> = (0..n).step_by(chunk).collect();
    let next = Mutex::new(0usize);
    type Part = (Vec<usize>, Vec<u32>, Vec<f64>);
    let parts: Vec<Mutex<Option<Result<Part, String>>>> =
        starts.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut counts = vec![0u32; spec.dim as usize];
                let (mut touched, mut bounds) = (Vec::new(), Vec::new());
                loop {
                    let k = {
                        let mut g = next.lock().unwrap_or_else(|p| p.into_inner());
                        let k = *g;
                        *g += 1;
                        k
                    };
                    let Some(&start) = starts.get(k) else { break };
                    let end = (start + chunk).min(n);
                    let mut lens = Vec::with_capacity(end - start);
                    let (mut cols, mut vals) = (Vec::new(), Vec::new());
                    let mut outcome = Ok(());
                    for doc in &docs[start..end] {
                        let before = cols.len();
                        outcome = hash_doc(
                            spec,
                            doc,
                            &mut counts,
                            &mut touched,
                            &mut bounds,
                            &mut cols,
                            &mut vals,
                        );
                        if outcome.is_err() {
                            break;
                        }
                        lens.push(cols.len() - before);
                    }
                    let result = outcome.map(|()| (lens, cols, vals));
                    *parts[k].lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
                }
            });
        }
    });
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0usize);
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for part in parts {
        let (lens, cols, vals) = part
            .into_inner()
            .unwrap_or_else(|p| p.into_inner())
            .ok_or("a hashing thread stopped before its chunk was done")??;
        for len in lens {
            let last = *indptr.last().unwrap_or(&0);
            indptr.push(last + len);
        }
        indices.extend_from_slice(&cols);
        data.extend_from_slice(&vals);
    }
    let csr = Csr {
        n_cols: spec.dim as usize,
        indptr,
        indices,
        data,
    };
    if csr.n_rows() != n {
        return Err(format!("hashed {} rows of {n} documents", csr.n_rows()));
    }
    Ok(csr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> NGramSpec {
        NGramSpec::new(3, 5, 1 << 16).expect("valid")
    }

    /// The reference, transliterated: n outermost, every gram hashed from scratch.
    fn reference(spec: NGramSpec, doc: &str) -> Vec<(u32, f64)> {
        let chars: Vec<char> = doc.chars().collect();
        let mut counts = std::collections::BTreeMap::<u32, f64>::new();
        for n in spec.n_min as usize..=spec.n_max as usize {
            if chars.len() < n {
                continue;
            }
            for j in 0..=chars.len() - n {
                let gram: String = chars[j..j + n].iter().collect();
                let col = (fnv1a(FNV_OFFSET, gram.as_bytes()) & u64::from(spec.dim - 1)) as u32;
                *counts.entry(col).or_insert(0.0) += 1.0;
            }
        }
        let norm = counts.values().map(|v| v * v).sum::<f64>().sqrt();
        counts.into_iter().map(|(c, v)| (c, v / norm)).collect()
    }

    #[test]
    fn fnv1a_matches_the_published_vectors() {
        // FNV-1a 64 of "" and "a" (http://www.isthe.com/chongo/tech/comp/fnv/).
        assert_eq!(fnv1a(FNV_OFFSET, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(FNV_OFFSET, b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn edge_documents_hash_like_the_transliterated_reference() {
        let docs = [
            "",
            "ab",
            "abc",
            "abcde",
            "aaaaaaaaaaaaaaaa",
            "crlf\r\nline\r\n",
            "nul\0inside\0",
            "e\u{301}\u{301} combining",
            "astral \u{1F9EA}\u{1F9EC}\u{10FFFF} plane",
            "\u{FFFD}\u{FEFF}bom",
        ];
        let csr = transform(spec(), &docs, 3).expect("hashes");
        for (i, doc) in docs.iter().enumerate() {
            let (cols, vals) = csr.row(i);
            let got: Vec<(u32, f64)> = cols.iter().copied().zip(vals.iter().copied()).collect();
            assert_eq!(got, reference(spec(), doc), "{doc:?}");
        }
        assert_eq!(csr.row(0).0.len(), 0, "an empty document is an empty row");
        assert_eq!(csr.row(1).0.len(), 0, "shorter than n_min is an empty row");
    }

    #[test]
    fn hashing_does_not_depend_on_the_thread_count() {
        let owned: Vec<String> = (0..203)
            .map(|i| format!("doc {i} {}", "x\u{e9}y ".repeat(i % 17)))
            .collect();
        let docs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let one = transform(spec(), &docs, 1).expect("hashes");
        one.validate().expect("valid csr");
        for threads in [2, 3, 7, 64, 1000] {
            assert_eq!(transform(spec(), &docs, threads).expect("hashes"), one);
        }
    }

    #[test]
    fn invalid_parameters_are_refused_as_the_reference_refuses_them() {
        assert!(NGramSpec::new(0, 5, 1 << 16).is_err());
        assert!(NGramSpec::new(4, 3, 1 << 16).is_err());
        assert!(NGramSpec::new(3, 5, 8).is_err());
        assert!(NGramSpec::new(3, 5, 1000).is_err());
        assert!(NGramSpec::new(1, 1, 16).is_ok());
    }
}
