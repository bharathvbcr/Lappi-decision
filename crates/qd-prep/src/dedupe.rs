//! `qd_data.dedupe.dedupe`'s MinHash path: sign, band, confirm, cluster, keep.
//!
//! The reference takes every searched content unit (rows grouped by `identity_key` and the
//! digest of their text), signs its shingle set, bands the signatures
//! (`qd_data.minhash.candidate_pairs`, with v6's agreement prefilter when one is set), confirms
//! each candidate with an exact Jaccard over the two shingle sets, and unions every confirmed
//! pair whose units sit in different repos -- a pair inside one repo is counted, never merged,
//! because one repo is one side of the split. In each connected component one unit survives:
//!
//! - rule 0, `lexical` (v5): the bytewise-smallest unit key;
//! - rule 1, `split_priority` (v6, GAP-QD-DATA-DEDUPE-KEEP-RULE-LEXICAL-SPLIT-BLIND-2026-10-03):
//!   the unit of the most protected split (held-out 2, val 1, train 0), then the smallest key.
//!
//! This module does exactly that over the same units, reusing [`crate::wire`]'s shingle sets
//! and [`crate::minhash`]'s signing (one embedded `QDPMHIN1` request) and [`crate::lsh`]'s
//! banding. Shingling, a unit's split rank (`qd_data.dedupe.unit_split_ranks`) and the
//! exact-content path stay in Python, which owns them; this computes what the reference
//! computes from them. Python is the oracle: `python/tests/test_qd_prep_dedupe_parity.py`.
//!
//! Request (`QDPDDIN1`), little-endian throughout:
//!
//! ```text
//! magic 8 | threshold f64 | bands u32 | rows u32 | min_agreement_permille u32 | keep_rule u32
//!         | max_pairs u64 | n_units u64
//!         | n_units x u32 key lengths | the keys' bytes, concatenated
//!         | n_units x u32 repo lengths | the repos' bytes, concatenated
//!         | n_units x u8 split ranks
//!         | a complete QDPMHIN1 request (see `crate::wire`) whose document i is unit i
//! ```
//!
//! Reply (`QDPDDOK1`):
//!
//! ```text
//! magic 8 | truncated u8 | n_candidate_pairs u64 | n_cross_repo_pairs u64
//!         | n_within_repo_pairs u64 | n_clusters u64
//!         | per cluster: kept u32 | n_dropped u32 | n_dropped x u32 | min_edge_jaccard f64
//! ```
//!
//! Clusters come in the reference's order (by their smallest key), each dropped list in key
//! order. A truncated candidate search still clusters the pairs it found, as the reference
//! does before it reports `NotRun`; the flag says so.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use crate::lsh;
use crate::wire::{self, Cursor};

pub const INPUT_MAGIC: &[u8; 8] = b"QDPDDIN1";
pub const OUTPUT_MAGIC: &[u8; 8] = b"QDPDDOK1";
/// The embedded MinHash request's bound plus room for the keys and repos.
pub const MAX_INPUT_BYTES: u64 = wire::MAX_INPUT_BYTES + (1 << 30);
/// `qd_data.config.DEDUPE_KEEP_LEXICAL`.
pub const KEEP_LEXICAL: u32 = 0;
/// `qd_data.config.DEDUPE_KEEP_SPLIT_PRIORITY`.
pub const KEEP_SPLIT_PRIORITY: u32 = 1;
/// The index of `heldout` in `qd_data.config.SPLITS`.
pub const MAX_SPLIT_RANK: u8 = 2;

/// A parsed request.
#[derive(Debug)]
pub struct Request<'b> {
    pub threshold: f64,
    pub bands: usize,
    pub rows: usize,
    pub min_agreement_permille: u32,
    pub keep_rule: u32,
    pub max_pairs: u64,
    pub keys: Vec<&'b [u8]>,
    pub repos: Vec<&'b [u8]>,
    pub ranks: Vec<u8>,
    pub minhash: wire::Request<'b>,
}

/// One connected component of more than one unit.
#[derive(Debug, PartialEq)]
pub struct Cluster {
    pub kept: u32,
    pub dropped: Vec<u32>,
    pub min_edge_jaccard: f64,
}

/// What the MinHash path decided.
#[derive(Debug, PartialEq)]
pub struct Outcome {
    pub truncated: bool,
    pub n_candidate_pairs: u64,
    pub n_cross_repo_pairs: u64,
    pub n_within_repo_pairs: u64,
    pub clusters: Vec<Cluster>,
}

/// `n` byte strings: `n` u32 lengths, then their bytes concatenated.
fn byte_strings<'b>(
    c: &mut Cursor<'b>,
    buf: &'b [u8],
    n: usize,
    (lengths_what, bytes_what): (&str, &str),
) -> Result<Vec<&'b [u8]>, String> {
    let lengths = c.take(
        n.checked_mul(4).ok_or("the length count overflows")?,
        lengths_what,
    )?;
    let mut out = Vec::with_capacity(n);
    for len in buf[lengths].as_chunks::<4>().0 {
        let range = c.take(u32::from_le_bytes(*len) as usize, bytes_what)?;
        out.push(&buf[range]);
    }
    Ok(out)
}

/// Parse and validate a request. Nothing is signed until all of it has been read.
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
    let threshold = f64::from_bits(c.u64("threshold")?);
    if !(threshold > 0.0 && threshold <= 1.0) {
        return Err(format!("threshold {threshold} is outside (0, 1]"));
    }
    let bands = u64::from(c.u32("bands")?);
    let rows = u64::from(c.u32("rows")?);
    if bands > lsh::MAX_BANDED_VALUES
        || rows > lsh::MAX_BANDED_VALUES
        || bands * rows > lsh::MAX_BANDED_VALUES
    {
        return Err(format!(
            "{bands} bands of {rows} rows: each, and their product, must be at most {}",
            lsh::MAX_BANDED_VALUES
        ));
    }
    let min_agreement_permille = c.u32("min_agreement_permille")?;
    if min_agreement_permille > lsh::MAX_AGREEMENT_PERMILLE {
        return Err(format!(
            "min_agreement_permille {min_agreement_permille}; the bound is {}",
            lsh::MAX_AGREEMENT_PERMILLE
        ));
    }
    let keep_rule = c.u32("keep_rule")?;
    if keep_rule != KEEP_LEXICAL && keep_rule != KEEP_SPLIT_PRIORITY {
        return Err(format!(
            "keep_rule {keep_rule} is neither {KEEP_LEXICAL} (lexical) nor \
             {KEEP_SPLIT_PRIORITY} (split_priority)"
        ));
    }
    let max_pairs = c.u64("max_pairs")?;
    if max_pairs > lsh::MAX_PAIRS {
        return Err(format!(
            "max_pairs {max_pairs}; the bound is {}",
            lsh::MAX_PAIRS
        ));
    }
    let n_units = c.u64("n_units")?;
    if n_units > wire::MAX_DOCS {
        return Err(format!("{n_units} units; the bound is {}", wire::MAX_DOCS));
    }
    let n = n_units as usize;
    let keys = byte_strings(&mut c, buf, n, ("key lengths", "key bytes"))?;
    let repos = byte_strings(&mut c, buf, n, ("repo lengths", "repo bytes"))?;
    let ranks = buf[c.take(n, "split ranks")?].to_vec();
    if let Some((i, r)) = ranks.iter().enumerate().find(|(_, r)| **r > MAX_SPLIT_RANK) {
        return Err(format!(
            "unit {i} has split rank {r}; the ranks are 0..={MAX_SPLIT_RANK}"
        ));
    }
    let mut unique: HashSet<&[u8]> = HashSet::with_capacity(n);
    for &key in &keys {
        if !unique.insert(key) {
            return Err(format!(
                "unit key {:?} appears twice; a content unit is one key",
                String::from_utf8_lossy(key)
            ));
        }
    }
    let minhash = wire::parse(&buf[c.at..])?;
    if minhash.docs.len() != n {
        return Err(format!(
            "{n} units but the MinHash request holds {} shingle sets",
            minhash.docs.len()
        ));
    }
    if (bands * rows) as usize > minhash.family.num_perm() {
        return Err(format!(
            "{bands} bands of {rows} rows read past the {} permutations signed",
            minhash.family.num_perm()
        ));
    }
    Ok(Request {
        threshold,
        bands: bands as usize,
        rows: rows as usize,
        min_agreement_permille,
        keep_rule,
        max_pairs,
        keys,
        repos,
        ranks,
        minhash,
    })
}

/// A disjoint-set forest over unit indices; which member survives is the keep rule's.
struct UnionFind(Vec<u32>);

impl UnionFind {
    fn find(&mut self, x: u32) -> u32 {
        let mut root = x;
        while self.0[root as usize] != root {
            root = self.0[root as usize];
        }
        let mut at = x;
        while self.0[at as usize] != root {
            let next = self.0[at as usize];
            self.0[at as usize] = root;
            at = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra.max(rb) as usize] = ra.min(rb);
        }
    }
}

impl Request<'_> {
    /// The MinHash path's decisions. The result does not depend on `threads`.
    pub fn run(&self, threads: usize) -> Result<Outcome, String> {
        let n = self.keys.len();
        let num_perm = self.minhash.family.num_perm();
        let width = self.bands * self.rows;
        let signatures = self.minhash.sign(threads)?;
        let mut banded = Vec::with_capacity(n * width);
        for sig in signatures.chunks_exact(num_perm) {
            banded.extend_from_slice(&sig[..width]);
        }
        let banding = lsh::Request {
            bands: self.bands,
            rows: self.rows,
            max_pairs: self.max_pairs,
            keys: self.keys.clone(),
            values: banded,
            min_agreement_permille: self.min_agreement_permille,
        };
        let (pairs, truncated) = banding.candidate_pairs(threads)?;
        let sets: Vec<HashSet<&[u8]>> = self
            .minhash
            .docs
            .iter()
            .map(|doc| self.minhash.shingles(doc).collect())
            .collect();

        let mut forest = UnionFind((0..n as u32).collect());
        let mut edges: Vec<(u32, u32, f64)> = Vec::new();
        let mut n_within = 0u64;
        for &(a, b) in &pairs {
            let (sa, sb) = (&sets[a as usize], &sets[b as usize]);
            let (small, large) = if sa.len() <= sb.len() {
                (sa, sb)
            } else {
                (sb, sa)
            };
            let inter = small.iter().filter(|s| large.contains(*s)).count();
            let union = sa.len() + sb.len() - inter;
            // `exact_jaccard`: |A & B| / |A | B|, both exact integers, one correctly rounded
            // division; two empty sets are 0.0 (unreachable: an empty set cannot be signed).
            let j = if union == 0 {
                0.0
            } else {
                inter as f64 / union as f64
            };
            if j < self.threshold {
                continue;
            }
            if self.repos[a as usize] == self.repos[b as usize] {
                n_within += 1;
                continue;
            }
            edges.push((a, b, j));
            forest.union(a, b);
        }

        let mut components: HashMap<u32, Vec<u32>> = HashMap::new();
        for unit in 0..n as u32 {
            let root = forest.find(unit);
            components.entry(root).or_default().push(unit);
        }
        let mut min_edge: HashMap<u32, f64> = HashMap::new();
        for &(a, _, j) in &edges {
            let root = forest.find(a);
            let slot = min_edge.entry(root).or_insert(j);
            if j < *slot {
                *slot = j;
            }
        }
        let mut clusters: Vec<(Vec<u32>, f64)> = components
            .into_iter()
            .filter(|(_, members)| members.len() > 1)
            .map(|(root, mut members)| {
                members.sort_by(|x, y| self.keys[*x as usize].cmp(self.keys[*y as usize]));
                (members, min_edge.get(&root).copied().unwrap_or(0.0))
            })
            .collect();
        clusters.sort_by(|(x, _), (y, _)| self.keys[x[0] as usize].cmp(self.keys[y[0] as usize]));
        let clusters = clusters
            .into_iter()
            .map(|(members, min_edge_jaccard)| {
                let kept = if self.keep_rule == KEEP_SPLIT_PRIORITY {
                    *members
                        .iter()
                        .min_by_key(|u| {
                            (Reverse(self.ranks[**u as usize]), self.keys[**u as usize])
                        })
                        .expect("a cluster has members")
                } else {
                    members[0]
                };
                Cluster {
                    kept,
                    dropped: members.into_iter().filter(|u| *u != kept).collect(),
                    min_edge_jaccard,
                }
            })
            .collect();
        Ok(Outcome {
            truncated,
            n_candidate_pairs: pairs.len() as u64,
            n_cross_repo_pairs: edges.len() as u64,
            n_within_repo_pairs: n_within,
            clusters,
        })
    }
}

/// The reply's bytes.
pub fn encode_output(outcome: &Outcome) -> Vec<u8> {
    let mut out = OUTPUT_MAGIC.to_vec();
    out.push(u8::from(outcome.truncated));
    for v in [
        outcome.n_candidate_pairs,
        outcome.n_cross_repo_pairs,
        outcome.n_within_repo_pairs,
        outcome.clusters.len() as u64,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for c in &outcome.clusters {
        out.extend_from_slice(&c.kept.to_le_bytes());
        out.extend_from_slice(&(c.dropped.len() as u32).to_le_bytes());
        for d in &c.dropped {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&c.min_edge_jaccard.to_bits().to_le_bytes());
    }
    out
}

/// `qd-prep dedupe`: request bytes in, reply bytes and a summary line out.
pub fn run_dedupe(buf: &[u8], threads: usize) -> Result<(Vec<u8>, String), String> {
    let request = parse(buf)?;
    let outcome = request.run(threads)?;
    let dropped: usize = outcome.clusters.iter().map(|c| c.dropped.len()).sum();
    Ok((
        encode_output(&outcome),
        format!(
            "qd-prep dedupe: {} units, {} candidate pairs{}, {} cross-repo and {} within-repo \
             confirmed, {} clusters dropping {dropped} units ({} keep) on {threads} thread(s)",
            request.keys.len(),
            outcome.n_candidate_pairs,
            if outcome.truncated {
                " (truncated)"
            } else {
                ""
            },
            outcome.n_cross_repo_pairs,
            outcome.n_within_repo_pairs,
            outcome.clusters.len(),
            if request.keep_rule == KEEP_SPLIT_PRIORITY {
                "split_priority"
            } else {
                "lexical"
            },
        ),
    ))
}
