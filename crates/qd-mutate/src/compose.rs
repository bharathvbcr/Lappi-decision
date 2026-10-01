//! `compose` — PR-shaped `code.defect_class` rows, built from a corpus's own files.
//!
//! The v3 corpus supplies no defect row past about 1.1k tokens, because `bigcode/commitpackft`
//! itself holds no long file: measured over the four local dumps, the largest post-image is
//! 4,428 characters and the largest 3-context diff 1,011 Qwen tokens. A defect deep in a long
//! diff cannot be found in that source at any size filter. This module builds the long rows out
//! of what the source does have: many small real commits, composed into one multi-file unified
//! diff the way a pull request reads.
//!
//! # What a composed row is
//!
//! `min_files` to `max_files` files, each rendered as
//!
//! ```text
//! diff --git a/<path> b/<path>
//! --- a/<path>
//! +++ b/<path>
//! <that file's hunks>
//! ```
//!
//! in an order drawn uniformly at random, so the needle file's index is uniform over the PR and
//! the defect sits at every depth by construction.
//!
//! # Files and their two renderings
//!
//! The unit is a **file**: one pool record, one real commit. Every file in the universe has two
//! kinds of rendering:
//!
//! * **pristine** — the commit, unmodified: `normalize` + [`diffspan::unified_multi`] from the
//!   pre-image, exactly the path `generate.rs` renders a clean example by. [`prepare`] proves
//!   the path by re-rendering **every** stored clean example of the base corpus from its pool
//!   record and refusing the run on one byte of difference.
//! * **mutated** — one of the generator's mutated examples of that record: from the base corpus
//!   (v3) for files it mutated, and from an *auxiliary* corpus — the same generator, unchanged,
//!   run at `--clean-permille 0` over every file the base never mutated (see [`aux_pool`]) —
//!   for the rest. A file with no mutated rendering (no function body, no site inside a hunk)
//!   is not in the universe.
//!
//! # Symmetric roles: every file is the needle exactly once
//!
//! A mutated row holds one needle (rendered mutated) and fillers (rendered pristine); a clean
//! row is all pristine. If fillers were drawn from files that are never needles, file
//! *novelty* would predict the needle in training — a filler is seen many times, a needle once
//! — and that cue is absent at test, where every file is new. So the universe is exactly the
//! set of needle files: each mutated row is assigned a distinct needle file, and fillers are
//! drawn only from those same files. Every file is therefore the needle once and pristine in
//! its other appearances, and its needle rate is `1 / appearances`, the same order for every
//! file. Fillers are drawn least-used-first under a floor: no file passes `soft_max_uses`
//! until every file has `floor_uses`, and `max_filler_uses` is the hard stop. Diagnostic 1 —
//! no file with five or more appearances has needle rate 0 or 1 — is checked on the output and
//! refuses the run.
//!
//! This drops, **within train and for the composed family only**, the generator's rule that a
//! function never appears both mutated and clean. That rule's stated purpose (`generate.rs`,
//! "Split safety") is that both forms of a function land on the same side of the split; that
//! still holds, because both forms of a file are its repo's, and the repo's split decides.
//! `qd_train.mutate_adapter.contrastive_pairs` already keeps both forms on one side. The base
//! corpus's own single-file rows are unchanged.
//!
//! # The invariants, and where each is enforced
//!
//! * **Splits.** The repo-level split is owned by Python (`qd_data.split.assign_repo`) and is not
//!   re-implemented here, for the reason `noul_rows/allowlist.rs` gives: a second implementation
//!   is the copy that drifts. A Python tool writes a [`SplitMap`] — every pool repo to `train`,
//!   `val` or `heldout` — pinned to the pool's sha256. A repo missing from it refuses the run; it
//!   is never defaulted. A row is composed from one split only; `heldout` repos are never
//!   composed from. The loader (`qd_data.defect_class`) re-derives every constituent's split at
//!   the training run's own config and refuses the corpus on any mix.
//! * **No exact twin across a split boundary.** A file any of whose renderings also occurs in a
//!   repo of another split (forks carry identical commits) is excluded. Near-duplicates that are
//!   not byte-identical are not detected here, and that residual is reported as unverified.
//! * **No row is a near-duplicate of one of its parts.** No constituent may carry more than
//!   `max_share_permille` of the row's estimate. Two rows with the same constituent set are
//!   refused.
//!
//! # Length
//!
//! Lengths are spread by choosing how many files to compose. This crate has no tokenizer (the
//! `tokenizers` crate is approved for the Metal backend only), so lengths are budgeted with
//! [`estimate_tokens`], a pre-tokenizer piece count scaled by [`TOKENS_PER_PIECE`]. The estimate
//! is a budget, not a measurement: the pipeline's exact count is the length of record, and the
//! hard width cap is enforced there (`--max-seq-len`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diffspan;
use crate::generate::DIFF_CONTEXT;
use crate::lang::LangId;
use crate::manifest::sha256_hex;
use crate::normalize;
use crate::ops::MutationClass;
use crate::pool::{self, FunctionIdentity, PoolRecord};
use crate::span::LineSpan;

/// What [`SplitMap::schema`] must say.
pub const SPLIT_MAP_SCHEMA: &str = "qd-compose-split-map/v1";
/// What a composed corpus's manifest says it is.
pub const MANIFEST_SCHEMA: &str = "qd-compose/v1";
/// Header lines in front of every file's hunks: `diff --git`, `---`, `+++`.
pub const FILE_HEADER_LINES: u32 = 3;
/// Exact-token count per pre-tokenizer piece, measured 2026-10-01 over sums of 8 random
/// `commitpackft-corpus-v3` diffs against the Qwen3.5-2B-Base tokenizer: median 0.902, p0.1
/// 0.829, p99.9 0.975 (20,000 draws). The budget uses the median; the pipeline measures.
pub const TOKENS_PER_PIECE: f64 = 0.902;
/// Appearances from which diagnostic 1 judges a file's needle rate.
pub const DIAGNOSTIC_MIN_APPEARANCES: u64 = 5;
/// A letter run longer than this counts one extra piece per this many letters: BPE splits long
/// identifiers.
const LETTER_RUN_PIECE: usize = 8;
/// The same for runs of punctuation.
const PUNCT_RUN_PIECE: usize = 3;
/// Draws per filler slot, of which the least-used eligible one is taken.
const DRAWS_PER_SLOT: usize = 24;
/// Attempts at one row before the run is refused as unable to meet its specification.
const ATTEMPTS_PER_ROW: usize = 400;
/// Why both renderings of a file may appear in the composed family, carried in the manifest.
pub const SYMMETRIC_ROLES_REASON: &str = "Every composed file is the needle (rendered mutated) in \
exactly one row and pristine in its other appearances, so file novelty does not predict the \
needle. This drops, within train and for the composed family only, generate.rs's rule that a \
function never appears both mutated and clean. That rule's stated purpose (generate.rs, 'Split \
safety') is that both forms land on one side of the split, which still holds: both forms of a \
file are its repo's, and assign_repo splits by repo. qd_train.mutate_adapter.contrastive_pairs \
already keeps both forms on one side. The base corpus's single-file rows are unchanged \
(Fable round I, 2026-10-01).";

#[derive(Debug, thiserror::Error)]
pub enum ComposeError {
    #[error("input: {0}")]
    Input(String),
    #[error(
        "{total} {what} name a repo the split map does not hold (first: {first:?}); the split is \
         never defaulted"
    )]
    SplitMissing {
        what: &'static str,
        total: usize,
        first: Vec<String>,
    },
    #[error(
        "{mismatches} of {checked} stored clean diffs do not re-render byte for byte from their \
         pool record (first: {first:?}); a pristine file rendered by this path would not be the \
         shape the corpus renders"
    )]
    Parity {
        checked: u64,
        mismatches: u64,
        first: Vec<String>,
    },
    #[error("invariant: {0}")]
    Invariant(String),
    #[error(
        "{split}: composed {made} of {wanted} rows; {detail} (refusals so far: {refusals:?})"
    )]
    Shortfall {
        split: &'static str,
        made: usize,
        wanted: usize,
        detail: String,
        refusals: BTreeMap<String, u64>,
    },
}

/// The repo-level split, as the canonical Python function assigns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Heldout,
}

impl Split {
    pub fn parse(raw: &str) -> Option<Split> {
        match raw {
            "train" => Some(Split::Train),
            "val" => Some(Split::Val),
            "heldout" => Some(Split::Heldout),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Heldout => "heldout",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitParams {
    pub seed: u64,
    pub train_fraction: f64,
    pub val_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitMapPool {
    /// File name of the pool, re-anchored under `data/pool/` by the loader.
    pub file: String,
    pub sha256: String,
    pub records: u64,
}

/// Every pool repo's split, written by `tools/compose_split_map.py` through
/// `qd_data.split.assign_repo`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitMap {
    pub schema: String,
    pub split: SplitParams,
    pub pool: SplitMapPool,
    pub repos: BTreeMap<String, String>,
}

impl SplitMap {
    /// Refuse a map that could not have come from the canonical step.
    pub fn check(&self) -> Result<(), ComposeError> {
        if self.schema != SPLIT_MAP_SCHEMA {
            return Err(ComposeError::Input(format!(
                "split map schema {:?}, expected {SPLIT_MAP_SCHEMA:?}",
                self.schema
            )));
        }
        let p = &self.split;
        if !(p.train_fraction > 0.0
            && p.val_fraction > 0.0
            && p.train_fraction + p.val_fraction < 1.0)
        {
            return Err(ComposeError::Input(format!(
                "split fractions train={} val={} leave no held-out repos",
                p.train_fraction, p.val_fraction
            )));
        }
        if self.pool.sha256.len() != 64 || !self.pool.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(ComposeError::Input(format!(
                "split map pool sha256 {:?} is not a sha256",
                self.pool.sha256
            )));
        }
        if let Some((repo, bad)) = self.repos.iter().find(|(_, s)| Split::parse(s).is_none()) {
            return Err(ComposeError::Input(format!(
                "split map gives repo {repo:?} the split {bad:?}, which is not train, val or heldout"
            )));
        }
        Ok(())
    }

    fn split_of(&self, repo: &str) -> Option<Split> {
        self.repos.get(repo).and_then(|s| Split::parse(s))
    }
}

/// The fields of a corpus row this module reads. Everything else in the row is ignored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusRow {
    pub id: String,
    pub pool_id: String,
    pub repo: String,
    pub path: String,
    pub language: LangId,
    pub class: MutationClass,
    #[serde(default)]
    pub operator: Option<String>,
    pub silent: bool,
    #[serde(default)]
    pub span: Option<LineSpan>,
    pub function: FunctionIdentity,
    pub after: String,
    pub diff: String,
    pub hunk_constrained: bool,
}

/// A corpus a composed corpus reads, named and pinned in its manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusRef {
    /// The corpus directory's name, re-anchored under `data/pool/` by the loader.
    pub name: String,
    pub examples_sha256: String,
}

/// The pool, as a corpus manifest records it. The loader's licence join reads this key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRef {
    pub path: String,
    pub sha256: String,
    pub records: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitMapRef {
    pub file: String,
    pub sha256: String,
}

/// A corpus read from disk and checked against its manifest.
pub struct LoadedCorpus {
    pub rows: Vec<CorpusRow>,
    pub reference: CorpusRef,
    pub manifest: serde_json::Value,
}

/// Everything [`prepare`] needs, already read and checked against its recorded digests.
pub struct Inputs {
    pub base: LoadedCorpus,
    /// The generator's mutations of the files the base never mutated (see [`aux_pool`]).
    pub aux: Option<LoadedCorpus>,
    pub pool: Vec<PoolRecord>,
    pub split_map: SplitMap,
    pub pool_ref: PoolRef,
    pub split_map_ref: SplitMapRef,
}

fn read_file(p: &Path) -> Result<Vec<u8>, ComposeError> {
    std::fs::read(p).map_err(|e| ComposeError::Input(format!("reading {}: {e}", p.display())))
}

fn name_of(p: &Path) -> Result<String, ComposeError> {
    p.file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .ok_or_else(|| ComposeError::Input(format!("{} has no file name", p.display())))
}

/// Read one corpus directory (`examples.jsonl` + `manifest.json`), checking its digest and count.
pub fn load_corpus(dir: &Path) -> Result<LoadedCorpus, ComposeError> {
    let manifest: serde_json::Value = serde_json::from_slice(&read_file(&dir.join("manifest.json"))?)
        .map_err(|e| ComposeError::Input(format!("{}: manifest: {e}", dir.display())))?;
    let examples_sha = manifest
        .get("examples_sha256")
        .and_then(|x| x.as_str())
        .ok_or_else(|| ComposeError::Input(format!("{}: manifest has no examples_sha256", dir.display())))?
        .to_string();
    let bytes = read_file(&dir.join("examples.jsonl"))?;
    let actual = sha256_hex(&bytes);
    if actual != examples_sha {
        return Err(ComposeError::Input(format!(
            "{}/examples.jsonl hashes to {actual}, its manifest records {examples_sha}",
            dir.display()
        )));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| ComposeError::Input(format!("{}: not UTF-8: {e}", dir.display())))?;
    let mut rows = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str::<CorpusRow>(line).map_err(|e| {
            ComposeError::Input(format!("{}/examples.jsonl:{}: {e}", dir.display(), lineno + 1))
        })?);
    }
    if let Some(n) = manifest.get("totals").and_then(|t| t.get("examples")).and_then(|x| x.as_u64())
        && n != rows.len() as u64
    {
        return Err(ComposeError::Input(format!(
            "{} holds {} rows, its manifest records {n}",
            dir.display(),
            rows.len()
        )));
    }
    Ok(LoadedCorpus {
        rows,
        reference: CorpusRef {
            name: name_of(dir)?,
            examples_sha256: examples_sha,
        },
        manifest,
    })
}

fn manifest_pool(manifest: &serde_json::Value, what: &str) -> Result<(String, u64), ComposeError> {
    let pool = manifest
        .get("pool")
        .ok_or_else(|| ComposeError::Input(format!("{what} manifest has no pool")))?;
    let sha = pool
        .get("sha256")
        .and_then(|x| x.as_str())
        .ok_or_else(|| ComposeError::Input(format!("{what} manifest pool has no sha256")))?;
    let records = pool
        .get("records")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| ComposeError::Input(format!("{what} manifest pool has no records")))?;
    Ok((sha.to_string(), records))
}

/// Read the base corpus, its pool, a split map and, optionally, the auxiliary corpus, refusing
/// any digest that does not hold.
pub fn load(
    corpus_dir: &Path,
    pool_path: &Path,
    split_map_path: &Path,
    aux_dir: Option<&Path>,
) -> Result<Inputs, ComposeError> {
    let base = load_corpus(corpus_dir)?;
    let (pool_sha, pool_records) = manifest_pool(&base.manifest, "corpus")?;
    let pool_bytes = read_file(pool_path)?;
    let actual_pool = sha256_hex(&pool_bytes);
    if actual_pool != pool_sha {
        return Err(ComposeError::Input(format!(
            "{} hashes to {actual_pool}, the corpus manifest records {pool_sha}: the corpus was \
             generated from some other pool",
            pool_path.display()
        )));
    }
    let pool = pool::read_jsonl(pool_bytes.as_slice())
        .map_err(|e| ComposeError::Input(format!("pool: {e}")))?;
    if pool.len() as u64 != pool_records {
        return Err(ComposeError::Input(format!(
            "pool holds {} records, the corpus manifest records {pool_records}",
            pool.len()
        )));
    }
    let map_bytes = read_file(split_map_path)?;
    let split_map: SplitMap = serde_json::from_slice(&map_bytes)
        .map_err(|e| ComposeError::Input(format!("split map: {e}")))?;
    split_map.check()?;
    if split_map.pool.sha256 != pool_sha || split_map.pool.records != pool_records {
        return Err(ComposeError::Input(format!(
            "the split map was computed over pool {} ({} records), not this pool {} ({})",
            split_map.pool.sha256, split_map.pool.records, pool_sha, pool_records
        )));
    }
    let aux = aux_dir.map(load_corpus).transpose()?;
    Ok(Inputs {
        base,
        aux,
        pool,
        split_map,
        pool_ref: PoolRef {
            path: pool_path.display().to_string(),
            sha256: pool_sha,
            records: pool_records,
        },
        split_map_ref: SplitMapRef {
            file: name_of(split_map_path)?,
            sha256: sha256_hex(&map_bytes),
        },
    })
}

/// The records the base corpus never mutated, in `train` or `val`, in a language the generator
/// reads, with a change to render: the pool the auxiliary corpus is generated from.
///
/// Returned as the JSONL the generator reads and its records. Deterministic, so [`prepare`] can
/// re-derive it and refuse an auxiliary corpus generated from any other bytes.
pub fn aux_pool(inputs: &Inputs) -> Result<(String, Vec<&PoolRecord>), ComposeError> {
    let mutated = mutated_files(&inputs.base.rows);
    let mut out = String::new();
    let mut records = Vec::new();
    for record in &inputs.pool {
        let split = inputs.split_map.split_of(&record.repo).ok_or_else(|| {
            ComposeError::SplitMissing {
                what: "pool records",
                total: 1,
                first: vec![record.repo.clone()],
            }
        })?;
        if split == Split::Heldout
            || record.language().is_none()
            || mutated.contains(&(record.repo.as_str(), record.path.as_str()))
            || record.prior_source.is_none()
        {
            continue;
        }
        out.push_str(
            &serde_json::to_string(record)
                .map_err(|e| ComposeError::Input(format!("serialising {}: {e}", record.id)))?,
        );
        out.push('\n');
        records.push(record);
    }
    Ok((out, records))
}

fn mutated_files(rows: &[CorpusRow]) -> HashSet<(&str, &str)> {
    rows.iter()
        .filter(|r| r.class != MutationClass::Clean)
        .map(|r| (r.repo.as_str(), r.path.as_str()))
        .collect()
}

// -- the token estimate ------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharClass {
    Newline,
    Space,
    Letter,
    Digit,
    Other,
}

fn class_of(c: char) -> CharClass {
    if c == '\n' || c == '\r' {
        CharClass::Newline
    } else if c.is_whitespace() {
        CharClass::Space
    } else if c.is_alphabetic() {
        CharClass::Letter
    } else if c.is_numeric() {
        CharClass::Digit
    } else {
        CharClass::Other
    }
}

/// Pieces a byte-level BPE pre-tokenizer of the Qwen kind splits `text` into, approximately:
/// a letter run (plus one per [`LETTER_RUN_PIECE`] letters past the first), one per digit, one
/// per newline run, one per whitespace run except a single space that prefixes a word or a
/// symbol, and a punctuation run (plus one per [`PUNCT_RUN_PIECE`] past the first).
pub fn proxy_pieces(text: &str) -> u64 {
    let chars: Vec<CharClass> = text.chars().map(class_of).collect();
    let n = chars.len();
    let mut pieces = 0u64;
    let mut i = 0usize;
    let run_end = |from: usize, class: CharClass| {
        let mut j = from;
        while j < n && chars[j] == class {
            j += 1;
        }
        j
    };
    while i < n {
        match chars[i] {
            CharClass::Letter => {
                let j = run_end(i, CharClass::Letter);
                pieces += 1 + ((j - i - 1) / LETTER_RUN_PIECE) as u64;
                i = j;
            }
            CharClass::Digit => {
                pieces += 1;
                i += 1;
            }
            CharClass::Newline => {
                i = run_end(i, CharClass::Newline);
                pieces += 1;
            }
            CharClass::Space => {
                let j = run_end(i, CharClass::Space);
                if j - i == 1 && j < n && matches!(chars[j], CharClass::Letter | CharClass::Other) {
                    // One space attaches to the piece after it.
                    i = j;
                    continue;
                }
                pieces += 1;
                i = j;
            }
            CharClass::Other => {
                let j = run_end(i, CharClass::Other);
                pieces += 1 + ((j - i - 1) / PUNCT_RUN_PIECE) as u64;
                i = j;
            }
        }
    }
    pieces
}

/// The length budget for `text`, in estimated Qwen3.5 tokens. See [`TOKENS_PER_PIECE`].
pub fn estimate_tokens(text: &str) -> u32 {
    let est = (proxy_pieces(text) as f64 * TOKENS_PER_PIECE).round();
    if est >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        est as u32
    }
}

// -- the span walk -----------------------------------------------------------------------------

/// Why a span could not be carried into a diff's line coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanRefusal {
    /// The line exists in the post-image but no hunk represents it.
    OutsideDiff,
    /// The diff is empty.
    EmptyDiff,
    /// A line the format does not allow, or a span whose end maps before its start.
    Malformed(String),
}

impl SpanRefusal {
    pub fn code(&self) -> &'static str {
        match self {
            SpanRefusal::OutsideDiff => "needle_span_outside_diff",
            SpanRefusal::EmptyDiff => "needle_span_empty_diff",
            SpanRefusal::Malformed(_) => "needle_span_malformed_diff",
        }
    }
}

/// The new-side start of a hunk header `@@ -a[,b] +c[,d] @@`, or `None`.
fn new_side_start(line: &str) -> Option<u64> {
    let rest = line.strip_prefix("@@ -")?;
    let digits = |s: &str| s.bytes().take_while(u8::is_ascii_digit).count();
    let n = digits(rest);
    if n == 0 {
        return None;
    }
    let mut rest = &rest[n..];
    if let Some(r) = rest.strip_prefix(',') {
        let m = digits(r);
        if m == 0 {
            return None;
        }
        rest = &r[m..];
    }
    let rest = rest.strip_prefix(" +")?;
    let n = digits(rest);
    if n == 0 {
        return None;
    }
    let start: u64 = rest[..n].parse().ok()?;
    let mut tail = &rest[n..];
    if let Some(r) = tail.strip_prefix(',') {
        let m = digits(r);
        if m == 0 {
            return None;
        }
        tail = &r[m..];
    }
    tail.starts_with(" @@").then_some(start)
}

/// The 1-based line of `diff` that represents post-image line `after_line`.
///
/// A port of `qd_train.mutate_adapter.diff_offset_of_after_line`, the walk the loader rebases
/// every uncomposed span with, and exactly as strict: a hunk header sets the new-side counter,
/// `+` and context lines consume one, `-` lines none, an empty line is allowed only after the
/// final newline, and anything else refuses. `tests/compose.rs` holds the parity cases.
pub fn diff_line_of_after_line(diff: &str, after_line: u32) -> Result<u32, SpanRefusal> {
    if after_line < 1 {
        return Err(SpanRefusal::Malformed(format!(
            "after_line is 1-based, got {after_line}"
        )));
    }
    if diff.trim().is_empty() {
        return Err(SpanRefusal::EmptyDiff);
    }
    let mut offset = 0usize;
    let mut new_line: Option<u64> = None;
    for (index, line) in diff.split('\n').enumerate() {
        let width = line.len() + 1;
        match line.as_bytes().first() {
            Some(b'@') => {
                new_line = Some(new_side_start(line).ok_or_else(|| {
                    SpanRefusal::Malformed(format!("unparseable hunk header {line:?}"))
                })?);
            }
            Some(b'+') | Some(b' ') => {
                let Some(current) = new_line else {
                    return Err(SpanRefusal::Malformed(format!(
                        "diff line {line:?} precedes any hunk header"
                    )));
                };
                if current == u64::from(after_line) {
                    return u32::try_from(index + 1)
                        .map_err(|_| SpanRefusal::Malformed("diff too long".to_string()));
                }
                new_line = Some(current + 1);
            }
            Some(b'-') => {
                if new_line.is_none() {
                    return Err(SpanRefusal::Malformed(format!(
                        "diff line {line:?} precedes any hunk header"
                    )));
                }
            }
            None => {
                if offset < diff.len() {
                    return Err(SpanRefusal::Malformed(
                        "a blank line inside a unified diff".to_string(),
                    ));
                }
            }
            Some(_) => {
                return Err(SpanRefusal::Malformed(format!(
                    "diff line {line:?} does not begin with '+', '-', ' ' or '@'"
                )));
            }
        }
        offset += width;
    }
    Err(SpanRefusal::OutsideDiff)
}

/// A post-image span, as 1-based inclusive lines of `diff`. Both ends are walked separately.
pub fn diff_line_span(diff: &str, span: LineSpan) -> Result<LineSpan, SpanRefusal> {
    let start = diff_line_of_after_line(diff, span.start)?;
    let end = diff_line_of_after_line(diff, span.end)?;
    if end < start {
        return Err(SpanRefusal::Malformed(format!(
            "after lines {}..{} map to diff lines {start}..{end}",
            span.start, span.end
        )));
    }
    Ok(LineSpan::new(start, end))
}

// -- files -------------------------------------------------------------------------------------

/// Which corpus a mutated rendering came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Base,
    Aux,
}

/// One mutated rendering of a file: a generator example, its span carried into its own diff.
#[derive(Debug, Clone)]
pub struct Rendering {
    pub row: CorpusRow,
    pub source: Source,
    /// The span in the rendering's own diff's line coordinates.
    pub local_span: LineSpan,
    /// The estimate of the rendering's block (headers included).
    pub est: u32,
}

/// One real commit with its pristine rendering and at least one mutated one.
#[derive(Debug, Clone)]
pub struct FileUnit {
    pub pool_id: String,
    pub repo: String,
    pub path: String,
    pub language: LangId,
    /// The commit, unmodified.
    pub pristine: String,
    /// The estimate of the pristine block (headers included).
    pub pristine_est: u32,
    /// The base corpus's clean example of this record, when the pristine form is one.
    pub pristine_example_id: Option<String>,
    /// The normalized post-image: a clean row's `after` when this file anchors it.
    pub after: String,
    pub renderings: Vec<Rendering>,
}

/// Per-split file universes, after every exclusion.
pub struct Prepared {
    pub units: BTreeMap<Split, Vec<FileUnit>>,
    pub refusals: BTreeMap<String, u64>,
    pub parity_checked: u64,
    pub heldout_skipped: u64,
    pub base: CorpusRef,
    pub aux: Option<CorpusRef>,
    pub aux_pool_sha256: Option<String>,
    pub pool_ref: PoolRef,
    pub split_map_ref: SplitMapRef,
    pub split_params: SplitParams,
}

fn bump(map: &mut BTreeMap<String, u64>, key: &str) {
    *map.entry(key.to_string()).or_insert(0) += 1;
}

fn digest(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}

fn count_lines(diff: &str) -> u32 {
    u32::try_from(diff.bytes().filter(|&b| b == b'\n').count()).unwrap_or(u32::MAX)
}

/// The estimate of one file's block as a row renders it, headers included.
fn block_estimate(path: &str, diff: &str) -> u32 {
    let mut block = String::with_capacity(diff.len() + 4 * path.len() + 32);
    render_file(&mut block, path, diff);
    estimate_tokens(&block)
}

fn path_is_renderable(path: &str) -> bool {
    !path.is_empty() && !path.chars().any(char::is_control)
}

/// Render one pool record's commit pristine, the way `generate.rs` renders a clean example
/// under `DiffShape::MultiHunk`: `(diff, normalized post-image)`, or `None` when the record
/// holds no change to read.
pub fn render_pristine(
    record: &PoolRecord,
) -> Result<Option<(String, String)>, diffspan::DiffRefusal> {
    let (source, _) = normalize::normalize(&record.source);
    let Some(prior_raw) = record.prior_source.as_deref() else {
        return Ok(None);
    };
    let (prior, _) = normalize::normalize(prior_raw);
    if prior == source {
        return Ok(None);
    }
    let diff = diffspan::unified_multi(&prior, &source, DIFF_CONTEXT)?;
    Ok(Some((diff, source)))
}

/// Build the per-split file universes, checking every invariant the module docstring names.
pub fn prepare(inputs: &Inputs) -> Result<Prepared, ComposeError> {
    let map = &inputs.split_map;
    let mut refusals: BTreeMap<String, u64> = BTreeMap::new();

    // Every repo every input names must be in the map. Collected, then refused together.
    let mut missing: BTreeSet<String> = BTreeSet::new();
    let aux_rows: &[CorpusRow] = inputs.aux.as_ref().map(|a| a.rows.as_slice()).unwrap_or(&[]);
    for r in inputs.base.rows.iter().chain(aux_rows) {
        if map.split_of(&r.repo).is_none() {
            missing.insert(r.repo.clone());
        }
    }
    for r in &inputs.pool {
        if map.split_of(&r.repo).is_none() {
            missing.insert(r.repo.clone());
        }
    }
    if !missing.is_empty() {
        return Err(ComposeError::SplitMissing {
            what: "corpus rows or pool records",
            total: missing.len(),
            first: missing.into_iter().take(5).collect(),
        });
    }
    let split_of = |repo: &str| map.split_of(repo).expect("checked above");

    let pool_by_id: HashMap<&str, &PoolRecord> =
        inputs.pool.iter().map(|r| (r.id.as_str(), r)).collect();
    if pool_by_id.len() != inputs.pool.len() {
        return Err(ComposeError::Input("the pool holds a duplicate id".to_string()));
    }

    // The auxiliary corpus must have been generated, mutated-only, from exactly `aux_pool`.
    let mut aux_pool_sha256 = None;
    if let Some(aux) = &inputs.aux {
        let (jsonl, records) = aux_pool(inputs)?;
        let expected = sha256_hex(jsonl.as_bytes());
        let (sha, n) = manifest_pool(&aux.manifest, "aux corpus")?;
        if sha != expected || n != records.len() as u64 {
            return Err(ComposeError::Input(format!(
                "the auxiliary corpus was generated from pool {sha} ({n} records), not from this \
                 corpus's never-mutated records {expected} ({})",
                records.len()
            )));
        }
        let clean = aux.manifest.get("totals").and_then(|t| t.get("clean")).and_then(|x| x.as_u64());
        let renderer = aux
            .manifest
            .get("diff")
            .and_then(|d| d.get("renderer"))
            .and_then(|x| x.as_str());
        if clean != Some(0) || renderer != Some("multi_hunk") {
            return Err(ComposeError::Input(format!(
                "the auxiliary corpus must be generated with --clean-permille 0 in the multi-hunk \
                 shape; its manifest records clean={clean:?} renderer={renderer:?}"
            )));
        }
        aux_pool_sha256 = Some(expected);
    }

    // Parity: every stored clean diff re-renders byte for byte from its pool record.
    let mut parity_checked = 0u64;
    let mut parity_bad: Vec<String> = Vec::new();
    let mut clean_example_of: HashMap<&str, &CorpusRow> = HashMap::new();
    for r in inputs.base.rows.iter().filter(|r| r.class == MutationClass::Clean) {
        let record = pool_by_id.get(r.pool_id.as_str()).ok_or_else(|| {
            ComposeError::Input(format!(
                "clean example {} names pool id {} the pool lacks",
                r.id, r.pool_id
            ))
        })?;
        parity_checked += 1;
        match render_pristine(record) {
            Ok(Some((diff, _))) if diff == r.diff => {}
            _ => parity_bad.push(r.id.clone()),
        }
        clean_example_of.insert(r.pool_id.as_str(), r);
    }
    if !parity_bad.is_empty() {
        return Err(ComposeError::Parity {
            checked: parity_checked,
            mismatches: parity_bad.len() as u64,
            first: parity_bad.into_iter().take(5).collect(),
        });
    }

    // Mutated renderings by pool id.
    let mut renderings_of: HashMap<&str, Vec<(&CorpusRow, Source)>> = HashMap::new();
    for (rows, source) in [(inputs.base.rows.as_slice(), Source::Base), (aux_rows, Source::Aux)] {
        for r in rows.iter().filter(|r| r.class != MutationClass::Clean) {
            if !pool_by_id.contains_key(r.pool_id.as_str()) {
                return Err(ComposeError::Input(format!(
                    "mutated example {} names pool id {} the pool lacks",
                    r.id, r.pool_id
                )));
            }
            if r.span.is_none() {
                return Err(ComposeError::Input(format!("mutated example {} carries no span", r.id)));
            }
            renderings_of.entry(r.pool_id.as_str()).or_default().push((r, source));
        }
    }
    if aux_rows.iter().any(|r| r.class == MutationClass::Clean) {
        return Err(ComposeError::Input("the auxiliary corpus holds a clean example".to_string()));
    }

    // Every diff text in every split, for the exact-twin exclusion: every corpus row, and every
    // record's pristine rendering.
    let mut splits_of_diff: HashMap<[u8; 32], BTreeSet<Split>> = HashMap::new();
    for r in inputs.base.rows.iter().chain(aux_rows) {
        splits_of_diff.entry(digest(&r.diff)).or_default().insert(split_of(&r.repo));
    }
    let mut pristine_of: HashMap<&str, (String, String)> = HashMap::new();
    for record in &inputs.pool {
        if record.language().is_none() {
            continue;
        }
        match render_pristine(record) {
            Ok(Some((diff, after))) => {
                splits_of_diff.entry(digest(&diff)).or_default().insert(split_of(&record.repo));
                pristine_of.insert(record.id.as_str(), (diff, after));
            }
            Ok(None) => {}
            Err(_) => bump(&mut refusals, "pristine_diff_refused"),
        }
    }
    let crosses = |text: &str| splits_of_diff.get(&digest(text)).is_some_and(|s| s.len() > 1);

    let mut units: BTreeMap<Split, Vec<FileUnit>> = BTreeMap::new();
    let mut seen_pristine: HashSet<(Split, [u8; 32])> = HashSet::new();
    let mut heldout_skipped = 0u64;
    for record in &inputs.pool {
        let split = split_of(&record.repo);
        let Some(language) = record.language() else {
            bump(&mut refusals, "file_language_unrecognised");
            continue;
        };
        if split == Split::Heldout {
            heldout_skipped += 1;
            continue;
        }
        if !path_is_renderable(&record.path) {
            bump(&mut refusals, "file_path_unrenderable");
            continue;
        }
        let Some((pristine, after)) = pristine_of.get(record.id.as_str()) else {
            bump(&mut refusals, "file_no_pristine_change");
            continue;
        };
        if pristine.trim().is_empty() || !pristine.ends_with('\n') {
            bump(&mut refusals, "file_pristine_empty_or_unterminated");
            continue;
        }
        let Some(candidates) = renderings_of.get(record.id.as_str()) else {
            bump(&mut refusals, "file_no_mutated_rendering");
            continue;
        };
        if crosses(pristine) || candidates.iter().any(|(r, _)| crosses(&r.diff)) {
            bump(&mut refusals, "file_exact_twin_across_splits");
            continue;
        }
        if !seen_pristine.insert((split, digest(pristine))) {
            bump(&mut refusals, "file_duplicate_pristine_in_split");
            continue;
        }
        let mut renderings = Vec::new();
        for &(r, source) in candidates {
            if r.path != record.path || r.repo != record.repo || !r.diff.ends_with('\n') {
                bump(&mut refusals, "rendering_disagrees_with_its_record");
                continue;
            }
            let span = r.span.expect("checked above");
            match diff_line_span(&r.diff, span) {
                Ok(local_span) => renderings.push(Rendering {
                    row: r.clone(),
                    source,
                    local_span,
                    est: block_estimate(&r.path, &r.diff),
                }),
                Err(refusal) => bump(&mut refusals, refusal.code()),
            }
        }
        if renderings.is_empty() {
            bump(&mut refusals, "file_no_usable_mutated_rendering");
            continue;
        }
        units.entry(split).or_default().push(FileUnit {
            pool_id: record.id.clone(),
            repo: record.repo.clone(),
            path: record.path.clone(),
            language,
            pristine_est: block_estimate(&record.path, pristine),
            pristine: pristine.clone(),
            pristine_example_id: clean_example_of.get(record.id.as_str()).map(|r| r.id.clone()),
            after: after.clone(),
            renderings,
        });
    }

    Ok(Prepared {
        units,
        refusals,
        parity_checked,
        heldout_skipped,
        base: inputs.base.reference.clone(),
        aux: inputs.aux.as_ref().map(|a| a.reference.clone()),
        aux_pool_sha256,
        pool_ref: inputs.pool_ref.clone(),
        split_map_ref: inputs.split_map_ref.clone(),
        split_params: inputs.split_map.split.clone(),
    })
}

// -- composition -------------------------------------------------------------------------------

/// How a run is configured. Every field lands in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Options {
    pub seed: u64,
    pub train_rows: usize,
    pub val_rows: usize,
    /// A row's target estimated tokens is drawn uniformly from `[min_tokens, max_tokens]`.
    pub min_tokens: u32,
    pub max_tokens: u32,
    /// No row's estimate may exceed this, whatever its target.
    pub hard_max_tokens: u32,
    pub min_files: usize,
    pub max_files: usize,
    /// Rows in a thousand composed `clean`.
    pub clean_permille: u32,
    /// No constituent may carry more than this share of its row's estimate.
    pub max_share_permille: u32,
    /// A row's estimate must land within this many permille of its target.
    pub tolerance_permille: u32,
    /// The JSONL row bound the loader enforces (`DataConfig.max_row_bytes`).
    pub max_row_bytes: usize,
    /// Until every file has this many appearances, none may pass `soft_max_uses`.
    pub floor_uses: u64,
    pub soft_max_uses: u64,
    /// No file appears in more rows than this.
    pub max_filler_uses: u64,
}

impl Options {
    pub fn check(&self) -> Result<(), ComposeError> {
        let bad = |m: String| Err(ComposeError::Input(m));
        if self.min_files < 2 || self.min_files > self.max_files {
            return bad(format!(
                "files {}..{} is not a range of at least two",
                self.min_files, self.max_files
            ));
        }
        if self.min_tokens == 0
            || self.min_tokens > self.max_tokens
            || self.max_tokens > self.hard_max_tokens
        {
            return bad(format!(
                "tokens {}..{} (hard max {}) is not an ordered range",
                self.min_tokens, self.max_tokens, self.hard_max_tokens
            ));
        }
        if self.clean_permille > 1000 || self.max_share_permille == 0 || self.max_share_permille > 1000
        {
            return bad("permille out of range".to_string());
        }
        if self.tolerance_permille == 0 || self.tolerance_permille >= 1000 {
            return bad("tolerance_permille must be in (0, 1000)".to_string());
        }
        if self.floor_uses == 0
            || self.floor_uses > self.soft_max_uses
            || self.soft_max_uses > self.max_filler_uses
        {
            return bad(format!(
                "uses floor {} <= soft max {} <= hard max {} does not hold",
                self.floor_uses, self.soft_max_uses, self.max_filler_uses
            ));
        }
        if self.train_rows + self.val_rows == 0 {
            return bad("no rows requested".to_string());
        }
        Ok(())
    }
}

/// One constituent of a composed row, in the order it appears in the diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constituent {
    pub pool_id: String,
    pub repo: String,
    pub path: String,
    pub language: LangId,
    /// `needle` (rendered mutated) or `filler` (rendered pristine).
    pub role: String,
    /// The generator example this block is, when it is one: the needle's mutated example, or a
    /// filler's base-corpus clean example.
    pub example_id: Option<String>,
    /// The corpus `example_id` is from.
    pub example_source: Option<Source>,
    /// First and last line of this file's block in the composed diff, header included.
    pub first_line: u32,
    pub last_line: u32,
    pub est_tokens: u32,
}

/// One composed row: the shape `qd_train.mutate_adapter.parse_example` reads, plus the fields
/// `qd_data.defect_class` needs to load it without re-walking the diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposedRow {
    pub id: String,
    /// The anchor's pool id: the needle's, or on a clean row its first file in pool order.
    pub pool_id: String,
    pub repo: String,
    /// The repository: a composed row has no single file, and the rendered context's `file:`
    /// header names the anchor repo.
    pub path: String,
    pub language: LangId,
    pub class: MutationClass,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    pub silent: bool,
    /// The needle's span over the needle file's mutated post-image (`after`). Absent on a clean
    /// row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<LineSpan>,
    pub function: FunctionIdentity,
    pub after: String,
    pub diff: String,
    pub hunk_constrained: bool,
    pub detail: String,
    pub seed: u64,
    pub tool_version: String,
    pub composed: bool,
    pub split: Split,
    /// The needle's span in the composed diff's line coordinates, 1-based inclusive. `null` on a
    /// clean row. The loader takes this rather than walking the diff, whose `diff --git` lines
    /// the single-file walk refuses.
    pub diff_span: Option<LineSpan>,
    pub needle_index: Option<u32>,
    pub n_files: u32,
    pub est_tokens: u32,
    pub constituents: Vec<Constituent>,
}

/// Fixed edges, so two runs' histograms compare.
const LENGTH_EDGES: [u32; 9] = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];

fn length_bin(est: u32) -> String {
    for w in LENGTH_EDGES.windows(2) {
        if est >= w[0] && est < w[1] {
            return format!("{:05}-{:05}", w[0], w[1]);
        }
    }
    format!("{:05}+", LENGTH_EDGES[LENGTH_EDGES.len() - 1])
}

fn depth_bucket(fraction: f64) -> &'static str {
    if fraction < 0.2 {
        "0-20%"
    } else if fraction < 0.4 {
        "20-40%"
    } else if fraction < 0.6 {
        "40-60%"
    } else if fraction < 0.8 {
        "60-80%"
    } else {
        "80-100%"
    }
}

/// Diagnostic 1: each file's needle rate over its appearances.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NeedleRateReport {
    pub files: u64,
    pub files_with_min_appearances: u64,
    pub min_appearances: u64,
    /// Over files with at least [`DIAGNOSTIC_MIN_APPEARANCES`] appearances.
    pub rate_min: f64,
    pub rate_median: f64,
    pub rate_max: f64,
    /// Files with at least [`DIAGNOSTIC_MIN_APPEARANCES`] appearances whose needle rate is 0 or 1.
    /// The run is refused unless this is zero.
    pub violations: u64,
    pub appearances_min: u64,
    pub appearances_median: u64,
    pub appearances_max: u64,
    /// Files that ended below `floor_uses` appearances, and their share.
    pub files_below_floor: u64,
    pub share_below_floor: f64,
    pub appearances_histogram: BTreeMap<u64, u64>,
}

/// What one split's composition did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SplitReport {
    pub rows: u64,
    pub by_class: BTreeMap<String, u64>,
    pub files_available: u64,
    pub files_with_base_rendering: u64,
    pub files_with_aux_rendering_only: u64,
    pub files_used: u64,
    pub file_slots: u64,
    pub needle_rate: NeedleRateReport,
    pub needle_sources: BTreeMap<String, u64>,
    pub file_est_histogram: BTreeMap<String, u64>,
    /// The row's estimate, binned.
    pub length_est_histogram: BTreeMap<String, u64>,
    pub files_per_row: BTreeMap<u32, u64>,
    /// `needle_index / (n_files - 1)`, binned, over mutated rows.
    pub needle_index_depth: BTreeMap<String, u64>,
    /// The needle span's first line over the diff's lines, binned, over mutated rows.
    pub needle_line_depth: BTreeMap<String, u64>,
    pub languages_needle: BTreeMap<String, u64>,
    pub languages_filler_slots: BTreeMap<String, u64>,
    /// Mutated rows whose span covers a `-` line of the needle's diff (the base corpus's rebase
    /// allows it).
    pub spans_with_interior_removed_lines: u64,
    pub est_tokens_total: u64,
    pub attempts_refused: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenEstimate {
    pub method: String,
    pub tokens_per_piece: f64,
    pub calibration: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComposeTotals {
    pub examples: u64,
    pub by_split: BTreeMap<String, u64>,
    pub by_class: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComposeManifest {
    pub schema: String,
    pub tool_version: String,
    pub seed: u64,
    pub seed_derivation: String,
    /// `train_only`, `val_only` or `train_and_val`: which splits this corpus holds rows of.
    pub mode: String,
    pub base_corpus: CorpusRef,
    pub aux_corpus: Option<CorpusRef>,
    pub aux_pool_sha256: Option<String>,
    pub pool: PoolRef,
    pub split_map: SplitMapRef,
    pub split_params: SplitParams,
    pub options: Options,
    pub symmetric_roles: String,
    pub token_estimate: TokenEstimate,
    pub examples_sha256: String,
    pub totals: ComposeTotals,
    pub prepare_refusals: BTreeMap<String, u64>,
    pub heldout_records_skipped: u64,
    pub clean_diff_parity_checked: u64,
    pub splits: BTreeMap<String, SplitReport>,
}

pub struct Composed {
    pub rows: Vec<ComposedRow>,
    pub manifest: ComposeManifest,
}

impl Composed {
    /// The rows as JSONL, and its digest.
    pub fn to_jsonl(&self) -> Result<(String, String), serde_json::Error> {
        let mut out = String::new();
        for row in &self.rows {
            out.push_str(&serde_json::to_string(row)?);
            out.push('\n');
        }
        let digest = sha256_hex(out.as_bytes());
        Ok((out, digest))
    }
}

fn split_rng(seed: u64, split: Split) -> ChaCha20Rng {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(b"compose\0");
    h.update(split.as_str().as_bytes());
    ChaCha20Rng::from_seed(h.finalize().into())
}

fn render_file(out: &mut String, path: &str, diff: &str) {
    out.push_str("diff --git a/");
    out.push_str(path);
    out.push_str(" b/");
    out.push_str(path);
    out.push_str("\n--- a/");
    out.push_str(path);
    out.push_str("\n+++ b/");
    out.push_str(path);
    out.push('\n');
    out.push_str(diff);
}

fn shuffle<T>(items: &mut [T], rng: &mut ChaCha20Rng) {
    for i in (1..items.len()).rev() {
        let j = rng.random_range(0..=i);
        items.swap(i, j);
    }
}

/// The pool a split's fillers are drawn from: the needle files, by pristine estimate.
struct FillerPool<'a> {
    units: &'a [FileUnit],
    /// Unit indices sorted by `(pristine_est, pool_id)`, and their estimates in that order.
    order: Vec<usize>,
    sizes: Vec<u32>,
    /// Appearances so far, the needle appearance included; indexed by unit.
    uses: Vec<u64>,
    in_pool: Vec<bool>,
    /// Pool files still below the floor; while any is left, no file passes the soft maximum.
    below_floor: u64,
    floor: u64,
    soft_max: u64,
    hard_max: u64,
}

impl<'a> FillerPool<'a> {
    fn new(units: &'a [FileUnit], members: &[usize], opts: &Options) -> Self {
        let mut order: Vec<usize> = members.to_vec();
        order.sort_by(|&a, &b| {
            units[a]
                .pristine_est
                .cmp(&units[b].pristine_est)
                .then_with(|| units[a].pool_id.cmp(&units[b].pool_id))
        });
        let sizes = order.iter().map(|&i| units[i].pristine_est).collect();
        let mut in_pool = vec![false; units.len()];
        for &m in members {
            in_pool[m] = true;
        }
        FillerPool {
            units,
            order,
            sizes,
            uses: vec![0; units.len()],
            in_pool,
            below_floor: members.len() as u64,
            floor: opts.floor_uses,
            soft_max: opts.soft_max_uses,
            hard_max: opts.max_filler_uses,
        }
    }

    fn limit(&self) -> u64 {
        if self.below_floor > 0 { self.soft_max } else { self.hard_max }
    }

    fn record_use(&mut self, unit: usize) {
        self.uses[unit] += 1;
        if self.in_pool[unit] && self.uses[unit] == self.floor {
            self.below_floor -= 1;
        }
    }

    /// A file with pristine estimate in `[lo, hi]`, not in the row and not sharing a path with
    /// it: the least used of [`DRAWS_PER_SLOT`] uniform draws over the window, under the current
    /// use limit. (A scan that also offered every below-floor file in the window was measured
    /// on the 25k build and moved nothing: the files left below the floor are large ones few
    /// windows reach at all.)
    fn pick(
        &self,
        lo: u32,
        hi: u32,
        in_row: &HashSet<usize>,
        paths: &HashSet<&str>,
        rng: &mut ChaCha20Rng,
    ) -> Option<usize> {
        if lo > hi {
            return None;
        }
        let a = self.sizes.partition_point(|&s| s < lo);
        let b = self.sizes.partition_point(|&s| s <= hi);
        if a >= b {
            return None;
        }
        let limit = self.limit();
        let mut best: Option<usize> = None;
        for _ in 0..DRAWS_PER_SLOT {
            let f = self.order[rng.random_range(a..b)];
            if in_row.contains(&f)
                || paths.contains(self.units[f].path.as_str())
                || self.uses[f] >= limit
            {
                continue;
            }
            if best.is_none_or(|b| self.uses[f] < self.uses[b]) {
                best = Some(f);
            }
        }
        best
    }
}

/// A planned row: its class, target, and (on a mutated row) its needle file and rendering.
struct Plan {
    target: u32,
    needle: Option<(usize, usize)>,
}

/// The largest block a row of `target` may plan for: the share bound at the shortest estimate
/// the row may be accepted at, so a block planned under it never fails the share check after
/// rendering.
fn share_cap(target: u32, opts: &Options) -> u32 {
    let floor = u64::from(target) * u64::from(1000 - opts.tolerance_permille) / 1000;
    (floor * u64::from(opts.max_share_permille) / 1000) as u32
}

/// Choose one row's pristine files around its needle. `Err` names why this attempt failed.
fn choose_fillers(
    pool: &FillerPool<'_>,
    units: &[FileUnit],
    plan: &Plan,
    opts: &Options,
    rng: &mut ChaCha20Rng,
) -> Result<Vec<usize>, &'static str> {
    let cap = share_cap(plan.target, opts);
    let mut in_row: HashSet<usize> = HashSet::new();
    let mut paths: HashSet<&str> = HashSet::new();
    let mut remaining = i64::from(plan.target);
    let mut fixed = 0usize;
    if let Some((u, r)) = plan.needle {
        in_row.insert(u);
        paths.insert(units[u].path.as_str());
        remaining -= i64::from(units[u].renderings[r].est);
        fixed = 1;
    }
    let smallest = i64::from(*pool.sizes.first().ok_or("no_files")?);
    let largest = i64::from((*pool.sizes.last().ok_or("no_files")?).min(cap));
    if remaining < smallest || largest < 1 {
        return Err("needle_leaves_no_room");
    }
    let s_lo = ((remaining + largest - 1) / largest).max((opts.min_files - fixed) as i64);
    let s_hi = (remaining / smallest).min((opts.max_files - fixed) as i64);
    if s_lo > s_hi || s_hi < 1 {
        return Err("file_count_infeasible");
    }
    let slots = rng.random_range(s_lo..=s_hi);
    let mut chosen = Vec::with_capacity(slots as usize);
    for j in 0..slots {
        let left = slots - j;
        let desired = remaining / left;
        let (lo, hi) = if left == 1 {
            let tol = i64::from(plan.target) * i64::from(opts.tolerance_permille) / 1000 / 2;
            (remaining - tol, remaining + tol)
        } else {
            (desired * 2 / 3, desired * 3 / 2)
        };
        let lo = lo.max(1);
        let hi = hi.min(i64::from(cap));
        if hi < lo {
            return Err("slot_window_empty");
        }
        let f = pool
            .pick(lo as u32, hi as u32, &in_row, &paths, rng)
            .ok_or("no_file_in_window")?;
        in_row.insert(f);
        paths.insert(units[f].path.as_str());
        remaining -= i64::from(units[f].pristine_est);
        chosen.push(f);
    }
    Ok(chosen)
}

/// A rendered row before its fields are filled in.
struct Rendered {
    diff: String,
    constituents: Vec<Constituent>,
    needle_at: Option<(u32, LineSpan)>,
    est: u32,
}

fn render_row(
    units: &[FileUnit],
    needle: Option<(usize, usize)>,
    fillers: &[usize],
    rng: &mut ChaCha20Rng,
) -> Rendered {
    let mut parts: Vec<(usize, bool)> = fillers.iter().map(|&f| (f, false)).collect();
    if let Some((u, _)) = needle {
        parts.push((u, true));
    }
    shuffle(&mut parts, rng);
    let mut diff = String::new();
    let mut constituents = Vec::with_capacity(parts.len());
    let mut needle_at = None;
    let mut line = 0u32;
    for (index, &(u, is_needle)) in parts.iter().enumerate() {
        let unit = &units[u];
        let first_line = line + 1;
        let (body, example_id, example_source, est) = if is_needle {
            let r = &unit.renderings[needle.expect("a needle part has a rendering").1];
            let span = LineSpan::new(
                first_line + FILE_HEADER_LINES - 1 + r.local_span.start,
                first_line + FILE_HEADER_LINES - 1 + r.local_span.end,
            );
            needle_at = Some((index as u32, span));
            (r.row.diff.as_str(), Some(r.row.id.clone()), Some(r.source), r.est)
        } else {
            let source = unit.pristine_example_id.as_ref().map(|_| Source::Base);
            (unit.pristine.as_str(), unit.pristine_example_id.clone(), source, unit.pristine_est)
        };
        render_file(&mut diff, &unit.path, body);
        line += FILE_HEADER_LINES + count_lines(body);
        constituents.push(Constituent {
            pool_id: unit.pool_id.clone(),
            repo: unit.repo.clone(),
            path: unit.path.clone(),
            language: unit.language,
            role: if is_needle { "needle" } else { "filler" }.to_string(),
            example_id,
            example_source,
            first_line,
            last_line: line,
            est_tokens: est,
        });
    }
    let est = estimate_tokens(&diff);
    Rendered {
        diff,
        constituents,
        needle_at,
        est,
    }
}

/// Assign each mutated row a distinct needle file, smallest share bound first, so a short row
/// is never left with only large files. Within a bound the choice is uniform.
fn assign_needles(
    plans: &mut [Plan],
    mutated_rows: &[usize],
    needle_files: &[usize],
    units: &[FileUnit],
    opts: &Options,
    rng: &mut ChaCha20Rng,
) -> Result<(), String> {
    // Each needle file's rendering is drawn once, uniformly among its renderings.
    let mut pending: Vec<(u32, usize, usize)> = needle_files
        .iter()
        .map(|&u| {
            let r = rng.random_range(0..units[u].renderings.len());
            (units[u].renderings[r].est, u, r)
        })
        .collect();
    pending.sort_unstable();
    let mut rows: Vec<usize> = mutated_rows.to_vec();
    rows.sort_by_key(|&i| (plans[i].target, i));
    for i in rows {
        let cap = share_cap(plans[i].target, opts);
        let fits = pending.partition_point(|&(est, _, _)| est <= cap);
        if fits == 0 {
            return Err(format!(
                "no remaining needle file fits a row of target {} (share bound {cap})",
                plans[i].target
            ));
        }
        let (_, u, r) = pending.remove(rng.random_range(0..fits));
        plans[i].needle = Some((u, r));
    }
    Ok(())
}

/// Compose one split's rows.
fn compose_split(
    split: Split,
    wanted: usize,
    units: &[FileUnit],
    opts: &Options,
) -> Result<(Vec<ComposedRow>, SplitReport), ComposeError> {
    let mut rng = split_rng(opts.seed, split);
    let mut report = SplitReport {
        files_available: units.len() as u64,
        files_with_base_rendering: units
            .iter()
            .filter(|u| u.renderings.iter().any(|r| r.source == Source::Base))
            .count() as u64,
        files_with_aux_rendering_only: units
            .iter()
            .filter(|u| u.renderings.iter().all(|r| r.source == Source::Aux))
            .count() as u64,
        ..SplitReport::default()
    };
    for u in units {
        *report.file_est_histogram.entry(length_bin(u.pristine_est)).or_insert(0) += 1;
    }
    if wanted == 0 {
        return Ok((Vec::new(), report));
    }
    let shortfall = |made: usize, detail: String, refusals: &BTreeMap<String, u64>| {
        ComposeError::Shortfall {
            split: split.as_str(),
            made,
            wanted,
            detail,
            refusals: refusals.clone(),
        }
    };

    // Plan every row first: class and target, drawn once and held through every attempt.
    let mut plans: Vec<Plan> = (0..wanted)
        .map(|_| {
            let clean = rng.random_range(0..1000u32) < opts.clean_permille;
            let target = rng.random_range(opts.min_tokens..=opts.max_tokens);
            (clean, target)
        })
        .map(|(clean, target)| Plan {
            target,
            needle: if clean { None } else { Some((usize::MAX, usize::MAX)) },
        })
        .collect();
    let mutated_rows: Vec<usize> = (0..wanted).filter(|&i| plans[i].needle.is_some()).collect();
    if mutated_rows.is_empty() {
        return Err(shortfall(
            0,
            "no mutated row was planned, so no file is a needle and the filler universe is empty"
                .to_string(),
            &report.attempts_refused,
        ));
    }
    if mutated_rows.len() > units.len() {
        return Err(shortfall(
            0,
            format!(
                "{} mutated rows need as many distinct needle files, and only {} files have a \
                 mutated rendering",
                mutated_rows.len(),
                units.len()
            ),
            &report.attempts_refused,
        ));
    }
    // The universe: as many files as mutated rows, each the needle exactly once.
    let mut all: Vec<usize> = (0..units.len()).collect();
    shuffle(&mut all, &mut rng);
    let needle_files: Vec<usize> = all[..mutated_rows.len()].to_vec();
    assign_needles(&mut plans, &mutated_rows, &needle_files, units, opts, &mut rng)
        .map_err(|d| shortfall(0, d, &report.attempts_refused))?;
    let mut pool = FillerPool::new(units, &needle_files, opts);
    for &u in &needle_files {
        pool.record_use(u);
    }

    let mut rows = Vec::with_capacity(wanted);
    let mut seen_sets: HashSet<[u8; 32]> = HashSet::new();
    for (index, plan) in plans.iter().enumerate() {
        let mut accepted: Option<(Vec<usize>, Rendered)> = None;
        for _ in 0..ATTEMPTS_PER_ROW {
            let fillers = match choose_fillers(&pool, units, plan, opts, &mut rng) {
                Ok(f) => f,
                Err(why) => {
                    bump(&mut report.attempts_refused, why);
                    continue;
                }
            };
            let mut ids: Vec<&str> = fillers.iter().map(|&f| units[f].pool_id.as_str()).collect();
            if let Some((u, _)) = plan.needle {
                ids.push(units[u].pool_id.as_str());
            }
            ids.sort_unstable();
            let set_key: [u8; 32] = Sha256::digest(ids.join("\0").as_bytes()).into();
            if seen_sets.contains(&set_key) {
                bump(&mut report.attempts_refused, "duplicate_constituent_set");
                continue;
            }
            let rendered = render_row(units, plan.needle, &fillers, &mut rng);
            let lo = u64::from(plan.target) * u64::from(1000 - opts.tolerance_permille) / 1000;
            let hi = u64::from(plan.target) * u64::from(1000 + opts.tolerance_permille) / 1000;
            if u64::from(rendered.est) < lo || u64::from(rendered.est) > hi {
                bump(&mut report.attempts_refused, "estimate_off_target");
                continue;
            }
            if rendered.est > opts.hard_max_tokens {
                bump(&mut report.attempts_refused, "over_hard_max_tokens");
                continue;
            }
            let share_cap = u64::from(rendered.est) * u64::from(opts.max_share_permille) / 1000;
            if rendered.constituents.iter().any(|c| u64::from(c.est_tokens) > share_cap) {
                bump(&mut report.attempts_refused, "constituent_over_share");
                continue;
            }
            seen_sets.insert(set_key);
            accepted = Some((fillers, rendered));
            break;
        }
        let Some((fillers, rendered)) = accepted else {
            return Err(shortfall(
                rows.len(),
                format!("row {index} (target {}) failed {ATTEMPTS_PER_ROW} attempts", plan.target),
                &report.attempts_refused,
            ));
        };
        for &f in &fillers {
            pool.record_use(f);
        }
        let row = build_row(split, index, units, plan, rendered, &fillers, opts)?;
        verify_row(&row, units)?;
        note_row(&mut report, &row);
        rows.push(row);
    }

    // Diagnostic 1 is read off the emitted rows, not the sampler's bookkeeping; the two must
    // agree.
    let counts = appearances(&rows);
    for &u in &needle_files {
        let from_rows = counts.get(units[u].pool_id.as_str()).map_or(0, |c| c.0);
        if pool.uses[u] != from_rows {
            return Err(ComposeError::Invariant(format!(
                "{}: the sampler counted {} appearances of {}, the rows hold {from_rows}",
                split.as_str(),
                pool.uses[u],
                units[u].pool_id
            )));
        }
    }
    let universe: Vec<&str> = needle_files.iter().map(|&u| units[u].pool_id.as_str()).collect();
    report.needle_rate = needle_rate_report(&counts, &universe, opts)
        .map_err(|m| ComposeError::Invariant(format!("{}: {m}", split.as_str())))?;
    report.files_used = needle_files.iter().filter(|&&u| pool.uses[u] > 0).count() as u64;
    report.file_slots = needle_files.iter().map(|&u| pool.uses[u]).sum();
    for p in &plans {
        if let Some((u, r)) = p.needle {
            let key = match units[u].renderings[r].source {
                Source::Base => "base",
                Source::Aux => "aux",
            };
            *report.needle_sources.entry(key.to_string()).or_insert(0) += 1;
        }
    }
    if report.needle_rate.violations > 0 {
        return Err(ComposeError::Invariant(format!(
            "{}: diagnostic 1 failed: {} file(s) with >= {DIAGNOSTIC_MIN_APPEARANCES} appearances \
             have needle rate 0 or 1",
            split.as_str(),
            report.needle_rate.violations
        )));
    }
    Ok((rows, report))
}

/// Per pool id: `(appearances, appearances as the needle)` over `rows`.
pub fn appearances(rows: &[ComposedRow]) -> HashMap<&str, (u64, u64)> {
    let mut out: HashMap<&str, (u64, u64)> = HashMap::new();
    for row in rows {
        for c in &row.constituents {
            let e = out.entry(c.pool_id.as_str()).or_insert((0, 0));
            e.0 += 1;
            if c.role == "needle" {
                e.1 += 1;
            }
        }
    }
    out
}

/// Diagnostic 1 over `counts`: refuses a file outside `universe`, and reports each universe
/// file's needle rate. The caller refuses the run on any violation.
pub fn needle_rate_report(
    counts: &HashMap<&str, (u64, u64)>,
    universe: &[&str],
    opts: &Options,
) -> Result<NeedleRateReport, String> {
    let members: HashSet<&str> = universe.iter().copied().collect();
    if let Some(stray) = counts.keys().find(|k| !members.contains(*k)) {
        return Err(format!("{stray} appears in a row but is not in the file universe"));
    }
    let mut out = NeedleRateReport {
        files: universe.len() as u64,
        min_appearances: DIAGNOSTIC_MIN_APPEARANCES,
        ..NeedleRateReport::default()
    };
    let mut rates: Vec<f64> = Vec::new();
    let mut appearances: Vec<u64> = Vec::new();
    for id in universe {
        let (a, needles) = counts.get(id).copied().unwrap_or((0, 0));
        appearances.push(a);
        *out.appearances_histogram.entry(a).or_insert(0) += 1;
        if a < opts.floor_uses {
            out.files_below_floor += 1;
        }
        if a >= DIAGNOSTIC_MIN_APPEARANCES {
            if needles == 0 || needles == a {
                out.violations += 1;
            }
            rates.push(needles as f64 / a as f64);
        }
    }
    out.files_with_min_appearances = rates.len() as u64;
    rates.sort_by(f64::total_cmp);
    appearances.sort_unstable();
    if let (Some(&lo), Some(&hi)) = (rates.first(), rates.last()) {
        out.rate_min = lo;
        out.rate_max = hi;
        out.rate_median = rates[rates.len() / 2];
    }
    if let (Some(&lo), Some(&hi)) = (appearances.first(), appearances.last()) {
        out.appearances_min = lo;
        out.appearances_max = hi;
        out.appearances_median = appearances[appearances.len() / 2];
    }
    out.share_below_floor = if universe.is_empty() {
        0.0
    } else {
        out.files_below_floor as f64 / universe.len() as f64
    };
    Ok(out)
}

fn build_row(
    split: Split,
    index: usize,
    units: &[FileUnit],
    plan: &Plan,
    rendered: Rendered,
    fillers: &[usize],
    opts: &Options,
) -> Result<ComposedRow, ComposeError> {
    let n_files = rendered.constituents.len() as u32;
    let id = format!("compose:{}:{index:06}", split.as_str());
    match (plan.needle, rendered.needle_at) {
        (Some((u, r)), Some((needle_index, diff_span))) => {
            let nd = &units[u].renderings[r].row;
            Ok(ComposedRow {
                id,
                pool_id: nd.pool_id.clone(),
                repo: nd.repo.clone(),
                path: nd.repo.clone(),
                language: nd.language,
                class: nd.class,
                operator: nd.operator.clone(),
                silent: nd.silent,
                span: nd.span,
                function: FunctionIdentity {
                    repo: nd.function.repo.clone(),
                    path: nd.repo.clone(),
                    symbol: nd.function.symbol.clone(),
                    arity: nd.function.arity,
                },
                after: nd.after.clone(),
                diff: rendered.diff,
                hunk_constrained: nd.hunk_constrained,
                detail: format!("{} of {n_files} files; needle {}", needle_index + 1, nd.id),
                seed: opts.seed,
                tool_version: crate::TOOL_VERSION.to_string(),
                composed: true,
                split,
                diff_span: Some(diff_span),
                needle_index: Some(needle_index),
                n_files,
                est_tokens: rendered.est,
                constituents: rendered.constituents,
            })
        }
        (None, None) => {
            // The anchor is the row's first file in pool-unit order, which does not move with
            // the file order.
            let anchor = *fillers
                .iter()
                .min()
                .ok_or_else(|| ComposeError::Invariant(format!("{id}: a clean row holds no file")))?;
            let unit = &units[anchor];
            let function = &unit.renderings[0].row.function;
            Ok(ComposedRow {
                id,
                pool_id: unit.pool_id.clone(),
                repo: unit.repo.clone(),
                path: unit.repo.clone(),
                language: unit.language,
                class: MutationClass::Clean,
                operator: None,
                silent: false,
                span: None,
                function: FunctionIdentity {
                    repo: function.repo.clone(),
                    path: unit.repo.clone(),
                    symbol: function.symbol.clone(),
                    arity: function.arity,
                },
                after: unit.after.clone(),
                diff: rendered.diff,
                hunk_constrained: true,
                detail: format!("{n_files} unmodified commits"),
                seed: opts.seed,
                tool_version: crate::TOOL_VERSION.to_string(),
                composed: true,
                split,
                diff_span: None,
                needle_index: None,
                n_files,
                est_tokens: rendered.est,
                constituents: rendered.constituents,
            })
        }
        _ => Err(ComposeError::Invariant(format!("{id}: the plan and the rendering disagree on the needle"))),
    }
}

/// Every check a row must pass before it is kept, re-derived from the rendered text.
fn verify_row(row: &ComposedRow, units: &[FileUnit]) -> Result<(), ComposeError> {
    let fail = |m: String| Err(ComposeError::Invariant(format!("{}: {m}", row.id)));
    let lines: Vec<&str> = row.diff.split('\n').collect();
    let total = count_lines(&row.diff);
    if lines.len() as u32 != total + 1 || !row.diff.ends_with('\n') {
        return fail("the composed diff does not end in exactly one newline".to_string());
    }
    if row.constituents.len() as u32 != row.n_files {
        return fail("constituent count disagrees with n_files".to_string());
    }
    let mut expected_first = 1u32;
    for c in &row.constituents {
        if c.first_line != expected_first || c.last_line < c.first_line + FILE_HEADER_LINES {
            return fail(format!("block of {} does not tile the diff", c.pool_id));
        }
        let at = |n: u32| lines[(n - 1) as usize];
        if at(c.first_line) != format!("diff --git a/{} b/{}", c.path, c.path)
            || at(c.first_line + 1) != format!("--- a/{}", c.path)
            || at(c.first_line + 2) != format!("+++ b/{}", c.path)
        {
            return fail(format!("block of {} does not open with its file headers", c.pool_id));
        }
        if !at(c.first_line + 3).starts_with("@@ ") {
            return fail(format!("block of {} does not open with a hunk header", c.pool_id));
        }
        expected_first = c.last_line + 1;
    }
    if expected_first != total + 1 {
        return fail("blocks do not cover the diff".to_string());
    }
    let paths: HashSet<&str> = row.constituents.iter().map(|c| c.path.as_str()).collect();
    if paths.len() != row.constituents.len() {
        return fail("two files share a path".to_string());
    }
    let needles: Vec<&Constituent> = row.constituents.iter().filter(|c| c.role == "needle").collect();
    match (row.class, row.diff_span, needles.as_slice()) {
        (MutationClass::Clean, None, []) => Ok(()),
        (MutationClass::Clean, _, _) => fail("a clean row carries a span or a needle".to_string()),
        (_, Some(span), [needle]) => {
            let body_first = needle.first_line + FILE_HEADER_LINES;
            if span.start < body_first || span.end > needle.last_line || span.end < span.start {
                return fail(format!("span {span} leaves the needle's block"));
            }
            for n in [span.start, span.end] {
                let l = lines[(n - 1) as usize];
                if !(l.starts_with('+') || l.starts_with(' ')) {
                    return fail(format!("span endpoint line {n} is {l:?}, not '+' or context"));
                }
            }
            // Within one hunk: inside a block's body, only a hunk header begins with '@'.
            if (span.start..=span.end).any(|n| lines[(n - 1) as usize].starts_with('@')) {
                return fail("span crosses a hunk header".to_string());
            }
            // The text the span points at is the needle rendering's own span text.
            let rendering = units
                .iter()
                .find(|u| u.pool_id == needle.pool_id)
                .and_then(|u| {
                    u.renderings
                        .iter()
                        .find(|r| Some(r.row.id.as_str()) == needle.example_id.as_deref())
                })
                .ok_or_else(|| ComposeError::Invariant(format!("{}: needle not found", row.id)))?;
            let own: Vec<&str> = rendering.row.diff.split('\n').collect();
            for k in 0..=(span.end - span.start) {
                if lines[(span.start + k - 1) as usize]
                    != own[(rendering.local_span.start + k - 1) as usize]
                {
                    return fail("span text differs from the needle's own span text".to_string());
                }
            }
            if row.needle_index.map(|i| row.constituents[i as usize].role.as_str()) != Some("needle")
            {
                return fail("needle_index does not name the needle".to_string());
            }
            Ok(())
        }
        _ => fail("a mutated row must hold exactly one needle and a span".to_string()),
    }
}

fn note_row(report: &mut SplitReport, row: &ComposedRow) {
    report.rows += 1;
    *report.by_class.entry(row.class.as_str().to_string()).or_insert(0) += 1;
    *report.length_est_histogram.entry(length_bin(row.est_tokens)).or_insert(0) += 1;
    *report.files_per_row.entry(row.n_files).or_insert(0) += 1;
    report.est_tokens_total += u64::from(row.est_tokens);
    for c in &row.constituents {
        let lang = c.language.as_str().to_string();
        if c.role == "needle" {
            *report.languages_needle.entry(lang).or_insert(0) += 1;
        } else {
            *report.languages_filler_slots.entry(lang).or_insert(0) += 1;
        }
    }
    if let (Some(index), Some(span)) = (row.needle_index, row.diff_span) {
        let frac = if row.n_files > 1 {
            f64::from(index) / f64::from(row.n_files - 1)
        } else {
            0.0
        };
        *report.needle_index_depth.entry(depth_bucket(frac).to_string()).or_insert(0) += 1;
        let total = count_lines(&row.diff);
        let line_frac = if total > 1 {
            f64::from(span.start - 1) / f64::from(total - 1)
        } else {
            0.0
        };
        *report.needle_line_depth.entry(depth_bucket(line_frac).to_string()).or_insert(0) += 1;
        let lines: Vec<&str> = row.diff.split('\n').collect();
        if (span.start..=span.end).any(|n| lines[(n - 1) as usize].starts_with('-')) {
            report.spans_with_interior_removed_lines += 1;
        }
    }
}

/// Compose both splits. Refuses rather than shortens: a run that cannot meet its specification
/// is an error naming what failed, not a smaller corpus.
pub fn compose(prepared: &Prepared, opts: &Options) -> Result<Composed, ComposeError> {
    opts.check()?;
    let mut rows = Vec::new();
    let mut splits = BTreeMap::new();
    let empty: Vec<FileUnit> = Vec::new();
    for (split, wanted) in [(Split::Train, opts.train_rows), (Split::Val, opts.val_rows)] {
        let units = prepared.units.get(&split).unwrap_or(&empty);
        let (r, report) = compose_split(split, wanted, units, opts)?;
        rows.extend(r);
        splits.insert(split.as_str().to_string(), report);
    }
    let mut by_split = BTreeMap::new();
    let mut by_class = BTreeMap::new();
    for r in &rows {
        *by_split.entry(r.split.as_str().to_string()).or_insert(0) += 1;
        *by_class.entry(r.class.as_str().to_string()).or_insert(0) += 1;
    }
    let mode = match (opts.train_rows > 0, opts.val_rows > 0) {
        (true, true) => "train_and_val",
        (true, false) => "train_only",
        _ => "val_only",
    };
    let mut composed = Composed {
        manifest: ComposeManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            tool_version: crate::TOOL_VERSION.to_string(),
            seed: opts.seed,
            seed_derivation:
                "ChaCha20Rng::from_seed(sha256(seed.to_le_bytes() || b\"compose\\0\" || split))"
                    .to_string(),
            mode: mode.to_string(),
            base_corpus: prepared.base.clone(),
            aux_corpus: prepared.aux.clone(),
            aux_pool_sha256: prepared.aux_pool_sha256.clone(),
            pool: prepared.pool_ref.clone(),
            split_map: prepared.split_map_ref.clone(),
            split_params: prepared.split_params.clone(),
            options: opts.clone(),
            symmetric_roles: SYMMETRIC_ROLES_REASON.to_string(),
            token_estimate: token_estimate(),
            examples_sha256: String::new(),
            totals: ComposeTotals {
                examples: rows.len() as u64,
                by_split,
                by_class,
            },
            prepare_refusals: prepared.refusals.clone(),
            heldout_records_skipped: prepared.heldout_skipped,
            clean_diff_parity_checked: prepared.parity_checked,
            splits,
        },
        rows,
    };
    let (_, digest) = composed
        .to_jsonl()
        .map_err(|e| ComposeError::Input(format!("serialising rows: {e}")))?;
    composed.manifest.examples_sha256 = digest;
    for row in &composed.rows {
        let bytes = serde_json::to_string(row)
            .map(|s| s.len() + 1)
            .map_err(|e| ComposeError::Input(format!("serialising {}: {e}", row.id)))?;
        if bytes > opts.max_row_bytes {
            return Err(ComposeError::Invariant(format!(
                "{}: {bytes} bytes, over the {}-byte row bound the loader enforces",
                row.id, opts.max_row_bytes
            )));
        }
    }
    Ok(composed)
}

fn token_estimate() -> TokenEstimate {
    TokenEstimate {
        method: "pre-tokenizer piece count (compose::proxy_pieces) x tokens_per_piece".to_string(),
        tokens_per_piece: TOKENS_PER_PIECE,
        calibration: "median exact/proxy over 20,000 sums of 8 commitpackft-corpus-v3 diffs, \
                      Qwen3.5-2B-Base tokenizer: p0.1 0.829, p50 0.902, p99.9 0.975 (2026-10-01). \
                      A budget; the pipeline's exact count is the length of record"
            .to_string(),
    }
}

/// The universe a run would draw from, without composing: per split, files and their
/// pristine-estimate histogram.
pub fn census(prepared: &Prepared) -> BTreeMap<String, SplitReport> {
    let mut out = BTreeMap::new();
    for split in [Split::Train, Split::Val] {
        let mut report = SplitReport::default();
        if let Some(units) = prepared.units.get(&split) {
            report.files_available = units.len() as u64;
            report.files_with_base_rendering = units
                .iter()
                .filter(|u| u.renderings.iter().any(|r| r.source == Source::Base))
                .count() as u64;
            report.files_with_aux_rendering_only = units
                .iter()
                .filter(|u| u.renderings.iter().all(|r| r.source == Source::Aux))
                .count() as u64;
            for u in units {
                *report.file_est_histogram.entry(length_bin(u.pristine_est)).or_insert(0) += 1;
            }
        }
        out.insert(split.as_str().to_string(), report);
    }
    out
}
