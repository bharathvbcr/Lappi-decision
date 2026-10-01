//! `compose` — PR-shaped `code.defect_class` rows, built from a corpus's own examples.
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
//! 3 to 12 files, each rendered as
//!
//! ```text
//! diff --git a/<path> b/<path>
//! --- a/<path>
//! +++ b/<path>
//! <that file's hunks, exactly as the corpus rendered them>
//! ```
//!
//! in an order drawn uniformly at random, so the needle file's index is uniform over the PR and
//! the defect sits at every depth by construction.
//!
//! * A **mutated** row holds exactly one *needle*: one of the corpus's mutated examples, its
//!   diff byte for byte (the commit's real hunks with the injected edit among them). Its class
//!   and operator are the needle's, and its span is the needle's span, carried into the composed
//!   diff's line coordinates.
//! * A **clean** row holds only fillers. Its class is `clean` and its span abstains.
//! * Every other file is a *filler*: a real commit, unmodified.
//!
//! # The invariants, and where each is enforced
//!
//! * **At most one injected defect per row.** Fillers come only from files whose `(repo, path)`
//!   has no mutated example anywhere in the corpus, so a filler can never be some other row's
//!   needle, and the generator's structural rule — the same function never appears both mutated
//!   and clean (`generate.rs`, "Split safety") — holds over the corpus and the composed rows
//!   together. [`prepare`] checks it rather than assuming it.
//! * **Fillers render exactly as the corpus renders a clean example.** A filler that is the
//!   corpus's own clean example is used as stored. Any other filler is rendered by the same
//!   `normalize` + [`diffspan::unified_multi`] path, and [`prepare`] proves the path by
//!   re-rendering **every** stored clean example from its pool record and refusing the run on one
//!   byte of difference.
//! * **Splits.** The repo-level split is owned by Python (`qd_data.split.assign_repo`) and is not
//!   re-implemented here, for the reason `noul_rows/allowlist.rs` gives: a second implementation
//!   is the copy that drifts. A Python tool writes a [`SplitMap`] — every pool repo to `train`,
//!   `val` or `heldout` — pinned to the pool's sha256. A repo missing from it refuses the run; it
//!   is never defaulted. A row is composed from one split only; `heldout` repos are never
//!   composed from. The loader (`qd_data.defect_class`) re-derives every constituent's split at
//!   the training run's own config and refuses the corpus on any mix, so a hand-edited map cannot
//!   reach a training process either.
//! * **No exact twin across a split boundary.** A constituent whose diff text also occurs in a
//!   repo of another split (forks carry identical commits) is excluded. Near-duplicates that are
//!   not byte-identical are not detected here; the pipeline's row-level MinHash check cannot see
//!   them inside a long row, and that residual is reported as unverified.
//! * **No row is a near-duplicate of one of its parts.** No constituent may carry more than
//!   `max_share_permille` of the row's estimated tokens, so a short row is never a J >= 0.8
//!   near-duplicate of the corpus row it contains (which the pipeline's dedupe could resolve by
//!   dropping the corpus row). Two rows with the same constituent set are refused.
//!
//! # Length
//!
//! Lengths are spread by choosing how many commits to compose. This crate has no tokenizer (the
//! `tokenizers` crate is approved for the Metal backend only), so lengths are budgeted with
//! [`estimate_tokens`], a pre-tokenizer piece count scaled by [`TOKENS_PER_PIECE`]. The estimate
//! is a budget, not a measurement: the pipeline's exact count is the length of record, and the
//! hard width cap is enforced there.

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
/// A letter run longer than this counts one extra piece per this many letters: BPE splits long
/// identifiers.
const LETTER_RUN_PIECE: usize = 8;
/// The same for runs of punctuation.
const PUNCT_RUN_PIECE: usize = 3;
/// Draws per filler slot, of which the least-used eligible one is taken.
const DRAWS_PER_SLOT: usize = 12;
/// Unused needles scanned past the cursor for one that fits a row's share bound.
const NEEDLE_LOOKAHEAD: usize = 512;
/// Attempts at one row before the run is refused as unable to meet its specification.
const ATTEMPTS_PER_ROW: usize = 400;

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
         pool record (first: {first:?}); a filler rendered by this path would not be the shape \
         the corpus renders"
    )]
    Parity {
        checked: u64,
        mismatches: u64,
        first: Vec<String>,
    },
    #[error("invariant: {0}")]
    Invariant(String),
    #[error(
        "{split}: composed {made} of {wanted} rows; row {made} failed {ATTEMPTS_PER_ROW} attempts \
         (refusals so far: {refusals:?})"
    )]
    Shortfall {
        split: &'static str,
        made: usize,
        wanted: usize,
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

/// The corpus the needles come from, named in the composed manifest so the loader reads it first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseCorpus {
    /// The corpus directory's name, a sibling of the composed corpus under `data/pool/`.
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

/// Everything [`prepare`] needs, already read and checked against its recorded digests.
pub struct Inputs {
    pub corpus_rows: Vec<CorpusRow>,
    pub pool: Vec<PoolRecord>,
    pub split_map: SplitMap,
    pub base: BaseCorpus,
    pub pool_ref: PoolRef,
    pub split_map_ref: SplitMapRef,
}

/// Read a corpus directory, its pool and a split map, refusing any digest that does not hold.
pub fn load(
    corpus_dir: &Path,
    pool_path: &Path,
    split_map_path: &Path,
) -> Result<Inputs, ComposeError> {
    let read = |p: &Path| {
        std::fs::read(p).map_err(|e| ComposeError::Input(format!("reading {}: {e}", p.display())))
    };
    let manifest_bytes = read(&corpus_dir.join("manifest.json"))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| ComposeError::Input(format!("corpus manifest: {e}")))?;
    let field = |v: &serde_json::Value, k: &str| -> Result<String, ComposeError> {
        v.get(k)
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .ok_or_else(|| ComposeError::Input(format!("corpus manifest has no string {k:?}")))
    };
    let examples_sha = field(&manifest, "examples_sha256")?;
    let pool_meta = manifest
        .get("pool")
        .ok_or_else(|| ComposeError::Input("corpus manifest has no \"pool\"".to_string()))?;
    let pool_sha = field(pool_meta, "sha256")?;
    let pool_records = pool_meta
        .get("records")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| ComposeError::Input("corpus manifest pool has no records".to_string()))?;
    let declared_examples = manifest
        .get("totals")
        .and_then(|t| t.get("examples"))
        .and_then(|x| x.as_u64());

    let examples_bytes = read(&corpus_dir.join("examples.jsonl"))?;
    let actual = sha256_hex(&examples_bytes);
    if actual != examples_sha {
        return Err(ComposeError::Input(format!(
            "{} hashes to {actual}, its manifest records {examples_sha}",
            corpus_dir.join("examples.jsonl").display()
        )));
    }
    let text = std::str::from_utf8(&examples_bytes)
        .map_err(|e| ComposeError::Input(format!("corpus is not UTF-8: {e}")))?;
    let mut corpus_rows = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: CorpusRow = serde_json::from_str(line).map_err(|e| {
            ComposeError::Input(format!("corpus examples.jsonl:{}: {e}", lineno + 1))
        })?;
        corpus_rows.push(row);
    }
    if let Some(n) = declared_examples
        && n != corpus_rows.len() as u64
    {
        return Err(ComposeError::Input(format!(
            "corpus holds {} rows, its manifest records {n}",
            corpus_rows.len()
        )));
    }

    let pool_bytes = read(pool_path)?;
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

    let map_bytes = read(split_map_path)?;
    let split_map: SplitMap = serde_json::from_slice(&map_bytes)
        .map_err(|e| ComposeError::Input(format!("split map: {e}")))?;
    split_map.check()?;
    if split_map.pool.sha256 != pool_sha || split_map.pool.records != pool_records {
        return Err(ComposeError::Input(format!(
            "the split map was computed over pool {} ({} records), not this pool {} ({})",
            split_map.pool.sha256, split_map.pool.records, pool_sha, pool_records
        )));
    }

    let name_of = |p: &Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
            .ok_or_else(|| ComposeError::Input(format!("{} has no file name", p.display())))
    };
    Ok(Inputs {
        corpus_rows,
        pool,
        split_map,
        base: BaseCorpus {
            name: name_of(corpus_dir)?,
            examples_sha256: examples_sha,
        },
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

// -- candidates --------------------------------------------------------------------------------

/// A mutated corpus example usable as a row's one defect.
#[derive(Debug, Clone)]
pub struct Needle {
    pub row: CorpusRow,
    /// The span in the needle's own diff's line coordinates.
    pub local_span: LineSpan,
    pub est: u32,
    pub lines: u32,
}

/// A real commit, unmodified.
#[derive(Debug, Clone)]
pub struct Filler {
    pub pool_id: String,
    pub repo: String,
    pub path: String,
    pub language: LangId,
    /// The corpus's clean example for this record, when it has one.
    pub example_id: Option<String>,
    /// The normalized post-image: a clean row's `after` when this filler anchors it.
    pub after: String,
    pub function: Option<FunctionIdentity>,
    pub diff: String,
    pub est: u32,
    pub lines: u32,
}

/// Per-split candidate pools, after every exclusion.
pub struct Prepared {
    pub needles: BTreeMap<Split, Vec<Needle>>,
    pub fillers: BTreeMap<Split, Vec<Filler>>,
    pub refusals: BTreeMap<String, u64>,
    pub parity_checked: u64,
    pub heldout_skipped: BTreeMap<String, u64>,
    pub base: BaseCorpus,
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

/// The estimate of one file's block as a row renders it, headers included: the unit lengths are
/// budgeted in, so a row of many small files is not under-counted by its headers.
fn block_estimate(path: &str, diff: &str) -> u32 {
    let mut block = String::with_capacity(diff.len() + 4 * path.len() + 32);
    render_file(&mut block, path, diff);
    estimate_tokens(&block)
}

fn path_is_renderable(path: &str) -> bool {
    !path.is_empty() && !path.chars().any(char::is_control)
}

/// Render one pool record's commit as a clean diff, the way `generate.rs` renders a clean example
/// under `DiffShape::MultiHunk`. `None` when the record holds no change to read.
fn render_clean(record: &PoolRecord) -> Result<Option<(String, String)>, diffspan::DiffRefusal> {
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

/// Build the needle and filler pools, checking every invariant the module docstring names.
pub fn prepare(inputs: &Inputs) -> Result<Prepared, ComposeError> {
    let map = &inputs.split_map;
    let mut refusals: BTreeMap<String, u64> = BTreeMap::new();
    let mut heldout_skipped: BTreeMap<String, u64> = BTreeMap::new();

    // Every repo either input names must be in the map. Collected, then refused together.
    let mut missing: BTreeSet<String> = BTreeSet::new();
    for r in &inputs.corpus_rows {
        if map.split_of(&r.repo).is_none() {
            missing.insert(r.repo.clone());
        }
    }
    let corpus_missing = missing.len();
    for r in &inputs.pool {
        if map.split_of(&r.repo).is_none() {
            missing.insert(r.repo.clone());
        }
    }
    if !missing.is_empty() {
        return Err(ComposeError::SplitMissing {
            what: if corpus_missing > 0 {
                "corpus and pool rows"
            } else {
                "pool records"
            },
            total: missing.len(),
            first: missing.into_iter().take(5).collect(),
        });
    }

    let pool_by_id: HashMap<&str, &PoolRecord> =
        inputs.pool.iter().map(|r| (r.id.as_str(), r)).collect();
    if pool_by_id.len() != inputs.pool.len() {
        return Err(ComposeError::Input(
            "the pool holds a duplicate id".to_string(),
        ));
    }

    // (repo, path) of every mutated example in the corpus, in every split.
    let mutated_files: HashSet<(&str, &str)> = inputs
        .corpus_rows
        .iter()
        .filter(|r| r.class != MutationClass::Clean)
        .map(|r| (r.repo.as_str(), r.path.as_str()))
        .collect();

    // Parity: every stored clean diff re-renders byte for byte from its pool record.
    let mut parity_checked = 0u64;
    let mut parity_bad: Vec<String> = Vec::new();
    let mut clean_example_of: HashMap<&str, &CorpusRow> = HashMap::new();
    for r in inputs
        .corpus_rows
        .iter()
        .filter(|r| r.class == MutationClass::Clean)
    {
        let record = pool_by_id.get(r.pool_id.as_str()).ok_or_else(|| {
            ComposeError::Input(format!(
                "clean example {} names pool id {} the pool lacks",
                r.id, r.pool_id
            ))
        })?;
        parity_checked += 1;
        match render_clean(record) {
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

    // Every diff text in every split, for the exact-twin exclusion.
    let mut splits_of_diff: HashMap<[u8; 32], BTreeSet<Split>> = HashMap::new();
    let split_of = |repo: &str| map.split_of(repo).expect("checked above");
    for r in &inputs.corpus_rows {
        splits_of_diff
            .entry(digest(&r.diff))
            .or_default()
            .insert(split_of(&r.repo));
    }

    // Fillers: records whose file has no mutated example anywhere.
    let mut raw_fillers: Vec<(Split, Filler, [u8; 32])> = Vec::new();
    for record in &inputs.pool {
        let Some(language) = record.language() else {
            bump(&mut refusals, "filler_language_unrecognised");
            continue;
        };
        if mutated_files.contains(&(record.repo.as_str(), record.path.as_str())) {
            continue;
        }
        if !path_is_renderable(&record.path) {
            bump(&mut refusals, "filler_path_unrenderable");
            continue;
        }
        let split = split_of(&record.repo);
        let (diff, after, example_id, function) = match clean_example_of.get(record.id.as_str()) {
            Some(ex) => (
                ex.diff.clone(),
                ex.after.clone(),
                Some(ex.id.clone()),
                Some(ex.function.clone()),
            ),
            None => match render_clean(record) {
                Ok(Some((diff, after))) => (diff, after, None, None),
                Ok(None) => {
                    bump(&mut refusals, "filler_no_change");
                    continue;
                }
                Err(_) => {
                    bump(&mut refusals, "filler_diff_refused");
                    continue;
                }
            },
        };
        if diff.trim().is_empty() || !diff.ends_with('\n') {
            bump(&mut refusals, "filler_diff_empty_or_unterminated");
            continue;
        }
        let key = digest(&diff);
        splits_of_diff.entry(key).or_default().insert(split);
        let est = block_estimate(&record.path, &diff);
        let lines = count_lines(&diff);
        raw_fillers.push((
            split,
            Filler {
                pool_id: record.id.clone(),
                repo: record.repo.clone(),
                path: record.path.clone(),
                language,
                example_id,
                after,
                function,
                diff,
                est,
                lines,
            },
            key,
        ));
    }

    let mut fillers: BTreeMap<Split, Vec<Filler>> = BTreeMap::new();
    let mut seen_in_split: HashSet<(Split, [u8; 32])> = HashSet::new();
    for (split, filler, key) in raw_fillers {
        if split == Split::Heldout {
            bump(&mut heldout_skipped, "fillers");
            continue;
        }
        if splits_of_diff.get(&key).is_some_and(|s| s.len() > 1) {
            bump(&mut refusals, "filler_exact_twin_across_splits");
            continue;
        }
        if !seen_in_split.insert((split, key)) {
            bump(&mut refusals, "filler_duplicate_diff_in_split");
            continue;
        }
        fillers.entry(split).or_default().push(filler);
    }

    let mut needles: BTreeMap<Split, Vec<Needle>> = BTreeMap::new();
    for r in inputs
        .corpus_rows
        .iter()
        .filter(|r| r.class != MutationClass::Clean)
    {
        let split = split_of(&r.repo);
        if split == Split::Heldout {
            bump(&mut heldout_skipped, "needles");
            continue;
        }
        if !pool_by_id.contains_key(r.pool_id.as_str()) {
            return Err(ComposeError::Input(format!(
                "mutated example {} names pool id {} the pool lacks",
                r.id, r.pool_id
            )));
        }
        let Some(span) = r.span else {
            return Err(ComposeError::Input(format!(
                "mutated example {} carries no span",
                r.id
            )));
        };
        if !path_is_renderable(&r.path) {
            bump(&mut refusals, "needle_path_unrenderable");
            continue;
        }
        if !r.diff.ends_with('\n') {
            bump(&mut refusals, "needle_diff_unterminated");
            continue;
        }
        if splits_of_diff
            .get(&digest(&r.diff))
            .is_some_and(|s| s.len() > 1)
        {
            bump(&mut refusals, "needle_exact_twin_across_splits");
            continue;
        }
        let local_span = match diff_line_span(&r.diff, span) {
            Ok(s) => s,
            Err(refusal) => {
                bump(&mut refusals, refusal.code());
                continue;
            }
        };
        needles.entry(split).or_default().push(Needle {
            row: r.clone(),
            local_span,
            est: block_estimate(&r.path, &r.diff),
            lines: count_lines(&r.diff),
        });
    }

    // The invariant the filler rule exists for, checked rather than assumed.
    for (split, fs) in &fillers {
        for f in fs {
            if mutated_files.contains(&(f.repo.as_str(), f.path.as_str())) {
                return Err(ComposeError::Invariant(format!(
                    "{}: filler {} is a file the corpus also mutates",
                    split.as_str(),
                    f.pool_id
                )));
            }
        }
    }

    Ok(Prepared {
        needles,
        fillers,
        refusals,
        parity_checked,
        heldout_skipped,
        base: inputs.base.clone(),
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
    /// No filler appears in more rows than this.
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
        if self.clean_permille > 1000
            || self.max_share_permille == 0
            || self.max_share_permille > 1000
        {
            return bad("permille out of range".to_string());
        }
        if self.tolerance_permille == 0 || self.tolerance_permille >= 1000 {
            return bad("tolerance_permille must be in (0, 1000)".to_string());
        }
        if self.max_filler_uses == 0 {
            return bad("max_filler_uses must be positive".to_string());
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
    /// `needle` or `filler`.
    pub role: String,
    /// The corpus example this file's hunks are, when they are one.
    pub example_id: Option<String>,
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
    /// The anchor's pool id: the needle's, or on a clean row its first-drawn filler's.
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
    /// The needle's span over the needle file's post-image (`after`). Absent on a clean row.
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

/// What one split's composition did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SplitReport {
    pub rows: u64,
    pub by_class: BTreeMap<String, u64>,
    pub needles_available: u64,
    pub needles_used: u64,
    /// Always zero: a needle is used once. Reported so the claim is a number.
    pub needles_reused: u64,
    pub fillers_available: u64,
    pub fillers_available_from_clean_examples: u64,
    pub fillers_used_unique: u64,
    pub filler_slots: u64,
    pub filler_reuse_max: u64,
    pub filler_reuse_mean: f64,
    /// uses -> fillers with that many uses (0 included).
    pub filler_reuse_histogram: BTreeMap<u64, u64>,
    pub filler_tokens_available_est: u64,
    pub filler_est_histogram: BTreeMap<String, u64>,
    pub needle_est_histogram: BTreeMap<String, u64>,
    /// The row's estimate, binned.
    pub length_est_histogram: BTreeMap<String, u64>,
    pub files_per_row: BTreeMap<u32, u64>,
    /// `needle_index / (n_files - 1)`, binned, over mutated rows.
    pub needle_index_depth: BTreeMap<String, u64>,
    /// The needle span's first line over the diff's lines, binned, over mutated rows.
    pub needle_line_depth: BTreeMap<String, u64>,
    /// `(n_files, needle_index)` counts, so uniformity can be read per row width.
    pub needle_index_by_files: BTreeMap<String, u64>,
    pub languages_needle: BTreeMap<String, u64>,
    pub languages_filler_slots: BTreeMap<String, u64>,
    /// Mutated rows whose span covers a `-` line of the needle's diff (v3's rebase allows it).
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
    pub base_corpus: BaseCorpus,
    pub pool: PoolRef,
    pub split_map: SplitMapRef,
    pub split_params: SplitParams,
    pub options: Options,
    pub token_estimate: TokenEstimate,
    pub examples_sha256: String,
    pub totals: ComposeTotals,
    pub prepare_refusals: BTreeMap<String, u64>,
    pub heldout_skipped: BTreeMap<String, u64>,
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

/// A part of a row while it is being chosen.
#[derive(Clone, Copy)]
enum Part {
    Needle(usize),
    Filler(usize),
}

struct SplitComposer<'a> {
    needles: &'a [Needle],
    fillers: &'a [Filler],
    /// Filler indices sorted by `(est, pool_id)`, and their estimates in that order.
    order: Vec<usize>,
    sizes: Vec<u32>,
    uses: Vec<u64>,
    /// A filler used this many times is no longer drawn.
    max_uses: u64,
    needle_used: Vec<bool>,
    needle_order: Vec<usize>,
    cursor: usize,
    seen_sets: HashSet<[u8; 32]>,
}

impl<'a> SplitComposer<'a> {
    fn new(
        needles: &'a [Needle],
        fillers: &'a [Filler],
        max_uses: u64,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let mut order: Vec<usize> = (0..fillers.len()).collect();
        order.sort_by(|&a, &b| {
            fillers[a]
                .est
                .cmp(&fillers[b].est)
                .then_with(|| fillers[a].pool_id.cmp(&fillers[b].pool_id))
        });
        let sizes = order.iter().map(|&i| fillers[i].est).collect();
        let mut needle_order: Vec<usize> = (0..needles.len()).collect();
        // Fisher-Yates over the needles, so which needle a row gets does not follow corpus order.
        for i in (1..needle_order.len()).rev() {
            let j = rng.random_range(0..=i);
            needle_order.swap(i, j);
        }
        SplitComposer {
            needles,
            fillers,
            order,
            sizes,
            uses: vec![0; fillers.len()],
            max_uses,
            needle_used: vec![false; needles.len()],
            needle_order,
            cursor: 0,
            seen_sets: HashSet::new(),
        }
    }

    /// The first unused needle past the cursor whose estimate is at most `cap`.
    fn find_needle(&mut self, cap: u32) -> Option<usize> {
        while self.cursor < self.needle_order.len()
            && self.needle_used[self.needle_order[self.cursor]]
        {
            self.cursor += 1;
        }
        let end = (self.cursor + NEEDLE_LOOKAHEAD).min(self.needle_order.len());
        (self.cursor..end)
            .map(|k| self.needle_order[k])
            .find(|&n| !self.needle_used[n] && self.needles[n].est <= cap)
    }

    /// A filler with estimate in `[lo, hi]`, not yet in the row and not sharing a path with it:
    /// the least used of [`DRAWS_PER_SLOT`] draws.
    fn pick_filler(
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
        let mut best: Option<usize> = None;
        for _ in 0..DRAWS_PER_SLOT {
            let f = self.order[rng.random_range(a..b)];
            if in_row.contains(&f)
                || paths.contains(self.fillers[f].path.as_str())
                || self.uses[f] >= self.max_uses
            {
                continue;
            }
            if best.is_none_or(|b| self.uses[f] < self.uses[b]) {
                best = Some(f);
            }
        }
        best
    }

    /// Choose one row's parts. `Err` names why this attempt failed.
    fn choose(
        &mut self,
        clean: bool,
        target: u32,
        opts: &Options,
        rng: &mut ChaCha20Rng,
    ) -> Result<Vec<Part>, &'static str> {
        // Copied out so the row's path set borrows the pools, not `self`.
        let needles: &'a [Needle] = self.needles;
        let fillers: &'a [Filler] = self.fillers;
        let cap = (u64::from(target) * u64::from(opts.max_share_permille) / 1000) as u32;
        let mut parts: Vec<Part> = Vec::new();
        let mut in_row: HashSet<usize> = HashSet::new();
        let mut paths: HashSet<&'a str> = HashSet::new();
        let mut remaining = i64::from(target);
        if !clean {
            let n = self.find_needle(cap).ok_or("no_needle_fits_share")?;
            parts.push(Part::Needle(n));
            paths.insert(needles[n].row.path.as_str());
            remaining -= i64::from(needles[n].est);
        }
        let fixed = parts.len();
        let smallest = i64::from(*self.sizes.first().ok_or("no_fillers")?);
        let largest = i64::from((*self.sizes.last().ok_or("no_fillers")?).min(cap));
        if remaining < smallest {
            return Err("needle_leaves_no_room");
        }
        // Filler slots that can fill `remaining`, within the file-count range.
        let s_lo = ((remaining + largest - 1) / largest).max((opts.min_files - fixed) as i64);
        let s_hi = (remaining / smallest).min((opts.max_files - fixed) as i64);
        if s_lo > s_hi || s_hi < 1 {
            return Err("file_count_infeasible");
        }
        let slots = rng.random_range(s_lo..=s_hi);
        for j in 0..slots {
            let left = slots - j;
            let desired = remaining / left;
            let (lo, hi) = if left == 1 {
                let tol = i64::from(target) * i64::from(opts.tolerance_permille) / 1000 / 2;
                (remaining - tol, remaining + tol)
            } else {
                (desired * 2 / 3, desired * 3 / 2)
            };
            let lo = lo.max(1);
            let hi = hi.min(i64::from(cap));
            if hi < lo {
                return Err("slot_window_empty");
            }
            let f = self
                .pick_filler(lo as u32, hi as u32, &in_row, &paths, rng)
                .ok_or("no_filler_in_window")?;
            in_row.insert(f);
            paths.insert(fillers[f].path.as_str());
            remaining -= i64::from(fillers[f].est);
            parts.push(Part::Filler(f));
        }
        Ok(parts)
    }
}

fn est_of(composer: &SplitComposer<'_>, part: Part) -> u32 {
    match part {
        Part::Needle(n) => composer.needles[n].est,
        Part::Filler(f) => composer.fillers[f].est,
    }
}

/// Compose one split's rows.
fn compose_split(
    split: Split,
    wanted: usize,
    needles: &[Needle],
    fillers: &[Filler],
    opts: &Options,
) -> Result<(Vec<ComposedRow>, SplitReport), ComposeError> {
    let mut rng = split_rng(opts.seed, split);
    let mut report = SplitReport {
        needles_available: needles.len() as u64,
        fillers_available: fillers.len() as u64,
        fillers_available_from_clean_examples: fillers
            .iter()
            .filter(|f| f.example_id.is_some())
            .count() as u64,
        filler_tokens_available_est: fillers.iter().map(|f| u64::from(f.est)).sum(),
        ..SplitReport::default()
    };
    for f in fillers {
        *report
            .filler_est_histogram
            .entry(length_bin(f.est))
            .or_insert(0) += 1;
    }
    for n in needles {
        *report
            .needle_est_histogram
            .entry(length_bin(n.est))
            .or_insert(0) += 1;
    }
    let mut composer = SplitComposer::new(needles, fillers, opts.max_filler_uses, &mut rng);
    let mut rows = Vec::with_capacity(wanted);

    while rows.len() < wanted {
        // Class and target are drawn once per row and held through every attempt. Re-drawing the
        // target on a failed attempt would bias the lengths toward whatever is easiest to fill.
        let clean = rng.random_range(0..1000u32) < opts.clean_permille;
        let target = rng.random_range(opts.min_tokens..=opts.max_tokens);
        let mut accepted: Option<(
            Vec<Part>,
            String,
            Vec<Constituent>,
            Option<(u32, LineSpan, u32)>,
            u32,
        )> = None;
        for _ in 0..ATTEMPTS_PER_ROW {
            let parts = match composer.choose(clean, target, opts, &mut rng) {
                Ok(p) => p,
                Err(why) => {
                    bump(&mut report.attempts_refused, why);
                    continue;
                }
            };
            // Same constituent set as an earlier row: refused.
            let mut ids: Vec<&str> = parts
                .iter()
                .map(|&p| match p {
                    Part::Needle(n) => composer.needles[n].row.pool_id.as_str(),
                    Part::Filler(f) => composer.fillers[f].pool_id.as_str(),
                })
                .collect();
            ids.sort_unstable();
            let set_key: [u8; 32] = Sha256::digest(ids.join("\0").as_bytes()).into();
            if composer.seen_sets.contains(&set_key) {
                bump(&mut report.attempts_refused, "duplicate_constituent_set");
                continue;
            }
            // File order: uniform.
            let mut ordered = parts.clone();
            for i in (1..ordered.len()).rev() {
                let j = rng.random_range(0..=i);
                ordered.swap(i, j);
            }
            let mut diff = String::new();
            let mut constituents = Vec::with_capacity(ordered.len());
            let mut needle_at: Option<(u32, LineSpan, u32)> = None;
            let mut line = 0u32;
            for (index, &part) in ordered.iter().enumerate() {
                let (path, body) = match part {
                    Part::Needle(n) => {
                        (&composer.needles[n].row.path, &composer.needles[n].row.diff)
                    }
                    Part::Filler(f) => (&composer.fillers[f].path, &composer.fillers[f].diff),
                };
                let first_line = line + 1;
                render_file(&mut diff, path, body);
                line += FILE_HEADER_LINES + count_lines(body);
                let c = match part {
                    Part::Needle(n) => {
                        let nd = &composer.needles[n];
                        let span = LineSpan::new(
                            first_line + FILE_HEADER_LINES - 1 + nd.local_span.start,
                            first_line + FILE_HEADER_LINES - 1 + nd.local_span.end,
                        );
                        needle_at = Some((index as u32, span, first_line));
                        Constituent {
                            pool_id: nd.row.pool_id.clone(),
                            repo: nd.row.repo.clone(),
                            path: nd.row.path.clone(),
                            language: nd.row.language,
                            role: "needle".to_string(),
                            example_id: Some(nd.row.id.clone()),
                            first_line,
                            last_line: line,
                            est_tokens: nd.est,
                        }
                    }
                    Part::Filler(f) => {
                        let fl = &composer.fillers[f];
                        Constituent {
                            pool_id: fl.pool_id.clone(),
                            repo: fl.repo.clone(),
                            path: fl.path.clone(),
                            language: fl.language,
                            role: "filler".to_string(),
                            example_id: fl.example_id.clone(),
                            first_line,
                            last_line: line,
                            est_tokens: fl.est,
                        }
                    }
                };
                constituents.push(c);
            }
            let est = estimate_tokens(&diff);
            let lo = u64::from(target) * u64::from(1000 - opts.tolerance_permille) / 1000;
            let hi = u64::from(target) * u64::from(1000 + opts.tolerance_permille) / 1000;
            if u64::from(est) < lo || u64::from(est) > hi {
                bump(&mut report.attempts_refused, "estimate_off_target");
                continue;
            }
            if est > opts.hard_max_tokens {
                bump(&mut report.attempts_refused, "over_hard_max_tokens");
                continue;
            }
            let share_cap = u64::from(est) * u64::from(opts.max_share_permille) / 1000;
            if ordered
                .iter()
                .any(|&p| u64::from(est_of(&composer, p)) > share_cap)
            {
                bump(&mut report.attempts_refused, "constituent_over_share");
                continue;
            }
            composer.seen_sets.insert(set_key);
            accepted = Some((ordered, diff, constituents, needle_at, est));
            break;
        }
        let Some((ordered, diff, constituents, needle_at, est)) = accepted else {
            return Err(ComposeError::Shortfall {
                split: split.as_str(),
                made: rows.len(),
                wanted,
                refusals: report.attempts_refused.clone(),
            });
        };

        // Commit the choice.
        for &p in &ordered {
            match p {
                Part::Needle(n) => composer.needle_used[n] = true,
                Part::Filler(f) => composer.uses[f] += 1,
            }
        }
        let index = rows.len();
        let n_files = ordered.len() as u32;
        let row = match needle_at {
            Some((needle_index, diff_span, _)) => {
                let nd = ordered
                    .iter()
                    .find_map(|&p| match p {
                        Part::Needle(n) => Some(&composer.needles[n]),
                        Part::Filler(_) => None,
                    })
                    .expect("a mutated row holds its needle");
                ComposedRow {
                    id: format!("compose:{}:{index:06}", split.as_str()),
                    pool_id: nd.row.pool_id.clone(),
                    repo: nd.row.repo.clone(),
                    path: nd.row.repo.clone(),
                    language: nd.row.language,
                    class: nd.row.class,
                    operator: nd.row.operator.clone(),
                    silent: nd.row.silent,
                    span: nd.row.span,
                    function: FunctionIdentity {
                        repo: nd.row.function.repo.clone(),
                        path: nd.row.repo.clone(),
                        symbol: nd.row.function.symbol.clone(),
                        arity: nd.row.function.arity,
                    },
                    after: nd.row.after.clone(),
                    diff,
                    hunk_constrained: nd.row.hunk_constrained,
                    detail: format!(
                        "{} of {n_files} files; needle {}",
                        needle_index + 1,
                        nd.row.id
                    ),
                    seed: opts.seed,
                    tool_version: crate::TOOL_VERSION.to_string(),
                    composed: true,
                    split,
                    diff_span: Some(diff_span),
                    needle_index: Some(needle_index),
                    n_files,
                    est_tokens: est,
                    constituents,
                }
            }
            None => {
                // The anchor is the row's filler that sorts first in the pool, a choice that does
                // not move with the file order.
                let anchor = ordered
                    .iter()
                    .filter_map(|&p| match p {
                        Part::Filler(f) => Some(f),
                        Part::Needle(_) => None,
                    })
                    .min()
                    .ok_or_else(|| {
                        ComposeError::Invariant("a clean row holds no filler".to_string())
                    })?;
                let fl = &composer.fillers[anchor];
                let function = fl.function.clone().unwrap_or_else(|| FunctionIdentity {
                    repo: fl.repo.clone(),
                    path: fl.path.clone(),
                    symbol: "<pull-request>".to_string(),
                    arity: 0,
                });
                ComposedRow {
                    id: format!("compose:{}:{index:06}", split.as_str()),
                    pool_id: fl.pool_id.clone(),
                    repo: fl.repo.clone(),
                    path: fl.repo.clone(),
                    language: fl.language,
                    class: MutationClass::Clean,
                    operator: None,
                    silent: false,
                    span: None,
                    function: FunctionIdentity {
                        repo: function.repo,
                        path: fl.repo.clone(),
                        symbol: function.symbol,
                        arity: function.arity,
                    },
                    after: fl.after.clone(),
                    diff,
                    hunk_constrained: true,
                    detail: format!("{n_files} unmodified commits"),
                    seed: opts.seed,
                    tool_version: crate::TOOL_VERSION.to_string(),
                    composed: true,
                    split,
                    diff_span: None,
                    needle_index: None,
                    n_files,
                    est_tokens: est,
                    constituents,
                }
            }
        };
        verify_row(&row, &composer)?;
        note_row(&mut report, &row);
        rows.push(row);
    }

    report.needles_used = composer.needle_used.iter().filter(|&&u| u).count() as u64;
    report.fillers_used_unique = composer.uses.iter().filter(|&&u| u > 0).count() as u64;
    report.filler_slots = composer.uses.iter().sum();
    report.filler_reuse_max = composer.uses.iter().copied().max().unwrap_or(0);
    report.filler_reuse_mean = if report.fillers_used_unique == 0 {
        0.0
    } else {
        report.filler_slots as f64 / report.fillers_used_unique as f64
    };
    for &u in &composer.uses {
        *report.filler_reuse_histogram.entry(u).or_insert(0) += 1;
    }
    Ok((rows, report))
}

/// Every check a row must pass before it is kept, re-derived from the rendered text.
fn verify_row(row: &ComposedRow, composer: &SplitComposer<'_>) -> Result<(), ComposeError> {
    let fail = |m: String| Err(ComposeError::Invariant(format!("{}: {m}", row.id)));
    let lines: Vec<&str> = row.diff.split('\n').collect();
    let total = count_lines(&row.diff);
    if lines.len() as u32 != total + 1 || !row.diff.ends_with('\n') {
        return fail("the composed diff does not end in exactly one newline".to_string());
    }
    if row.constituents.len() as u32 != row.n_files {
        return fail("constituent count disagrees with n_files".to_string());
    }
    // Blocks tile the diff, each opening with its three headers.
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
            return fail(format!(
                "block of {} does not open with its file headers",
                c.pool_id
            ));
        }
        if !at(c.first_line + 3).starts_with("@@ ") {
            return fail(format!(
                "block of {} does not open with a hunk header",
                c.pool_id
            ));
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
    let needles: Vec<&Constituent> = row
        .constituents
        .iter()
        .filter(|c| c.role == "needle")
        .collect();
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
                    return fail(format!(
                        "span endpoint line {n} is {l:?}, not '+' or context"
                    ));
                }
            }
            // Within one hunk: no hunk header between the endpoints. Positional: inside a block's
            // body, only a hunk header begins with '@'.
            if (span.start..=span.end).any(|n| lines[(n - 1) as usize].starts_with('@')) {
                return fail("span crosses a hunk header".to_string());
            }
            // The text the span points at is the needle's own span text.
            let nd = composer
                .needles
                .iter()
                .find(|n| n.row.id.as_str() == needle.example_id.as_deref().unwrap_or(""))
                .ok_or_else(|| ComposeError::Invariant(format!("{}: needle not found", row.id)))?;
            let own: Vec<&str> = nd.row.diff.split('\n').collect();
            for k in 0..=(span.end - span.start) {
                if lines[(span.start + k - 1) as usize]
                    != own[(nd.local_span.start + k - 1) as usize]
                {
                    return fail("span text differs from the needle's own span text".to_string());
                }
            }
            if row
                .needle_index
                .map(|i| row.constituents[i as usize].role.as_str())
                != Some("needle")
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
    *report
        .by_class
        .entry(row.class.as_str().to_string())
        .or_insert(0) += 1;
    *report
        .length_est_histogram
        .entry(length_bin(row.est_tokens))
        .or_insert(0) += 1;
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
        *report
            .needle_index_depth
            .entry(depth_bucket(frac).to_string())
            .or_insert(0) += 1;
        let total = count_lines(&row.diff);
        let line_frac = if total > 1 {
            f64::from(span.start - 1) / f64::from(total - 1)
        } else {
            0.0
        };
        *report
            .needle_line_depth
            .entry(depth_bucket(line_frac).to_string())
            .or_insert(0) += 1;
        *report
            .needle_index_by_files
            .entry(format!("{:02}:{:02}", row.n_files, index))
            .or_insert(0) += 1;
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
    let empty_n: Vec<Needle> = Vec::new();
    let empty_f: Vec<Filler> = Vec::new();
    for (split, wanted) in [(Split::Train, opts.train_rows), (Split::Val, opts.val_rows)] {
        let needles = prepared.needles.get(&split).unwrap_or(&empty_n);
        let fillers = prepared.fillers.get(&split).unwrap_or(&empty_f);
        let (r, report) = compose_split(split, wanted, needles, fillers, opts)?;
        rows.extend(r);
        splits.insert(split.as_str().to_string(), report);
    }
    let mut by_split = BTreeMap::new();
    let mut by_class = BTreeMap::new();
    for r in &rows {
        *by_split.entry(r.split.as_str().to_string()).or_insert(0) += 1;
        *by_class.entry(r.class.as_str().to_string()).or_insert(0) += 1;
    }
    let mut composed = Composed {
        manifest: ComposeManifest {
            schema: MANIFEST_SCHEMA.to_string(),
            tool_version: crate::TOOL_VERSION.to_string(),
            seed: opts.seed,
            seed_derivation:
                "ChaCha20Rng::from_seed(sha256(seed.to_le_bytes() || b\"compose\\0\" || split))"
                    .to_string(),
            base_corpus: prepared.base.clone(),
            pool: prepared.pool_ref.clone(),
            split_map: prepared.split_map_ref.clone(),
            split_params: prepared.split_params.clone(),
            options: opts.clone(),
            token_estimate: token_estimate(),
            examples_sha256: String::new(),
            totals: ComposeTotals {
                examples: rows.len() as u64,
                by_split,
                by_class,
            },
            prepare_refusals: prepared.refusals.clone(),
            heldout_skipped: prepared.heldout_skipped.clone(),
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
            .unwrap_or(usize::MAX);
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

/// The supply a run would draw from, without composing: per split, the needle and filler pools
/// and their estimated-size histograms.
pub fn census(prepared: &Prepared) -> BTreeMap<String, SplitReport> {
    let mut out = BTreeMap::new();
    for split in [Split::Train, Split::Val] {
        let mut report = SplitReport::default();
        if let Some(n) = prepared.needles.get(&split) {
            report.needles_available = n.len() as u64;
            for x in n {
                *report
                    .needle_est_histogram
                    .entry(census_bin(x.est))
                    .or_insert(0) += 1;
            }
        }
        if let Some(f) = prepared.fillers.get(&split) {
            report.fillers_available = f.len() as u64;
            report.fillers_available_from_clean_examples =
                f.iter().filter(|x| x.example_id.is_some()).count() as u64;
            report.filler_tokens_available_est = f.iter().map(|x| u64::from(x.est)).sum();
            for x in f {
                *report
                    .filler_est_histogram
                    .entry(census_bin(x.est))
                    .or_insert(0) += 1;
            }
        }
        out.insert(split.as_str().to_string(), report);
    }
    out
}

fn census_bin(est: u32) -> String {
    const EDGES: [u32; 10] = [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1500];
    for w in EDGES.windows(2) {
        if est >= w[0] && est < w[1] {
            return format!("{:04}-{:04}", w[0], w[1]);
        }
    }
    format!("{:04}+", EDGES[EDGES.len() - 1])
}
