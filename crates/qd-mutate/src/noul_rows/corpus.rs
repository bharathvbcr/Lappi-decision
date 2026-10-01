//! Source (b), v2 forms: the `code.defect_class` corpus's own diffs, from train-split repos,
//! scrambled the way a corrupted patch is scrambled.
//!
//! v1 scrambled windows of pool files with their tokens AND lines shuffled. The OOD suite's
//! scrambled cases (`qd_train.ood.build_ood_suite`) instead line-shuffle diff hunks and keep
//! every token whole, and J4 abstained on 3 / 4 / 8 of their 60: the form did not transfer
//! (Fable F2, 2026-10-01). v2 draws from the corpus the model is trained on (corpus v3) and
//! writes two forms in equal halves:
//!
//! * [`FORM_LINES`] -- every hunk's body lines permuted and, for a multi-hunk diff, the hunks'
//!   order permuted. Each line and each token is the diff's own; only the order is broken.
//! * [`FORM_TOKENS`] -- the same, then the tokens of every body line shuffled too.
//!
//! Each hunk keeps its header on its first line, and the header's counts still match its body
//! (a permutation keeps the marker counts), so the row has the corpus's hunk shape and the
//! label cannot be read off the shape.
//!
//! Disjoint from the suite by source: the suite scrambles the needle suite's own filler
//! templates under `src/scrambled_<i>.<ext>`; these are real commitpackft files under their own
//! paths, licensed per row through the pool join, exactly as the main corpus is.
//!
//! A scrambled row and the corpus row it came from share a repo, so they are one split unit and
//! dedupe never chains them across repos. [`MAX_SHINGLE_JACCARD`] keeps the two well below the
//! dedupe threshold anyway, so a scrambled row never stands in for its original.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

use qd_mutate::lang::LangId;
use qd_mutate::manifest::sha256_hex;

use crate::hunk::{has_char_in, item_rng, order_key};
use crate::scramble::{shuffle_tokens, LANGUAGES};

/// The key every v2 scrambled draw is derived under, distinct from v1's `scrambled`.
pub const SOURCE_KEY: &str = "scrambled-corpus";
pub const FORM_LINES: &str = "lines";
pub const FORM_TOKENS: &str = "tokens";

/// Corpus v3 holds 49,953 rows; a file this many times larger is not that corpus.
const MAX_LINES: usize = 1_000_000;
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;
/// Body lines (all hunks together). Fewer and a permutation leaves little to read as broken.
const MIN_BODY_LINES: usize = 6;
/// Lines with two or more distinct tokens, for [`FORM_TOKENS`]: v1's bound.
const MIN_SHUFFLABLE_LINES: usize = 4;
/// v1's length bounds: a line longer than this is minified or generated, and the diff band
/// keeps length from being a cue against the corpus's 5th..95th percentile.
const MAX_LINE_CHARS: usize = 200;
const MIN_DIFF_CHARS: usize = 120;
const MAX_DIFF_CHARS: usize = 2_400;
/// At least half of the body positions must hold a different line than the original's.
const MIN_MOVED_NUM: usize = 1;
const MIN_MOVED_DEN: usize = 2;
/// Word 5-shingle Jaccard against the original diff must stay below this. The dedupe pass
/// links pairs at 0.8 (`qd_data.config.DataConfig.dedupe_threshold`); this keeps a margin.
pub const MAX_SHINGLE_JACCARD: f64 = 0.5;
const SHINGLE: usize = 5;
const MAX_TRIES: usize = 8;

/// The fields of a corpus row this source reads. Every other field is ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct CorpusExample {
    pub id: String,
    pub pool_id: String,
    pub repo: String,
    pub path: String,
    pub language: String,
    pub diff: String,
}

#[derive(Deserialize)]
struct CorpusManifest {
    examples_sha256: String,
    pool: CorpusPool,
}

#[derive(Deserialize)]
struct CorpusPool {
    sha256: String,
}

/// What the manifest records about the corpus read.
#[derive(Debug, Clone, Serialize)]
pub struct CorpusMeta {
    /// The corpus directory's name (`commitpackft-corpus-v3`): re-anchored under `data/pool/`.
    pub dir: String,
    pub examples_sha256: String,
    /// The pool the corpus was generated from: the allowlist's, checked.
    pub pool_sha256: String,
    pub examples_read: u64,
    /// Rows whose pool file the allowlist admits (train-split repo, admitted licence) and whose
    /// language is one of the four.
    pub examples_admitted: u64,
    pub excluded: BTreeMap<String, u64>,
}

pub struct Corpus {
    pub examples: Vec<CorpusExample>,
    pub meta: CorpusMeta,
}

/// Read `dir/examples.jsonl`, refused unless it hashes to `dir/manifest.json`'s
/// `examples_sha256` and that manifest names `pool_sha256` (the allowlist's pool) as its pool.
/// Keeps only rows of `allowed` pool files in the four languages.
pub fn read(
    dir: &Path,
    allowed: &BTreeMap<String, String>,
    pool_sha256: &str,
) -> Result<Corpus, String> {
    let manifest_path = dir.join("manifest.json");
    let examples_path = dir.join("examples.jsonl");
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|e| format!("reading {}: {e}", manifest_path.display()))?;
    let manifest: CorpusManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("{} is not a corpus manifest: {e}", manifest_path.display()))?;
    if manifest.pool.sha256 != pool_sha256 {
        return Err(format!(
            "{} was generated from the pool {}, but the allowlist's pool is {pool_sha256}: its \
             pool ids would be judged against another pool",
            manifest_path.display(),
            manifest.pool.sha256
        ));
    }
    let bytes =
        fs::read(&examples_path).map_err(|e| format!("reading {}: {e}", examples_path.display()))?;
    let sha = sha256_hex(&bytes);
    if sha != manifest.examples_sha256 {
        return Err(format!(
            "{} hashes to {sha}, but its manifest's examples_sha256 is {}",
            examples_path.display(),
            manifest.examples_sha256
        ));
    }
    let wanted: BTreeSet<&str> = LANGUAGES.iter().map(|l| l.as_str()).collect();
    let mut examples = Vec::new();
    let mut excluded: BTreeMap<String, u64> = BTreeMap::new();
    let mut read = 0u64;
    for (i, line) in bytes.split(|b| *b == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if i >= MAX_LINES {
            return Err(format!("{} holds more than {MAX_LINES} lines", examples_path.display()));
        }
        if line.len() > MAX_LINE_BYTES {
            return Err(format!("{}:{}: {} bytes", examples_path.display(), i + 1, line.len()));
        }
        let ex: CorpusExample = serde_json::from_slice(line)
            .map_err(|e| format!("{}:{} is not a corpus row: {e}", examples_path.display(), i + 1))?;
        read += 1;
        if !allowed.contains_key(&ex.pool_id) {
            // Val or held-out repo, or a licence the allowlist did not admit: never read further.
            *excluded.entry("pool_id_not_allowlisted".to_string()).or_insert(0) += 1;
            continue;
        }
        if !wanted.contains(ex.language.as_str()) {
            *excluded.entry(format!("language:{}", ex.language)).or_insert(0) += 1;
            continue;
        }
        examples.push(ex);
    }
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{} has no directory name", dir.display()))?
        .to_string();
    let meta = CorpusMeta {
        dir: name,
        examples_sha256: sha,
        pool_sha256: pool_sha256.to_string(),
        examples_read: read,
        examples_admitted: examples.len() as u64,
        excluded,
    };
    Ok(Corpus { examples, meta })
}

/// Admitted examples by language, in keyed-hash order.
pub fn candidates(examples: &[CorpusExample], seed: u64) -> BTreeMap<LangId, Vec<&CorpusExample>> {
    let mut out: BTreeMap<LangId, Vec<(&CorpusExample, [u8; 32])>> = BTreeMap::new();
    for ex in examples {
        let Some(lang) = LANGUAGES.iter().find(|l| l.as_str() == ex.language) else { continue };
        out.entry(*lang).or_default().push((ex, order_key(seed, SOURCE_KEY, &ex.id)));
    }
    out.into_iter()
        .map(|(lang, mut v)| {
            v.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.id.cmp(&b.0.id)));
            (lang, v.into_iter().map(|(e, _)| e).collect())
        })
        .collect()
}

/// Why an example produced no row. Counted in the manifest, never silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Skip {
    CarriageReturn,
    NotAHunk,
    TooShort,
    LongLine,
    InvisibleFormat,
    TooFewShufflableLines,
    Unscrambled,
    TooCloseToOriginal,
    DiffLength,
}

impl Skip {
    pub fn as_str(self) -> &'static str {
        match self {
            Skip::CarriageReturn => "carriage_return",
            Skip::NotAHunk => "line_outside_a_hunk_body",
            Skip::TooShort => "fewer_body_lines_than_min",
            Skip::LongLine => "line_over_max_chars",
            Skip::InvisibleFormat => "invisible_format_character",
            Skip::TooFewShufflableLines => "too_few_shufflable_lines",
            Skip::Unscrambled => "shuffle_moved_too_little",
            Skip::TooCloseToOriginal => "shingle_jaccard_to_original_over_max",
            Skip::DiffLength => "diff_length_out_of_band",
        }
    }
}

struct Hunk<'a> {
    header: &'a str,
    /// Body lines, each with its marker.
    body: Vec<&'a str>,
}

fn parse(diff: &str) -> Result<Vec<Hunk<'_>>, Skip> {
    let mut hunks: Vec<Hunk> = Vec::new();
    for line in diff.lines() {
        if line.starts_with("@@ -") {
            hunks.push(Hunk { header: line, body: Vec::new() });
            continue;
        }
        let Some(h) = hunks.last_mut() else { return Err(Skip::NotAHunk) };
        if !matches!(line.chars().next(), Some(' ' | '-' | '+')) {
            return Err(Skip::NotAHunk);
        }
        h.body.push(line);
    }
    if hunks.is_empty() || hunks.iter().any(|h| h.body.is_empty()) {
        return Err(Skip::NotAHunk);
    }
    Ok(hunks)
}

/// A permutation of `0..n` that is not the identity when one exists (`n >= 2`).
fn permutation(n: usize, rng: &mut ChaCha20Rng) -> Vec<usize> {
    let identity: Vec<usize> = (0..n).collect();
    let mut p = identity.clone();
    if n < 2 {
        return p;
    }
    for _ in 0..MAX_TRIES {
        p.shuffle(rng);
        if p != identity {
            break;
        }
    }
    p
}

fn shingles(text: &str) -> BTreeSet<Vec<&str>> {
    let words: Vec<&str> = text.split_whitespace().collect();
    words.windows(SHINGLE).map(<[&str]>::to_vec).collect()
}

/// Word 5-shingle Jaccard; two texts too short for a shingle are compared as their word lists.
pub fn shingle_jaccard(a: &str, b: &str) -> f64 {
    let (sa, sb) = (shingles(a), shingles(b));
    if sa.is_empty() && sb.is_empty() {
        let wa: Vec<&str> = a.split_whitespace().collect();
        let wb: Vec<&str> = b.split_whitespace().collect();
        return if wa == wb { 1.0 } else { 0.0 };
    }
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    inter as f64 / union as f64
}

/// The scrambled diff for `ex` in `form`, or why it has none.
pub fn render(
    ex: &CorpusExample,
    form: &str,
    seed: u64,
    invisible: &[(u32, u32)],
) -> Result<String, Skip> {
    if ex.diff.contains('\r') {
        return Err(Skip::CarriageReturn);
    }
    let hunks = parse(&ex.diff)?;
    if ex.diff.lines().any(|l| l.chars().count() > MAX_LINE_CHARS) {
        return Err(Skip::LongLine);
    }
    if has_char_in(&ex.diff, invisible) {
        return Err(Skip::InvisibleFormat);
    }
    let total: usize = hunks.iter().map(|h| h.body.len()).sum();
    if total < MIN_BODY_LINES {
        return Err(Skip::TooShort);
    }
    let mut rng = item_rng(seed, &format!("{SOURCE_KEY}:{form}"), &ex.id);

    // Line order within each hunk, then hunk order.
    let mut moved = 0usize;
    let mut bodies: Vec<Vec<String>> = Vec::with_capacity(hunks.len());
    for h in &hunks {
        let p = permutation(h.body.len(), &mut rng);
        moved += p.iter().enumerate().filter(|&(i, &j)| h.body[i] != h.body[j]).count();
        bodies.push(p.iter().map(|&j| h.body[j].to_string()).collect());
    }
    if moved * MIN_MOVED_DEN < total * MIN_MOVED_NUM {
        return Err(Skip::Unscrambled);
    }
    if form == FORM_TOKENS {
        let mut shufflable = 0usize;
        let mut changed = 0usize;
        for body in &mut bodies {
            for line in body.iter_mut() {
                let (mark, text) = line.split_at(1);
                if let Some(shuffled) = shuffle_tokens(text, &mut rng) {
                    shufflable += 1;
                    changed += usize::from(shuffled != text);
                    *line = format!("{mark}{shuffled}");
                }
            }
        }
        if shufflable < MIN_SHUFFLABLE_LINES {
            return Err(Skip::TooFewShufflableLines);
        }
        if changed * 2 < shufflable {
            return Err(Skip::Unscrambled);
        }
    }
    let order = permutation(hunks.len(), &mut rng);
    let mut out = String::with_capacity(ex.diff.len());
    for &i in &order {
        out.push_str(hunks[i].header);
        out.push('\n');
        for line in &bodies[i] {
            out.push_str(line);
            out.push('\n');
        }
    }
    if out == ex.diff {
        return Err(Skip::Unscrambled);
    }
    if shingle_jaccard(&out, &ex.diff) >= MAX_SHINGLE_JACCARD {
        return Err(Skip::TooCloseToOriginal);
    }
    if !(MIN_DIFF_CHARS..=MAX_DIFF_CHARS).contains(&out.chars().count()) {
        return Err(Skip::DiffLength);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIFF: &str = "\
@@ -10,9 +10,9 @@
 def load_config(path, defaults=None):
     defaults = defaults or {}
     with open(path) as handle:
-        raw = handle.read()
+        raw = handle.read().strip()
     values = dict(defaults)
     for line in raw.splitlines():
         key, _, value = line.partition('=')
         values[key.strip()] = value.strip()
     return values
";

    const TWO: &str = "\
@@ -3,5 +3,5 @@
 import os
 import sys
-from json import loads
+from json import loads, dumps
 from pathlib import Path
 from typing import Any
@@ -40,6 +40,6 @@
     for name in sorted(names):
         entry = table.get(name)
-        if entry is None:
+        if entry is None or entry.retired:
             continue
         out.append(entry.render(width=width))
     return out
";

    fn ex(id: &str, diff: &str) -> CorpusExample {
        CorpusExample {
            id: id.to_string(),
            pool_id: id.to_string(),
            repo: "o/r".to_string(),
            path: "pkg/config.py".to_string(),
            language: "python".to_string(),
            diff: diff.to_string(),
        }
    }

    fn body_multiset(diff: &str) -> BTreeMap<String, Vec<String>> {
        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut head = String::new();
        for l in diff.lines() {
            if l.starts_with("@@ -") {
                head = l.to_string();
                out.entry(head.clone()).or_default();
            } else {
                out.get_mut(&head).unwrap().push(l.to_string());
            }
        }
        for v in out.values_mut() {
            v.sort();
        }
        out
    }

    #[test]
    fn the_lines_form_permutes_lines_within_hunks_and_keeps_every_token() {
        let mut rendered = 0;
        for seed in 0..40 {
            let Ok(out) = render(&ex("a", DIFF), FORM_LINES, seed, &[(0xFEFF, 0xFEFF)]) else {
                continue;
            };
            rendered += 1;
            assert!(out.starts_with("@@ -10,9 +10,9 @@\n"), "{out}");
            assert_eq!(body_multiset(&out), body_multiset(DIFF));
            assert_ne!(out, DIFF);
            assert!(shingle_jaccard(&out, DIFF) < MAX_SHINGLE_JACCARD);
        }
        assert!(rendered >= 30, "only {rendered} of 40 seeds rendered");
        assert_eq!(render(&ex("a", DIFF), FORM_LINES, 3, &[]),
                   render(&ex("a", DIFF), FORM_LINES, 3, &[]));
    }

    #[test]
    fn a_multi_hunk_diff_keeps_each_header_with_its_own_lines_and_its_order_can_move() {
        let mut moved_order = 0;
        for seed in 0..40 {
            let Ok(out) = render(&ex("b", TWO), FORM_LINES, seed, &[]) else { continue };
            assert_eq!(body_multiset(&out), body_multiset(TWO));
            assert!(out.starts_with("@@ -"));
            moved_order += usize::from(out.starts_with("@@ -40,6"));
        }
        assert!(moved_order >= 30, "hunk order moved on {moved_order} of 40 seeds");
    }

    #[test]
    fn the_tokens_form_also_shuffles_tokens_inside_lines_keeping_markers_and_indent() {
        let mut rendered = 0;
        for seed in 0..40 {
            let Ok(out) = render(&ex("c", DIFF), FORM_TOKENS, seed, &[]) else { continue };
            rendered += 1;
            let key = |l: &str| {
                let indent = l.len() - l[1..].trim_start().len() - 1;
                let mut t: Vec<&str> = l[1..].split_whitespace().collect();
                t.sort();
                format!("{}{indent}{}", &l[..1], t.join(" "))
            };
            let mut got: Vec<String> = out.lines().skip(1).map(key).collect();
            let mut want: Vec<String> = DIFF.lines().skip(1).map(key).collect();
            got.sort();
            want.sort();
            assert_eq!(got, want);
            let originals: BTreeSet<&str> = DIFF.lines().collect();
            assert!(out.lines().skip(1).filter(|l| !originals.contains(l)).count() >= 4, "{out}");
        }
        assert!(rendered >= 30, "only {rendered} of 40 seeds rendered");
    }

    #[test]
    fn every_reason_an_example_has_no_row_is_named() {
        assert_eq!(render(&ex("d", "@@ -1,2 +1,2 @@\n a\n-b\n+c\n"), FORM_LINES, 0, &[]),
                   Err(Skip::TooShort));
        assert_eq!(render(&ex("e", &DIFF.replace('\n', "\r\n")), FORM_LINES, 0, &[]),
                   Err(Skip::CarriageReturn));
        assert_eq!(render(&ex("f", &format!("{DIFF}\\ No newline at end of file\n")),
                          FORM_LINES, 0, &[]), Err(Skip::NotAHunk));
        assert_eq!(render(&ex("g", &DIFF.replace("dict(", "dict\u{FEFF}(")), FORM_LINES, 0,
                          &[(0xFEFF, 0xFEFF)]), Err(Skip::InvisibleFormat));
        let same = format!("@@ -1,8 +1,8 @@\n{}", " x = 1\n".repeat(8));
        assert_eq!(render(&ex("h", &same), FORM_LINES, 0, &[]), Err(Skip::Unscrambled));
        let flat = format!("@@ -1,8 +1,8 @@\n{}", (0..8).map(|i| format!(" x{i}\n"))
            .collect::<String>());
        assert_eq!(render(&ex("i", &flat), FORM_TOKENS, 0, &[]),
                   Err(Skip::TooFewShufflableLines));
        let names: BTreeSet<&str> = [Skip::TooShort, Skip::NotAHunk, Skip::Unscrambled,
            Skip::TooCloseToOriginal, Skip::DiffLength, Skip::LongLine]
            .into_iter().map(Skip::as_str).collect();
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn the_jaccard_sees_a_copy_and_not_a_stranger() {
        assert_eq!(shingle_jaccard(DIFF, DIFF), 1.0);
        assert!(shingle_jaccard(DIFF, TWO) < 0.05);
        assert_eq!(shingle_jaccard("a b", "a b"), 1.0);
    }
}
