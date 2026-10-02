//! Source (b): real code, scrambled. A window of a commitpackft pool file from a train-split
//! repo, laid out as the corpus lays out a diff, then with the tokens of every line shuffled and
//! the lines themselves shuffled -- code-shaped text that is no program.
//!
//! Disjoint from the OOD suite's scrambled cases by source: the suite line-shuffles the needle
//! suite's own filler templates, several hunks at a time, under `src/scrambled_<i>.<ext>`. These
//! are real files from the pool (so the licence is the file's, joined through the pool), one
//! hunk each, under the file's own path.

use std::collections::{BTreeMap, BTreeSet};

use rand::Rng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;

use qd_mutate::lang::LangId;
use qd_mutate::pool::PoolRecord;

use crate::hunk::{self, draw_shape, has_char_in, item_rng, mark_block, order_key, Line};

pub const SOURCE: &str = "scrambled";

/// The languages of the `code.defect_class` corpus: the in-distribution languages, scrambled.
/// Owned by `qd-lang`, which the serving runtime's admission check reads too.
pub const LANGUAGES: [LangId; 4] = qd_lang::DEFECT_CLASS_POOL_LANGUAGES;

/// Window length in lines. The corpus's diffs are 8 / 10 / 16 / 31 lines at the 5th / 50th /
/// 75th / 90th percentiles; a window draws from that bulk.
const MIN_WINDOW: usize = 8;
const MAX_WINDOW: usize = 24;
/// A line longer than this is minified or generated, and scrambling it says nothing new.
const MAX_LINE_CHARS: usize = 200;
const MIN_DIFF_CHARS: usize = 120;
const MAX_DIFF_CHARS: usize = 2_400;
/// Lines with two or more distinct tokens; fewer and the result is barely scrambled.
const MIN_SHUFFLABLE_LINES: usize = 4;
const MAX_TRIES: usize = 8;

/// Why a record produced no row. Counted in the manifest, never silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    CarriageReturn,
    TooShort,
    LongLine,
    InvisibleFormat,
    TooFewShufflableLines,
    Unscrambled,
    DiffLength,
}

impl Skip {
    pub fn as_str(self) -> &'static str {
        match self {
            Skip::CarriageReturn => "carriage_return",
            Skip::TooShort => "file_shorter_than_window",
            Skip::LongLine => "line_over_max_chars",
            Skip::InvisibleFormat => "invisible_format_character",
            Skip::TooFewShufflableLines => "too_few_shufflable_lines",
            Skip::Unscrambled => "shuffle_left_it_readable",
            Skip::DiffLength => "diff_length_out_of_band",
        }
    }
}

/// Allowlisted records, by language, in keyed-hash order.
pub fn candidates<'a>(
    records: &'a [PoolRecord],
    allowed: &BTreeMap<String, String>,
    seed: u64,
) -> BTreeMap<LangId, Vec<&'a PoolRecord>> {
    let mut out: BTreeMap<LangId, Vec<(&PoolRecord, [u8; 32])>> = BTreeMap::new();
    for record in records {
        if !allowed.contains_key(&record.id) {
            continue;
        }
        let Some(lang) = record.language() else { continue };
        if !LANGUAGES.contains(&lang) {
            continue;
        }
        out.entry(lang)
            .or_default()
            .push((record, order_key(seed, SOURCE, &record.id)));
    }
    out.into_iter()
        .map(|(lang, mut v)| {
            v.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.id.cmp(&b.0.id)));
            (lang, v.into_iter().map(|(r, _)| r).collect())
        })
        .collect()
}

/// Shuffle the whitespace-separated tokens of `text`, keeping its indentation. `None` when it
/// has fewer than two distinct tokens, so no order of them differs from this one.
pub(crate) fn shuffle_tokens(text: &str, rng: &mut ChaCha20Rng) -> Option<String> {
    let indent_len = text.len() - text.trim_start().len();
    let (indent, rest) = text.split_at(indent_len);
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.iter().collect::<BTreeSet<_>>().len() < 2 {
        return None;
    }
    let mut shuffled = tokens.clone();
    for _ in 0..MAX_TRIES {
        shuffled.shuffle(rng);
        if shuffled != tokens {
            break;
        }
    }
    Some(format!("{indent}{}", shuffled.join(" ")))
}

/// The scrambled one-hunk diff for `record`, or why it has none.
pub fn render(record: &PoolRecord, seed: u64, invisible: &[(u32, u32)]) -> Result<String, Skip> {
    if record.source.contains('\r') {
        return Err(Skip::CarriageReturn);
    }
    let lines: Vec<&str> = record.source.lines().collect();
    let n = lines.len();
    if n < MIN_WINDOW {
        return Err(Skip::TooShort);
    }
    let mut rng = item_rng(seed, SOURCE, &record.id);
    let len = rng.random_range(MIN_WINDOW..=MAX_WINDOW).min(n);
    // The block the window centres on: one of the pool's own hunks (the lines its commit
    // touched), or a random line when the record carries none.
    let (block_start, block_end) = match record.hunks.as_deref() {
        Some(hunks) if !hunks.is_empty() => {
            let h = hunks[rng.random_range(0..hunks.len())];
            let s = (h.start.max(1) as usize - 1).min(n - 1);
            let e = (h.end as usize).clamp(s + 1, n);
            (s, e)
        }
        _ => {
            let s = rng.random_range(0..n);
            (s, (s + 1).min(n))
        }
    };
    let lo = (block_start + 1).saturating_sub(len);
    let hi = block_start.min(n - len);
    let window_start = rng.random_range(lo.min(hi)..=hi);
    let texts: Vec<String> = lines[window_start..window_start + len]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    if texts.iter().any(|t| t.chars().count() > MAX_LINE_CHARS) {
        return Err(Skip::LongLine);
    }
    if texts.iter().any(|t| has_char_in(t, invisible)) {
        return Err(Skip::InvisibleFormat);
    }
    let mut bs = block_start - window_start;
    let mut be = (block_end - window_start).min(len);
    // Keep one context line: a hunk that is all change is not one the corpus holds.
    if bs == 0 && be == len {
        be = len - 1;
    }
    if be <= bs {
        bs = be.saturating_sub(1);
    }
    let marked = mark_block(texts, bs, be, draw_shape(&mut rng));

    let mut shufflable = 0usize;
    let mut changed = 0usize;
    let mut body: Vec<Line> = Vec::with_capacity(marked.len());
    for line in marked {
        match shuffle_tokens(&line.text, &mut rng) {
            Some(text) => {
                shufflable += 1;
                changed += usize::from(text != line.text);
                body.push(Line { mark: line.mark, text });
            }
            None => body.push(line),
        }
    }
    if shufflable < MIN_SHUFFLABLE_LINES {
        return Err(Skip::TooFewShufflableLines);
    }
    let before = body.clone();
    for _ in 0..MAX_TRIES {
        body.shuffle(&mut rng);
        if body != before {
            break;
        }
    }
    if body == before || changed * 2 < shufflable {
        return Err(Skip::Unscrambled);
    }
    let diff = hunk::render(window_start as u32 + 1, &body);
    if !(MIN_DIFF_CHARS..=MAX_DIFF_CHARS).contains(&diff.chars().count()) {
        return Err(Skip::DiffLength);
    }
    Ok(diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qd_mutate::span::LineSpan;

    const PY: &str = "\
import os
import sys


def load_config(path, defaults=None):
    defaults = defaults or {}
    with open(path) as handle:
        raw = handle.read()
    values = dict(defaults)
    for line in raw.splitlines():
        key, _, value = line.partition('=')
        values[key.strip()] = value.strip()
    return values


def main(argv):
    config = load_config(argv[1])
    print(config.get('name', 'unknown'))
    return 0
";

    fn record(id: &str, source: &str, hunks: Option<Vec<LineSpan>>) -> PoolRecord {
        serde_json::from_value(serde_json::json!({
            "id": id, "repo": "o/r", "path": "pkg/config.py", "source": source,
            "hunks": hunks.map(|h| h.iter().map(|s| serde_json::json!({
                "start_line": s.start, "end_line": s.end})).collect::<Vec<_>>()),
        }))
        .unwrap()
    }

    #[test]
    fn a_window_is_one_hunk_with_its_markers_and_indentation_kept_and_its_order_broken() {
        let rec = record("c:pkg/config.py", PY, Some(vec![LineSpan { start: 10, end: 12 }]));
        let mut rendered = 0;
        for seed in 0..30 {
            let Ok(diff) = render(&rec, seed, &[(0xFEFF, 0xFEFF)]) else { continue };
            rendered += 1;
            assert_eq!(diff.matches("@@ -").count(), 1, "{diff}");
            let body: Vec<&str> = diff.lines().skip(1).collect();
            assert!(body.iter().all(|l| crate::hunk::Mark::from_prefix(
                l.chars().next().unwrap_or('?')).is_some()), "{body:?}");
            // Every token is the file's own, and the lines no longer run in the file's order.
            let vocab: BTreeSet<&str> = PY.split_whitespace().collect();
            assert!(body.iter().flat_map(|l| l[1..].split_whitespace()).all(|t| vocab.contains(t)));
            let text: Vec<&str> = body.iter().map(|l| &l[1..]).collect();
            assert!(!PY.contains(&text.join("\n")), "{diff}");
        }
        assert!(rendered >= 20, "only {rendered} of 30 seeds rendered");
        assert_eq!(render(&rec, 3, &[]), render(&rec, 3, &[]));
    }

    #[test]
    fn every_reason_a_record_has_no_row_is_named() {
        let short = record("c:a.py", "x = 1\ny = 2\n", None);
        assert_eq!(render(&short, 0, &[]), Err(Skip::TooShort));
        let crlf = record("c:b.py", &PY.replace('\n', "\r\n"), None);
        assert_eq!(render(&crlf, 0, &[]), Err(Skip::CarriageReturn));
        let bom = record("c:c.py", &PY.replace("import sys", "import\u{FEFF} sys"), None);
        let reasons: BTreeSet<_> = (0..30)
            .filter_map(|s| render(&bom, s, &[(0xFEFF, 0xFEFF)]).err())
            .collect::<Vec<_>>()
            .into_iter()
            .map(Skip::as_str)
            .collect();
        assert!(reasons.contains("invisible_format_character"), "{reasons:?}");
        let flat = record("c:d.py", &"x\n".repeat(30), None);
        assert_eq!(render(&flat, 0, &[]), Err(Skip::TooFewShufflableLines));
    }

    #[test]
    fn candidates_are_the_allowlisted_files_of_the_four_languages_only() {
        let records = vec![
            record("c:pkg/config.py", PY, None),
            serde_json::from_value(serde_json::json!({
                "id": "c:main.swift", "repo": "o/s", "path": "main.swift", "source": PY})).unwrap(),
            serde_json::from_value(serde_json::json!({
                "id": "c:held.py", "repo": "o/h", "path": "held.py", "source": PY})).unwrap(),
        ];
        let allowed: BTreeMap<String, String> = ["c:pkg/config.py", "c:main.swift"]
            .iter()
            .map(|k| ((*k).to_string(), "mit".to_string()))
            .collect();
        let got = candidates(&records, &allowed, 0);
        assert_eq!(got.len(), 1);
        assert_eq!(got[&LangId::Python].len(), 1);
        assert_eq!(got[&LangId::Python][0].id, "c:pkg/config.py");
    }
}
