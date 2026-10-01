//! Source (a): English prose. SQuAD v2 paragraphs from train-split `qa.answer_span` titles,
//! wrapped and presented as a one-hunk diff -- a defect question asked of something that is not
//! code at all.
//!
//! Disjoint from the OOD suite's prose by source: the suite draws MMLU / CommonsenseQA **val**
//! questions and presents them raw under `notes.txt`. These are SQuAD paragraphs, in the
//! corpus's own hunk shape, under paths that are never `notes.txt` and are as often a code
//! file's as a document's -- so the path alone does not predict the label.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read};

use rand::Rng;
use rand::seq::IndexedRandom;
use serde::Deserialize;

use crate::hunk::{self, draw_shape, has_char_in, item_rng, mark_block, order_key, Shape};

pub const SOURCE: &str = "prose";
pub const LANGUAGE: &str = "prose";

/// The corpus's diffs run 159..1,435 characters between their 5th and 95th percentiles
/// (`commitpackft-corpus-v2`, 2026-09-30); a paragraph outside a similar band would make length
/// a cue. SQuAD train paragraphs: 411 / 679 / 1,135 at the 10th / 50th / 90th.
pub const MIN_CHARS: usize = 240;
pub const MAX_CHARS: usize = 1_400;
/// At most this many paragraphs per article, so ~830 rows spread over ~335 titles rather
/// than concentrating on the articles whose paragraphs hash first.
pub const PER_TITLE_CAP: usize = 3;
/// SQuAD v2 train is 130,319 rows; a file this many times larger is not that file.
const MAX_LINES: usize = 2_000_000;
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

const PATHS: [&str; 8] = [
    "docs/{slug}.md",
    "docs/guides/{slug}.md",
    "doc/{slug}.rst",
    "{slug}/README.md",
    "src/{snake}/__init__.py",
    "internal/{snake}/doc.go",
    "src/{snake}.rs",
    "src/{camel}.ts",
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Paragraph {
    pub title: String,
    pub context: String,
}

#[derive(Deserialize)]
struct SquadLine {
    title: String,
    context: String,
}

/// Every distinct paragraph of an allowlisted title that can become a row, and what was skipped.
pub fn read_paragraphs(
    reader: impl Read,
    allowed: &BTreeMap<String, String>,
    invisible: &[(u32, u32)],
) -> Result<(Vec<Paragraph>, BTreeMap<String, u64>), String> {
    let mut skipped: BTreeMap<String, u64> = BTreeMap::new();
    let mut seen: BTreeSet<Paragraph> = BTreeSet::new();
    let mut eligible: Vec<Paragraph> = Vec::new();
    let mut buffered = BufReader::new(reader);
    let mut buf = Vec::new();
    let mut n_lines = 0usize;
    loop {
        buf.clear();
        let read = buffered
            .read_until(b'\n', &mut buf)
            .map_err(|e| format!("reading SQuAD: {e}"))?;
        if read == 0 {
            break;
        }
        n_lines += 1;
        if n_lines > MAX_LINES {
            return Err(format!("SQuAD file holds more than {MAX_LINES} lines"));
        }
        if buf.len() > MAX_LINE_BYTES {
            return Err(format!("SQuAD line {n_lines} is {} bytes", buf.len()));
        }
        let text = std::str::from_utf8(&buf)
            .map_err(|e| format!("SQuAD line {n_lines} is not UTF-8: {e}"))?
            .trim();
        if text.is_empty() {
            continue;
        }
        let row: SquadLine = serde_json::from_str(text)
            .map_err(|e| format!("SQuAD line {n_lines} is not a row: {e}"))?;
        if !allowed.contains_key(&row.title) {
            // Not counted by title: the allowlist step already counted every excluded title by
            // the reason it was excluded, and that count is in the manifest.
            continue;
        }
        let paragraph = Paragraph {
            title: row.title,
            context: row.context,
        };
        // Several questions share one paragraph; it is one candidate, counted once.
        if !seen.insert(paragraph.clone()) {
            continue;
        }
        let chars = paragraph.context.chars().count();
        let reason = if paragraph.context.contains(['\n', '\r']) {
            Some("multi_line_paragraph")
        } else if has_char_in(&paragraph.context, invisible) {
            Some("invisible_format_character")
        } else if chars < MIN_CHARS {
            Some("shorter_than_min_chars")
        } else if chars > MAX_CHARS {
            Some("longer_than_max_chars")
        } else {
            None
        };
        match reason {
            Some(reason) => *skipped.entry(reason.to_string()).or_insert(0) += 1,
            None => eligible.push(paragraph),
        }
    }
    Ok((eligible, skipped))
}

/// `n` paragraphs in keyed-hash order, at most [`PER_TITLE_CAP`] per title.
pub fn select(paragraphs: &[Paragraph], n: usize, seed: u64) -> Result<Vec<&Paragraph>, String> {
    let mut order: Vec<(&Paragraph, [u8; 32])> = paragraphs
        .iter()
        .map(|p| (p, order_key(seed, SOURCE, &format!("{}\u{1f}{}", p.title, p.context))))
        .collect();
    order.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
    let mut per_title: BTreeMap<&str, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(n);
    for (p, _) in order {
        if out.len() == n {
            break;
        }
        let used = per_title.entry(p.title.as_str()).or_insert(0);
        if *used >= PER_TITLE_CAP {
            continue;
        }
        *used += 1;
        out.push(p);
    }
    if out.len() < n {
        return Err(format!(
            "only {} eligible paragraphs at {PER_TITLE_CAP} per title; {n} were asked for",
            out.len()
        ));
    }
    Ok(out)
}

/// Lower-case ASCII words of `title`, joined by `-`; `section` when nothing survives.
fn slug(title: &str) -> String {
    let mut out = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if c.is_ascii() && !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "section".to_string() } else { out }
}

fn camel(slug: &str) -> String {
    let mut out = String::new();
    for (i, part) in slug.split('-').enumerate() {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            if i == 0 {
                out.push(first);
            } else {
                out.push(first.to_ascii_uppercase());
            }
            out.extend(chars);
        }
    }
    out
}

/// Greedy word wrap at `width` characters; a word longer than `width` sits on its own line.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.chars().count() + 1 + word.chars().count() > width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// `(path, diff)` for one paragraph, every choice drawn from the paragraph's own RNG.
pub fn render(p: &Paragraph, seed: u64) -> (String, String) {
    let mut rng = item_rng(seed, SOURCE, &format!("{}\u{1f}{}", p.title, p.context));
    let width = rng.random_range(60..=100usize);
    let lines = wrap(&p.context, width);
    let n = lines.len();
    let shape = draw_shape(&mut rng);
    // At least one context line, and a Replace needs two lines to have two halves.
    let max_k = (n - 1).clamp(1, 4);
    let k = match shape {
        Shape::Replace if max_k >= 2 => rng.random_range(2..=max_k),
        _ => rng.random_range(1..=max_k.min(3)),
    };
    let start = rng.random_range(0..=n - k);
    let body = mark_block(lines, start, start + k, shape);
    let diff = hunk::render(rng.random_range(1..=400u32), &body);
    let s = slug(&p.title);
    let pattern = PATHS.choose(&mut rng).copied().unwrap_or(PATHS[0]);
    let path = pattern
        .replace("{slug}", &s)
        .replace("{snake}", &s.replace('-', "_"))
        .replace("{camel}", &camel(&s));
    (path, diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARA: &str = "The quick survey of the northern valley found that the river had moved \
        nearly two kilometres east over a century, cutting new channels through farmland and \
        leaving oxbow lakes where the old meanders once ran. Local records describe floods in \
        1890 and 1921 that reshaped the banks, and the survey team compared those accounts with \
        aerial photographs taken after the second world war.";

    fn allowed(titles: &[&str]) -> BTreeMap<String, String> {
        titles
            .iter()
            .map(|t| ((*t).to_string(), format!("squad-title:{t}")))
            .collect()
    }

    fn line(title: &str, context: &str) -> String {
        format!(
            "{}\n",
            serde_json::json!({"id": "q", "title": title, "context": context, "question": "?"})
        )
    }

    #[test]
    fn only_allowlisted_titles_are_read_and_each_paragraph_once() {
        let file = [
            line("Kept", PARA),
            line("Kept", PARA),
            line("Held", &PARA.replace("river", "canal")),
            line("Kept", "too short"),
            line("Kept", &format!("{PARA}\u{200B}")),
        ]
        .concat();
        let (paras, skipped) =
            read_paragraphs(file.as_bytes(), &allowed(&["Kept"]), &[(0x200B, 0x200F)]).unwrap();
        assert_eq!(paras.len(), 1);
        assert_eq!(paras[0].title, "Kept");
        assert_eq!(skipped.get("shorter_than_min_chars"), Some(&1));
        assert_eq!(skipped.get("invisible_format_character"), Some(&1));
        assert!(paras.iter().all(|p| p.title != "Held"));
    }

    #[test]
    fn a_paragraph_renders_as_one_wrapped_hunk_under_a_path_that_is_never_notes_txt() {
        let p = Paragraph {
            title: "Super_Bowl_50".to_string(),
            context: PARA.to_string(),
        };
        for seed in 0..40 {
            let (path, diff) = render(&p, seed);
            assert!(diff.starts_with("@@ -"), "{diff}");
            assert_eq!(diff.matches("@@ -").count(), 1);
            assert!(!path.contains("notes") && !path.ends_with(".txt"), "{path}");
            assert!(path.contains("super"), "{path}");
            let body: Vec<&str> = diff.lines().skip(1).collect();
            assert!(body.iter().all(|l| l.len() <= 101), "{body:?}");
            assert!(body.iter().any(|l| l.starts_with(' ')), "no context line: {body:?}");
            assert!(body.iter().any(|l| !l.starts_with(' ')), "no change: {body:?}");
            let words: String = body.iter().map(|l| &l[1..]).collect::<Vec<_>>().join(" ");
            assert_eq!(words, PARA.split_whitespace().collect::<Vec<_>>().join(" "));
        }
        assert_eq!(render(&p, 7), render(&p, 7));
    }

    #[test]
    fn selection_is_capped_per_title_and_refuses_a_short_supply() {
        let paras: Vec<Paragraph> = (0..10)
            .map(|i| Paragraph {
                title: if i < 8 { "A".into() } else { "B".into() },
                context: format!("{PARA} {i}"),
            })
            .collect();
        let got = select(&paras, 5, 0).unwrap();
        assert_eq!(got.iter().filter(|p| p.title == "A").count(), 3);
        assert_eq!(got.iter().filter(|p| p.title == "B").count(), 2);
        assert!(select(&paras, 6, 0).unwrap_err().contains("only 5 eligible"));
        assert_eq!(slug("Beyoncé"), "beyonc");
        assert_eq!(slug("Super_Bowl_50"), "super-bowl-50");
        assert_eq!(camel("super-bowl-50"), "superBowl50");
        assert_eq!(slug("…"), "section");
    }
}
