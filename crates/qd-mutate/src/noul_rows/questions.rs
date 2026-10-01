//! Source (a), v2 form: short questions. SQuAD v2 **train-split** questions of the same
//! allowlisted titles v1's paragraphs come from, one to three of one paragraph's questions per
//! row, wrapped narrowly and presented as a one-hunk diff.
//!
//! v1's prose rows are whole paragraphs. The OOD suite's prose is a short MMLU / CommonsenseQA
//! question, and J4 abstained on 3 / 9 / 2 of its 60 -- below J1's 17, which had no noul rows
//! at all: supervision on one form sharpened the boundary against the other (Fable F2,
//! 2026-10-01). v2 keeps half the paragraphs and makes the other half short questions.
//!
//! Disjoint from the suite by source (SQuAD, not MMLU / CSQA) and by split (train titles only,
//! through the allowlist), and from the general **val** split by the same title split: a
//! question of a train title is never a val row. Paths are v1's, never the suite's `notes.txt`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read};

use rand::seq::SliceRandom;
use rand::Rng;
use serde::Deserialize;

use crate::hunk::{self, draw_shape, has_char_in, item_rng, mark_block, order_key, Shape};
use crate::prose::{path_for, wrap};

pub const FORM: &str = "question";
/// The key every question-row draw is derived under, distinct from the paragraphs' `prose`.
const KEY: &str = "prose-question";
/// Whitespace words per row, so every row carries a word 8-gram and the suite-disjointness
/// check (`qd_train.replay.decontaminate`) compares it rather than skipping it.
pub const MIN_ROW_WORDS: usize = 8;
const MIN_QUESTION_WORDS: usize = 3;
const MAX_QUESTION_CHARS: usize = 200;
/// Questions per row: one, one, two or three, drawn uniformly -- half the rows are a single
/// question, as every suite prose case is.
const QUESTIONS_PER_ROW: [usize; 4] = [1, 1, 2, 3];
const MAX_QUESTIONS_PER_ROW: usize = 3;
/// Narrower than the paragraphs' 60..=100, so a single question still spans two lines.
const MIN_WIDTH: usize = 30;
const MAX_WIDTH: usize = 70;
/// At most this many rows per title, as for paragraphs.
pub const PER_TITLE_CAP: usize = 3;
/// SQuAD v2 train is 130,319 rows; a file this many times larger is not that file.
const MAX_LINES: usize = 2_000_000;
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

/// One paragraph's usable questions, in file order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub title: String,
    pub context: String,
    pub questions: Vec<String>,
}

#[derive(Deserialize)]
struct SquadLine {
    title: String,
    context: String,
    question: String,
}

fn bump(map: &mut BTreeMap<String, u64>, reason: &str) {
    *map.entry(format!("{FORM}:{reason}")).or_insert(0) += 1;
}

/// Every allowlisted title's questions, grouped by paragraph, and what was skipped. A question
/// whose text (whitespace-normalised) appeared before is skipped, so no two rows share one.
pub fn read(
    reader: impl Read,
    allowed: &BTreeMap<String, String>,
    invisible: &[(u32, u32)],
) -> Result<(Vec<Group>, BTreeMap<String, u64>), String> {
    let mut skipped: BTreeMap<String, u64> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut groups: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
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
            .map_err(|e| format!("SQuAD line {n_lines} is not a question row: {e}"))?;
        if !allowed.contains_key(&row.title) {
            // Counted by the allowlist step, by the reason the title was excluded.
            continue;
        }
        let normalised = row.question.split_whitespace().collect::<Vec<_>>().join(" ");
        let reason = if row.question.contains(['\n', '\r']) {
            Some("multi_line")
        } else if has_char_in(&row.question, invisible) {
            Some("invisible_format_character")
        } else if normalised.split(' ').count() < MIN_QUESTION_WORDS || normalised.is_empty() {
            Some("fewer_words_than_min")
        } else if normalised.chars().count() > MAX_QUESTION_CHARS {
            Some("longer_than_max_chars")
        } else if !seen.insert(normalised.clone()) {
            Some("duplicate_text")
        } else {
            None
        };
        match reason {
            Some(r) => bump(&mut skipped, r),
            None => groups.entry((row.title, row.context)).or_default().push(normalised),
        }
    }
    let groups = groups
        .into_iter()
        .map(|((title, context), questions)| Group { title, context, questions })
        .collect();
    Ok((groups, skipped))
}

/// `(path, diff)` for one paragraph's questions, or why it has none.
pub fn render(g: &Group, seed: u64) -> Result<(String, String), &'static str> {
    let mut rng = item_rng(seed, KEY, &format!("{}\u{1f}{}", g.title, g.context));
    let mut questions = g.questions.clone();
    questions.shuffle(&mut rng);
    let mut k = QUESTIONS_PER_ROW[rng.random_range(0..QUESTIONS_PER_ROW.len())].min(questions.len());
    let words = |k: usize| questions[..k].iter().map(|q| q.split(' ').count()).sum::<usize>();
    while k < questions.len().min(MAX_QUESTIONS_PER_ROW) && words(k) < MIN_ROW_WORDS {
        k += 1;
    }
    if k == 0 || words(k) < MIN_ROW_WORDS {
        return Err("fewer_row_words_than_min");
    }
    let text = questions[..k].join(" ");
    let width = rng.random_range(MIN_WIDTH..=MAX_WIDTH);
    let mut lines = wrap(&text, width);
    if lines.len() < 2 {
        let longest = text.split(' ').map(|w| w.chars().count()).max().unwrap_or(0);
        lines = wrap(&text, longest.max(text.chars().count().div_ceil(2)));
    }
    if lines.len() < 2 {
        return Err("one_line_after_wrap");
    }
    let n = lines.len();
    let shape = draw_shape(&mut rng);
    let max_k = (n - 1).clamp(1, 4);
    let block = match shape {
        Shape::Replace if max_k >= 2 => rng.random_range(2..=max_k),
        _ => rng.random_range(1..=max_k.min(3)),
    };
    let start = rng.random_range(0..=n - block);
    let body = mark_block(lines, start, start + block, shape);
    let diff = hunk::render(rng.random_range(1..=400u32), &body);
    let path = path_for(&g.title, &mut rng);
    Ok((path, diff))
}

/// `n` rows from `groups` in keyed-hash order, at most [`PER_TITLE_CAP`] per title, each group
/// used once; and the groups that rendered no row, by reason.
#[allow(clippy::type_complexity)]
pub fn select(
    groups: &[Group],
    n: usize,
    seed: u64,
) -> Result<(Vec<(&Group, String, String)>, BTreeMap<String, u64>), String> {
    let mut order: Vec<(&Group, [u8; 32])> = groups
        .iter()
        .map(|g| (g, order_key(seed, KEY, &format!("{}\u{1f}{}", g.title, g.context))))
        .collect();
    order.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| a.0.title.cmp(&b.0.title))
            .then_with(|| a.0.context.cmp(&b.0.context))
    });
    let mut per_title: BTreeMap<&str, usize> = BTreeMap::new();
    let mut skipped: BTreeMap<String, u64> = BTreeMap::new();
    let mut out = Vec::with_capacity(n);
    for (g, _) in order {
        if out.len() == n {
            break;
        }
        let used = per_title.entry(g.title.as_str()).or_insert(0);
        if *used >= PER_TITLE_CAP {
            continue;
        }
        match render(g, seed) {
            Ok((path, diff)) => {
                *used += 1;
                out.push((g, path, diff));
            }
            Err(reason) => bump(&mut skipped, reason),
        }
    }
    if out.len() < n {
        return Err(format!(
            "only {} question rows at {PER_TITLE_CAP} per title; {n} were asked for",
            out.len()
        ));
    }
    Ok((out, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(title: &str, context: &str, question: &str) -> String {
        format!(
            "{}\n",
            serde_json::json!({"title": title, "context": context, "question": question})
        )
    }

    fn allowed(titles: &[&str]) -> BTreeMap<String, String> {
        titles.iter().map(|t| ((*t).to_string(), format!("squad-title:{t}"))).collect()
    }

    #[test]
    fn questions_are_read_from_allowlisted_titles_once_each_grouped_by_paragraph() {
        let file = [
            line("Kept", "P1", "In what year did the northern survey of the valley begin?"),
            line("Kept", "P1", "Who  funded the museum exhibit about the floods?"),
            line("Kept", "P2", "In what year did the northern survey of the valley begin?"),
            line("Kept", "P2", "Why?"),
            line("Kept", "P2", "Which\u{200B} village rebuilt further up the slope?"),
            line("Held", "P9", "What did the held-out article say about the river?"),
        ]
        .concat();
        let (groups, skipped) =
            read(file.as_bytes(), &allowed(&["Kept"]), &[(0x200B, 0x200F)]).unwrap();
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert_eq!(groups[0].questions, vec![
            "In what year did the northern survey of the valley begin?".to_string(),
            "Who funded the museum exhibit about the floods?".to_string(),
        ]);
        assert_eq!(skipped.get("question:duplicate_text"), Some(&1));
        assert_eq!(skipped.get("question:fewer_words_than_min"), Some(&1));
        assert_eq!(skipped.get("question:invisible_format_character"), Some(&1));
        assert!(groups.iter().all(|g| g.title != "Held"));
    }

    #[test]
    fn a_row_is_one_hunk_of_one_to_three_whole_questions_on_at_least_two_lines() {
        let g = Group {
            title: "Super_Bowl_50".to_string(),
            context: "ctx".to_string(),
            questions: vec![
                "Which team won Super Bowl 50 after the long season?".to_string(),
                "Where was the game played that year?".to_string(),
                "Who was named the most valuable player of the game?".to_string(),
            ],
        };
        let mut single = 0;
        for seed in 0..60 {
            let (path, diff) = render(&g, seed).unwrap();
            assert_eq!(diff.matches("@@ -").count(), 1, "{diff}");
            assert!(!path.contains("notes") && !path.ends_with(".txt"), "{path}");
            let body: Vec<&str> = diff.lines().skip(1).collect();
            assert!(body.len() >= 2, "{diff}");
            assert!(body.iter().all(|l| l.chars().count() <= MAX_WIDTH + 1), "{body:?}");
            assert!(body.iter().any(|l| l.starts_with(' ')), "no context line: {body:?}");
            let text = body.iter().map(|l| &l[1..]).collect::<Vec<_>>().join(" ");
            let used = g.questions.iter().filter(|q| text.contains(q.as_str())).count();
            let total: usize = g.questions.iter().filter(|q| text.contains(q.as_str()))
                .map(|q| q.len()).sum::<usize>() + used.saturating_sub(1);
            assert!((1..=3).contains(&used) && total == text.len(), "{text}");
            single += usize::from(used == 1);
        }
        assert!((15..=45).contains(&single), "{single} of 60 rows were one question");
        assert_eq!(render(&g, 4), render(&g, 4));
    }

    #[test]
    fn short_questions_are_joined_until_a_row_has_eight_words_or_refused() {
        let g = Group {
            title: "T".to_string(),
            context: "c".to_string(),
            questions: vec!["Who built it?".to_string(), "When exactly was it built?".to_string()],
        };
        for seed in 0..20 {
            let (_, diff) = render(&g, seed).unwrap();
            let words: usize = diff.lines().skip(1).map(|l| l[1..].split_whitespace().count())
                .sum();
            assert!(words >= MIN_ROW_WORDS, "{diff}");
        }
        let lone = Group { questions: vec!["Who built it?".to_string()], ..g };
        assert_eq!(render(&lone, 0), Err("fewer_row_words_than_min"));
    }

    #[test]
    fn selection_is_capped_per_title_and_refuses_a_short_supply() {
        let q = |i: usize| format!("Which of the eight surveys numbered {i} found the river?");
        let groups: Vec<Group> = (0..6)
            .map(|i| Group {
                title: if i < 5 { "A".into() } else { "B".into() },
                context: format!("c{i}"),
                questions: vec![q(i)],
            })
            .collect();
        let (got, _) = select(&groups, 4, 0).unwrap();
        assert_eq!(got.iter().filter(|(g, _, _)| g.title == "A").count(), 3);
        assert_eq!(got.iter().filter(|(g, _, _)| g.title == "B").count(), 1);
        assert!(select(&groups, 5, 0).unwrap_err().contains("only 4 question rows"));
    }
}
