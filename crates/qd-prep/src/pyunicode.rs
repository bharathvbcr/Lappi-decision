//! Python's `str.lower()` and `re.findall(r"\w+", ...)` over a `str`, evaluated in Rust.
//!
//! `qd_train.replay.word_ngrams` is `re.findall(r"\w+", text.lower())`. Neither half has an
//! exact std equivalent: Rust's `char::is_alphanumeric` counts `Other_Alphabetic` code points
//! (Devanagari vowel signs, circled letters) that Python's `\w` does not, `str::to_lowercase`
//! carries Rust's Unicode version rather than Python's, and both sides apply a context rule
//! to U+03A3 that reads two properties of their own tables. So nothing here consults Rust's
//! Unicode data: every answer comes from [`crate::pyunicode_tables`], read off CPython's own
//! `re` and `str.lower()` by `tools/qd_prep_unicode_tables.py`, under the Unicode version it
//! names ([`unicode_version`]).
//!
//! What cannot be made to agree is refused, not approximated: a code point that version
//! calls unassigned (`Cn`) is one a later Unicode version may make a word character or give a
//! lower-case mapping, so [`check_assigned`] names it and the caller stops.

use crate::pyunicode_tables::{
    LOWER, SIGMA_CASED, SIGMA_IGNORABLE, UNASSIGNED, UNIDATA_VERSION, WORD,
};

/// U+03A3 GREEK CAPITAL LETTER SIGMA, the one code point `str.lower()` maps by context.
const CAPITAL_SIGMA: char = '\u{03A3}';
const SMALL_SIGMA: char = '\u{03C3}';
const FINAL_SIGMA: char = '\u{03C2}';
const UNUSED: u32 = u32::MAX;

/// The `unicodedata.unidata_version` the tables were read under.
pub fn unicode_version() -> &'static str {
    UNIDATA_VERSION
}

fn in_ranges(c: char, ranges: &[(u32, u32)]) -> bool {
    let c = u32::from(c);
    // The last range starting at or below `c`, if any, contains it or nothing does.
    let after = ranges.partition_point(|&(lo, _)| lo <= c);
    after > 0 && ranges[after - 1].1 >= c
}

/// `re.fullmatch(r"\w", ch)` under the tables' Python.
pub fn is_word(c: char) -> bool {
    in_ranges(c, WORD)
}

/// `unicodedata.category(ch) == "Cn"` under the tables' Unicode version.
pub fn is_unassigned(c: char) -> bool {
    in_ranges(c, UNASSIGNED)
}

/// The first code point of `text` the tables' Unicode version leaves unassigned, as a
/// refusal: its `\w` and `lower()` answers are the ones a newer Python may give differently.
pub fn check_assigned(text: &str) -> Result<(), String> {
    match text.char_indices().find(|&(_, c)| is_unassigned(c)) {
        None => Ok(()),
        Some((at, c)) => Err(format!(
            "U+{:04X} at byte {at} is unassigned in Unicode {UNIDATA_VERSION}, the version \
             these word and lower-case tables were read under; a newer Python may treat it \
             differently, so this text is refused rather than tokenised by a guess",
            u32::from(c)
        )),
    }
}

/// `str.lower()`'s mapping for one code point other than U+03A3, pushed onto `out`.
fn push_lower(c: char, out: &mut String) {
    let key = u32::from(c);
    match LOWER.binary_search_by_key(&key, |&(k, _)| k) {
        Err(_) => out.push(c),
        Ok(i) => {
            for &m in LOWER[i].1.iter().take_while(|&&m| m != UNUSED) {
                // The generator wrote only code points `str.lower()` returned.
                out.push(char::from_u32(m).unwrap_or(char::REPLACEMENT_CHARACTER));
            }
        }
    }
}

/// CPython's `handle_capital_sigma`: U+03A3 at `i` is final when the nearest code point
/// before it that is not case-ignorable is cased, and the nearest after it that is not
/// case-ignorable is not cased (or there is none). Both read the ORIGINAL text.
fn sigma_is_final(chars: &[char], i: usize) -> bool {
    let ignorable = |c: char| in_ranges(c, SIGMA_IGNORABLE);
    let cased = |c: char| in_ranges(c, SIGMA_CASED);
    let before = chars[..i].iter().rev().find(|&&c| !ignorable(c));
    if !before.is_some_and(|&c| cased(c)) {
        return false;
    }
    match chars[i + 1..].iter().find(|&&c| !ignorable(c)) {
        None => true,
        Some(&c) => !cased(c),
    }
}

/// `text.lower()`, as CPython computes it.
pub fn lower(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == CAPITAL_SIGMA {
            out.push(if sigma_is_final(&chars, i) {
                FINAL_SIGMA
            } else {
                SMALL_SIGMA
            });
        } else {
            push_lower(c, &mut out);
        }
    }
    out
}

/// `re.findall(r"\w+", text)`: the maximal runs of word characters, in order.
pub fn words(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (at, c) in text.char_indices() {
        match (is_word(c), start) {
            (true, None) => start = Some(at),
            (false, Some(s)) => {
                out.push(&text[s..at]);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(&text[s..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_are_sorted_and_disjoint() {
        for (name, table) in [
            ("WORD", WORD),
            ("SIGMA_IGNORABLE", SIGMA_IGNORABLE),
            ("SIGMA_CASED", SIGMA_CASED),
            ("UNASSIGNED", UNASSIGNED),
        ] {
            for (lo, hi) in table {
                assert!(lo <= hi, "{name}: ({lo:#x}, {hi:#x})");
            }
            for w in table.windows(2) {
                assert!(w[0].1 + 1 < w[1].0, "{name}: {:?} then {:?}", w[0], w[1]);
            }
        }
        assert!(LOWER.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(LOWER.iter().all(|(k, _)| *k != u32::from(CAPITAL_SIGMA)));
    }

    /// Expected values are CPython 3.14.7's (Unicode 16.0.0) `str.lower()` and
    /// `re.findall(r"\w+", ...)`, computed 2026-10-02.
    #[test]
    fn lower_and_words_match_python_on_the_hard_cases() {
        // U+0130 lowers to two code points, the second a combining dot (not \w).
        assert_eq!(lower("\u{0130}stanbul"), "i\u{0307}stanbul");
        assert_eq!(words(&lower("\u{0130}stanbul")), vec!["i", "stanbul"]);
        // Final sigma: after a cased letter, before a non-cased one or the end.
        assert_eq!(lower("ΟΔΟΣ"), "οδος");
        assert_eq!(lower("ΟΔΟΣ ΚΑΙ"), "οδος και");
        assert_eq!(lower("Σ"), "σ");
        assert_eq!(lower("ΑΣΑ"), "ασα");
        // Case-ignorable code points are skipped on both sides.
        assert_eq!(lower("Α'Σ'"), "α'ς'");
        // Devanagari vowel signs are Mc/Mn: Python's \w stops at them; Rust's
        // is_alphanumeric would not.
        assert_eq!(words("नमस्ते"), vec!["नमस", "त"]);
        // Circled letters are So (Other_Alphabetic in Rust): not \w in Python, but they lower.
        assert_eq!(lower("Ⓐ"), "ⓐ");
        assert!(words("Ⓐⓑ").is_empty());
        // Arabic-Indic digits, CJK and the underscore are \w.
        assert_eq!(words("x_y ٣٤ 漢字!"), vec!["x_y", "٣٤", "漢字"]);
        assert!(check_assigned("plain ascii, Ελληνικά, 漢字").is_ok());
        assert!(
            check_assigned("a\u{0378}b").is_err(),
            "U+0378 is unassigned"
        );
    }
}
