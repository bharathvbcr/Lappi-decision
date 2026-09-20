//! Derivation B of the span label: a textual before/after line diff, with the changed range read
//! off the hunk.
//!
//! This module deliberately knows **nothing** about the mutator. Its only inputs are two strings.
//! It shares the *convention* with [`crate::span`] — 1-based, inclusive both ends, over the
//! post-mutation file — and shares no code path with it. That is the whole point: agreement
//! between the two is evidence, and a shared helper would make it a tautology.

use crate::span::LineSpan;

/// Split for diffing, under the same convention [`crate::span::line_of`] counts by.
///
/// A text ending in `\n` yields a trailing empty element — the phantom final line. Dropping it here
/// and not there is precisely the off-by-one this crate exists to prevent, so it is kept.
fn lines_for_diff(text: &str) -> Vec<&str> {
    text.split('\n').collect()
}

/// **Derivation B.** The changed line range, in post-mutation coordinates.
///
/// Returns `None` when the two texts are identical — a mutation that changed nothing is not a
/// mutation, and an example claiming a span over an empty diff is a poisoned label.
///
/// The hunk is minimised from both ends (longest common prefix, then longest common suffix), which
/// is the same range a unified diff would print. A pure deletion leaves an empty after-range; its
/// span is the single line containing the join, matching derivation A's rule for an empty byte
/// range.
pub fn span_from_text_diff(before: &str, after: &str) -> Option<LineSpan> {
    if before == after {
        return None;
    }
    let b = lines_for_diff(before);
    let a = lines_for_diff(after);

    let mut prefix = 0usize;
    while prefix < b.len() && prefix < a.len() && b[prefix] == a[prefix] {
        prefix += 1;
    }

    let max_suffix = b.len().min(a.len()) - prefix;
    let mut suffix = 0usize;
    while suffix < max_suffix
        && b[b.len() - 1 - suffix] == a[a.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let changed_start = prefix; // 0-based, inclusive
    let changed_end = a.len() - suffix; // 0-based, exclusive

    if changed_end <= changed_start {
        // Pure deletion: nothing in the after-text belongs to the hunk. The span is the line the
        // deletion joined, clamped into the file. `a.len()` is at least 1 for any string, so the
        // clamp can only ever pull the line back to a real one.
        let line = u32::try_from(changed_start.min(a.len().saturating_sub(1)) + 1).ok()?;
        return Some(LineSpan::new(line, line));
    }

    let start = u32::try_from(changed_start + 1).ok()?;
    let end = u32::try_from(changed_end).ok()?;
    Some(LineSpan::new(start, end.max(start)))
}

/// A unified-style diff of `before` against `after`, for the example record.
///
/// Whole-file context is not carried: the record already holds both texts, and a diff here is for a
/// human reading a dropped example. Line numbers in the header are 1-based in each file's own
/// coordinates.
pub fn unified(before: &str, after: &str, context: usize) -> String {
    let b = lines_for_diff(before);
    let a = lines_for_diff(after);
    let mut prefix = 0usize;
    while prefix < b.len() && prefix < a.len() && b[prefix] == a[prefix] {
        prefix += 1;
    }
    let max_suffix = b.len().min(a.len()) - prefix;
    let mut suffix = 0usize;
    while suffix < max_suffix && b[b.len() - 1 - suffix] == a[a.len() - 1 - suffix] {
        suffix += 1;
    }
    let b_start = prefix.saturating_sub(context);
    let b_end = (b.len() - suffix + context).min(b.len());
    let a_start = prefix.saturating_sub(context);
    let a_end = (a.len() - suffix + context).min(a.len());

    let mut out = String::new();
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        b_start + 1,
        b_end.saturating_sub(b_start),
        a_start + 1,
        a_end.saturating_sub(a_start)
    ));
    for line in b.get(b_start..prefix).unwrap_or_default() {
        out.push_str(&format!(" {line}\n"));
    }
    for line in b.get(prefix..b.len() - suffix).unwrap_or_default() {
        out.push_str(&format!("-{line}\n"));
    }
    for line in a.get(prefix..a.len() - suffix).unwrap_or_default() {
        out.push_str(&format!("+{line}\n"));
    }
    for line in a.get(a.len() - suffix..a_end).unwrap_or_default() {
        out.push_str(&format!(" {line}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_texts_have_no_span() {
        assert_eq!(span_from_text_diff("a\nb\n", "a\nb\n"), None);
    }

    #[test]
    fn a_single_changed_line_is_that_line() {
        assert_eq!(
            span_from_text_diff("a\nb\nc\n", "a\nX\nc\n"),
            Some(LineSpan::new(2, 2))
        );
    }

    #[test]
    fn a_pure_deletion_is_the_join_line() {
        assert_eq!(
            span_from_text_diff("a\nb\nc\n", "a\nc\n"),
            Some(LineSpan::new(2, 2))
        );
    }

    #[test]
    fn an_insertion_is_the_inserted_line() {
        assert_eq!(
            span_from_text_diff("a\nc\n", "a\nb\nc\n"),
            Some(LineSpan::new(2, 2))
        );
    }

    #[test]
    fn a_change_on_the_last_line_without_a_trailing_newline() {
        assert_eq!(span_from_text_diff("a\nb", "a\nX"), Some(LineSpan::new(2, 2)));
    }

    #[test]
    fn deleting_the_whole_file_clamps_to_line_one() {
        assert_eq!(span_from_text_diff("a\nb\n", ""), Some(LineSpan::new(1, 1)));
    }

    #[test]
    fn a_multi_line_replacement_covers_every_changed_line() {
        assert_eq!(
            span_from_text_diff("a\nb\nc\nd\n", "a\nX\nY\nd\n"),
            Some(LineSpan::new(2, 3))
        );
    }
}
