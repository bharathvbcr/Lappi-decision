//! Derivation A of the span label: the mutator's own byte bookkeeping, converted to lines.
//!
//! Lines are **1-based and inclusive on both ends**, counted over the **post-mutation** file. That
//! is the same convention the runtime's `span` slot returns; `tests/span_convention.rs` round-trips
//! a known span through both to keep the two from drifting apart.

use serde::{Deserialize, Serialize};

/// A 1-based, inclusive-on-both-ends line range.
///
/// The field **names on the wire** are `start_line` and `end_line`, not `start` and `end`. That is
/// not cosmetic: `docs/schema-api.md` specifies the runtime's `span` slot as
/// `{start_line, end_line}`, `qd-runtime`'s `SpanValue` declares exactly those two fields with
/// `#[serde(deny_unknown_fields)]`, and `python/qd_data/schema.py` documents the same pair. A row
/// emitted as `{"start": …, "end": …}` would fail to deserialize on all three surfaces — and the
/// span label is the one field in this crate's output whose whole value is that the consumer can
/// read it.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize, PartialOrd, Ord)]
pub struct LineSpan {
    #[serde(rename = "start_line")]
    pub start: u32,
    #[serde(rename = "end_line")]
    pub end: u32,
}

impl LineSpan {
    pub fn new(start: u32, end: u32) -> Self {
        LineSpan { start, end }
    }

    /// Both ranges are inclusive, so touching endpoints count as intersecting.
    pub fn intersects(&self, other: &LineSpan) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    pub fn line_count(&self) -> u32 {
        self.end.saturating_sub(self.start).saturating_add(1)
    }

    /// A span is well-formed when it is non-empty and inside a file of `total_lines` lines.
    pub fn is_well_formed(&self, total_lines: u32) -> bool {
        self.start >= 1 && self.end >= self.start && self.end <= total_lines
    }
}

impl std::fmt::Display for LineSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..={}", self.start, self.end)
    }
}

/// The 1-based line number of the line containing `offset`.
///
/// `offset` is a byte offset and is clamped to the text length, so an offset at EOF answers the
/// last line rather than panicking — the "last line with no trailing newline" case.
///
/// Counting newlines strictly *before* the offset means an offset sitting on a `\n` answers the
/// line that newline terminates, not the next one. That is what makes an edit whose replacement
/// ends in `\n` report the line it filled rather than the line after it.
pub fn line_of(text: &str, offset: usize) -> u32 {
    let clamped = offset.min(text.len());
    let newlines = text.as_bytes()[..clamped]
        .iter()
        .filter(|b| **b == b'\n')
        .count();
    // A file cannot have more lines than it has bytes plus one, so u32 is safe for any file this
    // crate will parse (bounded at 10 MiB by `parse::MAX_SOURCE_BYTES`). Saturating rather than
    // wrapping: a silently wrapped line number is exactly the failure this module exists to stop.
    u32::try_from(newlines).unwrap_or(u32::MAX - 1).saturating_add(1)
}

/// Total lines in `text`, under the same convention `line_of` uses.
///
/// A file ending in `\n` has a phantom final line, matching `line_of(text, text.len())`. Both
/// derivations agree on this or they disagree about every trailing edit.
pub fn total_lines(text: &str) -> u32 {
    line_of(text, text.len())
}

/// **Derivation A.** The post-mutation byte range `[start, end)` as a line span.
///
/// An empty range (`end <= start`) is a pure deletion: the span is the single line containing the
/// join. `diffspan` reaches the same answer from the text alone, by a different route.
pub fn span_from_byte_range(text: &str, start: usize, end: usize) -> LineSpan {
    let start_line = line_of(text, start);
    if end <= start {
        return LineSpan::new(start_line, start_line);
    }
    let end_line = line_of(text, end - 1);
    LineSpan::new(start_line, end_line.max(start_line))
}

/// Extract the 1-based inclusive line range `span` from `text`, terminator-free.
///
/// Used by tests and by the example writer to show what the span actually points at. Returns
/// `None` for a span that is not inside the text rather than clamping — a clamped span is a wrong
/// span that reads as a right one.
pub fn lines_in_span(text: &str, span: LineSpan) -> Option<Vec<&str>> {
    if !span.is_well_formed(total_lines(text)) {
        return None;
    }
    let all: Vec<&str> = text.split('\n').collect();
    let start = usize::try_from(span.start).ok()?.checked_sub(1)?;
    let end = usize::try_from(span.end).ok()?;
    all.get(start..end).map(<[&str]>::to_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_of_counts_newlines_before_the_offset() {
        let t = "a\nb\nc";
        assert_eq!(line_of(t, 0), 1);
        assert_eq!(line_of(t, 1), 1, "an offset on the newline is still line 1");
        assert_eq!(line_of(t, 2), 2);
        assert_eq!(line_of(t, 4), 3);
        assert_eq!(line_of(t, 999), 3, "clamped, not panicking");
    }

    #[test]
    fn a_trailing_newline_makes_a_phantom_final_line() {
        assert_eq!(total_lines("a\n"), 2);
        assert_eq!(total_lines("a"), 1);
        assert_eq!(total_lines(""), 1);
    }

    #[test]
    fn an_empty_range_is_the_line_it_sits_on() {
        let t = "aa\nbb\ncc\n";
        assert_eq!(span_from_byte_range(t, 3, 3), LineSpan::new(2, 2));
    }

    #[test]
    fn a_replacement_ending_in_a_newline_does_not_bleed_into_the_next_line() {
        // "aa\nXX\ncc\n" with [3, 6) == "XX\n" replaced. The changed line is 2, not 2..3.
        let t = "aa\nXX\ncc\n";
        assert_eq!(span_from_byte_range(t, 3, 6), LineSpan::new(2, 2));
    }

    #[test]
    fn multibyte_text_does_not_shift_the_line_count() {
        let t = "let 日本 = 1;\nlet 🎌 = 2;\nx\n";
        // The offset of the second line's first byte.
        let second = t.find("let 🎌").expect("fixture contains the second line");
        assert_eq!(line_of(t, second), 2);
    }
}
