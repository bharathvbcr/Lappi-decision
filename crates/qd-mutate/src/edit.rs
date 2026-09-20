//! Byte-range bookkeeping — the input to span derivation A.
//!
//! An [`EditSet`] is a sorted, non-overlapping set of byte replacements over the normalized source.
//! Applying it yields the post-mutation text **and** the post-mutation byte range the edits
//! occupy. Nothing here consults line numbers; the conversion to lines happens in [`crate::span`],
//! and the cross-check against the text diff happens in [`crate::diffspan`].

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EditError {
    #[error("edit range {start}..{end} is inverted")]
    Inverted { start: usize, end: usize },
    #[error("edit range {start}..{end} escapes a {len}-byte source")]
    OutOfBounds {
        start: usize,
        end: usize,
        len: usize,
    },
    #[error("edit range {start}..{end} splits a multi-byte character")]
    NotCharBoundary { start: usize, end: usize },
    #[error("edits {a} and {b} overlap")]
    Overlap { a: usize, b: usize },
    #[error("an edit set must contain at least one edit")]
    Empty,
    #[error("an edit set is capped at {cap} edits; this one has {found}")]
    TooManyEdits { cap: usize, found: usize },
}

/// A single byte replacement. `text` may be empty (a deletion) and `start == end` is allowed
/// (an insertion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

impl Edit {
    pub fn replace(start: usize, end: usize, text: impl Into<String>) -> Self {
        Edit {
            start,
            end,
            text: text.into(),
        }
    }

    pub fn delete(start: usize, end: usize) -> Self {
        Edit {
            start,
            end,
            text: String::new(),
        }
    }

    pub fn insert(at: usize, text: impl Into<String>) -> Self {
        Edit {
            start: at,
            end: at,
            text: text.into(),
        }
    }
}

/// `cosmetic.rename_local` is the only operator that emits many edits, and a local referenced more
/// than this many times inside one body is not a local worth renaming. A bound, because every
/// fan-out in this crate has one.
pub const MAX_EDITS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditSet {
    edits: Vec<Edit>,
}

/// The result of applying an edit set: the new text, and the byte range in **that** text which the
/// edits occupy. The range runs from the first edit's post-image start to the last edit's
/// post-image end, so a multi-site rename reports the whole affected region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl EditSet {
    /// Validate and sort. Overlapping edits are refused rather than merged: two operators that
    /// both wanted the same bytes is a bug in candidate generation, and silently picking one would
    /// emit a diff nobody chose.
    pub fn new(mut edits: Vec<Edit>) -> Result<Self, EditError> {
        if edits.is_empty() {
            return Err(EditError::Empty);
        }
        if edits.len() > MAX_EDITS {
            return Err(EditError::TooManyEdits {
                cap: MAX_EDITS,
                found: edits.len(),
            });
        }
        for e in &edits {
            if e.end < e.start {
                return Err(EditError::Inverted {
                    start: e.start,
                    end: e.end,
                });
            }
        }
        edits.sort_by_key(|e| (e.start, e.end));
        for pair in edits.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            // Touching is fine (a.end == b.start); genuine overlap is not. Two insertions at the
            // same point are an overlap too: their order would decide the output.
            if b.start < a.end || (a.start == b.start && a.end == b.end) {
                return Err(EditError::Overlap {
                    a: a.start,
                    b: b.start,
                });
            }
        }
        Ok(EditSet { edits })
    }

    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    /// The pre-image byte range the edits cover, for the hunk-intersection check.
    pub fn source_range(&self) -> (usize, usize) {
        let start = self.edits.first().map_or(0, |e| e.start);
        let end = self.edits.last().map_or(0, |e| e.end);
        (start, end)
    }

    /// Apply to `src`, returning the new text and the post-mutation byte range.
    ///
    /// Every boundary is checked against `src` here rather than trusted from the tree: tree-sitter
    /// node ranges are always on character boundaries, but an operator that did arithmetic on one
    /// may not be, and a span that splits a multi-byte character panics the first consumer that
    /// slices with it.
    pub fn apply(&self, src: &str) -> Result<Applied, EditError> {
        for e in &self.edits {
            if e.end > src.len() {
                return Err(EditError::OutOfBounds {
                    start: e.start,
                    end: e.end,
                    len: src.len(),
                });
            }
            if !src.is_char_boundary(e.start) || !src.is_char_boundary(e.end) {
                return Err(EditError::NotCharBoundary {
                    start: e.start,
                    end: e.end,
                });
            }
        }

        let mut out = String::with_capacity(src.len() + 64);
        let mut cursor = 0usize;
        let mut first_post_start: Option<usize> = None;
        let mut last_post_end = 0usize;

        for e in &self.edits {
            out.push_str(&src[cursor..e.start]);
            let post_start = out.len();
            out.push_str(&e.text);
            let post_end = out.len();
            if first_post_start.is_none() {
                first_post_start = Some(post_start);
            }
            last_post_end = post_end;
            cursor = e.end;
        }
        out.push_str(&src[cursor..]);

        let start = first_post_start.unwrap_or(0);
        Ok(Applied {
            text: out,
            start,
            end: last_post_end.max(start),
        })
    }
}

/// Expand `[start, end)` outward to whole lines, but only across whitespace.
///
/// Block-body replacements need this. Without it, an edit that begins immediately after a `{`
/// reports the signature line as changed, while the text diff — which sees the signature line
/// unchanged — reports the line below. The two derivations would disagree on every stub, and the
/// disagreement would be the *checker's* fault rather than the mutator's. Snapping to lines makes
/// the two agree for the right reason: the edit really does replace whole lines.
///
/// Non-whitespace on either side leaves that end alone, so an in-line edit (`<` to `<=`) stays
/// in-line and both derivations report the single line it sits on.
pub fn snap_to_lines(src: &str, start: usize, end: usize) -> (usize, usize) {
    let bytes = src.as_bytes();

    let mut new_start = start;
    while new_start > 0 {
        let b = bytes[new_start - 1];
        if b == b'\n' {
            break;
        }
        if b == b' ' || b == b'\t' {
            new_start -= 1;
            continue;
        }
        // Real code before the edit on this line: leave the start where it is.
        new_start = start;
        break;
    }

    let mut probe = end;
    let mut new_end = end;
    while probe < bytes.len() {
        let b = bytes[probe];
        if b == b'\n' {
            new_end = probe + 1;
            break;
        }
        if b == b' ' || b == b'\t' {
            probe += 1;
            continue;
        }
        break;
    }
    if probe == bytes.len() && probe > end {
        // Trailing whitespace to EOF with no newline: take it, so the span ends at the last line.
        new_end = bytes.len();
    }

    (new_start, new_end)
}

/// The leading whitespace of the line containing `offset`.
pub fn indent_of_line(src: &str, offset: usize) -> String {
    let clamped = offset.min(src.len());
    let line_start = src[..clamped].rfind('\n').map_or(0, |i| i + 1);
    src[line_start..]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_replacement_reports_its_post_image_range() {
        let set = EditSet::new(vec![Edit::replace(2, 4, "XYZ")]).expect("valid");
        let applied = set.apply("aabbcc").expect("applies");
        assert_eq!(applied.text, "aaXYZcc");
        assert_eq!(&applied.text[applied.start..applied.end], "XYZ");
    }

    #[test]
    fn multiple_edits_report_the_whole_affected_region() {
        let set = EditSet::new(vec![Edit::replace(0, 1, "X"), Edit::replace(4, 5, "Y")])
            .expect("valid");
        let applied = set.apply("abcde").expect("applies");
        assert_eq!(applied.text, "XbcdY");
        assert_eq!(applied.start, 0);
        assert_eq!(applied.end, 5);
    }

    #[test]
    fn overlapping_edits_are_refused_not_merged() {
        let err = EditSet::new(vec![Edit::replace(0, 3, "X"), Edit::replace(2, 5, "Y")])
            .expect_err("overlap");
        assert!(matches!(err, EditError::Overlap { .. }));
    }

    #[test]
    fn an_edit_splitting_a_multibyte_character_is_refused() {
        let src = "日本語";
        let set = EditSet::new(vec![Edit::replace(1, 2, "x")]).expect("constructs");
        assert!(matches!(
            set.apply(src),
            Err(EditError::NotCharBoundary { .. })
        ));
    }

    #[test]
    fn snap_expands_across_whitespace_only() {
        let src = "fn f() {\n    let x = 1;\n}\n";
        let start = src.find("let").expect("fixture");
        let end = src.find(";").expect("fixture") + 1;
        let (s, e) = snap_to_lines(src, start, end);
        assert_eq!(&src[s..e], "    let x = 1;\n");
    }

    #[test]
    fn snap_leaves_an_inline_edit_alone() {
        let src = "if a < b { x }\n";
        let start = src.find('<').expect("fixture");
        let (s, e) = snap_to_lines(src, start, start + 1);
        assert_eq!((s, e), (start, start + 1), "code on both sides: no snapping");
    }
}
