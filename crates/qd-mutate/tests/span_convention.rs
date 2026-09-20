//! The span convention, round-tripped through both derivations and through the runtime's shape.
//!
//! `src/span.rs` promises this file exists: *"that is the same convention the runtime's `span` slot
//! returns; `tests/span_convention.rs` round-trips a known span through both to keep the two from
//! drifting apart."* Three things are asserted here and nowhere else.
//!
//! 1. **The two derivations agree on a known span.** Byte bookkeeping and a textual line diff are
//!    different code paths on purpose; a shared helper would make their agreement a tautology.
//! 2. **The convention is 1-based and inclusive on both ends, over the post-mutation file.**
//!    Asserted against hand-counted line numbers, not against another function in this crate.
//! 3. **The wire shape matches the runtime's.** `docs/schema-api.md` and `qd-runtime`'s `SpanValue`
//!    both spell it `{start_line, end_line}`, and `SpanValue` carries `deny_unknown_fields`. A span
//!    this crate emits must deserialize there. `qd-runtime` is not a dependency of this crate — it
//!    is another lane's crate and coupling the two builds would be worse than the duplication — so
//!    the runtime's declared shape is mirrored here and the mirror is what the test binds.

use qd_mutate::diffspan::span_from_text_diff;
use qd_mutate::span::{line_of, lines_in_span, span_from_byte_range, total_lines, LineSpan};

/// A byte-for-byte mirror of `qd_runtime::schema::SpanValue`, including `deny_unknown_fields`.
///
/// If the runtime's declaration changes, this test keeps passing and the two drift — so the mirror
/// is small, obvious, and named after what it mirrors.
#[derive(Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSpanValue {
    start_line: usize,
    end_line: usize,
}

const FIXTURE: &str = "\
fn one() {
    let a = 1;
    let b = 2;
    a + b
}
";

#[test]
fn a_known_span_reaches_the_same_answer_by_both_routes() {
    // Replace line 3 (`    let b = 2;`) with two different lines. Counted by hand, the fixture's
    // lines are 1 `fn one() {`, 2 `    let a = 1;`, 3 `    let b = 2;`, 4 `    a + b`, 5 `}`, 6 "".
    let start = FIXTURE.find("    let b").expect("fixture");
    let end = FIXTURE.find("    a + b").expect("fixture");
    assert_eq!(&FIXTURE[start..end], "    let b = 2;\n");

    let replacement = "    let b = 20;\n    let c = 3;\n";
    let after = format!("{}{replacement}{}", &FIXTURE[..start], &FIXTURE[end..]);

    let from_bytes = span_from_byte_range(&after, start, start + replacement.len());
    let from_diff = span_from_text_diff(FIXTURE, &after).expect("the texts differ");

    assert_eq!(from_bytes, from_diff, "the two derivations disagree");
    assert_eq!(
        from_bytes,
        LineSpan::new(3, 4),
        "hand-counted: the change occupies post-mutation lines 3 and 4"
    );
}

#[test]
fn a_replacement_that_rewrites_a_line_to_itself_is_a_disagreement_and_must_stay_one() {
    // This is the asymmetry the two derivations exist to catch, pinned so nobody "fixes" it.
    //
    // The edit's byte range covers line 3 and line 4, but line 3 comes out textually identical, so
    // the only line that *changed* is 4. Derivation A answers 3..=4 because that is what the
    // mutator wrote; derivation B answers 4..=4 because that is what the text shows. B is the
    // better label and A is not wrong about what it did — they are answering slightly different
    // questions, and the generator's response is to **drop the example**, never to pick a winner.
    //
    // Teaching A to trim unchanged leading lines would make it converge on B's algorithm, and two
    // derivations that share an algorithm are one derivation with extra steps. Their independence
    // is the whole evidentiary value; the drop is the price and it is the right price.
    let start = FIXTURE.find("    let b").expect("fixture");
    let end = FIXTURE.find("    a + b").expect("fixture");
    let replacement = "    let b = 2;\n    let c = 3;\n";
    let after = format!("{}{replacement}{}", &FIXTURE[..start], &FIXTURE[end..]);

    let from_bytes = span_from_byte_range(&after, start, start + replacement.len());
    let from_diff = span_from_text_diff(FIXTURE, &after).expect("the texts differ");
    assert_eq!(from_bytes, LineSpan::new(3, 4), "what the mutator wrote");
    assert_eq!(from_diff, LineSpan::new(4, 4), "what the text shows changed");
    assert_ne!(
        from_bytes, from_diff,
        "if these ever agree, one derivation has started copying the other"
    );
}

#[test]
fn the_span_is_one_based_and_inclusive_on_both_ends() {
    let span = LineSpan::new(3, 4);
    let lines = lines_in_span(FIXTURE, span).expect("inside the file");
    assert_eq!(
        lines,
        vec!["    let b = 2;", "    a + b"],
        "an inclusive end means line 4 is in the span, not one past it"
    );
    assert_eq!(span.line_count(), 2);

    // 1-based: the first line is 1, never 0, and a span starting at 0 is not well-formed.
    assert_eq!(line_of(FIXTURE, 0), 1);
    assert!(!LineSpan::new(0, 1).is_well_formed(total_lines(FIXTURE)));
    assert!(LineSpan::new(1, 1).is_well_formed(total_lines(FIXTURE)));
}

#[test]
fn a_single_line_span_has_equal_ends_rather_than_a_zero_width() {
    let start = FIXTURE.find("let a = 1").expect("fixture");
    let after = FIXTURE.replacen("let a = 1", "let a = 9", 1);
    let from_bytes = span_from_byte_range(&after, start, start + "let a = 9".len());
    assert_eq!(from_bytes, LineSpan::new(2, 2));
    assert_eq!(from_bytes, span_from_text_diff(FIXTURE, &after).expect("differ"));
    assert_eq!(from_bytes.line_count(), 1, "an inclusive one-line span counts 1");
}

#[test]
fn the_wire_shape_is_the_one_the_runtime_declares() {
    let span = LineSpan::new(41, 47);
    let json = serde_json::to_string(&span).expect("serialises");
    assert_eq!(
        json, r#"{"start_line":41,"end_line":47}"#,
        "the runtime's SpanValue has deny_unknown_fields; `start`/`end` would be rejected there"
    );

    // Forwards: a span this crate emits deserializes as the runtime's value.
    let as_runtime: RuntimeSpanValue = serde_json::from_str(&json).expect("the runtime can read it");
    assert_eq!(
        as_runtime,
        RuntimeSpanValue {
            start_line: 41,
            end_line: 47
        }
    );

    // Backwards: the runtime's answer — the literal example from `docs/schema-api.md` — reads back
    // as a `LineSpan` with the same numbers.
    let from_runtime: LineSpan =
        serde_json::from_str(r#"{"start_line": 41, "end_line": 47}"#).expect("reads back");
    assert_eq!(from_runtime, span);
}

#[test]
fn the_old_field_names_are_refused_rather_than_silently_defaulted() {
    // A `{start, end}` row is the shape this crate used to emit. It must fail loudly, because a
    // span that quietly deserialized to 0..0 would point every example at line zero.
    let parsed: Result<LineSpan, _> = serde_json::from_str(r#"{"start":41,"end":47}"#);
    assert!(parsed.is_err(), "a stale span shape must not parse");
}

#[test]
fn a_trailing_newline_and_its_absence_agree_on_the_last_line() {
    // "Last line of file with no trailing newline | Span end runs past EOF" — hardening §1.
    let with_newline = "a\nb\n";
    let without = "a\nb";
    assert_eq!(total_lines(with_newline), 3, "the phantom final line is counted");
    assert_eq!(total_lines(without), 2);

    let after_with = "a\nX\n";
    let after_without = "a\nX";
    for (before, after) in [(with_newline, after_with), (without, after_without)] {
        let start = 2usize;
        let end = start + 1;
        let from_bytes = span_from_byte_range(after, start, end);
        let from_diff = span_from_text_diff(before, after).expect("differ");
        assert_eq!(from_bytes, from_diff, "disagreement on {before:?}");
        assert_eq!(from_bytes, LineSpan::new(2, 2));
        assert!(
            from_bytes.is_well_formed(total_lines(after)),
            "the span end ran past EOF on {after:?}"
        );
    }
}
