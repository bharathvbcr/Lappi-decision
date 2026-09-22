"""Where does a span live in the text the model reads, when that text is a diff?

The span head points at a line of the context. In ``after`` mode the context IS the
post-image, so a span line is a line of it. In diff mode the context is the unified diff,
where the same source line appears as a ``+`` or context line at an unrelated offset --
which is why span supervision was suppressed in diff mode rather than pointed at the wrong
text, leaving no mode in which both heads are supervised.

The risk here is not that the function returns nothing. It is that it returns a plausible
offset for the wrong line: a mapping that counts ``-`` lines, or that reads the old-side
start from the hunk header, is off by a handful of lines and trains the pointer on a line
that is merely nearby. A falling span loss on a fabricated target looks exactly like
learning, so every test below checks the CONTENT at the returned offset, not just that one
came back.
"""

from __future__ import annotations

import pytest

from qd_train.mutate_adapter import (
    CONTEXT_AFTER,
    CONTEXT_DIFF,
    EmptyDiffContext,
    LineSpan,
    MalformedExample,
    MutateExample,
    SpanOutsideDiff,
    diff_offset_of_after_line,
    to_decision,
)

#: The shape `diffspan::unified` emits, taken from a real corpus row: one hunk, no file
#: headers, every line led by '@', ' ', '+' or '-', and a trailing newline.
REAL = (
    b"@@ -3,7 +3,7 @@\n"
    b' import "testing"\n'
    b" \n"
    b" func (g *Game) RollMany(runs int, pins int) {\n"
    b"-\tfor i := 0; i < runs; i++ {\n"
    b"+\tfor i := 0; i <= runs; i++ {\n"
    b" \t\tg.Roll(pins)\n"
    b" \t}\n"
    b" }\n"
)


def _line_at(diff: bytes, offset: int) -> bytes:
    return diff[offset:].split(b"\n", 1)[0]


def _tool():
    """``tools/rung0_real_run.py``, imported only by the tests that need it.

    It imports torch at module scope, so importing it at the top of this file would skip
    every test here on a machine without torch -- including the dozen that only need the
    adapter and have no reason to be gated on it.
    """
    import sys
    from pathlib import Path

    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import rung0_real_run

    return rung0_real_run


def test_the_mutated_line_maps_to_the_added_line_and_not_the_removed_one() -> None:
    """The real row this fixture came from: span 6..6, and after line 6 is the ``+`` line.

    A mapping that counted ``-`` lines toward the new side would land one line earlier, on
    the removed text -- which is the pre-image, the very thing the model is not shown.
    """
    off = diff_offset_of_after_line(REAL, 6)
    assert _line_at(REAL, off) == b"+\tfor i := 0; i <= runs; i++ {"


def test_every_line_of_the_hunk_maps_to_its_own_source_line() -> None:
    """Walked end to end, because an off-by-one shows up at one end and not in the middle."""
    expected = {
        3: b' import "testing"',
        4: b" ",
        5: b" func (g *Game) RollMany(runs int, pins int) {",
        6: b"+\tfor i := 0; i <= runs; i++ {",
        7: b" \t\tg.Roll(pins)",
        8: b" \t}",
        9: b" }",
    }
    for after_line, want in expected.items():
        assert _line_at(REAL, diff_offset_of_after_line(REAL, after_line)) == want, (
            f"after line {after_line} mapped to the wrong diff line"
        )


def test_a_line_before_the_hunk_is_refused_rather_than_clamped_to_its_first_line() -> None:
    """Line 1 is in the file but not in the diff: the hunk starts at 3. Clamping would give
    the pointer a target that is simply wrong, which trains it to point at hunk starts."""
    with pytest.raises(SpanOutsideDiff):
        diff_offset_of_after_line(REAL, 1)


def test_a_line_after_the_hunk_is_refused_too() -> None:
    with pytest.raises(SpanOutsideDiff):
        diff_offset_of_after_line(REAL, 99)


def test_the_new_side_start_is_read_and_not_the_old_side() -> None:
    """The two sides differ whenever the hunk adds or removes lines above the span. Reading
    ``-`` where ``+`` was meant is the single likeliest way to get this wrong, and on a
    balanced hunk like REAL it is invisible -- so the fixture here is deliberately skewed."""
    skewed = (
        b"@@ -10,3 +40,4 @@\n"
        b" keep\n"
        b"+added one\n"
        b"+added two\n"
        b" tail\n"
    )
    assert _line_at(skewed, diff_offset_of_after_line(skewed, 40)) == b" keep"
    assert _line_at(skewed, diff_offset_of_after_line(skewed, 41)) == b"+added one"
    assert _line_at(skewed, diff_offset_of_after_line(skewed, 42)) == b"+added two"
    assert _line_at(skewed, diff_offset_of_after_line(skewed, 43)) == b" tail"
    with pytest.raises(SpanOutsideDiff):
        diff_offset_of_after_line(skewed, 10)


def test_multiple_hunks_each_position_their_own_lines() -> None:
    """Both corpora emit exactly one hunk per diff today. The parser handles more because
    the generator may not always, and a second hunk silently inheriting the first hunk's
    counter would misplace every line in it."""
    two = (
        b"@@ -1,2 +1,2 @@\n"
        b" alpha\n"
        b"-beta\n"
        b"+BETA\n"
        b"@@ -20,2 +20,2 @@\n"
        b" gamma\n"
        b"-delta\n"
        b"+DELTA\n"
    )
    assert _line_at(two, diff_offset_of_after_line(two, 1)) == b" alpha"
    assert _line_at(two, diff_offset_of_after_line(two, 2)) == b"+BETA"
    assert _line_at(two, diff_offset_of_after_line(two, 20)) == b" gamma"
    assert _line_at(two, diff_offset_of_after_line(two, 21)) == b"+DELTA"
    with pytest.raises(SpanOutsideDiff):
        diff_offset_of_after_line(two, 10), "the gap between the hunks is in neither"


def test_a_hunk_header_without_counts_is_read_as_one_line() -> None:
    """``@@ -5 +5 @@`` is legal unified diff for a single-line hunk."""
    terse = b"@@ -5 +5 @@\n-old\n+new\n"
    assert _line_at(terse, diff_offset_of_after_line(terse, 5)) == b"+new"


def test_an_unreadable_hunk_header_stops_rather_than_skipping() -> None:
    """A header this cannot parse makes every offset after it wrong, not merely unknown, so
    skipping it would produce confident nonsense for the rest of the diff."""
    with pytest.raises(MalformedExample, match="unparseable hunk header"):
        diff_offset_of_after_line(b"@@ garbage @@\n+x\n", 1)


def test_a_line_with_an_unknown_lead_stops_rather_than_being_skipped() -> None:
    """Skipping an unrecognised line shifts every line number after it. Neither corpus
    contains one -- no ``\\ No newline`` marker, no ``---``/``+++`` headers, measured over
    41728 and 50177 diffs -- so meeting one means the dialect changed."""
    with pytest.raises(MalformedExample, match="is not one of"):
        diff_offset_of_after_line(b"@@ -1,1 +1,1 @@\n\\ No newline at end of file\n", 1)


def test_content_before_any_hunk_header_has_no_line_number_to_carry() -> None:
    with pytest.raises(MalformedExample, match="precedes any hunk header"):
        diff_offset_of_after_line(b"+orphan\n@@ -1,1 +1,1 @@\n+x\n", 1)


def test_an_empty_diff_is_its_own_refusal_and_not_a_malformed_row() -> None:
    """``decisions_of`` counts refusals by exception name: an empty diff is a statement
    about the corpus, a malformed line is a statement about one row, and merging them makes
    the leak read as noise."""
    with pytest.raises(EmptyDiffContext):
        diff_offset_of_after_line(b"", 1)
    with pytest.raises(EmptyDiffContext):
        diff_offset_of_after_line(b"   \n", 1)


def test_the_line_number_is_one_based_like_every_other_span_line_in_this_module() -> None:
    with pytest.raises(MalformedExample, match="1-based"):
        diff_offset_of_after_line(REAL, 0)


# -- what `to_decision` does with it -------------------------------------------------------


def _example(after: bytes, diff: bytes, start: int, end: int) -> MutateExample:
    return MutateExample(
        example_id="ex-1",
        repo="r",
        path="bowling_test.go",
        symbol="RollMany",
        arity=2,
        language="go",
        mutation_class="logic",
        after=after,
        span=LineSpan(start_line=start, end_line=end),
        silent=False,
        hunk_constrained=True,
        seed=1,
        diff=diff,
    )


#: The post-image the REAL fixture's hunk was cut from: after line 6 is the mutated line.
AFTER = b"\n".join(
    [
        b"package main",
        b"",
        b'import "testing"',
        b"",
        b"func (g *Game) RollMany(runs int, pins int) {",
        b"\tfor i := 0; i <= runs; i++ {",
        b"\t\tg.Roll(pins)",
        b"\t}",
        b"}",
        b"",
    ]
)


def test_diff_mode_still_gives_the_span_head_nothing_by_default() -> None:
    """The 120-plus diff rows in the ledger trained this way. Turning it on by default
    would move the span gradient into the trunk and make every one of them incomparable
    with every row after it, for a flag nobody set."""
    decision = to_decision(
        _example(AFTER, REAL, 6, 6), max_context_bytes=4096, context_source=CONTEXT_DIFF
    )
    assert decision.gold_span_line is None
    assert decision.gold_span_end_line is None


def test_span_in_diff_points_at_the_line_of_the_diff_the_model_actually_reads() -> None:
    """The whole point: a supervised span head in the mode the plan says the model runs in.

    The assertion is on the CONTENT of the line the index selects, not merely that an index
    came back -- an off-by-one would still produce a falling span loss, on a target that is
    the wrong line.
    """
    decision = to_decision(
        _example(AFTER, REAL, 6, 6),
        max_context_bytes=4096,
        context_source=CONTEXT_DIFF,
        span_in_diff=True,
    )
    assert decision.gold_span_line is not None
    start = decision.context.starts[decision.gold_span_line]
    assert REAL[start:].split(b"\n", 1)[0] == b"+\tfor i := 0; i <= runs; i++ {"
    assert decision.gold_span_end_line == decision.gold_span_line


def test_span_in_diff_does_not_disturb_after_mode() -> None:
    """In after mode the span already points into the text the model reads, so the flag is
    inert there -- and the tool refuses it outright rather than recording a protocol change
    that did not happen."""
    plain = to_decision(
        _example(AFTER, REAL, 6, 6), max_context_bytes=4096, context_source=CONTEXT_AFTER
    )
    flagged = to_decision(
        _example(AFTER, REAL, 6, 6),
        max_context_bytes=4096,
        context_source=CONTEXT_AFTER,
        span_in_diff=True,
    )
    assert plain.gold_span_line == flagged.gold_span_line is not None
    assert plain.gold_span_end_line == flagged.gold_span_end_line


def test_the_added_line_null_is_what_makes_a_diff_span_score_readable() -> None:
    """Without it the pointer is scored against uniform chance, which in a diff is wrong.

    Measured on the real corpus the moment this was wired: the span head reached 92.3%
    against a uniform rate of 8.8%, which reads as a head that has learned to locate
    changes. The added-line null on the same rows is 90.8% -- a policy that reads no code
    at all and aims at a '+'. The head beats it by 1.5 points, not by 83.5. This test is
    the arithmetic that makes that number appear.
    """
    tool = _tool()

    # Two added lines among four candidates, gold on one of them: a random-'+' policy is
    # right half the time, while a uniform policy over all four would be right a quarter.
    two_added = to_decision(
        _example(
            b"a\nb\nc\nd\n",
            b"@@ -1,2 +1,4 @@\n a\n+b\n+c\n d\n",
            2,
            2,
        ),
        max_context_bytes=4096,
        context_source=CONTEXT_DIFF,
        span_in_diff=True,
    )
    chance, pointing, gold_added = tool.plus_line_chance([two_added])
    assert (pointing, gold_added) == (1, 1)
    assert chance == pytest.approx(0.5), (
        "one of the two added lines, so a policy that aims at an added line is right half "
        "the time -- not one in four, which is what counting every candidate line gives"
    )


def test_a_row_whose_gold_is_not_an_added_line_scores_zero_under_the_added_line_null() -> None:
    """The null is weak on such a corpus, and that has to be visible rather than averaged
    into a single number that looks like a strong baseline everywhere."""
    tool = _tool()

    # The gold is a CONTEXT line, which an added-line policy never selects.
    context_gold = to_decision(
        _example(b"a\nb\nc\n", b"@@ -1,3 +1,3 @@\n a\n-x\n+b\n c\n", 3, 3),
        max_context_bytes=4096,
        context_source=CONTEXT_DIFF,
        span_in_diff=True,
    )
    chance, pointing, gold_added = tool.plus_line_chance([context_gold])
    assert (pointing, gold_added) == (1, 0)
    assert chance == 0.0


def test_rows_that_abstain_are_not_in_the_added_line_null_at_all() -> None:
    """Same decomposition the pointing metrics make: a row with no gold line has nothing
    for any pointing policy to be right or wrong about."""
    tool = _tool()

    abstaining = to_decision(
        _example(AFTER, REAL, 6, 6), max_context_bytes=4096, context_source=CONTEXT_DIFF
    )
    assert abstaining.gold_span_line is None
    assert tool.plus_line_chance([abstaining]) == (0.0, 0, 0)


def test_a_span_no_hunk_represents_is_refused_rather_than_pointed_somewhere_nearby() -> None:
    """After line 1 is in the file and in no hunk. Silently clamping it to the hunk's first
    line would teach the pointer that every out-of-hunk span means 'the top of the hunk'."""
    with pytest.raises(SpanOutsideDiff):
        to_decision(
            _example(AFTER, REAL, 1, 1),
            max_context_bytes=4096,
            context_source=CONTEXT_DIFF,
            span_in_diff=True,
        )
