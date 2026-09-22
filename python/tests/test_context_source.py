"""What the choice head reads, and the leak that made the first answer wrong.

``GAP-RUNG0-CHOICE-SEES-THE-POST-IMAGE-NOT-THE-DIFF`` recorded that the choice slot is
asked "what kind of change is this?" while being shown only the file the change produced.
The fix is ``--context-source diff``. The trap is that on this corpus the diff is not
merely more informative -- for one class it IS the label.

``crates/qd-mutate/src/generate.rs`` emits ``clean`` rows with ``before == after`` and
``diff: String::new()``. Measured over all 50,178 examples of the commitpackft corpus:
8,450 rows have an empty diff, all 8,450 are ``clean``, and no ``clean`` row has a
non-empty one (``AUDIT/after-vs-diff-leak.log``). So "is my context empty?" answers the
question exactly, and a model that learned that would score well while having learned
nothing about change.

These tests pin both halves: the diff can be read, and a void cannot.
"""

from __future__ import annotations

import pytest
from data_fixtures import mutate_row as row

from qd_train.mutate_adapter import (
    CONTEXT_AFTER,
    CONTEXT_DIFF,
    CONTEXT_SOURCES,
    EmptyDiffContext,
    MalformedExample,
    parse_example,
    refuse_leaky_diff_corpus,
    to_decision,
)

WIDE = 8192


def decode(decision) -> bytes:
    """The bytes the model would actually be given."""
    return bytes(decision.context.ids)


# ---------------------------------------------------------------------------
# The source is selected, not assumed
# ---------------------------------------------------------------------------


def test_the_two_sources_are_named_and_after_is_the_default():
    """A default of `diff` would silently redefine every recorded row's meaning."""
    assert CONTEXT_SOURCES == (CONTEXT_AFTER, CONTEXT_DIFF)
    default = to_decision(parse_example(row()), max_context_bytes=WIDE)
    assert decode(default) == row()["after"].encode("utf-8")


def test_diff_mode_encodes_the_diff_and_not_the_post_image():
    example = parse_example(row())
    decision = to_decision(example, max_context_bytes=WIDE, context_source=CONTEXT_DIFF)
    assert decode(decision) == row()["diff"].encode("utf-8")
    assert decode(decision) != row()["after"].encode("utf-8")


def test_an_unknown_context_source_is_refused_rather_than_defaulted():
    with pytest.raises(MalformedExample, match="context_source"):
        to_decision(parse_example(row()), max_context_bytes=WIDE, context_source="before")


def test_parse_example_carries_the_diff_and_tolerates_its_absence():
    """Rows written before qd-mutate emitted a diff still parse; they simply cannot be
    asked for one."""
    assert parse_example(row()).diff == row()["diff"].encode("utf-8")
    without = row()
    del without["diff"]
    assert parse_example(without).diff is None


# ---------------------------------------------------------------------------
# The span head is not pointed at the wrong text
# ---------------------------------------------------------------------------


def test_diff_mode_gives_the_span_head_no_target():
    """Span offsets are into `after`. Every byte moves when the diff is encoded, so a
    span carried over would point somewhere arbitrary -- and a confidently wrong pointer
    trains worse than an absent one."""
    example = parse_example(row())
    assert example.span is not None

    on_after = to_decision(example, max_context_bytes=WIDE)
    assert on_after.gold_span_line is not None, "the post-image path still supervises spans"

    on_diff = to_decision(example, max_context_bytes=WIDE, context_source=CONTEXT_DIFF)
    assert on_diff.gold_span_line is None
    assert on_diff.gold_span_end_line is None


# ---------------------------------------------------------------------------
# A void is refused, not encoded
# ---------------------------------------------------------------------------


def test_an_empty_diff_is_refused_because_on_this_corpus_it_names_the_class():
    """The load-bearing test. Pre-fix this returned a ByteDecision over zero bytes, and
    every clean row in a diff-mode run would have carried its own label in its length."""
    example = parse_example(row(**{"class": "clean", "span": None, "diff": ""}))
    with pytest.raises(EmptyDiffContext) as caught:
        to_decision(example, max_context_bytes=WIDE, context_source=CONTEXT_DIFF)
    assert "empty one" in str(caught.value)


def test_a_missing_diff_is_refused_by_the_same_branch_as_an_empty_one():
    without = row()
    del without["diff"]
    with pytest.raises(EmptyDiffContext) as caught:
        to_decision(parse_example(without), max_context_bytes=WIDE, context_source=CONTEXT_DIFF)
    assert "none" in str(caught.value)


def test_a_whitespace_only_diff_counts_as_empty():
    """`" \\n"` carries no more evidence than `""` and must not slip past on length."""
    example = parse_example(row(**{"diff": "  \n\t\n"}))
    with pytest.raises(EmptyDiffContext):
        to_decision(example, max_context_bytes=WIDE, context_source=CONTEXT_DIFF)


def test_an_empty_diff_is_fine_in_after_mode():
    """After mode never reads the field, so a row with no diff is an ordinary row."""
    example = parse_example(row(**{"diff": ""}))
    decision = to_decision(example, max_context_bytes=WIDE)
    assert decode(decision) == row()["after"].encode("utf-8")


# ---------------------------------------------------------------------------
# The corpus is judged before training, not one row at a time
# ---------------------------------------------------------------------------


def leaky_corpus() -> list[dict]:
    """The shape qd-mutate actually emits: clean rows empty, every other row not."""
    return [
        row(**{"id": "m1", "class": "logic"}),
        row(**{"id": "m2", "class": "stub"}),
        row(**{"id": "c1", "class": "clean", "span": None, "diff": ""}),
        row(**{"id": "c2", "class": "clean", "span": None, "diff": ""}),
    ]


def test_after_mode_never_refuses_a_corpus_for_its_diffs():
    assert refuse_leaky_diff_corpus(leaky_corpus(), context_source=CONTEXT_AFTER) == {}


def test_a_corpus_whose_clean_rows_are_all_empty_is_refused_before_any_training():
    with pytest.raises(EmptyDiffContext) as caught:
        refuse_leaky_diff_corpus(leaky_corpus(), context_source=CONTEXT_DIFF)
    message = str(caught.value)
    assert "clean 2/2" in message
    assert "ALL of them" in message, "a perfectly separable class must be named as one"
    assert "--context-source after" in message, "the refusal names the way out"


def test_a_repaired_corpus_passes():
    """Once every clean row carries the agent's own commit diff, nothing is separable by
    length and the check gets out of the way."""
    repaired = [
        dict(r, diff="--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n") if r["class"] == "clean" else r
        for r in leaky_corpus()
    ]
    assert refuse_leaky_diff_corpus(repaired, context_source=CONTEXT_DIFF) == {}


def test_a_partly_empty_class_is_still_refused_but_not_called_total():
    """Half-repaired is still leaky -- an empty context is a void whatever its class --
    but the message must not claim a separability it did not measure."""
    partial = leaky_corpus()
    partial[2] = dict(partial[2], diff="@@ -1 +1 @@\n-x\n+y\n")
    with pytest.raises(EmptyDiffContext) as caught:
        refuse_leaky_diff_corpus(partial, context_source=CONTEXT_DIFF)
    message = str(caught.value)
    assert "clean 1/2" in message
    assert "ALL of them" not in message


# The matching recipe-hash tests live in ``test_rung0_real_run.py``: that module imports
# torch at scope and is gated for it, and this one is deliberately torch-free.
