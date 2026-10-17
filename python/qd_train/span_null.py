"""The random-``+``-line null for the span head, and the gate that reads it.

A unified diff marks its own answer: the changed line starts with ``+``. Both corpora average
about 1.5 added lines per hunk, so a policy that reads no code and aims at a ``+`` line scores
far above the uniform-over-candidates rate the original span gates compare against. Measured
2026-09-22 (AUDIT/span-in-diff.md, ``GAP-THE-DIFF-MARKS-ITS-OWN-ANSWER-FOR-THE-SPAN-HEAD``):
span start top-1 63.61% against 89.91% for "point at a random added line", while the old gate
read 63.61% against 8.4% and passed on all 8 seeds. The human ruled the comparison a gate on
2026-10-17 (AUDIT/harden-2026-10-17/human-ruling-2026-10-17.md, Q2): the head's top-1 must
beat this null on the same rows.

The candidate set is the span head's own: ``plan.line_starts[i]`` is
``decision.context.starts`` (``qd_train.byte_batch.plan_batch``), and the head scores exactly
those line starts plus the abstention (``qd_train.heads.serving_scores``). A candidate ``+``
line is a line start whose first byte is ``+``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from .tristate import NotRun, Ran, TriState

__all__ = [
    "PLUS",
    "SPAN_END_TOP1_METRIC",
    "SPAN_NULL_METRIC",
    "SPAN_START_TOP1_METRIC",
    "plus_line_null",
    "span_beats_null_state",
]

#: The byte a unified diff marks an added line with.
PLUS: Final[int] = ord("+")

#: The metrics ``tools/rung0_real_run.py`` writes on every span-scored eval row: the head's
#: start and end top-1 over the pointing rows (``Ran.value`` the rate, ``n_total`` the pointing
#: rows), and the null beside them (``value`` the null, ``n`` the rows whose gold is an added
#: line, ``n_total`` the pointing rows).
SPAN_START_TOP1_METRIC: Final[str] = "val_span_start_top1_over_chance"
SPAN_END_TOP1_METRIC: Final[str] = "val_span_end_top1_over_chance"
SPAN_NULL_METRIC: Final[str] = "val_span_pointing_added_line_chance"


def plus_line_null(
    rows: Iterable[tuple[Sequence[int], int | None]],
) -> tuple[float, int, int]:
    """What "point at a uniformly random candidate ``+`` line" scores, over the same rows.

    Each row is ``(first_bytes, gold_line)``: the first byte of every candidate line (the span
    head's candidate set, in order) and the ordinal of the gold start line, or ``None`` for a
    row whose gold is the abstention. Per pointing row the null is
    ``(gold lines among the candidate + lines) / (candidate + lines)`` -- ``1/added`` when the
    gold line is itself an added line and ``0`` when it is not -- averaged over the pointing
    rows. Abstaining rows are not pointing rows and enter neither count.

    Returns ``(null, pointing_rows, rows_whose_gold_is_an_added_line)``. The two counts are
    different facts and both are carried: a corpus whose gold is often not an added line has a
    weak null, and that has to be visible rather than averaged away.
    """
    total = 0.0
    pointing = 0
    gold_is_added = 0
    for first_bytes, gold in rows:
        if gold is None:
            continue
        if not 0 <= gold < len(first_bytes):
            raise ValueError(
                f"gold line {gold} is outside this row's {len(first_bytes)} candidate lines"
            )
        pointing += 1
        if first_bytes[gold] != PLUS:
            continue
        gold_is_added += 1
        total += 1.0 / sum(1 for b in first_bytes if b == PLUS)  # at least 1: the gold itself
    return (total / pointing if pointing else 0.0), pointing, gold_is_added


def span_beats_null_state(
    start_top1: TriState | None, end_top1: TriState | None, null: TriState | None
) -> TriState:
    """``span_beats_null``: both pointers' top-1 exceed the null, computed on the same rows.

    ``NotRun`` -- which blocks promotion -- when any of the three is absent or did not run, or
    when they were not computed over the same pointing rows (``n_total`` differs), or when
    there are no pointing rows. Otherwise ``Ran``: ``passed`` iff start AND end top-1 are each
    strictly above the null, ``value`` the smaller margin, with the pointing-row count as full
    coverage and every number in ``detail``.
    """
    named = (
        (SPAN_START_TOP1_METRIC, start_top1),
        (SPAN_END_TOP1_METRIC, end_top1),
        (SPAN_NULL_METRIC, null),
    )
    for name, state in named:
        if state is None:
            return NotRun(reason=f"the row carries no {name}: the span head's top-1 against the "
                          "random-+-line null was not computed")
        if isinstance(state, NotRun):
            return NotRun(reason=f"{name} did not run ({state.reason})")
    assert isinstance(start_top1, Ran) and isinstance(end_top1, Ran) and isinstance(null, Ran)
    rows = null.n_total
    if rows is None or rows <= 0:
        return NotRun(reason=f"{SPAN_NULL_METRIC} states no pointing rows to measure over")
    if start_top1.n_total != rows or end_top1.n_total != rows:
        return NotRun(reason=(
            f"the top-1 and the null were not computed over the same rows: start n_total "
            f"{start_top1.n_total}, end n_total {end_top1.n_total}, null n_total {rows}"
        ))
    values = (start_top1.value, end_top1.value, null.value)
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return NotRun(reason="a span top-1 or the null carries no numeric value")
    start, end, baseline = (float(v) for v in values)  # type: ignore[arg-type]
    margin = min(start, end) - baseline
    return Ran(
        passed=start > baseline and end > baseline,
        value=margin,
        n=rows,
        n_total=rows,
        detail=(
            f"span start top-1 {start:.2%} and end top-1 {end:.2%} against the random-+-line "
            f"null {baseline:.2%} on the same {rows} pointing rows "
            f"({null.n} of them with a gold that is an added line); smaller margin "
            f"{margin:+.2%}"
        ),
    )
