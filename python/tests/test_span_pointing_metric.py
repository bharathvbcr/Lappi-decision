"""A head that abstains on everything has located nothing, and must not score 100%.

`evaluate` counted every scored row into `span_start_top1`, including rows whose gold IS
the abstention. That figure is a mixture of two different successes -- "pointed at the right
line" and "correctly declined to point" -- and the mixture rises when the corpus gets
*easier to abstain on*.

`--context-source diff` made the mixture degenerate. Span offsets are into `after`, and
encoding the diff moves every byte, so `to_decision` suppresses span supervision entirely
and every gold becomes the abstention. The first diff arm duly reported::

    span start 100.0% end 100.0% (chance 8.4%, 10635 rows)   [span 9.062->0.000]

100.0% against a "chance" rate computed over line starts the head was never asked to choose
between, and a span loss of exactly zero -- the signature of a constant target. Nothing was
measured and it read as the best result in the file.

So the gate reads a decomposed figure over POINTING rows, and reports `NotRun` when there
are none. The undecomposed fields are kept beside it, unchanged: the 61 capacity rows and
the learning curve were written with the mixture, and silently redefining it would make
every new row incomparable with the rows it extends.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

torch = pytest.importorskip("torch")

import rung0_real_run as tool  # noqa: E402

from qd_train.tristate import NotRun, Ran  # noqa: E402


def _after(**over) -> dict[str, object]:
    """The shape `evaluate` returns, with the fields the span gates read."""
    base = {
        "span_start_top1": 1.0,
        "span_end_top1": 1.0,
        "span_chance": 0.084,
        "span_n": 10_635,
        "span_pointing_n": 0,
        "span_pointing_start_top1": 0.0,
        "span_pointing_end_top1": 0.0,
        "span_pointing_chance": 0.0,
    }
    base.update(over)
    return base


def _gate_for(after: dict[str, object]):
    """The two span gates exactly as `main` builds them."""
    pointing_n = int(after["span_pointing_n"])  # type: ignore[arg-type]
    if not pointing_n:
        return NotRun(
            reason=(
                f"no gold span points at a line: all {int(after['span_n'])} scored "  # type: ignore[arg-type]
                "row(s) abstain, so pointer accuracy has nothing to be measured over. "
                "A head that abstains on everything scores 100% here and has located "
                "nothing."
            )
        )
    return tool._accuracy_gate(
        float(after["span_pointing_start_top1"]),  # type: ignore[arg-type]
        float(after["span_pointing_chance"]),  # type: ignore[arg-type]
        n=pointing_n,
        what="span start",
        baseline_name="uniform-pointer chance over the candidate line starts",
    )


def test_evaluate_reports_the_pointing_decomposition_at_all() -> None:
    """The fields the gate needs exist. Pre-fix they did not, and the gate had only the
    mixture to read."""
    import inspect

    source = inspect.getsource(tool.evaluate)
    for field in (
        "span_pointing_n",
        "span_pointing_start_top1",
        # END as well as START. Decomposing one and not the other prints them side by side
        # as though they were the same measurement: the four-class arm showed `start 1.6%`
        # over pointing rows beside `end 17.9%` over all rows, which reads as a head that
        # locates the end of a span whose start it cannot find.
        "span_pointing_end_top1",
        "span_pointing_chance",
    ):
        assert field in source, f"evaluate does not report {field}"
    assert "span_is_noul" in source, (
        "evaluate does not consult span_is_noul, so it cannot tell a row that pointed from "
        "a row that abstained"
    )


def test_an_all_abstain_corpus_is_not_run_rather_than_a_perfect_score() -> None:
    """The load-bearing case. This is exactly what the first diff arm produced."""
    verdict = _gate_for(_after())
    assert isinstance(verdict, NotRun), f"expected NotRun, got {verdict}"
    assert "abstain" in verdict.reason
    assert "located nothing" in verdict.reason


def test_a_corpus_with_pointers_is_still_measured() -> None:
    """The control. A gate that refused everything would be as useless as one that passed
    everything."""
    verdict = _gate_for(
        _after(
            span_pointing_n=8_000,
            span_pointing_start_top1=0.095,
            span_pointing_chance=0.032,
        )
    )
    assert isinstance(verdict, Ran), f"expected Ran, got {verdict}"
    assert verdict.passed, "9.5% against 3.2% chance should pass"
    assert verdict.n_total == 8_000, "the gate must report the POINTING row count"


def test_the_gate_counts_pointing_rows_not_every_scored_row() -> None:
    """A four-class corpus abstains on its clean rows. Counting those into the pointer
    figure inflates it by the clean share -- 8,449 of 50,177 on commitpackft -- and the
    inflation grows as the head gets better at abstaining."""
    verdict = _gate_for(
        _after(
            span_n=10_000,
            span_pointing_n=8_300,
            span_pointing_start_top1=0.10,
            span_pointing_chance=0.032,
        )
    )
    assert isinstance(verdict, Ran)
    assert verdict.n_total == 8_300 and verdict.n_total != 10_000


def test_both_pointers_are_decomposed_not_only_the_start() -> None:
    """The end gate must read the pointing figure too.

    It did not at first: `val_span_end_top1_over_chance` was gated on `pointing_n` but still
    read `span_end_top1` over every scored row. The v2 four-class arm printed the result --
    `span start 1.6% end 17.9%` -- and the gap was the abstentions showing through one of
    the two, not an asymmetry in the head.
    """
    import inspect

    source = inspect.getsource(tool.main)
    end_gate = source[source.index("val_span_end_top1_over_chance") :][:600]
    assert "span_pointing_end_top1" in end_gate, (
        "the end gate reads the undecomposed figure while the start gate reads the "
        "decomposed one, so the pair is not comparable with itself"
    )
    assert "span_pointing_chance" in end_gate


def test_a_pointing_head_at_chance_fails_rather_than_passing_on_abstentions() -> None:
    """Pre-fix, a head that pointed at chance but abstained perfectly could clear the gate
    on the strength of the abstentions alone."""
    verdict = _gate_for(
        _after(
            span_n=10_000,
            span_pointing_n=5_000,
            span_pointing_start_top1=0.031,
            span_pointing_chance=0.032,
        )
    )
    assert isinstance(verdict, Ran)
    assert not verdict.passed, "a pointer at chance must not pass"
