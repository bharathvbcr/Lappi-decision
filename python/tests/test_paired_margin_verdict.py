"""`paired_margin_test`'s detail has to describe the interval it actually computed.

`passed` is a bool and the gate is right to be. The DETAIL is read by people, and "not a
win" covers two findings that are not the same:

* an interval straddling zero — the comparison could not separate the two arms;
* an interval entirely below zero — the control won, and by a measurable amount.

The old string appended *"CI includes zero, so this is not a win"* to both. That is a false
statement in an append-only ledger whenever the interval does not, in fact, include zero.

It stayed invisible for as long as the gate did. Across 988 rows `paired_margin_vs_linear`
was never evaluated; the first arm to evaluate it produced intervals that straddled zero
(`[-0.0660, +0.0174]`), where the sentence was true. The first decisive loss it ever scored
— `-0.1260, [-0.1356, -0.1164]` on the commitpackft corpus, 2026-09-21 — was also the first
time the sentence was wrong. Wiring a gate is what makes its messages falsifiable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train.eval_harness import paired_margin_test  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402


def _arms(model_right: int, baseline_right: int, n: int):
    """Two correctness vectors with the given counts, arranged to be maximally separated
    so the bootstrap interval is tight rather than borderline."""
    model = np.zeros(n, dtype=bool)
    baseline = np.zeros(n, dtype=bool)
    model[:model_right] = True
    baseline[n - baseline_right:] = True
    return model, baseline


def test_a_decisive_loss_does_not_claim_the_interval_includes_zero():
    """The bug: a CI entirely below zero was described as straddling it."""
    model, baseline = _arms(model_right=100, baseline_right=900, n=1000)
    out = paired_margin_test(model, baseline)
    assert isinstance(out, Ran)
    assert out.passed is False
    assert out.value < 0
    assert "CI includes zero" not in out.detail
    assert "the whole interval is below zero" in out.detail


def test_a_decisive_loss_says_how_far_the_baseline_is_ahead():
    model, baseline = _arms(model_right=100, baseline_right=900, n=1000)
    out = paired_margin_test(model, baseline)
    assert f"beats the model by {-out.value:.4f}" in out.detail


def test_a_straddling_interval_still_says_so():
    """The other branch must keep working: near-identical arms are inconclusive."""
    rng = np.random.default_rng(0)
    model = rng.random(400) < 0.5
    baseline = rng.random(400) < 0.5
    out = paired_margin_test(model, baseline)
    assert isinstance(out, Ran)
    if not out.passed and "the whole interval is below zero" not in out.detail:
        assert "CI includes zero" in out.detail


def test_a_win_carries_no_verdict_clause_at_all():
    model, baseline = _arms(model_right=900, baseline_right=100, n=1000)
    out = paired_margin_test(model, baseline)
    assert out.passed is True
    assert "not a win" not in out.detail
    assert "below zero" not in out.detail


def test_the_three_branches_are_mutually_exclusive():
    """No result may describe itself as both inconclusive and decisive."""
    cases = [
        _arms(900, 100, 1000),   # win
        _arms(100, 900, 1000),   # decisive loss
        _arms(500, 500, 1000),   # tied
    ]
    for model, baseline in cases:
        detail = paired_margin_test(model, baseline).detail
        assert not (
            "CI includes zero" in detail and "the whole interval is below zero" in detail
        )


def test_the_reported_bounds_bracket_the_point_estimate():
    """Guards the sentence against the numbers: a verdict about an interval is only
    meaningful if the interval printed is the one the verdict was derived from."""
    model, baseline = _arms(model_right=100, baseline_right=900, n=1000)
    out = paired_margin_test(model, baseline)
    inside = out.detail.split("CI [")[1].split("]")[0]
    lo, hi = (float(part) for part in inside.split(", "))
    assert lo <= out.value <= hi
    assert hi < 0.0


@pytest.mark.parametrize("n", [4, 50, 1000])
def test_the_verdict_holds_at_every_size(n):
    model, baseline = _arms(model_right=0, baseline_right=n, n=n)
    out = paired_margin_test(model, baseline)
    assert out.passed is False
    assert "CI includes zero" not in out.detail
