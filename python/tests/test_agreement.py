"""The teacher-agreement gate, and its honesty about its own sample size."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.agreement import (  # noqa: E402
    cohens_kappa,
    confusion,
    disagreements,
    kappa_gate,
    kappa_with_ci,
)
from qd_train.tristate import Ran  # noqa: E402

CLASSES = ["stub", "logic", "cosmetic", "clean"]


def _pair(n: int, agree: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    human = list(rng.choice(CLASSES, size=n))
    teacher = [
        x if rng.random() < agree else str(rng.choice([c for c in CLASSES if c != x]))
        for x in human
    ]
    return human, teacher


def test_perfect_agreement_is_one():
    labels = ["stub", "logic", "clean", "cosmetic"] * 5
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_independent_raters_land_near_zero():
    rng = np.random.default_rng(0)
    a = list(rng.choice(CLASSES, size=2000))
    b = list(rng.choice(CLASSES, size=2000))
    assert abs(cohens_kappa(a, b)) < 0.1


def test_both_raters_constant_is_zero_not_one():
    """Perfect agreement carrying no information must not read as a passing gate.

    p_e == 1 makes kappa 0/0. A rubric that drives the teacher to answer one class
    every time agrees with a human who did the same, perfectly and uselessly.
    """
    a = ["clean"] * 50
    assert cohens_kappa(a, list(a)) == 0.0


def test_kappa_is_symmetric():
    h, t = _pair(200, 0.8, 3)
    assert cohens_kappa(h, t) == pytest.approx(cohens_kappa(t, h))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="disagree in length"):
        cohens_kappa(["a", "b"], ["a"])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="zero items"):
        cohens_kappa([], [])


# --------------------------------------------------------------------------
# The interval
# --------------------------------------------------------------------------

def test_ci_brackets_the_point_estimate():
    h, t = _pair(120, 0.75, 5)
    r = kappa_with_ci(h, t, n_boot=2000, seed=5)
    assert r.ci_low <= r.kappa <= r.ci_high


def test_the_interval_narrows_with_more_items():
    """The measured basis for calling the n=50 gate underpowered."""
    widths = []
    for n in (50, 400):
        h, t = _pair(n, 0.74, 11)
        r = kappa_with_ci(h, t, n_boot=1500, seed=11)
        widths.append(r.ci_high - r.ci_low)
    assert widths[1] < widths[0] / 1.5, f"CI did not narrow meaningfully: {widths}"


def test_n50_interval_is_wide_enough_to_straddle_the_threshold():
    """Documents the power problem as an executable claim, not a footnote."""
    h, t = _pair(50, 0.74, 7)
    r = kappa_with_ci(h, t, n_boot=4000, seed=7)
    assert r.ci_high - r.ci_low > 0.20, f"expected a wide interval at n=50, got {r}"


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_a_clearly_good_teacher_settles_as_a_pass():
    h, t = _pair(400, 0.95, 2)
    g = kappa_gate(kappa_with_ci(h, t, n_boot=1500, seed=2))
    assert isinstance(g, Ran) and g.passed is True
    assert "across the whole interval" in g.detail
    assert "UNDERPOWERED" not in g.detail


def test_a_clearly_bad_teacher_settles_as_a_fail():
    h, t = _pair(400, 0.35, 4)
    g = kappa_gate(kappa_with_ci(h, t, n_boot=1500, seed=4))
    assert isinstance(g, Ran) and g.passed is False
    assert "rewrite the rubric" in g.detail


def test_a_borderline_result_is_reported_as_unsettled():
    """0.62 must not read as a clean pass at n=50."""
    h, t = _pair(50, 0.74, 7)
    g = kappa_gate(kappa_with_ci(h, t, n_boot=4000, seed=7))
    assert "UNDERPOWERED" in g.detail
    assert "does not settle this either way" in g.detail


def test_the_gate_does_not_move_the_threshold():
    """Rule 2: an agent may report a gate failed; it may not retune it."""
    h, t = _pair(300, 0.5, 9)
    r = kappa_with_ci(h, t, n_boot=1000, seed=9)
    assert kappa_gate(r).passed == (r.kappa >= 0.6)
    # A caller may pass a different threshold explicitly, but 0.6 is the default.
    assert kappa_gate(r, threshold=0.6).passed == (r.kappa >= 0.6)


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------

def test_disagreements_reports_shown_and_total():
    """A truncated list must never read as the complete set."""
    items = [f"diff-{i}" for i in range(100)]
    human = ["stub"] * 100
    teacher = ["clean" if i % 2 == 0 else "stub" for i in range(100)]
    out = disagreements(items, human, teacher, limit=10)
    assert len(out) == 10
    assert out[0]["_shown"] == "10 of 50 disagreements"


def test_disagreements_are_only_the_disagreeing_rows():
    out = disagreements(["a", "b"], ["stub", "clean"], ["stub", "logic"])
    assert len(out) == 1 and out[0]["item"] == "b"


def test_disagreements_length_mismatch_raises():
    with pytest.raises(ValueError, match="lengths differ"):
        disagreements(["a"], ["stub", "clean"], ["stub"])


def test_confusion_counts_rows_as_first_rater():
    classes, m = confusion(["stub", "stub", "clean"], ["stub", "clean", "clean"])
    i = {c: k for k, c in enumerate(classes)}
    assert m[i["stub"], i["stub"]] == 1
    assert m[i["stub"], i["clean"]] == 1
    assert m[i["clean"], i["clean"]] == 1
    assert m.sum() == 3
