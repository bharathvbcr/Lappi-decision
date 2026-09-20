"""Fitting the runtime's calibration table, and the scale trap in the middle of it."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from qd_train.calibration_fit import (
    CHOICE,
    MAX_OPTIONS,
    MIN_OPTIONS,
    RESERVED_NOUL_ROWS,
    SCORE,
    CalibrationEntryFit,
    ece_gate,
    expected_calibration_error,
    fit_entry,
    fit_noul_margin,
    fit_table,
    fit_temperature,
    letters_key,
    probability_cutoff_from_nonconformity,
)
from qd_train.eval_harness import conformal_sets, split_conformal_threshold
from qd_train.tristate import NotRun, Ran

SCHEMA_RS = (
    Path(__file__).resolve().parents[2] / "crates" / "qd-runtime" / "src" / "schema.rs"
)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _calibrated(n: int = 4000, k: int = 4, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that are calibrated by construction: labels are *drawn from* them."""
    rng = np.random.default_rng(seed)
    probs = _softmax(rng.normal(size=(n, k)) * 1.5)
    labels = np.array([rng.choice(k, p=row) for row in probs])
    return probs, labels


# ---------------------------------------------------------------------------
# The scale trap. This is the reason the module exists in this shape.
# ---------------------------------------------------------------------------


def _regime(scale: float, n: int = 4000, k: int = 4, seed: int = 11):
    """Calibrated probabilities at a chosen sharpness: labels are drawn from them.

    `scale` sets how accurate the model is, which is what decides whether `q_hat` lands
    above or below 0.5 — and therefore which way the scale bug breaks.
    """
    rng = np.random.default_rng(seed)
    probs = _softmax(rng.normal(size=(n, k)) * scale)
    labels = np.array([rng.choice(k, p=row) for row in probs])
    return probs, labels


@pytest.mark.parametrize(
    "scale,expect_wrong_narrower",
    [
        (1.5, True),  # weak model: q_hat ~0.86 > 0.5, sets collapse, coverage ~0.09
        (10.0, False),  # strong model: q_hat ~0.37 < 0.5, sets widen and go vacuous
    ],
)
def test_using_the_nonconformity_quantile_as_the_cutoff_breaks_the_guarantee(
    scale: float, expect_wrong_narrower: bool
):
    """`conformal_quantile = split_conformal_threshold(...)` is the obvious line, and wrong.

    Rust filters `p >= entry.conformal_quantile`, so it wants a *probability* cutoff, while
    `split_conformal_threshold` returns a *nonconformity* quantile.

    Both regimes are pinned because the error changes direction with model accuracy, and
    because at `q_hat == 0.5` the two scales coincide and the bug disappears entirely — a
    single fixture near that point would pass with the bug fully present.
    """
    alpha = 0.1
    probs, labels = _regime(scale)
    cal, test = slice(0, 2000), slice(2000, 4000)
    q_hat = split_conformal_threshold(probs[cal], labels[cal], alpha=alpha)
    cutoff = probability_cutoff_from_nonconformity(q_hat)
    assert cutoff == pytest.approx(1.0 - q_hat)
    assert abs(q_hat - 0.5) > 0.05, (
        f"q_hat={q_hat:.3f} is too near 0.5, where the bug has no effect; this fixture "
        "would prove nothing"
    )

    y = labels[test]
    right = [np.flatnonzero(row >= cutoff).tolist() for row in probs[test]]
    wrong = [np.flatnonzero(row >= q_hat).tolist() for row in probs[test]]

    cov_right = float(np.mean([y[i] in s for i, s in enumerate(right)]))
    cov_wrong = float(np.mean([y[i] in s for i, s in enumerate(wrong)]))
    size_right = float(np.mean([len(s) for s in right]))
    size_wrong = float(np.mean([len(s) for s in wrong]))

    assert cov_right >= (1 - alpha) - 0.03, (
        f"the converted cutoff must keep the guarantee, got {cov_right:.3f}"
    )
    assert abs(cov_wrong - (1 - alpha)) > abs(cov_right - (1 - alpha)), (
        f"the un-converted threshold must miss the target by more: "
        f"wrong={cov_wrong:.3f} right={cov_right:.3f} target={1 - alpha}"
    )
    if expect_wrong_narrower:
        assert size_wrong < size_right and cov_wrong < 0.5, (
            f"a weak model should collapse the sets: size {size_wrong:.2f} cov {cov_wrong:.3f}"
        )
    else:
        assert size_wrong > size_right, (
            f"a strong model should widen the sets: {size_wrong:.2f} vs {size_right:.2f}"
        )


def test_the_supported_conversion_agrees_with_the_harness_set_builder():
    """`conformal_sets` and the Rust-side cutoff must select the same rows.

    `conformal_sets(probs, q_hat)` filters `p >= 1 - q_hat` internally; Rust filters
    `p >= conformal_quantile`. Those are the same set only if the conversion is applied.
    """
    probs, labels = _calibrated(n=1200, seed=3)
    q_hat = split_conformal_threshold(probs, labels, alpha=0.1)
    cutoff = probability_cutoff_from_nonconformity(q_hat)

    harness = conformal_sets(probs, q_hat)
    rust_side = [np.flatnonzero(row >= cutoff).tolist() for row in probs]
    assert harness == rust_side


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_a_conformal_quantile_off_the_probability_scale_is_refused(bad: float):
    with pytest.raises(ValueError):
        probability_cutoff_from_nonconformity(bad)


# ---------------------------------------------------------------------------
# ECE: the gate the ledger has always listed and nothing computed
# ---------------------------------------------------------------------------


def test_a_calibrated_model_has_near_zero_ece():
    probs, labels = _calibrated(n=8000, seed=5)
    ece, populated = expected_calibration_error(probs, labels)
    assert ece < 0.05, f"labels drawn from the probabilities should be calibrated, got {ece:.4f}"
    assert populated > 1, "a one-bin ECE is a statement about one bin"


def test_an_overconfident_model_is_caught():
    """Sharpen a calibrated model without changing its predictions; only ECE should move."""
    probs, labels = _calibrated(n=8000, seed=5)
    sharp = _softmax(np.log(np.clip(probs, 1e-12, None)) * 4.0)
    assert (sharp.argmax(axis=1) == probs.argmax(axis=1)).all(), "accuracy must be unchanged"

    base, _ = expected_calibration_error(probs, labels)
    worse, _ = expected_calibration_error(sharp, labels)
    assert worse > base + 0.1, f"overconfidence should show: {base:.4f} -> {worse:.4f}"

    gate = ece_gate(sharp, labels, threshold=0.05)
    assert isinstance(gate, Ran)
    assert not gate.passed


def test_ece_below_the_sample_floor_is_not_run_not_a_pass():
    """A check that could not run must not report what a check that ran and passed reports."""
    probs, labels = _calibrated(n=20, seed=1)
    gate = ece_gate(probs, labels, min_samples=100)
    assert isinstance(gate, NotRun)
    assert not hasattr(gate, "passed")
    assert "20" in gate.reason


def test_ece_carries_both_counts():
    probs, labels = _calibrated(n=500, seed=2)
    gate = ece_gate(probs, labels)
    assert isinstance(gate, Ran)
    assert gate.n == gate.n_total == 500


def test_confidence_of_exactly_one_lands_in_the_last_bin_not_outside_it():
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    labels = np.array([0, 0])
    ece, populated = expected_calibration_error(probs, labels, n_bins=5)
    assert populated == 1
    assert ece == pytest.approx(0.0), "perfectly confident and perfectly right is zero error"


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


def test_fit_temperature_recovers_a_known_distortion():
    """Divide logits by 3, and the fitted temperature should come back near 3."""
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(6000, 5)) * 2.0
    labels = np.array([rng.choice(5, p=row) for row in _softmax(logits)])

    assert fit_temperature(logits, labels) == pytest.approx(1.0, abs=0.15)
    assert fit_temperature(logits * 3.0, labels) == pytest.approx(3.0, rel=0.15)


def test_fit_temperature_lowers_nll_it_does_not_just_return_one():
    rng = np.random.default_rng(17)
    logits = rng.normal(size=(3000, 4)) * 5.0
    labels = np.array([rng.choice(4, p=row) for row in _softmax(logits / 5.0)])
    t = fit_temperature(logits, labels)

    def nll(temp: float) -> float:
        p = _softmax(logits / temp)
        return float(-np.log(p[np.arange(len(labels)), labels]).mean())

    assert nll(t) < nll(1.0), "a fitted temperature that does not beat T=1 has not fitted anything"


def test_non_finite_logits_are_refused_rather_than_fitted():
    logits = np.array([[0.0, 1.0], [np.inf, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        fit_temperature(logits, np.array([0, 1]))


# ---------------------------------------------------------------------------
# noul margin
# ---------------------------------------------------------------------------


def test_the_noul_margin_buys_precision():
    rng = np.random.default_rng(23)
    margins = rng.uniform(0, 1, size=2000)
    # Correctness rises with margin, which is the property abstention exploits.
    correct = rng.uniform(size=2000) < (0.5 + 0.5 * margins)
    thr = fit_noul_margin(margins, correct, target_precision=0.9)
    keep = margins >= thr
    assert keep.any()
    assert correct[keep].mean() >= 0.9
    assert correct[keep].mean() > correct.mean(), "abstaining must improve on answering everything"


def test_an_unreachable_precision_target_does_not_silently_return_zero():
    """Returning 0.0 would mean 'answer everything' — the opposite of what was asked."""
    margins = np.linspace(0.0, 1.0, 100)
    correct = np.zeros(100, dtype=bool)
    assert fit_noul_margin(margins, correct, target_precision=0.99) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The Rust seam
# ---------------------------------------------------------------------------


def test_the_key_counts_rows_including_noul():
    assert letters_key(CHOICE, MAX_OPTIONS + RESERVED_NOUL_ROWS) == "choice:17"
    assert letters_key(SCORE, MIN_OPTIONS + RESERVED_NOUL_ROWS) == "score:3"
    with pytest.raises(ValueError):
        letters_key(CHOICE, MIN_OPTIONS)  # named count, noul row forgotten
    with pytest.raises(ValueError):
        letters_key("span", 5)  # span has one entry, not one per row count


def test_constants_still_match_schema_rs():
    """The Python mirrors are duplication; this is what stops it rotting.

    `calibration_fit` restates MAX_OPTIONS, MIN_OPTIONS, RESERVED_NOUL_ROWS and the
    SlotKind strings because the Rust crate is not importable here. If schema.rs changes,
    this fails rather than letting the two drift into a fifth 'one name, two quantities'.
    """
    src = SCHEMA_RS.read_text()

    def const(name: str) -> int:
        m = re.search(rf"pub const {name}:\s*\w+\s*=\s*(\d+)\s*;", src)
        assert m, f"{name} not found in {SCHEMA_RS}"
        return int(m.group(1))

    assert const("MAX_OPTIONS") == MAX_OPTIONS
    assert const("MIN_OPTIONS") == MIN_OPTIONS
    assert const("RESERVED_NOUL_ROWS") == RESERVED_NOUL_ROWS
    for kind in (CHOICE, SCORE, "span"):
        assert f'=> "{kind}"' in src, f"SlotKind::as_str no longer emits {kind!r}"


def test_an_entry_validates_the_same_ranges_rust_does():
    ok = CalibrationEntryFit(temperature=1.5, conformal_quantile=0.8, noul_margin=0.02)
    assert set(ok.to_json()) == {"temperature", "conformal_quantile", "noul_margin"}

    for kwargs in (
        {"temperature": 0.0, "conformal_quantile": 0.8, "noul_margin": 0.02},
        {"temperature": -1.0, "conformal_quantile": 0.8, "noul_margin": 0.02},
        {"temperature": math.nan, "conformal_quantile": 0.8, "noul_margin": 0.02},
        {"temperature": 1.0, "conformal_quantile": 1.5, "noul_margin": 0.02},
        {"temperature": 1.0, "conformal_quantile": 0.8, "noul_margin": -0.1},
    ):
        with pytest.raises(ValueError):
            CalibrationEntryFit(**kwargs)


def test_fit_entry_applies_the_conversion_so_a_caller_cannot_skip_it():
    entry = fit_entry(temperature=1.0, nonconformity_quantile=0.1, noul_margin=0.02)
    assert entry.conformal_quantile == pytest.approx(0.9)


def test_a_table_without_a_span_entry_says_so_rather_than_defaulting():
    letters = {
        (CHOICE, 3): fit_entry(temperature=1.0, nonconformity_quantile=0.1, noul_margin=0.02)
    }
    table = fit_table("fitted-v1", letters)
    assert table["name"] == "fitted-v1"
    assert table["span"] is None
    assert list(table["letters"]) == ["choice:3"]
    assert table["letters"]["choice:3"]["conformal_quantile"] == pytest.approx(0.9)

    with pytest.raises(ValueError):
        fit_table("", letters)
