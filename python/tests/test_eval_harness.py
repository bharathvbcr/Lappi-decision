"""S7 — test the tests.

*"Feed the eval harness a constant-prediction model and a shuffled-label model: the
degenerate-head assertion and the shuffled-label control must fail them."*

If the harness passes either, the harness is broken and no number it produces means
anything. These tests are therefore run before any real eval, and their result is a
ledger row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.baseline import CharNGramHasher, LinearBaseline  # noqa: E402
from qd_train.eval_harness import (  # noqa: E402
    conformal_sets,
    degenerate_head_check,
    evaluate,
    paired_margin_test,
    shuffled_label_control,
    split_conformal_threshold,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

RNG = np.random.default_rng(1234)


def _healthy_probs(n: int = 400, k: int = 4, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A model that is right ~75% of the time with honest spread."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    logits = rng.normal(0, 1.0, size=(n, k))
    logits[np.arange(n), y] += 2.2
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True), y


# --------------------------------------------------------------------------
# S7.1 — the constant-prediction model must fail
# --------------------------------------------------------------------------

def test_constant_prediction_model_fails_the_degenerate_assertion():
    """Always the same class, confidently. The canonical collapsed head."""
    n, k = 300, 4
    probs = np.full((n, k), 0.001)
    probs[:, 2] = 1.0 - 0.001 * (k - 1)
    result = degenerate_head_check(probs)
    assert isinstance(result, Ran)
    assert result.passed is False, "a constant-prediction head must not pass"
    assert "DEGENERATE" in result.detail


def test_constant_but_soft_model_still_fails_on_class_share():
    """The blind spot of an entropy-only check: soft distribution, one argmax.

    Entropy is high, so an entropy-only assertion passes it. The class-share limb
    is what catches it — which is why the check has three independent limbs.
    """
    n, k = 300, 4
    probs = np.full((n, k), 0.24)
    probs[:, 1] = 0.28  # every argmax is class 1, but entropy is near-maximal
    probs /= probs.sum(axis=1, keepdims=True)

    from qd_train.eval_harness import DEFAULT_ENTROPY_FLOOR

    assert float(-np.sum(probs * np.log(probs), axis=1).mean()) > DEFAULT_ENTROPY_FLOOR
    result = degenerate_head_check(probs)
    assert isinstance(result, Ran) and result.passed is False
    assert "takes 1.000 of predictions" in result.detail


def test_dead_features_fail_even_when_predictions_look_fine():
    """A head reading constant features fails neither entropy nor class-share."""
    probs, _ = _healthy_probs()
    dead = np.ones((len(probs), 16), dtype=np.float64)
    assert degenerate_head_check(probs).passed is True
    result = degenerate_head_check(probs, features=dead)
    assert isinstance(result, Ran) and result.passed is False
    assert "feature variance" in result.detail


def test_a_healthy_head_passes():
    probs, _ = _healthy_probs()
    features = RNG.normal(0, 1, size=(len(probs), 16))
    result = degenerate_head_check(probs, features=features)
    assert isinstance(result, Ran) and result.passed is True


# --------------------------------------------------------------------------
# S7.2 — the shuffled-label model must fail
# --------------------------------------------------------------------------

def test_shuffled_label_model_scoring_above_chance_is_flagged_as_leakage():
    labels = np.array([0] * 250 + [1] * 250)  # chance (majority) = 0.5
    result = shuffled_label_control(0.72, labels, n_eval=500)
    assert isinstance(result, Ran) and result.passed is False
    assert "LEAKAGE" in result.detail


def test_shuffled_label_model_at_chance_passes():
    labels = np.array([0] * 250 + [1] * 250)
    result = shuffled_label_control(0.505, labels, n_eval=500)
    assert isinstance(result, Ran) and result.passed is True


def test_chance_is_the_majority_rate_not_one_over_k():
    """On an imbalanced set, 1/k sets the bar BELOW chance and lets leakage through.

    90/10 split: a model that learned nothing scores 0.90 by always guessing the
    majority. Against 1/k = 0.5 that would look like a strong signal.
    """
    labels = np.array([0] * 900 + [1] * 100)
    assert shuffled_label_control(0.90, labels, n_eval=1000).passed is True
    assert shuffled_label_control(0.97, labels, n_eval=1000).passed is False


# --------------------------------------------------------------------------
# "No number is reported while it fails"
# --------------------------------------------------------------------------

def test_no_metric_is_reported_from_a_degenerate_head():
    """The rule enforced structurally: metrics come back NotRun, not a number."""
    n, k = 300, 4
    probs = np.full((n, k), 0.001)
    probs[:, 2] = 0.997
    y = np.random.default_rng(3).integers(0, k, size=n)

    report = evaluate(probs, y)
    assert report.usable is False
    assert isinstance(report.metrics["accuracy"], NotRun)
    assert "degenerate" in report.metrics["accuracy"].reason.lower()
    # And there is genuinely no number to misread.
    assert not hasattr(report.metrics["accuracy"], "value")


def test_a_healthy_run_does_report_metrics():
    probs, y = _healthy_probs()
    report = evaluate(probs, y, features=RNG.normal(0, 1, size=(len(probs), 16)))
    assert report.usable is True
    acc = report.metrics["accuracy"]
    assert isinstance(acc, Ran) and 0.5 < acc.value < 1.0


def test_missing_baseline_is_not_run_not_a_win():
    probs, y = _healthy_probs()
    report = evaluate(probs, y)
    assert isinstance(report.metrics["paired_margin_vs_linear"], NotRun)


# --------------------------------------------------------------------------
# The paired margin
# --------------------------------------------------------------------------

def test_a_clear_win_passes_the_paired_test():
    rng = np.random.default_rng(11)
    base = rng.random(600) < 0.55
    model = base | (rng.random(600) < 0.35)  # strictly dominates
    result = paired_margin_test(model, base)
    assert isinstance(result, Ran) and result.passed is True
    assert result.value > 0


def test_a_tie_does_not_pass():
    rng = np.random.default_rng(12)
    a = rng.random(600) < 0.6
    b = rng.random(600) < 0.6
    result = paired_margin_test(a, b)
    assert isinstance(result, Ran) and result.passed is False
    assert "not a win" in result.detail


def test_a_positive_point_estimate_with_a_straddling_ci_is_not_a_win():
    """The specific failure the CI exists to prevent: a small lead on few examples."""
    model = np.array([True] * 11 + [False] * 9)
    base = np.array([True] * 9 + [False] * 11)
    result = paired_margin_test(model, base)
    assert isinstance(result, Ran)
    assert result.value > 0, "point estimate favours the model"
    assert result.passed is False, "but 20 examples cannot establish it"


def test_mismatched_shapes_are_not_run_not_false():
    assert isinstance(paired_margin_test(np.ones(5, bool), np.ones(7, bool)), NotRun)


# --------------------------------------------------------------------------
# Split conformal — margin, not entropy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_sets_achieve_their_coverage(alpha: float):
    probs, y = _healthy_probs(n=4000, seed=21)
    cal, test = slice(0, 2000), slice(2000, 4000)
    thr = split_conformal_threshold(probs[cal], y[cal], alpha=alpha)
    sets = conformal_sets(probs[test], thr)
    covered = np.mean([y[test][i] in s for i, s in enumerate(sets)])
    # Split conformal guarantees >= 1-alpha marginal coverage; allow finite-sample slack.
    assert covered >= (1 - alpha) - 0.03, f"coverage {covered:.3f} below 1-alpha={1 - alpha}"


def test_an_empty_conformal_set_is_allowed_and_not_widened():
    """An empty set means 'confident about nothing at this level' -> noul.

    It is information. Silently widening it to all classes would turn an abstention
    into a maximally uninformative answer that still looks like an answer.
    """
    probs = np.array([[0.26, 0.25, 0.25, 0.24]])
    assert conformal_sets(probs, 0.5) == [[]]


# --------------------------------------------------------------------------
# The linear baseline: the control arm
# --------------------------------------------------------------------------

def test_baseline_learns_a_separable_signal():
    docs = [f"fn compute_{i}() {{ todo!() }}" for i in range(60)]
    docs += [f"fn compute_{i}() {{ let x = {i}; x * 2 }}" for i in range(60)]
    labels = ["stub"] * 60 + ["clean"] * 60
    clf = LinearBaseline(hasher=CharNGramHasher(dim=4096), seed=0)
    clf.fit(docs, labels)
    acc = np.mean([p == t for p, t in zip(clf.predict(docs), labels, strict=True)])
    assert acc > 0.9, f"baseline only reached {acc:.3f} on a separable task; it is too weak to be a control"


def test_baseline_is_deterministic_across_fits():
    docs = [f"fn a_{i}() {{ todo!() }}" for i in range(30)] + [f"fn b_{i}() {{ {i} }}" for i in range(30)]
    labels = ["stub"] * 30 + ["clean"] * 30
    p1 = LinearBaseline(hasher=CharNGramHasher(dim=4096), seed=5).fit(docs, labels)
    p2 = LinearBaseline(hasher=CharNGramHasher(dim=4096), seed=5).fit(docs, labels)
    assert np.allclose(p1.weights, p2.weights)


def test_an_unconverged_baseline_is_not_run_not_a_low_score():
    """A model cannot beat a control that never finished training."""
    docs = [f"fn a_{i}() {{ todo!() }}" for i in range(30)] + [f"fn b_{i}() {{ {i} }}" for i in range(30)]
    labels = ["stub"] * 30 + ["clean"] * 30
    clf = LinearBaseline(hasher=CharNGramHasher(dim=4096), seed=0, max_iter=2, tol=1e-12)  # cannot possibly converge
    clf.fit(docs, labels)
    conv = clf.convergence()
    assert isinstance(conv, NotRun)
    assert "did not converge" in conv.reason


def test_unfitted_baseline_refuses_to_predict():
    with pytest.raises(RuntimeError, match="fit\\(\\) before"):
        LinearBaseline().predict(["anything"])


def test_hasher_is_stable_across_processes():
    """FNV-1a, not Python's hash(): salted hash() would change every run."""
    a = CharNGramHasher(dim=1024).transform(["fn main() { todo!() }"])
    b = CharNGramHasher(dim=1024).transform(["fn main() { todo!() }"])
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.indptr, b.indptr)
    assert np.allclose(a.data, b.data)
    assert a.data.size > 0


def test_hasher_rejects_non_power_of_two_dim():
    with pytest.raises(ValueError, match="power of two"):
        CharNGramHasher(dim=1000)


def test_empty_document_does_not_produce_nan():
    """A zero row must stay absent rather than becoming NaN under L2 normalization."""
    out = CharNGramHasher(dim=256).transform(["", "fn x() {}"])
    assert not np.isnan(out.data).any()
    assert out.indptr[1] - out.indptr[0] == 0, "the empty doc must contribute no nonzeros"
    assert out.indptr[2] - out.indptr[1] > 0, "the real doc must contribute some"
    # And it must survive a forward pass without poisoning the other row.
    W = np.ones((256, 3), dtype=np.float64)
    assert not np.isnan(out.matmul(W)).any()


def test_csr_matmul_matches_a_dense_reference():
    """The sparse path is the only path now; prove it against dense arithmetic."""
    docs = ["fn compute() { todo!() }", "", "let x = 1; let y = 2;"]
    csr = CharNGramHasher(dim=512).transform(docs)
    dense = np.zeros((len(docs), 512))
    for i in range(len(docs)):
        for j in range(csr.indptr[i], csr.indptr[i + 1]):
            dense[i, csr.indices[j]] = csr.data[j]

    rng = np.random.default_rng(0)
    W = rng.normal(size=(512, 5))
    assert np.allclose(csr.matmul(W), dense @ W)

    D = rng.normal(size=(len(docs), 5))
    assert np.allclose(csr.rmatmul(D), dense.T @ D)


def test_csr_select_preserves_rows():
    docs = [f"fn f{i}() {{ {i} }}" for i in range(6)]
    csr = CharNGramHasher(dim=512).transform(docs)
    picked = np.array([4, 1, 0])
    sub = csr.select(picked)
    assert sub.shape == (3, 512)
    W = np.random.default_rng(1).normal(size=(512, 3))
    assert np.allclose(sub.matmul(W), csr.matmul(W)[picked])
