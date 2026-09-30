"""The eval harness, and the controls that decide whether its numbers mean anything.

S7 ("test the tests") is the reason this module is written before any real eval:
the harness is fed a constant-prediction model and a shuffled-label model, and it
**must fail them**. A harness that passes a broken model produces numbers that look
exactly like real ones.

The plan's rule — *"No number is reported while it fails"* — is enforced structurally
here: when the degenerate-head assertion fails, `EvalReport.metrics` come back as
`NotRun` carrying that reason. A caller cannot read an accuracy off a degenerate run,
because there is no accuracy to read.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np

from .tristate import NotRun, Ran, TriState

__all__ = [
    "DEFAULT_ENTROPY_FLOOR",
    "DEFAULT_FEATURE_VAR_FLOOR",
    "DEFAULT_MAX_CLASS_SHARE",
    "PERMUTATION_CONSISTENCY_FLOOR",
    "EvalReport",
    "conformal_sets",
    "degenerate_head_check",
    "evaluate",
    "paired_margin_by_key",
    "paired_margin_test",
    "permutation_consistency_state",
    "shuffled_label_control",
    "split_conformal_threshold",
]

# Floors from the plan's "degenerate-head assertion". They are thresholds, and
# thresholds are read-only to an agent (rule 2): report a failure, never retune.
DEFAULT_ENTROPY_FLOOR = 0.15      # nats, mean predictive entropy
DEFAULT_MAX_CLASS_SHARE = 0.95    # no single predicted class above this share
DEFAULT_FEATURE_VAR_FLOOR = 1e-6  # mean per-dim variance of pooled features

#: The gate from `docs/schema-api.md`: "permutation consistency (>= 95%) is a training gate
#: and not only a runtime check: a model that fails it makes the second pass fire constantly
#: and the abstain rate blows the cap." A threshold from the plan, read-only to an agent.
#: Owned here, torch-free, because rung 0 and the FT eval both gate on it.
PERMUTATION_CONSISTENCY_FLOOR = 0.95


def permutation_consistency_state(*, agree: int, asked: int, total: int) -> TriState:
    """The ``permutation_consistency`` gate from its counts, one wording for every caller.

    ``asked`` rows had a derangement (two or more options) and were decoded in both orders;
    ``agree`` of them gave the same answer. ``total - asked`` rows had no derangement: they
    leave through the denominator, never the numerator, because a row that cannot disagree
    would inflate the rate with a question it was never asked.
    """
    if not 0 <= agree <= asked <= total:
        raise ValueError(f"counts must satisfy 0 <= agree <= asked <= total, got "
                         f"{agree}, {asked}, {total}")
    if asked == 0:
        return NotRun(
            reason=(
                f"no row of {total} had two or more live options, so no derangement exists "
                "and permutation consistency was not measured on anything"
            )
        )
    rate = agree / asked
    return Ran(
        passed=rate >= PERMUTATION_CONSISTENCY_FLOOR,
        value=rate,
        n=agree,
        n_total=asked,
        detail=(
            f"the choice head agreed with itself across a derangement on {agree} of "
            f"{asked} rows ({rate:.1%}) against a {PERMUTATION_CONSISTENCY_FLOOR:.0%} "
            f"floor; {total - asked} row(s) had fewer than two live options and were "
            "excluded rather than counted as agreeing"
        ),
    )


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def degenerate_head_check(
    probs: np.ndarray,
    *,
    features: np.ndarray | None = None,
    entropy_floor: float = DEFAULT_ENTROPY_FLOOR,
    max_class_share: float = DEFAULT_MAX_CLASS_SHARE,
    feature_var_floor: float = DEFAULT_FEATURE_VAR_FLOOR,
) -> TriState:
    """Is this head actually discriminating, or has it collapsed?

    Three independent symptoms, because each alone has a blind spot: a constant head
    with a soft distribution passes the entropy check but fails the class-share one;
    a confidently-wrong head passes class-share but fails entropy; a head reading
    dead features fails neither but has nothing to read.
    """
    if probs.ndim != 2 or probs.shape[0] == 0:
        return NotRun(
            reason=f"degenerate check needs a 2-D non-empty probs array, got {probs.shape}"
        )

    failures: list[str] = []

    mean_entropy = float(_entropy(probs).mean())
    if mean_entropy < entropy_floor:
        failures.append(f"mean predictive entropy {mean_entropy:.4f} < floor {entropy_floor}")

    preds = np.argmax(probs, axis=1)
    counts = np.bincount(preds, minlength=probs.shape[1])
    top_share = float(counts.max() / counts.sum())
    if top_share > max_class_share:
        failures.append(
            f"predicted class {int(counts.argmax())} takes {top_share:.3f} of predictions "
            f"> max {max_class_share}"
        )

    if features is not None:
        if features.ndim != 2 or features.shape[0] == 0:
            return NotRun(reason=f"features must be 2-D non-empty, got {features.shape}")
        var = float(np.var(features, axis=0).mean())
        if var < feature_var_floor:
            failures.append(f"mean feature variance {var:.3e} < floor {feature_var_floor:.1e}")

    detail = (
        f"entropy={mean_entropy:.4f}, top_class_share={top_share:.3f}"
        + (
            ""
            if features is None
            else f", feature_var={float(np.var(features, axis=0).mean()):.3e}"
        )
    )
    if failures:
        return Ran(passed=False, value=mean_entropy, n=len(probs), n_total=len(probs),
                   detail="DEGENERATE: " + "; ".join(failures))
    return Ran(passed=True, value=mean_entropy, n=len(probs), n_total=len(probs), detail=detail)


def shuffled_label_control(
    shuffled_accuracy: float,
    labels: list[str] | np.ndarray,
    *,
    n_eval: int,
    z: float = 3.0,
) -> TriState:
    """A model trained on shuffled labels must fall to chance.

    Chance is the **majority-class rate**, not 1/k: on an imbalanced set a model that
    learned nothing still scores the majority rate by always guessing it. Using 1/k
    would set the bar below chance and let real leakage through.

    Failing this means the split leaks — through file paths, repo names, or
    near-duplicates. The split is rebuilt before any other run.
    """
    if n_eval <= 0:
        return NotRun(reason="shuffled-label control had no evaluation examples")
    arr = np.asarray(labels)
    if arr.size == 0:
        return NotRun(reason="shuffled-label control had no labels to establish the chance rate")

    _, counts = np.unique(arr, return_counts=True)
    chance = float(counts.max() / counts.sum())
    # Binomial standard error at the chance rate; z=3 is ~99.7%.
    se = math.sqrt(max(chance * (1.0 - chance), 1e-12) / n_eval)
    ceiling = chance + z * se

    passed = shuffled_accuracy <= ceiling
    return Ran(
        passed=passed,
        value=shuffled_accuracy,
        n=n_eval,
        n_total=n_eval,
        detail=(
            f"shuffled-label accuracy {shuffled_accuracy:.4f} vs chance (majority) {chance:.4f}, "
            f"ceiling {ceiling:.4f} at z={z}"
            + ("" if passed else " -- LEAKAGE: rebuild the split before any other run")
        ),
    )


def paired_margin_test(
    model_correct: np.ndarray,
    baseline_correct: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> TriState:
    """Does the model beat the baseline by a *paired* margin?

    Paired over the same examples, by bootstrap over example indices. Pairing is the
    point: the two arms see identical items, so example difficulty cancels and the
    test measures the difference rather than the variance of the set.

    Passes only when the lower bound of the CI on (model - baseline) is above zero:
    a positive point estimate whose interval straddles zero is not a win.
    """
    m = np.asarray(model_correct, dtype=bool)
    b = np.asarray(baseline_correct, dtype=bool)
    if m.shape != b.shape:
        return NotRun(reason=f"paired test needs equal shapes, got {m.shape} vs {b.shape}")
    if m.size == 0:
        return NotRun(reason="paired test had no examples")

    diff = m.astype(np.float64) - b.astype(np.float64)
    point = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))

    passed = lo > 0.0
    # Three outcomes, not two. `passed` is a bool and the gate is right to be, but the
    # DETAIL is read by people, and "not a win" covers two very different findings: an
    # interval straddling zero says the comparison could not separate them, and an interval
    # entirely below zero says the control won and by how much. Appending the straddling
    # sentence to both was a false statement in an append-only ledger, and it stayed
    # invisible until 2026-09-21 because the gate had never once been evaluated -- the
    # first decisive loss it ever scored was also the first time the sentence was wrong.
    if passed:
        verdict = ""
    elif hi < 0.0:
        verdict = (
            f" -- the whole interval is below zero, so this is not an inconclusive result: "
            f"the baseline beats the model by {-point:.4f} and the comparison separates them"
        )
    else:
        verdict = " -- CI includes zero, so this is not a win"
    return Ran(
        passed=passed,
        value=point,
        n=len(diff),
        n_total=len(diff),
        detail=(
            f"paired margin {point:+.4f}, {int((1 - alpha) * 100)}% CI [{lo:+.4f}, {hi:+.4f}] "
            f"over {n_boot} bootstrap resamples" + verdict
        ),
    )


def paired_margin_by_key(
    model_correct: Mapping[Hashable, bool],
    baseline_correct: Mapping[Hashable, bool],
    *,
    seed: int = 0,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> TriState:
    """:func:`paired_margin_test` over two arms scored by row key rather than by position.

    Positional pairing is only sound when both arms enumerated the rows in one order, which
    is true inside one process and false across two: the FT model's verdicts come from the
    shard reader's bucketed batch order and the control's from the renderer's row order. So
    the pairing is done here by key, and **the key sets must be identical**. A row one arm
    scored and the other did not is not a row either arm can be credited with -- dropping it
    silently would let a refusal on one side reshape the population the margin is over. So a
    mismatch is ``NotRun`` naming both counts and an example of each, never an intersection.
    """
    model_keys = set(model_correct)
    base_keys = set(baseline_correct)
    if model_keys != base_keys:
        only_model = sorted(map(repr, model_keys - base_keys))
        only_base = sorted(map(repr, base_keys - model_keys))
        return NotRun(
            reason=(
                f"the two arms scored different rows: {len(only_model)} only by the model "
                f"(e.g. {only_model[:3]}), {len(only_base)} only by the control "
                f"(e.g. {only_base[:3]}). A paired margin needs the identical population, "
                "so nothing was paired"
            )
        )
    keys = sorted(model_keys, key=repr)
    return paired_margin_test(
        np.array([bool(model_correct[k]) for k in keys], dtype=bool),
        np.array([bool(baseline_correct[k]) for k in keys], dtype=bool),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )


def split_conformal_threshold(
    cal_probs: np.ndarray, cal_labels: np.ndarray, *, alpha: float = 0.1
) -> float:
    """Split-conformal threshold, returned on the **nonconformity** scale.

    The audits attribute "entropy as confidence" to the Laya spec as a defect, and no
    entropy is computed here. The nonconformity score is ``1 - p(true class)``: a direct
    statement about the true label's calibrated probability, which is what a coverage
    guarantee needs.

    **Read the scale before using the return value.** This is ``q̂`` on the nonconformity
    scale, so it is *large* for a weak model. Three different numbers in this repo get
    called a threshold:

    * this ``q̂``, on the nonconformity scale;
    * ``CalibrationEntry.conformal_quantile`` in ``crates/qd-runtime/src/calibration.rs``,
      which is a **probability** cutoff (``p >= conformal_quantile``) and equals ``1 - q̂``;
    * the **margin** ``p_top - p_second``, which is the reported ``score`` and the basis for
      ``noul_margin`` — and is neither of the above.

    :func:`conformal_sets` takes this value directly and converts internally. Anything
    crossing to the Rust table must go through
    :func:`qd_train.calibration_fit.probability_cutoff_from_nonconformity`; passing ``q̂``
    straight into ``conformal_quantile`` silently breaks coverage, and which direction it
    breaks depends on the model's accuracy.

    This docstring previously opened "threshold on the *margin*", which described the third
    quantity while the body computed the first.
    """
    if cal_probs.ndim != 2 or len(cal_probs) == 0:
        raise ValueError(f"calibration probs must be 2-D non-empty, got {cal_probs.shape}")
    if len(cal_probs) != len(cal_labels):
        raise ValueError(f"probs/labels length mismatch: {len(cal_probs)} vs {len(cal_labels)}")

    scores = 1.0 - cal_probs[np.arange(len(cal_labels)), np.asarray(cal_labels, dtype=int)]
    n = len(scores)
    # Finite-sample correction: the (n+1)(1-alpha)/n quantile gives >= 1-alpha coverage.
    q = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q, method="higher"))


def conformal_sets(probs: np.ndarray, threshold: float) -> list[list[int]]:
    """Every class whose calibrated probability clears the conformal threshold.

    A set may be empty (the model is confident about nothing at this level) — which
    the runtime maps to `noul`. An empty set is information, not a failure, and is
    never silently widened to "all classes".
    """
    return [np.flatnonzero(row >= 1.0 - threshold).tolist() for row in probs]


@dataclass(frozen=True, slots=True)
class EvalReport:
    degenerate: TriState
    metrics: dict[str, TriState]
    controls: dict[str, TriState]

    @property
    def usable(self) -> bool:
        return isinstance(self.degenerate, Ran) and self.degenerate.passed

    def summary(self) -> str:
        head = "USABLE" if self.usable else "NOT USABLE (degenerate head)"
        lines = [f"eval report: {head}", f"  degenerate: {self.degenerate}"]
        lines += [f"  metric {k}: {v}" for k, v in sorted(self.metrics.items())]
        lines += [f"  control {k}: {v}" for k, v in sorted(self.controls.items())]
        return "\n".join(lines)


def evaluate(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    features: np.ndarray | None = None,
    baseline_correct: np.ndarray | None = None,
    shuffled_accuracy: float | None = None,
    seed: int = 0,
) -> EvalReport:
    """Run the harness. **No number is reported while the degenerate check fails.**

    That rule is enforced here rather than left to the caller: when the head is
    degenerate every metric comes back `NotRun` with that reason, so a report
    cannot quote an accuracy from a collapsed model.
    """
    degenerate = degenerate_head_check(probs, features=features)
    y = np.asarray(labels, dtype=int)

    controls: dict[str, TriState] = {}
    if shuffled_accuracy is None:
        controls["shuffled_label"] = NotRun(reason="no shuffled-label model was evaluated")
    else:
        controls["shuffled_label"] = shuffled_label_control(
            shuffled_accuracy, y, n_eval=len(y)
        )
    controls["degenerate_head"] = degenerate

    if not (isinstance(degenerate, Ran) and degenerate.passed):
        reason = (
            degenerate.reason if isinstance(degenerate, NotRun)
            else f"head is degenerate ({degenerate.detail}); "
            "no metric is reported from a collapsed head"
        )
        return EvalReport(
            degenerate=degenerate,
            metrics={
                "accuracy": NotRun(reason=reason),
                "paired_margin_vs_linear": NotRun(reason=reason),
            },
            controls=controls,
        )

    preds = np.argmax(probs, axis=1)
    correct = preds == y
    metrics: dict[str, TriState] = {
        "accuracy": Ran(passed=True, value=float(correct.mean()), n=len(y), n_total=len(y))
    }
    if baseline_correct is None:
        metrics["paired_margin_vs_linear"] = NotRun(
            reason="the linear baseline was not evaluated on these examples, "
            "so there is nothing to pair against"
        )
    else:
        metrics["paired_margin_vs_linear"] = paired_margin_test(
            correct, baseline_correct, seed=seed
        )

    return EvalReport(degenerate=degenerate, metrics=metrics, controls=controls)
