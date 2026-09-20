"""Teacher agreement: Cohen's kappa against the hand labels, with its uncertainty.

The plan's gate is *"the teacher must reach Cohen's kappa >= 0.6 against 50 hand
labels. Below that, the rubric is rewritten, not the model."*

That gate is measured on **50 items**, and a point estimate from 50 items is not a
decision — it is a decision with a confidence interval wide enough to contain both
outcomes. This module therefore always reports the interval alongside the estimate,
and `kappa_gate` states explicitly whether the interval settles the question.

It does **not** move the 0.6 threshold. Thresholds are read-only (repo rule 2): an
agent may report that a gate is underpowered; it may not retune it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tristate import Ran, TriState

__all__ = [
    "KappaResult",
    "cohens_kappa",
    "confusion",
    "disagreements",
    "kappa_gate",
    "kappa_with_ci",
]


@dataclass(frozen=True, slots=True)
class KappaResult:
    kappa: float
    p_observed: float
    p_expected: float
    n: int
    ci_low: float
    ci_high: float
    ci_level: float
    classes: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"kappa={self.kappa:.3f} "
            f"[{int(self.ci_level * 100)}% CI {self.ci_low:.3f}, {self.ci_high:.3f}] "
            f"on n={self.n} (p_o={self.p_observed:.3f}, p_e={self.p_expected:.3f})"
        )


def _classes(a: list[str], b: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(a) | set(b)))


def cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """(p_o - p_e) / (1 - p_e).

    Degenerate case worth naming: when both raters put everything in one class,
    p_e == 1 and kappa is 0/0. That is *perfect agreement carrying no information*,
    not agreement by chance, so it returns 0.0 rather than 1.0 — a rubric that makes
    the teacher answer one class every time must not read as a passing gate.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"raters disagree in length: {len(rater_a)} vs {len(rater_b)}")
    if not rater_a:
        raise ValueError("cannot compute kappa on zero items")

    classes = _classes(rater_a, rater_b)
    idx = {c: i for i, c in enumerate(classes)}
    k, n = len(classes), len(rater_a)

    m = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(rater_a, rater_b, strict=True):
        m[idx[x], idx[y]] += 1.0

    p_o = float(np.trace(m) / n)
    p_e = float(np.sum(m.sum(axis=0) * m.sum(axis=1)) / (n * n))
    if abs(1.0 - p_e) < 1e-12:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def kappa_with_ci(
    rater_a: list[str],
    rater_b: list[str],
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> KappaResult:
    """Kappa plus a bootstrap CI over items.

    Bootstrapping over *items* is the right resample unit: the uncertainty being
    measured is "would another 50 diffs have given the same answer", which is what
    the gate actually asks.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"raters disagree in length: {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n < 2:
        raise ValueError(f"need at least 2 items for an interval, got {n}")

    point = cohens_kappa(rater_a, rater_b)
    a = np.asarray(rater_a, dtype=object)
    b = np.asarray(rater_b, dtype=object)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, n, size=n)
        boots[i] = cohens_kappa(list(a[pick]), list(b[pick]))

    alpha = 1.0 - level
    classes = _classes(rater_a, rater_b)
    m = np.zeros((len(classes), len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for x, y in zip(rater_a, rater_b, strict=True):
        m[idx[x], idx[y]] += 1.0

    return KappaResult(
        kappa=point,
        p_observed=float(np.trace(m) / n),
        p_expected=float(np.sum(m.sum(axis=0) * m.sum(axis=1)) / (n * n)),
        n=n,
        ci_low=float(np.quantile(boots, alpha / 2)),
        ci_high=float(np.quantile(boots, 1 - alpha / 2)),
        ci_level=level,
        classes=classes,
    )


def kappa_gate(result: KappaResult, *, threshold: float = 0.6) -> TriState:
    """Does the teacher clear the agreement gate, and does the evidence settle it?

    Three outcomes, and the middle one is the honest answer the plan's binary
    phrasing cannot express:

    - CI entirely above the threshold -> `Ran(passed=True)`. Settled.
    - CI entirely below -> `Ran(passed=False)`. Settled: rewrite the rubric.
    - CI straddles the threshold -> `Ran` carrying the point verdict, with the
      detail stating plainly that this sample size does not settle it.

    The straddle is not an edge case at n=50; it is the *expected* result for any
    kappa near 0.6. Reporting 0.62 as a pass, with no interval, is how an
    underpowered gate becomes a confident decision.
    """
    settled_pass = result.ci_low >= threshold
    settled_fail = result.ci_high < threshold
    passed = result.kappa >= threshold

    if settled_pass:
        detail = f"{result} -- clears {threshold} across the whole interval"
    elif settled_fail:
        detail = f"{result} -- below {threshold} across the whole interval; rewrite the rubric"
    else:
        detail = (
            f"{result} -- UNDERPOWERED: the interval straddles {threshold}, so n={result.n} "
            f"does not settle this either way. The point estimate says "
            f"{'pass' if passed else 'fail'}. More hand labels, or accept a weaker guarantee "
            f"and say so in the ledger."
        )
    return Ran(passed=passed, value=result.kappa, n=result.n, n_total=result.n, detail=detail)


def confusion(rater_a: list[str], rater_b: list[str]) -> tuple[tuple[str, ...], np.ndarray]:
    """Confusion matrix, rows = rater_a, cols = rater_b."""
    classes = _classes(rater_a, rater_b)
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for x, y in zip(rater_a, rater_b, strict=True):
        m[idx[x], idx[y]] += 1
    return classes, m


def disagreements(
    items: list[str], human: list[str], teacher: list[str], *, limit: int = 50
) -> list[dict[str, str]]:
    """Where the two disagree — the only part of the run that improves the rubric.

    Kappa says *whether* to rewrite the rubric; this says *what to rewrite*. Capped,
    and the caller is told both numbers so a truncated list is never read as the
    complete set of disagreements.
    """
    if not (len(items) == len(human) == len(teacher)):
        raise ValueError(
            f"lengths differ: items={len(items)}, human={len(human)}, teacher={len(teacher)}"
        )
    out = [
        {"item": it, "human": h, "teacher": t}
        for it, h, t in zip(items, human, teacher, strict=True)
        if h != t
    ]
    total = len(out)
    shown = out[:limit]
    for row in shown:
        row["_shown"] = f"{len(shown)} of {total} disagreements"
    return shown
