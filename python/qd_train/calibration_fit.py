"""Fitting the calibration table that ``qd-runtime`` serves.

``GAP-RT-CALIBRATION-NOT-FITTED``: ``crates/qd-runtime/src/calibration.rs`` ships
``CalibrationTable::reference()``, named ``"reference-uncalibrated-v1"``, with
``temperature`` pinned at ``1.0``, ``conformal_quantile`` hand-set to ``0.5 / rows`` and
``noul_margin`` hand-set to ``0.02``. Those are placeholders, and the name says so. This
module fits them from held-out data and emits exactly the shape that table deserialises.

It also implements ``ece``, which ``qd_train.ledger`` has listed as a **gate** since the
schema was written while nothing in the repo computed it. A gate whose number no code
produces is a gate that gets filled in by hand.

Why this matters beyond housekeeping: the nearest comparable system
(``AUDIT/prior-art-jev-nimble-2026-09-19.md``) explicitly disclaims calibration — softmax,
temperature fixed at 1.0, and a README warning that 0.9 does not mean right 90% of the
time. Calibrated abstention is the differentiator, and right now ours is *also* unfitted.

## Three quantities, two names — read this before touching the conversion

This is the fifth instance in this repo of "one name, two quantities, both sides green"
(after ``GAP-RT-WIRE-CONTEXT-ENCODING``, ``GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS``, the S2
parity gate, and the span gold/candidate pair). It is latent rather than live only because
nothing fits the table yet — this module is what would have made it live.

Three distinct numbers are in play, and two of them are called a "threshold":

1. **Nonconformity quantile** ``q̂`` — what
   :func:`qd_train.eval_harness.split_conformal_threshold` returns. Its score is
   ``1 - p(true class)``, so ``q̂`` lives on the *nonconformity* scale and is **small** for
   a good model (around ``alpha``).
2. **Probability cutoff** ``1 - q̂`` — what ``CalibrationEntry.conformal_quantile`` means in
   Rust: *"rows at or above this calibrated probability enter the set"*, implemented as
   ``p >= entry.conformal_quantile``. It is **large** for a good model.
3. **Margin** ``p_top - p_second`` — the reported ``score``, and the basis for
   ``noul_margin``. Not either of the above, and not entropy.

``split_conformal_threshold``'s own docstring says it thresholds "on the *margin*", while
its body computes ``1 - p(true class)``. That is (1) described as (3).

The trap: ``conformal_quantile = split_conformal_threshold(...)`` is the obvious line to
write, and it is wrong by ``1 - x``. **Which way it breaks depends on the model**, measured
on synthetic data in ``test_calibration_fit.py`` at ``alpha = 0.1`` (target coverage 0.90):

=================  =====  ==========================================================
accuracy            ``q̂``  effect of using ``q̂`` as the cutoff
=================  =====  ==========================================================
0.618               0.86   sets collapse; coverage **0.093**
0.888               0.55   coverage 0.859, just under the target
0.927               0.37   sets widen; coverage 0.947, and far less informative
=================  =====  ==========================================================

``q̂`` is the quantile of ``1 - p(true class)``, so it is large exactly when the model is
weak. A strong model drives it below ``0.5`` and the error flips from under-covering to
over-covering.

**The worst case is neither end.** At ``q̂ = 0.5`` the two scales coincide and the bug has
*no effect at all*. A test fixture that happens to sit near that accuracy passes with the
bug fully present — which is why the regression test below pins both regimes rather than
picking one.

:func:`probability_cutoff_from_nonconformity` is the only supported conversion, and
:func:`fit_entry` takes the nonconformity quantile rather than a ready-made cutoff so that
a caller holding ``q̂`` cannot skip it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schema_mirror import CHOICE, MAX_OPTIONS, MIN_OPTIONS, RESERVED_NOUL_ROWS, SCORE, SPAN
from .tristate import NotRun, Ran, TriState

__all__ = [
    "CHOICE",
    "MAX_OPTIONS",
    "MIN_OPTIONS",
    "RESERVED_NOUL_ROWS",
    "SCORE",
    "SPAN",
    "CalibrationEntryFit",
    "ece_gate",
    "expected_calibration_error",
    "fit_entry",
    "fit_noul_margin",
    "fit_table",
    "fit_temperature",
    "letters_key",
    "probability_cutoff_from_nonconformity",
]

# The schema constants are imported from `qd_train.schema_mirror` and re-exported through
# `__all__` above, so existing `from .calibration_fit import CHOICE` sites keep working.
# `test_schema_mirror.py` pins every one of them against schema.rs.


def letters_key(kind: str, rows: int) -> str:
    """The Rust table's key: ``letters_key`` in calibration.rs is ``"{kind}:{rows}"``.

    ``rows`` is the **total decode rows including the reserved noul row**
    (``schema.rs:114``: ``options.len() + RESERVED_NOUL_ROWS``), not the number of named
    options. Passing the named count silently fits the wrong entry, and every lookup for
    the real shape then refuses with ``CalibrationEntryMissing`` — which at least fails
    loudly, unlike the conformal-scale bug above.
    """
    if kind not in (CHOICE, SCORE):
        raise ValueError(f"letters_key is for choice/score only, got {kind!r}; span has one entry")
    if rows < MIN_OPTIONS + RESERVED_NOUL_ROWS:
        raise ValueError(f"rows={rows} is below the minimum {MIN_OPTIONS} options plus noul")
    return f"{kind}:{rows}"


def _check_probs(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError(f"probs must be 2-D and non-empty, got shape {p.shape}")
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"probs/labels length mismatch: {p.shape[0]} vs {y.shape[0]}")
    if not np.isfinite(p).all():
        raise ValueError("probs contain a non-finite value")
    if y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError(f"labels out of range for {p.shape[1]} columns")
    return p, y


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> tuple[float, int]:
    """Equal-width binned ECE over the top-class confidence, and the bins that had mass.

    Returns ``(ece, n_populated_bins)``. The second value is not decoration: ECE over a
    distribution concentrated in one bin is a statement about one bin, and reporting the
    scalar alone would hide that. An empty bin contributes nothing and is not counted as
    agreement.
    """
    p, y = _check_probs(probs, labels)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    # Right-closed bins so that confidence exactly 1.0 lands in the last bin rather than
    # falling outside it.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)

    total = len(conf)
    ece = 0.0
    populated = 0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        populated += 1
        ece += (count / total) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece, populated


def ece_gate(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.05,
    n_bins: int = 15,
    min_samples: int = 100,
) -> TriState:
    """The ``ece`` gate ``qd_train.ledger`` has always listed and nothing ever computed.

    Refuses rather than reports below ``min_samples``: an ECE over a handful of examples is
    dominated by binning noise, and a small-sample number that clears the bar is exactly
    the "approved means unexamined" failure the repo rules name.
    """
    p, y = _check_probs(probs, labels)
    n = len(y)
    if n < min_samples:
        return NotRun(
            reason=(
                f"ECE needs at least {min_samples} examples to be meaningful; got {n}. "
                "A binned calibration error over a small sample measures the binning."
            )
        )
    ece, populated = expected_calibration_error(p, y, n_bins=n_bins)
    return Ran(
        passed=ece <= threshold,
        value=ece,
        n=n,
        n_total=n,
        detail=(
            f"ECE {ece:.4f} against a {threshold:.4f} bar, {n_bins} bins of which "
            f"{populated} carried mass"
        ),
    )


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 200,
) -> float:
    """Temperature that minimises held-out NLL, by ternary search on ``log T``.

    Dependency-free on purpose: scipy is present in this venv but is not a declared
    dependency of the package, and a one-dimensional bounded search is 20 lines.

    The search is on ``log T`` because the NLL is far closer to symmetric there — halving
    and doubling the temperature are the comparable perturbations, not ``+-0.5``.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError(f"logits must be 2-D and non-empty, got shape {z.shape}")
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"logits/labels length mismatch: {z.shape[0]} vs {y.shape[0]}")
    if not np.isfinite(z).all():
        raise ValueError("logits contain a non-finite value")
    if not 0.0 < lo < hi:
        raise ValueError(f"need 0 < lo < hi, got lo={lo} hi={hi}")

    rows = np.arange(len(y))

    def nll(t: float) -> float:
        s = z / t
        s = s - s.max(axis=1, keepdims=True)
        return float(-(s[rows, y] - np.log(np.exp(s).sum(axis=1))).mean())

    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        if nll(math.exp(m1)) < nll(math.exp(m2)):
            b = m2
        else:
            a = m1
    return float(math.exp((a + b) / 2.0))


def fit_span_temperature(
    start: list[np.ndarray],
    end: list[np.ndarray],
    gold_start: list[int],
    gold_end: list[int],
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 200,
) -> float:
    """The span entry's temperature: :func:`fit_temperature`'s search over ragged rows.

    A span slot's rows are one per line start plus the abstention, so each row is as wide as
    its context has lines and no 2-D array holds them. One temperature serves both pointers
    (``answer.rs`` reads both through the table's one span entry), so the NLL is the mean over
    ``2n`` pointer rows interleaved ``start_0, end_0, start_1, ...``, each at its own gold --
    the same per-row arithmetic as :func:`fit_temperature`'s ``nll``, row by row. Written for
    ``GAP-CALIB-SPAN-NOT-FITTABLE-FROM-VERDICTS``; ``qd-calib-fit``'s ``fit_span_temperature``
    is the Rust owner this pins.
    """
    if not (len(start) == len(end) == len(gold_start) == len(gold_end)) or not start:
        raise ValueError("start/end/gold_start/gold_end must be equal-length and non-empty")
    rows: list[np.ndarray] = []
    golds: list[int] = []
    for s, e, gs, ge in zip(start, end, gold_start, gold_end, strict=True):
        for z, g in ((s, gs), (e, ge)):
            z = np.asarray(z, dtype=np.float64)
            if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all():
                raise ValueError("each pointer row must be 1-D, finite, with at least 2 rows")
            if not 0 <= int(g) < len(z):
                raise ValueError(f"gold row {g} is outside {len(z)} pointer rows")
            rows.append(z)
            golds.append(int(g))
    if not 0.0 < lo < hi:
        raise ValueError(f"need 0 < lo < hi, got lo={lo} hi={hi}")

    def nll(t: float) -> float:
        per_row = np.empty(len(rows), dtype=np.float64)
        for i, (z, y) in enumerate(zip(rows, golds, strict=True)):
            s = z / t
            s = s - s.max()
            per_row[i] = s[y] - np.log(np.exp(s).sum())
        return float(-per_row.mean())

    a, b = math.log(lo), math.log(hi)
    for _ in range(iters):
        m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        if nll(math.exp(m1)) < nll(math.exp(m2)):
            b = m2
        else:
            a = m1
    return float(math.exp((a + b) / 2.0))


def probability_cutoff_from_nonconformity(q_hat: float) -> float:
    """Convert a nonconformity quantile to the probability cutoff Rust expects.

    The **only** supported bridge between
    :func:`qd_train.eval_harness.split_conformal_threshold` and
    ``CalibrationEntry.conformal_quantile``. See this module's docstring for why writing
    ``conformal_quantile = split_conformal_threshold(...)`` is both the obvious line and a
    silent over-coverage bug.
    """
    if not (isinstance(q_hat, (int, float)) and math.isfinite(q_hat)):
        raise ValueError(f"nonconformity quantile must be a finite number, got {q_hat!r}")
    if not 0.0 <= q_hat <= 1.0:
        raise ValueError(f"nonconformity quantile must be in [0, 1], got {q_hat}")
    return 1.0 - float(q_hat)


def fit_noul_margin(
    margins: np.ndarray, correct: np.ndarray, *, target_precision: float = 0.95
) -> float:
    """Smallest margin at which answering reaches ``target_precision``.

    Abstention is only worth anything if the answers that survive it are better than the
    ones that do not. This sweeps the observed margins and returns the lowest threshold
    whose retained set is accurate enough; if no threshold reaches the target, it returns
    the largest observed margin, which abstains on everything but one — a refusal to
    pretend, not a silent fallback to 0.
    """
    m = np.asarray(margins, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    if m.ndim != 1 or len(m) == 0:
        raise ValueError(f"margins must be 1-D and non-empty, got shape {m.shape}")
    if len(m) != len(c):
        raise ValueError(f"margins/correct length mismatch: {len(m)} vs {len(c)}")
    if not np.isfinite(m).all():
        raise ValueError("margins contain a non-finite value")
    if not 0.0 < target_precision <= 1.0:
        raise ValueError(f"target_precision must be in (0, 1], got {target_precision}")

    for thr in np.unique(m):
        keep = m >= thr
        if keep.any() and c[keep].mean() >= target_precision:
            return float(thr)
    return float(m.max())


@dataclass(frozen=True, slots=True)
class CalibrationEntryFit:
    """One fitted entry, field-for-field with Rust's ``CalibrationEntry``.

    Field names and ranges match ``calibration.rs`` so that :meth:`to_json` deserialises
    there directly. ``CalibrationEntry::validate`` rejects a non-finite or non-positive
    temperature and anything outside ``[0, 1]`` for the other two; the same checks run here
    so a bad fit fails where it is produced rather than where it is loaded.
    """

    temperature: float
    conformal_quantile: float
    noul_margin: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0, got {self.temperature}")
        for name in ("conformal_quantile", "noul_margin"):
            v = getattr(self, name)
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be finite and in [0, 1], got {v}")

    def to_json(self) -> dict[str, float]:
        return {
            "temperature": self.temperature,
            "conformal_quantile": self.conformal_quantile,
            "noul_margin": self.noul_margin,
        }


def fit_entry(
    *,
    temperature: float,
    nonconformity_quantile: float,
    noul_margin: float,
) -> CalibrationEntryFit:
    """Build one entry, converting the conformal threshold to Rust's scale on the way in.

    Takes ``nonconformity_quantile`` rather than a ready-made ``conformal_quantile``
    precisely so the conversion cannot be skipped by a caller who has ``q̂`` in hand and
    assumes the names line up.
    """
    return CalibrationEntryFit(
        temperature=temperature,
        conformal_quantile=probability_cutoff_from_nonconformity(nonconformity_quantile),
        noul_margin=noul_margin,
    )


def fit_table(
    name: str,
    letters: dict[tuple[str, int], CalibrationEntryFit],
    span: CalibrationEntryFit | None = None,
) -> dict[str, object]:
    """Assemble the JSON ``CalibrationTable`` deserialises from.

    ``span`` is a single entry, not one per row count: a span slot's row count is the
    context's line count and cannot be enumerated (``calibration.rs``). Leaving it ``None``
    means a span request refuses with ``CalibrationEntryMissing``, which is the correct
    outcome for a table that was never fitted for spans — not a default.
    """
    if not name:
        raise ValueError("a calibration table must be named; the name is what a hash pins")
    out: dict[str, dict[str, float]] = {}
    for (kind, rows), entry in letters.items():
        out[letters_key(kind, rows)] = entry.to_json()
    return {
        "name": name,
        "letters": dict(sorted(out.items())),
        "span": None if span is None else span.to_json(),
    }
