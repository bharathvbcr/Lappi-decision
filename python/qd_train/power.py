"""What difference a sweep could have seen, stated before it runs.

`agreement.py` makes this argument for Cohen's kappa at n=50: *"Reporting 0.62 as a pass,
with no interval, is how an underpowered gate becomes a confident decision."* The same
sentence is true of a seed sweep, and on 2026-09-21 it cost this project a capacity
experiment.

The 4096 arms compared model capacities at 8 seeds each and reported both below the
majority-class prior. Read as a capacity result, that is "capacity does not help". But
nothing in the run said what difference 8 seeds at that per-seed spread could resolve, so
"no effect" and "an effect this sweep could never have seen" arrive in the ledger looking
identical -- which is the repository's own rule about a check that could not run, one level
up from a check to a comparison.

## What this computes

The smallest true difference a two-sided test at ``alpha`` would detect with probability
``power``::

    difference = (z[1 - alpha/2] + z[power]) * sd * sqrt(k / n)

The leading constant is ~2.80 at the conventional 0.05/0.80, and it is computed from
:class:`statistics.NormalDist` rather than written down, so the assumption is visible and
the two knobs are real parameters instead of a folklore number.

``k`` is 2 when both sides are measured at ``n`` runs each, and 1 when the other side is a
fixed number. **Both cases occur in this repository and they differ by a factor of 1.41**,
which is larger than several of the effects these sweeps look for. A rung-0 arm compared
against the majority-class baseline is the one-sided-reference case: the baseline is a
property of the validation split, not a quantity with seed noise. An arm compared against
another arm is the two-sample case. Getting this backwards overstates sensitivity by 41%,
so ``against_known_reference`` is a required keyword with no default -- a caller has to say
which comparison it is making.

## The one thing this number is not

It assumes ``sd`` is **known**. At the seed counts these sweeps actually use -- 3, 5, 8 --
it is estimated from the same handful of runs, and the honest quantile is Student's t, not
z. So every number :func:`resolvable_difference` returns is a **lower bound on what it takes
to detect an effect**, optimistic by construction.

:func:`estimated_sd_penalty` says by how much, for the caller's own case, and
:func:`resolution_state` puts that figure on the row. Until 2026-09-21 it printed one
hardcoded "~14% at n=5" on every row instead. That figure is right for exactly one case --
two-sample at 5 per arm, where ``t[0.975, 8] + t[0.80, 8]`` is 3.1949 against the normal
2.8016 -- and it was printed unchanged on a one-sample row at n=8, where the honest figure
is 16.4%. Larger, not smaller, in a detail string a reader is meant to act on. Across the
seed counts in use the correction runs from 32.7% (one-sample, n=5) down to 7.5%
(two-sample, n=8 per arm): no single number covers that, and the one that was printed was
the smallest of them.

It was found by a concurrent lane refusing to put a figure derived from it into an audit
file, and reproduced from both ends before being changed.

Correcting it needs an inverse-t, which this module now computes from ``math`` alone -- the
regularised incomplete beta by Lentz's continued fraction, pinned in ``test_power.py``
against the standard two-sided 95% table from df=1 to df=120. SciPy is not a dependency of
this project (`pyproject.toml` declares numpy, datasketch and pyyaml) and does not become
one. A sweep that is underpowered against this bound is
underpowered, full stop; a sweep that clears it by a little has not been shown to clear the
real one. Both halves of that are in :func:`resolution_state`'s detail.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from .tristate import NotRun, Ran, TriState

__all__ = ["CONVENTIONAL_ALPHA", "CONVENTIONAL_POWER", "estimated_sd_penalty",
           "resolution_state", "resolvable_difference", "t_quantile"]

#: Two-sided significance and the power conventionally paired with it. Named rather than
#: defaulted-in-place so that a sweep quoting different ones has to say so.
CONVENTIONAL_ALPHA = 0.05
CONVENTIONAL_POWER = 0.80

_NORMAL = NormalDist()


def _constant(alpha: float, power: float) -> float:
    """``z[1 - alpha/2] + z[power]`` -- ~2.8016 at 0.05/0.80."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    return _NORMAL.inv_cdf(1.0 - alpha / 2.0) + _NORMAL.inv_cdf(power)


# ---------------------------------------------------------------------------------------
# How optimistic the bound above is, for the row's OWN case.
#
# The module reported a single hardcoded "~14% at n=5" on every resolution row. That number
# is correct and it is correct for ONE case: a two-sample comparison at 5 per arm, where
# `t[0.975, 8] + t[0.80, 8]` is 3.1949 against the normal 2.8016. It was printed unchanged
# on a one-sample row at n=8, where the true figure is 16.4% -- larger, not smaller, and
# quoted in a detail string a reader is meant to act on.
#
# The fix needs t at arbitrary df, and SciPy is not a dependency of this project. It does
# not have to be: the regularised incomplete beta by Lentz's continued fraction is forty
# lines of `math`, and `test_power.py` pins it against the standard table before anything
# uses it. Recomputed per row rather than tabulated, because the table would be one more
# place for a case to be quoted that is not the reader's.
# ---------------------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta, Lentz's method.

    Raises rather than returning its last iterate: a quantile that did not converge is not
    a quantile, and this feeds a number a reader uses to decide whether a result stands.
    """
    tiny, eps, max_iter = 1e-300, 3e-16, 500
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            return h
    raise ArithmeticError(
        f"incomplete beta did not converge in {max_iter} iterations at a={a}, b={b}, x={x}"
    )


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front + a * math.log(x) + b * math.log1p(-x)) * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        log_front + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """``t[p, df]`` for ``p >= 0.5``, by bisection on the CDF above.

    Bounded: the bracket doubles at most until 1e6 and the bisection runs a fixed 200
    times, so no input makes this spin. `test_power.py` checks it against the standard
    two-sided 95% table from df=1 to df=120.
    """
    if not 0.5 <= p < 1.0:
        raise ValueError(f"t_quantile is for p in [0.5, 1), got {p}")
    if df <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {df}")
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e6:  # pragma: no cover - unreachable for p < 1 at any positive df
            raise ArithmeticError(f"t quantile did not bracket at p={p}, df={df}")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def estimated_sd_penalty(
    *,
    n_per_arm: int,
    against_known_reference: bool,
    alpha: float = CONVENTIONAL_ALPHA,
    power: float = CONVENTIONAL_POWER,
) -> float:
    """How much wider the honest resolvable difference is than what this module returns.

    :func:`resolvable_difference` assumes ``sd`` is known. It never is here -- it is
    estimated from the same handful of runs -- so the honest quantiles are Student's t on
    ``df``, and the ratio of the two constants is the factor by which every number this
    module reports is too small.

    ``df`` is ``n - 1`` against a fixed reference and ``2(n - 1)`` against another arm,
    matching the one- and two-sample cases :func:`resolvable_difference` already
    distinguishes. Both terms take t, which is the convention that reproduces this module's
    own previously-hardcoded figure: 3.1949 against 2.8016 at two-sample n=5 per arm, the
    14% the docstring quoted.

    Returns a multiplier >= 1. At the seed counts these sweeps use it ranges from 32.7%
    (one-sample, n=5) to 7.5% (two-sample, n=8 per arm) -- which is the point: one number
    could not have covered that spread, and the one that was printed was the smallest of
    them.
    """
    if n_per_arm < 2:
        raise ValueError(
            f"n_per_arm={n_per_arm} leaves no degrees of freedom to estimate sd from; "
            "this correction is about sd being estimated, and one run estimates nothing"
        )
    df = float(n_per_arm - 1 if against_known_reference else 2 * (n_per_arm - 1))
    student = t_quantile(1.0 - alpha / 2.0, df) + t_quantile(power, df)
    return student / _constant(alpha, power)


def resolvable_difference(
    *,
    sd: float,
    n_per_arm: int,
    against_known_reference: bool,
    alpha: float = CONVENTIONAL_ALPHA,
    power: float = CONVENTIONAL_POWER,
) -> float:
    """The smallest difference this many runs could detect.

    ``sd`` is the per-run spread at a fixed configuration -- the seed sd -- and has to come
    from somewhere other than the sweep being planned, or the sweep is grading its own
    sensitivity with the numbers it is about to interpret.

    ``against_known_reference`` says what the arm is being compared to: ``True`` for a fixed
    number such as the majority-class baseline, ``False`` for another arm measured at the
    same ``n``. It has no default because the two answers differ by 1.41x and the wrong one
    is always the flattering one.

    A single run per arm has no resolution at all rather than infinite resolution, so
    ``n_per_arm`` below 1 is refused rather than returning something a caller might print.
    """
    if not math.isfinite(sd) or sd < 0.0:
        raise ValueError(f"sd must be finite and non-negative, got {sd}")
    if n_per_arm < 1:
        raise ValueError(f"n_per_arm must be at least 1, got {n_per_arm}")
    variances = 1.0 if against_known_reference else 2.0
    return _constant(alpha, power) * sd * math.sqrt(variances / n_per_arm)


def resolution_state(
    *,
    sd: float | None,
    n_per_arm: int,
    target: float | None,
    against_known_reference: bool,
    alpha: float = CONVENTIONAL_ALPHA,
    power: float = CONVENTIONAL_POWER,
) -> TriState:
    """Can this sweep see the effect it is looking for? Pre-registered, before it runs.

    Three outcomes, and the third is the one this exists for:

    - ``sd`` or ``target`` absent -> :class:`NotRun`. A sweep that never stated what it was
      looking for, or never had a prior spread to judge itself against, has not been shown
      to be adequately powered -- and must not record that fact the same way a sweep that
      checked and passed does.
    - resolvable difference at or below ``target`` -> ``Ran(passed=True)``.
    - resolvable difference above ``target`` -> ``Ran(passed=False)``, and the detail says
      plainly that a null from this sweep is not evidence of absence.

    ``passed=False`` is deliberately not a refusal. Running an underpowered sweep can be the
    right call -- it may be all the box time there is. What is not acceptable is reading its
    null as a result, and that is a property of the row, not of the run.
    """
    if sd is None or target is None:
        missing = " and ".join(
            name for name, value in (("prior sd", sd), ("target difference", target))
            if value is None
        )
        return NotRun(
            f"no {missing} supplied, so what this sweep could resolve at n={n_per_arm} is "
            "unknown. A null from it is not evidence of absence, and recording nothing here "
            "would let it read as though the question had been asked and settled."
        )
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(f"target must be finite and positive, got {target}")

    difference = resolvable_difference(
        sd=sd,
        n_per_arm=n_per_arm,
        against_known_reference=against_known_reference,
        alpha=alpha,
        power=power,
    )
    adequate = difference <= target
    against = "a fixed reference" if against_known_reference else "another arm"
    verdict = (
        f"resolves {difference:.4f} against {against} at n={n_per_arm} per arm, "
        f"sd={sd:.4f}, alpha={alpha}, power={power}; target {target:.4f}"
    )
    # This row's own correction, not a constant. The figure here used to read "~14% at
    # n=5" on every row regardless of n or of which comparison was being made -- true of a
    # two-sample comparison at 5 per arm and of nothing else, and printed unchanged on a
    # one-sample row at n=8 where the honest figure is 16.4%. A number a reader is meant
    # to act on has to be the number for their case.
    penalty = estimated_sd_penalty(
        n_per_arm=n_per_arm,
        against_known_reference=against_known_reference,
        alpha=alpha,
        power=power,
    )
    honest = difference * penalty
    correction = (
        f"sd is ESTIMATED from these {n_per_arm} runs, not known, so the honest quantile is "
        f"Student's t on {n_per_arm - 1 if against_known_reference else 2 * (n_per_arm - 1)} "
        f"degrees of freedom and this bound is {100 * (penalty - 1):.1f}% too small: "
        f"{honest:.4f}, not {difference:.4f}"
    )
    if adequate:
        detail = (
            f"{verdict} -- adequate against the KNOWN-sd bound, and that bound is the "
            f"optimistic one: {correction}. A margin narrower than that has not been "
            "demonstrated"
        )
    else:
        detail = (
            f"{verdict} -- UNDERPOWERED: this sweep cannot see its own target effect, so a "
            f"null from it is not evidence of absence. It is worse than it looks: "
            f"{correction}"
        )
    return Ran(
        passed=adequate,
        value=difference,
        n=n_per_arm,
        n_total=n_per_arm,
        detail=detail,
    )
