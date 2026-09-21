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
z. That constant is larger: about 3.20 at n=5 per arm against 2.80, so the true resolvable
difference is roughly 14% wider there than this returns.

So every number here is a **lower bound on what it takes to detect an effect** -- optimistic
by construction. It is reported that way rather than corrected, because correcting it
properly needs an inverse-t and SciPy is not a dependency of this project (`pyproject.toml`
declares numpy, datasketch and pyyaml). A sweep that is underpowered against this bound is
underpowered, full stop; a sweep that clears it by a little has not been shown to clear the
real one. Both halves of that are in :func:`resolution_state`'s detail.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from .tristate import NotRun, Ran, TriState

__all__ = ["CONVENTIONAL_ALPHA", "CONVENTIONAL_POWER", "resolution_state",
           "resolvable_difference"]

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
    if adequate:
        detail = (
            f"{verdict} -- adequate, but against the KNOWN-sd bound: with sd estimated from "
            "this many runs the honest quantile is Student's t, which is wider (~14% at "
            "n=5), so a margin narrower than that has not been demonstrated"
        )
    else:
        detail = (
            f"{verdict} -- UNDERPOWERED: this sweep cannot see its own target effect, so a "
            "null from it is not evidence of absence. The bound is optimistic on top of "
            "that, because it assumes sd is known rather than estimated from these runs"
        )
    return Ran(
        passed=adequate,
        value=difference,
        n=n_per_arm,
        n_total=n_per_arm,
        detail=detail,
    )
