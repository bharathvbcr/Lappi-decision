"""What a sweep could have seen, and the two ways of getting that wrong.

`agreement.py` makes this argument at n=50 for Cohen's kappa. These tests make it for a
seed sweep, using the numbers from the 2026-09-21 rung-0 capacity arms -- the run that
turned an unresolvable measurement into a reported null.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qd_train.power import (
    CONVENTIONAL_ALPHA,
    CONVENTIONAL_POWER,
    estimated_sd_penalty,
    resolution_state,
    resolvable_difference,
    t_quantile,
)
from qd_train.tristate import NotRun, Ran

REPO = Path(__file__).resolve().parents[2]


# -- the constant, which is computed rather than remembered --------------------------------


def test_the_leading_constant_is_the_two_quantiles_it_claims_to_be() -> None:
    """2.8 circulates as folklore. It is `z[0.975] + z[0.80]`, and computing it means alpha
    and power are real knobs rather than a number somebody wrote down once."""
    from statistics import NormalDist

    expected = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    assert math.isclose(expected, 2.801585, abs_tol=1e-5)

    # sd=1, n=2, two arms -> sqrt(2/2) = 1, so the difference IS the constant.
    got = resolvable_difference(
        sd=1.0, n_per_arm=2, against_known_reference=False,
        alpha=CONVENTIONAL_ALPHA, power=CONVENTIONAL_POWER,
    )
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_tightening_alpha_or_raising_power_costs_resolution() -> None:
    """Both directions, because a formula that ignored one of them would still pass a test
    that only moved the other."""
    base = resolvable_difference(sd=0.02, n_per_arm=8, against_known_reference=True)
    stricter = resolvable_difference(
        sd=0.02, n_per_arm=8, against_known_reference=True, alpha=0.01
    )
    stronger = resolvable_difference(
        sd=0.02, n_per_arm=8, against_known_reference=True, power=0.95
    )
    assert stricter > base
    assert stronger > base


def test_more_seeds_resolve_more_at_the_square_root_rate() -> None:
    """Quadrupling the seeds halves the resolvable difference. The rate is the reason a
    sweep cannot buy its way out of a bad sd with a couple more runs."""
    few = resolvable_difference(sd=0.02, n_per_arm=4, against_known_reference=True)
    many = resolvable_difference(sd=0.02, n_per_arm=16, against_known_reference=True)
    assert math.isclose(few / many, 2.0, rel_tol=1e-12)


# -- the comparison kind, which is the easy thing to get backwards -------------------------


def test_comparing_against_a_fixed_reference_resolves_more_than_against_another_arm() -> None:
    """A rung-0 arm against the majority-class baseline has one noisy side; against another
    arm it has two. The factor is sqrt(2), and the flattering answer is the wrong one."""
    fixed = resolvable_difference(sd=0.011, n_per_arm=8, against_known_reference=True)
    two_arm = resolvable_difference(sd=0.011, n_per_arm=8, against_known_reference=False)
    assert math.isclose(two_arm / fixed, math.sqrt(2.0), rel_tol=1e-12)


def test_the_comparison_kind_has_no_default() -> None:
    """Required keyword, because the two answers differ by 41% -- larger than several of the
    effects these sweeps look for -- and a default would be silently right half the time."""
    with pytest.raises(TypeError, match="against_known_reference"):
        resolvable_difference(sd=0.011, n_per_arm=8)  # type: ignore[call-arg]


# -- the 4096 arms, which are why this module exists ---------------------------------------


def test_the_4096_256x4_arm_is_inside_its_own_noise_floor() -> None:
    """The measurement this module was written for.

    Measured, not relayed: ``ledger/gh200-rung0-capacity-4096-e10-2026-09-21.jsonl`` on the
    GH200, arm ``rung0-scratch:256x8:4layer:ctx4096``, n=8, mean 0.4803, sd 0.0111, against
    a majority-class baseline of 0.484.

    So it came in 0.37pp BELOW the prior over 8 seeds. That is not a null result -- 8 seeds
    at that spread resolve 1.10pp against a fixed reference, so -0.37pp and 0.00pp are the
    same measurement. Read as "capacity does not help", it is a conclusion the run could
    not support.

    **And resolution is not even that arm's worst problem**, per the concurrent lane: it
    COLLAPSED on 6 of its 8 seeds, train accuracy at or below the 43.5% training majority.
    A held-out number from a model that never fit is not weak evidence about
    generalisation, it is no evidence, and the resolvable difference does not arise. Kept
    as this module's worked example because the arithmetic is what is being tested -- but
    a sweep needs BOTH checks, and this module is only the second one. Seeding
    ``--prior-sd`` from a collapsed arm's spread would be measuring the noise of a model
    that was not learning.
    """
    floor = resolvable_difference(sd=0.0111, n_per_arm=8, against_known_reference=True)
    assert 0.0100 < floor < 0.0120, floor
    assert abs(0.4803 - 0.484) < floor, "the observed difference must be inside the floor"


def test_the_4096_128x2_arm_by_contrast_is_resolvable() -> None:
    """The control, and the reason this is a measurement rather than a blanket objection to
    small sweeps.

    Same ledger, arm ``rung0-scratch:128x4:2layer:ctx4096``, n=8, mean 0.4544, sd 0.0212.
    That is -2.96pp against a 2.10pp floor, so it IS outside the noise. One arm of the
    sweep supports its claim and the other does not, which is exactly the distinction a row
    carrying only the point estimate cannot make.
    """
    floor = resolvable_difference(sd=0.0212, n_per_arm=8, against_known_reference=True)
    assert abs(0.4544 - 0.484) > floor


# -- the tri-state, which is the whole point -----------------------------------------------


def test_no_prior_sd_is_not_run_rather_than_silence() -> None:
    """A sweep that never had a prior spread has not been shown to be adequately powered,
    and recording nothing lets it read as though the question was asked and settled. This is
    the repository's rule about a check that could not run, one level up."""
    state = resolution_state(
        sd=None, n_per_arm=8, target=0.01, against_known_reference=True
    )
    assert isinstance(state, NotRun)
    assert "prior sd" in state.reason
    assert "not evidence of absence" in state.reason


def test_no_target_is_also_not_run() -> None:
    """A sweep that never said what it was looking for cannot be judged against what it
    could see, and the two missing values are named separately so the message says which."""
    state = resolution_state(
        sd=0.011, n_per_arm=8, target=None, against_known_reference=True
    )
    assert isinstance(state, NotRun)
    assert "target difference" in state.reason


def test_an_underpowered_sweep_fails_and_says_a_null_proves_nothing() -> None:
    """`passed=False` is not a refusal -- running an underpowered sweep can be the only box
    time there is. What is refused is reading its null as a result."""
    state = resolution_state(
        sd=0.0110, n_per_arm=8, target=0.005, against_known_reference=True
    )
    assert isinstance(state, Ran) and not state.passed
    assert "UNDERPOWERED" in state.detail
    assert "not evidence of absence" in state.detail


def test_an_adequate_sweep_still_says_the_bound_is_optimistic() -> None:
    """The bound assumes sd is KNOWN. At 3, 5 or 8 seeds it is estimated from those same
    runs, where the honest quantile is Student's t and about 14% wider at n=5. A sweep that
    clears the bound by less than that has not been shown to clear the real one, and the
    detail has to say so or the pass reads as stronger than it is."""
    state = resolution_state(
        sd=0.0110, n_per_arm=8, target=0.05, against_known_reference=True
    )
    assert isinstance(state, Ran) and state.passed
    assert "KNOWN-sd" in state.detail
    assert "Student's t" in state.detail


def test_the_state_carries_both_counts() -> None:
    """A sample size without a population reads as full coverage. Here the seed count is
    both, and it is stated rather than implied."""
    state = resolution_state(
        sd=0.011, n_per_arm=8, target=0.05, against_known_reference=True
    )
    assert isinstance(state, Ran)
    assert state.n == state.n_total == 8


# -- refusals -------------------------------------------------------------------------------


def test_a_single_run_per_arm_is_refused_rather_than_answered() -> None:
    """One run has no resolution at all. Returning a number a caller could print would make
    "n=1" look like a measurement with a wide error bar rather than no measurement."""
    with pytest.raises(ValueError, match="n_per_arm must be at least 1"):
        resolvable_difference(sd=0.01, n_per_arm=0, against_known_reference=True)


def test_a_negative_or_non_finite_sd_is_refused() -> None:
    """Both, because NaN propagates silently through the multiplication and would emerge as
    a resolvable difference of NaN, which compares False against every target and so reads
    as adequately powered."""
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sd must be finite and non-negative"):
            resolvable_difference(sd=bad, n_per_arm=8, against_known_reference=True)


def test_a_non_positive_target_is_refused() -> None:
    """A target of zero is satisfied by nothing and would report every sweep underpowered;
    a negative one is satisfied by everything."""
    for bad in (0.0, -0.01):
        with pytest.raises(ValueError, match="target must be finite and positive"):
            resolution_state(
                sd=0.011, n_per_arm=8, target=bad, against_known_reference=True
            )


# -- the sweep runner records it ------------------------------------------------------------


def test_every_sweep_runner_pre_registers_what_it_can_resolve() -> None:
    """The open item this closes: "pre-register the resolvable difference in every sweep
    runner's header, beside the seed count". Asserted on the source because reaching the
    metric needs a corpus and a GPU.

    **Every**, plural, and deliberately. Four times on 2026-09-21 a fix was applied to one
    member of a list that had already been written down -- the price literal, the provenance
    fingerprint, the call-site scope, and this. Writing the test for one runner and calling
    the item closed is the same move a fifth time.

    The comparison kind is asserted by POSITION, inside each `resolution_state(` call. Both
    tools score against a fixed reference -- rung 0 against the majority-class baseline, the
    FT lane against a floor computed from the corpus -- and neither is a quantity with seed
    noise. Passing the two-arm form would claim 1.41x the sensitivity these comparisons have.
    """
    def call_text(source: str, opener: str) -> str:
        """The whole call, by balancing parentheses.

        Slicing to the first ``),`` cuts ``n_per_arm=len(args.seeds)`` in half and reports
        the argument as absent -- a test failing on its own parsing rather than on the
        thing it checks.
        """
        start = source.index(opener)
        depth = 0
        for i in range(start + len(opener) - 1, len(source)):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    return source[start : i + 1]
        raise AssertionError(f"unbalanced parentheses after {opener}")

    runners = {
        "rung0_real_run.py": "n_per_arm=args.seeds",
        "real_ft_run.py": "n_per_arm=len(args.seeds)",
    }
    for name, seed_expression in runners.items():
        source = (REPO / "tools" / name).read_text(encoding="utf-8")
        assert '"--prior-sd"' in source, name
        assert '"--target-difference"' in source, name
        assert '"sweep_can_resolve"' in source, name

        call = call_text(source, "resolution_state(")
        assert seed_expression in call, (
            f"{name}: the resolution must be computed from the seed count the run actually "
            "used, and --seeds is a scalar in one tool and a list in the other"
        )
        assert "against_known_reference=True" in call, (
            f"{name}: this tool compares against a fixed reference, not another arm"
        )


# --------------------------------------------------------------------------
# The correction this module used to quote as a constant.
#
# `resolution_state` printed "~14% at n=5" on every row it wrote, regardless of n and
# regardless of whether the comparison was against a fixed reference or another arm. It
# reached a real audit that way: a concurrent lane read it off a one-sample row at n=8,
# tried to reproduce it, could not, and declined to put any figure derived from it into an
# AUDIT file. Which was the right call, and is what found this.
#
# The figure is correct for exactly one case. Everything below is about making the row
# state ITS case.
# --------------------------------------------------------------------------

#: t[0.975, df], the standard two-sided 95% table. Written down rather than computed,
#: because the point is to check the computation against something it did not produce.
TWO_SIDED_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
}


@pytest.mark.parametrize("df", sorted(TWO_SIDED_95))
def test_the_inverse_t_reproduces_the_standard_table(df: int) -> None:
    """Sixteen degrees of freedom from 1 to 120, against the table every textbook prints.

    This is the load-bearing check of the whole section: an incomplete-beta continued
    fraction that is subtly wrong still returns plausible numbers, and every figure this
    module now puts on a row comes out of it. The table's own precision is three decimals,
    so that is the tolerance.
    """
    assert t_quantile(0.975, df) == pytest.approx(TWO_SIDED_95[df], abs=5e-4)


def test_the_t_quantile_agrees_with_the_normal_in_the_limit() -> None:
    """At large df, Student's t IS the normal, and a continued fraction that drifts there
    is one that is wrong everywhere and only visible here."""
    from statistics import NormalDist

    for p in (0.80, 0.95, 0.975, 0.995):
        assert t_quantile(p, 100_000) == pytest.approx(NormalDist().inv_cdf(p), abs=1e-3)


def test_a_quantile_outside_the_supported_range_is_refused() -> None:
    """Refused rather than extrapolated. The bisection brackets upward from zero, so p
    below 0.5 would return 0.0 -- a plausible-looking number that is not a quantile."""
    for bad_p in (0.0, 0.25, 0.4999, 1.0, 1.5):
        with pytest.raises(ValueError, match="t_quantile is for p"):
            t_quantile(bad_p, 8)
    for bad_df in (0, -1, -0.5):
        with pytest.raises(ValueError, match="degrees of freedom"):
            t_quantile(0.975, bad_df)


def test_the_penalty_reproduces_the_figure_this_module_used_to_hardcode() -> None:
    """Where "~14% at n=5" came from, established rather than assumed.

    It is the TWO-sample case at 5 per arm: df = 2(n-1) = 8, and t on both terms,
    ``t[0.975,8] + t[0.80,8]`` = 3.1949 against the normal 2.8016. That identifies the
    convention the old constant was computed under, which is what makes replacing it a
    correction of scope rather than a change of definition.
    """
    assert estimated_sd_penalty(
        n_per_arm=5, against_known_reference=False
    ) == pytest.approx(3.1949 / 2.8016, rel=2e-4)


def test_the_penalty_is_larger_for_the_case_the_constant_was_printed_on() -> None:
    """The defect, as a number.

    A one-sample comparison at n=8 -- a rung-0 arm against the majority-class baseline,
    which is most of what this repository measures -- carries a 16.4% correction, not 14%.
    The row said 14%. Under-stating the correction on the side that says "your margin is
    smaller than you think" is the direction that matters.
    """
    one_sample_n8 = estimated_sd_penalty(n_per_arm=8, against_known_reference=True)
    old_constant = estimated_sd_penalty(n_per_arm=5, against_known_reference=False)
    assert one_sample_n8 == pytest.approx(1.164, abs=1e-3)
    assert one_sample_n8 > old_constant


def test_the_penalty_falls_with_n_and_with_the_second_sample() -> None:
    """Both directions, because a correction that moved the wrong way would still look
    like a correction. More runs estimate sd better; a two-sample comparison has twice the
    degrees of freedom at the same n, so it pays less for the same ignorance."""
    for known in (True, False):
        penalties = [
            estimated_sd_penalty(n_per_arm=n, against_known_reference=known)
            for n in (3, 5, 8, 20)
        ]
        assert penalties == sorted(penalties, reverse=True), penalties
        assert penalties[-1] > 1.0, "the correction never reaches zero at finite n"
    for n in (3, 5, 8, 20):
        assert estimated_sd_penalty(n_per_arm=n, against_known_reference=True) > \
            estimated_sd_penalty(n_per_arm=n, against_known_reference=False)


def test_a_single_run_cannot_estimate_the_spread_it_is_being_corrected_for() -> None:
    """n=1 leaves zero degrees of freedom. Returning a correction there would put a number
    on a row whose sd came from nowhere."""
    with pytest.raises(ValueError, match="no degrees of freedom"):
        estimated_sd_penalty(n_per_arm=1, against_known_reference=True)


def test_the_row_states_its_own_correction_not_a_constant() -> None:
    """The fix, end to end, on the shape that reached the audit.

    Two states differing only in ``n_per_arm`` must not carry the same percentage, and
    neither may carry the old hardcoded one. Asserted on the detail string because that is
    what a reader reads -- the number being right in a function nobody prints is not the
    property that failed.
    """
    detail_n8 = resolution_state(
        sd=0.0289, n_per_arm=8, target=0.05, against_known_reference=True
    ).detail
    detail_n5 = resolution_state(
        sd=0.0289, n_per_arm=5, target=0.05, against_known_reference=True
    ).detail

    assert "16.4% too small" in detail_n8, detail_n8
    assert "32.7% too small" in detail_n5, detail_n5
    assert "7 degrees of freedom" in detail_n8
    assert "4 degrees of freedom" in detail_n5
    for detail in (detail_n8, detail_n5):
        assert "~14% at" not in detail, (
            "the row still quotes the constant that is true of one case and was printed on "
            "all of them"
        )


def test_both_verdicts_carry_the_correction() -> None:
    """Adequate and underpowered are different sentences and the correction belongs in
    both. A bound that is 16% too small matters most to the row that just cleared it."""
    adequate = resolution_state(
        sd=0.01, n_per_arm=8, target=0.05, against_known_reference=True
    )
    underpowered = resolution_state(
        sd=0.05, n_per_arm=8, target=0.01, against_known_reference=True
    )
    assert adequate.passed and not underpowered.passed
    for state in (adequate, underpowered):
        assert "ESTIMATED from these 8 runs" in state.detail
        assert "16.4% too small" in state.detail


def test_the_corrected_bound_is_stated_as_a_number_not_only_a_percentage() -> None:
    """A reader comparing a measured margin against a floor needs the floor, not a
    percentage to apply to it themselves -- which is one more step at which the case can
    be got wrong."""
    state = resolution_state(
        sd=0.0289, n_per_arm=8, target=0.05, against_known_reference=True
    )
    assert isinstance(state, Ran) and state.value is not None
    honest = state.value * estimated_sd_penalty(
        n_per_arm=8, against_known_reference=True
    )
    assert f"{honest:.4f}, not {state.value:.4f}" in state.detail
