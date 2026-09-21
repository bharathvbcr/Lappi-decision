"""``tools/ledger_arms.py``: the three things a log parser got wrong.

Each test here fails against a specific wrong implementation, and each of those wrong
implementations is one somebody wrote:

* grouping on ``recipe_hash`` alone merges a capacity sweep's arms into one population --
  the three 4096 arms share recipe ``aa8aeac9b4`` and differ only in ``backbone_commit``;
* reading the accuracy without the collapse gate reports a constant predictor's score as a
  result, which is how "capacity helps +2.59pp" was reported from the e10 arms on
  2026-09-21;
* using the one-sample floor on an arm-vs-arm difference claims 1.41x more sensitivity than
  the comparison has, which flipped a verdict the same day.

The last test reads the committed GH200 ledger rather than a fixture, so the row shape this
module parses stays pinned to the row shape the runners actually write.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ledger_arms  # noqa: E402

#: The sweep whose three arms share one recipe. Committed at 7f191bd, off the rented box.
CAPACITY_E30 = REPO / "ledger" / "gh200-rung0-capacity-4096-e30-2026-09-21.jsonl"


def _row(
    *,
    backbone: str,
    val: float,
    fit: bool = True,
    recipe: str = "r" * 64,
    baseline: float = 0.484,
    metrics: bool = True,
) -> dict:
    """One row carrying only the fields ``ledger_arms`` reads."""
    row: dict = {"protocol": {"recipe_hash": recipe, "backbone_commit": backbone}, "metrics": {}}
    if metrics:
        row["metrics"] = {
            ledger_arms.VAL: {
                "state": "ran",
                "passed": val > baseline,
                "value": val,
                "detail": (
                    f"choice top-1 {val * 100:.1f}% of 288 held-out rows, against a "
                    f"majority-class baseline of {baseline * 100:.1f}%."
                ),
            },
            ledger_arms.TRAIN: {
                "state": "ran",
                "passed": fit,
                "value": 0.65 if fit else 0.40,
                "detail": "against a training-set majority share of 46.9%.",
            },
        }
    return row


def test_two_backbones_at_one_recipe_are_two_arms() -> None:
    """The grouping bug. A capacity sweep's arms differ in the model, not in the recipe.

    Against a version that keys on ``recipe_hash`` alone this returns one arm of four
    seeds whose spread is mostly the capacity difference it was supposed to measure.
    """
    rows = [
        _row(backbone="rung0-scratch:128x4:2layer:ctx4096", val=0.38),
        _row(backbone="rung0-scratch:128x4:2layer:ctx4096", val=0.39),
        _row(backbone="rung0-scratch:512x8:6layer:ctx4096", val=0.40),
        _row(backbone="rung0-scratch:512x8:6layer:ctx4096", val=0.41),
    ]
    arms = ledger_arms.arms_of(rows)
    assert len(arms) == 2, f"grouped into {len(arms)} arm(s); the backbone is half the key"
    assert {len(a.values) for a in arms.values()} == {2}


def test_one_backbone_at_two_recipes_is_also_two_arms() -> None:
    """The other half of the same key, which a learning curve depends on.

    Its points run one model over different training-set sizes, so they share a backbone
    and differ only in ``recipe_hash`` -- via ``train_subsample``, which ``0e897b3`` put
    into the recipe for exactly this reason.
    """
    rows = [
        _row(backbone="rung0-scratch:128x4:2layer:ctx8192", val=0.52, recipe="a" * 64),
        _row(backbone="rung0-scratch:128x4:2layer:ctx8192", val=0.54, recipe="b" * 64),
    ]
    assert len(ledger_arms.arms_of(rows)) == 2


def test_a_seed_that_did_not_clear_the_training_majority_is_not_in_the_fitted_sample() -> None:
    """Collapse comes from the gate the runner already recorded.

    A collapsed seed's held-out number is what a constant predictor scores, so including
    it in the sample being interpreted is how a majority-class predictor gets reported as
    a result.
    """
    rows = [
        _row(backbone="b", val=0.484, fit=False),
        _row(backbone="b", val=0.483, fit=False),
        _row(backbone="b", val=0.42, fit=True),
        _row(backbone="b", val=0.44, fit=True),
    ]
    arm = next(iter(ledger_arms.arms_of(rows).values()))
    assert arm.collapsed == 2
    assert len(arm.values) == 4, "every measured seed belongs to the all-seeds sample"
    assert sorted(arm.fitted) == [0.42, 0.44]


def test_a_collapsed_arms_tight_spread_is_reported_beside_the_fitted_one_not_instead() -> None:
    """Both numbers, never one standing in for the other.

    The e10 arms gave 1.11pp and 0.67pp against 4.41-5.58pp for the fitted e30 arms. That
    tightness is a constant predictor having almost nothing to vary; a report that showed
    only the all-seeds spread would hand the next sweep a prior three times too confident.
    """
    rows = [_row(backbone="b", val=v, fit=False) for v in (0.484, 0.485, 0.483, 0.484)]
    rows += [_row(backbone="b", val=v, fit=True) for v in (0.40, 0.46)]
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of(rows)))
    assert "all seeds: n=6" in text
    assert "fitted seeds: n=2" in text
    assert "COLLAPSED on 4 of 6" in text


def test_a_row_missing_its_metrics_is_unmeasured_not_a_zero() -> None:
    """A seed that recorded nothing and a seed that scored nothing are different facts."""
    rows = [_row(backbone="b", val=0.5), _row(backbone="b", val=0.0, metrics=False)]
    arm = next(iter(ledger_arms.arms_of(rows).values()))
    assert arm.unmeasured == 1
    assert arm.values == [0.5], "an unmeasured row must not enter the sample as a number"
    assert arm.n_rows == 2, "and must not vanish from the row count either"


def test_an_arm_with_one_fitted_seed_reports_no_spread_rather_than_a_zero_one() -> None:
    """One run per arm has no resolution at all, which is not the same as perfect."""
    rows = [_row(backbone="b", val=0.45, fit=True)]
    rows += [_row(backbone="b", val=v, fit=False) for v in (0.484, 0.485)]
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of(rows)))
    assert "fitted seeds: n=1 -- no spread to report" in text


def test_a_baseline_no_row_states_is_said_rather_than_silently_dropped() -> None:
    """A delta that cannot be computed is named; the arm is still reported."""
    row = _row(backbone="b", val=0.5)
    row["metrics"][ledger_arms.VAL]["detail"] = "no baseline in this sentence"
    text = "\n".join(ledger_arms.render(ledger_arms.arms_of([row, _row_no_base()])))
    assert "NOT STATED by any row" in text


def _row_no_base() -> dict:
    row = _row(backbone="b", val=0.51)
    row["metrics"][ledger_arms.VAL]["detail"] = "also no baseline here"
    return row


def test_a_pairwise_line_names_both_arms_distinguishably() -> None:
    """A learning curve's points share a backbone, so the backbone cannot be the label.

    Against the first version of this file the line reads
    ``rung0-scratch:128x4:2layer:ctx8192 minus rung0-scratch:128x4:2layer:ctx8192`` -- two
    different arms with the same name, on the report that compares them. Found by running
    the tool on the curve's own partial rows before its numbers mattered, which is the only
    reason it is fixed rather than shipped.
    """
    backbone = "rung0-scratch:128x4:2layer:ctx8192"
    quarter = [_row(backbone=backbone, val=v, recipe="1" * 64) for v in (0.49, 0.50, 0.48)]
    whole = [_row(backbone=backbone, val=v, recipe="2" * 64) for v in (0.52, 0.53, 0.51)]
    line = next(
        ln for ln in ledger_arms.render_pairwise(ledger_arms.arms_of(quarter + whole))
        if "minus" in ln
    )
    left, right = line.split("minus", 1)
    assert "1111111111" in right and "2222222222" in left, (
        f"the two sides of the comparison are not distinguishable:\n{line}"
    )


def test_the_arm_vs_arm_floor_is_the_wider_one() -> None:
    """The sqrt(2). Two noisy arms, not an arm against a fixed number.

    Against a version that passes ``against_known_reference=True`` for a difference
    between arms, the printed floor is 1.41x too small -- and 1.41x too small is always
    the direction that turns a null into a finding.
    """
    left = [_row(backbone="aaa", val=v) for v in (0.30, 0.34, 0.32, 0.36)]
    right = [_row(backbone="bbb", val=v) for v in (0.40, 0.44, 0.42, 0.46)]
    text = "\n".join(ledger_arms.render_pairwise(ledger_arms.arms_of(left + right)))
    # The known-sd column of both lines, so this compares like with like: the sqrt(2) is a
    # property of the comparison, and the t penalty is a separate factor that differs
    # between the two cases (df = n-1 against df = 2(n-1)) and would otherwise be folded in.
    pairwise = float(
        text.split("known-sd / estimated-sd ", 1)[1].split(" / ", 1)[0]
    )
    one_arm = "\n".join(ledger_arms.render(ledger_arms.arms_of(left)))
    fixed = float(
        one_arm.split("known-sd / estimated-sd: ", 1)[1].split(" / ", 1)[0]
    )
    # Both numbers are read back off a line rounded to two decimals, so the tolerance is
    # the printing precision and not the arithmetic's. It is still an order of magnitude
    # tighter than the gap being tested: against the one-sample floor this reads 3.62
    # against an expected 5.12, which no rounding accounts for.
    assert pairwise == pytest.approx(fixed * 2**0.5, abs=0.02)


def test_both_floors_are_printed_and_the_honest_one_is_the_wider() -> None:
    """The known-sd bound and the estimated-sd bound, as two numbers on the line.

    ``resolvable_difference`` assumes ``sd`` is known; it is estimated from these runs, so
    the honest quantile is Student's t. A reader given one number and a percentage has to
    pick the case themselves, and on 2026-09-21 an AUDIT file picked the wrong one -- it
    applied this module's two-sample n=5 figure of 14% to a one-sample n=8 comparison whose
    true penalty is 16.4%, erring on the flattering side.
    """
    rows = [_row(backbone="b", val=v) for v in (0.50, 0.52, 0.54, 0.51, 0.53)]
    line = next(
        ln for ln in ledger_arms.render(ledger_arms.arms_of(rows)) if "known-sd" in ln
    )
    fixed = line.split("known-sd / estimated-sd: ", 1)[1]
    known, honest = (float(x) for x in fixed.split("pp", 1)[0].split(" / "))
    assert honest > known, f"the estimated-sd bound must be the wider one:\n{line}"
    assert honest == pytest.approx(
        known * ledger_arms.estimated_sd_penalty(n_per_arm=5, against_known_reference=True),
        abs=0.01,
    ), "the penalty is not the one qd_train.power computes for this case"


def _arm_at(backbone: str, *, mean: float, sd: float, n: int = 5) -> list[dict]:
    """``n`` seeds with exactly this mean and sample sd, so a fixture can be aimed."""
    step = sd / statistics.stdev([i - (n - 1) / 2 for i in range(n)])
    return [_row(backbone=backbone, val=mean + step * (i - (n - 1) / 2)) for i in range(n)]


def test_a_difference_between_the_two_bounds_is_not_called_visible() -> None:
    """The band where a null gets read as a finding.

    A difference that clears the known-sd floor but not the honest one has not been shown
    to be visible, and calling it VISIBLE is the flattering read. It is the exact error the
    curve's audit made before the correction: +3.78pp reported as clearing 2.86pp, when the
    bound that applies is 3.33pp.

    The fixture is AIMED at the band rather than hoped into it. At n=5 per arm the floor is
    ``2.80159 * s * sqrt(2/5)`` and the honest one is 14.0% wider, so a pooled sd of 1.00pp
    puts the band at roughly [1.77, 2.02]pp and a 1.90pp difference lands inside it. If the
    arithmetic ever moves, this fails rather than skipping -- a test that skips is a test
    that did not run, which is the thing this repository refuses.
    """
    left = _arm_at("aaa", mean=0.3000, sd=0.0100)
    right = _arm_at("bbb", mean=0.3190, sd=0.0100)
    text = "\n".join(ledger_arms.render_pairwise(ledger_arms.arms_of(left + right)))

    known, honest = (
        float(x)
        for x in text.split("known-sd / estimated-sd ", 1)[1].split("pp", 1)[0].split(" / ")
    )
    diff = abs(statistics.mean([r["metrics"][ledger_arms.VAL]["value"] for r in right])
               - statistics.mean([r["metrics"][ledger_arms.VAL]["value"] for r in left])) * 100
    assert known < diff <= honest, (
        f"the fixture no longer lands in the band: {diff:.2f}pp against [{known}, {honest}]"
    )
    assert "clears the known-sd bound only" in text, text
    assert "VISIBLE" not in text.replace("clears the known-sd bound only", ""), text


def test_a_ledger_that_is_not_there_is_refused_rather_than_reported_over_the_rest() -> None:
    """Fail closed. A report over the files that happened to exist is a coverage claim
    nobody can reconstruct."""
    with pytest.raises(FileNotFoundError, match="coverage claim"):
        ledger_arms.read_rows([CAPACITY_E30, REPO / "ledger" / "does-not-exist.jsonl"])


def test_the_committed_capacity_ledger_reads_as_three_arms_of_eight() -> None:
    """Against the real rows, so the fixture above cannot drift from the written shape.

    These 24 rows are the 4096 e30 sweep: three capacities, 8 seeds each, no seed
    collapsed -- every one trained above its training-set majority, which is what makes
    all 24 held-out numbers statements about generalisation.
    """
    arms = ledger_arms.arms_of(ledger_arms.read_rows([CAPACITY_E30]))
    assert len(arms) == 3, f"expected three arms, got {sorted(k[1] for k in arms)}"
    assert {a.recipe_hash for a in arms.values()} == {
        next(iter(arms.values())).recipe_hash
    }, "the three arms share one recipe; they are separated by the backbone"
    for arm in arms.values():
        assert len(arm.values) == 8, f"{arm.backbone_commit} has {len(arm.values)} measured"
        assert arm.unmeasured == 0
        assert arm.collapsed == 0, f"{arm.backbone_commit} collapsed on {arm.collapsed}"
        assert arm.baseline == pytest.approx(0.484, abs=5e-4)


# -- render_gates: a gate nobody evaluated and a gate that passed must not look alike ----


def _gate_row(gates: dict, controls: dict) -> dict:
    """A row carrying only the blocks `render_gates` reads."""
    return {
        "protocol": {"recipe_hash": "r" * 64, "backbone_commit": "b"},
        "metrics": {},
        "gates": gates,
        "controls": controls,
    }


def test_a_gate_that_never_ran_is_counted_not_omitted():
    """The 988 rows before 2026-09-21 carried nine of these, every one `not_run`.

    A summary that printed only the gates that ran would have shown an empty section and
    said nothing was wrong, which is how the condition survived for months.
    """
    rows = [_gate_row({"ece": {"state": "not_run", "reason": "never evaluated"}}, {})]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "ece" in out
    assert "not_run 1" in out


def test_pass_and_fail_and_not_run_are_three_distinct_counts():
    rows = [
        _gate_row({"ece": {"state": "ran", "passed": True, "value": 0.01}}, {}),
        _gate_row({"ece": {"state": "ran", "passed": False, "value": 0.19}}, {}),
        _gate_row({"ece": {"state": "not_run", "reason": "never evaluated"}}, {}),
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "FAIL 1" in out
    assert "pass 1" in out
    assert "not_run 1" in out


def test_a_failing_gate_is_not_softened_into_a_pass():
    """`passed: False` on a row that ran is a FAIL, never folded in with not_run.

    paired_margin_vs_linear went negative on the first real row of 2026-09-21. A summary
    that reported it as anything other than a failure would be reporting a win.
    """
    rows = [
        _gate_row(
            {"paired_margin_vs_linear": {
                "state": "ran", "passed": False, "value": -0.0243,
                "detail": "paired margin -0.0243, CI includes zero, so this is not a win",
            }},
            {},
        )
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "FAIL 1" in out
    assert "pass" not in out.split("paired_margin_vs_linear")[1].split("\n")[0]
    assert "not a win" in out


def test_values_are_averaged_only_over_the_rows_that_ran():
    """A not_run row contributes no value; counting it as zero would move the mean."""
    rows = [
        _gate_row({"g": {"state": "ran", "passed": True, "value": 1.0}}, {}),
        _gate_row({"g": {"state": "ran", "passed": True, "value": 3.0}}, {}),
        _gate_row({"g": {"state": "not_run", "reason": "no"}}, {}),
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "value mean +2.0000" in out


def test_controls_are_reported_beside_gates_not_instead_of_them():
    rows = [
        _gate_row(
            {"ece": {"state": "ran", "passed": False, "value": 0.19}},
            {"degenerate_head": {"state": "ran", "passed": True, "value": 0.89}},
        )
    ]
    out = "\n".join(ledger_arms.render_gates(rows))
    assert "gates:" in out
    assert "controls:" in out
    assert "degenerate_head" in out
    assert "ece" in out


def test_the_committed_rows_still_report_every_gate_as_never_run():
    """Pins the finding to the rows themselves: the capacity sweep evaluated nothing.

    If a later change made these rows carry a verdict, this test fails and the claim in the
    handoff has to be rewritten rather than quietly becoming false.
    """
    path = (
        REPO / "ledger" / "gh200-rung0-capacity-4096-e30-2026-09-21.jsonl"
    )
    rows = ledger_arms.read_rows([path])
    assert len(rows) == 24
    out = "\n".join(ledger_arms.render_gates(rows))
    for name in (
        "ece", "needle_hunk_recall", "ood_abstain", "paired_margin_vs_linear",
        "permutation_consistency", "degenerate_head", "privileged_hunk",
        "shuffled_label", "transfer_gate",
    ):
        assert f"{name:28} not_run 24" in out, name
