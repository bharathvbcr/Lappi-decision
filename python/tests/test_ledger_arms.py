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


def test_the_arm_vs_arm_floor_is_the_wider_one() -> None:
    """The sqrt(2). Two noisy arms, not an arm against a fixed number.

    Against a version that passes ``against_known_reference=True`` for a difference
    between arms, the printed floor is 1.41x too small -- and 1.41x too small is always
    the direction that turns a null into a finding.
    """
    left = [_row(backbone="aaa", val=v) for v in (0.30, 0.34, 0.32, 0.36)]
    right = [_row(backbone="bbb", val=v) for v in (0.40, 0.44, 0.42, 0.46)]
    text = "\n".join(ledger_arms.render_pairwise(ledger_arms.arms_of(left + right)))
    pairwise = float(text.split("floor at n=4 ", 1)[1].split("pp", 1)[0])

    one_arm = "\n".join(ledger_arms.render(ledger_arms.arms_of(left)))
    fixed = float(one_arm.split("difference at this n: ", 1)[1].split("pp", 1)[0])
    # Both numbers are read back off a line rounded to two decimals, so the tolerance is
    # the printing precision and not the arithmetic's. It is still an order of magnitude
    # tighter than the gap being tested: against the one-sample floor this reads 3.62
    # against an expected 5.12, which no rounding accounts for.
    assert pairwise == pytest.approx(fixed * 2**0.5, abs=0.02)


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
