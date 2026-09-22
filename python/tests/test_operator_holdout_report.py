"""The report that decides what the operator-holdout arms mean.

Its arithmetic is two subtractions, so the risk is not the arithmetic. It is that a cell
with a missing arm, an unmeasured seed or an absent sibling measurement renders as a clean
comparison -- which is the repository's own "a check that could not run must never report
the same result as a check that ran and passed", in the file that decides whether rung 0
shares the control's shortcut.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import operator_holdout_report as rep  # noqa: E402


def _row(
    *,
    operator: str = "",
    held: str = "",
    dropped: int = 0,
    op_value: float | None = 0.9,
    sib_value: float | None = 0.9,
    overall: float = 0.8,
    fit_passed: bool = True,
    op_rows: int = 5448,
    sib_rows: int = 427,
) -> dict:
    """One ledger row in the shape ``rung0_real_run`` writes.

    ``None`` for either value means the metric is present but ``not_run`` -- the shape a
    row takes when the measurement could not be made, which must never read as a zero.
    """
    recipe: dict[str, object] = {}
    if held:
        recipe["hold_out_operator"] = held
    elif operator:
        recipe["measure_operator"] = operator
    if dropped:
        recipe["drop_random_train"] = dropped

    def metric(value: float | None, n_total: int) -> dict:
        if value is None:
            return {"state": "not_run", "reason": "nothing to measure it over"}
        return {"state": "ran", "passed": True, "value": value, "n_total": n_total}

    return {
        "recipe": recipe,
        "metrics": {
            rep.OPERATOR: metric(op_value, op_rows),
            rep.SIBLINGS: metric(sib_value, sib_rows),
            rep.OVERALL: metric(overall, 12792),
            rep.FIT: {"state": "ran", "passed": fit_passed, "value": 0.7},
        },
    }


def test_the_condition_comes_from_the_recipe_and_not_from_a_label() -> None:
    """The recipe is what the hash covers; a run label is a string somebody typed."""
    assert rep.condition_of({"hold_out_operator": "stub.panic"}) == ("stub.panic", rep.HOLDOUT)
    assert rep.condition_of(
        {"measure_operator": "stub.panic", "drop_random_train": 16037}
    ) == ("stub.panic", rep.SIZEMATCH)
    assert rep.condition_of({"measure_operator": "stub.panic"}) == ("stub.panic", rep.REFERENCE)
    assert rep.condition_of({}) is None, (
        "a row that named no operator belongs to another experiment and must not be given "
        "a condition it never ran under"
    )
    assert rep.condition_of({"hold_out_operator": "stub.panic", "drop_random_train": 9}) == (
        "stub.panic",
        rep.HOLDOUT,
    ), "holding out is the condition even if rows were also dropped; it is the stronger fact"


def test_a_holdout_arm_that_keeps_its_siblings_reads_as_generator_recognition() -> None:
    """The finding this experiment exists to detect: the operator collapses, its siblings --
    same class, same shifted prior, seen in training -- do not."""
    rows = [
        _row(operator="stub.panic", op_value=0.99, sib_value=0.90),
        _row(held="stub.panic", op_value=0.05, sib_value=0.88),
    ]
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "+94.00pp drop" in text
    assert "fell 92.00pp further than its siblings" in text, text


def test_when_the_siblings_fall_too_the_drop_is_not_evidence_about_generators() -> None:
    """Holding out an operator also moves the class prior, because every operator here
    produces exactly one class. A report that called this a shortcut would be reading the
    prior shift it caused."""
    rows = [
        _row(operator="stub.panic", op_value=0.99, sib_value=0.95),
        _row(held="stub.panic", op_value=0.50, sib_value=0.40),
    ]
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "moved the class prior" in text, text
    assert "further than its siblings" not in text


def test_an_operator_that_does_not_fall_reads_as_reading_the_change() -> None:
    rows = [
        _row(operator="cosmetic.rename_local", op_value=0.60, sib_value=0.60),
        _row(held="cosmetic.rename_local", op_value=0.62, sib_value=0.59),
    ]
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "does not depend on having seen the generator" in text, text


def test_a_missing_sibling_measurement_withholds_the_reading_rather_than_guessing() -> None:
    """Without siblings on both arms the drop cannot be separated from prior shift, and a
    report that named a reading anyway would be inventing the control it lacks."""
    rows = [
        _row(operator="lonely.op", op_value=0.99, sib_value=None),
        _row(held="lonely.op", op_value=0.05, sib_value=None),
    ]
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "reading: NOT AVAILABLE" in text, text
    assert "a fact about the arm, not yet evidence about generators" in text
    assert "+94.00pp drop" in text, "the measured drop is still reported, just not read"


def test_a_missing_arm_is_named_rather_than_rendering_as_a_clean_comparison() -> None:
    rows = [_row(operator="stub.panic"), _row(held="stub.panic")]
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "MISSING ARM(S): sizematch" in text, text
    assert "cannot be separated from the smaller training set" in text


def test_a_seed_that_measured_nothing_is_counted_and_never_read_as_zero() -> None:
    """A seed that recorded no measurement and a seed that scored zero are different facts,
    and averaging the first in as a zero is how an arm is made to look like a collapse."""
    rows = [
        _row(operator="stub.panic", op_value=0.90),
        _row(operator="stub.panic", op_value=None),
    ]
    cell = rep.cells_of(rows)[("stub.panic", rep.REFERENCE)]
    assert cell.unmeasured == 1
    assert cell.operator_acc == [0.90], "the unmeasured seed contributes no value at all"
    assert cell.n_seeds == 2, "and is still counted, so the arm cannot claim 1 of 1"
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "recorded no operator metric at all" in text


def test_a_collapsed_seed_is_surfaced_because_its_held_out_number_is_the_prior() -> None:
    rows = [
        _row(operator="stub.panic", fit_passed=False),
        _row(operator="stub.panic", fit_passed=True),
    ]
    cell = rep.cells_of(rows)[("stub.panic", rep.REFERENCE)]
    assert cell.collapsed == 1
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "did not clear the training-set majority" in text


def test_a_delta_against_an_empty_arm_is_withheld_rather_than_computed() -> None:
    assert rep._delta([], [0.5]) == "not measured"
    assert rep._delta([0.5], []) == "not measured"
    assert rep._delta([0.9], [0.5]) == "+40.00pp drop"


def test_a_single_seed_says_so_instead_of_reporting_a_spread_of_zero() -> None:
    assert "no spread" in rep._mean_sd([0.9])
    assert rep._mean_sd([]) == "not measured"
    assert "+/-" in rep._mean_sd([0.9, 0.8])


def test_rows_from_another_experiment_are_skipped_not_miscounted() -> None:
    """The capacity grid's rows sit in ledgers beside these and name no operator."""
    rows = [_row(), _row(operator="stub.panic")]
    cells = rep.cells_of(rows)
    assert set(cells) == {("stub.panic", rep.REFERENCE)}
    assert sum(c.n_seeds for c in cells.values()) == 1
