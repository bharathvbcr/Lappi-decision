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

import pytest

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
    margin: float | None = -0.08,
    seed: int | None = None,
    rev: str = "",
    row_id: str = "",
) -> dict:
    """One ledger row in the shape ``rung0_real_run`` writes.

    ``None`` for either value means the metric is present but ``not_run`` -- the shape a
    row takes when the measurement could not be made, which must never read as a zero.
    ``seed=None`` leaves out ``protocol`` entirely, which every real row carries; the tests
    that are not about seeds use that so each row is its own population member.
    """
    recipe: dict[str, object] = {}
    if held:
        recipe["hold_out_operator"] = held
    elif operator:
        recipe["measure_operator"] = operator
    if dropped:
        recipe["drop_random_train"] = dropped
    if rev:
        recipe["rev"] = rev
    extra: dict[str, object] = {}
    if seed is not None:
        extra["protocol"] = {"seed": seed}
    if row_id:
        extra["row_id"] = row_id

    def metric(value: float | None, n_total: int) -> dict:
        if value is None:
            return {"state": "not_run", "reason": "nothing to measure it over"}
        return {"state": "ran", "passed": True, "value": value, "n_total": n_total}

    return {
        "recipe": recipe,
        # The runner records the margin as a GATE, not a metric, which is why the report
        # reads both containers.
        "gates": {rep.MARGIN: metric(margin, 12792)},
        "metrics": {
            rep.OPERATOR: metric(op_value, op_rows),
            rep.SIBLINGS: metric(sib_value, sib_rows),
            rep.OVERALL: metric(overall, 12792),
            rep.FIT: {"state": "ran", "passed": fit_passed, "value": 0.7},
        },
        **extra,
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


def test_an_arm_with_no_fitted_control_reports_no_margin_rather_than_a_loss() -> None:
    """Every holdout and size-matched arm trains on a reduced set, and until the controls
    are fitted on those sets there is no opponent at all. A report that showed an absent
    margin as a zero, or omitted it silently, would turn "the model was not compared" into
    "the model did not win" -- which is the finding this experiment might overturn.
    """
    rows = [_row(held="stub.panic", margin=None), _row(held="stub.panic", margin=None)]
    cell = rep.cells_of(rows)[("stub.panic", rep.HOLDOUT)]
    assert cell.margins == [] and cell.margin_not_run == 2
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "Not measured -- never a loss" in text, text
    assert "fit_operator_holdout_controls.sh" in text, "and it says how to get one"


def test_a_positive_margin_is_counted_because_none_has_ever_been_seen() -> None:
    """120-plus rows across two corpora, three widths and four rates have produced zero
    positive margins. If an operator-holdout arm produces one it is the first, so the count
    is reported beside the mean rather than left to be inferred from a sign."""
    rows = [_row(operator="stub.panic", margin=0.03), _row(operator="stub.panic", margin=-0.01)]
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "1 of 2 seed(s) positive" in text, text


def test_rows_from_another_experiment_are_skipped_not_miscounted() -> None:
    """The capacity grid's rows sit in ledgers beside these and name no operator."""
    rows = [_row(), _row(operator="stub.panic")]
    cells = rep.cells_of(rows)
    assert set(cells) == {("stub.panic", rep.REFERENCE)}
    assert sum(c.n_seeds for c in cells.values()) == 1


def test_a_rerun_of_the_same_seed_is_refused_rather_than_pooled() -> None:
    """The margins are closed by RE-RUNNING arms that already have rows. Grouping ignores
    rev on purpose, so the re-run's seed 0 and the first run's seed 0 would land in one
    cell: two "seeds" where one exists, the spread understated, and the first run's not_run
    margin averaged in beside the re-run's measured one. Refused, naming both rows."""
    rows = [
        _row(held="stub.panic", seed=0, rev="ecbd370a4d40", row_id="first", margin=None),
        _row(held="stub.panic", seed=0, rev="52c8fe9fc39a", row_id="rerun", margin=-0.05),
    ]
    with pytest.raises(rep.RepeatedSeed) as refused:
        rep.cells_of(rows)
    msg = str(refused.value)
    assert "stub.panic holdout seed 0" in msg
    assert "row first (rev ecbd370a4d40)" in msg and "row rerun (rev 52c8fe9fc39a)" in msg


def test_a_refusal_says_how_many_it_did_not_show() -> None:
    """A capped list presented as the whole list is a coverage claim nobody made."""
    rows = [_row(operator="stub.panic", seed=s, row_id=f"a{s}") for s in range(7)]
    rows += [_row(operator="stub.panic", seed=s, row_id=f"b{s}") for s in range(7)]
    with pytest.raises(rep.RepeatedSeed) as refused:
        rep.cells_of(rows)
    assert "showing 5 of 7" in str(refused.value)


def test_the_same_seed_in_different_cells_is_not_a_repeat() -> None:
    """Every arm runs seeds 0..7, and every operator's arms share them. Only a repeat inside
    one (operator, condition) cell is two measurements of one seed."""
    rows = [
        _row(operator="stub.panic", seed=0, rev="ecbd370a"),
        _row(held="stub.panic", seed=0, rev="ecbd370a"),
        _row(operator="stub.panic", dropped=16037, seed=0, rev="ecbd370a"),
        _row(operator="stub.default_return", seed=0, rev="728562b3"),
    ]
    cells = rep.cells_of(rows)
    assert len(cells) == 4
    assert all(c.unseeded == 0 for c in cells.values())


def test_one_ledger_read_twice_is_refused_on_real_rows(tmp_path, capsys) -> None:
    """The likeliest way to double a population is to name one ledger twice on the command
    line. Checked on the committed rows, so the seed field is where the real runner puts it."""
    ledger = REPO / "ledger" / "gh200-operator-holdout-model-2026-09-22.jsonl"
    assert rep.main([str(ledger)]) == 0
    capsys.readouterr()
    assert rep.main([str(ledger), str(ledger)]) == 2
    assert "refusing to pool" in capsys.readouterr().err


def test_a_missing_margin_prints_the_reason_the_run_recorded() -> None:
    """The report used to say "no control was fitted on this arm's training set" for every
    not_run margin. That is the usual cause and not the only one -- an unconverged control
    and an over-budget projection are recorded the same way -- and the row carries the
    reason, so the report prints what the run said rather than what it probably meant."""
    rows = [_row(held="stub.panic", margin=None)]
    text = "\n".join(rep.render(rep.cells_of(rows)))
    assert "[1 seed(s)] nothing to measure it over" in text, text
    assert "no control was fitted on this arm's training set" not in text


def test_rows_without_a_seed_say_the_repeat_check_could_not_run() -> None:
    """A check that could not run must not read like one that ran and found nothing."""
    text = "\n".join(rep.render(rep.cells_of([_row(operator="stub.panic")])))
    assert "1 row(s) carry no protocol.seed" in text, text


def test_the_docstring_no_longer_says_the_control_cannot_be_fitted() -> None:
    """It said fit_linear_control.py "has no holdout flags", which stopped being true when
    the flags landed; the launcher's header was corrected and this one was missed."""
    src = (REPO / "tools" / "operator_holdout_report.py").read_text()
    assert "has no holdout flags" not in src
