"""The operator-holdout scripts' contract: the arm launcher, its control fitter, and the
operator list they share.

Running the launcher would spend GPU hours and running the fitter would fit the corpus, so
those two are read as text. The operator list is a sourced file with no side effects, so it
is executed under bash and its behaviour asserted. The defects pinned here are a stale claim
in a header that a reader would act on, an override that could silently run zero arms while
still printing the marker a watcher treats as success, two scripts that could run over
different operators, and a re-run that would pool its seeds with the run it repeats.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "launch_operator_holdout_model.sh"
FITTER = REPO / "tools" / "fit_operator_holdout_controls.sh"
OPERATORS_FILE = REPO / "tools" / "operator_holdout_operators.sh"
DEFAULT_OPERATORS = ("stub.panic", "logic.change_constant", "cosmetic.rename_local")
SOURCE_LINE = 'source "$(dirname "${BASH_SOURCE[0]}")/operator_holdout_operators.sh"'


def _source_operators(override: str | None) -> subprocess.CompletedProcess[str]:
    """Source the shared list the way both scripts do, then print what it produced."""
    env = {k: v for k, v in os.environ.items() if k != "QD_HOLDOUT_OPERATORS"}
    if override is not None:
        env["QD_HOLDOUT_OPERATORS"] = override
    return subprocess.run(
        ["bash", "-c", f'source "{OPERATORS_FILE}"; echo "ran:$OPERATORS"'],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _code_lines(src: str) -> str:
    """The executable lines: the headers are allowed to describe what used to be there."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def test_the_operator_list_is_overridable_without_editing_the_script() -> None:
    """The three defaults are the dominant operator in each class, which is the weakest
    version of this test: holding out the biggest operator moves the class prior hardest,
    and that is precisely the confound the sibling metric then has to rule out. Measuring a
    minor operator must not require a second copy of the script that can drift from it.
    """
    done = _source_operators("logic.negate_condition stub.default_return")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ran:logic.negate_condition stub.default_return"


def test_the_default_operator_list_is_unchanged() -> None:
    """72 rows were written against these three. A default that quietly changed would make
    the next run incomparable with them while looking like a rerun."""
    done = _source_operators(None)
    assert done.returncode == 0, done.stderr
    assert tuple(done.stdout.strip().removeprefix("ran:").split()) == DEFAULT_OPERATORS


def test_an_empty_override_refuses_instead_of_running_nothing() -> None:
    """`for OP in $OPERATORS` over an empty string runs zero arms, falls through to the
    DONE marker, and exits 0. A watcher grepping for that marker cannot tell that from a
    finished run, so the ledger would simply be missing rows nobody was told about.

    Executed rather than read: the refusal has to stop the SOURCING script, which is what
    `exit` inside a sourced file does and what a `return` would silently not do.
    """
    for empty in ("", "   "):
        done = _source_operators(empty)
        assert done.returncode == 2, (empty, done.stdout, done.stderr)
        assert "ran:" not in done.stdout, "the script carried on past the refusal"
        assert "refusing" in done.stderr


def test_the_launcher_and_the_fitter_take_their_operators_from_the_same_file() -> None:
    """The fitter kept its own hardcoded list after the launcher learned to take an
    override, so the minor operators could run as arms while their controls could not be
    fitted without editing the fitter -- and all 48 of their holdout and size-matched rows
    came back with no margin. One file now owns the list; both scripts must source it before
    their loop, and neither may define OPERATORS itself.
    """
    for path in (SCRIPT, FITTER):
        src = path.read_text()
        code = _code_lines(src)
        assert SOURCE_LINE in code, f"{path.name} must source the shared operator list"
        assert code.index(SOURCE_LINE) < code.index("for OP in $OPERATORS"), (
            f"{path.name} sources the list after its loop, which runs over nothing"
        )
        assert not re.search(r"^\s*OPERATORS=", code, re.M), (
            f"{path.name} defines OPERATORS itself, which is the second copy that drifted"
        )


def test_a_rerun_can_write_to_its_own_ledger() -> None:
    """The report groups rows by (operator, condition). A second run of the same seeds
    appended to the first run's ledger would be read as sixteen seeds where eight exist,
    with the not_run margins of the first run mixed in beside the measured ones."""
    code = _code_lines(SCRIPT.read_text())
    line = re.search(r'^LEDGER="\$\{QD_HOLDOUT_LEDGER:-([^}]*)\}"', code, re.M)
    assert line is not None, "LEDGER must take a QD_HOLDOUT_LEDGER override"
    assert line.group(1) == (
        "/home/ubuntu/qwen-decision/ledger/gh200-operator-holdout-model-2026-09-22.jsonl"
    ), "the default must stay the ledger the first 144 rows were written to"
    assert '--ledger "$LEDGER"' in code


def test_the_fitter_reads_the_held_out_count_from_its_own_log() -> None:
    """The size-matched control drops as many rows as the holdout removed. The fitter read
    that count from the holdout ARM's log, so no control could be fitted for an operator
    until a GPU arm had already trained without one -- the exact ordering that left every
    holdout and size-matched margin not_run. fit_linear_control.py prints the same line,
    through the same filter_train_rows, so the fitter reads its own."""
    code = _code_lines(FITTER.read_text())
    assert 'HOLD_LOG=/home/ubuntu/opctl-holdout-"$OP".log' in code
    assert "ophold-holdout-" not in code, "no fit may wait on an arm's log"


def test_the_header_no_longer_claims_the_control_cannot_be_fitted() -> None:
    """It said "fit_linear_control.py has no holdout flags". That stopped being true: it
    takes --hold-out-operator and --drop-random-train. A reader who believed the old
    sentence would conclude the missing margins were impossible rather than unpurchased,
    which is the opposite of what the not_run rows mean."""
    src = SCRIPT.read_text()
    assert "has no holdout flags" not in src
    assert "--hold-out-operator" in src, "the header should say what the flags are"


def test_no_script_claims_the_arm_never_fits_inline() -> None:
    """Both headers said rung0_real_run.py "never fits inline". It does, whenever the fit
    projects under LINEAR_CONTROL_TIME_BUDGET_S; on this corpus no arm's projection came
    near that, which is a fact about the corpus and not about the code. The replacement
    names the budget, so the number is checked against the constant it describes."""
    runner = (REPO / "tools" / "rung0_real_run.py").read_text()
    budget = re.search(
        r"^LINEAR_CONTROL_TIME_BUDGET_S: Final\[float\] = ([0-9.]+)", runner, re.M
    )
    assert budget is not None
    budget_s = f"{float(budget.group(1)):g}s"
    for path in (SCRIPT, FITTER):
        # Comment lines joined back into prose, so a re-wrap cannot hide the claim.
        prose = re.sub(r"\n\s*#\s*", " ", path.read_text())
        assert "never fits inline" not in prose, f"{path.name} still says it"
        assert f"under its {budget_s} LINEAR_CONTROL_TIME_BUDGET_S" in prose, (
            f"{path.name} must state the budget, and state it as {budget_s}"
        )


def test_the_header_carries_the_measured_cost_not_the_projection() -> None:
    """The reason those 48 rows had no margin is cost, not capability -- and the cost has
    now been paid and measured, so the header must carry the measurement.

    This test previously asserted the header said "101.4 minute", which was
    `projected_fit_seconds` quoted as if it were a duration. It is a worst-case bound at
    max_iter; the fits converge in ~440 iterations and take 204-352s, so that number
    overstated a $0.91 job as a $22 one. Quoting a projection as a cost is the defect, so
    the test now forbids exactly that.
    """
    src = SCRIPT.read_text()
    assert "2190.4s" in src or "36.5 minutes" in src, "the measured total must be stated"
    assert "17.6x" in src, "and the overshoot, so the projection is not trusted again"
    assert "nine is ~15 hours" not in src, "the false claim itself must not survive"
    low = src.lower()
    assert "backfill" in low and "not" in low, (
        "it must say that warming the cache leaves already-written rows unchanged"
    )
    assert "re-run" in low, "and that a re-run is what actually produces the margins"
