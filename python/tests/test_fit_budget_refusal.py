"""A fit must refuse a cap it cannot finish inside, rather than be killed by it.

`fit_linear_control.py` computed `projected_fit_seconds`, printed it, and started anyway.
When a caller wrapped it in a shorter `timeout` the result was the worst available outcome:
every core of the box busy for the length of the cap, the process killed before it
converged, and nothing written to the cache -- which the arm that reads the cache records
as `paired_margin_vs_linear: not_run`, indistinguishable from a fit that was never
launched. The hour leaves no trace saying it was spent.

Measured on 2026-09-22: `tools/fit_operator_holdout_controls.sh` wrapped a 101.4-minute
projection in `timeout 3600` and queued nine of them. Commit 5fd0ea8 had already recorded
this once ("The linear control could not finish inside the cap it was launched under"), so
these tests pin the precondition rather than the incident.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "fit_operator_holdout_controls.sh"


def _tool():
    """Import the tool, skipping when torch is absent.

    The module reaches `rung0_real_run` for its imported defaults, which pulls in torch.
    The refusal under test needs none of it, but the import does.
    """
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "tools"))
    import fit_linear_control

    return fit_linear_control


def test_a_projection_inside_the_cap_starts() -> None:
    assert _tool().fit_budget_refusal(60.0, 10.0) is None


def test_no_cap_means_any_duration_is_acceptable() -> None:
    """The default for an interactive fit that owns its own terminal: the caller waits as
    long as it takes, and there is no `timeout` to be killed by."""
    assert _tool().fit_budget_refusal(99_999.0, None) is None


def test_a_projection_over_the_cap_refuses_and_names_both_numbers() -> None:
    """A refusal that did not carry the projection would leave the caller guessing at what
    cap would actually work, which is how a cap gets raised twice."""
    msg = _tool().fit_budget_refusal(101.4 * 60, 60.0)
    assert msg is not None
    assert "101.4 minute(s)" in msg
    assert "60" in msg
    assert "NOTHING would be cached" in msg


def test_the_refusal_forbids_the_shortcut_that_would_weaken_the_control() -> None:
    """The cheapest way to fit inside a cap is to cut --max-iter, and it is the one way
    that corrupts the result: a control that stopped early is a weaker opponent, and every
    margin measured against it afterwards is inflated. Rule 2 makes that a human's call."""
    msg = _tool().fit_budget_refusal(10_000.0, 1.0)
    assert msg is not None
    assert "--max-iter" in msg


def test_the_cap_is_a_boundary_not_a_strict_inequality() -> None:
    """A projection of exactly the cap is affordable; refusing it would be off by one in
    the direction that wastes a fit nobody had to skip."""
    tool = _tool()
    assert tool.fit_budget_refusal(600.0, 10.0) is None
    assert tool.fit_budget_refusal(600.1, 10.0) is not None


def test_the_cli_exposes_the_cap_so_a_caller_can_pass_its_own() -> None:
    """A refusal no caller can reach is not a fix. The wrapping script must be able to hand
    down the same number it gives `timeout`."""
    tool = _tool()
    with pytest.raises(SystemExit):
        tool.main(["--help"])
    parser_src = (REPO / "tools" / "fit_linear_control.py").read_text()
    assert '"--max-fit-minutes"' in parser_src


def test_the_script_feeds_one_number_to_both_the_timeout_and_the_tool() -> None:
    """The defect was two numbers that could disagree: a `timeout 3600` beside a tool with
    no cap at all. One constant must reach both, or they drift apart again the next time
    somebody raises one of them.

    This reads the script as text deliberately -- running it would fit the corpus.
    """
    src = SCRIPT.read_text()
    assert re.search(r"^FIT_CAP_MIN=(\d+)", src, re.M), "the cap must be a named constant"
    cap_min = int(re.search(r"^FIT_CAP_MIN=(\d+)", src, re.M).group(1))
    assert re.search(r"^FIT_CAP_S=\$\(\(\s*FIT_CAP_MIN \* 60\s*\)\)", src, re.M), (
        "the seconds form must be DERIVED from the minutes form, not written out again"
    )
    assert '--max-fit-minutes "$FIT_CAP_MIN"' in src, (
        "the tool must be told the same cap the timeout enforces"
    )
    assert 'timeout --signal=TERM --kill-after=120 "$FIT_CAP_S"' in src, (
        "the timeout must use the derived constant, not a literal"
    )
    assert not re.search(r"timeout[^\n]*\b3600\b", src), (
        "no literal 3600 may survive beside the constant; that is the drift this prevents"
    )
    assert cap_min > 101.4, (
        "the cap must exceed the measured 101.4-minute projection, or every fit this "
        "script queues is unrunnable before it starts -- which is the bug itself"
    )


def test_the_script_will_not_spend_fifteen_hours_without_being_told_to() -> None:
    """Nine fits at ~101 minutes is about 15 hours of a rented box, roughly $22 -- over the
    line rule 4 draws around a job that needs a human yes. The script must fail closed."""
    src = SCRIPT.read_text()
    # Anchored on the GUARD, not on the first mention of the name: the header documents
    # the variable several paragraphs earlier, and a test that matched that would pass on
    # a script whose guard had been deleted.
    guard = src.find('if [ "${QD_CONTROL_FIT_ACK:-0}" != "1" ]')
    assert guard != -1, "the acknowledgement must be an executable guard, not just prose"
    assert "exit 2" in src[guard : guard + 800], "the unacknowledged path must not fall through"


def test_no_docstring_still_calls_the_fit_single_threaded() -> None:
    """Both docstrings said the control fit was single-threaded CPU work, and a caller
    believed them: `fit_operator_holdout_controls.sh` was written to run fits beside GPU
    arms on the theory that they cost nothing the GPU wanted. Measured, one fit drove load
    average to 64.8 on a 64-core box -- it is a dense BLAS GEMM -- and a span arm's seed
    went from ~92s to 140.5s while one ran beside it.

    The claim is what made the scheduling wrong, so the claim is what is pinned. When a
    document and the code disagree the code wins, and then the document gets fixed.
    """
    for path in (
        REPO / "tools" / "fit_linear_control.py",
        REPO / "python" / "qd_train" / "control_cache.py",
    ):
        src = path.read_text()
        assert "is single-threaded CPU work" not in src, (
            f"{path.name} still asserts the fit is single-threaded; it is BLAS-parallel "
            "and saturates every core, which is why nine fits cost nine times one"
        )


def test_the_script_says_that_warming_the_cache_does_not_backfill_written_rows() -> None:
    """`rung0_real_run.py` reads the cache once, before its seed loop, and never fits
    inline. Somebody who ran this expecting the 72 existing rows to gain margins would
    spend 15 hours and get nothing, and the header is where that is cheapest to learn."""
    src = SCRIPT.read_text()
    assert "does NOT retrofit rows already written" in src
    assert "re-run" in src.lower()
