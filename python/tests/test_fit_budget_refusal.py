"""A fit must refuse a cap it cannot finish inside, rather than be killed by it.

`fit_linear_control.py` computed `projected_fit_seconds`, printed it, and started anyway.
When a caller wrapped it in a shorter `timeout` the result was the worst available outcome:
every core of the box busy for the length of the cap, the process killed before it
converged, and nothing written to the cache -- which the arm that reads the cache records
as `paired_margin_vs_linear: not_run`, indistinguishable from a fit that was never
launched. The hour leaves no trace saying it was spent.

Commit 5fd0ea8 had already recorded this once ("The linear control could not finish inside
the cap it was launched under"), so these tests pin the precondition rather than the
incident.

A second defect sits on top of the first and is pinned here too. The projection the
precondition consumes is a WORST-CASE bound at `max_iter`, not an estimate: measured
2026-09-22, a fit projected at 101.4 minutes converged in 443 iterations and 345.7s, an
overshoot of 17.6x. Caps must therefore be set above the projection, while COSTS must never
be quoted from it -- doing that turned a $0.91 job into a documented $22 one.

Worth stating plainly, because these tests were written believing otherwise: on this corpus
the original `timeout 3600` would never have killed a fit, and the claim that it doomed
every fit was itself read off the bad projection. The precondition is defensible on the
general case -- a cap and a tool that never compare notes keep 5fd0ea8 alive -- not on a
failure that was actually happening here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "fit_operator_holdout_controls.sh"

#: Words that mark a retracted figure rather than an asserted one.
_RETRACTION = ("earlier", "overshoot", "overshot", "17.6x", "wrong")


def assert_only_as_retraction(src: str, phrase: str) -> None:
    """The phrase may appear only where the surrounding text retracts it.

    A blunt "this string must not appear" forbids the sentence that corrects the mistake,
    which pushes the fix towards deleting the history instead of recording it. What must
    not survive is the CLAIM; the correction is the point.
    """
    for match in re.finditer(re.escape(phrase), src):
        window = src[max(0, match.start() - 400) : match.end() + 400].lower()
        assert any(mark in window for mark in _RETRACTION), (
            f"{phrase!r} appears at offset {match.start()} with nothing nearby marking it "
            "as a retracted figure, so it reads as a current claim"
        )


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
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"timeout[^\n]*\b3600\b", code), (
        "no literal 3600 may survive in EXECUTABLE lines beside the constant; that is the "
        "drift this prevents. The header may still quote it when describing the old bug."
    )
    assert cap_min > 101.4, (
        "the cap must exceed the measured 101.4-minute projection, or every fit this "
        "script queues is unrunnable before it starts -- which is the bug itself"
    )


def test_the_script_quotes_measured_fit_times_and_not_the_projection() -> None:
    """The script briefly carried a `QD_CONTROL_FIT_ACK` gate demanding a human
    acknowledgement before spending "~15 hours, roughly $22". Both numbers came from
    `projected_fit_seconds`. The seven fits actually took 2190.4s in total -- 36.5 minutes,
    about $0.91 -- so the gate was friction protecting nobody from a cost that did not
    exist, and it was removed rather than left as a monument to a bad estimate.

    What replaces it is the measurement. A header that states per-fit times somebody can
    check is worth more than a confirmation prompt keyed to a number nobody measured.
    """
    src = SCRIPT.read_text()
    assert "2190.4s" in src, "the measured total must be in the header"
    assert "345.7s" in src and "204.2s" in src, "per-fit times, so the spread is visible"
    assert "17.6x" in src, "and the overshoot factor, so the projection is not trusted again"
    assert "QD_CONTROL_FIT_ACK" not in src, (
        "the acknowledgement gate guarded a cost that was off by 17.6x; it should be gone "
        "rather than re-tuned, because the thing that was wrong was the estimate"
    )
    assert_only_as_retraction(src, "15 hours")


def test_the_refusal_documents_that_it_consumes_a_worst_case_bound() -> None:
    """The guard is only safe if whoever sets a cap knows what it is compared against.
    Against a worst-case bound, a cap chosen from observed times over-refuses: a 60-minute
    cap would reject a fit that finishes in six. That is cheap to state and expensive to
    rediscover, and it is the reason FIT_CAP_MIN is 150 rather than 10."""
    src = (REPO / "tools" / "fit_linear_control.py").read_text()
    assert "worst-case bound" in src
    assert "345.7s" in src, "with the measurement that showed the gap"
    assert "17.6x" in src


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


def _refused_inline_fit(monkeypatch, cache_dir):
    """Drive the arm's inline-fit refusal with a budget no projection can clear."""
    pytest.importorskip("torch", reason="rung0_real_run imports torch at module scope")
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "python"))
    from dataclasses import dataclass

    import rung0_real_run

    from qd_train.mutate_adapter import MUTATION_CLASSES

    class _Ctx:
        def __init__(self, text: str) -> None:
            self.ids = list(text.encode("utf-8"))
            self.n_bytes_kept = len(self.ids)

    @dataclass(frozen=True, slots=True)
    class _D:
        context: object
        gold_option: int

    train = [
        _D(_Ctx(f"def f{i}(): return {i}\n" * 3), i % len(MUTATION_CLASSES))
        for i in range(40)
    ]
    monkeypatch.setattr(rung0_real_run, "LINEAR_CONTROL_TIME_BUDGET_S", 0.0)
    correct, not_run = rung0_real_run.linear_baseline_correctness(
        train, train[:20], seed=0, max_iter=6000, cache_dir=cache_dir
    )
    assert correct is None, "a refused control was scored anyway"
    assert not_run is not None
    return not_run.reason


def test_the_arm_s_refusal_calls_the_projection_a_bound_not_a_cost(monkeypatch) -> None:
    """This reason is written into every row whose arm missed the cache, and it said the
    projection was "a statement about the control's cost at this corpus size". Read that
    way it priced a fit that converges in minutes at 25.7 hours -- which is how 104 rows
    of this experiment came to read as though their opponent were unaffordable. The
    projection prices max_iter iterations; the row must say so."""
    reason = _refused_inline_fit(monkeypatch, None)
    assert "statement about the control's cost" not in reason
    assert "6000 max_iter iterations" in reason
    assert "worst-case bound" in reason and "NOT what the fit would cost" in reason
    assert "re-run this arm" in reason and "does not backfill" in reason
    assert "no --control-cache was given" in reason


def test_the_arm_s_refusal_names_the_cache_and_key_it_missed(monkeypatch, tmp_path) -> None:
    """A miss and a missing cache flag are different failures with different fixes, and the
    key is what lets somebody check the fit tool wrote the entry this arm asked for."""
    reason = _refused_inline_fit(monkeypatch, tmp_path)
    assert f"{tmp_path} has no control for this run's training set (key " in reason
    assert re.search(r"\(key [0-9a-f]{16};", reason), reason


def test_the_script_says_that_warming_the_cache_does_not_backfill_written_rows() -> None:
    """`rung0_real_run.py` reads the cache once, before its seed loop, and never fits
    inline. Somebody who ran this expecting the 72 existing rows to gain margins would
    spend the fits and still have no margin, and the header is where that is cheapest to
    learn. (The spend turned out to be 36.5 minutes rather than the 15 hours first written
    here, but the point is unchanged: it buys cached verdicts, not rows.)"""
    src = SCRIPT.read_text()
    assert "does not retrofit rows already written" in src.lower()
    assert "re-run" in src.lower()
