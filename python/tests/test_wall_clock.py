"""``wall_clock_s`` must be the run's duration, not the recorder's lifetime.

``RunRecorder._finish`` computes ``wall = time.monotonic() - self._t0`` and ``_t0`` is set
in ``__enter__``. That is correct for a caller whose ``with`` block CONTAINS the run, and
silently wrong for a caller that does the work first and enters the recorder afterwards to
report it -- which records how long it took to write the row.

Measured over ``ledger/*.jsonl`` before this change: 346 of 799 rows carried a
``wall_clock_s`` under 0.1s, including all 53 rows from ``tools/rung0_real_run.py``, whose
runs the same process printed at 239.0s. ``cost_usd`` is derived from the same number, so
those rows price a GPU hour at zero.

The recorder cannot detect which shape its caller has -- only the caller knows whether the
``with`` block contains the run. So the caller says, and ``None`` is a positive statement
("this recorder wraps the run") rather than an absence.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.ledger import (
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    RunRecorder,
    verify_no_fork,
)
from qd_train.run_control import CostEstimate, WallClockCap
from qd_train.tristate import NotRun

REPO = Path(__file__).resolve().parents[2]


def _protocol(seed: int = 0) -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64,
        tokenizer_hash="t" * 64,
        backbone_commit="b" * 40,
        recipe_hash="r" * 64,
        seed=seed,
    )


def _env() -> Environment:
    return Environment(
        torch="2.12.1", transformers_sha="abc123", device="mps", host="test",
        fla_present=NotRun(reason="no CUDA on this host"),
        causal_conv1d_present=NotRun(reason="no CUDA on this host"),
    )


def _rate(usd_per_hour: float) -> CostEstimate:
    """A billed rate, for the two tests that assert cost follows the measured duration."""
    return CostEstimate(
        cap=WallClockCap(cap_s=3600.0),
        usd_per_hour=usd_per_hour,
        n_gpus=1,
        instance="test-1gpu",
    )


def _recorder(led: Ledger, **kw: object) -> RunRecorder:
    # `cost` is required and has no default, and these tests are about the CLOCK. The env
    # below is a local device, so `None` is the honest answer and the recorder accepts it;
    # test_row_cost.py is where the cost argument itself is exercised.
    kw.setdefault("cost", None)
    return RunRecorder(
        led, protocol=_protocol(), run_kind="ft", repo=REPO, env=_env(), **kw  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Characterisation: what the unmodified recorder does. Both of these pass
# before and after the change -- they pin the behaviour the fix must preserve.
# --------------------------------------------------------------------------

def test_a_recorder_that_wraps_the_run_times_the_run(tmp_path: Path):
    """``None`` is the wrapping caller's answer, and it must keep timing the block.

    Written first without the argument, where it passed against the pre-fix code; making
    the argument required is what added the ``None``. It pins the behaviour the caller
    that really does wrap its run still depends on -- ``record_build`` below, and
    ``real_ft_run``'s ft path.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=None):
        time.sleep(0.05)
    assert led.rows()[0].wall_clock_s >= 0.05


def test_a_recorder_entered_after_the_work_times_only_itself(tmp_path: Path):
    """The defect, stated as a fact about the mechanism rather than a complaint.

    This is what every ``rung0_real_run.py`` row in the ledger was: the work happened
    outside the block, so the recorder reported the block. It stays true of ``None``
    afterwards -- which is precisely why ``None`` cannot be the default. A caller that
    has not thought about it gets this, silently.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    time.sleep(0.05)  # "training", outside the block
    with _recorder(led, wall_clock_s=None):
        pass
    assert led.rows()[0].wall_clock_s < 0.05


# --------------------------------------------------------------------------
# The fix.
# --------------------------------------------------------------------------

def test_a_measured_wall_clock_is_recorded_verbatim(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=239.0):
        pass
    assert led.rows()[0].wall_clock_s == 239.0


def test_cost_comes_from_the_measured_wall_clock_not_the_recorder_lifetime(tmp_path: Path):
    """The mutation this kills: setting ``wall`` from the argument and leaving ``cost_usd``
    derived from ``self._t0``. They are computed on adjacent lines and a fix that moves one
    and not the other still passes every test above.

    One hour at $2.20/h is $2.20. A recorder that priced its own lifetime would bill about
    six millionths of a cent.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=3600.0, cost=_rate(2.20)):
        pass
    assert led.rows()[0].cost_usd == pytest.approx(2.20)


def test_a_measured_wall_clock_survives_a_failing_run(tmp_path: Path):
    """A run that raises still cost what it cost. ``_finish`` has three callers -- the
    completed path, the exception path and the signal path -- and a fix applied to only
    the first leaves a failed GPU run priced at zero.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(RuntimeError), _recorder(led, wall_clock_s=118.5, cost=_rate(2.20)):
        raise RuntimeError("CUDA OOM")
    row = led.rows()[0]
    assert row.status == "failed"
    assert row.wall_clock_s == 118.5
    assert row.cost_usd == pytest.approx(2.20 * 118.5 / 3600.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_a_wall_clock_that_is_not_a_measurement_is_refused(tmp_path: Path, bad: float):
    """``LedgerRow`` rejects a negative duration with ``wall_clock_s < 0``. NaN fails that
    comparison -- ``nan < 0`` is False -- so a NaN duration reached the ledger, and a NaN
    cost with it. A number that is not a measurement must not be recorded as one.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(ValueError, match="measured"), _recorder(led, wall_clock_s=bad):
        pass


# --------------------------------------------------------------------------
# Which of the two numbers a row is carrying.
#
# `wall_clock_s` now holds the right duration, but a reader of a row still cannot tell a
# caller's measurement from a wrapped block without opening the tool that wrote it. The
# field below says. It CAN carry a default where `wall_clock_s` could not, and the
# difference is the point: "unrecorded" is true of a row that never stated a source, where
# a defaulted duration was a number nobody measured presented as one somebody did.
# --------------------------------------------------------------------------

def test_a_caller_measured_row_says_so(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=239.0):
        pass
    assert led.rows()[0].wall_clock_source == "caller"


def test_a_wrapped_row_says_so(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=None):
        pass
    assert led.rows()[0].wall_clock_source == "recorder"


def test_a_row_written_before_this_field_existed_reads_back_unrecorded(tmp_path: Path):
    """The mutation this kills: defaulting old rows to "recorder".

    It would be true of every row written before a409895, and it would still be a claim
    the file does not make. 799 rows already exist; a reader must be able to tell "this row
    says it was recorder-timed" from "this row says nothing", or the field launders an
    inference into a record.
    """
    path = tmp_path / "runs.jsonl"
    led = Ledger(path)
    with _recorder(led, wall_clock_s=12.5):
        pass
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["wall_clock_source"] == "caller"
    del raw["wall_clock_source"]
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    back = Ledger(path).rows()[0]
    assert back.wall_clock_source == "unrecorded"
    assert back.wall_clock_s == 12.5  # the duration is still whatever it was


def test_the_new_field_does_not_break_a_chain_written_without_it(tmp_path: Path):
    """Chain verification hashes each raw LINE (`sha256(line)`), not a re-serialisation of
    the parsed row, so a row gaining a field does not invalidate a file written before it
    existed. This asserts that rather than assuming it -- it is the reason the field was
    safe to add to an append-only ledger at all.
    """
    path = tmp_path / "runs.jsonl"
    led = Ledger(path)
    with _recorder(led, wall_clock_s=1.0):
        pass
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    del first["wall_clock_source"]
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")

    # A second row appended now carries the field and chains onto the old-format line.
    with _recorder(Ledger(path), wall_clock_s=2.0):
        pass
    rows = Ledger(path).rows()
    assert [r.wall_clock_source for r in rows] == ["unrecorded", "caller"]
    verify_no_fork([path])  # raises if the chain is broken


@pytest.mark.parametrize("bad", ["measured", "", "CALLER", "wall"])
def test_an_unknown_wall_clock_source_is_refused(bad: str):
    with pytest.raises(ValueError, match="wall_clock_source"):
        LedgerRow(
            row_id="r", written_at="w", prev_row_hash=None, protocol=_protocol(),
            run_kind="ft", status="completed", quick=False, quick_reason=None,
            code_commit="c", env=_env(), metrics={}, noul_rate=NotRun(reason="x"),
            controls={}, gates={}, wall_clock_s=1.0, cost_usd=0.0,
            wall_clock_source=bad,
        )


def test_the_refusal_happens_before_the_run_not_after_it(tmp_path: Path):
    """Validating in ``_finish`` would refuse the argument only once the GPU time had
    already been spent, and the row would be lost with it. The check belongs at
    construction.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(ValueError, match="measured"):
        _recorder(led, wall_clock_s=math.nan)
    assert led.rows() == []


# --------------------------------------------------------------------------
# Stating the duration from inside the block.
#
# The argument above forces a choice between two things a caller may want, and
# `tools/rung0_real_run.py` wanted both: it MEASURES the training loop, and it needs the
# recorder to WRAP that loop so a run killed part-way through still writes a row. Passing
# the measured figure to the constructor requires having it, which requires the training to
# be over, which puts the training outside the block -- and the guarantee in RunRecorder's
# docstring is a property of the block.
#
# `measured()` separates the two. The recorder is entered before the work, with None; the
# duration is stated when it is known. `_finish` reads `self.wall_clock_s` at exit rather
# than at construction, so a figure stated in between is indistinguishable from one passed
# in -- which is the point, because those rows have to stay comparable with the 53 written
# before this existed.
# --------------------------------------------------------------------------

def test_a_duration_stated_inside_the_block_is_recorded_as_the_callers(tmp_path: Path):
    """Both halves, because a fix that moved only the first would still pass one of them.

    The mutation this kills is `measured()` storing the figure somewhere `_finish` does not
    read: the row then silently carries the block's lifetime, and `wall_clock_source` still
    says "recorder", which is true and useless.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=None) as rec:
        time.sleep(0.05)  # "training"
        rec.measured(239.0)
        time.sleep(0.05)  # reporting, inside the block and not part of the run
    row = led.rows()[0]
    assert row.wall_clock_s == 239.0
    assert row.wall_clock_source == "caller"


def test_a_duration_stated_inside_the_block_prices_the_run(tmp_path: Path):
    """`wall` and `cost_usd` are computed on adjacent lines in `_finish` from the same
    attribute, and a change that reached one and not the other would pass the test above
    while billing a GPU hour at the cost of writing a row.

    One hour at $2.20/h is $2.20. The recorder's own lifetime here is milliseconds.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=None, cost=_rate(2.20)) as rec:
        rec.measured(3600.0)
    assert led.rows()[0].cost_usd == pytest.approx(2.20)


def test_a_duration_stated_before_a_crash_survives_it(tmp_path: Path):
    """A run that trained for an hour and then raised in its reporting still cost an hour.

    This is why `measured()` is called the moment training returns rather than at the end
    of the block: everything after that point is reporting, and a failure in it must not
    take the row's measured duration down with it.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(RuntimeError), _recorder(
        led, wall_clock_s=None, cost=_rate(2.20)
    ) as rec:
        rec.measured(3600.0)
        raise RuntimeError("evaluation OOM")
    row = led.rows()[0]
    assert row.status == "failed"
    assert row.wall_clock_s == 3600.0
    assert row.wall_clock_source == "caller"
    assert row.cost_usd == pytest.approx(2.20)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_a_stated_duration_that_is_not_a_measurement_is_refused(tmp_path: Path, bad: float):
    """The same refusal the constructor makes, at the other door.

    A refused statement is not a statement: the block goes on to exit normally, and the row
    it writes falls back and says "recorder" rather than carrying a NaN as though somebody
    had measured one.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=None) as rec, pytest.raises(
        ValueError, match="measured duration"
    ):
        rec.measured(bad)
    row = led.rows()[0]
    assert row.wall_clock_source == "recorder"
    assert math.isfinite(row.wall_clock_s)


def test_a_second_statement_is_refused(tmp_path: Path):
    """Keeping the last figure is the wrong default and the failure is invisible.

    `rung0_real_run.py` opens one recorder per seed inside a loop. A refactor that hoisted
    the recorder out of the loop would call this once per seed against one recorder, and
    "keep the last" would record the final seed's minutes as the sweep's -- a row that is
    wrong by a factor of `--seeds` and looks exactly like a correct one.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=None) as rec:
        rec.measured(120.0)
        with pytest.raises(ValueError, match="already stated"):
            rec.measured(240.0)
    assert led.rows()[0].wall_clock_s == 120.0


def test_a_statement_that_contradicts_the_constructor_is_refused(tmp_path: Path):
    """A caller that passed a measured figure in is a caller that did the work outside the
    block. Calling `measured()` as well means one of the two is about a different run.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=239.0) as rec, pytest.raises(
        ValueError, match="already stated"
    ):
        rec.measured(12.0)
    assert led.rows()[0].wall_clock_s == 239.0


def test_a_statement_after_the_row_is_written_is_refused(tmp_path: Path):
    """By then it changes nothing, and a caller that believes otherwise has a bug that a
    silent no-op would hide -- a row priced at the recorder's lifetime, with the code that
    was supposed to fix that sitting two lines below it.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, wall_clock_s=None) as rec:
        pass
    with pytest.raises(ValueError, match="after the row is written"):
        rec.measured(239.0)
    assert led.rows()[0].wall_clock_source == "recorder"
