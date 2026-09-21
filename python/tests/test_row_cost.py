"""A row's ``cost_usd`` comes from the estimate that gated the run, or the run is refused.

``a409895`` made ``wall_clock_s`` the run's duration. ``cost_usd`` is derived from it, and
was still wrong for a second, independent reason: ``RunRecorder`` took a bare
``cost_usd_per_hour: float = 0.0``, and no runner but ``tools/real_ft_run.py`` passed one.
So every rung 0 row from the rented GH200 recorded $0.00 for a real GPU hour.

``c74f4ba`` fixed the other half -- ``CostEstimate.for_device`` now refuses to invent a rate
for cuda -- but that object gates the *launch*. The *row* still read a separate float that
defaulted to zero. Two quantities, one name, one layer apart:
``python/qd_train/trainer.py`` already computed ``control.cost.cost_for(wall)`` correctly
and put it in ``TrainResult`` rather than the row, so the right number existed and was
discarded.

The rule this file pins: a run on hardware that is paid for by the hour cannot record a row
without saying what it cost. ``CostEstimate.for_device`` prices ``cpu`` and ``mps`` at zero,
so a local run says zero with an object rather than by omission -- which is the difference
between a zero somebody measured and a zero nobody wrote down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder
from qd_train.run_control import CostEstimate, WallClockCap
from qd_train.tristate import NotRun

REPO = Path(__file__).resolve().parents[2]

#: An hour, so a rate and a cost are the same number and an arithmetic slip is visible.
ONE_HOUR_S = 3600.0
GH200_RATE = 1.49


def _protocol(seed: int = 0) -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64,
        tokenizer_hash="t" * 64,
        backbone_commit="b" * 40,
        recipe_hash="r" * 64,
        seed=seed,
    )


def _env(device: str = "mps") -> Environment:
    return Environment(
        torch="2.12.1", transformers_sha="abc123", device=device, host="test",
        fla_present=NotRun(reason="no CUDA on this host"),
        causal_conv1d_present=NotRun(reason="no CUDA on this host"),
    )


def _rented(rate: float = GH200_RATE) -> CostEstimate:
    return CostEstimate(
        cap=WallClockCap(cap_s=ONE_HOUR_S),
        usd_per_hour=rate,
        n_gpus=1,
        instance="lambda-1xGH200",
    )


def _recorder(led: Ledger, **kw: object) -> RunRecorder:
    kw.setdefault("env", _env())
    return RunRecorder(
        led, protocol=_protocol(), run_kind="ft", repo=REPO, **kw  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# The cost the row records is the cost the estimate says.
# --------------------------------------------------------------------------

def test_the_row_costs_what_the_estimate_says(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, env=_env("cuda"), wall_clock_s=ONE_HOUR_S, cost=_rented()):
        pass
    assert led.rows()[0].cost_usd == pytest.approx(GH200_RATE)


def test_cost_is_taken_from_the_measured_duration_not_the_recorder_lifetime(tmp_path: Path):
    """The mutation this kills: pricing the recorder's own block.

    A recorder entered after the work lives for microseconds, so a cost derived from its
    lifetime is ~0 for any rate. Half an hour at $1.49/h is $0.745; the wrong answer here
    is not a near miss, it is three orders of magnitude.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, env=_env("cuda"), wall_clock_s=ONE_HOUR_S / 2, cost=_rented()):
        pass
    assert led.rows()[0].cost_usd == pytest.approx(GH200_RATE / 2)


def test_a_failed_run_still_cost_what_it_cost(tmp_path: Path):
    """A run that OOMs after 40 minutes was billed for 40 minutes. ``_finish`` has three
    callers and a fix applied to the completed path alone prices every crash at zero."""
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(RuntimeError), _recorder(
        led, env=_env("cuda"), wall_clock_s=ONE_HOUR_S, cost=_rented()
    ):
        raise RuntimeError("CUDA out of memory")
    row = led.rows()[0]
    assert row.status == "failed"
    assert row.cost_usd == pytest.approx(GH200_RATE)


# --------------------------------------------------------------------------
# A rented device cannot record a row that says nothing about cost.
# --------------------------------------------------------------------------

def test_a_rented_device_without_an_estimate_is_refused(tmp_path: Path):
    """The GH200 case, exactly: 13 rung 0 rows recorded cost_usd 0.0 for real GPU hours.

    ``cost=None`` is how a local run says "nothing is billed here". On a device that is
    paid for by the hour it is an omission, and an omission that reads as $0.00 is the
    defect this file exists for.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(ValueError, match="cost"):
        _recorder(led, env=_env("cuda"), wall_clock_s=1.0, cost=None)
    assert led.rows() == []


@pytest.mark.parametrize("device", sorted(CostEstimate.LOCAL_DEVICES))
def test_a_local_device_may_state_no_cost(tmp_path: Path, device: str):
    """cpu and mps are the machine the work is already being done on."""
    led = Ledger(tmp_path / "runs.jsonl")
    with _recorder(led, env=_env(device), wall_clock_s=ONE_HOUR_S, cost=None):
        pass
    assert led.rows()[0].cost_usd == 0.0


def test_a_local_run_may_also_state_a_zero_estimate(tmp_path: Path):
    """``for_device`` prices a local device at zero and returns an object, which is a zero
    somebody wrote down rather than one nobody did. Both spellings are allowed; only the
    rented-and-silent combination is not."""
    led = Ledger(tmp_path / "runs.jsonl")
    local = CostEstimate.for_device(cap=WallClockCap(cap_s=ONE_HOUR_S), device="cpu")
    with _recorder(led, env=_env("cpu"), wall_clock_s=ONE_HOUR_S, cost=local):
        pass
    assert led.rows()[0].cost_usd == 0.0


def test_the_refusal_happens_before_the_run_not_after_it(tmp_path: Path):
    """Refusing in ``_finish`` would refuse once the GPU hour had already been spent, and
    would take the row down with it -- leaving no record of a run that really happened."""
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(ValueError, match="cost"):
        _recorder(led, env=_env("cuda"), wall_clock_s=1.0, cost=None)
    assert not (tmp_path / "runs.jsonl").exists() or led.rows() == []


def test_cost_is_required_and_has_no_default(tmp_path: Path):
    """The whole defect was a default. Omitting it is a TypeError, not a zero."""
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(TypeError, match="cost"):
        RunRecorder(  # type: ignore[call-arg]
            led, protocol=_protocol(), run_kind="ft", repo=REPO, env=_env(),
            wall_clock_s=1.0,
        )
