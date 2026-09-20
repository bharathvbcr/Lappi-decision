"""The schedule, the cap, the price and the position -- all torch-free, all run here.

These are the parts whose bugs are silent, so they are the parts that get the suite that
actually executes on this host rather than the one reported as "not run".
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from qd_train.run_control import (
    APPROVAL_FREE_USD,
    MAX_CAP_S,
    MAX_GRAD_ACCUM,
    Checkpoint,
    CostEstimate,
    LaunchRefused,
    LossLog,
    LossPoint,
    LRSchedule,
    Position,
    RunControl,
    WallClockCap,
)


def _schedule(**over) -> LRSchedule:
    base = {"peak_lr": 1e-3, "total_steps": 100, "warmup_steps": 10, "min_lr": 1e-5}
    base.update(over)
    return LRSchedule(**base)  # type: ignore[arg-type]


def _cheap(cap: WallClockCap) -> CostEstimate:
    """A single-GPU estimate under $20, so it needs no human yes."""
    return CostEstimate(usd_per_hour=0.5, cap=cap, n_gpus=1, instance="test-1xA10")


# --- the schedule -------------------------------------------------------------------------


def test_warmup_is_linear_and_reaches_the_peak_on_its_last_step():
    s = _schedule(peak_lr=1.0, warmup_steps=4, total_steps=100, min_lr=0.0)
    assert [s.lr_at(i) for i in range(4)] == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cosine_decays_from_the_peak_towards_min_lr():
    s = _schedule(peak_lr=1.0, warmup_steps=1, total_steps=101, min_lr=0.1)
    rates = [s.lr_at(i) for i in range(1, 101)]
    assert rates[0] == pytest.approx(1.0)
    assert all(a >= b for a, b in itertools.pairwise(rates)), "cosine must not rise"
    # The last step is one step short of progress==1, so it approaches min_lr without reaching it.
    assert 0.1 < rates[-1] < 0.11


def test_the_schedule_is_a_pure_function_of_the_step():
    s = _schedule()
    assert [s.lr_at(i) for i in range(50)] == [s.lr_at(i) for i in range(50)]


def test_asking_past_the_end_of_the_schedule_raises_rather_than_clamping():
    s = _schedule(total_steps=10, warmup_steps=1)
    assert s.lr_at(9) > 0
    with pytest.raises(ValueError, match="past the end"):
        s.lr_at(10)


def test_a_negative_step_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        _schedule().lr_at(-1)


def test_a_warmup_longer_than_the_run_is_refused():
    with pytest.raises(ValueError, match="end before the warmup"):
        _schedule(warmup_steps=100, total_steps=100)


def test_a_floor_above_the_peak_is_refused():
    with pytest.raises(ValueError, match="decay upwards"):
        _schedule(peak_lr=1e-4, min_lr=1e-3)


def test_a_non_finite_peak_is_refused():
    with pytest.raises(ValueError, match="finite"):
        _schedule(peak_lr=float("inf"))


def test_the_schedule_round_trips_through_json():
    s = _schedule()
    assert LRSchedule.from_json(json.loads(json.dumps(s.to_json()))) == s


# --- the cap ------------------------------------------------------------------------------


def test_a_cap_must_be_positive_and_finite():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            WallClockCap(cap_s=bad)


def test_a_cap_above_the_programs_own_cap_is_refused_as_read_only():
    WallClockCap(cap_s=MAX_CAP_S)  # the boundary itself is allowed
    with pytest.raises(ValueError, match="Rule 2: a kill criterion is read-only"):
        WallClockCap(cap_s=MAX_CAP_S + 1.0)


def test_the_cap_expires_at_the_boundary_not_after_it():
    cap = WallClockCap(cap_s=10.0)
    assert not cap.expired(9.999)
    assert cap.expired(10.0)
    assert cap.remaining_s(4.0) == pytest.approx(6.0)
    assert cap.remaining_s(99.0) == 0.0


# --- the price ----------------------------------------------------------------------------


def test_the_estimate_prices_the_cap_not_the_hoped_for_duration():
    # DESIGN-4: 40 h x $31.92 is the number that belongs on the line, not 34 h x $31.92.
    cost = CostEstimate(
        usd_per_hour=31.92, cap=WallClockCap(cap_s=40 * 3600.0), n_gpus=8, instance="8xH100 SXM"
    )
    assert cost.projected_usd == pytest.approx(1276.8)
    assert cost.cost_for(34 * 3600.0) == pytest.approx(1085.28)


def test_rule_4_multi_gpu_always_needs_a_human_yes():
    cost = CostEstimate(
        usd_per_hour=0.01, cap=WallClockCap(cap_s=60.0), n_gpus=8, instance="8xH100 SXM"
    )
    assert cost.projected_usd < APPROVAL_FREE_USD
    assert cost.requires_human_approval, "8 GPUs needs a yes however cheap the run is"


def test_rule_4_a_single_gpu_job_needs_a_yes_only_once_it_reaches_twenty_dollars():
    cap = WallClockCap(cap_s=3600.0)
    assert not CostEstimate(
        usd_per_hour=APPROVAL_FREE_USD - 0.01, cap=cap, n_gpus=1, instance="1xA10"
    ).requires_human_approval
    assert CostEstimate(
        usd_per_hour=APPROVAL_FREE_USD, cap=cap, n_gpus=1, instance="1xA10"
    ).requires_human_approval


def test_an_unnamed_instance_is_refused_because_a_rate_needs_a_row_to_check_against():
    with pytest.raises(ValueError, match="name the machine"):
        CostEstimate(usd_per_hour=1.0, cap=WallClockCap(cap_s=60.0), n_gpus=1, instance="  ")


# --- run control --------------------------------------------------------------------------


def test_a_run_needing_approval_refuses_to_be_configured_without_one():
    cap = WallClockCap(cap_s=3600.0)
    cost = CostEstimate(usd_per_hour=31.92, cap=cap, n_gpus=8, instance="8xH100 SXM")
    with pytest.raises(LaunchRefused, match="human yes"):
        RunControl(schedule=_schedule(), cap=cap, cost=cost)
    # With a name attached it is permitted -- the check is for a recorded yes, not a veto.
    RunControl(schedule=_schedule(), cap=cap, cost=cost, approved_by="bharath 2026-09-19")


def test_an_estimate_that_prices_a_different_cap_is_refused():
    cap = WallClockCap(cap_s=60.0)
    other = CostEstimate(usd_per_hour=1.0, cap=WallClockCap(cap_s=600.0), n_gpus=1, instance="x")
    with pytest.raises(ValueError, match="not this run's worst case"):
        RunControl(schedule=_schedule(), cap=cap, cost=other)


def test_elapsed_before_start_raises_rather_than_reporting_zero():
    cap = WallClockCap(cap_s=60.0)
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap))
    with pytest.raises(RuntimeError, match="before start"):
        ctl.elapsed_s()


def test_one_control_drives_one_run():
    cap = WallClockCap(cap_s=60.0)
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap))
    ctl.start()
    with pytest.raises(RuntimeError, match="called twice"):
        ctl.start()


def test_the_cap_is_measured_against_the_injected_clock():
    cap = WallClockCap(cap_s=10.0)
    now = [100.0]
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), clock=lambda: now[0])
    ctl.start()
    assert not ctl.expired()
    now[0] = 109.0
    assert not ctl.expired()
    now[0] = 110.0
    assert ctl.expired()
    assert ctl.cost_so_far() == pytest.approx(0.5 * 10.0 / 3600.0)


def test_grad_accum_is_bounded():
    cap = WallClockCap(cap_s=60.0)
    for bad in (0, -1, MAX_GRAD_ACCUM + 1):
        with pytest.raises(ValueError, match="grad_accum"):
            RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), grad_accum=bad)


def test_max_micro_batches_bounds_an_endless_source():
    cap = WallClockCap(cap_s=60.0)
    ctl = RunControl(
        schedule=_schedule(total_steps=100), cap=cap, cost=_cheap(cap), grad_accum=4
    )
    assert ctl.max_micro_batches == 400


def test_checkpoint_every_zero_means_only_at_the_end():
    cap = WallClockCap(cap_s=60.0)
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), checkpoint_every=0)
    assert not any(ctl.should_checkpoint(i) for i in range(1, 50))
    ctl3 = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), checkpoint_every=3)
    assert [i for i in range(1, 10) if ctl3.should_checkpoint(i)] == [3, 6, 9]


# --- position and loss log ------------------------------------------------------------------


def test_a_position_is_two_non_negative_integers():
    assert Position(epoch=0, index=0).to_json() == {"epoch": 0, "index": 0}
    with pytest.raises(ValueError, match="non-negative"):
        Position(epoch=0, index=-1)
    with pytest.raises(TypeError):
        Position(epoch="0", index=0)  # type: ignore[arg-type]


def test_a_non_finite_or_negative_loss_is_refused_at_the_point_it_is_recorded():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            LossPoint(optimizer_step=0, epoch=0, batch_index=0, loss=bad)
    with pytest.raises(ValueError, match="masking or normalisation bug"):
        LossPoint(optimizer_step=0, epoch=0, batch_index=0, loss=-0.5)


def test_the_loss_log_refuses_a_replayed_or_rewound_step():
    log = LossLog()
    log.append(LossPoint(optimizer_step=0, epoch=0, batch_index=0, loss=1.0))
    log.append(LossPoint(optimizer_step=1, epoch=0, batch_index=1, loss=0.9))
    for bad in (0, 1):
        with pytest.raises(ValueError, match="strictly increasing"):
            log.append(LossPoint(optimizer_step=bad, epoch=0, batch_index=2, loss=0.8))


def test_the_digest_separates_losses_one_ulp_apart():
    """float.hex is exact; a digest over %g or repr-with-precision would collapse these."""
    a, b = 0.1, float(np.nextafter(0.1, 1.0))
    assert a != b
    assert f"{a:.6g}" == f"{b:.6g}", "the two are indistinguishable at printing precision"
    log_a = LossLog([LossPoint(optimizer_step=0, epoch=0, batch_index=0, loss=a)])
    log_b = LossLog([LossPoint(optimizer_step=0, epoch=0, batch_index=0, loss=b)])
    assert log_a.digest() != log_b.digest()
    assert log_a != log_b


def test_a_loss_log_round_trips_through_json_bit_exactly():
    log = LossLog(
        [
            LossPoint(optimizer_step=i, epoch=0, batch_index=i, loss=1.0 / (i + 3))
            for i in range(20)
        ]
    )
    back = LossLog.from_json(json.loads(json.dumps(log.to_json())))
    assert back == log
    assert back.digest() == log.digest()
    assert back.losses() == log.losses()


def test_a_snapshot_does_not_follow_the_log_it_came_from():
    log = LossLog([LossPoint(optimizer_step=0, epoch=0, batch_index=0, loss=1.0)])
    snap = log.snapshot()
    log.append(LossPoint(optimizer_step=1, epoch=0, batch_index=1, loss=0.5))
    assert len(snap) == 1 and len(log) == 2


# --- the checkpoint ---------------------------------------------------------------------


def _checkpoint(**over) -> Checkpoint:
    base = {
        "position": Position(epoch=0, index=8),
        "optimizer_step": 4,
        "seed": 7,
        "schedule": _schedule(),
        "loss_log": LossLog(
            LossPoint(optimizer_step=i, epoch=0, batch_index=i, loss=1.0 - i / 10)
            for i in range(4)
        ),
        "model_state": {"w": [1.0, 2.0]},
    }
    base.update(over)
    return Checkpoint(**base)  # type: ignore[arg-type]


def test_a_checkpoint_past_its_own_schedule_is_refused():
    with pytest.raises(ValueError, match="different runs"):
        _checkpoint(optimizer_step=101)


def test_a_checkpoint_that_would_replay_a_logged_step_is_refused():
    with pytest.raises(ValueError, match="replay a logged step"):
        _checkpoint(optimizer_step=3)


def test_a_checkpoint_round_trips_through_json_with_its_digest_intact():
    ckpt = _checkpoint()
    raw = json.loads(json.dumps(ckpt.to_json()))
    back = Checkpoint.from_json(raw)
    assert back.loss_log == ckpt.loss_log
    assert back.loss_log.digest() == ckpt.loss_log.digest()
    assert back.position == ckpt.position
    assert back.schedule == ckpt.schedule
    assert back.model_state == ckpt.model_state


def test_a_tampered_checkpoint_is_refused_on_read():
    raw = _checkpoint().to_json()
    raw["loss_log"][0]["loss_hex"] = 9.0.hex()
    with pytest.raises(ValueError, match="modified after it was written"):
        Checkpoint.from_json(raw)


def test_a_checkpoint_round_trips_through_a_file(tmp_path):
    ckpt = _checkpoint()
    path = ckpt.write(tmp_path / "nested" / "ckpt.json")
    assert path.exists()
    assert Checkpoint.read(path).loss_log.digest() == ckpt.loss_log.digest()


def test_the_schedule_carried_by_a_checkpoint_recomputes_the_same_rates():
    ckpt = _checkpoint()
    assert ckpt.schedule.lr_at(ckpt.optimizer_step) == _schedule().lr_at(4)
    assert math.isfinite(ckpt.schedule.lr_at(ckpt.optimizer_step))


# ---------------------------------------------------------------------------
# AccumulationGroup: the four rules, one owner
# ---------------------------------------------------------------------------


def _group(*, grad_accum: int = 3, cap_s: float = 600.0, clock=None):
    from qd_train.run_control import AccumulationGroup

    cap = WallClockCap(cap_s=cap_s)
    kwargs = dict(
        schedule=LRSchedule(peak_lr=1e-3, total_steps=10),
        cap=cap,
        cost=CostEstimate(usd_per_hour=0.0, cap=cap, n_gpus=1, instance="test"),
        grad_accum=grad_accum,
    )
    if clock is not None:
        kwargs["clock"] = clock
    control = RunControl(**kwargs)
    control.start()
    return AccumulationGroup(control, violation=ValueError), control


class _Boom(Exception):
    pass


def test_a_group_is_not_ready_until_grad_accum_micro_batches():
    group, _ = _group(grad_accum=3)
    for i in range(2):
        group.add(1.0, where=f"b{i}")
        assert not group.ready
        assert group.open
    group.add(1.0, where="b2")
    assert group.ready


def test_rule_1_a_partial_group_is_refused_when_the_source_ends():
    """A step on fewer micro-batches is a different effective batch size."""
    group, _ = _group(grad_accum=3)
    group.add(1.0, where="b0")
    with pytest.raises(ValueError, match="effective batch size"):
        group.refuse_partial(where="end of epoch")


def test_rule_1_a_closed_group_is_not_refused():
    group, _ = _group(grad_accum=1)
    group.add(1.0, where="b0")
    group.drain()
    group.refuse_partial(where="end of epoch")  # must not raise


def test_rule_2_the_cap_cannot_stop_the_loop_mid_group():
    """Stopping mid-group discards a half-accumulated gradient with nothing saying so."""
    ticks = iter([0.0] + [1000.0] * 20)
    last = [0.0]

    def clock():
        last[0] = next(ticks, last[0])
        return last[0]

    group, _ = _group(grad_accum=3, cap_s=1.0, clock=clock)
    assert group.should_stop_for_cap(), "an empty group is a boundary; the cap applies"
    group.add(1.0, where="b0")
    assert not group.should_stop_for_cap(), "the cap must wait for the group to close"
    group.add(1.0, where="b1")
    group.add(1.0, where="b2")
    group.drain()
    assert group.should_stop_for_cap()


def test_rule_3_a_non_finite_loss_stops_the_run():
    for bad in (float("nan"), float("inf"), float("-inf")):
        group, _ = _group()
        with pytest.raises(ValueError, match="non-finite loss"):
            group.add(bad, where="b0")


def test_rule_3_covers_the_extras_too():
    group, _ = _group()
    with pytest.raises(ValueError, match="non-finite span loss"):
        group.add(1.0, where="b0", span=float("nan"))


def test_rule_4_the_group_loss_is_the_mean_over_its_own_micro_batches():
    group, _ = _group(grad_accum=4)
    for value in (1.0, 2.0, 3.0, 4.0):
        group.add(value, where="b", half=value / 2)
    mean, extras = group.drain()
    assert mean == pytest.approx(2.5)
    assert extras["half"] == pytest.approx(1.25)
    assert not group.open, "drain resets the group"


def test_rule_4_an_empty_group_cannot_be_drained():
    """0.0 would enter the loss log as a measurement of a step never taken."""
    group, _ = _group()
    with pytest.raises(ValueError, match="never taken"):
        group.drain()


def test_the_caller_supplies_the_exception_type_it_raises():
    """`TrainerContractViolation` is pinned by test_trainer.py; the rules are still shared."""
    from qd_train.run_control import AccumulationGroup

    cap = WallClockCap(cap_s=60.0)
    control = RunControl(
        schedule=LRSchedule(peak_lr=1e-3, total_steps=10),
        cap=cap,
        cost=CostEstimate(usd_per_hour=0.0, cap=cap, n_gpus=1, instance="test"),
    )
    group = AccumulationGroup(control, violation=_Boom)
    with pytest.raises(_Boom):
        group.add(float("nan"), where="b0")


def test_a_group_refuses_something_that_is_not_a_run_control():
    from qd_train.run_control import AccumulationGroup

    with pytest.raises(TypeError, match="must be a RunControl"):
        AccumulationGroup(object())  # type: ignore[arg-type]
