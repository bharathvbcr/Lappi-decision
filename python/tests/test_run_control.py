"""The schedule, the cap, the price and the position -- all torch-free, all run here.

These are the parts whose bugs are silent, so they are the parts that get the suite that
actually executes on this host rather than the one reported as "not run".
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from qd_train.run_control import (
    APPROVAL_FREE_USD,
    CAP_EXIT_CODE,
    EMPTY_PREFIX_DIGEST,
    MAX_CAP_S,
    MAX_CHECKPOINT_BYTES,
    MAX_GRAD_ACCUM,
    MAX_SIDECAR_BYTES,
    MAX_SIDECAR_TENSORS,
    Checkpoint,
    ConsumedPrefix,
    CostEstimate,
    LaunchRefused,
    LossLog,
    LossPoint,
    LRSchedule,
    Position,
    RunControl,
    TensorRef,
    WallClockCap,
    _first_non_finite,
    _payload_digest,
    hard_exit_on_cap,
)

#: `python/`, so the subprocess in the crash test imports the same `qd_train` this does.
PYTHON_ROOT = Path(__file__).resolve().parents[1]


def _schedule(**over) -> LRSchedule:
    base = {"peak_lr": 1e-3, "total_steps": 100, "warmup_steps": 10, "min_lr": 1e-5}
    base.update(over)
    return LRSchedule(**base)  # type: ignore[arg-type]


def _cheap(cap: WallClockCap) -> CostEstimate:
    """A single-GPU estimate under $20, so it needs no human yes."""
    return CostEstimate(usd_per_hour=0.5, cap=cap, n_gpus=1, instance="test-1xA10")


def _noop_terminate(reason: str) -> None:
    """A terminate action that records nothing and kills nothing.

    Rule 4's fourth requirement is that a terminate action *exists* and is armed; what it
    does is the caller's. Tests that are about the other three requirements pass this so the
    real one -- which interrupts the main thread and then exits the process -- never runs
    inside pytest.
    """


class _Recorder:
    """A terminate action that remembers it was called, for the watchdog tests."""

    def __init__(self) -> None:
        self.reasons: list[str] = []
        self.fired = threading.Event()

    def __call__(self, reason: str) -> None:
        self.reasons.append(reason)
        self.fired.set()


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
        usd_per_hour=31.92,
        cap=WallClockCap(cap_s=40 * 3600.0),
        n_gpus=8,
        instance="8xH100 SXM",
        usd_per_gpu_hour=3.99,
    )
    assert cost.projected_usd == pytest.approx(1276.8)
    assert cost.cost_for(34 * 3600.0) == pytest.approx(1085.28)


def test_rule_4_multi_gpu_always_needs_a_human_yes():
    cost = CostEstimate(
        usd_per_hour=0.08,
        cap=WallClockCap(cap_s=60.0),
        n_gpus=8,
        instance="8xH100 SXM",
        usd_per_gpu_hour=0.01,
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
    cost = CostEstimate(
        usd_per_hour=31.92, cap=cap, n_gpus=8, instance="8xH100 SXM", usd_per_gpu_hour=3.99
    )
    with pytest.raises(LaunchRefused, match="human yes"):
        RunControl(schedule=_schedule(), cap=cap, cost=cost, auto_terminate=_noop_terminate)
    # With a name attached it is permitted -- the check is for a recorded yes, not a veto.
    RunControl(
        schedule=_schedule(),
        cap=cap,
        cost=cost,
        approved_by="bharath 2026-09-19",
        auto_terminate=_noop_terminate,
    )


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


# ---------------------------------------------------------------------------
# COSTCTL: the machinery between a training run and a surprise bill.
#
# Every rate below is one the human can actually rent at, so the arithmetic is
# pinned against real money rather than against invented numbers:
#
#   1x A10 24GB    $1.29/h    1x A100 40GB   $1.99/h
#   1x H100 80GB   $4.29/h    4x A6000 48GB  $4.36/h
#   8x A100 40GB  $15.92/h    8x A100 80GB  $22.32/h
#   2x H100 80GB SXM5  $8.38/h per instance, $4.19/GPU/h   <- the live target
# ---------------------------------------------------------------------------

#: (name, per-instance $/h, n_gpus, per-GPU $/h or None when n_gpus <= 1)
REAL_RATES: list[tuple[str, float, int, float | None]] = [
    ("1x A10 24GB", 1.29, 1, None),
    ("1x A100 40GB", 1.99, 1, None),
    ("1x H100 80GB", 4.29, 1, None),
    ("4x A6000 48GB", 4.36, 4, 1.09),
    ("8x A100 40GB", 15.92, 8, 1.99),
    ("8x A100 80GB", 22.32, 8, 2.79),
    ("2x H100 80GB SXM5", 8.38, 2, 4.19),
]


def _estimate(name, rate, n, per_gpu, cap):
    return CostEstimate(
        usd_per_hour=rate, cap=cap, n_gpus=n, instance=name, usd_per_gpu_hour=per_gpu
    )


def test_the_projection_is_rate_times_cap_hours_for_every_rate_that_can_be_rented():
    """Units, at a two-hour cap. Seconds/3600 -> hours, float throughout, no rounding."""
    cap = WallClockCap(cap_s=7200.0)
    expected = {
        "1x A10 24GB": 2.58,
        "1x A100 40GB": 3.98,
        "1x H100 80GB": 8.58,
        "4x A6000 48GB": 8.72,
        "8x A100 40GB": 31.84,
        "8x A100 80GB": 44.64,
        "2x H100 80GB SXM5": 16.76,
    }
    for name, rate, n, per_gpu in REAL_RATES:
        est = _estimate(name, rate, n, per_gpu, cap)
        assert est.projected_usd == pytest.approx(expected[name]), name
        # The cap priced, not the hoped-for duration: a run that ends early costs less, and
        # the estimate is still the cap's number.
        assert est.cost_for(3600.0) == pytest.approx(rate), name
        assert est.cost_for(0.0) == 0.0, name


def test_the_projection_is_correct_for_a_rate_the_code_has_never_seen():
    """There is no instance -> rate table anywhere, and there must not be one.

    A rate is a required constructor argument with no default, so an unrecognised machine is
    priced by whatever the provider charges for it rather than by a lookup that can go stale
    or miss. Two consequences worth pinning: an unknown instance cannot silently become $0,
    because the field has no default at all; and the arithmetic does not care what the
    machine is called.
    """
    cap = WallClockCap(cap_s=7200.0)
    for rate in (0.37, 1.49, 3.19, 7.77, 13.0, 999.99):
        est = CostEstimate(
            usd_per_hour=rate, cap=cap, n_gpus=1, instance="a machine nobody has priced here"
        )
        assert est.projected_usd == pytest.approx(rate * 2.0)
    with pytest.raises(TypeError):
        CostEstimate(cap=cap, n_gpus=1, instance="no rate given")  # type: ignore[call-arg]


def test_the_approval_threshold_is_a_closed_boundary_and_one_cent_either_side_of_it():
    """Rule 2: the threshold is read-only. This pins where it is, it does not move it."""
    cap = WallClockCap(cap_s=7200.0)  # $20.00 at exactly $10.00/h

    def free(rate):
        return not CostEstimate(
            usd_per_hour=rate, cap=cap, n_gpus=1, instance="boundary"
        ).requires_human_approval

    assert free(0.0), "a machine that costs nothing needs no yes"
    assert free(9.99), "$19.98 at the cap"
    assert free(9.995), "$19.99 at the cap"
    assert not free(10.0), "$20.00 at the cap is >= the threshold, not above it"
    assert not free(10.005), "$20.01 at the cap"
    # And the comparison is on the real value, not on a rounded one: $19.9998 is under.
    assert free(9.9999)
    assert CostEstimate(
        usd_per_hour=9.9999, cap=cap, n_gpus=1, instance="boundary"
    ).projected_usd == pytest.approx(19.9998)


def test_the_two_gpu_target_is_priced_from_the_instance_column_not_the_gpu_column():
    """2x H100 80GB SXM5: $8.38/h per instance, $4.19/GPU/h. They differ by exactly 2x.

    Before this check, ``CostEstimate`` had no concept of which column a rate came from:
    ``n_gpus`` was stored and used for nothing but the ``> 1`` test, and never multiplied
    into ``projected_usd``. Typing $4.19 where $8.38 belonged produced a perfectly valid
    estimate that under-reported the bill by half, and the object could not tell.
    """
    cap = WallClockCap(cap_s=7200.0)
    right = CostEstimate(
        usd_per_hour=8.38, cap=cap, n_gpus=2, instance="2x H100 80GB SXM5", usd_per_gpu_hour=4.19
    )
    assert right.projected_usd == pytest.approx(16.76)
    # $16.76 is under $20, so the arithmetic alone would clear it -- rule 4's multi-GPU
    # clause is what makes this a yes, and it is the only thing that does.
    assert right.projected_usd < APPROVAL_FREE_USD
    assert right.requires_human_approval

    with pytest.raises(ValueError, match="wrong column"):
        CostEstimate(
            usd_per_hour=4.19,
            cap=cap,
            n_gpus=2,
            instance="2x H100 80GB SXM5",
            usd_per_gpu_hour=4.19,
        )
    # ... and the per-instance rate typed into the per-GPU field is refused the same way.
    with pytest.raises(ValueError, match="wrong column"):
        CostEstimate(
            usd_per_hour=8.38,
            cap=cap,
            n_gpus=2,
            instance="2x H100 80GB SXM5",
            usd_per_gpu_hour=8.38,
        )


def test_a_multi_gpu_estimate_that_names_only_one_column_is_refused():
    cap = WallClockCap(cap_s=7200.0)
    for name, rate, n, _per_gpu in REAL_RATES:
        if n <= 1:
            continue
        with pytest.raises(ValueError, match="per-GPU column"):
            CostEstimate(usd_per_hour=rate, cap=cap, n_gpus=n, instance=name)


def test_every_real_multi_gpu_rate_agrees_across_its_two_columns():
    cap = WallClockCap(cap_s=7200.0)
    for name, rate, n, per_gpu in REAL_RATES:
        if per_gpu is None:
            continue
        est = _estimate(name, rate, n, per_gpu, cap)
        assert est.usd_per_hour == pytest.approx(per_gpu * n), name
        assert f"${per_gpu:.2f}/GPU/h x {n}" in est.approval_line(), name


def test_what_the_column_check_cannot_catch_is_a_falsified_gpu_count():
    """Named so it is not mistaken for coverage it does not have.

    The two columns agree trivially at ``n_gpus == 1``, so an estimate that *understates* the
    GPU count passes every check here. That is the one shape of this error the object cannot
    detect without a price list, and it is the shape that changes the approval decision
    rather than only the number: the real 8x A100 40GB costs $31.84 at a two-hour cap and
    needs a human yes, while its per-GPU column declared as a single GPU projects $3.98 and
    clears approval untouched.
    """
    cap = WallClockCap(cap_s=7200.0)
    truth = CostEstimate(
        usd_per_hour=15.92, cap=cap, n_gpus=8, instance="8x A100 40GB", usd_per_gpu_hour=1.99
    )
    assert truth.projected_usd == pytest.approx(31.84)
    assert truth.requires_human_approval

    understated = CostEstimate(usd_per_hour=1.99, cap=cap, n_gpus=1, instance="8x A100 40GB")
    assert understated.projected_usd == pytest.approx(3.98)
    assert not understated.requires_human_approval, (
        "this is the hole, pinned deliberately: n_gpus is taken on trust, and nothing in this "
        "module can check it against a price list"
    )
    # The window in which the 8x error flips the decision rather than only the number, at a
    # two-hour cap and a declared single GPU: a true instance rate in [$10/h, $20/h) is
    # >= $20 at the cap while any per-GPU column under $10/h reads as approval-free.
    assert CostEstimate(
        usd_per_hour=10.0, cap=cap, n_gpus=1, instance="edge"
    ).requires_human_approval
    assert not CostEstimate(
        usd_per_hour=9.99, cap=cap, n_gpus=1, instance="edge"
    ).requires_human_approval


def test_an_estimate_that_does_not_name_its_machine_cannot_be_built():
    """``instance`` used to default to ``"unspecified"``, which its own guard then accepted.

    The check rejected ``""`` and ``"  "`` with a message about rates read off the wrong row
    of a price list, and let the placeholder its own signature supplied go through: an 8-GPU
    estimate built without naming anything printed ``unspecified: 8 GPU(s) at $22.32/h``.
    """
    cap = WallClockCap(cap_s=7200.0)
    with pytest.raises(TypeError):
        CostEstimate(usd_per_hour=22.32, cap=cap, n_gpus=8)  # type: ignore[call-arg]
    for blank in ("", "  ", "\t\n"):
        with pytest.raises(ValueError, match="name the machine"):
            CostEstimate(usd_per_hour=1.0, cap=cap, n_gpus=1, instance=blank)
    with pytest.raises(ValueError, match="name the machine"):
        CostEstimate(usd_per_hour=1.0, cap=cap, n_gpus=1, instance=None)  # type: ignore[arg-type]


def test_the_approval_line_never_prints_a_number_that_contradicts_its_own_verdict():
    """A line that says ``$20.00 at the cap -- no approval required`` is a lie told in $0.0002.

    ``:.2f`` rounds to nearest, so the printed projection and the decision were computed from
    different numbers. Measured before this fix: ``$9.9999/h`` printed
    ``at $10.00/h, capped at 2.00 h -> $20.00 at the cap -- no approval required``.
    """
    cap = WallClockCap(cap_s=7200.0)
    line = CostEstimate(
        usd_per_hour=9.9999, cap=cap, n_gpus=1, instance="edge"
    ).approval_line()
    assert "no approval required" in line
    assert "$20.00 at the cap" not in line
    assert "19.9998" in line

    # A rate that is not free must never print as free.
    cheap = CostEstimate(usd_per_hour=0.004, cap=cap, n_gpus=1, instance="edge").approval_line()
    assert "$0.00/h" not in cheap
    assert "0.004" in cheap

    # A cap that is real must not print as no cap at all, for the same reason.
    short = CostEstimate(
        usd_per_hour=4.29, cap=WallClockCap(cap_s=10.0), n_gpus=1, instance="edge"
    ).approval_line()
    assert "capped at 0.00 h" not in short
    assert "0.00277" in short
    # ... and a cap that genuinely rounds to two places still reads like hours.
    assert "capped at 0.01 h" in CostEstimate(
        usd_per_hour=4.29, cap=WallClockCap(cap_s=30.0), n_gpus=1, instance="edge"
    ).approval_line()

    # And the ordinary case still reads like money.
    plain = CostEstimate(usd_per_hour=4.29, cap=cap, n_gpus=1, instance="1x H100 80GB")
    assert "at $4.29/h per instance" in plain.approval_line()
    assert "capped at 2.00 h" in plain.approval_line()
    assert "-> $8.58 at the cap -- no approval required" in plain.approval_line()


# --- the clock the cap is measured against ---------------------------------


def test_a_clock_that_steps_backwards_is_refused_rather_than_clamped():
    """``max(0.0, clock() - t0)`` hid an NTP step, a DST transition and a bad test double.

    Measured before this fix, against a clock stepped back 109 s: ``elapsed_s()`` went 9.0 ->
    0.0 -> 9.0, ``cost_so_far()`` went $0.010725 -> $0.000000, and the cap fired 109 s of
    clock-time late. The recorded cost went *down* while the meter ran, and nothing said so.
    """
    cap = WallClockCap(cap_s=10.0)
    now = [1000.0]
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), clock=lambda: now[0])
    ctl.start()
    now[0] = 1009.0
    assert ctl.elapsed_s() == pytest.approx(9.0)
    assert ctl.cost_so_far() > 0.0

    now[0] = 900.0  # NTP steps the clock back
    with pytest.raises(RuntimeError, match="clock went backwards"):
        ctl.elapsed_s()
    with pytest.raises(RuntimeError, match="clock went backwards"):
        ctl.expired()
    with pytest.raises(RuntimeError, match="clock went backwards"):
        ctl.cost_so_far()


def test_a_clock_that_never_moves_is_not_a_clock_that_went_backwards():
    """The check is on regression, not on progress: a stalled clock must still report."""
    cap = WallClockCap(cap_s=10.0)
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), clock=lambda: 5.0)
    ctl.start()
    assert [ctl.elapsed_s() for _ in range(5)] == [0.0] * 5


def test_a_clock_that_returns_a_non_finite_reading_is_refused():
    """``max(0.0, nan)`` is ``0.0``, so a NaN clock read as a run that had only just begun.

    Measured before this fix: ``elapsed_s()`` -> 0.0, ``expired()`` -> False and
    ``cost_so_far()`` -> $0.00, for as long as the clock stayed broken. That is the
    ``nan > threshold`` shape with the comparison hidden inside ``max``.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        cap = WallClockCap(cap_s=10.0)
        ctl = RunControl(
            schedule=_schedule(), cap=cap, cost=_cheap(cap), clock=lambda bad=bad: bad
        )
        with pytest.raises(ValueError, match="must be finite"):
            ctl.start()

    # And a clock that only breaks mid-run is refused at the reading, not swallowed.
    readings = iter([0.0, 1.0, float("nan")])
    cap = WallClockCap(cap_s=10.0)
    ctl = RunControl(
        schedule=_schedule(), cap=cap, cost=_cheap(cap), clock=lambda: next(readings)
    )
    ctl.start()
    assert not ctl.expired()
    with pytest.raises(ValueError, match="must be finite"):
        ctl.expired()


def test_a_negative_elapsed_is_a_broken_clock_not_a_cheap_run():
    """``cost_for(-3600.0)`` used to return ``-4.29``: a bill that earned money.

    ``trainer.py`` feeds ``control.elapsed_s()`` straight into ``cost_for`` for the ledger
    row, so a negative reading wrote a negative dollar figure into an append-only file.
    ``remaining_s(-1e9)`` answered with a billion seconds of headroom against a 10 s cap.
    """
    cap = WallClockCap(cap_s=10.0)
    est = CostEstimate(usd_per_hour=4.29, cap=cap, n_gpus=1, instance="1x H100 80GB")
    for bad in (-1e-9, -3600.0, -1e9):
        with pytest.raises(ValueError, match="non-negative"):
            est.cost_for(bad)
        with pytest.raises(ValueError, match="non-negative"):
            cap.expired(bad)
        with pytest.raises(ValueError, match="non-negative"):
            cap.remaining_s(bad)
    assert est.cost_for(0.0) == 0.0
    assert cap.remaining_s(0.0) == 10.0


# --- rule 4's fourth requirement: auto-terminate ---------------------------


def test_rule_4_refuses_a_run_that_has_no_auto_terminate():
    """Rule 4 names four things. Three were enforced; this is the fourth.

    ``expired()`` is a poll a cooperative loop makes at optimizer-step boundaries. Before
    this check, an 8-GPU run could be fully configured -- cap, estimate, recorded human yes
    -- with nothing whatsoever that stops it if the loop never asks.
    """
    cap = WallClockCap(cap_s=7200.0)
    cost = CostEstimate(
        usd_per_hour=22.32, cap=cap, n_gpus=8, instance="8x A100 80GB", usd_per_gpu_hour=2.79
    )
    with pytest.raises(LaunchRefused, match="needs auto-terminate"):
        RunControl(schedule=_schedule(), cap=cap, cost=cost, approved_by="bharath 2026-09-20")
    ctl = RunControl(
        schedule=_schedule(),
        cap=cap,
        cost=cost,
        approved_by="bharath 2026-09-20",
        auto_terminate=_noop_terminate,
    )
    assert ctl.auto_terminate is _noop_terminate
    # A run rule 4 does not gate is not forced to carry one.
    RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap))
    with pytest.raises(TypeError, match="auto_terminate must be callable"):
        RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), auto_terminate="kill it")


def test_the_watchdog_terminates_a_step_that_never_polls_the_cap():
    """The money test: a hung step, a wedged dataloader, a loop that forgot to ask.

    Nothing here calls ``expired()``. Measured before this fix, on the same shape: sleeping
    four times the cap left ``expired()`` answering True to nobody, with the process still
    running and still billing, and ``RunControl`` carrying no stop, terminate, kill or
    watchdog member at all.
    """
    cap = WallClockCap(cap_s=0.05)
    fired = _Recorder()
    ctl = RunControl(
        schedule=_schedule(),
        cap=cap,
        cost=CostEstimate(usd_per_hour=4.29, cap=cap, n_gpus=1, instance="1x H100 80GB"),
        auto_terminate=fired,
    )
    ctl.start()
    try:
        assert fired.fired.wait(timeout=10.0), "the cap passed and nothing terminated the run"
    finally:
        ctl.stop()
    assert ctl.terminated_for_cap
    assert len(fired.reasons) == 1, "a watchdog that fires twice is a watchdog with no latch"
    assert "cap reached with the loop still running" in fired.reasons[0]
    assert "1x H100 80GB" in fired.reasons[0], "the reason carries the priced line"


def test_the_watchdog_is_measured_against_the_injected_clock_not_the_wall():
    """So a test with a frozen clock arms a watchdog that never fires, as it must."""
    cap = WallClockCap(cap_s=0.05)
    fired = _Recorder()
    now = [0.0]
    ctl = RunControl(
        schedule=_schedule(),
        cap=cap,
        cost=_cheap(cap),
        clock=lambda: now[0],
        auto_terminate=fired,
    )
    ctl.start()
    try:
        assert not fired.fired.wait(timeout=0.5), "real time is not this run's clock"
        now[0] = 99.0
        assert fired.fired.wait(timeout=10.0), "the injected clock passed the cap"
    finally:
        ctl.stop()


def test_a_watchdog_that_cannot_read_the_clock_terminates_rather_than_going_quiet():
    """A check that could not run must never report what a check that ran and passed reports."""
    cap = WallClockCap(cap_s=0.05)
    fired = _Recorder()
    readings = [0.0]

    def clock():
        if len(readings) > 1:
            return float("nan")
        readings.append(1.0)
        return 0.0

    ctl = RunControl(
        schedule=_schedule(), cap=cap, cost=_cheap(cap), clock=clock, auto_terminate=fired
    )
    ctl.start()
    try:
        assert fired.fired.wait(timeout=10.0), "an unreadable clock left the cap unenforced"
    finally:
        ctl.stop()
    assert "could not be evaluated" in fired.reasons[0]


def test_stop_disarms_the_watchdog_and_is_idempotent():
    cap = WallClockCap(cap_s=0.05)
    fired = _Recorder()
    ctl = RunControl(
        schedule=_schedule(), cap=cap, cost=_cheap(cap), auto_terminate=fired
    )
    ctl.start()
    ctl.stop()
    ctl.stop()
    assert not fired.fired.wait(timeout=0.5), "a disarmed watchdog must not fire"
    assert not ctl.terminated_for_cap


def test_a_control_used_as_a_context_manager_starts_and_disarms():
    cap = WallClockCap(cap_s=0.05)
    fired = _Recorder()
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap), auto_terminate=fired)
    with ctl as entered:
        assert entered is ctl
        assert ctl.started
    assert not fired.fired.wait(timeout=0.5)


def test_a_run_with_no_auto_terminate_arms_no_thread():
    """The default path is unchanged: no terminate action, no watchdog, no behaviour added.

    Compares the set of live threads rather than the count: an earlier test's watchdog may
    still be winding down after its ``stop()``, and a count that drops would fail a test that
    is asking only whether this control *added* one.
    """
    cap = WallClockCap(cap_s=0.01)
    before = {id(t) for t in threading.enumerate()}
    ctl = RunControl(schedule=_schedule(), cap=cap, cost=_cheap(cap))
    ctl.start()
    time.sleep(0.05)
    added = [t for t in threading.enumerate() if id(t) not in before]
    assert added == [], f"a run with no terminate action armed {added}"
    assert not ctl.terminated_for_cap
    ctl.stop()  # must be safe on a control that never armed one


def test_the_default_terminate_action_escalates_and_is_bounded():
    """``hard_exit_on_cap`` is not called here -- it exits the process. Its shape is read.

    Rule 5: this is a source-level check, not a run of the ladder. What it pins is that each
    rung exists and that nothing between them is unbounded.
    """
    import inspect

    src = inspect.getsource(hard_exit_on_cap)
    assert "interrupt_main" in src, "rung 1: let a cooperative loop unwind and checkpoint"
    assert "SIGTERM" in src, "rung 2: for a main thread blocked inside a C call"
    assert "os._exit" in src, "rung 3: for a process that ignores SIGTERM"
    assert src.count("time.sleep(TERMINATE_GRACE_S)") == 2, "every wait between rungs is bounded"
    assert CAP_EXIT_CODE == 124, "the GNU timeout convention, and deliberately not 3 (NotRun)"


#: A throwaway process for the two rungs the source-level test above cannot run, because
#: running them means killing the process under test. It arms the REAL watchdog through
#: RunControl with the default terminate action and then blocks its main thread inside a C
#: call -- a read on a pipe nothing ever writes -- which is exactly the state rung 1's
#: interrupt cannot reach: `interrupt_main` schedules a handler for the next bytecode
#: boundary and sends no signal, so a thread parked in `read(2)` never sees it. The grace is
#: shortened from 30s so the test finishes in seconds; the ladder's order and the waits
#: between rungs are otherwise the shipped ones.
_LADDER_CHILD = """\
import os, signal, sys
import qd_train.run_control as rc
rc.TERMINATE_GRACE_S = 0.5
if sys.argv[1] == "ignore-sigterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
cap = rc.WallClockCap(cap_s=0.2)
ctl = rc.RunControl(
    schedule=rc.LRSchedule(peak_lr=1e-3, total_steps=10),
    cap=cap,
    cost=rc.CostEstimate(usd_per_hour=0.5, cap=cap, n_gpus=1, instance="test-1xA10"),
    auto_terminate=rc.hard_exit_on_cap,
)
ctl.start()
r, _w = os.pipe()
print("BLOCKED", flush=True)
os.read(r, 1)
print("UNREACHABLE", flush=True)
"""


def _run_ladder_child(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "ladder_child.py"
    script.write_text(_LADDER_CHILD, encoding="utf-8")
    # The command is the interpreter running this test, on a temp file it just wrote.
    return subprocess.run(
        [sys.executable, str(script), mode],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(PYTHON_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )


def test_rung_two_sigterm_ends_a_main_thread_blocked_in_a_c_call(tmp_path):
    """Rung 2, run rather than read. HANDOFF/costctl-2026-09-20.md reported rungs 2 and 3
    NOT RUN and asked for "one deliberate manual exercise against a throwaway process
    before the rental"; this is that exercise, made repeatable."""
    proc = _run_ladder_child(tmp_path, "default")
    assert "BLOCKED" in proc.stdout, f"the child never reached its blocking call: {proc}"
    assert "UNREACHABLE" not in proc.stdout, "the read returned, so nothing was blocked"
    assert "still alive after the interrupt; SIGTERM" in proc.stderr, proc.stderr
    assert proc.returncode == -signal.SIGTERM, (
        f"expected death by SIGTERM (rung 2), got returncode {proc.returncode}: "
        f"{proc.stderr}"
    )
    assert "exiting 124" not in proc.stderr, "rung 3 ran, so rung 2 did not end it"


def test_rung_three_exits_124_when_sigterm_is_ignored(tmp_path):
    """Rung 3: a process that swallows SIGTERM still stops billing, with the GNU timeout
    convention's status so a wrapper script reads it without being taught a new number."""
    proc = _run_ladder_child(tmp_path, "ignore-sigterm")
    assert "BLOCKED" in proc.stdout, f"the child never reached its blocking call: {proc}"
    assert "UNREACHABLE" not in proc.stdout
    assert "still alive after SIGTERM; exiting 124 without unwinding" in proc.stderr, (
        proc.stderr
    )
    assert proc.returncode == CAP_EXIT_CODE == 124, (proc.returncode, proc.stderr)


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


def _consumed(n: int) -> str:
    """The digest of ``n`` folded batches, so a fixture's position and evidence agree."""
    prefix = ConsumedPrefix()
    for i in range(n):
        prefix.fold(i.to_bytes(8, "big"))
    return prefix.hexdigest()


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
        "consumed_digest": _consumed(8),
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


# --- what a checkpoint refuses to be ------------------------------------------------------


def test_state_a_checkpoint_could_not_write_is_refused_where_it_is_built():
    """The rented-hour failure: `write()` used to be the first thing that ever looked.

    Measured against the pre-fix code with torch 2.12.1: `Rung0Step.state()` returned live
    tensors, `Checkpoint(...)` accepted them, `to_json()` accepted them, and `write()` died
    with `TypeError: Object of type Tensor is not JSON serializable`. A run discovers that
    at its first checkpoint boundary and again at its last.
    """
    with pytest.raises(TypeError, match="which this checkpoint cannot write"):
        _checkpoint(model_state={"w": object()})
    # Nested, because a state dict is a tree and the top level is the easy case.
    with pytest.raises(TypeError, match=r"model_state\['opt'\]\['exp_avg'\]"):
        _checkpoint(model_state={"opt": {"exp_avg": complex(1, 2)}})
    with pytest.raises(TypeError, match="JSON object keys must be str"):
        _checkpoint(model_state={"opt": {0: [1.0]}})


def test_a_diverged_parameter_is_refused_rather_than_written_as_a_bare_nan():
    """`json.dumps` writes NaN as a bare token that is not JSON and that strict readers refuse."""
    with pytest.raises(ValueError, match="which JSON cannot represent"):
        _checkpoint(model_state={"w": [1.0, float("nan")]})


def test_a_checkpoint_whose_position_and_evidence_disagree_is_refused():
    """Two statements of one fact. A digest that says "ate nothing" at index 8 is a lie."""
    with pytest.raises(ValueError, match="describe different runs"):
        _checkpoint(consumed_digest=EMPTY_PREFIX_DIGEST)
    with pytest.raises(ValueError, match="describe different runs"):
        _checkpoint(position=Position(epoch=0, index=0), optimizer_step=0, loss_log=LossLog())


def test_the_consumed_prefix_is_order_sensitive_and_cannot_be_split_to_collide():
    a, b = ConsumedPrefix(), ConsumedPrefix()
    a.fold(b"x")
    a.fold(b"y")
    b.fold(b"y")
    b.fold(b"x")
    assert a.hexdigest() != b.hexdigest(), "order must matter or a reordered epoch passes"
    one, two = ConsumedPrefix(), ConsumedPrefix()
    one.fold(b"ab")
    two.fold(b"a", b"b")
    assert one.hexdigest() != two.hexdigest(), "parts are length-prefixed"
    assert ConsumedPrefix().hexdigest() == EMPTY_PREFIX_DIGEST
    assert a.n_folded == 2


# --- the file, and the crash -------------------------------------------------------------


def test_editing_the_weights_in_a_written_checkpoint_is_refused_on_read(tmp_path):
    """`loss_digest` covered the trajectory and left the parameters unguarded.

    Pre-fix this returned a Checkpoint carrying `[9.0, 9.0]` without a word: the file's
    only checksum was over the loss log, so every other field -- the weights, the position,
    the schedule, the seed -- loaded unverified.
    """
    path = _checkpoint().write(tmp_path / "ckpt.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model_state"]["w"] = [9.0, 9.0]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="payload_digest"):
        Checkpoint.read(path)


def test_a_checkpoint_with_its_checksum_removed_is_refused_rather_than_skipped(tmp_path):
    """An absent checksum is not a passed checksum.

    Pre-fix, `from_json` read the digest with `raw.get("loss_digest")` and skipped the
    comparison when it was `None` -- so deleting one key turned the only integrity check
    in the file off, and the read succeeded.
    """
    path = _checkpoint().write(tmp_path / "ckpt.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["payload_digest"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="no payload_digest"):
        Checkpoint.read(path)

    raw = json.loads(_checkpoint().write(tmp_path / "b.json").read_text(encoding="utf-8"))
    del raw["loss_digest"]
    raw["payload_digest"] = _payload_digest(
        {k: v for k, v in raw.items() if k != "payload_digest"}
    )
    (tmp_path / "b.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="no loss_digest"):
        Checkpoint.read(tmp_path / "b.json")


def test_a_checkpoint_too_large_to_be_a_checkpoint_is_refused(tmp_path, monkeypatch):
    """Bounded like every other payload here, and with a measured number behind the bound.

    2,080,749 bytes for 44,736 parameters -- 46.5 bytes each -- measured on 2026-09-20 with
    `Rung0Step` on torch 2.12.1. The 2B backbone would be 65.2 GB of decimal text per
    checkpoint.

    The number has not moved and what it bounds has: the weights now live in the sidecar,
    so this bounds the position, the schedule, the loss log and the per-tensor digests. That
    is strictly less than it used to admit, so nothing was widened to let anything through
    (rule 2). `MAX_SIDECAR_BYTES` is the *new* bound on the *new* payload, and it is pinned
    separately below -- a payload that could not be written at all before is not a payload
    whose bound was relaxed.
    """
    assert MAX_CHECKPOINT_BYTES == 1 << 30, "the shipped bound is still a real bound"
    monkeypatch.setattr("qd_train.run_control.MAX_CHECKPOINT_BYTES", 2_000)
    big = _checkpoint(model_state={"w": [float(i) for i in range(1_000)]})
    with pytest.raises(ValueError, match="over MAX_CHECKPOINT_BYTES"):
        big.write(tmp_path / "too-big.json")
    assert not list(tmp_path.iterdir()), "a refused write leaves nothing behind"


def test_the_sidecar_bound_is_derived_from_the_model_it_has_to_hold(monkeypatch, tmp_path):
    """`MAX_SIDECAR_BYTES` is a new bound on a new payload, and its number has a derivation.

    Rule 2 forbids moving a gate to make something pass. This is not that: before the
    sidecar existed a model state of this size could not be written *at all*, so there is no
    older, tighter bound that was loosened. What the number must be is a question with an
    answer in this repository rather than a guess, and this pins the arithmetic so the two
    cannot drift apart in silence.
    """
    from qd_train.memory import ADAMW_FP32, QWEN3_5_2B_TEXT

    widest = QWEN3_5_2B_TEXT.trainable_params() * (
        4 + ADAMW_FP32.states_per_param * ADAMW_FP32.state_bytes
    )
    assert widest == 22_581_901_056, "the derivation in the docstring is this number"
    assert MAX_SIDECAR_BYTES == 32 << 30
    assert widest < MAX_SIDECAR_BYTES, (
        "the bound must admit the largest checkpoint this repo can construct -- the full "
        f"text tower with fp32 weights and two fp32 moments, {widest:,} bytes"
    )
    assert 2 * widest > MAX_SIDECAR_BYTES, (
        "a bound with more than 2x headroom over the widest real checkpoint is not bounding "
        "anything"
    )

    # And it is enforced, not merely declared.
    monkeypatch.setattr("qd_train.run_control.MAX_SIDECAR_BYTES", 8)
    ckpt = _checkpoint(model_state={"w": TensorRef(dtype="float32", shape=(4,), data=bytes(16))})
    with pytest.raises(ValueError, match="over MAX_SIDECAR_BYTES"):
        ckpt.write(tmp_path / "wide.json")
    assert not list(tmp_path.iterdir()), "a refused write leaves neither file behind"


def test_a_write_killed_part_way_leaves_the_previous_checkpoint_readable(tmp_path):
    """The rented-hour case, with a real interrupted write rather than a story about one.

    The child sets `RLIMIT_FSIZE` below the size of the checkpoint it is about to write, so
    the write dies at a byte offset the kernel picks and the process exits non-zero. That
    is a kill mid-write without the timing race a signal would need.

    Pre-fix, `write_text` truncated the target and wrote into it, so the file left behind
    was neither checkpoint: `Checkpoint.read` raised `JSONDecodeError` and the previous
    hour's work was gone. Post-fix the bytes land in a temporary file and only a rename
    ever touches the target.
    """
    target = tmp_path / "ckpt.json"
    first = _checkpoint()
    first.write(target)
    before = target.read_text(encoding="utf-8")

    big = _checkpoint(model_state={"w": [float(i) for i in range(20_000)]})
    limit = len(json.dumps(big.to_json())) // 2
    script = tmp_path / "child.py"
    script.write_text(
        "import pickle, resource, signal, sys\n"
        "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({limit}, {limit}))\n"
        "ckpt, target = pickle.load(open(sys.argv[1], 'rb'))\n"
        "ckpt.write(target)\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.pkl"
    with payload.open("wb") as fh:
        pickle.dump((big, target), fh)

    # The command is the interpreter running this test, on a temp file it just wrote.
    proc = subprocess.run(
        [sys.executable, str(script), str(payload)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(PYTHON_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert proc.returncode != 0, (
        "the write completed inside the file-size limit, so nothing was interrupted and "
        f"this test proved nothing. stdout={proc.stdout!r}"
    )

    assert target.read_text(encoding="utf-8") == before, "the previous checkpoint was damaged"
    assert Checkpoint.read(target).consumed_digest == first.consumed_digest
    assert not list(tmp_path.glob("ckpt.json.tmp-*")), "the failed write left no debris"


# ---------------------------------------------------------------------------
# The sidecar: the half of a checkpoint JSON could not hold
#
# Measured 2026-09-20: JSON costs 46.5 bytes per parameter, because `t.tolist()` renders
# every float as decimal text. For this program's 1,401,501,504 trainable parameters that
# is 65.2 GB per checkpoint against 2.80 GB for the same tensors as bf16 -- a factor of 23,
# and a rented disk full before a two-hour run ends.
#
# Moving the weights out creates one new failure mode that did not exist with one file: a
# fresh metadata file pairing with a stale sidecar, which reads as a valid checkpoint and
# is not. Most of what follows is about that.
# ---------------------------------------------------------------------------


def _bytes_for(dtype: str, values: list[int]) -> bytes:
    """Raw little-endian bytes from raw bit patterns, with no float conversion anywhere."""
    size = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "uint8": 1, "int64": 8}
    return b"".join(v.to_bytes(size[dtype], "little") for v in values)


def test_the_non_finite_scan_is_exact_over_every_sixteen_bit_pattern():
    """The refusal that would have been retired in silence, reimplemented against bytes.

    `_refuse_unwritable` refuses a non-finite float in the JSON body for two reasons: JSON
    writes it as a bare `NaN`/`Infinity` token that is not JSON, and a non-finite parameter
    is a run that has already diverged. Only the first stops applying in a sidecar, where
    the bit pattern round-trips exactly. Dropping the check with the format would retire a
    refusal without saying so.

    The oracle here is independent of the implementation: this masks the whole 16-bit value
    against the format's exponent field; `_first_non_finite` slices the high bytes out of
    the buffer and runs `bytes.translate` over them. Exhaustive, so "it works on the values
    I thought of" is not the claim.
    """
    for dtype, exp_mask in (("float16", 0x7C00), ("bfloat16", 0x7F80)):
        wrong = []
        for bits in range(1 << 16):
            expected = 0 if (bits & exp_mask) == exp_mask else None
            got = _first_non_finite(_bytes_for(dtype, [bits]), dtype)
            if got != expected:
                wrong.append((hex(bits), got, expected))
        assert not wrong, f"{dtype}: {len(wrong)} of 65,536 patterns misread, e.g. {wrong[:4]}"

    # And it finds the *first* one, in a buffer where most elements are fine.
    ok, nan = 0x3F80, 0x7FC1
    data = _bytes_for("bfloat16", [ok] * 500 + [nan] + [ok] * 500)
    assert _first_non_finite(data, "bfloat16") == 500
    # A dtype with no non-finite value has nothing to find, rather than a false positive.
    assert _first_non_finite(b"\xff" * 64, "uint8") is None
    assert _first_non_finite(b"\xff" * 64, "int64") is None


def test_a_tensor_that_does_not_match_its_own_shape_is_refused():
    """A length that disagrees with the shape is a truncated tensor, and reviving it reshapes."""
    with pytest.raises(ValueError, match="is 16 bytes and this one carries 12"):
        TensorRef(dtype="float32", shape=(4,), data=bytes(12))
    with pytest.raises(ValueError, match="not a dtype this checkpoint can store"):
        TensorRef(dtype="float8_e4m3fn", shape=(1,), data=bytes(1))
    with pytest.raises(ValueError, match="non-negative ints"):
        TensorRef(dtype="float32", shape=(-1,), data=b"")
    # A zero-element tensor is a real tensor, not an error.
    assert TensorRef(dtype="float32", shape=(0,), data=b"").nbytes == 0


def test_a_diverged_parameter_is_refused_in_the_sidecar_too():
    """The bound the format change must not quietly drop."""
    nan = _bytes_for("bfloat16", [0x3F80, 0x7FC0, 0x3F80])
    with pytest.raises(ValueError, match=r"element 1 .* is not finite"):
        TensorRef(dtype="bfloat16", shape=(3,), data=nan)
    inf = _bytes_for("float32", [0x3F800000, 0xFF800000])
    with pytest.raises(ValueError, match=r"element 1 .* is not finite"):
        TensorRef(dtype="float32", shape=(2,), data=inf)
    with pytest.raises(ValueError, match="already diverged"):
        _checkpoint(model_state={"w": TensorRef(dtype="bfloat16", shape=(3,), data=nan)})


def test_a_checkpoint_with_no_tensors_still_writes_exactly_one_file(tmp_path):
    """The single-file path is unchanged: everything the RESUME lane pinned still holds."""
    ckpt = _checkpoint()
    path = ckpt.write(tmp_path / "plain.json")
    assert [f.name for f in tmp_path.iterdir()] == ["plain.json"]
    assert json.loads(path.read_text(encoding="utf-8"))["sidecar"] is None
    assert Checkpoint.read(path).model_state == ckpt.model_state


def test_a_body_that_declares_a_sidecar_and_is_handed_none_is_refused():
    """An absent sidecar is not an empty one -- the rule an absent digest already follows."""
    ckpt = _checkpoint(model_state={"w": TensorRef(dtype="float32", shape=(2,), data=bytes(8))})
    raw = ckpt.to_json()
    assert raw["sidecar"]["n_tensors"] == 1
    assert raw["sidecar"]["nbytes"] == 8
    with pytest.raises(ValueError, match="none was supplied"):
        Checkpoint.from_json(raw)
    # And the field itself cannot be deleted to turn the question off.
    body = {k: v for k, v in raw.items() if k not in ("payload_digest", "sidecar")}
    body["payload_digest"] = _payload_digest(body)
    with pytest.raises(ValueError, match="no `sidecar` field"):
        Checkpoint.from_json(body)


def test_a_sidecar_from_another_checkpoint_cannot_pair_with_this_metadata():
    """The failure mode two files have and one file does not: a stale pair that reads valid."""
    mine = TensorRef(dtype="float32", shape=(2,), data=_bytes_for("float32", [0x3F800000] * 2))
    theirs = TensorRef(dtype="float32", shape=(2,), data=_bytes_for("float32", [0x40000000] * 2))
    raw = _checkpoint(model_state={"w": mine}).to_json()
    key = "model_state['w']"
    with pytest.raises(ValueError, match="not a matched pair"):
        Checkpoint.from_json(raw, tensors={key: theirs})
    # Right count, right names, right shapes -- only the bytes differ. That is exactly the
    # case a manifest without digests would have accepted.
    assert theirs.shape == mine.shape and theirs.dtype == mine.dtype
    # A sidecar for a checkpoint that declares none is equally refused.
    plain = _checkpoint().to_json()
    with pytest.raises(ValueError, match="sidecar nobody wrote"):
        Checkpoint.from_json(plain, tensors={key: mine})
    # And the round trip works when they do match.
    assert Checkpoint.from_json(raw, tensors={key: mine}).model_state["w"] == mine


def test_the_payload_digest_reaches_into_the_sidecar_manifest():
    """`payload_digest` covered the whole body; the weights left the body, so it has to follow.

    Editing the recorded digest of a tensor is how you would make a swapped sidecar pass the
    pair check. It does not, because the manifest is inside the body the payload digest
    covers -- the same guarantee, reaching one file further.
    """
    ref = TensorRef(dtype="float32", shape=(2,), data=bytes(8))
    raw = _checkpoint(model_state={"w": ref}).to_json()
    raw["model_state"]["w"]["__tensor_ref__"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="modified after it was written"):
        Checkpoint.from_json(raw, tensors={"model_state['w']": ref})

    # Rebuild the payload digest over the edited body, as a real tamper would, and the
    # per-tensor comparison is what catches it.
    body = {k: v for k, v in raw.items() if k != "payload_digest"}
    body["payload_digest"] = _payload_digest(
        {k: v for k, v in body.items() if k != "payload_digest"}
    )
    with pytest.raises(ValueError, match="hashes to"):
        Checkpoint.from_json(body, tensors={"model_state['w']": ref})


def test_the_sidecars_name_carries_its_digest_so_a_rewrite_cannot_clobber_the_old_one():
    """The naming decision is the atomicity argument, so it is pinned rather than described.

    A fixed sidecar name has the defect this whole design exists to avoid: rewriting a
    checkpoint at one path replaces the tensors first, and a kill before the metadata lands
    leaves the old metadata pointing at the new tensors. With the digest in the name,
    different content is a different file.
    """
    a = TensorRef(dtype="float32", shape=(2,), data=_bytes_for("float32", [0x3F800000] * 2))
    b = TensorRef(dtype="float32", shape=(2,), data=_bytes_for("float32", [0x40000000] * 2))
    da = _checkpoint(model_state={"w": a}).to_json()["sidecar"]["digest"]
    db = _checkpoint(model_state={"w": b}).to_json()["sidecar"]["digest"]
    assert da != db
    slot = Path("/run/latest.json")
    assert Checkpoint.sidecar_path(slot, da) != Checkpoint.sidecar_path(slot, db)
    assert Checkpoint.sidecar_path(slot, da).parent == slot.parent
    assert Checkpoint.sidecar_path(slot, da).name.endswith(".safetensors")
    # Pure function of the two arguments, so write and read agree without storing a path.
    assert Checkpoint.sidecar_path(slot, da) == Checkpoint.sidecar_path("/run/latest.json", da)
    # The tensor's position in the tree is part of the set digest, so the same bytes under a
    # different name are a different sidecar -- a state dict cannot be rearranged silently.
    assert _checkpoint(model_state={"x": a}).to_json()["sidecar"]["digest"] != da
    # A digest that is not a digest cannot become a filename.
    with pytest.raises(ValueError, match="64 lowercase hex"):
        Checkpoint.sidecar_path(slot, "../../etc/passwd")


def test_a_checkpoint_with_more_tensors_than_the_bound_is_refused(tmp_path, monkeypatch):
    """Every fan-out here is bounded, including this one."""
    assert MAX_SIDECAR_TENSORS == 1 << 20
    monkeypatch.setattr("qd_train.run_control.MAX_SIDECAR_TENSORS", 2)
    state = {
        f"w{i}": TensorRef(dtype="float32", shape=(1,), data=_bytes_for("float32", [i]))
        for i in range(3)
    }
    with pytest.raises(ValueError, match="over MAX_SIDECAR_TENSORS"):
        _checkpoint(model_state=state).write(tmp_path / "many.json")
    assert not list(tmp_path.iterdir())


def test_everything_the_resume_lane_pinned_survives_the_format_change():
    """A sidecar that dropped one of these would reintroduce exactly what was fixed hours ago."""
    ref = TensorRef(dtype="float32", shape=(2,), data=bytes(8))
    # consumed_digest: still required, still checked against the position.
    with pytest.raises(ValueError, match="describe different runs"):
        _checkpoint(model_state={"w": ref}, consumed_digest=EMPTY_PREFIX_DIGEST)
    with pytest.raises(TypeError, match="consumed_digest"):
        _checkpoint(model_state={"w": ref}, consumed_digest=None)
    # the schedule and the loss log are still checked against the step.
    with pytest.raises(ValueError, match="different runs"):
        _checkpoint(model_state={"w": ref}, optimizer_step=101)
    with pytest.raises(ValueError, match="replay a logged step"):
        _checkpoint(model_state={"w": ref}, optimizer_step=3)
    # a live object is still refused where it is built, not at write().
    with pytest.raises(TypeError, match="which this checkpoint cannot write"):
        _checkpoint(model_state={"w": object()})
    # an absent checksum is still not a passed one.
    raw = _checkpoint(model_state={"w": ref}).to_json()
    del raw["payload_digest"]
    with pytest.raises(ValueError, match="no payload_digest"):
        Checkpoint.from_json(raw, tensors={"model_state['w']": ref})


# --- the pair, on a real filesystem -------------------------------------------------------
#
# `safetensors` is in stack/train.lock and in the training venv; the torch-free venv
# deliberately has neither it nor torch. These skip there, one skip per test rather than one
# per module, so the suite's denominator still says how much was not run.


def _needs_safetensors():
    return pytest.importorskip("safetensors", reason="sidecar I/O needs safetensors")


def _f32(values: list[float]) -> TensorRef:
    return TensorRef(
        dtype="float32",
        shape=(len(values),),
        data=b"".join(struct.pack("<f", v) for v in values),
    )


def test_a_checkpoint_with_tensors_round_trips_through_two_files(tmp_path):
    """The whole point: the bytes go to the sidecar and come back identical."""
    _needs_safetensors()
    state = {"model": {"w": _f32([1.5, -2.5, 3.0])}, "optimizer": {"step": 4}}
    ckpt = _checkpoint(model_state=state)
    path = ckpt.write(tmp_path / "nested" / "step-4.json")
    written = sorted(f.name for f in path.parent.iterdir())
    assert len(written) == 2 and written[1] == "step-4.json"
    assert written[0].startswith("step-4.") and written[0].endswith(".safetensors")

    back = Checkpoint.read(path)
    assert back.model_state["model"]["w"] == state["model"]["w"]
    assert back.model_state["model"]["w"].data == state["model"]["w"].data
    assert back.model_state["optimizer"]["step"] == 4
    assert back.consumed_digest == ckpt.consumed_digest
    assert back.loss_log.digest() == ckpt.loss_log.digest()

    body = path.read_text(encoding="utf-8")
    assert "cuda" not in body and "mps" not in body and "device" not in body
    assert str(tmp_path) not in body, "a checkpoint that stores an absolute path cannot move"


def test_the_sidecar_is_exact_for_every_bfloat16_bit_pattern(tmp_path):
    """bf16 through a real file, measured rather than asserted.

    `tolist()` goes through a Python float. For *finite* bf16 that round trip happens to be
    exact -- 8 mantissa bits into 52 and back -- so the old path was not lossy, it was
    merely 23x the size and unable to represent the non-finite patterns at all. The raw-byte
    path is exact for all 65,536 by construction, and this is the measurement of that claim
    rather than a restatement of it.
    """
    _needs_safetensors()
    finite = [b for b in range(1 << 16) if (b & 0x7F80) != 0x7F80]
    assert len(finite) == 65_280, "256 of the 65,536 bf16 patterns are NaN or infinity"
    ref = TensorRef(
        dtype="bfloat16",
        shape=(len(finite),),
        data=b"".join(b.to_bytes(2, "little") for b in finite),
    )
    path = _checkpoint(model_state={"w": ref}).write(tmp_path / "bf16.json")
    back = Checkpoint.read(path).model_state["w"]
    assert back.data == ref.data, "a bf16 bit pattern changed on the way through the file"
    assert back.dtype == "bfloat16" and back.shape == (len(finite),)


def test_an_interrupted_successor_leaves_the_whole_previous_pair_readable(tmp_path):
    """The RESUME lane's property, extended across two files.

    A real interrupted write on 2026-09-20 left 84,731 bytes of a half-written successor
    over a 582-byte predecessor and **neither** survived. The fix made one file atomic. Two
    files are two chances to have half a checkpoint, and the order is what closes it: the
    sidecar is written first and names nothing, the JSON is written last and is the commit.

    This drives the exact interruption that matters -- the successor's tensors are on disk
    and its metadata is not -- and asks whether the predecessor is still a checkpoint.
    """
    _needs_safetensors()
    slot = tmp_path / "run.json"
    first = _checkpoint(model_state={"w": _f32([1.0, 1.0])})
    first.write(slot)
    first_json = slot.read_text(encoding="utf-8")
    first_side = Checkpoint.sidecar_path(slot, json.loads(first_json)["sidecar"]["digest"])

    second = _checkpoint(model_state={"w": _f32([2.0, 2.0])})
    second_side = Checkpoint.sidecar_path(slot, second.to_json()["sidecar"]["digest"])
    Checkpoint._write_sidecar(second_side, second.tensors())

    assert second_side.exists(), "the interruption must be after the sidecar lands"
    assert second_side != first_side, "a fixed sidecar name would have clobbered the old one"
    assert slot.read_text(encoding="utf-8") == first_json
    assert first_side.exists(), "the predecessor's tensors were destroyed by its successor"
    recovered = Checkpoint.read(slot)
    assert recovered.model_state["w"].data == first.model_state["w"].data
    assert recovered.consumed_digest == first.consumed_digest


def test_a_stale_sidecar_under_the_right_name_is_still_refused(tmp_path):
    """The name is a convenience; the digest is the check. Both are needed and only one decides."""
    _needs_safetensors()
    slot = tmp_path / "latest.json"
    _checkpoint(model_state={"w": _f32([1.0, 1.0])}).write(slot)
    live = Checkpoint.sidecar_path(slot, json.loads(slot.read_text())["sidecar"]["digest"])
    # Forge a sidecar with the right *name* and the wrong *content*.
    impostor = _checkpoint(model_state={"w": _f32([9.0, 9.0])})
    tmp_side = tmp_path / "impostor.safetensors"
    Checkpoint._write_sidecar(tmp_side, impostor.tensors())
    live.write_bytes(tmp_side.read_bytes())
    with pytest.raises(ValueError, match="not a matched pair"):
        Checkpoint.read(slot)


def test_one_edited_byte_in_the_sidecar_is_refused_on_read(tmp_path):
    """`payload_digest` protected the weights while they were in the body. They left the body."""
    _needs_safetensors()
    path = _checkpoint(model_state={"w": _f32([1.0, 2.0, 3.0])}).write(tmp_path / "c.json")
    side = Checkpoint.sidecar_path(path, json.loads(path.read_text())["sidecar"]["digest"])
    blob = bytearray(side.read_bytes())
    blob[-1] ^= 0x01
    side.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="hashes to"):
        Checkpoint.read(path)


def test_a_checkpoint_whose_sidecar_is_gone_is_refused_by_name(tmp_path):
    """Half a checkpoint is not a checkpoint with fewer tensors."""
    _needs_safetensors()
    path = _checkpoint(model_state={"w": _f32([1.0])}).write(tmp_path / "c.json")
    side = Checkpoint.sidecar_path(path, json.loads(path.read_text())["sidecar"]["digest"])
    side.unlink()
    with pytest.raises(FileNotFoundError, match=side.name):
        Checkpoint.read(path)


def test_rewriting_a_slot_retires_the_previous_sidecar_and_only_after_it_commits(tmp_path):
    """Retention, bounded: a slot rewritten N times holds one sidecar, not N.

    The deletion is the one in this module, it is derived from the predecessor's own record
    rather than from a glob, and it happens strictly after the successor's JSON is durable.
    A crash during it leaves a stale sidecar -- wasted disk -- and never a live checkpoint
    missing its tensors.
    """
    _needs_safetensors()
    slot = tmp_path / "latest.json"
    names = []
    for i in range(4):
        _checkpoint(model_state={"w": _f32([float(i), float(i)])}).write(slot)
        digest = json.loads(slot.read_text())["sidecar"]["digest"]
        names.append(Checkpoint.sidecar_path(slot, digest))
        sidecars = sorted(f.name for f in tmp_path.glob("*.safetensors"))
        assert len(sidecars) == 1, f"after {i + 1} writes the directory holds {sidecars}"
    assert len({n.name for n in names}) == 4, "each write had different content, so a new name"
    assert Checkpoint.read(slot).model_state["w"] == _f32([3.0, 3.0])

    # A second slot in the same directory is untouched by the first slot's retirement.
    other = tmp_path / "keep.json"
    _checkpoint(model_state={"w": _f32([7.0, 7.0])}).write(other)
    _checkpoint(model_state={"w": _f32([8.0, 8.0])}).write(slot)
    assert Checkpoint.read(other).model_state["w"] == _f32([7.0, 7.0])


def test_the_sidecar_is_what_makes_this_model_checkpointable_at_all(tmp_path):
    """The measurement the lane exists for, taken here rather than quoted from a handoff.

    **The values matter, and that is the finding.** A tensor of 0.5, 1.5, 2.5 renders as
    `"0.5"` and costs under 6 bytes a number; a trained weight renders as
    `"-0.8201345205307007"` and costs 20, because `tolist()` widens a float32 to a float64
    whose shortest round-tripping decimal is 17 significant figures. So this uses values
    with full float32 mantissas rather than convenient ones -- the compact case measures
    nothing and would pass whatever the sidecar cost.

    Measured on the real rung-0 step with AdamW moments present, 2026-09-20: **68.9 bytes
    per model parameter as JSON against 12.1 as a JSON+sidecar pair, a ratio of 5.7**. The
    bound below is loose enough not to be a flake and tight enough to fail if the sidecar
    ever stops earning itself.
    """
    _needs_safetensors()
    n = 20_000
    # Full-mantissa values: a linear congruential walk through the float32 bit space,
    # skipping the non-finite exponent, so every number needs its 17 digits.
    bits, values = 12345, []
    while len(values) < n:
        bits = (bits * 1103515245 + 12345) & 0xFFFFFFFF
        v = struct.unpack("<f", struct.pack("<I", bits))[0]
        if math.isfinite(v):
            values.append(v)
    ref = TensorRef(
        dtype="float32", shape=(n,), data=b"".join(struct.pack("<f", v) for v in values)
    )
    p = _checkpoint(model_state={"w": ref}).write(tmp_path / "side.json")
    side = Checkpoint.sidecar_path(p, json.loads(p.read_text())["sidecar"]["digest"])
    with_sidecar = p.stat().st_size + side.stat().st_size
    as_json = len(
        json.dumps(_checkpoint(model_state={"w": values}).to_json(), sort_keys=True).encode()
    )

    assert as_json / n > 15.0, f"JSON is {as_json / n:.1f} bytes per fp32 parameter"
    assert with_sidecar / n < 4.5, f"the pair is {with_sidecar / n:.2f} bytes per parameter"
    assert as_json > 4 * with_sidecar, (
        f"JSON {as_json:,} bytes vs sidecar pair {with_sidecar:,} bytes for {n:,} fp32 "
        "numbers -- if this ratio ever falls to 1 the sidecar has stopped earning itself"
    )
    # The JSON body is the part that is now independent of the model's size.
    assert p.stat().st_size < 0.02 * as_json, (
        f"the metadata is {p.stat().st_size:,} bytes; it must not scale with the weights"
    )


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


# -- pricing the machine the run is actually on ------------------------------------------
#
# GAP-EVERY-RUN-PRICED-ITSELF-AT-ZERO-ON-ZERO-GPUS. Four tools reached CostEstimate through
# the same literal -- `usd_per_hour=0.0, n_gpus=0, instance=f"local-{device}"` -- on every
# device including a rented GH200. `_control`'s own docstring said what should happen
# instead ("a rented machine sets a real rate here") and nothing enforced it.


def test_the_zero_defaults_turn_rule_4_off_for_any_cap() -> None:
    """The arithmetic, pinned, because it is the whole finding.

    ``requires_human_approval`` is ``n_gpus > 1 or projected_usd >= APPROVAL_FREE_USD``.
    At ``(0, 0.0)`` the first disjunct is False and the second is ``0.0 * cap >= 20``, which
    is False at **every** cap this program admits -- including MAX_CAP_S, the 40-hour ceiling
    an 8xH100 block run would use. So the human-yes gate could not fire on any run, and the
    per-GPU column check, gated on ``if self.n_gpus > 1``, could not run either.
    """
    for cap_s in (1.0, 3600.0, MAX_CAP_S):
        zeroed = CostEstimate(
            cap=WallClockCap(cap_s=cap_s), usd_per_hour=0.0, n_gpus=0, instance="local-cuda"
        )
        assert zeroed.projected_usd == 0.0
        assert not zeroed.requires_human_approval, (
            f"at cap {cap_s}s the zero defaults still ask for approval, so this test is "
            "no longer describing the defect it was written for"
        )


def test_a_cuda_device_is_not_priced_at_zero_on_the_callers_behalf() -> None:
    """A cuda device is hardware rented by the hour. The refusal names all three missing
    values at once rather than one per attempt -- an operator on a rented box should learn
    the whole requirement from one message."""
    with pytest.raises(ValueError) as excinfo:
        CostEstimate.for_device(cap=WallClockCap(cap_s=600.0), device="cuda")
    message = str(excinfo.value)
    for name in ("instance", "usd_per_hour", "n_gpus"):
        assert name in message, f"the refusal does not name {name}"
    assert "requires_human_approval" in message, (
        "the refusal should say WHY the default is not harmless, not merely that it is "
        "missing: the two values that look like defaults are the two that turn rule 4 off"
    )


def test_a_local_device_is_priced_at_zero_because_that_is_true() -> None:
    """Not a loophole. A Mac that is already bought costs nothing marginal, and refusing to
    price it would push every local run into inventing a rate -- which is how a fabricated
    number gets into a ledger that is append-only."""
    local = CostEstimate.for_device(cap=WallClockCap(cap_s=600.0), device="mps")
    assert local.usd_per_hour == 0.0
    assert local.n_gpus == 0
    assert local.instance == "local-mps"
    assert not local.requires_human_approval


def test_a_priced_multi_gpu_run_finally_reaches_the_approval_gate() -> None:
    """The gate rule 4 exists for, reachable for the first time. Eight GPUs is
    ``n_gpus > 1``, so approval is required whatever the cap and whatever the rate."""
    cost = CostEstimate.for_device(
        cap=WallClockCap(cap_s=MAX_CAP_S),
        device="cuda",
        n_gpus=8,
        usd_per_hour=23.92,
        usd_per_gpu_hour=2.99,
        instance="lambda-8xH100",
    )
    assert cost.requires_human_approval
    assert cost.projected_usd == pytest.approx(23.92 * 40.0)
    assert "NEEDS A HUMAN YES" in cost.approval_line()


def test_the_per_gpu_column_check_runs_once_a_real_gpu_count_is_given() -> None:
    """DESIGN-4's own error -- a rate read off the per-GPU column instead of the per-node
    one -- is caught only above one GPU. At ``n_gpus=0`` it was unreachable, so the check
    that exists for an 8xH100 launch had never applied to one."""
    with pytest.raises(ValueError, match="came off the wrong column"):
        CostEstimate.for_device(
            cap=WallClockCap(cap_s=3600.0),
            device="cuda",
            n_gpus=8,
            usd_per_hour=2.99,  # the per-GPU figure in the per-instance slot
            usd_per_gpu_hour=2.99,
            instance="lambda-8xH100",
        )


def test_a_single_gpu_run_under_the_threshold_still_needs_no_approval() -> None:
    """Rule 4's other half, unchanged: "Single-GPU jobs under $20 do not." Pricing the
    machine honestly must not turn every rented single-GPU probe into an approval request,
    or the requirement gets routed around."""
    cost = CostEstimate.for_device(
        cap=WallClockCap(cap_s=3600.0),
        device="cuda",
        n_gpus=1,
        usd_per_hour=1.49,
        instance="lambda-1xGH200",
    )
    assert cost.projected_usd == pytest.approx(1.49)
    assert cost.projected_usd < APPROVAL_FREE_USD
    assert not cost.requires_human_approval
