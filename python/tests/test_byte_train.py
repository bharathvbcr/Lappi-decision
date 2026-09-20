"""Rung 0's training step and loop.

Torch-gated: this module ``importorskip``s, so in the repo venv it reports as **skipped**
rather than passing vacuously. The index arithmetic it depends on is tested torch-free in
``test_byte_batch.py``; what is asserted here is the part that needs real parameters --
that one encode feeds both heads, that the loss actually falls, and that the loop can say
why it stopped.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

from data_fixtures import mutate_row as row  # noqa: E402
from qd_train.byte_batch import MAX_PADDING_WASTE, plan_batch  # noqa: E402
from qd_train.byte_context import ID_PAD  # noqa: E402
from qd_train.byte_decider import ByteDeciderConfig  # noqa: E402
from qd_train.byte_train import (  # noqa: E402
    MAX_BATCHES_PER_CALL,
    Rung0ContractViolation,
    Rung0Model,
    Rung0Step,
    plan_to_tensors,
    train_rung0,
)
from qd_train.mutate_adapter import CLEAN, parse_example, to_decision  # noqa: E402
from qd_train.run_control import (  # noqa: E402
    CostEstimate,
    LRSchedule,
    RunControl,
    WallClockCap,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

WIDE = dict(pad_id=ID_PAD, max_context_bytes=512, max_option_bytes=96)
TINY = ByteDeciderConfig(width=32, n_layers=1, n_heads=2, max_context_bytes=512)


def decision(**over):
    return to_decision(parse_example(row(**over)), max_context_bytes=512)


def clean(**over):
    return decision(id=over.pop("id", "c"), **{"class": CLEAN}, span=None, **over)


def batch(n: int = 2):
    """A ragged batch: one mutated row and one clean (abstaining) row."""
    ds = [decision(id=f"m{i}", after="aa\nbb\ncc") for i in range(n - 1)]
    ds.append(clean(id="c0", after="xx\nyy"))
    return plan_batch(ds, **WIDE)


def control(*, total_steps: int = 4, cap_s: float = 600.0, grad_accum: int = 1, clock=None):
    schedule = LRSchedule(peak_lr=1e-2, warmup_steps=1, total_steps=total_steps)
    cap = WallClockCap(cap_s=cap_s)
    kwargs = dict(
        schedule=schedule,
        cap=cap,
        cost=CostEstimate(
            cap=cap, usd_per_hour=0.0, n_gpus=1, instance="local-mps"
        ),
        grad_accum=grad_accum,
    )
    if clock is not None:
        kwargs["clock"] = clock
    return RunControl(**kwargs)


def model_and_step(**over):
    model = Rung0Model(TINY)
    return model, Rung0Step(model, **over)


# ---------------------------------------------------------------------------
# One encode, two heads
# ---------------------------------------------------------------------------


def test_both_heads_receive_gradient_from_one_step():
    """If the span head were detached, its loss would fall out of the graph silently."""
    model, step = model_and_step()
    step.accumulate(batch())

    decider_grads = [
        p.grad for p in model.decider.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    span_grads = [
        p.grad
        for p in model.span_head.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert decider_grads, "the byte decider received no gradient"
    assert span_grads, "the span pointer head received no gradient"


def test_the_context_is_encoded_once_for_both_heads():
    """Two encodes would diverge under dropout and the span head would read unseen states."""
    model, _ = model_and_step()
    calls = []
    original = model.decider.encode_context

    def counting(context_ids, context_mask):
        calls.append(context_ids.shape)
        return original(context_ids, context_mask)

    model.decider.encode_context = counting
    model.losses(batch())
    assert len(calls) == 1, f"encode_context ran {len(calls)} times, not once"


def test_both_losses_are_carried_not_only_the_total():
    """A falling total with a flat span term is a model that learned the class only."""
    _, step = model_and_step()
    losses = step.accumulate(batch())
    assert losses.choice > 0.0 and losses.span > 0.0
    assert losses.total == pytest.approx(losses.choice + step.span_weight * losses.span)


def test_the_span_weight_actually_weights_the_span_term():
    _, heavy = model_and_step(span_weight=3.0)
    losses = heavy.accumulate(batch())
    assert losses.total == pytest.approx(losses.choice + 3.0 * losses.span)


# ---------------------------------------------------------------------------
# It learns
# ---------------------------------------------------------------------------


def test_the_loss_falls_on_a_batch_the_model_sees_repeatedly():
    """The end-to-end check: real parameters, real optimizer, a loss that actually moves."""
    torch.manual_seed(0)
    _, step = model_and_step()
    plan = batch()

    first = step.accumulate(plan)
    step.apply(lr=1e-2)
    for _ in range(30):
        step.accumulate(plan)
        step.apply(lr=1e-2)
    last = step.accumulate(plan)

    assert last.total < first.total, f"loss rose: {first.total} -> {last.total}"
    assert last.choice < first.choice
    assert last.span < first.span


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_zero_span_weight_is_refused():
    """Zero would train nothing on spans while still logging a span loss."""
    model = Rung0Model(TINY)
    with pytest.raises(ValueError, match="span_weight must be positive"):
        Rung0Step(model, span_weight=0.0)


def test_a_non_positive_lr_is_refused():
    _, step = model_and_step()
    step.accumulate(batch())
    with pytest.raises(ValueError, match="lr must be positive"):
        step.apply(lr=0.0)


def test_resuming_with_a_different_span_weight_is_refused():
    """The objective would change mid-run and the loss curve would not say so."""
    _, step = model_and_step(span_weight=1.0)
    state = step.state()
    _, other = model_and_step(span_weight=2.0)
    with pytest.raises(ValueError, match="would change the objective"):
        other.load_state(state)


def test_a_checkpoint_round_trips_the_model():
    torch.manual_seed(0)
    _, step = model_and_step()
    plan = batch()
    step.accumulate(plan)
    step.apply(lr=1e-2)
    before = step.accumulate(plan).total

    _, restored = model_and_step()
    restored.load_state(step.state())
    after = restored.accumulate(plan).total
    assert after == pytest.approx(before, rel=1e-5)


def test_an_empty_plan_cannot_reach_the_model():
    plan = batch()
    # `batch_size` reads `example_ids`, so that is the field an empty batch empties.
    empty = type(plan)(
        **{**{f: getattr(plan, f) for f in plan.__slots__}, "example_ids": (), "context_ids": ()}
    )
    assert empty.batch_size == 0
    with pytest.raises(ValueError, match="nothing to train on"):
        plan_to_tensors(empty)


def test_the_loop_refuses_something_that_is_not_a_rung0_step():
    with pytest.raises(TypeError, match="must be a Rung0Step"):
        train_rung0([batch()], step=object(), control=control())


@pytest.mark.parametrize("bad", [0, -1, MAX_BATCHES_PER_CALL + 1])
def test_an_out_of_range_max_batches_is_refused(bad: int):
    _, step = model_and_step()
    with pytest.raises(ValueError, match="max_batches must be in"):
        train_rung0([batch()], step=step, control=control(), max_batches=bad)


# ---------------------------------------------------------------------------
# Why it stopped
# ---------------------------------------------------------------------------


def test_running_out_of_data_says_data_exhausted():
    _, step = model_and_step()
    result = train_rung0([batch(), batch()], step=step, control=control(total_steps=10))
    assert result.termination == "data_exhausted"
    assert result.optimizer_steps == 2
    assert result.micro_batches == 2


def test_running_out_of_schedule_says_steps_exhausted():
    _, step = model_and_step()
    plans = [batch() for _ in range(10)]
    result = train_rung0(plans, step=step, control=control(total_steps=3))
    assert result.termination == "steps_exhausted"
    assert result.optimizer_steps == 3


def test_the_wall_clock_cap_stops_the_loop_and_says_so():
    """The cap is read from RunControl against an injected clock, never re-decided here."""
    ticks = iter([0.0] + [100.0] * 50)
    last = [0.0]

    def clock():
        last[0] = next(ticks, last[0])
        return last[0]

    _, step = model_and_step()
    result = train_rung0(
        [batch() for _ in range(10)],
        step=step,
        control=control(total_steps=10, cap_s=1.0, clock=clock),
    )
    assert result.termination == "wall_clock_cap"
    assert result.optimizer_steps == 0, "the cap is checked before the micro-batch, not after"


def test_gradient_accumulation_takes_one_step_per_group():
    _, step = model_and_step()
    plans = [batch() for _ in range(6)]
    result = train_rung0(plans, step=step, control=control(total_steps=10, grad_accum=3))
    assert result.micro_batches == 6
    assert result.optimizer_steps == 2


def test_a_source_that_ends_mid_group_is_refused_not_silently_truncated():
    """Rule 1, via the shared AccumulationGroup. Dropping the tail loses work silently."""
    _, step = model_and_step()
    plans = [batch() for _ in range(4)]
    with pytest.raises(Rung0ContractViolation, match="effective batch size"):
        train_rung0(plans, step=step, control=control(total_steps=10, grad_accum=3))


def test_exhausting_max_batches_is_refused_rather_than_called_steps_exhausted():
    """A truncated run must not be indistinguishable from a complete one."""
    _, step = model_and_step()
    plans = [batch() for _ in range(10)]
    with pytest.raises(Rung0ContractViolation, match="not the one this run planned for"):
        train_rung0(plans, step=step, control=control(total_steps=10), max_batches=2)


def test_a_non_finite_loss_stops_the_run():
    """Rule 3. Continuing trains on NaN and still prints numbers."""

    class NanStep(Rung0Step):
        def accumulate(self, plan):
            real = super().accumulate(plan)
            return type(real)(choice=real.choice, span=real.span, total=float("nan"))

    model = Rung0Model(TINY)
    with pytest.raises(Rung0ContractViolation, match="non-finite loss"):
        train_rung0([batch()], step=NanStep(model), control=control(total_steps=5))


def test_the_cap_does_not_discard_a_half_accumulated_gradient():
    """Rule 2: the cap is read at a group boundary, so overshoot is at most one step."""
    ticks = iter([0.0, 0.0, 0.0] + [1000.0] * 40)
    last = [0.0]

    def clock():
        last[0] = next(ticks, last[0])
        return last[0]

    _, step = model_and_step()
    plans = [batch() for _ in range(9)]
    result = train_rung0(
        plans, step=step, control=control(total_steps=10, cap_s=1.0, grad_accum=3, clock=clock)
    )
    assert result.termination == "wall_clock_cap"
    assert result.micro_batches % 3 == 0, (
        f"stopped {result.micro_batches} micro-batches in, which is mid-group: a "
        "half-accumulated gradient was computed and thrown away"
    )


# ---------------------------------------------------------------------------
# What it recorded
# ---------------------------------------------------------------------------


def test_the_loss_log_and_the_two_slot_logs_stay_aligned():
    _, step = model_and_step()
    result = train_rung0([batch() for _ in range(3)], step=step, control=control(total_steps=5))
    assert len(result.loss_log) == len(result.choice_log) == len(result.span_log) == 3
    triples = zip(result.loss_log, result.choice_log, result.span_log, strict=True)
    for point, choice, span in triples:
        assert point.loss == pytest.approx(choice + step.span_weight * span)


def test_the_checkpoint_carries_the_position_the_run_reached():
    _, step = model_and_step()
    result = train_rung0([batch() for _ in range(3)], step=step, control=control(total_steps=5))
    assert result.checkpoint.position.index == result.micro_batches
    assert result.checkpoint.optimizer_step == result.optimizer_steps
    assert result.checkpoint.model_state, "a checkpoint without model state cannot resume"


def test_checkpoints_fire_on_the_cadence_run_control_names():
    seen = []
    _, step = model_and_step()
    ctl = control(total_steps=10)
    object.__setattr__(ctl, "checkpoint_every", 2)
    train_rung0(
        [batch() for _ in range(5)], step=step, control=ctl, on_checkpoint=seen.append
    )
    assert [c.optimizer_step for c in seen] == [2, 4]


def test_padding_is_a_gate_not_a_number_nobody_reads():
    _, step = model_and_step()
    result = train_rung0([batch()], step=step, control=control(total_steps=5))
    assert isinstance(result.padding, Ran)
    assert 0.0 <= float(result.padding.value) < 1.0
    assert result.padding.passed == (float(result.padding.value) <= MAX_PADDING_WASTE)
    assert result.padding.n_total > 0


def test_a_run_that_consumed_nothing_reports_padding_as_not_run():
    """0.0 would read as perfect bucketing on a run that trained on nothing."""
    ticks = iter([0.0] + [1000.0] * 20)
    last = [0.0]

    def clock():
        last[0] = next(ticks, last[0])
        return last[0]

    _, step = model_and_step()
    result = train_rung0(
        [batch()], step=step, control=control(total_steps=5, cap_s=1.0, clock=clock)
    )
    assert result.micro_batches == 0
    assert isinstance(result.padding, NotRun)
    assert not hasattr(result.padding, "passed")


def test_the_result_reports_how_many_rows_it_trained_on():
    _, step = model_and_step()
    plans = [batch(), batch()]
    result = train_rung0(plans, step=step, control=control(total_steps=5))
    assert result.rows == sum(p.batch_size for p in plans)
