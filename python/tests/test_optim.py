"""``MasterWeightAdamW``: does it actually keep the precision it claims, and refuse loudly?

The defect this class exists for is measured in ``tools/moment_precision.py``: at
``beta2=0.999`` a bf16 ``exp_avg_sq`` settles at 0.5 against a true 1.0 after 383 steps and
cannot recover when the gradient scale changes. These tests pin the fix, the refusals, and
-- the part most likely to rot -- that a checkpoint round trip does not quietly throw the
fp32 master away.

Torch-gated: skips in the repo venv rather than passing vacuously. To run::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project \\
        --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \\
        python -m pytest python/tests/test_optim.py -o addopts= -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.memory import ADAMW_BF16, ADAMW_FP32, OptimizerSpec  # noqa: E402
from qd_train.optim import MasterWeightAdamW, build_optimizer  # noqa: E402

MASTER_SPEC = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)


def _bf16_param(n: int = 256, value: float = 1.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((n,), value, dtype=torch.bfloat16))


def _drive(opt, p: torch.nn.Parameter, *, grad: float, steps: int) -> None:
    for _ in range(steps):
        p.grad = torch.full_like(p, grad)
        opt.step()
        opt.zero_grad(set_to_none=True)


# -- the defect, and that this fixes it ---------------------------------------------------


def test_plain_adamw_on_bf16_freezes_its_second_moment() -> None:
    """Characterisation of the thing being fixed, so the fix has something to be better than.

    This asserts the *defect*: if a future torch changes AdamW to keep fp32 state for bf16
    parameters, this test fails and `MasterWeightAdamW` may no longer be needed.
    """
    p = _bf16_param()
    opt = torch.optim.AdamW([p], lr=0.0, betas=(0.9, 0.999))
    _drive(opt, p, grad=1.0, steps=2000)
    settled = float(opt.state[p]["exp_avg_sq"][0].item())
    assert settled == pytest.approx(0.5, abs=1e-3), (
        f"bf16 exp_avg_sq settled at {settled}, not the measured 0.5. The premise of "
        "MasterWeightAdamW has changed and should be re-measured."
    )


def test_the_master_optimizer_tracks_the_second_moment_that_bf16_loses() -> None:
    """Against the analytic EMA, not against the asymptote.

    From a zero start with a constant ``g**2 = 1``, the exact value after ``n`` steps is
    ``1 - beta2**n``; at n=2000 that is 0.864797. Comparing against the *limit* (1.0) would
    pass for anything close to converged and would not notice a moment that tracks slowly.
    The contrast is the point: at the very same step count plain bf16 sits at 0.5, frozen.
    """
    steps = 2000
    exact = 1.0 - 0.999**steps
    p = _bf16_param()
    opt = MasterWeightAdamW([p], lr=0.0)
    _drive(opt, p, grad=1.0, steps=steps)
    v = float(next(iter(opt.state.values()))["exp_avg_sq"][0].item())
    assert v == pytest.approx(exact, rel=1e-5), f"exp_avg_sq reached {v}, exact is {exact}"

    plain = _bf16_param()
    po = torch.optim.AdamW([plain], lr=0.0, betas=(0.9, 0.999))
    _drive(po, plain, grad=1.0, steps=steps)
    bf = float(po.state[plain]["exp_avg_sq"][0].item())
    assert abs(v - exact) < abs(bf - exact), (
        f"the fp32 master ({v}) is no closer to the exact EMA ({exact}) than bf16 ({bf})"
    )


def test_the_second_moment_still_follows_a_change_in_gradient_scale() -> None:
    """The failure that matters for a real run: bf16 froze at 32.0 against a true 100.0."""
    p = _bf16_param()
    opt = MasterWeightAdamW([p], lr=0.0)
    _drive(opt, p, grad=1.0, steps=2000)
    _drive(opt, p, grad=10.0, steps=20000)
    v = float(next(iter(opt.state.values()))["exp_avg_sq"][0].item())
    assert v == pytest.approx(100.0, rel=1e-2), (
        f"exp_avg_sq reached {v} after the gradient magnitude rose 10x; expected ~100.0. "
        "bf16 reaches 32.0 here."
    )


def test_the_live_parameters_stay_in_their_own_dtype() -> None:
    """The master is fp32; the model is not. If the live parameter were promoted, every
    matmul after the first step would run at fp32 speed and the memory budget would be
    wrong by the size of the model."""
    p = _bf16_param()
    opt = MasterWeightAdamW([p], lr=1e-3)
    _drive(opt, p, grad=1.0, steps=3)
    assert p.dtype == torch.bfloat16


def test_a_step_actually_moves_the_parameters() -> None:
    p = _bf16_param()
    before = p.detach().clone()
    opt = MasterWeightAdamW([p], lr=1e-2)
    _drive(opt, p, grad=1.0, steps=5)
    assert not torch.equal(p.detach(), before), "the parameters did not move"


def test_sub_bf16_updates_accumulate_instead_of_rounding_away() -> None:
    """The second half of the precision story, and the one a single step cannot show.

    An update smaller than bf16's spacing is lost entirely by a bf16 optimizer -- each step
    rounds back to where it started. Against an fp32 master the same updates accumulate
    until they cross a representable boundary.
    """
    lr = 1e-6  # deliberately below bf16's ability to record a single step at this scale
    short = 200

    plain = _bf16_param(value=1.0)
    po = torch.optim.AdamW([plain], lr=lr)
    _drive(po, plain, grad=1.0, steps=short)
    assert float(plain[0].item()) == 1.0, (
        "plain AdamW moved the bf16 parameter at an lr this test assumes it cannot record; "
        "re-measure if torch changed"
    )

    mastered = _bf16_param(value=1.0)
    mo = MasterWeightAdamW([mastered], lr=lr)
    _drive(mo, mastered, grad=1.0, steps=short)
    # The LIVE parameter is a rounded view of the master and is expected to be unmoved
    # here -- that is not the failure, it is the mechanism. The master is where the
    # accumulation lives, and asserting on the live value alone would test the wrong object.
    assert float(mastered[0].item()) == 1.0
    master_moved = abs(float(mo.state_dict()["masters"][0][0].item()) - 1.0)
    assert master_moved > 0.0, (
        "the fp32 master did not accumulate updates that bf16 rounds away, which is the "
        "whole point of holding one"
    )

    # And the payoff: keep going and the accumulation crosses a representable boundary, so
    # an update that plain bf16 discards forever eventually becomes a real parameter change.
    _drive(mo, mastered, grad=1.0, steps=4000)
    assert float(mastered[0].item()) != 1.0, (
        "after 4200 steps the accumulated master never surfaced in the live parameter"
    )
    plain_long = _bf16_param(value=1.0)
    po2 = torch.optim.AdamW([plain_long], lr=lr)
    _drive(po2, plain_long, grad=1.0, steps=4200)
    assert float(plain_long[0].item()) == 1.0, (
        "plain bf16 AdamW moved after 4200 steps; the contrast this test draws is gone"
    )


# -- the resume path: where a silent precision loss would hide ----------------------------


def test_a_checkpoint_round_trip_restores_the_master_not_a_rounded_copy() -> None:
    """A resume that rebuilds the master from the bf16 parameter throws away exactly the
    precision this class keeps, and would do it silently."""
    p = _bf16_param()
    opt = MasterWeightAdamW([p], lr=1e-4)
    _drive(opt, p, grad=1.0, steps=50)
    saved = opt.state_dict()
    master_before = saved["masters"][0].detach().clone()
    # The bf16 parameter is a *rounded* view of the master -- confirm they really differ,
    # or this test proves nothing.
    assert not torch.equal(master_before, p.detach().to(torch.float32)), (
        "the master and the rounded bf16 parameter are identical here, so this test cannot "
        "distinguish a restored master from a rebuilt one"
    )

    fresh_p = _bf16_param()
    fresh = MasterWeightAdamW([fresh_p], lr=1e-4)
    fresh.load_state_dict(saved)
    assert torch.equal(fresh.state_dict()["masters"][0], master_before)


def test_a_resumed_run_continues_the_trajectory_it_was_cut_from() -> None:
    uninterrupted = _bf16_param()
    opt = MasterWeightAdamW([uninterrupted], lr=1e-3)
    _drive(opt, uninterrupted, grad=0.5, steps=40)

    first = _bf16_param()
    o1 = MasterWeightAdamW([first], lr=1e-3)
    _drive(o1, first, grad=0.5, steps=20)
    saved = o1.state_dict()

    second = _bf16_param()
    o2 = MasterWeightAdamW([second], lr=1e-3)
    o2.load_state_dict(saved)
    _drive(o2, second, grad=0.5, steps=20)

    assert torch.equal(second.detach(), uninterrupted.detach()), (
        "a run resumed at the half-way point did not reproduce the uninterrupted "
        "trajectory"
    )


def test_a_partial_state_is_refused_rather_than_half_loaded() -> None:
    p = _bf16_param()
    opt = MasterWeightAdamW([p], lr=1e-4)
    full = opt.state_dict()
    for drop in ("masters", "inner"):
        partial = {k: v for k, v in full.items() if k != drop}
        fresh = MasterWeightAdamW([_bf16_param()], lr=1e-4)
        with pytest.raises(ValueError, match="missing"):
            fresh.load_state_dict(partial)


def test_a_state_for_a_different_model_is_refused() -> None:
    opt = MasterWeightAdamW([_bf16_param(n=256)], lr=1e-4)
    other = MasterWeightAdamW([_bf16_param(n=128)], lr=1e-4)
    with pytest.raises(ValueError, match="different model"):
        other.load_state_dict(opt.state_dict())

    two = MasterWeightAdamW([_bf16_param(), _bf16_param()], lr=1e-4)
    with pytest.raises(ValueError, match="different model"):
        two.load_state_dict(opt.state_dict())


# -- refusals -----------------------------------------------------------------------------


def test_an_all_fp32_model_is_refused() -> None:
    p = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
    with pytest.raises(ValueError, match="only fp32 parameters"):
        MasterWeightAdamW([p], lr=1e-4)


# -- mixed precision, which is what the real model actually is ----------------------------


def test_a_mixed_dtype_model_gives_a_master_only_to_what_needs_one() -> None:
    """``QwenDecisionStep`` optimises a bf16 tower together with an fp32 ``SpanPointerHead``.
    An fp32 parameter is already its own master; copying it would double its memory to hold
    the same bits."""
    bf = _bf16_param(n=64)
    fp = torch.nn.Parameter(torch.ones(32, dtype=torch.float32))
    opt = MasterWeightAdamW([bf, fp], lr=1e-3)
    masters = opt.state_dict()["masters"]
    assert masters[0] is not bf, "the bf16 parameter should have a separate fp32 master"
    assert masters[1] is fp, "the fp32 parameter should be its own master, not a copy"


def test_both_halves_of_a_mixed_model_actually_train() -> None:
    """The fp32 half has no entry in ``_copy_back``; if ``step`` walked only that list for
    gradients too, the fp32 parameters would never move and nothing would say so."""
    bf = _bf16_param(n=64)
    fp = torch.nn.Parameter(torch.ones(32, dtype=torch.float32))
    before_bf, before_fp = bf.detach().clone(), fp.detach().clone()
    opt = MasterWeightAdamW([bf, fp], lr=1e-2)
    for _ in range(10):
        bf.grad = torch.full_like(bf, 1.0)
        fp.grad = torch.full_like(fp, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert not torch.equal(bf.detach(), before_bf), "the bf16 half did not train"
    assert not torch.equal(fp.detach(), before_fp), "the fp32 half did not train"
    assert bf.dtype == torch.bfloat16 and fp.dtype == torch.float32


def test_a_mixed_model_round_trips_through_a_checkpoint() -> None:
    bf = _bf16_param(n=64)
    fp = torch.nn.Parameter(torch.ones(32, dtype=torch.float32))
    opt = MasterWeightAdamW([bf, fp], lr=1e-3)
    for _ in range(20):
        bf.grad = torch.full_like(bf, 0.5)
        fp.grad = torch.full_like(fp, 0.5)
        opt.step()
        opt.zero_grad(set_to_none=True)
    saved = opt.state_dict()
    want = [m.detach().clone() for m in saved["masters"]]

    fresh = MasterWeightAdamW(
        [_bf16_param(n=64), torch.nn.Parameter(torch.ones(32, dtype=torch.float32))],
        lr=1e-3,
    )
    fresh.load_state_dict(saved)
    for got, expected in zip(fresh.state_dict()["masters"], want, strict=True):
        assert torch.equal(got, expected)


def test_an_optimizer_over_nothing_is_refused() -> None:
    frozen = torch.nn.Parameter(torch.ones(16, dtype=torch.bfloat16), requires_grad=False)
    with pytest.raises(ValueError, match="no parameters"):
        MasterWeightAdamW([frozen], lr=1e-4)


def test_an_oversized_model_is_refused_before_it_allocates() -> None:
    from qd_train import optim as optim_mod

    p = _bf16_param(n=1024)
    original = optim_mod.MAX_MASTER_PARAMS
    optim_mod.MAX_MASTER_PARAMS = 512
    try:
        with pytest.raises(ValueError, match="MAX_MASTER_PARAMS"):
            MasterWeightAdamW([p], lr=1e-4)
    finally:
        optim_mod.MAX_MASTER_PARAMS = original


def test_the_lr_written_into_param_groups_is_the_lr_that_is_used() -> None:
    """``TrainStep.apply`` sets the lr by writing into ``param_groups``. If this class
    exposed its own groups rather than the inner optimizer's, every schedule would be
    silently ignored and the run would train at the constructor's lr forever."""
    p = _bf16_param()
    opt = MasterWeightAdamW([p], lr=1e-9)
    for group in opt.param_groups:
        group["lr"] = 1e-2
    before = p.detach().clone()
    _drive(opt, p, grad=1.0, steps=5)
    moved = float((p.detach().to(torch.float32) - before.to(torch.float32)).abs().max())
    assert moved > 1e-4, (
        f"the parameter moved {moved} after 5 steps at lr=1e-2 written into param_groups; "
        "the schedule is not reaching the optimizer that steps"
    )


# -- the builder, which ties the spec to the thing the spec describes ---------------------


def test_the_builder_returns_what_the_spec_describes() -> None:
    bf16 = [_bf16_param()]
    # total_steps below the bf16 settling step, so this test is about the SPEC and not
    # about the schedule guard; the guard has its own tests below.
    assert isinstance(
        build_optimizer(bf16, spec=ADAMW_BF16, lr=1e-4, total_steps=10),
        torch.optim.AdamW,
    )
    # A master-weight spec is admitted at ANY length -- fp32 moments settle on the right
    # value -- so this one deliberately asks for a schedule no bf16 run could have.
    assert isinstance(
        build_optimizer(
            [_bf16_param()], spec=MASTER_SPEC, lr=1e-4, total_steps=1_000_000
        ),
        MasterWeightAdamW,
    )


def test_the_builder_refuses_a_spec_that_does_not_describe_the_parameters() -> None:
    """ADAMW_FP32 over a bf16 tower budgets 4-byte states against the 2-byte states torch
    builds. Over-budgeting never crashes, which is why it survived once already."""
    with pytest.raises(ValueError, match="layout nothing builds"):
        build_optimizer([_bf16_param()], spec=ADAMW_FP32, lr=1e-4, total_steps=10)


def test_the_builder_refuses_a_spec_with_the_wrong_state_count() -> None:
    sgd_like = OptimizerSpec("SGD+momentum", 1, 2)
    with pytest.raises(ValueError, match="exactly two"):
        build_optimizer([_bf16_param()], spec=sgd_like, lr=1e-4, total_steps=10)


def test_the_master_spec_budgets_what_the_optimizer_allocates() -> None:
    """20 B/param: 2 bf16 weights + 2 bf16 grads + 4 fp32 grad cast + 8 moments + 4 master.

    Not 16. The fp32 gradient cast is 4 B/param that the recipe cannot avoid -- ``step``
    casts every gradient up before the inner optimizer runs -- and the GH200 measured the
    five components at 35.05 GiB over 1,881,825,088 parameters, which is 20.0 B/param to
    the digit.
    """
    assert MASTER_SPEC.bytes_per_param(param_bytes=2, grad_bytes=2) == 20
    p = _bf16_param(n=4096)
    opt = MasterWeightAdamW([p], lr=1e-4)
    _drive(opt, p, grad=1.0, steps=2)
    master_bytes = sum(m.numel() * m.element_size() for m in opt.state_dict()["masters"])
    state_bytes = sum(
        v.numel() * v.element_size()
        for s in opt.state.values()
        for v in s.values()
        if torch.is_tensor(v) and v.numel() > 1
    )
    n = p.numel()
    assert master_bytes / n == 4.0, f"master is {master_bytes / n} B/param, expected 4"
    assert state_bytes / n == 8.0, f"moments are {state_bytes / n} B/param, expected 8"


# -- the schedule guard -------------------------------------------------------------------
#
# This module could measure that a bf16 second moment stops moving at step 384 and settles
# 50% low, say so at length in its own docstring, and still hand back the optimizer that
# does it -- because nothing here knew how long the run would be. These are what changed.


def test_bf16_settles_half_low_and_fp32_settles_on_the_answer() -> None:
    """The measurement the guard is built on, reproduced against torch's own rounding.

    ``tools/moment_precision.py`` is the authority and drives ``torch.optim.AdamW`` itself;
    this is the scalar EMA, so a guard can ask without allocating an optimizer. They agree
    on the value and differ by one on the step -- that tool reports the last update that
    MOVED the value, this the first that did not.
    """
    from qd_train.optim import moment_settling

    bf16 = moment_settling(dtype=torch.bfloat16)
    assert bf16.settled_at_step == 384
    assert bf16.settled_value == pytest.approx(0.5, abs=1e-9)
    assert not bf16.is_faithful

    fp32 = moment_settling(dtype=torch.float32)
    assert fp32.settled_value == pytest.approx(1.0, abs=1e-4)
    assert fp32.is_faithful


def test_stopping_is_not_the_defect_stopping_wrong_is() -> None:
    """fp32 stops moving too, at step 10,301 -- on the right answer. A guard keyed on "does
    the moment freeze" would refuse the correct optimizer along with the broken one, which
    is why ``MomentSettling`` carries the VALUE and not just the step."""
    from qd_train.optim import moment_settling

    fp32 = moment_settling(dtype=torch.float32)
    assert fp32.settled_at_step is not None
    assert fp32.settled_at_step < 20_000
    # Stops, and survives a schedule far longer than where it stopped.
    assert fp32.survives(fp32.settled_at_step * 10)


def test_fp16_looks_survivable_and_is_not() -> None:
    """The case neither the docstrings nor the tools had measured. fp16 lasts three times
    longer than bf16 before it freezes, which makes it look like the safe half-precision
    choice, and it still settles 27% low."""
    from qd_train.optim import moment_settling

    fp16 = moment_settling(dtype=torch.float16)
    assert fp16.settled_at_step > moment_settling(dtype=torch.bfloat16).settled_at_step
    assert not fp16.is_faithful
    assert fp16.relative_error > 0.2


def test_a_schedule_that_outlives_its_second_moment_is_refused() -> None:
    """The hardening, stated as a test. 1,000 steps of bf16 moments is a run whose last 616
    steps are all mis-scaled by 1.41x, invisibly."""
    from qd_train.optim import build_optimizer

    with pytest.raises(ValueError, match="does not survive it"):
        build_optimizer([_bf16_param()], spec=ADAMW_BF16, lr=1e-4, total_steps=1_000)


def test_a_schedule_that_ends_before_the_freeze_is_admitted() -> None:
    """The bound is not a ban on bf16 moments. A run that finishes before step 384 has a
    second moment that tracked the whole way, and refusing it would be the same kind of
    error in the other direction."""
    from qd_train.optim import build_optimizer

    assert build_optimizer(
        [_bf16_param()], spec=ADAMW_BF16, lr=1e-4, total_steps=383
    ) is not None


def test_the_exception_has_to_be_said_out_loud() -> None:
    """There is a real case for accepting degraded moments -- a device where 16 B/param does
    not fit -- and it must be spelled, land in the recipe, and never be reachable by
    forgetting to pass something. Named after ``write_shards(allow_contradictions=...)``."""
    from qd_train.optim import build_optimizer

    assert build_optimizer(
        [_bf16_param()],
        spec=ADAMW_BF16,
        lr=1e-4,
        total_steps=100_000,
        allow_frozen_moments=True,
    ) is not None


def test_total_steps_is_required_rather_than_defaulted() -> None:
    """A default would make the guard opt-in, and an opt-in guard against an invisible
    failure is a guard that is off. Every caller already computes this number before it
    builds a step."""
    import inspect

    from qd_train.optim import build_optimizer

    param = inspect.signature(build_optimizer).parameters["total_steps"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_schedule_of_no_steps_is_refused() -> None:
    from qd_train.optim import build_optimizer

    with pytest.raises(ValueError, match="total_steps must be at least 1"):
        build_optimizer([_bf16_param()], spec=ADAMW_BF16, lr=1e-4, total_steps=0)


def test_an_impossible_beta2_is_refused_rather_than_looped_over() -> None:
    """``beta2 >= 1`` never converges and ``beta2 <= 0`` is not an EMA; either would spin to
    ``MAX_SETTLING_STEPS`` and report ``None``, which reads as "still tracking"."""
    from qd_train.optim import moment_settling

    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="beta2 must be in"):
            moment_settling(dtype=torch.float32, beta2=bad)


def test_the_step_refuses_a_schedule_its_moments_cannot_serve() -> None:
    """The guard reaches the real FT step, not just the builder. ``QwenDecisionStep`` is
    where a full train's optimizer is actually constructed."""
    from qd_train.backbone import QwenDecisionStep

    assert "total_steps" in inspect_signature_params(QwenDecisionStep.__init__)
    assert "allow_frozen_moments" in inspect_signature_params(QwenDecisionStep.__init__)


def inspect_signature_params(fn) -> set[str]:
    import inspect

    return set(inspect.signature(fn).parameters)


# -- layer-wise lr: one writer of group["lr"], honoured by every driver ---------------------


def _two_group_optimizer(params: list[torch.nn.Parameter]) -> torch.optim.AdamW:
    """The first tensor at 0.1x, the rest at 1.0x -- the shape `layerwise_param_groups`
    builds, without needing a tower whose names carry ``layers.<i>.``."""
    return torch.optim.AdamW(
        [
            {"params": params[:1], "lr_scale": 0.1, "name": "base_lower"},
            {"params": params[1:], "lr_scale": 1.0, "name": "base"},
        ],
        lr=1.0,
    )


def _driver_step(which: str) -> object:
    tools = Path(__file__).resolve().parents[2] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    if which == "Rung0Step":
        from qd_train.byte_train import Rung0Model, Rung0Step

        return Rung0Step(Rung0Model())
    if which == "RealFtStep":
        from real_ft_run import RealFtStep

        return RealFtStep(seed=0, device="cpu", vocab=32, width=8, hidden=16, heads=2,
                          lr=1e-3, span_weight=1.0)
    from ft_toy_run import ToyFtStep

    return ToyFtStep(seed=0, device="cpu")


@pytest.mark.parametrize("which", ["Rung0Step", "RealFtStep", "ToyFtStep"])
def test_every_driver_applies_the_schedule_times_the_groups_lr_scale(which: str) -> None:
    """The defect: each driver wrote ``group["lr"] = lr`` into every group, so a group built
    at 0.1x trained at 1x from its first step. Pre-fix this read 1e-3 for the scaled group
    at all three of these drivers (``QwenDecisionStep`` is pinned in test_backbone.py)."""
    step = _driver_step(which)
    params = list(step.optimizer.param_groups[0]["params"])  # type: ignore[attr-defined]
    step.optimizer = _two_group_optimizer(params)  # type: ignore[attr-defined]
    step.apply(lr=1e-3)  # type: ignore[attr-defined]
    lower, base = step.optimizer.param_groups  # type: ignore[attr-defined]
    assert lower["lr"] == pytest.approx(1e-4), f"{which} ignored lr_scale"
    assert base["lr"] == pytest.approx(1e-3), which


def test_apply_lr_refuses_a_non_positive_rate_and_a_bad_scale() -> None:
    from qd_train.optim import apply_lr

    opt = _two_group_optimizer(
        [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    )
    for bad in (0.0, -1e-3, float("nan")):
        with pytest.raises(ValueError, match="lr must be positive and finite"):
            apply_lr(opt, bad)
    opt.param_groups[0]["lr_scale"] = 0.0
    with pytest.raises(ValueError, match="lr_scale must be finite and positive"):
        apply_lr(opt, 1e-3)


def test_layerwise_groups_follow_rsi_s_rule_and_refuse_an_empty_split() -> None:
    from qd_train.optim import layerwise_param_groups

    names = [f"layers.{i}.mlp.weight" for i in range(4)] + ["norm.weight", "embed_tokens.weight"]
    named = [(n, torch.nn.Parameter(torch.zeros(2))) for n in names]
    extra = [torch.nn.Parameter(torch.zeros(2))]
    base, lower = layerwise_param_groups(
        named, lower_layers_n=2, lower_lr_scale=0.1, extra=extra
    )
    assert lower["lr_scale"] == 0.1 and base["lr_scale"] == 1.0
    assert [id(p) for p in lower["params"]] == [id(named[0][1]), id(named[1][1])]
    assert len(base["params"]) == 5  # layers 2,3 + norm + embed + the span head
    with pytest.raises(ValueError, match="matched no trainable parameter"):
        layerwise_param_groups(
            [("norm.weight", named[4][1])], lower_layers_n=2, lower_lr_scale=0.1
        )
    with pytest.raises(ValueError, match="deepest layer is 3"):
        layerwise_param_groups(named, lower_layers_n=5, lower_lr_scale=0.1)
    with pytest.raises(ValueError, match="at least 1"):
        layerwise_param_groups(named, lower_layers_n=0, lower_lr_scale=0.1)


def test_build_optimizer_carries_group_scales_into_both_recipes() -> None:
    from qd_train.optim import apply_lr, build_optimizer

    for spec, dtype in ((ADAMW_FP32, torch.float32), (MASTER_SPEC, torch.bfloat16)):
        a = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
        b = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
        opt = build_optimizer(
            [{"params": [a], "lr_scale": 0.1}, {"params": [b], "lr_scale": 1.0}],
            spec=spec, lr=1e-3, total_steps=10,
        )
        assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-4, 1e-3]), spec.name
        apply_lr(opt, 2e-3)
        assert [g["lr"] for g in opt.param_groups] == pytest.approx([2e-4, 2e-3]), spec.name


# -- beta2 as a parameter, and the fidelity check asked at the beta2 actually used -----------


def test_beta2_reaches_both_optimizers_and_defaults_to_torch_s_value() -> None:
    from qd_train.optim import DEFAULT_BETA2, build_optimizer

    assert DEFAULT_BETA2 == 0.999
    fp32 = build_optimizer(
        [torch.nn.Parameter(torch.zeros(4))], spec=ADAMW_FP32, lr=1e-3, total_steps=10
    )
    assert fp32.param_groups[0]["betas"] == (0.9, 0.999)
    for spec, p in (
        (ADAMW_FP32, torch.nn.Parameter(torch.zeros(4))),
        (MASTER_SPEC, _bf16_param()),
    ):
        opt = build_optimizer([p], spec=spec, lr=1e-3, total_steps=10, beta2=0.95)
        assert opt.param_groups[0]["betas"] == (0.9, 0.95), spec.name


def test_the_bf16_fidelity_check_is_asked_at_the_optimizers_own_beta2() -> None:
    """bf16 at 0.95 settles 1.95% low after 64 steps -- measured by ``moment_settling`` on
    this host -- against 50% low after 384 at 0.999. So a 100-step bf16 run is admitted at
    0.999 and refused at 0.95. Before beta2 was a parameter the check could only ever ask
    0.999's question, whatever optimizer it was guarding."""
    from qd_train.optim import build_optimizer, moment_settling

    s = moment_settling(dtype=torch.bfloat16, beta2=0.95)
    assert s.settled_at_step == 64 and not s.is_faithful
    build_optimizer([_bf16_param()], spec=ADAMW_BF16, lr=1e-4, total_steps=100)
    with pytest.raises(ValueError, match=r"beta2=0\.95"):
        build_optimizer(
            [_bf16_param()], spec=ADAMW_BF16, lr=1e-4, total_steps=100, beta2=0.95
        )


# -- fused AdamW over the masters (Tier B: a numerics change, so opt-in and recorded) -------


def test_fused_builds_torchs_fused_kernel_over_the_masters_and_default_does_not() -> None:
    fused = build_optimizer([_bf16_param()], spec=MASTER_SPEC, lr=1e-4, total_steps=100,
                            fused=True)
    plain = build_optimizer([_bf16_param()], spec=MASTER_SPEC, lr=1e-4, total_steps=100)
    assert fused.param_groups[0]["fused"] is True
    assert not plain.param_groups[0]["fused"], "the default must stay torch's foreach AdamW"


def test_fused_tracks_the_default_update_closely_but_is_its_own_recipe() -> None:
    """Same rule, different rounding: close, which is why it is Tier B and not Tier A."""
    torch.manual_seed(0)
    grads = [torch.randn(256) for _ in range(20)]
    out = []
    for fused in (False, True):
        p = torch.nn.Parameter(torch.linspace(-1, 1, 256).to(torch.bfloat16))
        opt = MasterWeightAdamW([p], lr=1e-3, fused=fused)
        for g in grads:
            p.grad = g.to(torch.bfloat16)
            opt.step()
            opt.zero_grad(set_to_none=True)
        out.append(opt._masters[0].detach().clone())
    assert torch.allclose(out[0], out[1], rtol=0, atol=1e-6)


def test_fused_on_a_recipe_without_masters_is_refused() -> None:
    with pytest.raises(ValueError, match="fp32-master recipe only"):
        build_optimizer([_bf16_param()], spec=ADAMW_BF16, lr=1e-4, total_steps=100,
                        fused=True)


# -- KahanBf16AdamW: the 16-bit recipe (bf16 weights + bf16 compensation + fp32 moments) ------
#
# The question each test below answers is whether the compensated recipe reaches where the
# master recipe reaches, at 14 B/param instead of 20, and whether it refuses the inputs that
# would make that false without a word. The reference is MasterWeightAdamW, the recipe v5
# trained every row with; the contrast is plain torch.optim.AdamW over bf16, which loses the
# updates. The trajectory is a noisy quadratic at the real tower's numbers: weights ~N(0,
# 0.02), the lr real_ft_run.REAL_BACKBONE_LR = 1e-5.

from qd_train.memory import ADAMW_KAHAN, ADAMW_MASTER  # noqa: E402
from qd_train.optim import (  # noqa: E402
    KAHAN_RECIPE,
    KahanBf16AdamW,
)

REAL_LR = 1e-5


def _quadratic_run(kind: str, *, steps: int, lr: float = REAL_LR, n: int = 4096, seed: int = 0):
    """Train one bf16 vector toward a target under shared noise. Returns (w0, value, opt, p).

    ``value`` is the full-precision weight each recipe holds: the master for the master
    recipe, ``p + c`` for the compensated one, the bf16 weight itself for plain AdamW. The
    gradient is computed from that value, as a real backward reads the weights it trains.
    """
    gen = torch.Generator().manual_seed(seed)
    w0 = torch.randn(n, generator=gen) * 0.02
    target = w0 + torch.randn(n, generator=gen) * 0.005
    p = torch.nn.Parameter(w0.to(torch.bfloat16))
    if kind == "master":
        opt = MasterWeightAdamW([p], lr=lr)
    elif kind == "kahan":
        opt = KahanBf16AdamW([p], lr=lr)
    else:
        opt = torch.optim.AdamW([p], lr=lr)

    def value() -> torch.Tensor:
        if kind == "master":
            return opt._masters[0].detach().clone()
        if kind == "kahan":
            st = opt.state.get(p)  # allocated at the first step
            return p.detach().float() + (st["kahan_comp"].float() if st else 0.0)
        return p.detach().float()

    for _ in range(steps):
        noise = torch.randn(n, generator=gen) * 0.01
        p.grad = ((value() - target) + noise).to(torch.bfloat16)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return w0, value(), opt, p


def test_plain_bf16_adamw_loses_most_of_the_update_at_the_real_lr() -> None:
    """The defect, characterised, so the fix has something to beat.

    At lr 1e-5 an Adam step on a 0.02-magnitude weight is below bf16's half-spacing, so most
    steps round away. Measured on this host (2026-10-06, 2,000 steps): plain bf16 ends 95.7%
    of the master's distance away from the master. If torch ever changes this, the recipe
    comparison below has to be re-measured.
    """
    w0, master, _, _ = _quadratic_run("master", steps=2000)
    _, plain, _, _ = _quadratic_run("plain", steps=2000)
    moved = float((master - w0).norm())
    assert moved > 0.1, "the master trajectory did not move; the comparison is vacuous"
    err = float((plain - master).norm()) / moved
    assert err > 0.5, f"plain bf16 tracked the master to {err:.3e}; the premise has changed"


def test_kahan_tracks_the_master_trajectory_at_the_real_lr() -> None:
    """The fix: within 0.5% of the master's movement, where plain bf16 misses by ~96%.

    Measured on this host: 2.7e-4 at 2,000 steps and 5.2e-4 at 500. The bound is ten times
    the larger. The residual is the compensation's own 8-bit rounding, which the master does
    not have; it is not a drift that grows with steps (the 2,000-step error is smaller).
    """
    for steps in (500, 2000):
        w0, master, _, _ = _quadratic_run("master", steps=steps)
        _, kahan, _, _ = _quadratic_run("kahan", steps=steps)
        moved = float((master - w0).norm())
        err = float((kahan - master).norm()) / moved
        assert err < 5e-3, f"{steps} steps: compensated recipe is {err:.3e} off the master"


def test_kahan_update_rule_is_torch_adamw_on_fp32_parameters() -> None:
    """An fp32 parameter takes the same rule with no compensation, so it must reproduce
    torch.optim.AdamW exactly: same decay, bias correction, eps placement and lr_scale."""
    torch.manual_seed(1)
    init = torch.randn(300)
    grads = [torch.randn(300) for _ in range(60)]
    mine = torch.nn.Parameter(init.clone())
    ref = torch.nn.Parameter(init.clone())
    anchor = _bf16_param(n=8)  # an fp32-only model is refused; the anchor never gets a grad
    opt = KahanBf16AdamW(
        [{"params": [mine], "lr_scale": 0.5}, {"params": [anchor]}],
        lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1,
    )
    torch_opt = torch.optim.AdamW([ref], lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.1)
    for g in grads:
        mine.grad, ref.grad = g.clone(), g.clone()
        opt.step()
        torch_opt.step()
    assert torch.equal(mine.detach(), ref.detach()), (
        float((mine - ref).abs().max())
    )
    assert torch.equal(opt.state[mine]["exp_avg_sq"], torch_opt.state[ref]["exp_avg_sq"])


def test_kahan_keeps_the_layout_its_budget_describes() -> None:
    """14 B/param: bf16 weight 2 + bf16 grad 2 + bf16 compensation 2 + fp32 moments 8."""
    assert ADAMW_KAHAN.bytes_per_param(param_bytes=2, grad_bytes=2) == 14
    assert ADAMW_MASTER.bytes_per_param(param_bytes=2, grad_bytes=2) == 20
    p = _bf16_param(n=4096)
    opt = KahanBf16AdamW([p], lr=1e-4)
    _drive(opt, p, grad=1.0, steps=2)
    st = opt.state[p]
    n = p.numel()
    assert p.dtype == torch.bfloat16 and st["kahan_comp"].dtype == torch.bfloat16
    assert st["exp_avg"].dtype == st["exp_avg_sq"].dtype == torch.float32
    assert st["kahan_comp"].numel() * st["kahan_comp"].element_size() / n == 2.0
    moments = st["exp_avg"].numel() * 4 + st["exp_avg_sq"].numel() * 4
    assert moments / n == 8.0


def test_kahan_second_moment_follows_a_change_in_gradient_scale() -> None:
    """fp32 moments: the failure plain bf16 has (frozen at 32 against 100) is absent."""
    p = _bf16_param()
    opt = KahanBf16AdamW([p], lr=0.0)
    _drive(opt, p, grad=1.0, steps=2000)
    _drive(opt, p, grad=10.0, steps=20000)
    assert float(opt.state[p]["exp_avg_sq"][0]) == pytest.approx(100.0, rel=1e-2)


def test_kahan_slices_do_not_change_the_step() -> None:
    """The bounded-memory slicing is an implementation detail: any slice size, same bits."""
    torch.manual_seed(2)
    grads = [torch.randn(1000).to(torch.bfloat16) for _ in range(30)]
    out = []
    for chunk in (7, 1000, 1 << 24):
        p = torch.nn.Parameter(torch.linspace(-0.05, 0.05, 1000).to(torch.bfloat16))
        opt = KahanBf16AdamW([p], lr=1e-4, chunk_elems=chunk)
        for g in grads:
            p.grad = g.clone()
            opt.step()
        out.append((p.detach().clone(), opt.state[p]["kahan_comp"].clone()))
    for p_k, c_k in out[1:]:
        assert torch.equal(p_k, out[0][0]) and torch.equal(c_k, out[0][1])


def test_kahan_resume_continues_the_trajectory_bit_for_bit() -> None:
    """The compensation is saved exactly; a resume that rebuilt it from the bf16 weight would
    throw away the bits the recipe exists for, silently."""
    grads = [torch.full((64,), 0.3 * ((-1) ** i)) for i in range(40)]

    def drive(opt, p, gs):
        for g in gs:
            p.grad = g.to(torch.bfloat16)
            opt.step()
            opt.zero_grad(set_to_none=True)

    whole = torch.nn.Parameter(torch.linspace(-0.02, 0.02, 64).to(torch.bfloat16))
    wo = KahanBf16AdamW([whole], lr=1e-5)
    drive(wo, whole, grads)

    first = torch.nn.Parameter(torch.linspace(-0.02, 0.02, 64).to(torch.bfloat16))
    fo = KahanBf16AdamW([first], lr=1e-5)
    drive(fo, first, grads[:20])
    saved = fo.state_dict()
    assert saved["recipe"] == KAHAN_RECIPE

    second = torch.nn.Parameter(first.detach().clone())  # the weights come back via the model
    so = KahanBf16AdamW([second], lr=1e-5)
    so.load_state_dict(saved)
    drive(so, second, grads[20:])
    assert torch.equal(second.detach(), whole.detach())
    assert torch.equal(so.state[second]["kahan_comp"], wo.state[whole]["kahan_comp"])


def _split_cases() -> list[torch.Tensor]:
    """Values the split must hold on: every scale bf16 reaches, values a small fraction of a
    spacing off the bf16 grid (exact ties included), power-of-two edges, and signed zeros."""
    gen = torch.Generator().manual_seed(3)
    cases = [torch.randn(200_000, generator=gen) * s for s in (1e-30, 1e-4, 0.02, 1.0, 1e30)]
    grid = torch.randn(50_000, generator=gen).to(torch.bfloat16).to(torch.float32)
    spacing = grid.abs() * 2.0**-7
    for frac in (0.5, -0.5, 0.4999, 2.0**-17, 2.0**-23, 2.0**-30):
        cases.append(grid + frac * spacing)
    p2 = torch.tensor([2.0**k for k in range(-120, 120)])
    for d in (1e-7, -1e-7, 2.0**-9, -(2.0**-9)):
        cases += [p2 * (1 + d), -p2 * (1 + d)]
    cases.append(torch.tensor([0.0, -0.0]))
    return [c[torch.isfinite(c)] for c in cases]


def test_the_kahan_split_is_exact_canonical_and_recoverable() -> None:
    """The three properties the checkpoint format rests on.
    1. fp32(hi) + fp32(lo) is exact, so masters lose nothing.
    2. bf16(hi + lo) == hi everywhere, ties included, which is the identity ckpt_average
       matches a master to its weight by.
    3. lo == bf16(master - hi), so a resume recovers the compensation from the live weight.
    Without the canonical re-split, property 2 fails on exact ties (the 0.5 rows)."""
    from qd_train.optim import kahan_split

    ties_moved = 0
    for w in _split_cases():
        hi, lo = kahan_split(w, torch.bfloat16)
        total = hi.float() + lo.float()
        assert torch.equal(total - hi.float(), lo.float()), "the sum was not exact"
        assert torch.equal(total.to(torch.bfloat16), hi), "a master would not round to its weight"
        assert torch.equal((total - hi.float()).to(torch.bfloat16), lo)
        naive = w.to(torch.bfloat16)
        ties_moved += int((naive != hi).sum())
    assert ties_moved > 0, "no case exercised the tie the canonical re-split exists for"


def test_kahan_masters_reconstruct_and_round_to_their_weights_after_training() -> None:
    _, _, opt, p = _quadratic_run("kahan", steps=300)
    master = opt.state_dict()["masters"][0]
    assert master.dtype == torch.float32
    assert torch.equal(master.to(torch.bfloat16), p.detach())
    assert torch.equal(p.detach().float() + opt.state[p]["kahan_comp"].float(), master)


def test_kahan_allocates_nothing_until_it_steps() -> None:
    """A scoring step builds the training step and never steps it. Under this recipe it must
    pay nothing for that: no moments, no compensation."""
    a, b = _bf16_param(n=64), torch.nn.Parameter(torch.zeros(8))
    opt = KahanBf16AdamW([a, b], lr=1e-4)
    assert opt.state == {}
    saved = opt.state_dict()
    assert saved["state"] == {}
    assert [m.dtype for m in saved["masters"]] == [torch.float32, torch.float32]
    a.grad = torch.ones_like(a)
    opt.step()
    assert set(opt.state) == {a}, "only the parameter that had a gradient got state"


def test_a_kahan_checkpoint_is_as_large_as_a_master_checkpoint() -> None:
    """14 B/param with the bf16 tower: masters 4 + moments 8 in the optimizer state, the
    compensation recovered rather than stored. The first version stored both and was 16."""
    p = _bf16_param(n=4096)
    opt = KahanBf16AdamW([p], lr=1e-4)
    _drive(opt, p, grad=1.0, steps=3)
    saved = opt.state_dict()
    held = sum(m.numel() * m.element_size() for m in saved["masters"]) + sum(
        v.numel() * v.element_size()
        for e in saved["state"].values() for v in e.values() if torch.is_tensor(v)
    )
    assert held / p.numel() == 12.0
    assert all("kahan_comp" not in e for e in saved["state"].values())


def test_a_compensated_weight_without_optimizer_state_is_refused() -> None:
    p = _bf16_param(n=4)
    opt = KahanBf16AdamW([p], lr=1e-4)
    forged = {**opt.state_dict()}
    forged["masters"] = [p.detach().float() + 2.0**-12]
    with pytest.raises(ValueError, match="no optimizer state"):
        KahanBf16AdamW([p], lr=1e-4).load_state_dict(forged)


def test_kahan_refuses_a_state_it_did_not_write_or_that_does_not_match() -> None:
    p = _bf16_param(n=32)
    opt = KahanBf16AdamW([p], lr=1e-4)
    _drive(opt, p, grad=0.5, steps=3)
    good = opt.state_dict()

    master_state = MasterWeightAdamW([_bf16_param(n=32)], lr=1e-4).state_dict()
    with pytest.raises(ValueError, match="missing"):
        KahanBf16AdamW([_bf16_param(n=32)], lr=1e-4).load_state_dict(master_state)
    with pytest.raises(ValueError, match="recipe"):
        KahanBf16AdamW([p], lr=1e-4).load_state_dict({**good, "recipe": "master"})
    for drop in ("state", "masters", "param_groups", "recipe"):
        with pytest.raises(ValueError, match="missing"):
            KahanBf16AdamW([p], lr=1e-4).load_state_dict(
                {k: v for k, v in good.items() if k != drop}
            )
    with pytest.raises(ValueError, match="different model"):
        KahanBf16AdamW([_bf16_param(n=16)], lr=1e-4).load_state_dict(good)
    with pytest.raises(ValueError, match="different model"):
        KahanBf16AdamW([_bf16_param(n=32), _bf16_param(n=32)], lr=1e-4).load_state_dict(good)
    # Weights from another checkpoint: the masters do not round to them.
    other = _bf16_param(n=32, value=3.0)
    with pytest.raises(ValueError, match="different checkpoints"):
        KahanBf16AdamW([other], lr=1e-4).load_state_dict(good)


def test_kahan_refuses_what_it_cannot_compensate_or_bound() -> None:
    from qd_train import optim as optim_mod

    with pytest.raises(ValueError, match="no parameters"):
        KahanBf16AdamW(
            [torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)],
            lr=1e-4,
        )
    with pytest.raises(ValueError, match="only fp32"):
        KahanBf16AdamW([torch.nn.Parameter(torch.ones(4))], lr=1e-4)
    with pytest.raises(ValueError, match="bf16 and fp32 parameters only"):
        KahanBf16AdamW([torch.nn.Parameter(torch.ones(4, dtype=torch.float16))], lr=1e-4)
    shared = _bf16_param(n=4)
    with pytest.raises(ValueError, match="more than one group"):
        KahanBf16AdamW([{"params": [shared]}, {"params": [shared]}], lr=1e-4)
    strided = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16).t())
    with pytest.raises(ValueError, match="not contiguous"):
        KahanBf16AdamW([strided], lr=1e-4)
    for bad in (dict(lr=float("nan")), dict(lr=-1.0), dict(lr=1e-4, eps=0.0),
                dict(lr=1e-4, betas=(0.9, 1.0)), dict(lr=1e-4, weight_decay=float("inf")),
                dict(lr=1e-4, chunk_elems=0)):
        with pytest.raises(ValueError):
            KahanBf16AdamW([_bf16_param(n=4)], **bad)
    original = optim_mod.MAX_KAHAN_PARAMS
    optim_mod.MAX_KAHAN_PARAMS = 512
    try:
        with pytest.raises(ValueError, match="MAX_KAHAN_PARAMS"):
            KahanBf16AdamW([_bf16_param(n=1024)], lr=1e-4)
    finally:
        optim_mod.MAX_KAHAN_PARAMS = original


def test_kahan_refuses_a_nan_lr_written_into_its_groups_before_it_steps() -> None:
    """apply_lr validates, but a group can be written directly; a NaN lr would write NaN into
    every weight. The step checks the values it is about to use."""
    p = _bf16_param(n=4)
    opt = KahanBf16AdamW([p], lr=1e-4)
    opt.param_groups[0]["lr"] = float("nan")
    p.grad = torch.ones_like(p)
    before = p.detach().clone()
    with pytest.raises(ValueError, match="lr"):
        opt.step()
    assert torch.equal(p.detach(), before)


def test_kahan_skips_parameters_without_a_gradient_and_empty_ones() -> None:
    a, b = _bf16_param(n=8), _bf16_param(n=8)
    empty = torch.nn.Parameter(torch.zeros(0, dtype=torch.bfloat16))
    opt = KahanBf16AdamW([a, b, empty], lr=1e-2)
    a.grad = torch.ones_like(a)
    empty.grad = torch.zeros_like(empty)
    before_b = b.detach().clone()
    opt.step()
    assert torch.equal(b.detach(), before_b) and b not in opt.state
    assert opt.state[a]["step"] == 1 and not torch.equal(a.detach(), before_b)
    assert opt.state[empty]["step"] == 1


def test_the_builder_builds_the_kahan_recipe_from_its_spec_and_refuses_a_wrong_one() -> None:
    opt = build_optimizer([_bf16_param()], spec=ADAMW_KAHAN, lr=1e-4, total_steps=1_000_000)
    assert isinstance(opt, KahanBf16AdamW), "fp32 moments survive any schedule"
    with pytest.raises(ValueError, match="fp32-master recipe only"):
        build_optimizer([_bf16_param()], spec=ADAMW_KAHAN, lr=1e-4, total_steps=10, fused=True)
    wrong = OptimizerSpec("kahan-with-bf16-moments", 2, 2, compensation_bytes=2)
    with pytest.raises(ValueError, match="layout nothing builds"):
        build_optimizer([_bf16_param()], spec=wrong, lr=1e-4, total_steps=10)
    scaled = build_optimizer(
        [{"params": [_bf16_param()], "lr_scale": 0.1}, {"params": [_bf16_param()]}],
        spec=ADAMW_KAHAN, lr=1e-3, total_steps=10,
    )
    assert [g["lr"] for g in scaled.param_groups] == pytest.approx([1e-4, 1e-3])
    from qd_train.optim import apply_lr

    apply_lr(scaled, 2e-3)
    assert [g["lr"] for g in scaled.param_groups] == pytest.approx([2e-4, 2e-3])


def test_a_spec_cannot_hold_both_a_master_and_a_compensation() -> None:
    with pytest.raises(ValueError, match="both an fp32 master and a compensation"):
        OptimizerSpec("both", 2, 4, keeps_fp32_master=True, compensation_bytes=2)
    with pytest.raises(ValueError, match="non-negative"):
        OptimizerSpec("negative", 2, 4, compensation_bytes=-2)
