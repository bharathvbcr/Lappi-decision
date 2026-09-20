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
    assert isinstance(build_optimizer(bf16, spec=ADAMW_BF16, lr=1e-4), torch.optim.AdamW)
    assert isinstance(
        build_optimizer([_bf16_param()], spec=MASTER_SPEC, lr=1e-4), MasterWeightAdamW
    )


def test_the_builder_refuses_a_spec_that_does_not_describe_the_parameters() -> None:
    """ADAMW_FP32 over a bf16 tower budgets 4-byte states against the 2-byte states torch
    builds. Over-budgeting never crashes, which is why it survived once already."""
    with pytest.raises(ValueError, match="layout nothing builds"):
        build_optimizer([_bf16_param()], spec=ADAMW_FP32, lr=1e-4)


def test_the_builder_refuses_a_spec_with_the_wrong_state_count() -> None:
    sgd_like = OptimizerSpec("SGD+momentum", 1, 2)
    with pytest.raises(ValueError, match="exactly two"):
        build_optimizer([_bf16_param()], spec=sgd_like, lr=1e-4)


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
