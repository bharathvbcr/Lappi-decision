"""``tools/memory_budget.py`` budgets only layouts the trainer builds, and defaults to v5's.

The defect pinned here: the tool's verdict defaulted to "bf16 weights+grads, fp32 AdamW
states", which no optimizer in this repository builds -- ``build_optimizer`` refuses
ADAMW_FP32 over a bf16 tower -- at 12 B/param against the 20 B/param v5's master recipe
measured. A "fits" from that row was a fit for a run nobody could launch, in the direction
that OOMs.

Torch-gated (``build_optimizer`` constructs real optimizers)::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project \\
        --python /Users/bharath/.venvs/ml/bin/python --with pytest \\
        python -m pytest python/tests/test_memory_budget.py -o addopts= -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import memory_budget  # noqa: E402

from qd_train.memory import ADAMW_FP32, ADAMW_KAHAN, ADAMW_MASTER  # noqa: E402
from qd_train.optim import (  # noqa: E402
    KahanBf16AdamW,
    MasterWeightAdamW,
    build_optimizer,
)

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
BUILT = {
    "master": MasterWeightAdamW,
    "kahan": KahanBf16AdamW,
    "bf16": torch.optim.AdamW,
    "fp32": torch.optim.AdamW,
}


def test_every_budgeted_recipe_is_one_the_trainer_builds() -> None:
    """Each row's optimizer spec, over parameters of the row's dtype, goes through the one
    builder every training step uses. A row it refuses is a budget for nothing."""
    assert set(memory_budget.RECIPES) == set(BUILT)
    for name, (_label, kw) in memory_budget.RECIPES.items():
        p = torch.nn.Parameter(torch.zeros(16, dtype=DTYPES[str(kw["param_dtype"])]))
        # 10 steps: under bf16's 384-step freeze, so the plain bf16 row is about its layout,
        # not about the schedule guard that has tests of its own.
        opt = build_optimizer([p], spec=kw["optimizer"], lr=1e-4, total_steps=10)
        assert type(opt) is BUILT[name], name


def test_the_old_default_layout_is_one_nothing_builds() -> None:
    """Why the old row is gone, stated as the refusal it met."""
    p = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="layout nothing builds"):
        build_optimizer([p], spec=ADAMW_FP32, lr=1e-4, total_steps=10)
    for _label, kw in memory_budget.RECIPES.values():
        assert not (kw["optimizer"] is ADAMW_FP32 and kw["param_dtype"] == "bf16")


def test_the_default_verdict_budgets_the_recipe_v5_trained_with() -> None:
    assert memory_budget.DEFAULT_RECIPE == "master"
    _label, kw = memory_budget.RECIPES["master"]
    assert kw["optimizer"] is ADAMW_MASTER
    assert ADAMW_MASTER.bytes_per_param(param_bytes=2, grad_bytes=2) == 20


def test_the_gpu_footprint_probe_measures_the_same_specs_it_budgets() -> None:
    """``tools/gh200_footprint.py`` is how a box checks these budgets. It used to build its
    own copy of the master spec; it now takes each one from ``qd_train.memory``, and the
    state it expects to find includes the kahan recipe's compensation buffer."""
    import gh200_footprint

    for name, spec in gh200_footprint.RECIPES.items():
        assert spec is memory_budget.RECIPES[name][1]["optimizer"], name
    assert gh200_footprint.described_state_bytes(gh200_footprint.RECIPES["kahan"]) == 10
    assert gh200_footprint.described_state_bytes(gh200_footprint.RECIPES["master"]) == 8
    p = torch.nn.Parameter(torch.zeros(64, dtype=torch.bfloat16))
    opt = build_optimizer([p], spec=gh200_footprint.RECIPES["kahan"], lr=1e-4, total_steps=10)
    # State is allocated at the first step, as torch's AdamW allocates its moments; the
    # probe reads it after its steps (gh200_footprint measures the second step's peak).
    p.grad = torch.ones_like(p)
    opt.step()
    held = sum(
        v.numel() * v.element_size()
        for s in opt.state.values()
        for v in s.values()
        if torch.is_tensor(v)
    )
    assert held / p.numel() == gh200_footprint.described_state_bytes(ADAMW_KAHAN)


def test_the_tool_runs_and_names_its_verdict_recipe(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["memory_budget.py", "--width", "2048", "--device-gb", "80", "--require-fit", "80"],
    )
    code = memory_budget.main()
    out = capsys.readouterr().out
    assert "verdict recipe: master" in out
    for label, _kw in memory_budget.RECIPES.values():
        assert label in out
    assert code == 0, out[-2000:]
