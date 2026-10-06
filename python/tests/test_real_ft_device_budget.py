"""``tools/real_ft_run.py`` checks memory.py's estimate against the device before training.

The residual of GAP-MPS-ALLOCATOR-CACHE-SWAPS-INSTEAD-OF-REFUSING: nothing compared the
estimate to the device, so a run that could not fit started anyway. ``device_budget`` is the
comparison; ``_train`` refuses on a failed one before the first step and records the result on
the ft row either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run  # noqa: E402
from test_memory import _snapshot  # noqa: E402

from qd_train.tristate import NotRun, Ran  # noqa: E402

GIB = 1024**3


def _batch(rows: int, width: int, index: int = 0):
    import numpy as np

    from qd_train.artifacts import Batch

    return Batch(
        tokens=np.ones((rows, width), dtype=np.int32),
        lengths=np.full(rows, width, dtype=np.int32),
        bucket=0, index=index,
    )


def test_the_budget_is_checked_at_shapes_the_plan_actually_has() -> None:
    """--batch-tokens 32768 on the GH200 (2026-09-30) was refused at 105 GiB: the estimate
    paired the plan's most rows (~163 short rows) with its widest width (1,625), a batch of
    ~265k positions that no plan contains -- every real one is at most 32,768."""
    plan = [_batch(163, 200, 0), _batch(20, 1625, 1), _batch(163, 200, 2)]
    assert real_ft_run._batch_shapes(plan) == [(20, 1625), (163, 200)]


def test_the_trainer_budgets_the_worst_real_shape() -> None:
    src = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert "rows=max(int(b.tokens.shape[0]) for b in plan)" not in src
    assert "footprint_at(tower, rows=r, width=w) for r, w in _batch_shapes(plan)" in src


def test_cuda_over_budget_is_a_failed_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10 * GIB, 96 * GIB))
    got = real_ft_run.device_budget(20 * GIB, "cuda")
    assert isinstance(got, Ran) and got.passed is False
    assert got.value == 20 * GIB
    assert f"against {10 * GIB} B" in got.detail and "mem_get_info" in got.detail


def test_cuda_within_budget_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (90 * GIB, 96 * GIB))
    got = real_ft_run.device_budget(40 * GIB, "cuda")
    assert isinstance(got, Ran) and got.passed is True


def test_mps_reads_the_recommended_working_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 48 * GIB)
    assert real_ft_run.device_budget(60 * GIB, "mps").passed is False
    assert real_ft_run.device_budget(30 * GIB, "mps").passed is True


def test_cpu_is_not_checked_and_says_so() -> None:
    got = real_ft_run.device_budget(1, "cpu")
    assert isinstance(got, NotRun) and "cpu" in got.reason


def test_train_refuses_a_failed_budget_before_building_the_step() -> None:
    """The call site, read from source: the refusal sits between the remap and the step, and
    the result reaches the ft row. A source check, because exercising it needs the real
    tower -- the behaviour of the check itself is covered above."""
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    remap = source.index("tower = remap_text_tower(tower, reader.remap)")
    check = source.index("budget = device_budget(tower.footprint.total_bytes, device)")
    step = source.index("step = QwenDecisionStep(")
    assert remap < check < step
    assert 'raise SystemExit(f"device budget: {budget.detail}")' in source
    assert 'recorder.metric("device_budget", budget)' in source


class _Loaded(Exception):
    """Stops ``_real_step`` at the load, carrying what the load was asked for."""


def _capture_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import qd_train.backbone as backbone

    def fake_load(snapshot, **kwargs):
        raise _Loaded(kwargs)

    monkeypatch.setattr(backbone, "load_text_tower", fake_load)


def _real_step_on(snapshot: Path) -> None:
    from types import SimpleNamespace

    from qd_train.memory import ADAMW_KAHAN

    real_ft_run._real_step(
        backbone=snapshot, reader=SimpleNamespace(remap=None),
        plan=[SimpleNamespace(tokens=torch.zeros((1, 8)))], device="cpu", dtype="bf16",
        spec=ADAMW_KAHAN, attn_implementation="sdpa", seed=0, lr=1e-5, total_steps=10,
        span_weight=1.0, width=8,
    )


def test_the_real_step_budgets_the_base_on_disk_not_the_2b(tmp_path, monkeypatch) -> None:
    """DevMap audit #3: ``_real_step`` loaded every backbone with the default spec, the
    2B's, so any other base was refused at load (or, had the check been weaker, budgeted
    as a 2B). It now hands the load the spec measured from the snapshot's own headers."""
    _capture_load(monkeypatch)
    with pytest.raises(_Loaded) as caught:
        _real_step_on(_snapshot(tmp_path, top_tie=True))
    spec = caught.value.args[0]["spec"]
    assert spec.name == "Qwen/Qwen3.5-4B-Base (text tower)"
    assert (spec.hidden_size, spec.linear_value_heads) == (2560, 32)


def test_the_real_step_refuses_a_base_whose_head_would_be_a_guess(tmp_path, monkeypatch) -> None:
    _capture_load(monkeypatch)
    with pytest.raises(SystemExit, match="no top-level"):
        _real_step_on(_snapshot(tmp_path, top_tie=None))


def test_the_sidecar_preflight_sits_before_any_step_and_only_where_full_checkpoints_go() -> None:
    """The call site, read from source, as the device budget's is above: after the step is
    built and before training, under exactly the condition that builds a full-checkpoint
    sink. The arithmetic itself is checked against a real step in test_backbone.py."""
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    built = source.index("step, tower, budget = _real_step(")
    check = source.index("checkpoint_sidecar_bytes(step, spec)")
    trained = source.index("result = train_ft(")
    sink = source.index("on_checkpoint = CheckpointSink(")
    assert built < check < sink < trained
    guard = "if checkpoint_every and checkpoint_dir is not None:"
    assert source[source.rindex(guard, 0, check):check].count("\n") <= 3
    assert source[source.rindex(guard, 0, sink):sink].count("\n") <= 1
