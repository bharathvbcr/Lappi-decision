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

from qd_train.tristate import NotRun, Ran  # noqa: E402

GIB = 1024**3


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
