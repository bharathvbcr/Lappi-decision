"""``tools/real_ft_run.py`` writes the finished arm's checkpoint, not only interval ones.

The GH200 campaign of 2026-09-30 derived ``--checkpoint-every 1747`` for a 1,505-step epoch:
the loop's interval never fired, nothing wrote the end state, and all three phase-3 seeds
trained to completion without saving a weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run  # noqa: E402


class _Ckpt:
    def __init__(self, step: int, log: list[int]) -> None:
        self.optimizer_step = step
        self._log = log

    def write(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(self.optimizer_step), encoding="utf-8")
        self._log.append(self.optimizer_step)
        return target


def test_an_arm_shorter_than_the_interval_still_saves_its_end_state(tmp_path: Path) -> None:
    written: list[int] = []
    sink = real_ft_run.CheckpointSink(tmp_path / "ckpt" / "epoch-seed0-cuda.json")
    sink.final(_Ckpt(1505, written))
    assert written == [1505]
    assert (tmp_path / "ckpt" / "epoch-seed0-cuda.json").read_text() == "1505"


def test_the_end_state_is_not_written_twice_when_the_last_interval_took_it(
    tmp_path: Path,
) -> None:
    written: list[int] = []
    sink = real_ft_run.CheckpointSink(tmp_path / "c.json")
    sink(_Ckpt(100, written))
    sink(_Ckpt(200, written))
    sink.final(_Ckpt(200, written))
    assert written == [100, 200]
    sink.final(_Ckpt(250, written))
    assert written == [100, 200, 250]


def test_the_trainer_writes_the_end_state() -> None:
    src = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert "on_checkpoint.final(result.checkpoint)" in src
