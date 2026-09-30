"""``tools/real_ft_run.py`` prints a flushed progress line while an arm trains.

The GH200 hour-0 throughput run of 2026-09-30 trained for 25 minutes with an empty log:
stdout was a file, so block-buffered, and the loop reported nothing. ``ProgressLine`` is the
``on_progress`` it now passes, throttled so a ~1 s step does not print a line per step.
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

from qd_train.trainer import Progress  # noqa: E402


def _p(step: int, elapsed: float) -> Progress:
    return Progress(
        optimizer_step=step, total_steps=100, micro_batches=step, supervised_tokens=step,
        total_positions=1000 * step, elapsed_s=elapsed, loss=1.5,
    )


def test_the_first_step_prints_and_later_steps_are_throttled() -> None:
    lines: list[str] = []
    line = real_ft_run.ProgressLine("epoch cuda seed=0", every_s=60.0, emit=lines.append)
    for step, t in [(1, 2.0), (2, 30.0), (3, 61.9), (4, 62.0), (5, 100.0), (6, 122.0)]:
        line(_p(step, t))
    assert [ln.split("step ")[1].split(" ")[0] for ln in lines] == ["1/100", "4/100", "6/100"]


def test_the_line_carries_rate_eta_loss_and_peak() -> None:
    lines: list[str] = []
    line = real_ft_run.ProgressLine(
        "epoch cuda seed=0", peak_bytes=lambda: 44 * (1 << 30), emit=lines.append
    )
    line(_p(10, 20.0))
    assert lines == [
        "  progress epoch cuda seed=0: step 10/100 elapsed 20s 2.00s/step 500 pos/s "
        "eta 180s loss 1.5000 peak 44.0GiB"
    ]


@pytest.mark.parametrize("every_s", [0.0, -1.0, float("nan"), float("inf")])
def test_a_meaningless_interval_is_refused(every_s: float) -> None:
    with pytest.raises(ValueError, match="every_s"):
        real_ft_run.ProgressLine("x", every_s=every_s)


def test_the_trainer_is_handed_a_progress_line() -> None:
    """Source check: the one ``train_ft`` call in the tool passes ``on_progress``."""
    src = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    call = src[src.index("result = train_ft(") :]
    call = call[: call.index("\n    )\n")]
    assert "on_progress=ProgressLine(" in call
