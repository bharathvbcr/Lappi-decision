"""``tools/real_ft_run.py``: per-step wall time on the ft row (``StepTimer``).

campaign/next-train-first-box-2026-10-06.DRAFT.json P1 reads "the optimizer step time
(median of steps 101-600)". Before this, the tool recorded only a whole-run ``steps_per_s``
and a cumulative progress line, neither of which is a per-step median. These tests pin:

* each interval and each ``apply`` is closed by the device sync, so the time covers the
  queued device work and not the launch;
* the metric is ``NotRun`` until every step of the window was timed, so a short, capped
  or resumed run never records a median of part of the window;
* only the window's steps are kept (bounded whatever the schedule's length);
* a checkpoint write is in no step's interval;
* end to end on the CPU stand-in and the tiny real tower, the ft row carries both metrics.

Torch-gated like ``test_real_ft_probe_shapes.py``, whose corpus and runner these reuse.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_rungd_flags import (  # noqa: E402
    _only,
    _run,
    corpus,  # noqa: F401  (the module-scoped fixture, used by name)
)

from qd_train.tristate import NotRun, Ran  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}

    def metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value


class _Clock:
    """Advances only when told to; every read and every sync is logged in order."""

    def __init__(self) -> None:
        self.t = 0.0
        self.events: list[str] = []

    def __call__(self) -> float:
        self.events.append("clock")
        return self.t

    def sync(self) -> None:
        self.events.append("sync")


class _Step:
    def __init__(self, clock: _Clock, apply_s: float) -> None:
        self.clock = clock
        self.apply_s = apply_s
        self.applied = 0

    def apply(self, *, lr: float) -> None:
        self.clock.t += self.apply_s
        self.applied += 1


def _progress(step: int) -> Any:
    return rft.Progress(optimizer_step=step, total_steps=10, micro_batches=step,
                        supervised_tokens=0, total_positions=0, elapsed_s=1.0, loss=1.0)


def _timer(window=(3, 5), first_step=0) -> tuple[rft.StepTimer, _Recorder, _Clock, _Step]:
    rec, clock = _Recorder(), _Clock()
    timer = rft.StepTimer(rec, sync=clock.sync, sync_name="fake.sync()", window=window,
                          first_step=first_step, clock=clock)
    step = _Step(clock, apply_s=0.25)
    timer.wrap(step)
    timer.start()
    return timer, rec, clock, step


def _drive(timer, clock, step, n: int, *, forward_s=lambda k: 1.0, start: int = 1) -> None:
    for k in range(start, start + n):
        clock.t += forward_s(k)  # forward + backward of step k
        step.apply(lr=1e-3)
        timer(_progress(k))


def test_before_any_step_both_metrics_are_not_run():
    _, rec, _, _ = _timer()
    for name in (rft.StepTimer.STEP_METRIC, rft.StepTimer.APPLY_METRIC):
        assert isinstance(rec.metrics[name], NotRun)
        assert "0 of the 3 steps of window 3-5" in rec.metrics[name].reason


def test_a_covered_window_records_the_median_of_exactly_its_steps():
    timer, rec, clock, step = _timer()
    # Step k's forward+backward takes k seconds; the window is steps 3, 4, 5.
    _drive(timer, clock, step, 8, forward_s=lambda k: float(k))
    whole = rec.metrics[rft.StepTimer.STEP_METRIC]
    apply = rec.metrics[rft.StepTimer.APPLY_METRIC]
    assert isinstance(whole, Ran) and isinstance(apply, Ran)
    assert whole.value == pytest.approx(4.25)  # median of 3.25, 4.25, 5.25
    assert whole.n == whole.n_total == 3
    assert apply.value == pytest.approx(0.25)
    assert "fake.sync()" in whole.detail and "steps 3-5" in whole.detail
    # Bounded: steps outside the window are timed and dropped.
    assert sorted(timer.step_s) == [3, 4, 5] and sorted(timer.apply_s) == [3, 4, 5]


def test_a_run_short_of_the_window_end_records_no_median():
    timer, rec, clock, step = _timer()
    _drive(timer, clock, step, 4)  # steps 1-4: 3 and 4 of window 3-5
    m = rec.metrics[rft.StepTimer.STEP_METRIC]
    assert isinstance(m, NotRun)
    assert "2 of the 3 steps" in m.reason and "no median of a partial window" in m.reason


def test_a_resumed_run_cannot_cover_a_window_it_started_inside():
    timer, rec, clock, step = _timer(first_step=3)
    _drive(timer, clock, step, 4, start=4)  # steps 4-7
    m = rec.metrics[rft.StepTimer.STEP_METRIC]
    assert isinstance(m, NotRun) and "resumed at step 3" in m.reason


def test_every_clock_read_follows_a_sync():
    """The time must cover the device's queued work, not the launch."""
    timer, _, clock, step = _timer()
    _drive(timer, clock, step, 2)
    reads = [i for i, e in enumerate(clock.events) if e == "clock"]
    assert reads and all(clock.events[i - 1] == "sync" for i in reads), clock.events


def test_a_checkpoint_write_is_in_no_steps_interval():
    timer, rec, clock, step = _timer()
    writes: list[int] = []

    def sink(ckpt: Any) -> None:
        clock.t += 100.0  # a slow write
        writes.append(ckpt)

    on_checkpoint = timer.excluding(sink)
    for k in range(1, 6):
        clock.t += 1.0
        step.apply(lr=1e-3)
        timer(_progress(k))
        if k == 3:
            on_checkpoint(k)
    assert writes == [3]
    whole = rec.metrics[rft.StepTimer.STEP_METRIC]
    assert isinstance(whole, Ran) and max(timer.step_s.values()) == pytest.approx(1.25)
    assert "1 interval(s) restarted" in whole.detail
    assert timer.excluding(None) is None


def test_a_step_that_bypassed_the_timed_apply_is_refused():
    rec, clock = _Recorder(), _Clock()
    timer = rft.StepTimer(rec, sync=clock.sync, sync_name="s", window=(1, 2), clock=clock)
    timer.start()
    with pytest.raises(RuntimeError, match="without the timed apply"):
        timer(_progress(1))


@pytest.mark.parametrize("window", [(0, 5), (6, 5)])
def test_a_meaningless_window_is_refused(window):
    with pytest.raises(ValueError, match="window"):
        rft.StepTimer(_Recorder(), sync=lambda: None, sync_name="s", window=window)


def test_the_default_window_is_the_drafts():
    assert rft.STEP_TIME_WINDOW == (101, 600)


def test_each_device_names_its_sync():
    assert rft.device_sync("cuda")[1] == "torch.cuda.synchronize()"
    assert rft.device_sync("cuda")[0] is torch.cuda.synchronize
    assert rft.device_sync("mps")[0] is torch.mps.synchronize
    assert rft.device_sync("cpu")[1].startswith("none")
    with pytest.raises(ValueError, match="no synchronisation"):
        rft.device_sync("tpu")


@pytest.mark.parametrize("real", [False, True], ids=["stand-in", "tiny-real-tower"])
def test_the_ft_row_carries_both_step_time_metrics(corpus, tmp_path, real):  # noqa: F811
    """End to end: a 3-step arm cannot cover 101-600, and its row says so rather than
    carrying a median; the metric names exist on the row at all, which they did not."""
    ft = _only(_run(corpus, tmp_path / "l.jsonl", "--max-steps", "3", real=real, short=True),
               "ft")
    for name in (rft.StepTimer.STEP_METRIC, rft.StepTimer.APPLY_METRIC):
        state = ft.metrics[name].to_json()
        assert state["state"] == "not_run", state
        assert "3 of the 500" not in state["reason"]
        assert "0 of the 500 steps of window 101-600" in state["reason"]


def test_the_ft_row_carries_a_median_when_the_window_is_covered(
    corpus, tmp_path, monkeypatch  # noqa: F811
):
    """The same run with the window moved inside it: a Ran median over exactly its steps."""
    monkeypatch.setattr(rft, "STEP_TIME_WINDOW", (2, 3))
    real_init = rft.StepTimer.__init__

    def init(self, recorder, **kw):
        kw["window"] = rft.STEP_TIME_WINDOW
        real_init(self, recorder, **kw)

    monkeypatch.setattr(rft.StepTimer, "__init__", init)
    ft = _only(_run(corpus, tmp_path / "l.jsonl", "--max-steps", "3", real=True, short=True),
               "ft")
    for name in (rft.StepTimer.STEP_METRIC, rft.StepTimer.APPLY_METRIC):
        state = ft.metrics[name].to_json()
        assert state["state"] == "ran" and state["n"] == state["n_total"] == 2, state
        assert state["value"] > 0.0
    whole = ft.metrics[rft.StepTimer.STEP_METRIC].to_json()["value"]
    apply = ft.metrics[rft.StepTimer.APPLY_METRIC].to_json()["value"]
    assert apply <= whole
