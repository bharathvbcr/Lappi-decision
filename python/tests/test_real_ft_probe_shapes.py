"""``tools/real_ft_run.py --probe-shapes MARGIN_GIB``: the 2x H100 memory probe.

The human chose a 2x H100 (80 GB) box for v5 on 2026-10-03. Fable's ruling: before seed 0,
step the real recipe once on every distinct ``(rows, width)`` batch shape of v5's plan,
costliest first, and record the allocator's peaks on a quick ft row. ``qd_train.memory``
prices the narrowest bucket with the most rows above the widest
(AUDIT/finalize-2026-10-03/v5_h100_budget.py reproduces F's recorded estimate to the byte),
so a probe of the widest bucket alone would miss the costliest step. These tests pin:

* ``real_ft_run.probe_shape_order`` keeps one batch per shape, the first of each, costliest first;
* ``real_ft_run.MemoryProbe`` fails before step 1, passes only when every shape stepped within
  budget, and on a probe cut short says how far it got and which shape was next;
* the recipe key and the rule-8 reason appear only when the probe is on;
* argv refuses every flag a probe would record without it determining anything;
* end to end on the CPU stand-in, with torch.cuda's readers faked: the plan is cut to the
  distinct shapes, the ft row carries the metric, the key and the reason, and a probe over
  budget exits non-zero after its row is written. Off cuda, unfaked, it is refused before
  any row.

Torch-gated like ``test_real_ft_rungd_flags.py``, whose CPU corpus and runner these reuse.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_rungd_flags import (  # noqa: E402
    _OFF,
    REV,
    _only,
    _run,
    _tripwire,
    corpus,  # noqa: F401  (the module-scoped fixture, used by name)
)

from qd_train.ledger import Ledger  # noqa: E402

GIB = 1 << 30


def _batch(rows: int, width: int) -> Any:
    return SimpleNamespace(tokens=np.zeros((rows, width), dtype=np.uint32))


# --- 1. the probe's plan --------------------------------------------------------------------


def test_one_batch_per_shape_the_first_of_each_costliest_first():
    plan = [_batch(4, 100), _batch(3, 200), _batch(4, 100), _batch(258, 137), _batch(3, 200),
            _batch(2, 300), _batch(1, 600)]
    # Positions: 4x100=400, 3x200=600, 258x137=35,346, 2x300=600, 1x600=600. Most positions
    # first; among the three 600s, widest first.
    assert rft.probe_shape_order(plan) == [3, 6, 5, 1, 0]


def test_a_plan_of_one_shape_probes_one_batch():
    assert rft.probe_shape_order([_batch(3, 64)] * 5) == [0]


# --- 2. the metric ------------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}

    def metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value


def _progress(step: int, total: int) -> Any:
    return rft.Progress(optimizer_step=step, total_steps=total, micro_batches=step,
                        supervised_tokens=0, total_positions=0, elapsed_s=1.0, loss=1.0)


def _probe(peaks: list[tuple[int, int]], *, shapes=((258, 137), (3, 10240)),
           margin_gib: float = 2.0, total_gib: float = 80.0) -> tuple[Any, _Recorder]:
    rec = _Recorder()
    feed = iter(peaks)
    probe = rft.MemoryProbe(
        rec, shapes=list(shapes), margin_bytes=int(margin_gib * GIB),
        total_bytes=int(total_gib * GIB), read_peaks=lambda: next(feed),
    )
    return probe, rec


def _state(rec: _Recorder) -> dict[str, Any]:
    return rec.metrics[rft.MemoryProbe.METRIC].to_json()


def test_the_metric_fails_before_any_step_and_names_the_first_shape():
    probe, rec = _probe([])
    state = _state(rec)
    assert state["passed"] is False and state["n"] == 0 and state["n_total"] == 2
    assert "none yet, first 258x137" in state["detail"]
    assert probe.passed is False


def test_a_probe_passes_only_when_every_shape_stepped_within_budget():
    probe, rec = _probe([(60 * GIB, 70 * GIB), (61 * GIB, 72 * GIB)])
    probe(_progress(1, 2))
    assert probe.passed is False, "one of two shapes is not a probe"
    assert "next 3x10240" in _state(rec)["detail"]
    probe(_progress(2, 2))
    state = _state(rec)
    assert probe.passed is True and state["passed"] is True
    assert state["value"] == 72 * GIB and state["n"] == state["n_total"] == 2
    assert "max_memory_reserved 72.00 GiB" in state["detail"]


def test_a_reserved_peak_past_the_margin_fails_even_when_every_shape_stepped():
    probe, rec = _probe([(60 * GIB, 70 * GIB), (61 * GIB, int(78.5 * GIB))])
    probe(_progress(1, 2))
    probe(_progress(2, 2))
    assert probe.passed is False and _state(rec)["passed"] is False


def test_the_budget_is_reserved_not_allocated():
    """Allocated well inside, reserved past the line: the caching allocator holds reserved."""
    probe, _ = _probe([(50 * GIB, 79 * GIB)], shapes=((3, 10240),))
    probe(_progress(1, 1))
    assert probe.passed is False


@pytest.mark.parametrize(
    ("shapes", "margin_gib", "total_gib", "match"),
    [((), 2.0, 80.0, "at least one batch shape"),
     (((3, 64),), 80.0, 80.0, "leaves no budget"),
     (((3, 64),), 0.0, 80.0, "leaves no budget")],
)
def test_a_probe_with_nothing_to_measure_is_refused(shapes, margin_gib, total_gib, match):
    with pytest.raises(ValueError, match=match):
        _probe([], shapes=shapes, margin_gib=margin_gib, total_gib=total_gib)


# --- 3. the recipe and rule 8 -----------------------------------------------------------------


def test_the_probe_adds_its_recipe_key_only_when_on():
    assert rft._recipe_pieces(**_OFF) == {}
    assert rft._recipe_pieces(**_OFF, probe_shapes=2.0) == {"probe_shapes_margin_gib": 2.0}
    assert "probe_shapes_margin_gib" in rft.RECIPE_PIECE_KEYS


def test_a_probe_row_is_quick_and_says_why():
    clean = rft.CorpusFacts(snapshot_not_run=None, history_rows={})
    kw: dict[str, Any] = {"tag": "epoch", "device": "cuda", "real_backbone": True,
                          "corpus": clean}
    assert rft.quick_reasons(**kw) == []
    reasons = rft.quick_reasons(**kw, probe_shapes=2.0)
    assert reasons == [rft.probe_shapes_reason(2.0)]
    assert "memory probe (--probe-shapes 2)" in reasons[0]


# --- 4. argv --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--epoch", "--no-memorise", "--probe-shapes", "0"], "finite and positive"),
        (["--epoch", "--no-memorise", "--probe-shapes", "nan"], "finite and positive"),
        (["--probe-shapes", "2"], "give --epoch and --no-memorise"),
        (["--epoch", "--probe-shapes", "2"], "give --epoch and --no-memorise"),
        (["--epoch", "--no-memorise", "--probe-shapes", "2", "--max-steps", "5"],
         "--max-steps"),
        (["--epoch", "--no-memorise", "--probe-shapes", "2", "--checkpoint-dir", "c"],
         "--checkpoint-dir"),
        (["--epoch", "--no-memorise", "--probe-shapes", "2", "--retain-tower-every", "10"],
         "--retain-tower-every"),
        (["--epoch", "--no-memorise", "--probe-shapes", "2", "--noul-weight", "4"],
         "--noul-weight"),
        (["--epoch", "--no-memorise", "--probe-shapes", "2", "--needle"], "--needle"),
        (["--epoch", "--no-memorise", "--probe-shapes", "2", "--score-val",
          "--shuffled-label", "abcdefgh"], "--shuffled-label"),
    ],
)
def test_a_probe_flag_that_would_record_something_it_did_not_determine_is_refused(
    flags, match, tmp_path, monkeypatch
):
    monkeypatch.setattr(rft, "ShardReader", _tripwire)
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), "--rev", REV, "--seeds", "0", "--devices", "cpu",
                  *flags])


# --- 5. end to end on the CPU stand-in, torch.cuda's readers faked ---------------------------


def _fake_cuda(mp: pytest.MonkeyPatch, *, reserved_gib: float) -> None:
    mp.setattr(rft, "PROBE_DEVICES", ("cuda", "cpu"))
    mp.setattr(rft.torch.cuda, "get_device_properties",
               lambda device: SimpleNamespace(total_memory=80 * GIB))
    mp.setattr(rft.torch.cuda, "max_memory_allocated", lambda *a, **k: 40 * GIB)
    mp.setattr(rft.torch.cuda, "max_memory_reserved",
               lambda *a, **k: int(reserved_gib * GIB))


def test_a_probe_steps_each_shape_once_and_records_its_peak_on_a_quick_row(
    corpus, tmp_path, monkeypatch  # noqa: F811
):
    _fake_cuda(monkeypatch, reserved_gib=50.0)
    ledger = tmp_path / "l.jsonl"
    ft = _only(_run(corpus, ledger, "--probe-shapes", "2", short=True), "ft")
    probe = ft.metrics[rft.MemoryProbe.METRIC].to_json()
    assert probe["passed"] is True and probe["n"] == probe["n_total"] >= 1
    assert ft.metrics["corpus.plan_batches"].to_json()["value"] == probe["n_total"]
    assert ft.recipe is not None and ft.recipe["probe_shapes_margin_gib"] == 2.0
    assert ft.quick and rft.probe_shapes_reason(2.0) in (ft.quick_reason or "")


def test_a_probe_over_budget_exits_non_zero_after_writing_its_row(
    corpus, tmp_path, monkeypatch  # noqa: F811
):
    _fake_cuda(monkeypatch, reserved_gib=79.0)
    ledger = tmp_path / "l.jsonl"
    with pytest.raises(SystemExit, match="memory probe did not pass"):
        _run(corpus, ledger, "--probe-shapes", "2", short=True)
    ft = _only(Ledger(ledger).rows(), "ft")
    assert ft.metrics[rft.MemoryProbe.METRIC].to_json()["passed"] is False


def test_a_probe_off_cuda_is_refused_before_any_row(corpus, tmp_path):  # noqa: F811
    ledger = tmp_path / "l.jsonl"
    with pytest.raises(ValueError, match="probe_shapes refuses device 'cpu'"):
        _run(corpus, ledger, "--probe-shapes", "2")
    assert not ledger.exists() or Ledger(ledger).rows() == []
