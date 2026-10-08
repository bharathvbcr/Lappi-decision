"""``tools/gh200_footprint.py``: where it writes, and the shape it says it measured.

Before ``--out`` the probe always overwrote ``/home/ubuntu/gh200_footprint_<optimizer>.json``
and recorded no shape, so a second run silently replaced the first one's measurement and a
reader could not tell whether two arms were probed at the same rows, widths, loss and
checkpointing. P1's reading rule compares the master and kahan peaks "at the same shape".

CPU only: the device is faked at ``torch.cuda`` and the allocator measurement at its seam,
``measure_widths``; the tower is a tiny module standing in for ``load_text_tower``'s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import gh200_footprint as gf  # noqa: E402


def _tower(**over: Any) -> Any:
    fields = dict(
        model=torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16), vocab_size=16, hidden_size=4,
        n_tensors_loaded=1, gradient_checkpointing=True, checkpoint_skip_layers=(),
        attn_implementation="sdpa", dtype="bf16",
    )
    fields.update(over)
    return SimpleNamespace(**fields)


def _fake_device(mp: pytest.MonkeyPatch, *, tower: Any | None = None,
                 measured: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Fake CUDA and the loader; returns the list the seam's calls are logged into."""
    calls: list[dict[str, Any]] = []
    mp.setattr(gf.torch.cuda, "is_available", lambda: True)
    mp.setattr(gf.torch.cuda, "get_device_properties",
               lambda i: SimpleNamespace(name="FAKE GH200", total_memory=96 * gf.GiB,
                                         major=9, minor=0))
    mp.setattr(gf, "load_text_tower", lambda *a, **k: tower if tower is not None else _tower())

    def measure(tower, opt, spec, *, rows, widths):
        calls.append({"rows": rows, "widths": widths, "spec": spec.name})
        return measured if measured is not None else [
            {"width": w, "rows": rows, "measured_bytes": 10 * gf.GiB + w} for w in widths
        ]

    mp.setattr(gf, "measure_widths", measure)
    return calls


def test_out_writes_the_shape_it_measured(tmp_path, monkeypatch):
    calls = _fake_device(monkeypatch)
    out = tmp_path / "fp_master.json"
    assert gf.main(["--optimizer", "bf16", "--out", str(out)]) == 0
    body = json.loads(out.read_text())
    assert body["recipe"] == "bf16"
    assert body["shape"] == {
        "rows": 1, "widths": list(gf.WIDTHS), "loss": gf.LOSS,
        "gradient_checkpointing": True, "checkpoint_skip_layers": [],
        "attn_implementation": "sdpa", "dtype": "bf16", "snapshot": str(gf.SNAPSHOT),
        "peak": gf.PEAK,
    }
    assert calls == [{"rows": 1, "widths": gf.WIDTHS, "spec": gf.RECIPES["bf16"].name}]
    assert [r["width"] for r in body["results"]] == list(gf.WIDTHS)


def test_the_shape_is_read_off_the_tower_not_restated(tmp_path, monkeypatch):
    _fake_device(monkeypatch, tower=_tower(gradient_checkpointing=False,
                                           checkpoint_skip_layers=(3, 7)))
    out = tmp_path / "fp.json"
    gf.main(["--out", str(out)])
    shape = json.loads(out.read_text())["shape"]
    assert shape["gradient_checkpointing"] is False
    assert shape["checkpoint_skip_layers"] == [3, 7]


def test_an_existing_out_is_refused_before_the_tower_loads(tmp_path, monkeypatch):
    out = tmp_path / "fp.json"
    out.write_text("the first probe's measurement")

    def boom(*a, **k):
        raise AssertionError("the tower was loaded for a run whose output was refused")

    _fake_device(monkeypatch)
    monkeypatch.setattr(gf, "load_text_tower", boom)
    with pytest.raises(SystemExit, match="already exists"):
        gf.main(["--out", str(out)])
    assert out.read_text() == "the first probe's measurement"


def test_an_out_that_appears_during_the_probe_is_still_not_overwritten(tmp_path, monkeypatch):
    """The final write refuses too (write_text_atomic), not only the early check."""
    out = tmp_path / "fp.json"
    _fake_device(monkeypatch)
    real = gf.measure_widths

    def measure_then_race(*a, **k):
        out.write_text("a concurrent probe")
        return real(*a, **k)

    monkeypatch.setattr(gf, "measure_widths", measure_then_race)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        gf.main(["--out", str(out)])
    assert out.read_text() == "a concurrent probe"
    assert [p.name for p in tmp_path.iterdir()] == ["fp.json"], "no temp file left behind"


def test_without_out_the_default_path_is_written_and_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "DEFAULT_OUT", str(tmp_path / "gh200_footprint_{optimizer}.json"))
    _fake_device(monkeypatch)
    target = tmp_path / "gh200_footprint_bf16.json"
    target.write_text("old")
    gf.main([])
    assert json.loads(target.read_text())["recipe"] == "bf16"


def test_no_cuda_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(gf.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="no CUDA device"):
        gf.main(["--out", str(tmp_path / "fp.json")])
    assert not (tmp_path / "fp.json").exists()
