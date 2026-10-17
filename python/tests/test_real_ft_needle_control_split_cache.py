"""``--needle-control`` reads the split through ``--split-cache`` when given one (ft-505105265aa).

The J6 shape trains in one process and runs ``--needle-control`` in a second, which re-imports,
rebuilds the whole split (~200 s on the v5 data, the GPU idle) and only then loads the model.
The tool already keys a cache for every start; the second process just was not handed one. The
smaller and safer fix over folding the control into the train+score process (a different row,
recipe and exit path) is to hand it ``--split-cache``, and this pins that the control's argv is
accepted with the flag and that its split read goes through the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run  # noqa: E402
import split_cache  # noqa: E402
from test_real_ft_score_checkpoint import tiny_master_checkpoints  # noqa: E402
from test_real_ft_shuffled_label import REV  # noqa: E402


class _CacheConsulted(Exception):
    pass


def test_a_needle_control_process_reads_its_split_through_the_cache(tmp_path, monkeypatch):
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    seen: dict[str, object] = {}

    def consulted(directory, **kwargs):
        seen["directory"] = directory
        seen["rebuild"] = kwargs["rebuild"]
        raise _CacheConsulted

    monkeypatch.setattr(split_cache, "cached_split_rows", consulted)
    with pytest.raises(_CacheConsulted):
        real_ft_run.main([
            "--out", str(tiny.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
            "--score-val", "--needle", "--needle-control", "256",
            "--score-checkpoint", str(tiny.paths[0]), "--real-backbone", str(tiny.snapshot),
            "--ft-ledger", str(tiny.ft_ledger), "--ft-row-id", tiny.ids[0],
            "--ledger", str(tmp_path / "eval.jsonl"), "--tokenizer-json", str(tiny.tokenizer),
            "--split-cache", str(cache),
        ])
    assert seen["directory"] == cache
    assert callable(seen["rebuild"])
