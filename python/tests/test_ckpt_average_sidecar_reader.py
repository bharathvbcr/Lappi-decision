"""ckpt_average reads a checkpoint sidecar through qd_train.memory (ft-e0211ba7412).

Its own header read (the third private one) had no offset, dtype, overlap or truncation check.
Held to the same shared list as every other reader:
``crates/qd-export/tests/fixtures/safetensors-adversarial.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
from test_safetensors_one_reader import (  # noqa: E402
    ALL,
    _build,
    _ids,
    _sidecar_of,
    _write_checkpoint,
)


@pytest.mark.parametrize("case", ALL, ids=_ids(ALL))
def test_the_averagers_sidecar_open_gives_the_shared_verdict(case, tmp_path):
    ck = _write_checkpoint(tmp_path)
    _build(case, _sidecar_of(ck))
    if case["expect"] == "ok":
        ckpt_average._SidecarTensors(ck)
    else:
        with pytest.raises(ckpt_average.AverageRefusal):
            ckpt_average._SidecarTensors(ck)
