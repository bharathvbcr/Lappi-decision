"""``tools/mac_zero_shot.py`` runs on cuda, priced, so rung 1 is re-measured on the box.

The rung-1 row of record could not come from MPS: torch bf16 on MPS drifts ~0.8 nats on the
letter logits (GAP-TORCH-MPS-BF16-LETTER-LOGITS-DRIFT). The tool took ``mps`` or ``cpu`` only,
so phase 0 had no way to take it on the GH200.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _help() -> str:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    done = subprocess.run(
        [sys.executable, str(REPO / "tools" / "mac_zero_shot.py"), "--help"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_cuda_is_a_device_and_it_is_priced() -> None:
    text = _help()
    assert "{mps,cpu,cuda}" in text
    assert "--usd-per-hour" in text and "--instance" in text


def test_a_cuda_run_without_a_rate_is_refused_by_the_cost_estimate() -> None:
    """What the cuda branch builds, with the rate left out: refused, never priced at zero."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "python"))
    import real_ft_run as ft

    with pytest.raises((ValueError, SystemExit)):
        ft._cost(device="cuda", n_gpus=1, usd_per_hour=None, instance=None, cap_s=1800.0)
