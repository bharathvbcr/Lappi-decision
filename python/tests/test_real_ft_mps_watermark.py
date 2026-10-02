"""``tools/real_ft_run.py`` caps the MPS allocator's cache before torch initialises it.

The full-vocabulary smoke of 2026-09-29 held 16 GiB of live tensors and a 61 GiB footprint
on a 64 GiB Mac: torch's default watermarks (1.7 / 1.4 of the recommended working set) let
the caching allocator grow into swap, and the run paged for 58 minutes. The tool now sets
1.0 / 0.9 unless the caller set a value. In a subprocess, because the variables only matter
before the first import of torch, and this test process has imported it already.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
KEYS = ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")
PROBE = (
    "import os, sys; sys.argv = ['real_ft_run.py']; "
    f"sys.path.insert(0, {str(REPO / 'tools')!r}); import real_ft_run; "
    f"print(','.join(os.environ.get(k, '') for k in {KEYS!r}))"
)


def _import_and_read(env: dict[str, str]) -> str:
    done = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip().splitlines()[-1]


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in KEYS}


def test_the_tool_caps_the_mps_cache_when_the_caller_did_not() -> None:
    pytest.importorskip("torch")
    assert _import_and_read(_clean_env()) == "1.0,0.9"


def test_an_explicit_value_wins() -> None:
    pytest.importorskip("torch")
    env = {**_clean_env(), KEYS[0]: "0.5", KEYS[1]: "0.4"}
    assert _import_and_read(env) == "0.5,0.4"
