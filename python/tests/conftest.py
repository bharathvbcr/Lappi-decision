"""Fixtures more than one test module needs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
#: ``None`` where cargo is absent; the modules that need the probe skip on it.
CARGO = shutil.which("cargo")


@pytest.fixture(scope="session")
def probe() -> Path:
    """``qd-margin-probe``, built once per session from this checkout."""
    if CARGO is None:
        pytest.skip("cargo is not on PATH: the probe was not built")
    subprocess.run(
        [CARGO, "build", "--quiet", "--manifest-path", str(REPO / "Cargo.toml"),
         "-p", "qd-runtime", "--bin", "qd-margin-probe"],
        check=True, timeout=900,
    )
    binary = REPO / "target" / "debug" / "qd-margin-probe"
    assert binary.is_file(), binary
    return binary


@pytest.fixture(scope="session")
def noul_rows_bin() -> Path:
    """``qd-noul-rows``, built once per session from this checkout."""
    if CARGO is None:
        pytest.skip("cargo is not on PATH: qd-noul-rows was not built")
    subprocess.run(
        [CARGO, "build", "--quiet", "--manifest-path", str(REPO / "Cargo.toml"),
         "-p", "qd-mutate", "--bin", "qd-noul-rows"],
        check=True, timeout=900,
    )
    binary = REPO / "target" / "debug" / "qd-noul-rows"
    assert binary.is_file(), binary
    return binary
