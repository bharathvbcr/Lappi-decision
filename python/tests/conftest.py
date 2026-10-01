"""Fixtures more than one test module needs."""

from __future__ import annotations

import os
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


#: ``tools/real_tokenizer_pipeline.PREP_BIN_ENV``; ``test_qd_prep_minhash_parity.py`` asserts
#: the two agree.
QD_PREP_BIN_ENV = "QD_PREP_BIN"


@pytest.fixture(scope="session")
def qd_prep_bin() -> Path:
    """``qd-prep``, which MinHash signing runs in: the one ``QD_PREP_BIN`` names, else built
    once per session (release) from this checkout. Without either, SKIPPED with the reason --
    a dedupe that could not run is not a dedupe that passed."""
    named = os.environ.get(QD_PREP_BIN_ENV, "")
    if named:
        return Path(named)
    if CARGO is None:
        pytest.skip(f"cargo is not on PATH and {QD_PREP_BIN_ENV} is unset: qd-prep was not built")
    subprocess.run(
        [CARGO, "build", "--quiet", "--release", "--manifest-path", str(REPO / "Cargo.toml"),
         "-p", "qd-prep", "--bin", "qd-prep"],
        check=True, timeout=900,
    )
    binary = REPO / "target" / "release" / "qd-prep"
    assert binary.is_file(), binary
    return binary


@pytest.fixture
def qd_prep(qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``QD_PREP_BIN`` set to :func:`qd_prep_bin` for one test, so dedupe and split sign in
    qd-prep (``real_tokenizer_pipeline.native_minhash``); there is no Python fallback."""
    monkeypatch.setenv(QD_PREP_BIN_ENV, str(qd_prep_bin))
    return qd_prep_bin


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
