"""Fixtures more than one test module needs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
#: ``None`` where cargo is absent; the modules that need a cargo-built binary skip on it.
CARGO = shutil.which("cargo")

#: Exported by ``tools/mac_heavy.sh`` to the job it runs under the Mac's one-heavy-job lock.
#: On darwin a fixture runs cargo only under it: a cargo build is a heavy job, and a plain pytest
#: that built qd-prep while another job held the lock is how the rule was broken on 2026-10-06
#: (GAP-PYTEST-QD-PREP-FIXTURE-BUILDS-OUTSIDE-THE-MAC-LOCK-2026-10-06).
MAC_HEAVY_LABEL_ENV = "MAC_HEAVY_LABEL"

#: Each env var names a prebuilt binary, so a run needs no cargo at all (the cross-built one,
#: one from another checkout, or one built under the lock earlier).
QD_MARGIN_PROBE_BIN_ENV = "QD_MARGIN_PROBE_BIN"
QD_CALIB_FIT_BIN_ENV = "QD_CALIB_FIT_BIN"
QD_GATE_REPORT_BIN_ENV = "QD_GATE_REPORT_BIN"
#: ``tools/real_tokenizer_pipeline.PREP_BIN_ENV``; ``test_qd_prep_minhash_parity.py`` asserts
#: the two agree.
QD_PREP_BIN_ENV = "QD_PREP_BIN"
QD_NOUL_ROWS_BIN_ENV = "QD_NOUL_ROWS_BIN"


def cargo_binary(
    env: str,
    package: str,
    binary: str,
    *,
    release: bool,
    environ: Mapping[str, str] = os.environ,
    cargo: str | None = CARGO,
    platform: str = sys.platform,
) -> Path:
    """``binary`` from ``package``: the one ``$env`` names, else one cargo builds from this
    checkout. Every session fixture below that needs a cargo-built binary comes through here.

    - ``$env`` names a path: that path, which must be an executable file. A name that is not
      one fails the run; building around a wrong name would test a binary nobody asked for.
    - No cargo on PATH: SKIPPED, naming ``$env`` -- a check that could not run is not one that
      passed.
    - darwin, not under ``tools/mac_heavy.sh`` (no ``MAC_HEAVY_LABEL``): SKIPPED, naming both
      ways to run it. The build would be a heavy job outside the lock.
    - Otherwise built once (``--release`` or debug) at ``CARGO_BUILD_JOBS``, which
      mac_heavy.sh sets to 2, into ``$CARGO_TARGET_DIR`` when that is set (a lane's own target
      dir), else this checkout's ``target/``.
    """
    named = environ.get(env, "")
    if named:
        path = Path(named)
        if not (path.is_file() and os.access(path, os.X_OK)):
            pytest.fail(f"{env}={named} is not an executable file; refusing to test around it")
        return path
    if cargo is None:
        pytest.skip(f"cargo is not on PATH and {env} is unset: {binary} was not built")
    if platform == "darwin" and not environ.get(MAC_HEAVY_LABEL_ENV):
        pytest.skip(
            f"building {binary} is a heavy job and this run is not under tools/mac_heavy.sh "
            f"({MAC_HEAVY_LABEL_ENV} is unset): set {env} to a built {binary}, or run the "
            "tests under bash tools/mac_heavy.sh <label> <command>"
        )
    subprocess.run(
        [cargo, "build", "--quiet", *(["--release"] if release else []),
         "--manifest-path", str(REPO / "Cargo.toml"), "-p", package, "--bin", binary],
        check=True, timeout=900, env=dict(environ),
    )
    target = Path(environ.get("CARGO_TARGET_DIR") or REPO / "target")
    built = target / ("release" if release else "debug") / binary
    assert built.is_file(), built
    return built


@pytest.fixture(scope="session")
def probe() -> Path:
    """``qd-margin-probe`` (debug)."""
    return cargo_binary(QD_MARGIN_PROBE_BIN_ENV, "qd-runtime", "qd-margin-probe", release=False)


@pytest.fixture(scope="session")
def calib_fit_bin() -> Path:
    """``qd-calib-fit`` (release, which the benchmark beside its parity tests needs)."""
    return cargo_binary(QD_CALIB_FIT_BIN_ENV, "qd-runtime", "qd-calib-fit", release=True)


@pytest.fixture(scope="session")
def gate_report_bin() -> Path:
    """``qd-gate-report`` (release)."""
    return cargo_binary(QD_GATE_REPORT_BIN_ENV, "qd-runtime", "qd-gate-report", release=True)


@pytest.fixture(scope="session")
def qd_prep_bin() -> Path:
    """``qd-prep`` (release), which MinHash signing runs in."""
    return cargo_binary(QD_PREP_BIN_ENV, "qd-prep", "qd-prep", release=True)


@pytest.fixture
def qd_prep(qd_prep_bin: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``QD_PREP_BIN`` set to :func:`qd_prep_bin` for one test, so dedupe and split sign in
    qd-prep (``real_tokenizer_pipeline.native_minhash``); there is no Python fallback."""
    monkeypatch.setenv(QD_PREP_BIN_ENV, str(qd_prep_bin))
    return qd_prep_bin


@pytest.fixture(scope="session")
def noul_rows_bin() -> Path:
    """``qd-noul-rows`` (debug)."""
    return cargo_binary(QD_NOUL_ROWS_BIN_ENV, "qd-mutate", "qd-noul-rows", release=False)
