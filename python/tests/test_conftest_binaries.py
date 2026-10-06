"""conftest's cargo-built binaries: named, skipped, refused or built -- never built behind the
Mac's one-heavy-job lock (GAP-PYTEST-QD-PREP-FIXTURE-BUILDS-OUTSIDE-THE-MAC-LOCK-2026-10-06).

On 2026-10-06 a plain pytest built qd-prep while another job held tools/mac_heavy.sh's lock, and
the `probe` and `noul_rows_bin` fixtures had no way to be handed a binary at all, so a full
suite with cargo on PATH always ran cargo. Every case here calls ``cargo_binary`` with its own
environment, cargo and platform; nothing in this file runs cargo.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import conftest
import pytest
from conftest import MAC_HEAVY_LABEL_ENV, cargo_binary

REPO = Path(__file__).resolve().parents[2]
CONFTEST = Path(conftest.__file__).resolve()
MAC_HEAVY = REPO / "tools" / "mac_heavy.sh"

#: Every cargo-built binary a fixture hands out, with the env var that names a prebuilt one.
FIXTURES = {
    "probe": "QD_MARGIN_PROBE_BIN",
    "calib_fit_bin": "QD_CALIB_FIT_BIN",
    "gate_report_bin": "QD_GATE_REPORT_BIN",
    "qd_prep_bin": "QD_PREP_BIN",
    "noul_rows_bin": "QD_NOUL_ROWS_BIN",
}


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def no_cargo_runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Records every subprocess.run conftest makes; a test that expects none asserts it."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(conftest.subprocess, "run", fake_run)
    return calls


def test_a_named_binary_is_used_and_nothing_is_built(tmp_path, no_cargo_runs) -> None:
    binary = _executable(tmp_path / "qd-margin-probe")
    got = cargo_binary(
        "QD_MARGIN_PROBE_BIN", "qd-runtime", "qd-margin-probe", release=False,
        environ={"QD_MARGIN_PROBE_BIN": str(binary)}, cargo="/bin/cargo", platform="darwin",
    )
    assert got == binary
    assert no_cargo_runs == []


@pytest.mark.parametrize("kind", ["missing", "not executable", "a directory"])
def test_a_name_that_is_not_an_executable_fails_the_run(tmp_path, no_cargo_runs, kind) -> None:
    """Building around a wrong name would test a binary nobody asked for."""
    path = tmp_path / "qd-prep"
    if kind == "not executable":
        path.write_text("", encoding="utf-8")
    elif kind == "a directory":
        path.mkdir()
    with pytest.raises(pytest.fail.Exception, match=r"QD_PREP_BIN=.* is not an executable"):
        cargo_binary(
            "QD_PREP_BIN", "qd-prep", "qd-prep", release=True,
            environ={"QD_PREP_BIN": str(path)}, cargo="/bin/cargo", platform="linux",
        )
    assert no_cargo_runs == []


def test_no_cargo_is_a_skip_naming_the_env_var(no_cargo_runs) -> None:
    with pytest.raises(pytest.skip.Exception, match="QD_NOUL_ROWS_BIN is unset"):
        cargo_binary(
            "QD_NOUL_ROWS_BIN", "qd-mutate", "qd-noul-rows", release=False,
            environ={}, cargo=None, platform="linux",
        )
    assert no_cargo_runs == []


def test_on_the_mac_a_build_outside_the_lock_is_a_skip_not_a_build(no_cargo_runs) -> None:
    with pytest.raises(pytest.skip.Exception) as skipped:
        cargo_binary(
            "QD_PREP_BIN", "qd-prep", "qd-prep", release=True,
            environ={}, cargo="/bin/cargo", platform="darwin",
        )
    reason = str(skipped.value)
    assert "QD_PREP_BIN" in reason and "tools/mac_heavy.sh" in reason
    assert no_cargo_runs == []


@pytest.mark.parametrize(
    ("platform", "environ"),
    [("darwin", {MAC_HEAVY_LABEL_ENV: "suite"}), ("linux", {})],
    ids=["mac-under-the-lock", "linux"],
)
def test_otherwise_it_builds_into_the_lanes_own_target_dir(
    tmp_path, no_cargo_runs, platform, environ
) -> None:
    target = tmp_path / "target"
    (target / "release").mkdir(parents=True)
    _executable(target / "release" / "qd-gate-report")
    got = cargo_binary(
        "QD_GATE_REPORT_BIN", "qd-runtime", "qd-gate-report", release=True,
        environ={**environ, "CARGO_TARGET_DIR": str(target)}, cargo="/bin/cargo",
        platform=platform,
    )
    assert got == target / "release" / "qd-gate-report"
    [args] = no_cargo_runs
    assert args[:3] == ["/bin/cargo", "build", "--quiet"] and "--release" in args
    assert args[-4:] == ["-p", "qd-runtime", "--bin", "qd-gate-report"]


def _fixture_calls() -> dict[str, ast.Call | None]:
    """Each session fixture's ``cargo_binary(...)`` call, read from conftest's source."""
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    found: dict[str, ast.Call | None] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FIXTURES:
            calls = [
                n for n in ast.walk(node)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "cargo_binary"
            ]
            found[node.name] = calls[0] if len(calls) == 1 else None
    return found


def test_every_binary_fixture_goes_through_cargo_binary_with_its_env_var() -> None:
    calls = _fixture_calls()
    assert set(calls) == set(FIXTURES), "a binary fixture is missing from conftest"
    for name, env in FIXTURES.items():
        call = calls[name]
        assert call is not None, f"{name} does not call cargo_binary exactly once"
        first = call.args[0]
        assert isinstance(first, ast.Name), name
        assert getattr(conftest, first.id) == env, name


def test_cargo_is_run_in_one_place_only() -> None:
    """The class, not the case: a sixth fixture that calls cargo itself would bypass the
    lock rule and the env override, which is how `probe` and `noul_rows_bin` came to have no
    override."""
    source = CONFTEST.read_text(encoding="utf-8")
    assert source.count("subprocess.run(") == 1
    assert source.index("subprocess.run(") > source.index("def cargo_binary(")
    assert source.index("subprocess.run(") < source.index("@pytest.fixture")


def test_mac_heavy_hands_its_job_the_label(tmp_path: Path) -> None:
    """The signal cargo_binary reads. The outer label, if this suite itself runs under
    mac_heavy.sh, must not leak through as the inner job's."""
    env = {
        **os.environ,
        MAC_HEAVY_LABEL_ENV: "outer",
        "MAC_HEAVY_LOCK": str(tmp_path / "lock.d"),
        "MAC_HEAVY_POLL_S": "1",
        "MAC_HEAVY_MIN_FREE_GB": "0",
        "MAC_HEAVY_MAX_LOAD": "100000",
        "MAC_HEAVY_MAX_PROCS": "10000000",
    }
    proc = subprocess.run(
        ["bash", str(MAC_HEAVY), "inner-job", "bash", "-c", f'echo "${MAC_HEAVY_LABEL_ENV}"'],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "inner-job"
