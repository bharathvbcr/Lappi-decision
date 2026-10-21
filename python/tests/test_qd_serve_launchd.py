"""tools/qd_serve_launchd.py and the training guard in tools/mac_heavy.sh.

Nothing here touches launchd: the LaunchAgents directory is a temp dir, launchctl is a fake that
records its argv, and the daemon is a stub that speaks `qd serve`'s control ops
(`crates/qd-runtime/src/wire.rs` ControlOp, `service.rs` StatusReport) on a Unix socket.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "qd_serve_launchd.py"
MAC_HEAVY = ROOT / "tools" / "mac_heavy.sh"
SHA = "6f7b9ba73bca3a0b2e4ec9b2c0fc2aa6e7459f7e3286b26386633a222667dfcb"

FAKE_LAUNCHCTL = """#!/usr/bin/env python3
import json, os, sys
state = os.environ["FAKE_STATE"]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
verb = sys.argv[1]
if os.environ.get("FAKE_FAIL") == verb:
    sys.stderr.write("fake launchctl: " + verb + " refused\\n")
    sys.exit(5)
if verb == "print":
    sys.exit(0 if os.path.exists(state) else 113)
if verb == "bootstrap":
    open(state, "w").close()
if verb == "bootout" and os.path.exists(state):
    os.remove(state)
"""


class StubDaemon:
    """A qd-serve-shaped listener: ping, status, and `ok` for anything else."""

    def __init__(
        self, path: Path, warm: bool, weight_hash: str = SHA, garbage: bool = False
    ) -> None:
        self.warm = warm
        self.weight_hash = weight_hash
        self.garbage = garbage
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        self.listener.listen(8)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _reply(self, line: dict[str, object]) -> dict[str, object]:
        if line.get("op") == "ping":
            return {"status": "ack", "op": "ping", "acted": False}
        if line.get("op") == "status":
            backend = (
                {"name": "qd-metal/stub", "weight_hash": self.weight_hash, "is_model": True}
                if self.warm
                else None
            )
            return {
                "status": "status",
                "runtime_version": "stub-1",
                "warm": self.warm,
                "degraded": False,
                "backend": backend,
                "shutdown_requested": False,
            }
        self.warm = True
        return {"status": "ok"}

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            with conn:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                if self.garbage:
                    conn.sendall(b"not json\n")
                else:
                    conn.sendall(json.dumps(self._reply(json.loads(data))).encode() + b"\n")

    def close(self) -> None:
        self.listener.close()


@pytest.fixture
def sockdir() -> Iterator[Path]:
    # AF_UNIX paths are capped near 104 bytes on macOS, which pytest's tmp_path can exceed.
    path = Path(tempfile.mkdtemp(prefix="qd", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def env(tmp_path: Path, sockdir: Path) -> dict[str, Path]:
    release = tmp_path / "release"
    release.mkdir()
    model = release / "model.safetensors"
    model.write_bytes(b"weights")
    manifest = {
        "expected_identity": {"weight_hash": SHA},
        "files": {
            "model.safetensors": {"bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}
        },
    }
    (release / "release_manifest.json").write_text(json.dumps(manifest))
    binary = tmp_path / "qd-metal-serve"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    fake = tmp_path / "launchctl"
    fake.write_text(FAKE_LAUNCHCTL)
    fake.chmod(0o755)
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    return {
        "release": release,
        "binary": binary,
        "fake": fake,
        "agents": agents,
        "sock": sockdir / "qd.sock",
        "stdout": tmp_path / "logs" / "out.log",
        "stderr": tmp_path / "logs" / "err.log",
        "state": tmp_path / "state",
        "calls": tmp_path / "calls.jsonl",
    }


def write_banner(env: dict[str, Path], sha: str = SHA, release: Path | None = None) -> None:
    env["stderr"].parent.mkdir(parents=True, exist_ok=True)
    env["stderr"].write_text(
        f"qd-metal-serve: listening on {env['sock']} (release {release or env['release']}, "
        f"weight_hash {sha}, calibration c; idle eviction 1 ms)\n"
    )


def run_tool(
    env: dict[str, Path], verb: str, *extra: str, fail: str = ""
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        "-I",
        str(TOOL),
        verb,
        "--binary",
        str(env["binary"]),
        "--release-dir",
        str(env["release"]),
        "--release-sha",
        SHA,
        "--socket",
        str(env["sock"]),
        "--stdout-log",
        str(env["stdout"]),
        "--stderr-log",
        str(env["stderr"]),
        "--health-wait-s",
        "1",
    ]
    if verb in ("install", "restart"):
        args += [
            "--launch-agents-dir",
            str(env["agents"]),
            "--launchctl",
            str(env["fake"]),
            "--domain",
            "gui/501",
        ]
    return subprocess.run(
        [*args, *extra],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            **os.environ,
            "FAKE_STATE": str(env["state"]),
            "FAKE_LOG": str(env["calls"]),
            "FAKE_FAIL": fail,
        },
    )


def calls(env: dict[str, Path]) -> list[list[str]]:
    if not env["calls"].exists():
        return []
    return [json.loads(row) for row in env["calls"].read_text().splitlines()]


LABEL = "com.bharath.lappi.qd-serve"


def test_install_renders_the_plist_from_explicit_arguments_and_bootstraps(
    env: dict[str, Path],
) -> None:
    daemon = StubDaemon(env["sock"], warm=False)
    write_banner(env)
    try:
        done = run_tool(env, "install", "--idle-timeout-ms", "1800000")
    finally:
        daemon.close()
    assert done.returncode == 0, done.stderr
    plist = plistlib.loads((env["agents"] / f"{LABEL}.plist").read_bytes())
    assert plist["ProgramArguments"] == [
        str(env["binary"]),
        "--release",
        str(env["release"]),
        "--socket",
        str(env["sock"]),
        "--idle-timeout-ms",
        "1800000",
    ]
    assert plist["StandardErrorPath"] == str(env["stderr"]) and plist["KeepAlive"] is True
    assert [c[0] for c in calls(env)] == ["print", "bootstrap"]
    assert calls(env)[1] == ["bootstrap", "gui/501", str(env["agents"] / f"{LABEL}.plist")]
    assert "cold" in done.stdout


def test_a_second_install_kickstarts_and_a_changed_one_reloads(env: dict[str, Path]) -> None:
    daemon = StubDaemon(env["sock"], warm=False)
    write_banner(env)
    try:
        assert run_tool(env, "install").returncode == 0
        env["calls"].unlink()
        assert run_tool(env, "install").returncode == 0
        assert [c[0] for c in calls(env)] == ["print", "kickstart"]
        assert calls(env)[1] == ["kickstart", "-k", f"gui/501/{LABEL}"]
        env["calls"].unlink()
        assert run_tool(env, "install", "--idle-timeout-ms", "5").returncode == 0
        assert [c[0] for c in calls(env)] == ["print", "bootout", "bootstrap"]
    finally:
        daemon.close()


def test_restart_needs_an_installed_matching_plist_and_kickstarts(env: dict[str, Path]) -> None:
    refused = run_tool(env, "restart")
    assert refused.returncode == 2 and "run install" in refused.stderr
    assert calls(env) == []
    daemon = StubDaemon(env["sock"], warm=False)
    write_banner(env)
    try:
        assert run_tool(env, "install").returncode == 0
        env["calls"].unlink()
        assert run_tool(env, "restart").returncode == 0
    finally:
        daemon.close()
    assert [c[0] for c in calls(env)] == ["print", "kickstart"]


def test_a_sha_that_is_not_the_manifests_is_refused_before_anything_is_touched(
    env: dict[str, Path],
) -> None:
    manifest = json.loads((env["release"] / "release_manifest.json").read_text())
    manifest["expected_identity"]["weight_hash"] = "0" * 64
    (env["release"] / "release_manifest.json").write_text(json.dumps(manifest))
    done = run_tool(env, "install")
    assert done.returncode == 2 and "release sha mismatch" in done.stderr
    assert calls(env) == [] and not list(env["agents"].iterdir())


@pytest.mark.parametrize("damage", ["no-model", "short-model", "no-manifest", "no-binary"])
def test_a_missing_or_wrong_model_is_refused(env: dict[str, Path], damage: str) -> None:
    if damage == "no-model":
        (env["release"] / "model.safetensors").unlink()
    elif damage == "short-model":
        (env["release"] / "model.safetensors").write_bytes(b"x")
    elif damage == "no-manifest":
        (env["release"] / "release_manifest.json").unlink()
    else:
        env["binary"].unlink()
    done = run_tool(env, "install")
    assert done.returncode == 2, done.stderr
    assert calls(env) == [] and not list(env["agents"].iterdir())


def test_verify_files_catches_same_size_corruption(env: dict[str, Path]) -> None:
    (env["release"] / "model.safetensors").write_bytes(b"weighTs")
    assert run_tool(env, "render").returncode == 0
    done = run_tool(env, "render", "--verify-files")
    assert done.returncode == 2 and "does not hash" in done.stderr


def test_no_daemon_is_a_failed_health_check_not_a_pass(env: dict[str, Path]) -> None:
    write_banner(env)
    done = run_tool(env, "install")
    assert done.returncode == 1 and "ping over" in done.stderr


def test_a_daemon_bound_to_another_release_fails_health(env: dict[str, Path]) -> None:
    daemon = StubDaemon(env["sock"], warm=False)
    write_banner(env, sha="1" * 64)
    try:
        done = run_tool(env, "health")
    finally:
        daemon.close()
    assert done.returncode == 1 and "another release" in done.stderr


def test_a_missing_bind_banner_fails_health(env: dict[str, Path]) -> None:
    daemon = StubDaemon(env["sock"], warm=False)
    try:
        done = run_tool(env, "health")
    finally:
        daemon.close()
    assert done.returncode == 1 and "bind" in done.stderr


def test_warm_request_loads_the_model_and_the_wire_identity_is_checked(
    env: dict[str, Path], tmp_path: Path
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema_version": 1, "task": "t"}))
    write_banner(env)
    good = StubDaemon(env["sock"], warm=False)
    try:
        done = run_tool(env, "health", "--warm-request", str(request))
    finally:
        good.close()
    assert done.returncode == 0 and "warm, identity verified on the wire" in done.stdout, (
        done.stderr
    )
    env["sock"].unlink()
    bad = StubDaemon(env["sock"], warm=False, weight_hash="2" * 64)
    try:
        done = run_tool(env, "health", "--warm-request", str(request))
    finally:
        bad.close()
    assert done.returncode == 1 and "not the release" in done.stderr


def test_a_launchctl_failure_is_exit_3(env: dict[str, Path]) -> None:
    done = run_tool(env, "install", fail="bootstrap")
    assert done.returncode == 3 and "bootstrap" in done.stderr


# ---- held, and the mac_heavy training guard ------------------------------------------------------


def held(env: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), "held", "--socket", str(env["sock"]), "--timeout-s", "1"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_held_distinguishes_free_held_and_unknown(env: dict[str, Path]) -> None:
    assert held(env).returncode == 0  # no socket file
    env["sock"].write_text("")
    assert held(env).returncode == 11  # a file that is not a socket
    env["sock"].unlink()
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(env["sock"]))
    stale.close()  # the file stays, nothing listens: what a crashed daemon leaves
    assert env["sock"].exists() and held(env).returncode == 0
    env["sock"].unlink()
    for warm in (False, True):
        daemon = StubDaemon(env["sock"], warm=warm)
        try:
            done = held(env)
        finally:
            daemon.close()
        env["sock"].unlink()
        assert done.returncode == 10, done.stderr
        assert ("model resident" in done.stdout) == warm
    garbage = StubDaemon(env["sock"], warm=False, garbage=True)
    try:
        assert held(env).returncode == 11
    finally:
        garbage.close()


def mac_heavy(
    env: dict[str, Path], tmp_path: Path, marker: Path, **extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(MAC_HEAVY), "t", "bash", "-c", f"echo ran > {marker}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            **os.environ,
            "MAC_HEAVY_LOCK": str(tmp_path / "lock.d"),
            "MAC_HEAVY_POLL_S": "1",
            "MAC_HEAVY_MIN_FREE_GB": "0",
            "MAC_HEAVY_MAX_LOAD": "100000",
            "MAC_HEAVY_MAX_PROCS": "10000000",
            "MAC_HEAVY_QD_SOCKET": str(env["sock"]),
            **extra,
        },
    )


@pytest.mark.parametrize("warm", [True, False])
def test_a_training_job_is_refused_while_a_daemon_holds_the_socket(
    env: dict[str, Path], tmp_path: Path, warm: bool
) -> None:
    marker = tmp_path / "marker"
    daemon = StubDaemon(env["sock"], warm=warm)
    try:
        refused = mac_heavy(env, tmp_path, marker, MAC_HEAVY_TRAINING="1")
        unmarked = mac_heavy(env, tmp_path, marker)
    finally:
        daemon.close()
    assert refused.returncode == 77, refused.stderr
    assert "did NOT run" in refused.stderr and not (tmp_path / "lock.d").exists()
    assert unmarked.returncode == 0 and marker.exists()  # only a marked job is guarded


def test_a_training_job_runs_when_no_daemon_listens_and_fails_closed_when_unreadable(
    env: dict[str, Path], tmp_path: Path
) -> None:
    marker = tmp_path / "marker"
    free = mac_heavy(env, tmp_path, marker, MAC_HEAVY_TRAINING="1")
    assert free.returncode == 0 and marker.exists(), free.stderr
    marker.unlink()
    env["sock"].write_text("")
    unreadable = mac_heavy(env, tmp_path, marker, MAC_HEAVY_TRAINING="1")
    assert unreadable.returncode == 78 and not marker.exists()
    bad = mac_heavy(env, tmp_path, marker, MAC_HEAVY_TRAINING="yes")
    assert bad.returncode == 64
