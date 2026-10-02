"""tools/mac_heavy.sh: one heavy job on the Mac at a time.

GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02. Each test points the lock at a
temporary directory, so none of them touches the real lock.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "mac_heavy.sh"


def _run(lock: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = {
        **os.environ,
        "MAC_HEAVY_LOCK": str(lock),
        "MAC_HEAVY_POLL_S": "1",
        "MAC_HEAVY_MIN_FREE_GB": "0",
        # The busy-machine gate reads the real host; only the test about that gate tightens it,
        # so no other test waits on whatever else this Mac is doing.
        "MAC_HEAVY_MAX_LOAD": "100000",
        "MAC_HEAVY_MAX_PROCS": "10000000",
        **env,
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_runs_the_command_under_the_lock_and_releases_it(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    seen = tmp_path / "seen"
    proc = _run(lock, "t", "bash", "-c", f'cat "{lock}/owner" > "{seen}"; echo "$CARGO_BUILD_JOBS"')
    assert proc.returncode == 0, proc.stderr
    owner = seen.read_text()
    assert "label=t" in owner and "pid=" in owner and "start=" in owner
    # 2, not 4, since the second launchd-SIGBUS panic (GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-
    # 2026-10-02): the human chose "resume, gentler" -- cargo at -j 2.
    assert proc.stdout.strip() == "2"
    assert not lock.exists()


def test_the_commands_exit_status_is_returned_and_the_lock_released(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    proc = _run(lock, "t", "bash", "-c", "exit 7")
    assert proc.returncode == 7
    assert not lock.exists()


def test_a_held_lock_is_never_run_through_and_never_broken(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    lock.mkdir()
    (lock / "owner").write_text("label=manual: another session\nstart=2026-10-02T14:00:00Z\n")
    marker = tmp_path / "ran"
    proc = _run(lock, "t", "touch", str(marker), MAC_HEAVY_MAX_WAIT_S="2")
    assert proc.returncode == 75
    assert "did NOT run" in proc.stderr
    assert not marker.exists()
    assert (lock / "owner").read_text().startswith("label=manual: another session")


def test_a_dead_holder_is_reported_stale_but_not_broken(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    lock.mkdir()
    dead = subprocess.run(["bash", "-c", "echo $$"], capture_output=True, text=True, check=True)
    (lock / "owner").write_text(f"label=gone\npid={dead.stdout.strip()}\n")
    proc = _run(lock, "t", "true", MAC_HEAVY_MAX_WAIT_S="1")
    assert proc.returncode == 75
    assert "STALE" in proc.stderr
    assert lock.exists()


def test_too_little_free_disk_refuses_before_running(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    marker = tmp_path / "ran"
    proc = _run(lock, "t", "touch", str(marker), MAC_HEAVY_MIN_FREE_GB="1000000")
    assert proc.returncode == 74
    assert not marker.exists()
    assert not lock.exists()


def test_bad_settings_and_missing_command_are_refused(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    assert _run(lock, "only-a-label").returncode == 64
    assert _run(lock, "t", "true", MAC_HEAVY_MAX_WAIT_S="soon").returncode == 64
    assert _run(lock, "t", "true", MAC_HEAVY_MAX_RSS_MB="lots").returncode == 64
    assert _run(lock, "t", "true", MAC_HEAVY_MAX_LOAD="1.5").returncode == 64
    assert _run(lock, "t", "true", MAC_HEAVY_MAX_PROCS="").returncode == 64
    assert not lock.exists()


# The guards below were added after the second launchd-SIGBUS panic
# (AUDIT/mac-stability-2026-10-02/report.md): JetsamEvent reports showed single test processes at
# 36-120 GiB resident on this 64 GiB Mac, and both panics came amid heavy process churn.


def test_a_job_tree_over_the_rss_cap_is_killed_and_reported(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    pidfile = tmp_path / "grandchild.pid"
    # The allocation is in a grandchild, so the cap must sum the whole tree, not the direct child.
    hog = (
        "import os, time; open(os.environ['PIDFILE'], 'w').write(str(os.getpid())); "
        "b = b'\\x01' * (300 * 1024 * 1024); time.sleep(60)"
    )
    started = time.monotonic()
    proc = _run(
        lock,
        "t",
        "bash",
        "-c",
        f'python3 -c "{hog}"',
        MAC_HEAVY_MAX_RSS_MB="100",
        MAC_HEAVY_RSS_POLL_S="1",
        PIDFILE=str(pidfile),
    )
    assert proc.returncode == 70, proc.stderr
    assert "over the RSS cap" in proc.stderr
    assert time.monotonic() - started < 40
    grandchild = int(pidfile.read_text())
    assert subprocess.run(["kill", "-0", str(grandchild)], check=False).returncode != 0
    assert not lock.exists()


def test_a_busy_machine_waits_then_refuses_without_running(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    marker = tmp_path / "ran"
    for over in ({"MAC_HEAVY_MAX_LOAD": "0"}, {"MAC_HEAVY_MAX_PROCS": "1"}):
        proc = _run(lock, "t", "touch", str(marker), MAC_HEAVY_MAX_WAIT_S="2", **over)
        assert proc.returncode == 76, (over, proc.stderr)
        assert "did NOT run" in proc.stderr and "busy" in proc.stderr
        assert not marker.exists()
        assert not lock.exists()


def test_stopping_the_wrapper_stops_the_job_tree_and_releases_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    pidfile = tmp_path / "child.pid"
    env = {
        **os.environ,
        "MAC_HEAVY_LOCK": str(lock),
        "MAC_HEAVY_POLL_S": "1",
        "MAC_HEAVY_RSS_POLL_S": "1",
        "MAC_HEAVY_MIN_FREE_GB": "0",
        "MAC_HEAVY_MAX_LOAD": "100000",
        "MAC_HEAVY_MAX_PROCS": "10000000",
    }
    wrapper = subprocess.Popen(
        ["bash", str(SCRIPT), "t", "bash", "-c", f'echo $$ > "{pidfile}"; exec sleep 60'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        child = int(pidfile.read_text())
        wrapper.terminate()
        wrapper.wait(timeout=20)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
    assert wrapper.returncode == 143
    deadline = time.monotonic() + 15
    while subprocess.run(["kill", "-0", str(child)], check=False).returncode == 0:
        assert time.monotonic() < deadline, "the job outlived its wrapper"
        time.sleep(0.2)
    assert not lock.exists()
