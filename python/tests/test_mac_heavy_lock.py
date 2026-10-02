"""tools/mac_heavy.sh: one heavy job on the Mac at a time.

GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02. Each test points the lock at a
temporary directory, so none of them touches the real lock.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "mac_heavy.sh"


def _run(lock: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = {
        **os.environ,
        "MAC_HEAVY_LOCK": str(lock),
        "MAC_HEAVY_POLL_S": "1",
        "MAC_HEAVY_MIN_FREE_GB": "0",
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
    assert not lock.exists()
