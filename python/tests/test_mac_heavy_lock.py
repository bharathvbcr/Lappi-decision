"""tools/mac_heavy.sh: one heavy job on the Mac at a time.

GAP-MAC-KERNEL-PANIC-CONCURRENT-HEAVY-LOAD-2026-10-02. Each test points the lock at a
temporary directory, so none of them touches the real lock.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

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
        # The reader hook is honoured only with this marker; no production launch sets it.
        **({"MAC_HEAVY_TEST_HOOK": "1"} if "MAC_HEAVY_FOOTPRINT_CMD" in env else {}),
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
    assert _run(lock, "t", "true", MAC_HEAVY_MAX_FOOTPRINT_MB="lots").returncode == 64
    assert _run(lock, "t", "true", MAC_HEAVY_MAX_LOAD="1.5").returncode == 64
    assert _run(lock, "t", "true", MAC_HEAVY_MAX_PROCS="").returncode == 64
    assert not lock.exists()


# The guards below were added after the second launchd-SIGBUS panic
# (AUDIT/mac-stability-2026-10-02/report.md): JetsamEvent reports showed single test processes at
# 36-120 GiB resident on this 64 GiB Mac, and both panics came amid heavy process churn.


def test_a_job_tree_over_the_footprint_cap_is_killed_and_reported(tmp_path: Path) -> None:
    # The real reader (tools/mac_footprint.py) on a real process tree, no injected hook.
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
        MAC_HEAVY_MAX_FOOTPRINT_MB="100",
        MAC_HEAVY_RSS_POLL_S="1",
        PIDFILE=str(pidfile),
    )
    assert proc.returncode == 70, proc.stderr
    assert "over the footprint cap" in proc.stderr
    assert "result does not count" in proc.stderr
    assert time.monotonic() - started < 40
    grandchild = int(pidfile.read_text())
    assert subprocess.run(["kill", "-0", str(grandchild)], check=False).returncode != 0
    assert not lock.exists()


def test_the_real_reader_reports_a_live_process_and_refuses_a_dead_one() -> None:
    reader = SCRIPT.parent / "mac_footprint.py"
    live = subprocess.run(
        ["python3", "-I", str(reader), str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert live.returncode == 0, live.stderr
    assert int(live.stdout) > 0
    dead = subprocess.run(["bash", "-c", "echo $$"], capture_output=True, text=True, check=True)
    gone = subprocess.run(
        ["python3", "-I", str(reader), dead.stdout.strip()],
        capture_output=True,
        text=True,
        check=False,
    )
    assert gone.returncode == 1 and gone.stdout == ""
    bad = subprocess.run(
        ["python3", "-I", str(reader), "x"], capture_output=True, text=True, check=False
    )
    assert bad.returncode == 2


def _hook(tmp_path: Path, body: str) -> str:
    hook = tmp_path / "reader.sh"
    hook.write_text("#!/bin/bash\n" + body + "\n")
    hook.chmod(0o755)
    return str(hook)


def test_an_injected_footprint_over_the_cap_stops_the_job_with_exit_70(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    proc = _run(
        lock,
        "t",
        "sleep",
        "60",
        MAC_HEAVY_MAX_FOOTPRINT_MB="1000",
        MAC_HEAVY_RSS_POLL_S="1",
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, "echo 1001"),
    )
    assert proc.returncode == 70, proc.stderr
    assert "footprint" in proc.stderr and "1001 MiB" in proc.stderr
    assert not lock.exists()


def test_a_footprint_at_the_cap_is_not_killed(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    proc = _run(
        lock,
        "t",
        "sleep",
        "2",
        MAC_HEAVY_MAX_FOOTPRINT_MB="1000",
        MAC_HEAVY_RSS_POLL_S="1",
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, "echo 1000"),
    )
    assert proc.returncode == 0, proc.stderr


def test_a_reader_that_fails_three_polls_in_a_row_stops_the_job_with_exit_71(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "lock.d"
    pidfile = tmp_path / "child.pid"
    started = time.monotonic()
    proc = _run(
        lock,
        "t",
        "bash",
        "-c",
        f'echo $$ > "{pidfile}"; exec sleep 60',
        MAC_HEAVY_RSS_POLL_S="1",
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, "exit 3"),
    )
    assert proc.returncode == 71, proc.stderr
    assert "memory guard could not run" in proc.stderr
    assert "3 of 3" in proc.stderr
    assert time.monotonic() - started < 40
    child = int(pidfile.read_text())
    assert subprocess.run(["kill", "-0", str(child)], check=False).returncode != 0
    assert not lock.exists()


def test_a_non_integer_reader_answer_counts_as_a_failure(tmp_path: Path) -> None:
    proc = _run(
        tmp_path / "lock.d",
        "t",
        "sleep",
        "60",
        MAC_HEAVY_RSS_POLL_S="1",
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, "echo lots"),
    )
    assert proc.returncode == 71, proc.stderr


def test_an_intermittent_reader_failure_is_not_fatal(tmp_path: Path) -> None:
    counter = tmp_path / "n"
    body = (
        f'n=$(cat "{counter}" 2>/dev/null || echo 0); echo $((n + 1)) > "{counter}"; '
        "[ $((n % 3)) -eq 0 ] && exit 3; echo 1"
    )
    proc = _run(
        tmp_path / "lock.d",
        "t",
        "sleep",
        "5",
        MAC_HEAVY_RSS_POLL_S="1",
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, body),
    )
    assert proc.returncode == 0, proc.stderr


def test_the_retired_rss_variable_is_refused_loudly(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    marker = tmp_path / "ran"
    proc = _run(lock, "t", "touch", str(marker), MAC_HEAVY_MAX_RSS_MB="32768")
    assert proc.returncode == 64
    assert "MAC_HEAVY_MAX_FOOTPRINT_MB" in proc.stderr
    assert not marker.exists()
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


def test_an_oversized_setting_is_refused_not_read_as_not_over(tmp_path: Path) -> None:
    # 20 digits is all digits, but bash's [ -gt ] errors on it and an error read as "not over"
    # ran the job uncapped.
    lock = tmp_path / "lock.d"
    marker = tmp_path / "ran"
    huge = "9" * 20
    for var in (
        "MAC_HEAVY_MAX_FOOTPRINT_MB",
        "MAC_HEAVY_MAX_WAIT_S",
        "MAC_HEAVY_MAX_LOAD",
        "MAC_HEAVY_MAX_PROCS",
        "MAC_HEAVY_RSS_POLL_S",
    ):
        proc = _run(lock, "t", "touch", str(marker), **{var: huge})
        assert proc.returncode == 64, (var, proc.stderr)
        assert not marker.exists()
    assert not lock.exists()


_NUMERIC_SETTINGS = (
    "MAC_HEAVY_MAX_WAIT_S",
    "MAC_HEAVY_MIN_FREE_GB",
    "MAC_HEAVY_POLL_S",
    "MAC_HEAVY_MAX_FOOTPRINT_MB",
    "MAC_HEAVY_RSS_POLL_S",
    "MAC_HEAVY_MAX_LOAD",
    "MAC_HEAVY_MAX_PROCS",
)


@pytest.mark.parametrize("value", ["09", "0900", "08", "010", "00", "007", "09999"])
@pytest.mark.parametrize("var", _NUMERIC_SETTINGS)
def test_a_leading_zero_setting_is_refused_not_read_as_octal(
    tmp_path: Path, var: str, value: str
) -> None:
    """Bash arithmetic reads ``09`` as an invalid octal (the lock loop then fell through and ran
    the job unlocked) and ``0900`` as 0 under a free-space floor; both fail open. Only a
    canonical decimal is a setting."""
    lock = tmp_path / "lock.d"
    marker = tmp_path / "ran"
    proc = _run(lock, "t", "touch", str(marker), **{var: value})
    assert proc.returncode == 64, (var, value, proc.stderr)
    assert "canonical" in proc.stderr
    assert not marker.exists()
    assert not lock.exists()


def test_a_leading_zero_poll_never_runs_through_or_breaks_a_held_lock(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    lock.mkdir()
    (lock / "owner").write_text("label=other\npid=1\n")
    marker = tmp_path / "ran"
    proc = _run(lock, "t", "touch", str(marker), MAC_HEAVY_POLL_S="09")
    assert proc.returncode == 64, proc.stderr
    assert not marker.exists()
    assert (lock / "owner").read_text() == "label=other\npid=1\n"


def test_canonical_decimals_including_zero_are_accepted(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    proc = _run(
        lock,
        "t",
        "true",
        MAC_HEAVY_MIN_FREE_GB="0",
        MAC_HEAVY_POLL_S="9",
        MAC_HEAVY_MAX_WAIT_S="900",
    )
    assert proc.returncode == 0, proc.stderr


def test_an_oversized_reader_answer_is_a_reader_failure(tmp_path: Path) -> None:
    proc = _run(
        tmp_path / "lock.d",
        "t",
        "sleep",
        "60",
        MAC_HEAVY_RSS_POLL_S="1",
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, "echo " + "9" * 30),
    )
    assert proc.returncode == 71, proc.stderr
    assert "integer expression" not in proc.stderr


def _alive(pid: int) -> bool:
    probe = subprocess.run(["kill", "-0", str(pid)], check=False, capture_output=True)
    return probe.returncode == 0


def test_a_process_the_job_leaves_behind_does_not_outlive_the_wrapper(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    pidfile = tmp_path / "orphan.pid"
    bystander = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        # The root starts a child and exits first, so the child is reparented to init.
        job = (
            "python3 -c \"import os, time; "
            "open(os.environ['PIDFILE'], 'w').write(str(os.getpid())); "
            'time.sleep(60)" & sleep 1; exit 0'
        )
        proc = _run(lock, "t", "bash", "-c", job, PIDFILE=str(pidfile))
        assert proc.returncode == 0, proc.stderr
        orphan = int(pidfile.read_text())
        deadline = time.monotonic() + 15
        while _alive(orphan):
            assert time.monotonic() < deadline, "the job's orphan outlived its wrapper"
            time.sleep(0.2)
        assert "left behind" in proc.stderr
        assert bystander.poll() is None, "the sweep killed a process outside the job"
        assert not lock.exists()
    finally:
        bystander.kill()
        bystander.wait()


def test_the_footprint_hook_is_refused_without_the_test_marker(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    marker = tmp_path / "ran"
    proc = _run(
        lock,
        "t",
        "touch",
        str(marker),
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, "echo 0"),
        MAC_HEAVY_TEST_HOOK="",
    )
    assert proc.returncode == 64, proc.stderr
    assert "MAC_HEAVY_FOOTPRINT_CMD" in proc.stderr
    assert not marker.exists()
    assert not lock.exists()


_REAL_MPS = os.environ.get("QD_TEST_REAL_MPS") == "1"
ML_PYTHON = Path("/Users/bharath/.venvs/ml/bin/python")


@pytest.mark.skipif(
    not _REAL_MPS,
    reason="opt-in: set QD_TEST_REAL_MPS=1 (and run under tools/mac_heavy.sh) to allocate 3 GiB "
    "on the real GPU; this check did NOT run",
)
def test_a_real_mps_allocation_over_the_footprint_cap_is_stopped(tmp_path: Path) -> None:
    assert ML_PYTHON.exists(), f"{ML_PYTHON} is missing; the opt-in test cannot run"
    code = (
        "import time, torch; "
        "x = torch.empty(int(3 * 2**30 / 4), device='mps'); x.fill_(1); torch.mps.synchronize(); "
        "time.sleep(60)"
    )
    started = time.monotonic()
    proc = _run(
        tmp_path / "lock.d",
        "t",
        str(ML_PYTHON),
        "-c",
        code,
        MAC_HEAVY_MAX_FOOTPRINT_MB="2000",
        MAC_HEAVY_RSS_POLL_S="1",
    )
    assert proc.returncode == 70, proc.stderr
    assert "over the footprint cap" in proc.stderr
    assert time.monotonic() - started < 55


def test_rewriting_the_script_in_place_mid_run_does_not_change_a_running_wrapper(
    tmp_path: Path,
) -> None:
    """Bash reads a running script incrementally, so an in-place edit (truncate and write, ``sed
    -i`` on some filesystems, the Edit tool) resumed every live wrapper at a shifted offset: the
    re-score jobs died with exit 143 on 2026-10-12. The body is one function, parsed whole before
    anything runs, so the rewrite is invisible and the wrapper still returns the job's status."""
    import shutil

    tools = tmp_path / "tools"
    tools.mkdir()
    script = tools / "mac_heavy.sh"
    shutil.copy(SCRIPT, script)
    shutil.copy(SCRIPT.parent / "mac_footprint.py", tools / "mac_footprint.py")
    lock = tmp_path / "lock.d"
    env = {
        **os.environ,
        "MAC_HEAVY_LOCK": str(lock),
        "MAC_HEAVY_POLL_S": "1",
        "MAC_HEAVY_RSS_POLL_S": "1",
        "MAC_HEAVY_MIN_FREE_GB": "0",
        "MAC_HEAVY_MAX_LOAD": "100000",
        "MAC_HEAVY_MAX_PROCS": "10000000",
    }
    proc = subprocess.Popen(
        ["bash", str(script), "t", "bash", "-c", "sleep 4; exit 7"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 30
    while not (lock / "owner").exists():
        assert time.monotonic() < deadline, "the wrapper never took the lock"
        time.sleep(0.1)
    time.sleep(1.5)  # the job is running and the wrapper is in its polling loop
    size = script.stat().st_size
    with script.open("r+b") as fh:  # in place: the same inode and length, different bytes
        fh.write(b"\n" * size)
    _out, err = proc.communicate(timeout=60)
    assert proc.returncode == 7, (proc.returncode, err)
    assert not lock.exists()


# memguard binds first (CLAUDE.md): it SIGTERMs any single process over its -cap-gib threshold
# and leaves no log, so the footprint cap must sit below that threshold, read from its plist.


def _plist(tmp_path: Path, gib: str) -> Path:
    plist = tmp_path / "memguard.plist"
    plist.write_text(
        "<plist><dict><key>ProgramArguments</key><array>\n"
        "<string>/x/memguard</string>\n"
        f"<string>-cap-gib</string><string>{gib}</string>\n"
        "<string>-floor-gib</string><string>2</string>\n"
        "</array></dict></plist>\n"
    )
    return plist


def test_ops_default_cap_sits_below_memguards_threshold_read_from_its_plist(tmp_path: Path) -> None:
    # Proved through behaviour: a 16 GiB threshold gives a 12288 MiB default cap, so 12289 is killed.
    over = _run(
        tmp_path / "lock.d", "t", "sleep", "30",
        MAC_HEAVY_MEMGUARD_PLIST=str(_plist(tmp_path, "16")), MAC_HEAVY_RSS_POLL_S="1",
        MAC_HEAVY_FOOTPRINT_CMD=_hook(tmp_path, "echo 12289"),
    )
    assert over.returncode == 70, over.stderr
    assert "> 12288 MiB" in over.stderr


def test_a_footprint_cap_at_or_above_memguards_threshold_is_refused(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    marker = tmp_path / "ran"
    plist = str(_plist(tmp_path, "32"))
    for cap in ("32768", "53248"):
        proc = _run(lock, "t", "touch", str(marker), MAC_HEAVY_MEMGUARD_PLIST=plist, MAC_HEAVY_MAX_FOOTPRINT_MB=cap)
        assert proc.returncode == 64, proc.stderr
        assert "memguard" in proc.stderr and "did NOT run" in proc.stderr
    assert not marker.exists()
    assert not lock.exists()


def test_an_unreadable_memguard_threshold_refuses_rather_than_guessing(tmp_path: Path) -> None:
    bad = tmp_path / "bad.plist"
    bad.write_text("<plist><dict></dict></plist>\n")
    proc = _run(
        tmp_path / "lock.d", "t", "true",
        MAC_HEAVY_MEMGUARD_PLIST=str(bad), MAC_HEAVY_MEMGUARD_MAIN=str(tmp_path / "nope.go"),
    )
    assert proc.returncode == 64
    assert "cannot be read" in proc.stderr


def test_a_job_ended_by_an_external_sigterm_gets_a_distinct_exit_and_message(tmp_path: Path) -> None:
    lock = tmp_path / "lock.d"
    proc = _run(lock, "t", "bash", "-c", "kill -TERM $$; sleep 5", MAC_HEAVY_MEMGUARD_PLIST=str(_plist(tmp_path, "32")))
    assert proc.returncode == 79, proc.stderr
    assert "external SIGTERM -- likely memguard" in proc.stderr
    assert "finished with exit" not in proc.stderr
    assert not lock.exists()
