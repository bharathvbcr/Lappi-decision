"""Adversarial tests for lane L3's ``tools/footprint_peak.py`` (adv/ft-57fc7484-l3).

``test_BREAK_*`` assert what a reviewer would expect and FAIL on the lane; the rest held.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "footprint_peak.py"
sys.path.insert(0, str(REPO / "tools"))

import footprint_peak  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="reads libproc")


def _run(timeline: Path, command: list[str], interval: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), "--timeline", str(timeline), "--interval-s", interval, "--",
         *command],
        capture_output=True, text=True, check=False, timeout=120,
    )


def test_BREAK_a_spike_between_two_samples_is_missed(tmp_path: Path) -> None:
    """A command holds 400 MiB for a fraction of a second, between samples. The tool reports the
    sampled maximum, so the 'peak' is the interpreter's idle footprint. The kernel keeps the
    answer (``ri_lifetime_max_phys_footprint`` / ``ri_interval_max_phys_footprint``) but the
    tool never reads it. Expected: peak >= 400 MiB. A short-lived ``qd-prep`` child is the
    same shape: it is never in a ``ps`` snapshot at all."""
    child = "b = bytearray(b'\\x01') * (400 << 20)"
    code = (
        "import subprocess, sys, time\n"
        "time.sleep(0.3)\n"
        f"subprocess.run([sys.executable, '-c', {child!r}])\n"
        "time.sleep(1.6)\n"
    )
    done = _run(tmp_path / "t.txt", [sys.executable, "-c", code], "1.0")
    match = re.search(r"peak (\d+) MiB", done.stderr)
    assert match, done.stderr
    assert int(match.group(1)) >= 400, f"reported {match.group(1)} MiB; the command held 400"


def test_BREAK_the_command_is_launched_before_the_timeline_is_openable(tmp_path: Path) -> None:
    """``--timeline`` in a directory that does not exist: the tool dies with a traceback after
    ``Popen``, the command keeps running unmeasured and unsupervised. Expected: refuse before
    launching (the marker is never written)."""
    marker = tmp_path / "ran"
    code = f"import time, pathlib; time.sleep(0.6); pathlib.Path({str(marker)!r}).write_text('x')"
    done = _run(tmp_path / "no-such-dir" / "t.txt", [sys.executable, "-c", code], "0.2")
    assert done.returncode != 0
    time.sleep(1.5)
    assert not marker.exists(), "the command ran to completion with no measurement"


def test_BREAK_a_command_path_with_spaces_loses_the_stage_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ps -o command=`` is split on whitespace, so ``/Users/me/my repo/target/release/qd-prep
    minhash`` is labelled ``my``, not ``qd-prep:minhash``: the timeline cannot be put against the
    stage that was running. (The peak itself is unaffected; low severity.)"""
    out = "10 1 /Users/me/my repo/target/release/qd-prep minhash --input a\n11 10 /usr/bin/true\n"
    monkeypatch.setattr(
        footprint_peak.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=out)
    )
    names = {pid: name for pid, _ppid, name in footprint_peak.process_table()}
    assert names[10] == "qd-prep:minhash", names


def test_a_command_that_ends_at_once_is_never_a_pass_without_samples(tmp_path: Path) -> None:
    done = _run(tmp_path / "t.txt", [sys.executable, "-c", "pass"], "0.05")
    match = re.search(r"exit (\d+), (\d+) samples, (\d+) failed", done.stderr)
    assert match, done.stderr
    if int(match.group(2)) == 0:
        assert done.returncode == footprint_peak.NO_SAMPLE
    else:
        assert done.returncode == 0 and (tmp_path / "t.txt").read_text(encoding="utf-8").strip()


def test_a_failing_command_status_passes_through_with_a_peak(tmp_path: Path) -> None:
    done = _run(tmp_path / "t.txt", [sys.executable, "-c", "import time; time.sleep(0.5); exit(3)"],
                "0.1")
    assert done.returncode == 3
    assert int(re.search(r"peak (\d+) MiB", done.stderr).group(1)) > 0


def test_a_signal_killed_command_is_not_reported_as_success(tmp_path: Path) -> None:
    code = "import os, signal, time; time.sleep(0.3); os.kill(os.getpid(), signal.SIGKILL)"
    done = _run(tmp_path / "t.txt", [sys.executable, "-c", code], "0.1")
    assert done.returncode not in (0, footprint_peak.NO_SAMPLE) and done.returncode > 128
