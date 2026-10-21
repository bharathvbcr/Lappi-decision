"""``tools/footprint_peak.py`` measures a command's tree peak phys_footprint and passes its exit on.

The tool is how the v6 containment scan's "peak below memguard's line" claim is measured, so it
is tested against a process that really holds a known amount of memory, in a child of the
command (the sum is over the tree, not the root).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "footprint_peak.py"
sys.path.insert(0, str(REPO / "tools"))

import footprint_peak  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="reads libproc")

HOLD_MIB = 300
#: Allocates and touches HOLD_MIB in a grandchild, then outlives a few samples.
HOLDER = (
    "import subprocess, sys, time\n"
    f"child = 'b = bytearray(b\"\\\\x01\") * ({HOLD_MIB} << 20); import time; time.sleep(3)'\n"
    "subprocess.run([sys.executable, '-c', child])\n"
)


def _run(tmp_path: Path, command: list[str]) -> tuple[subprocess.CompletedProcess[str], Path]:
    timeline = tmp_path / "timeline.txt"
    done = subprocess.run(
        [sys.executable, str(TOOL), "--timeline", str(timeline), "--interval-s", "0.2", "--",
         *command],
        capture_output=True, text=True, check=False, timeout=120,
    )
    return done, timeline


def test_the_peak_counts_a_descendants_memory_and_the_exit_is_the_commands(tmp_path: Path) -> None:
    done, timeline = _run(tmp_path, [sys.executable, "-c", HOLDER + "sys.exit(7)\n"])
    match = re.search(r"peak (\d+) MiB, wall [\d.]+ s, exit (\d+), (\d+) samples", done.stderr)
    assert match, done.stderr
    assert done.returncode == 7 and match.group(2) == "7"
    assert int(match.group(1)) >= HOLD_MIB, done.stderr
    lines = timeline.read_text(encoding="utf-8").splitlines()
    assert len(lines) == int(match.group(3)) >= 3
    # The interpreter's name is "python3.13" from a venv and "Python" from a framework build.
    assert any(":python" in line.split(None, 2)[-1].lower() for line in lines)


def test_an_existing_timeline_is_refused(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.txt"
    timeline.write_text("keep\n", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(TOOL), "--timeline", str(timeline), "--",
         sys.executable, "-c", "pass"],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 2
    assert timeline.read_text(encoding="utf-8") == "keep\n"


def test_descendants_follow_parent_links_and_ignore_other_trees() -> None:
    table = [(1, 0, "init"), (10, 1, "a"), (11, 10, "b"), (12, 11, "c"), (20, 1, "other")]
    assert footprint_peak.descendants(10, table) == {10: "a", 11: "b", 12: "c"}
    assert footprint_peak.descendants(99, table) == {}


def test_a_spike_the_root_freed_before_the_next_sample_is_read_from_the_kernel(
    tmp_path: Path,
) -> None:
    """The root holds 300 MiB for an instant and then idles for seconds: no sample sees it, the
    kernel's lifetime max (ri_lifetime_max_phys_footprint) does. The summary names the stage and
    says the peak is a lower bound."""
    code = (
        "import time\n"
        f"b = bytearray(b'\\x01') * ({HOLD_MIB} << 20)\n"
        "del b\n"
        "time.sleep(2.5)\n"
    )
    done, _timeline = _run(tmp_path, [sys.executable, "-c", code])
    stages = re.search(r"per stage, lifetime max MiB: \{(.*)\}", done.stderr)
    assert stages, done.stderr
    held = [int(v) for v in re.findall(r": (\d+)", stages.group(1))]
    assert max(held) >= HOLD_MIB, done.stderr
    assert "LOWER BOUND" in done.stderr


def test_sampling_spawns_no_process_and_still_names_the_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``ps`` run inside a mac_heavy job is a setuid tree member whose footprint the cap's
    reader cannot read (EPERM); three such failures in a row stop the job. The sampling loop walks
    the tree through libproc, so it must work with ``subprocess.run`` unavailable."""
    import ctypes

    from mac_footprint import child_pids, command_line

    def no_spawn(*_a: object, **_k: object) -> None:
        raise AssertionError("a process was spawned to sample the tree")

    monkeypatch.setattr(footprint_peak.subprocess, "run", no_spawn)
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    child = subprocess.Popen(["sleep", "30"])
    try:
        assert child.pid in child_pids(libproc, os.getpid())
        assert "sleep 30" in command_line(libproc, child.pid)
        tally = footprint_peak.Tally()
        got = tally.take(libproc, os.getpid())
        assert got is not None
        assert "sleep" in tally.stages()
    finally:
        child.kill()
        child.wait()


@pytest.mark.parametrize(("command", "name"), [
    ("/Users/me/my repo/target/release/qd-prep minhash --input a", "qd-prep:minhash"),
    ("/x/qd-prep lsh --input a --output b", "qd-prep:lsh"),
    ("/x/target/release/qd-prep", "qd-prep"),
    ("/usr/bin/python3 -u tools/containment_scan.py", "python3"),
    ("python3 /x/qd-prep minhash", "python3"),
    ("", ""),
])
def test_stage_names(command: str, name: str) -> None:
    assert footprint_peak.stage_name(command) == name
