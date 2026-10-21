"""``tools/footprint_peak.py`` measures a command's tree peak phys_footprint and passes its exit on.

The tool is how the v6 containment scan's "peak below memguard's line" claim is measured, so it
is tested against a process that really holds a known amount of memory, in a child of the
command (the sum is over the tree, not the root).
"""

from __future__ import annotations

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
    "child = 'b = bytearray(b\"\\\\x01\") * (%d << 20); import time; time.sleep(3)'\n"
    "subprocess.run([sys.executable, '-c', child])\n" % HOLD_MIB
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
    assert any(":python" in line.split(None, 2)[-1] for line in lines)


def test_an_existing_timeline_is_refused(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.txt"
    timeline.write_text("keep\n", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(TOOL), "--timeline", str(timeline), "--", sys.executable, "-c", "pass"],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 2
    assert timeline.read_text(encoding="utf-8") == "keep\n"


def test_descendants_follow_parent_links_and_ignore_other_trees() -> None:
    table = [(1, 0, "init"), (10, 1, "a"), (11, 10, "b"), (12, 11, "c"), (20, 1, "other")]
    assert footprint_peak.descendants(10, table) == {10: "a", 11: "b", 12: "c"}
    assert footprint_peak.descendants(99, table) == {}
