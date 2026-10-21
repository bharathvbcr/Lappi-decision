"""Run one command and record its process tree's peak phys_footprint.

    python3 tools/footprint_peak.py --timeline FILE -- <command> [args...]

``tools/mac_heavy.sh`` kills a job over its footprint cap but never says how close a passing job
came. This runs beside it (``bash tools/mac_heavy.sh LABEL python3 tools/footprint_peak.py ...``)
and measures with the same reader (``tools/mac_footprint.py``: ``ri_phys_footprint``, the figure
memguard and Jetsam act on), summed over the command and every descendant. The sum is the number
mac_heavy compares to its cap, so it is the number a "below the cap" claim must use.

Every ``--interval-s`` seconds it appends one line to the timeline: UTC time, the tree's footprint
in MiB, then ``pid:name:MiB`` for each process over 64 MiB, so the peak can be put against the
stage that was running (a ``qd-prep minhash`` child, a ``qd-prep lsh`` child, the Python parent
alone). On exit it prints one summary line (peak MiB, wall seconds, command exit status, samples,
failed reads) and exits with the command's status. A sample the reader could not take is counted
and reported, never read as 0 MiB: a run with no sample at all exits 70, not 0.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mac_footprint import phys_footprint_bytes

#: Exit status when the command ran but no footprint sample could be read.
NO_SAMPLE: Final[int] = 70
#: Processes under this are left out of a timeline line's per-process list.
LISTED_MIB: Final[int] = 64
MIB: Final[int] = 1024 * 1024


def descendants(root: int, table: list[tuple[int, int, str]]) -> dict[int, str]:
    """``{pid: name}`` of ``root`` and every descendant in a ``(pid, ppid, name)`` table."""
    names = {pid: name for pid, _ppid, name in table}
    children: dict[int, list[int]] = {}
    for pid, ppid, _name in table:
        children.setdefault(ppid, []).append(pid)
    found: dict[int, str] = {}
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in found or pid not in names:
            continue
        found[pid] = names[pid]
        stack.extend(children.get(pid, ()))
    return found


def process_table() -> list[tuple[int, int, str]]:
    """``(pid, ppid, name)`` per process. The name is the executable's basename, plus the first
    argument for a ``qd-prep`` process: its subcommand (``minhash``, ``lsh``, ``containment``) is
    which stage of a scan the child is."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True,
    ).stdout
    table: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            name = Path(parts[2]).name
            if name == "qd-prep" and len(parts) == 4:
                name += ":" + parts[3].split(None, 1)[0]
            table.append((int(parts[0]), int(parts[1]), name))
    return table


def sample(libproc: ctypes.CDLL, root: int) -> tuple[int, list[tuple[int, str, int]]] | None:
    """``(tree MiB, [(pid, name, MiB)])`` for the tree under ``root``; ``None`` if nothing read."""
    per: list[tuple[int, str, int]] = []
    total = 0
    for pid, name in descendants(root, process_table()).items():
        try:
            nbytes = phys_footprint_bytes(libproc, pid)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                continue
            return None
        total += nbytes
        per.append((pid, name, -(-nbytes // MIB)))
    return (-(-total // MIB), per) if per else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeline", type=Path, required=True,
                        help="the file the per-sample lines are appended to; must not exist")
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command; give it after --")
    if not 0.05 <= args.interval_s <= 60:
        parser.error("--interval-s is between 0.05 and 60 seconds")
    if args.timeline.exists():
        parser.error(f"{args.timeline} exists; refusing to overwrite a timeline")
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    peak = 0
    taken = 0
    failed = 0
    started = time.monotonic()
    proc = subprocess.Popen(command)
    with args.timeline.open("x", encoding="utf-8") as fh:
        while True:
            got = sample(libproc, proc.pid)
            if got is None:
                if proc.poll() is None:
                    failed += 1
            else:
                total, per = got
                taken += 1
                peak = max(peak, total)
                listed = " ".join(f"{p}:{n}:{m}" for p, n, m in per if m >= LISTED_MIB)
                fh.write(f"{datetime.now(UTC).strftime('%H:%M:%SZ')} {total} {listed}\n")
                fh.flush()
            try:
                status = proc.wait(timeout=args.interval_s)
                break
            except subprocess.TimeoutExpired:
                continue
    wall = time.monotonic() - started
    print(f"footprint_peak: peak {peak} MiB, wall {wall:.1f} s, exit {status}, {taken} samples, "
          f"{failed} failed reads, command {command[0]}", file=sys.stderr, flush=True)
    if taken == 0:
        return NO_SAMPLE
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    raise SystemExit(main())
