"""Run one command and report its process tree's footprint, with the lower bound named as one.

    python3 tools/footprint_peak.py --timeline FILE -- <command> [args...]

``tools/mac_heavy.sh`` kills a job over its cap but never says how close a passing job came. This
runs beside it (``bash tools/mac_heavy.sh LABEL python3 tools/footprint_peak.py ...``) and reads
with the same reader, ``tools/mac_footprint.py`` (``proc_pid_rusage``: ``ri_phys_footprint``, the
figure memguard and Jetsam act on). There is one reader; this file only runs the command, walks
the tree and reports.

What is measured, and what each number can claim:

* ``sampled tree``: the sum of the tree's current footprints at one sample. The peak of those
  is a LOWER bound of the tree's true peak: a process that allocates and exits between two
  samples is in none of them.
* ``lifetime max`` per process: ``ri_lifetime_max_phys_footprint``, kept by the kernel, read at
  every sample. Exact for any process this saw alive at least once, however briefly it held its
  memory. One process's lifetime max is also a lower bound of the tree's peak.
* ``reaped rss``: the exit report (``wait4``'s ``ru_maxrss``) of the command, which covers every
  descendant that was waited for, however short-lived. It is RSS, not footprint, and it names
  the largest single process only. It is what catches a child that was never sampled.
* ``seen sum``: the sum of the lifetime maxima of the processes seen. It is not a bound of
  anything unseen, and it over-counts when the processes did not overlap.

The headline ``peak N MiB`` is the largest of the lower bounds and is labelled as one. Every
``--interval-s`` seconds one line goes to the timeline: UTC time, the sampled tree sum in MiB,
then ``pid:name:MiB`` for each process over 64 MiB. The summary also names each process stage
(``Python``, ``qd-prep:minhash``, ``qd-prep:lsh``, ``qd-prep:containment``...) with the largest
lifetime max among its processes. A sample the reader could not take is counted, never read as
0 MiB: a run with no sample at all exits 70, not 0. The timeline is opened before the command
starts, so a path that cannot be written refuses instead of leaving the command unmeasured.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mac_footprint import rusage

#: Exit status when the command ran but no footprint sample could be read.
NO_SAMPLE: Final[int] = 70
#: Processes under this are left out of a timeline line's per-process list.
LISTED_MIB: Final[int] = 64
MIB: Final[int] = 1024 * 1024
#: An absolute path to a ``qd-prep`` executable (spaces allowed in the path), then its subcommand.
_QD_PREP: Final = re.compile(r"^/(?:.*/)?qd-prep(?:\s+([A-Za-z][\w-]*))?(?:\s|$)")


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


def stage_name(command: str) -> str:
    """The executable's basename, and for ``qd-prep`` its subcommand (``minhash``, ``lsh``,
    ``containment``): which stage of a scan the child is. ``ps`` gives the command line as one
    string, so a path with spaces is read as a whole, not split at the first space."""
    match = _QD_PREP.match(command)
    if match:
        return "qd-prep" + (f":{match.group(1)}" if match.group(1) else "")
    return Path(command.split(None, 1)[0]).name if command.strip() else ""


def process_table() -> list[tuple[int, int, str]]:
    """``(pid, ppid, stage_name)`` per process."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True,
    ).stdout
    table: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            table.append((int(parts[0]), int(parts[1]), stage_name(parts[2])))
    return table


class Tally:
    """What the samples add up to."""

    def __init__(self) -> None:
        self.sampled_tree = 0
        self.samples = 0
        self.failed = 0
        #: pid -> (name, largest ``ri_lifetime_max_phys_footprint`` read, bytes)
        self.lifetime: dict[int, tuple[str, int]] = {}

    def take(
        self, libproc: ctypes.CDLL, root: int
    ) -> tuple[int, list[tuple[int, str, int]]] | None:
        """One sample of the tree under ``root``: ``(tree MiB, [(pid, name, MiB)])`` or ``None``
        when nothing could be read. Also folds each process's kernel lifetime max in."""
        per: list[tuple[int, str, int]] = []
        total = 0
        for pid, name in descendants(root, process_table()).items():
            try:
                info = rusage(libproc, pid)
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    continue
                return None
            now = int(info.ri_phys_footprint)
            total += now
            kept_name, kept = self.lifetime.get(pid, ("", 0))
            if kept_name and name == "<defunct>":
                name = kept_name  # a zombie's command line is gone; keep what it was called
            self.lifetime[pid] = (name, max(kept, int(info.ri_lifetime_max_phys_footprint), now))
            per.append((pid, name, -(-now // MIB)))
        if not per:
            return None
        self.samples += 1
        self.sampled_tree = max(self.sampled_tree, total)
        return -(-total // MIB), per

    def stages(self) -> dict[str, int]:
        """Per stage name, the largest lifetime max (MiB) among its processes."""
        out: dict[str, int] = {}
        for name, nbytes in self.lifetime.values():
            out[name] = max(out.get(name, 0), -(-nbytes // MIB))
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def seen_sum(self) -> int:
        return -(-sum(n for _name, n in self.lifetime.values()) // MIB)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeline", type=Path, required=True,
                        help="the file the per-sample lines are written to; must not exist")
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command; give it after --")
    if not 0.05 <= args.interval_s <= 60:
        parser.error("--interval-s is between 0.05 and 60 seconds")
    try:
        timeline = args.timeline.open("x", encoding="utf-8")
    except OSError as exc:
        parser.error(f"--timeline {args.timeline}: {exc}; the command was not started")
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    tally = Tally()
    started = time.monotonic()
    with timeline as fh:
        proc = subprocess.Popen(command)
        while True:
            got = tally.take(libproc, proc.pid)
            if got is None:
                tally.failed += 1
            else:
                total, per = got
                listed = " ".join(f"{p}:{n}:{m}" for p, n, m in per if m >= LISTED_MIB)
                fh.write(f"{datetime.now(UTC).strftime('%H:%M:%SZ')} {total} {listed}\n")
                fh.flush()
            done, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if done:
                break
            time.sleep(args.interval_s)
    wall = time.monotonic() - started
    code = os.waitstatus_to_exitcode(status)
    proc.returncode = code
    reaped_rss = -(-int(usage.ru_maxrss) // MIB)  # macOS reports bytes
    sampled_mib = -(-tally.sampled_tree // MIB)
    stages = tally.stages()
    one_process = max(stages.values(), default=0)
    peak = max(sampled_mib, one_process, reaped_rss)
    print(f"footprint_peak: peak {peak} MiB, wall {wall:.1f} s, exit {code}, {tally.samples} "
          f"samples, {tally.failed} failed reads, command {command[0]}", file=sys.stderr)
    print(f"footprint_peak: LOWER BOUND of the tree's peak: the largest of sampled tree sum "
          f"{sampled_mib} MiB, one process's kernel lifetime max {one_process} MiB, and the "
          f"largest process RSS in the exit report (reaped children included) {reaped_rss} MiB; "
          f"sum of seen processes' lifetime maxima {tally.seen_sum()} MiB; per stage, lifetime "
          f"max MiB: {stages}", file=sys.stderr, flush=True)
    if tally.samples == 0:
        return NO_SAMPLE
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
