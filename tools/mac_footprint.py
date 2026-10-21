"""Summed phys_footprint of the given pids, in MiB, for tools/mac_heavy.sh.

    python3 -I tools/mac_footprint.py <pid> [<pid> ...]

Prints one integer: the sum over the live pids of ri_phys_footprint (proc_pid_rusage, flavor
RUSAGE_INFO_V4), rounded up to whole MiB. phys_footprint is the figure the kernel's memory
accounting and Jetsam act on; it includes Metal/MPS allocations that `ps` RSS does not show.

Summing over a process tree can double-count shared IOKit memory across processes. A GPU job is one
process and the cap is a safety bound, so over-counting is the intended side to err on.

Exit status: 0 on success; 1 when no pid could be read or any pid failed for a reason other than
having exited (ESRCH); 2 on bad usage or a platform without libproc. A pid that exited between the
process-table snapshot and this read is skipped, but if every pid is gone the reader still fails, so
the caller never mistakes "read nothing" for "0 MiB".
"""

from __future__ import annotations

import ctypes
import errno
import sys

RUSAGE_INFO_V4 = 4
_FIELDS = [
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
    "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility",
    "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time",
    "ri_serviced_system_time",
    "ri_logical_writes",
    "ri_lifetime_max_phys_footprint",
    "ri_instructions",
    "ri_cycles",
    "ri_billed_energy",
    "ri_serviced_energy",
    "ri_interval_max_phys_footprint",
    "ri_runnable_time",
]


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _FIELDS]


def rusage(libproc: ctypes.CDLL, pid: int) -> RusageInfoV4:
    """``proc_pid_rusage(pid, RUSAGE_INFO_V4)`` of a live process; ``OSError`` otherwise (ESRCH
    when it has exited)."""
    info = RusageInfoV4()
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libproc.proc_pid_rusage.restype = ctypes.c_int
    if libproc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.addressof(info)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"proc_pid_rusage({pid}) failed")
    return info


def phys_footprint_bytes(libproc: ctypes.CDLL, pid: int) -> int:
    """The process's footprint now: what the cap in tools/mac_heavy.sh sums."""
    return int(rusage(libproc, pid).ri_phys_footprint)


def lifetime_max_phys_footprint_bytes(libproc: ctypes.CDLL, pid: int) -> int:
    """The largest footprint the process has had since it started, kept by the kernel: exact for
    that process however briefly it held the memory, unlike any sample of
    :func:`phys_footprint_bytes`. Readable only while the process is alive."""
    return int(rusage(libproc, pid).ri_lifetime_max_phys_footprint)


def main(argv: list[str]) -> int:
    try:
        pids = [int(a) for a in argv]
    except ValueError:
        print("mac_footprint: pids must be integers", file=sys.stderr)
        return 2
    if not pids or any(p <= 0 for p in pids):
        print("usage: mac_footprint.py <pid> [<pid> ...]", file=sys.stderr)
        return 2
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    except OSError as exc:
        print(f"mac_footprint: libproc is unavailable: {exc}", file=sys.stderr)
        return 2
    total = 0
    read = 0
    for pid in pids:
        try:
            total += phys_footprint_bytes(libproc, pid)
            read += 1
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                continue
            print(f"mac_footprint: {exc}", file=sys.stderr)
            return 1
    if read == 0:
        print("mac_footprint: none of the pids could be read", file=sys.stderr)
        return 1
    print(-(-total // (1024 * 1024)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
