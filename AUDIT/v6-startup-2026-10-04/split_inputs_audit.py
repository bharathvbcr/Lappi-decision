"""Which files does the split rebuild read? Every one, observed, so the split cache's key can be
checked against it.

Analysis only (stdlib + the repo's own modules); never imported by a tool. Runs
``tools/real_ft_run.py``'s ``main`` with the given argv and, while ``ft_split_rows`` runs and
only then, records through a ``sys.audit`` hook installed in THIS analysis process:

- every ``open`` of a path (str, bytes or PathLike; an fd is skipped), with whether it was opened
  for reading only;
- every ``os.listdir`` / ``os.scandir`` of a path;
- every ``subprocess.Popen`` executable and argv[0].

When ``ft_split_rows`` returns it writes ``OUT_JSON``: the regular files that were opened for
reading only, still exist, and are not ``.pyc`` -- each with its size, whether it sits under
``sys.prefix``/``sys.base_prefix`` (the interpreter and its packages) or is a ``*.py`` under the
repo's ``python/`` or ``tools/`` (the code the cache keys as code) -- plus the directories
listed and the executables spawned; then stops (``SystemExit``) before the val set.

What it cannot see: reads made outside Python's ``open`` (a C library opening a file itself, or
a subprocess's own reads). The rebuild's readers are Python ``open``/``Path.read_*`` calls on
JSON/JSONL/text; the subprocesses are ``qd-prep``, whose inputs are request files this process
wrote. Stated in the HANDOFF beside the result.

Usage: python split_inputs_audit.py <OUT_JSON> <real_ft_run.py argv...>
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
from pathlib import Path
from typing import Any

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "tools"))
sys.path.insert(0, str(WT / "python"))

import real_ft_run as rft  # noqa: E402

_LOCK = threading.Lock()
_ACTIVE = False
_OPENS: dict[str, bool] = {}  # abspath -> read-only on every open
_LISTED: set[str] = set()
_SPAWNED: set[str] = set()
_ERRORS: list[str] = []


def _path(value: Any) -> str | None:
    if isinstance(value, int):
        return None
    try:
        raw = os.fsdecode(os.fspath(value))
    except TypeError:
        return None
    return str(Path(raw).absolute())


def _read_only(mode: Any, flags: Any) -> bool:
    if isinstance(mode, str):
        return "r" in mode and not any(c in mode for c in "wax+")
    if isinstance(flags, int):
        return flags & (os.O_WRONLY | os.O_RDWR) == 0
    return False


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if not _ACTIVE:
        return
    try:
        if event == "open":
            path = _path(args[0])
            if path is not None:
                ro = _read_only(args[1] if len(args) > 1 else None,
                                args[2] if len(args) > 2 else None)
                with _LOCK:
                    _OPENS[path] = _OPENS.get(path, True) and ro
        elif event in ("os.listdir", "os.scandir"):
            path = _path(args[0])
            if path is not None:
                with _LOCK:
                    _LISTED.add(path)
        elif event == "subprocess.Popen":
            executable, argv = args[0], args[1]
            first = argv[0] if isinstance(argv, (list, tuple)) and argv else argv
            for v in (executable, first):
                p = _path(v) if v is not None else None
                if p is not None:
                    with _LOCK:
                        _SPAWNED.add(p)
    except Exception as exc:  # a hook must never raise into the audited call
        _ERRORS.append(f"{event}: {exc!r}")


def _under(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def main() -> None:
    out_json = Path(sys.argv[1])
    argv = sys.argv[2:]
    inner = rft.ft_split_rows
    prefixes = sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
    code_roots = [os.path.realpath(WT / "python"), os.path.realpath(WT / "tools")]

    def audited(**kw: Any) -> Any:
        global _ACTIVE
        sys.addaudithook(_hook)
        _ACTIVE = True
        try:
            train, val = inner(**kw)
        finally:
            _ACTIVE = False
        files = []
        for path, ro in sorted(_OPENS.items()):
            real = os.path.realpath(path)
            try:
                st = Path(real).stat()
            except OSError:
                continue  # a temp file the rebuild wrote, read and removed
            if not stat.S_ISREG(st.st_mode) or real.endswith(".pyc"):
                continue
            files.append({
                "path": path, "realpath": real, "bytes": st.st_size, "read_only": ro,
                "interpreter": _under(real, prefixes),
                "code_py": real.endswith(".py") and _under(real, code_roots),
            })
        report = {
            "argv": argv,
            "kwargs": {k: (str(v) if isinstance(v, Path) else repr(v)) for k, v in kw.items()},
            "rows": {"train": len(train), "val": len(val)},
            "files": files,
            "listed": sorted(_LISTED),
            "spawned": sorted(_SPAWNED),
            "hook_errors": _ERRORS,
        }
        out_json.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
        data = [f for f in files if not f["interpreter"] and not f["code_py"]]
        print(f"[inputs] {len(files)} files read ({len(data)} neither interpreter nor keyed code), "
              f"{len(_LISTED)} dirs listed, {len(_SPAWNED)} executables, "
              f"{len(_ERRORS)} hook errors -> {out_json}", file=sys.stderr, flush=True)
        raise SystemExit("split_inputs_audit: recorded; stopping before the val set")

    rft.ft_split_rows = audited
    rft.main(argv)


if __name__ == "__main__":
    main()
