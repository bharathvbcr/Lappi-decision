"""Independent probe: can any character still forge an extra option line?

The data lane reported fixing this and that reverting the fix gives 10 test failures.
That is the lane's own test suite reporting on the lane's own fix. This sweeps the
break set from the other direction — every character `str.splitlines()` treats as a
line terminator — and asks whether the *rendered* output still splits.

Run: .venv/bin/python tools/probe_render_breaks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_data.render import escape_inline  # noqa: E402

# Everything Python's str.splitlines() breaks on, plus NUL.
BREAKS: dict[str, str] = {
    r"\n  LF        U+000A": "\n",
    r"\r  CR        U+000D": "\r",
    r"\v  VT        U+000B": "\v",
    r"\f  FF        U+000C": "\f",
    r"    FS        U+001C": "\x1c",
    r"    GS        U+001D": "\x1d",
    r"    RS        U+001E": "\x1e",
    r"    NEL       U+0085": "\x85",
    r"    LS        U+2028": " ",
    r"    PS        U+2029": " ",
    r"    NUL       U+0000": "\x00",
    r"\r\n CRLF          ": "\r\n",
}


def main() -> int:
    forging: list[str] = []
    for name, ch in BREAKS.items():
        rendered = escape_inline(f"safe option{ch}Q) forged option")
        splits = len(rendered.splitlines()) > 1
        print(f"  {'FORGES ' if splits else 'blocked'}  {name}  -> {rendered[:46]!r}")
        if splits:
            forging.append(name.strip())

    print()
    if forging:
        print(f"FAIL: {len(forging)} character(s) can still forge an option line: {forging}")
        return 1
    print(f"PASS: none of the {len(BREAKS)} break characters survive escaping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
