"""Count `unwrap()`/`expect()` in production Rust paths, excluding inline test modules.

The repo rule is "no unwrap() on anything that can fail in production paths (tests are
fine)". A plain `rg -c` cannot honour that exemption, because Rust's test modules live
*inside* the source file behind `#[cfg(test)]` — so a naive count conflates a panicking
production path with a perfectly legitimate test assertion, and the rule looks violated
when it is not (or, worse, looks satisfied when a real one hides among them).

Everything from the first `#[cfg(test)]` line onward is treated as test code.

Comment lines are skipped too. The first version of this script reported exactly one
production `unwrap()` in the workspace, and it turned out to be inside a `//` comment
documenting the `let g = m.lock().unwrap()` pattern that a mutation operator looks for.
A checker that counts prose as code cries wolf once and is then ignored, which is worse
than not having it.

This is a line-based heuristic, stated as one: it skips `//` lines and `/* */` blocks
but does not parse Rust, so an `.unwrap()` inside a string literal would still be
counted. That is deliberate — over-reporting a literal is cheap to dismiss, and the
alternative is a Rust parser in a lint script.
"""

from __future__ import annotations

import pathlib
import re
import sys

PATTERN = re.compile(r"\.unwrap\(\)|\.expect\(")


def split_at_test_module(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.strip().startswith("#[cfg(test)]"):
            return i
    return len(lines)


def code_lines(lines: list[str]) -> list[tuple[int, str]]:
    """(1-based line number, text) for lines that are not comments."""
    out: list[tuple[int, str]] = []
    in_block = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if in_block:
            if "*/" in line:
                in_block = False
                line = line.split("*/", 1)[1].strip()
            else:
                continue
        if line.startswith("/*"):
            if "*/" not in line:
                in_block = True
                continue
            line = line.split("*/", 1)[1].strip()
        if line.startswith("//"):
            continue
        # Strip a trailing line comment so `foo.unwrap(); // note` still counts once
        # and `foo(); // mentions .unwrap()` does not count at all.
        if "//" in line:
            line = line.split("//", 1)[0]
        if line.strip():
            out.append((i + 1, line))
    return out


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    total_prod = 0
    rows: list[tuple[int, int, str, list[int]]] = []

    for path in sorted(root.rglob("src/**/*.rs")):
        lines = path.read_text(encoding="utf-8").splitlines()
        cut = split_at_test_module(lines)
        prod_lines = [n for n, line in code_lines(lines[:cut]) if PATTERN.search(line)]
        test_count = sum(1 for _, line in code_lines(lines[cut:]) if PATTERN.search(line))
        if prod_lines:
            total_prod += len(prod_lines)
            rows.append((len(prod_lines), test_count, str(path.relative_to(root)), prod_lines))

    for prod, test, rel, where in sorted(rows, reverse=True):
        print(f"  {prod:3d} production ({test:3d} in tests)  {rel}")
        print(f"        lines: {where[:12]}{' ...' if len(where) > 12 else ''}")

    print(f"\nTOTAL unwrap/expect on production paths: {total_prod}")
    return 1 if total_prod else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
