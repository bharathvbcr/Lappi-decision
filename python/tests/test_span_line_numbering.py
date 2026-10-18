"""Span line numbers are counted on LF alone, the rule qd-mutate numbers by (M21).

``str.splitlines()`` also splits on CR, VT, FF, FS, GS, RS, U+0085, U+2028 and U+2029, so a
changed-line span numbered with it is off by one after any of them when read by qd-mutate
(GAP-COMMITPACKFT-CHANGED-LINE-SPANS-SPLITLINES-VS-QD-MUTATE-NEWLINES-2026-10-12).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from qd_data.lines import lf_lines  # noqa: E402

SEPARATORS = ["\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", chr(0x2028), chr(0x2029)]


def _id(sep: str) -> str:
    return f"U+{ord(sep):04X}"


def _qd_mutate_span(old: str, new: str) -> tuple[int, int]:
    """The changed span exactly as qd-mutate's diffspan reads it: split on LF, trim the common
    prefix and suffix, 1-based inclusive."""
    b, a = old.split("\n"), new.split("\n")
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(a), len(b)) - prefix and a[-1 - suffix] == b[-1 - suffix]:
        suffix += 1
    return prefix + 1, len(a) - suffix


@pytest.mark.parametrize("sep", SEPARATORS, ids=_id)
def test_changed_line_spans_agree_with_lf_numbering(sep):
    import build_commitpackft_pool as bcp

    old = f"a{sep}b\nc\nd\n"
    new = f"a{sep}b\nc\nD\n"
    assert bcp.changed_line_spans(old, new) == [(3, 3)]
    assert bcp.changed_line_spans(old, new)[0] == _qd_mutate_span(old, new)


@pytest.mark.parametrize("sep", SEPARATORS, ids=_id)
def test_lf_lines_does_not_split_on_the_separator(sep):
    assert lf_lines(f"x{sep}y\nz") == [f"x{sep}y", "z"]


def test_lf_lines_keeps_the_phantom_final_line():
    assert lf_lines("a\nb\n") == ["a", "b", ""]
    assert lf_lines("a\r\nb") == ["a\r", "b"]
    assert lf_lines("") == [""]


def test_no_span_path_site_uses_splitlines():
    """Every function that numbers lines for a span label goes through lf_lines."""
    sites = {"tools/build_commitpackft_pool.py": ["changed_line_spans"]}
    for rel, names in sites.items():
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for name in names:
            assert name in fns, f"{rel}:{name} moved; update this test"
            calls = [
                n.func.attr
                for n in ast.walk(fns[name])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            ]
            assert "splitlines" not in calls, f"{rel}:{name} calls .splitlines()"
