"""The one line-numbering rule for span labels: a line ends at LF and at nothing else.

qd-mutate numbers lines by counting LF bytes (crates/qd-mutate/src/span.rs line_of,
diffspan.rs lines_for_diff and span.rs lines_in_span all split on '\\n'). Python's
str.splitlines() also breaks on CR, VT, FF, FS, GS, RS, U+0085, U+2028 and U+2029, so a span
numbered with it and read by qd-mutate points at the wrong line whenever the file holds one
of them (GAP-COMMITPACKFT-CHANGED-LINE-SPANS-SPLITLINES-VS-QD-MUTATE-NEWLINES-2026-10-12).
Every site that numbers lines for a span label calls lf_lines.
"""

from __future__ import annotations

__all__ = ["lf_lines"]


def lf_lines(text: str) -> list[str]:
    """``text`` split on LF only, terminators removed, the phantom final element kept.

    A text ending in LF yields a trailing empty element, exactly as ``split('\\n')`` does in
    qd-mutate (``total_lines`` counts it too). A CR before the LF stays in the line, as it
    does there.
    """
    return text.split("\n")
