"""L-lint throwaway: what makes each E501 line long?

Reads a concise ruff report (argv[1]) and, for every E501 line, says which token kind
covers it: a line inside a multi-line (triple-quoted) string, a comment, a line carrying
a string or f-string literal, or plain code. Prints one row per file. Run from the
worktree root.
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

FSTRING_START = getattr(tokenize, "FSTRING_START", -1)


def classify(src: str, line_no: int) -> str:
    kind = "code"
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.start[0] > line_no:
            break
        if not tok.start[0] <= line_no <= tok.end[0]:
            continue
        if tok.type == tokenize.STRING and tok.start[0] < tok.end[0]:
            return "inside-multiline-string"
        if tok.type == tokenize.COMMENT:
            kind = "comment"
        elif tok.type in (tokenize.STRING, FSTRING_START) and kind == "code":
            kind = "string-on-line"
    return kind


def main() -> int:
    report = Path(sys.argv[1]).read_text().splitlines()
    hits: dict[str, list[int]] = defaultdict(list)
    for row in report:
        m = re.match(r"^(\S+):(\d+):\d+: E501", row)
        if m:
            hits[m.group(1)].append(int(m.group(2)))
    for path, line_nos in sorted(hits.items()):
        src = Path(path).read_text()
        kinds = Counter(classify(src, n) for n in line_nos)
        print(f"{path} {len(line_nos)} {dict(kinds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
