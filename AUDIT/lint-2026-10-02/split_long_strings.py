"""L-lint throwaway: split over-long single-line string literals by implicit concatenation.

    python AUDIT/lint-2026-10-02/split_long_strings.py FILE [FILE ...]

Deliberately dumb. For each line longer than LIMIT characters it looks for the one string
literal that straddles column LIMIT, and splits it after the last space that keeps the
first piece within the limit. When the literal ends its line and the next line opens with
a literal of the same prefix and quote (a run of adjacent literals, one per line), the
remainder is carried into the start of that next literal, so a paragraph reflows instead
of growing stub lines; the next line is then re-checked on the next pass. Otherwise the
remainder goes on a new line, indented to the literal's own start column, and carries
whatever followed the literal. Adjacent literals are joined by the parser into one
constant, so the program is unchanged; ast_equal.py is the check that says so, file by
file.

A line is left alone (and listed on stderr, for a hand edit) when:

* no single-line literal straddles the limit (the length is code, or a comment);
* the literal is triple-quoted;
* the literal sits at bracket depth 0, where a line break would end the statement;
* the literal has no top-level space far enough left to split at.

Escapes are atomic (a split never lands inside ``\\x``, ``\\N{...}`` and the like), and an
f-string is split only in its literal text, never inside a replacement field. An f-string
piece left with no replacement field loses its ``f`` prefix and has its doubled braces
undoubled, so ruff's F541 does not fire on it.

Repeats until no line changes. Prints, per file, how many splits it made.
"""

from __future__ import annotations

import io
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

LIMIT = 100
OPEN = {"(", "[", "{"}
CLOSE = {")", "]", "}"}


@dataclass
class Literal:
    line: int  # 1-based
    start: int  # 0-based column of the prefix
    end: int  # 0-based column one past the closing quote
    depth: int  # bracket depth outside any f-string


def literals(src: str) -> list[Literal]:
    """Single-line string literals (plain or f-string) at the top level of the code."""
    out: list[Literal] = []
    depth = 0
    fstack: list[tuple[int, int]] = []  # (line, col) of each open FSTRING_START
    fstart = getattr(tokenize, "FSTRING_START", None)
    fend = getattr(tokenize, "FSTRING_END", None)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if fstart is not None and tok.type == fstart:
            fstack.append(tok.start)
            continue
        if fend is not None and tok.type == fend:
            begin = fstack.pop()
            if not fstack and begin[0] == tok.end[0]:
                out.append(Literal(begin[0], begin[1], tok.end[1], depth))
            continue
        if fstack:
            continue
        if tok.type == tokenize.OP:
            if tok.string in OPEN:
                depth += 1
            elif tok.string in CLOSE:
                depth -= 1
        elif tok.type == tokenize.STRING and tok.start[0] == tok.end[0]:
            out.append(Literal(tok.start[0], tok.start[1], tok.end[1], depth))
    return out


def split_points(body: str, raw: bool, fmt: bool) -> tuple[list[int], list[int]]:
    """Offsets k where body[:k] ends in a top-level space, and where it ends in a \\n escape."""
    points: list[int] = []
    newlines: list[int] = []
    i = 0
    field = 0
    while i < len(body):
        c = body[i]
        if not raw and c == "\\":
            if body.startswith("N{", i + 1):
                close = body.find("}", i)
                i = close + 1 if close != -1 else len(body)
            else:
                if body.startswith("n", i + 1) and field == 0:
                    newlines.append(i + 2)
                i += 2
            continue
        if fmt:
            if field == 0:
                if body.startswith("{{", i) or body.startswith("}}", i):
                    i += 2
                    continue
                if c == "{":
                    field = 1
                    i += 1
                    continue
            else:
                if c == "{":
                    field += 1
                elif c == "}":
                    field -= 1
                i += 1
                continue
        if c == " ":
            points.append(i + 1)
        i += 1
    return points, newlines


def has_field(body: str) -> bool:
    return bool(body.replace("{{", "").replace("}}", "").count("{"))


def render(prefix: str, quote: str, body: str) -> str:
    if "f" in prefix.lower() and not has_field(body):
        prefix = prefix.replace("f", "").replace("F", "")
        body = body.replace("{{", "{").replace("}}", "}")
    return f"{prefix}{quote}{body}{quote}"


def parts(token: str) -> tuple[str, str, str, bool]:
    """(prefix, quote, body, triple) of one literal's source text."""
    q_at = min(i for i in (token.find("'"), token.find('"')) if i != -1)
    prefix, quote = token[:q_at], token[q_at]
    triple = token[q_at : q_at + 3] == quote * 3
    return prefix, quote, token[q_at + 1 : -1], triple


def follower(lines: list[str], lit: Literal, found: list[Literal]) -> Literal | None:
    """The literal that opens the next line, when ``lit`` is the last token on its own.

    That is a run of adjacent literals, one per line. The overflow of ``lit`` can then be
    carried into the start of the next literal (a reflow) instead of becoming a stub line.
    """
    if lines[lit.line - 1].rstrip("\r\n")[lit.end :].strip() or lit.line >= len(lines):
        return None
    nxt_text = lines[lit.line]
    col = len(nxt_text) - len(nxt_text.lstrip(" "))
    for cand in found:
        if cand.line == lit.line + 1 and cand.start == col and cand.depth == lit.depth:
            return cand
    return None


def split_once(src: str) -> str | None:
    lines = src.splitlines(keepends=True)
    found = literals(src)
    for lit in found:
        text = lines[lit.line - 1]
        bare = text.rstrip("\r\n")
        if len(bare) <= LIMIT or not (lit.start < LIMIT < lit.end):
            continue
        prefix, quote, body, triple = parts(bare[lit.start : lit.end])
        if triple or lit.depth == 0:
            continue
        raw = "r" in prefix.lower()
        fmt = "f" in prefix.lower()
        room = LIMIT - lit.start - len(prefix) - 2
        spaces, newlines = split_points(body, raw, fmt)
        # Prefer to break after a newline escape (a literal holding source code then breaks at
        # its own line ends), unless that would leave the first piece under a third full.
        at_nl = [k for k in newlines if room // 3 <= k <= room and k < len(body)]
        fits = at_nl or [k for k in spaces if k <= room and k < len(body)]
        if not fits:
            continue
        k = fits[-1]
        head = render(prefix, quote, body[:k])
        newline = text[len(bare) :]
        nxt = follower(lines, lit, found)
        if nxt is not None:
            nxt_bare = lines[nxt.line - 1].rstrip("\r\n")
            n_prefix, n_quote, n_body, n_triple = parts(nxt_bare[nxt.start : nxt.end])
            if (n_prefix, n_quote, n_triple) == (prefix, quote, False):
                merged = render(prefix, quote, body[k:] + n_body)
                lines[lit.line - 1] = bare[: lit.start] + head + newline
                lines[nxt.line - 1] = (
                    nxt_bare[: nxt.start]
                    + merged
                    + nxt_bare[nxt.end :]
                    + lines[nxt.line - 1][len(nxt_bare) :]
                )
                return "".join(lines)
        tail = render(prefix, quote, body[k:])
        lines[lit.line - 1] = (
            bare[: lit.start] + head + "\n" + " " * lit.start + tail + bare[lit.end :] + newline
        )
        return "".join(lines)
    return None


def main() -> int:
    for name in sys.argv[1:]:
        path = Path(name)
        src = path.read_text(encoding="utf-8")
        count = 0
        while (new := split_once(src)) is not None:
            src = new
            count += 1
        path.write_text(src, encoding="utf-8")
        print(f"{name}: {count} splits")
        for line_no, line in enumerate(src.splitlines(), 1):
            if len(line) > LIMIT:
                print(f"  left for a hand edit: {name}:{line_no} ({len(line)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
