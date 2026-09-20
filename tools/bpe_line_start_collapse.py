"""Measure the span head's line-start collapse against the REAL Qwen tokenizer.

``GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE`` records a refusal in
``qd_train.shards._span_token_positions``: if two context lines map to the same token,
the pointer head cannot tell them apart, so the row is refused rather than deduplicated.
Until this script ran, that refusal had only ever been exercised by a two-token stand-in,
and its sibling records said so -- the case it guards was untriggered and the cost of the
rule was unknown.

**It is not a corner case.** Measured 2026-09-20 over this repository's own tracked
``.py`` and ``.rs`` sources, using the repo's own ``escape_block`` (which is what actually
reaches the tokenizer) and its own ``line_start_indices``:

* every-line-start (today's rule): **60.1% of contexts refused**, 940 line starts lost;
* non-empty line starts: **0.0% refused**, 0 lost;
* non-whitespace line starts: **0.0% refused**, 0 lost.

**One token causes all of it.** Every one of the 940 losses is the single vocabulary entry
``'\\n\\n\\n'``. The arithmetic is exact: that token covers characters ``[i, i+3)``, and a
run of three newlines opens two blank lines, at ``i+1`` and ``i+2`` -- both inside it. A
*single* blank line is ``'\\n\\n'``, which spans ``[i, i+2)``, so the following line starts
at ``i+2``, in the next token, and does not collapse. So the trigger is precisely **two or
more consecutive blank lines**, which is ordinary formatting in most source files.

Why this script and not a test: the repo venv has no ``transformers`` by design
(``python/tests/test_shards.py`` notes the tokenizer is injected as a plain callable so the
line arithmetic needs neither torch nor transformers). Run it with the ML venv:

    /Users/bharath/.venvs/ml/bin/python tools/bpe_line_start_collapse.py

**What this script does NOT do:** it does not change the candidate-set rule. That rule is
owned across both languages -- ``crates/qd-runtime/src/context.rs::Context::line_starts``
pushes a start for every ``\\n``, blank lines included, and the head is sized at
``line_count + RESERVED_NOUL_ROWS``. Excluding blank lines would be a serving-contract
change in both lanes at once, so it is recorded as a decision and not taken here.
"""

from __future__ import annotations

import collections
import subprocess
import sys
from pathlib import Path

# Inlined rather than assigned to REPO first, matching tools/probe_render_breaks.py:
# ruff's E402 exemption covers a `sys.path` modification before the imports, but an
# ordinary assignment in between is not one, and suppressing that with a noqa would be
# hiding the question rather than answering it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_data.render import escape_block
from qd_train.artifacts import line_start_indices

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-2B-Base"
#: Each context is capped so one enormous file cannot dominate the rate. This is
#: CONSERVATIVE with respect to the finding: a longer context has more line starts and
#: therefore more chances to collapse, so the real rate under the render caps is at least
#: this one.
MAX_CHARS = 8000

#: The candidate-set rules compared, named once so the measurement and the report cannot
#: drift into covering different sets.
RULE_NAMES = (
    "A every line start (today)",
    "B non-empty line starts",
    "C non-whitespace line starts",
)


def _line_text(region: str, start: int) -> str:
    end = region.find("\n", start)
    return region[start:] if end < 0 else region[start:end]


def _token_index_for_char(offsets: list[tuple[int, int]], char: int) -> int | None:
    for i, (a, b) in enumerate(offsets):
        if a <= char < b:
            return i
    return None


def _sources() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout
    paths = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        p = REPO / raw.decode()
        if p.suffix in {".py", ".rs"} and p.is_file():
            paths.append(p)
    if not paths:
        raise SystemExit("git listed no sources; refusing to report a vacuous clean scan")
    return paths


def main() -> int:
    # Imported here rather than at module scope: the repo venv deliberately carries no
    # transformers, and a bare ImportError traceback would read like the tool is broken
    # rather than like it is pointed at the wrong interpreter.
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        raise SystemExit(
            "transformers is not importable from this interpreter. The repo venv does not "
            "carry it on purpose; run this with /Users/bharath/.venvs/ml/bin/python."
        ) from None

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"tokenizer {type(tok).__name__} for {MODEL}: vocab={tok.vocab_size} fast={tok.is_fast}")

    refused: collections.Counter[str] = collections.Counter()
    lost: collections.Counter[str] = collections.Counter()
    shapes: collections.Counter[str] = collections.Counter()
    examined = 0

    for path in _sources():
        try:
            region = escape_block(path.read_text(encoding="utf-8")[:MAX_CHARS])
        except (OSError, UnicodeDecodeError):
            continue
        starts = line_start_indices(region)
        if len(starts) < 2:
            continue
        examined += 1
        enc = tok(region, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        ids = list(enc["input_ids"])

        rules = dict(
            zip(
                RULE_NAMES,
                (
                    list(starts),
                    [s for s in starts if _line_text(region, s) != ""],
                    [s for s in starts if _line_text(region, s).strip() != ""],
                ),
                strict=True,
            )
        )
        for name, selected in rules.items():
            if len(selected) < 2:
                continue
            owner: dict[int, list[int]] = collections.defaultdict(list)
            for s in selected:
                t = _token_index_for_char(offsets, s)
                if t is not None:
                    owner[t].append(s)
            collisions = sum(len(v) - 1 for v in owner.values() if len(v) > 1)
            if collisions:
                refused[name] += 1
                lost[name] += collisions
                if name.startswith("A"):
                    for t, v in owner.items():
                        if len(v) > 1:
                            shapes[repr(tok.decode([ids[t]]))] += len(v) - 1

    print(f"corpus: {examined} tracked .py/.rs files with 2+ lines, capped at {MAX_CHARS} chars\n")
    # Iterated from an explicit list, not from the Counter's keys: a rule that refused
    # NOTHING is the result worth seeing, and `Counter.__or__` drops zero counts, so
    # building the row set from the tally would silently hide exactly the rows that carry
    # the finding. Reporting a zero and reporting nothing are different statements.
    for name in RULE_NAMES:
        n = refused[name]
        pct = 100.0 * n / examined if examined else 0.0
        print(
            f"  {name:30} refused {n:4}/{examined:<4} = {pct:5.1f}%"
            f"   line starts lost: {lost[name]}"
        )

    print("\n  token shapes that swallow a second line start, under rule A:")
    for text, n in shapes.most_common(10):
        print(f"    {n:5}  {text}")
    if not shapes:
        print("    (none -- this tokenizer does not merge across newlines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
