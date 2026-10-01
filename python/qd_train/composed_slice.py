"""The report-only composed long-context slice: its cases, hunks, hit rule and tables.

Fable's ruling on the slice (relayed by the compose lane, 2026-10-01) sets its terms:

* **What it is.** The slice is the compose lane's 1,250 ``compose:val:*`` and 300
  ``compose:diag:*`` rows, built at width 8,192 into a shard set of its own. The set's header
  says ``report_only`` and ``span_collapse_policy: refuse-gold``.
* **It is never a gate.** No gate reads it and it is no gate's population (rule 2). Every
  number here is a diagnostic, recorded beside its ``n``.
* **Condition 6: two populations from the one set.**
  * ``refuse_gold`` is every written row.
  * ``refuse_any`` is the rows whose span sequence's candidate positions hold no duplicate,
    i.e. no two rendered lines start on one token.
  * Every cell carries both populations. Their difference is the measured selection bias of
    refuse-any: evidence for the human's open call on val, not a decision.
* **Condition 8: the shared-token hit rule.** A prediction on a token that several lines share
  is a needle-hunk hit only if EVERY line on that token is inside the gold hunk. How many
  predictions landed on a shared token is reported per cell.

What a row looks like (``crates/qd-mutate/src/compose.rs`` ``ComposedRow``, per the compose
lane):

* ``diff`` is a concatenation of ``n_files`` file blocks.
* Each block spans ``[first_line, last_line]``, 1-based over ``diff`` lines. It opens with 3
  header lines (``diff --git``, ``---``, ``+++``) and its body starts on an ``@@`` line.
* The needle block is the one at ``needle_index``. Its ``diff_span`` is
  ``{start_line, end_line}`` over ``diff``.
* A clean row has neither a needle nor a span: its span slot abstains.
* The rendered context is ``file: <path>``, a blank line, then the diff
  (``CONTEXT_HEADER_LINES`` = 2). So rendered line ``c`` (0-based) is diff line
  ``c - CONTEXT_HEADER_LINES + 1``.
* Hunks count the ``@@`` lines over the whole diff. A block's header lines belong to no hunk.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qd_data.defect_class import CONTEXT_HEADER_LINES

from .needle import DEPTH_BUCKETS, depth_bucket_label, wilson_interval
from .tristate import NotRun, Ran, TriState

#: A corpus id's shard row id: ``qd_data.mixture.rewrite_defect_class``'s
#: ``f"qdm:{family_id}:{example_id}"``.
SHARD_ROW_PREFIX: Final[str] = "qdm:code.defect_class:"
VAL_PREFIX: Final[str] = "compose:val:"
DIAG_PREFIX: Final[str] = "compose:diag:"
DIAG_HALVES: Final[tuple[str, ...]] = ("seen_filler", "unseen")
CLASSES: Final[tuple[str, ...]] = ("stub", "logic", "cosmetic", "clean")
CLEAN_CLASS: Final[str] = "clean"
#: ``diff --git``, ``---``, ``+++``: a file block's lines before its first hunk.
BLOCK_HEADER_LINES: Final[int] = 3
#: Compose's ``min_files``: a composed row has at least three files, so depth is defined.
MIN_FILES: Final[int] = 3
#: The slice is 1,550 rows; a file far past that is not the slice.
MAX_SLICE_ROWS: Final[int] = 4096
#: Upper edges (inclusive) of the length bins, in real tokens; past the last is ``gt8k``. The
#: slice is built at --max-seq-len 8192, so ``gt8k`` is empty by construction and appears only
#: if that stops being true.
LENGTH_BINS: Final[tuple[tuple[int, str], ...]] = (
    (1024, "le1k"), (2048, "1k-2k"), (4096, "2k-4k"), (8192, "4k-8k"),
)
LENGTH_OVER: Final[str] = "gt8k"
#: Upper edges (inclusive) of the files-per-row bins; past the last is ``32+``.
FILES_BINS: Final[tuple[tuple[int, str], ...]] = (
    (3, "3"), (7, "4-7"), (15, "8-15"), (31, "16-31"),
)
FILES_OVER: Final[str] = "32+"
#: The depth "bucket" of a clean row, which has no needle.
CLEAN_DEPTH: Final[str] = "clean"
POPULATIONS: Final[tuple[str, ...]] = ("refuse_gold", "refuse_any")
POPULATION_DETAIL: Final[Mapping[str, str]] = {
    "refuse_gold": "every row the refuse-gold slice wrote",
    "refuse_any": (
        "the rows whose span candidates hold no shared token (no two lines start on one "
        "token) -- the subset refuse-any would have written"
    ),
}
#: What every slice metric says of itself.
REPORT_ONLY: Final[str] = "report-only composed slice, not a gate and no gate's population"


class SliceRefusal(ValueError):
    """The slice's rows or their shard sequences do not say what this module reads."""


@dataclass(frozen=True, slots=True)
class ComposedCase:
    """One composed row, as much of it as the slice's tables read."""

    row_id: str
    mutation_class: str
    n_files: int
    needle_index: int | None
    diff_span: tuple[int, int] | None
    diag_half: str | None
    needle_train_appearances: int | None
    #: Diff line ``i + 1``'s hunk, ``None`` on a block's header lines.
    hunk_of_diff_line: tuple[int | None, ...]

    @property
    def is_clean(self) -> bool:
        return self.needle_index is None

    @property
    def slice_set(self) -> str:
        """``val`` for a ``compose:val:*`` row, ``diag.<half>`` for a diag row."""
        return "val" if self.diag_half is None else f"diag.{self.diag_half}"

    @property
    def rendered_lines(self) -> int:
        return CONTEXT_HEADER_LINES + len(self.hunk_of_diff_line)

    @property
    def gold_line(self) -> int | None:
        """The 0-based rendered line the span gold starts on; ``None`` for a clean row."""
        if self.diff_span is None:
            return None
        return self.diff_span[0] + CONTEXT_HEADER_LINES - 1

    @property
    def gold_hunk(self) -> int | None:
        if self.diff_span is None:
            return None
        return self.hunk_of_diff_line[self.diff_span[0] - 1]

    @property
    def depth_bucket(self) -> str:
        if self.needle_index is None:
            return CLEAN_DEPTH
        return depth_bucket_label(self.needle_index / (self.n_files - 1))

    @property
    def files_bin(self) -> str:
        return _bin(self.n_files, FILES_BINS, FILES_OVER)

    def hunk_of_context_line(self, line: int) -> int | None:
        """The hunk holding rendered line ``line`` (0-based), ``None`` for a header line."""
        if line < 0:
            raise SliceRefusal(f"{self.row_id}: line {line} is negative")
        if line < CONTEXT_HEADER_LINES:
            return None
        j = line - CONTEXT_HEADER_LINES
        if j >= len(self.hunk_of_diff_line):
            raise SliceRefusal(
                f"{self.row_id}: line {line} is past the context's {self.rendered_lines} lines"
            )
        return self.hunk_of_diff_line[j]


def _bin(value: int, edges: Sequence[tuple[int, str]], over: str) -> str:
    for edge, label in edges:
        if value <= edge:
            return label
    return over


def length_bin(tokens: int) -> str:
    if tokens < 1:
        raise SliceRefusal(f"a sequence of {tokens} tokens")
    return _bin(tokens, LENGTH_BINS, LENGTH_OVER)


def _int(obj: Mapping[str, object], key: str, where: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SliceRefusal(f"{where}: {key} is {value!r}, not an integer")
    return value


def parse_case(obj: Mapping[str, object], *, where: str) -> ComposedCase:
    """One corpus line, checked against everything the tables rely on."""
    corpus_id = obj.get("id")
    if not isinstance(corpus_id, str) or not corpus_id.startswith((VAL_PREFIX, DIAG_PREFIX)):
        raise SliceRefusal(f"{where}: id {corpus_id!r} is not a compose:val/diag row")
    where = f"{where} ({corpus_id})"
    if obj.get("composed") is not True:
        raise SliceRefusal(f"{where}: composed is {obj.get('composed')!r}, not true")
    mutation_class = obj.get("class")
    if mutation_class not in CLASSES:
        raise SliceRefusal(f"{where}: class {mutation_class!r} is not one of {CLASSES}")
    diff = obj.get("diff")
    if not isinstance(diff, str) or not diff:
        raise SliceRefusal(f"{where}: no diff")
    # The rendered context drops the diff's final newline (rewrite_defect_class), so these
    # are exactly the lines after the two header lines.
    lines = (diff[:-1] if diff.endswith("\n") else diff).split("\n")
    n_files = _int(obj, "n_files", where)
    constituents = obj.get("constituents")
    if not isinstance(constituents, list) or len(constituents) != n_files:
        raise SliceRefusal(f"{where}: {n_files} files but constituents is not a list of them")
    if n_files < MIN_FILES:
        raise SliceRefusal(f"{where}: {n_files} files, under compose's minimum {MIN_FILES}")

    header_lines: set[int] = set()
    blocks: list[tuple[int, int]] = []
    roles: list[str] = []
    expected_first = 1
    for k, block in enumerate(constituents):
        at = f"{where} constituent {k}"
        if not isinstance(block, dict):
            raise SliceRefusal(f"{at}: not an object")
        first, last = _int(block, "first_line", at), _int(block, "last_line", at)
        if first != expected_first or last < first + BLOCK_HEADER_LINES:
            raise SliceRefusal(
                f"{at}: lines {first}..{last}; the blocks must tile the diff from line 1, each "
                f"its {BLOCK_HEADER_LINES} header lines and a body"
            )
        heads = lines[first - 1: first + BLOCK_HEADER_LINES]
        if len(heads) < BLOCK_HEADER_LINES + 1 or not (
            heads[0].startswith("diff --git ") and heads[1].startswith("--- ")
            and heads[2].startswith("+++ ") and heads[3].startswith("@@ ")
        ):
            raise SliceRefusal(
                f"{at}: lines {first}..{first + BLOCK_HEADER_LINES} are not a file header "
                "(diff --git, ---, +++) followed by a hunk header"
            )
        header_lines.update(range(first, first + BLOCK_HEADER_LINES))
        role = block.get("role")
        if role not in ("needle", "filler"):
            raise SliceRefusal(f"{at}: role {role!r}")
        roles.append(str(role))
        blocks.append((first, last))
        expected_first = last + 1
    if expected_first != len(lines) + 1:
        raise SliceRefusal(
            f"{where}: the blocks end at line {expected_first - 1}, the diff at {len(lines)}"
        )

    hunk = -1
    hunks: list[int | None] = []
    for i, text in enumerate(lines, 1):
        if i in header_lines:
            hunks.append(None)
            continue
        if text.startswith("@@ "):
            hunk += 1
        hunks.append(hunk)

    needle_index, raw_span = obj.get("needle_index"), obj.get("diff_span")
    if mutation_class == CLEAN_CLASS:
        if needle_index is not None or raw_span is not None:
            raise SliceRefusal(f"{where}: a clean row with a needle or a span")
        span = None
    else:
        needle_index = _int(obj, "needle_index", where)
        if not 0 <= needle_index < n_files or roles[needle_index] != "needle" or (
            roles.count("needle") != 1
        ):
            raise SliceRefusal(
                f"{where}: needle_index {needle_index} is not the one needle among {roles}"
            )
        if not isinstance(raw_span, dict) or set(raw_span) != {"start_line", "end_line"}:
            raise SliceRefusal(f"{where}: diff_span {raw_span!r} is not {{start_line, end_line}}")
        start, end = _int(raw_span, "start_line", where), _int(raw_span, "end_line", where)
        first, last = blocks[needle_index]
        if not first + BLOCK_HEADER_LINES <= start <= end <= last:
            raise SliceRefusal(
                f"{where}: diff_span {start}..{end} is not inside the needle block's body "
                f"{first + BLOCK_HEADER_LINES}..{last}"
            )
        span = (start, end)

    is_diag = corpus_id.startswith(DIAG_PREFIX)
    half, appearances = obj.get("diag_half"), obj.get("needle_train_appearances")
    if is_diag:
        if half not in DIAG_HALVES:
            raise SliceRefusal(f"{where}: diag_half {half!r} is not one of {DIAG_HALVES}")
        appearances = _int(obj, "needle_train_appearances", where)
        if appearances < 0 or (half == "unseen") != (appearances == 0):
            raise SliceRefusal(
                f"{where}: needle_train_appearances {appearances} for the {half} half"
            )
    elif half is not None or appearances is not None:
        raise SliceRefusal(f"{where}: a compose:val row carries diag fields")

    return ComposedCase(
        row_id=SHARD_ROW_PREFIX + corpus_id,
        mutation_class=str(mutation_class),
        n_files=n_files,
        needle_index=needle_index if isinstance(needle_index, int) else None,
        diff_span=span,
        diag_half=str(half) if is_diag else None,
        needle_train_appearances=appearances if is_diag else None,
        hunk_of_diff_line=tuple(hunks),
    )


def load_cases(paths: Iterable[Path], *, max_rows: int = MAX_SLICE_ROWS) -> dict[str, ComposedCase]:
    """Every row of the slice's corpus files, keyed by shard row id; a repeat is refused."""
    cases: dict[str, ComposedCase] = {}
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for n, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                if len(cases) >= max_rows:
                    raise SliceRefusal(
                        f"{path}: more than {max_rows} rows; the slice is 1,550 and this is not it"
                    )
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise SliceRefusal(f"{path}:{n}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise SliceRefusal(f"{path}:{n}: not an object")
                case = parse_case(obj, where=f"{path}:{n}")
                if case.row_id in cases:
                    raise SliceRefusal(f"{path}:{n}: {case.row_id} appears twice")
                cases[case.row_id] = case
    return cases


def lines_on_token(candidates: Sequence[int], head_row: int) -> tuple[int, ...]:
    """The rendered lines on the token the pointer head chose, ``()`` for its abstention.

    ``candidates`` is the sequence's line-start positions, one per rendered line (two lines
    on one token repeat it). The head's rows are the UNIQUE positions in ascending order, then
    the abstention (``qd_train.heads.plan_span_batch``), so row ``j`` is a token and every line
    starting on it.
    """
    unique = sorted(set(int(p) for p in candidates))
    if head_row == len(unique):
        return ()
    if not 0 <= head_row < len(unique):
        raise SliceRefusal(
            f"head row {head_row} is not one of {len(unique)} tokens or the abstention"
        )
    token = unique[head_row]
    return tuple(c for c, p in enumerate(candidates) if int(p) == token)


def check_alignment(case: ComposedCase, candidates: Sequence[int], gold_head_row: int) -> None:
    """Refuse a sequence whose candidates do not line up with its corpus row.

    One candidate per rendered line, and the head's gold row is the gold start line's token --
    a token no other line shares (refuse-gold's guarantee) -- or the abstention for a clean row.
    A mapping off by one line would score every row against the wrong hunk, so it is never
    scored at all.
    """
    if len(candidates) != case.rendered_lines:
        raise SliceRefusal(
            f"{case.row_id}: {len(candidates)} line-start candidates for {case.rendered_lines} "
            "rendered lines, so a candidate index is not a line number"
        )
    expected = lines_on_token(candidates, gold_head_row)
    if case.gold_line is None:
        if expected:
            raise SliceRefusal(f"{case.row_id}: a clean row whose span gold is not the abstention")
        return
    if expected != (case.gold_line,):
        raise SliceRefusal(
            f"{case.row_id}: the shard's gold token starts lines {expected}, the corpus's gold "
            f"start is rendered line {case.gold_line}"
        )


def needle_hunk_hit(case: ComposedCase, lines: Sequence[int]) -> bool:
    """Condition 8: a hit iff the prediction is a token and EVERY line on it is in the gold
    hunk. The abstention and a block's header line are misses."""
    if case.gold_hunk is None:
        raise SliceRefusal(f"{case.row_id}: a clean row has no needle hunk to hit")
    return bool(lines) and all(case.hunk_of_context_line(c) == case.gold_hunk for c in lines)


@dataclass(frozen=True, slots=True)
class SliceVerdict:
    """One slice row as decoded: what the tables count."""

    case: ComposedCase
    #: The longer of the row's sequences, in real tokens: what the model read.
    length_tokens: int
    #: Whether the span sequence's candidates repeat a token; ``None`` when the row has no
    #: span sequence (refuse-gold wrote no span slot for it).
    shared_candidates: bool | None
    span_correct: bool | None
    #: The rendered lines on the predicted start token (``()`` for the abstention); ``None``
    #: without a span sequence.
    predicted_lines: tuple[int, ...] | None
    choice_correct: bool | None

    def in_population(self, population: str) -> bool:
        if population == "refuse_gold":
            return True
        if population == "refuse_any":
            # A row without a span sequence had its slot refused for a collision on its gold
            # line, which refuse-any would have refused too.
            return self.shared_candidates is False
        raise SliceRefusal(f"population {population!r} is not one of {POPULATIONS}")

    @property
    def hunk_hit(self) -> bool | None:
        if self.case.is_clean or self.predicted_lines is None:
            return None
        return needle_hunk_hit(self.case, self.predicted_lines)

    @property
    def shared_prediction(self) -> bool | None:
        if self.case.is_clean or self.predicted_lines is None:
            return None
        return len(self.predicted_lines) > 1


def _cells(v: SliceVerdict) -> list[tuple[str, str]]:
    lb, depth = length_bin(v.length_tokens), v.case.depth_bucket
    cells = [("all", "all"), ("length", lb), ("depth", depth), ("files", v.case.files_bin)]
    if v.case.diag_half is None:
        cells.append(("length_x_depth", f"{lb}.{depth}"))
    return cells


def _rate(flags: list[bool], *, what: str, population: str) -> TriState:
    if not flags:
        return NotRun(reason=f"no {what} in this cell ({population})")
    k, n = sum(flags), len(flags)
    lo, hi = wilson_interval(k, n)
    return Ran(
        passed=True, value=k / n, n=k, n_total=n,
        detail=(
            f"{k} of {n} {what} ({k / n:.3f}, Wilson 95% [{lo:.3f}, {hi:.3f}]); "
            f"{population}: {POPULATION_DETAIL[population]}; {REPORT_ONLY}"
        ),
    )


def _order(cut: str, cell: str) -> tuple[int, str]:
    order = {
        "all": ["all"],
        "length": [label for _, label in LENGTH_BINS] + [LENGTH_OVER],
        "depth": [*DEPTH_BUCKETS, CLEAN_DEPTH],
        "files": [label for _, label in FILES_BINS] + [FILES_OVER],
    }.get(cut)
    return (order.index(cell) if order and cell in order else len(order or ()), cell)


def slice_metrics(verdicts: Sequence[SliceVerdict]) -> dict[str, TriState]:
    """Every table, as ``composed.<set>.<population>.<cut>.<cell>.<metric>``.

    ``<set>`` is ``val`` (the compose:val rows) or ``diag.seen_filler``/``diag.unseen`` (each
    diag half alone). Cuts: ``all``, ``length``, ``depth`` (the needle's file position over
    ``n_files - 1``; ``clean`` for clean rows) and ``files`` for every set, plus
    ``length_x_depth`` for ``val``. A cell exists when some row of the set falls in it. Its
    metrics, per population:

    * ``span_top1``: the span verdict exactly right (a clean row: the abstention);
    * ``hunk_hit``: condition 8's rule, needle rows only;
    * ``shared_token_predictions``: of those, predictions on a shared token;
    * ``choice_top1``: the defect_class verdict right.

    ``composed.<set>.delta.<cut>.<cell>.<metric>`` is refuse-any's rate minus
    refuse-gold's, wherever both ran. ``composed.<set>.<population>.rows`` and
    ``composed.<set>.span_slot_absent`` count the rows.
    """
    by_set: dict[str, list[SliceVerdict]] = {}
    for v in verdicts:
        by_set.setdefault(v.case.slice_set, []).append(v)
    out: dict[str, TriState] = {}
    for slice_set in sorted(by_set):
        rows = by_set[slice_set]
        absent = sum(1 for v in rows if v.shared_candidates is None)
        out[f"composed.{slice_set}.span_slot_absent"] = Ran(
            passed=True, value=absent, n=absent, n_total=len(rows),
            detail=(f"rows with no span sequence: refuse-gold refused the slot for a collision "
                    f"on its gold line; their choice slot is still scored; {REPORT_ONLY}"),
        )
        cells = sorted({c for v in rows for c in _cells(v)}, key=lambda c: (c[0], _order(*c)))
        for population in POPULATIONS:
            members = [v for v in rows if v.in_population(population)]
            out[f"composed.{slice_set}.{population}.rows"] = Ran(
                passed=True, value=len(members), n=len(members), n_total=len(rows),
                detail=f"{population}: {POPULATION_DETAIL[population]}; {REPORT_ONLY}",
            )
            for cut, cell in cells:
                inside = [v for v in members if (cut, cell) in _cells(v)]
                name = f"composed.{slice_set}.{population}.{cut}.{cell}"
                out[f"{name}.span_top1"] = _rate(
                    [bool(v.span_correct) for v in inside if v.span_correct is not None],
                    what="span verdicts right", population=population,
                )
                hits = [v for v in inside if v.hunk_hit is not None]
                out[f"{name}.hunk_hit"] = _rate(
                    [bool(v.hunk_hit) for v in hits],
                    what="needle predictions in the needle hunk (all lines on its token)",
                    population=population,
                )
                shared = sum(1 for v in hits if v.shared_prediction)
                out[f"{name}.shared_token_predictions"] = (
                    Ran(passed=True, value=shared, n=shared, n_total=len(hits),
                        detail=(f"{shared} of {len(hits)} needle predictions on a token that "
                                f"several lines share; {population}; {REPORT_ONLY}"))
                    if hits else NotRun(reason=f"no needle prediction in this cell ({population})")
                )
                out[f"{name}.choice_top1"] = _rate(
                    [bool(v.choice_correct) for v in inside if v.choice_correct is not None],
                    what="defect_class verdicts right", population=population,
                )
        for cut, cell in cells:
            for metric in ("span_top1", "hunk_hit", "choice_top1"):
                gold = out[f"composed.{slice_set}.refuse_gold.{cut}.{cell}.{metric}"]
                anyp = out[f"composed.{slice_set}.refuse_any.{cut}.{cell}.{metric}"]
                name = f"composed.{slice_set}.delta.{cut}.{cell}.{metric}"
                if isinstance(gold, Ran) and isinstance(anyp, Ran):
                    assert isinstance(gold.value, float) and isinstance(anyp.value, float)
                    out[name] = Ran(
                        passed=True, value=anyp.value - gold.value,
                        detail=(
                            f"refuse-any {anyp.value:.3f} (n {anyp.n_total}) minus refuse-gold "
                            f"{gold.value:.3f} (n {gold.n_total}): the measured selection bias "
                            "of refuse-any -- evidence for the human's open call on val, not a "
                            f"decision; {REPORT_ONLY}"
                        ),
                    )
                else:
                    out[name] = NotRun(
                        reason="a population has no row for this metric in this cell"
                    )
    return out
