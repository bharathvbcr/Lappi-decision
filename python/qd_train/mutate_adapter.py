"""qd-mutate's labelled examples, converted into byte-space decisions for rung 0.

This is the seam between ``crates/qd-mutate`` (Rust, emits JSONL) and
:mod:`qd_train.byte_decider`. Every cross-language seam in this repo has produced a
"one name, two quantities" defect -- five of them so far -- so the conversions this module
performs are named, individually tested, and refuse rather than round.

## The three conversions, and which one is dangerous

**1. Wire field names.** ``LineSpan`` serialises as ``{start_line, end_line}``, not
``{start, end}``. ``crates/qd-mutate/src/span.rs`` says why at length: ``qd-runtime``'s
``SpanValue`` declares those two names with ``deny_unknown_fields``, so a row spelled
``{"start": ...}`` fails to deserialise on all three surfaces. :func:`parse_line_span`
accepts only the wire spelling.

**2. 1-based inclusive lines to 0-based offsets.** ``span`` is 1-based, inclusive on both
ends, over ``after`` -- the *post*-mutation text. Line ``k`` begins at the byte after the
``(k-1)``-th newline, which is ``line_starts(after)[k - 1]``. Mechanical, and tested.

**3. The phantom final line. This is the dangerous one.**

``span.rs``::

    pub fn total_lines(text: &str) -> u32 { line_of(text, text.len()) }

and ``line_of`` is *newline count plus one*. Its docstring is explicit: **"A file ending in
`\\n` has a phantom final line."** So qd-mutate believes ``b"a\\n"`` has **two** lines, while
:func:`qd_train.byte_context.line_starts` reports **one** -- because the second one has no
bytes and begins at ``len(text)``, one past the end.

That line is reachable, not hypothetical: ``span_from_byte_range`` computes
``line_of(text, start)``, and an edit appended at the very end of a file that ends in a
newline lands exactly there. A consumer that silently mapped it to the last real line would
train the span head to point at the wrong line every time an edit was appended at EOF, and
the loss would fall while doing it.

:func:`span_start_offset` raises :class:`PhantomFinalLine`. The example is dropped and
counted, never repaired.

## What is *not* an abstention

``clean`` is a real class, not an abstention -- the model must be able to say "this change
is fine". The **span** of a clean example is ``None``, and *that* is a genuine ``noul``:
``generate.rs`` says clean "points at nothing because nothing was found". So a clean example
supervises the choice slot with a real label and the span slot with an abstention, which is
the shape that teaches the span head when to abstain at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .byte_context import EncodedContext, SpanOutsideWindow, encode_context, line_starts

__all__ = [
    "CLEAN",
    "MUTATION_CLASSES",
    "ByteDecision",
    "LineSpan",
    "MalformedExample",
    "MutateExample",
    "PhantomFinalLine",
    "contrastive_pairs",
    "mutate_total_lines",
    "pair_key",
    "parse_example",
    "parse_line_span",
    "read_examples",
    "span_end_offset",
    "span_start_offset",
    "to_decision",
]

#: ``MutationClass`` in ``crates/qd-mutate/src/ops.rs``, which carries
#: ``#[serde(rename_all = "lowercase")]``. Order is the declaration order, which is what an
#: option list must be stable against; ``test_mutate_adapter.py`` reads ops.rs and fails if
#: either the set or the spelling moves.
MUTATION_CLASSES: Final[tuple[str, ...]] = ("stub", "logic", "cosmetic", "clean")
CLEAN: Final[str] = "clean"

#: The question put to the choice slot. Bytes, because the model reads bytes.
DEFAULT_QUESTION: Final[bytes] = b"What kind of change is this?"


class MalformedExample(ValueError):
    """A qd-mutate row that cannot be trusted. Never repaired, only rejected."""


class PhantomFinalLine(ValueError):
    """The span points at qd-mutate's phantom final line, which holds no bytes.

    See this module's docstring. Raised rather than clamped to the last real line.
    """


@dataclass(frozen=True, slots=True)
class LineSpan:
    """1-based, inclusive on both ends, over the post-mutation text."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise MalformedExample(f"start_line is 1-based, got {self.start_line}")
        if self.end_line < self.start_line:
            raise MalformedExample(
                f"end_line {self.end_line} precedes start_line {self.start_line}; "
                "a span is inclusive on both ends, so this is not an empty span but a broken one"
            )

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def parse_line_span(raw: object, *, where: str) -> LineSpan:
    """Accept only ``{start_line, end_line}``. See conversion 1 in the module docstring."""
    if not isinstance(raw, dict):
        raise MalformedExample(f"{where}: span must be an object, got {type(raw).__name__}")
    missing = {"start_line", "end_line"} - set(raw)
    if missing:
        extra = set(raw) - {"start_line", "end_line"}
        hint = ""
        if extra & {"start", "end"}:
            hint = (
                " -- this row uses {start, end}, which qd-runtime's SpanValue refuses under "
                "deny_unknown_fields; the wire spelling is {start_line, end_line}"
            )
        raise MalformedExample(f"{where}: span is missing {sorted(missing)}{hint}")
    for key in ("start_line", "end_line"):
        if not isinstance(raw[key], int) or isinstance(raw[key], bool):
            raise MalformedExample(f"{where}: span.{key} must be an int, got {raw[key]!r}")
    return LineSpan(start_line=raw["start_line"], end_line=raw["end_line"])


@dataclass(frozen=True, slots=True)
class MutateExample:
    """One row of qd-mutate's JSONL, with only the fields this lane consumes.

    Unconsumed fields (``diff``, ``node_kind``, ``detail``, ``normalization`` ...) are not
    carried. That is deliberate: a field this module stores but never reads would invite a
    downstream consumer to read it *through* here and couple to a shape nothing checks.
    """

    example_id: str
    repo: str
    path: str
    symbol: str
    arity: int
    language: str
    mutation_class: str
    after: bytes
    span: LineSpan | None
    silent: bool
    hunk_constrained: bool
    seed: int

    def __post_init__(self) -> None:
        if self.mutation_class not in MUTATION_CLASSES:
            raise MalformedExample(
                f"{self.example_id}: class {self.mutation_class!r} is not one of "
                f"{MUTATION_CLASSES}"
            )
        if self.mutation_class == CLEAN and self.span is not None:
            raise MalformedExample(
                f"{self.example_id}: a clean example points at nothing, but carries a span"
            )
        if self.mutation_class != CLEAN and self.span is None:
            raise MalformedExample(
                f"{self.example_id}: class {self.mutation_class!r} has no span; an unlocated "
                "mutation cannot supervise the span slot and must not be silently demoted to noul"
            )


def parse_example(obj: dict[str, Any]) -> MutateExample:
    """Parse one qd-mutate row. Missing required fields are an error, never a default."""

    def need(key: str) -> Any:
        if key not in obj:
            raise MalformedExample(f"row is missing required field {key!r}")
        return obj[key]

    ident = need("function")
    if not isinstance(ident, dict):
        raise MalformedExample("function must be an object")
    for key in ("repo", "path", "symbol", "arity"):
        if key not in ident:
            raise MalformedExample(f"function is missing {key!r}")

    after = need("after")
    if not isinstance(after, str):
        raise MalformedExample(f"after must be a string, got {type(after).__name__}")

    example_id = str(need("id"))
    span_raw = obj.get("span")
    span = None if span_raw is None else parse_line_span(span_raw, where=example_id)

    return MutateExample(
        example_id=example_id,
        repo=str(ident["repo"]),
        path=str(ident["path"]),
        symbol=str(ident["symbol"]),
        arity=int(ident["arity"]),
        language=str(need("language")),
        mutation_class=str(need("class")),
        # qd-mutate normalises CRLF to LF before mutating, so `after` is LF-only and UTF-8.
        after=after.encode("utf-8"),
        span=span,
        silent=bool(need("silent")),
        hunk_constrained=bool(need("hunk_constrained")),
        seed=int(need("seed")),
    )


def read_examples(path: Path) -> Iterator[MutateExample]:
    """Stream a JSONL file. A bad line names its line number and stops the read.

    Stops rather than skips: a corpus with silently dropped rows has a coverage number nobody
    can reconstruct, which is the failure `n`/`n_total` exists to prevent.
    """
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise MalformedExample(f"{path}:{lineno}: not JSON: {e}") from e
            try:
                yield parse_example(obj)
            except MalformedExample as e:
                raise MalformedExample(f"{path}:{lineno}: {e}") from e


def mutate_total_lines(raw: bytes) -> int:
    """Line count under **qd-mutate's** convention: newline count plus one.

    Deliberately not the same as ``len(line_starts(raw))``. The difference is the phantom
    final line, and naming both conventions separately is what keeps the gap visible instead
    of letting one silently stand in for the other.
    """
    return raw.count(b"\n") + 1


def _span_offset(after: bytes, span: LineSpan, which: str) -> int:
    """Byte offset in ``after`` where ``span.<which>_line`` begins.

    Both ends go through here so they cannot drift: the phantom-final-line check in
    particular has to run on whichever line is being resolved, and a copy of it that only
    ever guarded ``start_line`` would let an end pointer address the phantom.
    """
    if not after:
        raise MalformedExample("an empty post-mutation text has no lines to point at")

    line = span.start_line if which == "start" else span.end_line
    total = mutate_total_lines(after)
    if span.start_line > total:
        raise MalformedExample(
            f"start_line {span.start_line} is past the end of a {total}-line file"
        )
    if span.end_line > total:
        raise MalformedExample(
            f"end_line {span.end_line} is past the end of a {total}-line file"
        )

    starts = line_starts(after)
    if line > len(starts):
        raise PhantomFinalLine(
            f"{which}_line {line} is qd-mutate's phantom final line: the file ends "
            f"in a newline, so qd-mutate counts {total} lines while only {len(starts)} hold "
            "bytes. The phantom begins one past the end and cannot be pointed at."
        )
    return starts[line - 1]


def span_start_offset(after: bytes, span: LineSpan) -> int:
    """Byte offset in ``after`` where ``span.start_line`` begins.

    Raises :class:`PhantomFinalLine` when the span points at the line qd-mutate counts and
    byte space cannot represent. See conversion 3 in the module docstring.
    """
    return _span_offset(after, span, "start")


def span_end_offset(after: bytes, span: LineSpan) -> int:
    """Byte offset in ``after`` where ``span.end_line`` begins.

    The **end line's start offset**, not the end of the span: the pointer head selects a
    line start for each of its two pointers, so both quantities are line-start offsets.
    """
    return _span_offset(after, span, "end")


@dataclass(frozen=True, slots=True)
class ByteDecision:
    """One example in the shape :mod:`qd_train.byte_decider` consumes."""

    example_id: str
    context: EncodedContext
    question: bytes
    options: tuple[bytes, ...]
    #: Index into ``options``. ``clean`` is a real option here, never an abstention.
    gold_option: int
    #: Index into ``context.starts``, or ``None`` for a genuine span abstention.
    gold_span_line: int | None
    #: Index into ``context.starts`` for the span's **last** line, or ``None`` when
    #: ``gold_span_line`` is. ``SpanPointerHead`` trains two pointers; giving the end one
    #: the start line would be a fabricated target that still produces a falling loss.
    gold_span_end_line: int | None
    hunk_constrained: bool

    def __post_init__(self) -> None:
        if not 0 <= self.gold_option < len(self.options):
            raise MalformedExample(
                f"{self.example_id}: gold_option {self.gold_option} is outside "
                f"{len(self.options)} options"
            )
        if (self.gold_span_line is None) != (self.gold_span_end_line is None):
            raise MalformedExample(
                f"{self.example_id}: gold_span_line and gold_span_end_line are one fact -- a "
                "span either abstains or points -- and exactly one of them is None"
            )
        for name, value in (
            ("gold_span_line", self.gold_span_line),
            ("gold_span_end_line", self.gold_span_end_line),
        ):
            if value is not None and not 0 <= value < len(self.context.starts):
                raise MalformedExample(
                    f"{self.example_id}: {name} {value} is outside the "
                    f"{len(self.context.starts)} lines of the encoded context"
                )
        if (
            self.gold_span_line is not None
            and self.gold_span_end_line is not None
            and self.gold_span_end_line < self.gold_span_line
        ):
            raise MalformedExample(
                f"{self.example_id}: the span ends at line {self.gold_span_end_line} and "
                f"starts at {self.gold_span_line}; an end before its start is not a span"
            )

    @property
    def span_is_noul(self) -> bool:
        return self.gold_span_line is None

    @property
    def span_abstain_row(self) -> int:
        """Row the span head's abstention occupies: one past the last line.

        Mirrors ``answer.rs``'s ``span_rows = line_count + RESERVED_NOUL_ROWS`` with the
        abstention last, which is the same layout ``byte_decider.abstain_column`` produces
        for a choice slot.
        """
        return len(self.context.starts)


def to_decision(
    example: MutateExample,
    *,
    max_context_bytes: int,
    question: bytes = DEFAULT_QUESTION,
    options: tuple[str, ...] = MUTATION_CLASSES,
) -> ByteDecision:
    """Convert one example. Raises rather than returning a degraded decision.

    Truncation keeps the tail of ``after``, so a span in the dropped head is a
    :class:`SpanOutsideWindow` refusal: the span head must not be taught to point at whatever
    truncation happened to leave behind.
    """
    if example.mutation_class not in options:
        raise MalformedExample(
            f"{example.example_id}: class {example.mutation_class!r} is not in the option "
            f"set {options}; a gold answer outside the options is an unanswerable example"
        )

    context = encode_context(example.after, max_bytes=max_context_bytes)
    dropped = len(example.after) - context.n_bytes_kept

    gold_span_line: int | None = None
    gold_span_end_line: int | None = None
    if example.span is not None:
        full_offset = span_start_offset(example.after, example.span)
        if full_offset < dropped:
            raise SpanOutsideWindow(
                f"{example.example_id}: the span starts at byte {full_offset}, but truncation "
                f"dropped the first {dropped} bytes ({context.coverage()} kept)"
            )
        gold_span_line = context.check_span(full_offset - dropped)
        # The end line is inside the window whenever the start is, because truncation keeps
        # the tail -- but it is resolved rather than assumed, so a future head-keeping
        # truncation fails here instead of mislabelling the end pointer.
        end_offset = span_end_offset(example.after, example.span)
        if end_offset < dropped:
            raise SpanOutsideWindow(
                f"{example.example_id}: the span ends at byte {end_offset}, but truncation "
                f"dropped the first {dropped} bytes ({context.coverage()} kept)"
            )
        gold_span_end_line = context.check_span(end_offset - dropped)

    return ByteDecision(
        example_id=example.example_id,
        context=context,
        question=question,
        options=tuple(o.encode("utf-8") for o in options),
        gold_option=options.index(example.mutation_class),
        gold_span_line=gold_span_line,
        gold_span_end_line=gold_span_end_line,
        hunk_constrained=example.hunk_constrained,
    )


def pair_key(example: MutateExample) -> str:
    """The identity two contrastive examples share.

    Matches ``FunctionIdentity``'s own ``Display`` in ``pool.rs``: ``repo:path:symbol/arity``.
    Arity is part of it because two overloads of a name are different functions, and pairing
    across them would produce a "pair" whose two halves share no code.
    """
    return f"{example.repo}:{example.path}:{example.symbol}/{example.arity}"


def contrastive_pairs(
    examples: Iterable[MutateExample],
) -> list[tuple[MutateExample, MutateExample]]:
    """Pair each mutated example with a clean one from the same function.

    Nimble's result (``AUDIT/prior-art-jev-nimble-2026-09-19.md``) is that paired examples
    differing in one fact carry far more signal than independent ones. Here the pair is
    stronger than Nimble's: both halves are the *same function*, one mutated and one not, and
    the label is a property of the tree-sitter edit rather than something a second model had
    to verify.

    A function with no clean counterpart yields no pair. Unpaired examples are still usable
    for training -- this returns pairs, and the caller decides -- but the count of pairs is
    the number to report, because reporting rows would overstate the contrastive signal.
    """
    by_key: dict[str, list[MutateExample]] = {}
    for ex in examples:
        by_key.setdefault(pair_key(ex), []).append(ex)

    pairs: list[tuple[MutateExample, MutateExample]] = []
    for group in by_key.values():
        cleans = [e for e in group if e.mutation_class == CLEAN]
        mutated = [e for e in group if e.mutation_class != CLEAN]
        if not cleans or not mutated:
            continue
        # One clean anchors the group; pairing every clean with every mutation would
        # multiply the same context and inflate any count taken over pairs.
        anchor = min(cleans, key=lambda e: e.example_id)
        pairs.extend((m, anchor) for m in sorted(mutated, key=lambda e: e.example_id))
    return pairs
