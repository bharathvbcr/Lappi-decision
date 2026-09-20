"""S4: the data path. Pre-tokenized `uint32` memmap shards, length buckets, no packing.

The plan: *"Pre-tokenized memmap shards (the nanolab pattern), length buckets per hunk
size, **no packing** (unsupported for the hybrid)."* Gate: *"Padding waste <= 15% measured
over one epoch of the mixture."*

Everything about the *format* lives in :mod:`qd_train.artifacts` and is imported from
there, never restated: ``ShardHeader``, ``Batch``, ``RemapTable``, ``bucket_for``,
``assign_buckets``, ``padding_waste``, ``assert_shard_trainable``, ``TOKEN_DTYPE``. This
module is the writer and the reader; `artifacts.py` is what they both mean by a shard.
``docs/training-contract.md`` explains why that separation is load-bearing.

## What is on disk

A shard set is a directory:

``header.json``
    ``ShardHeader.to_json()``. Re-verified on read via ``ShardHeader.from_json``, which
    recomputes ``shard_hash`` and refuses a file edited after it was written.
``tokens.u32``
    Every sequence's post-remap ids, back to back, as raw little-endian ``uint32``.
``offsets.npy``
    ``int64[n_sequences + 1]``. Sequence *i* is ``tokens[offsets[i]:offsets[i + 1]]``.
``supervision.npz``
    ``slot_kind uint8[n]``, ``target_index int32[n]``, ``span_target int32[n, 2]`` -- the
    ``Batch`` channels, one entry per sequence -- plus ``candidate_positions int32[]`` and
    ``candidate_offsets int64[n + 1]``, the pointer head's line-start candidates in the
    same ragged form as the tokens. Stored rather than recomputed so the shard *states*
    what it supervises: the first version relied on the convention that the gold is the
    final token, and that convention had no room for a span at all, which is the defect
    ``GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`` was opened for. ``Batch.line_starts`` wants
    a ``[B, L]`` bool mask; the reader materialises it per batch, at that batch's width,
    because a mask that wide is mostly false and grows with the longest sequence in the
    whole set rather than with the number of lines it actually has.
``coverage.json``
    What was written and what was left out, as a tri-state. Separate from ``header.json``
    because that file's shape belongs to the contract; see :func:`write_shards`.

**Storing sequences back-to-back on disk is not packing.** Packing means two examples in
one *training row*, which is what SAFETY-2 forbids: a GDN layer carries recurrent state
along the sequence, so the second example reads the first example's state and no attention
mask expresses a reset. Here ``offsets`` keeps every sequence addressable on its own and
:meth:`ShardReader.batches` puts exactly one sequence in each row of each batch, padding
the rest. ``test_shards.py::test_every_batch_row_holds_exactly_one_sequence`` is the test
that this stays true; ``ShardHeader`` refusing ``packed=True`` is the structural half.

## Why the writer takes rows *and* a manifest path

``open_training_data`` is rule 3's door, and it is the only way this module learns a
corpus's ``data_snapshot_hash``. But a manifest carries provenance, not text -- a
``ManifestEntry`` has a ``content_hash`` and no prompt -- so the text has to arrive
separately, as the ``DataRow`` objects the mixture already built.

That would be a hole big enough to drive the held-out set through: pass held-out rows
alongside a training manifest and rule 3 is satisfied on paper while the trainer reads
exactly the data it must not. So :func:`write_shards` matches the two against each other
by ``row_content_hash`` and refuses any disagreement in either direction. The manifest
decides which rows may be written; the rows only supply their text.

## The torch boundary

Nothing here imports torch or transformers -- they are an optional ``mac`` extra and are
not in the repo venv. The tokenizer arrives as an injected ``Callable[[str], list[int]]``,
so format, ordering, bucketing and the padding-waste gate are all testable on any machine.
A span row additionally needs the tokenizer's character offsets (:data:`TokenOffsets`),
for the same reason and on the same terms: injected, not imported.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from qd_data.config import DataConfig
from qd_data.render import DEFAULT_CAPS, M_CTX_END, RenderCaps, render
from qd_data.rows import DataRow, row_content_hash
from qd_data.schema import NOUL_LETTER, ChoiceSlot, ScoreSlot, Slot, SpanSlot

from .artifacts import (
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    TOKEN_DTYPE,
    Batch,
    RemapTable,
    ShardContractViolation,
    ShardHeader,
    assert_shard_trainable,
    assign_buckets,
    bucket_for,
    line_start_indices,
    padding_waste,
)
from .data_access import open_training_data
from .tristate import NotRun, Ran, TriState, parse_tristate

__all__ = [
    "COVERAGE_NAME",
    "DEFAULT_MAX_SEQUENCES",
    "DEFAULT_MAX_TOTAL_TOKENS",
    "HEADER_NAME",
    "MAX_ROWS_PER_BATCH",
    "OFFSETS_NAME",
    "PAD_ID",
    "SUPERVISION_NAME",
    "TOKENS_NAME",
    "Decode",
    "SequenceSpec",
    "ShardReader",
    "TokenOffsets",
    "UnencodableGold",
    "answer_letter",
    "choose_buckets",
    "line_starts",
    "slot_kind_of",
    "training_texts",
    "write_shards",
]

HEADER_NAME: Final[str] = "header.json"
TOKENS_NAME: Final[str] = "tokens.u32"
OFFSETS_NAME: Final[str] = "offsets.npy"
COVERAGE_NAME: Final[str] = "coverage.json"
SUPERVISION_NAME: Final[str] = "supervision.npz"

#: ``(char_start, char_end)`` per token, aligned one-to-one with ``tokenize``'s output --
#: the shape a HuggingFace fast tokenizer returns for ``return_offsets_mapping=True``.
#: Required only when a span row is present; see :func:`write_shards`.
TokenOffsets = Callable[[str], list[tuple[int, int]]]

#: One token id back to the text it stands for -- ``tokenizer.decode([id])``. Optional, and
#: the only thing that can check a span against what the tokenizer *says* rather than
#: against arithmetic over what it claims. See :func:`_assert_spans_decode_to_their_text`.
Decode = Callable[[Sequence[int]], str]

#: Padding filler. Arbitrary in principle -- ``Batch.lengths`` is what the trainer masks
#: loss with, so no padded position contributes -- but it must still be a *valid* embedding
#: index, because an out-of-range id would fault the lookup before the mask is ever applied.
#: New id 0 always exists: ``RemapTable`` refuses a remap that keeps no tokens.
PAD_ID: Final[int] = 0

#: A batch is bounded in rows as well as in tokens. Without this, a narrow bucket turns a
#: generous ``batch_tokens`` into a batch of tens of thousands of rows, and the first sign
#: of it is an allocator failure some minutes into a run.
MAX_ROWS_PER_BATCH: Final[int] = 4096

#: Write-side ceilings. Both are arguments with these as defaults, so a caller may tighten
#: them; neither may be absent, because an unbounded write is an unbounded memmap.
DEFAULT_MAX_SEQUENCES: Final[int] = 50_000_000
DEFAULT_MAX_TOTAL_TOKENS: Final[int] = 100_000_000_000


class UnencodableGold(Exception):
    """This row's supervision cannot be expressed in a ``Batch``.

    Two kinds of span gold used to land here, and both are now encodable:

    * A span with a real line range. ``Batch`` had no channel for a position, so the only
      gold available was the noul letter and the span head could only have learned to
      abstain. ``GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL``, resolved by ``slot_kind`` /
      ``target_index`` / ``span_target``.
    * A span whose gold is *abstain*. ``SLOT_SPAN`` rows had to carry real positions and
      ``NO_SPAN`` meant "not a span row", so an unanswerable question had no encoding at
      all -- while ``qd-runtime/src/answer.rs:64`` sized the pointer head at
      ``line_count + RESERVED_NOUL_ROWS`` and could say "no evidence" perfectly well.
      ``GAP-S4-ABSTAINING-SPAN-HAS-NO-ENCODING``, resolved by :data:`SPAN_ABSTAIN`.

    What lands here now is a row this writer will not guess at: a gold that names no
    rendered option, a slot type with no supervision kind, a span whose gold lines fall
    outside its context, a context whose lines collapse onto too few tokens to tell apart,
    and a span row with no offset mapping to place it. Each is refused rather than
    approximated, because every one of them ends as a training example that teaches
    something specific and wrong while the loss curve looks ordinary.
    """


# -- lines, and where they land in a tokenization ---------------------------------------


def line_starts(text: str) -> list[int]:
    """Character offsets of every line start: 0-based offsets, for 1-based line numbers.

    **The rule is not stated here.** It is :func:`qd_train.artifacts.line_start_indices`,
    beside ``bucket_for``, and this is the writer's spelling of it -- a ``list`` of
    character offsets, which is what ``_context_line_chars`` and the tests already read.

    It used to be a second implementation, which is what
    ``GAP-S4-LINE-STARTS-SECOND-IMPLEMENTATION`` was opened for, and the gap was not
    hypothetical: this function reported **one** line for the empty string while
    ``qd_train.byte_context.line_starts`` -- the *other* function of this name in this
    package, rung 0's, over bytes -- reported **none**, and both were green, because the
    five vectors each was pinned to come from the Rust suite and not one of them is empty.
    ``Context::line_count`` (context.rs:167) says none, so the byte path was right and
    this was wrong. The consequence was not an exception: an empty context yielded one
    candidate the runtime would never offer, against a pointer head sized at
    ``line_count + RESERVED_NOUL_ROWS`` = the abstain row alone.

    ``artifacts`` is now the single owner, and the two are held together vector for vector
    by ``test_shards.py::test_the_two_python_line_rules_agree_vector_for_vector``. The
    Rust remains the definition for the whole system and is pinned separately, against
    ``answering_procedure.rs``'s own table, in
    ``test_shards.py::test_line_starts_matches_the_rust_definition``.

    The Rust counts bytes and this counts characters. For line *numbering* the two agree:
    ``\\n`` is one byte and one character, so the same newlines are found at the same
    ordinals. Characters are what is wanted here because a tokenizer's offset mapping is
    in characters.
    """
    return list(line_start_indices(text))


def _token_index_for_char(offsets: Sequence[tuple[int, int]], char_pos: int, *, where: str) -> int:
    """The first token whose character span contains ``char_pos``.

    A BPE token can straddle a line boundary -- the token holding the first character of
    line N may also hold the tail of line N-1 -- so "the token at this line start" is the
    token *containing* that character, which is what the pointer head would have to point
    at. When no token contains it the mapping has failed, and this raises rather than
    picking a neighbour: a span pointing at the wrong token teaches the model to cite the
    wrong evidence, and nothing downstream could tell.
    """
    for i, (start, end) in enumerate(offsets):
        if start <= char_pos < end:
            return i
    raise UnencodableGold(
        f"{where}: character {char_pos} lies in no token's offset span. The tokenizer's "
        "offsets and its ids describe different strings, or the offsets omit this region. "
        "Refused rather than mapped to a neighbouring token."
    )


def slot_kind_of(slot: Slot) -> int:
    """The ``artifacts`` supervision kind for a rendered slot."""
    if isinstance(slot, ChoiceSlot):
        return SLOT_CHOICE
    if isinstance(slot, ScoreSlot):
        return SLOT_SCORE
    if isinstance(slot, SpanSlot):
        return SLOT_SPAN
    raise UnencodableGold(
        f"slot {slot.name!r} is a {type(slot).__name__}, which has no supervision kind in "
        "artifacts. A row whose kind the trainer cannot read is a row it would supervise "
        "by guesswork."
    )


# -- picking the text a sequence is made of --------------------------------------------


def answer_letter(row: DataRow, slot_name: str, letter_to_value: dict[str, str]) -> str:
    """The letter this row's gold answer denotes, in *this* rendering's alphabet.

    The alphabet is per-example: ``qd_data.render`` shuffles a choice slot's options per
    training example, so the letter for a given option differs between examples by design.
    Reading it back out of the ``SlotRender`` that was actually produced is what keeps the
    label attached to the rendering -- deriving it from the request's option order instead
    would bake in one permutation and mislabel every shuffled epoch, which is the failure
    ``GoldAnswer``'s docstring already warns about one level up.
    """
    gold = next((g for g in row.gold if g.slot_name == slot_name), None)
    if gold is None:
        raise UnencodableGold(
            f"row {row.row_id!r}: slot {slot_name!r} has no gold answer. A rendered prompt "
            "with no supervision is an unlabelled sequence, and training on it teaches the "
            "model to continue the prompt rather than to answer it."
        )
    if gold.is_noul:
        return NOUL_LETTER

    wanted = str(gold.value)
    matches = [letter for letter, value in letter_to_value.items() if value == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise UnencodableGold(
            f"row {row.row_id!r}: slot {slot_name!r} has gold {gold.value!r}, which is not "
            f"among the rendered options {sorted(letter_to_value.values())}. The label and "
            "the prompt describe different questions."
        )
    raise UnencodableGold(
        f"row {row.row_id!r}: slot {slot_name!r} renders {wanted!r} under "
        f"{len(matches)} letters ({sorted(matches)}), so the gold letter is ambiguous."
    )


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    """One training sequence before tokenization: its text and its supervision.

    ``span_char_starts`` is the ``(start, end)`` pair of **character offsets into
    ``text``** at which the gold's first and last evidence lines begin, or ``None`` for a
    non-span row. Characters rather than tokens because the tokenizer has not run yet; the
    conversion is :func:`_token_index_for_char`, and keeping the two steps apart is what
    lets the line arithmetic be tested without a tokenizer at all.
    """

    slot_name: str
    text: str
    slot_kind: int
    span_char_starts: tuple[int, int] | None
    #: True only for a span row whose gold is abstain. Such a row still carries
    #: ``line_char_starts`` -- the pointer head needs its candidate set either way -- but
    #: has no ``span_char_starts``, because there is no evidence to point at.
    span_abstains: bool = False
    #: Every context line start, as offsets into ``text``: the pointer head's candidate set.
    #: Set for every span row, ``None`` otherwise. The gold in ``span_char_starts`` is drawn
    #: from this same list, so the two cannot disagree about where a line begins.
    line_char_starts: tuple[int, ...] | None = None


def _context_line_chars(row: DataRow, prefix: str, region: str) -> list[int]:
    """Every context line start, as character offsets into the rendered prompt.

    One computation behind both the candidate set and the gold: the gold is
    ``result[line - 1]``, so a line the gold names and the same line in the candidate set
    are the same number by construction rather than by two agreeing derivations. ``Batch``
    now requires a span gold to land on a candidate, and that check is only worth anything
    while the two come from here.

    The gold's line numbers are 1-based and inclusive, counted over the **raw** context
    (``qd_data.mixture._line_span``). The prompt carries the *escaped* context, so this
    leans on ``escape_block``'s stated invariant that it preserves the newline count
    exactly -- line N of the escaped region is line N of the raw one. That invariant is
    load-bearing here rather than incidental, so it is checked rather than trusted: if the
    two line counts ever disagree, every span label moves by the difference, silently,
    which is precisely the systematic off-by-one ``docs/hardening.md`` section 1 describes.
    """
    raw_lines = len(line_starts(row.request.context.decode("utf-8")))
    region_starts = line_starts(region)
    if len(region_starts) != raw_lines:
        raise UnencodableGold(
            f"row {row.row_id!r}: the escaped context has {len(region_starts)} lines but the "
            f"raw context has {raw_lines}. escape_block is documented to preserve the "
            "newline count exactly; it did not here, so every span label would be shifted."
        )
    # Locate the region by the delimiter `context_region` itself keys off, not by counting
    # back from the end of the prefix: that arithmetic has to know every character render
    # puts after the context, and getting it wrong shifts every span label by a constant --
    # `docs/hardening.md` section 1's systematic off-by-one, which is invisible in loss.
    # The slice is then checked against the region, so a drift in render's layout raises
    # here instead of silently relabelling.
    end = prefix.rindex("\n" + M_CTX_END)
    base = end - len(region)
    if prefix[base:end] != region:
        raise UnencodableGold(
            f"row {row.row_id!r}: the context region was not found where the prompt's "
            "delimiters say it is, so a span offset into it would point at other text"
        )
    return [base + s for s in region_starts]


def training_texts(
    row: DataRow, *, seed: int, caps: RenderCaps = DEFAULT_CAPS
) -> list[SequenceSpec]:
    """One :class:`SequenceSpec` per slot on this row, in the request's slot order.

    The text is the canonical rendering -- ``qd_data.render.render``, the single renderer
    the serving path also uses -- followed by one answer token. There is no second prompt
    format here; a divergence between the training prompt and the served prompt is what
    ``test_render.py::test_training_and_serving_render_byte_identically`` exists to catch,
    and it can only keep catching it while this path goes through ``render``.

    Every row gets that trailing token, span rows included, and it is the same shape for
    all of them: ``target_index`` is then ``lengths - 2``, which is the last token of
    ``<|qd_answer|>`` -- exactly the position the model is asked to answer from at serving
    time. For a choice or score row the trailing token is the gold letter. For a span row
    the gold is in ``span_target`` instead, and the trailing token is the only letter a
    span slot's option block offers: ``noul``. **A trainer must route on ``slot_kind``.**
    Applying the letter loss to a ``SLOT_SPAN`` row would train it to abstain, which is the
    bug ``GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`` was opened for; ``slot_kind`` exists so
    that routing is explicit rather than conventional.
    """
    rendered = render(row.request, caps=caps, seed=seed)
    region = rendered.context_region()
    out: list[SequenceSpec] = []
    for rendered_slot, slot in zip(rendered.slots, row.request.slots, strict=True):
        if rendered_slot.name != slot.name:
            raise UnencodableGold(
                f"row {row.row_id!r}: rendered slot {rendered_slot.name!r} does not match "
                f"request slot {slot.name!r}; the supervision would name the wrong slot"
            )
        kind = slot_kind_of(slot)
        gold = next(g for g in row.gold if g.slot_name == slot.name)

        span_chars: tuple[int, int] | None = None
        line_chars: tuple[int, ...] | None = None
        abstains = False
        if kind == SLOT_SPAN:
            # Never `answer_letter` here. A span slot's option block offers exactly one
            # letter -- noul -- so asking for "the letter this gold denotes" would either
            # fail or hand back the abstain letter as though it were the answer. The gold
            # goes in span_target; the trailing token is structural, and `slot_kind` is
            # what tells the trainer not to read it as a label.
            letter = NOUL_LETTER
            # Every span row needs its candidate set, abstaining ones included: the pointer
            # head still ranges over this context's line starts, and the abstain answer is
            # the reserved extra row beside them, not the absence of candidates.
            candidates = _context_line_chars(row, rendered.prefix, region)
            line_chars = tuple(candidates)
            if gold.is_noul:
                abstains = True
            else:
                value = gold.value
                if not (isinstance(value, tuple) and len(value) == 2):
                    raise UnencodableGold(
                        f"row {row.row_id!r}: slot {slot.name!r} is a span but its gold is "
                        f"{value!r}, not a (start_line, end_line) pair"
                    )
                start_line, end_line = int(value[0]), int(value[1])
                if not 1 <= start_line <= end_line <= len(candidates):
                    raise UnencodableGold(
                        f"row {row.row_id!r}: slot {slot.name!r} has gold lines "
                        f"{(start_line, end_line)}, outside the context's 1..{len(candidates)} "
                        "lines. A span pointing past the context would be clamped into "
                        "pointing at the wrong evidence."
                    )
                span_chars = (candidates[start_line - 1], candidates[end_line - 1])
        else:
            letter = answer_letter(row, slot.name, rendered_slot.letter_to_value)

        out.append(
            SequenceSpec(
                slot_name=slot.name,
                text=rendered.prefix + rendered_slot.suffix + letter,
                slot_kind=kind,
                span_char_starts=span_chars,
                span_abstains=abstains,
                line_char_starts=line_chars,
            )
        )
    if not out:
        raise UnencodableGold(f"row {row.row_id!r}: the request renders no slots")
    return out


# -- bucketing -------------------------------------------------------------------------


def choose_buckets(lengths: Sequence[int], *, n_buckets: int = 8) -> tuple[int, ...]:
    """Bucket boundaries drawn from an observed length distribution.

    Equal-count quantiles rather than equal-width bins: padding waste is driven by the
    *mass* of sequences sitting far below their bucket's ceiling, so the boundaries belong
    where the sequences are. The largest boundary is always the longest sequence, because
    ``bucket_for`` refuses a length that fits no bucket and ``ShardHeader`` refuses a
    largest bucket below ``max_seq_len`` -- a sequence with nowhere to go must be a loud
    write-time refusal, never a quiet truncation at read time.

    This picks *boundaries*; it does not touch the gate. ``MAX_PADDING_WASTE`` is read-only
    (rule 2), and lowering waste by bucketing better is the work the gate exists to demand.
    """
    if not lengths:
        raise ValueError(
            "cannot choose buckets from zero lengths: the distribution that would decide "
            "the boundaries does not exist, and a default bucketing would be measured "
            "against a corpus it was never fitted to"
        )
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    values = np.asarray(sorted(int(n) for n in lengths), dtype=np.int64)
    if int(values[0]) <= 0:
        raise ValueError(f"every length must be positive, got {int(values[0])}")

    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_buckets + 1)[1:])
    boundaries = sorted({int(np.ceil(q)) for q in quantiles} | {int(values[-1])})
    return tuple(boundaries)


# -- the writer ------------------------------------------------------------------------


def _tokenize_checked(
    tokenize: Callable[[str], list[int]], text: str, *, where: str
) -> np.ndarray:
    """Run the injected tokenizer and refuse anything that is not a list of token ids.

    The tokenizer is the one part of this path the module does not own, and a bad return
    value from it fails much later and much less legibly: a float array indexes the remap
    by truncation, and an empty one produces a zero-length sequence that ``bucket_for``
    rejects with a message about buckets rather than about the tokenizer.
    """
    raw = tokenize(text)
    ids = np.asarray(raw)
    if ids.ndim != 1:
        raise ShardContractViolation(
            f"{where}: the tokenizer returned a {ids.ndim}-D result; expected a flat "
            "sequence of token ids"
        )
    if ids.size < 2:
        # Two, not one. `trainer.ft_supervision` supervises position `lengths - 2` -- the
        # one whose next token is the answer -- so a one-token row is a prompt with no
        # answer. The trainer calls that "a writer-side error", so the writer refuses it
        # here, where the row it came from can still be named.
        raise ShardContractViolation(
            f"{where}: the tokenizer returned {ids.size} token(s). A training example is "
            "a prompt followed by its answer token, so it needs at least two; an empty "
            "sequence additionally fits no bucket."
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise ShardContractViolation(
            f"{where}: the tokenizer returned dtype {ids.dtype}; token ids must be "
            "integers. A float id would index the remap by silent truncation."
        )
    return ids


def _assert_spans_decode_to_their_text(
    spec: SequenceSpec,
    ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    positions: Sequence[int],
    *,
    decode: Decode,
    where: str,
) -> None:
    """Check the line->token mapping against **decoded text**, not against itself.

    This is the check ``GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED`` names as the one that
    would settle it, and ``GAP-S4-GOLD-ON-CANDIDATE-CANNOT-CATCH-LINE-SHIFT`` records why
    nothing else in this path can. Every other span invariant here compares the gold to
    the candidate set, and the two are projected from one list of offsets: a mapping that
    is consistently wrong moves both together, so the gold still lands on a candidate and
    every suite stays green while the head learns to cite the line next door.

    Asking the tokenizer to *decode* breaks that circle, because the answer comes from the
    tokenizer's vocabulary rather than from the offsets under test. Three statements:

    1. **Each checked token decodes to exactly the characters its offsets claim.** A
       tokenizer whose ``return_offsets_mapping`` describes a normalised copy of the text
       satisfies every length and reach check in this module and fails here.
    2. **The ids round-trip to the text they were produced from.** Not implied by (1):
       HuggingFace's ``decode`` applies ``clean_up_tokenization_spaces`` over a *sequence*,
       so a tokenizer can be honest token by token and still not reproduce the text.
    3. **Every character offset recorded as a line start is a line start of the decoded
       text**, under :func:`~qd_train.artifacts.line_start_indices` -- the same rule the
       runtime serves. This is the statement the gap asks for: the positions the pointer
       head will range over are line starts of the text the tokenizer actually produces,
       established against decoded text rather than against the candidate set.

    **What this does not establish.** It does not verify the gold's *line number* -- that
    leg (gold line -> character offset in the region) is character-space arithmetic that
    needs no tokenizer, and it is checked against an independent baseline by
    ``test_shards.py::test_a_span_row_carries_gold_token_positions``, which splits the
    rendered region itself rather than reusing the offsets. Nor does it make a real BPE
    tokenizer *have* a token that starts line N; a merge across the newline is
    ``GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE``'s refusal, not this one's. In the repo venv
    it runs only against a byte-level stand-in, where (1)-(3) cannot fail for the reason
    they exist. They are not green for the real tokenizer until one runs them.
    """
    text = spec.text
    for pos in positions:
        if not 0 <= pos < len(ids):
            raise ShardContractViolation(
                f"{where}: position {pos} is outside the {len(ids)} token(s) to decode"
            )
        first, last = offsets[pos]
        claimed = text[first:last]
        piece = decode([int(ids[pos])])
        if not isinstance(piece, str):
            raise ShardContractViolation(
                f"{where}: decode returned {type(piece).__name__}, not str"
            )
        if piece != claimed:
            raise ShardContractViolation(
                f"{where}: token {pos} decodes to {piece[:40]!r} but its offsets claim "
                f"characters [{first}, {last}) of the text, which are {claimed[:40]!r}. "
                "The tokenizer's offsets describe a different string than the one it "
                "tokenized, so every line start resolves to a plausible wrong token."
            )

    decoded = decode([int(i) for i in ids])
    if not isinstance(decoded, str):
        raise ShardContractViolation(
            f"{where}: decode returned {type(decoded).__name__}, not str"
        )
    grid = set(line_start_indices(decoded))
    astray = sorted(c for c in (spec.line_char_starts or ()) if c not in grid)
    if astray:
        raise ShardContractViolation(
            f"{where}: character offset(s) {astray[:5]} are recorded as the start of a "
            "context line, but they are not line starts of the text these ids decode to. "
            "The pointer head would range over positions that do not begin a line in what "
            "the tokenizer actually produced -- checked against decoded text rather than "
            "against the candidate set, which is projected from the same offsets and so "
            "could never show it (GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED)."
        )
    if decoded != text:
        at = next(
            (i for i, (a, b) in enumerate(zip(decoded, text, strict=False)) if a != b),
            min(len(decoded), len(text)),
        )
        raise ShardContractViolation(
            f"{where}: these ids do not decode to the text the spans were measured "
            f"against. They first differ at character {at}: decoded {decoded[at : at + 20]!r} "
            f"against {text[at : at + 20]!r}. Every offset into that text is then measured "
            "against a string the model will not see."
        )


def _span_token_positions(
    spec: SequenceSpec,
    ids: Sequence[int],
    *,
    token_offsets: TokenOffsets | None,
    decode: Decode | None = None,
    where: str,
) -> tuple[tuple[int, int], tuple[int, ...]] | None:
    """``((start, end), candidates)`` in token positions, or ``None`` for a non-span row.

    Both come off the *same* offset mapping, and the gold is one of the candidates by
    construction -- the character offsets it was built from are entries of
    ``spec.line_char_starts``. ``Batch`` requires a span gold to coincide with a candidate;
    that check is only meaningful because the two are projected together here, rather than
    derived twice and expected to agree.

    An abstaining row returns ``SPAN_ABSTAIN`` in both positions and still gets its
    candidates: the pointer head ranges over this context's line starts either way, and
    the abstention is the reserved row beside them.

    Refuses rather than approximates, because each of these silently moves the evidence the
    model is taught to cite: no offsets supplied; offsets that do not line up with the ids;
    a character in no token's span; two different gold lines collapsing onto one token; and
    a candidate set smaller than the context's line count.

    ``decode`` is optional and additive: when the tokenizer can turn an id back into text,
    :func:`_assert_spans_decode_to_their_text` checks every candidate and every gold
    position against what it says, which is the one check here that does not consult the
    offsets it is checking.
    """
    if spec.line_char_starts is None:
        return None
    n_tokens = len(ids)
    if not spec.line_char_starts:
        raise UnencodableGold(
            f"{where}: this is a span row whose context has no lines, so the pointer head "
            "has nothing to range over. qd-runtime sizes it at line_count + "
            "RESERVED_NOUL_ROWS (answer.rs:64) and Context::line_count is 0 for an empty "
            "context (context.rs:167), so what is served here is the abstain row alone, "
            "and Batch refuses a SLOT_SPAN row with an empty candidate set. Refused rather "
            "than given one candidate at offset 0 -- which is what this writer produced "
            "until the line rule converged on artifacts.line_start_indices, and it is a "
            "train/serve mismatch no loss curve shows."
        )
    if token_offsets is None:
        raise UnencodableGold(
            f"{where}: this is a span row, and mapping its context lines to token positions "
            "needs the tokenizer's character offsets. Pass token_offsets= to write_shards. "
            "Token ids alone cannot say which characters a token covers, and guessing is "
            "how a span comes to point at the wrong evidence."
        )
    offsets = list(token_offsets(spec.text))
    if len(offsets) != n_tokens:
        raise ShardContractViolation(
            f"{where}: token_offsets returned {len(offsets)} offsets but tokenize returned "
            f"{n_tokens} ids for the same text. The two describe different tokenizations, "
            "so a position derived from one would not index the other."
        )
    # The offsets must describe *this* text, not merely have the right length. A tokenizer
    # fed a normalised copy -- newlines stripped, whitespace collapsed -- returns the right
    # number of offsets for the wrong string, and every line start then resolves to a token
    # that is plausible, in range, and wrong. Requiring them to reach the end of the text is
    # what distinguishes the two; special tokens carrying (0, 0) are unaffected.
    reach = max((end for _, end in offsets), default=0)
    if reach != len(spec.text):
        raise ShardContractViolation(
            f"{where}: token_offsets reach character {reach} of a {len(spec.text)}-character "
            "text, so they describe a different string than the one tokenized -- a "
            "normalised copy, most likely. Every line start would map to a wrong token."
        )

    candidates = tuple(
        _token_index_for_char(offsets, c, where=where) for c in spec.line_char_starts
    )
    if len(set(candidates)) != len(candidates):
        raise UnencodableGold(
            f"{where}: {len(candidates)} context lines map to only "
            f"{len(set(candidates))} distinct tokens, so two lines share one candidate and "
            "the pointer head cannot tell them apart. qd-runtime sizes the head at "
            "line_count + RESERVED_NOUL_ROWS (answer.rs:64), so training over a smaller "
            "candidate set is a train/serve mismatch. Refused rather than deduplicated; "
            "see GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE."
        )
    if max(candidates) >= n_tokens:
        raise ShardContractViolation(
            f"{where}: a line-start candidate maps to token {max(candidates)} of {n_tokens}"
        )

    if spec.span_abstains or spec.span_char_starts is None:
        if decode is not None:
            _assert_spans_decode_to_their_text(
                spec, ids, offsets, candidates, decode=decode, where=where
            )
        return ((SPAN_ABSTAIN, SPAN_ABSTAIN), candidates)

    start_char, end_char = spec.span_char_starts
    start_tok = _token_index_for_char(offsets, start_char, where=where)
    end_tok = _token_index_for_char(offsets, end_char, where=where)
    if start_tok > end_tok:
        raise UnencodableGold(
            f"{where}: the gold's first line maps to token {start_tok} and its last to "
            f"{end_tok}, which is earlier. The offsets are not monotonic in the text."
        )
    if start_char != end_char and start_tok == end_tok:
        raise UnencodableGold(
            f"{where}: the gold's first and last evidence lines both fall inside token "
            f"{start_tok}, so a multi-line span would be recorded as a single-line one. "
            "The tokenization cannot express this span; refused rather than narrowed."
        )
    if end_tok >= n_tokens:
        raise ShardContractViolation(
            f"{where}: span end maps to token {end_tok} of {n_tokens}"
        )
    # Batch enforces this too, at read time. Enforced here as well so a mapping fault is
    # refused while the row that produced it can still be named, rather than surfacing as
    # a batch index during an epoch.
    missing = [p for p in (start_tok, end_tok) if p not in set(candidates)]
    if missing:
        raise UnencodableGold(
            f"{where}: gold position(s) {missing} are not line-start candidates. The gold "
            "and the candidate set are built from one list of line offsets, so this means "
            "the offset mapping is not monotonic in the text."
        )
    if decode is not None:
        # The gold positions first, so that when a whole-corpus mapping is shifted the
        # message names the gold rather than an arbitrary candidate.
        _assert_spans_decode_to_their_text(
            spec, ids, offsets, (start_tok, end_tok), decode=decode, where=where
        )
        _assert_spans_decode_to_their_text(
            spec, ids, offsets, candidates, decode=decode, where=where
        )
    return ((start_tok, end_tok), candidates)


def _match_rows_to_manifest(
    rows: Sequence[DataRow], entry_hashes: dict[str, int], *, path: Path
) -> None:
    """Every supplied row is in the manifest, and every manifest entry has a row.

    This is what makes "rows plus a manifest path" safe. ``open_training_data`` checked the
    *manifest*; nothing about that check constrains the ``DataRow`` objects handed in
    beside it, so without matching them the held-out rule would be satisfied by a training
    manifest while held-out rows were tokenized into the shard. Checked in both directions:
    an extra row is data the manifest never cleared, and a missing one is a corpus that is
    not the one the ``data_snapshot_hash`` in the header describes.
    """
    supplied: dict[str, int] = {}
    for row in rows:
        digest = row_content_hash(row)
        supplied[digest] = supplied.get(digest, 0) + 1

    unknown = sorted(set(supplied) - set(entry_hashes))
    if unknown:
        offenders = sorted(r.row_id for r in rows if row_content_hash(r) in set(unknown))
        raise ShardContractViolation(
            f"{path}: {len(unknown)} supplied row(s) do not appear in this manifest; the "
            f"first ids are {offenders[:5]}. The manifest is what rule 3 cleared, so a row "
            "that is not in it has not been cleared by anything. Rows and manifest must "
            "describe one corpus."
        )
    missing = sorted(set(entry_hashes) - set(supplied))
    if missing:
        raise ShardContractViolation(
            f"{path}: {len(missing)} manifest entry/entries have no supplied row. The "
            "header would pin a data_snapshot_hash for a corpus larger than what was "
            "written, so every later comparison against that hash would be false."
        )
    for digest, count in supplied.items():
        if count != entry_hashes[digest]:
            raise ShardContractViolation(
                f"{path}: a row with content hash {digest[:16]}… was supplied {count} "
                f"time(s) but appears {entry_hashes[digest]} time(s) in the manifest."
            )


def write_shards(
    manifest_path: Path,
    rows: Sequence[DataRow],
    *,
    out_dir: Path,
    remap: RemapTable,
    tokenize: Callable[[str], list[int]],
    config: DataConfig,
    repo_root: Path,
    token_offsets: TokenOffsets | None = None,
    decode: Decode | None = None,
    buckets: Sequence[int] | None = None,
    seed: int | None = None,
    caps: RenderCaps = DEFAULT_CAPS,
    allow_unencodable: bool = False,
    allow_not_run_snapshot: bool = False,
    max_sequences: int = DEFAULT_MAX_SEQUENCES,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
) -> ShardHeader:
    """Tokenize a cleared corpus into a shard set and return its header.

    ``manifest_path`` goes through ``open_training_data`` -- rule 3's door, and the source
    of the ``data_snapshot_hash`` pinned in the header. ``rows`` supply the text the
    manifest does not carry and are matched against it entry for entry; see
    :func:`_match_rows_to_manifest` for why that match is not a formality.

    ``tokenize`` is injected rather than imported so this path stays torch-free.
    ``remap.tokenizer_hash`` is the header's tokenizer hash -- there is deliberately no
    separate argument for it, because two ways to say which tokenizer was used is two ways
    for a shard set to disagree with the remap it was built against.

    ``token_offsets`` returns one ``(char_start, char_end)`` per token for the same text,
    and is **required as soon as one span row is present**: a span's gold is a pair of
    context *line numbers*, and turning a line into a token position needs to know which
    characters each token covers. Token ids alone cannot answer that, and a BPE token may
    straddle a line boundary, so this is not something to approximate -- a span pointing at
    the wrong token teaches the model to cite the wrong evidence and nothing downstream
    could tell. Absent it, a span row is refused rather than guessed at.

    ``decode`` is optional and is the check that ``GAP-SPAN-HEAD-LINE-MAPPING-BPE-
    UNVERIFIED`` asks for before any span number is reported: given ``tokenizer.decode``,
    every gold position and every candidate is checked against the text the tokenizer says
    that id stands for, rather than against arithmetic over the offsets being checked. It
    is the only check here that can catch a mapping which is *consistently* wrong, because
    every other one compares the gold to a candidate set projected from the same offsets
    (``GAP-S4-GOLD-ON-CANDIDATE-CANNOT-CATCH-LINE-SHIFT``). It is optional rather than
    required because the repo venv has no tokenizer to decode with; **a real-tokenizer run
    passes it**, and a run that omits it has not had its span mapping verified.

    ``buckets`` defaults to :func:`choose_buckets` over the measured lengths. ``seed``
    defaults to ``config.seed`` and drives the per-example option shuffle.

    ``allow_unencodable`` governs rows whose gold no ``Batch`` can express -- since the
    supervision channel landed, that is the **abstaining span** and nothing else (see
    :class:`UnencodableGold`). The default refuses the whole write. Passing ``True`` writes
    the rest and records the exclusion in ``coverage.json`` as a ``Ran`` whose ``n`` and
    ``n_total`` carry *both* numbers, so a partial corpus can never be read as a complete
    one; ``ShardReader.coverage`` surfaces it. It is never silent and never a substitution.
    """
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    if max_sequences <= 0 or max_total_tokens <= 0:
        raise ValueError(
            f"max_sequences and max_total_tokens must be positive, got {max_sequences} "
            f"and {max_total_tokens}"
        )

    handle = open_training_data(
        manifest_path,
        config=config,
        repo_root=repo_root,
        allow_not_run_snapshot=allow_not_run_snapshot,
    )
    manifest = handle.manifest

    entry_hashes: dict[str, int] = {}
    for entry in manifest.entries:
        entry_hashes[entry.content_hash] = entry_hashes.get(entry.content_hash, 0) + 1
    _match_rows_to_manifest(rows, entry_hashes, path=manifest_path)

    if not rows:
        raise ShardContractViolation(
            f"{manifest_path}: no rows. A shard set over zero sequences cannot be given a "
            "positive n_sequences, and writing one would put an empty corpus behind a "
            "header that looks like any other."
        )

    shuffle_seed = config.seed if seed is None else seed
    ordered = sorted(rows, key=lambda r: r.row_id)

    # Pass one: tokenize and remap. Held in memory as int32 arrays because the bucket
    # boundaries cannot be chosen until every length is known, and the header cannot be
    # built until the boundaries are.
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    kinds: list[int] = []
    spans: list[tuple[int, int]] = []
    candidates: list[tuple[int, ...]] = []
    excluded: list[str] = []
    total_tokens = 0
    for row in ordered:
        # A row is written whole or not at all. Its slots are one example's supervision, so
        # committing the ones that encoded and dropping the rest would leave a corpus that
        # answers some of each question -- and `excluded` counts rows, so the coverage
        # figure would be wrong as well as the corpus.
        staged: list[tuple[np.ndarray, str, int, tuple[int, int], tuple[int, ...]]] = []
        try:
            for spec in training_texts(row, seed=shuffle_seed, caps=caps):
                where = f"row {row.row_id!r} slot {spec.slot_name!r}"
                ids = _tokenize_checked(tokenize, spec.text, where=where)
                # Not caught: RemapTable.encode raises on an id the remap dropped, and that
                # exception is the S2<->S4 cross-lane check firing. Substituting a token
                # here would turn a coverage bug into a training example that teaches the
                # wrong thing, and the only symptom would be slightly worse loss.
                new_ids = remap.encode(ids)
                if int(new_ids.min()) < 0:
                    raise ShardContractViolation(
                        f"{where}: the remap produced a negative id, which cannot be stored "
                        f"as {TOKEN_DTYPE}"
                    )
                projected = _span_token_positions(
                    spec,
                    ids,
                    token_offsets=token_offsets,
                    decode=decode,
                    where=where,
                )
                if projected is None:
                    if spec.slot_kind == SLOT_SPAN:
                        raise ShardContractViolation(
                            f"{where}: a SLOT_SPAN row reached the writer with no candidates"
                        )
                    staged.append((new_ids, where, spec.slot_kind, (NO_SPAN, NO_SPAN), ()))
                else:
                    staged.append(
                        (new_ids, where, spec.slot_kind, projected[0], projected[1])
                    )
        except UnencodableGold as exc:
            if not allow_unencodable:
                raise
            excluded.append(f"{row.row_id}: {exc}")
            continue

        for new_ids, where, kind, span, cands in staged:
            sequences.append(new_ids)
            labels.append(where)
            kinds.append(kind)
            spans.append(span)
            candidates.append(cands)
            total_tokens += int(new_ids.size)
            if len(sequences) > max_sequences:
                raise ShardContractViolation(
                    f"{manifest_path}: over the {max_sequences} sequence bound at {where}"
                )
            if total_tokens > max_total_tokens:
                raise ShardContractViolation(
                    f"{manifest_path}: over the {max_total_tokens} token bound at {where}"
                )

    if not sequences:
        raise ShardContractViolation(
            f"{manifest_path}: every row was excluded, so there is nothing to write. "
            f"{len(excluded)} row(s) carried gold no Batch can express."
        )

    lengths = [int(s.size) for s in sequences]
    chosen = tuple(int(b) for b in buckets) if buckets is not None else choose_buckets(lengths)
    # The writer asks `bucket_for` -- the same function the sampler asks -- rather than
    # comparing against `chosen[-1]` itself. Two plausible spellings of "does this fit"
    # that disagree on the boundary case would put a sequence in the shard that the reader
    # then refuses, and the write would look clean. Doing it here also means an oversized
    # sequence is named while the row it came from is still in hand.
    for label, n in zip(labels, lengths, strict=True):
        try:
            bucket_for(n, chosen)
        except ValueError as exc:
            raise ShardContractViolation(
                f"{manifest_path}: {label} is {n} tokens, which fits no bucket in "
                f"{list(chosen)}. Refused rather than truncated: truncation drops the "
                "answer token off the end of the example, and the row still trains."
            ) from exc

    header = ShardHeader(
        split=manifest.split,
        data_snapshot_hash=handle.data_snapshot_hash,
        tokenizer_hash=remap.tokenizer_hash,
        remap_hash=remap.remap_hash(),
        vocab_size=remap.vocab_size,
        n_sequences=len(sequences),
        total_tokens=total_tokens,
        max_seq_len=max(lengths),
        buckets=chosen,
        created_at=datetime.now(UTC).isoformat(),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])

    tokens_path = out_dir / TOKENS_NAME
    flat = np.memmap(tokens_path, dtype=TOKEN_DTYPE, mode="w+", shape=(total_tokens,))
    for i, seq in enumerate(sequences):
        flat[offsets[i] : offsets[i + 1]] = seq.astype(TOKEN_DTYPE, copy=False)
    flat.flush()
    del flat

    np.save(out_dir / OFFSETS_NAME, offsets)
    # target_index is lengths-2 for every row this writer produces: the answer token is the
    # last one, so the position whose *next* token is the gold is the one before it. It is
    # stored rather than recomputed on read so the shard says what it supervises instead of
    # the reader assuming a convention -- which is the whole reason the channel exists.
    # The candidate sets are stored ragged -- positions plus offsets, the same shape as the
    # tokens themselves -- rather than as the [n, max_len] bool mask `Batch` wants. A mask
    # that wide is mostly false and grows with the longest sequence in the whole set; the
    # ragged form grows with the number of context lines, which is what it actually is.
    # `ShardReader.batches` materialises the mask per batch, at that batch's width.
    cand_offsets = np.zeros(len(candidates) + 1, dtype=np.int64)
    np.cumsum(np.asarray([len(c) for c in candidates], dtype=np.int64), out=cand_offsets[1:])
    flat_candidates = [p for c in candidates for p in c]
    np.savez(
        out_dir / SUPERVISION_NAME,
        slot_kind=np.asarray(kinds, dtype=np.uint8),
        target_index=(np.asarray(lengths, dtype=np.int32) - 2),
        span_target=np.asarray(spans, dtype=np.int32).reshape(len(spans), 2),
        candidate_offsets=cand_offsets,
        candidate_positions=np.asarray(flat_candidates, dtype=np.int32),
    )
    (out_dir / HEADER_NAME).write_text(
        json.dumps(header.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / COVERAGE_NAME).write_text(
        json.dumps(
            _coverage(len(ordered), excluded).to_json(), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return header


def _coverage(n_rows_in: int, excluded: list[str]) -> TriState:
    """Rows written against rows offered, as a tri-state carrying both numbers."""
    written = n_rows_in - len(excluded)
    if not excluded:
        return Ran(
            passed=True,
            value=written,
            n=written,
            n_total=n_rows_in,
            detail="every row in the manifest was tokenized into the shard set",
        )
    return Ran(
        passed=False,
        value=written,
        n=written,
        n_total=n_rows_in,
        detail=(
            f"{len(excluded)} of {n_rows_in} row(s) were excluded because their gold "
            "cannot be expressed as a final answer token; see UnencodableGold and "
            "GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL. First: " + "; ".join(excluded[:3])
        ),
    )


# -- the reader ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Plan:
    """One batch's membership, before any token is read."""

    bucket: int
    width: int
    rows: tuple[int, ...]


class ShardReader:
    """Reads a shard set and yields ``Batch`` objects in a reproducible order.

    The constructor calls ``assert_shard_trainable``: rule 3 at the shard boundary.
    ``open_training_data`` guards manifests, and a shard is a derived artifact the trainer
    opens directly, so without the same check here a training process could reach held-out
    data by indirection through a file that never passed the manifest door.

    ``batches`` order is a pure function of ``(seed, epoch, batch_tokens)`` and the shard
    set. There is no iterator RNG, no shuffled member state and nothing consumed on a
    previous call: S5's bit-exact resume reconstructs a position from ``(epoch, index)``
    alone, which is only sound if asking twice gives the same answer twice.
    """

    def __init__(self, root: Path, *, config: DataConfig, repo_root: Path) -> None:
        self.root = Path(root)
        self._config = config
        self._repo_root = Path(repo_root)

        raw = json.loads((self.root / HEADER_NAME).read_text(encoding="utf-8"))
        # from_json recomputes shard_hash and refuses a header edited after it was written.
        self.header: ShardHeader = ShardHeader.from_json(raw)
        self.checks: dict[str, TriState] = assert_shard_trainable(
            self.header, config=config, path=self.root
        )

        self._offsets: np.ndarray = np.load(self.root / OFFSETS_NAME)
        if self._offsets.dtype != np.int64 or self._offsets.ndim != 1:
            raise ShardContractViolation(
                f"{self.root}: offsets must be 1-D int64, got {self._offsets.ndim}-D "
                f"{self._offsets.dtype}"
            )
        if self._offsets.size != self.header.n_sequences + 1:
            raise ShardContractViolation(
                f"{self.root}: {self._offsets.size} offsets describe "
                f"{self._offsets.size - 1} sequences but the header declares "
                f"{self.header.n_sequences}"
            )
        if int(self._offsets[0]) != 0:
            raise ShardContractViolation(f"{self.root}: offsets must start at 0")
        if int(self._offsets[-1]) != self.header.total_tokens:
            raise ShardContractViolation(
                f"{self.root}: offsets end at {int(self._offsets[-1])} but the header "
                f"declares {self.header.total_tokens} tokens"
            )
        self._lengths: np.ndarray = np.diff(self._offsets)
        if self._lengths.size and int(self._lengths.min()) <= 0:
            raise ShardContractViolation(
                f"{self.root}: at least one sequence is empty or the offsets are not "
                "increasing"
            )
        if self._lengths.size and int(self._lengths.max()) != self.header.max_seq_len:
            raise ShardContractViolation(
                f"{self.root}: the longest sequence is {int(self._lengths.max())} tokens "
                f"but the header declares max_seq_len={self.header.max_seq_len}"
            )

        tokens_path = self.root / TOKENS_NAME
        expected = self.header.total_tokens * np.dtype(TOKEN_DTYPE).itemsize
        actual = tokens_path.stat().st_size
        if actual != expected:
            raise ShardContractViolation(
                f"{tokens_path}: {actual} bytes on disk, but the header describes "
                f"{self.header.total_tokens} {TOKEN_DTYPE} tokens ({expected} bytes). The "
                "shard set is truncated or was written by a different format."
            )
        self._tokens = np.memmap(
            tokens_path, dtype=TOKEN_DTYPE, mode="r", shape=(self.header.total_tokens,)
        )

        self._load_supervision()

        coverage_path = self.root / COVERAGE_NAME
        if coverage_path.exists():
            self.coverage: TriState = parse_tristate(
                json.loads(coverage_path.read_text(encoding="utf-8")),
                field=f"{coverage_path}",
            )
        else:
            self.coverage = NotRun(
                reason=(
                    f"{coverage_path} is absent, so how much of the manifest reached this "
                    "shard set was not recorded. Absent coverage is unknown coverage, not "
                    "full coverage."
                )
            )

    def _load_supervision(self) -> None:
        """Read the per-sequence supervision channel and check it against the tokens.

        Absent, it is refused rather than defaulted to CPT. A shard set written by this
        writer is FT -- every sequence is a prompt with an answer token -- so silently
        yielding CPT batches would run next-token loss over the whole prompt and call it
        fine-tuning. That is a different objective, not a missing field.
        """
        path = self.root / SUPERVISION_NAME
        if not path.exists():
            raise ShardContractViolation(
                f"{path} is absent. This shard set's sequences are prompt-plus-answer rows, "
                "so their supervision is not derivable from the tokens alone; yielding CPT "
                "batches instead would silently train a different objective."
            )
        n = self.header.n_sequences
        with np.load(path) as z:
            kinds = z["slot_kind"].astype(np.uint8, copy=True)
            target = z["target_index"].astype(np.int32, copy=True)
            spans = z["span_target"].astype(np.int32, copy=True)
            cand_offsets = z["candidate_offsets"].astype(np.int64, copy=True)
            cand_positions = z["candidate_positions"].astype(np.int32, copy=True)
        if kinds.shape != (n,) or target.shape != (n,) or spans.shape != (n, 2):
            raise ShardContractViolation(
                f"{path}: supervision arrays are {kinds.shape}, {target.shape}, "
                f"{spans.shape}; expected ({n},), ({n},) and ({n}, 2)"
            )
        if cand_offsets.shape != (n + 1,):
            raise ShardContractViolation(
                f"{path}: candidate_offsets is {cand_offsets.shape}, expected ({n + 1},)"
            )
        if int(cand_offsets[0]) != 0 or int(cand_offsets[-1]) != cand_positions.size:
            raise ShardContractViolation(
                f"{path}: candidate_offsets run 0..{int(cand_offsets[-1])} but there are "
                f"{cand_positions.size} candidate positions"
            )
        # target_index[i] must leave a next token inside sequence i -- the same rule
        # Batch._check_supervision enforces per batch, checked here over the whole set so a
        # bad shard is refused at open rather than at some batch in the middle of an epoch.
        if int(target.min()) < 0 or bool(np.any(target >= self._lengths - 1)):
            bad = int(np.flatnonzero((target < 0) | (target >= self._lengths - 1))[0])
            raise ShardContractViolation(
                f"{path}: target_index[{bad}]={int(target[bad])} is not in "
                f"[0, {int(self._lengths[bad])}-1); the supervised position's next token is "
                "the gold, so the last position supervises nothing"
            )
        is_span = kinds == SLOT_SPAN
        if bool(np.any(spans[~is_span] != NO_SPAN)):
            raise ShardContractViolation(
                f"{path}: a non-span sequence carries a span position; non-span rows must "
                f"be {NO_SPAN}"
            )
        abstaining = is_span & (spans[:, 0] == SPAN_ABSTAIN) & (spans[:, 1] == SPAN_ABSTAIN)
        half = is_span & ((spans[:, 0] == SPAN_ABSTAIN) != (spans[:, 1] == SPAN_ABSTAIN))
        if bool(half.any()):
            bad = int(np.flatnonzero(half)[0])
            raise ShardContractViolation(
                f"{path}: span_target[{bad}] abstains in one position only; a span row "
                "abstains in both or neither"
            )
        pointing = is_span & ~abstaining
        if bool(pointing.any()) and bool(np.any(spans[pointing, 1] >= self._lengths[pointing])):
            bad = int(np.flatnonzero(pointing & (spans[:, 1] >= self._lengths))[0])
            raise ShardContractViolation(
                f"{path}: span_target[{bad}] ends at {int(spans[bad, 1])} but that sequence "
                f"holds {int(self._lengths[bad])} tokens: the span points past it"
            )

        # The candidate set, checked at open for exactly the properties Batch checks per
        # batch. A shard whose candidates are wrong is wrong for every epoch it is read in,
        # so it is better refused here than at some batch index part-way through one.
        for i in np.flatnonzero(is_span):
            positions = cand_positions[cand_offsets[i] : cand_offsets[i + 1]]
            n_real = int(self._lengths[i])
            if positions.size == 0:
                raise ShardContractViolation(
                    f"{path}: sequence {i} is SLOT_SPAN but has no line-start candidates; a "
                    "sequence has at least one line, so an empty set is a mapping failure"
                )
            if int(positions.min()) < 0 or int(positions.max()) >= n_real:
                raise ShardContractViolation(
                    f"{path}: sequence {i} has a line-start candidate outside its "
                    f"{n_real} real tokens; the pointer head could answer with padding"
                )
            if pointing[i]:
                allowed = set(int(p) for p in positions)
                for name, pos in (("start", int(spans[i, 0])), ("end", int(spans[i, 1]))):
                    if pos not in allowed:
                        raise ShardContractViolation(
                            f"{path}: span_target[{i}] {name}={pos} is not one of that "
                            "sequence's line-start candidates, so the gold names a position "
                            "the pointer head cannot select"
                        )

        self._slot_kind = kinds
        self._target_index = target
        self._span_target = spans
        self._candidate_offsets = cand_offsets
        self._candidate_positions = cand_positions

    def __len__(self) -> int:
        return int(self.header.n_sequences)

    def lengths(self) -> list[int]:
        return [int(n) for n in self._lengths]

    def padding_waste(self) -> TriState:
        """The S4 gate, measured over one epoch, by the contract's own function.

        Delegates to ``artifacts.padding_waste`` rather than recomputing: the gate and the
        number it is applied to have to come from one place, and the threshold itself is
        read-only under rule 2.

        The figure it returns is the one this reader actually pays, because
        :meth:`batches` pads every row to its bucket's full width -- the same width
        ``padding_waste`` charges for. Padding to the longest sequence *in the batch*
        instead would be cheaper and would make the measured gate describe a padding
        scheme nobody trains with.
        """
        return padding_waste(self.lengths(), self.header.buckets)

    def candidates(self, i: int) -> np.ndarray:
        """Sequence *i*'s line-start token positions: the pointer head's candidate set.

        Empty for a non-span sequence, which has no pointer head and therefore nothing to
        choose between.
        """
        if not 0 <= i < len(self):
            raise IndexError(f"sequence {i} out of range for {len(self)} sequences")
        lo, hi = int(self._candidate_offsets[i]), int(self._candidate_offsets[i + 1])
        return self._candidate_positions[lo:hi]

    def sequence(self, i: int) -> np.ndarray:
        """Sequence *i* as post-remap ``int32`` ids."""
        if not 0 <= i < len(self):
            raise IndexError(f"sequence {i} out of range for {len(self)} sequences")
        start, end = int(self._offsets[i]), int(self._offsets[i + 1])
        return np.asarray(self._tokens[start:end], dtype=np.int32)

    def _plan(self, *, batch_tokens: int, seed: int, epoch: int) -> list[_Plan]:
        """Every batch of the epoch, as membership only. Pure in its arguments.

        Two independent shuffles, each with its own derived seed: one over the members of
        a bucket, one over the resulting batches. Deriving both from
        ``np.random.SeedSequence`` rather than from a module-level generator is what keeps
        the order reproducible -- a shared generator would make any bucket's order depend
        on how many draws the previous bucket happened to take.
        """
        if seed < 0 or epoch < 0:
            raise ValueError(f"seed and epoch must be non-negative, got {seed} and {epoch}")
        if batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive, got {batch_tokens}")

        buckets = self.header.buckets
        members: dict[int, list[int]] = {}
        for i, b in enumerate(assign_buckets(self.lengths(), buckets)):
            members.setdefault(b, []).append(i)

        too_narrow = sorted(b for b in members if buckets[b] > batch_tokens)
        if too_narrow:
            raise ValueError(
                f"batch_tokens={batch_tokens} is below bucket width {buckets[too_narrow[0]]}, "
                f"which holds {len(members[too_narrow[0]])} sequence(s). A single sequence "
                "from that bucket already exceeds the budget, so no batch can honour it. "
                "Raise batch_tokens or rebucket; silently emitting an over-budget batch is "
                "how an out-of-memory failure gets blamed on the model."
            )

        plans: list[_Plan] = []
        for b in sorted(members):
            width = int(buckets[b])
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, epoch, batch_tokens, b])
            )
            order = rng.permutation(np.asarray(members[b], dtype=np.int64))
            per_batch = min(batch_tokens // width, MAX_ROWS_PER_BATCH)
            for start in range(0, order.size, per_batch):
                chunk = order[start : start + per_batch]
                plans.append(_Plan(bucket=b, width=width, rows=tuple(int(i) for i in chunk)))

        # Interleave the buckets. Without this the epoch runs every short sequence before
        # every long one, so the optimizer sees a length curriculum nobody asked for.
        order_rng = np.random.default_rng(
            np.random.SeedSequence([seed, epoch, batch_tokens, 0xB17C])
        )
        shuffled = order_rng.permutation(len(plans))
        return [plans[int(i)] for i in shuffled]

    def batches(self, *, batch_tokens: int, seed: int, epoch: int) -> Iterator[Batch]:
        """One epoch of batches. Same arguments, same order, every time.

        Each row of each batch holds exactly one sequence, padded out to its bucket's
        width with :data:`PAD_ID`. Two sequences are never concatenated into one row --
        SAFETY-2, and the reason ``ShardHeader`` refuses ``packed=True``.

        Every batch carries the supervision channel: ``slot_kind`` says how each row is
        supervised, ``target_index`` the position whose next token is the gold letter, and
        ``span_target`` the gold evidence positions for ``SLOT_SPAN`` rows. ``span_target``
        is passed only when the batch actually holds a span row, because
        ``Batch._check_supervision`` refuses span positions on a batch with no span in it.
        """
        for index, plan in enumerate(self._plan(batch_tokens=batch_tokens, seed=seed, epoch=epoch)):
            rows = np.asarray(plan.rows, dtype=np.int64)
            tokens = np.full((len(plan.rows), plan.width), PAD_ID, dtype=np.int32)
            lengths = np.zeros(len(plan.rows), dtype=np.int64)
            for r, i in enumerate(plan.rows):
                seq = self.sequence(i)
                tokens[r, : seq.size] = seq
                lengths[r] = seq.size
            kinds = self._slot_kind[rows]
            spans = self._span_target[rows]
            has_span = bool((kinds == SLOT_SPAN).any())
            mask: np.ndarray | None = None
            if has_span:
                # Materialised at this batch's width, and only when a span row is present:
                # Batch refuses a candidate set on a batch with nothing to point.
                mask = np.zeros((len(plan.rows), plan.width), dtype=np.bool_)
                for r, i in enumerate(plan.rows):
                    mask[r, self.candidates(i)] = True
            yield Batch(
                tokens=tokens,
                lengths=lengths,
                bucket=plan.bucket,
                index=index,
                slot_kind=kinds,
                target_index=self._target_index[rows],
                span_target=spans if has_span else None,
                line_starts=mask,
            )

    def to_json(self) -> dict[str, Any]:
        """What a ledger row records about this shard set."""
        return {
            "root": str(self.root),
            "header": self.header.to_json(),
            "checks": {k: v.to_json() for k, v in sorted(self.checks.items())},
            "coverage": self.coverage.to_json(),
            "padding_waste": self.padding_waste().to_json(),
        }
