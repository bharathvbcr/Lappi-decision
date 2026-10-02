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
``span_check.json``
    Whether this set's span rows had their line->token mapping checked against *decoded*
    text, as a tri-state, and over how many rows. ``write_shards``' ``decode`` argument is
    optional -- the repo venv has no tokenizer to decode with -- and until this file
    existed a set written without it was byte-identical to one written with it, down to
    ``ShardReader.to_json()``, which is what a ledger row carries. So "the only check that
    does not consult the offsets it is checking was never run" and "it ran and passed" read
    the same downstream, which is the one thing this repository refuses to let happen.
    ``NotRun`` when ``decode`` was absent, and ``NotRun`` again when the set holds no span
    rows at all -- 0 of 0 verified is not a pass, for the same reason
    ``artifacts.padding_waste`` refuses to score an empty shard set.

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

import hashlib
import json
import os
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from qd_data.config import DataConfig
from qd_data.defect_class import NOUL_ROUTE_CONTRAST, NOUL_ROUTE_KEY
from qd_data.errors import QdRefusal
from qd_data.fingerprint import code_fingerprint
from qd_data.general import REPLAY_ONLY, REPLAY_ROLE_KEY
from qd_data.render import (
    DEFAULT_CAPS,
    M_CTX_END,
    PROMPT_FORMAT,
    RenderCaps,
    RenderedPrompt,
    render,
)
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
    ContrastRows,
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
    "CONTRADICTION_NAME",
    "COVERAGE_NAME",
    "DEFAULT_MAX_SEQUENCES",
    "DEFAULT_MAX_TOTAL_TOKENS",
    "GOLD_ROLE",
    "HEADER_NAME",
    "MAX_POSITIONS_PER_BATCH",
    "MAX_ROWS_PER_BATCH",
    "OFFSETS_NAME",
    "PAD_ID",
    "REMAP_NAME",
    "SEQUENCE_INDEX_FORMAT",
    "SEQUENCE_INDEX_NAME",
    "SPAN_CHECK_NAME",
    "SPAN_COLLAPSE_POLICIES",
    "SPAN_COLLAPSE_REFUSE_ANY",
    "SPAN_COLLAPSE_REFUSE_GOLD",
    "SUPERVISION_NAME",
    "TOKENS_NAME",
    "Decode",
    "EncodedSlot",
    "SequenceIndex",
    "SequenceSpec",
    "ShardReader",
    "SlotExclusion",
    "TokenOffsets",
    "UnencodableGold",
    "answer_letter",
    "assemble_batch",
    "choose_buckets",
    "corpus_contradictions",
    "encode_slot",
    "line_starts",
    "rendered_training_texts",
    "slot_kind_of",
    "training_texts",
    "write_shards",
]

HEADER_NAME: Final[str] = "header.json"
TOKENS_NAME: Final[str] = "tokens.u32"
OFFSETS_NAME: Final[str] = "offsets.npy"
COVERAGE_NAME: Final[str] = "coverage.json"
SUPERVISION_NAME: Final[str] = "supervision.npz"
SPAN_CHECK_NAME: Final[str] = "span_check.json"

#: What a span slot whose context lines collapse under BPE -- two line starts inside one
#: token -- does (GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE).
#:
#: * ``refuse-any`` (the default, and the only policy a gate shard set is written under):
#:   any collapse refuses the slot.
#: * ``refuse-gold`` (Fable round K, training shards and the report-only val slice only --
#:   ``write_shards(report_only=True)``): the slot is refused only when the
#:   gold's first or last line starts inside a token another line also starts in. Lines that
#:   collapse elsewhere **share one candidate**: ``candidates`` stays one entry per context
#:   line, the collapsed lines carry the same token index, and the batch's ``line_starts``
#:   mask -- a set of positions -- holds that token once. Measured on 300 composed rows,
#:   every collapse (696 of 696) was two consecutive blank context lines. A defect gold
#:   sits on such a pair in 16 of 20,898 composed and 18 of 41,504 v3 mutated rows
#:   (``test_pipeline_defect_class.py`` pins the counts); when the pair collapses, the
#:   slot is refused. The runtime's span candidates must build line start -> token with this same
#:   rule for a model trained under it (a requirement on G9(a)).
SPAN_COLLAPSE_REFUSE_ANY: Final[str] = "refuse-any"
SPAN_COLLAPSE_REFUSE_GOLD: Final[str] = "refuse-gold"
SPAN_COLLAPSE_POLICIES: Final[tuple[str, ...]] = (
    SPAN_COLLAPSE_REFUSE_ANY,
    SPAN_COLLAPSE_REFUSE_GOLD,
)

#: The self-consistency report :func:`corpus_contradictions` produces, written beside the
#: shards. It exists as a file so that "the corpus was checked" is a recorded artifact
#: rather than the absence of an exception: a set written by an older writer has no such
#: file, and that is a different state from a set whose check ran and found nothing.
CONTRADICTION_NAME: Final[str] = "contradictions.json"

#: Which ``(row_id, slot_name)`` each written sequence is, in write order, and every
#: ``(row_id, slot_name)`` that was excluded, with its scope and refusal. The contract a
#: consumer pairs labels to sequences by; ``coverage.json``'s free-text ``detail`` is a
#: human summary and not parseable as one. Written atomically, and its sha256 is pinned in
#: the header as ``sequence_index_hash``, which ``shard_hash`` covers. Schema:
#:
#: * ``format``: :data:`SEQUENCE_INDEX_FORMAT`;
#: * ``rows_in``: rows offered to the writer;
#: * ``role``: ``"gold"`` or ``"replay_only"`` -- see ``write_shards(replay=)``;
#: * ``sequences``: one ``{"row_id", "slot_name", "slot_kind"}`` per sequence, where list
#:   position ``i`` is sequence ``i`` of ``tokens.u32`` / ``supervision.npz``;
#: * ``excluded``: one ``{"row_id", "slot_name", "scope", "refusal", "detail"}`` per slot
#:   that produced no sequence. ``scope`` is ``"row"`` when the whole row was refused before
#:   any slot was tokenized (``render`` refused it, or its gold names no option) and
#:   ``"slot"`` when only that slot's sequence was refused. ``refusal`` is the exception's
#:   class name; ``detail`` its message.
#:
#: Every ``(row_id, slot_name)`` the rows offer is in exactly one of the two lists.
SEQUENCE_INDEX_NAME: Final[str] = "sequence_index.json"
SEQUENCE_INDEX_FORMAT: Final[str] = "qd-sequence-index/1"
#: The ``role`` of a shard set whose rows are trained against their gold.
GOLD_ROLE: Final[str] = "gold"

#: Stem of the remap this set's ids were written under. ``RemapTable.write`` appends
#: ``.npz`` and ``.json``, so the two files are ``remap.npz`` and ``remap.json``.
#:
#: The header pins ``remap_hash`` and ``assert_shard_trainable`` reports it as
#: ``shard_provenance_pinned``. Until this existed that hash named an artifact **nothing
#: kept**: ``RemapTable.write`` had exactly one caller in the repository and it was a
#: round-trip test, so the only real remap ever built -- by
#: ``tools/real_tokenizer_pipeline.py`` -- lived in one process and died with it. A shard
#: set is a file of renumbered ids; without the table that renumbered them, not one token
#: can be turned back into text, and the pinned hash cannot be checked against anything.
#: Re-deriving it means re-tokenizing the whole corpus with the same tokenizer, which is
#: the thing the hash exists to make unnecessary.
REMAP_NAME: Final[str] = "remap"

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

#: ...and in **positions**, which is the number that costs memory. :data:`MAX_ROWS_PER_BATCH`
#: bounds rows; a batch's cost is ``rows x width``, and until 2026-09-20 nothing bounded
#: that. ``_plan`` refused a ``batch_tokens`` *below* the widest bucket -- saying out loud
#: that "silently emitting an over-budget batch is how an out-of-memory failure gets blamed
#: on the model" -- and accepted any value above it. Measured on the real shard set:
#: ``batch_tokens=10_000_000_000`` was accepted and planned a batch of 1,380,880 positions,
#: roughly 260 GB of activations.
#:
#: This is a **backstop, not a gate**. It exists to turn a mis-typed ``batch_tokens`` (a
#: stray zero) into a refusal instead of an allocator failure; it is not a statement about
#: what fits. The bound that describes a real device is device-dependent and lives in
#: :mod:`qd_train.memory` -- ``max_positions_that_fit`` -- which ``tools/memory_budget.py``
#: prints per bucket. The value here is ~1.9x the largest batch that fits the biggest card
#: on the 2026-09-20 rental menu (96 GiB, bf16 weights and gradients, fp32 AdamW states,
#: per-layer gradient checkpointing: about 547,000 positions), so no recipe that fits any
#: real GPU reaches it.
MAX_POSITIONS_PER_BATCH: Final[int] = 1 << 20

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

    It is **not** the only row-level refusal ``write_shards`` handles. ``render`` refuses a
    context over ``RenderCaps.max_context_bytes``, a whitespace-only context, an over-long
    option and an over-long rendered prompt, and those are ``QdRefusal``s raised one frame
    inside :func:`training_texts`. They say the same thing this exception says -- this row
    cannot become an example -- so ``write_shards`` treats them identically. Measured on
    2026-09-20 over a real corpus: four of 321 rows carried a context above the cap, and
    while the ``except`` here named ``UnencodableGold`` alone, ``allow_unencodable=True``
    did not cover them and the whole write died at the first one with nothing counted.
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


#: Cells (positions x tokens) one comparison block of :func:`_token_indices_for_chars` may
#: hold: 16M booleans, 16 MiB per mask. A needle case is ~500 line starts over ~8.5K tokens
#: (~4M cells), so it is one block; the bound only splits a pathological input.
TOKEN_INDEX_BLOCK_CELLS: Final[int] = 1 << 24


def _token_indices_for_chars(
    offsets: Sequence[tuple[int, int]], char_positions: Sequence[int], *, where: str
) -> tuple[int, ...]:
    """For each of ``char_positions``, in order, the first token whose span contains it.

    A BPE token can straddle a line boundary -- the token holding the first character of
    line N may also hold the tail of line N-1 -- so "the token at this line start" is the
    token *containing* that character, which is what the pointer head would have to point
    at. When no token contains one the mapping has failed, and this raises -- naming the
    first such position in ``char_positions`` order -- rather than picking a neighbour: a
    span pointing at the wrong token teaches the model to cite the wrong evidence, and
    nothing downstream could tell.

    "First" is the lowest token index ``i`` with ``start_i <= c < end_i``, whatever the
    offsets look like: a byte-level BPE gives several tokens the same span (Qwen3.5: "Ṣ" ->
    (2,3) (2,3) (2,3)), special tokens carry (0, 0), and nothing here assumes the spans are
    sorted. It is one vectorised comparison of every position against every span, in blocks
    of at most :data:`TOKEN_INDEX_BLOCK_CELLS`, rather than a Python scan of the offsets per
    position: that scan was O(lines x tokens) interpreted steps, 18.0 of the 37.4 cProfiled
    seconds of the ``--needle`` suite's preparation in a phase-4 training prelude (2026-10-01).
    ``python/tests/test_token_index_for_chars.py`` holds it to the scan it replaced.
    """
    positions = np.asarray(char_positions, dtype=np.int64).reshape(-1)
    if positions.size == 0:
        return ()
    spans = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    if spans.shape[0] == 0:
        raise _no_token_contains(int(positions[0]), where=where)
    starts, ends = spans[:, 0][None, :], spans[:, 1][None, :]
    step = max(1, TOKEN_INDEX_BLOCK_CELLS // spans.shape[0])
    found = np.empty(positions.size, dtype=np.int64)
    for lo in range(0, positions.size, step):
        block = positions[lo : lo + step, None]
        inside = (starts <= block) & (block < ends)
        first = inside.argmax(axis=1)
        hit = inside[np.arange(first.size), first]
        if not bool(hit.all()):
            raise _no_token_contains(int(block[int(np.flatnonzero(~hit)[0]), 0]), where=where)
        found[lo : lo + step] = first
    return tuple(int(i) for i in found)


def _no_token_contains(char_pos: int, *, where: str) -> UnencodableGold:
    return UnencodableGold(
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
    conversion is :func:`_token_indices_for_chars`, and keeping the two steps apart is what
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


@dataclass(frozen=True, slots=True)
class SlotExclusion:
    """One ``(row_id, slot_name)`` that produced no sequence, and why. See
    :data:`SEQUENCE_INDEX_NAME` for what ``scope`` distinguishes."""

    row_id: str
    slot_name: str
    scope: str
    refusal: str
    detail: str

    def to_json(self) -> dict[str, str]:
        return {
            "row_id": self.row_id,
            "slot_name": self.slot_name,
            "scope": self.scope,
            "refusal": self.refusal,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SequenceIndex:
    """The parsed :data:`SEQUENCE_INDEX_NAME`: sequence ``i`` is ``sequences[i]``."""

    rows_in: int
    #: :data:`GOLD_ROLE` or ``qd_data.general.REPLAY_ONLY``: which kind of shard
    #: set this is. Pinned with the index, so a replay set cannot be read as a gold one.
    role: str
    sequences: tuple[tuple[str, str], ...]
    slot_kinds: tuple[int, ...]
    excluded: tuple[SlotExclusion, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format": SEQUENCE_INDEX_FORMAT,
            "rows_in": self.rows_in,
            "role": self.role,
            "sequences": [
                {"row_id": row_id, "slot_name": slot_name, "slot_kind": kind}
                for (row_id, slot_name), kind in zip(
                    self.sequences, self.slot_kinds, strict=True
                )
            ],
            "excluded": [e.to_json() for e in self.excluded],
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, where: str) -> SequenceIndex:
        """Parse and refuse anything that is not exactly the schema, naming the fault."""
        if raw.get("format") != SEQUENCE_INDEX_FORMAT:
            raise ShardContractViolation(
                f"{where}: format {raw.get('format')!r}, expected {SEQUENCE_INDEX_FORMAT!r}"
            )
        try:
            sequences = tuple((str(s["row_id"]), str(s["slot_name"])) for s in raw["sequences"])
            kinds = tuple(int(s["slot_kind"]) for s in raw["sequences"])
            excluded = tuple(
                SlotExclusion(
                    row_id=str(e["row_id"]), slot_name=str(e["slot_name"]),
                    scope=str(e["scope"]), refusal=str(e["refusal"]), detail=str(e["detail"]),
                )
                for e in raw["excluded"]
            )
            rows_in = int(raw["rows_in"])
            role = str(raw["role"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShardContractViolation(f"{where}: malformed sequence index: {exc!r}") from exc
        bad_scope = sorted({e.scope for e in excluded} - {"row", "slot"})
        if bad_scope:
            raise ShardContractViolation(f"{where}: unknown exclusion scope(s) {bad_scope}")
        keys = list(sequences) + [(e.row_id, e.slot_name) for e in excluded]
        if len(set(keys)) != len(keys):
            dup = next(k for k in keys if keys.count(k) > 1)
            raise ShardContractViolation(
                f"{where}: (row_id, slot_name) {dup!r} appears more than once across the "
                "written and excluded lists, so a label paired by it would be ambiguous"
            )
        if role not in (GOLD_ROLE, REPLAY_ONLY):
            raise ShardContractViolation(
                f"{where}: role {role!r} is not {GOLD_ROLE!r} or {REPLAY_ONLY!r}"
            )
        return cls(
            rows_in=rows_in, role=role, sequences=sequences, slot_kinds=kinds,
            excluded=excluded,
        )


def _write_bytes_atomically(path: Path, payload: bytes) -> None:
    """Write-then-rename in the same directory, fsynced, so a reader sees all or nothing."""
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


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
    return rendered_training_texts(row, seed=seed, caps=caps)[1]


def rendered_training_texts(
    row: DataRow, *, seed: int, caps: RenderCaps = DEFAULT_CAPS
) -> tuple[RenderedPrompt, list[SequenceSpec]]:
    """:func:`training_texts`, with the one ``render`` it was built from.

    For a caller that needs the rendered slots too -- ``tools/real_ft_run.py``'s ``_labels``
    reads each slot's ``letter_to_value`` -- so the row is rendered once rather than once
    there and again here. ``render`` was the largest single cost of relabelling the phase-4
    train split (258,072 rows, 2026-10-01), and it is a pure function of the request, the
    caps and the seed, so the second call could only ever return the first's answer.
    Raises exactly what :func:`training_texts` raises, at the same points.
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
    return rendered, out


# -- bucketing -------------------------------------------------------------------------


def choose_buckets(lengths: Sequence[int], *, n_buckets: int = 32) -> tuple[int, ...]:
    """Bucket boundaries that minimise padded positions over an observed length distribution.

    Every sequence is padded to its bucket's boundary (``ShardReader.batches`` pads to the
    full width, and :func:`qd_train.artifacts.padding_waste` charges exactly that), so the
    padding a boundary set costs is ``sum(boundary(n) - n)``. This returns the set of at
    most ``n_buckets`` boundaries for which that sum is smallest -- an exact optimum, not a
    heuristic, computed by :func:`_min_padding_boundaries`. The largest boundary is always
    the longest sequence, because ``bucket_for`` refuses a length that fits no bucket and
    ``ShardHeader`` refuses a largest bucket below ``max_seq_len`` -- a sequence with
    nowhere to go must be a loud write-time refusal, never a quiet truncation at read time.

    This picks *boundaries*; it does not touch the gate. ``MAX_PADDING_WASTE`` is read-only
    (rule 2), and lowering waste by bucketing better is the work the gate exists to demand.

    ## Why not equal-count quantiles

    This used to place boundaries at 32 equal-count quantiles, on the argument that waste
    is driven by where the sequences' *mass* sits. That holds for a unimodal body and fails
    on a tail: the last quantile bucket runs from the p96.9 length to the maximum, and every
    sequence in it pads to the maximum. The first ``code.defect_class`` build
    (ledger row 8f8558a9, ``AUDIT/shard-lengths-2026-09-29.json``: 78,643 sequences, median
    236 tokens, p99 908, max 37,098) put its 2,449 sequences above 713 tokens -- 494 of them
    this repository's own history at 2k-37k tokens -- into one bucket padded to 37,098, and
    wasted **74.47%** of all positions against the 15% gate. More quantiles do not reach
    it: at 64 the same set still wasted 57.37%, because a quantile boundary follows the
    count, and the tail is too few sequences to earn a boundary of its own.

    Measured, quantile against minimum-padding, same ``n_buckets``:

    | set | 8 | 16 | 32 | 64 |
    | --- | --- | --- | --- | --- |
    | 2026-09-29 defect build, quantile | 92.51% | 85.87% | 74.47% | 57.37% |
    | 2026-09-29 defect build, min-padding | 19.66% | 9.85% | **4.79%** | 2.23% |
    | 2026-09-20 GH200 set (341 seqs), quantile | 25.66% | 13.64% | 6.56% | 3.43% |
    | 2026-09-20 GH200 set, min-padding | 13.35% | 6.43% | 2.65% | 0.92% |

    ## Why 32

    The default was raised from 8 to 32 on the GH200 set, where the cost of more boundaries
    was measured on the axis that pays for it: ``ShardReader._plan`` chunks a bucket into
    batches of ``min(batch_tokens // width, MAX_ROWS_PER_BATCH)``, so more boundaries thin
    each bucket's membership and the bill arrives in each bucket's last, partly-filled
    chunk. That trade is unchanged by how the boundaries are placed, and 32 is kept.

    ## What it does NOT promise

    Optimal for a given ``n_buckets`` is not "under the gate". The GH200 set at 4 boundaries
    wastes 30.00% even optimally. At 32, no shape measured on 2026-09-29 failed -- 200
    corpora each of lognormal 1-1M tokens (100-800 rows, worst 3.92%), lognormal 200-35k
    (2-50 rows, worst 0.32%) and log-uniform 1-1M (2,000 rows, worst 5.21%), and 0 of the
    300 lognormal corpora the quantile rule failed 122 times -- but that is measurement,
    not a bound: a corpus with more than 32 distinct lengths of comparable token mass
    spread over many octaves must share buckets between lengths far apart. Such a corpus
    fails the gate by being that corpus; the remedy is the corpus or a larger
    ``n_buckets`` at the call site, never a lower bar. Two properties do hold for every
    input and are tested: more boundaries never waste more (the optimum over a larger
    budget includes the smaller one's), and no row is orphaned (the maximum is always a
    boundary).

    What it costs in batches, at ``batch_tokens`` = the longest sequence: on the defect
    build 878 batches (89.6 rows each) against 2,632 at 2 boundaries, 0.02% of rows in a
    bucket's under-tenth-full last chunk; on the GH200 set 108 batches (3.2 rows each)
    against the quantile rule's 115.
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
    distinct, counts = np.unique(values, return_counts=True)
    return _min_padding_boundaries(distinct, counts, n_buckets)


def _min_padding_boundaries(
    distinct: np.ndarray, counts: np.ndarray, n_buckets: int
) -> tuple[int, ...]:
    """The at-most-``n_buckets`` boundary set minimising total padding, exactly.

    With the distinct lengths ``v_1 < ... < v_m`` and prefix counts ``C`` and sums ``S``,
    a bucket holding ``v_{i+1}..v_j`` has boundary ``v_j`` (a boundary strictly between
    two distinct lengths only adds padding) and costs ``v_j (C_j - C_i) - (S_j - S_i)``.
    That cost is Monge -- ``cost(a,d) + cost(b,c) - cost(a,c) - cost(b,d) =
    (v_d - v_c)(C_b - C_a) >= 0`` for ``a <= b <= c <= d`` -- so the optimal split point
    is monotone in ``j`` and each layer of the dynamic programme is solved by divide and
    conquer in ``O(m log m)``, ``O(n_buckets m log m)`` overall. Integer arithmetic
    throughout, and ties go to the smallest split point, so the same lengths always give
    the same boundaries -- a header pins them.
    """
    m = int(distinct.size)
    if m <= n_buckets:
        return tuple(int(v) for v in distinct)
    v = distinct.astype(np.int64)
    c = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    s = np.concatenate(([0], np.cumsum(v * counts, dtype=np.int64)))
    # Larger than any real cost (at most max_len * n_sequences, bounded by the writer's
    # DEFAULT_MAX_TOTAL_TOKENS-scale inputs) and far enough from int64's ceiling that one
    # more cost added to it cannot overflow.
    unreachable = np.int64(1 << 62)
    prev = np.full(m + 1, unreachable, dtype=np.int64)
    prev[0] = 0
    splits: list[np.ndarray] = []
    for _ in range(n_buckets):
        cur = np.full(m + 1, unreachable, dtype=np.int64)
        arg = np.zeros(m + 1, dtype=np.int64)
        # (lo, hi, opt_lo, opt_hi) over j in [lo, hi]; an explicit stack, not recursion.
        stack = [(1, m, 0, m - 1)]
        while stack:
            lo, hi, opt_lo, opt_hi = stack.pop()
            if lo > hi:
                continue
            mid = (lo + hi) // 2
            top = min(mid - 1, opt_hi)
            cand = np.arange(opt_lo, top + 1)
            cost = prev[cand] + v[mid - 1] * (c[mid] - c[cand]) - (s[mid] - s[cand])
            k = int(np.argmin(cost))
            cur[mid] = cost[k]
            arg[mid] = opt_lo + k
            stack.append((lo, mid - 1, opt_lo, int(arg[mid])))
            stack.append((mid + 1, hi, int(arg[mid]), opt_hi))
        splits.append(arg)
        prev = cur
    boundaries: list[int] = []
    j = m
    for arg in reversed(splits):
        if j == 0:
            break
        boundaries.append(int(v[j - 1]))
        j = int(arg[j])
    if j != 0:  # pragma: no cover - the programme always reaches the empty prefix
        raise AssertionError(f"bucket partition ended at distinct length {j}, not 0")
    return tuple(sorted(boundaries))


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


@dataclass(frozen=True, slots=True)
class EncodedSlot:
    """One slot's sequence as the writer stores it: post-remap ids and span supervision.

    ``span`` is ``(NO_SPAN, NO_SPAN)`` and ``candidates`` empty for a letter slot; a span
    slot carries its gold token positions (or ``SPAN_ABSTAIN`` twice) and its line starts.
    """

    ids: np.ndarray
    slot_kind: int
    span: tuple[int, int]
    candidates: tuple[int, ...]


def encode_slot(
    spec: SequenceSpec,
    *,
    tokenize: Callable[[str], list[int]],
    remap: RemapTable,
    token_offsets: TokenOffsets | None,
    decode: Decode | None,
    where: str,
    span_collapse_policy: str = SPAN_COLLAPSE_REFUSE_ANY,
) -> EncodedSlot:
    """Tokenize, remap and project one slot's sequence -- the writer's only way to do it.

    Public so an eval that builds sequences the shard writer never saw (the needle suite)
    encodes them through the same checks rather than a second copy of them. Raises
    :class:`UnencodableGold` when the span cannot be placed in this tokenization; the
    writer decides whether that excludes the slot or aborts.
    """
    ids = _tokenize_checked(tokenize, spec.text, where=where)
    # Not caught: RemapTable.encode raises on an id the remap dropped, and that exception
    # is the S2<->S4 cross-lane check firing. Substituting a token here would turn a
    # coverage bug into a training example that teaches the wrong thing, and the only
    # symptom would be slightly worse loss.
    new_ids = remap.encode(ids)
    if int(new_ids.min()) < 0:
        raise ShardContractViolation(
            f"{where}: the remap produced a negative id, which cannot be stored as {TOKEN_DTYPE}"
        )
    projected = _span_token_positions(
        spec, ids, token_offsets=token_offsets, decode=decode, where=where,
        span_collapse_policy=span_collapse_policy,
    )
    if projected is None:
        if spec.slot_kind == SLOT_SPAN:
            raise ShardContractViolation(
                f"{where}: a SLOT_SPAN row reached the writer with no candidates"
            )
        return EncodedSlot(new_ids, spec.slot_kind, (NO_SPAN, NO_SPAN), ())
    return EncodedSlot(new_ids, spec.slot_kind, projected[0], projected[1])


def assemble_batch(
    sequences: Sequence[np.ndarray],
    *,
    kinds: np.ndarray,
    target_index: np.ndarray,
    spans: np.ndarray,
    candidates: Sequence[Sequence[int] | np.ndarray],
    width: int,
    bucket: int,
    index: int,
) -> Batch:
    """One padded batch from its sequences and their supervision, one row per sequence.

    The only place a ``Batch`` is put together from stored sequences: the shard reader and
    the needle suite both come through here. ``span_target`` and ``line_starts`` are passed
    only when a span row is present, because ``Batch`` refuses either on a batch with
    nothing to point.
    """
    n = len(sequences)
    tokens = np.full((n, width), PAD_ID, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int64)
    for r, seq in enumerate(sequences):
        tokens[r, : seq.size] = seq
        lengths[r] = seq.size
    has_span = bool((kinds == SLOT_SPAN).any())
    mask: np.ndarray | None = None
    if has_span:
        mask = np.zeros((n, width), dtype=np.bool_)
        for r, cands in enumerate(candidates):
            mask[r, np.asarray(cands, dtype=np.int64)] = True
    return Batch(
        tokens=tokens,
        lengths=lengths,
        bucket=bucket,
        index=index,
        slot_kind=kinds,
        target_index=target_index,
        span_target=spans if has_span else None,
        line_starts=mask,
    )


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

    1. **Each checked token's run decodes to exactly the characters its offsets claim.**
       The run is the token plus the tokens after it that start inside its claimed span:
       the byte pieces of a character a byte-level BPE split, which each claim the whole
       character and decode alone to U+FFFD. For a token its successor does not overlap
       -- the ordinary case -- the run is that one token. A tokenizer whose
       ``return_offsets_mapping`` describes a normalised copy of the text satisfies every
       length and reach check in this module and fails here
       (``test_span_decode_multibyte.py``).
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
        # A byte-level BPE splits a multi-byte character into several tokens that each
        # claim the whole character (Qwen3.5: "Ṣ" -> (2,3) (2,3) (2,3); merged into a
        # preceding space -> (19,21) (20,21) (20,21)). One byte of it decodes to U+FFFD, so
        # the unit compared is the run of tokens that start inside the span claimed so far.
        # A token that starts at or past its predecessor's end -- every token of a
        # tokenizer that splits on character boundaries -- ends the run at length one.
        run_end = pos + 1
        while (
            run_end < len(ids)
            and first <= offsets[run_end][0] < last
            and offsets[run_end][1] > offsets[run_end][0]
        ):
            last = max(last, offsets[run_end][1])
            run_end += 1
        claimed = text[first:last]
        piece = decode([int(i) for i in ids[pos:run_end]])
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
    span_collapse_policy: str = SPAN_COLLAPSE_REFUSE_ANY,
) -> tuple[tuple[int, int], tuple[int, ...]] | None:
    """``((start, end), candidates)`` in token positions, or ``None`` for a non-span row.

    ``span_collapse_policy`` decides what collapsed line starts do; see
    :data:`SPAN_COLLAPSE_REFUSE_ANY` and :data:`SPAN_COLLAPSE_REFUSE_GOLD`.

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
    # A text the tokenizer itself rewrites: Qwen3.5's normalizer is NFC, and a character
    # NFC changes (Bengali U+09DF, a composition exclusion, decomposes) makes the ids decode
    # to a different string than the one the offsets index. The offsets are aligned to the
    # original, but nothing can then check them against decoded text, so the slot is refused
    # here, counted, rather than reaching the decode check below -- which still aborts the
    # write for any other mismatch, the tokenizer-wiring fault it exists for. Only text that
    # is not itself NFC-stable can take this exit, so an NFC-stable text never does.
    if (
        decode is not None
        and not unicodedata.is_normalized("NFC", spec.text)
        and decode([int(i) for i in ids]) == unicodedata.normalize("NFC", spec.text)
    ):
        raise UnencodableGold(
            f"{where}: the context is not NFC-stable and the tokenizer's NFC normalizer "
            "rewrites it, so its ids decode to a different string than the line offsets "
            "index and the line mapping cannot be verified against decoded text. "
            "Refused for this slot rather than trusted unverified."
        )

    candidates = _token_indices_for_chars(offsets, spec.line_char_starts, where=where)
    if span_collapse_policy not in SPAN_COLLAPSE_POLICIES:
        raise ValueError(
            f"span_collapse_policy must be one of {SPAN_COLLAPSE_POLICIES}, "
            f"got {span_collapse_policy!r}"
        )
    collapsed = len(set(candidates)) != len(candidates)
    if collapsed and span_collapse_policy == SPAN_COLLAPSE_REFUSE_ANY:
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
    start_tok, end_tok = _token_indices_for_chars(offsets, (start_char, end_char), where=where)
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
    if collapsed:
        # Only reached under refuse-gold: collapsed lines elsewhere share their token, but a
        # gold line that shares one is a target the pointer cannot tell from its neighbour.
        shared = [p for p in (start_tok, end_tok) if candidates.count(p) > 1]
        if shared:
            raise UnencodableGold(
                f"{where}: the gold's line start shares token(s) {sorted(set(shared))} with "
                "another context line, so the pointer head cannot tell the gold from its "
                "neighbour. Refused under span_collapse_policy=refuse-gold, which keeps a "
                "collapse only when it misses the gold."
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


def _check_replay_roles(rows: Sequence[DataRow], *, replay: bool, path: Path) -> None:
    """A gold shard set holds no replay-only row, and a replay set holds nothing else.

    ``qd_data.general.partition_replay`` marks replay rows
    ``metadata[REPLAY_ROLE_KEY] == REPLAY_ONLY``: they are trained toward the base
    model's own answers and their gold is never read. Written into a gold set they
    would be cross-entropy-trained to that gold after all, which is the one thing the
    role exists to prevent; a gold row in a replay set would be KL-trained with its
    label ignored. Refused for the whole write, before any tokenization.
    """
    if replay:
        wrong = [r.row_id for r in rows if r.metadata.get(REPLAY_ROLE_KEY) != REPLAY_ONLY]
        kind = "gold (not replay-only) row(s) offered to a replay shard set"
    else:
        wrong = [r.row_id for r in rows if r.metadata.get(REPLAY_ROLE_KEY) == REPLAY_ONLY]
        kind = "replay-only row(s) offered to a gold shard set"
    if wrong:
        raise ShardContractViolation(
            f"{path}: {len(wrong)} {kind}, first {sorted(wrong)[:3]}. Replay rows go to"
            " their own set (write_shards(replay=True)); gold rows never do."
        )


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


def corpus_contradictions(
    sequences: Sequence[np.ndarray],
    *,
    target_index: Sequence[int],
    slot_kinds: Sequence[int],
    span_targets: Sequence[tuple[int, int]],
    labels: Sequence[str],
    max_examples: int = 3,
) -> dict[str, object]:
    """Sequences with **identical prefixes and different golds**, over a whole corpus.

    A causal model conditions on ``tokens[:target_index + 1]`` and nothing else, so two
    sequences that agree there and disagree about the answer cannot both be fitted. The
    optimum over such a group is its empirical label distribution: two rows, two golds,
    and the loss floors at ``ln 2`` no matter how long it trains. On a rented GPU the only
    symptom is a curve that plateaus and an eval that reports 50%.

    This is the canonical owner of that question. ``write_shards`` asks it before writing
    a byte, and ``tools/real_ft_run.py`` asks it again of the set it is about to train on;
    both get the same answer from the same code, which is the point -- the two used to be
    one function in the training tool and one absence at the write boundary, so a corpus
    could be *written* self-contradictory and only a GPU-hours-long run would say so.

    Measured over the **set**, not per batch: a contradiction does not need the two rows
    to land in one batch to be unfittable.

    ``per_kind`` is keyed by the integer slot kind rather than by a display name. Naming is
    the caller's boundary, and a mapping that has to be kept total over four constants in
    two modules is one more thing that can disagree with itself.
    """
    n = len(sequences)
    if not (n == len(target_index) == len(slot_kinds) == len(span_targets) == len(labels)):
        raise ValueError(
            f"corpus_contradictions was given {n} sequence(s) but "
            f"{len(target_index)} target index/indices, {len(slot_kinds)} slot kind(s), "
            f"{len(span_targets)} span target(s) and {len(labels)} label(s). A row whose "
            "gold came from a different index is the defect this function exists to find."
        )

    groups: dict[bytes, list[int]] = {}
    for i in range(n):
        at = int(target_index[i])
        # Cast before hashing: the writer holds freshly encoded arrays and the reader holds
        # a memmap, and two arrays with the same ids in different dtypes have different
        # bytes. Without this the two callers would silently disagree about whether a
        # corpus contradicts itself, which is the failure mode one level up.
        prefix = np.asarray(sequences[i][: at + 1], dtype=TOKEN_DTYPE).tobytes()
        groups.setdefault(hashlib.sha256(prefix).digest(), []).append(i)

    colliding = 0
    contradicting: list[dict[str, object]] = []
    per_kind: dict[int, dict[str, int]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        colliding += len(members)
        golds: list[str] = []
        for i in members:
            if int(slot_kinds[i]) == SLOT_SPAN:
                golds.append(f"span:{int(span_targets[i][0])},{int(span_targets[i][1])}")
            else:
                golds.append(f"letter:{int(sequences[i][-1])}")
        if len(set(golds)) < 2:
            continue
        kinds = {int(slot_kinds[i]) for i in members}
        for kind in kinds:
            bucket = per_kind.setdefault(kind, {"groups": 0, "rows": 0})
            bucket["groups"] += 1
            bucket["rows"] += sum(1 for i in members if int(slot_kinds[i]) == kind)
        contradicting.append({
            "rows": [str(labels[i]) for i in members],
            "kinds": sorted(kinds),
            "golds": golds,
            "prefix_tokens": int(target_index[members[0]]) + 1,
        })
    return {
        "sequences": n,
        "rows_sharing_a_prefix": colliding,
        "contradicting_groups": len(contradicting),
        "contradicting_rows": sum(len(c["rows"]) for c in contradicting),  # type: ignore[arg-type]
        "per_kind": {str(k): v for k, v in sorted(per_kind.items())},
        "examples": contradicting[:max_examples],
    }


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
    allow_contradictions: bool = False,
    replay: bool = False,
    corpus_rev: str = "",
    max_sequences: int = DEFAULT_MAX_SEQUENCES,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    max_seq_len: int | None = None,
    span_collapse_policy: str = SPAN_COLLAPSE_REFUSE_ANY,
    report_only: bool = False,
    exclusions_sha256: str = "",
    contrast_rows: ContrastRows | None = None,
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

    Whether ``decode`` was supplied is itself recorded, in ``span_check.json``, because a
    check that did not run must not read as one that ran and passed -- and without that
    file the two writes produce byte-identical artifacts.

    ``replay`` says which kind of set this is. ``False`` (a gold set) refuses any row marked
    ``metadata["replay_role"] == "replay_only"``; ``True`` (the replay set that
    ``qd_train.replay`` reads) refuses any row that is not. See :func:`_check_replay_roles`.
    The role is recorded in :data:`SEQUENCE_INDEX_NAME`.

    ``allow_unencodable`` governs rows this writer cannot turn into a sequence: a gold no
    ``Batch`` can express (see :class:`UnencodableGold`) **and** a row ``render`` refuses
    outright -- over the context cap, whitespace-only, over the option or rendered-prompt
    cap. Both are row-level and neither is recoverable by guessing, so both are handled the
    same way. A ``ShardContractViolation`` and a dropped token are deliberately *not* in
    that set: those say the artifacts disagree with each other, not that one row is
    unusable, and absorbing them would turn a contract fault into a smaller corpus.

    The default refuses the whole write and re-raises the original exception. Passing
    ``True`` writes the rest and records the exclusion in ``coverage.json`` as a ``Ran``
    whose ``n`` and ``n_total`` carry *both* numbers, so a partial corpus can never be read
    as a complete one; ``ShardReader.coverage`` surfaces it, and the detail names what
    refused each row. It is never silent and never a substitution.

    ``max_seq_len`` is a hard width: a row any of whose slot sequences is longer is refused
    whole (``OverMaxSeqLen``), under the same two rules -- the write fails without
    ``allow_unencodable`` and records the exclusion with it. Never truncated: truncation
    drops the answer token. ``None`` (the default) caps nothing, and the widest bucket is the
    longest sequence as before.

    ``span_collapse_policy`` is :data:`SPAN_COLLAPSE_REFUSE_ANY` by default. A caller may pass
    :data:`SPAN_COLLAPSE_REFUSE_GOLD` for a TRAINING set, or for a val set that is
    ``report_only`` -- outside every gate population, written refuse-gold so its span slots
    are the population training reads (Fable round K). A gate val set keeps the default, so
    every gate's span population stays the one it was measured on (rule 2). Both choices are
    written into the header and covered by its hash; gate readers refuse a report-only or
    non-refuse-any header (``ShardHeader.require_gate_population``).
    """
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    if max_sequences <= 0 or max_total_tokens <= 0:
        raise ValueError(
            f"max_sequences and max_total_tokens must be positive, got {max_sequences} "
            f"and {max_total_tokens}"
        )
    if max_seq_len is not None and (
        not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool) or max_seq_len < 2
    ):
        raise ValueError(f"max_seq_len must be an int of at least 2, got {max_seq_len!r}")
    if span_collapse_policy not in SPAN_COLLAPSE_POLICIES:
        raise ValueError(
            f"span_collapse_policy must be one of {SPAN_COLLAPSE_POLICIES}, "
            f"got {span_collapse_policy!r}"
        )
    if not isinstance(report_only, bool):
        raise ValueError(f"report_only must be a bool, got {report_only!r}")
    # The header's contrast record and the rows must agree: a set carrying contrast rows that
    # its header does not name, or naming rows it does not carry, would let a trainer read a
    # v5 set as another (qd_train.contrast).
    n_contrast = sum(r.metadata.get(NOUL_ROUTE_KEY) == NOUL_ROUTE_CONTRAST for r in rows)
    if n_contrast != (0 if contrast_rows is None else contrast_rows.count) or (
        contrast_rows is not None and replay
    ):
        raise ShardContractViolation(
            f"{manifest_path}: {n_contrast} contrast row(s) among the rows, and the header "
            f"would record {contrast_rows!r} on a {'replay' if replay else 'gold'} set; a gold "
            "train set names exactly the contrast rows it carries, and a replay set carries none"
        )

    handle = open_training_data(
        manifest_path,
        config=config,
        repo_root=repo_root,
        allow_not_run_snapshot=allow_not_run_snapshot,
    )
    manifest = handle.manifest
    if report_only and (
        manifest.split != "val" or span_collapse_policy != SPAN_COLLAPSE_REFUSE_GOLD
    ):
        raise ShardContractViolation(
            f"{manifest_path}: report_only is a val set written {SPAN_COLLAPSE_REFUSE_GOLD}; "
            f"this is the {manifest.split!r} split under {span_collapse_policy}"
        )
    if (
        span_collapse_policy != SPAN_COLLAPSE_REFUSE_ANY
        and manifest.split != "train"
        and not report_only
    ):
        raise ShardContractViolation(
            f"{manifest_path}: span_collapse_policy={span_collapse_policy} is a training-side "
            f"policy, and this is the {manifest.split!r} split. Val and gate sets keep "
            f"{SPAN_COLLAPSE_REFUSE_ANY}, so every gate's span population is the one it was "
            "measured on (rule 2); only a report_only val set may differ."
        )

    entry_hashes: dict[str, int] = {}
    for entry in manifest.entries:
        entry_hashes[entry.content_hash] = entry_hashes.get(entry.content_hash, 0) + 1
    _match_rows_to_manifest(rows, entry_hashes, path=manifest_path)
    _check_replay_roles(rows, replay=replay, path=manifest_path)

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
    #: `(row_id, slot_name)` per written sequence, and every slot that wrote none: the
    #: structured record SEQUENCE_INDEX_NAME carries. `excluded` above counts ROWS with
    #: nothing written and feeds coverage.json; this counts SLOTS.
    written_keys: list[tuple[str, str]] = []
    slot_exclusions: list[SlotExclusion] = []
    total_tokens = 0
    n_span_sequences = 0
    for row in ordered:
        # Two scopes of refusal, split where `training_texts` returns. Everything it raises
        # is about the ROW -- `render` refused the request (over a cap, whitespace-only), or
        # a gold names no rendered option -- so no slot of it can be written. Everything
        # raised after it is about ONE SLOT's sequence: its tokenization cannot place that
        # slot's span. Each slot is its own sequence with its own supervision, so a span
        # that collapses under BPE says nothing about whether the row's choice slot can be
        # trained. Refusing the whole row for it dropped the class label too, unevenly by
        # class (GAP-A3-BPE-SPAN-COLLAPSE-DROPS-DEFECT-CLASS-LABELS: stub 3,567 of 6,393),
        # which moved the majority rate the choice head is judged against.
        try:
            specs = training_texts(row, seed=shuffle_seed, caps=caps)
        except (UnencodableGold, QdRefusal) as exc:
            # QdRefusal alongside UnencodableGold: `render` refuses an over-cap context, a
            # whitespace-only one, an over-long option and an over-long prompt, and those
            # arrive from inside `training_texts`. They are row-level refusals with the
            # same meaning, and naming only UnencodableGold here made `allow_unencodable`
            # -- the mechanism that carries both numbers -- blind to a class real corpora
            # actually produce. ShardContractViolation and TokenNotInRemap are outside
            # QdRefusal by construction and still escape: those are contract faults.
            if not allow_unencodable:
                raise
            excluded.append(f"{row.row_id}: {type(exc).__name__}: {exc}")
            slot_exclusions.extend(
                SlotExclusion(
                    row_id=row.row_id, slot_name=slot.name, scope="row",
                    refusal=type(exc).__name__, detail=str(exc),
                )
                for slot in row.request.slots
            )
            continue

        staged: list[tuple[np.ndarray, str, str, int, tuple[int, int], tuple[int, ...]]] = []
        refused_here: list[SlotExclusion] = []
        for spec in specs:
            where = f"row {row.row_id!r} slot {spec.slot_name!r}"
            try:
                encoded = encode_slot(
                    spec, tokenize=tokenize, remap=remap, token_offsets=token_offsets,
                    decode=decode, where=where, span_collapse_policy=span_collapse_policy,
                )
            except UnencodableGold as exc:
                # Only UnencodableGold is slot-scoped: it says this slot's span cannot be
                # placed in this tokenization. ShardContractViolation from the same call
                # says the offsets and ids disagree, which is a fault in the tokenizer
                # wiring, not in one row, and still aborts the write.
                if not allow_unencodable:
                    raise
                refused_here.append(
                    SlotExclusion(
                        row_id=row.row_id, slot_name=spec.slot_name, scope="slot",
                        refusal=type(exc).__name__, detail=str(exc),
                    )
                )
                continue
            staged.append(
                (encoded.ids, where, spec.slot_name, spec.slot_kind, encoded.span,
                 encoded.candidates)
            )
        if max_seq_len is not None and any(int(s[0].size) > max_seq_len for s in staged):
            longest = max(int(s[0].size) for s in staged)
            detail = f"{longest} tokens, over max_seq_len={max_seq_len}"
            if not allow_unencodable:
                raise ShardContractViolation(
                    f"{manifest_path}: row {row.row_id!r} is {detail}. Refused rather than "
                    "truncated: truncation drops the answer token off the end of the example"
                )
            excluded.append(f"{row.row_id}: OverMaxSeqLen: {detail}")
            slot_exclusions.extend(refused_here)
            slot_exclusions.extend(
                SlotExclusion(
                    row_id=row.row_id, slot_name=s[2], scope="row", refusal="OverMaxSeqLen",
                    detail=detail,
                )
                for s in staged
            )
            continue
        slot_exclusions.extend(refused_here)
        if not staged:
            # Every slot refused on its own account: the row wrote nothing, so it is a row
            # exclusion for coverage.json as well as N slot exclusions in the index.
            excluded.append(
                f"{row.row_id}: every slot refused: "
                + "; ".join(f"{e.slot_name}: {e.refusal}: {e.detail}" for e in refused_here)
            )
            continue

        for new_ids, where, slot_name, kind, span, cands in staged:
            written_keys.append((row.row_id, slot_name))
            sequences.append(new_ids)
            labels.append(where)
            kinds.append(kind)
            spans.append(span)
            candidates.append(cands)
            total_tokens += int(new_ids.size)
            n_span_sequences += 1 if kind == SLOT_SPAN else 0
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

    # Before a byte is written. `target_index` is `lengths - 2` here for the same reason it
    # is below -- the answer token is last, so the supervised position is the one before it
    # -- and it is computed from the same expression rather than read back from the file
    # this function has not written yet.
    contradictions = corpus_contradictions(
        sequences,
        target_index=[n - 2 for n in lengths],
        slot_kinds=kinds,
        span_targets=spans,
        labels=labels,
    )
    if int(contradictions["contradicting_rows"]) and not allow_contradictions:  # type: ignore[arg-type]
        first = contradictions["examples"][0]  # type: ignore[index]
        raise ShardContractViolation(
            f"{manifest_path}: {contradictions['contradicting_rows']} sequence(s) in "
            f"{contradictions['contradicting_groups']} group(s) share a prefix up to their "
            "supervised position and disagree about the gold. A causal model conditions on "
            "that prefix and nothing else, so no parameters fit both: the loss floors at "
            "the group's label entropy and an eval reports the coin flip as if it were "
            "accuracy. Refused at the write boundary rather than reported by the trainer, "
            "because the cheapest place to find this is before the GPU is rented. First "
            f"group: {first['rows']} with golds {first['golds']} over "  # type: ignore[index]
            f"{first['prefix_tokens']} prefix token(s). Pass allow_contradictions=True to "  # type: ignore[index]
            "write it anyway, which is a decision about the corpus, not about this check."
        )

    index = SequenceIndex(
        rows_in=len(ordered),
        role=REPLAY_ONLY if replay else GOLD_ROLE,
        sequences=tuple(written_keys),
        slot_kinds=tuple(kinds),
        excluded=tuple(slot_exclusions),
    )
    index_bytes = (json.dumps(index.to_json(), indent=1, sort_keys=True) + "\n").encode("utf-8")

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
        # The code that turned the corpus into these rows, pinned beside the corpus it was
        # turned from. `data_snapshot_hash` above covers the corpus and `remap_hash` the
        # vocabulary; between them sits `qd_data`, and until this line nothing covered it.
        # Taken here rather than passed in: the fingerprint must describe the modules that
        # are LOADED in the process doing the writing, which is the only thing that
        # actually shaped `sequences`.
        code_fingerprint=code_fingerprint(),
        # Pins WHICH REVISION the corpus was read at. Empty is honest: a caller that does
        # not know its rev must not put a guess here, because a wrong rev in the header is
        # worse than an absent one -- absent reads NotRun, wrong reads passed=True.
        corpus_rev=corpus_rev,
        sequence_index_hash=hashlib.sha256(index_bytes).hexdigest(),
        # Empty for refuse-any, so a default set's header and hash are what they were.
        span_collapse_policy=(
            "" if span_collapse_policy == SPAN_COLLAPSE_REFUSE_ANY else span_collapse_policy
        ),
        report_only=report_only,
        # Empty unless the caller's train rows passed through an exclusion list
        # (qd_train.exclusions), so every set written without one is what it was.
        exclusions_sha256=exclusions_sha256,
        # None unless the caller derived v5 contrast rows (qd_train.contrast) into `rows`.
        contrast_rows=contrast_rows,
        # The layout every sequence above was rendered in: `training_texts` renders through
        # qd_data.render, so it is that module's format, read off it rather than restated.
        prompt_format=PROMPT_FORMAT,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Before the header, which pins its hash: a header naming an index that is not yet on
    # disk is refused by the reader, never trusted.
    _write_bytes_atomically(out_dir / SEQUENCE_INDEX_NAME, index_bytes)
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
    (out_dir / SPAN_CHECK_NAME).write_text(
        json.dumps(
            _span_check(n_span_sequences, decode=decode).to_json(), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    # Written whether or not it found anything, and carrying `allowed` so a set that was
    # written over the refusal says so in its own artifacts. A clean report and a missing
    # file are different states, and only one of them means the check ran.
    (out_dir / CONTRADICTION_NAME).write_text(
        json.dumps(
            {**contradictions, "allowed": bool(allow_contradictions)}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    # The table these ids were renumbered by, beside the ids. See `REMAP_NAME`: the header
    # pins its hash, and until this line the artifact that hash names was not kept by
    # anything. `RemapTable.write` re-derives nothing -- it stores the two tables and a
    # sidecar -- and `RemapTable.read` recomputes the hash on load rather than trusting it.
    written_remap_hash = remap.write(out_dir / REMAP_NAME)
    if written_remap_hash != header.remap_hash:  # pragma: no cover - both come from `remap`
        raise ShardContractViolation(
            f"{out_dir}: the remap written beside the shards hashes to "
            f"{written_remap_hash!r} but the header pins {header.remap_hash!r}. The set "
            "would describe a vocabulary it was not written under."
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
            f"{len(excluded)} of {n_rows_in} row(s) could not be turned into a sequence -- "
            "either a gold no Batch can express (UnencodableGold, "
            "GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL) or a row render refused outright, over "
            "a cap or empty. Each exclusion below names which. First: "
            + "; ".join(excluded[:3])
        ),
    )


def _span_check(n_span_sequences: int, *, decode: Decode | None) -> TriState:
    """Whether this set's span mapping was checked against decoded text, and over how many.

    Three answers, never two. A set with no span rows is ``NotRun`` rather than a pass:
    0 of 0 verified says nothing, and ``artifacts.padding_waste`` refuses an empty input on
    exactly the same ground. A set written without ``decode`` is ``NotRun`` because that
    check did not execute -- ``NotRun`` has no ``passed`` field to be misread as one.
    """
    if n_span_sequences == 0:
        return NotRun(
            reason=(
                "this shard set holds no SLOT_SPAN sequences, so there was no line-to-token "
                "mapping to verify. 0 of 0 verified is not a pass."
            )
        )
    if decode is None:
        return NotRun(
            reason=(
                f"write_shards was called without decode=, so the {n_span_sequences} span "
                "sequence(s) here had their gold and candidate positions checked only "
                "against the offset mapping they were derived from. That circle is "
                "GAP-S4-GOLD-ON-CANDIDATE-CANNOT-CATCH-LINE-SHIFT: a mapping that is "
                "consistently wrong moves gold and candidates together and every other "
                "check stays green. Supplying decode= is what "
                "GAP-SPAN-HEAD-LINE-MAPPING-BPE-UNVERIFIED asks for."
            )
        )
    return Ran(
        passed=True,
        value=n_span_sequences,
        n=n_span_sequences,
        n_total=n_span_sequences,
        detail=(
            "every span sequence's gold positions and full candidate set were checked "
            "against tokenizer.decode: each checked token decodes to the characters its "
            "offsets claim, the ids round-trip to the text the spans were measured "
            "against, and every recorded candidate is a line start of that decoded text"
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
    previous call.

    Measured on 2026-09-20, on the corpus ``test_shards.py`` builds: identical batch
    contents asked twice in one process, asked again after an intervening epoch, asked from
    a second process, and asked from a third under ``PYTHONHASHSEED=1234`` -- one digest,
    ``0e5b7bd2864f72d6``, for all four. The property holds, and it is a numpy-and-integers
    property, so it holds on any device.

    **What it does not say.** S5's resume reconstructs a position from ``(epoch, index)``,
    and that is sound only if the *order* is already pinned by something else -- because
    ``batch_tokens`` is one of this function's arguments, not a consequence of the other
    two. The same corpus at ``batch_tokens=4w`` and ``4w+1`` yields 17 batches either way,
    numbered 0..16, sharing not one batch: ``011d15855b660101`` against
    ``0e5b7bd2864f72d6``. An index is a position *inside* an order. Which order was eaten is
    ``run_control.Checkpoint.consumed_digest``'s question, and before that field existed the
    trainer accepted the wrong answer to it.
    """

    def __init__(
        self,
        root: Path,
        *,
        config: DataConfig,
        repo_root: Path,
        expect_rev: str | None = None,
        allow_rev_mismatch: bool = False,
    ) -> None:
        self.root = Path(root)
        self._config = config
        self._repo_root = Path(repo_root)

        raw = json.loads((self.root / HEADER_NAME).read_text(encoding="utf-8"))
        # from_json recomputes shard_hash and refuses a header edited after it was written.
        self.header: ShardHeader = ShardHeader.from_json(raw)
        self.checks: dict[str, TriState] = assert_shard_trainable(
            self.header,
            config=config,
            path=self.root,
            repo_root=self._repo_root,
            expect_rev=expect_rev,
            allow_rev_mismatch=allow_rev_mismatch,
        )
        # After the rule-3 door, so a held-out set is refused as one whatever its format.
        self.header.require_prompt_format(where=str(self.root))

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
        self.sequence_index: SequenceIndex | None = None
        self.slot_coverage: TriState = self._load_sequence_index()

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

        span_check_path = self.root / SPAN_CHECK_NAME
        if span_check_path.exists():
            self.span_check: TriState = parse_tristate(
                json.loads(span_check_path.read_text(encoding="utf-8")),
                field=f"{span_check_path}",
            )
        else:
            # Absent on every shard set written before the file existed, and on any written
            # by something else. Unknown, never verified -- the same rule coverage follows,
            # and the reason the file was added at all.
            self.span_check = NotRun(
                reason=(
                    f"{span_check_path} is absent, so whether this set's span mapping was "
                    "ever checked against decoded text was not recorded. Unverified and "
                    "unrecorded read the same here only because the writer said nothing; "
                    "not_run is the honest reading of silence."
                )
            )

        self.remap: RemapTable | None = None
        self.checks["shard_remap_matches_header"] = self._load_remap()

    def _load_remap(self) -> TriState:
        """Read the remap beside the shards and check it against the header's ``remap_hash``.

        Sets ``self.remap`` and returns the check. Three answers, never two:

        * ``Ran(passed=True)`` -- the table is there and hashes to what the header pins, so
          the ids in ``tokens.u32`` can be turned back into text and the provenance the
          header claims is a fact about a file rather than a claim about a memory.
        * ``Ran(passed=False)`` -- the table is there and hashes to something else. The set
          and its vocabulary disagree; every id in it means a different token than the
          header says. Refused rather than recorded, below.
        * ``NotRun`` -- absent. Every shard set written before ``REMAP_NAME`` existed is in
          this state, and so is any set written by something else, so an absent table is
          not a contract violation. It is *unknown* provenance, and it must not read the
          same as provenance that was checked: ``shard_provenance_pinned`` reports the
          header's ``remap_hash`` either way, which is exactly the trap this answers.
        """
        stem = self.root / REMAP_NAME
        if not (stem.with_suffix(".npz").exists() and stem.with_suffix(".json").exists()):
            return NotRun(
                reason=(
                    f"{stem}.npz/.json is absent, so the vocabulary this set's ids were "
                    f"renumbered under is not available and the header's "
                    f"remap_hash={self.header.remap_hash[:16]}… names no artifact here. "
                    "Re-deriving it means re-tokenizing the corpus with the same tokenizer."
                )
            )
        # `RemapTable.read` recomputes the hash from the tables and refuses a sidecar that
        # disagrees, so a tampered file raises here rather than returning a passing check.
        remap = RemapTable.read(stem)
        found = remap.remap_hash()
        if found != self.header.remap_hash:
            raise ShardContractViolation(
                f"{stem}: the remap beside these shards hashes to {found!r} but "
                f"{self.root / HEADER_NAME} pins {self.header.remap_hash!r}. Every id in "
                "this set would mean a different token than the header says it does."
            )
        if remap.vocab_size != self.header.vocab_size:
            raise ShardContractViolation(
                f"{stem}: the remap keeps {remap.vocab_size} tokens but the header declares "
                f"vocab_size={self.header.vocab_size}"
            )
        self.remap = remap
        return Ran(
            passed=True,
            value=found,
            n=remap.vocab_size,
            n_total=remap.source_vocab_size,
            detail=(
                f"{stem}.npz re-read and re-hashed; matches the header. "
                f"tokenizer_hash={remap.tokenizer_hash[:16]}…"
            ),
        )

    def _load_sequence_index(self) -> TriState:
        """Read :data:`SEQUENCE_INDEX_NAME`, check it against the header and the supervision.

        Sets ``self.sequence_index`` and returns slot coverage: sequences written against
        ``(row, slot)`` pairs offered. ``NotRun`` for a set whose header pins no index --
        every set written before the index existed -- so "unrecorded" never reads as
        "nothing excluded". A pinned index that is absent, re-hashes differently, or
        disagrees with the supervision in length or slot kind is refused outright: a label
        paired to the wrong sequence trains silently.
        """
        pinned = self.header.sequence_index_hash
        path = self.root / SEQUENCE_INDEX_NAME
        if not pinned:
            return NotRun(
                reason=(
                    f"{self.root / HEADER_NAME} pins no sequence_index_hash, so which row and "
                    "slot each sequence is, and which slots were excluded, was not recorded "
                    "by the writer. Pair labels by re-deriving them, or rewrite the set."
                )
            )
        if not path.exists():
            raise ShardContractViolation(
                f"{path} is absent but the header pins sequence_index_hash={pinned[:16]}…"
            )
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != pinned:
            raise ShardContractViolation(
                f"{path} hashes to {found!r} but the header pins {pinned!r}; the index was "
                "modified after it was written, or belongs to another set"
            )
        index = SequenceIndex.from_json(json.loads(raw), where=str(path))
        n = self.header.n_sequences
        if len(index.sequences) != n:
            raise ShardContractViolation(
                f"{path}: {len(index.sequences)} indexed sequence(s) but the header declares {n}"
            )
        mismatch = np.flatnonzero(np.asarray(index.slot_kinds, dtype=np.int64) != self._slot_kind)
        if mismatch.size:
            i = int(mismatch[0])
            raise ShardContractViolation(
                f"{path}: sequence {i} is indexed as slot kind {index.slot_kinds[i]} but "
                f"supervision.npz holds {int(self._slot_kind[i])}; the index and the shards "
                "describe different orders"
            )
        self.sequence_index = index
        n_total = n + len(index.excluded)
        by_scope = {
            scope: sum(1 for e in index.excluded if e.scope == scope) for scope in ("row", "slot")
        }
        return Ran(
            passed=not index.excluded,
            value=n,
            n=n,
            n_total=n_total,
            detail=(
                f"{n} of {n_total} (row, slot) sequence(s) written; excluded "
                f"{by_scope['slot']} slot(s) on their own account and {by_scope['row']} "
                f"slot(s) of rows refused whole, over {index.rows_in} row(s) offered"
            ),
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
        if batch_tokens > MAX_POSITIONS_PER_BATCH:
            raise ValueError(
                f"batch_tokens={batch_tokens} exceeds MAX_POSITIONS_PER_BATCH="
                f"{MAX_POSITIONS_PER_BATCH}. A batch's memory cost is rows x width, and "
                f"batch_tokens is the only ceiling on that product -- MAX_ROWS_PER_BATCH="
                f"{MAX_ROWS_PER_BATCH} bounds the rows, not the positions. This refusal is "
                "a backstop against a mis-typed value, not a statement about what fits: for "
                "the bound that describes an actual device, run tools/memory_budget.py, "
                "which prints qd_train.memory.max_positions_that_fit per bucket."
            )

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
            yield assemble_batch(
                [self.sequence(i) for i in plan.rows],
                kinds=self._slot_kind[rows],
                target_index=self._target_index[rows],
                spans=self._span_target[rows],
                candidates=[self.candidates(i) for i in plan.rows],
                width=plan.width, bucket=plan.bucket, index=index,
            )

    def to_json(self) -> dict[str, Any]:
        """What a ledger row records about this shard set."""
        return {
            "root": str(self.root),
            "header": self.header.to_json(),
            "checks": {k: v.to_json() for k, v in sorted(self.checks.items())},
            "coverage": self.coverage.to_json(),
            "slot_coverage": self.slot_coverage.to_json(),
            "span_check": self.span_check.to_json(),
            "padding_waste": self.padding_waste().to_json(),
        }
