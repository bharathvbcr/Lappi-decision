"""The two artifacts S2 and S4 produce and the trainer consumes — as code, not prose.

S2 writes a [`RemapTable`]; S4 writes token shards described by a [`ShardHeader`]; the
trainer reads both. Three lanes, two shared formats.

**This module exists because a prose contract between lanes has already failed twice in this
repo, on the same day.** `GAP-RT-WIRE-CONTEXT-ENCODING`: `qd_data` emitted `context_b64`
while `qd-runtime` accepted only a JSON integer array, and no caller could satisfy both even
though both lanes' suites were green. `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`: one name, two
quantities, again both suites green. In each case the contract was a paragraph, each lane
read it, and each lane implemented something the other could not consume.

So the format here is *executable*. A lane does not implement the shard header from a
description; it imports `ShardHeader` and is bound by its `__post_init__`. There is nothing
left to interpret differently.

## The invariants, and what each is defending against

**`packed` is always `False`.** `docs/plan-corrections.md` SAFETY-2: sequence packing is
unsupported for the hybrid. A GDN layer carries recurrent state *along* the sequence, so two
examples concatenated into one row bleed state across the boundary, and no attention mask
expresses "reset the recurrence here". The damage is silent — loss still falls. Constructing
a header with `packed=True` raises.

**Every token in a shard must survive the remap.** S2 drops ~168K of 248,320 rows; if S4
tokenizes with the full vocabulary and a dropped id reaches the trainer, the embedding lookup
is out of bounds at best and points at an unrelated row at worst. [`RemapTable.encode`]
**raises** on a dropped id and names it. It never substitutes an `UNK`, because a substituted
token is a training example quietly teaching the wrong thing.

**A shard carries its own provenance.** `qd_train.data_access.open_training_data` is the one
door onto training data and it refuses held-out input — but it guards *manifests*. A shard is
derived from a manifest and read by the trainer directly, so without provenance the trainer
would read held-out data through an artifact that never passed the check. Rule 3 would be
satisfied on paper and bypassed by indirection. [`assert_shard_trainable`] is the same check
at the new boundary, and a header without `data_snapshot_hash` cannot be built.

**One bucketing function.** [`bucket_for`] is used by the writer *and* the sampler. Two
implementations of "which bucket is this length in" that disagree at a boundary produce a
padding-waste figure measured against a bucketing nobody trains with.

**One line rule.** [`line_start_indices`] answers "where does a line begin", and a span
slot's pointer head ranges over exactly those positions — so two implementations that
disagree relabel every span by a constant, which `docs/hardening.md` §1 records as
systematic and invisible in accuracy. There were two, both in this package and both called
`line_starts`: S4's over characters and rung 0's over bytes
(`qd_train.byte_context.line_starts`). They disagreed on the empty context — one line
against none — and both suites were green, because the five vectors each was pinned to are
the Rust suite's own and not one of them is empty. The rule now lives here; the byte path
keeps its own offsets because its model's positions *are* bytes, and
`test_artifacts.py::test_the_byte_path_line_rule_is_this_one` holds the two together.
`GAP-S4-LINE-STARTS-SECOND-IMPLEMENTATION`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Self

import numpy as np

from qd_data.config import SPLITS, DataConfig

from .data_access import assert_path_not_held_out
from .tristate import NotRun, Ran, TriState

__all__ = [
    "DROPPED",
    "NO_SPAN",
    "REMAP_FORMAT",
    "SHARD_FORMAT",
    "SLOT_CHOICE",
    "SLOT_LM",
    "SLOT_SCORE",
    "SLOT_SPAN",
    "SPAN_ABSTAIN",
    "TOKEN_DTYPE",
    "Batch",
    "RemapTable",
    "ShardContractViolation",
    "ShardHeader",
    "TokenNotInRemap",
    "assert_remap_covers",
    "assert_shard_trainable",
    "assign_buckets",
    "bucket_for",
    "line_start_indices",
    "padding_waste",
]

REMAP_FORMAT: Final[str] = "qd-remap-v1"
SHARD_FORMAT: Final[str] = "qd-shard-v1"

#: `old_to_new[i]` for a token the remap dropped. Not a sentinel to be tested for and
#: replaced — a value whose presence in real data is an error.
DROPPED: Final[int] = -1

#: Token ids are stored as `uint32`. The source vocabulary is 248,320, so `uint16` cannot
#: hold a pre-remap id, and `int64` doubles every shard on disk for no reachable value.
TOKEN_DTYPE: Final[str] = "uint32"

#: A shard's padding waste gate. `docs/...` S4: "Padding waste <= 15% measured over one epoch
#: of the mixture." Read-only here; rule 2 puts moving it out of an agent's reach.
MAX_PADDING_WASTE: Final[float] = 0.15


class TokenNotInRemap(Exception):
    """A token id reached the remap that the remap dropped. Never recoverable by substitution."""


class ShardContractViolation(Exception):
    """A shard artifact violates the format this module defines."""


def _sha256_hex(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
        h.update(b"\x00")
    return h.hexdigest()


# --- S2's artifact ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RemapTable:
    """The 248,320 -> ~80K vocabulary remap.

    `old_to_new` is `int32[source_vocab_size]`, carrying [`DROPPED`] for a token the remap
    does not keep. `new_to_old` is `int32[vocab_size]`. The two are inverse on the kept set,
    and `__post_init__` checks it rather than trusting the producer: a remap whose tables
    disagree relabels tokens silently, and the loss curve looks fine.
    """

    old_to_new: np.ndarray
    new_to_old: np.ndarray
    tokenizer_hash: str
    special_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.old_to_new.dtype != np.int32 or self.new_to_old.dtype != np.int32:
            raise ShardContractViolation(
                f"remap tables must be int32; got old_to_new={self.old_to_new.dtype}, "
                f"new_to_old={self.new_to_old.dtype}"
            )
        if self.old_to_new.ndim != 1 or self.new_to_old.ndim != 1:
            raise ShardContractViolation("remap tables must be 1-D")
        if not self.tokenizer_hash:
            raise ShardContractViolation(
                "tokenizer_hash is required: a remap is only meaningful against the tokenizer "
                "whose ids it renumbers, and a swapped tokenizer maps wrong ids silently"
            )

        kept = np.flatnonzero(self.old_to_new >= 0)
        if kept.size != self.new_to_old.size:
            raise ShardContractViolation(
                f"{kept.size} kept entries in old_to_new but new_to_old has "
                f"{self.new_to_old.size}; the tables describe different vocabularies"
            )
        if self.new_to_old.size == 0:
            raise ShardContractViolation("a remap that keeps no tokens is not a remap")

        # Inverse on the kept set, both directions. Checked with array ops rather than a loop
        # because this runs over 248,320 entries on every load.
        identity = np.arange(self.new_to_old.size, dtype=np.int32)
        if not np.array_equal(self.old_to_new[self.new_to_old], identity):
            bad = int(np.flatnonzero(self.old_to_new[self.new_to_old] != identity)[0])
            raise ShardContractViolation(
                f"tables are not inverse: new id {bad} maps to old id "
                f"{int(self.new_to_old[bad])}, which maps back to "
                f"{int(self.old_to_new[self.new_to_old[bad]])}"
            )
        if not np.array_equal(self.new_to_old[self.old_to_new[kept]], kept.astype(np.int32)):
            raise ShardContractViolation("tables are not inverse in the old -> new direction")

        missing = [t for t in self.special_ids if not self._kept(t)]
        if missing:
            raise ShardContractViolation(
                f"the remap dropped special token id(s) {missing}. A dropped BOS/EOS/pad is not "
                "a vocabulary saving; it is a model that cannot express the end of a sequence"
            )

    def _kept(self, old_id: int) -> bool:
        return 0 <= old_id < self.old_to_new.size and bool(self.old_to_new[old_id] >= 0)

    @property
    def source_vocab_size(self) -> int:
        return int(self.old_to_new.size)

    @property
    def vocab_size(self) -> int:
        """Rows in the remapped `lm_head`. This is the number a training step pays for."""
        return int(self.new_to_old.size)

    def remap_hash(self) -> str:
        """Identity of this remap. Pinned in the shard header and in the ledger row.

        Over `new_to_old` rather than both tables: the two are inverse, so `new_to_old` plus
        the source size determines `old_to_new` exactly, and hashing both would let a
        corrupted `old_to_new` produce a *different* hash for the same vocabulary.
        """
        return _sha256_hex(
            REMAP_FORMAT.encode(),
            self.tokenizer_hash.encode(),
            str(self.source_vocab_size).encode(),
            self.new_to_old.tobytes(),
        )

    def encode(self, old_ids: np.ndarray) -> np.ndarray:
        """Old ids -> new ids. **Raises** on a dropped id; never substitutes.

        The temptation is an `UNK` fallback. A fallback turns a coverage bug into a corpus
        where some fraction of tokens silently mean "something was here", which trains a model
        to predict that and shows up as nothing worse than slightly worse loss.
        """
        ids = np.asarray(old_ids)
        if ids.size == 0:
            return np.empty(0, dtype=np.int32)
        if ids.min() < 0 or ids.max() >= self.source_vocab_size:
            bad = int(ids.max()) if ids.max() >= self.source_vocab_size else int(ids.min())
            raise TokenNotInRemap(
                f"token id {bad} is outside the source vocabulary [0, {self.source_vocab_size})"
            )
        mapped = self.old_to_new[ids]
        dropped = np.flatnonzero(mapped < 0)
        if dropped.size:
            first = int(dropped[0])
            raise TokenNotInRemap(
                f"{dropped.size} of {ids.size} token(s) were dropped by the remap; the first is "
                f"old id {int(ids.flat[first])} at position {first}. The remap and the corpus "
                "disagree — rebuild the remap over this corpus rather than substituting a token."
            )
        return mapped.astype(np.int32, copy=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": REMAP_FORMAT,
            "tokenizer_hash": self.tokenizer_hash,
            "source_vocab_size": self.source_vocab_size,
            "vocab_size": self.vocab_size,
            "special_ids": list(self.special_ids),
            "remap_hash": self.remap_hash(),
        }

    def write(self, path: Path) -> str:
        """Write `<path>.npz` plus a `<path>.json` sidecar. Returns the remap hash."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path.with_suffix(".npz"),
            old_to_new=self.old_to_new,
            new_to_old=self.new_to_old,
        )
        path.with_suffix(".json").write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.remap_hash()

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read and re-verify. The sidecar's `remap_hash` is recomputed, never trusted."""
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("format") != REMAP_FORMAT:
            raise ShardContractViolation(
                f"{path}: format is {meta.get('format')!r}, expected {REMAP_FORMAT!r}"
            )
        with np.load(path.with_suffix(".npz")) as z:
            table = cls(
                old_to_new=z["old_to_new"].astype(np.int32, copy=False),
                new_to_old=z["new_to_old"].astype(np.int32, copy=False),
                tokenizer_hash=meta["tokenizer_hash"],
                special_ids=tuple(int(t) for t in meta.get("special_ids", ())),
            )
        recomputed = table.remap_hash()
        if recomputed != meta.get("remap_hash"):
            raise ShardContractViolation(
                f"{path}: remap_hash on disk is {meta.get('remap_hash')!r} but the tables hash "
                f"to {recomputed!r}. The artifact was modified after it was written."
            )
        return table


# --- S4's artifact ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShardHeader:
    """What a set of token shards is, and what it was built from.

    Every hash here is a pin the trainer checks before the first step. A shard set whose
    `remap_hash` does not match the remap actually loaded is not a slow run, it is a run whose
    every token id means something else.
    """

    split: str
    data_snapshot_hash: str
    tokenizer_hash: str
    remap_hash: str
    vocab_size: int
    n_sequences: int
    total_tokens: int
    max_seq_len: int
    buckets: tuple[int, ...]
    packed: bool = False
    dtype: str = TOKEN_DTYPE
    format: str = SHARD_FORMAT
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.format != SHARD_FORMAT:
            raise ShardContractViolation(f"format {self.format!r}, expected {SHARD_FORMAT!r}")
        if self.packed:
            raise ShardContractViolation(
                "packed=True is refused. docs/plan-corrections.md SAFETY-2: packing is "
                "unsupported for the hybrid — a GDN layer carries recurrent state along the "
                "sequence, so two examples in one row bleed state across the boundary and no "
                "attention mask expresses a reset. The failure is silent; loss still falls."
            )
        if self.split not in SPLITS:
            raise ShardContractViolation(f"split {self.split!r} not one of {SPLITS}")
        if self.dtype != TOKEN_DTYPE:
            raise ShardContractViolation(
                f"dtype {self.dtype!r}: token ids are stored as {TOKEN_DTYPE}. uint16 cannot "
                "hold a pre-remap id from a 248,320-token vocabulary"
            )
        for name in ("data_snapshot_hash", "tokenizer_hash", "remap_hash"):
            if not getattr(self, name):
                raise ShardContractViolation(
                    f"{name} is required. Without it the trainer cannot tell whether these "
                    "shards were built from the data, tokenizer and remap it is about to use"
                )
        for name in ("vocab_size", "n_sequences", "total_tokens", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ShardContractViolation(f"{name} must be positive, got {getattr(self, name)}")
        if not self.buckets:
            raise ShardContractViolation("at least one bucket boundary is required")
        if list(self.buckets) != sorted(set(self.buckets)):
            raise ShardContractViolation(
                f"buckets must be strictly increasing and unique, got {self.buckets}"
            )
        if self.buckets[-1] < self.max_seq_len:
            raise ShardContractViolation(
                f"largest bucket {self.buckets[-1]} is below max_seq_len {self.max_seq_len}: "
                "a sequence that fits no bucket would be dropped or truncated silently"
            )
        if self.total_tokens < self.n_sequences:
            raise ShardContractViolation(
                f"total_tokens {self.total_tokens} < n_sequences {self.n_sequences}: at least "
                "one sequence is empty"
            )

    def shard_hash(self) -> str:
        return _sha256_hex(
            SHARD_FORMAT.encode(),
            self.split.encode(),
            self.data_snapshot_hash.encode(),
            self.tokenizer_hash.encode(),
            self.remap_hash.encode(),
            json.dumps(
                {
                    "vocab_size": self.vocab_size,
                    "n_sequences": self.n_sequences,
                    "total_tokens": self.total_tokens,
                    "max_seq_len": self.max_seq_len,
                    "buckets": list(self.buckets),
                },
                sort_keys=True,
            ).encode(),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "split": self.split,
            "data_snapshot_hash": self.data_snapshot_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "remap_hash": self.remap_hash,
            "vocab_size": self.vocab_size,
            "n_sequences": self.n_sequences,
            "total_tokens": self.total_tokens,
            "max_seq_len": self.max_seq_len,
            "buckets": list(self.buckets),
            "packed": self.packed,
            "dtype": self.dtype,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
            "shard_hash": self.shard_hash(),
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        header = cls(
            split=raw["split"],
            data_snapshot_hash=raw["data_snapshot_hash"],
            tokenizer_hash=raw["tokenizer_hash"],
            remap_hash=raw["remap_hash"],
            vocab_size=int(raw["vocab_size"]),
            n_sequences=int(raw["n_sequences"]),
            total_tokens=int(raw["total_tokens"]),
            max_seq_len=int(raw["max_seq_len"]),
            buckets=tuple(int(b) for b in raw["buckets"]),
            packed=bool(raw.get("packed", False)),
            dtype=raw.get("dtype", TOKEN_DTYPE),
            format=raw.get("format", SHARD_FORMAT),
            created_at=raw.get("created_at", ""),
        )
        if "shard_hash" in raw and raw["shard_hash"] != header.shard_hash():
            raise ShardContractViolation(
                f"shard_hash on disk is {raw['shard_hash']!r} but the header hashes to "
                f"{header.shard_hash()!r}; the artifact was modified after it was written"
            )
        return header


#: Supervision kinds. `SLOT_LM` is CPT — next-token over the whole sequence, no slot.
#:
#: It is a *known* kind, so a batch carrying it earns a named refusal rather than the
#: generic "unknown slot kind" one, but it may **not** appear in `Batch.slot_kind`: a batch
#: carrying the slot channel is an FT batch, and an LM row inside one has no gold letter for
#: its `target_index` to name. `Batch._check_supervision` refuses it and says why;
#: `GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED`.
SLOT_LM: Final[int] = 0
SLOT_CHOICE: Final[int] = 1
SLOT_SCORE: Final[int] = 2
SLOT_SPAN: Final[int] = 3
_SLOT_KINDS: Final[frozenset[int]] = frozenset({SLOT_LM, SLOT_CHOICE, SLOT_SCORE, SLOT_SPAN})

#: Real tokens a `Batch` row must carry. Two, not one: every objective in this contract
#: supervises a position by its **next** token, so a one-token row has no supervised
#: position at all. See `Batch`'s docstring and
#: `GAP-TRAINER-CPT-REFUSES-LENGTH-ONE-ROWS`. Written as a name rather than a literal `2`
#: because `qd_train.shards._tokenize_checked` states the same floor at write time and the
#: two are one decision.
_MIN_ROW_TOKENS: Final[int] = 2

#: `span_target` for a row that is not a span slot. Not a position.
NO_SPAN: Final[int] = -1

#: `span_target` for a span row whose gold answer is **abstain** — distinct from `NO_SPAN`,
#: which means "this row is not a span at all".
#:
#: The runtime can already express this: `qd-runtime/src/answer.rs:64` sizes the pointer head
#: at `context.line_count() + RESERVED_NOUL_ROWS`, and that extra row is the abstention. Until
#: this sentinel existed the training side had no counterpart, so an unanswerable span could
#: not be encoded at all — and because `qa.answerability` is one of the two held-out families
#: (`qd_data.config.DEFAULT_HELD_OUT_FAMILIES`), the span slot is the *only* place a model
#: would learn to abstain on this corpus. The runtime could say "no evidence"; training could
#: never teach it to. `GAP-S4-ABSTAINING-SPAN-HAS-NO-ENCODING`.
SPAN_ABSTAIN: Final[int] = -2


@dataclass(frozen=True, slots=True)
class Batch:
    """One padded batch: what S4's reader yields and the trainer consumes.

    Defined here rather than in either lane because it is the seam between them, and a seam
    described in two prompts is a seam two lanes implement differently.

    `tokens` is `int32[B, L]` already remapped — new ids, not source ids. `lengths` is the
    real token count per row, so the trainer masks loss rather than training on padding.
    `index` is the batch's deterministic position in the epoch, which is what makes S5's
    bit-exact resume possible: the order is a pure function of `(seed, epoch)`, never of
    hidden iterator state.

    # Supervision, and why a letter is not enough

    The first version of this type carried only `tokens` and `lengths`, on the assumption that
    a training example is a sequence whose final token *is* the answer. That holds for `choice`
    and `score`, which decode to a letter. **It does not hold for `span`.**

    `qd_data.render._slot_suffix` gives a `SpanSlot` an empty letter alphabet on purpose — a
    span is a *position*, decoded by a pointer head over line-start tokens, not a vocabulary
    token. The only letter in its option set is `noul`. So under the original type the sole
    gold token available to a span slot was the abstain letter, and **the span head could only
    ever have been trained to abstain** — silently, with a loss curve that looked fine, on the
    slot the plan's headline example (`evidence`) depends on. Found by the S4 lane, which
    escalated it rather than routing around it; `GAP-S4-SPAN-GOLD-HAS-NO-BATCH-CHANNEL`.

    The three supervision fields are all-or-nothing:

    * **CPT**: all three `None`. The objective is next-token over the sequence.
    * **FT**: `slot_kind` and `target_index` required. `span_target` required exactly when some
      row is `SLOT_SPAN`, and `NO_SPAN` on every row that is not.

    There is no third case, and the two settlements below both follow from that.

    `target_index[i]` is the position whose **next** token is the gold letter, so it is
    strictly less than `lengths[i] - 1`. For a span row it identifies the query position the
    pointer head reads from; the gold positions are in `span_target[i]`.

    # Two rows this type used to permit and the trainer then refused

    A constructor that accepts what its only consumer rejects is a contract with two
    readings, and this type exists precisely so there is one. Both of these were opened as
    gaps by the trainer lane against *this* file, and both are settled here rather than in a
    comment on the loop:

    **`SLOT_LM` cannot appear in `slot_kind`** — `GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED`.
    `SLOT_LM` is the CPT kind, *"next-token over the whole sequence, no slot"*, and the CPT
    case above is all three fields `None`. So a batch carrying `slot_kind` is an FT batch by
    the taxonomy's own terms, and an LM row inside one has no gold letter — its
    `target_index`, defined as *"the position whose next token is the gold letter"*, names
    nothing. Two readings were available (an LM row deliberately mixed into fine-tuning,
    versus a mislabelled choice/score row) and **both leave the loss curve looking fine**.
    The deciding evidence is that this contract has no don't-care sentinel for
    `target_index`: `span_target` got `NO_SPAN` the moment a row needed to say "not me", and
    the absence of the equivalent here is what says an LM row was never meant to be
    expressible. `qd_train.shards.slot_kind_of` agrees by construction — it returns
    `SLOT_CHOICE`, `SLOT_SCORE` or `SLOT_SPAN` and raises on anything else, so nothing in
    the repo ever writes one.

    **A row must carry at least two real tokens** —
    `GAP-TRAINER-CPT-REFUSES-LENGTH-ONE-ROWS`. Under CPT, position `p` is supervised only
    when `p < lengths - 1`; under FT, `target_index` must lie in `[0, lengths - 1)`, an empty
    range at `lengths == 1`. So a one-token row supervises nothing under *either* objective
    while still being loaded, padded and paid for — the same shape of error as training on
    padding, in the opposite direction. FT already refused it as a side effect of the
    `target_index` range check, and `qd_train.shards._tokenize_checked` refuses it at write
    time (`ids.size < 2`), where the message can name the source row; only this constructor
    permitted it, and only for CPT. Stating it once here makes the constructor and both
    consumers agree.
    """

    tokens: np.ndarray
    lengths: np.ndarray
    bucket: int
    index: int
    slot_kind: np.ndarray | None = None
    target_index: np.ndarray | None = None
    span_target: np.ndarray | None = None
    line_starts: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ShardContractViolation(f"tokens must be 2-D [B, L], got {self.tokens.shape}")
        if self.tokens.dtype != np.int32:
            raise ShardContractViolation(
                f"tokens must be int32 (post-remap ids), got {self.tokens.dtype}"
            )
        if self.lengths.ndim != 1 or self.lengths.size != self.tokens.shape[0]:
            raise ShardContractViolation(
                f"lengths must be 1-D with one entry per row: {self.lengths.shape} vs "
                f"{self.tokens.shape}"
            )
        if self.lengths.size == 0:
            raise ShardContractViolation("an empty batch is not a batch")
        if int(self.lengths.min()) <= 0:
            raise ShardContractViolation("every row must carry at least one real token")
        if int(self.lengths.min()) < _MIN_ROW_TOKENS:
            short = np.flatnonzero(self.lengths < _MIN_ROW_TOKENS)
            raise ShardContractViolation(
                f"row(s) {short.tolist()[:16]} carry "
                f"{self.lengths[short].tolist()[:16]} real token(s); every row needs at least "
                f"{_MIN_ROW_TOKENS}. CPT supervises position p only when p < lengths-1, and "
                "FT's target_index must lie in [0, lengths-1) -- an empty range here -- so a "
                "one-token row supervises nothing under either objective while still being "
                "loaded, padded and paid for. qd_train.shards._tokenize_checked refuses these "
                "at write time, where the message can name the source row"
            )
        if int(self.lengths.max()) > self.tokens.shape[1]:
            raise ShardContractViolation(
                f"length {int(self.lengths.max())} exceeds the padded width "
                f"{self.tokens.shape[1]}: a row claims more tokens than it has"
            )
        if self.index < 0:
            raise ShardContractViolation(f"index must be non-negative, got {self.index}")
        self._check_supervision()

    def _check_supervision(self) -> None:
        present = [f is not None for f in (self.slot_kind, self.target_index)]
        if not any(present):
            if self.span_target is not None:
                raise ShardContractViolation(
                    "span_target without slot_kind/target_index: nothing says which rows are "
                    "spans, so the pointer loss would be applied to whatever happened to be there"
                )
            return  # CPT: next-token over the sequence, no slot supervision
        if not all(present):
            raise ShardContractViolation(
                "slot_kind and target_index are all-or-nothing: a batch carrying one without "
                "the other supervises a position whose kind is unknown"
            )

        b = self.tokens.shape[0]
        kinds = self.slot_kind
        idx = self.target_index
        assert kinds is not None and idx is not None  # narrowed by the checks above
        if kinds.shape != (b,) or idx.shape != (b,):
            raise ShardContractViolation(
                f"slot_kind {kinds.shape} and target_index {idx.shape} must both be [{b}]"
            )
        unknown = sorted({int(k) for k in kinds.tolist()} - _SLOT_KINDS)
        if unknown:
            raise ShardContractViolation(
                f"unknown slot kind(s) {unknown}; expected {sorted(_SLOT_KINDS)}"
            )
        lm_rows = np.flatnonzero(kinds == SLOT_LM)
        if lm_rows.size:
            raise ShardContractViolation(
                f"row(s) {lm_rows.tolist()[:16]} carry SLOT_LM in a batch that also carries the "
                "slot channel. SLOT_LM is the CPT kind -- next-token over the whole sequence, "
                "no slot -- and the contract's CPT case is all three supervision fields None, "
                "so a batch with slot_kind is an FT batch. Such a row has no gold letter, so "
                "its target_index names nothing: it is either an LM row meant to be mixed into "
                "fine-tuning, for which this contract has no target_index don't-care sentinel, "
                "or a mislabelled choice/score row. The two train different objectives and both "
                "leave the loss curve looking fine, so the writer says which rather than the "
                "trainer guessing. GAP-TRAINER-FT-SLOT-LM-ROW-UNDEFINED"
            )
        # The supervised position's NEXT token is the gold letter, so it cannot be the last one.
        if int(idx.min()) < 0 or bool(np.any(idx >= self.lengths - 1)):
            bad = int(np.flatnonzero((idx < 0) | (idx >= self.lengths - 1))[0])
            raise ShardContractViolation(
                f"target_index[{bad}]={int(idx[bad])} is not in [0, lengths[{bad}]-1) with "
                f"lengths[{bad}]={int(self.lengths[bad])}: the supervised position's next token "
                "is the gold, so the last position supervises nothing"
            )

        is_span = kinds == SLOT_SPAN
        if not bool(is_span.any()):
            # Checked here rather than in `_check_line_starts`, which is only reached once a
            # span row exists — putting it there made it unreachable, and the test for it
            # passed vacuously until the test itself was run.
            if self.line_starts is not None:
                raise ShardContractViolation(
                    "line_starts without any SLOT_SPAN row: the candidate set is only "
                    "meaningful for a pointer head"
                )
            if self.span_target is not None and bool(np.any(self.span_target != NO_SPAN)):
                raise ShardContractViolation(
                    "span_target carries positions but no row is SLOT_SPAN"
                )
            return
        if self.span_target is None:
            raise ShardContractViolation(
                f"{int(is_span.sum())} row(s) are SLOT_SPAN but span_target is absent. A span "
                "answer is a position, not a letter — without this channel the only gold "
                "available is the noul letter, and the span head learns to abstain always"
            )
        if self.span_target.shape != (b, 2):
            raise ShardContractViolation(
                f"span_target must be [{b}, 2] (start, end), got {self.span_target.shape}"
            )
        starts, ends = self.span_target[:, 0], self.span_target[:, 1]
        if bool(np.any(self.span_target[~is_span] != NO_SPAN)):
            raise ShardContractViolation(
                f"a non-span row carries a span position; non-span rows must be {NO_SPAN}"
            )

        # A span row either points at evidence or abstains. Half of each is a row whose gold
        # nobody can read: it would train the start head on a position and the end head on the
        # abstain row, for the same answer.
        abstaining = is_span & (starts == SPAN_ABSTAIN) & (ends == SPAN_ABSTAIN)
        half = is_span & ((starts == SPAN_ABSTAIN) != (ends == SPAN_ABSTAIN))
        if bool(half.any()):
            bad = int(np.flatnonzero(half)[0])
            raise ShardContractViolation(
                f"span_target[{bad}] is ({int(starts[bad])}, {int(ends[bad])}): a span row "
                f"abstains in both positions or neither, never one"
            )
        pointing = is_span & ~abstaining

        if bool(np.any(starts[pointing] < 0)) or bool(np.any(ends[pointing] < 0)):
            raise ShardContractViolation(
                f"a SLOT_SPAN row must carry real start/end positions, or {SPAN_ABSTAIN} "
                f"(SPAN_ABSTAIN) in both to abstain"
            )
        if bool(np.any(starts[pointing] > ends[pointing])):
            bad = int(np.flatnonzero(pointing & (starts > ends))[0])
            raise ShardContractViolation(
                f"span_target[{bad}] is ({int(starts[bad])}, {int(ends[bad])}): start is after end"
            )
        if bool(np.any(ends[pointing] >= self.lengths[pointing])):
            bad = int(np.flatnonzero(pointing & (ends >= self.lengths))[0])
            raise ShardContractViolation(
                f"span_target[{bad}] ends at {int(ends[bad])} but that row holds "
                f"{int(self.lengths[bad])} tokens: the span points past the sequence"
            )

        self._check_line_starts(is_span, pointing, starts, ends)

    def _check_line_starts(
        self,
        is_span: np.ndarray,
        pointing: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> None:
        """The pointer head's candidate set, and the invariant that makes a gold checkable.

        `docs/schema-api.md` decodes a span as a *pointer head over line-start tokens*, so the
        candidates are not "every position" — and training over every position while serving
        over line starts is a train/serve mismatch that no loss curve shows.

        The invariant worth having is the last one below: **a span gold must land on a
        line-start token.** A line→token mapping is several conversions deep (gold line → line
        start in the escaped region → character offset → token index), and a mapping that is
        wrong by a line or by an escape still yields a plausible in-range token. Requiring the
        gold to coincide with a candidate is what turns that from plausible into checked.
        """
        b = self.tokens.shape[0]
        if self.line_starts is None:
            raise ShardContractViolation(
                f"{int(is_span.sum())} row(s) are SLOT_SPAN but line_starts is absent. The "
                "pointer head ranges over line-start tokens; without the candidate set the "
                "trainer would either score every position — a train/serve mismatch against "
                "qd-runtime — or re-derive line starts beside qd_data.render, which is how the "
                "same span label comes to point at two different lines"
            )
        if self.line_starts.dtype != np.bool_:
            raise ShardContractViolation(
                f"line_starts must be a bool mask, got {self.line_starts.dtype}"
            )
        if self.line_starts.shape != self.tokens.shape:
            raise ShardContractViolation(
                f"line_starts must be [{b}, {self.tokens.shape[1]}] like tokens, got "
                f"{self.line_starts.shape}"
            )

        for i in np.flatnonzero(is_span):
            row, n = self.line_starts[i], int(self.lengths[i])
            if bool(row[n:].any()):
                raise ShardContractViolation(
                    f"line_starts[{i}] marks a candidate at or past position {n}, which is "
                    "padding: the pointer head would be free to answer with a padded position"
                )
            if not bool(row[:n].any()):
                raise ShardContractViolation(
                    f"line_starts[{i}] marks no candidate in {n} real token(s); a sequence has "
                    "at least one line, so an empty candidate set is a mapping failure"
                )

        for i in np.flatnonzero(pointing):
            row = self.line_starts[i]
            for name, pos in (("start", int(starts[i])), ("end", int(ends[i]))):
                if not bool(row[pos]):
                    raise ShardContractViolation(
                        f"span_target[{i}] {name}={pos} is not a line-start token. The gold must "
                        "coincide with a candidate the pointer head can actually select — a "
                        "line→token mapping that is off by a line still lands on a plausible "
                        "in-range token, and this is the check that separates the two"
                    )

    @property
    def padding_fraction(self) -> float:
        """Padded positions in this batch that carry no token."""
        total = self.tokens.size
        return (total - int(self.lengths.sum())) / total


# --- shared functions both the writer and the sampler must agree on -------------------------


def bucket_for(length: int, buckets: Sequence[int]) -> int:
    """Index of the smallest bucket that fits `length`.

    One implementation, imported by the writer and by the sampler. Two implementations that
    disagree at a boundary give a padding-waste number measured against a bucketing nobody
    trains with — and the disagreement is invisible because both are individually plausible.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    for i, b in enumerate(buckets):
        if length <= b:
            return i
    raise ValueError(
        f"length {length} exceeds the largest bucket {buckets[-1]}; a sequence that fits no "
        "bucket must be refused at write time, not truncated at read time"
    )


def assign_buckets(lengths: Sequence[int], buckets: Sequence[int]) -> list[int]:
    """`bucket_for` over many lengths."""
    return [bucket_for(int(n), buckets) for n in lengths]


def line_start_indices(text: str | bytes) -> tuple[int, ...]:
    """Indices at which each line begins: 0-based indices, for 1-based line numbers.

    **The one statement of "where does a line begin" on the Python side**, for the same
    reason [`bucket_for`] is the one statement of "which bucket is this length in". A span
    slot's pointer head ranges over exactly these positions, so a second implementation
    that disagrees does not fail — it relabels every span by a constant, and
    `docs/hardening.md` §1 records that as systematic and invisible in accuracy.

    The rule is `crates/qd-runtime/src/context.rs`'s, which owns it for the whole system:

    * `\\n` terminates a line;
    * a trailing `\\n` does **not** open an empty final line;
    * `\\r` is ordinary content, never a second terminator (CRLF is where byte offsets and
      line numbers diverge, so a bare CR is not allowed to silently become one);
    * and an **empty input has no lines at all** — `Context::line_count` (context.rs:167)
      special-cases it to `0`, and `answer.rs:64` sizes the pointer head at
      `line_count + RESERVED_NOUL_ROWS`, so an empty context serves the abstain row alone.

    That last rule is the one the two Python implementations disagreed on. `\\n` is one
    byte and one character, so the *grid* is the same whichever unit is counted: this
    returns character indices for a `str` and byte indices for `bytes`, and the number of
    lines is the same either way. S4 wants characters, because a tokenizer's offset
    mapping is expressed in them; rung 0's `qd_train.byte_context` wants bytes, because
    its model's positions *are* bytes.
    """
    if isinstance(text, str):
        newline: str | int = "\n"
    elif isinstance(text, bytes | bytearray):
        newline = 0x0A
    else:
        raise TypeError(
            f"line_start_indices takes str or bytes, got {type(text).__name__}. A "
            "sequence of anything else has no line rule, and coercing one here is how a "
            "caller comes to count lines in something that is not text."
        )
    if not text:
        return ()
    last = len(text) - 1
    out = [0]
    for i, ch in enumerate(text):
        if ch == newline and i != last:
            out.append(i + 1)
    return tuple(out)


def padding_waste(lengths: Sequence[int], buckets: Sequence[int]) -> TriState:
    """Fraction of padded positions that carry no token, against the S4 gate.

    Returns `NotRun` on an empty input rather than a vacuous pass: a waste figure over zero
    sequences is `0/0`, and reporting that as 0% waste would let an empty shard set clear the
    gate. Python's `all([])` being `True` is the same trap one level down.
    """
    if not lengths:
        return NotRun(
            reason="no sequences: padding waste over an empty shard set is 0/0, and reporting "
            "it as 0% would let an empty set pass the gate"
        )
    real = 0
    padded = 0
    for n in lengths:
        real += int(n)
        padded += int(buckets[bucket_for(int(n), buckets)])
    waste = (padded - real) / padded
    return Ran(
        passed=waste <= MAX_PADDING_WASTE,
        value=round(waste, 6),
        n=len(lengths),
        n_total=len(lengths),
        detail=(
            f"{padded - real} padding of {padded} positions ({waste:.2%}) over "
            f"{len(lengths)} sequences; gate is <= {MAX_PADDING_WASTE:.0%}. buckets={list(buckets)}"
        ),
    )


def assert_remap_covers(token_ids: np.ndarray, remap: RemapTable) -> None:
    """Every id in `token_ids` survives `remap`, or raise naming the first that does not.

    The cross-lane check between S2 and S4. Each lane is individually correct — S2 keeps the
    tokens it measured, S4 tokenizes the corpus it was given — and the system is still broken
    if those two corpora are not the same one. That is the exact shape of the two contract
    failures this module's docstring opens with, so it is asserted here rather than assumed.
    """
    remap.encode(np.asarray(token_ids).reshape(-1))


def assert_shard_trainable(
    header: ShardHeader, *, config: DataConfig, path: Path, repo_root: Path
) -> dict[str, TriState]:
    """Rule 3 at the shard boundary. Refuses a held-out split, loudly.

    `open_training_data` guards manifests, but the trainer reads *shards*. Without this the
    held-out check is satisfied on paper and bypassed by indirection: nothing stops a shard set
    built from the held-out manifest being opened by a training process.

    **Two questions, and this used to ask only one.** The split check below reads
    `header.split` — what the shard set *says about itself*. It does not read where the shard
    set *is*. Measured before `repo_root` was threaded here: a shard set at
    `data/heldout/shards-train` whose header honestly declares `split="train"` was refused by
    [`qd_train.data_access.assert_path_not_held_out`] and **admitted** here, `passed=True`. The
    two doors disagreed, and the trainer goes through this one.

    `ShardReader` had taken a `repo_root` since it was written and passed it nowhere — the
    parameter for the check existed, and the check did not. Both questions are asked now, the
    path one first, and the location is the one a header cannot lie about.

    `repo_root` is required rather than optional on purpose. An optional root would let a
    caller omit it and get a pass that skipped the path check, which is the shape
    `assert_path_not_held_out` refuses an empty `held_out_roots` for, one level up.
    """
    assert_path_not_held_out(path, config=config, repo_root=repo_root)
    if header.split == "heldout":
        raise ShardContractViolation(
            f"{path}: this shard set's split is 'heldout'. Rule 3 — held-out data is never read "
            "by a training process. The shards are derived from a manifest the same rule "
            "already refused; reading them here would bypass that check by indirection."
        )
    if header.split not in ("train", "val"):
        raise ShardContractViolation(
            f"{path}: split {header.split!r} is not a trainable split"
        )
    return {
        # Recorded, not merely performed: a check whose only trace is the absence of an
        # exception is indistinguishable from a check that was never wired in. That is how
        # this one came to be missing for as long as it was.
        "shard_path_not_held_out": Ran(passed=True, value=str(path)),
        "shard_split_trainable": Ran(passed=True, value=header.split),
        "shard_provenance_pinned": Ran(
            passed=True,
            value=header.shard_hash(),
            detail=(
                f"data_snapshot_hash={header.data_snapshot_hash[:16]}… "
                f"tokenizer_hash={header.tokenizer_hash[:16]}… "
                f"remap_hash={header.remap_hash[:16]}…"
            ),
        ),
        "shard_not_packed": Ran(
            passed=True,
            value=False,
            detail="SAFETY-2: packing is unsupported for the hybrid; the header cannot be "
            "constructed with packed=True",
        ),
        "held_out_families_configured": Ran(
            passed=True, value=len(config.held_out_families),
            detail=f"{sorted(config.held_out_families)}",
        ),
    }
