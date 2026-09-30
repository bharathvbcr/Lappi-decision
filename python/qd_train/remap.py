"""S2: the 248,320 -> ~80K vocabulary remap — built, applied, and verified.

The plan: *"The 248,320 -> ~80K remap moves ahead of training: it is exact, CPU-only, and
removes a 248K-row `lm_head` from every training step."* Gate: *"Loss on 1K held-out
sequences identical before and after remap to 1e-5; peak memory per sequence recorded."*

The artifact this lane produces is [`qd_train.artifacts.RemapTable`]. That class is the
executable contract shared with S4 and the trainer, and it is **imported**, not restated:
every invariant about table shape, inverseness, special tokens, hashing and on-disk layout
lives there. This module is the *policy* on top of it — which tokens are kept, how the
weights are sliced, and what "identical loss" was actually measured.

## The coverage policy, and why it is the one that refuses

**Keep every token the corpus uses, plus every special id. Never drop a used token.
If that set does not fit the budget, refuse.**

The alternative — keep the top-K by frequency and let the tail fall off — is the policy that
looks reasonable on a frequency plot and is wrong here. A dropped token that the corpus
actually contains has no representation at all: [`RemapTable.encode`] raises on it, by
design, because the only other options are an `UNK` substitution (which trains the model that
some positions mean "something was here") or an out-of-range embedding lookup. So a
top-K policy does not trade a little accuracy for a smaller head; it produces a corpus that
cannot be encoded at all, discovered at the first shard write in S4 rather than here.

Concretely, therefore:

- There is deliberately **no `min_count`** and no frequency threshold. A token with count 1
  is kept exactly as hard as a token with count 10^9. Frequency is not consulted at all
  beyond `count > 0`.
- `target_vocab_size` is a **ceiling, not a target**. The remap keeps what the corpus needs;
  the ceiling only says how large that is allowed to get before the build refuses. Passing a
  ceiling of 80,000 to a corpus needing 79,004 tokens yields a 79,004-row head, not an
  80,000-row one. Filling to the ceiling would buy nothing and cost rows in every step.
- Exceeding the ceiling raises [`RemapCoverageError`]. It is a refusal, not a warning, and
  the fix is upstream: a narrower corpus, or a larger budget with the memory arithmetic
  redone. The fix is never "drop the rarest few thousand".
- A count that was **truncated by a bound** is also a refusal. A partial count silently
  under-reports the used set, which is the same defect as truncation with a friendlier name.

**Kept ids are renumbered in ascending old-id order.** Not by descending frequency: the hash
in [`RemapTable.remap_hash`] is over `new_to_old`, so a frequency ordering would make the
remap's identity depend on tie-breaking between equally frequent tokens, and two lanes
counting the same corpus in a different order would produce two different `remap_hash`
values for the same vocabulary. Ascending old id is total, tie-free and auditable by eye.

**No alignment padding.** Rounding `vocab_size` up to a multiple of 64 or 128 for the GEMM is
a real consideration and is deliberately *not* done here: `vocab_size` is pinned in the shard
header and in the ledger row, so a lane that pads and a lane that does not would compute two
different `remap_hash` values over one corpus — the exact shape of
`GAP-RT-WIRE-CONTEXT-ENCODING` and `GAP-SCHEMA-LABEL-SET-HASH-TWO-MEANINGS`. If padding is
wanted it belongs in one place, upstream of the hash, as a decision recorded in the plan.

## The tie between `lm_head` and `embed_tokens`

`AUDIT/external-surfaces.md` records, for `Qwen/Qwen3.5-2B-Base`: `tie_word_embeddings =
true`, and `model.safetensors.index.json` contains **no `lm_head` key** — the embedding and
the output head are one tensor. Slicing a tied pair into two independent tensors doubles the
memory this lane exists to save and breaks the tie silently: the two copies then drift apart
under training, and nothing reports it.

So [`apply_remap_to_model`] detects the tie **twice**, by two independent signals, and says
which one it believed:

1. `config.tie_word_embeddings` — metadata, and it can be stale or simply wrong for a model
   assembled in memory rather than loaded from a checkpoint.
2. Storage identity of the two weight tensors — the fact on the machine.

When they disagree, **storage wins**, because it is the thing the forward pass actually uses,
and the config flag is then **rewritten to match** the observed reality. Leaving a
`tie_word_embeddings=True` flag on a model whose weights are in fact independent is a
landmine: a later `tie_weights()` call would overwrite one head with the other. The
disagreement is recorded in [`TieObservation`] either way, so it reaches the ledger rather
than only the logs.

## What "identical loss" means, and what it cannot mean

Read literally, the gate is unsatisfiable. Cross-entropy is
`-(z_y - logsumexp(z))`, and after the remap the `logsumexp` runs over ~80K rows instead of
248,320. Dropping 168K rows from the denominator *lowers* the loss by a margin that is
nowhere near 1e-5 — it is a property of the remap, not a bug in it. Any implementation
reporting "loss identical to 1e-5" against the full-vocabulary loss is reporting something
that did not happen.

What the gate can and does mean here is that the remap is **exact**: the surviving rows come
through unchanged, so the loss computed over the surviving vocabulary is identical before and
after. [`remap_parity_report`] therefore measures three things and keeps them apart:

- `logit_exactness` — `max |logits_after[..., j] - logits_before[..., new_to_old[j]]|` over
  every kept column. This is the real claim "the remap is exact" and it is this module's own
  invariant, not a plan gate, so it carries its own named tolerance.
- `loss_parity` — cross-entropy over the *kept* vocabulary, before vs after. This is the
  plan's gate and it carries the plan's 1e-5.
- `full_vocab_loss_delta` — cross-entropy over all 248,320 rows minus the restricted one.
  Reported as a number, gated on nothing. It is the expected, intended consequence of the
  remap, and suppressing it is how the ambiguity above gets laundered into a green tick.

`GAP-S2-PARITY-GATE-TWO-MEANINGS` records the ambiguity in the plan sentence.

## torch

Everything above the parity check is torch-free and tested in the repo venv. torch is an
optional `mac` extra; [`apply_remap_to_model`] and [`remap_parity_report`] import it lazily
and return `NotRun` — never a pass — when it is absent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import numpy as np

from .artifacts import DROPPED, RemapTable, TokenNotInRemap
from .tristate import NotRun, Ran, TriState, aggregate

if TYPE_CHECKING:  # pragma: no cover - typing only; torch is not a repo-venv dependency
    import torch
    from torch import nn

__all__ = [
    "LOGIT_EXACTNESS_TOL",
    "LOSS_PARITY_TOL",
    "MAX_PARITY_LOGIT_BYTES",
    "MAX_PARITY_SEQ_LEN",
    "PARITY_SEQUENCES",
    "CorpusCounts",
    "EmbeddingModel",
    "ParityReport",
    "PeakMemoryRecord",
    "RemapApplication",
    "RemapCoverageError",
    "TieObservation",
    "apply_remap_to_model",
    "build_remap",
    "count_corpus_tokens",
    "remap_parity_report",
    "verify_remap_parity",
]

#: The plan's gate: *"Loss on 1K held-out sequences identical before and after remap to
#: 1e-5"*. Rule 2 — gates are read-only, so this is a module constant and not a parameter.
LOSS_PARITY_TOL: Final[float] = 1e-5

#: This module's own invariant on the sliced head, deliberately named apart from the plan's
#: gate above so that a threshold chosen here is never reported as one the plan set.
LOGIT_EXACTNESS_TOL: Final[float] = 1e-5

#: The plan's N. A parity run that examined fewer carries `n`/`n_total` saying so.
PARITY_SEQUENCES: Final[int] = 1000

#: A parity sequence longer than this is refused rather than truncated: truncating changes
#: the loss, so a truncated sequence would compare two different computations and pass.
#: The value is the model's native context (`AUDIT/external-surfaces.md`:
#: `max_position_embeddings = 262144`), i.e. a structural ceiling — the bound that actually
#: binds in practice is `MAX_PARITY_LOGIT_BYTES` below.
MAX_PARITY_SEQ_LEN: Final[int] = 262_144

#: The parity check is the one place in this program that **must** materialize logits at the
#: full `[1, T, 248320]`: it compares the pre-remap head against the post-remap one, so the
#: fused path in `qd_train.fused_ce` — which exists precisely so the trainer never allocates
#: that tensor — is unavailable here by construction. At T=512 that is already 508 MB in
#: float32, and the restricted gather is live at the same time. A sequence whose logits would
#: exceed this budget is refused with the arithmetic in the message, because the alternative
#: is an allocator failure partway through a 1000-sequence run that has to start over.
MAX_PARITY_LOGIT_BYTES: Final[int] = 2 * 1024**3


class RemapCoverageError(Exception):
    """The corpus cannot be covered by a remap under the requested budget.

    Distinct from [`qd_train.artifacts.TokenNotInRemap`], which is raised at *use* time when a
    token reaches a finished remap that drops it. This one is raised at *build* time and says
    the remap cannot be built at all without dropping something the corpus uses.
    """


@runtime_checkable
class EmbeddingModel(Protocol):
    """The slice of the `transformers.PreTrainedModel` surface this module uses.

    Stated as a protocol rather than importing `PreTrainedModel` for two reasons: torch and
    transformers are an optional extra, and these four accessors are the documented API for
    swapping embeddings, so binding to them keeps the tie bookkeeping in the library's hands
    instead of hard-coding `model.model.embed_tokens` and guessing at the next model family.
    """

    config: object

    def get_input_embeddings(self) -> nn.Module: ...

    def set_input_embeddings(self, value: nn.Module) -> None: ...

    def get_output_embeddings(self) -> nn.Module | None: ...

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None: ...

    # Supplied by `nn.Module` for every model this could be called with. Declared so the
    # parity run can find the device and put the model in eval mode without a cast.
    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def eval(self) -> EmbeddingModel: ...

    def __call__(self, input_ids: torch.Tensor) -> object: ...


# --- Counting the corpus --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusCounts:
    """Per-token occurrence counts over a corpus, and what was actually counted.

    `n_sequences` and `n_tokens` are carried so that a downstream report can say what the
    remap was built over. A count is either complete or it raised; there is no "partial"
    state, because a partial count under-reports the used set and the resulting remap would
    drop a token the corpus contains.
    """

    counts: np.ndarray
    n_sequences: int
    n_tokens: int

    def __post_init__(self) -> None:
        if self.counts.dtype != np.int64:
            raise RemapCoverageError(f"counts must be int64, got {self.counts.dtype}")
        if self.counts.ndim != 1:
            raise RemapCoverageError("counts must be 1-D, indexed by old token id")

    @property
    def source_vocab_size(self) -> int:
        return int(self.counts.size)

    @property
    def n_distinct(self) -> int:
        """Distinct token ids the corpus used. The floor on any remap's `vocab_size`."""
        return int(np.count_nonzero(self.counts))


def count_corpus_tokens(
    sequences: Iterable[np.ndarray],
    *,
    source_vocab_size: int,
    max_sequences: int,
) -> CorpusCounts:
    """Count old token ids over a corpus, refusing anything it could not fully see.

    `max_sequences` is mandatory and has no default. It bounds the iteration, and exceeding
    it **raises** rather than returning a truncated count: a count that stopped early reports
    fewer distinct tokens than the corpus contains, and a remap built on it drops the ones it
    never saw. That failure surfaces as a `TokenNotInRemap` during S4's shard write, far from
    this function, so it is refused here instead.

    Memory is bounded by `source_vocab_size` regardless of how many tokens are counted; only
    the sequence count needs an explicit bound.
    """
    if source_vocab_size <= 0:
        raise RemapCoverageError(f"source_vocab_size must be positive, got {source_vocab_size}")
    if max_sequences <= 0:
        raise RemapCoverageError(f"max_sequences must be positive, got {max_sequences}")

    counts = np.zeros(source_vocab_size, dtype=np.int64)
    n_sequences = 0
    n_tokens = 0

    for seq in sequences:
        if n_sequences >= max_sequences:
            raise RemapCoverageError(
                f"more than max_sequences={max_sequences} sequences were offered. The count "
                "was not completed, and a partial count silently under-reports the used "
                "token set. Raise the bound deliberately rather than remapping on a "
                "corpus this function only partly saw."
            )
        ids = np.asarray(seq).reshape(-1)
        n_sequences += 1
        if ids.size == 0:
            continue
        if not np.issubdtype(ids.dtype, np.integer):
            raise RemapCoverageError(
                f"sequence {n_sequences - 1} has dtype {ids.dtype}; token ids must be integers"
            )
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= source_vocab_size:
            raise RemapCoverageError(
                f"sequence {n_sequences - 1} contains id {hi if hi >= source_vocab_size else lo} "
                f"outside the source vocabulary [0, {source_vocab_size}). The corpus was "
                "tokenized with a different tokenizer than the one this remap is for."
            )
        counts += np.bincount(ids.astype(np.int64, copy=False), minlength=source_vocab_size)
        n_tokens += int(ids.size)

    return CorpusCounts(counts=counts, n_sequences=n_sequences, n_tokens=n_tokens)


def _normalize_counts(
    counts: Mapping[int, int] | np.ndarray | CorpusCounts, *, source_vocab_size: int
) -> np.ndarray:
    """Accept the two shapes a count naturally arrives in, and reject everything else."""
    if isinstance(counts, CorpusCounts):
        if counts.source_vocab_size != source_vocab_size:
            raise RemapCoverageError(
                f"counts were taken over a {counts.source_vocab_size}-token vocabulary but "
                f"source_vocab_size={source_vocab_size} was requested"
            )
        return counts.counts

    if isinstance(counts, np.ndarray):
        if counts.ndim != 1:
            raise RemapCoverageError("a dense count array must be 1-D, indexed by old token id")
        if counts.size != source_vocab_size:
            raise RemapCoverageError(
                f"dense count array has {counts.size} entries but source_vocab_size is "
                f"{source_vocab_size}; the two describe different vocabularies"
            )
        if not np.issubdtype(counts.dtype, np.integer):
            raise RemapCoverageError(f"counts must be integers, got dtype {counts.dtype}")
        dense = counts.astype(np.int64, copy=True)
    elif isinstance(counts, Mapping):
        dense = np.zeros(source_vocab_size, dtype=np.int64)
        for raw_id, raw_count in counts.items():
            old_id = int(raw_id)
            if old_id < 0 or old_id >= source_vocab_size:
                raise RemapCoverageError(
                    f"count key {old_id} is outside the source vocabulary "
                    f"[0, {source_vocab_size})"
                )
            dense[old_id] += int(raw_count)
    else:
        raise RemapCoverageError(
            f"counts must be a CorpusCounts, a dense 1-D array or a mapping, got "
            f"{type(counts).__name__}"
        )

    if np.any(dense < 0):
        first = int(np.flatnonzero(dense < 0)[0])
        raise RemapCoverageError(
            f"count for old id {first} is negative ({int(dense[first])}); counts are occurrences"
        )
    return dense


# --- Building the remap ---------------------------------------------------------------


def build_remap(
    *,
    counts: Mapping[int, int] | np.ndarray | CorpusCounts,
    source_vocab_size: int,
    tokenizer_hash: str,
    special_ids: Sequence[int],
    target_vocab_size: int | None,
) -> RemapTable:
    """Build the remap that keeps every token the corpus uses, or refuse.

    `target_vocab_size` is a **ceiling** and is a required argument: `None` means "no
    ceiling" and has to be written out, because a defaulted-away budget is how a head
    silently ends up larger than the memory arithmetic the plan was built on.

    See the module docstring for the coverage policy and the reasoning behind it. The
    returned table is validated by [`RemapTable.__post_init__`] — this function does not
    re-check inverseness or the special-token rule, it produces input the contract accepts.
    """
    if source_vocab_size <= 0:
        raise RemapCoverageError(f"source_vocab_size must be positive, got {source_vocab_size}")
    if target_vocab_size is not None and target_vocab_size <= 0:
        raise RemapCoverageError(
            f"target_vocab_size must be positive or None, got {target_vocab_size}"
        )

    dense = _normalize_counts(counts, source_vocab_size=source_vocab_size)

    specials: list[int] = []
    for raw in special_ids:
        sid = int(raw)
        if sid < 0 or sid >= source_vocab_size:
            raise RemapCoverageError(
                f"special id {sid} is outside the source vocabulary [0, {source_vocab_size}). "
                "A special token the vocabulary does not contain is a tokenizer mismatch."
            )
        specials.append(sid)

    used = np.flatnonzero(dense > 0)
    # Sorted union. Ascending old id, so the renumbering is total and tie-free — see the
    # module docstring on why frequency order would make `remap_hash` count-dependent.
    kept = np.union1d(used, np.asarray(sorted(set(specials)), dtype=used.dtype))
    kept = kept.astype(np.int64, copy=False)

    if target_vocab_size is not None and kept.size > target_vocab_size:
        n_used = int(used.size)
        n_specials_added = int(kept.size) - n_used
        raise RemapCoverageError(
            f"the corpus uses {n_used} distinct token id(s) and {n_specials_added} further "
            f"special id(s) must be kept, so this remap needs {kept.size} rows, which exceeds "
            f"the ceiling of {target_vocab_size} by {kept.size - target_vocab_size}. "
            "This is a refusal, not a truncation: dropping the rarest tokens to fit would "
            "produce a corpus that RemapTable.encode cannot encode at all, discovered at the "
            "first shard write rather than here. Narrow the corpus, or raise the ceiling with "
            "the lm_head memory arithmetic redone."
        )

    old_to_new = np.full(source_vocab_size, DROPPED, dtype=np.int32)
    old_to_new[kept] = np.arange(kept.size, dtype=np.int32)

    return RemapTable(
        old_to_new=old_to_new,
        new_to_old=kept.astype(np.int32, copy=False),
        tokenizer_hash=tokenizer_hash,
        special_ids=tuple(sorted(set(specials))),
    )


def full_vocab_remap(
    *,
    source_vocab_size: int,
    tokenizer_hash: str,
    special_ids: Sequence[int],
) -> RemapTable:
    """The identity remap: every tokenizer id is kept, at its own index.

    The corpus-built remap of [`build_remap`] cannot encode text it was not counted over,
    and a served model is sent exactly that text (GAP-REMAP-CANNOT-ENCODE-THE-ROWS-IT-WAS-
    NOT-BUILT-FROM, decided (c) on 2026-09-29). This keeps the whole tokenizer vocabulary,
    so `RemapTable.encode` refuses nothing the tokenizer can produce, while every consumer
    that pins a remap table -- the shard header, `remap_text_tower`, the scorer -- works
    unchanged. `remap_text_tower` then slices only the embedding rows past the tokenizer
    (Qwen3.5 pads 248,077 ids to 248,320 rows), which no token can reach.
    """
    if source_vocab_size <= 0:
        raise RemapCoverageError(f"source_vocab_size must be positive, got {source_vocab_size}")
    specials = sorted({int(raw) for raw in special_ids})
    outside = [sid for sid in specials if sid < 0 or sid >= source_vocab_size]
    if outside:
        raise RemapCoverageError(
            f"special id(s) {outside} are outside the source vocabulary [0, "
            f"{source_vocab_size}). A special token the vocabulary does not contain is a "
            "tokenizer mismatch."
        )
    ids = np.arange(source_vocab_size, dtype=np.int32)
    return RemapTable(
        old_to_new=ids,
        new_to_old=ids.copy(),
        tokenizer_hash=tokenizer_hash,
        special_ids=tuple(specials),
    )


# --- Applying the remap to weights -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class TieObservation:
    """What the two independent tie signals said, and which one was believed.

    `config_flag` is `None` when the config carries no `tie_word_embeddings` at all, which is
    a third state and not the same as `False`.
    """

    config_flag: bool | None
    storage_shared: bool
    config_flag_rewritten: bool
    detail: str

    @property
    def agreed(self) -> bool:
        """False when the flag and the storage disagreed, or the flag was absent."""
        return self.config_flag is not None and self.config_flag == self.storage_shared

    @property
    def believed(self) -> str:
        """Always storage. Named explicitly so a report states it rather than implying it."""
        return "storage identity (data_ptr)"

    def to_json(self) -> dict[str, Any]:
        return {
            "config_flag": self.config_flag,
            "storage_shared": self.storage_shared,
            "agreed": self.agreed,
            "believed": self.believed,
            "config_flag_rewritten": self.config_flag_rewritten,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RemapApplication:
    """What [`apply_remap_to_model`] did, in numbers a ledger row can carry."""

    tie: TieObservation
    old_vocab_size: int
    new_vocab_size: int
    hidden_size: int
    rows_removed: int
    bytes_freed: int
    output_head_present: bool
    element_size_bytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "tie": self.tie.to_json(),
            "old_vocab_size": self.old_vocab_size,
            "new_vocab_size": self.new_vocab_size,
            "hidden_size": self.hidden_size,
            "rows_removed": self.rows_removed,
            "bytes_freed": self.bytes_freed,
            "output_head_present": self.output_head_present,
            "element_size_bytes": self.element_size_bytes,
        }


def _storage_key(t: torch.Tensor) -> tuple[str, int]:
    """A key equal for two tensors backed by the same allocation.

    `Tensor.data_ptr()` is the address of element zero, so it would already differ between a
    tensor and a nonzero-offset view of it. The untyped storage pointer identifies the
    allocation itself, which is what "these two are one tensor" actually means. Device is part
    of the key because two allocations on different devices can share an address value.
    """
    return (str(t.device), int(t.untyped_storage().data_ptr()))


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a is b:
        return True
    if a.numel() == 0 or b.numel() == 0:
        # A zero-element tensor has no allocation to compare; an address match would be
        # meaningless and a mismatch would be equally meaningless. Fail closed.
        raise ValueError(
            "cannot decide storage identity: one of the weight tensors has zero elements, "
            "so its storage pointer carries no information. Refusing rather than guessing."
        )
    return _storage_key(a) == _storage_key(b)


def _set_vocab_size(config: object, new_vocab_size: int) -> list[str]:
    """Write the new size everywhere the config already states one. Returns what was set.

    Only attributes that already exist are written. Inventing `vocab_size` on a config that
    never had one would be a guess about a model family this code has not seen.
    """
    written: list[str] = []
    if hasattr(config, "vocab_size"):
        config.vocab_size = new_vocab_size  # type: ignore[attr-defined]
        written.append("config.vocab_size")
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        text_config.vocab_size = new_vocab_size
        written.append("config.text_config.vocab_size")
    return written


def apply_remap_to_model(model: EmbeddingModel, remap: RemapTable) -> RemapApplication:
    """Slice `embed_tokens` and `lm_head` down to the remapped vocabulary, in place.

    The model is **mutated**. `verify_remap_parity` needs the pre-remap model as well, so a
    caller running the parity gate must keep a separate copy (`copy.deepcopy(model)`) before
    calling this; that costs a second set of weights and there is no way around it, because
    the gate compares two forward passes of the same inputs.

    The tie between the output head and the input embedding is detected by two independent
    signals and preserved; see the module docstring. When the two disagree, storage wins and
    `config.tie_word_embeddings` is rewritten to match what the weights actually are.

    Raises `ValueError` if the model's vocabulary does not match the remap's
    `source_vocab_size` — a remap built for a different tokenizer renumbers every row.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only in a torch-free env
        raise RuntimeError(
            "apply_remap_to_model needs torch, which is the optional `mac` extra "
            f"(pip install -e '.[mac]'): {exc}"
        ) from exc

    in_emb = model.get_input_embeddings()
    in_weight = in_emb.weight
    if in_weight.ndim != 2:
        raise ValueError(f"input embedding weight must be 2-D, got shape {tuple(in_weight.shape)}")
    old_vocab_size, hidden_size = int(in_weight.shape[0]), int(in_weight.shape[1])

    # Directional, not equality. An embedding LARGER than the tokenizer's vocabulary is
    # ordinary alignment padding -- Qwen3.5-2B-Base is 248,320 = 1940 x 128 rows over a
    # 248,077-token tokenizer -- and `index_select` below never selects those rows. An
    # embedding SMALLER than the remap's source vocabulary is fatal in the way the old
    # message described, and still refuses.
    if old_vocab_size < remap.source_vocab_size:
        raise ValueError(
            f"the model's input embedding has only {old_vocab_size} rows but the remap was "
            f"built over a {remap.source_vocab_size}-token vocabulary. Ids this remap keeps "
            "would index past the end of the embedding: it was built for the wrong "
            "tokenizer."
        )
    highest_kept = int(max(remap.new_to_old)) if len(remap.new_to_old) else -1
    if highest_kept >= old_vocab_size:
        raise ValueError(
            f"the remap keeps old id {highest_kept}, which is outside the model's "
            f"{old_vocab_size}-row input embedding. Checked directly rather than inferred "
            f"from source_vocab_size={remap.source_vocab_size}, because a remap whose ids "
            "exceed its own declared source size passes every size comparison."
        )

    out_head = model.get_output_embeddings()
    config_flag_raw = getattr(model.config, "tie_word_embeddings", None)
    config_flag = None if config_flag_raw is None else bool(config_flag_raw)

    if out_head is None:
        storage_shared = False
        tie_detail = (
            "the model exposes no output embedding, so there is no head to tie; only the "
            "input embedding was sliced"
        )
    else:
        out_weight = out_head.weight
        if out_weight.ndim != 2 or int(out_weight.shape[0]) != old_vocab_size:
            raise ValueError(
                f"output head weight has shape {tuple(out_weight.shape)}; expected "
                f"({old_vocab_size}, ...) to match the input embedding"
            )
        storage_shared = _same_storage(out_weight, in_weight)
        if config_flag is None:
            tie_detail = (
                "config carries no tie_word_embeddings; storage says "
                f"{'shared' if storage_shared else 'independent'} and that is what was used"
            )
        elif config_flag == storage_shared:
            tie_detail = (
                f"config.tie_word_embeddings={config_flag} agrees with storage identity"
            )
        else:
            tie_detail = (
                f"DISAGREEMENT: config.tie_word_embeddings={config_flag} but the weight "
                f"tensors are {'shared' if storage_shared else 'independent'} in memory. "
                "Storage was believed, because it is what the forward pass reads, and the "
                "config flag has been rewritten to match — an unreconciled flag would let a "
                "later tie_weights() call overwrite one head with the other."
            )

    index = torch.as_tensor(
        np.asarray(remap.new_to_old, dtype=np.int64), device=in_weight.device
    )
    new_vocab_size = remap.vocab_size

    with torch.no_grad():
        new_in = nn.Embedding(
            new_vocab_size,
            hidden_size,
            padding_idx=getattr(in_emb, "padding_idx", None),
            device=in_weight.device,
            dtype=in_weight.dtype,
        )
        new_in.weight.copy_(in_weight.detach().index_select(0, index))
    new_in.weight.requires_grad_(in_weight.requires_grad)

    if new_in.padding_idx is not None and new_in.padding_idx >= new_vocab_size:
        raise ValueError(
            f"the embedding's padding_idx {new_in.padding_idx} is outside the remapped "
            f"vocabulary [0, {new_vocab_size}); it must be remapped by the caller first"
        )

    model.set_input_embeddings(new_in)

    if out_head is not None:
        has_bias = getattr(out_head, "bias", None) is not None
        with torch.no_grad():
            new_out = nn.Linear(
                hidden_size,
                new_vocab_size,
                bias=has_bias,
                device=out_head.weight.device,
                dtype=out_head.weight.dtype,
            )
            if storage_shared:
                # Share the Parameter object itself, not a copy: that is what makes the two
                # one allocation, and it is the whole point of preserving the tie.
                new_out.weight = new_in.weight
            else:
                new_out.weight.copy_(out_head.weight.detach().index_select(0, index))
                new_out.weight.requires_grad_(out_head.weight.requires_grad)
            if has_bias:
                new_out.bias.copy_(out_head.bias.detach().index_select(0, index))
                new_out.bias.requires_grad_(out_head.bias.requires_grad)
        model.set_output_embeddings(new_out)

    _set_vocab_size(model.config, new_vocab_size)
    config_flag_rewritten = False
    if out_head is not None and hasattr(model.config, "tie_word_embeddings"):
        if config_flag != storage_shared:
            config_flag_rewritten = True
        model.config.tie_word_embeddings = storage_shared  # type: ignore[attr-defined]

    # Verify the mutation rather than assume it. A set_output_embeddings that rebuilt the
    # head, or a model that keeps its own reference, would leave the tie broken here, and
    # the whole saving this lane exists for would be silently gone.
    applied_in = model.get_input_embeddings()
    applied_out = model.get_output_embeddings()
    if int(applied_in.weight.shape[0]) != new_vocab_size:
        raise RuntimeError(
            f"after set_input_embeddings the model reports {applied_in.weight.shape[0]} "
            f"embedding rows, expected {new_vocab_size}"
        )
    if out_head is not None:
        if applied_out is None:
            raise RuntimeError(
                "the model had an output head before the remap and reports none after it"
            )
        if int(applied_out.weight.shape[0]) != new_vocab_size:
            raise RuntimeError(
                f"after set_output_embeddings the head has {applied_out.weight.shape[0]} rows, "
                f"expected {new_vocab_size}"
            )
        if _same_storage(applied_out.weight, applied_in.weight) != storage_shared:
            raise RuntimeError(
                "the tie was not preserved: storage sharing before the remap was "
                f"{storage_shared} and is now {not storage_shared}. Slicing a tied pair into "
                "two tensors doubles the memory this remap exists to save."
            )

    element_size = int(in_weight.element_size())
    rows_removed = old_vocab_size - new_vocab_size
    # One tensor's worth when tied, two when not — that is the number the tie is worth.
    n_tensors = 1 if (out_head is None or storage_shared) else 2
    return RemapApplication(
        tie=TieObservation(
            config_flag=config_flag,
            storage_shared=storage_shared,
            config_flag_rewritten=config_flag_rewritten,
            detail=tie_detail,
        ),
        old_vocab_size=old_vocab_size,
        new_vocab_size=new_vocab_size,
        hidden_size=hidden_size,
        rows_removed=rows_removed,
        bytes_freed=rows_removed * hidden_size * element_size * n_tensors,
        output_head_present=out_head is not None,
        element_size_bytes=element_size,
    )


# --- Peak memory ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeakMemoryRecord:
    """Per-sequence memory, with "peak" and "allocated" kept strictly apart.

    `peak_bytes` is `Ran` only where torch exposes a real high-water mark. On MPS it does not:
    torch 2.12.1's `torch.mps` offers `current_allocated_memory`, `driver_allocated_memory`
    and `recommended_max_memory` and no peak or reset API (verified against the installed
    build, not recalled). A point reading taken after the forward is not a high-water mark, so
    it is carried in `allocated_delta_bytes` under a different name — reporting it as a peak
    is exactly the substitution that makes an unmeasured quantity look measured.
    """

    device: str
    method: str
    peak_bytes: TriState
    allocated_delta_bytes: TriState
    n_sequences: int

    def summary(self, label: str) -> str:
        if isinstance(self.peak_bytes, Ran):
            return (
                f"{label}_peak_bytes_per_sequence={self.peak_bytes.value} over "
                f"{self.n_sequences} sequence(s) on {self.device} via {self.method}"
            )
        delta = self.allocated_delta_bytes
        delta_txt = delta.value if isinstance(delta, Ran) else "NOT RUN"
        return (
            f"{label}_peak_bytes_per_sequence=NOT RUN ({self.peak_bytes.reason}) "
            f"[{label}_allocated_delta_bytes={delta_txt}]"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "method": self.method,
            "peak_bytes": self.peak_bytes.to_json(),
            "allocated_delta_bytes": self.allocated_delta_bytes.to_json(),
            "n_sequences": self.n_sequences,
        }


class _MemoryProbe:
    """Device-appropriate memory sampling around one forward pass.

    `begin`/`end` bracket a single sequence; the record returned by `record()` carries the
    **maximum over every sequence bracketed**, which is what "peak memory per sequence" means
    for a run of more than one. A probe that never bracketed anything returns `NotRun` rather
    than a zero.
    """

    def __init__(self, device: torch.device, *, label: str) -> None:
        import torch as _torch

        self._torch = _torch
        self._device = device
        self._kind = device.type
        self._label = label
        self._before = 0
        self._samples = 0
        self._max_peak = 0
        self._max_delta = 0

    def begin(self) -> None:
        if self._kind == "cuda":
            self._torch.cuda.reset_peak_memory_stats(self._device)
            self._before = int(self._torch.cuda.memory_allocated(self._device))
        elif self._kind == "mps":
            self._before = int(self._torch.mps.current_allocated_memory())
        else:
            self._before = 0

    def end(self) -> None:
        if self._kind == "cuda":
            self._max_peak = max(
                self._max_peak, int(self._torch.cuda.max_memory_allocated(self._device))
            )
            after = int(self._torch.cuda.memory_allocated(self._device))
        elif self._kind == "mps":
            after = int(self._torch.mps.current_allocated_memory())
        else:
            self._samples += 1
            return
        self._max_delta = max(self._max_delta, after - self._before)
        self._samples += 1

    def record(self) -> PeakMemoryRecord:
        if self._samples == 0:
            reason = f"{self._label}: no forward pass was bracketed, so nothing was measured"
            return PeakMemoryRecord(
                device=str(self._device),
                method="none",
                peak_bytes=NotRun(reason=reason),
                allocated_delta_bytes=NotRun(reason=reason),
                n_sequences=0,
            )
        if self._kind == "cuda":
            return PeakMemoryRecord(
                device=str(self._device),
                method="torch.cuda.max_memory_allocated after reset_peak_memory_stats",
                peak_bytes=Ran(
                    passed=True,
                    value=self._max_peak,
                    detail="true high-water mark; recorded, not gated on any budget",
                ),
                allocated_delta_bytes=Ran(passed=True, value=self._max_delta),
                n_sequences=self._samples,
            )
        if self._kind == "mps":
            return PeakMemoryRecord(
                device=str(self._device),
                method="torch.mps.current_allocated_memory sampled before and after",
                peak_bytes=NotRun(
                    reason=(
                        f"torch {self._torch.__version__} exposes no peak-memory API for MPS: "
                        "torch.mps has current_allocated_memory, driver_allocated_memory and "
                        "recommended_max_memory only, with no reset or high-water call. The "
                        "point reading is carried as allocated_delta_bytes instead — see "
                        "GAP-S2-MPS-NO-PEAK-MEMORY-API."
                    )
                ),
                allocated_delta_bytes=Ran(
                    passed=True,
                    value=self._max_delta,
                    detail="allocated bytes after the forward minus before; not a high-water mark",
                ),
                n_sequences=self._samples,
            )
        return PeakMemoryRecord(
            device=str(self._device),
            method="none",
            peak_bytes=NotRun(
                reason=(
                    f"no per-sequence memory API for device type {self._kind!r}. Process RSS is "
                    "a monotonic, process-wide high-water mark and is not a per-sequence peak, "
                    "so no number is reported rather than a misleading one."
                )
            ),
            allocated_delta_bytes=NotRun(
                reason=f"torch reports no allocator statistics for device type {self._kind!r}"
            ),
            n_sequences=self._samples,
        )


# --- The parity gate ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Every number the parity run measured, kept apart by what it means.

    `as_tristate` is the gate; the rest is what a ledger row carries. `full_vocab_loss_delta`
    is deliberately reported and deliberately not gated — see the module docstring.
    """

    n_examined: int
    n_offered: int
    max_abs_loss_delta: float | None
    max_abs_logit_delta: float | None
    mean_full_vocab_loss: float | None
    mean_restricted_loss: float | None
    memory_before: PeakMemoryRecord | None
    memory_after: PeakMemoryRecord | None
    failures: tuple[str, ...]
    not_run_reason: str | None
    model_dtype: str | None
    old_vocab_size: int
    new_vocab_size: int

    @property
    def full_vocab_loss_delta(self) -> float | None:
        """Full-vocabulary loss minus restricted loss. The remap's intended effect, not a bug."""
        if self.mean_full_vocab_loss is None or self.mean_restricted_loss is None:
            return None
        return self.mean_full_vocab_loss - self.mean_restricted_loss

    def _memory_summary(self) -> str:
        if self.memory_before is None or self.memory_after is None:
            return "peak memory: NOT RUN (no forward pass was executed)"
        return (
            self.memory_before.summary("pre_remap")
            + "; "
            + self.memory_after.summary("post_remap")
        )

    def as_tristate(self) -> TriState:
        """The gate. `NotRun` whenever the comparison did not execute; never a default pass.

        Memory recording is reported inside the detail rather than aggregated in: the loss
        comparison either ran or it did not, and a host without a peak-memory API must not
        turn a measured loss result into an unmeasured one. The detail always states the
        memory outcome explicitly so a pass can never be read as "memory was recorded".
        """
        if self.not_run_reason is not None:
            return NotRun(reason=self.not_run_reason)

        memory = self._memory_summary()
        vocab = f"{self.old_vocab_size} -> {self.new_vocab_size}"

        if self.failures:
            return Ran(
                passed=False,
                value=self.max_abs_loss_delta,
                n=self.n_examined,
                n_total=self.n_offered,
                detail=(
                    f"{len(self.failures)} sequence(s) failed: {'; '.join(self.failures[:5])}"
                    + (" ..." if len(self.failures) > 5 else "")
                    + f" | vocab {vocab} | {memory}"
                ),
            )

        parts: dict[str, TriState] = {
            "logit_exactness": Ran(
                passed=(self.max_abs_logit_delta or 0.0) <= LOGIT_EXACTNESS_TOL,
                value=self.max_abs_logit_delta,
                n=self.n_examined,
                n_total=self.n_offered,
                detail=(
                    f"max |logits_after[j] - logits_before[new_to_old[j]]| = "
                    f"{self.max_abs_logit_delta}, tol {LOGIT_EXACTNESS_TOL} "
                    "(this module's invariant, not a plan gate)"
                ),
            ),
            "loss_parity": Ran(
                passed=(self.max_abs_loss_delta or 0.0) <= LOSS_PARITY_TOL,
                value=self.max_abs_loss_delta,
                n=self.n_examined,
                n_total=self.n_offered,
                detail=(
                    f"max |loss_after - loss_before| over the KEPT vocabulary = "
                    f"{self.max_abs_loss_delta}, gate {LOSS_PARITY_TOL}"
                ),
            ),
        }
        gate = aggregate(parts, name="remap_parity")
        if isinstance(gate, NotRun):  # pragma: no cover - both parts are always Ran here
            return gate
        return Ran(
            passed=gate.passed,
            value=self.max_abs_loss_delta,
            n=self.n_examined,
            n_total=self.n_offered,
            detail=(
                f"loss parity {self.max_abs_loss_delta} <= {LOSS_PARITY_TOL} over the kept "
                f"vocabulary; logit exactness {self.max_abs_logit_delta} <= "
                f"{LOGIT_EXACTNESS_TOL}; full-vocabulary loss is higher by "
                f"{self.full_vocab_loss_delta} and that is the remap's intended effect, not a "
                f"failure | dtype {self.model_dtype} | vocab {vocab} | {memory}"
                + ("" if gate.passed else f" | FAILED: {gate.detail}")
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "n_examined": self.n_examined,
            "n_offered": self.n_offered,
            "max_abs_loss_delta": self.max_abs_loss_delta,
            "max_abs_logit_delta": self.max_abs_logit_delta,
            "mean_full_vocab_loss": self.mean_full_vocab_loss,
            "mean_restricted_loss": self.mean_restricted_loss,
            "full_vocab_loss_delta": self.full_vocab_loss_delta,
            "memory_before": None if self.memory_before is None else self.memory_before.to_json(),
            "memory_after": None if self.memory_after is None else self.memory_after.to_json(),
            "failures": list(self.failures),
            "not_run_reason": self.not_run_reason,
            "model_dtype": self.model_dtype,
            "old_vocab_size": self.old_vocab_size,
            "new_vocab_size": self.new_vocab_size,
            "gate": self.as_tristate().to_json(),
        }


def _not_run_report(reason: str, remap: RemapTable) -> ParityReport:
    return ParityReport(
        n_examined=0,
        n_offered=0,
        max_abs_loss_delta=None,
        max_abs_logit_delta=None,
        mean_full_vocab_loss=None,
        mean_restricted_loss=None,
        memory_before=None,
        memory_after=None,
        failures=(),
        not_run_reason=reason,
        model_dtype=None,
        old_vocab_size=remap.source_vocab_size,
        new_vocab_size=remap.vocab_size,
    )


def _logits_of(output: object) -> torch.Tensor:
    """transformers returns a `CausalLMOutput`; a bare module returns the tensor."""
    logits = getattr(output, "logits", output)
    if not hasattr(logits, "shape"):
        raise TypeError(
            f"the model returned {type(output).__name__}, which has neither `.logits` nor a "
            "tensor shape. A parity check cannot compare what it cannot read."
        )
    return logits  # type: ignore[return-value]


def remap_parity_report(
    *,
    model_before: EmbeddingModel | None,
    model_after: EmbeddingModel | None,
    remap: RemapTable,
    sequences: Iterable[np.ndarray],
    max_sequences: int = PARITY_SEQUENCES,
    max_seq_len: int = MAX_PARITY_SEQ_LEN,
    max_logit_bytes: int = MAX_PARITY_LOGIT_BYTES,
) -> ParityReport:
    """Measure the parity gate. The canonical implementation; `verify_remap_parity` is its gate.

    `sequences` carry **old** (pre-remap) token ids; the remapped ids are derived here so that
    the two runs are provably the same corpus. A sequence containing a token the remap dropped
    is a measured failure of the gate, not an exception: the remap and the verification corpus
    disagree, and that is a result, not a crash.

    Both models are required because the comparison is per-sequence and needs both forward
    passes; see [`apply_remap_to_model`] on the cost.
    """
    if model_before is None or model_after is None:
        missing = [
            name
            for name, m in (("model_before", model_before), ("model_after", model_after))
            if m is None
        ]
        return _not_run_report(
            f"remap parity: {' and '.join(missing)} was not supplied, so no weights were "
            "available to compare. A gate with no model has verified nothing.",
            remap,
        )
    try:
        import torch
    except ImportError as exc:
        return _not_run_report(
            f"remap parity: torch is not installed in this environment ({exc}). torch is the "
            "optional `mac` extra; the comparison did not execute.",
            remap,
        )

    if max_sequences <= 0:
        return _not_run_report(
            f"remap parity: max_sequences={max_sequences} permits no sequences, so nothing "
            "was compared.",
            remap,
        )

    new_to_old = torch.as_tensor(np.asarray(remap.new_to_old, dtype=np.int64))

    try:
        param = next(iter(model_before.parameters()))
        device, dtype = param.device, param.dtype
    except (AttributeError, StopIteration):
        return _not_run_report(
            "remap parity: model_before exposes no parameters, so there are no weights to "
            "compare and no device to run on.",
            remap,
        )

    new_to_old = new_to_old.to(device)
    # Dropout active in one of the two runs would make the losses differ for a reason that
    # has nothing to do with the remap, so this is asserted rather than assumed of the caller.
    model_before.eval()
    model_after.eval()

    n_examined = 0
    n_offered = 0
    max_loss_delta = 0.0
    max_logit_delta = 0.0
    full_losses: list[float] = []
    restricted_losses: list[float] = []
    failures: list[str] = []
    # Two probes, not one: the saving this lane exists for is the difference between them,
    # and a single figure spanning both forwards would be dominated by the pre-remap head
    # and would say nothing about what a training step now costs.
    probe_before = _MemoryProbe(device, label="pre_remap")
    probe_after = _MemoryProbe(device, label="post_remap")

    with torch.no_grad():
        for raw in sequences:
            n_offered += 1
            if n_examined >= max_sequences:
                # Not an error: the cap is the plan's N. `n`/`n_total` carry the shortfall so
                # the result can never read as complete coverage.
                continue

            old_ids = np.asarray(raw).reshape(-1)
            if old_ids.size < 2:
                failures.append(
                    f"sequence {n_offered - 1}: {old_ids.size} token(s); a causal-LM loss needs "
                    "at least two positions"
                )
                continue
            if old_ids.size > max_seq_len:
                failures.append(
                    f"sequence {n_offered - 1}: {old_ids.size} tokens exceeds max_seq_len="
                    f"{max_seq_len}; refused rather than truncated, because truncating would "
                    "compare two different computations"
                )
                continue
            # The payload bound. Both logit tensors are live at once during the comparison.
            logit_bytes = int(old_ids.size) * (remap.source_vocab_size + remap.vocab_size) * 4
            if logit_bytes > max_logit_bytes:
                failures.append(
                    f"sequence {n_offered - 1}: comparing {old_ids.size} positions over "
                    f"{remap.source_vocab_size} + {remap.vocab_size} rows needs "
                    f"{logit_bytes} bytes of float32 logits, over the {max_logit_bytes}-byte "
                    "budget. The parity check cannot use the fused cross-entropy path — it "
                    "exists to compare the two heads' logits — so shorter verification "
                    "sequences are the fix, not a larger allocation."
                )
                continue
            try:
                new_ids = remap.encode(old_ids)
            except TokenNotInRemap as exc:
                failures.append(f"sequence {n_offered - 1}: {exc}")
                continue

            x_old = torch.as_tensor(old_ids.astype(np.int64, copy=False), device=device)[None, :]
            x_new = torch.as_tensor(new_ids.astype(np.int64, copy=False), device=device)[None, :]

            probe_before.begin()
            logits_full = _logits_of(model_before(x_old)).float()
            probe_before.end()

            probe_after.begin()
            logits_new = _logits_of(model_after(x_new)).float()
            probe_after.end()

            # Directional, exactly as in `apply_remap_to_model`: a head WIDER than the
            # tokenizer is alignment padding -- Qwen3.5-2B-Base scores 248,320 rows over a
            # 248,077-token tokenizer -- whose rows no id reaches. This was `!=` after the
            # surgery had been fixed to accept padding, so the gate refused every sequence
            # of the one model it exists for, the first time it was run against it. NARROWER
            # is the fatal direction: ids the remap keeps would have no logit at all.
            if logits_full.shape[-1] < remap.source_vocab_size:
                failures.append(
                    f"sequence {n_offered - 1}: model_before produced "
                    f"{logits_full.shape[-1]} logits, fewer than the "
                    f"{remap.source_vocab_size}-token vocabulary the remap was built over"
                )
                continue
            if logits_new.shape[-1] != remap.vocab_size:
                failures.append(
                    f"sequence {n_offered - 1}: model_after produced "
                    f"{logits_new.shape[-1]} logits, expected {remap.vocab_size}"
                )
                continue

            restricted = logits_full.index_select(-1, new_to_old)
            max_logit_delta = max(
                max_logit_delta, float((logits_new - restricted).abs().max().item())
            )

            targets = x_new[:, 1:].reshape(-1)
            loss_before = torch.nn.functional.cross_entropy(
                restricted[:, :-1].reshape(-1, remap.vocab_size), targets
            )
            loss_after = torch.nn.functional.cross_entropy(
                logits_new[:, :-1].reshape(-1, remap.vocab_size), targets
            )
            # Over every row the unremapped head scores, padding included: that is the
            # softmax the model computes without the remap, so it is the loss the remap's
            # intended effect is measured from.
            loss_full = torch.nn.functional.cross_entropy(
                logits_full[:, :-1].reshape(-1, logits_full.shape[-1]),
                x_old[:, 1:].reshape(-1).to(torch.int64),
            )

            max_loss_delta = max(max_loss_delta, float((loss_after - loss_before).abs().item()))
            restricted_losses.append(float(loss_before.item()))
            full_losses.append(float(loss_full.item()))
            n_examined += 1

    if n_examined == 0 and not failures:
        return _not_run_report(
            f"remap parity: {n_offered} sequence(s) were offered and none could be compared, "
            "so nothing was verified. An aggregate over zero checks is not a pass.",
            remap,
        )

    return ParityReport(
        n_examined=n_examined,
        n_offered=n_offered,
        max_abs_loss_delta=max_loss_delta if n_examined else None,
        max_abs_logit_delta=max_logit_delta if n_examined else None,
        mean_full_vocab_loss=float(np.mean(full_losses)) if full_losses else None,
        mean_restricted_loss=float(np.mean(restricted_losses)) if restricted_losses else None,
        memory_before=probe_before.record(),
        memory_after=probe_after.record(),
        failures=tuple(failures),
        not_run_reason=None,
        model_dtype=str(dtype),
        old_vocab_size=remap.source_vocab_size,
        new_vocab_size=remap.vocab_size,
    )


def verify_remap_parity(
    *,
    model_before: EmbeddingModel | None,
    model_after: EmbeddingModel | None,
    remap: RemapTable,
    sequences: Iterable[np.ndarray],
    max_sequences: int = PARITY_SEQUENCES,
    max_seq_len: int = MAX_PARITY_SEQ_LEN,
    max_logit_bytes: int = MAX_PARITY_LOGIT_BYTES,
) -> TriState:
    """The S2 gate: loss identical before and after the remap over the kept vocabulary.

    Returns `NotRun` — never a pass — when the comparison could not execute: no torch, no
    weights, no sequences. Peak memory per sequence is recorded in the result's detail, and is
    itself `NotRun` on a device where torch exposes no high-water API rather than being
    silently reported as a point reading.

    For the individual numbers (a ledger row wants them separately), call
    [`remap_parity_report`] and use its `to_json`.
    """
    return remap_parity_report(
        model_before=model_before,
        model_after=model_after,
        remap=remap,
        sequences=sequences,
        max_sequences=max_sequences,
        max_seq_len=max_seq_len,
        max_logit_bytes=max_logit_bytes,
    ).as_tristate()
