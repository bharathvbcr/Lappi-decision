"""Does ``train_ft`` train on a shard set that came out of the real pipeline?

``tools/ft_toy_run.py`` proved S4's loop executes, on a **toy** vocabulary over twelve
synthetic rows built inside that tool, and then named what was left:

    *"Run the same treatment one rung up: real shards, real tokenizer, real remap."*

This is that run. The corpus is the one ``tools/real_tokenizer_pipeline.py`` writes with the
live ``Qwen/Qwen3.5-2B-Base`` tokenizer -- 321 sequences, 2,485,641 tokens, a 13,787-token
remapped vocabulary, lengths from 320 to 34,522 -- read back through the real
:class:`qd_train.shards.ShardReader` and driven through the real
:func:`qd_train.trainer.train_ft`. Nothing about the data is synthesised here; what this
tool supplies by default is a randomly-initialised backbone small enough to run on a Mac,
with ``--real-backbone`` swapping in the real text tower instead, and the
arithmetic that says what the corpus admits.

Two arms, because the real corpus cannot answer both questions at once.

**Arm 1, the real epoch.** One epoch over every sequence in the set. This is the arm that
says ``train_ft`` consumes what ``ShardReader`` produces -- every bucket, every slot kind,
every batch shape -- and it reports the trainer's own ``train.*`` metrics: supervised
tokens, span rows, padding fraction. It makes **no** convergence claim: 321 real prompts of
mean 7,743 tokens are not memorisable in 106 optimizer steps by anything that fits on this
host, and a loss that merely fell would prove nothing (the lesson ``tools/rung0_toy_run.py``
paid for).

**Arm 2, the memorisation.** A bounded subset of the *same real batches*, repeated, so the
floor is memorisation and "reached the floor the corpus admits" is a statement with content.
The subset rule is stated, not tuned: every batch of the real epoch whose padded width is at
most ``--max-width``. Under rule 8 that is a subsample, so this arm is always ``quick``.
The epoch arm is ``quick`` for whichever of :func:`quick_reasons` apply to it -- a
truncated schedule, a NotRun snapshot, repository-history rows, a non-campaign device, the
stand-in backbone -- and for none of them otherwise. Seed count is the family's, enforced
by ``Ledger.promotion_verdict`` and the campaign driver's ``min_seeds``.

## What the abstention question turned into

The toy corpus carried one abstaining row per slot kind, deliberately: *"a corpus of only-
abstaining or only-answering rows would let a head that always abstains, or never does,
reach a floor that reads as learning."* The real corpus is the degenerate case that guard
was written against, and nothing counts it. Of its 243 letter rows -- 123 ``choice`` and 120
``score`` -- **none** has ``noul`` as its gold. Every ``noul`` in the set is the structural
trailing token of a ``span`` row, and ``ft_supervision`` routes exactly those out of the
letter channel. So the abstain row of a choice or score slot is present in the vocabulary,
reachable at serve time, and supervised zero times.

That is structural twice over, not an accident of sampling. The two sources produce five
families; ``DataConfig.held_out_families`` reserves two of them -- ``code.language_id`` and
``qa.answerability`` -- as rule 3's task holdout, so the trainable mixture is exactly
``code.commit_intent`` (choice), ``code.change_scope`` (score) and ``qa.answer_span``
(span), which is what the per-kind family counts below show. Of the five, only
``qa.answer_span`` ever emits a ``noul`` gold (``qd_data.mixture.rewrite_squad``) and it is
a span slot; ``qa.answerability`` turns an unanswerable question into gold ``"no"`` rather
than an abstention, and is held out in any case. The one family that would put an
abstention in the letter channel is ``intent.classification`` on ``clinc/clinc_oos``, which
is not in this corpus.

So this tool decodes the abstention for the slot kind that has one, reports the other two as
**NOT RUN with that reason**, and measures the consequence instead: what training on this
corpus does to the probability mass on the abstain row it never supervises.

## What is reused rather than restated

The floor formulas, the causal block and the counterfactual verdict come from
``tools/ft_toy_run.py`` by import. A second copy of the conditional-entropy computation in
this file would be free to drift from the one the FT lane calibrated against a corpus whose
floor is non-zero, and the drift would look exactly like a correct tool.

The letter row order comes from :func:`qd_data.render.render` for each real row, and the
token id of each letter is read off the shard set itself: a row's trailing token **is** its
gold letter, so 321 rows give the letter-to-id map by correspondence and not by assumption.
The correspondence is checked against ``supervision.npz``'s ``slot_kind`` for all 321.

RUN
---
    /Users/bharath/.venvs/ml/bin/python tools/real_tokenizer_pipeline.py \\
      --out /tmp/qd-real --max-pairs 400 --rev <sha> --ledger
    /Users/bharath/.venvs/ml/bin/python tools/real_ft_run.py --out /tmp/qd-real \\
      --rev <the same full sha> --ledger <path>

``--rev`` is a full 40-character sha on both: every run of this tool writes ledger rows.

The repo venv carries no torch by design, so this refuses there rather than reporting a
vacuous pass. Exit status is non-zero if any claim it makes fails.
"""

from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

# Inline for the reason tools/bpe_line_start_collapse.py states: ruff's E402 exemption
# covers `sys.path` modification before the imports, but an ordinary assignment in between
# is not one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Read from argv rather than from parsed arguments, and deliberately: cuBLAS reads this when
# it initialises, which is the first matmul, and `argparse` has not run by then. With
# deterministic algorithms in force and this unset, torch raises at the first addmm rather
# than silently using a nondeterministic one -- so the failure mode of getting this wrong is
# loud, and the failure mode of parsing argv here instead is a 32 MB cuBLAS workspace on a
# run that did not ask for one. Only set when asked, so an ordinary run is untouched.
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# The MPS caching allocator's ceiling, as fractions of the device's recommended working set
# (48 GiB on a 64 GiB Mac). torch's defaults are 1.7 / 1.4, which let the cache -- not the
# tensors -- grow past physical memory into swap: the full-vocabulary smoke of 2026-09-29
# held 16 GiB of live tensors and 61 GiB of footprint, and ran for 58 minutes paging
# (GAP-MPS-ALLOCATOR-CACHE-SWAPS-INSTEAD-OF-REFUSING). At 1.0 / 0.9 the allocator reclaims
# its cache before the ceiling and raises an out-of-memory error past it: a refusal in
# seconds instead of an hour that measures nothing. Read by the allocator when MPS first
# initialises, hence here, before torch is imported; setdefault, so an explicit value wins.
# Literals, not named constants: ruff's E402 exemption covers os.environ changes before the
# imports, and an ordinary assignment in between is not one.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - the repo venv has no torch by design
    raise SystemExit(
        "torch is not importable from this interpreter. The repo venv carries none on "
        "purpose (pyproject: torch is the optional `mac` extra). Run this with\n"
        "    /Users/bharath/.venvs/ml/bin/python tools/real_ft_run.py --out <dir>\n"
        "Refusing rather than reporting a vacuous pass."
    ) from exc

import numpy as np

# An average's manifest is ckpt_average's format, so reading it back -- and checking its
# sources and its weights against it -- is ckpt_average's too, not a second reader here.
# A Metal export's weights are the same file layout (tower.* and span_head.* beside a
# <name>.manifest.json), so read_average reads them too, after their own manifest's checks.
from ckpt_average import SOURCES as AVERAGE_SOURCES
from ckpt_average import (
    AverageManifest,
    AverageRefusal,
    manifest_path,
    read_average,
    read_manifest,
    verify_sources,
)

# The floor formulas and the causal block, from the lane that calibrated them. Private by
# name because they are this repository's, not a public API -- but a second copy of the
# conditional-entropy computation is the thing to avoid, not an underscore.
from ft_toy_run import (
    FLOOR_SLACK,
    _Block,
    _entropy,
    _letter_floor,
    _span_floor,
)

# Resolving --rev to a commit is repo_git's job, not a second rev-parse here: this tool and
# real_tokenizer_pipeline.py must agree on the string that goes in and comes out of a shard
# header, and two copies of "peel it to a commit" is exactly how they would stop agreeing.
from repo_git import require_full_sha, resolve_rev

# Counting GPUs needs torch and `run_control` is torch-free by contract, so the count lives
# in tools. Its own module rather than this one because `rung0_real_run.py` needs the same
# answer, and importing this file to get it would load a text tower to count a GPU.
from run_cost import n_gpus_for_device
from torch import nn

from qd_data.config import DataConfig
from qd_data.defect_class import CHOICE_SLOT, DEFECT_FAMILY_ID, SPAN_SLOT
from qd_data.errors import QdRefusal
from qd_data.render import DEFAULT_CAPS, second_pass_permutation
from qd_data.rows import DataRow
from qd_data.schema import NOUL_LETTER
from qd_train import composed_slice as cslice
from qd_train.artifacts import (
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    Batch,
    ShardContractViolation,
)
from qd_train.calibration_fit import ece_gate, letters_key
from qd_train.eval_harness import (
    degenerate_head_check,
    permutation_consistency_state,
    permute_within_groups,
    shuffled_label_control,
)
from qd_train.fused_ce import fused_linear_cross_entropy, resolve_chunk_size
from qd_train.heads import (
    RESERVED_NOUL_ROWS,
    SpanPointerHead,
    plan_span_batch,
    serving_scores,
)
from qd_train.ledger import (
    DEFAULT_LEDGER_PATH,
    SUPPLEMENT_KEY,
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    RunRecorder,
)
from qd_train.memory import ADAMW_BF16, ADAMW_FP32, OptimizerSpec
from qd_train.needle import (
    NEEDLE_CASES_PER_DEPTH,
    NEEDLE_HIT_RULE,
    NEEDLE_MIN_RECALL,
    NEEDLE_TARGET_TOKENS,
    NeedleCase,
    build_suite,
    hunk_of_context_line,
    needle_defect_row,
    score_suite,
)
from qd_train.ood import (
    OOD_CASES_PER_CATEGORY,
    OOD_CATEGORIES,
    OOD_MAX_IN_DISTRIBUTION_ABSTAIN,
    OOD_MIN_ABSTAIN,
    OodCase,
    build_ood_suite,
    ood_defect_row,
    ood_gate,
)
from qd_train.optim import DEFAULT_BETA2, apply_lr
from qd_train.power import resolution_state
from qd_train.replay import PriorCache, PriorKLReplay, ReplayRefusal, check_attestation
from qd_train.run_control import (
    MAX_CAP_S,
    CostEstimate,
    LRSchedule,
    RunControl,
    WallClockCap,
    hard_exit_on_cap,
)
from qd_train.shards import (
    HEADER_NAME,
    MAX_POSITIONS_PER_BATCH,
    ShardReader,
    UnencodableGold,
    answer_letter,
    assemble_batch,
    corpus_contradictions,
    encode_slot,
    rendered_training_texts,
    training_texts,
)
from qd_train.trainer import (
    TRAIN_ATTENTION_MASKS,
    ChoicePermutation,
    Progress,
    SpanScoringStep,
    ft_supervision,
    newline_terminated_ids,
    train_ft,
)
from qd_train.tristate import NotRun, Ran, TriState, aggregate, parse_tristate

REPO = Path(__file__).resolve().parents[1]

#: Recipe keys the ported pieces add -- each ONLY when its piece is on, so a run that uses
#: none of them hashes exactly as the rows before them did. Mirrored into the verdict and
#: score rows beside BACKBONE_KEYS, for the same reason those are.
RECIPE_PIECE_KEYS: Final[tuple[str, ...]] = (
    "lower_layers_n",
    "lower_lr_scale",
    "beta2",
    "option_permutation_seed",
    "replay_shard_hash",
    "replay_attestation_sha256",
    "replay_weight",
    "replay_every",
    "replay_direction",
    "wall_clock_cap_s",
    "no_memorise",
    "batch_tokens",
    # --shuffled-label: so any score row of a model trained on permuted golds -- a
    # --score-checkpoint of its weights included -- hashes apart from the real model's.
    "shuffled_label",
    # --checkpoint-skip-layers (Tier A) and --fused-adamw (Tier B): Fable's tier rule records
    # the fused flag on every row, and a piece the ft recipe names must not drop out of the
    # rows scored after it.
    "checkpoint_skip_layers",
    "optimizer_fused",
    # --train-attention-mask none (Tier B, Fable's no-mask ruling): on every scored row too.
    "train_attention_mask",
)

#: The wall-clock cap a run here carries when ``--wall-clock-cap-s`` is not given -- the one
#: every row before 2026-09-29 was taken under. `RunControl` stops at a group boundary and the
#: ledger row records `termination`, so a capped run is legible as capped, not as finished.
WALL_CLOCK_CAP_S = 1_800.0

#: The most ``--wall-clock-cap-s`` may be. The campaign approval is "up to 2-3 days" (72 h),
#: but ``MAX_CAP_S`` -- the program's own cap, 40 h, read-only under rule 2 -- is lower, and
#: ``WallClockCap`` refuses anything above it. The lower of the two binds; it is checked at
#: argv time so the refusal comes before a shard set is read or a tower loaded.
MAX_WALL_CLOCK_CAP_S: Final[float] = min(72 * 3600.0, MAX_CAP_S)

#: The devices the GH200 campaign trains on. A row from any other device is a smoke of the
#: path, not a measurement a decision rests on (``quick_reasons``).
CAMPAIGN_DEVICES: Final[frozenset[str]] = frozenset({"cuda"})

#: RSI-Jev's layer-wise default (fit.py): decoder layers 0-7 at 0.1x.
RSI_LOWER_LAYERS_N: Final[int] = 8
RSI_LOWER_LR_SCALE: Final[float] = 0.1
#: One replay micro-batch per this many training micro-batches: 1 in 7 of all micro-batches,
#: ~14%, the "~15% replay" of docs/train-plan-2026-09-28.md (A5).
DEFAULT_REPLAY_EVERY: Final[int] = 6


@dataclasses.dataclass(frozen=True)
class ReplayPlan:
    """Replay batches (same remap as the train set, already width-filtered) and the term."""

    batches: list[Batch]
    shard_hash: str
    cache_path: Path
    letter_ids: tuple[int, ...]
    weight: float
    every: int
    attestation_sha256: str


def _recipe_pieces(
    *, lower_layers_n: int, lower_lr_scale: float, beta2: float,
    permutation: ChoicePermutation | None, replay: ReplayPlan | None,
    cap_s: float = WALL_CLOCK_CAP_S, no_memorise: bool = False,
    batch_tokens: int | None = None, shuffled_label: Mapping[str, object] | None = None,
    checkpoint_skip_layers: int = 0, fused_adamw: bool = False,
    train_attention_mask: str = "padding",
) -> dict[str, object]:
    """The recipe keys for whichever ported pieces are on. Empty when none is.

    The wall-clock cap, ``--no-memorise`` and ``--batch-tokens`` ride here too, on the same
    terms: named only when they differ from what every earlier row ran under, so those rows
    hash as before. ``batch_tokens`` is ``None`` at the default (the widest bucket), which
    ``_resolve_batch_tokens`` decides. So does ``--checkpoint-skip-layers``: the policy is
    already in every recipe as ``gradient_checkpointing``, and a selective one is named
    beside it only when it is on. And ``--train-attention-mask``: ``"padding"`` is every row
    so far, ``"none"`` (Tier B) names itself.
    """
    if train_attention_mask not in TRAIN_ATTENTION_MASKS:
        raise ValueError(
            f"train_attention_mask must be one of {TRAIN_ATTENTION_MASKS}, got "
            f"{train_attention_mask!r}"
        )
    out: dict[str, object] = {}
    if fused_adamw:
        out["optimizer_fused"] = True
    if train_attention_mask != "padding":
        out["train_attention_mask"] = train_attention_mask
    if checkpoint_skip_layers:
        out["checkpoint_skip_layers"] = checkpoint_skip_layers
    if cap_s != WALL_CLOCK_CAP_S:
        out["wall_clock_cap_s"] = cap_s
    if no_memorise:
        out["no_memorise"] = True
    if batch_tokens is not None:
        out["batch_tokens"] = batch_tokens
    if lower_layers_n:
        out["lower_layers_n"] = lower_layers_n
        out["lower_lr_scale"] = lower_lr_scale
    if beta2 != DEFAULT_BETA2:
        out["beta2"] = beta2
    if permutation is not None:
        out["option_permutation_seed"] = permutation.seed
    if replay is not None:
        out["replay_shard_hash"] = replay.shard_hash
        out["replay_attestation_sha256"] = replay.attestation_sha256
        out["replay_weight"] = replay.weight
        out["replay_every"] = replay.every
        out["replay_direction"] = "base_to_model"
    if shuffled_label is not None:
        out["shuffled_label"] = dict(shuffled_label)
    return out


def _batch_shapes(plan: Sequence[Batch]) -> list[tuple[int, int]]:
    """The distinct ``(rows, width)`` shapes the plan's batches really have, sorted."""
    return sorted({(int(b.tokens.shape[0]), int(b.tokens.shape[1])) for b in plan})


def _resolve_batch_tokens(given: int | None, *, widest: int) -> tuple[int, int | None]:
    """``--batch-tokens`` against the shard set: ``(the value to plan with, its recipe key)``.

    The default is the widest bucket, what every row before the flag trained at -- one
    optimizer step per ~1.4k positions on the phase-3 set (row d732111f). Asking for exactly
    that is the default too, so it hashes the same. Below the widest bucket the planner would
    have no batch for that bucket's rows; above ``MAX_POSITIONS_PER_BATCH`` it refuses.
    Both are refused here, on argv's time, before a tower loads.
    """
    if given is None or given == widest:
        return widest, None
    if not widest <= given <= MAX_POSITIONS_PER_BATCH:
        raise SystemExit(
            f"--batch-tokens {given} is outside [{widest}, {MAX_POSITIONS_PER_BATCH}]: the "
            f"widest bucket ({widest}) must fit in one batch, and the shard reader refuses "
            "more positions per batch than MAX_POSITIONS_PER_BATCH"
        )
    return given, given


def _prior_cache(
    step: object, replay: ReplayPlan, *, backbone_keys: Mapping[str, object], device: str,
    seed: int, resuming: bool,
) -> PriorCache:
    """Load the base's cached letter logits for this replay plan, or build them now.

    The key names what "the base" is: the real tower's snapshot, vocabulary and kernel, or
    -- for the stand-in, whose base is a seeded random init -- its shape and seed; plus the
    device, because two devices' logits differ in the last bits. A cache for another base is
    refused, never rebuilt over. And on a resume a missing cache is refused: the step has
    loaded trained weights by then, and a cache built from them would distil toward the
    trained model, not the base.
    """
    identity = {
        k: backbone_keys[k]
        for k in ("backbone_snapshot", "backbone_vocab", "attn_implementation", "hidden",
                  "heads")
        if k in backbone_keys
    }
    if "hidden" in identity:
        identity["seed"] = seed
    key = {"base": identity, "device": device, "replay_shard_hash": replay.shard_hash}
    expect = {
        **key,
        "letter_ids": list(replay.letter_ids),
        "plan_digest": PriorCache.plan_digest(replay.batches),
    }
    if replay.cache_path.exists():
        return PriorCache.load(replay.cache_path, expect_key=expect)
    if resuming:
        raise ReplayRefusal(
            f"--replay-cache {replay.cache_path} does not exist and this run resumes from a "
            "checkpoint: the model is no longer the base, so a cache built now would anchor "
            "replay to the trained weights. Point at the cache the original run built."
        )
    t0 = time.monotonic()
    cache = PriorCache.build(step, replay.batches, letter_ids=replay.letter_ids, key=key)
    cache.save(replay.cache_path)
    print(
        f"  replay: cached the base's letter logits on {cache.row.size} rows -> "
        f"{replay.cache_path} ({time.monotonic() - t0:.1f}s)",
        flush=True,
    )
    return cache

#: The stand-in's shape and learning rate. 3e-3 is right for a randomly-initialised 128-wide
#: block and catastrophic for pretrained weights.
STANDIN_HIDDEN: Final[int] = 128
STANDIN_HEADS: Final[int] = 4
STANDIN_LR: Final[float] = 3e-3

#: The real tower's learning rate. Measured on a GH200: at STANDIN_LR the span channel missed
#: its floor by 714.50 against a bar of 0.05; at this rate it reached 5.07e-05.
REAL_BACKBONE_LR: Final[float] = 1e-5

#: The optimizer recipes this tool can run, by the name `--optimizer` takes.
#:
#: `bf16` is what torch.optim.AdamW builds for a bf16 tower: 8 B/param all-in, and an
#: `exp_avg_sq` that stops moving after 383 steps because a 1e-3 relative increment is below
#: bfloat16's 2^-8 spacing (tools/moment_precision.py measures it settling at 0.5 against a
#: true 1.0, and reaching 32.0 when the truth is 100.0). `master` keeps an fp32 master and
#: fp32 moments -- 20 B/param, measured on a GH200 -- and tracks the EMA exactly.
#:
#: Neither is the default by accident: `bf16` stays the default because it is what every
#: row in the ledger so far used, and changing that silently would make new rows
#: incomparable to old ones without anything saying so.
OPTIMIZER_RECIPES: Final[dict[str, str]] = {
    "bf16": "torch.optim.AdamW over the bf16 parameters (8 B/param)",
    "master": "fp32 master weights and fp32 moments (20 B/param)",
}

#: The recipe keys that say which backbone a run used. One list, because the ft recipe, the
#: run's return value and the verdict recipe all need the same answer and all three feed a
#: protocol hash -- three hand-maintained copies is how they come to disagree.
BACKBONE_KEYS: Final[tuple[str, ...]] = (
    "hidden",
    "heads",
    "backbone_snapshot",
    "backbone_params",
    "backbone_vocab",
    "optimizer_recipe",
)
KIND_NAMES: dict[int, str] = {SLOT_CHOICE: "choice", SLOT_SCORE: "score", SLOT_SPAN: "span"}

#: The attention kernel, named rather than inherited. `sdpa` is what transformers resolves
#: to unaided on both of this project's hosts today -- so this is the value every existing
#: row was taken at, and recording it changes no number while making the next change visible.
#: It is a DEFAULT here and REQUIRED on `load_text_tower`: a library is entitled to pick for
#: a caller that has an opinion, and this tool has one.
DEFAULT_ATTN_IMPLEMENTATION: Final[str] = "sdpa"

#: Hard caps. This tool answers "does it train on real shards"; a schedule long enough to be
#: interesting is long enough to hide a wiring bug behind a plausible curve.
MAX_PASSES = 2_000
MAX_SEEDS = 8
#: ``--seeds`` when it is not given. Resolved after parsing rather than as argparse's default
#: so a ``--score-plan`` run can tell "not given" from "given as 0 1 2" and refuse the latter.
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

#: Seconds a single forward+backward probe is given before it is called a refusal. Generous:
#: the widest real batch is 34,522 positions and `mps` takes 13.6s for the attention alone.
PROBE_TIMEOUT_S = 240.0


# --- the corpus behind the shard set -------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Label:
    """One written sequence's supervision, reconstructed without a tokenizer.

    ``write_shards`` sorts its rows by ``row_id`` and emits one sequence per slot, skipping
    a whole row whose render refuses. That ordering is replicated here so each sequence of
    the shard set can be named -- and then *checked* against ``supervision.npz``'s
    ``slot_kind``, because a replication that silently drifted would mislabel every row.
    """

    row_id: str
    family_id: str
    slot_name: str
    slot_kind: int
    gold_letter: str
    letters: tuple[str, ...]
    #: ``DataRow.metadata["language"]`` where the row has one -- ``code.defect_class`` rows
    #: do, the commitpackft rewriters' rows do not. What ``ece.lang.*`` groups by.
    language: str | None = None


def _labels(rows: list[DataRow], *, config: DataConfig) -> tuple[list[Label], list[str]]:
    """``(labels, excluded)`` in the order ``write_shards`` writes them.

    No tokenizer: everything here is the renderer's, and the four rows this corpus loses are
    lost inside ``training_texts`` (``ContextTooLargeRefusal``), before any tokenization.
    """
    labels: list[Label] = []
    excluded: list[str] = []
    for row in sorted(rows, key=lambda r: r.row_id):
        staged: list[Label] = []
        try:
            # `seed=` is not optional here. `render` shuffles a choice slot's options **per
            # example** when it is given one, and `training_texts` passes ``config.seed``
            # -- so rendering without it produces a different permutation and the gold
            # letter comes back wrong for some rows and right for others. Measured on this
            # corpus with the seed omitted: `code.change_scope` (a score slot, whose bins
            # are ordinal and not shuffled) was consistent, while `code.commit_intent`
            # returned gold 'A' as token 32 for 37 rows and token 33 for 31 others. A
            # per-example alphabet read at the wrong seed is exactly the mislabelling
            # `answer_letter`'s docstring warns about. One render serves both the alphabets
            # and the sequences: `training_texts` rendering the row a second time returned
            # the same prompt, at the price of the costliest step of relabelling the phase-4
            # train split (2026-10-01).
            rendered, specs = rendered_training_texts(row, seed=config.seed, caps=DEFAULT_CAPS)
            by_name = {slot.name: slot for slot in rendered.slots}
            raw_language = row.metadata.get("language")
            language = raw_language if isinstance(raw_language, str) and raw_language else None
            for spec in specs:
                slot = by_name[spec.slot_name]
                letters = tuple(slot.letter_to_value)
                # A span row's gold is a pair of line numbers, not a letter, and
                # `_slot_suffix` gives a span slot no alphabet but `noul` -- so
                # `training_texts` writes `noul` as its trailing token and `answer_letter`
                # refuses it outright: *"gold (9, 9), which is not among the rendered
                # options ['noul']"*. Measured over this corpus: it refuses exactly the 39
                # pointing span rows and admits the 39 abstaining ones, which is the
                # clearest statement available that the two channels are different kinds
                # of answer. The writer never asks it about a span row, and neither does
                # this.
                gold = (
                    NOUL_LETTER
                    if spec.slot_kind == SLOT_SPAN
                    else answer_letter(row, spec.slot_name, slot.letter_to_value)
                )
                staged.append(
                    Label(
                        row_id=row.row_id,
                        family_id=row.family_id,
                        slot_name=spec.slot_name,
                        slot_kind=spec.slot_kind,
                        gold_letter=gold,
                        letters=letters,
                        language=language,
                    )
                )
        except (UnencodableGold, QdRefusal) as exc:
            # The same `except` `write_shards` uses under `allow_unencodable=True`, so the
            # rows dropped here are the rows dropped there.
            excluded.append(f"{row.row_id}: {type(exc).__name__}: {exc}")
            continue
        labels.extend(staged)
    return labels, excluded


def _letter_ids(reader: ShardReader, labels: list[Label]) -> dict[str, int]:
    """``letter -> post-remap token id``, read off the shard set by correspondence.

    ``training_texts`` ends every sequence with its gold letter, so sequence *i*'s last
    token **is** ``labels[i].gold_letter`` under this remap. Over 321 rows that pins every
    letter the corpus uses, with no tokenizer and no assumption about how Qwen numbers
    single characters. A letter that maps to two ids, or an id claimed by two letters, is a
    refusal: it would mean the replication of the writer's order is wrong, and every label
    after it is on the wrong sequence.
    """
    by_letter: dict[str, set[int]] = collections.defaultdict(set)
    for i, label in enumerate(labels):
        by_letter[label.gold_letter].add(int(reader.sequence(i)[-1]))
    out: dict[str, int] = {}
    for letter, ids in sorted(by_letter.items()):
        if len(ids) != 1:
            raise SystemExit(
                f"letter {letter!r} appears as the trailing token under {len(ids)} different "
                f"ids {sorted(ids)}. Either the shard set and this tool disagree about which "
                "sequence is which row, or a letter is not one token."
            )
        out[letter] = ids.pop()
    collisions = collections.Counter(out.values())
    doubled = {letter: i for letter, i in out.items() if collisions[i] > 1}
    if doubled:
        raise SystemExit(f"two letters share one token id: {doubled}")
    return out


# --- inventory -------------------------------------------------------------------------------


def _contradictions(reader: ShardReader, labels: list[Label]) -> dict[str, object]:
    """Sequences with **identical prefixes and different golds**, over the whole shard set.

    A causal model conditions on ``tokens[:target_index + 1]`` and nothing else, so two
    sequences that agree there and disagree about the answer cannot both be fitted. The
    optimum over such a group is its empirical label distribution, which is what
    ``_letter_floor`` and ``_span_floor`` compute -- so this is the same fact the floors
    already carry, stated as a count of rows rather than as a number of nats.

    It is measured over the **set**, not per batch: a contradiction does not need the two
    rows to land in one batch to be unfittable, and the floors are per batch because that
    is the unit the loss is reduced over.

    Nothing else in the pipeline reports this. ``dedupe`` sees these pairs -- they are
    byte-identical in ``dedupe_text`` -- confirms them at Jaccard 1.0, and **keeps** them
    deliberately: *"same repo means same side of the split, so this pair cannot leak, and
    deleting it would thin the mixture for nothing."* That reasoning is about contamination
    and is sound; it is silent about trainability, and the count it does keep (``n_within``)
    is never read by anything downstream as a supervision signal.

    The grouping itself lives in :func:`qd_train.shards.corpus_contradictions`, which
    ``write_shards`` also calls before writing. This function is the reading end of that
    one owner: it supplies the arrays from a set already on disk and renames the integer
    slot kinds for the report. Two implementations of "do these two rows contradict" is
    two answers to a question that must have one.
    """
    report = corpus_contradictions(
        [reader.sequence(i) for i in range(len(reader))],
        target_index=[int(x) for x in reader._target_index],
        slot_kinds=[label.slot_kind for label in labels],
        span_targets=[(int(s[0]), int(s[1])) for s in reader._span_target],
        labels=[label.row_id for label in labels],
    )
    per_kind = {KIND_NAMES[int(k)]: v for k, v in report["per_kind"].items()}  # type: ignore[union-attr]
    examples = [
        {**e, "kinds": sorted(KIND_NAMES[int(k)] for k in e["kinds"])}  # type: ignore[index]
        for e in report["examples"]  # type: ignore[union-attr]
    ]
    return {**report, "per_kind": per_kind, "examples": examples}


def _inventory(reader: ShardReader, labels: list[Label], excluded: list[str]) -> dict[str, object]:
    """What this shard set actually holds, per slot kind, with both numbers everywhere."""
    lengths = np.asarray(reader.lengths())
    kinds = np.asarray([label.slot_kind for label in labels])
    shard_kinds = reader._slot_kind.astype(np.int64)
    if kinds.size != shard_kinds.size or not bool(np.array_equal(kinds, shard_kinds)):
        differ = (
            int((kinds != shard_kinds).sum())
            if kinds.size == shard_kinds.size
            else "n/a (different lengths)"
        )
        raise SystemExit(
            f"the reconstructed row order disagrees with supervision.npz: {kinds.size} "
            f"labels against {shard_kinds.size} sequences, {differ} slot kinds differ. "
            "Every label would be on the wrong sequence."
        )

    span_target = reader._span_target
    per_kind: dict[str, object] = {}
    for kind, name in sorted(KIND_NAMES.items()):
        sel = kinds == kind
        n = int(sel.sum())
        if not n:
            continue
        sub = lengths[sel]
        gold_noul = sum(
            1 for label in labels if label.slot_kind == kind and label.gold_letter == NOUL_LETTER
        )
        entry: dict[str, object] = {
            "n": n,
            "n_total": len(labels),
            "tokens": int(sub.sum()),
            "len_min": int(sub.min()),
            "len_max": int(sub.max()),
            "len_mean": round(float(sub.mean()), 1),
            "gold_is_noul": gold_noul,
            "families": dict(
                collections.Counter(
                    label.family_id for label in labels if label.slot_kind == kind
                )
            ),
            "row_widths": sorted(
                {len(label.letters) for label in labels if label.slot_kind == kind}
            ),
        }
        if kind == SLOT_SPAN:
            st = span_target[sel]
            entry["pointer_abstaining"] = int(
                ((st[:, 0] == SPAN_ABSTAIN) & (st[:, 1] == SPAN_ABSTAIN)).sum()
            )
            cand = [reader.candidates(i).size for i in np.flatnonzero(sel)]
            entry["candidates_total"] = int(sum(cand))
            entry["candidates_min"] = int(min(cand))
            entry["candidates_max"] = int(max(cand))
        per_kind[name] = entry

    letter_rows = sum(
        1 for label in labels if label.slot_kind in (SLOT_CHOICE, SLOT_SCORE)
    )
    letter_noul = sum(
        1
        for label in labels
        if label.slot_kind in (SLOT_CHOICE, SLOT_SCORE) and label.gold_letter == NOUL_LETTER
    )
    return {
        "root": str(reader.root),
        "sequences": len(reader),
        "tokens": int(lengths.sum()),
        "excluded_rows": len(excluded),
        "excluded_examples": excluded[:3],
        "coverage": reader.coverage.to_json(),
        "span_check": reader.span_check.to_json(),
        "remap_check": reader.checks["shard_remap_matches_header"].to_json(),
        "padding_waste": reader.padding_waste().to_json(),
        "buckets": list(reader.header.buckets),
        "vocab_size": int(reader.header.vocab_size),
        "per_kind": per_kind,
        "letter_rows": letter_rows,
        "letter_rows_whose_gold_is_noul": letter_noul,
    }


def _batch_inventory(reader: ShardReader, *, batch_tokens: int, seed: int) -> dict[str, object]:
    """What the epoch's batches look like: shape, slot kinds, and the letter channel's waste.

    ``live_chunks``/``total_chunks`` is the one number in this tool that is about cost rather
    than correctness. ``ft_supervision`` masks in exactly one position per non-span row, and
    ``_FusedLinearCE.forward`` walks every chunk of positions regardless of the mask, so an
    FT batch pays a CPT-sized projection for an FT-sized supervision. Counted here because a
    rented GPU pays it per step.
    """
    rows: list[dict[str, object]] = []
    live_chunks = total_chunks = 0
    for batch in reader.batches(batch_tokens=batch_tokens, seed=seed, epoch=0):
        supervision = ft_supervision(batch)
        n_rows, width = batch.tokens.shape
        chunk = resolve_chunk_size(n_rows * (width - 1), int(reader.header.vocab_size), None)
        flat = supervision.mask.reshape(-1)
        n_chunks = -(-flat.size // chunk)
        live = len({int(i) // chunk for i in np.flatnonzero(flat)})
        live_chunks += live
        total_chunks += n_chunks
        rows.append(
            {
                "index": batch.index,
                "bucket": batch.bucket,
                "rows": n_rows,
                "width": width,
                "kinds": {
                    KIND_NAMES[int(k)]: int(v)
                    for k, v in sorted(collections.Counter(batch.slot_kind.tolist()).items())
                },
                "supervised": int(supervision.n_supervised),
                "span_rows": supervision.span.n_spans if supervision.span is not None else 0,
                "chunks": n_chunks,
                "live_chunks": live,
            }
        )
    with_span = [r for r in rows if r["span_rows"]]
    return {
        "batches": len(rows),
        "batches_with_a_span_row": len(with_span),
        "batches_letter_only": sum(1 for r in rows if not r["span_rows"]),
        "batches_span_only": sum(1 for r in rows if not r["supervised"]),
        "widths": sorted({int(r["width"]) for r in rows}),
        "letter_channel_live_chunks": live_chunks,
        "letter_channel_total_chunks": total_chunks,
        "rows": rows,
    }


# --- can this device run this batch at all? --------------------------------------------------


def _probe_one(device: str, rows: int, width: int, hidden: int, heads: int) -> dict[str, object]:
    """One forward+backward, in a **subprocess**.

    Not defensiveness. Measured on this host at ``width=34522``: ``mps`` takes the fused
    attention kernel under ``no_grad`` and succeeds, and takes the math path with grad, where
    ``at::native::mps::tiled_bmm_out_mps_impl`` segfaults the interpreter. A ``SIGSEGV`` is
    not an exception; an in-process probe would take the tool, its ledger row and every
    number it had measured with it. Out of process it is an exit status, which is data.
    """
    started = time.monotonic()
    try:
        done = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--probe",
                f"{device}:{rows}:{width}:{hidden}:{heads}",
            ],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "device": device, "rows": rows, "width": width, "ok": False,
            "why": f"no answer within {PROBE_TIMEOUT_S}s",
        }
    wall = time.monotonic() - started
    if done.returncode == 0:
        return {
            "device": device, "rows": rows, "width": width, "ok": True,
            "wall_s": round(wall, 3), "why": done.stdout.strip()[-200:],
        }
    tail = (done.stderr.strip() or done.stdout.strip()).splitlines()
    return {
        "device": device, "rows": rows, "width": width, "ok": False,
        "wall_s": round(wall, 3),
        "returncode": done.returncode,
        "why": (
            f"exit {done.returncode}"
            + (" (SIGSEGV)" if done.returncode == -11 or done.returncode == 139 else "")
            + ": "
            + " | ".join(line.strip() for line in tail[-3:])[:300]
        ),
    }


def _run_probe(spec: str) -> int:
    """``--probe device:rows:width:hidden:heads``. The child half of :func:`_probe_one`."""
    device, rows, width, hidden, heads = spec.split(":")
    block = _Block(int(hidden), int(heads)).to(device)
    x = torch.randn(int(rows), int(width), int(hidden), device=device, requires_grad=True)
    out = block(x)
    out[:, -1].square().mean().backward()
    if device == "mps":
        torch.mps.synchronize()
    print(f"{device} rows={rows} width={width} fwd+bwd ok")
    return 0


# --- the model ---------------------------------------------------------------------------------


class RealFtStep:
    """A :class:`qd_train.trainer.SpanScoringStep` over the shard set's own vocabulary.

    The backbone is a stand-in -- one causal block, randomly initialised -- and says so. What
    is **not** a stand-in: the ``lm_head`` is ``[vocab_size, hidden]`` at the shard header's
    real ``vocab_size``, the letter loss is
    :func:`qd_train.fused_ce.fused_linear_cross_entropy`, the span loss is
    :meth:`qd_train.heads.SpanPointerHead.loss`, and the batches are whatever
    :meth:`qd_train.shards.ShardReader.batches` hands over.

    The position table is sized to the widest batch this run will see and refuses a wider
    one, rather than truncating: a silently clipped position is a row whose answer token is
    no longer at ``target_index``.
    """

    def __init__(self, *, seed: int, device: str, vocab: int, width: int, hidden: int,
                 heads: int, lr: float, span_weight: float,
                 beta2: float = DEFAULT_BETA2, span_channel_off: bool = False) -> None:
        torch.manual_seed(seed)
        # The same opt-in `QwenDecisionStep` takes, for the same one caller: --shuffled-label.
        if span_channel_off:
            if span_weight != 0.0:
                raise ValueError(
                    f"span_channel_off takes span_weight 0.0 exactly, got {span_weight}"
                )
        elif not span_weight > 0.0:
            raise ValueError(
                f"span_weight must be positive, got {span_weight}; zero would train the span "
                "head on nothing while its loss still appeared in the log"
            )
        self.device = device
        # Required, with no default, and the same refusal `QwenDecisionStep` makes. The two
        # had drifted in the direction the backbone handoff warned about: `QwenDecisionStep`
        # carried a `span_weight` that no caller in this repository ever passed, and this
        # stand-in carried none at all, so the tool's two branches summed their channels by
        # two different rules while claiming to exercise one contract.
        self.span_weight = float(span_weight)
        self.max_width = int(width)
        self.embed = nn.Embedding(vocab, hidden).to(device)
        self.position = nn.Parameter(torch.zeros(self.max_width, hidden, device=device))
        self.block = _Block(hidden, heads).to(device)
        self.ln_f = nn.LayerNorm(hidden).to(device)
        self.lm_head = nn.Linear(hidden, vocab, bias=False).to(device)
        self.span_head = SpanPointerHead(hidden).to(device)
        # beta2 through the same parameter the real branch takes; the default is torch's
        # own 0.999, so a run that does not pass --beta2 builds the optimizer it always did.
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, beta2))
        #: Per-micro-batch component losses. `TrainResult.loss_log` carries the combined
        #: number only, and a falling total with a flat span term is a model that learned the
        #: letter and nothing about *where*.
        self.letter_log: list[float] = []
        self.span_log: list[float] = []

    def parameters(self) -> list[torch.nn.Parameter]:
        return [
            *self.embed.parameters(),
            self.position,
            *self.block.parameters(),
            *self.ln_f.parameters(),
            *self.lm_head.parameters(),
            *self.span_head.parameters(),
        ]

    def hidden(self, batch: Batch) -> torch.Tensor:
        """``[B, L, H]`` for ``batch``.

        Takes the whole ``Batch`` rather than its token array so that this and
        ``qd_train.backbone.QwenDecisionStep.hidden`` are one signature. The stand-in needs
        only the tokens; the real tower needs ``batch.lengths`` for its attention mask, and
        a caller holding a bare array cannot supply that.
        """
        tokens = batch.tokens
        if tokens.shape[1] > self.max_width:
            raise ValueError(
                f"a batch {tokens.shape[1]} wide reached a step whose position table is "
                f"{self.max_width}. Clipping it would move every row's target_index."
            )
        ids = torch.as_tensor(tokens.astype(np.int64), device=self.device)
        return self.ln_f(self.block(self.embed(ids) + self.position[: ids.shape[1]]))

    def _letter_loss(self, hidden: torch.Tensor, supervision) -> torch.Tensor | None:
        """``None`` when the batch is all-span: its letter channel is legitimately empty."""
        if supervision.n_supervised == 0:
            return None
        return fused_linear_cross_entropy(
            hidden[:, :-1],
            self.lm_head.weight,
            torch.as_tensor(supervision.targets.astype(np.int64), device=self.device),
            mask=torch.as_tensor(supervision.mask.copy(), device=self.device),
        )

    def accumulate(self, batch: Batch, supervision) -> float:
        loss = self._letter_loss(self.hidden(batch), supervision)
        if loss is None:  # pragma: no cover - `_refuse_unsupervised_rows` refuses this first
            raise RuntimeError("a span-free batch reached accumulate with no supervised token")
        loss.backward()
        self.letter_log.append(float(loss.item()))
        self.span_log.append(0.0)
        return float(loss.item())

    def accumulate_span(self, batch: Batch, supervision) -> float:
        span = supervision.span
        if span is None:  # pragma: no cover - `_train` only routes here when span is not None
            raise RuntimeError("accumulate_span was handed a supervision with no span channel")
        hidden = self.hidden(batch)
        plan = plan_span_batch(span, device=self.device)
        rows = torch.as_tensor(span.rows, device=self.device)
        span_loss = self.span_head.loss(hidden[rows], plan)
        letter = self._letter_loss(hidden, supervision)
        weighted = self.span_weight * span_loss
        total = weighted if letter is None else weighted + letter
        total.backward()
        self.letter_log.append(0.0 if letter is None else float(letter.item()))
        self.span_log.append(float(span_loss.item()))
        return float(total.item())

    def apply(self, *, lr: float) -> None:
        # `qd_train.optim.apply_lr`, the one writer of group["lr"]; see its docstring.
        apply_lr(self.optimizer, lr)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state(self) -> dict[str, object]:
        return {"micro_batches": len(self.letter_log)}

    def load_state(self, state) -> None:
        """This tool never resumes; a checkpoint it cannot restore would be a silent lie."""
        raise NotImplementedError(
            "RealFtStep does not implement resume. tools/real_ft_run.py runs from scratch, "
            "and a load_state that quietly did nothing would let a resumed run report the "
            "checkpoint's loss log against freshly-initialised parameters."
        )


# --- reading the trained model the way the runtime does ------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BatchReadout:
    """What one forward pass over one batch hands the decode: per letter row, its scores over
    the slot's own rows in decode-row order; per span row, its start and end pointer scores
    over the runtime's rows (:func:`qd_train.heads.serving_scores`, the abstention last).

    One tower's readout holds that tower's raw values (the slice of ``lm_head`` the runtime
    decodes, and the pointer head's scores); an ensemble's holds the mean of its towers'
    log-probabilities (:func:`combine_readouts`). The decode reads either the same way.
    """

    letters: dict[int, torch.Tensor]
    plan: Any
    starts: list[torch.Tensor] | None
    ends: list[torch.Tensor] | None
    span_index: dict[int, int]


def _readout(
    step: Any, batch: Batch, letter_rows: Mapping[int, Sequence[int]]
) -> BatchReadout:
    """One step's forward over ``batch``: ``letter_rows`` maps a batch row to the token ids of
    the letters it is decoded over, in decode-row order.

    The tensor ops are the ones the decode always ran -- the full-vocabulary head over every
    position, then indexed -- because a head computed over a subset of positions is not the
    same GEMM and need not round the same on a GPU. Each row's slice is cloned so the
    full-vocabulary tensor is freed when this returns.
    """
    hidden = step.hidden(batch)
    letters: dict[int, torch.Tensor] = {}
    # A batch of span rows reads no letter, and the full-vocabulary head over every position
    # is the costliest tensor here -- 8 GB in fp32 at the needle suite's 8K tokens. Skipped
    # only when no row reads it.
    if letter_rows:
        logits = step.lm_head(hidden)
        for r, ids in letter_rows.items():
            at = int(batch.target_index[r])  # type: ignore[index]
            letters[r] = logits[r, at, torch.as_tensor(list(ids), device=step.device)].clone()
        del logits
    supervision = ft_supervision(batch)
    plan = starts = ends = None
    span_index: dict[int, int] = {}
    if supervision.span is not None:
        plan = plan_span_batch(supervision.span, device=step.device)
        start_scores, end_scores = step.span_head(
            hidden[torch.as_tensor(supervision.span.rows, device=step.device)], plan
        )
        starts = serving_scores(start_scores, plan)
        ends = serving_scores(end_scores, plan)
        span_index = {int(r): k for k, r in enumerate(supervision.span.rows)}
    return BatchReadout(letters, plan, starts, ends, span_index)


#: How a logit ensemble combines its towers, recorded on every ensemble row. The decode is the
#: argmax of this mean, the abstention is the noul row winning it, and the second pass reads
#: the same mean of the permuted batch. Per tower the log-softmax differs from the raw scores
#: by one constant per row, so the argmax and the softmax over the slot's rows equal those of
#: the mean of the raw scores: only the reported values (``row_logits``) are log-probabilities.
ENSEMBLE_COMBINE: Final[str] = (
    "mean over towers of each tower's log-softmax over the slot's own rows (letters: the "
    "slice of lm_head at the slot's letters; span: each pointer's serving_scores, abstention "
    "included), in float64"
)


def on_host(value: object) -> np.ndarray:
    """``value`` as a host array. A torch tensor on any device (CUDA, MPS) is detached and
    copied to the CPU first: ``np.asarray`` of a device tensor raises, and so does one of a
    tensor that requires grad. J4's ens3 gate row on the GH200 (2026-10-01) died on the
    former in :func:`combine_readouts`."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def same_span_plan(a: Any, b: Any) -> bool:
    """Whether two towers planned one batch's span rows identically, read on the host.

    A :class:`qd_train.heads.SpanPlan` is compared in every field (candidates, their
    validity, counts, query positions, gold rows, abstention) and in ``runtime_rows``; the
    same row counts with a different gold row are a different batch. Anything else exposing
    ``runtime_rows`` is compared on that alone.
    """
    if a is None or b is None:
        return a is None and b is None

    def fields(plan: Any) -> dict[str, np.ndarray]:
        names = (
            [f.name for f in dataclasses.fields(plan)] if dataclasses.is_dataclass(plan) else []
        )
        return {
            **{name: on_host(getattr(plan, name)) for name in names},
            "runtime_rows": on_host(plan.runtime_rows),
        }

    fa, fb = fields(a), fields(b)
    return fa.keys() == fb.keys() and all(
        fa[k].shape == fb[k].shape and bool(np.array_equal(fa[k], fb[k])) for k in fa
    )


def combine_readouts(readouts: Sequence[BatchReadout]) -> BatchReadout:
    """The logit ensemble of several towers' readouts of ONE batch (:data:`ENSEMBLE_COMBINE`).

    Every tower must have read the same rows under the same span plan
    (:func:`same_span_plan`, on the host, wherever the towers run); anything else is a
    different batch, and is refused rather than averaged.
    """
    if len(readouts) < 2:
        raise ValueError(f"an ensemble combines two or more towers, got {len(readouts)}")
    first = readouts[0]
    for other in readouts[1:]:
        if set(other.letters) != set(first.letters) or other.span_index != first.span_index:
            raise ValueError("ensemble towers read different rows of one batch")
        if not same_span_plan(first.plan, other.plan):
            raise ValueError("ensemble towers planned one batch's span rows differently")

    def mean_log_probs(scores: Sequence[torch.Tensor]) -> torch.Tensor:
        # On the host: MPS has no float64 at all, and a row's scores are a handful of values
        # (a slot's letters, a context's line starts), so the copy costs nothing. The decode
        # reads the result the same way wherever it lives.
        shapes = {tuple(s.shape) for s in scores}
        if len(shapes) != 1:
            raise ValueError(f"ensemble towers scored one row over different row counts {shapes}")
        return torch.stack(
            [torch.log_softmax(s.detach().cpu().to(torch.float64), dim=-1) for s in scores]
        ).mean(dim=0)

    letters = {
        r: mean_log_probs([ro.letters[r] for ro in readouts]) for r in first.letters
    }
    starts = ends = None
    if first.starts is not None and first.ends is not None:
        starts = [
            mean_log_probs([ro.starts[k] for ro in readouts])  # type: ignore[index]
            for k in range(len(first.starts))
        ]
        ends = [
            mean_log_probs([ro.ends[k] for ro in readouts])  # type: ignore[index]
            for k in range(len(first.ends))
        ]
    return BatchReadout(letters, first.plan, starts, ends, dict(first.span_index))


class TowerEnsemble:
    """Several scored towers decoded as one model: a 3-seed LOGIT ensemble.

    Holds the per-seed steps, each loaded with its own checkpoint, on one device. The decode
    reads every tower's :func:`_readout` of a batch and decodes their
    :func:`combine_readouts`, so the abstention rule, the permuted second pass, the needle
    pointer and the OOD rule all see the ensemble's averaged log-probabilities and nothing
    else. ``max_width`` is the narrowest tower's bound.
    """

    def __init__(self, steps: Sequence[Any]) -> None:
        if len(steps) < 2:
            raise ValueError(f"an ensemble needs two or more towers, got {len(steps)}")
        devices = {str(s.device) for s in steps}
        if len(devices) != 1:
            raise ValueError(f"ensemble towers are on different devices {sorted(devices)}")
        self.steps = list(steps)
        self.device = steps[0].device
        self.max_width = min(int(s.max_width) for s in steps)

    def readout(self, batch: Batch, letter_rows: Mapping[int, Sequence[int]]) -> BatchReadout:
        return combine_readouts([_readout(s, batch, letter_rows) for s in self.steps])


def batch_readout(
    step: Any, batch: Batch, letter_rows: Mapping[int, Sequence[int]]
) -> BatchReadout:
    """``step``'s readout of ``batch``: one tower's, or an ensemble's combined one."""
    if isinstance(step, TowerEnsemble):
        return step.readout(batch, letter_rows)
    return _readout(step, batch, letter_rows)


def _decode(
    step: RealFtStep | TowerEnsemble,
    batches: list[Batch],
    labels_for: dict[int, list[Label]],
    letter_id: dict[str, int],
    *,
    noul_first: bool = False,
    pointer_scores: bool = False,
) -> dict[str, object]:
    """Decode every row of every batch the way ``crates/qd-runtime/src/answer.rs`` does.

    * ``choice``/``score``: a ``QueryKind::Letters`` query is a slice of ``lm_head`` at this
      slot's rendered letters, in rendered order, so the verdict is an argmax over those rows
      at ``target_index``. Abstention iff ``top == rows - RESERVED_NOUL_ROWS``.
    * ``span``: ``PointerStart``/``PointerEnd`` through
      :func:`qd_train.heads.serving_scores` -- the slice the runtime will accept, so no
      padded column it never sees can win.

    ``noul_first`` reads the same trained model under the row order a renderer that listed
    ``noul`` first would produce. It retrains nothing: the letter channel never sees a row,
    so the loss is not merely uninformative about the difference, it is the same number.

    A letter row is decoded over the letters it OFFERS, so every one of them needs a token
    id. ``letter_id`` read off the shard set by gold correspondence (``_letter_ids``) knows
    only letters some row has as its gold; main extends it from the tokenizer's vocabulary
    (``vocab_letter_ids``) when one is available. A row offering a letter that still has no
    id is not decoded -- decoding it over a subset of its rows would not be the runtime's
    query -- and is COUNTED in ``rows_not_decoded`` / ``letters_without_id`` rather than
    raising: until 2026-09-29 this was a bare ``KeyError`` part-way through a paid run.

    Never consults the loss. Returns per-row verdicts and the counts.

    ``step`` may be a :class:`TowerEnsemble`: every score below is then the ensemble's mean
    of per-tower log-probabilities (:func:`combine_readouts`), decoded by the same rules.
    ``pointer_scores`` adds each span row's ``start_logits``/``end_logits`` (the runtime's
    rows, the abstention last) to its verdict -- what ``--suite-logits`` writes for the
    needle suite; off, the verdicts are what they always were.
    """
    verdicts: list[dict[str, object]] = []
    not_decoded: list[str] = []
    letters_without_id: set[str] = set()
    with torch.no_grad():
        for b, batch in enumerate(batches):
            # Which letters each letter row is decoded over, decided before the forward pass:
            # a row offering a letter with no id is counted and never read.
            ordered_for: dict[int, list[str]] = {}
            letter_rows: dict[int, list[int]] = {}
            for r in range(batch.tokens.shape[0]):
                label = labels_for[b][r]
                if label.slot_kind == SLOT_SPAN:
                    continue
                order = [x for x in label.letters if x != NOUL_LETTER]
                ordered = (
                    [NOUL_LETTER, *order] if noul_first else [*order, NOUL_LETTER]
                )
                unknown = [x for x in ordered if x not in letter_id]
                if unknown:
                    not_decoded.append(label.row_id)
                    letters_without_id.update(unknown)
                    continue
                ordered_for[r] = ordered
                letter_rows[r] = [letter_id[x] for x in ordered]
            readout = batch_readout(step, batch, letter_rows)
            plan, start_rows, end_rows = readout.plan, readout.starts, readout.ends
            for r in range(batch.tokens.shape[0]):
                label = labels_for[b][r]
                if label.slot_kind == SLOT_SPAN:
                    assert plan is not None and start_rows is not None and end_rows is not None
                    k = readout.span_index[r]
                    n_rows = int(plan.runtime_rows[k])
                    noul_row = n_rows - RESERVED_NOUL_ROWS
                    # The runtime's row count for this context, restated from the head's own
                    # plan: `answer.rs:70` is `line_count + RESERVED_NOUL_ROWS` and
                    # `noul_row = plan.rows - RESERVED_NOUL_ROWS`. `serving_scores` has
                    # already refused any row whose finite columns are not exactly
                    # `0..n_candidates`, over real candidate sets of 14 to 60 lines.
                    if noul_row != int(plan.n_candidates[k]):  # pragma: no cover
                        raise AssertionError(
                            f"row {label.row_id}: the abstention sits at {noul_row} but the "
                            f"context has {int(plan.n_candidates[k])} line starts"
                        )
                    top_start = int(start_rows[k].argmax())
                    top_end = int(end_rows[k].argmax())
                    abstained = top_start == noul_row or top_end == noul_row
                    expected = bool(plan.abstaining[k])
                    pointer = (
                        {
                            "start_logits": start_rows[k].double().tolist(),
                            "end_logits": end_rows[k].double().tolist(),
                        }
                        if pointer_scores else {}
                    )
                    verdicts.append({
                        **pointer,
                        "kind": "span",
                        "row_id": label.row_id,
                        "slot_name": label.slot_name,
                        # What the report-only per-family metrics group by
                        # (``*.family.<family_id>``); nothing a gate reads.
                        "family_id": label.family_id,
                        "prefix_key": hashlib.sha256(
                            batch.tokens[r, : int(batch.target_index[r]) + 1].tobytes()  # type: ignore[index]
                        ).hexdigest(),
                        "expected_abstain": expected,
                        "rows": n_rows,
                        "noul_row": noul_row,
                        "top": [top_start, top_end],
                        "runtime_verdict": (
                            "abstain" if abstained else f"lines {top_start}..{top_end}"
                        ),
                        "correct": (
                            abstained == expected
                            and (
                                expected
                                or (
                                    top_start == int(plan.gold_start[k])
                                    and top_end == int(plan.gold_end[k])
                                )
                            )
                        ),
                        "abstain_correct": abstained == expected,
                    })
                    continue
                if r not in letter_rows:
                    continue  # counted in not_decoded above
                ordered = ordered_for[r]
                row_tokens = letter_rows[r]
                row_scores = readout.letters[r]
                noul_row = len(row_tokens) - RESERVED_NOUL_ROWS
                at = int(batch.target_index[r])  # type: ignore[index]
                top = int(row_scores.argmax())
                expected = label.gold_letter == NOUL_LETTER
                gold_row = ordered.index(label.gold_letter)
                abstained = top == noul_row
                verdicts.append({
                    "kind": KIND_NAMES[label.slot_kind],
                    "row_id": label.row_id,
                    "slot_name": label.slot_name,
                    "family_id": label.family_id,
                    "prefix_key": hashlib.sha256(
                        batch.tokens[r, : at + 1].tobytes()
                    ).hexdigest(),
                    "expected_abstain": expected,
                    "rows": len(row_tokens),
                    "noul_row": noul_row,
                    "top": top,
                    "gold_row": gold_row,
                    "runtime_verdict": (
                        "abstain" if abstained else f"option ordinal {top}"
                    ),
                    "correct": top == gold_row,
                    "abstain_correct": abstained == expected,
                    "language": label.language,
                    # The mass this model puts on a row it was never supervised at. Softmax
                    # over the slot's own rows, which is what `QueryKind::Letters` decodes.
                    "noul_probability": float(torch.softmax(row_scores, dim=-1)[noul_row]),
                    # The whole distribution the verdict was an argmax of, in decode-row
                    # order. What `ece` and `degenerate_head` read (letter_distributions),
                    # and what a calibration fit would: the argmax alone is not enough for
                    # either. bf16/fp32 -> fp64 is exact, so this is the same numbers `top`
                    # saw -- an ensemble's float64 mean log-probabilities included.
                    "row_logits": row_scores.double().tolist(),
                })

    by_kind: dict[str, dict[str, float]] = {}
    for v in verdicts:
        bucket = by_kind.setdefault(
            str(v["kind"]),
            {"n": 0, "correct": 0, "abstaining": 0, "abstaining_correct": 0,
             "noul_probability_sum": 0.0},
        )
        bucket["n"] += 1
        bucket["correct"] += int(bool(v["correct"]))
        bucket["abstaining"] += int(bool(v["expected_abstain"]))
        bucket["abstaining_correct"] += int(
            bool(v["expected_abstain"]) and bool(v["abstain_correct"])
        )
        bucket["noul_probability_sum"] += float(v.get("noul_probability", 0.0))
    for name, bucket in by_kind.items():
        bucket["noul_probability_mean"] = (
            bucket["noul_probability_sum"] / bucket["n"] if name != "span" else float("nan")
        )
    # Rows that share a prefix must decode identically -- the model cannot see anything
    # else, so this is a determinism check that doubles as proof the prefixes really are
    # identical. It is also the whole cost of the contradiction: within a group whose golds
    # differ, one identical verdict serves them all, so at most one member can be right.
    by_prefix: dict[str, list[dict[str, object]]] = {}
    for v in verdicts:
        key = str(v.get("prefix_key", ""))
        if key:
            by_prefix.setdefault(key, []).append(v)
    inconsistent = [
        group for group in by_prefix.values()
        if len(group) > 1 and len({str(g["runtime_verdict"]) for g in group}) > 1
    ]
    capped_groups = [g for g in by_prefix.values() if len(g) > 1]
    best_possible = sum(1 for _ in capped_groups)  # one correct member per group, at best
    return {
        "noul_first": noul_first,
        "by_kind": by_kind,
        "verdicts": verdicts,
        "prefix_groups": len(capped_groups),
        "prefix_groups_that_decoded_inconsistently": len(inconsistent),
        "rows_in_prefix_groups": sum(len(g) for g in capped_groups),
        "best_possible_correct_in_prefix_groups": best_possible,
        "all_correct": all(bool(v["correct"]) for v in verdicts),
        "letter_abstaining_rows": sum(
            int(by_kind.get(k, {}).get("abstaining", 0)) for k in ("choice", "score")
        ),
        "letter_abstaining_decoded_as_abstain": sum(
            int(by_kind.get(k, {}).get("abstaining_correct", 0)) for k in ("choice", "score")
        ),
        "span_abstaining_rows": int(by_kind.get("span", {}).get("abstaining", 0)),
        "span_abstaining_decoded_as_abstain": int(
            by_kind.get("span", {}).get("abstaining_correct", 0)
        ),
        "rows_decoded": len(verdicts),
        "rows_not_decoded": len(not_decoded),
        "letters_without_id": sorted(letters_without_id),
    }


# --- the runs -------------------------------------------------------------------------------


def _floor_state(gap: object, rows: object, *, detail: str) -> TriState:
    """Three answers for "did this channel reach the floor this plan admits", never two.

    The single owner. :func:`_record_verdict` calls it for the ledger row and :func:`main`
    calls it for the exit status, which is the same arrangement as
    :func:`_counterfactual_holds` and for the same reason: a row and an exit status computed
    from two different conditions is ``GAP-FT-TOY-VERDICT-ROW-CONTRADICTED-ITS-OWN-RUN``.

    **A plan can hold no row of a kind, and that used to read as success.** ``--max-width
    479`` selects only bucket-0 batches, and no letter row in this corpus is shorter than
    1,359 tokens, so that plan's letter channel is empty. :func:`_evaluate` then returns
    ``letter_gap = nan`` -- and ``nan > FLOOR_SLACK`` is ``False``, so **nothing was appended
    to** ``failures`` **for the letter channel** while the row written for the same run said
    ``Ran(passed=False, value=NaN, n=0, n_total=0)``. Measured, on the unmodified tool:
    ``--max-width 479 --passes 3 --seeds 0`` gave a plan of 1 batch, 42 sequences, slot kinds
    ``['span']``, that row, and a ``failures`` list naming only the two span claims. It also
    put four bare ``NaN`` tokens into the ledger, which is not JSON by the spec.

    Zero rows at their floor is neither a pass nor a failure; it is ``NotRun``. A gap that is
    not finite over rows that *do* exist is a broken measurement, which is a failure and must
    never be a pass -- and its value is carried as a string so the ledger stays valid JSON.
    """
    n = int(rows)  # type: ignore[arg-type,call-overload]
    if not n:
        return NotRun(
            reason=(
                "this plan holds no supervised row of this kind, so there is no floor to "
                "reach. A gap over zero rows is nan, nan compares False against every bar, "
                "and 0 of 0 rows at their floor is not a pass."
            )
        )
    value = float(gap)  # type: ignore[arg-type]
    if not math.isfinite(value):
        return Ran(
            passed=False,
            value=str(value),
            n=0,
            n_total=n,
            detail=(
                f"the gap over this plan is {value}, which is not a number. Over {n} row(s) "
                "that is a broken measurement, not a floor that was reached."
            ),
        )
    return Ran(passed=value <= FLOOR_SLACK, value=round(value, 8), n=n, n_total=n, detail=detail)


#: The arms this tool runs. A checkpoint belongs to one of them and `Checkpoint` cannot say
#: which: it carries a seed, a schedule and a position, all of which two arms can share.
ARM_TAGS: Final[tuple[str, ...]] = ("memorise", "epoch")

#: The devices it runs them on. In the filename for the same reason the tag is: `cpu` and
#: `mps` at one seed on one arm differ in NEITHER seed nor schedule nor batch order, so the
#: trainer's resume checks all pass on a checkpoint taken on the other one.
ARM_DEVICES: Final[tuple[str, ...]] = ("cpu", "mps", "cuda")


#: Seconds between progress lines. A line per step would be ~1 s apart on the GH200 and bury
#: everything else; a minute is legible over `tail -f` and ~10 lines per 1500 s arm.
PROGRESS_EVERY_S: Final[float] = 60.0


class ProgressLine:
    """``train_ft``'s ``on_progress``, throttled to one flushed line per ``every_s``.

    The GH200 hour-0 run of 2026-09-30 trained for 25 minutes with nothing in its log:
    stdout was a file, so block-buffered, and the loop said nothing. The line carries what
    a person watching a rented box needs -- step of total, positions per second (padding
    included: that is what the device processed), an ETA to the schedule's end, the loss
    and, on cuda, the allocator's peak. The first step always prints, so a run that is
    stepping at all says so within one step. ``clock`` is the loop's own elapsed time, so
    no second clock is read here.
    """

    def __init__(
        self, label: str, *, every_s: float = PROGRESS_EVERY_S,
        peak_bytes: Callable[[], int] | None = None,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        if not (math.isfinite(every_s) and every_s > 0.0):
            raise ValueError(f"every_s must be finite and positive, got {every_s!r}")
        self.label = label
        self.every_s = every_s
        self._peak = peak_bytes
        self._emit = emit if emit is not None else (lambda s: print(s, flush=True))
        self._last: float | None = None

    def __call__(self, p: Progress) -> None:
        if self._last is not None and p.elapsed_s - self._last < self.every_s:
            return
        self._last = p.elapsed_s
        rate = p.total_positions / p.elapsed_s if p.elapsed_s > 0 else 0.0
        per_step = p.elapsed_s / p.optimizer_step
        eta = per_step * (p.total_steps - p.optimizer_step)
        line = (
            f"  progress {self.label}: step {p.optimizer_step}/{p.total_steps} "
            f"elapsed {p.elapsed_s:.0f}s {per_step:.2f}s/step {rate:.0f} pos/s "
            f"eta {eta:.0f}s loss {p.loss:.4f}"
        )
        if self._peak is not None:
            line += f" peak {self._peak() / (1 << 30):.1f}GiB"
        self._emit(line)


class CheckpointSink:
    """``train_ft``'s ``on_checkpoint``, plus the write at the end of the arm.

    The loop writes only at interval boundaries. The GH200 campaign of 2026-09-30 derived an
    interval of 1,747 steps for a 1,505-step epoch, so all three phase-3 seeds trained to
    completion and saved nothing: the rows and val scores exist, the weights do not.
    ``final`` writes the finished state unless the last interval already wrote that step.
    Timed and sized, because the interval is a cost decision and nothing here could price
    it: this model's checkpoint is ~8.5 GB -- weights plus both AdamW moments.
    """

    def __init__(self, target: Path) -> None:
        self.target = target
        self.last_step: int | None = None

    def __call__(self, ckpt: Any) -> None:
        t0 = time.monotonic()
        written = ckpt.write(self.target)
        took = time.monotonic() - t0
        self.last_step = int(ckpt.optimizer_step)
        payload = sum(
            p.stat().st_size for p in written.parent.glob(f"{written.stem}*") if p.is_file()
        )
        print(
            f"  checkpoint: step {ckpt.optimizer_step} -> {written} "
            f"({payload / (1 << 30):.2f} GiB in {took:.1f}s)",
            flush=True,
        )

    def final(self, ckpt: Any) -> None:
        if self.last_step != int(ckpt.optimizer_step):
            self(ckpt)


def _checkpoint_name(tag: str, seed: int, device: str) -> str:
    """The one place a checkpoint's filename is spelled.

    The (tag x seed x device) product is what this tool actually runs, and every cell of it
    needs its own file. An earlier spelling here was ``f"{tag}-seed{seed}.json"``, which
    collides across devices -- and a collision is worse than an overwrite, because the
    survivor passes every check ``train_ft`` has. A resume is refused across seeds and
    across schedules; ``cpu`` and ``mps`` at one seed on one arm differ in neither.
    """
    if tag not in ARM_TAGS:
        raise ValueError(f"tag must be one of {ARM_TAGS}, got {tag!r}")
    if device not in ARM_DEVICES:
        raise ValueError(f"device must be one of {ARM_DEVICES}, got {device!r}")
    if seed < 0:
        raise ValueError(f"seed must not be negative, got {seed}")
    return f"{tag}-seed{seed}-{device}.json"


def _resume_arm(path: Path) -> tuple[str, int, str]:
    """Which cell a checkpoint belongs to. The inverse of [`_checkpoint_name`].

    Refuses anything this tool did not write rather than guessing. A hand-written
    ``latest.json``, a file from an arm this tool does not have, or the spelling from
    before the device was in the name are all things that would otherwise be routed
    somewhere by accident -- and the destination would accept them, because the trainer
    checks the seed and the schedule and cannot check the hardware.
    """
    if path.suffix != ".json":
        raise ValueError(
            f"{path.name}: a checkpoint written by this tool ends in .json, and its sidecar "
            "is found from that name"
        )
    parts = path.stem.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(
            f"{path.name}: not a name this tool writes. Expected "
            f"<tag>-seed<N>-<device>.json with tag in {ARM_TAGS} and device in "
            f"{ARM_DEVICES}."
        )
    tag, seed_part, device = parts
    if tag not in ARM_TAGS:
        raise ValueError(f"{path.name}: {tag!r} is not one of this tool's arms {ARM_TAGS}")
    if device not in ARM_DEVICES:
        raise ValueError(
            f"{path.name}: {device!r} is not a device this tool runs {ARM_DEVICES}"
        )
    if not seed_part.startswith("seed") or not seed_part[4:].isdigit():
        raise ValueError(
            f"{path.name}: {seed_part!r} is not 'seed' followed by a number, so which seed "
            "this checkpoint was taken at cannot be read off it"
        )
    return tag, int(seed_part[4:]), device


def _channel_balance(run: Mapping[str, object]) -> TriState:
    """What the optimizer was actually asked to minimise, at the first step that had both.

    ``train.final_loss`` is a sum, and a sum does not say which of its terms moved. Rung 0
    reached its held-out majority-class baseline four times while its total loss fell 90%,
    and the cause was this quantity: the span channel opened 6.2x above the choice channel,
    they were summed unweighted, and the optimizer served the pointer. Measured there, the
    choice head sat at the training set's own majority share -- a constant predictor -- on
    every seed at ``span_weight >= 0.5`` and broke free below 0.2. See
    ``AUDIT/rung0-span-weight-2026-09-20.json``.

    The number recorded here is ``span_weight * span_first / letter_first``: the effective
    ratio, which is what the gradient sees, rather than the raw losses. It is recorded and
    **not gated**, deliberately. Rung 0's threshold was measured on a 1.5M-parameter byte
    model and this path trains a 1.4B backbone; carrying a bar across that gap would be an
    inference wearing a gate's clothes, which is the failure this repository keeps finding.
    What the ledger gets is the quantity, so the question can be asked of real rows later
    instead of re-derived from a run that is gone.

    The two losses come from ONE micro-batch -- the first in which both channels are live,
    which is the first span batch, since ``accumulate_span`` is the only path that computes
    both. ``letter_first`` and ``span_first`` are the wrong pair for this and were what this
    function read at first: each is the opening value of its own log, and a span-free batch
    logs a letter loss beside a span of 0.0, so the two can be a step apart. That is the
    right pair for "where did each channel start" and the wrong one for a ratio, which is a
    statement about a single gradient.

    ``NotRun`` when either channel is absent or non-finite: a plan whose batches never carry
    both channels at once has no ratio, and 0.0 or nan would both read as a balance that was
    measured and found benign.
    """
    letter = float(run["letter_at_first_joint_batch"])  # type: ignore[arg-type]
    span = float(run["span_at_first_joint_batch"])  # type: ignore[arg-type]
    weight = float(run["span_weight"])  # type: ignore[arg-type]
    if not (math.isfinite(letter) and math.isfinite(span)):
        return NotRun(
            reason=(
                f"no micro-batch in this plan carried both channels at once (letter "
                f"{letter}, span {span}), so there is no ratio between them. A substituted "
                "0.0 would claim a balanced objective that was never measured."
            )
        )
    if letter <= 0.0:
        return NotRun(
            reason=(
                f"the letter channel opened at {letter}, so the ratio against it is not "
                "defined. It is not evidence that the span channel did not dominate."
            )
        )
    return Ran(
        passed=True,
        value=weight * span / letter,
        detail=(
            f"span {span:.4f} x span_weight {weight} against letter {letter:.4f}, both from "
            f"the FIRST MICRO-BATCH that carried both: an effective "
            f"{weight * span / letter:.2f}:1. "
            "Recorded, not gated -- rung 0 collapsed above roughly 1.2:1 on a 1.5M-parameter "
            "byte model, and that bar has not been shown to transfer to this backbone."
        ),
    )


def _first_joint_batch(
    letter_log: Sequence[float], span_log: Sequence[float]
) -> tuple[float, float] | None:
    """The first micro-batch in which BOTH channels were live, or ``None``.

    Both steps append to both logs on every micro-batch -- ``accumulate`` logs a span of 0.0
    and ``accumulate_span`` logs both -- so the two are parallel by construction and the
    batches where both are positive are exactly the span batches.

    **Mismatched lengths return ``None`` rather than raising**, and that is the whole reason
    this is a function. The obvious spelling is ``zip(..., strict=True)``, which is correct
    about the invariant and wrong about the consequence: it would raise *after* ``train_ft``
    has returned, so a broken pair of logs would destroy the verdict row -- the evaluation,
    the floor gates, the decodes -- for a number that is one line of a detail string. A
    measurement that could not be made is ``NotRun``, which is what ``None`` becomes here,
    and the rest of the row still gets written. Fail closed on the CLAIM, not on the run.
    """
    if len(letter_log) != len(span_log):
        return None
    for lt, sp in zip(letter_log, span_log, strict=True):
        if lt > 0.0 and sp > 0.0:
            return (lt, sp)
    return None


def _span_floor_cause(over_the_plan: float, mean_per_batch: float) -> str:
    """Why the span floor is where it is, decided from the number rather than asserted.

    **This sentence used to be a constant, and it went stale the first time the corpus was
    regenerated.** It read "That floor is ln 2 because every span row in this corpus has a
    prompt-identical twin with a contradictory gold; the mean per-batch floor understates it
    by 3x because the sampler splits most of the pairs" -- true of the shard set of
    2026-09-20, and written into the ledger unchanged beside a measured floor of 0.000000
    on the shard set of 2026-09-21, whose 78 span rows carry no contradictory twin at all.
    A hardcoded explanation that contradicts its own measured value is a claim wearing a
    measurement's clothes, and the ledger is the last place it belongs.

    So the cause is read off the two floors. A plan-level floor materially above the
    per-batch mean is the contradictory-twin signature: rows that cannot both be fitted are
    unfittable whether or not the sampler put them in one batch, and only the plan-level
    computation can see the pair once it is split. A plan-level floor of zero says the
    opposite -- no two rows in this plan share a prefix and disagree.
    """
    if over_the_plan <= 0.0:
        return (
            "The floor is 0.0: no two rows in this plan share a prefix and carry different "
            "golds, so nothing here is unfittable and the channel can in principle reach "
            "zero. A gap is therefore optimisation, not the corpus."
        )
    if over_the_plan > mean_per_batch * 1.5:
        return (
            f"The plan-level floor {over_the_plan:.6f} sits "
            f"{over_the_plan / mean_per_batch:.1f}x above the mean per-batch floor "
            f"{mean_per_batch:.6f}, which is the prompt-identical-twin signature: rows with "
            "contradictory golds are unfittable whether or not the sampler split them, and "
            "only the plan-level computation sees the pair. The per-batch mean is not the "
            "bound."
        )
    return (
        f"The plan-level floor {over_the_plan:.6f} is close to the mean per-batch floor "
        f"{mean_per_batch:.6f}, so what is unfittable here is unfittable within batches "
        "rather than across them."
    )


def _counterfactual_holds(shipped: dict[str, object], defect: dict[str, object]) -> bool:
    """Did reading one trained model under the noul-first row order leave the pointers alone?

    ``noul_first`` reorders the rendered LETTER rows and nothing else. A span slot has no
    letter alphabet beyond ``noul`` -- ``qd_data.render._slot_suffix`` gives its option block
    no pairs -- so its abstention is a pointer row the reordering cannot touch. If it moved,
    the two arms differ in more than the thing under test and the comparison proves nothing.

    That is the **whole** condition, deliberately. The FT lane's version of this also
    required that no letter abstention survive noul-first, which on its corpus was right and
    on this one is unsatisfiable: there is no letter abstention to flip. Bundling the two
    would put ``passed=False`` in a ledger row on a run whose every claim held -- which is
    ``GAP-FT-TOY-VERDICT-ROW-CONTRADICTED-ITS-OWN-RUN``, and this tool reproduced it before
    this function existed. The letter half is reported separately, as ``NotRun`` with its
    reason.

    The single owner of this verdict: :func:`main` and :func:`_record_verdict` both call it.
    """
    return (
        defect["span_abstaining_decoded_as_abstain"]
        == shipped["span_abstaining_decoded_as_abstain"]
        and defect["span_abstaining_rows"] == shipped["span_abstaining_rows"]
    )


def _cost(
    *, device: str, n_gpus: int | None = None, usd_per_hour: float | None = None,
    usd_per_gpu_hour: float | None = None, instance: str | None = None,
    cap_s: float = WALL_CLOCK_CAP_S,
) -> CostEstimate:
    """What this run costs, in one place.

    Two things need the answer and they must not be able to disagree: the ``RunControl``
    that gates the launch under rule 4, and the ledger row that records what was spent.
    Before this, the row read a separate ``cost_usd_per_hour`` float that defaulted to zero
    and that only one of two call sites passed -- so the estimate could be right while the
    row said a GH200 hour cost nothing.

    The cap is the same ``cap_s`` the control uses (``--wall-clock-cap-s``), because
    ``projected_usd`` is priced from the cap; a cost built against a different cap would
    answer a different question about the same run -- and ``RunControl`` refuses the pair.
    """
    return CostEstimate.for_device(
        cap=WallClockCap(cap_s=cap_s),
        device=device,
        n_gpus=n_gpus,
        usd_per_hour=usd_per_hour,
        usd_per_gpu_hour=usd_per_gpu_hour,
        instance=instance,
    )


def _control(
    steps: int, *, device: str, lr: float, checkpoint_every: int = 0,
    n_gpus: int | None = None, usd_per_hour: float | None = None,
    usd_per_gpu_hour: float | None = None, instance: str | None = None,
    approved_by: str = "", cap_s: float = WALL_CLOCK_CAP_S,
) -> RunControl:
    """The cap, the schedule and the price of a local run.

    ``usd_per_hour=0.0`` with ``n_gpus=0`` is a measured fact about a Mac that is already
    bought, not a way around rule 4: a rented machine sets a real rate here and
    ``RunControl`` refuses to start without ``approved_by``.

    That paragraph described a contract nothing enforced. This function passed those two
    zeros on **every** device, so every GH200 run recorded itself as ``local-cuda`` on zero
    GPUs at zero dollars an hour -- and at ``(0, 0.0)`` ``requires_human_approval`` is False
    for any cap, so the human-yes gate could not fire and the per-GPU column check was
    skipped. :meth:`CostEstimate.for_device` is the enforcement: it prices ``cpu`` and
    ``mps`` at zero and refuses to invent a rate for anything else.

    ``checkpoint_every=0`` -- the default, and what every run before 2026-09-21 used --
    means the loop never calls ``on_checkpoint`` and nothing reaches a disk. That was
    survivable while this tool ran for two minutes and is not survivable for a full train.

    ``auto_terminate`` is rule 4's fourth requirement, and ``RunControl`` refuses a run that
    needs a human yes without one. Under the old fixed 30-minute cap no run here ever
    needed one ($0.75 at a GH200's rate); a campaign-length cap does ($44.70 at 30 h), so
    it is passed exactly when the estimate requires approval -- a cheap run keeps the
    cooperative cap alone, as every earlier row did. With it armed, the watchdog fires
    within about a second of the cap while the loop checks only between optimizer steps, so
    a capped run will usually end as a ``killed`` row rather than ``completed`` with
    ``train.termination == 'wall_clock_cap'``. Neither can promote.
    """
    cap = WallClockCap(cap_s=cap_s)
    cost = _cost(
        device=device, n_gpus=n_gpus, usd_per_hour=usd_per_hour,
        usd_per_gpu_hour=usd_per_gpu_hour, instance=instance, cap_s=cap_s,
    )
    return RunControl(
        schedule=LRSchedule(
            peak_lr=lr, total_steps=steps, warmup_steps=max(1, steps // 20), min_lr=lr / 10
        ),
        cap=cap,
        cost=cost,
        grad_accum=1,
        checkpoint_every=checkpoint_every,
        # Threaded through so the refusal RunControl already makes is reachable. It fires on
        # `cost.requires_human_approval and not approved_by.strip()` -- which, while the
        # rate was 0.0 and n_gpus 0, could not fire at all.
        approved_by=approved_by,
        auto_terminate=hard_exit_on_cap if cost.requires_human_approval else None,
    )


def _backbone_commit(recipe: Mapping[str, object]) -> str:
    """Name the backbone that actually ran, or refuse.

    ``backbone_commit`` is one of the five components of ``Protocol``, whose own docstring
    says two rows are comparable only if their protocol hashes match. So a format string
    that assumes the stand-in does not merely mislabel a real-tower run -- it declares that
    run comparable to a 128x4 single block. Five rows in ``ledger/gh200-2026-09-20.jsonl``
    carry ``backbone_commit="scratch:128x4:1block"`` over notes reading "Backbone is the
    REAL text tower ... 1,881,825,088 trainable parameters"; the structured field is the
    one a query filters on, so it is the worse of the two to have wrong.

    ``qd_train.backbone.load_text_tower`` already states the convention this implements --
    "the local checkpoint directory -- the one whose hash is ``backbone_commit``" -- and
    ``tools/real_tokenizer_pipeline.py`` already follows it, reading the same revision out
    of the HF cache's ``refs/main``. The snapshot directory's name *is* that revision
    (verified: ``refs/main`` and the sole snapshot dir are both
    ``b1485b2fa6dfa1287294f269f5fb618e03d52d7c``), which is why the name travels between
    machines where the absolute path does not.

    Refusing rather than inventing is the pipeline's rule too: "A protocol naming no
    backbone identifies nothing; refusing to write a row rather than inventing one."
    """
    snapshot = recipe.get("backbone_snapshot")
    if snapshot is not None:
        text = str(snapshot)
        if "/" in text or "\\" in text:
            raise ValueError(
                f"backbone_snapshot is {text!r}, which is a path rather than a revision. "
                "This field feeds recipe_hash and backbone_commit, and an absolute path "
                "differs between machines -- /home/ubuntu/... on a rented box and "
                "/Users/bharath/... here -- so the same run would get two protocol hashes "
                "and stop being comparable to itself. Store the snapshot directory's name, "
                "which is the revision (`tower.snapshot.name`)."
            )
        return f"{text}:vocab{recipe['backbone_vocab']}"
    hidden, heads = recipe.get("hidden"), recipe.get("heads")
    if hidden is None or heads is None:
        raise ValueError(
            "this recipe names no backbone: it carries neither backbone_snapshot (the real "
            f"tower) nor hidden/heads (the stand-in), only {sorted(recipe)}. A protocol "
            "naming no backbone identifies nothing, and every comparison against the row it "
            "would write is meaningless. Refusing to write a row rather than inventing one."
        )
    return f"scratch:{hidden}x{heads}:1block"


def _protocol(*, reader: ShardReader, seed: int, recipe: dict[str, object]) -> Protocol:
    """The shard set's own hashes, not this tool's. ``data_snapshot_hash`` and
    ``tokenizer_hash`` come out of the header the pipeline wrote, so a ledger row names the
    corpus and the tokenizer that actually produced these tokens."""
    return Protocol(
        data_snapshot_hash=reader.header.data_snapshot_hash,
        tokenizer_hash=reader.header.tokenizer_hash,
        backbone_commit=_backbone_commit(recipe),
        recipe_hash=hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        seed=seed,
    )


def device_budget(estimate_bytes: int, device: str) -> TriState:
    """``memory.py``'s step estimate against what ``device`` can hold, before training.

    cuda: the free bytes ``torch.cuda.mem_get_info`` reports now, so another tenant's memory
    counts against the run. mps: ``torch.mps.recommended_max_memory``, the working set the
    watermark cap above is a fraction of. cpu: not checked, and said so.
    """
    gib = 1024**3
    if device == "cuda":
        available, source = int(torch.cuda.mem_get_info()[0]), "torch.cuda.mem_get_info free"
    elif device == "mps":
        available, source = (
            int(torch.mps.recommended_max_memory()), "torch.mps.recommended_max_memory"
        )
    else:
        return NotRun(reason=f"device {device!r} has no memory budget this tool can read")
    return Ran(
        passed=estimate_bytes <= available,
        # The estimate is the value; the budget is in the detail. Not n / n_total: those
        # are a count examined out of a total, and an estimate over budget is exactly the
        # case where the first exceeds the second.
        value=estimate_bytes,
        detail=(
            f"memory.py estimates {estimate_bytes} B ({estimate_bytes / gib:.2f} GiB) for "
            f"this step (tensors only, with its stated allowance) against {available} B "
            f"({available / gib:.2f} GiB) from {source}"
        ),
    )


def _recorder(ledger: Ledger, *, reader: ShardReader, seed: int, recipe: dict[str, object],
              run_kind: str, quick_reasons: Sequence[str], notes: str,
              wall_clock_s: float | None, cost: CostEstimate | None,
              protocol: Protocol | None = None) -> RunRecorder:
    """Both of this tool's row kinds go through here, and they need different answers.

    ``None`` from :func:`_train`, whose ``with`` block contains ``train_ft``. A measured
    elapsed from :func:`_record_verdict`, which describes a decode that finished before its
    recorder existed -- 162 of this tool's rows recorded the time taken to write the row
    because that distinction had no way to be stated.

    ``quick`` is the run's own facts (:func:`quick_reasons`), every one that applies. It was
    ``True`` on every row this tool ever wrote, so nothing the campaign trained could have
    promoted however it ran. An ``ft`` row's truncation is added by the recorder itself, from
    the ``train.termination`` the loop writes inside its block.

    ``protocol`` is given only by a supplement row (:func:`_record_shuffled_label`), which
    carries the protocol of the eval row it supplements so that it joins that row's seed
    family; its own settings are in ``recipe``. Every other row hashes its own recipe.
    """
    reasons = [r for r in quick_reasons if r.strip()]
    return RunRecorder(
        ledger,
        entry_point=Path(__file__),
        protocol=(
            _protocol(reader=reader, seed=seed, recipe=recipe) if protocol is None else protocol
        ),
        run_kind=run_kind,  # type: ignore[arg-type]
        repo=REPO,
        env=Environment.detect(device=str(recipe["device"])),
        wall_clock_s=wall_clock_s,
        cost=cost,
        # The same dict `_protocol` hashed into `recipe_hash`, stored as well as hashed.
        # The hash makes two recipes incomparable and says nothing about how they differ;
        # a sweep's rows could not name their own arm without the launch command.
        recipe=recipe,
        quick=bool(reasons),
        quick_reason="; ".join(reasons) if reasons else None,
        notes=notes,
    )


def optimizer_spec(dtype: str, optimizer_recipe: str) -> OptimizerSpec:
    """The optimizer layout a step of this dtype and recipe is built with -- one owner.

    ADAMW_BF16 is not ADAMW_FP32: torch.optim.AdamW keeps exp_avg and exp_avg_sq in the
    parameter dtype, so a bf16 tower gets 2-byte states, and load_text_tower refuses the
    mismatch rather than budgeting a layout nothing builds. The master spec is the other real
    recipe -- fp32 master, fp32 moments -- and QwenDecisionStep builds whichever the tower
    names. Scoring a checkpoint builds the same step, so it asks here too: on 2026-09-30 a
    bf16 score of a ``master``-trained checkpoint was given ADAMW_BF16, whose moments
    ``build_optimizer`` refuses for the 1,505-step schedule, though scoring takes no step.
    """
    if dtype == "fp32":
        return ADAMW_FP32
    if dtype != "bf16":
        raise ValueError(f"no optimizer layout for dtype {dtype!r}")
    if optimizer_recipe == "master":
        return OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    return ADAMW_BF16


def _real_step(
    *, backbone: Path, reader: ShardReader, plan: list[Batch], device: str, dtype: str,
    spec: OptimizerSpec, attn_implementation: str, seed: int, lr: float, total_steps: int,
    span_weight: float, width: int, lower_layers_n: int = 0, lower_lr_scale: float = 1.0,
    beta2: float = DEFAULT_BETA2, eval_widths: Sequence[int] = (),
    span_channel_off: bool = False, checkpoint_skip_layers: int = 0, fused_adamw: bool = False,
    train_attention_mask: str = "padding",
) -> tuple[Any, Any, TriState]:
    """The real tower, remapped to the shard set, budgeted, and wrapped in a step.

    ``eval_widths`` are the widths of batches the step will only DECODE -- the needle
    suite's ~8K cases. They raise the step's ``max_width`` bound and nothing else: the
    budget prices ``plan``'s forward+backward with optimizer state, which would misprice a
    no-grad forward, so an eval-only shape is not budgeted here. A step bounded by
    ``plan`` alone refused the needle suite after a full val pass on 2026-09-30.

    The one place the real backbone becomes a step: ``_train`` builds it in bf16 to train,
    and ``--score-checkpoint`` builds it in fp32 to receive a checkpoint's weights. Returns
    ``(step, tower, budget)``.
    """
    # Imported here, not at module scope: qd_train.backbone needs transformers and
    # safetensors, which are the optional `mac` extra.
    from qd_train.backbone import (
        QwenDecisionStep,
        footprint_at,
        linear_attention_on_reference_path,
        load_text_tower,
        remap_text_tower,
    )

    tower = load_text_tower(
        backbone,
        gradient_checkpointing=True,
        optimizer=spec,
        attn_implementation=attn_implementation,
        device=device,
        dtype=dtype,
        # A real batch's shape; the budget below takes the worst of all of them.
        rows=int(plan[0].tokens.shape[0]),
        width=int(plan[0].tokens.shape[1]),
        checkpoint_skip_layers=checkpoint_skip_layers,
    )
    # On CUDA the linear-attention layers must be on fla's kernels. transformers falls back
    # to its torch reference silently apart from a log line, and puts that path at more
    # than an order of magnitude slower: a run on it is one that hits its wall-clock cap
    # having trained a fraction of the plan. CPU and MPS have no fla kernel, so there the
    # reference path is the only one and is recorded rather than refused.
    slow = linear_attention_on_reference_path(tower.linear_attention_kernels)
    if device == "cuda" and "chunk_gated_delta_rule" in slow:
        raise SystemExit(
            "linear attention is bound to transformers' torch reference on cuda "
            f"({tower.linear_attention_kernels}); fla's chunk_gated_delta_rule did not "
            "import. Refusing rather than training on the slow path."
        )
    # The shard set's ids are post-remap, so the tied embedding has to be sliced to the
    # same vocabulary or every id indexes a different row than the one it names. The
    # reader's own table is used rather than a second one read from disk here.
    if reader.remap is None:
        raise ValueError(
            f"{reader.header.shard_hash()}: this shard set carries no remap table, but "
            "the real tower's embedding is 248,320 rows and the set's ids are post-remap. "
            "Training would index the wrong row for every token. Refusing."
        )
    tower = remap_text_tower(tower, reader.remap)
    # The footprint the row records and the budget checks is the costliest batch the
    # plan really contains. It used to pair the plan's most rows with its widest width
    # -- harmless at batch_tokens = widest bucket, where they nearly coincide, and ~8x
    # too high at --batch-tokens 32768, which it refused at 105 GiB on 2026-09-30.
    tower = dataclasses.replace(
        tower,
        footprint=max(
            (footprint_at(tower, rows=r, width=w) for r, w in _batch_shapes(plan)),
            key=lambda f: f.total_bytes,
        ),
    )
    # Before a step is paid for: memory.py's estimate for this batch shape and optimizer,
    # against what the device can hold. A refusal here costs seconds; the unchecked
    # full-vocabulary smoke of 2026-09-29 paged for 58 minutes. A pass is necessary, not
    # sufficient -- the estimate covers tensors, and MPS's Metal-side allocations are not
    # in it -- so it is recorded on the row, and the MPS watermark cap stays the backstop.
    budget = device_budget(tower.footprint.total_bytes, device)
    if isinstance(budget, Ran) and not budget.passed:
        raise SystemExit(f"device budget: {budget.detail}")
    # `total_steps` travels into the step, so a schedule that would outlive its own second
    # moment is refused before the tower is trained rather than discovered in a loss
    # curve that shows nothing.
    step = QwenDecisionStep(
        # The same `seed` this run records in its protocol and names its checkpoint
        # with. Until QwenDecisionStep took one, that seed governed the batch order and
        # not the span head's initialisation, so two runs at one seed were two runs.
        tower,
        seed=seed,
        lr=lr,
        total_steps=total_steps,
        span_weight=span_weight,
        max_width=max((width, *eval_widths)),
        lower_layers_n=lower_layers_n,
        lower_lr_scale=lower_lr_scale,
        beta2=beta2,
        span_channel_off=span_channel_off,
        fused_adamw=fused_adamw,
        train_attention_mask=train_attention_mask,
    )
    return step, tower, budget


def _train(
    *, reader: ShardReader, plan: list[Batch], passes: int, device: str, seed: int,
    hidden: int, heads: int, lr: float, span_weight: float, ledger: Ledger, tag: str,
    quick_reasons: Sequence[str], backbone: Path | None = None,
    optimizer_recipe: str = "bf16",
    checkpoint_dir: Path | None = None, checkpoint_every: int = 0,
    resume_from: object | None = None, deterministic: bool = False,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    n_gpus: int | None = None, usd_per_hour: float | None = None,
    usd_per_gpu_hour: float | None = None, instance: str | None = None,
    approved_by: str = "",
    lower_layers_n: int = 0, lower_lr_scale: float = 1.0, beta2: float = DEFAULT_BETA2,
    permutation: ChoicePermutation | None = None,
    alphabets: Mapping[int, list[tuple[str, ...] | None]] | None = None,
    replay: ReplayPlan | None = None,
    eval_widths: Sequence[int] = (),
    cap_s: float = WALL_CLOCK_CAP_S, no_memorise: bool = False,
    batch_tokens: int | None = None, shuffled_label: Mapping[str, object] | None = None,
    checkpoint_skip_layers: int = 0, fused_adamw: bool = False,
    train_attention_mask: str = "padding",
) -> dict[str, object]:
    """Run ``train_ft`` over ``plan`` repeated ``passes`` times. One optimizer step per batch.

    The batches are the real reader's, re-indexed: ``_train`` requires strictly increasing
    indices inside an epoch because S5 reconstructs a resume position from them, and a
    repeated plan would otherwise go backwards.

    The ported pieces are all off by default and each lands in ``recipe`` only when on, so
    a run that uses none of them hashes exactly as every row before them did:
    ``lower_layers_n``/``lower_lr_scale`` (layer-wise lr, real backbone only), ``beta2``,
    ``permutation`` (with ``alphabets[b]``: plan batch ``b``'s per-row choice letters) and
    ``replay`` (prior_kl toward the base's cached answers).

    ``shuffled_label`` is --shuffled-label's recipe entry: ``plan`` already carries the
    permuted golds (:func:`apply_shuffled_golds`), and the step is built with its span
    channel off, which is the only way either step accepts ``span_weight`` 0.
    """
    width = max(int(b.tokens.shape[1]) for b in plan)
    steps = len(plan) * passes
    pieces = _recipe_pieces(
        lower_layers_n=lower_layers_n, lower_lr_scale=lower_lr_scale, beta2=beta2,
        permutation=permutation, replay=replay, cap_s=cap_s, no_memorise=no_memorise,
        batch_tokens=batch_tokens, shuffled_label=shuffled_label,
        checkpoint_skip_layers=checkpoint_skip_layers, fused_adamw=fused_adamw,
        train_attention_mask=train_attention_mask,
    )
    if checkpoint_skip_layers and backbone is None:
        raise ValueError(
            "checkpoint_skip_layers needs the real backbone: the stand-in is one block with "
            "no checkpointing to be selective about"
        )
    if train_attention_mask != "padding" and backbone is None:
        raise ValueError(
            "train_attention_mask needs the real backbone: the stand-in has no SDPA layer to "
            "switch, so a recipe naming the switch would describe nothing that ran"
        )
    if fused_adamw and (backbone is None or optimizer_recipe != "master"):
        raise ValueError(
            "fused_adamw is built for the real backbone's fp32-master optimizer only; the "
            "recipe would record a fused optimizer this run does not build"
        )
    if permutation is not None and alphabets is None:
        raise ValueError("option permutation needs each plan batch's per-row alphabets")
    recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py", "tag": tag, "device": device,
        "lr": lr, "passes": passes, "batches": len(plan), "width": width,
        # The objective is part of the recipe, the same way `lr` is. Two runs that summed
        # their two channels by different rules are not one protocol, and without this key
        # they hash identically -- which is how a sweep over the objective becomes a single
        # row in the ledger that contradicts itself.
        "span_weight": span_weight,
        # In the recipe for the same reason `span_weight` is: two runs that used different
        # kernels for the same matmul are not one protocol, and the measured spread between
        # them is larger than several of the effects this tool is used to look for.
        "deterministic": deterministic,
        "shard_hash": reader.header.shard_hash(),
    }
    # Which backbone ran, built once and used three times: here, in this function's return
    # value, and -- through that -- in the verdict row's recipe. All three feed a protocol
    # hash, and `_backbone_commit` refuses a recipe that names no backbone at all.
    backbone_keys: dict[str, object] = {}
    # Which path this row's numbers came off: kernels, determinism, checkpointing policy,
    # optimizer implementation. A metric, not recipe keys -- recording it must not move the
    # recipe hash of a run that is otherwise the one every earlier row describes -- and on
    # every ft row, so two rows are never compared without saying what each ran on.
    train_path: dict[str, object] = {
        "deterministic": deterministic,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch": torch.__version__,
        "compile": "off",
    }
    budget: TriState = NotRun(
        reason="the stand-in backbone is one block; memory.py budgets the real tower only"
    )
    if backbone is None:
        if lower_layers_n:
            raise ValueError(
                "layer-wise lr needs --real-backbone: the stand-in is one block with no "
                "'layers.<i>.' to split"
            )
        step: SpanScoringStep = RealFtStep(
            seed=seed, device=device, vocab=int(reader.header.vocab_size), width=width,
            hidden=hidden, heads=heads, lr=lr, span_weight=span_weight, beta2=beta2,
            span_channel_off=shuffled_label is not None,
        )
        train_path["backbone"] = "stand-in: one causal block, no linear attention"
        # Only meaningful for the stand-in, so only recorded for it: under --real-backbone
        # these determine nothing and would still move recipe_hash.
        backbone_keys["hidden"] = hidden
        backbone_keys["heads"] = heads
        what_ran = (
            f"Backbone is a randomly-initialised {hidden}x{heads} single block; this is a "
            "statement about the loop and the data, not an evaluation of any model."
        )
    else:
        spec = optimizer_spec("bf16", optimizer_recipe)
        step, tower, budget = _real_step(
            backbone=backbone, reader=reader, plan=plan, device=device, dtype="bf16",
            spec=spec, attn_implementation=attn_implementation, seed=seed, lr=lr,
            total_steps=steps, span_weight=span_weight, width=width,
            lower_layers_n=lower_layers_n, lower_lr_scale=lower_lr_scale, beta2=beta2,
            eval_widths=eval_widths, span_channel_off=shuffled_label is not None,
            checkpoint_skip_layers=checkpoint_skip_layers, fused_adamw=fused_adamw,
            train_attention_mask=train_attention_mask,
        )
        train_path.update(
            linear_attention_kernels=dict(tower.linear_attention_kernels),
            gradient_checkpointing=tower.gradient_checkpointing,
            checkpoint_skip_layers=list(tower.checkpoint_skip_layers),
            # Read off the step, which is what the training forward consults. "none" runs
            # with SDPA's flash backend as the only one enabled (training_attention).
            train_attention_mask=step.train_attention_mask,
            train_sdpa_backends=(
                ["flash"] if step.train_attention_mask == "none" else "unrestricted"
            ),
            optimizer=type(step.optimizer).__name__,
            optimizer_inner_fused=bool(
                getattr(step.optimizer, "param_groups", [{}])[0].get("fused") or False
            ),
            optimizer_inner_foreach=getattr(step.optimizer, "param_groups", [{}])[0].get(
                "foreach"
            ),
        )
        # `tower.snapshot.name`, not `str(backbone)`: the directory name is the HF revision
        # (refs/main and the snapshot dir agree), while the absolute path is
        # /home/ubuntu/... on the rented box and /Users/bharath/... here. Since this feeds
        # recipe_hash, the path would give the same run two protocol hashes on two machines
        # -- the exact failure --hidden is refused a few lines up to prevent.
        # In the recipe, so a library upgrade that moves the kernel moves recipe_hash
        # rather than moving the numbers quietly. Read off the tower, which reports what the
        # model RESOLVED rather than what was asked for.
        backbone_keys["attn_implementation"] = tower.attn_implementation
        backbone_keys["optimizer_recipe"] = optimizer_recipe
        backbone_keys["backbone_snapshot"] = tower.snapshot.name
        backbone_keys["backbone_params"] = tower.footprint.trainable_params
        backbone_keys["backbone_vocab"] = tower.vocab_size
        recipe["gradient_checkpointing"] = tower.gradient_checkpointing
        what_ran = (
            f"Optimizer recipe {optimizer_recipe!r}: {OPTIMIZER_RECIPES[optimizer_recipe]}. "
            f"Backbone is the REAL text tower from {backbone.name}: "
            f"{tower.n_tensors_loaded} tensors, {tower.footprint.trainable_params:,} "
            f"trainable parameters after the remap to {tower.vocab_size} rows, "
            f"gradient_checkpointing={tower.gradient_checkpointing}, dtype={tower.dtype}."
        )
    if not isinstance(step, SpanScoringStep):  # pragma: no cover - the protocol is structural
        raise TypeError(
            f"{type(step).__name__} does not satisfy SpanScoringStep, so train_ft would "
            "refuse every batch carrying a span row rather than training it"
        )
    # The ported pieces ride with the backbone keys: into this recipe, into the run dict, and
    # through RECIPE_PIECE_KEYS into the verdict and score rows, so every row of a run that
    # used one says so and hashes apart from the rows of runs that did not.
    backbone_keys.update(pieces)
    # Before any optimizer step, while the step is still the base: the prior this run's
    # replay term anchors to. Built (or loaded and checked) here so a mismatched cache is
    # refused before a single step is paid for.
    train_step: SpanScoringStep = step
    cache: PriorCache | None = None
    if replay is not None:
        cache = _prior_cache(
            step, replay, backbone_keys=backbone_keys, device=device, seed=seed,
            resuming=resume_from is not None,
        )
        train_step = PriorKLReplay(
            step, batches=replay.batches, cache=cache, weight=replay.weight,
            every=replay.every,
        )
        what_ran += (
            f" Replay: prior_kl (base->model) at weight {replay.weight} on one replay "
            f"micro-batch every {replay.every}, over {len(cache.row)} cached base rows from "
            f"shard set {replay.shard_hash[:16]}."
        )
    # Validated over the whole plan BEFORE training, so a row whose option block cannot be
    # located refuses the run now rather than at batch 900 of a rented epoch.
    permuted_per_pass = 0
    if permutation is not None:
        if alphabets is None:  # pragma: no cover - refused at the top of this function
            raise ValueError("option permutation needs each plan batch's per-row alphabets")
        for b, batch in enumerate(plan):
            permuted_per_pass += permutation.apply(batch, alphabets[b])[1]
        what_ran += (
            f" Option permutation seed {permutation.seed}: {permuted_per_pass} choice rows "
            "re-permuted per pass on top of the shard set's own shuffle."
        )
    if shuffled_label is not None:
        what_ran += (
            f" SHUFFLED-LABEL CONTROL: {shuffled_label.get('family')} choice golds permuted "
            f"at seed {seed} and the span channel trained at weight {span_weight}; this row "
            "is the control's model, not a measurement of the recipe."
        )
    recipe.update(backbone_keys)
    recorder = _recorder(
        ledger, reader=reader, seed=seed, recipe=recipe, run_kind="ft",
        # None: the block below contains train_ft, so the recorder's own lifetime IS the run.
        wall_clock_s=None,
        # The same estimate the control below is gated on, not a bare rate beside it. It was
        # 0.0 on all 799 rows this project had written, because a float defaulted and nothing
        # passed it; an estimate cannot default, and on a non-local device the recorder
        # refuses a row that omits it.
        cost=_cost(
            device=device, n_gpus=n_gpus, usd_per_hour=usd_per_hour,
            usd_per_gpu_hour=usd_per_gpu_hour, instance=instance, cap_s=cap_s,
        ),
        quick_reasons=quick_reasons,
        notes=(
            f"tools/real_ft_run.py [{tag}] -- qd_train.trainer.train_ft over a shard set "
            f"written by tools/real_tokenizer_pipeline.py with the live Qwen tokenizer. "
            + what_ran
        ),
    )
    recorder.metric("device_budget", budget)
    recorder.metric(
        "train.path",
        Ran(
            passed=True,
            value=json.dumps(train_path, sort_keys=True, separators=(",", ":")),
            detail="what this row ran on; compare rows only where this agrees or says why not",
        ),
    )

    supervised = [ft_supervision(b) for b in plan]
    # Per batch as well as over the plan. The per-batch numbers are what each batch's loss
    # is reduced over and are reported; the bound a RUN can reach is `_plan_floors`, which
    # groups across batches -- see its docstring for the 3x these two differ by on real
    # data, and why comparing a run against the per-batch mean is the wrong test.
    letter_floors = [_letter_floor(b) for b in plan]
    span_floors: list[float | None] = [
        _span_floor(b) if s.span is not None else None
        for b, s in zip(plan, supervised, strict=True)
    ]
    # Both floors are means over the batches that HAVE a row of that kind. Over none, a mean
    # is nan and a substituted 0.0 would be a vacuous claim that the floor is zero -- so
    # neither is recorded as a number: the metric is NotRun and says why.
    measured_letters = [
        f for f, s in zip(letter_floors, supervised, strict=True) if s.n_supervised
    ]
    letter_floor = float(np.mean(measured_letters)) if measured_letters else float("nan")
    measured_spans = [f for f in span_floors if f is not None]
    span_floor = float(np.mean(measured_spans)) if measured_spans else float("nan")
    for name, measured, value, detail in (
        ("corpus.letter_floor", bool(measured_letters), letter_floor,
         "mean over the plan's batches of the conditional entropy of the gold letter given "
         "the exact prefix the model conditions on"),
        ("corpus.span_floor", bool(measured_spans), span_floor,
         "the same for the two gold pointers, over 2K decisions to match the head's mean"),
    ):
        recorder.metric(
            name,
            Ran(passed=True, value=value, detail=detail)
            if measured
            else NotRun(
                reason=(
                    f"no batch in this plan carries a row this floor is defined over, so "
                    f"{name} has nothing to average. A mean over no batches is nan and a "
                    "substituted 0.0 would claim a floor of zero that was never measured."
                )
            ),
        )
    # Whether this run's numbers can be got back. NOT a claim that two runs agreed -- that
    # is a different measurement and one row cannot make it. What a COMPLETED deterministic
    # run does establish is narrower and checkable: torch raises where an op has no
    # deterministic implementation, so reaching the end means every op this model used had
    # one. Off, the honest answer is that nothing was established, which is what NotRun is
    # for -- and the measured consequence is on the record rather than left to be assumed.
    recorder.metric(
        "deterministic_kernels",
        Ran(
            passed=True,
            value="torch.use_deterministic_algorithms(True)",
            detail=(
                "in force for the whole run, set before the tower loaded. torch raises "
                "rather than falling back, so completing the run means no op silently used "
                "a nondeterministic kernel"
            ),
        )
        if deterministic
        else NotRun(
            reason=(
                "this run used torch's default kernels, so its numbers cannot be got "
                "back exactly. Measured on a GH200 at a FIXED seed, eight repeats of one "
                "configuration at 128 steps: without deterministic kernels the final span "
                "loss ranged 0.000000 to 1.505752 with 3 of 8 over the 0.05 bar, and with "
                "them all eight were bit-identical. The kernels are the source. A longer "
                "schedule hides it rather than fixing it -- at 512 steps the same channel "
                "lands in [0.00000, 0.00012] on 5 of 5 seeds without determinism, because "
                "a converged run stops amplifying the perturbation. Read this row's span "
                "number as one draw unless its budget converged."
            )
        ),
    )
    for name, value, detail in (
        ("corpus.plan_batches", len(plan), f"real batches repeated {passes}x"),
        ("corpus.plan_rows", sum(int(b.tokens.shape[0]) for b in plan), "rows in the plan"),
        ("corpus.plan_span_rows", sum(s.span.n_spans for s in supervised if s.span), "span rows"),
        ("corpus.shard_sequences", len(reader), "sequences in the shard set the plan is from"),
        ("corpus.plan_max_width", width, "the widest padded batch this run trains on"),
    ):
        recorder.metric(name, Ran(passed=True, value=value, detail=detail))

    if permutation is not None:
        recorder.metric(
            "option_permutation.rows_per_pass",
            Ran(
                passed=True, value=permuted_per_pass,
                detail=(
                    f"choice rows re-permuted per pass at seed {permutation.seed}, located and "
                    "checked over the whole plan before training"
                ),
            ),
        )
    if cache is not None:
        recorder.metric(
            "replay.base_rows_cached",
            Ran(passed=True, value=int(cache.row.size),
                detail=f"base letter logits over {len(cache.key['letter_ids'])} letters"),
        )

    def source():
        index = 0
        for _ in range(passes):
            for b, batch in enumerate(plan):
                out = dataclasses.replace(batch, index=index)
                # A pure function of (seed, index, row), so a resume regenerates the batches
                # it skips bit-for-bit and consumed_digest still matches.
                if permutation is not None and alphabets is not None:
                    out = permutation.apply(out, alphabets[b])[0]
                yield out
                index += 1

    started = time.monotonic()
    # The hook nobody passed. `HANDOFF/resume-2026-09-20.md` recorded that `on_checkpoint`
    # was referenced in trainer.py, byte_train.py and the tests and nowhere else, so
    # nothing on disk survived a kill -- "which is still nobody's job". One path, rewritten
    # in place: `Checkpoint.write` is atomic (temp file in the target's directory, fsync,
    # rename, fsync the directory), so the previous checkpoint is readable right up to the
    # instant the new one replaces it. Keeping a series would be a retention policy, and
    # this is a resume point rather than a history.
    on_checkpoint = None
    if checkpoint_every and checkpoint_dir is not None:
        on_checkpoint = CheckpointSink(checkpoint_dir / _checkpoint_name(tag, seed, device))

    result = train_ft(
        source(),
        epoch=0,
        step=train_step,
        control=_control(
            steps, device=device, lr=lr, checkpoint_every=checkpoint_every,
            n_gpus=n_gpus, usd_per_hour=usd_per_hour,
            usd_per_gpu_hour=usd_per_gpu_hour, instance=instance,
            approved_by=approved_by, cap_s=cap_s,
        ),
        recorder=recorder,
        on_checkpoint=on_checkpoint,
        resume_from=resume_from,
        on_progress=ProgressLine(
            f"{tag} {device} seed={seed}",
            peak_bytes=torch.cuda.max_memory_allocated if device == "cuda" else None,
        ),
    )
    wall = time.monotonic() - started
    if on_checkpoint is not None:
        on_checkpoint.final(result.checkpoint)
    losses = result.loss_log.losses()
    letter = [x for x in step.letter_log if x > 0.0]
    spans = [x for x in step.span_log if x > 0.0]
    # The two channels' first values are taken from each log independently above, so they
    # can come from DIFFERENT micro-batches: a span-free batch logs a letter loss and a
    # span of 0.0. That is the right pair for "where did each channel start" and the wrong
    # pair for a RATIO, which is a statement about one gradient. `accumulate_span` is the
    # only path that computes both, so the batches where both logs are live are exactly the
    # span batches, and the first of those is the one the ratio is about.
    both = _first_joint_batch(step.letter_log, step.span_log)
    final = _evaluate(step, plan, supervised, letter_floors, span_floors)
    return {
        "tag": tag, "device": device, "seed": seed,
        # Carried out for the same reason as the backbone keys: the verdict row is billed on
        # the same machine at the same rate, and must price itself from this run's estimate
        # rather than build a second one from arguments it does not have.
        "cost": _cost(
            device=device, n_gpus=n_gpus, usd_per_hour=usd_per_hour,
            usd_per_gpu_hour=usd_per_gpu_hour, instance=instance, cap_s=cap_s,
        ),
        # So the verdict row names the same backbone this row does, rather than restating it.
        **backbone_keys,
        "steps_requested": steps,
        "optimizer_steps": result.optimizer_steps,
        "termination": result.termination,
        "micro_batches": result.micro_batches,
        "supervised_tokens": result.supervised_tokens,
        "span_rows": result.span_rows,
        "padding_fraction": round(result.padding_fraction, 6),
        "params": sum(p.numel() for p in step.parameters()),
        "letter_first": letter[0] if letter else float("nan"),
        "letter_last": letter[-1] if letter else float("nan"),
        "letter_floor": letter_floor,
        "span_first": spans[0] if spans else float("nan"),
        "span_last": spans[-1] if spans else float("nan"),
        "span_floor": span_floor,
        "span_weight": span_weight,
        "letter_at_first_joint_batch": both[0] if both else float("nan"),
        "span_at_first_joint_batch": both[1] if both else float("nan"),
        "total_first": losses[0],
        "total_last": losses[-1],
        "wall_clock_s": round(wall, 3),
        "steps_per_s": round(result.optimizer_steps / wall, 2) if wall > 0 else float("inf"),
        "ft_row_id": result.row_id,
        "final": final,
        **(
            {
                "replayed": train_step.replayed,
                "replay_kl_first": train_step.replay_log[0] if train_step.replay_log else None,
                "replay_kl_last": train_step.replay_log[-1] if train_step.replay_log else None,
            }
            if isinstance(train_step, PriorKLReplay)
            else {}
        ),
        **({"option_rows_permuted": permuted_per_pass * passes} if permutation else {}),
        "_step": step,
    }


def _plan_floors(plan: list[Batch], supervised: list[object]) -> tuple[float, float]:
    """The floors over the **whole plan**, which is the only place they are lower bounds.

    ``_letter_floor`` and ``_span_floor`` group by prefix inside one batch, which is right
    for the number a single batch's loss is reduced over -- and wrong as a bound on what a
    run can reach, because one set of parameters answers every batch. Two rows with the same
    prefix and different golds are unfittable whether or not the sampler put them in the same
    batch, and a per-batch computation cannot see the pair once it is split.

    Measured on the real shard set's span rows: the mean per-batch floor is 0.2237 and the
    plan-level floor is 0.6931 -- exactly ``ln 2``, because every one of the 78 span rows has
    a prompt-identical twin with a contradictory gold, and the sampler happened to split
    most of the pairs. A run compared against the per-batch mean would have looked like it
    was failing to converge by 3x when it was sitting on its true floor.

    Same formulas as the two it generalises; the grouping is the whole difference.
    """
    groups: dict[bytes, list[int]] = {}
    n_supervised = 0
    for batch, sup in zip(plan, supervised, strict=True):
        for i in range(batch.tokens.shape[0]):
            if not sup.mask[i].any():  # type: ignore[attr-defined]
                continue
            n_supervised += 1
            at = int(batch.target_index[i])  # type: ignore[index]
            groups.setdefault(batch.tokens[i, : at + 1].tobytes(), []).append(
                int(batch.tokens[i, at + 1])
            )
    # nan, not 0.0: a floor over no supervised row is not a floor of zero. The callers
    # guard on the row count before reading it (`_floor_state`), and nan cannot be
    # mistaken for a measurement if one ever stops.
    letter = (
        sum(_entropy(g) for g in groups.values()) / n_supervised
        if n_supervised
        else float("nan")
    )

    span_groups: dict[bytes, list[tuple[int, int]]] = {}
    n_spans = 0
    for batch, sup in zip(plan, supervised, strict=True):
        if sup.span is None:  # type: ignore[attr-defined]
            continue
        head_plan = plan_span_batch(sup.span)  # type: ignore[attr-defined]
        for k, row in enumerate(sup.span.rows):  # type: ignore[attr-defined]
            at = int(sup.span.query_index[k])  # type: ignore[attr-defined]
            span_groups.setdefault(batch.tokens[row, : at + 1].tobytes(), []).append(
                (int(head_plan.gold_start[k]), int(head_plan.gold_end[k]))
            )
            n_spans += 1
    span = (
        sum(
            _entropy([g[0] for g in group]) + _entropy([g[1] for g in group])
            for group in span_groups.values()
        )
        / (2 * n_spans)
        if n_spans
        else float("nan")
    )
    return letter, span


def _evaluate(
    step: RealFtStep,
    plan: list[Batch],
    supervised: list[object],
    letter_floors: list[float],
    span_floors: list[float | None],
) -> dict[str, object]:
    """Each batch's two losses against **its own** floor, after the last optimizer step.

    The training log's last entry is one batch's loss; comparing it against a mean of
    floors would let a batch sitting above its own floor hide behind batches below theirs.
    This is a no-grad pass over the whole plan, and the claim this tool makes is the worst
    gap it finds.
    """
    plan_letter_floor, plan_span_floor = _plan_floors(plan, supervised)
    letter_weighted = letter_n = 0.0
    span_weighted = span_n = 0.0
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for b, (batch, sup) in enumerate(zip(plan, supervised, strict=True)):
            hidden = step.hidden(batch)
            entry: dict[str, float] = {"batch": float(b)}
            if sup.n_supervised:  # type: ignore[attr-defined]
                loss = float(step._letter_loss(hidden, sup).item())
                entry["letter"] = loss
                entry["letter_floor_in_batch"] = letter_floors[b]
                # `fused_linear_cross_entropy` reduces with "mean" over supervised
                # positions, so the plan's loss is the count-weighted mean of the batches'.
                letter_weighted += loss * sup.n_supervised  # type: ignore[attr-defined]
                letter_n += sup.n_supervised  # type: ignore[attr-defined]
            if sup.span is not None:  # type: ignore[attr-defined]
                plan_b = plan_span_batch(sup.span, device=step.device)  # type: ignore[attr-defined]
                rows_t = torch.as_tensor(sup.span.rows, device=step.device)  # type: ignore[attr-defined]
                loss = float(step.span_head.loss(hidden[rows_t], plan_b).item())
                entry["span"] = loss
                entry["span_floor_in_batch"] = span_floors[b] or 0.0
                span_weighted += loss * sup.span.n_spans  # type: ignore[attr-defined]
                span_n += sup.span.n_spans  # type: ignore[attr-defined]
            rows.append(entry)
    letter_mean = letter_weighted / letter_n if letter_n else float("nan")
    span_mean = span_weighted / span_n if span_n else float("nan")
    return {
        "per_batch": rows,
        "letter_loss": letter_mean,
        "letter_floor_over_the_plan": plan_letter_floor,
        "letter_gap": letter_mean - plan_letter_floor,
        "letter_floor_mean_per_batch": float(np.mean(letter_floors)),
        "span_loss": span_mean,
        "span_floor_over_the_plan": plan_span_floor,
        "span_gap": span_mean - plan_span_floor,
        "span_floor_mean_per_batch": float(
            np.mean([f for f in span_floors if f is not None])
        )
        if any(f is not None for f in span_floors)
        else float("nan"),
        "letter_rows": int(letter_n),
        "span_rows": int(span_n),
    }


def _record_verdict(run: dict[str, object], *, ledger: Ledger, reader: ShardReader,
                    shipped: dict[str, object], defect: dict[str, object],
                    inventory: dict[str, object], batch_chunks: dict[str, int],
                    quick_reasons: Sequence[str], decode_s: float,
                    resolution: TriState) -> str:
    """One row per run for what happened **after** the last optimizer step.

    Separate from the ``ft`` row because ``train_ft`` owns its recorder's context manager and
    writes on exit, so a post-training fact cannot be in it.
    """
    recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py", "tag": f"{run['tag']}-verdict",
        "device": run["device"],
        # Mirrored from the run, not restated: the verdict row has to name the same backbone
        # the ft row named, and under --real-backbone there is no hidden/heads to name.
        **{k: run[k] for k in (*BACKBONE_KEYS, *RECIPE_PIECE_KEYS) if k in run},
        "shard_hash": reader.header.shard_hash(),
    }
    recorder = _recorder(
        ledger, reader=reader, seed=int(run["seed"]), recipe=recipe, run_kind="smoke",
        quick_reasons=quick_reasons,
        # The decode this row reports on, which ran before this function was called -- NOT
        # the parent run's duration. 186 ft rows and 162 verdict rows each claiming the same
        # seconds would sum to twice the GPU time actually spent.
        wall_clock_s=decode_s,
        # The decode was billed on the same machine as the run it reports on, so it prices
        # itself from that run's estimate. The durations differ and must; the rate does not.
        cost=run["cost"],  # type: ignore[arg-type]
        notes=(
            f"tools/real_ft_run.py verdict for ft row {run['ft_row_id']} "
            f"({run['device']} seed={run['seed']}): where the abstention decoded on a real "
            "shard set, read the way crates/qd-runtime/src/answer.rs reads it."
        ),
    )
    by_kind = shipped["by_kind"]  # type: ignore[index]
    with recorder:
        # Pre-registered, and on the VERDICT row because that is the row carrying the
        # comparisons. Against a fixed reference: this tool scores a channel's loss against
        # a floor computed from the corpus, which is a property of the data rather than a
        # quantity with seed noise -- so one variance, not two.
        recorder.metric("sweep_can_resolve", resolution)
        recorder.metric(
            "ft_run_row_id",
            Ran(passed=True, value=run["ft_row_id"], detail="the train_ft row this is of"),
        )
        recorder.metric("objective_channel_balance_at_open", _channel_balance(run))
        for name, key, what in (
            ("letter_channel_first", "letter_first", "letter loss at the first micro-batch"),
            ("letter_channel_last", "letter_last", "letter loss at the last"),
            ("span_channel_first", "span_first", "span loss at the first micro-batch"),
            ("span_channel_last", "span_last", "span loss at the last"),
        ):
            value = float(run[key])  # type: ignore[arg-type]
            recorder.metric(
                name,
                Ran(passed=True, value=value, detail=what)
                if math.isfinite(value)
                else NotRun(
                    reason=(
                        f"no micro-batch in this plan carried this channel, so {what} is "
                        "nan. The ledger records that rather than a 0.0 which would read "
                        "as a channel that was measured and found at zero."
                    )
                ),
            )
        final = run["final"]
        recorder.metric(
            "letter_loss_reached_its_floor",
            _floor_state(
                final["letter_gap"],  # type: ignore[index]
                final["letter_rows"],  # type: ignore[index]
                detail=(
                    f"letter loss {final['letter_loss']:.6f} against the conditional label "  # type: ignore[index]
                    f"entropy of the WHOLE plan {final['letter_floor_over_the_plan']:.6f} "  # type: ignore[index]
                    f"over {final['letter_rows']} supervised rows; bar is floor + "  # type: ignore[index]
                    f"{FLOOR_SLACK}. The mean per-batch floor is "
                    f"{final['letter_floor_mean_per_batch']:.6f} and is not the bound"  # type: ignore[index]
                ),
            ),
        )
        recorder.metric(
            "span_loss_reached_its_floor",
            _floor_state(
                final["span_gap"],  # type: ignore[index]
                final["span_rows"],  # type: ignore[index]
                detail=(
                    f"span loss {final['span_loss']:.6f} against the pointer entropy of the "  # type: ignore[index]
                    f"WHOLE plan {final['span_floor_over_the_plan']:.6f} over "  # type: ignore[index]
                    f"{final['span_rows']} span rows; bar is floor + {FLOOR_SLACK}. "  # type: ignore[index]
                    + _span_floor_cause(
                        float(final["span_floor_over_the_plan"]),  # type: ignore[index]
                        float(final["span_floor_mean_per_batch"]),  # type: ignore[index]
                    )
                ),
            ),
        )
        recorder.metric(
            "prefix_identical_rows_decode_identically",
            Ran(
                passed=int(shipped["prefix_groups_that_decoded_inconsistently"]) == 0,
                value=int(shipped["prefix_groups_that_decoded_inconsistently"]),
                n=int(shipped["prefix_groups"]) - int(
                    shipped["prefix_groups_that_decoded_inconsistently"]
                ),
                n_total=int(shipped["prefix_groups"]),
                detail=(
                    "a deterministic model on identical input must produce identical "
                    f"verdicts. {shipped['rows_in_prefix_groups']} rows sit in "
                    f"{shipped['prefix_groups']} prefix-identical groups, so at most "
                    f"{shipped['best_possible_correct_in_prefix_groups']} of them can ever "
                    "be answered correctly -- the corpus caps span accuracy at one per group"
                ),
            )
            if int(shipped["prefix_groups"])
            else NotRun(reason="no two rows in this plan share a prefix"),
        )
        for kind, counts in sorted(by_kind.items()):  # type: ignore[union-attr]
            recorder.metric(
                f"decoded_as_the_runtime_reads_it.{kind}",
                Ran(
                    passed=counts["correct"] == counts["n"],
                    value=counts["correct"], n=int(counts["correct"]), n_total=int(counts["n"]),
                    detail=(
                        f"{int(counts['abstaining'])} of {int(counts['n'])} {kind} rows "
                        "abstain; verdict read at rows - RESERVED_NOUL_ROWS, never from loss"
                    ),
                ),
            )
            abstaining = int(counts["abstaining"])
            recorder.metric(
                f"abstention_decoded_where_answer_rs_reads_it.{kind}",
                Ran(
                    passed=counts["abstaining_correct"] == abstaining,
                    value=int(counts["abstaining_correct"]),
                    n=int(counts["abstaining_correct"]), n_total=abstaining,
                    detail=(
                        "abstaining rows whose verdict is abstain, read at "
                        "rows - RESERVED_NOUL_ROWS and checked against the head's own "
                        "n_candidates. passed=False here is the corpus, not the layout: "
                        "every span row has a prompt-identical twin with a contradictory "
                        "gold, so the model splits its mass and the argmax is a coin flip. "
                        "GAP-REALFT-CONTRADICTORY-SPAN-SUPERVISION"
                    ),
                )
                if abstaining
                else NotRun(
                    reason=(
                        f"this shard set holds no {kind} row whose gold is noul: "
                        f"{inventory['letter_rows_whose_gold_is_noul']} of "
                        f"{inventory['letter_rows']} letter rows abstain. Of the five "
                        "families its two sources produce, only qa.answer_span emits a noul "
                        "gold and it is a span slot. 0 of 0 abstentions decoded correctly is "
                        "not a pass -- the abstain row of a "
                        f"{kind} slot was supervised zero times."
                    )
                ),
            )
        for kind in ("choice", "score"):
            if kind not in by_kind:  # type: ignore[operator]
                continue
            recorder.metric(
                f"noul_probability_after_training.{kind}",
                Ran(
                    passed=False,
                    value=round(float(by_kind[kind]["noul_probability_mean"]), 8),  # type: ignore[index]
                    n=int(by_kind[kind]["n"]),  # type: ignore[index]
                    n_total=int(by_kind[kind]["n"]),  # type: ignore[index]
                    detail=(
                        "mean softmax mass on the abstain row, over this slot kind's own "
                        "rendered rows, after training on a corpus that never supervises it. "
                        "passed=False is the finding, not a regression: cross-entropy on the "
                        "other letters drives this toward zero, so the corpus trains the "
                        "model AGAINST abstaining. GAP-REALFT-LETTER-ABSTENTION-NEVER-SUPERVISED"
                    ),
                ),
            )
        recorder.metric(
            "reordering_letters_does_not_move_the_span_abstention",
            Ran(
                passed=_counterfactual_holds(shipped, defect),
                value=(
                    f"noul-last {shipped['span_abstaining_decoded_as_abstain']}/"
                    f"{shipped['span_abstaining_rows']}, noul-first "
                    f"{defect['span_abstaining_decoded_as_abstain']}/"
                    f"{defect['span_abstaining_rows']}"
                ),
                detail=(
                    "one trained model read under two row orders. A span abstention is a "
                    "POINTER row, not a letter row, so reordering the rendered letters must "
                    "leave it exactly where it was; an arm that moved it differs in more "
                    "than the thing under test. This metric asserts that and nothing else -- "
                    "in particular it does NOT demand that every abstaining row decode as "
                    "abstain, which this corpus forbids (see "
                    "abstention_decoded_where_answer_rs_reads_it.span). "
                    "_counterfactual_holds is the single owner and main() calls the same "
                    "function for the exit status, so the row and the exit status cannot "
                    "disagree -- which is what GAP-FT-TOY-VERDICT-ROW-CONTRADICTED-ITS-OWN-RUN "
                    "was about, reproduced here before this fix."
                ),
            ),
        )
        recorder.metric(
            "letter_row_order_decides_the_letter_answer",
            NotRun(
                reason=(
                    "the counterfactual the FT lane's toy run makes -- read one trained model "
                    "under noul-first and watch every letter abstention stop being an "
                    "abstention -- cannot be run on this corpus, because it holds "
                    f"{inventory['letter_rows_whose_gold_is_noul']} letter abstentions of "
                    f"{inventory['letter_rows']} letter rows. Nothing to flip is not a pass. "
                    "GAP-REALFT-LETTER-ABSTENTION-NEVER-SUPERVISED"
                )
            )
            if not int(shipped["letter_abstaining_rows"])
            else Ran(
                passed=int(defect["letter_abstaining_decoded_as_abstain"]) == 0
                and shipped["letter_abstaining_decoded_as_abstain"]
                == shipped["letter_abstaining_rows"],
                value=(
                    f"noul-last {shipped['letter_abstaining_decoded_as_abstain']}/"
                    f"{shipped['letter_abstaining_rows']}, noul-first "
                    f"{defect['letter_abstaining_decoded_as_abstain']}/"
                    f"{defect['letter_abstaining_rows']}"
                ),
                detail="every letter abstention must stop being one under the noul-first order",
            ),
        )
        contra = inventory["contradictions"]
        recorder.metric(
            "corpus_supervision_is_self_consistent",
            Ran(
                passed=int(contra["contradicting_rows"]) == 0,  # type: ignore[index]
                value=int(contra["contradicting_rows"]),  # type: ignore[index]
                n=int(contra["contradicting_rows"]),  # type: ignore[index]
                n_total=int(contra["sequences"]),  # type: ignore[index]
                detail=(
                    f"{contra['contradicting_rows']} sequences in "  # type: ignore[index]
                    f"{contra['contradicting_groups']} groups share a prefix up to "  # type: ignore[index]
                    "target_index and carry different golds, so a causal model cannot fit "
                    "both and the corpus's entropy floor is above zero by construction. "
                    "dedupe sees these pairs at Jaccard 1.0 and keeps them by design "
                    "(same repo_key, so no leak); nothing downstream reads that as a "
                    "supervision signal. GAP-REALFT-CONTRADICTORY-SPAN-SUPERVISION"
                ),
            ),
        )
        recorder.metric(
            "letter_channel_projection_waste",
            Ran(
                passed=False,
                value=int(batch_chunks["total"]) - int(batch_chunks["live"]),
                n=int(batch_chunks["live"]),
                n_total=int(batch_chunks["total"]),
                detail=(
                    "position chunks fused_linear_cross_entropy projects over one real "
                    "epoch, against the chunks that hold a supervised position. FT masks in "
                    "one position per non-span row and _FusedLinearCE.forward walks every "
                    "chunk regardless, so the letter channel pays a CPT-sized projection. "
                    "passed=False is the measurement, not a regression. "
                    "GAP-REALFT-FUSED-CE-PROJECTS-FULLY-MASKED-CHUNKS"
                ),
            ),
        )
        recorder.metric("shard_padding_waste", reader.padding_waste())
        recorder.metric("shard_coverage", reader.coverage)
        recorder.metric("shard_span_mapping_decode_verified", reader.span_check)
        recorder.metric("shard_remap_matches_header", reader.checks["shard_remap_matches_header"])
        recorder.noul_rate = NotRun(
            reason=(
                "a memorisation run over a subsample of the training split has no held-out "
                "population to compute an abstain rate over -- and this corpus supervises no "
                "letter-channel abstention at all, so a rate computed on it would describe "
                "the corpus, not the model"
            )
        )
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id


# --- scoring on rows the model did not train on ----------------------------------------


@dataclasses.dataclass(frozen=True)
class ValSet:
    """The val shard set, relabelled and batched, ready for :func:`_decode`.

    GAP-RUNG3-NOTHING-SCORES-A-TRAINED-MODEL-ON-ROWS-IT-DID-NOT-TRAIN-ON: every number this
    tool wrote was about rows the model trained on. ``real_tokenizer_pipeline.py
    --val-shards`` writes the val split under the train set's remap; this is what reads it.
    """

    reader: ShardReader
    labels: list[Label]
    plan: list[Batch]
    labels_for: dict[int, list[Label]]
    letter_id: dict[str, int]


def ft_splits(
    *,
    commitpackft: Path | None,
    max_pairs: int,
    rev: str,
    config: DataConfig,
    defect_class: Path | None = None,
    defect_download: Path | None = None,
    defect_max_rows: int | None = None,
    repo_history: bool = True,
    general_record: Path | None = None,
    general_max_rows: int | None = None,
    replay_partition: bool = False,
    defect_noul: Path | None = None,
) -> dict[str, list[DataRow]]:
    """Every split of the corpus this tool's shard sets were built from, by split name.

    The same "which rows" ``tools/real_tokenizer_pipeline.py``'s ``run`` answered, from the
    same functions in the same order, so a set built from the commitpackft download is
    relabelled from the download and a set built with ``--defect-class`` gets its
    ``code.defect_class`` rows back through the same ``load_defect_rows`` call (same
    corpus, licence download and sha256-ordered cap). ``rev`` must already be resolved
    (``repo_git.resolve_rev``). One owner for this rebuild: ``main``, :func:`ft_split_rows`
    (``tools/ft_linear_control.py``) and ``tools/replay_decontam.py`` (which needs the
    ``heldout`` split as well) all call it, rather than each carrying a copy that could
    drift from the shard set.

    ``repo_history=False`` mirrors the pipeline's ``--no-repo-history``, through the same
    ``base_sources`` the pipeline itself calls.

    ``general_record`` / ``general_max_rows`` mirror the pipeline's ``--general-record`` /
    ``--general-max-rows``: the general families come back through the pipeline's own
    ``general_rows`` (same sha-checked caches, same per-file cap), inserted after the base
    and defect sources exactly as ``run`` inserts them (real SQuAD replaces the repository
    prose stand-in under the same source), and ``build_mixture`` gets the record's CLINC
    domain map, without which it builds no two-stage CLINC rows at all.
    ``replay_partition`` mirrors the pipeline's ``--replay-shards``: its
    ``split_off_replay`` takes the replay-only rows out of the gold ``train`` split, so
    they are never gold-trained here either. val and held-out are untouched by it.
    ``defect_noul`` mirrors the pipeline's ``--defect-noul``: the same ``load_defect_rows``
    call appends the noul corpus's rows after the defect corpus's, in the same order.

    ``build_mixture`` runs at the pipeline's ``PIPELINE_MAX_CONSISTENCY_ROWS``, not the
    library default: the two agree below 250,000 rows and build different row sets above
    it. A consistency pass that could not run is refused, as the pipeline refuses it: no
    shard set was ever written from such a mixture, so this rebuild has diverged from it.
    """
    if defect_class is None and (
        defect_download is not None or defect_max_rows is not None or defect_noul is not None
    ):
        raise ValueError(
            "defect_download/defect_max_rows/defect_noul without defect_class read nothing"
        )
    if general_record is None and (general_max_rows is not None or replay_partition):
        raise ValueError(
            "general_max_rows/replay_partition without general_record read nothing: the "
            "replay slice is drawn from the general families' training rows"
        )
    import real_tokenizer_pipeline as pipeline

    from qd_data.config import SPLITS
    from qd_data.dedupe import dedupe
    from qd_data.mixture import build_mixture
    from qd_data.split import split

    raw: dict[str, list[Any]] = dict(
        pipeline.base_sources(
            repo_history=repo_history, commitpackft=commitpackft, max_pairs=max_pairs,
            blank_line_runs=False, rev=rev,
        ).raw
    )
    if defect_class is not None:
        from qd_data.defect_class import DEFECT_SOURCE_ID, load_defect_rows

        load = load_defect_rows(
            defect_class,
            download_root=(
                pipeline.DEFAULT_DEFECT_DOWNLOAD if defect_download is None else defect_download
            ),
            config=config, repo_root=REPO, max_rows=defect_max_rows, noul_dir=defect_noul,
        )
        raw[DEFECT_SOURCE_ID] = list(load.rows)
    clinc_domain_map = None
    if general_record is not None:
        general = pipeline.general_rows(
            general_record,
            max_rows_per_file=(
                pipeline.DEFAULT_GENERAL_MAX_ROWS if general_max_rows is None
                else general_max_rows
            ),
        )
        for dataset, rows in general.raw.items():
            raw[dataset] = list(rows)
        clinc_domain_map = general.clinc_domain_map
    mixture = build_mixture(
        raw, config=config, clinc_domain_map=clinc_domain_map,
        max_consistency_rows=pipeline.PIPELINE_MAX_CONSISTENCY_ROWS,
    )
    if isinstance(mixture.prompt_consistency, NotRun):
        raise SystemExit(
            "build_mixture could not run its consistency pass on the rebuild, and the "
            "pipeline refuses to write a shard set from such a mixture, so this rebuild is "
            f"not the one any shard set was written from: {mixture.prompt_consistency.reason}"
        )
    # Signed by crates/qd-prep (QD_PREP_BIN), byte-identical to qd_data.minhash, which is its
    # parity oracle (pipeline.native_minhash): the signatures were most of this rebuild's time.
    with pipeline.native_minhash(mixture.rows, config=config):
        report = dedupe(list(mixture.rows), config=config)
        split_report = split(report, config=config)
    if replay_partition:
        split_report, _replay, _partition = pipeline.split_off_replay(
            split_report, seed=config.seed
        )
    return {name: list(split_report.rows_by_split.get(name, ())) for name in SPLITS}


def ft_split_rows(
    *,
    commitpackft: Path | None,
    max_pairs: int,
    rev: str,
    config: DataConfig,
    defect_class: Path | None = None,
    defect_download: Path | None = None,
    defect_max_rows: int | None = None,
    repo_history: bool = True,
    general_record: Path | None = None,
    general_max_rows: int | None = None,
    replay_partition: bool = False,
    defect_noul: Path | None = None,
) -> tuple[list[DataRow], list[DataRow]]:
    """``(train_rows, val_rows)``: exactly the two splits ``main`` trains and scores on."""
    splits = ft_splits(
        commitpackft=commitpackft, max_pairs=max_pairs, rev=rev, config=config,
        defect_class=defect_class, defect_download=defect_download,
        defect_max_rows=defect_max_rows, repo_history=repo_history,
        general_record=general_record, general_max_rows=general_max_rows,
        replay_partition=replay_partition, defect_noul=defect_noul,
    )
    return splits["train"], splits["val"]


#: Where ``tools/real_tokenizer_pipeline.py`` writes the train manifest, under its --out.
TRAIN_MANIFEST: Final[str] = "data/pool/train.json"
#: Where it writes the replay slice's manifest -- only under its ``--replay-shards``, whose
#: ``split_off_replay`` took those rows out of the gold train split.
REPLAY_MANIFEST: Final[str] = "data/pool/train-replay.json"


def check_defect_source(out: Path, *, defect_class: Path | None) -> None:
    """Refuse a rebuild that disagrees with the shard set about the defect-class source.

    Read off the train manifest the pipeline wrote beside the shards: ``mixture.n_input``
    is the row count each source actually FED this build. Not ``admitted_source_ids``,
    which lists every source the licence registry admits whether or not this build read
    it -- measured 2026-09-29, a plain 100-pair build lists 'qd-mutate/commitpackft' there
    with no defect row in it, and the first version of this check refused it.

    A set built with ``--defect-class`` and relabelled without it would rebuild a different
    row set; the reverse adds rows the shards never held. Both are refused here, before any
    tower loads.
    """
    from qd_data.defect_class import DEFECT_SOURCE_ID

    path = out / TRAIN_MANIFEST
    if not path.is_file():
        raise SystemExit(
            f"{path} is absent, so which sources this shard set was built from cannot be "
            "checked against this run's --defect-class. Refusing rather than guessing."
        )
    n_input = json.loads(path.read_text(encoding="utf-8")).get("mixture", {}).get("n_input")
    if not isinstance(n_input, dict):
        raise SystemExit(
            f"{path} carries no mixture.n_input, so which sources fed this build cannot be "
            "checked against this run's --defect-class"
        )
    built_with = int(n_input.get(DEFECT_SOURCE_ID, 0)) > 0
    if built_with and defect_class is None:
        raise SystemExit(
            f"{path}: {n_input[DEFECT_SOURCE_ID]} {DEFECT_SOURCE_ID!r} rows fed this shard set "
            "-- it was built with tools/real_tokenizer_pipeline.py --defect-class, and "
            "without the same --defect-class here its labels would be rebuilt from a "
            "different row set"
        )
    if defect_class is not None and not built_with:
        raise SystemExit(
            f"--defect-class was given but no {DEFECT_SOURCE_ID!r} row fed {path}: this shard "
            "set was built without it"
        )


#: Sources only ``--general-record`` supplies. A set fed by any of them is relabelled only
#: with the same ``--general-record``, which ``ft_splits`` reads through the pipeline's own
#: ``general_rows``.
GENERAL_ONLY_SOURCES: Final[tuple[str, ...]] = (
    "cais/mmlu", "tau/commonsense_qa", "clinc/clinc_oos",
)


@dataclasses.dataclass(frozen=True)
class CorpusFacts:
    """What the train manifest beside the shards says about the corpus, for rule 8."""

    #: ``None`` when the snapshot ran; else ``NotRun``'s reason (a capped read, a family
    #: with no rows, an unchecked consistency pass).
    snapshot_not_run: str | None
    #: ``{source: rows}`` read from this repository's git history.
    history_rows: dict[str, int]


def general_record_datasets(record: Path | None) -> frozenset[str] | None:
    """The datasets a fetch record names, read through the pipeline's own record parser --
    before the rebuild, which sha-checks every cache. ``None`` when no record was given."""
    if record is None:
        return None
    import real_tokenizer_pipeline as pipeline

    _raw, entries = pipeline.fetch_record_entries(record)
    return frozenset(str(e["dataset"]) for e in entries)


def corpus_facts(
    out: Path, *, data_snapshot_hash: str, repo_history: bool, commitpackft: Path | None,
    general_datasets: frozenset[str] | None = None, replay_partition: bool = False,
) -> CorpusFacts:
    """Read the train manifest the pipeline wrote, and refuse what this rebuild cannot match.

    Refused: a manifest that is absent, that names a different ``data_snapshot_hash`` than
    the shard header (it is not this set's manifest), whose repository-history rows
    disagree with ``repo_history`` -- a set built with ``--no-repo-history`` and relabelled
    with history would rebuild rows the shards never held, and the reverse would drop rows
    they did -- and any disagreement about the general families or the replay slice:

    * ``general_datasets`` is what this run's ``--general-record`` names
      (:func:`general_record_datasets`), ``None`` without one. A set fed by a family only a
      record supplies is refused without a record, and with a record that does not name
      that family; a record none of whose datasets fed the set is refused too. A dataset
      the record names may legitimately feed nothing (a split ``REFUSED_READS`` forbids),
      so that alone is not a refusal.
    * ``rajpurkar/squad_v2`` counts as repository history only when the record does not
      supply it: the pipeline replaces the prose stand-in with real SQuAD.
    * ``replay_partition`` must agree with the replay manifest the pipeline writes beside
      the train manifest only under its ``--replay-shards``: without the partition the
      rebuild would gold-train rows the set keeps for replay, and with it on a set that
      had none it would drop rows the shards hold.
    """
    path = out / TRAIN_MANIFEST
    if not path.is_file():
        raise SystemExit(f"{path} is absent, so the corpus behind this shard set cannot be read")
    raw = json.loads(path.read_text(encoding="utf-8"))
    recorded = raw.get("data_snapshot_hash")
    if recorded != data_snapshot_hash:
        raise SystemExit(
            f"{path} records data_snapshot_hash {recorded!r} but the shard header pins "
            f"{data_snapshot_hash!r}: it is not the manifest this shard set was written from"
        )
    status = parse_tristate(raw.get("status"), field=f"{path}:status")
    n_input = raw.get("mixture", {}).get("n_input")
    if not isinstance(n_input, dict):
        raise SystemExit(f"{path} carries no mixture.n_input, so its sources cannot be read")
    general = sorted(s for s in GENERAL_ONLY_SOURCES if int(n_input.get(s, 0)) > 0)
    if general and general_datasets is None:
        raise SystemExit(
            f"{path}: {general} fed this shard set -- it was built with --general-record, and "
            "without the same --general-record here the general families are not rebuilt, "
            "so its labels cannot be reconstructed. Refusing rather than pairing labels to a "
            "different row set."
        )
    if general_datasets is not None:
        unnamed = sorted(set(general) - general_datasets)
        if unnamed:
            raise SystemExit(
                f"{path}: {unnamed} fed this shard set but this run's --general-record does "
                f"not name them (it names {sorted(general_datasets)}): not the record the set "
                "was built from"
            )
        # Only the families a record alone supplies can tell whether it fed the set: a SQuAD
        # count is the same whether the rows came from the record or from this repository's
        # prose stand-in. A SQuAD-only record is therefore not refused here; pair_labels,
        # which compares row ids, is the refusal for it.
        record_only = general_datasets & set(GENERAL_ONLY_SOURCES)
        if record_only and not any(int(n_input.get(d, 0)) > 0 for d in record_only):
            raise SystemExit(
                f"--general-record names {sorted(general_datasets)} but none of them fed "
                f"{path}: this shard set was built without it"
            )
    built_with_replay = (out / REPLAY_MANIFEST).is_file()
    if built_with_replay and not replay_partition:
        raise SystemExit(
            f"{out / REPLAY_MANIFEST} exists: this shard set was built with the pipeline's "
            "--replay-shards, whose replay-only rows left the gold train split. Pass "
            "--replay-partition, or the rebuild gold-trains rows the set keeps for replay."
        )
    if replay_partition and not built_with_replay:
        raise SystemExit(
            f"--replay-partition was given but {out / REPLAY_MANIFEST} is absent: this shard "
            "set was built without the pipeline's --replay-shards, and partitioning the "
            "rebuild would drop rows the shards hold"
        )
    squad_from_history = general_datasets is None or "rajpurkar/squad_v2" not in general_datasets
    history_sources = (("rajpurkar/squad_v2",) if squad_from_history else ()) + (
        ("bigcode/commitpackft",) if commitpackft is None else ()
    )
    history_rows = {s: int(n_input[s]) for s in history_sources if int(n_input.get(s, 0)) > 0}
    if history_rows and not repo_history:
        raise SystemExit(
            f"{path}: {history_rows} rows from this repository's history fed this shard set, "
            "but --no-repo-history was passed; the rebuild would not have them"
        )
    # With --commitpackft and a record that supplies SQuAD, the rebuild reads no history row
    # whatever --no-repo-history says, so the flag decides nothing and is not checked.
    if repo_history and history_sources and not history_rows:
        raise SystemExit(
            f"{path}: no row from this repository's history fed this shard set -- it was built "
            "with --no-repo-history. Pass the same flag here, or the rebuild adds rows the "
            "shards never held."
        )
    return CorpusFacts(
        snapshot_not_run=status.reason if isinstance(status, NotRun) else None,
        history_rows=history_rows,
    )


def quick_reasons(
    *, tag: str, device: str, real_backbone: bool, corpus: CorpusFacts,
    termination: str | None = None, memorise_detail: str = "",
) -> list[str]:
    """Every rule-8 reason that stands for one row, from the run's facts. Empty means none.

    Rule 8: *"Fewer than 3 seeds, a truncated schedule or a subsample is marked quick."*

    * **Subsample.** The memorisation arm (a subset of batches, repeated), a data snapshot
      that is ``NotRun`` (a capped read is a sample; so is a mixture with a family that
      produced nothing), and rows from this repository's own history -- the precedent in
      ``real_tokenizer_pipeline.quick_reason_for`` and ``rung0_real_run.quick_reason_for``,
      both of which call a corpus drawn from this repository a subsample of the plan's pool.
    * **Truncated schedule.** ``termination`` other than ``steps_exhausted``, for the rows
      written after the loop (verdict, eval). The ``ft`` row's own truncation is added by
      ``RunRecorder`` from ``train.termination``, since it is written inside the loop.
    * **Not the campaign's run**, stated beside rule 8 because the tool made every row quick
      before this existed and these conditions were among the reasons: a device outside
      :data:`CAMPAIGN_DEVICES` (a Mac or CPU run is a smoke of the path) and the stand-in
      backbone (a statement about the loop, not about any model).
    * **Seeds** are not decided here. A campaign unit runs one seed, so a row cannot see its
      family. The family's seed count is enforced where the family is visible:
      ``Ledger.promotion_verdict`` (three distinct seeds under one protocol-minus-seed) and
      the campaign driver's ``gate.min_seeds``.

    Only ever a list of reasons to be quick; nothing here clears a flag a caller set.
    """
    reasons: list[str] = []
    if tag == "memorise":
        reasons.append(
            "the memorisation arm is a subsample by construction"
            + (f": {memorise_detail}" if memorise_detail else "")
        )
    if termination is not None and termination != "steps_exhausted":
        reasons.append(
            f"train.termination is {termination!r}, not 'steps_exhausted': a truncated schedule"
        )
    if corpus.snapshot_not_run is not None:
        reasons.append(
            "the shard set's data snapshot is NotRun, so the corpus is a capped or partial "
            f"sample: {corpus.snapshot_not_run}"
        )
    if corpus.history_rows:
        reasons.append(
            f"the corpus holds rows drawn from this repository's own history "
            f"({corpus.history_rows}) rather than the plan's pool, which rule 8 counts as a "
            "subsample"
        )
    if device not in CAMPAIGN_DEVICES:
        reasons.append(
            f"device {device!r} is not a campaign device ({sorted(CAMPAIGN_DEVICES)}): a "
            "smoke of the path, not the measurement"
        )
    if not real_backbone:
        reasons.append(
            "the backbone is the randomly-initialised stand-in, a statement about the loop "
            "and the data rather than about any model"
        )
    return reasons


def merge_letter_ids(train: dict[str, int], val: dict[str, int]) -> dict[str, int]:
    """One ``letter -> id`` map over both sets, or a refusal.

    Each map is read off its own shard set by correspondence, so a letter only ever used
    as a gold in one set is known only there -- and a val row whose options include it
    still has to be decoded. Both sets share one remap, so a letter that reads as two ids,
    or two letters as one id, means the sets disagree about which sequence is which row.
    """
    merged = dict(train)
    for letter, token in val.items():
        if merged.setdefault(letter, token) != token:
            raise SystemExit(
                f"letter {letter!r} is token {merged[letter]} in the train set and {token} "
                "in the val set, which share one remap. One of the two relabellings is on "
                "the wrong sequences."
            )
    doubled = {t for t, n in collections.Counter(merged.values()).items() if n > 1}
    if doubled:
        raise SystemExit(f"two letters share a token id across train and val: {sorted(doubled)}")
    return merged


def pair_labels(reader: ShardReader, labels: list[Label], *, require_index: bool) -> list[Label]:
    """``labels`` in the shard set's sequence order, paired by ``(row_id, slot_name)``.

    When the set carries ``sequence_index.json`` (``reader.sequence_index``), sequence ``i``
    gets exactly the label whose id the index names -- never a position in a reconstructed
    order -- and any disagreement is a refusal: an index id this rebuild has no label for,
    a rebuilt label the writer neither wrote nor excluded, a duplicate id, or a slot kind
    that differs. A set written before the index existed has none; it is only accepted when
    ``require_index`` is false, and then ``_inventory`` still checks the reconstructed order
    against ``supervision.npz``. A ``--defect-class`` set requires the index: its writer
    drops span slots at tokenization, which no tokenizer-free rebuild can predict.
    """
    index = getattr(reader, "sequence_index", None)
    if index is None:
        if require_index:
            raise SystemExit(
                f"{reader.root} carries no sequence index, so its labels could only be paired "
                "by reconstructing the writer's order -- which a defect-class set breaks at "
                "tokenization. Rebuild the shard set with the current writer."
            )
        return labels
    from qd_train.shards import GOLD_ROLE

    # A replay_only set's rows are anchored to the base's answers, not trained against a
    # gold, so pairing gold labels to it would supervise rows that were never meant to be.
    role = getattr(index, "role", None)
    if role != GOLD_ROLE:
        raise SystemExit(
            f"{reader.root}: sequence index role is {role!r}, not {GOLD_ROLE!r}; its rows are "
            "not trained against a gold label and cannot be relabelled for FT"
        )
    by_id: dict[tuple[str, str], Label] = {}
    for label in labels:
        key = (label.row_id, label.slot_name)
        if key in by_id:
            raise SystemExit(f"the rebuild produced {key} twice")
        by_id[key] = label
    missing = [key for key in index.sequences if key not in by_id]
    if missing:
        raise SystemExit(
            f"{reader.root}: {len(missing)} sequence(s) name a (row_id, slot_name) this "
            f"rebuild does not have, first {missing[:3]}. The shard set was not built from "
            "these rows."
        )
    excluded = {(e.row_id, e.slot_name) for e in index.excluded}
    stray = sorted(set(by_id) - set(index.sequences) - excluded)
    if stray:
        raise SystemExit(
            f"{reader.root}: the rebuild has {len(stray)} slot(s) the writer neither wrote nor "
            f"excluded, first {stray[:3]}. The shard set was not built from these rows."
        )
    paired = [by_id[key] for key in index.sequences]
    wrong_kind = [
        (i, key) for i, (key, kind) in enumerate(zip(index.sequences, index.slot_kinds,
                                                      strict=True))
        if by_id[key].slot_kind != kind
    ]
    if wrong_kind:
        raise SystemExit(f"{reader.root}: slot kinds disagree at {wrong_kind[:3]}")
    return paired


@dataclasses.dataclass(frozen=True)
class TrainRelabel:
    """The train shard set relabelled from the rebuild: what training needs before a tower."""

    labels: list[Label]
    inventory: dict[str, object]
    #: ``letter -> post-remap id``: read off the train golds by correspondence, then, with a
    #: tokenizer.json, every option letter read off its vocabulary and cross-checked.
    letter_id: dict[str, int]


def relabel_train(
    reader: ShardReader, rows: list[DataRow], *, config: DataConfig, require_index: bool,
    tokenizer_json: Path | None,
) -> TrainRelabel:
    """Relabel, pair, inventory and letter the train split against the train shard set.

    Paired by id against the writer's sequence index where the set has one; see
    :func:`pair_labels` for why a ``--defect-class`` set must. A ``--general-record`` set
    must too: the writer that built it records one, and the replay partition moves gold rows
    within the train split, so a reconstructed order is never what such a set is checked
    against. Every letter a row OFFERS needs an id to be decoded, not only the letters some
    row has as its gold: with ``tokenizer_json`` the vocabulary supplies them, cross-checked
    against the gold ids.
    """
    labels, excluded = _labels(rows, config=config)
    labels = pair_labels(reader, labels, require_index=require_index)
    inventory = _inventory(reader, labels, excluded)
    inventory["contradictions"] = _contradictions(reader, labels)
    letter_id = _letter_ids(reader, labels)
    if tokenizer_json is not None:
        letter_id = vocab_letter_ids(reader, tokenizer_json=tokenizer_json, letter_id=letter_id)
    return TrainRelabel(labels=labels, inventory=inventory, letter_id=letter_id)


def letters_no_val_gold_confirms(val: ValSet, ood: OodSuite) -> list[str]:
    """Letters a val or OOD row offers -- so may decode to -- that no val row has as its gold.

    A letter's id comes from the tokenizer's vocabulary, and the vocabulary is trusted only
    because it agrees with the ids the shard sets carry for the letters that ARE golds:
    ``merge_letter_ids`` (the val golds) and ``vocab_letter_ids`` (the train golds). Without
    the train relabel, only the val golds confirm it, so a letter offered and never a val
    gold would be decoded on the vocabulary's word alone. Empty means every offered letter
    was confirmed by the val set's own correspondence.
    """
    offered = {x for label in val.labels if label.slot_kind != SLOT_SPAN for x in label.letters}
    if ood.val is not None:
        offered |= {
            x for label in ood.val.labels if label.slot_kind != SLOT_SPAN for x in label.letters
        }
    return sorted(offered - {label.gold_letter for label in val.labels})


def open_val_reader(
    out: Path, *, config: DataConfig, rev: str, train: ShardReader
) -> ShardReader:
    """The val shard set's reader, refused unless it is there and shares ``train``'s remap."""
    val_dir = out / "shards" / "val"
    if not (val_dir / HEADER_NAME).exists():
        raise SystemExit(
            f"--score-val found no val shard set at {val_dir}. Build one with "
            "tools/real_tokenizer_pipeline.py --val-shards, which also builds the remap over "
            "the val rows -- a set scored under a remap that dropped their tokens cannot be "
            "written at all (GAP-REMAP-CANNOT-ENCODE-THE-ROWS-IT-WAS-NOT-BUILT-FROM)."
        )
    reader = ShardReader(val_dir, config=config, repo_root=out, expect_rev=rev)
    reader.header.require_gate_population(where=str(val_dir))
    if reader.header.remap_hash != train.header.remap_hash:
        raise SystemExit(
            f"the val set's remap {reader.header.remap_hash[:16]} is not the train set's "
            f"{train.header.remap_hash[:16]}: the same id would name a different embedding "
            "row in the model being scored"
        )
    return reader


def val_plan(reader: ShardReader, *, config: DataConfig) -> list[Batch]:
    """The val set's batches, in the one order every scoring of it uses: the widest bucket
    per batch, the protocol seed, epoch 0."""
    return list(
        reader.batches(batch_tokens=int(max(reader.header.buckets)), seed=config.seed, epoch=0)
    )


@dataclasses.dataclass(frozen=True)
class ValPlan:
    """The val set's batches without its labels: all that sizes a scoring step.

    What the needle worker opens (:func:`needle_worker_main`): it decodes the needle suite it
    is handed, never a val row, so it reads the val set's shapes and not its golds.
    """

    reader: ShardReader
    plan: list[Batch]


def open_val_set(
    out: Path, *, config: DataConfig, rev: str, rows: list[DataRow], train: ShardReader,
    letter_id: dict[str, int], require_index: bool = False,
) -> ValSet:
    """Read, relabel and batch the val shard set, refusing anything that would misscore it.

    Every refusal is decided here, before a tower loads: a val set written under another
    remap indexes different embedding rows for the same ids, a relabelling that disagrees
    with ``supervision.npz`` puts every label on the wrong sequence (``_inventory``), and
    an option letter no set ever used as a gold has no id to decode it with.
    """
    reader = open_val_reader(out, config=config, rev=rev, train=train)
    labels, excluded = _labels(rows, config=config)
    labels = pair_labels(reader, labels, require_index=require_index)
    _inventory(reader, labels, excluded)
    merged = merge_letter_ids(letter_id, _letter_ids(reader, labels))
    unknown = sorted(
        {x for label in labels if label.slot_kind != SLOT_SPAN for x in label.letters}
        - set(merged)
    )
    if unknown:
        # Counted, not refused: _decode skips exactly the rows offering these and reports
        # them as rows_not_decoded, which _record_score puts on the eval row beside the
        # rows that were. Pass the tokenizer (--real-backbone, or --tokenizer-json) and
        # main reads every offered letter's id off the vocabulary instead.
        print(
            f"val rows offer letter(s) {unknown} with no known token id (never a gold in "
            "either set, and no tokenizer.json given): those rows will not be decoded"
        )
    plan = val_plan(reader, config=config)
    return ValSet(
        reader=reader,
        labels=labels,
        plan=plan,
        labels_for=_labels_by_batch(
            reader, plan, labels, config=config,
            batch_tokens=int(max(reader.header.buckets)),
        ),
        letter_id=merged,
    )


def score_states(scored: dict[str, object], labels: list[Label]) -> dict[str, TriState]:
    """Per slot kind: the decoded top-1 against the best answer that ignores the input.

    ``scored`` is :func:`_decode`'s output over the val plan. For ``choice`` and ``score``
    the baseline is the val set's most common gold letter; for ``span`` it is abstaining on
    every row, the only constant a pointer can answer. ``passed`` is the model above that
    baseline. Val's OWN majority is the stricter reading -- the model never saw these labels,
    and a constant tuned to them is the best a constant can do.
    """
    by_kind = scored["by_kind"]
    if not isinstance(by_kind, dict):  # pragma: no cover - _decode's own shape
        raise TypeError("scored['by_kind'] is not a mapping")
    states: dict[str, TriState] = {}
    for kind_id, kind in sorted(KIND_NAMES.items()):
        bucket = by_kind.get(kind)
        name = f"val_top1.{kind}"
        if not bucket or not int(bucket["n"]):
            states[name] = NotRun(reason=f"the val set holds no {kind} row to score")
            continue
        n, correct = int(bucket["n"]), int(bucket["correct"])
        if kind_id == SLOT_SPAN:
            base_n = int(bucket["abstaining"])
            base_what = "abstaining on every row"
        else:
            golds = collections.Counter(
                label.gold_letter for label in labels if label.slot_kind == kind_id
            )
            letter, base_n = golds.most_common(1)[0]
            base_what = f"answering {letter!r} on every row, the val set's most common gold"
        states[name] = Ran(
            passed=correct > base_n,
            value=correct / n,
            n=correct,
            n_total=n,
            detail=(
                f"{correct} of {n} val {kind} rows ({correct / n:.1%}) decoded to the gold "
                f"the way answer.rs decodes a {'pointer' if kind_id == SLOT_SPAN else 'Letters'}"
                f" query, on rows this model never trained on; {base_what} scores {base_n} "
                f"of {n} ({base_n / n:.1%}), a gap of {(correct - base_n) / n:+.1%}"
            ),
        )
    return states


#: Why ``ece.lang.*`` cannot be computed from rows without a language, carried INTO the
#: ``ece`` gate. The plan's calibration gate is per-k and per-language
#: (docs/ledger-schema.md, "an aggregate hides exactly the failure it is meant to catch"), so
#: a gate aggregated over the per-k half would pass on a breakdown the plan does not accept.
#: ``code.defect_class`` rows carry ``metadata["language"]`` and are split by it; the
#: commitpackft rewriters' rows carry none. GAP-FT-ECE-HAS-NO-LANGUAGE-TO-SPLIT-BY.
NO_LANGUAGE_REASON: Final[str] = (
    "these rows carry no language: qd_data.mixture's commitpackft rewriters put none in "
    "DataRow.metadata (only code.defect_class rows have one), so ece.lang.* cannot be "
    "computed for them and the per-language half of the plan's calibration gate is unmeasured"
)


def language_eces(verdicts: Sequence[Mapping[str, object]]) -> dict[str, TriState]:
    """``ece.lang.{language}`` for every language the letter rows carry, fail-closed.

    Letter rows without a language are ``ece.lang`` :class:`NotRun` with
    :data:`NO_LANGUAGE_REASON` -- all of them or some of them, because a gate that passed on
    the rows that had a language would be silent about the ones that did not. A language
    whose rows span two slot shapes is not run either: stacking distributions over different
    row counts needs padding, which :func:`letter_distributions` refuses for the same reason.
    Empty when there are no letter rows at all.
    """
    letter_rows = [v for v in verdicts if str(v["kind"]) != "span"]
    by_language: dict[str, list[Mapping[str, object]]] = {}
    missing = 0
    for v in letter_rows:
        language = v.get("language")
        if isinstance(language, str) and language:
            by_language.setdefault(language, []).append(v)
        else:
            missing += 1
    states: dict[str, TriState] = {}
    if missing:
        states["ece.lang"] = NotRun(
            reason=f"{missing} of {len(letter_rows)} letter rows: {NO_LANGUAGE_REASON}"
        )
    for language, rows in sorted(by_language.items()):
        groups = letter_distributions(rows)
        if len(groups) != 1:
            states[f"ece.lang.{language}"] = NotRun(
                reason=(
                    f"{language} rows span slot shapes {sorted(groups)}; one ECE over "
                    "distributions of different row counts needs padding, which is refused"
                )
            )
            continue
        ((probs, gold),) = groups.values()
        states[f"ece.lang.{language}"] = ece_gate(probs, gold)
    return states


#: Why a ``*.family`` metric is not run over verdicts that carry no ``family_id``.
NO_FAMILY_REASON: Final[str] = (
    "carry no family_id: _decode writes the label's family onto every verdict, so these were "
    "not produced by it and cannot be grouped by family"
)


def family_eces(verdicts: Sequence[Mapping[str, object]]) -> dict[str, TriState]:
    """``ece.family.{family_id}.{kind}.k{options}``: REPORT-ONLY, never in the ``ece`` gate.

    The same :func:`ece_gate` per slot shape as ``ece.{kind}.k{options}``, over one family's
    letter rows -- so a family under the sample floor is ``not_run`` with ``ece_gate``'s own
    reason, and ``passed`` is that metric's bar, as on the per-k and per-language metrics.
    Per shape because a family can ask over more than one option count (CLINC's
    within-domain intents), and one ECE over different row counts needs padding, which
    :func:`letter_distributions` refuses. Rows without a family are ``ece.family``
    :class:`NotRun`, never dropped. Empty when there are no letter rows at all.
    """
    letter_rows = [v for v in verdicts if str(v["kind"]) != "span"]
    by_family: dict[str, list[Mapping[str, object]]] = {}
    missing = 0
    for v in letter_rows:
        family = v.get("family_id")
        if isinstance(family, str) and family:
            by_family.setdefault(family, []).append(v)
        else:
            missing += 1
    states: dict[str, TriState] = {}
    if missing:
        states["ece.family"] = NotRun(
            reason=f"{missing} of {len(letter_rows)} letter rows {NO_FAMILY_REASON}"
        )
    for family, rows in sorted(by_family.items()):
        for key, (probs, gold) in letter_distributions(rows).items():
            states[f"ece.family.{family}.{key}"] = ece_gate(probs, gold)
    return states


def family_heads(verdicts: Sequence[Mapping[str, object]]) -> dict[str, TriState]:
    """``degenerate_head.family.{family_id}.{kind}.k{options}``: REPORT-ONLY, never in the
    ``degenerate_head`` control (Fable G1, GAP-DEGENERATE-HEAD-FAILS-ON-BINARY-SLOTS-TOO).

    :func:`degenerate_head_check`'s two numbers over one family's rows of one slot shape: mean
    predictive entropy (the value) and the top predicted row's share. ``passed`` is always
    true: whether the control's entropy floor and class-share cap apply per family is the
    human's ruling, and a per-family pass/fail here would be that ruling made where nobody
    looks. A letter row without a family leaves one ``degenerate_head.family`` NotRun and no
    per-family number, as ``qd-gate-report`` does. ``qd-gate-report`` recomputes each value.
    """
    letter_rows = [v for v in verdicts if str(v["kind"]) != "span"]
    by_family: dict[str, list[Mapping[str, object]]] = {}
    missing = 0
    for v in letter_rows:
        family = v.get("family_id")
        if isinstance(family, str) and family:
            by_family.setdefault(family, []).append(v)
        else:
            missing += 1
    if missing:
        return {"degenerate_head.family": NotRun(
            reason=f"{missing} of {len(letter_rows)} letter rows {NO_FAMILY_REASON}"
        )}
    states: dict[str, TriState] = {}
    for family, rows in sorted(by_family.items()):
        for key, (probs, _) in letter_distributions(rows).items():
            head = degenerate_head_check(probs)
            name = f"degenerate_head.family.{family}.{key}"
            if not isinstance(head, Ran):
                states[name] = head
                continue
            counts = np.bincount(np.argmax(probs, axis=1), minlength=probs.shape[1])
            share = float(counts.max() / len(probs))
            states[name] = Ran(
                passed=True, value=head.value, n=head.n, n_total=head.n_total,
                detail=(
                    f"mean predictive entropy {float(head.value):.4f} nats, top predicted "  # type: ignore[arg-type]
                    f"share {share:.3f}; report-only per family: the control judges each slot "
                    "shape over every family together, and whether its floor and cap apply "
                    "per family is the human's ruling"
                ),
            )
    return states


def report_eces(verdicts: Sequence[Mapping[str, object]]) -> dict[str, TriState]:
    """``ece.report.pooled`` and ``ece.report.lang.{language}``, ``none`` for letter rows that
    carry no language: REPORT-ONLY, never in the ``ece`` gate (Fable G2,
    GAP-ECE-GATE-IS-NOT-RUN-ON-THE-FULL-MIXTURE).

    ECE by top-1 confidence over letter rows of every slot shape together. Each row's
    distribution is :func:`letter_distributions`' softmax over its own rows, zero-padded to the
    widest row: padding moves neither a row's top probability nor its top row, which is all a
    top-1 ECE reads (a per-shape table would read the padded columns, which is why the gate's
    inputs refuse it). ``passed`` is always true: the gate's bar is stated, not applied. Rows
    must already have passed :func:`letter_distributions`, as in :func:`calibration_states`.
    """
    letter_rows = [v for v in verdicts if str(v["kind"]) != "span"]
    by_language: dict[str, list[Mapping[str, object]]] = {}
    for v in letter_rows:
        language = v.get("language")
        by_language.setdefault(
            language if isinstance(language, str) and language else "none", []
        ).append(v)
    groups = [("ece.report.pooled", letter_rows)] + [
        (f"ece.report.lang.{language}", rows) for language, rows in sorted(by_language.items())
    ]
    states: dict[str, TriState] = {}
    for name, rows in groups:
        if not rows:
            continue
        probs = np.zeros((len(rows), max(int(v["rows"]) for v in rows)))  # type: ignore[call-overload]
        for i, v in enumerate(rows):
            z = np.asarray([[float(x) for x in v["row_logits"]]], dtype=np.float64)  # type: ignore[union-attr]
            z = np.exp(z - z.max(axis=1, keepdims=True))
            probs[i, : z.shape[1]] = (z / z.sum(axis=1, keepdims=True))[0]
        state = ece_gate(probs, np.asarray([v["gold_row"] for v in rows], dtype=int))
        if not isinstance(state, Ran):
            states[name] = state
            continue
        shapes = sorted({f"{v['kind']}.k{int(v['rows']) - RESERVED_NOUL_ROWS}" for v in rows})  # type: ignore[call-overload]
        states[name] = Ran(
            passed=True, value=state.value, n=state.n, n_total=state.n_total,
            detail=(f"{state.detail}; letter rows of slot shapes {shapes} pooled by top-1 "
                    "confidence; report-only, not the ece gate, whose bar is not applied here"),
        )
    return states


def letter_distributions(
    verdicts: Sequence[Mapping[str, object]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each letter row's softmax over its own decode rows, and its gold row, by slot shape.

    Keyed ``{kind}.k{options}``, one group per entry of the runtime's calibration table
    (``letters_key``, ``{kind}:{options + noul}`` in calibration.rs): a distribution over 3
    rows and one over 6 are different quantities, and stacking them needs padding, which
    ``rung0_real_run.ece_state`` refuses for the same reason. Span rows are not here -- a
    pointer's row count is its context's line count, not a slot shape.
    """
    groups: dict[str, tuple[list[list[float]], list[int]]] = {}
    for v in verdicts:
        kind = str(v["kind"])
        if kind == "span":
            continue
        row_logits, gold_row, rows = v.get("row_logits"), v.get("gold_row"), v.get("rows")
        if not (
            isinstance(row_logits, list) and isinstance(gold_row, int) and isinstance(rows, int)
        ):
            raise TypeError(
                f"row {v.get('row_id')}: a letter verdict without row_logits, gold_row and "
                "rows was not produced by _decode"
            )
        if len(row_logits) != rows:
            raise ValueError(
                f"row {v.get('row_id')}: {len(row_logits)} logits for {rows} decode rows"
            )
        letters_key(kind, rows)  # refuses a shape the runtime table could not hold
        logits, golds = groups.setdefault(f"{kind}.k{rows - RESERVED_NOUL_ROWS}", ([], []))
        logits.append([float(x) for x in row_logits])
        golds.append(gold_row)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (logits, golds) in sorted(groups.items()):
        z = np.asarray(logits, dtype=np.float64)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        out[key] = (z / z.sum(axis=1, keepdims=True), np.asarray(golds, dtype=int))
    return out


def calibration_states(
    scored: Mapping[str, object],
) -> tuple[dict[str, TriState], TriState, TriState]:
    """Per slot shape: ``ece`` and ``degenerate_head``; then the gate and the control.

    Both read the model's own distributions, uncalibrated, at the functions' default
    thresholds -- rule 2 makes a threshold read-only, and passing one from here would be
    retuning it where nobody looks (the rung-0 call site says the same). The gate aggregates
    every per-k ECE AND every per-language one (:func:`language_eces`), so it is ``not_run``
    while any letter row has no language; every number is recorded as a metric regardless.
    The per-family ECEs (:func:`family_eces`) are metrics only and never reach the gate.
    """
    verdicts = scored["verdicts"]
    if not isinstance(verdicts, list):
        raise TypeError("scored['verdicts'] is not a list")
    metrics: dict[str, TriState] = {}
    eces: dict[str, TriState] = {}
    degenerate: dict[str, TriState] = {}
    for key, (probs, gold) in letter_distributions(verdicts).items():
        eces[f"ece.{key}"] = ece_gate(probs, gold)
        degenerate[f"degenerate_head.{key}"] = degenerate_head_check(probs)
    eces.update(language_eces(verdicts))
    metrics.update(eces)
    metrics.update(degenerate)
    # Into the metrics only: `eces` is what the gate aggregates, and a per-family ECE there
    # would change the gate's population (rule 2).
    metrics.update(family_eces(verdicts))
    # Report-only (Fable G1/G2), into the metrics only: they never enter `eces` or
    # `degenerate`, which are all the gate and the control aggregate.
    metrics.update(family_heads(verdicts))
    metrics.update(report_eces(verdicts))
    if not eces:
        eces["ece.lang"] = NotRun(reason="no letter rows were decoded")
    return metrics, aggregate(eces, name="ece"), aggregate(degenerate, name="degenerate_head")


def second_pass_batches(
    val: ValSet, spec: ChoicePermutation, *, seed: int
) -> tuple[list[Batch], dict[int, list[Label]], dict[tuple[str, str], tuple[int, ...]]]:
    """Every val batch holding a choice row, its choice rows' options deranged. CPU only.

    Built whole before any forward pass, so a row :meth:`ChoicePermutation.apply` refuses
    fails the run in seconds rather than after the first decode. Each row's order is
    ``qd_data.render.second_pass_permutation`` of ``(seed, row_id, slot_name)`` -- the
    derangement ``docs/schema-api.md`` requires, not the trainer's uniform per-pass shuffle,
    whose fixed points let a position-biased model agree with itself. Returns the batches,
    their labels by position, and each asked row's permutation (``perm[j]`` is the option
    first shown at ``j``, now shown at ``j``'s place in the second pass). A row with fewer
    than two options is left in place and not in the map: it has no derangement.
    """
    alphabets = _alphabets(val.plan, val.labels_for)
    batches: list[Batch] = []
    labels_for: dict[int, list[Label]] = {}
    perms: dict[tuple[str, str], tuple[int, ...]] = {}
    for b, batch in enumerate(val.plan):
        rows = val.labels_for[b]
        if not any(label.slot_kind == SLOT_CHOICE for label in rows):
            continue
        drawn: dict[int, tuple[int, ...]] = {}
        for r, letters in enumerate(alphabets[b]):
            if letters is None:
                continue
            label = rows[r]
            if len(letters) < 2:
                drawn[r] = tuple(range(len(letters)))
                continue
            perm = second_pass_permutation(
                len(letters), seed=seed, example_id=label.row_id, slot_name=label.slot_name
            )
            if any(perm[j] == j for j in range(len(perm))):
                raise SystemExit(
                    f"row {label.row_id}: second-pass permutation {perm} leaves an option in "
                    "place; it is not the derangement docs/schema-api.md requires"
                )
            key = (label.row_id, label.slot_name)
            if key in perms:
                raise SystemExit(f"two val choice rows share {key}; they cannot be paired")
            perms[key] = perm
            drawn[r] = perm
        permuted, _ = spec.apply(
            batch, alphabets[b], permutation_for=lambda _index, r, _m, drawn=drawn: drawn[r]
        )
        labels_for[len(batches)] = rows
        batches.append(permuted)
    return batches, labels_for, perms


def permutation_agreement(
    scored: Mapping[str, object],
    second: Mapping[str, object],
    perms: Mapping[tuple[str, str], tuple[int, ...]],
) -> TriState:
    """``permutation_consistency`` from the first pass's verdicts and the second pass's.

    Only ``top``, ``noul_row`` and the row's key are read from the second pass: its
    ``gold_row`` and ``correct`` are against labels that were not permuted, so they are
    meaningless, and nothing from it reaches the verdicts, accuracy or calibration. Both
    passes abstaining is agreement -- the runtime answers ``noul`` either way -- and one
    abstaining is not.
    """
    first = _choice_verdicts(scored)
    outcomes = permutation_outcomes(first, second, perms)
    return permutation_consistency_state(
        agree=sum(1 for _, agreed in outcomes.values() if agreed), asked=len(outcomes),
        total=len(first),
    )


def _choice_verdicts(decode: Mapping[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    """A decode's choice verdicts by ``(row_id, slot_name)``, the key a derangement has."""
    return {
        (str(v["row_id"]), str(v["slot_name"])): v
        for v in decode["verdicts"]  # type: ignore[union-attr]
        if v["kind"] == "choice"
    }


def permutation_outcomes(
    first: Mapping[tuple[str, str], Mapping[str, object]],
    second: Mapping[str, object],
    perms: Mapping[tuple[str, str], tuple[int, ...]],
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Per ASKED first-pass choice row (one with a derangement): the second pass's top row,
    and whether the two passes agreed.

    The one owner of the agreement rule, so the gate (:func:`permutation_agreement`), the
    per-family breakdown and ``--verdicts-out`` count the same rows the same way: both
    passes abstaining is agreement, one abstaining is not, and otherwise the permuted
    winner must map back (``perm[top2]``) to the first pass's. A row decoded in the first
    pass and not the second is refused rather than dropped.
    """
    again = _choice_verdicts(second)
    out: dict[tuple[str, str], tuple[int, bool]] = {}
    for key, v in first.items():
        perm = perms.get(key)
        if perm is None:
            continue
        w = again.get(key)
        if w is None:
            raise SystemExit(f"row {key} was decoded in the first pass and not the second")
        top1, top2 = int(v["top"]), int(w["top"])  # type: ignore[call-overload]
        abstained1 = top1 == int(v["noul_row"])  # type: ignore[call-overload]
        abstained2 = top2 == int(w["noul_row"])  # type: ignore[call-overload]
        agreed = (
            (abstained1 and abstained2) if abstained1 or abstained2 else perm[top2] == top1
        )
        out[key] = (top2, agreed)
    return out


def annotate_second_pass(
    scored: Mapping[str, object],
    second: Mapping[str, object],
    perms: Mapping[tuple[str, str], tuple[int, ...]],
) -> None:
    """Write each asked choice row's second pass onto its FIRST-pass verdict, in place:
    ``top_permuted`` (the permuted pass's top row, in ITS order), ``permutation_agreed``
    (:func:`permutation_outcomes`) and ``perm`` (``perm[j]`` is the option first shown at
    ``j``). What :func:`permutation_family_metrics` and ``--verdicts-out`` read; nothing a
    gate reads. A row with no derangement gets none of the three: it was never asked.
    """
    first = _choice_verdicts(scored)
    for key, (top2, agreed) in permutation_outcomes(first, second, perms).items():
        first[key].update(
            {"top_permuted": top2, "permutation_agreed": agreed, "perm": list(perms[key])}
        )


def permutation_family_metrics(
    scored: Mapping[str, object], gate: TriState
) -> dict[str, TriState]:
    """``permutation_consistency.family.{family_id}``: REPORT-ONLY, never the gate.

    Each family's share of the gate under the gate's own convention -- ``n`` agreeing of
    ``n_total`` asked -- so the families' ``n`` and ``n_total`` sum to the gate's.
    ``passed`` is always true: whether the 95% floor applies per family is the human's
    ruling, and a per-family pass/fail here would be that ruling made where nobody looks.
    Read from :func:`annotate_second_pass`'s fields. Fails closed without raising: a gate
    that did not run gives every family its reason, and counts that do not sum to the
    gate's give one ``permutation_consistency.family`` :class:`NotRun`, never a number.
    """
    first = _choice_verdicts(scored)
    by_family: dict[str, list[Mapping[str, object]]] = {}
    missing = 0
    for v in first.values():
        family = v.get("family_id")
        if isinstance(family, str) and family:
            by_family.setdefault(family, []).append(v)
        else:
            missing += 1
    if missing:
        return {"permutation_consistency.family": NotRun(
            reason=f"{missing} of {len(first)} choice rows {NO_FAMILY_REASON}"
        )}
    if isinstance(gate, NotRun):
        return {
            f"permutation_consistency.family.{family}": NotRun(
                reason=f"the permutation_consistency gate was not run: {gate.reason}"
            )
            for family in sorted(by_family)
        }
    out: dict[str, TriState] = {}
    agree_sum = asked_sum = 0
    for family, rows in sorted(by_family.items()):
        asked = [v for v in rows if "permutation_agreed" in v]
        agree = sum(1 for v in asked if v["permutation_agreed"] is True)
        agree_sum, asked_sum = agree_sum + agree, asked_sum + len(asked)
        name = f"permutation_consistency.family.{family}"
        if not asked:
            out[name] = NotRun(
                reason=(
                    f"none of {len(rows)} {family} choice rows had two or more live options, "
                    "so no derangement exists for this family"
                )
            )
            continue
        out[name] = Ran(
            passed=True, value=agree / len(asked), n=agree, n_total=len(asked),
            detail=(
                f"{agree} of {len(asked)} {family} choice rows ({agree / len(asked):.1%}) "
                f"agreed with themselves across the derangement; {len(rows) - len(asked)} "
                "had fewer than two live options. Reported per family, not the gate: "
                "permutation_consistency is judged over every family together"
            ),
        )
    if isinstance(gate, Ran) and (gate.n, gate.n_total) != (agree_sum, asked_sum):
        return {"permutation_consistency.family": NotRun(
            reason=(
                f"the families sum to {agree_sum} of {asked_sum} and the gate is {gate.n} of "
                f"{gate.n_total}: these verdicts do not carry the second pass the gate was "
                "scored from, so no per-family number is reported"
            )
        )}
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class SecondPass:
    """:func:`second_pass_batches`' output, built before a tower loads; or why it was not."""

    batches: list[Batch]
    labels_for: dict[int, list[Label]]
    perms: dict[tuple[str, str], tuple[int, ...]]
    not_run: str | None = None


def prepare_second_pass(
    val: ValSet, *, reader: ShardReader, tokenizer_json: Path | None, seed: int
) -> SecondPass:
    if tokenizer_json is None:
        return SecondPass(
            [], {}, {},
            not_run=(
                "no tokenizer.json: which token ids end an option line is a fact about the "
                "vocabulary, and without it the options cannot be moved"
            ),
        )
    spec = _permutation_spec(reader, tokenizer_json=tokenizer_json, letter_id=val.letter_id,
                             seed=seed)
    return SecondPass(*second_pass_batches(val, spec, seed=seed))


def score_permutation_consistency(
    step: RealFtStep, val: ValSet, second_pass: SecondPass, scored: Mapping[str, object]
) -> tuple[TriState, dict[str, object] | None]:
    """Decode the val set's choice rows a second time, options deranged, and compare.

    Returns the gate and the second pass's decode (``None`` when it did not run), which the
    ``ood_abstain`` in-distribution bound reads rather than decoding the val set a third time.
    The gate is computed first; then each asked row's second pass is written onto its
    ``scored`` verdict (:func:`annotate_second_pass`) for the per-family metrics and
    ``--verdicts-out``, which both scoring paths read from ``scored``.
    """
    if second_pass.not_run is not None:
        return NotRun(reason=second_pass.not_run), None
    second = _decode(step, second_pass.batches, second_pass.labels_for, val.letter_id)
    gate = permutation_agreement(scored, second, second_pass.perms)
    annotate_second_pass(scored, second, second_pass.perms)
    return gate, second


def choice_rule_abstentions(
    first: Mapping[str, object],
    second: Mapping[str, object],
    perms: Mapping[tuple[str, str], tuple[int, ...]],
    *,
    gold_noul: bool = False,
) -> dict[str, bool]:
    """Per first-pass choice row whose gold is not ``noul``: would the runtime abstain?

    ``crates/qd-runtime/src/answer.rs``'s generic-route choice rule, minus the calibrated
    margin: abstain when either pass's top row is ``noul``, or when the permuted pass's
    winner maps back to a different option. A row with no derangement (fewer than two
    options) has no second pass and abstains only on ``noul``. Keyed ``row_id``.

    ``gold_noul=True`` asks the same rule of the complement -- the rows whose gold IS
    ``noul``, which the in-distribution bound excludes by contract -- for the report-only
    ``ood_abstain.in_distribution.gold_noul.*`` counts. The gate's call never passes it.
    """
    again = {
        (str(v["row_id"]), str(v["slot_name"])): v
        for v in second["verdicts"]  # type: ignore[union-attr]
        if v["kind"] == "choice"
    }
    out: dict[str, bool] = {}
    for v in first["verdicts"]:  # type: ignore[union-attr]
        if v["kind"] != "choice" or bool(v["expected_abstain"]) is not gold_noul:
            continue
        key = (str(v["row_id"]), str(v["slot_name"]))
        top1 = int(v["top"])  # type: ignore[call-overload]
        abstained = top1 == int(v["noul_row"])  # type: ignore[call-overload]
        perm = perms.get(key)
        if perm is not None:
            w = again.get(key)
            if w is None:
                raise SystemExit(f"row {key} was decoded in the first pass and not the second")
            top2 = int(w["top"])  # type: ignore[call-overload]
            abstained = (
                abstained or top2 == int(w["noul_row"])  # type: ignore[call-overload]
                or perm[top2] != top1
            )
        out[key[0]] = abstained
    return out


#: Tokenizers :func:`_matching_tokenizer` loaded, by the hash they were checked against. The
#: needle and OOD suites each asked for one: two loads, 7.9 s under cProfile (2026-10-01).
_LOADED_TOKENIZERS: dict[str, Any] = {}


def _matching_tokenizer(reader: ShardReader, *, what: str) -> Any:
    """The pipeline's tokenizer, refused unless it hashes to the one ``reader`` was built with.

    Loaded once per process and per hash: it is stateless at ``memo_limit=0``, so a second
    load could only return the first's tokenizer.
    """
    import real_tokenizer_pipeline as pipeline

    if reader.remap is None:
        raise SystemExit(f"{reader.root}: no remap table beside the shards")
    want = reader.header.tokenizer_hash
    tok = _LOADED_TOKENIZERS.get(want)
    if tok is None:
        tok = pipeline.RealTokenizer.load(memo_limit=0)
        if tok.hash() != want:
            raise SystemExit(
                f"the tokenizer loaded for {what} hashes to {tok.hash()[:16]}, not the "
                f"val set's {want[:16]}: its ids would not be this model's"
            )
        _LOADED_TOKENIZERS[want] = tok
    return tok


def encode_slot_batch(
    row: DataRow, *, slot_kind: int, tok: Any, reader: ShardReader, config: DataConfig,
    index: int, row_id: str, where: str,
) -> tuple[Batch, Label, Any]:
    """One slot of an eval-only row as its own one-row batch, with its label.

    For suites the shard writer never saw (needle, OOD): rendered by ``training_texts``,
    encoded by ``encode_slot`` and batched by ``assemble_batch`` -- the writer's and the
    reader's own paths -- and labelled by :func:`_labels`, so letters and language come from
    the same place a val row's do. ``row_id`` is the suite's case id, which is how the
    suite's scorer finds the verdict.
    """
    if reader.remap is None:
        raise SystemExit(f"{reader.root}: no remap table beside the shards")
    (spec,) = [
        s for s in training_texts(row, seed=config.seed, caps=DEFAULT_CAPS)
        if s.slot_kind == slot_kind
    ]
    labels, excluded = _labels([row], config=config)
    if excluded:
        raise SystemExit(f"{where}: the row was refused: {excluded[0]}")
    (label,) = [x for x in labels if x.slot_name == spec.slot_name]
    encoded = encode_slot(
        spec, tokenize=tok.tokenize, remap=reader.remap, token_offsets=tok.offsets,
        decode=tok.decode, where=where,
    )
    n = int(encoded.ids.size)
    batch = assemble_batch(
        [encoded.ids], kinds=np.asarray([slot_kind]), target_index=np.asarray([n - 2]),
        spans=np.asarray([encoded.span]), candidates=[encoded.candidates],
        width=n, bucket=n, index=index,
    )
    return batch, dataclasses.replace(label, row_id=row_id), encoded


@dataclasses.dataclass(frozen=True, slots=True)
class NeedleSuite:
    """The needle suite, encoded one case per batch before a tower loads; or why not."""

    cases: list[NeedleCase]
    batches: list[Batch]
    labels_for: dict[int, list[Label]]
    #: Real token count per case, re-measured: ``build_suite`` sizes by a 3-chars/token guess.
    token_lengths: list[int]
    seed: int = 0
    not_run: str | None = None
    #: sha256 over every case id and its encoded ids: what a worker process scored must be
    #: this suite, byte for byte, before its predictions are recorded against it.
    digest: str = ""


#: Every needle batch is padded to one width, a multiple of this. MPS keeps a compiled graph
#: per distinct input shape and ``torch.mps.empty_cache`` does not release it: measured
#: 2026-09-30 on the seed-0 checkpoint, six cases padded to one width held driver memory at
#: 18.81-18.83 GiB, and one unpadded case of a new length added 3.9 GiB. 300 cases of 300
#: lengths do not fit. Right padding is masked, and the padded and unpadded verdicts of the
#: probed case were identical.
NEEDLE_WIDTH_MULTIPLE: Final[int] = 64


def _repad(batch: Batch, width: int) -> Batch:
    """``batch`` (one row) re-assembled at ``width``, its supervision unchanged."""
    n = int(batch.lengths[0])
    candidates = (
        [np.flatnonzero(batch.line_starts[0])] if batch.line_starts is not None else [()]
    )
    spans = (
        batch.span_target if batch.span_target is not None
        else np.full((1, 2), NO_SPAN, dtype=np.int64)
    )
    return assemble_batch(
        [batch.tokens[0, :n]], kinds=np.asarray(batch.slot_kind),
        target_index=np.asarray(batch.target_index), spans=spans, candidates=candidates,
        width=width, bucket=width, index=batch.index,
    )


def prepare_needle(
    reader: ShardReader, *, config: DataConfig, enabled: bool,
    target_tokens: int = NEEDLE_TARGET_TOKENS,
) -> NeedleSuite:
    """Build, render and encode the needle suite under ``reader``'s remap and tokenizer.

    ``target_tokens`` is the gate's ``NEEDLE_TARGET_TOKENS`` for the gate suite; only the
    ``--needle-control`` diagnostic builds a suite at another length, and it records it on a
    row of its own, never under the gate.

    Each case is a ``code.defect_class`` row (``needle_defect_row``), and only its span slot
    is encoded, through ``encode_slot`` -- the shard writer's own path. Two checks per case
    make the hunk mapping trustworthy before anything is decoded: one candidate per rendered
    context line, and the encoded gold's line falling in the needle hunk. Either failing is
    a refusal, because a mapping off by one line scores the model against the wrong hunk.
    """
    if not enabled:
        return NeedleSuite([], [], {}, [], not_run="--needle was not given")
    from qd_data.defect_class import CONTEXT_HEADER_LINES

    tok = _matching_tokenizer(reader, what="the needle suite")
    cases = build_suite(
        target_tokens=target_tokens, cases_per_depth=NEEDLE_CASES_PER_DEPTH,
        seed=config.seed,
    )
    batches: list[Batch] = []
    labels_for: dict[int, list[Label]] = {}
    lengths: list[int] = []
    for i, case in enumerate(cases):
        where = f"needle case {case.case_id}"
        batch, label, encoded = encode_slot_batch(
            needle_defect_row(case, config=config), slot_kind=SLOT_SPAN, tok=tok,
            reader=reader, config=config, index=i, row_id=case.case_id, where=where,
        )
        body = case.context[:-1] if case.context.endswith("\n") else case.context
        n_lines = CONTEXT_HEADER_LINES + len(body.split("\n"))
        if len(encoded.candidates) != n_lines:
            raise SystemExit(
                f"{where}: {len(encoded.candidates)} line-start candidates for {n_lines} "
                "context lines, so a candidate index is not a line number"
            )
        gold_line = encoded.candidates.index(encoded.span[0])
        if hunk_of_context_line(case, gold_line) != case.needle_index:
            raise SystemExit(
                f"{where}: the encoded gold starts on line {gold_line}, in hunk "
                f"{hunk_of_context_line(case, gold_line)}, not the needle hunk "
                f"{case.needle_index}"
            )
        batches.append(batch)
        labels_for[i] = [label]
        lengths.append(int(encoded.ids.size))
    width = -(-max(lengths) // NEEDLE_WIDTH_MULTIPLE) * NEEDLE_WIDTH_MULTIPLE
    return NeedleSuite(
        cases, [_repad(b, width) for b in batches], labels_for, lengths, seed=config.seed,
        digest=needle_suite_digest(cases, batches),
    )


def release_device_cache() -> None:
    """Hand the allocator's cached blocks back to the device, on whichever backend is up.

    A decode that follows a long one inherits its cache: MPS counts those blocks against its
    high-watermark cap, so an eval that fits alone can run out of memory after another.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclasses.dataclass(frozen=True, slots=True)
class NeedleDecoded:
    """A decode of the needle suite: what the gate scores, and what it was scored from.

    ``predictions`` is the hunk of each case's predicted start line, ``None`` on abstain --
    all ``score_suite`` reads. ``verdicts`` keeps the pointer itself per case (start, end,
    abstention row), which the hunk index throws away: without it a miss cannot be told
    apart as an adjacent hunk (a mapping defect) or a filler far away (the model).
    """

    predictions: dict[str, int | None]
    verdicts: tuple[dict[str, object], ...] = ()


#: Fields every raw needle verdict carries; the worker's payload is refused without them.
NEEDLE_VERDICT_FIELDS: Final[tuple[str, ...]] = (
    "suite", "case_id", "language", "depth_bucket", "depth_fraction", "needle_index",
    "n_hunks", "token_length", "start", "end", "noul_row", "rows", "abstained",
    "predicted_hunk", "hit",
)


def needle_predictions(
    step: RealFtStep | TowerEnsemble, suite: NeedleSuite, letter_id: Mapping[str, int],
    *, logits: bool = False,
) -> NeedleDecoded:
    """Decode the suite: per case, the hunk of the predicted start line (``None`` on
    abstain), and the raw pointer verdict it came from.

    ``logits`` (``--suite-logits``) adds each case's pointer scores over the runtime's rows
    -- ``start_logits`` and ``end_logits``, the abstention at ``noul_row`` -- to its raw
    verdict; for an ensemble they are its mean log-probabilities. Off, the verdicts are what
    they always were."""
    bound = getattr(step, "max_width", None)
    widest = max(int(b.tokens.shape[1]) for b in suite.batches)
    if bound is not None and int(bound) < widest:
        raise SystemExit(
            f"the step is bounded at {bound} tokens and the needle suite is padded to "
            f"{widest}: build it with eval_widths=suite_widths(...)"
        )
    release_device_cache()
    verdicts: list[Mapping[str, object]] = []
    for i, batch in enumerate(suite.batches):
        decoded = _decode(
            step, [batch], {0: suite.labels_for[i]}, dict(letter_id), pointer_scores=logits
        )
        verdicts.extend(decoded["verdicts"])  # type: ignore[arg-type]
    by_case = {str(v["row_id"]): v for v in verdicts}
    predictions: dict[str, int | None] = {}
    raw: list[dict[str, object]] = []
    for i, case in enumerate(suite.cases):
        v = by_case.get(case.case_id)
        if v is None:
            continue  # score_suite reports a case with no prediction as not_run
        start, end = (int(x) for x in v["top"])  # type: ignore[union-attr]
        noul = int(v["noul_row"])  # type: ignore[call-overload]
        hunk = None if noul in (start, end) else hunk_of_context_line(case, start)
        predictions[case.case_id] = hunk
        raw.append({
            "suite": "needle", "case_id": case.case_id, "language": case.language,
            "depth_bucket": case.depth_bucket, "depth_fraction": case.depth_fraction,
            "needle_index": case.needle_index, "n_hunks": case.n_hunks,
            "token_length": suite.token_lengths[i], "start": start, "end": end,
            "noul_row": noul, "rows": v.get("rows"), "abstained": hunk is None,
            "predicted_hunk": hunk, "hit": hunk == case.needle_index,
            **{k: v[k] for k in SUITE_LOGIT_KEYS if logits and k in v},
        })
    return NeedleDecoded(predictions, tuple(raw))


#: What ``--suite-logits`` adds to a needle case's raw verdict: the pointer's scores over
#: the runtime's rows, the abstention included (at ``noul_row``). An OOD case's line always
#: carries both passes' letter rows (``row_logits_1``/``row_logits_2``, noul included).
SUITE_LOGIT_KEYS: Final[tuple[str, ...]] = ("start_logits", "end_logits")


def score_needle(
    step: RealFtStep | TowerEnsemble, suite: NeedleSuite, letter_id: Mapping[str, int],
    *, decoded: NeedleDecoded | None = None, logits: bool = False,
) -> tuple[TriState, dict[str, TriState], tuple[dict[str, object], ...]]:
    """``needle_hunk_recall``, its by-depth metrics, and the raw per-case verdicts, under
    the approved contract.

    ``decoded`` is a worker process's (``--needle-predictions-out``); without it the suite
    is decoded here, with ``step`` (and ``logits`` as :func:`needle_predictions` takes it).
    Under ``logits`` every verdict must carry :data:`SUITE_LOGIT_KEYS`, wherever it was
    decoded: a line without them would read the same as one never asked for scores.
    """
    if suite.not_run is not None:
        return NotRun(reason=suite.not_run), {}, ()
    if decoded is None:
        decoded = needle_predictions(step, suite, letter_id, logits=logits)
    if logits:
        for v in decoded.verdicts:
            missing = [k for k in SUITE_LOGIT_KEYS if k not in v]
            if missing:
                raise SystemExit(
                    f"needle verdict {v.get('case_id')} lacks {', '.join(missing)} under "
                    "--suite-logits: the decode was asked for the pointer scores and returned "
                    "none, so its suite lines would read as a run that never asked"
                )
    predictions = decoded.predictions
    report, gate = score_suite(suite.cases, dict(predictions), min_recall=NEEDLE_MIN_RECALL)
    metrics: dict[str, TriState] = {}
    for bucket in report.by_depth:
        metrics[f"needle_hunk_recall.depth.{bucket.label}"] = Ran(
            passed=bucket.recall >= NEEDLE_MIN_RECALL, value=bucket.recall,
            n=bucket.correct, n_total=bucket.total,
            detail=f"95% Wilson CI [{bucket.lo:.3f}, {bucket.hi:.3f}]",
        )
    abstained = sum(1 for p in predictions.values() if p is None)
    lengths = sorted(suite.token_lengths)
    metrics["needle_suite_tokens"] = Ran(
        passed=True, value=lengths[len(lengths) // 2], n=len(lengths), n_total=len(lengths),
        detail=(
            f"real tokens per case, min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
            f"max {lengths[-1]}, against a {NEEDLE_TARGET_TOKENS} target sized by a "
            f"3-chars/token guess; {abstained} of {len(predictions)} cases abstained. The "
            "model trained on sequences of at most ~1.1K tokens"
        ),
    ) if lengths else NotRun(reason="the suite is empty")
    return gate, metrics, decoded.verdicts


#: The needle worker decodes 300 ~8.5K cases at ~6.5 s each on the Mac (2026-09-30 probe),
#: plus a checkpoint load: about 35 minutes. Twice that, so a hang is a refusal, not a wait.
NEEDLE_WORKER_TIMEOUT_S: Final[float] = 2 * 3600.0


#: What the scoring process hands its needle worker (:func:`write_needle_handoff`). A reader
#: refuses any other value, so a change to what is written is a new format, never a quiet
#: reinterpretation of the old one.
NEEDLE_HANDOFF_FORMAT: Final[str] = "qd-needle-handoff/1"
#: Every field of a :class:`Batch`, in the order :func:`batches_digest` hashes them.
_BATCH_ARRAYS: Final[tuple[str, ...]] = (
    "tokens", "lengths", "slot_kind", "target_index", "span_target", "line_starts",
)


def batches_digest(batches: Sequence[Batch]) -> str:
    """sha256 over every array and integer of every batch, dtypes and shapes included.

    Wider than :class:`NeedleSuite`'s ``digest`` (case ids and unpadded tokens): this covers
    everything ``_decode`` reads -- the padding, the slot kind, the target index, the span
    gold and the candidate mask -- so a round trip through a file that changed any of them,
    even only a dtype, does not hash the same.
    """
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(f"batch {int(batch.bucket)} {int(batch.index)}\n".encode())
        for name in _BATCH_ARRAYS:
            array = getattr(batch, name)
            if array is None:
                digest.update(f"{name} none\n".encode())
                continue
            array = np.ascontiguousarray(array)
            digest.update(f"{name} {array.dtype.str} {array.shape}\n".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def needle_suite_digest(cases: Sequence[NeedleCase], batches: Sequence[Batch]) -> str:
    """:class:`NeedleSuite`'s ``digest``: sha256 over every case id and its unpadded ids."""
    digest = hashlib.sha256()
    for case, batch in zip(cases, batches, strict=True):
        digest.update(case.case_id.encode("utf-8"))
        digest.update(batch.tokens[0, : int(batch.lengths[0])].astype(np.int64).tobytes())
    return digest.hexdigest()


def _plan_shapes_digest(plan: Sequence[Batch]) -> str:
    """sha256 over the plan's ``(rows, width)`` per batch, in order: what sizes a step."""
    shapes = [[int(b.tokens.shape[0]), int(b.tokens.shape[1])] for b in plan]
    return hashlib.sha256(json.dumps(shapes).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class NeedleHandoff:
    """What the needle worker decodes, as the scoring process built and checked it."""

    suite: NeedleSuite
    letter_id: dict[str, int]
    eval_widths: list[int]


def write_needle_handoff(
    path: Path, suite: NeedleSuite, *, letter_id: Mapping[str, int],
    eval_widths: Sequence[int], train: ShardReader, val: ValSet,
) -> None:
    """The built needle suite, and what the worker's step is sized by, as one ``.npz``.

    The worker used to rebuild all of it from argv -- the corpus rebuild, the train and val
    relabels, the second pass, the needle and OOD suites -- a second full prelude, at 100% of
    one core with the GPU idle (~5 minutes on the GH200 for J7g, 2026-10-01), to decode a
    suite this process had already built and checked. Now it reads this file instead.

    Written with ``allow_pickle`` off: every batch field as a stacked array (one row per case,
    one padded width), the cases and their labels as JSON. Recorded beside them: the suite's
    own ``digest``, :func:`batches_digest` over the batches as built, the train and val shard
    hashes and the val plan's shapes -- every one of which the worker re-derives and must
    match before it decodes anything.
    """
    if suite.not_run is not None or not suite.cases:
        raise SystemExit(f"no needle suite to hand over: {suite.not_run or 'it is empty'}")
    batches = suite.batches
    widths = {int(b.tokens.shape[1]) for b in batches}
    if len(batches) != len(suite.cases) or len(widths) != 1 or any(
        int(b.tokens.shape[0]) != 1 for b in batches
    ):
        raise SystemExit(
            "the needle suite is not one single-row batch per case at one padded width; "
            "it cannot be handed over as stacked arrays"
        )
    present = {name: [getattr(b, name) is not None for b in batches] for name in _BATCH_ARRAYS}
    if any(len(set(flags)) != 1 for flags in present.values()):
        raise SystemExit("needle batches disagree about which supervision fields they carry")
    arrays: dict[str, np.ndarray] = {
        name: np.concatenate([np.ascontiguousarray(getattr(b, name)) for b in batches])
        for name, flags in present.items() if flags[0]
    }
    labels = [suite.labels_for[i] for i in range(len(suite.cases))]
    meta = {
        "format": NEEDLE_HANDOFF_FORMAT,
        "cases": [dataclasses.asdict(c) for c in suite.cases],
        "labels": [[dataclasses.asdict(x) for x in group] for group in labels],
        "token_lengths": [int(n) for n in suite.token_lengths],
        "seed": int(suite.seed),
        "digest": suite.digest,
        "batches_digest": batches_digest(batches),
        "buckets": [int(b.bucket) for b in batches],
        "indices": [int(b.index) for b in batches],
        "letter_id": [[letter, int(i)] for letter, i in letter_id.items()],
        "eval_widths": [int(w) for w in eval_widths],
        "train_shard_hash": train.header.shard_hash(),
        "val_shard_hash": val.reader.header.shard_hash(),
        "val_plan_shapes": _plan_shapes_digest(val.plan),
    }
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    with path.open("xb") as fh:
        np.savez(fh, **arrays)  # type: ignore[arg-type]


def read_needle_handoff(
    path: Path, *, train: ShardReader, val: ValPlan, seed: int
) -> NeedleHandoff:
    """:func:`write_needle_handoff`'s file, refused unless everything it records re-derives.

    Re-derived here, from what was read: the suite ``digest`` and :func:`batches_digest`
    (so the batches decoded are, array for array, the ones the scoring process built), the
    train and val shard hashes from this process's own readers, the val plan's shapes from
    this process's own plan, and the seed. What is NOT re-run is the suite's construction and
    its per-case checks (one candidate per context line, the gold in the needle hunk): they
    ran once, in the scoring process, before this file was written, which is all they were
    ever for. The digest check is therefore a round-trip check of the parent's suite, no
    longer an independent rebuild of it -- and the parent still refuses predictions whose
    digest is not its own suite's.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
    except (OSError, ValueError) as exc:
        raise SystemExit(f"needle handoff {path}: unreadable: {exc}") from exc
    if "meta" not in arrays:
        raise SystemExit(f"needle handoff {path}: no meta record")
    meta = json.loads(arrays.pop("meta").tobytes().decode("utf-8"))
    if meta.get("format") != NEEDLE_HANDOFF_FORMAT:
        raise SystemExit(
            f"needle handoff {path}: format {meta.get('format')!r}, not {NEEDLE_HANDOFF_FORMAT!r}"
        )
    pinned = {
        "train shard hash": (meta["train_shard_hash"], train.header.shard_hash()),
        "val shard hash": (meta["val_shard_hash"], val.reader.header.shard_hash()),
        "val plan shapes": (meta["val_plan_shapes"], _plan_shapes_digest(val.plan)),
        "suite seed": (meta["seed"], seed),
    }
    wrong = {k: v for k, v in pinned.items() if v[0] != v[1]}
    if wrong:
        raise SystemExit(
            f"needle handoff {path} was written for another run: "
            + "; ".join(f"{k}: handoff {a!r}, here {b!r}" for k, (a, b) in wrong.items())
        )
    cases = [NeedleCase(**c) for c in meta["cases"]]
    n = len(cases)
    if set(arrays) - set(_BATCH_ARRAYS) or any(a.shape[0] != n for a in arrays.values()):
        raise SystemExit(f"needle handoff {path}: arrays {sorted(arrays)} are not one row per case")
    if not (len(meta["labels"]) == len(meta["buckets"]) == len(meta["indices"])
            == len(meta["token_lengths"]) == n):
        raise SystemExit(f"needle handoff {path}: its records do not count {n} cases each")
    try:
        batches = [
            Batch(
                **{name: (arrays[name][i : i + 1] if name in arrays else None)
                   for name in _BATCH_ARRAYS},
                bucket=int(meta["buckets"][i]), index=int(meta["indices"][i]),
            )
            for i in range(n)
        ]
    except ShardContractViolation as exc:
        raise SystemExit(f"needle handoff {path}: a case is not a valid batch: {exc}") from exc
    labels_for = {
        i: [Label(**{**x, "letters": tuple(x["letters"])}) for x in group]
        for i, group in enumerate(meta["labels"])
    }
    for what, recorded, found in (
        ("suite digest", meta["digest"], needle_suite_digest(cases, batches)),
        ("batches digest", meta["batches_digest"], batches_digest(batches)),
    ):
        if recorded != found:
            raise SystemExit(
                f"needle handoff {path}: its {what} is {str(recorded)[:16]} but what it holds "
                f"hashes to {found[:16]}; it was not read back as written"
            )
    suite = NeedleSuite(
        cases, batches, labels_for, [int(x) for x in meta["token_lengths"]],
        seed=int(meta["seed"]), digest=str(meta["digest"]),
    )
    return NeedleHandoff(
        suite=suite,
        letter_id={str(letter): int(i) for letter, i in meta["letter_id"]},
        eval_widths=[int(w) for w in meta["eval_widths"]],
    )


def run_needle_worker(
    argv: Sequence[str], suite: NeedleSuite, *, letter_id: Mapping[str, int],
    eval_widths: Sequence[int], train: ShardReader, val: ValSet,
) -> NeedleDecoded:
    """Score the needle suite in a fresh process and return its per-case predictions.

    A separate process because MPS keeps a compiled graph per distinct input shape that
    ``torch.mps.empty_cache`` does not release: on 2026-09-30 the val pass left 37.56 GiB of
    such allocations, and the needle pass after it ran out of memory twice (48 GiB cap on
    a 64 GiB Mac), though alone it holds ~18.8 GiB. A process that exits returns all of
    it. The worker is handed the suite this process built (:func:`write_needle_handoff`)
    rather than rebuilding it from argv, and its predictions are refused unless they carry
    this suite's digest -- same cases, same ids, byte for byte.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="qd-needle-worker-") as tmp:
        out = Path(tmp) / "predictions.json"
        handoff = Path(tmp) / "needle-handoff.npz"
        write_needle_handoff(
            handoff, suite, letter_id=letter_id, eval_widths=eval_widths, train=train, val=val,
        )
        cmd = [sys.executable, str(Path(__file__).resolve()), *argv,
               "--needle-handoff", str(handoff), "--needle-predictions-out", str(out)]
        print(
            f"needle worker: starting {len(suite.cases)} cases in a fresh process, handed the "
            f"suite this process built ({handoff.stat().st_size} bytes, digest "
            f"{suite.digest[:16]})",
            flush=True,
        )
        try:
            done = subprocess.run(cmd, timeout=NEEDLE_WORKER_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(
                f"the needle worker ran past {NEEDLE_WORKER_TIMEOUT_S:.0f} s and was killed"
            ) from exc
        if done.returncode != 0:
            raise SystemExit(f"the needle worker exited {done.returncode}; nothing recorded")
        payload = json.loads(out.read_text(encoding="utf-8"))
    if payload.get("digest") != suite.digest:
        raise SystemExit(
            "the needle worker scored a different suite (digest "
            f"{str(payload.get('digest'))[:16]} vs {suite.digest[:16]})"
        )
    predictions = payload.get("predictions")
    ids = {c.case_id for c in suite.cases}
    if not isinstance(predictions, dict) or set(predictions) != ids:
        raise SystemExit("the needle worker's predictions do not name exactly the suite's cases")
    for case_id, hunk in predictions.items():
        if hunk is not None and (not isinstance(hunk, int) or isinstance(hunk, bool)):
            raise SystemExit(f"needle worker prediction for {case_id} is {hunk!r}")
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list) or not all(isinstance(v, dict) for v in verdicts):
        raise SystemExit("the needle worker returned no raw verdicts")
    if sorted(str(v.get("case_id")) for v in verdicts) != sorted(ids):
        raise SystemExit("the needle worker's raw verdicts do not name exactly the suite's cases")
    for v in verdicts:
        missing = [k for k in NEEDLE_VERDICT_FIELDS if k not in v]
        if missing:
            raise SystemExit(f"needle worker verdict {v.get('case_id')} lacks {missing}")
        if v["predicted_hunk"] != predictions[str(v["case_id"])]:
            raise SystemExit(
                f"needle worker verdict {v['case_id']} names hunk {v['predicted_hunk']!r} "
                f"and its prediction {predictions[str(v['case_id'])]!r}"
            )
    return NeedleDecoded(dict(predictions), tuple(verdicts))


def needle_worker_main(
    args: argparse.Namespace, *, reader: ShardReader, config: DataConfig, rev: str
) -> int:
    """The ``--needle`` worker: decode the handed-over suite and write its predictions.

    Reads only what the decode needs: the train shard set (its remap and shard hash, for the
    checkpoint's pairing with its ft rows), the val set's batch plan (the shapes the step is
    sized by -- its labels are never read), and the handoff (:func:`read_needle_handoff`).
    The corpus rebuild, both relabels, the second pass and both suites are not repeated: the
    scoring process ran them, with every check they carry, before it wrote the handoff.
    """
    if args.devices is None or len(args.devices) != 1:  # main refuses this at argv time
        raise SystemExit("the needle worker runs on exactly one --devices entry")
    val_reader = open_val_reader(args.out, config=config, rev=rev, train=reader)
    plan = ValPlan(val_reader, val_plan(val_reader, config=config))
    handoff = read_needle_handoff(args.needle_handoff, train=reader, val=plan, seed=config.seed)
    print(
        f"needle worker: {len(handoff.suite.cases)} cases read from the scoring process's "
        f"handoff (digest {handoff.suite.digest[:16]}, re-derived from what was read); the "
        "corpus rebuild, the relabels and the suite construction were NOT repeated here -- "
        "the scoring process ran them and their checks before handing the suite over",
        flush=True,
    )
    worker_step, *_ = _checkpoint_step(
        args, reader=reader, val=plan, device=args.devices[0],
        eval_widths=handoff.eval_widths, suite_seed=config.seed,
    )
    # --suite-logits rides in on the scoring process's own argv; the scores go back in the
    # verdicts, and score_needle refuses a verdict without them under the flag.
    decoded = needle_predictions(
        worker_step, handoff.suite, handoff.letter_id, logits=args.suite_logits
    )
    args.needle_predictions_out.write_text(
        json.dumps({
            "digest": handoff.suite.digest, "predictions": decoded.predictions,
            "verdicts": list(decoded.verdicts),
        }),
        encoding="utf-8",
    )
    print(f"needle worker: {len(decoded.predictions)} predictions -> {args.needle_predictions_out}")
    return 0


def needle_recipe(suite: NeedleSuite) -> dict[str, object]:
    """What the needle gate was scored under, for the eval row's recipe -- only when it
    ran, so a row without it keeps the recipe hash it always had."""
    return {
        "min_recall": NEEDLE_MIN_RECALL, "cases_per_depth": NEEDLE_CASES_PER_DEPTH,
        "target_tokens": NEEDLE_TARGET_TOKENS, "hit_rule": NEEDLE_HIT_RULE,
        "suite_seed": suite.seed, "cases": len(suite.cases),
    }


#: At most this many ``--needle-control`` arms: each is 300 decodes, so the fan-out is bounded.
NEEDLE_CONTROL_MAX_ARMS: Final[int] = 8
#: Why a ``--needle-control`` row is quick whatever else holds: it is a diagnostic of one
#: checkpoint at one seed, on suites the gate does not use. It cannot promote and is not
#: evidence for or against the gate.
NEEDLE_CONTROL_QUICK_REASON: Final[str] = (
    "needle length-control diagnostic: one checkpoint, one seed, suites at lengths the gate "
    "does not use; not the needle_hunk_recall gate"
)


def parse_needle_control(text: str) -> tuple[int, ...]:
    """``--needle-control``'s target lengths: distinct, ascending, each one ``build_suite``
    takes, and at most :data:`NEEDLE_CONTROL_MAX_ARMS` of them."""
    try:
        lengths = tuple(int(x) for x in text.split(","))
    except ValueError as exc:
        raise SystemExit(f"--needle-control {text!r}: comma-separated token counts") from exc
    if len(set(lengths)) != len(lengths) or list(lengths) != sorted(lengths):
        raise SystemExit(f"--needle-control {text!r}: lengths must be distinct and ascending")
    if lengths[0] < 256:
        raise SystemExit(f"--needle-control {text!r}: build_suite needs at least 256 tokens")
    if len(lengths) > NEEDLE_CONTROL_MAX_ARMS:
        raise SystemExit(
            f"--needle-control {text!r}: at most {NEEDLE_CONTROL_MAX_ARMS} arms, got {len(lengths)}"
        )
    return lengths


def needle_control_metrics(
    length: int, suite: NeedleSuite, decoded: NeedleDecoded, *, trained_width: int | None,
) -> dict[str, TriState]:
    """One arm's metrics under ``needle_hunk_recall.control.<length>``: the gate's own rule
    and threshold applied to a suite of another length, recorded as metrics only.

    The suites are independent ``build_suite`` draws from the same seed, and the fillers are
    fixed short templates, so a longer arm has more distractor hunks as well as more tokens:
    the sweep moves length and hunk count together and cannot separate them.
    """
    key = f"needle_hunk_recall.control.{length}"
    report, gate = score_suite(suite.cases, dict(decoded.predictions), min_recall=NEEDLE_MIN_RECALL)
    lengths = sorted(suite.token_lengths)
    longest = lengths[-1]
    width = (
        "the trained width is not recorded on the ft row" if trained_width is None else
        f"{'within' if longest <= trained_width else 'beyond'} the trained width {trained_width}"
    )
    hunks = sorted(c.n_hunks for c in suite.cases)
    metrics: dict[str, TriState] = {
        key: (
            dataclasses.replace(
                gate,
                detail=f"length control at {length} target tokens, NOT the gate: {gate.detail}",
            )
            if isinstance(gate, Ran) else gate
        ),
        f"needle_suite_tokens.control.{length}": Ran(
            passed=True, value=lengths[len(lengths) // 2], n=len(lengths), n_total=len(lengths),
            detail=(
                f"real tokens min {lengths[0]}, median {lengths[len(lengths) // 2]}, max "
                f"{longest} ({width}); hunks per case {hunks[0]}..{hunks[-1]}"
            ),
        ),
    }
    for bucket in report.by_depth:
        metrics[f"{key}.depth.{bucket.label}"] = Ran(
            passed=bucket.recall >= NEEDLE_MIN_RECALL, value=bucket.recall,
            n=bucket.correct, n_total=bucket.total,
            detail=f"95% Wilson CI [{bucket.lo:.3f}, {bucket.hi:.3f}]",
        )
    k = sum(1 for p in decoded.predictions.values() if p is None)
    metrics[f"{key}.abstained"] = Ran(
        passed=True, value=k / len(decoded.predictions), n=k, n_total=len(decoded.predictions),
        detail="cases whose pointer named the abstention row (reported, not a gate)",
    )
    return metrics


def run_needle_control(
    args: argparse.Namespace, *, reader: ShardReader, val: ValSet, device: str, ledger: Ledger,
    config: DataConfig, gate_suite: NeedleSuite, reasons_for: Callable[..., list[str]],
) -> tuple[str, list[dict[str, object]], int]:
    """``--needle-control``: the needle suite at each given length, scored by the gate's rule,
    on a quick row of its own. Returns ``(row id, raw verdict lines, row seed)``.

    The row's recipe is its own (``tag`` ``epoch-needle-length-control`` and the arms), so it
    is a seed family of its own: it never joins, supplements or blocks the gate's eval rows,
    and it names no ``eval_row_id``. The gate-length arm must be the gate suite byte for
    byte (digest), or the sweep is refused: its 8K point is then the gate's own suite.

    An average (``.safetensors``) is paired as its score row is (:func:`_averaged_weights`):
    the row's tag is ``avg-needle-length-control``, its recipe carries the ``averaged``
    block, it names every input's ft row under ``ft_run_row_ids`` and never one of them under
    ``ft_run_row_id``, and its protocol seed is the suite seed, which is the row seed returned.
    """
    lengths: tuple[int, ...] = args.needle_control
    suites: dict[int, NeedleSuite] = {}
    for n in lengths:
        suite = prepare_needle(val.reader, config=config, enabled=True, target_tokens=n)
        if n == NEEDLE_TARGET_TOKENS and suite.digest != gate_suite.digest:
            raise SystemExit(
                f"the {n}-token control arm hashes to {suite.digest[:16]}, not the gate "
                f"suite's {gate_suite.digest[:16]}: the target_tokens path is not the gate's"
            )
        suites[n] = suite
        print(
            f"needle control {n}: {len(suite.cases)} cases, real tokens "
            f"{min(suite.token_lengths)}..{max(suite.token_lengths)}", flush=True,
        )
    widths = [int(b.tokens.shape[1]) for s in suites.values() for b in s.batches]
    step, ft, recipe, seed, meta = _checkpoint_step(
        args, reader=reader, val=val, device=device, eval_widths=widths,
        suite_seed=config.seed,
    )
    # For an average or an ensemble `ft` is its first input's row; the pairing has checked
    # that every input's row has this shard set and this recipe, which fix the plan's widest
    # batch.
    plan_width = ft["metrics"].get("corpus.plan_max_width", {}).get("value")
    trained_width = int(plan_width) if isinstance(plan_width, int) else None
    decode_at = time.monotonic()
    metrics: dict[str, TriState] = {}
    lines: list[dict[str, object]] = []
    for n, suite in suites.items():
        decoded = needle_predictions(step, suite, val.letter_id, logits=args.suite_logits)
        metrics.update(needle_control_metrics(n, suite, decoded, trained_width=trained_width))
        lines.extend({**v, "target_tokens": n} for v in decoded.verdicts)
        print(f"  needle control {n}: {metrics[f'needle_hunk_recall.control.{n}'].to_json()}")
    decode_s = time.monotonic() - decode_at
    model = scored_model(args, ft, meta)
    control_recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py",
        "tag": f"{model.tag}-needle-length-control",
        "device": device,
        **{k: recipe[k] for k in (*BACKBONE_KEYS, *RECIPE_PIECE_KEYS) if k in recipe},
        "score_dtype": args.score_dtype,
        "scored_checkpoint": model.scored_checkpoint,
        **model.recipe_block(),
        "shard_hash": reader.header.shard_hash(),
        "val_shard_hash": val.reader.header.shard_hash(),
        "needle_control": {
            "target_tokens": list(lengths), "cases_per_depth": NEEDLE_CASES_PER_DEPTH,
            "hit_rule": NEEDLE_HIT_RULE, "min_recall": NEEDLE_MIN_RECALL,
            "suite_seed": config.seed, "gate_suite_digest": gate_suite.digest,
        },
    }
    reasons = model_reasons(model, device=device, reasons_for=reasons_for)
    reasons.append(NEEDLE_CONTROL_QUICK_REASON)
    recorder = _recorder(
        ledger, reader=reader, seed=seed, recipe=control_recipe, run_kind="eval",
        quick_reasons=reasons, wall_clock_s=decode_s,
        cost=_cost(
            device=device, n_gpus=n_gpus_for_device(device), usd_per_hour=args.usd_per_hour,
            usd_per_gpu_hour=args.usd_per_gpu_hour, instance=args.instance,
            cap_s=args.wall_clock_cap_s,
        ),
        notes=(
            f"tools/real_ft_run.py --needle-control {','.join(map(str, lengths))} on "
            f"{model.scored_checkpoint} ({model.described}), {args.score_dtype} "
            f"on {device}: the needle gate's rule at other lengths, a diagnostic for "
            "GAP-NEEDLE-FAILS-AT-8K-CAUSE-UNRESOLVED. Length and hunk count move together "
            "(fixed filler templates), so this cannot separate them."
        ),
    )
    with recorder:
        # ft_run_row_ids for an average or an ensemble, never ft_run_row_id: that names ONE
        # row, and shuffled_label_target reads it so.
        recorder.metric(*model.ft_row_metric())
        for name, state in metrics.items():
            recorder.metric(name, state)
        recorder.noul_rate = NotRun(reason="a needle length-control row decodes no val row")
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id, lines, seed


#: Rows per general cache file read for the OOD prose pool. The val split is a keyed hash
#: per split unit, so reading fewer rows changes which texts are available, never which
#: split a text is in; 2,000 per file gave 2,699 val-split MMLU/CSQA questions.
OOD_GENERAL_MAX_ROWS: Final[int] = 2_000
#: The general families whose context is plain English prose.
OOD_PROSE_FAMILIES: Final[tuple[str, ...]] = (
    "knowledge.multiple_choice", "commonsense.multiple_choice",
)


@dataclasses.dataclass(frozen=True, slots=True)
class OodSuite:
    """The OOD suite's choice rows, encoded, with their permuted second pass; or why not."""

    cases: list[OodCase]
    val: ValSet | None
    second_pass: SecondPass | None
    record_sha256: str = ""
    seed: int = 0
    not_run: str | None = None


def prepare_ood(
    reader: ShardReader, *, config: DataConfig, rev: str, letter_id: Mapping[str, int],
    tokenizer_json: Path | None, general_record: Path | None, enabled: bool,
) -> OodSuite:
    """Build and encode the OOD suite, and its second pass, before a tower loads.

    The prose pool is the general records' **val** split, rebuilt by :func:`ft_splits`
    from ``general_record`` alone -- never the held-out split, which ``ft_splits`` returns
    and this does not read.
    """
    if not enabled:
        return OodSuite([], None, None, not_run="--ood was not given")
    if general_record is None or tokenizer_json is None:
        raise SystemExit("--ood needs --ood-general-record and the backbone's tokenizer.json")
    splits = ft_splits(
        commitpackft=None, max_pairs=0, rev=rev, config=config, repo_history=False,
        general_record=general_record, general_max_rows=OOD_GENERAL_MAX_ROWS,
    )
    prose = [
        row.request.context.decode("utf-8")
        for row in splits["val"]
        if row.family_id in OOD_PROSE_FAMILIES
    ]
    cases = build_ood_suite(prose, seed=config.seed)
    tok = _matching_tokenizer(reader, what="the OOD suite")
    plan: list[Batch] = []
    labels_for: dict[int, list[Label]] = {}
    for i, case in enumerate(cases):
        batch, label, _ = encode_slot_batch(
            ood_defect_row(case, config=config), slot_kind=SLOT_CHOICE, tok=tok,
            reader=reader, config=config, index=i, row_id=case.case_id,
            where=f"OOD case {case.case_id}",
        )
        plan.append(batch)
        labels_for[i] = [label]
    val = ValSet(
        reader=reader, labels=[x for rows in labels_for.values() for x in rows], plan=plan,
        labels_for=labels_for, letter_id=dict(letter_id),
    )
    second_pass = prepare_second_pass(
        val, reader=reader, tokenizer_json=tokenizer_json, seed=config.seed
    )
    return OodSuite(
        cases, val, second_pass,
        record_sha256=hashlib.sha256(general_record.read_bytes()).hexdigest(),
        seed=config.seed,
    )


def score_ood(
    step: RealFtStep, suite: OodSuite, *, scored: Mapping[str, object],
    val_second: Mapping[str, object] | None, val_second_pass: SecondPass,
) -> tuple[TriState, dict[str, TriState], tuple[dict[str, object], ...]]:
    """``ood_abstain``, its metrics, and each OOD case's raw verdict: the suite's
    abstentions and the val set's, and both passes' letter distributions per case -- what
    a margin or calibration diagnostic reads, and what the counts alone cannot show."""
    margin: TriState = NotRun(
        reason=(
            "the calibrated-margin half of the runtime's abstain rule needs a fitted "
            "calibration table, and none exists (GAP-RT-CALIBRATION-NOT-FITTED)"
        )
    )
    if suite.not_run is not None:
        return NotRun(reason=suite.not_run), {}, ()
    assert suite.val is not None and suite.second_pass is not None  # set whenever it ran
    if suite.second_pass.not_run is not None or val_second is None:
        why = suite.second_pass.not_run or val_second_pass.not_run or "no val second pass"
        return NotRun(reason=f"the permuted second pass did not run: {why}"), {
            "ood_abstain.margin": margin,
        }, ()
    first, second, ood = decode_ood(step, suite)
    indist = choice_rule_abstentions(scored, val_second, val_second_pass.perms)
    metrics: dict[str, TriState] = {"ood_abstain.margin": margin}
    metrics.update(ood_category_metrics(
        suite, ood, prefix="ood_abstain", what="(reported, not the gate)"
    ))
    in_k = sum(indist.values())
    metrics["ood_abstain.in_distribution"] = (
        Ran(
            passed=True, value=in_k / len(indist), n=in_k, n_total=len(indist),
            detail="val choice rows the runtime rule would abstain on (reported, not the gate)",
        )
        if indist else NotRun(reason="no val choice row was scored")
    )
    metrics.update(in_distribution_family_metrics(
        scored, indist,
        choice_rule_abstentions(scored, val_second, val_second_pass.perms, gold_noul=True),
    ))
    gate = ood_gate(
        ood_abstained=sum(ood.values()), ood_total=len(ood),
        in_abstained=in_k, in_total=len(indist),
    )
    return gate, metrics, ood_verdict_lines(suite, first, second, ood)


def decode_ood(
    step: RealFtStep | TowerEnsemble, suite: OodSuite
) -> tuple[dict[str, object], dict[str, object], dict[str, bool]]:
    """Both passes over the OOD suite -- as rendered, then deranged -- and the runtime rule's
    abstention per decoded case. What the ``ood_abstain`` gate and the OOD-only diagnostic
    (:func:`run_ood_diagnostic`) both decode, by the one rule."""
    assert suite.val is not None and suite.second_pass is not None  # set whenever it ran
    first = _decode(step, suite.val.plan, suite.val.labels_for, suite.val.letter_id)
    second = _decode(
        step, suite.second_pass.batches, suite.second_pass.labels_for, suite.val.letter_id
    )
    return first, second, choice_rule_abstentions(first, second, suite.second_pass.perms)


def ood_category_metrics(
    suite: OodSuite, ood: Mapping[str, bool], *, prefix: str, what: str
) -> dict[str, TriState]:
    """``<prefix>.<category>``: the runtime rule's abstentions on each OOD category's decoded
    cases. The one count the ``ood_abstain`` gate's report and the OOD-only diagnostic make."""
    metrics: dict[str, TriState] = {}
    for category in OOD_CATEGORIES:
        ids = [c.case_id for c in suite.cases if c.category == category and c.case_id in ood]
        k = sum(1 for i in ids if ood[i])
        metrics[f"{prefix}.{category}"] = (
            Ran(
                passed=True, value=k / len(ids), n=k, n_total=len(ids),
                detail=f"abstained on {k} of {len(ids)} {category} cases {what}",
            )
            if ids else NotRun(reason=f"no {category} case was decoded")
        )
    return metrics


def in_distribution_family_metrics(
    scored: Mapping[str, object], indist: Mapping[str, bool], gold_noul: Mapping[str, bool],
) -> dict[str, TriState]:
    """REPORT-ONLY breakdowns of ``ood_abstain.in_distribution``, never the gate.

    * ``ood_abstain.in_distribution.family.{family_id}``: the runtime rule's abstentions on
      one family's val choice rows, out of that family's rows in the bound -- the same
      ``indist`` the gate reads, so the families' ``n`` and ``n_total`` sum to the pooled
      metric's.
    * ``ood_abstain.in_distribution.gold_noul.family.{family_id}``: the rows the bound
      EXCLUDES by contract ("none of whose golds is noul", ``qd_train.ood``), and how many of
      them the same rule abstains on -- the right answer there. Not in any pooled number.

    Keyed by ``row_id`` like :func:`choice_rule_abstentions`. Every family with a val choice
    row gets both names; a family with no row on one side is :class:`NotRun` saying so.
    """
    family_of: dict[str, object] = {
        str(v["row_id"]): v.get("family_id")
        for v in scored["verdicts"]  # type: ignore[union-attr]
        if v["kind"] == "choice"
    }
    out: dict[str, TriState] = {}
    for prefix, abstentions, what, empty in (
        ("ood_abstain.in_distribution", indist,
         "val choice rows the runtime rule would abstain on, of this family's rows in the "
         "in-distribution bound (reported per family, not the gate)",
         "every {family} val choice row has a noul gold, which the bound excludes"),
        ("ood_abstain.in_distribution.gold_noul", gold_noul,
         "val choice rows whose gold is noul -- excluded from the in-distribution bound by "
         "its contract -- that the runtime rule abstains on (reported, in no pooled number)",
         "no {family} val choice row has a noul gold"),
    ):
        missing = [r for r in abstentions if not isinstance(family_of.get(r), str)]
        if missing:
            out[f"{prefix}.family"] = NotRun(
                reason=f"{len(missing)} of {len(abstentions)} choice rows {NO_FAMILY_REASON}"
            )
            continue
        by_family: dict[str, list[bool]] = {
            str(f): [] for f in family_of.values() if isinstance(f, str)
        }
        for row_id, abstained in abstentions.items():
            by_family[str(family_of[row_id])].append(abstained)
        for family, flags in sorted(by_family.items()):
            name = f"{prefix}.family.{family}"
            if not flags:
                out[name] = NotRun(reason=empty.format(family=family))
                continue
            k = sum(flags)
            out[name] = Ran(
                passed=True, value=k / len(flags), n=k, n_total=len(flags),
                detail=f"{k} of {len(flags)}: {what}",
            )
    return out


def ood_verdict_lines(
    suite: OodSuite, first: Mapping[str, object], second: Mapping[str, object],
    abstained: Mapping[str, bool],
) -> tuple[dict[str, object], ...]:
    """One raw verdict per decoded OOD case: both passes' top rows and letter distributions,
    the derangement between them, and whether the runtime rule abstained."""
    assert suite.second_pass is not None  # only called once the suite ran
    passes = [
        {
            str(v["row_id"]): v for v in decode["verdicts"]  # type: ignore[union-attr]
            if v["kind"] == "choice"
        }
        for decode in (first, second)
    ]
    out: list[dict[str, object]] = []
    for case in suite.cases:
        if case.case_id not in abstained:
            continue
        one = passes[0][case.case_id]
        two = passes[1].get(case.case_id)
        perm = suite.second_pass.perms.get((case.case_id, str(one["slot_name"])))
        out.append({
            "suite": "ood", "case_id": case.case_id, "category": case.category,
            "language": case.language, "abstained": abstained[case.case_id],
            "top1": one["top"], "top2": None if two is None else two["top"],
            "noul_row": one["noul_row"], "rows": one.get("rows"),
            "perm": None if perm is None else list(perm),
            "noul_probability_1": one.get("noul_probability"),
            "noul_probability_2": None if two is None else two.get("noul_probability"),
            "row_logits_1": one.get("row_logits"),
            "row_logits_2": None if two is None else two.get("row_logits"),
        })
    return tuple(out)


def suite_widths(
    needle_suite: NeedleSuite, ood_suite: OodSuite, *, composed: ComposedSlice | None = None,
) -> list[int]:
    """Every eval-only batch width the step will decode, for its ``max_width`` bound: the
    suites', and the composed slice's when a plan scores it (up to 7,801 tokens)."""
    widths = [int(b.tokens.shape[1]) for b in needle_suite.batches]
    if ood_suite.val is not None:
        widths += [int(b.tokens.shape[1]) for b in ood_suite.val.plan]
    if ood_suite.second_pass is not None:
        widths += [int(b.tokens.shape[1]) for b in ood_suite.second_pass.batches]
    if composed is not None:
        widths += [int(b.tokens.shape[1]) for b in composed.plan]
    return widths


def ood_recipe(suite: OodSuite) -> dict[str, object]:
    """What ``ood_abstain`` was scored under, for the recipe -- only when it ran."""
    return {
        "min_abstain": OOD_MIN_ABSTAIN, "max_in_distribution": OOD_MAX_IN_DISTRIBUTION_ABSTAIN,
        "cases_per_category": OOD_CASES_PER_CATEGORY, "categories": list(OOD_CATEGORIES),
        "general_record_sha256": suite.record_sha256,
        "general_max_rows": OOD_GENERAL_MAX_ROWS, "suite_seed": suite.seed,
        "rule": "noul-either-pass-or-permuted-disagreement; margin half not applied",
    }


@dataclasses.dataclass(frozen=True, slots=True)
class SuiteGate:
    """A gate scored on a generated suite: its state, its metrics, and -- only when it ran --
    what it was scored under, which goes in the eval row's recipe under ``recipe_key``."""

    name: str
    state: TriState
    metrics: Mapping[str, TriState]
    recipe_key: str
    recipe: Mapping[str, object] | None
    #: The raw per-case verdicts the gate was scored from, for ``--suite-verdicts-out``.
    verdicts: tuple[dict[str, object], ...] = ()


SuiteScore = tuple[TriState, dict[str, TriState], tuple[dict[str, object], ...]]


def needle_gate(scored: SuiteScore, suite: NeedleSuite) -> SuiteGate:
    return SuiteGate("needle_hunk_recall", scored[0], scored[1], "needle",
                     None if suite.not_run is not None else needle_recipe(suite), scored[2])


def ood_suite_gate(scored: SuiteScore, suite: OodSuite) -> SuiteGate:
    return SuiteGate("ood_abstain", scored[0], scored[1], "ood",
                     None if suite.not_run is not None else ood_recipe(suite), scored[2])


def suite_verdict_lines(
    gates: Sequence[SuiteGate], *, eval_row_id: str, seed: int
) -> list[dict[str, object]]:
    """The JSONL lines ``--suite-verdicts-out`` writes for one eval row: every suite gate's
    raw per-case verdicts, each naming the row and seed it was scored under."""
    return [
        {"eval_row_id": eval_row_id, "seed": int(seed), "gate": g.name, **v}
        for g in gates for v in g.verdicts
    ]


def write_suite_verdicts_jsonl(path: Path, lines: Sequence[Mapping[str, object]]) -> None:
    """Every line to ``path``, atomically, refusing to overwrite."""
    from qd_train.replay import ReplayRefusal as _Refusal
    from qd_train.replay import write_text_atomic

    for i, line in enumerate(lines):
        missing = [k for k in ("eval_row_id", "seed", "gate", "suite", "case_id") if k not in line]
        if missing:
            raise ValueError(f"suite verdict line {i} is missing {missing}")
    try:
        write_text_atomic(path, "".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    except _Refusal as exc:
        raise SystemExit(f"--suite-verdicts-out: {exc}") from exc


def _record_score(run: dict[str, object], scored: dict[str, object], *, ledger: Ledger,
                  reader: ShardReader, val: ValSet, quick_reasons: Sequence[str],
                  decode_s: float, permutation: TriState,
                  suite_gates: Sequence[SuiteGate] = ()) -> str:
    """One ``eval`` row per epoch run: what its model does on the val set.

    Pinned to the TRAIN set's protocol, like the verdict row, so the rows of one
    configuration group by seed; the val set is named in the recipe by its shard hash.
    """
    recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py", "tag": f"{run['tag']}-score-val",
        "device": run["device"],
        **{k: run[k] for k in (*BACKBONE_KEYS, *RECIPE_PIECE_KEYS) if k in run},
        **{k: run[k] for k in SCORED_CHECKPOINT_KEYS if k in run},
        "shard_hash": reader.header.shard_hash(),
        "val_shard_hash": val.reader.header.shard_hash(),
    }
    for suite_gate in suite_gates:
        if suite_gate.recipe is not None:
            recipe[suite_gate.recipe_key] = dict(suite_gate.recipe)
    averaged = run.get("averaged")
    model = run.get("scored_model")
    if model is not None and not isinstance(model, ScoredModel):  # pragma: no cover
        raise TypeError(f"run['scored_model'] is {type(model).__name__}, not a ScoredModel")
    if model is not None and model.block_key == "ensemble":
        notes = (
            f"tools/real_ft_run.py --score-checkpoint of {model.described}, scored in "
            f"{run['score_dtype']} on {run['device']} at T = 1 on {len(val.reader)} val "
            "sequences none of them trained on, the way crates/qd-runtime/src/answer.rs "
            "decodes them, every rule (abstention, the permuted second pass, the needle "
            "pointer, the OOD rule) applied to the ensemble's averaged log-probabilities. The "
            "permuted second pass and the needle and OOD suites are built at the protocol seed "
            f"{run['seed']}, as every per-seed score row's are. " + ENSEMBLE_PROMOTION_NOTE
        )
    elif model is not None and model.tag == AVERAGED_NP_TAG:
        notes = (
            f"tools/real_ft_run.py --score-checkpoint of {model.described}, "
            f"{run['scored_checkpoint']}, scored in {run['score_dtype']} on {run['device']} at "
            f"T = 1 on {len(val.reader)} val sequences none of them trained on, the way "
            "crates/qd-runtime/src/answer.rs decodes them. "
            + (
                "lambda was GIVEN, not derived from the measured c, and this row is quick "
                "for it. " if model.lambda_chosen else
                "lambda is a function of the measured c (Fable I-2), never chosen on a gate. "
            )
            + "The permuted second pass and the needle and OOD suites are built at the "
            f"protocol seed {run['seed']}, as every per-seed score row's are. "
            + AVERAGED_PROMOTION_NOTE
        )
    elif model is not None and model.trained_by is not None:
        notes = (
            f"tools/real_ft_run.py --score-checkpoint of {model.described}, scored in "
            f"{run['score_dtype']} on {run['device']} at T = 1 on {len(val.reader)} val "
            "sequences it never trained on, the way crates/qd-runtime/src/answer.rs decodes "
            "them. The weights are another trainer's export, not a checkpoint this tool "
            f"wrote; recipe.{TRAINED_BY_KEY} names who trained them and on what."
        )
    elif averaged is None:
        notes = (
            f"tools/real_ft_run.py --score-val for ft row {run['ft_row_id']} "
            f"({run['device']} seed={run['seed']}): the epoch model decoded on "
            f"{len(val.reader)} val sequences it never trained on, the way "
            "crates/qd-runtime/src/answer.rs decodes them."
            + (
                f" Scored from the saved checkpoint {run['scored_checkpoint']} in "
                f"{run['score_dtype']} on {run['device']}, not in the training process; "
                "a different device and dtype from the ft row's own score row."
                if "scored_checkpoint" in run else ""
            )
        )
    else:
        assert isinstance(averaged, Mapping)  # what _averaged_weights writes
        notes = (
            f"tools/real_ft_run.py --score-checkpoint of an AVERAGE, {run['scored_checkpoint']}"
            f": the mean of seeds {averaged['seeds']}' weights taken from their "
            f"{averaged['source']} (ft rows {', '.join(averaged['ft_row_ids'])}; manifest "
            f"sha256 {averaged['manifest_sha256']}), scored in {run['score_dtype']} on "
            f"{run['device']} on {len(val.reader)} val sequences none of them trained on, the "
            "way crates/qd-runtime/src/answer.rs decodes them. The permuted second pass and "
            f"the needle and OOD suites are built at the protocol seed "
            f"{averaged['suite_seed']}, as every per-seed score row's are. "
            + AVERAGED_PROMOTION_NOTE
        )
    plan_note = run.get("plan_note")
    if plan_note is not None:
        notes = f"{notes} {plan_note}"
    recorder = _recorder(
        ledger, reader=reader, seed=int(run["seed"]), recipe=recipe, run_kind="eval",
        quick_reasons=quick_reasons,
        # The decode, not the training run: see _record_verdict.
        wall_clock_s=decode_s,
        cost=run["cost"],  # type: ignore[arg-type]
        notes=notes,
    )
    with recorder:
        if model is None:
            recorder.metric(
                "ft_run_row_id",
                Ran(passed=True, value=run["ft_row_id"],
                    detail="the train_ft row whose model this is"),
            )
        else:
            # ft_run_row_ids for an average or an ensemble: ft_run_row_id names ONE row, and
            # shuffled_label_target reads it so.
            recorder.metric(*model.ft_row_metric())
        for name, state in score_states(scored, val.labels).items():
            recorder.metric(name, state)
        calibration, ece, degenerate = calibration_states(scored)
        for name, state in calibration.items():
            recorder.metric(name, state)
        recorder.gate("ece", ece)
        recorder.control("degenerate_head", degenerate)
        recorder.gate("permutation_consistency", permutation)
        for name, state in permutation_family_metrics(scored, permutation).items():
            recorder.metric(name, state)
        for suite_gate in suite_gates:
            for name, state in suite_gate.metrics.items():
                recorder.metric(name, state)
            recorder.gate(suite_gate.name, suite_gate.state)
        decoded_abstain = sum(
            1 for v in scored["verdicts"]  # type: ignore[union-attr]
            if str(v["runtime_verdict"]) == "abstain"
        )
        rows = len(scored["verdicts"])  # type: ignore[arg-type]
        recorder.metric(
            "val_decoded_as_abstain",
            Ran(
                passed=True, value=decoded_abstain, n=decoded_abstain, n_total=rows,
                detail=(
                    "val rows whose runtime verdict was the abstention, over every kind. A "
                    "count, not the gate: the letter channel supervises no abstention"
                ),
            ),
        )
        recorder.metric("val_shard_coverage", val.reader.coverage)
        decoded, skipped = int(scored["rows_decoded"]), int(scored["rows_not_decoded"])  # type: ignore[call-overload]
        recorder.metric(
            "val_rows_decoded",
            Ran(
                passed=skipped == 0, value=decoded, n=decoded, n_total=decoded + skipped,
                detail=(
                    "every val row decoded" if not skipped else
                    f"{skipped} row(s) offer letter(s) {scored['letters_without_id']} with no "
                    "known token id and were not decoded; every score on this row is over "
                    f"the {decoded} that were"
                ),
            ),
        )
        recorder.noul_rate = NotRun(
            reason=(
                "this corpus supervises no letter-channel abstention, so an abstain rate "
                "over its val rows would describe the corpus, not the model; the decoded "
                "count is recorded as val_decoded_as_abstain"
            )
        )
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id


#: Keys a ``--score-checkpoint`` run adds to its eval row's recipe -- only then, so every
#: score row written by a training run keeps the recipe hash it always had. ``averaged`` is
#: on a scored average's row alone: its seeds, ft rows, manifest sha256 and source.
#: ``ensemble`` is on a scored logit ensemble's row alone: its seeds, ft rows, each tower's
#: checkpoint and digests, and how the towers are combined. ``trained_by`` is on a row
#: scoring a model another trainer made (a Metal export) alone: that trainer, the device it
#: trained on, the export's source and its manifest sha256.
SCORED_CHECKPOINT_KEYS: Final[tuple[str, ...]] = (
    "score_dtype", "scored_checkpoint", "averaged", "ensemble", "trained_by",
)

#: What ``--score-checkpoint`` given this suffix scores: an average ``tools/ckpt_average.py``
#: wrote, with its ``.manifest.json`` beside it -- unless the name is a Metal export's
#: (:func:`_is_metal_export`). Anything else is one seed's ``<tag>-seed<N>-<device>.json``.
AVERAGED_SUFFIX: Final[str] = ".safetensors"
#: A model the Rust trainer (``crates/qd-train``, over canonical tessl's Metal kernels)
#: trained: human ask 7 of ``AUDIT/ojas-training-2026-10-01/fable-advice.md``, a scoring
#: input only. One seed's ``tower.*`` and ``span_head.*``, named as every per-seed checkpoint
#: is but as a ``.safetensors`` -- ``<tag>-seed<N>-metal.safetensors`` -- with the Rust
#: export's manifest at ``<name>.manifest.json`` (:func:`_metal_export_weights`).
METAL_DEVICE: Final[str] = "metal"
#: The trainer a Metal export's manifest and its ft row's ``recipe.trainer`` both name.
METAL_TRAINER: Final[str] = "qd-train-metal"
#: A Metal export's manifest ``from``: the Rust export. An average's is one of
#: ``ckpt_average.SOURCES``; a ``from`` that is neither is refused.
METAL_EXPORT_SOURCE: Final[str] = "qd-train-export"
#: The recipe block every row scoring a model another trainer made carries
#: (:meth:`ScoredModel.recipe_block`), so it never reads as a model this tool trained.
TRAINED_BY_KEY: Final[str] = "trained_by"
#: A Metal export's manifest fields and their JSON types; each one is checked.
METAL_MANIFEST_FIELDS: Final[Mapping[str, type | tuple[type, ...]]] = {
    "from": str, "trainer": str, "device": str, "seed": int, "optimizer_step": int,
    "ft_row_id": str, "vocab_size": int, "span_weight": (int, float),
    "safetensors_sha256": str, "n_tensors": int,
}
#: What the scoring step is built from (:func:`_scoring_step`), which a Metal ft row's
#: recipe must therefore name itself: a ``.json`` checkpoint's older rows may lack
#: ``optimizer_recipe`` and are read at ``bf16``, a new trainer's are not guessed at.
METAL_ROW_RECIPE_KEYS: Final[tuple[str, ...]] = (
    "attn_implementation", "lr", "span_weight", "optimizer_recipe",
)
_SHA256_HEX: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
#: A scored average's run tag; :func:`_record_score` makes its recipe tag ``avg-score-val``,
#: as an epoch arm's is ``epoch-score-val``.
AVERAGED_TAG: Final[str] = "avg"
#: A scored NORM-PRESERVING average's run tag (``ckpt_average.py --norm-preserving``, Fable
#: I-2): recipe tags ``avg-np-score-val`` and ``avg-np-needle-length-control``.
AVERAGED_NP_TAG: Final[str] = "avg-np"
#: Why a norm-preserving average's row is quick when its lambda was not derived from c.
LAMBDA_CHOSEN_REASON: Final[str] = (
    "a norm-preserving average whose lambda was given (--lambda-not-derived-from-c), not "
    "derived from the measured c: a chosen number could have been chosen on a gate"
)
#: Rule 8 for an average: fewer inputs than this is a seed shortfall, and the row is quick.
MIN_AVERAGED_SEEDS: Final[int] = 3
#: On every scored average's row. Rule 8's seed count and the promotion join are about seed
#: families; an average is one row of a family of its own, and nothing here decides whether
#: it may stand in for its three.
AVERAGED_PROMOTION_NOTE: Final[str] = (
    "Whether an average can be the promoted artifact is the human's decision; this row "
    "does not make it."
)
#: A scored logit ensemble's run tag is this plus its tower count -- ``ens3`` -- and its
#: recipe tag ``ens3-score-val``. Rule 8 counts its seeds as an average's
#: (:data:`MIN_AVERAGED_SEEDS`).
ENSEMBLE_TAG_PREFIX: Final[str] = "ens"
#: On every scored ensemble's row: what may serve is the human's decision, as for an average.
ENSEMBLE_PROMOTION_NOTE: Final[str] = (
    "Whether a logit ensemble can be the promoted artifact is the human's decision; this row "
    "does not make it, and serving N towers is a runtime change of its own."
)


def score_paths(value: Path | Sequence[Path]) -> tuple[Path, ...]:
    """``--score-checkpoint`` as a tuple: one path (one seed's checkpoint, or an average) or
    several (the per-seed checkpoints of a logit ensemble)."""
    if isinstance(value, Path):
        return (value,)
    paths = tuple(value)
    if not paths or not all(isinstance(p, Path) for p in paths):
        raise SystemExit(f"--score-checkpoint {value!r}: one or more checkpoint paths")
    return paths


def ft_row_ids(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """``--ft-row-id`` as a tuple: none (an average names its own), one, or one per tower."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _is_average(path: Path | Sequence[Path]) -> bool:
    paths = score_paths(path)
    return (
        len(paths) == 1 and paths[0].suffix == AVERAGED_SUFFIX and not _is_metal_export(paths)
    )


def _is_metal_export(path: Path | Sequence[Path]) -> bool:
    """One ``.safetensors`` whose name ends in ``-metal``: one seed's Metal export, routed by
    its device as every per-seed checkpoint is. :func:`_metal_export_seed` refuses any other
    part of the name it cannot read, and the manifest refuses an average named like one."""
    paths = score_paths(path)
    return (
        len(paths) == 1 and paths[0].suffix == AVERAGED_SUFFIX
        and paths[0].stem.endswith(f"-{METAL_DEVICE}")
    )


def _is_ensemble(path: Path | Sequence[Path]) -> bool:
    return len(score_paths(path)) > 1


@dataclasses.dataclass(frozen=True, slots=True)
class ScoredModel:
    """What a ``--score-checkpoint`` row says about the model it scored -- one owner for the
    three kinds, so the score row, the needle-control row and the diagnostic rows name a
    seed's checkpoint, an average and an ensemble the same way.

    ``tag`` is the run tag (``epoch``, ``avg``, ``avg-np``, ``ens3``); ``block_key`` and
    ``block`` are the recipe's provenance block (``averaged`` or ``ensemble``), absent for
    one seed's checkpoint, which names its ft row under ``ft_run_row_id`` instead.
    """

    tag: str
    scored_checkpoint: str
    terminations: tuple[str | None, ...]
    ft_row_ids: tuple[str, ...]
    block_key: str | None
    block: Mapping[str, object] | None
    #: For notes: what the model is, in a phrase.
    described: str
    #: A norm-preserving average whose lambda was given rather than derived from c.
    lambda_chosen: bool = False
    #: A model another trainer made (a Metal export): the :data:`TRAINED_BY_KEY` block every
    #: row scoring it carries. ``None`` for a model this tool trained.
    trained_by: Mapping[str, object] | None = None
    #: That model's ft row's ``quick_reason`` when the row is quick, which its rows carry.
    ft_quick_reason: str | None = None

    @property
    def n_inputs(self) -> int:
        return len(self.ft_row_ids)

    def recipe_block(self) -> dict[str, object]:
        """``{block_key: block}``, or nothing for one seed's checkpoint; and the
        :data:`TRAINED_BY_KEY` block for a model another trainer made."""
        block: dict[str, object] = (
            {} if self.block_key is None or self.block is None
            else {self.block_key: dict(self.block)}
        )
        if self.trained_by is not None:
            block[TRAINED_BY_KEY] = dict(self.trained_by)
        return block

    def ft_row_metric(self) -> tuple[str, TriState]:
        """``ft_run_row_id`` for one seed's model; ``ft_run_row_ids`` for any other, never
        the singular -- ``shuffled_label_target`` reads that as ONE row."""
        if self.block_key is None:
            return "ft_run_row_id", Ran(
                passed=True, value=self.ft_row_ids[0],
                detail="the train_ft row whose model this is",
            )
        detail = (
            "the train_ft rows whose weights were averaged into this model"
            if self.block_key == "averaged" else
            "the train_ft rows whose towers this logit ensemble decodes together"
        )
        return "ft_run_row_ids", Ran(passed=True, value=",".join(self.ft_row_ids), detail=detail)

    def seed_reasons(self) -> list[str]:
        """Rule 8 for a several-seed model: fewer than :data:`MIN_AVERAGED_SEEDS` is quick.
        And a norm-preserving average whose lambda was chosen, not derived from c: a number
        a person picked could have been picked on a gate."""
        reasons: list[str] = []
        if self.block_key is not None and self.n_inputs < MIN_AVERAGED_SEEDS:
            what = "an average" if self.block_key == "averaged" else "an ensemble"
            reasons.append(
                f"{what} of {self.n_inputs} seeds: rule 8 marks fewer than "
                f"{MIN_AVERAGED_SEEDS} seeds quick"
            )
        if self.lambda_chosen:
            reasons.append(LAMBDA_CHOSEN_REASON)
        return reasons

    def provenance_reasons(self) -> list[str]:
        """Rule 8 carried from the ft row of a model another trainer made: a model a quick
        run trained is no less quick, whatever device scores it."""
        if self.ft_quick_reason is None or self.trained_by is None:
            return []
        return [
            f"ft row {self.ft_row_ids[0]}, which {self.trained_by['trainer']} trained on "
            f"{self.trained_by['device']}, is quick ({self.ft_quick_reason}); a model a quick "
            "run trained is no less quick"
        ]


def scored_model(
    args: argparse.Namespace, ft: Mapping[str, Any], meta: Mapping[str, Any]
) -> ScoredModel:
    """:class:`ScoredModel` for what :func:`_checkpoint_step` loaded."""
    averaged = meta.get("averaged")
    ensemble = meta.get("ensemble")
    if averaged is not None:
        assert isinstance(averaged, Mapping)  # what _averaged_weights writes
        norm = averaged.get("norm_preserving")
        if norm is not None:
            assert isinstance(norm, Mapping)  # what _averaged_weights writes
            return ScoredModel(
                tag=AVERAGED_NP_TAG,
                scored_checkpoint=str(meta["scored_checkpoint"]),
                terminations=tuple(meta["terminations"]),
                ft_row_ids=tuple(averaged["ft_row_ids"]),
                block_key="averaged", block=averaged,
                described=(
                    f"a NORM-PRESERVING AVERAGE of seeds {averaged['seeds']} from their "
                    f"{averaged['source']} (base {norm['base_snapshot']}, c = {norm['c']:.6f}, "
                    f"lambda = {norm['lambda']:.6f}"
                    + ("" if norm["lambda_derived_from_c"] else ", NOT derived from c")
                    + f"), ft rows {', '.join(averaged['ft_row_ids'])}, manifest sha256 "
                    f"{averaged['manifest_sha256']}"
                ),
                lambda_chosen=not bool(norm["lambda_derived_from_c"]),
            )
        return ScoredModel(
            tag=AVERAGED_TAG,
            scored_checkpoint=str(meta["scored_checkpoint"]),
            terminations=tuple(meta["terminations"]),
            ft_row_ids=tuple(averaged["ft_row_ids"]),
            block_key="averaged", block=averaged,
            described=(
                f"an AVERAGE of seeds {averaged['seeds']} from their {averaged['source']}, "
                f"ft rows {', '.join(averaged['ft_row_ids'])}, manifest sha256 "
                f"{averaged['manifest_sha256']}"
            ),
        )
    if ensemble is not None:
        assert isinstance(ensemble, Mapping)  # what _ensemble_inputs writes
        return ScoredModel(
            tag=f"{ENSEMBLE_TAG_PREFIX}{ensemble['n_towers']}",
            scored_checkpoint=str(meta["scored_checkpoint"]),
            terminations=tuple(meta["terminations"]),
            ft_row_ids=tuple(ensemble["ft_row_ids"]),
            block_key="ensemble", block=ensemble,
            described=(
                f"a {ensemble['n_towers']}-tower LOGIT ENSEMBLE of seeds {ensemble['seeds']} "
                f"(ft rows {', '.join(ensemble['ft_row_ids'])}; towers "
                + ", ".join(
                    f"{t['checkpoint']}:{t['sidecar_digest']}" for t in ensemble["towers"]
                )
                + f"), combined as the {ensemble['combine']}"
            ),
        )
    trained_by = meta.get(TRAINED_BY_KEY)
    if trained_by is not None:
        assert isinstance(trained_by, Mapping)  # what _metal_export_weights writes
        return ScoredModel(
            tag="epoch",
            scored_checkpoint=str(meta["scored_checkpoint"]),
            terminations=(ft["metrics"]["train.termination"]["value"],),
            ft_row_ids=(str(ft["row_id"]),),
            block_key=None, block=None,
            described=(
                f"ft row {ft['row_id']}, a model TRAINED BY {trained_by['trainer']} on "
                f"{trained_by['device']} -- not by this tool -- read from its export "
                f"{meta['scored_checkpoint']} (manifest sha256 {trained_by['manifest_sha256']})"
            ),
            trained_by=trained_by,
            ft_quick_reason=str(ft["quick_reason"]) if ft["quick"] else None,
        )
    path = score_paths(args.score_checkpoint)[0]
    return ScoredModel(
        tag="epoch",
        scored_checkpoint=f"{path.name}:{meta['sidecar']['digest']}",
        terminations=(ft["metrics"].get("train.termination", {}).get("value"),),
        ft_row_ids=(str(ft["row_id"]),),
        block_key=None, block=None,
        described=f"ft row {ft['row_id']}",
    )


def _ft_row(ledger_path: Path, row_id: str) -> dict[str, Any]:
    """The one ft row ``row_id`` names (a full id or a unique prefix), or a refusal."""
    if len(row_id) < 8:
        raise SystemExit(f"--ft-row-id {row_id!r}: give at least 8 characters of the row id")
    found = [
        row for row in (
            json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if str(row.get("row_id", "")).startswith(row_id)
    ]
    if len(found) != 1:
        raise SystemExit(f"{ledger_path}: {len(found)} rows match --ft-row-id {row_id!r}, not 1")
    row = found[0]
    if row.get("run_kind") != "ft" or row.get("status") != "completed":
        raise SystemExit(
            f"row {row['row_id']} is a {row.get('status')} {row.get('run_kind')!r} row; "
            "--score-checkpoint scores the model a completed ft row trained"
        )
    return row


#: What :func:`_checkpoint_step` returns: ``(step, ft row, its recipe, seed, meta)``.
LoadedModel = tuple[Any, dict[str, Any], dict[str, Any], int, dict[str, Any]]


def _checkpoint_step(
    args: argparse.Namespace, *, reader: ShardReader, val: ValSet | ValPlan, device: str,
    eval_widths: Sequence[int], suite_seed: int,
) -> LoadedModel:
    """``--score-checkpoint``'s weights in a step, every pairing checked before they load.

    Returns ``(step, ft row, its recipe, seed, checkpoint meta)``. The one
    loader for both the scoring process and its needle worker, for one seed's checkpoint
    (:func:`_seed_weights`), one seed's Metal export (:func:`_metal_export_weights`) and an
    average (:func:`_averaged_weights`): only where the weights come from and what is checked
    about them differ, never the step they load into.

    ``seed`` is the eval row's protocol seed: the checkpoint's own for one seed's, and
    ``suite_seed`` -- the protocol seed every suite is built at -- for an average or an
    ensemble, which is no one seed's model. ``meta['optimizer_step']`` is the schedule the
    step is built for. ``val`` is read for its plan alone, the batch shapes the step is sized
    by, so the needle worker passes a :class:`ValPlan` and the scoring process its
    :class:`ValSet`.

    A logit ensemble (several ``--score-checkpoint`` files) returns a :class:`TowerEnsemble`
    of one step per tower: every tower is paired with its ft row and every body checked
    (:func:`_ensemble_inputs`) before any tensor is read, then each tower's weights are read,
    loaded and dropped in turn.
    """
    from qd_train.run_control import Checkpoint

    if _is_ensemble(args.score_checkpoint):
        ft, recipe, seed, meta, paths = _ensemble_inputs(
            args, reader=reader, suite_seed=suite_seed
        )
        towers: list[Any] = []
        for path, tower in zip(paths, meta["ensemble"]["towers"], strict=True):
            weights, header = Checkpoint.read_weights(path, subtrees=("tower", "span_head"))
            if header["payload_digest"] != tower["payload_digest"]:
                raise SystemExit(
                    f"{path.name} changed between its pairing and its read (payload_digest "
                    f"{str(header['payload_digest'])[:16]} vs {tower['payload_digest'][:16]})"
                )
            step = _scoring_step(
                args, reader=reader, val=val, device=device, recipe=recipe, seed=seed,
                optimizer_step=int(meta["optimizer_step"]), eval_widths=eval_widths,
            )
            step.load_weights(weights)
            del weights
            towers.append(step)
        return TowerEnsemble(towers), ft, recipe, seed, meta
    if _is_average(args.score_checkpoint):
        ft, recipe, seed, meta, weights = _averaged_weights(
            args, reader=reader, suite_seed=suite_seed
        )
    elif _is_metal_export(args.score_checkpoint):
        ft, recipe, seed, meta, weights = _metal_export_weights(args, reader=reader)
    else:
        ft, recipe, seed, meta, weights = _seed_weights(args, reader=reader)
    step = _scoring_step(
        args, reader=reader, val=val, device=device, recipe=recipe, seed=seed,
        optimizer_step=int(meta["optimizer_step"]), eval_widths=eval_widths,
    )
    step.load_weights(weights)
    return step, ft, recipe, seed, meta


def _scoring_step(
    args: argparse.Namespace, *, reader: ShardReader, val: ValSet | ValPlan, device: str,
    recipe: Mapping[str, Any], seed: int, optimizer_step: int, eval_widths: Sequence[int],
) -> Any:
    """The step a scored checkpoint's weights load into: the ft row's recipe, built at
    ``--score-dtype``, bounded by the val plan's widest batch and ``eval_widths``."""
    step, _, _ = _real_step(
        backbone=args.real_backbone, reader=reader, plan=val.plan, device=device,
        dtype=args.score_dtype,
        spec=optimizer_spec(args.score_dtype, str(recipe.get("optimizer_recipe", "bf16"))),
        attn_implementation=str(recipe["attn_implementation"]), seed=seed,
        lr=float(recipe["lr"]), total_steps=optimizer_step,
        span_weight=float(recipe["span_weight"]),
        width=max(int(b.tokens.shape[1]) for b in val.plan), eval_widths=eval_widths,
    )
    return step


def _ft_row_mismatches(
    ft: Mapping[str, Any], *, seed: int, trained_on: str, reader: ShardReader,
    real_backbone: Path,
) -> dict[str, tuple[object, object]]:
    """What in ``ft`` does not describe one seed's epoch checkpoint scored here: its arm, its
    device, its seed, its shard set and its backbone. Empty when it does."""
    recipe = ft["recipe"]
    expected = {
        "recipe tag": (recipe.get("tag"), "epoch"),
        "recipe device": (recipe.get("device"), trained_on),
        "protocol seed": (ft["protocol"]["seed"], seed),
        "shard_hash": (recipe.get("shard_hash"), reader.header.shard_hash()),
        "backbone_snapshot": (recipe.get("backbone_snapshot"), real_backbone.name),
    }
    return {k: v for k, v in expected.items() if v[0] != v[1]}


def _seed_weights(
    args: argparse.Namespace, *, reader: ShardReader
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    """One seed's ``<tag>-seed<N>-<device>.json``: ``(ft row, recipe, seed, meta, weights)``,
    the file paired with ``--ft-row-id``'s row before its tower and span head are read."""
    from qd_train.run_control import Checkpoint

    (path,) = score_paths(args.score_checkpoint)
    (row_id,) = ft_row_ids(args.ft_row_id)
    tag, seed, trained_on = _resume_arm(path)
    if tag != "epoch":
        raise SystemExit(f"{path.name}: only an epoch checkpoint is scored")
    if list(args.seeds) != [seed]:
        raise SystemExit(f"{path.name} is seed {seed}; --seeds says {args.seeds}")
    ft = _ft_row(args.ft_ledger, row_id)
    recipe = ft["recipe"]
    wrong = _ft_row_mismatches(
        ft, seed=seed, trained_on=trained_on, reader=reader, real_backbone=args.real_backbone
    )
    if wrong:
        raise SystemExit(
            f"ft row {ft['row_id']} does not describe {path.name} scored "
            f"against this shard set and backbone: "
            + "; ".join(f"{k}: row says {a!r}, here {b!r}" for k, (a, b) in wrong.items())
        )
    steps = int(ft["metrics"]["train.optimizer_steps"]["value"])
    weights, meta = Checkpoint.read_weights(path, subtrees=("tower", "span_head"))
    if meta["optimizer_step"] != steps or meta["seed"] != seed:
        raise SystemExit(
            f"{path.name} is at optimizer step {meta['optimizer_step']}, seed "
            f"{meta['seed']}; ft row {ft['row_id']} ended at step {steps}, seed {seed}. This "
            "is not the model that row trained."
        )
    return ft, recipe, seed, meta, weights


def _metal_export_seed(path: Path) -> int:
    """The seed in a Metal export's name, ``epoch-seed<N>-metal.safetensors``, or a
    refusal: only an epoch arm is scored, and a name that does not say its seed is not one
    the Rust export writes."""
    parts = path.stem.rsplit("-", 2)
    if (len(parts) != 3 or parts[2] != METAL_DEVICE or not parts[1].startswith("seed")
            or not parts[1][4:].isdigit()):
        raise SystemExit(
            f"Metal export {path.name}: not a name the Rust export writes. Expected "
            f"<tag>-seed<N>-{METAL_DEVICE}{AVERAGED_SUFFIX}, which says the seed it was taken at"
        )
    if parts[0] != "epoch":
        raise SystemExit(f"Metal export {path.name}: only an epoch checkpoint is scored")
    return int(parts[1][4:])


def _read_metal_manifest(path: Path) -> AverageManifest:
    """The Rust export's manifest beside ``path``, every field present and of its type, its
    source, trainer and device the Metal export's; refused otherwise. Returned as the
    ``AverageManifest`` ``read_average`` reads the weights through -- the same file layout,
    never an average: nothing here reads its ``inputs`` or ``ft_row_ids``."""
    where = f"Metal export {path.name}"
    file = manifest_path(path)
    if not file.is_file():
        raise SystemExit(
            f"{where} has no manifest at {file}; without it nothing says which ft row, step "
            "and seed these weights are"
        )
    raw = file.read_bytes()
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{where}: its manifest {file} is not JSON ({exc})") from exc
    if not isinstance(body, dict):
        raise SystemExit(f"{where}: its manifest {file} is not a JSON object")
    source = body.get("from")
    if source in AVERAGE_SOURCES:
        raise SystemExit(
            f"{where}: its manifest's 'from' is {source!r}, an average's manifest "
            f"(tools/ckpt_average.py). An average is not named <tag>-seed<N>-{METAL_DEVICE}"
            f"{AVERAGED_SUFFIX}; this name says one seed's Metal export"
        )
    if source != METAL_EXPORT_SOURCE:
        raise SystemExit(
            f"{where}: its manifest's 'from' is {source!r}, neither a Metal export's "
            f"{METAL_EXPORT_SOURCE!r} nor an average's {AVERAGE_SOURCES}"
        )
    problems: list[str] = []
    for key, kind in METAL_MANIFEST_FIELDS.items():
        value = body.get(key)
        if not isinstance(value, kind) or isinstance(value, bool):
            kinds = kind if isinstance(kind, tuple) else (kind,)
            problems.append(f"{key} is {value!r}, not {' or '.join(k.__name__ for k in kinds)}")
    if not problems:
        if body["trainer"] != METAL_TRAINER:
            problems.append(f"trainer is {body['trainer']!r}, not {METAL_TRAINER!r}")
        if body["device"] != METAL_DEVICE:
            problems.append(
                f"device is {body['device']!r}, not {METAL_DEVICE!r}: the name says this is "
                "a Metal export, and the export says it trained elsewhere"
            )
        if _SHA256_HEX.fullmatch(body["safetensors_sha256"]) is None:
            problems.append("safetensors_sha256 is not a lowercase hex sha256")
    if problems:
        raise SystemExit(
            f"{where}: its manifest {file} is not one a Metal export can be scored from: "
            + "; ".join(problems)
        )
    return AverageManifest(
        weights=path, path=file, sha256=hashlib.sha256(raw).hexdigest(), body=body
    )


def _metal_row_problems(ft: Mapping[str, Any]) -> list[str]:
    """What a Metal export's ft row lacks beyond :func:`_ft_row_mismatches`: the trainer
    that wrote it, what the scoring step is built from, whether it is quick (rule 8, stated
    either way), and how its run ended."""
    recipe = ft["recipe"]
    problems: list[str] = []
    if recipe.get("trainer") != METAL_TRAINER:
        problems.append(
            f"recipe trainer: row says {recipe.get('trainer')!r}, a Metal export's ft row "
            f"says {METAL_TRAINER!r}"
        )
    missing = [k for k in METAL_ROW_RECIPE_KEYS if k not in recipe]
    if missing:
        problems.append(f"its recipe names no {missing}, which the scoring step is built from")
    quick = ft.get("quick")
    if not isinstance(quick, bool):
        problems.append(
            f"quick is {quick!r}: a Metal ft row says whether it is quick (rule 8), true or "
            "false, and its rows inherit it"
        )
    elif quick and not (isinstance(ft.get("quick_reason"), str) and ft["quick_reason"].strip()):
        problems.append("quick is true and quick_reason says nothing")
    metrics = ft.get("metrics") or {}
    steps = metrics.get("train.optimizer_steps", {}).get("value")
    if not isinstance(steps, int) or isinstance(steps, bool):
        problems.append(f"metrics train.optimizer_steps is {steps!r}, not a step count")
    termination = metrics.get("train.termination", {}).get("value")
    if not isinstance(termination, str):
        problems.append(
            f"metrics train.termination is {termination!r}: a run that does not say how it "
            "ended cannot say its schedule was not truncated"
        )
    return problems


def _metal_export_weights(
    args: argparse.Namespace, *, reader: ShardReader
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    """One seed's Metal export (``epoch-seed<N>-metal.safetensors``):
    ``(ft row, recipe, seed, meta, weights)``, every pairing checked before a tensor is read.

    * the name is an epoch arm's and its seed is ``--seeds``' one;
    * the manifest (:func:`_read_metal_manifest`) says ``from`` :data:`METAL_EXPORT_SOURCE`,
      ``trainer`` :data:`METAL_TRAINER` and ``device`` :data:`METAL_DEVICE`, and names its
      seed, optimizer step, ft row, ``vocab_size``, ``span_weight``, ``n_tensors`` and
      ``safetensors_sha256``;
    * ``--ft-row-id``'s row describes it as a ``.json`` checkpoint's row must
      (:func:`_ft_row_mismatches`: an epoch arm, on ``metal``, at its seed, on this run's
      shard set and backbone) and as a Metal export's must (:func:`_metal_row_problems`);
    * the manifest names that row, its step and its seed, and the row's span weight;
    * the ``.safetensors`` hashes to the manifest's sha256 and holds only ``tower.*`` and
      ``span_head.*``, ``n_tensors`` of them (``ckpt_average.read_average``).

    ``meta[TRAINED_BY_KEY]`` is what every row scoring it records about who trained it.
    """
    (path,) = score_paths(args.score_checkpoint)
    (row_id,) = ft_row_ids(args.ft_row_id)
    where = f"Metal export {path.name}"
    seed = _metal_export_seed(path)
    if list(args.seeds) != [seed]:
        raise SystemExit(f"{where} is seed {seed}; --seeds says {args.seeds}")
    manifest = _read_metal_manifest(path)
    body = manifest.body
    ft = _ft_row(args.ft_ledger, row_id)
    protocol = ft.get("protocol")
    if not isinstance(ft.get("recipe"), Mapping) or not (
        isinstance(protocol, Mapping) and isinstance(protocol.get("seed"), int)
    ):
        raise SystemExit(
            f"{where}: ft row {ft['row_id']} records no recipe or no protocol seed, so "
            "nothing in it can be checked against this export"
        )
    recipe = ft["recipe"]
    problems = [
        f"{k}: row says {a!r}, here {b!r}"
        for k, (a, b) in _ft_row_mismatches(
            ft, seed=seed, trained_on=METAL_DEVICE, reader=reader,
            real_backbone=args.real_backbone,
        ).items()
    ]
    problems.extend(_metal_row_problems(ft))
    if not problems:
        steps = int(ft["metrics"]["train.optimizer_steps"]["value"])
        if body["ft_row_id"] != ft["row_id"]:
            problems.append(
                f"the manifest's ft_row_id is {body['ft_row_id']!r}: it is not this row's export"
            )
        if body["seed"] != seed:
            problems.append(f"the manifest says seed {body['seed']} and the name seed {seed}")
        if body["optimizer_step"] != steps:
            problems.append(
                f"the export is at optimizer step {body['optimizer_step']} and the row ended "
                f"at optimizer step {steps}"
            )
        if float(body["span_weight"]) != float(recipe["span_weight"]):
            problems.append(
                f"the export says span_weight {body['span_weight']} and the row trained at "
                f"{recipe['span_weight']}"
            )
    if problems:
        raise SystemExit(
            f"{where} and ft row {ft['row_id']} do not describe one model scored against "
            "this shard set and backbone: " + "; ".join(problems)
        )
    try:
        weights = read_average(manifest)
    except AverageRefusal as exc:
        raise SystemExit(f"{where}: {exc}") from exc
    meta = {
        "optimizer_step": int(body["optimizer_step"]),
        "seed": seed,
        "scored_checkpoint": f"{path.name}:{body['safetensors_sha256']}",
        TRAINED_BY_KEY: {
            "trainer": METAL_TRAINER, "device": METAL_DEVICE, "source": METAL_EXPORT_SOURCE,
            "manifest_sha256": manifest.sha256,
        },
    }
    return ft, recipe, seed, meta, weights


def _one_configuration(rows: Sequence[Mapping[str, Any]], *, what: str) -> int:
    """Refuse unless ``rows`` are completed, NOT quick, and one configuration at one step;
    return that step. The check an average and an ensemble both make of their ft rows.

    One configuration: one protocol minus the seed (``recipe_hash``, ``data_snapshot_hash``,
    ``tokenizer_hash``, ``backbone_commit``), one stored recipe, one
    ``train.optimizer_steps``.
    """
    quick = [str(r["row_id"]) for r in rows if r.get("quick") is not False]
    if quick:
        raise SystemExit(
            f"ft row(s) {quick} of {what} are quick (or do not say they are not): rule 8 "
            "excludes a quick run from every decision, and a model made of one is no less quick"
        )
    ids = [str(r["row_id"]) for r in rows]
    disagree: list[str] = []
    for key in ("recipe_hash", "data_snapshot_hash", "tokenizer_hash", "backbone_commit"):
        values = [r["protocol"].get(key) for r in rows]
        if len(set(values)) != 1:
            disagree.append(f"protocol.{key} {dict(zip(ids, values, strict=True))}")
    steps_by_row = [r["metrics"].get("train.optimizer_steps", {}).get("value") for r in rows]
    if len(set(steps_by_row)) != 1:
        disagree.append(f"train.optimizer_steps {dict(zip(ids, steps_by_row, strict=True))}")
    if any(r["recipe"] != rows[0]["recipe"] for r in rows[1:]):
        disagree.append("recipe (the stored recipe dicts differ)")
    if disagree:
        raise SystemExit(
            f"the ft rows of {what} are not one configuration at one step: "
            + "; ".join(disagree)
        )
    return int(steps_by_row[0])


def _ensemble_inputs(
    args: argparse.Namespace, *, reader: ShardReader, suite_seed: int
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any], list[Path]]:
    """A logit ensemble's ``(first ft row, recipe, protocol seed, meta, tower paths)``, every
    pairing checked and no tensor read.

    * two or more per-seed epoch checkpoints, no average among them, no file or seed twice;
    * ``--ft-row-id`` names one ft row per checkpoint, in the same order, each in
      ``--ft-ledger``, completed and **not quick**, no row twice;
    * ``--seeds`` is the checkpoints' seeds, in that order;
    * the rows are one configuration at one step (:func:`_one_configuration`);
    * row ``i`` describes checkpoint ``i`` as a single seed's score requires
      (:func:`_ft_row_mismatches`): its arm, device, seed, shard set and backbone;
    * each checkpoint's JSON body verifies (``payload_digest``), is at the rows' step and at
      its own seed, and declares a sidecar -- whose digest the row records per tower.

    ``meta['ensemble']`` is what the eval row's recipe records about the ensemble.
    """
    from qd_train.run_control import Checkpoint

    paths = list(score_paths(args.score_checkpoint))
    ids = list(ft_row_ids(args.ft_row_id))
    if len(paths) < 2:
        raise SystemExit(
            f"a logit ensemble is two or more checkpoints, got {len(paths)}: one checkpoint is "
            "scored as itself"
        )
    if any(p.suffix == AVERAGED_SUFFIX for p in paths):
        raise SystemExit(
            "a logit ensemble's towers are per-seed epoch checkpoints (.json); an average "
            "(.safetensors) is scored on its own"
        )
    if len({p.resolve() for p in paths}) != len(paths):
        raise SystemExit("an ensemble names one checkpoint twice; it would count one seed twice")
    if len(ids) != len(paths):
        raise SystemExit(
            f"--ft-row-id names {len(ids)} ft row(s) for {len(paths)} checkpoints: an ensemble "
            "pairs each checkpoint with its own ft row, in order"
        )
    arms: list[tuple[int, str]] = []
    for path in paths:
        try:
            tag, seed, trained_on = _resume_arm(path)
        except ValueError as exc:
            raise SystemExit(f"--score-checkpoint {exc}") from exc
        if tag != "epoch":
            raise SystemExit(f"{path.name}: only an epoch checkpoint is scored")
        arms.append((seed, trained_on))
    seeds = [seed for seed, _ in arms]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(
            f"the ensemble's checkpoints are seeds {seeds}: one seed twice is not "
            f"{len(seeds)} seeds"
        )
    if [int(s) for s in args.seeds] != seeds:
        raise SystemExit(
            f"the ensemble's checkpoints are seeds {seeds}, in that order; --seeds says "
            f"{list(args.seeds)}"
        )
    rows: list[dict[str, Any]] = []
    for path, row_id in zip(paths, ids, strict=True):
        try:
            rows.append(_ft_row(args.ft_ledger, row_id))
        except SystemExit as exc:
            raise SystemExit(
                f"the ensemble's tower {path.name} names ft row {row_id}: {exc}"
            ) from exc
    full_ids = [str(r["row_id"]) for r in rows]
    if len(set(full_ids)) != len(full_ids):
        raise SystemExit(f"the ensemble names ft rows {full_ids}: one row twice")
    steps = _one_configuration(rows, what=f"the {len(rows)}-tower ensemble")
    problems: list[str] = []
    for path, row, (seed, trained_on) in zip(paths, rows, arms, strict=True):
        wrong = _ft_row_mismatches(
            row, seed=seed, trained_on=trained_on, reader=reader,
            real_backbone=args.real_backbone,
        )
        if wrong:
            problems.append(
                f"ft row {row['row_id']} and {path.name}: "
                + "; ".join(f"{k}: row says {a!r}, here {b!r}" for k, (a, b) in wrong.items())
            )
    if problems:
        raise SystemExit(
            "the ensemble's ft rows do not describe its checkpoints scored against this shard "
            "set and backbone: " + " | ".join(problems)
        )
    towers: list[dict[str, object]] = []
    for path, (seed, _) in zip(paths, arms, strict=True):
        try:
            body = Checkpoint.read_body(path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"the ensemble's tower {path.name}: {exc}") from exc
        sidecar = body.get("sidecar")
        if not isinstance(sidecar, Mapping) or not isinstance(sidecar.get("digest"), str):
            raise SystemExit(f"{path.name} declares no tensor sidecar, so it carries no tower")
        if int(body["optimizer_step"]) != steps or int(body["seed"]) != seed:
            raise SystemExit(
                f"{path.name} is at optimizer step {body['optimizer_step']}, seed "
                f"{body['seed']}; its ft row ended at step {steps}, seed {seed}. This is not "
                "the model that row trained."
            )
        towers.append({
            "checkpoint": path.name, "seed": seed,
            "sidecar_digest": str(sidecar["digest"]),
            "payload_digest": str(body["payload_digest"]),
        })
    recipe = rows[0]["recipe"]
    meta = {
        "optimizer_step": steps,
        "seed": suite_seed,
        "scored_checkpoint": ",".join(f"{t['checkpoint']}:{t['sidecar_digest']}" for t in towers),
        "terminations": [r["metrics"].get("train.termination", {}).get("value") for r in rows],
        "ensemble": {
            "seeds": seeds, "ft_row_ids": full_ids, "n_towers": len(towers),
            "towers": towers, "combine": ENSEMBLE_COMBINE,
            # As an average's: the eval row's protocol seed and every suite's build seed.
            "protocol_seed": suite_seed, "suite_seed": suite_seed,
        },
    }
    return rows[0], recipe, suite_seed, meta, paths


def _averaged_weights(
    args: argparse.Namespace, *, reader: ShardReader, suite_seed: int
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    """An average's ``(first ft row, recipe, protocol seed, meta, weights)``, every pairing
    checked before the weights are read.

    * the manifest is one ``tools/ckpt_average.py`` wrote, naming one ft row per input;
    * ``--seeds`` lists exactly the average's seeds;
    * every row it names is in ``--ft-ledger``, a completed ft row, and **not quick**;
    * the rows are one configuration at one step: one protocol minus the seed
      (``recipe_hash``, ``data_snapshot_hash``, ``tokenizer_hash``, ``backbone_commit``), one
      recipe, one ``train.optimizer_steps``, and it is the manifest's ``optimizer_step``;
    * row ``i`` describes input ``i``: an epoch arm, its seed, the device in the input's
      file name, this run's shard set and this run's backbone snapshot;
    * each input's JSON body on disk hashes to the ``payload_digest`` the manifest recorded,
      and a body that is not on disk is refused (``ckpt_average.verify_sources``);
    * the ``.safetensors`` hashes to the manifest's ``safetensors_sha256``.

    ``meta['averaged']`` is what the eval row's recipe records about the average.
    """
    path = args.score_checkpoint
    try:
        manifest = read_manifest(path)
    except AverageRefusal as exc:
        raise SystemExit(f"--score-checkpoint {path.name}: {exc}") from exc
    seeds = manifest.seeds
    if sorted(int(s) for s in args.seeds) != sorted(seeds):
        raise SystemExit(
            f"{path.name} averages seeds {seeds}; --seeds says {list(args.seeds)}. Pass the "
            "average's own seeds."
        )
    rows: list[dict[str, Any]] = []
    for row_id in manifest.ft_row_ids:
        try:
            rows.append(_ft_row(args.ft_ledger, row_id))
        except SystemExit as exc:
            raise SystemExit(f"{path.name}'s manifest names ft row {row_id}: {exc}") from exc
    ids = [str(r["row_id"]) for r in rows]
    steps = _one_configuration(rows, what=path.name)
    if manifest.optimizer_step != steps:
        raise SystemExit(
            f"{path.name} averages checkpoints at optimizer step {manifest.optimizer_step}; "
            f"its ft rows ended at step {steps}. These are not the models those rows trained."
        )
    recipe = rows[0]["recipe"]
    problems: list[str] = []
    for row, record in zip(rows, manifest.inputs, strict=True):
        try:
            tag, seed, trained_on = _resume_arm(Path(record["path"]))
        except ValueError as exc:
            raise SystemExit(f"{path.name}'s input {record['path']}: {exc}") from exc
        expected = {
            "input arm": (tag, "epoch"),
            "recipe tag": (row["recipe"].get("tag"), "epoch"),
            "recipe device": (row["recipe"].get("device"), trained_on),
            "protocol seed": (row["protocol"]["seed"], record["seed"]),
            "input file seed": (seed, record["seed"]),
            "shard_hash": (row["recipe"].get("shard_hash"), reader.header.shard_hash()),
            "backbone_snapshot": (row["recipe"].get("backbone_snapshot"), args.real_backbone.name),
        }
        wrong = {k: v for k, v in expected.items() if v[0] != v[1]}
        if wrong:
            problems.append(
                f"ft row {row['row_id']} and input {Path(record['path']).name}: "
                + "; ".join(f"{k}: row says {a!r}, here {b!r}" for k, (a, b) in wrong.items())
            )
    if problems:
        raise SystemExit(
            f"the ft rows {path.name}'s manifest names do not describe its inputs scored "
            "against this shard set and backbone: " + " | ".join(problems)
        )
    norm = manifest.norm_preserving
    if norm is not None and norm["base_snapshot"] != args.real_backbone.name:
        raise SystemExit(
            f"{path.name} is a norm-preserving average that moved from the base "
            f"{norm['base_snapshot']!r}, and it is scored on the backbone "
            f"{args.real_backbone.name!r}: base + lambda * delta means nothing on another base"
        )
    try:
        verify_sources(manifest)
        weights = read_average(manifest)
    except AverageRefusal as exc:
        raise SystemExit(f"--score-checkpoint {path.name}: {exc}") from exc
    if float(weights["span_weight"]) != float(recipe["span_weight"]):
        raise SystemExit(
            f"{path.name} was averaged at span_weight {weights['span_weight']} and its ft "
            f"rows trained at {recipe['span_weight']}"
        )
    meta = {
        "optimizer_step": steps,
        "seed": suite_seed,
        "scored_checkpoint": f"{path.name}:{manifest.body['safetensors_sha256']}",
        "terminations": [
            r["metrics"].get("train.termination", {}).get("value") for r in rows
        ],
        "averaged": {
            "seeds": seeds, "ft_row_ids": ids, "manifest_sha256": manifest.sha256,
            "source": manifest.source, "n_inputs": len(rows),
            # The eval row's protocol seed, and the seed every suite this run scores (val's
            # permuted second pass, needle, OOD) was built at: config.seed, as for every
            # per-seed score row -- prepare_second_pass, prepare_needle and prepare_ood take
            # it, never the run seed.
            "protocol_seed": suite_seed, "suite_seed": suite_seed,
            # Only on a norm-preserving average (tag avg-np): the numbers its manifest was
            # made with, so a plain average's recipe is what it always was.
            **({} if norm is None else {"norm_preserving": {
                k: norm[k] for k in (
                    "formula", "c", "pairwise_cosine", "lambda_from_c", "lambda",
                    "lambda_derived_from_c", "base_snapshot",
                )
            }}),
        },
    }
    return rows[0], recipe, suite_seed, meta, weights


def _score_checkpoint(
    args: argparse.Namespace, *, reader: ShardReader, val: ValSet, device: str,
    ledger: Ledger, reasons_for: Callable[..., list[str]], second_pass: SecondPass,
    needle_suite: NeedleSuite, ood_suite: OodSuite, suite_seed: int,
    needle_decoded: NeedleDecoded | None = None, loaded: LoadedModel | None = None,
    plan_note: str | None = None,
) -> tuple[str, dict[str, object], TriState, list[SuiteGate], int]:
    """Score a saved epoch checkpoint, or an average of several, on the val set.

    Returns ``(eval row id, scored, permutation_consistency, suite gates, row seed)``.

    ``loaded`` is the model :func:`_checkpoint_step` already loaded for these ``args`` (once
    per ``--score-plan`` kind, for all its passes); without it this loads it.
    ``plan_note`` is the plan's sentence for the row's notes -- never its recipe, so a
    plan's row hashes as the same command's single run does.

    Every pairing that could silently score the wrong weights under the wrong row is checked
    before the tower loads: the file's arm, seed and training device against the ft row; the
    ft row's shard set and backbone against this run's; the checkpoint's optimizer step and
    seed against the ft row's (for an average, :func:`_averaged_weights`). Only the tower and
    span head are read, cast to ``--score-dtype``. From there one code path scores both: the
    val decode, the permuted second pass, the needle and OOD suites, and :func:`_record_score`.

    ``suite_seed`` is ``config.seed``, the seed the suites were built at. An average's eval
    row takes it as its protocol seed and records it.
    """
    for suite in (needle_suite, ood_suite):
        if suite.not_run is None and suite.seed != suite_seed:
            raise SystemExit(
                f"a suite was built at seed {suite.seed} and this run's suite seed is "
                f"{suite_seed}; the row would record a seed its suites were not built at"
            )
    if loaded is None:
        loaded = _checkpoint_step(
            args, reader=reader, val=val, device=device,
            eval_widths=suite_widths(needle_suite, ood_suite), suite_seed=suite_seed,
        )
    step, ft, recipe, seed, meta = loaded
    model = scored_model(args, ft, meta)
    run: dict[str, object] = {
        "tag": model.tag, "device": device, "seed": seed,
        "cost": _cost(
            device=device, n_gpus=n_gpus_for_device(device), usd_per_hour=args.usd_per_hour,
            usd_per_gpu_hour=args.usd_per_gpu_hour, instance=args.instance,
            cap_s=args.wall_clock_cap_s,
        ),
        **{k: recipe[k] for k in (*BACKBONE_KEYS, *RECIPE_PIECE_KEYS) if k in recipe},
        "score_dtype": args.score_dtype,
        "scored_checkpoint": model.scored_checkpoint,
        **model.recipe_block(),
        # Read by _record_score for the notes and the ft row metric; not a recipe key.
        "scored_model": model,
    }
    if model.block_key is None:
        run["ft_row_id"] = ft["row_id"]
    if plan_note is not None:
        run["plan_note"] = plan_note  # read by _record_score for the notes; not a recipe key
    decode_at = time.monotonic()
    scored = _decode(step, val.plan, val.labels_for, val.letter_id)
    permutation, val_second = score_permutation_consistency(step, val, second_pass, scored)
    gates = [
        needle_gate(
            score_needle(step, needle_suite, val.letter_id,
                         decoded=needle_decoded, logits=args.suite_logits),
            needle_suite,
        ),
        ood_suite_gate(
            score_ood(step, ood_suite, scored=scored, val_second=val_second,
                      val_second_pass=second_pass),
            ood_suite,
        ),
    ]
    decode_s = time.monotonic() - decode_at
    row_id = _record_score(
        run, scored, ledger=ledger, reader=reader, val=val,
        quick_reasons=model_reasons(model, device=device, reasons_for=reasons_for),
        decode_s=decode_s, permutation=permutation, suite_gates=gates,
    )
    return row_id, scored, permutation, gates, seed


def model_reasons(
    model: ScoredModel, *, device: str, reasons_for: Callable[..., list[str]]
) -> list[str]:
    """Every rule-8 reason for a row scoring ``model``: each input's termination (one
    reason shared by three rows is one reason), and a seed shortfall."""
    reasons: list[str] = []
    for termination in model.terminations:
        for reason in reasons_for(
            "epoch", device, None if termination is None else str(termination)
        ):
            if reason not in reasons:
                reasons.append(reason)
    reasons.extend(model.seed_reasons())
    reasons.extend(model.provenance_reasons())
    return reasons


# --- the report-only composed slice (a --score-plan 'composed' pass) ------------------------
#
# Fable G5(ii)'s slice ruling, conditions 6 and 8 (qd_train.composed_slice): 1,550 composed
# long-context rows, written refuse-gold into a report-only val set of their own. Never a gate
# and no gate's population; every number is a diagnostic beside its n. The set, its build row
# and its measured exclusions are pinned together, and a rebuilt slice re-pins all three
# before it is scored.

#: The slice's build row (longctx lane, 2026-10-01) and the ledger that holds it.
COMPOSED_SLICE_LEDGER: Final[Path] = REPO / "ledger" / "mac-slice-v4-2026-10-01.jsonl"
COMPOSED_SLICE_ROW: Final[str] = "dafe86af-ac2a-42ca-a3cd-d2b624740783"
#: The slice set's ``shard_hash``, which covers ``report_only`` and ``span_collapse_policy``.
COMPOSED_SLICE_SHARD_HASH: Final[str] = (
    "042d9b5034f9f1b2a4213be1601ed881d16adf9422fc14eefbd812aad28eac9d"
)
#: The slice corpus: its manifest is committed, its examples are not (they ship beside the
#: checkout, like every other corpus under data/pool).
COMPOSED_SLICE_CORPUS: Final[Path] = REPO / "data" / "pool" / "commitpackft-composed-slice-v1"
#: The ``gate`` on the slice's suite-verdict lines. No gate is called this, so qd-gate-report
#: refuses such a line rather than counting it toward anything.
COMPOSED_SLICE_GATE: Final[str] = "composed_slice.report_only"
#: Rule 8 for a slice row, whatever it scores.
COMPOSED_SLICE_QUICK_REASON: Final[str] = (
    "the report-only composed long-context slice (Fable G5(ii)): a diagnostic, never a gate "
    "and no gate's population, so it cannot promote anything"
)
#: The part name a kind's slice lines are written under (:func:`plan_output`), apart from its
#: gate lines, which qd-gate-report reads and would refuse beside these.
COMPOSED_SLICE_PART: Final[str] = "composed"


@dataclasses.dataclass(frozen=True)
class ComposedSlice:
    """The slice, opened and checked before any tower loads.

    Its reader, labels by batch and letter ids; its corpus rows; each span sequence's
    candidates (one per rendered line) and gold head row from the shard's own supervision;
    each choice sequence's length; and how every slot and row that was not written is
    counted. ``recipe`` is what a scored row records about it.
    """

    reader: ShardReader
    plan: list[Batch]
    labels_for: dict[int, list[Label]]
    letter_id: dict[str, int]
    cases: dict[str, cslice.ComposedCase]
    span: dict[str, cslice.SpanSequence]
    choice_length: dict[str, int]
    slot_excluded: dict[tuple[str, str], str]
    row_exclusions: tuple[tuple[cslice.ComposedCase, str], ...]
    recipe: dict[str, object]


def composed_slice_build(ledger_path: Path, row_id: str, *, shard_hash: str) -> LedgerRow:
    """The slice's build row, refused unless it is there, once, and names ``shard_hash``."""
    found = [r for r in Ledger(ledger_path).rows() if r.row_id == row_id]
    if len(found) != 1:
        raise SystemExit(
            f"the composed slice's build row {row_id} is in {ledger_path} {len(found)} times, "
            "not once"
        )
    row = found[0]
    header = row.metrics.get("report_only_slice_header")
    reached = row.metrics.get("report_only_slice_rows")
    recipe = row.recipe or {}
    if not isinstance(header, Ran) or header.value != shard_hash:
        raise SystemExit(
            f"build row {row_id} does not name the slice's shard hash {shard_hash[:16]} "
            f"(report_only_slice_header: {header})"
        )
    if not isinstance(reached, Ran) or reached.n is None or reached.n_total is None:
        raise SystemExit(f"build row {row_id} records no report_only_slice_rows n and n_total")
    missing = [k for k in ("rev", "defect_max_rows", "report_only_slice_manifest_sha256")
               if k not in recipe]
    if recipe.get("report_only") is not True or missing:
        raise SystemExit(
            f"build row {row_id} is not a report-only slice build (report_only "
            f"{recipe.get('report_only')!r}, recipe lacks {missing})"
        )
    return row


def open_composed_slice(
    slice_out: Path, *, corpus_dir: Path, download_root: Path, repo_root: Path,
    config: DataConfig, train: ShardReader, letter_id: Mapping[str, int],
    build_ledger: Path = COMPOSED_SLICE_LEDGER, build_row: str = COMPOSED_SLICE_ROW,
    shard_hash: str = COMPOSED_SLICE_SHARD_HASH,
    measured: Mapping[tuple[str, str], str] = cslice.MEASURED_EXCLUSIONS,
) -> ComposedSlice:
    """Open ``<slice_out>/shards/val-report-only-composed`` and refuse every way it could
    misscore, before a tower loads.

    Refused unless: the build row names this set's shard hash and the header says
    ``report_only`` and refuse-gold (the opposite of the gate readers' check); the set was
    written at the build's revision under the train set's remap; the corpus is the build's
    (manifest sha256), read by the pipeline's own slice loader; every label pairs with the
    writer's sequence index and ``supervision.npz``; every letter has an id; the exclusions
    are the measured list; ``dropped_before_write`` agrees with the build's
    ``report_only_slice_rows``; and every span sequence's candidates line up with its corpus
    row (one per rendered line, the shard's gold token on the gold start line).
    """
    import real_tokenizer_pipeline as pipeline

    from qd_data.mixture import rewrite_defect_class

    build = composed_slice_build(build_ledger, build_row, shard_hash=shard_hash)
    recipe = dict(build.recipe or {})
    set_dir = Path(slice_out) / "shards" / pipeline.REPORT_ONLY_VAL_DIR
    if not (set_dir / HEADER_NAME).exists():
        raise SystemExit(f"--composed-slice found no slice set at {set_dir}")
    try:
        reader = ShardReader(
            set_dir, config=config, repo_root=Path(slice_out), expect_rev=str(recipe["rev"])
        )
        cslice.require_report_only_slice(reader.header, where=str(set_dir))
    except (ShardContractViolation, cslice.SliceRefusal) as exc:
        raise SystemExit(f"--composed-slice: {exc}") from exc
    if reader.header.shard_hash() != shard_hash:
        raise SystemExit(
            f"{set_dir}: shard_hash {reader.header.shard_hash()[:16]}, not the pinned slice's "
            f"{shard_hash[:16]}"
        )
    if reader.header.remap_hash != train.header.remap_hash:
        raise SystemExit(
            f"the slice's remap {reader.header.remap_hash[:16]} is not the train set's "
            f"{train.header.remap_hash[:16]}: the same id would name a different embedding row"
        )
    loaded = pipeline.load_report_only_slice(
        Path(corpus_dir), download_root=Path(download_root), config=config,
        repo_root=Path(repo_root), base_max_rows=int(recipe["defect_max_rows"]),
    )
    if loaded.manifest_sha256 != recipe["report_only_slice_manifest_sha256"]:
        raise SystemExit(
            f"{corpus_dir}: manifest sha256 {loaded.manifest_sha256[:16]}, the build read "
            f"{str(recipe['report_only_slice_manifest_sha256'])[:16]}"
        )
    rows = [
        rewrite_defect_class(r, family_id=DEFECT_FAMILY_ID, index=i, config=config)
        for i, r in enumerate(loaded.rows)
    ]
    labels, excluded = _labels(rows, config=config)
    labels = pair_labels(reader, labels, require_index=True)
    _inventory(reader, labels, excluded)
    merged = merge_letter_ids(dict(letter_id), _letter_ids(reader, labels))
    unknown = sorted(
        {x for label in labels if label.slot_kind != SLOT_SPAN for x in label.letters}
        - set(merged)
    )
    if unknown:
        raise SystemExit(
            f"slice rows offer letter(s) {unknown} with no token id: those slots would be "
            "neither decoded nor excluded"
        )
    index = reader.sequence_index
    if index is None:  # pragma: no cover - pair_labels(require_index=True) refused it
        raise SystemExit(f"{set_dir} carries no sequence index")
    try:
        cases = cslice.load_cases([Path(corpus_dir) / "examples.jsonl"])
        loaded_ids = {f"qdm:{DEFECT_FAMILY_ID}:{r.example_id}" for r in loaded.rows}
        if set(cases) != loaded_ids:
            raise cslice.SliceRefusal(
                f"the slice's cases ({len(cases)}) and the loader's rows ({len(loaded_ids)}) "
                "are not one set of ids"
            )
        found: dict[tuple[str, str], str] = {}
        for e in index.excluded:
            key = (e.row_id, e.slot_name)
            if key in found:
                raise cslice.SliceRefusal(f"{key} is excluded twice")
            found[key] = cslice.exclusion_bucket(e.scope, e.refusal, e.detail)
        cslice.check_measured_exclusions(found, measured=measured)
        dropped = cslice.dropped_before_write(
            cases, [rid for rid, _ in index.sequences] + [e.row_id for e in index.excluded]
        )
        reached = build.metrics["report_only_slice_rows"]
        assert isinstance(reached, Ran) and reached.n is not None and reached.n_total is not None
        if len(cases) != reached.n_total or len(dropped) != reached.n_total - reached.n:
            raise cslice.SliceRefusal(
                f"{len(cases)} slice rows and {len(dropped)} dropped before write, against "
                f"build row {build_row}'s report_only_slice_rows {reached.n} of "
                f"{reached.n_total} (n_total - n = {reached.n_total - reached.n} dropped)"
            )
        batch_tokens = int(max(reader.header.buckets))
        plan = val_plan(reader, config=config)
        labels_for = _labels_by_batch(
            reader, plan, labels, config=config, batch_tokens=batch_tokens
        )
        members = reader._plan(batch_tokens=batch_tokens, seed=config.seed, epoch=0)
        lengths = reader.lengths()
        span: dict[str, cslice.SpanSequence] = {}
        choice_length: dict[str, int] = {}
        for p, batch in zip(members, plan, strict=True):
            sup = ft_supervision(batch)
            heads = None if sup.span is None else plan_span_batch(sup.span)
            span_k = {} if sup.span is None else {
                int(r): k for k, r in enumerate(sup.span.rows)
            }
            for r, i in enumerate(p.rows):
                label = labels[i]
                if label.row_id not in cases:
                    raise cslice.SliceRefusal(f"sequence {i} is {label.row_id}, not a slice row")
                into = span if label.slot_kind == SLOT_SPAN else choice_length
                if label.row_id in into:
                    raise cslice.SliceRefusal(f"{label.row_id} has two {label.slot_name} sequences")
                if label.slot_kind == SLOT_SPAN:
                    if heads is None or r not in span_k:  # pragma: no cover - Batch's contract
                        raise cslice.SliceRefusal(f"span sequence {i} has no span supervision")
                    seq = cslice.SpanSequence(
                        candidates=tuple(int(x) for x in reader.candidates(i)),
                        gold_head_row=int(heads.gold_start[span_k[r]]),
                        length_tokens=int(lengths[i]),
                    )
                    cslice.check_alignment(cases[label.row_id], seq.candidates, seq.gold_head_row)
                    span[label.row_id] = seq
                elif label.slot_name == CHOICE_SLOT:
                    choice_length[label.row_id] = int(lengths[i])
                else:
                    raise cslice.SliceRefusal(
                        f"sequence {i} is slot {label.slot_name!r}, neither {SPAN_SLOT!r} nor "
                        f"{CHOICE_SLOT!r}"
                    )
    except cslice.SliceRefusal as exc:
        raise SystemExit(f"--composed-slice: {exc}") from exc
    shared = sum(1 for s in span.values() if len(set(s.candidates)) != len(s.candidates))
    print(
        f"composed slice: {len(cases)} rows ({loaded.by_half}), {len(reader)} sequences in "
        f"{len(plan)} batches; {len(span)} span sequences, {shared} with a shared line-start "
        f"token (refuse-any keeps {len(span) - shared}); excluded {sorted(found.items())}; "
        f"{len(dropped)} dropped before write; shard_hash {shard_hash[:16]}, build row "
        f"{build_row[:8]}"
    )
    return ComposedSlice(
        reader=reader, plan=plan, labels_for=labels_for, letter_id=merged, cases=cases,
        span=span, choice_length=choice_length, slot_excluded=found,
        row_exclusions=tuple(dropped),
        recipe={
            "build_row": build_row, "build_ledger": Path(build_ledger).name,
            "shard_hash": shard_hash, "manifest_sha256": loaded.manifest_sha256,
            "examples_sha256": loaded.examples_sha256, "rows": len(cases),
            "sequences": len(reader),
            "measured_exclusions": [[r, s, b] for (r, s), b in sorted(found.items())],
            "dropped_before_write": len(dropped),
            "populations": list(cslice.POPULATIONS),
            "hit_rule": (
                "a prediction on a token several lines share is a needle-hunk hit only if "
                "every line on it is in the gold hunk"
            ),
        },
    )


def composed_slice_verdicts(
    composed: ComposedSlice, scored_verdicts: Sequence[Mapping[str, object]], *, logits: bool,
) -> tuple[list[cslice.SliceVerdict], list[dict[str, object]]]:
    """The scorer's per-sequence verdicts as one :class:`cslice.SliceVerdict` per slice row
    that was not dropped, and each row's suite-verdict line.

    Every written slot has exactly one verdict and every excluded slot none; a verdict for
    anything else is refused. A line carries the choice verdict's letter rows, and under
    ``logits`` (``--suite-logits``) the span pointer's ``start_logits``/``end_logits`` -- a
    span verdict without them is refused, as :func:`score_needle` refuses one.
    """
    by_slot: dict[tuple[str, str], Mapping[str, object]] = {}
    for v in scored_verdicts:
        key = (str(v["row_id"]), str(v["slot_name"]))
        if key in by_slot:
            raise SystemExit(f"--composed-slice: {key} was decoded twice")
        by_slot[key] = v
    dropped = {c.row_id for c, _ in composed.row_exclusions}
    stray = sorted(k for k in by_slot if k[0] not in composed.cases or k[0] in dropped)
    if stray:
        raise SystemExit(f"--composed-slice: verdicts for no slice row: {stray[:3]}")
    verdicts: list[cslice.SliceVerdict] = []
    lines: list[dict[str, object]] = []
    try:
        for row_id, case in sorted(composed.cases.items()):
            if row_id in dropped:
                continue
            span_v = by_slot.get((row_id, SPAN_SLOT))
            choice_v = by_slot.get((row_id, CHOICE_SLOT))
            verdict = cslice.slice_verdict(
                case, span=composed.span.get(row_id), span_verdict=span_v,
                span_excluded=composed.slot_excluded.get((row_id, SPAN_SLOT)),
                choice_length=composed.choice_length.get(row_id), choice_verdict=choice_v,
                choice_excluded=composed.slot_excluded.get((row_id, CHOICE_SLOT)),
            )
            line = cslice.verdict_line(verdict)
            if choice_v is not None:
                line["choice_row_logits"] = choice_v.get("row_logits")
            if logits and span_v is not None:
                missing = [k for k in SUITE_LOGIT_KEYS if k not in span_v]
                if missing:
                    raise SystemExit(
                        f"--composed-slice: span verdict {row_id} lacks {', '.join(missing)} "
                        "under --suite-logits"
                    )
                line.update({k: span_v[k] for k in SUITE_LOGIT_KEYS})
            verdicts.append(verdict)
            lines.append(line)
    except cslice.SliceRefusal as exc:
        raise SystemExit(f"--composed-slice: {exc}") from exc
    return verdicts, lines


def run_composed_slice(
    args: argparse.Namespace, *, loaded: LoadedModel, reader: ShardReader,
    composed: ComposedSlice, device: str, ledger: Ledger, reasons_for: Callable[..., list[str]],
    plan_note: str | None = None,
) -> tuple[str, tuple[dict[str, object], ...], int]:
    """The slice on ``loaded``: one decode of every slice sequence, conditions 6 and 8's
    tables (:func:`cslice.slice_metrics`) on a quick row tagged ``<model tag>-composed-slice``,
    and its per-row lines. Returns ``(row id, lines, row seed)``.

    The decode is :func:`_decode`'s, the one every gate uses, at T = 1; a logit ensemble's
    mean log-probabilities go through it unchanged.
    """
    step, ft, recipe, seed, meta = loaded
    model = scored_model(args, ft, meta)
    decode_at = time.monotonic()
    scored = _decode(
        step, composed.plan, composed.labels_for, composed.letter_id,
        pointer_scores=args.suite_logits,
    )
    decode_s = time.monotonic() - decode_at
    if int(scored["rows_not_decoded"]):  # type: ignore[call-overload]
        raise SystemExit(
            f"--composed-slice: {scored['rows_not_decoded']} slice sequences were not decoded "
            f"(letters without an id: {scored.get('letters_without_id')})"
        )
    verdicts, lines = composed_slice_verdicts(
        composed, scored["verdicts"], logits=args.suite_logits,  # type: ignore[arg-type]
    )
    try:
        metrics = cslice.slice_metrics(
            verdicts, row_exclusions=composed.row_exclusions, corpus=composed.cases
        )
    except cslice.SliceRefusal as exc:
        raise SystemExit(f"--composed-slice: {exc}") from exc
    slice_recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py",
        "tag": f"{model.tag}-composed-slice",
        "device": device,
        **{k: recipe[k] for k in (*BACKBONE_KEYS, *RECIPE_PIECE_KEYS) if k in recipe},
        "score_dtype": args.score_dtype,
        "scored_checkpoint": model.scored_checkpoint,
        **model.recipe_block(),
        "shard_hash": reader.header.shard_hash(),
        "composed_slice": composed.recipe,
    }
    reasons = model_reasons(model, device=device, reasons_for=reasons_for)
    reasons.append(COMPOSED_SLICE_QUICK_REASON)
    notes = (
        f"tools/real_ft_run.py --score-plan composed-slice pass of {model.scored_checkpoint} "
        f"({model.described}), {args.score_dtype} on {device} at T = 1: the report-only "
        f"composed long-context slice (build row {composed.recipe['build_row']}, shard_hash "
        f"{str(composed.recipe['shard_hash'])[:16]}), {len(verdicts)} rows decoded the way "
        "crates/qd-runtime/src/answer.rs decodes them. Fable G5(ii) conditions 6 and 8: span "
        "top-1 and the needle-hunk hit under refuse_gold and refuse_any with their difference, "
        "choice top-1 under both_policies, a prediction on a shared token a hit only if every "
        f"line on it is in the gold hunk; exclusions counted, never misses. {REPORT_ONLY_NOTE}"
    )
    recorder = _recorder(
        ledger, reader=reader, seed=seed, recipe=slice_recipe, run_kind="eval",
        quick_reasons=reasons, wall_clock_s=decode_s,
        cost=_cost(
            device=device, n_gpus=n_gpus_for_device(device), usd_per_hour=args.usd_per_hour,
            usd_per_gpu_hour=args.usd_per_gpu_hour, instance=args.instance,
            cap_s=args.wall_clock_cap_s,
        ),
        notes=notes if plan_note is None else f"{notes} {plan_note}",
    )
    with recorder:
        recorder.metric(*model.ft_row_metric())
        for name, state in metrics.items():
            recorder.metric(name, state)
        recorder.noul_rate = NotRun(reason="a composed-slice row decodes no val row")
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id, tuple(lines), seed


#: What every slice row's notes end on.
REPORT_ONLY_NOTE: Final[str] = (
    "Report-only: not a gate, no gate's population, never a promotion input."
)


# --- several models in one invocation (--score-plan) ---------------------------------------
#
# Fable I diagnostic (2026-10-01): seed 0, 1 and 2, their average and their 3-seed ensemble
# on the OOD suite, and the ensemble's full gate row. Five --score-checkpoint runs would
# rebuild the same shard set, val set and suites five times (minutes each on the full
# corpus); a plan builds them once and runs one kind after another, each kind exactly what
# --score-checkpoint with its --ft-row-id and --seeds would score, checked by the same
# function (`_check_score_checkpoint_flags`) and paired by the same one (`_checkpoint_step`).

#: What a plan kind may run on its model. ``gates``: the val pass and every gate, the row
#: ``--score-checkpoint`` writes. ``ood``: the OOD suite alone, a quick diagnostic row
#: (:func:`run_ood_diagnostic`) -- never a gate, never a promotion input. ``composed``: the
#: report-only composed slice (:func:`run_composed_slice`), likewise a quick diagnostic row.
PLAN_PASSES: Final[tuple[str, ...]] = ("gates", "ood", "composed")
#: Kinds one plan may hold. Each loads a model; a bound on fan-out, not a judgement.
PLAN_MAX_KINDS: Final[int] = 8
#: Bytes a plan file may hold: a list of kinds, nothing more.
PLAN_MAX_BYTES: Final[int] = 64 * 1024
#: A kind's name: it lands in output file names and on every line, so it is plain.
PLAN_KIND_NAME: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
PLAN_KIND_REQUIRED: Final[frozenset[str]] = frozenset({"name", "checkpoints", "seeds", "passes"})
PLAN_KIND_KEYS: Final[frozenset[str]] = PLAN_KIND_REQUIRED | {"ft_row_ids"}


@dataclasses.dataclass(frozen=True, slots=True)
class PlanKind:
    """One model a plan scores, and which passes it runs on it."""

    name: str
    checkpoints: tuple[Path, ...]
    ft_row_ids: tuple[str, ...] | None
    seeds: tuple[int, ...]
    passes: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ScorePlan:
    """A ``--score-plan`` file, read and checked: its kinds in run order, and its digest."""

    path: Path
    sha256: str
    kinds: tuple[PlanKind, ...]

    def wants(self, name: str) -> bool:
        return any(name in kind.passes for kind in self.kinds)

    def note(self, kind: PlanKind) -> str:
        """The sentence a kind's rows carry in their notes (never their recipe)."""
        return (
            f"Scored as kind {kind.name!r} ({', '.join(kind.passes)}) of --score-plan "
            f"{self.path.name} (sha256 {self.sha256}): one invocation that built the shard "
            f"set, val set and suites once for its {len(self.kinds)} kind(s)."
        )


def _plan_strings(value: object, *, what: str, where: str) -> tuple[str, ...]:
    if not (isinstance(value, list) and 1 <= len(value) <= MAX_SEEDS
            and all(isinstance(x, str) and x for x in value)):
        raise SystemExit(f"{where}: {what} is a list of 1 to {MAX_SEEDS} non-empty strings")
    return tuple(value)


def read_score_plan(path: Path) -> ScorePlan:
    """Read and check a plan, refusing anything it does not define."""
    where = f"--score-plan {path}"
    try:
        size = path.stat().st_size
        if size > PLAN_MAX_BYTES:
            raise SystemExit(f"{where}: {size} bytes, more than a plan's {PLAN_MAX_BYTES}")
        raw = path.read_bytes()
        body = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{where}: {exc}") from exc
    if not isinstance(body, dict) or set(body) != {"kinds"}:
        raise SystemExit(f"{where}: a plan is an object with one key, 'kinds'")
    listed = body["kinds"]
    if not isinstance(listed, list) or not 1 <= len(listed) <= PLAN_MAX_KINDS:
        raise SystemExit(f"{where}: 'kinds' is a list of 1 to {PLAN_MAX_KINDS} kinds")
    kinds: list[PlanKind] = []
    for i, entry in enumerate(listed):
        at = f"{where} kind {i}"
        if not isinstance(entry, dict):
            raise SystemExit(f"{at}: a kind is an object")
        unknown = sorted(set(entry) - PLAN_KIND_KEYS)
        missing = sorted(PLAN_KIND_REQUIRED - set(entry))
        if unknown or missing:
            raise SystemExit(f"{at}: unknown keys {unknown}, missing keys {missing}")
        name = entry["name"]
        if not isinstance(name, str) or PLAN_KIND_NAME.fullmatch(name) is None:
            raise SystemExit(f"{at}: name {name!r} is not {PLAN_KIND_NAME.pattern}")
        at = f"{where} kind {name!r}"
        seeds = entry["seeds"]
        if not (isinstance(seeds, list) and 1 <= len(seeds) <= MAX_SEEDS
                and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)):
            raise SystemExit(f"{at}: seeds is a list of 1 to {MAX_SEEDS} integers")
        passes = entry["passes"]
        if not (isinstance(passes, list) and passes
                and all(isinstance(p, str) and p in PLAN_PASSES for p in passes)
                and len(set(passes)) == len(passes)):
            raise SystemExit(f"{at}: passes is a non-empty list of distinct {list(PLAN_PASSES)}")
        if "gates" in passes and "ood" in passes:
            raise SystemExit(
                f"{at}: 'gates' already decodes the OOD suite (the ood_abstain gate, with the "
                "in-distribution bound); an 'ood' pass beside it would decode it twice"
            )
        kinds.append(PlanKind(
            name=name,
            checkpoints=tuple(
                Path(p) for p in _plan_strings(entry["checkpoints"], what="checkpoints", where=at)
            ),
            ft_row_ids=(
                None if "ft_row_ids" not in entry
                else _plan_strings(entry["ft_row_ids"], what="ft_row_ids", where=at)
            ),
            seeds=tuple(seeds),
            passes=tuple(passes),
        ))
    names = [k.name for k in kinds]
    if len(set(names)) != len(names):
        raise SystemExit(f"{where}: kind names repeat: {names}")
    return ScorePlan(path=path, sha256=hashlib.sha256(raw).hexdigest(), kinds=tuple(kinds))


def plan_output(path: Path, kind: str, part: str | None = None) -> Path:
    """A kind's own output file beside ``path``: ``<stem>-<kind><suffix>``, or
    ``<stem>-<kind>.<part><suffix>`` for a part kept apart from it. A kind name holds no dot
    (:data:`PLAN_KIND_NAME`), so no kind's file is another kind's part."""
    middle = kind if part is None else f"{kind}.{part}"
    return path.with_name(f"{path.stem}-{middle}{path.suffix}")


def plan_kind_args(args: argparse.Namespace, kind: PlanKind) -> argparse.Namespace:
    """``args`` as ``--score-checkpoint`` with this kind's checkpoints, ft rows and seeds
    would have parsed them: one checkpoint a Path and one ft row id a str, several lists."""
    out = copy.copy(args)
    out.score_checkpoint = (
        kind.checkpoints[0] if len(kind.checkpoints) == 1 else list(kind.checkpoints)
    )
    out.ft_row_id = (
        None if kind.ft_row_ids is None
        else kind.ft_row_ids[0] if len(kind.ft_row_ids) == 1 else list(kind.ft_row_ids)
    )
    out.seeds = list(kind.seeds)
    return out


def _check_score_plan_flags(args: argparse.Namespace, *, seeds_given: bool) -> ScorePlan:
    """Read the plan and refuse, from argv alone, every way it could score nothing, score
    something twice, or write over an earlier result."""
    plan = read_score_plan(args.score_plan)
    clashing = [
        flag for flag, given in (
            ("--score-checkpoint", args.score_checkpoint is not None),
            ("--ft-row-id", args.ft_row_id is not None), ("--seeds", seeds_given),
            ("--needle-control", args.needle_control is not None),
            ("--needle-predictions-out", args.needle_predictions_out is not None),
            ("--epoch", args.epoch), ("--resume-from", args.resume_from is not None),
        ) if given
    ]
    if clashing:
        raise SystemExit(
            f"--score-plan names every kind's checkpoints, ft rows and seeds itself and trains "
            f"nothing; {', '.join(clashing)} would be ignored or contradict it"
        )
    if not args.devices or len(args.devices) != 1:
        raise SystemExit("--score-plan scores on one device: pass exactly one --devices entry")
    if args.needle and args.devices[0] == "mps":
        raise SystemExit(
            "--score-plan decodes the needle suite in this process, and on MPS the needle after "
            "a val pass ran out of memory (run_needle_worker): score the needle on MPS with "
            "--score-checkpoint, whose worker process returns that memory"
        )
    if args.needle and not plan.wants("gates"):
        raise SystemExit("--needle is a gate; no kind of this plan runs the 'gates' pass")
    if args.ood and not (plan.wants("gates") or plan.wants("ood")):
        raise SystemExit("--ood would build a suite no kind of this plan decodes")
    if plan.wants("ood") and not args.ood:
        raise SystemExit("an 'ood' pass decodes the OOD suite: it needs --ood")
    if args.verdicts_out is not None and not plan.wants("gates"):
        raise SystemExit("--verdicts-out writes val verdicts; no kind of this plan decodes val")
    if plan.wants("composed") != (args.composed_slice is not None):
        raise SystemExit(
            "a 'composed' pass scores the slice --composed-slice names, and --composed-slice is "
            "read only by one: give both or neither"
        )
    if plan.wants("composed") and args.devices[0] == "mps":
        raise SystemExit(
            "--score-plan decodes the composed slice (3,099 sequences up to 7,801 tokens) in "
            "this process, and on MPS the needle-length decode after a val pass ran out of "
            "memory (run_needle_worker): score the slice on cuda"
        )
    for kind in plan.kinds:
        try:
            _check_score_checkpoint_flags(plan_kind_args(args, kind))
        except SystemExit as exc:
            raise SystemExit(f"--score-plan kind {kind.name!r}: {exc}") from exc
        suite_parts = [None, *([COMPOSED_SLICE_PART] if "composed" in kind.passes else [])]
        for flag, out, part in (
            ("--verdicts-out", args.verdicts_out if "gates" in kind.passes else None, None),
            *(("--suite-verdicts-out", args.suite_verdicts_out, p) for p in suite_parts),
        ):
            if out is not None and plan_output(out, kind.name, part).exists():
                raise SystemExit(
                    f"{flag} of kind {kind.name!r}: {plan_output(out, kind.name, part)} "
                    "already exists"
                )
    return plan


#: The ``gate`` an OOD diagnostic row's suite-verdict lines carry: not ``ood_abstain``, so
#: nothing that reads gate lines (qd-gate-report) can count them toward that gate.
OOD_DIAGNOSTIC_GATE: Final[str] = "ood_abstain.diagnostic"
#: Rule 8 for an OOD-only diagnostic row, whatever it scores.
OOD_DIAGNOSTIC_QUICK_REASON: Final[str] = (
    "an OOD-suite-only diagnostic: no val pass, so no in-distribution bound and no ood_abstain "
    "gate; a diagnostic row is never a gate row"
)


def run_ood_diagnostic(
    args: argparse.Namespace, *, loaded: LoadedModel, reader: ShardReader, val: ValSet,
    device: str, ledger: Ledger, reasons_for: Callable[..., list[str]], ood_suite: OodSuite,
    suite_seed: int, plan_note: str | None = None,
) -> tuple[str, tuple[dict[str, object], ...], int]:
    """The OOD suite alone on ``loaded``: both passes, the runtime rule per case, a quick
    diagnostic row (tag ``<model tag>-ood-diagnostic``) and its per-case lines.

    Decoded by :func:`decode_ood` and counted by :func:`ood_category_metrics`, as the
    ``ood_abstain`` gate's report is, under ``ood_diagnostic.*``: no val pass runs, so there
    is no in-distribution bound and no gate. Returns ``(row id, lines, row seed)``.
    """
    if ood_suite.not_run is not None or ood_suite.seed != suite_seed:
        raise SystemExit(
            f"the OOD diagnostic needs the OOD suite built at the suite seed {suite_seed}: "
            f"{ood_suite.not_run or f'it was built at {ood_suite.seed}'}"
        )
    assert ood_suite.second_pass is not None  # set whenever the suite ran
    if ood_suite.second_pass.not_run is not None:
        raise SystemExit(
            f"the OOD diagnostic's second pass did not run: {ood_suite.second_pass.not_run}"
        )
    step, ft, recipe, seed, meta = loaded
    model = scored_model(args, ft, meta)
    decode_at = time.monotonic()
    first, second, ood = decode_ood(step, ood_suite)
    decode_s = time.monotonic() - decode_at
    what = "(an OOD-only diagnostic, no gate)"
    metrics = ood_category_metrics(ood_suite, ood, prefix="ood_diagnostic", what=what)
    abstained = sum(ood.values())
    metrics["ood_diagnostic.all"] = (
        Ran(passed=True, value=abstained / len(ood), n=abstained, n_total=len(ood),
            detail=f"abstained on {abstained} of {len(ood)} OOD cases {what}")
        if ood else NotRun(reason="no OOD case was decoded")
    )
    diagnostic_recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py",
        "tag": f"{model.tag}-ood-diagnostic",
        "device": device,
        **{k: recipe[k] for k in (*BACKBONE_KEYS, *RECIPE_PIECE_KEYS) if k in recipe},
        "score_dtype": args.score_dtype,
        "scored_checkpoint": model.scored_checkpoint,
        **model.recipe_block(),
        "shard_hash": reader.header.shard_hash(),
        "val_shard_hash": val.reader.header.shard_hash(),
        "ood": ood_recipe(ood_suite),
    }
    reasons = model_reasons(model, device=device, reasons_for=reasons_for)
    reasons.append(OOD_DIAGNOSTIC_QUICK_REASON)
    notes = (
        f"tools/real_ft_run.py --score-plan OOD diagnostic of {model.scored_checkpoint} "
        f"({model.described}), {args.score_dtype} on {device} at T = 1: the OOD suite's two "
        "passes decoded the way crates/qd-runtime/src/answer.rs decodes them, the runtime "
        "rule's abstentions counted per category, and each case's letter rows (noul "
        "included) kept on its suite-verdict line. No val pass and no gate."
    )
    recorder = _recorder(
        ledger, reader=reader, seed=seed, recipe=diagnostic_recipe, run_kind="eval",
        quick_reasons=reasons, wall_clock_s=decode_s,
        cost=_cost(
            device=device, n_gpus=n_gpus_for_device(device), usd_per_hour=args.usd_per_hour,
            usd_per_gpu_hour=args.usd_per_gpu_hour, instance=args.instance,
            cap_s=args.wall_clock_cap_s,
        ),
        notes=notes if plan_note is None else f"{notes} {plan_note}",
    )
    with recorder:
        recorder.metric(*model.ft_row_metric())
        for name, state in metrics.items():
            recorder.metric(name, state)
        recorder.noul_rate = NotRun(reason="an OOD diagnostic row decodes no val row")
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id, ood_verdict_lines(ood_suite, first, second, ood), seed


def run_score_plan(
    args: argparse.Namespace, plan: ScorePlan, *, reader: ShardReader, val: ValSet,
    device: str, ledger: Ledger, reasons_for: Callable[..., list[str]],
    second_pass: SecondPass, needle_suite: NeedleSuite, ood_suite: OodSuite, suite_seed: int,
    composed: ComposedSlice | None = None,
) -> list[tuple[str, str, str]]:
    """Every kind of ``plan``, in order, over the one set of suites main built: load the
    kind's model, run its passes on it, write its files, free it before the next loads.

    The needle suite, when asked for, is decoded in this process: the worker exists for
    MPS (:func:`run_needle_worker`), which :func:`_check_score_plan_flags` refuses here.
    ``composed`` is the slice a 'composed' pass scores (:func:`open_composed_slice`); its
    lines go to the kind's ``composed`` part of ``--suite-verdicts-out``, apart from its gate
    lines. Returns ``(kind, pass, row id)`` per row written.
    """
    if plan.wants("composed") and composed is None:
        raise SystemExit("a 'composed' pass needs the slice, and none was opened")
    if plan.wants("ood") and (
        ood_suite.not_run is not None or ood_suite.second_pass is None
        or ood_suite.second_pass.not_run is not None
    ):
        raise SystemExit(
            "an 'ood' pass needs the OOD suite and its second pass, and they did not build: "
            f"{ood_suite.not_run or (ood_suite.second_pass and ood_suite.second_pass.not_run)}"
        )
    widths = suite_widths(needle_suite, ood_suite, composed=composed)
    recorded: list[tuple[str, str, str]] = []
    for kind in plan.kinds:
        kind_args = plan_kind_args(args, kind)
        note = plan.note(kind)
        print(f"\nscore plan: kind {kind.name} ({', '.join(kind.passes)})", flush=True)
        loaded = _checkpoint_step(
            kind_args, reader=reader, val=val, device=device, eval_widths=widths,
            suite_seed=suite_seed,
        )
        verdict_lines: list[dict[str, object]] = []
        suite_lines: list[dict[str, object]] = []
        if "gates" in kind.passes:
            row_id, scored, _, gates, row_seed = _score_checkpoint(
                kind_args, reader=reader, val=val, device=device, ledger=ledger,
                reasons_for=reasons_for, second_pass=second_pass, needle_suite=needle_suite,
                ood_suite=ood_suite, suite_seed=suite_seed, loaded=loaded, plan_note=note,
            )
            verdict_lines = [
                {**line, "score_kind": kind.name}
                for line in _verdict_lines(scored, eval_row_id=row_id, seed=row_seed)
            ]
            suite_lines.extend(
                {**line, "score_kind": kind.name}
                for line in suite_verdict_lines(gates, eval_row_id=row_id, seed=row_seed)
            )
            recorded.append((kind.name, "gates", row_id))
            for g in gates:
                print(f"  {kind.name} {g.name}: {json.dumps(g.state.to_json())[:300]}")
        if "ood" in kind.passes:
            row_id, lines, row_seed = run_ood_diagnostic(
                kind_args, loaded=loaded, reader=reader, val=val, device=device, ledger=ledger,
                reasons_for=reasons_for, ood_suite=ood_suite, suite_seed=suite_seed,
                plan_note=note,
            )
            suite_lines.extend(
                {"eval_row_id": row_id, "seed": int(row_seed), "gate": OOD_DIAGNOSTIC_GATE,
                 "score_kind": kind.name, **v}
                for v in lines
            )
            recorded.append((kind.name, "ood", row_id))
        composed_lines: list[dict[str, object]] = []
        if "composed" in kind.passes:
            assert composed is not None  # refused above
            row_id, lines, row_seed = run_composed_slice(
                kind_args, loaded=loaded, reader=reader, composed=composed, device=device,
                ledger=ledger, reasons_for=reasons_for, plan_note=note,
            )
            composed_lines = [
                {"eval_row_id": row_id, "seed": int(row_seed), "gate": COMPOSED_SLICE_GATE,
                 "score_kind": kind.name, **v}
                for v in lines
            ]
            recorded.append((kind.name, "composed", row_id))
        # The model's last reference: the next kind loads into the memory this returns.
        del loaded
        gc.collect()
        release_device_cache()
        if args.verdicts_out is not None and "gates" in kind.passes:
            write_verdicts_jsonl(plan_output(args.verdicts_out, kind.name), verdict_lines)
        if args.suite_verdicts_out is not None:
            if "gates" in kind.passes or "ood" in kind.passes:
                write_suite_verdicts_jsonl(
                    plan_output(args.suite_verdicts_out, kind.name), suite_lines
                )
            if "composed" in kind.passes:
                write_suite_verdicts_jsonl(
                    plan_output(args.suite_verdicts_out, kind.name, COMPOSED_SLICE_PART),
                    composed_lines,
                )
        print(f"score plan: kind {kind.name} done -> "
              f"{[(p, r) for name, p, r in recorded if name == kind.name]}")
    return recorded


# --- the shuffled-label control (--shuffled-label) -----------------------------------------
#
# docs/hardening.md: a model trained on PERMUTED labels must fall to chance, and if it does
# not, the split leaks. `eval_harness.shuffled_label_control` is that check, used as it is.
# This arm trains the model it needs -- the target eval row's recipe on its own shard set,
# with the code.defect_class choice golds permuted -- and records the control on a row that
# SUPPLEMENTS the target eval row (`ledger.SUPPLEMENT_KEY`), so promotion reads the two as
# one unit.

#: The family whose split the control tests, and the only one whose golds it permutes or
#: scores. A phase-3 set is this family alone. In a phase-4 mixture the general families
#: have holdouts of their own, and permuting their letters too would move the letter
#: marginal the chance rate is computed from without testing anything about this split.
SHUFFLED_LABEL_FAMILY: Final[str] = DEFECT_FAMILY_ID

#: What a --shuffled-label ft row's recipe names. The same on every seed, so the control's
#: ft rows are one seed family of their own; the per-seed facts (the permutation's digest,
#: how many golds moved) go on the control row, beside the eval row id.
SHUFFLED_LABEL_RECIPE: Final[Mapping[str, object]] = {
    "family": SHUFFLED_LABEL_FAMILY,
    "permuted": "choice gold letters",
    "groups": "equal option count",
    "permutation_seed": "the run seed",
    "span_channel": "off",
}

#: The ft recipe keys a shuffled run may differ from its target's in: the control's own key
#: and the span weight it forces to 0. Everything else must be the target's recipe.
SHUFFLED_LABEL_EXEMPT: Final[frozenset[str]] = frozenset({"shuffled_label", "span_weight"})

#: Ft recipe keys only a loaded tower can state. Compared after training, against the
#: shuffled run's own ft row, rather than guessed before it.
TOWER_ONLY_KEYS: Final[frozenset[str]] = frozenset(
    {"backbone_params", "backbone_vocab", "gradient_checkpointing"}
)

_ABSENT: Final[str] = "<absent>"


@dataclasses.dataclass(frozen=True, slots=True)
class ShuffledGolds:
    """Which training sequences --shuffled-label relabels, and to what.

    ``golds`` maps ``(row_id, slot_name)`` to ``(true gold, permuted gold)`` for every choice
    sequence of ``family``; ``moved`` counts the ones whose gold changed, and ``digest`` is
    the sha256 of the sorted ``row_id, slot_name, permuted gold`` lines.
    """

    family: str
    seed: int
    golds: Mapping[tuple[str, str], tuple[str, str]]
    moved: int
    digest: str

    def recipe(self) -> dict[str, object]:
        """The control row's account of the permutation: the rule and this seed's draw."""
        return {
            **SHUFFLED_LABEL_RECIPE,
            "permutation_seed": self.seed,
            "permutation_sha256": self.digest,
            "rows_permuted": len(self.golds),
            "rows_moved": self.moved,
        }


def shuffle_choice_golds(labels: Sequence[Label], *, family: str, seed: int) -> ShuffledGolds:
    """Permute ``family``'s choice gold LETTERS within groups of equal option count.

    Letters, not classes: ``render`` shuffles each row's options per example, so the decode
    scores a letter, and the chance rate the control compares against is the majority
    LETTER rate of the scored rows. Permuting classes instead would leave the class prior
    learnable -- a model answering the commonest class everywhere would beat the letter
    rate without any leak -- and the control would fail for a reason that is not the split.
    The draw is :func:`qd_train.eval_harness.permute_within_groups`, rung 0's rule.
    """
    picked = [x for x in labels if x.family_id == family and x.slot_kind == SLOT_CHOICE]
    if not picked:
        raise SystemExit(
            f"--shuffled-label: the train set holds no {family} choice sequence to permute"
        )
    keys = [(x.row_id, x.slot_name) for x in picked]
    if len(set(keys)) != len(keys):
        raise SystemExit("--shuffled-label: two train sequences share one (row_id, slot_name)")
    groups = [len([letter for letter in x.letters if letter != NOUL_LETTER]) for x in picked]
    drawn = permute_within_groups([x.gold_letter for x in picked], groups, seed=seed)
    invalid = [
        (key, gold) for key, x, gold in zip(keys, picked, drawn, strict=True)
        if gold not in x.letters
    ]
    if invalid:
        raise SystemExit(
            f"--shuffled-label: {len(invalid)} permuted gold(s) are letters their row does not "
            f"offer, first {invalid[:3]}: rows of one option count offer different letters"
        )
    golds = {key: (x.gold_letter, gold) for key, x, gold in zip(keys, picked, drawn, strict=True)}
    lines = "\n".join(f"{r}\t{s}\t{g}" for (r, s), (_, g) in sorted(golds.items()))
    return ShuffledGolds(
        family=family, seed=seed, golds=golds,
        moved=sum(1 for true, new in golds.values() if true != new),
        digest=hashlib.sha256(lines.encode("utf-8")).hexdigest(),
    )


def apply_shuffled_golds(
    plan: Sequence[Batch], labels_for: Mapping[int, list[Label]], shuffled: ShuffledGolds,
    letter_id: Mapping[str, int],
) -> tuple[list[Batch], int]:
    """``(plan with each permuted gold written over its answer token, tokens rewritten)``.

    ``ft_supervision`` reads the letter target from ``tokens[r, target_index + 1]``, so that
    token is the label. Each one is checked to be the TRUE gold's id before it is replaced,
    and every relabelled sequence must be found in the plan exactly once.
    """
    out: list[Batch] = []
    seen: set[tuple[str, str]] = set()
    rewritten = 0
    for b, batch in enumerate(plan):
        tokens: np.ndarray | None = None
        for r, label in enumerate(labels_for[b]):
            key = (label.row_id, label.slot_name)
            pair = shuffled.golds.get(key)
            if pair is None:
                continue
            if key in seen:
                raise SystemExit(f"--shuffled-label: {key} is in the plan twice")
            seen.add(key)
            true, new = pair
            if batch.target_index is None:  # pragma: no cover - an FT batch always has it
                raise SystemExit(f"batch {b} carries no target_index")
            at = int(batch.target_index[r]) + 1
            if int(batch.tokens[r, at]) != letter_id[true]:
                raise SystemExit(
                    f"--shuffled-label: batch {b} row {r} ({key}) ends in token "
                    f"{int(batch.tokens[r, at])}, not its gold {true!r} ({letter_id[true]})"
                )
            if new == true:
                continue
            if tokens is None:
                tokens = batch.tokens.copy()
            tokens[r, at] = letter_id[new]
            rewritten += 1
        out.append(batch if tokens is None else dataclasses.replace(batch, tokens=tokens))
    missing = sorted(set(shuffled.golds) - seen)
    if missing:
        raise SystemExit(
            f"--shuffled-label: {len(missing)} relabelled sequence(s) are not in the plan, "
            f"first {missing[:3]}"
        )
    return out, rewritten


@dataclasses.dataclass(frozen=True, slots=True)
class ShuffledLabel:
    """The control's target (an eval row and the ft row whose model it scored) and golds."""

    eval_row: LedgerRow
    ft_row: LedgerRow
    golds: ShuffledGolds


def shuffled_label_target(
    rows: Sequence[LedgerRow], row_id: str, *, seed: int
) -> tuple[LedgerRow, LedgerRow]:
    """``(eval row, its ft row)`` for --shuffled-label, or a refusal naming every problem.

    The eval row is the one ``row_id`` names (a unique prefix of 8+ characters): a completed
    --score-val row of this tool, at this run's seed, that is not itself a supplement --
    ``ledger._promotion_units`` refuses a supplement of a supplement, and one at another
    seed. Its ft row is the one its ``ft_run_row_id`` metric names, in the same ledger.
    """
    if len(row_id) < 8:
        raise SystemExit(f"--shuffled-label {row_id!r}: give at least 8 characters of the id")
    hits = [r for r in rows if r.row_id.startswith(row_id)]
    if len(hits) != 1:
        raise SystemExit(f"--shuffled-label {row_id!r} matches {len(hits)} row(s), not 1")
    row = hits[0]
    recipe = dict(row.recipe or {})
    problems: list[str] = []
    if row.run_kind != "eval" or row.status != "completed":
        problems.append(f"it is a {row.status} {row.run_kind!r} row, not a completed eval row")
    if recipe.get("tool") != "tools/real_ft_run.py" or recipe.get("tag") != "epoch-score-val":
        problems.append(
            f"its recipe is {recipe.get('tool')!r} tag {recipe.get('tag')!r}, not this "
            "tool's 'epoch-score-val'"
        )
    if SUPPLEMENT_KEY in recipe:
        problems.append(f"it supplements {recipe[SUPPLEMENT_KEY]}; a supplement joins nothing")
    if row.protocol.seed != seed:
        problems.append(f"it is seed {row.protocol.seed} and this run is seed {seed}")
    ft_ref = row.metrics.get("ft_run_row_id")
    if not (isinstance(ft_ref, Ran) and isinstance(ft_ref.value, str)):
        problems.append("it names no ft_run_row_id, so the recipe it scored cannot be read")
    if problems:
        raise SystemExit(f"--shuffled-label: eval row {row.row_id}: " + "; ".join(problems))
    assert isinstance(ft_ref, Ran)  # narrowed by the refusal above
    ft_hits = [r for r in rows if r.row_id == ft_ref.value]
    if len(ft_hits) != 1:
        raise SystemExit(
            f"--shuffled-label: eval row {row.row_id} scored ft row {ft_ref.value}, which this "
            f"ledger holds {len(ft_hits)} time(s), not once"
        )
    ft = ft_hits[0]
    ft_recipe = dict(ft.recipe or {})
    if (ft.run_kind, ft.status, ft_recipe.get("tag")) != ("ft", "completed", "epoch") or (
        ft.protocol.seed != seed or "shuffled_label" in ft_recipe
    ):
        raise SystemExit(
            f"--shuffled-label: ft row {ft.row_id} is a {ft.status} {ft.run_kind!r} row, tag "
            f"{ft_recipe.get('tag')!r}, seed {ft.protocol.seed}"
            + (", itself a shuffled-label model" if "shuffled_label" in ft_recipe else "")
            + f"; the control needs the completed epoch arm at seed {seed}"
        )
    return row, ft


def recipe_differences(
    target: Mapping[str, object], ours: Mapping[str, object], *, skip: frozenset[str]
) -> list[str]:
    """Every key outside ``skip`` on which the two recipes differ, absence included."""
    return [
        f"{key}: target {target.get(key, _ABSENT)!r}, this run {ours.get(key, _ABSENT)!r}"
        for key in sorted((set(target) | set(ours)) - skip)
        if target.get(key, _ABSENT) != ours.get(key, _ABSENT)
    ]


def planned_ft_recipe(
    args: argparse.Namespace, *, device: str, plan: Sequence[Batch], reader: ShardReader,
    permutation: ChoicePermutation | None, replay: ReplayPlan | None,
    batch_tokens: int | None,
) -> dict[str, object]:
    """The epoch arm's ft recipe as far as argv and the plan decide it, before any tower loads.

    The keys :func:`_train` writes, minus :data:`TOWER_ONLY_KEYS`, which the post-training
    check (:func:`check_shuffled_ft_row`) reads off the row actually written.
    """
    recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py", "tag": "epoch", "device": device, "lr": args.lr,
        "passes": 1, "batches": len(plan), "width": max(int(b.tokens.shape[1]) for b in plan),
        "span_weight": args.span_weight, "deterministic": args.deterministic,
        "shard_hash": reader.header.shard_hash(),
        **_recipe_pieces(
            lower_layers_n=args.lower_layers_n, lower_lr_scale=args.lower_layers_lr_scale,
            beta2=args.beta2, permutation=permutation, replay=replay,
            cap_s=args.wall_clock_cap_s, no_memorise=args.no_memorise,
            batch_tokens=batch_tokens, checkpoint_skip_layers=args.checkpoint_skip_layers,
            fused_adamw=args.fused_adamw, train_attention_mask=args.train_attention_mask,
        ),
    }
    if args.real_backbone is None:
        recipe.update(hidden=args.hidden, heads=args.heads)
    else:
        recipe.update(
            optimizer_recipe=args.optimizer, backbone_snapshot=args.real_backbone.name,
            attn_implementation=args.attn_implementation,
        )
    return recipe


def prepare_shuffled_label(
    row_id: str, *, rows: Sequence[LedgerRow], reader: ShardReader, val: ValSet,
    labels: Sequence[Label], seed: int, planned: Mapping[str, object],
    quick: Sequence[str],
) -> ShuffledLabel:
    """Find the target, refuse every way this run would not be its control, draw the golds.

    Decided before any tower loads: the target's identity (:func:`shuffled_label_target`),
    its shard set, val set and protocol hashes against this run's, its ft recipe against
    ``planned`` outside :data:`SHUFFLED_LABEL_EXEMPT` and :data:`TOWER_ONLY_KEYS`, and
    quickness -- a quick row in a seed family blocks the whole family, so a supplement may
    not be quick where the row it supplements is not.
    """
    eval_row, ft_row = shuffled_label_target(rows, row_id, seed=seed)
    eval_recipe = dict(eval_row.recipe or {})
    problems = recipe_differences(
        dict(ft_row.recipe or {}), planned, skip=SHUFFLED_LABEL_EXEMPT | TOWER_ONLY_KEYS
    )
    for name, theirs, ours in (
        ("eval shard_hash", eval_recipe.get("shard_hash"), reader.header.shard_hash()),
        ("val_shard_hash", eval_recipe.get("val_shard_hash"), val.reader.header.shard_hash()),
        ("data_snapshot_hash", eval_row.protocol.data_snapshot_hash,
         reader.header.data_snapshot_hash),
        ("tokenizer_hash", eval_row.protocol.tokenizer_hash, reader.header.tokenizer_hash),
    ):
        if theirs != ours:
            problems.append(f"{name}: target {theirs!r}, this run {ours!r}")
    if quick and not eval_row.quick:
        problems.append(
            f"this run would be quick ({'; '.join(quick)}) and eval row {eval_row.row_id} is "
            "not; a quick supplement blocks its whole seed family"
        )
    if problems:
        raise SystemExit(
            f"--shuffled-label: this run is not eval row {eval_row.row_id}'s control "
            f"(ft row {ft_row.row_id}) in anything but the shuffle and the span weight:\n  "
            + "\n  ".join(problems)
        )
    return ShuffledLabel(
        eval_row=eval_row, ft_row=ft_row,
        golds=shuffle_choice_golds(labels, family=SHUFFLED_LABEL_FAMILY, seed=seed),
    )


def check_shuffled_ft_row(
    rows: Sequence[LedgerRow], ft_row_id: str, shuffled: ShuffledLabel
) -> None:
    """After training: the ft row this run wrote against the target's, every key compared.

    The pre-training check could not see what only the loaded tower states
    (:data:`TOWER_ONLY_KEYS`, the resolved attention kernel, ``backbone_commit``); this reads
    them off the row actually written. A mismatch writes no control row.
    """
    hits = [r for r in rows if r.row_id == ft_row_id]
    if len(hits) != 1:
        raise SystemExit(f"--shuffled-label: this run's ft row {ft_row_id} is not in the ledger")
    ours, target = hits[0], shuffled.ft_row
    recipe = dict(ours.recipe or {})
    problems = recipe_differences(dict(target.recipe or {}), recipe, skip=SHUFFLED_LABEL_EXEMPT)
    for name in ("backbone_commit", "data_snapshot_hash", "tokenizer_hash"):
        theirs, mine = getattr(target.protocol, name), getattr(ours.protocol, name)
        if theirs != mine:
            problems.append(f"{name}: target {theirs!r}, this run {mine!r}")
    if recipe.get("span_weight") != 0.0 or recipe.get("shuffled_label") != dict(
        SHUFFLED_LABEL_RECIPE
    ):
        problems.append(
            f"span_weight {recipe.get('span_weight')!r} and shuffled_label "
            f"{recipe.get('shuffled_label')!r} are not the control's"
        )
    if problems:
        raise SystemExit(
            f"--shuffled-label: ft row {ft_row_id} trained something other than ft row "
            f"{target.row_id} with its golds shuffled, so no control row is written:\n  "
            + "\n  ".join(problems)
        )


def shuffled_label_state(
    scored: Mapping[str, object], labels: Sequence[Label], *, family: str
) -> TriState:
    """``shuffled_label_control`` over ``family``'s val choice rows, against their REAL golds.

    Chance is the majority gold letter of the rows actually scored, as the harness defines
    it. A row the decode skipped (a letter with no token id) leaves the denominator short,
    and the state says so through ``n_total`` rather than passing on a capped sample.
    """
    population = {
        (x.row_id, x.slot_name): x.gold_letter
        for x in labels if x.family_id == family and x.slot_kind == SLOT_CHOICE
    }
    if not population:
        return NotRun(reason=f"the val set holds no {family} choice row to score")
    verdicts = scored["verdicts"]
    if not isinstance(verdicts, list):  # pragma: no cover - _decode's own shape
        raise TypeError("scored['verdicts'] is not a list")
    picked = [
        v for v in verdicts
        if v.get("kind") == "choice" and (str(v["row_id"]), str(v["slot_name"])) in population
    ]
    if not picked:
        return NotRun(reason=f"none of {len(population)} {family} val choice rows was decoded")
    correct = sum(1 for v in picked if bool(v["correct"]))
    state = shuffled_label_control(
        correct / len(picked),
        [population[(str(v["row_id"]), str(v["slot_name"]))] for v in picked],
        n_eval=len(picked),
    )
    if isinstance(state, Ran) and len(picked) < len(population):
        state = dataclasses.replace(
            state, n_total=len(population),
            detail=f"{state.detail}; {len(population) - len(picked)} of {len(population)} "
            "rows were not decoded",
        )
    return state


def _record_shuffled_label(
    run: dict[str, object], scored: dict[str, object], *, ledger: Ledger, reader: ShardReader,
    val: ValSet, shuffled: ShuffledLabel, quick_reasons: Sequence[str], decode_s: float,
) -> str:
    """The control row: ``controls.shuffled_label``, supplementing the target eval row.

    The target eval row's PROTOCOL, so it lands in that row's seed family; its own settings
    in ``recipe``, which names the eval row under ``SUPPLEMENT_KEY``. Nothing else measured
    is recorded on it: ``ledger._joined`` lets any measured failure fail the unit, and this
    model's degenerate head is expected -- it is a statement about the shuffled model, not
    about the target.
    """
    target = shuffled.eval_row
    recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py", "tag": "epoch-shuffled-label",
        SUPPLEMENT_KEY: target.row_id, "device": run["device"],
        "shuffled_label": shuffled.golds.recipe(),
        "shuffled_ft_row_id": run["ft_row_id"], "target_ft_row_id": shuffled.ft_row.row_id,
        "span_weight": run["span_weight"], "shard_hash": reader.header.shard_hash(),
        "val_shard_hash": val.reader.header.shard_hash(),
    }
    recorder = _recorder(
        ledger, reader=reader, seed=int(run["seed"]), recipe=recipe,
        run_kind="eval", quick_reasons=quick_reasons,
        # The decode, not the training run: see _record_verdict.
        wall_clock_s=decode_s,
        cost=run["cost"],  # type: ignore[arg-type]
        protocol=Protocol(**target.protocol.to_json()),
        notes=(
            f"tools/real_ft_run.py --shuffled-label for eval row {target.row_id}: ft row "
            f"{run['ft_row_id']} trained ft row {shuffled.ft_row.row_id}'s recipe with "
            f"{shuffled.golds.family} choice golds permuted ({shuffled.golds.moved} of "
            f"{len(shuffled.golds.golds)} moved) and the span channel off; its "
            f"{shuffled.golds.family} val choice rows decoded the way answer.rs does and "
            "scored against their REAL golds."
        ),
    )
    with recorder:
        recorder.metric(
            "shuffled_ft_row_id",
            Ran(passed=True, value=run["ft_row_id"],
                detail="the train_ft row of the shuffled-label model"),
        )
        recorder.control(
            "shuffled_label", shuffled_label_state(scored, val.labels, family=SHUFFLED_LABEL_FAMILY)
        )
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id


def _check_shuffled_label_flags(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    """Refuse every combination --shuffled-label cannot be the control in, then force the
    span weight to 0. Called after the ``--span-weight`` guard, which it relaxes for itself."""
    if args.shuffled_label is None:
        return
    absent = [
        flag for flag, given in (
            ("--epoch", args.epoch), ("--no-memorise", args.no_memorise),
            ("--score-val", args.score_val),
        ) if not given
    ]
    if absent:
        raise SystemExit(
            f"--shuffled-label trains the epoch arm on permuted golds and scores it on the val "
            f"set; it needs {', '.join(absent)}"
        )
    clashing = [
        flag for flag, given in (
            ("--score-checkpoint", args.score_checkpoint is not None),
            ("--score-plan", args.score_plan is not None),
            ("--needle", args.needle), ("--ood", args.ood),
            ("--needle-control", args.needle_control is not None),
            ("--verdicts-out", args.verdicts_out is not None),
            ("--suite-verdicts-out", args.suite_verdicts_out is not None),
        ) if given
    ]
    if clashing:
        raise SystemExit(
            f"--shuffled-label writes one control row and scores no gate; {', '.join(clashing)} "
            "would measure the shuffled model as if it were the recipe's"
        )
    # A checkpoint is named by (tag, seed, device) alone, so the control's would be
    # epoch-seed<N>-<device>.json -- the real arm's file, in the directory a mirrored launch
    # line names, rewritten with the shuffled model's weights. Nothing consumes the control's.
    saving = [
        flag for flag, given in (
            ("--checkpoint-dir", args.checkpoint_dir is not None),
            ("--checkpoint-every", bool(args.checkpoint_every)),
            ("--resume-from", args.resume_from is not None),
        ) if given
    ]
    if saving:
        raise SystemExit(
            f"--shuffled-label saves and resumes no weights: {', '.join(saving)} would write "
            "or read epoch-seed<N>-<device>.json, the real epoch arm's checkpoint name"
        )
    if len(args.seeds) != 1 or not args.devices or len(args.devices) != 1:
        raise SystemExit(
            "--shuffled-label supplements one eval row, which has one seed and one device: "
            "pass exactly one --seeds and one --devices entry"
        )
    # argparse accepts any unique prefix, and "--sp" is unique to --span-weight here.
    given = [t for t in raw_argv if t.startswith("--sp") and "--span-weight".startswith(
        t.split("=", 1)[0])]
    if given:
        raise SystemExit(
            "--shuffled-label trains the span channel at weight 0, so the real span gold "
            "cannot pull the shared tower toward the defect; --span-weight would determine "
            "nothing"
        )
    args.span_weight = 0.0


def _check_score_checkpoint_flags(args: argparse.Namespace) -> None:
    """Every refusal a ``--score-checkpoint`` scoring is decidable on from argv: one seed's
    checkpoint, an average or a logit ensemble, and what each must be paired with. A
    ``--score-plan`` kind is checked by this same function, on its own namespace."""
    averaged = _is_average(args.score_checkpoint)
    ensemble = _is_ensemble(args.score_checkpoint)
    needed = {
        "--score-val": args.score_val, "--real-backbone": args.real_backbone,
        "--ft-ledger": args.ft_ledger, "--devices": args.devices,
    }
    if not averaged:
        needed["--ft-row-id"] = args.ft_row_id
    absent = [flag for flag, value in needed.items() if not value]
    if absent:
        raise SystemExit(f"--score-checkpoint needs {', '.join(absent)}")
    if averaged and args.ft_row_id is not None:
        raise SystemExit(
            "--score-checkpoint of an average (.safetensors) is paired with the ft rows "
            "its manifest names, one per input; --ft-row-id would name one of them"
        )
    if ensemble:
        towers = score_paths(args.score_checkpoint)
        if any(p.suffix == AVERAGED_SUFFIX for p in towers):
            raise SystemExit(
                "a logit ensemble's towers are per-seed epoch checkpoints (.json); an "
                "average (.safetensors) is scored on its own"
            )
        if len(ft_row_ids(args.ft_row_id)) != len(towers) or len(args.seeds) != len(towers):
            raise SystemExit(
                f"a logit ensemble of {len(towers)} checkpoints needs one --ft-row-id and "
                f"one --seeds entry per checkpoint, in its order; got "
                f"{len(ft_row_ids(args.ft_row_id))} and {len(args.seeds)}"
            )
    elif not averaged and len(ft_row_ids(args.ft_row_id)) != 1:
        raise SystemExit("one --score-checkpoint is paired with one --ft-row-id")
    if len(args.devices) != 1 or (not averaged and not ensemble and len(args.seeds) != 1):
        raise SystemExit(
            "--score-checkpoint scores one checkpoint on one device: pass exactly one "
            "--devices entry and the checkpoint's one seed in --seeds"
        )
    if averaged and len(args.seeds) < 2:
        raise SystemExit(
            "--score-checkpoint of an average needs --seeds to list the average's seeds, "
            "two or more (J7: --seeds 0 1 2); they are checked against its manifest"
        )
    if args.epoch or args.resume_from is not None:
        raise SystemExit(
            "--score-checkpoint trains nothing; --epoch and --resume-from would train"
        )


def _check_suite_logits_flags(args: argparse.Namespace) -> None:
    """``--suite-logits`` is refused wherever it would write nothing or be dropped.

    It is read by every needle decode of a scored checkpoint: in-process for
    ``--needle-control`` and a ``--score-plan`` gate pass, and in the worker process
    (:func:`needle_worker_main`) for a single ``--score-checkpoint`` gate row, which gets
    the flag on the scoring process's own argv and hands the scores back in its verdicts
    (:func:`score_needle` refuses one without them). The OOD suite's lines carry both
    passes' letter rows with or without it.
    """
    if not args.suite_logits:
        return
    if args.score_checkpoint is None and args.score_plan is None:
        raise SystemExit(
            "--suite-logits writes a scored checkpoint's per-case scores: it needs "
            "--score-checkpoint or --score-plan"
        )
    if args.suite_verdicts_out is None:
        raise SystemExit(
            "--suite-logits adds scores to the --suite-verdicts-out lines; without that file "
            "they would be decoded and dropped"
        )


def _check_piece_flags(args: argparse.Namespace) -> None:
    """Refuse, at argv time, every combination of the ported-piece flags that would record a
    value that determined nothing, or run a piece without what it needs. Resolves defaults
    in place (``lower_layers_lr_scale``, ``beta2``, ``tokenizer_json``)."""
    if args.checkpoint_skip_layers < 0:
        raise SystemExit(
            f"--checkpoint-skip-layers must not be negative, got {args.checkpoint_skip_layers}"
        )
    if args.checkpoint_skip_layers and args.real_backbone is None:
        raise SystemExit(
            "--checkpoint-skip-layers needs --real-backbone: the stand-in is one block with "
            "no checkpointing to be selective about"
        )
    if args.train_attention_mask != "padding":
        if args.real_backbone is None:
            raise SystemExit(
                f"--train-attention-mask {args.train_attention_mask} needs --real-backbone: "
                "the stand-in has no SDPA layer to switch"
            )
        if args.replay_shards is not None:
            raise SystemExit(
                f"--train-attention-mask {args.train_attention_mask} with --replay-shards: "
                "replay's KL term runs its own forward through hidden(), which keeps the "
                "mask, so the switch would cover part of training while the recipe named all"
            )
    if args.fused_adamw and (args.real_backbone is None or args.optimizer != "master"):
        raise SystemExit(
            "--fused-adamw needs --real-backbone and --optimizer master: it fuses the fp32-"
            "master optimizer's inner AdamW, and on any other recipe it would build nothing"
        )
    if args.lower_layers_n < 0:
        raise SystemExit(f"--lower-layers-n must not be negative, got {args.lower_layers_n}")
    if args.lower_layers_n:
        if args.real_backbone is None:
            raise SystemExit(
                "--lower-layers-n needs --real-backbone: the stand-in is one block with no "
                "decoder layers to split, and the recipe would record a split of nothing"
            )
        if args.lower_layers_lr_scale is None:
            args.lower_layers_lr_scale = RSI_LOWER_LR_SCALE
    elif args.lower_layers_lr_scale is not None:
        raise SystemExit("--lower-layers-lr-scale without --lower-layers-n scales no layer")
    else:
        args.lower_layers_lr_scale = 1.0
    if args.beta2 is None:
        args.beta2 = DEFAULT_BETA2
    elif not 0.0 < args.beta2 < 1.0:
        raise SystemExit(f"--beta2 must be in (0, 1), got {args.beta2}")
    # The snapshot's own tokenizer, when one is there: read for every offered letter's id
    # (decode) and by --option-permutation-seed.
    if args.tokenizer_json is None and args.real_backbone is not None:
        candidate = args.real_backbone / "tokenizer.json"
        if candidate.is_file():
            args.tokenizer_json = candidate
    if args.option_permutation_seed is not None:
        if not args.epoch:
            raise SystemExit(
                "--option-permutation-seed applies to the epoch arm only (--epoch): the "
                "memorisation arm's floors are statements about FIXED rows, and a per-pass "
                "permutation would change the rows the floor was computed over"
            )
        if args.tokenizer_json is None or not args.tokenizer_json.is_file():
            raise SystemExit(
                "--option-permutation-seed needs the tokenizer.json the shards were built "
                f"with (got {args.tokenizer_json}): which token ids end a line is a fact "
                "about the vocabulary, and the option-line search refuses to guess it"
            )
    if args.tokenizer_json is not None and not args.tokenizer_json.is_file():
        raise SystemExit(f"--tokenizer-json {args.tokenizer_json} does not exist")
    replay_flags = {
        "--replay-attestation": args.replay_attestation,
        "--replay-cache": args.replay_cache,
        "--replay-weight": args.replay_weight,
    }
    if args.replay_shards is None:
        given = [k for k, v in replay_flags.items() if v is not None]
        if args.replay_every != DEFAULT_REPLAY_EVERY:
            given.append("--replay-every")
        if given:
            raise SystemExit(f"{', '.join(given)} without --replay-shards does nothing")
    else:
        missing = [k for k, v in replay_flags.items() if v is None]
        if missing:
            raise SystemExit(f"--replay-shards needs {', '.join(missing)}")
        if not args.epoch:
            raise SystemExit(
                "--replay-shards applies to the epoch arm only (--epoch), for the same "
                "reason permutation does: the memorisation arm's floors are about its rows"
            )
        if not (math.isfinite(args.replay_weight) and args.replay_weight > 0.0):
            raise SystemExit(f"--replay-weight must be positive, got {args.replay_weight}")
        if args.replay_every < 1:
            raise SystemExit(f"--replay-every must be at least 1, got {args.replay_every}")
        if not args.replay_attestation.is_file():
            raise SystemExit(f"--replay-attestation {args.replay_attestation} does not exist")
        if not (args.replay_shards / HEADER_NAME).is_file():
            raise SystemExit(f"--replay-shards {args.replay_shards} holds no shard header")
    if args.verdicts_out is not None:
        if not args.score_val:
            raise SystemExit(
                "--verdicts-out writes the val verdicts --score-val decodes; without it "
                "there are none"
            )
        if args.verdicts_out.exists():
            raise SystemExit(f"--verdicts-out {args.verdicts_out} already exists")
    if args.suite_verdicts_out is not None:
        if not (args.needle or args.ood or args.composed_slice is not None):
            raise SystemExit(
                "--suite-verdicts-out writes the needle and OOD suites' raw verdicts and the "
                "composed slice's per-row lines; without --needle, --ood or --composed-slice "
                "there are none"
            )
        if args.suite_verdicts_out.exists():
            raise SystemExit(f"--suite-verdicts-out {args.suite_verdicts_out} already exists")


def _remap_post(reader: ShardReader) -> Any:
    """``source id -> post-remap id`` for this set, ``None`` for a token the remap dropped."""
    if reader.remap is None:
        raise SystemExit(f"{reader.root}: no remap table beside the shards")
    old_to_new = reader.remap.old_to_new

    def post(source_id: int) -> int | None:
        if not 0 <= source_id < old_to_new.size:
            return None
        new = int(old_to_new[source_id])
        return new if new >= 0 else None

    return post


def vocab_letter_ids(
    reader: ShardReader, *, tokenizer_json: Path, letter_id: Mapping[str, int]
) -> dict[str, int]:
    """Every option letter's post-remap id, read off the tokenizer's vocabulary.

    A superset of ``letter_id`` (the ids read off the shard set by gold correspondence),
    and cross-checked against it: a tokenizer.json that disagrees on any letter is not the
    one the shards were built with and is refused. A letter the remap dropped is absent --
    no row of this set can offer it.
    """
    from qd_data.schema import OPTION_LETTERS

    vocab = json.loads(tokenizer_json.read_text(encoding="utf-8"))["model"]["vocab"]
    post = _remap_post(reader)
    letters: dict[str, int] = {}
    for letter in (*OPTION_LETTERS, NOUL_LETTER):
        if letter in vocab and (new := post(int(vocab[letter]))) is not None:
            letters[letter] = new
    for letter, new in letter_id.items():
        if letters.get(letter) != new:
            raise SystemExit(
                f"{tokenizer_json}: letter {letter!r} is post-remap id {letters.get(letter)} "
                f"by this tokenizer and {new} by the shard set. It is not the tokenizer the "
                "shards were built with."
            )
    return letters


def _permutation_spec(
    reader: ShardReader, *, tokenizer_json: Path, letter_id: Mapping[str, int], seed: int
) -> ChoicePermutation:
    """Post-remap letter ids and newline-terminated ids, read off ``tokenizer_json``.

    Cross-checked against ``_letter_ids`` -- the ids read off the shard set by
    correspondence -- so a tokenizer.json that is not the one the shards were built with is
    refused rather than used to cut option lines in the wrong places.
    """
    letters = vocab_letter_ids(reader, tokenizer_json=tokenizer_json, letter_id=letter_id)
    vocab = json.loads(tokenizer_json.read_text(encoding="utf-8"))["model"]["vocab"]
    post = _remap_post(reader)
    line_ends = frozenset(
        new for sid in newline_terminated_ids(vocab) if (new := post(sid)) is not None
    )
    if NOUL_LETTER not in letters:
        raise SystemExit(f"the noul letter {NOUL_LETTER!r} is not in this shard set's remap")
    return ChoicePermutation(
        seed=seed, letter_ids=letters, noul_id=letters[NOUL_LETTER], line_end_ids=line_ends
    )


def _alphabets(
    plan: list[Batch], labels_for: Mapping[int, list[Label]]
) -> dict[int, list[tuple[str, ...] | None]]:
    """Per plan batch, per row: the choice letters in rendered order (noul excluded), or None."""
    return {
        b: [
            tuple(x for x in label.letters if x != NOUL_LETTER)
            if label.slot_kind == SLOT_CHOICE
            else None
            for label in labels_for[b]
        ]
        for b in range(len(plan))
    }


def replay_corpus_identity(
    *, rev: str, max_pairs: int, commitpackft: Path | None, defect_class: Path | None,
    defect_max_rows: int | None, repo_history: bool = True,
    general_record: Path | None = None, general_max_rows: int | None = None,
    defect_noul: Path | None = None,
) -> dict[str, object]:
    """What ``ft_splits`` was called with, as the replay attestation records it. One
    function, used by ``tools/replay_decontam.py`` to write it and by ``_replay_plan`` to
    check it, so the two cannot spell the corpus differently. ``repo_history`` is named
    only when False, and the general record only when given, so every attestation written
    before either existed still matches. The record is named by its sha256, as the
    pipeline's recipe names it: its general families are val and held-out targets too, and
    an attestation made without them compared the replay set against fewer rows."""
    if general_record is None and general_max_rows is not None:
        raise ValueError("general_max_rows without general_record read nothing")
    general: dict[str, object] = {}
    if general_record is not None:
        import real_tokenizer_pipeline as pipeline

        general = {
            "general_record_sha256": hashlib.sha256(general_record.read_bytes()).hexdigest(),
            "general_max_rows": (
                pipeline.DEFAULT_GENERAL_MAX_ROWS if general_max_rows is None
                else general_max_rows
            ),
        }
    return {
        "rev": rev,
        "max_pairs": max_pairs,
        "commitpackft": None if commitpackft is None else commitpackft.name,
        "defect_class": None if defect_class is None else defect_class.name,
        "defect_max_rows": defect_max_rows,
        **({} if repo_history else {"repo_history": False}),
        **general,
        # Named by its examples' sha256, and only when given, so every attestation written
        # before it still matches: a corpus with the noul rows is not the corpus without.
        **(
            {} if defect_noul is None else {
                "defect_noul_examples_sha256": str(json.loads(
                    (defect_noul / "manifest.json").read_text(encoding="utf-8")
                )["examples_sha256"]),
            }
        ),
    }


def check_replay_role(reader: ShardReader) -> None:
    """Refuse a replay set whose sequence index is not marked ``replay_only``.

    The mirror of ``pair_labels``' gold check, so neither direction can be crossed: a gold
    set is never replayed and a replay set is never relabelled. Fails closed -- a set with
    no sequence index, or an index with no role, is refused, because nothing then says its
    rows were written for replay.
    """
    from qd_data.general import REPLAY_ONLY

    index = getattr(reader, "sequence_index", None)
    role = None if index is None else getattr(index, "role", None)
    if role != REPLAY_ONLY:
        raise SystemExit(
            f"--replay-shards {reader.root}: sequence index role is {role!r}, not "
            f"{REPLAY_ONLY!r}. Write replay sets with write_shards(replay=True); a gold set, "
            "or one whose role is unrecorded, is not replayed."
        )


def _replay_plan(
    args: argparse.Namespace, *, config: DataConfig, train: ShardReader, width: int,
    letter_id: Mapping[str, int], val_rows: list[DataRow], rev: str,
) -> ReplayPlan:
    """Open the replay shard set and hold it to its attestation before anything trains."""
    from qd_data.render import render
    from qd_train.replay import prompt_content

    replay_reader = ShardReader(args.replay_shards, config=config, repo_root=args.out)
    check_replay_role(replay_reader)
    if replay_reader.header.remap_hash != train.header.remap_hash:
        raise SystemExit(
            f"--replay-shards was tokenized under remap {replay_reader.header.remap_hash[:16]} "
            f"and the train set under {train.header.remap_hash[:16]}; the same id would name "
            "different tokens in the two"
        )
    raw_attestation = args.replay_attestation.read_bytes()
    attestation = json.loads(raw_attestation)
    corpus = replay_corpus_identity(
        rev=rev, max_pairs=args.max_pairs, commitpackft=args.commitpackft,
        defect_class=args.defect_class, defect_max_rows=args.defect_max_rows,
        repo_history=args.repo_history, general_record=args.general_record,
        general_max_rows=args.general_max_rows, defect_noul=args.defect_noul,
    )
    if attestation.get("corpus") != corpus:
        raise SystemExit(
            f"the attestation was made against corpus {attestation.get('corpus')} and this "
            f"run's is {corpus}: its val/heldout target sets are not this run's"
        )
    # Only the val set is recomputed here. The heldout split is Rule 3's: it is checked by
    # tools/replay_decontam.py, a process that does not train, and this run holds it to
    # that check through the corpus identity above rather than by reading it.
    val_texts: dict[str, str] = {}
    for row in val_rows:
        rendered = render(row.request, seed=None)
        for slot in rendered.slots:
            val_texts[f"{row.row_id}#{slot.name}"] = prompt_content(
                rendered.prompt_for(slot.name)
            )
    try:
        check_attestation(
            attestation, replay_shard_hash=replay_reader.header.shard_hash(),
            verify_targets={"val": val_texts}, required_targets=("val", "heldout"),
        )
    except ReplayRefusal as exc:
        raise SystemExit(f"--replay-attestation: {exc}") from exc
    batches = [
        b
        for b in replay_reader.batches(
            batch_tokens=int(max(train.header.buckets)), seed=config.seed, epoch=0
        )
        if int(b.tokens.shape[1]) <= width
    ]
    if not batches:
        raise SystemExit(f"no replay batch is at most {width} tokens wide")
    return ReplayPlan(
        batches=batches,
        shard_hash=replay_reader.header.shard_hash(),
        cache_path=args.replay_cache,
        letter_ids=tuple(sorted(letter_id.values())),
        weight=float(args.replay_weight),
        every=int(args.replay_every),
        attestation_sha256=hashlib.sha256(raw_attestation).hexdigest(),
    )


def write_verdicts_jsonl(path: Path, lines: list[dict[str, object]]) -> None:
    """Every line to ``path``, atomically, refusing to overwrite. One JSON object per line."""
    from qd_train.replay import ReplayRefusal as _Refusal
    from qd_train.replay import write_text_atomic

    required = ("eval_row_id", "seed", "row_id", "kind", "correct")
    for i, line in enumerate(lines):
        missing = [k for k in required if k not in line]
        if missing or not isinstance(line["correct"], bool):
            raise ValueError(f"verdict line {i} is missing {missing} or has a non-bool correct")
    body = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    try:
        write_text_atomic(path, body)
    except _Refusal as exc:
        raise SystemExit(f"--verdicts-out: {exc}") from exc


def _verdict_lines(
    scored: Mapping[str, object], *, eval_row_id: str, seed: int
) -> list[dict[str, object]]:
    """The JSONL lines ``--verdicts-out`` writes for one eval row."""
    out: list[dict[str, object]] = []
    for v in scored["verdicts"]:  # type: ignore[union-attr]
        line: dict[str, object] = {
            "eval_row_id": eval_row_id,
            "seed": int(seed),
            "row_id": v["row_id"],
            "kind": v["kind"],
            "correct": bool(v["correct"]),
            "top": v["top"],
            "gold_row": v.get("gold_row"),
        }
        if "slot_name" in v:
            line["slot_name"] = v["slot_name"]
        if "expected_abstain" in v:
            # What an always-abstaining span head is scored against
            # (ft_linear_control's paired_margin_vs_abstain_constant.span).
            line["expected_abstain"] = bool(v["expected_abstain"])
        # What a margin or calibration fit reads: the argmax alone cannot be refitted. And
        # the family and the permuted pass, from which every per-family metric on the eval
        # row (permutation_consistency, the in-distribution abstentions, ece) is recomputed.
        for key in (*VERDICT_DISTRIBUTION_KEYS, *VERDICT_FAMILY_KEYS):
            if key in v:
                line[key] = v[key]
        out.append(line)
    return out


#: Verdict fields ``--verdicts-out`` carries whenever the decoder wrote them: the letter
#: distribution and where the abstention sits in it.
VERDICT_DISTRIBUTION_KEYS: Final[tuple[str, ...]] = (
    "noul_row", "rows", "language", "noul_probability", "row_logits",
)

#: The row's family (``_decode``) and, on a choice row with a derangement, its second pass
#: (:func:`annotate_second_pass`): ``top_permuted`` in the permuted pass's own order,
#: ``permutation_agreed``, and ``perm``. Carried only where they were written.
VERDICT_FAMILY_KEYS: Final[tuple[str, ...]] = (
    "family_id", "top_permuted", "permutation_agreed", "perm",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="the pipeline's --out directory")
    parser.add_argument("--passes", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"default: {' '.join(map(str, DEFAULT_SEEDS))}")
    # Pre-registration, beside the seed count it is about. Without these two the row records
    # NotRun rather than nothing, because a sweep that never said what it was looking for
    # must not be indistinguishable from one that looked and found nothing.
    parser.add_argument(
        "--prior-sd",
        type=float,
        help=(
            "seed sd from a PREVIOUS measurement of this configuration, used to state what "
            "--seeds can resolve. Not from this sweep, and not from an arm that collapsed: "
            "the spread of a model that never fit is not the spread of one that did"
        ),
    )
    parser.add_argument(
        "--target-difference",
        type=float,
        help=(
            "the smallest effect worth detecting, in the units of the channel loss it is "
            "compared against. Stated up front so the verdict row can say whether this "
            "sweep could have seen it"
        ),
    )
    # Sentinels, not values: which default is right depends on --real-backbone, and a
    # default that silently applies to the wrong backbone is fault 1 below.
    parser.add_argument(
        "--span-weight", type=float, default=1.0,
        help=(
            "multiplier on the span loss in the summed objective, for BOTH the stand-in and "
            "the real backbone. Default 1.0, which is what every run before 2026-09-20 used "
            "-- deliberately unchanged: rung 0 collapsed at this setting, but that was a "
            "1.5M-parameter byte model and moving this default on the strength of it would "
            "be transferring a threshold that has not been measured here"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help=(
            "where to write a resume point. One file per (tag, seed), rewritten in place; "
            "Checkpoint.write is atomic, so the previous one is readable until the instant "
            "the new one replaces it"
        ),
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help=(
            "optimizer steps between checkpoints. 0, the default, means the loop never "
            "calls on_checkpoint and nothing survives a kill -- which is what every run "
            "before 2026-09-21 did"
        ),
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help=(
            "a checkpoint written by --checkpoint-dir. The schedule, seed and epoch must "
            "match the run it was taken from; train_ft refuses otherwise rather than "
            "resuming into a different run"
        ),
    )
    parser.add_argument(
        "--hidden", type=int, default=None,
        help=f"stand-in only; default {STANDIN_HIDDEN}",
    )
    parser.add_argument(
        "--heads", type=int, default=None,
        help=f"stand-in only; default {STANDIN_HEADS}",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help=(
            f"default {STANDIN_LR:g} for the stand-in, {REAL_BACKBONE_LR:g} for "
            "--real-backbone. The stand-in's rate on pretrained weights destroys them."
        ),
    )
    parser.add_argument(
        "--max-width", type=int, default=5383,
        help="arm 2 trains on every real batch at most this wide. The rule is stated rather "
             "than tuned: it is what this Mac can repeat often enough to reach a floor.",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help=(
            "as the pipeline's --max-pairs; default 400. Refused under --no-repo-history "
            "without --commitpackft, where the rebuild reads nothing it bounds"
        ),
    )
    parser.add_argument(
        "--no-repo-history", dest="repo_history", action="store_false",
        help=(
            "the shard set was built with tools/real_tokenizer_pipeline.py "
            "--no-repo-history: rebuild its labels without reading this repository's git "
            "history. Checked against the train manifest's sources; a mismatch is refused"
        ),
    )
    parser.add_argument(
        "--rev", default="0632f693d3b765b726499e7b4bf19c67959b75cb",
        help="the corpus revision the shard set was built at, as a full 40-character sha",
    )
    parser.add_argument(
        "--commitpackft",
        type=Path,
        default=None,
        help=(
            "the commitpackft download the shard set was built from, exactly as passed to "
            "tools/real_tokenizer_pipeline.py --commitpackft. The labels are rebuilt by "
            "re-running that corpus, so without it a set built from the download is "
            "refused: its rows are not this repository's history"
        ),
    )
    parser.add_argument(
        "--defect-class", type=Path, default=None,
        help=(
            "the qd-mutate corpus directory the shard set was built with, exactly as passed "
            "to tools/real_tokenizer_pipeline.py --defect-class. Required when the set's "
            "train manifest admits the code.defect_class source, refused when it does not"
        ),
    )
    parser.add_argument(
        "--defect-download", type=Path, default=None,
        help="as the pipeline's --defect-download; default the pipeline's own default",
    )
    parser.add_argument(
        "--defect-max-rows", type=int, default=None,
        help="as the pipeline's --defect-max-rows: the same sha256-ordered cap, or none",
    )
    parser.add_argument(
        "--defect-noul", type=Path, default=None,
        help=(
            "the qd-noul-rows corpus the shard set was built with, exactly as passed to "
            "tools/real_tokenizer_pipeline.py --defect-noul. A rebuild without it (or with it, "
            "for a set built without) is refused by the sequence-index pairing"
        ),
    )
    parser.add_argument(
        "--general-record", type=Path, default=None,
        help=(
            "the general-family fetch record the shard set was built with, exactly as passed "
            "to tools/real_tokenizer_pipeline.py --general-record. Required when the set's "
            "train manifest was fed a family only the record supplies (MMLU, CSQA, CLINC), "
            "refused when none of the record's datasets fed it"
        ),
    )
    parser.add_argument(
        "--general-max-rows", type=int, default=None,
        help="as the pipeline's --general-max-rows: the same per-file bound; default its default",
    )
    parser.add_argument(
        "--replay-partition", action="store_true",
        help=(
            "the shard set was built with the pipeline's --replay-shards, whose replay-only "
            "rows left the gold train split: rebuild the split the same way. Needs "
            "--general-record; checked against the replay manifest beside the train manifest"
        ),
    )
    parser.add_argument("--epoch", action="store_true", help="also run arm 1, the real epoch")
    parser.add_argument(
        "--no-memorise", action="store_true",
        help=(
            "skip arm 2, the memorisation, and run only the epoch arm. Needs --epoch. The "
            "memorisation arm is a Mac smoke convenience and a subsample by construction: "
            "on a rented box it is time spent on rows no decision can use. Recorded in the "
            "recipe"
        ),
    )
    parser.add_argument(
        "--wall-clock-cap-s", type=float, default=WALL_CLOCK_CAP_S,
        help=(
            f"the wall-clock cap on EACH arm of each seed, in seconds; default "
            f"{WALL_CLOCK_CAP_S:g}. At most {MAX_WALL_CLOCK_CAP_S:g} "
            f"({MAX_WALL_CLOCK_CAP_S / 3600:g} h): the program's MAX_CAP_S, which is below "
            "the campaign's 72 h approval. The cost estimate is priced from it, so above "
            "rule 4's $20 line a run needs --approved-by and is armed with auto-terminate. "
            "Recorded in the recipe when it differs from the default"
        ),
    )
    parser.add_argument(
        "--batch-tokens", type=int, default=None,
        help=(
            "positions per optimizer step (rows x padded width). Default: the widest bucket, "
            "what every earlier row trained at -- ~1.4k positions a step on the phase-3 set, "
            "which leaves a GH200 waiting on the host. Larger is a different recipe (fewer, "
            "bigger steps; the lr is tuned for the default), so it is recorded in the recipe "
            "when it differs"
        ),
    )
    parser.add_argument(
        "--score-val",
        action="store_true",
        help=(
            "after each seed's epoch arm, score that model on the val shard set the "
            "pipeline wrote with --val-shards (shards/val), decoding every row the way "
            "crates/qd-runtime/src/answer.rs does, and write one eval row per seed. Needs "
            "--epoch: the memorisation arm trains on a subset, and a val number is about "
            "the model that saw the whole train split once"
        ),
    )
    parser.add_argument(
        "--score-checkpoint",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "score a SAVED epoch checkpoint (<tag>-seed<N>-<device>.json, with its sidecar "
            "beside it) on the val set instead of training: reads only its tower and span "
            "head, never its optimizer. Needs --score-val, --real-backbone, one --devices "
            "entry, and --ft-ledger/--ft-row-id naming the ft row that wrote it. Or an "
            "AVERAGE: the .safetensors tools/ckpt_average.py wrote, with its .manifest.json "
            "beside it, paired with the ft rows the manifest names (--ft-ledger, no "
            "--ft-row-id) and scored under --seeds listing the average's seeds. Or a LOGIT "
            "ENSEMBLE: two or more per-seed checkpoints, each paired with its own ft row "
            "(--ft-row-id, one per checkpoint, in order) under --seeds listing their seeds in "
            "order; every decode reads the mean of the towers' log-probabilities (tag ens<N>). "
            "Or a METAL EXPORT: epoch-seed<N>-metal.safetensors, one seed's weights the Rust "
            "trainer (crates/qd-train) exported, with its .manifest.json beside it, paired "
            "with the ft row it wrote (--ft-ledger/--ft-row-id) under its one --seeds; every "
            "row it is scored into records recipe.trained_by"
        ),
    )
    parser.add_argument(
        "--ft-ledger", type=Path, default=None,
        help="the ledger holding the ft row(s) that wrote --score-checkpoint",
    )
    parser.add_argument(
        "--ft-row-id", nargs="+", default=None,
        help=(
            "that ft row's row_id (the full uuid or a unique prefix of it); for an ensemble, "
            "one per --score-checkpoint, in the same order"
        ),
    )
    parser.add_argument(
        "--score-plan", type=Path, default=None, metavar="PLAN.json",
        help=(
            "score several models in ONE invocation, the shard set, val set and suites built "
            "once: {\"kinds\": [{\"name\", \"checkpoints\", \"ft_row_ids\" (not for an "
            "average), \"seeds\", \"passes\"}]}, each kind exactly what --score-checkpoint "
            "and its --ft-row-id/--seeds would name. passes: \"gates\" (the val pass and every "
            "gate: the --score-checkpoint row), \"ood\" (the OOD suite alone: a quick "
            "diagnostic row, tag <kind tag>-ood-diagnostic) or \"composed\" (the report-only "
            "composed slice, --composed-slice: a quick row, tag <kind tag>-composed-slice). One "
            "model is loaded per kind and freed before the next; --verdicts-out/"
            "--suite-verdicts-out are written per kind as <stem>-<name><suffix>, a composed "
            "pass's lines as <stem>-<name>.composed<suffix>. Needs --score-val, "
            "--real-backbone, --ft-ledger and one --devices entry; refuses --score-checkpoint, "
            "--ft-row-id and --seeds"
        ),
    )
    parser.add_argument(
        "--composed-slice", type=Path, default=None, metavar="SLICE_OUT",
        help=(
            "a --score-plan 'composed' pass's slice: the report-only slice build's --out, "
            "holding shards/val-report-only-composed. Pinned to build row "
            f"{COMPOSED_SLICE_ROW[:8]} ({COMPOSED_SLICE_LEDGER.name}) and its shard hash; "
            "refused unless its header says report_only and refuse-gold"
        ),
    )
    parser.add_argument(
        "--composed-slice-corpus", type=Path, default=COMPOSED_SLICE_CORPUS,
        help=(
            "the slice corpus (manifest.json and examples.jsonl), read by the pipeline's own "
            "report-only slice loader with --defect-download's licences"
        ),
    )
    parser.add_argument(
        "--score-dtype", choices=["fp32", "bf16"], default="fp32",
        help=(
            "the dtype a --score-checkpoint tower is scored in. fp32 by default: torch bf16 "
            "on MPS drifts ~0.8 nats on letter logits (GAP-TORCH-MPS-BF16-LETTER-LOGITS-DRIFT)"
        ),
    )
    parser.add_argument(
        "--needle", action="store_true",
        help=(
            "with --score-val: also score needle_hunk_recall on qd_train.needle's generated "
            "~8K-token multi-hunk suite, under the contract approved on 2026-09-30 (min recall "
            f"{NEEDLE_MIN_RECALL} on the worst depth bucket, {NEEDLE_CASES_PER_DEPTH} cases "
            "per depth, hit = the hunk of the predicted start line). Needs --real-backbone"
        ),
    )
    parser.add_argument(
        "--ood", action="store_true",
        help=(
            "with --score-val: also score ood_abstain on qd_train.ood's generated suite under "
            "the contract approved on 2026-09-30 (OOD abstain Wilson lower >= "
            f"{OOD_MIN_ABSTAIN}, in-distribution Wilson upper <= "
            f"{OOD_MAX_IN_DISTRIBUTION_ABSTAIN}). Needs --real-backbone and --ood-general-record"
        ),
    )
    parser.add_argument(
        "--ood-general-record", type=Path, default=None,
        help=(
            "the general fetch record the OOD prose is drawn from (its val split only). "
            "Separate from --general-record, which changes which rows the val set is "
            "relabelled from"
        ),
    )
    parser.add_argument(
        "--needle-predictions-out", type=Path, default=None,
        help=(
            "internal: run as the --needle worker. Decode the needle suite with the "
            "--score-checkpoint weights, write its predictions here, record nothing. The "
            "scoring process starts this itself, in a fresh process (see run_needle_worker)"
        ),
    )
    parser.add_argument(
        "--needle-handoff", type=Path, default=None,
        help=(
            "internal, with --needle-predictions-out: the needle suite the scoring process "
            "built, which the worker decodes instead of rebuilding the corpus and the suite "
            "(see write_needle_handoff)"
        ),
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZER_RECIPES),
        default="bf16",
        help=(
            "optimizer recipe under --real-backbone. 'bf16' (default) is what "
            "torch.optim.AdamW builds and what every existing ledger row used; its "
            "exp_avg_sq stops moving after 383 steps. 'master' keeps fp32 master weights "
            "and fp32 moments at 20 B/param against 8. Recorded in the recipe, so two runs "
            "differing in it are not comparable."
        ),
    )
    parser.add_argument(
        "--real-backbone",
        type=Path,
        help=(
            "local Qwen3.5-2B-Base snapshot directory. Without it the backbone is the "
            "randomly-initialised stand-in block, which is a statement about the loop and "
            "the data rather than about any model."
        ),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "cuda", "mps"],
        help=(
            "run only these devices instead of auto-detecting. The real tower on cpu is "
            "hours per pass and causal-conv1d's kernel refuses cpu tensors outright, so on "
            "a CUDA box with --real-backbone this is normally just: --devices cuda"
        ),
    )
    parser.add_argument(
        "--instance",
        help=(
            "what machine this is, as a price list names it -- e.g. 'lambda-1xGH200'. "
            "REQUIRED with --devices cuda and refused at argv time without it: a cuda "
            "device is rented by the hour, and the zero-rate default it used to get made "
            "requires_human_approval False for any cap"
        ),
    )
    parser.add_argument(
        "--usd-per-hour",
        type=float,
        help="the WHOLE instance's rate. Required with --devices cuda",
    )
    parser.add_argument(
        "--usd-per-gpu-hour",
        type=float,
        help=(
            "the per-GPU column of the same price list. Required by CostEstimate above one "
            "GPU, where the two columns differ by exactly n_gpus and reading the wrong one "
            "under-reports the run by that factor -- DESIGN-4's own error"
        ),
    )
    parser.add_argument(
        "--approved-by",
        default="",
        help=(
            "who said yes. RunControl refuses to start a run whose capped cost needs a "
            "human and has none: rule 4, multi-GPU always and single-GPU at $20"
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default=DEFAULT_ATTN_IMPLEMENTATION,
        help=(
            f"attention kernel for the real tower; default {DEFAULT_ATTN_IMPLEMENTATION!r}. "
            "transformers validates the name and refuses one it does not implement. It is "
            "in the recipe, so two runs on different kernels are two protocols rather than "
            "one row that contradicts itself"
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "run under torch.use_deterministic_algorithms(True). MEASURED on a GH200, "
            "eight repeats of one configuration at one seed, 128 steps: without it the "
            "final span loss ranged 0.000000 to 1.505752 with 3 of 8 over the bar; with "
            "it all eight came back bit-identical, 0.000078 letter and 0.000000 span to "
            # argparse %%-formats help strings, so a literal percent has to be doubled.
            "every digit. The price is 131.3s against 111.4s, 18%% wall clock. Not the "
            "default only because torch RAISES where an op has no deterministic "
            "implementation, which would make this tool unrunnable on a model that has "
            "one -- every op in THIS model has one, so a full train should pass it"
        ),
    )
    # --- ported pieces (RSI-Jev MIT @8f34a4f, decider Apache-2.0 @23579f7). All off by
    # default; each lands in the recipe only when on.
    parser.add_argument(
        "--fused-adamw", action="store_true",
        help=(
            "build the fp32-master optimizer's inner AdamW as torch's fused kernel: one launch "
            "per step and no full-size fp32 temporaries (GH200 2026-10-01: optimizer step "
            "70 -> 40 ms and peak -7 GiB at 4 x 8,441). Different rounding from the default "
            "foreach AdamW, so a numerics change: in the recipe as optimizer_fused. Needs "
            "--real-backbone and --optimizer master"
        ),
    )
    parser.add_argument(
        "--train-attention-mask", choices=TRAIN_ATTENTION_MASKS, default="padding",
        help=(
            "what the TRAINING forward attends with. 'padding' (default, every row so far): "
            "the mask from each row's length. 'none': is_causal only, on SDPA's flash "
            "backend alone, raising rather than falling back (GH200 2026-10-01 probe at 4 x "
            "8,441: 15,280 -> 25,703 pos/s). Batches are right-padded, so no real position "
            "reads a pad either way, but the kernel changes: Tier B, in the recipe as "
            "train_attention_mask. Scoring always masks. Needs --real-backbone; refused with "
            "--replay-shards, whose KL forward keeps the mask"
        ),
    )
    parser.add_argument(
        "--checkpoint-skip-layers", type=int, default=0,
        help=(
            "selective activation checkpointing: N decoder layers, evenly spaced, run without "
            "it and keep their activations, saving their recompute. 0 (default) checkpoints "
            "every layer, as every earlier run did. Read back per layer and priced by the "
            "device budget; in the recipe only when non-zero. Needs --real-backbone"
        ),
    )
    parser.add_argument(
        "--lower-layers-n", type=int, default=0,
        help=(
            "layer-wise lr (RSI-Jev fit.py): train decoder layers 0..N-1 at "
            "--lower-layers-lr-scale times the schedule. 0 (default) is off. RSI's value is "
            f"{RSI_LOWER_LAYERS_N}. Needs --real-backbone"
        ),
    )
    parser.add_argument(
        "--lower-layers-lr-scale", type=float, default=None,
        help=f"with --lower-layers-n; default {RSI_LOWER_LR_SCALE} (RSI's)",
    )
    parser.add_argument(
        "--beta2", type=float, default=None,
        help=(
            f"AdamW beta2; default {DEFAULT_BETA2} (unchanged). decider trains at 0.95. The "
            "bf16 recipe's moment-fidelity check is asked at this value: bf16 at 0.95 "
            "settles 1.95%% low after 64 steps and is refused past that; --optimizer "
            "master is not affected"
        ),
    )
    parser.add_argument(
        "--option-permutation-seed", type=int, default=None,
        help=(
            "re-permute every choice row's options per pass (epoch arm only), keeping the "
            "gold on its option, seeded by this. Needs --epoch and a tokenizer.json "
            "(--tokenizer-json, or the --real-backbone snapshot's)"
        ),
    )
    parser.add_argument(
        "--tokenizer-json", type=Path, default=None,
        help="tokenizer.json the shards were tokenized with; default: the snapshot's",
    )
    parser.add_argument(
        "--replay-shards", type=Path, default=None,
        help=(
            "a shard set (same remap as the train set) to replay with prior_kl toward the "
            "base's own cached answers, epoch arm only. Needs --replay-attestation, "
            "--replay-cache, --replay-weight and --epoch"
        ),
    )
    parser.add_argument(
        "--replay-attestation", type=Path, default=None,
        help="the clean attestation tools/replay_decontam.py wrote for --replay-shards",
    )
    parser.add_argument(
        "--replay-cache", type=Path, default=None,
        help="the base-logit cache: read if present (key-checked), else built once and written",
    )
    parser.add_argument("--replay-weight", type=float, default=None,
                        help="weight on prior_kl; no default -- say it")
    parser.add_argument(
        "--replay-every", type=int, default=DEFAULT_REPLAY_EVERY,
        help=f"one replay micro-batch per N training ones; default {DEFAULT_REPLAY_EVERY}",
    )
    parser.add_argument(
        "--needle-control", type=str, default=None, metavar="TOKENS,TOKENS,...",
        help=(
            "with --score-checkpoint --needle: score the needle suite at each of these target "
            "lengths (e.g. 1024,2048,4096,8192) under the gate's hit rule, on a quick "
            "diagnostic row of its own -- never the needle_hunk_recall gate, never the gate "
            "row. The 8192 arm must hash to the gate suite"
        ),
    )
    parser.add_argument(
        "--suite-verdicts-out", type=Path, default=None,
        help=(
            "with --needle and/or --ood: write every suite case's raw verdict as one JSONL "
            "line (the needle's pointer and hunk, the OOD case's two passes and letter "
            "distributions), atomically; refused if the file exists"
        ),
    )
    parser.add_argument(
        "--suite-logits", action="store_true",
        help=(
            "with --score-checkpoint and --suite-verdicts-out: each needle line also carries "
            "the pointer's start_logits/end_logits over the runtime's rows, the abstention "
            "at noul_row (an ensemble's are its mean log-probabilities). OOD lines carry "
            "both passes' letter rows, noul included, with or without it"
        ),
    )
    parser.add_argument(
        "--verdicts-out", type=Path, default=None,
        help=(
            "with --score-val: write every val verdict as one JSONL line (eval_row_id, seed, "
            "row_id, slot_name, kind, correct, top, gold_row), atomically; refused if the "
            "file exists"
        ),
    )
    parser.add_argument(
        "--shuffled-label", default=None, metavar="EVAL_ROW_ID",
        help=(
            "train the shuffled-label CONTROL for this --score-val eval row (an id or unique "
            f"prefix, read from --ledger and supplemented there): the epoch arm on the same "
            f"shard set and recipe as that row's model, with the {DEFECT_FAMILY_ID} choice "
            "golds permuted within option-count groups at the run's seed and the span channel "
            "at weight 0, then scored on its val choice rows against the real golds. Writes "
            "the ft row (its own family) and one eval row carrying controls.shuffled_label "
            "and recipe.eval_row_id, in the target's seed family. Needs --epoch, "
            "--no-memorise, --score-val and one --seeds/--devices entry each"
        ),
    )
    parser.add_argument("--probe", help=argparse.SUPPRESS)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    seeds_given = args.seeds is not None
    if not seeds_given:
        args.seeds = list(DEFAULT_SEEDS)
    # One --score-checkpoint is a Path and one --ft-row-id a str, as they always were;
    # several are a logit ensemble's towers and their ft rows, kept as lists.
    if args.score_checkpoint is not None and len(args.score_checkpoint) == 1:
        args.score_checkpoint = args.score_checkpoint[0]
    if args.ft_row_id is not None and len(args.ft_row_id) == 1:
        args.ft_row_id = args.ft_row_id[0]
    _check_piece_flags(args)

    # Resolve the sentinels against the backbone that was actually chosen. --hidden and
    # --heads are REFUSED rather than ignored under --real-backbone: the real tower's width
    # and head count come from its config.json, so accepting them would record a number that
    # determined nothing -- and `recipe` feeds `recipe_hash`.
    if args.real_backbone is not None:
        cannot_apply = [
            flag
            for flag, value in (("--hidden", args.hidden), ("--heads", args.heads))
            if value is not None
        ]
        if cannot_apply:
            raise SystemExit(
                f"{', '.join(cannot_apply)} cannot apply under --real-backbone: the real "
                "tower's hidden size and head count come from its own config.json. Accepting "
                "them would put a number in the ledger recipe that determined nothing, and "
                "recipe feeds recipe_hash -- two identical runs would get different protocol "
                "hashes. Drop them, or drop --real-backbone."
            )
        args.lr = REAL_BACKBONE_LR if args.lr is None else args.lr
    else:
        if args.optimizer != "bf16":
            raise SystemExit(
                f"--optimizer {args.optimizer} cannot apply without --real-backbone: the "
                "stand-in is a randomly-initialised 128-wide block trained for a few "
                "hundred steps, so the recipe it uses determines nothing about any model. "
                "Accepting it would put a value in the ledger recipe that did not affect "
                "the run, and recipe feeds recipe_hash."
            )
        args.hidden = STANDIN_HIDDEN if args.hidden is None else args.hidden
        args.heads = STANDIN_HEADS if args.heads is None else args.heads
        args.lr = STANDIN_LR if args.lr is None else args.lr

    # Checkpointing is refused on the STAND-IN branch, and that is not a limitation being
    # worked around -- it is the whole point. `RealFtStep.load_state` raises by design
    # ("a checkpoint it cannot restore would be a silent lie"), so a stand-in run that
    # wrote one would produce a file that looks like a resume point and is not. Only
    # `QwenDecisionStep` carries the optimizer state a resume needs.
    if (args.checkpoint_every or args.checkpoint_dir) and args.real_backbone is None:
        raise SystemExit(
            "--checkpoint-every/--checkpoint-dir need --real-backbone: the stand-in step "
            "does not implement load_state, so a checkpoint written from it could never be "
            "resumed. Writing one anyway would put a file on disk that looks like a resume "
            "point and is not, which is the failure RealFtStep.load_state raises to prevent."
        )
    if args.checkpoint_every and args.checkpoint_dir is None:
        raise SystemExit(
            f"--checkpoint-every {args.checkpoint_every} was given without "
            "--checkpoint-dir, so there is nowhere to write. A run that believes it is "
            "checkpointing and is not is worse than one that knows it is not."
        )
    if args.checkpoint_dir is not None and not args.checkpoint_every:
        raise SystemExit(
            "--checkpoint-dir was given without --checkpoint-every, so the loop would never "
            "call on_checkpoint and the directory would stay empty. Pass both, or neither."
        )
    if args.checkpoint_every < 0:
        raise SystemExit(
            f"--checkpoint-every must not be negative, got {args.checkpoint_every}"
        )
    resume_cell: tuple[str, int, str] | None = None
    if args.resume_from is not None:
        if args.real_backbone is None:
            raise SystemExit(
                "--resume-from needs --real-backbone: the stand-in step cannot restore a "
                "checkpoint, and pretending to would report a resumed run that started "
                "from scratch."
            )
        if not args.resume_from.is_file():
            raise SystemExit(f"--resume-from {args.resume_from} does not exist")
        # Routed against the PLAN, at argv time. Handing one checkpoint to every cell of
        # the (tag x seed x device) product and letting the trainer sort it out resumes one
        # cell and aborts the sweep at the next -- after a 24-second tower load, with a
        # traceback instead of an answer. All three of these are decidable from argv.
        try:
            resume_tag, resume_seed, resume_device = _resume_arm(args.resume_from)
        except ValueError as exc:
            raise SystemExit(f"--resume-from {exc}") from exc
        planned_devices = list(args.devices) if args.devices else list(ARM_DEVICES)
        if resume_device not in planned_devices:
            raise SystemExit(
                f"--resume-from is a checkpoint from {resume_device} and this run's "
                f"--devices is {planned_devices}. A resume onto other hardware passes every "
                "check train_ft has -- same seed, same schedule, same batch order -- and "
                "reproduces nothing."
            )
        if resume_seed not in list(args.seeds):
            raise SystemExit(
                f"--resume-from was taken at seed {resume_seed} and this run's --seeds is "
                f"{list(args.seeds)}. train_ft refuses a seed mismatch, so every cell would "
                "abort; refusing here costs no tower load."
            )
        if resume_tag == "epoch" and not args.epoch:
            raise SystemExit(
                "--resume-from is an 'epoch' checkpoint but --epoch was not passed, so this "
                "run has no arm to resume it into."
            )
        resume_cell = (resume_tag, resume_seed, resume_device)
    # Refused here rather than inside the step, which is constructed after the shard set has
    # been read, inventoried and batched -- and on the real backbone, after a 24-second
    # tower load. The verdict was decidable from argv.
    if not args.span_weight > 0.0:
        raise SystemExit(
            f"--span-weight must be positive, got {args.span_weight}; zero would train the "
            "span head on nothing while its loss still appeared in the log"
        )
    _check_shuffled_label_flags(args, raw_argv)
    if args.needle_control is not None:
        args.needle_control = parse_needle_control(args.needle_control)
        if not (args.needle and args.score_checkpoint is not None):
            raise SystemExit(
                "--needle-control scores a saved checkpoint against the gate suite's digest: "
                "it needs --score-checkpoint and --needle"
            )
        clashing = [
            flag for flag, given in (
                ("--ood", args.ood), ("--verdicts-out", args.verdicts_out is not None),
                ("--needle-predictions-out", args.needle_predictions_out is not None),
            ) if given
        ]
        if clashing:
            raise SystemExit(
                f"--needle-control writes its own diagnostic row and decodes no val row or "
                f"gate; {', '.join(clashing)} would record nothing"
            )
    score_plan: ScorePlan | None = None
    if args.score_plan is not None:
        score_plan = _check_score_plan_flags(args, seeds_given=seeds_given)
    elif args.score_checkpoint is not None:
        _check_score_checkpoint_flags(args)
    elif args.ft_ledger is not None or args.ft_row_id is not None:
        raise SystemExit(
            "--ft-ledger/--ft-row-id only mean something with --score-checkpoint (or "
            "--score-plan)"
        )
    if args.composed_slice is not None and score_plan is None:
        raise SystemExit("--composed-slice is read only by a --score-plan 'composed' pass")
    _check_suite_logits_flags(args)
    if (args.ood_general_record is not None) != args.ood:
        raise SystemExit("--ood and --ood-general-record are given together or not at all")
    if args.ood and not (args.score_val and args.real_backbone is not None):
        raise SystemExit(
            "--ood scores the model the val pass scores and encodes with the real "
            "tokenizer: it needs --score-val and --real-backbone"
        )
    if args.needle_predictions_out is not None and not (
        args.needle and args.score_checkpoint is not None
    ):
        raise SystemExit("--needle-predictions-out is the --needle worker of --score-checkpoint")
    if (args.needle_handoff is None) != (args.needle_predictions_out is None):
        raise SystemExit(
            "--needle-handoff and --needle-predictions-out are the needle worker's two ends and "
            "are given together: the worker decodes only a suite the scoring process handed it"
        )
    if args.needle and not (args.score_val and args.real_backbone is not None):
        raise SystemExit(
            "--needle scores the model the val pass scores and encodes with the real "
            "tokenizer: it needs --score-val and --real-backbone"
        )
    if (args.score_val and not args.epoch and args.score_checkpoint is None
            and args.score_plan is None):
        raise SystemExit(
            "--score-val scores the epoch arm's model and --epoch was not passed, so there "
            "would be nothing to score. The memorisation arm trains on a subset of the "
            "train split and is not the model a val number is about."
        )
    if args.no_memorise and not args.epoch:
        raise SystemExit(
            "--no-memorise skips the memorisation arm, and without --epoch there is no "
            "other arm: the run would train nothing"
        )
    if args.no_memorise and resume_cell is not None and resume_cell[0] == "memorise":
        raise SystemExit(
            "--resume-from is a 'memorise' checkpoint but --no-memorise was passed, so this "
            "run has no arm to resume it into"
        )
    if not (math.isfinite(args.wall_clock_cap_s)
            and 0.0 < args.wall_clock_cap_s <= MAX_WALL_CLOCK_CAP_S):
        raise SystemExit(
            f"--wall-clock-cap-s {args.wall_clock_cap_s!r} is outside (0, "
            f"{MAX_WALL_CLOCK_CAP_S:g}]. The campaign approval is up to 72 h, but MAX_CAP_S "
            f"({MAX_CAP_S / 3600:g} h) is the program's own cap and is read-only (rule 2): "
            "report that a cap was hit, do not raise it"
        )
    # Every run of this tool writes ledger rows, so the corpus revision is always a full sha:
    # a recipe that says "HEAD" names a corpus that no longer exists after the next commit.
    try:
        require_full_sha(args.rev)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.general_record is None and (
        args.general_max_rows is not None or args.replay_partition
    ):
        raise SystemExit(
            "--general-max-rows/--replay-partition without --general-record read nothing: "
            "the replay slice is drawn from the general families' training rows"
        )
    if not args.repo_history and args.commitpackft is None:
        if args.max_pairs is not None:
            raise SystemExit(
                "--max-pairs bounds the repository-history rows and the --commitpackft "
                "sample; under --no-repo-history without --commitpackft the rebuild reads "
                "neither, so it would determine nothing. Drop it."
            )
        # Unread by the rebuild in this mode; a number rather than None because ft_splits
        # and the replay attestation take an int.
        args.max_pairs = 400
    elif args.max_pairs is None:
        args.max_pairs = 400

    if args.probe:
        return _run_probe(args.probe)
    if args.out is None:
        parser.error("--out is required")
    if not 1 <= args.passes <= MAX_PASSES:
        parser.error(f"--passes must be in [1, {MAX_PASSES}]")
    if not 1 <= len(args.seeds) <= MAX_SEEDS:
        parser.error(f"--seeds must name between 1 and {MAX_SEEDS} seeds")
    # Decided from argv, before a 24-second tower load and before any GPU time is spent.
    # `CostEstimate.for_device` refuses the same case, but it is reached per-arm inside
    # `_train`, which is after the work has started on a rented box.
    if "cuda" in (args.devices or []) and (
        args.instance is None or args.usd_per_hour is None
    ):
        # SystemExit rather than parser.error: the latter exits 2 with the message on
        # stderr only, so the reason is not in the exception and a caller driving main()
        # gets a bare 2. Every other refusal in this tool raises with its reason attached.
        raise SystemExit(
            "--devices cuda needs --instance and --usd-per-hour. A cuda device is hardware "
            "rented by the hour; the zero-rate, zero-GPU default that stood in for a price "
            "makes requires_human_approval False for ANY cap and skips the per-GPU column "
            "check, so the two values that look like harmless defaults are the two that "
            "turn rule 4 off. Example: --instance lambda-1xGH200 --usd-per-hour 1.49"
        )
    # Rule 4 at argv time, priced from the cap this run will actually carry. RunControl makes
    # the same refusal, but per arm inside `_train` -- after the shard set is read and, on
    # the real backbone, after the tower has loaded on a box billed by the hour.
    for device in args.devices or []:
        estimate = _cost(
            device=device, n_gpus=n_gpus_for_device(device), usd_per_hour=args.usd_per_hour,
            usd_per_gpu_hour=args.usd_per_gpu_hour, instance=args.instance,
            cap_s=args.wall_clock_cap_s,
        )
        if estimate.requires_human_approval and not args.approved_by.strip():
            raise SystemExit(
                f"rule 4: {estimate.approval_line()}. Each arm of each seed carries this cap, "
                "so each needs a human yes: pass --approved-by '<who said yes>' once a human "
                "has. An agent cannot supply it on a human's behalf."
            )

    if args.deterministic:
        # Before the tower loads, so nothing has run on a nondeterministic kernel by the
        # time this takes effect. torch raises rather than falling back, which is the
        # property that makes a completed run evidence: every op this model uses had a
        # deterministic implementation, rather than "we asked and something quietly said
        # no".
        torch.use_deterministic_algorithms(True)

    config = DataConfig()
    shard_dir = args.out / "shards" / "train"
    # Resolved once, and the same string is used for both halves below. --rev is what this
    # tool RECONSTRUCTS its labels at, so it is exactly the revision the shard set has to
    # have been built from -- and the two would silently disagree if one of them peeled a
    # symbolic name and the other did not.
    rev = resolve_rev(REPO, args.rev)
    if rev != args.rev:
        print(f"corpus revision: {args.rev} -> {rev}")
    # Passing it turns a silent mismatch into a refusal: the 321-against-341 drift was
    # caught only because it moved the row count, and a revision that changes which rows
    # exist without changing how many would have paired every label with the wrong
    # sequence and said nothing.
    reader = ShardReader(shard_dir, config=config, repo_root=args.out, expect_rev=rev)
    if args.needle_predictions_out is not None:
        # The --needle worker decodes the suite its scoring process handed it; everything
        # below builds what that process already built.
        return needle_worker_main(args, reader=reader, config=config, rev=rev)

    check_defect_source(args.out, defect_class=args.defect_class)
    # Before the rebuild, which on the full defect corpus is minutes of work: what the
    # manifest says about the corpus decides both whether this rebuild can match it and,
    # below, which of this run's rows are quick.
    corpus = corpus_facts(
        args.out, data_snapshot_hash=reader.header.data_snapshot_hash,
        repo_history=args.repo_history, commitpackft=args.commitpackft,
        general_datasets=general_record_datasets(args.general_record),
        replay_partition=args.replay_partition,
    )
    train_rows, val_rows = ft_split_rows(
        commitpackft=args.commitpackft, max_pairs=args.max_pairs, rev=rev, config=config,
        defect_class=args.defect_class, defect_download=args.defect_download,
        defect_max_rows=args.defect_max_rows, repo_history=args.repo_history,
        general_record=args.general_record, general_max_rows=args.general_max_rows,
        replay_partition=args.replay_partition, defect_noul=args.defect_noul,
    )
    require_index = args.defect_class is not None or args.general_record is not None
    # --score-checkpoint trains nothing, and the train split's labels, inventory,
    # contradictions and epoch plan feed no row it writes: on the phase-4 set they were ~37 s
    # of a 143 s Mac prelude (one unprofiled run, 2026-10-01), paid with the GPU idle; see
    # AUDIT/perf-ft-run-prep-2026-10-01.md for the interleaved A/B. What they did for scoring
    # was confirm the tokenizer's letter ids against the train golds; the val golds confirm
    # them too (open_val_set's merge_letter_ids), and where some offered letter is a gold
    # nowhere in val, the train relabel still runs (below) -- so it is skipped only with a
    # tokenizer.json to read every letter from, and only when nothing rests on it.
    train_side: TrainRelabel | None = None
    if args.score_checkpoint is None or args.tokenizer_json is None:
        train_side = relabel_train(
            reader, train_rows, config=config, require_index=require_index,
            tokenizer_json=args.tokenizer_json,
        )
        letter_id = train_side.letter_id
    else:
        letter_id = vocab_letter_ids(reader, tokenizer_json=args.tokenizer_json, letter_id={})
    # Opened before any tower loads, so every way it could misscore is refused on argv's
    # time rather than after an epoch has been paid for.
    val_set: ValSet | None = None
    second_pass = SecondPass([], {}, {}, not_run="no val set: --score-val was not given")
    needle_suite = NeedleSuite([], [], {}, [], not_run="no val set: --score-val was not given")
    ood_suite = OodSuite([], None, None, not_run="no val set: --score-val was not given")
    if args.score_val:
        val_set = open_val_set(
            args.out, config=config, rev=rev,
            rows=list(val_rows),
            train=reader, letter_id=letter_id,
            require_index=require_index,
        )
        print(
            f"val set: {len(val_set.reader)} sequences in {len(val_set.plan)} batches, "
            f"remap {val_set.reader.header.remap_hash[:16]} (the train set's)"
        )
        second_pass = prepare_second_pass(
            val_set, reader=val_set.reader, tokenizer_json=args.tokenizer_json,
            seed=config.seed,
        )
        print(
            f"permutation second pass: {len(second_pass.perms)} choice rows deranged in "
            f"{len(second_pass.batches)} batches (seed {config.seed})"
            if second_pass.not_run is None else
            f"permutation second pass: not run -- {second_pass.not_run}"
        )
        needle_suite = prepare_needle(val_set.reader, config=config, enabled=args.needle)
        print(
            f"needle suite: {len(needle_suite.cases)} cases, real tokens "
            f"{min(needle_suite.token_lengths)}..{max(needle_suite.token_lengths)}"
            if needle_suite.not_run is None else
            f"needle suite: not run -- {needle_suite.not_run}"
        )
        ood_suite = prepare_ood(
            val_set.reader, config=config, rev=rev, letter_id=val_set.letter_id,
            tokenizer_json=args.tokenizer_json, general_record=args.ood_general_record,
            enabled=args.ood,
        )
        print(
            f"OOD suite: {len(ood_suite.cases)} cases, "
            f"{0 if ood_suite.second_pass is None else len(ood_suite.second_pass.perms)} "
            "deranged for the second pass"
            if ood_suite.not_run is None else f"OOD suite: not run -- {ood_suite.not_run}"
        )
    train_skipped: str | None = None
    if train_side is None:
        if val_set is None:  # --score-checkpoint is refused at argv time without --score-val
            raise SystemExit("--score-checkpoint needs --score-val's val set")
        unconfirmed = letters_no_val_gold_confirms(val_set, ood_suite)
        if unconfirmed:
            print(
                f"train relabel: run after all -- val or OOD rows offer letter(s) {unconfirmed} "
                "that no val row has as its gold, so their ids are confirmed against the "
                "train golds, as before --score-checkpoint skipped them"
            )
            train_side = relabel_train(
                reader, train_rows, config=config, require_index=require_index,
                tokenizer_json=args.tokenizer_json,
            )
            if train_side.letter_id != letter_id:
                raise SystemExit(
                    f"the train relabel reads letter ids {train_side.letter_id} and the "
                    f"vocabulary {letter_id}; one of them is not this shard set's"
                )
        else:
            train_skipped = (
                "--score-checkpoint trains nothing and no row it writes reads the train "
                "split's labels, inventory or contradictions; every letter a val or OOD row "
                f"offers was confirmed against the val golds, and the ids come from "
                f"{args.tokenizer_json}"
            )
    batch_tokens, recipe_batch_tokens = _resolve_batch_tokens(
        args.batch_tokens, widest=int(max(reader.header.buckets))
    )
    batch_info = (
        None if args.score_checkpoint is not None
        else _batch_inventory(reader, batch_tokens=batch_tokens, seed=config.seed)
    )

    print(f"shard set: {shard_dir}")
    if train_side is None:
        print(f"  train relabel, per-kind inventory and contradictions: NOT RUN -- {train_skipped}")
    else:
        inventory = train_side.inventory
        print(
            f"  rows in -> out: {inventory['coverage']['n_total']} -> "  # type: ignore[index]
            f"{inventory['coverage']['n']}"  # type: ignore[index]
            f"  ({inventory['excluded_rows']} excluded)  tokens {inventory['tokens']}"
            f"  vocab {inventory['vocab_size']}"
        )
        for name, entry in inventory["per_kind"].items():  # type: ignore[union-attr]
            print(
                f"  {name:7s} n={entry['n']:4d}  len {entry['len_min']}..{entry['len_max']} "
                f"(mean {entry['len_mean']})  gold is noul: {entry['gold_is_noul']}"
                + (
                    f"  pointer-abstaining {entry['pointer_abstaining']}/{entry['n']}, "
                    f"candidates {entry['candidates_total']}"
                    if name == "span"
                    else f"  rendered rows {entry['row_widths']}"
                )
            )
        print(
            f"  letter rows whose gold is noul: "
            f"{inventory['letter_rows_whose_gold_is_noul']} of {inventory['letter_rows']}"
        )
        contra = inventory["contradictions"]
        print(
            f"  prompt-identical rows: {contra['rows_sharing_a_prefix']} of {contra['sequences']};"  # type: ignore[index]
            f" of those, {contra['contradicting_rows']} rows in "  # type: ignore[index]
            f"{contra['contradicting_groups']} groups carry DIFFERENT golds "  # type: ignore[index]
            f"({contra['per_kind']})"  # type: ignore[index]
        )
    print(f"  letters -> post-remap ids: {letter_id}")
    # Read off the reader, which checked them when it opened the set, whichever branch ran.
    print(f"  padding waste: {reader.padding_waste().to_json()['detail']}")
    print(f"  span_check: {json.dumps(reader.span_check.to_json())[:180]}")
    print(
        "  remap beside the shards: "
        f"{json.dumps(reader.checks['shard_remap_matches_header'].to_json())[:180]}"
    )
    if batch_info is None:
        print(
            f"  epoch at batch_tokens={batch_tokens}: NOT RUN -- --score-checkpoint trains "
            "nothing, so no epoch is planned and no device is probed"
        )
    else:
        print(
            f"  epoch at batch_tokens={batch_tokens}: {batch_info['batches']} batches, "
            f"{batch_info['batches_with_a_span_row']} carry a span row, "
            f"{batch_info['batches_span_only']} have no letter row; letter-channel chunks "
            f"{batch_info['letter_channel_live_chunks']} live of "
            f"{batch_info['letter_channel_total_chunks']}"
        )

    if args.devices:
        devices = list(args.devices)
        print(f"devices: {devices} (from --devices; auto-detection skipped)")
    else:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        else:
            print("cuda: NOT RUN -- torch.cuda.is_available() is False on this host")
        if torch.backends.mps.is_available():
            devices.append("mps")
        else:
            print("mps: NOT RUN -- torch.backends.mps.is_available() is False on this host")

    # --score-plan, like --score-checkpoint below, trains nothing: one kind after another over
    # the suites built above, each written as it finishes.
    if score_plan is not None:
        if val_set is None:  # pragma: no cover - _check_score_plan_flags needs --score-val
            raise SystemExit("--score-plan needs --score-val's val set")
        composed: ComposedSlice | None = None
        if score_plan.wants("composed"):
            import real_tokenizer_pipeline as pipeline

            assert args.composed_slice is not None  # _check_score_plan_flags refused otherwise
            composed = open_composed_slice(
                args.composed_slice, corpus_dir=args.composed_slice_corpus,
                download_root=(
                    pipeline.DEFAULT_DEFECT_DOWNLOAD if args.defect_download is None
                    else args.defect_download
                ),
                repo_root=REPO, config=config, train=reader, letter_id=val_set.letter_id,
            )
        for kind_name, what, row_id in run_score_plan(
            args, score_plan, reader=reader, val=val_set, device=devices[0],
            ledger=Ledger(args.ledger),
            reasons_for=lambda tag, device, termination=None: quick_reasons(
                tag=tag, device=device, real_backbone=True, corpus=corpus,
                termination=termination,
            ),
            second_pass=second_pass, needle_suite=needle_suite, ood_suite=ood_suite,
            suite_seed=config.seed, composed=composed,
        ):
            print(f"score plan: {kind_name} {what} row {row_id}")
        return 0

    # --score-checkpoint trains nothing, so nothing below here runs: the probes, the plans and
    # the arms are all about training. Its eval row is written by `_record_score`, the same
    # writer an epoch arm's --score-val uses.
    if args.score_checkpoint is not None:
        if val_set is None:
            raise SystemExit("--score-checkpoint needs --score-val's val set")
        if args.needle_control is not None:
            control_row_id, control_lines, control_seed = run_needle_control(
                args, reader=reader, val=val_set, device=devices[0], ledger=Ledger(args.ledger),
                config=config, gate_suite=needle_suite,
                reasons_for=lambda tag, device, termination=None: quick_reasons(
                    tag=tag, device=device, real_backbone=True, corpus=corpus,
                    termination=termination,
                ),
            )
            # The row's own protocol seed, as for the score row below: never --seeds[0],
            # which for an average is one of its inputs.
            if args.suite_verdicts_out is not None:
                write_suite_verdicts_jsonl(
                    args.suite_verdicts_out,
                    [{"eval_row_id": control_row_id, "seed": control_seed,
                      "gate": "needle_hunk_recall.control", **v} for v in control_lines],
                )
                print(f"suite verdicts: {len(control_lines)} lines -> {args.suite_verdicts_out}")
            print(f"needle control row {control_row_id}")
            return 0
        worker_decoded = (
            run_needle_worker(
                raw_argv, needle_suite, letter_id=val_set.letter_id,
                eval_widths=suite_widths(needle_suite, ood_suite), train=reader, val=val_set,
            )
            if needle_suite.not_run is None else None
        )

        def eval_reasons(tag: str, device: str, termination: str | None = None) -> list[str]:
            return quick_reasons(
                tag=tag, device=device, real_backbone=True, corpus=corpus,
                termination=termination,
            )

        score_row_id, scored, permutation, suite_gates, row_seed = _score_checkpoint(
            args, reader=reader, val=val_set, device=devices[0],
            ledger=Ledger(args.ledger), reasons_for=eval_reasons, second_pass=second_pass,
            needle_suite=needle_suite, ood_suite=ood_suite, suite_seed=config.seed,
            needle_decoded=worker_decoded,
        )
        for name, state in score_states(scored, val_set.labels).items():
            print(f"  {name}: {json.dumps(state.to_json())[:300]}")
        _, ece, degenerate = calibration_states(scored)
        for name, state in (("ece", ece), ("degenerate_head", degenerate),
                            ("permutation_consistency", permutation),
                            *((g.name, g.state) for g in suite_gates)):
            print(f"  {name}: {json.dumps(state.to_json())[:300]}")
        # The row's own protocol seed: the checkpoint's for one seed's, the protocol seed
        # for an average -- never --seeds[0], which for an average is one of three.
        if args.verdicts_out is not None:
            write_verdicts_jsonl(
                args.verdicts_out,
                _verdict_lines(scored, eval_row_id=score_row_id, seed=row_seed),
            )
            print(f"verdicts -> {args.verdicts_out}")
        if args.suite_verdicts_out is not None:
            suite_lines = suite_verdict_lines(
                suite_gates, eval_row_id=score_row_id, seed=row_seed
            )
            write_suite_verdicts_jsonl(args.suite_verdicts_out, suite_lines)
            print(f"suite verdicts: {len(suite_lines)} lines -> {args.suite_verdicts_out}")
        print(f"score row {score_row_id}")
        return 0

    # Only --score-checkpoint, which returned above, skips the train relabel and the plan.
    if train_side is None or batch_info is None:
        raise RuntimeError("training reached without the train relabel or the epoch plan")
    labels, inventory = train_side.labels, train_side.inventory

    # Which device can take which bucket, measured out of process. The probe builds the
    # STAND-IN block, not the real tower, so it answers "can this device take this shape"
    # and not "does the real model fit" -- stated here because eight green lines under
    # --real-backbone otherwise read as the second.
    probe_hidden = STANDIN_HIDDEN if args.hidden is None else args.hidden
    probe_heads = STANDIN_HEADS if args.heads is None else args.heads
    print("\nfeasibility, one forward+backward per bucket width, in a subprocess:")
    if args.real_backbone is not None:
        print(
            f"  (a {probe_hidden}x{probe_heads} stand-in, NOT the real tower: this establishes "
            "the device and the shape, not that the real model fits)"
        )
    widest_per_bucket: dict[int, tuple[int, int]] = {}
    for row in batch_info["rows"]:  # type: ignore[union-attr]
        b = int(row["bucket"])
        cand = (int(row["rows"]), int(row["width"]))
        if b not in widest_per_bucket or cand[0] > widest_per_bucket[b][0]:
            widest_per_bucket[b] = cand
    probes: list[dict[str, object]] = []
    for device in devices:
        for b in sorted(widest_per_bucket):
            rows_n, width = widest_per_bucket[b]
            probe = _probe_one(device, rows_n, width, probe_hidden, probe_heads)
            probe["bucket"] = b
            probes.append(probe)
            print(
                f"  {device} bucket {b} ({rows_n} x {width}): "
                + ("ok " + f"{probe.get('wall_s')}s" if probe["ok"] else f"REFUSED {probe['why']}")
            )

    plan_all = list(reader.batches(batch_tokens=batch_tokens, seed=config.seed, epoch=0))
    labels_by_batch_all = _labels_by_batch(
        reader, plan_all, labels, config=config, batch_tokens=batch_tokens
    )
    keep = [i for i, b in enumerate(plan_all) if b.tokens.shape[1] <= args.max_width]
    plan_small = [plan_all[i] for i in keep]
    labels_small = {new: labels_by_batch_all[old] for new, old in enumerate(keep)}

    kinds_in_set = {name for name in inventory["per_kind"]}  # type: ignore[union-attr]
    kinds_in_plan = {
        label.slot_kind for group in labels_small.values() for label in group
    }
    print(
        f"\narm 2 plan: every real batch at most {args.max_width} wide -> "
        f"{len(plan_small)} of {len(plan_all)} batches, "
        f"{sum(int(b.tokens.shape[0]) for b in plan_small)} of {len(reader)} sequences, "
        f"slot kinds {sorted(KIND_NAMES[k] for k in kinds_in_plan)} of {sorted(kinds_in_set)}"
    )
    if not plan_small and not args.no_memorise:
        raise SystemExit(f"no real batch is at most {args.max_width} wide")

    # The epoch arm's ported pieces, built and checked before any tower loads.
    permutation: ChoicePermutation | None = None
    epoch_alphabets: dict[int, list[tuple[str, ...] | None]] | None = None
    if args.option_permutation_seed is not None:
        permutation = _permutation_spec(
            reader, tokenizer_json=args.tokenizer_json, letter_id=letter_id,
            seed=args.option_permutation_seed,
        )
        epoch_alphabets = _alphabets(plan_all, labels_by_batch_all)
    replay_plan: ReplayPlan | None = None
    if args.replay_shards is not None:
        replay_plan = _replay_plan(
            args, config=config, train=reader,
            width=max(int(b.tokens.shape[1]) for b in plan_all),
            letter_id=letter_id, val_rows=list(val_rows), rev=rev,
        )
        print(
            f"replay: {len(replay_plan.batches)} batches from {args.replay_shards} "
            f"(attestation {replay_plan.attestation_sha256[:16]}), weight "
            f"{replay_plan.weight}, every {replay_plan.every}"
        )
    verdict_lines: list[dict[str, object]] = []
    suite_lines: list[dict[str, object]] = []


    ledger = Ledger(args.ledger)
    # Read once, before any arm runs. `train_ft` checks it against the schedule, the seed
    # and the epoch and refuses a mismatch, so a checkpoint from a different run fails at
    # the first arm rather than after the tower has loaded for the second.
    resume_checkpoint = None
    if args.resume_from is not None:
        from qd_train.run_control import Checkpoint

        resume_checkpoint = Checkpoint.read(args.resume_from)
        print(
            f"resume: {args.resume_from} at optimizer step "
            f"{resume_checkpoint.optimizer_step}, batch index "
            f"{resume_checkpoint.position.index}, seed {resume_checkpoint.seed}"
        )
    # What the backbone actually was, in the ledger's own words. Branching here and not only
    # in `notes` is the whole point: a row written for a --real-backbone run used to say it
    # ran "a randomly-initialised 128x4 single block", interpolating --hidden and --heads,
    # which that path ignores. The ledger is append-only, so a wrong claim in it is permanent.
    backbone_said = (
        f"a randomly-initialised {args.hidden}x{args.heads} single block"
        if args.real_backbone is None
        else f"the real text tower from {args.real_backbone.name}"
    )
    memorise_detail = (
        f"{len(plan_small)} of {len(plan_all)} real batches "
        f"({sum(int(b.tokens.shape[0]) for b in plan_small)} of {len(reader)} sequences) "
        f"repeated {args.passes}x against {backbone_said}"
    )

    def reasons_for(tag: str, device: str, termination: str | None = None) -> list[str]:
        """This run's rule-8 reasons for one (arm, device); see :func:`quick_reasons`."""
        return quick_reasons(
            tag=tag, device=device, real_backbone=args.real_backbone is not None,
            corpus=corpus, termination=termination, memorise_detail=memorise_detail,
        )

    report: dict[str, object] = {
        "torch": torch.__version__,
        "inventory": inventory,
        "letter_ids": letter_id,
        "batches": {k: v for k, v in batch_info.items() if k != "rows"},
        "batches_with_span": [r for r in batch_info["rows"] if r["span_rows"]],  # type: ignore[union-attr]
        "probes": probes,
        "arm2": [],
        "arm1": [],
    }
    failures: list[str] = []

    # --shuffled-label, decided before any tower loads. `plan_all` is replaced by the plan
    # carrying the permuted golds; under --no-memorise nothing else reads it after this.
    shuffled: ShuffledLabel | None = None
    if args.shuffled_label is not None:
        if val_set is None:  # pragma: no cover - _check_shuffled_label_flags needs --score-val
            raise SystemExit("--shuffled-label needs --score-val's val set")
        shuffled = prepare_shuffled_label(
            args.shuffled_label, rows=ledger.rows(), reader=reader, val=val_set, labels=labels,
            seed=int(args.seeds[0]),
            planned=planned_ft_recipe(
                args, device=devices[0], plan=plan_all, reader=reader,
                permutation=permutation, replay=replay_plan, batch_tokens=recipe_batch_tokens,
            ),
            quick=reasons_for("epoch", devices[0]),
        )
        plan_all, rewritten = apply_shuffled_golds(
            plan_all, labels_by_batch_all, shuffled.golds, letter_id
        )
        print(
            f"\nSHUFFLED-LABEL CONTROL for eval row {shuffled.eval_row.row_id} (ft row "
            f"{shuffled.ft_row.row_id}): {len(shuffled.golds.golds)} {SHUFFLED_LABEL_FAMILY} "
            f"choice golds permuted at seed {shuffled.golds.seed}, {shuffled.golds.moved} moved "
            f"({rewritten} answer tokens rewritten), permutation "
            f"{shuffled.golds.digest[:16]}; span weight {args.span_weight}"
        )

    if args.no_memorise:
        print("\narm 2: NOT RUN -- --no-memorise")
    for device in [] if args.no_memorise else devices:
        refused = [
            p for p in probes
            if p["device"] == device and not p["ok"]
            and int(p["width"]) <= args.max_width  # type: ignore[arg-type]
        ]
        if refused:
            print(f"\narm 2 on {device}: NOT RUN -- {refused[0]['why']}")
            report["arm2"].append(  # type: ignore[union-attr]
                {"device": device, "not_run": str(refused[0]["why"])}
            )
            continue
        for seed in args.seeds:
            run = _train(
                reader=reader, plan=plan_small, passes=args.passes, device=device, seed=seed,
                hidden=args.hidden, heads=args.heads, lr=args.lr,
                span_weight=args.span_weight, ledger=ledger,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_every=args.checkpoint_every,
                # Only the cell it was taken from. One checkpoint handed to every cell
                # resumes one and aborts the rest on a seed or schedule mismatch.
                resume_from=(
                    resume_checkpoint
                    if resume_cell == ("memorise", seed, device)
                    else None
                ),
                optimizer_recipe=args.optimizer,
                backbone=args.real_backbone,
                deterministic=args.deterministic,
                attn_implementation=args.attn_implementation,
                n_gpus=n_gpus_for_device(device), usd_per_hour=args.usd_per_hour,
                usd_per_gpu_hour=args.usd_per_gpu_hour, instance=args.instance,
                approved_by=args.approved_by,
                tag="memorise", quick_reasons=reasons_for("memorise", device),
                lower_layers_n=args.lower_layers_n,
                lower_lr_scale=args.lower_layers_lr_scale, beta2=args.beta2,
                cap_s=args.wall_clock_cap_s, batch_tokens=recipe_batch_tokens,
                checkpoint_skip_layers=args.checkpoint_skip_layers, fused_adamw=args.fused_adamw,
                train_attention_mask=args.train_attention_mask,
            )
            step = run.pop("_step")
            decode_at = time.monotonic()
            shipped = _decode(step, plan_small, labels_small, letter_id, noul_first=False)
            defect = _decode(step, plan_small, labels_small, letter_id, noul_first=True)
            decode_s = time.monotonic() - decode_at
            run["verdict_row_id"] = _record_verdict(
                run, ledger=ledger, reader=reader, shipped=shipped, defect=defect,
                inventory=inventory, decode_s=decode_s,
                quick_reasons=reasons_for("memorise", device, str(run["termination"])),
                resolution=resolution_state(
                    sd=args.prior_sd,
                    n_per_arm=len(args.seeds),
                    target=args.target_difference,
                    against_known_reference=True,
                ),
                batch_chunks={
                    "live": int(batch_info["letter_channel_live_chunks"]),  # type: ignore[arg-type]
                    "total": int(batch_info["letter_channel_total_chunks"]),  # type: ignore[arg-type]
                },
            )
            run["decoded"] = shipped["by_kind"]
            report["arm2"].append(run)  # type: ignore[union-attr]
            where = f"{device} seed={seed}"
            final = run["final"]
            # `_floor_state` decides both this and the ledger row, so the two cannot
            # disagree. A channel with no row in the plan is NotRun here and NotRun there.
            for channel in ("letter", "span"):
                state = _floor_state(
                    final[f"{channel}_gap"], final[f"{channel}_rows"], detail=""  # type: ignore[index]
                )
                if isinstance(state, Ran) and not state.passed:
                    failures.append(
                        f"{where}: the {channel} channel did not reach the floor this plan "
                        f"admits -- gap {final[f'{channel}_gap']}, bar {FLOOR_SLACK}, over "  # type: ignore[index]
                        f"{final[f'{channel}_rows']} row(s)"  # type: ignore[index]
                    )
            if shipped["rows_not_decoded"]:
                failures.append(
                    f"{where}: {shipped['rows_not_decoded']} row(s) offer letter(s) "
                    f"{shipped['letters_without_id']} with no known token id and were not "
                    "decoded; pass --tokenizer-json (or --real-backbone) so every offered "
                    "letter has one"
                )
            if shipped["prefix_groups_that_decoded_inconsistently"]:
                failures.append(
                    f"{where}: {shipped['prefix_groups_that_decoded_inconsistently']} "
                    "prefix-identical group(s) decoded to different verdicts, which a "
                    "deterministic model on identical input cannot do"
                )
            if not _counterfactual_holds(shipped, defect):
                failures.append(
                    f"{where}: reordering the LETTER rows moved the span abstention, so the "
                    "two arms differ in more than the thing under test"
                )
            print(
                f"{device} seed={seed}: total {run['total_first']:.4f} -> "
                f"{run['total_last']:.6f}  letter {run['letter_first']:.4f} -> "
                f"{run['letter_last']:.6f} (floor {run['letter_floor']:.6f})  span "
                f"{run['span_first']:.4f} -> {run['span_last']:.6f} (floor "
                f"{run['span_floor']:.6f})  span abstain "
                f"{shipped['span_abstaining_decoded_as_abstain']}/"
                f"{shipped['span_abstaining_rows']}  letter abstain "
                f"{shipped['letter_abstaining_decoded_as_abstain']}/"
                f"{shipped['letter_abstaining_rows']}  {run['wall_clock_s']}s  ft row "
                f"{run['ft_row_id']}  verdict row {run['verdict_row_id']}"
            )

    if args.epoch:
        for device in devices:
            refused = [p for p in probes if p["device"] == device and not p["ok"]]
            if refused:
                why = "; ".join(
                    f"bucket {p['bucket']} ({p['rows']}x{p['width']}): {p['why']}"
                    for p in refused
                )
                print(f"\narm 1 on {device}: NOT RUN -- {why}")
                report["arm1"].append({"device": device, "not_run": why})  # type: ignore[union-attr]
                continue
            for seed in args.seeds:
                run = _train(
                    reader=reader, plan=plan_all, passes=1, device=device, seed=seed,
                    hidden=args.hidden, heads=args.heads, lr=args.lr,
                    span_weight=args.span_weight, ledger=ledger,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_every=args.checkpoint_every,
                    resume_from=(
                        resume_checkpoint
                        if resume_cell == ("epoch", seed, device)
                        else None
                    ),
                    optimizer_recipe=args.optimizer,
                    backbone=args.real_backbone,
                    deterministic=args.deterministic,
                    attn_implementation=args.attn_implementation,
                    n_gpus=n_gpus_for_device(device), usd_per_hour=args.usd_per_hour,
                    usd_per_gpu_hour=args.usd_per_gpu_hour, instance=args.instance,
                    approved_by=args.approved_by,
                    tag="epoch", quick_reasons=reasons_for("epoch", device),
                    cap_s=args.wall_clock_cap_s, no_memorise=args.no_memorise,
                    batch_tokens=recipe_batch_tokens,
                    lower_layers_n=args.lower_layers_n,
                    lower_lr_scale=args.lower_layers_lr_scale, beta2=args.beta2,
                    permutation=permutation, alphabets=epoch_alphabets, replay=replay_plan,
                    eval_widths=suite_widths(needle_suite, ood_suite),
                    shuffled_label=None if shuffled is None else SHUFFLED_LABEL_RECIPE,
                    checkpoint_skip_layers=args.checkpoint_skip_layers,
                    fused_adamw=args.fused_adamw,
                    train_attention_mask=args.train_attention_mask,
                )
                step = run.pop("_step")
                if shuffled is not None and val_set is not None:
                    # The control, not a score of the recipe: no _record_score row, whose
                    # recipe would hash the shuffled model into the real model's family.
                    check_shuffled_ft_row(ledger.rows(), str(run["ft_row_id"]), shuffled)
                    decode_at = time.monotonic()
                    scored = _decode(step, val_set.plan, val_set.labels_for, val_set.letter_id)
                    decode_s = time.monotonic() - decode_at
                    run["control_row_id"] = _record_shuffled_label(
                        run, scored, ledger=ledger, reader=reader, val=val_set,
                        shuffled=shuffled, decode_s=decode_s,
                        quick_reasons=reasons_for("epoch", device, str(run["termination"])),
                    )
                    print(f"  shuffled-label control row {run['control_row_id']}")
                if val_set is not None and shuffled is None:
                    decode_at = time.monotonic()
                    scored = _decode(step, val_set.plan, val_set.labels_for, val_set.letter_id)
                    # Not ``permutation``: that local is the option permutation spec the next
                    # seed's ``_train`` is handed, and a gate there broke every seed after the
                    # first.
                    permutation_gate, val_second = score_permutation_consistency(
                        step, val_set, second_pass, scored
                    )
                    suite_gates = [
                        needle_gate(
                            score_needle(step, needle_suite, val_set.letter_id), needle_suite
                        ),
                        ood_suite_gate(
                            score_ood(step, ood_suite, scored=scored, val_second=val_second,
                                      val_second_pass=second_pass),
                            ood_suite,
                        ),
                    ]
                    decode_s = time.monotonic() - decode_at
                    run["score_row_id"] = _record_score(
                        run, scored, ledger=ledger, reader=reader, val=val_set,
                        quick_reasons=reasons_for("epoch", device, str(run["termination"])),
                        decode_s=decode_s, permutation=permutation_gate, suite_gates=suite_gates,
                    )
                    if args.verdicts_out is not None:
                        verdict_lines.extend(
                            _verdict_lines(
                                scored, eval_row_id=str(run["score_row_id"]), seed=seed
                            )
                        )
                    if args.suite_verdicts_out is not None:
                        suite_lines.extend(
                            suite_verdict_lines(
                                suite_gates, eval_row_id=str(run["score_row_id"]), seed=seed
                            )
                        )
                    for name, state in score_states(scored, val_set.labels).items():
                        print(f"  {device} seed={seed} {name}: {json.dumps(state.to_json())[:300]}")
                    _, ece, degenerate = calibration_states(scored)
                    for name, state in (("ece", ece), ("degenerate_head", degenerate),
                                        ("permutation_consistency", permutation_gate),
                                        *((g.name, g.state) for g in suite_gates)):
                        print(f"  {device} seed={seed} {name}: {json.dumps(state.to_json())[:300]}")
                    print(f"  score row {run['score_row_id']}")
                report["arm1"].append(run)  # type: ignore[union-attr]
                print(
                    f"arm1 {device} seed={seed}: {run['micro_batches']} batches, "
                    f"{run['supervised_tokens']} supervised tokens, {run['span_rows']} span "
                    f"rows, padding {run['padding_fraction']:.6f}, total "
                    f"{run['total_first']:.4f} -> {run['total_last']:.4f}, "
                    f"{run['wall_clock_s']}s, termination {run['termination']}, ft row "
                    f"{run['ft_row_id']}"
                )
                if not (run["letter_last"] < run["letter_first"]):
                    failures.append(f"arm1 {device} seed={seed}: the letter loss did not fall")

    if args.verdicts_out is not None:
        # Once, after every seed, so the file is either every eval row's verdicts or absent.
        # Absent, too, when no eval row was written: an empty file would read as a scored
        # model with no verdicts rather than as a score that never ran.
        if verdict_lines:
            write_verdicts_jsonl(args.verdicts_out, verdict_lines)
            print(f"verdicts: {len(verdict_lines)} lines -> {args.verdicts_out}")
        else:
            print(f"verdicts: NOT WRITTEN -- no eval row was recorded, {args.verdicts_out} absent")
    if args.suite_verdicts_out is not None:
        if suite_lines:
            write_suite_verdicts_jsonl(args.suite_verdicts_out, suite_lines)
            print(f"suite verdicts: {len(suite_lines)} lines -> {args.suite_verdicts_out}")
        else:
            print(
                "suite verdicts: NOT WRITTEN -- no suite case was decoded, "
                f"{args.suite_verdicts_out} absent"
            )
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if failures:
        print(f"\nFAILED: {len(failures)} claim(s) did not hold:")
        for line in failures:
            print(f"  - {line}")
        return 1
    return 0


def _labels_by_batch(
    reader: ShardReader, plan: list[Batch], labels: list[Label], *, config: DataConfig,
    batch_tokens: int,
) -> dict[int, list[Label]]:
    """Which :class:`Label` each row of each batch is.

    ``ShardReader._plan`` is a pure function of ``(seed, epoch, batch_tokens)`` and it is the
    only thing that knows which sequence landed in which row, so the membership is taken from
    it rather than re-derived. Checked: every row's stored ``target_index`` and ``slot_kind``
    must match the label's, over every row of every batch.
    """
    # The plan ``plan`` was yielded from, so it must be the same batch_tokens: a different
    # value is a different grouping, and the check below would refuse it row by row.
    plans = reader._plan(batch_tokens=batch_tokens, seed=config.seed, epoch=0)
    if len(plans) != len(plan):
        raise SystemExit(f"{len(plans)} planned batches against {len(plan)} yielded")
    out: dict[int, list[Label]] = {}
    for b, (p, batch) in enumerate(zip(plans, plan, strict=True)):
        group = [labels[i] for i in p.rows]
        for r, (i, label) in enumerate(zip(p.rows, group, strict=True)):
            if int(batch.slot_kind[r]) != label.slot_kind:  # type: ignore[index]
                raise SystemExit(
                    f"batch {b} row {r} is slot kind {int(batch.slot_kind[r])} but sequence "  # type: ignore[index]
                    f"{i} was labelled {label.slot_kind} ({label.row_id})"
                )
        out[b] = group
    return out


if __name__ == "__main__":
    raise SystemExit(main())
