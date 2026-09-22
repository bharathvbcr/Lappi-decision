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
most ``--max-width``. Under rule 8 that is a subsample on a truncated schedule and both arms
are ``quick``; they promote nothing.

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
    /Users/bharath/.venvs/ml/bin/python tools/real_ft_run.py --out /tmp/qd-real --ledger

The repo venv carries no torch by design, so this refuses there rather than reporting a
vacuous pass. Exit status is non-zero if any claim it makes fails.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
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
from repo_git import resolve_rev

# Counting GPUs needs torch and `run_control` is torch-free by contract, so the count lives
# in tools. Its own module rather than this one because `rung0_real_run.py` needs the same
# answer, and importing this file to get it would load a text tower to count a GPU.
from run_cost import n_gpus_for_device
from torch import nn

from qd_data.config import DataConfig
from qd_data.errors import QdRefusal
from qd_data.render import DEFAULT_CAPS, render
from qd_data.rows import DataRow
from qd_data.schema import NOUL_LETTER
from qd_train.artifacts import (
    SLOT_CHOICE,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
    Batch,
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
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.power import resolution_state
from qd_train.run_control import CostEstimate, LRSchedule, RunControl, WallClockCap
from qd_train.shards import (
    ShardReader,
    UnencodableGold,
    answer_letter,
    corpus_contradictions,
    training_texts,
)
from qd_train.trainer import SpanScoringStep, ft_supervision, train_ft
from qd_train.tristate import NotRun, Ran, TriState

REPO = Path(__file__).resolve().parents[1]

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

#: Seconds a single forward+backward probe is given before it is called a refusal. Generous:
#: the widest real batch is 34,522 positions and `mps` takes 13.6s for the attention alone.
PROBE_TIMEOUT_S = 240.0

#: The wall-clock cap every run here carries. `RunControl` stops at a group boundary and the
#: ledger row records `termination`, so a capped run is legible as capped, not as finished.
WALL_CLOCK_CAP_S = 1_800.0


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
            # `answer_letter`'s docstring warns about.
            rendered = render(row.request, caps=DEFAULT_CAPS, seed=config.seed)
            by_name = {slot.name: slot for slot in rendered.slots}
            for spec in training_texts(row, seed=config.seed, caps=DEFAULT_CAPS):
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
                 heads: int, lr: float, span_weight: float) -> None:
        torch.manual_seed(seed)
        if not span_weight > 0.0:
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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
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
        for group in self.optimizer.param_groups:
            group["lr"] = lr
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


def _decode(
    step: RealFtStep,
    batches: list[Batch],
    labels_for: dict[int, list[Label]],
    letter_id: dict[str, int],
    *,
    noul_first: bool = False,
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

    Never consults the loss. Returns per-row verdicts and the counts.
    """
    verdicts: list[dict[str, object]] = []
    with torch.no_grad():
        for b, batch in enumerate(batches):
            hidden = step.hidden(batch)
            logits = step.lm_head(hidden)
            supervision = ft_supervision(batch)
            plan = start_rows = end_rows = None
            if supervision.span is not None:
                plan = plan_span_batch(supervision.span, device=step.device)
                starts, ends = step.span_head(
                    hidden[torch.as_tensor(supervision.span.rows, device=step.device)], plan
                )
                start_rows = serving_scores(starts, plan)
                end_rows = serving_scores(ends, plan)
                span_index = {int(r): k for k, r in enumerate(supervision.span.rows)}
            for r in range(batch.tokens.shape[0]):
                label = labels_for[b][r]
                if label.slot_kind == SLOT_SPAN:
                    assert plan is not None and start_rows is not None and end_rows is not None
                    k = span_index[r]
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
                    verdicts.append({
                        "kind": "span",
                        "row_id": label.row_id,
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
                order = [x for x in label.letters if x != NOUL_LETTER]
                ordered = (
                    [NOUL_LETTER, *order] if noul_first else [*order, NOUL_LETTER]
                )
                row_tokens = [letter_id[x] for x in ordered]
                noul_row = len(row_tokens) - RESERVED_NOUL_ROWS
                at = int(batch.target_index[r])  # type: ignore[index]
                top = int(
                    logits[r, at, torch.as_tensor(row_tokens, device=step.device)].argmax()
                )
                expected = label.gold_letter == NOUL_LETTER
                gold_row = ordered.index(label.gold_letter)
                abstained = top == noul_row
                verdicts.append({
                    "kind": KIND_NAMES[label.slot_kind],
                    "row_id": label.row_id,
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
                    # The mass this model puts on a row it was never supervised at. Softmax
                    # over the slot's own rows, which is what `QueryKind::Letters` decodes.
                    "noul_probability": float(
                        torch.softmax(
                            logits[r, at, torch.as_tensor(row_tokens, device=step.device)],
                            dim=-1,
                        )[noul_row]
                    ),
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
) -> CostEstimate:
    """What this run costs, in one place.

    Two things need the answer and they must not be able to disagree: the ``RunControl``
    that gates the launch under rule 4, and the ledger row that records what was spent.
    Before this, the row read a separate ``cost_usd_per_hour`` float that defaulted to zero
    and that only one of two call sites passed -- so the estimate could be right while the
    row said a GH200 hour cost nothing.

    The cap is the same ``WALL_CLOCK_CAP_S`` the control uses, because ``projected_usd`` is
    priced from the cap; a cost built against a different cap would answer a different
    question about the same run.
    """
    return CostEstimate.for_device(
        cap=WallClockCap(cap_s=WALL_CLOCK_CAP_S),
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
    approved_by: str = "",
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
    """
    cap = WallClockCap(cap_s=WALL_CLOCK_CAP_S)
    return RunControl(
        schedule=LRSchedule(
            peak_lr=lr, total_steps=steps, warmup_steps=max(1, steps // 20), min_lr=lr / 10
        ),
        cap=cap,
        cost=_cost(
            device=device, n_gpus=n_gpus, usd_per_hour=usd_per_hour,
            usd_per_gpu_hour=usd_per_gpu_hour, instance=instance,
        ),
        grad_accum=1,
        checkpoint_every=checkpoint_every,
        # Threaded through so the refusal RunControl already makes is reachable. It fires on
        # `cost.requires_human_approval and not approved_by.strip()` -- which, while the
        # rate was 0.0 and n_gpus 0, could not fire at all.
        approved_by=approved_by,
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


def _recorder(ledger: Ledger, *, reader: ShardReader, seed: int, recipe: dict[str, object],
              run_kind: str, quick_reason: str, notes: str,
              wall_clock_s: float | None, cost: CostEstimate | None) -> RunRecorder:
    """Both of this tool's row kinds go through here, and they need different answers.

    ``None`` from :func:`_train`, whose ``with`` block contains ``train_ft``. A measured
    elapsed from :func:`_record_verdict`, which describes a decode that finished before its
    recorder existed -- 162 of this tool's rows recorded the time taken to write the row
    because that distinction had no way to be stated.
    """
    return RunRecorder(
        ledger,
        entry_point=Path(__file__),
        protocol=_protocol(reader=reader, seed=seed, recipe=recipe),
        run_kind=run_kind,  # type: ignore[arg-type]
        repo=REPO,
        env=Environment.detect(device=str(recipe["device"])),
        wall_clock_s=wall_clock_s,
        cost=cost,
        # The same dict `_protocol` hashed into `recipe_hash`, stored as well as hashed.
        # The hash makes two recipes incomparable and says nothing about how they differ;
        # a sweep's rows could not name their own arm without the launch command.
        recipe=recipe,
        quick=True,
        quick_reason=quick_reason,
        notes=notes,
    )


def _train(
    *, reader: ShardReader, plan: list[Batch], passes: int, device: str, seed: int,
    hidden: int, heads: int, lr: float, span_weight: float, ledger: Ledger, tag: str,
    quick_reason: str, backbone: Path | None = None, optimizer_recipe: str = "bf16",
    checkpoint_dir: Path | None = None, checkpoint_every: int = 0,
    resume_from: object | None = None, deterministic: bool = False,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    n_gpus: int | None = None, usd_per_hour: float | None = None,
    usd_per_gpu_hour: float | None = None, instance: str | None = None,
    approved_by: str = "",
) -> dict[str, object]:
    """Run ``train_ft`` over ``plan`` repeated ``passes`` times. One optimizer step per batch.

    The batches are the real reader's, re-indexed: ``_train`` requires strictly increasing
    indices inside an epoch because S5 reconstructs a resume position from them, and a
    repeated plan would otherwise go backwards.
    """
    width = max(int(b.tokens.shape[1]) for b in plan)
    steps = len(plan) * passes
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
    if backbone is None:
        step: SpanScoringStep = RealFtStep(
            seed=seed, device=device, vocab=int(reader.header.vocab_size), width=width,
            hidden=hidden, heads=heads, lr=lr, span_weight=span_weight,
        )
        # Only meaningful for the stand-in, so only recorded for it: under --real-backbone
        # these determine nothing and would still move recipe_hash.
        backbone_keys["hidden"] = hidden
        backbone_keys["heads"] = heads
        what_ran = (
            f"Backbone is a randomly-initialised {hidden}x{heads} single block; this is a "
            "statement about the loop and the data, not an evaluation of any model."
        )
    else:
        # Imported here, not at module scope: qd_train.backbone needs transformers and
        # safetensors, which are the optional `mac` extra. A module-level import would make
        # this tool unimportable wherever the stand-in path is the only one available --
        # which is every machine without a checkpoint.
        from qd_train.backbone import (
            QwenDecisionStep,
            load_text_tower,
            remap_text_tower,
        )
        from qd_train.memory import ADAMW_BF16, OptimizerSpec

        # ADAMW_BF16 is not ADAMW_FP32: torch.optim.AdamW keeps exp_avg and exp_avg_sq in
        # the parameter dtype, so a bf16 tower gets 2-byte states, and load_text_tower
        # refuses the mismatch rather than budgeting a layout nothing builds. The master
        # spec is the other real recipe -- fp32 master, fp32 moments -- and
        # QwenDecisionStep builds whichever the tower names.
        spec = (
            OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
            if optimizer_recipe == "master"
            else ADAMW_BF16
        )
        tower = load_text_tower(
            backbone,
            gradient_checkpointing=True,
            optimizer=spec,
            attn_implementation=attn_implementation,
            device=device,
            dtype="bf16",
            rows=max(int(b.tokens.shape[0]) for b in plan),
            width=width,
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
        # `steps` is computed at the top of this function and was always available here;
        # it now travels into the step, so a schedule that would outlive its own second
        # moment is refused before the tower is trained rather than discovered in a loss
        # curve that shows nothing.
        step = QwenDecisionStep(
            # The same `seed` this run records in its protocol and names its checkpoint
            # with. Until QwenDecisionStep took one, that seed governed the batch order and
            # not the span head's initialisation, so two runs at one seed were two runs.
            tower,
            seed=seed,
            lr=lr,
            total_steps=steps,
            span_weight=span_weight,
            max_width=width,
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
            usd_per_gpu_hour=usd_per_gpu_hour, instance=instance,
        ),
        quick_reason=quick_reason,
        notes=(
            f"tools/real_ft_run.py [{tag}] -- qd_train.trainer.train_ft over a shard set "
            f"written by tools/real_tokenizer_pipeline.py with the live Qwen tokenizer. "
            + what_ran
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

    def source():
        index = 0
        for _ in range(passes):
            for batch in plan:
                yield dataclasses.replace(batch, index=index)
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
        target = checkpoint_dir / _checkpoint_name(tag, seed, device)

        def on_checkpoint(ckpt: Any, _target: Path = target) -> None:
            # Timed and sized, because the interval is a cost decision and nothing here
            # could price it. This model's checkpoint is ~8.5 GB -- weights plus both AdamW
            # moments -- and at ~1.5 s/step a 20-step interval spends more wall clock
            # writing than training. The number belongs on screen next to the step it was
            # taken at, not in a handoff someone has to remember.
            t0 = time.monotonic()
            written = ckpt.write(_target)
            took = time.monotonic() - t0
            payload = sum(
                p.stat().st_size
                for p in written.parent.glob(f"{written.stem}*")
                if p.is_file()
            )
            print(
                f"  checkpoint: step {ckpt.optimizer_step} -> {written} "
                f"({payload / (1 << 30):.2f} GiB in {took:.1f}s)",
                flush=True,
            )

    result = train_ft(
        source(),
        epoch=0,
        step=step,
        control=_control(
            steps, device=device, lr=lr, checkpoint_every=checkpoint_every,
            n_gpus=n_gpus, usd_per_hour=usd_per_hour,
            usd_per_gpu_hour=usd_per_gpu_hour, instance=instance,
            approved_by=approved_by,
        ),
        recorder=recorder,
        on_checkpoint=on_checkpoint,
        resume_from=resume_from,
    )
    wall = time.monotonic() - started
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
            usd_per_gpu_hour=usd_per_gpu_hour, instance=instance,
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
                    quick_reason: str, decode_s: float, resolution: TriState) -> str:
    """One row per run for what happened **after** the last optimizer step.

    Separate from the ``ft`` row because ``train_ft`` owns its recorder's context manager and
    writes on exit, so a post-training fact cannot be in it.
    """
    recipe: dict[str, object] = {
        "tool": "tools/real_ft_run.py", "tag": f"{run['tag']}-verdict",
        "device": run["device"],
        # Mirrored from the run, not restated: the verdict row has to name the same backbone
        # the ft row named, and under --real-backbone there is no hidden/heads to name.
        **{k: run[k] for k in BACKBONE_KEYS if k in run},
        "shard_hash": reader.header.shard_hash(),
    }
    recorder = _recorder(
        ledger, reader=reader, seed=int(run["seed"]), recipe=recipe, run_kind="smoke",
        quick_reason=quick_reason,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="the pipeline's --out directory")
    parser.add_argument("--passes", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--rev", default="0632f693d3b765b726499e7b4bf19c67959b75cb")
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
    parser.add_argument("--epoch", action="store_true", help="also run arm 1, the real epoch")
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
    parser.add_argument("--probe", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

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

    import real_tokenizer_pipeline as pipeline

    # The same "which rows" the builder answered, from the same function, so a set built
    # from the download is relabelled from the download.
    commits, _, _ = pipeline.code_rows(
        commitpackft=args.commitpackft, max_pairs=args.max_pairs, rev=rev
    )
    spans, _ = pipeline.span_rows(max_rows=args.max_pairs, blank_line_runs=False, rev=rev)
    from qd_data.dedupe import dedupe
    from qd_data.mixture import build_mixture
    from qd_data.split import split

    mixture = build_mixture(
        {"bigcode/commitpackft": list(commits), "rajpurkar/squad_v2": list(spans)},
        config=config,
    )
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    train_rows = list(split_report.rows_by_split.get("train", ()))
    labels, excluded = _labels(train_rows, config=config)

    inventory = _inventory(reader, labels, excluded)
    inventory["contradictions"] = _contradictions(reader, labels)
    letter_id = _letter_ids(reader, labels)
    batch_tokens = int(max(reader.header.buckets))
    batch_info = _batch_inventory(reader, batch_tokens=batch_tokens, seed=config.seed)

    print(f"shard set: {shard_dir}")
    print(
        f"  rows in -> out: {inventory['coverage']['n_total']} -> {inventory['coverage']['n']}"
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
    print(f"  letters -> post-remap ids: {letter_id}")
    contra = inventory["contradictions"]
    print(
        f"  prompt-identical rows: {contra['rows_sharing_a_prefix']} of {contra['sequences']};"  # type: ignore[index]
        f" of those, {contra['contradicting_rows']} rows in "  # type: ignore[index]
        f"{contra['contradicting_groups']} groups carry DIFFERENT golds "  # type: ignore[index]
        f"({contra['per_kind']})"  # type: ignore[index]
    )
    print(f"  padding waste: {inventory['padding_waste']['detail']}")
    print(f"  span_check: {json.dumps(inventory['span_check'])[:180]}")
    print(f"  remap beside the shards: {json.dumps(inventory['remap_check'])[:180]}")
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
    labels_by_batch_all = _labels_by_batch(reader, plan_all, labels, config=config)
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
    if not plan_small:
        raise SystemExit(f"no real batch is at most {args.max_width} wide")


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
    quick_small = (
        f"a subsample on a truncated schedule: {len(plan_small)} of {len(plan_all)} real "
        f"batches ({sum(int(b.tokens.shape[0]) for b in plan_small)} of {len(reader)} "
        f"sequences) repeated {args.passes}x against {backbone_said}. Rule 8: excluded from "
        "every decision."
    )
    quick_epoch = (
        f"one epoch over {len(reader)} real sequences against {backbone_said}, "
        f"{len(plan_all)} optimizer steps. Rule 8: a truncated schedule is quick and "
        "promotes nothing."
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

    for device in devices:
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
                tag="memorise", quick_reason=quick_small,
            )
            step = run.pop("_step")
            decode_at = time.monotonic()
            shipped = _decode(step, plan_small, labels_small, letter_id, noul_first=False)
            defect = _decode(step, plan_small, labels_small, letter_id, noul_first=True)
            decode_s = time.monotonic() - decode_at
            run["verdict_row_id"] = _record_verdict(
                run, ledger=ledger, reader=reader, shipped=shipped, defect=defect,
                inventory=inventory, quick_reason=quick_small, decode_s=decode_s,
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
                    tag="epoch", quick_reason=quick_epoch,
                )
                run.pop("_step")
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

    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if failures:
        print(f"\nFAILED: {len(failures)} claim(s) did not hold:")
        for line in failures:
            print(f"  - {line}")
        return 1
    return 0


def _labels_by_batch(
    reader: ShardReader, plan: list[Batch], labels: list[Label], *, config: DataConfig
) -> dict[int, list[Label]]:
    """Which :class:`Label` each row of each batch is.

    ``ShardReader._plan`` is a pure function of ``(seed, epoch, batch_tokens)`` and it is the
    only thing that knows which sequence landed in which row, so the membership is taken from
    it rather than re-derived. Checked: every row's stored ``target_index`` and ``slot_kind``
    must match the label's, over every row of every batch.
    """
    plans = reader._plan(
        batch_tokens=int(max(reader.header.buckets)), seed=config.seed, epoch=0
    )
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
