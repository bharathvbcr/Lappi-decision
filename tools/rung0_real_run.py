"""Rung 0 trained on a real corpus and measured on files it never saw.

``docs/lappi.md`` describes a ladder whose ordering was "a decision, not a default": the
cheap rungs run first, so the expensive one is justified by measurement rather than by
plan. Rung 0 is the cheapest, and its status is **model built, untrained** -- all twelve of
its ledger rows are ``run_kind=smoke, quick=True`` from ``tools/rung0_toy_run.py``, which
trains one memorisable batch and says so. Rung 3, the most expensive and least evidenced
rung, is where every GPU hour has gone.

This tool is the missing measurement. It is the same pipeline the toy run exercises --
``qd-mutate`` -> :func:`qd_train.mutate_adapter.to_decision` ->
:func:`qd_train.byte_batch.plan_batch` -> :func:`qd_train.byte_train.train_rung0`
-- pointed at a real corpus, with the one change that makes the number mean something:
**the evaluation files are not the training files.**

## What is measured, and against what

Accuracy on its own is unreadable. ``MUTATION_CLASSES`` is not uniform and
``--clean-permille`` sets how much of the corpus is ``clean`` by construction, so a model
that answers the majority class every time scores whatever that class's share is. Every
accuracy here is reported beside that baseline, and the gap between them is the only part
that is evidence. A model at the baseline has learned the prior and nothing else.

## Why the split is by file

Mutations are generated per function, so one file yields many examples that share
surrounding context, naming and style. Splitting examples at random would put siblings of
a validation example in training and report memorisation as generalisation. The split is
over *paths*, hashed, so a file is wholly on one side.

That is weaker than the plan's real split -- same repository, same author, same idioms on
both sides -- and it is named here rather than left for a reader to assume otherwise.
Every row this tool writes is ``quick``: the corpus is this repository rather than the
pool the plan names, which rule 8 calls a subsample.

## Rule 3

``qd_train.mutate_adapter.read_examples`` is rung 0's held-out door and refuses a
held-out path. This tool generates its corpus into a scratch directory under ``--out`` and
never names ``data/heldout``; the door is still the thing that enforces it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

# Read from argv rather than from parsed arguments, and deliberately: cuBLAS reads this when
# it initialises, which is the first matmul, and `argparse` has not run by then. With
# deterministic algorithms in force and this unset, torch raises at the first addmm rather
# than silently using a nondeterministic one -- so the failure mode of getting this wrong is
# loud, and the failure mode of parsing argv here instead is a 32 MB cuBLAS workspace on a
# run that did not ask for one. Only set when asked, so an ordinary run is untouched.
#
# The same shape as tools/real_ft_run.py, and copied rather than re-derived: two spellings
# of this would be two ways to get it subtly wrong.
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from repo_git import git_bytes, tracked_paths  # noqa: E402
from run_cost import n_gpus_for_device  # noqa: E402

from qd_train.baseline import LinearBaseline  # noqa: E402
from qd_train.byte_batch import BatchPlan, plan_batch, span_supervision  # noqa: E402
from qd_train.byte_context import ID_PAD, SpanOutsideWindow  # noqa: E402
from qd_train.byte_decider import ByteDeciderConfig  # noqa: E402
from qd_train.byte_train import Rung0Model, Rung0Step, train_rung0  # noqa: E402
from qd_train.calibration_fit import ece_gate  # noqa: E402
from qd_train.control_cache import control_key, load_control, store_control  # noqa: E402
from qd_train.eval_harness import (  # noqa: E402
    degenerate_head_check,
    paired_margin_test,
    shuffled_label_control,
)
from qd_train.heads import plan_span_batch, serving_scores  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.mutate_adapter import (  # noqa: E402
    CONTEXT_AFTER,
    CONTEXT_DIFF,
    CONTEXT_SOURCES,
    MUTATION_CLASSES,
    EmptyDiffContext,
    MalformedExample,
    PhantomFinalLine,
    SpanOutsideDiff,
    parse_example,
    refuse_leaky_diff_corpus,
    to_decision,
)
from qd_train.power import resolution_state  # noqa: E402
from qd_train.run_control import CostEstimate, LRSchedule, RunControl, WallClockCap  # noqa: E402
from qd_train.tristate import NotRun, Ran, TriState  # noqa: E402

#: Extensions qd-mutate has a grammar for, and which this repository actually contains.
POOL_SUFFIXES: Final[dict[str, str]] = {".rs": "rust", ".py": "python"}

#: A file bigger than this is skipped rather than truncated: qd-mutate parses whole files,
#: and a partial file is a different file.
MAX_POOL_BYTES: Final[int] = 120_000

#: Hard bounds. A schedule long enough to be interesting is long enough to hide a wiring
#: bug behind a plausible curve, and these are the ceilings that keep a typo cheap.
MAX_FILES: Final[int] = 4_000
MAX_EXAMPLES: Final[int] = 20_000
MAX_EPOCHS: Final[int] = 200

#: Rung 0's default context, and why it is a flag rather than a constant.
#:
#: ``to_decision`` keeps the TAIL of the post-mutation file and refuses a span that falls in
#: the dropped head -- correct, since a span head must not be taught to point at whatever
#: truncation left behind. So the window decides how much of a real corpus exists at all.
#: Measured over 2,156 qd-mutate examples from 169 of this repository's files:
#:
#: | bytes | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
#: | kept  | 6.4% | 9.0% | 15.5% | 26.6% | 49.4% | 70.1% | 86.4% |
#:
#: Every refusal is ``SpanOutsideWindow``; nothing else refuses at any width. At the 1024
#: ``ByteDeciderConfig`` defaults to, **91% of the corpus is unusable**, which is a fact
#: about the corpus rather than about the model and has to be settled before any accuracy
#: from this tool means anything. ``docs/lappi.md`` calls capacity "the first thing to raise
#: if rung 0 underperforms"; this is the prior question of whether there is enough corpus to
#: underperform on.
DEFAULT_CONTEXT_BYTES: Final[int] = 4096

#: Named rather than written inline into the parser, because `tools/fit_linear_control.py`
#: has to split the corpus EXACTLY as this tool does or the cache key it computes is not
#: the key this tool looks up. Restating the values there diverged them immediately: the
#: fit defaulted to 0.20 while this parser defaulted to 0.25, which would have produced two
#: different splits, a guaranteed cache miss, and an hour of CPU spent on a verdict nothing
#: could read. One owner per default; the other tool imports these.
DEFAULT_VAL_SHARE: Final[float] = 0.25
DEFAULT_BATCH_SIZE: Final[int] = 16

#: Peak of the LR schedule. Unchanged from the literal it replaces, so every row written
#: before this became a flag hashes exactly as it did -- the recipe already carried an `lr`
#: key, stating a number nothing obliged the run to use, because the schedule was built from
#: a SECOND literal in `train_once`. One owner now, reaching both.
DEFAULT_PEAK_LR: Final[float] = 3e-3

#: Attention is quadratic in the context, so this is a real ceiling and not a typo guard.
MAX_CONTEXT_BYTES: Final[int] = 32_768

#: The wall-clock cap every rung 0 seed runs under, and the cap its cost is priced
#: against. Named rather than repeated: `CostEstimate` computes `projected_usd` from
#: the cap, so a row priced against a different cap answers a different question about
#: the same run than the control that gated it does.
RUN_CAP_S: Final[float] = 3600.0


# -- the corpus --------------------------------------------------------------------------


def build_pool(*, rev: str, max_files: int) -> tuple[list[dict[str, object]], bool]:
    """Tracked source files at ``rev`` as qd-mutate pool records.

    ``rev`` is explicit rather than ``HEAD`` for the reason the rung 3 pipeline gives: several
    lanes commit to this worktree, and two runs against "HEAD" hours apart read different
    corpora while reporting the same protocol.

    Returns the records and whether the bound actually bound, because a capped corpus and a
    complete one must not report the same way.
    """
    if not 1 <= max_files <= MAX_FILES:
        raise ValueError(f"max_files must be in [1, {MAX_FILES}], got {max_files}")
    records: list[dict[str, object]] = []
    for name in tracked_paths(REPO, rev=rev, suffixes=frozenset(POOL_SUFFIXES)):
        language = POOL_SUFFIXES[Path(name).suffix]
        try:
            raw = git_bytes(REPO, "show", f"{rev}:{name}")
        except subprocess.CalledProcessError:
            continue
        if len(raw) > MAX_POOL_BYTES:
            continue
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not source.strip():
            continue
        if len(records) >= max_files:
            return records, True
        records.append({
            "id": f"pool-{len(records):05d}",
            "repo": "qwen-decision",
            "path": name,
            "language": language,
            "source": source,
        })
    return records, False


def generate_examples(
    pool: Sequence[dict[str, object]], *, out_dir: Path, seed: int, limit: int, binary: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run qd-mutate over the pool and read back its examples and manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "pool.jsonl"
    pool_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool), encoding="utf-8"
    )
    examples_path = out_dir / "examples.jsonl"
    manifest_path = out_dir / "manifest.json"
    subprocess.run(
        [
            str(binary), "generate",
            "--pool", str(pool_path),
            "--out", str(examples_path),
            "--manifest", str(manifest_path),
            "--seed", str(seed),
            "--limit", str(limit),
        ],
        check=True, capture_output=True, text=True,
    )
    examples = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return examples, json.loads(manifest_path.read_text(encoding="utf-8"))


def split_by_file(
    examples: Sequence[dict[str, object]], *, val_share: float
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Train and validation, disjoint by source path.

    Hashed rather than shuffled so the assignment is a property of the path and does not
    move when the corpus grows: adding files must not silently move an old file across the
    split and turn a held-out measurement into a training one.
    """
    if not 0.0 < val_share < 1.0:
        raise ValueError(f"val_share must be in (0, 1), got {val_share}")
    cut = int(val_share * (1 << 16))
    train: list[dict[str, object]] = []
    val: list[dict[str, object]] = []
    for example in examples:
        path = str(example.get("function", {}).get("path", ""))  # type: ignore[union-attr]
        digest = hashlib.sha256(path.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big")
        (val if bucket < cut else train).append(example)
    return train, val


def recipe_of(args: argparse.Namespace) -> dict[str, object]:
    """Every field that makes two runs different protocols rather than one repeated.

    Named rather than written inline into the hash. Every field here went into
    ``recipe_hash`` and was stored nowhere readable, so a row could say two arms differ
    and not say how: the concurrent lane recovered a learning curve's point labels on
    2026-09-21 by re-hashing four candidate ``train_subsample`` values with the other
    seven fields pinned at the launch command's -- which works, and needs the launch
    command. One object reaches both the hash that makes the arms incomparable and the
    row that says what they were, so the two cannot drift.

    A function rather than an expression inside ``main`` so the two invariants it carries
    are testable: that an ``after`` run hashes exactly as it did before ``--context-source``
    existed, and that a ``diff`` run cannot hash like one.
    """
    recipe: dict[str, object] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_share": args.val_share,
        # The peak of the schedule, which was a literal in two places -- here and in
        # `train_once` -- so the recipe stated a number the run was not obliged to use. It
        # is now one value reaching both, and `DEFAULT_PEAK_LR` keeps its old value so every
        # row written before it became a flag hashes exactly as it did.
        "lr": args.lr,
        # The objective is part of the recipe. Without this the five points of the
        # span-weight sweep hash identically, and two runs that optimised different things
        # become one protocol in the ledger -- which is exactly the comparison the sweep
        # exists to make.
        "span_weight": args.span_weight,
        # Deterministic and nondeterministic runs are different protocols, not the same
        # protocol measured twice. Without this they hash identically and the ledger treats
        # a reproducible number and a draw from a 2.8-point spread as comparable rows.
        "deterministic": args.deterministic,
        # The control and the arm it controls for MUST NOT hash alike. Everything else
        # about them is identical by design -- same corpus, same schedule, same capacity --
        # so without these two fields a control run and a real run share a recipe_hash, and
        # a reader pooling by protocol would average a model trained on destroyed labels
        # into the measurement it exists to validate.
        "shuffle_train_labels": args.shuffle_train_labels,
        "shuffle_seed": args.shuffle_seed if args.shuffle_train_labels else None,
        # How much of the training set was used. A learning curve's whole content is that
        # its points differ in this and nothing else, and `data_snapshot_hash` cannot see
        # it: it comes from the manifest, which a subsampled run does not change. Without
        # this a half-data arm and a full-data one agree on every protocol field there is,
        # and the ledger reads two populations as one protocol measured twice.
        "train_subsample": args.train_subsample,
        "rev": args.rev,
    }
    # WHAT the model read is the most load-bearing protocol field there is: a post-image
    # run and a diff run are not one protocol measured twice, they are two questions. So
    # it must reach the hash -- but ADDED ONLY WHEN IT IS NOT THE DEFAULT. Writing it
    # unconditionally would move the hash of every `after` run too, and the 61 capacity
    # rows and the learning curve already in the ledger were written without it; new
    # post-image rows would then be incomparable with the rows they exist to extend, for
    # a field whose value never varied. Omitted-at-default keeps both invariants: every
    # `after` run hashes as it always did, and no `diff` run can collide with one.
    if args.context_source != CONTEXT_AFTER:
        recipe["context_source"] = args.context_source
    # Same rule, same reason: added only when set, so no row written before these existed
    # moves. A run that trained without an operator and one that trained on everything are
    # two protocols, and so are the holdout and its size-matched control -- which differ in
    # nothing else at all, and would otherwise be the one pair in this experiment guaranteed
    # to hash alike.
    if getattr(args, "hold_out_operator", ""):
        recipe["hold_out_operator"] = args.hold_out_operator
    if getattr(args, "drop_random_train", 0):
        recipe["drop_random_train"] = args.drop_random_train
        recipe["drop_random_seed"] = args.drop_random_seed
    # Only when it is NOT the one `hold_out_operator` already implies. Recording it twice
    # would give the same run two hashes depending on which flag the caller spelled out,
    # which is the opposite of what the hash is for. A reader wanting the measured operator
    # reads it the way `measure_operator` computes it, not from this key alone.
    if measure_operator(args) and not getattr(args, "hold_out_operator", ""):
        recipe["measure_operator"] = args.measure_operator
    # Same omitted-at-default rule, and here it is load-bearing rather than tidy: with this
    # set the span head has a target, so its gradient reaches the shared trunk and the
    # choice head's number moves. A diff run with it and one without are two protocols, and
    # writing the field unconditionally would move the hash of all 120-plus diff rows that
    # trained with the span head given nothing.
    if getattr(args, "span_in_diff", False):
        recipe["span_in_diff"] = True
    return recipe


def filter_train_rows(
    train_raw: list[dict[str, object]],
    *,
    hold_out_operator: str = "",
    drop_random_train: int = 0,
    drop_random_seed: int = 0,
) -> tuple[list[dict[str, object]], str]:
    """The training-side filter for the operator-holdout arms, and its one-line note.

    Applied to TRAINING ONLY and after the split, so the held-out operator's rows stay in
    validation. Filtering the corpus instead would take them off both sides and leave
    nothing to ask the question of. ``drop_random_train`` is the counterpart control: it
    removes the same NUMBER of rows at random, because without it a collapse is explained
    as well by "the training set shrank" as by "the fingerprint went", and ``stub.panic``
    alone is 43% of this corpus.

    **One owner, called by both tools.** ``tools/fit_linear_control.py`` must fit the linear
    control on exactly the training set the run trains on, or the two arms of
    ``paired_margin_vs_linear`` are not opponents. The cache key is computed over the
    training DOCUMENTS, so a second copy of this that drifted would not produce a wrong
    margin -- it would produce a cache miss and a ``not_run`` margin, which is safe and
    useless. Sharing the filter is what makes the margin available at all.
    """
    if hold_out_operator:
        before = len(train_raw)
        kept = [r for r in train_raw if r.get("operator") != hold_out_operator]
        if len(kept) == before:
            raise SystemExit(
                f"--hold-out-operator {hold_out_operator!r} matched no training row. "
                "A typo would otherwise train on everything and report the unseen condition "
                "as though the operator had been removed."
            )
        return kept, (
            f"holding out {hold_out_operator}: {before - len(kept)} of {before} "
            f"training row(s) removed ({(before - len(kept)) / before:.0%})"
        )
    if drop_random_train:
        before = len(train_raw)
        if drop_random_train >= before:
            raise SystemExit(
                f"--drop-random-train {drop_random_train} would empty a training set "
                f"of {before}"
            )
        # Sorted, so the kept rows stay in corpus order: the control cache keys on the
        # training documents, and two runs that kept the same rows in different orders
        # would hash differently and never share a fit.
        keep = random.Random(drop_random_seed).sample(
            range(before), before - drop_random_train
        )
        return [train_raw[i] for i in sorted(keep)], (
            f"size-matched control: {drop_random_train} of {before} training row(s) "
            f"dropped at random (seed {drop_random_seed})"
        )
    return train_raw, ""


#: The byte a unified diff marks an added line with. Named because `ids` holds byte values
#: and a bare 43 at the comparison site is a magic number in the one place it matters.
_PLUS: Final[int] = ord("+")


def plus_line_chance(decisions: Sequence[object]) -> tuple[float, int, int]:
    """What "point at a uniformly random ADDED line" scores, over the same scored rows.

    The honest null for span pointing in diff space, and the reason it is needed: the
    uniform-over-all-candidate-lines rate that ``span_pointing_chance`` computes is the
    right null when the model reads a file, where nothing marks the changed line. In a
    unified diff the changed line is marked with a ``+``, and both corpora average about
    1.5 added lines per hunk -- so a policy that reads no code at all and points at an
    added line scores on the order of 65%, against a uniform rate near 9%. Reporting a
    pointer's accuracy only against the uniform rate would present that marker as a result.

    Returns ``(chance, pointing_rows, rows_whose_gold_is_an_added_line)``. The second and
    third are carried rather than folded in because they are different facts: a corpus
    where the gold is often NOT an added line is one where this null is weak and the
    uniform rate is closer to right, and that has to be visible rather than averaged away.
    """
    total = 0.0
    pointing = 0
    gold_is_added = 0
    for d in decisions:
        line = getattr(d, "gold_span_line", None)
        if line is None:
            continue
        pointing += 1
        context = d.context  # type: ignore[attr-defined]
        # `ids` are byte values and `starts` index into them, which is the identity
        # EncodedContext exists to preserve -- so the first byte of a line is ids[start]
        # with no second mapping to keep in agreement.
        ids, starts = context.ids, context.starts
        added = sum(1 for s in starts if ids[s] == _PLUS)
        if ids[starts[line]] != _PLUS:
            continue
        gold_is_added += 1
        total += 1.0 / added  # at least 1: the gold line itself is one
    return (total / pointing if pointing else 0.0), pointing, gold_is_added


def operator_and_sibling_rows(
    scored_operators: Sequence[str], gold: Sequence[int], measured: str
) -> tuple[list[int], list[int], int]:
    """Row indices for `measured`, for its same-class siblings, and the impure count.

    Siblings are the rows that make the held-out number readable. Every operator in this
    corpus belongs to exactly one class, so removing one moves the class prior -- and a
    model that then under-predicts that class scores badly on the held-out rows for a
    reason unrelated to recognising generators. Siblings sit under the identical shifted
    prior and training saw them, so they hold the prior fixed while the generator varies.

    Returns indices rather than values so the caller pairs them against the same arrays
    everything else on the row is computed from, instead of a second copy that could drift.
    """
    marked = [i for i, op in enumerate(scored_operators) if op == measured]
    if not marked:
        return [], [], 0
    counts: dict[int, int] = {}
    for i in marked:
        counts[int(gold[i])] = counts.get(int(gold[i]), 0) + 1
    # The class the operator produces, by majority rather than by assuming purity: an
    # operator that straddles classes would otherwise define its own sibling set away.
    sibling_class = max(counts, key=lambda k: (counts[k], -k))
    impure = len(marked) - counts[sibling_class]
    siblings = [
        i
        for i, op in enumerate(scored_operators)
        if op and op != measured and int(gold[i]) == sibling_class
    ]
    return marked, siblings, impure


def measure_operator(args: argparse.Namespace) -> str:
    """The operator whose validation rows get their own accuracy, or "".

    One owner, because the rule -- `--hold-out-operator` implies measuring the operator it
    held out -- is read in three places (the recipe, the pre-flight, the metric) and three
    copies of it would be three chances to disagree about which rows a number describes.
    """
    return getattr(args, "measure_operator", "") or getattr(args, "hold_out_operator", "")


def decisions_of(
    examples: Sequence[dict[str, object]],
    *,
    config: ByteDeciderConfig,
    context_source: str = CONTEXT_AFTER,
    span_in_diff: bool = False,
) -> tuple[list, dict[str, int], list[str], list[str]]:
    """Parse and convert, counting every refusal by kind rather than dropping quietly.

    Returns the decisions, the refusals by kind, **the source path of each decision**, and
    **the operator that produced each decision** (``""`` for ``clean``, which had none).

    Both parallel lists are returned rather than recovered by zipping against the input,
    because this function DROPS refused examples: 1,599 raw examples become 763 decisions at
    8192 bytes, so ``zip(examples, decisions)`` pairs each decision with the wrong file and
    every per-row operation downstream is quietly wrong. They are built here, where the row
    being dropped is still in hand.

    The operator is carried because ``MutateExample`` deliberately does not: it is one of
    the "unconsumed fields" that module refuses to store, and that is the right call for the
    training seam. But the operator-holdout experiment needs to know which validation rows a
    given operator produced (``AUDIT/operator-holdout.md``), and recovering it by re-parsing
    in the caller would reintroduce exactly the index drift this function exists to prevent.
    """
    out = []
    paths: list[str] = []
    operators: list[str] = []
    refused: dict[str, int] = {}
    for obj in examples:
        try:
            decision = to_decision(
                parse_example(obj),
                max_context_bytes=config.max_context_bytes,
                context_source=context_source,
                span_in_diff=span_in_diff,
            )
        # `EmptyDiffContext` and `SpanOutsideDiff` belong here for the reason
        # `EmptyDiffContext`'s own docstring gives -- "decisions_of counts refusals by
        # exception name and these two must not share a line" -- and neither was in this
        # tuple, so a corpus with SOME void diffs, which the perfectly-separable pre-flight
        # passes, killed the run with an uncaught exception instead of counting the rows it
        # could not use. A refusal that crashes is not the same as a refusal that is
        # counted, and only the second leaves a number in the row.
        except (
            EmptyDiffContext,
            MalformedExample,
            PhantomFinalLine,
            SpanOutsideDiff,
            SpanOutsideWindow,
        ) as exc:
            name = type(exc).__name__
            refused[name] = refused.get(name, 0) + 1
            continue
        out.append(decision)
        paths.append(str(obj.get("function", {}).get("path", "")))  # type: ignore[union-attr]
        # `clean` rows carry no operator; `""` rather than `None` so the list is uniformly
        # typed and a caller comparing against an operator name never matches them.
        operators.append(str(obj.get("operator") or ""))
    return out, refused, paths, operators


# -- batching ----------------------------------------------------------------------------


def bucketed_batches(
    decisions: Sequence, *, batch_size: int, config: ByteDeciderConfig
) -> list[BatchPlan]:
    """Length-sorted batches, so a batch's width is its own rows and not the corpus maximum.

    ``plan_batch`` pads to the batch maximum, which is the right choice and is exactly why
    the batch's membership matters: one long row drags the whole batch's width up and every
    short row in it pays for the difference in padding.
    """
    return [
        plan_batch(
            chunk,
            pad_id=ID_PAD,
            max_context_bytes=config.max_context_bytes,
            max_option_bytes=96,
        )
        for chunk in bucketed_chunks(decisions, batch_size=batch_size)
    ]


def bucketed_chunks(decisions: Sequence, *, batch_size: int) -> list[list]:
    """The chunks :func:`bucketed_batches` will plan, in the order it will plan them.

    Factored out because two things need this ordering and they must not disagree: the
    batcher, and anything pairing a per-row result back to the decision that produced it.
    A second implementation of "sorted by kept bytes, chunked, short chunk dropped" is a
    copy that drifts, and the failure it produces -- predictions scored against another
    example's answer -- is invisible to a count and to a spot check.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    ordered = sorted(decisions, key=lambda d: d.context.n_bytes_kept)
    chunks = []
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        if len(chunk) < 2:
            # A one-row batch has no contrast for the choice head and cannot be scored
            # against a per-batch floor. Dropped and counted by the caller's arithmetic
            # (len(plans) * batch_size against len(decisions)), never padded out with a
            # duplicate row, which would be supervision this corpus does not contain.
            continue
        chunks.append(chunk)
    return chunks


def cycle(plans: Sequence[BatchPlan], *, n: int) -> Iterator[BatchPlan]:
    """``n`` plans, repeating the epoch. Ordered, so a run is reproducible from its seed."""
    for i in range(n):
        yield plans[i % len(plans)]


# -- measurement -------------------------------------------------------------------------


def label_entropy(decisions: Sequence) -> float:
    """The choice loss of a model that has learned the class prior and nothing else.

    ``tools/rung0_toy_run.py`` computes the same floor per context group and says why:
    "Computing it turns 'the loss fell' into 'the loss reached the floor', which is the
    difference between a curve that moved and a head that is wired to the right rows." On a
    real corpus no two rows share a context, so the conditional floor collapses to the
    marginal one -- the entropy of the label distribution.

    This is the instrument the real run was missing. The TOTAL loss falls from ~14.5 to
    ~1.4 and looks like healthy training, because the span channel outweighs the choice
    channel about 6:1 at initialisation. Against this number the choice channel's ~1.13 is
    legible as what it is: the prior, exactly, and nothing conditional on the context.
    """
    counts: dict[int, int] = {}
    for d in decisions:
        counts[d.gold_option] = counts.get(d.gold_option, 0) + 1
    n = sum(counts.values())
    if n == 0:
        raise ValueError("cannot compute a label entropy over zero decisions")
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def _prior_gate(measured: float, floor: float, *, n: int) -> TriState:
    """Whether the choice head learned anything the class prior does not already give.

    ``passed`` requires the loss to sit meaningfully BELOW the prior's entropy. A head at
    the floor has converged to answering the marginal distribution, which is the same fact
    the majority-class accuracy reports and is worth stating twice: an accuracy can land on
    the baseline by luck on 303 rows, a loss landing on the floor to three decimals cannot.
    """
    if n == 0:
        return NotRun(
            reason=(
                "no training decisions, so there is no label distribution and no floor to "
                "compare the choice loss against"
            )
        )
    # One percent of a nat: below that the two are the same number at this precision.
    margin = 0.01
    return Ran(
        passed=measured < floor - margin,
        value=measured,
        n=n,
        n_total=n,
        detail=(
            f"choice loss converged to {measured:.4f} nats against a label-distribution "
            f"entropy of {floor:.4f} over {n} training decisions -- a gap of "
            f"{measured - floor:+.4f}. At the floor the head has learned the class prior "
            "and nothing conditional on the context, and the TOTAL loss does not show it "
            "because the span channel outweighs this one about 6:1 at initialisation."
        ),
    )


def shuffle_train_labels(decisions: Sequence, *, seed: int) -> list:
    """Permute the gold labels among the training decisions, destroying the signal.

    The control this feeds asks whether a model trained on destroyed labels still beats
    chance on the held-out set. If it does, the split leaks -- through file paths, repo
    names or near-duplicates -- and every accuracy measured on that split is worthless.
    Across 988 ledger rows this control has never once been evaluated, because
    ``eval_harness.shuffled_label_control`` needs the accuracy of a shuffled-label model
    and nothing in this repository ever trained one.

    A PERMUTATION rather than random labels, because the control compares against the
    majority-class rate: permuting preserves the label distribution exactly, so the bar it
    is measured against remains the bar that applies. Drawing fresh random labels would
    shift the marginal and move the ceiling with it.

    Permuted **within groups that share an option count**. ``ByteDecision`` is frozen and
    its ``__post_init__`` refuses a ``gold_option`` outside ``options``, so a global
    permutation across decisions offering different numbers of options would raise partway
    through -- and would do so only on a corpus where the counts differ, which is not this
    one today and is not a property to rely on silently.
    """
    rng = random.Random(seed)
    by_arity: dict[int, list[int]] = {}
    for index, decision in enumerate(decisions):
        by_arity.setdefault(len(decision.options), []).append(index)

    shuffled = list(decisions)
    for indices in by_arity.values():
        labels = [decisions[i].gold_option for i in indices]
        rng.shuffle(labels)
        for i, label in zip(indices, labels, strict=True):
            shuffled[i] = dataclasses.replace(decisions[i], gold_option=label)
    return shuffled


def majority_baseline(decisions: Sequence) -> tuple[float, str]:
    """The share of the most common gold class, and which class it is.

    The number every accuracy in this report has to beat to mean anything.
    """
    counts: dict[int, int] = {}
    for d in decisions:
        counts[d.gold_option] = counts.get(d.gold_option, 0) + 1
    best = max(counts, key=lambda k: counts[k])
    return counts[best] / len(decisions), MUTATION_CLASSES[best]


def evaluate(model: Rung0Model, plans: Sequence[BatchPlan], *, device: str) -> dict[str, object]:
    """Argmax agreement with the gold over every plan, through the serving-side helpers.

    Read through :func:`qd_train.heads.serving_scores` rather than off the padded matrix,
    for the reason the toy run gives: the padded matrix has columns no runtime ever sees,
    and scoring them measures a model the caller cannot use.
    """
    model.eval()
    choice_hit = choice_n = 0
    start_hit = end_hit = span_n = 0
    # Rows whose gold span is a real line, counted apart from rows whose gold is the
    # abstention. See the loop below: the undecomposed figure is a mixture, and a corpus
    # where every gold abstains scores 100% on it having pointed at nothing.
    pointing_n = pointing_start_hit = pointing_end_hit = 0
    pointing_chance = 0.0
    # The per-row choice distribution, kept so `degenerate_head_check` can be evaluated.
    # It needs nothing but these numbers, which this loop already computes -- which is why
    # `degenerate_head` sat at not_run on 988 rows for want of four lines rather than for
    # want of a measurement. Softmax over the live options, in float32 because a bf16
    # softmax rounds small probabilities to zero and would understate the entropy the
    # check thresholds on.
    choice_probs: list[list[float]] = []
    # The gold for each row of `choice_probs`, taken from the SAME loop iteration.
    #
    # Not `[d.gold_option for d in val_d]`, which is the trap: `bucketed_batches` sorts by
    # `context.n_bytes_kept` and drops any trailing one-row chunk, so the decisions arrive
    # here length-sorted and possibly fewer. On this corpus 288 val decisions at batch 16
    # divide exactly, so the two lists would be the same LENGTH and a length check would
    # pass while every probability was scored against another example's answer.
    choice_gold: list[int] = []
    # Chance for a POINTER is 1/candidates, not 0. A span head choosing uniformly among a
    # row's line starts scores that, so it is the number a measured span accuracy has to
    # beat -- and it is accumulated per row because rows have different line counts.
    span_chance = 0.0
    with torch.no_grad():
        for plan in plans:
            ctx = torch.tensor(plan.context_ids, dtype=torch.long, device=device)
            mask = torch.tensor(plan.context_mask, dtype=torch.bool, device=device)
            hidden = model.decider.encode_context(ctx, mask)
            logits = model.decider.score_from_hidden(
                hidden,
                mask,
                torch.tensor(plan.option_ids, dtype=torch.long, device=device),
                torch.tensor(plan.option_mask, dtype=torch.bool, device=device),
                torch.tensor(plan.n_live_options, dtype=torch.long, device=device),
            )
            choice_probs.extend(torch.softmax(logits.float(), dim=1).tolist())
            for i, top in enumerate(logits.argmax(dim=1).tolist()):
                choice_hit += 1 if top == plan.choice_target[i] else 0
                choice_gold.append(int(plan.choice_target[i]))
                choice_n += 1
            span_plan = plan_span_batch(span_supervision(plan), device=device)
            if span_plan.n_spans:
                start, end = model.span_head(hidden, span_plan)
                start_rows = serving_scores(start, span_plan)
                end_rows = serving_scores(end, span_plan)
                for k in range(span_plan.n_spans):
                    gold_start = int(span_plan.gold_start[k])
                    gold_end = int(span_plan.gold_end[k])
                    hit_start = int(start_rows[k].argmax()) == gold_start
                    hit_end = int(end_rows[k].argmax()) == gold_end
                    start_hit += 1 if hit_start else 0
                    end_hit += 1 if hit_end else 0
                    # `serving_scores` returns exactly the rows a runtime would accept, so
                    # its length is the real number of choices this pointer had.
                    span_chance += 1.0 / max(1, int(start_rows[k].numel()))
                    span_n += 1
                    # Rows whose gold is a real line, kept apart from rows whose gold is the
                    # abstention. Mixing them reports "correctly abstained" and "correctly
                    # pointed" as one number, and the mixture rises when the corpus gets
                    # EASIER to abstain on. In `--context-source diff` every gold is the
                    # abstention -- span offsets are into `after` and every byte has moved --
                    # so the mixture reads 100.0% against a chance rate computed over line
                    # starts the head was never asked to choose between. A vacuous 100% that
                    # looks like a triumph is worse than a `not_run`.
                    if not plan.span_is_noul[k]:
                        pointing_n += 1
                        pointing_start_hit += 1 if hit_start else 0
                        # BOTH pointers, or the pair is a mixture of one decomposed number
                        # and one inflated one. Measured on the four-class v2 arm before this
                        # was fixed: start 1.6% (pointing rows) printed beside end 17.9% (all
                        # rows), which reads as a head that finds the end of a span it cannot
                        # find the start of, and is really just the abstentions showing
                        # through on one of the two.
                        pointing_end_hit += 1 if hit_end else 0
                        pointing_chance += 1.0 / max(1, int(start_rows[k].numel()))
    model.train()
    return {
        "choice_top1": choice_hit / choice_n if choice_n else 0.0,
        "choice_n": choice_n,
        "span_start_top1": start_hit / span_n if span_n else 0.0,
        "span_end_top1": end_hit / span_n if span_n else 0.0,
        "span_n": span_n,
        "span_chance": span_chance / span_n if span_n else 0.0,
        # The same three quantities over POINTING rows only. Added beside the undecomposed
        # fields rather than replacing them: the 61 capacity rows and the learning curve were
        # written with the mixture, and silently redefining it would make every new row
        # incomparable with the rows it extends. The GATE reads these.
        "span_pointing_n": pointing_n,
        "span_pointing_start_top1": pointing_start_hit / pointing_n if pointing_n else 0.0,
        "span_pointing_end_top1": pointing_end_hit / pointing_n if pointing_n else 0.0,
        "span_pointing_chance": pointing_chance / pointing_n if pointing_n else 0.0,
        "choice_probs": choice_probs,
        "choice_gold": choice_gold,
    }


def degenerate_head_state(rows: Sequence[Sequence[float]]) -> TriState:
    """``degenerate_head_check`` over the held-out choice distributions.

    Ragged input is refused rather than padded. Bucketed batching can in principle hand
    back rows of different widths, and padding them to a common width with zeros would
    invent probability mass the model never emitted -- lowering the measured entropy and
    making a healthy head look degenerate. A check that reports the wrong verdict is worse
    than one that reports NotRun, so the ragged case says so and names the widths.
    """
    if not rows:
        return NotRun(reason="the degenerate-head check had no held-out rows to read")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        return NotRun(
            reason=(
                f"held-out choice distributions are ragged ({sorted(widths)} columns), and "
                "padding them to a common width would invent probability mass the model "
                "never emitted, understating the entropy this check thresholds on"
            )
        )
    return degenerate_head_check(np.asarray(rows, dtype=np.float64))


def sattolo_permutation(n: int, *, seed_text: str) -> list[int]:
    """A uniformly random CYCLIC permutation of ``range(n)`` -- Sattolo's algorithm.

    ``result[new_position] = old_index``.

    Sattolo rather than Fisher-Yates, and the difference is the whole gate.
    ``docs/schema-api.md`` states it: *"The permutation must be a derangement -- no option
    may keep its position. A uniform shuffle (Fisher-Yates) leaves fixed points, and when
    the winning row happens to be one, a purely position-biased model agrees with itself
    across both passes and the abstention never fires. The check then passes precisely on
    the cases it exists to catch."* And, for this gate specifically: *"a
    permutation-consistency figure measured over uniform shuffles is measuring something
    weaker than the runtime enforces, so the >= 95% gate must be computed over derangements
    or it is not the same quantity."*

    The single difference in code is ``randrange(i)`` rather than ``randrange(i + 1)``,
    which is why it is worth a docstring this long: the two algorithms differ by one
    character and produce gates that measure different things.

    Seeded per example rather than from a sweep-wide RNG, so the permutation a row got is
    reproducible from the row alone -- the training-gate analogue of the runtime seeding
    its permutation from ``DecisionRequest::digest()``.
    """
    if n < 2:
        raise ValueError(f"a derangement needs at least 2 items, got {n}")
    rng = random.Random(hashlib.sha256(seed_text.encode("utf-8")).digest())
    items = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)  # NOT randrange(i + 1): that is Fisher-Yates and has fixed points
        items[i], items[j] = items[j], items[i]
    return items


#: The gate from `docs/schema-api.md`: "permutation consistency (>= 95%) is a training gate
#: and not only a runtime check: a model that fails it makes the second pass fire constantly
#: and the abstain rate blows the cap." A threshold from the plan, read-only to an agent.
PERMUTATION_CONSISTENCY_FLOOR: Final[float] = 0.95


#: The linear control's iteration budget, which must match `rung0_linear_control`'s.
#: Restated rather than imported only because that module imports this one; the value is
#: asserted equal to its source in `test_shuffled_label_control.py`, so the two cannot
#: drift. Above the library default of 500, which does NOT converge on this corpus -- and
#: an unconverged control is refused, so the budget is what makes the gate reachable at
#: all. The ITERATION BUDGET moves, never the tolerance: the tolerance is what makes the
#: control worth beating.
LINEAR_CONTROL_MAX_ITER: Final[int] = 6_000

#: The linear control is fitted once per sweep, on CPU, with the GPU doing nothing. On the
#: rung-0 corpus that is 38s and unremarkable. On the commitpackft corpus it projects to
#: ~150 hours, and an arm that entered it would be terminated by its own wall-clock cap
#: having written no rows -- which is precisely the 2026-09-21 failure, whose whole cost
#: was that a slow fit and a hang are indistinguishable from outside. 15 minutes is well
#: above the measured 38s and far below anything that could hide.
LINEAR_CONTROL_TIME_BUDGET_S: Final[float] = 900.0


def permutation_consistency(
    model: Rung0Model, plans: Sequence[BatchPlan], *, device: str
) -> TriState:
    """Does the choice head give the same answer when the options are deranged?

    Two passes over the same rows, the second with each row's live options cyclically
    permuted, the answer mapped back to the original option index and compared. This is the
    training-gate half of what the runtime does per request, where disagreement becomes
    ``noul``; a model that fails it makes the second pass fire constantly and the abstain
    rate blows its cap.

    Rows with fewer than two live options are EXCLUDED rather than counted as agreeing.
    They have no derangement, so they cannot disagree, and scoring them as agreements would
    inflate the rate with rows that were never asked the question. They are carried in the
    coverage pair instead, so a corpus that quietly became mostly single-option rows shows
    up as a shrinking denominator rather than as a rising score.
    """
    model.eval()
    agree = 0
    asked = 0
    total = 0
    with torch.no_grad():
        for plan in plans:
            ctx = torch.tensor(plan.context_ids, dtype=torch.long, device=device)
            mask = torch.tensor(plan.context_mask, dtype=torch.bool, device=device)
            hidden = model.decider.encode_context(ctx, mask)

            option_ids = [list(row) for row in plan.option_ids]
            option_mask = [list(row) for row in plan.option_mask]
            n_live = list(plan.n_live_options)
            total += len(n_live)

            perms: list[list[int] | None] = []
            permuted_ids = []
            permuted_mask = []
            for i, live in enumerate(n_live):
                if live < 2:
                    perms.append(None)
                    permuted_ids.append(option_ids[i])
                    permuted_mask.append(option_mask[i])
                    continue
                perm = sattolo_permutation(live, seed_text=plan.example_ids[i])
                perms.append(perm)
                # Only the live prefix moves; padded columns stay where they are, because
                # they are not options and the model is told so by `n_live_options`.
                ids = [option_ids[i][perm[j]] for j in range(live)] + option_ids[i][live:]
                msk = [option_mask[i][perm[j]] for j in range(live)] + option_mask[i][live:]
                permuted_ids.append(ids)
                permuted_mask.append(msk)

            first = model.decider.score_from_hidden(
                hidden, mask,
                torch.tensor(plan.option_ids, dtype=torch.long, device=device),
                torch.tensor(plan.option_mask, dtype=torch.bool, device=device),
                torch.tensor(n_live, dtype=torch.long, device=device),
            ).argmax(dim=1).tolist()
            second = model.decider.score_from_hidden(
                hidden, mask,
                torch.tensor(permuted_ids, dtype=torch.long, device=device),
                torch.tensor(permuted_mask, dtype=torch.bool, device=device),
                torch.tensor(n_live, dtype=torch.long, device=device),
            ).argmax(dim=1).tolist()

            for i, perm in enumerate(perms):
                if perm is None:
                    continue
                asked += 1
                # `second[i]` is a position in the PERMUTED order; perm maps it back.
                if second[i] < len(perm) and perm[second[i]] == first[i]:
                    agree += 1
    model.train()

    if asked == 0:
        return NotRun(
            reason=(
                f"no row of {total} had two or more live options, so no derangement exists "
                "and permutation consistency was not measured on anything"
            )
        )
    rate = agree / asked
    return Ran(
        passed=rate >= PERMUTATION_CONSISTENCY_FLOOR,
        value=rate,
        n=agree,
        n_total=asked,
        detail=(
            f"the choice head agreed with itself across a derangement on {agree} of "
            f"{asked} rows ({rate:.1%}) against a {PERMUTATION_CONSISTENCY_FLOOR:.0%} "
            f"floor; {total - asked} row(s) had fewer than two live options and were "
            "excluded rather than counted as agreeing"
        ),
    )


def quick_reason_for(
    *,
    examples_supplied: bool,
    seeds: int,
    train_subsample: float,
    shuffled: bool,
    pool_records: int | None,
    corpus_hash: str,
) -> str:
    """Why this row is `quick`, from the run's own facts rather than a fixed sentence.

    The sentence this replaces read *"the corpus is this repository's own sources rather
    than the pool the plan names"*. It was true of every row ever written, because no run
    had ever been given another corpus. The commitpackft corpus exists now, and a run on it
    would have recorded that sentence anyway -- a false statement, in an append-only
    ledger, about the single thing rule 8 turns on.

    The flag itself stays `True` regardless. Rule 2 makes promotion a human's decision and
    not an agent's, and "the reason no longer applies" is precisely the argument an agent
    should not be able to make on its own behalf. What changes is that the row now says
    which conditions actually stand, so the human deciding has the facts rather than a
    sentence that outlived them.

    One condition is deliberately NOT inferred: whether a supplied corpus is the pool the
    plan names. This run cannot verify that -- `--examples` accepts any file -- so it
    records the snapshot hash and the pool's record count and says plainly that it cannot
    tell. Asserting it from a filename or a record count would be a guess wearing the
    clothes of a check.
    """
    reasons: list[str] = []
    if not examples_supplied:
        reasons.append(
            "the corpus was built from this repository's own tracked sources rather than "
            "the pool the plan names, which rule 8 counts as a subsample"
        )
    if seeds < 3:
        reasons.append(f"{seeds} seed(s), fewer than the 3 rule 8 requires")
    if train_subsample < 1.0:
        reasons.append(
            f"the training set is subsampled to {train_subsample:.0%} of the split"
        )
    if shuffled:
        reasons.append(
            "the training labels were permuted, so this row is a control and not a "
            "measurement of the model"
        )

    corpus = (
        f"data_snapshot {corpus_hash[:16]}"
        + (f", {pool_records} pool record(s)" if pool_records is not None else "")
    )
    if reasons:
        return "; ".join(reasons) + f" ({corpus})"
    return (
        "no rule-8 condition this run can determine stands: "
        f"{seeds} seeds, full schedule, no subsample, real labels, corpus supplied as a "
        f"pre-generated set ({corpus}). Recorded quick nonetheless -- whether that corpus "
        "is the pool the plan names is not something this run can verify, and clearing "
        "the flag is a promotion decision that rule 2 makes a human's, not an agent's."
    )


def linear_baseline_correctness(
    train_d: Sequence,
    scored_val: Sequence,
    *,
    seed: int,
    max_iter: int,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray | None, TriState | None]:
    """Fit the linear control and return which of the scored rows it got right.

    Returns ``(correct, None)`` on success and ``(None, NotRun)`` when the control did not
    converge -- which is the control's own rule, from ``rung0_linear_control``: *"A model
    cannot beat a baseline that never finished training, and scoring it as though it had is
    how a weak control manufactures a win."* An unconverged fit produces a weak opponent
    and therefore a flattering margin, so the gate reports NotRun rather than a win.

    Fitted ONCE per sweep rather than once per seed: the baseline is a property of the
    split, not of the model's initialisation, and refitting it 24 times would cost real
    minutes to recompute an identical array.

    ``context_texts`` is imported here rather than at module scope because
    ``rung0_linear_control`` imports *this* module, so a top-level import is a cycle. It is
    imported rather than copied for a reason that outranks the awkwardness: the paired
    margin is only meaningful if both arms saw the same bytes, and a second rendering here
    would be free to drift into scoring a different task. The dependency direction is worth
    straightening eventually -- both tools want one renderer and neither owns it -- and
    that is `GAP-CONTEXT-TEXTS-HAS-NO-OWNER`.
    """
    from rung0_linear_control import context_texts

    train_docs, train_labels = context_texts(train_d)
    val_docs, val_labels = context_texts(scored_val)

    model = LinearBaseline(seed=seed, max_iter=max_iter)
    key = control_key(
        train_docs=train_docs,
        train_labels=train_labels,
        val_docs=val_docs,
        seed=seed,
        max_iter=max_iter,
        hasher_params=(model.hasher.n_min, model.hasher.n_max, model.hasher.dim),
        l2_grid=model.l2_grid,
        tol=model.tol,
        lr=model.lr,
    )
    if cache_dir is not None:
        hit = load_control(cache_dir, key, expected_n=len(val_docs))
        if hit is not None:
            print(
                f"  linear control: CACHE HIT {key[:16]} -- fitted in {hit.fitted_s:.1f}s "
                f"at {hit.fitted_at} on {hit.n_train} doc(s), L2 {hit.l2:g}, "
                f"{hit.iterations} iteration(s). No GPU time was spent waiting for it."
            )
            return hit.correct, None

    projected_s = model.projected_fit_seconds(
        train_docs, n_classes=len(set(train_labels))
    )
    if projected_s > LINEAR_CONTROL_TIME_BUDGET_S:
        return None, NotRun(
            reason=(
                f"the paired margin has no opponent: fitting the linear control on "
                f"{len(train_docs)} training document(s) projects to {projected_s / 3600:.1f} "
                f"hours, over the {LINEAR_CONTROL_TIME_BUDGET_S / 60:.0f} minute budget. "
                "The control is REFUSED rather than attempted: a run that disappears into "
                "an unbounded CPU fit with the GPU idle looks exactly like a hung one, "
                "which is what happened on 2026-09-21 and cost a whole arm. This is a "
                "statement about the control's cost at this corpus size, not about the "
                "model -- no margin was measured, and none may be inferred. Fit it once "
                "off the GPU with tools/fit_linear_control.py, which writes the cache this "
                "run just missed, and the gate reports on the next run."
            ),
        )
    _fit_started = time.monotonic()
    model.fit(train_docs, train_labels)
    fitted_s = time.monotonic() - _fit_started
    convergence = model.convergence()
    if not (isinstance(convergence, Ran) and convergence.passed):
        reason = (
            convergence.reason
            if isinstance(convergence, NotRun)
            else f"the linear control did not converge ({convergence.detail})"
        )
        return None, NotRun(
            reason=(
                f"the paired margin has no opponent: {reason}. A model cannot beat a "
                "baseline that never finished training, and scoring it as though it had "
                "is how a weak control manufactures a win."
            )
        )

    predicted = model.predict(val_docs)
    correct = np.asarray(
        [p == gold for p, gold in zip(predicted, val_labels, strict=True)], dtype=bool
    )
    if cache_dir is not None:
        fit = model.fit_
        assert fit is not None  # convergence() above already refused an unfitted model
        store_control(
            cache_dir,
            key,
            correct,
            fitted_s=fitted_s,
            n_train=len(train_docs),
            l2=fit.l2,
            iterations=fit.iterations,
            final_grad_norm=fit.final_grad_norm,
        )
    return correct, None


def ece_state(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> TriState:
    """``ece_gate`` over the held-out distributions, refusing the same ragged case.

    Kept beside :func:`degenerate_head_state` and refusing on the same grounds: zero-padding
    a short row would make the model look MORE confident than it was, and confidence is
    exactly what a calibration error measures. The length disagreement is checked too --
    probabilities and golds that have drifted apart would silently score the wrong pairs.
    """
    if not rows:
        return NotRun(reason="the ECE gate had no held-out rows to read")
    if len(rows) != len(labels):
        return NotRun(
            reason=(
                f"{len(rows)} held-out distribution(s) against {len(labels)} gold label(s); "
                "these must be the same examples in the same order or the gate scores "
                "predictions against the wrong answers"
            )
        )
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        return NotRun(
            reason=(
                f"held-out choice distributions are ragged ({sorted(widths)} columns), and "
                "padding them would overstate the model's confidence, which is the quantity "
                "a calibration error is about"
            )
        )
    return ece_gate(
        np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=int)
    )


def _accuracy_gate(
    measured: float, baseline: float, *, n: int, what: str, baseline_name: str
) -> TriState:
    """Whether the model beat the trivial answer, carrying both numbers.

    ``baseline`` is passed rather than assumed because the trivial answer is not the same
    for the two heads, and getting that wrong is how a head that learned nothing reports as
    passing. For the CHOICE head it is the majority class: ``stub`` is 51.5% of this corpus,
    so answering one constant scores 51.5%. For the SPAN head it is **chance over the
    candidate line starts**, 1/candidates -- not 0. An earlier version of this tool passed
    0.0 for the span baseline and duly recorded ``passed=True`` for a span head scoring
    0.6%, which is the exact failure CLAUDE.md names: a check that could not distinguish
    anything reporting like one that ran and passed.

    ``NotRun`` on an empty evaluation set rather than 0.0: nothing was measured, and a 0%
    reading would look like a model that failed rather than a measurement that did not
    happen.
    """
    if n == 0:
        return NotRun(
            reason=(
                f"no {what} rows were evaluated, so there is no accuracy to compare against "
                f"the {baseline_name}"
            )
        )
    return Ran(
        passed=measured > baseline,
        value=measured,
        n=round(measured * n),
        n_total=n,
        detail=(
            f"{what} top-1 {measured:.1%} of {n} held-out rows, against a {baseline_name} "
            f"of {baseline:.1%}. The gap is {measured - baseline:+.1%}; a model at "
            "the baseline has learned the prior and nothing else."
        ),
    )


def _fit_gate(measured: float, train_majority: float, *, n: int) -> TriState:
    """Whether the choice head beat a constant predictor on its OWN training data.

    Deliberately NOT ``_accuracy_gate`` against the training set, though the arithmetic is
    the same comparison. The two answer different questions and a failure means opposite
    things, so they carry opposite verdicts:

    * ``_accuracy_gate`` asks **did it generalise**. At the baseline, the model fitted
      something that did not transfer, and the corpus is the suspect.
    * this asks **did it fit at all**. At the training majority share the head is a
      constant predictor -- it has not learned one conditional thing about a single row it
      was optimised on -- and no held-out number from that run is evidence about
      generalisation, because there was nothing to generalise.

    Collapsing them would report the second as the first, which is what happened here for
    four experiments. Rung 0's held-out accuracy sat exactly at the baseline and was read
    as "fits but does not transfer" on the strength of a total loss that fell 90%. Train
    accuracy was not being measured; when it was, it equalled the training majority share
    to the row, on every seed. ``span_weight`` is why -- at 1.0 the span channel outweighs
    the choice channel 6.2:1 in an unweighted sum, and at >= 0.5 this gate fails on every
    seed while below 0.2 it passes. See ``AUDIT/rung0-span-weight-2026-09-20.json``.

    The margin is exact equality, not a tolerance. A head that emits one class scores the
    majority share to the row, and any real fit clears it by whole percentage points; a
    tolerance here would only blur the one signal the gate exists to give.
    """
    if n == 0:
        return NotRun(
            reason=(
                "no training rows were evaluated, so whether the choice head fitted them "
                "is unmeasured -- which is not the same as its having failed to"
            )
        )
    return Ran(
        passed=measured > train_majority,
        value=measured,
        n=round(measured * n),
        n_total=n,
        detail=(
            f"choice top-1 {measured:.1%} of {n} rows the model TRAINED on, against a "
            f"training-set majority share of {train_majority:.1%}. The gap is "
            f"{measured - train_majority:+.1%}. At or below it the head is a constant "
            "predictor and the run's held-out number says nothing about generalisation."
        ),
    )


def _channel_ratio(run: dict[str, object]) -> str:
    """The span:choice loss ratio at the first step, as a phrase for the summary.

    Guarded rather than divided, because a choice loss of exactly 0.0 at step one is not
    impossible and a ZeroDivisionError raised while explaining a failed run would replace
    the explanation with a traceback.
    """
    span = float(run["span_first"])  # type: ignore[arg-type]
    choice = float(run["choice_first"])  # type: ignore[arg-type]
    if choice <= 0.0:
        return f"span {span:.3f} against a choice loss of {choice:.3f}, which has no ratio"
    return f"span {span:.3f} against choice {choice:.3f}, a {span / choice:.1f}:1 ratio"


# -- the run -----------------------------------------------------------------------------


def run_cost_estimate(
    *,
    cap: WallClockCap,
    device: str,
    instance: str | None,
    usd_per_hour: float | None,
    usd_per_gpu_hour: float | None,
) -> CostEstimate:
    """What this run costs, in one place.

    Two things need the answer and must not be able to disagree: the ``RunControl`` that
    gates the launch under rule 4, and the ledger row that records what was spent. The row
    used to read a separate rate that defaulted to zero, which is how 13 GH200 rows came to
    price a real GPU hour at $0.00.

    It is a function rather than a value carried in the run dict because that dict is a
    record of JSON-serialisable facts -- ``tools/rung0_toy_run.py`` prints its equivalent as
    a report -- and a ``CostEstimate`` in it is a TypeError waiting for whoever adds the
    next ``json.dumps``. It was one here: this function exists because that serialisation
    broke on the first end-to-end run after the cost wiring landed.

    ``for_device`` prices cpu and mps at zero -- a Mac already bought costs nothing per hour
    -- and refuses to invent a rate for anything else.
    """
    return CostEstimate.for_device(
        cap=cap,
        device=device,
        n_gpus=n_gpus_for_device(device),
        usd_per_hour=usd_per_hour,
        usd_per_gpu_hour=usd_per_gpu_hour,
        instance=instance,
    )


def train_once(
    *,
    train_plans: Sequence[BatchPlan],
    val_plans: Sequence[BatchPlan],
    baseline: float,
    choice_floor: float,
    train_decisions: int,
    span_weight: float,
    peak_lr: float,
    device: str,
    seed: int,
    epochs: int,
    config: ByteDeciderConfig,
    instance: str | None,
    usd_per_hour: float | None,
    usd_per_gpu_hour: float | None,
    approved_by: str,
) -> dict[str, object]:
    """One seed end to end: train on the train files, measure on the validation files."""
    torch.manual_seed(seed)
    steps = epochs * len(train_plans)
    model = Rung0Model(config).to(device)
    # `Rung0Step` has always taken a span_weight and this tool has always left it at 1.0,
    # which is an unweighted sum of two channels whose scales differ by 6.2:1 at
    # initialisation (span 12.520, choice 2.026). Under that sum the optimiser serves the
    # pointer, and `GAP-RUNG0-TOTAL-LOSS-IS-THE-SPAN-CHANNELS-LOSS` is the same fact seen
    # from the reporting side. Exposed so the ratio is a measurement rather than a default.
    step = Rung0Step(model, span_weight=span_weight)
    cap = WallClockCap(cap_s=RUN_CAP_S)
    control = RunControl(
        schedule=LRSchedule(
            peak_lr=peak_lr, warmup_steps=max(1, steps // 10), total_steps=steps
        ),
        cap=cap,
        # `for_device` prices cpu and mps at zero -- a Mac that is already bought costs
        # nothing per hour -- and refuses to invent a rate for anything else. The literal
        # this replaces applied the Mac's price to whatever `--device` named, and `--device`
        # is a free string: every run this tool made on the rented GH200 recorded
        # `instance="local-cuda"` on `n_gpus=0` at `usd_per_hour=0.0`. Not an under-report
        # of a cost -- an assertion that the machine was a local one with no GPUs in it.
        cost=run_cost_estimate(
            cap=cap, device=device, instance=instance,
            usd_per_hour=usd_per_hour, usd_per_gpu_hour=usd_per_gpu_hour,
        ),
        approved_by=approved_by,
        grad_accum=1,
    )

    before = evaluate(model, val_plans, device=device)
    t0 = time.monotonic()
    result = train_rung0(
        cycle(train_plans, n=steps), step=step, control=control, seed=seed, max_batches=steps
    )
    wall = time.monotonic() - t0
    after = evaluate(model, val_plans, device=device)
    # The same measurement on the TRAINING batches. A held-out number at the baseline has
    # two possible causes and they call for opposite fixes: a model that also scores the
    # baseline on its own training data has not fitted anything and needs capacity or a
    # longer schedule, while one that scores far above it on training and at it on held-out
    # has fitted and cannot transfer, which is a corpus problem. The loss curve alone does
    # not separate these, because a falling loss is consistent with both.
    on_train = evaluate(model, train_plans, device=device)
    # The second pass, on the held-out rows only. Computed here because it needs the model
    # and the model does not leave this function.
    permutation = permutation_consistency(model, val_plans, device=device)

    waste_num = sum(
        sum(1 for row in p.context_mask for live in row if not live) for p in train_plans
    )
    waste_den = sum(p.batch_size * p.context_width for p in train_plans)
    return {
        "device": device,
        "seed": seed,
        "epochs": epochs,
        "steps": steps,
        "optimizer_steps": result.optimizer_steps,
        "termination": result.termination,
        "params": sum(p.numel() for p in model.parameters()),
        "train_batches": len(train_plans),
        "val_batches": len(val_plans),
        "total_first": result.loss_log.losses()[0],
        "total_last": result.loss_log.losses()[-1],
        "choice_first": result.choice_log[0],
        "choice_last": result.choice_log[-1],
        "span_first": result.span_log[0],
        "span_last": result.span_log[-1],
        "val_before": before,
        "val_after": after,
        "train_after": on_train,
        "permutation_consistency": permutation,
        "baseline": baseline,
        "choice_floor": choice_floor,
        "train_decisions": train_decisions,
        "train_padding_waste": waste_num / waste_den if waste_den else 0.0,
        "wall_clock_s": wall,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="scratch directory for the corpus")
    parser.add_argument("--rev", required=True, help="the revision the corpus is read at")
    parser.add_argument("--device", default="mps")
    # The price of the device above. `--device` is a free string and this tool priced
    # whatever it named at a Mac's rate, so these are required with cuda and refused at
    # argv time without them.
    parser.add_argument(
        "--instance",
        help=(
            "what machine this is, as a price list names it -- e.g. 'lambda-1xGH200'. "
            "REQUIRED with --device cuda: a cuda device is rented by the hour, and the "
            "zero-rate default it used to get made requires_human_approval False for any cap"
        ),
    )
    parser.add_argument(
        "--usd-per-hour",
        type=float,
        help="the WHOLE instance's rate. Required with --device cuda",
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
    parser.add_argument("--seeds", type=int, default=3)
    # Pre-registration. A sweep that reports "no effect" without these two numbers has not
    # said whether it could have seen one, and the 4096 capacity arms are what that costs:
    # 256x4 came in at -0.39pp against a floor of 1.09pp, which is not a null, it is a
    # measurement that never had the resolution to be one.
    parser.add_argument(
        "--prior-sd",
        type=float,
        help=(
            "seed sd from a PREVIOUS measurement at this configuration, used to state what "
            "--seeds can resolve. From somewhere other than this sweep: a run that "
            "estimates its own sensitivity from the numbers it is about to interpret has "
            "graded its own exam. And not from an arm that COLLAPSED -- the 4096 256x4 and "
            "512x6 arms sat at or below the training majority on 6 and 7 of 8 seeds, and "
            "the spread of a model that never fit is not the spread of one that did"
        ),
    )
    parser.add_argument(
        "--target-difference",
        type=float,
        help=(
            "the smallest effect worth detecting, in the same units as the accuracy it is "
            "compared against. Stated up front so the row can say whether this sweep could "
            "see it; without it the resolution metric records NotRun rather than nothing"
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=400)
    parser.add_argument("--limit", type=int, default=4000, help="examples qd-mutate may emit")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--context-bytes",
        type=int,
        default=DEFAULT_CONTEXT_BYTES,
        help="the byte window; see DEFAULT_CONTEXT_BYTES for what each width keeps",
    )
    # Capacity, as flags rather than constants, because the 8192 run settled that context is
    # NOT the binding constraint: it nearly doubled the usable corpus and moved the held-out
    # number by 0.0%. Width and depth are what remains untested.
    parser.add_argument(
        "--train-subsample",
        type=float,
        default=1.0,
        help=(
            "fraction of TRAINING FILES to keep, for a learning curve. The validation side "
            "is never subsampled, so every point is scored on the same rows"
        ),
    )
    parser.add_argument(
        "--hold-out-operator",
        default="",
        help=(
            "remove every row this mutation operator produced from TRAINING, leaving them "
            "in validation, and report accuracy on them separately. The linear control "
            "collapses under this -- stub.panic 99.10%% to 3.08%%, cosmetic.rename_local "
            "58.71%% to 0.00%% (AUDIT/operator-holdout.md) -- which says it classifies by "
            "recognising the generator rather than by reading the change. Whether the model "
            "does the same is the open half of that finding"
        ),
    )
    parser.add_argument(
        "--drop-random-train",
        type=int,
        default=0,
        help=(
            "the size-matched control for --hold-out-operator: drop this many training rows "
            "at random instead of an operator's. Without it a collapse is explained as well "
            "by the smaller training set as by the missing fingerprint"
        ),
    )
    parser.add_argument("--drop-random-seed", type=int, default=0)
    parser.add_argument(
        "--measure-operator",
        default="",
        help=(
            "report accuracy on this operator's validation rows WITHOUT removing it from "
            "training. --hold-out-operator implies it, so the holdout arm needs only that "
            "flag; the size-matched control arm needs this one, because otherwise the two "
            "arms report different quantities and the comparison the holdout exists to "
            "make cannot be made. Validation is never filtered, so both arms score the "
            "identical rows and the difference between them is the operator's absence"
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_PEAK_LR,
        help=(
            "peak of the LR schedule. A flag because the capacity sweep on the diff task "
            "found accuracy falling monotonically in width -- 76.3%% at 128, 70.9%% at 256, "
            "58.1%% at 512 -- with the seed spread exploding from 5.1 to 29.7 points. That "
            "is the signature of an optimiser over-stepping, not of a model short of "
            "capacity, and it could not be told apart while this was a literal. It is in "
            "the recipe, so two runs at different rates cannot hash alike"
        ),
    )
    parser.add_argument(
        "--span-weight",
        type=float,
        default=1.0,
        help=(
            "multiplier on the span loss in the summed objective. Must be positive -- "
            "Rung0Step refuses zero, because that trains the span head on nothing while its "
            "loss still appears in the log"
        ),
    )
    # The control that has never run. `eval_harness.shuffled_label_control` has existed all
    # along and every one of 988 ledger rows records `shuffled_label` as not_run, because
    # the harness needs the accuracy of a model trained on destroyed labels and nothing
    # trained one. These two flags are that model.
    parser.add_argument(
        "--shuffle-train-labels",
        action="store_true",
        help=(
            "permute the gold labels among the TRAINING decisions and evaluate on the "
            "untouched validation split, recording the `shuffled_label` control on the "
            "row. A run with this flag is the control, not a measurement of the model: "
            "its accuracy is expected at or below the held-out majority rate, and an "
            "accuracy ABOVE that ceiling means the split leaks and every other number "
            "measured on it is worthless"
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help=(
            "seed for the label permutation, separate from --seeds so a sweep can vary "
            "model init while holding the destroyed labelling fixed, which is what makes "
            "the seed spread attributable to init rather than to a different shuffle"
        ),
    )
    parser.add_argument(
        "--context-source",
        choices=CONTEXT_SOURCES,
        default=CONTEXT_AFTER,
        help=(
            "what the model reads. after is the post-image and is what every row "
            "before 2026-09-22 was measured on; diff is the unified diff, which is "
            "what the plan says the model reads. The choice head is asked what kind "
            "of change this is, and for 54 percent of the commitpackft corpus the "
            "post-image carries no evidence a change happened at all -- measured on "
            "the same control and the same rows, post-image 55.8 percent against the "
            "diff 93.8 percent. Encoding the diff gives the span head NO target, "
            "because span offsets are into after and every byte has moved"
        ),
    )
    parser.add_argument(
        "--span-in-diff",
        action="store_true",
        help=(
            "supervise the span head in --context-source diff by resolving each span into "
            "the diff's own byte space, instead of giving the head nothing. Span offsets "
            "are into `after`, and encoding the diff moves every byte, so without this "
            "there is NO mode in which both heads are supervised: after mode has a choice "
            "task where 54%% of the corpus shows no evidence a change happened, and diff "
            "mode has a span head with no target -- which is what 'span NOT MEASURED (all "
            "N scored rows abstain)' has been reporting. Verified before it was wired: "
            "over both corpora, 83456 of 83456 span endpoints resolve to the exact source "
            "line, none mismatched and none outside a hunk. Off by default because the "
            "span gradient reaches the trunk, so a run with it is a different protocol "
            "from the diff rows already in the ledger"
        ),
    )
    parser.add_argument(
        "--control-cache",
        type=Path,
        default=None,
        help=(
            "directory holding fitted linear-control verdicts. The control is a property "
            "of the corpus and the split, not of the model, so it is identical across "
            "every seed of a sweep and across sweeps on the same corpus -- and on the "
            "commitpackft corpus fitting it costs hours of CPU during which the GPU is "
            "idle. Warm it off the GPU with tools/fit_linear_control.py. Omitted means no "
            "cache: the control is fitted in-process if it fits the time budget, and "
            "reported NotRun if it does not"
        ),
    )
    parser.add_argument("--width", type=int, default=ByteDeciderConfig().width)
    parser.add_argument("--layers", type=int, default=ByteDeciderConfig().n_layers)
    parser.add_argument("--heads", type=int, default=ByteDeciderConfig().n_heads)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "run under torch.use_deterministic_algorithms(True). MEASURED on a GH200 "
            "2026-09-21: without it, two runs at seed 0 with every other input identical "
            "scored 50.3%% and 53.1%% on the same held-out rows -- 2.8 points apart at a "
            "FIXED seed, which is half the size of the whole signal this corpus contains. "
            "A per-seed number from a run without this is a draw, not a measurement, and a "
            "sweep comparing points cannot attribute a difference of a few points to the "
            "thing it varied."
        ),
    )
    parser.add_argument("--val-share", type=float, default=DEFAULT_VAL_SHARE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO / "target" / "release" / "qd-mutate",
        help="the qd-mutate binary",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        help=(
            "a qd-mutate examples.jsonl generated elsewhere. Requires --manifest-in. "
            "Generating a corpus needs the Rust binary and training needs a GPU, and those "
            "are not always the same machine"
        ),
    )
    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="the manifest.json beside --examples; it is what data_snapshot_hash is taken from",
    )
    args = parser.parse_args(argv)

    # Decided from argv, before the corpus is built and before any GPU time is spent.
    # `CostEstimate.for_device` refuses the same case, but it is reached per-seed inside
    # `train_once`, which is after the work has started on a rented box.
    if args.device == "cuda" and (args.instance is None or args.usd_per_hour is None):
        raise SystemExit(
            "--device cuda needs --instance and --usd-per-hour. A cuda device is hardware "
            "rented by the hour; the zero-rate, zero-GPU default that stood in for a price "
            "makes requires_human_approval False for ANY cap and skips the per-GPU column "
            "check, so the two values that look like harmless defaults are the two that "
            "turn rule 4 off. It also recorded the box as instance='local-cuda', a machine "
            "that does not exist. Example: --instance lambda-1xGH200 --usd-per-hour 1.49"
        )

    if args.deterministic:
        # Before the corpus is built and long before any model runs, so nothing has touched
        # a nondeterministic kernel by the time this takes effect. torch RAISES rather than
        # falling back, which is the property that makes a completed run evidence: every op
        # this model uses had a deterministic implementation, rather than "we asked and
        # something quietly said no".
        torch.use_deterministic_algorithms(True)

    if not 1 <= args.epochs <= MAX_EPOCHS:
        raise SystemExit(f"--epochs must be in [1, {MAX_EPOCHS}], got {args.epochs}")
    if not 1 <= args.limit <= MAX_EXAMPLES:
        raise SystemExit(f"--limit must be in [1, {MAX_EXAMPLES}], got {args.limit}")
    if not 1 <= args.context_bytes <= MAX_CONTEXT_BYTES:
        raise SystemExit(
            f"--context-bytes must be in [1, {MAX_CONTEXT_BYTES}], got {args.context_bytes}"
        )
    # Checked here rather than left to `Rung0Step`, which is constructed inside the seed
    # loop: by then the corpus has been generated and split, and on the GH200 that is
    # minutes of work thrown away to reach a refusal that was decidable from argv.
    if not args.span_weight > 0.0:
        raise SystemExit(
            f"--span-weight must be positive, got {args.span_weight}; zero would train the "
            "span head on nothing while its loss still appeared in the log"
        )
    # Both or neither. A corpus without its manifest has no data_snapshot_hash, and a run
    # that invented one would put a fabricated value in the protocol every later comparison
    # is made against.
    if bool(args.examples) != bool(args.manifest_in):
        raise SystemExit(
            "--examples and --manifest-in go together: the examples are what is trained on "
            "and the manifest is what the protocol's data_snapshot_hash is taken from. One "
            "without the other either trains on an unidentified corpus or identifies a "
            "corpus it did not train on."
        )
    # ByteDeciderConfig.__post_init__ refuses a width that does not divide by n_heads and
    # any non-positive dimension, so a bad sweep point fails here rather than producing a
    # model with a silently dropped head's worth of capacity.
    config = ByteDeciderConfig(
        max_context_bytes=args.context_bytes,
        width=args.width,
        n_layers=args.layers,
        n_heads=args.heads,
    )

    if args.examples:
        print(f"corpus: reading a pre-generated set from {args.examples}")
        examples = [
            json.loads(line)
            for line in args.examples.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(args.manifest_in.read_text(encoding="utf-8"))
        if not examples:
            raise SystemExit(f"{args.examples} holds no examples; there is nothing to train on")
    else:
        if not args.binary.exists():
            raise SystemExit(
                f"{args.binary} does not exist; cargo build --release -p qd-mutate, or pass "
                "--examples/--manifest-in to use a set generated on another machine"
            )
        print(f"corpus: reading tracked sources at {args.rev}")
        pool, capped = build_pool(rev=args.rev, max_files=args.max_files)
        print(f"  pool: {len(pool)} file(s){' (CAPPED)' if capped else ''}")
        examples, manifest = generate_examples(
            pool, out_dir=args.out, seed=0, limit=args.limit, binary=args.binary
        )
    print(f"  {len(examples)} example(s)")

    # Before a byte of training: in diff mode, refuse a corpus in which an empty context
    # names a class. Raised here rather than discovered row by row inside decisions_of,
    # because the diagnosis is about the corpus and a rented GPU should not pay for the
    # parse to reach the same conclusion one example at a time.
    refuse_leaky_diff_corpus(examples, context_source=args.context_source)
    print(f"  context source: {args.context_source}")

    train_raw, val_raw = split_by_file(examples, val_share=args.val_share)
    train_paths = {e.get("function", {}).get("path") for e in train_raw}  # type: ignore[union-attr]
    val_paths = {e.get("function", {}).get("path") for e in val_raw}  # type: ignore[union-attr]
    overlap = train_paths & val_paths
    if overlap:
        raise SystemExit(f"the split is not file-disjoint: {sorted(overlap)[:3]}")
    print(
        f"  split by file: {len(train_raw)} train from {len(train_paths)} file(s), "
        f"{len(val_raw)} val from {len(val_paths)} file(s), 0 files on both sides"
    )

    train_raw, note = filter_train_rows(
        train_raw,
        hold_out_operator=args.hold_out_operator,
        drop_random_train=args.drop_random_train,
        drop_random_seed=args.drop_random_seed,
    )
    if note:
        print(f"  {note}")

    if args.span_in_diff and args.context_source != CONTEXT_DIFF:
        raise SystemExit(
            "--span-in-diff has no effect outside --context-source diff: in after mode the "
            "span already points into the text the model reads. Refusing rather than "
            "ignoring it, because a recipe would otherwise record a protocol change that "
            "did not happen and the row would be incomparable with identical runs."
        )
    train_d, train_refused, train_paths_of, _ = decisions_of(
        train_raw,
        config=config,
        context_source=args.context_source,
        span_in_diff=args.span_in_diff,
    )
    val_d, val_refused, _, val_operators = decisions_of(
        val_raw,
        config=config,
        context_source=args.context_source,
        span_in_diff=args.span_in_diff,
    )
    print(f"  decisions: {len(train_d)} train (refused {train_refused or 'none'}), "
          f"{len(val_d)} val (refused {val_refused or 'none'})")
    if not train_d or not val_d:
        raise SystemExit("one side of the split is empty; nothing can be measured")

    # The operator of each SCORED row, in the order `evaluate` produces its per-row
    # distributions. Built through `bucketed_chunks` -- the single implementation of that
    # order -- and paired back by object identity, because a second sort here would be the
    # copy its docstring warns about and would silently score predictions against another
    # example's answer.
    # The added-line null, over exactly the rows `evaluate` scores: `bucketed_chunks` is
    # the one implementation of that order, and `bucketed_batches` drops a trailing
    # one-row chunk -- so computing this over `val_d` would average a null over rows the
    # model was never scored on and report it as the same quantity.
    added_chance, added_pointing, added_gold = plus_line_chance(
        [d for chunk in bucketed_chunks(val_d, batch_size=args.batch_size) for d in chunk]
    )
    if added_pointing:
        print(
            f"  added-line null: {added_chance:.1%} over {added_pointing} pointing row(s) "
            f"({added_gold} whose gold is an added line) -- what a pointer that reads no "
            "code scores by aiming at a '+' line"
        )

    scored_operators: list[str] | None = None
    measured = measure_operator(args)
    if measured:
        operator_of = {id(d): op for d, op in zip(val_d, val_operators, strict=True)}
        scored_operators = [
            operator_of[id(d)]
            for chunk in bucketed_chunks(val_d, batch_size=args.batch_size)
            for d in chunk
        ]
        marked = sum(1 for op in scored_operators if op == measured)
        if not marked:
            raise SystemExit(
                f"{measured!r} produced no SCORED validation row out of "
                f"{len(scored_operators)}. There is nothing to measure the effect on, and "
                "an arm that reports no number is not a control for one that does."
            )
        print(
            f"  {marked} scored validation row(s) come from {measured}, which training "
            + ("never saw" if args.hold_out_operator else "did see")
        )

    baseline, majority = majority_baseline(val_d)
    print(f"  majority-class baseline on val: {baseline:.1%} ({majority})")

    if args.train_subsample < 1.0:
        # Subsampled by FILE, not by row, and the validation side is never touched. Taking
        # a fraction of the rows would leave every training file represented and measure
        # density rather than coverage; a learning curve over files answers the question
        # actually being asked, which is whether MORE REPOSITORIES would help.
        by_file: dict[str, list] = {}
        for path, decision in zip(train_paths_of, train_d, strict=True):
            by_file.setdefault(path, []).append(decision)
        names = sorted(by_file)
        keep = max(1, round(args.train_subsample * len(names)))
        # Hashed, so the 25% set is a subset of the 50% set: a learning curve whose points
        # are drawn from unrelated samples measures sampling noise as well as size.
        ordered = sorted(names, key=lambda n: hashlib.sha256(n.encode("utf-8")).hexdigest())
        kept = set(ordered[:keep])
        train_d = [d for name in names if name in kept for d in by_file[name]]
        print(
            f"  train subsample {args.train_subsample:.0%}: {keep} of {len(names)} file(s), "
            f"{len(train_d)} decision(s); validation untouched"
        )
        if not train_d:
            raise SystemExit("the subsample kept no training decisions")

    # After any subsample, so the control measures the arm it is a control FOR, and before
    # the floors below, which are properties of what is actually trained on -- under a
    # permutation the entropy is unchanged but the majority share is not guaranteed to be,
    # and both must describe the labels the model really saw.
    if args.shuffle_train_labels:
        train_d = shuffle_train_labels(train_d, seed=args.shuffle_seed)
        print(
            f"  SHUFFLED-LABEL CONTROL: training labels permuted at seed "
            f"{args.shuffle_seed}; validation untouched. A model that still beats the "
            "held-out majority rate here means the SPLIT LEAKS and every accuracy "
            "measured on it is worthless."
        )

    # After any subsample, because the floor is a property of what is actually trained on.
    choice_floor = label_entropy(train_d)
    # The same reasoning for the train-side majority share, and it is a DIFFERENT number
    # from the validation baseline above -- the split is by file, so the two sides carry
    # different class mixes (48.0% against 52.1% on this corpus). Comparing the train
    # accuracy to the val baseline would be the repo's "one name, two quantities" defect
    # again, and it would have hidden the collapse: 48.0% read against 52.1% looks like a
    # model doing slightly worse than the prior rather than one sitting exactly on it.
    train_majority, train_majority_class = majority_baseline(train_d)
    print(
        f"  majority-class share on train: {train_majority:.1%} ({train_majority_class}) "
        "-- what a constant predictor scores on the rows it is optimised on"
    )
    print(
        f"  choice-loss floor (label-distribution entropy of the training set): "
        f"{choice_floor:.4f} nats -- a head that reaches this has learned the prior"
    )

    # The opponent for `paired_margin_vs_linear`, the gate `qd_train.ledger` has listed
    # since S6 and nothing has ever evaluated. `paired_margin_test` existed; what was
    # missing was a baseline scored on THE SAME examples in THE SAME order, which is what
    # `bucketed_chunks` makes available. Fitted once per sweep, not once per seed.
    #
    # Placed HERE, before the batch summary, deliberately. It is unrecorded setup -- no
    # RunRecorder is open yet -- and it is not cheap: measured at 109.6s on 727 documents
    # at this iteration budget, about as long as a full training seed. Putting it after the
    # `batches:` line would widen the window between that line and the seed loop, and
    # `test_a_run_killed_before_training_finishes_still_writes_a_priced_row` uses that line
    # as the point after which a kill must produce a row. It caught this when the fit sat
    # on the wrong side of it. Its duration is printed so the setup cost is visible rather
    # than merely absent from the ledger.
    scored_val = [d for chunk in bucketed_chunks(val_d, batch_size=args.batch_size)
                  for d in chunk]
    print(
        f"  linear control: fitting on {len(train_d)} train doc(s), to score the "
        f"{len(scored_val)} validation row(s) the model is scored on"
    )
    _fit_t0 = time.monotonic()
    baseline_correct, baseline_not_run = linear_baseline_correctness(
        train_d,
        scored_val,
        seed=0,
        max_iter=LINEAR_CONTROL_MAX_ITER,
        cache_dir=args.control_cache,
    )
    _fit_s = time.monotonic() - _fit_t0
    if baseline_correct is None:
        print(f"  linear control NOT RUN after {_fit_s:.1f}s: "
              f"{baseline_not_run.reason}")  # type: ignore[union-attr]
    else:
        print(
            f"  linear control accuracy on those rows: "
            f"{float(baseline_correct.mean()):.1%} (fitted in {_fit_s:.1f}s, unrecorded "
            "setup: no row covers this time)"
        )

    train_plans = bucketed_batches(train_d, batch_size=args.batch_size, config=config)
    val_plans = bucketed_batches(val_d, batch_size=args.batch_size, config=config)
    if not train_plans or not val_plans:
        raise SystemExit("too few decisions to form a batch on one side of the split")
    waste = sum(
        sum(1 for row in p.context_mask for live in row if not live) for p in train_plans
    ) / sum(p.batch_size * p.context_width for p in train_plans)
    print(f"  batches: {len(train_plans)} train, {len(val_plans)} val; "
          f"train padding waste {waste:.2%}\n")

    ledger = Ledger(args.ledger)
    corpus_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    quick_reason = quick_reason_for(
        examples_supplied=bool(args.examples),
        seeds=args.seeds,
        train_subsample=args.train_subsample,
        shuffled=args.shuffle_train_labels,
        pool_records=((manifest.get("pool") or {}).get("records")),
        corpus_hash=corpus_hash,
    )
    recipe = recipe_of(args)
    # `sort_keys=True` and no `separators`, unchanged: this is the hash the 61 capacity
    # rows and the learning curve running on the box were written with, and changing the
    # bytes it hashes would make every row written after today incomparable with them.
    recipe_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode("utf-8")
    ).hexdigest()

    runs: list[dict[str, object]] = []
    for seed in range(args.seeds):
        protocol = Protocol(
            data_snapshot_hash=corpus_hash,
            tokenizer_hash="bytes-utf8-256",
            backbone_commit=(
                f"rung0-scratch:{config.width}x{config.n_heads}:{config.n_layers}layer:"
                f"ctx{config.max_context_bytes}"
            ),
            recipe_hash=recipe_hash,
            seed=seed,
        )
        with RunRecorder(
            ledger,
            entry_point=Path(__file__),
            protocol=protocol,
            run_kind="ft",
            repo=REPO,
            env=Environment.detect(device=args.device),
            # None here, and stated by `recorder.measured()` the moment training returns.
            # The block WRAPS the training now, so a run killed before training finishes
            # still writes a row -- and the ordinary path carries the same training
            # duration it always did, under wall_clock_source="caller". A killed run falls
            # back to the recorder's lifetime under "recorder", which is what it was
            # billed for.
            wall_clock_s=None,
            # Same factory the RunControl above was built from, so the row and the gate
            # cannot price the run differently. Rebuilt rather than carried in `run`,
            # which holds JSON-serialisable facts only.
            cost=run_cost_estimate(
                cap=WallClockCap(cap_s=RUN_CAP_S), device=args.device,
                instance=args.instance, usd_per_hour=args.usd_per_hour,
                usd_per_gpu_hour=args.usd_per_gpu_hour,
            ),
            quick=True,
            # The same object `recipe_hash` was computed from, so the row says what the
            # hash only distinguishes.
            recipe=recipe,
            quick_reason=quick_reason,
            notes=(
                "tools/rung0_real_run.py -- rung 0 trained on a real qd-mutate corpus and "
                "measured on files it never saw, split by path"
                + (
                    " -- SHUFFLED-LABEL CONTROL: the training labels were permuted, so "
                    "this row is not a measurement of the model. Its held-out accuracy is "
                    "the control's value and belongs at or below the majority rate."
                    if args.shuffle_train_labels
                    else ""
                )
            ),
        ) as recorder:
            run = train_once(
                train_plans=train_plans,
                val_plans=val_plans,
                baseline=baseline,
                choice_floor=choice_floor,
                train_decisions=len(train_d),
                span_weight=args.span_weight,
                peak_lr=args.lr,
                device=args.device,
                seed=seed,
                epochs=args.epochs,
                config=config,
                instance=args.instance,
                usd_per_hour=args.usd_per_hour,
                usd_per_gpu_hour=args.usd_per_gpu_hour,
                approved_by=args.approved_by,
            )
            recorder.measured(float(run["wall_clock_s"]))  # type: ignore[arg-type]
            runs.append(run)
            after = run["val_after"]  # type: ignore[index]
            before = run["val_before"]  # type: ignore[index]
            # The control, recorded on the row of the run that IS the control. Chance is
            # the held-out majority rate, computed by the harness from the validation gold
            # -- which the shuffle never touched, so it is the same bar the real arms are
            # measured against. Set only under --shuffle-train-labels: a run trained on
            # true labels has not evaluated this control and must keep saying so, which is
            # what RunRecorder's NotRun default does.
            # On EVERY run, control or not. It reads the held-out choice distribution this
            # evaluation already produced, so it costs no GPU time and there is no reason
            # for a row to omit it. It is also the check that catches what `_fit_gate`
            # cannot: _fit_gate reads TRAIN accuracy against the train majority, so a head
            # that fits the training set and then answers one class on everything held out
            # clears it while being exactly the degenerate case.
            recorder.control(
                "degenerate_head",
                degenerate_head_state(after["choice_probs"]),  # type: ignore[index,arg-type]
            )
            # The `ece` gate, likewise never computed on any row. `calibration_fit.ece_gate`
            # says so in its own docstring -- "the ece gate qd_train.ledger has always
            # listed and nothing ever computed" -- so it was built and never called. It
            # needs the same distributions the control above reads, plus the held-out gold.
            # Its threshold and bin count are left at the function's defaults: rule 2 makes
            # a threshold read-only to an agent, and passing one here would be retuning it
            # from the call site.
            # `paired_margin_vs_linear`. Paired over the same examples in the same order,
            # which is the whole point of the test: example difficulty cancels, so it
            # measures the difference rather than the variance of the set. The model's
            # correctness comes from the argmax of the same distributions everything else
            # here reads, against the same `choice_gold`.
            # Accuracy restricted to the rows the held-out operator produced. The pairing is
            # by object identity through `bucketed_chunks` -- the one implementation of the
            # scoring order -- rather than a second sort that could drift from it and score
            # each prediction against another example's answer.
            if scored_operators is not None:
                probs = np.asarray(after["choice_probs"], dtype=np.float64)  # type: ignore[index,arg-type]
                gold = np.asarray(after["choice_gold"], dtype=int)  # type: ignore[index,arg-type]
                marked, siblings, impure = operator_and_sibling_rows(
                    scored_operators, gold, measured
                )
                hits = int((np.argmax(probs[marked], axis=1) == gold[marked]).sum())
                # ONE key across both arms, because the comparison is between them and a
                # key that changed with the condition could not be joined. So the key may
                # not assert the condition either: the holdout arm and its size-matched
                # control both report this, and only one of them held the operator out.
                held = bool(args.hold_out_operator)
                recorder.metric(
                    "val_choice_top1_on_measured_operator",
                    Ran(
                        passed=True,
                        value=hits / len(marked),
                        n=hits,
                        n_total=len(marked),
                        detail=(
                            f"top-1 on the {len(marked)} validation row(s) produced by "
                            f"{measured!r}, which training "
                            + ("never saw" if held else "did see")
                            + ". The linear control scores 3.08% on stub.panic and 0.00% "
                            "on cosmetic.rename_local when they are held out, having "
                            "learned the generator rather than the change; this is the "
                            "same question asked of the model. The two arms score the "
                            "identical validation rows, so their difference is the "
                            "operator's absence from training and nothing else."
                        ),
                    ),
                )
                # The confound the size-matched control cannot reach. Every operator in
                # this corpus belongs to exactly one class, so holding one out also moves
                # the class prior -- stub.panic alone is 42.8% of the rows, and removing it
                # takes `stub` from 46% of training to about 7%. A model that then
                # under-predicts `stub` scores badly on the held-out rows for a reason that
                # has nothing to do with recognising generators, and `--drop-random-train`
                # does not separate the two: it drops uniformly, so it preserves the prior
                # it needs to disturb.
                #
                # Its siblings do separate them. They are validation rows of the SAME class
                # from a DIFFERENT operator, so they sit under the identical shifted prior
                # and training did see them. Sibling accuracy high while the held-out
                # operator's is low means the prior is intact and the fingerprint was the
                # signal; both low means the arm moved the prior and the held-out number
                # says little by itself.
                if not siblings:
                    recorder.metric(
                        "val_choice_top1_on_measured_operator_siblings",
                        NotRun(
                            reason=(
                                f"no scored validation row shares {measured!r}'s class "
                                "while coming from another operator, so there is nothing "
                                "under the same shifted prior to compare against. The "
                                "held-out number cannot be separated from prior shift on "
                                "this corpus."
                            )
                        ),
                    )
                else:
                    sib_hits = int(
                        (np.argmax(probs[siblings], axis=1) == gold[siblings]).sum()
                    )
                    recorder.metric(
                        "val_choice_top1_on_measured_operator_siblings",
                        Ran(
                            passed=True,
                            value=sib_hits / len(siblings),
                            n=sib_hits,
                            n_total=len(siblings),
                            detail=(
                                f"top-1 on the {len(siblings)} validation row(s) of "
                                f"{measured!r}'s own class produced by OTHER operators, "
                                "which training did see. Same class, same shifted prior, "
                                "different generator -- so the gap between this and the "
                                "measured operator's own accuracy is what the generator's "
                                "absence cost, with the prior held fixed."
                                + (
                                    ""
                                    if not impure
                                    else f" WARNING: {impure} of {len(marked)} measured "
                                    "row(s) carry a different class, so this operator is "
                                    "not class-pure and the sibling set is approximate."
                                )
                            ),
                        ),
                    )

            if baseline_correct is None:
                recorder.gate("paired_margin_vs_linear", baseline_not_run)  # type: ignore[arg-type]
            else:
                model_correct = np.argmax(
                    np.asarray(after["choice_probs"], dtype=np.float64), axis=1  # type: ignore[index,arg-type]
                ) == np.asarray(after["choice_gold"], dtype=int)  # type: ignore[index,arg-type]
                recorder.gate(
                    "paired_margin_vs_linear",
                    paired_margin_test(model_correct, baseline_correct, seed=seed),
                )
            recorder.gate(
                "permutation_consistency",
                run["permutation_consistency"],  # type: ignore[index,arg-type]
            )
            recorder.gate(
                "ece",
                ece_state(
                    after["choice_probs"],  # type: ignore[index,arg-type]
                    after["choice_gold"],  # type: ignore[index,arg-type]
                ),
            )
            if args.shuffle_train_labels:
                recorder.control(
                    "shuffled_label",
                    shuffled_label_control(
                        float(after["choice_top1"]),  # type: ignore[index]
                        # The gold of the rows actually SCORED, for the same reason the
                        # gate above uses it: bucketing drops any trailing one-row chunk,
                        # so val_d can contain examples this accuracy never covered, and
                        # the majority share of a different set is a different ceiling.
                        after["choice_gold"],  # type: ignore[index,arg-type]
                        n_eval=int(after["choice_n"]),  # type: ignore[index]
                    ),
                )
            # Against the majority-class baseline, which is a property of the split
            # rather than a quantity with seed noise -- so one variance, not two. The
            # two-arm form would claim 1.41x more sensitivity than this comparison has.
            recorder.metric(
                "sweep_can_resolve",
                resolution_state(
                    sd=args.prior_sd,
                    n_per_arm=args.seeds,
                    target=args.target_difference,
                    against_known_reference=True,
                ),
            )
            recorder.metric(
                "val_choice_top1_over_baseline",
                _accuracy_gate(
                    float(after["choice_top1"]),  # type: ignore[index]
                    baseline,
                    n=int(after["choice_n"]),  # type: ignore[index]
                    what="choice",
                    baseline_name="majority-class baseline",
                ),
            )
            # Over POINTING rows only, and NotRun when there are none. The undecomposed
            # figure counts a correctly-withheld pointer as a correctly-placed one, so a
            # corpus on which every gold is the abstention scores 100% on it -- which is
            # exactly what `--context-source diff` produces, because span offsets are into
            # `after` and encoding the diff moves every byte. Reporting that as a pass would
            # be a check that could not run returning the same answer as one that ran.
            pointing_n = int(after["span_pointing_n"])  # type: ignore[index]
            no_pointers = NotRun(
                reason=(
                    f"no gold span points at a line: all {int(after['span_n'])} scored "  # type: ignore[index]
                    "row(s) abstain, so pointer accuracy has nothing to be measured over. "
                    "A head that abstains on everything scores 100% here and has located "
                    "nothing."
                )
            )
            recorder.metric(
                "val_span_start_top1_over_chance",
                _accuracy_gate(
                    float(after["span_pointing_start_top1"]),  # type: ignore[index]
                    float(after["span_pointing_chance"]),  # type: ignore[index]
                    n=pointing_n,
                    what="span start",
                    baseline_name="uniform-pointer chance over the candidate line starts",
                )
                if pointing_n
                else no_pointers,
            )
            recorder.metric(
                "val_span_end_top1_over_chance",
                _accuracy_gate(
                    float(after["span_pointing_end_top1"]),  # type: ignore[index]
                    float(after["span_pointing_chance"]),  # type: ignore[index]
                    n=pointing_n,
                    what="span end",
                    baseline_name="uniform-pointer chance over the candidate line starts",
                )
                if pointing_n
                else no_pointers,
            )
            # Beside the two gates above and never instead of them, because rule 2 makes
            # their baseline read-only. But the uniform rate those gates use is the right
            # null only when nothing marks the changed line, and in a unified diff a `+`
            # does: both corpora average about 1.5 added lines per hunk, so a policy that
            # reads no code and points at an added line scores far above uniform. A pointer
            # reported against the uniform rate alone would present that marker as a result.
            recorder.metric(
                "val_span_pointing_added_line_chance",
                Ran(
                    passed=True,
                    value=added_chance,
                    n=added_gold,
                    n_total=added_pointing,
                    detail=(
                        f"what 'point at a uniformly random ADDED line' scores on the same "
                        f"{added_pointing} pointing row(s): {added_chance:.2%}, against the "
                        f"uniform-over-all-candidates rate of "
                        f"{float(after['span_pointing_chance']):.2%} the gates above use. "  # type: ignore[index]
                        f"{added_gold} of those rows have a gold that IS an added line; on "
                        "the rest this null scores zero, so a corpus where the gold is "
                        "usually not an added line makes this null weak and the uniform "
                        "rate closer to right. Read the pointer's accuracy against the "
                        "LARGER of the two."
                    ),
                )
                if added_pointing
                else no_pointers,
            )
            recorder.metric(
                "train_choice_top1_over_train_majority",
                _fit_gate(
                    float(run["train_after"]["choice_top1"]),  # type: ignore[index]
                    train_majority,
                    n=int(run["train_after"]["choice_n"]),  # type: ignore[index]
                ),
            )
            recorder.metric(
                "choice_loss_below_the_class_prior",
                _prior_gate(
                    float(run["choice_last"]),  # type: ignore[arg-type]
                    float(run["choice_floor"]),  # type: ignore[arg-type]
                    n=int(run["train_decisions"]),  # type: ignore[arg-type]
                ),
            )
            recorder.metric(
                "train.termination",
                Ran(passed=run["termination"] != "wall_clock_cap", value=run["termination"]),
            )
            recorder.metric(
                "train_padding_waste",
                Ran(
                    passed=True,
                    value=run["train_padding_waste"],
                    n=len(train_plans),
                    n_total=len(train_plans),
                    detail="fraction of context positions that were padding across the epoch",
                ),
            )

        print(
            f"{args.device} seed={seed}: total {run['total_first']:.4f} -> "
            f"{run['total_last']:.4f}  "
            # The two channels separately, because the total hides their ratio and the
            # ratio is what decides which head the optimizer actually serves.
            f"[choice {run['choice_first']:.3f}->{run['choice_last']:.3f} "
            f"(floor {choice_floor:.3f}) "
            f"span {run['span_first']:.3f}->{run['span_last']:.3f}]  "
            f"choice val {float(before['choice_top1']):.1%} -> "  # type: ignore[index]
            f"{float(after['choice_top1']):.1%} "  # type: ignore[index]
            f"(baseline {baseline:.1%}, {int(after['choice_n'])} rows)  "  # type: ignore[index]
            f"[train {float(run['train_after']['choice_top1']):.1%}]  "  # type: ignore[index]
            # The POINTING figure, and a plain statement when there is nothing to point at.
            # The undecomposed number reads 100.0% on an all-abstain corpus, which is what
            # `--context-source diff` produces; printing that beside a chance rate computed
            # over line starts the head was never asked to choose between is how a vacuous
            # result gets read as a triumph on the way past.
            + (
                f"span start {float(after['span_pointing_start_top1']):.1%} "  # type: ignore[index]
                f"end {float(after['span_pointing_end_top1']):.1%} "  # type: ignore[index]
                f"(chance {float(after['span_pointing_chance']):.1%}, "  # type: ignore[index]
                f"{int(after['span_pointing_n'])} pointing rows)  "  # type: ignore[index]
                if int(after["span_pointing_n"])  # type: ignore[index]
                else f"span NOT MEASURED (all {int(after['span_n'])} scored rows abstain)  "  # type: ignore[index]
            )
            + f"{run['wall_clock_s']:.1f}s  {run['optimizer_steps']} steps"
        )

    gaps = [
        float(r["val_after"]["choice_top1"]) - baseline  # type: ignore[index]
        for r in runs
    ]
    mean = sum(gaps) / len(gaps)
    spread = max(gaps) - min(gaps)
    print(
        f"\nchoice accuracy over baseline across {len(runs)} seed(s): "
        f"mean {mean:+.1%}, spread {spread:.1%}"
    )
    print(
        "Every row is quick=True and promotes nothing (rule 8). What this establishes is a "
        "floor rung 3 has to beat to justify its cost, measured rather than assumed."
    )
    # Two failures wear the same held-out number and call for opposite fixes, so the
    # summary names which one happened rather than printing one sentence for both. An
    # earlier version printed only the second and sent four experiments after capacity,
    # context and corpus size while the objective was the constraint.
    collapsed = [
        r for r in runs
        if float(r["train_after"]["choice_top1"]) <= train_majority  # type: ignore[index]
    ]
    if collapsed:
        print(
            f"\nThe choice head COLLAPSED TO A CONSTANT PREDICTOR on {len(collapsed)} of "
            f"{len(runs)} seed(s): train accuracy at or below the training-set majority "
            f"share of {train_majority:.1%}, on rows it was optimised on. The held-out "
            "number from those seeds is not evidence about generalisation -- there was no "
            "fit to generalise. Look at the objective before capacity or corpus: at "
            f"--span-weight {args.span_weight} the two channels open at "
            f"{_channel_ratio(runs[0])}, and AUDIT/rung0-span-weight-2026-09-20.json "
            "measures this gate failing on every seed at >= 0.5 and passing below 0.2."
        )
    elif mean <= 0:
        print(
            "\nRung 0 fitted its training files and did NOT beat the majority class on "
            "held-out ones. Unlike a collapse this is a genuine generalisation result, and "
            "it points at the corpus and the split rather than at the objective."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
