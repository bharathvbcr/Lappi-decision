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
import hashlib
import json
import math
import os
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

import torch  # noqa: E402
from repo_git import git_bytes, tracked_paths  # noqa: E402
from run_cost import n_gpus_for_device  # noqa: E402

from qd_train.byte_batch import BatchPlan, plan_batch, span_supervision  # noqa: E402
from qd_train.byte_context import ID_PAD, SpanOutsideWindow  # noqa: E402
from qd_train.byte_decider import ByteDeciderConfig  # noqa: E402
from qd_train.byte_train import Rung0Model, Rung0Step, train_rung0  # noqa: E402
from qd_train.heads import plan_span_batch, serving_scores  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.mutate_adapter import (  # noqa: E402
    MUTATION_CLASSES,
    MalformedExample,
    PhantomFinalLine,
    parse_example,
    to_decision,
)
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

#: Attention is quadratic in the context, so this is a real ceiling and not a typo guard.
MAX_CONTEXT_BYTES: Final[int] = 32_768


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


def decisions_of(
    examples: Sequence[dict[str, object]], *, config: ByteDeciderConfig
) -> tuple[list, dict[str, int], list[str]]:
    """Parse and convert, counting every refusal by kind rather than dropping quietly.

    Returns the decisions, the refusals by kind, and **the source path of each decision**.

    The paths are returned rather than recovered by zipping against the input, because this
    function DROPS refused examples: 1,599 raw examples become 763 decisions at 8192 bytes,
    so ``zip(examples, decisions)`` pairs each decision with the wrong file and every
    per-file operation downstream is quietly wrong. The parallel list is built here, where
    the row being dropped is still in hand.
    """
    out = []
    paths: list[str] = []
    refused: dict[str, int] = {}
    for obj in examples:
        try:
            decision = to_decision(
                parse_example(obj), max_context_bytes=config.max_context_bytes
            )
        except (MalformedExample, PhantomFinalLine, SpanOutsideWindow) as exc:
            name = type(exc).__name__
            refused[name] = refused.get(name, 0) + 1
            continue
        out.append(decision)
        paths.append(str(obj.get("function", {}).get("path", "")))  # type: ignore[union-attr]
    return out, refused, paths


# -- batching ----------------------------------------------------------------------------


def bucketed_batches(
    decisions: Sequence, *, batch_size: int, config: ByteDeciderConfig
) -> list[BatchPlan]:
    """Length-sorted batches, so a batch's width is its own rows and not the corpus maximum.

    ``plan_batch`` pads to the batch maximum, which is the right choice and is exactly why
    the batch's membership matters: one long row drags the whole batch's width up and every
    short row in it pays for the difference in padding.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    ordered = sorted(decisions, key=lambda d: d.context.n_bytes_kept)
    plans: list[BatchPlan] = []
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        if len(chunk) < 2:
            # A one-row batch has no contrast for the choice head and cannot be scored
            # against a per-batch floor. Dropped and counted by the caller's arithmetic
            # (len(plans) * batch_size against len(decisions)), never padded out with a
            # duplicate row, which would be supervision this corpus does not contain.
            continue
        plans.append(
            plan_batch(
                chunk,
                pad_id=ID_PAD,
                max_context_bytes=config.max_context_bytes,
                max_option_bytes=96,
            )
        )
    return plans


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
            for i, top in enumerate(logits.argmax(dim=1).tolist()):
                choice_hit += 1 if top == plan.choice_target[i] else 0
                choice_n += 1
            span_plan = plan_span_batch(span_supervision(plan), device=device)
            if span_plan.n_spans:
                start, end = model.span_head(hidden, span_plan)
                start_rows = serving_scores(start, span_plan)
                end_rows = serving_scores(end, span_plan)
                for k in range(span_plan.n_spans):
                    gold_start = int(span_plan.gold_start[k])
                    gold_end = int(span_plan.gold_end[k])
                    start_hit += 1 if int(start_rows[k].argmax()) == gold_start else 0
                    end_hit += 1 if int(end_rows[k].argmax()) == gold_end else 0
                    # `serving_scores` returns exactly the rows a runtime would accept, so
                    # its length is the real number of choices this pointer had.
                    span_chance += 1.0 / max(1, int(start_rows[k].numel()))
                    span_n += 1
    model.train()
    return {
        "choice_top1": choice_hit / choice_n if choice_n else 0.0,
        "choice_n": choice_n,
        "span_start_top1": start_hit / span_n if span_n else 0.0,
        "span_end_top1": end_hit / span_n if span_n else 0.0,
        "span_n": span_n,
        "span_chance": span_chance / span_n if span_n else 0.0,
    }


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


def train_once(
    *,
    train_plans: Sequence[BatchPlan],
    val_plans: Sequence[BatchPlan],
    baseline: float,
    choice_floor: float,
    train_decisions: int,
    span_weight: float,
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
    cap = WallClockCap(cap_s=3600.0)
    control = RunControl(
        schedule=LRSchedule(peak_lr=3e-3, warmup_steps=max(1, steps // 10), total_steps=steps),
        cap=cap,
        # `for_device` prices cpu and mps at zero -- a Mac that is already bought costs
        # nothing per hour -- and refuses to invent a rate for anything else. The literal
        # this replaces applied the Mac's price to whatever `--device` named, and `--device`
        # is a free string: every run this tool made on the rented GH200 recorded
        # `instance="local-cuda"` on `n_gpus=0` at `usd_per_hour=0.0`. Not an under-report
        # of a cost -- an assertion that the machine was a local one with no GPUs in it.
        cost=CostEstimate.for_device(
            cap=cap,
            device=device,
            n_gpus=n_gpus_for_device(device),
            usd_per_hour=usd_per_hour,
            usd_per_gpu_hour=usd_per_gpu_hour,
            instance=instance,
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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=400)
    parser.add_argument("--limit", type=int, default=4000, help="examples qd-mutate may emit")
    parser.add_argument("--batch-size", type=int, default=16)
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
        "--span-weight",
        type=float,
        default=1.0,
        help=(
            "multiplier on the span loss in the summed objective. Must be positive -- "
            "Rung0Step refuses zero, because that trains the span head on nothing while its "
            "loss still appears in the log"
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
    parser.add_argument("--val-share", type=float, default=0.25)
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

    train_d, train_refused, train_paths_of = decisions_of(train_raw, config=config)
    val_d, val_refused, _ = decisions_of(val_raw, config=config)
    print(f"  decisions: {len(train_d)} train (refused {train_refused or 'none'}), "
          f"{len(val_d)} val (refused {val_refused or 'none'})")
    if not train_d or not val_d:
        raise SystemExit("one side of the split is empty; nothing can be measured")

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
    runs: list[dict[str, object]] = []
    for seed in range(args.seeds):
        run = train_once(
            train_plans=train_plans,
            val_plans=val_plans,
            baseline=baseline,
            choice_floor=choice_floor,
            train_decisions=len(train_d),
            span_weight=args.span_weight,
            device=args.device,
            seed=seed,
            epochs=args.epochs,
            config=config,
            instance=args.instance,
            usd_per_hour=args.usd_per_hour,
            usd_per_gpu_hour=args.usd_per_gpu_hour,
            approved_by=args.approved_by,
        )
        runs.append(run)
        after = run["val_after"]  # type: ignore[index]
        before = run["val_before"]  # type: ignore[index]
        protocol = Protocol(
            data_snapshot_hash=corpus_hash,
            tokenizer_hash="bytes-utf8-256",
            backbone_commit=(
                f"rung0-scratch:{config.width}x{config.n_heads}:{config.n_layers}layer:"
                f"ctx{config.max_context_bytes}"
            ),
            recipe_hash=hashlib.sha256(
                json.dumps(
                    {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "val_share": args.val_share,
                        "lr": 3e-3,
                        # The objective is part of the recipe. Without this the five points
                        # of the span-weight sweep hash identically, and two runs that
                        # optimised different things become one protocol in the ledger --
                        # which is exactly the comparison the sweep exists to make.
                        "span_weight": args.span_weight,
                        # Deterministic and nondeterministic runs are different protocols,
                        # not the same protocol measured twice. Without this they hash
                        # identically and the ledger treats a reproducible number and a
                        # draw from a 2.8-point spread as comparable rows.
                        "deterministic": args.deterministic,
                        "rev": args.rev,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            seed=seed,
        )
        with RunRecorder(
            ledger,
            protocol=protocol,
            run_kind="ft",
            repo=REPO,
            env=Environment.detect(device=args.device),
            # train_once() returned before this block was entered, so the recorder's own
            # lifetime is the time to write metrics -- microseconds against a run that
            # takes minutes. The measured figure is the one the log already prints.
            wall_clock_s=float(run["wall_clock_s"]),  # type: ignore[arg-type]
            quick=True,
            quick_reason=(
                "the corpus is this repository's own sources rather than the pool the plan "
                "names, which rule 8 counts as a subsample"
            ),
            notes=(
                "tools/rung0_real_run.py -- rung 0 trained on a real qd-mutate corpus and "
                "measured on files it never saw, split by path"
            ),
        ) as recorder:
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
            recorder.metric(
                "val_span_start_top1_over_chance",
                _accuracy_gate(
                    float(after["span_start_top1"]),  # type: ignore[index]
                    float(after["span_chance"]),  # type: ignore[index]
                    n=int(after["span_n"]),  # type: ignore[index]
                    what="span start",
                    baseline_name="uniform-pointer chance over the candidate line starts",
                ),
            )
            recorder.metric(
                "val_span_end_top1_over_chance",
                _accuracy_gate(
                    float(after["span_end_top1"]),  # type: ignore[index]
                    float(after["span_chance"]),  # type: ignore[index]
                    n=int(after["span_n"]),  # type: ignore[index]
                    what="span end",
                    baseline_name="uniform-pointer chance over the candidate line starts",
                ),
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
            f"span start {float(after['span_start_top1']):.1%} "  # type: ignore[index]
            f"end {float(after['span_end_top1']):.1%} "  # type: ignore[index]
            f"(chance {float(after['span_chance']):.1%}, "  # type: ignore[index]
            f"{int(after['span_n'])} rows)  "  # type: ignore[index]
            f"{run['wall_clock_s']:.1f}s  {run['optimizer_steps']} steps"
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
