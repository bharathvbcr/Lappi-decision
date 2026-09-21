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
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

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


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, check=True, text=True
    ).stdout


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


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
    for name in sorted(_git("ls-tree", "-r", "--name-only", rev).splitlines()):
        language = POOL_SUFFIXES.get(Path(name).suffix)
        if language is None:
            continue
        try:
            raw = _git_bytes("show", f"{rev}:{name}")
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
) -> tuple[list, dict[str, int]]:
    """Parse and convert, counting every refusal by kind rather than dropping quietly."""
    out = []
    refused: dict[str, int] = {}
    for obj in examples:
        try:
            out.append(to_decision(parse_example(obj), max_context_bytes=config.max_context_bytes))
        except (MalformedExample, PhantomFinalLine, SpanOutsideWindow) as exc:
            name = type(exc).__name__
            refused[name] = refused.get(name, 0) + 1
    return out, refused


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
                    span_n += 1
    model.train()
    return {
        "choice_top1": choice_hit / choice_n if choice_n else 0.0,
        "choice_n": choice_n,
        "span_start_top1": start_hit / span_n if span_n else 0.0,
        "span_end_top1": end_hit / span_n if span_n else 0.0,
        "span_n": span_n,
    }


def _accuracy_gate(measured: float, baseline: float, *, n: int, what: str) -> TriState:
    """Whether the model beat answering the majority class, carrying both numbers.

    ``NotRun`` on an empty evaluation set rather than 0.0: nothing was measured, and a 0%
    reading would look like a model that failed rather than a measurement that did not
    happen.
    """
    if n == 0:
        return NotRun(
            reason=(
                f"no {what} rows were evaluated, so there is no accuracy to compare against "
                "the majority-class baseline"
            )
        )
    return Ran(
        passed=measured > baseline,
        value=measured,
        n=round(measured * n),
        n_total=n,
        detail=(
            f"{what} top-1 {measured:.1%} of {n} held-out rows, against a majority-class "
            f"baseline of {baseline:.1%}. The gap is {measured - baseline:+.1%}; a model at "
            "the baseline has learned the prior and nothing else."
        ),
    )


# -- the run -----------------------------------------------------------------------------


def train_once(
    *,
    train_plans: Sequence[BatchPlan],
    val_plans: Sequence[BatchPlan],
    baseline: float,
    device: str,
    seed: int,
    epochs: int,
    config: ByteDeciderConfig,
) -> dict[str, object]:
    """One seed end to end: train on the train files, measure on the validation files."""
    torch.manual_seed(seed)
    steps = epochs * len(train_plans)
    model = Rung0Model(config).to(device)
    step = Rung0Step(model)
    cap = WallClockCap(cap_s=3600.0)
    control = RunControl(
        schedule=LRSchedule(peak_lr=3e-3, warmup_steps=max(1, steps // 10), total_steps=steps),
        cap=cap,
        # A Mac that is already bought costs nothing per hour. Rule 4 is about rented GPUs.
        cost=CostEstimate(cap=cap, usd_per_hour=0.0, n_gpus=0, instance=f"local-{device}"),
        grad_accum=1,
    )

    before = evaluate(model, val_plans, device=device)
    t0 = time.monotonic()
    result = train_rung0(
        cycle(train_plans, n=steps), step=step, control=control, seed=seed, max_batches=steps
    )
    wall = time.monotonic() - t0
    after = evaluate(model, val_plans, device=device)

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
        "baseline": baseline,
        "train_padding_waste": waste_num / waste_den if waste_den else 0.0,
        "wall_clock_s": wall,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="scratch directory for the corpus")
    parser.add_argument("--rev", required=True, help="the revision the corpus is read at")
    parser.add_argument("--device", default="mps")
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
    parser.add_argument("--val-share", type=float, default=0.25)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO / "target" / "release" / "qd-mutate",
        help="the qd-mutate binary",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.epochs <= MAX_EPOCHS:
        raise SystemExit(f"--epochs must be in [1, {MAX_EPOCHS}], got {args.epochs}")
    if not 1 <= args.limit <= MAX_EXAMPLES:
        raise SystemExit(f"--limit must be in [1, {MAX_EXAMPLES}], got {args.limit}")
    if not args.binary.exists():
        raise SystemExit(f"{args.binary} does not exist; cargo build --release -p qd-mutate")
    if not 1 <= args.context_bytes <= MAX_CONTEXT_BYTES:
        raise SystemExit(
            f"--context-bytes must be in [1, {MAX_CONTEXT_BYTES}], got {args.context_bytes}"
        )
    config = ByteDeciderConfig(max_context_bytes=args.context_bytes)

    print(f"corpus: reading tracked sources at {args.rev}")
    pool, capped = build_pool(rev=args.rev, max_files=args.max_files)
    print(f"  pool: {len(pool)} file(s){' (CAPPED)' if capped else ''}")

    examples, manifest = generate_examples(
        pool, out_dir=args.out, seed=0, limit=args.limit, binary=args.binary
    )
    print(f"  qd-mutate emitted {len(examples)} example(s)")

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

    train_d, train_refused = decisions_of(train_raw, config=config)
    val_d, val_refused = decisions_of(val_raw, config=config)
    print(f"  decisions: {len(train_d)} train (refused {train_refused or 'none'}), "
          f"{len(val_d)} val (refused {val_refused or 'none'})")
    if not train_d or not val_d:
        raise SystemExit("one side of the split is empty; nothing can be measured")

    baseline, majority = majority_baseline(val_d)
    print(f"  majority-class baseline on val: {baseline:.1%} ({majority})")

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
            device=args.device,
            seed=seed,
            epochs=args.epochs,
            config=config,
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
                ),
            )
            recorder.metric(
                "val_span_start_top1",
                _accuracy_gate(
                    float(after["span_start_top1"]),  # type: ignore[index]
                    0.0,
                    n=int(after["span_n"]),  # type: ignore[index]
                    what="span start",
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
            f"choice val {float(before['choice_top1']):.1%} -> "  # type: ignore[index]
            f"{float(after['choice_top1']):.1%} "  # type: ignore[index]
            f"(baseline {baseline:.1%}, {int(after['choice_n'])} rows)  "  # type: ignore[index]
            f"span start {float(after['span_start_top1']):.1%} "  # type: ignore[index]
            f"end {float(after['span_end_top1']):.1%} "  # type: ignore[index]
            f"({int(after['span_n'])} rows)  "  # type: ignore[index]
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
    if mean <= 0:
        print(
            "\nRung 0 did NOT beat the majority class on held-out files. That is a real "
            "answer and the ladder's cheapest one: it says raise capacity or context before "
            "spending another GPU hour on rung 3, and ByteDeciderConfig names capacity as "
            "the first thing to raise."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
