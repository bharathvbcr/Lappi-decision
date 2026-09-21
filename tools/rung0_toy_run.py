"""Does rung 0's training path execute at all, and does the loss move where it should?

``python/qd_train/`` has never trained anything. Every module below it is covered by tests,
but five of those test files ``importorskip`` torch -- which is absent from the repo venv by
design -- so ``make gates`` has never executed a single parameter update. A green gate and an
untrained trainer are not in tension here; they are the same fact seen twice.

This tool closes that by running the smallest real thing: :func:`qd_train.byte_train.train_rung0`
over a memorisable toy batch, on ``mps`` and on ``cpu``, for a handful of steps, with a row
written to the ledger for every run. **A loss that does not fall on a batch the model can
memorise is a wiring bug**, and that is the only claim this tool is built to make. It says
nothing about whether the model is any good.

Three things are measured, in this order:

1. **Does it run.** Parse -> :func:`qd_train.mutate_adapter.to_decision` ->
   :func:`qd_train.byte_batch.plan_batch` -> :class:`qd_train.byte_train.Rung0Model` ->
   ``train_rung0``, on a real device, with a real optimizer step.
2. **Does the loss move.** The same batch every step, so the floor is memorisation, not
   generalisation. Both slot losses are carried separately: a falling total with a flat span
   term is a model that learned the mutation class and nothing about *where*.
3. **Are the heads wired to the rows the runtime reads.** With numbers, not prose --
   see below.

## The abstain row's POSITION, which is the half that hides

``GAP-SCHEMA-NOUL-LETTER-BUDGET`` settled the *count*: a maximal choice slot is 16 named
options plus one reserved row, 17 rows. The dangerous half was never the count. A head trained
with the abstention **first** and served by a runtime that reads it **last** puts every
no-evidence answer on option A, and no loss curve shows it: the training loss falls exactly as
fast either way, because the head is learning a consistent convention -- just not the one the
serving side reads.

So this tool measures the position from both ends. It checks what the code does
(:func:`_wiring`), and it *builds the defect on purpose* (:func:`_counterfactual`): a head
trained with the abstain gold at row 0, then read the way ``answer.rs`` reads it
(``noul_row = plan.rows - RESERVED_NOUL_ROWS``). If that head's argmax is a real line while
its loss sits at the same floor as the correct one, the failure mode is demonstrated rather
than argued, and the correct head's contrasting number means something.

## What this tool is NOT

It is not a gate and it promotes nothing. Every row it writes is ``quick`` with a stated
reason: the schedule is four to sixty steps over a corpus of a handful of synthetic rows.
Rule 8 excludes it from every decision, which is correct -- the question here is "does the
path execute", and the answer to that is not a number anyone should tune against.

It also does not touch the 2B lane. No Qwen weights are loaded and none are needed: rung 0 is
byte-level and from scratch, which is exactly why it is the rung that can run on this Mac.

RUN
---
    /Users/bharath/.venvs/ml/bin/python tools/rung0_toy_run.py

The repo venv deliberately carries no torch, so this refuses there with that instruction
rather than reporting a vacuous pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

# Both inserts are inline for the reason tools/bpe_line_start_collapse.py states: ruff's E402
# exemption covers `sys.path` modification before the imports, but an ordinary assignment in
# between is not one. `python/tests` is on the path for `data_fixtures`, which is the repo's
# one owner of a valid qd-mutate row and says of itself that it is "not a test module";
# restating its row here would be a second corpus free to drift from the one the suite uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - the repo venv has no torch by design
    raise SystemExit(
        "torch is not importable from this interpreter. The repo venv carries none on "
        "purpose (pyproject: torch is the optional `mac` extra). Run this with\n"
        "    /Users/bharath/.venvs/ml/bin/python tools/rung0_toy_run.py\n"
        "Refusing rather than reporting a vacuous pass."
    ) from exc

import numpy as np
from data_fixtures import mutate_row

from qd_train.artifacts import SPAN_ABSTAIN
from qd_train.byte_batch import plan_batch, span_supervision
from qd_train.byte_context import ID_PAD
from qd_train.byte_decider import ByteDecider, ByteDeciderConfig, abstain_column
from qd_train.byte_train import Rung0Model, Rung0Step, train_rung0
from qd_train.heads import (
    RESERVED_NOUL_ROWS,
    SpanPointerHead,
    plan_span_batch,
    serving_scores,
    span_head_rows,
)
from qd_train.ledger import DEFAULT_LEDGER_PATH, Environment, Ledger, Protocol, RunRecorder
from qd_train.mutate_adapter import CLEAN, MUTATION_CLASSES, parse_example, to_decision
from qd_train.run_control import CostEstimate, LRSchedule, RunControl, WallClockCap
from qd_train.schema_mirror import MAX_OPTIONS
from qd_train.trainer import SpanSupervision
from qd_train.tristate import NotRun, Ran

REPO = Path(__file__).resolve().parents[1]

#: The toy corpus is capped hard. This tool exists to answer "does it run", and a corpus
#: large enough to be interesting is a corpus large enough to hide a wiring bug behind a
#: plausible-looking curve.
MAX_ROWS = 64
MAX_STEPS = 400

#: How far above :func:`_choice_floor` a converged run may sit. Not a tuned threshold: the
#: floor is exact, and this is the slack a finite schedule leaves. A run that misses it has
#: either not converged or is not optimising what the floor is computed over -- both worth a
#: failed metric rather than a shrug.
CHOICE_FLOOR_SLACK = 0.05

#: Small enough that a handful of steps on CPU is measurable in seconds, wide enough that the
#: attention has something to do. `cua-s1-forms`' published shape is (2, 128, 4); this is one
#: notch below it because the question is the plumbing, not the capacity.
TOY = ByteDeciderConfig(width=64, n_layers=2, n_heads=4, max_context_bytes=512)


def _body(i: int) -> str:
    """A three-line function, distinct per row.

    Distinct **matters**. The first version of this corpus reused one body for every mutated
    row while cycling the gold class, so four rows carried identical bytes and three different
    labels. That corpus is not memorisable, and the loss stopped at its label entropy rather
    than at zero: measured 0.5199 against a computed floor of 0.519860, on all three seeds.
    The head was right and the data was wrong -- which is the reading a plateau invites you to
    get backwards, so :func:`_choice_floor` now computes the floor instead of assuming it.
    """
    return f"fn f{i}(x: i32) -> bool {{\n    x > {i}\n}}\n"


def _rows(n: int) -> list[dict]:
    """``n`` deterministic qd-mutate rows, alternating mutated and clean.

    A clean row carries no span, so it trains the span head's **abstention**; a mutated row
    trains a pointer. A corpus of one or the other would let a head that always abstains, or
    never does, reach a floor that reads as learning.
    """
    if not 2 <= n <= MAX_ROWS:
        raise ValueError(f"n must be in [2, {MAX_ROWS}], got {n}")
    out: list[dict] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(
                mutate_row(
                    id=f"m{i}",
                    after=_body(i),
                    span={"start_line": 2, "end_line": 2},
                    **{"class": MUTATION_CLASSES[i % 3]},
                )
            )
        else:
            out.append(mutate_row(id=f"c{i}", after=_body(i), span=None, **{"class": CLEAN}))
    return out


def _choice_floor(plan) -> float:
    """The lowest choice loss this corpus admits: its conditional label entropy.

    Rows with identical context bytes but different gold options cannot both be fitted, and
    the optimum over such a group is the empirical label distribution. Computing it turns
    "the loss fell" into "the loss reached the floor", which is the difference between a
    curve that moved and a head that is wired to the right rows -- a head pointed at the
    wrong column can still produce a falling curve, but it cannot reach this number.
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, context in enumerate(plan.context_ids):
        groups.setdefault(context, []).append(plan.choice_target[i])
    total = 0.0
    for golds in groups.values():
        counts: dict[int, int] = {}
        for g in golds:
            counts[g] = counts.get(g, 0) + 1
        n = len(golds)
        total += -sum(c * math.log(c / n) for c in counts.values())
    return total / plan.batch_size


def _plan(n: int):
    """One collated batch. Returned rather than a stream: the same batch every step is what
    makes the floor memorisation."""
    decisions = [
        to_decision(parse_example(row), max_context_bytes=TOY.max_context_bytes)
        for row in _rows(n)
    ]
    return plan_batch(
        decisions,
        pad_id=ID_PAD,
        max_context_bytes=TOY.max_context_bytes,
        max_option_bytes=96,
    )


def _control(steps: int, *, device: str) -> RunControl:
    """The cap, the schedule and the price of a local run.

    Zero is a measured fact about this host, not a way around rule 4: a Mac that is already
    bought costs nothing per hour, ``requires_human_approval`` is therefore False, and no
    GPU is rented. This used to say *"a run on a rented machine sets a real rate here"* and
    nothing made it -- ``device`` is a parameter, and the literal priced whatever arrived
    as a Mac. :meth:`CostEstimate.for_device` is what makes it true: cpu and mps are priced
    exactly as before, and anything else is refused until a caller states the rate.
    """
    cap = WallClockCap(cap_s=900.0)
    return RunControl(
        schedule=LRSchedule(peak_lr=3e-3, warmup_steps=max(1, steps // 10), total_steps=steps),
        cap=cap,
        cost=CostEstimate.for_device(cap=cap, device=device),
        grad_accum=1,
    )


def _accuracy(model: Rung0Model, plan, device: str) -> dict[str, float]:
    """Argmax agreement with the gold, on the batch just trained. Memorisation, by design.

    Read through the same helpers the serving side uses: the choice head's own placement, and
    :func:`qd_train.heads.serving_scores`, which returns exactly the rows ``qd-runtime`` will
    accept. Reading the padded matrix directly would score columns no runtime ever sees.
    """
    model.eval()
    with torch.no_grad():
        tensors_ctx = torch.tensor(plan.context_ids, dtype=torch.long, device=device)
        tensors_mask = torch.tensor(plan.context_mask, dtype=torch.bool, device=device)
        hidden = model.decider.encode_context(tensors_ctx, tensors_mask)
        logits = model.decider.score_from_hidden(
            hidden,
            tensors_mask,
            torch.tensor(plan.option_ids, dtype=torch.long, device=device),
            torch.tensor(plan.option_mask, dtype=torch.bool, device=device),
            torch.tensor(plan.n_live_options, dtype=torch.long, device=device),
        )
        choice_hit = sum(
            1
            for i, top in enumerate(logits.argmax(dim=1).tolist())
            if top == plan.choice_target[i]
        )
        span_plan = plan_span_batch(span_supervision(plan), device=device)
        start, end = model.span_head(hidden, span_plan)
        start_rows = serving_scores(start, span_plan)
        end_rows = serving_scores(end, span_plan)
        start_hit = sum(
            1
            for k in range(span_plan.n_spans)
            if int(start_rows[k].argmax()) == int(span_plan.gold_start[k])
        )
        end_hit = sum(
            1
            for k in range(span_plan.n_spans)
            if int(end_rows[k].argmax()) == int(span_plan.gold_end[k])
        )
    n = plan.batch_size
    model.train()
    return {
        "choice_top1": choice_hit / n,
        "span_start_top1": start_hit / n,
        "span_end_top1": end_hit / n,
        "n_rows": n,
    }


def _train_once(*, device: str, seed: int, steps: int, rows: int) -> dict:
    """One run on one device with one seed. Every number this tool reports comes from here."""
    torch.manual_seed(seed)
    plan = _plan(rows)
    model = Rung0Model(TOY).to(device)
    step = Rung0Step(model)
    control = _control(steps, device=device)

    t0 = time.monotonic()
    result = train_rung0([plan] * steps, step=step, control=control, seed=seed)
    wall = time.monotonic() - t0

    accuracy = _accuracy(model, plan, device)
    return {
        "device": device,
        "seed": seed,
        "steps_requested": steps,
        "optimizer_steps": result.optimizer_steps,
        "termination": result.termination,
        "params": sum(p.numel() for p in model.parameters()),
        "rows_per_batch": plan.batch_size,
        "context_width": plan.context_width,
        "choice_first": result.choice_log[0],
        "choice_last": result.choice_log[-1],
        "choice_floor": _choice_floor(plan),
        "span_first": result.span_log[0],
        "span_last": result.span_log[-1],
        "total_first": result.loss_log.losses()[0],
        "total_last": result.loss_log.losses()[-1],
        "wall_clock_s": wall,
        "steps_per_s": result.optimizer_steps / wall if wall > 0 else float("inf"),
        "padding_waste": (
            result.padding.value if isinstance(result.padding, Ran) else None
        ),
        "accuracy": accuracy,
    }


def _abstaining_supervision(length: int, starts: list[int]) -> SpanSupervision:
    """One abstaining span row over a context with line starts at ``starts``.

    ``SPAN_ABSTAIN`` rather than a hand-picked sentinel: ``SpanSupervision.__post_init__``
    checks the flag and the sentinel against each other, so a different negative number is
    refused rather than quietly trained on.
    """
    mask = np.zeros((1, length), dtype=bool)
    mask[0, starts] = True
    return SpanSupervision(
        rows=np.arange(1, dtype=np.int64),
        query_index=np.array([length - 1], dtype=np.int64),
        start=np.array([SPAN_ABSTAIN], dtype=np.int64),
        end=np.array([SPAN_ABSTAIN], dtype=np.int64),
        line_starts=mask,
        abstaining=np.array([True], dtype=bool),
    )


def _wiring(device: str) -> dict:
    """The row layout of both heads, as integers, against what ``qd-runtime`` will read.

    Nothing here is trained. These are shape facts, and they are the facts that a loss curve
    cannot report on: a head can be perfectly trained onto the wrong row.
    """
    torch.manual_seed(0)
    decider = ByteDecider(ByteDeciderConfig(width=32, n_layers=1, n_heads=2)).to(device)
    n = MAX_OPTIONS
    context = torch.randint(1, 200, (1, 24), device=device)
    mask = torch.ones((1, 24), dtype=torch.bool, device=device)
    options = torch.randint(1, 200, (1, n, 6), device=device)
    option_mask = torch.ones((1, n, 6), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits = decider(
            context, mask, options, option_mask, torch.tensor([n], device=device)
        )
    finite = torch.isfinite(logits[0]).nonzero().flatten().tolist()

    # The span side: a three-line context, abstaining, so the gold lands on the reserved row.
    span_plan = plan_span_batch(_abstaining_supervision(24, [0, 8, 16]), device=device)

    return {
        "reserved_noul_rows": RESERVED_NOUL_ROWS,
        "max_options": MAX_OPTIONS,
        "choice": {
            "n_live_options": n,
            "rows_total": int(logits.shape[1]),
            "rows_total_expected": MAX_OPTIONS + RESERVED_NOUL_ROWS,
            "abstain_column": abstain_column(n),
            "abstain_is_last_selectable": abstain_column(n) == max(finite),
            "selectable_columns": finite,
        },
        "span": {
            "line_count": 3,
            "rows_total": span_head_rows(3),
            "runtime_rows": int(span_plan.runtime_rows[0]),
            "gold_row_when_abstaining": int(span_plan.gold_start[0]),
            "abstain_row_read_as_runtime_does": span_head_rows(3) - RESERVED_NOUL_ROWS,
            "abstain_is_last": int(span_plan.gold_start[0]) == span_head_rows(3) - 1,
        },
    }


def _counterfactual(device: str, *, steps: int = 60) -> dict:
    """Build the defect on purpose, so the correct head's number means something.

    Two span heads, same data, same steps, same seed. One is trained with the abstain gold on
    the reserved row (what :func:`qd_train.heads.plan_span_batch` produces); the other with the
    gold on row 0. Both are then read the way ``crates/qd-runtime/src/answer.rs`` reads them --
    ``noul_row = plan.rows - RESERVED_NOUL_ROWS`` -- and the answer is whichever row wins.

    The point of the comparison is the loss: if the two floors are comparable while the served
    answers differ, then the training curve genuinely cannot tell the two apart, which is the
    claim ``GAP-SCHEMA-NOUL-LETTER-BUDGET``'s residual rests on.
    """
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}], got {steps}")
    width, starts, length = 16, [0, 5, 10, 15], 20
    plan = plan_span_batch(_abstaining_supervision(length, starts), device=device)
    noul_row = int(plan.runtime_rows[0]) - RESERVED_NOUL_ROWS

    out: dict[str, dict] = {}
    for label, gold in (("abstain_last_as_shipped", noul_row), ("abstain_first_defect", 0)):
        torch.manual_seed(11)
        head = SpanPointerHead(width).to(device)
        hidden = torch.randn(1, length, width, device=device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=5e-2)
        target = torch.tensor([gold], device=device)
        loss = torch.tensor(float("nan"))
        for _ in range(steps):
            start, end = head(hidden, plan)
            loss = torch.nn.functional.cross_entropy(
                start, target
            ) + torch.nn.functional.cross_entropy(end, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            start, _ = head(hidden, plan)
            served = serving_scores(start, plan)[0]
        top = int(served.argmax())
        out[label] = {
            "trained_gold_row": gold,
            "final_loss": float(loss.detach()),
            "served_argmax_row": top,
            "runtime_reads_noul_at_row": noul_row,
            "runtime_verdict": "abstain" if top == noul_row else f"line ordinal {top}",
        }
    out["context"] = {"n_lines": len(starts), "steps": steps}
    return out


def _protocol(*, seed: int, device: str, steps: int, rows: int) -> Protocol:
    """Real hashes, no markers. Rung 0 has no pretrained backbone and no BPE tokenizer, but it
    is not a ``build`` row either: its data, its codec and its architecture are all real and
    all identifiable, so each is hashed rather than marked ``n/a``."""

    def digest(obj: object) -> str:
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return Protocol(
        data_snapshot_hash=digest(_rows(rows)),
        tokenizer_hash=f"bytes:{digest({'pad': ID_PAD, 'codec': 'qd_train.byte_context'})}",
        backbone_commit=f"scratch:{digest(TOY.__dict__ if hasattr(TOY, '__dict__') else str(TOY))}",
        recipe_hash=digest({"steps": steps, "rows": rows, "device": device, "tool": "rung0_toy"}),
        seed=seed,
    )


def _record(run: dict, *, ledger: Ledger, wiring: dict, steps: int, rows: int) -> str:
    """One ledger row per run. Always ``quick``: a toy corpus on a truncated schedule."""
    recorder = RunRecorder(
        ledger,
        protocol=_protocol(seed=run["seed"], device=run["device"], steps=steps, rows=rows),
        run_kind="smoke",
        repo=REPO,
        # The device this run used, not the best one this host offers. Without it every row
        # here said "mps", including the three that ran on the CPU -- and the whole point of
        # running both is a comparison the ledger can be read back for.
        env=Environment.detect(device=run["device"]),
        # The run finished before `_record` was called; the recorder would otherwise time
        # its own metric writes.
        wall_clock_s=float(run["wall_clock_s"]),
        quick=True,
        quick_reason=(
            f"toy run: {steps} optimizer steps over {rows} synthetic rows on "
            f"{run['device']}, one batch repeated. Rule 8: excluded from every decision."
        ),
        notes=(
            "tools/rung0_toy_run.py -- does the rung 0 training path execute, and does the "
            "loss fall on a memorisable batch. Not an evaluation of the model."
        ),
    )
    with recorder:
        for name, value in (
            ("choice_loss_first", run["choice_first"]),
            ("choice_loss_last", run["choice_last"]),
            ("span_loss_first", run["span_first"]),
            ("span_loss_last", run["span_last"]),
            ("wall_clock_s", run["wall_clock_s"]),
            ("steps_per_s", run["steps_per_s"]),
        ):
            recorder.metric(
                name, Ran(passed=True, value=value, detail=f"{name} on {run['device']}")
            )
        recorder.metric(
            "choice_top1_on_trained_batch",
            Ran(
                passed=run["accuracy"]["choice_top1"] == 1.0,
                value=run["accuracy"]["choice_top1"],
                n=int(run["accuracy"]["choice_top1"] * run["accuracy"]["n_rows"]),
                n_total=run["accuracy"]["n_rows"],
                detail="argmax on the batch it memorised; not held-out and not an eval",
            ),
        )
        recorder.metric(
            "loss_fell",
            Ran(
                passed=run["total_last"] < run["total_first"],
                value=run["total_first"] - run["total_last"],
                detail=(
                    f"total {run['total_first']:.4f} -> {run['total_last']:.4f} over "
                    f"{run['optimizer_steps']} steps"
                ),
            ),
        )
        recorder.metric(
            "choice_loss_reached_its_floor",
            Ran(
                passed=run["choice_last"] <= run["choice_floor"] + CHOICE_FLOOR_SLACK,
                value=run["choice_last"] - run["choice_floor"],
                detail=(
                    f"choice {run['choice_last']:.6f} against this corpus's conditional label "
                    f"entropy {run['choice_floor']:.6f}; bar is floor + {CHOICE_FLOOR_SLACK}"
                ),
            ),
        )
        choice, span = wiring["choice"], wiring["span"]
        recorder.metric(
            "abstain_row_is_last_both_heads",
            Ran(
                passed=bool(choice["abstain_is_last_selectable"] and span["abstain_is_last"]),
                value=(
                    f"choice col {choice['abstain_column']} of {choice['rows_total']}; "
                    f"span row {span['gold_row_when_abstaining']} of {span['rows_total']}"
                ),
                detail="Python half of GAP-SCHEMA-NOUL-LETTER-BUDGET's position residual",
            ),
        )
        recorder.noul_rate = NotRun(
            reason="a toy memorisation run has no held-out population to compute a noul rate over"
        )
    if recorder.row is None:  # pragma: no cover - RunRecorder always writes on exit
        raise RuntimeError("RunRecorder exited without writing a row")
    return recorder.row.row_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    args = parser.parse_args(argv)

    if not 1 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be in [1, {MAX_STEPS}]")
    if not 2 <= args.rows <= MAX_ROWS:
        parser.error(f"--rows must be in [2, {MAX_ROWS}]")

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    else:
        print("mps: NOT RUN -- torch.backends.mps.is_available() is False on this host")

    ledger = Ledger(args.ledger)
    report: dict[str, object] = {
        "torch": torch.__version__,
        "devices_run": devices,
        "wiring": _wiring("cpu"),
        "counterfactual": _counterfactual("cpu"),
        "runs": [],
    }
    for device in devices:
        for seed in args.seeds:
            run = _train_once(device=device, seed=seed, steps=args.steps, rows=args.rows)
            run["ledger_row_id"] = _record(
                run,
                ledger=ledger,
                wiring=report["wiring"],  # type: ignore[arg-type]
                steps=args.steps,
                rows=args.rows,
            )
            report["runs"].append(run)  # type: ignore[union-attr]
            print(
                f"{device} seed={seed}: total {run['total_first']:.4f} -> "
                f"{run['total_last']:.4f}  choice {run['choice_first']:.4f} -> "
                f"{run['choice_last']:.6f} (floor {run['choice_floor']:.6f})  span "
                f"{run['span_first']:.4f} -> {run['span_last']:.6f}  top1 "
                f"{run['accuracy']['choice_top1']:.2f}/{run['accuracy']['span_start_top1']:.2f}"
                f"  {run['wall_clock_s']:.2f}s ({run['steps_per_s']:.1f} step/s)  "
                f"row {run['ledger_row_id']}"
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
