"""How many rows actually fit, against how many ``max_positions_that_fit`` says.

Two residuals were left open when the footprint was first measured, and both are
load-bearing:

1. Every width measured so far ran at **rows=1**. The decision that justified renting a
   96 GB box was "8 rows fit at width 34,522 and 9 do not", and that decision was
   ``estimate_step`` arithmetic.
2. Every width measured so far ran with **gradient checkpointing ON**. The one verdict
   sitting inside the safety band -- 34,522 at ``recompute='none'``, 84.53 GB enumerated
   against 108.15 GB with the allowance -- is the band the run never entered.

This settles both by measurement. It walks rows upward until the device refuses, so an
**OOM is the answer, not a failure**: the largest row count that completes a real
forward+backward+step is the measured capacity, and the first that does not is the wall.
``torch.cuda.OutOfMemoryError`` is caught per attempt and the allocator is reset, because a
sweep that dies on its first refusal measures nothing past it.

Run on the box::

    PYTHONPATH=/home/ubuntu/qwen-decision/python /home/ubuntu/qd-venv/bin/python \
        /home/ubuntu/qwen-decision/tools/gh200_rows.py

With no flags it measures what it always measured: the 2B snapshot under the plain bf16
recipe at the two widths below, the backward driven by a proxy loss. ``--snapshot`` (a larger
base, budgeted from its own checkpoint headers by ``qd_train.memory.spec_from_checkpoint``),
``--optimizer`` (any recipe ``real_ft_run.py --optimizer`` takes), ``--width``, ``--loss
fused-ce`` (the trainer's own loss head, which the proxy never builds) and ``--out`` exist for
the next plan's P2: the 4B under the compensated recipe at v5's widest bucket.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from qd_train.backbone import load_text_tower
from qd_train.fused_ce import fused_linear_cross_entropy
from qd_train.memory import (
    ADAMW_BF16,
    ADAMW_KAHAN,
    ADAMW_MASTER,
    QWEN3_5_2B_TEXT,
    ActivationModel,
    ModelSpec,
    OptimizerSpec,
    estimate_step,
    max_positions_that_fit,
    spec_from_checkpoint,
)
from qd_train.optim import build_optimizer

SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)
GiB = 1024**3

#: The widths the decision was made at: the real shard set's widest bucket, and a mid one.
CASES = ((34522, "the widest real bucket"), (8192, "a mid bucket"))

#: The recipes a sweep can measure under, by ``real_ft_run.py --optimizer`` name, from the one
#: owner of each spec.
RECIPES: dict[str, OptimizerSpec] = {
    "bf16": ADAMW_BF16, "master": ADAMW_MASTER, "kahan": ADAMW_KAHAN,
}


@dataclass(frozen=True)
class Sweep:
    """What one invocation measures: which tower, under which recipe, at which widths."""

    snapshot: Path
    model: ModelSpec
    recipe: str
    optimizer: OptimizerSpec
    cases: tuple[tuple[int, str], ...]
    out: Path
    #: What the backward is driven by. ``proxy`` is ``hidden.float().pow(2).mean()``, what
    #: this tool always ran: it never builds the loss head, so ``estimate_step``'s
    #: ``loss_head_bytes`` (counted in the prediction) goes unmeasured -- the 0.84-0.97x
    #: GH200 figure is backbone-only. ``fused-ce`` drives the real
    #: ``fused_linear_cross_entropy`` against the tied head, so the measurement covers it
    #: (GAP-LOSS-HEAD-TERM-NEVER-MEASURED-2026-10-06).
    loss: str = "proxy"


#: The ``--loss`` choices; see [`Sweep.loss`].
LOSSES: tuple[str, ...] = ("proxy", "fused-ce")


def sweep_from_args(argv: list[str] | None = None) -> Sweep:
    """Parse the flags. With none, exactly the sweep this tool always ran."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--optimizer", choices=sorted(RECIPES), default="bf16")
    ap.add_argument("--width", type=int, action="append", default=None,
                    help="a width to sweep; repeatable. Default: 34,522 and 8,192.")
    ap.add_argument("--loss", choices=LOSSES, default="proxy",
                    help="what drives the backward; fused-ce measures the loss head too")
    ap.add_argument("--out", type=Path, default=Path("/home/ubuntu/gh200_rows.json"))
    args = ap.parse_args(argv)
    widths = args.width
    if widths is not None and any(w <= 0 for w in widths):
        ap.error(f"--width must be positive, got {widths}")
    cases = CASES if widths is None else tuple((w, "--width") for w in widths)
    # The 2B's measured constant when the snapshot is the 2B; any other snapshot is budgeted
    # from its own headers, because a 2B spec would describe a tower that was not loaded
    # (load_text_tower refuses that pair).
    model = QWEN3_5_2B_TEXT if args.snapshot == SNAPSHOT else spec_from_checkpoint(args.snapshot)
    return Sweep(
        args.snapshot, model, args.optimizer, RECIPES[args.optimizer], cases, args.out,
        args.loss,
    )


#: Stop climbing here. 64 rows at 34,522 is 2.2M positions, far past anything this device
#: admits; reaching it means the sweep is not measuring what it thinks it is.
MAX_ROWS = 64


def forward_backward(tower, ids: torch.Tensor, *, loss: str) -> float:
    """The forward and the backward one probe step runs, on whatever device ``ids`` is on.
    Returns the loss value."""
    if loss not in LOSSES:
        raise ValueError(f"no loss {loss!r}; known: {LOSSES}")
    out = tower.model(input_ids=ids)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    if loss == "fused-ce":
        # The trainer's own loss over the tied head, at its own chunking.
        n = ids.numel()
        targets = torch.randint(0, tower.vocab_size, (n,), device=ids.device)
        value = fused_linear_cross_entropy(
            hidden.reshape(n, -1), tower.lm_head_weight, targets
        )
    else:
        value = hidden.float().pow(2).mean()
    value.backward()
    return float(value.detach())


def attempt(tower, opt, rows: int, width: int, *, loss: str = "proxy") -> tuple[bool, int, str]:
    """One real step at this shape. Returns (fitted, peak_bytes, detail)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        ids = torch.randint(0, tower.vocab_size, (rows, width), device="cuda")
        forward_backward(tower, ids, loss=loss)
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated()
        del ids
        return True, peak, "fwd+bwd+step ok"
    except torch.cuda.OutOfMemoryError as exc:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, 0, str(exc).split("\n")[0][:160]


def main(argv: list[str] | None = None) -> int:
    sweep = sweep_from_args(argv)
    spec, model, cases = sweep.optimizer, sweep.model, sweep.cases
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This script exists to measure one; refusing to report a "
            "CPU number under a name that says cuda."
        )
    dev = torch.cuda.get_device_properties(0)
    # FREE bytes, not total: the CUDA context is already resident and the allocator cannot
    # use every byte the device reports. Measured here at 3.3%-12.1% depending on the shape,
    # which is why `max_positions_that_fit` is fed what is usable and its answer is still
    # treated as an upper bound to be confirmed by a real step.
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    device_bytes = int(free_bytes)
    print(f"device: {dev.name}, {total_bytes / GiB:.2f} GiB total, "
          f"{device_bytes / GiB:.2f} GiB free to the allocator")
    print(f"torch : {torch.__version__}\n")

    # Named rather than inherited: attention is where the activation memory is, so a
    # footprint taken on another kernel is a measurement of something else.
    tower = load_text_tower(
        sweep.snapshot, gradient_checkpointing=True, optimizer=spec,
        attn_implementation="sdpa",
        device="cuda", dtype="bf16", rows=1, width=cases[0][0], spec=model,
    )
    # total_steps=1 -- a row-count feasibility probe takes one step per case, so the
    # bf16 moment is nowhere near the step at which it stops tracking.
    opt = build_optimizer(
        list(tower.model.parameters()), spec=spec, lr=1e-5, total_steps=1
    )
    print(f"tower : {model.name}, recipe {sweep.recipe} ({spec.name}), "
          f"{tower.n_tensors_loaded} tensors, vocab {tower.vocab_size}, "
          f"grad_ckpt {tower.gradient_checkpointing}\n")

    results = []
    for width, label in cases:
        predicted_positions = max_positions_that_fit(
            model, device_bytes=device_bytes, width=width,
            optimizer=spec, param_dtype="bf16", grad_dtype="bf16",
            activation_dtype="bf16",
        )
        predicted_rows = predicted_positions // width
        print(f"width {width} ({label}): estimate_step admits {predicted_rows} row(s) "
              f"[{predicted_positions:,} positions]")

        fitted = 0
        rows = 1
        while rows <= MAX_ROWS:
            ok, peak, detail = attempt(tower, opt, rows, width, loss=sweep.loss)
            mark = "fits" if ok else "OOM "
            extra = f"peak {peak / GiB:6.2f} GiB" if ok else detail[:70]
            print(f"    rows {rows:3d}  {mark}  {extra}")
            if not ok:
                break
            fitted = rows
            rows += 1
        results.append({
            "width": width, "measured_max_rows": fitted,
            "predicted_max_rows": predicted_rows,
            "predicted_positions": predicted_positions,
        })
        verdict = (
            "matches" if fitted == predicted_rows
            else ("CONSERVATIVE, the budget under-admits" if fitted > predicted_rows
                  else "OPTIMISTIC, the budget admits more than fits")
        )
        print(f"  -> measured {fitted}, predicted {predicted_rows}: {verdict}\n")

    # The band nobody entered: recompute='none' at the widest bucket. The first tower and its
    # optimizer are released first: two towers and two optimizers at once would measure a
    # device holding both, which no run does.
    del opt, tower
    torch.cuda.empty_cache()
    print("recompute='none' at the widest bucket -- the verdict inside the safety band:")
    none_fp = estimate_step(
        model, rows=1, width=cases[0][0], optimizer=spec,
        param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16",
        activations=ActivationModel(recompute="none"),
    )
    print(f"  enumerated {none_fp.total_bytes / GiB:7.2f} GiB "
          f"(with allowance {(none_fp.total_bytes + none_fp.safety_bytes) / GiB:7.2f} GiB) "
          f"against {device_bytes / GiB:.2f} GiB free")
    no_ckpt = load_text_tower(
        sweep.snapshot, gradient_checkpointing=False, optimizer=spec,
        attn_implementation="sdpa",
        device="cuda", dtype="bf16", rows=1, width=cases[0][0], spec=model,
    )
    no_opt = build_optimizer(
        list(no_ckpt.model.parameters()), spec=spec, lr=1e-5, total_steps=1
    )
    ok, peak, detail = attempt(no_ckpt, no_opt, 1, cases[0][0], loss=sweep.loss)
    if ok:
        print(f"  MEASURED: 1 row FITS at {peak / GiB:.2f} GiB -- the budget said it would "
              "not, so the enumerated activation term over-states at recompute='none'")
    else:
        print(f"  MEASURED: 1 row does NOT fit -- {detail[:100]}")
        print("  the budget and the device agree; this is a measurement, not a failure")
    results.append({
        "width": cases[0][0], "recompute": "none", "rows": 1,
        "fits": ok, "peak_bytes": peak if ok else None,
        "enumerated_bytes": none_fp.total_bytes,
        "enumerated_with_allowance_bytes": none_fp.total_bytes + none_fp.safety_bytes,
        "device_bytes": device_bytes,
    })

    out = sweep.out
    out.write_text(json.dumps({
        "device": dev.name, "model": model.name, "params_total": model.params_total,
        "recipe": sweep.recipe, "optimizer_spec": spec.name, "loss": sweep.loss,
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
