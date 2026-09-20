"""Measure a real training step's peak CUDA memory against what `memory.py` predicts.

``HANDOFF/backbone-2026-09-20.md`` names this as the first CUDA box's job:

    On the first CUDA box, close the arithmetic. Record ``torch.cuda.max_memory_allocated``
    against ``StepFootprint.total_bytes`` for the same ``rows``/``width``. That single
    comparison settles the 35% ``SAFETY_FRACTION`` and closes
    ``GAP-BACKBONE-GH200-FOOTPRINT-IS-ARITHMETIC-NOT-MEASURED``.

**What is compared, and what is not.** ``StepFootprint`` keeps every component separate on
purpose -- *"a number that cannot be attributed cannot be argued with"* -- so this compares
the backbone terms it predicts (params, grads, optimizer, master, activations, recurrent
state) against a real forward, backward and optimizer step over the real tower. It does
**not** drive ``SpanPointerHead`` or ``fused_linear_cross_entropy``, so ``loss_head_bytes``
and ``score_matrix_bytes`` are excluded from the predicted side and named in the output
rather than folded in. Comparing a total that includes two terms the run never exercised
would be the same "one name, two quantities" error this repository keeps finding.

**Why the optimizer state is measured on the second step.** AdamW allocates ``exp_avg`` and
``exp_avg_sq`` lazily, inside the first ``step()``. A peak read after one step measures a
step that was still allocating; every step after it is the steady state. Both are printed.

**Why ADAMW_BF16 and not ADAMW_FP32.** An earlier version of this script budgeted with
``ADAMW_FP32`` -- two 4-byte states per parameter -- while loading a **bf16** tower, and
measured 14.02 GiB against the 28.04 GiB it had predicted. Plain ``torch.optim.AdamW`` keeps
``exp_avg`` and ``exp_avg_sq`` in the *parameter's* dtype and holds no fp32 master copy, so a
bf16 tower gets 2-byte states: **4 B/param of optimizer state**, and 8 B/param once the bf16
weights and bf16 grads are counted with them -- against ADAMW_FP32's 16 B/param total.
``load_text_tower`` now refuses that mismatch outright, because a footprint that does not
describe the run cannot be used to decide whether the next run fits, and deciding that is the
only reason it exists.

Both figures are stated because an earlier draft of this sentence carried only the 8 and
attached it to the *states*, which is the total. Measured on the GH200 over the real tower:
optimizer state alone is 7.01 GiB = 4.00 B/param, and 2 + 2 + 4 = 8 B/param is the 14.02 GiB
above. An isolated probe confirms the composition -- exactly two state tensors, ``exp_avg``
and ``exp_avg_sq``, both ``torch.bfloat16`` at 2 bytes, plus a scalar ``step``, and no fp32
master. One name, two quantities, in a file whose subject is keeping them apart.

That is not the same as choosing bf16 moments. ``exp_avg_sq`` accumulates across a whole run
and bf16 has 8 bits of mantissa, and ``tools/moment_precision.py`` measures what that costs:
at ``beta2=0.999`` the per-step relative increment is ``1e-3`` against bfloat16's ``2**-8``
spacing, so the second moment settles at 0.5 against a true 1.0 after 383 steps and cannot
follow a later change in gradient scale. ``qd_train.optim.MasterWeightAdamW`` implements the
alternative, and ``--optimizer master`` measures it here: the choice between them is a recipe
decision with a measured price on both sides, which is the only honest way to have one.

Run it on the box, not here::

    PYTHONPATH=/home/ubuntu/qwen-decision/python /home/ubuntu/qd-venv/bin/python \
        /home/ubuntu/qwen-decision/tools/gh200_footprint.py --optimizer master
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch

from qd_train.backbone import load_text_tower
from qd_train.memory import ADAMW_BF16, QWEN3_5_2B_TEXT, OptimizerSpec, estimate_step
from qd_train.optim import build_optimizer

#: The fp32-master recipe: bf16 weights and grads, an fp32 master copy, fp32 moments.
#: 16 B/param against ADAMW_BF16's 8. `qd_train.optim.MasterWeightAdamW` builds it.
ADAMW_MASTER: OptimizerSpec = OptimizerSpec(
    "AdamW+master", 2, 4, keeps_fp32_master=True
)

RECIPES = {"bf16": ADAMW_BF16, "master": ADAMW_MASTER}

SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)

#: The real shard set's bucket ceilings, plus two narrow ones for the shape of the curve.
WIDTHS = (2048, 4096, 8192, 14759, 34522)

GiB = 1024**3


def backbone_predicted_bytes(
    rows: int, width: int, spec: OptimizerSpec = ADAMW_BF16
) -> tuple[int, dict[str, int]]:
    """The footprint terms this script actually exercises, and the ones it does not."""
    fp = estimate_step(QWEN3_5_2B_TEXT, rows=rows, width=width, optimizer=spec)
    exercised = {
        "param_bytes": fp.param_bytes,
        "grad_bytes": fp.grad_bytes,
        "optimizer_bytes": fp.optimizer_bytes,
        "master_bytes": fp.master_bytes,
        "activation_bytes": fp.activation_bytes,
        "recurrent_state_bytes": fp.recurrent_state_bytes,
    }
    excluded = {
        "loss_head_bytes": fp.loss_head_bytes,
        "score_matrix_bytes": fp.score_matrix_bytes,
        "safety_bytes": fp.safety_bytes,
    }
    return sum(exercised.values()), {**exercised, **excluded}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--optimizer",
        choices=sorted(RECIPES),
        default="bf16",
        help=(
            "which recipe to measure. 'bf16' is what torch.optim.AdamW builds for a bf16 "
            "tower (8 B/param all-in); 'master' is the fp32-master recipe (16 B/param), "
            "whose second moment does not freeze -- see tools/moment_precision.py."
        ),
    )
    args = ap.parse_args(argv)
    spec = RECIPES[args.optimizer]

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This script exists to measure one; refusing to report a "
            "CPU number under a name that says cuda."
        )
    dev = torch.cuda.get_device_properties(0)
    print(f"device   : {dev.name}, {dev.total_memory / GiB:.2f} GiB, sm_{dev.major}{dev.minor}")
    print(f"torch    : {torch.__version__}")
    print()

    tower = load_text_tower(
        SNAPSHOT,
        gradient_checkpointing=True,
        optimizer=spec,
        device="cuda",
        dtype="bf16",
        rows=1,
        width=WIDTHS[-1],
    )
    print(f"loaded   : {tower.n_tensors_loaded} tensors, vocab {tower.vocab_size}, "
          f"hidden {tower.hidden_size}, grad_ckpt {tower.gradient_checkpointing}")
    trainable = sum(p.numel() for p in tower.model.parameters() if p.requires_grad)
    print(f"trainable: {trainable:,} parameters")

    print(f"recipe   : {args.optimizer} -- {spec}")
    opt = build_optimizer(list(tower.model.parameters()), spec=spec, lr=1e-5)
    vocab = tower.vocab_size
    rows = 1
    results = []

    for width in WIDTHS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, vocab, (rows, width), device="cuda")
        peaks = []
        try:
            for _ in range(2):
                out = tower.model(input_ids=ids)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                # A scalar that touches every hidden state, so the backward walks the whole
                # tower. NOT the real objective -- the CE and span-head terms are excluded
                # from the predicted side too, and named in the output.
                loss = hidden.float().pow(2).mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                peaks.append(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            measured = peaks[-1]
            predicted, parts = backbone_predicted_bytes(rows, width, spec)
            results.append(
                {
                    "width": width,
                    "rows": rows,
                    "measured_bytes": measured,
                    "first_step_bytes": peaks[0],
                    "predicted_backbone_bytes": predicted,
                    "ratio_predicted_over_measured": predicted / measured,
                    "excluded_from_prediction": {
                        k: parts[k]
                        for k in ("loss_head_bytes", "score_matrix_bytes", "safety_bytes")
                    },
                }
            )
            print(
                f"  width {width:6d}  measured {measured / GiB:7.2f} GiB   "
                f"predicted(backbone) {predicted / GiB:7.2f} GiB   "
                f"pred/meas {predicted / measured:5.2f}x"
            )
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"width": width, "rows": rows, "oom": str(exc)[:200]})
            print(f"  width {width:6d}  OOM -- that is a measurement, not a failure")
            torch.cuda.empty_cache()
        del ids

    # What the optimizer actually built, against what ADAMW_FP32 describes.
    # `MasterWeightAdamW.state` delegates to the inner optimizer, so this reads the real
    # moments either way. The fp32 MASTER copies are counted separately below, because they
    # are not optimizer state -- they are a second copy of the weights, and folding them in
    # would be one name over two quantities again.
    state_bytes = sum(
        v.numel() * v.element_size()
        for s in opt.state.values()
        for v in s.values()
        if torch.is_tensor(v)
    )
    master_bytes = 0
    if spec.keeps_fp32_master:
        master_bytes = sum(
            m.numel() * m.element_size() for m in opt.state_dict()["masters"]
        )
        print(f"fp32 master copies: {master_bytes / GiB:.2f} GiB "
              f"= {master_bytes / trainable:.2f} B/param (counted apart from the states)")
    per_param = state_bytes / trainable if trainable else 0.0
    print()
    print(f"optimizer state actually allocated: {state_bytes / GiB:.2f} GiB "
          f"= {per_param:.2f} B/param over {trainable:,} params")
    described = spec.states_per_param * spec.state_bytes
    print(f"{args.optimizer} spec as memory.py describes it: {spec}")
    print(f"  states the spec describes: {described} B/param; measured {per_param:.2f}")
    agrees = abs(per_param - described) < 0.01
    print("  AGREES" if agrees else "  DISAGREES -- the arithmetic describes a layout "
          "this run did not build")

    out = Path(f"/home/ubuntu/gh200_footprint_{args.optimizer}.json")
    out.write_text(json.dumps({
        "device": dev.name,
        "recipe": args.optimizer,
        "optimizer_spec": {
            "name": spec.name, "states_per_param": spec.states_per_param,
            "state_bytes": spec.state_bytes, "keeps_fp32_master": spec.keeps_fp32_master,
        },
        "optimizer_state_bytes_measured": state_bytes,
        "optimizer_state_bytes_per_param_measured": per_param,
        "optimizer_state_bytes_per_param_described": described,
        "spec_agrees_with_measurement": agrees,
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
