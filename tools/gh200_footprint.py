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
and bf16 has 8 bits of mantissa; the fp32-moment recipe needs ``keeps_fp32_master=True``,
which nothing implements yet. This script measures what is built, and the gap between that and
what should be built is a recipe decision, not an error to route around.

Run it on the box, not here::

    PYTHONPATH=/home/ubuntu/qwen-decision/python /home/ubuntu/qd-venv/bin/python \
        /home/ubuntu/qwen-decision/tools/gh200_footprint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch

from qd_train.backbone import load_text_tower
from qd_train.memory import ADAMW_BF16, QWEN3_5_2B_TEXT, estimate_step

SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)

#: The real shard set's bucket ceilings, plus two narrow ones for the shape of the curve.
WIDTHS = (2048, 4096, 8192, 14759, 34522)

GiB = 1024**3


def backbone_predicted_bytes(rows: int, width: int) -> tuple[int, dict[str, int]]:
    """The footprint terms this script actually exercises, and the ones it does not."""
    fp = estimate_step(QWEN3_5_2B_TEXT, rows=rows, width=width, optimizer=ADAMW_BF16)
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


def main() -> int:
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
        optimizer=ADAMW_BF16,
        device="cuda",
        dtype="bf16",
        rows=1,
        width=WIDTHS[-1],
    )
    print(f"loaded   : {tower.n_tensors_loaded} tensors, vocab {tower.vocab_size}, "
          f"hidden {tower.hidden_size}, grad_ckpt {tower.gradient_checkpointing}")
    trainable = sum(p.numel() for p in tower.model.parameters() if p.requires_grad)
    print(f"trainable: {trainable:,} parameters")

    opt = torch.optim.AdamW(tower.model.parameters(), lr=1e-5)
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
            predicted, parts = backbone_predicted_bytes(rows, width)
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
    state_bytes = sum(
        v.numel() * v.element_size()
        for s in opt.state.values()
        for v in s.values()
        if torch.is_tensor(v)
    )
    per_param = state_bytes / trainable if trainable else 0.0
    print()
    print(f"optimizer state actually allocated: {state_bytes / GiB:.2f} GiB "
          f"= {per_param:.2f} B/param over {trainable:,} params")
    print(f"ADAMW_BF16 as memory.py describes it: {ADAMW_BF16}")
    print("if these disagree, the arithmetic describes a recipe not yet implemented")

    out = Path("/home/ubuntu/gh200_footprint.json")
    out.write_text(json.dumps({"device": dev.name, "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
