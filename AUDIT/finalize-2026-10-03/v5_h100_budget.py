"""Throwaway analysis: memory.py's step estimate for F's and v5's costliest batch shapes.

Reproduces F's recorded device_budget value (ft row 973cd4e3: 68,510,315,980 B) from the
v4 shard set's widths and remap, then prices v5's widest bucket (10,240) at the same
batch_tokens, so the H100's ~79 GiB can be judged before anything is rented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

W = Path("/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt")
sys.path.insert(0, str(W / "python"))

from qd_train.backbone import activation_model, uncheckpointed_layers  # noqa: E402
from qd_train.memory import QWEN3_5_2B_TEXT, OptimizerSpec, estimate_step  # noqa: E402

GIB = 1024**3
F_RECORDED = 68_510_315_980
BATCH_TOKENS = 35403
SHARDS = Path("/Users/bharath/qd-campaign/phase4-v4-2026-10-01/shards/train")

layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(24)]
skip = uncheckpointed_layers(6, layer_types)
master = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
acts = activation_model(gradient_checkpointing=True, skip_layers=skip, layer_types=layer_types)

header = json.loads((SHARDS / "header.json").read_text())
remap = json.loads((SHARDS / "remap.json").read_text())
print("header keys:", sorted(header)[:40])
print("remap keys:", sorted(remap)[:20] if isinstance(remap, dict) else type(remap))


def est(rows: int, width: int, vocab: int) -> int:
    return estimate_step(
        QWEN3_5_2B_TEXT, rows=rows, width=width, optimizer=master,
        param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16",
        activations=acts, vocab_size=vocab,
    ).total_bytes


def costliest(widths: list[int], vocab: int) -> tuple[int, int, int]:
    best = (0, 0, 0)
    for w in widths:
        rows = max(1, BATCH_TOKENS // w)
        b = est(rows, w, vocab)
        if b > best[0]:
            best = (b, rows, w)
    return best


if __name__ == "__main__":
    print("skip layers:", skip)
    vocab = int(remap["vocab_size"])
    f_widths = [int(w) for w in header["buckets"]]
    b, r, w = costliest(f_widths, vocab)
    print(f"F: remap vocab {vocab}; costliest {r}x{w}: {b} B = {b / GIB:.2f} GiB; "
          f"recorded {F_RECORDED} ({'MATCH' if b == F_RECORDED else 'DIFFERENT'})")
    shapes = {(max(1, BATCH_TOKENS // x), x) for x in f_widths}
    for rr, ww in sorted(shapes, key=lambda t: -est(*t, vocab))[:4]:
        print(f"   F shape {rr}x{ww}: {est(rr, ww, vocab) / GIB:.2f} GiB")
    for ww in (7936, 8192, 8704, 9216, 9728, 10240):
        rr = max(1, BATCH_TOKENS // ww)
        print(f"v5 candidate widest {rr}x{ww}: {est(rr, ww, vocab) / GIB:.2f} GiB")
    # Narrower buckets than F's 137, which v5's short pool rows could add.
    for ww in (48, 64, 80, 100, 120, 137, 160):
        rr = BATCH_TOKENS // ww
        print(f"narrow {rr}x{ww}: {est(rr, ww, vocab) / GIB:.2f} GiB")
