"""Kernel smoke on one H100 of the v5 box: throwaway analysis, report-only, never a ledger row.

The human's yes (lead session, ~20:06Z 2026-10-03): a short smoke on one H100 while v5's data
builds, <=30 min, <$5, separate from the v5 queue and its ledgers. Its purpose: show that the box's
x86 venv (torch 2.10.0+cu128, fla 0.5.2, triton 3.7.1, causal_conv1d 1.7.0 built here) compiles
and runs the tower's kernels on sm90 before the pre-registered probe needs them.

It builds the tower as tools/real_ft_run.py's _real_step does (qd_train.backbone.load_text_tower,
gradient checkpointing on, the master optimizer layout, F's --checkpoint-skip-layers 6, sdpa, bf16),
refuses if linear attention is on transformers' torch reference path (the trainer's own check), and
times forward+backward on random token ids at v5's shape extremes: the widest buckets
(3x10240, 4x8192) and the costliest narrow ones (737x48, 258x137). tower.model is the decoder
without the LM head, so the loss is the mean of the last hidden state: the backward runs through
every layer at every position. No LM head, no optimizer step: the peak memory here is not the probe's number,
and nothing here replaces the pre-registered probe or any estimate. Output: one JSON report.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/smoke/qd-v5build")
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from qd_train.backbone import linear_attention_on_reference_path, load_text_tower  # noqa: E402
from qd_train.memory import OptimizerSpec  # noqa: E402

BACKBONE = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)
OUT = Path(sys.argv[1])
SHAPES = [(3, 10240), (4, 8192), (737, 48), (258, 137)]
REPEATS = 3
MASTER = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)  # optimizer_spec("bf16", "master")

report: dict = {"purpose": "kernel smoke, report-only", "shapes": []}
t0 = time.time()
tower = load_text_tower(
    BACKBONE, gradient_checkpointing=True, optimizer=MASTER, attn_implementation="sdpa",
    device="cuda", dtype="bf16", rows=3, width=10240, checkpoint_skip_layers=6,
)
report["load_s"] = round(time.time() - t0, 1)
report["linear_attention_kernels"] = dict(tower.linear_attention_kernels)
slow = linear_attention_on_reference_path(tower.linear_attention_kernels)
report["on_reference_path"] = sorted(slow)
if "chunk_gated_delta_rule" in slow:
    OUT.write_text(json.dumps(report, indent=1))
    sys.exit("linear attention is on transformers' torch reference path; refusing (the trainer's check)")
model = tower.model
model.train()
props = torch.cuda.get_device_properties(0)
report["device"] = {"name": props.name, "total_memory": props.total_memory}
vocab = int(tower.vocab_size)
for rows, width in SHAPES:
    gen = torch.Generator(device="cuda").manual_seed(rows * 100_003 + width)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(REPEATS + 1):
        ids = torch.randint(0, vocab, (rows, width), device="cuda", generator=gen)
        torch.cuda.synchronize()
        t = time.time()
        # tower.model is the text decoder (Qwen3_5Model): it returns hidden states, no logits.
        out = model(input_ids=ids, use_cache=False)
        out.last_hidden_state.float().mean().backward()
        torch.cuda.synchronize()
        times.append(time.time() - t)
        model.zero_grad(set_to_none=True)
        del out
    entry = {
        "rows": rows, "width": width, "tokens": rows * width,
        "first_s": round(times[0], 3), "steady_s": [round(x, 3) for x in times[1:]],
        "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "max_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    report["shapes"].append(entry)
    print(json.dumps(entry), flush=True)
    torch.cuda.empty_cache()
report["total_s"] = round(time.time() - t0, 1)
OUT.write_text(json.dumps(report, indent=1))
print("KERNEL SMOKE OK", json.dumps({k: report[k] for k in ("load_s", "total_s", "on_reference_path")}))
