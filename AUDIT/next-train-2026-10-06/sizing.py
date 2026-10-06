"""Text-tower sizes for Qwen3.5 2B/4B/9B-Base and what each optimizer recipe needs (analysis only).

Throwaway analysis for HANDOFF/next-training-plan-2026-10-06.md; not shipped code.

Method:
- The parameter count is derived from each config by formula. The formula is checked first
  against the 2B checkpoint's safetensors header, which is the measured count; the script refuses
  to print anything if the two disagree.
- 4B/9B configs: copied from huggingface.co/Qwen/Qwen3.5-{4B,9B}-Base/raw/main/config.json
  through a summarising fetcher on 2026-10-06. That makes them [fetched], not [V]. Only the
  fields the formula reads are copied.
- Memory: qd_train.memory.estimate_step, the repository's own arithmetic. The static state
  (weights + grads + optimizer) is exact bookkeeping. The activation term is an enumerated lower
  bound plus the module's allowance. Measured on a GH200, it under-predicts the backbone by
  0.84-0.97x (GAP-MEMORY-PREDICTS-084-TO-097X-OF-MEASURED-BACKBONE-FOOTPRINT-ON-GH200), so
  every activation figure here is a floor, not a fit.

Run from the repo root. It prints to stdout; sizing.txt is that output, redirected:
    /Users/bharath/.venvs/ml/bin/python AUDIT/next-train-2026-10-06/sizing.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from qd_train.memory import (  # noqa: E402
    ADAMW_BF16,
    ADAMW_KAHAN,
    ADAMW_MASTER,
    ActivationModel,
    ModelSpec,
    estimate_step,
    max_positions_that_fit,
)

SNAP_2B = Path(
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)

# Fields read by the formula, from the fetched configs (text_config; tie from the top level
# where text_config does not carry it). 24/32 layers, every 4th full attention.
FETCHED = {
    "Qwen3.5-4B-Base": {
        "hidden_size": 2560, "intermediate_size": 9216, "num_hidden_layers": 32,
        "full_attention_interval": 4, "num_attention_heads": 16, "num_key_value_heads": 4,
        "head_dim": 256, "attn_output_gate": True, "linear_num_key_heads": 16,
        "linear_key_head_dim": 128, "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "vocab_size": 248320, "tie_word_embeddings": True,
        "mamba_ssm_dtype": "float32",
    },
    "Qwen3.5-9B-Base": {
        "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 32,
        "full_attention_interval": 4, "num_attention_heads": 16, "num_key_value_heads": 4,
        "head_dim": 256, "attn_output_gate": True, "linear_num_key_heads": 16,
        "linear_key_head_dim": 128, "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "vocab_size": 248320, "tie_word_embeddings": False,
        "mamba_ssm_dtype": "float32",
    },
}


def text_tower_params(t: dict) -> tuple[int, int]:
    """(total, embedding) for the text tower, by tensor, from the 2B header's tensor list."""
    h, inter, n = t["hidden_size"], t["intermediate_size"], t["num_hidden_layers"]
    every = t["full_attention_interval"]
    n_full = sum(1 for i in range(n) if (i + 1) % every == 0)
    n_lin = n - n_full
    q_out = t["num_attention_heads"] * t["head_dim"] * (2 if t["attn_output_gate"] else 1)
    kv = t["num_key_value_heads"] * t["head_dim"]
    full = q_out * h + 2 * kv * h + t["num_attention_heads"] * t["head_dim"] * h + 2 * t["head_dim"]
    nk, dk = t["linear_num_key_heads"], t["linear_key_head_dim"]
    nv, dv = t["linear_num_value_heads"], t["linear_value_head_dim"]
    qkv = 2 * nk * dk + nv * dv
    lin = (
        qkv * h            # in_proj_qkv
        + nv * dv * h      # in_proj_z
        + 2 * nv * h       # in_proj_a, in_proj_b
        + qkv * t["linear_conv_kernel_dim"]  # conv1d (depthwise)
        + 2 * nv           # dt_bias, A_log
        + dv               # gated norm
        + h * nv * dv      # out_proj
    )
    mlp = 3 * h * inter
    norms = 2 * h
    embedding = t["vocab_size"] * h
    head = 0 if t["tie_word_embeddings"] else embedding
    total = n_full * (full + mlp + norms) + n_lin * (lin + mlp + norms) + h + embedding + head
    return total, embedding


def header_count(snapshot: Path) -> int:
    total = 0
    for shard in sorted(snapshot.glob("*.safetensors")):
        with shard.open("rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        header.pop("__metadata__", None)
        for name, spec in header.items():
            if name.startswith(("model.visual.", "mtp.")):
                continue
            n = 1
            for d in spec["shape"]:
                n *= d
            total += n
    return total


def spec(name: str, t: dict, total: int, embedding: int) -> ModelSpec:
    n = t["num_hidden_layers"]
    n_full = sum(1 for i in range(n) if (i + 1) % t["full_attention_interval"] == 0)
    return ModelSpec(
        name=f"{name} (text tower, derived from config)",
        hidden_size=t["hidden_size"], intermediate_size=t["intermediate_size"],
        n_full_attention_layers=n_full, n_linear_attention_layers=n - n_full,
        q_heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"],
        head_dim=t["head_dim"], attn_output_gate=t["attn_output_gate"],
        linear_heads=t["linear_num_key_heads"], linear_head_dim=t["linear_key_head_dim"],
        linear_value_heads=t["linear_num_value_heads"],
        linear_value_head_dim=t["linear_value_head_dim"],
        vocab_size=t["vocab_size"], params_total=total, params_embedding=embedding,
        tied_embedding=t["tie_word_embeddings"],
        recurrent_state_bytes=4 if t["mamba_ssm_dtype"] == "float32" else 2,
    )


def main() -> int:
    cfg2 = json.loads((SNAP_2B / "config.json").read_text(encoding="utf-8"))["text_config"]
    derived2, emb2 = text_tower_params(cfg2)
    measured2 = header_count(SNAP_2B)
    print(f"2B formula {derived2:,} vs header {measured2:,}")
    if derived2 != measured2:
        print("REFUSED: the formula does not reproduce the measured 2B count")
        return 1

    models = [("Qwen3.5-2B-Base", cfg2, derived2, emb2)]
    for name, t in FETCHED.items():
        total, emb = text_tower_params(t)
        models.append((name, t, total, emb))

    recipes = [
        ("master (v5: fp32 master + fp32 moments)", ADAMW_MASTER),
        ("kahan (bf16 + bf16 compensation + fp32 moments)", ADAMW_KAHAN),
        ("bf16 (torch AdamW over bf16; moments stall)", ADAMW_BF16),
    ]
    gib = 1024**3
    # v5's batch: 35,403 tokens per batch (recipe batch_tokens), checkpoint_skip_layers 6,
    # which on 2B leaves its 6 full-attention layers un-checkpointed (backbone.activation_model).
    # Under the master recipe this reproduces v5's recorded device_budget (68,357,417,011 B)
    # to within 0.24%: 68,192,214,681 B.
    acts = ActivationModel(recompute="full", attention="flash", retained_full_layers=6)
    rows, width = 1, 35403
    # The worst measured under-prediction of the backbone footprint (GH200, 0.84x).
    worst = 0.84
    print()
    print("| Model | Text-tower params | Recipe | B/param | Static GiB | Step at v5's batch, GiB "
          "(predicted / ÷0.84) |")
    print("|---|---|---|---|---|---|")
    for name, t, total, emb in models:
        ms = spec(name, t, total, emb)
        for label, opt in recipes:
            bpp = opt.bytes_per_param(param_bytes=2, grad_bytes=2)
            f0 = estimate_step(ms, rows=1, width=1, activations=acts, optimizer=opt,
                               param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16")
            f = estimate_step(ms, rows=rows, width=width, activations=acts, optimizer=opt,
                              param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16")
            print(f"| {name} | {total:,} | {label} | {bpp} | {f0.static_bytes / gib:.1f} | "
                  f"{f.total_bytes / gib:.1f} / {f.total_bytes / gib / worst:.1f} |")

    # Tokens per batch that fit, per device, with every layer checkpointed (skip 0), at v5's
    # widest bucket (9,638). Budget = 0.84 x the device's usable memory, so the predictor's
    # worst measured under-prediction is already paid for. Usable memory: H100 is the 69.80
    # GiB torch.cuda.mem_get_info reported free on v5's box (V); GH200 is the 87.95 GiB peak
    # that completed on 09-20 (V, GAP-MEMORY-CAPACITY-PREDICTOR-ADMITS-MORE-ROWS-THAN-FIT);
    # H200 (141 GB) and B200 (180 GB) are nominal less 7%, never measured here (U).
    devices = [("H100 80GB", 69.80), ("GH200 96GB", 87.95), ("H200 141GB", 141e9 * 0.93 / gib),
               ("B200 180GB", 180e9 * 0.93 / gib)]
    full = ActivationModel(recompute="full", attention="flash")
    print()
    print("Tokens per batch that fit at width 9,638, every layer checkpointed, budget 0.84 x usable"
          " (v5 trained 35,403):")
    print("| Model | Recipe | " + " | ".join(d for d, _ in devices) + " |")
    print("|---|---|" + "---|" * len(devices))
    for name, t, total, emb in models:
        ms = spec(name, t, total, emb)
        for label, opt in recipes:
            cells = []
            for _dev, usable in devices:
                pos = max_positions_that_fit(
                    ms, device_bytes=int(usable * gib * worst), width=9638, optimizer=opt,
                    activations=full,
                )
                cells.append(f"{pos:,}" if pos else "none")
            print(f"| {name} | {label.split(' (')[0]} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
