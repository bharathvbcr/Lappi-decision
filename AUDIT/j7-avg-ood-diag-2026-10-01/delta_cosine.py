"""Throwaway analysis (Fable I-2 diagnostic 2; never ships): pairwise cosine of J4's three seeds'
fine-tuning deltas (fp32 master - base), overall, per module kind, and on the letter rows of the
tied embedding -- the noul letter Z (id 57) against the option letters A..P (ids 32..47).

Dilution predicts: the noul row's deltas near-orthogonal across seeds while the overall deltas
and the option letters' rows are not.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(16)
AVG = "/home/ubuntu/ckpt/p4-v3/avg/epoch-avg-seed012-masters.safetensors"
SEEDS = [
    "/home/ubuntu/ckpt/p4-v3/epoch-seed0-cuda.18bead03d1ac2793.safetensors",
    "/home/ubuntu/ckpt/p4-v3/epoch-seed1-cuda.666dd11d8492541b.safetensors",
    "/home/ubuntu/ckpt/p4-v3/epoch-seed2-cuda.da084ff2a63b5e7a.safetensors",
]
BASE = (
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c/model.safetensors-00001-of-00001.safetensors"
)
NOUL_ID = 57
OPTION_IDS = list(range(32, 48))
PAIRS = [(0, 1), (0, 2), (1, 2)]

with Path(AVG + ".manifest.json").open() as manifest:
    master_index = json.load(manifest)["master_index"]
base = safe_open(BASE, "pt")
seeds = [safe_open(p, "pt") for p in SEEDS]


def kind(name: str) -> str:
    if "embed_tokens" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".linear_attn." in name:
        return "linear_attn"
    if ".self_attn." in name:
        return "self_attn"
    if "norm" in name:
        return "norm"
    return "other"


acc: dict[str, dict[str, float]] = {}


def add(group: str, deltas: list[torch.Tensor]) -> None:
    g = acc.setdefault(group, {})
    for i in range(3):
        g[f"n{i}"] = g.get(f"n{i}", 0.0) + float((deltas[i] * deltas[i]).sum())
    for i, j in PAIRS:
        g[f"d{i}{j}"] = g.get(f"d{i}{j}", 0.0) + float((deltas[i] * deltas[j]).sum())


letter_rows: dict[int, list[torch.Tensor]] = {}
for name, idx in sorted(master_index.items()):
    if not name.startswith("tower."):
        continue
    key = name[len("tower."):]
    b = base.get_tensor("model.language_model." + key).to(torch.float64)
    deltas = [
        s.get_tensor(f"model_state['optimizer']['masters'][{idx}]").to(torch.float64) - b
        for s in seeds
    ]
    add("overall", deltas)
    add(kind(key), deltas)
    if "embed_tokens" in key:
        for row in [NOUL_ID, *OPTION_IDS]:
            letter_rows[row] = [d[row].clone() for d in deltas]
        rest = torch.ones(deltas[0].shape[0], dtype=torch.bool)
        rest[[NOUL_ID, *OPTION_IDS]] = False
        add("embed_without_letters", [d[rest] for d in deltas])
    del deltas, b

for row in [NOUL_ID, *OPTION_IDS]:
    add(f"row{row}", letter_rows[row])


def cos(g: dict[str, float]) -> dict[str, float]:
    return {f"{i}{j}": g[f"d{i}{j}"] / (g[f"n{i}"] ** 0.5 * g[f"n{j}"] ** 0.5) for i, j in PAIRS}


out = {}
for group, g in acc.items():
    out[group] = {
        "cos": cos(g),
        "norm": [g[f"n{i}"] ** 0.5 for i in range(3)],
    }
opt = [out[f"row{r}"]["cos"] for r in OPTION_IDS]
summary = {
    "noul_row_Z57": out["row57"],
    "option_rows_A_P_mean_cos": {p: sum(c[p] for c in opt) / len(opt) for p in ["01", "02", "12"]},
    "option_rows_A_P_mean_norm": [
        sum(out[f"row{r}"]["norm"][i] for r in OPTION_IDS) / len(OPTION_IDS) for i in range(3)
    ],
}
GROUPS = [
    "overall", "embed", "embed_without_letters", "mlp", "linear_attn", "self_attn", "norm", "other"
]
for group in GROUPS:
    if group in out:
        summary[group] = out[group]
json.dump({"summary": summary, "rows": {k: v for k, v in out.items() if re.match(r"row\d+", k)}},
          sys.stdout, indent=1)
print()
