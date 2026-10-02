"""Throwaway analysis (Fable I-2 diagnostic 2; never ships): pairwise cosine of N seeds'
fine-tuning deltas (fp32 master - base), overall, per module kind, and on the letter rows of the
tied embedding -- the noul letter Z (id 57) against the option letters A..P (ids 32..47).

Dilution predicts: the noul row's deltas near-orthogonal across seeds while the overall deltas
and the option letters' rows are not.

The masters are read by the name -> index map of an average of exactly these seeds
(``ckpt_average.py --from masters``'s manifest), which ``master_names`` proved one map for every
input. Each sidecar must be its manifest input's, in the manifest's order.

J4's run (delta-cosine.json, three seeds)::

    C=/home/ubuntu/ckpt/p4-v3
    python delta_cosine.py \\
      --avg-manifest $C/avg/epoch-avg-seed012-masters.safetensors.manifest.json \\
      --base <snapshot>/model.safetensors-00001-of-00001.safetensors \\
      $C/epoch-seed0-cuda.18bead03d1ac2793.safetensors \\
      $C/epoch-seed1-cuda.666dd11d8492541b.safetensors \\
      $C/epoch-seed2-cuda.da084ff2a63b5e7a.safetensors > delta-cosine.json
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

NOUL_ID = 57
OPTION_IDS = list(range(32, 48))

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("sidecars", nargs="+", type=Path, help="each seed's checkpoint sidecar")
parser.add_argument("--avg-manifest", type=Path, required=True)
parser.add_argument("--base", type=Path, required=True, help="the base snapshot's safetensors")
parser.add_argument("--threads", type=int, default=16)
args = parser.parse_args()
torch.set_num_threads(args.threads)

with args.avg_manifest.open() as manifest_file:
    manifest = json.load(manifest_file)
master_index = manifest["master_index"]
inputs = [Path(record["path"]) for record in manifest["inputs"]]
if len(inputs) != len(args.sidecars) or len(inputs) < 2:
    raise SystemExit(f"{len(args.sidecars)} sidecars for an average of {len(inputs)} inputs")
for record, sidecar in zip(inputs, args.sidecars, strict=True):
    if not sidecar.name.startswith(record.stem + ".") or sidecar.suffix != ".safetensors":
        raise SystemExit(f"{sidecar.name} is not {record.name}'s sidecar (manifest order)")
N = len(inputs)
PAIRS = list(itertools.combinations(range(N), 2))
base = safe_open(str(args.base), "pt")
seeds = [safe_open(str(p), "pt") for p in args.sidecars]


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
    for i in range(N):
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
        "norm": [g[f"n{i}"] ** 0.5 for i in range(N)],
    }
opt = [out[f"row{r}"]["cos"] for r in OPTION_IDS]
summary = {
    "noul_row_Z57": out["row57"],
    "option_rows_A_P_mean_cos": {
        f"{i}{j}": sum(c[f"{i}{j}"] for c in opt) / len(opt) for i, j in PAIRS
    },
    "option_rows_A_P_mean_norm": [
        sum(out[f"row{r}"]["norm"][i] for r in OPTION_IDS) / len(OPTION_IDS) for i in range(N)
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
