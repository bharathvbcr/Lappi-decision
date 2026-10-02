"""Fixture generator (reference oracle, not shipped): F's per-parameter AdamW table on the 2B.

Amendment 2 (ii) test 1 (``AUDIT/ojas-training-2026-10-01/fable-rung-b-bars.md``): the Rust
trainer's per-entry ``(lr_scale, weight_decay, eps, betas)`` table must equal what Lappi's
``layerwise_param_groups`` plus F's builder produce, name by name, on the real 2B parameter
names, with no tolerance. This script asks the Python code itself, rather than restating the
rule:

* the tower is ``Qwen3_5TextModel(text_config)`` built on the meta device from the snapshot's
  own ``config.json``, in bf16 (F's ``--optimizer master`` tower), exactly as
  ``backbone.load_text_tower`` builds its skeleton; its names are checked against the
  snapshot's ``model.language_model.*`` header (``backbone.text_tensor_index``), so they are
  the checkpoint's names and not only the class's;
* the span head is ``heads.SpanPointerHead(hidden)`` in f32, as ``QwenDecisionStep`` builds it;
* the optimizer is ``optim.build_optimizer(optim.layerwise_param_groups(tower,
  lower_layers_n=8, lower_lr_scale=0.1, extra=head), spec=real_ft_run.optimizer_spec("bf16",
  "master"), lr=1e-5, total_steps=9683)`` -- ``QwenDecisionStep.__init__``'s call with F's flags
  (``campaign/f-v4-preregistered.json``: ``--optimizer master --lr 1e-5 --lower-layers-n 8
  --lower-layers-lr-scale 0.1``; 9,683 is F's epoch length and moves nothing in the table);
* every parameter's row is read off the built optimizer's own param groups (through
  ``MasterWeightAdamW``'s master-to-live map), so a default the builder adds is in the table.

Floats are dumped as ``float.hex``. Run::

    PYTHONPATH=python:tools /Users/bharath/.venvs/ml/bin/python \\
        tools/qd_train_oracle_optimizer_table.py \\
        --snapshot ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \\
        --out crates/qd-train/tests/fixtures/optimizer-table-2b.json \\
        --config-out crates/qd-train/tests/fixtures/qwen35-2b-config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from qd_train.backbone import text_tensor_index
from qd_train.heads import SpanPointerHead
from qd_train.optim import DEFAULT_BETA2, LR_SCALE_KEY, build_optimizer, layerwise_param_groups
from real_ft_run import optimizer_spec

#: F's recipe (campaign/f-v4-preregistered.json), the flags this table depends on.
F_LR = 1e-5
F_LOWER_LAYERS_N = 8
F_LOWER_LR_SCALE = 0.1
F_EPOCH_STEPS = 9683


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"unknown({e})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--config-out", type=Path, required=True,
        help="where to copy the snapshot's config.json, byte for byte, so a Rust test can build "
        "ojas-qwen35's name map for the same tower without the snapshot",
    )
    args = ap.parse_args()
    config_bytes = (args.snapshot / "config.json").read_bytes()
    args.config_out.write_bytes(config_bytes)

    full = AutoConfig.from_pretrained(args.snapshot)
    text_config = full.text_config
    with torch.device("meta"):
        tower = Qwen3_5TextModel(text_config).to(dtype=torch.bfloat16)
        head = SpanPointerHead(text_config.hidden_size).to(dtype=torch.float32)

    tower_names = [n for n, _ in tower.named_parameters()]
    header = text_tensor_index(args.snapshot)
    if set(tower_names) != set(header):
        raise SystemExit(
            f"the model's parameter names and the snapshot's model.language_model.* header differ: "
            f"{sorted(set(tower_names) ^ set(header))[:8]}"
        )
    shapes = {n: list(p.shape) for n, p in tower.named_parameters()}
    for n, (_shard, shape) in header.items():
        if tuple(shapes[n]) != tuple(shape):
            raise SystemExit(f"{n}: model shape {shapes[n]} but header shape {list(shape)}")

    groups = layerwise_param_groups(
        tower.named_parameters(),
        lower_layers_n=F_LOWER_LAYERS_N,
        lower_lr_scale=F_LOWER_LR_SCALE,
        extra=head.parameters(),
    )
    opt = build_optimizer(
        groups,
        spec=optimizer_spec("bf16", "master"),
        lr=F_LR,
        total_steps=F_EPOCH_STEPS,
        beta2=DEFAULT_BETA2,
    )

    name_of = {id(p): n for n, p in tower.named_parameters()}
    name_of.update({id(p): f"span_head.{n}" for n, p in head.named_parameters()})
    live_of_master = {id(m): live for live, m in zip(opt._live, opt._masters, strict=True)}
    table: dict[str, dict[str, object]] = {}
    for g in opt.param_groups:
        for m in g["params"]:
            name = name_of[id(live_of_master[id(m)])]
            if name in table:
                raise SystemExit(f"{name} is in two param groups")
            beta1, beta2 = g["betas"]
            table[name] = {
                "group": g.get("name"),
                "lr_scale": float(g.get(LR_SCALE_KEY, 1.0)).hex(),
                "weight_decay": float(g["weight_decay"]).hex(),
                "eps": float(g["eps"]).hex(),
                "beta1": float(beta1).hex(),
                "beta2": float(beta2).hex(),
                "lr": float(g["lr"]).hex(),
            }
    head_names = [f"span_head.{n}" for n, _ in head.named_parameters()]
    every = tower_names + head_names
    if sorted(table) != sorted(every):
        raise SystemExit(f"the optimizer covers {len(table)} parameters, the model has {len(every)}")

    script = Path(__file__).read_bytes()
    out = {
        "what": (
            "F's per-parameter AdamW table on the Qwen3.5-2B names: layerwise_param_groups + "
            "build_optimizer as QwenDecisionStep calls them, read off the built optimizer"
        ),
        "generator": {
            "script": "tools/qd_train_oracle_optimizer_table.py",
            "script_sha256": hashlib.sha256(script).hexdigest(),
            "argv": sys.argv[1:],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "git_head": _git_head(),
        },
        "snapshot": args.snapshot.name,
        "config": {"file": args.config_out.name, "sha256": hashlib.sha256(config_bytes).hexdigest()},
        "recipe": {
            "lr": F_LR.hex(),
            "lower_layers_n": F_LOWER_LAYERS_N,
            "lower_lr_scale": F_LOWER_LR_SCALE.hex(),
            "total_steps": F_EPOCH_STEPS,
            "optimizer_recipe": "master",
            "beta2": DEFAULT_BETA2.hex(),
        },
        "names_checked_against": "backbone.text_tensor_index(snapshot): the model.language_model.* header",
        "tower_names": tower_names,
        "tower_shapes": [shapes[n] for n in tower_names],
        "head_names": head_names,
        "hidden_size": text_config.hidden_size,
        "table": table,
    }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    groups_seen = sorted({str(v["group"]) for v in table.values()})
    print(f"{len(table)} parameters ({len(tower_names)} tower + {len(head_names)} head), groups {groups_seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
