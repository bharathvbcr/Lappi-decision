"""Oracle for the Rust LoRA parameter plan and the LoRA-A init rule (reference only).

Builds a tiny Qwen3.5-shaped module tree (a GDN layer, a full-attention layer, MLPs; the GDN
``in_proj_a``/``in_proj_b`` present and not adapted), runs ``qd_train.lora.attach_lora`` at
each seed and ``qd_train.lora.lora_param_groups`` over it, and dumps what the Rust side must
reproduce without torch:

* the optimizer groups: name, ``lr_scale``, ``weight_decay`` and the member parameter names;
* the adapted modules in the order ``attach_lora`` drew their ``A`` (canonical: layer index, then
  ``LORA_TARGET_SUFFIXES``, not ``model.named_modules()``), each with ``in_features``, the
  float.hex of the first values of its ``A`` and the A tensor's ``TensorRef`` digest;
* ``hf_registration_order``: the same tree registered in HF's GDN order (``out_proj`` first), with
  the listing order and the digest, which must equal the ojas-order digest;
* the LoRA-A digest: ``run_control._sidecar_digest`` over the ``TensorRef`` of every
  ``<module>.lora_A.weight`` (float32, little-endian bytes), the same construction as the span
  head's content digest.

Regenerate (the command is also recorded in the fixture):

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_lora_plan.py \\
        --out crates/qd-train/tests/fixtures/lora-plan-oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

from qd_train import lora
from qd_train.heads import SpanPointerHead
from qd_train.run_control import TensorRef, _sidecar_digest

HIDDEN = 12
INTER = 20
GDN_QKV = 18
SEEDS = (0, 7, 1234)
HEAD_HIDDEN = 6


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HIDDEN, bias=False)


class SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, 16, bias=False)
        self.k_proj = nn.Linear(HIDDEN, 8, bias=False)
        self.v_proj = nn.Linear(HIDDEN, 8, bias=False)
        self.o_proj = nn.Linear(16, HIDDEN, bias=False)


class LinearAttn(nn.Module):
    def __init__(self, hf_order: bool = False) -> None:
        super().__init__()
        # hf_order: transformers' Qwen3_5GatedDeltaNet registers out_proj before in_proj_qkv and
        # in_proj_z (modeling_qwen3_5.py, transformers 5.19.0); the default is the ojas order.
        if hf_order:
            self.out_proj = nn.Linear(8, HIDDEN, bias=False)
        self.in_proj_qkv = nn.Linear(HIDDEN, GDN_QKV, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN, 8, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN, 2, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN, 2, bias=False)
        if not hf_order:
            self.out_proj = nn.Linear(8, HIDDEN, bias=False)


class Layer(nn.Module):
    def __init__(self, gdn: bool, hf_order: bool = False) -> None:
        super().__init__()
        if gdn:
            self.linear_attn = LinearAttn(hf_order)
        else:
            self.self_attn = SelfAttn()
        self.mlp = Mlp()
        self.input_layernorm = nn.LayerNorm(HIDDEN)


class Tower(nn.Module):
    def __init__(self, hf_order: bool = False) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, HIDDEN)
        self.layers = nn.ModuleList(
            [Layer(True, hf_order), Layer(False, hf_order), Layer(True, hf_order)]
        )
        self.norm = nn.LayerNorm(HIDDEN)


class Model(nn.Module):
    def __init__(self, hf_order: bool = False) -> None:
        super().__init__()
        self.model = Tower(hf_order)


def a_ref(t: torch.Tensor) -> TensorRef:
    host = t.detach().to("cpu").contiguous()
    return TensorRef(
        dtype="float32",
        shape=tuple(host.shape),
        data=host.reshape(-1).view(torch.uint8).numpy().tobytes(),
    )


def hexes(t: torch.Tensor, n: int) -> list[str]:
    return [float(v).hex() for v in t.reshape(-1)[:n].tolist()]


def seed_block(seed: int) -> dict[str, object]:
    model = Model()
    adapter = lora.attach_lora(model, seed=seed)
    in_features = {
        name: module.base.in_features
        for name, module in model.named_modules()
        if isinstance(module, lora.LoRALinear)
    }
    refs: dict[str, TensorRef] = {}
    modules = []
    for name in adapter.module_names:
        a = model.get_submodule(f"{name}.lora_A").weight
        ref = a_ref(a)
        refs[f"{name}.lora_A.weight"] = ref
        modules.append(
            {
                "module": name,
                "in_features": in_features[name],
                "shape": list(a.shape),
                "first_values": hexes(a, 6),
                "last_values": hexes(a.reshape(-1)[-6:], 6),
                "tensor_digest": ref.digest(),
            }
        )
    delta = lora.AnswerRowDelta(HEAD_HIDDEN)
    head = SpanPointerHead(HEAD_HIDDEN)
    groups = lora.lora_param_groups(model, delta=delta, span_head=head)
    named: dict[int, str] = {id(p): n for n, p in lora.lora_parameters(model)}
    named[id(delta.delta)] = "d17.delta"
    for n, p in head.named_parameters():
        named[id(p)] = n
    out_groups = [
        {
            "name": g["name"],
            "lr_scale": float(g["lr_scale"]).hex(),
            "weight_decay": float(g["weight_decay"]).hex(),
            "params": [named[id(p)] for p in g["params"]],
        }
        for g in groups
    ]
    parameter_table = [
        {"name": n, "shape": list(p.shape)} for n, p in lora.lora_parameters(model)
    ]
    return {
        "seed": seed,
        "a_digest": _sidecar_digest(refs),
        "modules": modules,
        "groups": out_groups,
        "adapter_parameters": parameter_table,
        "d17": {"name": "d17.delta", "shape": list(delta.delta.shape)},
        "span_head": [{"name": n, "shape": list(p.shape)} for n, p in head.named_parameters()],
    }


def hf_registration_block(seed: int) -> dict[str, object]:
    """The same architecture registered in HF's real GDN order: the digest and the registration
    order its adapted linears were listed in (the order a provider may hand the Rust verifier)."""
    model = Model(hf_order=True)
    registered = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and any(n.endswith(s) for s in lora.LORA_TARGET_SUFFIXES)
    ]
    adapter = lora.attach_lora(model, seed=seed)
    refs = {
        f"{name}.lora_A.weight": a_ref(model.get_submodule(f"{name}.lora_A").weight)
        for name in adapter.module_names
    }
    widths = dict(lora.adapted_in_features(model))
    return {
        "seed": seed,
        "registration_order": [{"module": n, "in_features": widths[n]} for n in registered],
        "draw_order": list(adapter.module_names),
        "a_digest": _sidecar_digest(refs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    blocks = [seed_block(s) for s in SEEDS]
    base_names = sorted(
        n
        for n, _ in Model().named_parameters()
        if not n.endswith(("lora_A.weight", "lora_B.weight"))
    )
    doc = {
        "what": "qd_train.lora.lora_param_groups and attach_lora over a tiny Qwen3.5-shaped tree",
        "command": "PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python "
        "tools/qd_train_oracle_lora_plan.py "
        "--out crates/qd-train/tests/fixtures/lora-plan-oracle.json",
        "generated_with": {"python": sys.version.split()[0], "torch": torch.__version__},
        "constants": {
            "LORA_LR": float(lora.LORA_LR).hex(),
            "D17_LR": float(lora.D17_LR).hex(),
            "HEAD_LR_SCALE": float(lora.HEAD_LR_SCALE).hex(),
            "LORA_WEIGHT_DECAY": float(lora.LORA_WEIGHT_DECAY).hex(),
            "LORA_RANK": lora.LORA_RANK,
            "LORA_TARGET_SUFFIXES": list(lora.LORA_TARGET_SUFFIXES),
            "LORA_UNADAPTED_LINEAR_SUFFIXES": list(lora.LORA_UNADAPTED_LINEAR_SUFFIXES),
        },
        "unadapted_base_names": base_names,
        "seeds": blocks,
        "hf_registration_order": [hf_registration_block(s) for s in SEEDS],
    }
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: seeds {list(SEEDS)}, {len(blocks[0]['modules'])} adapted modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
