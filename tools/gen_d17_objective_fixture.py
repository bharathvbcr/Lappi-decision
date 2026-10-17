#!/usr/bin/env python3
"""Parity fixture for qd-train's D17 host objective (crates/qd-train/src/d17.rs), generated with
the Python the readout is trained with (ft-2ed2fd55 criterion "D17 as a host entry in qd-train's
objective").

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/gen_d17_objective_fixture.py \
        --out crates/qd-train/tests/fixtures/d17-objective

Then pin it: `shasum -a 256 cases.json > SHA256SUMS` in that directory (the fixture set's
manifest, checked by crates/qd-train/tests/d17_objective.rs before the bytes are read).

This is the **reference oracle** (policy: Python only as a fixture generator that Rust tests
compare against). It runs qd_train.lora.AnswerRowDelta.loss (the rule named in the file) on the
CPU in float32, with autograd for the gradients, and the same computation in float64 as the
exact reference. The loss is a mean over the N supervised positions, so Rust's per-position
scale is 1/N. The generator is a sibling of gen_span_fixture.py because that one writes into
qd-metal's pinned span/ set; this one must not change a byte of it.

Encoding: embed as bf16 bit patterns (the checkpoint's dtype); every other f32 array as
binary32 bit patterns (u32); float64 values as binary64 bit patterns (u64). ids are the
vocabulary ids of the 17 answer rows in D17 row order (deliberately not contiguous and not
sorted, so a reader that took rows by position would fail); targets are vocabulary ids.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from qd_train.lora import D17_ROWS, AnswerRowDelta


def bits32(t: torch.Tensor) -> list[int]:
    return t.detach().to(torch.float32).contiguous().view(torch.int32).flatten().tolist()


def bits16(t: torch.Tensor) -> list[int]:
    raw = t.detach().to(torch.bfloat16).contiguous().view(torch.int16).flatten().tolist()
    return [b & 0xFFFF for b in raw]


def bits64(values: list[float]) -> list[int]:
    return [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]


def case(name: str, seed: int, hidden: int, vocab: int, targets_k: list[int]) -> dict:
    gen = torch.Generator().manual_seed(seed)
    n = len(targets_k)
    embed = (torch.randn(vocab, hidden, generator=gen) * 0.7).to(torch.bfloat16)
    ids = torch.randperm(vocab, generator=gen)[:D17_ROWS].tolist()
    module = AnswerRowDelta(hidden)
    with torch.no_grad():
        module.delta.copy_(torch.randn(D17_ROWS, hidden, generator=gen) * 0.2)
    h = torch.randn(n, hidden, generator=gen)
    targets = torch.tensor([ids[k] for k in targets_k])
    row_ids = torch.tensor(ids)

    h32 = h.clone().requires_grad_(True)
    loss = module.loss(h32, embed, row_ids, targets)
    loss.backward()
    logits = module.logits(h32, embed, row_ids).detach()

    # float64: the same rule, exact to ~1e-16, as the gate's reference.
    h64 = h.double().clone().requires_grad_(True)
    d64 = module.delta.detach().double().clone().requires_grad_(True)
    rows64 = embed.detach()[row_ids].double() + d64
    lg64 = h64 @ rows64.t()
    loss64 = F.cross_entropy(lg64, torch.tensor(targets_k))
    loss64.backward()

    return {
        "name": name, "hidden": hidden, "vocab": vocab, "n": n,
        "ids": ids, "targets": targets.tolist(), "target_rows": targets_k,
        "embed": bits16(embed), "delta": bits32(module.delta), "h": bits32(h),
        "logits": bits32(logits), "loss": bits32(loss.detach().reshape(1)),
        "dh": bits32(h32.grad), "d_delta": bits32(module.delta.grad),
        "ref64_logits": bits64(lg64.detach().flatten().tolist()),
        "ref64_loss": bits64([loss64.item()]),
        "ref64_dh": bits64(h64.grad.flatten().tolist()),
        "ref64_d_delta": bits64(d64.grad.flatten().tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [
        case("n1", 20261017, 16, 64, [3]),
        case("all17", 20261018, 24, 80, list(range(17))),
        case("repeated", 20261019, 32, 70, [4, 4, 4, 9, 16, 9, 0, 4]),
        case("noul_only", 20261020, 16, 40, [16, 16]),
        case("random_h64", 20261021, 64, 96, [1, 5, 2, 8, 13, 0, 11, 16, 7, 7, 3, 12]),
    ]
    doc = {
        "rule": "qd_train.lora.AnswerRowDelta.loss: mean 17-way cross entropy over "
                "embed.detach()[ids].float() + delta; gradients by autograd (mean => scale 1/N)",
        "command": "PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python "
                   "tools/gen_d17_objective_fixture.py "
                   "--out crates/qd-train/tests/fixtures/d17-objective",
        "torch": torch.__version__,
        "encoding": "embed bf16 bits; f32 arrays u32 bits; ref64_* u64 bits of binary64",
        "cases": cases,
    }
    (args.out / "cases.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
