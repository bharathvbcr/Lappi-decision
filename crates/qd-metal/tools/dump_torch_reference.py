#!/usr/bin/env python3
"""Torch reference for qd-metal's parity gate: real prompts, real 2B weights, CPU fp32.

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python \
        crates/qd-metal/tools/dump_torch_reference.py --out <scratch dir> [--n 24]

**CPU only.** The reference runs on ``device="cpu"`` so it never touches the GPU another lane may
be timing on.

Prompts are real Lappi prompts: ``code.defect_class`` requests built from the first rows of
``data/pool/commitpackft-mutated/examples.jsonl`` (real commits, real diffs) through
``qd_data.mixture._request`` and rendered by ``qd_data.render.render`` -- the single renderer the
training and serving paths share. The model is ``qd_train.backbone.load_text_tower``, the
canonical owner of the text tower's construction.

Two references per prompt, both fp32 activations on the real bf16 weights:

* ``fp32``: the model as transformers runs it in fp32. This is the reference the gate names.
* ``bf16in``: the same, with every ``nn.Linear`` input rounded to bf16 first. That is tessl's
  GEMM numerics (bf16 operands, fp32 accumulate), so a difference from it is wiring, not
  rounding. It is the tight comparison; ``fp32`` is the loose one.

The rounding hooks exist only inside :func:`bf16_linear_inputs`'s ``with`` block, and every
``fp32`` pass first asserts that no ``nn.Linear`` carries a forward-pre-hook. (A first version
built both passes' hooks eagerly, so its "fp32" pass was rounded too; that dump was discarded.)

Per prompt this writes the residual stream after every decoder layer (forward hooks on
``model.layers[i]``, pre final norm) at up to ``--rows`` evenly spaced positions including the
last, and the 17-row letter logits / log-softmax at the last position (final norm, then
``embed_tokens.weight[letter_ids]``, since the head is tied).

It also checks, for every prompt and every letter, the tokenization contract qd-metal relies on:
``ids(prompt) + [id(L)] == ids(prompt + L)`` and ``ids(prefix) + ids(suffix) == ids(prompt)``.
A failure aborts the dump.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from qd_data.defect_class import DEFECT_CLASSES
from qd_data.mixture import DEFECT_FAMILY_ID, _request
from qd_data.render import NOUL_LETTER, OPTION_LETTERS, render
from qd_data.schema import ChoiceSlot
from qd_train.backbone import load_text_tower
from qd_train.memory import ADAMW_FP32

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "data/pool/commitpackft-mutated/examples.jsonl"
SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots"
LETTERS = [*OPTION_LETTERS, NOUL_LETTER]


def snapshot_dir() -> Path:
    found = [p for p in SNAPSHOTS.glob("*") if (p / "config.json").is_file()]
    if len(found) != 1:
        sys.exit(f"expected exactly one Qwen3.5-2B-Base snapshot, found {found}")
    return found[0]


def encoder(tok: Tokenizer) -> Callable[[str], list[int]]:
    """Token ids without special tokens, as qd-metal encodes."""

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False).ids

    return enc


def prompts(n: int, tok: Tokenizer, min_tokens: int, max_tokens: int) -> list[dict]:
    out: list[dict] = []
    enc = encoder(tok)
    with EXAMPLES.open() as f:
        for line in f:
            row = json.loads(line)
            diff = row["diff"]
            body = diff[:-1] if diff.endswith("\n") else diff
            ctx = f"file: {row['function']['path']}\n\n{body}"
            req = _request(
                family_id=DEFECT_FAMILY_ID,
                context=ctx,
                slots=(ChoiceSlot(name="defect_class", options=DEFECT_CLASSES),),
                example_id=row["id"],
            )
            r = render(req, seed=None)
            prefix, suffix = r.prefix, r.slots[0].suffix
            ids = enc(prefix + suffix)
            if not (min_tokens <= len(ids) <= max_tokens):
                continue
            if enc(prefix) + enc(suffix) != ids:
                sys.exit(f"{row['id']}: prefix/suffix do not tokenize to the prompt's tokens")
            out.append(
                {
                    "id": row["id"],
                    "class": row["class"],
                    "prefix": prefix,
                    "suffix": suffix,
                    "ids": ids,
                    "prefix_tokens": len(enc(prefix)),
                }
            )
            if len(out) == n:
                break
    if len(out) < n:
        sys.exit(f"only {len(out)} prompts in [{min_tokens}, {max_tokens}] tokens")
    return out


def letter_ids(tok: Tokenizer, sample_prompts: list[dict]) -> list[int]:
    enc = encoder(tok)
    ids = []
    for letter in LETTERS:
        one = enc(letter)
        if len(one) != 1:
            sys.exit(f"letter {letter!r} is {one}, not one token")
        ids.append(one[0])
    for p in sample_prompts:
        text = p["prefix"] + p["suffix"]
        for letter, lid in zip(LETTERS, ids, strict=True):
            if enc(text + letter) != [*p["ids"], lid]:
                sys.exit(f"{p['id']}: the letter {letter!r} merges into the prompt's last token")
    return ids


def linears(model: torch.nn.Module) -> list[torch.nn.Linear]:
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


@contextmanager
def bf16_linear_inputs(model: torch.nn.Module) -> Iterator[None]:
    """Round every nn.Linear input to bf16 inside the block: tessl's GEMM operand precision."""

    def pre(_mod, args):
        (x, *rest) = args
        return (x.to(torch.bfloat16).to(torch.float32), *rest)

    handles = [m.register_forward_pre_hook(pre) for m in linears(model)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run(
    model, ids: list[int], rows: list[int], letters: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    captured: list[np.ndarray] = []

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h[0, rows].to(torch.float32).numpy().copy())

    handles = [layer.register_forward_hook(hook) for layer in model.layers]
    try:
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if len(captured) != len(model.layers):
        sys.exit(f"captured {len(captured)} layer outputs for {len(model.layers)} layers")
    last = out.last_hidden_state[0, -1].to(torch.float32)  # after the final norm
    head = model.embed_tokens.weight[letters].to(torch.float32)
    logits = head @ last
    logp = torch.log_softmax(logits.double(), dim=-1).to(torch.float32)
    return np.stack(captured), logits.numpy(), logp.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--rows", type=int, default=128, help="hidden rows kept per prompt")
    ap.add_argument("--min-tokens", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=1536)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    snap = snapshot_dir()
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    ps = prompts(args.n, tok, args.min_tokens, args.max_tokens)
    letters = letter_ids(tok, ps)

    torch.set_grad_enabled(False)
    tower = load_text_tower(
        snap,
        gradient_checkpointing=False,
        # Inference only: the spec feeds the load's memory-footprint arithmetic, nothing else.
        optimizer=ADAMW_FP32,
        attn_implementation="eager",
        device="cpu",
        dtype="fp32",
    )
    model = getattr(tower, "model", tower)
    model.eval()

    manifest = {
        "snapshot": str(snap),
        "torch": torch.__version__,
        "device": "cpu",
        "letters": LETTERS,
        "letter_ids": letters,
        "prompts": [],
    }
    for i, p in enumerate(ps):
        t = len(p["ids"])
        spaced = np.linspace(0, t - 1, min(args.rows, t)).round().astype(int).tolist()
        keep = sorted(set(spaced) | {t - 1})
        entry = dict(p, index=i, tokens=t, rows=keep)

        if any(m._forward_pre_hooks for m in linears(model)):
            sys.exit("an nn.Linear carries a forward-pre-hook before the fp32 pass")
        results = {"fp32": run(model, p["ids"], keep, letters)}
        with bf16_linear_inputs(model):
            results["bf16in"] = run(model, p["ids"], keep, letters)
        if np.array_equal(results["fp32"][1], results["bf16in"][1]):
            sys.exit(f"prompt {i}: fp32 and bf16in logits are identical; the rounding did not run")

        for name, (hidden, logits, logp) in results.items():
            np.save(args.out / f"hidden_{i:02d}_{name}.npy", hidden.astype(np.float32))
            entry[f"logits_{name}"] = logits.tolist()
            entry[f"logprobs_{name}"] = logp.tolist()
        manifest["prompts"].append(entry)
        top = int(np.argmax(entry["logits_fp32"]))
        gap = float(np.max(np.abs(results["fp32"][2] - results["bf16in"][2])))
        print(f"prompt {i}: {t} tokens, argmax fp32 {top}, |fp32 - bf16in| logp {gap:.4f}")
    (args.out / "manifest.json").write_text(json.dumps(manifest))
    print(f"wrote {len(ps)} prompts to {args.out}")


if __name__ == "__main__":
    main()
