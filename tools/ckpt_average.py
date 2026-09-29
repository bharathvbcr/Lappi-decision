"""Average the weights of N checkpoints written by ``real_ft_run.py --checkpoint-dir``.

Ported from RSI-Jev ``rsijev/train.py:189-200`` (``average_checkpoints``), MIT licence,
Copyright (c) 2026 Shanghua Gao, at commit 8f34a4f: sum the float copies, divide by N.
Their note, which is the reason to have it: averaging removes checkpoint-selection noise at
no training cost.

What this adds to theirs, all of it refusal:

* **Same schedule, same step, same model.** ``real_ft_run.py`` rewrites one file per
  ``(tag, seed, device)`` cell, so N checkpoints here are N *runs* (normally N seeds of one
  configuration), not N points along one run. Averaging only means something when they are
  the same configuration at the same point: the ``LRSchedule`` must be equal, the
  ``optimizer_step`` equal, ``vocab_size`` and ``span_weight`` equal, and the ``tower`` and
  ``span_head`` trees must match name for name, dtype for dtype and shape for shape. RSI's
  version indexes ``st[k]`` and would raise a bare ``KeyError`` on the first two and
  silently broadcast on none of them; this says which input differs and how.
* **No input twice.** The same file, or two files whose tensor sets have one digest, would
  weight one run double while the manifest listed N.
* **Weights only, and not a checkpoint.** The optimizer subtree is not averaged and the
  output is not written as a ``Checkpoint``: Adam moments averaged across runs describe no
  trajectory, and this repository's position (``run_control``: "a resume from weights alone
  restarts Adam's moments from zero and does not reproduce the trajectory") is that
  weights without moments are not a resume point. The output is a safetensors file of the
  averaged ``tower.*`` and ``span_head.*`` tensors plus a JSON manifest naming every input,
  its ``payload_digest``, and ``"resumable": false``.

Averaged in float32 and cast back to each tensor's own dtype, so a bf16 tower comes back
bf16 -- the dtype ``load_text_tower`` was budgeted for.

What this cannot check: which shard set each input trained on. A ``Checkpoint`` carries the
batch order's ``consumed_digest`` but not the shard hash; the ledger rows of the runs do.
Pass ``--ft-row-ids`` to record them in the manifest so the average names its sources.

Usage::

    python tools/ckpt_average.py --out /path/avg.safetensors CKPT.json CKPT.json [...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_train.run_control import Checkpoint, TensorRef

#: The model subtrees averaged. Everything else in ``model_state`` is either compared for
#: equality (the scalars below) or deliberately dropped (``optimizer``).
AVERAGED_TREES: Final[tuple[str, ...]] = ("tower", "span_head")
#: Scalars that must agree across inputs, or the tensors mean different things.
MATCHED_SCALARS: Final[tuple[str, ...]] = ("vocab_size", "span_weight")
#: RSI's "last k" is typically 3-5; beyond a handful this is a sweep, not an average.
MAX_INPUTS: Final[int] = 64


class AverageRefusal(ValueError):
    """The inputs cannot be averaged into one meaningful model."""


def _tree(ckpt: Checkpoint, name: str, *, where: str) -> Mapping[str, TensorRef]:
    tree = ckpt.model_state.get(name)
    if not isinstance(tree, Mapping) or not tree:
        raise AverageRefusal(
            f"{where}: model_state has no {name!r} tensors. Only a checkpoint written by "
            "QwenDecisionStep.state carries a tower; the stand-in never writes one."
        )
    for key, ref in tree.items():
        if not isinstance(ref, TensorRef):
            raise AverageRefusal(f"{where}: {name}[{key!r}] is {type(ref).__name__}, not a tensor")
    return tree


def check_compatible(ckpts: Sequence[Checkpoint], names: Sequence[str]) -> None:
    """Refuse unless every input is the same model at the same point of the same schedule."""
    if len(ckpts) < 2:
        raise AverageRefusal(f"averaging needs at least 2 checkpoints, got {len(ckpts)}")
    if len(ckpts) > MAX_INPUTS:
        raise AverageRefusal(f"{len(ckpts)} inputs exceeds MAX_INPUTS={MAX_INPUTS}")
    first, first_name = ckpts[0], names[0]
    schedule = first.schedule.to_json()
    digests: dict[str, str] = {}
    for ckpt, name in zip(ckpts, names, strict=True):
        if ckpt.schedule.to_json() != schedule:
            raise AverageRefusal(
                f"{name} was trained on schedule {ckpt.schedule.to_json()} and {first_name} "
                f"on {schedule}. Different schedules are different runs, not seeds of one."
            )
        if ckpt.optimizer_step != first.optimizer_step:
            raise AverageRefusal(
                f"{name} is at optimizer step {ckpt.optimizer_step} and {first_name} at "
                f"{first.optimizer_step}; an average across two points of training is "
                "neither of them"
            )
        for scalar in MATCHED_SCALARS:
            if ckpt.model_state.get(scalar) != first.model_state.get(scalar):
                raise AverageRefusal(
                    f"{name} has {scalar}={ckpt.model_state.get(scalar)!r} and {first_name} "
                    f"{first.model_state.get(scalar)!r}"
                )
        for tree_name in AVERAGED_TREES:
            mine = _tree(ckpt, tree_name, where=name)
            ref = _tree(first, tree_name, where=first_name)
            if set(mine) != set(ref):
                extra = sorted(set(mine) - set(ref))[:5]
                missing = sorted(set(ref) - set(mine))[:5]
                raise AverageRefusal(
                    f"{name}'s {tree_name} is a different architecture from {first_name}'s: "
                    f"extra {extra}, missing {missing}"
                )
            for key in ref:
                if (mine[key].dtype, mine[key].shape) != (ref[key].dtype, ref[key].shape):
                    raise AverageRefusal(
                        f"{name} {tree_name}[{key!r}] is {mine[key].dtype}{list(mine[key].shape)} "
                        f"and {first_name}'s is {ref[key].dtype}{list(ref[key].shape)}"
                    )
        digest = hashlib.sha256(
            "".join(
                f"{t}.{k}:{_tree(ckpt, t, where=name)[k].digest()}"
                for t in AVERAGED_TREES
                for k in sorted(_tree(ckpt, t, where=name))
            ).encode()
        ).hexdigest()
        if digest in digests:
            raise AverageRefusal(
                f"{name} and {digests[digest]} hold identical weights; averaging them would "
                "weight one run twice while the manifest counted two"
            )
        digests[digest] = name


def average(ckpts: Sequence[Checkpoint], names: Sequence[str]) -> dict[str, Any]:
    """``{"tower.<k>": tensor, "span_head.<k>": tensor}``, averaged in fp32, cast back."""
    import torch

    check_compatible(ckpts, names)

    def load(ref: TensorRef) -> Any:
        return (
            torch.frombuffer(bytearray(ref.data), dtype=getattr(torch, ref.dtype))
            .reshape(ref.shape)
        )

    out: dict[str, Any] = {}
    for tree_name in AVERAGED_TREES:
        keys = sorted(_tree(ckpts[0], tree_name, where=names[0]))
        for key in keys:
            refs = [_tree(c, tree_name, where=n)[key] for c, n in zip(ckpts, names, strict=True)]
            dtype = getattr(torch, refs[0].dtype)
            if not dtype.is_floating_point:
                values = [load(r) for r in refs]
                if not all(torch.equal(values[0], v) for v in values[1:]):
                    raise AverageRefusal(
                        f"{tree_name}[{key!r}] is a non-float tensor ({refs[0].dtype}) that "
                        "differs between inputs; it has no mean"
                    )
                out[f"{tree_name}.{key}"] = values[0].clone()
                continue
            acc = torch.zeros(refs[0].shape, dtype=torch.float32)
            for r in refs:
                acc += load(r).to(torch.float32)
            out[f"{tree_name}.{key}"] = (acc / len(refs)).to(dtype)
    return out


def write(
    tensors: Mapping[str, Any],
    out: Path,
    *,
    inputs: Sequence[Path],
    ckpts: Sequence[Checkpoint],
    ft_row_ids: Sequence[str],
) -> Path:
    """The safetensors file and ``<out>.manifest.json``, each written atomically.

    Refuses to overwrite either: an average is an artifact other rows will name.
    """
    from safetensors.torch import save

    from qd_train.run_control import _atomic_write_bytes

    manifest_path = out.with_name(out.name + ".manifest.json")
    for target in (out, manifest_path):
        if target.exists():
            raise AverageRefusal(f"{target} already exists; refusing to overwrite it")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = save(dict(tensors))
    _atomic_write_bytes(out, payload)
    body = {
        "tool": "tools/ckpt_average.py",
        "method": "arithmetic mean in float32, cast back to each tensor's dtype",
        "source": "RSI-Jev rsijev/train.py:189-200 average_checkpoints (MIT) @8f34a4f",
        "resumable": False,
        "why_not_resumable": (
            "weights only: the optimizer moments were not averaged and are not here, and "
            "a resume from weights alone restarts Adam's moments from zero"
        ),
        "n_inputs": len(inputs),
        "optimizer_step": ckpts[0].optimizer_step,
        "schedule": ckpts[0].schedule.to_json(),
        "vocab_size": ckpts[0].model_state.get("vocab_size"),
        "span_weight": ckpts[0].model_state.get("span_weight"),
        "inputs": [
            {
                "path": str(p),
                "seed": c.seed,
                "payload_digest": c.to_json()["payload_digest"],
            }
            for p, c in zip(inputs, ckpts, strict=True)
        ],
        "ft_row_ids": list(ft_row_ids),
        "safetensors_sha256": hashlib.sha256(payload).hexdigest(),
        "n_tensors": len(tensors),
    }
    _atomic_write_bytes(
        manifest_path, (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="the .safetensors to write")
    parser.add_argument(
        "--ft-row-ids", nargs="*", default=[],
        help="the ledger ft rows of the inputs, recorded in the manifest",
    )
    args = parser.parse_args(argv)
    resolved = [p.resolve() for p in args.checkpoints]
    if len(set(resolved)) != len(resolved):
        raise SystemExit("the same checkpoint was named twice; it would be weighted double")
    if args.ft_row_ids and len(args.ft_row_ids) != len(args.checkpoints):
        raise SystemExit(
            f"{len(args.ft_row_ids)} --ft-row-ids for {len(args.checkpoints)} checkpoints"
        )
    ckpts = [Checkpoint.read(p) for p in args.checkpoints]
    names = [str(p) for p in args.checkpoints]
    try:
        tensors = average(ckpts, names)
        manifest = write(
            tensors, args.out, inputs=args.checkpoints, ckpts=ckpts, ft_row_ids=args.ft_row_ids
        )
    except AverageRefusal as exc:
        raise SystemExit(f"refused: {exc}") from exc
    print(f"averaged {len(ckpts)} checkpoints at step {ckpts[0].optimizer_step} -> {args.out}")
    print(f"manifest: {manifest} (weights only; not a resume point)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
