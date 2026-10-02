"""The span-head init, written once and pinned by digest: ojas plan Q3 Amendment 2 (iv).

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_span_head_init.py \\
        --hidden 2048 --seed 0 \\
        --out data/checkpoints/span-head-init/span_head_init-seed0.safetensors \\
        --manifest crates/qd-train/tests/fixtures/span-head-init-seed0.manifest.json

A **fixture generator** (user policy: Python only as an oracle). The Rust trainer does not
port torch's random stream. It loads this file by path, recomputes the two digests the manifest
records, and refuses to start on a mismatch. A second owner of the stream would be a copy that
drifts.

**The head comes from ``QwenDecisionStep``'s own construction path** (``backbone.py:983-1000``):
``torch.manual_seed(seed)``, then ``SpanPointerHead(tower.hidden_size)`` built on the CPU and
moved to float32. The tower is a stand-in at the requested hidden size, from the rung (b)
oracle's tiny family. Nothing draws from the random stream between those lines, so the tower's
other sizes do not enter the head.

Three constructions must agree bit for bit before anything is written:

* F's path: a bf16 tower under the ``master`` recipe;
* an fp32 tower;
* ``torch.manual_seed(seed); SpanPointerHead(hidden)`` directly.

``RecordingStep``, which ``make_step`` builds, calls ``QwenDecisionStep.__init__`` first and
then sets four plain attributes. It draws nothing.

**What pins the file**, the formula Fable named:

* the sha256 of the file's bytes;
* the repository's content digest, ``run_control._sidecar_digest`` over the
  ``qd-tensor-ref-v1`` per-tensor digests, names ``span_head.*``, dtype ``float32``.

The content digest is computed from the bytes on disk through
``run_control.tensor_refs_from_safetensors``, the path a reader follows. It is checked equal to
the digest over the step's own ``state()['span_head']``. ``_sidecar_digest`` is private to
``run_control``; this oracle imports it rather than restating it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402
from qd_train_oracle_tiny import (  # noqa: E402
    SPAN_HEAD_PREFIX,
    _git_head,
    _version,
    make_step,
    sha256_file,
    write_snapshot,
)

from qd_train.heads import SpanPointerHead  # noqa: E402
from qd_train.run_control import (  # noqa: E402
    TensorRef,
    _sidecar_digest,
    tensor_refs_from_safetensors,
)

#: The stand-in tower: two layers (one GDN, one attention), the smallest tiny-family tower
#: load_text_tower accepts. Its depth and vocabulary never reach the head.
STAND_IN_LAYERS: Final[int] = 2
#: One batch row at the smallest bucket: the plan load_text_tower sizes its budget by.
STAND_IN_PLAN: Final[list[dict[str, Any]]] = [{"rows": [0], "width": 64}]


def head_from_step(hidden: int, seed: int, dtype: str, scratch: Path) -> dict[str, torch.Tensor]:
    """The span head ``QwenDecisionStep(seed=seed)`` builds over a ``hidden``-wide tower of
    ``dtype``, as ``span_head.*`` float32 tensors."""
    snapshot = scratch / f"snapshot-h{hidden}"
    if not snapshot.is_dir():
        write_snapshot(snapshot, STAND_IN_LAYERS, hidden)
    _, step = make_step(
        snapshot=snapshot,
        layers=STAND_IN_LAYERS,
        dtype=dtype,
        plan=STAND_IN_PLAN,
        steps=1,
        lr=1e-5,
        seed=seed,
        lower_layers_n=0,
        lower_lr_scale=1.0,
        hidden=hidden,
    )
    if step.tower.hidden_size != hidden:
        raise SystemExit(f"the stand-in tower is {step.tower.hidden_size} wide, not {hidden}")
    state = step.state()["span_head"]
    refs = {f"{SPAN_HEAD_PREFIX}{k}": v for k, v in state.items()}
    tensors = {
        f"{SPAN_HEAD_PREFIX}{n}": p.detach().to("cpu").contiguous().clone()
        for n, p in step.span_head.state_dict().items()
    }
    for name, t in tensors.items():
        if t.dtype != torch.float32:
            raise SystemExit(f"{name} is {t.dtype}; the span head is float32 (backbone.py:997)")
        ref: TensorRef = refs[name]
        if ref.digest() != _ref(t).digest():
            raise SystemExit(f"{name}: the step's state() and its module disagree")
    return tensors


def _ref(t: torch.Tensor) -> TensorRef:
    """A tensor as ``QwenDecisionStep.state`` turns one into a ``TensorRef`` (backbone.py)."""
    host = t.detach().to("cpu").contiguous()
    return TensorRef(
        dtype=str(host.dtype).removeprefix("torch."),
        shape=tuple(host.shape),
        data=host.reshape(-1).view(torch.uint8).numpy().tobytes(),
    )


def direct_head(hidden: int, seed: int) -> dict[str, torch.Tensor]:
    """``torch.manual_seed(seed); SpanPointerHead(hidden)``: the two lines restated, as a
    cross-check of the construction path, never as the source."""
    torch.manual_seed(seed)
    head = SpanPointerHead(hidden).to(dtype=torch.float32)
    return {f"{SPAN_HEAD_PREFIX}{n}": t.detach().clone() for n, t in head.state_dict().items()}


def same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(
        a[n].dtype == b[n].dtype and a[n].shape == b[n].shape and torch.equal(a[n], b[n]) for n in a
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(1)
    scratch = Path(tempfile.mkdtemp(prefix="qd-span-head-init-"))
    via_bf16 = head_from_step(args.hidden, args.seed, "bf16", scratch)
    via_fp32 = head_from_step(args.hidden, args.seed, "fp32", scratch)
    direct = direct_head(args.hidden, args.seed)
    checks = {
        "bf16_master_tower_equals_fp32_tower": same(via_bf16, via_fp32),
        "step_equals_direct_manual_seed_then_SpanPointerHead": same(via_bf16, direct),
    }
    if not all(checks.values()):
        raise SystemExit(f"the constructions disagree: {checks}; nothing written")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # No metadata: the file is the four tensors and nothing else, so a reader in any language
    # sees exactly what the digests cover.
    save_file(via_bf16, str(args.out))
    payload = args.out.read_bytes()
    on_disk = tensor_refs_from_safetensors(payload)
    if sorted(on_disk) != sorted(via_bf16):
        raise SystemExit(f"{args.out} holds {sorted(on_disk)}, not {sorted(via_bf16)}")
    content = _sidecar_digest(on_disk)
    in_memory = _sidecar_digest({n: _ref(t) for n, t in via_bf16.items()})
    if content != in_memory:
        raise SystemExit("the content digest from the file differs from the in-memory one")
    checks["content_digest_from_file_equals_in_memory"] = True

    compared: dict[str, Any] | None = None
    if args.compare_with is not None:
        other = {
            n: t
            for n, t in load_file(str(args.compare_with)).items()
            if n.startswith(SPAN_HEAD_PREFIX)
        }
        compared = {
            "file": str(args.compare_with),
            "sha256": sha256_file(args.compare_with),
            "span_head_tensors_bit_identical": same(via_bf16, other),
        }
        if not compared["span_head_tensors_bit_identical"]:
            raise SystemExit(f"{args.compare_with}'s span_head.* differ from this head")

    manifest: dict[str, Any] = {
        "what": f"the span head QwenDecisionStep(seed={args.seed}) initialises at hidden "
        f"{args.hidden}: ojas plan Q3 Amendment 2 (iv). The Rust trainer loads the file by "
        "path, recomputes sha256 and content_digest, and refuses to start on a mismatch",
        "file": {
            "path": _repo_relative(args.out),
            "tracked": _repo_relative(args.out).startswith("crates/"),
            "bytes": len(payload),
            "sha256": sha256_file(args.out),
            "safetensors_metadata": None,
        },
        "content_digest": {
            "value": content,
            "formula": "run_control._sidecar_digest over run_control.TensorRef.digest of each "
            "tensor: per tensor sha256(b'qd-tensor-ref-v1' || u32be(len(dtype)) || dtype ascii "
            "|| u32be(ndim) || u64be(each axis) || u64be(nbytes) || little-endian data); over "
            "the set sha256(b'qd-checkpoint-sidecar-v1' || u64be(count) || for each name in "
            "sorted order: u64be(len(name utf-8)) || name || the 32 raw digest bytes)",
            "names": f"{SPAN_HEAD_PREFIX}*",
            "dtype": "float32",
            "source": "python/qd_train/run_control.py _TENSOR_REF_DOMAIN, TensorRef.digest, "
            "_SIDECAR_DOMAIN, _sidecar_digest",
        },
        "tensors": {
            n: {
                "dtype": r.dtype,
                "shape": list(r.shape),
                "nbytes": r.nbytes,
                "digest": r.digest(),
            }
            for n, r in sorted(on_disk.items())
        },
        "construction": {
            "path": "python/qd_train/backbone.py:983-1000: torch.manual_seed(seed), then "
            "SpanPointerHead(tower.hidden_size) built on the CPU and moved to float32",
            "seed": args.seed,
            "hidden": args.hidden,
            "stand_in_tower": f"tiny Qwen3.5 family (tools/qd_train_oracle_tiny.py), "
            f"{STAND_IN_LAYERS} layers, hidden {args.hidden}; only its hidden size reaches the "
            "head",
            "checks": checks,
            "default_device_or_dtype_set_anywhere": "no call to torch.set_default_device or "
            "torch.set_default_dtype in tools/real_ft_run.py or python/qd_train (rg, "
            "2026-10-01; the one torch.device context there is the 'meta' skeleton at "
            "backbone.py:559, before the step), so a CUDA run's head is drawn on the CPU too "
            "[inferred: no CUDA host here]",
        },
        "compared_with": compared,
        "generator": {
            "script": "tools/qd_train_oracle_span_head_init.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "command": " ".join(
                [
                    "PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python",
                    "tools/qd_train_oracle_span_head_init.py",
                    *[str(a) for a in args.argv],
                ]
            ),
            "git_head": _git_head(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": _version("safetensors"),
            "device": "cpu",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the .safetensors to write")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="a safetensors whose span_head.* tensors must equal this head bit for bit",
    )
    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if args.hidden < 1 or args.seed < 0:
        raise SystemExit("--hidden must be positive and --seed non-negative")
    manifest = build(args)
    print(
        f"{args.out}: {manifest['file']['bytes']:,} bytes, sha256 {manifest['file']['sha256']}, "
        f"content digest {manifest['content_digest']['value']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
