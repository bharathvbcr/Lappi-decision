"""Fixture generator (reference oracle, not shipped): the span-head init's content digest.

Amendment 2 (iv) (``AUDIT/ojas-training-2026-10-01/fable-rung-b-bars.md``) pins the span-head
init file by its sha256 and by the repo's existing tensor-set digest:
``run_control._sidecar_digest`` over ``run_control.TensorRef`` (domain ``qd-tensor-ref-v1``),
with names ``span_head.*`` and dtype ``float32``. The Rust trainer recomputes both and refuses
on a mismatch. This script computes the digest with Python's own functions on the tiny head
(the ``span_head.*`` tensors of rung (b)'s ``tiny-published/init.safetensors``), building each
``TensorRef`` exactly as ``QwenDecisionStep`` does for a checkpoint (``backbone.py`` ``refs``:
dtype ``str(t.dtype)`` without ``torch.``, the shape, and the raw little-endian bytes), so
the Rust port is checked against Python's own code rather than a restatement of it. Run::

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_head_digest.py \\
        --init crates/qd-train/tests/fixtures/tiny-published/init.safetensors \\
        --out crates/qd-train/tests/fixtures/span-head-init-digest-tiny.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from qd_train.run_control import TensorRef, _sidecar_digest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tensors = {k: v for k, v in load_file(str(args.init)).items() if k.startswith("span_head.")}
    if sorted(tensors) != sorted(
        ["span_head.abstain_end", "span_head.abstain_start", "span_head.end_proj.weight", "span_head.start_proj.weight"]
    ):
        raise SystemExit(f"{args.init}: span_head tensors are {sorted(tensors)}")
    refs: dict[str, TensorRef] = {}
    for name, t in tensors.items():
        host = t.detach().to("cpu").contiguous()
        if host.dtype != torch.float32:
            raise SystemExit(f"{name} is {host.dtype}, not float32")
        refs[name] = TensorRef(
            dtype=str(host.dtype).removeprefix("torch."),
            shape=tuple(host.shape),
            data=host.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    script = Path(__file__).read_bytes()
    out = {
        "what": "run_control._sidecar_digest over the span_head.* TensorRefs of the tiny init (Amendment 2 (iv))",
        "generator": {
            "script": "tools/qd_train_oracle_head_digest.py",
            "script_sha256": hashlib.sha256(script).hexdigest(),
            "argv": sys.argv[1:],
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "source": args.init.name,
        "source_sha256": hashlib.sha256(args.init.read_bytes()).hexdigest(),
        "hidden_size": int(tensors["span_head.abstain_start"].shape[0]),
        "tensors": {
            name: {"dtype": r.dtype, "shape": list(r.shape), "digest": r.digest()} for name, r in sorted(refs.items())
        },
        "content_digest": _sidecar_digest(refs),
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"content_digest {out['content_digest']} over {len(refs)} tensors, hidden {out['hidden_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
