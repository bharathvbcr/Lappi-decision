"""Where the fp32-master recipe's memory actually goes, stage by stage.

``tools/gh200_footprint.py --optimizer master`` measures 42.13 GiB at rows=1 width=2048
against ``estimate_step``'s 28.40 GiB -- a 49% shortfall, where the bf16 recipe's is 22%.
``SAFETY_FRACTION = 0.35`` covers the second and not the first, so the budget is wrong for
this recipe in the direction that matters: it would say a run fits when it does not.

A ratio is not a diagnosis. This attributes the gap by reading
``torch.cuda.max_memory_allocated`` between the stages of one step, so each component is a
measured number rather than a share of a residual::

    load -> forward+backward -> cast grads to fp32 -> optimizer step

and it runs the step twice with ``foreach`` on and off, because torch's fused multi-tensor
path allocates temporaries proportional to the whole parameter set and that is the obvious
candidate for a cost no per-parameter budget would contain.

Run on the box::

    PYTHONPATH=/home/ubuntu/qwen-decision/python /home/ubuntu/qd-venv/bin/python \
        /home/ubuntu/qwen-decision/tools/master_overhead.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch

from qd_train.backbone import load_text_tower
from qd_train.memory import OptimizerSpec

SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)
MASTER = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
WIDTH = 2048
GiB = 1024**3


def _mark(label: str, previous: int) -> int:
    """Print the peak since the last mark, and return the new baseline."""
    peak = torch.cuda.max_memory_allocated()
    live = torch.cuda.memory_allocated()
    print(f"  {label:34s} peak {peak / GiB:7.2f} GiB   live {live / GiB:7.2f} GiB   "
          f"(+{(peak - previous) / GiB:6.2f} since last)")
    torch.cuda.reset_peak_memory_stats()
    return live


def run(foreach: bool) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"\n=== foreach={foreach} ===")

    # Named rather than inherited: attention is where the activation memory is, so a
    # footprint taken on another kernel is a measurement of something else.
    tower = load_text_tower(
        SNAPSHOT, gradient_checkpointing=True, optimizer=MASTER,
        attn_implementation="sdpa",
        device="cuda", dtype="bf16", rows=1, width=WIDTH,
    )
    n = sum(p.numel() for p in tower.model.parameters() if p.requires_grad)
    base = _mark("tower loaded (bf16 weights)", 0)

    live = [p for p in tower.model.parameters() if p.requires_grad]
    masters = [p.detach().clone().to(torch.float32).requires_grad_(True) for p in live]
    after_master = _mark("fp32 masters cloned", base)

    inner = torch.optim.AdamW(masters, lr=1e-5, foreach=foreach)

    ids = torch.randint(0, tower.vocab_size, (1, WIDTH), device="cuda")
    out = tower.model(input_ids=ids)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    hidden.float().pow(2).mean().backward()
    after_bwd = _mark("forward + backward (bf16 grads)", after_master)

    for p, m in zip(live, masters, strict=True):
        m.grad = p.grad.detach().to(torch.float32)
    after_cast = _mark("grads cast to fp32", after_bwd)

    inner.step()
    _mark("optimizer step (moments + temps)", after_cast)

    # Steady state: a second step, with the moments already allocated.
    for p in live:
        p.grad = None
    for _p, m in zip(live, masters, strict=True):
        m.grad = None
    out = tower.model(input_ids=ids)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    hidden.float().pow(2).mean().backward()
    for p, m in zip(live, masters, strict=True):
        m.grad = p.grad.detach().to(torch.float32)
    inner.step()
    steady = torch.cuda.max_memory_allocated()
    print(f"  {'steady-state second step':34s} peak {steady / GiB:7.2f} GiB")

    print(f"  per-parameter accounting over {n:,} params:")
    for label, total in (
        ("bf16 weights", n * 2), ("bf16 grads", n * 2), ("fp32 master", n * 4),
        ("fp32 grads (the cast)", n * 4), ("fp32 moments", n * 8),
    ):
        print(f"    {label:24s} {total / GiB:6.2f} GiB")
    enumerated = n * (2 + 2 + 4 + 4 + 8)
    print(f"    {'enumerated total':24s} {enumerated / GiB:6.2f} GiB   "
          f"vs measured steady {steady / GiB:.2f} GiB")

    result = {
        "foreach": foreach, "trainable": n, "steady_peak_bytes": steady,
        "enumerated_bytes": enumerated,
        "unexplained_bytes": steady - enumerated,
    }
    del inner, masters, live, tower, ids, out, hidden
    torch.cuda.empty_cache()
    return result


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This script exists to attribute a CUDA measurement; refusing "
            "to report a CPU number under a name that says cuda."
        )
    print(f"device: {torch.cuda.get_device_properties(0).name}")
    print(f"torch : {torch.__version__}")
    results = [run(foreach=True), run(foreach=False)]

    print("\n=== what foreach costs ===")
    on, off = results
    saved = on["steady_peak_bytes"] - off["steady_peak_bytes"]
    print(f"  foreach=True  steady peak {on['steady_peak_bytes'] / GiB:.2f} GiB")
    print(f"  foreach=False steady peak {off['steady_peak_bytes'] / GiB:.2f} GiB")
    print(f"  difference    {saved / GiB:.2f} GiB "
          f"({saved / on['steady_peak_bytes']:.1%} of the foreach=True peak)")
    for r in results:
        print(f"  foreach={r['foreach']!s:5s} unexplained by the per-parameter terms: "
              f"{r['unexplained_bytes'] / GiB:6.2f} GiB")

    out = Path("/home/ubuntu/master_overhead.json")
    out.write_text(json.dumps({"width": WIDTH, "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
