"""Where torch's MPS time goes on the 2B, and the torch half of the MLX parity check.

AUDIT/mac-speed-2026-09-28.md section 5 items 2-4, on real corpus tokens:

* ``matmul``: a 4096x2048 @ 2048x6144 bf16 matmul loop on MPS -> achieved TFLOP/s (item 4).
* ``full_ce``: forward + full-vocab logits over every position + cross-entropy, the audit's
  ``model(input_ids=x, labels=x)`` baseline.
* ``last_only``: the same forward, logits only at the last position and only for the item's
  letter rows -- ``labels=None`` (item 3).
* ``gdn_identity``: ``last_only`` with every Gated-DeltaNet mixer replaced by a shape-preserving
  identity (it returns its normed input), which removes the GDN's cost and changes the output:
  a timing probe only (item 2).

``--parity-out`` also writes, for the first ``--n-parity`` val items, the log-softmax over each
item's decode rows at the last prompt position -- the torch side of ``mac_mlx_bench.py parity``.

The tower is ``qd_train.backbone.load_text_tower`` (the repo's own loader), NOT remapped: parity
and timing are about the base model in the original vocabulary. One ``throughput`` row.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mac_bench_common as common
import torch

from qd_train.ledger import Ledger, RunRecorder
from qd_train.tristate import NotRun

M, K, N = 4096, 2048, 6144


def _sync() -> None:
    torch.mps.synchronize()


def _timed(fn, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        _sync()
        out.append(time.perf_counter() - t)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", type=Path, required=True, help="mac_zero_shot --dump-items output")
    ap.add_argument("--header", type=Path, required=True, help="the items' shard header.json")
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--parity-out", type=Path, required=True)
    ap.add_argument("--n-parity", type=int, default=50)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument(
        "--dtype", choices=["bf16", "fp32"], default="bf16",
        help="tower dtype. fp32 is the parity reference; timings are only taken in bf16",
    )
    ap.add_argument(
        "--parity-only", action="store_true",
        help="write the parity file and skip the matmul and forward timings",
    )
    args = ap.parse_args(argv)
    if not 1 <= args.n_parity <= 500 or not 1 <= args.reps <= 100:
        raise SystemExit("--n-parity must be in [1,500] and --reps in [1,100]")

    items = [json.loads(line) for line in args.items.read_text().splitlines() if line.strip()]
    stream = [t for it in items for t in it["prompt_ids"]]
    if len(stream) < max(args.lengths):
        raise SystemExit(f"{len(stream)} real tokens available, fewer than {max(args.lengths)}")
    letter_ids = items[0]["letter_ids"]

    recipe = {
        "tool": "tools/mac_torch_attrib.py", "backend": "torch-mps", "dtype": args.dtype,
        "parity_only": args.parity_only, "head": "fp32 (h.float() @ W.float())",
        "attn_implementation": "sdpa", "backbone_snapshot": args.snapshot.name,
        "backbone_vocab": 248320,
        "lengths": args.lengths, "warmup": args.warmup, "reps": args.reps,
        "n_parity": args.n_parity, "matmul_shape": [M, K, N], "batch": 1,
    }
    rec = RunRecorder(
        Ledger(args.ledger),
        entry_point=Path(__file__),
        protocol=common.protocol(
            header=args.header,
            recipe=recipe,
        ),
        run_kind="throughput",
        repo=common.REPO,
        env=common.environment(),
        wall_clock_s=None,
        cost=None,
        quick=True,
        quick_reason=common.QUICK_REASON,
        recipe=recipe,
        notes=(
            "AUDIT/mac-speed-2026-09-28.md section 5 items 2-4 on real val-prompt tokens "
            "(concatenated to each length; synthetic length, real ids). Timings exclude load; "
            "gdn_identity changes the model's output and is a timing probe only. Peak memory "
            "is torch.mps.driver_allocated_memory() sampled after each mode (not a true peak); "
            "the process max RSS is in the handoff from /usr/bin/time -l."
        ),
    )
    with rec:
        rec.metric("entry_tool_sha256", common.tool_digest(Path(__file__)))
        # -- item 4: the matmul rate ----------------------------------------------------------
        if not args.parity_only:
            ts = _matmul_times()
            flop = 2 * M * K * N
            rec.metric("matmul.bf16_tflops_median", common.num(
                flop / statistics.median(ts) / 1e12,
                f"torch.matmul bf16 {M}x{K} @ {K}x{N} on mps, 50 reps after 10 warmup, median "
                f"{statistics.median(ts) * 1e3:.3f} ms (min {min(ts) * 1e3:.3f})"))
            rec.metric("matmul.bf16_tflops_best",
                       common.num(flop / min(ts) / 1e12, "same, best rep"))

        from qd_train.backbone import load_text_tower
        from qd_train.memory import ADAMW_BF16, ADAMW_FP32

        tower = load_text_tower(
            args.snapshot, gradient_checkpointing=True,
            optimizer=ADAMW_FP32 if args.dtype == "fp32" else ADAMW_BF16,
            attn_implementation="sdpa", device="mps", dtype=args.dtype, rows=1,
            width=max(args.lengths),
        )
        model = tower.model.eval()
        weight = tower.lm_head_weight
        letters = torch.as_tensor(letter_ids, device="mps")

        # -- parity: log-softmax over each item's decode rows at its last prompt position ------
        par = items[: args.n_parity]
        t_par = time.perf_counter()
        with torch.no_grad(), args.parity_out.open("w", encoding="utf-8") as fh:
            for it in par:
                ids = torch.as_tensor([it["prompt_ids"]], device="mps")
                h = model(input_ids=ids).last_hidden_state[:, -1]
                rows = torch.as_tensor(it["letter_ids"], device="mps")
                z = (h.float() @ weight[rows].float().T)[0]
                full_top = int((h @ weight.T)[0].argmax())
                fh.write(json.dumps({
                    "row_id": it["row_id"], "n_tokens": len(it["prompt_ids"]),
                    "letter_logits": z.tolist(),
                    "letter_logsoftmax": torch.log_softmax(z, -1).tolist(),
                    "full_vocab_argmax": full_top,
                }) + "\n")
        _sync()
        rec.metric("parity.torch_items", common.num(
            len(par), f"items written to {args.parity_out.name} in "
            f"{time.perf_counter() - t_par:.1f}s"))

        # -- items 2 and 3: forward modes -------------------------------------------------------
        def full_ce(ids: torch.Tensor) -> None:
            h = model(input_ids=ids).last_hidden_state
            logits = (h @ weight.T).float()
            torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])

        def last_only(ids: torch.Tensor) -> None:
            h = model(input_ids=ids).last_hidden_state[:, -1]
            _ = h @ weight[letters].T

        gdn = [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
        rec.metric("model.gdn_mixers", common.num(len(gdn), "Qwen3_5GatedDeltaNet modules found"))
        for length in [] if args.parity_only else args.lengths:
            ids = torch.as_tensor([stream[:length]], device="mps")
            for mode, fn in (("full_ce", full_ce), ("last_only", last_only)):
                with torch.no_grad():
                    ts = _timed(functools.partial(fn, ids), args.warmup, args.reps)
                _record(rec, mode, length, ts)
            originals = [m.forward for m in gdn]
            try:
                for m in gdn:
                    m.forward = _identity_mixer  # type: ignore[method-assign]
                with torch.no_grad():
                    ts = _timed(functools.partial(last_only, ids), args.warmup, args.reps)
                _record(rec, "gdn_identity", length, ts)
            finally:
                for m, f in zip(gdn, originals, strict=True):
                    m.forward = f  # type: ignore[method-assign]
        rec.noul_rate = NotRun(reason="a throughput benchmark decodes no verdicts")
    row = rec.row
    if row is None:
        raise SystemExit("no row written")
    for k, v in sorted(row.metrics.items()):
        print(f"  {k}: {v.to_json().get('value')}  {str(v.to_json().get('detail', ''))[:140]}")
    print(f"ledger row: {row.row_id}  {args.ledger}")
    return 0


def _matmul_times() -> list[float]:
    a = torch.randn(M, K, device="mps", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="mps", dtype=torch.bfloat16)
    return _timed(functools.partial(torch.matmul, a, b), warmup=10, reps=50)


def _identity_mixer(hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
    return hidden_states


def _record(rec: RunRecorder, mode: str, length: int, ts: list[float]) -> None:
    mean, best = statistics.mean(ts), min(ts)
    alloc = torch.mps.driver_allocated_memory() / 1e9
    rec.metric(f"fwd.{mode}.L{length}.s_mean", common.num(
        mean, f"{mode} forward, batch 1 x {length} real tokens, {len(ts)} reps: mean {mean:.4f}s "
        f"min {best:.4f}s, {length / mean:.0f} tok/s mean; mps driver alloc {alloc:.1f} GB"))
    rec.metric(f"fwd.{mode}.L{length}.tok_per_s", common.num(length / mean, "mean-based"))
    print(f"{mode} L={length}: mean {mean:.3f}s min {best:.3f}s {length / mean:.0f} tok/s "
          f"alloc {alloc:.1f}GB", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
