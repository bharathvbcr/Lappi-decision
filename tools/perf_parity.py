#!/usr/bin/env python3
"""One arm of a Tier-A / Tier-B parity comparison, through the REAL training loop.

``real_ft_run._train`` -- the production tower build, budget, recorder, ``train_ft`` loop and
ledger row -- over a fixed batch subset of a real shard set, ``passes`` times. Each arm is a
fresh process, because two arms are two code trees (``--code-root``) that both name their
package ``qd_train``.

What an arm writes (one JSONL line to ``--result``), and what ``--compare`` reads:

* ``loss_log_digest`` -- the ``train.loss_log_digest`` metric off the ft row the loop wrote,
  the comparator the resume proof (ft 7a93bb47 against f0f0fa95) used;
* ``letter`` / ``span`` -- every micro-batch's two channel losses, as ``float.hex``;
* ``consumed_digest`` -- the batch-order digest the loop folds (``trainer._fold`` over the
  re-indexed source ``_train`` builds), so a loss difference can never be a data-order
  difference in disguise.

``--deterministic`` is real_ft_run's own: CUBLAS_WORKSPACE_CONFIG before cuBLAS initialises,
then ``torch.use_deterministic_algorithms(True)``, and ``deterministic=True`` into the recipe.

``--compare A.jsonl:tag B.jsonl:tag ...`` (no GPU) reports, for every pair against the
first, whether all three are identical and, where not, the first differing step and the
largest per-step deviation in each channel.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _append(path: Path, row: dict[str, Any]) -> None:
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def run_arm(args: argparse.Namespace) -> int:
    if args.deterministic:
        # What real_ft_run.py does at import for --deterministic, done before torch loads.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(code_root / "tools"))
    import real_ft_run
    import torch
    from perf_step import _select_batches

    from qd_data.config import DataConfig
    from qd_train.artifacts import SLOT_SPAN
    from qd_train.ledger import Ledger
    from qd_train.run_control import ConsumedPrefix
    from qd_train.shards import ShardReader
    from qd_train.trainer import _fold

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if args.allow_stale_shards:
        import functools

        import qd_train.shards as shards_mod

        shards_mod.assert_shard_trainable = functools.partial(
            shards_mod.assert_shard_trainable, allow_stale_code=True
        )
    config = DataConfig()
    reader = ShardReader(args.out / "shards" / "train", config=config, repo_root=args.out)
    plan, selection = _select_batches(
        reader, batch_tokens=args.batch_tokens, seed=config.seed, width_min=args.width_min,
        width_max=args.width_max, n=args.n_batches, scan_limit=100_000,
    )
    if len(plan) < args.n_batches and not args.allow_fewer:
        raise SystemExit(f"only {len(plan)} batches in range; pass --allow-fewer to cycle them")

    # The loop's own source, rebuilt: _train re-indexes plan x passes exactly like this.
    consumed = ConsumedPrefix()
    index = 0
    for _ in range(args.passes):
        for batch in plan:
            _fold(consumed, dataclasses.replace(batch, index=index))
            index += 1

    if args.dry_run:
        print(json.dumps({"dry_run": "ok", "batches": len(plan),
                          "widths": sorted(set(selection["widths"])),
                          "span_batches": int(sum(1 for b in plan
                                                  if (b.slot_kind == SLOT_SPAN).any())),
                          "consumed_digest": consumed.hexdigest(),
                          "real_ft_run": real_ft_run.__file__}), flush=True)
        return 0
    extra: dict[str, Any] = {}
    if args.checkpoint_skip_layers:
        extra["checkpoint_skip_layers"] = args.checkpoint_skip_layers
    ledger = Ledger(args.ledger)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    run = real_ft_run._train(
        reader=reader, plan=plan, passes=args.passes, device="cuda", seed=args.seed,
        hidden=0, heads=0, lr=args.lr, span_weight=1.0, ledger=ledger, tag=args.tag,
        quick_reasons=[
            f"perf parity arm: {len(plan)} batches x {args.passes} passes on a fixed "
            "subset; a comparison of code paths, not a training run"
        ],
        backbone=args.backbone, optimizer_recipe="master", deterministic=args.deterministic,
        attn_implementation="sdpa", n_gpus=1, usd_per_hour=2.29, instance="lambda-1xgh200",
        cap_s=args.cap_s, batch_tokens=args.batch_tokens, **extra,
    )
    wall = time.perf_counter() - t0
    step = run.pop("_step")
    row = next(r for r in ledger.rows() if r.row_id == run["ft_row_id"])
    metrics = row.metrics
    digest = metrics["train.loss_log_digest"]
    out = {
        "tag": args.tag,
        "code_root": str(code_root),
        "code_commit": row.code_commit,
        "ft_row_id": row.row_id,
        "recipe_hash": row.protocol.recipe_hash,
        "deterministic": args.deterministic,
        "checkpoint_skip_layers": args.checkpoint_skip_layers,
        "shape": {"batch_tokens": args.batch_tokens, "widths": selection["widths"],
                  "rows": selection["rows"], "passes": args.passes},
        "span_batches": int(sum(1 for b in plan if (b.slot_kind == SLOT_SPAN).any())),
        "steps": run["optimizer_steps"],
        "termination": run["termination"],
        "loss_log_digest": digest.value if hasattr(digest, "value") else digest["value"],
        "consumed_digest": consumed.hexdigest(),
        "letter": [float(x).hex() for x in step.letter_log],
        "span": [float(x).hex() for x in step.span_log],
        "train_path": (metrics.get("train.path").value
                       if hasattr(metrics.get("train.path"), "value")
                       else (metrics.get("train.path") or {}).get("value")),
        "wall_s": round(wall, 2),
        "train_wall_s": run["wall_clock_s"],
        "steps_per_s": run["steps_per_s"],
        "positions": int(sum(int(b.tokens.size) for b in plan)) * args.passes,
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    out["pos_per_s"] = round(out["positions"] / float(run["wall_clock_s"]), 1)
    _append(args.result, out)
    print(json.dumps({k: out[k] for k in ("tag", "steps", "loss_log_digest", "consumed_digest",
                                          "pos_per_s", "peak_alloc_gib", "termination")}),
          flush=True)
    return 0


def _load(spec: str) -> dict[str, Any]:
    path, _, tag = spec.partition(":")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    hits = [r for r in rows if r["tag"] == tag]
    if len(hits) != 1:
        raise SystemExit(f"{spec}: {len(hits)} rows tagged {tag!r}, need exactly one")
    return hits[0]


def compare(specs: list[str]) -> int:
    base = _load(specs[0])
    worst = 0
    for spec in specs[1:]:
        other = _load(spec)
        report: dict[str, Any] = {"base": base["tag"], "other": other["tag"]}
        for key in ("loss_log_digest", "consumed_digest", "steps"):
            report[f"{key}_identical"] = base[key] == other[key]
        for ch in ("letter", "span"):
            a = [float.fromhex(x) for x in base[ch]]
            b = [float.fromhex(x) for x in other[ch]]
            diffs = [i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y]
            report[f"{ch}_identical"] = a == b
            report[f"{ch}_first_diff_step"] = diffs[0] if diffs else None
            report[f"{ch}_max_abs_dev"] = max(
                (abs(x - y) for x, y in zip(a, b, strict=False)), default=0.0
            )
        report["identical"] = all(
            report[k] for k in ("loss_log_digest_identical", "consumed_digest_identical",
                                "steps_identical", "letter_identical", "span_identical")
        )
        report["pos_per_s"] = {base["tag"]: base["pos_per_s"], other["tag"]: other["pos_per_s"]}
        report["peak_alloc_gib"] = {base["tag"]: base["peak_alloc_gib"],
                                    other["tag"]: other["peak_alloc_gib"]}
        print(json.dumps(report, sort_keys=True))
        worst = max(worst, 0 if report["identical"] else 1)
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare", nargs="+", metavar="FILE:TAG")
    ap.add_argument("--code-root", type=Path)
    ap.add_argument("--out", type=Path, help="shard set root (has shards/train)")
    ap.add_argument("--backbone", type=Path)
    ap.add_argument("--batch-tokens", type=int)
    ap.add_argument("--width-min", type=int, default=0)
    ap.add_argument("--width-max", type=int, default=1 << 30)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--allow-fewer", action="store_true")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cap-s", type=float, default=900.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--checkpoint-skip-layers", type=int, default=0)
    ap.add_argument("--allow-stale-shards", action="store_true")
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--result", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--dry-run", action="store_true", help="stop before the tower loads")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args.compare)
    missing = [k for k in ("code_root", "out", "backbone", "batch_tokens", "ledger", "result",
                           "tag") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"an arm needs {missing}")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
