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

``--p2 FILE --base TAG... --cand TAG...`` (no GPU) is Fable's Tier-B screen
(campaign/f-j7prime-preregistered.json ``no_mask.p2_rule``) over default-kernel arms on the
same batches: per step t and channel, the baseline spread S(t) = max over baseline pairs of
|B_i(t) - B_j(t)| and the cross deviation D(t) = max over (candidate, baseline) pairs of
|M_i(t) - B_j(t)|; the channel passes when D(t) <= S(t) on at least ``--min-frac`` of its live
steps. A channel with no live step (no arm ever saw it non-zero) is NOT RUN, and a shape with a
channel not run does not pass.
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
    span_batches = int(sum(1 for b in plan if (b.slot_kind == SLOT_SPAN).any()))
    if span_batches < args.min_span_batches:
        raise SystemExit(
            f"{span_batches} of {len(plan)} selected batches carry a span row; this arm needs "
            f"at least {args.min_span_batches}, or its span channel would be compared on nothing"
        )

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
                          "span_batches": span_batches,
                          "consumed_digest": consumed.hexdigest(),
                          "real_ft_run": real_ft_run.__file__}), flush=True)
        return 0
    extra: dict[str, Any] = {}
    if args.checkpoint_skip_layers:
        extra["checkpoint_skip_layers"] = args.checkpoint_skip_layers
    if args.train_attention_mask != "padding":
        extra["train_attention_mask"] = args.train_attention_mask
    ledger = Ledger(args.ledger)
    on_cuda = args.device == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    run = real_ft_run._train(
        reader=reader, plan=plan, passes=args.passes, device=args.device, seed=args.seed,
        hidden=64, heads=2, lr=args.lr, span_weight=1.0, ledger=ledger, tag=args.tag,
        quick_reasons=[
            f"perf parity arm: {len(plan)} batches x {args.passes} passes on a fixed "
            "subset; a comparison of code paths, not a training run"
        ],
        backbone=None if args.stand_in else args.backbone,
        optimizer_recipe="bf16" if args.stand_in else "master", deterministic=args.deterministic,
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
        "train_attention_mask": args.train_attention_mask,
        "shape": {"batch_tokens": args.batch_tokens, "widths": selection["widths"],
                  "rows": selection["rows"], "passes": args.passes},
        "span_batches": span_batches,
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
        "peak_alloc_gib": (round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                           if on_cuda else None),
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


#: Fable's P2 rule: D(t) <= S(t) on at least this fraction of a channel's live steps.
P2_MIN_FRAC = 0.9
#: Fable: the baseline spread is measured on at least this many repeats.
P2_MIN_BASELINES = 3


def _channel(rows: list[dict[str, Any]], ch: str) -> list[list[float]]:
    return [[float.fromhex(x) for x in r[ch]] for r in rows]


def p2_screen(
    bases: list[dict[str, Any]], cands: list[dict[str, Any]], *, min_frac: float = P2_MIN_FRAC
) -> dict[str, Any]:
    """Fable's P2 screen over parity-arm result rows (see the module docstring).

    Refuses arms that are not comparable step for step: fewer than three baselines, no
    candidate, different step counts, or a different batch order (``consumed_digest``). A
    channel is *live* at step t when any arm's value there is non-zero (a span channel is
    0.0 on a letter-only batch in every arm, which is evidence of nothing); a channel with no
    live step is ``not_run``, and the verdict is ``pass`` only when every channel ran and
    passed, ``fail`` when one ran and failed, ``not_run`` otherwise.
    """
    if len(bases) < P2_MIN_BASELINES:
        raise SystemExit(
            f"P2 needs at least {P2_MIN_BASELINES} baseline repeats for its spread, got "
            f"{len(bases)}"
        )
    if not cands:
        raise SystemExit("P2 needs at least one candidate arm")
    arms = [*bases, *cands]
    digests = {r["consumed_digest"] for r in arms}
    if len(digests) != 1:
        raise SystemExit(f"P2 arms ran over different batches: consumed digests {sorted(digests)}")
    lengths = {(r["steps"], len(r["letter"]), len(r["span"])) for r in arms}
    if len(lengths) != 1:
        raise SystemExit(f"P2 arms differ in steps or logged channel lengths: {sorted(lengths)}")
    out: dict[str, Any] = {
        "base": [r["tag"] for r in bases], "cand": [r["tag"] for r in cands],
        "min_frac": min_frac, "channels": {},
    }
    for ch in ("letter", "span"):
        b = _channel(bases, ch)
        m = _channel(cands, ch)
        n = len(b[0])
        live = [t for t in range(n) if any(arm[t] != 0.0 for arm in (*b, *m))]
        if not live:
            out["channels"][ch] = {"status": "not_run", "live_steps": 0,
                                   "reason": "no arm logged a non-zero value in this channel"}
            continue
        s = {t: max(abs(b[i][t] - b[j][t]) for i in range(len(b)) for j in range(i + 1, len(b)))
             for t in live}
        d = {t: max(abs(x[t] - y[t]) for x in m for y in b) for t in live}
        ok = [t for t in live if d[t] <= s[t]]
        max_s = max(s.values())
        over = [t for t in live if d[t] > max_s]
        frac = len(ok) / len(live)
        out["channels"][ch] = {
            "status": "ran", "live_steps": len(live), "steps_d_le_s": len(ok),
            "frac_d_le_s": frac, "passed": frac >= min_frac,
            "median_d": sorted(d.values())[len(d) // 2], "max_d": max(d.values()),
            "median_s": sorted(s.values())[len(s) // 2], "max_s": max_s,
            "first_step_d_over_max_s": over[0] if over else None,
            "final_base": [arm[-1] for arm in b], "final_cand": [arm[-1] for arm in m],
        }
    statuses = [c["status"] for c in out["channels"].values()]
    if any(c.get("passed") is False for c in out["channels"].values()):
        out["verdict"] = "fail"
    elif all(st == "ran" for st in statuses):
        out["verdict"] = "pass"
    else:
        out["verdict"] = "not_run"
    return out


#: The shapes whose P2 verdicts the outcome run needs; a verdict's shape is its baseline tags'
#: prefix (``A-mask-1`` -> ``A``), as perf_nomask_p2_body.sh names them.
P2_SHAPES = ("A", "B")


def p2_gate(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """Whether the outcome run may go: the latest P2 verdict of every shape in ``P2_SHAPES``
    is ``pass``. A shape with no verdict, or a ``not_run`` one, cancels it as a ``fail`` does
    (Fable: a P2 fail cancels the outcome run; an unexamined screen is not a passed one)."""
    latest: dict[str, str] = {}
    for row in rows:
        latest[str(row["base"][0]).split("-", 1)[0]] = str(row["verdict"])
    state = {shape: latest.get(shape, "missing") for shape in P2_SHAPES}
    ok = all(v == "pass" for v in state.values())
    return ok, f"P2 verdicts {state}: outcome run {'may go' if ok else 'cancelled'}"


def _tagged(path: Path, tags: list[str]) -> list[dict[str, Any]]:
    return [_load(f"{path}:{tag}") for tag in tags]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compare", nargs="+", metavar="FILE:TAG")
    ap.add_argument("--p2", type=Path, metavar="FILE",
                    help="Fable's P2 screen over the --base and --cand arms in FILE")
    ap.add_argument("--base", nargs="+", default=[], metavar="TAG")
    ap.add_argument("--cand", nargs="+", default=[], metavar="TAG")
    ap.add_argument("--p2-out", type=Path, help="append the P2 verdict row here too")
    ap.add_argument("--p2-gate", type=Path, metavar="VERDICTS",
                    help="exit 0 only if every shape's latest P2 verdict in VERDICTS is pass")
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
    ap.add_argument("--train-attention-mask", choices=("padding", "none"), default="padding")
    ap.add_argument("--min-span-batches", type=int, default=0,
                    help="refuse a selection with fewer span-carrying batches than this")
    ap.add_argument("--allow-stale-shards", action="store_true")
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--result", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--dry-run", action="store_true", help="stop before the tower loads")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stand-in", action="store_true",
                    help="harness smoke only: the one-block stand-in instead of the backbone")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args.compare)
    if args.p2_gate:
        if not args.p2_gate.is_file():
            print(f"{args.p2_gate} does not exist: P2 has not run, so the outcome run is "
                  "cancelled", flush=True)
            return 5
        rows = [json.loads(line) for line in args.p2_gate.read_text().splitlines()
                if line.strip()]
        ok, detail = p2_gate(rows)
        print(detail, flush=True)
        return 0 if ok else 5
    if args.p2:
        report = p2_screen(_tagged(args.p2, args.base), _tagged(args.p2, args.cand))
        print(json.dumps(report, sort_keys=True), flush=True)
        if args.p2_out:
            _append(args.p2_out, report)
        return {"pass": 0, "fail": 1}.get(report["verdict"], 2)
    missing = [k for k in ("code_root", "out", "backbone", "batch_tokens", "ledger", "result",
                           "tag") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"an arm needs {missing}")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
