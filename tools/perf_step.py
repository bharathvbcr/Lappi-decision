#!/usr/bin/env python3
"""Profile and time the REAL fine-tuning step: real tower, real shard batches, real losses.

What runs is ``real_ft_run._real_step`` -- the production tower build, remap, footprint and
device budget -- driven exactly as ``trainer._train_loop`` drives it at ``grad_accum=1``:
``accumulate`` or ``accumulate_span`` per batch, then ``apply``. Nothing here is a toy.

``--code-root`` names the tree whose ``tools/real_ft_run.py`` (and, through it, whose
``python/``) is imported, so one script measures the unmodified checkout and a modified
overlay alike -- one tree per process, because both are the package ``qd_train``.

Each ``--configs`` entry toggles, in-process and on the one loaded tower:

* ``off:N``      N decoder layers (evenly spaced) run WITHOUT activation checkpointing;
                 ``off:0`` is the production setting, ``off:L`` is none at all;
* ``+fused``     the master optimizer's inner AdamW rebuilt with ``fused=True``;
* ``+compile``   ``nn.Module.compile(dynamic=True)`` on every MLP and RMSNorm module.

Weights drift across configs inside one process, so this is a THROUGHPUT and MEMORY
measurement only. Loss parity is ``--mode parity``, one arm per fresh process.

Every config writes one JSONL row: ``ran``, ``oom``, ``error`` or ``not_run`` (with why).
A config the budget skipped is ``not_run``; it never reads as a pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

GIB = 1024**3


def _append(path: Path, row: dict[str, Any]) -> None:
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _parse_config(text: str, n_layers: int) -> dict[str, Any]:
    parts = text.split("+")
    head = parts[0]
    # skip:N is the product rule (qd_train.backbone.uncheckpointed_layers, read from the
    # --code-root tree); off:N is this script's own even spread, for trees that predate it.
    rule = "skip" if head.startswith("skip:") else "off"
    if rule == "skip":
        head = "off:" + head[5:]
    if not head.startswith("off:"):
        raise SystemExit(f"config {text!r}: must start with off:N or skip:N")
    n_off = n_layers if head == "off:all" else int(head[4:])
    if not 0 <= n_off <= n_layers:
        raise SystemExit(f"config {text!r}: off:{n_off} outside 0..{n_layers}")
    flags = set(parts[1:])
    # nNN: this config's timed-step count, overriding --steps.
    counts = {f for f in flags if f.startswith("n") and f[1:].isdigit()}
    steps = int(next(iter(counts))[1:]) if counts else None
    flags -= counts
    unknown = flags - {"fused", "compile", "profile", "syncdebug", "memhist", "det", "nomask"}
    if unknown or len(counts) > 1:
        raise SystemExit(f"config {text!r}: unknown toggles {sorted(unknown)} or two nNN")
    return {"name": text, "n_off": n_off, "flags": flags, "steps": steps, "rule": rule}


def _off_layers(n_off: int, n_layers: int) -> list[int]:
    if n_off == 0:
        return []
    if n_off == n_layers:
        return list(range(n_layers))
    # Evenly spaced, so the off set mixes the hybrid's linear and full-attention layers.
    return sorted({math.floor((i + 0.5) * n_layers / n_off) for i in range(n_off)})


def _select_batches(reader: Any, *, batch_tokens: int, seed: int, width_min: int,
                    width_max: int, n: int, scan_limit: int) -> tuple[list[Any], dict[str, Any]]:
    picked: list[Any] = []
    seen = 0
    t0 = time.perf_counter()
    for batch in reader.batches(batch_tokens=batch_tokens, seed=seed, epoch=0):
        seen += 1
        w = int(batch.tokens.shape[1])
        if width_min <= w <= width_max:
            picked.append(batch)
            if len(picked) >= n:
                break
        if seen >= scan_limit:
            break
    wall = time.perf_counter() - t0
    return picked, {
        "scanned": seen,
        "picked": len(picked),
        "scan_wall_s": round(wall, 3),
        "per_batch_build_ms": round(1000 * wall / max(seen, 1), 3),
        "widths": [int(b.tokens.shape[1]) for b in picked],
        "rows": [int(b.tokens.shape[0]) for b in picked],
        "positions": [int(b.tokens.size) for b in picked],
        "real_tokens": [int(b.lengths.sum()) for b in picked],
    }


def _set_checkpointing(model: Any, off: list[int]) -> None:
    off_set = set(off)
    for i, layer in enumerate(model.layers):
        layer.gradient_checkpointing = i not in off_set


def _compile_targets(model: Any) -> list[Any]:
    out = []
    for _name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls.endswith("MLP") or cls.endswith("RMSNorm") or cls.endswith("RMSNormGated"):
            out.append(mod)
    return out


def _set_compile(model: Any, on: bool) -> int:
    targets = _compile_targets(model)
    for mod in targets:
        if on:
            mod.compile(dynamic=True)
        else:
            mod._compiled_call_impl = None
    return len(targets)


def _set_fused(step: Any, torch: Any, fused: bool) -> str:
    """Rebuild the master optimizer's inner AdamW only when the wanted kind differs.

    The production optimizer object is left untouched until a ``+fused`` config asks.
    """
    opt = step.optimizer
    inner = getattr(opt, "_inner", None)
    if inner is None:
        raise RuntimeError(f"{type(opt).__name__} has no _inner; not the master optimizer")
    is_fused = bool(inner.param_groups[0].get("fused"))
    if is_fused == fused:
        return "fused" if fused else "as-built"
    groups = []
    for g in inner.param_groups:
        extra = {k: v for k, v in g.items() if k not in ("params", "foreach", "fused")}
        groups.append({**extra, "params": g["params"]})
    # Throughput only: the moments restart. Never used for a parity arm.
    new = torch.optim.AdamW(groups, fused=True) if fused else torch.optim.AdamW(groups)
    opt._inner = new
    del inner
    gc.collect()
    return "fused" if fused else "rebuilt-default"


def _set_nomask(step: Any, on: bool) -> str:
    """The training forward without the padding mask: the product switch
    (`QwenDecisionStep.train_attention_mask`, `real_ft_run --train-attention-mask none`), so a
    benchmark measures exactly what a run would -- including flash as the only SDPA backend.

    Shard batches are right-padded, so under causal attention no real position can see a pad;
    without an explicit mask SDPA takes flash (is_causal) instead of mem-efficient. Different
    kernels, so a numerics change (Tier B) -- measured here for speed only.
    """
    step.train_attention_mask = "none" if on else "padding"
    return "none (is_causal, flash only)" if on else "padding-mask"


def _one_step(step: Any, batch: Any, sup: Any, lr: float, torch: Any, sync: bool) -> dict:
    t0 = time.perf_counter()
    loss = step.accumulate(batch, sup) if sup.span is None else step.accumulate_span(batch, sup)
    t1 = time.perf_counter()
    step.apply(lr=lr)
    if sync:
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    return {"loss": float(loss), "acc_s": t1 - t0, "apply_s": t2 - t1, "step_s": t2 - t0}


def _profile_table(prof: Any, top: int) -> tuple[str, dict[str, Any]]:
    events = prof.key_averages()
    table = events.table(sort_by="self_device_time_total", row_limit=top, max_name_column_width=90)
    kernels: list[tuple[float, str, int]] = []
    ops: list[tuple[float, str, int]] = []
    for evt in events:
        dev = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if dev <= 0:
            continue
        is_kernel = str(getattr(evt, "device_type", "")).endswith("CUDA")
        (kernels if is_kernel else ops).append((dev, evt.key, int(evt.count)))
    kernels.sort(reverse=True)
    ops.sort(reverse=True)
    kernel_us = sum(d for d, _k, _c in kernels)
    # The GDN path, named from the kernels that actually ran: fla's Triton kernels carry
    # chunk/delta/solve_tril/recompute_w_u in their names; the torch fallback runs none.
    gdn_marks = ("gated_delta", "chunk", "solve_tril", "recompute_w_u", "delta_rule",
                 "causal_conv1d", "l2norm", "fused_recurrent")
    gdn = [(d, k, c) for d, k, c in kernels if any(m in k.lower() for m in gdn_marks)]
    return table, {
        "kernel_self_us_total": kernel_us,
        "top_kernels": [{"us": round(d, 1), "name": k[:160], "count": c}
                        for d, k, c in kernels[:top]],
        "top_ops": [{"us": round(d, 1), "name": k[:160], "count": c} for d, k, c in ops[:top]],
        "gdn_kernel_us": round(sum(d for d, _k, _c in gdn), 1),
        "gdn_kernels": [{"us": round(d, 1), "name": k[:160], "count": c} for d, k, c in gdn],
    }


def run_profile(args: argparse.Namespace) -> int:
    t_start = time.perf_counter()
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root / "tools"))
    import numpy as np  # noqa: F401
    import real_ft_run
    import torch

    from qd_data.config import DataConfig
    from qd_train.shards import ShardReader
    from qd_train.trainer import ft_supervision

    base = {
        "shape": args.shape,
        "code_root": str(code_root),
        "real_ft_run": str(Path(real_ft_run.__file__).resolve()),
        "torch": torch.__version__,
        "host": os.uname().nodename,
    }
    try:
        import transformers

        base["transformers"] = transformers.__version__
    except ImportError as exc:  # pragma: no cover - the real branch needs it
        base["transformers"] = f"missing: {exc}"
    try:
        import fla  # type: ignore[import-not-found]

        base["fla"] = getattr(fla, "__version__", "?")
    except ImportError as exc:
        base["fla"] = f"missing: {exc}"

    config = DataConfig()
    if args.allow_stale_shards:
        # Deliberate and recorded: J2's probe-8k set was written by older qd_data code. Its
        # tokens are real; a throughput or same-batches parity measurement does not read
        # its labels as labels. The production reader is never patched.
        import functools

        import qd_train.shards as shards_mod

        shards_mod.assert_shard_trainable = functools.partial(
            shards_mod.assert_shard_trainable, allow_stale_code=True
        )
        base["allow_stale_shards"] = True
    reader = ShardReader(args.out / "shards" / "train", config=config, repo_root=args.out)
    batches, sel = _select_batches(
        reader, batch_tokens=args.batch_tokens, seed=config.seed, width_min=args.width_min,
        width_max=args.width_max, n=args.n_batches, scan_limit=args.scan_limit,
    )
    if not batches:
        _append(args.results, {**base, "status": "error", "reason": "no batch in width range",
                               "selection": sel})
        return 2
    sups = [ft_supervision(b) for b in batches]
    t0 = time.perf_counter()
    for b in batches:
        ft_supervision(b)
    sel["supervise_ms_per_batch"] = round(1000 * (time.perf_counter() - t0) / len(batches), 3)
    sel["span_batches"] = sum(1 for s in sups if s.span is not None)
    base["selection"] = sel
    print(json.dumps({"selection": {k: sel[k] for k in ("scanned", "picked", "scan_wall_s",
                                                        "per_batch_build_ms", "span_batches")},
                      "widths": sorted(set(sel["widths"]))}), flush=True)

    import inspect

    import transformers.modeling_layers as ml  # type: ignore[import-not-found]

    src = inspect.getsource(ml.GradientCheckpointingLayer.__call__)
    base["gc_layer_checks_flag"] = "self.gradient_checkpointing" in src
    if not base["gc_layer_checks_flag"]:
        _append(args.results, {**base, "status": "error",
                               "reason": "GradientCheckpointingLayer.__call__ no longer reads "
                                         "self.gradient_checkpointing; per-layer toggles invalid"})
        return 2

    if args.dry_run:
        # CPU: the profiler's key on a tiny op, so a GPU slot is not spent learning it moved.
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            (torch.ones(64, 64) @ torch.ones(64, 64)).sum()
        _profile_table(prof, 5)
        _append(args.results, {**base, "status": "dry_run_ok",
                               "wall_s": round(time.perf_counter() - t_start, 1)})
        print("dry run ok", flush=True)
        if not args.dry_run_step:
            return 0

    device = args.device
    spec = real_ft_run.optimizer_spec("bf16", "master")
    width = max(int(b.tokens.shape[1]) for b in batches)
    t_load = time.perf_counter()
    step, tower, budget = real_ft_run._real_step(
        backbone=args.backbone, reader=reader, plan=batches, device=device, dtype="bf16",
        spec=spec, attn_implementation="sdpa", seed=args.seed, lr=1e-5,
        total_steps=10_000, span_weight=1.0, width=width,
    )
    base["load_s"] = round(time.perf_counter() - t_load, 1)
    base["budget"] = budget.to_json() if hasattr(budget, "to_json") else str(budget)
    base["attn_implementation"] = tower.attn_implementation
    model = tower.model
    n_layers = len(model.layers)
    base["n_layers"] = n_layers
    base["layer_types"] = list(getattr(model.config, "layer_types", []) or [])
    layer_types = base["layer_types"]
    import qd_train.backbone as qb

    skip_rule = getattr(qb, "uncheckpointed_layers", None)
    if skip_rule is None and any(c.startswith("skip:") for c in args.configs):
        raise SystemExit(f"{code_root} has no uncheckpointed_layers; skip:N needs that tree")
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq  # type: ignore

        # What transformers' use_kernel_func_from_hub_with_fallback bound at import: the
        # closure's `implementation` is fla's kernel or the torch reference fallback.
        bound: dict[str, str] = {}
        for fn_name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
            fn = getattr(mq, fn_name)
            cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=False))
            impl = cells.get("implementation")
            impl_v = impl.cell_contents if impl is not None else None
            bound[fn_name] = (
                f"{getattr(impl_v, '__module__', '?')}.{getattr(impl_v, '__qualname__', '?')}"
                if impl_v is not None else "unresolved"
            )
        base["gdn_bound"] = bound
    except Exception as exc:
        base["gdn_bound"] = f"unreadable: {type(exc).__name__}: {exc}"
    shown = ("load_s", "n_layers", "attn_implementation", "gdn_bound")
    print(json.dumps({k: base[k] for k in shown}),
          flush=True)

    on_cuda = device == "cuda"
    configs = [_parse_config(c, n_layers) for c in args.configs]
    oom_at: int | None = None
    est_s: dict[str, float] = {}
    # --rounds R interleaves: every config once per round, R rounds, so drift on the device
    # (clocks, thermals, another tenant) lands on every arm alike; --summarize takes min-of-N.
    schedule = [(rnd, cfg) for rnd in range(args.rounds) for cfg in configs]
    for rnd, cfg in schedule:
        name = cfg["name"]
        row: dict[str, Any] = {**base, "config": name, "n_off": cfg["n_off"],
                               "flags": sorted(cfg["flags"]), "round": rnd}
        n_timed = cfg["steps"] or args.steps
        elapsed = time.perf_counter() - t_start
        guess = est_s.get("per_step", 2.0) * (args.warmup + n_timed) + (
            150 if "compile" in cfg["flags"] else 10)
        if elapsed + guess > args.budget_s:
            _append(args.results, {**row, "status": "not_run",
                                   "reason": f"budget: {elapsed:.0f}s elapsed + ~{guess:.0f}s "
                                             f"> {args.budget_s}s"})
            continue
        if oom_at is not None and cfg["n_off"] >= oom_at and "compile" not in cfg["flags"]:
            _append(args.results, {**row, "status": "not_run",
                                   "reason": f"off:{oom_at} already OOMed at this shape"})
            continue
        off = (
            list(skip_rule(cfg["n_off"], layer_types)) if cfg["rule"] == "skip"
            else _off_layers(cfg["n_off"], n_layers)
        )
        row["off_layers"] = off
        try:
            _set_checkpointing(model, off)
            # real_ft_run --deterministic is exactly this call, with CUBLAS_WORKSPACE_CONFIG
            # set before cuBLAS initialises (the session script exports it for the process).
            torch.use_deterministic_algorithms("det" in cfg["flags"])
            row["deterministic"] = torch.are_deterministic_algorithms_enabled()
            row["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            row["optimizer"] = _set_fused(step, torch, "fused" in cfg["flags"])
            row["attention_mask"] = _set_nomask(step, "nomask" in cfg["flags"])
            row["compiled_modules"] = _set_compile(model, "compile" in cfg["flags"])
            if "compile" in cfg["flags"]:
                torch._dynamo.utils.counters.clear()
                # Every module compiled through nn.Module.compile shares the code object
                # Module._call_impl, so MLP, RMSNorm and RMSNormGated entries all count
                # against ONE recompile limit (8 by default); past it dynamo runs eager.
                for knob in ("recompile_limit", "cache_size_limit"):
                    if hasattr(torch._dynamo.config, knob):
                        setattr(torch._dynamo.config, knob, 64)
                row["recompile_limit"] = 64
            step.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if on_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            order = [i % len(batches) for i in range(args.warmup + n_timed)]
            t_c0 = time.perf_counter()
            for i in order[: args.warmup]:
                _one_step(step, batches[i], sups[i], 1e-5, torch, on_cuda)
            row["warmup_s"] = round(time.perf_counter() - t_c0, 2)
            # --overlap: no synchronize after each optimizer step, exactly as trainer._train_loop
            # runs (its only per-step sync is float(loss)), so the next batch's host work
            # overlaps the optimizer's kernels; one sync closes the window. Without it, every
            # step is timed in isolation, which session 1 did and which reads ~13% slower than
            # the production loop at shape A (0.95 vs J6(b)'s logged 0.83 s/step).
            if on_cuda:
                torch.cuda.synchronize()
            t_loop = time.perf_counter()
            timed = [_one_step(step, batches[i], sups[i], 1e-5, torch,
                               on_cuda and not args.overlap)
                     for i in order[args.warmup:]]
            if on_cuda:
                torch.cuda.synchronize()
            loop_wall = time.perf_counter() - t_loop
            pos = sum(int(batches[i].tokens.size) for i in order[args.warmup:])
            real = sum(int(batches[i].lengths.sum()) for i in order[args.warmup:])
            wall = loop_wall if args.overlap else sum(t["step_s"] for t in timed)
            row["timing"] = "overlapped (one sync per window)" if args.overlap else "per-step sync"
            row["step_s_window_mean"] = round(loop_wall / len(timed), 4)
            est_s["per_step"] = max(est_s.get("per_step", 0.0), wall / len(timed))
            row.update({
                "status": "ran",
                "steps": len(timed),
                "wall_s": round(wall, 3),
                "pos_per_s": round(pos / wall, 1),
                "real_tok_per_s": round(real / wall, 1),
                "step_s_median": round(statistics.median(t["step_s"] for t in timed), 4),
                "step_s_min": round(min(t["step_s"] for t in timed), 4),
                "acc_s_median": round(statistics.median(t["acc_s"] for t in timed), 4),
                "apply_s_median": round(statistics.median(t["apply_s"] for t in timed), 4),
                "losses": [round(t["loss"], 5) for t in timed],
            })
            if on_cuda:
                row["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
                row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 2)
            if "compile" in cfg["flags"]:
                c = torch._dynamo.utils.counters
                row["dynamo"] = {
                    str(cat): {str(k)[:120]: v for k, v in list(vals.items())[:20]}
                    for cat, vals in c.items()
                }
            if "syncdebug" in cfg["flags"] and on_cuda:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    torch.cuda.set_sync_debug_mode("warn")
                    try:
                        i = order[-1]
                        _one_step(step, batches[i], sups[i], 1e-5, torch, False)
                        torch.cuda.synchronize()
                    finally:
                        torch.cuda.set_sync_debug_mode(0)
                sites: dict[str, int] = {}
                for w in caught:
                    key = f"{Path(w.filename).name}:{w.lineno}"
                    sites[key] = sites.get(key, 0) + 1
                row["host_syncs_one_step"] = {"count": len(caught), "sites": sites}
            if "profile" in cfg["flags"]:
                from torch.profiler import ProfilerActivity, profile

                acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_cuda else [])
                prof_steps = order[-args.profile_steps:]
                if on_cuda:
                    torch.cuda.synchronize()
                t_p0 = time.perf_counter()
                with profile(activities=acts, record_shapes=False) as prof:
                    for i in prof_steps:
                        _one_step(step, batches[i], sups[i], 1e-5, torch, on_cuda)
                p_wall = time.perf_counter() - t_p0
                table, summary = _profile_table(prof, args.top)
                summary["wall_us"] = p_wall * 1e6
                summary["gpu_busy_frac"] = (
                    summary["kernel_self_us_total"] / summary["wall_us"] if on_cuda else None
                )
                summary["steps"] = len(prof_steps)
                row["profile"] = summary
                tpath = args.results.with_name(
                    f"{args.results.stem}-{args.shape}-{name.replace(':', '').replace('+', '_')}"
                    "-top.txt")
                tpath.write_text(table, encoding="utf-8")
                row["profile_table"] = str(tpath)
                del prof
            if "memhist" in cfg["flags"] and on_cuda:
                torch.cuda.memory._record_memory_history(max_entries=200_000)
                i = order[-1]
                _one_step(step, batches[i], sups[i], 1e-5, torch, True)
                snap = args.results.with_name(f"{args.results.stem}-{args.shape}-memsnap.pickle")
                torch.cuda.memory._dump_snapshot(str(snap))
                torch.cuda.memory._record_memory_history(enabled=None)
                row["memory_snapshot"] = str(snap)
        except torch.cuda.OutOfMemoryError as exc:
            row.update({"status": "oom", "reason": str(exc).splitlines()[0][:400]})
            if not cfg["flags"] & {"compile"}:
                oom_at = cfg["n_off"] if oom_at is None else min(oom_at, cfg["n_off"])
            if on_cuda:
                row["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
        except Exception as exc:
            row.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:600],
                        "trace": traceback.format_exc()[-2000:]})
        finally:
            step.optimizer.zero_grad(set_to_none=True)
            _set_compile(model, False)
            _set_nomask(step, False)
            gc.collect()
            if on_cuda:
                torch.cuda.empty_cache()
        row["elapsed_s"] = round(time.perf_counter() - t_start, 1)
        _append(args.results, row)
        print(json.dumps({k: row.get(k) for k in ("shape", "config", "status", "pos_per_s",
                                                  "step_s_median", "acc_s_median",
                                                  "apply_s_median", "peak_alloc_gib",
                                                  "reason")}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--code-root", type=Path)
    ap.add_argument("--out", type=Path, help="shard set root (has shards/train)")
    ap.add_argument("--backbone", type=Path)
    ap.add_argument("--shape")
    ap.add_argument("--batch-tokens", type=int)
    ap.add_argument("--width-min", type=int, default=0)
    ap.add_argument("--width-max", type=int)
    ap.add_argument("--n-batches", type=int, default=24)
    ap.add_argument("--scan-limit", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--profile-steps", type=int, default=3)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", nargs="+", default=["off:0+profile+syncdebug"])
    ap.add_argument("--results", type=Path)
    ap.add_argument("--budget-s", type=float, default=540.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overlap", action="store_true",
                    help="time like the production loop: no sync after each step")
    ap.add_argument("--rounds", type=int, default=1,
                    help="interleave: every config once per round, this many rounds")
    ap.add_argument("--summarize", type=Path, help="min-of-N over a results file; no GPU")
    ap.add_argument("--allow-stale-shards", action="store_true",
                    help="read a shard set whose qd_data code moved (recorded on every row)")
    ap.add_argument("--dry-run-step", action="store_true",
                    help="with --dry-run, also build the tower and run the configs")
    args = ap.parse_args(argv)
    if args.summarize is not None:
        return summarize(args.summarize)
    missing = [k for k in ("code_root", "out", "backbone", "shape", "batch_tokens", "width_max",
                           "results") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"a profile run needs {missing}")
    return run_profile(args)


def summarize(path: Path) -> int:
    """Min-of-N per (shape, config) over the rounds that RAN; a config with a round that did
    not run says so beside its number rather than being summarised over fewer rounds."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if "config" in r:
            groups.setdefault((r["shape"], r["config"]), []).append(r)
    for (shape, config), rs in sorted(groups.items()):
        ran = [r for r in rs if r.get("status") == "ran"]
        out = {
            "shape": shape, "config": config, "rounds": len(rs), "rounds_ran": len(ran),
            "not_ran": sorted({f"{r.get('status')}: {r.get('reason', '')}"[:120]
                               for r in rs if r.get("status") != "ran"}),
        }
        if ran:
            best = min(ran, key=lambda r: r.get("step_s_window_mean", r["step_s_median"]))
            out.update({
                "min_step_s_window_mean": best.get("step_s_window_mean"),
                "min_step_s_median": best["step_s_median"],
                "timing": sorted({r.get("timing", "per-step sync") for r in ran}),
                "max_pos_per_s": max(r["pos_per_s"] for r in ran),
                "pos_per_s_by_round": [r["pos_per_s"] for r in ran],
                "peak_alloc_gib_max": max(r.get("peak_alloc_gib", 0.0) for r in ran),
                "deterministic": sorted({bool(r.get("deterministic")) for r in ran}),
            })
        print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
