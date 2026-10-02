"""Letter-logit teacher scoring in torch: the bf16 half of the 4-bit vs bf16 teacher comparison.

``tools/mac_mlx_bench.py teacher`` scores real val letter items with the 27B 4-bit MLX build;
whether its labels match the bf16 model's was never measured (train-plan: "how far they are
from bf16 labels is unmeasured"). This tool scores the SAME items the same way -- one prefill,
the model's own output head at the last position, the option letters' logits only -- through
``transformers``' own loader, and writes each item's letter distribution in the parity format.
``mac_mlx_bench.py teacher --reference <this file>`` then records the divergence.

Why not ``qd_train.backbone.load_text_tower``: it builds the 2B's text tower exactly (320
tensors, a tied head, a training budget). The 27B has an untied ``lm_head`` and 64 layers;
``from_pretrained`` is the vendor's loader for the checkpoint as published, and
``logits_to_keep=1`` asks it for the last position only. Prompts are re-tokenized with the
teacher's tokenizer from the text the 2B tokenizer decodes, exactly as the MLX mode does
(``mac_bench_common.prepare_teacher_items``).

On a rented card the row is priced (``--instance``, ``--usd-per-hour``, ``--wall-clock-cap-s``)
through ``real_tokenizer_pipeline._pipeline_cost``, the pipeline's own pricing, so this tool
holds no second spelling of it. One ``throughput`` row per invocation, marked quick.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mac_bench_common as common

from qd_train.ledger import Ledger, RunRecorder
from qd_train.tristate import NotRun, Ran

DTYPES = ("bf16", "fp32")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True, help="local HF snapshot directory")
    ap.add_argument("--items", type=Path, required=True, help="val letter items (jsonl)")
    ap.add_argument("--header", type=Path, required=True, help="the items' shard header.json")
    ap.add_argument("--base-tokenizer", type=Path, required=True, help="the 2B tokenizer.json")
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--logits-out", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], required=True)
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--project", type=int, nargs="+", default=[20_000, 50_000])
    ap.add_argument("--instance", help="the priced machine, on a rented card")
    ap.add_argument("--usd-per-hour", type=float, help="the rate from the provider's price page")
    ap.add_argument("--wall-clock-cap-s", type=float, default=3600.0)
    args = ap.parse_args(argv)
    if not 1 <= args.n <= 1000:
        raise SystemExit("--n must be in [1,1000]")
    if not 60 <= args.wall_clock_cap_s <= 6 * 3600:
        raise SystemExit("--wall-clock-cap-s must be in [60, 21600]")

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForImageTextToText

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but torch sees no CUDA device")
    items = [json.loads(x) for x in args.items.read_text().splitlines() if x.strip()]
    todo = items[: args.n]
    if len(todo) < args.n:
        raise SystemExit(f"--n {args.n}, but {args.items} holds only {len(items)} items")

    cfg = json.loads((args.model / "config.json").read_text())
    vocab = cfg.get("text_config", cfg).get("vocab_size")
    snap = args.model.name
    recipe = {
        "tool": "tools/teacher_torch_score.py", "backend": f"torch-{args.device}",
        "model_dir": args.model.parent.parent.name, "backbone_snapshot": snap,
        "backbone_vocab": vocab, "n": args.n, "dtype": args.dtype,
        "scoring": "one prefill, logits_to_keep=1, model's own head, option-letter rows only",
    }
    env = common.environment(args.device)
    pipeline = importlib.import_module("real_tokenizer_pipeline")
    cost = pipeline._pipeline_cost(env.device, args)
    rec = RunRecorder(
        Ledger(args.ledger),
        entry_point=Path(__file__),
        protocol=common.protocol(header=args.header, recipe=recipe),
        run_kind="throughput",
        repo=common.REPO,
        env=env,
        wall_clock_s=None,
        cost=cost,
        quick=True,
        quick_reason=common.QUICK_REASON,
        recipe=recipe,
        notes=f"torch {args.dtype} teacher scoring on {args.device}, "
              f"{args.model.parent.parent.name} ({snap}); letters written to "
              f"{args.logits_out.name}.",
    )
    with rec:
        rec.metric("entry_tool_sha256", common.tool_digest(Path(__file__)))
        base = Tokenizer.from_file(str(args.base_tokenizer))
        teacher_tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        prepared, same_ids = common.prepare_teacher_items(
            todo,
            decode=lambda ids: base.decode(ids, skip_special_tokens=False),
            encode=lambda text: teacher_tok.encode(text, add_special_tokens=False).ids,
        )
        rec.metric("tokenizer.prompt_ids_identical", Ran(
            passed=same_ids == len(todo), value=same_ids, n=same_ids, n_total=len(todo),
            detail="items whose teacher re-tokenization equals the 2B ids exactly"))

        torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        t0 = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch_dtype, device_map=args.device
        ).eval()
        rec.metric("load_s", common.num(
            time.perf_counter() - t0,
            f"{type(model).__name__}.from_pretrained({args.dtype}, device_map={args.device})"))
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def score(ids: list[int], letter_ids: list[int]) -> list[float]:
            with torch.no_grad():
                x = torch.as_tensor([ids], device=args.device)
                logits = model(input_ids=x, logits_to_keep=1).logits
                z = logits[0, -1, torch.as_tensor(letter_ids, device=args.device)].float()
            return [float(v) for v in z.tolist()]

        score(prepared[0][1], prepared[0][2])  # warmup
        ts: list[float] = []
        toks = correct = 0
        t_all = time.perf_counter()
        with args.logits_out.open("w", encoding="utf-8") as out:
            for row_id, ids, lids, gold in prepared:
                t = time.perf_counter()
                z = score(ids, lids)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
                toks += len(ids)
                correct += int(max(range(len(z)), key=z.__getitem__) == gold)
                record = common.letter_record(row_id=row_id, n_tokens=len(ids), letter_logits=z)
                out.write(json.dumps(record) + "\n")
                out.flush()
        wall = time.perf_counter() - t_all
        n = len(prepared)
        ips = n / wall
        rec.metric("teacher.items_per_s", common.num(
            ips, f"{n} items, batch 1, sequential, {wall:.1f}s wall; per-item median "
            f"{statistics.median(ts):.3f}s max {max(ts):.3f}s"))
        rec.metric("teacher.tok_per_s", common.num(
            toks / wall, f"{toks} prompt tokens in {wall:.1f}s"))
        rec.metric("teacher.mean_prompt_tokens", common.num(toks / n, "teacher tokens per item"))
        rec.metric("teacher.letter_top1_on_val", Ran(
            passed=True, value=round(correct / n, 4), n=correct, n_total=n,
            detail="the teacher's zero-shot argmax over decode rows vs gold; informational, quick"))
        for target in args.project:
            rec.metric(f"teacher.projected_hours_{target}", common.num(
                target / ips / 3600,
                f"INFERRED: {target} items at the measured {ips:.3f} items/s, assuming the "
                "campaign's prompt-length distribution matches these val items; batch 1"))
        if args.device == "cuda":
            rec.metric("memory.cuda_peak_gb", common.num(
                torch.cuda.max_memory_allocated() / 1e9,
                "torch.cuda.max_memory_allocated() after scoring (weights included)"))
        else:
            rec.metric("memory.cuda_peak_gb", NotRun(reason=f"device {args.device}, not cuda"))
        rec.noul_rate = NotRun(reason="a benchmark decodes no verdicts against a gate")
        print(f"teacher: {ips:.3f} items/s, {toks / wall:.0f} tok/s, top1 {correct}/{n}",
              flush=True)
    row = rec.row
    if row is None:
        raise SystemExit("no row written")
    for k, v in sorted(row.metrics.items()):
        j = v.to_json()
        print(f"  {k}: {j.get('value')}  {str(j.get('detail', ''))[:160]}")
    print(f"ledger row: {row.row_id}  {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
