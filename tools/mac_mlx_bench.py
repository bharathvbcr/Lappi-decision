"""MLX on the Mac: 2B letter-logit parity against torch, 2B prefill rate, and 27B teacher scoring.

Every mode scores the way a letter-logit teacher or the runtime would: ONE prefill over the
prompt, the final hidden state at the last position, and the logits of the item's option-letter
tokens only -- no generation, no KV cache kept, no full-vocab slab over every position.

* ``parity``  -- 2B from the HF snapshot, against ``mac_torch_attrib.py --parity-out``: max abs
  difference of the log-softmax over each item's decode rows, and argmax agreement.
* ``prefill`` -- 2B scoring-path tok/s at fixed lengths of real (concatenated) prompt tokens.
* ``teacher`` -- a 27B MLX checkpoint on real val letter items: items/s, tok/s, peak memory,
  and projections for labelling N items (INFERRED: assumes the campaign's prompt-length
  distribution matches these items'). Prompts are re-tokenized with the 27B's own tokenizer
  from the text the 2B tokenizer decodes, and the identity of the two encodings is measured.

One ``throughput`` row per invocation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mac_bench_common as common
import mlx.core as mx
from mlx_lm import load

from qd_train.ledger import Ledger, RunRecorder
from qd_train.tristate import NotRun, Ran


def _text_model(model):  # type: ignore[no-untyped-def]
    lm = getattr(model, "language_model", model)
    return lm, lm.model


def _head(lm, h: mx.array) -> mx.array:  # type: ignore[no-untyped-def]
    if getattr(lm.args, "tie_word_embeddings", False) or not hasattr(lm, "lm_head"):
        return lm.model.embed_tokens.as_linear(h)
    return lm.lm_head(h)


def score(  # type: ignore[no-untyped-def]
    model, ids: list[int], letter_ids: list[int], *, fp32_head: bool = False
) -> mx.array:
    """Letter logits (fp32) at the last position of one prefill.

    ``fp32_head`` computes the head on fp32 hidden states against fp32 letter rows, so
    the logits are not rounded to bf16 before the comparison (parity); the teacher and
    prefill paths keep the model's own head dtype.
    """
    lm, inner = _text_model(model)
    h = inner(mx.array([ids]))[:, -1, :]
    if fp32_head:
        rows = lm.model.embed_tokens.weight[mx.array(letter_ids)].astype(mx.float32)
        z = (h.astype(mx.float32) @ rows.T)[0]
    else:
        z = _head(lm, h)[0, mx.array(letter_ids)].astype(mx.float32)
    mx.eval(z)
    return z


def _load(path: Path, dtype: str):  # type: ignore[no-untyped-def]
    """``mlx_lm.load``, optionally upcasting every floating tensor to fp32 BEFORE sanitize.

    mlx_lm's qwen3_5 ``sanitize`` stores ``1 + w`` for the RMSNorm weights of an HF-format
    snapshot, and does it in the checkpoint's dtype -- bf16 here -- while transformers
    computes ``1.0 + weight.float()`` at run time. Upcasting at ``mx.load`` makes the whole
    model fp32 including that shift, which is what the parity reference needs.
    """
    if dtype == "bfloat16":
        return load(str(path))
    original = mx.load

    def _upcast(file: str, *a: object, **k: object):  # type: ignore[no-untyped-def]
        out = original(file, *a, **k)
        return {
            name: (v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v)
            for name, v in out.items()
        }

    mx.load = _upcast
    try:
        return load(str(path))
    finally:
        mx.load = original


def _log_softmax(z: mx.array) -> mx.array:
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["parity", "prefill", "teacher"])
    ap.add_argument("--model", type=Path, required=True, help="local snapshot directory")
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--header", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--torch-parity", type=Path, help="parity: mac_torch_attrib --parity-out")
    ap.add_argument("--base-tokenizer", type=Path, help="teacher: the 2B tokenizer.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 8192])
    ap.add_argument("--project", type=int, nargs="+", default=[20_000, 50_000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = ap.parse_args(argv)
    if not 1 <= args.n <= 1000 or not 1 <= args.reps <= 50:
        raise SystemExit("--n must be in [1,1000] and --reps in [1,50]")
    items = [json.loads(x) for x in args.items.read_text().splitlines() if x.strip()]

    snap = args.model.name
    cfg = json.loads((args.model / "config.json").read_text())
    quant = cfg.get("quantization")
    vocab = cfg.get("text_config", cfg).get("vocab_size")
    recipe = {
        "tool": "tools/mac_mlx_bench.py", "mode": args.mode, "backend": "mlx-metal",
        "model_dir": args.model.parent.parent.name, "backbone_snapshot": snap,
        "backbone_vocab": vocab,
        "quantization": quant, "n": args.n, "lengths": args.lengths, "reps": args.reps,
        "scoring": "one prefill, last-position hidden, option-letter rows only",
        "dtype": args.dtype, "parity_head": "fp32" if args.mode == "parity" else "model",
    }
    rec = RunRecorder(
        Ledger(args.ledger),
        entry_point=Path(__file__),
        protocol=common.protocol(
            header=args.header, recipe=recipe
        ),
        run_kind="throughput",
        repo=common.REPO,
        env=common.environment(),
        wall_clock_s=None,
        cost=None,
        quick=True,
        quick_reason=common.QUICK_REASON,
        recipe=recipe,
        notes=f"MLX {args.mode} on {args.model.parent.parent.name} ({snap}); quantization={quant}. "
              "Peak memory is mx.get_peak_memory() (MLX allocator peak, includes weights).",
    )
    with rec:
        rec.metric("entry_tool_sha256", common.tool_digest(Path(__file__)))
        import mlx_lm

        t0 = time.perf_counter()
        model, tok = _load(args.model, args.dtype)
        load_s = time.perf_counter() - t0
        rec.metric("load_s", common.num(load_s, f"mlx_lm {mlx_lm.__version__} load(), weights "
                                                f"{mx.get_active_memory() / 1e9:.2f} GB active"))
        mx.reset_peak_memory()
        if args.mode == "parity":
            _parity(rec, model, items, args)
        elif args.mode == "prefill":
            _prefill(rec, model, items, args)
        else:
            _teacher(rec, model, tok, items, args)
        rec.metric("memory.mlx_peak_gb", common.num(
            mx.get_peak_memory() / 1e9, "mx.get_peak_memory() after the timed work"))
        rec.noul_rate = NotRun(reason="a benchmark decodes no verdicts against a gate")
    row = rec.row
    if row is None:
        raise SystemExit("no row written")
    for k, v in sorted(row.metrics.items()):
        j = v.to_json()
        print(f"  {k}: {j.get('value')}  {str(j.get('detail', ''))[:160]}")
    print(f"ledger row: {row.row_id}  {args.ledger}")
    return 0


def _parity(rec, model, items, args) -> None:  # type: ignore[no-untyped-def]
    if args.torch_parity is None:
        raise SystemExit("parity needs --torch-parity")
    ref = {r["row_id"]: r for r in map(json.loads, args.torch_parity.read_text().splitlines())}
    todo = [it for it in items if it["row_id"] in ref][: args.n]
    diffs, agree = [], 0
    for it in todo:
        z = score(model, it["prompt_ids"], it["letter_ids"], fp32_head=True)
        ls = _log_softmax(z).tolist()
        tl = ref[it["row_id"]]["letter_logsoftmax"]
        diffs.append(max(abs(a - b) for a, b in zip(ls, tl, strict=True)))
        agree += int(max(range(len(ls)), key=ls.__getitem__)
                     == max(range(len(tl)), key=tl.__getitem__))
    n = len(todo)
    if n == 0:
        raise SystemExit("no item overlaps the torch parity file")
    rec.metric("parity.max_abs_logsoftmax_diff", Ran(
        passed=True, value=round(max(diffs), 5), n=n, n_total=len(ref),
        detail=(f"max over {n} items of max |mlx - torch| log-softmax over each item's decode "
                f"rows; median per-item max {statistics.median(diffs):.5f}; mlx {args.dtype} "
                f"vs {args.torch_parity.name}; fp32 head on both sides")))
    rec.metric("parity.argmax_agreement", Ran(
        passed=agree == n, value=agree, n=agree, n_total=n,
        detail=f"{agree} of {n} items pick the same decode row in mlx and torch"))


def _prefill(rec, model, items, args) -> None:  # type: ignore[no-untyped-def]
    stream = [t for it in items for t in it["prompt_ids"]]
    letters = items[0]["letter_ids"]
    for length in args.lengths:
        if len(stream) < length:
            rec.metric(f"prefill.L{length}.tok_per_s",
                       NotRun(reason=f"only {len(stream)} real tokens available"))
            continue
        ids = stream[:length]
        score(model, ids, letters)
        score(model, ids, letters)
        ts = []
        for _ in range(args.reps):
            t = time.perf_counter()
            score(model, ids, letters)
            ts.append(time.perf_counter() - t)
        mean = statistics.mean(ts)
        rec.metric(f"prefill.L{length}.tok_per_s", common.num(
            length / mean,
            f"batch 1 x {length} real tokens (concatenated val prompts), scoring path, "
            f"{args.reps} reps after 2 warmup: mean {mean:.4f}s min {min(ts):.4f}s; peak so far "
            f"{mx.get_peak_memory() / 1e9:.2f} GB"))
        print(f"prefill L={length}: {length / mean:.0f} tok/s ({mean:.3f}s)", flush=True)


def _teacher(rec, model, tok, items, args) -> None:  # type: ignore[no-untyped-def]
    if args.base_tokenizer is None:
        raise SystemExit("teacher needs --base-tokenizer (the 2B tokenizer.json)")
    from tokenizers import Tokenizer

    base = Tokenizer.from_file(str(args.base_tokenizer))
    todo = items[: args.n]
    same_ids = 0
    prepared = []
    letter_single = True
    for it in todo:
        text = base.decode(it["prompt_ids"], skip_special_tokens=False)
        ids = tok.encode(text, add_special_tokens=False)
        same_ids += int(ids == it["prompt_ids"])
        lids = []
        for lid in it["letter_ids"]:
            s = base.decode([lid], skip_special_tokens=False)
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                letter_single = False
            lids.append(enc[0])
        prepared.append((ids, lids, it["gold_row"]))
    rec.metric("tokenizer.prompt_ids_identical", Ran(
        passed=same_ids == len(todo), value=same_ids, n=same_ids, n_total=len(todo),
        detail="items whose 27B re-tokenization equals the 2B ids exactly"))
    rec.metric("tokenizer.letters_single_token", Ran(
        passed=letter_single, value=letter_single,
        detail="every option letter (as the 2B decodes it) is one 27B token"))
    score(model, prepared[0][0], prepared[0][1])  # warmup
    ts, toks, correct = [], 0, 0
    t_all = time.perf_counter()
    for ids, lids, gold in prepared:
        t = time.perf_counter()
        z = score(model, ids, lids).tolist()
        ts.append(time.perf_counter() - t)
        toks += len(ids)
        correct += int(max(range(len(z)), key=z.__getitem__) == gold)
    wall = time.perf_counter() - t_all
    n = len(prepared)
    ips = n / wall
    rec.metric("teacher.items_per_s", common.num(
        ips, f"{n} items, batch 1, sequential, {wall:.1f}s wall; per-item median "
        f"{statistics.median(ts):.3f}s max {max(ts):.3f}s"))
    rec.metric("teacher.tok_per_s", common.num(toks / wall, f"{toks} prompt tokens in {wall:.1f}s"))
    rec.metric("teacher.mean_prompt_tokens", common.num(toks / n, "27B tokens per item"))
    rec.metric("teacher.letter_top1_on_val", Ran(
        passed=True, value=round(correct / n, 4), n=correct, n_total=n,
        detail="the teacher's zero-shot argmax over decode rows vs gold; informational, quick"))
    for target in args.project:
        rec.metric(f"teacher.projected_hours_{target}", common.num(
            target / ips / 3600,
            f"INFERRED: {target} items at the measured {ips:.3f} items/s, assuming the campaign's "
            "prompt-length distribution matches these val items; batch 1, no prefix reuse"))
    print(f"teacher: {ips:.3f} items/s, {toks / wall:.0f} tok/s, top1 {correct}/{n}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
