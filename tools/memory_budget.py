"""Will this shard set train on this GPU? Answered here, before anything is rented.

``qd_train.memory`` holds the arithmetic; this is its caller. It reads a **real** shard
set's header for the bucket widths and a **real** checkpoint's safetensors header for the
parameter counts, so neither number is typed in by hand, and it prints the budget per
bucket per device.

It **exits 1** when a bucket in the shard set does not fit the device being budgeted. That
is the point: a report nobody acts on and a gate that refuses are different things, and the
failure this replaces -- ``torch.OutOfMemoryError`` partway into a paid run -- is loud
already. It is merely late.

Rule 5: every number here is **arithmetic**, not a measurement. There is no CUDA device on
this host, so nothing in this tool has been checked against an allocator. The parameter
counts are measured (from tensor shapes); the activation term is an enumerated lower bound
plus a stated allowance. ``StepFootprint.provenance`` says so on every breakdown it prints.

Usage::

    python tools/memory_budget.py --shards /tmp/qd-real/shards/train
    python tools/memory_budget.py --device-gb 40 --device-gb 80 --recompute none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from qd_train.memory import (
    ADAMW_BF16,
    ADAMW_FP32,
    ADAMW_KAHAN,
    ADAMW_MASTER,
    QWEN3_5_2B_TEXT,
    ActivationModel,
    CheckpointRefused,
    MemoryRefused,
    ModelSpec,
    StepFootprint,
    estimate_step,
    max_positions_that_fit,
    refuse_unless_it_fits,
    spec_from_checkpoint,
)

GB = 1000**3

#: The rental menu, as the coordinator posted it on 2026-09-20. Rates are per *instance*.
#: ``usable_gpus`` is 1 on every row and that is not a typo: nothing in this repository can
#: drive a second GPU. There is no DDP, no FSDP, no ``torchrun``, no ``init_process_group``
#: and no ``world_size``; ``train_ft`` takes no device argument at all, because the device
#: belongs to the caller's ``TrainStep``. A multi-GPU row therefore buys idle silicon.
MENU: tuple[tuple[str, float, int, float], ...] = (
    # (name, usd_per_hour, gpus_in_instance, gib_per_gpu)
    ("1x A10 24GB", 1.29, 1, 24.0),
    ("1x A100 40GB", 1.99, 1, 40.0),
    ("4x A6000 48GB", 4.36, 4, 48.0),
    ("2x H100 80GB", 8.38, 2, 80.0),
    ("8x A100 40GB", 15.92, 8, 40.0),
)

#: The recipes the trainer builds, keyed by the name ``real_ft_run.py --optimizer`` takes
#: (``fp32`` is ``--train-dtype fp32``), as (label, kwargs for `estimate_step`).
#:
#: Only layouts something builds. This table once defaulted its verdict to "bf16
#: weights+grads, fp32 AdamW states", which no optimizer here builds (``build_optimizer``
#: refuses ADAMW_FP32 over a bf16 tower). At 12 B/param it was under the 20 that v5's master
#: recipe measured, so a "fits" from it could OOM. ``tests/test_memory_budget.py`` builds each
#: row's optimizer to keep it that way.
RECIPES: dict[str, tuple[str, dict[str, object]]] = {
    "master": (
        "bf16 + fp32 master + fp32 AdamW (--optimizer master; v5's recipe)",
        {"optimizer": ADAMW_MASTER, "param_dtype": "bf16", "grad_dtype": "bf16",
         "activation_dtype": "bf16"},
    ),
    "kahan": (
        "bf16 + bf16 Kahan compensation + fp32 AdamW (--optimizer kahan)",
        {"optimizer": ADAMW_KAHAN, "param_dtype": "bf16", "grad_dtype": "bf16",
         "activation_dtype": "bf16"},
    ),
    "bf16": (
        "bf16 + bf16 AdamW states (--optimizer bf16; second moment freezes at step 384)",
        {"optimizer": ADAMW_BF16, "param_dtype": "bf16", "grad_dtype": "bf16",
         "activation_dtype": "bf16"},
    ),
    "fp32": (
        "fp32 tower + fp32 AdamW (--train-dtype fp32)",
        {"optimizer": ADAMW_FP32, "param_dtype": "fp32", "grad_dtype": "fp32",
         "activation_dtype": "fp32"},
    ),
}

#: The verdict's recipe when ``--recipe`` is not given: the one every v5 row trained with.
DEFAULT_RECIPE = "master"


def read_shard_header(shards: Path) -> dict:
    path = shards / "header.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist. This tool budgets a real shard set; point --shards at "
            "one, or pass --width to budget a width without one."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_for(model: ModelSpec, *, device_bytes: int, width: int, **kw: object) -> int:
    positions = max_positions_that_fit(model, device_bytes=device_bytes, width=width, **kw)
    return positions // width if positions else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, default=None, help="a shard set's train directory")
    ap.add_argument("--snapshot", type=Path, default=None, help="checkpoint snapshot directory")
    ap.add_argument(
        "--device-gb",
        type=float,
        action="append",
        default=None,
        help="a device budget in GB; repeatable. Default: every distinct size on the menu.",
    )
    ap.add_argument("--width", type=int, action="append", default=None, help="extra widths")
    ap.add_argument("--recompute", choices=("none", "full"), default="full")
    ap.add_argument("--attention", choices=("flash", "math"), default="flash")
    ap.add_argument(
        "--vocab",
        type=int,
        default=None,
        help="trainable vocabulary; default is the shard set's remapped size when --shards "
        "is given, otherwise the checkpoint's own",
    )
    ap.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default=DEFAULT_RECIPE,
        help=f"the recipe the pass/fail verdict budgets (default {DEFAULT_RECIPE}: v5's)",
    )
    ap.add_argument(
        "--require-fit",
        type=float,
        default=None,
        help="exit 1 unless every bucket fits a device of this many GB",
    )
    args = ap.parse_args()

    try:
        model = spec_from_checkpoint(args.snapshot) if args.snapshot else QWEN3_5_2B_TEXT
    except CheckpointRefused as exc:
        raise SystemExit(f"--snapshot refused: {exc}") from exc
    acts = ActivationModel(recompute=args.recompute, attention=args.attention)

    widths: list[int] = []
    vocab = args.vocab
    if args.shards is not None:
        header = read_shard_header(args.shards)
        widths.extend(int(b) for b in header["buckets"])
        if vocab is None:
            vocab = int(header["vocab_size"])
        print(
            f"shard set {args.shards}: {header['n_sequences']:,} sequences, "
            f"{header['total_tokens']:,} tokens, vocab {header['vocab_size']:,}, "
            f"max_seq_len {header['max_seq_len']:,}"
        )
    widths.extend(args.width or [])
    if not widths:
        widths = [2048, 4096, 8192, 16384, 34522]
    widths = sorted(set(widths))

    print(f"model: {model.name}")
    print(f"  measured parameters      {model.params_total:>15,}")
    print(f"  tied embedding           {model.params_embedding:>15,}  "
          f"({100 * model.params_embedding / model.params_total:.1f}%)")
    print(f"  trainable at vocab {vocab or model.vocab_size:>7,}  "
          f"{model.trainable_params(vocab_size=vocab):>15,}")
    print(f"  layers: {model.n_linear_attention_layers} linear_attention + "
          f"{model.n_full_attention_layers} full_attention")
    print(f"  activation policy: recompute={args.recompute}, attention={args.attention}\n")

    if args.device_gb:
        devices = [(f"{g:g} GB", g) for g in args.device_gb]
    else:
        seen: dict[float, str] = {}
        for name, _rate, _n, gib in MENU:
            seen.setdefault(gib, name)
        devices = [(f"{g:g} GB ({n})", g) for g, n in sorted(seen.items())]

    print("=" * 100)
    print("STATIC STATE (weights + gradients + optimizer), before a single activation")
    print("=" * 100)
    for label, kw in RECIPES.values():
        f = estimate_step(model, rows=1, width=1, activations=acts, vocab_size=vocab, **kw)
        print(f"  {f.static_bytes / GB:7.2f} GB  ({f.static_bytes / 1024**3:7.2f} GiB)  {label}")

    for dev_label, dev_gb in devices:
        dev_bytes = int(dev_gb * GB)
        print("\n" + "=" * 100)
        print(f"{dev_label}  --  ONE card")
        print("=" * 100)
        for label, kw in RECIPES.values():
            one = estimate_step(
                model, rows=1, width=1, activations=acts, vocab_size=vocab, **kw
            )
            if one.static_bytes >= dev_bytes:
                over = (one.static_bytes - dev_bytes) / GB
                print(f"  {label}: STATIC STATE ALONE IS {over:.2f} GB OVER THE CARD "
                      f"-- not a tuning problem")
                continue
            cells = []
            for w in widths:
                rows = _rows_for(
                    model, device_bytes=dev_bytes, width=w, activations=acts,
                    vocab_size=vocab, **kw,
                )
                cells.append(f"{w:,}:{rows}" if rows else f"{w:,}:NONE")
            print(f"  {label}")
            print(f"    rows that fit, by width -- {'  '.join(cells)}")

    # -- the verdict ---------------------------------------------------------
    label, kw = RECIPES[args.recipe]
    print(f"\nverdict recipe: {args.recipe}")
    print("\n" + "=" * 100)
    print(f"VERDICT under {label!r}")
    print("=" * 100)
    failures: list[str] = []
    if args.require_fit is None:
        print("  --require-fit not given, so no device was checked. Rule 5: this is NOT a "
              "pass, it is an unchecked budget. Pass --require-fit <GB> to make it a gate.")
    else:
        dev_bytes = int(args.require_fit * GB)
        for w in widths:
            f: StepFootprint = estimate_step(
                model, rows=1, width=w, activations=acts, vocab_size=vocab, **kw
            )
            try:
                refuse_unless_it_fits(f, device_bytes=dev_bytes, where=f"bucket width {w:,}")
            except MemoryRefused as exc:
                failures.append(str(exc).splitlines()[0])
            else:
                rows = _rows_for(
                    model, device_bytes=dev_bytes, width=w, activations=acts,
                    vocab_size=vocab, **kw,
                )
                print(f"  width {w:>7,}: FITS at {rows} row(s) per batch "
                      f"({f.total_bytes / GB:.2f} GB for one row)")
        if failures:
            print(f"\n  REFUSED: {len(failures)} of {len(widths)} bucket width(s) do not fit "
                  f"{args.require_fit:g} GB:")
            for line in failures:
                print(f"    - {line}")
        else:
            print(f"\n  every one of the {len(widths)} bucket width(s) fits "
                  f"{args.require_fit:g} GB")

    f = estimate_step(
        model, rows=1, width=max(widths), activations=acts, vocab_size=vocab, **kw
    )
    print("\n" + "=" * 100)
    print(f"BREAKDOWN at the widest bucket, {label!r}")
    print("=" * 100)
    print(f.breakdown())
    print(f"\n{f.provenance}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
