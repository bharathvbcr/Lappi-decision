"""Reference values for crates/qd-train's trainer half, dumped from the Python it replaces.

A reference oracle only (house language policy): nothing here trains or ships. It imports the
repository's own ``qd_train.run_control`` and ``qd_train.ledger`` and torch's own AdamW, runs
them on fixed inputs, and writes what they produce to a JSON fixture that the Rust tests
compare against -- floats as ``float.hex`` so a comparison is of bits, not of renderings.

What it pins, and the Rust test that reads it:

* ``schedule`` -- ``LRSchedule.lr_at`` for every step of small schedules and sampled steps of
  large ones, including ``real_ft_run._control``'s ``max(1, steps // 20)`` / ``lr / 10``, and
  the same warmup with the ``--min-lr 0`` floor (``tests/schedule_oracle.rs``). Same-host
  claim: ``math.cos`` is this host's libm.
* ``floats`` -- ``repr`` and ``float.hex`` of edge and random floats; ``json`` -- canonical
  ``json.dumps`` of nested values; ``consumed`` -- ``ConsumedPrefix`` digests;
  ``loss_log`` -- ``LossLog.digest``; ``isoformat`` -- ``datetime.isoformat`` in UTC
  (``tests/pyjson_oracle.rs``).
* ``adamw`` -- ``torch.optim.AdamW(foreach=False)`` on CPU f32 tensors over several steps with a
  schedule, weight decay 0.01 and a clip coefficient applied in place, as
  ``MasterWeightAdamW`` + ``clip_grad_norm_`` drive it (``tests/adamw_oracle.rs``).
* ``clip`` -- ``torch.nn.utils.clip_grad_norm_``'s total norm for fixed gradients.
* ``ledger`` -- a ``LedgerRow`` line written by ``_canonical(row.to_json())`` with injected
  ``row_id``/``written_at``/``code_commit``/``host``, re-read by ``LedgerRow.from_json`` and
  chain-verified, for the Rust writer to reproduce byte for byte (``tests/ledger_oracle.rs``).

CPU only: no MPS or Metal tensor is created (the Mac GPU is reserved; ``torch.device("cpu")``
throughout).

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_trainer.py \\
        --out crates/qd-train/tests/fixtures/trainer-oracle.json

    # verify a ledger file the Rust writer produced, with Python's own reader
    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_trainer.py \\
        --verify-ledger /path/to/ledger/mac-ojas-x.jsonl

    # verify a Metal export the Rust writer produced, with the scorer's own reader
    # (real_ft_run._is_metal_export / _metal_export_seed / _read_metal_manifest and
    # ckpt_average.read_average: everything before a tower loads)
    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_trainer.py \\
        --verify-export /path/to/epoch-seed0-metal.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from qd_train.ledger import (
    REQUIRED_CONTROLS,
    REQUIRED_GATES,
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    _canonical,
)
from qd_train.run_control import ConsumedPrefix, LossLog, LossPoint, LRSchedule
from qd_train.tristate import NotRun, Ran

SEED = 20261001


def h(x: float) -> str:
    return float(x).hex()


def schedules() -> list[dict]:
    out = []
    configs = [
        # (peak, total, warmup, min): LRSchedule directly
        (1e-3, 10, 0, 0.0),
        (1e-3, 10, 3, 1e-4),
        (3e-4, 7, 1, 0.0),
        # Rung (d) as fable-rung-d-resize.md writes it literally. min_lr 1e-6 is one ulp below
        # real_ft_run's lr / 10 (1.0000000000000002e-06), which the (1e-5, 200) case below uses.
        (1e-5, 200, 10, 1e-6),
    ]
    # real_ft_run._control: warmup max(1, steps // 20), floor lr / 10 (real_ft_run.py:1833-1834).
    # (1e-5, 200) is rung (d)'s schedule: F's recipe at --max-steps 200, every step dumped.
    for lr, steps in [(1e-5, 2), (1e-5, 19), (1e-5, 20), (1e-5, 21), (1e-5, 100), (1e-5, 200),
                      (2e-5, 1505), (1e-5, 18197)]:
        configs.append((lr, steps, max(1, steps // 20), lr / 10))
    # The same rule under --min-lr 0 (v5 recipe.added[1]; LrSchedule::real_ft_with_floor):
    # warmup max(1, steps // 20), floor 0. 9683 is F's epoch, the length v5's base ran.
    for lr, steps in [(1e-5, 20), (1e-5, 200), (1e-5, 9683)]:
        configs.append((lr, steps, max(1, steps // 20), 0.0))
    for peak, total, warmup, floor in configs:
        s = LRSchedule(peak_lr=peak, total_steps=total, warmup_steps=warmup, min_lr=floor)
        steps = list(range(total)) if total <= 200 else sorted(
            {0, 1, warmup - 1, warmup, warmup + 1, total // 2, total - 2, total - 1}
            | {random.Random(total).randrange(total) for _ in range(40)}
        )
        out.append({
            "peak_lr": h(peak), "total_steps": total, "warmup_steps": warmup, "min_lr": h(floor),
            "steps": steps, "lr": [h(s.lr_at(i)) for i in steps],
        })
    return out


def floats() -> list[dict]:
    rng = random.Random(SEED)
    xs = [0.0, -0.0, 1.0, -1.0, 0.1, 1e-5, 1e-4, 0.0001, 0.00001, 1e15, 1e16, 1e17, 35403.0,
          32400.0, 1800.0, 0.5, 2.0 ** -1074, 2.0 ** -1022, 1.7976931348623157e308,
          0.8353729248046875, 591.820476162, 1e-05 / 10, 123456789012345678.0, 9.999999999999999e15,
          1e-7, 2.5e-8, 0.018589392289327065, 1.0000000000000002]
    for _ in range(300):
        bits = rng.getrandbits(64)
        x = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if math.isfinite(x):
            xs.append(x)
    for _ in range(300):
        xs.append(rng.uniform(-10, 10) * 10.0 ** rng.randint(-12, 20))
    return [
        {"bits": struct.unpack("<Q", struct.pack("<d", x))[0], "repr": repr(x), "hex": x.hex()}
        for x in xs
    ]


def to_wire(v):
    """A Python value as the fixture carries it: floats tagged so Rust rebuilds Json::Float."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return {"$int": v}
    if isinstance(v, float):
        return {"$f": v.hex()}
    if isinstance(v, list):
        return [to_wire(x) for x in v]
    if isinstance(v, dict):
        return {"$obj": [[k, to_wire(x)] for k, x in v.items()]}
    raise TypeError(type(v))


def json_cases() -> list[dict]:
    values = [
        {"b": 1e-5, "a": [1, None, True, False], "z": {"y": 0.1, "x": "s"}},
        {"lr": 1e-05, "span_weight": 1.0, "batch_tokens": 35403, "wall_clock_cap_s": 32400.0,
         "tag": "epoch", "deterministic": False},
        # The fraktur U is deliberate: a non-BMP character, which ensure_ascii writes as a
        # surrogate pair, is the case the Rust writer has to reproduce.
        {"esc": "tab\there \"q\" \\ \n\r\b\f \x01 \x7f é ✓ 𝔘", "big": 1e16, "neg": -2.5e-7},  # noqa: RUF001
        [0.0, -0.0, 5e-324, 1e300, "x"],
    ]
    out = []
    for v in values:
        out.append({
            "value": to_wire(v),
            "ascii": json.dumps(v, sort_keys=True, separators=(",", ":")),
            "utf8": json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        })
    return out


def consumed_cases() -> list[dict]:
    cases = [
        [], [[b"ab"]], [[b"a", b"b"]], [[b"a"], [b"b"]], [[b"", b"x" * 300, bytes(range(256))]],
    ]
    # One `trainer._fold`-shaped batch: index, bucket, int32 tokens [2,3], int32 lengths, then
    # the four optional arrays (two present, two absent), little-endian as numpy holds them.
    tokens = struct.pack("<6i", 5, 6, 7, 8, 9, 0)
    lengths = struct.pack("<2i", 3, 2)
    slot = struct.pack("<2b", 1, 3)
    target = struct.pack("<2q", 1, 0)
    cases.append([[
        (4).to_bytes(8, "big"), (2).to_bytes(8, "big"), tokens, lengths, slot, target, b"", b"",
    ]])
    out = []
    for folds in cases:
        p = ConsumedPrefix()
        for parts in folds:
            p.fold(*parts)
        out.append({
            "folds": [[part.hex() for part in parts] for parts in folds],
            "digest": p.hexdigest(),
        })
    return out


def loss_log_case() -> dict:
    points = [(0, 0, 0, 2.5), (1, 0, 1, 1.25), (2, 0, 3, 0.8353729248046875), (5, 1, 0, 1e-9)]
    log = LossLog(
        LossPoint(optimizer_step=s, epoch=e, batch_index=i, loss=x) for s, e, i, x in points
    )
    return {
        "points": [[s, e, i, h(x)] for s, e, i, x in points],
        "json": json.dumps(log.to_json(), sort_keys=True, separators=(",", ":")),
        "digest": log.digest(),
    }


def isoformat_cases() -> list[dict]:
    ts = [(1_790_000_000, 123_456), (1_790_000_000, 0), (951_782_400, 0), (1_759_336_929, 195_699),
          (4_102_444_799, 999_999)]
    out = []
    for secs, micros in ts:
        dt = datetime.fromtimestamp(secs, UTC).replace(microsecond=micros)
        out.append({"secs": secs, "micros": micros, "iso": dt.isoformat()})
    return out


def adamw_case() -> dict:
    import torch

    torch.manual_seed(SEED)
    cpu = torch.device("cpu")
    shapes = [(7,), (3, 5), (16,)]
    params = [torch.randn(s, dtype=torch.float32, device=cpu) for s in shapes]
    init = [p.detach().clone() for p in params]
    for p in params:
        p.requires_grad_(True)
    lr, beta2, steps = 1e-3, 0.999, 6
    sched = LRSchedule(
        peak_lr=lr, total_steps=steps, warmup_steps=max(1, steps // 20), min_lr=lr / 10
    )
    opt = torch.optim.AdamW(
        params, lr=lr, betas=(0.9, beta2), eps=1e-8, weight_decay=0.01, foreach=False
    )
    record = []
    for step in range(steps):
        grads = [torch.randn(s, dtype=torch.float32) * (0.5 + step) for s in shapes]
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        total = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False)
        coef = min(1.0 / (float(total) + 1e-6), 1.0)
        rate = sched.lr_at(step)
        for group in opt.param_groups:
            group["lr"] = rate
        opt.step()
        record.append({
            "lr": h(rate),
            "grads": [[h(x) for x in g.reshape(-1).tolist()] for g in grads],
            # After clip_grad_norm_'s in-place multiply: what AdamW actually read.
            "clipped_grads": [[h(x) for x in p.grad.reshape(-1).tolist()] for p in params],
            "torch_total_norm": h(float(total)),
            "coef": h(coef),
            "params": [[h(x) for x in p.detach().reshape(-1).tolist()] for p in params],
            "m": [[h(x) for x in opt.state[p]["exp_avg"].reshape(-1).tolist()] for p in params],
            "v": [[h(x) for x in opt.state[p]["exp_avg_sq"].reshape(-1).tolist()] for p in params],
        })
    return {
        "torch": torch.__version__,
        "shapes": [list(s) for s in shapes],
        "init": [[h(x) for x in p.reshape(-1).tolist()] for p in init],
        "beta2": h(beta2), "eps": h(1e-8), "weight_decay": h(0.01), "max_grad_norm": h(1.0),
        "steps": record,
    }


def ledger_case() -> dict:
    """A row shaped as crates/qd-train's ledger.rs writes one, written by Python's own code."""
    recipe = {
        "tool": "crates/qd-train-metal", "trainer": "qd-train-metal", "tag": "epoch",
        "device": "metal", "lr": 1e-05, "passes": 1,
        "batches": 3, "width": 1625, "span_weight": 1.0, "deterministic": True,
        "shard_hash": "d773b87666e1b042279271ab0f891246b7268d4ce0cad2c3e677bb415c147e1a",
        "backbone_snapshot": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c", "backbone_vocab": 248320,
        "backbone_params": 1881825088, "attn_implementation": "sdpa", "optimizer_recipe": "master",
        "provider": "ojas-qwen35 over tessl", "operands": "bf16", "optimizer_groups": "single",
        "wall_clock_cap_s": 21600.0, "batch_tokens": 35403, "no_memorise": True,
    }
    recipe_hash = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    protocol = Protocol(
        data_snapshot_hash="e9e55ba7b9975872cf1680e71a05f68d5e34fba38aed16987309ea26a7abf225",
        tokenizer_hash="7fbd94d096a01bca55f22c1852ed490a6c19759560577b1ffb68e207932308a9",
        backbone_commit="b1485b2fa6dfa1287294f269f5fb618e03d52d7c:vocab248320",
        recipe_hash=recipe_hash, seed=0,
    )
    points = [(0, 0, 0, 2.5), (1, 0, 1, 1.25), (2, 0, 2, 0.8353729248046875)]
    log = LossLog(
        LossPoint(optimizer_step=s, epoch=e, batch_index=i, loss=x) for s, e, i, x in points
    )
    metrics = {
        "train.termination": Ran(
            passed=True, value="steps_exhausted", detail="ft stopped after 3 optimizer step(s)"
        ),
        "train.optimizer_steps": Ran(passed=True, value=3),
        "train.micro_batches": Ran(passed=True, value=3),
        "train.supervised_tokens": Ran(passed=True, value=7, n=7, n_total=4875),
        "train.span_rows": Ran(passed=True, value=2),
        "train.padding_fraction": Ran(passed=True, value=1200 / 4875, n=1200, n_total=4875),
        "train.final_loss": Ran(passed=True, value=0.8353729248046875),
        "train.loss_log_digest": Ran(passed=True, value=log.digest()),
        "train.consumed_digest": Ran(
            passed=True, value="ab" * 32,
            detail="ConsumedPrefix over the 3 batch(es) this run consumed",
        ),
        "train.projected_usd_at_cap": Ran(
            passed=True, value=0.0,
            detail="local-metal: the Mac's own GPU, not billed, capped at 6.00 h -> $0.00 at the "
                   "cap -- no approval required"),
        "train.path": Ran(
            passed=True, value="ojas-qwen35 over tessl",
            detail="what this row ran on; compare rows only where this agrees or says why not",
        ),
        "train.head_init_digest": Ran(
            passed=True, value="01" * 32,
            detail="sha256 of span_head_init-seed0.safetensors, the span head's initial weights"),
        "train.eta_projected_s": Ran(
            passed=True, value=12000.0,
            detail="projected at optimizer step 10 from this process's elapsed time; the rule "
                   "stops a run projected past 20700.0 s (the 21600.0 s cap less 900.0 s)"),
        "deterministic_kernels": NotRun(reason="one run; repeat-run equality was not measured"),
    }
    env = Environment(
        torch="n/a: Rust trainer, no torch in the process",
        transformers_sha="n/a: Rust trainer, no transformers in the process",
        device="metal", host="oracle-host",
        fla_present=NotRun(reason="fla dry run not executed"),
        causal_conv1d_present=NotRun(reason="causal-conv1d dry run not executed"),
    )
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(Path(tmp) / "mac-ojas-oracle.jsonl")
        # Three rows: a run that finished; one cut by the cap; one stopped by the ETA rule, whose
        # projection (20725.0 s) is past the line and so is recorded as not passing.
        for k, (status, quick_reason, ended, projected) in enumerate([
            ("completed", "Metal/tessl trainer, 1 seed, oracle fixture", None, 12000.0),
            ("completed", "Metal/tessl trainer, 1 seed, oracle fixture", "wall_clock_cap", 12000.0),
            ("completed", "Metal/tessl trainer, 1 seed, oracle fixture", "eta_rule", 20725.0),
        ]):
            m = dict(metrics)
            if ended is not None:
                m["train.termination"] = Ran(passed=False, value=ended,
                                             detail="ft stopped after 3 optimizer step(s)")
                quick_reason = (
                    f"{quick_reason}; train.termination is '{ended}', not 'steps_exhausted': the "
                    "schedule did not run to its end, which rule 8 calls a truncated schedule"
                )
            m["train.eta_projected_s"] = Ran(
                passed=projected <= 20700.0, value=projected,
                detail=metrics["train.eta_projected_s"].detail)
            row = LedgerRow(
                row_id=f"00000000-0000-4000-8000-00000000000{k}",
                written_at="2026-10-01T18:02:09.195699+00:00",
                prev_row_hash=None, protocol=protocol, run_kind="ft", status=status, quick=True,
                quick_reason=quick_reason, code_commit="593b568" + "0" * 33, env=env, metrics=m,
                noul_rate=NotRun(reason="noul rate not computed by this run"),
                controls={c: NotRun(reason=f"control {c!r} was never evaluated by this run")
                          for c in REQUIRED_CONTROLS},
                gates={g: NotRun(reason=f"gate {g!r} was never evaluated by this run")
                       for g in REQUIRED_GATES},
                wall_clock_s=1234.5, wall_clock_source="recorder", cost_usd=0.0,
                notes="crates/qd-train-metal rung (d) oracle row", recipe=recipe,
            )
            ledger.append(row)
        ledger.verify_chain()
        for line in ledger.raw_lines():
            back = LedgerRow.from_json(json.loads(line))
            assert _canonical(back.to_json()).encode("utf-8") == line
            rows.append(line.decode("utf-8"))
    return {
        "recipe_hash": recipe_hash,
        "protocol_hash": protocol.hash(),
        "loss_points": [[s, e, i, h(x)] for s, e, i, x in points],
        "lines": rows,
    }


def clip_case() -> dict:
    import torch

    torch.manual_seed(SEED + 1)
    grads = [torch.randn(s, dtype=torch.float32) * 3 for s in [(5,), (2, 3)]]
    ps = [torch.zeros_like(g, requires_grad=True) for g in grads]
    for p, g in zip(ps, grads, strict=True):
        p.grad = g.clone()
    total = torch.nn.utils.clip_grad_norm_(ps, 1.0, foreach=False)
    return {
        "grads": [[h(x) for x in g.reshape(-1).tolist()] for g in grads],
        "torch_total_norm": h(float(total)),
        "clipped": [[h(x) for x in p.grad.reshape(-1).tolist()] for p in ps],
    }


def verify_ledger(path: Path) -> int:
    ledger = Ledger(path)
    ledger.verify_chain()
    for row in ledger.rows():
        recipe_hash = hashlib.sha256(
            json.dumps(row.recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if recipe_hash != row.protocol.recipe_hash:
            print(f"{row.row_id}: recipe_hash {row.protocol.recipe_hash} != hash of stored "
                  f"recipe {recipe_hash}")
            return 1
        print(f"{row.row_id}: ok run_kind={row.run_kind} status={row.status} quick={row.quick} "
              f"steps={row.metrics['train.optimizer_steps'].to_json().get('value')}")
    print(f"{path}: chain verified, {len(ledger.raw_lines())} row(s)")
    return 0


def verify_export(path: Path) -> int:
    """Run the scorer's own pre-load checks on a Rust-written Metal export: the route by name,
    the seed in the name, the manifest's fields and types, and the weights' sha256, tensor
    trees and count. The ft-row pairing needs a ledger row and a shard set, and is not run."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import real_ft_run as rft
    from ckpt_average import read_average

    if not rft._is_metal_export(path) or rft._is_average(path):
        print(f"{path.name}: the scorer does not route this name as a Metal export")
        return 1
    seed = rft._metal_export_seed(path)
    manifest = rft._read_metal_manifest(path)
    weights = read_average(manifest)
    n = sum(len(weights[t]) for t in ("tower", "span_head"))
    print(f"{path.name}: seed {seed}; manifest ok (from={manifest.body['from']!r}, "
          f"trainer={manifest.body['trainer']!r}, device={manifest.body['device']!r}, "
          f"ft_row_id={manifest.body['ft_row_id']}, manifest sha256 {manifest.sha256}); "
          f"weights ok ({n} tensors, vocab_size={weights['vocab_size']}, "
          f"span_weight={weights['span_weight']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify-ledger", type=Path)
    ap.add_argument("--verify-export", type=Path)
    args = ap.parse_args(argv)
    if args.verify_ledger is not None:
        return verify_ledger(args.verify_ledger)
    if args.verify_export is not None:
        return verify_export(args.verify_export)
    if args.out is None:
        ap.error("--out or --verify-ledger")
    body = {
        "generator": "tools/qd_train_oracle_trainer.py",
        "python": sys.version.split()[0],
        "schedule": schedules(),
        "floats": floats(),
        "json": json_cases(),
        "consumed": consumed_cases(),
        "loss_log": loss_log_case(),
        "isoformat": isoformat_cases(),
        "adamw": adamw_case(),
        "clip": clip_case(),
        "ledger": ledger_case(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
