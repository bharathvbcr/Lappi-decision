"""A1 zero-shot baseline (rung 1): the BASE Qwen3.5-2B weights, decoded on a val shard set.

No training. The model is the untouched pretrained tower, remapped to the shard set's
vocabulary exactly as ``tools/real_ft_run.py`` remaps it before training, and every val row is
decoded by ``real_ft_run._decode`` -- the same function ``--score-val`` uses, which reads the
slot's own rendered letter rows at ``target_index`` the way ``crates/qd-runtime/src/answer.rs``
decodes a ``QueryKind::Letters`` query. Nothing here re-implements the decode; a second copy
is how two scorers come to disagree.

The val set, its labels and its letter ids are opened by ``real_ft_run.open_val_set`` from the
same corpus reconstruction ``real_ft_run.main`` performs, so a set this tool scores is a set
``--score-val`` would score, and it refuses the same things.

What this measures and what it does not:

* ``val_top1.choice`` / ``val_top1.score``: the pretrained letter channel, zero-shot. This is
  the rung-1 number.
* ``val_top1.span``: the span pointer head does not exist in the base model. ``QwenDecisionStep``
  builds it randomly initialised (seeded), so a span number here describes a random head over
  pretrained hidden states. It is recorded because the row would otherwise hide that the val
  set has span rows, and it is labelled as such in the row's notes.

One ``eval`` row per val set, ``quick=True``: a single seed and no training schedule at all.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch

# QD_TOOLS points the tool at another checkout's tools/ (e.g. a frozen worktree), so the
# code that decodes and labels is that checkout's, not whatever this tree holds right now.
TOOLS = Path(os.environ.get("QD_TOOLS", str(Path(__file__).resolve().parent))).resolve()
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# Imported by name through importlib, not by `import` statements, and deliberately:
# `RunRecorder`'s `code_that_ran` hashes the entry point's sibling imports FROM THE ENTRY
# POINT'S DIRECTORY. With QD_TOOLS set, the decode code that runs is TOOLS/real_ft_run.py
# while the sibling beside this file is another tree's copy, so a static import would put
# the wrong file's digest on the row. The decode closure that actually ran is recorded
# separately, as `decode_code_that_ran`, from TOOLS.
ft = importlib.import_module("real_ft_run")
pipeline = importlib.import_module("real_tokenizer_pipeline")
resolve_rev = importlib.import_module("repo_git").resolve_rev
n_gpus_for_device = importlib.import_module("run_cost").n_gpus_for_device


def _sync(device: str) -> None:
    """Wait for the device, so the decode's wall clock is the decode's and not its launch."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.split import split  # noqa: E402
from qd_train.ledger import Environment, Ledger, RunRecorder, what_ran_state  # noqa: E402
from qd_train.shards import ShardReader  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

QUICK_REASON = (
    "zero-shot baseline of the untrained base weights: one seed, no training schedule. "
    "Rule 8: quick, and excluded from every decision"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the pipeline's --out dir")
    parser.add_argument("--real-backbone", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, required=True)
    parser.add_argument("--rev", required=True)
    parser.add_argument("--commitpackft", type=Path, default=None)
    parser.add_argument(
        "--device", choices=["mps", "cpu", "cuda"], default="mps",
        help=(
            "cuda re-measures rung 1 on the rented box, where bf16 letter logits do not "
            "carry MPS's drift (GAP-TORCH-MPS-BF16-LETTER-LOGITS-DRIFT); it needs "
            "--instance and --usd-per-hour, because a rented hour is not free"
        ),
    )
    parser.add_argument("--instance", default=None, help="the priced machine, for cuda")
    parser.add_argument(
        "--usd-per-hour", type=float, default=None,
        help="the instance rate from the provider's price page at launch, for cuda",
    )
    parser.add_argument(
        "--wall-clock-cap-s", type=float, default=1800.0,
        help="the cap the cost estimate is priced from; the decode itself is minutes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--notes", default="", help="appended to the row's notes")
    parser.add_argument(
        "--dump-items", type=Path, default=None,
        help=(
            "write every val LETTER row as JSONL in the ORIGINAL tokenizer's ids (prompt ids "
            "through target_index, and each rendered option letter's id in decode-row order) "
            "and exit before any model loads. Input for the MLX parity and teacher timings"
        ),
    )
    args = parser.parse_args(argv)

    config = DataConfig()
    rev = resolve_rev(ft.REPO, args.rev)
    reader = ShardReader(
        args.out / "shards" / "train", config=config, repo_root=args.out, expect_rev=rev
    )
    commits, _, _ = pipeline.code_rows(
        commitpackft=args.commitpackft, max_pairs=args.max_pairs, rev=rev
    )
    spans, _ = pipeline.span_rows(max_rows=args.max_pairs, blank_line_runs=False, rev=rev)
    mixture = build_mixture(
        {"bigcode/commitpackft": list(commits), "rajpurkar/squad_v2": list(spans)},
        config=config,
    )
    split_report = split(dedupe(list(mixture.rows), config=config), config=config)
    train_labels, excluded = ft._labels(
        list(split_report.rows_by_split.get("train", ())), config=config
    )
    ft._inventory(reader, train_labels, excluded)
    letter_id = ft._letter_ids(reader, train_labels)
    val = ft.open_val_set(
        args.out, config=config, rev=rev,
        rows=list(split_report.rows_by_split.get("val", ())),
        train=reader, letter_id=letter_id,
    )
    print(f"val set: {len(val.reader)} sequences in {len(val.plan)} batches")
    if args.dump_items is not None:
        return _dump_items(val, reader, args.dump_items)

    from qd_train.backbone import QwenDecisionStep, load_text_tower, remap_text_tower
    from qd_train.memory import ADAMW_BF16

    width = max(int(b.tokens.shape[1]) for b in val.plan)
    t0 = time.monotonic()
    tower = load_text_tower(
        args.real_backbone,
        gradient_checkpointing=True,
        optimizer=ADAMW_BF16,
        attn_implementation=ft.DEFAULT_ATTN_IMPLEMENTATION,
        device=args.device,
        dtype="bf16",
        rows=max(int(b.tokens.shape[0]) for b in val.plan),
        width=width,
    )
    if reader.remap is None:
        raise SystemExit("the train shard set carries no remap; the val ids would misindex")
    tower = remap_text_tower(tower, reader.remap)
    tower.model.eval()
    # The optimizer is built and never stepped: `lr`/`total_steps` are required by the
    # constructor and determine nothing here, so they are not in the recipe.
    step = QwenDecisionStep(
        tower, seed=args.seed, lr=ft.REAL_BACKBONE_LR, total_steps=1, max_width=width
    )
    load_s = time.monotonic() - t0
    val_tokens = int(sum(int(b.lengths.sum()) for b in val.plan))

    recipe: dict[str, object] = {
        "tool": "tools/mac_zero_shot.py", "tag": "zero-shot-base", "device": args.device,
        "attn_implementation": tower.attn_implementation,
        "backbone_snapshot": tower.snapshot.name,
        "backbone_params": tower.footprint.trainable_params,
        "backbone_vocab": tower.vocab_size,
        "shard_hash": reader.header.shard_hash(),
        "val_shard_hash": val.reader.header.shard_hash(),
    }
    # Constructed before the decode and wrapping it, so a decode killed part-way still
    # writes a row; `measured()` then states the decode's own duration, not the load's.
    recorder = RunRecorder(
        Ledger(args.ledger),
        entry_point=Path(__file__),
        protocol=ft._protocol(reader=reader, seed=args.seed, recipe=recipe),
        run_kind="eval",
        repo=ft.REPO,
        env=Environment.detect(device=args.device),
        wall_clock_s=None,
        # A Mac run is priced at zero by leaving it out, as every earlier row did. A rented
        # device is priced from the arguments, and _cost refuses a missing instance or rate
        # by name rather than recording a GH200 hour as free.
        cost=(
            ft._cost(
                device="cuda", n_gpus=n_gpus_for_device("cuda"),
                usd_per_hour=args.usd_per_hour, instance=args.instance,
                cap_s=args.wall_clock_cap_s,
            )
            if args.device == "cuda"
            else None
        ),
        quick=True,
        quick_reason=QUICK_REASON,
        recipe=recipe,
        notes=(
            f"A1 zero-shot: BASE {tower.snapshot.name} weights (no training), remapped to "
            f"{tower.vocab_size} rows, decoded by real_ft_run._decode on {len(val.reader)} val "
            f"sequences of {args.out.name}. val_top1.span is a RANDOMLY INITIALISED span head "
            "(seeded) over pretrained hidden states, not a zero-shot pointer: the base model "
            f"has none. wall_clock_s is the decode only; tower load took {load_s:.1f}s. "
            + args.notes
        ).strip(),
    )
    with recorder:
        recorder.metric(
            "decode_code_that_ran",
            what_ran_state(ft.REPO / "python" / "qd_train", TOOLS / "real_ft_run.py"),
        )
        _sync(args.device)
        t1 = time.monotonic()
        scored = ft._decode(step, val.plan, val.labels_for, val.letter_id)
        _sync(args.device)
        decode_s = time.monotonic() - t1
        recorder.measured(decode_s)
        peak_gb = (
            torch.mps.driver_allocated_memory() / 1e9 if args.device == "mps" else float("nan")
        )
        for name, state in ft.score_states(scored, val.labels).items():
            recorder.metric(name, state)
        calibration, ece, degenerate = ft.calibration_states(scored)
        for name, state in calibration.items():
            recorder.metric(name, state)
        recorder.gate("ece", ece)
        recorder.control("degenerate_head", degenerate)
        n_rows = len(scored["verdicts"])  # type: ignore[arg-type]
        abstained = sum(
            1 for v in scored["verdicts"]  # type: ignore[union-attr]
            if str(v["runtime_verdict"]) == "abstain"
        )
        recorder.metric(
            "val_decoded_as_abstain",
            Ran(passed=True, value=abstained, n=abstained, n_total=n_rows,
                detail="val rows whose runtime verdict was the abstention, every kind"),
        )
        recorder.metric("val_shard_coverage", val.reader.coverage)
        recorder.metric(
            "throughput.decode_tok_per_s",
            Ran(passed=True, value=round(val_tokens / decode_s, 1),
                detail=(f"{val_tokens} real (unpadded) val tokens in {decode_s:.2f}s, full "
                        "forward + remapped lm_head over every position, no_grad, bf16")),
        )
        recorder.metric(
            "memory.mps_driver_allocated_gb",
            Ran(passed=True, value=round(peak_gb, 2),
                detail="torch.mps.driver_allocated_memory() after the decode")
            if args.device == "mps"
            else NotRun(reason="not an mps run"),
        )
        recorder.noul_rate = NotRun(
            reason="this corpus supervises no letter-channel abstention; see val_decoded_as_abstain"
        )
    row = recorder.row
    if row is None:
        raise SystemExit("the recorder exited without a row")
    for k, v in sorted(row.metrics.items()):
        if k.startswith(("val_top1", "throughput", "memory", "val_decoded")):
            print(f"  {k}: {json.dumps(v.to_json())[:300]}")
    print(f"decode {decode_s:.1f}s, load {load_s:.1f}s")
    print(f"ledger row: {row.row_id}  {args.ledger}")
    # Deterministic digest of the verdicts, so a rerun can be compared without the ledger.
    digest = hashlib.sha256(
        json.dumps([v["runtime_verdict"] for v in scored["verdicts"]]).encode()  # type: ignore[union-attr]
    ).hexdigest()
    print(f"verdict digest: {digest[:16]}")
    return 0


def _dump_items(val: ft.ValSet, train: ShardReader, path: Path) -> int:
    """Val letter rows in the base tokenizer's ids, decode rows in ``_decode``'s order."""
    remap = train.remap
    if remap is None:
        raise SystemExit("the train shard set carries no remap; ids cannot be mapped back")
    new_to_old = remap.new_to_old
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for b, batch in enumerate(val.plan):
            for r in range(batch.tokens.shape[0]):
                label = val.labels_for[b][r]
                if label.slot_kind == ft.SLOT_SPAN:
                    continue
                at = int(batch.target_index[r])  # type: ignore[index]
                ordered = [x for x in label.letters if x != ft.NOUL_LETTER] + [ft.NOUL_LETTER]
                ids = [int(new_to_old[int(t)]) for t in batch.tokens[r, : at + 1]]
                fh.write(json.dumps({
                    "row_id": label.row_id,
                    "kind": ft.KIND_NAMES[label.slot_kind],
                    "prompt_ids": ids,
                    "letters": ordered,
                    "letter_ids": [int(new_to_old[val.letter_id[x]]) for x in ordered],
                    "gold_row": ordered.index(label.gold_letter),
                }) + "\n")
                n += 1
    print(f"wrote {n} val letter items to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
