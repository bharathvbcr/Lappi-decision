"""S2's remap parity gate, run once against the REAL Qwen3.5-2B text tower.

``qd_train.remap.apply_remap_to_model`` slices the vocabulary out of the tied embedding. Until
this ran it had been exercised only against stub models (GAP-S2-NO-REAL-CHECKPOINT-PARITY),
and what a stub cannot establish is the failure that record names: that the real class keeps
a second reference to the 248,320-row embedding, so the slice installs a small tensor beside
a large one that stays alive -- the saving silently does not happen, and every test still
passes because the loss is unaffected.

The record asked for the gate against ``Qwen3_5ForCausalLM``. What this repository trains
is not that class: ``qd_train.backbone`` builds ``Qwen3_5TextModel`` from the checkpoint's
text tensors, with no LM head, because the tied embedding IS the head (see that module's
docstring). So this measures the model the trainer actually builds, through the path the
trainer actually takes -- ``load_text_tower`` then ``remap_text_tower`` -- and adds nothing
of its own to the surgery. Three things are measured:

1. ``remap_text_tower`` itself: it slices, then verifies every kept row of the embedding in
   both directions against the checkpoint, and raises if one is wrong.
2. The module tree after the remap. A stale reference that the optimizer can see would be a
   parameter or buffer still carrying the source vocabulary's row count, and the parameter
   count would fall by less than ``(V_old - V_new) * hidden``. Both are checked exactly.
3. ``qd_train.remap.remap_parity_report`` -- the plan's gate: loss over the kept vocabulary
   identical before and after, the post-remap logits equal to the pre-remap logits at the
   kept ids, and the full-vocabulary loss delta reported as the remap's intended effect.
   The tower returns hidden states, so a thin module scores them against the tower's own
   input embedding -- the same ``[V, H]`` tensor ``fused_linear_cross_entropy`` receives as
   its weight during training -- and hands that to the report.

What this does NOT measure, and says so on its row: the sequences come from the shard set
whose remap is under test, so every token is inside the kept vocabulary by construction.
That isolates the surgery from COVERAGE -- whether held-out text uses tokens the remap
dropped -- which is the other half of the plan's gate and needs held-out sequences.

Float32 on CPU, deliberately: the gate is an exactness check, and two fp32 copies of the
1.88B-parameter tower fit a 64 GB Mac with room for the ``[1, T, 248320]`` logits the
comparison has to materialise. Costs nothing.

Usage::

    python tools/real_tokenizer_pipeline.py --out /tmp/shards --rev <sha>
    python tools/remap_parity_real.py --shards /tmp/shards/shards/train --ledger
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train.backbone import (  # noqa: E402
    GradientCheckpointingDisabled,
    load_text_tower,
    remap_text_tower,
)
from qd_train.ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.memory import ADAMW_FP32  # noqa: E402
from qd_train.remap import (  # noqa: E402
    MAX_PARITY_LOGIT_BYTES,
    MAX_PARITY_SEQ_LEN,
    PARITY_SEQUENCES,
    remap_parity_report,
)
from qd_train.shards import ShardReader  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

MODEL = "Qwen/Qwen3.5-2B-Base"
#: Where the HF cache records which revision of ``MODEL`` is on this host.
MODEL_REF = (
    Path.home() / ".cache/huggingface/hub"
    / f"models--{MODEL.replace('/', '--')}" / "refs" / "main"
)


class TiedHead(torch.nn.Module):
    """``Qwen3_5TextModel`` scored against its own input embedding.

    The tower has no output head by construction; its input embedding is the head. This is
    the scoring the trainer does inside ``fused_linear_cross_entropy``, materialised,
    because the parity report compares logits and the tower does not produce any.
    """

    def __init__(self, tower: torch.nn.Module) -> None:
        super().__init__()
        self.tower = tower
        self.config = tower.config

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.tower(input_ids=input_ids).last_hidden_state
        return hidden @ self.tower.get_input_embeddings().weight.T


def eligible_sequences(
    reader: ShardReader, *, source_vocab: int, limit: int
) -> tuple[list[np.ndarray], int, int]:
    """Pre-remap id sequences the report can take, in shard order, and how many were not.

    A sequence is eligible when its full-vocabulary logits fit the report's byte cap and it
    has a next token to predict. Longer ones are skipped here rather than handed over to be
    refused one by one, and the count skipped is returned so it is reported, never hidden.
    """
    assert reader.remap is not None
    new_to_old = np.asarray(reader.remap.new_to_old, dtype=np.int64)
    max_len = min(MAX_PARITY_SEQ_LEN, MAX_PARITY_LOGIT_BYTES // (source_vocab * 4))
    chosen: list[np.ndarray] = []
    skipped = 0
    for i in range(len(reader)):
        seq = reader.sequence(i)
        if not 2 <= len(seq) <= max_len:
            skipped += 1
            continue
        if len(chosen) < limit:
            chosen.append(new_to_old[seq])
    return chosen, skipped, len(reader)


def module_tree_check(model: torch.nn.Module, *, full_rows: tuple[int, ...]) -> list[str]:
    """Every parameter or buffer that still has one of the pre-remap row counts.

    Two counts, because the real checkpoint has two: its embedding is 248,320 rows, padded
    for alignment over a 248,077-token tokenizer. The first version of this check looked
    only for the tokenizer's count, so a stale copy of the real embedding would have passed
    it -- found on the first run, when the parameter drop also disagreed with the formula
    by exactly the 243 padding rows.
    """
    named = list(model.named_parameters()) + list(model.named_buffers())
    return [name for name, t in named if t.ndim >= 1 and int(t.shape[0]) in full_rows]


def open_shards(shards: Path) -> ShardReader:
    """The shard set, through the rule-3 door, refused if it carries no remap to test."""
    reader = ShardReader(shards, config=DataConfig(), repo_root=REPO)
    if reader.remap is None:
        raise SystemExit(
            f"{shards} carries no remap table, so there is no remap to test. A parity run "
            "with nothing to compare would report a pass it never measured."
        )
    return reader


def measure(
    *, reader: ShardReader, shards: Path, snapshot: Path, n_sequences: int, device: str
) -> dict:
    """Everything the row records, measured; raises only on a contract violation."""
    remap = reader.remap
    assert remap is not None  # open_shards refused a set without one
    sequences, skipped, total = eligible_sequences(
        reader, source_vocab=remap.source_vocab_size, limit=n_sequences
    )
    if not sequences:
        raise SystemExit(f"none of {total} sequence(s) in {shards} fit the logit byte cap")
    width = max(len(s) for s in sequences)
    print(f"shards   : {shards} -- {total} sequence(s), {skipped} too short or over the "
          "logit cap, "
          f"{len(sequences)} offered (limit {n_sequences}), widest {width}")
    print(f"remap    : {remap.source_vocab_size} -> {remap.vocab_size}")

    t0 = time.monotonic()
    with warnings.catch_warnings():
        # Inference only: no backward pass runs, so there are no activations to recompute
        # and the warning's arithmetic -- about training memory -- does not apply here.
        warnings.simplefilter("ignore", GradientCheckpointingDisabled)
        before = load_text_tower(
            snapshot,
            gradient_checkpointing=False,
            # Required, and irrelevant to an inference-only parity run: it sizes the
            # footprint annotation on the returned TextTower, which this tool never reports.
            optimizer=ADAMW_FP32,
            attn_implementation="sdpa",
            device=device,
            dtype="fp32",
            rows=1,
            width=width,
        )
    load_s = time.monotonic() - t0
    hidden = before.hidden_size
    params_before = sum(p.numel() for p in before.model.parameters())
    print(f"tower    : {before.n_tensors_loaded} tensors, vocab {before.vocab_size}, "
          f"hidden {hidden}, {params_before:,} params, loaded in {load_s:.1f}s")

    # The embedding's own row count, not the tokenizer's: the slice removes every row the
    # remap does not keep, alignment padding included.
    rows_before = int(before.model.get_input_embeddings().weight.shape[0])
    after = remap_text_tower(copy.deepcopy(before), remap)
    params_after = sum(p.numel() for p in after.model.parameters())
    stale = module_tree_check(
        after.model, full_rows=(rows_before, remap.source_vocab_size)
    )
    expected_drop = (rows_before - remap.vocab_size) * hidden
    print(f"remapped : {params_after:,} params (drop {params_before - params_after:,}, "
          f"expected {expected_drop:,}); stale full-vocab tensors: {stale or 'none'}")

    t1 = time.monotonic()
    with torch.inference_mode():
        report = remap_parity_report(
            model_before=TiedHead(before.model),
            model_after=TiedHead(after.model),
            remap=remap,
            sequences=sequences,
            max_sequences=n_sequences,
        )
    parity_s = time.monotonic() - t1
    gate = report.as_tristate()
    print(f"parity   : {json.dumps(gate.to_json())[:600]}")
    print(f"           {parity_s:.1f}s over {report.n_examined} sequence(s)")

    return {
        "gate": gate,
        "metrics": {
            "remap_rows_verified_both_directions": Ran(
                passed=True,
                value=remap.vocab_size,
                n=remap.vocab_size,
                n_total=remap.vocab_size,
                detail=(
                    "remap_text_tower compared every kept embedding row with the "
                    "checkpoint in both directions and did not raise"
                ),
            ),
            "remap_no_stale_full_vocab_tensor": Ran(
                passed=not stale,
                value=len(stale),
                detail=(
                    f"parameters/buffers still carrying {rows_before} (embedding) or "
                    f"{remap.source_vocab_size} (tokenizer) rows after the remap: "
                    f"{stale or 'none'}"
                ),
            ),
            "remap_parameter_drop": Ran(
                passed=(params_before - params_after) == expected_drop,
                value=params_before - params_after,
                detail=(
                    f"{params_before} -> {params_after} parameters; expected exactly "
                    f"({rows_before} embedding rows - {remap.vocab_size} kept) x {hidden} = "
                    f"{expected_drop}. The embedding is {rows_before - remap.source_vocab_size} "
                    f"row(s) wider than the {remap.source_vocab_size}-token tokenizer "
                    "(alignment padding), and the slice drops those too."
                ),
            ),
            "remap_parity_sequences_offered": Ran(
                passed=True,
                value=len(sequences),
                n=len(sequences),
                n_total=total,
                detail=(
                    f"{skipped} of {total} shard sequence(s) were not offered: under 2 "
                    f"tokens, or over the {MAX_PARITY_LOGIT_BYTES}-byte logit cap. The "
                    "rest were offered in shard order up to the limit. All come from the "
                    "shard set under test, so none can contain a dropped token: this "
                    "measures the surgery, not held-out coverage."
                ),
            ),
        },
        "wall_clock_s": load_s + parity_s,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, type=Path, help="a shard set with a remap")
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="checkpoint directory; defaults to the HF cache's refs/main snapshot",
    )
    ap.add_argument("--n-sequences", type=int, default=PARITY_SEQUENCES)
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument(
        "--ledger",
        nargs="?",
        const=str(DEFAULT_LEDGER_PATH),
        default=None,
        help="append one `eval` row; without it the numbers are printed and not recorded",
    )
    args = ap.parse_args(argv)

    if not MODEL_REF.exists():
        raise SystemExit(f"{MODEL_REF} is absent, so no snapshot revision can be named")
    revision = MODEL_REF.read_text(encoding="utf-8").strip()
    snapshot = args.snapshot or MODEL_REF.parents[1] / "snapshots" / revision

    reader = open_shards(args.shards)
    if args.ledger is None:
        measured = measure(
            reader=reader, shards=args.shards, snapshot=snapshot,
            n_sequences=args.n_sequences, device=args.device,
        )
        return 0 if isinstance(measured["gate"], Ran) and measured["gate"].passed else 1

    # Everything the row's identity needs is in the shard header, which is cheap, so the
    # recorder is opened BEFORE the work and wraps it: a run killed while loading or
    # scoring the tower still writes a row saying so.
    header = reader.header
    assert reader.remap is not None
    recipe = {
        "tool": "tools/remap_parity_real.py",
        "model": MODEL,
        "shards_corpus_rev": header.corpus_rev,
        "remap_hash": header.remap_hash,
        "n_sequences": args.n_sequences,
        "dtype": "fp32",
        "device": args.device,
    }
    recorder = RunRecorder(
        Ledger(Path(args.ledger)),
        entry_point=Path(__file__),
        protocol=Protocol(
            data_snapshot_hash=header.data_snapshot_hash,
            tokenizer_hash=reader.remap.tokenizer_hash,
            backbone_commit=revision,
            recipe_hash=hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()
            ).hexdigest(),
            seed=DataConfig().seed,
        ),
        run_kind="eval",
        repo=REPO,
        env=Environment.detect(
            transformers_sha=_transformers_version(), device=args.device
        ),
        # Stated from inside the block by recorder.measured(); None here is what lets a
        # killed run fall back to the recorder's own lifetime.
        wall_clock_s=None,
        # Local and already paid for; Environment.detect names the device, and anything
        # billed by the hour is refused an unstated zero by the recorder itself.
        cost=None,
        recipe=recipe,
        quick=True,
        quick_reason=(
            "sequences drawn from the shard set whose remap is under test, so held-out "
            "coverage -- the other half of the plan's parity gate -- is not measured"
        ),
        notes=(
            "S2 remap parity against the real Qwen3.5-2B text tower "
            "(GAP-S2-NO-REAL-CHECKPOINT-PARITY): load_text_tower -> remap_text_tower -> "
            "remap_parity_report, fp32"
        ),
    )
    with recorder:
        measured = measure(
            reader=reader, shards=args.shards, snapshot=snapshot,
            n_sequences=args.n_sequences, device=args.device,
        )
        recorder.measured(measured["wall_clock_s"])
        for name, value in measured["metrics"].items():
            recorder.metric(name, value)
        recorder.gate("remap_parity", measured["gate"])
        recorder.noul_rate = NotRun(
            reason="a parity run scores logits for identity; it makes no decisions"
        )
    row = recorder.row
    if row is None:
        raise SystemExit("the recorder exited without a row; nothing was recorded")
    print(f"\nledger row: {row.row_id}\n  {Path(args.ledger)}")
    return 0


def _transformers_version() -> str:
    import transformers

    return f"transformers=={transformers.__version__}"


if __name__ == "__main__":
    raise SystemExit(main())
