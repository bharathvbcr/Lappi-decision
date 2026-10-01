"""Reference oracle for `crates/qd-train` (L-data lane). Python here is an oracle only.

Builds a tiny shard set **through the real Python pipeline** -- `small_corpus` ->
`build_mixture` -> `dedupe` -> `split` -> `build_manifests` -> `write_shards`, the recipe
`python/tests/test_shards.py::_snapshot` uses, with its byte tokenizer -- and dumps what the
Python code makes of it, so the Rust port can be held to byte equality:

* the manifests (`data/pool/{train,val}.json`, `data/heldout/heldout.json`) and the train
  shard set (`shards/train/`), as written;
* door fixtures under `door/` -- a held-out manifest moved out of the held-out tree, a train
  manifest carrying a held-out family, one recording a different holdout, `not_run` and
  `ran, passed=false` snapshots, a tampered manifest, a header declaring `split: "heldout"` --
  each opened by Python's own `open_training_data` / `assert_shard_trainable`, whose verdict
  is recorded beside it (`oracle/door.json`);
* a report-only val shard set (`shards/val-report-only/`), which Python's `ShardReader`
  admits and the Rust training reader refuses on purpose;
* `ShardReader.batches(batch_tokens, seed, epoch)` order and the consumed digest, folded
  through the real `trainer._fold` + `run_control.ConsumedPrefix`, for 2 seeds x 2 epochs x 2
  batch_tokens values (`oracle/batches.json`);
* the numpy RNG intermediates `ShardReader._plan` consumes (`oracle/rng.json`), so an order
  mismatch can be located to SeedSequence, PCG64 or the shuffle;
* `trainer.ft_supervision` and `heads.plan_span_batch` per batch (`oracle/supervision.json`);
* Python `json.dumps` renderings of floats and strings (`oracle/pyjson.json`);
* `DataConfig()` defaults and the contract constants (`oracle/meta.json`).

Usage (repository root)::

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_shards.py \\
        --out crates/qd-train/tests/fixtures/shards-tiny

Refuses to write into a non-empty `--out` unless `--replace` is given. Nothing here trains,
and nothing here reads real held-out data: every row is synthetic (`small_corpus`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "python" / "tests"))

from data_fixtures import small_corpus  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.dedupe import dedupe  # noqa: E402
from qd_data.errors import HeldOutViolation  # noqa: E402
from qd_data.manifest import Manifest, build_manifests  # noqa: E402
from qd_data.mixture import build_mixture  # noqa: E402
from qd_data.schema import canonical_json  # noqa: E402
from qd_data.split import HELD_OUT, split  # noqa: E402
from qd_train import artifacts, shards  # noqa: E402
from qd_train.artifacts import (  # noqa: E402
    RemapTable,
    ShardContractViolation,
    ShardHeader,
    assert_shard_trainable,
)
from qd_train.data_access import open_training_data  # noqa: E402
from qd_train.heads import RESERVED_NOUL_ROWS, plan_span_batch  # noqa: E402
from qd_train.run_control import ConsumedPrefix  # noqa: E402
from qd_train.shards import ShardReader, write_shards  # noqa: E402
from qd_train.trainer import _fold, ft_supervision  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

#: The corpus size `test_shards.py` uses; smaller ones lose the intent family entirely
#: (`intent_vocabulary_too_small`) and the snapshot is `not_run`.
CORPUS_N = 24
SOURCE_VOCAB = 512
TOKENIZER_HASH = "tokhash-oracle-0123456789abcdef"
CORPUS_REV = "oracle-rev-0000000000000000000000000000000000"
PROVENANCE = {
    "built_at_utc": "2026-10-01T00:00:00+00:00",
    "oracle": "tools/qd_train_oracle_shards.py",
}
#: The data seed `real_ft_run` orders batches with, and a second one.
SEEDS = (DataConfig().seed, 7)
EPOCHS = (0, 1)


def byte_tokenize(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def byte_offsets(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for ci, ch in enumerate(text):
        out.extend((ci, ci + 1) for _ in ch.encode("utf-8"))
    return out


def byte_decode(ids: Sequence[int]) -> str:
    return bytes(int(i) for i in ids).decode("utf-8", errors="strict")


def byte_remap() -> RemapTable:
    old_to_new = np.full(SOURCE_VOCAB, -1, dtype=np.int32)
    for i in range(256):
        old_to_new[i] = i
    return RemapTable(
        old_to_new=old_to_new,
        new_to_old=np.arange(256, dtype=np.int32),
        tokenizer_hash=TOKENIZER_HASH,
        special_ids=(),
    )


#: Absolute paths in Python's messages are rewritten to this token, so the fixture does not
#: carry the path of the checkout that generated it.
FIXTURE_TOKEN = "<fixture>"


def _write_json(path: Path, payload: Any, *, out: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    path.write_text(text.replace(str(out), FIXTURE_TOKEN), encoding="utf-8")


def _verdict(fn) -> dict[str, Any]:
    """Run one Python door call and record what it did, refusal included."""
    try:
        result = fn()
    except (HeldOutViolation, ShardContractViolation, ValueError) as exc:
        return {"admitted": False, "error": type(exc).__name__, "message": str(exc)}
    out: dict[str, Any] = {"admitted": True}
    if hasattr(result, "data_snapshot_hash"):
        out["data_snapshot_hash"] = result.data_snapshot_hash
    return out


def _manifest_with(base: Manifest, **changes: Any) -> Manifest:
    return dataclasses.replace(base, **changes)


def build(out: Path) -> dict[str, Any]:
    config = DataConfig()
    mixture = build_mixture(small_corpus(CORPUS_N), config=config)
    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report,
        provenance=dict(PROVENANCE),
    )
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        path = out / "data" / (HELD_OUT if name == HELD_OUT else "pool") / f"{name}.json"
        manifest.write(path)
        paths[name] = path
    rows = {k: tuple(v) for k, v in split_report.rows_by_split.items()}

    train_dir = out / "shards" / "train"
    write_shards(
        paths["train"], rows["train"], out_dir=train_dir, remap=byte_remap(),
        tokenize=byte_tokenize, token_offsets=byte_offsets, decode=byte_decode,
        config=config, repo_root=out, corpus_rev=CORPUS_REV,
    )
    report_only_dir = out / "shards" / "val-report-only"
    write_shards(
        paths["val"], rows["val"], out_dir=report_only_dir, remap=byte_remap(),
        tokenize=byte_tokenize, token_offsets=byte_offsets, decode=byte_decode,
        config=config, repo_root=out, corpus_rev=CORPUS_REV,
        span_collapse_policy=shards.SPAN_COLLAPSE_REFUSE_GOLD, report_only=True,
    )
    return {"config": config, "manifests": manifests, "paths": paths, "train_dir": train_dir,
            "report_only_dir": report_only_dir}


def door_fixtures(out: Path, built: dict[str, Any]) -> dict[str, Any]:
    """Manifests and a header that the door must refuse, each opened by Python first."""
    config: DataConfig = built["config"]
    manifests: dict[str, Manifest] = built["manifests"]
    train, heldout = manifests["train"], manifests[HELD_OUT]
    door = out / "door"
    door.mkdir(parents=True, exist_ok=True)

    cases: dict[str, Path] = {}
    # Layer 3: a held-out manifest moved to a path with no held-out marker in it.
    cases["heldout-moved"] = door / "heldout-moved.json"
    heldout.write(cases["heldout-moved"])
    # Layer 4: a train manifest carrying one held-out family entry, relabelled split=train.
    leak = next(e for e in heldout.entries if e.family_id in config.held_out_families)
    leaked = dataclasses.replace(leak, split="train", row_id=leak.row_id + "-leaked")
    cases["train-with-heldout-family"] = door / "train-with-heldout-family.json"
    _manifest_with(train, entries=(*train.entries, leaked)).write(
        cases["train-with-heldout-family"]
    )
    # Layer 4: a manifest recording a different holdout than the config names.
    cases["train-other-holdout"] = door / "train-other-holdout.json"
    _manifest_with(train, held_out_families=("code.change_scope", "code.language_id")).write(
        cases["train-other-holdout"]
    )
    # Snapshot status: not_run, and ran-and-failed.
    cases["train-not-run"] = door / "train-not-run.json"
    _manifest_with(
        train, status=NotRun(reason="dedupe hit its bound (oracle fixture)")
    ).write(cases["train-not-run"], allow_not_run=True)
    cases["train-failed"] = door / "train-failed.json"
    _manifest_with(
        train, status=Ran(passed=False, detail="split gate failed (oracle fixture)")
    ).write(cases["train-failed"])
    # Tampered: one content hash edited after the file was written.
    cases["train-tampered"] = door / "train-tampered.json"
    raw = json.loads(built["paths"]["train"].read_text(encoding="utf-8"))
    raw["entries"][0]["content_hash"] = "0" * 64
    cases["train-tampered"].write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    verdicts: dict[str, Any] = {}
    for name in ("train", "val"):
        verdicts[f"pool/{name}"] = _verdict(
            lambda p=built["paths"][name]: open_training_data(p, config=config, repo_root=out)
        )
    verdicts["heldout/heldout"] = _verdict(
        lambda: open_training_data(built["paths"][HELD_OUT], config=config, repo_root=out)
    )
    for name, path in cases.items():
        verdicts[f"door/{name}"] = _verdict(
            lambda p=path: open_training_data(p, config=config, repo_root=out)
        )

    # Rule 3 at the shard boundary: the train header, declaring split=heldout.
    train_header = ShardHeader.from_json(
        json.loads((built["train_dir"] / shards.HEADER_NAME).read_text(encoding="utf-8"))
    )
    heldout_header = dataclasses.replace(train_header, split="heldout")
    header_path = door / "header-split-heldout.json"
    header_path.write_text(
        json.dumps(heldout_header.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verdicts["door/header-split-heldout"] = _verdict(
        lambda: assert_shard_trainable(
            heldout_header, config=config, path=built["train_dir"], repo_root=out
        )
    )
    # The same supervision arrays, deflated. np.savez never writes this; np.load reads it
    # anyway, and the Rust reader refuses it by name rather than carrying a deflate decoder.
    with np.load(built["train_dir"] / shards.SUPERVISION_NAME) as z:
        arrays = {k: z[k] for k in z.files}
    np.savez_compressed(door / "supervision-deflated.npz", **arrays)
    # The report-only val set: Python's ShardReader admits it; the Rust training reader does not.
    verdicts["shards/val-report-only"] = _verdict(
        lambda: ShardReader(built["report_only_dir"], config=config, repo_root=out)
    )
    for name, verdict in verdicts.items():
        expect_admitted = name in ("pool/train", "pool/val", "shards/val-report-only")
        if verdict["admitted"] != expect_admitted:
            raise SystemExit(
                f"oracle: Python's door did not behave as expected on {name}: {verdict}"
            )
    return verdicts


def _batch_record(batch: artifacts.Batch) -> dict[str, Any]:
    return {
        "index": int(batch.index),
        "bucket": int(batch.bucket),
        "width": int(batch.tokens.shape[1]),
        "lengths": [int(n) for n in batch.lengths],
    }


def batch_orders(reader: ShardReader, batch_tokens: Sequence[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in SEEDS:
        for epoch in EPOCHS:
            for bt in batch_tokens:
                prefix = ConsumedPrefix()
                plans = reader._plan(batch_tokens=bt, seed=seed, epoch=epoch)
                batches: list[dict[str, Any]] = []
                for batch, plan in zip(
                    reader.batches(batch_tokens=bt, seed=seed, epoch=epoch), plans, strict=True
                ):
                    one = ConsumedPrefix()
                    _fold(one, batch)
                    _fold(prefix, batch)
                    record = _batch_record(batch)
                    record["rows"] = list(plan.rows)
                    record["fold_digest"] = one.hexdigest()
                    batches.append(record)
                out.append({
                    "seed": seed, "epoch": epoch, "batch_tokens": bt,
                    "n_batches": len(batches), "consumed_digest": prefix.hexdigest(),
                    "n_folded": prefix.n_folded, "batches": batches,
                })
    return out


def rng_intermediates(reader: ShardReader, batch_tokens: int) -> dict[str, Any]:
    seed, epoch = SEEDS[0], EPOCHS[0]
    members: dict[int, list[int]] = {}
    for i, b in enumerate(artifacts.assign_buckets(reader.lengths(), reader.header.buckets)):
        members.setdefault(b, []).append(i)
    streams = []
    for tag in [*sorted(members), 0xB17C]:
        entropy = [seed, epoch, batch_tokens, tag]
        ss = np.random.SeedSequence(entropy)
        state = ss.generate_state(4, np.uint64)
        raw = np.random.PCG64(np.random.SeedSequence(entropy)).random_raw(8)
        perm16 = np.random.default_rng(np.random.SeedSequence(entropy)).permutation(16)
        streams.append({
            "entropy": entropy,
            "pool": [int(x) for x in ss.pool],
            "generate_state_u64": [int(x) for x in state],
            "random_raw": [int(x) for x in raw],
            "permutation_16": [int(x) for x in perm16],
        })
    big = [2**32 + 5, 0, 123456789012, 1]
    ss = np.random.SeedSequence(big)
    streams.append({
        "entropy": big,
        "pool": [int(x) for x in ss.pool],
        "generate_state_u64": [int(x) for x in ss.generate_state(4, np.uint64)],
        "random_raw": [int(x) for x in np.random.PCG64(np.random.SeedSequence(big)).random_raw(8)],
        "permutation_16": [
            int(x) for x in np.random.default_rng(np.random.SeedSequence(big)).permutation(16)
        ],
    })
    return {"seed": seed, "epoch": epoch, "batch_tokens": batch_tokens, "streams": streams}


def supervision_dump(reader: ShardReader, batch_tokens: Sequence[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seed, epoch = SEEDS[0], EPOCHS[0]
    for bt in batch_tokens:
        per_batch: list[dict[str, Any]] = []
        for batch in reader.batches(batch_tokens=bt, seed=seed, epoch=epoch):
            sup = ft_supervision(batch)
            rows, cols = np.nonzero(sup.mask)
            record: dict[str, Any] = {
                "index": int(batch.index),
                "mask_shape": [int(d) for d in sup.mask.shape],
                "letter_positions": [[int(r), int(c)] for r, c in zip(rows, cols, strict=True)],
                "letter_targets": [int(sup.targets[r, c]) for r, c in zip(rows, cols, strict=True)],
                "n_supervised": int(sup.n_supervised),
                "n_answers": int(sup.n_answers),
                "span": None,
            }
            if sup.span is not None:
                span = sup.span
                plan = plan_span_batch(span)
                record["span"] = {
                    "rows": [int(r) for r in span.rows],
                    "query_index": [int(q) for q in span.query_index],
                    "start": [int(s) for s in span.start],
                    "end": [int(e) for e in span.end],
                    "abstaining": [bool(a) for a in span.abstaining],
                    "candidate_counts": [int(c) for c in span.candidate_counts()],
                    "plan": {
                        "candidate_pos": plan.candidate_pos.tolist(),
                        "candidate_valid": plan.candidate_valid.tolist(),
                        "n_candidates": plan.n_candidates.tolist(),
                        "query_index": plan.query_index.tolist(),
                        "gold_start": plan.gold_start.tolist(),
                        "gold_end": plan.gold_end.tolist(),
                        "abstaining": plan.abstaining.tolist(),
                        "runtime_rows": plan.runtime_rows.tolist(),
                        "max_rows": int(plan.max_rows),
                        "n_spans": int(plan.n_spans),
                    },
                }
            per_batch.append(record)
        out.append({"seed": seed, "epoch": epoch, "batch_tokens": bt, "batches": per_batch})
    return out


def reader_facts(reader: ShardReader) -> dict[str, Any]:
    lengths = reader.lengths()
    return {
        "shard_hash": reader.header.shard_hash(),
        "n_sequences": len(reader),
        "total_tokens": int(reader.header.total_tokens),
        "max_seq_len": int(reader.header.max_seq_len),
        "buckets": list(reader.header.buckets),
        "lengths": lengths,
        "sequence_sha256": [
            hashlib.sha256(np.ascontiguousarray(reader.sequence(i)).tobytes()).hexdigest()
            for i in range(len(reader))
        ],
        "candidates": [[int(p) for p in reader.candidates(i)] for i in range(len(reader))],
        "slot_kind": [int(k) for k in reader._slot_kind],
        "target_index": [int(t) for t in reader._target_index],
        "span_target": [[int(a), int(b)] for a, b in reader._span_target],
        "to_json": reader.to_json(),
    }


def pyjson_cases() -> dict[str, Any]:
    floats = [0.8, 0.9, 0.05, 1.0, 1e-05, 2.5e-05, 0.0001, 1.5e-07, 1e15, 1e16, 1.2345e16,
              1e22, 9.999999999999999e15, 0.1 + 0.2, 5e-324, 123456789.123, -2.5, -0.0,
              1e100, 1.7976931348623157e308]
    obj = {"b": [1, 2.5, "é"], "a": {"z": None, "y": True}}
    strings = ["plain", "é ü", "tab\tnl\nquote\"bs\\", "\x01\x1f\x7f", "😀 astral", "ﬆ"]
    return {
        "floats": [{"value": f, "repr": repr(f), "canonical": canonical_json(f)} for f in floats],
        "strings": [
            {
                "value": s,
                "canonical": canonical_json(s),
                "sorted_default": json.dumps(s, sort_keys=True),
            }
            for s in strings
        ],
        "object": {
            "value": obj,
            "canonical": canonical_json(obj),
            "sorted_default": json.dumps(obj, sort_keys=True),
        },
    }


def meta(config: DataConfig) -> dict[str, Any]:
    return {
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "corpus_n": CORPUS_N,
        "corpus_rev": CORPUS_REV,
        "tokenizer_hash": TOKENIZER_HASH,
        "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "data_config": {
            "held_out_families": list(config.held_out_families),
            "held_out_roots": [str(p) for p in config.held_out_roots],
            "held_out_path_markers": list(config.held_out_path_markers),
            "seed": config.seed,
        },
        "constants": {
            "PAD_ID": shards.PAD_ID,
            "MAX_ROWS_PER_BATCH": shards.MAX_ROWS_PER_BATCH,
            "MAX_POSITIONS_PER_BATCH": shards.MAX_POSITIONS_PER_BATCH,
            "MAX_PADDING_WASTE": artifacts.MAX_PADDING_WASTE,
            "NO_SPAN": artifacts.NO_SPAN,
            "SPAN_ABSTAIN": artifacts.SPAN_ABSTAIN,
            "SLOT_LM": artifacts.SLOT_LM,
            "SLOT_CHOICE": artifacts.SLOT_CHOICE,
            "SLOT_SCORE": artifacts.SLOT_SCORE,
            "SLOT_SPAN": artifacts.SLOT_SPAN,
            "RESERVED_NOUL_ROWS": RESERVED_NOUL_ROWS,
            "CONSUMED_PREFIX_DOMAIN": ConsumedPrefix.DOMAIN.decode(),
            "SHARD_FORMAT": artifacts.SHARD_FORMAT,
            "REMAP_FORMAT": artifacts.REMAP_FORMAT,
            "SEQUENCE_INDEX_FORMAT": shards.SEQUENCE_INDEX_FORMAT,
        },
    }


def v4_digests(out: Path, rev: str) -> None:
    """Print what `crates/qd-train/tests/v4.rs` pins: the full epoch-0 plan's membership hash
    (u64-LE bucket then u64-LE rows, per batch) and the consumed digest over its first 8
    batches, at `batch_tokens = 2 * max_seq_len` and `DataConfig().seed`. Reads, writes nothing."""
    config = DataConfig()
    reader = ShardReader(out / "shards" / "train", config=config, repo_root=out, expect_rev=rev)
    bt = 2 * reader.header.max_seq_len
    plans = reader._plan(batch_tokens=bt, seed=config.seed, epoch=0)
    membership = hashlib.sha256()
    for plan in plans:
        membership.update(int(plan.bucket).to_bytes(8, "little"))
        for row in plan.rows:
            membership.update(int(row).to_bytes(8, "little"))
    prefix = ConsumedPrefix()
    for index, batch in enumerate(reader.batches(batch_tokens=bt, seed=config.seed, epoch=0)):
        if index == 8:
            break
        _fold(prefix, batch)
    print(json.dumps({
        "sequences": len(reader), "batches": len(plans), "batch_tokens": bt, "seed": config.seed,
        "membership_sha256": membership.hexdigest(), "consumed_digest_first_8": prefix.hexdigest(),
    }, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replace", action="store_true",
                        help="delete --out first if it exists (it must be a fixture directory)")
    parser.add_argument("--v4-out", type=Path,
                        help="print the v4 plan digests tests/v4.rs pins, for this out dir")
    parser.add_argument("--v4-rev", default="881ab304f15ea13529002391dda8520c2ea47af4",
                        help="corpus revision of the v4 set (ledger row d96409bd recipe.rev)")
    args = parser.parse_args()
    if args.v4_out is not None:
        v4_digests(args.v4_out.resolve(), args.v4_rev)
        return
    if args.out is None:
        raise SystemExit("--out is required (or --v4-out)")
    out: Path = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.replace:
            raise SystemExit(f"{out} is not empty; pass --replace to regenerate it")
        if not (out / "oracle" / "meta.json").is_file():
            raise SystemExit(f"{out} does not look like an oracle fixture; refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    built = build(out)
    config: DataConfig = built["config"]
    verdicts = door_fixtures(out, built)
    reader = ShardReader(built["train_dir"], config=config, repo_root=out, expect_rev=CORPUS_REV)
    widest = int(max(reader.header.buckets))
    # Two budgets that change the per-bucket chunking, not only the seed entropy.
    batch_tokens = (2 * widest, 4 * widest + 1)

    oracle = out / "oracle"
    _write_json(oracle / "meta.json", meta(config) | {"batch_tokens": list(batch_tokens)}, out=out)
    _write_json(oracle / "door.json", verdicts, out=out)
    _write_json(oracle / "reader.json", reader_facts(reader), out=out)
    _write_json(oracle / "batches.json", batch_orders(reader, batch_tokens), out=out)
    _write_json(oracle / "rng.json", rng_intermediates(reader, batch_tokens[0]), out=out)
    _write_json(oracle / "supervision.json", supervision_dump(reader, batch_tokens), out=out)
    _write_json(oracle / "pyjson.json", pyjson_cases(), out=out)
    leaked = [p for p in out.rglob("*") if p.is_file() and str(out).encode() in p.read_bytes()]
    if leaked:
        raise SystemExit(f"oracle: the checkout path leaked into {leaked}")
    print(f"oracle: {len(reader)} sequences, buckets {list(reader.header.buckets)}, "
          f"batch_tokens {batch_tokens}, written to {out}")


if __name__ == "__main__":
    main()
