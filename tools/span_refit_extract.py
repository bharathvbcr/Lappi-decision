"""Extract frozen span features for campaign/span-refit-mac-preregistered.json.

Throwaway analysis (house Python policy, category 3): it never ships.

For each span sequence of one shard set, through one tower (the v5 average, or v5 seed 1's
epoch checkpoint), this writes to a disk cache:

* ``hq.f32``         [N, H] fp32 -- the query state, gathered at ``plan.query_index`` exactly
  as heads.SpanPointerHead.forward gathers it;
* ``cand_start.bf16`` [M, H] bf16 -- one vector per candidate column at ``plan.candidate_pos``
  (the line starts, as served);
* ``cand_end.bf16``   [M, H] bf16 -- the same columns at each line's LAST token, the arm-(e)
  derivation (span_refit_common.derive_columns; decode only);
* ``rows.jsonl``      per-row metadata, appended per batch after the memmaps are flushed;
* ``layout.json``     the row order, candidate offsets and every input's identity.

Hidden state = ``QwenDecisionStep.hidden`` (post-final-norm last_hidden_state) of a step built
by tools/real_ft_run.py ``_real_step`` at fp32 (score_dtype fp32, as row 6af73bef). On the val
set it also decodes, on the fp32 hidden states, the loaded tower's own span head -- and on the
averaged tower, seed 1's head too -- through ``step.span_head`` + ``serving_scores`` and
_decode's rule: the exact, uncached number arm (a)'s validity gate reads.

Resumable: rows already in rows.jsonl are skipped; the layout must match or it refuses.
``--census-only`` loads no tower: it runs the selection and the arm-(e) derivation (prereg
amendment 1 applied; the original rule's failures are reported beside it, with the last
line's tokens) and writes census-amendment1.json (no model, no device, no lock needed).
census.json in cache/avg-val is the pre-amendment census the amendment cites by sha256
(b05069fd...); it is never overwritten.

Rule 3: reads only the prereg's val shard (scoring) or train shard (the fit set). ShardReader
refuses a held-out set; this refuses any other path.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import span_refit_common as C  # noqa: E402

C.repo_imports()

from qd_data.config import DataConfig  # noqa: E402
from qd_train.artifacts import SLOT_SPAN  # noqa: E402
from qd_train.heads import SpanPointerHead, plan_span_batch, serving_scores  # noqa: E402
from qd_train.shards import ShardReader, assemble_batch  # noqa: E402
from qd_train.trainer import ft_supervision  # noqa: E402

DEFAULT_BACKBONE = Path(
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)
DEFAULT_FT_LEDGER = Path(
    "/Users/bharath/qd-campaign/ledgers-h100-2026-10-05/ledger/h100x2-v5-2026-10-03.jsonl"
)


def rss_gib() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


# --- selection -------------------------------------------------------------------------------


def open_reader(prereg: dict[str, Any], split: str) -> tuple[ShardReader, dict[str, Any]]:
    spec = prereg["inputs"]["val_shard" if split == "val" else "train_shard_for_fit"]
    shard_dir = Path(spec["path"])
    if shard_dir.name != split or shard_dir.parent.name != "shards":
        raise SystemExit(f"{shard_dir}: not a shards/{split} directory")
    reader = ShardReader(shard_dir, config=DataConfig(), repo_root=shard_dir.parent.parent)
    if reader.header.split != split:
        raise SystemExit(f"{shard_dir} declares split {reader.header.split!r}, not {split!r}")
    if reader.header.shard_hash() != spec["shard_hash"]:
        raise SystemExit(
            f"{shard_dir} hashes to {reader.header.shard_hash()}, the prereg pins "
            f"{spec['shard_hash']}"
        )
    if reader.header.tokenizer_hash != prereg["inputs"]["tokenizer_hash"]:
        raise SystemExit(f"{shard_dir}: tokenizer_hash differs from the prereg's")
    if reader.sequence_index is None or reader.remap is None:
        raise SystemExit(f"{shard_dir}: no sequence index or remap beside the shards")
    # The tower's embedding is sliced by the TRAIN set's remap (the one the v5 rows trained
    # under); val must share it, as real_ft_run.open_val_reader requires.
    train_header = Path(prereg["inputs"]["train_shard_for_fit"]["path"]) / "header.json"
    train_remap = json.loads(train_header.read_text(encoding="utf-8"))["remap_hash"]
    if reader.header.remap_hash != train_remap:
        raise SystemExit(f"{shard_dir}: remap {reader.header.remap_hash[:16]} is not the train "
                         f"set's {train_remap[:16]}")
    return reader, spec


def select(reader: ShardReader, prereg: dict[str, Any], split: str) -> tuple[list[int], dict[str, Any]]:
    """Which sequences, and (train) which of them are the 10,000 / 2,000 inner split.

    Refuses while any amendment of the pre-registration lists ``pending_before_any_run`` items
    (Amendment 3): "No run is registered until the pending items are fixed". The val branch
    already refuses at its population pin; the train branch had no pin to fail, so it drew
    the fit set from the v6 train set today. Both refuse here, before reading either set.
    """
    pending = [a["id"] for a in prereg.get("amendments", []) if a.get("pending_before_any_run")]
    if pending:
        raise SystemExit(
            f"amendment(s) {pending} of the pre-registration list pending_before_any_run items; "
            f"no extraction (split={split}) is registered until a later amendment fixes them"
        )
    span = np.flatnonzero(reader._slot_kind == SLOT_SPAN)
    seqs = reader.sequence_index.sequences
    lengths = np.asarray(reader.lengths())
    if split == "val":
        keys = [f"{seqs[i][0]}|{seqs[i][1]}" for i in span]
        pin = C.population_pin(keys)
        if pin != C.POPULATION_PIN:
            raise SystemExit(f"val span population hashes to {pin}, not {C.POPULATION_PIN}")
        return [int(i) for i in span], {"population_pin": pin, "n": int(span.size)}
    fit = prereg["fit_set"]
    n_total, seed, cap = int(fit["n_sequences"]), int(fit["seed"]), int(fit["length_cap_tokens"])
    shares = {"qa.answer_span": 0.71, "code.defect_class": 0.29}
    if "71% qa.answer_span, 29% code.defect_class" not in fit["stratification"]:
        raise SystemExit("the prereg's stratification text moved; refusing to guess the mix")
    want = {f: int(round(n_total * s)) for f, s in shares.items()}
    want["qa.answer_span"] = n_total - want["code.defect_class"]
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    info: dict[str, Any] = {"seed": seed, "length_cap_tokens": cap, "want": want, "by_family": {}}
    for fam in C.FAMILIES:
        members = np.asarray([i for i in span if C.family_of(seqs[i][0]) == fam], dtype=np.int64)
        eligible = members[lengths[members] <= cap]
        info["by_family"][fam] = {
            "span_sequences": int(members.size),
            "excluded_over_length_cap": int(members.size - eligible.size),
            "eligible": int(eligible.size),
        }
        if eligible.size < want[fam]:
            raise SystemExit(f"{fam}: {eligible.size} eligible sequences, fewer than {want[fam]}")
        picked.extend(int(i) for i in rng.choice(eligible, size=want[fam], replace=False))
    info["excluded_over_length_cap"] = sum(v["excluded_over_length_cap"] for v in info["by_family"].values())
    perm = rng.permutation(len(picked))
    inner_fit = sorted(picked[int(j)] for j in perm[:10_000])
    inner_select = sorted(picked[int(j)] for j in perm[10_000:])
    info["inner_split"] = {
        "rule": "after the per-family draws, the same generator permutes the 12,000; the first "
                "10,000 are the inner fit set, the last 2,000 the lambda-selection set",
        "inner_fit": inner_fit, "inner_select": inner_select,
    }
    return sorted(picked), info


def columns_for(reader: ShardReader, order: list[int], decode_id) -> dict[int, C.Columns]:
    out: dict[int, C.Columns] = {}
    n2o = np.asarray(reader.remap.new_to_old, dtype=np.int64)
    for i in order:
        ids = n2o[reader.sequence(i)]
        pieces = [decode_id(int(t)) for t in ids]
        out[i] = C.derive_columns(reader.candidates(i), pieces)
    return out


def make_decoder(reader: ShardReader):
    from real_ft_run import _matching_tokenizer

    tok = _matching_tokenizer(reader, what="the span-refit arm-(e) derivation")
    memo: dict[int, str] = {}

    def decode_id(t: int) -> str:
        s = memo.get(t)
        if s is None:
            s = tok.decode([t])
            memo[t] = s
        return s

    return decode_id


def gold_columns(reader: ShardReader, i: int, cols: C.Columns) -> tuple[int, int, bool]:
    s, e = (int(x) for x in reader._span_target[i])
    abstaining = s < 0 and e < 0
    if abstaining:
        n = len(cols.starts)
        return n, n, True
    return cols.starts.index(s), cols.starts.index(e), False


QA_POINTING = "qa.answer_span/pointing"


def arm_e_void(failed_in_qa_pointing: int, qa_pointing_rows: int) -> bool:
    """Amendment 1's void rule: derivation failures exceed 0.5% of qa.answer_span/pointing
    rows (the reading cell). Counted over that cell's own rows; the defect cells report
    their failure counts beside their results."""
    return failed_in_qa_pointing > 0.005 * qa_pointing_rows


def census(reader: ShardReader, order: list[int], cols: dict[int, C.Columns], split: str,
           decode_id) -> dict[str, Any]:
    seqs = reader.sequence_index.sequences
    n2o = np.asarray(reader.remap.new_to_old, dtype=np.int64)
    by_cell: dict[str, dict[str, int]] = {}
    plus_zero = plus_gold_single = 0
    total_cols = 0
    original_fail_tokens: dict[str, int] = {}
    amended_fail_tokens: dict[str, int] = {}
    statuses: dict[str, dict[str, int]] = {"original_rule": {}, "amended_rule": {}}
    for i in order:
        c = cols[i]
        fam = C.family_of(seqs[i][0])
        gs, ge, ab = gold_columns(reader, i, c)
        cell = f"{fam}/{'abstain' if ab else 'pointing'}"
        d = by_cell.setdefault(cell, {
            "rows": 0, "end_failed_original_rule": 0, "end_failed_amended_rule": 0,
            "last_line_by_amendment_1": 0, "collapsed_rows": 0})
        d["rows"] += 1
        d["end_failed_original_rule"] += int(c.original_status != "ok")
        d["end_failed_amended_rule"] += int(c.status != "ok")
        d["last_line_by_amendment_1"] += int(c.last_rule == "amendment-1")
        d["collapsed_rows"] += int(len(c.starts) != len(c.line_ends))
        total_cols += len(c.starts)
        for key, st in (("original_rule", c.original_status), ("amended_rule", c.status)):
            k = st.split("-column-")[0]
            statuses[key][k] = statuses[key].get(k, 0) + 1
        if c.original_status != "ok" or c.status != "ok":
            ids = n2o[reader.sequence(i)]
            last = c.starts[-1]
            tok = repr(decode_id(int(ids[last])))
            nxt = repr(decode_id(int(ids[last + 1]))) if last + 1 < ids.size else "<end>"
            label = f"last-start token {tok}, next {nxt}"
            if c.original_status != "ok":
                original_fail_tokens[label] = original_fail_tokens.get(label, 0) + 1
            if c.status != "ok":
                amended_fail_tokens[label] = amended_fail_tokens.get(label, 0) + 1
        if fam == "code.defect_class" and not ab:
            plus_zero += int(not c.plus)
            plus_gold_single += int(gs == ge and gs in c.plus)
    qa = by_cell.get(QA_POINTING, {"rows": 0, "end_failed_amended_rule": 0})
    return {
        "split": split, "rows": len(order), "candidate_columns": total_cols, "by_cell": by_cell,
        "end_status_counts": statuses,
        "original_rule_failures_by_last_tokens": dict(
            sorted(original_fail_tokens.items(), key=lambda kv: -kv[1])),
        "amended_rule_failures_by_last_tokens": dict(
            sorted(amended_fail_tokens.items(), key=lambda kv: -kv[1])),
        "arm_e_validity": {
            "rule": "amendment 1: void iff derivation failures exceed 0.5% of "
                    "qa.answer_span/pointing rows (failures counted in that cell); defect "
                    "cells report their counts",
            "qa_pointing_rows": qa["rows"],
            "qa_pointing_failures": qa["end_failed_amended_rule"],
            "all_rows_failures": sum(v["end_failed_amended_rule"] for v in by_cell.values()),
            "void": bool(split == "val" and arm_e_void(qa["end_failed_amended_rule"], qa["rows"])),
            "applies": split == "val",
        },
        "defect_pointing_rows_with_no_plus_line": plus_zero,
        "defect_pointing_rows_whose_gold_is_one_plus_line": plus_gold_single,
    }


# --- the tower -------------------------------------------------------------------------------


def seed1_json(prereg: dict[str, Any]) -> Path:
    sidecar = Path(prereg["inputs"]["seed1_checkpoint"]["path"])
    stem = sidecar.name.split(".")[0]
    return sidecar.with_name(f"{stem}.json")


def load_tower(args, prereg, reader, batches, device: str) -> tuple[Any, dict[str, Any], Any]:
    """The step, its weights loaded, built the way --score-checkpoint builds one; plus the
    seed-1 head for arm (b) when the tower is the average."""
    from ckpt_average import read_average, read_manifest
    from qd_train.backbone import revive_tensors
    from qd_train.run_control import Checkpoint
    from real_ft_run import _ft_row, _one_configuration, _real_step, optimizer_spec

    avg_path = Path(prereg["inputs"]["averaged_tower"]["path"])
    manifest = read_manifest(avg_path)  # reads only the manifest JSON
    if manifest.body["safetensors_sha256"] != prereg["inputs"]["averaged_tower"]["sha256"]:
        raise SystemExit("the average's manifest names another safetensors than the prereg")
    rows = [_ft_row(args.ft_ledger, rid) for rid in manifest.ft_row_ids]
    steps = _one_configuration(rows, what=avg_path.name)
    recipe = rows[0]["recipe"]
    seeds = [int(r["seed"]) for r in manifest.inputs]
    s1_row = rows[seeds.index(1)]
    if int(s1_row["protocol"]["seed"]) != 1:
        raise SystemExit(f"ft row {s1_row['row_id']} is not seed 1")
    if args.backbone.name != recipe["backbone_snapshot"]:
        raise SystemExit(f"--backbone {args.backbone.name} is not {recipe['backbone_snapshot']}")

    s1_json = seed1_json(prereg)
    s1_sidecar = Path(prereg["inputs"]["seed1_checkpoint"]["path"])
    provenance: dict[str, Any] = {"ft_row_ids": manifest.ft_row_ids, "optimizer_step": steps}
    if args.tower == "avg":
        weights = read_average(manifest)  # checks the file's sha256 against the manifest
        provenance.update({
            "path": str(avg_path), "sha256": manifest.body["safetensors_sha256"],
            "manifest_sha256": manifest.sha256,
            "verify_sources": "NOT RUN: the five input JSON bodies (/home/ubuntu/ckpt/v5/"
                              "epoch-seed*-cuda.json) left with the deleted box; the safetensors "
                              "sha256 is checked against the manifest and the prereg instead",
        })
        step_seed = int(DataConfig().seed)
        if float(weights["span_weight"]) != float(recipe["span_weight"]):
            raise SystemExit("the average's span_weight is not its ft rows'")
    else:
        got = C.sha256_file(s1_sidecar)
        if got != prereg["inputs"]["seed1_checkpoint"]["sha256"]:
            raise SystemExit(f"{s1_sidecar} hashes to {got}, not the prereg's")
        weights, meta = Checkpoint.read_weights(s1_json, subtrees=("tower", "span_head"))
        if int(meta["optimizer_step"]) != steps or int(meta["seed"]) != 1:
            raise SystemExit(f"{s1_json}: step {meta['optimizer_step']}, seed {meta['seed']}")
        declared = meta.get("sidecar")
        digest = declared.get("digest") if isinstance(declared, dict) else declared
        if not isinstance(digest, str) or Checkpoint.sidecar_path(s1_json, digest) != s1_sidecar:
            raise SystemExit(f"{s1_json}'s sidecar is not {s1_sidecar}")
        provenance.update({"path": str(s1_sidecar), "body": str(s1_json), "sha256": got,
                           "ft_row_id": s1_row["row_id"],
                           "payload_digest": meta["payload_digest"]})
        step_seed = 1
    step, _, budget = _real_step(
        backbone=args.backbone, reader=reader, plan=batches, device=device, dtype="fp32",
        spec=optimizer_spec("fp32", str(recipe.get("optimizer_recipe", "bf16"))),
        attn_implementation=str(recipe["attn_implementation"]), seed=step_seed,
        lr=float(recipe["lr"]), total_steps=steps, span_weight=float(recipe["span_weight"]),
        width=max(int(b.tokens.shape[1]) for b in batches),
    )
    step.load_weights(weights)
    del weights
    provenance["device_budget"] = budget.to_json()
    if step.tower.hidden_size != args.hidden:
        raise SystemExit(f"tower hidden {step.tower.hidden_size} != --hidden {args.hidden}")
    seed1_head = None
    if args.tower == "avg" and args.split == "val":
        sub, meta = Checkpoint.read_weights(s1_json, subtrees=("span_head",))
        seed1_head = SpanPointerHead(step.tower.hidden_size).to(
            device=step.device, dtype=torch.float32)
        seed1_head.load_state_dict(revive_tensors(sub["span_head"], where="span_head"), strict=True)
        provenance["seed1_head_payload_digest"] = meta["payload_digest"]
    return step, provenance, seed1_head


# --- the run ---------------------------------------------------------------------------------


def plan_batches(pending: list[int], lengths: np.ndarray, batch_tokens: int, max_rows: int) -> list[list[int]]:
    out: list[list[int]] = []
    cur: list[int] = []
    for i in pending:  # already length-sorted
        width = max([int(lengths[j]) for j in cur] + [int(lengths[i])])
        if cur and ((len(cur) + 1) * width > batch_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def build_batch(reader: ShardReader, group: list[int], index: int):
    rows = np.asarray(group, dtype=np.int64)
    seqs = [reader.sequence(i) for i in group]
    return assemble_batch(
        seqs, kinds=reader._slot_kind[rows], target_index=reader._target_index[rows],
        spans=reader._span_target[rows], candidates=[reader.candidates(i) for i in group],
        width=max(s.size for s in seqs), bucket=0, index=index,
    )


def tops_of(head, hs, plan) -> list[tuple[int, int]]:
    s, e = head(hs, plan)
    ss, ee = serving_scores(s, plan), serving_scores(e, plan)
    return [(int(a.argmax()), int(b.argmax())) for a, b in zip(ss, ee)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tower", choices=("avg", "seed1"), required=True)
    p.add_argument("--split", choices=("val", "train"), required=True)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    p.add_argument("--limit", type=int, default=None, help="smoke: an evenly spaced subset")
    p.add_argument("--batch-tokens", type=int, default=8192)
    p.add_argument("--max-rows", type=int, default=32)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--census-only", action="store_true")
    p.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    p.add_argument("--ft-ledger", type=Path, default=DEFAULT_FT_LEDGER)
    args = p.parse_args(argv)
    if args.tower == "seed1" and args.split == "train":
        raise SystemExit("the prereg fits on the averaged tower only; seed 1's tower reads val (arm d)")
    if args.batch_tokens < 2084 or args.max_rows < 1:
        raise SystemExit("--batch-tokens must hold one sequence of the 2,084-token cap")
    t0 = time.monotonic()
    prereg = C.load_prereg()
    prereg_sha = C.prereg_sha256()
    reader, spec = open_reader(prereg, args.split)
    order_all, selection = select(reader, prereg, args.split)
    lengths = np.asarray(reader.lengths())
    order_all = sorted(order_all, key=lambda i: (int(lengths[i]), i))
    order = order_all
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        stride = max(1, math.ceil(len(order_all) / args.limit))
        order = order_all[::stride][: args.limit]
    out = args.out or (C.CACHE / f"{args.tower}-{args.split}" if not args.limit
                       else C.CACHE / f"{args.tower}-{args.split}-limit{args.limit}")
    out.mkdir(parents=True, exist_ok=True)
    decode_id = make_decoder(reader)
    t1 = time.monotonic()
    cols = columns_for(reader, order, decode_id)
    print(f"derived columns for {len(order)} rows in {time.monotonic() - t1:.1f}s", flush=True)
    if args.census_only:
        report = census(reader, order, cols, args.split, decode_id)
        report.update({"prereg_sha256": prereg_sha, "shard_hash": spec["shard_hash"],
                       "limit": args.limit,
                       "selection": {k: v for k, v in selection.items() if k != "inner_split"}})
        (out / "census-amendment1.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=1))
        return 0

    offsets = np.zeros(len(order) + 1, dtype=np.int64)
    for k, i in enumerate(order):
        offsets[k + 1] = offsets[k] + len(cols[i].starts)
    layout = {
        "tower": args.tower, "split": args.split, "shard_dir": spec["path"],
        "shard_hash": spec["shard_hash"], "prereg_sha256": prereg_sha, "limit": args.limit,
        "hidden": args.hidden, "order": order, "cand_offsets": offsets.tolist(),
        "n_candidate_vectors": int(offsets[-1]), "selection": selection,
        "hidden_state": "QwenDecisionStep.hidden: last_hidden_state (post-final-norm), fp32 tower",
        "cache_dtypes": {"hq": "float32", "cand_start": "bfloat16", "cand_end": "bfloat16"},
    }
    lpath = out / "layout.json"
    if lpath.exists():
        old = json.loads(lpath.read_text(encoding="utf-8"))
        keep = {k: v for k, v in old.items() if k not in ("weights", "prereg_sha256", "device")}
        new = {k: v for k, v in layout.items() if k not in ("weights", "prereg_sha256", "device")}
        if keep != new:
            raise SystemExit(f"{lpath} describes another extraction; use a fresh --out")
        layout = old
    n, m, h = len(order), int(offsets[-1]), args.hidden
    mode = "r+" if (out / "hq.f32").exists() else "w+"
    hq_mm = np.memmap(out / "hq.f32", dtype=np.float32, mode=mode, shape=(n, h))
    st_mm = C.bf16_memmap(out / "cand_start.bf16", (m, h), mode)
    en_mm = C.bf16_memmap(out / "cand_end.bf16", (m, h), mode)
    done = {int(r["seq"]) for r in C.open_cache(out).rows} if (out / "rows.jsonl").exists() else set()
    slot_of = {i: k for k, i in enumerate(order)}
    pending = [i for i in order if i not in done]
    print(f"{len(done)} rows already cached, {len(pending)} to extract into {out}", flush=True)
    if not pending:
        return 0
    groups = plan_batches(pending, lengths, args.batch_tokens, args.max_rows)
    batches = [build_batch(reader, g, b) for b, g in enumerate(groups)]
    step, provenance, seed1_head = load_tower(args, prereg, reader, batches, args.device)
    if "weights" not in layout:
        layout["weights"] = provenance
        layout["device"] = args.device
        lpath.write_text(json.dumps(layout) + "\n", encoding="utf-8")
    elif layout["weights"].get("sha256") != provenance["sha256"]:
        raise SystemExit(f"{lpath} was extracted from other weights")
    print(f"tower loaded ({args.tower}, {args.device}) in {time.monotonic() - t0:.0f}s; "
          f"max RSS {rss_gib():.1f} GiB", flush=True)
    from real_ft_run import release_device_cache

    seqs = reader.sequence_index.sequences
    t_loop = time.monotonic()
    for b, (group, batch) in enumerate(zip(groups, batches)):
        sup = ft_supervision(batch)
        if sup.span is None or sup.span.rows.size != len(group):
            raise SystemExit(f"batch {b}: not every row is a span row")
        plan = plan_span_batch(sup.span, device=step.device)
        k_rows = len(group)
        width = int(plan.candidate_pos.shape[1])
        end_pos = torch.zeros((k_rows, width), dtype=torch.int64)
        for k, i in enumerate(group):
            c = cols[i]
            nk = int(plan.n_candidates[k])
            if nk != len(c.starts) or plan.candidate_pos[k, :nk].tolist() != list(c.starts):
                raise SystemExit(f"seq {i}: the plan's candidates are not the derived starts")
            gs, ge, ab = gold_columns(reader, i, c)
            if (int(plan.gold_start[k]), int(plan.gold_end[k]), bool(plan.abstaining[k])) != (gs, ge, ab):
                raise SystemExit(f"seq {i}: the plan's gold columns are not the derived ones")
            end_pos[k, :nk] = torch.as_tensor([max(0, e) for e in c.ends], dtype=torch.int64)
        with torch.no_grad():
            hidden = step.hidden(batch)
            hs = hidden[torch.as_tensor(sup.span.rows, device=step.device)].float()
            del hidden
            hq, hc, he = C.gather_features(hs, plan.candidate_pos, plan.query_index,
                                           end_pos.to(step.device))
            live_own = live_s1 = None
            if args.split == "val":
                live_own = tops_of(step.span_head, hs, plan)
                if seed1_head is not None:
                    live_s1 = tops_of(seed1_head, hs, plan)
            hq, hc, he = hq.cpu(), hc.cpu(), he.cpu()
            del hs
        records = []
        for k, i in enumerate(group):
            slot = slot_of[i]
            nk = int(plan.n_candidates[k])
            lo = int(offsets[slot])
            hq_mm[slot] = hq[k].numpy()
            st_mm[lo:lo + nk] = C.to_bf16_bits(hc[k, :nk])
            en_mm[lo:lo + nk] = C.to_bf16_bits(he[k, :nk])
            c = cols[i]
            gs, ge, ab = gold_columns(reader, i, c)
            rec = {
                "slot": slot, "seq": i, "row_id": seqs[i][0], "slot_name": seqs[i][1],
                "family": C.family_of(seqs[i][0]), "abstaining": ab, "gold_start": gs,
                "gold_end": ge, "n_candidates": nk, "n_lines": len(c.line_ends),
                "length": int(lengths[i]), "plus_columns": list(c.plus),
                "end_status": c.status, "delimiter_token": c.delimiter_token,
                "last_rule": c.last_rule, "end_status_original_rule": c.original_status,
            }
            if live_own is not None:
                rec["live_top_own_head"] = list(live_own[k])
            if live_s1 is not None:
                rec["live_top_seed1_head"] = list(live_s1[k])
            records.append(rec)
        hq_mm.flush()
        st_mm.flush()
        en_mm.flush()
        C.append_jsonl(out / "rows.jsonl", records)
        if args.device == "mps":
            release_device_cache()
        if b % 20 == 0 or b == len(groups) - 1:
            el = time.monotonic() - t_loop
            print(f"batch {b + 1}/{len(groups)} rows {sum(len(g) for g in groups[:b + 1])} "
                  f"elapsed {el:.0f}s max RSS {rss_gib():.1f} GiB", flush=True)
    print(f"done: {n} rows, {m} candidate columns, wall {time.monotonic() - t0:.0f}s, "
          f"max RSS {rss_gib():.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
