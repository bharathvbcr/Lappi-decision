"""L-recency: where F's noul rows sit in F's batch plan. Report-only; nothing gates on it.

Fable's post-F ruling (AUDIT/post-f-2026-10-02/fable-post-f-ruling.md:30) asks, from F's batch
plans, for the fraction of the 2,010 prose-noul rows and of all 5,004 defect-noul-v3b rows that
fall in the last 10% of each seed's optimizer steps, beside F's per-seed ood_abstain.prose. The
hypothesis it tests: F's plan is globally shuffled, the lr decays by cosine to lr/10 and the final
loss is ~1e-4, so the rows trained last are the ones the final weights hold. A hypothesis, not a
claim; n = 3 seeds.

Throwaway analysis (a permitted Python role under the language policy). It trains nothing, loads
no tower, touches no GPU or MPS device, and writes one JSON file.

What it does, in order, failing closed at each step:

1. Reads F's three ft rows and three eval rows from the ledger by id, and checks that each eval
   row names its ft row.
2. Checks every module of the code root (an export of F's commit a502670) against the sha256
   prefixes in the ft row's ``code_that_ran`` detail, and that the lines that build F's plan are
   present verbatim (``code_anchors``). ``real_ft_run.main`` at a502670 builds ``plan_all`` ONCE,
   at ``DataConfig().seed``, and hands that same list to ``_train`` for every seed in --seeds; this
   tool reproduces that plan through ``ShardReader._plan``/``batches`` directly, never through
   ``main`` (whose device probes would touch MPS).
3. Opens F's train shard set with a502670's ``ShardReader`` (shard hash, rev, qd_data
   fingerprint and sequence index all checked by the reader itself), builds the plan and checks
   it against every plan quantity the ft rows record. On any mismatch it writes the checks and
   stops: no fraction is computed on a plan that is not F's.
4. Labels every sequence from the writer's own sequence index (what F's run paired its labels
   against, ``relabel_train``/``pair_labels``), joins the v3b corpus by
   ``qdm:code.defect_class:<id>`` (``qd_data.mixture``), and counts, in three units, how many of
   each group's sequences/rows sit in the last 10% of the plan's steps.
5. References: the uniform expectations, exact unstratified and bucket-stratified hypergeometric
   tails, a Monte Carlo of the stratified null for the row unit, and optionally the same counts
   over plans this planner builds at other seeds.
6. Optionally folds the whole plan through a502670's ``trainer._fold`` to get the
   ``consumed_digest`` F's final checkpoints carry, so a single read-only check on the box can
   confirm or refute that the three seeds consumed one order.

Run it under the Mac's heavy-job lock; HANDOFF/recency-2026-10-02.md has the exact command.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: F's three ft rows and their eval rows, ledger/gh200-p4-v4-2026-10-01.jsonl.
FT_ROWS: dict[int, str] = {
    0: "973cd4e3-e0d2-4ff8-8588-b761cb842b75",
    1: "95fa4854-6146-4824-b7ad-22bb54744ff1",
    2: "32990e1a-0c61-49fa-9217-800eb845f27e",
}
EVAL_ROWS: dict[int, str] = {
    0: "f4feac15-db49-4159-bb9b-695866c855cc",
    1: "aeca8d69-4733-4593-92de-2073021ed684",
    2: "8c3a774a-87d7-4452-a33b-49cc6fbd911e",
}
F_COMMIT = "a5026707b6e3e57c253be32003ba32420f2d78e2"
#: defect-noul-v3b examples.jsonl, as F's lane linked it (HANDOFF/gh200-phase4-2026-10-01.md).
V3B_SHA256 = "b194f14c75e1ac959f3658e0dda6a07ef4c41ff026a416a39dd526a9dfb13a5e"
DEFECT_FAMILY = "code.defect_class"
CHOICE_SLOT = "defect_class"
SPAN_SLOT = "defect_span"
LATE_FRACTION_NUM, LATE_FRACTION_DEN = 1, 10
MAX_NULL_SEEDS = 2000
MAX_MC_DRAWS = 200_000

#: The lines of a502670's real_ft_run.py and shards.py that make F's plan what this tool
#: rebuilds. Each must occur exactly once in the code root, or the tool refuses.
CODE_ANCHORS: dict[str, tuple[str, str]] = {
    "config_is_default": ("tools/real_ft_run.py", "    config = DataConfig()\n"),
    "batch_tokens_from_argv": (
        "tools/real_ft_run.py",
        "    batch_tokens, recipe_batch_tokens = _resolve_batch_tokens(\n",
    ),
    "plan_built_once_at_config_seed": (
        "tools/real_ft_run.py",
        "    plan_all = list(reader.batches(batch_tokens=batch_tokens, seed=config.seed, epoch=0))\n",
    ),
    "every_seed_trains_plan_all": (
        "tools/real_ft_run.py",
        "                    reader=reader, plan=plan_all, passes=1, device=device, seed=seed,\n",
    ),
    "source_reindexes_in_plan_order": (
        "tools/real_ft_run.py",
        "                out = dataclasses.replace(batch, index=index)\n",
    ),
    "bucket_shuffle_seed": (
        "python/qd_train/shards.py",
        "                np.random.SeedSequence([seed, epoch, batch_tokens, b])\n",
    ),
    "batch_order_shuffle_seed": (
        "python/qd_train/shards.py",
        "            np.random.SeedSequence([seed, epoch, batch_tokens, 0xB17C])\n",
    ),
    "default_seed": ("python/qd_data/config.py", "    seed: int = 20260919\n"),
    "row_id_format": (
        "python/qd_data/mixture.py",
        '    row_id = f"qdm:{family_id}:{raw.example_id}"\n',
    ),
}


class Refused(SystemExit):
    """A precondition failed; nothing past it is computed."""

    def __init__(self, msg: str) -> None:
        print(f"REFUSED: {msg}", flush=True)
        super().__init__(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru) if sys.platform == "darwin" else int(ru) * 1024


# --- the ledger -------------------------------------------------------------------------------


def read_ledger(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                rid = row.get("row_id")
                if not isinstance(rid, str) or rid in rows:
                    raise Refused(f"{path}:{n}: missing or duplicate row_id {rid!r}")
                rows[rid] = row
    return rows


def metric(row: dict[str, Any], name: str) -> dict[str, Any]:
    m = row.get("metrics", {}).get(name)
    if not isinstance(m, dict) or m.get("state") != "ran":
        raise Refused(f"row {row['row_id']}: metric {name} is absent or not ran: {m!r}")
    return m


def f_rows(ledger: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Per seed: what the ft row records about the plan, and the eval row's prose count."""
    out: dict[int, dict[str, Any]] = {}
    for seed, ft_id in FT_ROWS.items():
        ft = ledger.get(ft_id)
        ev = ledger.get(EVAL_ROWS[seed])
        if ft is None or ev is None:
            raise Refused(f"seed {seed}: ft {ft_id} or eval {EVAL_ROWS[seed]} not in the ledger")
        if ft.get("run_kind") != "ft" or ev.get("run_kind") != "eval":
            raise Refused(f"seed {seed}: run kinds {ft.get('run_kind')}/{ev.get('run_kind')}")
        if ft["protocol"]["seed"] != seed or ev["protocol"]["seed"] != seed:
            raise Refused(f"seed {seed}: protocol seeds {ft['protocol']['seed']}/{ev['protocol']['seed']}")
        if ft_id not in str(ev.get("notes", "")):
            raise Refused(f"seed {seed}: eval row {ev['row_id']} does not name ft row {ft_id}")
        if ft.get("code_commit") != F_COMMIT or ev.get("code_commit") != F_COMMIT:
            raise Refused(f"seed {seed}: code_commit is not {F_COMMIT}")
        if ft.get("quick") is not False or ft.get("status") != "completed":
            raise Refused(f"seed {seed}: ft row is quick or not completed")
        recipe = ft["recipe"]
        pad = metric(ft, "train.padding_fraction")
        prose = metric(ev, "ood_abstain.prose")
        out[seed] = {
            "ft_row_id": ft_id,
            "eval_row_id": ev["row_id"],
            "recipe": {
                k: recipe.get(k)
                for k in ("batches", "width", "shard_hash", "batch_tokens", "passes", "lr", "tag")
            },
            "recipe_has_option_permutation_seed": "option_permutation_seed" in recipe,
            "metrics": {
                "corpus.plan_batches": metric(ft, "corpus.plan_batches")["value"],
                "corpus.plan_max_width": metric(ft, "corpus.plan_max_width")["value"],
                "corpus.plan_rows": metric(ft, "corpus.plan_rows")["value"],
                "corpus.shard_sequences": metric(ft, "corpus.shard_sequences")["value"],
                "train.optimizer_steps": metric(ft, "train.optimizer_steps")["value"],
                "train.micro_batches": metric(ft, "train.micro_batches")["value"],
                "train.termination": metric(ft, "train.termination")["value"],
                "train.padding_fraction.n": pad["n"],
                "train.padding_fraction.n_total": pad["n_total"],
                "train.final_loss": metric(ft, "train.final_loss")["value"],
            },
            "code_that_ran": metric(ft, "code_that_ran")["detail"],
            "ood_abstain.prose": {"n": prose["n"], "n_total": prose["n_total"],
                                  "detail": prose.get("detail")},
        }
    return out


# --- the code root ----------------------------------------------------------------------------


def check_code(code_root: Path, detail: str) -> list[dict[str, Any]]:
    """Every ``name=prefix`` of code_that_ran against the code root's file. Fails closed."""
    checks = []
    for item in detail.split():
        name, _, prefix = item.partition("=")
        if not prefix:
            raise Refused(f"code_that_ran item {item!r} is not name=prefix")
        rel = name if "/" in name else f"python/qd_train/{name}"
        path = code_root / rel
        if not path.is_file():
            raise Refused(f"code_that_ran names {rel}, absent from {code_root}")
        got = sha256_file(path)
        checks.append({"file": rel, "expected_prefix": prefix, "got": got,
                       "ok": got.startswith(prefix)})
    bad = [c for c in checks if not c["ok"]]
    if bad:
        raise Refused(f"{len(bad)} module(s) differ from F's code_that_ran: {bad}")
    return checks


def check_anchors(code_root: Path) -> dict[str, Any]:
    found = {}
    for key, (rel, text) in CODE_ANCHORS.items():
        lines = (code_root / rel).read_text(encoding="utf-8").splitlines(keepends=True)
        at = [i + 1 for i, line in enumerate(lines) if line == text]
        if len(at) != 1:
            raise Refused(f"anchor {key}: {text.strip()!r} occurs {len(at)} times in {rel}")
        found[key] = {"file": rel, "line": at[0], "text": text.strip()}
    return found


# --- the exact references ---------------------------------------------------------------------


def log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_pmf(pop: int, succ: int, draws: int) -> tuple[int, list[float]]:
    """pmf of the late count when ``draws`` members are placed uniformly among ``pop``, of which
    ``succ`` are late. Returns (lowest support value, pmf over the support)."""
    lo, hi = max(0, draws - (pop - succ)), min(draws, succ)
    denom = log_comb(pop, draws)
    pmf = [math.exp(log_comb(succ, x) + log_comb(pop - succ, draws - x) - denom)
           for x in range(lo, hi + 1)]
    return lo, pmf


def convolve(a: tuple[int, list[float]], b: tuple[int, list[float]]) -> tuple[int, list[float]]:
    import numpy as np

    return a[0] + b[0], list(np.convolve(np.asarray(a[1]), np.asarray(b[1])))


def tails(dist: tuple[int, list[float]], x: int) -> dict[str, float]:
    lo, pmf = dist
    total = sum(pmf)
    mean = sum((lo + i) * p for i, p in enumerate(pmf)) / total
    ge = sum(p for i, p in enumerate(pmf) if lo + i >= x) / total
    le = sum(p for i, p in enumerate(pmf) if lo + i <= x) / total
    return {"expected": mean, "p_ge": min(1.0, ge), "p_le": min(1.0, le),
            "p_two_sided": min(1.0, 2 * min(ge, le)), "pmf_mass": total}


# --- the analysis -----------------------------------------------------------------------------


def late_positions(n_steps: int) -> int:
    """First plan position in the last 10% of steps: the final floor(N/10) steps."""
    return n_steps - (n_steps * LATE_FRACTION_NUM) // LATE_FRACTION_DEN


def build_groups(index: Any, v3b: list[dict[str, Any]]) -> dict[str, Any]:
    """Map each v3b row to its sequences in the shard set, by the writer's own index."""
    by_row: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for i, (row_id, slot_name) in enumerate(index.sequences):
        by_row[row_id].append((i, slot_name))
    excluded_by_row: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for e in index.excluded:
        excluded_by_row[e.row_id].append({"slot_name": e.slot_name, "scope": e.scope,
                                          "refusal": e.refusal})
    rows: list[dict[str, Any]] = []
    absent: list[str] = []
    for r in v3b:
        rid = f"qdm:{DEFECT_FAMILY}:{r['id']}"
        seqs = by_row.get(rid, [])
        if not seqs:
            absent.append(rid)
        rows.append({"row_id": rid, "source": r["noul_source"], "form": r["noul_form"],
                     "seqs": seqs, "excluded": excluded_by_row.get(rid, [])})
    return {"rows": rows, "absent": absent}


def count_units(rows: list[dict[str, Any]], pos: list[int], late_from: int) -> dict[str, Any]:
    seq_all = [i for r in rows for i, _ in r["seqs"]]
    seq_choice = [i for r in rows for i, s in r["seqs"] if s == CHOICE_SLOT]
    seq_span = [i for r in rows for i, s in r["seqs"] if s == SPAN_SLOT]
    rows_present = [r for r in rows if r["seqs"]]

    def late(seqs: list[int]) -> int:
        return sum(1 for i in seqs if pos[i] >= late_from)

    rows_any = sum(1 for r in rows_present if any(pos[i] >= late_from for i, _ in r["seqs"]))
    rows_all = sum(1 for r in rows_present if all(pos[i] >= late_from for i, _ in r["seqs"]))

    def unit(n_late: int, n: int) -> dict[str, Any]:
        return {"late": n_late, "n": n, "fraction": (n_late / n) if n else None}

    return {
        "sequences_all_slots": unit(late(seq_all), len(seq_all)),
        "sequences_choice_slot": unit(late(seq_choice), len(seq_choice)),
        "sequences_span_slot": unit(late(seq_span), len(seq_span)),
        "rows_any_sequence_late": unit(rows_any, len(rows_present)),
        "rows_every_sequence_late": unit(rows_all, len(rows_present)),
    }


def references(rows: list[dict[str, Any]], pos: list[int], bucket_of: list[int],
               late_from: int, counts: dict[str, Any], mc_draws: int,
               mc_seed: int) -> dict[str, Any]:
    """Uniform expectations, exact hypergeometric tails (plain and bucket-stratified), and a
    Monte Carlo of the stratified null for the row unit."""
    import numpy as np

    n_seq = len(pos)
    late_mask = [p >= late_from for p in pos]
    k_late = sum(late_mask)
    bucket_n: collections.Counter[int] = collections.Counter(bucket_of)
    bucket_late: collections.Counter[int] = collections.Counter(
        b for b, is_late in zip(bucket_of, late_mask, strict=True) if is_late
    )
    out: dict[str, Any] = {
        "population_sequences": n_seq,
        "late_sequences": k_late,
        "expected_fraction_by_sequences": k_late / n_seq,
    }
    for unit, slot in (("sequences_all_slots", None), ("sequences_choice_slot", CHOICE_SLOT),
                       ("sequences_span_slot", SPAN_SLOT)):
        members = [i for r in rows for i, s in r["seqs"] if slot is None or s == slot]
        x = counts[unit]["late"]
        plain = tails(hypergeom_pmf(n_seq, k_late, len(members)), x)
        per_bucket: collections.Counter[int] = collections.Counter(bucket_of[i] for i in members)
        dist: tuple[int, list[float]] = (0, [1.0])
        for b, n_b in sorted(per_bucket.items()):
            dist = convolve(dist, hypergeom_pmf(bucket_n[b], bucket_late[b], n_b))
        strat = tails(dist, x)
        out[unit] = {
            "hypergeometric_unstratified": plain,
            "hypergeometric_bucket_stratified": strat,
            "members_by_bucket": {str(b): n for b, n in sorted(per_bucket.items())},
        }
    # Row unit: a row is late if any of its sequences is. Under the stratified null each
    # bucket's late members are a uniform subset of its members, independently across buckets.
    rng = np.random.default_rng(mc_seed)
    present = [r for r in rows if r["seqs"]]
    seq_ids = sorted({i for r in present for i, _ in r["seqs"]})
    local = {i: j for j, i in enumerate(seq_ids)}
    by_bucket_list: dict[int, list[int]] = collections.defaultdict(list)
    for i in seq_ids:
        by_bucket_list[bucket_of[i]].append(local[i])
    by_bucket = {b: np.asarray(m, dtype=np.int64) for b, m in by_bucket_list.items()}
    row_members = [[local[i] for i, _ in r["seqs"]] for r in present]
    max_len = max(len(m) for m in row_members)
    member_matrix = np.full((len(row_members), max_len), -1, dtype=np.int64)
    for k, m in enumerate(row_members):
        member_matrix[k, : len(m)] = m
    valid = member_matrix >= 0
    draws = np.empty(mc_draws, dtype=np.int64)
    for d in range(mc_draws):
        is_late = np.zeros(len(seq_ids) + 1, dtype=bool)
        for b, mem in by_bucket.items():
            x_b = rng.hypergeometric(bucket_late[b], bucket_n[b] - bucket_late[b], len(mem))
            if x_b:
                chosen = rng.choice(len(mem), size=x_b, replace=False)
                is_late[mem[chosen]] = True
        draws[d] = int((is_late[member_matrix] & valid).any(axis=1).sum())
    x = counts["rows_any_sequence_late"]["late"]
    out["rows_any_sequence_late"] = {
        "monte_carlo_bucket_stratified": {
            "draws": mc_draws, "rng_seed": mc_seed,
            "expected": float(draws.mean()),
            "p_ge": float((draws >= x).mean()), "p_le": float((draws <= x).mean()),
            "p_two_sided": float(min(1.0, 2 * min((draws >= x).mean(), (draws <= x).mean()))),
            "min": int(draws.min()), "max": int(draws.max()),
        }
    }
    return out


def planner_null(reader: Any, *, batch_tokens: int, seeds: range, groups: dict[str, list],
                 observed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same counts over plans this planner builds at other seeds (epoch 0, same
    batch_tokens): how unusual F's plan is among the plans _plan produces."""
    samples: dict[str, dict[str, list[int]]] = {
        g: {u: [] for u in observed[g]} for g in groups
    }
    started = time.monotonic()
    for s in seeds:
        plan = reader._plan(batch_tokens=batch_tokens, seed=s, epoch=0)
        pos = [0] * len(reader)
        for p, b in enumerate(plan):
            for i in b.rows:
                pos[i] = p
        late_from = late_positions(len(plan))
        for g, rows in groups.items():
            c = count_units(rows, pos, late_from)
            for u in samples[g]:
                samples[g][u].append(c[u]["late"])
    out: dict[str, Any] = {"seeds": [seeds.start, seeds.stop - 1], "n_plans": len(seeds),
                           "wall_s": round(time.monotonic() - started, 1), "groups": {}}
    for g in groups:
        out["groups"][g] = {}
        for u, xs in samples[g].items():
            x = observed[g][u]["late"]
            n = len(xs)
            out["groups"][g][u] = {
                "mean": sum(xs) / n, "min": min(xs), "max": max(xs),
                "p_ge": sum(1 for v in xs if v >= x) / n,
                "p_le": sum(1 for v in xs if v <= x) / n,
                "seeds_0_1_2_counterfactual": xs[:3] if seeds.start == 0 and n >= 3 else None,
            }
    return out


def consumed_digest(reader: Any, *, batch_tokens: int, seed: int) -> dict[str, Any]:
    """Fold the whole plan through a502670's trainer._fold, as _train_loop does (passes=1, so
    source()'s re-index leaves every index equal to its position)."""
    from qd_train.run_control import ConsumedPrefix
    from qd_train.trainer import _fold

    started = time.monotonic()
    prefix = ConsumedPrefix()
    for position, batch in enumerate(
        reader.batches(batch_tokens=batch_tokens, seed=seed, epoch=0)
    ):
        if int(batch.index) != position:
            raise Refused(f"batch at position {position} has index {batch.index}")
        _fold(prefix, dataclasses.replace(batch, index=position))
    return {"consumed_digest": prefix.hexdigest(), "batches_folded": prefix.n_folded,
            "wall_s": round(time.monotonic() - started, 1)}


# --- main -------------------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    t0 = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--code-root", type=Path, required=True,
                    help="an export of F's commit a502670 (python/ and tools/)")
    ap.add_argument("--out", type=Path, required=True, help="F's pipeline --out (v4 shards)")
    ap.add_argument("--rev", required=True, help="F's --rev, the corpus revision of the shards")
    ap.add_argument("--v3b", type=Path, required=True, help="defect-noul-v3b examples.jsonl")
    ap.add_argument("--ledger", type=Path, required=True, help="F's ledger jsonl")
    ap.add_argument("--result", type=Path, required=True, help="the JSON to write (must not exist)")
    ap.add_argument("--planner-null-seeds", type=int, default=0,
                    help=f"plans at seeds 0..N-1 for the planner null, 0 to skip, <= {MAX_NULL_SEEDS}")
    ap.add_argument("--mc-draws", type=int, default=20_000, help=f"<= {MAX_MC_DRAWS}")
    ap.add_argument("--consumed-digest", action="store_true",
                    help="also fold the whole plan into the trainer's consumed digest")
    args = ap.parse_args(argv)
    if args.result.exists():
        raise Refused(f"--result {args.result} exists; this tool never overwrites a result")
    if not args.result.parent.is_dir():
        raise Refused(f"--result's directory {args.result.parent} does not exist")
    if not 0 <= args.planner_null_seeds <= MAX_NULL_SEEDS:
        raise Refused(f"--planner-null-seeds must be in [0, {MAX_NULL_SEEDS}]")
    if not 1 <= args.mc_draws <= MAX_MC_DRAWS:
        raise Refused(f"--mc-draws must be in [1, {MAX_MC_DRAWS}]")
    code_root = args.code_root.resolve()

    ledger = read_ledger(args.ledger)
    per_seed = f_rows(ledger)
    detail = per_seed[0]["code_that_ran"]
    if any(per_seed[s]["code_that_ran"] != detail for s in per_seed):
        raise Refused("the three ft rows record different code_that_ran")
    code_checks = check_code(code_root, detail)
    anchors = check_anchors(code_root)
    print(f"code root {code_root}: {len(code_checks)} modules match code_that_ran; "
          f"{len(anchors)} anchors present", flush=True)

    sys.path.insert(0, str(code_root / "python"))
    import numpy as np
    from qd_data.config import DataConfig
    from qd_train import shards as shards_mod
    from qd_train.shards import ShardReader

    import qd_data.config as config_mod
    import qd_train.run_control as run_control_mod
    import qd_train.trainer as trainer_mod

    for mod in (shards_mod, config_mod, run_control_mod, trainer_mod):
        if not Path(str(mod.__file__)).resolve().is_relative_to(code_root):
            raise Refused(f"{mod.__name__} imported from {mod.__file__}, not {code_root}")
    config = DataConfig()
    shard_dir = args.out / "shards" / "train"
    reader = ShardReader(shard_dir, config=config, repo_root=args.out, expect_rev=args.rev)
    if reader.sequence_index is None:
        raise Refused(f"{shard_dir} has no sequence index; labels cannot be paired by id")
    t_open = time.monotonic()
    print(f"reader open: {len(reader)} sequences, shard_hash {reader.header.shard_hash()[:16]} "
          f"({t_open - t0:.0f} s, rss {rss_bytes() / 1e9:.2f} GB)", flush=True)

    batch_tokens = int(per_seed[0]["recipe"]["batch_tokens"])
    plan_seed = int(config.seed)
    plan = reader._plan(batch_tokens=batch_tokens, seed=plan_seed, epoch=0)
    n_steps = len(plan)
    pos = [-1] * len(reader)
    bucket_of = [-1] * len(reader)
    for p, b in enumerate(plan):
        for i in b.rows:
            if pos[i] != -1:
                raise Refused(f"sequence {i} appears twice in the plan")
            pos[i] = p
            bucket_of[i] = int(b.bucket)
    padded_positions = sum(len(b.rows) * int(b.width) for b in plan)
    got = {
        "shard_hash": reader.header.shard_hash(),
        "batches": n_steps,
        "width": max(int(b.width) for b in plan),
        "plan_rows": sum(len(b.rows) for b in plan),
        "shard_sequences": len(reader),
        "padding_n_total": padded_positions,
        "padding_n": padded_positions - int(reader.header.total_tokens),
        "every_sequence_once": all(p >= 0 for p in pos),
    }
    identity: dict[str, Any] = {}
    for seed, rec in per_seed.items():
        m, r = rec["metrics"], rec["recipe"]
        expected = {
            "shard_hash": r["shard_hash"],
            "batches": r["batches"],
            "width": r["width"],
            "plan_rows": m["corpus.plan_rows"],
            "shard_sequences": m["corpus.shard_sequences"],
            "padding_n_total": m["train.padding_fraction.n_total"],
            "padding_n": m["train.padding_fraction.n"],
            "every_sequence_once": True,
        }
        consistency = {
            "recipe.batches == corpus.plan_batches": r["batches"] == m["corpus.plan_batches"],
            "recipe.width == corpus.plan_max_width": r["width"] == m["corpus.plan_max_width"],
            "optimizer_steps == micro_batches == batches": (
                m["train.optimizer_steps"] == m["train.micro_batches"] == r["batches"]
            ),
            "passes == 1": r["passes"] == 1,
            "termination == steps_exhausted": m["train.termination"] == "steps_exhausted",
            "no option_permutation_seed in recipe": not rec["recipe_has_option_permutation_seed"],
        }
        checks = {k: {"expected": expected[k], "got": got[k], "ok": expected[k] == got[k]}
                  for k in expected}
        identity[str(seed)] = {
            "ft_row_id": rec["ft_row_id"],
            "checks": checks,
            "row_consistency": consistency,
            "ok": all(c["ok"] for c in checks.values()) and all(consistency.values()),
        }
    result: dict[str, Any] = {
        "what": ("L-recency: the fraction of defect-noul-v3b rows (prose and all) in the last "
                 "10% of F's optimizer steps, beside F's per-seed ood_abstain.prose. Report-only "
                 "(Fable's post-F ruling, AUDIT/post-f-2026-10-02/fable-post-f-ruling.md:30)."),
        "tool": "AUDIT/post-f-2026-10-02/recency/recency_diagnostic.py",
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "argv": argv,
        "analysis_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parent, timeout=60,
        ).stdout.strip() or None,
        "plan_code_commit": F_COMMIT,
        "code_root": str(code_root),
        "code_that_ran_checked": len(code_checks),
        "code_anchors": anchors,
        "plan": {
            "plan_seed": plan_seed,
            "plan_seed_source": "DataConfig().seed (qd_data/config.py); real_ft_run.main at "
                                "a502670 never sets it from --seeds",
            "epoch": 0,
            "batch_tokens": batch_tokens,
            "steps": n_steps,
            "shared_by_seeds": sorted(per_seed),
        },
        "identity": identity,
    }

    def write(obj: dict[str, Any]) -> None:
        obj["wall_s"] = round(time.monotonic() - t0, 1)
        obj["max_rss_bytes"] = rss_bytes()
        obj["platform"] = platform.platform()
        obj["python"] = platform.python_version()
        obj["numpy"] = np.__version__
        body = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(args.result, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, body)
            os.fsync(fd)
        finally:
            os.close(fd)
        print(f"RESULT {args.result} sha256 {hashlib.sha256(body).hexdigest()}", flush=True)

    if not all(v["ok"] for v in identity.values()):
        result["stopped"] = "identity check failed; no fraction computed on a plan that is not F's"
        write(result)
        return 3
    print("identity: all checks pass for seeds 0, 1, 2", flush=True)

    # The corpus.
    if sha256_file(args.v3b) != V3B_SHA256:
        raise Refused(f"{args.v3b} is not the v3b corpus F linked (sha256 {V3B_SHA256})")
    v3b = [json.loads(line) for line in args.v3b.open(encoding="utf-8") if line.strip()]
    by_source = collections.Counter(r["noul_source"] for r in v3b)
    built = build_groups(reader.sequence_index, v3b)
    if built["absent"]:
        raise Refused(f"{len(built['absent'])} v3b row(s) have no sequence in the shard set, "
                      f"e.g. {built['absent'][:3]}")
    all_rows = built["rows"]
    groups = {
        "prose": [r for r in all_rows if r["source"] == "prose"],
        "v3b_all": all_rows,
        "scrambled": [r for r in all_rows if r["source"] == "scrambled"],
        "unseen-language": [r for r in all_rows if r["source"] == "unseen-language"],
    }
    slot_shapes = collections.Counter(
        tuple(sorted(s for _, s in r["seqs"])) for r in all_rows
    )
    v3b_excluded = [{"row_id": r["row_id"], **e} for r in all_rows for e in r["excluded"]]

    late_from = late_positions(n_steps)
    # The smallest i with (i+1)/N > 9/10: one step more than the primary window when 9N/10 is
    # not an integer.
    late_from_alt = (9 * n_steps) // 10
    windows = {
        "primary": {"first_position": late_from, "last_position": n_steps - 1,
                    "steps": n_steps - late_from,
                    "definition": "positions i >= N - floor(N/10): the final floor(N/10) steps"},
        "alternative": {"first_position": late_from_alt, "last_position": n_steps - 1,
                        "steps": n_steps - late_from_alt,
                        "definition": "positions i with (i+1)/N > 0.9"},
    }
    counts = {g: count_units(rows, pos, late_from) for g, rows in groups.items()}
    counts_alt = {g: count_units(rows, pos, late_from_alt) for g, rows in groups.items()}
    alt_differs = {
        g: {u: counts_alt[g][u]["late"] for u in counts[g]
            if counts_alt[g][u]["late"] != counts[g][u]["late"]}
        for g in groups
    }
    refs = {
        g: references(rows, pos, bucket_of, late_from, counts[g], args.mc_draws,
                      mc_seed=20261002 + k)
        for k, (g, rows) in enumerate(groups.items())
    }
    print(f"counts done ({time.monotonic() - t0:.0f} s)", flush=True)

    result.update({
        "corpus": {
            "path": str(args.v3b), "sha256": V3B_SHA256, "rows": len(v3b),
            "rows_by_noul_source": dict(by_source),
            "rows_by_noul_form": dict(collections.Counter(r["noul_form"] for r in v3b)),
            "row_id_form": f"qdm:{DEFECT_FAMILY}:<examples.jsonl id>",
            "rows_absent_from_shard_set": 0,
            "slot_shapes": {"+".join(k): v for k, v in sorted(slot_shapes.items())},
            "v3b_slots_excluded_by_writer": v3b_excluded,
            "sequence_index_exclusions_total": len(reader.sequence_index.excluded),
        },
        "late_window": windows,
        "expected_fraction_by_steps": windows["primary"]["steps"] / n_steps,
        "counts": counts,
        "counts_alternative_window_differs": alt_differs,
        "counts_alternative_window": counts_alt,
        "references": refs,
        "per_seed": {
            str(seed): {
                "ft_row_id": rec["ft_row_id"],
                "eval_row_id": rec["eval_row_id"],
                "ood_abstain.prose": {"n": rec["ood_abstain.prose"]["n"],
                                      "n_total": rec["ood_abstain.prose"]["n_total"]},
                "train.final_loss": rec["metrics"]["train.final_loss"],
                "a_prose_late_fraction": counts["prose"],
                "b_v3b_late_fraction": counts["v3b_all"],
                "plan_identical_to_seeds": sorted(per_seed),
            }
            for seed, rec in per_seed.items()
        },
    })

    if args.planner_null_seeds:
        result["planner_null"] = planner_null(
            reader, batch_tokens=batch_tokens, seeds=range(args.planner_null_seeds),
            groups=groups, observed=counts,
        )
        print(f"planner null done ({time.monotonic() - t0:.0f} s)", flush=True)
    if args.consumed_digest:
        result["consumed_digest"] = consumed_digest(reader, batch_tokens=batch_tokens,
                                                    seed=plan_seed)
        print(f"consumed digest {result['consumed_digest']['consumed_digest']} "
              f"({time.monotonic() - t0:.0f} s)", flush=True)

    result["inputs"] = {
        "ledger": {"path": str(args.ledger), "sha256": sha256_file(args.ledger)},
        "v3b_examples": {"path": str(args.v3b), "sha256": V3B_SHA256},
        **{
            f"shards/train/{name}": {"path": str(shard_dir / name),
                                     "sha256": sha256_file(shard_dir / name)}
            for name in ("header.json", "sequence_index.json", "offsets.npy", "supervision.npz")
        },
        "shards/train/tokens.u32": (
            {"read": "folded into consumed_digest"} if args.consumed_digest
            else {"read": "not read: the plan is a function of lengths (offsets.npy) only"}
        ),
    }
    write(result)
    for seed in sorted(per_seed):
        rec = result["per_seed"][str(seed)]
        pa, pb = rec["a_prose_late_fraction"], rec["b_v3b_late_fraction"]
        print(f"seed {seed}: prose abstain {rec['ood_abstain.prose']['n']}/"
              f"{rec['ood_abstain.prose']['n_total']} | (a) prose rows any-late "
              f"{pa['rows_any_sequence_late']['late']}/{pa['rows_any_sequence_late']['n']}, "
              f"choice seqs {pa['sequences_choice_slot']['late']}/{pa['sequences_choice_slot']['n']}"
              f" | (b) v3b rows any-late {pb['rows_any_sequence_late']['late']}/"
              f"{pb['rows_any_sequence_late']['n']}, choice seqs "
              f"{pb['sequences_choice_slot']['late']}/{pb['sequences_choice_slot']['n']}")
    print(f"DONE in {time.monotonic() - t0:.0f} s, max rss {rss_bytes() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
