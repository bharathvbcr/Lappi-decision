"""Shared plumbing for the Mac GPU-lane benchmarks (``tools/mac_torch_attrib.py``,
``tools/mac_mlx_bench.py``): where the code comes from, and the protocol a row carries.

Only the Protocol is shared. Each tool constructs its own ``RunRecorder`` with
``entry_point=Path(__file__)``, so a row's ``code_that_ran`` is the invoking tool's closure --
this module included, as a sibling import -- rather than this helper's.

The protocol is real, not invented: ``data_snapshot_hash`` and ``tokenizer_hash`` are read off
the header of the shard set the prompts were taken from (``--header``), ``backbone_commit`` is
the snapshot revision and vocab (`real_ft_run._backbone_commit`), and ``recipe_hash`` is
computed by ``real_ft_run._protocol`` itself -- no new spelling of the hash.

``QD_REPO`` names the checkout whose ``python/`` is imported and whose git state is recorded as
``code_commit`` -- a frozen worktree during the GPU lane, so the rows carry a clean commit while
other lanes edit the main tree. It defaults to this file's own repository.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(os.environ.get("QD_REPO", str(Path(__file__).resolve().parents[1]))).resolve()
for _p in (REPO / "python", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qd_train.ledger import Environment, Protocol  # noqa: E402
from qd_train.tristate import Ran  # noqa: E402

QUICK_REASON = (
    "a throughput/parity benchmark: one seed, no training schedule, a fixed item sample. "
    "Rule 8: quick, excluded from every decision"
)


def header_hashes(header: Path) -> tuple[str, str]:
    """``(data_snapshot_hash, tokenizer_hash)`` from a shard set's ``header.json``."""
    raw = json.loads(header.read_text(encoding="utf-8"))
    data, tok = raw.get("data_snapshot_hash"), raw.get("tokenizer_hash")
    if not (isinstance(data, str) and data and isinstance(tok, str) and tok):
        raise SystemExit(f"{header} carries no data_snapshot_hash/tokenizer_hash; refusing")
    return data, tok


def protocol(*, header: Path, recipe: dict[str, Any], seed: int = 0) -> Protocol:
    """The protocol of a benchmark row, built by ``real_ft_run._protocol`` itself.

    Delegated rather than re-spelled: seven tools already hash a recipe and two spellings
    disagree (``test_tool_call_sites``, GAP-SIX-SPELLINGS-OF-ONE-RECIPE-HASH), so an eighth
    implementation would be one more place for the next tidy-up to rename every hash. The
    header is read as JSON rather than through ``ShardReader``: only its two hashes are
    needed, and the reader's code-fingerprint check is about training on the set, which a
    benchmark over its prompts does not do. ``recipe`` must carry ``backbone_snapshot`` and
    ``backbone_vocab``, which ``_backbone_commit`` turns into ``<snapshot>:vocab<N>``.
    """
    data, tok = header_hashes(header)
    ft = importlib.import_module("real_ft_run")
    stand_in = SimpleNamespace(
        header=SimpleNamespace(data_snapshot_hash=data, tokenizer_hash=tok)
    )
    built: Protocol = ft._protocol(reader=stand_in, seed=seed, recipe=recipe)
    return built


def environment(device: str = "mps") -> Environment:
    """The device the run used. ``mps`` by default: torch-MPS and MLX both run on this Mac's
    Metal GPU, which bills nothing. ``tools/teacher_torch_score.py`` passes its own, because it
    also runs on a rented CUDA card and a row claiming ``mps`` there would carry no cost."""
    return Environment.detect(device=device)


def prepare_teacher_items(
    items: list[dict[str, Any]],
    *,
    decode: Callable[[list[int]], str],
    encode: Callable[[str], list[int]],
) -> tuple[list[tuple[str, list[int], list[int], int]], int]:
    """Re-tokenize 2B letter items with a teacher's own tokenizer.

    ``decode`` is the 2B tokenizer's (special tokens kept); ``encode`` is the teacher's
    (no special tokens added). Returns ``(prepared, same_ids)``: each prepared item is
    ``(row_id, teacher_ids, teacher_letter_ids, gold_row)``, and ``same_ids`` counts items whose
    teacher ids equal the 2B ids exactly. An option letter that is not exactly one teacher
    token is refused rather than truncated to its first token, which would score a different
    string than the option.
    """
    prepared = []
    same_ids = 0
    for it in items:
        ids = encode(decode(it["prompt_ids"]))
        same_ids += int(ids == it["prompt_ids"])
        letter_ids = []
        for lid in it["letter_ids"]:
            enc = encode(decode([lid]))
            if len(enc) != 1:
                raise SystemExit(
                    f"item {it['row_id']}: option letter id {lid} encodes to {len(enc)} teacher "
                    "tokens; a letter logit is defined only for a one-token letter"
                )
            letter_ids.append(enc[0])
        prepared.append((it["row_id"], ids, letter_ids, int(it["gold_row"])))
    return prepared, same_ids


def letter_record(*, row_id: str, n_tokens: int, letter_logits: Sequence[float]) -> dict[str, Any]:
    """One item's letter logits and their log-softmax, in the parity-file format.

    The log-softmax is computed here, in float64, so every tool that writes the format
    normalises the same way whatever its framework.
    """
    z = [float(v) for v in letter_logits]
    if not z or not all(math.isfinite(v) for v in z):
        raise SystemExit(f"item {row_id}: letter logits {z[:8]} are empty or not finite")
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return {
        "row_id": row_id,
        "n_tokens": int(n_tokens),
        "letter_logits": z,
        "letter_logsoftmax": [v - lse for v in z],
    }


def read_letter_records(path: Path) -> dict[str, list[float]]:
    """``row_id -> letter_logsoftmax`` from a parity file. Refuses a duplicate or malformed row."""
    out: dict[str, list[float]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        rid, ls = rec.get("row_id"), rec.get("letter_logsoftmax")
        if not isinstance(rid, str) or not isinstance(ls, list) or not ls:
            raise SystemExit(f"{path}:{n}: no row_id or letter_logsoftmax")
        if rid in out:
            raise SystemExit(f"{path}:{n}: row_id {rid!r} appears twice")
        out[rid] = [float(v) for v in ls]
    return out


def letter_divergence(
    ours: Mapping[str, Sequence[float]], ref: Mapping[str, Sequence[float]]
) -> dict[str, float | int]:
    """How far ``ours`` is from ``ref``, over the row ids both scored.

    Both map ``row_id -> log-softmax over the item's letters``. Reports the number compared,
    the max and median per-item max-abs log-prob difference, argmax agreement, and the mean
    and max KL(ref || ours) in nats -- KL is the number that matters for soft labels, since a
    distillation loss is a cross-entropy against the teacher's distribution. Refuses an empty
    overlap and an item whose two sides score a different number of letters.
    """
    shared = sorted(set(ours) & set(ref))
    if not shared:
        raise SystemExit("the two letter files share no row_id; nothing to compare")
    diffs: list[float] = []
    kls: list[float] = []
    agree = 0
    for rid in shared:
        a, b = list(ours[rid]), list(ref[rid])
        if len(a) != len(b):
            raise SystemExit(f"item {rid}: {len(a)} letters on one side, {len(b)} on the other")
        diffs.append(max(abs(x - y) for x, y in zip(a, b, strict=True)))
        kls.append(sum(math.exp(y) * (y - x) for x, y in zip(a, b, strict=True)))
        agree += int(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__))
    return {
        "n": len(shared),
        "n_ours": len(ours),
        "n_ref": len(ref),
        "max_abs_logsoftmax_diff": max(diffs),
        "median_item_max_abs_diff": statistics.median(diffs),
        "argmax_agreement": agree,
        "mean_kl_ref_to_ours": statistics.fmean(kls),
        "max_kl_ref_to_ours": max(kls),
    }


def record_divergence(
    rec: Any, ours: Mapping[str, Sequence[float]], reference: Path, *, ours_label: str
) -> None:
    """Record ``letter_divergence(ours, reference)`` as ``teacher.divergence.*`` metrics.

    Nothing here passes or fails: whether a divergence is small enough is a decision about
    the labels, taken by a person reading these numbers, not a threshold this function owns.
    """
    ref = read_letter_records(reference)
    d = letter_divergence(ours, ref)
    n = int(d["n"])
    where = (
        f"{ours_label} vs {reference.name}, over {n} shared items "
        f"({d['n_ref']} in the reference)"
    )
    rec.metric("teacher.divergence.mean_kl_nats", Ran(
        passed=True, value=round(float(d["mean_kl_ref_to_ours"]), 6), n=n, n_total=int(d["n_ref"]),
        detail=f"mean KL(reference || ours) over each item's letter distribution; {where}"))
    rec.metric("teacher.divergence.max_kl_nats", Ran(
        passed=True, value=round(float(d["max_kl_ref_to_ours"]), 6), n=n, n_total=int(d["n_ref"]),
        detail=f"largest single-item KL(reference || ours); {where}"))
    rec.metric("teacher.divergence.max_abs_logsoftmax_diff", Ran(
        passed=True, value=round(float(d["max_abs_logsoftmax_diff"]), 5), n=n,
        n_total=int(d["n_ref"]),
        detail=f"max per-item max |ours - reference| letter log-prob; median per-item "
               f"{d['median_item_max_abs_diff']:.5f}; {where}"))
    agree = int(d["argmax_agreement"])
    rec.metric("teacher.divergence.argmax_agreement", Ran(
        passed=True, value=agree, n=agree, n_total=n,
        detail=f"{agree} of {n} items give the same top letter; {where}"))


def tool_digest(tool: Path) -> Ran:
    """The invoking tool's bytes, pinned as a metric (it may live outside ``REPO``)."""
    return Ran(
        passed=True,
        value=hashlib.sha256(tool.read_bytes()).hexdigest(),
        detail=f"{tool.name}; qd_train imported from {REPO / 'python'}",
    )


def num(value: float, detail: str) -> Ran:
    return Ran(passed=True, value=round(float(value), 4), detail=detail)
