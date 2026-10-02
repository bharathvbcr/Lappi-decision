"""Attest a replay shard set free of 8-gram overlap with every val and held-out set.

The check is RSI-Jev's (``scripts/build_specialist_replay_corpus.py:44-86,158-172``, MIT,
Copyright (c) 2026 Shanghua Gao, @8f34a4f): a replay row is contaminated when it contains at
least half the word 8-grams of some target row, and any hit refuses the replay set. The
logic is ``qd_train.replay.decontaminate``; this tool feeds it and writes the attestation
``tools/real_ft_run.py --replay-shards`` requires.

**Why a separate process.** The targets include the ``heldout`` split, and rule 3 says
held-out data is never read by a training process. This tool trains nothing: it rebuilds
the corpus the way ``real_ft_run.ft_splits`` does, renders every ``val`` and ``heldout`` row,
and compares text. ``real_ft_run`` then re-checks only what it may read -- the ``val``
digest -- and holds the ``heldout`` half to this attestation through the corpus identity
(revision, ``--max-pairs``, ``--commitpackft``) recorded in it.

**Text, not token ids.** Both sides are compared as ``qd_train.replay.prompt_content`` of a
rendered prompt: question, context and option values, never the template (markers, task,
the ``noul`` line), which every row of a family shares. Replay rows arrive as token ids and
are decoded through the shard set's remap and the tokenizer that built it; byte-level BPE
decodes to exactly the text it encoded.

**The hit list (``--hits-out``).** The attestation names at most 50 hits
(``MAX_HIT_EXAMPLES``), so the rows to rebuild without cannot be read off it. ``--hits-out``
writes every (replay sequence, target row) pair at or over the threshold -- each sequence
resolved to its ``(row_id, slot)`` through the shard set's sequence index and to its
``identity_key`` through the replay manifest beside it -- and the distinct identity keys the
pipeline's ``--replay-exclude`` takes. It is a separate file, so the attestation format does
not change.

Usage::

    python tools/replay_decontam.py --out OUT --replay-shards DIR \\
        --tokenizer-json SNAPSHOT/tokenizer.json --attestation-out FILE [--rev REV] \\
        [--hits-out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_git import resolve_rev

from qd_data.config import DataConfig
from qd_data.errors import QdRefusal
from qd_data.general import REPLAY_ONLY
from qd_data.render import render
from qd_data.rows import DataRow
from qd_train.replay import (
    DEFAULT_N,
    DEFAULT_THRESHOLD,
    DecontamReport,
    decontaminate,
    prompt_content,
    write_text_atomic,
)
from qd_train.shards import ShardReader

REPO = Path(__file__).resolve().parents[1]
#: The splits a replay row must not overlap: everything a number is ever reported on.
TARGET_SPLITS: tuple[str, ...] = ("val", "heldout")
#: What a hit list names as its writer; ``tools/real_tokenizer_pipeline.py --replay-exclude``
#: reads only a file that says this.
HITS_TOOL = "tools/replay_decontam.py --hits-out"
#: The manifest the pipeline writes beside the replay shard set, under its ``--out``.
REPLAY_MANIFEST = Path("data") / "pool" / "train-replay.json"


def row_texts(rows: list[DataRow]) -> tuple[dict[str, str], dict[str, str]]:
    """``({row_id#slot: content}, {row_id: refusal})`` over every row, rendered as served.

    A row ``render`` refuses -- a context over ``RenderCaps.max_context_bytes``, say -- is
    counted in the second map and skipped, never a crash: ``write_shards`` refuses the
    same row, so it is in no shard set and no number is reported on it, and there is
    nothing for a replay row to leak into. Measured on the default corpus: 4 of 50 val
    rows at 146,009 bytes (GAP-REPLAY-DECONTAM-TOOL-CRASHES-ON-UNRENDERABLE-FT-VAL-ROW).
    """
    out: dict[str, str] = {}
    unrenderable: dict[str, str] = {}
    for row in rows:
        try:
            rendered = render(row.request, seed=None)
        except QdRefusal as exc:
            unrenderable[row.row_id] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        for slot in rendered.slots:
            out[f"{row.row_id}#{slot.name}"] = prompt_content(rendered.prompt_for(slot.name))
    return out, unrenderable


def target_texts(
    splits: dict[str, list[DataRow]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Every target split's texts, and its unrenderable rows. Fails closed on a split
    that has rows and renders none: a comparison against nothing would attest clean."""
    texts: dict[str, dict[str, str]] = {}
    skipped: dict[str, dict[str, str]] = {}
    for name in TARGET_SPLITS:
        texts[name], skipped[name] = row_texts(splits[name])
        if splits[name] and not texts[name]:
            raise SystemExit(
                f"target {name!r}: all {len(splits[name])} row(s) are unrenderable, so "
                "nothing could be compared against it"
            )
    return texts, skipped


def replay_texts(reader: ShardReader, tokenizer_json: Path) -> dict[str, str]:
    """``{sequence index: content}`` for every sequence of the replay shard set."""
    index = reader.sequence_index
    if index is None or index.role != REPLAY_ONLY:
        raise SystemExit(
            f"{reader.root}: not a replay shard set (sequence index role "
            f"{index.role if index is not None else None!r}); only a set written with "
            "write_shards(replay=True) can be attested as replay"
        )
    if reader.remap is None:
        raise SystemExit(
            f"{reader.root}: no remap table beside the shards, so its ids cannot be turned "
            "back into text and nothing about it can be attested"
        )
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(str(tokenizer_json))
    new_to_old = reader.remap.new_to_old
    out: dict[str, str] = {}
    for i in range(len(reader)):
        source = [int(new_to_old[int(t)]) for t in reader.sequence(i)]
        out[str(i)] = prompt_content(tok.decode(source, skip_special_tokens=False))
    return out


def replay_identity_keys(out: Path, *, data_snapshot_hash: str) -> dict[str, str]:
    """``{row_id: identity_key}`` from the replay manifest beside the shards.

    Refused unless the manifest is the one the replay shard set was written from: its
    ``data_snapshot_hash`` is the one the shard header pins.
    """
    path = out / REPLAY_MANIFEST
    if not path.is_file():
        raise SystemExit(f"{path} is absent, so no replay row can be named by identity key")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("data_snapshot_hash") != data_snapshot_hash:
        raise SystemExit(
            f"{path} records data_snapshot_hash {raw.get('data_snapshot_hash')!r} but the "
            f"replay shard header pins {data_snapshot_hash!r}: not this set's manifest"
        )
    return {str(e["row_id"]): str(e["identity_key"]) for e in raw["entries"]}


def hit_list(
    report: DecontamReport,
    *,
    sequences: Sequence[tuple[str, str]],
    identity_of: Mapping[str, str],
    replay_shard_hash: str,
    corpus: Mapping[str, Any],
    attestation_sha256: str,
) -> dict[str, Any]:
    """The ``--hits-out`` body: every pair, each replay sequence resolved to its row.

    ``report``'s replay ids are sequence indices (as :func:`replay_texts` keys them);
    ``sequences[i]`` is sequence ``i``'s ``(row_id, slot_name)``. A hit row the replay
    manifest does not name stops the tool: an exclusion keyed by a guess is no exclusion.
    """
    pairs: list[dict[str, Any]] = []
    for p in report.pairs:
        seq = int(p.replay_row)
        row_id, slot = sequences[seq]
        if row_id not in identity_of:
            raise SystemExit(
                f"replay sequence {seq} is row {row_id!r}, which the replay manifest does not "
                "name; refusing to write a hit list with a row it cannot key"
            )
        pairs.append({
            "sequence": seq, "row_id": row_id, "slot_name": slot,
            "identity_key": identity_of[row_id], "target": p.target,
            "target_row": p.target_row, "shared": p.shared, "target_ngrams": p.target_ngrams,
            "containment": p.containment,
        })
    pairs.sort(key=lambda d: (d["sequence"], d["target"], d["target_row"]))
    return {
        "tool": HITS_TOOL,
        "version": 1,
        "n": report.n,
        "threshold": report.threshold,
        "replay_shard_hash": replay_shard_hash,
        "attestation_sha256": attestation_sha256,
        "corpus": dict(corpus),
        "hits": dict(report.hits),
        # h: the distinct replay rows with at least one pair (= rows hit in any target set).
        "replay_rows_hit": len({d["sequence"] for d in pairs}),
        "identity_keys": sorted({d["identity_key"] for d in pairs}),
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the pipeline's --out")
    parser.add_argument("--replay-shards", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--attestation-out", type=Path, required=True)
    parser.add_argument("--rev", default="0632f693d3b765b726499e7b4bf19c67959b75cb")
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--commitpackft", type=Path, default=None)
    parser.add_argument("--defect-class", type=Path, default=None)
    parser.add_argument("--defect-download", type=Path, default=None)
    parser.add_argument("--defect-max-rows", type=int, default=None)
    parser.add_argument("--defect-noul", type=Path, default=None)
    parser.add_argument(
        "--general-record", type=Path, default=None,
        help="as the pipeline's --general-record: its general families are val and held-out "
             "targets too, so an attestation made without it compares against fewer rows",
    )
    parser.add_argument("--general-max-rows", type=int, default=None)
    parser.add_argument(
        "--no-repo-history", dest="repo_history", action="store_false",
        help="as the pipeline's and real_ft_run.py's --no-repo-history",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--hits-out", type=Path, default=None,
        help="also write every (replay sequence, target row) pair at or over the threshold, "
             "with each sequence's row_id, slot and identity_key, to this file: what the "
             "pipeline's --replay-exclude reads. The attestation is unchanged by it",
    )
    args = parser.parse_args(argv)
    for path in (args.attestation_out, args.hits_out):
        if path is not None and path.exists():
            raise SystemExit(f"{path} already exists; refusing to overwrite it")

    from real_ft_run import check_defect_source, ft_splits, replay_corpus_identity

    config = DataConfig()
    rev = resolve_rev(REPO, args.rev)
    check_defect_source(args.out, defect_class=args.defect_class)
    splits = ft_splits(
        commitpackft=args.commitpackft, max_pairs=args.max_pairs, rev=rev, config=config,
        defect_class=args.defect_class, defect_download=args.defect_download,
        defect_max_rows=args.defect_max_rows, repo_history=args.repo_history,
        general_record=args.general_record, general_max_rows=args.general_max_rows,
        defect_noul=args.defect_noul,
    )
    targets, unrenderable = target_texts(splits)
    reader = ShardReader(args.replay_shards, config=config, repo_root=args.out)
    identity_of: dict[str, str] | None = None
    if args.hits_out is not None:
        # Read before anything is compared or written, so a missing or foreign manifest
        # stops the run rather than leaving an attestation with no hit list beside it.
        identity_of = replay_identity_keys(
            args.out, data_snapshot_hash=reader.header.data_snapshot_hash
        )
    replay = replay_texts(reader, args.tokenizer_json)
    report = decontaminate(replay, targets, n=args.n, threshold=args.threshold)
    body = {
        **report.to_json(),
        "tool": "tools/replay_decontam.py",
        # Target rows render refuses: in no shard set, never scored, so not compared.
        "unrenderable_targets": {
            name: {"n": len(rows), "rows": dict(sorted(rows.items()))}
            for name, rows in unrenderable.items()
        },
        "replay_shard_hash": reader.header.shard_hash(),
        "corpus": replay_corpus_identity(
            rev=rev, max_pairs=args.max_pairs, commitpackft=args.commitpackft,
            defect_class=args.defect_class, defect_max_rows=args.defect_max_rows,
            repo_history=args.repo_history, general_record=args.general_record,
            general_max_rows=args.general_max_rows, defect_noul=args.defect_noul,
        ),
    }
    attestation_text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    write_text_atomic(args.attestation_out, attestation_text)
    if args.hits_out is not None and identity_of is not None:
        index = reader.sequence_index
        if index is None:  # replay_texts already refused this; kept for the type
            raise SystemExit(f"{reader.root}: no sequence index")
        hits_body = hit_list(
            report, sequences=index.sequences, identity_of=identity_of,
            replay_shard_hash=reader.header.shard_hash(), corpus=body["corpus"],
            attestation_sha256=hashlib.sha256(attestation_text.encode("utf-8")).hexdigest(),
        )
        hits_text = json.dumps(hits_body, indent=2, sort_keys=True) + "\n"
        write_text_atomic(args.hits_out, hits_text)
        print(
            f"hit list: {len(hits_body['pairs'])} pair(s) over {hits_body['replay_rows_hit']} "
            f"replay row(s), {len(hits_body['identity_keys'])} identity key(s) -> "
            f"{args.hits_out} sha256 {hashlib.sha256(hits_text.encode('utf-8')).hexdigest()}"
        )
    print(
        f"replay {reader.header.shard_hash()[:16]}: {report.replay_rows_checked} of "
        f"{report.replay_rows_total} rows checked ({report.replay_rows_too_short} too short "
        f"for a {args.n}-gram), against "
        + ", ".join(
            f"{name} {t['rows_indexed']}/{t['rows']} rows" for name, t in report.targets.items()
        )
        + "; unrenderable targets skipped: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in unrenderable.items())
    )
    print(f"hits per target: {report.hits} -> {'CLEAN' if report.clean else 'CONTAMINATED'}")
    print(f"attestation: {args.attestation_out}")
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
