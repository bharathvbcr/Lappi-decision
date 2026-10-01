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

Usage::

    python tools/replay_decontam.py --out OUT --replay-shards DIR \\
        --tokenizer-json SNAPSHOT/tokenizer.json --attestation-out FILE [--rev REV]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    decontaminate,
    prompt_content,
    write_text_atomic,
)
from qd_train.shards import ShardReader

REPO = Path(__file__).resolve().parents[1]
#: The splits a replay row must not overlap: everything a number is ever reported on.
TARGET_SPLITS: tuple[str, ...] = ("val", "heldout")


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
    args = parser.parse_args(argv)
    if args.attestation_out.exists():
        raise SystemExit(f"{args.attestation_out} already exists; refusing to overwrite it")

    import json

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
    write_text_atomic(args.attestation_out, json.dumps(body, indent=2, sort_keys=True) + "\n")
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
