"""Write the text of a built train shard set's rows as ``examples.jsonl`` files, for
``qd-train-data --examples``.

A shard set holds token ids, not text (GAP-LDATA-TRAIN-DATA-GUARD-SHARD-SETS-HAVE-NO-TEXT-2026-10-17),
so ``qd-train-data``'s n-gram and exact-short tiers need the rows' text from a rebuild. This
rebuilds the split exactly as ``tools/containment_scan.py`` does (``real_ft_run.ft_split_report``
over the same corpus flags, under the default v6 data rules), renders every slot of every train
row, and keeps the ones whose row id is in the built train manifest (``data/pool/train.json``).

Two files, one per text definition, each one line per slot keyed ``row_id#slot``:

* ``stripped.jsonl``: the scan's own definition (``qd_train.containment_strip.strip_template``
  over train, val and held-out together: constant question lines and family-constant options
  removed; what is left of an intent.* row is its utterance);
* ``unstripped.jsonl``: ``prompt_content``'s text, byte for byte, which is the stricter reading.

Every manifest row id must be found in the rebuilt train split, or the tool refuses and names
the count: a text file that does not cover the shard set's rows is not evidence about it. A row
whose id contains ``:contrast:`` is derived after the split from a surviving twin
(``qd_train.contrast``); it has no row of its own in the rebuild, so its text is decoded from the
token ids of the written shard set (``--shard-dir``), in both files, unstripped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from containment_scan import REPO, TRAIN, _rendered, scan_sets
from repo_git import resolve_rev

from qd_data.config import DataConfig

def _manifest_ids(path: Path) -> dict[str, tuple[str, str]]:
    """``{row_id: (source_id, family_id)}`` of the train manifest, in its order."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    entries = doc["entries"]
    if doc["n_rows"] != len(entries):
        raise SystemExit(f"{path}: n_rows {doc['n_rows']} but {len(entries)} entries")
    return {e["row_id"]: (e["source_id"], e["family_id"]) for e in entries}


def decode_shard_rows(
    shard_dir: Path, row_ids: set[str], decode: Callable[[np.ndarray], str],
) -> list[tuple[str, str]]:
    """``(row_id#slot, text)`` of every sequence of ``row_ids`` in a written shard set, decoded
    from the token ids on disk (through the remap's ``new_to_old``), in sequence order.

    For rows derived after the split (v5/v6 contrast rows, ``qd_train.contrast``): they have no
    row in the rebuilt split, but their exact training text is in the shard. Refuses a row id
    that has no sequence there, so a derived row can never be silently left unscanned."""
    index = json.loads((shard_dir / "sequence_index.json").read_text(encoding="utf-8"))
    sequences = index["sequences"]
    offsets = np.load(shard_dir / "offsets.npy")
    if len(offsets) != len(sequences) + 1:
        raise SystemExit(f"{shard_dir}: {len(offsets)} offsets for {len(sequences)} sequences")
    tokens = np.memmap(shard_dir / "tokens.u32", dtype=np.uint32, mode="r")
    new_to_old = np.load(shard_dir / "remap.npz")["new_to_old"]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, seq in enumerate(sequences):
        if seq["row_id"] not in row_ids:
            continue
        seen.add(seq["row_id"])
        ids = new_to_old[np.asarray(tokens[int(offsets[i]) : int(offsets[i + 1])], dtype=np.int64)]
        out.append((f"{seq['row_id']}#{seq['slot_name']}", decode(ids)))
    absent = sorted(row_ids - seen)
    if absent:
        raise SystemExit(f"{len(absent)} derived row(s) have no sequence in {shard_dir}; "
                         f"first {absent[:5]}")
    return out


def _write(path: Path, lines: list[tuple[str, str, str, str]]) -> str:
    import hashlib

    body = "".join(
        json.dumps(
            {"id": key, "source_id": source, "family_id": family, "context": text,
             "options": []},
            ensure_ascii=False,
        ) + "\n"
        for key, source, family, text in lines
    ).encode("utf-8")
    with path.open("xb") as fh:
        fh.write(body)
    return hashlib.sha256(body).hexdigest()


def pick(
    rows: Sequence[tuple[str, str, str, str]], found: set[str], source_of: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """``(key, source_id, family, text)`` of the slots whose row id (the key before ``#``) is in
    ``found``, in the order of ``rows`` -- the split's own order, never a set's or a completion
    order -- so two runs write the same bytes."""
    out: list[tuple[str, str, str, str]] = []
    for key, _identity, family, text in rows:
        rid = key.split("#", 1)[0]
        if rid in found:
            out.append((key, source_of[rid], family, text))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument(
        "--shard-dir", type=Path, required=True,
        help="the written train shard set: rows derived after the split (contrast rows) have no "
             "row in the rebuild, so their text is decoded from its token ids",
    )
    p.add_argument("--rev", required=True)
    p.add_argument("--defect-class", type=Path, default=None)
    p.add_argument("--defect-download", type=Path, default=None)
    p.add_argument("--defect-noul", type=Path, default=None)
    p.add_argument("--general-record", type=Path, default=None)
    p.add_argument("--general-max-rows", type=int, default=None)
    p.add_argument("--decisions-pool", type=Path, default=None)
    p.add_argument("--drop-before-dedupe", type=Path, default=None, dest="pre_dedupe_drops")
    p.add_argument("--no-repo-history", dest="repo_history", action="store_false")
    args = p.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; refusing to overwrite it")
    wanted = _manifest_ids(args.train_manifest)

    from real_ft_run import ft_split_report
    from real_tokenizer_pipeline import DEFAULT_MAX_PAIRS, data_rules

    config = data_rules(DataConfig(), v6=True)
    report = ft_split_report(
        commitpackft=None, max_pairs=DEFAULT_MAX_PAIRS, rev=resolve_rev(REPO, args.rev),
        config=config, defect_class=args.defect_class,
        defect_download=args.defect_download, defect_max_rows=None,
        repo_history=args.repo_history, general_record=args.general_record,
        general_max_rows=args.general_max_rows, defect_noul=args.defect_noul,
        decisions_pool=args.decisions_pool, pre_dedupe_drops=args.pre_dedupe_drops,
    )
    train_rows = tuple(report.rows_by_split.get(TRAIN, ()))
    source_of = {r.row_id: r.source_id for r in train_rows}
    found = {rid for rid in wanted if rid in source_of}
    derived = {rid for rid in wanted if rid not in found and ":contrast:" in rid}
    missing = sorted(rid for rid in wanted if rid not in found and rid not in derived)
    print(f"manifest rows {len(wanted)}; in the rebuilt train split {len(found)}; "
          f"derived (contrast) rows {len(derived)}; missing {len(missing)}")
    if missing:
        raise SystemExit(f"{len(missing)} manifest row(s) are neither in the rebuilt train "
                         f"split nor derived contrast rows; first {missing[:5]}")
    sets, _record = scan_sets(report, config=config, template_strip=True)
    stripped = next(s for s in sets if s.name == TRAIN)
    unstripped, _refused = _rendered(train_rows)

    a = pick(stripped.rows, found, source_of)
    b = pick([
        (key, identity, family, parts.joined())
        for key, identity, family, _slot, parts in unstripped
    ], found, source_of)
    covered = {k.split("#", 1)[0] for k, *_ in a}
    uncovered = sorted(found - covered)
    print(f"stripped slots {len(a)} over {len(covered)} rows; unstripped slots {len(b)}; "
          f"found rows with no rendered slot {len(uncovered)}")
    if uncovered:
        raise SystemExit(f"{len(uncovered)} manifest row(s) rendered no slot; "
                         f"first {uncovered[:5]}")
    if derived:
        from real_tokenizer_pipeline import RealTokenizer

        tok = RealTokenizer.load(memo_limit=0)
        extra = decode_shard_rows(args.shard_dir, derived, lambda ids: tok.decode(ids))
        # No template strip applies to a decoded row: both files carry its whole text.
        for key, text in extra:
            rid = key.split("#", 1)[0]
            line = (key, wanted[rid][0], wanted[rid][1], text)
            a.append(line)
            b.append(line)
        print(f"derived rows decoded from {args.shard_dir}: {len(extra)} slots over "
              f"{len(derived)} rows")
    args.out_dir.mkdir(parents=True)
    h1 = _write(args.out_dir / "stripped.jsonl", a)
    h2 = _write(args.out_dir / "unstripped.jsonl", b)
    print(f"stripped.jsonl sha256 {h1}\nunstripped.jsonl sha256 {h2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
