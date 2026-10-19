"""``tools/train_examples_export.py``: the text file ``qd-train-data --examples`` reads.

A manifest that disagrees with itself is refused, the written lines are the shape
``qd_prep::pool::read_example_texts`` parses (id, source_id, family_id, context, options), and an
existing output is never overwritten. The rebuild itself (``ft_split_report``) is heavy and is
exercised by the build that uses this tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))


def _manifest(tmp_path: Path, n_rows: int, ids: list[str]) -> Path:
    path = tmp_path / "train.json"
    path.write_text(json.dumps({
        "n_rows": n_rows,
        "entries": [
            {"row_id": i, "source_id": f"src-{k}", "family_id": "fam"} for k, i in enumerate(ids)
        ],
    }), encoding="utf-8")
    return path


def test_manifest_ids_maps_rows_to_sources(tmp_path: Path) -> None:
    from train_examples_export import _manifest_ids

    assert _manifest_ids(_manifest(tmp_path, 2, ["a", "b"])) == {
        "a": ("src-0", "fam"), "b": ("src-1", "fam"),
    }


def test_manifest_with_wrong_row_count_is_refused(tmp_path: Path) -> None:
    from train_examples_export import _manifest_ids

    with pytest.raises(SystemExit, match="n_rows 3 but 2 entries"):
        _manifest_ids(_manifest(tmp_path, 3, ["a", "b"]))


def test_written_lines_carry_the_fields_qd_train_data_reads(tmp_path: Path) -> None:
    from train_examples_export import _write

    path = tmp_path / "x.jsonl"
    digest = _write(path, [("r1#slot", "clinc/clinc_oos", "intent.domain", "what is the time")])
    (line,) = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {
        "id": "r1#slot", "source_id": "clinc/clinc_oos", "family_id": "intent.domain",
        "context": "what is the time", "options": [],
    }
    import hashlib

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_pick_keeps_input_order_and_only_manifest_rows() -> None:
    from train_examples_export import pick

    rows = [
        ("z#s", "i", "f", "t1"), ("a#s", "i", "f", "t2"), ("skip#s", "i", "f", "t3"),
        ("m#s2", "i", "f", "t4"), ("a#s2", "i", "f", "t5"),
    ]
    got = pick(rows, {"z", "a", "m"}, {"z": "S1", "a": "S2", "m": "S3"})
    assert [(k, s) for k, s, _f, _t in got] == [
        ("z#s", "S1"), ("a#s", "S2"), ("m#s2", "S3"), ("a#s2", "S2"),
    ]


def _shard(tmp_path: Path) -> Path:
    import numpy as np

    d = tmp_path / "shard"
    d.mkdir()
    seqs = [("r0", "s"), ("c1", "s"), ("r2", "s"), ("c1", "t")]
    (d / "sequence_index.json").write_text(json.dumps({"sequences": [
        {"row_id": r, "slot_kind": 1, "slot_name": s} for r, s in seqs
    ]}), encoding="utf-8")
    np.save(d / "offsets.npy", np.array([0, 2, 4, 5, 7], dtype=np.int64))
    np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint32).tofile(d / "tokens.u32")
    np.savez(d / "remap.npz", new_to_old=np.arange(100, 110, dtype=np.int32))
    return d


def test_decode_shard_rows_decodes_only_the_named_rows_in_sequence_order(tmp_path: Path) -> None:
    from train_examples_export import decode_shard_rows

    got = decode_shard_rows(_shard(tmp_path), {"c1"}, lambda ids: "-".join(map(str, ids)))
    assert got == [("c1#s", "103-104"), ("c1#t", "106-107")]


def test_a_derived_row_with_no_sequence_is_refused(tmp_path: Path) -> None:
    from train_examples_export import decode_shard_rows

    with pytest.raises(SystemExit, match="no sequence"):
        decode_shard_rows(_shard(tmp_path), {"c1", "ghost"}, lambda ids: "x")


def test_pick_reads_the_row_id_before_the_last_hash_so_a_row_id_may_hold_one() -> None:
    """A code row's id ends ``...path.py#0``; its slot key is ``<id>#<slot>``. Splitting at the
    first ``#`` found no slot for 43,466 rows of the v6 rebuild (W2-d)."""
    from train_examples_export import pick

    rows = [("qdm:x:a.py#0#defect_span", "i", "f", "t1"), ("qdm:x:b.py#1#choice", "i", "f", "t2")]
    got = pick(rows, {"qdm:x:a.py#0"}, {"qdm:x:a.py#0": "S"})
    assert got == [("qdm:x:a.py#0#defect_span", "S", "f", "t1")]


def test_an_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    from train_examples_export import _write

    path = tmp_path / "x.jsonl"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _write(path, [("k", "s", "f", "t")])
    assert path.read_text(encoding="utf-8") == "keep"
