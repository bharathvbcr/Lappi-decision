"""The train manifest is the canonical owner of the request shapes each family trained on
(GAP-QD-EXPORT-NO-TRACKED-PER-FAMILY-SLOT-SHAPE-REGISTRY-2026-10-17).

qd-export reads them from the manifest it already binds, so every entry carries its request's
slot shape, spelled as the calibration table keys it (choice:N and score:N count the reserved
noul row, span). The shapes are not in the hashed body: recording them must not move any
data_snapshot_hash the ledger already holds.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from test_manifest import _snapshot

from qd_data.manifest import Manifest, data_snapshot_hash
from qd_data.schema import ChoiceSlot, ScoreSlot, SpanSlot, slot_shape_keys


def _train() -> Manifest:
    return _snapshot(n=24)["train"]


def test_every_entry_records_the_shape_of_its_request() -> None:
    train = _train()
    assert train.entries
    for e in train.entries:
        assert e.slot_shape, f"{e.row_id} records no slot shape"
        assert list(e.slot_shape) == sorted(e.slot_shape)
        for key in e.slot_shape:
            kind, _, rows = key.partition(":")
            assert key == "span" or (kind in ("choice", "score") and rows.isdigit() and int(rows) >= 3)


def test_the_choice_key_counts_the_reserved_noul_row() -> None:
    keys = slot_shape_keys(
        (ChoiceSlot(name="a", options=("x", "y", "z", "w")), SpanSlot(name="s"), ScoreSlot(name="q", bins=4))
    )
    assert keys == ("choice:5", "score:5", "span")


def test_the_shape_is_written_and_read_back(tmp_path: Path) -> None:
    train = _train()
    path = tmp_path / "train.json"
    train.write(path, allow_not_run=True)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert all(isinstance(e["slot_shape"], list) and e["slot_shape"] for e in on_disk["entries"])
    assert Manifest.read(path).entries == train.entries


def test_editing_one_slot_shape_after_writing_is_refused_on_read(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    _train().write(path, allow_not_run=True)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["slot_shape_digest"]["version"] == 1
    doc["entries"][0]["slot_shape"] = ["choice:16"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="slot_shape"):
        Manifest.read(path)


def test_the_digest_has_a_pinned_value() -> None:
    """crates/qd-export/tests/train_manifest_shapes.rs pins the same vector for the Rust reader."""
    train = _train()
    e = dataclasses.replace(train.entries[0], row_id="r1", content_hash="0" * 64, slot_shape=("choice:5", "span"))
    one = dataclasses.replace(train, entries=(e,))
    assert one.slot_shape_digest() == {
        "version": 1,
        "sha256": "ac5cb778fa420b231bf726ea7bf8e99d7d8e8319c29090c075fa4bc2aa6bfa39",
    }


def test_recording_shapes_does_not_move_the_snapshot_hash() -> None:
    snap = _snapshot(n=24)
    stripped = {
        name: dataclasses.replace(
            m, entries=tuple(dataclasses.replace(e, slot_shape=None) for e in m.entries)
        )
        for name, m in snap.items()
    }
    assert data_snapshot_hash(snap) == data_snapshot_hash(stripped)
