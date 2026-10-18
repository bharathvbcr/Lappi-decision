"""The val rows' wire requests, the input of ``qd-release-check build-requests``
(GAP-QD-RELEASE-CHECK-NO-TRACKED-REQUEST-BUILDER-2026-10-17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from data_fixtures import small_corpus

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.mixture import build_mixture
from qd_data.schema import slot_shape_keys
from qd_data.split import split
from qd_data.val_requests import write_val_requests


def _rows() -> dict[str, list]:
    config = DataConfig.legacy_v5()
    mixture = build_mixture(small_corpus(24), config=config)
    report = split(dedupe(list(mixture.rows), config=config), config=config)
    return {name: list(rows) for name, rows in report.rows_by_split.items()}


def test_every_val_row_becomes_one_wire_request_and_nothing_else(tmp_path: Path) -> None:
    by_split = _rows()
    val = by_split["val"]
    assert val
    split_of = {r.row_id: s for s, rows in by_split.items() for r in rows}
    out = tmp_path / "val.jsonl"
    assert write_val_requests(val, out, split_of=split_of) == len(val)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [r.request.to_wire() for r in val]
    for line, row in zip(lines, val, strict=True):
        assert "gold" not in line
        assert line["task"] == row.request.task
        assert slot_shape_keys(row.request.slots)
    assert not list(tmp_path.glob("*.partial"))


def test_a_train_row_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    by_split = _rows()
    split_of = {r.row_id: s for s, rows in by_split.items() for r in rows}
    out = tmp_path / "val.jsonl"
    with pytest.raises(ValueError, match="not 'val'"):
        write_val_requests(by_split["val"] + by_split["train"][:1], out, split_of=split_of)
    assert not out.exists()
    with pytest.raises(ValueError, match="None"):
        write_val_requests(by_split["val"][:1], out, split_of={})
    assert not out.exists()
