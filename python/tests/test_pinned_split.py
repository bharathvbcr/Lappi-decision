"""A source that pins its splits decides the row's split, not the repo hash.

``cais/mmlu`` publishes test/dev/validation splits and ``Source.pinned_splits`` maps them to
ours (test and dev train, validation val); ``general.rewrite_mmlu`` records the result as
``metadata[PINNED_SPLIT_KEY]``. ``split()`` ignored it and hashed every MMLU subject by
``repo_key``, so validation subjects landed in train. A pinned value outside ``SPLITS`` is
refused rather than defaulted.
"""

from __future__ import annotations

import dataclasses

import pytest
from data_fixtures import commitpackft_row

from qd_data.config import SPLITS, DataConfig
from qd_data.dedupe import dedupe
from qd_data.mixture import build_mixture
from qd_data.sources import PINNED_SPLIT_KEY
from qd_data.split import assign_repo, split


def _rows() -> list:
    config = DataConfig()
    mixture = build_mixture(
        {"bigcode/commitpackft": [commitpackft_row(i) for i in range(30)]}, config=config,
        families=["code.commit_intent"],
    )
    return list(mixture.rows)


def _pin(row, value: str):
    return dataclasses.replace(row, metadata={**row.metadata, PINNED_SPLIT_KEY: value})


def test_a_pinned_split_overrides_the_repo_hash() -> None:
    config = DataConfig()
    rows = _rows()
    hashed = {
        r.row_id: assign_repo(r.repo_key, seed=config.seed,
                              train_fraction=config.train_fraction,
                              val_fraction=config.val_fraction)
        for r in rows
    }
    target = next(r for r in rows if hashed[r.row_id] == "train")
    pinned = [_pin(r, "val") if r.row_id == target.row_id else r for r in rows]
    report = split(dedupe(pinned, config=config), config=config)
    where = {a.row_id: a for a in report.assignments}
    assert where[target.row_id].split == "val"
    assert where[target.row_id].repo_split == "val"
    # Unpinned rows are untouched.
    for r in rows:
        if r.row_id != target.row_id:
            assert where[r.row_id].split == hashed[r.row_id]


def test_a_pinned_value_outside_splits_is_refused() -> None:
    config = DataConfig()
    rows = _rows()
    bad = [_pin(rows[0], "validation"), *rows[1:]]
    assert "validation" not in SPLITS
    with pytest.raises(ValueError, match="is not one"):
        split(dedupe(bad, config=config), config=config)
