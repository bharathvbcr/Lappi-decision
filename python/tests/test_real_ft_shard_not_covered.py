"""``tools/real_ft_run.py`` records the not_covered verdict of every shard set it reads.

The train set's verdict rides ``_recorder``; the val set's rides ``open_val_reader`` into the
metrics every row of the run carries. Promotion refuses on either
(``test_promotion_shard_not_covered.py``). Torch-gated, like the other real_ft_run tests: the
tool raises ``SystemExit`` at import without torch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    SHARD_NOT_COVERED_METRIC,
    VAL_SHARD_HASH_METRIC,
    VAL_SHARD_NOT_COVERED_METRIC,
    Ledger,
    Protocol,
)
from qd_train.shards import HEADER_NAME  # noqa: E402
from qd_train.tristate import NotRun, Ran  # noqa: E402

V2_NOT_COVERED = Ran(passed=True, value="manifest.py,val_requests.py")


def _protocol(**_: object) -> Protocol:
    return Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                    backbone_commit="b" * 40, recipe_hash="e" * 64, seed=0)


def _recorder(tmp_path: Path, reader: object):
    return rft._recorder(
        Ledger(tmp_path / "l.jsonl"), reader=reader, seed=0, recipe={"device": "cpu"},
        run_kind="eval", quick_reasons=[], notes="", wall_clock_s=None, cost=None,
    )


def test_the_train_sets_verdict_rides_the_recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rft, "_protocol", _protocol)
    monkeypatch.setattr(rft, "_RUN_METRICS", {})
    rec = _recorder(tmp_path, SimpleNamespace(checks={"shard_code_not_covered": V2_NOT_COVERED}))
    assert rec.metrics[SHARD_NOT_COVERED_METRIC] == V2_NOT_COVERED
    # A reader that recorded no verdict is a NotRun on the row, never an absent (= covered) one.
    absent = _recorder(tmp_path, SimpleNamespace(checks={}))
    assert isinstance(absent.metrics[SHARD_NOT_COVERED_METRIC], NotRun)


def test_opening_the_val_reader_marks_every_later_row_with_the_val_sets_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val_dir = tmp_path / "shards" / "val"
    val_dir.mkdir(parents=True)
    (val_dir / HEADER_NAME).write_text("{}", encoding="utf-8")
    header = SimpleNamespace(
        remap_hash="r" * 64, require_gate_population=lambda where: None,
        shard_hash=lambda: "v" * 64,
    )
    opened = SimpleNamespace(header=header, checks={"shard_code_not_covered": V2_NOT_COVERED})
    monkeypatch.setattr(rft, "ShardReader", lambda *a, **k: opened)
    monkeypatch.setattr(rft, "_protocol", _protocol)
    monkeypatch.setattr(rft, "_RUN_METRICS", {})
    covered = Ran(passed=True, value="")
    train = SimpleNamespace(header=SimpleNamespace(remap_hash="r" * 64),
                            checks={"shard_code_not_covered": covered})

    rft.open_val_reader(tmp_path, config=DataConfig(), rev="x", train=train)  # type: ignore[arg-type]

    rec = _recorder(tmp_path, train)
    assert rec.metrics[VAL_SHARD_NOT_COVERED_METRIC] == V2_NOT_COVERED
    assert rec.metrics[SHARD_NOT_COVERED_METRIC] == covered  # the train set is covered
    # The val set's identity rides beside the verdict, for the byte-identity lift (the ft row's
    # recipe names only the train set).
    assert rec.metrics[VAL_SHARD_HASH_METRIC].value == "v" * 64
