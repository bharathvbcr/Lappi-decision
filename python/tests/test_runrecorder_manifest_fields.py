"""``RunRecorder`` carries the train-manifest facts into the terminal row it writes.

W2-1 added ``LedgerRow.train_manifest_sha256`` and ``LedgerRow.val_without_train`` and no
producer wrote them. The recorder is the one writer every ft tool goes through, so it takes
both as constructor arguments and puts them on the row it appends on every exit path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qd_train.ledger import (
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    RunRecorder,
    _population_paired_margin,
)

REPO = Path(__file__).resolve().parents[2]
SHA = hashlib.sha256(b"a train manifest file").hexdigest()
FAMILY = "synth.general/access_route"
BLOCK = {"count": 1, "families": {FAMILY: 3}}


def _recorder(tmp_path: Path, **kw: object) -> RunRecorder:
    proto = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                     backbone_commit="b" * 40, recipe_hash="r" * 64, seed=0)
    return RunRecorder(
        Ledger(tmp_path / "ledger.jsonl"), protocol=proto, run_kind="ft", repo=REPO,
        env=Environment.detect(device="cpu"), wall_clock_s=1.0, cost=None,
        **kw,  # type: ignore[arg-type]
    )


def test_the_terminal_row_carries_both_fields(tmp_path: Path) -> None:
    rec = _recorder(tmp_path, train_manifest_sha256=SHA, val_without_train=BLOCK)
    with rec:
        pass
    assert rec.row is not None
    assert rec.row.train_manifest_sha256 == SHA
    assert rec.row.val_without_train == BLOCK
    on_disk = Ledger(tmp_path / "ledger.jsonl")
    on_disk.verify_chain()
    (row,) = on_disk.rows()
    assert row.train_manifest_sha256 == SHA and row.val_without_train == BLOCK
    assert json.loads(on_disk.raw_lines()[-1])["train_manifest_sha256"] == SHA


def test_a_failed_run_still_carries_them(tmp_path: Path) -> None:
    rec = _recorder(tmp_path, train_manifest_sha256=SHA)
    with pytest.raises(RuntimeError), rec:
        raise RuntimeError("boom")
    assert rec.row is not None and rec.row.status == "failed"
    assert rec.row.train_manifest_sha256 == SHA


def test_a_recorder_given_neither_writes_neither_key(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    with rec:
        pass
    line = json.loads(Ledger(tmp_path / "ledger.jsonl").raw_lines()[-1])
    assert "train_manifest_sha256" not in line and "val_without_train" not in line


@pytest.mark.parametrize("bad", ["abc", "A" * 64, ""])
def test_a_bad_digest_is_refused_at_construction_not_after_the_run(
    tmp_path: Path, bad: str
) -> None:
    with pytest.raises(ValueError, match="train_manifest_sha256"):
        _recorder(tmp_path, train_manifest_sha256=bad)


def test_the_population_reading_names_a_family_from_a_recorded_row(tmp_path: Path) -> None:
    rec = _recorder(tmp_path, val_without_train=BLOCK)
    with rec:
        pass
    assert rec.row is not None
    assert rec.row.families_without_train() == {FAMILY}
    assert callable(_population_paired_margin)
    assert LedgerRow.from_json(rec.row.to_json()).val_without_train == BLOCK
