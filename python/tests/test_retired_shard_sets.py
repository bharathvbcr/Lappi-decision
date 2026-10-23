"""The retired v5r pair cannot be opened: no new read (Fable 2026-10-22, LoRA-2B amendment 3).

``RETIRED_SHARD_SETS`` is checked in ``ShardHeader.from_json``, the door ``ShardReader`` and
every header-parsing tool go through. The tests build a header, retire its hash, and ask the
door; and they hold the code's list equal to the repo-side retirement record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from qd_train import artifacts  # noqa: E402
from qd_train.artifacts import ShardContractViolation, ShardHeader  # noqa: E402

RECORD = REPO / "AUDIT" / "x5-v5r-retire-2026-10-22" / "retirement.json"
REASON = "retired: contaminated (LoRA-2B amendment 3)"


def _header() -> ShardHeader:
    return ShardHeader(
        split="train",
        data_snapshot_hash="a" * 64,
        tokenizer_hash="b" * 64,
        remap_hash="c" * 64,
        vocab_size=128,
        n_sequences=4,
        total_tokens=40,
        max_seq_len=16,
        buckets=(16,),
    )


def test_a_retired_hash_is_refused_at_the_header_door(monkeypatch: pytest.MonkeyPatch) -> None:
    header = _header()
    raw = header.to_json()
    assert ShardHeader.from_json(raw).shard_hash() == header.shard_hash()  # open until retired
    monkeypatch.setitem(artifacts.RETIRED_SHARD_SETS, header.shard_hash(), REASON)
    with pytest.raises(ShardContractViolation, match="retired: contaminated"):
        ShardHeader.from_json(raw)


def test_the_code_list_is_the_record() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert {s["shard_hash"] for s in record["sets"]} == set(artifacts.RETIRED_SHARD_SETS)
    assert {s["retired"] for s in record["sets"]} == {REASON.removeprefix("retired: ")}
    assert set(artifacts.RETIRED_SHARD_SETS.values()) == {REASON}
    assert record["status"] == REASON
    for s in record["sets"]:
        assert s["shard_hash"].startswith(s["short"])
    assert {s["short"] for s in record["sets"]} == {
        "108082545a66", "a485e7f7056d", "e8cffbdd9574"
    }
    assert {s["data_snapshot_hash"] for s in record["sets"]} == set(
        artifacts.RETIRED_DATA_SNAPSHOTS
    )
    assert set(artifacts.RETIRED_DATA_SNAPSHOTS.values()) == {REASON}


def test_the_v5r_sets_are_named_by_their_full_hashes() -> None:
    """The pair (Fable's first ruling) and train-no-decode (ruling (a), same day)."""
    assert set(artifacts.RETIRED_SHARD_SETS) == {
        "108082545a66ec3e0b008f8ad7a8425625cd9bc323630f8d0cd186f40e00cdf4",
        "a485e7f7056d22b9f7e6baf8f19ae64b313db9f3b77b504a094ff1caae3f5d0f",
        "e8cffbdd95747bf5e1dfe4e5505d5aa7b34fb81b229ce65d7b238373d06927f9",
    }


def test_the_p1_queue_reads_v6_and_is_held_by_an_unfilled_placeholder() -> None:
    queue = json.loads(
        (REPO / "campaign" / "next-train-first-box-queue-2026-10-08.json").read_text()
    )
    argv = [a for ph in queue["phases"] for u in ph["units"] for a in u["argv"]]
    assert not any("v5_data_dir" in a or "drop_before_dedupe" in a for a in argv)
    assert "--drop-before-dedupe" not in argv
    assert "{v6_data_dir}" in argv
    assert queue["hold"]["status"] == "held"
    # The driver refuses an unfilled placeholder, so the hold is enforced, not advisory.
    assert queue["placeholders"]["v6_shards_verified"] is None


def test_a_header_from_a_retired_data_snapshot_is_refused_whatever_its_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-hashed copy (any covered field changed) or a rebuild from the same snapshot."""
    snapshot = "f" * 64
    monkeypatch.setitem(artifacts.RETIRED_DATA_SNAPSHOTS, snapshot, REASON)
    for over in ({}, {"corpus_rev": "rebuilt"}, {"n_sequences": 9, "total_tokens": 90}):
        fields: dict[str, object] = {
            "split": "train", "data_snapshot_hash": snapshot, "tokenizer_hash": "b" * 64,
            "remap_hash": "c" * 64, "vocab_size": 128, "n_sequences": 4, "total_tokens": 40,
            "max_seq_len": 16, "buckets": (16,), **over,
        }
        header = ShardHeader(**fields)  # type: ignore[arg-type]
        with pytest.raises(ShardContractViolation, match="retired: contaminated"):
            ShardHeader.from_json(header.to_json())
