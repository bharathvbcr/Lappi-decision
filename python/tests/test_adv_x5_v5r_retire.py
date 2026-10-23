"""Adversarial tests for lane X5 (v5r retirement). Tests marked BREAK fail on lane/x5-v5r-retire.

The others (HELD) pass and pin the attacks the guard survives. No v5r file is read: the retired
hash is simulated by retiring a synthetic header's own hash with monkeypatch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from qd_train import artifacts  # noqa: E402
from qd_train.artifacts import ShardContractViolation, ShardHeader  # noqa: E402

V5R = (
    "108082545a66ec3e0b008f8ad7a8425625cd9bc323630f8d0cd186f40e00cdf4",
    "a485e7f7056d22b9f7e6baf8f19ae64b313db9f3b77b504a094ff1caae3f5d0f",
)


def _header(**over: object) -> ShardHeader:
    kw: dict[str, object] = dict(
        split="train", data_snapshot_hash="a" * 64, tokenizer_hash="b" * 64,
        remap_hash="c" * 64, vocab_size=128, n_sequences=4, total_tokens=40, max_seq_len=16,
        buckets=(16,),
    )
    kw.update(over)
    return ShardHeader(**kw)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- HELD


def test_held_registry_keys_are_full_lowercase_64_hex() -> None:
    for key in artifacts.RETIRED_SHARD_SETS:
        assert re.fullmatch(r"[0-9a-f]{64}", key), key
    assert set(V5R) == set(artifacts.RETIRED_SHARD_SETS)


@pytest.mark.parametrize("mangle", [str.upper, lambda s: s + " ", lambda s: " " + s,
                                    lambda s: s[:12], lambda s: s[:-1], lambda s: ""])
def test_held_a_mangled_recorded_hash_is_refused_not_opened(
    monkeypatch: pytest.MonkeyPatch, mangle
) -> None:
    header = _header()
    monkeypatch.setitem(artifacts.RETIRED_SHARD_SETS, header.shard_hash(), "r")
    raw = header.to_json()
    raw["shard_hash"] = mangle(raw["shard_hash"])
    with pytest.raises(ShardContractViolation):  # mismatch or retired; never opens
        ShardHeader.from_json(raw)


def test_held_a_header_with_no_shard_hash_field_is_still_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = _header()
    monkeypatch.setitem(artifacts.RETIRED_SHARD_SETS, header.shard_hash(), "retired: x")
    raw = header.to_json()
    del raw["shard_hash"]  # Python skips the comparison; the retired check uses the computed hash
    with pytest.raises(ShardContractViolation, match="retired"):
        ShardHeader.from_json(raw)


def test_held_a_copied_header_with_other_unhashed_fields_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    header = _header()
    monkeypatch.setitem(artifacts.RETIRED_SHARD_SETS, header.shard_hash(), "retired: x")
    raw = header.to_json()
    raw["created_at"] = "2030-01-01T00:00:00+00:00"  # not covered by the hash
    copy = tmp_path / "copy" / "header.json"
    copy.parent.mkdir()
    copy.write_text(json.dumps(raw))
    with pytest.raises(ShardContractViolation, match="retired"):
        ShardHeader.from_json(json.loads(copy.read_text()))


def test_held_the_guard_runs_before_assert_shard_trainable_in_the_reader() -> None:
    text = (REPO / "python" / "qd_train" / "shards.py").read_text()
    door = text.index("ShardHeader.from_json(raw)")
    check = text.index("assert_shard_trainable(\n            self.header", door)
    assert door < check  # allow_stale_code lives in assert_shard_trainable, after the door


# ----------------------------------------------------------------------------- BREAKS


def test_break_rust_header_reader_has_no_retired_set_check() -> None:
    """crates/qd-train/src/shards.rs opens train shards for the Rust trainers
    (qd-train, qd-train-metal) and carries no retired-set list. A v5r directory is trainable
    from Rust."""
    src = (REPO / "crates" / "qd-train" / "src" / "shards.rs").read_text()
    assert all(h in src for h in V5R), "Rust ShardHeader::from_json does not refuse the v5r pair"


def test_break_a_rehashed_copy_of_the_retired_tokens_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard keys on the header hash, not on the data. A copy of the same tokens whose
    header differs in any covered field (here corpus_rev) and is re-hashed carries a new
    shard_hash and opens. (A deliberate forgery, so low severity; also what a rebuild under
    drifted qd_data code produces: a v5-rule rebuild gets a different code_fingerprint, hence
    a different shard_hash, hence is not retired.)"""
    retired = _header()
    monkeypatch.setitem(artifacts.RETIRED_SHARD_SETS, retired.shard_hash(), "retired: x")
    sibling = _header(corpus_rev="rebuilt")
    assert sibling.shard_hash() != retired.shard_hash()
    with pytest.raises(ShardContractViolation):
        ShardHeader.from_json(sibling.to_json())


def test_break_the_p1_queue_still_builds_the_v5_rule_data() -> None:
    """campaign/next-train-first-box-queue-2026-10-08.json is untouched: once its placeholders
    are filled it launches `real_ft_run.py --rev 8e6a009...` (the v5 rule) and nothing in the
    file or the driver marks it held. The retirement guard cannot catch the rebuilt set (see
    the test above), so the box would train on a re-contaminated set."""
    queue = json.loads(
        (REPO / "campaign" / "next-train-first-box-queue-2026-10-08.json").read_text()
    )
    builds_v5 = any(
        "8e6a00952a09e31a09cc3f5270229a935b7235cc" in u["argv"]
        for ph in queue["phases"]
        for u in ph["units"]
    )
    held = any(k in queue for k in ("hold", "held", "withdrawn", "retired", "status"))
    assert not builds_v5 or held, "queue builds v5-rule data and carries no hold marker"


def test_break_p1_data_fact_result_is_still_machine_readable_as_current() -> None:
    """The void is a free-text sibling key. `result` (211 of 216, floor_108_met: true) and
    `shard_set` (the v5r hash) stay in place, untouched, and a reader that takes
    data_fact.result gets the v5r numbers (test_read_p1_parity does this on the DRAFT)."""
    reg = json.loads((REPO / "campaign" / "next-train-first-box-preregistered.json").read_text())
    fact = reg["P1"]["data_fact"]
    assert "void" in fact
    assert "result" not in fact or fact.get("void_structured") is True, (
        "data_fact.result still carries the v5r numbers with floor_108_met true"
    )


def test_break_span_refit_train_extraction_is_not_refused_on_amendment_3() -> None:
    """Amendment 3 says 'No run is registered until the pending items are fixed', and the
    scorer is the fail-closed gate. But the train-split selection (the fit set) has no pin
    check and runs on the v6 train set today."""
    pytest.importorskip("torch")
    import span_refit_extract as X

    prereg = json.loads((REPO / "campaign" / "span-refit-mac-preregistered.json").read_text())
    n = 30000
    seqs = [(f"src:qa.answer_span:r{i}" if i % 2 else f"src:code.defect_class:r{i}", "span")
            for i in range(n)]
    reader = SimpleNamespace(
        _slot_kind=__import__("numpy").full(n, X.SLOT_SPAN),
        sequence_index=SimpleNamespace(sequences=seqs),
        lengths=lambda: [100] * n,
    )
    with pytest.raises(SystemExit):
        X.select(reader, prereg, "train")
