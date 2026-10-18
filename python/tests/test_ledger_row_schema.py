"""W2-1: the ledger row learns ``val_without_train`` and ``train_manifest_sha256``, and the
Python writer's ``started`` row is what the Rust mirror reads (``crates/qd-train/tests/
ledger_schema.rs``).

Both new fields are optional, omitted from the line when ``None`` and read back as ``None`` when
absent, so every row written before them parses as it did and every chain still verifies. The
committed ``ledger/`` files are the evidence for that, read here and never written.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.ledger import (
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    _population_paired_margin,
)
from qd_train.tristate import NotRun, Ran

REPO = Path(__file__).resolve().parents[2]
FAMILY = "synth.general/access_route"
SHA = hashlib.sha256(b"a train manifest file").hexdigest()


def _row(row_id: str = "r1", **extra: object) -> LedgerRow:
    env = Environment(
        torch="2.12.1", transformers_sha="abc", device="mps", host="test",
        fla_present=NotRun(reason="no CUDA"), causal_conv1d_present=NotRun(reason="no CUDA"),
    )
    return LedgerRow(
        row_id=row_id, written_at="2026-10-18T00:00:00+00:00", prev_row_hash=None,
        protocol=Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                          backbone_commit="b" * 40, recipe_hash="r" * 64, seed=0),
        run_kind="ft", status="completed", quick=True, quick_reason="test",
        code_commit="c" * 40, env=env, metrics={}, noul_rate=NotRun(reason="x"),
        controls={}, gates={}, wall_clock_s=1.0, cost_usd=0.0,
        wall_clock_source="recorder", **extra,  # type: ignore[arg-type]
    )


# -- val_without_train --------------------------------------------------------


def test_val_without_train_round_trips_through_the_line() -> None:
    block = {"count": 1, "families": {FAMILY: 3}}
    raw = json.loads(json.dumps(_row(val_without_train=block).to_json()))
    assert raw["val_without_train"] == block
    back = LedgerRow.from_json(raw)
    assert back.val_without_train == block
    assert back.families_without_train() == {FAMILY}


def test_a_row_recording_nothing_writes_neither_key_and_reads_back_none() -> None:
    raw = _row().to_json()
    assert "val_without_train" not in raw and "train_manifest_sha256" not in raw
    back = LedgerRow.from_json(raw)
    assert back.val_without_train is None and back.train_manifest_sha256 is None
    # null is the same claim as absent
    back = LedgerRow.from_json({**raw, "val_without_train": None, "train_manifest_sha256": None})
    assert back.val_without_train is None and back.train_manifest_sha256 is None


def test_an_empty_block_is_recorded_not_absent() -> None:
    row = _row(val_without_train={"count": 0, "families": {}})
    assert row.to_json()["val_without_train"] == {"count": 0, "families": {}}
    assert row.families_without_train() == frozenset()


def test_a_malformed_block_is_refused() -> None:
    with pytest.raises(ValueError, match="not a mapping"):
        _row(val_without_train=["a"])
    with pytest.raises(ValueError, match="families"):
        _row(val_without_train={"count": 1, "families": [FAMILY]})
    with pytest.raises(ValueError, match="keys must be strings"):
        _row(val_without_train={1: 2})


def _unit_state(*, recorded: bool) -> list[LedgerRow]:
    return [_row("ctl", **({"val_without_train": {"count": 1, "families": {FAMILY: 3}}}
                           if recorded else {}))]


def test_the_population_reading_names_val_without_train_as_the_cause() -> None:
    unit = _unit_state(recorded=True)
    absent = _population_paired_margin(unit, (FAMILY,))
    assert isinstance(absent, NotRun)
    assert "val rows and no train rows" in absent.reason and "ctl" in absent.reason
    unit[0].metrics["paired_margin_vs_linear.choice." + FAMILY] = NotRun(reason="no train rows")
    did_not_run = _population_paired_margin(unit, (FAMILY,))
    assert isinstance(did_not_run, NotRun) and "val rows and no train rows" in did_not_run.reason


def test_a_row_not_recording_the_block_gets_the_old_reading() -> None:
    reading = _population_paired_margin(_unit_state(recorded=False), (FAMILY,))
    assert isinstance(reading, NotRun)
    assert "cause:" not in reading.reason and "is absent from rows ctl" in reading.reason


def test_a_family_that_ran_is_not_blamed() -> None:
    unit = _unit_state(recorded=True)
    unit[0].metrics["paired_margin_vs_linear.choice." + FAMILY] = Ran(
        passed=True, value=0.3, n=10, n_total=10)
    assert isinstance(_population_paired_margin(unit, (FAMILY,)), Ran)


# -- train_manifest_sha256 ----------------------------------------------------


def test_train_manifest_sha256_round_trips_and_is_not_in_the_protocol() -> None:
    row = _row(train_manifest_sha256=SHA)
    raw = json.loads(json.dumps(row.to_json()))
    assert raw["train_manifest_sha256"] == SHA
    assert "train_manifest_sha256" not in raw["protocol"]
    assert LedgerRow.from_json(raw).train_manifest_sha256 == SHA
    assert row.protocol_hash == _row().protocol_hash


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "g" * 64, SHA + "0", 5])
def test_a_train_manifest_sha256_that_is_not_lowercase_hex_is_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="train_manifest_sha256"):
        _row(train_manifest_sha256=bad)


# -- the Rust oracle's fixture is a ledger Python reads ------------------------

FIXTURE = REPO / "crates" / "qd-train" / "tests" / "fixtures" / "ledger-schema"


def test_the_rust_oracle_fixture_is_what_python_reads_from_it() -> None:
    led = Ledger(FIXTURE / "ledger.jsonl")
    led.verify_chain()
    want = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    assert [[r.row_id, r.status] for r in led.rows()] == want["rows"]
    assert [[r.row_id, r.status] for r in led.rows(include_started=True)] == (
        want["rows_include_started"])
    assert [hashlib.sha256(b).hexdigest() for b in led.raw_lines()] == want["raw_line_sha256"]
    killed = [r for r in led.rows() if r.status == "killed"]
    assert len(killed) == 1, "the reconciled killed row"
    both, neither = led.rows()[2], led.rows()[3]
    assert both.val_without_train == want["val_without_train"]
    assert both.train_manifest_sha256 == want["train_manifest_sha256"]
    assert neither.val_without_train is None and neither.train_manifest_sha256 is None


# -- every committed row still parses, and every chain still verifies -----------


def test_every_committed_ledger_parses_unchanged_and_its_chain_verifies() -> None:
    files = sorted((REPO / "ledger").glob("*.jsonl"))
    assert files, "no committed ledgers found"
    lines = 0
    for path in files:
        led = Ledger(path)
        led.verify_chain()
        for line in led.raw_lines():
            row = LedgerRow.from_json(json.loads(line))
            assert row.val_without_train is None and row.train_manifest_sha256 is None, path
            out = row.to_json()
            assert "val_without_train" not in out and "train_manifest_sha256" not in out
            lines += 1
    assert lines > 100
