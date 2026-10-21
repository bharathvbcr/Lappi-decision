"""Python's half of the `lora` label gate's cross-language parity (ft-57fc7484 L5).

The same row bytes -- ``crates/qd-train/tests/fixtures/lora-gate-cases.json``, written by
``tools/gen_lora_gate_cases.py`` -- go to Rust's ``check_row_line``
(``crates/qd-train/tests/lora_gate_parity.rs``, which asserts each case's ``expect``) and, here,
through both of this ledger's doors: ``verify_chain`` over a file holding the bytes, and
``Ledger.append`` of the row those bytes parse to. Each must give the case's ``expect``. The
Rust-writes, Python-verifies direction is ``pythons_verifier_accepts_a_lora_row`` in
``crates/qd-train/tests/lora_row.rs`` (``--ignored``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import gen_lora_gate_cases as gen  # noqa: E402

from qd_train.ledger import Ledger, LedgerRow  # noqa: E402

CASES = json.loads(gen.FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_is_the_generators_output() -> None:
    assert gen.FIXTURE.read_text(encoding="utf-8") == gen.render()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_python_gives_the_row_bytes_the_verdict_rust_does(case: dict[str, str], tmp_path: Path):
    line = case["line"].encode("utf-8")
    accept = case["expect"] == "accept"

    verified = Ledger(tmp_path / "verify.jsonl")
    verified.path.write_bytes(line + b"\n")
    if accept:
        verified.verify_chain()
        assert len(verified.rows()) == 1
    else:
        with pytest.raises(ValueError, match=r"optimizer_recipe|lora"):
            verified.verify_chain()

    appended = Ledger(tmp_path / "append.jsonl")
    if accept:
        row = LedgerRow.from_json(json.loads(line))
        appended.append(row)
        appended.verify_chain()
    else:
        # The row parses structurally; the append door is what refuses it.
        raw = json.loads(line)
        row = _structural(raw)
        with pytest.raises(ValueError, match=r"optimizer_recipe|lora"):
            appended.append(row)
        assert appended.raw_lines() == []


def _structural(raw: dict) -> LedgerRow:
    """The row ``raw`` describes without its label gate, as a writer would hold it before
    ``Ledger.append``. ``from_json`` runs the gate (it is the read door), so build the object the
    way ``from_json`` does and stop before the check."""
    original = LedgerRow.check_optimizer_label
    try:
        LedgerRow.check_optimizer_label = lambda self: None  # type: ignore[method-assign]
        return LedgerRow.from_json(raw)
    finally:
        LedgerRow.check_optimizer_label = original  # type: ignore[method-assign]
