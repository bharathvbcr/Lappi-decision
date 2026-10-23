"""Python's half of the span-refit span-null check's cross-language parity.

The same row bytes -- ``crates/qd-train/tests/fixtures/span-refit-gate-cases.json``, written by
``tools/gen_span_refit_gate_cases.py`` -- go to Rust's ``check_row_line``
(``crates/qd-train/tests/span_refit_gate_parity.rs``, which asserts each case's ``expect``) and,
here, through both of this ledger's doors: ``verify_chain`` over a file holding the bytes, and
``Ledger.append`` of the row those bytes parse to. Each must give the case's ``expect``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import gen_span_refit_gate_cases as gen  # noqa: E402

from qd_train.ledger import Ledger, LedgerRow  # noqa: E402

CASES = json.loads(gen.FIXTURE.read_text(encoding="utf-8"))
REFUSAL = r"span-null scorer|pointing-row count"


def test_the_fixture_is_the_generators_output() -> None:
    assert gen.FIXTURE.read_text(encoding="utf-8") == gen.render()


def test_the_cases_cover_what_the_lead_named() -> None:
    names = {c["name"] for c in CASES}
    assert {
        "new_sha_valid", "new_sha_missing_the_metrics", "new_sha_counts_disagree",
        "arm_a_after_validity_gate_failed", "old_sha_without_metrics",
    } <= names


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_python_gives_the_row_bytes_the_verdict_rust_does(case: dict[str, str], tmp_path: Path):
    line = case["line"].encode("utf-8")
    accept = case["expect"] == "accept"

    verified = Ledger(tmp_path / "verify.jsonl")
    verified.path.write_bytes(line + b"\n")
    appended = Ledger(tmp_path / "append.jsonl")
    if accept:
        verified.verify_chain()
        assert len(verified.rows()) == 1
        appended.append(LedgerRow.from_json(json.loads(line)))
        appended.verify_chain()
    else:
        with pytest.raises(ValueError, match=REFUSAL):
            verified.verify_chain()
        row = _structural(json.loads(line))
        with pytest.raises(ValueError, match=REFUSAL):
            appended.append(row)
        assert appended.raw_lines() == []


def _structural(raw: dict) -> LedgerRow:
    """The row ``raw`` describes without the span-null check, as a writer holds it before
    ``Ledger.append`` (``from_json`` runs the check: it is the read door)."""
    original = LedgerRow.check_span_refit_span_null
    try:
        LedgerRow.check_span_refit_span_null = lambda self: None  # type: ignore[method-assign]
        return LedgerRow.from_json(raw)
    finally:
        LedgerRow.check_span_refit_span_null = original  # type: ignore[method-assign]
