"""Adversarial (ft-57fc7484 L5, cc60581): Rust refuses a `lora`-labelled ft row that lacks the seven
LORA_ROW_METRICS and a train_manifest_sha256 (crates/qd-train/src/ledger.rs check_lora_label).
The Python side must not accept what Rust refuses: the ledger is one file, and Python's
Ledger.append / verify_chain / rows() are the verifier."""
from __future__ import annotations

from pathlib import Path

from qd_train.ledger import Ledger, LedgerRow, NotRun
from test_ledger import _env, _protocol


def _bare_lora_row() -> LedgerRow:
    return LedgerRow(
        row_id="adv-lora-bare", written_at="now", prev_row_hash=None, protocol=_protocol(1),
        run_kind="ft", status="completed", quick=True, quick_reason="adversarial",
        code_commit="c", env=_env(), metrics={}, noul_rate=NotRun(reason="n/a"),
        controls={}, gates={}, wall_clock_s=1.0, cost_usd=0.0,
        recipe={"optimizer_recipe": "lora"},
    )


def test_python_append_refuses_a_lora_row_with_no_lora_metrics(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    try:
        led.append(_bare_lora_row())
    except (ValueError, RuntimeError):
        return
    raise AssertionError(
        "Python appended a recipe.optimizer_recipe=='lora' ft row with no metrics and no "
        "train_manifest_sha256; Rust's append_with refuses exactly this row"
    )


def test_python_verifier_rejects_a_lora_row_rust_would_refuse(tmp_path: Path):
    """Even if a writer got such a row in, verify_chain/rows() should not call it clean."""
    led = Ledger(tmp_path / "runs.jsonl")
    row = _bare_lora_row()
    # Bypass append's own checks (if any) by writing the line the way Rust's append_line would.
    import json

    led.path.write_bytes(
        json.dumps(row.to_json(), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    try:
        led.verify_chain()
    except Exception:
        return
    raise AssertionError("verify_chain accepted a lora row with no metrics")
