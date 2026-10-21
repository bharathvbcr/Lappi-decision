"""The `lora` label gate (ft-57fc7484 L5), Python side: ``LedgerRow`` refuses exactly what Rust's
``check_optimizer_label`` (crates/qd-train/src/ledger.rs) refuses, for every writer and reader of
a row (``Ledger.append``, ``rows``, ``verify_chain`` all build a ``LedgerRow``).

The Rust-to-Python direction is the ``pythons_verifier_accepts_a_lora_row`` test in
crates/qd-train/tests/lora_row.rs (``--ignored``): Rust writes a lora row, this ledger verifies it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qd_train.ledger import (
    OPTIMIZER_RECIPE_LABELS,
    Ledger,
    LedgerRow,
)
from qd_train.tristate import NotRun, Ran
from test_ledger import _env, _protocol

REPO = Path(__file__).resolve().parents[2]
HEX = "ab" * 32
RUST_METRICS = {
    "train.span_head_init_digest": Ran(passed=True, value=HEX),
    "lora.d17_init_digest": Ran(passed=True, value="cd" * 32),
    "lora.a_init_digest": Ran(passed=True, value="ef" * 32),
    "lora.base_safetensors_sha256": Ran(passed=True, value="01" * 32),
    "lora.dtype_map_sha256": Ran(passed=True, value="23" * 32),
    "corpus.plan_seed": Ran(passed=True, value=0),
    "lora.param_groups": Ran(passed=True, value="lora x2 at lr_scale 1, d17 x1 at lr_scale 0.1"),
}
DRIVER_METRICS = {
    "init.span_head_digest": Ran(passed=True, value=HEX),
    "init.d17_digest": Ran(passed=True, value="cd" * 32),
    "init.lora_a_digest": Ran(passed=True, value="ef" * 32),
    "init.base_safetensors_sha256": Ran(passed=True, value="01" * 32),
    "init.dtype_map_sha256": Ran(passed=True, value="23" * 32),
    "corpus.plan_seed": Ran(passed=True, value="0"),
}


def _row(label: str = "lora", *, metrics=None, status: str = "completed",
         manifest: str | None = "c" * 64, row_id: str = "lora-gate") -> LedgerRow:
    row = LedgerRow(
        row_id=row_id, written_at="now", prev_row_hash=None, protocol=_protocol(1),
        run_kind="ft", status=status, quick=True, quick_reason="gate test",
        code_commit="c", env=_env(), metrics=dict(metrics or {}), noul_rate=NotRun(reason="n/a"),
        controls={}, gates={}, wall_clock_s=1.0, cost_usd=0.0,
        recipe={"optimizer_recipe": label}, train_manifest_sha256=manifest,
    )
    row.check_optimizer_label()  # the gate Ledger.append and from_json run
    return row


def test_the_label_set_is_the_one_rust_and_real_ft_run_use() -> None:
    rust = (REPO / "crates/qd-train/src/ledger.rs").read_text()
    body = re.search(r"OPTIMIZER_RECIPE_LABELS: \[&str; \d+\] = \[([^\]]*)\]", rust)
    assert body is not None
    assert set(re.findall(r'"([^"]+)"', body.group(1))) == set(OPTIMIZER_RECIPE_LABELS)
    ft = (REPO / "tools/real_ft_run.py").read_text()
    block = re.search(r"OPTIMIZER_RECIPES: Final\[dict\[str, str\]\] = \{(.*?)\n\}", ft, re.S)
    assert block is not None
    assert set(re.findall(r'^    "([a-z0-9]+)":', block.group(1), re.M)) == set(
        OPTIMIZER_RECIPE_LABELS
    )


def test_a_complete_rust_spelled_row_and_a_complete_driver_spelled_row_are_accepted(
    tmp_path: Path,
) -> None:
    led = Ledger(tmp_path / "runs.jsonl")
    led.append(_row(metrics=RUST_METRICS, row_id="rust"))
    led.append(_row(metrics=DRIVER_METRICS, manifest=None, row_id="driver"))
    led.verify_chain()
    assert [r.row_id for r in led.rows()] == ["rust", "driver"]


@pytest.mark.parametrize("label", ["LORA", "Lora", " lora", "lora ", "lora\n", "lоra", "adamw", ""])
def test_a_label_outside_the_closed_set_is_refused(label: str) -> None:
    with pytest.raises(ValueError, match="optimizer_recipe"):
        _row(label, metrics=RUST_METRICS)


@pytest.mark.parametrize("label", sorted(OPTIMIZER_RECIPE_LABELS - {"lora"}))
def test_every_other_known_label_needs_no_evidence(label: str) -> None:
    _row(label)


def _bad_values() -> list[tuple[str, object]]:
    bad_hex = [
        Ran(passed=True, value=""), Ran(passed=True, value="ab"),
        Ran(passed=True, value="A" * 64), Ran(passed=True, value="z" * 64),
        Ran(passed=True, value=None), Ran(passed=True, value=7),
        Ran(passed=False, value=HEX), NotRun(reason="never measured"),
    ]
    cases: list[tuple[str, object]] = [
        (name, v) for name in list(RUST_METRICS)[:5] for v in bad_hex
    ]
    cases += [
        ("corpus.plan_seed", v) for v in (
            Ran(passed=True, value="0"), Ran(passed=True, value=-1), Ran(passed=True, value=1.5),
            Ran(passed=True, value=True), Ran(passed=True, value=None),
            Ran(passed=False, value=0), NotRun(reason="never measured"),
        )
    ]
    cases += [
        ("lora.param_groups", v) for v in (
            Ran(passed=True, value=""), Ran(passed=True, value="lora"),
            Ran(passed=True, value="lora x0 at lr_scale 1"),
            Ran(passed=True, value="lora x2 at lr_scale nan"),
            Ran(passed=True, value="lora x2 at lr_scale 1_0"),
            Ran(passed=True, value="lora x2 at lr_scale 1, "),
            Ran(passed=True, value=3), Ran(passed=False, value="lora x2 at lr_scale 1"),
            NotRun(reason="never measured"),
        )
    ]
    return cases


@pytest.mark.parametrize(("name", "value"), _bad_values())
def test_a_rust_spelled_row_with_a_bad_metric_value_is_refused(name: str, value: object) -> None:
    metrics = {**RUST_METRICS, name: value}
    with pytest.raises(ValueError, match=re.escape(name)):
        _row(metrics=metrics)


def test_missing_metrics_manifest_sha_and_mixed_spellings_are_refused() -> None:
    for name in RUST_METRICS:
        partial = {k: v for k, v in RUST_METRICS.items() if k != name}
        with pytest.raises(ValueError):
            _row(metrics=partial)
    with pytest.raises(ValueError, match="train_manifest_sha256"):
        _row(metrics=RUST_METRICS, manifest=None)
    with pytest.raises(ValueError):
        _row(metrics={**RUST_METRICS, "init.d17_digest": DRIVER_METRICS["init.d17_digest"]},
             manifest=None, row_id="mixed-missing")
    partial_driver = {k: v for k, v in DRIVER_METRICS.items() if k != "init.lora_a_digest"}
    with pytest.raises(ValueError):
        _row(metrics=partial_driver, manifest=None)
    with pytest.raises(ValueError):
        _row(metrics={})


def test_started_failed_and_killed_lora_rows_stay_writable_without_evidence() -> None:
    for status in ("started", "failed", "killed"):
        _row(status=status, manifest=None)


def test_verify_chain_and_rows_refuse_a_hand_written_bare_lora_line(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "runs.jsonl")
    ok = _row(metrics=RUST_METRICS)
    # Serialise the evidence-free row the way a raw append_line writer would.
    raw = ok.to_json()
    raw["row_id"] = "bare"
    raw["metrics"] = {}
    led.path.write_bytes(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(ValueError):
        led.verify_chain()
    with pytest.raises(ValueError):
        led.rows()
