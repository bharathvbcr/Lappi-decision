"""A task-holdout family with no rows is named in the export and not scanned.

``qd-prep containment`` refuses a scan whose target set is empty. The LoRA-2B v6 rebuild's scan
(W2-d, 2026-10-18) died on ``heldout-family:code.language_id`` (zero rows in this corpus; the v5
attestation lists it with 0 rows). ``drop_empty_targets`` leaves an empty set out of the scans,
and ``export_block`` names it, so the attestation never shows it as scanned. ``scan_specs`` is
unchanged (its request bytes are pinned by ``test_benchmark_targets_v6``).

Fails on the pre-fix code: ``drop_empty_targets`` and ``export_block(empty_sets=)`` do not exist,
and ``main`` handed the empty family's scan to the binary, which refused it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import containment_scan  # noqa: E402
from containment_scan import ScanSet, drop_empty_targets, export_block, scan_specs  # noqa: E402


def _set(name: str, n: int) -> ScanSet:
    rows = tuple((f"{name}-{i}", f"id{i}", "fam", "some text here") for i in range(n))
    return ScanSet(name=name, rows=rows, unrenderable={})


def test_empty_task_holdout_family_is_not_a_scan_target_and_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(containment_scan, "engine_sha256", lambda _p: "0" * 64)
    sets = [
        _set("train", 3), _set("val", 2), _set("heldout", 2),
        _set("heldout-family:code.language_id", 0), _set("heldout-family:qa.answerability", 2),
    ]
    full = scan_specs(sets)
    assert "heldout-family:code.language_id" in {s.target for s in full}
    scans, empty = drop_empty_targets(sets, full)
    assert empty == ["heldout-family:code.language_id"]
    assert "heldout-family:code.language_id" not in {s.target for s in scans}
    assert "heldout-family:qa.answerability" in {s.target for s in scans}
    assert len(scans) < len(full)
    block = export_block(sets, engine=REPO / "Cargo.toml", strip={}, empty_sets=empty)
    assert block["empty_sets_not_scanned"] == ["heldout-family:code.language_id"]
    assert "empty_sets_not_scanned" not in export_block(
        sets, engine=REPO / "Cargo.toml", strip={}
    )
