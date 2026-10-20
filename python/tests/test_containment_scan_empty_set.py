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
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

import containment_scan  # noqa: E402
from containment_scan import ScanSet, drop_empty_targets, export_block, scan_specs  # noqa: E402


def _set(name: str, n: int) -> ScanSet:
    rows = tuple((f"{name}-{i}", f"id{i}", "fam", "some text here") for i in range(n))
    return ScanSet(name=name, rows=rows, unrenderable={})


def test_the_scan_names_its_corpus_as_the_build_does() -> None:
    """The scan wrote only ``benchmark_eval_splits_are_targets`` beside the v5 name, while the
    v6 build also names ``dedupe_keep_rule`` and ``lsh_min_agreement_permille`` (every
    ``V6_CORPUS_KEYS`` fingerprint), so the build refused the scan's list as 'another corpus'
    (W2-d, chain9). Both name the corpus through ``name_corpus`` now."""
    from real_tokenizer_pipeline import data_rules, name_corpus

    from qd_data.config import DataConfig

    identity = {"rev": "a" * 40, "max_pairs": 400, "repo_history": False}
    for v6 in (True, False):
        config = data_rules(DataConfig(), v6=v6)
        got = containment_scan.scan_corpus_name(identity, config)
        assert got == name_corpus(identity, config=config)
    v6_name = containment_scan.scan_corpus_name(identity, data_rules(DataConfig(), v6=True))
    assert {"dedupe_keep_rule", "lsh_min_agreement_permille"} <= set(v6_name)


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
