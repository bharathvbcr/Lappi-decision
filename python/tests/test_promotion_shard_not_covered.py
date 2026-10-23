"""A row trained on a shard set whose verdict carried ``not_covered`` does not promote.

Fable, 2026-10-23 (ft-2eb31e7d9863d70a25807b3425fa6057): a v2 shard set was written under a
fingerprint rule that did not hash ``manifest.py`` and ``val_requests.py``, so its
``shard_code_current`` pass says nothing about them. Promotable runs on it are allowed only after
a one-time rebuild under rule 3, compared byte for byte against the pinned set, has an AUDIT
record. Until then the set is report-only, enforced here and not by convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_promotion_respec import _family, _protocol, _record

from qd_train import ledger as ledger_mod
from qd_train.ledger import (
    SHARD_NOT_COVERED_METRIC,
    VAL_SHARD_NOT_COVERED_METRIC,
    Ledger,
    PromotionVerdict,
)
from qd_train.tristate import NotRun, Ran

SET_HASH = "94bef49aa2c13ed675d64d1d93b75ce201c2b47932596908e927c7172e01a0ac"
OTHER_HASH = "58bd753fc93391f191f9f7ff2bf8a5ea3d1e990f83c33752c7399a951a62435b"
V2_NOT_COVERED = Ran(passed=True, value="manifest.py,val_requests.py",
                     detail="not_covered: [manifest.py, val_requests.py]")
V3_COVERED = Ran(passed=True, value="", detail="not_covered: []")


def _led(
    tmp_path: Path, state: Ran | NotRun | None, shard_hash: str = SET_HASH,
    val_state: Ran | NotRun | None = None,
) -> Ledger:
    """Eval rows naming a train set and a val set, each with its own coverage verdict."""
    metrics: dict[str, Ran | NotRun] = {}
    if state is not None:
        metrics[SHARD_NOT_COVERED_METRIC] = state
    if val_state is not None:
        metrics[VAL_SHARD_NOT_COVERED_METRIC] = val_state
    return _family(tmp_path, pooled=Ran(passed=True, value=1.0, n=300, n_total=300),
                   train_recipe={"shard_hash": shard_hash, "val_shard_hash": OTHER_HASH},
                   train_metrics=metrics)


def _verdict(led: Ledger, tmp_path: Path, identity: Path | None) -> PromotionVerdict:
    return led.promotion_verdict(
        _protocol(0).hash_without_seed(), decisions_path=_record(tmp_path),
        byte_identity_path=identity,
    )


def _identity(tmp_path: Path, *records: dict[str, object]) -> Path:
    path = tmp_path / "byte_identity.json"
    path.write_text(json.dumps({"records": list(records)}), encoding="utf-8")
    return path


def _blocked_for_coverage(verdict: PromotionVerdict) -> bool:
    return not verdict.promoted and any("did not cover" in r for r in verdict.reasons)


def test_a_set_with_no_not_covered_metric_is_unchanged(tmp_path: Path) -> None:
    assert _verdict(_led(tmp_path, None), tmp_path, tmp_path / "absent.json").promoted


def test_a_current_rule_set_names_nothing_uncovered_and_promotes(tmp_path: Path) -> None:
    assert _verdict(_led(tmp_path, V3_COVERED), tmp_path, tmp_path / "absent.json").promoted


def test_a_row_on_a_not_covered_set_is_refused_without_a_byte_identity_record(
    tmp_path: Path,
) -> None:
    verdict = _verdict(_led(tmp_path, V2_NOT_COVERED), tmp_path, tmp_path / "absent.json")
    assert _blocked_for_coverage(verdict)
    assert any("manifest.py,val_requests.py" in r and "report-only" in r for r in verdict.reasons)


def test_the_default_record_path_does_not_exempt_the_pinned_sets_today(tmp_path: Path) -> None:
    """No byte-identity record is committed for 94bef49a / 58bd753f until the rebuild has run."""
    verdict = _verdict(_led(tmp_path, V2_NOT_COVERED), tmp_path, None)
    assert _blocked_for_coverage(verdict)
    assert ledger_mod.BYTE_IDENTITY_RECORD_PATH.name == "byte_identity.json"


def test_an_identical_record_for_the_set_lifts_the_refusal(tmp_path: Path) -> None:
    record = _identity(tmp_path, {"shard_hash": SET_HASH, "identical": True,
                                  "audit": "AUDIT/x6-fingerprint-list-2026-10-23/byte_identity.md"})
    assert _verdict(_led(tmp_path, V2_NOT_COVERED), tmp_path, record).promoted


@pytest.mark.parametrize("record", [
    {"shard_hash": SET_HASH, "identical": False, "audit": "AUDIT/x.md"},
    {"shard_hash": SET_HASH, "identical": "true", "audit": "AUDIT/x.md"},
    {"shard_hash": SET_HASH, "identical": True},
    {"shard_hash": SET_HASH, "identical": True, "audit": "  "},
    {"shard_hash": OTHER_HASH, "identical": True, "audit": "AUDIT/x.md"},
])
def test_a_record_that_is_not_identical_not_cited_or_for_another_set_does_not_lift_it(
    tmp_path: Path, record: dict[str, object]
) -> None:
    verdict = _verdict(_led(tmp_path, V2_NOT_COVERED), tmp_path, _identity(tmp_path, record))
    assert _blocked_for_coverage(verdict)


def test_an_unreadable_record_refuses_and_says_so(tmp_path: Path) -> None:
    bad = tmp_path / "byte_identity.json"
    bad.write_text("{not json", encoding="utf-8")
    verdict = _verdict(_led(tmp_path, V2_NOT_COVERED), tmp_path, bad)
    assert _blocked_for_coverage(verdict)
    assert any("could not be read" in r for r in verdict.reasons)


def test_only_the_val_set_being_v2_refuses_and_names_the_val_split(tmp_path: Path) -> None:
    """Promotion is decided on val metrics: a covered train set does not excuse an uncovered
    val set."""
    led = _led(tmp_path, V3_COVERED, val_state=V2_NOT_COVERED)
    verdict = _verdict(led, tmp_path, tmp_path / "absent.json")
    assert _blocked_for_coverage(verdict)
    assert any("val shard set" in r and "val_shard_hash" in r for r in verdict.reasons)
    assert not any("train shard set" in r for r in verdict.reasons)


def test_the_record_must_be_for_that_split_s_set_not_the_other_one(tmp_path: Path) -> None:
    """A record for the TRAIN hash does not exempt an uncovered VAL set, and vice versa."""
    led = _led(tmp_path, V2_NOT_COVERED, val_state=V2_NOT_COVERED)
    train_only = _identity(tmp_path, {"shard_hash": SET_HASH, "identical": True, "audit": "a.md"})
    blocked = _verdict(led, tmp_path, train_only)
    assert _blocked_for_coverage(blocked)
    assert any("val shard set" in r for r in blocked.reasons)
    assert not any("train shard set" in r for r in blocked.reasons)
    both = _identity(
        tmp_path,
        {"shard_hash": SET_HASH, "identical": True, "audit": "a.md"},
        {"shard_hash": OTHER_HASH, "identical": True, "audit": "b.md"},
    )
    assert _verdict(led, tmp_path, both).promoted


def test_a_covered_val_set_with_an_uncovered_train_set_still_refuses(tmp_path: Path) -> None:
    led = _led(tmp_path, V2_NOT_COVERED, val_state=V3_COVERED)
    verdict = _verdict(led, tmp_path, tmp_path / "absent.json")
    assert _blocked_for_coverage(verdict)
    assert any("train shard set" in r for r in verdict.reasons)


def test_a_coverage_verdict_that_did_not_run_refuses(tmp_path: Path) -> None:
    verdict = _verdict(
        _led(tmp_path, NotRun(reason="no code_fingerprint")), tmp_path, tmp_path / "absent.json"
    )
    assert _blocked_for_coverage(verdict)
