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
    VAL_SHARD_HASH_METRIC,
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
    """A record file beside the evidence files it cites (each cited file is created: a citation
    must name an existing file, see ``test_a_citation_naming_no_file_does_not_lift``)."""
    for rec in records:
        audit = rec.get("audit")
        if not (isinstance(audit, str) and audit.strip()):
            continue
        cited = Path(audit)
        # Only a plain relative path is ever written: never an absolute one or one that climbs
        # out of tmp_path (those are the refused citations, and must not touch the disk).
        if cited.is_absolute() or ".." in cited.parts or audit.startswith("AUDIT/missing"):
            continue
        (tmp_path / cited).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / cited).write_text("evidence", encoding="utf-8")
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


@pytest.mark.parametrize("audit", ["AUDIT/missing/evidence.md", "/etc/hosts", "../escape.md"])
def test_a_citation_naming_no_file_in_the_repo_does_not_lift(tmp_path: Path, audit: str) -> None:
    (tmp_path.parent / "escape.md").write_text("outside", encoding="utf-8")
    record = _identity(tmp_path, {"shard_hash": SET_HASH, "identical": True, "audit": audit})
    assert _blocked_for_coverage(_verdict(_led(tmp_path, V2_NOT_COVERED), tmp_path, record))


def test_conflicting_records_for_one_set_lift_nothing_whichever_came_last(
    tmp_path: Path,
) -> None:
    for n, order in enumerate(((True, False), (False, True))):
        work = tmp_path / f"order{n}"
        work.mkdir()
        records = [
            {"shard_hash": SET_HASH, "identical": flag, "audit": f"{i}.md"}
            for i, flag in enumerate(order)
        ]
        verdict = _verdict(_led(work, V2_NOT_COVERED), work, _identity(work, *records))
        assert _blocked_for_coverage(verdict), order
        assert any("disagree" in r for r in verdict.reasons), order


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


def test_an_ft_row_naming_its_val_set_only_by_the_hash_metric_can_be_lifted(
    tmp_path: Path,
) -> None:
    """An ft row's recipe carries shard_hash only; open_val_reader records the val hash as a
    metric. A record for that hash lifts it; a metric that disagrees with the recipe names no
    set; a row naming no val hash at all cannot be lifted by any record."""
    metrics = {VAL_SHARD_NOT_COVERED_METRIC: V2_NOT_COVERED,
               VAL_SHARD_HASH_METRIC: Ran(passed=True, value=OTHER_HASH)}
    record = _identity(tmp_path, {"shard_hash": OTHER_HASH, "identical": True, "audit": "v.md"})

    def ft_led(work: Path, metrics: dict[str, Ran], recipe: dict[str, object]) -> Ledger:
        work.mkdir()
        return _family(work, pooled=Ran(passed=True, value=1.0, n=300, n_total=300),
                       train_recipe=recipe, train_metrics=metrics)

    ft = {"shard_hash": SET_HASH}
    assert _verdict(ft_led(tmp_path / "ok", metrics, ft), tmp_path, record).promoted
    clash = {**ft, "val_shard_hash": SET_HASH}
    assert _blocked_for_coverage(_verdict(ft_led(tmp_path / "clash", metrics, clash),
                                          tmp_path, record))
    nameless = {VAL_SHARD_NOT_COVERED_METRIC: V2_NOT_COVERED}
    assert _blocked_for_coverage(_verdict(ft_led(tmp_path / "none", nameless, ft),
                                          tmp_path, record))


def test_every_ft_row_writer_takes_the_verdict_from_one_function_that_fails_closed() -> None:
    """``shard_coverage_state`` is what real_ft_run, lora_reference_run and open_val_reader call:
    the reader's verdict, or a NotRun when it recorded none (never an absent metric)."""
    from qd_train.ledger import shard_coverage_state

    assert shard_coverage_state({"shard_code_not_covered": V2_NOT_COVERED}) == V2_NOT_COVERED
    assert isinstance(shard_coverage_state({}), NotRun)


@pytest.mark.parametrize("split", ["train", "val"])
def test_a_not_run_coverage_metric_is_not_covered_and_only_a_record_lifts_it(
    tmp_path: Path, split: str
) -> None:
    """A check that could not run never passes as clean: a NotRun coverage metric refuses
    exactly as a non-empty not_covered does, and is lifted only by a byte-identity record."""
    not_run = NotRun(reason="the shard reader recorded no shard_code_not_covered verdict")
    train_state, val_state = (not_run, None) if split == "train" else (None, not_run)
    refused = tmp_path / "refused"
    refused.mkdir()
    verdict = _verdict(_led(refused, train_state, val_state=val_state), refused,
                       tmp_path / "absent.json")
    assert _blocked_for_coverage(verdict)
    assert any(f"its {split} shard set" in r and "did not run" in r for r in verdict.reasons)

    set_hash = SET_HASH if split == "train" else OTHER_HASH
    lifted = tmp_path / "lifted"
    lifted.mkdir()
    record = _identity(lifted, {"shard_hash": set_hash, "identical": True, "audit": "e.md"})
    assert _verdict(_led(lifted, train_state, val_state=val_state), lifted, record).promoted


def test_a_row_with_the_metric_absent_is_historical_and_unmarked(tmp_path: Path) -> None:
    """Rows written before the metric existed carry none: unchanged, not refused."""
    assert _verdict(_led(tmp_path, None), tmp_path, tmp_path / "absent.json").promoted


def test_a_coverage_verdict_that_did_not_run_refuses(tmp_path: Path) -> None:
    verdict = _verdict(
        _led(tmp_path, NotRun(reason="no code_fingerprint")), tmp_path, tmp_path / "absent.json"
    )
    assert _blocked_for_coverage(verdict)
