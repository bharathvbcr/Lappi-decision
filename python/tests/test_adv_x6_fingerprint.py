"""Adversarial tests against lane X6 (fingerprint rule 3 and report-only enforcement).

Each ``test_BREAK_*`` is expected to FAIL against the X6 branch; each ``test_HELD_*`` is an
attack that X6 resists and is kept as a regression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_promotion_respec import _family, _protocol, _record

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import qd_data.fingerprint as fp  # noqa: E402
from qd_data.config import DataConfig  # noqa: E402
from qd_train import artifacts  # noqa: E402
from qd_train.artifacts import ShardContractViolation, ShardHeader  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    SHARD_NOT_COVERED_METRIC,
    VAL_SHARD_HASH_METRIC,
    VAL_SHARD_NOT_COVERED_METRIC,
)
from qd_train.tristate import Ran  # noqa: E402

TRAIN = "94bef49aa2c13ed675d64d1d93b75ce201c2b47932596908e927c7172e01a0ac"
VAL = "58bd753fc93391f191f9f7ff2bf8a5ea3d1e990f83c33752c7399a951a62435b"
NC = Ran(passed=True, value="manifest.py,val_requests.py", detail="")
COVERED = Ran(passed=True, value="", detail="")


def _header(**over: object) -> ShardHeader:
    fields: dict[str, object] = {
        "split": "train", "data_snapshot_hash": "a" * 64, "tokenizer_hash": "b" * 64,
        "remap_hash": "c" * 64, "vocab_size": 128, "n_sequences": 4, "total_tokens": 40,
        "max_seq_len": 16, "buckets": (16,),
    }
    fields.update(over)
    return ShardHeader(**fields)  # type: ignore[arg-type]


def _check(header: ShardHeader):
    return artifacts._shard_code_check(header, path=Path("h.json"))


# ---------------------------------------------------------------- target 1 / 2: rule branching


@pytest.mark.parametrize("version", [4, 0, -1, 99])
def test_HELD_unknown_numeric_version_is_refused(version: int) -> None:
    with pytest.raises(ShardContractViolation):
        _header(code_fingerprint=fp.shard_code_fingerprint(), code_fingerprint_version=version)


@pytest.mark.parametrize("bad", [4, 0, -1, "3", "2", 3.0, 2.0, True, None, [3]])
def test_HELD_from_json_refuses_a_mistyped_or_unknown_version(bad: object) -> None:
    raw = _header(
        code_fingerprint=fp.shard_code_fingerprint(), code_fingerprint_version=3
    ).to_json()
    raw["code_fingerprint_version"] = bad
    with pytest.raises(ShardContractViolation):
        ShardHeader.from_json(raw)


def test_HELD_a_v3_header_recording_the_16_module_list_is_not_verified() -> None:
    forged = _header(
        code_fingerprint=fp.shard_code_fingerprint(rule=2), code_fingerprint_version=3
    )
    state = _check(forged)
    assert isinstance(state, Ran) and not state.passed


@pytest.mark.parametrize("drop", sorted(fp.SHARD_CONTENT_MODULES_V2))
def test_HELD_a_v2_header_recording_fewer_than_16_modules_is_not_verified(drop: str) -> None:
    rec = dict(fp.shard_code_fingerprint(rule=2))
    del rec[drop]
    state = _check(_header(code_fingerprint=rec, code_fingerprint_version=2))
    assert isinstance(state, Ran) and not state.passed, drop


def test_HELD_a_v2_header_with_an_extra_module_is_not_verified() -> None:
    rec = dict(fp.shard_code_fingerprint(rule=2))
    rec["manifest.py"] = fp.shard_code_fingerprint(rule=3)["manifest.py"]
    state = _check(_header(code_fingerprint=rec, code_fingerprint_version=2))
    assert isinstance(state, Ran) and not state.passed


def _trainable(header: ShardHeader, tmp_path: Path) -> dict[str, object]:
    return artifacts.assert_shard_trainable(
        header, config=DataConfig(), path=tmp_path / "shards" / "header.json",
        repo_root=tmp_path, allow_stale_code=True,
    )


def test_BREAK_not_covered_understates_for_a_truncated_v2_header_read_with_allow_stale_code(
    tmp_path: Path,
) -> None:
    """A v2 header that recorded 15 modules (decisions.py dropped) fails shard_code_current,
    but allow_stale_code reads it anyway; its not_covered says only the two rule-3 additions,
    computed from the version alone. decisions.py is equally uncovered."""
    rec = dict(fp.shard_code_fingerprint(rule=2))
    del rec["decisions.py"]
    checks = _trainable(_header(code_fingerprint=rec, code_fingerprint_version=2), tmp_path)
    assert not checks["shard_code_current"].passed  # type: ignore[union-attr]
    nc = checks["shard_code_not_covered"]
    assert isinstance(nc, Ran)
    assert "decisions.py" in str(nc.value), f"not_covered says {nc.value!r}"


def test_BREAK_a_drifted_v3_header_read_with_allow_stale_code_reports_not_covered_clean(
    tmp_path: Path,
) -> None:
    """A v3 header whose modules have drifted (shard_code_current failed) is reported with
    shard_code_not_covered == "" : the same value a clean current-rule set carries, so a row
    written from it is unmarked and promotable."""
    rec = dict(fp.shard_code_fingerprint(rule=3))
    rec["render.py"] = "0" * 64
    checks = _trainable(_header(code_fingerprint=rec, code_fingerprint_version=3), tmp_path)
    assert not checks["shard_code_current"].passed  # type: ignore[union-attr]
    nc = checks["shard_code_not_covered"]
    assert not (isinstance(nc, Ran) and nc.value == ""), (
        "drifted v3 set reads as a clean 'nothing uncovered'"
    )


# ---------------------------------------------------------------- target 3: report-only enforcement


def _led(tmp_path: Path, metrics: dict[str, Ran], recipe: dict[str, object]):
    return _family(tmp_path, pooled=Ran(passed=True, value=1.0, n=300, n_total=300),
                   train_recipe=recipe, train_metrics=metrics)


def _identity(tmp_path: Path, *records: dict[str, object]) -> Path:
    path = tmp_path / "byte_identity.json"
    path.write_text(json.dumps({"records": list(records)}), encoding="utf-8")
    return path


def _verdict(led, tmp_path: Path, identity: Path):
    return led.promotion_verdict(
        _protocol(0).hash_without_seed(), decisions_path=_record(tmp_path),
        byte_identity_path=identity,
    )


def _blocked(v) -> bool:
    return not v.promoted and any("did not cover" in r for r in v.reasons)


def test_BREAK_a_later_identical_false_record_does_not_revoke_an_earlier_true_one(
    tmp_path: Path,
) -> None:
    led = _led(tmp_path, {SHARD_NOT_COVERED_METRIC: NC},
               {"shard_hash": TRAIN, "val_shard_hash": VAL})
    ident = _identity(
        tmp_path,
        {"shard_hash": TRAIN, "identical": True, "audit": "AUDIT/a.md"},
        {"shard_hash": TRAIN, "identical": False, "audit": "AUDIT/b.md"},
    )
    assert _blocked(_verdict(led, tmp_path, ident)), "conflicting records still lift the refusal"


def test_BREAK_an_audit_citation_naming_no_file_lifts_the_refusal(tmp_path: Path) -> None:
    led = _led(tmp_path, {SHARD_NOT_COVERED_METRIC: NC},
               {"shard_hash": TRAIN, "val_shard_hash": VAL})
    ident = _identity(
        tmp_path, {"shard_hash": TRAIN, "identical": True, "audit": "AUDIT/does/not/exist.md"}
    )
    assert _blocked(_verdict(led, tmp_path, ident)), "the cited evidence file does not exist"


def test_HELD_an_uppercase_hash_does_not_lift(tmp_path: Path) -> None:
    led = _led(tmp_path, {SHARD_NOT_COVERED_METRIC: NC},
               {"shard_hash": TRAIN, "val_shard_hash": VAL})
    ident = _identity(
        tmp_path, {"shard_hash": TRAIN.upper(), "identical": True, "audit": "AUDIT/a.md"}
    )
    assert _blocked(_verdict(led, tmp_path, ident))


def test_HELD_a_train_record_does_not_lift_val(tmp_path: Path) -> None:
    led = _led(tmp_path, {SHARD_NOT_COVERED_METRIC: NC, VAL_SHARD_NOT_COVERED_METRIC: NC},
               {"shard_hash": TRAIN, "val_shard_hash": VAL})
    ident = _identity(tmp_path, {"shard_hash": TRAIN, "identical": True, "audit": "a.md"})
    v = _verdict(led, tmp_path, ident)
    assert _blocked(v) and any("val shard set" in r for r in v.reasons)


def test_BREAK_a_val_record_cannot_lift_a_row_whose_recipe_names_no_val_shard_hash(
    tmp_path: Path,
) -> None:
    """real_ft_run's ft recipe (_train) carries `shard_hash` only, while the val metric rides
    every row from `_RUN_METRICS` once open_val_reader ran (before _train, in main). With a
    valid byte-identity record for both sets the ft row still refuses: the lift looks up
    recipe['val_shard_hash'] == None. The lift is unreachable for ft rows."""
    # The ft row records its val set where open_val_reader writes it (item 3): the hash metric.
    # With no val hash anywhere on the row no record can name its set (see the companion test
    # in test_promotion_shard_not_covered.py).
    led = _led(tmp_path, {SHARD_NOT_COVERED_METRIC: COVERED, VAL_SHARD_NOT_COVERED_METRIC: NC,
                          VAL_SHARD_HASH_METRIC: Ran(passed=True, value=VAL)},
               {"shard_hash": TRAIN})
    # The cited evidence must exist (item 4 of the lead's review): the record sits beside it.
    (tmp_path / "a.md").write_text("train byte comparison", encoding="utf-8")
    (tmp_path / "b.md").write_text("val byte comparison", encoding="utf-8")
    ident = _identity(
        tmp_path,
        {"shard_hash": TRAIN, "identical": True, "audit": "a.md"},
        {"shard_hash": VAL, "identical": True, "audit": "b.md"},
    )
    v = _verdict(led, tmp_path, ident)
    assert not _blocked(v), v.reasons


# ---------------------------------------------------------------- target 3: every ft-row writer


def test_BREAK_the_rust_metal_ft_row_records_the_not_covered_metric() -> None:
    """crates/qd-train-metal writes run_kind 'ft' rows through qd_train::ledger::FtRow with the
    reader in hand (`reader.check("shard_code_not_covered")` exists) and never records
    shard.code_not_covered, so a Rust-written ft row on a v2 set is unmarked."""
    src = (REPO / "crates" / "qd-train-metal" / "src" / "run.rs").read_text(encoding="utf-8")
    assert SHARD_NOT_COVERED_METRIC in src, "run.rs never writes shard.code_not_covered"


def test_BREAK_the_lora_reference_run_records_the_not_covered_metric() -> None:
    """tools/lora_reference_run.py (the LoRA-2B torch reference arm, reads a ShardReader and
    writes run_kind 'ft' rows through its own RunRecorder, not real_ft_run._recorder)."""
    src = (REPO / "tools" / "lora_reference_run.py").read_text(encoding="utf-8")
    assert "not_covered" in src, "lora_reference_run.py never records the coverage verdict"


# ---------------------------------------------------------------- target 4: the original gap


def test_BREAK_the_original_gap_is_resolved_by_a_later_append() -> None:
    gap_id = "GAP-SHARD-FINGERPRINT-LIST-FIX-REFUSES-EVERY-PINNED-V2-SET-2026-10-23"
    last = None
    for line in (REPO / "gaps.jsonl").read_text(encoding="utf-8").splitlines():
        if gap_id in line:
            rec = json.loads(line)
            if rec.get("id") == gap_id:
                last = rec
    assert last is not None
    assert str(last["status"]).startswith("resolved"), f"still {last['status']!r}"
