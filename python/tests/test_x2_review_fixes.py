"""Fixes for the adversarial review of lane X2 that `test_adv_x2_prereg.py` does not itself pin.

* the adversarial fixture's recorded Python verdicts are Python's current ones (so the Rust test
  that reads them cannot go stale);
* a q8 row and a bf16 row of an otherwise equal protocol do not pool (promotion reader);
* the recorder records NotRun, never a ran reading, over zero pointing rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_ledger import _env, _protocol

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "python" / "tests"))

import adv_x2_gen as adv  # noqa: E402
import span_refit_rows as srr  # noqa: E402
from ft_linear_control import Refused  # noqa: E402

from qd_train import ledger as L  # noqa: E402
from qd_train.span_null import (  # noqa: E402
    SPAN_END_TOP1_METRIC,
    SPAN_NULL_METRIC,
    SPAN_START_TOP1_METRIC,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

ADV_FIXTURE = REPO / "crates/qd-train/tests/fixtures/adv-x2-span-refit-cases.json"
HEX = "ab" * 32


def test_the_adversarial_fixtures_python_verdicts_are_current() -> None:
    cases = json.loads(ADV_FIXTURE.read_text(encoding="utf-8"))
    assert {c["name"] for c in cases} == set(adv.cases)
    stale = [
        c["name"] for c in cases if adv.verdict(adv.cases[c["name"]]) != c["python"]
    ]
    assert not stale, f"rerun python/tests/adv_x2_gen.py <fixture>: stale {stale}"


# --- q8 pooling --------------------------------------------------------------------------------


def _row(rid: str, seed: int, recipe: dict) -> L.LedgerRow:
    return L.LedgerRow(
        row_id=rid, written_at="2026-10-22T00:00:00+00:00", prev_row_hash=None,
        protocol=_protocol(seed),
        run_kind="ft", status="completed", quick=False, quick_reason=None, code_commit="c",
        env=_env(), metrics={}, noul_rate=NotRun(reason="n/a"), controls={}, gates={},
        wall_clock_s=1.0, cost_usd=0.0, recipe=recipe,
    )


def _q8(rid: str, seed: int, **base: object) -> L.LedgerRow:
    b = {"format": "bf16-fakequant-q8-g64", "file_sha256": HEX, **base}
    return _row(rid, seed, {"family": "lora-2b-q8", "base": b})


def test_a_q8_row_and_a_bf16_row_of_an_equal_protocol_do_not_pool(tmp_path: Path) -> None:
    q8 = _q8("q8-0", 0)
    bf16 = _row("bf16-1", 1, {"family": "lora-2b"})
    assert q8.protocol.hash_without_seed() == bf16.protocol.hash_without_seed()  # one pool key
    refusals = L._base_pool_refusals([q8, bf16])
    assert any("never pooled with a bf16 row" in r for r in refusals), refusals
    ledger = L.Ledger(tmp_path / "l.jsonl")
    ledger.append(q8)
    ledger.append(bf16)
    verdict = ledger.promotion_verdict(q8.protocol.hash_without_seed())
    assert not verdict.promoted
    assert any("never pooled with a bf16 row" in r for r in verdict.reasons), verdict.reasons


def test_q8_rows_that_differ_in_the_base_file_sha_do_not_pool() -> None:
    other = _q8("q8-1", 1, file_sha256="cd" * 32)
    assert L._base_pool_refusals([_q8("q8-0", 0), other])


def test_a_q8_row_without_its_base_format_and_sha_is_refused() -> None:
    bare = _row("q8-bare", 0, {"family": "lora-2b-q8"})
    assert any("must record recipe.base.format" in r for r in L._base_pool_refusals([bare]))
    no_sha = _q8("q8-nosha", 0, file_sha256=None)
    assert any("must record recipe.base.format" in r for r in L._base_pool_refusals([no_sha]))


def test_equal_q8_rows_pool_and_bf16_rows_pool() -> None:
    assert L._base_pool_refusals([_q8("a", 0), _q8("b", 1), _q8("c", 2)]) == []
    plain = {"family": "lora-2b"}
    assert L._base_pool_refusals([_row("a", 0, plain), _row("b", 1, plain)]) == []


# --- the recorder ------------------------------------------------------------------------------


def _block(n: int, n_total: int) -> dict:
    return {"span_pointers": {
        "cache": "fp32_live",
        SPAN_START_TOP1_METRIC: {"n": n, "n_total": n_total},
        SPAN_END_TOP1_METRIC: {"n": n, "n_total": n_total}}}


def test_the_recorder_writes_not_run_over_zero_pointing_rows() -> None:
    null = {"value": 0.0, "n": 0, "n_total": 0}
    got = srr.span_null_metrics({"nulls": {"pointing_added_line": null}}, _block(0, 0), "a")
    assert set(got) == {SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC, SPAN_NULL_METRIC}
    assert all(isinstance(m, NotRun) for m in got.values())


def test_the_recorder_refuses_zero_for_some_and_rows_for_others() -> None:
    null = {"value": 0.5, "n": 1, "n_total": 4}
    with pytest.raises(Refused):
        srr.span_null_metrics({"nulls": {"pointing_added_line": null}}, _block(0, 0), "a")


def test_all_three_not_run_is_accepted_by_the_ledger_and_mixed_is_not() -> None:
    sha = next(iter(L.SPAN_REFIT_SPAN_NULL_SCORER_SHAS))
    recipe = {"tag": "span-refit-c", "script_sha256": {"span_refit_score.py": sha}}
    none = {n: NotRun(reason="no pointing row") for n in L.SPAN_REFIT_METRIC_NAMES}
    row = _row("nr", 0, recipe)
    row.metrics.update(none)
    row.check_span_refit_span_null()
    row.metrics[SPAN_END_TOP1_METRIC] = Ran(passed=True, value=0.5, n=1, n_total=2)
    with pytest.raises(ValueError, match="span-null scorer"):
        row.check_span_refit_span_null()
