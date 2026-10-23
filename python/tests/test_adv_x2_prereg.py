"""Adversarial review of lane X2 (span-refit span null, lora-2b-q8 prereg).

Each test states the invariant the amendment / pre-registration claims and attacks it. A test
that fails is a break in X2's work; one that passes is a held attack.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
import span_refit_rows as srr  # noqa: E402
from ft_linear_control import Refused  # noqa: E402

from qd_train import ledger as L  # noqa: E402
from qd_train.span_null import (  # noqa: E402
    PLUS,
    SPAN_END_TOP1_METRIC,
    SPAN_NULL_METRIC,
    SPAN_START_TOP1_METRIC,
)
from qd_train.tristate import NotRun, Ran  # noqa: E402

THREE = (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC, SPAN_NULL_METRIC)
NEW_SHA = next(iter(L.SPAN_REFIT_SPAN_NULL_SCORER_SHAS))
FIXTURE = REPO / "crates/qd-train/tests/fixtures/span-refit-gate-cases.json"


def _scorer():
    spec = importlib.util.spec_from_file_location("adv_scorer", REPO / "tools" / "span_refit_score.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vc(mod, rows: list[dict]):
    vc = mod.ValCache.__new__(mod.ValCache)
    vc.rows = rows
    vc.abstaining = np.array([bool(r["abstaining"]) for r in rows])
    vc.gold = np.array([[r["gold_start"], r["gold_end"]] for r in rows], dtype=np.int64).reshape(-1, 2)
    return vc


def _row(n_cand: int, plus: list[int], gs: int, ge: int, abstain: bool = False) -> dict:
    return {"abstaining": abstain, "n_candidates": n_cand, "plus_columns": plus,
            "gold_start": gs, "gold_end": ge}


# --- target 2: scorer vs real_ft_run.span_pointer_states ---------------------------------------


def _differential(seed: int) -> None:
    rnd = random.Random(seed)
    mod = _scorer()
    rows, verdicts, labels, tops = [], [], [], []
    for i in range(rnd.randint(1, 30)):
        n = rnd.randint(1, 8)
        plus = sorted(rnd.sample(range(n), rnd.randint(1, n)))
        abstain = rnd.random() < 0.3
        gs = -1 if abstain else rnd.randrange(n)
        ge = -1 if abstain else rnd.randrange(n)
        rows.append(_row(n, plus, gs, ge, abstain))
        ts, te = rnd.randrange(n), rnd.randrange(n)
        tops.append([ts, te])
        first = tuple(PLUS if j in plus else 0 for j in range(n))
        verdicts.append({"kind": "span", "expected_abstain": abstain, "row_id": f"r{i}",
                         "slot_name": "s", "noul_row": n, "gold_start": gs, "gold_end": ge,
                         "top": [ts, te]})
        labels.append(SimpleNamespace(row_id=f"r{i}", slot_name="s", line_first_bytes=first))
    vc = _vc(mod, rows)
    mine = vc.pointer_top1(np.array(tops, dtype=np.int64))
    mine_null = mod.pointing_added_line_null(vc) if not vc.abstaining.all() else None
    theirs = rft.span_pointer_states({"verdicts": verdicts}, labels)  # type: ignore[arg-type]
    if vc.abstaining.all():
        assert all(isinstance(t, NotRun) for t in theirs.values())
        return
    for name in (SPAN_START_TOP1_METRIC, SPAN_END_TOP1_METRIC):
        t = theirs[name]
        assert isinstance(t, Ran)
        assert (t.n, t.n_total) == (mine[name]["n"], mine[name]["n_total"]), name
    t = theirs[SPAN_NULL_METRIC]
    assert isinstance(t, Ran) and mine_null is not None
    assert (t.n, t.n_total) == (mine_null["n"], mine_null["n_total"])
    assert t.value == pytest.approx(mine_null["value"])


@pytest.mark.parametrize("seed", range(40))
def test_scorer_pointer_counts_equal_span_pointer_states(seed: int) -> None:
    _differential(seed)


def test_zero_pointing_rows_is_not_a_ran_reading() -> None:
    """real_ft_run: no pointing row -> NotRun. The scorer + recorder must not make that Ran."""
    mod = _scorer()
    vc = _vc(mod, [_row(3, [], -1, -1, True), _row(2, [], -1, -1, True)])
    tops = np.array([[0, 0], [1, 1]], dtype=np.int64)
    p = vc.pointer_top1(tops)
    try:
        null = mod.pointing_added_line_null(vc)
    except Exception:  # noqa: BLE001  a loud refusal is acceptable
        return
    block = {"span_pointers": {"cache": "fp32_live", **p}}
    try:
        got = srr.span_null_metrics({"nulls": {"pointing_added_line": null}}, block, "a")
    except Refused:
        return
    assert not any(isinstance(m, Ran) for m in got.values()), (
        f"zero pointing rows recorded as ran: {got}")


def test_all_rows_pointing_counts() -> None:
    mod = _scorer()
    vc = _vc(mod, [_row(3, [0], 0, 0), _row(3, [1], 1, 2)])
    p = vc.pointer_top1(np.array([[0, 0], [1, 1]], dtype=np.int64))
    assert p[SPAN_START_TOP1_METRIC] == {"n": 2, "n_total": 2}
    assert p[SPAN_END_TOP1_METRIC] == {"n": 1, "n_total": 2}


def _recorded(**over) -> dict:
    null = {"value": 0.6, "n": 40, "n_total": 50}
    null.update(over.pop("null", {}))
    block = {"span_pointers": {"cache": "bf16_cache",
                               SPAN_START_TOP1_METRIC: {"n": 30, "n_total": 50},
                               SPAN_END_TOP1_METRIC: {"n": 28, "n_total": 50}}}
    for k, v in over.items():
        block["span_pointers"][k] = v
    return srr.span_null_metrics({"nulls": {"pointing_added_line": null}}, block, "c")


def test_recorder_refuses_a_missing_pointer() -> None:
    block = {"span_pointers": {"cache": "x", SPAN_START_TOP1_METRIC: {"n": 1, "n_total": 2}}}
    with pytest.raises(Refused):
        srr.span_null_metrics({"nulls": {"pointing_added_line": {"value": 0.5, "n": 1, "n_total": 2}}},
                              block, "c")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -0.5, 1.5])
def test_recorder_refuses_a_non_probability_null(bad: float) -> None:
    with pytest.raises((Refused, ValueError)):
        got = _recorded(null={"value": bad})
        # if it did not refuse, the value must at least not be a NaN/inf/out-of-range 'ran'
        assert all(math.isfinite(m.value) and 0.0 <= m.value <= 1.0 for m in got.values()
                   if isinstance(m, Ran) and m.value is not None)
        raise AssertionError(f"recorded null value {bad!r} as ran")


def test_recorder_refuses_start_and_null_stating_different_row_counts() -> None:
    """The recorder is the writer; the ledger check is the backstop, not the first line."""
    with pytest.raises((Refused, ValueError)):
        _recorded(null={"n_total": 49})
        raise AssertionError("start n_total 50 vs null n_total 49 was recorded")


@pytest.mark.parametrize("hit", [{"n": 51, "n_total": 50}, {"n": -1, "n_total": 50},
                                 {"n": 1.5, "n_total": 50}, {"n": True, "n_total": 50}])
def test_recorder_refuses_nonsense_counts(hit: dict) -> None:
    with pytest.raises((Refused, ValueError, TypeError)):
        _recorded(**{SPAN_START_TOP1_METRIC: hit})
        raise AssertionError(f"recorded {hit}")


# --- target 3: python gate attacks (Python side only; Rust is the cargo test) ------------------


def _line_of(name: str) -> dict:
    case = next(c for c in json.loads(FIXTURE.read_text()) if c["name"] == name)
    return json.loads(case["line"])


def _verdict(raw: dict) -> str:
    from qd_train.ledger import Ledger
    line = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l.jsonl"
        p.write_bytes(line.encode() + b"\n")
        try:
            Ledger(p).verify_chain()
        except ValueError as e:
            return "refuse" if ("span-null scorer" in str(e) or "pointing-row count" in str(e)) else f"other:{e}"
        except Exception as e:  # noqa: BLE001
            return f"other:{type(e).__name__}"
    return "accept"


def _stripped(**recipe_over) -> dict:
    raw = _line_of("new_sha_valid")
    raw["metrics"] = {}
    raw["recipe"].update(recipe_over)
    return raw


@pytest.mark.parametrize("tag", ["Span-Refit-a", " span-refit-a", "SPAN-REFIT-A"])
def test_a_tag_that_differs_in_case_or_whitespace_does_not_escape_the_check(tag: str) -> None:
    """A new-sha completed row with no span metrics must be refused whatever the tag's spelling
    (the scorer sha, not the tag, is what proves the null was emitted)."""
    assert _verdict(_stripped(tag=tag)) == "refuse"


@pytest.mark.parametrize("sha", [NEW_SHA.upper(), NEW_SHA + " ", " " + NEW_SHA])
def test_a_new_sha_spelled_differently_does_not_escape_the_check(sha: str) -> None:
    raw = _stripped()
    raw["recipe"]["script_sha256"] = {"span_refit_score.py": sha}
    assert _verdict(raw) == "refuse"


def test_the_validity_gate_exemption_is_arm_a_only() -> None:
    """Only arm (a) after its validity gate failed records no span reading. An arm (b) row
    claiming a failed arm-(a) gate must not take the exemption."""
    raw = _stripped(tag="span-refit-b")
    raw["gates"] = {L.SPAN_REFIT_VALIDITY_GATE: {"state": "ran", "passed": False, "value": 5000.0}}
    assert _verdict(raw) == "refuse"


def test_the_exemption_needs_the_gate_to_have_failed_not_merely_to_exist() -> None:
    raw = _stripped(tag="span-refit-a")
    raw["gates"] = {L.SPAN_REFIT_VALIDITY_GATE: {"state": "ran", "passed": True, "value": 5596.0}}
    assert _verdict(raw) == "refuse"
    raw["gates"] = {L.SPAN_REFIT_VALIDITY_GATE: {"state": "not_run", "reason": "x"}}
    assert _verdict(raw) == "refuse"


def test_old_sha_row_with_a_forged_new_sha_field_is_not_checked_as_new() -> None:
    """Held expectation: only script_sha256['span_refit_score.py'] decides; a forged extra field
    cannot make an old-scorer row (which has no null) look new."""
    raw = _stripped()
    raw["recipe"]["script_sha256"] = {"span_refit_score.py": "b" * 64}
    raw["recipe"]["scorer_sha256"] = NEW_SHA
    assert _verdict(raw) == "accept"


# --- target 1: q8 pooling ----------------------------------------------------------------------


def test_some_reader_keeps_q8_rows_apart_from_bf16_rows() -> None:
    """The pre-registration says the pool key is (family id, base.format, base-file sha256).

    The attack: a q8 row and a bf16 row whose Protocol is equal, so ``Protocol.hash_without_seed``
    (the only pooling the ledger does) puts them in one seed family. The separation is enforced
    at the promotion reader (``_base_pool_refusals`` in ``promotion_verdict``), not in Protocol,
    because a Protocol field would re-key every protocol hash (lead ruling, 2026-10-22; the
    original version of this test asserted a Protocol field and was rewritten to assert the
    outcome). The outcome: ``promotion_verdict`` over the one pool refuses and names
    ``recipe.base.format``.
    """
    from test_ledger import _env, _protocol

    def row(rid: str, seed: int, recipe: dict) -> L.LedgerRow:
        return L.LedgerRow(
            row_id=rid, written_at="2026-10-22T00:00:00+00:00", prev_row_hash=None,
            protocol=_protocol(seed), run_kind="ft", status="completed", quick=False,
            quick_reason=None, code_commit="c", env=_env(), metrics={},
            noul_rate=NotRun(reason="n/a"), controls={}, gates={}, wall_clock_s=1.0,
            cost_usd=0.0, recipe=recipe)

    q8 = row("q8-0", 0, {"family": "lora-2b-q8", "base": {
        "format": "bf16-fakequant-q8-g64", "file_sha256": "ab" * 32}})
    bf16 = row("bf16-1", 1, {"family": "lora-2b"})
    assert q8.protocol.hash_without_seed() == bf16.protocol.hash_without_seed()
    with tempfile.TemporaryDirectory() as td:
        ledger = L.Ledger(Path(td) / "l.jsonl")
        ledger.append(q8)
        ledger.append(bf16)
        verdict = ledger.promotion_verdict(q8.protocol.hash_without_seed())
    assert not verdict.promoted
    assert any("recipe.base.format" in r and "never pooled with a bf16 row" in r
               for r in verdict.reasons), verdict.reasons


def test_d3_verdict_from_a_looser_copy_is_distinguishable_from_the_committed_prereg(
    tmp_path: Path,
) -> None:
    """Formerly ``test_d3_load_spec_refuses_a_bar_looser_than_section_5``, rewritten (lead ruling
    on item 7, 2026-10-22): ``tools/d3_quant_gate.py`` deliberately holds no second copy of the
    bars (rule 2), so ``load_spec`` loads whatever pre-registration it is handed. The guard that
    was accepted is audit: (1) the committed pre-registration's bars equal section 5 (pinned by
    ``test_lora_2b_q8_prereg.py``), and (2) a verdict produced from a looser copy records that
    copy's path and sha256 and the bars it ran with, all different from the committed file's, so
    an auditor can tell the two apart.
    """
    import hashlib

    import d3_quant_gate as gate
    import test_lora_2b_q8_prereg as pin

    pin.test_numbers_equal_the_owner_decision_prose()  # (1) the committed bars are section 5
    committed = REPO / "campaign/lora-2b-q8-preregistered.json"
    doc = copy.deepcopy(json.loads(committed.read_text()))
    doc["d3_gate"]["argmax_agreement_min"] = 0.5
    doc["d3_gate"]["min_prompts"] = 1
    loose = tmp_path / "loose.json"
    loose.write_text(json.dumps(doc))
    assert isinstance(gate.load_spec(loose), gate.GateSpec)  # the tool does load it

    def verdict_of(prereg: Path) -> dict:
        out = tmp_path / f"{prereg.stem}-verdict.json"
        gate.main(["--preregistration", str(prereg), "--model", str(tmp_path / "no-model"),
                   "--items", str(tmp_path / "items.jsonl"), "--out", str(out)])
        return json.loads(out.read_text())

    loose_v, real_v = verdict_of(loose), verdict_of(committed)
    assert loose_v["preregistration"] == str(loose) != real_v["preregistration"]
    assert loose_v["preregistration_sha256"] == hashlib.sha256(loose.read_bytes()).hexdigest()
    assert real_v["preregistration_sha256"] == hashlib.sha256(committed.read_bytes()).hexdigest()
    assert loose_v["preregistration_sha256"] != real_v["preregistration_sha256"]
    assert loose_v["spec"]["argmax_agreement_min"] == 0.5  # the loose bars are on the record
    assert real_v["spec"]["argmax_agreement_min"] == 0.99


@pytest.mark.parametrize("agree,delta,ok", [(0.99, 0.02, True), (0.9899999, 0.02, False),
                                            (0.99, 0.0200001, False), (1.0, 0.0, True)])
def test_d3_decide_boundaries_follow_section_5(agree: float, delta: float, ok: bool) -> None:
    import d3_quant_gate as gate
    spec = gate.load_spec(REPO / "campaign/lora-2b-q8-preregistered.json")
    assert isinstance(spec, gate.GateSpec)
    got = gate.decide(spec, gate.Metrics(n=500, argmax_agreement=agree, mean_abs_delta_logp=delta),
                      n_total=500)
    assert isinstance(got, Ran) and got.passed is ok
    assert isinstance(gate.decide(spec, gate.Metrics(n=499, argmax_agreement=1.0,
                                                     mean_abs_delta_logp=0.0), n_total=499), NotRun)
