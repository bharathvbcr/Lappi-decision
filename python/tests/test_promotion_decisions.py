"""Fable round G, G1 and G8: the promotion population is data a human sets, and an average
of seeds has a promotion kind of its own that the open human items still block.

Rule 2 is the point of every test here. ``promotion_population`` is READ from
``docs/promotion-decisions.json`` and never inferred from code, and the ``avg`` kind adds
checks -- it never relaxes one. The test that matters most is
``test_an_average_is_not_promotable_under_the_current_decisions``: an average whose own gate
row is entirely green and whose inputs are clean is still REFUSED, and every reason is an
open human item. Its positive control (``..._promotes_once_every_human_item_is_decided``) is
what proves the refusal comes from the decisions record and not from a fixture that could
never have promoted anyway.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.ledger import (
    DEFAULT_DECISIONS_PATH,
    REQUIRED_CONTROLS,
    REQUIRED_DECISIONS,
    REQUIRED_GATES,
    DecisionsRecordError,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
    load_promotion_decisions,
)
from qd_train.tristate import NotRun, Ran

REPO = Path(__file__).resolve().parents[2]
#: real_ft_run's config.seed: what an averaged eval row's protocol carries (no single seed's).
PROTOCOL_SEED = 20260919


def _protocol(seed: int, *, recipe: str = "r", data: str = "d") -> Protocol:
    return Protocol(
        data_snapshot_hash=data * 64, tokenizer_hash="t" * 64, backbone_commit="b" * 40,
        recipe_hash=recipe * 64, seed=seed,
    )


def _env() -> Environment:
    return Environment(
        torch="2.12.1", transformers_sha="abc123", device="mps", host="test",
        fla_present=NotRun(reason="test"), causal_conv1d_present=NotRun(reason="test"),
    )


def _ft_row(led: Ledger, seed: int, *, termination: str | None = "steps_exhausted",
            quick: bool = False, recipe: str = "r", data: str = "d") -> str:
    with RunRecorder(
        led, protocol=_protocol(seed, recipe=recipe, data=data), run_kind="ft", repo=REPO,
        env=_env(), wall_clock_s=None, cost=None, quick=quick,
        quick_reason="subsample" if quick else None,
    ) as rec:
        if termination is not None:
            rec.metric("train.termination",
                       Ran(passed=termination != "wall_clock_cap", value=termination))
    return led.rows()[-1].row_id


def _avg_row(led: Ledger, ft_row_ids: list[str], seeds: list[int], *,
             failing_gate: str | None = None, margin_on_a_supplement: bool = False,
             averaged: bool = True) -> str:
    """An averaged eval row as real_ft_run --score-checkpoint AVG.safetensors writes it,
    with every gate and control green unless told otherwise."""
    recipe: dict[str, object] = {"tool": "tools/real_ft_run.py", "tag": "avg-score-val"}
    if averaged:
        recipe["averaged"] = {
            "seeds": seeds, "ft_row_ids": ft_row_ids, "manifest_sha256": "m" * 64,
            "source": "masters", "n_inputs": len(ft_row_ids), "protocol_seed": PROTOCOL_SEED,
            "suite_seed": PROTOCOL_SEED,
        }
    with RunRecorder(
        led, protocol=_protocol(PROTOCOL_SEED, recipe="e"), run_kind="eval", repo=REPO,
        env=_env(), wall_clock_s=None, cost=None, recipe=recipe,
    ) as rec:
        for g in REQUIRED_GATES:
            if margin_on_a_supplement and g == "paired_margin_vs_linear":
                continue
            passed = g != failing_gate
            rec.gate(g, Ran(passed=passed, value=1.0 if passed else 0.5, n=300, n_total=300))
        for c in REQUIRED_CONTROLS:
            rec.control(c, Ran(passed=True, n=300, n_total=300))
    avg_id = led.rows()[-1].row_id
    if margin_on_a_supplement:
        with RunRecorder(
            led, protocol=_protocol(PROTOCOL_SEED, recipe="c"), run_kind="eval", repo=REPO,
            env=_env(), wall_clock_s=None, cost=None,
            recipe={"tool": "ft_linear_control", "eval_row_id": avg_id},
        ) as rec:
            rec.gate("paired_margin_vs_linear", Ran(passed=True, value=0.1, n=300, n_total=300))
    return avg_id


def _clean_average(tmp_path: Path, **avg_kw: object) -> tuple[Ledger, Ledger, str]:
    """J4's shape: three ft rows in one ledger, the average's eval row in another."""
    ft = Ledger(tmp_path / "ft.jsonl")
    ids = [_ft_row(ft, seed) for seed in (0, 1, 2)]
    avg = Ledger(tmp_path / "avg.jsonl")
    return avg, ft, _avg_row(avg, ids, [0, 1, 2], **avg_kw)  # type: ignore[arg-type]


def _record(tmp_path: Path, **changes: dict[str, object]) -> Path:
    """The repo's record with some entries changed -- what a human edit would look like."""
    record = json.loads(DEFAULT_DECISIONS_PATH.read_text(encoding="utf-8"))
    for name, fields in changes.items():
        record["decisions"][name].update(fields)
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _decided(value: object, **extra: object) -> dict[str, object]:
    return {"status": "decided", "value": value, "decided_by": "a human (test)",
            "decided_on": "2026-10-02", "decision_ref": "test fixture", **extra}


def _everything_decided(tmp_path: Path, *, average: bool = True) -> Path:
    return _record(
        tmp_path,
        promotion_population=_decided("code.defect_class only", families=["code.defect_class"]),
        average_may_promote=_decided(average),
        ece_population=_decided("per shape and per language"),
        degenerate_head_floor=_decided("0.15 nats"),
        privileged_hunk_pass_rule=_decided("defined"),
        transfer_gate_definition=_decided("defined"),
    )


# --------------------------------------------------------------------------
# G1: promotion_population is the human's data, carried on every verdict
# --------------------------------------------------------------------------

def test_the_record_in_the_repo_states_the_pooled_population_as_open():
    """The CURRENT population, as built, with its source cited -- and no ruling on it."""
    decisions = load_promotion_decisions()
    pop = decisions.population
    assert pop.status == "open"
    assert pop.families == "all"
    assert pop.value.startswith("pooled")
    assert pop.gap == "GAP-GATES-POOL-THE-GENERAL-FAMILIES-INTO-DEFECT-CONTRACTS"
    assert "permutation_agreement" in pop.source and "choice_rule_abstentions" in pop.source
    assert (pop.decided_by, pop.decided_on, pop.decision_ref) == (None, None, None)
    assert pop.record_sha256 == hashlib.sha256(DEFAULT_DECISIONS_PATH.read_bytes()).hexdigest()
    assert set(REQUIRED_DECISIONS) <= set(decisions.decisions)


def test_a_seed_family_verdict_carries_the_recorded_population(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        _ft_row(led, seed)
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert verdict.kind == "seeds"
    assert verdict.promotion_population == load_promotion_decisions().population
    assert "promotion_population: pooled" in str(verdict)
    assert "open (GAP-GATES-POOL-THE-GENERAL-FAMILIES-INTO-DEFECT-CONTRACTS)" in str(verdict)


def test_the_population_is_read_from_the_record_never_inferred(tmp_path: Path):
    """Change the data, and the verdict changes with it; nothing in code picks the value."""
    path = _record(
        tmp_path,
        promotion_population=_decided("code.defect_class only",
                                      families=["code.defect_class"]),
    )
    led = Ledger(tmp_path / "runs.jsonl")
    _ft_row(led, 1)
    pop = led.promotion_verdict(_protocol(1).hash_without_seed(),
                                decisions_path=path).promotion_population
    assert pop is not None
    assert (pop.value, pop.families, pop.status) == (
        "code.defect_class only", ("code.defect_class",), "decided")
    assert (pop.decided_by, pop.decision_ref) == ("a human (test)", "test fixture")


@pytest.mark.parametrize("breakage", [
    "missing_file", "decided_without_a_name", "open_with_a_decider", "a_missing_question",
    "bad_status", "bad_date", "wrong_schema", "families_empty", "open_average_says_yes",
])
def test_a_record_that_does_not_validate_refuses_the_verdict(tmp_path: Path, breakage: str):
    """Fail closed: a verdict that cannot say which population it judged does not promote."""
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        _ft_row(led, seed)
    record = json.loads(DEFAULT_DECISIONS_PATH.read_text(encoding="utf-8"))
    entry = record["decisions"]["promotion_population"]
    if breakage == "decided_without_a_name":
        entry.update(status="decided", decided_on="2026-10-02", decision_ref="x")
    elif breakage == "open_with_a_decider":
        entry.update(decided_by="someone")
    elif breakage == "a_missing_question":
        del record["decisions"]["transfer_gate_definition"]
    elif breakage == "bad_status":
        entry.update(status="tentative")
    elif breakage == "bad_date":
        entry.update(status="decided", decided_by="h", decided_on="yesterday", decision_ref="x")
    elif breakage == "wrong_schema":
        record["schema"] = "qd.promotion-decisions.v0"
    elif breakage == "families_empty":
        entry.update(families=[])
    elif breakage == "open_average_says_yes":
        record["decisions"]["average_may_promote"]["value"] = True
    path = tmp_path / "decisions.json"
    if breakage != "missing_file":
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DecisionsRecordError):
        load_promotion_decisions(path)
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed(), decisions_path=path)
    assert not verdict.promoted
    assert verdict.promotion_population is None
    assert any("promotion decisions record" in r for r in verdict.reasons), verdict.reasons


def test_an_all_green_seed_family_still_promotes_exactly_as_before(tmp_path: Path):
    """The field is added; the seed-family kind's pass/fail logic is not touched."""
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env(),
                         wall_clock_s=None, cost=None) as rec:
            for g in REQUIRED_GATES:
                rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
            rec.metric("train.termination", Ran(passed=True, value="steps_exhausted"))
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert verdict.promoted, str(verdict)


# --------------------------------------------------------------------------
# G8: the avg kind
# --------------------------------------------------------------------------

def test_an_average_is_not_promotable_under_the_current_decisions(tmp_path: Path):
    """The required test. Every gate and control on the average's own row passes at full
    coverage, its three inputs are clean -- and it is REFUSED, by the open human items and by
    nothing else."""
    avg, ft, avg_id = _clean_average(tmp_path)
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[ft])
    assert verdict.kind == "avg"
    assert not verdict.promoted, str(verdict)
    record = load_promotion_decisions()
    open_gaps = {d.gap for d in record.open()}
    assert "GAP-AVERAGE-PROMOTION-IS-THE-HUMANS-DECISION" in open_gaps
    assert len(open_gaps) == len(REQUIRED_DECISIONS)
    for gap in open_gaps:
        assert any(gap in r for r in verdict.reasons), (gap, verdict.reasons)
    # Nothing but the human items refuses it: proof the fixture itself was promotable.
    assert all(r.startswith("open human decision") for r in verdict.reasons), verdict.reasons
    assert verdict.promotion_population == record.population


def test_the_same_average_promotes_once_every_human_item_is_decided(tmp_path: Path):
    """The positive control. Without it, the test above could pass on a fixture that never
    could have promoted."""
    avg, ft, avg_id = _clean_average(tmp_path)
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[ft],
                                        decisions_path=_everything_decided(tmp_path))
    assert verdict.promoted, str(verdict)
    assert verdict.promotion_population is not None
    assert verdict.promotion_population.families == ("code.defect_class",)
    assert "average of seeds [0, 1, 2]" in verdict.reasons[0]


def test_a_human_no_on_averages_refuses_every_average(tmp_path: Path):
    avg, ft, avg_id = _clean_average(tmp_path)
    verdict = avg.promotion_verdict_avg(
        avg_id, input_ledgers=[ft], decisions_path=_everything_decided(tmp_path, average=False))
    assert not verdict.promoted
    assert verdict.reasons == (
        "the human decided that an average may not be the promoted artifact "
        "(average_may_promote, GAP-AVERAGE-PROMOTION-IS-THE-HUMANS-DECISION, ref: test fixture)",
    )


def test_an_averages_own_gate_row_must_be_full(tmp_path: Path):
    avg, ft, avg_id = _clean_average(tmp_path, failing_gate="needle_hunk_recall")
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[ft],
                                        decisions_path=_everything_decided(tmp_path))
    assert not verdict.promoted
    assert verdict.reasons == (
        f"{avg_id}: gate 'needle_hunk_recall' ran and FAILED [300/300]",
    )


def test_an_averages_supplement_joins_its_gate_row(tmp_path: Path):
    """The linear control lands on a row of its own naming the average, as for one seed."""
    avg, ft, avg_id = _clean_average(tmp_path, margin_on_a_supplement=True)
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[ft],
                                        decisions_path=_everything_decided(tmp_path))
    assert verdict.promoted, str(verdict)
    assert len(verdict.rows) == 5  # the average, its supplement, its three inputs


def _with_inputs(tmp_path: Path, rows: list[dict[str, object]], seeds: list[int] | None = None
                 ) -> tuple[Ledger, Ledger, str]:
    ft = Ledger(tmp_path / "ft.jsonl")
    ids = [_ft_row(ft, **kw) for kw in rows]  # type: ignore[arg-type]
    avg = Ledger(tmp_path / "avg.jsonl")
    seeds = seeds if seeds is not None else [int(kw["seed"]) for kw in rows]  # type: ignore[call-overload]
    return avg, ft, _avg_row(avg, ids, seeds)


@pytest.mark.parametrize(("rows", "expected"), [
    ([{"seed": 0, "quick": True}, {"seed": 1}, {"seed": 2}],
     "is quick (subsample); a quick run cannot be an average's input"),
    ([{"seed": 0, "termination": "wall_clock_cap"}, {"seed": 1}, {"seed": 2}],
     "train.termination is 'wall_clock_cap', not 'steps_exhausted'"),
    ([{"seed": 0, "termination": None}, {"seed": 1}, {"seed": 2}],
     "carries no train.termination"),
    ([{"seed": 0}, {"seed": 1}, {"seed": 2, "recipe": "x"}],
     "the inputs span 2 recipe_hash values"),
    ([{"seed": 0}, {"seed": 1}, {"seed": 2, "data": "x"}],
     "the inputs span 2 data_snapshot_hash values"),
    ([{"seed": 0}, {"seed": 1}],
     "2 distinct seed(s) [0, 1] among the inputs; an average promotes only over at least 3"),
])
def test_an_averages_inputs_are_non_quick_full_schedule_seeds_of_one_recipe_and_snapshot(
    tmp_path: Path, rows: list[dict[str, object]], expected: str,
):
    avg, ft, avg_id = _with_inputs(tmp_path, rows)
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[ft],
                                        decisions_path=_everything_decided(tmp_path))
    assert not verdict.promoted
    assert any(expected in r for r in verdict.reasons), verdict.reasons


def test_an_input_the_ledgers_do_not_hold_refuses(tmp_path: Path):
    avg, _, avg_id = _clean_average(tmp_path)
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[],
                                        decisions_path=_everything_decided(tmp_path))
    assert not verdict.promoted
    assert sum("is in none of the ledgers read" in r for r in verdict.reasons) == 3


def test_an_average_naming_other_seeds_than_its_inputs_refuses(tmp_path: Path):
    avg, ft, avg_id = _with_inputs(tmp_path, [{"seed": 0}, {"seed": 1}, {"seed": 2}],
                                   seeds=[0, 1, 3])
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[ft],
                                        decisions_path=_everything_decided(tmp_path))
    assert not verdict.promoted
    assert any("names seeds [0, 1, 3]; its inputs are seeds [0, 1, 2]" in r
               for r in verdict.reasons), verdict.reasons


def test_a_row_that_is_not_an_average_is_refused_by_the_avg_kind(tmp_path: Path):
    avg, ft, avg_id = _clean_average(tmp_path, averaged=False)
    verdict = avg.promotion_verdict_avg(avg_id, input_ledgers=[ft],
                                        decisions_path=_everything_decided(tmp_path))
    assert not verdict.promoted
    assert verdict.reasons == (
        f"{avg_id}: recipe carries no 'averaged' block; only an eval row "
        "real_ft_run.py --score-checkpoint wrote for an average is an avg candidate",
    )


def test_a_quick_average_row_refuses(tmp_path: Path):
    """real_ft_run marks an average quick on a seed shortfall or a per-seed rule-8 reason."""
    ft = Ledger(tmp_path / "ft.jsonl")
    ids = [_ft_row(ft, seed) for seed in (0, 1, 2)]
    avg = Ledger(tmp_path / "avg.jsonl")
    recipe = {"tool": "tools/real_ft_run.py", "tag": "avg-score-val", "averaged": {
        "seeds": [0, 1, 2], "ft_row_ids": ids, "manifest_sha256": "m" * 64,
        "source": "masters", "n_inputs": 3}}
    with RunRecorder(avg, protocol=_protocol(PROTOCOL_SEED, recipe="e"), run_kind="eval",
                     repo=REPO, env=_env(), wall_clock_s=None, cost=None, recipe=recipe,
                     quick=True, quick_reason="fewer than 3 seeds") as rec:
        for g in REQUIRED_GATES:
            rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
        for c in REQUIRED_CONTROLS:
            rec.control(c, Ran(passed=True, n=300, n_total=300))
    verdict = avg.promotion_verdict_avg(avg.rows()[-1].row_id, input_ledgers=[ft],
                                        decisions_path=_everything_decided(tmp_path))
    assert not verdict.promoted
    assert any("marked quick (fewer than 3 seeds)" in r for r in verdict.reasons)


def test_the_cli_refuses_the_average_under_the_current_record(tmp_path: Path):
    avg, ft, avg_id = _clean_average(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "qd_train.ledger", "verdict", "--ledger", str(avg.path),
         "--row-id", avg_id, "--kind", "avg", "--input-ledger", str(ft.path)],
        capture_output=True, text=True, timeout=120, cwd=REPO / "python", check=False,
    )
    assert proc.returncode == 1, proc.stderr
    assert proc.stdout.startswith("REFUSED [avg]"), proc.stdout
    assert "GAP-AVERAGE-PROMOTION-IS-THE-HUMANS-DECISION" in proc.stdout
    assert "promotion_population: pooled" in proc.stdout


def test_the_cli_avg_kind_needs_a_row_id(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "qd_train.ledger", "verdict", "--ledger",
         str(tmp_path / "x.jsonl"), "--seed-family", "f" * 64, "--kind", "avg"],
        capture_output=True, text=True, timeout=120, cwd=REPO / "python", check=False,
    )
    assert proc.returncode == 2
    assert "--kind avg needs --row-id" in proc.stderr
