"""The two promotion re-specifications the human approved on 2026-10-08
(HANDOFF/promotion-respec-proposal-2026-10-05.md):

- ``paired_margin_population``: paired_margin_vs_linear is judged on the promotion population's
  families, from each seed's ``paired_margin_vs_linear.choice.<family>``; every other family is
  report-only (GAP-PAIRED-MARGIN-POOL-INCLUDES-A-FAMILY-WITH-NO-TRAINING-ROWS-2026-10-05).
- ``shuffled_label_seeds``: at least PROMOTION_MIN_SEEDS seeds carry a ran-and-passed
  shuffled_label control; one that ran and failed on any seed still blocks
  (GAP-SHUFFLED-LABEL-REQUIRED-ON-EVERY-SEED-2026-10-05).

The decided entries are read from docs/promotion-decisions.respec-2026-10-08.patch.json, the
exact text the human is to copy into docs/promotion-decisions.json, so these tests also prove
the patch is a record the verdict applies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promotion_fixtures import record_scored_row_metrics, write_record

from qd_train.ledger import (
    PROMOTION_MIN_SEEDS,
    REQUIRED_CONTROLS,
    REQUIRED_DECISIONS,
    REQUIRED_GATES,
    DecisionsRecordError,
    Environment,
    Ledger,
    PromotionVerdict,
    Protocol,
    RunRecorder,
    load_promotion_decisions,
)
from qd_train.tristate import NotRun, Ran, TriState

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "docs" / "promotion-decisions.respec-2026-10-08.patch.json"
MARGIN = "paired_margin_population"
SHUFFLED = "shuffled_label_seeds"
POPULATION_FAMILY = "code.defect_class"

#: v5 seed 0's control row 0b0ac9a8, in shape: the pooled gate cannot run (synth.general has no
#: training rows), the defect family passes, intent.in_scope's interval lies below zero.
POOLED_NOT_RUN = NotRun(reason="a task's control did not run: synth.general/access_route")
DEFECT_PASS = Ran(passed=True, value=0.3898, n=2304, n_total=2304)
IN_SCOPE_FAIL = Ran(passed=False, value=-0.0172, n=600, n_total=600)
SYNTH_NOT_RUN = NotRun(reason="synth.general has no training rows")
GREEN = Ran(passed=True, value=1.0, n=300, n_total=300)
_ABSENT = "absent"


def _patch_entries() -> dict[str, dict[str, object]]:
    return json.loads(PATCH.read_text(encoding="utf-8"))["decisions"]


def _open(value: str) -> dict[str, object]:
    return {"status": "open", "value": value, "decided_by": None, "decided_on": None,
            "decision_ref": None}


def _record(tmp_path: Path, *, margin: dict[str, object] | None = None,
            shuffled: dict[str, object] | None = None, population: bool = True) -> Path:
    """The as-built fixture record with promotion_population decided to code.defect_class
    (unless ``population`` is False), and the two re-spec entries: open by default, or the
    patch's decided entry with any field overridden."""
    record = json.loads(write_record(tmp_path).read_text(encoding="utf-8"))
    patch = _patch_entries()
    entries = record["decisions"]
    entries[MARGIN] = {**patch[MARGIN], **_open("pooled, as built")}
    entries[SHUFFLED] = {**patch[SHUFFLED], **_open("every seed, as built")}
    if margin is not None:
        entries[MARGIN] = {**patch[MARGIN], **margin}
    if shuffled is not None:
        entries[SHUFFLED] = {**patch[SHUFFLED], **shuffled}
    if population:
        entries["promotion_population"].update(
            status="decided", value="code.defect_class only", families=[POPULATION_FAMILY],
            decided_by="a human (test)", decided_on="2026-10-08", decision_ref="test fixture")
    path = tmp_path / f"respec-{len(list(tmp_path.glob('respec-*.json')))}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


DECIDED: dict[str, object] = {}


def _protocol(seed: int) -> Protocol:
    return Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                    backbone_commit="b" * 40, recipe_hash="e" * 64, seed=seed)


def _env() -> Environment:
    return Environment(torch="2.12.1", transformers_sha="abc123", device="mps", host="test",
                       fla_present=NotRun(reason="test"),
                       causal_conv1d_present=NotRun(reason="test"))


def _family(tmp_path: Path, n_seeds: int = 3, *, pooled: TriState = POOLED_NOT_RUN,
            defect: dict[int, TriState | str] | None = None,
            shuffled: dict[int, TriState | str] | None = None) -> Ledger:
    """``n_seeds`` eval rows differing only in seed, each with the control row that
    supplements it (tools/ft_linear_control.py's shape: the pooled gate plus the per-family
    margins). Everything else green. ``defect`` and ``shuffled`` override per seed; the
    string ``"absent"`` leaves the metric or control off the rows (RunRecorder then records an
    unreported control as not_run)."""
    led = Ledger(tmp_path / "family.jsonl")
    for seed in range(n_seeds):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env(),
                         wall_clock_s=None, cost=None) as rec:
            for g in REQUIRED_GATES:
                if g != "paired_margin_vs_linear":
                    rec.gate(g, GREEN)
            for c in REQUIRED_CONTROLS:
                if c != "shuffled_label":
                    rec.control(c, GREEN)
            record_scored_row_metrics(rec)
        eval_id = led.rows()[-1].row_id
        with RunRecorder(led, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env(),
                         wall_clock_s=None, cost=None,
                         recipe={"tool": "ft_linear_control", "eval_row_id": eval_id}) as rec:
            rec.gate("paired_margin_vs_linear", pooled)
            d = (defect or {}).get(seed, DEFECT_PASS)
            if not isinstance(d, str):
                rec.metric(f"paired_margin_vs_linear.choice.{POPULATION_FAMILY}", d)
            rec.metric("paired_margin_vs_linear.choice.intent.in_scope", IN_SCOPE_FAIL)
            rec.metric("paired_margin_vs_linear.choice.synth.general", SYNTH_NOT_RUN)
            s = (shuffled or {}).get(seed, GREEN)
            if not isinstance(s, str):
                rec.control("shuffled_label", s)
    return led


def _verdict(led: Ledger, path: Path) -> PromotionVerdict:
    return led.promotion_verdict(_protocol(0).hash_without_seed(), decisions_path=path)


def _has(verdict: PromotionVerdict, text: str) -> bool:
    return any(text in r for r in verdict.reasons)


# --------------------------------------------------------------------------
# The record: both keys are required, and the patch is a record the verdict reads
# --------------------------------------------------------------------------

def test_both_respec_keys_are_required_decisions():
    assert MARGIN in REQUIRED_DECISIONS and SHUFFLED in REQUIRED_DECISIONS


@pytest.mark.parametrize("missing", [MARGIN, SHUFFLED])
def test_a_record_without_a_respec_key_is_refused(tmp_path: Path, missing: str):
    path = _record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["decisions"][missing]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DecisionsRecordError, match=missing):
        load_promotion_decisions(path)


def test_the_patch_holds_both_entries_as_the_proposal_words_them(tmp_path: Path):
    record = load_promotion_decisions(_record(tmp_path, margin=DECIDED, shuffled=DECIDED))
    m, s = record.decisions[MARGIN], record.decisions[SHUFFLED]
    assert m.gap == "GAP-PAIRED-MARGIN-POOL-INCLUDES-A-FAMILY-WITH-NO-TRAINING-ROWS-2026-10-05"
    assert s.gap == "GAP-SHUFFLED-LABEL-REQUIRED-ON-EVERY-SEED-2026-10-05"
    for d in (m, s):
        assert d.status == "decided" and d.decided_on == "2026-10-08"
        assert "HANDOFF/promotion-respec-proposal-2026-10-05.md" in (d.decision_ref or "")
    assert str(m.value).startswith("the promotion population: each seed's")
    assert str(s.value).startswith("3: at least three seeds")


# --------------------------------------------------------------------------
# paired_margin_population
# --------------------------------------------------------------------------

def test_open_margin_decision_keeps_the_pooled_gate_as_built(tmp_path: Path):
    """v5's shape: the pooled gate is not_run, so every seed blocks, whatever the family did."""
    verdict = _verdict(_family(tmp_path), _record(tmp_path))
    assert not verdict.promoted
    assert _has(verdict, "gate 'paired_margin_vs_linear' did not run")
    assert not any("paired_margin_population" in r for r in verdict.readings)


def test_open_margin_decision_ignores_the_family_metrics(tmp_path: Path):
    """As built, a pooled pass promotes even where the population family failed."""
    led = _family(tmp_path, pooled=GREEN, defect={0: Ran(passed=False, value=-0.1, n=2304,
                                                         n_total=2304)})
    assert _verdict(led, _record(tmp_path)).promoted


def test_a_passing_population_family_promotes_and_other_families_are_report_only(
    tmp_path: Path,
):
    """intent.in_scope ran and failed and synth.general did not run on every seed; neither is
    in the population, so neither blocks."""
    verdict = _verdict(_family(tmp_path), _record(tmp_path, margin=DECIDED))
    assert verdict.promoted, str(verdict)
    moved = [r for r in verdict.readings if "'paired_margin_vs_linear'" in r]
    assert len(moved) == 3
    assert "as built not_run" in moved[0]
    assert "paired_margin_population = per family over the population: PASS 0.3898" in moved[0]


def test_a_failing_population_family_blocks(tmp_path: Path):
    led = _family(tmp_path, defect={1: Ran(passed=False, value=-0.02, n=2304, n_total=2304)})
    verdict = _verdict(led, _record(tmp_path, margin=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "gate 'paired_margin_vs_linear' ran and FAILED (read under the "
                         "decisions record) [2304/2304]")


def test_a_population_family_that_did_not_run_is_not_run(tmp_path: Path):
    led = _family(tmp_path, defect={2: NotRun(reason="the defect control did not converge")})
    verdict = _verdict(led, _record(tmp_path, margin=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "paired_margin_vs_linear.choice.code.defect_class did not run")
    assert _has(verdict, "the defect control did not converge")


def test_a_missing_population_family_metric_is_not_run(tmp_path: Path):
    verdict = _verdict(_family(tmp_path, defect={0: _ABSENT}),
                       _record(tmp_path, margin=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "paired_margin_vs_linear.choice.code.defect_class is absent")


def test_a_capped_population_family_is_not_complete_coverage(tmp_path: Path):
    led = _family(tmp_path, defect={0: Ran(passed=True, value=0.39, n=2000, n_total=2304)})
    verdict = _verdict(led, _record(tmp_path, margin=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "gate 'paired_margin_vs_linear' passed on only 2000/2304")


def test_a_margin_decision_without_a_decided_population_is_not_run(tmp_path: Path):
    led = _family(tmp_path, pooled=GREEN)
    verdict = _verdict(led, _record(tmp_path, margin=DECIDED, population=False))
    assert not verdict.promoted
    assert _has(verdict, "paired_margin_population is decided over the promotion population, "
                         "and promotion_population is not decided")


def test_an_unknown_margin_value_is_not_run_never_as_built(tmp_path: Path):
    """The pooled gate passes; a ruling the code cannot apply still does not fall back to it."""
    led = _family(tmp_path, pooled=GREEN)
    verdict = _verdict(led, _record(tmp_path, margin={"value": "each family separately"}))
    assert not verdict.promoted
    assert _has(verdict, "paired_margin_population is decided as 'each family separately', a "
                         "value this verdict does not know how to apply")


# --------------------------------------------------------------------------
# shuffled_label_seeds
# --------------------------------------------------------------------------

def _five(tmp_path: Path, shuffled: dict[int, TriState | str]) -> Ledger:
    return _family(tmp_path, 5, pooled=GREEN, shuffled=shuffled)


def test_open_shuffled_decision_requires_the_control_on_every_seed(tmp_path: Path):
    verdict = _verdict(_five(tmp_path, {3: _ABSENT, 4: _ABSENT}), _record(tmp_path))
    assert not verdict.promoted
    assert _has(verdict, "control 'shuffled_label' did not run")


def test_three_of_five_seeds_passed_promotes(tmp_path: Path):
    assert PROMOTION_MIN_SEEDS == 3
    verdict = _verdict(_five(tmp_path, {3: _ABSENT, 4: NotRun(reason="J5' not run")}),
                       _record(tmp_path, shuffled=DECIDED))
    assert verdict.promoted, str(verdict)
    line = next(r for r in verdict.readings if "shuffled_label_seeds" in r)
    assert "on 3 seed(s) [0, 1, 2]" in line
    assert "not counted: seed 3" in line and "seed 4" in line


def test_two_of_five_seeds_passed_blocks(tmp_path: Path):
    verdict = _verdict(_five(tmp_path, {2: _ABSENT, 3: _ABSENT, 4: _ABSENT}),
                       _record(tmp_path, shuffled=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "control 'shuffled_label' ran and passed at complete coverage on 2 "
                         "seed(s) [0, 1]")


def test_three_passed_and_one_failed_blocks(tmp_path: Path):
    failed = Ran(passed=False, value=0.41, n=300, n_total=300)
    verdict = _verdict(_five(tmp_path, {3: failed, 4: _ABSENT}),
                       _record(tmp_path, shuffled=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "control 'shuffled_label' ran and FAILED [300/300]; under "
                         "shuffled_label_seeds a failure on any seed still blocks")


def test_a_control_that_did_not_run_never_counts(tmp_path: Path):
    not_run = NotRun(reason="the shuffled run was killed")
    verdict = _verdict(_five(tmp_path, {2: not_run, 3: not_run, 4: not_run}),
                       _record(tmp_path, shuffled=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "on 2 seed(s) [0, 1]")


def test_a_capped_shuffled_pass_does_not_count(tmp_path: Path):
    capped = Ran(passed=True, value=0.23, n=200, n_total=300)
    verdict = _verdict(_five(tmp_path, {2: capped, 3: _ABSENT, 4: _ABSENT}),
                       _record(tmp_path, shuffled=DECIDED))
    assert not verdict.promoted
    assert _has(verdict, "on 2 seed(s) [0, 1]")


def test_an_unknown_shuffled_value_is_not_run_never_as_built(tmp_path: Path):
    """Every seed's control passed; a ruling the code cannot apply still refuses."""
    verdict = _verdict(_five(tmp_path, {}), _record(tmp_path, shuffled={"value": "two seeds"}))
    assert not verdict.promoted
    assert _has(verdict, "control 'shuffled_label' did not run (read under the decisions "
                         "record) (shuffled_label_seeds is decided as 'two seeds', a value "
                         "this verdict does not know how to apply)")


def test_both_respecs_together_promote_v5s_shape(tmp_path: Path):
    """Pooled margin not_run on every seed, the shuffled control on three of five seeds."""
    led = _family(tmp_path, 5, shuffled={3: _ABSENT, 4: _ABSENT})
    assert not _verdict(led, _record(tmp_path)).promoted
    verdict = _verdict(led, _record(tmp_path, margin=DECIDED, shuffled=DECIDED))
    assert verdict.promoted, str(verdict)
