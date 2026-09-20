"""S6: the ledger, and the tri-state that keeps 'not run' from reading as 'passed'.

Every test here corresponds to a line in docs/ledger-schema.md or docs/hardening.md.
The plan's S6 is not green until a deliberately killed run produces a row that says so
(``test_killed_run_writes_a_row_saying_so``), and S7 is not green until the harness
fails two deliberately broken models (``test_eval_harness.py``).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.ledger import (  # noqa: E402
    NOT_APPLICABLE,
    Environment,
    Ledger,
    LedgerChainError,
    LedgerRow,
    Protocol,
    RunRecorder,
)
from qd_train.tristate import NotRun, Ran, aggregate, parse_tristate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _protocol(seed: int) -> Protocol:
    return Protocol(
        data_snapshot_hash="d" * 64,
        tokenizer_hash="t" * 64,
        backbone_commit="b" * 40,
        recipe_hash="r" * 64,
        seed=seed,
    )


def _env() -> Environment:
    return Environment(
        torch="2.12.1", transformers_sha="abc123", device="mps", host="test",
        fla_present=NotRun(reason="no CUDA on this host"),
        causal_conv1d_present=NotRun(reason="no CUDA on this host"),
    )


def _all_green(rec: RunRecorder) -> None:
    from qd_train.ledger import REQUIRED_CONTROLS, REQUIRED_GATES

    for g in REQUIRED_GATES:
        rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
    for c in REQUIRED_CONTROLS:
        rec.control(c, Ran(passed=True, n=300, n_total=300))


# --------------------------------------------------------------------------
# The tri-state invariant
# --------------------------------------------------------------------------

def test_not_run_has_no_passed_attribute():
    """The structural guarantee: a report cannot render NotRun as passing."""
    nr = NotRun(reason="no CUDA")
    assert not hasattr(nr, "passed")


def test_not_run_requires_a_reason():
    with pytest.raises(ValueError, match="non-empty reason"):
        NotRun(reason="")
    with pytest.raises(ValueError, match="non-empty reason"):
        NotRun(reason="   ")


def test_ran_without_passed_is_refused_at_parse():
    with pytest.raises(ValueError, match="without a 'passed' field"):
        parse_tristate({"state": "ran"}, field="x")


def test_not_run_carrying_passed_is_refused_at_parse():
    """The exact shape by which a skipped suite becomes a green one."""
    with pytest.raises(ValueError, match="carries a 'passed' field"):
        parse_tristate({"state": "not_run", "reason": "skipped", "passed": True}, field="x")


def test_there_is_no_third_state():
    with pytest.raises(ValueError, match="no third state"):
        parse_tristate({"state": "unknown"}, field="x")


def test_coverage_is_carried_as_a_pair_or_not_at_all():
    with pytest.raises(ValueError, match="carried together or not at all"):
        Ran(passed=True, n=5)
    with pytest.raises(ValueError, match="n exceeds n_total"):
        Ran(passed=True, n=500, n_total=300)


def test_capped_sample_is_not_complete_coverage():
    assert Ran(passed=True, n=50, n_total=300).is_complete_coverage is False
    assert Ran(passed=True, n=300, n_total=300).is_complete_coverage is True
    # Coverage unstated is NOT coverage complete.
    assert Ran(passed=True).is_complete_coverage is False


def test_aggregate_is_never_more_confident_than_its_least_informed_input():
    assert aggregate([Ran(passed=True), Ran(passed=True)]).passed is True
    agg = aggregate({"a": Ran(passed=True), "b": NotRun(reason="no GPU")})
    assert isinstance(agg, NotRun)
    assert "no GPU" in agg.reason


def test_aggregate_over_zero_checks_is_not_a_pass():
    """all([]) is True in Python. An empty gate has not passed; it is empty."""
    assert isinstance(aggregate([]), NotRun)


@pytest.mark.parametrize("n_not_run", range(1, 6))
def test_any_not_run_input_blocks_the_aggregate(n_not_run: int):
    parts = [Ran(passed=True) for _ in range(5 - n_not_run)]
    parts += [NotRun(reason=f"r{i}") for i in range(n_not_run)]
    assert isinstance(aggregate(parts), NotRun)


# --------------------------------------------------------------------------
# S6: a run cannot finish without a row
# --------------------------------------------------------------------------

def test_completed_run_writes_a_row(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    with RunRecorder(led, protocol=_protocol(1), run_kind="ft", repo=REPO, env=_env()) as rec:
        rec.metric("accuracy.k4", Ran(passed=True, value=0.83, n=300, n_total=300))
    rows = led.rows()
    assert len(rows) == 1 and rows[0].status == "completed"


def test_failed_run_writes_a_row_saying_so(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    with pytest.raises(RuntimeError):
        with RunRecorder(led, protocol=_protocol(1), run_kind="cpt", repo=REPO, env=_env()):
            raise RuntimeError("deliberate explosion")
    rows = led.rows()
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "deliberate explosion" in rows[0].notes


def test_killed_run_writes_a_row_saying_so(tmp_path: Path):
    """S6's explicit gate: 'A deliberately broken run must produce a row that says so.'

    Runs in a subprocess so a real SIGTERM is delivered to a real process.
    """
    ledger_path = tmp_path / "runs.jsonl"
    script = f"""
import sys, os, time, signal
sys.path.insert(0, {str(REPO / "python")!r})
from qd_train.ledger import Ledger, Protocol, RunRecorder, Environment
from qd_train.tristate import NotRun
led = Ledger({str(ledger_path)!r})
p = Protocol('d'*64, 't'*64, 'b'*40, 'r'*64, 1)
env = Environment(torch='x', transformers_sha='y', device='cpu', host='t',
                  fla_present=NotRun(reason='n/a'), causal_conv1d_present=NotRun(reason='n/a'))
with RunRecorder(led, protocol=p, run_kind='cpt', repo={str(REPO)!r}, env=env):
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(5)
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0, "process should have died from SIGTERM"
    rows = Ledger(ledger_path).rows()
    assert len(rows) == 1, f"killed run left no row; stderr={proc.stderr}"
    assert rows[0].status == "killed"
    assert "SIGTERM" in rows[0].notes


def test_unreported_gates_become_not_run_not_absent(tmp_path: Path):
    """A gate the run forgot must refuse promotion, not silently drop out of it."""
    led = Ledger(tmp_path / "runs.jsonl")
    with RunRecorder(led, protocol=_protocol(1), run_kind="ft", repo=REPO, env=_env()) as rec:
        rec.gate("ece", Ran(passed=True, value=0.02))
    row = led.rows()[0]
    from qd_train.ledger import REQUIRED_GATES

    assert set(row.gates) == set(REQUIRED_GATES)
    assert sum(isinstance(g, NotRun) for g in row.gates.values()) == len(REQUIRED_GATES) - 1


# --------------------------------------------------------------------------
# Promotion
# --------------------------------------------------------------------------

def test_three_clean_seeds_promote(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()) as rec:
            _all_green(rec)
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert verdict.promoted, str(verdict)


def test_two_seeds_do_not_promote(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()) as rec:
            _all_green(rec)
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert not verdict.promoted
    assert any("differing only in seed" in r for r in verdict.reasons)


def test_a_quick_run_cannot_promote(tmp_path: Path):
    """Rule 8: fewer than 3 seeds, a truncated schedule or a subsample is `quick`."""
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env(),
            quick=(seed == 2), quick_reason="truncated schedule" if seed == 2 else None,
        ) as rec:
            _all_green(rec)
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert not verdict.promoted
    assert any("quick runs cannot promote" in r for r in verdict.reasons)


@pytest.mark.parametrize("missing_gate", ["paired_margin_vs_linear", "ood_abstain",
                                          "needle_hunk_recall", "permutation_consistency", "ece"])
def test_a_single_not_run_gate_blocks_promotion(tmp_path: Path, missing_gate: str):
    """Condition 4, the one that is easy to get wrong: not_run BLOCKS, it does not pass."""
    from qd_train.ledger import REQUIRED_CONTROLS, REQUIRED_GATES

    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()) as rec:
            for g in REQUIRED_GATES:
                if g != missing_gate:
                    rec.gate(g, Ran(passed=True, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert not verdict.promoted
    assert any(missing_gate in r and "did not run" in r for r in verdict.reasons)


def test_quick_without_reason_is_refused():
    with pytest.raises(ValueError, match="requires quick_reason"):
        LedgerRow(
            row_id="x", written_at="now", prev_row_hash=None, protocol=_protocol(1),
            run_kind="ft", status="completed", quick=True, quick_reason=None,
            code_commit="c", env=_env(), metrics={}, noul_rate=NotRun(reason="n/a"),
            controls={}, gates={}, wall_clock_s=1.0, cost_usd=0.0,
        )


# --------------------------------------------------------------------------
# Append-only and chain integrity
# --------------------------------------------------------------------------

def test_chain_verifies_on_an_honest_ledger(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env()):
            pass
    led.verify_chain()


def test_editing_a_row_breaks_the_chain(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env()) as rec:
            rec.gate("ece", Ran(passed=False, value=0.9))
    # Flip a failing gate to passing, exactly the tamper the chain exists to catch.
    lines = led.path.read_bytes().split(b"\n")
    obj = json.loads(lines[0])
    obj["gates"]["ece"]["passed"] = True
    lines[0] = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    led.path.write_bytes(b"\n".join(lines))
    with pytest.raises(LedgerChainError, match="History was edited"):
        led.verify_chain()


def test_deleting_a_row_breaks_the_chain(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env()):
            pass
    lines = [ln for ln in led.path.read_bytes().split(b"\n") if ln.strip()]
    led.path.write_bytes(b"\n".join([lines[0], lines[2]]) + b"\n")
    with pytest.raises(LedgerChainError):
        led.verify_chain()


def test_duplicate_row_id_is_refused(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    row = LedgerRow(
        row_id="fixed-id", written_at="now", prev_row_hash=None, protocol=_protocol(1),
        run_kind="eval", status="completed", quick=False, quick_reason=None,
        code_commit="c", env=_env(), metrics={}, noul_rate=NotRun(reason="n/a"),
        controls={}, gates={}, wall_clock_s=1.0, cost_usd=0.0,
    )
    led.append(row)
    with pytest.raises(ValueError, match="append-only"):
        led.append(row)


def test_protocol_hash_mismatch_is_detected(tmp_path: Path):
    """Rewriting a protocol component without recomputing its hash is caught."""
    led = Ledger(tmp_path / "runs.jsonl")
    with RunRecorder(led, protocol=_protocol(1), run_kind="eval", repo=REPO, env=_env()):
        pass
    obj = json.loads(led.path.read_bytes().strip())
    obj["protocol"]["seed"] = 99
    led.path.write_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(LedgerChainError, match="was edited"):
        led.verify_chain()


def test_seed_family_groups_rows_that_differ_only_in_seed():
    fam = {_protocol(s).hash_without_seed() for s in (1, 2, 3)}
    assert len(fam) == 1
    other = Protocol("different", "t" * 64, "b" * 40, "r" * 64, 1)
    assert other.hash_without_seed() not in fam


# ---------------------------------------------------------------------------
# run_kind "build": the row a lane that compiles and runs a test suite writes.
#
# Repo rule 5 says a number in a report cites a ledger row or is not in the
# report. Before this existed there was no run_kind whose required fields a
# build lane could honestly fill, so three lanes in a row printed command
# output instead and said so (GAP-MUTATE-NO-LEDGER-ROW-KIND). These tests pin
# the two things that make the new kind safe rather than merely available: it
# never promotes, and it cannot be confused with a training row in either
# direction.
# ---------------------------------------------------------------------------


def _build_protocol(*, commands: tuple[str, ...] = ("cargo test --workspace",)) -> Protocol:
    return Protocol.for_build(commands=commands, toolchain="cargo 1.98.0")


def test_a_build_lane_has_a_run_kind_to_write(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    with RunRecorder(
        led, protocol=_build_protocol(), run_kind="build", repo=REPO, env=_env()
    ) as rec:
        rec.metric("suite.cargo_test_workspace", Ran(passed=True, value=344, n=344, n_total=344))
    rows = led.rows()
    assert len(rows) == 1
    assert rows[0].run_kind == "build"
    assert rows[0].status == "completed"
    assert rows[0].metrics["suite.cargo_test_workspace"].value == 344


def test_a_build_protocol_says_which_components_it_does_not_have():
    proto = _build_protocol()
    # The three model-protocol components do not exist for a build. They carry an
    # explicit not-applicable marker rather than a plausible-looking hash: a
    # fabricated value in the decision record is worse than no row at all, which
    # is exactly why the earlier lanes declined to invent one.
    assert proto.data_snapshot_hash == NOT_APPLICABLE
    assert proto.tokenizer_hash == NOT_APPLICABLE
    assert proto.backbone_commit == NOT_APPLICABLE
    # `recipe_hash` is real: it identifies the command set, so two build runs of
    # different commands are not silently comparable.
    assert proto.recipe_hash != NOT_APPLICABLE
    assert len(proto.recipe_hash) == 64
    other = Protocol.for_build(commands=("pytest python/tests",), toolchain="cargo 1.98.0")
    assert other.recipe_hash != proto.recipe_hash
    same = _build_protocol()
    assert same.recipe_hash == proto.recipe_hash, "the same commands must hash the same"


def test_a_build_row_cannot_promote_anything(tmp_path: Path):
    """Three completed, non-quick build rows with every gate and control green.

    That is every condition in docs/ledger-schema.md's promotion list, and it
    must still refuse — because a build row's gates are vacuous, not satisfied.
    Leaving this to "a build lane would never report gates" is the failure mode:
    `_fill_unreported` would have to be relied on to keep a decision honest.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        proto = Protocol.for_build(
            commands=("cargo test --workspace",), toolchain="cargo 1.98.0", seed=seed
        )
        with RunRecorder(led, protocol=proto, run_kind="build", repo=REPO, env=_env()) as rec:
            _all_green(rec)
    verdict = led.promotion_verdict(_build_protocol().hash_without_seed())
    assert not verdict.promoted
    assert any("build" in r for r in verdict.reasons), verdict.reasons


def test_a_training_row_may_not_borrow_the_build_marker(tmp_path: Path):
    """The marker is one-directional: only a build row may carry it.

    Without this, `data_snapshot_hash="n/a"` on a `cpt` row would pass every
    check in the module and produce a training row whose protocol identifies
    nothing, which is the same unfalsifiable comparison the marker exists to
    make visible.
    """
    with pytest.raises(ValueError, match="build"):
        LedgerRow(
            row_id="x",
            written_at="2026-09-19T00:00:00+00:00",
            prev_row_hash=None,
            protocol=Protocol(NOT_APPLICABLE, "t" * 64, "b" * 40, "r" * 64, 1),
            run_kind="cpt",
            status="completed",
            quick=False,
            quick_reason=None,
            code_commit="abc",
            env=_env(),
            metrics={},
            noul_rate=NotRun(reason="not computed"),
            controls={},
            gates={},
            wall_clock_s=1.0,
            cost_usd=0.0,
        )


def test_a_build_row_must_carry_the_marker_it_claims(tmp_path: Path):
    """And the other direction: a build row with a real-looking snapshot hash.

    A build lane has no data snapshot. A row saying it does would make the row
    comparable to training rows it has nothing to do with.
    """
    with pytest.raises(ValueError, match="build"):
        LedgerRow(
            row_id="x",
            written_at="2026-09-19T00:00:00+00:00",
            prev_row_hash=None,
            protocol=_protocol(1),
            run_kind="build",
            status="completed",
            quick=False,
            quick_reason=None,
            code_commit="abc",
            env=_env(),
            metrics={},
            noul_rate=NotRun(reason="not computed"),
            controls={},
            gates={},
            wall_clock_s=1.0,
            cost_usd=0.0,
        )
