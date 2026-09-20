"""S6: the ledger, and the tri-state that keeps 'not run' from reading as 'passed'.

Every test here corresponds to a line in docs/ledger-schema.md or docs/hardening.md.
The plan's S6 is not green until a deliberately killed run produces a row that says so
(``test_killed_run_writes_a_row_saying_so``), and S7 is not green until the harness
fails two deliberately broken models (``test_eval_harness.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.ledger import (
    NOT_APPLICABLE,
    Environment,
    Ledger,
    LedgerChainError,
    LedgerRow,
    Protocol,
    RunRecorder,
)
from qd_train.tristate import NotRun, Ran, aggregate, parse_tristate

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


def _all_green(rec: RunRecorder, *, termination: str = "steps_exhausted") -> None:
    from qd_train.ledger import REQUIRED_CONTROLS, REQUIRED_GATES

    for g in REQUIRED_GATES:
        rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
    for c in REQUIRED_CONTROLS:
        rec.control(c, Ran(passed=True, n=300, n_total=300))
    # What `qd_train.trainer._train` writes on every exit path. A training row that does
    # not say how it ended cannot be shown to have finished its schedule, so promotion now
    # refuses it -- which means a fixture standing for a complete run has to say so.
    rec.metric(
        "train.termination",
        Ran(passed=termination != "wall_clock_cap", value=termination),
    )


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
    with (
        pytest.raises(RuntimeError),
        RunRecorder(led, protocol=_protocol(1), run_kind="cpt", repo=REPO, env=_env()),
    ):
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
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
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
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()
        ) as rec:
            _all_green(rec)
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert verdict.promoted, str(verdict)


def test_two_seeds_do_not_promote(tmp_path: Path):
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2):
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()
        ) as rec:
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


def test_a_capped_run_cannot_promote_even_when_it_calls_itself_complete(tmp_path: Path):
    """Rule 8's "truncated schedule", derived from evidence instead of taken on trust.

    Measured against the pre-fix code on 2026-09-20: this exact ledger -- three seeds,
    every required gate and control passing, one row honestly recording
    `train.termination == 'wall_clock_cap'`, all three `quick=False` -- returned
    `promoted=True` with the single reason line *"3 completed rows, seeds [1, 2, 3], every
    gate and control ran and passed"*. `quick` was the only thing standing between a capped
    run and a promotion, and `quick` is whatever the caller typed.
    """
    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()
        ) as rec:
            _all_green(
                rec, termination="wall_clock_cap" if seed == 2 else "steps_exhausted"
            )
    assert all(not r.quick for r in led.rows()), "every row calls itself complete"
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert not verdict.promoted, str(verdict)
    assert any("truncated schedule" in r for r in verdict.reasons), str(verdict)


def test_a_training_row_that_never_says_how_it_ended_cannot_promote(tmp_path: Path):
    """An absent answer is not a passed one -- the same rule the gates follow.

    `_train` writes `train.termination` on every exit path, so a `cpt`/`ft`/`prune_heal`
    row without it did not come from the trainer, and nothing about it establishes that a
    schedule was finished.
    """
    from qd_train.ledger import REQUIRED_CONTROLS, REQUIRED_GATES

    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()
        ) as rec:
            for g in REQUIRED_GATES:
                rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
    verdict = led.promotion_verdict(_protocol(1).hash_without_seed())
    assert not verdict.promoted, str(verdict)
    assert any("no train.termination" in r for r in verdict.reasons), str(verdict)

    # And a run kind with no training loop is not asked a question it cannot answer.
    other = Ledger(tmp_path / "eval.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(
            other, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env()
        ) as rec:
            for g in REQUIRED_GATES:
                rec.gate(g, Ran(passed=True, value=1.0, n=300, n_total=300))
            for c in REQUIRED_CONTROLS:
                rec.control(c, Ran(passed=True, n=300, n_total=300))
    assert other.promotion_verdict(_protocol(1).hash_without_seed()).promoted


@pytest.mark.parametrize("missing_gate", ["paired_margin_vs_linear", "ood_abstain",
                                          "needle_hunk_recall", "permutation_consistency", "ece"])
def test_a_single_not_run_gate_blocks_promotion(tmp_path: Path, missing_gate: str):
    """Condition 4, the one that is easy to get wrong: not_run BLOCKS, it does not pass."""
    from qd_train.ledger import REQUIRED_CONTROLS, REQUIRED_GATES

    led = Ledger(tmp_path / "runs.jsonl")
    for seed in (1, 2, 3):
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="ft", repo=REPO, env=_env()
        ) as rec:
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
        with RunRecorder(
            led, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env()
        ) as rec:
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


# ---------------------------------------------------------------------------
# The recording path: how a lane actually gets a row id to cite.
#
# `run_kind: "build"` made an honest row *possible*; it did not make one easy.
# Between the kind and a written row sat: construct a Protocol, construct an
# Environment, open a Ledger, wrap the commands in a RunRecorder, run them,
# parse two different harnesses' output into coverage pairs, and decide what a
# suite that never launched should look like. Measured: with the kind available
# and documented, `ledger/` stayed empty and four lanes printed command output
# instead. A rule harder to obey than to skip is a rule that gets skipped, so
# the path below is one command.
# ---------------------------------------------------------------------------


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "qd_train.ledger", *args],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(cwd or REPO),
        env=env,
    )


# -- the output parsers -----------------------------------------------------


CARGO_SAMPLE = """\
running 107 tests
test result: ok. 107 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.73s

running 24 tests
test result: FAILED. 24 passed; 1 failed; 2 ignored; 0 measured; 3 filtered out; finished in 7.09s

   Doc-tests qd_mutate
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""

PYTEST_SAMPLE = """\
........................................................................ [ 19%]
..........................                                               [100%]
1104 passed, 7 skipped in 29.02s
"""


def test_cargo_counts_are_summed_over_every_test_binary():
    """One `cargo test --workspace` prints one summary per binary, not one total.

    A parser that reads only the last line reports the doc-tests' `0 passed` as
    the workspace result. That is wrong in the direction that looks harmless,
    which is the direction nobody checks.
    """
    from qd_train.ledger import parse_cargo_test_output

    counts = parse_cargo_test_output(CARGO_SAMPLE)
    assert counts is not None
    assert counts.ran_ok == 131  # 107 + 24 + 0
    assert counts.ran_failed == 1
    # `ignored` and `filtered out` were collected and did not run, so they belong
    # in n_total and nowhere else: a filter that silently shrank the run is then
    # visible as n < n_total rather than as a smaller, cleaner-looking pass count.
    assert counts.not_run == 5  # 2 ignored + 3 filtered out
    assert counts.n == 132
    assert counts.n_total == 137


def test_pytest_skips_are_visible_in_the_coverage_pair():
    from qd_train.ledger import parse_pytest_output

    counts = parse_pytest_output(PYTEST_SAMPLE)
    assert counts is not None
    assert counts.ran_ok == 1104
    assert counts.ran_failed == 0
    assert counts.not_run == 7
    assert counts.n == 1104
    assert counts.n_total == 1111
    assert counts.as_tristate(exit_code=0).is_complete_coverage is False


def test_a_parser_that_recognises_nothing_says_so_rather_than_counting_zero():
    """`None`, not zero-of-everything.

    Zero passed, zero failed, zero skipped is a real and possible measurement --
    an empty suite. A parser returning it because it did not understand the
    output has manufactured that measurement, and the caller cannot tell the two
    apart.
    """
    from qd_train.ledger import parse_cargo_test_output, parse_pytest_output

    assert parse_cargo_test_output("error: could not compile `qd_mutate`") is None
    assert parse_pytest_output("ERROR: file or directory not found: nope/") is None


# -- what a suite result may and may not become -----------------------------


def test_a_command_that_cannot_be_launched_is_not_run_never_a_zero(tmp_path: Path):
    """The failure the tri-state exists for, at the point a build lane meets it."""
    from qd_train.ledger import run_suite

    out = run_suite("missing_tool", ("qd-no-such-binary-exists", "--version"), cwd=tmp_path)
    assert isinstance(out.result, NotRun)
    assert out.exit_code is None
    assert "could not be launched" in out.result.reason


def test_a_timeout_is_not_run_rather_than_a_failure(tmp_path: Path):
    """A suite that was cut off produced no result; it did not produce a bad one."""
    from qd_train.ledger import run_suite

    out = run_suite(
        "sleeper", (sys.executable, "-c", "import time; time.sleep(30)"),
        cwd=tmp_path, timeout_s=1.0,
    )
    assert isinstance(out.result, NotRun)
    assert "timed out" in out.result.reason


def test_a_green_command_with_no_parsable_summary_is_not_recorded_as_a_pass(tmp_path: Path):
    """Exit 0 with no counts is the shape of a suite that collected nothing.

    pytest exits 0 on "no tests ran" under some configurations -- which is why
    this repo's addopts carry `--strict-config`. Recording that as
    `Ran(passed=True)` with coverage unstated would put a green suite with no
    tests into the decision record, so it is `not_run` with the reason instead.
    """
    from qd_train.ledger import run_suite

    out = run_suite(
        "silent_success", (sys.executable, "-c", "print('nothing to report')"),
        cwd=tmp_path, parser="pytest",
    )
    assert isinstance(out.result, NotRun), out.result
    assert out.exit_code == 0
    assert "no parsable" in out.result.reason


def test_a_failing_command_with_no_summary_is_recorded_as_a_failure(tmp_path: Path):
    """The other direction. A compile error is a command that ran and failed.

    Calling that `not_run` would let a broken build sit in the ledger as "nothing
    was measured here", which reads far better than it deserves.
    """
    from qd_train.ledger import run_suite

    out = run_suite(
        "broken_build",
        (sys.executable, "-c", "import sys; sys.stderr.write('E0001\\n'); sys.exit(101)"),
        cwd=tmp_path, parser="cargo",
    )
    assert isinstance(out.result, Ran)
    assert out.result.passed is False
    assert out.exit_code == 101
    # Coverage is unstated, not complete: nothing was counted.
    assert out.result.is_complete_coverage is False
    assert out.result.n is None


def test_a_shell_pipeline_is_refused_rather_than_run_as_arguments():
    """`cargo test | tail` loses the exit code -- this repo's harness says so in
    as many words. Splitting it into argv would hand `|` to cargo as a test-name
    filter, and cargo would exit 0 having run nothing."""
    from qd_train.ledger import parse_command

    with pytest.raises(ValueError, match=r"pipeline|redirect"):
        parse_command("cargo test --workspace | tail -5")
    with pytest.raises(ValueError, match=r"pipeline|redirect"):
        parse_command("cargo test --workspace > out.txt")
    assert parse_command("cargo test --workspace") == ("cargo", "test", "--workspace")


# -- the one command a lane runs --------------------------------------------


def test_a_lane_records_a_build_run_and_gets_a_row_id(tmp_path: Path):
    from qd_train.ledger import record_build_run

    led = Ledger(tmp_path / "runs.jsonl")
    row = record_build_run(
        ledger=led,
        repo=REPO,
        suites=(("unit", (sys.executable, "-c", "print('3 passed, 1 skipped in 0.10s')")),),
        toolchain="python 3.14.7",
        cwd=tmp_path,
        env=_env(),
    )
    assert row.run_kind == "build"
    assert row.status == "completed"
    suite = row.metrics["suite.unit"]
    assert isinstance(suite, Ran) and suite.passed is True
    assert (suite.value, suite.n, suite.n_total) == (3, 3, 4)
    # And it is on disk, readable by row id, with the chain intact.
    led.verify_chain()
    assert [r.row_id for r in led.rows()] == [row.row_id]


def test_a_failing_suite_makes_the_row_say_failed(tmp_path: Path):
    """`status` is not cosmetic: promotion condition 1 reads it."""
    from qd_train.ledger import record_build_run

    led = Ledger(tmp_path / "runs.jsonl")
    row = record_build_run(
        ledger=led,
        repo=REPO,
        suites=(
            (
                "unit",
                (sys.executable, "-c", "print('2 failed, 1 passed in 0.1s'); raise SystemExit(1)"),
            ),
        ),
        toolchain="python 3.14.7",
        cwd=tmp_path,
        env=_env(),
    )
    assert row.status == "failed"
    suite = row.metrics["suite.unit"]
    assert isinstance(suite, Ran) and suite.passed is False
    assert (suite.n, suite.n_total) == (3, 3)


def test_the_cli_prints_the_row_id_a_lane_must_cite(tmp_path: Path):
    """The point of the path: one command, one id, pasteable into a handoff."""
    ledger_path = tmp_path / "runs.jsonl"
    proc = _run_cli(
        "record",
        "--ledger", str(ledger_path),
        "--toolchain", "python 3.14.7",
        "--suite", f"unit={sys.executable} -c \"print('5 passed in 0.1s')\"",
    )
    assert proc.returncode == 0, proc.stderr
    row_id = proc.stdout.strip().splitlines()[-1].strip()
    rows = Ledger(ledger_path).rows()
    assert len(rows) == 1
    assert rows[0].row_id == row_id, f"CLI printed {row_id!r}, ledger holds {rows[0].row_id!r}"


def test_the_cli_exits_non_zero_when_a_suite_fails(tmp_path: Path):
    """A recorder that always exits 0 turns a red suite into a green lane."""
    ledger_path = tmp_path / "runs.jsonl"
    proc = _run_cli(
        "record",
        "--ledger", str(ledger_path),
        "--toolchain", "python 3.14.7",
        "--suite", f"unit={sys.executable} -c \"raise SystemExit(1)\"",
    )
    assert proc.returncode != 0
    # The row is still written. A failed run that leaves no row is the one
    # failure this module exists to prevent.
    assert len(Ledger(ledger_path).rows()) == 1


def test_the_cli_verifies_the_chain_and_reports_a_break(tmp_path: Path):
    """docs/ledger-schema.md has claimed a verify command since S6."""
    ledger_path = tmp_path / "runs.jsonl"
    led = Ledger(ledger_path)
    for seed in (1, 2, 3):
        with RunRecorder(led, protocol=_protocol(seed), run_kind="eval", repo=REPO, env=_env()):
            pass
    ok = _run_cli("verify", "--ledger", str(ledger_path))
    assert ok.returncode == 0, ok.stderr

    lines = [ln for ln in ledger_path.read_bytes().split(b"\n") if ln.strip()]
    ledger_path.write_bytes(b"\n".join([lines[0], lines[2]]) + b"\n")
    broken = _run_cli("verify", "--ledger", str(ledger_path))
    assert broken.returncode != 0
    assert "prev_row_hash" in (broken.stdout + broken.stderr)


def test_the_recorded_build_row_still_cannot_promote(tmp_path: Path):
    """End to end, through the real path rather than a hand-built row."""
    from qd_train.ledger import record_build_run

    led = Ledger(tmp_path / "runs.jsonl")
    row = record_build_run(
        ledger=led,
        repo=REPO,
        suites=(("unit", (sys.executable, "-c", "print('1 passed in 0.1s')")),),
        toolchain="python 3.14.7",
        cwd=tmp_path,
        env=_env(),
    )
    verdict = led.promotion_verdict(row.protocol.hash_without_seed())
    assert not verdict.promoted
    assert any("never promotes" in r for r in verdict.reasons), verdict.reasons


# -- the append-only chain, under the conditions it is actually written in ---


def test_concurrent_appends_do_not_break_the_chain(tmp_path: Path):
    """Several sessions write this repo at once; the ledger has to survive that.

    `append` reads `last_line_hash()` and then opens the file. Between those two
    steps another process can append, and both writers then claim the same
    predecessor -- a chain `verify_chain` refuses forever after, on a log whose
    whole premise is that you cannot go back and fix it. O_APPEND prevents a torn
    line; it does nothing for the read-then-write window.
    """
    ledger_path = tmp_path / "runs.jsonl"
    start_at = time.time() + 2.0
    script = f"""
import sys, time
sys.path.insert(0, {str(REPO / "python")!r})
from qd_train.ledger import Ledger, LedgerRow, Protocol, Environment
from qd_train.tristate import NotRun
led = Ledger({str(ledger_path)!r})
env = Environment(torch='x', transformers_sha='y', device='cpu', host='t',
                  fla_present=NotRun(reason='n/a'), causal_conv1d_present=NotRun(reason='n/a'))
while time.time() < {start_at!r}:
    pass
for i in range(5):
    led.append(LedgerRow(
        row_id=f"{{sys.argv[1]}}-{{i}}", written_at='2026-09-19T00:00:00+00:00',
        prev_row_hash=None, protocol=Protocol('d'*64, 't'*64, 'b'*40, 'r'*64, 1),
        run_kind='eval', status='completed', quick=False, quick_reason=None,
        code_commit='c', env=env, metrics={{}}, noul_rate=NotRun(reason='n/a'),
        controls={{}}, gates={{}}, wall_clock_s=0.0, cost_usd=0.0,
    ))
"""
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, f"w{w}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for w in range(6)
    ]
    failures = []
    for p in procs:
        _, err = p.communicate(timeout=180)
        if p.returncode != 0:
            failures.append(err.strip().splitlines()[-1] if err.strip() else "no stderr")

    led = Ledger(ledger_path)
    assert not failures, "a writer crashed instead of waiting its turn:\n  " + "\n  ".join(failures)
    assert len(led.raw_lines()) == 30, f"lost rows: {len(led.raw_lines())} of 30"
    led.verify_chain()  # the assertion: 30 concurrent appends, one unbroken chain


def test_green_counts_with_a_red_exit_code_are_not_a_pass(tmp_path: Path):
    """The counts are not the only witness, and they are not the last word.

    A harness can report zero failures and still exit non-zero: a collection
    error, a plugin that raised in teardown, a link step that failed after the
    tests themselves were fine. Reading only the summary line records that as
    green. Both signals have to agree.
    """
    from qd_train.ledger import run_suite

    out = run_suite(
        "green_counts_red_exit",
        (sys.executable, "-c", "print('5 passed in 0.10s'); raise SystemExit(1)"),
        cwd=tmp_path,
    )
    assert isinstance(out.result, Ran)
    assert out.result.value == 5, "the counts are still recorded"
    assert out.result.passed is False, "exit 1 with a clean summary is not a pass"


def test_two_suites_cannot_share_a_name(tmp_path: Path):
    """One metric key per suite, or the second result silently replaces the first
    and the row reports half of what was run as though it were all of it."""
    from qd_train.ledger import record_build_run

    with pytest.raises(ValueError, match="duplicate suite name"):
        record_build_run(
            ledger=Ledger(tmp_path / "runs.jsonl"),
            repo=REPO,
            suites=(("unit", ("true",)), ("unit", ("false",))),
            toolchain="python 3.14.7",
        )


def test_a_suite_that_could_not_run_does_not_exit_zero(tmp_path: Path):
    """`$?` is a trace like any other, and it gets the tri-state's rule.

    Exit 0 for "the GPU suite was not run here" is how a lane's CI, or a lane's
    own eyes, read an unexamined suite as an examined one. It exits 3: not 0,
    and not the same as a suite that ran and failed.
    """
    from qd_train.ledger import EXIT_SUITE_NOT_RUN

    ledger_path = tmp_path / "runs.jsonl"
    proc = _run_cli(
        "record",
        "--ledger", str(ledger_path),
        "--toolchain", "python 3.14.7",
        "--suite", "gpu=qd-no-such-binary-exists --version",
    )
    assert proc.returncode == EXIT_SUITE_NOT_RUN, (proc.returncode, proc.stderr)
    rows = Ledger(ledger_path).rows()
    assert len(rows) == 1
    assert isinstance(rows[0].metrics["suite.gpu"], NotRun)
    assert rows[0].status == "completed", "not-run is not a failure; the run itself completed"


def test_the_cli_reports_the_promotion_verdict_for_a_row(tmp_path: Path):
    """The refusal a lane can read without writing Python."""
    from qd_train.ledger import record_build_run

    ledger_path = tmp_path / "runs.jsonl"
    led = Ledger(ledger_path)
    row = record_build_run(
        ledger=led,
        repo=REPO,
        suites=(("unit", (sys.executable, "-c", "print('1 passed in 0.1s')")),),
        toolchain="python 3.14.7",
        cwd=tmp_path,
        env=_env(),
    )
    proc = _run_cli("verdict", "--ledger", str(ledger_path), "--row-id", row.row_id)
    assert proc.returncode == 1, proc.stderr
    assert "REFUSED" in proc.stdout
    assert "never promotes" in proc.stdout


# ---------------------------------------------------------------------------
# Found by the first real row, not by reasoning about it.
#
# Row 06d15c1b-e643-4fae-a478-32b7581ef632 recorded
# `suite.cargo_test_workspace: FAILED value=299 coverage=301/301` — a complete
# coverage pair. The baseline an hour earlier was 347 passed. cargo had not lost
# 46 tests: it had *aborted* after the first failing test binary and never built
# or ran the rest. The recorder asked cargo what it did, cargo answered for the
# binaries it reached, and the pair came out n == n_total.
#
# That is the exact shape this repo forbids — a run that was cut short reading
# as one with full coverage — reproduced inside the tool written to prevent it.
# The counts a harness reports are only ever about the part it reached, so the
# tri-state's own rule applies to the denominator: unstated, not complete.
# ---------------------------------------------------------------------------


CARGO_ABORTED_SAMPLE = """\
running 107 tests
test result: ok. 107 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.73s

running 16 tests
test result: FAILED. 14 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.11s

failures:
    both_lanes_refuse_the_same_payloads_and_name_the_context_checks_identically

error: test failed, to rerun pass `-p qd-runtime --test wire_context_crosslang`
"""

PYTEST_ABORTED_SAMPLE = """\
python/tests/test_wire_gap_pins.py .F
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed, 1 passed in 0.31s
"""


def test_a_cargo_run_that_aborted_early_does_not_claim_complete_coverage():
    """cargo's default is fail-fast: the binaries after the failing one never run.

    Their tests are not `ignored` and not `filtered out` — cargo never mentions
    them at all, so summing what it printed produces n == n_total for a run that
    covered a fraction of the workspace.
    """
    from qd_train.ledger import parse_cargo_test_output

    counts = parse_cargo_test_output(CARGO_ABORTED_SAMPLE)
    assert counts is not None
    assert counts.ran_ok == 121 and counts.ran_failed == 2
    assert counts.collection_complete is False, "cargo said it stopped; believe it"

    result = counts.as_tristate(exit_code=101)
    assert result.passed is False
    assert result.value == 121, "the counts it did report are still recorded"
    assert (result.n, result.n_total) == (None, None), "a denominator nobody measured"
    assert result.is_complete_coverage is False
    assert result.coverage_str() == "coverage unstated"
    assert "aborted" in result.detail


def test_a_pytest_run_stopped_by_exitfirst_does_not_claim_complete_coverage():
    """Same hole, other harness: `-x` leaves the rest of the suite unmentioned."""
    from qd_train.ledger import parse_pytest_output

    counts = parse_pytest_output(PYTEST_ABORTED_SAMPLE)
    assert counts is not None
    assert counts.collection_complete is False
    result = counts.as_tristate(exit_code=1)
    assert result.is_complete_coverage is False
    assert (result.n, result.n_total) == (None, None)


def test_a_complete_run_still_carries_its_coverage_pair():
    """The fix must not answer "unstated" to everything, which would be the same
    failure pointed the other way: a coverage pair that is never populated tells
    a reader nothing and cannot show a shrinking collection."""
    from qd_train.ledger import parse_cargo_test_output, parse_pytest_output

    cargo = parse_cargo_test_output(CARGO_SAMPLE)
    assert cargo is not None and cargo.collection_complete is True
    assert cargo.as_tristate(exit_code=0).coverage_str() == "132/137"

    pyt = parse_pytest_output(PYTEST_SAMPLE)
    assert pyt is not None and pyt.collection_complete is True
    assert pyt.as_tristate(exit_code=0).coverage_str() == "1104/1111"


# ---------------------------------------------------------------------------
# The device a row claims is the device the run used
# ---------------------------------------------------------------------------


def test_detect_reports_the_device_the_run_chose_not_the_host_ceiling():
    """A run placed on a device must not be recorded as the host's best device.

    Measured before `detect` took a device: `tools/rung0_toy_run.py` ran the same toy
    schedule three times on `mps` and three times on `cpu`, and all six rows said
    `device: "mps"` -- because `detect` answered `torch.backends.mps.is_available()`,
    which is a fact about the host and the same for every run on it. A ledger whose rows
    all name one device cannot be read back for a comparison between two.
    """
    from qd_train.ledger import Environment as Env

    for chosen in ("cpu", "mps", "cuda:8xH100-80GB"):
        assert Env.detect(device=chosen).device == chosen


def test_detect_still_auto_detects_when_no_device_is_chosen():
    """The control. A `device` parameter that silently became mandatory, or that turned
    auto-detection off, would break `record_build_run` -- whose row should carry what the
    host is, because it never chose anything."""
    import importlib.util

    auto = Environment.detect().device
    if importlib.util.find_spec("torch") is None:
        assert auto == "cpu"
    else:
        import torch as _torch

        expected = "mps" if _torch.backends.mps.is_available() else "cpu"
        assert auto == expected or auto.startswith("cuda:")


def test_a_recorded_row_carries_the_device_it_was_given():
    """End to end: the parameter has to survive into the written row, not just the object."""
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        ledger = Ledger(Path(raw) / "runs.jsonl")
        with RunRecorder(
            ledger,
            protocol=_protocol(seed=0),
            run_kind="smoke",
            repo=Path(raw),
            env=Environment.detect(device="cpu"),
            quick=True,
            quick_reason="device-provenance test",
        ):
            pass
        assert ledger.rows()[0].env.device == "cpu"
