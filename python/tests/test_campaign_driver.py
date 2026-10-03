"""The GH200 campaign driver: caps, resume, the go/no-go stop, the refusals before launch,
and the Mac-pull handoff that gates every termination.

Every unit here is a small fake tool that writes real ledger rows through ``RunRecorder``
into a scratch ledger, so the driver is exercised against the same row format the real
tools write -- never against ``ledger/runs.jsonl``. The "Mac" is the real
``tools/sync_box.sh pull`` in ``QD_BOX=local`` mode, run in a background thread.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import campaign_driver as cd  # noqa: E402

FAKE_TOOL = '''
import argparse, os, sys, time
from pathlib import Path
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO / "python"))
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder
from qd_train.tristate import NotRun, Ran
p = argparse.ArgumentParser()
p.add_argument("repo"); p.add_argument("--ledger", required=True)
p.add_argument("--kind", default="ft"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--tool", default="tools/real_ft_run.py")
p.add_argument("--gate", choices=["pass", "fail", "not_run", "none"], default="none")
p.add_argument("--steps", type=int, default=0); p.add_argument("--wall", type=float, default=1.0)
p.add_argument("--sleep", type=float, default=0.0); p.add_argument("--exit", type=int, default=0)
p.add_argument("--no-row", action="store_true"); p.add_argument("--pidfile")
p.add_argument("--count"); p.add_argument("--raise", dest="boom", action="store_true")
p.add_argument("--quick", action="store_true")
a, rest = p.parse_known_args(sys.argv[1:])
if a.count:
    with open(a.count, "a") as fh: fh.write(" ".join(rest) + "\\n")
if a.pidfile:
    Path(a.pidfile).write_text(str(os.getpid()))
time.sleep(a.sleep)
if not a.no_row:
    proto = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                     backbone_commit="fake", recipe_hash="r" * 64, seed=a.seed)
    with RunRecorder(Ledger(a.ledger), protocol=proto, run_kind=a.kind, repo=REPO,
                     env=Environment.detect(device="cpu"), wall_clock_s=a.wall, cost=None,
                     quick=a.quick, quick_reason="driver test" if a.quick else None,
                     recipe={"tool": a.tool}) as rec:
        if a.boom:
            raise RuntimeError("simulated crash inside the recorder block")
        if a.steps:
            rec.metric("train.optimizer_steps", Ran(passed=True, value=a.steps))
            rec.metric("train.termination", Ran(passed=True, value="steps_exhausted"))
        if a.gate == "pass":
            rec.gate("paired_margin_vs_linear", Ran(passed=True, value=0.1, n=10, n_total=10))
        elif a.gate == "fail":
            rec.gate("paired_margin_vs_linear", Ran(passed=False, value=-0.2, n=10, n_total=10))
        elif a.gate == "not_run":
            rec.gate("paired_margin_vs_linear", NotRun(reason="no verdicts"))
sys.exit(a.exit)
'''


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    tool = tmp_path / "fake_tool.py"
    tool.write_text(FAKE_TOOL, encoding="utf-8")
    return {"tool": tool, "ledger": tmp_path / "ledger.jsonl", "state": tmp_path / "state",
            "term": tmp_path / "terminated", "mac": tmp_path / "mac",
            "count": tmp_path / "count.txt", "root": tmp_path}


def pull_once(state_dir: Path, dest: Path) -> subprocess.CompletedProcess[str]:
    """The real Mac-side pull, in local mode."""
    return subprocess.run(
        ["bash", str(REPO / "tools" / "sync_box.sh"), "pull"], capture_output=True, text=True,
        timeout=120, check=False,
        env={**os.environ, "QD_BOX": "local", "QD_PULL_DEST": str(dest),
             "QD_PULL_SYNC_DIR": str(state_dir / "sync"), "QD_PYTHON": sys.executable},
    )


@pytest.fixture
def puller() -> Iterator[Callable[[Path, Path], None]]:
    """Start a background Mac puller for a campaign's state dir; stopped at teardown."""
    stops: list[threading.Event] = []
    threads: list[threading.Thread] = []

    def start(state_dir: Path, dest: Path) -> None:
        stop = threading.Event()

        def loop() -> None:
            while not stop.is_set():
                pull_once(state_dir, dest)
                stop.wait(0.2)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        stops.append(stop)
        threads.append(t)

    yield start
    for stop in stops:
        stop.set()
    for t in threads:
        t.join(timeout=30)


def unit(env: dict[str, Path], name: str, *args: str, **extra: object) -> dict[str, object]:
    argv = [sys.executable, "-P", str(env["tool"]), str(REPO), "--ledger", str(env["ledger"]),
            "--count", str(env["count"]), name, *args]
    return {"name": name, "argv": argv, **extra}


def config(env: dict[str, Path], phases: list[dict[str, object]], **over: object) -> Path:
    cfg: dict[str, object] = {
        "name": "test", "state_dir": str(env["state"]), "ledger": str(env["ledger"]),
        "device": "cpu", "instance": "local-cpu", "usd_per_hour": 0.0, "n_gpus": 0,
        "campaign_cap_hours": 1.0,
        "terminate_command": ["/usr/bin/touch", str(env["term"])],
        "pull_grace_minutes": 0.05, "pull_poll_s": 0.1,
        "kill_grace_s": 2, "phases": phases,
    }
    cfg.update(over)
    path = env["root"] / f"campaign-{time.monotonic_ns()}.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def gate_phase(env: dict[str, Path], results: list[str]) -> dict[str, object]:
    return {
        "name": "3-go-no-go", "cap_hours": 0.5,
        "units": [unit(env, f"control-s{i}", "--kind", "eval", "--seed", str(i), "--tool",
                       "tools/ft_linear_control.py", "--gate", g.removesuffix("-quick"),
                       *(["--quick"] if g.endswith("-quick") else []))
                  for i, g in enumerate(results)],
        "gate": {"ledger": str(env["ledger"]), "name": "paired_margin_vs_linear",
                 "tool": "tools/ft_linear_control.py", "min_seeds": 3},
    }


def after_phase(env: dict[str, Path]) -> dict[str, object]:
    return {"name": "4-workhorse", "cap_hours": 0.5, "units": [unit(env, "workhorse")]}


def invocations(env: dict[str, Path]) -> list[str]:
    if not env["count"].exists():
        return []
    return [ln.split()[0] for ln in env["count"].read_text().splitlines() if ln.strip()]


# --- refusals before launch -----------------------------------------------------------


def test_a_missing_terminate_command_fails_closed(env: dict[str, Path]) -> None:
    path = config(env, [after_phase(env)])
    raw = json.loads(path.read_text())
    del raw["terminate_command"]
    path.write_text(json.dumps(raw))
    with pytest.raises(cd.ConfigRefused, match="no terminate_command"):
        cd.load_config(path)


def test_rule_four_needs_a_name_above_twenty_dollars(env: dict[str, Path]) -> None:
    gh200 = {"usd_per_hour": 1.49, "device": "cuda", "instance": "lambda-1xGH200", "n_gpus": 1,
             "usd_per_hour_source": "lambda.ai/pricing read at launch (test)",
             "pull_grace_minutes": 45}
    path = config(env, [after_phase(env)], campaign_cap_hours=60, **gh200)
    with pytest.raises(cd.ConfigRefused, match="approved_by is empty"):
        cd.load_config(path)
    ok = config(env, [after_phase(env)], campaign_cap_hours=60,
                approved_by="user decision 2026-09-28", **gh200)
    cfg = cd.load_config(ok)
    # 60 h cap + the default 45 min pull grace window: the box bills while it waits.
    assert cd.campaign_usd(cfg) == pytest.approx(1.49 * 60.75)
    assert "$90.52" in "\n".join(cd.cost_lines(cfg))


def test_a_rented_gpu_is_never_priced_at_zero(env: dict[str, Path]) -> None:
    with pytest.raises(cd.ConfigRefused, match="not free"):
        cd.load_config(config(env, [after_phase(env)], device="cuda", n_gpus=1))


def test_the_campaign_cap_is_bounded_by_the_approved_three_days(env: dict[str, Path]) -> None:
    with pytest.raises(cd.ConfigRefused, match="campaign_cap_hours"):
        cd.load_config(config(env, [after_phase(env)], campaign_cap_hours=80))


def test_a_long_phase_that_does_not_checkpoint_is_refused(env: dict[str, Path]) -> None:
    long = {"name": "4-workhorse", "cap_hours": 30, "units": [unit(env, "workhorse")]}
    with pytest.raises(cd.ConfigRefused, match="does not checkpoint"):
        cd.load_config(config(env, [long], campaign_cap_hours=40))


def test_a_phase_cap_above_the_program_cap_is_refused(env: dict[str, Path]) -> None:
    huge = {"name": "x", "cap_hours": 41, "units": [unit(env, "u")]}
    with pytest.raises(cd.ConfigRefused, match="MAX_CAP_S"):
        cd.load_config(config(env, [huge], campaign_cap_hours=60))


# --- the go/no-go ---------------------------------------------------------------------


def test_go_runs_every_phase_then_syncs_and_terminates(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    path = config(env, [gate_phase(env, ["pass", "pass", "pass"]), after_phase(env)])
    assert cd.main(["--config", str(path)]) == cd.EXIT_DONE
    assert "workhorse" in invocations(env)
    assert env["term"].exists()
    # A snapshot per phase plus the final one, each pulled, verified and acknowledged.
    sync = env["state"] / "sync"
    assert sorted(x.name for x in sync.glob("pulled-*.ok")) == [
        "pulled-1.ok", "pulled-2.ok", "pulled-3.ok"]
    final = json.loads((sync / "phase-3.done").read_text())
    assert final["final"] is True
    assert (env["mac"] / "phase-3" / "ledger" / "ledger.jsonl").read_bytes() == (
        env["ledger"].read_bytes())
    state = json.loads((env["state"] / cd.STATE_NAME).read_text())
    assert state["outcome"] == "completed every phase"
    assert state["guard_pid"] is None


@pytest.mark.parametrize("results", [["pass", "fail", "pass"], ["pass", "not_run", "pass"],
                                     ["pass", "pass"], ["pass", "pass-quick", "pass"]])
def test_no_go_stops_before_the_workhorse_phase(
    env: dict[str, Path], results: list[str], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    path = config(env, [gate_phase(env, results), after_phase(env)])
    assert cd.main(["--config", str(path)]) == cd.EXIT_STOPPED
    assert "workhorse" not in invocations(env), "phase 4 ran after a NO-GO"
    assert env["term"].exists(), "a stopped campaign must still end the billing"
    state = json.loads((env["state"] / cd.STATE_NAME).read_text())
    assert state["phases"]["3-go-no-go"]["status"] == "stopped_at_gate"
    assert "4-workhorse" not in state["phases"]


def test_a_passing_gate_on_a_quick_row_is_not_evidence_for_go(env: dict[str, Path]) -> None:
    """Rule 8: a quick row is excluded from decisions, and GO is one. Three passing seeds,
    one of them quick, is two seeds of evidence -- and the refusal names the quick row."""
    ledger = env["ledger"]
    for seed, quick in ((0, False), (1, True), (2, False)):
        subprocess.run(
            [sys.executable, "-P", str(env["tool"]), str(REPO), "--ledger", str(ledger),
             "--kind", "eval", "--seed", str(seed), "--tool", "tools/ft_linear_control.py",
             "--gate", "pass", *(["--quick"] if quick else [])],
            check=True, timeout=60,
        )
    gate = cd.GateSpec(ledger, "paired_margin_vs_linear", "tools/ft_linear_control.py", 2)
    go, why = cd.gate_verdict(gate, 0)
    assert not go
    assert "seed 1: quick (driver test)" in why


def test_no_verdict_row_at_all_is_a_no_go(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    phase = gate_phase(env, ["none"])
    phase["units"] = [unit(env, "control", "--kind", "eval", "--tool", "tools/other.py")]
    path = config(env, [phase, after_phase(env)])
    assert cd.main(["--config", str(path)]) == cd.EXIT_STOPPED
    state = json.loads((env["state"] / cd.STATE_NAME).read_text())
    assert "never scored" in state["phases"]["3-go-no-go"]["gate"]["why"]


# --- caps and failures ----------------------------------------------------------------


def test_a_unit_past_its_phase_cap_is_killed_with_its_process_group(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    pidfile = env["root"] / "pid"
    slow = {"name": "slow", "cap_hours": 3 / 3600,
            "units": [unit(env, "sleeper", "--sleep", "600", "--pidfile", str(pidfile))]}
    path = config(env, [slow, after_phase(env)])
    started = time.monotonic()
    assert cd.main(["--config", str(path)]) == cd.EXIT_FAILED
    assert time.monotonic() - started < 60
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "workhorse" not in invocations(env)
    assert env["term"].exists()


def test_a_crash_is_not_accepted_but_a_failed_claim_can_be(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    puller(env["root"] / "state2", env["root"] / "mac2")
    claim = {"name": "0-smoke", "cap_hours": 0.5, "units": [
        unit(env, "claimfail", "--exit", "1", accept_returncodes=[0, 1])]}
    path = config(env, [claim, after_phase(env)])
    assert cd.main(["--config", str(path)]) == cd.EXIT_DONE

    crash_env = dict(env, state=env["root"] / "state2", count=env["root"] / "count2")
    crash = {"name": "0-smoke", "cap_hours": 0.5, "units": [
        unit(crash_env, "crash", "--exit", "1", "--no-row", accept_returncodes=[0, 1])]}
    path = config(crash_env, [crash, after_phase(crash_env)])
    assert cd.main(["--config", str(path)]) == cd.EXIT_FAILED
    assert "workhorse" not in invocations(crash_env)


def test_a_unit_that_writes_no_row_is_a_failure(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    phase = {"name": "p", "cap_hours": 0.5, "units": [unit(env, "silent", "--no-row")]}
    assert cd.main(["--config", str(config(env, [phase, after_phase(env)]))]) == cd.EXIT_FAILED


# --- resume ---------------------------------------------------------------------------


def test_resume_skips_finished_units_and_reruns_the_interrupted_one(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    phases = [
        {"name": "0-smoke", "cap_hours": 0.5, "units": [unit(env, "smoke")]},
        {"name": "1-diag", "cap_hours": 0.5,
         "units": [unit(env, "diag-a"), unit(env, "diag-b")]},
    ]
    path = config(env, phases)
    cfg = cd.load_config(path)
    # A driver that died inside diag-b: smoke and diag-a are done, diag-b was running.
    state = cd.load_state(cfg, time.time())
    state["phases"] = {
        "0-smoke": {"status": "done", "units": {"smoke": {"status": "done", "attempts": 1}},
                    "ledger_lines_at_start": 0, "started_at": time.time()},
        "1-diag": {"status": "running", "started_at": time.time(), "ledger_lines_at_start": 0,
                   "gate_lines_at_start": None,
                   "units": {"diag-a": {"status": "done", "attempts": 1},
                             "diag-b": {"status": "running", "attempts": 1,
                                        "ledger_lines_at_start": 0}}},
    }
    cd.write_state(cfg.state_dir / cd.STATE_NAME, state)
    assert cd.main(["--config", str(path)]) == cd.EXIT_DONE
    assert invocations(env) == ["diag-b"]
    final = json.loads((env["state"] / cd.STATE_NAME).read_text())
    assert final["phases"]["1-diag"]["units"]["diag-b"]["attempts"] == 2


def test_resume_under_a_different_config_is_refused(env: dict[str, Path]) -> None:
    path = config(env, [after_phase(env)])
    cfg = cd.load_config(path)
    cd.write_state(cfg.state_dir / cd.STATE_NAME,
                   {**cd.load_state(cfg, time.time()), "phases": {"x": {}}})
    other = config(env, [after_phase(env)], campaign_cap_hours=0.9)
    with pytest.raises(cd.ConfigRefused, match="different campaign"):
        cd.load_state(cd.load_config(other), time.time())


def test_checkpoint_cadence_is_derived_from_measured_throughput(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    ckdir = env["root"] / "ck"
    ckdir.mkdir()
    phases = [
        {"name": "0-smoke", "cap_hours": 0.5,
         "units": [unit(env, "smoke", "--steps", "360", "--wall", "3600")]},
        {"name": "4-workhorse", "cap_hours": 30, "units": [unit(
            env, "workhorse", "--checkpoint-dir", str(ckdir), "--checkpoint-every",
            "{checkpoint_every}",
            checkpoint={"dir": str(ckdir), "every_hours": 2, "throughput_from": "0-smoke"})]},
    ]
    cfg = cd.load_config(config(env, phases, campaign_cap_hours=40))
    workhorse = cfg.phases[1].units[0]
    state = cd.load_state(cfg, time.time())
    with pytest.raises(cd.ConfigRefused, match="measured rate"):
        cd.unit_argv(workhorse, cfg, state, resuming=False)

    assert cd.main(["--config", str(config(env, phases[:1]))]) == cd.EXIT_DONE
    state["phases"]["0-smoke"] = {"status": "done", "ledger_lines_at_start": 0}
    argv = cd.unit_argv(workhorse, cfg, state, resuming=False)
    # 360 steps / 3600 s = 0.1 steps/s; 2 h -> 720 steps.
    assert argv[argv.index("--checkpoint-every") + 1] == "720"
    assert "--resume-from" not in argv
    (ckdir / "epoch-seed0-cuda.json").write_text("{}")
    argv = cd.unit_argv(workhorse, cfg, state, resuming=True)
    assert argv[-2:] == ["--resume-from", str(ckdir / "epoch-seed0-cuda.json")]


def test_gate_verdict_reads_only_the_named_tool_and_this_phase(env: dict[str, Path]) -> None:
    ledger = env["ledger"]
    assert not ledger.exists()
    gate = cd.GateSpec(ledger, "paired_margin_vs_linear", "tools/ft_linear_control.py", 1)
    go, why = cd.gate_verdict(gate, 0)
    assert not go and "never scored" in why


def test_a_crash_that_writes_a_failed_row_is_not_accepted(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    puller(env["state"], env["mac"])
    """The optim.py refusal the GPU lane hit is raised inside the run; RunRecorder then writes
    a row whose status is not `completed`. An accepted exit code must not launder that."""
    phase = {"name": "1-diag", "cap_hours": 0.5, "units": [
        unit(env, "refused", "--raise", accept_returncodes=[0, 1])]}
    assert cd.main(["--config", str(config(env, [phase, after_phase(env)]))]) == cd.EXIT_FAILED
    assert "workhorse" not in invocations(env)
    state = json.loads((env["state"] / cd.STATE_NAME).read_text())
    assert state["phases"]["1-diag"]["status"] == "failed"


def test_a_real_backbone_unit_must_state_its_optimizer(env: dict[str, Path]) -> None:
    argv = ["python", "tools/real_ft_run.py", "--real-backbone", "/snap", "--devices", "cuda",
            "--wall-clock-cap-s", "3600"]
    phase = {"name": "1-diag", "cap_hours": 1, "units": [{"name": "a2", "argv": argv}]}
    with pytest.raises(cd.ConfigRefused, match="explicit"):
        cd.load_config(config(env, [phase]))
    phase["units"][0]["argv"] = [*argv, "--optimizer", "master"]  # type: ignore[index]
    assert cd.load_config(config(env, [phase])).phases[0].units[0].argv[-1] == "master"


@pytest.mark.parametrize(
    ("cap", "match"),
    [(None, "without --wall-clock-cap-s"), (["3601"], "not in \\(0, 3600\\]"),
     (["0"], "not in \\(0, 3600\\]"), (["nan"], "not in"), ([], "needs a number"),
     (["soon"], "needs a number")],
)
def test_a_trainer_unit_states_a_cap_that_fits_its_phase(
    env: dict[str, Path], cap: list[str] | None, match: str
) -> None:
    """real_ft_run's own default cap is 30 minutes; a phase unit that inherited it would
    stop there and write only quick rows. One above the phase cap never gets to end it."""
    argv = ["python", "tools/real_ft_run.py", "--devices", "cpu"]
    if cap is not None:
        argv += ["--wall-clock-cap-s", *cap]
    phase = {"name": "3-go", "cap_hours": 1, "units": [{"name": "s0", "argv": argv}]}
    with pytest.raises(cd.ConfigRefused, match=match):
        cd.load_config(config(env, [phase]))


def test_a_trainer_unit_cap_at_or_under_its_phase_cap_is_accepted(env: dict[str, Path]) -> None:
    argv = ["python", "tools/real_ft_run.py", "--devices", "cpu", "--wall-clock-cap-s", "3600"]
    phase = {"name": "3-go", "cap_hours": 1, "units": [{"name": "s0", "argv": argv}]}
    assert cd.load_config(config(env, [phase])).phases[0].units[0].argv[-1] == "3600"


def test_two_units_may_not_share_a_checkpoint_dir(env: dict[str, Path]) -> None:
    ck = str(env["root"] / "ck")
    spec = {"dir": ck, "every_hours": 2, "throughput_from": "0-smoke"}
    args = ("--checkpoint-dir", ck, "--checkpoint-every", "{checkpoint_every}")
    phases = [
        {"name": "0-smoke", "cap_hours": 0.5, "units": [unit(env, "smoke")]},
        {"name": "4-workhorse", "cap_hours": 30, "units": [
            unit(env, "seed0", *args, checkpoint=spec),
            unit(env, "seed1", *args, checkpoint=spec)]},
    ]
    with pytest.raises(cd.ConfigRefused, match="share a checkpoint dir"):
        cd.load_config(config(env, phases, campaign_cap_hours=40))


def test_a_reissued_pid_is_not_mistaken_for_the_guard() -> None:
    assert cd._alive(os.getpid())
    assert not cd._is_guard(os.getpid()), "the test runner is alive but is not a guard"


def _finish_state(env: dict[str, Path], **over: object) -> tuple[cd.Config, dict[str, object]]:
    """A campaign whose phases ran; what is left is the gated finalize."""
    path = config(env, [after_phase(env)], **over)
    cfg = cd.load_config(path)
    state = cd.load_state(cfg, time.time())
    env["ledger"].write_text('{"row": 1}\n')
    cd.write_state(cfg.state_dir / cd.STATE_NAME, state)
    return cfg, state


# --- the handoff: termination waits for the Mac ------------------------------------------


def test_no_acknowledgement_and_no_durable_copy_keeps_the_box_alive(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg, state = _finish_state(env, usd_per_hour=1.49, device="cuda", n_gpus=1,
                               instance="lambda-1xGH200", usd_per_hour_source="test",
                               approved_by="test")
    started = time.monotonic()
    code = cd.finalize(cfg, state, "completed every phase", cd.EXIT_DONE, phase="x")
    assert code == cd.EXIT_KEPT_ALIVE
    assert time.monotonic() - started >= 3.0 * 0.9, "it must wait the grace window first"
    assert not env["term"].exists(), "terminated with no proof the data is off the box"
    out = capsys.readouterr().out
    assert "INSTANCE KEPT ALIVE" in out and "$1.49/h" in out
    saved = json.loads((cfg.state_dir / cd.STATE_NAME).read_text())
    assert saved["finalized"] == "kept_alive"


def test_a_late_pull_then_finalize_terminates(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    cfg, state = _finish_state(env)
    assert cd.finalize(cfg, state, "done", cd.EXIT_DONE, phase="x") == cd.EXIT_KEPT_ALIVE
    puller(env["state"], env["mac"])
    assert cd.main(["--config", str(sorted(env["root"].glob("campaign-*.json"))[-1]),
                    "--finalize"]) == cd.EXIT_DONE
    assert env["term"].exists()


def test_an_acknowledgement_of_a_different_manifest_is_not_accepted(env: dict[str, Path]) -> None:
    cfg, state = _finish_state(env)
    seq = cd.write_snapshot(cfg, state, phase="x", final=True)
    (cfg.sync_dir / f"pulled-{seq}.ok").write_text(json.dumps({"done_sha256": "0" * 64}))
    ok, why = cd.pulled_ok(cfg, seq)
    assert not ok and "not this manifest" in why


def test_a_durable_copy_on_the_instance_disk_is_not_durable(env: dict[str, Path]) -> None:
    durable = env["root"] / "persistent"
    durable.mkdir()
    cfg, state = _finish_state(env, durable_copy_dir=str(durable))
    assert cd.finalize(cfg, state, "done", cd.EXIT_DONE, phase="x") == cd.EXIT_KEPT_ALIVE
    assert not env["term"].exists()
    events = json.loads((cfg.state_dir / cd.STATE_NAME).read_text())["events"]
    assert any("same device" in e["text"] for e in events)


def test_a_verified_durable_copy_on_another_device_permits_termination(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = env["root"] / "persistent"
    durable.mkdir()
    cfg, state = _finish_state(env, durable_copy_dir=str(durable))
    # Stands in for a Lambda persistent filesystem mount; the hashes are still checked.
    monkeypatch.setattr(cd, "_same_device", lambda a, b: False)
    assert cd.finalize(cfg, state, "done", cd.EXIT_DONE, phase="x") == cd.EXIT_DONE
    assert env["term"].exists()


def test_a_durable_copy_that_differs_is_refused(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    durable = env["root"] / "persistent"
    durable.mkdir()
    cfg, state = _finish_state(env, durable_copy_dir=str(durable))
    monkeypatch.setattr(cd, "_same_device", lambda a, b: False)
    real_copy = cd.copy_to_durable

    def corrupting_copy(c: cd.Config, seq: int) -> str:
        msg = real_copy(c, seq)
        (durable / c.name / f"phase-{seq}" / "ledger" / "ledger.jsonl").write_text("tampered\n")
        return msg

    monkeypatch.setattr(cd, "copy_to_durable", corrupting_copy)
    assert cd.finalize(cfg, state, "done", cd.EXIT_DONE, phase="x") == cd.EXIT_KEPT_ALIVE
    assert not env["term"].exists()


def test_the_up_front_estimate_includes_the_grace_window(env: dict[str, Path]) -> None:
    cfg = cd.load_config(config(env, [after_phase(env)], usd_per_hour=2.0, device="cuda",
                                n_gpus=1, usd_per_hour_source="test", approved_by="test",
                                campaign_cap_hours=10, pull_grace_minutes=45))
    assert cd.campaign_usd(cfg) == pytest.approx(2.0 * 10.75)
    assert "45 min pull grace window" in "\n".join(cd.cost_lines(cfg))


def test_retired_sync_command_and_unsourced_rates_are_refused(env: dict[str, Path]) -> None:
    with pytest.raises(cd.ConfigRefused, match="retired"):
        cd.load_config(config(env, [after_phase(env)], sync_command=["true"]))
    with pytest.raises(cd.ConfigRefused, match="usd_per_hour_source"):
        cd.load_config(config(env, [after_phase(env)], usd_per_hour=1.49, device="cuda",
                              n_gpus=1, approved_by="test"))


def test_the_guard_stops_the_run_and_still_waits_for_the_mac(
    env: dict[str, Path], puller: Callable[[Path, Path], None]
) -> None:
    cfg, state = _finish_state(env)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                               start_new_session=True)
    state["running_pgid"] = sleeper.pid
    cd.write_state(cfg.state_dir / cd.STATE_NAME, state)
    # No Mac yet: the guard stops the unit, snapshots, waits, and keeps the box alive.
    assert cd.guard_fire(cfg) == cd.EXIT_KEPT_ALIVE
    assert sleeper.wait(timeout=10) is not None
    assert not env["term"].exists()
    # The Mac comes back: a second firing (or --finalize) terminates.
    puller(env["state"], env["mac"])
    assert cd.guard_fire(cfg) == cd.EXIT_FAILED  # the campaign was capped, not completed
    assert env["term"].exists()


def test_only_one_process_finalizes(env: dict[str, Path]) -> None:
    cfg, state = _finish_state(env)
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                               "campaign_driver.py"], start_new_session=True)
    try:
        (cfg.state_dir / cd.LOCK_NAME).write_text(str(holder.pid))
        assert cd.finalize(cfg, state, "done", cd.EXIT_DONE, phase="x") == cd.EXIT_DONE
        assert not env["term"].exists(), "a second finalizer must defer to the lock holder"
        assert not (cfg.sync_dir).exists()
    finally:
        holder.kill()
    # A lock whose holder is dead is broken, not obeyed forever.
    assert cd.acquire_lock(cfg)


def test_a_campaign_process_is_known_past_column_80() -> None:
    """Linux procps cut ``ps -o command=`` at 80 columns when stdout is a pipe, so a driver or
    guard started from a long path read as not ours: a second finalizer ran beside a live one
    and a live guard read as dead (the H100 box suite at 8e6a009). macOS never cut it, so this
    fails only on Linux before the fix."""
    pad = "/" + "x" * 160
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", pad,
                               "campaign_driver.py", "--guard"], start_new_session=True)
    try:
        line = cd._cmdline(holder.pid)
        assert pad in line and line.rstrip().endswith("campaign_driver.py --guard"), line
        assert cd._is_campaign_process(holder.pid)
        assert cd._is_guard(holder.pid)
    finally:
        holder.kill()
        holder.wait(timeout=30)


# --- the Mac side ------------------------------------------------------------------------


def test_pull_needs_both_paths(tmp_path: Path) -> None:
    out = subprocess.run(["bash", str(REPO / "tools" / "sync_box.sh"), "pull"],
                         capture_output=True, text=True, timeout=60, check=False,
                         env={**os.environ, "QD_BOX": "local"})
    assert out.returncode == 1 and "QD_PULL_DEST" in out.stdout


def test_pull_verifies_acknowledges_and_refuses_a_corrupt_snapshot(env: dict[str, Path]) -> None:
    cfg, state = _finish_state(env)
    assert pull_once(env["state"], env["mac"]).returncode == 4  # nothing published yet
    seq = cd.write_snapshot(cfg, state, phase="0", final=False)
    first = pull_once(env["state"], env["mac"])
    assert first.returncode == 0, first.stdout + first.stderr
    assert f"snapshot {seq} PULLED AND VERIFIED" in first.stdout
    assert cd.pulled_ok(cfg, seq)[0]
    assert pull_once(env["state"], env["mac"]).returncode == 4  # acknowledged, not re-pulled

    seq2 = cd.write_snapshot(cfg, state, phase="1", final=True)
    (cfg.sync_dir / f"phase-{seq2}" / "ledger" / "ledger.jsonl").write_text("bit rot\n")
    bad = pull_once(env["state"], env["mac"])
    assert bad.returncode == 1 and "does NOT match its manifest" in bad.stdout
    assert not (cfg.sync_dir / f"pulled-{seq2}.ok").exists(), "acknowledged unverified data"


def test_the_pull_loop_ends_at_the_final_snapshot(env: dict[str, Path]) -> None:
    cfg, state = _finish_state(env)
    cd.write_snapshot(cfg, state, phase="0", final=False)
    cd.write_snapshot(cfg, state, phase="end", final=True)
    out = subprocess.run(
        ["bash", str(REPO / "tools" / "campaign_pull_loop.sh")], capture_output=True, text=True,
        timeout=120, check=False,
        env={**os.environ, "QD_BOX": "local", "QD_PULL_DEST": str(env["mac"]),
             "QD_PULL_SYNC_DIR": str(env["state"] / "sync"), "QD_PYTHON": sys.executable,
             "QD_PULL_INTERVAL_S": "1"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "final snapshot acknowledged" in out.stdout
    assert cd.pulled_ok(cfg, 2)[0]


def test_the_pull_loop_is_bounded(env: dict[str, Path]) -> None:
    out = subprocess.run(
        ["bash", str(REPO / "tools" / "campaign_pull_loop.sh")], capture_output=True, text=True,
        timeout=120, check=False,
        env={**os.environ, "QD_BOX": "local", "QD_PULL_DEST": str(env["mac"]),
             "QD_PULL_SYNC_DIR": str(env["state"] / "sync"), "QD_PYTHON": sys.executable,
             "QD_PULL_INTERVAL_S": "1", "QD_PULL_LOOP_MAX_HOURS": "0.0005"})
    assert out.returncode == 1 and "without the FINAL snapshot" in out.stdout
