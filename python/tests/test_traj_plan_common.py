"""``campaign/post-f-queue/traj_plan_common.sh``: v6's trajectory block, driven on the Mac.

``traj_score_plans`` scores every retained snapshot of a seed through ``tools/real_ft_run.py
--score-plan`` 'trajectory' kinds, up to ``TRAJ_PLAN_MAX_KINDS`` per process, where v5's waiter
(``box_q_v5traj.sh``) starts one process per snapshot. These tests run the function under
``/bin/bash`` with a fake ``python`` -- it records each call's argv and plan, writes the suite-
verdict files of the kinds a scenario says it scored, and exits with the scenario's code -- and a
fake ``timeout`` that records its seconds. What is pinned: the snapshots and their order are v5's;
each plan names v5's files; a snapshot already scored is never scored twice; a failed plan's
in-flight snapshot is left NOT RUN and the rest requeued; a plan that scored nothing, the cap
and a timeout stop the block; everything not scored is said NOT RUN; and the chunk size is
``real_ft_run.PLAN_MAX_KINDS``. No GPU, no torch: the rows themselves are
``test_score_plan_trajectory.py``'s.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "campaign" / "post-f-queue" / "traj_plan_common.sh"
BASH = "/bin/bash"
PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
FT = "8079c966-0000-4000-8000-000000000000"
REC = "/home/ubuntu/rec.json"
SPLIT = ("--out", "/home/ubuntu/v5-data", "--no-repo-history", "--rev", "0" * 40)
COST = ("--instance", "lambda:gpu0", "--usd-per-hour", "4.19", "--usd-per-gpu-hour", "4.19")
#: v5 seed 2's retained steps on the H100 (--retain-tower-every 1000, final 12176): 13 snapshots.
STEPS = (*range(1000, 13000, 1000), 12176)
V5_ORDER = [12176, *range(1000, 13000, 1000)]

FAKE_PYTHON = r"""#!{real}
import json, os, sys
ROOT = {root!r}
args = sys.argv[1:]
scen = json.load(open(os.path.join(ROOT, "scenario.json")))
count = os.path.join(ROOT, "calls.count")
n = int(open(count).read()) if os.path.exists(count) else 0
open(count, "w").write(str(n + 1))
plan_path = args[args.index("--score-plan") + 1]
plan = json.load(open(plan_path))
with open(os.path.join(ROOT, "calls.jsonl"), "a") as f:
    f.write(json.dumps({{"argv": args, "plan": plan}}) + "\n")
step = (scen.get("plans") or [{{}}])[min(n, len(scen.get("plans") or [{{}}]) - 1)]
score = step.get("score")
stem = args[args.index("--suite-verdicts-out") + 1]
base, ext = os.path.splitext(stem)
for kind in plan["kinds"][:score]:
    out = f"{{base}}-{{kind['name']}}{{ext}}"
    assert not os.path.exists(out), out
    open(out, "w").write(json.dumps({{"kind": kind["name"]}}) + "\n")
print(f"score plan: {{len(plan['kinds'][:score])}} kinds scored")
sys.exit(step.get("rc", 0))
"""

FAKE_TIMEOUT = r"""#!/bin/bash
echo "$1" >> {root}/timeout.log
shift
"$@"
rc=$?
if [ -e {root}/timeout.rc ]; then exit "$(cat {root}/timeout.rc)"; fi
exit $rc
"""


class Block:
    """A temp box: a checkpoint dir holding ``steps``, an output dir, and the fakes."""

    def __init__(self, tmp: Path, steps=STEPS, scenario: dict | None = None):
        self.root = tmp
        self.ckpt = tmp / "ckpt"
        self.out = tmp / "out"
        self.bin = tmp / "bin"
        for d in (self.ckpt, self.out, self.bin):
            d.mkdir()
        for s in steps:
            (self.ckpt / f"epoch-seed2-cuda-step{s}.json").write_text("{}", encoding="utf-8")
        # Not snapshots of seed 2: another seed's, the final checkpoint, a malformed step.
        for name in ("epoch-seed3-cuda-step1000.json", "epoch-seed2-cuda.json",
                     "epoch-seed2-cuda-stepX.json"):
            (self.ckpt / name).write_text("{}", encoding="utf-8")
        (tmp / "scenario.json").write_text(json.dumps(scenario or {}), encoding="utf-8")
        real = shutil.which("python3", path=PATH) or "/usr/bin/python3"
        self.python = self._exec("python", FAKE_PYTHON.format(real=real, root=str(tmp)))
        self._exec("timeout", FAKE_TIMEOUT.format(root=str(tmp)))

    def _exec(self, name: str, body: str) -> Path:
        path = self.bin / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        return path

    def run(self, *, ft: str = FT, cap: int = 3600, min_left: int = 120, ckpt: str | None = None,
            t0: str = "now", pipefail: bool = False) -> subprocess.CompletedProcess:
        start = "$(date +%s)" if t0 == "now" else shlex.quote(t0)
        call = " ".join(shlex.quote(str(a)) for a in (
            "v6traj-s2", 2, ft, ckpt or self.ckpt, self.out, self.root / "ledger.jsonl", cap,
            min_left,
        ))
        script = (
            ("set -o pipefail\n" if pipefail else "")
            + 'say() { echo "=== $*"; }\n'
            f"source {shlex.quote(str(SCRIPT))}\n"
            f"PY={shlex.quote(str(self.python))}\nREC={REC}\n"
            f"TRAJ_SPLIT=({' '.join(SPLIT)})\nTRAJ_COST=({' '.join(COST)})\n"
            f"T0={start}\n"
            f'traj_score_plans {call} "$T0"\n'
            'echo "rc=$?"\n'
        )
        return subprocess.run(
            [BASH, "-c", script], capture_output=True, text=True, check=False, timeout=60,
            env={"PATH": f"{self.bin}:{PATH}"}, cwd=self.root,
        )

    def calls(self) -> list[dict]:
        path = self.root / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]

    def timeouts(self) -> list[int]:
        path = self.root / "timeout.log"
        return [int(x) for x in path.read_text().split()] if path.exists() else []

    def scored(self) -> set[int]:
        return {int(m.group(1)) for p in self.out.iterdir()
                if (m := re.fullmatch(r"traj-s2-step(\d+)\.jsonl", p.name))}


def _opts(argv: list[str]):
    """``opt(flag)``: the value after ``flag`` in ``argv``."""
    return lambda flag: argv[argv.index(flag) + 1]


def _plan_steps(call: dict) -> list[int]:
    return [int(k["name"].removeprefix("step")) for k in call["plan"]["kinds"]]


def _not_run(text: str) -> list[int]:
    found = re.findall(r"steps ([\d ]+) NOT RUN", text)
    assert len(found) <= 1, found
    return [int(s) for s in found[0].split()] if found else []


# --- static --------------------------------------------------------------------------------------


def test_bash_n_parses_it():
    r = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_shellcheck_is_clean():
    sc = shutil.which("shellcheck", path=f"/opt/homebrew/bin:{PATH}")
    if sc is None:
        pytest.skip("shellcheck is not installed here: NOT RUN (not a pass)")
    r = subprocess.run([sc, "-x", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def _tool_constant(name: str) -> object:
    """A module-level constant of tools/real_ft_run.py, read with ast: importing the tool
    imports torch, which this test's venv does not hold."""
    tree = ast.parse((REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)  # type: ignore[arg-type]
    raise AssertionError(f"{name} is not a module-level annotated constant of real_ft_run.py")


def test_the_chunk_is_the_tools_plan_bound():
    text = SCRIPT.read_text(encoding="utf-8")
    (value,) = re.findall(r"^TRAJ_PLAN_MAX_KINDS=(\d+)$", text, flags=re.M)
    assert int(value) == _tool_constant("PLAN_MAX_KINDS")


# --- the block -----------------------------------------------------------------------------------


def test_every_snapshot_in_v5s_order_two_plans_one_startup_each(tmp_path):
    block = Block(tmp_path)
    r = block.run()
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    calls = block.calls()
    assert [_plan_steps(c) for c in calls] == [V5_ORDER[:8], V5_ORDER[8:]]
    assert block.scored() == set(STEPS)
    assert _not_run(r.stdout) == []
    for call, seconds in zip(calls, block.timeouts(), strict=True):
        argv = call["argv"]
        assert argv[:2] == ["-u", "tools/real_ft_run.py"]
        assert argv[2:2 + len(SPLIT)] == list(SPLIT)
        assert all(a in argv for a in COST)
        opt = _opts(argv)
        assert opt("--devices") == "cuda" and "--ood" in argv
        assert opt("--ood-general-record") == REC
        assert opt("--ft-ledger") == opt("--ledger") == str(tmp_path / "ledger.jsonl")
        assert opt("--suite-verdicts-out") == str(block.out / "traj-s2.jsonl")
        assert int(opt("--wall-clock-cap-s")) == seconds <= 3600
        # A plan names checkpoints, ft rows and seeds itself; --score-val would make it a gate run.
        for flag in ("--seeds", "--ft-row-id", "--score-checkpoint", "--score-val", "--needle"):
            assert flag not in argv
        for kind in call["plan"]["kinds"]:
            step = kind["name"].removeprefix("step")
            assert kind == {
                "name": f"step{step}",
                "checkpoints": [str(block.ckpt / f"epoch-seed2-cuda-step{step}.json")],
                "ft_row_ids": [FT], "seeds": [2], "passes": ["trajectory"],
            }
            assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", kind["name"])


def test_a_snapshot_already_scored_is_not_scored_twice(tmp_path):
    block = Block(tmp_path)
    (block.out / "traj-s2-step3000.jsonl").write_text("{}\n", encoding="utf-8")
    r = block.run()
    assert "step 3000: " in r.stdout and "refusing to score it twice" in r.stdout
    planned = [s for c in block.calls() for s in _plan_steps(c)]
    assert planned == [s for s in V5_ORDER if s != 3000]
    assert _not_run(r.stdout) == []


@pytest.mark.parametrize("pipefail", [False, True])
def test_a_failed_plan_leaves_its_inflight_snapshot_not_run_and_requeues_the_rest(
    tmp_path, pipefail
):
    """The plan's exit status is timeout's whether or not the caller set pipefail (the v5
    waiters do; a caller that does not must not read tee's 0 as a clean exit)."""
    block = Block(tmp_path, scenario={"plans": [{"score": 3, "rc": 1}, {}]})
    r = block.run(pipefail=pipefail)
    calls = block.calls()
    assert _plan_steps(calls[0]) == V5_ORDER[:8]
    # 12176, 1000 and 2000 scored; 3000 was in flight when it exited.
    assert "step 3000 is the one it was scoring" in r.stdout
    assert [_plan_steps(c) for c in calls[1:]] == [V5_ORDER[4:12], V5_ORDER[12:]]
    assert block.scored() == set(STEPS) - {3000}
    assert _not_run(r.stdout) == [3000]


def test_a_plan_that_scored_nothing_stops_the_block(tmp_path):
    block = Block(tmp_path, scenario={"plans": [{"score": 0, "rc": 1}]})
    r = block.run()
    assert len(block.calls()) == 1
    assert "before scoring any snapshot" in r.stdout and "not retried" in r.stdout
    assert _not_run(r.stdout) == V5_ORDER


def test_a_plan_that_exits_0_short_of_its_kinds_stops_the_block(tmp_path):
    block = Block(tmp_path, scenario={"plans": [{"score": 5, "rc": 0}]})
    r = block.run()
    assert len(block.calls()) == 1
    assert "a plan that skips a kind is not understood" in r.stdout
    assert _not_run(r.stdout) == V5_ORDER[5:]


def test_the_cap_stopping_a_plan_stops_the_block(tmp_path):
    block = Block(tmp_path, scenario={"plans": [{"score": 2, "rc": 1}]})
    (tmp_path / "timeout.rc").write_text("124", encoding="utf-8")
    r = block.run()
    assert len(block.calls()) == 1
    assert "the cap stopped it after 2 of 8 snapshots" in r.stdout
    assert _not_run(r.stdout) == V5_ORDER[2:]


def test_no_plan_starts_with_less_than_the_minimum_left(tmp_path):
    block = Block(tmp_path)
    r = block.run(t0="1")  # started in 1970: the cap is long gone
    assert block.calls() == [] and "rc=0" in r.stdout
    assert _not_run(r.stdout) == V5_ORDER


def test_no_snapshot_is_said_not_run(tmp_path):
    block = Block(tmp_path, steps=())
    r = block.run()
    assert block.calls() == [] and "trajectory NOT RUN" in r.stdout and "rc=0" in r.stdout


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"ft": "8079c966"}, "is not a row id"),
        ({"ckpt": "/home/ubuntu/ckpt/v5 x"}, "it would be written into the plan unescaped"),
        ({"ckpt": 'relative"quote'}, "it would be written into the plan unescaped"),
        ({"cap": -1}, "is not a whole number"),
    ],
)
def test_what_cannot_be_written_into_a_plan_is_refused_before_anything_runs(tmp_path, kw, match):
    block = Block(tmp_path)
    r = block.run(**kw)
    assert match in r.stdout and "rc=3" in r.stdout
    assert block.calls() == []
