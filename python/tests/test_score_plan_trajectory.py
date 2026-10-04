"""``--score-plan`` 'trajectory' kinds: v5's OOD-only trajectory row, many snapshots, one startup.

v5 scores the OOD suite on each retained snapshot of a seed with one ``--score-checkpoint`` call
per snapshot (``campaign/post-f-queue/box_q_v5traj.sh``: ``--ood`` without ``--score-val``, tag
``trajectory-ood``). Every call rebuilds the split, the val set and the OOD suite: ~6.5 min of
wall time for ~16 s of GPU on the H100 (HANDOFF/lead-pipeline-2026-10-03.md, ~07:15Z
2026-10-04), so the 3,600 s cap fits ~9 of 13 snapshots. A plan builds them once.

Before this, no plan kind could express that row: the plan's ``ood`` pass writes the plan
DIAGNOSTIC row (tag ``<model>-ood-diagnostic``, metrics ``ood_diagnostic.*``, gate
``ood_abstain.diagnostic``, no ``checkpoint_step``), and every plan needed ``--score-val``. What
is pinned here:

* a ``trajectory`` kind is one snapshot's OOD-only trajectory row and nothing else; a plan is all
  trajectory kinds or none; a trajectory plan is scored without ``--score-val`` and needs
  ``--ood``, as the per-snapshot call is; the ``ood`` pass and its diagnostic row are unchanged;
* PLUMBING PARITY ON THE TOY CORPUS, NOT 2B PARITY: a tiny real-backbone tower trained for three
  steps with a snapshot at every step, steps 3 and 1 scored the v5 way (two ``--score-checkpoint``
  calls) and the plan way (one ``--score-plan``), and the two compared row by row and line by
  line. Exactly three things may differ, and each is asserted rather than ignored: the row ids
  (and so ``prev_row_hash``, ``written_at`` and ``wall_clock_s``), the plan's sentence appended
  to the row's ``notes``, and the ``score_kind`` key every plan line carries. The same
  comparison on two real v5 snapshots is NOT RUN (it needs the H100 box's snapshots; the exact
  command is in HANDOFF/v6-startup-2026-10-04.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_checkpoint_retain import _main  # noqa: E402
from test_real_ft_family_metrics import _tokenizer_json  # noqa: E402
from test_real_ft_rungd_flags import REV, _only, _run, corpus  # noqa: E402, F401

from qd_train.ood import OOD_CATEGORIES, OodCase  # noqa: E402

# --- reading a plan ------------------------------------------------------------------------------


def _kind(name: str, *passes: str,
          checkpoints: tuple[str, ...] = ("/c/epoch-seed0-cuda-step2.json",),
          ft_row_ids: tuple[str, ...] = ("aaaaaaaa",), seeds: tuple[int, ...] = (0,)) -> dict:
    return {"name": name, "checkpoints": list(checkpoints), "ft_row_ids": list(ft_row_ids),
            "seeds": list(seeds), "passes": list(passes)}


def _plan(tmp_path: Path, *kinds: dict) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"kinds": list(kinds)}), encoding="utf-8")
    return path


def test_trajectory_is_a_plan_pass_and_a_plan_of_them_is_ood_only(tmp_path):
    plan = rft.read_score_plan(_plan(
        tmp_path, _kind("step2", "trajectory"),
        _kind("step1", "trajectory", checkpoints=("/c/epoch-seed0-cuda-step1.json",)),
    ))
    assert "trajectory" in rft.PLAN_PASSES
    assert [k.passes for k in plan.kinds] == [("trajectory",), ("trajectory",)]
    assert plan.ood_only


@pytest.mark.parametrize("other", ["gates", "ood", "composed"])
def test_a_trajectory_kind_runs_nothing_else(tmp_path, other):
    """``gates`` and ``ood`` would decode the OOD suite a second time; a trajectory point is one
    snapshot's OOD-only row, so a pass beside it is refused rather than given a meaning."""
    with pytest.raises(SystemExit, match="a 'trajectory' kind is one snapshot's OOD-only row"):
        rft.read_score_plan(_plan(tmp_path, _kind("step2", "trajectory", other)))


def test_the_plans_other_kinds_are_not_ood_only(tmp_path):
    plan = rft.read_score_plan(_plan(tmp_path, _kind("seed0", "ood")))
    assert not plan.ood_only


def test_a_trajectory_kinds_namespace_is_the_ood_only_mode_and_no_other_kinds_is():
    """``_ood_only`` is the one owner of "this namespace is the trajectory row": argv checks it,
    and ``_seed_weights`` scores a snapshot before the final step only under it. A trajectory
    kind's namespace is that mode; an 'ood' kind's and the plan's own are not."""
    args = argparse.Namespace(score_checkpoint=None, score_plan=Path("plan.json"),
                              score_val=False, ood=True, ft_row_id=None, seeds=[])
    trajectory = rft.PlanKind("step2", (Path("/c/epoch-seed0-cuda-step2.json"),), ("aaaaaaaa",),
                              (0,), ("trajectory",))
    diagnostic = dataclasses.replace(trajectory, name="seed0", passes=("ood",))
    assert rft._ood_only(rft.plan_kind_args(args, trajectory))
    assert not rft._ood_only(rft.plan_kind_args(args, diagnostic))
    assert not rft._ood_only(args), "the plan's own namespace is no one model"
    with_val = argparse.Namespace(**{**vars(args), "score_val": True})
    assert not rft._ood_only(rft.plan_kind_args(with_val, trajectory))


def test_the_kind_name_lands_on_v5s_per_snapshot_file_name():
    """Characterization (true before and after): a kind named ``step<S>`` under v5's
    ``--suite-verdicts-out`` stem writes the very file v5's per-snapshot call writes, so the
    plan's own exists-refusal is v5's 'refusing to score it twice'."""
    assert rft.plan_output(Path("/v5/traj-s2.jsonl"), "step12176") == Path(
        "/v5/traj-s2-step12176.jsonl"
    )


# --- argv ----------------------------------------------------------------------------------------


def _argv(tmp_path: Path, plan: Path, *extra: str) -> list[str]:
    return ["--out", str(tmp_path), "--rev", "0" * 40, "--score-plan", str(plan),
            "--real-backbone", "/x", "--ft-ledger", "/l", "--devices", "cpu", *extra]


_OOD = ("--ood", "--ood-general-record", "/g.json")


def test_a_trajectory_plan_passes_every_argv_check_without_score_val(tmp_path):
    """It stops at the first thing that reads the disk: resolving --rev, which no repository
    here has. Before 'trajectory' existed the plan was refused at its first kind."""
    plan = _plan(tmp_path, _kind("step2", "trajectory"))
    with pytest.raises(subprocess.CalledProcessError, match="rev-parse"):
        rft.main(_argv(tmp_path, plan, *_OOD))


@pytest.mark.parametrize(
    ("kinds", "extra", "match"),
    [
        ([_kind("step2", "trajectory")], ("--score-val", *_OOD),
         "a 'trajectory' kind is the OOD-only row, scored without --score-val"),
        ([_kind("step2", "trajectory")], (),
         "a 'trajectory' pass decodes the OOD suite: it needs --ood"),
        ([_kind("step2", "trajectory")], ("--needle", *_OOD), "--needle is a gate"),
        ([_kind("step2", "trajectory"), _kind("seed0", "ood")], ("--score-val", *_OOD),
         "one plan is trajectory kinds or none"),
        ([_kind("step2", "trajectory", checkpoints=("/c/epoch-seed0-cuda.json",
                                                    "/c/epoch-seed1-cuda.json"),
                ft_row_ids=("aaaaaaaa", "bbbbbbbb"), seeds=(0, 1))], _OOD,
         "kind 'step2': .*a logit ensemble is scored with --score-val"),
    ],
)
def test_a_trajectory_plan_is_refused_where_the_per_snapshot_call_is(tmp_path, kinds, extra, match):
    with pytest.raises(SystemExit, match=match):
        rft.main(_argv(tmp_path, _plan(tmp_path, *kinds), *extra))


def test_an_ood_diagnostic_plan_still_needs_score_val(tmp_path):
    """Characterization (true before and after): the 'ood' pass is untouched."""
    with pytest.raises(SystemExit, match="kind 'seed0': --score-checkpoint needs --score-val"):
        rft.main(_argv(tmp_path, _plan(tmp_path, _kind("seed0", "ood")), *_OOD))


# --- plumbing parity on the toy corpus (NOT 2B parity) -------------------------------------------


def _toy_ood(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """The OOD suite's construction replaced by the toy val set's choice rows as cases, with the
    real second pass over them (``test_ood_only_scoring``'s stand-in: the real suite needs the
    Qwen tokenizer and a general record). Returns the list every call of the val set's OWN
    second pass is appended to: an OOD-only run must never build it."""
    opened: dict[str, rft.ValSet] = {}
    real_open, real_second = rft.open_val_set, rft.prepare_second_pass
    main_second: list[object] = []

    def open_val_set(*a, **k):
        opened["val"] = real_open(*a, **k)
        return opened["val"]

    def prepare_second_pass(val, **k):
        main_second.append(val)
        return real_second(val, **k)

    def toy_ood(reader, *, config, rev, letter_id, tokenizer_json, general_record, enabled):
        assert enabled
        val = opened["val"]
        choice = [lab for lab in val.labels if lab.slot_kind == rft.SLOT_CHOICE]
        cases = [
            OodCase(case_id=lab.row_id, category=OOD_CATEGORIES[i % len(OOD_CATEGORIES)],
                    language="prose", path="p", context="c")
            for i, lab in enumerate(choice)
        ]
        return rft.OodSuite(
            cases, val, real_second(val, reader=reader, tokenizer_json=tokenizer_json,
                                    seed=config.seed),
            record_sha256=hashlib.sha256(general_record.read_bytes()).hexdigest(),
            seed=config.seed,
        )

    monkeypatch.setattr(rft, "open_val_set", open_val_set)
    monkeypatch.setattr(rft, "prepare_second_pass", prepare_second_pass)
    monkeypatch.setattr(rft, "prepare_ood", toy_ood)
    return main_second


def _eval_rows(ledger: Path) -> list[dict]:
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("run_kind") == "eval"]


#: What two runs that wrote the same row in different processes and ledgers must differ in: the
#: row's identity, its place in its ledger's hash chain, and when and how long it ran.
_IDENTITY = ("row_id", "prev_row_hash", "written_at", "wall_clock_s")


def _without(row: dict, keys: tuple[str, ...]) -> dict:
    return {k: v for k, v in row.items() if k not in keys}


def _lines(path: Path, row_id: str) -> list[dict]:
    """A suite-verdict file's lines, its row id replaced by a placeholder, after checking the
    file is exactly the sorted-key form its writer emits (so comparing re-dumps compares bytes)."""
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(x) for x in text.splitlines()]
    assert text == "".join(json.dumps(x, sort_keys=True) + "\n" for x in lines)
    assert lines and {x["eval_row_id"] for x in lines} == {row_id}
    return [{**x, "eval_row_id": "<row>"} for x in lines]


def test_plumbing_parity_toy_corpus_plan_trajectory_rows_equal_per_snapshot_rows(
    corpus, tmp_path, monkeypatch  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    """PLUMBING PARITY ON THE TOY CORPUS, NOT 2B PARITY. Steps 3 then 1 (the final step first,
    as v5 orders them) scored twice: by two v5-style ``--score-checkpoint`` calls and by one
    ``--score-plan`` of two trajectory kinds. Every row field and every suite-verdict byte must
    agree except the three named differences, which are asserted."""
    ck = tmp_path / "ck"
    ft = _only(
        _run(corpus, tmp_path / "ft.jsonl", "--deterministic", "--max-steps", "3",
             "--checkpoint-dir", str(ck), "--retain-tower-every", "1", real=True, short=True),
        "ft",
    )
    main_second = _toy_ood(monkeypatch)
    record = tmp_path / "general-record.json"
    record.write_text("{}", encoding="utf-8")
    tokenizer = _tokenizer_json(tmp_path / "tokenizer.json")
    common = (
        "--out", str(corpus.out), "--rev", REV, "--devices", "cpu",
        "--real-backbone", str(corpus.snapshot), "--ft-ledger", str(tmp_path / "ft.jsonl"),
        "--tokenizer-json", str(tokenizer), "--ood", "--ood-general-record", str(record),
    )
    steps = (3, 1)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()

    # v5: one process per snapshot (box_q_v5traj.sh).
    old_ledger = tmp_path / "old.jsonl"
    for step in steps:
        code, text = _main(
            corpus, *common, "--seeds", "0", "--ft-row-id", ft.row_id,
            "--score-checkpoint", str(ck / f"epoch-seed0-cpu-step{step}.json"),
            "--ledger", str(old_ledger),
            "--suite-verdicts-out", str(old_dir / f"traj-s0-step{step}.jsonl"),
        )
        assert code == 0, text

    # v6: one process, one kind per snapshot.
    plan_path = tmp_path / "traj-plan.json"
    plan_path.write_text(json.dumps({"kinds": [
        {"name": f"step{step}", "checkpoints": [str(ck / f"epoch-seed0-cpu-step{step}.json")],
         "ft_row_ids": [ft.row_id], "seeds": [0], "passes": ["trajectory"]}
        for step in steps
    ]}), encoding="utf-8")
    new_ledger = tmp_path / "new.jsonl"
    code, text = _main(
        corpus, *common, "--score-plan", str(plan_path), "--ledger", str(new_ledger),
        "--suite-verdicts-out", str(new_dir / "traj-s0.jsonl"),
    )
    assert code == 0, text
    assert "opened for the OOD suite, not decoded" in text
    assert main_second == [], "the val set's own second pass is never built, by either path"

    old_rows, new_rows = _eval_rows(old_ledger), _eval_rows(new_ledger)
    assert len(old_rows) == len(new_rows) == len(steps)
    plan = rft.read_score_plan(plan_path)
    for step, kind, old, new in zip(steps, plan.kinds, old_rows, new_rows, strict=True):
        assert old["recipe"]["tag"] == new["recipe"]["tag"] == rft.OOD_TRAJECTORY_TAG
        assert new["metrics"]["checkpoint_step"]["value"] == step
        assert set(new["metrics"]) == set(old["metrics"])
        assert {
            "ft_run_row_id", "checkpoint_step", "ood_abstain.all",
            *(f"ood_abstain.{c}" for c in OOD_CATEGORIES),
        } <= set(new["metrics"])
        assert not any(k.startswith("ood_diagnostic") for k in new["metrics"])
        # Named difference 2 of 3: the plan's sentence, appended to the very same notes.
        assert new["notes"] == f"{old['notes']} {plan.note(kind)}"
        # Everything else in the row -- metrics, gates, recipe, protocol, quick and its reason,
        # cost, env, controls -- is the per-snapshot call's, field for field.
        assert _without(new, (*_IDENTITY, "notes")) == _without(old, (*_IDENTITY, "notes"))

        old_lines = _lines(old_dir / f"traj-s0-step{step}.jsonl", old["row_id"])
        new_file = rft.plan_output(new_dir / "traj-s0.jsonl", kind.name)
        assert new_file.name == f"traj-s0-step{step}.jsonl", "v5's per-snapshot file name"
        new_lines = _lines(new_file, new["row_id"])
        assert {x["gate"] for x in new_lines} == {rft.OOD_TRAJECTORY_GATE}
        # Named difference 3 of 3: every plan line names its kind.
        assert {x.pop("score_kind") for x in new_lines} == {kind.name}
        assert [json.dumps(x, sort_keys=True) for x in new_lines] == [
            json.dumps(x, sort_keys=True) for x in old_lines
        ]
