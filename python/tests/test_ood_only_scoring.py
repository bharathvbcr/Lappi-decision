"""``tools/real_ft_run.py --score-checkpoint X --ood`` without ``--score-val``: the trajectory row.

v5 scores the 180-case OOD suite on every ``--retain-tower-every`` snapshot of a seed, report-only
(``campaign/v5-preregistered.DRAFT.json``: rows tagged ``trajectory-ood`` with
``metrics.checkpoint_step``). What is pinned here:

* the OOD-only mode is exactly ``--score-checkpoint`` + ``--ood`` without ``--score-val``; a
  ``--score-plan`` kind and the needle worker never enter it, and every other scoring mode
  still needs ``--score-val``;
* its row is the plan diagnostic's decode and counts under ``ood_abstain.<category>`` (the
  names the pre-registration's reading uses), tagged ``trajectory-ood``, quick, with no gate,
  plus ``checkpoint_step``; the plan's diagnostic row is unchanged;
* end to end on the toy corpus: a snapshot before the final step is scored, the val set is
  opened but its second pass is not built, and the suite lines carry the trajectory's gate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_checkpoint_retain import _main  # noqa: E402
from test_ood_abstain import PERM, PROSE, _v  # noqa: E402
from test_real_ft_family_metrics import _tokenizer_json  # noqa: E402
from test_real_ft_rungd_flags import REV, _only, _run, corpus  # noqa: E402, F401
from test_real_ft_score_checkpoint import _CapturedRecorder  # noqa: E402

from qd_train.ledger import Ledger  # noqa: E402
from qd_train.ood import OOD_CATEGORIES, OodCase, build_ood_suite  # noqa: E402
from qd_train.tristate import NotRun  # noqa: E402

# --- which invocations are the OOD-only mode -----------------------------------------------------


def _ns(**over: object) -> argparse.Namespace:
    base = {"score_checkpoint": Path("epoch-seed0-cpu-step2.json"), "score_plan": None,
            "score_val": False, "ood": True}
    base.update(over)
    return argparse.Namespace(**base)


def test_the_ood_only_mode_is_score_checkpoint_and_ood_without_score_val():
    assert rft._ood_only(_ns())
    assert not rft._ood_only(_ns(score_val=True)), "with --score-val it is the gate row"
    assert not rft._ood_only(_ns(ood=False)), "without --ood there is nothing to score"
    assert not rft._ood_only(_ns(score_checkpoint=None))


def test_a_score_plan_kind_never_enters_the_ood_only_mode():
    """``plan_kind_args`` keeps ``score_plan``: a kind's passes say what it decodes."""
    args = _ns(score_checkpoint=None, score_plan=Path("plan.json"), ft_row_id=None, seeds=[])
    kind = rft.PlanKind("seed0", (Path("/c/epoch-seed0-cuda.json"),), ("aaaaaaaa",), (0,),
                        ("ood",))
    assert not rft._ood_only(rft.plan_kind_args(args, kind))


def test_the_needle_worker_never_enters_the_ood_only_mode():
    """The worker is handed the scoring process's own argv, which has --score-val --needle
    (--needle needs --score-val); parsed here the way main parses it."""
    seen: list[bool] = []

    def stop(args: argparse.Namespace) -> None:
        seen.append(rft._ood_only(args))
        raise SystemExit("stopped after the argv checks")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rft, "_check_suite_logits_flags", stop)
        with pytest.raises(SystemExit, match="stopped"):
            rft.main(["--out", "/o", "--rev", "0" * 40, "--score-checkpoint",
                      "/c/epoch-seed0-cpu.json", "--ft-ledger", "/l", "--ft-row-id",
                      "abcdefgh", "--seeds", "0", "--real-backbone", "/x", "--score-val",
                      "--needle", "--ood", "--ood-general-record", "/g.json", "--devices",
                      "cpu"])
    assert seen == [False]


# --- argv ----------------------------------------------------------------------------------------

_OOD_ONLY = ["--score-checkpoint", "/c/epoch-seed0-cpu-step2.json", "--ood",
             "--ood-general-record", "/g.json", "--real-backbone", "/x", "--ft-ledger", "/l",
             "--ft-row-id", "abcdefgh", "--devices", "cpu", "--seeds", "0"]


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["--out", str(tmp_path), "--rev", "0" * 40, *extra]


def test_a_well_formed_ood_only_argv_passes_every_argv_check(tmp_path):
    """It stops at the first thing that reads the disk: resolving --rev, which no repository
    here has. Before this mode existed it was refused for want of --score-val."""
    with pytest.raises(subprocess.CalledProcessError, match="rev-parse"):
        rft.main(_argv(tmp_path, *_OOD_ONLY))


@pytest.mark.parametrize(
    ("extra", "drop", "match"),
    [
        ([], ("--ood", "--ood-general-record", "/g.json"), "--score-checkpoint needs --score-val"),
        (["--verdicts-out", "{t}/v.jsonl"], (), "there are none"),
        (["--needle"], (), "--needle scores the model the val pass scores"),
        ([], ("--real-backbone", "/x"), "needs --real-backbone"),
    ],
)
def test_ood_only_argv_is_refused_where_it_would_decode_val(tmp_path, extra, drop, match):
    argv = [a for a in _OOD_ONLY if a not in drop] + [x.format(t=tmp_path) for x in extra]
    with pytest.raises(SystemExit, match=match):
        rft.main(_argv(tmp_path, *argv))


def test_a_logit_ensemble_is_not_an_ood_only_row(tmp_path):
    argv = [a for a in _OOD_ONLY if a not in ("--ft-row-id", "abcdefgh", "--seeds", "0")]
    argv[1:2] = ["/c/epoch-seed0-cpu.json", "/c/epoch-seed1-cpu.json"]
    with pytest.raises(SystemExit, match="a logit ensemble is scored with --score-val"):
        rft.main(_argv(tmp_path, *argv, "--ft-row-id", "aaaaaaaa", "bbbbbbbb",
                       "--seeds", "0", "1"))


def test_a_score_plan_still_needs_score_val(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"kinds": [
        {"name": "seed0", "checkpoints": ["/c/epoch-seed0-cuda.json"],
         "ft_row_ids": ["aaaaaaaa"], "seeds": [0], "passes": ["ood"]},
    ]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="kind 'seed0': --score-checkpoint needs --score-val"):
        rft.main(_argv(tmp_path, "--score-plan", str(plan), "--ood", "--ood-general-record",
                       "/g.json", "--real-backbone", "/x", "--ft-ledger", "/l",
                       "--devices", "cuda"))


# --- the row --------------------------------------------------------------------------------------


def _ood_row(monkeypatch, row: rft.OodOnlyRow | None, meta: dict) -> tuple[dict, dict]:
    """``run_ood_diagnostic`` over a stubbed decode, the score-plan test's suite: within
    each category but prose three abstentions of four, every prose case agreeing."""
    cases = build_ood_suite(PROSE, per_category=4, seed=0)
    tops = [(4, 0), (1, 4), (1, PERM.index(2)), (1, PERM.index(1))]
    seen: collections.Counter[str] = collections.Counter()
    pick = {}
    for c in cases:
        pick[c.case_id] = tops[3] if c.category == "prose" else tops[seen[c.category] % 4]
        seen[c.category] += 1
    passes = [
        {"verdicts": [{**_v(c.case_id, pick[c.case_id][k]), "row_logits": [float(k)] * 5}
                      for c in cases]}
        for k in (0, 1)
    ]
    decodes: list[int] = []

    def decode(*a, **k):
        decodes.append(1)
        return passes[(len(decodes) - 1) % 2]

    monkeypatch.setattr(rft, "_decode", decode)
    val = rft.ValSet(reader=None, labels=[], plan=[], labels_for={}, letter_id={})  # type: ignore[arg-type]
    perms = {(c.case_id, "defect_class"): PERM for c in cases}
    suite = rft.OodSuite(cases, val, rft.SecondPass([], {}, perms), seed=7)
    captured: dict[str, object] = {}

    def recorder(ledger, **kw):
        captured.update(kw)
        captured["recorder"] = _CapturedRecorder(captured)
        return captured["recorder"]

    monkeypatch.setattr(rft, "_recorder", recorder)
    monkeypatch.setattr(rft, "_cost", lambda **k: None)
    reader = SimpleNamespace(header=SimpleNamespace(shard_hash=lambda: "s" * 64))
    args = SimpleNamespace(
        score_checkpoint=Path("epoch-seed0-cuda-step2.json"), score_dtype="fp32",
        usd_per_hour=2.29, usd_per_gpu_hour=None, instance="gh200", wall_clock_cap_s=600.0,
    )
    ft = {"row_id": "ft0", "metrics": {"train.termination": {"value": "steps_exhausted"}}}
    loaded = (None, ft, {"lr": 1e-5}, 0, meta)
    kw = {} if row is None else {"row": row}
    row_id, lines, seed = rft.run_ood_diagnostic(
        args, loaded=loaded, reader=reader, val=SimpleNamespace(reader=reader),  # type: ignore[arg-type]
        device="cuda", ledger=None, ood_suite=suite, suite_seed=7,  # type: ignore[arg-type]
        reasons_for=lambda tag, device, termination=None: [], **kw,
    )
    assert row_id == "control-row" and seed == 0 and lines
    return captured, {"lines": lines}


def test_the_trajectory_row_counts_under_ood_abstain_with_its_step_and_no_gate(monkeypatch):
    meta = {"sidecar": {"digest": "d" * 64}, "optimizer_step": 2}
    captured, _ = _ood_row(monkeypatch, rft.OOD_TRAJECTORY_ROW, meta)
    metrics = captured["metrics"]
    assert metrics["ood_abstain.prose"].value == 0.0  # every prose case agreed
    assert metrics["ood_abstain.scrambled"].value == 0.75
    assert metrics["ood_abstain.all"].n == sum(
        metrics[f"ood_abstain.{c}"].n for c in OOD_CATEGORIES
    )
    assert metrics["checkpoint_step"].value == 2
    assert not any(k.startswith("ood_diagnostic") for k in metrics)
    recipe = captured["recipe"]
    assert recipe["tag"] == "trajectory-ood"
    assert recipe["scored_checkpoint"] == f"epoch-seed0-cuda-step2.json:{'d' * 64}"
    assert rft.OOD_TRAJECTORY_QUICK_REASON in captured["quick_reasons"]
    assert rft.OOD_DIAGNOSTIC_QUICK_REASON not in captured["quick_reasons"]
    assert "--score-checkpoint OOD-only trajectory point of" in captured["notes"]
    assert "after optimizer step 2 " in captured["notes"]
    assert isinstance(captured["recorder"].noul_rate, NotRun)  # type: ignore[union-attr]


def test_the_plans_diagnostic_row_is_what_it_was(monkeypatch):
    """Characterization (passes before and after): the default row is the plan's diagnostic,
    tag, metric names, quick reason and notes as written on 2026-10-01; its meta need not
    carry an optimizer step."""
    captured, _ = _ood_row(monkeypatch, None, {"sidecar": {"digest": "d" * 64}})
    assert set(captured["metrics"]) == {
        "ft_run_row_id", *(f"ood_diagnostic.{c}" for c in OOD_CATEGORIES), "ood_diagnostic.all",
    }
    assert captured["recipe"]["tag"] == "epoch-ood-diagnostic"
    assert captured["quick_reasons"] == [rft.OOD_DIAGNOSTIC_QUICK_REASON]
    assert captured["notes"] == (
        f"tools/real_ft_run.py --score-plan OOD diagnostic of epoch-seed0-cuda-step2.json:"
        f"{'d' * 64} (ft row ft0), fp32 on cuda at T = 1: the OOD suite's two passes decoded "
        "the way crates/qd-runtime/src/answer.rs decodes them, the runtime rule's abstentions "
        "counted per category, and each case's letter rows (noul included) kept on its "
        "suite-verdict line. No val pass and no gate."
    )


# --- end to end on the toy corpus ----------------------------------------------------------------


def test_a_snapshot_before_the_final_step_is_scored_ood_only_end_to_end(
    corpus, tmp_path, monkeypatch  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    """Trained on the toy corpus with a snapshot at every step, then each of steps 1 and 3
    scored by ``main`` without ``--score-val``. Only the OOD suite's construction is replaced
    -- the real one needs the Qwen tokenizer and a general record -- by the toy val set's
    choice rows as cases, with the real second pass over them."""
    ck = tmp_path / "ck"
    ft = _only(
        _run(corpus, tmp_path / "ft.jsonl", "--deterministic", "--max-steps", "3",
             "--checkpoint-dir", str(ck), "--retain-tower-every", "1", real=True, short=True),
        "ft",
    )
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
    record = tmp_path / "general-record.json"
    record.write_text("{}", encoding="utf-8")
    tokenizer = _tokenizer_json(tmp_path / "tokenizer.json")
    eval_ledger = tmp_path / "eval.jsonl"
    for step in (1, 3):
        suite_out = tmp_path / f"suite-{step}.jsonl"
        code, text = _main(
            corpus, "--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
            "--real-backbone", str(corpus.snapshot), "--ft-ledger", str(tmp_path / "ft.jsonl"),
            "--ft-row-id", ft.row_id, "--tokenizer-json", str(tokenizer),
            "--score-checkpoint", str(ck / f"epoch-seed0-cpu-step{step}.json"),
            "--ood", "--ood-general-record", str(record), "--ledger", str(eval_ledger),
            "--suite-verdicts-out", str(suite_out),
        )
        assert code == 0, text
        assert "opened for the OOD suite, not decoded" in text
        lines = [json.loads(x) for x in suite_out.read_text(encoding="utf-8").splitlines()]
        assert lines and {x["gate"] for x in lines} == {rft.OOD_TRAJECTORY_GATE}
    assert main_second == [], "the val set's own second pass is never built"
    rows = [r for r in Ledger(eval_ledger).rows() if r.run_kind == "eval"]
    assert [r.metrics["checkpoint_step"].value for r in rows] == [1, 3]
    for row in rows:
        recipe = dict(row.recipe or {})
        assert recipe["tag"] == "trajectory-ood"
        assert row.quick and rft.OOD_TRAJECTORY_QUICK_REASON in (row.quick_reason or "")
        # The recorder lists every gate; a trajectory row evaluates none of them.
        assert row.gates and all(isinstance(g, NotRun) for g in row.gates.values()), row.gates
        assert row.metrics["ft_run_row_id"].value == ft.row_id
        total = row.metrics["ood_abstain.all"]
        assert total.n_total == sum(
            row.metrics[f"ood_abstain.{c}"].n_total for c in OOD_CATEGORIES
        ) > 0
    assert [str(dict(r.recipe or {})["scored_checkpoint"]).split(":")[0] for r in rows] == [
        "epoch-seed0-cpu-step1.json", "epoch-seed0-cpu-step3.json",
    ]
