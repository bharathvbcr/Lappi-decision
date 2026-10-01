"""``tools/real_ft_run.py --score-plan PLAN.json``: several models in one invocation.

Fable I diagnostic (2026-10-01): seed 0, 1 and 2, their average and their ensemble on the OOD
suite, plus the ensemble's full gate row. Five ``--score-checkpoint`` runs rebuild the same
shard set, val set and suites five times; a plan builds them once and runs one kind after
another.

What is pinned here:

* the ORACLE: each kind of a plan writes the row and the verdict lines its own
  ``--score-checkpoint`` run writes -- recipe, protocol, metrics, gates, quick reason and
  every verdict -- for a seed, an average and an ensemble, end to end through ``main()`` on
  three tiny ``--optimizer master`` checkpoints; only the notes (the plan's sentence) and the
  lines' ``score_kind`` differ;
* the prelude is built once (``ft_split_rows``, ``open_val_set``), and each kind's model is
  unreachable before the next one loads;
* the OOD-only diagnostic row decodes the suite by the gate's own rule and counts what the
  gate's report counts, under ``ood_diagnostic.*``, quick, with no gate;
* every refusal of the plan file and of the argv around it, before anything loads.
"""

from __future__ import annotations

import collections
import gc
import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
import real_ft_run as rft  # noqa: E402
from test_ood_abstain import PERM, PROSE, _v  # noqa: E402
from test_real_ft_score_checkpoint import (  # noqa: E402
    _CapturedRecorder,
    tiny_master_checkpoints,
)

from qd_train.ledger import Ledger  # noqa: E402
from qd_train.ood import build_ood_suite  # noqa: E402
from qd_train.tristate import NotRun  # noqa: E402

# --- the plan file -----------------------------------------------------------------------------


def _write_plan(tmp_path: Path, kinds: object, name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(kinds if isinstance(kinds, dict) else {"kinds": kinds}),
                    encoding="utf-8")
    return path


SEED0 = {"name": "seed0", "checkpoints": ["/c/epoch-seed0-cuda.json"],
         "ft_row_ids": ["aaaaaaaa"], "seeds": [0], "passes": ["ood"]}
AVG = {"name": "avg", "checkpoints": ["/c/avg.safetensors"], "seeds": [0, 1, 2],
       "passes": ["ood"]}
ENS3 = {"name": "ens3", "checkpoints": [f"/c/epoch-seed{s}-cuda.json" for s in (0, 1, 2)],
        "ft_row_ids": ["aaaaaaaa", "bbbbbbbb", "cccccccc"], "seeds": [0, 1, 2],
        "passes": ["gates"]}


def test_a_plan_is_read_into_its_kinds_in_order(tmp_path):
    path = _write_plan(tmp_path, [SEED0, AVG, ENS3])
    plan = rft.read_score_plan(path)
    assert [k.name for k in plan.kinds] == ["seed0", "avg", "ens3"]
    assert plan.kinds[1].ft_row_ids is None and plan.kinds[2].seeds == (0, 1, 2)
    assert plan.kinds[2].checkpoints[1] == Path("/c/epoch-seed1-cuda.json")
    assert plan.wants("ood") and plan.wants("gates")
    import hashlib

    assert plan.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rft.plan_output(Path("/o/suite.jsonl"), "ens3") == Path("/o/suite-ens3.jsonl")


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"kinds": [SEED0], "note": "x"}, "one key, 'kinds'"),
        ({"kinds": []}, "list of 1 to"),
        ({"kinds": [dict(SEED0, name=f"k{i}") for i in range(rft.PLAN_MAX_KINDS + 1)]},
         "list of 1 to"),
        ({"kinds": ["seed0"]}, "a kind is an object"),
        ({"kinds": [dict(SEED0, extra=1)]}, "unknown keys \\['extra'\\]"),
        ({"kinds": [{k: v for k, v in SEED0.items() if k != "passes"}]},
         "missing keys \\['passes'\\]"),
        ({"kinds": [dict(SEED0, name="Seed 0")]}, "is not"),
        ({"kinds": [dict(SEED0, name="../x")]}, "is not"),
        ({"kinds": [SEED0, SEED0]}, "repeat"),
        ({"kinds": [dict(SEED0, passes=[])]}, "non-empty list of distinct"),
        ({"kinds": [dict(SEED0, passes=["ood", "ood"])]}, "non-empty list of distinct"),
        ({"kinds": [dict(SEED0, passes=["val"])]}, "non-empty list of distinct"),
        ({"kinds": [dict(SEED0, passes=["gates", "ood"])]}, "decode it twice"),
        ({"kinds": [dict(SEED0, seeds=[True])]}, "integers"),
        ({"kinds": [dict(SEED0, seeds="0")]}, "integers"),
        ({"kinds": [dict(SEED0, checkpoints=[])]}, "checkpoints is a list"),
        ({"kinds": [dict(SEED0, checkpoints=[""])]}, "checkpoints is a list"),
        ({"kinds": [dict(SEED0, ft_row_ids="aaaaaaaa")]}, "ft_row_ids is a list"),
    ],
)
def test_a_plan_that_says_anything_it_does_not_define_is_refused(tmp_path, body, match):
    with pytest.raises(SystemExit, match=match):
        rft.read_score_plan(_write_plan(tmp_path, body))


def test_a_plan_file_that_is_not_a_plan_is_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"bad\.json"):
        rft.read_score_plan(bad)
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (rft.PLAN_MAX_BYTES + 1))
    with pytest.raises(SystemExit, match="more than a plan's"):
        rft.read_score_plan(big)
    with pytest.raises(SystemExit, match=r"missing\.json"):
        rft.read_score_plan(tmp_path / "missing.json")


# --- argv around a plan, refused before anything loads --------------------------------------


def _argv(tmp_path: Path, plan: Path, *extra: str, device: str = "cuda") -> list[str]:
    return ["--out", str(tmp_path), "--rev", "0" * 40, "--score-val", "--real-backbone", "/x",
            "--ft-ledger", "/l", "--devices", device, "--instance", "gh200",
            "--usd-per-hour", "2.29", "--score-plan", str(plan), *extra]


@pytest.mark.parametrize(
    ("kinds", "extra", "match"),
    [
        ([SEED0], ["--score-checkpoint", "/c/x.json"], "--score-checkpoint"),
        ([SEED0], ["--ft-row-id", "aaaaaaaa"], "--ft-row-id"),
        ([SEED0], ["--seeds", "0", "1", "2"], "--seeds"),
        ([SEED0], ["--seed", "0"], "--seeds"),
        ([ENS3], ["--needle", "--needle-control", "1024"], "--needle-control"),
        ([ENS3], ["--epoch"], "--epoch"),
        ([SEED0], ["--ood", "--ood-general-record", "/g.json", "--devices", "cuda", "cpu"],
         "one device"),
        ([ENS3], ["--needle"], "ran out of memory"),
        ([SEED0], ["--needle", "--ood", "--ood-general-record", "/g.json"],
         "--needle is a gate"),
        ([dict(ENS3, passes=["gates"])], [], None),
        ([SEED0], [], "it needs --ood"),
        ([SEED0], ["--ood", "--ood-general-record", "/g.json", "--verdicts-out", "v.jsonl"],
         "no kind of this plan decodes val"),
        ([dict(ENS3, ft_row_ids=["aaaaaaaa"])], [], "kind 'ens3': a logit ensemble of 3"),
        ([dict(AVG, seeds=[0])], ["--ood", "--ood-general-record", "/g.json"],
         "kind 'avg': --score-checkpoint of an average needs --seeds"),
        ([dict(AVG, ft_row_ids=["aaaaaaaa"])], ["--ood", "--ood-general-record", "/g.json"],
         "kind 'avg': --score-checkpoint of an average"),
    ],
)
def test_plan_argv_is_refused_before_anything_loads(tmp_path, kinds, extra, match):
    plan = _write_plan(tmp_path, kinds)
    device = "mps" if "--needle" in extra and match == "ran out of memory" else "cuda"
    extra = [str(tmp_path / x) if x.endswith(".jsonl") else x for x in extra]
    if match is None:
        # A well-formed plan passes every argv check and stops at the first thing that
        # reads the disk: resolving --rev, which no repository here has.
        with pytest.raises(subprocess.CalledProcessError, match="rev-parse"):
            rft.main(_argv(tmp_path, plan, *extra, device=device))
        return
    with pytest.raises(SystemExit, match=match):
        rft.main(_argv(tmp_path, plan, *extra, device=device))


def test_a_kinds_output_file_that_exists_is_refused_before_anything_loads(tmp_path):
    plan = _write_plan(tmp_path, [SEED0, AVG])
    (tmp_path / "suite-avg.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"kind 'avg'.*suite-avg\.jsonl already exists"):
        rft.main(_argv(tmp_path, plan, "--ood", "--ood-general-record", "/g.json",
                       "--suite-verdicts-out", str(tmp_path / "suite.jsonl")))


def test_suite_logits_with_the_needle_is_kept_under_a_plan(tmp_path):
    """A plan decodes the needle in-process (cuda), so --suite-logits is not dropped there;
    the single --score-checkpoint gate row still refuses it (its worker drops it)."""
    plan = _write_plan(tmp_path, [ENS3])
    with pytest.raises(subprocess.CalledProcessError, match="rev-parse"):
        rft.main(_argv(tmp_path, plan, "--needle", "--suite-logits",
                       "--suite-verdicts-out", str(tmp_path / "s.jsonl")))


# --- the OOD-only diagnostic row ---------------------------------------------------------------


def test_the_ood_diagnostic_counts_what_the_gates_report_counts_and_is_no_gate(monkeypatch):
    cases = build_ood_suite(PROSE, per_category=4, seed=0)
    # Within each category but prose: noul in pass 1, noul in pass 2, a moved winner, and
    # agreement -- three abstentions of four. Every prose case agrees.
    tops = [(4, 0), (1, 4), (1, PERM.index(2)), (1, PERM.index(1))]
    seen: collections.Counter[str] = collections.Counter()
    pick = {}
    for c in cases:
        pick[c.case_id] = tops[3] if c.category == "prose" else tops[seen[c.category] % 4]
        seen[c.category] += 1
    first = {"verdicts": [
        {**_v(c.case_id, pick[c.case_id][0]), "row_logits": [0.0] * 5} for c in cases
    ]}
    second = {"verdicts": [
        {**_v(c.case_id, pick[c.case_id][1]), "row_logits": [1.0] * 5} for c in cases
    ]}
    decodes: list[dict] = []

    def decode(*a, **k):
        out = (first, second)[len(decodes) % 2]
        decodes.append(out)
        return out

    monkeypatch.setattr(rft, "_decode", decode)
    val = rft.ValSet(reader=None, labels=[], plan=[], labels_for={}, letter_id={})  # type: ignore[arg-type]
    perms = {(c.case_id, "defect_class"): PERM for c in cases}
    suite = rft.OodSuite(cases, val, rft.SecondPass([], {}, perms), seed=7)

    # The gate's own decode of the same two passes.
    in_perms = {("v0", "defect_class"): PERM}
    _, gate_metrics, gate_lines = rft.score_ood(
        None, suite, scored={"verdicts": [_v("v0", 1)]},  # type: ignore[arg-type]
        val_second={"verdicts": [_v("v0", PERM.index(1))]},
        val_second_pass=rft.SecondPass([], {}, in_perms),
    )

    captured: dict[str, object] = {}

    def recorder(ledger, **kw):
        captured.update(kw)
        captured["recorder"] = _CapturedRecorder(captured)
        return captured["recorder"]

    monkeypatch.setattr(rft, "_recorder", recorder)
    monkeypatch.setattr(rft, "_cost", lambda **k: None)
    header = SimpleNamespace(shard_hash=lambda: "s" * 64)
    reader = SimpleNamespace(header=header)
    args = SimpleNamespace(
        score_checkpoint=Path("epoch-seed0-cuda.json"), score_dtype="fp32", usd_per_hour=2.29,
        usd_per_gpu_hour=None, instance="gh200", wall_clock_cap_s=600.0,
    )
    ft = {"row_id": "ft0", "metrics": {"train.termination": {"value": "steps_exhausted"}}}
    loaded = (None, ft, {"lr": 1e-5}, 0, {"sidecar": {"digest": "d" * 64}})
    row_id, lines, seed = rft.run_ood_diagnostic(
        args, loaded=loaded, reader=reader, val=SimpleNamespace(reader=reader),  # type: ignore[arg-type]
        device="cuda", ledger=None, ood_suite=suite, suite_seed=7,  # type: ignore[arg-type]
        reasons_for=lambda tag, device, termination=None: [],
        plan_note="Scored as kind 'seed0'.",
    )
    assert row_id == "control-row" and seed == 0
    assert lines == gate_lines, "the same per-case lines the gate row keeps"
    metrics = captured["metrics"]
    for category in ("prose", "unseen-language", "scrambled"):
        mine, gates = metrics[f"ood_diagnostic.{category}"], gate_metrics[f"ood_abstain.{category}"]
        assert (mine.value, mine.n, mine.n_total) == (gates.value, gates.n, gates.n_total)
    assert metrics["ood_diagnostic.prose"].value == 0.0  # every prose case agreed
    assert metrics["ood_diagnostic.scrambled"].value == 0.75
    assert metrics["ood_diagnostic.all"].n == sum(
        gate_metrics[f"ood_abstain.{c}"].n for c in ("prose", "unseen-language", "scrambled")
    )
    assert not any(k.startswith("ood_abstain") for k in metrics), "a diagnostic is no gate"
    assert metrics["ft_run_row_id"].value == "ft0"
    recipe = captured["recipe"]
    assert recipe["tag"] == "epoch-ood-diagnostic" and recipe["ood"] == rft.ood_recipe(suite)
    assert recipe["scored_checkpoint"] == f"epoch-seed0-cuda.json:{'d' * 64}"
    assert rft.OOD_DIAGNOSTIC_QUICK_REASON in captured["quick_reasons"]
    assert captured["run_kind"] == "eval" and "Scored as kind 'seed0'." in captured["notes"]
    assert isinstance(captured["recorder"].noul_rate, NotRun)  # type: ignore[union-attr]


def test_an_ood_pass_without_its_second_pass_refuses_before_any_model_loads(monkeypatch):
    cases = build_ood_suite(PROSE, per_category=1, seed=0)
    val = rft.ValSet(reader=None, labels=[], plan=[], labels_for={}, letter_id={})  # type: ignore[arg-type]
    suite = rft.OodSuite(cases, val, rft.SecondPass([], {}, {}, not_run="no tokenizer"))

    def never(*a, **k):
        raise AssertionError("a model was loaded for a pass that cannot run")

    monkeypatch.setattr(rft, "_checkpoint_step", never)
    plan = rft.ScorePlan(path=Path("p.json"), sha256="0" * 64, kinds=(
        rft.PlanKind("seed0", (Path("/c/epoch-seed0-cuda.json"),), ("aaaaaaaa",), (0,),
                     ("ood",)),
    ))
    with pytest.raises(SystemExit, match="no tokenizer"):
        rft.run_score_plan(
            SimpleNamespace(), plan, reader=None, val=None, device="cuda",  # type: ignore[arg-type]
            ledger=None, reasons_for=lambda *a: [], second_pass=None,  # type: ignore[arg-type]
            needle_suite=rft.NeedleSuite([], [], {}, [], not_run="x"), ood_suite=suite,
            suite_seed=0,
        )


def test_each_kinds_files_are_written_before_the_next_kind_loads(tmp_path, monkeypatch):
    """A kind's result survives a later kind's failure (an out-of-memory on the ensemble is
    the one this was written for): its row is in the ledger and its files on disk before the
    next model loads. The gate pass gets the loaded model and no worker's predictions, so the
    needle is decoded in this process."""
    plan = rft.ScorePlan(path=Path("p.json"), sha256="0" * 64, kinds=(
        rft.PlanKind("seed0", (Path("/c/epoch-seed0-cuda.json"),), ("aaaaaaaa",), (0,),
                     ("gates",)),
        rft.PlanKind("ens3", tuple(Path(f"/c/epoch-seed{s}-cuda.json") for s in (0, 1, 2)),
                     ("aaaaaaaa", "bbbbbbbb", "cccccccc"), (0, 1, 2), ("gates",)),
    ))
    verdicts, suite = tmp_path / "v.jsonl", tmp_path / "s.jsonl"
    loads: list[str] = []

    def load(args, **k):
        for done in loads:
            assert rft.plan_output(verdicts, done).is_file()
            assert rft.plan_output(suite, done).is_file()
        loads.append("ens3" if isinstance(args.score_checkpoint, list) else "seed0")
        return ("model", loads[-1])

    calls: list[dict] = []

    def score(args, **k):
        calls.append({"loaded": k["loaded"], "needle_decoded": k.get("needle_decoded"),
                      "note": k["plan_note"], "seeds": args.seeds})
        verdict = {"row_id": "r", "kind": "choice", "correct": True, "top": 0}
        gate = rft.SuiteGate("needle_hunk_recall", NotRun(reason="x"), {}, "needle", None,
                             ({"suite": "needle", "case_id": "c0", "start_logits": [0.0]},))
        return f"row-{len(calls)}", {"verdicts": [verdict]}, None, [gate], 0

    monkeypatch.setattr(rft, "_checkpoint_step", load)
    monkeypatch.setattr(rft, "_score_checkpoint", score)
    monkeypatch.setattr(rft, "release_device_cache", lambda: None)
    args = SimpleNamespace(verdicts_out=verdicts, suite_verdicts_out=suite, score_plan=plan.path,
                           score_checkpoint=None, ft_row_id=None, seeds=[0, 1, 2])
    recorded = rft.run_score_plan(
        args, plan, reader=None, val=None, device="cuda", ledger=None,  # type: ignore[arg-type]
        reasons_for=lambda *a: [], second_pass=None,  # type: ignore[arg-type]
        needle_suite=rft.NeedleSuite([], [], {}, [], not_run="x"),
        ood_suite=rft.OodSuite([], None, None, not_run="x"), suite_seed=0,
    )
    assert recorded == [("seed0", "gates", "row-1"), ("ens3", "gates", "row-2")]
    assert [c["loaded"] for c in calls] == [("model", "seed0"), ("model", "ens3")]
    assert all(c["needle_decoded"] is None for c in calls), "decoded here, not by a worker"
    assert [c["seeds"] for c in calls] == [[0], [0, 1, 2]]
    assert "kind 'ens3' (gates)" in calls[1]["note"]
    (line,) = _verdicts(rft.plan_output(suite, "ens3"))
    assert line["score_kind"] == "ens3" and line["eval_row_id"] == "row-2"
    assert line["gate"] == "needle_hunk_recall" and line["start_logits"] == [0.0]


# --- end to end: each kind is its own single run ----------------------------------------------

_VOLATILE = ("row_id", "written_at", "prev_row_hash", "wall_clock_s", "cost_usd", "notes")


def _row_body(row) -> dict:
    return {k: v for k, v in row.to_json().items() if k not in _VOLATILE}


def _verdicts(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_each_kind_of_a_plan_is_the_row_its_single_run_writes(tmp_path, monkeypatch):
    from test_real_ft_shuffled_label import REV

    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    avg = tmp_path / "avg" / "avg.safetensors"
    assert ckpt_average.main([*map(str, tiny.paths), "--out", str(avg), "--from", "masters",
                              "--ft-row-ids", *tiny.ids]) == 0
    base = ["--out", str(tiny.out), "--rev", REV, "--devices", "cpu", "--score-val",
            "--real-backbone", str(tiny.snapshot), "--ft-ledger", str(tiny.ft_ledger),
            "--tokenizer-json", str(tiny.tokenizer)]
    kinds = {
        "seed1": {"checkpoints": [str(tiny.paths[1])], "ft_row_ids": [tiny.ids[1]],
                  "seeds": [1]},
        "avg": {"checkpoints": [str(avg)], "seeds": [0, 1, 2]},
        "ens3": {"checkpoints": list(map(str, tiny.paths)), "ft_row_ids": list(tiny.ids),
                 "seeds": [0, 1, 2]},
    }
    singles = {}
    for name, kind in kinds.items():
        ledger, verdicts = tmp_path / f"eval-{name}.jsonl", tmp_path / f"verdicts-{name}.jsonl"
        assert rft.main([
            *base, "--ledger", str(ledger), "--verdicts-out", str(verdicts),
            "--seeds", *map(str, kind["seeds"]), "--score-checkpoint", *kind["checkpoints"],
            *(["--ft-row-id", *kind["ft_row_ids"]] if "ft_row_ids" in kind else []),
        ]) == 0
        (row,) = Ledger(ledger).rows()
        singles[name] = (row, _verdicts(verdicts))

    counts: collections.Counter[str] = collections.Counter()
    real_split, real_open, real_load = rft.ft_split_rows, rft.open_val_set, rft._checkpoint_step

    def split(*a, **k):
        counts["ft_split_rows"] += 1
        return real_split(*a, **k)

    def open_val(*a, **k):
        counts["open_val_set"] += 1
        return real_open(*a, **k)

    held: list[weakref.ref] = []

    def load(*a, **k):
        gc.collect()
        alive = [r for r in held if r() is not None]
        assert not alive, f"{len(alive)} of the previous kind's steps are still reachable"
        loaded = real_load(*a, **k)
        held.extend(weakref.ref(s) for s in (loaded[0], *getattr(loaded[0], "steps", ())))
        counts["models"] += 1
        return loaded

    monkeypatch.setattr(rft, "ft_split_rows", split)
    monkeypatch.setattr(rft, "open_val_set", open_val)
    monkeypatch.setattr(rft, "_checkpoint_step", load)
    plan = _write_plan(tmp_path, [
        {"name": name, **kind, "passes": ["gates"]} for name, kind in kinds.items()
    ])
    ledger, verdicts = tmp_path / "eval-plan.jsonl", tmp_path / "verdicts-plan.jsonl"
    assert rft.main([*base, "--ledger", str(ledger), "--verdicts-out", str(verdicts),
                     "--score-plan", str(plan)]) == 0
    assert counts == {"ft_split_rows": 1, "open_val_set": 1, "models": 3}

    rows = Ledger(ledger).rows()
    assert [dict(r.recipe or {})["tag"] for r in rows] == [
        "epoch-score-val", "avg-score-val", "ens3-score-val"
    ]
    sha = rft.read_score_plan(plan).sha256
    for row, name in zip(rows, kinds, strict=True):
        single, single_lines = singles[name]
        assert _row_body(row) == _row_body(single), name
        assert row.notes.startswith(single.notes) and f"kind '{name}'" in row.notes
        assert sha in row.notes and sha not in json.dumps(dict(row.recipe or {}))
        lines = _verdicts(rft.plan_output(verdicts, name))
        assert {(x["eval_row_id"], x["score_kind"]) for x in lines} == {(row.row_id, name)}

        def strip(xs):
            return [{k: v for k, v in x.items() if k not in ("eval_row_id", "score_kind")}
                    for x in xs]

        assert strip(lines) == strip(single_lines), name
    assert not verdicts.exists(), "a plan writes one file per kind, never the stem itself"
