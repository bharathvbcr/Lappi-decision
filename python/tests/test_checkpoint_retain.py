"""``tools/real_ft_run.py --retain-tower-every N``: the trajectory's snapshots (v5 recipe.added[0]).

What is pinned here:

* a snapshot is ``<tag>-seed<N>-<device>-step<S>.json`` beside the cell's one resume checkpoint,
  holds the tower and span head and no optimizer state, and nothing resumes from it;
* snapshots are taken after every optimizer step that is a multiple of N and after the final
  step, independently of ``--checkpoint-every``, whose resume checkpoint is written exactly as
  before -- also when the run resumed part-way;
* the flag moves no recipe key and changes nothing that trains (one deterministic run with it
  and one without share their loss log);
* ``--score-checkpoint`` reads the final snapshot as the model its ft row trained, and refuses
  an earlier one for a val/gate row;
* a run that would write over snapshots already on disk is refused before any tower loads.

Torch-gated, like ``test_real_ft_rungd_flags.py``, whose toy corpus and harness it uses.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_family_metrics import _tokenizer_json  # noqa: E402
from test_real_ft_rungd_flags import (  # noqa: E402, F401 - corpus is a fixture
    REV,
    Corpus,
    _deterministic_restored,
    _only,
    _patch_module,
    _run,
    _tiny_backbone,
    _value,
    corpus,
)

from qd_train.run_control import (  # noqa: E402
    Checkpoint,
    LossLog,
    LRSchedule,
    Position,
    TensorRef,
)

# --- names --------------------------------------------------------------------------------------


def test_a_snapshot_name_round_trips_and_sits_beside_the_resume_checkpoint():
    name = rft._snapshot_name("epoch", 0, "cpu", 20)
    assert name == "epoch-seed0-cpu-step20.json"
    assert name.startswith(rft._checkpoint_name("epoch", 0, "cpu").removesuffix(".json"))
    assert rft._snapshot_arm(Path("/d") / name) == ("epoch", 0, "cpu", 20)
    assert rft._snapshot_arm(Path("epoch-seed0-cpu.json")) is None
    assert rft._snapshot_arm(Path("epoch-seed0-cpu-step20.safetensors")) is None


@pytest.mark.parametrize("bad", ["epoch-seed0-cpu-step0.json", "epoch-seed0-cpu-stepx.json"])
def test_a_malformed_snapshot_name_is_refused(bad):
    with pytest.raises(ValueError, match="optimizer step >= 1"):
        rft._snapshot_arm(Path(bad))


@pytest.mark.parametrize("step", [0, -1, True])
def test_a_snapshot_is_named_after_a_real_optimizer_step(step):
    with pytest.raises(ValueError, match="optimizer step >= 1"):
        rft._snapshot_name("epoch", 0, "cpu", step)


def test_nothing_resumes_from_a_snapshot():
    with pytest.raises(ValueError, match="retain-tower-every snapshot"):
        rft._resume_arm(Path("epoch-seed0-cpu-step2.json"))
    assert rft._resume_arm(Path("epoch-seed0-cpu.json")) == ("epoch", 0, "cpu")


# --- the interval the loop is handed, and resume alignment --------------------------------------


@pytest.mark.parametrize(
    ("every", "retain", "interval"),
    [(0, 0, 0), (5, 0, 5), (0, 4, 4), (6, 4, 2), (3, 2, 1), (4, 4, 4)],
)
def test_the_loop_is_called_at_every_boundary_of_either(every, retain, interval):
    assert rft._checkpoint_interval(every, retain) == interval


def test_a_resume_off_the_interval_is_refused():
    assert rft.retention_resume_problem(0, checkpoint_every=6, retain_every=4) is None
    assert rft.retention_resume_problem(4, checkpoint_every=6, retain_every=4) is None
    assert rft.retention_resume_problem(7, checkpoint_every=0, retain_every=0) is None
    problem = rft.retention_resume_problem(3, checkpoint_every=6, retain_every=4)
    assert problem is not None and "not a multiple of the checkpoint interval 2" in problem


# --- the sink, driven as train_ft drives it ------------------------------------------------------


def _ref(t: torch.Tensor) -> TensorRef:
    """A tensor as ``QwenDecisionStep.state`` hands it to a checkpoint."""
    return TensorRef(
        dtype=str(t.dtype).removeprefix("torch."), shape=tuple(t.shape),
        data=t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(),
    )


def _full_ckpt(step: int, *, total: int) -> Checkpoint:
    """What ``train_ft`` hands ``on_checkpoint``: the step's whole state, optimizer included."""
    return Checkpoint(
        position=Position(epoch=0, index=step), optimizer_step=step, seed=0,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=total, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog().snapshot(), consumed_digest=f"{step + 1:064x}",
        model_state={
            "tower": {"w": _ref(torch.full((2, 3), float(step)))},
            "span_head": {"b": _ref(torch.full((3,), -float(step)))},
            "span_weight": 1.0, "vocab_size": 256,
            "optimizer": {"m": _ref(torch.ones(2, 3))},
        },
    )


def _drive(sink: rft.RetainingSink, *, first: int, total: int, interval: int) -> None:
    """``train_ft``'s calls: at ``should_checkpoint(step - first)``, then ``final``."""
    for step in range(first + 1, total + 1):
        if (step - first) % interval == 0:
            sink(_full_ckpt(step, total=total))
    sink.final(_full_ckpt(total, total=total))


def _on_disk(directory: Path) -> tuple[list[int], int | None]:
    snaps = sorted(rft.existing_snapshots(directory, "epoch", 0, "cpu"))
    resume = directory / rft._checkpoint_name("epoch", 0, "cpu")
    return snaps, (Checkpoint.read_body(resume)["optimizer_step"] if resume.exists() else None)


def test_snapshots_and_the_resume_checkpoint_are_independent(tmp_path, capsys):
    full = rft.CheckpointSink(tmp_path / rft._checkpoint_name("epoch", 0, "cpu"))
    sink = rft.RetainingSink(
        directory=tmp_path, tag="epoch", seed=0, device="cpu", retain_every=4, full=full,
        full_every=6, first_step=0,
    )
    _drive(sink, first=0, total=13, interval=rft._checkpoint_interval(6, 4))
    out = capsys.readouterr().out
    # Resume checkpoint: steps 6 and 12 (every 6), then 13 at the end -- what it always did.
    assert re.findall(r"checkpoint: step (\d+)", out) == ["6", "12", "13"]
    # Snapshots: 4, 8, 12 (every 4) and 13 (the final step), each once.
    assert re.findall(r"snapshot: step (\d+)", out) == ["4", "8", "12", "13"]
    assert _on_disk(tmp_path) == ([4, 8, 12, 13], 13)


def test_a_resumed_run_counts_resume_checkpoints_from_where_it_resumed(tmp_path, capsys):
    full = rft.CheckpointSink(tmp_path / rft._checkpoint_name("epoch", 0, "cpu"))
    sink = rft.RetainingSink(
        directory=tmp_path, tag="epoch", seed=0, device="cpu", retain_every=4, full=full,
        full_every=6, first_step=4,
    )
    assert rft.retention_resume_problem(4, checkpoint_every=6, retain_every=4) is None
    _drive(sink, first=4, total=13, interval=2)
    out = capsys.readouterr().out
    # RunControl.should_checkpoint(step - first): 10 is the resumed run's first boundary.
    assert re.findall(r"checkpoint: step (\d+)", out) == ["10", "13"]
    assert re.findall(r"snapshot: step (\d+)", out) == ["8", "12", "13"]


def test_retention_alone_writes_no_resume_checkpoint(tmp_path, capsys):
    sink = rft.RetainingSink(
        directory=tmp_path, tag="epoch", seed=0, device="cpu", retain_every=3, full=None,
        full_every=0, first_step=0,
    )
    _drive(sink, first=0, total=6, interval=3)
    assert re.findall(r"snapshot: step (\d+)", capsys.readouterr().out) == ["3", "6"]
    assert _on_disk(tmp_path) == ([3, 6], None)


def test_a_snapshot_is_the_tower_and_span_head_and_reads_as_weights(tmp_path):
    sink = rft.RetainingSink(
        directory=tmp_path, tag="epoch", seed=0, device="cpu", retain_every=2, full=None,
        full_every=0, first_step=0,
    )
    sink(_full_ckpt(2, total=4))
    path = tmp_path / "epoch-seed0-cpu-step2.json"
    body = Checkpoint.read_body(path)
    assert set(body["model_state"]) == set(rft.SNAPSHOT_KEYS)
    assert body["optimizer_step"] == 2 and body["schedule"]["total_steps"] == 4
    weights, meta = Checkpoint.read_weights(path, subtrees=("tower", "span_head"))
    assert meta["optimizer_step"] == 2
    assert weights["tower"]["w"] == _ref(torch.full((2, 3), 2.0))
    assert weights["span_head"]["b"] == _ref(torch.full((3,), -2.0))
    with pytest.raises(ValueError, match="optimizer"):
        Checkpoint.read_weights(path, subtrees=("optimizer",))


def test_the_resume_checkpoint_is_priced_without_the_snapshots_beside_it(tmp_path, capsys):
    """``CheckpointSink`` summed ``<stem>*``, which under retention also matches every
    ``<stem>-step<S>`` snapshot: a 2 GiB (sparse) one here, priced as the resume checkpoint's."""
    snapshot = tmp_path / "epoch-seed0-cpu-step2.json"
    with snapshot.open("wb") as fh:
        fh.truncate(2 << 30)
    rft.CheckpointSink(tmp_path / "epoch-seed0-cpu.json")(_full_ckpt(3, total=3))
    assert "(0.00 GiB in" in capsys.readouterr().out


def test_a_snapshot_already_on_disk_past_the_start_is_a_clash(tmp_path):
    (tmp_path / "epoch-seed0-cpu-step4.json").write_text("{}", encoding="utf-8")
    (tmp_path / "epoch-seed1-cpu-step8.json").write_text("{}", encoding="utf-8")
    assert rft.snapshot_clash(tmp_path, "epoch", 0, "cpu", first_step=4) is None
    clash = rft.snapshot_clash(tmp_path, "epoch", 0, "cpu", first_step=0)
    assert clash is not None and "1 epoch seed 0 cpu snapshot(s) past step 0" in clash
    assert rft.snapshot_clash(tmp_path / "absent", "epoch", 0, "cpu", first_step=0) is None


# --- argv ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--retain-tower-every", "-1"], "must not be negative"),
        (["--retain-tower-every", "2", "--checkpoint-dir", "{d}"], "need --real-backbone"),
        (["--real-backbone", "/x", "--retain-tower-every", "2"], "nowhere to keep"),
        (["--real-backbone", "/x", "--retain-tower-every", "2", "--checkpoint-dir", "{d}",
          "--score-checkpoint", "/c/epoch-seed0-cpu.json"], "train nothing"),
        (["--real-backbone", "/x", "--shuffled-label", "abcdefgh", "--epoch", "--no-memorise",
          "--score-val", "--checkpoint-dir", "{d}", "--retain-tower-every", "2", "--seeds", "0",
          "--devices", "cpu"], "--retain-tower-every would write"),
        (["--real-backbone", "/x", "--resume-from", "{snap}", "--seeds", "0", "--epoch"],
         "retain-tower-every snapshot"),
    ],
)
def test_retention_argv_is_refused_before_anything_loads(tmp_path, extra, match):
    snap = tmp_path / "epoch-seed0-cpu-step2.json"
    snap.write_text("{}", encoding="utf-8")
    argv = ["--out", str(tmp_path), "--rev", "0" * 40] + [
        a.format(d=tmp_path / "ck", snap=snap) for a in extra
    ]
    if "--seeds" not in argv:
        argv += ["--seeds", "0"]
    with pytest.raises(SystemExit, match=match):
        rft.main(argv)


# --- end to end on the toy corpus ----------------------------------------------------------------


def _main(toy: Corpus, *argv: str) -> tuple[int, str]:
    """``main`` under the rung (d) harness's patches, its exit code and stdout."""
    out = io.StringIO()
    with (
        pytest.MonkeyPatch.context() as mp,
        _deterministic_restored(),
        contextlib.redirect_stdout(out),
    ):
        _patch_module(mp, rft, toy)
        _tiny_backbone(mp)
        code = rft.main(list(argv))
    return code, out.getvalue()


def _rows(ledger: Path) -> list[object]:
    """The rows a run appended to ``ledger``: none when it never created the file."""
    return list(rft.Ledger(ledger).rows()) if ledger.exists() else []


_TRAIN = ("--deterministic", "--max-steps", "3")


def test_retention_keeps_the_trajectory_and_changes_nothing_that_trains(
    corpus, tmp_path, capsys  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    plain = _only(_run(corpus, tmp_path / "plain.jsonl", *_TRAIN, real=True, short=True), "ft")
    ck = tmp_path / "ck"
    kept = _only(
        _run(corpus, tmp_path / "kept.jsonl", *_TRAIN, "--checkpoint-dir", str(ck),
             "--checkpoint-every", "3", "--retain-tower-every", "2", real=True, short=True),
        "ft",
    )
    out = capsys.readouterr().out
    assert re.findall(r"snapshot: step (\d+)", out) == ["2", "3"]
    assert re.findall(r"checkpoint: step (\d+)", out) == ["3"]
    assert _on_disk(ck) == ([2, 3], 3)
    # Kept, not trained: no recipe key, the same hash and the same loss log.
    assert kept.recipe == plain.recipe
    assert kept.protocol.recipe_hash == plain.protocol.recipe_hash
    assert _value(kept, "train.loss_log_digest") == _value(plain, "train.loss_log_digest")
    assert _value(kept, "train.retain_tower_every") == 2
    assert "train.retain_tower_every" not in plain.metrics
    # The final snapshot is the resume checkpoint's tower and span head, bit for bit.
    snap, _ = Checkpoint.read_weights(ck / "epoch-seed0-cpu-step3.json",
                                      subtrees=("tower", "span_head"))
    full, _ = Checkpoint.read_weights(ck / "epoch-seed0-cpu.json",
                                      subtrees=("tower", "span_head"))
    for part in ("tower", "span_head"):
        assert snap[part].keys() == full[part].keys()
        for key in snap[part]:
            assert snap[part][key] == full[part][key], (part, key)
    early, _ = Checkpoint.read_weights(ck / "epoch-seed0-cpu-step2.json", subtrees=("tower",))
    assert any(early["tower"][k] != full["tower"][k] for k in full["tower"])

    # Scored: the final snapshot is the model the ft row trained.
    common = (
        "--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
        "--score-val", "--real-backbone", str(corpus.snapshot),
        "--ft-ledger", str(tmp_path / "kept.jsonl"), "--ft-row-id", kept.row_id,
        "--tokenizer-json", str(_tokenizer_json(tmp_path / "tokenizer.json")),
    )
    code, text = _main(corpus, *common, "--ledger", str(tmp_path / "eval.jsonl"),
                       "--score-checkpoint", str(ck / "epoch-seed0-cpu-step3.json"))
    assert code == 0, text
    (row,) = [r for r in rft.Ledger(tmp_path / "eval.jsonl").rows() if r.run_kind == "eval"]
    assert str((row.recipe or {})["scored_checkpoint"]).startswith("epoch-seed0-cpu-step3.json:")
    # An earlier snapshot is not that model: no val or gate row from it.
    with pytest.raises(SystemExit, match="not the model that row trained"):
        _main(corpus, *common, "--ledger", str(tmp_path / "eval2.jsonl"),
              "--score-checkpoint", str(ck / "epoch-seed0-cpu-step2.json"))
    assert not _rows(tmp_path / "eval2.jsonl")


def test_retention_alone_needs_no_checkpoint_every_and_will_not_overwrite(
    corpus, tmp_path  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    ck = tmp_path / "ck"
    _run(corpus, tmp_path / "a.jsonl", *_TRAIN, "--checkpoint-dir", str(ck),
         "--retain-tower-every", "2", real=True, short=True)
    assert _on_disk(ck) == ([2, 3], None)
    stamps = {p.name: p.stat().st_mtime_ns for p in ck.iterdir()}
    with pytest.raises(SystemExit, match="already holds 2 epoch seed 0 cpu snapshot"):
        _main(corpus, "--out", str(corpus.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
              "--epoch", "--no-memorise", "--ledger", str(tmp_path / "b.jsonl"),
              "--real-backbone", str(corpus.snapshot), *_TRAIN, "--checkpoint-dir", str(ck),
              "--retain-tower-every", "2")
    assert {p.name: p.stat().st_mtime_ns for p in ck.iterdir()} == stamps
    assert not _rows(tmp_path / "b.jsonl")
