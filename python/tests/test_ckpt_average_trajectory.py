"""``tools/ckpt_average.py --same-seed-trajectory``: one run's snapshots at distinct steps.

v5's trajectory plan (``campaign/v5-preregistered.DRAFT.json``, recipe.added[0]) asks for one
last-3 average per seed "iff tools/ckpt_average.py --from tower gains a same-seed
unequal-step allowance". What is pinned here:

* without the flag an average across steps is refused, as it always was;
* with it, the inputs must be one run: one seed, steps strictly ascending, one schedule,
  scalars and trees, no repeated weights, and each loss log the start of the next; the
  manifest records the steps in a ``trajectory`` block and the latest as its step;
* ``read_manifest`` -- and so ``real_ft_run.py --score-checkpoint`` -- refuses such an
  average by name: the scoring half is not written (``TRAJECTORY_NOT_SCORABLE``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("safetensors", reason="safetensors writes the average")
pytest.importorskip("transformers", reason="the toy run's tower is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
from test_checkpoint_retain import _TRAIN  # noqa: E402
from test_real_ft_rungd_flags import _only, _run, corpus  # noqa: E402, F401

from qd_train.run_control import (  # noqa: E402
    Checkpoint,
    LossLog,
    LossPoint,
    LRSchedule,
    Position,
    TensorRef,
)

ROW = "e2e00000-0000-4000-8000-000000000000"
LOSSES = [3.0, 2.5, 2.25, 2.0, 1.75, 1.5]


def _ref(t: torch.Tensor) -> TensorRef:
    return TensorRef(
        dtype=str(t.dtype).removeprefix("torch."), shape=tuple(t.shape),
        data=t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(),
    )


def _snapshot(
    tmp_path: Path, step: int, *, seed: int = 0, losses: list[float] = LOSSES,
    total: int = 6, value: float | None = None, span_weight: float = 1.0,
) -> Path:
    """A ``--retain-tower-every`` snapshot's shape: tower and span head, no optimizer."""
    w = float(step) if value is None else value
    path = tmp_path / f"epoch-seed{seed}-cpu-step{step}.json"
    Checkpoint(
        position=Position(epoch=0, index=step), optimizer_step=step, seed=seed,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=total, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog(
            # Logged at the step it ran as, 0-based: S completed steps log 0 .. S - 1.
            LossPoint(optimizer_step=i, epoch=0, batch_index=i, loss=losses[i])
            for i in range(step)
        ),
        consumed_digest=f"{step:064x}",
        model_state={
            "tower": {"w": _ref(torch.full((2, 3), w))},
            "span_head": {"b": _ref(torch.full((3,), -w, dtype=torch.bfloat16))},
            "span_weight": span_weight, "vocab_size": 256,
        },
    ).write(path)
    return path


def _main(*argv: object) -> int:
    return ckpt_average.main([str(a) for a in argv])


def test_without_the_flag_an_average_across_steps_is_refused(tmp_path):
    """Characterization: the seeds-at-one-step rule, unchanged."""
    paths = [_snapshot(tmp_path, s) for s in (2, 4, 6)]
    with pytest.raises(SystemExit, match="an average across two points of training"):
        _main(*paths, "--out", tmp_path / "avg.safetensors")


def test_one_runs_snapshots_average_with_their_steps_recorded(tmp_path, capsys):
    from safetensors.torch import load_file

    paths = [_snapshot(tmp_path, s) for s in (2, 4, 6)]
    out = tmp_path / "avg" / "last3.safetensors"
    assert _main(*paths, "--out", out, "--same-seed-trajectory", "--ft-row-ids", ROW) == 0
    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["trajectory"] == {
        "seed": 0, "optimizer_steps": [2, 4, 6], "ft_row_id": ROW,
        "rule": ckpt_average.TRAJECTORY_RULE,
    }
    assert manifest["optimizer_step"] == 6 and manifest["n_inputs"] == 3
    assert manifest["from"] == "tower" and manifest["ft_row_ids"] == [ROW]
    assert [r["seed"] for r in manifest["inputs"]] == [0, 0, 0]
    assert [Path(r["path"]).name for r in manifest["inputs"]] == [p.name for p in paths]
    tensors = load_file(out)
    assert torch.equal(tensors["tower.w"], torch.full((2, 3), 4.0))
    assert tensors["span_head.b"].dtype == torch.bfloat16
    assert torch.equal(tensors["span_head.b"], torch.full((3,), -4.0, dtype=torch.bfloat16))
    assert "same-seed trajectory: seed 0, steps [2, 4, 6]" in capsys.readouterr().out


def test_the_ft_row_is_optional_and_recorded_as_none(tmp_path):
    paths = [_snapshot(tmp_path, s) for s in (5, 6)]
    out = tmp_path / "avg.safetensors"
    assert _main(*paths, "--out", out, "--same-seed-trajectory") == 0
    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["trajectory"]["ft_row_id"] is None and manifest["ft_row_ids"] == []


def test_a_trajectory_average_is_not_scored(tmp_path):
    paths = [_snapshot(tmp_path, s) for s in (4, 6)]
    out = tmp_path / "avg.safetensors"
    assert _main(*paths, "--out", out, "--same-seed-trajectory", "--ft-row-ids", ROW) == 0
    with pytest.raises(ckpt_average.AverageRefusal, match=r"same-seed trajectory average .*"
                       r"steps \[4, 6\]\): real_ft_run.py --score-checkpoint pairs"):
        ckpt_average.read_manifest(out)


@pytest.mark.parametrize(
    ("make", "extra", "match"),
    [
        (lambda t: [_snapshot(t, 2), _snapshot(t, 4, seed=1)], [], "these are seeds \\[0, 1\\]"),
        (lambda t: [_snapshot(t, 4), _snapshot(t, 2)], [], "ascending order; got \\[4, 2\\]"),
        (lambda t: [_snapshot(t, 2), _snapshot(t, 4, losses=[3.0, 9.0, 2.25, 2.0, 1.75, 1.5])],
         [], "is not the start of"),
        (lambda t: [_snapshot(t, 2), _snapshot(t, 4, total=8)], [], "Different schedules"),
        (lambda t: [_snapshot(t, 2), _snapshot(t, 4, span_weight=0.5)], [], "span_weight"),
        (lambda t: [_snapshot(t, 2), _snapshot(t, 4, value=2.0)], [], "identical weights"),
        (lambda t: [_snapshot(t, 2), _snapshot(t, 4)], ["--from", "masters"], "--from tower"),
        (lambda t: [_snapshot(t, 2), _snapshot(t, 4)], ["--ft-row-ids", ROW, ROW],
         "names its one ft row, got 2"),
    ],
)
def test_what_is_not_one_run_is_refused(tmp_path, make, extra, match):
    paths = make(tmp_path)
    out = tmp_path / "avg.safetensors"
    with pytest.raises(SystemExit, match=match):
        _main(*paths, "--out", out, "--same-seed-trajectory", *extra)
    assert not out.exists() and not ckpt_average.manifest_path(out).exists()


def test_real_snapshots_of_one_toy_run_are_one_run(
    corpus, tmp_path  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    """The loss-log prefix rule on what the trainer actually writes: the toy run's
    --retain-tower-every snapshots average; the same files out of order do not."""
    ck = tmp_path / "ck"
    ft = _only(_run(corpus, tmp_path / "ft.jsonl", *_TRAIN, "--checkpoint-dir", str(ck),
                    "--retain-tower-every", "1", real=True, short=True), "ft")
    snaps = [ck / f"epoch-seed0-cpu-step{s}.json" for s in (1, 2, 3)]
    out = tmp_path / "last3.safetensors"
    assert _main(*snaps, "--out", out, "--same-seed-trajectory", "--ft-row-ids", ft.row_id) == 0
    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["trajectory"]["optimizer_steps"] == [1, 2, 3]
    assert manifest["optimizer_step"] == 3
    with pytest.raises(SystemExit, match="ascending order"):
        _main(snaps[1], snaps[0], "--out", tmp_path / "other.safetensors",
              "--same-seed-trajectory")
