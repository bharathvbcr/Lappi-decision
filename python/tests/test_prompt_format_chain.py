"""v5's prompt format, carried from the train shard header to everything scored or exported.

L-v5-fmt stamped ``ShardHeader.prompt_format`` and made qd-export stamp a release from its
source manifest's top-level ``prompt_format`` (absent = 1). The chain between them is this
lane's (GAP-L-V5FMT-SOURCE-MANIFEST-PROMPT-FORMAT-CHAIN-2026-10-02). Pinned here:

* the ft recipe names ``prompt_format``, read off the train header, never off
  ``qd_data.render.PROMPT_FORMAT``, and only when it is not 1;
* a ``QwenDecisionStep`` checkpoint states the format in ``model_state`` (only when not 1), a
  ``--retain-tower-every`` snapshot keeps it, and a resume across formats is refused;
* ``tools/ckpt_average.py`` reads every input's format, refuses two formats, and writes the
  top-level ``prompt_format`` only when it is not 1;
* ``--score-checkpoint`` refuses an ft row whose recipe format (absent = 1) is not
  ``PROMPT_FORMAT`` (the DRAFT's ``format.refusal_both_ways.eval``).

Format 1 stays absent everywhere, so a v4 recipe, checkpoint body and average manifest are the
bytes they were.
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
import real_ft_run as rft  # noqa: E402
import test_backbone as tb  # noqa: E402
from test_checkpoint_retain import _TRAIN  # noqa: E402
from test_ckpt_average_trajectory import _ref  # noqa: E402
from test_real_ft_rungd_flags import _only, _run, corpus  # noqa: E402, F401

from qd_data.render import PROMPT_FORMAT  # noqa: E402
from qd_train.backbone import BackboneContractViolation, QwenDecisionStep  # noqa: E402
from qd_train.run_control import (  # noqa: E402
    Checkpoint,
    LossLog,
    LossPoint,
    LRSchedule,
    Position,
)

ROW = "f0000000-0000-4000-8000-000000000000"

#: What an average's manifest held before ``prompt_format`` existed: a format-1 average must
#: still write exactly these keys, so its bytes -- and the sha256 eval rows name it by -- do not
#: move.
V4_AVERAGE_KEYS = {
    "accumulator", "from", "ft_row_ids", "inputs", "master_index", "method", "n_inputs",
    "n_tensors", "optimizer_step", "resumable", "safetensors_sha256", "schedule", "source",
    "span_weight", "tensor_sources", "tool", "vocab_size", "why_not_resumable",
}


def test_format_2_is_what_this_build_renders():
    """The premise of every test below: v5-build renders format 2."""
    assert PROMPT_FORMAT == 2


# --- the recipe ---------------------------------------------------------------------------

_OFF = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
        "permutation": None, "replay": None}


def test_the_recipe_names_a_format_other_than_1_and_1_is_absent():
    assert rft._recipe_pieces(**_OFF, prompt_format=1) == {}
    assert rft._recipe_pieces(**_OFF) == {}
    assert rft._recipe_pieces(**_OFF, prompt_format=2) == {"prompt_format": 2}
    assert "prompt_format" in rft.RECIPE_PIECE_KEYS, "mirrored into every row scored from it"


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "2", None])
def test_a_recipe_format_that_is_not_an_int_of_at_least_1_is_refused(bad):
    with pytest.raises(ValueError, match="prompt_format must be an int >= 1"):
        rft._recipe_pieces(**_OFF, prompt_format=bad)


# --- the eval refusal ---------------------------------------------------------------------


def _ledger(tmp_path: Path, recipe: dict[str, object]) -> Path:
    path = tmp_path / "ft.jsonl"
    row = {"row_id": ROW, "run_kind": "ft", "status": "completed", "recipe": recipe}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_score_checkpoint_refuses_a_model_of_another_prompt_format(tmp_path):
    """A v4 row (no key: format 1) is refused before any weights are read; a format-2 row is
    the model this build scores."""
    with pytest.raises(SystemExit, match="trained on prompt format 1 and this build renders "
                                         "format 2"):
        rft._ft_row(_ledger(tmp_path, {"tag": "epoch"}), ROW)
    row = rft._ft_row(_ledger(tmp_path, {"tag": "epoch", "prompt_format": 2}), ROW)
    assert row["row_id"] == ROW


def test_score_checkpoint_refuses_a_row_with_no_recipe_by_name(tmp_path):
    """Not read as format 1: a row with no recipe says nothing about any format."""
    path = tmp_path / "ft.jsonl"
    path.write_text(json.dumps({"row_id": ROW, "run_kind": "ft", "status": "completed"}) + "\n",
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="has no recipe"):
        rft._ft_row(path, ROW)


@pytest.mark.parametrize("bad", [0, True, "2", 2.0])
def test_score_checkpoint_refuses_a_recipe_format_that_is_not_an_int(tmp_path, bad):
    with pytest.raises(SystemExit, match="is not an int >= 1"):
        rft._ft_row(_ledger(tmp_path, {"tag": "epoch", "prompt_format": bad}), ROW)


# --- the step's checkpoint ------------------------------------------------------------------


def _step(tmp_path: Path, prompt_format: int = 1) -> QwenDecisionStep:
    tower, _ = tb._tiny_tower(tmp_path, gradient_checkpointing=False)
    return QwenDecisionStep(
        tower, seed=0, lr=1e-3, total_steps=4, max_width=64, prompt_format=prompt_format,
    )


def test_the_step_states_its_format_and_a_resume_across_formats_is_refused(tmp_path):
    v5, v4 = _step(tmp_path / "a", 2), _step(tmp_path / "b")
    state_5, state_4 = v5.state(), v4.state()
    assert state_5["prompt_format"] == 2
    assert "prompt_format" not in state_4, "format 1 writes the state it always did"
    with pytest.raises(BackboneContractViolation, match="prompt format 2 and this step on 1"):
        v4.load_state(state_5)
    with pytest.raises(BackboneContractViolation, match="prompt format 1 and this step on 2"):
        v5.load_state(state_4)
    _step(tmp_path / "c", 2).load_state(state_5)


@pytest.mark.parametrize("bad", [0, True, 2.0])
def test_a_step_format_that_is_not_an_int_of_at_least_1_is_refused(tmp_path, bad):
    tower, _ = tb._tiny_tower(tmp_path, gradient_checkpointing=False)
    with pytest.raises(ValueError, match="prompt_format must be an int >= 1"):
        QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=4, max_width=64, prompt_format=bad)


# --- ckpt_average ---------------------------------------------------------------------------


def _ckpt(tmp_path: Path, seed: int, **state: object) -> Path:
    """One seed's checkpoint at step 4: a tower, a span head and ``state``'s extra keys."""
    path = tmp_path / f"epoch-seed{seed}-cpu.json"
    Checkpoint(
        position=Position(epoch=0, index=4), optimizer_step=4, seed=seed,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=4, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog(
            LossPoint(optimizer_step=i, epoch=0, batch_index=i, loss=2.0 - i / 8) for i in range(4)
        ),
        consumed_digest=f"{seed:064x}",
        model_state={
            "tower": {"w": _ref(torch.full((2, 3), float(seed + 1)))},
            "span_head": {"b": _ref(torch.full((3,), -float(seed + 1)))},
            "span_weight": 1.0, "vocab_size": 256, **state,
        },
    ).write(path)
    return path


def _average(tmp_path: Path, *paths: Path) -> dict[str, object]:
    out = tmp_path / "avg.safetensors"
    assert ckpt_average.main([*(str(p) for p in paths), "--out", str(out)]) == 0
    return json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))


def test_an_average_of_format_2_inputs_states_format_2(tmp_path):
    m = _average(tmp_path, _ckpt(tmp_path, 0, prompt_format=2), _ckpt(tmp_path, 1, prompt_format=2))
    assert m["prompt_format"] == 2 and type(m["prompt_format"]) is int


def test_an_average_of_format_1_inputs_writes_the_manifest_it_always_wrote(tmp_path):
    m = _average(tmp_path, _ckpt(tmp_path, 0), _ckpt(tmp_path, 1))
    assert set(m) == V4_AVERAGE_KEYS


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        ({"prompt_format": 2}, {}, "trained on prompt format 1 and .* on 2"),
        ({}, {"prompt_format": 2}, "trained on prompt format 2 and .* on 1"),
        ({"prompt_format": 2}, {"prompt_format": 3}, "prompt format 3 and .* on 2"),
        ({"prompt_format": True}, {"prompt_format": True}, "is not an int >= 1"),
        ({"prompt_format": 0}, {"prompt_format": 0}, "is not an int >= 1"),
    ],
)
def test_inputs_of_two_formats_or_a_malformed_one_are_refused(tmp_path, a, b, match):
    out = tmp_path / "avg.safetensors"
    with pytest.raises(SystemExit, match=match):
        ckpt_average.main([str(_ckpt(tmp_path, 0, **a)), str(_ckpt(tmp_path, 1, **b)),
                           "--out", str(out)])
    assert not out.exists() and not ckpt_average.manifest_path(out).exists()


@pytest.mark.parametrize("bad", [True, 2.0, "2", 0])
def test_read_manifest_refuses_a_malformed_stated_format(tmp_path, bad):
    out = tmp_path / "avg.safetensors"
    assert ckpt_average.main([
        str(_ckpt(tmp_path, 0, prompt_format=2)), str(_ckpt(tmp_path, 1, prompt_format=2)),
        "--out", str(out), "--ft-row-ids", ROW, ROW.replace("f", "e", 1),
    ]) == 0
    assert ckpt_average.read_manifest(out).body["prompt_format"] == 2
    path = ckpt_average.manifest_path(out)
    body = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**body, "prompt_format": bad}), encoding="utf-8")
    with pytest.raises(ckpt_average.AverageRefusal, match="is not an int >= 1"):
        ckpt_average.read_manifest(out)


# --- end to end, on what the trainer writes --------------------------------------------------


def test_a_toy_run_carries_its_shard_sets_format_from_the_header_to_the_average(
    corpus, tmp_path  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    """The real tiny tower on CPU: the recipe's format is the train header's, the resume
    checkpoint and every retained snapshot state it, the last-3 average's manifest states it,
    and --score-checkpoint's row lookup accepts the row."""
    header = json.loads((corpus.out / "shards" / "train" / "header.json").read_text("utf-8"))
    assert header.get("prompt_format") == PROMPT_FORMAT, "the toy set is a format-2 set"
    ck = tmp_path / "ck"
    ledger = tmp_path / "ft.jsonl"
    ft = _only(_run(corpus, ledger, *_TRAIN, "--checkpoint-dir", str(ck),
                    "--checkpoint-every", "3", "--retain-tower-every", "1", real=True,
                    short=True), "ft")
    assert ft.recipe["prompt_format"] == header["prompt_format"]
    assert Checkpoint.read_body(ck / "epoch-seed0-cpu.json")["model_state"]["prompt_format"] == 2
    snaps = [ck / f"epoch-seed0-cpu-step{s}.json" for s in (1, 2, 3)]
    for snap in snaps:
        assert Checkpoint.read_body(snap)["model_state"]["prompt_format"] == 2, snap.name
    out = tmp_path / "last3.safetensors"
    argv = [*(str(s) for s in snaps), "--out", str(out), "--same-seed-trajectory",
            "--ft-row-ids", ft.row_id]
    assert ckpt_average.main(argv) == 0
    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["prompt_format"] == 2
    assert rft._ft_row(ledger, ft.row_id)["row_id"] == ft.row_id
