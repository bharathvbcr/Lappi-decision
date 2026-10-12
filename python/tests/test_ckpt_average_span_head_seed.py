"""``tools/ckpt_average.py --span-head-seed N`` (Fable's board-sweep ruling 2g).

The v5 five-seed average beat every seed on the tower metrics and collapsed on span (0.773 against
0.907-0.915): the seeds' span heads are near-orthogonal, so their mean is 1/sqrt(5) of a head.
v6 pre-registers the tower average with ONE seed's span head. The ruling:

* the head is taken from the input whose checkpoint *body* says ``seed == N`` -- never from a file
  name;
* refused when no input, or more than one input, carries that seed; refused with
  ``--same-seed-trajectory``; refused for a v6 run whose seed-N ``span_head_init`` manifest is not
  the shared one;
* the manifest records ``span_head_source: {mode: "seed"|"averaged", seed, input, payload_digest}``
  so a primary and a fallback row can never be pooled;
* it composes with ``--from masters`` and ``--norm-preserving`` (which then average the tower only).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
import torch  # noqa: E402
from test_real_ft_score_checkpoint import _schedule  # noqa: E402

STEPS = 1505


def _f32(values) -> bytes:
    values = list(values)
    return struct.pack(f"<{len(values)}f", *values)


def _ref(values):
    from qd_train.run_control import TensorRef

    return TensorRef("float32", (len(values),), _f32(values))


def _ckpt(tmp_path: Path, name: str, seed: int, w: list[float], head: list[float], *,
          masters: bool = False) -> Path:
    """A tiny checkpoint body: tower tensor ``w``, span head ``b``; the file NAME is free."""
    from qd_train.run_control import Checkpoint, LossLog, Position

    tower, span = {"w": _ref(w)}, {"b": _ref(head)}
    state = {"tower": tower, "span_head": span, "span_weight": 1.0, "vocab_size": 8}
    if masters:
        state["optimizer"] = {"masters": [tower["w"], span["b"]]}
    path = tmp_path / "ckpt" / name
    Checkpoint(
        position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=seed,
        schedule=_schedule(), loss_log=LossLog().snapshot(),
        consumed_digest=f"{seed + 1:064x}", model_state=state,
    ).write(path)
    return path


def _three(tmp_path: Path, **kw) -> list[Path]:
    # File names deliberately disagree with the bodies: seed 1's body is in "epoch-seed0-...".
    return [
        _ckpt(tmp_path, "epoch-seed2-cuda.json", 1, [3.0, 0.0], [10.0, 20.0], **kw),
        _ckpt(tmp_path, "epoch-seed0-cuda.json", 0, [0.0, 3.0], [1.0, 2.0], **kw),
        _ckpt(tmp_path, "epoch-seed1-cuda.json", 2, [0.0, 0.0], [100.0, 200.0], **kw),
    ]


def _run(tmp_path: Path, paths: list[Path], *extra: str) -> tuple[dict, dict]:
    out = tmp_path / "avg" / "a.safetensors"
    code = ckpt_average.main([*map(str, paths), "--out", str(out), *extra])
    assert code == 0
    from safetensors.torch import load_file

    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    return manifest, load_file(str(out))


def _digest(path: Path) -> str:
    from qd_train.run_control import Checkpoint

    return str(Checkpoint.read_body(path)["payload_digest"])


def test_the_flag_takes_the_head_of_the_input_whose_body_says_that_seed(tmp_path):
    paths = _three(tmp_path)
    manifest, out = _run(tmp_path, paths, "--span-head-seed", "1")
    assert torch.equal(out["span_head.b"], torch.tensor([10.0, 20.0]))  # body seed 1, not a name
    assert torch.equal(out["tower.w"], torch.tensor([1.0, 1.0]))  # the tower is still the mean
    assert manifest["span_head_source"] == {
        "mode": "seed", "seed": 1, "input": str(paths[0]), "payload_digest": _digest(paths[0]),
    }
    assert manifest["tensor_sources"]["span_head.b"] == "span_head_seed"


def test_without_the_flag_the_head_is_the_mean_and_the_manifest_is_the_one_it_always_was(tmp_path):
    """An absent ``span_head_source`` reads as averaged; a plain average's manifest (whose sha256
    eval rows name it by) does not move (test_prompt_format_chain pins its key set)."""
    manifest, out = _run(tmp_path, _three(tmp_path), "--ft-row-ids", "a", "b", "c")
    assert torch.equal(out["span_head.b"], torch.tensor([37.0, 74.0]))
    assert "span_head_source" not in manifest
    weights = tmp_path / "avg" / "a.safetensors"
    assert ckpt_average.read_manifest(weights).span_head_source is None


def test_a_seed_no_input_carries_is_refused(tmp_path):
    with pytest.raises(SystemExit, match=r"no input.*seed 7"):
        _run(tmp_path, _three(tmp_path), "--span-head-seed", "7")


def test_a_seed_two_inputs_carry_is_refused(tmp_path):
    paths = [
        _ckpt(tmp_path, "a.json", 1, [3.0, 0.0], [10.0, 20.0]),
        _ckpt(tmp_path, "b.json", 1, [0.0, 3.0], [1.0, 2.0]),
        _ckpt(tmp_path, "c.json", 2, [0.0, 0.0], [5.0, 6.0]),
    ]
    with pytest.raises(SystemExit, match=r"more than one input .*seed 1"):
        _run(tmp_path, paths, "--span-head-seed", "1")


def test_the_flag_is_refused_with_same_seed_trajectory(tmp_path):
    with pytest.raises(SystemExit, match="same-seed-trajectory"):
        _run(tmp_path, _three(tmp_path), "--span-head-seed", "1", "--same-seed-trajectory")


def test_it_composes_with_from_masters(tmp_path):
    paths = _three(tmp_path, masters=True)
    manifest, out = _run(tmp_path, paths, "--from", "masters", "--span-head-seed", "2")
    assert torch.equal(out["span_head.b"], torch.tensor([100.0, 200.0]))
    assert torch.equal(out["tower.w"], torch.tensor([1.0, 1.0]))
    assert manifest["from"] == "masters" and manifest["span_head_source"]["seed"] == 2
    assert "span_head.b" not in manifest["master_index"]  # the head did not come from a master


def test_it_composes_with_norm_preserving(tmp_path):
    from safetensors.torch import save_file

    paths = _three(tmp_path, masters=True)
    snap = tmp_path / "snap"
    snap.mkdir()
    from qd_train.backbone import TEXT_PREFIX

    save_file({TEXT_PREFIX + "w": torch.tensor([0.5, 0.5])}, str(snap / "model.safetensors"))
    manifest, out = _run(
        tmp_path, paths, "--from", "masters", "--norm-preserving", "--base-snapshot", str(snap),
        "--span-head-seed", "0",
    )
    assert torch.equal(out["span_head.b"], torch.tensor([1.0, 2.0]))
    assert manifest["span_head_source"]["seed"] == 0
    assert "norm_preserving" in manifest


def test_a_v6_run_needs_the_shared_init_manifest_and_a_seed_specific_one_is_refused(tmp_path):
    paths = _three(tmp_path)
    with pytest.raises(SystemExit, match=r"--v6-run.*--span-head-init-manifest"):
        _run(tmp_path, paths, "--span-head-seed", "1", "--v6-run")
    own = tmp_path / "own.manifest.json"
    own.write_text(json.dumps({"construction": {"seed": 1}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the shared"):
        _run(tmp_path, paths, "--span-head-seed", "1", "--v6-run", "--span-head-init-manifest",
             str(own))
    shared = tmp_path / "shared.manifest.json"
    shared.write_text(
        json.dumps({"construction": {"seed": 0, "shared_across_seeds": True},
                    "content_digest": {"value": "d" * 64}}),
        encoding="utf-8",
    )
    manifest, _ = _run(tmp_path, paths, "--span-head-seed", "1", "--v6-run",
                       "--span-head-init-manifest", str(shared))
    assert manifest["span_head_source"]["init_manifest"]["shared_across_seeds"] is True
    assert manifest["span_head_source"]["init_manifest"]["content_digest"] == "d" * 64


def test_the_v6_flags_mean_nothing_without_a_seed(tmp_path):
    with pytest.raises(SystemExit, match="--span-head-seed"):
        _run(tmp_path, _three(tmp_path), "--v6-run")


def test_read_manifest_checks_a_seed_source_against_its_inputs(tmp_path):
    paths = _three(tmp_path)
    manifest, _ = _run(tmp_path, paths, "--span-head-seed", "1", "--ft-row-ids", "a", "b", "c")
    weights = tmp_path / "avg" / "a.safetensors"
    assert ckpt_average.read_manifest(weights).span_head_source["seed"] == 1
    path = ckpt_average.manifest_path(weights)
    for tamper in (
        {"mode": "seed", "seed": 9, "input": str(paths[0]), "payload_digest": "0" * 64},
        {"mode": "seed", "seed": 1, "input": str(paths[1]), "payload_digest": _digest(paths[1])},
        {"mode": "median", "seed": None, "input": None, "payload_digest": None},
    ):
        path.write_text(json.dumps({**manifest, "span_head_source": tamper}), encoding="utf-8")
        with pytest.raises(ckpt_average.AverageRefusal, match="span_head_source"):
            ckpt_average.read_manifest(weights)
    legacy = {k: v for k, v in manifest.items() if k != "span_head_source"}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert ckpt_average.read_manifest(weights).span_head_source is None  # pre-flag averages read
