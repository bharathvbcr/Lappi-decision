"""``tools/ckpt_average.py --from masters --norm-preserving --base-snapshot DIR`` (Fable I-2).

The uniform mean of N fine-tuning deltas whose pairwise cosine is c has
``sqrt((1 + (N-1)c) / N)`` of one delta's norm: it shrinks every seed-specific direction.
The norm-preserving average moves from the base by ``lambda * mean delta`` with
``lambda = sqrt(N / (1 + (N-1)c))``, and c is MEASURED from the inputs' fp32 masters against
the base -- never chosen on a gate. Pinned in closed form (orthogonal deltas give
``lambda = sqrt(3)``, identical deltas give 1), against a float64 numpy oracle on random
deltas, through the manifest reader, and through ``real_ft_run.py --score-checkpoint``,
whose row is tagged ``avg-np``.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import ckpt_average  # noqa: E402
import numpy as np  # noqa: E402
import real_ft_run  # noqa: E402
import torch  # noqa: E402
from test_real_ft_score_checkpoint import (  # noqa: E402
    AVG_IDS,
    STEPS,
    SUITE_SEED,
    _avg_args,
    _avg_control_args,
    _avg_row,
    _ledger,
    _reader,
    _run_avg_needle_control,
    _schedule,
    tiny_master_checkpoints,
)

from qd_train.backbone import TEXT_PREFIX  # noqa: E402


def _f32(values) -> bytes:
    values = list(values)
    return struct.pack(f"<{len(values)}f", *values)


def _input(tmp_path: Path, seed: int, tower: dict[str, list[float]], b: list[float],
           *, masters_for=None) -> Path:
    """A ``--optimizer master`` body whose tower is fp32 (so each master IS its tensor), with
    a master for every tower tensor named in ``masters_for`` (all of them by default) and
    for the span head."""
    from qd_train.run_control import Checkpoint, LossLog, Position, TensorRef

    refs = {k: TensorRef("float32", (len(v),), _f32(v)) for k, v in tower.items()}
    head = TensorRef("float32", (len(b),), _f32(b))
    keys = sorted(tower) if masters_for is None else list(masters_for)
    path = tmp_path / "ckpt" / f"epoch-seed{seed}-cuda.json"
    Checkpoint(
        position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=seed,
        schedule=_schedule(), loss_log=LossLog().snapshot(),
        consumed_digest=f"{seed + 1:064x}",
        model_state={
            "tower": refs, "span_head": {"b": head},
            "optimizer": {"masters": [*(refs[k] for k in keys), head]},
            "span_weight": 1.0, "vocab_size": 8,
        },
    ).write(path)
    return path


def _base(tmp_path: Path, tensors: dict[str, list[float]]) -> Path:
    from safetensors.torch import save_file

    snap = tmp_path / "snap"
    snap.mkdir(parents=True, exist_ok=True)
    save_file(
        {TEXT_PREFIX + k: torch.tensor(v, dtype=torch.float32) for k, v in tensors.items()},
        str(snap / "model.safetensors"),
    )
    return snap


def _np_main(tmp_path: Path, paths: list[Path], base: Path, *extra: str) -> tuple[dict, dict]:
    out = tmp_path / "avg" / "avgnp.safetensors"
    assert ckpt_average.main([
        *map(str, paths), "--out", str(out), "--from", "masters", "--norm-preserving",
        "--base-snapshot", str(base), "--ft-row-ids", *(AVG_IDS[s] for s in range(len(paths))),
        *extra,
    ]) == 0
    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    from safetensors.torch import load_file

    return manifest, load_file(str(out))


BASE_W = [0.5, -1.0, 2.0]


def test_orthogonal_deltas_give_lambda_sqrt3_and_keep_one_deltas_norm(tmp_path):
    eye = np.eye(3)
    paths = [
        _input(tmp_path, s, {"w": list(np.add(BASE_W, eye[s]))}, [float(s), 1.0])
        for s in range(3)
    ]
    manifest, out = _np_main(tmp_path, paths, _base(tmp_path, {"w": BASE_W}))
    block = manifest["norm_preserving"]
    assert block["c"] == 0.0 and block["pairwise_cosine"] == {"0-1": 0.0, "0-2": 0.0, "1-2": 0.0}
    assert block["lambda"] == block["lambda_from_c"] == math.sqrt(3)
    assert block["lambda_derived_from_c"] is True and block["n"] == 3
    assert block["delta_norms"] == [1.0, 1.0, 1.0]
    expected = (torch.tensor(BASE_W, dtype=torch.float64)
                + math.sqrt(3) * torch.full((3,), 1 / 3, dtype=torch.float64))
    assert torch.equal(out["tower.w"], expected.to(torch.float32))
    # The moved delta keeps one input's norm, which the plain mean (norm 1/sqrt(3)) does not.
    moved = out["tower.w"].double() - torch.tensor(BASE_W, dtype=torch.float64)
    assert float(moved.norm()) == pytest.approx(1.0, abs=1e-6)
    # The span head has no base: its plain mean, and the manifest says so.
    assert torch.equal(out["span_head.b"], torch.tensor([1.0, 1.0]))
    assert block["span_head"] == ckpt_average.SPAN_HEAD_NO_BASE
    assert manifest["method"] == ckpt_average.NORM_PRESERVING_FORMULA
    assert manifest["from"] == "masters" and block["base_snapshot"] == "snap"
    sha = hashlib.sha256((tmp_path / "snap" / "model.safetensors").read_bytes()).hexdigest()
    assert block["base_files"] == {"model.safetensors": sha}


def test_identical_deltas_give_lambda_one_and_the_plain_mean(tmp_path):
    delta = [1.0, 2.0, -3.0]
    paths = [
        _input(tmp_path, s, {"w": list(np.add(BASE_W, delta))}, [float(s), -float(s)])
        for s in range(3)
    ]
    manifest, out = _np_main(tmp_path, paths, _base(tmp_path, {"w": BASE_W}))
    block = manifest["norm_preserving"]
    assert block["c"] == pytest.approx(1.0, abs=1e-15)
    assert block["lambda"] == pytest.approx(1.0, abs=1e-15)
    assert torch.allclose(out["tower.w"], torch.tensor(np.add(BASE_W, delta), dtype=torch.float32))


def test_c_and_lambda_match_a_float64_oracle_over_every_tower_master(tmp_path):
    """Random deltas over two tower tensors, one tower tensor WITHOUT a master (it has a base:
    lambda moves it, but c is measured over masters only), and a span head."""
    rng = np.random.default_rng(7)
    base = {"w": list(rng.normal(size=5)), "v": list(rng.normal(size=4)),
            "frozen": list(rng.normal(size=2))}
    shared = rng.normal(size=9)
    tower_in, paths = [], []
    for s in range(3):
        d = 0.6 * shared + rng.normal(size=9)
        tower = {
            "w": [float(x) for x in np.float32(np.add(base["w"], d[:5]))],
            "v": [float(x) for x in np.float32(np.add(base["v"], d[5:]))],
            "frozen": [float(x) for x in np.float32(base["frozen"])],
        }
        tower_in.append(tower)
        paths.append(_input(tmp_path, s, tower, list(rng.normal(size=3)),
                            masters_for=["w", "v"]))
    manifest, out = _np_main(tmp_path, paths, _base(tmp_path, base))
    block = manifest["norm_preserving"]
    b32 = {k: np.float32(v).astype(np.float64) for k, v in base.items()}
    deltas = [
        np.concatenate([np.float32(t["v"]).astype(np.float64) - b32["v"],
                        np.float32(t["w"]).astype(np.float64) - b32["w"]])
        for t in tower_in
    ]
    pairs = {}
    for i in range(3):
        for j in range(i + 1, 3):
            pairs[f"{i}-{j}"] = float(
                deltas[i] @ deltas[j] / (np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j]))
            )
    c = sum(pairs.values()) / 3
    assert block["pairwise_cosine"] == pytest.approx(pairs, abs=1e-12)
    assert block["c"] == pytest.approx(c, abs=1e-12) and 0.1 < c < 0.9
    lam = math.sqrt(3 / (1 + 2 * c))
    assert block["lambda"] == pytest.approx(lam, abs=1e-12)
    assert block["c_measured_over"]["tensors"] == 2
    assert block["c_measured_over"]["parameters"] == 9
    for key in ("w", "v"):
        mean = np.mean([np.float32(t[key]).astype(np.float64) for t in tower_in], axis=0)
        expected = (b32[key] + lam * (mean - b32[key])).astype(np.float32)
        assert np.array_equal(out[f"tower.{key}"].numpy(), expected), key
    # A tower tensor every input left at its base stays at its base whatever lambda is.
    assert np.array_equal(out["tower.frozen"].numpy(), np.float32(base["frozen"]))


def test_a_lambda_not_derived_from_c_is_marked_so(tmp_path):
    eye = np.eye(3)
    paths = [
        _input(tmp_path, s, {"w": list(np.add(BASE_W, eye[s]))}, [0.0, 1.0]) for s in range(3)
    ]
    manifest, out = _np_main(tmp_path, paths, _base(tmp_path, {"w": BASE_W}),
                             "--lambda-not-derived-from-c", "2.0")
    block = manifest["norm_preserving"]
    assert block["lambda"] == 2.0 and block["lambda_derived_from_c"] is False
    assert block["lambda_from_c"] == math.sqrt(3)
    expected = torch.tensor(BASE_W, dtype=torch.float64) + 2.0 * torch.full(
        (3,), 1 / 3, dtype=torch.float64
    )
    assert torch.equal(out["tower.w"], expected.to(torch.float32))


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("no_base", "is not a directory"),
        ("empty_base", "no .safetensors"),
        ("base_lacks_tensor", "has no base tensor"),
        ("base_other_shape", "a remapped vocabulary"),
        ("zero_delta", "moved no tower master"),
    ],
)
def test_a_norm_preserving_average_without_its_base_is_refused(tmp_path, case, match):
    eye = np.eye(3)
    rows = [list(np.add(BASE_W, eye[s])) for s in range(3)]
    if case == "zero_delta":
        rows[1] = list(BASE_W)
    paths = [_input(tmp_path, s, {"w": rows[s]}, [float(s), 1.0]) for s in range(3)]
    base = tmp_path / "snap"
    if case == "empty_base":
        base.mkdir()
    elif case == "base_lacks_tensor":
        base = _base(tmp_path, {"other": BASE_W})
    elif case == "base_other_shape":
        base = _base(tmp_path, {"w": [*BASE_W, 0.0]})
    elif case == "zero_delta":
        base = _base(tmp_path, {"w": BASE_W})
    with pytest.raises(SystemExit, match=match):
        ckpt_average.main([*map(str, paths), "--out", str(tmp_path / "o.safetensors"),
                           "--from", "masters", "--norm-preserving", "--base-snapshot",
                           str(base)])
    assert not (tmp_path / "o.safetensors").exists()


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--norm-preserving"], "needs --from masters and --base-snapshot"),
        (["--from", "masters", "--norm-preserving"], "--base-snapshot"),
        (["--norm-preserving", "--base-snapshot", "/b"], "--from masters"),
        (["--from", "masters", "--base-snapshot", "/b"], "only with --norm-preserving"),
        (["--from", "masters", "--lambda-not-derived-from-c", "1.5"],
         "only with --norm-preserving"),
    ],
)
def test_norm_preserving_flags_are_refused_out_of_place(tmp_path, extra, match):
    with pytest.raises(SystemExit, match=match):
        ckpt_average.main(["/a/epoch-seed0-cuda.json", "/a/epoch-seed1-cuda.json",
                           "--out", str(tmp_path / "o.safetensors"), *extra])


def _np_manifest(tmp_path: Path, *, base: str = "snap", c: float = 0.2,
                 derived: bool = True, tamper: dict | None = None) -> Path:
    """An avg-np manifest over tiny stand-in tensors, by the tool's own writer."""
    from test_real_ft_score_checkpoint import _sources

    records = _sources(tmp_path)
    out = tmp_path / "avg" / "avgnp.safetensors"
    from_c = math.sqrt(3 / (1 + 2 * c))
    block = {
        "formula": ckpt_average.NORM_PRESERVING_FORMULA, "n": 3, "c": c,
        "pairwise_cosine": {"0-1": c, "0-2": c, "1-2": c}, "lambda_from_c": from_c,
        "lambda": from_c if derived else 2.0, "lambda_derived_from_c": derived,
        "base_snapshot": base, "span_head": ckpt_average.SPAN_HEAD_NO_BASE,
        **(tamper or {}),
    }
    ckpt_average.write(
        {"tower.w": torch.zeros(4, dtype=torch.bfloat16), "span_head.b": torch.zeros(2)}, out,
        source="masters", inputs=records, optimizer_step=STEPS,
        schedule=_schedule().to_json(), scalars={"vocab_size": 8, "span_weight": 1.0},
        ft_row_ids=[AVG_IDS[s] for s in (0, 1, 2)],
        tensor_sources={"tower.w": "master", "span_head.b": "master"},
        master_index={"tower.w": 0, "span_head.b": 1}, norm_preserving=block,
    )
    return out


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ({"lambda": 1.9}, "is said to come from c"),
        ({"lambda_from_c": 1.0}, "is not sqrt"),
        ({"n": 2}, "norm_preserving.n"),
        ({"formula": "theta = whatever"}, "formula"),
        ({"c": float("nan")}, "not a finite number"),
    ],
)
def test_a_manifest_whose_lambda_is_not_cs_is_refused(tmp_path, tamper, match):
    avg = _np_manifest(tmp_path, tamper=tamper)
    with pytest.raises(ckpt_average.AverageRefusal, match=match):
        ckpt_average.read_manifest(avg)


class _Reached(Exception):
    pass


def test_an_avg_np_scored_on_another_base_is_refused_before_the_tower(tmp_path, monkeypatch):
    from types import SimpleNamespace

    avg = _np_manifest(tmp_path, base="some-other-snapshot")
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))

    def never(**kwargs):
        raise AssertionError("a tower was built for an average on the wrong base")

    monkeypatch.setattr(real_ft_run, "_real_step", never)
    val = SimpleNamespace(plan=[SimpleNamespace(tokens=np.zeros((1, 64)))])
    with pytest.raises(SystemExit, match="moved from the base 'some-other-snapshot'"):
        real_ft_run._score_checkpoint(
            _avg_args(avg, ledger), reader=_reader(), val=val, device="cuda",
            ledger=None,  # type: ignore[arg-type]
            reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
            needle_suite=real_ft_run.NeedleSuite([], [], {}, [], not_run="no"),
            ood_suite=real_ft_run.OodSuite([], None, None, not_run="no"), suite_seed=SUITE_SEED,
        )


@pytest.mark.parametrize("derived", [True, False])
def test_an_avg_np_needle_control_row_is_tagged_and_names_its_lambda(
    tmp_path, monkeypatch, derived
):
    avg = _np_manifest(tmp_path, derived=derived)
    ledger = _ledger(tmp_path, *(_avg_row(s) for s in (0, 1, 2)))
    (_, _, seed), captured, _ = _run_avg_needle_control(monkeypatch, _avg_control_args(avg, ledger))
    recipe = captured["recipe"]
    assert recipe["tag"] == "avg-np-needle-length-control" and seed == SUITE_SEED
    norm = recipe["averaged"]["norm_preserving"]
    assert norm["lambda_derived_from_c"] is derived and norm["base_snapshot"] == "snap"
    assert norm["c"] == 0.2 and "NORM-PRESERVING" in captured["notes"]
    assert captured["metrics"]["ft_run_row_ids"].value == ",".join(
        AVG_IDS[s] for s in (0, 1, 2)
    )
    assert (real_ft_run.LAMBDA_CHOSEN_REASON in captured["quick_reasons"]) is (not derived)


def test_a_norm_preserving_average_of_three_tiny_checkpoints_is_scored_end_to_end(
    tmp_path, monkeypatch
):
    """The real path on CPU: three tiny ``--optimizer master`` checkpoints fine-tuned from the
    tiny snapshot, averaged with --norm-preserving against that snapshot, scored by
    --score-checkpoint. The row is avg-np-score-val and carries the measured c and lambda."""
    from test_real_ft_shuffled_label import REV

    from qd_train.ledger import Ledger

    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    avg = tmp_path / "avg" / "avgnp.safetensors"
    assert ckpt_average.main([
        *map(str, tiny.paths), "--out", str(avg), "--from", "masters", "--norm-preserving",
        "--base-snapshot", str(tiny.snapshot), "--ft-row-ids", *tiny.ids,
    ]) == 0
    block = json.loads(ckpt_average.manifest_path(avg).read_text())["norm_preserving"]
    assert block["lambda_derived_from_c"] is True and block["lambda"] >= 1.0
    ledger = tmp_path / "eval.jsonl"
    assert real_ft_run.main([
        "--out", str(tiny.out), "--rev", REV, "--devices", "cpu", "--seeds", "0", "1", "2",
        "--score-val", "--score-checkpoint", str(avg), "--real-backbone", str(tiny.snapshot),
        "--ft-ledger", str(tiny.ft_ledger), "--ledger", str(ledger),
        "--tokenizer-json", str(tiny.tokenizer),
    ]) == 0
    (row,) = Ledger(ledger).rows()
    recipe = dict(row.recipe or {})
    assert recipe["tag"] == "avg-np-score-val"
    assert recipe["averaged"]["norm_preserving"]["c"] == block["c"]
    assert recipe["averaged"]["norm_preserving"]["lambda"] == block["lambda"]
    assert row.protocol.seed == SUITE_SEED
    assert row.metrics["ft_run_row_ids"].value == ",".join(tiny.ids)
    assert "NORM-PRESERVING" in row.notes and "human's decision" in row.notes
    assert real_ft_run.LAMBDA_CHOSEN_REASON not in (row.quick_reason or "")
