"""``tools/real_ft_run.py --score-checkpoint epoch-seed<N>-metal.safetensors``: a Metal export.

Human ask 7 of Fable's ojas advice (``AUDIT/ojas-training-2026-10-01/fable-advice.md``,
approved 2026-10-01): the campaign's PyTorch scorer accepts a model the Rust trainer
(``crates/qd-train``, over canonical tessl's Metal kernels) trained -- a scoring input only.
No gate, threshold, seed rule or verdict logic changes.

The artifact is one seed's weights, not a ``Checkpoint``: ``tower.*`` and ``span_head.*`` in
a ``.safetensors`` named like every per-seed checkpoint, ``<tag>-seed<N>-metal``, and the
Rust export's manifest beside it (``<name>.manifest.json``). It is paired with the ``ft``
row the Rust trainer wrote, as a ``.json`` checkpoint is paired with its row
(``--ft-row-id``, one ``--seeds``), and the same refusal discipline holds: every way the
pairing could put the wrong weights under the wrong row is refused before a tower loads.

What is pinned here:

* the contract, literally: the manifest's ``from`` is ``qd-train-export``, its ``trainer``
  ``qd-train-metal``, its ``device`` ``metal``; the ft row's ``recipe.device`` is ``metal``
  and its ``recipe.trainer`` ``qd-train-metal``; the manifest names the ft row by id;
* every row scored from it carries ``trained_by`` (trainer, device, source, manifest
  sha256), so it can never read as a torch-trained row, and is quick when its ft row is;
* every malformed variant is refused, naming the Metal export;
* END TO END on CPU: seed 0's tiny tower exported as a Metal artifact scores exactly as the
  same weights scored from their ``.json`` checkpoint -- every verdict line equal -- while
  the two rows differ in what they say trained the model;
* the torch paths are untouched: an average keeps its route, a ``.json`` keeps its own.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
from test_real_ft_score_checkpoint import (  # noqa: E402
    _NO_NEEDLE,
    _NO_OOD,
    SHARD_HASH,
    STEPS,
    SUITE_SEED,
    _ledger,
    _Reached,
    _reader,
    _run_avg_needle_control,
    tiny_master_checkpoints,
)

#: The contract, spelled here rather than imported: a rename on either side fails this file.
METAL = "metal"
TRAINER = "qd-train-metal"
SOURCE = "qd-train-export"
EXPORT = "epoch-seed0-metal.safetensors"
METAL_ID = "3e7a1000-aaaa-4bbb-8ccc-00000000000a"
METAL_QUICK = "1 seed, truncated schedule, Metal/tessl trainer (parity ladder rung d)"


def _metal_row(*, row_id: str = METAL_ID, seed: int = 0, steps: int = STEPS,
               drop: tuple[str, ...] = (), **over) -> dict:
    """The ft row the Rust trainer writes: quick, on metal, naming its trainer."""
    recipe = {
        "tag": "epoch", "device": METAL, "trainer": TRAINER, "shard_hash": SHARD_HASH,
        "backbone_snapshot": "snap", "attn_implementation": "sdpa", "lr": 1e-5,
        "span_weight": 1.0, "optimizer_recipe": "master",
    }
    row: dict = {
        "row_id": row_id, "run_kind": "ft", "status": "completed",
        "quick": True, "quick_reason": METAL_QUICK,
        "recipe": recipe, "protocol": {"seed": seed},
        "metrics": {
            "train.optimizer_steps": {"value": steps},
            "train.termination": {"value": "steps_exhausted"},
        },
    }
    for key, value in over.items():
        if key in ("quick", "quick_reason"):
            row[key] = value
        else:
            recipe[key] = value
    for key in drop:
        if key in recipe:
            del recipe[key]
        elif key in row:
            del row[key]
        else:
            del row["metrics"][key]
    return row


def _manifest_body(payload: bytes, n_tensors: int, **over) -> dict:
    body = {
        "from": SOURCE, "trainer": TRAINER, "device": METAL, "seed": 0,
        "optimizer_step": STEPS, "ft_row_id": METAL_ID, "vocab_size": 8, "span_weight": 1.0,
        "safetensors_sha256": hashlib.sha256(payload).hexdigest(), "n_tensors": n_tensors,
    }
    body.update(over)
    return body


def _export(tmp_path: Path, *, name: str = EXPORT, tensors=None, drop: tuple[str, ...] = (),
            manifest: bool = True, **over) -> Path:
    """A Metal export as the Rust writer lays it out: the safetensors, and its manifest at
    ``<name>.manifest.json``. Stand-in tensors unless ``tensors`` is given."""
    import torch
    from safetensors.torch import save

    if tensors is None:
        tensors = {"tower.w": torch.zeros(4, dtype=torch.bfloat16), "span_head.b": torch.zeros(2)}
    payload = save(dict(tensors))
    out = tmp_path / "export" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    if manifest:
        body = _manifest_body(payload, len(tensors), **over)
        for key in drop:
            del body[key]
        out.with_name(out.name + ".manifest.json").write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return out


def _args(export: Path, ledger: Path, *, seeds=(0,), ft_row_id: str = METAL_ID[:8]):
    return argparse.Namespace(
        score_checkpoint=export, seeds=list(seeds), ft_ledger=ledger, ft_row_id=ft_row_id,
        real_backbone=Path("/models/snap"), score_dtype="fp32",
    )


def _score(args: argparse.Namespace, *, val: object = None):
    return rft._score_checkpoint(
        args, reader=_reader(), val=val, device="cuda", ledger=None,  # type: ignore[arg-type]
        reasons_for=lambda *a, **k: [], second_pass=None,  # type: ignore[arg-type]
        needle_suite=_NO_NEEDLE, ood_suite=_NO_OOD, suite_seed=SUITE_SEED,
    )


# --- routing: by the file's name, as every per-seed checkpoint is ---------------------------


def test_a_metal_export_is_one_seeds_model_and_an_average_keeps_its_route():
    assert not rft._is_average(Path(EXPORT))
    assert not rft._is_average(Path("/runs/x/epoch-seed2-metal.safetensors"))
    # Every average name in this repository's ledgers, HANDOFFs and tests keeps its route.
    for name in ("avg.safetensors", "epoch-avg-seed012-masters.safetensors",
                 "epoch-avgnp-seed012-masters.safetensors", "AVG.safetensors"):
        assert rft._is_average(Path(name)), name


# --- the pairing, before any tower -----------------------------------------------------------


def test_a_well_formed_metal_export_passes_every_pairing_check_and_reaches_the_tower(
    tmp_path, monkeypatch
):
    """The positive control for every refusal below: the fixture, unaltered, reaches the
    step -- at the checkpoint's own seed, the ft row's step count, its recipe and
    --score-dtype -- with the exported weights read and verified."""
    import numpy as np

    export = _export(tmp_path)
    ledger = _ledger(tmp_path, _metal_row())
    seen: dict[str, object] = {}

    def reached(**kwargs):
        seen.update(kwargs)
        raise _Reached

    monkeypatch.setattr(rft, "_real_step", reached)
    val = SimpleNamespace(plan=[SimpleNamespace(tokens=np.zeros((1, 64)))])
    with pytest.raises(_Reached):
        _score(_args(export, ledger), val=val)
    assert seen["seed"] == 0 and seen["total_steps"] == STEPS
    assert seen["dtype"] == "fp32" and seen["span_weight"] == 1.0
    assert seen["attn_implementation"] == "sdpa" and seen["lr"] == 1e-5


def _refusal_case(case: str, tmp_path: Path) -> argparse.Namespace:
    """The ``--score-checkpoint`` arguments of one malformed variant, its files written."""
    row = _metal_row()
    export_kw: dict = {}
    seeds: tuple[int, ...] = (0,)
    if case == "torch_row":
        row = _metal_row(device="cuda", drop=("trainer",))
    elif case == "row_unknown_device":
        row = _metal_row(device="rocm")
    elif case == "row_no_trainer":
        row = _metal_row(drop=("trainer",))
    elif case == "row_other_trainer":
        row = _metal_row(trainer="torch")
    elif case == "row_no_quick":
        row = _metal_row(drop=("quick",))
    elif case == "row_quick_no_reason":
        row = _metal_row(quick_reason=None)
    elif case == "row_quick_not_bool":
        row = _metal_row(quick="yes")
    elif case == "row_no_attn":
        row = _metal_row(drop=("attn_implementation",))
    elif case == "row_no_optimizer_recipe":
        row = _metal_row(drop=("optimizer_recipe",))
    elif case == "row_no_termination":
        row = _metal_row(drop=("train.termination",))
    elif case == "row_other_seed":
        row = _metal_row(seed=1)
    elif case == "row_other_shard_set":
        row = _metal_row(shard_hash="b" * 64)
    elif case == "row_other_backbone":
        row = _metal_row(backbone_snapshot="other")
    elif case == "row_other_steps":
        row = _metal_row(steps=STEPS - 1)
    elif case == "row_memorise_tag":
        row = _metal_row(tag="memorise")
    elif case == "row_no_protocol":
        row = _metal_row(drop=("protocol",))
    elif case == "row_no_recipe":
        row = _metal_row(drop=("recipe",))
    elif case == "average_manifest":
        export_kw = {"from": "masters"}
    elif case == "unknown_source":
        export_kw = {"from": "torch-export"}
    elif case == "no_source":
        export_kw = {"drop": ("from",)}
    elif case == "manifest_torch_device":
        export_kw = {"device": "cuda"}
    elif case == "manifest_unknown_device":
        export_kw = {"device": "rocm"}
    elif case == "manifest_no_device":
        export_kw = {"drop": ("device",)}
    elif case == "manifest_other_trainer":
        export_kw = {"trainer": "torch"}
    elif case == "manifest_no_trainer":
        export_kw = {"drop": ("trainer",)}
    elif case == "manifest_other_seed":
        export_kw = {"seed": 1}
    elif case == "manifest_no_seed":
        export_kw = {"drop": ("seed",)}
    elif case == "manifest_other_step":
        export_kw = {"optimizer_step": STEPS - 1}
    elif case == "manifest_no_step":
        export_kw = {"drop": ("optimizer_step",)}
    elif case == "manifest_other_row":
        export_kw = {"ft_row_id": "3e7a1000-aaaa-4bbb-8ccc-00000000000b"}
    elif case == "manifest_no_row":
        export_kw = {"drop": ("ft_row_id",)}
    elif case == "manifest_no_sha":
        export_kw = {"drop": ("safetensors_sha256",)}
    elif case == "manifest_bad_sha":
        export_kw = {"safetensors_sha256": "not-a-sha"}
    elif case == "manifest_no_count":
        export_kw = {"drop": ("n_tensors",)}
    elif case == "manifest_no_vocab":
        export_kw = {"drop": ("vocab_size",)}
    elif case == "manifest_other_span_weight":
        export_kw = {"span_weight": 0.5}
    elif case == "no_manifest":
        export_kw = {"manifest": False}
    elif case == "memorise_name":
        export_kw = {"name": "memorise-seed0-metal.safetensors"}
    elif case == "unparseable_name":
        export_kw = {"name": "latest-metal.safetensors"}
    elif case == "other_seeds":
        seeds = (1,)
    return _args(_export(tmp_path, **export_kw), _ledger(tmp_path, row), seeds=seeds)


REFUSALS = [
    ("torch_row", "recipe device"),
    ("row_unknown_device", "recipe device"),
    ("row_no_trainer", "trainer"),
    ("row_other_trainer", "trainer"),
    ("row_no_quick", "quick"),
    ("row_quick_no_reason", "quick_reason"),
    ("row_quick_not_bool", "quick"),
    ("row_no_attn", "attn_implementation"),
    ("row_no_optimizer_recipe", "optimizer_recipe"),
    ("row_no_termination", "train.termination"),
    ("row_other_seed", "protocol seed"),
    ("row_other_shard_set", "shard_hash"),
    ("row_other_backbone", "backbone_snapshot"),
    ("row_other_steps", "optimizer step"),
    ("row_memorise_tag", "recipe tag"),
    ("row_no_protocol", "protocol seed"),
    ("row_no_recipe", "no recipe"),
    ("average_manifest", "an average's manifest"),
    ("unknown_source", "'from' is 'torch-export'"),
    ("no_source", "'from'"),
    ("manifest_torch_device", "device"),
    ("manifest_unknown_device", "device"),
    ("manifest_no_device", "device"),
    ("manifest_other_trainer", "trainer"),
    ("manifest_no_trainer", "trainer"),
    ("manifest_other_seed", "seed"),
    ("manifest_no_seed", "seed"),
    ("manifest_other_step", "optimizer step"),
    ("manifest_no_step", "optimizer_step"),
    ("manifest_other_row", "ft_row_id"),
    ("manifest_no_row", "ft_row_id"),
    ("manifest_no_sha", "safetensors_sha256"),
    ("manifest_bad_sha", "safetensors_sha256"),
    ("manifest_no_count", "n_tensors"),
    ("manifest_no_vocab", "vocab_size"),
    ("manifest_other_span_weight", "span_weight"),
    ("no_manifest", "no manifest"),
    ("memorise_name", "only an epoch"),
    ("unparseable_name", "seed"),
    ("other_seeds", "--seeds"),
]


@pytest.mark.parametrize(("case", "match"), REFUSALS)
def test_a_malformed_metal_export_is_refused_before_the_tower(tmp_path, monkeypatch, case, match):
    """Each variant changes one thing the scorer is told about the model, and each is a
    refusal that names the Metal export -- never the average path's or the .json path's
    refusal reached by accident."""
    args = _refusal_case(case, tmp_path)

    def never(**kwargs):
        raise AssertionError("the tower was built for a Metal export that cannot be paired")

    monkeypatch.setattr(rft, "_real_step", never)
    with pytest.raises(SystemExit, match=match) as refused:
        _score(args)
    assert "Metal export" in str(refused.value), str(refused.value)


def test_altered_exported_weights_are_refused(tmp_path, monkeypatch):
    export = _export(tmp_path)
    export.write_bytes(export.read_bytes() + b" ")
    monkeypatch.setattr(rft, "_real_step", lambda **k: pytest.fail("tower built"))
    with pytest.raises(SystemExit, match="safetensors_sha256") as refused:
        _score(_args(export, _ledger(tmp_path, _metal_row())))
    assert "Metal export" in str(refused.value)


def test_an_exported_tensor_outside_the_tower_and_span_head_is_refused(tmp_path, monkeypatch):
    import torch

    export = _export(tmp_path, tensors={
        "tower.w": torch.zeros(4, dtype=torch.bfloat16), "span_head.b": torch.zeros(2),
        "optimizer.exp_avg": torch.zeros(4),
    })
    monkeypatch.setattr(rft, "_real_step", lambda **k: pytest.fail("tower built"))
    with pytest.raises(SystemExit, match="outside") as refused:
        _score(_args(export, _ledger(tmp_path, _metal_row())))
    assert "Metal export" in str(refused.value)


# --- argv: one seed's model, paired with one ft row -----------------------------------------


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["--out", str(tmp_path), "--rev", "0" * 40, "--score-checkpoint",
            str(tmp_path / EXPORT), "--score-val", "--real-backbone", "/x",
            "--ft-ledger", "/l", "--devices", "mps", *extra]


def test_metal_export_argv_is_accepted(tmp_path, monkeypatch):
    """One Metal export, its one ft row and its one seed get past every argv check: the
    first thing after them, resolving --rev, is reached."""

    def reached(*a, **k):
        raise _Reached

    monkeypatch.setattr(rft, "resolve_rev", reached)
    with pytest.raises(_Reached):
        rft.main(_argv(tmp_path, "--ft-row-id", METAL_ID[:8], "--seeds", "0"))


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--seeds", "0"], "needs --ft-row-id"),
        (["--seeds", "0", "1", "--ft-row-id", METAL_ID[:8]], "one seed"),
    ],
)
def test_metal_export_argv_is_refused_before_anything_loads(tmp_path, extra, match):
    with pytest.raises(SystemExit, match=match):
        rft.main(_argv(tmp_path, *extra))


def test_a_score_plan_kind_of_a_metal_export_is_checked_as_one_seeds(tmp_path):
    """A --score-plan kind is checked by the same function, on its own namespace."""
    kind = rft.PlanKind(name="metal", checkpoints=(tmp_path / EXPORT,),
                        ft_row_ids=(METAL_ID,), seeds=(0,), passes=("gates",))
    base = argparse.Namespace(
        score_val=True, real_backbone=Path("/x"), ft_ledger=Path("/l"), devices=["cuda"],
        epoch=False, resume_from=None, score_checkpoint=None, ft_row_id=None, seeds=[],
    )
    rft._check_score_checkpoint_flags(rft.plan_kind_args(base, kind))


def test_a_metal_export_among_ensemble_towers_is_refused(tmp_path):
    """Characterization (unchanged at 6a06bd3): an ensemble's towers are .json checkpoints."""
    base = argparse.Namespace(
        score_val=True, real_backbone=Path("/x"), ft_ledger=Path("/l"), devices=["cuda"],
        epoch=False, resume_from=None, seeds=[0, 1],
        score_checkpoint=[tmp_path / EXPORT, tmp_path / "epoch-seed1-cuda.json"],
        ft_row_id=[METAL_ID, "7f2c11db"],
    )
    with pytest.raises(SystemExit, match="scored on its own"):
        rft._check_score_checkpoint_flags(base)


def test_a_json_checkpoint_named_for_metal_is_not_a_device_this_tool_trains_on(tmp_path):
    """Characterization (unchanged at 6a06bd3): the Rust trainer writes no ``.json``
    ``Checkpoint``; a file spelled as one is refused by the name check."""
    with pytest.raises(ValueError, match="not a device this tool runs"):
        rft._resume_arm(tmp_path / "epoch-seed0-metal.json")


# --- what the rows say about the model --------------------------------------------------------


def _control_args(export: Path, ledger: Path) -> argparse.Namespace:
    return argparse.Namespace(
        **vars(_args(export, ledger)), needle_control=(1024, 2048, 4096),
        usd_per_hour=2.29, usd_per_gpu_hour=None, instance="gh200", wall_clock_cap_s=5400.0,
        suite_logits=False,
    )


@pytest.mark.parametrize("quick", [True, False])
def test_a_metal_exports_needle_control_row_names_its_trainer_and_inherits_quick(
    tmp_path, monkeypatch, quick
):
    """The needle-control row (the same ``recipe_block`` every scored row is built with)
    names the trainer, the device it trained on, the export's source and its manifest; its
    ft row is ONE row; and the row is quick for its ft row when that row is quick -- here
    on a campaign device, where nothing else about the scoring would make it so."""
    export = _export(tmp_path)
    row = _metal_row() if quick else _metal_row(quick=False, quick_reason=None)
    ledger = _ledger(tmp_path, row)
    manifest_file = export.with_name(export.name + ".manifest.json")
    (row_id, _, seed), captured, _ = _run_avg_needle_control(
        monkeypatch, _control_args(export, ledger)
    )
    recipe = captured["recipe"]
    assert row_id == "control-row" and seed == 0 and captured["seed"] == 0
    assert recipe["tag"] == "epoch-needle-length-control"
    assert recipe["trained_by"] == {
        "trainer": TRAINER, "device": METAL, "source": SOURCE,
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    }
    sha = json.loads(manifest_file.read_text(encoding="utf-8"))["safetensors_sha256"]
    assert recipe["scored_checkpoint"] == f"{EXPORT}:{sha}"
    assert "averaged" not in recipe and "ensemble" not in recipe
    assert captured["metrics"]["ft_run_row_id"].value == METAL_ID
    assert "ft_run_row_ids" not in captured["metrics"]
    inherited = [r for r in captured["quick_reasons"] if METAL_QUICK in r]
    assert len(inherited) == (1 if quick else 0), captured["quick_reasons"]
    if quick:
        assert METAL_ID in inherited[0] and TRAINER in inherited[0]
    assert TRAINER in captured["notes"] and METAL in captured["notes"]


# --- end to end on CPU: the same weights, two loaders, one decode ---------------------------


def _write_metal_export(tiny, out_dir: Path) -> tuple[Path, Path]:
    """Seed 0's tiny tower and span head, exported the way the Rust trainer lays its export
    out: ``tower.*`` and ``span_head.*`` at their own dtypes, and the manifest. Returns
    ``(export, ft ledger holding the torch rows and the Metal row)``."""
    from safetensors.torch import save

    from qd_train.backbone import revive_tensors
    from qd_train.run_control import Checkpoint

    weights, header = Checkpoint.read_weights(tiny.paths[0], subtrees=("tower", "span_head"))
    tensors = {
        f"{tree}.{k}": t
        for tree in ("tower", "span_head")
        for k, t in revive_tensors(weights[tree], where=tree).items()
    }
    payload = save(tensors)
    export = out_dir / EXPORT
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_bytes(payload)
    body = _manifest_body(
        payload, len(tensors), optimizer_step=int(header["optimizer_step"]),
        vocab_size=int(weights["vocab_size"]), span_weight=float(weights["span_weight"]),
    )
    export.with_name(export.name + ".manifest.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch_rows = [
        json.loads(line)
        for line in tiny.ft_ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    metal = copy.deepcopy(torch_rows[0])
    metal["row_id"] = METAL_ID
    metal["recipe"].update({"device": METAL, "trainer": TRAINER})
    metal["quick"], metal["quick_reason"] = True, METAL_QUICK
    ledger = out_dir / "ft-with-metal.jsonl"
    ledger.write_text(
        "".join(json.dumps(r) + "\n" for r in (*torch_rows, metal)), encoding="utf-8"
    )
    return export, ledger


def _score_main(tiny, tmp_path: Path, label: str, ft_ledger: Path, checkpoint: Path,
                row_id: str) -> tuple[Path, Path]:
    from test_real_ft_shuffled_label import REV

    ledger = tmp_path / f"eval-{label}.jsonl"
    verdicts = tmp_path / f"verdicts-{label}.jsonl"
    assert rft.main([
        "--out", str(tiny.out), "--rev", REV, "--devices", "cpu", "--seeds", "0",
        "--score-val", "--score-checkpoint", str(checkpoint), "--ft-row-id", row_id,
        "--real-backbone", str(tiny.snapshot), "--ft-ledger", str(ft_ledger),
        "--ledger", str(ledger), "--tokenizer-json", str(tiny.tokenizer),
        "--verdicts-out", str(verdicts),
    ]) == 0
    return ledger, verdicts


def test_a_metal_export_of_a_tiny_tower_is_scored_end_to_end_on_cpu(tmp_path, monkeypatch):
    """Seed 0's tiny weights, scored once from their ``.json`` checkpoint (the torch path)
    and once from a Metal export of the same bytes, through ``main()``: every verdict line
    is equal, so the export loads exactly the model the checkpoint holds. The rows differ
    only in what they say about who trained it: the Metal row carries ``trained_by``, names
    the Metal ft row, inherits that row's quick reason, and hashes apart."""
    from qd_train.ledger import Ledger

    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    export, ft_ledger = _write_metal_export(tiny, tmp_path / "metal")
    torch_ledger, torch_verdicts = _score_main(
        tiny, tmp_path, "torch", ft_ledger, tiny.paths[0], tiny.ids[0]
    )
    metal_ledger, metal_verdicts = _score_main(
        tiny, tmp_path, "metal", ft_ledger, export, METAL_ID
    )

    def lines(path: Path) -> list[dict]:
        out = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = json.loads(raw)
            del line["eval_row_id"]
            out.append(line)
        return out

    assert lines(metal_verdicts) == lines(torch_verdicts) and lines(metal_verdicts)

    (torch_row,) = Ledger(torch_ledger).rows()
    (metal_row,) = Ledger(metal_ledger).rows()
    recipe = dict(metal_row.recipe or {})
    manifest_file = export.with_name(export.name + ".manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert metal_row.run_kind == "eval" and recipe["tag"] == "epoch-score-val"
    assert recipe["trained_by"] == {
        "trainer": TRAINER, "device": METAL, "source": SOURCE,
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    }
    assert recipe["scored_checkpoint"] == f"{EXPORT}:{manifest['safetensors_sha256']}"
    assert recipe["device"] == "cpu" and recipe["score_dtype"] == "fp32"
    assert metal_row.metrics["ft_run_row_id"].value == METAL_ID
    assert metal_row.protocol.seed == 0
    only_the_device = rft.quick_reasons(
        tag="epoch", device="cpu", real_backbone=True,
        corpus=rft.CorpusFacts(snapshot_not_run=None, history_rows={}),
    )
    assert metal_row.quick and metal_row.quick_reason is not None
    reasons = metal_row.quick_reason.split("; ")
    assert reasons[: len(only_the_device)] == only_the_device
    assert any(METAL_QUICK in r and METAL_ID in r for r in reasons), reasons
    assert TRAINER in metal_row.notes and "trained" in metal_row.notes

    torch_recipe = dict(torch_row.recipe or {})
    assert "trained_by" not in torch_recipe
    assert torch_row.metrics["ft_run_row_id"].value == tiny.ids[0]
    assert torch_row.quick_reason == "; ".join(only_the_device)
    assert metal_row.protocol.recipe_hash != torch_row.protocol.recipe_hash
    assert {k: v for k, v in recipe.items() if k not in ("trained_by", "scored_checkpoint")} == {
        k: v for k, v in torch_recipe.items() if k != "scored_checkpoint"
    }
