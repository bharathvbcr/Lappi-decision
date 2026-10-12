"""``tools/ckpt_average.py --from lora-merged`` and the refusal of an adapter-level average.

Fable D5 and the board-sweep ruling 2h: the release of a LoRA-2B model is a merged dense tower
(``W' = W + (alpha/r) B A``, computed in f32, rounded to the tower dtype once), and a seed average
is an average of merged weights. ``mean(B) mean(A) != mean(B A)``, so averaging A and B separately
is refused with a typed error, and a checkpoint that carries adapter state is refused by name
by every other source rather than averaged with its adapters silently dropped.

The LoRA checkpoint body this tool reads (the reference arm's layout, ``qd_train.lora``):
``model_state`` holds the frozen base ``tower``, ``lora_A`` and ``lora_B`` (module name ->
f32 ``[r, in]`` / ``[out, r]``), the scalars ``lora_r`` and ``lora_alpha``, the ``span_head`` and an
optional ``d17``.
"""

from __future__ import annotations

import json
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
MODULE = "layers.0.mlp.gate_proj"
OUT, IN, R, ALPHA = 6, 8, 2, 6  # scaling 3.0


def _ref(t: torch.Tensor):
    from qd_train.run_control import TensorRef

    dtype = str(t.dtype).removeprefix("torch.")
    return TensorRef(dtype, tuple(t.shape), t.contiguous().reshape(-1).view(torch.uint8)
                     .numpy().tobytes())


def _base() -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return (torch.randn(OUT, IN, generator=g) * 0.5).to(torch.bfloat16)


def _ckpt(tmp_path: Path, seed: int, *, base=None, r=R, alpha=ALPHA, d17=True, adapters=True,
          head=None, extra_tower=None, name=None) -> Path:
    from qd_train.run_control import Checkpoint, LossLog, Position

    g = torch.Generator().manual_seed(100 + seed)
    base = _base() if base is None else base
    tower = {f"{MODULE}.weight": _ref(base), "norm.weight": _ref(torch.ones(OUT))}
    tower.update(extra_tower or {})
    state: dict = {
        "tower": tower,
        "span_head": {"b": _ref(torch.tensor(head or [float(seed), 1.0]))},
        "span_weight": 1.0, "vocab_size": 8,
    }
    if adapters:
        state.update({
            "lora_A": {MODULE: _ref(torch.randn(r, IN, generator=g))},
            "lora_B": {MODULE: _ref(torch.randn(OUT, r, generator=g))},
            "lora_r": r, "lora_alpha": alpha,
        })
    if d17:
        state["d17"] = {"delta": _ref(torch.full((3,), float(seed)))}
    path = tmp_path / "ckpt" / (name or f"lora-seed{seed}.json")
    Checkpoint(
        position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=seed,
        schedule=_schedule(), loss_log=LossLog().snapshot(),
        consumed_digest=f"{seed + 1:064x}", model_state=state,
    ).write(path)
    return path


def _tensor(ref) -> torch.Tensor:
    return torch.frombuffer(bytearray(ref.data), dtype=getattr(torch, ref.dtype)).reshape(ref.shape)


def _read(path: Path, key: str):
    from qd_train.run_control import Checkpoint

    weights, _ = Checkpoint.read_weights(path, subtrees=(key,))
    return weights[key]


def _merged_oracle(paths: list[Path], scaling: float = ALPHA / R) -> torch.Tensor:
    acc = torch.zeros(OUT, IN, dtype=torch.float64)
    for p in paths:
        a = _tensor(_read(p, "lora_A")[MODULE])
        b = _tensor(_read(p, "lora_B")[MODULE])
        w = _tensor(_read(p, "tower")[f"{MODULE}.weight"]).to(torch.float32)
        acc += (w + scaling * (b @ a)).to(torch.float64)
    return (acc / len(paths)).to(torch.float32).to(torch.bfloat16)


def _run(tmp_path: Path, paths: list[Path], *extra: str, source: str = "lora-merged"):
    out = tmp_path / "avg" / "m.safetensors"
    code = ckpt_average.main([*map(str, paths), "--out", str(out), "--from", source, *extra])
    assert code == 0
    from safetensors.torch import load_file

    manifest = json.loads(ckpt_average.manifest_path(out).read_text(encoding="utf-8"))
    return manifest, load_file(str(out))


def _three(tmp_path: Path, **kw) -> list[Path]:
    return [_ckpt(tmp_path, s, **kw) for s in (0, 1, 2)]


def test_the_merged_average_is_the_mean_of_w_plus_scaled_ba_rounded_once(tmp_path):
    paths = _three(tmp_path)
    manifest, out = _run(tmp_path, paths)
    want = _merged_oracle(paths)
    assert out[f"tower.{MODULE}.weight"].dtype == torch.bfloat16
    assert torch.equal(out[f"tower.{MODULE}.weight"], want)
    assert torch.equal(out["tower.norm.weight"], torch.ones(OUT))  # unadapted: copied
    assert torch.equal(out["span_head.b"], torch.tensor([1.0, 1.0]))
    assert manifest["from"] == "lora-merged"
    assert manifest["tensor_sources"][f"tower.{MODULE}.weight"] == "lora-merged"
    assert manifest["tensor_sources"]["tower.norm.weight"] == "tower"
    assert manifest["lora"]["r"] == R and manifest["lora"]["alpha"] == ALPHA
    assert manifest["lora"]["scaling"] == ALPHA / R and manifest["lora"]["adapted_modules"] == 1
    assert manifest["span_head_source"] == {
        "mode": "averaged", "seed": None, "input": None, "payload_digest": None,
    }


def test_it_is_not_the_average_of_a_and_b_taken_separately(tmp_path):
    """The reason for the refusal: the product of means is a different weight."""
    paths = _three(tmp_path)
    a = torch.stack([_tensor(_read(p, "lora_A")[MODULE]) for p in paths]).mean(0)
    b = torch.stack([_tensor(_read(p, "lora_B")[MODULE]) for p in paths]).mean(0)
    naive = (_tensor(_read(paths[0], "tower")[f"{MODULE}.weight"]).to(torch.float32)
             + (ALPHA / R) * (b @ a)).to(torch.bfloat16)
    assert not torch.equal(naive, _merged_oracle(paths))
    _, out = _run(tmp_path, paths)
    assert not torch.equal(out[f"tower.{MODULE}.weight"], naive)


def test_d17_is_averaged_plainly_and_written_as_its_own_tree(tmp_path):
    _, out = _run(tmp_path, _three(tmp_path))
    assert torch.equal(out["d17.delta"], torch.full((3,), 1.0))


def test_d17_in_some_inputs_and_not_others_is_refused(tmp_path):
    paths = [_ckpt(tmp_path, 0), _ckpt(tmp_path, 1, d17=False), _ckpt(tmp_path, 2)]
    with pytest.raises(SystemExit, match="d17"):
        _run(tmp_path, paths)


@pytest.mark.parametrize("source", ["tower", "masters"])
def test_every_other_source_refuses_an_adapter_checkpoint_by_name(tmp_path, source):
    with pytest.raises(SystemExit, match=r"adapter.*--from lora-merged"):
        _run(tmp_path, _three(tmp_path), source=source)


def test_the_trajectory_average_refuses_it_too(tmp_path):
    with pytest.raises(SystemExit, match="adapter"):
        _run(tmp_path, _three(tmp_path), "--same-seed-trajectory", source="tower")


def test_the_typed_error_is_an_average_refusal_named_for_the_adapter_level_average(tmp_path):
    from qd_train.run_control import Checkpoint

    ckpts = [Checkpoint.read(p) for p in _three(tmp_path)]
    with pytest.raises(ckpt_average.AdapterLevelAverageRefused) as err:
        ckpt_average.average(ckpts, [str(p) for p in _three(tmp_path)])
    assert isinstance(err.value, ckpt_average.AverageRefusal)


def test_adapter_leaves_written_into_the_tower_tree_are_refused_in_every_mode(tmp_path):
    leaf = {f"{MODULE}.lora_A.weight": _ref(torch.zeros(R, IN))}
    paths = [_ckpt(tmp_path, s, adapters=False, d17=False, extra_tower=leaf) for s in (0, 1)]
    with pytest.raises(SystemExit, match="adapter"):
        _run(tmp_path, paths, source="tower")
    with pytest.raises(SystemExit, match="adapter"):
        _run(tmp_path, paths)


def test_merged_mode_refuses_a_checkpoint_with_no_adapters(tmp_path):
    paths = [_ckpt(tmp_path, 0), _ckpt(tmp_path, 1, adapters=False, d17=False)]
    with pytest.raises(SystemExit, match="lora_A"):
        _run(tmp_path, paths)


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"alpha": 8}, "lora_alpha"),
        ({"r": 4}, "lora_r"),
    ],
)
def test_inputs_must_agree_on_r_and_alpha(tmp_path, kw, needle):
    paths = [_ckpt(tmp_path, 0), _ckpt(tmp_path, 1, **kw)]
    with pytest.raises(SystemExit, match=needle):
        _run(tmp_path, paths)


def test_inputs_must_share_one_frozen_base(tmp_path):
    other = _base() + 1
    paths = [_ckpt(tmp_path, 0), _ckpt(tmp_path, 1, base=other.to(torch.bfloat16))]
    with pytest.raises(SystemExit, match="frozen base"):
        _run(tmp_path, paths)


def test_a_misshapen_adapter_is_refused(tmp_path):
    from qd_train.run_control import Checkpoint, LossLog, Position

    good = _ckpt(tmp_path, 0)
    bad_path = tmp_path / "ckpt" / "bad.json"
    state = {
        "tower": {f"{MODULE}.weight": _ref(_base()), "norm.weight": _ref(torch.ones(OUT))},
        "span_head": {"b": _ref(torch.tensor([1.0, 1.0]))}, "span_weight": 1.0, "vocab_size": 8,
        "lora_A": {MODULE: _ref(torch.randn(R, IN + 1))},
        "lora_B": {MODULE: _ref(torch.randn(OUT, R))}, "lora_r": R, "lora_alpha": ALPHA,
    }
    Checkpoint(
        position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=1,
        schedule=_schedule(), loss_log=LossLog().snapshot(), consumed_digest=f"{2:064x}",
        model_state=state,
    ).write(bad_path)
    with pytest.raises(SystemExit, match=r"\[8, 6\]|shape"):
        _run(tmp_path, [good, bad_path])


def test_an_adapter_for_a_module_the_tower_lacks_is_refused(tmp_path):
    from qd_train.run_control import Checkpoint, LossLog, Position

    paths = []
    for seed in (0, 1):
        path = tmp_path / "ckpt" / f"ghost{seed}.json"
        state = {
            "tower": {"norm.weight": _ref(torch.ones(OUT))},
            "span_head": {"b": _ref(torch.tensor([1.0, 1.0]))}, "span_weight": 1.0,
            "vocab_size": 8,
            "lora_A": {MODULE: _ref(torch.randn(R, IN))},
            "lora_B": {MODULE: _ref(torch.randn(OUT, R))}, "lora_r": R, "lora_alpha": ALPHA,
        }
        Checkpoint(
            position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=seed,
            schedule=_schedule(), loss_log=LossLog().snapshot(),
            consumed_digest=f"{seed + 1:064x}", model_state=state,
        ).write(path)
        paths.append(path)
    with pytest.raises(SystemExit, match=r"gate_proj\.weight"):
        _run(tmp_path, paths)


def test_the_span_head_seed_applies_to_a_lora_average_too(tmp_path):
    paths = [_ckpt(tmp_path, s, head=[float(10 * s), 1.0]) for s in (0, 1, 2)]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "".join(
            json.dumps({"row_id": f"row-{s}", "run_kind": "ft", "status": "completed",
                        "protocol": {"seed": s}, "recipe": {}}) + "\n"
            for s in (0, 1, 2)
        ),
        encoding="utf-8",
    )
    manifest, out = _run(tmp_path, paths, "--span-head-seed", "2", "--ft-ledger", str(ledger),
                         "--ft-row-ids", "row-0", "row-1", "row-2")
    assert torch.equal(out["span_head.b"], torch.tensor([20.0, 1.0]))
    assert manifest["span_head_source"]["mode"] == "seed"
    assert manifest["span_head_source"]["seed"] == 2


def test_a_scored_lora_average_with_d17_is_refused_by_the_full_ft_scorer(tmp_path):
    """real_ft_run's --score-checkpoint reads a tower and a span head and has no D17 readout."""
    paths = _three(tmp_path)
    _, _ = _run(tmp_path, paths, "--ft-row-ids", "a", "b", "c")
    manifest = ckpt_average.read_manifest(tmp_path / "avg" / "m.safetensors")
    ckpt_average.verify_sources(manifest)
    with pytest.raises(ckpt_average.AverageRefusal, match="d17"):
        ckpt_average.read_average(manifest)


def _rewrite(tmp_path: Path, source: Path, name: str, seed: int, **state_overrides) -> Path:
    from qd_train.run_control import Checkpoint, LossLog, Position

    weights, _ = Checkpoint.read_weights(
        source, subtrees=("tower", "span_head", "lora_A", "lora_B", "d17")
    )
    path = tmp_path / "ckpt" / name
    Checkpoint(
        position=Position(epoch=0, index=STEPS), optimizer_step=STEPS, seed=seed,
        schedule=_schedule(), loss_log=LossLog().snapshot(),
        consumed_digest=f"{seed + 1:064x}",
        model_state={**weights, "lora_r": R, "lora_alpha": ALPHA, **state_overrides},
    ).write(path)
    return path


def test_one_run_named_twice_by_its_adapters_is_refused(tmp_path):
    """Same adapters under two seeds would weight one run double while the manifest counted two."""
    first = _ckpt(tmp_path, 0)
    twin = _rewrite(tmp_path, first, "twin.json", seed=1)
    with pytest.raises(SystemExit, match="identical adapters"):
        _run(tmp_path, [first, twin])


def test_a_malformed_scale_is_refused(tmp_path):
    first = _ckpt(tmp_path, 0)
    bad = _rewrite(tmp_path, _ckpt(tmp_path, 1), "bad.json", seed=1, lora_r=True)
    with pytest.raises(SystemExit, match="lora_r"):
        _run(tmp_path, [first, bad])
