"""``load_text_tower(keep_stored_dtype=...)``: the 36 stored-F32 tensors stay f32 on request.

Fable's board-sweep ruling 2a: Qwen3.5-2B stores exactly 36 F32 tensors (``linear_attn.A_log``
and ``linear_attn.norm.weight`` in each of 18 GDN layers) and ``load_text_tower`` casts every
tensor to the run dtype. v5/v6 keep that cast (bytes unchanged, flag off). LoRA-2B keeps the
36 resident as f32 (they are D3's "norm parameters f32"), and the torch reference arm must
match what ojas's loader does -- so the load gains an opt-in name set, default empty.

Torch-gated like ``test_backbone.py``; run with the ml venv::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python /Users/bharath/.venvs/ml/bin/python \\
        --with pytest --with hypothesis python -m pytest \\
        python/tests/test_backbone_load_dtype.py -o addopts= -q
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")
pytest.importorskip("safetensors", reason="safetensors is an optional 'mac' extra")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_backbone import (  # noqa: E402
    TINY_LAYERS,
    _tiny_spec,
    _tiny_text_config,
)

from qd_train.backbone import (  # noqa: E402
    TEXT_PREFIX,
    BackboneContractViolation,
    GradientCheckpointingDisabled,
    load_text_tower,
)
from qd_train.memory import ADAMW_BF16  # noqa: E402

#: The name suffixes the real checkpoint stores as F32 (ruling 2a, read from the 2B header).
STORED_F32_SUFFIXES = ("linear_attn.A_log", "linear_attn.norm.weight")


def _write_mixed_snapshot(dirpath: Path) -> dict[str, torch.Tensor]:
    """A tiny snapshot stored the way the real one is: bf16 except the GDN A_log and norm."""
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextModel

    text = _tiny_text_config()
    torch.manual_seed(0)
    model = Qwen3_5TextModel(text)
    dirpath.mkdir(parents=True, exist_ok=True)
    Qwen3_5Config(text_config=text.to_dict(), tie_word_embeddings=True).save_pretrained(dirpath)
    stored = {}
    for name, tensor in model.state_dict().items():
        keep_f32 = name.endswith(STORED_F32_SUFFIXES)
        stored[name] = tensor.detach().clone().to(torch.float32 if keep_f32 else torch.bfloat16)
    save_file(
        {f"{TEXT_PREFIX}{k}": v.contiguous() for k, v in stored.items()},
        str(dirpath / "model.safetensors"),
    )
    return stored


def _load(snapshot: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GradientCheckpointingDisabled)
        return load_text_tower(
            snapshot,
            gradient_checkpointing=True,
            optimizer=ADAMW_BF16,
            attn_implementation="sdpa",
            dtype="bf16",
            rows=1,
            width=64,
            spec=_tiny_spec(),
            **kwargs,
        )


def _digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def test_the_fixture_stores_the_gdn_tensors_as_f32(tmp_path):
    stored = _write_mixed_snapshot(tmp_path / "snap")
    f32 = sorted(n for n, t in stored.items() if t.dtype == torch.float32)
    assert len(f32) == 2 * (TINY_LAYERS - TINY_LAYERS // 2)


def test_without_the_flag_every_tensor_is_cast_to_the_run_dtype(tmp_path):
    """v5/v6 bytes: the default load is unchanged, and its state digest is pinned to the cast."""
    stored = _write_mixed_snapshot(tmp_path / "snap")
    state = _load(tmp_path / "snap").model.state_dict()
    assert {t.dtype for t in state.values()} == {torch.bfloat16}
    expected = {k: v.to(torch.bfloat16) for k, v in stored.items()}
    assert _digest(state) == _digest(expected)


def test_with_the_flag_the_named_stored_f32_tensors_come_back_float32(tmp_path):
    stored = _write_mixed_snapshot(tmp_path / "snap")
    names = frozenset(n for n, t in stored.items() if t.dtype == torch.float32)
    state = _load(tmp_path / "snap", keep_stored_dtype=names).model.state_dict()
    for name, tensor in state.items():
        want = torch.float32 if name in names else torch.bfloat16
        assert tensor.dtype == want, name
        assert torch.equal(tensor, stored[name].to(want)), name


def test_a_kept_tensor_is_bit_identical_to_the_stored_f32_not_a_bf16_round_trip(tmp_path):
    stored = _write_mixed_snapshot(tmp_path / "snap")
    name = next(n for n in stored if n.endswith("linear_attn.A_log"))
    # Fixture A_log is exp-uniform noise, so a bf16 round trip would change it.
    assert not torch.equal(stored[name], stored[name].to(torch.bfloat16).to(torch.float32))
    state = _load(tmp_path / "snap", keep_stored_dtype={name}).model.state_dict()
    assert torch.equal(state[name], stored[name])


def test_a_name_the_checkpoint_does_not_have_is_refused(tmp_path):
    _write_mixed_snapshot(tmp_path / "snap")
    with pytest.raises(BackboneContractViolation, match=r"layers\.99\.linear_attn\.A_log"):
        _load(tmp_path / "snap", keep_stored_dtype={"layers.99.linear_attn.A_log"})
