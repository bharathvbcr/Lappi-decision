"""``tools/gh200_rows.py`` measures the base and recipe it is told to, and with no flags the one
it always measured.

The defect pinned here (DevMap audit #6, 2026-10-06): the row sweep was hard-wired to the 2B
snapshot and the plain bf16 recipe, so the next plan's P2 -- the 4B under the compensated
recipe -- had no instrument. The sweep itself needs CUDA; what it measures is decided on the
CPU, by ``sweep_from_args``, and that is what these check.

Torch-gated (the tool imports torch and ``qd_train.backbone`` at module scope)::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project \\
        --python /Users/bharath/.venvs/ml/bin/python --with pytest \\
        python -m pytest python/tests/test_gh200_rows.py -o addopts= -q
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import gh200_rows  # noqa: E402
import real_ft_run  # noqa: E402

from qd_train.memory import ADAMW_BF16, QWEN3_5_2B_TEXT, CheckpointRefused  # noqa: E402


def test_no_flags_is_the_sweep_it_always_ran() -> None:
    s = gh200_rows.sweep_from_args([])
    assert s.snapshot == gh200_rows.SNAPSHOT
    assert s.model is QWEN3_5_2B_TEXT
    assert (s.recipe, s.optimizer) == ("bf16", ADAMW_BF16)
    assert s.cases == ((34522, "the widest real bucket"), (8192, "a mid bucket"))
    assert s.out == Path("/home/ubuntu/gh200_rows.json")


@pytest.mark.parametrize("recipe", sorted(gh200_rows.RECIPES))
def test_each_recipe_is_the_spec_the_trainer_builds_under_that_name(recipe: str) -> None:
    """One owner per name: the sweep measures what ``real_ft_run.py --optimizer`` trains."""
    s = gh200_rows.sweep_from_args(["--optimizer", recipe])
    assert s.optimizer is real_ft_run.optimizer_spec("bf16", recipe)
    assert set(gh200_rows.RECIPES) <= set(real_ft_run.OPTIMIZER_RECIPES)


def test_widths_replace_the_default_cases_in_order() -> None:
    s = gh200_rows.sweep_from_args(["--width", "9638", "--width", "4096"])
    assert s.cases == ((9638, "--width"), (4096, "--width"))


@pytest.mark.parametrize(
    "argv",
    [
        ["--width", "0"],
        ["--width", "-9638"],
        ["--width", "wide"],
        ["--optimizer", "fp32"],
        ["--optimizer", "adamw"],
    ],
)
def test_a_sweep_that_describes_no_real_run_is_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        gh200_rows.sweep_from_args(argv)


def _four_b_snapshot(root: Path, *, tie: bool | None) -> Path:
    snap = root / "models--Qwen--Qwen3.5-4B-Base" / "snapshots" / "f00d"
    snap.mkdir(parents=True)
    text = {
        "hidden_size": 2560, "intermediate_size": 9216, "num_attention_heads": 16,
        "num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128, "vocab_size": 248_320,
        "mamba_ssm_dtype": "float32",
        "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8,
    }
    config: dict[str, object] = {"text_config": text}
    if tie is not None:
        config["tie_word_embeddings"] = tie
    (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header = {
        "model.language_model.embed_tokens.weight": {
            "dtype": "BF16", "shape": [248_320, 2560], "data_offsets": [0, 0]
        }
    }
    raw = json.dumps(header).encode()
    (snap / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return snap


def test_another_snapshot_is_budgeted_from_its_own_headers(tmp_path: Path) -> None:
    """A 2B spec over a 4B tower is the pair load_text_tower refuses; the sweep reads the
    4B's own shape, value heads included."""
    snap = _four_b_snapshot(tmp_path, tie=True)
    s = gh200_rows.sweep_from_args(["--snapshot", str(snap), "--optimizer", "kahan"])
    assert s.model.name == "Qwen/Qwen3.5-4B-Base (text tower)"
    assert (s.model.hidden_size, s.model.linear_value_heads) == (2560, 32)
    assert s.recipe == "kahan"


def test_a_snapshot_whose_head_is_a_guess_is_refused_before_any_gpu_work(tmp_path: Path) -> None:
    snap = _four_b_snapshot(tmp_path, tie=None)
    with pytest.raises(CheckpointRefused, match="no top-level"):
        gh200_rows.sweep_from_args(["--snapshot", str(snap)])


def test_the_loss_defaults_to_the_proxy_and_fused_ce_is_a_choice() -> None:
    assert gh200_rows.sweep_from_args([]).loss == "proxy"
    assert gh200_rows.sweep_from_args(["--loss", "fused-ce"]).loss == "fused-ce"
    with pytest.raises(SystemExit):
        gh200_rows.sweep_from_args(["--loss", "mse"])


def test_fused_ce_drives_the_loss_head_the_proxy_never_builds(tmp_path: Path) -> None:
    """The point of ``--loss fused-ce``: the tied head receives a gradient, so the step's
    peak includes the loss head ``estimate_step`` budgets. Under the proxy it receives
    none -- which is why the GH200's 0.84-0.97x figure is the backbone's alone. Run on a
    tiny CPU tower through the same function the GPU probe calls."""
    pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")
    from test_backbone import _tiny_tower

    grads = {}
    for loss in gh200_rows.LOSSES:
        tower, _ = _tiny_tower(tmp_path / loss)
        ids = torch.randint(0, tower.vocab_size, (2, 16))
        value = gh200_rows.forward_backward(tower, ids, loss=loss)
        assert value == value and value > 0, (loss, value)
        head = tower.lm_head_weight.grad
        # The embedding row lookups give the head a sparse gradient either way; the loss
        # head gives every row one.
        grads[loss] = int((head.abs().sum(dim=1) > 0).sum()) if head is not None else 0
    assert grads["fused-ce"] == tower.vocab_size
    assert grads["proxy"] < tower.vocab_size
    with pytest.raises(ValueError, match="no loss"):
        gh200_rows.forward_backward(tower, ids, loss="mse")
