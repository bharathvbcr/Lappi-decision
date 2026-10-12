"""``real_ft_run.OPTIMIZER_RECIPES`` knows ``"f32"`` and ``"lora"``; ``--score-checkpoint`` records
an average's ``span_head_source`` (Fable's board-sweep ruling 2g and 2h).

* ``"f32"`` labels the Metal full-fine-tune path (AdamW over f32 parameters, no master/bf16 split)
  and ``"lora"`` labels LoRA-2B (D6). Both are labels a ledger row carries and a reader must know;
  neither is built by this torch tool's bf16 steps, so ``optimizer_spec`` refuses them at bf16
  rather than quietly building another recipe's layout, and ``--optimizer`` does not offer them.
* A Mac row is never pooled with ``master``: ``optimizer_recipe`` is in the recipe that
  ``recipe_hash`` is the sha256 of.
* The averaged-model block of a scored row carries the manifest's ``span_head_source``, so a
  primary (averaged head) and a fallback (one seed's head) row differ in the row itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run  # noqa: E402

from qd_train.memory import ADAMW_FP32  # noqa: E402


def test_f32_and_lora_are_recipe_labels_and_master_is_still_one():
    assert set(real_ft_run.OPTIMIZER_RECIPES) == {"bf16", "master", "kahan", "f32", "lora"}


@pytest.mark.parametrize("label", ["f32", "lora"])
def test_the_new_labels_are_not_built_by_the_bf16_torch_steps(label):
    assert real_ft_run.optimizer_spec("fp32", label) is ADAMW_FP32
    with pytest.raises(ValueError, match="not built by this tool"):
        real_ft_run.optimizer_spec("bf16", label)
    assert label not in real_ft_run.TORCH_OPTIMIZER_RECIPES
    assert set(real_ft_run.TORCH_OPTIMIZER_RECIPES) == {"bf16", "master", "kahan"}


def test_a_lora_or_f32_row_never_shares_a_recipe_hash_with_master():
    reader = SimpleNamespace(header=SimpleNamespace(data_snapshot_hash="a" * 64,
                                                    tokenizer_hash="b" * 64))
    hashes = {
        label: real_ft_run._protocol(
            reader=reader, seed=0, recipe={"hidden": 8, "heads": 2, "optimizer_recipe": label}
        ).recipe_hash
        for label in ("master", "f32", "lora", "kahan", "bf16")
    }
    assert len(set(hashes.values())) == 5


def test_the_averaged_meta_carries_the_manifests_span_head_source():
    seed_mode = {"mode": "seed", "seed": 1, "input": "/x/a.json", "payload_digest": "c" * 64}
    manifest = SimpleNamespace(span_head_source=seed_mode)
    assert real_ft_run._span_head_source_meta(manifest) == {"span_head_source": seed_mode}
    legacy = SimpleNamespace(span_head_source=None)  # an average that predates the key
    assert real_ft_run._span_head_source_meta(legacy) == {}
    assert real_ft_run._span_head_source_meta(manifest)["span_head_source"] is not seed_mode
