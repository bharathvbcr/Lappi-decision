"""The scoring footprint (ft-a3b6c9ea916): weights + no-grad activations + the logit slab.

Torch-free, so it runs in the repo venv. A scoring step holds no gradient, no optimizer state
and no master copy, so a budget that prices them refuses work that fits: the 2026-10-06 audit
had a 9B scored fp32 priced as ADAMW_FP32 training (133.4 GiB) against fp32 weights of 33.4 GiB.
The tower-side half (no optimizer object, the verdicts unchanged) is
``test_real_ft_scoring_step.py``.
"""

from __future__ import annotations

import pytest

from qd_train.memory import (
    ADAMW_FP32,
    ADAMW_MASTER,
    ActivationModel,
    ModelSpec,
    estimate_scoring,
    estimate_step,
)

GIB = 1024**3

#: Qwen3.5-9B-Base's text tower (AUDIT/next-train-2026-10-06/sizing.py, from its config.json):
#: 32 layers, every fourth full attention, an untied head. 8,953,803,264 measured parameters.
NINE_B = ModelSpec(
    name="Qwen/Qwen3.5-9B-Base (text tower)",
    hidden_size=4096,
    intermediate_size=12288,
    n_full_attention_layers=8,
    n_linear_attention_layers=24,
    q_heads=16,
    kv_heads=4,
    head_dim=256,
    attn_output_gate=True,
    linear_heads=16,
    linear_head_dim=128,
    linear_value_heads=32,
    linear_value_head_dim=128,
    vocab_size=248_320,
    params_total=8_953_803_264,
    params_embedding=248_320 * 4096,
    tied_embedding=False,
    recurrent_state_bytes=4,
)

FP32 = {"param_dtype": "fp32", "grad_dtype": "fp32", "activation_dtype": "fp32"}


def test_a_scoring_footprint_prices_no_gradient_optimizer_or_master_copy():
    f = estimate_scoring(NINE_B, rows=1, width=2048, param_dtype="fp32", activation_dtype="fp32")
    assert (f.grad_bytes, f.optimizer_bytes, f.master_bytes, f.loss_head_bytes) == (0, 0, 0, 0)
    assert f.param_bytes == NINE_B.trainable_params() * 4
    assert f.purpose == "scoring"


def test_the_scoring_budget_is_weights_plus_no_grad_activations_plus_the_logit_slab():
    f = estimate_scoring(NINE_B, rows=3, width=512, param_dtype="fp32", activation_dtype="fp32")
    # The full-vocabulary head over every position, materialised in the tower's dtype
    # (`QwenDecisionStep.lm_head`; `real_ft_run._readout` takes rows of it).
    assert f.logit_slab_bytes == 3 * 512 * 248_320 * 4
    assert f.total_bytes == f.param_bytes + f.dynamic_bytes + f.logit_slab_bytes
    # No-grad: one layer is live at a time, never every layer's saved set. Even the
    # recompute="none" training enumeration would be 32 layers' worth.
    saved_all = estimate_step(
        NINE_B, rows=3, width=512, activations=ActivationModel(recompute="none"), **FP32
    )
    assert 0 < f.activation_bytes < saved_all.activation_bytes / 8
    assert "logit slab" in f.breakdown()


def test_a_9b_scoring_budget_is_not_refused_for_optimizer_state_alone():
    """The card's number: fp32 weights 33.4 GiB, priced as ADAMW_FP32 training 133.4 GiB."""
    card = 80 * GIB  # one H100
    train = estimate_step(NINE_B, rows=1, width=2048, optimizer=ADAMW_FP32, **FP32)
    master = estimate_step(NINE_B, rows=1, width=2048, optimizer=ADAMW_MASTER, **FP32)
    score = estimate_scoring(
        NINE_B, rows=1, width=2048, param_dtype="fp32", activation_dtype="fp32"
    )
    assert train.total_bytes > card and master.total_bytes > card  # what it was refused at
    assert round(train.param_bytes / GIB, 1) == 33.4
    assert score.fits(device_bytes=card)
    # Everything the scoring estimate drops is what training adds on top of the weights.
    assert score.param_bytes == train.param_bytes
    assert score.total_bytes < train.total_bytes - (train.grad_bytes + train.optimizer_bytes) + 1


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_a_scoring_shape_is_validated_like_a_training_one(bad):
    with pytest.raises(ValueError, match="rows"):
        estimate_scoring(NINE_B, rows=bad, width=8, param_dtype="fp32", activation_dtype="fp32")
    with pytest.raises(ValueError, match="width"):
        estimate_scoring(NINE_B, rows=1, width=bad, param_dtype="fp32", activation_dtype="fp32")


def test_a_dtype_with_no_element_size_is_still_refused_for_scoring():
    with pytest.raises(ValueError, match="param_dtype"):
        estimate_scoring(NINE_B, rows=1, width=8, param_dtype="fp8", activation_dtype="fp32")


def test_a_scoring_footprint_carries_its_purpose_and_a_training_one_still_says_train():
    t = estimate_step(NINE_B, rows=1, width=8, **FP32)
    assert t.purpose == "train" and t.logit_slab_bytes == 0
    assert "logit slab" not in t.breakdown()
