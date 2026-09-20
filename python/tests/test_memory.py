"""`qd_train.memory`: the arithmetic that says whether a step fits, before it runs.

Torch-free, so every one of these runs in the repo venv and is counted by ``make gates`` --
which is the point. The failure this module replaces happens on a rented GPU; the check for
it must not.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from qd_train.memory import (
    ADAMW_FP32,
    BYTES_PER_ELEMENT,
    MAX_SAFETY_FRACTION,
    QWEN3_5_2B_TEXT,
    ActivationModel,
    MemoryRefused,
    ModelSpec,
    OptimizerSpec,
    estimate_step,
    max_positions_that_fit,
    refuse_unless_it_fits,
)

GB = 1000**3
M = QWEN3_5_2B_TEXT
REMAP_VOCAB = 13_787

BF16 = {"param_dtype": "bf16", "grad_dtype": "bf16", "activation_dtype": "bf16"}

SNAPSHOTS = Path(
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots"
)


# --- the constant against the checkpoint it claims to describe ---------------------------


def _local_snapshot() -> Path | None:
    if not SNAPSHOTS.is_dir():
        return None
    found = sorted(p for p in SNAPSHOTS.glob("*") if p.is_dir())
    return found[-1] if found else None


def test_the_model_spec_matches_the_local_checkpoints_own_tensor_shapes():
    """The constant was measured once; this re-measures, so drift is caught not inherited."""
    snap = _local_snapshot()
    if snap is None:  # pragma: no cover - depends on the host's cache
        pytest.skip("no local Qwen3.5-2B-Base snapshot to check the constant against")

    header: dict = {}
    for shard in sorted(snap.glob("*.safetensors")):
        with shard.open("rb") as fh:
            length = struct.unpack("<Q", fh.read(8))[0]
            header.update(json.loads(fh.read(length)))
    header.pop("__metadata__", None)
    if not header:  # pragma: no cover - depends on the host's cache
        pytest.skip("the local snapshot carries no safetensors shards")

    def numel(spec: dict) -> int:
        n = 1
        for dim in spec["shape"]:
            n *= dim
        return n

    text = sum(
        numel(v) for k, v in header.items() if not k.startswith(("model.visual.", "mtp."))
    )
    assert text == M.params_total, (
        f"the checkpoint holds {text:,} text-tower parameters but ModelSpec says "
        f"{M.params_total:,}; a budget built on a stale count is a budget for another model"
    )

    cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))["text_config"]
    assert M.hidden_size == cfg["hidden_size"]
    assert M.intermediate_size == cfg["intermediate_size"]
    assert M.vocab_size == cfg["vocab_size"]
    assert M.q_heads == cfg["num_attention_heads"]
    assert M.kv_heads == cfg["num_key_value_heads"]
    assert M.head_dim == cfg["head_dim"]
    assert M.attn_output_gate == cfg["attn_output_gate"]
    assert M.n_full_attention_layers == cfg["layer_types"].count("full_attention")
    assert M.n_linear_attention_layers == cfg["layer_types"].count("linear_attention")
    assert M.params_embedding == cfg["vocab_size"] * cfg["hidden_size"]
    # config.json says the SSM state is float32; assuming the compute dtype halves it.
    assert (M.recurrent_state_bytes == 4) == (cfg.get("mamba_ssm_dtype") == "float32")


def _with(spec: ModelSpec, **changes: object) -> ModelSpec:
    """A copy of ``spec`` with fields replaced. ``ModelSpec`` is frozen and slotted."""
    fields = {f: getattr(spec, f) for f in spec.__slots__}  # type: ignore[attr-defined]
    fields.update(changes)
    return ModelSpec(**fields)  # type: ignore[arg-type]


def test_the_q_projection_carries_a_gate_and_the_arithmetic_knows_it():
    """`q_proj` is [4096, 2048] where q_heads x head_dim is 2048. Doubling is not optional."""
    assert M.attn_output_gate is True
    acts = ActivationModel()
    gated = acts.full_layer_elements(M)
    ungated = acts.full_layer_elements(_with(M, attn_output_gate=False))
    assert gated - ungated == M.q_heads * M.head_dim, (
        "the gate is exactly one q_heads x head_dim tensor per token; a model spec that "
        "ignores attn_output_gate understates the query activation by that much"
    )


# --- the remap is a quarter of the model -------------------------------------------------


def test_the_remap_removes_the_embedding_and_the_budget_sees_it():
    full = M.trainable_params()
    remapped = M.trainable_params(vocab_size=REMAP_VOCAB)
    assert full == M.params_total
    assert remapped == M.params_total - M.params_embedding + REMAP_VOCAB * M.hidden_size
    removed = full - remapped
    assert removed == 480_323_584, removed
    assert 0.25 < removed / full < 0.26, "the tied embedding is ~25.5% of this model"


def test_a_nonsense_vocabulary_is_refused_rather_than_budgeted():
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            M.trainable_params(vocab_size=bad)  # type: ignore[arg-type]


# --- the static term ---------------------------------------------------------------------


def test_adamw_keeps_two_states_per_parameter_and_the_spec_says_so():
    """Read from the code, not assumed: every optimizer this repo builds is torch AdamW."""
    assert ADAMW_FP32.states_per_param == 2
    assert ADAMW_FP32.state_bytes == 4
    assert ADAMW_FP32.keeps_fp32_master is False
    assert ADAMW_FP32.bytes_per_param(param_bytes=2, grad_bytes=2) == 12
    assert ADAMW_FP32.bytes_per_param(param_bytes=4, grad_bytes=4) == 16


def test_an_fp32_master_copy_costs_four_more_bytes_per_parameter():
    plain = ADAMW_FP32
    master = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    assert (
        master.bytes_per_param(param_bytes=2, grad_bytes=2)
        - plain.bytes_per_param(param_bytes=2, grad_bytes=2)
        == 4
    )
    a = estimate_step(M, rows=1, width=1024, optimizer=plain, vocab_size=REMAP_VOCAB, **BF16)
    b = estimate_step(M, rows=1, width=1024, optimizer=master, vocab_size=REMAP_VOCAB, **BF16)
    assert b.master_bytes == 4 * a.trainable_params
    assert a.master_bytes == 0
    assert b.static_bytes - a.static_bytes == b.master_bytes


def test_the_static_term_does_not_move_with_the_batch_shape():
    small = estimate_step(M, rows=1, width=479, vocab_size=REMAP_VOCAB, **BF16)
    large = estimate_step(M, rows=8, width=34_522, vocab_size=REMAP_VOCAB, **BF16)
    assert small.static_bytes == large.static_bytes
    assert large.dynamic_bytes > small.dynamic_bytes


def test_every_component_sums_to_the_total_with_nothing_folded_in():
    f = estimate_step(M, rows=3, width=8192, vocab_size=REMAP_VOCAB, **BF16)
    assert f.total_bytes == (
        f.param_bytes
        + f.grad_bytes
        + f.optimizer_bytes
        + f.master_bytes
        + f.activation_bytes
        + f.score_matrix_bytes
        + f.recurrent_state_bytes
        + f.safety_bytes
        + f.loss_head_bytes
    )
    assert f.total_bytes == f.static_bytes + f.dynamic_bytes + f.loss_head_bytes


# --- the dynamic term --------------------------------------------------------------------


def test_activations_are_linear_in_positions():
    one = estimate_step(M, rows=1, width=4096, vocab_size=REMAP_VOCAB, **BF16)
    four = estimate_step(M, rows=4, width=4096, vocab_size=REMAP_VOCAB, **BF16)
    assert four.activation_bytes == 4 * one.activation_bytes


def test_gradient_checkpointing_is_what_makes_the_widest_bucket_fit():
    """The decisive arithmetic: 34,522 on one 96 GiB card, with and without recompute."""
    device = 96 * GB
    on = estimate_step(
        M,
        rows=1,
        width=34_522,
        vocab_size=REMAP_VOCAB,
        activations=ActivationModel(recompute="full", attention="flash"),
        **BF16,
    )
    off = estimate_step(
        M,
        rows=1,
        width=34_522,
        vocab_size=REMAP_VOCAB,
        activations=ActivationModel(recompute="none", attention="flash"),
        **BF16,
    )
    assert on.fits(device_bytes=device)
    assert not off.fits(device_bytes=device)
    assert off.activation_bytes > 9 * on.activation_bytes, (
        "24 layers retained against one recomputed layer plus the boundaries"
    )


def test_the_materialised_score_matrix_is_quadratic_and_flash_removes_it():
    math_acts = ActivationModel(recompute="full", attention="math")
    flash_acts = ActivationModel(recompute="full", attention="flash")
    narrow = estimate_step(
        M, rows=1, width=8192, vocab_size=REMAP_VOCAB, activations=math_acts, **BF16
    )
    wide = estimate_step(
        M, rows=1, width=16384, vocab_size=REMAP_VOCAB, activations=math_acts, **BF16
    )
    assert wide.score_matrix_bytes == 4 * narrow.score_matrix_bytes, "doubling L quadruples it"
    flash = estimate_step(
        M, rows=1, width=16384, vocab_size=REMAP_VOCAB, activations=flash_acts, **BF16
    )
    assert flash.score_matrix_bytes == 0


def test_the_loss_head_is_a_fixed_cost_not_a_per_position_one():
    """`fused_ce` bounds the logit slab, so the head must not scale with sequence length."""
    a = estimate_step(M, rows=1, width=512, vocab_size=REMAP_VOCAB, **BF16)
    b = estimate_step(M, rows=1, width=34_522, vocab_size=REMAP_VOCAB, **BF16)
    assert a.loss_head_bytes == b.loss_head_bytes
    # And it shrinks with the remap, because the fp32 accumulator is the size of the weight.
    full = estimate_step(M, rows=1, width=512, **BF16)
    assert full.loss_head_bytes > 8 * a.loss_head_bytes


# --- refusing ----------------------------------------------------------------------------


def test_an_unsupplied_device_budget_refuses_rather_than_passing():
    """The rule this module exists for: could-not-check must not read as checked."""
    f = estimate_step(M, rows=1, width=479, vocab_size=REMAP_VOCAB, **BF16)
    with pytest.raises(MemoryRefused, match="never checked"):
        refuse_unless_it_fits(f, device_bytes=None, where="an unbudgeted run")


def test_a_step_that_does_not_fit_is_refused_with_the_arithmetic_in_the_message():
    f = estimate_step(
        M,
        rows=1,
        width=34_522,
        vocab_size=REMAP_VOCAB,
        activations=ActivationModel(recompute="none"),
        **BF16,
    )
    with pytest.raises(MemoryRefused) as excinfo:
        refuse_unless_it_fits(f, device_bytes=96 * GB, where="bucket width 34,522")
    message = str(excinfo.value)
    assert "bucket width 34,522" in message
    assert "optimizer state" in message, "the breakdown must be in the refusal, not a total"
    assert "ARITHMETIC, not a measurement" in message, "rule 5 travels with the number"


def test_a_step_that_fits_is_not_refused():
    f = estimate_step(M, rows=1, width=34_522, vocab_size=REMAP_VOCAB, **BF16)
    refuse_unless_it_fits(f, device_bytes=96 * GB, where="bucket width 34,522")


def test_refusing_without_saying_what_is_being_checked_is_itself_refused():
    f = estimate_step(M, rows=1, width=479, vocab_size=REMAP_VOCAB, **BF16)
    with pytest.raises(ValueError, match="where"):
        refuse_unless_it_fits(f, device_bytes=96 * GB, where="   ")


# --- the search --------------------------------------------------------------------------


def test_max_positions_returns_zero_rather_than_none_when_one_row_will_not_fit():
    """Zero is a real answer. `None` invites `or default`, which is the opposite of truth."""
    n = max_positions_that_fit(
        M,
        device_bytes=8 * GB,  # smaller than the static state alone
        width=34_522,
        vocab_size=REMAP_VOCAB,
        **BF16,
    )
    assert n == 0


def test_max_positions_is_the_largest_row_count_that_still_fits():
    device = 96 * GB
    width = 8192
    n = max_positions_that_fit(M, device_bytes=device, width=width, vocab_size=REMAP_VOCAB, **BF16)
    rows = n // width
    assert rows >= 1
    assert estimate_step(
        M, rows=rows, width=width, vocab_size=REMAP_VOCAB, **BF16
    ).fits(device_bytes=device)
    assert not estimate_step(
        M, rows=rows + 1, width=width, vocab_size=REMAP_VOCAB, **BF16
    ).fits(device_bytes=device)


def test_a_bigger_card_never_fits_fewer_rows():
    width = 14_759
    previous = -1
    for gb in (24, 40, 48, 80, 96):
        n = max_positions_that_fit(
            M, device_bytes=gb * GB, width=width, vocab_size=REMAP_VOCAB, **BF16
        )
        assert n >= previous, f"{gb} GB fit fewer positions than a smaller card"
        previous = n


# --- input validation --------------------------------------------------------------------


def test_an_unknown_dtype_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="not a dtype"):
        estimate_step(M, rows=1, width=512, param_dtype="fp8")
    assert "fp8" not in BYTES_PER_ELEMENT


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_nonsense_batch_shape_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        estimate_step(M, rows=bad, width=512)
    with pytest.raises((ValueError, TypeError)):
        estimate_step(M, rows=1, width=bad)


def test_a_safety_allowance_larger_than_the_thing_it_covers_is_refused():
    estimate_step(M, rows=1, width=512, safety_fraction=MAX_SAFETY_FRACTION)
    with pytest.raises(ValueError, match="safety_fraction"):
        estimate_step(M, rows=1, width=512, safety_fraction=MAX_SAFETY_FRACTION + 0.01)
    with pytest.raises(ValueError, match="safety_fraction"):
        estimate_step(M, rows=1, width=512, safety_fraction=-0.1)
    with pytest.raises(TypeError):
        estimate_step(M, rows=1, width=512, safety_fraction="lots")  # type: ignore[arg-type]


def test_a_non_finite_device_budget_is_refused():
    f = estimate_step(M, rows=1, width=512, vocab_size=REMAP_VOCAB, **BF16)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            f.fits(device_bytes=bad)
    with pytest.raises(TypeError):
        f.fits(device_bytes=math.inf)  # type: ignore[arg-type]


def test_an_activation_policy_outside_the_two_it_models_is_refused():
    with pytest.raises(ValueError, match="recompute"):
        ActivationModel(recompute="selective")
    with pytest.raises(ValueError, match="attention"):
        ActivationModel(attention="xformers")


def test_a_model_spec_with_no_layers_is_refused():
    with pytest.raises(ValueError, match="no layers"):
        _with(M, n_full_attention_layers=0, n_linear_attention_layers=0)


def test_an_embedding_larger_than_the_model_is_refused():
    with pytest.raises(ValueError, match="cannot be larger"):
        _with(M, params_embedding=M.params_total + 1)


def test_the_provenance_says_arithmetic_on_every_estimate():
    """Rule 5. A number that travels without this can be read as a measurement."""
    f = estimate_step(M, rows=1, width=512, vocab_size=REMAP_VOCAB, **BF16)
    assert f.provenance.startswith("ARITHMETIC, not a measurement")
    assert "No CUDA device" in f.provenance
