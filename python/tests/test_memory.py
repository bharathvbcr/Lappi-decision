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
    ADAMW_BF16,
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


def test_an_fp32_master_recipe_costs_eight_more_bytes_per_parameter():
    """This test previously asserted **four**, and the number was wrong.

    It encoded the belief that an fp32 master costs only the master copy. It does not: an
    optimizer stepping on fp32 masters cannot consume a bf16 gradient, so every gradient is
    cast up and both copies are live at the step. Measured on a GH200 over 1,881,825,088
    parameters, the stage between "forward + backward" and "optimizer step" adds exactly
    7.01 GiB -- 4.00 B/param -- on top of the 7.01 GiB of masters
    (``tools/master_overhead.py``). The old expectation under-stated the recipe by 49%
    against a 35% safety allowance, in the direction that says a run fits when it does not.
    """
    plain = ADAMW_FP32
    master = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    assert (
        master.bytes_per_param(param_bytes=2, grad_bytes=2)
        - plain.bytes_per_param(param_bytes=2, grad_bytes=2)
        == 8
    ), "the master copy AND the fp32 gradient it is stepped with"
    assert master.bytes_per_param(param_bytes=2, grad_bytes=2) == 20

    a = estimate_step(M, rows=1, width=1024, optimizer=plain, vocab_size=REMAP_VOCAB, **BF16)
    b = estimate_step(M, rows=1, width=1024, optimizer=master, vocab_size=REMAP_VOCAB, **BF16)
    n = a.trainable_params
    assert b.master_bytes == 4 * n
    assert a.master_bytes == 0
    assert b.grad_bytes - a.grad_bytes == 4 * n, "the fp32 cast, counted with the gradients"
    assert b.static_bytes - a.static_bytes == b.master_bytes + 4 * n


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


def test_an_fp32_master_recipe_budgets_the_gradient_cast_as_well_as_the_master():
    """Measured on a GH200 and missing from the model until then.

    A master recipe steps on fp32 masters, so every bf16 gradient is cast up and both
    copies are live when the optimizer steps. The stage between "forward + backward" and
    "optimizer step" adds exactly 7.01 GiB over 1,881,825,088 parameters -- 4.00 B/param --
    and `estimate_step` counted only the master. That under-stated the recipe by 49%
    against a 35% safety allowance, which is the direction that says a run fits when it
    does not.
    """
    master = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    plain = OptimizerSpec("AdamW", 2, 4)
    kw = {"rows": 1, "width": 2048, "param_dtype": "bf16", "grad_dtype": "bf16"}

    with_master = estimate_step(QWEN3_5_2B_TEXT, optimizer=master, **kw)
    without = estimate_step(QWEN3_5_2B_TEXT, optimizer=plain, **kw)
    n = with_master.trainable_params

    assert without.grad_bytes == n * 2, "a bf16 gradient is 2 B/param"
    assert with_master.grad_bytes == n * 6, (
        "an fp32-master recipe holds the bf16 gradient AND its fp32 cast: 2 + 4 B/param"
    )
    assert with_master.master_bytes == n * 4
    assert with_master.grad_bytes - without.grad_bytes == n * 4


def test_the_master_recipes_enumerated_total_matches_what_the_gh200_measured():
    """Against the stage-by-stage attribution in ``tools/master_overhead.py``.

    Enumerated there: 3.51 + 3.51 + 7.01 + 7.01 + 14.02 = 35.05 GiB, against a measured
    steady-state peak of 38.91 GiB at ``foreach=False`` and 42.13 GiB at ``foreach=True``.
    The model is a lower bound by construction, so it must not EXCEED the measurement, and
    the shortfall must sit inside the safety allowance.
    """
    master = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    fp = estimate_step(
        QWEN3_5_2B_TEXT, rows=1, width=2048, optimizer=master,
        param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16",
    )
    static = fp.param_bytes + fp.grad_bytes + fp.optimizer_bytes + fp.master_bytes
    gib = 1024 ** 3
    assert static / gib == pytest.approx(35.05, abs=0.05), (
        f"the enumerated static total is {static / gib:.2f} GiB; the GH200 measured the "
        "same five components at 35.05 GiB"
    )

    measured_foreach_off = 38.91 * gib
    backbone = static + fp.activation_bytes + fp.recurrent_state_bytes
    assert backbone < measured_foreach_off, (
        f"the enumerated backbone total {backbone / gib:.2f} GiB exceeds the measured "
        f"{measured_foreach_off / gib:.2f} GiB; a lower bound that is above the measurement "
        "is not a lower bound"
    )
    shortfall = (measured_foreach_off - backbone) / backbone
    assert shortfall < fp.safety_fraction, (
        f"the measurement is {shortfall:.1%} above the enumerated total, past the "
        f"{fp.safety_fraction:.0%} safety allowance"
    )


def test_the_footprint_tracks_the_gh200_peak_it_was_measured_against():
    """The arithmetic, against real ``torch.cuda.max_memory_allocated`` readings.

    ``tools/gh200_rows.py`` walked rows upward on a GH200 until the device refused, with
    ``ADAMW_BF16``, bf16, gradient checkpointing on, over the full 248,320-row vocabulary::

        width 34,522: 8 rows fit, peak 83.05 GiB; 9 OOMed asking for 3.56 GiB
        width  8,192: 36 rows fit, peak 87.95 GiB; 37 OOMed asking for 3.47 GiB

    ``total_bytes`` predicts 81.47 and 85.95 for those two shapes -- within 2.3%, and UNDER
    in both cases. Under is the unsafe direction, so the tolerance is asserted in both
    directions: a change that makes the model wildly conservative is also a regression,
    because an over-stated budget refuses runs that would have fitted.
    """
    gib = 1024 ** 3
    cases = ((34_522, 8, 83.05), (8_192, 36, 87.95))
    for width, rows, measured_gib in cases:
        fp = estimate_step(
            M, rows=rows, width=width, optimizer=ADAMW_BF16,
            param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16",
        )
        predicted_gib = fp.total_bytes / gib
        error = (predicted_gib - measured_gib) / measured_gib
        assert abs(error) < 0.05, (
            f"width {width} x {rows} rows: predicted {predicted_gib:.2f} GiB against a "
            f"measured {measured_gib:.2f} GiB ({error:+.1%}); the model has drifted from "
            "the only measurement there is"
        )


def test_the_capacity_answer_is_an_upper_bound_not_a_launch_target():
    """Characterisation, and a deliberate one.

    Fed the 91.42 GiB that ``mem_get_info`` reported free, the predictor admits 9 rows at
    width 34,522 where 8 fit, and 38 at width 8,192 where 36 fit. One row over at the wide
    bucket and **two** at the narrow one -- the overshoot is not a constant, which is the
    substance of the finding: it grows with the number of allocations, exactly as
    fragmentation does.

    The cause is not the arithmetic -- the test above holds that to 2.3% -- it is that no
    ``device_bytes`` is knowable in advance: the allocator could not hand out its own
    reported free memory to a fragmented workload. The pairs are written out rather than
    derived from a formula, because a formula here would be a guess dressed as a rule.
    """
    gib = 1024 ** 3
    usable_bytes = int(91.42 * gib)  # what mem_get_info reported free on the measured box
    # (width, rows the device fitted, rows this function admits at that free figure)
    measured = ((34_522, 8, 9), (8_192, 36, 38))
    for width, fitted, admitted in measured:
        positions = max_positions_that_fit(
            M, device_bytes=usable_bytes, width=width, optimizer=ADAMW_BF16,
            param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16",
        )
        predicted_rows = positions // width
        assert predicted_rows == admitted, (
            f"width {width}: the predictor admits {predicted_rows} row(s), and the "
            f"measurement recorded {admitted} against a device that fitted {fitted}. "
            "If this changed, re-run tools/gh200_rows.py and update the measurement rather "
            "than this number"
        )
        assert predicted_rows > fitted, (
            f"width {width}: the predictor no longer overshoots. That would be good news, "
            "but it has to be measured on a device rather than asserted here"
        )


def test_recompute_none_at_the_widest_bucket_does_not_fit_and_the_budget_says_so():
    """The one verdict that sat inside the safety band, now measured.

    The budget said 34,522 at ``recompute='none'`` would not fit a 96 GB device, and the
    device agreed: ``tools/gh200_rows.py`` OOMed on 1 row. The budget and the measurement
    are on the same side, so this pins the budget's side of it -- if a change ever made
    this configuration look admissible, the device has already said otherwise.
    """
    fp = estimate_step(
        M, rows=1, width=34_522, optimizer=ADAMW_BF16,
        param_dtype="bf16", grad_dtype="bf16", activation_dtype="bf16",
        activations=ActivationModel(recompute="none"),
    )
    gh200_bytes = int(94.50 * 1024 ** 3)
    assert not fp.fits(device_bytes=gh200_bytes), (
        f"the budget admits {fp.total_bytes / 1024 ** 3:.2f} GiB on a 94.50 GiB device at "
        "recompute='none'; the GH200 OOMed on one row of this shape"
    )


# --- selective checkpointing ---------------------------------------------------------------


def test_each_retained_layer_adds_exactly_its_own_saved_set():
    m = QWEN3_5_2B_TEXT
    full = ActivationModel(recompute="full")
    part = ActivationModel(recompute="full", retained_linear_layers=2, retained_full_layers=1)
    assert part.elements_per_token(m) - full.elements_per_token(m) == (
        2 * full.linear_layer_elements(m) + full.full_layer_elements(m)
    )
    assert part.describe() == "full-except-2-linear-1-full"
    assert full.describe() == "full" and ActivationModel(recompute="none").describe() == "none"
    fp = estimate_step(m, rows=1, width=8192, activations=part)
    assert fp.recompute == "full-except-2-linear-1-full"
    assert fp.activation_bytes > estimate_step(m, rows=1, width=8192).activation_bytes


def test_retaining_every_layer_prices_at_least_what_no_checkpointing_does():
    """Selective with every layer retained is 'none' plus the boundaries -- never less, so
    a policy that keeps everything is never admitted where 'none' would be refused."""
    m = QWEN3_5_2B_TEXT
    every = ActivationModel(
        recompute="full",
        retained_linear_layers=m.n_linear_attention_layers,
        retained_full_layers=m.n_full_attention_layers,
    )
    assert every.elements_per_token(m) >= ActivationModel(recompute="none").elements_per_token(m)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recompute": "none", "retained_linear_layers": 1},
        {"recompute": "full", "retained_full_layers": -1},
        {"recompute": "full", "retained_linear_layers": True},
    ],
)
def test_a_retention_that_describes_no_real_policy_is_refused(kwargs):
    with pytest.raises(ValueError):
        ActivationModel(**kwargs)


def test_more_retained_layers_than_the_model_has_is_refused():
    m = QWEN3_5_2B_TEXT
    acts = ActivationModel(recompute="full", retained_full_layers=m.n_full_attention_layers + 1)
    with pytest.raises(ValueError, match="exceed"):
        acts.elements_per_token(m)


# --- a spec measured from a checkpoint, and the 4B/9B shapes ---------------------------------
#
# The defects pinned here (DevMap audit, 2026-10-06): the spec had one linear-attention head
# count, so a 4B's 32 value heads were budgeted as 16; the only builder of a spec from a
# checkpoint lived in a CLI tool, read headers unbounded, and defaulted the tie and the gate
# when config.json did not state them; and a spec could say "untied" while the remapped
# budget cut one [V, H] matrix where an untied head has two.

#: Qwen3.5-4B-Base's text_config, the fields spec_from_checkpoint reads, as fetched from
#: huggingface.co on 2026-10-06 (AUDIT/next-train-2026-10-06/sizing.py). 32 layers, every
#: fourth full attention.
FOUR_B_TEXT = {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "vocab_size": 248_320,
    "mamba_ssm_dtype": "float32",
    "layer_types": ["linear_attention"] * 3 * 8 + ["full_attention"] * 8,
}


def _write_header_only(path: Path, tensors: dict[str, list[int]]) -> None:
    """A safetensors file whose header names ``tensors`` and carries no data: the reader
    under test never reads past the header."""
    header = {n: {"dtype": "BF16", "shape": s, "data_offsets": [0, 0]} for n, s in tensors.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _snapshot(
    tmp_path: Path,
    *,
    text: dict[str, object] | None = None,
    top_tie: object = True,
    extra: dict[str, list[int]] | None = None,
) -> Path:
    """A 4B-shaped snapshot in the HF cache layout: config.json plus one header-only shard."""
    text = dict(FOUR_B_TEXT if text is None else text)
    snap = tmp_path / "models--Qwen--Qwen3.5-4B-Base" / "snapshots" / "f00d"
    snap.mkdir(parents=True)
    config: dict[str, object] = {"text_config": text}
    if top_tie is not None:
        config["tie_word_embeddings"] = top_tie
    (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.language_model.embed_tokens.weight": [248_320, 2560],
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": [8192, 2560],
        "model.visual.blocks.0.attn.qkv.weight": [3456, 1152],
        "mtp.fc.weight": [2560, 5120],
        **(extra or {}),
    }
    _write_header_only(snap / "model.safetensors", tensors)
    return snap


def test_the_2b_snapshot_reproduces_the_2b_constant_field_for_field():
    from qd_train.memory import spec_from_checkpoint

    snap = _local_snapshot()
    if snap is None:  # pragma: no cover - depends on the host's cache
        pytest.skip("no local Qwen3.5-2B-Base snapshot to check the builder against")
    assert spec_from_checkpoint(snap) == QWEN3_5_2B_TEXT


def test_a_4b_checkpoint_is_budgeted_with_its_value_heads(tmp_path):
    """The 4B's GDN layer has 16 key heads and 32 value heads. q and k are key-heads wide;
    v, z, the norm, out_proj's input and the recurrent state are value-heads wide."""
    from qd_train.memory import spec_from_checkpoint

    spec = spec_from_checkpoint(_snapshot(tmp_path))
    assert spec.name == "Qwen/Qwen3.5-4B-Base (text tower)"
    assert (spec.linear_heads, spec.linear_value_heads) == (16, 32)
    assert spec.tied_embedding is True
    # Only the text tower counts; the vision and MTP tensors are dropped on load.
    assert spec.params_total == 248_320 * 2560 + 8192 * 2560
    # 2560*2 (residual, norm) + 8192*2 (qkv, conv) + 4096*3 (norm, z, out_proj input)
    # + 2560 (post norm) + 3*9216 (MLP). The one-head-count spec said 54,784: 14% short.
    assert ActivationModel(recompute="none").linear_layer_elements(spec) == 64_000
    f = estimate_step(
        spec, rows=1, width=1, activations=ActivationModel(recompute="full"),
        optimizer=ADAMW_BF16, **BF16,
    )
    # One live layer: [rows, value_heads, key_dim, value_dim] in fp32. It was 16x128x128.
    assert f.recurrent_state_bytes == 32 * 128 * 128 * 4


def test_the_2b_figures_are_unchanged_by_the_value_head_fields():
    """On the 2B the value heads equal the key heads and value_dim equals hidden_size, so
    the split changes no 2B number: v5's device_budget reproduction still holds."""
    acts = ActivationModel(recompute="none")
    assert acts.linear_layer_elements(M) == (
        2 * M.hidden_size + 2 * 3 * 16 * 128 + 16 * 128 + 2 * M.hidden_size + M.hidden_size
        + 3 * M.intermediate_size
    )


def test_value_heads_that_are_not_a_whole_multiple_of_key_heads_are_refused():
    with pytest.raises(ValueError, match="multiple"):
        _with(M, linear_value_heads=24)


def test_an_untied_head_is_counted_and_its_remap_is_refused(tmp_path):
    from qd_train.memory import spec_from_checkpoint

    snap = _snapshot(tmp_path, top_tie=False, extra={"lm_head.weight": [248_320, 2560]})
    spec = spec_from_checkpoint(snap)
    assert spec.tied_embedding is False
    assert spec.params_total == 2 * 248_320 * 2560 + 8192 * 2560
    assert spec.trainable_params() == spec.params_total
    with pytest.raises(ValueError, match="untied"):
        spec.trainable_params(vocab_size=REMAP_VOCAB)


@pytest.mark.parametrize(
    ("top_tie", "text_tie", "head_stored", "why"),
    [
        (True, None, True, "contradictory"),  # says tied, stores a head
        (False, None, False, "contradictory"),  # says untied, stores none
        (True, False, False, "contradictory"),  # the two flags disagree
        (None, True, False, "no top-level"),  # transformers would default it to false
        ("true", None, False, "not a bool"),
    ],
)
def test_a_head_whose_tie_would_be_a_guess_is_refused(
    tmp_path, top_tie, text_tie, head_stored, why
):
    from qd_train.memory import CheckpointRefused, spec_from_checkpoint

    text = dict(FOUR_B_TEXT)
    if text_tie is not None:
        text["tie_word_embeddings"] = text_tie
    extra = {"lm_head.weight": [248_320, 2560]} if head_stored else None
    snap = _snapshot(tmp_path, text=text, top_tie=top_tie, extra=extra)
    with pytest.raises(CheckpointRefused, match=why):
        spec_from_checkpoint(snap)


def test_head_is_tied_over_every_combination_of_flags_and_storage():
    """The whole truth table, not samples: a top-level flag, a text_config flag (each true,
    false, absent or not a bool) and the storage (a head tensor or none) -- 32 cases. Tied
    is answered only when every stated flag is true and nothing is stored; untied only when
    every stated flag is false and a head is stored; everything else is refused."""
    from qd_train.memory import CheckpointRefused, head_is_tied

    values = (True, False, None, "true")
    seen = {"tied": 0, "untied": 0, "refused": 0}
    for top in values:
        for text in values:
            for stored in (False, True):
                config: dict[str, object] = {"text_config": {}}
                if top is not None:
                    config["tie_word_embeddings"] = top
                if text is not None:
                    config["text_config"] = {"tie_word_embeddings": text}
                names = ["model.language_model.embed_tokens.weight"]
                if stored:
                    names.append("lm_head.weight")
                stated = [v for v in (top, text) if v is not None]
                if top is True and text in (True, None) and not stored:
                    want = True
                elif top is False and text in (False, None) and stored:
                    want = False
                else:
                    want = None
                case = (top, text, stored)
                if want is None:
                    with pytest.raises(CheckpointRefused):
                        head_is_tied(config, names, where="case")
                    seen["refused"] += 1
                else:
                    assert head_is_tied(config, names, where="case") is want, case
                    assert all(isinstance(v, bool) for v in stated), case
                    seen["tied" if want else "untied"] += 1
    assert seen == {"tied": 2, "untied": 2, "refused": 28}


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"linear_num_value_heads": None}, "linear_num_value_heads"),
        ({"attn_output_gate": None}, "attn_output_gate"),
        ({"mamba_ssm_dtype": None}, "mamba_ssm_dtype"),
        ({"mamba_ssm_dtype": "float8"}, "float8"),
        ({"vocab_size": True}, "vocab_size"),
        ({"hidden_size": 2048}, "embed_tokens"),
        ({"layer_types": ["linear_attention", "sliding_attention"]}, "layer kind"),
    ],
)
def test_spec_from_checkpoint_refuses_what_it_would_have_to_guess(tmp_path, change, why):
    from qd_train.memory import CheckpointRefused, spec_from_checkpoint

    text = {k: v for k, v in {**FOUR_B_TEXT, **change}.items() if v is not None}
    with pytest.raises(CheckpointRefused, match=why):
        spec_from_checkpoint(_snapshot(tmp_path, text=text))


def test_a_tensor_under_an_unknown_prefix_is_refused_not_counted(tmp_path):
    from qd_train.memory import CheckpointRefused, spec_from_checkpoint

    snap = _snapshot(tmp_path, extra={"model.audio.proj.weight": [8, 8]})
    with pytest.raises(CheckpointRefused, match="prefix"):
        spec_from_checkpoint(snap)


def test_a_tensor_in_two_shards_and_a_snapshot_with_no_shards_are_refused(tmp_path):
    from qd_train.memory import CheckpointRefused, checkpoint_tensor_index

    snap = _snapshot(tmp_path)
    _write_header_only(snap / "model-2.safetensors", {"mtp.fc.weight": [2560, 5120]})
    with pytest.raises(CheckpointRefused, match="two shards"):
        checkpoint_tensor_index(snap)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CheckpointRefused, match=r"no \.safetensors"):
        checkpoint_tensor_index(empty)


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        (b"\x01\x02", "truncated"),
        (struct.pack("<Q", 0), "outside"),
        (struct.pack("<Q", 64 * 1024 * 1024 + 1), "outside"),
        (struct.pack("<Q", 100) + b"{}", "holds 2"),
        (struct.pack("<Q", 2) + b"[]", "not an object"),
        (struct.pack("<Q", 3) + b"{x}", "not JSON"),
    ],
)
def test_a_safetensors_header_is_read_within_its_bounds_or_refused(tmp_path, raw, why):
    from qd_train.memory import CheckpointRefused, safetensors_header

    path = tmp_path / "bad.safetensors"
    path.write_bytes(raw)
    with pytest.raises(CheckpointRefused, match=why):
        safetensors_header(path)


def test_a_shape_that_is_not_a_list_of_dimensions_is_refused(tmp_path):
    from qd_train.memory import CheckpointRefused, checkpoint_tensor_index

    snap = tmp_path / "s"
    snap.mkdir()
    raw = json.dumps({"x": {"dtype": "BF16", "shape": [2, -1], "data_offsets": [0, 0]}}).encode()
    (snap / "m.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(CheckpointRefused, match="valid shape"):
        checkpoint_tensor_index(snap)
