"""Will this step fit on this GPU? Answered before the first step, not by the allocator.

Nothing in this repository could answer that question before this module. ``qd-preflight``
reads every device's ``total_mem_gib`` (``crates/qd-preflight/src/cuda.rs:174``) and spends
it on a human-readable string (``:217``); it compares the device *count* and never the
device *size*. ``remap._MemoryProbe`` samples ``torch.cuda.memory_allocated`` **around a
forward pass that has already happened**. Between them there was no estimate, so a batch
that could not fit was discovered by ``torch.OutOfMemoryError`` some minutes into a run --
which on a rented box is money already spent.

This module is the missing half: the arithmetic, from first principles, torch-free, so it
runs in the gate rather than on the machine being paid for.

## What is measured and what is stated

**Measured.** Parameter counts come from the checkpoint's own tensor shapes --
``tools/verify_checkpoint_inventory.py`` reads the safetensors header without torch, and
[`QWEN3_5_2B_TEXT`] carries what it found. So does the architecture: ``q_proj`` is
``[4096, 2048]`` and not ``[2048, 2048]`` because ``attn_output_gate`` is true and the
projection carries a gate beside the query, which is the kind of thing an assumed
architecture gets wrong by a factor of two.

**Stated.** [`ActivationModel`] *enumerates* the tensors a backward pass must keep, one
line per tensor, so a reader can check the list rather than trust a coefficient. It is a
**lower bound**: a real framework also keeps an RMSNorm's reciprocal, SDPA's per-row
logsumexp, the conv's padded input, and whatever its kernels choose. [`SAFETY_FRACTION`]
is the stated allowance for all of that, and it is reported separately from the enumerated
figure so the two are never confused. **An estimate from this module is arithmetic, not a
measurement**, and [`StepFootprint.provenance`] says so in words that survive being pasted
into a report.

## Why it refuses rather than warns

A preflight that prints a warning and continues has told the operator something they will
read after the allocator has already answered. [`refuse_unless_it_fits`] raises
[`MemoryRefused`] carrying the whole breakdown, because the failure it replaces is loud
already -- it is merely late, and expensive. And an *unknown* device size refuses too: a
budget nobody supplied is not a budget that was checked, and the one thing this module must
never do is let "could not check" read like "checked and fits".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

__all__ = [
    "ADAMW_BF16",
    "ADAMW_FP32",
    "BYTES_PER_ELEMENT",
    "MAX_SAFETY_FRACTION",
    "MAX_SEARCH_ROWS",
    "QWEN3_5_2B_TEXT",
    "SAFETY_FRACTION",
    "ActivationModel",
    "MemoryRefused",
    "ModelSpec",
    "OptimizerSpec",
    "StepFootprint",
    "estimate_step",
    "max_positions_that_fit",
    "refuse_unless_it_fits",
]

#: Bytes per element, by the names this repository uses for dtypes. ``float8`` is absent on
#: purpose: nothing here trains in it, and a name that resolves to a number invites an
#: estimate for a recipe that was never checked.
BYTES_PER_ELEMENT: Final[Mapping[str, int]] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp64": 8,
}

#: The allowance for what [`ActivationModel`] does not enumerate: kernel workspaces, the
#: allocator's fragmentation, an RMSNorm's saved reciprocal, SDPA's logsumexp, and the
#: temporaries a framework allocates between two layers. It is applied to the activation
#: term only -- weights, gradients and optimizer state are exact -- and it is carried in
#: [`StepFootprint`] as its own field so a report can state the enumerated number and the
#: allowance separately instead of quoting one total that hides both.
SAFETY_FRACTION: Final[float] = 0.35

#: An allowance above this is not an allowance, it is a guess wearing one. Refused rather
#: than accepted, for the same reason ``MAX_CAP_S`` refuses a cap above the program's cap.
MAX_SAFETY_FRACTION: Final[float] = 2.0


class MemoryRefused(RuntimeError):
    """This step was refused before it ran, because the arithmetic says it cannot fit."""


# --- the optimizer ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """How many bytes of *persistent* state an optimizer keeps per trainable parameter.

    ``states_per_param`` is a count of tensors the size of the parameter, not of bytes:
    AdamW keeps two (``exp_avg`` and ``exp_avg_sq``), SGD-with-momentum keeps one, plain
    SGD keeps none. Separating the count from ``state_bytes`` is what lets the same spec
    describe an fp32-state and a bf16-state AdamW without a second class.
    """

    name: str
    states_per_param: int
    state_bytes: int
    keeps_fp32_master: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an optimizer spec must name the optimizer it describes")
        if self.states_per_param < 0:
            raise ValueError(
                f"states_per_param must be non-negative, got {self.states_per_param}"
            )
        if self.state_bytes <= 0:
            raise ValueError(f"state_bytes must be positive, got {self.state_bytes}")

    def bytes_per_param(self, *, param_bytes: int, grad_bytes: int) -> int:
        """Weights + gradients + states + any fp32 master copy, per trainable parameter.

        A master recipe costs **8** more bytes per parameter than one without, not 4: the
        fp32 master *and* the fp32 gradient it is stepped with, because an optimizer over
        fp32 masters cannot consume a bf16 gradient and both copies are live when it steps.
        Measured on a GH200 at rows=1 width=2048 over 1,881,825,088 parameters: the stage
        between "forward + backward" and "optimizer step" adds exactly 7.01 GiB = 4.00
        B/param, on top of the 7.01 GiB the masters already cost. bf16 weights and grads
        under this recipe are therefore 2+2+4+8+4 = **20** B/param -- 35.05 GiB, which is
        the enumerated total ``tools/master_overhead.py`` measures to the digit -- and not
        the 16 B/param this returned before that run.
        """
        master = 4 if self.keeps_fp32_master else 0
        grad_cast = 4 if self.keeps_fp32_master else 0
        return (
            param_bytes
            + grad_bytes
            + grad_cast
            + self.states_per_param * self.state_bytes
            + master
        )


#: What ``tools/ft_toy_run.py:271``, ``tools/real_ft_run.py:574``, ``tools/rung0_toy_run.py:393``
#: and ``qd_train/byte_train.py:206`` all construct: ``torch.optim.AdamW``. Read, not assumed
#: -- every optimizer this repository instantiates is this one, and none of them passes
#: ``foreach``, ``fused`` or ``capturable``, so the states are two plain tensors per
#: parameter in the parameter's own dtype. ``state_bytes=4`` describes the fp32 case; pass a
#: different spec for a recipe that keeps them in bf16.
ADAMW_FP32: Final[OptimizerSpec] = OptimizerSpec(
    name="torch.optim.AdamW", states_per_param=2, state_bytes=4
)

#: The same optimizer over **bf16** parameters, which is what torch actually builds when the
#: model is bf16: `exp_avg` and `exp_avg_sq` follow the parameter's dtype, so they are 2 bytes
#: each and there is no fp32 master copy. Measured on a GH200 over 1,881,825,088 parameters --
#: 14.02 GiB of weights+grads+states, against the 28.04 GiB `ADAMW_FP32` predicts for the same
#: model.
#:
#: **This is the cheap option and the numerically poor one.** bf16 has 8 bits of mantissa, and
#: `exp_avg_sq` accumulates over a whole run; keeping it in bf16 is how a long run degrades in
#: a way no single step shows. It is named here so that a recipe which wants it says so, not
#: so that it becomes the default by being what torch does when nobody chooses.
ADAMW_BF16: Final[OptimizerSpec] = OptimizerSpec(
    name="torch.optim.AdamW", states_per_param=2, state_bytes=2
)


# --- the model ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A decoder's shape, in the terms the memory arithmetic needs.

    ``params_total`` and ``params_embedding`` are **measured** -- read off the checkpoint's
    tensor shapes -- rather than recomputed from the other fields, because a derived
    parameter count that disagrees with the file is a silent claim that the file is wrong.
    :meth:`trainable_params` is the one derived count, and only because the remap changes
    the vocabulary after the file was written.
    """

    name: str
    hidden_size: int
    intermediate_size: int
    n_full_attention_layers: int
    n_linear_attention_layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    attn_output_gate: bool
    linear_heads: int
    linear_head_dim: int
    vocab_size: int
    params_total: int
    params_embedding: int
    tied_embedding: bool
    recurrent_state_bytes: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a model spec must name the model it describes")
        for field_name in (
            "hidden_size",
            "intermediate_size",
            "q_heads",
            "kv_heads",
            "head_dim",
            "linear_heads",
            "linear_head_dim",
            "vocab_size",
            "params_total",
            "params_embedding",
            "recurrent_state_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive int, got {value!r}")
        if self.n_full_attention_layers < 0 or self.n_linear_attention_layers < 0:
            raise ValueError("layer counts must be non-negative")
        if self.n_full_attention_layers + self.n_linear_attention_layers == 0:
            raise ValueError("a model with no layers has no step to estimate")
        if self.params_embedding > self.params_total:
            raise ValueError(
                f"the embedding ({self.params_embedding:,}) cannot be larger than the model "
                f"({self.params_total:,})"
            )

    @property
    def n_layers(self) -> int:
        return self.n_full_attention_layers + self.n_linear_attention_layers

    def trainable_params(self, *, vocab_size: int | None = None) -> int:
        """Parameters that carry a gradient, with the embedding resized to ``vocab_size``.

        ``qd_train.remap`` exists to cut the vocabulary to what a corpus actually uses --
        248,320 to 13,787 on the real shard set -- and the embedding is tied, so that cut
        lands once and removes a quarter of the model. Passing the remapped size here is
        what makes the budget describe the run rather than the checkpoint.
        """
        if vocab_size is None:
            return self.params_total
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive int, got {vocab_size!r}")
        return self.params_total - self.params_embedding + vocab_size * self.hidden_size


#: Qwen3.5-2B-Base's **text** tower. Every number measured 2026-09-20 from the local
#: checkpoint at ``models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2f…``: parameter counts by
#: summing tensor shapes out of the safetensors header, architecture from ``config.json``.
#: The vision tower (331,416,576) and the MTP block (60,828,160) are dropped on load and
#: never trained, so neither is here.
#:
#: ``attn_output_gate`` is true and it is not decoration: ``q_proj`` is ``[4096, 2048]``
#: where ``num_attention_heads x head_dim`` is ``8 x 256 = 2048``. The projection emits a
#: gate beside the query, so the query activation is twice what the head count suggests.
QWEN3_5_2B_TEXT: Final[ModelSpec] = ModelSpec(
    name="Qwen/Qwen3.5-2B-Base (text tower)",
    hidden_size=2048,
    intermediate_size=6144,
    n_full_attention_layers=6,
    n_linear_attention_layers=18,
    q_heads=8,
    kv_heads=2,
    head_dim=256,
    attn_output_gate=True,
    linear_heads=16,
    linear_head_dim=128,
    vocab_size=248_320,
    params_total=1_881_825_088,
    params_embedding=508_559_360,
    tied_embedding=True,
    # config.json: "mamba_ssm_dtype": "float32". The recurrent state is not kept in the
    # compute dtype, and assuming it was would understate it by half.
    recurrent_state_bytes=4,
)


# --- activations --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActivationModel:
    """Elements a backward pass must keep, per token, enumerated per layer.

    Each term below is one saved tensor, named. The list is a **lower bound** on what a
    framework keeps; [`SAFETY_FRACTION`] is the allowance for the rest, and it is applied
    and reported separately.

    ``recompute="full"`` is per-layer gradient checkpointing: only the layer boundaries
    survive the forward pass, and one layer's worth is live at a time during backward.
    ``recompute="none"`` keeps every layer's activations at once.

    ``attention="flash"`` never materialises the ``[heads, L, L]`` score matrix.
    ``attention="math"`` does, and at 34,522 positions that single tensor is 19.07 GB per
    full-attention layer in bf16 -- more than the whole card, six times over. That is not a
    tuning knob; it is the difference between a bucket that trains and one that cannot.
    """

    recompute: str = "full"
    attention: str = "flash"

    def __post_init__(self) -> None:
        if self.recompute not in ("none", "full"):
            raise ValueError(f"recompute must be 'none' or 'full', got {self.recompute!r}")
        if self.attention not in ("flash", "math"):
            raise ValueError(f"attention must be 'flash' or 'math', got {self.attention!r}")

    def linear_layer_elements(self, m: ModelSpec) -> int:
        """One ``linear_attention`` layer's saved elements per token."""
        qkv = 3 * m.linear_heads * m.linear_head_dim  # in_proj_qkv -> [6144, 2048]
        return (
            m.hidden_size  # residual entering the layer
            + m.hidden_size  # input_layernorm output
            + qkv  # in_proj_qkv output
            + qkv  # conv1d output
            + m.linear_heads * m.linear_head_dim  # per-head norm output
            + m.hidden_size  # in_proj_z gate output
            + m.hidden_size  # gated attention output, o_proj input
            + m.hidden_size  # post_attention_layernorm output
            + 3 * m.intermediate_size  # gate_proj, up_proj, silu(gate)*up
        )

    def full_layer_elements(self, m: ModelSpec) -> int:
        """One ``full_attention`` layer's saved elements per token, score matrix excluded."""
        q = m.q_heads * m.head_dim * (2 if m.attn_output_gate else 1)
        return (
            m.hidden_size
            + m.hidden_size
            + q  # q_proj output, gate included
            + m.kv_heads * m.head_dim  # k_proj output
            + m.kv_heads * m.head_dim  # v_proj output
            + m.q_heads * m.head_dim  # attention output, before o_proj
            + m.hidden_size  # o_proj output
            + m.hidden_size  # post_attention_layernorm output
            + 3 * m.intermediate_size
        )

    def elements_per_token(self, m: ModelSpec) -> int:
        """Saved elements per token under this recompute policy."""
        if self.recompute == "none":
            return (
                m.n_linear_attention_layers * self.linear_layer_elements(m)
                + m.n_full_attention_layers * self.full_layer_elements(m)
            )
        # Per-layer checkpointing: one boundary tensor per layer plus the model's output,
        # and one layer's activations live at a time while backward recomputes it.
        boundaries = (m.n_layers + 1) * m.hidden_size
        peak = max(self.linear_layer_elements(m), self.full_layer_elements(m))
        return boundaries + peak

    def score_matrix_elements(self, m: ModelSpec, *, rows: int, width: int) -> int:
        """The materialised attention matrix, or zero when it is never formed.

        Under ``recompute="full"`` one layer is live at a time; under ``"none"`` every
        full-attention layer's matrix is retained at once.
        """
        if self.attention == "flash":
            return 0
        live_layers = 1 if self.recompute == "full" else m.n_full_attention_layers
        return live_layers * rows * m.q_heads * width * width


# --- the estimate -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepFootprint:
    """What one training step needs, with every component kept separately.

    Nothing here is a single total with its parts folded in: a number that cannot be
    attributed cannot be argued with, and the first question anyone asks of a budget is
    which term dominates it.
    """

    model: str
    trainable_params: int
    rows: int
    width: int
    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int
    master_bytes: int
    activation_bytes: int
    score_matrix_bytes: int
    recurrent_state_bytes: int
    loss_head_bytes: int
    safety_bytes: int
    recompute: str
    attention: str
    safety_fraction: float

    @property
    def static_bytes(self) -> int:
        """Weights, gradients, optimizer state and any master copy: independent of shape."""
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes + self.master_bytes

    @property
    def dynamic_bytes(self) -> int:
        """Everything that scales with ``rows x width``."""
        return (
            self.activation_bytes
            + self.score_matrix_bytes
            + self.recurrent_state_bytes
            + self.safety_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.static_bytes + self.dynamic_bytes + self.loss_head_bytes

    @property
    def positions(self) -> int:
        return self.rows * self.width

    @property
    def provenance(self) -> str:
        """One sentence a report can paste. Rule 5: this is arithmetic, not a measurement."""
        return (
            f"ARITHMETIC, not a measurement: parameter counts are measured from the "
            f"checkpoint's tensor shapes; the activation term is an enumerated lower bound "
            f"under recompute={self.recompute!r}, attention={self.attention!r}, plus a "
            f"{self.safety_fraction:.0%} stated allowance for what the enumeration omits. "
            f"No CUDA device was available to check it against."
        )

    def breakdown(self) -> str:
        """Every component, in bytes and GB, both numbers on every line."""
        gb = 1000**3
        lines = [
            f"{self.model}: {self.rows} row(s) x {self.width} position(s) "
            f"= {self.positions:,} positions",
            f"  trainable parameters      {self.trainable_params:>15,}",
            f"  weights                   {self.param_bytes:>15,} B  {self.param_bytes/gb:8.2f} GB",
            f"  gradients                 {self.grad_bytes:>15,} B  {self.grad_bytes/gb:8.2f} GB",
            f"  optimizer state           {self.optimizer_bytes:>15,} B  "
            f"{self.optimizer_bytes/gb:8.2f} GB",
            f"  fp32 master copy          {self.master_bytes:>15,} B  "
            f"{self.master_bytes/gb:8.2f} GB",
            f"  activations (enumerated)  {self.activation_bytes:>15,} B  "
            f"{self.activation_bytes/gb:8.2f} GB",
            f"  attention score matrix    {self.score_matrix_bytes:>15,} B  "
            f"{self.score_matrix_bytes/gb:8.2f} GB",
            f"  recurrent state           {self.recurrent_state_bytes:>15,} B  "
            f"{self.recurrent_state_bytes/gb:8.2f} GB",
            f"  loss head (fused CE)      {self.loss_head_bytes:>15,} B  "
            f"{self.loss_head_bytes/gb:8.2f} GB",
            f"  stated allowance ({self.safety_fraction:.0%})     {self.safety_bytes:>15,} B  "
            f"{self.safety_bytes/gb:8.2f} GB",
            f"  TOTAL                     {self.total_bytes:>15,} B  "
            f"{self.total_bytes/gb:8.2f} GB",
        ]
        return "\n".join(lines)

    def fits(self, *, device_bytes: int) -> bool:
        if not isinstance(device_bytes, int) or isinstance(device_bytes, bool):
            raise TypeError(f"device_bytes must be int, got {type(device_bytes).__name__}")
        if device_bytes <= 0:
            raise ValueError(f"device_bytes must be positive, got {device_bytes}")
        return self.total_bytes <= device_bytes


def _dtype_bytes(name: str, *, what: str) -> int:
    try:
        return BYTES_PER_ELEMENT[name]
    except KeyError:
        known = ", ".join(sorted(BYTES_PER_ELEMENT))
        raise ValueError(
            f"{what}={name!r} is not a dtype this module has an element size for; known: "
            f"{known}. Guessing a size would put a number in a budget that nobody checked."
        ) from None


def estimate_step(
    model: ModelSpec,
    *,
    rows: int,
    width: int,
    optimizer: OptimizerSpec = ADAMW_FP32,
    param_dtype: str = "bf16",
    grad_dtype: str = "bf16",
    activation_dtype: str = "bf16",
    activations: ActivationModel | None = None,
    vocab_size: int | None = None,
    ce_chunk_bytes: int = 128 * 1024 * 1024,
    safety_fraction: float = SAFETY_FRACTION,
) -> StepFootprint:
    """One training step's memory, component by component.

    ``vocab_size`` resizes the tied embedding -- pass the remapped size to budget the run
    this repository actually intends, or leave it ``None`` to budget the checkpoint as
    shipped. ``ce_chunk_bytes`` is ``fused_ce.TARGET_LOGIT_BYTES``: the logit slab is
    already bounded there, so it enters as a constant rather than as a term in ``L x V``.
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError(f"rows must be a positive int, got {rows!r}")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError(f"width must be a positive int, got {width!r}")
    if not isinstance(ce_chunk_bytes, int) or isinstance(ce_chunk_bytes, bool):
        raise TypeError(f"ce_chunk_bytes must be int, got {type(ce_chunk_bytes).__name__}")
    if ce_chunk_bytes <= 0:
        raise ValueError(f"ce_chunk_bytes must be positive, got {ce_chunk_bytes}")
    if not isinstance(safety_fraction, (int, float)) or isinstance(safety_fraction, bool):
        raise TypeError(
            f"safety_fraction must be a real number, got {type(safety_fraction).__name__}"
        )
    if not 0.0 <= safety_fraction <= MAX_SAFETY_FRACTION:
        raise ValueError(
            f"safety_fraction must be in [0, {MAX_SAFETY_FRACTION}], got {safety_fraction!r}. "
            "An allowance larger than the thing it covers is a guess, not an allowance."
        )

    acts = activations if activations is not None else ActivationModel()
    p_bytes = _dtype_bytes(param_dtype, what="param_dtype")
    g_bytes = _dtype_bytes(grad_dtype, what="grad_dtype")
    a_bytes = _dtype_bytes(activation_dtype, what="activation_dtype")

    trainable = model.trainable_params(vocab_size=vocab_size)
    positions = rows * width

    activation_bytes = positions * acts.elements_per_token(model) * a_bytes
    score_bytes = acts.score_matrix_elements(model, rows=rows, width=width) * a_bytes
    # The recurrent state is [rows, heads, key_dim, value_dim] per linear-attention layer
    # and does not scale with sequence length. Under checkpointing one layer is live.
    live_linear = (
        1 if acts.recompute == "full" else model.n_linear_attention_layers
    ) if model.n_linear_attention_layers else 0
    recurrent_bytes = (
        live_linear
        * rows
        * model.linear_heads
        * model.linear_head_dim
        * model.linear_head_dim
        * model.recurrent_state_bytes
    )
    # fused_linear_cross_entropy keeps one fp32 accumulator the size of the projection
    # weight, plus one bounded logit slab. Both are fixed costs, not per-position ones.
    effective_vocab = vocab_size if vocab_size is not None else model.vocab_size
    loss_head_bytes = effective_vocab * model.hidden_size * 4 + ce_chunk_bytes

    enumerated_dynamic = activation_bytes + score_bytes + recurrent_bytes
    safety_bytes = int(enumerated_dynamic * safety_fraction)

    return StepFootprint(
        model=model.name,
        trainable_params=trainable,
        rows=rows,
        width=width,
        param_bytes=trainable * p_bytes,
        # An fp32-master recipe cannot step on fp32 masters with low-precision gradients, so
        # every gradient is cast up and BOTH copies are live when the optimizer steps. That
        # second copy is 4 B/param and this model did not carry it: measured on a GH200 at
        # rows=1 width=2048, the stage between "forward + backward" and "optimizer step"
        # adds exactly 7.01 GiB over 1,881,825,088 parameters, which is 4.00 B/param.
        # Leaving it out made the budget under-state the master recipe by 49% -- past the
        # 35% safety allowance, in the direction that says a run fits when it does not.
        grad_bytes=trainable * (g_bytes + 4 if optimizer.keeps_fp32_master else g_bytes),
        optimizer_bytes=trainable * optimizer.states_per_param * optimizer.state_bytes,
        master_bytes=trainable * 4 if optimizer.keeps_fp32_master else 0,
        activation_bytes=activation_bytes,
        score_matrix_bytes=score_bytes,
        recurrent_state_bytes=recurrent_bytes,
        loss_head_bytes=loss_head_bytes,
        safety_bytes=safety_bytes,
        recompute=acts.recompute,
        attention=acts.attention,
        safety_fraction=float(safety_fraction),
    )


#: The row count at which the doubling search gives up. 16.7M rows of anything exceeds
#: every device that exists, so reaching this means the caller's arguments are wrong, not
#: that the device is enormous. Bounded because an unbounded search is the failure mode
#: this module was written to remove, and it would be embarrassing to add one.
MAX_SEARCH_ROWS: Final[int] = 1 << 24


def max_positions_that_fit(
    model: ModelSpec,
    *,
    device_bytes: int,
    width: int,
    optimizer: OptimizerSpec = ADAMW_FP32,
    param_dtype: str = "bf16",
    grad_dtype: str = "bf16",
    activation_dtype: str = "bf16",
    activations: ActivationModel | None = None,
    vocab_size: int | None = None,
    ce_chunk_bytes: int = 128 * 1024 * 1024,
    safety_fraction: float = SAFETY_FRACTION,
) -> int:
    """The largest ``rows x width`` this device admits, or 0 when even one row does not.

    Searched rather than solved: the score-matrix term is quadratic in ``width`` and the
    recurrent term is a step function of the recompute policy, so a closed form would have
    to be re-derived every time one of them changes. Zero is a real answer and is returned
    as one -- a caller that reads it as "unbounded" has the opposite of the truth, which is
    why this never returns ``None``.

    The parameters are spelled out rather than forwarded as ``**kwargs`` so that a typo in
    a caller's keyword is a ``TypeError`` here instead of a silently different budget.

    **``device_bytes`` is what the allocator can USE, not what the device reports.** Passing
    ``torch.cuda.get_device_properties(0).total_memory`` makes this optimistic, and
    optimistic here means a job is launched at a row count that OOMs. Measured on a GH200
    reporting 94.50 GiB (``tools/gh200_rows.py``), walking rows upward until the device
    refused:

    ======  =========  ========  =====================================
    width   predicted  measured  last peak / next allocation that OOMed
    ======  =========  ========  =====================================
    34,522  9          **8**     83.05 GiB / 3.56 GiB
    8,192   40         **36**    87.95 GiB / 3.47 GiB
    ======  =========  ========  =====================================

    The arithmetic itself is close: ``StepFootprint.total_bytes`` at the largest row count
    that fit was 81.47 GiB against a measured 83.05, and 85.95 against 87.95 -- within 2.3%
    both times, and *under* in both, which is the unsafe direction but a small one. What is
    not close is the device budget. ``mem_get_info()[0]`` reported 91.42 GiB free, and the
    real ceiling for this workload sits between 87.95 GiB (the largest peak that completed)
    and 89.96 GiB (where the next row would have landed). Free bytes are therefore **also**
    optimistic: the allocator cannot hand out every free byte to a fragmented workload.

    So there is no value of ``device_bytes`` that makes this exact, and this function does
    not pretend otherwise. **Its answer is an upper bound, to be confirmed by a real step.**
    ``tools/gh200_rows.py`` is that confirmation: it walks rows upward until the device
    refuses, and an OOM there is the measurement, not a failure. Nothing in this repository
    should launch a long run at this function's answer without having run that sweep for the
    same shape.
    """
    if not isinstance(device_bytes, int) or isinstance(device_bytes, bool):
        raise TypeError(f"device_bytes must be int, got {type(device_bytes).__name__}")
    if device_bytes <= 0:
        raise ValueError(f"device_bytes must be positive, got {device_bytes}")

    def at(rows: int) -> StepFootprint:
        return estimate_step(
            model,
            rows=rows,
            width=width,
            optimizer=optimizer,
            param_dtype=param_dtype,
            grad_dtype=grad_dtype,
            activation_dtype=activation_dtype,
            activations=activations,
            vocab_size=vocab_size,
            ce_chunk_bytes=ce_chunk_bytes,
            safety_fraction=safety_fraction,
        )

    if not at(1).fits(device_bytes=device_bytes):
        return 0

    # Double until it stops fitting, then bisect. Both loops are bounded.
    low, high = 1, 2
    while high <= MAX_SEARCH_ROWS:
        if not at(high).fits(device_bytes=device_bytes):
            break
        low, high = high, high * 2
    else:  # pragma: no cover - no real device admits MAX_SEARCH_ROWS rows of anything
        return low * width

    while low + 1 < high:
        mid = (low + high) // 2
        if at(mid).fits(device_bytes=device_bytes):
            low = mid
        else:
            high = mid
    return low * width


def refuse_unless_it_fits(
    footprint: StepFootprint, *, device_bytes: int | None, where: str
) -> None:
    """Raise [`MemoryRefused`] unless this step fits, and raise it when nobody said.

    ``device_bytes=None`` refuses. A budget that was never supplied has not been checked,
    and letting it pass is how "approved" comes to mean "unexamined" -- the one failure
    mode this whole module exists to remove.
    """
    if not where.strip():
        raise ValueError("refuse_unless_it_fits needs a `where` naming what is being checked")
    if device_bytes is None:
        raise MemoryRefused(
            f"{where}: no device memory budget was supplied, so whether this step fits was "
            f"never checked. Refusing rather than starting: an unchecked budget must not "
            f"read like a checked one.\n{footprint.breakdown()}\n{footprint.provenance}"
        )
    if footprint.fits(device_bytes=device_bytes):
        return
    gb = 1000**3
    over = footprint.total_bytes - device_bytes
    raise MemoryRefused(
        f"{where}: this step needs {footprint.total_bytes/gb:.2f} GB and the device has "
        f"{device_bytes/gb:.2f} GB -- {over/gb:.2f} GB over. Refused before the first step "
        f"rather than by the allocator partway into a paid run.\n"
        f"{footprint.breakdown()}\n{footprint.provenance}"
    )
