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

import json
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "ADAMW_BF16",
    "ADAMW_FP32",
    "ADAMW_KAHAN",
    "ADAMW_MASTER",
    "BYTES_PER_ELEMENT",
    "MAX_CHECKPOINT_TENSORS",
    "MAX_SAFETENSORS_HEADER_BYTES",
    "MAX_SAFETY_FRACTION",
    "MAX_SEARCH_ROWS",
    "QWEN3_5_2B_TEXT",
    "SAFETY_FRACTION",
    "TEXT_PREFIX",
    "ActivationModel",
    "CheckpointRefused",
    "MemoryRefused",
    "ModelSpec",
    "OptimizerSpec",
    "StepFootprint",
    "checkpoint_tensor_index",
    "estimate_step",
    "head_is_tied",
    "max_positions_that_fit",
    "refuse_unless_it_fits",
    "safetensors_header",
    "spec_from_checkpoint",
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
    #: Bytes per parameter of a Kahan compensation buffer: the rounding residual of each
    #: low-precision weight, kept beside it so updates smaller than the weight's spacing
    #: accumulate instead of rounding away. ``qd_train.optim.KahanBf16AdamW`` keeps one bf16
    #: buffer, so 2. Zero for every recipe without one.
    compensation_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an optimizer spec must name the optimizer it describes")
        if self.states_per_param < 0:
            raise ValueError(
                f"states_per_param must be non-negative, got {self.states_per_param}"
            )
        if self.state_bytes <= 0:
            raise ValueError(f"state_bytes must be positive, got {self.state_bytes}")
        if self.compensation_bytes < 0:
            raise ValueError(
                f"compensation_bytes must be non-negative, got {self.compensation_bytes}"
            )
        if self.compensation_bytes and self.keeps_fp32_master:
            raise ValueError(
                "a spec cannot keep both an fp32 master and a compensation buffer: the master "
                "already holds the bits the compensation exists to keep, so the budget would "
                "count a buffer nothing allocates"
            )

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

        A compensated recipe keeps no master and casts no gradient up for the whole model:
        bf16 weights, grads and compensation plus fp32 moments are 2+2+2+8 = **14** B/param.
        Its step casts one bounded chunk at a time (``KahanBf16AdamW.chunk_elems``), which is
        transient and not counted here.
        """
        master = 4 if self.keeps_fp32_master else 0
        grad_cast = 4 if self.keeps_fp32_master else 0
        return (
            param_bytes
            + grad_bytes
            + grad_cast
            + self.states_per_param * self.state_bytes
            + self.compensation_bytes
            + master
        )

    @property
    def checkpoint_bytes_per_param(self) -> int:
        """Bytes per optimized parameter in this optimizer's ``state_dict``: what a checkpoint
        sidecar holds for it beside the weights.

        The moments, plus an fp32 master for every parameter when the recipe keeps one
        (``MasterWeightAdamW``) or stores one in place of its compensation
        (``KahanBf16AdamW``: fp32 ``p + c``, the compensation recovered on load). So master
        and kahan both checkpoint 12, plain AdamW 2 x its state width. ``tests/test_optim.py``
        measures this off each built optimizer's real ``state_dict``.
        """
        masters = 4 if (self.keeps_fp32_master or self.compensation_bytes) else 0
        return self.states_per_param * self.state_bytes + masters


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

#: ``qd_train.optim.MasterWeightAdamW``: fp32 masters and fp32 moments over bf16 weights, 20
#: B/param measured on a GH200 (see :meth:`OptimizerSpec.bytes_per_param`). v5 ran it on
#: every row. One constant, because five files each built their own copy of it.
ADAMW_MASTER: Final[OptimizerSpec] = OptimizerSpec(
    name="AdamW+master", states_per_param=2, state_bytes=4, keeps_fp32_master=True
)

#: ``qd_train.optim.KahanBf16AdamW``: bf16 weights with a bf16 Kahan compensation buffer and
#: fp32 moments, no master: 14 B/param against the master recipe's 20. The 16-bit recipe for
#: a base that does not fit the master recipe; ``tests/test_optim.py`` measures how closely
#: it tracks the master trajectory, and against plain bf16, which loses the updates.
ADAMW_KAHAN: Final[OptimizerSpec] = OptimizerSpec(
    name="AdamW+kahan-bf16", states_per_param=2, state_bytes=4, compensation_bytes=2
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
    #: ``linear_attention`` KEY heads and their width (``linear_num_key_heads``,
    #: ``linear_key_head_dim``): q and k are this wide.
    linear_heads: int
    linear_head_dim: int
    #: ``linear_attention`` VALUE heads and their width (``linear_num_value_heads``,
    #: ``linear_value_head_dim``): v, the gate z, the per-head norm and ``out_proj``'s input
    #: are this wide, and so is the recurrent state's head count. Equal to the key heads on
    #: the 2B; twice them on Qwen3.5-4B/9B-Base, where counting key heads for both
    #: under-counted the activations. Required, with no default, because a default of "the
    #: key heads" is that under-count.
    linear_value_heads: int
    linear_value_head_dim: int
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
            "linear_value_heads",
            "linear_value_head_dim",
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
        if self.linear_value_heads % self.linear_heads:
            # transformers repeats q and k across the value heads (num_v // num_k); a ratio
            # that is not whole is a layer no kernel here runs.
            raise ValueError(
                f"linear_value_heads ({self.linear_value_heads}) must be a multiple of "
                f"linear_heads ({self.linear_heads})"
            )
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
        if not self.tied_embedding:
            # An untied head is a second [V, H] matrix the remap would have to cut too, and
            # nothing here cuts it: load_text_tower refuses an untied checkpoint. Budgeting
            # one cut would describe a run nobody can launch.
            raise ValueError(
                f"{self.name} has an untied output head; a remapped vocabulary is budgeted "
                "only for a tied embedding, the only kind the trainer loads"
            )
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
    linear_value_heads=16,
    linear_value_head_dim=128,
    vocab_size=248_320,
    params_total=1_881_825_088,
    params_embedding=508_559_360,
    tied_embedding=True,
    # config.json: "mamba_ssm_dtype": "float32". The recurrent state is not kept in the
    # compute dtype, and assuming it was would understate it by half.
    recurrent_state_bytes=4,
)


# --- a spec measured from a checkpoint ------------------------------------------------------

#: The prefix that separates the text tower from the vision tower and the MTP block in a
#: Qwen3.5 conditional-generation checkpoint. Measured from the 2B's header, not assumed:
#: ``model.language_model.`` (320), ``model.visual.`` (297) and ``mtp.`` (15) partition all
#: 632 tensors with nothing left over. ``qd_train.backbone`` loads exactly this prefix.
TEXT_PREFIX: Final[str] = "model.language_model."

#: Where a checkpoint stores an output head of its own -- only when it is untied.
_HEAD_PREFIX: Final[str] = "lm_head."

#: Stored, dropped on load, never trained: the vision tower and the multi-token-prediction
#: block.
_DROPPED_PREFIXES: Final[tuple[str, ...]] = ("model.visual.", "mtp.")

#: Safetensors headers are JSON behind a little-endian u64 length. A header larger than this
#: is refused rather than read into memory.
MAX_SAFETENSORS_HEADER_BYTES: Final[int] = 64 * 1024 * 1024

#: A bound on the tensors one snapshot may name, all shards together. The 2B has 632; a
#: checkpoint two orders of magnitude past that is not a Qwen3.5 base this code describes.
MAX_CHECKPOINT_TENSORS: Final[int] = 65_536

#: A bound on the shard fan-out. The 9B has 4.
_MAX_SHARDS: Final[int] = 1024

#: ``text_config.mamba_ssm_dtype`` -> bytes per recurrent-state element.
_STATE_DTYPE_BYTES: Final[Mapping[str, int]] = {"float32": 4, "bfloat16": 2, "float16": 2}


class CheckpointRefused(ValueError):
    """The checkpoint on disk is not one this module will describe, and it says why."""


def safetensors_header(path: Path) -> dict[str, Any]:
    """The JSON header of one safetensors file, without reading a byte of tensor data."""
    path = Path(path)
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise CheckpointRefused(f"{path}: truncated safetensors header length")
        n = struct.unpack("<Q", raw_len)[0]
        if n <= 0 or n > MAX_SAFETENSORS_HEADER_BYTES:
            raise CheckpointRefused(
                f"{path}: safetensors header claims {n} bytes, outside "
                f"(0, {MAX_SAFETENSORS_HEADER_BYTES}]. Refusing to read a header this size."
            )
        raw = handle.read(n)
    if len(raw) != n:
        raise CheckpointRefused(
            f"{path}: safetensors header claims {n} bytes and the file holds {len(raw)}"
        )
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointRefused(f"{path}: safetensors header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointRefused(
            f"{path}: safetensors header is a {type(header).__name__}, not an object"
        )
    return header


def checkpoint_tensor_index(snapshot: Path) -> dict[str, tuple[Path, tuple[int, ...]]]:
    """Every tensor in ``snapshot``'s shards, by its full name: ``(shard, shape)``.

    The one reader of a snapshot's tensor inventory: ``qd_train.backbone.text_tensor_index``
    filters it to the text tower, and [`spec_from_checkpoint`] counts it. A name stored in two
    shards is refused -- which one is the weight is not a question to guess at.
    """
    snapshot = Path(snapshot)
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise CheckpointRefused(
            f"{snapshot}: no .safetensors file. A text tower cannot be built from a snapshot "
            "that carries no weights, and an empty load would produce a randomly initialised "
            "model whose loss curve merely looks disappointing."
        )
    if len(shards) > _MAX_SHARDS:
        raise CheckpointRefused(f"{snapshot}: {len(shards)} shards, more than {_MAX_SHARDS}")
    index: dict[str, tuple[Path, tuple[int, ...]]] = {}
    for shard in shards:
        for name, entry in safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            if name in index:
                raise CheckpointRefused(
                    f"{snapshot}: tensor {name!r} appears in two shards "
                    f"({index[name][0].name} and {shard.name}); which one is the weight is "
                    "not a question this module will guess at."
                )
            if len(index) >= MAX_CHECKPOINT_TENSORS:
                raise CheckpointRefused(
                    f"{snapshot}: more than {MAX_CHECKPOINT_TENSORS} tensors; the 2B has 632. "
                    "This is a different model."
                )
            shape = entry.get("shape") if isinstance(entry, dict) else None
            if not isinstance(shape, list) or not all(
                isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
            ):
                raise CheckpointRefused(f"{shard}: tensor {name!r} has no valid shape: {shape!r}")
            index[name] = (shard, tuple(shape))
    return index


def head_is_tied(config: Mapping[str, Any], tensor_names: Iterable[str], *, where: str) -> bool:
    """Whether the output head *is* the embedding, decided by the config and the storage.

    Two signals, and both must agree. transformers ties a Qwen3.5 checkpoint's head by the
    **top-level** ``tie_word_embeddings`` (``PreTrainedModel`` reads ``self.config``, the
    composite config, and ``Qwen3_5Config`` defaults it to false whatever ``text_config``
    says); ``text_config`` may state its own. The storage is the other signal: an untied head
    is a tensor of its own under ``lm_head.``, and a tied one is not stored at all.

    True when every stated flag is true and no head is stored; False when every stated flag is
    false and a head is stored. Anything else is refused -- an absent top-level flag, a
    non-bool, two flags that disagree, or flags the storage contradicts -- because each is a
    model whose head would be a guess. ``ojas-qwen35``'s reader refuses a false flag at
    either level too; this one says why.
    """
    top = config.get("tie_word_embeddings")
    if top is None:
        raise CheckpointRefused(
            f"{where}: config.json states no top-level tie_word_embeddings. transformers "
            "reads that flag for this architecture and defaults it to false, so whether the "
            "head is the embedding would be a default, not a fact."
        )
    stated: dict[str, object] = {"tie_word_embeddings": top}
    text = config.get("text_config")
    if isinstance(text, Mapping) and text.get("tie_word_embeddings") is not None:
        stated["text_config.tie_word_embeddings"] = text["tie_word_embeddings"]
    for key, value in stated.items():
        if not isinstance(value, bool):
            raise CheckpointRefused(f"{where}: {key} is {value!r}, not a bool")
    stored = sorted(name for name in tensor_names if name.startswith(_HEAD_PREFIX))
    flags = set(stated.values())
    if flags == {True} and not stored:
        return True
    if flags == {False} and stored:
        return False
    raise CheckpointRefused(
        f"{where}: whether the output head is the embedding is contradictory: {stated}, and "
        f"the checkpoint stores {len(stored)} tensor(s) under {_HEAD_PREFIX!r} "
        f"({stored[:4]}). A tied head is not stored; an untied one is."
    )


def _config_field(text: Mapping[str, Any], key: str, kind: type, *, where: str) -> Any:
    value = text.get(key)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise CheckpointRefused(
            f"{where}: text_config.{key} is {value!r}, not a {kind.__name__}. A spec built "
            "on a default here would describe a model nobody read."
        )
    return value


def spec_from_checkpoint(snapshot: Path) -> ModelSpec:
    """A [`ModelSpec`] measured from a checkpoint on disk rather than taken from a constant.

    The parameter count is summed over the text tower's tensors and, when the head is untied,
    the head's -- read off the safetensors headers, not derived from the config. The
    architecture is read from ``config.json``'s ``text_config``, and every field is required:
    nothing here falls back to a default, because a defaulted head count or tie is exactly
    how a 2B-shaped budget gets applied to a 4B. ``tests/test_memory.py`` checks that the 2B
    snapshot reproduces [`QWEN3_5_2B_TEXT`] field for field.

    It describes an untied head too (the 9B's), so a budget can be computed for it; loading
    one is ``qd_train.backbone``'s refusal, not this function's.
    """
    snapshot = Path(snapshot)
    where = str(snapshot)
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointRefused(f"{where}: config.json is unreadable: {exc}") from exc
    text = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text, dict):
        raise CheckpointRefused(
            f"{where}: config.json has no text_config object. This describes the text tower "
            "of a Qwen3.5 conditional-generation checkpoint."
        )
    index = checkpoint_tensor_index(snapshot)
    known = (TEXT_PREFIX, _HEAD_PREFIX, *_DROPPED_PREFIXES)
    unknown = sorted(name for name in index if not name.startswith(known))
    if unknown:
        raise CheckpointRefused(
            f"{where}: {len(unknown)} tensor(s) under a prefix this module has not seen "
            f"({unknown[:4]}). Counting them or dropping them would be a guess."
        )
    tied = head_is_tied(config, index, where=where)

    def need_int(key: str) -> int:
        return int(_config_field(text, key, int, where=where))

    vocab, hidden = need_int("vocab_size"), need_int("hidden_size")
    embedding_name = TEXT_PREFIX + "embed_tokens.weight"
    embedding_shape = index[embedding_name][1] if embedding_name in index else None
    if embedding_shape != (vocab, hidden):
        raise CheckpointRefused(
            f"{where}: {embedding_name} is {embedding_shape}, and text_config says "
            f"({vocab}, {hidden})"
        )
    layer_types = _config_field(text, "layer_types", list, where=where)
    if not all(t in ("full_attention", "linear_attention") for t in layer_types):
        raise CheckpointRefused(
            f"{where}: text_config.layer_types names a layer kind other than full_attention "
            f"and linear_attention: {sorted(set(layer_types))}"
        )
    state_dtype = _config_field(text, "mamba_ssm_dtype", str, where=where)
    if state_dtype not in _STATE_DTYPE_BYTES:
        raise CheckpointRefused(
            f"{where}: text_config.mamba_ssm_dtype {state_dtype!r} is not one of "
            f"{sorted(_STATE_DTYPE_BYTES)}"
        )
    model_dir = snapshot.parent.parent.name
    return ModelSpec(
        name=f"{model_dir.removeprefix('models--').replace('--', '/')} (text tower)",
        hidden_size=hidden,
        intermediate_size=need_int("intermediate_size"),
        n_full_attention_layers=layer_types.count("full_attention"),
        n_linear_attention_layers=layer_types.count("linear_attention"),
        q_heads=need_int("num_attention_heads"),
        kv_heads=need_int("num_key_value_heads"),
        head_dim=need_int("head_dim"),
        attn_output_gate=bool(_config_field(text, "attn_output_gate", bool, where=where)),
        linear_heads=need_int("linear_num_key_heads"),
        linear_head_dim=need_int("linear_key_head_dim"),
        linear_value_heads=need_int("linear_num_value_heads"),
        linear_value_head_dim=need_int("linear_value_head_dim"),
        vocab_size=vocab,
        params_total=sum(
            math.prod(shape)
            for name, (_shard, shape) in index.items()
            if name.startswith((TEXT_PREFIX, _HEAD_PREFIX))
        ),
        params_embedding=vocab * hidden,
        tied_embedding=tied,
        recurrent_state_bytes=_STATE_DTYPE_BYTES[state_dtype],
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

    ``retained_linear_layers`` / ``retained_full_layers`` price **selective** checkpointing:
    under ``recompute="full"``, that many decoder layers of each kind are NOT checkpointed
    and keep every saved tensor, on top of the boundaries and the one recomputed layer. Zero
    of each, the default, is per-layer checkpointing everywhere -- exactly the model every
    earlier budget used.
    """

    recompute: str = "full"
    attention: str = "flash"
    retained_linear_layers: int = 0
    retained_full_layers: int = 0

    def __post_init__(self) -> None:
        if self.recompute not in ("none", "full"):
            raise ValueError(f"recompute must be 'none' or 'full', got {self.recompute!r}")
        if self.attention not in ("flash", "math"):
            raise ValueError(f"attention must be 'flash' or 'math', got {self.attention!r}")
        for name in ("retained_linear_layers", "retained_full_layers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if self.recompute == "none" and (self.retained_linear_layers or self.retained_full_layers):
            raise ValueError(
                "retained layers are a statement about which layers are NOT checkpointed, "
                "and under recompute='none' none is; the pair would price one policy and "
                "describe another"
            )

    @property
    def retained_layers(self) -> int:
        return self.retained_linear_layers + self.retained_full_layers

    def describe(self) -> str:
        """``"full"`` / ``"none"`` exactly as before; selective names its retained layers."""
        if not self.retained_layers:
            return self.recompute
        return (
            f"full-except-{self.retained_linear_layers}-linear-"
            f"{self.retained_full_layers}-full"
        )

    def _check_retained(self, m: ModelSpec) -> None:
        if (
            self.retained_linear_layers > m.n_linear_attention_layers
            or self.retained_full_layers > m.n_full_attention_layers
        ):
            raise ValueError(
                f"{self.retained_linear_layers} linear / {self.retained_full_layers} full "
                f"retained layers exceed {m.name}'s {m.n_linear_attention_layers} / "
                f"{m.n_full_attention_layers}"
            )

    def linear_layer_elements(self, m: ModelSpec) -> int:
        """One ``linear_attention`` layer's saved elements per token.

        q and k are key-heads wide; v, z, the per-head norm and ``out_proj``'s input are
        value-heads wide (transformers' ``Qwen3_5GatedDeltaNet``: ``conv_dim = key_dim * 2 +
        value_dim``, and ``in_proj_z`` and ``out_proj`` are ``value_dim``). On the 2B the two
        are equal and ``value_dim == hidden_size``, so its figures are what they always were.
        """
        key_dim = m.linear_heads * m.linear_head_dim
        value_dim = m.linear_value_heads * m.linear_value_head_dim
        qkv = 2 * key_dim + value_dim  # in_proj_qkv -> [6144, 2048] on the 2B
        return (
            m.hidden_size  # residual entering the layer
            + m.hidden_size  # input_layernorm output
            + qkv  # in_proj_qkv output
            + qkv  # conv1d output
            + value_dim  # per-head norm output
            + value_dim  # in_proj_z gate output
            + value_dim  # gated attention output, out_proj input
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
        # Selective: every retained layer keeps its whole set, all at once. Its boundary is
        # already counted above, so this over-counts by one hidden vector per retained
        # layer -- on the side that refuses a run that would have fit, never the other.
        self._check_retained(m)
        retained = (
            self.retained_linear_layers * self.linear_layer_elements(m)
            + self.retained_full_layers * self.full_layer_elements(m)
        )
        return boundaries + peak + retained

    def score_matrix_elements(self, m: ModelSpec, *, rows: int, width: int) -> int:
        """The materialised attention matrix, or zero when it is never formed.

        Under ``recompute="full"`` one layer is live at a time, plus every retained
        full-attention layer; under ``"none"`` every full-attention layer's matrix is
        retained at once.
        """
        if self.attention == "flash":
            return 0
        live_layers = (
            1 + self.retained_full_layers if self.recompute == "full"
            else m.n_full_attention_layers
        )
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
    if optimizer.compensation_bytes and param_dtype != "bf16":
        raise ValueError(
            f"optimizer spec {optimizer.name!r} keeps a {optimizer.compensation_bytes}-byte "
            f"compensation buffer, which only a bf16 tower has (KahanBf16AdamW refuses any "
            f"other low-precision dtype and gives fp32 parameters none); param_dtype is "
            f"{param_dtype!r}, so the budget would count a buffer nothing allocates"
        )

    trainable = model.trainable_params(vocab_size=vocab_size)
    positions = rows * width

    activation_bytes = positions * acts.elements_per_token(model) * a_bytes
    score_bytes = acts.score_matrix_elements(model, rows=rows, width=width) * a_bytes
    # The recurrent state is [rows, value_heads, key_head_dim, value_head_dim] per
    # linear-attention layer (q and k are repeated across the value heads) and does not
    # scale with sequence length. Under checkpointing one layer is live.
    live_linear = (
        1 + acts.retained_linear_layers if acts.recompute == "full"
        else model.n_linear_attention_layers
    ) if model.n_linear_attention_layers else 0
    recurrent_bytes = (
        live_linear
        * rows
        * model.linear_value_heads
        * model.linear_head_dim
        * model.linear_value_head_dim
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
        # The compensation buffer is optimizer state: it exists only for the optimizer, and a
        # spec that carries one budgets it here rather than under a field of its own.
        optimizer_bytes=trainable * (
            optimizer.states_per_param * optimizer.state_bytes + optimizer.compensation_bytes
        ),
        master_bytes=trainable * 4 if optimizer.keeps_fp32_master else 0,
        activation_bytes=activation_bytes,
        score_matrix_bytes=score_bytes,
        recurrent_state_bytes=recurrent_bytes,
        loss_head_bytes=loss_head_bytes,
        safety_bytes=safety_bytes,
        recompute=acts.describe(),
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
