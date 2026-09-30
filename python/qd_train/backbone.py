"""The real Qwen3.5-2B text tower: constructed from the checkpoint, remapped, checkpointed.

Nothing in this repository had ever constructed the model it exists to train. The only
``from_pretrained`` was ``stack/verify_fast_path.py:174``, a readiness gate; every caller of
[`qd_train.fused_ce.fused_linear_cross_entropy`] built a toy model inside the caller, and
``tools/real_ft_run.py`` says so in as many words -- *"what this tool supplies is a randomly-
initialised backbone"*. This module is the backbone, and it is the canonical owner: a second
construction of the text tower somewhere else is a second set of assumptions about which 320
tensors are the model.

## Which 320 tensors, and why the text-only config wins

The checkpoint's top-level config is ``Qwen3_5ForConditionalGeneration``. Its tensors divide
into exactly three prefixes -- ``model.language_model.`` (320), ``model.visual.`` (297) and
``mtp.`` (15) -- and only the first is trained here.

There were two honest ways to get it. Load the whole conditional-generation model and take
``.model.language_model``, or build a ``Qwen3_5TextModel`` from ``config.text_config`` and
load the text tensors into it. **The second is what this module does, because the key sets
are a bijection and that is a checkable claim rather than an argument.** Measured 2026-09-20
against transformers 5.12.1 and snapshot ``b1485b2f``: ``Qwen3_5TextModel(text_config)`` has
320 ``state_dict`` entries, the checkpoint has 320 ``model.language_model.*`` tensors, and
with the prefix stripped there are **0 keys missing, 0 extra and 0 shape mismatches**. The
first route would allocate the vision tower's 331,416,576 parameters and the MTP block's
60,828,160 -- 785 MB of bf16 -- in order to drop them, and would let a load succeed while
quietly holding a model this repository never trains.

[`load_text_tower`] re-checks the bijection on every load rather than trusting this
paragraph. A snapshot whose key set differs is refused, with both differences named: that is
the check that catches a re-download, a different revision, or a transformers upgrade that
renames a tensor. The alternative -- ``strict=False`` -- is how a model trains with randomly
initialised layers and a loss curve that looks merely disappointing.

## The head is the embedding, and it is not created twice

``Qwen3_5TextModel.get_output_embeddings()`` returns ``None``: a base tower has no LM head,
and the checkpoint has no ``lm_head`` key either, because ``tie_word_embeddings`` is true.
So the output head **is** ``embed_tokens.weight``, ``[V, H]`` -- which is exactly the
``weight`` argument [`qd_train.fused_ce.fused_linear_cross_entropy`] takes, untransposed.
This module therefore never constructs an ``nn.Linear`` head. Constructing one would double
the 508,559,360-parameter embedding that [`qd_train.remap`] exists to shrink, and the two
copies would then drift apart under training with nothing reporting it.

That is also why the remap is applied through [`qd_train.remap.apply_remap_to_model`] rather
than by slicing here: it already detects the tie by two independent signals and says which
one it believed. On this model it takes the ``out_head is None`` branch, which is correct
and is *not* the same as "there was no tie" -- the tie is total, because there is one tensor.
[`remap_text_tower`] then verifies the slice **in both directions over every kept row**,
because a tied embedding sliced wrong trains the wrong rows and no loss curve shows it.

## Gradient checkpointing is a decision, not a default

Without it one row of the 34,522-token bucket needs 108.15 GB against a 96 GB GH200; with
per-layer checkpointing that bucket fits at 8 rows. It did not exist anywhere in this
repository, and a rented GH200 is running.

So ``gradient_checkpointing`` is a **required** keyword on [`load_text_tower`] -- there is
no default to forget -- and turning it off emits a [`GradientCheckpointingDisabled`] warning
carrying the arithmetic. Enabling it is *verified* rather than assumed: transformers 5.12.1
implements it by way of ``GradientCheckpointingLayer.__call__``, gated on
``self.gradient_checkpointing and self.training``, and ``modeling_qwen3_5.py`` contains no
``_gradient_checkpointing_func`` call of its own. That is a mechanism inherited from a base
class, which a future version can drop while ``supports_gradient_checkpointing`` stays
``True``. [`load_text_tower`] reads the flag back off every decoder layer and refuses when
any of them did not take it.

Measurement, not belief: [`saved_activation_bytes`] counts what the autograd graph actually
retains, via ``torch.autograd.graph.saved_tensors_hooks``. It is exact, it needs no CUDA,
and it is the quantity checkpointing reduces. On an 8-layer tiny tower at ``[2, 128]``,
measured on this host: **1074 saved tensors / 32,463,552 bytes off, 14 tensors / 724,224
bytes on** -- a 97.8% reduction. ``test_backbone.py`` re-measures it rather than quoting it.

## The optimizer precision is still unmade, and this module keeps it that way

``python/qd_train/memory.py`` computes all four cells of the fp32-master / vocabulary grid
and hardcodes none. Neither does this: [`load_text_tower`] takes an
[`qd_train.memory.OptimizerSpec`] with **no default**, so every caller states which recipe
its budget describes. A default here would silently pick one of the four and report it as
the number.

## What this module does not do

It does not open training data. Rule 3's door is on the data path --
[`qd_train.artifacts.assert_shard_trainable`] and
[`qd_train.data_access.assert_path_not_held_out`] -- and [`QwenDecisionStep`] receives
``Batch`` objects that a reader already took through it. Adding a third check here over a
path this module never reads would be a check that cannot fire, which is worse than none.
"""

from __future__ import annotations

import json
import struct
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .artifacts import Batch, RemapTable
from .fused_ce import fused_linear_cross_entropy
from .heads import SpanPointerHead, plan_span_batch
from .memory import (
    BYTES_PER_ELEMENT,
    QWEN3_5_2B_TEXT,
    ActivationModel,
    ModelSpec,
    OptimizerSpec,
    StepFootprint,
    estimate_step,
)
from .optim import DEFAULT_BETA2, apply_lr, build_optimizer, layerwise_param_groups
from .remap import RemapApplication, apply_remap_to_model
from .trainer import Supervision

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "MAX_TEXT_TENSORS",
    "TEXT_PREFIX",
    "BackboneContractViolation",
    "GradientCheckpointingDisabled",
    "QwenDecisionStep",
    "TextTower",
    "footprint_at",
    "load_text_tower",
    "remap_text_tower",
    "saved_activation_bytes",
    "text_tensor_index",
]

#: The prefix that separates the text tower from the vision tower and the MTP block in
#: ``model.safetensors-*.safetensors``. Measured from the checkpoint header, not assumed:
#: the three prefixes partition all 632 tensors with nothing left over.
TEXT_PREFIX: Final[str] = "model.language_model."

#: A bound on the header scan. The real checkpoint has 320 text tensors; anything an order
#: of magnitude past that is a different model, and an unbounded loop over a header this
#: code did not write is the fan-out ``CLAUDE.md`` asks to bound.
MAX_TEXT_TENSORS: Final[int] = 4096

#: Safetensors headers are JSON prefixed by a little-endian u64 length. A header larger than
#: this is refused rather than read into memory.
_MAX_HEADER_BYTES: Final[int] = 64 * 1024 * 1024


class BackboneContractViolation(Exception):
    """The checkpoint, the model class and this module do not agree about the text tower."""


class GradientCheckpointingDisabled(UserWarning):
    """Raised as a warning when a tower is built without per-layer gradient checkpointing.

    A warning class of its own so a caller that genuinely wants it off can silence exactly
    this and stay loud about everything else -- and so a test can assert it was emitted.
    """


# --- reading the checkpoint's own header ---------------------------------------------------


def _safetensors_shards(snapshot: Path) -> list[Path]:
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise BackboneContractViolation(
            f"{snapshot}: no .safetensors file. A text tower cannot be built from a snapshot "
            "that carries no weights, and an empty load would produce a randomly initialised "
            "model whose loss curve merely looks disappointing."
        )
    return shards


def _header_of(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise BackboneContractViolation(f"{path}: truncated safetensors header length")
        n = struct.unpack("<Q", raw_len)[0]
        if n <= 0 or n > _MAX_HEADER_BYTES:
            raise BackboneContractViolation(
                f"{path}: safetensors header claims {n} bytes, outside "
                f"(0, {_MAX_HEADER_BYTES}]. Refusing to read a header this size."
            )
        return json.loads(handle.read(n))


def text_tensor_index(snapshot: Path) -> dict[str, tuple[Path, tuple[int, ...]]]:
    """Every text-tower tensor in ``snapshot``, keyed by its name with [`TEXT_PREFIX`] gone.

    The value carries the shard it lives in and its shape, so a caller can check the shape
    the file states against the shape the model wants without loading either.

    Names are returned *without* the prefix because that is the form
    ``Qwen3_5TextModel.state_dict()`` uses; keeping the raw name would make every comparison
    downstream restate the prefix, and a restated constant is one that can drift.
    """
    snapshot = Path(snapshot)
    index: dict[str, tuple[Path, tuple[int, ...]]] = {}
    for shard in _safetensors_shards(snapshot):
        header = _header_of(shard)
        for name, entry in header.items():
            if name == "__metadata__" or not name.startswith(TEXT_PREFIX):
                continue
            if len(index) >= MAX_TEXT_TENSORS:
                raise BackboneContractViolation(
                    f"{snapshot}: more than {MAX_TEXT_TENSORS} tensors under {TEXT_PREFIX!r}. "
                    "The real checkpoint has 320; this is a different model."
                )
            short = name[len(TEXT_PREFIX) :]
            if short in index:
                raise BackboneContractViolation(
                    f"{snapshot}: tensor {short!r} appears in two shards "
                    f"({index[short][0].name} and {shard.name}); which one is the weight is "
                    "not a question this module will guess at."
                )
            index[short] = (shard, tuple(entry["shape"]))
    if not index:
        raise BackboneContractViolation(
            f"{snapshot}: no tensor is named {TEXT_PREFIX!r}*. Either this is not a "
            "Qwen3.5 conditional-generation checkpoint, or the text tower is stored under a "
            "prefix this module has not seen. Refusing rather than loading nothing."
        )
    return index


# --- the tower -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextTower:
    """A loaded text tower and every number a ledger row wants about the load.

    ``model`` is a ``Qwen3_5TextModel``. It is typed as ``Any`` because transformers is the
    optional ``mac`` extra and importing the class at module scope would make this file
    unimportable in the repo venv, where the torch-free half of the suite runs.
    """

    model: Any
    snapshot: Path
    spec: ModelSpec
    n_tensors_loaded: int
    vocab_size: int
    hidden_size: int
    gradient_checkpointing: bool
    dtype: str
    device: str
    #: Which attention kernel ran, resolved rather than assumed. Until it was passed, this
    #: was whatever ``transformers`` picked: `sdpa` on both hosts today, but the two hosts
    #: are on different versions (5.12.1 here, 5.17.0 on the GH200) and a library upgrade or
    #: an installed `flash_attn` moves it without moving anything a ledger row records.
    attn_implementation: str
    optimizer: OptimizerSpec
    footprint: StepFootprint
    remap: RemapApplication | None = None

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """``[V, H]`` -- the tied embedding, which *is* the output head.

        Handed straight to [`qd_train.fused_ce.fused_linear_cross_entropy`] as its
        ``weight``. There is no second tensor, by design; see the module docstring.
        """
        return self.model.get_input_embeddings().weight

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot": str(self.snapshot),
            "model": self.spec.name,
            "n_tensors_loaded": self.n_tensors_loaded,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dtype": self.dtype,
            "device": self.device,
            "trainable_params": self.footprint.trainable_params,
            "footprint_total_bytes": self.footprint.total_bytes,
            "footprint_provenance": self.footprint.provenance,
            "remap": None if self.remap is None else self.remap.to_json(),
        }


def _verify_checkpointing_took(model: Any, *, enabled: bool) -> None:
    """Read the flag back off every decoder layer. Refuse a request that did not take.

    transformers 5.12.1 implements checkpointing in ``GradientCheckpointingLayer.__call__``,
    which ``Qwen3_5DecoderLayer`` inherits; ``modeling_qwen3_5.py`` has no call of its own.
    A version that drops the base class would leave ``supports_gradient_checkpointing`` true,
    ``gradient_checkpointing_enable()`` silent, and every activation retained -- an OOM on
    the rented card, hours in, with the flag reading ``True`` the whole time.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise BackboneContractViolation(
            f"{type(model).__name__} exposes no `.layers`, so whether gradient checkpointing "
            "took effect cannot be read back. Refusing: an unverified checkpointing flag is "
            "the one that costs a rented card."
        )
    disagreed = [
        i for i, layer in enumerate(layers) if bool(layer.gradient_checkpointing) != enabled
    ]
    if disagreed:
        raise BackboneContractViolation(
            f"gradient_checkpointing was requested {enabled} but layer(s) "
            f"{disagreed[:16]} of {len(layers)} report "
            f"{not enabled}. transformers implements this through "
            "GradientCheckpointingLayer.__call__, which Qwen3_5DecoderLayer inherits rather "
            "than defines; a version that changed that would fail exactly here. The flag was "
            "not believed, which is why this is an error and not a log line."
        )


def _verify_spec_describes(spec: ModelSpec, *, model: Any, config: Any) -> None:
    """Refuse a memory spec that describes a different model than the one just loaded.

    ``spec`` defaults to [`qd_train.memory.QWEN3_5_2B_TEXT`], which is right for the real
    checkpoint and wrong for anything else. Without this check a caller loading any other
    tower would receive a ``StepFootprint`` describing the 2B model -- a budget that reads
    exactly like a checked one and refers to a model nobody loaded.

    The four numbers compared are the ones the arithmetic is actually built on: hidden size
    and layer counts drive the activation term, and the vocabulary drives the tied
    embedding. ``params_total`` is not re-derived here -- ``ModelSpec`` documents it as
    measured from the checkpoint, and recomputing it would be the second opinion that
    module deliberately refuses to hold.
    """
    layer_types = list(getattr(config, "layer_types", []) or [])
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_linear = sum(1 for t in layer_types if t == "linear_attention")
    observed = {
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "n_full_attention_layers": n_full,
        "n_linear_attention_layers": n_linear,
    }
    disagreements = {
        field: (getattr(spec, field), value)
        for field, value in observed.items()
        if getattr(spec, field) != value
    }
    if disagreements:
        detail = ", ".join(
            f"{field}: spec says {want}, this model is {got}"
            for field, (want, got) in sorted(disagreements.items())
        )
        raise BackboneContractViolation(
            f"the memory spec {spec.name!r} does not describe the tower that was loaded "
            f"({detail}). The footprint would be arithmetic about a different model, which "
            "is indistinguishable in a report from a budget that was checked. Pass a "
            "ModelSpec matching this checkpoint."
        )


def load_text_tower(
    snapshot: str | Path,
    *,
    gradient_checkpointing: bool,
    optimizer: OptimizerSpec,
    attn_implementation: str,
    device: str = "cpu",
    dtype: str = "bf16",
    rows: int = 1,
    width: int = 34_522,
    spec: ModelSpec = QWEN3_5_2B_TEXT,
    config_overrides: Mapping[str, Any] | None = None,
) -> TextTower:
    """Build the text tower from ``snapshot`` and load exactly its 320 text tensors.

    Args:
        snapshot: the local checkpoint directory -- the one whose hash is ``backbone_commit``
            in the ledger rows.
        gradient_checkpointing: **required**. There is no default because off is what every
            caller gets by accident, and off is 108.15 GB for one row of the widest bucket.
            ``False`` warns [`GradientCheckpointingDisabled`] rather than passing quietly.
        optimizer: **required**, and deliberately without a default: the fp32-master choice
            is unmade (see ``memory.py``), and defaulting it here would pick one of the four
            cells and report it as the answer.
        attn_implementation: **required**, for the third time on this function and the same
            reason. Without it the kernel is whatever ``transformers`` resolves
            ``config._attn_implementation`` to, which depends on the library version and on
            what happens to be installed -- ``sdpa`` on both of this project's hosts today,
            which run 5.12.1 and 5.17.0. A default here would be this module deciding the
            arithmetic and not recording that it had. Passed through to the config, so an
            unsupported name is refused by transformers rather than silently ignored.
        device: where the weights land. ``"cpu"`` by default, because the machine that can
            load this checkpoint is not necessarily the machine that can train on it.
        dtype: element type for the loaded weights, named as ``memory.BYTES_PER_ELEMENT``
            names them.
        rows, width: the batch shape the returned ``footprint`` describes. The default width
            is the widest real bucket, so a caller that supplies neither gets the arithmetic
            for the case that actually binds rather than for a comfortable one.
        config_overrides: applied to the text config before construction. Present so a test
            can build a tiny tower through this same function; a production caller passes
            nothing.

    Returns:
        A [`TextTower`] carrying the model and what the load measured.

    Raises:
        BackboneContractViolation: if the snapshot's text key set and the model class's
            ``state_dict`` key set are not identical, in either direction or in any shape.
    """
    # `torch.optim.AdamW` keeps `exp_avg` and `exp_avg_sq` in the PARAMETER's dtype and holds
    # no fp32 master copy, which `memory.OptimizerSpec`'s docstring states and nothing
    # enforced. Passing ADAMW_FP32 (state_bytes=4) for a bf16 tower budgeted 16 B/param
    # against the 8 B/param torch actually allocates -- 28.04 GiB predicted, 14.02 GiB
    # measured on a GH200. Over-budgeting never crashes, which is why it survived; a
    # footprint that does not describe the run cannot decide whether the next one fits.
    if not optimizer.keeps_fp32_master:
        state_should_be = BYTES_PER_ELEMENT[dtype]
        if optimizer.state_bytes != state_should_be:
            raise BackboneContractViolation(
                f"optimizer spec {optimizer.name!r} keeps {optimizer.state_bytes}-byte states "
                f"but this tower's parameters are {dtype} ({state_should_be} bytes each), and "
                "torch.optim.AdamW keeps exp_avg and exp_avg_sq in the parameter's own dtype "
                "with no fp32 master. The budget would describe a layout nothing builds. Pass "
                f"a spec whose state_bytes is {state_should_be} (ADAMW_BF16 for bf16, "
                "ADAMW_FP32 for fp32), or a spec with keeps_fp32_master=True, which "
                "qd_train.optim.MasterWeightAdamW now implements -- and which is the "
                "numerically correct choice for a long run, measured: at beta2=0.999 a bf16 "
                "exp_avg_sq settles at 0.5 against a true 1.0 after 383 steps and cannot "
                "recover when the gradient scale changes (tools/moment_precision.py). It "
                "costs 16 B/param against 8. That is a recipe decision, not a default to "
                "inherit, which is why nothing here picks it for you."
            )

    try:
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.qwen3_5 import Qwen3_5TextModel
    except ImportError as exc:  # pragma: no cover - exercised only in a torch-free env
        raise RuntimeError(
            "load_text_tower needs torch, transformers and safetensors, which are the "
            f"optional `mac` extra (pip install -e '.[mac]'): {exc}"
        ) from exc

    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise BackboneContractViolation(f"{snapshot} is not a directory")

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype)
    if torch_dtype is None:
        raise BackboneContractViolation(
            f"dtype={dtype!r} is not one this module loads; known: bf16, fp16, fp32. "
            "Guessing an element size would put an unchecked number in the budget."
        )

    full_config = AutoConfig.from_pretrained(snapshot)
    text_config = getattr(full_config, "text_config", None)
    if text_config is None:
        raise BackboneContractViolation(
            f"{snapshot}: the config has no `text_config`. This module builds the text tower "
            f"of a conditional-generation checkpoint; {type(full_config).__name__} does not "
            "carry one, so which sub-config is the language model is not decided here."
        )
    for key, value in (config_overrides or {}).items():
        setattr(text_config, key, value)
    # Before either construction below, so the skeleton whose state_dict is compared against
    # the checkpoint is built the same way as the model that gets the weights. transformers
    # validates the name here and raises on one it does not implement.
    text_config._attn_implementation = attn_implementation

    index = text_tensor_index(snapshot)
    with torch.device("meta"):
        skeleton = Qwen3_5TextModel(text_config)
    wanted = {name: tuple(t.shape) for name, t in skeleton.state_dict().items()}

    missing = sorted(set(wanted) - set(index))
    extra = sorted(set(index) - set(wanted))
    mismatched = [
        (name, wanted[name], index[name][1])
        for name in sorted(set(wanted) & set(index))
        if wanted[name] != index[name][1]
    ]
    if missing or extra or mismatched:
        raise BackboneContractViolation(
            f"{snapshot}: the checkpoint's {TEXT_PREFIX!r} tensors and "
            f"{type(skeleton).__name__}'s state_dict are not the same set.\n"
            f"  wanted by the model, absent from the checkpoint ({len(missing)}): "
            f"{missing[:8]}\n"
            f"  in the checkpoint, unwanted by the model ({len(extra)}): {extra[:8]}\n"
            f"  present in both with different shapes ({len(mismatched)}): {mismatched[:8]}\n"
            "Loading the intersection would leave the rest randomly initialised, which is a "
            "model that trains and never says it is not the checkpoint."
        )

    model = Qwen3_5TextModel(text_config).to(dtype=torch_dtype)
    state: dict[str, torch.Tensor] = {}
    by_shard: dict[Path, list[str]] = {}
    for short, (shard, _shape) in index.items():
        by_shard.setdefault(shard, []).append(short)
    for shard, names in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for short in names:
                state[short] = handle.get_tensor(TEXT_PREFIX + short).to(torch_dtype)

    incompatible = model.load_state_dict(state, strict=True)
    if getattr(incompatible, "missing_keys", ()) or getattr(incompatible, "unexpected_keys", ()):
        raise BackboneContractViolation(  # pragma: no cover - strict=True raises first
            f"load_state_dict(strict=True) still reported {incompatible}"
        )
    model = model.to(device=device)

    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
        warnings.warn(
            "gradient checkpointing is OFF for this text tower: every layer's activations "
            "are retained. Measured with qd_train.memory at the 13,787-token remapped "
            "vocabulary, one row of the 34,522-token bucket needs 108.15 GB without it and "
            "25.85 GB with it; a 96 GB GH200 therefore admits 0 rows of that bucket off and "
            "8 rows on (87.33 GB). Off is what a caller gets by accident, so it is a "
            "warning and not a log line.",
            GradientCheckpointingDisabled,
            stacklevel=2,
        )
    _verify_checkpointing_took(model, enabled=gradient_checkpointing)

    embedding = model.get_input_embeddings().weight
    _verify_spec_describes(spec, model=model, config=text_config)
    footprint = estimate_step(
        spec,
        rows=rows,
        width=width,
        optimizer=optimizer,
        param_dtype=dtype,
        grad_dtype=dtype,
        activation_dtype=dtype,
        activations=ActivationModel(
            recompute="full" if gradient_checkpointing else "none"
        ),
        vocab_size=int(embedding.shape[0]),
    )
    return TextTower(
        model=model,
        snapshot=snapshot,
        spec=spec,
        n_tensors_loaded=len(state),
        vocab_size=int(embedding.shape[0]),
        hidden_size=int(embedding.shape[1]),
        gradient_checkpointing=gradient_checkpointing,
        dtype=dtype,
        # What the model resolved, not what was asked for. They agree today; asking the
        # model is what keeps the row true on the day they stop agreeing.
        attn_implementation=str(model.config._attn_implementation),
        device=str(device),
        optimizer=optimizer,
        footprint=footprint,
    )


# --- the remap ------------------------------------------------------------------------------


def footprint_at(
    tower: TextTower, *, rows: int, width: int, vocab_size: int | None = None
) -> StepFootprint:
    """``estimate_step`` for this tower at a ``rows x width`` batch.

    The one place a loaded tower's step is priced, so a caller budgeting several batch
    shapes uses the arithmetic the load recorded. ``vocab_size`` defaults to the tower's own.
    """
    return estimate_step(
        tower.spec,
        rows=rows,
        width=width,
        # The tower's own optimizer spec, carried rather than reconstructed: rebuilding it
        # from the footprint's byte totals would hardcode AdamW's two fp32 states and
        # silently re-budget an SGD run as an AdamW one.
        optimizer=tower.optimizer,
        param_dtype=tower.dtype,
        grad_dtype=tower.dtype,
        activation_dtype=tower.dtype,
        activations=ActivationModel(
            recompute="full" if tower.gradient_checkpointing else "none"
        ),
        vocab_size=tower.vocab_size if vocab_size is None else vocab_size,
    )


def remap_text_tower(tower: TextTower, remap: RemapTable) -> TextTower:
    """Slice the tied embedding to ``remap``'s vocabulary, and verify every kept row.

    The slicing itself is [`qd_train.remap.apply_remap_to_model`] -- this does not
    reimplement it. What is added here is the check that function cannot make: it verifies
    shapes, the tie and the row *count*, and nothing verifies the row *contents*.

    Both directions, over every kept row, not a sample:

    * forward -- ``new[j] == old[new_to_old[j]]`` for all ``j`` in the new vocabulary;
    * reverse -- ``new[old_to_new[o]] == old[o]`` for every old id the table kept.

    The reverse is implied by the forward when the table really is an inverse pair, and
    ``RemapTable`` validates that. Checking it anyway is what catches the case where the two
    statements of the mapping have drifted apart, which is the failure that trains the wrong
    rows with a loss curve that looks entirely normal.
    """
    import torch

    embedding = tower.model.get_input_embeddings()
    original = embedding.weight.detach().clone()
    application = apply_remap_to_model(tower.model, remap)

    sliced = tower.model.get_input_embeddings().weight.detach()
    new_to_old = torch.as_tensor(
        np.asarray(remap.new_to_old, dtype=np.int64), device=sliced.device
    )
    expected = original.index_select(0, new_to_old.to(original.device)).to(sliced.device)
    if not torch.equal(sliced, expected):
        wrong = torch.nonzero((sliced != expected).any(dim=1)).flatten()
        raise BackboneContractViolation(
            f"the remapped embedding disagrees with the checkpoint on {wrong.numel()} of "
            f"{sliced.shape[0]} row(s); first at new id {int(wrong[0])}, which should be old "
            f"id {int(new_to_old[int(wrong[0])])}. The embedding is tied to the output head, "
            "so a row in the wrong place trains one token and scores another, and no loss "
            "curve shows it."
        )

    old_to_new = np.asarray(remap.old_to_new)
    kept_old = np.flatnonzero(old_to_new >= 0).astype(np.int64)
    kept_new = old_to_new[kept_old].astype(np.int64)
    reverse_ok = torch.equal(
        sliced.index_select(0, torch.as_tensor(kept_new, device=sliced.device)),
        original.index_select(0, torch.as_tensor(kept_old, device=original.device)).to(
            sliced.device
        ),
    )
    if not reverse_ok:
        raise BackboneContractViolation(
            "old_to_new and new_to_old disagree about where a kept row landed: the forward "
            "direction verified and the reverse did not. The two statements of one mapping "
            "have drifted."
        )

    footprint = footprint_at(
        tower, rows=tower.footprint.rows, width=tower.footprint.width,
        vocab_size=application.new_vocab_size,
    )
    return TextTower(
        model=tower.model,
        snapshot=tower.snapshot,
        spec=tower.spec,
        n_tensors_loaded=tower.n_tensors_loaded,
        vocab_size=application.new_vocab_size,
        hidden_size=tower.hidden_size,
        gradient_checkpointing=tower.gradient_checkpointing,
        dtype=tower.dtype,
        device=tower.device,
        # Carried, not re-derived: the remap slices the tied embedding and touches nothing
        # about attention, and asking the config again here would re-read a value this
        # function cannot have changed.
        attn_implementation=tower.attn_implementation,
        optimizer=tower.optimizer,
        footprint=footprint,
        remap=application,
    )


# --- measuring what checkpointing actually did -----------------------------------------------


def saved_activation_bytes(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[int, int]:
    """``(n_tensors, n_bytes)`` the autograd graph retains for one forward pass.

    This is the quantity gradient checkpointing reduces, measured rather than inferred:
    ``saved_tensors_hooks`` sees every tensor the graph packs for backward. It works on CPU,
    which matters because ``torch.mps`` exposes no peak-memory API at all (``remap.py``
    records that as ``GAP-S2-MPS-NO-PEAK-MEMORY-API``) and no CUDA device is available here.

    Tensors are counted by identity, so one tensor saved by two nodes counts once. The
    forward runs under ``torch.enable_grad`` and the graph is discarded afterwards; nothing
    is stepped and no gradient is left behind.
    """
    import torch

    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            seen[id(t)] = t.numel() * t.element_size()
        return t

    def unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        model(input_ids=input_ids, attention_mask=attention_mask)
    return len(seen), sum(seen.values())


# --- the step ---------------------------------------------------------------------------------


def _group_recipe(group: Mapping[str, Any]) -> tuple[Any, ...]:
    """What of a parameter group is recipe rather than state: betas, lr_scale, name."""
    betas = group.get("betas")
    return (
        None if betas is None else tuple(float(b) for b in betas),
        float(group.get("lr_scale", 1.0)),
        group.get("name"),
    )


class QwenDecisionStep:
    """A [`qd_train.trainer.SpanScoringStep`] over the real text tower.

    This is the binding the repository did not have: ``trainer._train`` drives a
    ``TrainStep``, ``heads.SpanPointerHead.forward`` wants ``hidden`` ``[K, L, H]``, and
    until now nothing produced that ``hidden`` from real weights.

    Two contracts exist in this repository and only one of them is here.
    ``byte_train.Rung0Step.accumulate`` takes a ``BatchPlan`` and returns ``Rung0Losses``;
    it belongs to the from-scratch byte decider and ``tools/rung0_toy_run.py`` drives it.
    ``train_ft`` drives *this* one -- ``accumulate(batch, supervision) -> float`` -- which
    DevMap confirms: ``train_ft`` -> ``_train`` -> ``TrainStep``, with no edge to
    ``Rung0Step`` anywhere.

    ``accumulate_span`` is defined, so ``_train`` routes span batches here rather than
    refusing them; a step without it makes the loop refuse a ``SLOT_SPAN`` batch, which is
    the behaviour ``SpanScoringStep`` exists to make explicit.
    """

    def __init__(
        self,
        tower: TextTower,
        *,
        seed: int,
        lr: float,
        total_steps: int,
        span_weight: float = 1.0,
        max_grad_norm: float = 1.0,
        max_width: int = 34_522,
        allow_frozen_moments: bool = False,
        lower_layers_n: int = 0,
        lower_lr_scale: float = 1.0,
        beta2: float = DEFAULT_BETA2,
    ) -> None:
        import torch
        from torch import nn

        if not lr > 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not span_weight > 0.0:
            raise ValueError(
                f"span_weight must be positive, got {span_weight}; zero would train the span "
                "head on nothing while its loss still appeared in the log"
            )
        if not max_grad_norm > 0.0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
        if max_width < 2:
            raise ValueError(f"max_width must be at least 2, got {max_width}")

        # Before anything random is drawn. This class builds a randomly-initialised
        # SpanPointerHead, and until this line nothing in it was seeded at all -- so the
        # `seed` in the ledger's protocol, in `_checkpoint_name` and in every ledger row
        # this step ever wrote determined the BATCH ORDER and not the parameters the run
        # started from. Two runs at one seed opened at different losses, and the channel
        # whose head is random (span) scattered while the channel whose head is pretrained
        # (letter) did not. The stand-in branch, `real_ft_run.RealFtStep.__init__`, has
        # always called this on its first line; the real branch never did, and every GH200
        # measurement to date ran on the real branch.
        #
        # The global stream rather than a `torch.Generator`, matching the stand-in: torch's
        # module initialisers do not take a generator, and a second mechanism beside the one
        # that already works would be two owners of one guarantee. Seeding here also fixes
        # the stream the training forward passes draw from, because construction is the last
        # thing that happens before the loop.
        torch.manual_seed(seed)
        self.seed = int(seed)
        self.tower = tower
        self.device = tower.device
        self.span_weight = float(span_weight)
        self.max_grad_norm = float(max_grad_norm)
        self.max_width = int(max_width)
        # float32, deliberately, even when the tower is bf16: the head is two [H, H]
        # projections and two [H] vectors -- 16.8 MB at H=2048, against a 2.8 GB tower --
        # and its loss is a softmax over a candidate set that can run to hundreds of lines.
        # Computing that logsumexp with 8 bits of mantissa is where the abstain row stops
        # being a genuine competitor. `accumulate_span` casts `hidden` to match.
        self.span_head = SpanPointerHead(tower.hidden_size).to(
            device=tower.model.get_input_embeddings().weight.device,
            dtype=torch.float32,
        )
        self._nn = nn
        self._torch = torch
        # Built from the tower's OWN spec rather than hardcoded. `load_text_tower` validates
        # that spec against the tower's dtype and then this line ignored it, constructing a
        # plain AdamW whatever the caller asked for -- so a run could pass
        # keeps_fp32_master=True, have it accepted, and train with bf16 moments anyway. A
        # spec that is checked and then discarded is worse than no spec: it reads as a
        # guarantee.
        # `total_steps` travels with the spec because the spec alone cannot answer whether
        # this optimizer is fit for this run: a bf16 second moment is correct for 383 steps
        # and broken for 384, and the step that knows the dtype has never been the one that
        # knows the schedule. Passing it here is what closes that gap.
        #
        # `lower_layers_n > 0` is the layer-wise split (RSI-Jev fit.py; see
        # `qd_train.optim.layerwise_param_groups`): the tower's first `lower_layers_n`
        # decoder layers train at `lower_lr_scale` times the schedule. Zero, the default,
        # builds exactly the single-group optimizer this class always built.
        if lower_layers_n < 0:
            raise ValueError(f"lower_layers_n must not be negative, got {lower_layers_n}")
        if not lower_layers_n and lower_lr_scale != 1.0:
            raise ValueError(
                f"lower_lr_scale={lower_lr_scale} was given with lower_layers_n=0, so it "
                "would scale no layer while a recipe recorded it"
            )
        params: list[Any] = (
            layerwise_param_groups(
                tower.model.named_parameters(),
                lower_layers_n=lower_layers_n,
                lower_lr_scale=lower_lr_scale,
                extra=self.span_head.parameters(),
            )
            if lower_layers_n
            else list(self.parameters())
        )
        self.optimizer = build_optimizer(
            params,
            spec=tower.optimizer,
            lr=lr,
            total_steps=total_steps,
            allow_frozen_moments=allow_frozen_moments,
            beta2=beta2,
        )
        #: Component losses per micro-batch. ``TrainResult.loss_log`` carries the combined
        #: number only, and a falling total with a flat span term is a model that learned
        #: the letter and nothing about *where*.
        self.letter_log: list[float] = []
        self.span_log: list[float] = []

    def parameters(self) -> list[torch.nn.Parameter]:
        return [*self.tower.model.parameters(), *self.span_head.parameters()]

    # -- forward ---------------------------------------------------------------------------

    def hidden(self, batch: Batch) -> torch.Tensor:
        """``[B, L, H]`` -- the tower's last hidden state for this batch.

        The attention mask comes from ``Batch.lengths`` rather than from a padding-token
        comparison: a padding id that also occurs in real text would make the second method
        mask real positions, and ``lengths`` is the channel that states the answer.
        """
        torch = self._torch
        width = int(batch.tokens.shape[1])
        if width > self.max_width:
            raise ValueError(
                f"a batch {width} wide reached a step bounded at {self.max_width}. Clipping "
                "it would move every row's target_index, so it is refused instead."
            )
        weight = self.tower.lm_head_weight
        ids = torch.as_tensor(batch.tokens.astype(np.int64), device=weight.device)
        lengths = torch.as_tensor(
            np.asarray(batch.lengths).astype(np.int64), device=weight.device
        )
        mask = (
            torch.arange(width, device=weight.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).to(torch.int64)
        out = self.tower.model(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[..., V]`` logits from ``[..., H]`` hidden states, against the tied embedding.

        `RealFtStep` carries an `nn.Linear` under this name and `tools/real_ft_run.py`
        calls `step.lm_head(hidden)` on whichever step it was given, so the real step
        answers it too. There is no second weight to reach for: the tied embedding **is**
        the output head, which is why this is a method over `tower.lm_head_weight` rather
        than a module of its own.

        This materialises the full ``[..., V]`` slab, which is exactly what
        `fused_linear_cross_entropy` exists to avoid during training. It is here for
        evaluation, where the logits are the thing being inspected; `accumulate` does not
        call it.
        """
        return self._torch.nn.functional.linear(
            hidden.to(self.tower.lm_head_weight.dtype), self.tower.lm_head_weight
        )

    def _letter_loss(
        self, hidden: torch.Tensor, supervision: Supervision
    ) -> torch.Tensor | None:
        """``None`` when the batch is all-span: its letter channel is legitimately empty."""
        if supervision.n_supervised == 0:
            return None
        torch = self._torch
        weight = self.tower.lm_head_weight
        return fused_linear_cross_entropy(
            hidden[:, :-1],
            weight,
            torch.as_tensor(
                supervision.targets.astype(np.int64), device=weight.device
            ),
            mask=torch.as_tensor(supervision.mask.copy(), device=weight.device),
        )

    # -- TrainStep -------------------------------------------------------------------------

    def accumulate(self, batch: Batch, supervision: Supervision) -> float:
        loss = self._letter_loss(self.hidden(batch), supervision)
        if loss is None:  # pragma: no cover - `_refuse_unsupervised_rows` refuses this first
            raise RuntimeError(
                "a span-free batch reached accumulate with no supervised token"
            )
        loss.backward()
        value = float(loss.detach())
        self.letter_log.append(value)
        self.span_log.append(0.0)
        return value

    def accumulate_span(self, batch: Batch, supervision: Supervision) -> float:
        span = supervision.span
        if span is None:  # pragma: no cover - `_train` only routes here when span is set
            raise RuntimeError(
                "accumulate_span was handed a supervision with no span channel"
            )
        torch = self._torch
        hidden = self.hidden(batch)
        plan = plan_span_batch(span, device=hidden.device)
        rows = torch.as_tensor(span.rows.astype(np.int64), device=hidden.device)
        # float32 for the pointer softmax. The head's scores are a bilinear form over
        # hidden states and its loss is a cross-entropy over a candidate set that can run
        # to hundreds of lines; in bf16 the logsumexp of that is computed with 8 bits of
        # mantissa, and the abstain row competes with every line in it. The head's own
        # parameters are float32 for the same reason -- see __init__.
        span_loss = self.span_head.loss(hidden[rows].float(), plan)
        letter = self._letter_loss(hidden, supervision)
        total = (
            self.span_weight * span_loss
            if letter is None
            else letter + self.span_weight * span_loss
        )
        total.backward()
        self.letter_log.append(0.0 if letter is None else float(letter.detach()))
        self.span_log.append(float(span_loss.detach()))
        return float(total.detach())

    def apply(self, *, lr: float) -> None:
        # `apply_lr` rather than `group["lr"] = lr`, which flattened a layer-wise split back
        # to one rate at the first step. It refuses a non-positive lr, as this did.
        apply_lr(self.optimizer, lr)
        # Bounded before the step: an unclipped gradient is how a run ends with NaN
        # parameters and a loss log that stops rather than says why.
        self._nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # -- checkpointing ----------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The step's state as JSON values and [`qd_train.run_control.TensorRef`]s.

        ``TrainStep.state`` is documented "JSON-serialisable", and
        ``byte_train._portable`` honours that by writing every tensor out as a list of
        floats -- 46.5 bytes per parameter, measured at rung 0. At this model's size that is
        roughly 93 GB of decimal text per checkpoint, and non-finite values cannot be
        written at all.

        ``TensorRef`` is the repository's answer to that and it is **not** reimplemented
        here: ``Checkpoint.write`` splits these out into a safetensors sidecar bounded by
        ``MAX_SIDECAR_BYTES``, and ``Checkpoint.read`` rejoins them against a per-tensor
        digest. Writing a private sidecar beside that would be a second convention for one
        thing, and the two would drift on exactly the question of which file is the weights.

        ``_train`` calls this at the end of **every** run and at each checkpoint interval,
        not only when a checkpoint was asked for, so this has to be cheap enough to be
        called unconditionally. It is: the tensors are detached and copied to host memory,
        which is the same work the sidecar would do anyway.
        """
        import torch

        from .run_control import TensorRef

        def refs(module: Any) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for name, tensor in module.state_dict().items():
                host = tensor.detach().to("cpu").contiguous()
                # Reinterpreted as bytes rather than rendered as numbers: `TensorRef`
                # documents that a round trip through `float` is not the identity on every
                # bit pattern, and bfloat16 has no numpy dtype to render through at all.
                # Verified bit-exact on this host for bfloat16, float16, float32 and int64.
                out[name] = TensorRef(
                    dtype=str(host.dtype).removeprefix("torch."),
                    shape=tuple(host.shape),
                    data=host.reshape(-1).view(torch.uint8).numpy().tobytes(),
                )
            return out

        return {
            "tower": refs(self.tower.model),
            "span_head": refs(self.span_head),
            # The optimizer, because a checkpoint without it is not one. This module's own
            # `run_control` says so in its header -- "a resume from weights alone restarts
            # Adam's moments from zero and does not reproduce the trajectory, which is the
            # property S5 rests on" -- and `MAX_SIDECAR_BYTES` was derived from "weights and
            # optimizer state in fp32". Everything was sized for this and nothing put it
            # here, so every resume silently re-entered warm-up on a trained model.
            # Measured on a tiny tower: 12 steps taken as 6+6 across a written checkpoint
            # diverged from the uninterrupted 12 at 5 of 12 losses.
            "optimizer": self._optimizer_refs(),
            "micro_batches": len(self.letter_log),
            "span_weight": self.span_weight,
            "vocab_size": self.tower.vocab_size,
        }

    #: Where ``torch.optim.Optimizer.state_dict()`` keys a dict by parameter INDEX rather
    #: than by name. Everything else in an optimizer state is string-keyed, and the
    #: checkpoint body refuses a non-string key outright, so exactly this sub-tree is
    #: stringified on the way out and restored on the way back.
    _INT_KEYED: Final[str] = "state"

    def _optimizer_refs(self) -> dict[str, Any]:
        """The optimizer's ``state_dict`` as JSON values and [`TensorRef`].

        Shape-agnostic on purpose: ``torch.optim.AdamW`` returns ``{"state": ...,
        "param_groups": ...}`` and [`qd_train.optim.MasterWeightAdamW`] returns ``{"inner":
        ..., "masters": ...}``, and a walker that recurses handles both without either
        being named here. The one structural fact it does encode is [`_INT_KEYED`].
        """
        torch = self._torch

        from .run_control import TensorRef

        def convert(value: Any, *, numeric_keys: bool) -> Any:
            if isinstance(value, torch.Tensor):
                host = value.detach().to("cpu").contiguous()
                return TensorRef(
                    dtype=str(host.dtype).removeprefix("torch."),
                    shape=tuple(host.shape),
                    data=host.reshape(-1).view(torch.uint8).numpy().tobytes(),
                )
            if isinstance(value, dict):
                return {
                    str(key): convert(item, numeric_keys=key == self._INT_KEYED)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [convert(item, numeric_keys=numeric_keys) for item in value]
            return value

        return convert(self.optimizer.state_dict(), numeric_keys=False)

    def _revive_optimizer(self, body: Any) -> Any:
        """The inverse of [`_optimizer_refs`], including the integer keys."""
        import torch

        from .run_control import TensorRef

        def convert(value: Any, *, numeric_keys: bool) -> Any:
            if isinstance(value, TensorRef):
                dtype = getattr(torch, value.dtype)
                return (
                    torch.frombuffer(bytearray(value.data), dtype=dtype)
                    .reshape(value.shape)
                    .clone()
                )
            if isinstance(value, dict):
                return {
                    (int(key) if numeric_keys and key.lstrip("-").isdigit() else key):
                    convert(item, numeric_keys=key == self._INT_KEYED)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [convert(item, numeric_keys=numeric_keys) for item in value]
            return value

        return convert(body, numeric_keys=False)

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Revive what [`state`] produced. The digests were already checked by ``Checkpoint``."""
        import torch

        from .run_control import TensorRef

        missing = {"tower", "span_head", "span_weight", "vocab_size", "optimizer"} - set(
            state
        )
        if missing:
            raise BackboneContractViolation(
                f"this checkpoint state is missing {sorted(missing)}; it was not written by "
                "QwenDecisionStep.state and restoring from it would guess at the rest. "
                "'optimizer' is in that set deliberately: a state carrying only weights "
                "loads without complaint and resumes a run whose moments are zero, which "
                "looks like a working resume and is not one."
            )
        # Unexpected keys are refused too, not ignored. `state` writes exactly the six keys
        # checked here, and anything else was written by a wrapper this step is not inside:
        # a checkpoint taken under `qd_train.replay.PriorKLReplay` carries "replay", and
        # resuming it into a bare step would silently drop the replay term mid-run -- the
        # training source's consumed_digest never sees replay batches, so no other check
        # would notice.
        unexpected = set(state) - {
            "tower", "span_head", "span_weight", "vocab_size", "optimizer", "micro_batches"
        }
        if unexpected:
            raise BackboneContractViolation(
                f"this checkpoint state carries {sorted(unexpected)}, which "
                "QwenDecisionStep.state never writes. It was taken under a wrapper (e.g. "
                "replay) this run does not have; resuming it here would continue a different "
                "objective without saying so."
            )
        # The optimizer's own recipe, checked before anything is loaded. torch's
        # `load_state_dict` overwrites each group's hyperparameters with the saved ones, so a
        # checkpoint taken at beta2=0.95 or with a layer-wise split, resumed by a run built
        # without them, would silently train on the checkpoint's recipe while the ledger row
        # recorded this run's.
        revived_optimizer = self._revive_optimizer(state["optimizer"])
        saved_groups = (revived_optimizer.get("inner") or revived_optimizer).get("param_groups")
        mine = [_group_recipe(g) for g in self.optimizer.param_groups]
        theirs = [_group_recipe(g) for g in saved_groups or []]
        if mine != theirs:
            raise BackboneContractViolation(
                f"the checkpoint's optimizer groups are {theirs} and this step's are {mine} "
                "(betas, lr_scale, name). Resuming would train on the checkpoint's recipe "
                "under this run's ledger row; pass the flags the checkpoint was taken with."
            )
        if int(state["vocab_size"]) != self.tower.vocab_size:
            raise BackboneContractViolation(
                f"the checkpoint was written at vocab_size={state['vocab_size']} and this "
                f"tower is {self.tower.vocab_size}. A different remap renumbers every row."
            )

        def revive(entries: Mapping[str, Any], *, where: str) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for name, ref in entries.items():
                if not isinstance(ref, TensorRef):
                    raise BackboneContractViolation(
                        f"{where}[{name!r}] is a {type(ref).__name__}, not a TensorRef. A "
                        "checkpoint body that lost its tensors on the way through is not a "
                        "checkpoint with fewer tensors."
                    )
                dtype = getattr(torch, ref.dtype)
                out[name] = (
                    torch.frombuffer(bytearray(ref.data), dtype=dtype)
                    .reshape(ref.shape)
                    .clone()
                )
            return out

        self.tower.model.load_state_dict(
            revive(state["tower"], where="tower"), strict=True
        )
        self.span_head.load_state_dict(
            revive(state["span_head"], where="span_head"), strict=True
        )
        # After the parameters, never before: `torch.optim.Optimizer.load_state_dict` casts
        # each restored state tensor to the dtype and device of the parameter it belongs to,
        # so loading it against parameters that are about to be replaced would cast against
        # the wrong ones. The two halves are also restored together or not at all --
        # `MasterWeightAdamW.load_state_dict` refuses a partial state for the same reason.
        self.optimizer.load_state_dict(revived_optimizer)
        self.span_weight = float(state["span_weight"])
