"""The plain-torch LoRA reference arm for the LoRA-2B recipe family (reference and oracle only).

Fable's D7 (``AUDIT/lora-2b-2026-10-09/fable-ft-decisions.md``): the parity oracle for ojas's LoRA
trainer is a plain-torch LoRA arm in this package -- no ``peft``, ``bitsandbytes`` or ``unsloth``
(none is installed and none is added). It is policy-allowed under the language policy as a
reference oracle; it is not a shipped trainer, and it is not what a promotion-grade run uses.

What it is, and nothing more:

* **A/B leaves per target linear.** ``y = x W^T + (alpha/r) (x A^T) B^T`` with ``A`` of shape
  ``[r, in]`` and ``B`` of shape ``[out, r]``, held as ``nn.Linear`` sub-modules named ``lora_A``
  and ``lora_B`` so the state names line up with PEFT's (``<module>.lora_A.weight``).
* **PEFT's init and scaling.** ``A`` is ``kaiming_uniform_(a=sqrt(5))`` (so ``U(+-1/sqrt(in))``),
  ``B`` is zero, ``scaling = alpha / r``. Verified from the PEFT source by Fable's D5
  (``tuners/lora/layer.py``); the adapter-key prefix and ``adapter_config.json`` fields are the
  still-open ``GAP-FT-RULING-PEFT-KEYS-UNREAD-2026-10-09`` and are not produced here.
* **Frozen base.** Every parameter the adapter did not add has ``requires_grad=False`` -- the
  embedding, the norms and the 36 stored-F32 tensors included -- so autograd builds no edge to a
  base weight and no ``dW`` is ever computed (``tests/test_lora_reference.py`` walks the graph).
* **Targets (D6).** Attention ``q/k/v/o``, GDN ``linear_attn.in_proj_qkv / in_proj_z / out_proj``,
  MLP ``gate/up/down``: 150 linears at 2B (24 MLPs, 6 attention layers x 4, 18 GDN layers x 3).
  ``in_proj_a`` and ``in_proj_b`` (per-head gates, ``[heads, hidden]``) are not in the table and are
  not adapted.
* **D17.** The 17-row readout stays Lappi's; with a frozen embedding it trains as a delta
  ``D17 [17, H]`` (f32, init 0) added to the frozen embedding's answer rows, and the objective is a
  17-way cross entropy over exactly those rows in f32 (:class:`AnswerRowDelta`).
* **Unchanged Lappi machinery.** The shard reader, supervision, span head, ``LRSchedule``,
  ``apply_lr`` and ``build_optimizer`` are used as they are: :func:`lora_schedule` and
  :func:`lora_param_groups` only *configure* them with the pre-registered numbers
  (``campaign/lora-2b-preregistered.json``, which ``tests/test_lora_reference.py`` reads and holds
  these constants to).

**On "MasterWeightAdamW unchanged".** ``MasterWeightAdamW`` refuses an all-fp32 parameter list by
design (an fp32 parameter is its own master; its docstring says to use ``torch.optim.AdamW``). A
LoRA arm whose trainable leaves are the f32 adapters, ``D17`` and the f32 span head is exactly that
case, so the optimizer :func:`~qd_train.optim.build_optimizer` returns for ``ADAMW_FP32`` is
``torch.optim.AdamW`` over the same groups, with the same ``apply_lr`` driver. It is the same
arithmetic ``MasterWeightAdamW`` delegates to; nothing in ``optim.py`` is changed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qd_train.artifacts import Batch
from qd_train.fused_ce import fused_linear_cross_entropy
from qd_train.heads import plan_span_batch
from qd_train.memory import ADAMW_FP32, tensor_ranges
from qd_train.optim import apply_lr, build_optimizer
from qd_train.run_control import LRSchedule, TensorRef, _sidecar_digest
from qd_train.trainer import Supervision

__all__ = [
    "D17_LR",
    "D17_ROWS",
    "HEAD_LR_SCALE",
    "LORA_ALPHA",
    "LORA_LR",
    "LORA_RANK",
    "LORA_TARGET_SUFFIXES",
    "LORA_UNADAPTED_LINEAR_SUFFIXES",
    "LORA_WEIGHT_DECAY",
    "AnswerRowDelta",
    "LoRALinear",
    "LoraAdapter",
    "LoraContractViolation",
    "LoraReferenceStep",
    "StoredDtypes",
    "adapted_in_features",
    "attach_lora",
    "d17_init_digest",
    "lora_a_digest",
    "lora_a_rule_digest",
    "lora_model_state",
    "lora_param_groups",
    "lora_parameters",
    "lora_schedule",
    "merged_weights",
    "stored_dtypes",
]

#: D6: r=16, alpha=16, dropout 0 (there is no dropout path here, so 0 cannot be mis-set).
LORA_RANK: Final[int] = 16
LORA_ALPHA: Final[int] = 16
#: D6: LoRA at 2e-4; ``D17`` and the span head at 1e-4; weight decay 0.01 on every trainable tensor.
LORA_LR: Final[float] = 2e-4
D17_LR: Final[float] = 1e-4
LORA_WEIGHT_DECAY: Final[float] = 0.01
#: The one schedule peaks at :data:`LORA_LR`; the heads follow it at this ``lr_scale``, so they
#: share warmup and cosine and sit at :data:`D17_LR` at the peak.
HEAD_LR_SCALE: Final[float] = D17_LR / LORA_LR
#: Up to 16 letters plus noul (Lappi's served readout).
D17_ROWS: Final[int] = 17

#: Module-name suffixes the adapter wraps (D6's table). Matched on the dotted tail of a module name.
LORA_TARGET_SUFFIXES: Final[tuple[str, ...]] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
#: GDN linears the table does not name. Recorded so "all linears" is not read as including them.
LORA_UNADAPTED_LINEAR_SUFFIXES: Final[tuple[str, ...]] = (
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
)


class LoraContractViolation(ValueError):
    """The adapter was asked for something that would train a different model than named."""


class LoRALinear(nn.Module):
    """``base`` (frozen) plus the low-rank update ``(alpha/r) * B(A(x))``.

    The adapter matrices are f32 whatever the base's dtype is (D3: LoRA/head/norm parameters and
    their optimizer state are f32). The input is cast to the adapter's dtype for the low-rank
    branch and the update is cast back to the base output's dtype before the add, which is PEFT's
    own order of operations.
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise LoraContractViolation(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        if r < 1 or alpha < 1:
            raise LoraContractViolation(f"r and alpha must be positive, got r={r}, alpha={alpha}")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        device = base.weight.device
        # skip_init: nn.Linear's own kaiming init would draw from torch's global RNG (and then be
        # overwritten); A is drawn by attach_lora from its own generator, B is zeroed below.
        self.lora_A = nn.utils.skip_init(
            nn.Linear, base.in_features, r, bias=False, device=device, dtype=torch.float32
        )
        self.lora_B = nn.utils.skip_init(
            nn.Linear, r, base.out_features, bias=False, device=device, dtype=torch.float32
        )
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        update = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scaling
        return result + update.to(result.dtype)


@dataclass(frozen=True)
class LoraAdapter:
    """What :func:`attach_lora` did: the wrapped modules, in the order their ``A`` was drawn."""

    module_names: tuple[str, ...]
    r: int
    alpha: int
    seed: int

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


def _is_target(name: str, suffixes: Iterable[str]) -> bool:
    return any(name == s or name.endswith("." + s) for s in suffixes)


def attach_lora(
    model: nn.Module,
    *,
    seed: int,
    r: int = LORA_RANK,
    alpha: int = LORA_ALPHA,
    target_suffixes: Iterable[str] = LORA_TARGET_SUFFIXES,
) -> LoraAdapter:
    """Freeze ``model`` and wrap every target linear in a :class:`LoRALinear`.

    ``A`` is drawn from one CPU generator seeded with ``seed``, visiting the target modules in
    ``model.named_modules()`` order and calling ``kaiming_uniform_(a=sqrt(5))`` on each ``[r, in]``
    weight. That construction is the pre-registration's *rule* for the LoRA-A digest: a row
    records the digest, and a reader recomputes it from this function and the seed.

    Raises:
        LoraContractViolation: if no module matches, a match is not a plain ``nn.Linear``, or the
            model already carries an adapter. An adapter over nothing would train nothing and
            report a run.
    """
    suffixes = tuple(target_suffixes)
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise LoraContractViolation("this model already carries a LoRA adapter")
    targets = [
        (name, module) for name, module in model.named_modules() if _is_target(name, suffixes)
    ]
    if not targets:
        raise LoraContractViolation(
            f"no module name ends with any of {suffixes}; an adapter over no linear trains nothing"
        )
    for name, module in targets:
        if not isinstance(module, nn.Linear):
            raise LoraContractViolation(
                f"target {name!r} is a {type(module).__name__}, not nn.Linear"
            )
    for param in model.parameters():
        param.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    names: list[str] = []
    for name, module in targets:
        wrapped = LoRALinear(module, r=r, alpha=alpha)
        a_cpu = torch.empty(r, module.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a_cpu, a=math.sqrt(5), generator=generator)
        with torch.no_grad():
            wrapped.lora_A.weight.copy_(a_cpu)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, wrapped)
        names.append(name)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)
    return LoraAdapter(module_names=tuple(names), r=r, alpha=alpha, seed=seed)


def lora_parameters(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """``(name, parameter)`` for every adapter leaf, in module order."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield f"{name}.lora_A.weight", module.lora_A.weight
            yield f"{name}.lora_B.weight", module.lora_B.weight


def merged_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """``W' = W + (alpha/r) B A`` per adapted linear, computed in f32 (D5), keyed by module name.

    The caller rounds once to the release dtype; nothing is rounded here, so a seed average can
    average the f32 merged weights before its single rounding.
    """
    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            w = module.base.weight.detach().to(torch.float32)
            delta = module.lora_B.weight.detach() @ module.lora_A.weight.detach()
            out[f"{name}.weight"] = w + module.scaling * delta
    return out


class AnswerRowDelta(nn.Module):
    """D17: the trainable ``[K, H]`` delta over the frozen embedding's answer rows (D2).

    The served readout is ``softmax(h . (embed[answer_ids] + D17)^T)`` over exactly the answer rows;
    with a frozen, tied embedding there is no gradient into the table, so the trainable object is
    the delta, initialised to zero so an untrained delta reproduces the base's own letter rows.
    """

    def __init__(self, hidden_size: int, rows: int = D17_ROWS) -> None:
        super().__init__()
        if hidden_size < 1 or rows < 1:
            raise LoraContractViolation(f"bad D17 shape [{rows}, {hidden_size}]")
        self.delta = nn.Parameter(torch.zeros(rows, hidden_size, dtype=torch.float32))

    def rows(self, embedding: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
        """``embed[row_ids] + D17`` in f32, shape ``[K, H]``."""
        if row_ids.shape != (self.delta.shape[0],):
            raise LoraContractViolation(
                f"row_ids has shape {tuple(row_ids.shape)}, D17 has {self.delta.shape[0]} rows"
            )
        return embedding.detach()[row_ids.to(embedding.device)].to(torch.float32) + self.delta

    def logits(
        self, hidden: torch.Tensor, embedding: torch.Tensor, row_ids: torch.Tensor
    ) -> torch.Tensor:
        """``[N, K]`` f32 logits over the answer rows from ``hidden`` ``[N, H]``."""
        return hidden.to(torch.float32) @ self.rows(embedding, row_ids).t()

    def loss(
        self,
        hidden: torch.Tensor,
        embedding: torch.Tensor,
        row_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Mean 17-way cross entropy at the supervised positions.

        ``target_ids`` are vocabulary ids (what ``Supervision.targets`` holds at its mask); each
        must be one of ``row_ids``. A target outside the answer rows is refused, not skipped: a
        silent skip would shrink the denominator and report a lower loss.
        """
        ids = row_ids.to(target_ids.device)
        match = target_ids.unsqueeze(1) == ids.unsqueeze(0)  # [N, K]
        if not bool(match.any(dim=1).all()):
            raise LoraContractViolation("a supervised target is not one of the answer rows")
        if not bool((match.sum(dim=1) == 1).all()):
            raise LoraContractViolation("answer row ids are not distinct")
        index = match.to(torch.int64).argmax(dim=1)
        return F.cross_entropy(self.logits(hidden, embedding, row_ids), index.to(hidden.device))


def lora_schedule(total_steps: int) -> LRSchedule:
    """D6's schedule: Lappi's ``LRSchedule``, warmup ``max(1, steps // 20)``, cosine to 0."""
    return LRSchedule(
        peak_lr=LORA_LR,
        total_steps=total_steps,
        warmup_steps=max(1, total_steps // 20),
        min_lr=0.0,
    )


def lora_param_groups(
    model: nn.Module,
    *,
    delta: AnswerRowDelta | None = None,
    span_head: nn.Module | None = None,
) -> list[dict[str, Any]]:
    """Optimizer groups for ``build_optimizer``/``apply_lr``: adapters at 1.0x, heads at 0.5x.

    Every group carries ``weight_decay`` 0.01 (D6: on every trainable tensor). There is no 0.1x
    lower-layer split: D6 dropped it for LoRA.
    """
    groups: list[dict[str, Any]] = [
        {
            "name": "lora",
            "params": [p for _, p in lora_parameters(model)],
            "lr_scale": 1.0,
            "weight_decay": LORA_WEIGHT_DECAY,
        }
    ]
    if not groups[0]["params"]:
        raise LoraContractViolation("the model carries no LoRA adapter to optimise")
    if delta is not None:
        groups.append(
            {
                "name": "d17",
                "params": [delta.delta],
                "lr_scale": HEAD_LR_SCALE,
                "weight_decay": LORA_WEIGHT_DECAY,
            }
        )
    if span_head is not None:
        groups.append(
            {
                "name": "span_head",
                "params": list(span_head.parameters()),
                "lr_scale": HEAD_LR_SCALE,
                "weight_decay": LORA_WEIGHT_DECAY,
            }
        )
    return groups


# --- the driver's pieces: init digests, the lora-merged checkpoint body, the training step -------
#
# Added for ``tools/lora_reference_run.py`` (card ft-3b8126cd564d889af2b0169943020aa7). Everything
# above is unchanged. Each digest below is one of the seven that the pre-registration's
# ``init.record_in_every_row`` names; the formulas are in each docstring so the Rust writer can
# restate them byte for byte.


def _tensor_ref(tensor: torch.Tensor) -> TensorRef:
    """``tensor`` as a ``TensorRef`` (dtype, shape, raw little-endian bytes), as
    ``QwenDecisionStep.state`` builds one."""
    host = tensor.detach().to("cpu").contiguous()
    return TensorRef(
        dtype=str(host.dtype).removeprefix("torch."),
        shape=tuple(host.shape),
        data=host.reshape(-1).view(torch.uint8).numpy().tobytes(),
    )


def d17_init_digest(hidden_size: int, rows: int = D17_ROWS) -> str:
    """D17's init digest: ``TensorRef.digest()`` of a float32 zero tensor of shape
    ``[rows, hidden_size]`` (the pre-registration's ``init.d17.formula``)."""
    return _tensor_ref(torch.zeros(rows, hidden_size, dtype=torch.float32)).digest()


def lora_a_digest(model: nn.Module) -> str:
    """The LoRA-A digest of an adapter freshly attached by :func:`attach_lora`.

    ``run_control._sidecar_digest`` over ``{"<module>.lora_A.weight": TensorRef(A)}`` for every
    adapted module, ``A`` as the float32 ``[r, in]`` tensor. ``_sidecar_digest`` is: sha256 of its
    domain bytes, the tensor count (u64 big endian), then for each name in sorted order its byte
    length (u64 big endian), its UTF-8 bytes and the 32 raw bytes of that tensor's
    ``TensorRef.digest()``.
    """
    refs = {
        name: _tensor_ref(param)
        for name, param in lora_parameters(model)
        if name.endswith(".lora_A.weight")
    }
    if not refs:
        raise LoraContractViolation("the model carries no LoRA adapter to digest")
    return _sidecar_digest(refs)


def lora_a_rule_digest(
    in_features: Iterable[tuple[str, int]], *, seed: int, r: int = LORA_RANK
) -> str:
    """The LoRA-A digest the pre-registration's *rule* predicts, without a model.

    ``in_features`` is ``(module name, in_features)`` for each adapted linear in
    ``model.named_modules()`` order. ``A`` is ``kaiming_uniform_(a=sqrt(5))`` on each ``[r, in]``
    float32 tensor drawn from one CPU generator seeded with ``seed``; the digest is
    :func:`lora_a_digest`'s. A row's recorded digest must equal this one, or the row is void.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    refs: dict[str, TensorRef] = {}
    for name, n_in in in_features:
        a = torch.empty(r, n_in, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        refs[f"{name}.lora_A.weight"] = _tensor_ref(a)
    if not refs:
        raise LoraContractViolation("no module to draw an A for")
    return _sidecar_digest(refs)


def adapted_in_features(model: nn.Module) -> list[tuple[str, int]]:
    """``(module name, in_features)`` of every wrapped linear, in ``named_modules()`` order."""
    return [
        (name, module.base.in_features)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]


@dataclass(frozen=True)
class StoredDtypes:
    """What a snapshot's safetensors headers say about the text tower's stored dtypes."""

    #: ``sha256("\n".join(f"{name}\t{dtype}" for name in sorted(text names)))``, names without the
    #: ``model.language_model.`` prefix, dtype as the safetensors header spells it (F32, BF16).
    dtype_map_sha256: str
    #: Names stored F32, sorted: the set ``load_text_tower(keep_stored_dtype=...)`` keeps.
    f32_names: tuple[str, ...]
    #: ``sha256("\n".join(f32_names))``.
    f32_names_sha256: str
    #: sha256 of the snapshot's one ``*.safetensors`` file; for several files, sha256 of
    #: ``"<file name>\t<sha256>\n"`` lines in file-name order.
    safetensors_sha256: str


def stored_dtypes(snapshot: Path) -> StoredDtypes:
    """Read the stored-dtype facts off ``snapshot``'s headers (tensor data is only hashed)."""
    from qd_train.backbone import TEXT_PREFIX, text_tensor_index

    snapshot = Path(snapshot)
    index = text_tensor_index(snapshot)
    dtypes: dict[str, str] = {}
    for shard in sorted({shard for shard, _ in index.values()}):
        ranges = tensor_ranges(shard)
        for name, (path, _shape) in index.items():
            if path == shard:
                dtypes[name] = ranges[TEXT_PREFIX + name].dtype
    if set(dtypes) != set(index):
        raise LoraContractViolation("a text tensor's header entry could not be read")
    f32 = tuple(sorted(n for n, d in dtypes.items() if d == "F32"))
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise LoraContractViolation(f"{snapshot} holds no *.safetensors file")
    per_file: list[tuple[str, str]] = []
    for path in files:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                h.update(chunk)
        per_file.append((path.name, h.hexdigest()))
    whole = (
        per_file[0][1]
        if len(per_file) == 1
        else hashlib.sha256("".join(f"{n}\t{d}\n" for n, d in per_file).encode()).hexdigest()
    )
    return StoredDtypes(
        dtype_map_sha256=hashlib.sha256(
            "\n".join(f"{n}\t{dtypes[n]}" for n in sorted(dtypes)).encode()
        ).hexdigest(),
        f32_names=f32,
        f32_names_sha256=hashlib.sha256("\n".join(f32).encode()).hexdigest(),
        safetensors_sha256=whole,
    )


def lora_model_state(
    model: nn.Module,
    *,
    delta: AnswerRowDelta,
    span_head: nn.Module,
    span_weight: float,
    vocab_size: int,
    prompt_format: int = 1,
) -> dict[str, Any]:
    """The ``lora-merged`` checkpoint body ``tools/ckpt_average.py`` reads.

    ``tower`` is the frozen base under the dense tower's own names (``<module>.weight``, not
    ``<module>.base.weight``), ``lora_A``/``lora_B`` map module name to the f32 ``[r, in]`` /
    ``[out, r]`` tensor, ``lora_r``/``lora_alpha`` are ints, ``d17`` holds the ``delta`` tensor,
    and ``span_head``, ``span_weight``, ``vocab_size`` (and ``prompt_format`` when not 1) are the
    full fine-tune checkpoint's.
    """
    wrapped = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not wrapped:
        raise LoraContractViolation("the model carries no LoRA adapter to checkpoint")
    scalars = {(m.r, m.alpha) for m in wrapped.values()}
    if len(scalars) != 1:
        raise LoraContractViolation(f"adapters disagree about (r, alpha): {sorted(scalars)}")
    ((r, alpha),) = scalars
    tower: dict[str, TensorRef] = {}
    for key, tensor in model.state_dict().items():
        module, _, leaf = key.rpartition(".")
        if module.endswith((".lora_A", ".lora_B")):
            continue
        owner, _, inner = module.rpartition(".")
        if inner == "base" and owner in wrapped:
            key = f"{owner}.{leaf}"
        tower[key] = _tensor_ref(tensor)
    state: dict[str, Any] = {
        "tower": tower,
        "span_head": {k: _tensor_ref(v) for k, v in span_head.state_dict().items()},
        "lora_A": {n: _tensor_ref(m.lora_A.weight) for n, m in wrapped.items()},
        "lora_B": {n: _tensor_ref(m.lora_B.weight) for n, m in wrapped.items()},
        "lora_r": r,
        "lora_alpha": alpha,
        "d17": {"delta": _tensor_ref(delta.delta)},
        "span_weight": float(span_weight),
        "vocab_size": int(vocab_size),
    }
    if prompt_format != 1:
        state["prompt_format"] = prompt_format
    return state


class LoraReferenceStep:
    """A ``SpanScoringStep`` over a LoRA-adapted tower: the reference arm's training step.

    One optimizer step per :meth:`apply`; ``grad_accum`` is 1 by the recipe, so a step is one
    batch. The objective is the pre-registered one: the 17-way cross entropy over the answer rows
    (:class:`AnswerRowDelta`) at the supervised choice positions, plus ``span_weight`` (1.0) times
    the span pointer head's loss; the full-vocabulary letter cross entropy at the same positions
    is computed without gradient and only logged. The gradient is clipped to ``max_grad_norm``
    over every trainable tensor (adapters, D17, span head), as ``QwenDecisionStep`` clips; the
    logged ``grad_norm`` is the norm before clipping.

    ``log_path``, when given, receives one JSON line per optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        answer_ids: Iterable[int],
        span_head: nn.Module,
        total_steps: int,
        max_width: int,
        vocab_size: int,
        prompt_format: int = 1,
        log_path: Path | None = None,
        span_weight: float = 1.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        if not span_weight > 0.0 or not max_grad_norm > 0.0 or max_width < 2:
            raise LoraContractViolation("span_weight, max_grad_norm and max_width must be positive")
        embedding = model.get_input_embeddings().weight
        self.model = model
        self.device = embedding.device
        self.embedding = embedding
        self.span_head = span_head
        self.span_weight = float(span_weight)
        self.max_grad_norm = float(max_grad_norm)
        self.max_width = int(max_width)
        self.vocab_size = int(vocab_size)
        self.prompt_format = int(prompt_format)
        self.row_ids = torch.as_tensor(list(answer_ids), dtype=torch.int64, device=self.device)
        self.delta = AnswerRowDelta(embedding.shape[1], rows=int(self.row_ids.numel())).to(
            self.device
        )
        groups = lora_param_groups(model, delta=self.delta, span_head=span_head)
        self.trainable = [p for g in groups for p in g["params"]]
        self.optimizer = build_optimizer(
            groups, spec=ADAMW_FP32, lr=LORA_LR, total_steps=total_steps
        )
        self.steps_applied = 0
        self.micro_batches = 0
        self._pending: dict[str, Any] = {}
        self._log = None if log_path is None else log_path.open("w", encoding="utf-8")

    def parameters(self) -> list[nn.Parameter]:
        return list(self.trainable)

    def hidden(self, batch: Batch) -> torch.Tensor:
        width = int(batch.tokens.shape[1])
        if width > self.max_width:
            raise LoraContractViolation(
                f"a batch {width} wide reached a step bounded at {self.max_width}; clipping it "
                "would move every row's target_index"
            )
        ids = torch.as_tensor(batch.tokens.astype(np.int64), device=self.device)
        lengths = torch.as_tensor(np.asarray(batch.lengths).astype(np.int64), device=self.device)
        mask = (torch.arange(width, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)).to(
            torch.int64
        )
        return self.model(input_ids=ids, attention_mask=mask).last_hidden_state

    def _choice(self, hidden: torch.Tensor, supervision: Supervision) -> torch.Tensor | None:
        if supervision.n_supervised == 0:
            return None
        mask = torch.as_tensor(supervision.mask.copy(), device=self.device)
        targets = torch.as_tensor(supervision.targets.astype(np.int64), device=self.device)
        choice = self.delta.loss(
            hidden[:, :-1][mask], self.embedding, self.row_ids, targets[mask]
        )
        with torch.no_grad():
            self._pending["loss_vocab_ce"] = float(
                fused_linear_cross_entropy(
                    hidden[:, :-1].detach(), self.embedding.detach(), targets, mask=mask
                )
            )
        self._pending["loss_choice_17way"] = float(choice.detach())
        return choice

    def accumulate(self, batch: Batch, supervision: Supervision) -> float:
        total = self._choice(self.hidden(batch), supervision)
        if total is None:
            raise LoraContractViolation("a span-free batch reached accumulate with no choice row")
        return self._backward(batch, total)

    def accumulate_span(self, batch: Batch, supervision: Supervision) -> float:
        span = supervision.span
        if span is None:
            raise LoraContractViolation("accumulate_span was handed a batch with no span rows")
        hidden = self.hidden(batch)
        plan = plan_span_batch(span, device=hidden.device)
        rows = torch.as_tensor(span.rows.astype(np.int64), device=hidden.device)
        span_loss = self.span_head.loss(hidden[rows].float(), plan)
        self._pending["loss_span"] = float(span_loss.detach())
        choice = self._choice(hidden, supervision)
        weighted = self.span_weight * span_loss
        return self._backward(batch, weighted if choice is None else choice + weighted)

    def _backward(self, batch: Batch, total: torch.Tensor) -> float:
        if self._pending.get("micro_batches", 0):
            raise LoraContractViolation("grad_accum is 1: a second micro-batch before apply")
        total.backward()
        value = float(total.detach())
        self._pending.update(loss_total=value, batch_index=int(batch.index), micro_batches=1)
        self.micro_batches += 1
        return value

    def apply(self, *, lr: float) -> None:
        apply_lr(self.optimizer, lr)
        norm = float(nn.utils.clip_grad_norm_(self.trainable, self.max_grad_norm))
        if not math.isfinite(norm):
            raise LoraContractViolation(f"non-finite gradient norm {norm} before the step")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        line = {
            "step": self.steps_applied,
            "lr": lr,
            "loss_total": self._pending["loss_total"],
            "loss_choice_17way": self._pending.get("loss_choice_17way"),
            "loss_span": self._pending.get("loss_span"),
            "loss_vocab_ce": self._pending.get("loss_vocab_ce"),
            "grad_norm": norm,
            "batch_index": self._pending["batch_index"],
        }
        self._pending = {}
        self.steps_applied += 1
        if self._log is not None:
            self._log.write(json.dumps(line, sort_keys=True) + "\n")
            self._log.flush()

    def close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def state(self) -> dict[str, Any]:
        """The ``lora-merged`` checkpoint body (:func:`lora_model_state`): the loop writes it into
        every ``Checkpoint`` it builds, so the final one is what ``ckpt_average`` reads."""
        return lora_model_state(
            self.model,
            delta=self.delta,
            span_head=self.span_head,
            span_weight=self.span_weight,
            vocab_size=self.vocab_size,
            prompt_format=self.prompt_format,
        )

    def load_state(self, state: Any) -> None:
        raise LoraContractViolation(
            "the LoRA reference arm does not resume: a load_state that restored nothing would "
            "report a checkpoint's loss log against freshly initialised adapters"
        )
