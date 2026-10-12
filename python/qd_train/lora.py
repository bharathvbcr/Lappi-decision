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

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import nn
from torch.nn import functional as F

from qd_train.run_control import LRSchedule

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
    "attach_lora",
    "lora_param_groups",
    "lora_parameters",
    "lora_schedule",
    "merged_weights",
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
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=device, dtype=torch.float32)
        self.lora_B = nn.Linear(
            r, base.out_features, bias=False, device=device, dtype=torch.float32
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
