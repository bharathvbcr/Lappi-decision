"""Average the weights of N checkpoints written by ``real_ft_run.py --checkpoint-dir``.

Ported from RSI-Jev ``rsijev/train.py:189-200`` (``average_checkpoints``), MIT licence,
Copyright (c) 2026 Shanghua Gao, at commit 8f34a4f: sum the float copies, divide by N.
Their note, which is the reason to have it: averaging removes checkpoint-selection noise at
no training cost.

What this adds to theirs, all of it refusal:

* **Same schedule, same step, same model.** ``real_ft_run.py`` rewrites one file per
  ``(tag, seed, device)`` cell, so N checkpoints here are N *runs* (normally N seeds of one
  configuration), not N points along one run. Averaging only means something when they are
  the same configuration at the same point: the ``LRSchedule`` must be equal, the
  ``optimizer_step`` equal, ``vocab_size`` and ``span_weight`` equal, and the ``tower`` and
  ``span_head`` trees must match name for name, dtype for dtype and shape for shape, and the
  prompt format the weights were trained under must be one (``model_state['prompt_format']``,
  absent = 1; the manifest states it at top level when it is not 1, for qd-export). RSI's
  version indexes ``st[k]`` and would raise a bare ``KeyError`` on the first two and
  silently broadcast on none of them; this says which input differs and how.
* **No input twice.** The same file, or two files whose tensor sets have one digest, would
  weight one run double while the manifest listed N.
* **Weights only, and not a checkpoint.** The optimizer's moments are not averaged and the
  output is not written as a ``Checkpoint``: Adam moments averaged across runs describe no
  trajectory, and this repository's position (``run_control``: "a resume from weights alone
  restarts Adam's moments from zero and does not reproduce the trajectory") is that
  weights without moments are not a resume point. The output is a safetensors file of the
  averaged ``tower.*`` and ``span_head.*`` tensors plus a JSON manifest naming every input,
  its ``payload_digest``, and ``"resumable": false``.

**Two sources** (``--from``):

* ``tower``, the default and what this tool always did: each input's ``tower`` and
  ``span_head`` tensors, read whole by ``Checkpoint.read``, averaged in float32 and cast back
  to each tensor's own dtype, so a bf16 tower comes back bf16 -- the dtype
  ``load_text_tower`` was budgeted for.
* ``masters``, for checkpoints trained with ``--optimizer master``: each parameter's fp32
  master copy (``model_state['optimizer']['masters']``, kept by
  ``qd_train.optim.MasterWeightAdamW``) where it has one, and its tower tensor where it does
  not (a buffer, a frozen parameter). Accumulated in float64 and cast to the parameter's
  tower dtype ONCE, at the end -- not three bf16 roundings averaged. The manifest records,
  per tensor, which source it came from.

**How a master is matched to its parameter's name.** The masters are a bare list in the
order the optimizer was handed its parameters, and the checkpoint body is written with
sorted keys, so that order is not recoverable from the file. What is recoverable is the
invariant ``MasterWeightAdamW`` maintains: ``step`` ends with ``live.copy_(master)`` for
every low-precision parameter, and an fp32 parameter is its own master. So at every
checkpoint boundary master ``i``, cast to its parameter's dtype, is byte for byte the tower
or span-head tensor it belongs to -- and the body records every one of those tensors'
digests. :func:`master_names` is that identity, looked up by (dtype, shape, digest). It must
place each master on exactly one tensor, no two masters on one tensor, and the same names in
every input, or the average is refused. Pretrained Qwen3.5-2B-Base has 320 text tensors and
no two same-shape ones are byte-identical (measured 2026-10-01 on snapshot
``b1485b2f``), so the lookup has no tie to break from step 0.

**Memory under ``--from masters``.** ``Checkpoint.read_weights`` with the path
``("optimizer", "masters")`` reads one input's tower, span head and masters -- about
10.5 GiB for the real 2B, never its 14.08 GiB of Adam moments -- into float64 accumulators
the size of the model (about 14 GiB), and drops that input before reading the next. The peak
does not grow with N; :func:`average_masters` states it.

**``--norm-preserving --base-snapshot DIR``** (with ``--from masters``; Fable I-2): the
uniform mean of N deltas ``theta_i - base`` whose pairwise cosine is c keeps only
``sqrt((1 + (N-1)c)/N)`` of one delta's norm, so it shrinks every seed-specific direction.
This writes ``base + lambda * mean_i(theta_i - base)`` with ``lambda = sqrt(N / (1 +
(N-1)c))``, and c is MEASURED here -- the mean over input pairs of the cosine between their
fp32-master deltas against the base, over every tower tensor with a master, in float64,
streamed one tensor at a time (:func:`norm_preserving_average`) -- never chosen. The span
head has no base and is averaged plainly. The manifest's ``norm_preserving`` block records
c, every pair's cosine, N, lambda, the formula and the base's file digests;
``--lambda-not-derived-from-c`` replaces lambda and the block says so.

**``--same-seed-trajectory``** (opt-in, ``--from tower``; v5's last-3 average): the one
exception to "same step". The inputs are ONE run's ``--retain-tower-every`` snapshots at
distinct steps -- RSI's "last k" in its own sense, points along one run -- and
:func:`check_trajectory` refuses mixed seeds, steps that are not strictly ascending, a
schedule, scalar or tree that differs, and an input whose loss log is not the start of the
next one's. The manifest records the steps in a ``trajectory`` block and the latest as its
``optimizer_step``. ``real_ft_run.py --score-checkpoint`` does not score one:
:func:`read_manifest` refuses it by name (:data:`TRAJECTORY_NOT_SCORABLE` lists why).

What this cannot check: which shard set each input trained on. A ``Checkpoint`` carries the
batch order's ``consumed_digest`` but not the shard hash; the ledger rows of the runs do.
Pass ``--ft-row-ids`` to record them in the manifest so the average names its sources;
``real_ft_run.py --score-checkpoint`` refuses an average whose manifest names none.

Usage::

    python tools/ckpt_average.py --out /path/avg.safetensors CKPT.json CKPT.json [...]
    python tools/ckpt_average.py --from masters --ft-row-ids ID ID ID \\
        --out /path/avg.safetensors CKPT.json CKPT.json CKPT.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_train.memory import CheckpointRefused, tensor_ranges
from qd_train.run_control import (
    Checkpoint,
    LRSchedule,
    TensorRef,
    tensor_refs_from_safetensors,
)

#: The model subtrees averaged. Everything else in ``model_state`` is either compared for
#: equality (the scalars below) or deliberately dropped (the optimizer's moments).
AVERAGED_TREES: Final[tuple[str, ...]] = ("tower", "span_head")
#: Scalars that must agree across inputs, or the tensors mean different things.
MATCHED_SCALARS: Final[tuple[str, ...]] = ("vocab_size", "span_weight")
#: RSI's "last k" is typically 3-5; beyond a handful this is a sweep, not an average.
MAX_INPUTS: Final[int] = 64
#: ``--from``'s choices; ``tower`` is the default and what this tool always did.
SOURCES: Final[tuple[str, ...]] = ("tower", "masters", "lora-merged")
#: Where a ``--optimizer master`` checkpoint keeps its fp32 masters in ``model_state``.
MASTERS_PATH: Final[tuple[str, str]] = ("optimizer", "masters")
#: The manifest is written beside the weights as ``<out>.manifest.json``.
MANIFEST_SUFFIX: Final[str] = ".manifest.json"
#: The prompt layout an input's weights were trained under, as ``QwenDecisionStep.state``
#: writes it into ``model_state`` (only when it is not 1), and as the average's manifest states
#: it at top level (only when it is not 1) for qd-export, which reads an absent key as 1.
PROMPT_FORMAT_KEY: Final[str] = "prompt_format"
TOOL: Final[str] = "tools/ckpt_average.py"
#: Where the averaging rule was ported from, recorded in every manifest as ``source``.
RSI_SOURCE: Final[str] = "RSI-Jev rsijev/train.py:189-200 average_checkpoints (MIT) @8f34a4f"
#: The dtypes that have a mean. Anything else must be equal across inputs and is copied.
FLOAT_DTYPES: Final[frozenset[str]] = frozenset({"bfloat16", "float16", "float32", "float64"})
METHODS: Final[Mapping[str, str]] = {
    "tower": "arithmetic mean in float32, cast back to each tensor's dtype",
    "masters": (
        "arithmetic mean of each parameter's fp32 master (its tower tensor where it has no "
        "master), accumulated in float64 and cast to the tower tensor's dtype once"
    ),
    "lora-merged": (
        "arithmetic mean of each adapted weight's merged W' = W + (alpha/r) B A (B A in float32), "
        "accumulated in float64 and cast to the tower tensor's dtype once; the span head and D17 "
        "as the span-head rule says; never A and B averaged separately"
    ),
}
#: What the LoRA reference arm (``qd_train.lora``) puts in ``model_state`` besides the frozen base
#: ``tower``: ``lora_A``/``lora_B`` (module name -> f32 ``[r, in]``/``[out, r]``), the scalars
#: ``lora_r`` and ``lora_alpha``, and an optional ``d17`` tree.
LORA_TREES: Final[tuple[str, ...]] = ("lora_A", "lora_B")
LORA_SCALARS: Final[tuple[str, ...]] = ("lora_r", "lora_alpha")
D17_TREE: Final[str] = "d17"
#: Any of these keys in ``model_state``, or a tensor named ``*.lora_A.*``/``*.lora_B.*`` in the
#: ``tower`` or ``span_head`` tree, marks adapter-level state.
ADAPTER_STATE_KEYS: Final[frozenset[str]] = frozenset({*LORA_TREES, *LORA_SCALARS})
#: ``--span-head-seed``'s manifest key, and its two modes.
SPAN_HEAD_SOURCE_KEY: Final[str] = "span_head_source"
SPAN_HEAD_MODES: Final[tuple[str, ...]] = ("seed", "averaged")
#: What ``tensor_sources`` says of a span-head tensor taken from one input rather than averaged.
SPAN_HEAD_SEED_SOURCE: Final[str] = "span_head_seed"


class AverageRefusal(ValueError):
    """The inputs cannot be averaged into one meaningful model."""


class AdapterLevelAverageRefused(AverageRefusal):
    """The input carries LoRA adapter state and the average asked for would treat it as a dense
    model: A and B averaged separately (``mean(B) mean(A) != mean(B A)``), or dropped unseen."""


@dataclass(frozen=True, slots=True)
class InputRecord:
    """One input as the manifest names it: the file, its seed, and its body's digest."""

    path: Path
    seed: int
    payload_digest: str

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "resolved_path": str(self.path.resolve()),
            "seed": self.seed,
            "payload_digest": self.payload_digest,
        }


@dataclass(frozen=True, slots=True)
class _Facts:
    """What two inputs must agree on to be averaged: everything but the tensor bytes."""

    schedule: dict[str, Any]
    optimizer_step: int
    scalars: dict[str, Any]
    trees: dict[str, dict[str, tuple[str, tuple[int, ...]]]]
    weights_digest: str
    prompt_format: int


def prompt_format_of(state: Mapping[str, Any], *, where: str) -> int:
    """The prompt format ``state`` (an input's ``model_state`` or an average's manifest) says
    its weights were trained under: absent is 1, every checkpoint before v5's format 2; present,
    an int >= 1, never a bool, as qd-export's ``read_source_manifest`` takes it."""
    value = state.get(PROMPT_FORMAT_KEY, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AverageRefusal(f"{where}: {PROMPT_FORMAT_KEY} {value!r} is not an int >= 1")
    return value


def _tree(model_state: Mapping[str, Any], name: str, *, where: str) -> Mapping[str, TensorRef]:
    tree = model_state.get(name)
    if not isinstance(tree, Mapping) or not tree:
        raise AverageRefusal(
            f"{where}: model_state has no {name!r} tensors. Only a checkpoint written by "
            "QwenDecisionStep.state carries a tower; the stand-in never writes one."
        )
    for key, ref in tree.items():
        if not isinstance(ref, TensorRef):
            raise AverageRefusal(f"{where}: {name}[{key!r}] is {type(ref).__name__}, not a tensor")
    return tree


def refuse_adapter_state(model_state: Mapping[str, Any], *, where: str) -> None:
    """Refuse an input that carries LoRA adapter state, by name.

    Only ``--from lora-merged`` may read one. Every other source would average the dense trees
    and silently drop the adapters (a "model" that is the frozen base), or, with the adapter
    leaves written into the tower tree, average A and B separately -- which is not the average of
    the products. Works on a revived ``model_state`` and on a body's (only keys are read).
    """
    held = sorted(ADAPTER_STATE_KEYS & set(model_state))
    leaves = sorted(
        f"{tree}.{key}"
        for tree in AVERAGED_TREES
        if isinstance(model_state.get(tree), Mapping)
        for key in model_state[tree]
        if ".lora_A." in f".{key}." or ".lora_B." in f".{key}."
    )
    if held or leaves:
        raise AdapterLevelAverageRefused(
            f"{where}: carries LoRA adapter state ({(held + leaves)[:4]}). An adapter-level "
            "average -- A and B averaged separately, or the adapters left out -- is not the "
            "average of the models: mean(B) mean(A) != mean(B A). Average the merged weights "
            "with --from lora-merged"
        )


def _facts(
    model_state: Mapping[str, Any], *, schedule: Mapping[str, Any], optimizer_step: int,
    where: str, allow_adapters: bool = False,
) -> _Facts:
    if not allow_adapters:
        refuse_adapter_state(model_state, where=where)
    trees = {t: _tree(model_state, t, where=where) for t in AVERAGED_TREES}
    return _Facts(
        schedule=dict(schedule),
        optimizer_step=optimizer_step,
        scalars={s: model_state.get(s) for s in MATCHED_SCALARS},
        trees={t: {k: (r.dtype, r.shape) for k, r in trees[t].items()} for t in AVERAGED_TREES},
        weights_digest=hashlib.sha256(
            "".join(
                f"{t}.{k}:{trees[t][k].digest()}" for t in AVERAGED_TREES for k in sorted(trees[t])
            ).encode()
        ).hexdigest(),
        prompt_format=prompt_format_of(model_state, where=where),
    )


def _check_count(n: int) -> None:
    if n < 2:
        raise AverageRefusal(f"averaging needs at least 2 checkpoints, got {n}")
    if n > MAX_INPUTS:
        raise AverageRefusal(f"{n} inputs exceeds MAX_INPUTS={MAX_INPUTS}")


def _check_against(
    first: _Facts, first_name: str, mine: _Facts, name: str, digests: dict[str, str]
) -> None:
    """Refuse unless ``mine`` is ``first``'s model at ``first``'s point, and new."""
    _check_schedule(first, first_name, mine, name)
    if mine.optimizer_step != first.optimizer_step:
        raise AverageRefusal(
            f"{name} is at optimizer step {mine.optimizer_step} and {first_name} at "
            f"{first.optimizer_step}; an average across two points of training is "
            "neither of them"
        )
    _check_same_model(first, first_name, mine, name, digests)


def _check_schedule(first: _Facts, first_name: str, mine: _Facts, name: str) -> None:
    if mine.schedule != first.schedule:
        raise AverageRefusal(
            f"{name} was trained on schedule {mine.schedule} and {first_name} "
            f"on {first.schedule}. Different schedules are different runs, not seeds of one."
        )


def _check_same_model(
    first: _Facts, first_name: str, mine: _Facts, name: str, digests: dict[str, str]
) -> None:
    """The prompt format, the scalars, the trees name for name, and weights no other input
    already holds."""
    if mine.prompt_format != first.prompt_format:
        raise AverageRefusal(
            f"{name} was trained on prompt format {mine.prompt_format} and {first_name} on "
            f"{first.prompt_format}: an average across two prompt layouts is a model of "
            "neither, and its manifest could state only one"
        )
    for scalar in MATCHED_SCALARS:
        if mine.scalars[scalar] != first.scalars[scalar]:
            raise AverageRefusal(
                f"{name} has {scalar}={mine.scalars[scalar]!r} and {first_name} "
                f"{first.scalars[scalar]!r}"
            )
    for tree_name in AVERAGED_TREES:
        ours, ref = mine.trees[tree_name], first.trees[tree_name]
        if set(ours) != set(ref):
            extra = sorted(set(ours) - set(ref))[:5]
            missing = sorted(set(ref) - set(ours))[:5]
            raise AverageRefusal(
                f"{name}'s {tree_name} is a different architecture from {first_name}'s: "
                f"extra {extra}, missing {missing}"
            )
        for key in ref:
            if ours[key] != ref[key]:
                raise AverageRefusal(
                    f"{name} {tree_name}[{key!r}] is {ours[key][0]}{list(ours[key][1])} "
                    f"and {first_name}'s is {ref[key][0]}{list(ref[key][1])}"
                )
    if mine.weights_digest in digests:
        raise AverageRefusal(
            f"{name} and {digests[mine.weights_digest]} hold identical weights; averaging "
            "them would weight one run twice while the manifest counted two"
        )
    digests[mine.weights_digest] = name


def check_compatible(ckpts: Sequence[Checkpoint], names: Sequence[str]) -> None:
    """Refuse unless every input is the same model at the same point of the same schedule."""
    _check_count(len(ckpts))
    facts = [
        _facts(c.model_state, schedule=c.schedule.to_json(), optimizer_step=c.optimizer_step,
               where=n)
        for c, n in zip(ckpts, names, strict=True)
    ]
    digests: dict[str, str] = {}
    for mine, name in zip(facts, names, strict=True):
        _check_against(facts[0], names[0], mine, name, digests)


def _tensor(ref: TensorRef) -> Any:
    import torch

    return torch.frombuffer(bytearray(ref.data), dtype=getattr(torch, ref.dtype)).reshape(
        ref.shape
    )


def check_trajectory(ckpts: Sequence[Checkpoint], names: Sequence[str]) -> list[int]:
    """``--same-seed-trajectory``: refuse unless the inputs are ONE run's own snapshots at
    distinct steps, and return those steps.

    One seed; optimizer steps strictly ascending in the order given; one schedule; the same
    scalars and trees; no two inputs holding the same weights; and each input's loss log
    the start of the next one's -- the evidence, from the files alone, that they are points
    of one run (the same recipe and the same batches) rather than runs that happen to share
    a seed and a schedule. What it cannot see is the recipe itself; ``--ft-row-ids`` names
    the run's ledger row.
    """
    _check_count(len(ckpts))
    seeds = sorted({c.seed for c in ckpts})
    if len(seeds) != 1:
        raise AverageRefusal(
            f"--same-seed-trajectory averages one run's snapshots, and these are seeds {seeds}: "
            "an average across seeds at distinct steps is neither a trajectory nor an ensemble"
        )
    steps = [int(c.optimizer_step) for c in ckpts]
    if any(b <= a for a, b in itertools.pairwise(steps)):
        raise AverageRefusal(
            f"--same-seed-trajectory takes distinct optimizer steps in ascending order; got "
            f"{steps}. Name the snapshots in step order"
        )
    facts = [
        _facts(c.model_state, schedule=c.schedule.to_json(), optimizer_step=c.optimizer_step,
               where=n)
        for c, n in zip(ckpts, names, strict=True)
    ]
    digests: dict[str, str] = {}
    for mine, name in zip(facts, names, strict=True):
        _check_schedule(facts[0], names[0], mine, name)
        _check_same_model(facts[0], names[0], mine, name, digests)
    for i in range(1, len(ckpts)):
        head = ckpts[i - 1].loss_log.as_tuple()
        if ckpts[i].loss_log.as_tuple()[: len(head)] != head:
            raise AverageRefusal(
                f"{names[i - 1]}'s loss log ({len(head)} steps) is not the start of "
                f"{names[i]}'s: they are not points of one run"
            )
    return steps


def _mean_trees(ckpts: Sequence[Checkpoint], names: Sequence[str]) -> dict[str, Any]:
    """The averaging itself, once the inputs have been checked: float tensors in float32 and
    cast back, a non-float one copied when every input holds it unchanged."""
    import torch

    out: dict[str, Any] = {}
    for tree_name in AVERAGED_TREES:
        keys = sorted(_tree(ckpts[0].model_state, tree_name, where=names[0]))
        for key in keys:
            refs = [
                _tree(c.model_state, tree_name, where=n)[key]
                for c, n in zip(ckpts, names, strict=True)
            ]
            dtype = getattr(torch, refs[0].dtype)
            if not dtype.is_floating_point:
                values = [_tensor(r) for r in refs]
                if not all(torch.equal(values[0], v) for v in values[1:]):
                    raise AverageRefusal(
                        f"{tree_name}[{key!r}] is a non-float tensor ({refs[0].dtype}) that "
                        "differs between inputs; it has no mean"
                    )
                out[f"{tree_name}.{key}"] = values[0].clone()
                continue
            acc = torch.zeros(refs[0].shape, dtype=torch.float32)
            for r in refs:
                acc += _tensor(r).to(torch.float32)
            out[f"{tree_name}.{key}"] = (acc / len(refs)).to(dtype)
    return out


def average(ckpts: Sequence[Checkpoint], names: Sequence[str]) -> dict[str, Any]:
    """``{"tower.<k>": tensor, "span_head.<k>": tensor}``, averaged in fp32, cast back."""
    check_compatible(ckpts, names)
    return _mean_trees(ckpts, names)


#: What a ``--same-seed-trajectory`` manifest's ``trajectory.rule`` says the average is.
TRAJECTORY_RULE: Final[str] = (
    "the arithmetic mean of one run's own snapshots at distinct optimizer steps (one seed, "
    "one schedule, each input's loss log the start of the next's), not N seeds at one step"
)
#: Why ``real_ft_run.py --score-checkpoint`` cannot score one (``read_manifest`` says so).
TRAJECTORY_NOT_SCORABLE: Final[str] = (
    "real_ft_run.py --score-checkpoint pairs an average with one ft row per input, each a "
    "distinct seed at the average's one optimizer step (ckpt_average.read_manifest refuses "
    "repeated seeds and ft rows, verify_sources one step per input; real_ft_run "
    "_check_score_checkpoint_flags wants >= 2 seeds, _averaged_weights one row per input at "
    "the row's last step and inputs named <tag>-seed<N>-<device>.json); scoring this one "
    "needs each of those relaxed for the trajectory block alone, which is not written"
)


def trajectory_block(
    ckpts: Sequence[Checkpoint], steps: Sequence[int], ft_row_id: str | None
) -> dict[str, Any]:
    """The manifest's ``trajectory`` block: the seed, every input's step in order, the run's
    ft row when named, and the rule."""
    return {
        "seed": int(ckpts[0].seed), "optimizer_steps": [int(s) for s in steps],
        "ft_row_id": ft_row_id, "rule": TRAJECTORY_RULE,
    }



# --- --from masters --------------------------------------------------------------------------


def _masters_of(weights: Mapping[str, Any], *, where: str) -> list[TensorRef]:
    optimizer = weights.get(MASTERS_PATH[0])
    masters = optimizer.get(MASTERS_PATH[1]) if isinstance(optimizer, Mapping) else None
    if not isinstance(masters, list) or not masters:
        raise AverageRefusal(f"{where}: model_state['optimizer'] holds no fp32 masters")
    for i, ref in enumerate(masters):
        if not isinstance(ref, TensorRef) or ref.dtype != "float32":
            raise AverageRefusal(
                f"{where}: master {i} is "
                f"{ref.dtype if isinstance(ref, TensorRef) else type(ref).__name__}, not a "
                "float32 tensor; MasterWeightAdamW keeps every master in fp32"
            )
    return masters


def _as_dtype(ref: TensorRef, dtype: str) -> TensorRef:
    """``ref`` cast to ``dtype`` as ``MasterWeightAdamW.step``'s ``live.copy_(master)`` casts
    it: torch's round-to-nearest-even, on whichever device."""
    import torch

    cast = _tensor(ref).to(getattr(torch, dtype)).contiguous()
    return TensorRef(
        dtype=dtype, shape=ref.shape, data=cast.reshape(-1).view(torch.uint8).numpy().tobytes()
    )


def master_names(weights: Mapping[str, Any], *, where: str) -> list[str]:
    """``names[i]``: the ``tower.<k>`` or ``span_head.<k>`` whose fp32 master is
    ``optimizer.masters[i]``, by the identity ``cast(master_i) == live tensor``.

    Refused when a master casts to no tensor of its shape (the masters and the weights in
    one body disagree), to more than one (two parameters hold identical bytes, and which
    master is whose cannot be read off them), or when two masters land on one name.
    """
    index: dict[tuple[str, tuple[int, ...], str], list[str]] = {}
    dtypes: dict[tuple[int, ...], set[str]] = {}
    for tree_name in AVERAGED_TREES:
        for key, ref in _tree(weights, tree_name, where=where).items():
            if ref.dtype not in FLOAT_DTYPES:
                continue
            index.setdefault((ref.dtype, ref.shape, ref.digest()), []).append(
                f"{tree_name}.{key}"
            )
            dtypes.setdefault(ref.shape, set()).add(ref.dtype)
    names: list[str] = []
    for i, master in enumerate(_masters_of(weights, where=where)):
        hits: list[str] = []
        for dtype in sorted(dtypes.get(master.shape, ())):
            cast = master if dtype == master.dtype else _as_dtype(master, dtype)
            hits.extend(index.get((dtype, master.shape, cast.digest()), []))
        if not hits:
            raise AverageRefusal(
                f"{where}: master {i} {list(master.shape)}, cast to the dtype of every tensor "
                "of its shape, equals no tower or span_head tensor. MasterWeightAdamW copies "
                "each master into its parameter after every step, so the masters and the "
                "weights in this body do not describe one model."
            )
        if len(hits) > 1:
            raise AverageRefusal(
                f"{where}: master {i} {list(master.shape)} is ambiguous: it equals each of "
                f"{hits}, whose bytes are identical, so which master is whose cannot be read "
                "off the checkpoint"
            )
        names.append(hits[0])
    first_at: dict[str, int] = {}
    for i, name in enumerate(names):
        if name in first_at:
            raise AverageRefusal(
                f"{where}: two masters ({first_at[name]} and {i}) both equal {name}; each "
                "parameter has one master"
            )
        first_at[name] = i
    return names


@dataclass(frozen=True, slots=True)
class MastersAverage:
    """What :func:`average_masters` produced, and what the manifest says about it."""

    tensors: dict[str, Any]
    #: ``"master"`` or ``"tower"`` per output tensor.
    sources: dict[str, str]
    #: The master's index in ``optimizer.masters`` per tensor that came from one.
    master_index: dict[str, int]
    inputs: list[InputRecord]
    facts: _Facts
    #: Each tensor's tower dtype: what a float64 mean is cast to, once.
    out_dtype: dict[str, str]


def average_masters(
    paths: Sequence[Path], names: Sequence[str], *, keep_float64: bool = False
) -> MastersAverage:
    """Each parameter's fp32 masters averaged across the inputs, by name.

    One input is read at a time (:meth:`Checkpoint.read_weights` of ``tower``, ``span_head``
    and ``optimizer.masters``; never the moments) and folded into float64 accumulators, so
    the peak is one input plus the accumulators whatever N is. For the real 2B (1,881,825,088
    parameters): ~10.5 GiB of input, ~14.0 GiB of float64 accumulators and the embedding's
    transient copies (~5.7 GiB), about 30 GiB; then the bf16 output (~3.5 GiB) and its
    serialization (~3.5 GiB) as the accumulators drain.

    ``keep_float64`` returns each float tensor's float64 MEAN, uncast, with ``out_dtype``
    naming the dtype it is to be cast to once: what :func:`norm_preserving_average` moves
    from before its one cast.
    """
    import torch

    _check_count(len(paths))
    # From the bodies alone, before gigabytes are read: every input has masters to average.
    for path, name in zip(paths, names, strict=True):
        state = Checkpoint.read_body(path).get("model_state", {})
        refuse_adapter_state(state, where=name)
        optimizer = state.get("optimizer")
        if not isinstance(optimizer, dict) or MASTERS_PATH[1] not in optimizer:
            held = sorted(optimizer) if isinstance(optimizer, dict) else None
            raise AverageRefusal(
                f"{name} has no fp32 masters (model_state['optimizer'] holds {held}): it was "
                "not trained with --optimizer master. Average it with --from tower."
            )
    acc: dict[str, Any] = {}
    kept: dict[str, TensorRef] = {}
    out_dtype: dict[str, str] = {}
    mapping: list[str] | None = None
    first: _Facts | None = None
    digests: dict[str, str] = {}
    records: list[InputRecord] = []
    for path, name in zip(paths, names, strict=True):
        weights, meta = Checkpoint.read_weights(path, subtrees=(*AVERAGED_TREES, MASTERS_PATH))
        facts = _facts(
            weights, schedule=LRSchedule.from_json(meta["schedule"]).to_json(),
            optimizer_step=int(meta["optimizer_step"]), where=name,
        )
        first = facts if first is None else first
        _check_against(first, names[0], facts, name, digests)
        mine = master_names(weights, where=name)
        if mapping is None:
            mapping = mine
        elif mine != mapping:
            at = next(
                (i for i, (a, b) in enumerate(zip(mapping, mine, strict=False)) if a != b),
                min(len(mapping), len(mine)),
            )
            raise AverageRefusal(
                f"{name}'s masters map to other names than {names[0]}'s: master {at} is "
                f"{mine[at] if at < len(mine) else 'absent'} there and "
                f"{mapping[at] if at < len(mapping) else 'absent'} here; one recipe builds "
                "its optimizer over its parameters in one order"
            )
        master_of = {n: i for i, n in enumerate(mapping)}
        masters = _masters_of(weights, where=name)
        pending: dict[str, TensorRef] = {}
        for tree_name in AVERAGED_TREES:
            for key, ref in _tree(weights, tree_name, where=name).items():
                full = f"{tree_name}.{key}"
                out_dtype[full] = ref.dtype
                if ref.dtype not in FLOAT_DTYPES:
                    if full in kept and kept[full].digest() != ref.digest():
                        raise AverageRefusal(
                            f"{full} is a non-float tensor ({ref.dtype}) that differs between "
                            "inputs; it has no mean"
                        )
                    kept.setdefault(full, ref)
                    continue
                src = masters[master_of[full]] if full in master_of else ref
                if src.shape != ref.shape:
                    raise AverageRefusal(
                        f"{name}: the master mapped to {full} is {list(src.shape)} and the "
                        f"tensor is {list(ref.shape)}"
                    )
                pending[full] = src
        # The input's own references go now, so each tensor is freed as it is folded in.
        del weights, masters
        while pending:
            full, ref = pending.popitem()
            value = _tensor(ref).to(torch.float64)
            if full in acc:
                acc[full].add_(value)
            else:
                acc[full] = value
            del value, ref
        records.append(InputRecord(
            path=path, seed=int(meta["seed"]), payload_digest=str(meta["payload_digest"]),
        ))
    assert mapping is not None and first is not None  # at least two inputs were read
    tensors: dict[str, Any] = {}
    for full in sorted(set(acc) | set(kept)):
        if full in kept:
            tensors[full] = _tensor(kept.pop(full)).clone()
            continue
        mean = acc.pop(full).div_(len(paths))
        if keep_float64:
            tensors[full] = mean
            continue
        tensors[full] = _cast_once(mean, out_dtype[full])
        del mean
    master_of = {n: i for i, n in enumerate(mapping)}
    return MastersAverage(
        tensors=tensors,
        sources={full: "master" if full in master_of else "tower" for full in tensors},
        master_index={full: master_of[full] for full in tensors if full in master_of},
        inputs=records,
        facts=first,
        out_dtype=dict(out_dtype),
    )


def _cast_once(value: Any, dtype: str) -> Any:
    """A float64 tensor to ``dtype``, once: float64 -> float32 -> the tower dtype. torch's
    float64 -> bfloat16 conversion goes through float32 anyway, and saying so makes the one
    cast to bf16 the stated one."""
    import torch

    return value.to(torch.float32).to(getattr(torch, dtype))


# --- --norm-preserving -----------------------------------------------------------------------

#: The norm-preserving average (Fable I-2): the uniform mean of N fine-tuning deltas whose
#: pairwise cosine is c has norm sqrt((1 + (N-1)c)/N) of one delta's, so it shrinks every
#: seed-specific direction; lambda restores the norm. Recorded verbatim in the manifest.
NORM_PRESERVING_FORMULA: Final[str] = (
    "theta = base + lambda * mean_i(theta_i - base); lambda = sqrt(n / (1 + (n - 1) * c)), "
    "c = the mean over input pairs of the cosine between their deltas (fp32 master - base) "
    "over every tower tensor with a master, in float64"
)
#: What the span head gets under --norm-preserving: it is initialised at random, so it has
#: no base to move from, and it is averaged plainly.
SPAN_HEAD_NO_BASE: Final[str] = (
    "span_head: averaged plainly (the arithmetic mean of its masters); it has no base, so "
    "lambda is not applied to it"
)
#: The base tensor's name for a tower tensor: the repository's own mapping.
BASE_PREFIX_NOTE: Final[str] = "base name = qd_train.backbone.TEXT_PREFIX + tower key"
#: Elements per float64 slice when the deltas of one tensor are compared: bounds the
#: transient copies of the embedding (508.6M elements) at a few hundred MiB per input.
DELTA_CHUNK: Final[int] = 1 << 25


class _SidecarTensors:
    """One checkpoint's tensors read ONE AT A TIME from its sidecar, each checked against the
    body's recorded dtype, shape, size and digest -- the check ``Checkpoint.read_weights``
    makes (``run_control._join_tensors``) -- after the body's ``payload_digest`` verifies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.body = Checkpoint.read_body(path)
        declared = self.body.get("sidecar")
        if not isinstance(declared, Mapping) or not isinstance(declared.get("digest"), str):
            raise AverageRefusal(f"{path} declares no tensor sidecar")
        self.sidecar = Checkpoint.sidecar_path(path, declared["digest"])
        if not self.sidecar.is_file():
            raise AverageRefusal(f"{path}: its sidecar {self.sidecar.name} is not beside it")
        try:
            self.ranges = tensor_ranges(self.sidecar)
        except CheckpointRefused as exc:
            raise AverageRefusal(str(exc)) from exc

    def master(self, index: int) -> Any:
        """``optimizer.masters[index]`` as a float32 torch tensor, verified."""
        from qd_train.run_control import _TENSOR_REF_TAG, _join_tensors

        masters = self.body["model_state"][MASTERS_PATH[0]][MASTERS_PATH[1]]
        node = masters[index]
        entry = node.get(_TENSOR_REF_TAG) if isinstance(node, Mapping) else None
        if entry is None:
            raise AverageRefusal(f"{self.path}: master {index} is not a tensor in the body")
        key = str(entry["key"])
        found = self.ranges.get(key)
        if found is None:
            raise AverageRefusal(f"{self.sidecar.name} does not carry {key}")
        with self.sidecar.open("rb") as fh:
            fh.seek(found.start)
            data = fh.read(found.end - found.start)
        if len(data) != found.end - found.start:
            raise AverageRefusal(f"{self.sidecar}: {key} is truncated")
        try:
            ref = _join_tensors(node, {key: TensorRef(
                dtype=str(entry["dtype"]), shape=tuple(int(a) for a in entry["shape"]),
                data=data,
            )})
        except ValueError as exc:
            raise AverageRefusal(f"{self.path}: {exc}") from exc
        del data
        return _tensor(ref)


def _pair_key(i: int, j: int) -> str:
    return f"{i}-{j}"


@dataclass(frozen=True, slots=True)
class NormPreservingAverage:
    """What :func:`norm_preserving_average` produced: the average, and its manifest block."""

    average: MastersAverage
    block: dict[str, Any]


def norm_preserving_average(
    paths: Sequence[Path], names: Sequence[str], *, base: Path,
    lambda_not_derived_from_c: float | None = None,
) -> NormPreservingAverage:
    """``theta = base + lambda * mean_i(theta_i - base)`` (:data:`NORM_PRESERVING_FORMULA`).

    1. :func:`average_masters` with every input check it makes, keeping the float64 means.
    2. c, streamed tensor by tensor: for every tower tensor with a master, each input's
       master is read alone from its sidecar (verified), the base tensor beside it, and the
       float64 dot products and squared norms of the deltas are accumulated in slices of
       :data:`DELTA_CHUNK` elements. Per pair, cosine = dot / (norm_i * norm_j); c is their
       mean. lambda = sqrt(n / (1 + (n - 1) c)).
    3. Every float tower tensor moves from its base by lambda times the mean delta, in
       float64, and is cast once; the span head (no base) is the plain mean; a non-float
       tensor is copied as the masters average copies it.

    The peak is the float64 means (~14.0 GiB for the real 2B) plus one tensor of every
    input as fp32 (the embedding: 3 x 1.9 GiB) plus one slice: below the masters average's
    own ~30 GiB, which this begins with.

    ``lambda_not_derived_from_c`` replaces lambda by a number the caller chose, and the
    block says so (``lambda_derived_from_c: false``) beside the lambda c gives.
    """
    import torch
    from safetensors import safe_open

    from qd_train.backbone import TEXT_PREFIX, BackboneContractViolation, text_tensor_index

    if not base.is_dir():
        raise AverageRefusal(
            f"--base-snapshot {base} is not a directory: the norm-preserving average moves "
            "from the base the seeds were fine-tuned from, and there is none to move from"
        )
    try:
        base_index = text_tensor_index(base)
    except BackboneContractViolation as exc:
        raise AverageRefusal(f"--base-snapshot {base}: {exc}") from exc
    done = average_masters(paths, names, keep_float64=True)
    n = len(paths)
    tower = sorted(
        full for full, value in done.tensors.items()
        if full.startswith("tower.") and value.dtype == torch.float64
    )
    problems: list[str] = []
    for full in tower:
        key = full[len("tower."):]
        if key not in base_index:
            problems.append(f"{full} has no base tensor {TEXT_PREFIX}{key}")
        elif tuple(base_index[key][1]) != tuple(done.tensors[full].shape):
            problems.append(
                f"{full} is {list(done.tensors[full].shape)} and its base "
                f"{list(base_index[key][1])}: a remapped vocabulary has no base row for row"
            )
    if problems:
        raise AverageRefusal(
            f"--base-snapshot {base.name} is not the base of these checkpoints: "
            + "; ".join(problems[:5])
            + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
        )

    def base_tensor(key: str) -> Any:
        shard = base_index[key][0]
        with safe_open(str(shard), "pt") as handle:
            return handle.get_tensor(TEXT_PREFIX + key)

    measured = sorted(full for full in tower if full in done.master_index)
    if not measured:
        raise AverageRefusal("no tower tensor has a master, so there is no delta to measure c on")
    readers = [_SidecarTensors(p) for p in paths]
    sq = [0.0] * n
    dot = {_pair_key(i, j): 0.0 for i in range(n) for j in range(i + 1, n)}
    n_params = 0
    for full in measured:
        key = full[len("tower."):]
        flat_base = base_tensor(key).reshape(-1)
        masters = [r.master(done.master_index[full]).reshape(-1) for r in readers]
        if any(m.numel() != flat_base.numel() for m in masters):
            raise AverageRefusal(f"{full}: a master and its base hold different element counts")
        n_params += flat_base.numel()
        for lo in range(0, flat_base.numel(), DELTA_CHUNK):
            b = flat_base[lo:lo + DELTA_CHUNK].to(torch.float64)
            deltas = [m[lo:lo + DELTA_CHUNK].to(torch.float64) - b for m in masters]
            for i in range(n):
                sq[i] += float(torch.dot(deltas[i], deltas[i]))
                for j in range(i + 1, n):
                    dot[_pair_key(i, j)] += float(torch.dot(deltas[i], deltas[j]))
            del b, deltas
        del flat_base, masters
    zero = [names[i] for i in range(n) if not sq[i] > 0.0]
    if zero:
        raise AverageRefusal(
            f"{zero} moved no tower master away from the base: a zero delta has no direction, "
            "so c is undefined"
        )
    norms = [math.sqrt(s) for s in sq]
    cosine = {
        _pair_key(i, j): dot[_pair_key(i, j)] / (norms[i] * norms[j])
        for i in range(n) for j in range(i + 1, n)
    }
    c = sum(cosine.values()) / len(cosine)
    denominator = 1.0 + (n - 1) * c
    if not denominator > 0.0:
        raise AverageRefusal(
            f"c = {c!r} makes 1 + (n - 1) c = {denominator!r}, not positive: the mean delta is "
            "zero or the deltas cancel, and no lambda restores its norm"
        )
    from_c = math.sqrt(n / denominator)
    lam = from_c
    if lambda_not_derived_from_c is not None:
        lam = float(lambda_not_derived_from_c)
        if not (math.isfinite(lam) and lam > 0.0):
            raise AverageRefusal(f"--lambda-not-derived-from-c {lam!r} is not a positive number")
    for full in tower:
        mean = done.tensors[full]
        b = base_tensor(full[len("tower."):]).to(torch.float64)
        done.tensors[full] = _cast_once(mean.sub_(b).mul_(lam).add_(b), done.out_dtype[full])
        del mean, b
    span = [
        full for full, value in done.tensors.items()
        if full.startswith("span_head.") and value.dtype == torch.float64
    ]
    for full in span:
        done.tensors[full] = _cast_once(done.tensors[full], done.out_dtype[full])
    leftover = [k for k, v in done.tensors.items() if v.dtype == torch.float64
                and done.out_dtype.get(k) != "float64"]
    if leftover:  # pragma: no cover - every float tensor is a tower or span_head tensor
        raise AverageRefusal(f"tensors left uncast: {leftover[:5]}")
    block = {
        "formula": NORM_PRESERVING_FORMULA,
        "n": n,
        "c": c,
        "pairwise_cosine": cosine,
        "pairs_are": "input indices, in the order of 'inputs'",
        "delta_norms": norms,
        "lambda_from_c": from_c,
        "lambda": lam,
        "lambda_derived_from_c": lambda_not_derived_from_c is None,
        "c_measured_over": {
            "tensors": len(measured), "parameters": n_params,
            "rule": "every tower tensor with an fp32 master; float64",
        },
        "base_snapshot": base.name,
        "base_files": {
            str(p.name): _sha256_file(p)
            for p in sorted({shard for shard, _ in base_index.values()})
        },
        "base_mapping": BASE_PREFIX_NOTE,
        "lambda_applied_to": f"{len(tower)} tower tensors",
        "span_head": SPAN_HEAD_NO_BASE,
    }
    return NormPreservingAverage(average=done, block=block)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --- --from lora-merged ----------------------------------------------------------------------

#: What a ``lora-merged`` manifest's ``lora.formula`` says each adapted weight is.
LORA_MERGE_FORMULA: Final[str] = (
    "W' = W + (alpha / r) * (B @ A), B @ A in float32; the inputs' W' accumulated in float64 and "
    "cast to the tower dtype once"
)


@dataclass(frozen=True, slots=True)
class LoraMergedAverage:
    """What :func:`average_lora_merged` produced, and what the manifest says about it."""

    tensors: dict[str, Any]
    sources: dict[str, str]
    inputs: list[InputRecord]
    facts: _Facts
    block: dict[str, Any]


def _lora_scalar(state: Mapping[str, Any], key: str, *, where: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AverageRefusal(f"{where}: model_state[{key!r}] is {value!r}, not an int >= 1")
    return value


def _lora_trees(
    state: Mapping[str, Any], *, where: str
) -> tuple[Mapping[str, TensorRef], Mapping[str, TensorRef]]:
    out = []
    for name in LORA_TREES:
        tree = state.get(name)
        if not isinstance(tree, Mapping) or not tree:
            raise AverageRefusal(
                f"{where}: model_state has no {name!r} tensors. --from lora-merged averages a "
                "LoRA checkpoint's adapters (model_state['lora_A'], ['lora_B'], 'lora_r', "
                "'lora_alpha' and the frozen 'tower'); this is not one"
            )
        for key, ref in tree.items():
            if not isinstance(ref, TensorRef) or ref.dtype != "float32":
                raise AverageRefusal(
                    f"{where}: {name}[{key!r}] is not a float32 tensor; the adapters are f32"
                )
        out.append(tree)
    return out[0], out[1]


def _same_refs(
    first: Mapping[str, TensorRef], mine: Mapping[str, TensorRef], *, what: str, first_name: str,
    name: str, digests: bool,
) -> None:
    if set(first) != set(mine):
        raise AverageRefusal(
            f"{name}'s {what} differs from {first_name}'s: extra "
            f"{sorted(set(mine) - set(first))[:5]}, missing {sorted(set(first) - set(mine))[:5]}"
        )
    for key, ref in first.items():
        other = mine[key]
        if (ref.dtype, ref.shape) != (other.dtype, other.shape):
            raise AverageRefusal(
                f"{name} {what}[{key!r}] is {other.dtype}{list(other.shape)} and "
                f"{first_name}'s is {ref.dtype}{list(ref.shape)}"
            )
        if digests and ref.digest() != other.digest():
            raise AverageRefusal(
                f"{name} and {first_name} do not share one frozen base: {what}[{key!r}] differs. "
                "LoRA seeds adapt one frozen tower; an average over two bases is a model of neither"
            )


def average_lora_merged(paths: Sequence[Path], names: Sequence[str]) -> LoraMergedAverage:
    """``--from lora-merged``: the mean of each input's merged ``W' = W + (alpha/r) B A``.

    The frozen base is shared, so the mean of the merged weights is ``W + (alpha/r) mean(B A)``
    -- not ``W + (alpha/r) mean(B) mean(A)``, which is what an adapter-level average would give
    and what this refuses. Each input is read alone (its tower, adapters, span head and ``d17``),
    checked against the first, and folded into float64 accumulators; an adapted weight is cast
    to the tower dtype once, at the end, and every other tensor is the base's own.

    Refused: an input with no adapters, ``lora_r``/``lora_alpha`` that differ (they set the
    scale), towers that are not byte-identical (not one frozen base), adapter module sets or
    shapes that differ or do not fit the tower's weight, ``d17`` in some inputs and not others,
    and the checks every average makes (one schedule, one step, scalars, one prompt format, no
    input twice).
    """
    import torch

    _check_count(len(paths))
    for path, name in zip(paths, names, strict=True):
        state = Checkpoint.read_body(path).get("model_state", {})
        if not isinstance(state, Mapping) or not any(k in state for k in LORA_TREES):
            raise AverageRefusal(
                f"{name}: model_state has no 'lora_A' tensors. --from lora-merged averages a "
                "LoRA checkpoint; this is not one"
            )
    has_d17 = [D17_TREE in Checkpoint.read_body(p)["model_state"] for p in paths]
    if any(has_d17) and not all(has_d17):
        raise AverageRefusal(
            "d17 is in some inputs and not others: "
            + ", ".join(f"{n}: {'yes' if h else 'no'}" for n, h in zip(names, has_d17, strict=True))
        )
    subtrees = (*AVERAGED_TREES, *LORA_TREES, *((D17_TREE,) if all(has_d17) else ()))
    first_state: dict[str, Any] | None = None
    first_facts: _Facts | None = None
    first_scalars: tuple[int, int] | None = None
    seen_adapters: dict[str, str] = {}
    acc: dict[str, Any] = {}
    other_acc: dict[str, Any] = {}
    base_dtype: dict[str, str] = {}
    records: list[InputRecord] = []
    for path, name in zip(paths, names, strict=True):
        state, meta = Checkpoint.read_weights(path, subtrees=subtrees)
        facts = _facts(
            state, schedule=LRSchedule.from_json(meta["schedule"]).to_json(),
            optimizer_step=int(meta["optimizer_step"]), where=name, allow_adapters=True,
        )
        a_tree, b_tree = _lora_trees(state, where=name)
        scalars = (
            _lora_scalar(state, "lora_r", where=name), _lora_scalar(state, "lora_alpha", where=name)
        )
        if first_state is None:
            first_state, first_facts, first_scalars = state, facts, scalars
        else:
            assert first_facts is not None and first_scalars is not None
            _check_schedule(first_facts, names[0], facts, name)
            if facts.optimizer_step != first_facts.optimizer_step:
                raise AverageRefusal(
                    f"{name} is at optimizer step {facts.optimizer_step} and {names[0]} at "
                    f"{first_facts.optimizer_step}"
                )
            if facts.prompt_format != first_facts.prompt_format:
                raise AverageRefusal(f"{name} and {names[0]} were trained on different prompts")
            for scalar in MATCHED_SCALARS:
                if facts.scalars[scalar] != first_facts.scalars[scalar]:
                    raise AverageRefusal(
                        f"{name} has {scalar}={facts.scalars[scalar]!r} and {names[0]} "
                        f"{first_facts.scalars[scalar]!r}"
                    )
            for label, mine, theirs in zip(LORA_SCALARS, scalars, first_scalars, strict=True):
                if mine != theirs:
                    raise AverageRefusal(
                        f"{name} has {label}={mine} and {names[0]} {theirs}: the scale "
                        "alpha/r differs, so the adapters are not the same kind of update"
                    )
            _same_refs(first_state["tower"], state["tower"], what="tower", first_name=names[0],
                       name=name, digests=True)
            _same_refs(first_state["span_head"], state["span_head"], what="span_head",
                       first_name=names[0], name=name, digests=False)
            _same_refs(first_state["lora_A"], a_tree, what="lora_A", first_name=names[0],
                       name=name, digests=False)
            _same_refs(first_state["lora_B"], b_tree, what="lora_B", first_name=names[0],
                       name=name, digests=False)
            if D17_TREE in state:
                _same_refs(first_state[D17_TREE], state[D17_TREE], what=D17_TREE,
                           first_name=names[0], name=name, digests=False)
        r, alpha = scalars
        if set(a_tree) != set(b_tree):
            raise AverageRefusal(f"{name}: lora_A and lora_B name different modules")
        tower = state["tower"]
        adapter_digest = hashlib.sha256(
            "".join(
                f"{k}:{a_tree[k].digest()}:{b_tree[k].digest()}" for k in sorted(a_tree)
            ).encode()
        ).hexdigest()
        if adapter_digest in seen_adapters:
            raise AverageRefusal(
                f"{name} and {seen_adapters[adapter_digest]} hold identical adapters; averaging "
                "them would weight one run twice while the manifest counted two"
            )
        seen_adapters[adapter_digest] = name
        scaling = alpha / r
        for module in sorted(a_tree):
            key = f"{module}.weight"
            if key not in tower:
                raise AverageRefusal(
                    f"{name}: adapter for {module!r} but the tower has no {key!r}"
                )
            w_ref, a_ref, b_ref = tower[key], a_tree[module], b_tree[module]
            if (
                len(w_ref.shape) != 2 or a_ref.shape != (r, w_ref.shape[1])
                or b_ref.shape != (w_ref.shape[0], r)
            ):
                raise AverageRefusal(
                    f"{name}: {module!r} adapters A{list(a_ref.shape)} B{list(b_ref.shape)} do "
                    f"not fit {key} {list(w_ref.shape)} at r={r}"
                )
            if w_ref.dtype not in FLOAT_DTYPES:
                raise AverageRefusal(f"{name}: {key} is {w_ref.dtype}, which has no merge")
            merged = _tensor(w_ref).to(torch.float32) + scaling * (
                _tensor(b_ref).to(torch.float32) @ _tensor(a_ref).to(torch.float32)
            )
            base_dtype[key] = w_ref.dtype
            value = merged.to(torch.float64)
            if key in acc:
                acc[key].add_(value)
            else:
                acc[key] = value
            del merged, value
        for tree_name, tree in (("span_head", state["span_head"]),
                                *(((D17_TREE, state[D17_TREE]),) if D17_TREE in state else ())):
            for key, ref in tree.items():
                if ref.dtype not in FLOAT_DTYPES:
                    raise AverageRefusal(f"{name}: {tree_name}[{key!r}] is {ref.dtype}; no mean")
                full = f"{tree_name}.{key}"
                base_dtype[full] = ref.dtype
                value = _tensor(ref).to(torch.float32)
                if full in other_acc:
                    other_acc[full] += value
                else:
                    other_acc[full] = value.clone()
        records.append(InputRecord(
            path=path, seed=int(meta["seed"]), payload_digest=str(meta["payload_digest"]),
        ))
        del state
    assert first_state is not None and first_facts is not None and first_scalars is not None
    n = len(paths)
    tensors: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, ref in first_state["tower"].items():
        full = f"tower.{key}"
        if key in acc:
            tensors[full] = _cast_once(acc.pop(key).div_(n), base_dtype[key])
            sources[full] = "lora-merged"
        else:
            tensors[full] = _tensor(ref).clone()
            sources[full] = "tower"
    for full, total in other_acc.items():
        tensors[full] = (total / n).to(getattr(torch, base_dtype[full]))
        sources[full] = "tower"
    r, alpha = first_scalars
    block = {
        "r": r, "alpha": alpha, "scaling": alpha / r,
        "adapted_modules": len(first_state["lora_A"]),
        "formula": LORA_MERGE_FORMULA,
        "base_tower_digest": first_facts.weights_digest,
        "d17": D17_TREE in first_state,
    }
    return LoraMergedAverage(
        tensors=tensors, sources=sources, inputs=records, facts=first_facts, block=block
    )


# --- --span-head-seed ------------------------------------------------------------------------


def span_head_source_block(
    mode: str, *, seed: int | None = None, record: InputRecord | None = None
) -> dict[str, Any]:
    """The manifest's ``span_head_source``: how the span head was made, so a primary (averaged)
    and a fallback (one seed's) row can never be pooled."""
    if mode == "averaged":
        return {"mode": "averaged", "seed": None, "input": None, "payload_digest": None}
    assert record is not None and seed is not None
    return {
        "mode": "seed", "seed": seed, "input": str(record.path),
        "payload_digest": record.payload_digest,
    }


def ft_row_init(ledger: Path, row_id: str, *, seed: int) -> dict[str, Any] | None:
    """The span-head init identity the ft row ``row_id`` recorded for its run (its
    ``recipe.span_head_init``: ``sha256``, ``content_digest``, and ``shared`` for a shared head),
    or ``None`` when the run recorded none.

    A checkpoint body carries no init identity, so the ledger row of the run that wrote it is
    the only evidence; the row must be exactly one completed ft row of ``seed``.
    """
    try:
        rows = [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise AverageRefusal(f"--ft-ledger {ledger}: {exc}") from exc
    found = [r for r in rows if isinstance(r, dict) and r.get("row_id") == row_id]
    if len(found) != 1:
        raise AverageRefusal(f"--ft-ledger {ledger}: {len(found)} rows have row_id {row_id!r}")
    row = found[0]
    protocol = row.get("protocol")
    if row.get("run_kind") != "ft" or row.get("status") != "completed":
        raise AverageRefusal(f"ft row {row_id} is not a completed ft row")
    if not isinstance(protocol, dict) or protocol.get("seed") != seed:
        raise AverageRefusal(f"ft row {row_id} is not seed {seed}'s")
    recipe = row.get("recipe")
    init = recipe.get("span_head_init") if isinstance(recipe, dict) else None
    if init is None:
        return None
    if not isinstance(init, dict):
        raise AverageRefusal(f"ft row {row_id}: recipe.span_head_init is not an object")
    return {
        "ft_row_id": row_id, "sha256": init.get("sha256"),
        "content_digest": init.get("content_digest"), "shared": init.get("shared") is True,
    }


def check_span_head_inits(
    ledger: Path, row_ids: Sequence[str], seeds: Sequence[int], head_at: int, *, v6_run: bool
) -> dict[str, Any] | None:
    """What ``span_head_source`` records of the selected seed's init, after Fable's R7 check.

    A run is treated as v6 when the caller says ``--v6-run`` OR any input's ft row records a
    shared init, so forgetting the flag cannot skip the check. For a v6 run EVERY input's row must
    record a shared init and all of them the same one, by ``content_digest`` (a seed-specific
    init, none, or two different "shared" heads is refused: a head grown from an init the other
    seeds did not share is not a stand-in for the average's, and a run that records no init
    identity cannot be verified, so it is refused rather than trusted).
    """
    inits = [ft_row_init(ledger, r, seed=s) for r, s in zip(row_ids, seeds, strict=True)]
    v6 = v6_run or any(i is not None and i["shared"] for i in inits)
    chosen = inits[head_at]
    if v6:
        # R7 starts EVERY seed from one shared head, so the check is on every input's row, not
        # only the selected seed's: a row that says "shared" for a different head, or that
        # records no init, is not evidence that the heads being averaged had one start.
        for init, row_id, seed in zip(inits, row_ids, seeds, strict=True):
            if init is None or not init["shared"]:
                raise AverageRefusal(
                    f"v6 run: the ft row of seed {seed} ({row_id}) "
                    + ("records no span_head_init, so what init its head grew from cannot be "
                       "verified" if init is None
                       else "records a span-head init that is not the shared one")
                    + ". R7 starts every seed from one shared head; refusing rather than "
                    "trusting the caller"
                )
            digest = init["content_digest"]
            if not isinstance(digest, str) or not digest:
                raise AverageRefusal(
                    f"v6 run: the ft row of seed {seed} ({row_id}) records a shared span-head "
                    "init with no content_digest, so it cannot be compared with the other "
                    "seeds' inits; refusing"
                )
        digests = {i["content_digest"]: r for i, r in zip(inits, row_ids, strict=True) if i}
        if len(digests) != 1:
            raise AverageRefusal(
                "v6 run: the seeds' ft rows record "
                f"{len(digests)} different span-head init content digests "
                f"({', '.join(f'{r}: {d}' for d, r in sorted(digests.items()))}), not one init. "
                "R7 starts every seed from the same shared head; refusing"
            )
    return chosen


def select_span_head_input(
    paths: Sequence[Path], names: Sequence[str], seed: int
) -> int:
    """The index of the one input whose checkpoint BODY says ``seed`` (never a file name)."""
    seeds = [int(Checkpoint.read_body(p)["seed"]) for p in paths]
    hits = [i for i, s in enumerate(seeds) if s == seed]
    if not hits:
        raise AverageRefusal(
            f"--span-head-seed {seed}: no input's checkpoint body says seed {seed}; the inputs "
            f"are seeds {seeds}"
        )
    if len(hits) > 1:
        raise AverageRefusal(
            f"--span-head-seed {seed}: more than one input carries seed {seed} "
            f"({[names[i] for i in hits]}), so which head is meant is ambiguous"
        )
    return hits[0]


def take_span_head(
    tensors: dict[str, Any], sources: dict[str, str], path: Path, name: str
) -> list[str]:
    """Replace every ``span_head.*`` tensor in ``tensors`` by the one in ``path``'s checkpoint
    (verified by ``Checkpoint.read_weights``), and say so in ``sources``. Refuses a head whose
    names, dtypes or shapes are not the averaged one's. Returns the replaced names."""
    state, _ = Checkpoint.read_weights(path, subtrees=("span_head",))
    head = state["span_head"]
    mine = {f"span_head.{k}": ref for k, ref in head.items()}
    held = {k for k in tensors if k.startswith("span_head.")}
    if set(mine) != held:
        raise AverageRefusal(
            f"{name}'s span_head tensors {sorted(mine)[:5]} are not the average's "
            f"{sorted(held)[:5]}"
        )
    for full, ref in mine.items():
        current = tensors[full]
        if (tuple(current.shape), str(current.dtype).removeprefix("torch.")) != (
            ref.shape, ref.dtype
        ):
            raise AverageRefusal(
                f"{name}: {full} is {ref.dtype}{list(ref.shape)}, not the average's"
            )
        tensors[full] = _tensor(ref).clone()
        sources[full] = SPAN_HEAD_SEED_SOURCE
    return sorted(mine)


# --- the output and its manifest -------------------------------------------------------------


def manifest_path(out: Path) -> Path:
    """Where the manifest of the average at ``out`` is: ``<out>.manifest.json``."""
    return out.with_name(out.name + MANIFEST_SUFFIX)


def _refuse_existing(out: Path) -> None:
    for target in (out, manifest_path(out)):
        if target.exists():
            raise AverageRefusal(f"{target} already exists; refusing to overwrite it")


def write(
    tensors: Mapping[str, Any],
    out: Path,
    *,
    source: str,
    inputs: Sequence[InputRecord],
    optimizer_step: int,
    schedule: Mapping[str, Any],
    scalars: Mapping[str, Any],
    ft_row_ids: Sequence[str],
    tensor_sources: Mapping[str, str],
    master_index: Mapping[str, int],
    norm_preserving: Mapping[str, Any] | None = None,
    trajectory: Mapping[str, Any] | None = None,
    prompt_format: int = 1,
    span_head_source: Mapping[str, Any] | None = None,
    lora: Mapping[str, Any] | None = None,
) -> Path:
    """The safetensors file and ``<out>.manifest.json``, each written atomically.

    Refuses to overwrite either: an average is an artifact other rows will name.
    ``norm_preserving`` is :func:`norm_preserving_average`'s block, recorded under that key
    (and only then: a plain average's manifest is what it always was). ``trajectory`` is
    ``--same-seed-trajectory``'s block (:func:`trajectory_block`), likewise only when given.
    ``prompt_format`` is the inputs' one format (:func:`prompt_format_of`, checked equal across
    them), written as the top-level key qd-export stamps the release with -- only when it is
    not 1, so a v4 average's manifest, and the sha256 eval rows name it by, do not move.
    ``span_head_source`` (:func:`span_head_source_block`) is written whenever it is given, which
    ``main`` always does (ruling 2g: a primary and a fallback row must never pool); an average
    made before the key existed has none, and reads as averaged.
    ``lora`` is ``--from lora-merged``'s block, written only for that source.
    """
    from safetensors.torch import save

    from qd_train.run_control import _atomic_write_bytes

    stated = prompt_format_of({PROMPT_FORMAT_KEY: prompt_format}, where="write")
    if source not in SOURCES:
        raise AverageRefusal(f"source {source!r} is not one of {SOURCES}")
    if norm_preserving is not None and source != "masters":
        raise AverageRefusal("a norm-preserving average is made from the fp32 masters")
    if trajectory is not None and (source != "tower" or norm_preserving is not None):
        raise AverageRefusal("a same-seed trajectory average is a plain mean of the towers")
    if (lora is not None) != (source == "lora-merged"):
        raise AverageRefusal("a lora block belongs to --from lora-merged, and to nothing else")
    if span_head_source is not None and span_head_source.get("mode") not in SPAN_HEAD_MODES:
        raise AverageRefusal(f"span_head_source.mode must be one of {SPAN_HEAD_MODES}")
    if set(tensor_sources) != set(tensors):
        raise AverageRefusal("tensor_sources must name exactly the tensors written")
    _refuse_existing(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = save(dict(tensors))
    _atomic_write_bytes(out, payload)
    body = {
        "tool": TOOL,
        "method": METHODS[source] if norm_preserving is None else NORM_PRESERVING_FORMULA,
        "source": RSI_SOURCE,
        # The --from this average was made with. Not "source", which has always been the
        # attribution above.
        "from": source,
        "accumulator": "float32" if source == "tower" else "float64",
        "resumable": False,
        "why_not_resumable": (
            "weights only: the optimizer moments were not averaged and are not here, and "
            "a resume from weights alone restarts Adam's moments from zero"
        ),
        "n_inputs": len(inputs),
        "optimizer_step": optimizer_step,
        "schedule": dict(schedule),
        **{s: scalars.get(s) for s in MATCHED_SCALARS},
        "inputs": [record.to_json() for record in inputs],
        "ft_row_ids": list(ft_row_ids),
        "safetensors_sha256": hashlib.sha256(payload).hexdigest(),
        "n_tensors": len(tensors),
        "tensor_sources": dict(sorted(tensor_sources.items())),
        "master_index": dict(sorted(master_index.items())),
        **({} if norm_preserving is None else {"norm_preserving": dict(norm_preserving)}),
        **({} if trajectory is None else {"trajectory": dict(trajectory)}),
        **({} if span_head_source is None else {SPAN_HEAD_SOURCE_KEY: dict(span_head_source)}),
        **({} if lora is None else {"lora": dict(lora)}),
        **({} if stated == 1 else {PROMPT_FORMAT_KEY: stated}),
    }
    manifest = manifest_path(out)
    _atomic_write_bytes(
        manifest, (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


# --- reading an average back (real_ft_run.py --score-checkpoint) -----------------------------


@dataclass(frozen=True, slots=True)
class AverageManifest:
    """An average's manifest, as :func:`read_manifest` checked it."""

    weights: Path
    path: Path
    #: sha256 of the manifest file's bytes: what an eval row names the average by.
    sha256: str
    body: dict[str, Any]

    @property
    def source(self) -> str:
        """The ``--from`` the average was made with: ``masters`` or ``tower``."""
        return str(self.body["from"])

    @property
    def inputs(self) -> list[dict[str, Any]]:
        return list(self.body["inputs"])

    @property
    def seeds(self) -> list[int]:
        return [int(record["seed"]) for record in self.body["inputs"]]

    @property
    def ft_row_ids(self) -> list[str]:
        return list(self.body["ft_row_ids"])

    @property
    def optimizer_step(self) -> int:
        return int(self.body["optimizer_step"])

    @property
    def span_head_source(self) -> dict[str, Any] | None:
        """How the span head was made (:func:`span_head_source_block`), checked by
        :func:`read_manifest`; ``None`` for an average that predates the key (it is averaged)."""
        block = self.body.get(SPAN_HEAD_SOURCE_KEY)
        return None if block is None else dict(block)

    @property
    def norm_preserving(self) -> dict[str, Any] | None:
        """The ``--norm-preserving`` block, checked by :func:`read_manifest`; ``None`` for a
        plain average."""
        block = self.body.get("norm_preserving")
        return None if block is None else dict(block)


def _norm_preserving_problems(block: object, *, n_inputs: int, source: object) -> list[str]:
    """What is wrong with a manifest's ``norm_preserving`` block: every number a scored row
    will repeat must be there, and a lambda said to come from c must be the one c gives."""
    if not isinstance(block, dict):
        return ["norm_preserving is not an object"]
    problems: list[str] = []
    if source != "masters":
        problems.append("a norm-preserving average is made from the masters")
    if block.get("formula") != NORM_PRESERVING_FORMULA:
        problems.append("norm_preserving.formula is not this tool's")
    numbers = {k: block.get(k) for k in ("c", "lambda", "lambda_from_c")}
    for key, value in numbers.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(
            float(value)
        ):
            problems.append(f"norm_preserving.{key} is {value!r}, not a finite number")
    if block.get("n") != n_inputs:
        problems.append(f"norm_preserving.n is {block.get('n')!r} for {n_inputs} inputs")
    derived = block.get("lambda_derived_from_c")
    if not isinstance(derived, bool):
        problems.append("norm_preserving.lambda_derived_from_c is not a bool")
    if not isinstance(block.get("base_snapshot"), str) or not block.get("base_snapshot"):
        problems.append("norm_preserving names no base_snapshot")
    if problems:
        return problems
    c, lam, from_c = (float(numbers[k]) for k in ("c", "lambda", "lambda_from_c"))  # type: ignore[arg-type]
    denominator = 1.0 + (n_inputs - 1) * c
    if not denominator > 0.0 or not math.isclose(
        from_c, math.sqrt(n_inputs / denominator), rel_tol=1e-12, abs_tol=0.0
    ):
        problems.append(
            f"norm_preserving.lambda_from_c {from_c!r} is not sqrt(n / (1 + (n - 1) c)) at "
            f"n={n_inputs}, c={c!r}"
        )
    if derived and lam != from_c:
        problems.append(
            f"norm_preserving.lambda {lam!r} is said to come from c and is not lambda_from_c "
            f"{from_c!r}"
        )
    return problems


def _span_head_source_problems(block: object, inputs: Sequence[object]) -> list[str]:
    """What is wrong with a manifest's ``span_head_source``: a seed-mode source must name one
    of the manifest's own inputs, by the seed and digest that input recorded."""
    if not isinstance(block, dict):
        return ["span_head_source is not an object"]
    mode = block.get("mode")
    if mode not in SPAN_HEAD_MODES:
        return [f"span_head_source.mode is {mode!r}, not one of {SPAN_HEAD_MODES}"]
    if mode == "averaged":
        stray = [k for k in ("seed", "input", "payload_digest") if block.get(k) is not None]
        return [f"span_head_source is averaged and names {stray}"] if stray else []
    records = [r for r in inputs if isinstance(r, dict)]
    match = [r for r in records if r.get("path") == block.get("input")]
    if len(match) != 1:
        return [f"span_head_source.input {block.get('input')!r} is not one of the inputs"]
    problems = []
    if match[0].get("seed") != block.get("seed"):
        problems.append(
            f"span_head_source says seed {block.get('seed')!r} for an input of seed "
            f"{match[0].get('seed')!r}"
        )
    if match[0].get("payload_digest") != block.get("payload_digest"):
        problems.append("span_head_source.payload_digest is not that input's")
    return problems


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def read_manifest(weights: Path) -> AverageManifest:
    """The manifest beside ``weights``, refused unless it is one this tool wrote and it
    names its inputs, their seeds and digests, and one ft row per input."""
    if weights.suffix != ".safetensors":
        raise AverageRefusal(f"{weights.name}: an average this tool writes ends in .safetensors")
    path = manifest_path(weights)
    if not path.is_file():
        raise AverageRefusal(
            f"{weights.name} has no manifest at {path}; without it nothing says which "
            "checkpoints and which ft rows these weights are"
        )
    raw = path.read_bytes()
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise AverageRefusal(f"{path}: a manifest is a JSON object")
    if "trajectory" in body:
        # Said first and by name: the checks below would refuse it too, for repeated seeds,
        # which reads as a corrupt manifest rather than as a kind of average not scored here.
        block = body["trajectory"]
        steps = block.get("optimizer_steps") if isinstance(block, dict) else None
        raise AverageRefusal(
            f"{path} is a same-seed trajectory average (--same-seed-trajectory, steps "
            f"{steps}): {TRAJECTORY_NOT_SCORABLE}"
        )
    problems: list[str] = []
    if body.get("tool") != TOOL:
        problems.append(f"tool is {body.get('tool')!r}, not {TOOL!r}")
    if body.get("resumable") is not False:
        problems.append("it does not say resumable: false")
    if body.get("from") not in SOURCES:
        problems.append(f"'from' is {body.get('from')!r}, not one of {SOURCES}")
    inputs = body.get("inputs")
    fields = {"path": str, "resolved_path": str, "seed": int, "payload_digest": str}
    if not isinstance(inputs, list) or len(inputs) < 2:
        problems.append("it names fewer than 2 inputs")
        inputs = []
    for i, record in enumerate(inputs):
        if not isinstance(record, dict):
            problems.append(f"input {i} is not an object")
            continue
        wrong = [k for k, t in fields.items() if not isinstance(record.get(k), t)]
        if wrong:
            problems.append(f"input {i} lacks {wrong}")
        elif not _is_hex64(record["payload_digest"]):
            problems.append(f"input {i}'s payload_digest is not a sha256")
    if body.get("n_inputs") != len(inputs):
        problems.append(f"n_inputs is {body.get('n_inputs')!r} for {len(inputs)} inputs")
    seeds = [r.get("seed") for r in inputs if isinstance(r, dict)]
    if len(set(seeds)) != len(seeds):
        problems.append(f"its inputs' seeds {seeds} repeat")
    ids = body.get("ft_row_ids")
    if not isinstance(ids, list) or not ids:
        problems.append(
            "it names no ft_row_ids: pass --ft-row-ids when averaging, one ledger ft row "
            "per input, or nothing ties these weights to the runs that trained them"
        )
    elif len(ids) != len(inputs) or not all(isinstance(x, str) and x for x in ids):
        problems.append(f"ft_row_ids {ids} is not one row id per input")
    elif len(set(ids)) != len(ids):
        problems.append(f"ft_row_ids {ids} repeats a row")
    if not isinstance(body.get("optimizer_step"), int):
        problems.append("it records no optimizer_step")
    if not _is_hex64(body.get("safetensors_sha256")):
        problems.append("it records no safetensors_sha256")
    if not isinstance(body.get("vocab_size"), int) or not isinstance(
        body.get("span_weight"), (int, float)
    ):
        problems.append("it records no vocab_size and span_weight")
    try:
        prompt_format_of(body, where="it")
    except AverageRefusal as exc:
        problems.append(str(exc))
    if SPAN_HEAD_SOURCE_KEY in body:
        problems.extend(_span_head_source_problems(body[SPAN_HEAD_SOURCE_KEY], inputs))
    if (body.get("from") == "lora-merged") != isinstance(body.get("lora"), dict):
        problems.append("a lora block belongs to --from lora-merged, and to nothing else")
    if "norm_preserving" in body:
        problems.extend(_norm_preserving_problems(
            body["norm_preserving"], n_inputs=len(inputs), source=body.get("from")
        ))
    if problems:
        raise AverageRefusal(f"{path} is not a manifest an average can be scored from: "
                             + "; ".join(problems))
    return AverageManifest(
        weights=weights, path=path, sha256=hashlib.sha256(raw).hexdigest(), body=body
    )


def verify_sources(manifest: AverageManifest) -> None:
    """Each input's JSON body, where the manifest says it was, hashes to the
    ``payload_digest`` the manifest recorded.

    **A source that is not on disk is refused** (fail closed): a digest that cannot be
    checked must not read as one that matched. Only the body is read -- a few hundred KB
    for the real 2B, ``payload_digest`` covers it and through it every tensor's digest --
    so scoring elsewhere needs the three ``.json`` files copied to their recorded paths,
    never the 24.66 GiB sidecars.
    """
    for record in manifest.inputs:
        source = Path(record["resolved_path"])
        if not source.is_file():
            raise AverageRefusal(
                f"input {record['path']} (seed {record['seed']}) is not on disk at {source}, "
                "so its payload_digest cannot be checked against the manifest's. Refusing: "
                "copy that checkpoint's JSON body there (its sidecar is never read), or "
                "score where it is."
            )
        try:
            body = Checkpoint.read_body(source)
        except ValueError as exc:
            raise AverageRefusal(f"input {source}: {exc}") from exc
        if body.get("payload_digest") != record["payload_digest"]:
            raise AverageRefusal(
                f"{source} has payload_digest {str(body.get('payload_digest'))[:16]}... and "
                f"the manifest recorded {record['payload_digest'][:16]}...: it is not the "
                "checkpoint this average was made from"
            )
        if int(body["seed"]) != int(record["seed"]):
            raise AverageRefusal(
                f"{source} is seed {body['seed']} and the manifest says {record['seed']}"
            )
        if int(body["optimizer_step"]) != manifest.optimizer_step:
            raise AverageRefusal(
                f"{source} is at optimizer step {body['optimizer_step']} and the manifest "
                f"at {manifest.optimizer_step}"
            )


def read_average(manifest: AverageManifest) -> dict[str, Any]:
    """The averaged weights as ``QwenDecisionStep.load_weights`` takes them: ``tower`` and
    ``span_head`` trees of ``TensorRef`` plus ``vocab_size`` and ``span_weight``, after the
    file's sha256 is checked against the manifest's ``safetensors_sha256``."""
    payload = manifest.weights.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != manifest.body["safetensors_sha256"]:
        raise AverageRefusal(
            f"{manifest.weights.name} hashes to {actual[:16]}... and its manifest's "
            f"safetensors_sha256 is {manifest.body['safetensors_sha256'][:16]}...: these are "
            "not the weights the manifest describes"
        )
    state: dict[str, Any] = {tree: {} for tree in AVERAGED_TREES}
    for name, ref in tensor_refs_from_safetensors(payload).items():
        tree, _, key = name.partition(".")
        if tree == D17_TREE:
            raise AverageRefusal(
                f"{manifest.weights.name} holds a d17 tree: the full-fine-tune scorer reads a "
                "tower and a span head and has no D17 readout, so scoring it here would score a "
                "different model than the one averaged. Score a LoRA average with qd-metal"
            )
        if tree not in state or not key:
            raise AverageRefusal(
                f"{manifest.weights.name} holds {name!r}, outside {AVERAGED_TREES}"
            )
        state[tree][key] = ref
    n = sum(len(t) for t in state.values())
    if n != manifest.body.get("n_tensors"):
        raise AverageRefusal(
            f"{manifest.weights.name} holds {n} tensors and its manifest says "
            f"{manifest.body.get('n_tensors')}"
        )
    return {**state, "vocab_size": manifest.body["vocab_size"],
            "span_weight": manifest.body["span_weight"]}


# --- the command line ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="the .safetensors to write")
    parser.add_argument(
        "--from", dest="source", choices=SOURCES, default="tower",
        help=(
            "tower (default): average each input's tower and span_head tensors in float32. "
            "masters: average each parameter's fp32 master (a --optimizer master checkpoint's "
            "model_state['optimizer']['masters']) in float64, its tower tensor where it has "
            "none, and cast to the tower dtype once. lora-merged: a LoRA checkpoint (frozen "
            "tower, lora_A/lora_B, lora_r/lora_alpha): the mean of each adapted weight's merged "
            "W' = W + (alpha/r) B A, cast once; A and B are never averaged separately, and every "
            "other source refuses a checkpoint that carries adapters"
        ),
    )
    parser.add_argument(
        "--ft-row-ids", nargs="*", default=[],
        help="the ledger ft rows of the inputs, in input order, recorded in the manifest",
    )
    parser.add_argument(
        "--norm-preserving", action="store_true",
        help=(
            "with --from masters: base + lambda * mean(theta_i - base), lambda = "
            "sqrt(n / (1 + (n - 1) c)) where c is the mean pairwise cosine of the inputs' "
            "fine-tuning deltas against --base-snapshot, measured here (Fable I-2). The span "
            "head has no base and is averaged plainly. Recorded in the manifest"
        ),
    )
    parser.add_argument(
        "--base-snapshot", type=Path, default=None,
        help="with --norm-preserving: the pretrained snapshot the inputs were fine-tuned from",
    )
    parser.add_argument(
        "--lambda-not-derived-from-c", type=float, default=None, metavar="LAMBDA",
        help=(
            "with --norm-preserving: use this lambda INSTEAD of the one c gives. The manifest "
            "says lambda_derived_from_c: false and keeps the derived one beside it; a scored "
            "row of such an average is quick"
        ),
    )
    parser.add_argument(
        "--same-seed-trajectory", action="store_true",
        help=(
            "average ONE run's own snapshots at distinct optimizer steps (v5's last-3 "
            "average of --retain-tower-every snapshots) instead of N seeds at one step: one "
            "seed, steps ascending in the order given, one schedule, and each input's loss "
            "log the start of the next's. --from tower only; --ft-row-ids names the run's one "
            "ft row. The manifest's 'trajectory' block records the steps; real_ft_run.py "
            "--score-checkpoint refuses to score it"
        ),
    )
    parser.add_argument(
        "--span-head-seed", type=int, default=None, metavar="N",
        help=(
            "take the span head from the one input whose checkpoint body says seed N, instead of "
            "averaging the heads (v5's averaged heads are near-orthogonal and their mean "
            "collapses span). The tower is averaged as usual, under --from masters and "
            "--norm-preserving too. Refused when no input or more than one carries seed N, and "
            "with --same-seed-trajectory. The manifest records span_head_source"
        ),
    )
    parser.add_argument(
        "--span-head-shared-init", action="store_true",
        help=(
            "average the span heads, which is refused unless every input's head grew from ONE "
            "shared init (heads from independent inits are near-orthogonal and their mean "
            "collapses span: GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06). The proof is "
            "read from the ledger, never from the caller: --ft-ledger and one --ft-row-ids per "
            "input, every row recording the same shared recipe.span_head_init (R7, by "
            "content_digest). The alternative is --span-head-seed. Not needed for one input or "
            "--same-seed-trajectory (one run's snapshots)"
        ),
    )
    parser.add_argument(
        "--ft-ledger", type=Path, default=None,
        help=(
            "with --span-head-seed or --span-head-shared-init: the ledger holding the inputs' ft "
            "rows (--ft-row-ids, one per input in order). A checkpoint carries no span-head init "
            "identity; the row's recipe.span_head_init does, and is recorded in span_head_source"
        ),
    )
    parser.add_argument(
        "--v6-run", action="store_true",
        help=(
            "with --span-head-seed: require the selected seed's ft row to record the shared "
            "span-head init (R7). Also assumed whenever any input's row records a shared init"
        ),
    )
    args = parser.parse_args(argv)
    resolved = [p.resolve() for p in args.checkpoints]
    if len(set(resolved)) != len(resolved):
        raise SystemExit("the same checkpoint was named twice; it would be weighted double")
    if args.span_head_seed is not None and args.span_head_shared_init:
        raise SystemExit(
            "--span-head-seed takes one seed's head and --span-head-shared-init averages them: "
            "choose one"
        )
    if args.span_head_seed is None and not args.span_head_shared_init and (
        args.v6_run or args.ft_ledger is not None
    ):
        raise SystemExit(
            "--v6-run and --ft-ledger mean something only with --span-head-seed or "
            "--span-head-shared-init"
        )
    if args.span_head_shared_init and args.same_seed_trajectory:
        raise SystemExit(
            "--span-head-shared-init is for averaging several seeds' heads; "
            "--same-seed-trajectory is one run's snapshots and needs no proof"
        )
    if args.span_head_shared_init and (
        args.ft_ledger is None or len(args.ft_row_ids) != len(args.checkpoints)
    ):
        raise SystemExit(
            "--span-head-shared-init needs --ft-ledger and one --ft-row-ids per input: the init "
            "the heads grew from is read from the ft rows, never from the caller"
        )
    if args.span_head_seed is not None and args.same_seed_trajectory:
        raise SystemExit(
            "--span-head-seed picks one seed's head, and --same-seed-trajectory's inputs are one "
            "seed's snapshots: there is no other seed to take a head from"
        )
    if args.span_head_seed is not None and (
        args.ft_ledger is None or len(args.ft_row_ids) != len(args.checkpoints)
    ):
        raise SystemExit(
            "--span-head-seed needs --ft-ledger and one --ft-row-ids per input: the init the "
            "chosen head grew from is read from the ft rows, never from the caller"
        )
    if args.same_seed_trajectory:
        if args.source != "tower" or args.norm_preserving:
            raise SystemExit(
                "--same-seed-trajectory is a plain mean of one run's towers: --from tower, "
                "without --norm-preserving (a snapshot keeps no fp32 masters)"
            )
        if len(args.ft_row_ids) > 1:
            raise SystemExit(
                f"--same-seed-trajectory averages one run: --ft-row-ids names its one ft row, "
                f"got {len(args.ft_row_ids)}"
            )
    elif args.ft_row_ids and len(args.ft_row_ids) != len(args.checkpoints):
        raise SystemExit(
            f"{len(args.ft_row_ids)} --ft-row-ids for {len(args.checkpoints)} checkpoints"
        )
    if args.out.suffix != ".safetensors":
        raise SystemExit(f"--out {args.out}: the average is written as a .safetensors file")
    if args.norm_preserving and (args.source != "masters" or args.base_snapshot is None):
        raise SystemExit(
            "--norm-preserving moves from the base by the fp32 masters' mean delta: it needs "
            "--from masters and --base-snapshot"
        )
    if not args.norm_preserving and (
        args.base_snapshot is not None or args.lambda_not_derived_from_c is not None
    ):
        raise SystemExit(
            "--base-snapshot and --lambda-not-derived-from-c mean something only with "
            "--norm-preserving"
        )
    if (
        args.span_head_seed is None and not args.span_head_shared_init
        and not args.same_seed_trajectory and len(args.checkpoints) > 1
    ):
        raise SystemExit(
            f"refused: this would average {len(args.checkpoints)} span heads with no proof they "
            "grew from one shared init; heads from independent inits are near-orthogonal and "
            "their mean collapses span "
            "(GAP-WEIGHT-AVERAGE-COLLAPSES-THE-SPAN-HEAD-2026-10-06). Choose --span-head-seed N "
            "(take one seed's head) or prove the shared init with --span-head-shared-init "
            "--ft-ledger L --ft-row-ids ... (R7: every input's ft row records the same shared "
            "span_head_init)"
        )
    names = [str(p) for p in args.checkpoints]
    norm_block: dict[str, Any] | None = None
    trajectory: dict[str, Any] | None = None
    lora_block: dict[str, Any] | None = None
    try:
        _refuse_existing(args.out)
        if args.source != "lora-merged":
            for path, name in zip(args.checkpoints, names, strict=True):
                refuse_adapter_state(
                    Checkpoint.read_body(path).get("model_state", {}), where=name
                )
        head_at: int | None = None
        init_block: dict[str, Any] | None = None
        if args.span_head_seed is not None:
            head_at = select_span_head_input(args.checkpoints, names, args.span_head_seed)
            init_block = check_span_head_inits(
                args.ft_ledger, args.ft_row_ids,
                [int(Checkpoint.read_body(p)["seed"]) for p in args.checkpoints], head_at,
                v6_run=args.v6_run,
            )
        if args.span_head_shared_init:
            # Every input's row must record the one shared init (v6_run=True forces R7's check on
            # all of them); a seed-specific, absent or differing init refuses.
            init_block = check_span_head_inits(
                args.ft_ledger, args.ft_row_ids,
                [int(Checkpoint.read_body(p)["seed"]) for p in args.checkpoints], 0,
                v6_run=True,
            )
        if args.source == "lora-merged":
            if args.norm_preserving or args.same_seed_trajectory:
                raise AverageRefusal(
                    "--from lora-merged is a mean of merged weights: no --norm-preserving "
                    "(there are no fp32 masters) and no --same-seed-trajectory"
                )
            merged = average_lora_merged(args.checkpoints, names)
            tensors, inputs, facts = merged.tensors, merged.inputs, merged.facts
            sources, master_index, lora_block = merged.sources, {}, merged.block
        elif args.norm_preserving:
            made = norm_preserving_average(
                args.checkpoints, names, base=args.base_snapshot,
                lambda_not_derived_from_c=args.lambda_not_derived_from_c,
            )
            done, norm_block = made.average, made.block
            tensors, inputs, facts = done.tensors, done.inputs, done.facts
            sources, master_index = done.sources, done.master_index
        elif args.source == "masters":
            done = average_masters(args.checkpoints, names)
            tensors, inputs, facts = done.tensors, done.inputs, done.facts
            sources, master_index = done.sources, done.master_index
        else:
            ckpts = [Checkpoint.read(p) for p in args.checkpoints]
            if args.same_seed_trajectory:
                steps = check_trajectory(ckpts, names)
                tensors = _mean_trees(ckpts, names)
                trajectory = trajectory_block(
                    ckpts, steps, args.ft_row_ids[0] if args.ft_row_ids else None
                )
            else:
                tensors = average(ckpts, names)
            inputs = [
                InputRecord(path=p, seed=c.seed, payload_digest=c.to_json()["payload_digest"])
                for p, c in zip(args.checkpoints, ckpts, strict=True)
            ]
            # The last input's: a trajectory average is recorded at its latest step, which
            # for the seeds-at-one-step average is every input's.
            facts = _facts(
                ckpts[-1].model_state, schedule=ckpts[-1].schedule.to_json(),
                optimizer_step=ckpts[-1].optimizer_step, where=names[-1],
            )
            sources, master_index = {k: "tower" for k in tensors}, {}
        if head_at is None:
            head_source = span_head_source_block("averaged")
            if init_block is not None:
                head_source["init"] = {**init_block, "proven_across": list(args.ft_row_ids)}
        else:
            taken = take_span_head(tensors, sources, args.checkpoints[head_at], names[head_at])
            for key in taken:
                master_index.pop(key, None)
            head_source = span_head_source_block(
                "seed", seed=args.span_head_seed, record=inputs[head_at]
            )
            head_source["init"] = init_block
            if norm_block is not None:
                norm_block["span_head"] = (
                    f"taken from the input of seed {args.span_head_seed} (span_head_source), "
                    "not averaged"
                )
        manifest = write(
            tensors, args.out, source=args.source, inputs=inputs,
            optimizer_step=facts.optimizer_step, schedule=facts.schedule,
            scalars=facts.scalars, ft_row_ids=args.ft_row_ids, tensor_sources=sources,
            master_index=master_index, norm_preserving=norm_block, trajectory=trajectory,
            prompt_format=facts.prompt_format, span_head_source=head_source, lora=lora_block,
        )
    except AverageRefusal as exc:
        raise SystemExit(f"refused: {exc}") from exc
    from_masters = sum(1 for s in sources.values() if s == "master")
    print(
        f"averaged {len(inputs)} checkpoints at step {facts.optimizer_step} from "
        f"{args.source} ({from_masters} of {len(sources)} tensors from a master) -> {args.out}"
    )
    if norm_block is not None:
        print(
            f"norm-preserving: c = {norm_block['c']:.6f} (pairs {norm_block['pairwise_cosine']}),"
            f" lambda from c = {norm_block['lambda_from_c']:.6f}, applied lambda = "
            f"{norm_block['lambda']:.6f}"
            + ("" if norm_block["lambda_derived_from_c"] else " -- NOT DERIVED FROM c")
        )
    if trajectory is not None:
        print(
            f"same-seed trajectory: seed {trajectory['seed']}, steps "
            f"{trajectory['optimizer_steps']}; real_ft_run.py --score-checkpoint refuses to "
            "score it (ckpt_average.TRAJECTORY_NOT_SCORABLE)"
        )
    print(f"manifest: {manifest} (weights only; not a resume point)")
    # ru_maxrss is bytes on macOS and KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2
    print(f"peak resident set: {peak_gib:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
