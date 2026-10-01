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
  ``span_head`` trees must match name for name, dtype for dtype and shape for shape. RSI's
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
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

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
SOURCES: Final[tuple[str, ...]] = ("tower", "masters")
#: Where a ``--optimizer master`` checkpoint keeps its fp32 masters in ``model_state``.
MASTERS_PATH: Final[tuple[str, str]] = ("optimizer", "masters")
#: The manifest is written beside the weights as ``<out>.manifest.json``.
MANIFEST_SUFFIX: Final[str] = ".manifest.json"
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
}


class AverageRefusal(ValueError):
    """The inputs cannot be averaged into one meaningful model."""


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


def _facts(
    model_state: Mapping[str, Any], *, schedule: Mapping[str, Any], optimizer_step: int,
    where: str,
) -> _Facts:
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
    if mine.schedule != first.schedule:
        raise AverageRefusal(
            f"{name} was trained on schedule {mine.schedule} and {first_name} "
            f"on {first.schedule}. Different schedules are different runs, not seeds of one."
        )
    if mine.optimizer_step != first.optimizer_step:
        raise AverageRefusal(
            f"{name} is at optimizer step {mine.optimizer_step} and {first_name} at "
            f"{first.optimizer_step}; an average across two points of training is "
            "neither of them"
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


def average(ckpts: Sequence[Checkpoint], names: Sequence[str]) -> dict[str, Any]:
    """``{"tower.<k>": tensor, "span_head.<k>": tensor}``, averaged in fp32, cast back."""
    import torch

    check_compatible(ckpts, names)
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
        optimizer = Checkpoint.read_body(path).get("model_state", {}).get("optimizer")
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
        from qd_train.run_control import _MAX_SIDECAR_HEADER_BYTES

        self.path = path
        self.body = Checkpoint.read_body(path)
        declared = self.body.get("sidecar")
        if not isinstance(declared, Mapping) or not isinstance(declared.get("digest"), str):
            raise AverageRefusal(f"{path} declares no tensor sidecar")
        self.sidecar = Checkpoint.sidecar_path(path, declared["digest"])
        if not self.sidecar.is_file():
            raise AverageRefusal(f"{path}: its sidecar {self.sidecar.name} is not beside it")
        with self.sidecar.open("rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if not 0 < n <= _MAX_SIDECAR_HEADER_BYTES:
                raise AverageRefusal(f"{self.sidecar}: safetensors header claims {n} bytes")
            self.header = json.loads(fh.read(n))
        self.base = 8 + n

    def master(self, index: int) -> Any:
        """``optimizer.masters[index]`` as a float32 torch tensor, verified."""
        from qd_train.run_control import _TENSOR_REF_TAG, _join_tensors

        masters = self.body["model_state"][MASTERS_PATH[0]][MASTERS_PATH[1]]
        node = masters[index]
        entry = node.get(_TENSOR_REF_TAG) if isinstance(node, Mapping) else None
        if entry is None:
            raise AverageRefusal(f"{self.path}: master {index} is not a tensor in the body")
        key = str(entry["key"])
        info = self.header.get(key)
        if info is None:
            raise AverageRefusal(f"{self.sidecar.name} does not carry {key}")
        start, end = (int(x) for x in info["data_offsets"])
        with self.sidecar.open("rb") as fh:
            fh.seek(self.base + start)
            data = fh.read(end - start)
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
) -> Path:
    """The safetensors file and ``<out>.manifest.json``, each written atomically.

    Refuses to overwrite either: an average is an artifact other rows will name.
    ``norm_preserving`` is :func:`norm_preserving_average`'s block, recorded under that key
    (and only then: a plain average's manifest is what it always was).
    """
    from safetensors.torch import save

    from qd_train.run_control import _atomic_write_bytes

    if source not in SOURCES:
        raise AverageRefusal(f"source {source!r} is not one of {SOURCES}")
    if norm_preserving is not None and source != "masters":
        raise AverageRefusal("a norm-preserving average is made from the fp32 masters")
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
        "accumulator": "float64" if source == "masters" else "float32",
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
            "none, and cast to the tower dtype once"
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
    args = parser.parse_args(argv)
    resolved = [p.resolve() for p in args.checkpoints]
    if len(set(resolved)) != len(resolved):
        raise SystemExit("the same checkpoint was named twice; it would be weighted double")
    if args.ft_row_ids and len(args.ft_row_ids) != len(args.checkpoints):
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
    names = [str(p) for p in args.checkpoints]
    norm_block: dict[str, Any] | None = None
    try:
        _refuse_existing(args.out)
        if args.norm_preserving:
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
            tensors = average(ckpts, names)
            inputs = [
                InputRecord(path=p, seed=c.seed, payload_digest=c.to_json()["payload_digest"])
                for p, c in zip(args.checkpoints, ckpts, strict=True)
            ]
            facts = _facts(
                ckpts[0].model_state, schedule=ckpts[0].schedule.to_json(),
                optimizer_step=ckpts[0].optimizer_step, where=names[0],
            )
            sources, master_index = {k: "tower" for k in tensors}, {}
        manifest = write(
            tensors, args.out, source=args.source, inputs=inputs,
            optimizer_step=facts.optimizer_step, schedule=facts.schedule,
            scalars=facts.scalars, ft_row_ids=args.ft_row_ids, tensor_sources=sources,
            master_index=master_index, norm_preserving=norm_block,
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
    print(f"manifest: {manifest} (weights only; not a resume point)")
    # ru_maxrss is bytes on macOS and KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2
    print(f"peak resident set: {peak_gib:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
