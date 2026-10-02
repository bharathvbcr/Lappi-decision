"""Rung (b) reference: the Lappi fine-tune recipe on a tiny Qwen3.5 tower, run by PyTorch on CPU.

    PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/qd_train_oracle_tiny.py \\
        --out crates/qd-train/tests/fixtures/tiny-published

A **reference oracle** (user policy: Python only as an oracle), not a trainer. It writes the
fixture the Rust trainer over canonical tessl must match at rung (b) of Fable's parity ladder
(``AUDIT/ojas-training-2026-10-01/fable-advice.md`` Q3, rows 9-12 and 15 of Q1).

What runs is the repository's own recipe code, not a restatement of it:

* the tower is ``qd_train.backbone.load_text_tower`` over a snapshot written in the real
  on-disk layout (``model.language_model.*`` beside a ``Qwen3_5Config`` with a
  ``text_config``), with gradient checkpointing on and ``attn_implementation="sdpa"``, as
  ``tools/real_ft_run._real_step`` builds it;
* the step is ``qd_train.backbone.QwenDecisionStep`` with ``train_attention_mask="none"``
  (tessl runs unpadded sequences), the layer-wise split of ``--lower-layers-n``, and the
  optimizer ``qd_train.optim.build_optimizer`` builds from ``real_ft_run.optimizer_spec``;
* the loop is ``qd_train.trainer.train_ft`` itself, under ``real_ft_run._control`` -- the
  owner of F's schedule (``warmup = max(1, steps // 20)``, cosine to ``lr / 10``) and of
  ``grad_accum = 1``;
* the batches come from ``qd_train.shards.assemble_batch``, the one place a ``Batch`` is put
  together from stored sequences, over synthetic sequences written in the shard set's own
  ragged form (``tokens.u32``, ``offsets.npy``, ``supervision.npz``).

The one thing added is a subclass whose ``apply`` records the step's learning rate, the
pre-clip global gradient norm and the clip coefficient before delegating to
``QwenDecisionStep.apply`` unchanged. ``clip_grad_norm_`` is documented as
``get_total_norm`` followed by ``clip_grads_with_norm_`` (torch/nn/utils/clip_grad.py), so
the norm recorded is the one the clip used.

A second subclass, ``IndependentStep``, keeps that optimizer, clip and schedule but restates
the forward and loss per unpadded sequence, as tessl runs them, with every target read off the
stored sequence. Its gap to the reference (manifest ``independent_check_fp32``, and per lr in
``discrimination``) is the floor between two correct torch implementations: the margin rung
(b)'s bars actually leave a third one, the Rust trainer.

**Amendment 2** (``AUDIT/ojas-training-2026-10-01/fable-rung-b-bars.md``) adds three things:

* **Arm ``fp32_clip150``.** It is F's recipe at ``max_grad_norm = 150``, inside the measured
  pre-clip range, so both clip branches run. It refuses to finish if either branch is empty in
  its own trajectory.
* **The measure in force** (``weights_gap``'s ``amendment2_*``). Every ``span_head.*`` tensor is
  measured over the span head's largest ``|w|``, and every tower tensor over its own max. The
  pre-amendment numbers stay beside it.
* **A kernel-diverse independent floor** (``independent_check_kernel_diverse``). It runs eager
  attention and the token-by-token recurrent GDN (``torch_recurrent_gated_delta_rule``,
  published rule), sharing no kernel with the reference.

The tower is tessl's tiny shape family (``tools/qwen35_ref/make_train_fixture.py`` in
canonical tessl: hidden 64, head_dim 256, GDN key/value heads of 128, grouped KV, tied
vocabulary of 64), **not** Lappi's head_dim-16 test tower, which tessl's kernels are not
compiled for. Every parameter is re-initialised at a scale where it matters and rounded to
bf16, so the one ``init.safetensors`` loads bit-identically into the fp32 and the bf16 tower.

CPU only, by construction: MPS bf16 logits were measured off by up to 0.83 (ojas
``parity-plan.md`` section 3 row 5), so nothing here touches MPS.

The GDN on the torch side is transformers' own fallback, ``torch_chunk_gated_delta_rule``
(no fla on this host), which reads the **decayed** state -- the ``published`` rule of
``AUDIT/gdn-reference-and-contracts.md`` section 2.2 (CLAUDE.md rule 9). The manifest names
it, with the line numbers it was read at.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

# The owners of F's schedule and of the optimizer layout a dtype/recipe pair gets. Private by
# name, imported anyway: a second copy of `warmup = max(1, steps // 20)` here is exactly the
# drift this fixture exists to catch.
import real_ft_run  # noqa: E402
import torch  # noqa: E402

from qd_train.artifacts import (  # noqa: E402
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SCORE,
    SLOT_SPAN,
    SPAN_ABSTAIN,
)
from qd_train.backbone import (  # noqa: E402
    TEXT_PREFIX,
    GradientCheckpointingDisabled,
    QwenDecisionStep,
    load_text_tower,
)
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.memory import ModelSpec  # noqa: E402
from qd_train.optim import DEFAULT_BETA2, LR_SCALE_KEY  # noqa: E402
from qd_train.shards import PAD_ID, assemble_batch  # noqa: E402
from qd_train.trainer import train_ft  # noqa: E402
from qd_train.tristate import NotRun  # noqa: E402

#: F's recipe (campaign/f-v4-preregistered.json recipe.flags): --optimizer master --lr 1e-5
#: --lower-layers-n 8 --lower-layers-lr-scale 0.1, seeds 0, 1, 2. The tiny tower has 4 layers,
#: so the split is 2 of 4 (F's is 8 of 24).
F_LR: Final[float] = 1e-5
DEFAULT_LOWER_LAYERS_N: Final[int] = 2
DEFAULT_STEPS: Final[int] = 20
DEFAULT_LAYERS: Final[int] = 4
DEFAULT_SEED: Final[int] = 0
#: Synthetic data and weight-init streams. Fixed, recorded in the manifest.
DATA_SEED: Final[int] = 20261001
INIT_SEED: Final[int] = 20260930
#: The tiny tower (tessl tools/qwen35_ref/make_train_fixture.py:48-70).
HIDDEN: Final[int] = 64
INTERMEDIATE: Final[int] = 128
VOCAB: Final[int] = 64
#: Bucket widths the synthetic batches are padded to (PAD_ID), as ShardReader pads to a
#: bucket's width. Lengths cross the GDN chunk (64) and two 32-row attention blocks.
BUCKETS: Final[tuple[int, ...]] = (64, 128, 160)
#: Token ids with a role in the synthetic rows. 0 is PAD_ID and never a real token.
NEWLINE_ID: Final[int] = 10
LETTER_IDS: Final[tuple[int, ...]] = tuple(range(40, 57))  # 17 letters, as the real alphabet
NOUL_ID: Final[int] = 57
#: Rung (b)'s pass criteria (fable-advice.md Q3 row b). Read-only here (rule 2): the
#: discrimination table says whether each counterfactual would be caught at these bars.
TOL_FINAL_WEIGHTS: Final[float] = 1e-5
TOL_LOSS_EARLY: Final[float] = 1e-5  # steps 0-5
TOL_LOSS_LATE: Final[float] = 1e-4  # to step 20
#: The fixture budget the lead set (~20 MB).
MAX_FIXTURE_BYTES: Final[int] = 20 * 1024 * 1024
#: Each rung (b) arm: its tower dtype and its clip norm. ``fp32_clip150`` is Amendment 2
#: (iii)'s second arm: F's recipe with ``max_grad_norm = 150``, inside the measured
#: pre-clip range (62-438), so the composed step runs both clip branches.
ARMS: Final[dict[str, tuple[str, float]]] = {
    "fp32": ("fp32", 1.0),
    "master_bf16": ("bf16", 1.0),
    "fp32_clip150": ("fp32", 150.0),
}
#: The span head's tensor names in every file this oracle writes.
SPAN_HEAD_PREFIX: Final[str] = "span_head."
#: The arms the second process regenerates for the repeat check: the two fp32 ones.
REPEAT_ARMS: Final[tuple[str, ...]] = ("fp32", "fp32_clip150")
MODELING: Final[str] = "transformers/models/qwen3_5/modeling_qwen3_5.py"


# --- the tiny tower ------------------------------------------------------------------------


def tiny_text_config(layers: int, hidden: int = HIDDEN) -> Any:
    """tessl's tiny Qwen3.5 (make_train_fixture.py:48-70): the 2B's layer pattern, three GDN
    layers then full attention, cut to ``layers``; two layers are one of each."""
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    if layers < 2:
        raise SystemExit(f"--layers must be at least 2 (one GDN, one attention), got {layers}")
    types = (
        ["linear_attention", "full_attention"]
        if layers == 2
        else ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)]
    )
    if "full_attention" not in types:
        raise SystemExit(f"--layers {layers} holds no attention layer; use 2 or at least 4")
    return Qwen3_5TextConfig(
        hidden_size=hidden,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=layers,
        layer_types=types,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=VOCAB,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )


def reinit(model: Any, gen: torch.Generator) -> None:
    """Every parameter at a scale where it matters, then rounded to bf16.

    Restated from canonical tessl ``tools/qwen35_ref/make_train_fixture.py:73-93`` (``reinit``)
    rather than imported from a tree with live foreign edits. The zero-centred norms would
    otherwise sit at 0 and hide a ``1 + w`` applied in the wrong place.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.25)
            elif name.endswith("conv1d.weight") or name.endswith("A_log"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5 - 1.0)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=gen))
            elif p.dim() == 1:
                p.copy_(0.3 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) / p.shape[1] ** 0.5)
            p.copy_(p.to(torch.bfloat16).float())


def write_snapshot(dirpath: Path, layers: int, hidden: int = HIDDEN) -> dict[str, torch.Tensor]:
    """The tiny tower in the real checkpoint layout, bf16. Returns its tensors by full name."""
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextModel

    text = tiny_text_config(layers, hidden)
    torch.manual_seed(0)
    model = Qwen3_5TextModel(text).float()
    reinit(model, torch.Generator().manual_seed(INIT_SEED))
    dirpath.mkdir(parents=True, exist_ok=True)
    Qwen3_5Config(text_config=text.to_dict()).save_pretrained(dirpath)
    tensors = {
        f"{TEXT_PREFIX}{k}": v.detach().to(torch.bfloat16).contiguous()
        for k, v in model.state_dict().items()
    }
    save_file(tensors, str(dirpath / "model.safetensors"))
    return tensors


def tiny_spec(layers: int, n_params: int, hidden: int = HIDDEN) -> ModelSpec:
    cfg = tiny_text_config(layers, hidden)
    n_full = sum(1 for t in cfg.layer_types if t == "full_attention")
    return ModelSpec(
        name=f"tiny Qwen3.5 text tower, tessl shape family, {layers} layers (rung b oracle)",
        hidden_size=hidden,
        intermediate_size=INTERMEDIATE,
        n_full_attention_layers=n_full,
        n_linear_attention_layers=layers - n_full,
        q_heads=2,
        kv_heads=1,
        head_dim=256,
        attn_output_gate=True,
        linear_heads=1,
        linear_head_dim=128,
        vocab_size=VOCAB,
        params_total=n_params,
        params_embedding=VOCAB * hidden,
        tied_embedding=True,
        recurrent_state_bytes=4,
    )


# --- the synthetic shard set -----------------------------------------------------------------


@dataclass(frozen=True)
class Sequence_:
    """One stored sequence and its supervision, as ``write_shards`` stores them."""

    tokens: np.ndarray
    kind: int
    span: tuple[int, int]
    candidates: np.ndarray


#: Row kinds per batch, cycled over the steps: letter-only, mixed, span-only (the
#: ``letter is None`` branch of ``accumulate_span``), mixed with an abstaining span, and one
#: long single row. ``ab`` is an abstaining span row.
BATCH_PATTERNS: Final[tuple[tuple[str, ...], ...]] = (
    ("choice", "score"),
    ("span", "choice"),
    ("span", "ab"),
    ("score", "choice", "span"),
    ("choice",),
    ("ab", "choice", "score", "span"),
)


def _sequence(rng: np.random.Generator, kind: str, length: int) -> Sequence_:
    body = rng.integers(1, VOCAB, size=length).astype(np.int64)
    body[body == NEWLINE_ID] = NEWLINE_ID + 1
    if kind in ("choice", "score"):
        body[-1] = int(rng.choice(LETTER_IDS))
        return Sequence_(
            body.astype(np.uint32),
            SLOT_CHOICE if kind == "choice" else SLOT_SCORE,
            (NO_SPAN, NO_SPAN),
            np.zeros(0, dtype=np.int32),
        )
    # A span row: line starts after newline tokens, inside [1, length-2); the query position is
    # length-2 (target_index), as write_shards stores it; the last token is the noul letter.
    n_lines = int(rng.integers(3, 13))
    cands = np.sort(rng.choice(np.arange(1, length - 2), size=n_lines, replace=False))
    for p in cands:
        body[p - 1] = NEWLINE_ID
    body[-1] = NOUL_ID
    if kind == "ab":
        span = (SPAN_ABSTAIN, SPAN_ABSTAIN)
    else:
        a, b = sorted(int(x) for x in rng.choice(cands, size=2, replace=True))
        span = (a, b)
    return Sequence_(body.astype(np.uint32), SLOT_SPAN, span, cands.astype(np.int32))


def synth_shard_set(steps: int) -> tuple[list[Sequence_], list[dict[str, Any]]]:
    """Sequences and a batch plan: one batch per optimizer step (grad_accum is 1 in F)."""
    rng = np.random.default_rng(DATA_SEED)
    seqs: list[Sequence_] = []
    plan: list[dict[str, Any]] = []
    for index in range(steps):
        pattern = BATCH_PATTERNS[index % len(BATCH_PATTERNS)]
        rows: list[int] = []
        for kind in pattern:
            length = (
                int(rng.integers(130, 161)) if len(pattern) == 1 else int(rng.integers(24, 150))
            )
            rows.append(len(seqs))
            seqs.append(_sequence(rng, kind, length))
        widest = max(int(seqs[r].tokens.size) for r in rows)
        bucket = next(i for i, w in enumerate(BUCKETS) if w >= widest)
        plan.append({"index": index, "bucket": bucket, "width": BUCKETS[bucket], "rows": rows})
    return seqs, plan


def write_shard_form(out: Path, seqs: Sequence[Sequence_], plan: list[dict[str, Any]]) -> None:
    """``tokens.u32``, ``offsets.npy`` and ``supervision.npz`` exactly as
    ``qd_train.shards.write_shards`` lays them out (shards.py:14-31, 1815-1829), plus
    ``plan.json``: the batch order, one batch per optimizer step."""
    out.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate([s.tokens for s in seqs]).astype("<u4")
    (out / "tokens.u32").write_bytes(flat.tobytes())
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum([s.tokens.size for s in seqs], out=offsets[1:])
    np.save(out / "offsets.npy", offsets)
    cand_offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum([s.candidates.size for s in seqs], out=cand_offsets[1:])
    np.savez(
        out / "supervision.npz",
        slot_kind=np.asarray([s.kind for s in seqs], dtype=np.uint8),
        target_index=np.asarray([s.tokens.size - 2 for s in seqs], dtype=np.int32),
        span_target=np.asarray([s.span for s in seqs], dtype=np.int32).reshape(len(seqs), 2),
        candidate_offsets=cand_offsets,
        candidate_positions=np.concatenate([s.candidates for s in seqs]).astype(np.int32),
    )
    described = []
    for b, batch in zip(plan, torch_batches(seqs, plan), strict=True):
        kinds = [seqs[r].kind for r in b["rows"]]
        spans = [r for r in b["rows"] if seqs[r].kind == SLOT_SPAN]
        described.append(
            {
                **b,
                "span_plan": span_plan(batch),
                "lengths": [int(seqs[r].tokens.size) for r in b["rows"]],
                "slot_kinds": kinds,
                # The two denominators of the step's loss (backbone.py:1124-1185, heads.py:385-406):
                # the letter CE is a mean over n_supervised rows, the span CE a mean over 2*n_spans
                # pointer decisions; total = letter + span_weight * span.
                "n_supervised": sum(1 for k in kinds if k != SLOT_SPAN),
                "n_spans": len(spans),
                "n_abstaining": sum(
                    1 for r in spans if seqs[r].span == (SPAN_ABSTAIN, SPAN_ABSTAIN)
                ),
            }
        )
    (out / "plan.json").write_text(
        json.dumps(
            {
                "pad_id": PAD_ID,
                "buckets": list(BUCKETS),
                "grad_accum": 1,
                "batches": described,
            },
            indent=1,
        )
        + "\n"
    )


def span_plan(batch: Any) -> list[dict[str, Any]] | None:
    """The span head's row layout for this batch, from the owners: ``trainer.ft_supervision``
    then ``heads.plan_span_batch``. Rows ``0..n_candidates-1`` are line starts in ascending
    token order, row ``n_candidates`` is the abstention (qd-runtime answer.rs ``noul_row``)."""
    from qd_train.heads import plan_span_batch
    from qd_train.trainer import ft_supervision

    span = ft_supervision(batch).span
    if span is None:
        return None
    p = plan_span_batch(span)
    out = []
    for k in range(p.n_spans):
        n = int(p.n_candidates[k])
        out.append(
            {
                "batch_row": int(span.rows[k]),
                "query_index": int(p.query_index[k]),
                "candidate_pos": [int(x) for x in p.candidate_pos[k, :n]],
                "gold_start_row": int(p.gold_start[k]),
                "gold_end_row": int(p.gold_end[k]),
                "abstaining": bool(p.abstaining[k]),
                "runtime_rows": int(p.runtime_rows[k]),
            }
        )
    return out


def torch_batches(seqs: Sequence[Sequence_], plan: list[dict[str, Any]]) -> Iterator[Any]:
    for b in plan:
        rows = b["rows"]
        yield assemble_batch(
            [seqs[r].tokens.astype(np.int32) for r in rows],
            kinds=np.asarray([seqs[r].kind for r in rows], dtype=np.uint8),
            target_index=np.asarray([seqs[r].tokens.size - 2 for r in rows], dtype=np.int32),
            spans=np.asarray([seqs[r].span for r in rows], dtype=np.int32).reshape(len(rows), 2),
            candidates=[seqs[r].candidates for r in rows],
            width=b["width"],
            bucket=b["bucket"],
            index=b["index"],
        )


# --- the step, recorded --------------------------------------------------------------------


#: A counterfactual's edit of the optimizer after QwenDecisionStep built it, and of the lr the
#: loop hands `apply`. The reference arm uses neither.
OptimizerEdit = Callable[["RecordingStep"], None]
LrEdit = Callable[[int, float], float]


class RecordingStep(QwenDecisionStep):
    """``QwenDecisionStep`` whose ``apply`` first records what the clip and the schedule saw."""

    def __init__(
        self,
        *args: Any,
        lr_edit: LrEdit | None = None,
        keep_grads_at: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, Any]] = []
        self.lr_edit = lr_edit
        self.keep_grads_at = keep_grads_at
        self.kept_grads: dict[str, torch.Tensor] = {}

    def named_trainables(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [
            *((f"{TEXT_PREFIX}{n}", p) for n, p in self.tower.model.named_parameters()),
            *((f"span_head.{n}", p) for n, p in self.span_head.named_parameters()),
        ]

    def apply(self, *, lr: float) -> None:
        index = len(self.records)
        if self.lr_edit is not None:
            lr = self.lr_edit(index, lr)
        params = self.parameters()  # exactly the list QwenDecisionStep.apply clips
        grads = [p.grad for p in params if p.grad is not None]
        norm = torch.nn.utils.get_total_norm(grads)
        # clip_grads_with_norm_'s own arithmetic (clip_grad.py): max_norm / (norm + 1e-6),
        # clamped at 1, as a tensor in the norm's dtype.
        coef = torch.clamp(self.max_grad_norm / (norm + 1e-6), max=1.0)
        self.records.append(
            {
                "lr": float(lr),
                "grad_norm_preclip": float(norm),
                "clip_coef": float(coef),
                "clip_active": bool(float(coef) < 1.0),
                "grad_dtypes": sorted({str(g.dtype).removeprefix("torch.") for g in grads}),
            }
        )
        if self.keep_grads_at == index:
            self.kept_grads = {
                n: p.grad.detach().to(torch.float32).clone()
                for n, p in self.named_trainables()
                if p.grad is not None
            }
        super().apply(lr=lr)

    def weights_fp32(self) -> dict[str, torch.Tensor]:
        """The weights tessl's f32 model is compared against: the live parameter where it is
        f32, the optimizer's fp32 **master** where the live one is bf16
        (``live == RNE(master)``)."""
        opt = self.optimizer
        masters = getattr(opt, "_masters", None)
        master_of: dict[int, torch.Tensor] = {}
        if masters is not None:
            master_of = {id(live): m for live, m in zip(opt._live, masters, strict=True)}
        return {
            n: master_of.get(id(p), p).detach().to(torch.float32).clone()
            for n, p in self.named_trainables()
        }

    def param_recipe(self) -> dict[str, dict[str, Any]]:
        """Per parameter: its group's name, lr_scale, weight_decay, eps, betas -- read off the
        optimizer that steps (the inner AdamW over the masters, under the master recipe)."""
        opt = self.optimizer
        masters = getattr(opt, "_masters", None)
        stepped: dict[int, int] = {}
        if masters is not None:
            stepped = {id(live): id(m) for live, m in zip(opt._live, masters, strict=True)}
        group_of: dict[int, dict[str, Any]] = {}
        state_of: dict[int, Any] = {}
        for g in opt.param_groups:
            for p in g["params"]:
                group_of[id(p)] = g
                state_of[id(p)] = opt.state.get(p, {})
        out: dict[str, dict[str, Any]] = {}
        for n, p in self.named_trainables():
            key = stepped.get(id(p), id(p))
            g = group_of[key]
            taken = state_of[key].get("step")
            out[n] = {
                # torch's per-parameter state["step"] after the run: AdamW skips a parameter
                # whose .grad is None, so the span head does not step on a letter-only batch.
                "adamw_steps_taken": 0 if taken is None else int(taken),
                "shape": list(p.shape),
                "group": g.get("name", "default"),
                "lr_scale": float(g.get(LR_SCALE_KEY, 1.0)),
                "weight_decay": float(g["weight_decay"]),
                "eps": float(g["eps"]),
                "betas": [float(b) for b in g["betas"]],
                "tessl_default_excludes": tessl_excluded_from_weight_decay(
                    n.removeprefix(TEXT_PREFIX)
                ),
            }
        return out


class IndependentStep(RecordingStep):
    """The reference step with its forward and loss restated the way tessl runs them: one
    unpadded sequence at a time through the tower, the tied head's logits and torch's
    cross-entropy written out for the letter, the pointer scores written out for the span head,
    and every target read off the stored sequence (query ``length - 2``, candidates in stored
    order, abstention the row after the last candidate) rather than off ``ft_supervision`` and
    ``plan_span_batch``. The optimizer, clip and schedule are the reference's own (``apply`` is
    inherited), so the gap this arm leaves to the reference is the reduction-order floor
    between two correct torch implementations of rung (b)'s math -- the floor the Rust trainer
    is held to as well."""

    def __init__(
        self, *args: Any, seqs: Sequence[Sequence_], plan: list[dict[str, Any]], **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._seqs = seqs
        self._rows_of = {int(b["index"]): list(b["rows"]) for b in plan}

    def accumulate(self, batch: Any, supervision: Any) -> float:
        return self._unpadded(batch)

    def accumulate_span(self, batch: Any, supervision: Any) -> float:
        return self._unpadded(batch)

    def _unpadded(self, batch: Any) -> float:
        ce = torch.nn.functional.cross_entropy
        rows = [self._seqs[r] for r in self._rows_of[int(batch.index)]]
        n_letter = sum(1 for s in rows if s.kind != SLOT_SPAN)
        n_span = len(rows) - n_letter
        head = self.span_head
        letter = torch.zeros((), dtype=torch.float32)
        span = torch.zeros((), dtype=torch.float32)
        with self.training_attention():
            for s in rows:
                ids = torch.as_tensor(s.tokens.astype(np.int64))[None]
                h = self.tower.model(input_ids=ids, attention_mask=None).last_hidden_state[0]
                q = int(s.tokens.size) - 2
                if s.kind != SLOT_SPAN:
                    logits = (h[q] @ self.tower.lm_head_weight.T).float()
                    letter = letter + ce(logits[None], ids[0, q + 1 : q + 2], reduction="sum")
                    continue
                cands = [int(c) for c in s.candidates]
                gold = [len(cands) if t == SPAN_ABSTAIN else cands.index(t) for t in s.span]
                hf = h.float()
                for proj, abstain, g in (
                    (head.start_proj, head.abstain_start, gold[0]),
                    (head.end_proj, head.abstain_end, gold[1]),
                ):
                    qq = proj(hf[q])
                    scores = torch.cat([hf[torch.as_tensor(cands)] @ qq, (qq @ abstain)[None]])
                    span = span + ce(scores[None], torch.tensor([g]), reduction="sum")
            letter_mean = letter / n_letter if n_letter else None
            span_mean = span / (2 * n_span) if n_span else None
            parts = [x for x in (letter_mean,) if x is not None]
            if span_mean is not None:
                parts.append(self.span_weight * span_mean)
            total = parts[0] if len(parts) == 1 else parts[0] + parts[1]
            total.backward()
        self.letter_log.append(0.0 if letter_mean is None else float(letter_mean.detach()))
        self.span_log.append(0.0 if span_mean is None else float(span_mean.detach()))
        return float(total.detach())


def tessl_excluded_from_weight_decay(name: str) -> bool:
    """canonical tessl ``src/qwen35_adamw.rs`` ``excluded_from_weight_decay``, restated for the
    counterfactual arm and the per-parameter table only."""
    if "bias" in name or "layernorm" in name or "rmsnorm" in name:
        return True
    return any(seg == "norm" or seg.endswith("_norm") for seg in name.split("."))


# --- one arm ---------------------------------------------------------------------------------


@dataclass
class ArmResult:
    records: list[dict[str, Any]]
    losses: list[float]
    letter: list[float]
    span: list[float]
    loss_log_digest: str
    termination: str
    final: dict[str, torch.Tensor]
    init_span_head: dict[str, torch.Tensor]
    grads0: dict[str, torch.Tensor]
    param_recipe: dict[str, dict[str, Any]]
    kernels: dict[str, str]
    attn_implementation: str
    optimizer: dict[str, Any]
    schedule: dict[str, Any]
    max_grad_norm: float
    seconds: float


#: The GDN kernels an arm may run: transformers' chunked form (every reference arm) and its
#: token-by-token recurrent form (the kernel-diverse independent arm). Both read the
#: decayed state, the published rule (gdn_seam names the lines).
GDN_KERNELS: Final[frozenset[str]] = frozenset({"chunk", "recurrent"})


def use_recurrent_gdn(model: Any) -> int:
    """Route every GDN layer's training forward through ``torch_recurrent_gated_delta_rule``
    instead of the chunked kernel. Returns how many layers were rerouted; refuses none.

    The forward calls ``self.chunk_gated_delta_rule(..., cu_seqlens=...)``
    (modeling_qwen3_5.py); the recurrent form takes no ``cu_seqlens``, so the adapter refuses
    a packed call rather than dropping it, and otherwise passes every argument through.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    def recurrent(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            raise SystemExit("packed sequences reached the recurrent GDN; this oracle runs none")
        return torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    n = 0
    for module in model.modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            module.chunk_gated_delta_rule = recurrent
            n += 1
    if n == 0:
        raise SystemExit("no GDN layer found to reroute to the recurrent kernel")
    return n


def make_step(
    *,
    snapshot: Path,
    layers: int,
    dtype: str,
    plan: list[dict[str, Any]],
    steps: int,
    lr: float,
    seed: int,
    lower_layers_n: int,
    lower_lr_scale: float,
    max_grad_norm: float = 1.0,
    lr_edit: LrEdit | None = None,
    keep_grads: bool = False,
    seqs: Sequence[Sequence_] | None = None,
    independent: bool = False,
    hidden: int = HIDDEN,
    attn_implementation: str = "sdpa",
    gdn: str = "chunk",
) -> tuple[Any, RecordingStep]:
    """The tower and the step exactly as ``real_ft_run._real_step`` builds them (gradient
    checkpointing on, sdpa, the dtype's optimizer spec under the ``master`` recipe), on CPU.

    ``attn_implementation="eager"`` and ``gdn="recurrent"`` are the kernel-diverse
    independent arm's only (Amendment 2 (i)'s recommended measurement): every reference arm
    runs sdpa and the chunked GDN."""
    if gdn not in GDN_KERNELS:
        raise SystemExit(f"gdn must be one of {sorted(GDN_KERNELS)}, got {gdn!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GradientCheckpointingDisabled)
        tower = load_text_tower(
            snapshot,
            gradient_checkpointing=True,
            optimizer=real_ft_run.optimizer_spec(dtype, "master"),
            attn_implementation=attn_implementation,
            device="cpu",
            dtype=dtype,
            rows=max(len(b["rows"]) for b in plan),
            width=max(b["width"] for b in plan),
            spec=tiny_spec(layers, _snapshot_params(snapshot), hidden),
        )
    if gdn == "recurrent":
        use_recurrent_gdn(tower.model)
    extra: dict[str, Any] = {}
    if independent:
        if seqs is None:
            raise SystemExit("the independent step reads its targets off the stored sequences")
        extra = {"seqs": seqs, "plan": plan}
    step = (IndependentStep if independent else RecordingStep)(
        tower,
        **extra,
        seed=seed,
        lr=lr,
        total_steps=steps,
        span_weight=1.0,
        max_grad_norm=max_grad_norm,
        max_width=max(BUCKETS),
        lower_layers_n=lower_layers_n,
        lower_lr_scale=lower_lr_scale if lower_layers_n else 1.0,
        beta2=DEFAULT_BETA2,
        train_attention_mask="none",
        lr_edit=lr_edit,
        keep_grads_at=0 if keep_grads else None,
    )
    if any(p.device.type != "cpu" for p in step.parameters()):
        raise SystemExit("a parameter left the CPU; this oracle runs on CPU only")
    return tower, step


def run_arm(
    *,
    snapshot: Path,
    layers: int,
    dtype: str,
    seqs: Sequence[Sequence_],
    plan: list[dict[str, Any]],
    steps: int,
    lr: float,
    seed: int,
    lower_layers_n: int,
    lower_lr_scale: float,
    span_head_init: Mapping[str, torch.Tensor] | None,
    scratch: Path,
    keep_grads: bool = False,
    max_grad_norm: float = 1.0,
    optimizer_edit: OptimizerEdit | None = None,
    lr_edit: LrEdit | None = None,
    independent: bool = False,
    attn_implementation: str = "sdpa",
    gdn: str = "chunk",
) -> ArmResult:
    from torch.optim.optimizer import _default_to_fused_or_foreach

    t0 = time.monotonic()
    tower, step = make_step(
        snapshot=snapshot,
        layers=layers,
        dtype=dtype,
        plan=plan,
        steps=steps,
        lr=lr,
        seed=seed,
        lower_layers_n=lower_layers_n,
        lower_lr_scale=lower_lr_scale,
        max_grad_norm=max_grad_norm,
        lr_edit=lr_edit,
        keep_grads=keep_grads,
        seqs=seqs,
        independent=independent,
        attn_implementation=attn_implementation,
        gdn=gdn,
    )
    init_span_head = {
        f"span_head.{n}": p.detach().clone() for n, p in step.span_head.named_parameters()
    }
    if span_head_init is not None:
        for n, t in init_span_head.items():
            if not torch.equal(t, span_head_init[n]):
                raise SystemExit(
                    f"{n}: the span head this arm built at seed {seed} differs from the one in "
                    "init.safetensors; the arms would not start from one init"
                )
    if optimizer_edit is not None:
        optimizer_edit(step)

    control = real_ft_run._control(steps, device="cpu", lr=lr)
    inner = getattr(step.optimizer, "_inner", step.optimizer)
    fused, foreach = _default_to_fused_or_foreach(
        [p for g in inner.param_groups for p in g["params"]],
        differentiable=False,
        use_fused=False,
    )
    ledger_dir = Path(tempfile.mkdtemp(dir=scratch))
    recorder = RunRecorder(
        Ledger(ledger_dir / "ledger.jsonl"),
        protocol=Protocol(
            data_snapshot_hash=f"synthetic:{DATA_SEED}",
            # Not the build marker (ledger.NOT_APPLICABLE): an ft row refuses it. The ids are
            # synthetic and no tokenizer exists, which is what this string says.
            tokenizer_hash="synthetic-ids:no-tokenizer",
            backbone_commit=f"synthetic-tiny:{INIT_SEED}",
            recipe_hash="oracle",
            seed=seed,
        ),
        run_kind="ft",
        repo=ROOT,
        # The row is scratch (a temp ledger train_ft requires), but it still names the sources
        # that produced it, as every recorder in tools/ does (test_provenance_on_every_exit).
        entry_point=Path(__file__),
        wall_clock_s=None,
        cost=None,
        env=Environment(
            torch=torch.__version__,
            transformers_sha="none",
            device="cpu",
            host=platform.node() or "unknown",
            fla_present=NotRun(reason="CPU oracle; fla is CUDA-only"),
            causal_conv1d_present=NotRun(reason="CPU oracle; causal-conv1d is CUDA-only"),
        ),
        quick=True,
        quick_reason="rung (b) fixture generator: tiny synthetic tower, 1 seed, 20 steps",
    )
    result = train_ft(
        torch_batches(seqs, plan),
        epoch=0,
        step=step,
        control=control,
        recorder=recorder,
    )
    final = step.weights_fp32()
    groups = inner.param_groups
    return ArmResult(
        records=step.records,
        losses=[float(x) for x in result.loss_log.losses()],
        letter=[float(x) for x in step.letter_log],
        span=[float(x) for x in step.span_log],
        loss_log_digest=result.loss_log.digest(),
        termination=result.termination,
        final=final,
        init_span_head=init_span_head,
        grads0=step.kept_grads,
        param_recipe=step.param_recipe(),
        kernels=dict(tower.linear_attention_kernels),
        attn_implementation=tower.attn_implementation,
        optimizer={
            "class": f"{type(step.optimizer).__module__}.{type(step.optimizer).__qualname__}",
            "inner": f"{type(inner).__module__}.{type(inner).__qualname__}",
            "groups": [
                {
                    k: (v if not isinstance(v, tuple) else list(v))
                    for k, v in g.items()
                    if k != "params"
                }
                | {"n_tensors": len(g["params"])}
                for g in groups
            ],
            "torch_dispatch_on_this_host": {"fused": bool(fused), "foreach": bool(foreach)},
        },
        schedule={
            "peak_lr": control.schedule.peak_lr,
            "total_steps": control.schedule.total_steps,
            "warmup_steps": control.schedule.warmup_steps,
            "min_lr": control.schedule.min_lr,
            "grad_accum": control.grad_accum,
        },
        max_grad_norm=step.max_grad_norm,
        seconds=time.monotonic() - t0,
    )


def _snapshot_params(snapshot: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
        names = list(f.keys())  # a safe_open handle is not a mapping; keys() is its API
        return sum(int(np.prod(f.get_slice(k).get_shape())) for k in names)


# --- comparing weights and trajectories --------------------------------------------------------


def weights_gap(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """max over parameters of max|a - b| / max|b| -- the rung's final-weights measure -- and the
    same over the tower alone. The span head's two abstain vectors start at zero
    (heads.py:311-314), so their max is their own trained displacement and they dominate the
    all-parameter number; the tower-only number says what the tower itself distinguishes."""
    worst: dict[str, tuple[float, str]] = {"all": (0.0, ""), "tower": (0.0, "")}
    for n, ref in b.items():
        scale = float(ref.abs().max())
        gap = float((a[n] - ref).abs().max()) / (scale if scale > 0 else 1.0)
        for scope in ("all", "tower") if n.startswith(TEXT_PREFIX) else ("all",):
            if gap > worst[scope][0]:
                worst[scope] = (gap, n)
    # Amendment 2 (i): every span_head.* tensor over the span head's largest |w|, every tower
    # tensor over its own max. The pre-amendment numbers above stay, for the record.
    head = [n for n in b if n.startswith(SPAN_HEAD_PREFIX)]
    head_scale = max((float(b[n].abs().max()) for n in head), default=0.0)
    amended = (0.0, "")
    for n, ref in b.items():
        scale = head_scale if n.startswith(SPAN_HEAD_PREFIX) else float(ref.abs().max())
        gap = float((a[n] - ref).abs().max()) / (scale if scale > 0 else 1.0)
        if gap > amended[0]:
            amended = (gap, n)
    return {
        "max_rel_to_param_max": worst["all"][0],
        "param": worst["all"][1],
        "tower_max_rel_to_param_max": worst["tower"][0],
        "tower_param": worst["tower"][1],
        "amendment2_max_rel": amended[0],
        "amendment2_param": amended[1],
        "amendment2_span_head_scale": head_scale,
    }


def loss_gap(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    rel = [abs(x - y) / max(abs(y), 1e-30) for x, y in zip(a, b, strict=True)]
    return {"steps_0_5_max_rel": max(rel[:6]), "all_steps_max_rel": max(rel)}


def digest_floats(values: Sequence[float]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def digest_tensors(tensors: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for n in sorted(tensors):
        t = tensors[n].detach().contiguous()
        h.update(n.encode())
        h.update(str(t.dtype).encode())
        h.update(json.dumps(list(t.shape)).encode())
        h.update(
            t.view(torch.uint8).numpy().tobytes()
            if t.dtype == torch.bfloat16
            else t.numpy().tobytes()
        )
    return h.hexdigest()


def trajectory(
    arm: ArmResult, plan: Sequence[Mapping[str, Any]], seqs: Sequence[Sequence_]
) -> list[dict[str, Any]]:
    rows = []
    for i, rec in enumerate(arm.records):
        kinds = [seqs[r].kind for r in plan[i]["rows"]]
        rows.append(
            {
                "optimizer_step": i,
                "batch_index": plan[i]["index"],
                "lr": rec["lr"],
                "loss_total": arm.losses[i],
                # 0.0 with letter_present false is the span-only batch (accumulate_span's
                # `letter is None`), not a measured zero; likewise loss_span without span rows.
                "letter_present": any(k != SLOT_SPAN for k in kinds),
                "span_present": any(k == SLOT_SPAN for k in kinds),
                "loss_letter": arm.letter[i],
                "loss_span": arm.span[i],
                "grad_norm_preclip": rec["grad_norm_preclip"],
                "clip_coef": rec["clip_coef"],
                "clip_active": rec["clip_active"],
                "grad_dtypes": rec["grad_dtypes"],
            }
        )
    return rows


def arm_digests(arm: ArmResult) -> dict[str, str]:
    return {
        "losses_f64": digest_floats(arm.losses),
        "letter_f64": digest_floats(arm.letter),
        "span_f64": digest_floats(arm.span),
        "grad_norms_f64": digest_floats([r["grad_norm_preclip"] for r in arm.records]),
        "lrs_f64": digest_floats([r["lr"] for r in arm.records]),
        "final_weights": digest_tensors(arm.final),
        "loss_log_digest": arm.loss_log_digest,
    }


# --- counterfactuals: what this fixture can and cannot catch -----------------------------------


#: Per parameter (its full name, its current group) -> a value for the rebuilt optimizer.
PerParam = Callable[[str, dict[str, Any]], float]


def _group_wd(_name: str, g: dict[str, Any]) -> float:
    return float(g["weight_decay"])


def _group_scale(_name: str, g: dict[str, Any]) -> float:
    return float(g.get(LR_SCALE_KEY, 1.0))


def _rebuild_groups(
    step: RecordingStep, *, wd_of: PerParam = _group_wd, scale_of: PerParam = _group_scale
) -> None:
    """Replace the fp32 arm's AdamW with one whose groups split by per-parameter weight decay
    and lr scale. Counterfactual only; betas and eps are read off the optimizer it replaces."""
    old = step.optimizer
    if hasattr(old, "_masters"):
        raise SystemExit("counterfactuals run on the fp32 arm only")
    name_of = {id(p): n for n, p in step.named_trainables()}
    groups: dict[tuple[float, float, str], list[Any]] = {}
    for g in old.param_groups:
        for p in g["params"]:
            name = name_of[id(p)]
            key = (scale_of(name, g), wd_of(name, g), str(g.get("name", "default")))
            groups.setdefault(key, []).append(p)
    ref = old.param_groups[0]
    peak = ref["lr"] / float(ref.get(LR_SCALE_KEY, 1.0))
    step.optimizer = torch.optim.AdamW(
        [
            {"params": ps, LR_SCALE_KEY: s, "weight_decay": wd, "name": nm, "lr": peak * s}
            for (s, wd, nm), ps in groups.items()
        ],
        lr=peak,
        betas=ref["betas"],
        eps=ref["eps"],
    )


def _tessl_wd(name: str, g: dict[str, Any]) -> float:
    return (
        0.0
        if tessl_excluded_from_weight_decay(name.removeprefix(TEXT_PREFIX))
        else _group_wd(name, g)
    )


def _decay_at_full_lr(name: str, g: dict[str, Any]) -> float:
    # torch decays by 1 - group_lr * wd, group_lr = lr * scale; wd / scale undoes the scale.
    return _group_wd(name, g) / _group_scale(name, g)


def counterfactuals() -> dict[str, dict[str, Any]]:
    return {
        "tessl_default_weight_decay": {
            "what": "weight decay 0 on tessl's excluded_from_weight_decay names (norms, dt_bias), "
            "0.01 elsewhere; F decays every parameter",
            "optimizer_edit": lambda s: _rebuild_groups(s, wd_of=_tessl_wd),
        },
        "no_weight_decay": {
            "what": "weight decay 0 everywhere (a port that dropped it)",
            "optimizer_edit": lambda s: _rebuild_groups(s, wd_of=lambda n, g: 0.0),
        },
        "decay_not_lr_scaled": {
            "what": "the lower group's decay at the full lr (lr_scale applied to the step only); "
            "torch's decoupled decay is 1 - group_lr * wd",
            "optimizer_edit": lambda s: _rebuild_groups(s, wd_of=_decay_at_full_lr),
        },
        "no_lr_scale": {
            "what": "every group at 1.0x (the trainer before tessl's per-entry lr_scale lands)",
            "optimizer_edit": lambda s: _rebuild_groups(s, scale_of=lambda n, g: 1.0),
        },
        "no_clip": {
            "what": "no gradient clipping (max_grad_norm effectively infinite)",
            "max_grad_norm": 1e30,
        },
        "cosine_off_by_one": {
            "what": "cosine progress evaluated one step late: lr_at(min(step + 1, total - 1)) "
            "after warmup",
            "lr_edit": "shift",
        },
    }


# --- the fixture ---------------------------------------------------------------------------------


def save_safetensors(path: Path, tensors: Mapping[str, torch.Tensor], about: str) -> None:
    """One metadata key, deliberately: safetensors serialises ``__metadata__`` from a hash map
    whose order changes per process (measured: identical tensors, different file bytes with
    four keys), and a committed fixture has to regenerate byte-identically."""
    from safetensors.torch import save_file

    save_file(
        {n: t.contiguous() for n, t in sorted(tensors.items())},
        str(path),
        metadata={"about": f"tools/qd_train_oracle_tiny.py, rule=published: {about}"},
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gdn_seam() -> dict[str, Any]:
    """What the torch side's GDN is, read off the installed transformers source by line."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    src = inspect.getsource(mq)
    lines = src.splitlines()

    def line_of(needle: str, *, within: str | None = None) -> int:
        """The one line holding ``needle`` -- inside function ``within`` when given."""
        lo, hi = 0, len(lines)
        if within is not None:
            lo = line_of(f"def {within}(") - 1
            hi = next(
                (i for i in range(lo + 1, len(lines)) if lines[i].startswith("def ")), len(lines)
            )
        hits = [i + 1 for i in range(lo, hi) if needle in lines[i]]
        if len(hits) != 1:
            raise SystemExit(f"{MODELING}: {needle!r} found {len(hits)} times, expected once")
        return hits[0]

    return {
        "function": f"{mq.__name__}.torch_chunk_gated_delta_rule",
        "rule": "published",
        "why_published": (
            "the chunk form reads the state decayed by the cumulative gate through position t "
            "(k_cumdecay = attn @ (k_beta * exp(g_cum)); v_prime = k_cumdecay @ S), and the "
            "recurrent form decays before the read (S = S * exp(g_t); kv_mem = S k_t): "
            "e_t = v_t - alpha_t S_{t-1} k_t, the decayed read of "
            "AUDIT/gdn-reference-and-contracts.md section 2.2, not nanolab's rule='repo'"
        ),
        "lines": {
            "torch_chunk_gated_delta_rule": line_of("def torch_chunk_gated_delta_rule("),
            "k_cumdecay": line_of("k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))"),
            "v_prime": line_of("v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state"),
            "recurrent_decay_before_read": line_of(
                "last_recurrent_state = last_recurrent_state * g_t"
            ),
            "fallback_binding": line_of(
                "self.chunk_gated_delta_rule = chunk_gated_delta_rule"
                " or torch_chunk_gated_delta_rule"
            ),
            "fp32_core_cast": line_of(
                "x.transpose(1, 2).contiguous().to(torch.float32)"
                " for x in (query, key, value, beta, g)",
                within="torch_chunk_gated_delta_rule",
            ),
            "l2norm_eps": line_of(
                "def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):"
            ),
        },
        "file": MODELING,
        "chunk_size": int(
            inspect.signature(mq.torch_chunk_gated_delta_rule).parameters["chunk_size"].default
        ),
        "use_qk_l2norm_in_kernel": True,
        "seam": "transformers' (l2norm of q/k in the kernel, g = -exp(A_log) * "
        "softplus(a + dt_bias) and beta = sigmoid(b) given), the same seam tessl's gdn_train runs",
        "core_dtype": "float32 in both arms (q, k, v, beta, g cast to f32 before the recurrence; "
        "the output cast back to the tower dtype)",
        "kernel_diverse_independent_arm": {
            "function": f"{mq.__name__}.torch_recurrent_gated_delta_rule",
            "rule": "published",
            "why_published": "token by token, the state is decayed before it is read "
            "(S = S * exp(g_t); kv_mem = S k_t; delta = (v_t - kv_mem) * beta_t): the decayed "
            "read of AUDIT/gdn-reference-and-contracts.md section 2.2",
            "lines": {
                "torch_recurrent_gated_delta_rule": line_of(
                    "def torch_recurrent_gated_delta_rule("
                ),
                "decay_before_read": line_of("last_recurrent_state = last_recurrent_state * g_t"),
                "kv_mem_read": line_of(
                    "kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)",
                    within="torch_recurrent_gated_delta_rule",
                ),
                "fp32_core_cast": line_of(
                    "x.transpose(1, 2).contiguous().to(torch.float32)"
                    " for x in (query, key, value, beta, g)",
                    within="torch_recurrent_gated_delta_rule",
                ),
            },
            "routing": "use_recurrent_gdn (this script): every GDN layer's "
            "chunk_gated_delta_rule replaced by an adapter over the recurrent form; a packed "
            "call (cu_seqlens) is refused",
            "attention": "eager (transformers' eager_attention_forward, the causal mask built by "
            "the model from attention_mask=None), against the reference's sdpa flash",
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    scratch = Path(tempfile.mkdtemp(prefix="qd-oracle-tiny-"))
    snapshot = scratch / "snapshot"
    tower_init = write_snapshot(snapshot, args.layers)
    seqs, plan = synth_shard_set(args.steps)
    write_shard_form(out / "batches", seqs, plan)

    common = dict(
        snapshot=snapshot,
        layers=args.layers,
        seqs=seqs,
        plan=plan,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        lower_layers_n=args.lower_layers_n,
        lower_lr_scale=args.lower_layers_lr_scale,
        scratch=scratch,
    )
    arms: dict[str, ArmResult] = {}
    span_init: dict[str, torch.Tensor] | None = None
    for arm in args.arms:
        dtype, clip = ARMS[arm]
        res = run_arm(
            dtype=dtype,
            max_grad_norm=clip,
            span_head_init=span_init,
            keep_grads=arm == "fp32",
            **common,
        )
        span_init = span_init or res.init_span_head
        arms[arm] = res
        if arm == "fp32_clip150":
            active = [r["clip_active"] for r in res.records]
            if all(active) or not any(active):
                raise SystemExit(
                    f"arm {arm}: the clip was {'active' if all(active) else 'inactive'} on "
                    "every step, so this arm does not run both clip branches -- the one "
                    "thing Amendment 2 (iii) asks of it"
                )
        print(
            f"{arm}: {res.seconds:.1f}s, losses {res.losses[0]:.6f} -> {res.losses[-1]:.6f}, "
            f"termination {res.termination}",
            flush=True,
        )

    assert span_init is not None
    init = {**{n: t for n, t in tower_init.items()}, **span_init}
    save_safetensors(
        out / "init.safetensors", init, "the one init both arms load (tower bf16, span head f32)"
    )
    (out / "config.json").write_text((snapshot / "config.json").read_text())
    for arm, res in arms.items():
        d = out / arm
        d.mkdir(exist_ok=True)
        save_safetensors(
            d / "final.safetensors",
            res.final,
            f"arm {arm}: f32 weights after the last step (the fp32 masters under "
            "the master recipe)",
        )
        if res.grads0:
            save_safetensors(
                d / "grads_step0.safetensors",
                res.grads0,
                f"arm {arm}: the accumulated gradient of optimizer step 0, before clipping",
            )
        (d / "trajectory.json").write_text(json.dumps(trajectory(res, plan, seqs), indent=1) + "\n")

    manifest: dict[str, Any] = {
        "what": "rung (b) torch reference (fable-advice.md Q3 b, Amendments 1-2): the Lappi FT "
        "recipe on tessl's tiny Qwen3.5 shape family, CPU, fp32 (tight) and master-bf16 (loose) "
        "arms, and fp32_clip150 (Amendment 2 (iii): max_grad_norm 150, both clip branches), "
        "from one init and one batch order",
        "rule": "published",
        "generator": {
            "script": "tools/qd_train_oracle_tiny.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "argv": [str(a) for a in args.argv],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "numpy": np.__version__,
            "safetensors": _version("safetensors"),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "host": platform.platform(),
            "git_head": _git_head(),
        },
        "seeds": {"protocol_seed": args.seed, "data_seed": DATA_SEED, "init_seed": INIT_SEED},
        "model": {
            "config": "config.json (Qwen3_5Config with text_config; tessl reads text_config)",
            "family": "canonical tessl tools/qwen35_ref/make_train_fixture.py tiny_config, "
            f"{args.layers} layers in the 2B's pattern",
            "layer_types": list(tiny_text_config(args.layers).layer_types),
            "n_tower_params": _snapshot_params(snapshot),
            "init": "init.safetensors: tower under model.language_model.* in bf16 (every value "
            "bf16-exact: tessl reinit, make_train_fixture.py:73-93, then rounded), span "
            "head under span_head.* in f32 (SpanPointerHead's own init after "
            "torch.manual_seed(seed), backbone.py:983,999); both arms load it bit-identically",
            "names": "transformers' names below the tower with the checkpoint prefix "
            "model.language_model.; tessl's parameter_table drops the prefix",
        },
        "seam": {
            "torch": {
                "gdn": gdn_seam(),
                "linear_attention_kernels": next(iter(arms.values())).kernels,
                "attention": {
                    "attn_implementation": next(iter(arms.values())).attn_implementation,
                    "train_attention_mask": "none",
                    "sdpa_backends": "FLASH_ATTENTION only (QwenDecisionStep.training_attention, "
                    "backbone.py:1090-1104), is_causal, no mask; CPU flash",
                },
                "conv1d": "transformers' torch fallback (causal_conv1d not installed on CPU)",
                "gradient_checkpointing": "on, every layer (load_text_tower, use_reentrant=False)",
                "padding": "batches right-padded with PAD_ID to the bucket width (64/128/160); "
                "with no mask and causal attention/GDN/conv a real position never reads a "
                "pad, so tessl's unpadded per-sequence step sees the same math",
            },
            "tessl": "gdn_train (published rule, transformers' seam) + attn_train D=256 GQA; "
            "the Rust side names its own kernels in its row",
        },
        "recipe": {
            "source": "F: campaign/f-v4-preregistered.json recipe.flags (--optimizer master "
            "--lr 1e-5 --lower-layers-n 8 --lower-layers-lr-scale 0.1); see "
            "AUDIT/ojas-training-2026-10-01/f-optimizer-spec.md",
            "lr": args.lr,
            "lr_is_f": args.lr == F_LR,
            "lower_layers_n": args.lower_layers_n,
            "lower_layers_lr_scale": args.lower_layers_lr_scale if args.lower_layers_n else 1.0,
            "span_weight": 1.0,
            "train_attention_mask": "none",
            "loss": "per batch: letter = mean CE over n_supervised letter rows at target_index "
            "(full-vocab, tied head); span = (sum of start and end pointer CEs) / "
            "(2 * n_spans); total = letter + span_weight * span (letter absent on a "
            "span-only batch). grad_accum 1: one batch per optimizer step",
            "schedule": next(iter(arms.values())).schedule,
            "max_grad_norm": next(iter(arms.values())).max_grad_norm,
            "clip": "clip_grad_norm_ over tower + span head live grads: coef = min(1, "
            "max_norm / (norm + 1e-6)); under master-bf16 the tower's grads are bf16 and "
            "are scaled before the fp32 upcast",
        },
        "per_parameter": next(iter(arms.values())).param_recipe,
        "contents": {
            "config.json": "the tiny tower's Qwen3_5Config (text_config inside)",
            "init.safetensors": "the one init both arms load: tower bf16, span head f32",
            "batches/tokens.u32": "every sequence's ids back to back, little-endian u32 "
            "(shard form)",
            "batches/offsets.npy": "int64[n+1]: sequence i is tokens[offsets[i]:offsets[i+1]]",
            "batches/supervision.npz": "np.savez (stored zip): slot_kind u8, target_index i32 "
            "(= length-2), span_target i32[n,2] (-1 not a span, -2 "
            "abstain), candidate_offsets i64[n+1], candidate_positions i32",
            "batches/plan.json": "the batch order, one batch per optimizer step: rows (sequence "
            "ids), width, bucket, index, lengths, slot kinds, the loss "
            "denominators, and the span head's row layout per span row",
            "<arm>/trajectory.json": "per optimizer step: lr handed to apply, total/letter/span "
            "losses (letter_present/span_present say which channel "
            "existed), pre-clip global grad norm, clip coefficient",
            "<arm>/final.safetensors": "f32 weights after the last step (the fp32 masters under "
            "master-bf16), same names as init",
            "fp32/grads_step0.safetensors": "f32 accumulated gradient of optimizer step 0, before "
            "clipping (the forward/backward/supervision check, "
            "independent of the optimizer)",
        },
        "arms": {},
        "files": {},
    }
    for arm, res in arms.items():
        manifest["arms"][arm] = {
            "tower_dtype": ARMS[arm][0],
            "max_grad_norm": ARMS[arm][1],
            "optimizer": res.optimizer,
            "termination": res.termination,
            "digests": arm_digests(res),
            "trajectory": f"{arm}/trajectory.json",
            "final_weights": f"{arm}/final.safetensors",
            "clip_active_steps": [i for i, r in enumerate(res.records) if r["clip_active"]],
            "clip_inactive_steps": [i for i, r in enumerate(res.records) if not r["clip_active"]],
        }
    first = next(iter(arms.values()))
    kinds_of = [[seqs[r].kind for r in b["rows"]] for b in plan]
    lengths = [int(seqs[r].tokens.size) for b in plan for r in b["rows"]]
    manifest["coverage"] = {
        "letter_only_batches": [i for i, k in enumerate(kinds_of) if SLOT_SPAN not in k],
        "span_only_batches": [i for i, k in enumerate(kinds_of) if set(k) == {SLOT_SPAN}],
        "mixed_batches": [
            i for i, k in enumerate(kinds_of) if SLOT_SPAN in k and set(k) != {SLOT_SPAN}
        ],
        "abstaining_span_rows": sum(1 for s in seqs if s.span == (SPAN_ABSTAIN, SPAN_ABSTAIN)),
        "pointing_span_rows": sum(1 for s in seqs if s.kind == SLOT_SPAN and s.span[0] >= 0),
        "letter_rows": sum(1 for s in seqs if s.kind != SLOT_SPAN),
        "row_lengths_min_max": [min(lengths), max(lengths)],
        "rows_past_one_gdn_chunk_64": sum(1 for n in lengths if n > 64),
        "rows_past_two_chunks_128": sum(1 for n in lengths if n > 128),
        "clip_active_steps": [i for i, r in enumerate(first.records) if r["clip_active"]],
        "clip_inactive_steps": [i for i, r in enumerate(first.records) if not r["clip_active"]],
        "span_head_adamw_steps": first.param_recipe["span_head.start_proj.weight"][
            "adamw_steps_taken"
        ],
        "tower_adamw_steps": first.param_recipe[f"{TEXT_PREFIX}norm.weight"]["adamw_steps_taken"],
    }
    if "fp32" in arms and "master_bf16" in arms:
        manifest["arms"]["master_bf16"]["vs_fp32"] = {
            "final_weights": weights_gap(arms["master_bf16"].final, arms["fp32"].final),
            "losses": loss_gap(arms["master_bf16"].losses, arms["fp32"].losses),
        }

    if args.independent_check and "fp32" in arms:
        manifest["independent_check_fp32"] = independent_check(common, arms["fp32"])
        manifest["independent_check_kernel_diverse"] = independent_check(
            common,
            arms["fp32"],
            attn_implementation="eager",
            gdn="recurrent",
            label="independent, kernel-diverse",
        )
    if args.independent_check and "fp32_clip150" in arms:
        manifest["independent_check_fp32_clip150"] = independent_check(
            common,
            arms["fp32_clip150"],
            max_grad_norm=ARMS["fp32_clip150"][1],
            label="independent, clip 150",
        )
    floor_at_f = manifest.get("independent_check_fp32")
    if args.repeat_check and "fp32" in arms:
        manifest["repeat_equality_fp32"] = repeat_check(args, arms)
    if args.discrimination_lrs and "fp32" in arms:
        manifest["discrimination"] = discrimination(args, common, arms["fp32"], floor_at_f)

    return finish(out, manifest)


def repeat_check(args: argparse.Namespace, arms: Mapping[str, ArmResult]) -> dict[str, Any]:
    """The fp32 arms again, in a separate process, everything regenerated from the seeds."""
    again = [a for a in REPEAT_ARMS if a in arms]
    with tempfile.TemporaryDirectory(prefix="qd-oracle-repeat-") as tmp:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--out",
            tmp,
            "--arms",
            ",".join(again),
            "--no-repeat-check",
            "--no-independent-check",
            "--discrimination-lrs",
            "",
            "--steps",
            str(args.steps),
            "--layers",
            str(args.layers),
            "--lr",
            repr(args.lr),
            "--seed",
            str(args.seed),
            "--lower-layers-n",
            str(args.lower_layers_n),
            "--lower-layers-lr-scale",
            repr(args.lower_layers_lr_scale),
            "--threads",
            str(args.threads),
        ]
        subprocess.run(cmd, check=True)
        second = json.loads((Path(tmp) / "manifest.json").read_text())
        theirs_files = {k: v["sha256"] for k, v in second["files"].items()}
    mine_files = {k: sha256_file(args.out / k) for k in sorted(theirs_files)}
    by_arm: dict[str, Any] = {}
    for arm in again:
        mine = arm_digests(arms[arm])
        theirs = second["arms"][arm]["digests"]
        by_arm[arm] = {
            "first": mine,
            "second": theirs,
            "differing": sorted(k for k in mine if mine[k] != theirs.get(k)),
            "equal": mine == theirs,
        }
    return {
        "how": f"a second process (subprocess of this script, --arms {','.join(again)}) "
        "regenerating the snapshot, batches and runs from the seeds; same thread count. "
        "Compared: each arm's value digests and the bytes of every file both runs wrote",
        "first": by_arm["fp32"]["first"],
        "second": by_arm["fp32"]["second"],
        "by_arm": by_arm,
        "files_compared": sorted(theirs_files),
        "files_differing": sorted(k for k in theirs_files if mine_files[k] != theirs_files[k]),
        "differing": by_arm["fp32"]["differing"],
        "equal": all(r["equal"] for r in by_arm.values()) and mine_files == theirs_files,
    }


def independent_check(
    common: Mapping[str, Any],
    ref: ArmResult,
    *,
    max_grad_norm: float = 1.0,
    attn_implementation: str = "sdpa",
    gdn: str = "chunk",
    label: str = "independent",
) -> dict[str, Any]:
    """An fp32 arm again through ``IndependentStep``: the same optimizer, clip, schedule and
    init, the forward and loss restated per unpadded sequence. Measured the way rung (b)
    measures, so the result says whether two correct torch implementations clear its bars.

    With ``attn_implementation="eager"`` and ``gdn="recurrent"`` it is the kernel-diverse floor
    (Amendment 2 (i), recommended, not a gate): no attention or GDN kernel shared with the
    reference."""
    res = run_arm(
        dtype="fp32",
        span_head_init=ref.init_span_head,
        keep_grads=True,
        independent=True,
        max_grad_norm=max_grad_norm,
        attn_implementation=attn_implementation,
        gdn=gdn,
        **common,
    )
    w = weights_gap(res.final, ref.final)
    lo = loss_gap(res.losses, ref.losses)
    norms = [r["grad_norm_preclip"] for r in ref.records]
    theirs = [r["grad_norm_preclip"] for r in res.records]
    print(
        f"{label}: weights {w['max_rel_to_param_max']:.3e} (Amendment 2: "
        f"{w['amendment2_max_rel']:.3e}), losses {lo['all_steps_max_rel']:.3e}",
        flush=True,
    )
    shared = attn_implementation == "sdpa" and gdn == "chunk"
    return {
        "how": "IndependentStep (this script): one unpadded sequence at a time through the same "
        "tower, the tied head's logits and torch cross_entropy written out for the letter, the "
        "pointer scores written out for the span head, targets read off the stored sequences "
        "(not ft_supervision/plan_span_batch); the reference's optimizer, clip and schedule. "
        + (
            "Same GDN and attention kernels as the reference: it isolates padding, batching "
            "and loss-reduction order, not the kernels"
            if shared
            else f"Kernels NOT shared with the reference: attention {attn_implementation!r} "
            f"(the reference runs sdpa, flash backend), GDN {gdn!r} "
            "(torch_recurrent_gated_delta_rule, token by token; the reference runs "
            "torch_chunk_gated_delta_rule), both at the published rule"
        ),
        "kernels": {"attn_implementation": attn_implementation, "gdn": gdn},
        "max_grad_norm": max_grad_norm,
        "steps": len(res.records),
        # Report-only at rung (b) (Amendment 2 (i)). A reference arm that kept no step-0
        # gradient (only "fp32" keeps one) is not compared, and says so rather than reading 0.
        "grads_step0": weights_gap(res.grads0, ref.grads0)
        if ref.grads0
        else {"not_run": "the reference arm keeps no step-0 gradient; only arm fp32 does"},
        "grad_norm_preclip_max_rel": max(
            abs(a - b) / max(abs(b), 1e-30) for a, b in zip(theirs, norms, strict=True)
        ),
        "losses": lo,
        "final_weights": w,
        "measure": "Amendment 2 (i)",
        "within_rung_b_bars": {
            "final_weights": w["amendment2_max_rel"] <= TOL_FINAL_WEIGHTS,
            "losses": lo["steps_0_5_max_rel"] <= TOL_LOSS_EARLY
            and lo["all_steps_max_rel"] <= TOL_LOSS_LATE,
        },
        "within_rung_b_bars_pre_amendment": {
            "final_weights": w["max_rel_to_param_max"] <= TOL_FINAL_WEIGHTS,
        },
        "clip_active_steps_equal": [r["clip_active"] for r in res.records]
        == [r["clip_active"] for r in ref.records],
        "seconds": res.seconds,
    }


def discrimination(
    args: argparse.Namespace,
    common: Mapping[str, Any],
    ref_at_f: ArmResult,
    floor_at_f: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Each counterfactual against the fp32 reference at the same lr, measured the way rung (b)
    measures: final weights max|delta|/max|w| per parameter, losses relative per step. Beside
    each, its ratio to the independent restatement's gap at that lr (``independent_check``): a
    recipe difference within a small factor of what two correct implementations already differ
    by is not one the bar separates reliably, whichever side of 1e-5 it lands."""
    table: dict[str, Any] = {
        "how": "fp32 arm rerun with one recipe detail changed; 'caught' means the measured gap "
        "exceeds rung (b)'s bar (final weights 1e-5 of each parameter's max; loss 1e-5 "
        "relative over steps 0-5, 1e-4 to the end); 'caught_by_final_weights' is the "
        "pre-amendment measure (each tensor over its own max), "
        "'caught_by_final_weights_amendment2' the measure in force (span_head.* over the span "
        "head's max, Amendment 2 (i)); 'over_independent_floor' divides the gap "
        "by the independent restatement's gap at the same lr on the same measure",
        "by_lr": {},
    }
    for lr in args.discrimination_lrs:
        base = dict(common, lr=lr)
        ref = ref_at_f if lr == args.lr else run_arm(dtype="fp32", span_head_init=None, **base)
        floor = (
            floor_at_f if lr == args.lr and floor_at_f is not None else independent_check(base, ref)
        )
        fw = floor["final_weights"]
        warm = ref.schedule["warmup_steps"]
        total = ref.schedule["total_steps"]
        sched = real_ft_run._control(total, device="cpu", lr=lr).schedule
        rows: dict[str, Any] = {}
        for name, cf in counterfactuals().items():
            if name in ("no_lr_scale", "decay_not_lr_scaled") and not args.lower_layers_n:
                continue
            lr_edit: LrEdit | None = None
            if cf.get("lr_edit") == "shift":

                def lr_edit(
                    i: int, lr_now: float, _w: int = warm, _t: int = total, _s: Any = sched
                ) -> float:
                    return lr_now if i < _w else _s.lr_at(min(i + 1, _t - 1))

            res = run_arm(
                dtype="fp32",
                span_head_init=None,
                optimizer_edit=cf.get("optimizer_edit"),
                lr_edit=lr_edit,
                max_grad_norm=cf.get("max_grad_norm", 1.0),
                **base,
            )
            w = weights_gap(res.final, ref.final)
            lo = loss_gap(res.losses, ref.losses)
            rows[name] = {
                "what": cf["what"],
                "final_weights": w,
                "losses": lo,
                "caught_by_final_weights": w["max_rel_to_param_max"] > TOL_FINAL_WEIGHTS,
                "caught_by_final_weights_amendment2": w["amendment2_max_rel"] > TOL_FINAL_WEIGHTS,
                "caught_by_losses": lo["steps_0_5_max_rel"] > TOL_LOSS_EARLY
                or lo["all_steps_max_rel"] > TOL_LOSS_LATE,
                "over_independent_floor": {
                    "final_weights": w["max_rel_to_param_max"] / fw["max_rel_to_param_max"],
                    "final_weights_amendment2": w["amendment2_max_rel"] / fw["amendment2_max_rel"],
                    "final_weights_tower": w["tower_max_rel_to_param_max"]
                    / fw["tower_max_rel_to_param_max"],
                    "losses_all_steps": lo["all_steps_max_rel"]
                    / floor["losses"]["all_steps_max_rel"],
                },
            }
            print(
                f"  lr={lr:g} {name}: weights {w['max_rel_to_param_max']:.3e}, "
                f"losses {lo['all_steps_max_rel']:.3e}",
                flush=True,
            )
        table["by_lr"][repr(lr)] = {
            # The reference's own displacement from init on the same measure; a zero-init
            # parameter (the abstain vectors) is measured absolutely here.
            "reference_moved_from_init": weights_gap(
                ref.final, _init_fp32(common["snapshot"], ref)
            ),
            "independent_floor": {
                k: floor[k] for k in ("final_weights", "losses", "within_rung_b_bars")
            },
            "counterfactuals": rows,
        }
    return table


def _init_fp32(snapshot: Path, ref: ArmResult) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    t = {n: v.to(torch.float32) for n, v in load_file(str(snapshot / "model.safetensors")).items()}
    return {**t, **{n: v.to(torch.float32) for n, v in ref.init_span_head.items()}}


def finish(out: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    total = 0
    for p in files:
        size = p.stat().st_size
        total += size
        manifest["files"][str(p.relative_to(out))] = {"bytes": size, "sha256": sha256_file(p)}
    manifest["total_bytes_excluding_manifest"] = total
    if total > MAX_FIXTURE_BYTES:
        raise SystemExit(
            f"the fixture is {total:,} bytes, over the {MAX_FIXTURE_BYTES:,}-byte budget; cut "
            "--steps or --layers and say so"
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    return manifest


def _version(module: str) -> str:
    import importlib.metadata as md

    try:
        return md.version(module)
    except md.PackageNotFoundError:
        return "not installed"


def _git_head() -> str:
    """HEAD through ``tools/repo_git.py``, the one owner of git calls from ``tools/``. The
    script's own sha256 sits beside it in the manifest: the oracle may be uncommitted at
    HEAD when it runs, so HEAD alone does not say which oracle wrote the fixture."""
    from repo_git import resolve_rev

    try:
        return resolve_rev(ROOT, "HEAD")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return f"unknown: {exc}"


def _lrs(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _arms(text: str) -> list[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = sorted(set(names) - set(ARMS))
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"arms are a comma list of {sorted(ARMS)}, got {text!r}")
    return names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", type=_arms, default=list(ARMS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--lr", type=float, default=F_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--lower-layers-n",
        type=int,
        default=DEFAULT_LOWER_LAYERS_N,
        help="0 builds the single-group optimizer (no layer-wise split)",
    )
    parser.add_argument(
        "--lower-layers-lr-scale", type=float, default=real_ft_run.RSI_LOWER_LR_SCALE
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-repeat-check", dest="repeat_check", action="store_false")
    parser.add_argument(
        "--no-independent-check",
        dest="independent_check",
        action="store_false",
        help="skip the unpadded per-sequence restatement of the fp32 arm",
    )
    parser.add_argument(
        "--discrimination-lrs",
        type=_lrs,
        default=[F_LR, 1e-3],
        help="comma list; empty to skip the counterfactual table",
    )
    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if args.steps < 6:
        raise SystemExit("--steps must be at least 6: rung (b)'s early-loss bar covers steps 0-5")
    manifest = build(args)
    rep = manifest.get("repeat_equality_fp32")
    ind = manifest.get("independent_check_fp32")
    print(
        f"wrote {args.out}: {manifest['total_bytes_excluding_manifest']:,} bytes"
        + ("" if rep is None else f"; fp32 repeat equal: {rep['equal']}")
        + ("" if ind is None else f"; independent within bars: {ind['within_rung_b_bars']}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
