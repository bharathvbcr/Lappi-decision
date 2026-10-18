"""A scoring step builds no optimizer and is budgeted as scoring (ft-a3b6c9ea916, ruling 2j.9).

``_real_step`` is the one owner of step construction; it takes ``training``. Scoring
(``_scoring_step`` -> every ``--score-*`` path) builds the tower with ``training=False``:
no optimizer object, no fp32 master copy, a footprint of weights + no-grad activations + the
logit slab. The budget arithmetic is ``test_memory_scoring.py``; this is the tower-side half and
the end-to-end proof that the verdicts did not move.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run  # noqa: E402
import test_backbone as tb  # noqa: E402
from test_real_ft_score_checkpoint import (  # noqa: E402
    average_masters_with_proof,
    tiny_master_checkpoints,
)
from test_real_ft_shuffled_label import REV  # noqa: E402
from torch.utils import _pytree as pytree  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from qd_train.artifacts import Batch  # noqa: E402
from qd_train.backbone import (  # noqa: E402
    BackboneContractViolation,
    QwenDecisionStep,
    footprint_at,
)
from qd_train.memory import ADAMW_MASTER, estimate_scoring  # noqa: E402
from qd_train.optim import MasterWeightAdamW  # noqa: E402


@pytest.fixture
def masters_built(monkeypatch):
    """Counts the ``MasterWeightAdamW`` objects built (each clones every parameter to fp32)."""
    built: list[object] = []
    init = MasterWeightAdamW.__init__

    def counting(self, *args, **kwargs):
        built.append(self)
        init(self, *args, **kwargs)

    monkeypatch.setattr(MasterWeightAdamW, "__init__", counting)
    return built


def _batch(rows: int, width: int) -> Batch:
    rng = np.random.default_rng(0)
    return Batch(
        tokens=rng.integers(0, tb.TINY_VOCAB, size=(rows, width)).astype(np.int32),
        lengths=np.full(rows, width, dtype=np.int32),
        bucket=0, index=0,
    )


def _tower(tmp_path: Path, *, training: bool, spec=ADAMW_MASTER, dtype: str = "bf16"):
    return tb._tiny_tower(tmp_path, optimizer=spec, dtype=dtype, training=training)[0]


def test_a_scoring_step_holds_no_optimizer_and_no_master_copy(tmp_path, masters_built):
    step = QwenDecisionStep(
        _tower(tmp_path / "s", training=False), seed=0, lr=1e-3, total_steps=1, max_width=64
    )
    assert step.optimizer is None
    with torch.no_grad():
        step.lm_head(step.hidden(_batch(2, 8)))
    assert masters_built == [], "a scoring step allocated fp32 masters"

    # The control: the same tower, as a training step, does hold them.
    trained = QwenDecisionStep(
        _tower(tmp_path / "t", training=True), seed=0, lr=1e-3, total_steps=1, max_width=64
    )
    assert masters_built == [trained.optimizer]
    assert len(trained.optimizer.state_dict()["masters"]) > 0


def test_a_scoring_step_refuses_what_needs_an_optimizer(tmp_path):
    step = QwenDecisionStep(
        _tower(tmp_path, training=False), seed=0, lr=1e-3, total_steps=1, max_width=64
    )
    for call in (
        lambda: step.apply(lr=1e-3),
        lambda: step.state(),
        lambda: step.load_state({}),
        lambda: step.accumulate(_batch(1, 4), None),
        lambda: step.accumulate_span(_batch(1, 4), None),
    ):
        with pytest.raises(BackboneContractViolation, match="scoring step"):
            call()
    # What a scoring step is for still works: weights in.
    state = QwenDecisionStep(
        _tower(tmp_path / "t", training=True), seed=0, lr=1e-3, total_steps=1, max_width=64
    ).state()
    step.load_weights(state)


def test_a_scoring_step_refuses_optimizer_arguments_it_would_discard(tmp_path):
    tower = _tower(tmp_path, training=False)
    for kw in ({"lower_layers_n": 1}, {"fused_adamw": True}):
        with pytest.raises(ValueError, match="scoring step"):
            QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64, **kw)


def test_the_tower_carries_its_purpose_and_prices_itself_as_scoring(tmp_path):
    train = _tower(tmp_path / "t", training=True)
    score = _tower(tmp_path / "s", training=False)
    assert train.training is True and score.training is False
    assert "training" not in train.to_json() and score.to_json()["training"] is False
    f = score.footprint
    assert f.purpose == "scoring"
    assert (f.grad_bytes, f.optimizer_bytes, f.master_bytes) == (0, 0, 0)
    assert footprint_at(score, rows=2, width=16).purpose == "scoring"
    assert footprint_at(train, rows=2, width=16).purpose == "train"
    # Same weights either way; the difference is exactly what scoring never holds.
    ft = footprint_at(train, rows=2, width=16)
    fs = footprint_at(score, rows=2, width=16)
    assert fs.param_bytes == ft.param_bytes
    assert fs.total_bytes < ft.total_bytes


class _LiveBytes(TorchDispatchMode):
    """Peak bytes of the tensors the dispatched ops allocate while it is active: each output
    storage is counted when first produced and released when it is garbage collected."""

    def __init__(self) -> None:
        super().__init__()
        self.live = self.peak = 0
        self._tracked: dict[int, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for leaf in pytree.tree_leaves(out):
            if isinstance(leaf, torch.Tensor):
                storage = leaf.untyped_storage()
                ptr = storage.data_ptr()
                if ptr and ptr not in self._tracked:
                    self._tracked[ptr] = storage.nbytes()
                    self.live += storage.nbytes()
                    self.peak = max(self.peak, self.live)
                    weakref.finalize(storage, self._release, ptr)
        return out

    def _release(self, ptr: int) -> None:
        self.live -= self._tracked.pop(ptr, 0)


def test_the_scoring_budget_bounds_the_measured_tiny_tower_footprint(tmp_path, monkeypatch):
    """Measured on a tiny tower: weights exactly, the logit slab exactly, and the peak bytes a
    no-grad forward + the full-vocabulary head allocate against the activations + slab the
    budget names (it must hold the measurement, and not be absurdly above it).

    The tower is all full-attention. Under transformers' torch fallback for the linear-attention
    layers (no ``fla`` on this host) a GDN layer allocates chunked ``[B, Hv, N, 64, 64]`` fp32
    scratch the memory model has never had a term for, in training or scoring; the CUDA kernel
    path does not form it. That is a separate, older limit and it is not what this measures."""
    orig = tb._tiny_text_config

    def all_full(*args, **kwargs):
        cfg = orig(*args, **kwargs)
        cfg.layer_types = ["full_attention"] * tb.TINY_LAYERS
        return cfg

    monkeypatch.setattr(tb, "_tiny_text_config", all_full)
    tower = _tower(tmp_path, training=False)
    assert set(tower.model.config.layer_types) == {"full_attention"}
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    rows, width = 4, 64
    batch = _batch(rows, width)
    with torch.no_grad():
        step.lm_head(step.hidden(batch))  # warm: one-time buffers are not the forward
        meter = _LiveBytes()
        with meter:
            logits = step.lm_head(step.hidden(batch))

    f = footprint_at(tower, rows=rows, width=width)
    assert f.logit_slab_bytes == logits.numel() * logits.element_size()  # exact
    weights = sum(p.numel() * p.element_size() for p in tower.model.parameters())
    exact = dataclasses.replace(tb._tiny_spec(), params_total=sum(
        p.numel() for p in tower.model.parameters()
    ))
    assert estimate_scoring(
        exact, rows=rows, width=width, param_dtype="bf16", activation_dtype="bf16"
    ).param_bytes == weights
    budget = (
        f.activation_bytes + f.score_matrix_bytes + f.recurrent_state_bytes + f.safety_bytes
        + f.logit_slab_bytes
    )
    assert meter.peak > f.logit_slab_bytes, f"measured peak {meter.peak} B holds not even the slab"
    assert meter.peak <= budget, f"measured {meter.peak} B exceeds the scoring budget {budget} B"
    assert budget <= 8 * meter.peak, f"budget {budget} B is over 8x the measured {meter.peak} B"
    # And it is the scoring budget, not the training one with its optimizer and gradients.
    assert f.total_bytes == f.param_bytes + budget


# --- _real_step, the one owner ---------------------------------------------------------------


def _real_step_args(tiny, *, training: bool | None):
    """What ``_scoring_step`` passes, for the tiny fixture; ``training=None`` omits it."""
    from qd_train.shards import ShardReader

    reader = ShardReader(tiny.out / "shards" / "train", config=real_ft_run.DataConfig(),
                         repo_root=tiny.out, expect_rev=REV)
    kwargs = {} if training is None else {"training": training}
    return dict(
        backbone=tiny.snapshot, reader=reader,
        plan=[SimpleNamespace(tokens=torch.zeros((1, 8)))],
        device="cpu", dtype="fp32", spec=real_ft_run.optimizer_spec("fp32", "master"),
        attn_implementation="sdpa", seed=0, lr=1e-2, total_steps=2, span_weight=1.0, width=8,
        **kwargs,
    )


def test_real_step_builds_a_scoring_step_without_an_optimizer_and_keeps_eval_widths(
    tmp_path, monkeypatch
):
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    step, tower, _ = real_ft_run._real_step(
        **_real_step_args(tiny, training=False), eval_widths=(40,)
    )
    assert step.optimizer is None and tower.training is False
    assert tower.footprint.purpose == "scoring"
    # The 2026-09-30 refusal: a step bounded by the plan alone refused the needle suite after
    # a full val pass. eval_widths still raise max_width, scoring or not.
    assert step.max_width == 40
    train_step, train_tower, _ = real_ft_run._real_step(
        **_real_step_args(tiny, training=True), eval_widths=(40,)
    )
    assert train_step.optimizer is not None and train_tower.footprint.purpose == "train"
    assert train_step.max_width == 40
    # Training is the default: _train's call does not name it.
    default_step, _, _ = real_ft_run._real_step(**_real_step_args(tiny, training=None))
    assert default_step.optimizer is not None


def test_every_scoring_path_builds_through_scoring_step_with_training_off():
    """The source pin: `_scoring_step` is what every score path calls, and it is the one place
    that says ``training=False``; `_train`'s call says nothing."""
    src = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    scoring = src[src.index("def _scoring_step("):]
    scoring = scoring[: scoring.index("\ndef ")]
    assert "training=False" in scoring
    train_call = src[src.index("step, tower, budget = _real_step("):]
    assert "training" not in train_call[: train_call.index(")\n")]


def test_a_scoring_budget_prices_the_decode_only_widths_a_training_one_does_not(
    tmp_path, monkeypatch
):
    """The needle suite's ~8K cases are decoded by the scoring step, whose forward and logit
    slab the scoring footprint prices, so the budget is the costliest of the plan's shapes AND
    each eval width at one row. (`max_width` took them since 2026-09-30; the budget did not.)"""
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    _, scoring, _ = real_ft_run._real_step(
        **_real_step_args(tiny, training=False), eval_widths=(40, 24)
    )
    assert (scoring.footprint.rows, scoring.footprint.width) == (1, 40)
    _, training, _ = real_ft_run._real_step(
        **_real_step_args(tiny, training=True), eval_widths=(40, 24)
    )
    assert training.footprint.width == 8, "a training budget stays the plan's"


# --- the verdicts did not move ----------------------------------------------------------------


def _score_average(tiny, tmp_path: Path, tag: str) -> list[str]:
    """Average the tiny checkpoints, score the average, and return its verdict lines with the
    one volatile field (the eval row's uuid) removed."""
    avg = tmp_path / f"avg-{tag}" / "avg.safetensors"
    assert average_masters_with_proof(tmp_path, tiny, avg) == 0
    verdicts = tmp_path / f"verdicts-{tag}.jsonl"
    assert real_ft_run.main([
        "--out", str(tiny.out), "--rev", REV, "--devices", "cpu", "--seeds", "0", "1", "2",
        "--score-val", "--score-checkpoint", str(avg), "--real-backbone", str(tiny.snapshot),
        "--ft-ledger", str(tiny.ft_ledger), "--ledger", str(tmp_path / f"eval-{tag}.jsonl"),
        "--tokenizer-json", str(tiny.tokenizer), "--verdicts-out", str(verdicts),
    ]) == 0
    out = []
    for line in verdicts.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.pop("eval_row_id")
        out.append(json.dumps(row, sort_keys=True))
    return out


def test_scoring_verdicts_are_byte_identical_to_the_training_step_scoring_they_replace(
    tmp_path, monkeypatch
):
    tiny = tiny_master_checkpoints(tmp_path, monkeypatch)
    built: list[object] = []
    real = real_ft_run._real_step

    def recording(*args, **kwargs):
        out = real(*args, **kwargs)
        built.append(out[0])
        return out

    monkeypatch.setattr(real_ft_run, "_real_step", recording)
    scored = _score_average(tiny, tmp_path, "scoring")
    assert built and all(s.optimizer is None for s in built), "scoring built an optimizer"

    # The old behaviour: the same scoring path, forced to build the training step.
    built.clear()

    def as_before(*args, **kwargs):
        kwargs["training"] = True
        return recording(*args, **kwargs)

    monkeypatch.setattr(real_ft_run, "_real_step", as_before)
    before = _score_average(tiny, tmp_path, "training")
    assert built and all(s.optimizer is not None for s in built)
    assert len(scored) > 0
    assert scored == before


def test_a_9b_scoring_budget_passes_the_device_check_a_9b_training_budget_fails(monkeypatch):
    """`device_budget` is what `_real_step` raises on: the card's 80 GiB, the 9B's fp32 weights
    at 33.4 GiB. Training's number is refused; the scoring number is not."""
    from test_memory_scoring import FP32, GIB, NINE_B

    from qd_train.memory import ADAMW_FP32, estimate_step

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (80 * GIB, 80 * GIB))
    train = estimate_step(NINE_B, rows=1, width=2048, optimizer=ADAMW_FP32, **FP32)
    score = estimate_scoring(
        NINE_B, rows=1, width=2048, param_dtype="fp32", activation_dtype="fp32"
    )
    assert real_ft_run.device_budget(train.total_bytes, "cuda").passed is False
    assert real_ft_run.device_budget(score.total_bytes, "cuda").passed is True
