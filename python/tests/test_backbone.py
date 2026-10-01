"""The real text tower: its key set, its remap, its checkpointing, and the step it drives.

Torch-gated: this module ``importorskip``s torch and transformers, so in the repo venv it
reports as **skipped** rather than passing vacuously. To run it::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project \\
        --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \\
        python -m pytest python/tests/test_backbone.py -o addopts= -q

Most tests build a **tiny** Qwen3.5 text tower and save it as a snapshot in the real
on-disk layout -- ``model.language_model.*`` in a safetensors file beside a
``Qwen3_5Config`` with a ``text_config``. The class under test is the real one, so the API
facts transfer; what a tiny tower cannot prove is anything about the 320-tensor key set of
the actual checkpoint, and [`test_the_real_checkpoints_text_keys_are_a_bijection...`] is
the test that reads the real one. It builds the model on the ``meta`` device, so it
allocates nothing and still compares every key and every shape.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")
pytest.importorskip("safetensors", reason="safetensors is an optional 'mac' extra")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_train.artifacts import (  # noqa: E402
    NO_SPAN,
    SLOT_CHOICE,
    SLOT_SPAN,
    Batch,
)
from qd_train.backbone import (  # noqa: E402
    TEXT_PREFIX,
    BackboneContractViolation,
    GradientCheckpointingDisabled,
    QwenDecisionStep,
    footprint_at,
    load_text_tower,
    remap_text_tower,
    saved_activation_bytes,
    text_tensor_index,
)
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.memory import ADAMW_BF16, ADAMW_FP32, ModelSpec  # noqa: E402
from qd_train.remap import build_remap, full_vocab_remap  # noqa: E402
from qd_train.run_control import (  # noqa: E402
    CostEstimate,
    LRSchedule,
    RunControl,
    WallClockCap,
)
from qd_train.trainer import (  # noqa: E402
    SpanScoringStep,
    TrainerContractViolation,
    TrainStep,
    ft_supervision,
    train_ft,
)
from qd_train.tristate import NotRun  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(
    "/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/"
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)

TINY_VOCAB, TINY_HIDDEN, TINY_LAYERS = 64, 32, 4


# --- the tiny snapshot fixture -------------------------------------------------------------


def _tiny_text_config(vocab: int = TINY_VOCAB):
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=TINY_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
    )


def _tiny_spec(vocab: int = TINY_VOCAB) -> ModelSpec:
    """A ``ModelSpec`` describing the tiny tower, so the footprint is about what was loaded.

    ``load_text_tower`` refuses a spec that describes a different model; that refusal is
    itself under test in [`test_a_memory_spec_for_another_model_is_refused`]. This is the
    matching one.
    """
    cfg = _tiny_text_config(vocab)
    n_full = sum(1 for t in cfg.layer_types if t == "full_attention")
    return ModelSpec(
        name="tiny Qwen3.5 text tower (test fixture)",
        hidden_size=TINY_HIDDEN,
        intermediate_size=64,
        n_full_attention_layers=n_full,
        n_linear_attention_layers=TINY_LAYERS - n_full,
        q_heads=2,
        kv_heads=1,
        head_dim=16,
        attn_output_gate=True,
        linear_heads=2,
        linear_head_dim=16,
        vocab_size=vocab,
        params_total=4_000_000,
        params_embedding=vocab * TINY_HIDDEN,
        tied_embedding=True,
        recurrent_state_bytes=4,
    )


def _write_tiny_snapshot(dirpath: Path, *, seed: int = 0, vocab: int = TINY_VOCAB):
    """Write a tiny tower in the real on-disk layout. Returns the reference model."""
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextModel

    text = _tiny_text_config(vocab)
    torch.manual_seed(seed)
    model = Qwen3_5TextModel(text)
    dirpath.mkdir(parents=True, exist_ok=True)
    Qwen3_5Config(text_config=text.to_dict()).save_pretrained(dirpath)
    save_file(
        {
            f"{TEXT_PREFIX}{k}": v.detach().clone().contiguous()
            for k, v in model.state_dict().items()
        },
        str(dirpath / "model.safetensors"),
    )
    return model


def _tiny_tower(tmp_path: Path, *, gradient_checkpointing: bool = True, **kwargs):
    # `optimizer` and `dtype` are defaults rather than fixed arguments so a test can ask for
    # a bf16 tower under the fp32-master recipe; every existing caller passes neither and is
    # unaffected.
    kwargs.setdefault("optimizer", ADAMW_FP32)
    kwargs.setdefault("dtype", "fp32")
    # Same reason as the two above: every existing caller passes none and is unaffected,
    # and a test that wants to compare two kernels can ask for one.
    kwargs.setdefault("attn_implementation", "sdpa")
    reference = _write_tiny_snapshot(tmp_path / "snapshot")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GradientCheckpointingDisabled)
        tower = load_text_tower(
            tmp_path / "snapshot",
            gradient_checkpointing=gradient_checkpointing,
            rows=1,
            width=64,
            spec=_tiny_spec(),
            **kwargs,
        )
    return tower, reference


# --- the real checkpoint ---------------------------------------------------------------------


@pytest.mark.skipif(
    not SNAPSHOT.is_dir(), reason=f"the Qwen3.5-2B-Base snapshot is not at {SNAPSHOT}"
)
def test_the_real_checkpoints_text_keys_are_a_bijection_with_the_text_model_class():
    """The decision this module rests on, re-measured rather than quoted.

    ``load_text_tower`` builds a ``Qwen3_5TextModel`` from ``config.text_config`` instead of
    loading the whole ``Qwen3_5ForConditionalGeneration`` and taking ``.model.language_model``.
    That is only honest if the two key sets are the *same* set -- otherwise the load is
    quietly partial and the rest of the tower stays randomly initialised.

    Built on ``meta``, so this allocates no weights.
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5TextModel

    index = text_tensor_index(SNAPSHOT)
    config = AutoConfig.from_pretrained(SNAPSHOT)
    with torch.device("meta"):
        skeleton = Qwen3_5TextModel(config.text_config)
    wanted = {name: tuple(t.shape) for name, t in skeleton.state_dict().items()}
    found = {name: shape for name, (_shard, shape) in index.items()}

    assert set(wanted) == set(found), (
        f"model-only: {sorted(set(wanted) - set(found))[:8]}; "
        f"checkpoint-only: {sorted(set(found) - set(wanted))[:8]}"
    )
    mismatched = {k: (wanted[k], found[k]) for k in wanted if wanted[k] != found[k]}
    assert not mismatched, f"shape disagreements: {list(mismatched.items())[:8]}"
    assert len(wanted) == 320, f"the text tower is 320 tensors, this snapshot has {len(wanted)}"


@pytest.mark.skipif(
    not SNAPSHOT.is_dir(), reason=f"the Qwen3.5-2B-Base snapshot is not at {SNAPSHOT}"
)
def test_the_query_projection_is_doubled_by_the_output_gate_in_the_checkpoint_itself():
    """``attn_output_gate`` read off the weights, never off the config.

    ``num_attention_heads * head_dim`` is ``8 * 256 = 2048``; ``q_proj`` is ``[4096, 2048]``
    because the projection emits a gate beside the query. An architecture assumed from the
    head count understates the query activation by exactly 2x, and
    ``memory.QWEN3_5_2B_TEXT`` carries ``attn_output_gate=True`` on the strength of this.

    The checkpoint's ``config.json`` does carry ``text_config.attn_output_gate: true`` --
    and reading it would still be the wrong way round. Measured 2026-09-20: the string
    ``attn_output_gate`` appears **nowhere in the transformers 5.12.1 package at all**, and
    ``modeling_qwen3_5.py:657`` sizes the projection as
    ``nn.Linear(hidden_size, num_attention_heads * head_dim * 2)`` with the ``* 2``
    unconditional. So the config key is decorative in this version: a checkpoint declaring
    it ``false`` would still get the doubled projection. The tensor shape is the only
    load-bearing statement of the architecture there is.
    """
    index = text_tensor_index(SNAPSHOT)
    q_projections = {
        name: shape
        for name, (_s, shape) in index.items()
        if name.endswith("self_attn.q_proj.weight")
    }
    assert q_projections, "no self_attn.q_proj.weight in the text tower"
    for name, shape in q_projections.items():
        assert shape == (4096, 2048), f"{name} is {shape}, expected (4096, 2048)"


# --- construction -------------------------------------------------------------------------------


def test_the_tower_loads_every_text_tensor_and_the_weights_are_the_files_weights(tmp_path):
    """A load that produced random weights would pass every shape check ever written."""
    tower, reference = _tiny_tower(tmp_path)
    assert tower.n_tensors_loaded == len(reference.state_dict())
    assert tower.vocab_size == TINY_VOCAB
    assert tower.hidden_size == TINY_HIDDEN

    loaded = tower.model.state_dict()
    for name, expected in reference.state_dict().items():
        assert torch.equal(loaded[name], expected), f"{name} is not the tensor in the file"


def test_the_output_head_is_the_embedding_and_no_second_head_exists(tmp_path):
    """``tie_word_embeddings`` is total here: there is one tensor, not two that agree."""
    tower, _ = _tiny_tower(tmp_path)
    assert tower.model.get_output_embeddings() is None, (
        "a base text tower has no LM head; if this ever returns a module, the head stopped "
        "being the embedding and the remap's memory saving halved"
    )
    assert tower.lm_head_weight is tower.model.get_input_embeddings().weight
    assert tuple(tower.lm_head_weight.shape) == (TINY_VOCAB, TINY_HIDDEN)


def test_a_snapshot_whose_text_keys_differ_is_refused_rather_than_partly_loaded(tmp_path):
    """``strict=False`` is how a model trains with randomly initialised layers."""
    from safetensors.torch import load_file, save_file

    snapshot = tmp_path / "snapshot"
    _write_tiny_snapshot(snapshot)
    tensors = load_file(str(snapshot / "model.safetensors"))
    dropped = f"{TEXT_PREFIX}layers.0.mlp.up_proj.weight"
    assert dropped in tensors
    del tensors[dropped]
    save_file(tensors, str(snapshot / "model.safetensors"))

    with pytest.raises(BackboneContractViolation) as excinfo:
        load_text_tower(
            snapshot,
            gradient_checkpointing=True,
            optimizer=ADAMW_FP32,
            attn_implementation="sdpa",
            dtype="fp32",
            width=64,
            spec=_tiny_spec(),
        )
    assert "layers.0.mlp.up_proj.weight" in str(excinfo.value)


def test_a_memory_spec_for_another_model_is_refused(tmp_path):
    """A footprint about a model nobody loaded reads exactly like one that was checked."""
    snapshot = tmp_path / "snapshot"
    _write_tiny_snapshot(snapshot)
    with pytest.raises(BackboneContractViolation) as excinfo:
        load_text_tower(
            snapshot,
            gradient_checkpointing=True,
            optimizer=ADAMW_FP32,
            attn_implementation="sdpa",
            dtype="fp32",
            width=64,
        )  # spec defaults to QWEN3_5_2B_TEXT, which is not this tower
    message = str(excinfo.value)
    assert "hidden_size" in message and "vocab_size" in message


# --- gradient checkpointing -------------------------------------------------------------------


def test_gradient_checkpointing_actually_reduces_retained_activations(tmp_path):
    """The two behaviours, measured directly, because the flag is not the mechanism.

    There is no pre-fix version of this module to graft a test onto -- gradient
    checkpointing did not exist anywhere in the repository. So this measures the two states
    against each other and quotes both, which is the claim that matters: that the request
    changes what the autograd graph retains, not merely what an attribute says.

    ``saved_tensors_hooks`` is the instrument because it is exact and needs no CUDA. A
    forward-count hook is **not**: measured on this host, hooks registered with
    ``register_forward_hook`` reported 0 recomputed layers in both states, which would have
    read as "checkpointing does nothing" from an instrument that simply cannot see it.
    """
    tower, _ = _tiny_tower(tmp_path, gradient_checkpointing=True)
    ids = torch.randint(0, TINY_VOCAB, (2, 32))
    mask = torch.ones_like(ids)

    n_on, bytes_on = saved_activation_bytes(tower.model, ids, mask)
    tower.model.gradient_checkpointing_disable()
    n_off, bytes_off = saved_activation_bytes(tower.model, ids, mask)

    assert bytes_off > 0 and bytes_on > 0, "one of the measurements retained nothing at all"
    assert n_on < n_off, f"saved tensors did not fall: on={n_on} off={n_off}"
    assert bytes_on < bytes_off / 2, (
        f"checkpointing retained {bytes_on:,} B against {bytes_off:,} B off -- less than a "
        "halving is not per-layer checkpointing, whatever the flag says"
    )


def test_turning_gradient_checkpointing_off_is_loud(tmp_path):
    """Off is what every current caller gets, so off says so."""
    snapshot = tmp_path / "snapshot"
    _write_tiny_snapshot(snapshot)
    with pytest.warns(GradientCheckpointingDisabled, match="108.15 GB"):
        tower = load_text_tower(
            snapshot,
            gradient_checkpointing=False,
            optimizer=ADAMW_FP32,
            attn_implementation="sdpa",
            dtype="fp32",
            width=64,
            spec=_tiny_spec(),
        )
    assert tower.gradient_checkpointing is False
    assert tower.footprint.recompute == "none", (
        "the arithmetic must describe the run: a tower without checkpointing budgeted at "
        "recompute='full' is the 108.15 GB case reported as the 25.85 GB one"
    )


def test_checkpointing_on_binds_the_footprint_to_the_recompute_policy(tmp_path):
    tower, _ = _tiny_tower(tmp_path, gradient_checkpointing=True)
    assert tower.gradient_checkpointing is True
    assert tower.footprint.recompute == "full"


def test_a_request_that_did_not_reach_the_layers_is_refused(tmp_path):
    """transformers implements this in a base class the model does not itself call.

    ``Qwen3_5DecoderLayer`` inherits ``GradientCheckpointingLayer``; ``modeling_qwen3_5.py``
    contains no ``_gradient_checkpointing_func`` of its own. A version that stopped
    inheriting it would leave ``supports_gradient_checkpointing`` true and
    ``gradient_checkpointing_enable()`` silent. Simulated here by clearing one layer's flag.
    """
    from qd_train.backbone import _verify_checkpointing_took

    tower, _ = _tiny_tower(tmp_path, gradient_checkpointing=True)
    tower.model.layers[1].gradient_checkpointing = False
    with pytest.raises(BackboneContractViolation) as excinfo:
        _verify_checkpointing_took(tower.model, enabled=True)
    assert "[1]" in str(excinfo.value)


# --- the remap ------------------------------------------------------------------------------------


def _tiny_remap(kept: list[int], specials: list[int]):
    counts = np.zeros(TINY_VOCAB, dtype=np.int64)
    counts[kept] = 1
    return build_remap(
        counts=counts,
        source_vocab_size=TINY_VOCAB,
        tokenizer_hash="t" * 64,
        special_ids=specials,
        target_vocab_size=None,
    )


def test_footprint_at_is_the_estimate_the_tower_was_loaded_with(tmp_path):
    """One owner for "what does a step at this shape cost on this tower", so the budget a
    caller checks per batch shape is the same arithmetic the load recorded."""
    tower, _ = _tiny_tower(tmp_path)
    fp = tower.footprint
    assert footprint_at(tower, rows=fp.rows, width=fp.width) == fp
    wider = footprint_at(tower, rows=fp.rows, width=fp.width * 4)
    assert (wider.rows, wider.width) == (fp.rows, fp.width * 4)
    assert wider.total_bytes > fp.total_bytes


def test_the_remap_moves_every_kept_row_to_the_right_place_in_both_directions(tmp_path):
    """A tied embedding sliced wrong trains the wrong rows and no loss curve shows it.

    Checked over **every** kept row, not a sample, and in both directions: forward against
    ``new_to_old`` and reverse against ``old_to_new``. The reverse is implied by the forward
    when the table really is an inverse pair; checking it anyway is what catches the two
    statements of one mapping having drifted apart.
    """
    tower, _ = _tiny_tower(tmp_path)
    before = tower.model.get_input_embeddings().weight.detach().clone()
    table = _tiny_remap(kept=[1, 3, 5, 7, 9, 11, 13, 40], specials=[0, 2])

    remapped = remap_text_tower(tower, table)
    after = remapped.model.get_input_embeddings().weight.detach()

    assert remapped.vocab_size == table.vocab_size
    assert tuple(after.shape) == (table.vocab_size, TINY_HIDDEN)

    for new_id, old_id in enumerate(table.new_to_old):
        assert torch.equal(after[new_id], before[int(old_id)]), (
            f"new id {new_id} should carry old id {int(old_id)}"
        )

    old_to_new = np.asarray(table.old_to_new)
    kept_old = np.flatnonzero(old_to_new >= 0)
    assert kept_old.size == table.vocab_size
    for old_id in kept_old:
        assert torch.equal(after[int(old_to_new[old_id])], before[int(old_id)]), (
            f"old id {int(old_id)} did not land at new id {int(old_to_new[old_id])}"
        )


def test_the_full_vocabulary_remap_leaves_the_tower_exactly_as_loaded(tmp_path):
    """Sized to the checkpoint's rows, the identity remap slices nothing: every row stays
    where it was and the config still states the checkpoint's vocabulary, so a trained
    tower exports under the config.json it was loaded from."""
    tower, _ = _tiny_tower(tmp_path)
    before = tower.model.get_input_embeddings().weight.detach().clone()
    table = full_vocab_remap(
        source_vocab_size=TINY_VOCAB, tokenizer_hash="t" * 64, special_ids=[0, 2]
    )
    remapped = remap_text_tower(tower, table)
    after = remapped.model.get_input_embeddings().weight.detach()
    assert tuple(after.shape) == tuple(before.shape) == (TINY_VOCAB, TINY_HIDDEN)
    assert torch.equal(after, before)
    assert remapped.vocab_size == TINY_VOCAB
    assert remapped.model.config.vocab_size == TINY_VOCAB
    assert remapped.lm_head_weight is remapped.model.get_input_embeddings().weight


def test_the_remap_keeps_the_head_tied_to_the_embedding(tmp_path):
    """The whole saving: one tensor after the slice, as before it."""
    tower, _ = _tiny_tower(tmp_path)
    table = _tiny_remap(kept=[1, 3, 5], specials=[0])
    remapped = remap_text_tower(tower, table)
    assert remapped.model.get_output_embeddings() is None
    assert remapped.lm_head_weight is remapped.model.get_input_embeddings().weight
    assert remapped.remap is not None
    assert remapped.remap.new_vocab_size == table.vocab_size


def test_a_slice_that_moved_a_row_is_caught(tmp_path, monkeypatch):
    """The check earns its place only if it fails on a wrong slice.

    ``apply_remap_to_model`` verifies shapes, the tie and the row *count* and cannot verify
    row *contents*; this is the failure it would admit. The slice is corrupted after it
    returns, by rolling the sliced embedding one row -- every row is then off by one, which
    is exactly what an inverted or offset mapping produces.
    """
    import qd_train.backbone as backbone

    tower, _ = _tiny_tower(tmp_path)
    table = _tiny_remap(kept=[1, 3, 5, 7], specials=[0])
    real_apply = backbone.apply_remap_to_model

    def rolling_apply(model, remap):
        application = real_apply(model, remap)
        with torch.no_grad():
            weight = model.get_input_embeddings().weight
            weight.copy_(torch.roll(weight.detach().clone(), shifts=1, dims=0))
        return application

    monkeypatch.setattr(backbone, "apply_remap_to_model", rolling_apply)
    with pytest.raises(BackboneContractViolation, match="disagrees with the checkpoint"):
        remap_text_tower(tower, table)


def test_the_remap_shrinks_the_budget_it_reports(tmp_path):
    """The remap exists for the memory; the footprint has to move with it."""
    tower, _ = _tiny_tower(tmp_path)
    before = tower.footprint.trainable_params
    remapped = remap_text_tower(tower, _tiny_remap(kept=[1, 3, 5], specials=[0]))
    assert remapped.footprint.trainable_params < before
    assert remapped.footprint.recompute == tower.footprint.recompute
    assert remapped.optimizer is tower.optimizer, "the recipe must not change under a remap"


# --- the step -------------------------------------------------------------------------------


def _recorder(tmp_path: Path) -> RunRecorder:
    return RunRecorder(
        Ledger(tmp_path / "ledger.jsonl"),
        protocol=Protocol(
            data_snapshot_hash="d" * 64,
            tokenizer_hash="t" * 64,
            backbone_commit="b" * 40,
            recipe_hash="r" * 64,
            seed=7,
        ),
        run_kind="ft",
        repo=REPO,
        # None: every caller of this helper hands the recorder to train_ft, whose block
        # contains the run.
        wall_clock_s=None,
        cost=None,
        env=Environment(
            torch=torch.__version__,
            transformers_sha="none",
            device="cpu",
            host="test",
            fla_present=NotRun(reason="no CUDA on this host"),
            causal_conv1d_present=NotRun(reason="no CUDA on this host"),
        ),
    )


def _control(total_steps: int) -> RunControl:
    cap = WallClockCap(cap_s=600.0)
    return RunControl(
        schedule=LRSchedule(
            peak_lr=1e-3, total_steps=total_steps, warmup_steps=1, min_lr=1e-4
        ),
        cap=cap,
        cost=CostEstimate(usd_per_hour=0.5, cap=cap, n_gpus=1, instance="test-1xA10"),
    )


def _ft_batch(index: int, *, width: int = 6) -> Batch:
    lengths = [5, 4]
    tokens = np.zeros((2, width), dtype=np.int32)
    for row, n in enumerate(lengths):
        tokens[row, :n] = np.arange(1, n + 1, dtype=np.int32)
    return Batch(
        tokens=tokens,
        lengths=np.array(lengths, dtype=np.int32),
        bucket=0,
        index=index,
        slot_kind=np.array([SLOT_SPAN, SLOT_CHOICE], dtype=np.uint8),
        target_index=np.array([3, 2], dtype=np.int32),
        span_target=np.array([(0, 2), (NO_SPAN, NO_SPAN)], dtype=np.int32),
        # Width-aware. Until 2026-09-21 the first row was six hardcoded columns against a
        # second of `width`, so this parameter raised on every value except its default --
        # a knob that only works where it changes nothing.
        #
        # `i < lengths[0]` is not decoration: `Batch` refuses a candidate at or past the
        # row's real length, because the pointer head would otherwise be free to answer
        # with a padded position. A bare alternating pattern marks position 6 at width=7
        # and is rejected there. At width=6 this is the same six booleans as before, so no
        # existing caller moves.
        line_starts=np.array(
            [
                [i % 2 == 0 and i < lengths[0] for i in range(width)],
                [False] * width,
            ],
            dtype=bool,
        ),
    )


def test_the_step_satisfies_both_trainer_protocols(tmp_path):
    """``train_ft`` drives ``trainer.TrainStep``; ``byte_train.Rung0Step`` is a different one.

    ``Rung0Step.accumulate(plan: BatchPlan) -> Rung0Losses`` belongs to the from-scratch
    byte decider that ``tools/rung0_toy_run.py`` drives. DevMap's edges for ``train_ft`` run
    ``train_ft -> _train -> TrainStep`` with no edge to ``Rung0Step`` anywhere, which is
    why this step implements the former.
    """
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    assert isinstance(step, TrainStep)
    assert isinstance(step, SpanScoringStep), (
        "without accumulate_span the loop refuses every SLOT_SPAN batch"
    )


def test_only_the_span_channel_off_opt_in_takes_a_zero_span_weight(tmp_path):
    """``tools/real_ft_run.py --shuffled-label``'s step: weight 0 through ``span_channel_off``
    and nowhere else, and then a span batch sends the span head no gradient at all -- the
    real span gold cannot pull the tower toward the defect the choice golds were shuffled
    away from. The span loss is still computed and logged, unweighted."""
    tower, _ = _tiny_tower(tmp_path)
    with pytest.raises(ValueError, match="span_weight must be positive"):
        QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64, span_weight=0.0)
    with pytest.raises(ValueError, match=r"takes span_weight 0\.0 exactly"):
        QwenDecisionStep(
            tower, seed=0, lr=1e-3, total_steps=1, max_width=64, span_weight=1.0,
            span_channel_off=True,
        )
    step = QwenDecisionStep(
        tower, seed=0, lr=1e-3, total_steps=1, max_width=64, span_weight=0.0,
        span_channel_off=True,
    )
    batch = _ft_batch(0)
    total = step.accumulate_span(batch, ft_supervision(batch))
    grads = [p.grad for p in step.span_head.parameters()]
    assert grads and all(g is None or float(g.abs().sum()) == 0.0 for g in grads)
    assert step.span_log[-1] > 0.0 and total == pytest.approx(step.letter_log[-1])


def test_a_real_backbone_trains_through_train_ft_and_the_loss_falls(tmp_path):
    """The binding this lane exists to make: real weights driven by the real loop.

    A tiny tower, but a real ``Qwen3_5TextModel`` loaded from a real snapshot layout, real
    ``fused_linear_cross_entropy`` against the real tied embedding, the real
    ``SpanPointerHead``, and ``qd_train.trainer.train_ft`` itself. Twelve repeats of one
    batch, so the floor is memorisation and a falling loss means the gradient reaches the
    parameters -- which is the claim, and the only one twelve steps can support.
    """
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    result = train_ft(
        (_ft_batch(i) for i in range(12)),
        epoch=0,
        step=step,
        control=_control(12),
        recorder=_recorder(tmp_path),
    )
    assert result.termination == "steps_exhausted"
    assert result.optimizer_steps == 12
    assert result.span_rows == 12, "one span row per batch"
    assert result.supervised_tokens == 12, "one letter answer per batch"

    losses = result.loss_log.losses()
    assert all(np.isfinite(losses)), f"a non-finite loss reached the log: {losses}"
    assert losses[-1] < losses[0], f"the combined FT loss must fall: {losses}"
    assert len(step.span_log) == 12 and any(v > 0 for v in step.span_log), (
        "the span channel contributed nothing, so the pointer head trained on nothing"
    )
    assert len(ledger.rows()) == 1 and ledger.rows()[0].status == "completed"


def test_a_batch_wider_than_the_bound_is_refused_rather_than_clipped(tmp_path):
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=4)
    with pytest.raises(ValueError, match="would move every row's target_index"):
        step.hidden(_ft_batch(0, width=6))


def test_the_state_is_tensor_refs_and_not_a_private_sidecar(tmp_path):
    """One sidecar convention in this repository, and it is ``Checkpoint``'s.

    ``TrainStep.state`` is documented JSON-serialisable and ``byte_train._portable``
    honours that literally -- 46.5 bytes per parameter as decimal text, which for this model
    is roughly 93 GB per checkpoint. ``run_control.TensorRef`` is the repository's answer,
    and ``Checkpoint.write`` splits those into a safetensors sidecar bounded by
    ``MAX_SIDECAR_BYTES``.

    So this step returns ``TensorRef``s and writes no file of its own. A private sidecar
    beside ``Checkpoint``'s would be a second convention for one thing, and the two would
    drift on exactly the question of which file holds the weights.
    """
    from qd_train.run_control import TensorRef

    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    state = step.state()

    assert state["vocab_size"] == TINY_VOCAB
    assert set(state["tower"]) == set(tower.model.state_dict())
    assert all(isinstance(v, TensorRef) for v in state["tower"].values())
    assert all(isinstance(v, TensorRef) for v in state["span_head"].values())

    before = list(tmp_path.rglob("*.safetensors"))
    step.state()
    assert list(tmp_path.rglob("*.safetensors")) == before, (
        "state() wrote a file of its own; the sidecar belongs to Checkpoint.write"
    )


def test_the_state_round_trips_the_weights_bit_exactly(tmp_path):
    """A resume that restores almost the right weights is the worst kind."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    state = step.state()

    original = tower.model.get_input_embeddings().weight.detach().clone()
    with torch.no_grad():
        tower.model.get_input_embeddings().weight.add_(1.0)
    assert not torch.equal(tower.model.get_input_embeddings().weight.detach(), original)

    step.load_state(state)
    assert torch.equal(tower.model.get_input_embeddings().weight.detach(), original)


def test_a_checkpoint_from_a_different_vocabulary_is_refused(tmp_path):
    """A different remap renumbers every row, so the weights mean something else."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    state = dict(step.state())
    state["vocab_size"] = TINY_VOCAB + 1
    with pytest.raises(BackboneContractViolation, match="renumbers every row"):
        step.load_state(state)


def test_the_state_survives_a_real_checkpoint_write_and_read(tmp_path):
    """The integration, not the contract read off a docstring.

    ``state()`` returns ``TensorRef``s on the strength of ``run_control``'s documented
    contract, and a contract satisfied on paper is not one that round-trips. This drives the
    real ``Checkpoint.write`` and ``Checkpoint.read``: the weights must land in the
    safetensors sidecar, the JSON body must stay small, and the revived weights must be
    bit-identical -- ``Checkpoint.read`` checks a per-tensor digest, so anything less fails
    there rather than here.
    """
    from qd_train.run_control import Checkpoint, LossLog, Position

    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    checkpoint = Checkpoint(
        position=Position(epoch=0, index=1),
        optimizer_step=1,
        seed=7,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=4, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog().snapshot(),
        consumed_digest="0" * 64,
        model_state=step.state(),
    )
    path = tmp_path / "ckpt.json"
    checkpoint.write(path)

    sidecars = [p for p in tmp_path.iterdir() if p.suffix == ".safetensors"]
    assert len(sidecars) == 1, f"expected one sidecar beside the JSON, got {sidecars}"
    n_params = sum(p.numel() for p in tower.model.parameters())
    assert path.stat().st_size < n_params, (
        f"the JSON body is {path.stat().st_size:,} bytes for {n_params:,} parameters; the "
        "weights did not go to the sidecar"
    )

    original = tower.model.get_input_embeddings().weight.detach().clone()
    with torch.no_grad():
        tower.model.get_input_embeddings().weight.add_(1.0)
    step.load_state(Checkpoint.read(path).model_state)
    assert torch.equal(tower.model.get_input_embeddings().weight.detach(), original)


def test_weights_read_without_the_optimizer_restore_the_trained_step(tmp_path):
    """The evaluation path: ``Checkpoint.read_weights`` + ``load_weights``, no optimizer.

    Restores the tower AND the span head exactly (both are what a verdict depends on), and
    refuses a span weight other than the one the checkpoint was trained with.
    """
    from qd_train.run_control import Checkpoint, LossLog, Position

    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    Checkpoint(
        position=Position(epoch=0, index=1),
        optimizer_step=1,
        seed=7,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=4, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog().snapshot(),
        consumed_digest="0" * 64,
        model_state=step.state(),
    ).write(tmp_path / "ckpt.json")
    embed = tower.model.get_input_embeddings().weight.detach().clone()
    head = [p.detach().clone() for p in step.span_head.parameters()]
    with torch.no_grad():
        tower.model.get_input_embeddings().weight.add_(1.0)
        for p in step.span_head.parameters():
            p.add_(1.0)

    weights, meta = Checkpoint.read_weights(tmp_path / "ckpt.json", subtrees=("tower", "span_head"))
    assert "optimizer" not in weights and meta["optimizer_step"] == 1
    step.load_weights(weights)
    assert torch.equal(tower.model.get_input_embeddings().weight.detach(), embed)
    restored = zip(step.span_head.parameters(), head, strict=True)
    assert all(torch.equal(p.detach(), q) for p, q in restored)

    other = dict(weights)
    other["span_weight"] = float(weights["span_weight"]) + 1.0
    with pytest.raises(BackboneContractViolation, match="span_weight"):
        step.load_weights(other)
    with pytest.raises(BackboneContractViolation, match="missing"):
        step.load_weights({"tower": weights["tower"]})


def test_a_state_that_lost_its_tensors_is_refused(tmp_path):
    """A body that came back without its sidecar is not a body with fewer tensors."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    state = dict(step.state())
    state["tower"] = {name: None for name in state["tower"]}
    with pytest.raises(BackboneContractViolation, match="not a TensorRef"):
        step.load_state(state)


# --- the optimizer the tower's spec names ------------------------------------------------


def test_the_step_builds_the_optimizer_its_towers_spec_names(tmp_path):
    """`load_text_tower` validates `optimizer` and `QwenDecisionStep` used to ignore it.

    A run could ask for `keeps_fp32_master=True`, have it accepted by the loader, and then
    train with bf16 moments anyway -- at beta2=0.999 those settle at 0.5 against a true 1.0
    after 383 steps. A spec that is checked and then discarded reads as a guarantee, which
    is worse than no spec at all.
    """
    from qd_train.memory import OptimizerSpec
    from qd_train.optim import MasterWeightAdamW

    master_spec = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    tower, _ = _tiny_tower(tmp_path, dtype="bf16", optimizer=master_spec)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    assert isinstance(step.optimizer, MasterWeightAdamW), (
        f"the tower's spec asked for an fp32 master and the step built "
        f"{type(step.optimizer).__name__}"
    )

    plain, _ = _tiny_tower(tmp_path / "plain", dtype="bf16", optimizer=ADAMW_BF16)
    plain_step = QwenDecisionStep(plain, seed=0, lr=1e-3, total_steps=1, max_width=64)
    assert isinstance(plain_step.optimizer, torch.optim.AdamW)
    assert not isinstance(plain_step.optimizer, MasterWeightAdamW)


def test_a_bf16_tower_under_the_master_recipe_trains_and_stays_bf16(tmp_path):
    """End to end through the real step: the tower trains, and the live weights stay bf16 --
    if they were promoted to fp32 every matmul after the first step would run at fp32 speed
    and the memory budget would be wrong by the size of the model."""
    from qd_train.memory import OptimizerSpec

    master_spec = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    tower, _ = _tiny_tower(tmp_path, dtype="bf16", optimizer=master_spec)
    step = QwenDecisionStep(tower, seed=0, lr=1e-2, total_steps=5, max_width=64)
    before = tower.model.get_input_embeddings().weight.detach().clone()

    batch = _ft_batch(0)
    for _ in range(5):
        step.accumulate(batch, ft_supervision(batch))
        step.apply(lr=1e-2)

    after = tower.model.get_input_embeddings().weight.detach()
    assert after.dtype == torch.bfloat16, f"the live weights became {after.dtype}"
    assert not torch.equal(after, before), "the tower did not train"
    assert torch.isfinite(after).all(), "the tower has non-finite weights after 5 steps"


@pytest.mark.parametrize("recipe", ["fp32", "master"])
def test_layerwise_lr_gives_layer_zero_a_tenth_of_the_schedule_after_apply(tmp_path, recipe):
    """RSI-Jev's layer-wise rule through the real step: layers 0..1 of the tiny tower at
    0.1x. The number that matters is the lr layer 0's parameters are STEPPED at, which is
    what `apply` writes -- before `apply_lr`, `apply` wrote the flat schedule into every group
    and the 0.1x split existed only until the first step."""
    from qd_train.memory import OptimizerSpec

    if recipe == "master":
        spec = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
        tower, _ = _tiny_tower(tmp_path, dtype="bf16", optimizer=spec)
    else:
        tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(
        tower, seed=0, lr=1e-3, total_steps=2, max_width=64,
        lower_layers_n=2, lower_lr_scale=0.1,
    )
    batch = _ft_batch(0)
    step.accumulate(batch, ft_supervision(batch))
    step.apply(lr=4e-3)
    by_name = {g["name"]: g for g in step.optimizer.param_groups}
    assert by_name["base_lower"]["lr"] == pytest.approx(4e-4)
    assert by_name["base"]["lr"] == pytest.approx(4e-3)
    # Membership, by identity against the tower's own names -- a split whose groups held
    # the wrong tensors would pass the two lines above.
    layer0 = {id(p) for n, p in tower.model.named_parameters() if n.startswith("layers.0.")}
    layer3 = {id(p) for n, p in tower.model.named_parameters() if n.startswith("layers.3.")}
    if recipe == "fp32":
        lower_ids = {id(p) for p in by_name["base_lower"]["params"]}
        base_ids = {id(p) for p in by_name["base"]["params"]}
        assert layer0 <= lower_ids and not layer3 & lower_ids and layer3 <= base_ids
    span_ids = {id(p) for p in step.span_head.parameters()}
    assert len(by_name["base"]["params"]) + len(by_name["base_lower"]["params"]) == len(
        list(tower.model.parameters())
    ) + len(span_ids)


def _written_checkpoint(tmp_path: Path, *, seed: int, total_steps: int = 4, step_at: int = 1,
                        dtype: str = "bf16") -> tuple[Path, QwenDecisionStep]:
    from qd_train.memory import OptimizerSpec
    from qd_train.run_control import Checkpoint, LossLog, Position

    spec = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    tower, _ = _tiny_tower(tmp_path / f"s{seed}", dtype=dtype, optimizer=spec)
    step = QwenDecisionStep(tower, seed=seed, lr=1e-3, total_steps=total_steps, max_width=64)
    ckpt = Checkpoint(
        position=Position(epoch=0, index=step_at),
        optimizer_step=step_at,
        seed=seed,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=total_steps, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog().snapshot(),
        consumed_digest=f"{seed:064x}",
        model_state=step.state(),
    )
    path = tmp_path / f"ckpt-{seed}-{total_steps}-{step_at}.json"
    ckpt.write(path)
    return path, step


def _ckpt_average():
    tools = REPO / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import ckpt_average

    return ckpt_average


def test_checkpoint_averaging_is_the_fp32_mean_cast_back_and_writes_weights_only(tmp_path):
    """Two seeds of one configuration: the tower is loaded from one snapshot so it is
    identical, the span head is seeded so it differs. The average of the tower is the tower,
    the average of the head is the mean, both come back in their own dtype, and the output
    carries no optimizer state and says it is not resumable."""
    from safetensors.torch import load_file

    tool = _ckpt_average()
    a, step_a = _written_checkpoint(tmp_path, seed=1)
    b, step_b = _written_checkpoint(tmp_path, seed=2)
    out = tmp_path / "avg" / "avg.safetensors"
    assert tool.main([str(a), str(b), "--out", str(out), "--ft-row-ids", "r1", "r2"]) == 0
    got = load_file(str(out))
    assert not any(k.startswith("optimizer") for k in got)
    emb = "tower.embed_tokens.weight"
    assert got[emb].dtype == torch.bfloat16
    assert torch.equal(got[emb], step_a.tower.model.get_input_embeddings().weight.detach())
    head = dict(step_a.span_head.named_parameters())
    other = dict(step_b.span_head.named_parameters())
    name = next(iter(head))
    expected = ((head[name].detach().float() + other[name].detach().float()) / 2).to(
        head[name].dtype
    )
    assert torch.equal(got[f"span_head.{name}"], expected)
    manifest = json.loads((out.parent / "avg.safetensors.manifest.json").read_text())
    assert manifest["resumable"] is False and manifest["n_inputs"] == 2
    assert manifest["ft_row_ids"] == ["r1", "r2"]
    with pytest.raises(SystemExit, match="already exists"):
        tool.main([str(a), str(b), "--out", str(out)])


def test_checkpoint_averaging_refuses_mismatched_schedules_steps_and_duplicates(tmp_path):
    from qd_train.run_control import Checkpoint

    tool = _ckpt_average()
    a, _ = _written_checkpoint(tmp_path, seed=1)
    other_schedule, _ = _written_checkpoint(tmp_path, seed=2, total_steps=8)
    other_step, _ = _written_checkpoint(tmp_path, seed=3, step_at=2)
    out = tmp_path / "x.safetensors"
    with pytest.raises(SystemExit, match="Different schedules"):
        tool.main([str(a), str(other_schedule), "--out", str(out)])
    with pytest.raises(SystemExit, match="optimizer step 2"):
        tool.main([str(a), str(other_step), "--out", str(out)])
    with pytest.raises(SystemExit, match="named twice"):
        tool.main([str(a), str(a), "--out", str(out)])
    same = Checkpoint.read(a)
    with pytest.raises(tool.AverageRefusal, match="identical weights"):
        tool.check_compatible([same, Checkpoint.read(a)], ["a", "a-copy"])
    # A different architecture: one tensor dropped from the second input's tower.
    broken = Checkpoint.read(a)
    tower_state = dict(broken.model_state["tower"])
    tower_state.pop(sorted(tower_state)[0])
    broken.model_state["tower"] = tower_state
    with pytest.raises(tool.AverageRefusal, match="different architecture"):
        tool.check_compatible([Checkpoint.read(a), broken], ["a", "broken"])
    assert not out.exists()


# --- averaging the fp32 masters (`ckpt_average.py --from masters`, J7) ----------------------

#: The master recipe's layout, as ``real_ft_run.optimizer_spec("bf16", "master")`` builds it.
MASTER_SPEC_ARGS = ("AdamW+master", 2, 4)


def _seeded_batch(seed: int, step: int) -> Batch:
    """``_ft_batch`` with tokens of its own per (seed, step): one snapshot, three different
    trajectories. On one shared batch the letter channel would move every seed's tower the
    same way and the masters of the three would only differ through the span head."""
    batch = _ft_batch(step)
    tokens = batch.tokens.copy()
    for row, n in enumerate(batch.lengths):
        tokens[row, :n] = (np.arange(n) * (seed + 2) + 3 * step + seed) % (TINY_VOCAB - 1) + 1
    return Batch(
        tokens=tokens, lengths=batch.lengths, bucket=batch.bucket, index=batch.index,
        slot_kind=batch.slot_kind, target_index=batch.target_index,
        span_target=batch.span_target, line_starts=batch.line_starts,
    )


def _master_checkpoint(
    tmp_path: Path, *, seed: int, steps: int = 3, freeze: str | None = None,
    name: str | None = None,
) -> tuple[Path, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A ``--optimizer master`` checkpoint after ``steps`` real steps on batches of its own.

    Returns ``(path, fp32 master by parameter name, live weight by name)``. The names come
    from the live step -- ``MasterWeightAdamW._live`` paired with ``_masters`` by position,
    each live parameter named by identity through ``named_parameters`` -- never from the
    checkpoint, so they are ground truth for the mapping the average has to recover.
    ``freeze`` names a tower parameter taken out of the optimizer: it gets no master.
    """
    from qd_train.memory import OptimizerSpec
    from qd_train.optim import MasterWeightAdamW
    from qd_train.run_control import Checkpoint, LossLog, Position

    spec = OptimizerSpec(*MASTER_SPEC_ARGS, keeps_fp32_master=True)
    tower, _ = _tiny_tower(tmp_path, dtype="bf16", optimizer=spec)
    if freeze is not None:
        dict(tower.model.named_parameters())[freeze].requires_grad_(False)
    step = QwenDecisionStep(tower, seed=seed, lr=1e-2, total_steps=steps, max_width=64)
    assert isinstance(step.optimizer, MasterWeightAdamW)
    for k in range(steps):
        batch = _seeded_batch(seed, k)
        step.accumulate_span(batch, ft_supervision(batch))
        step.apply(lr=1e-2)
    by_id = {id(p): f"tower.{n}" for n, p in tower.model.named_parameters()}
    by_id.update({id(p): f"span_head.{n}" for n, p in step.span_head.named_parameters()})
    masters = {
        by_id[id(live)]: master.detach().to("cpu").clone()
        for live, master in zip(step.optimizer._live, step.optimizer._masters, strict=True)
    }
    lives = {f"tower.{n}": t.detach().to("cpu").clone()
             for n, t in tower.model.state_dict().items()}
    lives.update({f"span_head.{n}": t.detach().to("cpu").clone()
                  for n, t in step.span_head.state_dict().items()})
    path = tmp_path / (name or f"epoch-seed{seed}-cpu.json")
    Checkpoint(
        position=Position(epoch=0, index=steps),
        optimizer_step=steps,
        seed=seed,
        schedule=LRSchedule(peak_lr=1e-2, total_steps=steps, warmup_steps=1, min_lr=1e-3),
        loss_log=LossLog().snapshot(),
        consumed_digest=f"{seed + 1:064x}",
        model_state=step.state(),
    ).write(path)
    return path, masters, lives


def test_read_weights_reads_a_nested_path_and_not_its_siblings(tmp_path):
    """``("optimizer", "masters")`` is the fp32 masters and not the Adam moments beside them:
    7.04 GiB of a real checkpoint's 21.12 GiB of optimizer state. The masters come back at
    their own position, byte for byte what a full read gives."""
    from qd_train.run_control import Checkpoint

    path, _, _ = _master_checkpoint(tmp_path, seed=0, steps=2)
    weights, meta = Checkpoint.read_weights(path, subtrees=("tower", ("optimizer", "masters")))
    assert set(weights["optimizer"]) == {"masters"}, "the moments were read too"
    full = Checkpoint.read(path).model_state["optimizer"]["masters"]
    assert [m.digest() for m in weights["optimizer"]["masters"]] == [m.digest() for m in full]
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert meta["payload_digest"] == raw["payload_digest"]
    assert meta["schedule"] == raw["schedule"]
    with pytest.raises(ValueError, match="no 'nope'"):
        Checkpoint.read_weights(path, subtrees=(("optimizer", "nope"),))


def test_a_masters_average_takes_each_parameters_own_master_by_name(tmp_path):
    """The J7 average: per parameter, the mean of the three inputs' fp32 masters FOR THAT
    NAME, cast to the tower's dtype once; a parameter with no master (frozen here) averages
    its tower tensor. Ground truth is the live steps' own pairing, so a master put under
    any other name -- a positional mapping onto the sorted names, a span-head master under
    a tower name -- fails the equality below."""
    from safetensors.torch import load_file

    tool = _ckpt_average()
    frozen = "norm.weight"
    made = [
        _master_checkpoint(tmp_path / f"s{s}", seed=s, freeze=frozen) for s in (0, 1, 2)
    ]
    masters = [m for _, m, _ in made]
    lives = [w for _, _, w in made]
    assert set(masters[0]) == set(masters[1]) == set(masters[2])
    assert f"tower.{frozen}" not in masters[0]
    out = tmp_path / "avg" / "avg.safetensors"
    argv = [*(str(p) for p, _, _ in made), "--out", str(out), "--from", "masters",
            "--ft-row-ids", "r0", "r1", "r2"]
    assert tool.main(argv) == 0
    got = load_file(str(out))
    manifest = json.loads(tool.manifest_path(out).read_text(encoding="utf-8"))
    sources = manifest["tensor_sources"]
    assert set(got) == set(sources) == set(lives[0])
    moved_by_the_master = 0
    for name, tensor in got.items():
        live_mean = (sum(w[name].to(torch.float32) for w in lives) / 3).to(tensor.dtype)
        if name not in masters[0]:
            assert sources[name] == "tower" and name not in manifest["master_index"]
            assert torch.equal(tensor, live_mean), name
            continue
        assert sources[name] == "master", name
        acc = torch.zeros(tensor.shape, dtype=torch.float64)
        for m in masters:
            acc += m[name].to(torch.float64)
        expected = (acc / 3).to(torch.float32).to(tensor.dtype)
        assert torch.equal(tensor, expected), f"{name} is not the mean of its own masters"
        assert tensor.dtype == lives[0][name].dtype, f"{name} came back {tensor.dtype}"
        moved_by_the_master += not torch.equal(tensor, live_mean)
    assert sources[f"tower.{frozen}"] == "tower"
    assert moved_by_the_master > 0, (
        "no tensor differs from the bf16 average, so this test cannot tell masters from tower"
    )
    assert manifest["from"] == "masters" and manifest["resumable"] is False
    assert manifest["accumulator"] == "float64"
    assert manifest["ft_row_ids"] == ["r0", "r1", "r2"]
    for (path, _, _), record in zip(made, manifest["inputs"], strict=True):
        assert record["payload_digest"] == json.loads(path.read_text())["payload_digest"]
        assert record["resolved_path"] == str(path.resolve())
    assert len(manifest["master_index"]) == len(masters[0])


def test_a_masters_average_refuses_inputs_with_no_masters(tmp_path):
    """A checkpoint trained without ``--optimizer master`` has no fp32 masters; averaging
    its bf16 tower under the name 'masters' would record a source that was never read."""
    from qd_train.run_control import Checkpoint, LossLog, Position

    tool = _ckpt_average()
    paths = []
    for seed in (1, 2):
        tower, _ = _tiny_tower(tmp_path / f"p{seed}")  # fp32 under plain AdamW: no master
        step = QwenDecisionStep(tower, seed=seed, lr=1e-3, total_steps=4, max_width=64)
        assert "masters" not in step.state()["optimizer"]
        path = tmp_path / f"epoch-seed{seed}-cpu.json"
        Checkpoint(
            position=Position(epoch=0, index=1), optimizer_step=1, seed=seed,
            schedule=LRSchedule(peak_lr=1e-3, total_steps=4, warmup_steps=1, min_lr=1e-4),
            loss_log=LossLog().snapshot(), consumed_digest=f"{seed:064x}",
            model_state=step.state(),
        ).write(path)
        paths.append(str(path))
    out = tmp_path / "x.safetensors"
    with pytest.raises(SystemExit, match="no fp32 masters"):
        tool.main([*paths, "--out", str(out), "--from", "masters"])
    assert not out.exists()


def _ref(tensor: torch.Tensor):
    from qd_train.run_control import TensorRef

    host = tensor.detach().contiguous()
    return TensorRef(
        dtype=str(host.dtype).removeprefix("torch."), shape=tuple(host.shape),
        data=host.reshape(-1).view(torch.uint8).numpy().tobytes(),
    )


def test_the_master_mapping_refuses_a_master_it_cannot_place_exactly_once():
    """Each master must round to exactly one tower or span-head tensor of its shape, and no
    two masters to one. A guess between two candidates would be a coin flip on whose
    weights land where."""
    tool = _ckpt_average()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    def weights(tower: dict, masters: list) -> dict:
        return {
            "tower": {k: _ref(v.to(torch.bfloat16)) for k, v in tower.items()},
            "span_head": {"h": _ref(torch.tensor([7.0, 8.0]))},
            "optimizer": {"masters": [_ref(m) for m in masters]},
        }

    assert tool.master_names(weights({"a": x, "b": y}, [x, y, torch.tensor([7.0, 8.0])]),
                             where="ok") == ["tower.a", "tower.b", "span_head.h"]
    with pytest.raises(tool.AverageRefusal, match="ambiguous"):
        tool.master_names(weights({"a": x, "b": x}, [x]), where="twins")
    with pytest.raises(tool.AverageRefusal, match="no tower or span_head tensor"):
        tool.master_names(weights({"a": x}, [y]), where="stray")
    with pytest.raises(tool.AverageRefusal, match="two masters"):
        tool.master_names(weights({"a": x}, [x, x]), where="double")


def test_a_replay_checkpoint_resumed_into_a_bare_step_is_refused(tmp_path):
    """A checkpoint written under PriorKLReplay carries "replay". Resumed without the replay
    flags, the bare step used to ignore the key and continue without the replay term --
    and consumed_digest, which covers only the training source, could not notice."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    state = {**step.state(), "replay": {"micro_batches": 6, "replayed": 1}}
    with pytest.raises(BackboneContractViolation, match=r"\['replay'\]"):
        step.load_state(state)


@pytest.mark.parametrize(
    "saved_kwargs",
    [{"beta2": 0.95}, {"lower_layers_n": 2, "lower_lr_scale": 0.1}],
)
def test_a_checkpoint_taken_under_another_optimizer_recipe_is_refused(tmp_path, saved_kwargs):
    """torch's load_state_dict overwrites betas and every extra group key with the saved
    ones, so this resume used to succeed and train on the CHECKPOINT's beta2 or split while
    the ledger recorded the resuming run's flags."""
    tower, _ = _tiny_tower(tmp_path / "a")
    saved = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=2, max_width=64,
                             **saved_kwargs)
    state = saved.state()
    tower_b, _ = _tiny_tower(tmp_path / "b")
    plain = QwenDecisionStep(tower_b, seed=0, lr=1e-3, total_steps=2, max_width=64)
    before = plain.tower.model.get_input_embeddings().weight.detach().clone()
    with pytest.raises(BackboneContractViolation, match="optimizer groups"):
        plain.load_state(state)
    assert torch.equal(plain.tower.model.get_input_embeddings().weight, before), (
        "the refusal came after the weights had already been overwritten"
    )
    same = QwenDecisionStep(_tiny_tower(tmp_path / "c")[0], seed=0, lr=1e-3, total_steps=2,
                            max_width=64, **saved_kwargs)
    same.load_state(state)


def test_a_replay_wrapper_state_survives_a_real_checkpoint_write_and_read(tmp_path):
    """The wrapper's state through Checkpoint.write/read -- the JSON body and the sidecar --
    and back into a fresh wrapper, which restores its cursor and hands the step its own
    keys only."""
    from qd_train.replay import PriorCache, PriorKLReplay
    from qd_train.run_control import Checkpoint, LossLog, Position

    tower, _ = _tiny_tower(tmp_path / "a")
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=4, max_width=64)
    batches = [_ft_batch(0)]
    cache = PriorCache.build(step, batches, letter_ids=(1, 2, 3), key={"base": "tiny"})
    wrapped = PriorKLReplay(step, batches=batches, cache=cache, weight=0.1, every=1)
    wrapped.accumulate(batches[0], ft_supervision(batches[0]))
    wrapped.apply(lr=1e-3)
    path = tmp_path / "ckpt.json"
    Checkpoint(
        position=Position(epoch=0, index=1), optimizer_step=1, seed=0,
        schedule=LRSchedule(peak_lr=1e-3, total_steps=4, warmup_steps=1, min_lr=1e-4),
        loss_log=LossLog().snapshot(), consumed_digest="0" * 64, model_state=wrapped.state(),
    ).write(path)
    tower_b, _ = _tiny_tower(tmp_path / "b")
    fresh = PriorKLReplay(
        QwenDecisionStep(tower_b, seed=0, lr=1e-3, total_steps=4, max_width=64),
        batches=batches, cache=cache, weight=0.1, every=1,
    )
    fresh.load_state(Checkpoint.read(path).model_state)
    assert (fresh.micro_batches, fresh.replayed) == (1, 1)
    assert torch.equal(
        tower_b.model.get_input_embeddings().weight, tower.model.get_input_embeddings().weight
    )


def test_a_layerwise_scale_without_a_split_is_refused(tmp_path):
    tower, _ = _tiny_tower(tmp_path)
    with pytest.raises(ValueError, match="lower_layers_n=0"):
        QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64, lower_lr_scale=0.1)


# --- resume, through the REAL step -------------------------------------------------------
#
# `trainer.py` promises a bit-exact resume and `test_trainer.py` proves it -- for `TinyStep`,
# a test double. Every resume test in this repository uses one. These drive the same
# property through `QwenDecisionStep`, which is the step a full train actually runs.


def _twelve_step_losses(tmp_path: Path, *, legs: tuple[int, ...]) -> list[float]:
    """Run 12 FT steps as ``legs`` consecutive runs, resuming across each boundary.

    ``legs=(12,)`` is the uninterrupted reference. ``legs=(6, 6)`` is the same schedule
    across one kill. Every leg builds a FRESH step from the same deterministic snapshot, so
    anything the checkpoint fails to carry shows up as a divergence rather than being
    quietly supplied by the object that survived in memory.
    """
    from qd_train.run_control import Checkpoint

    resume_from = None
    consumed = 0
    losses: list[float] = []
    for leg, n in enumerate(legs):
        tower, _ = _tiny_tower(tmp_path)
        step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
        result = train_ft(
            (_ft_batch(i) for i in range(consumed + n)),
            epoch=0,
            step=step,
            control=_control(12),
            recorder=_recorder(tmp_path / f"leg{leg}"),
            resume_from=resume_from,
        )
        losses = result.loss_log.losses()
        consumed += n
        if leg + 1 < len(legs):
            # Through a real file, not the live object: a checkpoint that only works while
            # the process that wrote it is still alive is not a checkpoint.
            path = result.checkpoint.write(tmp_path / f"ckpt{leg}" / "run.json")
            resume_from = Checkpoint.read(path)
    return losses


def test_the_real_step_resumes_the_trajectory_it_was_cut_from(tmp_path):
    """The property a full train rests on, driven through the real step for the first time.

    A rented box interrupted at hour three has to come back to the run it was having, not to
    a model with the right weights and an optimizer that forgot everything. The difference
    is invisible in the weights -- they restore exactly -- and shows up only in what the
    next step does with them.
    """
    reference = _twelve_step_losses(tmp_path / "whole", legs=(12,))
    across_a_kill = _twelve_step_losses(tmp_path / "split", legs=(6, 6))
    assert len(reference) == 12
    assert across_a_kill == pytest.approx(reference, rel=1e-6, abs=1e-8), (
        "a run resumed at the half-way point did not reproduce the uninterrupted "
        "trajectory: the checkpoint did not carry everything the next step reads"
    )


def test_the_checkpoint_carries_the_optimizer_and_not_only_the_weights(tmp_path):
    """Stated directly, because the trajectory test above says only that something is
    missing and not what.

    ``torch.optim.AdamW`` keeps ``exp_avg`` and ``exp_avg_sq`` per parameter and a step
    count that drives bias correction. A resume that restores weights and leaves those at
    zero re-enters warm-up on a model that is no longer warming up: the first step after
    every resume boundary is taken with an empty second moment, which is the largest step
    the schedule can produce, applied to the most trained weights in the run.
    """
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=4, max_width=64)
    train_ft(
        (_ft_batch(i) for i in range(4)),
        epoch=0,
        step=step,
        control=_control(4),
        recorder=_recorder(tmp_path / "run"),
    )
    state = step.state()
    assert "optimizer" in state, (
        "QwenDecisionStep.state() carries the tower and the span head and not the "
        "optimizer, so every resume silently restarts AdamW's moments from zero"
    )


def test_the_master_recipe_resumes_too_including_its_fp32_masters(tmp_path):
    """The recipe a LONG run has to use, which is a different state shape.

    ``build_optimizer`` refuses a schedule past step 384 on bf16 moments, so any full train
    takes the ``keeps_fp32_master=True`` branch -- and ``MasterWeightAdamW.state_dict``
    returns ``{"inner": ..., "masters": ...}`` where ``torch.optim.AdamW`` returns
    ``{"state": ..., "param_groups": ...}``. A serialiser that handled only the second
    shape would work on every short test here and fail on the first real checkpoint.

    The masters specifically matter: ``MasterWeightAdamW.load_state_dict`` refuses a state
    without them, because rebuilding masters from the bf16 parameters throws away exactly
    the precision the class exists to keep, and does it silently.
    """
    from qd_train.memory import OptimizerSpec
    from qd_train.optim import MasterWeightAdamW
    from qd_train.run_control import Checkpoint, TensorRef

    master_spec = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)

    def leg(where: Path, n: int, resume_from):
        tower, _ = _tiny_tower(where, dtype="bf16", optimizer=master_spec)
        step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=8, max_width=64)
        assert isinstance(step.optimizer, MasterWeightAdamW)
        return step, train_ft(
            (_ft_batch(i) for i in range(n)),
            epoch=0,
            step=step,
            control=_control(8),
            recorder=_recorder(where / "rec"),
            resume_from=resume_from,
        )

    _, whole = leg(tmp_path / "whole", 8, None)
    first_step, first = leg(tmp_path / "split", 4, None)

    # The masters are fp32 while the live parameters are bf16, and that is the whole point:
    # a checkpoint that carried only the live weights would come back rounded.
    body = first_step.state()["optimizer"]
    masters = body["masters"]
    assert masters, "the master copies did not reach the checkpoint body"
    assert all(isinstance(m, TensorRef) for m in masters)
    assert {m.dtype for m in masters} == {"float32"}

    path = first.checkpoint.write(tmp_path / "ckpt" / "run.json")
    _, resumed = leg(tmp_path / "split", 8, Checkpoint.read(path))
    assert resumed.loss_log.losses() == pytest.approx(
        whole.loss_log.losses(), rel=1e-6, abs=1e-8
    )


def test_a_checkpoint_without_its_optimizer_is_refused_rather_than_half_loaded(tmp_path):
    """A weights-only state loads cleanly and resumes a run whose moments are zero, which
    looks like a working resume and is not one. That is the failure this whole pair of
    tests exists for, so it is refused by name."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=2, max_width=64)
    weights_only = {k: v for k, v in step.state().items() if k != "optimizer"}
    with pytest.raises(BackboneContractViolation, match="optimizer"):
        step.load_state(weights_only)


def test_the_optimizers_integer_keys_survive_the_json_body(tmp_path):
    """``torch.optim.Optimizer.state_dict()`` keys its ``state`` by parameter INDEX, and the
    checkpoint body refuses a non-string key outright -- so the indices are stringified on
    the way out. Restoring them as strings would hand torch a state whose parameters it
    cannot match, and it does not complain about that."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=2, max_width=64)
    train_ft(
        (_ft_batch(i) for i in range(2)),
        epoch=0,
        step=step,
        control=_control(2),
        recorder=_recorder(tmp_path / "rec"),
    )
    body = step.state()["optimizer"]
    assert body["state"], "the optimizer had taken steps but carried no per-parameter state"
    assert all(isinstance(k, str) for k in body["state"])
    revived = step._revive_optimizer(body)
    assert revived["state"], "the revived state lost its entries"
    assert all(isinstance(k, int) for k in revived["state"]), (
        "the parameter indices came back as strings; torch would not match them to "
        "parameters and would not say so"
    )


# -- the seed has to determine what the run starts from -----------------------------------
#
# GAP-REAL-STEP-INITIALISES-ITS-SPAN-HEAD-FROM-AN-UNSEEDED-RNG. `RealFtStep.__init__`, the
# STAND-IN branch of tools/real_ft_run.py, calls `torch.manual_seed(seed)` on its first
# line. This class -- the REAL branch, and the one every GH200 measurement ran on -- called
# nothing, so `seed` in the protocol, in the checkpoint name and in every ledger row
# governed the batch order and not the parameters. Six runs of one configuration on the
# GH200 opened at total losses of 253, 279, 298, 335, 383 and 490.


def test_two_steps_at_one_seed_start_from_the_same_parameters(tmp_path):
    """The claim `seed` makes in every ledger row this step writes.

    Built one after the other in ONE process, which is how a sweep runs them: the second
    construction draws from a global RNG the first one already advanced, so without seeding
    here it cannot land on the same head twice however the caller was written.
    """
    import torch

    tower_a, _ = _tiny_tower(tmp_path / "a")
    tower_b, _ = _tiny_tower(tmp_path / "b")
    a = QwenDecisionStep(tower_a, seed=7, lr=1e-3, total_steps=1, max_width=64)
    b = QwenDecisionStep(tower_b, seed=7, lr=1e-3, total_steps=1, max_width=64)

    for (name, pa), (_, pb) in zip(
        a.span_head.named_parameters(), b.span_head.named_parameters(), strict=True
    ):
        assert torch.equal(pa, pb), (
            f"span head parameter {name!r} differs between two steps at seed 7: the seed "
            "names the batch order and not the model the run starts from"
        )


def test_a_different_seed_starts_from_different_parameters(tmp_path):
    """The other half. A seed that changed nothing would satisfy the test above trivially
    -- a constant initialiser passes 'same seed, same parameters' and makes the seed a
    decoration."""
    import torch

    tower_a, _ = _tiny_tower(tmp_path / "a")
    tower_b, _ = _tiny_tower(tmp_path / "b")
    a = QwenDecisionStep(tower_a, seed=7, lr=1e-3, total_steps=1, max_width=64)
    b = QwenDecisionStep(tower_b, seed=8, lr=1e-3, total_steps=1, max_width=64)

    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(
            a.span_head.parameters(), b.span_head.parameters(), strict=True
        )
    ), "seeds 7 and 8 produced the same span head, so the seed selects nothing"


def test_the_seed_survives_an_arbitrarily_advanced_global_rng(tmp_path):
    """The case that makes this a constructor's job rather than a driver's.

    A driver could seed before each construction and get the same guarantee -- until the
    next caller forgets, or until something between the seeding and the construction draws
    a number. Here the ambient stream is advanced by a different amount before each build,
    which is what a real process does (a probe, a shuffled plan, a dropout mask), and the
    two steps must still agree.
    """
    import torch

    tower_a, _ = _tiny_tower(tmp_path / "a")
    tower_b, _ = _tiny_tower(tmp_path / "b")

    torch.manual_seed(1234)
    torch.randn(17)
    a = QwenDecisionStep(tower_a, seed=3, lr=1e-3, total_steps=1, max_width=64)

    torch.manual_seed(999)
    torch.randn(1_003)
    b = QwenDecisionStep(tower_b, seed=3, lr=1e-3, total_steps=1, max_width=64)

    for (name, pa), (_, pb) in zip(
        a.span_head.named_parameters(), b.span_head.named_parameters(), strict=True
    ):
        assert torch.equal(pa, pb), f"span head parameter {name!r} followed the ambient RNG"


def test_the_step_records_the_seed_it_was_built_with(tmp_path):
    """So a step can be asked, rather than the caller being trusted to have passed what it
    wrote into the ledger row beside it."""
    tower, _ = _tiny_tower(tmp_path)
    assert QwenDecisionStep(tower, seed=11, lr=1e-3, total_steps=1, max_width=64).seed == 11


def test_the_real_step_seeds_the_same_way_the_stand_in_does(tmp_path):
    """One behaviour, one spelling, checked across the two branches of the same tool.

    The defect was not that seeding is hard; it is that the tool has two backbone branches
    and only one of them did it. A test that reads both sources is what stops the next
    branch from being added without it.
    """
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert "torch.manual_seed(seed)" in source, "the stand-in branch stopped seeding"
    backbone_source = (
        REPO / "python" / "qd_train" / "backbone.py"
    ).read_text(encoding="utf-8")
    assert "torch.manual_seed(seed)" in backbone_source, (
        "the real branch does not seed, so every run on it is a fresh draw regardless of "
        "the seed its ledger row records"
    )


# -- the kernel and the library version, both of which decide a number ---------------------
#
# GAP-ATTENTION-IMPLEMENTATION-AND-TRANSFORMERS-VERSION-ARE-NOT-RECORDED. Two quantities
# that change a result, chosen by the library rather than by this project, and recorded
# nowhere: `Qwen3_5TextModel(text_config)` took whatever `config._attn_implementation`
# resolved to, and `Environment.detect` auto-detected torch's version while leaving
# transformers' as a parameter defaulting to "unknown". This project's two hosts run
# transformers 5.12.1 (Mac) and 5.17.0 (GH200); every ft row from either says "unknown".


def test_the_tower_reports_the_attention_kernel_it_resolved(tmp_path):
    """Not the string it was handed -- the one the model ended up with. They agree today,
    and asking the model is what keeps the row true on the day they stop."""
    tower, _ = _tiny_tower(tmp_path, attn_implementation="eager")
    assert tower.attn_implementation == "eager"
    assert str(tower.model.config._attn_implementation) == "eager"


def test_two_kernels_are_two_towers(tmp_path):
    """A field that reported the same value whichever kernel ran would satisfy the test
    above and record nothing."""
    eager, _ = _tiny_tower(tmp_path / "e", attn_implementation="eager")
    sdpa, _ = _tiny_tower(tmp_path / "s", attn_implementation="sdpa")
    assert eager.attn_implementation != sdpa.attn_implementation


def test_an_unimplemented_kernel_is_refused_rather_than_quietly_ignored(tmp_path):
    """Set on the config, so transformers validates it. A name this build does not
    implement has to raise: silently falling back would put a kernel in the recipe that did
    not run, which is worse than the unrecorded default it replaced."""
    # `is not supported` and not just the argument name: a loader that had never heard of
    # the argument raises TypeError naming it too, so the looser pattern passes against the
    # code this test exists to reject -- the same trap that let a --span-weight test pass
    # against a tool with no such flag.
    with pytest.raises(ValueError, match="is not supported"):
        _tiny_tower(tmp_path, attn_implementation="no_such_kernel")


def test_the_loader_requires_a_kernel_rather_than_choosing_one(tmp_path):
    """The third required argument on this function, after `gradient_checkpointing` and
    `optimizer`, and for the reason both of those carry: an opt-in record of an invisible
    choice is a record that is off."""
    import inspect

    parameter = inspect.signature(load_text_tower).parameters["attn_implementation"]
    assert parameter.default is inspect.Parameter.empty, (
        "a default here is this module deciding the arithmetic and not recording that it did"
    )


def test_the_remap_carries_the_kernel_through(tmp_path):
    """`remap_text_tower` rebuilds the TextTower around a sliced embedding. A field dropped
    there would read as the loader's default on exactly the towers that get trained, since
    the real path always remaps."""
    tower, _ = _tiny_tower(tmp_path, attn_implementation="eager")
    kept = list(range(0, TINY_VOCAB, 2))
    remapped = remap_text_tower(tower, _tiny_remap(kept, []))
    assert remapped.attn_implementation == "eager"


def test_the_environment_detects_transformers_rather_than_saying_unknown() -> None:
    """The asymmetry that was doing the damage: torch's version was detected and always
    right, transformers' was a parameter defaulting to "unknown" and was right only when a
    caller remembered. Only the tokenizer pipeline did, so every training row says
    "unknown" -- about a library whose version changes results, across two hosts that run
    different ones."""
    import transformers

    env = Environment.detect(device="cpu")
    assert env.transformers_sha != "unknown"
    assert transformers.__version__ in env.transformers_sha


def test_a_caller_that_knows_a_sha_still_overrides_the_detected_version() -> None:
    """Detection is for the caller that does not know, not a refusal to be told. A run
    against a transformers built from source has a commit and not a release number."""
    env = Environment.detect(transformers_sha="deadbee", device="cpu")
    assert env.transformers_sha == "deadbee"


# -- resuming onto a different corpus order, through the REAL step -------------------------
#
# `HANDOFF/resume-2026-09-20.md` and every handoff since have carried this as open: the
# trainer rehashes the consumed prefix and `test_trainer.py` covers the refusal -- with
# `TinyStep`, a test double, only. The end-to-end GH200 proof covered the happy path. So the
# refusal that protects a full train from resuming onto the wrong batches had never been
# exercised against a step that loads real weights and a real optimizer.
#
# Not the driver: `tools/real_ft_run.py` reconstructs its labels from git history and needs
# a shard set, so driving it here would be a minutes-long test of the pipeline. This closes
# the half that was actually untested -- the real step -- and the handoff says which half
# remains.


def _resumable_checkpoint(tmp_path: Path, *, n: int = 6):
    """Six real FT steps, checkpointed through a file. The same shape `_twelve_step_losses`
    uses for its first leg, kept separate so a change to the trajectory test cannot silently
    change what this one resumes from."""
    from qd_train.run_control import Checkpoint

    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    result = train_ft(
        (_ft_batch(i) for i in range(n)),
        epoch=0,
        step=step,
        control=_control(12),
        recorder=_recorder(tmp_path / "first"),
    )
    return Checkpoint.read(result.checkpoint.write(tmp_path / "ckpt" / "run.json"))


def test_the_real_step_refuses_a_resume_onto_a_different_corpus_order(tmp_path):
    """Landing on the index is not landing on the batch.

    The checkpoint records how many batches were consumed AND their hash. A source that
    yields the same COUNT of batches with different contents satisfies every positional
    check -- same seed, same epoch, same index -- and is a different corpus. Here the second
    leg's batches are one token wider, which is what a changed `batch_tokens` or a
    regenerated shard set does to them.
    """
    resume_from = _resumable_checkpoint(tmp_path)

    tower, _ = _tiny_tower(tmp_path / "second")
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    with pytest.raises(TrainerContractViolation, match="not the one the checkpoint was taken from"):
        train_ft(
            (_ft_batch(i, width=7) for i in range(12)),
            epoch=0,
            step=step,
            control=_control(12),
            recorder=_recorder(tmp_path / "second-run"),
            resume_from=resume_from,
        )


def test_the_refusal_names_both_digests_rather_than_saying_mismatch(tmp_path):
    """A refusal that says only "mismatch" leaves the operator to guess whether the
    checkpoint or the source is the wrong one, at hour three of a rented box. Both hashes in
    the message are what makes it actionable."""
    resume_from = _resumable_checkpoint(tmp_path)

    tower, _ = _tiny_tower(tmp_path / "second")
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    with pytest.raises(TrainerContractViolation) as excinfo:
        train_ft(
            (_ft_batch(i, width=7) for i in range(12)),
            epoch=0,
            step=step,
            control=_control(12),
            recorder=_recorder(tmp_path / "second-run"),
            resume_from=resume_from,
        )
    message = str(excinfo.value)
    assert resume_from.consumed_digest in message, "the refusal does not name what was expected"
    assert message.count("hash to") == 1 and "batch_tokens" in message, (
        "the refusal should name the hash it computed and the knob that changes it"
    )


def test_the_same_order_still_resumes_so_the_refusal_is_the_order(tmp_path):
    """The control. Without it, a refusal on every resume would pass the two tests above --
    and a trainer that refused all resumes would be worse than one that accepted the wrong
    ones, because it would be found immediately and reverted."""
    resume_from = _resumable_checkpoint(tmp_path)

    tower, _ = _tiny_tower(tmp_path / "second")
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    result = train_ft(
        (_ft_batch(i) for i in range(12)),
        epoch=0,
        step=step,
        control=_control(12),
        recorder=_recorder(tmp_path / "second-run"),
        resume_from=resume_from,
    )
    # 12, not 6: the loss log carries the whole trajectory across a resume, which is what
    # `test_the_real_step_resumes_the_trajectory_it_was_cut_from` compares against the
    # uninterrupted reference. What this control asserts is that the same order does NOT
    # raise -- deliberately overlapping that test, because a trainer that refused every
    # resume would satisfy the two refusal tests above, and would be worse than one that
    # accepted the wrong ones.
    assert len(result.loss_log.losses()) == 12, (
        "a resume onto the order it was cut from should complete the schedule"
    )


# --- selective checkpointing (Tier A: semantics-preserving) ----------------------------------


def _trajectory(
    where: Path, *, skip: int, n: int = 6, gradient_checkpointing: bool = True,
    lower_layers_n: int = 0, lower_lr_scale: float = 1.0,
):
    """``n`` real FT steps through ``train_ft``; every number a Tier-A comparison reads."""
    tower, _ = _tiny_tower(
        where, gradient_checkpointing=gradient_checkpointing, checkpoint_skip_layers=skip
    )
    step = QwenDecisionStep(
        tower, seed=0, lr=1e-3, total_steps=n, max_width=64,
        lower_layers_n=lower_layers_n, lower_lr_scale=lower_lr_scale,
    )
    result = train_ft(
        (_ft_batch(i) for i in range(n)),
        epoch=0,
        step=step,
        control=_control(n),
        recorder=_recorder(where / "rec"),
    )
    return {
        "digest": result.loss_log.digest(),
        "losses": result.loss_log.losses(),
        "letter": list(step.letter_log),
        "span": list(step.span_log),
        "consumed": result.checkpoint.consumed_digest,
    }


def test_selective_checkpointing_reaches_exactly_the_named_layers(tmp_path):
    from qd_train.backbone import uncheckpointed_layers

    tower, _ = _tiny_tower(tmp_path, checkpoint_skip_layers=2)
    types = list(tower.model.config.layer_types)
    assert tower.checkpoint_skip_layers == uncheckpointed_layers(2, types) == (1, 3)
    flags = [bool(layer.gradient_checkpointing) for layer in tower.model.layers]
    assert flags == [True, False, True, False]
    assert tower.to_json()["checkpoint_skip_layers"] == [1, 3]


def test_the_skip_order_takes_full_attention_layers_first_then_spreads_linear_ones():
    """Qwen3.5-2B's 24 layers are three linear-attention layers then one full-attention
    layer, six times. Full-attention layers are dearest to recompute per byte kept, so they
    go first; past six, linear layers are spread evenly. Pure in (n, layer_types)."""
    from qd_train.backbone import uncheckpointed_layers

    types = ["linear_attention"] * 3 + ["full_attention"]
    types = types * 6
    assert uncheckpointed_layers(0, types) == ()
    assert uncheckpointed_layers(6, types) == (3, 7, 11, 15, 19, 23)
    assert uncheckpointed_layers(3, types) == (7, 15, 23)
    assert uncheckpointed_layers(8, types) == (3, 5, 7, 11, 15, 17, 19, 23)
    assert uncheckpointed_layers(24, types) == tuple(range(24))
    with pytest.raises(ValueError):
        uncheckpointed_layers(25, types)
    with pytest.raises(ValueError):
        uncheckpointed_layers(-1, types)


def test_a_tower_checkpointed_everywhere_describes_itself_as_before(tmp_path):
    """The default must not move a single key a ledger row or a recipe reads."""
    tower, _ = _tiny_tower(tmp_path)
    assert tower.checkpoint_skip_layers == ()
    assert "checkpoint_skip_layers" not in tower.to_json()
    assert tower.footprint.recompute == "full"


def test_a_selective_request_that_reached_the_wrong_layers_is_refused(tmp_path):
    """Read back per layer: a skip that landed on layer 2 instead of 1 is not the policy."""
    from qd_train.backbone import _verify_checkpointing_took

    tower, _ = _tiny_tower(tmp_path, checkpoint_skip_layers=2)
    tower.model.layers[1].gradient_checkpointing = True
    tower.model.layers[2].gradient_checkpointing = False
    with pytest.raises(BackboneContractViolation) as excinfo:
        _verify_checkpointing_took(tower.model, enabled=True, skip_layers=(1, 3))
    assert "[1, 2]" in str(excinfo.value)


def test_skipping_layers_without_checkpointing_is_refused(tmp_path):
    with pytest.raises(BackboneContractViolation, match="needs gradient_checkpointing on"):
        _tiny_tower(tmp_path, gradient_checkpointing=False, checkpoint_skip_layers=1)


def test_the_budget_prices_selective_checkpointing_between_full_and_none(tmp_path):
    """Fail closed needs the budget to see the retained layers: priced as 'full' it would
    admit a run that does not fit; priced as 'none' it would refuse one that does."""
    full, _ = _tiny_tower(tmp_path / "full")
    part, _ = _tiny_tower(tmp_path / "part", checkpoint_skip_layers=2)
    none, _ = _tiny_tower(tmp_path / "none", gradient_checkpointing=False)
    a = [t.footprint.activation_bytes for t in (full, part, none)]
    assert a[0] < a[1] <= a[2] + full.footprint.positions * TINY_HIDDEN * 4 * (TINY_LAYERS + 1)
    assert part.footprint.recompute == "full-except-0-linear-2-full"
    assert footprint_at(part, rows=1, width=64).activation_bytes == a[1]


def test_the_remap_carries_the_checkpointing_policy_and_the_kernel_record(tmp_path):
    """The remap used to rebuild TextTower field by field, so a field added later was
    dropped on exactly the towers that train. It now replaces only what it changes."""
    tower, _ = _tiny_tower(tmp_path, checkpoint_skip_layers=1)
    remapped = remap_text_tower(tower, _tiny_remap(list(range(0, TINY_VOCAB, 2)), []))
    assert remapped.checkpoint_skip_layers == tower.checkpoint_skip_layers == (3,)
    assert remapped.linear_attention_kernels == tower.linear_attention_kernels
    assert remapped.footprint.recompute == "full-except-0-linear-1-full"


@pytest.mark.parametrize("skip", [1, 2, TINY_LAYERS])
def test_selective_checkpointing_leaves_the_trajectory_bit_identical(tmp_path, skip):
    """Tier A on CPU, whose kernels are deterministic: recompute replays the same kernels on
    the same inputs, so skipping it must not move one bit of any loss, in either channel,
    or the batch order. Identical, not close."""
    base = _trajectory(tmp_path / "full", skip=0)
    other = _trajectory(tmp_path / f"skip{skip}", skip=skip)
    assert other == base
    assert any(x > 0.0 for x in base["span"]), "the span channel must actually be exercised"


@pytest.mark.parametrize("skip", [2, TINY_LAYERS])
def test_selective_checkpointing_composes_with_the_layerwise_lr_split(tmp_path, skip):
    """Run F pairs --checkpoint-skip-layers 6 with --lower-layers-n 8, and skip 6's layers 3
    and 7 sit inside the lower group. The split only scales an optimizer group and the skip
    only drops a recompute, so with the split on the trajectory must still not move a bit --
    here with a skipped full-attention layer (1) inside the lower group, as in F."""
    from qd_train.backbone import uncheckpointed_layers

    types = list(_tiny_text_config().layer_types)
    assert [i for i in uncheckpointed_layers(skip, types) if i < 2], "no overlap to test"
    base = _trajectory(tmp_path / "full", skip=0, lower_layers_n=2, lower_lr_scale=0.1)
    other = _trajectory(tmp_path / f"skip{skip}", skip=skip, lower_layers_n=2, lower_lr_scale=0.1)
    assert other == base
    flat = _trajectory(tmp_path / f"flat{skip}", skip=skip)
    assert flat["digest"] != base["digest"], "the split must be live, or this proves nothing"


# --- training without the padding mask (Tier B candidate; Fable's no-mask ruling) -------------


def _sdpa_flags() -> tuple[bool, bool, bool, bool]:
    import torch

    b = torch.backends.cuda
    return (b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.math_sdp_enabled(),
            b.cudnn_sdp_enabled())


def _watch_attention(step) -> dict[str, list]:
    """Record what every tower call was handed as ``attention_mask``, and the SDPA backend
    flags in force each time a full-attention layer ran -- forward AND checkpoint recompute,
    which happens inside ``backward``."""
    seen: dict[str, list] = {"mask": [], "flags": []}
    model = step.tower.model

    def tower_pre(_module, _args, kwargs):
        seen["mask"].append(kwargs.get("attention_mask"))

    def attn_pre(_module, _args, _kwargs):
        seen["flags"].append(_sdpa_flags())

    model.register_forward_pre_hook(tower_pre, with_kwargs=True)
    for i, kind in enumerate(model.config.layer_types):
        if kind == "full_attention":
            model.layers[i].self_attn.register_forward_pre_hook(attn_pre, with_kwargs=True)
    return seen


FLASH_ONLY = (True, False, False, False)


@pytest.mark.parametrize("span", [False, True])
def test_train_attention_mask_none_trains_without_the_mask_on_the_flash_backend_only(
    tmp_path, span
):
    """`--train-attention-mask none`: the TRAINING forward gets no mask (`is_causal`), and the
    forward and its checkpoint recompute run with flash as the only SDPA backend, so on CUDA
    a shape flash cannot take raises instead of falling back to mem-efficient. Both entry
    points, the letter-only and the span batch."""
    tower, _ = _tiny_tower(tmp_path, gradient_checkpointing=True)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64,
                            train_attention_mask="none")
    before = _sdpa_flags()
    seen = _watch_attention(step)
    batch = _ft_batch(0)
    if span:
        step.accumulate_span(batch, ft_supervision(batch))
    else:
        letter_only = Batch(
            tokens=batch.tokens, lengths=batch.lengths, bucket=0, index=0,
            slot_kind=np.array([SLOT_CHOICE, SLOT_CHOICE], dtype=np.uint8),
            target_index=np.array([3, 2], dtype=np.int32),
            span_target=np.array([(NO_SPAN, NO_SPAN)] * 2, dtype=np.int32),
        )
        step.accumulate(letter_only, ft_supervision(letter_only))
    assert seen["mask"] == [None]
    n_full = sum(1 for t in tower.model.config.layer_types if t == "full_attention")
    assert len(seen["flags"]) == 2 * n_full, "forward plus the checkpoint recompute"
    assert set(seen["flags"]) == {FLASH_ONLY}
    assert _sdpa_flags() == before, "the backend restriction must not outlive the step"


def test_scoring_through_hidden_keeps_the_mask_under_train_attention_mask_none(tmp_path):
    """`hidden()` is shared with `_decode`, `_evaluate` and replay; the switch is training-only,
    so a scored row changes for one reason only -- the trained weights."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64,
                            train_attention_mask="none")
    seen = _watch_attention(step)
    step.hidden(_ft_batch(0))
    (mask,) = seen["mask"]
    assert mask is not None and mask.tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]
    assert set(seen["flags"]) == {_sdpa_flags()} and _sdpa_flags() != FLASH_ONLY


def test_the_default_trains_with_the_padding_mask_and_every_backend(tmp_path):
    tower, _ = _tiny_tower(tmp_path, gradient_checkpointing=True)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64)
    assert step.train_attention_mask == "padding"
    seen = _watch_attention(step)
    batch = _ft_batch(0)
    step.accumulate_span(batch, ft_supervision(batch))
    (mask,) = seen["mask"]
    assert mask is not None
    assert FLASH_ONLY not in seen["flags"]


def test_an_unknown_train_attention_mask_is_refused(tmp_path):
    tower, _ = _tiny_tower(tmp_path)
    with pytest.raises(ValueError, match="train_attention_mask"):
        QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64,
                         train_attention_mask="causal")


def test_dropping_the_mask_leaves_real_positions_unchanged_and_moves_only_pads(tmp_path):
    """Why the switch is admissible at all: batches are right-padded and attention and the
    GDN recurrence are causal, so no real position reads a pad. Real positions' hidden states
    agree; pad positions are free to differ. (CPU only -- on CUDA the kernel itself changes,
    which is what makes this Tier B.)"""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=1, max_width=64,
                            train_attention_mask="none")
    batch = _ft_batch(0, width=12)
    real = np.arange(12)[None, :] < np.asarray(batch.lengths)[:, None]
    with step.training_attention():
        free = step.hidden(batch, padding_mask=False).detach().numpy()
    masked = step.hidden(batch).detach().numpy()
    np.testing.assert_allclose(free[real], masked[real], rtol=0, atol=1e-6)
    assert not np.allclose(free[~real], masked[~real]), "pads must not have been masked"


def test_the_tower_records_which_linear_attention_kernels_it_bound(tmp_path):
    from qd_train.backbone import linear_attention_on_reference_path

    tower, _ = _tiny_tower(tmp_path)
    kernels = tower.linear_attention_kernels
    assert set(kernels) == {
        "chunk_gated_delta_rule", "recurrent_gated_delta_rule",
        "causal_conv1d_fn", "causal_conv1d_update",
    }
    assert all(isinstance(v, str) and v for v in kernels.values())
    assert tower.to_json()["linear_attention_kernels"] == dict(kernels)
    assert linear_attention_on_reference_path({
        "chunk_gated_delta_rule":
            "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule",
        "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",
    }) == ["chunk_gated_delta_rule"]


# --- the per-step channel log persists beside the checkpoint ---------------------------------


def test_the_checkpoint_carries_both_channels_per_step_and_a_resume_continues_them(tmp_path):
    """The ledger row holds only the combined log's digest; a later run is comparable step
    by step and channel by channel only if the checkpoint keeps the numbers themselves."""
    checkpoint = _resumable_checkpoint(tmp_path / "first")
    log = checkpoint.model_state["channel_log"]
    letter = [float.fromhex(x) for x in log["letter"]]
    span = [float.fromhex(x) for x in log["span"]]
    assert len(letter) == len(span) == 6 == checkpoint.model_state["micro_batches"]
    assert all(x > 0.0 for x in span), "every _ft_batch carries a span row"

    tower, _ = _tiny_tower(tmp_path / "second")
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    step.load_state(checkpoint.model_state)
    assert step.letter_log == letter and step.span_log == span


def test_a_checkpoint_written_before_the_channel_log_still_resumes(tmp_path):
    checkpoint = _resumable_checkpoint(tmp_path / "first")
    older = {k: v for k, v in checkpoint.model_state.items() if k != "channel_log"}
    tower, _ = _tiny_tower(tmp_path / "second")
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    step.load_state(older)
    assert step.letter_log == [] and step.span_log == []


def test_a_channel_log_that_disagrees_with_its_micro_batches_is_refused(tmp_path):
    checkpoint = _resumable_checkpoint(tmp_path / "first")
    broken = dict(checkpoint.model_state)
    broken["channel_log"] = {"letter": broken["channel_log"]["letter"][:-1],
                             "span": broken["channel_log"]["span"]}
    tower, _ = _tiny_tower(tmp_path / "second")
    step = QwenDecisionStep(tower, seed=0, lr=1e-3, total_steps=12, max_width=64)
    with pytest.raises(BackboneContractViolation, match="channel log"):
        step.load_state(broken)
