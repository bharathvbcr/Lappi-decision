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
    load_text_tower,
    remap_text_tower,
    saved_activation_bytes,
    text_tensor_index,
)
from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder  # noqa: E402
from qd_train.memory import ADAMW_BF16, ADAMW_FP32, ModelSpec  # noqa: E402
from qd_train.remap import build_remap  # noqa: E402
from qd_train.run_control import (  # noqa: E402
    CostEstimate,
    LRSchedule,
    RunControl,
    WallClockCap,
)
from qd_train.trainer import (  # noqa: E402
    SpanScoringStep,
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


def _tiny_text_config():
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=TINY_VOCAB,
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
    cfg = _tiny_text_config()
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
        params_embedding=TINY_VOCAB * TINY_HIDDEN,
        tied_embedding=True,
        recurrent_state_bytes=4,
    )


def _write_tiny_snapshot(dirpath: Path, *, seed: int = 0):
    """Write a tiny tower in the real on-disk layout. Returns the reference model."""
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextModel

    text = _tiny_text_config()
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
        line_starts=np.array(
            [[True, False, True, False, True, False], [False] * width], dtype=bool
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
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
    assert isinstance(step, TrainStep)
    assert isinstance(step, SpanScoringStep), (
        "without accumulate_span the loop refuses every SLOT_SPAN batch"
    )


def test_a_real_backbone_trains_through_train_ft_and_the_loss_falls(tmp_path):
    """The binding this lane exists to make: real weights driven by the real loop.

    A tiny tower, but a real ``Qwen3_5TextModel`` loaded from a real snapshot layout, real
    ``fused_linear_cross_entropy`` against the real tied embedding, the real
    ``SpanPointerHead``, and ``qd_train.trainer.train_ft`` itself. Twelve repeats of one
    batch, so the floor is memorisation and a falling loss means the gradient reaches the
    parameters -- which is the claim, and the only one twelve steps can support.
    """
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
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
    step = QwenDecisionStep(tower, lr=1e-3, max_width=4)
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
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
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
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
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
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
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
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
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


def test_a_state_that_lost_its_tensors_is_refused(tmp_path):
    """A body that came back without its sidecar is not a body with fewer tensors."""
    tower, _ = _tiny_tower(tmp_path)
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
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
    step = QwenDecisionStep(tower, lr=1e-3, max_width=64)
    assert isinstance(step.optimizer, MasterWeightAdamW), (
        f"the tower's spec asked for an fp32 master and the step built "
        f"{type(step.optimizer).__name__}"
    )

    plain, _ = _tiny_tower(tmp_path / "plain", dtype="bf16", optimizer=ADAMW_BF16)
    plain_step = QwenDecisionStep(plain, lr=1e-3, max_width=64)
    assert isinstance(plain_step.optimizer, torch.optim.AdamW)
    assert not isinstance(plain_step.optimizer, MasterWeightAdamW)


def test_a_bf16_tower_under_the_master_recipe_trains_and_stays_bf16(tmp_path):
    """End to end through the real step: the tower trains, and the live weights stay bf16 --
    if they were promoted to fp32 every matmul after the first step would run at fp32 speed
    and the memory budget would be wrong by the size of the model."""
    from qd_train.memory import OptimizerSpec

    master_spec = OptimizerSpec("AdamW+master", 2, 4, keeps_fp32_master=True)
    tower, _ = _tiny_tower(tmp_path, dtype="bf16", optimizer=master_spec)
    step = QwenDecisionStep(tower, lr=1e-2, max_width=64)
    before = tower.model.get_input_embeddings().weight.detach().clone()

    batch = _ft_batch(0)
    for _ in range(5):
        step.accumulate(batch, ft_supervision(batch))
        step.apply(lr=1e-2)

    after = tower.model.get_input_embeddings().weight.detach()
    assert after.dtype == torch.bfloat16, f"the live weights became {after.dtype}"
    assert not torch.equal(after, before), "the tower did not train"
    assert torch.isfinite(after).all(), "the tower has non-finite weights after 5 steps"
