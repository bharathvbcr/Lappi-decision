"""The plain-torch LoRA reference arm: Fable's (a') bounds on a tiny synthetic tower, CPU only.

D7 (a'): LoRA forward equal to the frozen base at B = 0; no ``dW`` allocated for a frozen weight and
the frozen leaves' gradients *absent*, not zero; scaling ``alpha / r``; only the adapter leaves (and
the heads the ruling names) train. The tower is built from random weights in-process -- no download.

Torch-gated. Run with the ml venv (pytest is not installed there; see ``test_backbone.py``'s header
for the ``uv run --with pytest`` form)::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python /Users/bharath/.venvs/ml/bin/python \\
        --with pytest --with hypothesis python -m pytest python/tests/test_lora_reference.py \\
        -o addopts= -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_backbone import TINY_LAYERS, TINY_VOCAB, _tiny_text_config  # noqa: E402

from qd_train import lora  # noqa: E402
from qd_train.heads import SpanPointerHead  # noqa: E402
from qd_train.memory import ADAMW_FP32  # noqa: E402
from qd_train.optim import apply_lr, build_optimizer  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "campaign" / "lora-2b-preregistered.json"

HIDDEN = 32


def _tower(dtype=torch.float32, seed: int = 0):
    from transformers.models.qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    return Qwen3_5TextModel(_tiny_text_config()).to(dtype=dtype)


def _ids(batch: int = 2, width: int = 7):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, TINY_VOCAB, (batch, width), generator=g)


def _hidden(model, ids):
    return model(input_ids=ids).last_hidden_state


def _accumulate_grad_leaves(loss) -> set[int]:
    """``id`` of every leaf tensor autograd will write a gradient into, walking the graph."""
    # Autograd nodes are handed to Python as short-lived wrappers whose ``id`` is reused once
    # one is collected, so every visited node is kept alive for the whole walk.
    seen, alive, stack, leaves = set(), [], [loss.grad_fn], set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        alive.append(node)
        variable = getattr(node, "variable", None)
        if variable is not None:
            leaves.add(id(variable))
        stack.extend(fn for fn, _ in node.next_functions)
    return leaves


# --- structure -------------------------------------------------------------------------------


def test_targets_are_the_ten_d6_linears_including_gdn_and_exclude_the_per_head_gates():
    model = _tower()
    n_full = sum(1 for t in model.config.layer_types if t == "full_attention")
    n_gdn = TINY_LAYERS - n_full
    adapter = lora.attach_lora(model, seed=0)
    assert len(adapter.module_names) == 3 * TINY_LAYERS + 4 * n_full + 3 * n_gdn
    names = set(adapter.module_names)
    assert any(n.endswith("linear_attn.in_proj_qkv") for n in names)
    assert any(n.endswith("linear_attn.in_proj_z") for n in names)
    assert any(n.endswith("linear_attn.out_proj") for n in names)
    assert not any(n.endswith(lora.LORA_UNADAPTED_LINEAR_SUFFIXES) for n in names)


def test_only_adapter_leaves_require_grad_and_everything_else_is_frozen():
    model = _tower()
    adapter = lora.attach_lora(model, seed=0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {n for n, _ in lora.lora_parameters(model)}
    assert len(trainable) == 2 * len(adapter.module_names)
    for name, p in model.named_parameters():
        if name not in trainable:
            assert not p.requires_grad, name


def test_a_is_kaiming_uniform_sqrt5_b_is_zero_and_the_draw_is_seeded():
    one, two, other = _tower(), _tower(), _tower()
    lora.attach_lora(one, seed=3)
    lora.attach_lora(two, seed=3)
    lora.attach_lora(other, seed=4)
    for name, p in lora.lora_parameters(one):
        if name.endswith("lora_B.weight"):
            assert not p.any()
        else:
            bound = 1.0 / math.sqrt(p.shape[1])  # gain sqrt(2/(1+5)) * sqrt(3/fan_in)
            assert p.abs().max() <= bound + 1e-7
            assert p.abs().max() > 0.5 * bound
    first = dict(lora.lora_parameters(one))
    for name, p in lora.lora_parameters(two):
        assert torch.equal(p, first[name])
    a_name = next(n for n in first if n.endswith("lora_A.weight"))
    assert not torch.equal(first[a_name], dict(lora.lora_parameters(other))[a_name])


def test_scaling_is_alpha_over_r_not_a_fixed_constant():
    model = _tower()
    adapter = lora.attach_lora(model, seed=0, r=4, alpha=12)
    assert adapter.scaling == 3.0
    module = model.get_submodule(adapter.module_names[0])
    torch.nn.init.normal_(module.lora_B.weight)
    x = torch.randn(5, module.base.in_features)
    expected = module.base(x) + 3.0 * (x @ module.lora_A.weight.t() @ module.lora_B.weight.t())
    assert torch.allclose(module(x), expected, atol=1e-6)


# --- identity at B = 0 -----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_at_b_zero_is_bit_equal_to_the_frozen_base(dtype):
    base = _tower(dtype)
    model = _tower(dtype)
    lora.attach_lora(model, seed=0)
    ids = _ids()
    with torch.no_grad():
        want = _hidden(base, ids)
        got = _hidden(model, ids)
    assert got.dtype == want.dtype
    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert (got.contiguous().view(bits) == want.contiguous().view(bits)).all()


def test_adapter_leaves_stay_f32_over_a_bf16_base():
    model = _tower(torch.bfloat16)
    lora.attach_lora(model, seed=0)
    assert {p.dtype for _, p in lora.lora_parameters(model)} == {torch.float32}


# --- no dW, no frozen gradient ---------------------------------------------------------------


def test_the_graph_reaches_only_adapter_leaves_so_no_dw_is_ever_computed():
    model = _tower()
    lora.attach_lora(model, seed=0)
    for _, p in lora.lora_parameters(model):
        torch.nn.init.normal_(p, std=0.05)
    loss = _hidden(model, _ids()).square().mean()
    reachable = _accumulate_grad_leaves(loss)
    adapters = {id(p) for _, p in lora.lora_parameters(model)}
    frozen = {id(p) for n, p in model.named_parameters() if id(p) not in adapters}
    assert reachable & frozen == set()
    assert reachable == adapters
    loss.backward()
    for name, p in model.named_parameters():
        if id(p) in adapters:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name  # absent, not zero


def test_at_b_zero_only_b_receives_a_nonzero_gradient():
    model = _tower()
    lora.attach_lora(model, seed=0)
    _hidden(model, _ids()).square().mean().backward()
    a_grads = [p.grad for n, p in lora.lora_parameters(model) if n.endswith("lora_A.weight")]
    b_grads = [p.grad for n, p in lora.lora_parameters(model) if n.endswith("lora_B.weight")]
    assert all(g is not None and not g.any() for g in a_grads)
    assert any(g is not None and g.any() for g in b_grads)


# --- refusals --------------------------------------------------------------------------------


def test_an_adapter_over_nothing_is_refused():
    with pytest.raises(lora.LoraContractViolation, match="trains nothing"):
        lora.attach_lora(_tower(), seed=0, target_suffixes=("no_such.linear",))


def test_a_second_attach_is_refused():
    model = _tower()
    lora.attach_lora(model, seed=0)
    with pytest.raises(lora.LoraContractViolation, match="already carries"):
        lora.attach_lora(model, seed=0)


# --- merge -----------------------------------------------------------------------------------


def test_merged_weights_reproduce_the_adapter_forward():
    model = _tower()
    adapter = lora.attach_lora(model, seed=0, r=4, alpha=8)
    for _, p in lora.lora_parameters(model):
        torch.nn.init.normal_(p, std=0.1)
    ids = _ids()
    with torch.no_grad():
        want = _hidden(model, ids)
        merged = lora.merged_weights(model)
        assert set(merged) == {f"{n}.weight" for n in adapter.module_names}
        plain = _tower()
        sd = plain.state_dict()
        for key, w in merged.items():
            sd[key] = w
        plain.load_state_dict(sd)
        got = _hidden(plain, ids)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


# --- D17 -------------------------------------------------------------------------------------


def _rows(delta_rows=lora.D17_ROWS):
    g = torch.Generator().manual_seed(5)
    embedding = torch.randn(TINY_VOCAB, HIDDEN, generator=g)
    return embedding, torch.arange(delta_rows)


def test_d17_at_zero_scores_exactly_the_frozen_embedding_rows():
    embedding, ids = _rows()
    delta = lora.AnswerRowDelta(HIDDEN)
    hidden = torch.randn(6, HIDDEN)
    assert torch.equal(delta.logits(hidden, embedding, ids), hidden @ embedding[ids].t())


def test_d17_loss_is_the_17_way_ce_and_only_the_delta_gets_a_gradient():
    embedding, ids = _rows()
    embedding.requires_grad_(True)
    delta = lora.AnswerRowDelta(HIDDEN)
    hidden = torch.randn(6, HIDDEN, requires_grad=True)
    targets = ids[torch.tensor([0, 3, 16, 3, 5, 9])]
    loss = delta.loss(hidden, embedding, ids, targets)
    ref = torch.nn.functional.cross_entropy(
        hidden @ embedding.detach()[ids].t(), torch.tensor([0, 3, 16, 3, 5, 9])
    )
    assert torch.allclose(loss, ref, atol=1e-6)
    loss.backward()
    assert delta.delta.grad is not None and delta.delta.grad.any()
    assert embedding.grad is None  # frozen: no gradient into the table


def test_d17_refuses_a_target_outside_the_answer_rows():
    embedding, ids = _rows()
    delta = lora.AnswerRowDelta(HIDDEN)
    with pytest.raises(lora.LoraContractViolation, match="not one of the answer rows"):
        delta.loss(torch.randn(1, HIDDEN), embedding, ids, torch.tensor([40]))


def test_d17_refuses_duplicate_answer_rows():
    embedding, _ = _rows()
    ids = torch.zeros(lora.D17_ROWS, dtype=torch.int64)
    with pytest.raises(lora.LoraContractViolation, match="not distinct"):
        lora.AnswerRowDelta(HIDDEN).loss(torch.randn(1, HIDDEN), embedding, ids, torch.tensor([0]))


# --- the unchanged Lappi optimizer and schedule drive it -------------------------------------


def test_optimizer_trains_adapters_and_heads_and_leaves_the_base_bit_identical():
    model = _tower()
    lora.attach_lora(model, seed=0)
    delta = lora.AnswerRowDelta(HIDDEN)
    head = SpanPointerHead(HIDDEN)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    schedule = lora.lora_schedule(40)
    groups = lora.lora_param_groups(model, delta=delta, span_head=head)
    opt = build_optimizer(groups, spec=ADAMW_FP32, lr=schedule.peak_lr, total_steps=40)

    embedding = model.get_input_embeddings().weight
    ids = torch.arange(lora.D17_ROWS)
    hidden = _hidden(model, _ids())[:, -1]
    loss = delta.loss(hidden, embedding, ids, ids[torch.tensor([1, 2])])
    loss = loss + head.start_proj(hidden).square().mean()
    loss.backward()
    apply_lr(opt, schedule.lr_at(schedule.warmup_steps))
    opt.step()

    assert [g["lr"] for g in opt.param_groups] == pytest.approx([2e-4, 1e-4, 1e-4])
    assert all(g["weight_decay"] == 0.01 for g in opt.param_groups)
    assert all(g["betas"] == (0.9, 0.999) and g["eps"] == 1e-8 for g in opt.param_groups)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), n
    assert delta.delta.abs().max() > 0


def test_schedule_is_lappis_warmup_steps_over_20_and_cosine_to_zero():
    s = lora.lora_schedule(400)
    assert (s.warmup_steps, s.min_lr, s.peak_lr) == (20, 0.0, 2e-4)
    assert lora.lora_schedule(10).warmup_steps == 1
    assert s.lr_at(399) < 1e-8


# --- the code's constants are the pre-registered ones ----------------------------------------


@pytest.mark.skipif(not PREREG.is_file(), reason="campaign/lora-2b-preregistered.json absent")
def test_constants_equal_the_preregistration():
    recipe = json.loads(PREREG.read_text(encoding="utf-8"))["recipe"]
    assert recipe["lora"]["r"] == lora.LORA_RANK
    assert recipe["lora"]["alpha"] == lora.LORA_ALPHA
    assert recipe["lora"]["dropout"] == 0
    assert tuple(recipe["lora"]["target_suffixes"]) == lora.LORA_TARGET_SUFFIXES
    assert tuple(recipe["lora"]["not_adapted"]) == lora.LORA_UNADAPTED_LINEAR_SUFFIXES
    assert recipe["optimizer"]["lr"]["lora"] == lora.LORA_LR
    assert recipe["optimizer"]["lr"]["d17"] == lora.D17_LR
    assert recipe["optimizer"]["lr"]["span_head"] == lora.D17_LR
    assert recipe["optimizer"]["weight_decay"] == lora.LORA_WEIGHT_DECAY
    assert recipe["head"]["d17_rows"] == lora.D17_ROWS
