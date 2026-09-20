"""Rung 0's model. Torch-gated: reports **skipped** in the repo venv, never passed.

torch lives in ``~/.venvs/ml``, which has **no pytest**, so this file cannot be run there by
``pytest`` as written. It was verified by driving the module directly under torch 2.12.1
(19 assertions, 0 failures), with the abstain-placement invariant additionally mutation-checked
two ways. Installing pytest into that shared venv would be a side effect on an environment
this project does not own, so it was not done.

The load-bearing test here is `test_the_abstain_column_is_per_example_not_the_padded_width`.
`answer.rs` reads the abstention at column `n_live_options`, not at the padded width, so a
model that writes it at the width puts "no evidence" mass on a dead column and no loss curve
says so.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

from qd_train.byte_context import BYTE_VOCAB_SIZE, ID_PAD  # noqa: E402
from qd_train.byte_decider import (  # noqa: E402
    RESERVED_NOUL_ROWS,
    ByteDecider,
    ByteDeciderConfig,
    abstain_column,
)

NEG_INF = float("-inf")


def _tiny() -> ByteDeciderConfig:
    return ByteDeciderConfig(width=32, n_layers=1, n_heads=2, max_context_bytes=64,
                             max_option_bytes=16)


def _batch(b: int, length: int, k: int, o: int, n_live: list[int]):
    g = torch.Generator().manual_seed(0)
    ctx = torch.randint(0, 256, (b, length), generator=g)
    ctx_mask = torch.ones(b, length, dtype=torch.bool)
    opt = torch.randint(0, 256, (b, k, o), generator=g)
    opt_mask = torch.ones(b, k, o, dtype=torch.bool)
    return ctx, ctx_mask, opt, opt_mask, torch.tensor(n_live)


# ---------------------------------------------------------------------------
# The abstain row layout. This is the one that matters.
# ---------------------------------------------------------------------------


def test_abstain_column_mirrors_the_rust_arithmetic():
    assert abstain_column(1) == 1
    assert abstain_column(3) == 3
    assert abstain_column(16) == 16
    with pytest.raises(ValueError):
        abstain_column(0)


def test_the_abstain_column_is_per_example_not_the_padded_width():
    """Ragged counts, deliberately chosen so the mask and the scatter are distinguishable.

    An earlier test in this repo used counts `[2, 1]` at width 3, where the narrow row's only
    invalid column was the one the abstain scatter overwrites -- so it passed with the mask
    deleted. Here width is 5 (K=4 options + 1) and the counts are `[3, 1]`, which leaves row 1
    with three dead columns (2, 3, 4), none of them the abstain column.
    """
    k = 4
    width = k + RESERVED_NOUL_ROWS
    # Distinct values so it is visible which column each logit came from.
    logits = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0, 99.0],
            [20.0, 21.0, 22.0, 23.0, 88.0],
        ]
    )
    out = ByteDecider._place_abstain(logits, torch.tensor([3, 1]))
    assert out.shape == (2, width)

    # Row 0: options 0..2 survive, abstain lands at 3, column 4 is dead.
    assert out[0, 0].item() == 10.0
    assert out[0, 2].item() == 12.0
    assert out[0, 3].item() == 99.0, "the abstain logit must be moved into column n, not copied"
    assert out[0, 4].item() == NEG_INF

    # Row 1: only option 0 survives, abstain lands at 1, columns 2..4 are dead.
    assert out[1, 0].item() == 20.0
    assert out[1, 1].item() == 88.0
    for dead in (2, 3, 4):
        assert out[1, dead].item() == NEG_INF, f"column {dead} must be dead on a 1-option row"


def test_the_abstain_logit_is_not_left_behind_at_the_padded_width():
    """The failure mode: abstention present at the far right *and* at column n."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 50.0]])
    out = ByteDecider._place_abstain(logits, torch.tensor([2]))
    assert out[0, 2].item() == 50.0
    assert out[0, 3].item() == NEG_INF, "the original far-right column must be masked away"


def test_a_full_width_row_puts_abstain_in_the_last_column():
    """When every option is live the abstention is the final column, and nothing is dead."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 7.0]])
    out = ByteDecider._place_abstain(logits, torch.tensor([3]))
    assert out[0, 3].item() == 7.0
    assert torch.isfinite(out).all(), "no column should be dead when all options are live"


def test_softmax_over_the_placed_logits_ignores_dead_columns():
    """The reason -inf and not a large negative: dead columns must take exactly zero mass."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out = ByteDecider._place_abstain(logits, torch.tensor([2]))
    p = out.softmax(dim=-1)
    assert p[0, 3].item() == 0.0
    assert p[0, 4].item() == 0.0
    assert p[0, :3].sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_returns_one_logit_per_option_plus_the_reserved_row():
    model = ByteDecider(_tiny())
    ctx, cm, opt, om, n = _batch(2, 48, 4, 12, [4, 2])
    out = model(ctx, cm, opt, om, n)
    assert out.shape == (2, 4 + RESERVED_NOUL_ROWS)
    assert torch.isfinite(out[0]).all(), "row with all options live has no dead column"
    assert out[1, 3].item() == NEG_INF and out[1, 4].item() == NEG_INF


def test_the_option_count_is_a_runtime_property_not_a_shape():
    """The point of option-as-query over a letter slice: no fixed ceiling.

    The same weights score 2 options and 40, which the 16-row lm_head slice design cannot do
    (`GAP-SCHEMA-NOUL-LETTER-BUDGET`).
    """
    model = ByteDecider(_tiny())
    for k in (2, 5, 40):
        ctx, cm, opt, om, n = _batch(1, 32, k, 8, [k])
        out = model(ctx, cm, opt, om, n)
        assert out.shape == (1, k + RESERVED_NOUL_ROWS)


def test_gradients_reach_the_abstain_query():
    """If abstention were a bolted-on threshold this parameter would have no gradient."""
    model = ByteDecider(_tiny())
    ctx, cm, opt, om, n = _batch(2, 32, 3, 8, [3, 2])
    logits = model(ctx, cm, opt, om, n)
    target = torch.tensor([0, 2])  # row 1's target IS the abstain column (n_live=2)
    torch.nn.functional.cross_entropy(logits, target).backward()

    assert model.noul_query.grad is not None
    assert torch.isfinite(model.noul_query.grad).all()
    assert model.noul_query.grad.abs().sum().item() > 0.0


def test_padding_in_the_context_is_ignored_not_attended():
    """Changing masked-out context bytes must not change the answer."""
    model = ByteDecider(_tiny()).eval()
    ctx, cm, opt, om, n = _batch(1, 32, 3, 8, [3])
    cm[0, 20:] = False
    ctx_b = ctx.clone()
    ctx_b[0, 20:] = ID_PAD - 1  # any other byte value
    with torch.no_grad():
        a = model(ctx, cm, opt, om, n)
        b = model(ctx_b, cm, opt, om, n)
    torch.testing.assert_close(a, b)


def test_a_context_longer_than_the_window_is_refused():
    model = ByteDecider(_tiny())
    ctx, cm, opt, om, n = _batch(1, 65, 3, 8, [3])
    with pytest.raises(ValueError, match="exceeds the 64-byte window"):
        model(ctx, cm, opt, om, n)


def test_a_live_count_above_the_padded_width_is_refused():
    model = ByteDecider(_tiny())
    ctx, cm, opt, om, n = _batch(1, 32, 3, 8, [4])
    with pytest.raises(ValueError, match="exceeds the padded"):
        model(ctx, cm, opt, om, n)


def test_a_row_with_no_live_options_is_refused():
    model = ByteDecider(_tiny())
    ctx, cm, opt, om, n = _batch(1, 32, 3, 8, [0])
    with pytest.raises(ValueError, match="at least one live option"):
        model(ctx, cm, opt, om, n)


# ---------------------------------------------------------------------------
# Config and size
# ---------------------------------------------------------------------------


def test_a_width_that_does_not_divide_by_heads_is_refused():
    with pytest.raises(ValueError, match="must divide by n_heads"):
        ByteDeciderConfig(width=130, n_heads=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"n_layers": 0},
        {"max_context_bytes": 0},
        {"max_option_bytes": 0},
        {"dropout": 1.0},
    ],
)
def test_degenerate_config_is_refused(kwargs: dict):
    with pytest.raises(ValueError):
        ByteDeciderConfig(**kwargs)


def test_the_default_model_is_small_enough_to_be_the_cheap_rung():
    """The claim rung 0 rests on: a fraction of a 2B, trained from scratch on this Mac.

    `cua-s1-forms` is 706,048 parameters over a 224-byte context. This carries a 1024-byte
    context, so it is larger, but it must stay in the same order of magnitude or it is not
    the cheap rung any more.
    """
    model = ByteDecider()
    total = sum(p.numel() for p in model.parameters())
    print(f"\nByteDecider default parameters: {total:,}")
    assert total < 2_000_000, f"{total:,} parameters is no longer a small model"
    assert model.embed.num_embeddings == BYTE_VOCAB_SIZE
    assert model.embed.padding_idx == ID_PAD
