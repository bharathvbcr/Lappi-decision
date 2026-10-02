"""``--noul-weight w`` (v5 ``arm_noul_weight.flag``, reading R5), at every layer it touches.

The flag multiplies the letter cross-entropy at supervised letter positions whose target is
the noul letter, and only in rows whose family is ``code.defect_class``. The loss stays
``sum(w_i * ce_i)`` over the COUNT of supervised positions -- no renormalisation -- so w is a
pure multiplier on those rows' pull. CLINC's Z-gold rows, every other family, and the span
channel (whose abstain row is not a letter position at all) are unweighted.

* ``fused_ce``: ``weights=None`` runs the pre-flag code; an all-ones weight is bit-identical
  to it; a weight is ``sum(w_i ce_i) / n`` in both the fused and the naive reference.
* ``backbone``: one function owns the position mask (``noul_weight_mask``); a defect-class Z
  row is weighted, a defect-class A row and a CLINC Z row are not; the span row is outside
  the letter channel. On a tiny real tower, ``w = 1`` is bit-identical to no flag in the loss
  and every gradient, and ``w = 4`` moves the loss by exactly ``3 * ce_defectZ / n``.
* ``real_ft_run``: recipe keys only when given, argv refusals, the plan count the Mac prelude
  reads, and the resume guard.

Torch- and transformers-gated, like ``test_backbone.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")
pytest.importorskip("transformers", reason="transformers is an optional 'mac' extra")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402
import test_backbone as tb  # noqa: E402
from test_real_ft_rungd_flags import _only, _run, _value, corpus  # noqa: E402, F401

from qd_data.defect_class import DEFECT_FAMILY_ID  # noqa: E402
from qd_data.general import CLINC_WITHIN_DOMAIN_FAMILY  # noqa: E402
from qd_data.schema import NOUL_LETTER  # noqa: E402
from qd_train.artifacts import NO_SPAN, SLOT_CHOICE, SLOT_SPAN  # noqa: E402
from qd_train.backbone import (  # noqa: E402
    BackboneContractViolation,
    NoulWeight,
    QwenDecisionStep,
    noul_weight_mask,
)
from qd_train.fused_ce import (  # noqa: E402
    fused_linear_cross_entropy,
    naive_linear_cross_entropy,
)
from qd_train.shards import assemble_batch  # noqa: E402
from qd_train.trainer import ft_supervision  # noqa: E402

#: A CLINC family: its oos rows are Z-gold letter rows and must stay unweighted.
CLINC_FAMILY_ID = CLINC_WITHIN_DOMAIN_FAMILY
#: Token ids inside the tiny tower's vocabulary (test_backbone.TINY_VOCAB = 64).
NOUL_ID, A_ID = 7, 5


# --- fused_ce: the per-position weight ---------------------------------------------------------


def _ce_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(3, 6, 8, generator=g, requires_grad=True)
    weight = torch.randn(11, 8, generator=g, requires_grad=True)
    targets = torch.randint(0, 11, (3, 6), generator=g)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[0, 2] = mask[1, 4] = mask[2, 1] = mask[2, 5] = True
    return hidden, weight, targets, mask


def _loss_and_grads(fn, *, weights=None, chunk_size=None):
    hidden, weight, targets, mask = _ce_inputs()
    kwargs = {"mask": mask, "weights": weights}
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    loss = fn(hidden, weight, targets, **kwargs)
    loss.backward()
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


@pytest.mark.parametrize("chunk_size", [None, 1, 4])
def test_all_ones_weights_are_bit_identical_to_no_weights(chunk_size):
    plain = _loss_and_grads(fused_linear_cross_entropy, chunk_size=chunk_size)
    ones = _loss_and_grads(
        fused_linear_cross_entropy, weights=torch.ones(3, 6), chunk_size=chunk_size
    )
    for a, b in zip(plain, ones, strict=True):
        assert torch.equal(a, b)


def test_a_weight_is_sum_w_ce_over_the_supervised_count_fused_and_naive():
    w = torch.ones(3, 6)
    w[2, 5] = 4.0  # one supervised position weighted
    w[0, 0] = 9.0  # an unsupervised one: must change nothing
    fused = _loss_and_grads(fused_linear_cross_entropy, weights=w, chunk_size=2)
    naive = _loss_and_grads(naive_linear_cross_entropy, weights=w)
    for a, b in zip(fused, naive, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    hidden, weight, targets, mask = _ce_inputs()
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy(
            (hidden @ weight.t()).reshape(-1, 11), targets.reshape(-1), reduction="none"
        ).reshape(3, 6)
    expected = (ce * w * mask).sum() / mask.sum()
    torch.testing.assert_close(fused[0], expected, rtol=1e-5, atol=1e-6)
    plain = _loss_and_grads(fused_linear_cross_entropy)[0]
    # Not renormalised: the weighted loss is the plain one plus 3 * ce[2, 5] / 4 positions.
    torch.testing.assert_close(fused[0] - plain, 3.0 * ce[2, 5] / 4, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: torch.ones(3, 5), "weights must match targets"),
        (lambda: torch.ones(3, 6, dtype=torch.int64), "weights must be floating"),
        (lambda: torch.full((3, 6), -1.0), "finite and non-negative"),
        (lambda: torch.full((3, 6), float("nan")), "finite and non-negative"),
        (lambda: torch.full((3, 6), float("inf")), "finite and non-negative"),
    ],
)
def test_a_weight_that_is_not_a_weight_is_refused(make, match):
    hidden, weight, targets, mask = _ce_inputs()
    for fn in (fused_linear_cross_entropy, naive_linear_cross_entropy):
        with pytest.raises(ValueError, match=match):
            fn(hidden, weight, targets, mask=mask, weights=make())


# --- the mask: which positions are weighted -----------------------------------------------------


def _batch(*, with_span: bool = False):
    """Rows: defect-class Z gold, defect-class A gold, CLINC Z gold (and a span row)."""
    seqs = [
        np.asarray([1, 2, 3, NOUL_ID, 0], dtype=np.int32),
        np.asarray([1, 2, 3, A_ID, 0], dtype=np.int32),
        np.asarray([1, 4, 3, NOUL_ID, 0], dtype=np.int32),
    ]
    kinds = [SLOT_CHOICE] * 3
    targets = [2, 2, 2]
    spans = [(NO_SPAN, NO_SPAN)] * 3
    candidates: list[list[int]] = [[], [], []]
    if with_span:
        # A span row whose only letter is the noul letter: it is supervised by the pointer
        # head, never by the letter channel, so no weight can reach it.
        seqs.append(np.asarray([1, 9, 10, 11, 12, NOUL_ID, 0], dtype=np.int32))
        kinds.append(SLOT_SPAN)
        targets.append(4)
        spans.append((1, 3))  # both ends are candidates (line starts), as the shard contract asks
        candidates.append([1, 3])
    width = max(s.size for s in seqs)
    return assemble_batch(
        seqs, kinds=np.asarray(kinds), target_index=np.asarray(targets),
        spans=np.asarray(spans, dtype=np.int64), candidates=candidates, width=width,
        bucket=width, index=0,
    )


def test_only_defect_class_z_positions_are_weighted():
    batch = _batch()
    sup = ft_supervision(batch)
    rows = np.asarray([True, True, False])  # the two defect-class rows; CLINC is out of scope
    mask = noul_weight_mask(sup, rows, NOUL_ID)
    assert mask.shape == sup.mask.shape
    assert mask.sum() == 1 and mask[0, 2]
    assert not mask[1].any(), "a defect-class A row is not weighted"
    assert not mask[2].any(), "a CLINC Z row is not weighted"


def test_the_span_row_is_outside_the_letter_channel_so_it_is_never_weighted():
    batch = _batch(with_span=True)
    sup = ft_supervision(batch)
    assert sup.span is not None and not sup.mask[3].any()
    mask = noul_weight_mask(sup, np.asarray([True, True, False, True]), NOUL_ID)
    assert mask.sum() == 1 and mask[0, 2]


def test_a_noul_weight_refuses_a_batch_it_was_not_planned_for():
    nw = NoulWeight(weight=4.0, scope=DEFECT_FAMILY_ID, noul_id=NOUL_ID,
                    rows_by_index={1: np.asarray([True, True, False])})
    batch = _batch()  # index 0
    with pytest.raises(BackboneContractViolation, match="batch index 0"):
        nw.position_weights(batch, ft_supervision(batch))
    nw_short = NoulWeight(weight=4.0, scope=DEFECT_FAMILY_ID, noul_id=NOUL_ID,
                          rows_by_index={0: np.asarray([True, True])})
    with pytest.raises(BackboneContractViolation, match="3 rows"):
        nw_short.position_weights(batch, ft_supervision(batch))


@pytest.mark.parametrize("w", [0.0, -1.0, math.nan, math.inf])
def test_a_noul_weight_that_is_not_positive_and_finite_is_refused(w):
    with pytest.raises(ValueError, match="finite and positive"):
        NoulWeight(weight=w, scope=DEFECT_FAMILY_ID, noul_id=NOUL_ID, rows_by_index={})


# --- the step, on a tiny real tower --------------------------------------------------------------


def _step(tmp_path: Path, noul: NoulWeight | None) -> QwenDecisionStep:
    tower, _ = tb._tiny_tower(tmp_path, gradient_checkpointing=False)
    return QwenDecisionStep(
        tower, seed=0, lr=1e-3, total_steps=4, max_width=64, noul_weight=noul,
    )


def _letter(step: QwenDecisionStep, batch):
    sup = ft_supervision(batch)
    step.optimizer.zero_grad(set_to_none=True)
    loss = step.accumulate(batch, sup)
    grads = [p.grad.detach().clone() for p in step.parameters() if p.grad is not None]
    return loss, grads


def _noul(w: float) -> NoulWeight:
    return NoulWeight(weight=w, scope=DEFECT_FAMILY_ID, noul_id=NOUL_ID,
                      rows_by_index={0: np.asarray([True, True, False])})


def test_w1_is_bit_identical_to_no_flag_in_the_loss_and_every_gradient(tmp_path):
    batch = _batch()
    plain_loss, plain_grads = _letter(_step(tmp_path / "a", None), batch)
    one_loss, one_grads = _letter(_step(tmp_path / "b", _noul(1.0)), batch)
    assert one_loss == plain_loss
    assert len(one_grads) == len(plain_grads) > 0
    for a, b in zip(plain_grads, one_grads, strict=True):
        assert torch.equal(a, b)


def test_w4_moves_the_loss_by_exactly_the_defect_z_rows_extra_pull(tmp_path):
    batch = _batch()
    sup = ft_supervision(batch)
    plain = _step(tmp_path / "a", None)
    with torch.no_grad():
        hidden = plain.hidden(batch)
        logits = plain.lm_head(hidden[:, :-1]).float()
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            torch.as_tensor(sup.targets.astype(np.int64)).reshape(-1), reduction="none",
        ).reshape(sup.targets.shape)
    n = sup.n_supervised
    plain_loss, plain_grads = _letter(plain, batch)
    four_loss, four_grads = _letter(_step(tmp_path / "b", _noul(4.0)), batch)
    assert math.isclose(four_loss - plain_loss, 3.0 * float(ce[0, 2]) / n, rel_tol=1e-4)
    assert any(not torch.equal(a, b) for a, b in zip(plain_grads, four_grads, strict=True))
    # The CLINC Z row's own term is unchanged: drop row 0 from scope and w moves nothing.
    none_in_scope = NoulWeight(weight=4.0, scope=DEFECT_FAMILY_ID, noul_id=NOUL_ID,
                               rows_by_index={0: np.asarray([False, True, False])})
    clinc_loss, _ = _letter(_step(tmp_path / "c", none_in_scope), batch)
    assert clinc_loss == plain_loss


def test_a_checkpoint_and_a_step_that_disagree_about_the_weight_are_refused(tmp_path):
    weighted = _step(tmp_path / "a", _noul(4.0))
    plain = _step(tmp_path / "b", None)
    state_w, state_p = weighted.state(), plain.state()
    assert state_w["noul_weight"] == {"weight": 4.0, "scope": DEFECT_FAMILY_ID}
    assert "noul_weight" not in state_p, "a run without the flag writes the state it always did"
    with pytest.raises(BackboneContractViolation, match="noul_weight"):
        plain.load_state(state_w)
    with pytest.raises(BackboneContractViolation, match="noul_weight"):
        weighted.load_state(state_p)
    other = _step(tmp_path / "c", _noul(3.0))
    with pytest.raises(BackboneContractViolation, match="noul_weight"):
        other.load_state(state_w)


# --- real_ft_run: the flag, the recipe and the plan count ---------------------------------------

_OFF = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
        "permutation": None, "replay": None}


def test_the_recipe_keys_appear_only_when_the_flag_is_given():
    assert rft._recipe_pieces(**_OFF) == {}
    assert rft._recipe_pieces(**_OFF, noul_weight=4.0) == {
        "noul_weight": 4.0, "noul_weight_scope": DEFECT_FAMILY_ID,
    }
    assert {"noul_weight", "noul_weight_scope"} <= set(rft.RECIPE_PIECE_KEYS)
    assert rft.NOUL_WEIGHT_SCOPE == DEFECT_FAMILY_ID


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--noul-weight", "0"], "finite and positive"),
        (["--noul-weight", "-4"], "finite and positive"),
        (["--noul-weight", "nan"], "finite and positive"),
        (["--noul-weight", "inf"], "finite and positive"),
        (["--epoch", "--noul-weight", "4"], "needs --real-backbone"),
    ],
)
def test_a_noul_weight_that_could_not_train_is_refused_at_argv_time(tmp_path, flags, match):
    with pytest.raises(SystemExit, match=match):
        rft.main(["--out", str(tmp_path), *flags])


def _label(family: str, gold: str) -> rft.Label:
    return rft.Label(row_id=f"{family}-{gold}", family_id=family, slot_name="s",
                     slot_kind=SLOT_CHOICE, gold_letter=gold, letters=("A", NOUL_LETTER))


def test_the_plan_count_is_what_the_trainer_weights():
    """The count the Mac prelude prints: defect-class Z positions of the plan, per pass."""
    batch = _batch()
    labels = {0: [_label(DEFECT_FAMILY_ID, NOUL_LETTER), _label(DEFECT_FAMILY_ID, "A"),
                  _label(CLINC_FAMILY_ID, NOUL_LETTER)]}
    plan = rft.noul_weight_plan(
        [batch], labels, weight=4.0, letter_id={NOUL_LETTER: NOUL_ID, "A": A_ID}
    )
    assert plan.weighted_positions == 1
    assert plan.supervised_positions == 3
    assert plan.weight_mass_ratio == (3 + 3.0 * 1) / 3
    assert [r.tolist() for r in plan.rows] == [[True, True, False]]
    assert plan.recipe() == {"noul_weight": 4.0, "noul_weight_scope": DEFECT_FAMILY_ID}
    by_index = plan.by_index(passes=2)
    assert sorted(by_index) == [0, 1]
    two = plan.cut(1)
    assert two.weighted_positions == 1
    with pytest.raises(ValueError, match="labels"):
        rft.noul_weight_plan([batch], {0: labels[0][:2]}, weight=4.0,
                             letter_id={NOUL_LETTER: NOUL_ID, "A": A_ID})


# --- end to end through main, on the toy code.defect_class corpus and the tiny tower -----------


def test_main_records_the_two_keys_and_the_count_and_w1_trains_bit_identically(
    corpus, tmp_path  # noqa: F811 - the fixture imported from test_real_ft_rungd_flags
):
    """--deterministic, CPU, the toy corpus: the ft row with --noul-weight 1 differs from the
    row without it in exactly the two recipe keys, carries the count metrics, and its loss
    log is the same bytes -- w = 1 trained the identical model."""
    flags = ("--deterministic", "--max-steps", "3")
    plain = _only(_run(corpus, tmp_path / "plain.jsonl", *flags, real=True, short=True), "ft")
    one = _only(
        _run(corpus, tmp_path / "one.jsonl", *flags, "--noul-weight", "1", real=True,
             short=True),
        "ft",
    )
    added = {k: one.recipe[k] for k in set(one.recipe) - set(plain.recipe)}
    assert added == {"noul_weight": 1.0, "noul_weight_scope": DEFECT_FAMILY_ID}
    assert {k: v for k, v in one.recipe.items() if k not in added} == plain.recipe
    assert "noul_weight" not in plain.recipe and "train.noul_weight.positions" not in plain.metrics
    positions = one.metrics["train.noul_weight.positions"].to_json()
    assert positions["value"] == positions["n"] and positions["n_total"] > 0
    assert _value(one, "train.noul_weight.mass_ratio") == 1.0
    assert _value(one, "train.loss_log_digest") == _value(plain, "train.loss_log_digest")
    four = _only(
        _run(corpus, tmp_path / "four.jsonl", *flags, "--noul-weight", "4", real=True,
             short=True),
        "ft",
    )
    k, n = (four.metrics["train.noul_weight.positions"].to_json()[x] for x in ("n", "n_total"))
    assert _value(four, "train.noul_weight.mass_ratio") == (n + 3.0 * k) / n
    # The toy corpus's weighted count decides whether w = 4 can move the loss at all; say
    # which case this run is in rather than assume it.
    if k:
        assert _value(four, "train.loss_log_digest") != _value(plain, "train.loss_log_digest")
    else:
        assert _value(four, "train.loss_log_digest") == _value(plain, "train.loss_log_digest")
