"""Which backbone a ``tools/real_ft_run.py`` row says it trained.

``backbone_commit`` is one of the five components of ``qd_train.ledger.Protocol``, whose
docstring is explicit that "two rows are comparable only if their protocol hashes match".
So the field does not merely label a row -- it decides which rows may be compared against
each other, and which three rows constitute a seed family for a promotion.

``tools/real_ft_run.py`` had no test file at all, which is how five rows in
``ledger/gh200-2026-09-20.jsonl`` came to carry ``backbone_commit="scratch:128x4:1block"``
over notes reading "Backbone is the REAL text tower ... 1,881,825,088 trainable
parameters". Those rows are in an append-only ledger and are not rewritten; see
``gaps.jsonl``. These tests are what stops a sixth.

Torch-gated: ``tools/real_ft_run.py`` raises ``SystemExit`` at import when torch is absent
(deliberately -- the repo venv carries none), so this module skips rather than passing
vacuously in the plain ``.venv``. To run it::

    PYTHONDONTWRITEBYTECODE=1 uv run --no-project \\
        --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis \\
        python -m pytest python/tests/test_real_ft_protocol.py -o addopts= -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from real_ft_run import BACKBONE_KEYS, _backbone_commit  # noqa: E402

sys.path.insert(0, str(REPO / "python"))

from qd_train.tristate import NotRun, Ran  # noqa: E402

#: A stand-in recipe, as ``_train`` builds it when ``--real-backbone`` is absent.
STANDIN: dict[str, object] = {
    "tool": "tools/real_ft_run.py", "tag": "memorise", "device": "cpu",
    "lr": 3e-3, "passes": 40, "batches": 8, "width": 2048, "shard_hash": "deadbeef",
    "hidden": 128, "heads": 4,
}

#: A real-tower recipe, with the values a GH200 run actually produced (ledger rows
#: 4541d72e / 5ebebda7 / 6374e188). ``backbone_snapshot`` is the HF revision, which on that
#: host is both the contents of ``refs/main`` and the name of the sole snapshot directory.
#:
#: ``backbone_params`` is 1.40B and not the 1.88B of the full text tower: the pipeline's
#: remap slices the tied embedding to the tokens the corpus actually uses, and
#: ``1,881,825,088 - 248,320*2048 + 14,016*2048 == 1,401,970,496`` exactly. Two real
#: quantities, one name -- so the fixture carries the pair that a real row carries.
REAL: dict[str, object] = {
    "tool": "tools/real_ft_run.py", "tag": "memorise", "device": "cuda",
    "lr": 1e-5, "passes": 40, "batches": 8, "width": 2048, "shard_hash": "deadbeef",
    "backbone_snapshot": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    "backbone_params": 1_401_970_496,
    "backbone_vocab": 14_016,
}


def test_the_stand_in_is_named_the_way_it_has_always_been_named() -> None:
    """Characterisation: the existing 97 stand-in rows stay comparable to new ones.

    If this changes, every ``scratch:128x4:1block`` row already in the ledger silently
    stops being in the same seed family as the next stand-in run.
    """
    assert _backbone_commit(STANDIN) == "scratch:128x4:1block"


def test_a_real_tower_run_is_not_labelled_a_scratch_block() -> None:
    """The bug, stated as a test.

    Pre-fix this returned ``scratch:128x4:1block`` for a run of the 1.88B tower -- or,
    after ``--hidden``/``--heads`` became sentinels, ``scratch:NonexNone:1block``. Either
    way a query filtering on ``backbone_commit`` grouped the real tower with the stand-in.
    """
    commit = _backbone_commit(REAL)
    assert not commit.startswith("scratch:"), commit
    assert "b1485b2fa6dfa1287294f269f5fb618e03d52d7c" in commit
    assert commit != _backbone_commit(STANDIN)


def test_the_remapped_vocabulary_is_part_of_the_identity() -> None:
    """A tower remapped to 14,016 rows is not the tower remapped to 248,320.

    The tied embedding is sliced by the remap, so the two are different models by nearly
    480M parameters. Runs that differ in it are not the same protocol and must not form a
    seed family together.
    """
    other = dict(REAL, backbone_vocab=248_320)
    assert _backbone_commit(other) != _backbone_commit(REAL)


def test_a_recipe_naming_no_backbone_is_refused_rather_than_invented() -> None:
    """``real_tokenizer_pipeline.py`` already states this rule, and refuses on it: a
    protocol naming no backbone identifies nothing, so no row is written rather than one
    being written with an invented name."""
    with pytest.raises(ValueError, match="names no backbone"):
        _backbone_commit({"tool": "tools/real_ft_run.py", "device": "cpu"})


def test_a_half_named_stand_in_is_refused() -> None:
    """``heads`` without ``hidden`` is not a backbone description, and ``scratch:128xNone``
    is not a name. The sentinel defaults made this reachable."""
    with pytest.raises(ValueError, match="names no backbone"):
        _backbone_commit({"hidden": 128, "device": "cpu"})


def test_an_absolute_path_in_the_snapshot_field_is_refused() -> None:
    """The second defect on the same line: ``str(backbone)`` rather than
    ``tower.snapshot.name``.

    An absolute path is ``/home/ubuntu/...`` on the rented GH200 and ``/Users/bharath/...``
    here, and this field feeds ``recipe_hash``. Storing one gives a bit-identical run two
    different protocol hashes depending on which machine ran it -- which is precisely what
    ``--hidden`` is refused under ``--real-backbone`` to prevent, a few lines earlier in
    the same function.
    """
    on_the_box = dict(
        REAL,
        backbone_snapshot=(
            "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/"
            "snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
        ),
    )
    with pytest.raises(ValueError, match="path rather than a revision"):
        _backbone_commit(on_the_box)


def test_the_keys_the_verdict_mirrors_are_enough_to_name_the_backbone() -> None:
    """``_record_verdict`` copies ``BACKBONE_KEYS`` out of the run instead of restating
    them, so the verdict row names the backbone the ft row named.

    That only holds while ``BACKBONE_KEYS`` covers everything ``_backbone_commit`` reads.
    A sixth key added to one and not the other would give the two rows of a single run two
    different backbones, which is the shape of the defect this file exists for.
    """
    for recipe in (STANDIN, REAL):
        mirrored = {k: recipe[k] for k in BACKBONE_KEYS if k in recipe}
        assert _backbone_commit(mirrored) == _backbone_commit(recipe)


# -- the objective's two channels --------------------------------------------------------
#
# Rung 0 reached its majority-class baseline four times while its total loss fell 90%,
# because the span channel opened 6.2x above the choice channel and the two were summed
# unweighted. `AUDIT/rung0-span-weight-2026-09-20.json` has the sweep. This path sums the
# same way, and what follows is what stops the same failure being invisible here.


def test_the_effective_ratio_is_what_is_recorded_not_the_raw_losses() -> None:
    """``span_weight`` is half the quantity. The gradient sees ``weight * span`` against
    ``letter``, so a run at weight 0.05 with a 6:1 raw ratio is a 0.3:1 objective and is
    not in the regime that collapsed rung 0. Recording the raw ratio would call those two
    runs the same."""
    from real_ft_run import _channel_balance

    heavy = _channel_balance({"letter_first": 2.0, "span_first": 12.0, "span_weight": 1.0})
    light = _channel_balance({"letter_first": 2.0, "span_first": 12.0, "span_weight": 0.05})
    assert isinstance(heavy, Ran) and isinstance(light, Ran)
    assert heavy.value == pytest.approx(6.0)
    assert light.value == pytest.approx(0.3)


def test_a_plan_with_only_one_channel_has_no_ratio() -> None:
    """``--max-width 479`` selects only bucket-0 batches and no letter row in this corpus is
    shorter than 1,359 tokens, so that plan's letter channel is empty and ``letter_first`` is
    nan. ``NotRun``: a substituted 0.0 would claim a balanced objective that was never
    measured, which is the failure ``_floor_state`` already exists to prevent."""
    from real_ft_run import _channel_balance

    state = _channel_balance(
        {"letter_first": float("nan"), "span_first": 12.0, "span_weight": 1.0}
    )
    assert isinstance(state, NotRun)
    assert not hasattr(state, "passed")


def test_a_zero_letter_channel_is_unmeasured_not_infinite() -> None:
    """Not a ZeroDivisionError and not ``inf``. A letter loss of exactly 0.0 at the first
    micro-batch says the ratio is undefined, and ``inf`` in a ledger row would read as a
    measured domination rather than an absent measurement -- besides not being JSON."""
    from real_ft_run import _channel_balance

    state = _channel_balance({"letter_first": 0.0, "span_first": 12.0, "span_weight": 1.0})
    assert isinstance(state, NotRun)
    assert "not defined" in state.reason


def test_the_balance_is_recorded_and_not_gated() -> None:
    """Rung 0's collapse threshold was measured on a 1.5M-parameter byte model. This path
    trains a 1.4B backbone, and carrying that bar across is an inference wearing a gate's
    clothes. The metric passes at any finite ratio; the number is the point."""
    from real_ft_run import _channel_balance

    catastrophic = _channel_balance(
        {"letter_first": 2.2, "span_first": 434.0, "span_weight": 1.0}
    )
    assert isinstance(catastrophic, Ran)
    assert catastrophic.passed
    assert catastrophic.value == pytest.approx(434.0 / 2.2)
    assert "not gated" in catastrophic.detail


def test_the_stand_in_and_the_real_step_weight_the_span_channel_the_same_way() -> None:
    """The drift the backbone handoff warned about, closed.

    ``QwenDecisionStep`` carried a ``span_weight`` that no caller in this repository ever
    passed, and ``RealFtStep`` carried none at all -- so the tool's two branches summed
    their channels by two different rules while claiming to exercise one contract. A
    stand-in that cannot reproduce the real step's objective cannot rehearse it.
    """
    import inspect

    from real_ft_run import RealFtStep

    from qd_train.backbone import QwenDecisionStep

    for step in (RealFtStep, QwenDecisionStep):
        assert "span_weight" in inspect.signature(step.__init__).parameters
    source = inspect.getsource(RealFtStep.accumulate_span)
    assert "self.span_weight * span_loss" in source


def test_the_stand_in_refuses_a_non_positive_span_weight() -> None:
    """The same refusal ``QwenDecisionStep`` makes, in the same words: zero would train the
    span head on nothing while its loss still appeared in the log."""
    from real_ft_run import RealFtStep

    with pytest.raises(ValueError, match="span_weight must be positive"):
        RealFtStep(
            seed=0, device="cpu", vocab=32, width=16, hidden=8, heads=2, lr=1e-3,
            span_weight=0.0,
        )


def test_the_span_weight_is_part_of_the_recipe() -> None:
    """Two runs that summed their channels by different rules are not one protocol. Without
    this key a sweep over the objective hashes to a single row that contradicts itself --
    the same defect fixed in ``tools/rung0_real_run.py`` the same day."""
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert '"span_weight": span_weight,' in source
    assert "span_weight=args.span_weight" in source
