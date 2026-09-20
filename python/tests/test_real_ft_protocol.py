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
