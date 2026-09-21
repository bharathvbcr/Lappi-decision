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
from real_ft_run import main as real_ft_main  # noqa: E402

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

    heavy = _channel_balance(
        {"letter_at_first_joint_batch": 2.0, "span_at_first_joint_batch": 12.0,
         "span_weight": 1.0}
    )
    light = _channel_balance(
        {"letter_at_first_joint_batch": 2.0, "span_at_first_joint_batch": 12.0,
         "span_weight": 0.05}
    )
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
        {"letter_at_first_joint_batch": float("nan"),
         "span_at_first_joint_batch": 12.0, "span_weight": 1.0}
    )
    assert isinstance(state, NotRun)
    assert not hasattr(state, "passed")


def test_a_zero_letter_channel_is_unmeasured_not_infinite() -> None:
    """Not a ZeroDivisionError and not ``inf``. A letter loss of exactly 0.0 at the first
    micro-batch says the ratio is undefined, and ``inf`` in a ledger row would read as a
    measured domination rather than an absent measurement -- besides not being JSON."""
    from real_ft_run import _channel_balance

    state = _channel_balance(
        {"letter_at_first_joint_batch": 0.0, "span_at_first_joint_batch": 12.0,
         "span_weight": 1.0}
    )
    assert isinstance(state, NotRun)
    assert "not defined" in state.reason


def test_the_balance_is_recorded_and_not_gated() -> None:
    """Rung 0's collapse threshold was measured on a 1.5M-parameter byte model. This path
    trains a 1.4B backbone, and carrying that bar across is an inference wearing a gate's
    clothes. The metric passes at any finite ratio; the number is the point."""
    from real_ft_run import _channel_balance

    catastrophic = _channel_balance(
        {"letter_at_first_joint_batch": 2.2,
         "span_at_first_joint_batch": 434.0, "span_weight": 1.0}
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


# -- the floor's explanation, decided rather than asserted --------------------------------


def test_a_zero_floor_is_not_explained_as_ln_2() -> None:
    """The defect this closes, caught in a ledger row.

    The sentence was a constant: "That floor is ln 2 because every span row in this corpus
    has a prompt-identical twin with a contradictory gold". True of the shard set of
    2026-09-20. Written unchanged into a row measuring a floor of 0.000000 on the shard set
    of 2026-09-21, whose 78 span rows carry no contradictory twin at all.
    """
    from real_ft_run import _span_floor_cause

    said = _span_floor_cause(0.0, 0.0)
    assert "ln 2" not in said
    assert "0.0" in said
    assert "optimisation, not the corpus" in said


def test_the_twin_signature_is_reported_when_the_numbers_show_it() -> None:
    """The 2026-09-20 shard set really did have it: a plan-level floor of 0.6931 -- ln 2 --
    against a mean per-batch floor of 0.2237, because the sampler split most of the pairs.
    The explanation has to survive being made conditional."""
    from real_ft_run import _span_floor_cause

    said = _span_floor_cause(0.6931, 0.2237)
    assert "twin" in said
    assert "3.1x" in said
    assert "not the bound" in said


def test_a_floor_that_lives_inside_batches_says_so() -> None:
    """The third case, which the constant could not express at all: something unfittable,
    but visible within a batch rather than only across the plan."""
    from real_ft_run import _span_floor_cause

    said = _span_floor_cause(0.30, 0.29)
    assert "within batches" in said
    assert "twin" not in said


def test_the_ratio_comes_from_one_micro_batch_not_two_separate_firsts() -> None:
    """The imprecision this closes, caught by two runs disagreeing.

    ``letter_first`` and ``span_first`` are each the opening value of their OWN log, and a
    span-free batch logs a letter loss beside a span of 0.0 -- so the two can be a step
    apart. A ratio is a statement about one gradient, and a ratio of two numbers from two
    different batches is not that. Measured: on the same shard set and seed the two
    readings gave 39.73:1 and 260.4:1, which is not a difference a rounding can explain.
    """
    from real_ft_run import _channel_balance

    run = {
        # What the independent-firsts reading would have used: the letter channel's own
        # opening value, from a batch with no span row at all.
        "letter_first": 6.9002,
        "span_first": 274.1343,
        # What one micro-batch actually carried.
        "letter_at_first_joint_batch": 2.0636,
        "span_at_first_joint_batch": 537.4454,
        "span_weight": 1.0,
    }
    state = _channel_balance(run)
    assert isinstance(state, Ran)
    assert state.value == pytest.approx(537.4454 / 2.0636, rel=1e-6)
    assert state.value != pytest.approx(274.1343 / 6.9002, rel=1e-3)
    assert "FIRST MICRO-BATCH that carried both" in state.detail


def test_a_plan_whose_batches_never_carry_both_channels_has_no_ratio() -> None:
    """``_train`` puts nan in both fields when no micro-batch had a live letter AND a live
    span. That is a plan of span-free batches, or of span-only ones -- neither has a ratio,
    and neither is evidence that the span channel did not dominate."""
    from real_ft_run import _channel_balance

    state = _channel_balance({
        "letter_at_first_joint_batch": float("nan"),
        "span_at_first_joint_batch": float("nan"),
        "span_weight": 1.0,
    })
    assert isinstance(state, NotRun)
    assert "carried both channels at once" in state.reason


def test_the_first_joint_batch_is_the_first_span_batch() -> None:
    """Both steps append to both logs on every micro-batch -- ``accumulate`` logs a span of
    0.0, ``accumulate_span`` logs both -- so the batches where both are positive are exactly
    the span batches, and the ratio is about the first of them."""
    from real_ft_run import _first_joint_batch

    # two span-free batches, then one carrying both, then another
    letters = [2.1, 2.0, 1.9, 1.8]
    spans = [0.0, 0.0, 537.4, 300.0]
    assert _first_joint_batch(letters, spans) == (1.9, 537.4)


def test_a_plan_with_no_span_batch_has_no_joint_batch() -> None:
    from real_ft_run import _first_joint_batch

    assert _first_joint_batch([2.1, 2.0], [0.0, 0.0]) is None


def test_mismatched_logs_do_not_destroy_a_finished_run() -> None:
    """The obvious spelling is ``zip(..., strict=True)``, which is right about the invariant
    and wrong about the consequence: it raises AFTER ``train_ft`` has returned, so a broken
    pair of logs would take the verdict row with it -- the evaluation, both floor gates, the
    decodes -- for a number that is one line of a detail string.

    ``None`` here becomes nan in the run dict and ``NotRun`` in the ledger, which is what a
    measurement that could not be made is. Fail closed on the claim, not on the run.
    """
    from real_ft_run import _channel_balance, _first_joint_batch

    assert _first_joint_batch([2.1, 2.0, 1.9], [0.0, 537.4]) is None
    downstream = _channel_balance({
        "letter_at_first_joint_batch": float("nan"),
        "span_at_first_joint_batch": float("nan"),
        "span_weight": 1.0,
    })
    assert isinstance(downstream, NotRun)


# -- checkpointing, the hook nobody passed ------------------------------------------------
#
# `HANDOFF/resume-2026-09-20.md`: "`Checkpoint.write` and `Checkpoint.read` are called in
# exactly two files, both of them test files. No driver in `tools/` writes a checkpoint, so
# nothing on disk survives a kill ... which is still nobody's job." These make it this
# tool's job, and refuse the three ways of believing it is done when it is not.


def test_the_stand_in_may_not_write_a_checkpoint_it_could_never_restore(tmp_path) -> None:
    """Not a limitation being worked around -- the point.

    ``RealFtStep.load_state`` raises by design: "a checkpoint it cannot restore would be a
    silent lie". A stand-in run that wrote one would leave a file on disk that looks like a
    resume point and is not, which is the exact failure that raise exists to prevent.
    """
    for flags in (
        ["--checkpoint-every", "10"],
        ["--checkpoint-dir", str(tmp_path)],
        ["--checkpoint-every", "10", "--checkpoint-dir", str(tmp_path)],
    ):
        with pytest.raises(SystemExit, match="need --real-backbone"):
            real_ft_main(["--out", str(tmp_path), *flags])


def test_an_interval_without_a_destination_is_refused(tmp_path) -> None:
    """A run that believes it is checkpointing and is not is worse than one that knows it
    is not: the first only finds out when it is killed."""
    with pytest.raises(SystemExit, match="nowhere to write"):
        real_ft_main([
            "--out", str(tmp_path), "--real-backbone", str(tmp_path),
            "--checkpoint-every", "10",
        ])


def test_a_destination_without_an_interval_is_refused(tmp_path) -> None:
    """The mirror image: the loop never calls ``on_checkpoint`` and the directory stays
    empty, which looks like a run that had nothing worth saving."""
    with pytest.raises(SystemExit, match="would never"):
        real_ft_main([
            "--out", str(tmp_path), "--real-backbone", str(tmp_path),
            "--checkpoint-dir", str(tmp_path / "ck"),
        ])


def test_resume_needs_a_step_that_can_restore_and_a_file_that_exists(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--resume-from needs --real-backbone"):
        real_ft_main([
            "--out", str(tmp_path), "--resume-from", str(tmp_path / "nope.json"),
        ])
    with pytest.raises(SystemExit, match="does not exist"):
        real_ft_main([
            "--out", str(tmp_path), "--real-backbone", str(tmp_path),
            "--resume-from", str(tmp_path / "nope.json"),
        ])


def test_the_control_carries_the_interval_into_the_loop(tmp_path) -> None:
    """``RunControl.checkpoint_every`` is what makes ``train_ft`` call the hook at all. It
    defaulted to 0 and the tool never set it, so passing ``on_checkpoint`` alone would have
    changed nothing."""
    from real_ft_run import _control

    assert _control(100, device="cpu", lr=1e-5).checkpoint_every == 0
    assert _control(
        100, device="cpu", lr=1e-5, checkpoint_every=25
    ).checkpoint_every == 25


# --- which run a checkpoint belongs to ------------------------------------------------
#
# `train_ft` refuses a resume whose seed, epoch or schedule differs from the run it is
# handed to, and rehashes the skipped prefix on top. That covers everything the LOOP can
# know. It does not cover which of this tool's arms the file came from, because `Checkpoint`
# does not carry a tag or a device -- correctly, those are this tool's concepts. So the
# driver owns that routing, and these are what hold it to it.


def test_two_devices_at_one_seed_do_not_share_a_checkpoint_path() -> None:
    """The collision the trainer cannot catch.

    A resume is refused across seeds and across schedules. ``cpu`` and ``mps`` at one seed
    on one arm differ in NEITHER: same seed, same schedule, same batch order. A filename
    without the device therefore lets one device's checkpoint overwrite the other's and
    hands the trainer a file that passes every check it has, taken on other hardware.
    """
    from real_ft_run import ARM_DEVICES, ARM_TAGS, _checkpoint_name

    names = [
        _checkpoint_name(tag, seed, device)
        for tag in ARM_TAGS
        for seed in (1, 2, 3)
        for device in ARM_DEVICES
    ]
    assert len(names) == len(set(names)), (
        "two cells of the (tag x seed x device) product write to one path: "
        f"{sorted(n for n in names if names.count(n) > 1)}"
    )
    assert _checkpoint_name("memorise", 1, "cpu") != _checkpoint_name("memorise", 1, "mps")


def test_the_name_a_checkpoint_is_written_under_is_the_name_that_is_parsed_back() -> None:
    """``_checkpoint_name`` and ``_resume_arm`` are inverses, or the routing is guesswork."""
    from pathlib import Path as _Path

    from real_ft_run import ARM_DEVICES, ARM_TAGS, _checkpoint_name, _resume_arm

    for tag in ARM_TAGS:
        for seed in (0, 7, 41):
            for device in ARM_DEVICES:
                name = _checkpoint_name(tag, seed, device)
                assert _resume_arm(_Path("/anywhere") / name) == (tag, seed, device)


def test_a_file_this_tool_did_not_write_is_refused_rather_than_guessed_at() -> None:
    from pathlib import Path as _Path

    from real_ft_run import _resume_arm

    for bad in (
        "latest.json",              # the obvious hand-written name
        "memorise-seed1.json",      # the spelling before the device was in it
        "memorise-seed1-tpu.json",  # a device this tool does not run
        "cpt-seed1-cpu.json",       # an arm this tool does not have
        "memorise-seedX-cpu.json",  # not a number
        "memorise-seed1-cpu.txt",   # not a checkpoint
    ):
        with pytest.raises(ValueError):
            _resume_arm(_Path(bad))


def test_a_checkpoint_is_refused_when_this_run_has_no_cell_for_it(tmp_path) -> None:
    """Refused against the PLAN, at argv time.

    Handing one checkpoint to every cell of the product and letting the trainer sort it out
    resumes one cell and aborts the sweep at the next -- after a 24-second tower load, with
    a traceback rather than an answer. These three are decidable from argv.
    """
    ckpt = tmp_path / "memorise-seed3-cuda.json"
    ckpt.write_text("{}", encoding="utf-8")
    common = ["--out", str(tmp_path), "--real-backbone", str(tmp_path),
              "--resume-from", str(ckpt)]

    with pytest.raises(SystemExit, match="checkpoint from cuda"):
        real_ft_main([*common, "--devices", "cpu", "--seeds", "3"])
    with pytest.raises(SystemExit, match="at seed 3"):
        real_ft_main([*common, "--devices", "cuda", "--seeds", "1", "2"])

    epoch_ckpt = tmp_path / "epoch-seed1-cpu.json"
    epoch_ckpt.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="--epoch was not passed"):
        real_ft_main([
            "--out", str(tmp_path), "--real-backbone", str(tmp_path),
            "--resume-from", str(epoch_ckpt), "--devices", "cpu", "--seeds", "1",
        ])
