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


# -- whether a run's numbers can be got back ---------------------------------------------
#
# GAP-FT-RUNS-ARE-NOT-REPRODUCIBLE-AT-A-FIXED-SEED. Once fee0f9a made the seed determine the
# starting point, eight repeats of ONE configuration at ONE seed opened at an identical
# 518.5886 and still finished with final span losses of 0.000000 .. 1.505752, three of them
# over the 0.05 bar -- the whole range the span-weight sweep had attributed to its arms,
# produced by a single arm. The letter channel's spread over the same eight runs was
# 1.0e-5 .. 1.3e-4, ~380x below its own bar.
#
# Then measured directly, which reversed the reading taken from phase 4 alone. Eight repeats
# of that same configuration WITH --deterministic came back bit-identical: 0.000078 letter
# and 0.000000 span to every digit, at a cost of 131.3s against 111.4s. So the kernels are
# the SOURCE. Phase 4 -- span gap [0.00000, 0.00012] at 512 steps without determinism -- says
# non-convergence is the AMPLIFIER: removing either removes the symptom, which is why
# inferring the cause from phase 4 alone was wrong.


def test_the_cublas_workspace_is_set_from_argv_because_argparse_is_too_late() -> None:
    """cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when it initialises, which is the first
    matmul -- long before ``main`` parses anything. Setting it from parsed arguments would
    be a setting that looks applied and is not, and with deterministic algorithms in force
    torch raises at the first addmm instead."""
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert 'if "--deterministic" in sys.argv:' in source
    assert 'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")' in source

    argv_at = source.index('if "--deterministic" in sys.argv:')
    torch_at = source.index("    import torch")
    assert argv_at < torch_at, (
        "the workspace is configured after torch is imported, so a CUDA context may "
        "already exist by then and the setting is decoration"
    )


def test_an_ordinary_run_does_not_get_the_workspace_set_for_it() -> None:
    """The other half: a 32 MB cuBLAS workspace on a run that did not ask for determinism
    is a cost paid for nothing, and a global that changes under runs that never mentioned it
    is how two runs come to differ for a reason neither recorded."""
    import os
    import subprocess
    import sys as _sys

    tools = str(REPO / "tools")
    probe = (
        "import os, sys;"
        "sys.argv=['real_ft_run.py','--out','/nowhere'];"
        f"sys.path.insert(0, {tools!r});"
        "import real_ft_run;"
        "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'UNSET'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    out = subprocess.run(
        [_sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "UNSET", out.stdout


def test_the_recipe_separates_a_deterministic_run_from_an_ordinary_one() -> None:
    """Two runs that used different kernels for the same matmul are not one protocol. The
    measured spread between them is larger than several of the effects this tool is used to
    look for, so collapsing them into one ``recipe_hash`` would put two populations in one
    row."""
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert '"deterministic": deterministic,' in source
    assert "deterministic=args.deterministic," in source


def test_a_run_that_did_not_ask_for_determinism_reports_not_run_rather_than_passing() -> None:
    """The discipline this repository is built on, at the one place it had not reached.

    "This run's numbers are reproducible" and "nobody checked" must not be one string in a
    ledger row -- and until this metric existed, a row said nothing at all, which reads as
    the former.
    """
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert 'recorder.metric(\n        "deterministic_kernels",' in source
    marker = '        "deterministic_kernels",'
    block = source[source.index(marker) : source.index(marker) + 2000]
    assert "if deterministic" in block and "else NotRun(" in block, (
        "the metric must be tri-state: a run that did not request deterministic kernels "
        "established nothing about reproducing its numbers"
    )
    assert "1.505752" in block and "0.00012" in block, (
        "the NotRun reason should carry BOTH measured numbers, because either one alone is "
        "misleading. The 128-step spread without determinism is what the kernels cost; the "
        "512-step collapse WITHOUT determinism is why a long run does not show it. A reason "
        "carrying only the first overstates how often this matters, and one carrying only "
        "the second reads as though the kernels were not the cause -- which is the error "
        "this string was corrected for."
    )


# -- the price of the machine, decided at argv time ----------------------------------------
#
# GAP-EVERY-RUN-PRICED-ITSELF-AT-ZERO-ON-ZERO-GPUS. This tool passed
# `usd_per_hour=0.0, n_gpus=0, instance=f"local-{device}"` on every device, so every GH200
# run recorded itself as a local run on zero GPUs at zero dollars an hour -- and at those
# two values `requires_human_approval` is False for any cap.


def test_a_cuda_run_without_a_price_is_refused_before_the_tower_loads(tmp_path) -> None:
    """At argv time, not at the first optimizer step.

    ``CostEstimate.for_device`` refuses the same case, but it is reached per-arm inside
    ``_train`` -- after a 24-second tower load, on a box being billed by the second. The
    argv check costs nothing and fires before any of that.
    """
    with pytest.raises(SystemExit, match="needs --instance and --usd-per-hour"):
        real_ft_main([
            "--out", str(tmp_path), "--real-backbone", str(tmp_path),
            "--devices", "cuda",
        ])
    with pytest.raises(SystemExit, match="needs --instance and --usd-per-hour"):
        real_ft_main([
            "--out", str(tmp_path), "--real-backbone", str(tmp_path),
            "--devices", "cuda", "--instance", "lambda-1xGH200",
        ])


def test_the_refusal_says_why_the_default_was_not_harmless(tmp_path) -> None:
    """"You forgot an argument" would get the argument added and the number guessed. The
    message has to carry the consequence: these are the two values that switch rule 4 off."""
    with pytest.raises(SystemExit) as excinfo:
        real_ft_main([
            "--out", str(tmp_path), "--real-backbone", str(tmp_path), "--devices", "cuda",
        ])
    message = str(excinfo.value)
    assert "requires_human_approval" in message and "ANY cap" in message
    assert "--usd-per-hour 1.49" in message, (
        "an operator on a rented box should be able to copy a working invocation out of "
        "the refusal rather than go and read the source"
    )


def test_a_local_run_needs_no_price_and_is_not_refused(tmp_path) -> None:
    """The control. A refusal that fired on cpu too would be found immediately and routed
    around by whoever hit it on a Mac -- and pricing an already-bought machine is the case
    the zero is honest for."""
    with pytest.raises(SystemExit) as excinfo:
        real_ft_main(["--out", str(tmp_path), "--devices", "cpu", "--checkpoint-every", "10"])
    # Refused for the checkpoint reason, which proves argv parsing got past the price check.
    assert "need --real-backbone" in str(excinfo.value)


def test_the_tool_no_longer_prices_every_device_at_zero() -> None:
    """Checked against the source, because the defect was a literal that read as deliberate.
    `_control` carried `usd_per_hour=0.0, n_gpus=0` for every device, under a docstring
    saying a rented machine sets a real rate here."""
    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    assert 'usd_per_hour=0.0, n_gpus=0' not in source, (
        "the zero-rate zero-GPU literal is back; it prices a rented box at nothing and "
        "makes requires_human_approval False for any cap"
    )
    assert "CostEstimate.for_device(" in source
    assert "approved_by=approved_by," in source, (
        "RunControl refuses a run that needs a human and has no approved_by; if the tool "
        "never passes one, that refusal can be satisfied only by not needing approval"
    )


# -- what code actually ran, beside a code_commit that cannot say -------------------------
#
# GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN, raised by the concurrent lane after measuring
# the GH200: the box is 8b9df39 plus 72 modified-or-untracked paths and 6894 insertions, so
# every GPU row's code_commit is "8b9df39-dirty" -- one bit standing for an unbounded amount
# of divergence, on the rows that carry every training number this project has. The box is
# synced by copying files into a clone pinned at an old commit, so -dirty is its normal
# state rather than an exception.


def test_both_row_kinds_record_what_ran() -> None:
    """An ft row and its verdict row are written at different moments by one process. A row
    that cannot rule out an edit between them is the hole being closed, and recording it on
    only one of the two would leave exactly that hole open.

    Asserted against the source because there is no cheap way to run both paths here, and
    because the failure being guarded is a call site quietly disappearing in a refactor --
    which is how this tool came to be the only one of four that recorded it at all.

    It used to count TWO hand-written `code_that_ran` calls, one per row kind. Counting
    them was the right check for an arrangement in which each row kind recorded its own,
    and that arrangement had a hole underneath it: both calls sat at the END of their
    blocks, so a run killed part-way wrote a row with no digest at all -- which is what
    happened to the SIGTERM'd fit in `ledger/gh200-commitpackft-2026-09-22.jsonl`.

    Now there is one construction site, `_recorder`, which both row kinds go through, and
    `RunRecorder.__enter__` takes the digest on entry. "Recorded on only one of the two"
    stopped being reachable rather than being checked for, which is the stronger outcome;
    what has to hold instead is that there is still exactly one site and it still hands
    over the entry point.
    """
    import re

    source = (REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8")
    sites = re.findall(r"RunRecorder\(", source)
    assert len(sites) == 1, (
        f"expected one recorder construction both row kinds share, found {len(sites)}; "
        "two sites is two chances to pass entry_point to only one of them"
    )
    assert re.search(r"entry_point\s*=\s*Path\(__file__\)", source), (
        "the shared recorder does not hand over its entry point, so neither row kind "
        "records what produced it"
    )
    # Every row kind reaches that one site: the ft row, its verdict row, and the eval row
    # --score-val writes. Named rather than counted: a count of 2 went stale the moment a
    # third kind was added through the SAME door, and a count cannot tell a moved site from
    # a new one.
    import ast

    writers = {
        fn.name
        for fn in ast.walk(ast.parse(source))
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_recorder"
    }
    # run_needle_control (2026-09-30): the --needle-control diagnostic's own eval row.
    # _record_shuffled_label (2026-09-30): the --shuffled-label control row, which carries
    # the protocol of the eval row it supplements through _recorder's `protocol`.
    # run_ood_diagnostic (2026-10-01): a --score-plan kind's OOD-only diagnostic row.
    assert writers == {
        "_train", "_record_verdict", "_record_score", "run_needle_control",
        "_record_shuffled_label", "run_ood_diagnostic",
    }, (
        "expected the ft, verdict, eval, needle-control, shuffled-label control and OOD "
        f"diagnostic rows to come from _recorder, got {sorted(writers)}"
    )


def test_code_fingerprint_still_defaults_to_qd_data() -> None:
    """The generalisation must not move what every shard header records. `qd_data`'s
    fingerprint is checked at `ShardReader.open()` against sets already on disk, so a
    changed default would invalidate them all."""
    from qd_data.fingerprint import code_fingerprint

    default = code_fingerprint()
    assert "render.py" in default and "mixture.py" in default
    assert not any(name in default for name in ("trainer.py", "backbone.py"))


def test_a_directory_with_no_sources_is_refused_rather_than_returning_empty(tmp_path) -> None:
    """An empty mapping from a path that holds no sources is indistinguishable from a
    package that has none, and both would record 'nothing changed' forever."""
    from qd_data.fingerprint import code_fingerprint

    with pytest.raises(RuntimeError, match="refusing to return an empty fingerprint"):
        code_fingerprint(tmp_path)


def test_the_toy_runner_this_tool_imports_from_refuses_a_rented_device() -> None:
    """The fourth of the four. ``real_ft_run`` reuses ``ft_toy_run``'s floor formulas by
    import, and the two files also shared the zero-price literal -- so fixing the importer
    and leaving the imported one is the accumulate-instead-of-replace failure in miniature.

    cpu is asserted to be unchanged rather than merely un-refused: ``for_device`` has to
    leave the honest case byte-for-byte as it was, or every local run's rows move for a
    change that was about rented hardware.
    """
    import ft_toy_run

    local = ft_toy_run._control(100, device="cpu")
    assert local.cost.usd_per_hour == 0.0
    assert local.cost.n_gpus == 0
    assert local.cost.instance == "local-cpu"

    with pytest.raises(ValueError, match="hardware being paid for by the hour"):
        ft_toy_run._control(100, device="cuda")


# -- the driver half of the resume-order refusal -------------------------------------------
#
# `train_ft` refuses a resume whose corpus order differs from the one the checkpoint was cut
# from -- it rehashes the skipped prefix and compares. test_backbone.py holds that, through
# the real step. What it cannot hold is whether THIS tool ever hands the checkpoint over, and
# a driver that quietly dropped it would run from scratch while reporting a resumed run.
#
# That is not hypothetical here. real_ft_run.py:1566 carries the note "The hook nobody
# passed": `on_checkpoint` existed, was correct, and was never wired, so checkpointing was
# dead for as long as nobody looked. Same shape, one argument over.


def _real_ft_tree():
    import ast

    return ast.parse((REPO / "tools" / "real_ft_run.py").read_text(encoding="utf-8"))


def test_the_driver_hands_its_checkpoint_to_the_trainer() -> None:
    """`_train` must pass `resume_from` into `train_ft`.

    Asserted through the AST rather than on the text, because the argument being present
    is the claim and its formatting is not: this call has been rewrapped twice today.
    """
    import ast

    for node in ast.walk(_real_ft_tree()):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_train"):
            continue
        assert "resume_from" in {a.arg for a in node.args.kwonlyargs} | {
            a.arg for a in node.args.args
        }, "_train cannot be told what to resume from"
        calls = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "train_ft"
        ]
        assert calls, "_train no longer calls train_ft; this test is checking nothing"
        for call in calls:
            assert "resume_from" in {kw.arg for kw in call.keywords}, (
                "_train calls train_ft without resume_from, so --resume-from is accepted, "
                "validated, routed to an arm and then dropped -- the run starts from "
                "scratch and every check train_ft has passes vacuously, because there is "
                "no checkpoint for it to compare against"
            )
        return
    raise AssertionError("_train is gone from real_ft_run.py")


def test_both_arms_route_the_checkpoint_not_just_the_first() -> None:
    """Every `_train` call site, because the tool trains two arms.

    An arm that dropped `resume_from` would be the one-of-N shape this repository found
    four times on 2026-09-21, and it would be invisible: the resumed arm would pass, the
    other would silently restart, and the row would say both were resumed.
    """
    import ast

    sites = [
        call
        for call in ast.walk(_real_ft_tree())
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_train"
    ]
    assert len(sites) >= 2, f"expected both arms to call _train, found {len(sites)}"
    missing = [
        site.lineno for site in sites if "resume_from" not in {kw.arg for kw in site.keywords}
    ]
    assert not missing, (
        f"_train call site(s) at line(s) {missing} do not pass resume_from, so that arm "
        "restarts from scratch while --resume-from reports a resume"
    )


def test_a_checkpoint_reaches_exactly_the_cell_it_was_cut_from() -> None:
    """The guard that makes the trainer's refusal meaningful rather than constant.

    Handing one checkpoint to every cell of the (tag x seed x device) product resumes the
    one it belongs to and aborts the rest on a seed or schedule mismatch -- so a tool that
    passed it everywhere would look correct on the arm that worked and fail the others for
    a reason that is not a defect. Each site is therefore conditioned on its own cell.
    """
    import ast

    for call in ast.walk(_real_ft_tree()):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_train"
        ):
            continue
        keyword = next(kw for kw in call.keywords if kw.arg == "resume_from")
        assert isinstance(keyword.value, ast.IfExp), (
            f"the _train call at line {call.lineno} passes resume_from unconditionally; it "
            "has to be conditioned on the cell the checkpoint was taken from, or every "
            "other cell aborts on a mismatch that is not a defect"
        )
        # And the condition names a cell, rather than something incidental like the device.
        source = ast.unparse(keyword.value.test)
        assert "resume_cell" in source, (
            f"line {call.lineno}: the condition is {source!r}, which does not compare "
            "against the cell the checkpoint was taken from"
        )
