"""The parts of ``tools/rung0_real_run.py`` that decide whether its numbers mean anything.

The tool's claim is "rung 0 scored X on files it never saw". Three things have to hold for
that sentence to be true rather than merely printed, and each is tested here:

* the split is disjoint **by file**, and stays that way as the corpus grows;
* the accuracy is reported against the majority-class baseline, because
  ``MUTATION_CLASSES`` is skewed and a model that answers ``stub`` every time scores 51.5%
  on this corpus;
* an empty evaluation set reads as ``NotRun``, not as 0%.

Torch-gated: the tool imports ``torch`` at module scope for the model it trains. The repo
venv has no torch by design, so these run in the ``mac`` extra alongside the other trainer
tests.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import rung0_real_run as tool  # noqa: E402

from qd_train.tristate import NotRun, Ran  # noqa: E402


def _example(path: str, cls: str = "stub") -> dict[str, object]:
    """The two fields the split and the baseline actually read."""
    return {"function": {"path": path}, "class": cls}


@functools.cache
def _recipe_of_a_row() -> dict[str, object]:
    """The recipe off a row this tool actually wrote, checked against that row's own hash.

    Cached: one real run serves every test that asks, and the run is sized to seconds.

    The check is the point. Reading the recipe alone would pass for a tool that stored a
    dict unrelated to the one it hashed -- which is a worse row than one storing nothing,
    because it reads as an answer. Re-hashing the stored recipe with the tool's own
    spelling (``sort_keys=True``, no ``separators``) has to reproduce ``recipe_hash``.
    """
    tmp = Path(tempfile.mkdtemp())
    examples = tmp / "examples.jsonl"
    body = "".join(f"line {i} of the body\n" for i in range(12))
    rows = [
        {
            "id": f"f{f}-{k}",
            "function": {"repo": "qwen-decision", "path": f"src/m{f}.py",
                         "symbol": f"fn_{k}", "arity": 1},
            "language": "python",
            "class": ("stub", "logic", "cosmetic", "clean")[(f + k) % 4],
            "after": body,
            "span": None if (f + k) % 4 == 3 else {"start_line": 2, "end_line": 3},
            "silent": False, "hunk_constrained": False, "seed": f * 10 + k,
        }
        for f in range(8) for k in range(3)
    ]
    examples.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"corpus_sha256": "0" * 64, "n_examples": len(rows)}),
                        encoding="utf-8")
    out = tmp / "scratch"
    out.mkdir()
    ledger = tmp / "l.jsonl"
    # The gates decide whether rung 0 learned anything; on eight synthetic files it plainly
    # did not, and `main` says so with a non-zero exit after the row is already written.
    with contextlib.suppress(SystemExit):
        tool.main([
            "--out", str(out), "--rev", "HEAD", "--examples", str(examples),
            "--manifest-in", str(manifest), "--device", "cpu", "--seeds", "1",
            "--epochs", "1", "--width", "16", "--layers", "1", "--heads", "1",
            "--context-bytes", "512", "--batch-size", "2", "--val-share", "0.25",
            "--ledger", str(ledger),
        ])
    assert ledger.is_file(), "the tool wrote no ledger row to read a recipe from"
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    recipe = row.get("recipe")
    assert isinstance(recipe, dict) and recipe, (
        "the row records no recipe, so it can only say THAT two arms differ and never how"
    )
    restated = hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert restated == row["protocol"]["recipe_hash"], (
        f"the row's stored recipe hashes to {restated[:16]}… but the row claims "
        f"{row['protocol']['recipe_hash'][:16]}…: the recipe recorded is not the recipe "
        "that identifies this run, which reads as an answer and is not one"
    )
    return recipe


# -- the split ---------------------------------------------------------------------------


def test_the_split_is_disjoint_by_file() -> None:
    """The property the whole tool rests on. Mutations are generated per function, so one
    file yields many examples sharing context, naming and style; a random split would put
    siblings of a validation example into training and report memorisation as
    generalisation."""
    examples = [_example(f"src/mod{i}.py") for i in range(200) for _ in range(5)]
    train, val = tool.split_by_file(examples, val_share=0.25)
    train_paths = {e["function"]["path"] for e in train}  # type: ignore[index]
    val_paths = {e["function"]["path"] for e in val}  # type: ignore[index]
    assert train_paths and val_paths
    assert not (train_paths & val_paths)


def test_every_example_lands_on_exactly_one_side() -> None:
    """A split that silently drops rows would make both sides look cleaner than the corpus."""
    examples = [_example(f"src/mod{i}.py") for i in range(120)]
    train, val = tool.split_by_file(examples, val_share=0.3)
    assert len(train) + len(val) == len(examples)


def test_a_files_side_does_not_move_when_the_corpus_grows() -> None:
    """Hashed rather than shuffled, and this is why.

    If assignment depended on corpus order or size, adding files would move an old file
    across the split -- turning a previously held-out measurement into a training one
    without any visible change. The tool would still print "files it never saw".
    """
    small = [_example(f"src/mod{i}.py") for i in range(30)]
    large = small + [_example(f"src/new{i}.py") for i in range(300)]

    _, val_small = tool.split_by_file(small, val_share=0.25)
    _, val_large = tool.split_by_file(large, val_share=0.25)

    was_val = {e["function"]["path"] for e in val_small}  # type: ignore[index]
    still_val = {e["function"]["path"] for e in val_large}  # type: ignore[index]
    assert was_val <= still_val, f"{sorted(was_val - still_val)} moved out of validation"


def test_a_val_share_outside_the_open_unit_interval_is_refused() -> None:
    """0.0 and 1.0 both produce an empty side, which measures nothing while looking fine."""
    for share in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="val_share"):
            tool.split_by_file([_example("a.py")], val_share=share)


# -- the baseline ------------------------------------------------------------------------


class _Decision:
    """Only ``gold_option`` is read by the baseline; the real ByteDecision needs a context."""

    def __init__(self, gold_option: int) -> None:
        self.gold_option = gold_option


def test_the_baseline_is_the_majority_class_share() -> None:
    decisions = [_Decision(0)] * 7 + [_Decision(1)] * 2 + [_Decision(2)]
    share, name = tool.majority_baseline(decisions)
    assert share == pytest.approx(0.7)
    assert name == tool.MUTATION_CLASSES[0]


def test_a_model_at_the_baseline_does_not_pass() -> None:
    """The gate's whole point. On this corpus ``stub`` is 51.5% of the rows, so 51.5%
    accuracy is what answering one constant scores -- evidence of nothing."""
    at = tool._accuracy_gate(
        0.515, 0.515, n=160, what="choice", baseline_name="majority-class baseline"
    )
    assert isinstance(at, Ran)
    assert not at.passed

    above = tool._accuracy_gate(
        0.60, 0.515, n=160, what="choice", baseline_name="majority-class baseline"
    )
    assert isinstance(above, Ran)
    assert above.passed
    assert "+8.5%" in above.detail


def test_a_span_head_at_chance_does_not_pass() -> None:
    """The defect this tool shipped with for one run, pinned.

    A pointer's trivial answer is 1/candidates, never 0. The first version passed 0.0 as
    the span baseline and recorded ``passed=True`` for a span head scoring 0.6% over ~100
    candidate lines -- a head doing nothing, reported as a head that beat its bar. Ledger
    rows 8b7291cd, 3767e56b and d9005962 carry that verdict and cannot be rewritten.
    """
    chance = 1 / 100
    at_chance = tool._accuracy_gate(
        chance, chance, n=160, what="span start", baseline_name="uniform-pointer chance"
    )
    assert isinstance(at_chance, Ran)
    assert not at_chance.passed

    # The measured numbers from the 3-seed run, against the chance they have to beat.
    for measured in (0.025, 0.006, 0.031):
        state = tool._accuracy_gate(
            measured, chance, n=160, what="span start", baseline_name="uniform-pointer chance"
        )
        assert isinstance(state, Ran)
        # Not asserting these fail -- 2.5% and 3.1% do exceed 1% chance. What is asserted
        # is that the comparison happens against chance at all: with the old 0.0 baseline
        # every one of them passed, including 0.6%, which is BELOW chance.
        assert state.passed == (measured > chance)
    below = tool._accuracy_gate(
        0.006, chance, n=160, what="span start", baseline_name="uniform-pointer chance"
    )
    assert not below.passed, "0.6% is below 1% chance and must not report as passing"


def test_an_empty_evaluation_set_reads_as_not_run_not_as_zero() -> None:
    """0% and "nothing was measured" are different facts, and only one of them is about
    the model. ``NotRun`` has no ``passed`` field to be misread as a failure either."""
    state = tool._accuracy_gate(
        0.0, 0.5, n=0, what="span start", baseline_name="majority-class baseline"
    )
    assert isinstance(state, NotRun)
    assert not hasattr(state, "passed")


# -- batching ----------------------------------------------------------------------------


class _Context:
    def __init__(self, n: int) -> None:
        self.n_bytes_kept = n


class _Sized:
    def __init__(self, n: int) -> None:
        self.context = _Context(n)


def test_batches_are_length_homogeneous() -> None:
    """``plan_batch`` pads to the batch maximum, so one long row makes every short row in
    its batch pay the difference. Sorting first is what keeps a batch's width its own."""
    decisions = [_Sized(n) for n in (100, 10, 90, 20, 80, 30)]
    ordered = sorted(decisions, key=lambda d: d.context.n_bytes_kept)
    chunks = [ordered[i : i + 2] for i in range(0, len(ordered), 2)]
    widths = [max(d.context.n_bytes_kept for d in c) for c in chunks]
    assert widths == sorted(widths)
    # Sorted chunking wastes strictly less than taking them in arrival order.
    unsorted_chunks = [decisions[i : i + 2] for i in range(0, len(decisions), 2)]
    sorted_waste = sum(
        max(d.context.n_bytes_kept for d in c) * len(c)
        - sum(d.context.n_bytes_kept for d in c)
        for c in chunks
    )
    unsorted_waste = sum(
        max(d.context.n_bytes_kept for d in c) * len(c)
        - sum(d.context.n_bytes_kept for d in c)
        for c in unsorted_chunks
    )
    assert sorted_waste < unsorted_waste


def test_a_batch_size_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        tool.bucketed_batches([], batch_size=0, config=tool.ByteDeciderConfig())


# -- the context window ------------------------------------------------------------------


def test_examples_without_a_manifest_is_refused(tmp_path) -> None:
    """A corpus and its identity travel together.

    ``--examples`` without ``--manifest-in`` would train on a corpus whose
    ``data_snapshot_hash`` had to be invented, putting a fabricated value in the protocol
    every later comparison is made against; the reverse identifies a corpus it did not
    train on. Both are worse than refusing.
    """
    corpus = tmp_path / "examples.jsonl"
    corpus.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="go together"):
        tool.main(["--out", str(tmp_path), "--rev", "HEAD", "--examples", str(corpus)])
    with pytest.raises(SystemExit, match="go together"):
        tool.main(["--out", str(tmp_path), "--rev", "HEAD", "--manifest-in", str(corpus)])


def test_an_empty_pre_generated_corpus_is_refused(tmp_path) -> None:
    """Fail closed. An empty file would otherwise reach the split, produce two empty sides
    and stop with a message about the split rather than about the corpus."""
    corpus = tmp_path / "examples.jsonl"
    corpus.write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="nothing to train on"):
        tool.main([
            "--out", str(tmp_path), "--rev", "HEAD",
            "--examples", str(corpus), "--manifest-in", str(manifest),
        ])


def test_decisions_and_their_paths_stay_aligned_through_refusals() -> None:
    """The alignment bug this return value exists to prevent.

    ``decisions_of`` DROPS refused examples -- 1,599 raw became 763 at 8192 bytes -- so
    recovering a decision's source file by zipping against the input pairs each decision
    with the wrong file, and every per-file operation downstream is quietly wrong while
    looking fine. The paths are built where the dropped row is still in hand.
    """
    config = tool.ByteDeciderConfig(max_context_bytes=1024)
    good = {
        "id": "ex-1", "pool_id": "p", "repo": "r", "path": "src/keep.py", "language": "Python",
        "class": "stub", "op": "stub_body", "before": "def f():\n    return 1\n",
        "after": "def f():\n    pass\n", "span": {"start_line": 2, "end_line": 2},
        "function": {"repo": "r", "path": "src/keep.py", "symbol": "f", "arity": 0},
        "hunk_constrained": False,
    }
    malformed = {"id": "ex-2", "function": {}}  # missing required fields -> MalformedExample
    decisions, refused, paths = tool.decisions_of([malformed, good, malformed], config=config)
    assert refused, "the malformed rows must be counted, not silently dropped"
    assert len(decisions) == len(paths), "a decision without its path is the alignment bug"
    if decisions:
        assert paths[0] == "src/keep.py", (
            f"the surviving decision came from src/keep.py but its path reads {paths[0]!r}; "
            "the refused rows shifted the pairing"
        )


def test_a_smaller_subsample_is_a_subset_of_a_larger_one() -> None:
    """Why the file order is hashed rather than shuffled or taken as-is.

    A learning curve whose 25% and 50% points are drawn from unrelated samples measures
    sampling noise alongside size, and the curve is unreadable. Hashing makes each point a
    strict subset of the next.
    """
    import hashlib as _h

    names = [f"src/mod{i}.py" for i in range(40)]
    ordered = sorted(names, key=lambda n: _h.sha256(n.encode("utf-8")).hexdigest())
    quarter = set(ordered[: max(1, round(0.25 * len(names)))])
    half = set(ordered[: max(1, round(0.50 * len(names)))])
    whole = set(ordered[: max(1, round(1.00 * len(names)))])
    assert quarter < half < whole
    assert whole == set(names)


def test_the_choice_floor_is_the_label_distribution_entropy() -> None:
    """A uniform four-class set floors at ln 4; a one-class set floors at 0.

    This is the instrument that made the real finding legible: the choice loss converged to
    1.128 nats against a measured floor of 1.130, so the head had learned the class prior
    exactly. The TOTAL loss fell 14.5 -> 1.4 over the same run and looked like training.
    """
    import math as _m

    four_ways = [_Decision(i % 4) for i in range(400)]
    assert tool.label_entropy(four_ways) == pytest.approx(_m.log(4), abs=1e-9)

    one_class = [_Decision(0) for _ in range(50)]
    assert tool.label_entropy(one_class) == pytest.approx(0.0, abs=1e-12)

    skewed = [_Decision(0)] * 80 + [_Decision(1)] * 65 + [_Decision(2)] * 31 + [_Decision(3)] * 5
    assert tool.label_entropy(skewed) == pytest.approx(1.1300, abs=5e-4)


def test_an_empty_training_set_has_no_floor_rather_than_a_zero_one() -> None:
    """0.0 is the entropy of a one-class corpus, which is a real and very different state
    from having no corpus. Returning it for both would make a head that learned nothing
    from nothing look like one that learned a deterministic rule."""
    with pytest.raises(ValueError, match="zero decisions"):
        tool.label_entropy([])


def test_a_choice_head_at_the_prior_does_not_pass() -> None:
    """The gate that would have caught this run without anyone reading a loss column."""
    at_floor = tool._prior_gate(1.128, 1.130, n=181)
    assert isinstance(at_floor, Ran)
    assert not at_floor.passed
    assert "learned the class prior" in at_floor.detail

    learned = tool._prior_gate(0.60, 1.130, n=181)
    assert isinstance(learned, Ran)
    assert learned.passed

    # Just inside the margin is still the floor: 1% of a nat is the precision at which
    # these two are the same number.
    assert not tool._prior_gate(1.125, 1.130, n=181).passed


def test_no_training_decisions_reads_as_not_run(  ) -> None:
    state = tool._prior_gate(1.0, 1.0, n=0)
    assert isinstance(state, NotRun)
    assert not hasattr(state, "passed")


def test_the_default_context_is_wider_than_the_config_default() -> None:
    """Measured, not preferred: at ``ByteDeciderConfig``'s 1024 only 9.0% of a real
    qd-mutate corpus survives ``SpanOutsideWindow``, against 26.6% at 4096. A tool whose
    corpus is 91% refused is measuring its truncation, not its model."""
    assert tool.ByteDeciderConfig().max_context_bytes < tool.DEFAULT_CONTEXT_BYTES
    assert tool.DEFAULT_CONTEXT_BYTES <= tool.MAX_CONTEXT_BYTES


# -- the objective -----------------------------------------------------------------------


def test_the_tool_exposes_the_steps_span_weight() -> None:
    """The bug this closes, which cost four experiments.

    ``Rung0Step`` has always taken a ``span_weight`` and this tool never passed one, so
    every rung 0 run before 2026-09-20 trained at the library default of 1.0 -- an
    unweighted sum of two channels whose losses open at 12.520 and 2.026. A default that
    is never named in a run's own recipe is not a choice anyone made.
    """
    import inspect

    assert "span_weight" in inspect.signature(tool.train_once).parameters


def test_a_non_positive_span_weight_is_refused_from_argv(tmp_path) -> None:
    """Refused at parse time, not inside the seed loop.

    ``Rung0Step`` does refuse it, but it is constructed after the corpus is generated,
    split and batched; on the GH200 that is minutes of work spent to reach a verdict that
    was decidable from argv. Fail closed, and fail early.
    """
    for bad in ("0", "-0.5"):
        # Matched on the REFUSAL, not on the flag name. "span-weight" alone also matches
        # argparse's "unrecognized arguments: --span-weight", so a tool that had never
        # heard of the flag would have passed this test.
        with pytest.raises(SystemExit, match="span-weight must be positive"):
            tool.main(["--out", str(tmp_path), "--rev", "HEAD", "--span-weight", bad])


def test_the_span_weight_changes_the_recipe_hash() -> None:
    """Two runs that optimised different objectives must not share a protocol.

    Without this the five points of the span-weight sweep hash identically and the ledger
    records one recipe for runs whose train accuracy differs by 29 points -- collapsing
    the exact comparison the sweep exists to make.
    """
    import hashlib
    import json

    def recipe(span_weight: float) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "epochs": 10,
                    "batch_size": 16,
                    "val_share": 0.25,
                    "lr": 3e-3,
                    "span_weight": span_weight,
                    "rev": "HEAD",
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    assert recipe(0.05) != recipe(1.0)
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert '"span_weight": args.span_weight,' in source


# -- fit against generalisation ----------------------------------------------------------


def test_a_constant_predictor_does_not_pass_the_fit_gate() -> None:
    """The measured signature of the collapse. At ``span_weight`` >= 0.5 every seed scored
    exactly the training-set majority share of 48.0% on its OWN training rows, which is
    what a head emitting one class scores. That must not read as a model that fitted."""
    state = tool._fit_gate(0.480, 0.480, n=763)
    assert isinstance(state, Ran)
    assert not state.passed
    assert "constant predictor" in state.detail


def test_the_fit_gate_passes_only_above_the_training_majority() -> None:
    """At ``span_weight`` 0.05 the same corpus reached 73.4% on training rows. The gate has
    to separate that from 48.0%, or the two regimes are one number in the ledger."""
    state = tool._fit_gate(0.734, 0.480, n=763)
    assert isinstance(state, Ran)
    assert state.passed
    assert "+25.4%" in state.detail


def test_an_unmeasured_training_set_is_not_a_failed_one() -> None:
    """``NotRun``, not ``passed=False``. Zero rows means the question was never asked, and
    recording that as a failed fit is how a check that could not run comes to look like a
    check that ran."""
    state = tool._fit_gate(0.0, 0.5, n=0)
    assert isinstance(state, NotRun)
    assert not hasattr(state, "passed")


def test_the_fit_gate_is_not_the_generalisation_gate() -> None:
    """Same arithmetic, opposite verdicts, so the prose must differ. ``_accuracy_gate`` at
    the baseline means "fitted something that did not transfer -- suspect the corpus";
    ``_fit_gate`` at the majority share means "fitted nothing -- the held-out number is not
    evidence at all". A single helper would report the second as the first."""
    generalisation = tool._accuracy_gate(
        0.521, 0.521, n=303, what="choice", baseline_name="majority-class baseline"
    )
    fit = tool._fit_gate(0.480, 0.480, n=763)
    assert isinstance(generalisation, Ran) and isinstance(fit, Ran)
    assert "held-out rows" in generalisation.detail
    assert "TRAINED on" in fit.detail
    assert "says nothing about generalisation" in fit.detail


def test_a_zero_choice_loss_does_not_crash_the_failure_summary() -> None:
    """The ratio is printed only when a run has already failed. A ZeroDivisionError raised
    while explaining the failure would replace the explanation with a traceback."""
    assert "no ratio" in tool._channel_ratio({"span_first": 12.52, "choice_first": 0.0})
    assert "6.2:1" in tool._channel_ratio({"span_first": 12.520, "choice_first": 2.026})


# --- determinism ----------------------------------------------------------------------
#
# MEASURED on a GH200 2026-09-21, two runs at seed 0 with every other input identical:
# choice val 50.3% and 53.1%, final choice loss 0.680 and 0.762, span end top-1 1.0% and
# 1.7%. 2.8 percentage points apart at a FIXED seed, against a corpus whose entire
# demonstrated signal is 5.3 points. Every per-seed rung 0 number predating this flag is a
# draw from that spread and no row says so.


def test_the_cublas_workspace_is_configured_before_torch_is_imported() -> None:
    """cuBLAS reads this when it initialises, which is the first matmul, and argparse has
    not run by then. Configured after the import it is decoration -- a setting that looks
    applied and is not."""
    source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
    assert 'if "--deterministic" in sys.argv:' in source
    assert 'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")' in source

    argv_at = source.index('if "--deterministic" in sys.argv:')
    torch_at = source.index("import torch  # noqa: E402")
    assert argv_at < torch_at, (
        "the workspace is configured after torch is imported, so a CUDA context may "
        "already exist by then and the setting does nothing"
    )


def test_an_ordinary_rung0_run_does_not_get_the_workspace_set_for_it() -> None:
    """A 32 MB cuBLAS workspace on a run that never asked for determinism is a cost paid
    for nothing, and a global that changes under runs that did not mention it is how two
    runs come to differ for a reason neither recorded."""
    import os
    import subprocess
    import sys as _sys

    tools = str(REPO / "tools")
    probe = (
        "import os, sys;"
        "sys.argv=['rung0_real_run.py','--out','/nowhere'];"
        f"sys.path.insert(0, {tools!r});"
        "import rung0_real_run;"
        "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'UNSET'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    out = subprocess.run(
        [_sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "UNSET", out.stdout


def test_the_flag_exists_and_defaults_to_off() -> None:
    """Off by default, because it costs wall clock and an ordinary probe should not pay it.
    Present, because the alternative is every rung 0 number being a draw."""
    import rung0_real_run

    parser = rung0_real_run._parser() if hasattr(rung0_real_run, "_parser") else None
    if parser is None:
        # The parser is built inside main(); assert on the source instead of restructuring
        # another lane's tool to make it importable.
        source = (REPO / "tools" / "rung0_real_run.py").read_text(encoding="utf-8")
        assert '"--deterministic",' in source
        assert 'action="store_true"' in source[source.index('"--deterministic",'):][:400]


def test_the_recipe_separates_a_deterministic_rung0_run_from_an_ordinary_one() -> None:
    """Two runs that used different kernels for the same matmul are not one protocol.

    The measured spread between them, 2.8 points, is larger than most of the effects this
    tool is used to look for, so hashing them alike would put two populations in one
    ``recipe_hash`` and make them comparable rows in the ledger.

    Asserted on the HASH, not on the source. It used to read the text between
    ``recipe_hash=hashlib.sha256(`` and ``).hexdigest(),``, which stopped existing when the
    recipe was given a name so the row could carry it -- a refactor that changed nothing
    about what is hashed and broke the test anyway. Reading the recipe off a row and
    re-hashing it asserts the same thing and survives the next such move.

    ``recipe`` in the row is exactly the object ``recipe_hash`` was taken of, which is
    checked here rather than assumed: a row whose stored recipe does not reproduce its own
    hash is worse than a row with no recipe at all.
    """
    recipe = _recipe_of_a_row()
    assert "deterministic" in recipe, (
        "the recipe hash does not cover --deterministic, so a deterministic run and an "
        "ordinary one hash to the same protocol"
    )
    # And the two levers this tool's sweeps vary are in there with it.
    assert "span_weight" in recipe
    assert "epochs" in recipe


def test_the_recipe_separates_a_subsampled_training_set_from_a_whole_one() -> None:
    """``--train-subsample`` changes how much data was trained on and nothing else.

    It is a learning curve: the point of it is that several runs differ ONLY in training-set
    size, and are compared. That is the same shape as ``--span-weight`` and
    ``--deterministic``, both of which are in the recipe with a comment saying why, and it
    is the shape the ledger cannot see through -- ``data_snapshot_hash`` is derived from the
    manifest, which a subsampled run does not change, so a 50% run and a whole one agree on
    every protocol field there is.

    Two runs whose entire experimental difference is invisible to ``recipe_hash`` are one
    protocol measured twice, and pooling a half-data arm with a full-data one is exactly the
    comparison the flag exists to make.

    Asserted off a row and its hash, like its neighbour: present in the recipe but absent
    from the hash is the failure, not absence altogether, and re-hashing the stored recipe
    is what rules that out.
    """
    recipe = _recipe_of_a_row()
    assert "train_subsample" in recipe, (
        "the recipe hash does not cover --train-subsample, so a run on half the training "
        "files and a run on all of them hash to the same protocol"
    )


def test_a_row_can_name_its_own_arm_without_the_launch_command() -> None:
    """GAP-A-ROW-CANNOT-SAY-WHICH-CURVE-POINT-IT-IS, raised by the concurrent lane.

    Every field this tool varies went into ``recipe_hash``'s input and was stored nowhere,
    so a row could say two arms are not comparable and not say how they differ. Labelling
    the three points of a learning curve on 2026-09-21 took re-hashing four candidate
    ``train_subsample`` values against seven fields pinned at the launch command's -- which
    is recovering the label from the log by a longer route, not from the row.

    The eight keys are asserted as a SET, so a field silently dropped from the recipe fails
    here even though it would leave every other test green: it would still hash, still
    identify, and quietly stop separating the arm it was added to separate.
    """
    assert set(_recipe_of_a_row()) == {
        "epochs", "batch_size", "val_share", "lr",
        "span_weight", "deterministic", "train_subsample", "rev",
    }


# -- the price of the machine ------------------------------------------------------------
#
# GAP-EVERY-RUN-PRICED-ITSELF-AT-ZERO-ON-ZERO-GPUS, the half that was left. This tool kept
# `usd_per_hour=0.0, n_gpus=0, instance=f"local-{device}"` under a comment reading "A Mac
# that is already bought costs nothing per hour" -- while `--device` is a free string and
# every GH200 run it made passed `--device cuda` straight through it.


def test_train_once_takes_the_price_of_the_machine_it_runs_on() -> None:
    """The same shape as ``span_weight`` above, and the same defect underneath: a quantity
    that decides what a run means, chosen by something other than this tool, recorded
    nowhere."""
    import inspect

    parameters = inspect.signature(tool.train_once).parameters
    for name in ("instance", "usd_per_hour", "usd_per_gpu_hour", "approved_by"):
        assert name in parameters, f"train_once cannot state {name}, so it cannot record it"


def test_a_cuda_run_without_a_price_is_refused_before_the_corpus_is_built(tmp_path) -> None:
    """Refused from argv, ahead of every other argv check, so nothing can mask it.

    ``--epochs 0`` is the instrument rather than the subject. It is refused a few lines
    further down, so a tool that reaches THAT refusal first is a tool that had not yet
    decided anything about the price -- which is what the pre-fix code does, in
    milliseconds, instead of spending minutes generating a corpus on a rented box before
    reaching a verdict that was decidable from argv.
    """
    argv = ["--out", str(tmp_path), "--rev", "HEAD", "--device", "cuda", "--epochs", "0"]
    # Matched on the REFUSAL, never on the flag name: "--instance" alone also matches
    # argparse's "unrecognized arguments: --instance", so a tool that had never heard of
    # the flag would pass. test_a_non_positive_span_weight_is_refused_from_argv found that
    # trap first; this is the same trap.
    with pytest.raises(SystemExit, match="needs --instance and --usd-per-hour"):
        tool.main(argv)
    with pytest.raises(SystemExit, match="needs --instance and --usd-per-hour"):
        tool.main([*argv, "--instance", "lambda-1xGH200"])


def test_the_refusal_says_what_the_two_defaults_actually_did(tmp_path) -> None:
    """"You forgot an argument" gets the argument added and the number guessed.

    The consequence has to be in the message, and here it is not merely an under-reported
    cost: at ``n_gpus=0, usd_per_hour=0.0`` both disjuncts of ``requires_human_approval``
    are False for any cap, and the row asserted ``instance="local-cuda"`` -- a local
    machine with no GPUs in it, for every run this tool made on a rented GH200.
    """
    with pytest.raises(SystemExit) as excinfo:
        tool.main([
            "--out", str(tmp_path), "--rev", "HEAD", "--device", "cuda", "--epochs", "0",
        ])
    message = str(excinfo.value)
    assert "requires_human_approval" in message and "ANY cap" in message
    assert "local-cuda" in message, (
        "the message describes an under-priced run but not the machine that did not exist"
    )
    assert "--usd-per-hour 1.49" in message, (
        "an operator on a rented box should be able to copy a working invocation out of "
        "the refusal rather than go and read the source"
    )


def test_a_local_run_needs_no_price_and_is_not_refused(tmp_path) -> None:
    """The control. A refusal that fired on mps too would be found within the hour and
    routed around by whoever hit it on a Mac -- and an already-bought machine is the case
    the zero is honest for. ``--epochs 0`` is the marker again: reaching its refusal is
    what proves the price check let an mps run past."""
    with pytest.raises(SystemExit, match=r"--epochs must be in"):
        tool.main([
            "--out", str(tmp_path), "--rev", "HEAD", "--device", "mps", "--epochs", "0",
        ])


def test_the_toy_runner_beside_it_refuses_the_same_case() -> None:
    """``rung0_toy_run`` carried the identical literal, under a docstring promising that "a
    run on a rented machine sets a real rate here". Nothing made that true: ``device`` is a
    parameter, and the literal priced whatever arrived as a Mac."""
    import rung0_toy_run

    local = rung0_toy_run._control(100, device="mps")
    assert local.cost.usd_per_hour == 0.0
    assert local.cost.n_gpus == 0
    assert local.cost.instance == "local-mps"

    with pytest.raises(ValueError, match="hardware being paid for by the hour"):
        rung0_toy_run._control(100, device="cuda")
