"""A cost and a clock reach a ledger row from a runner that was actually run.

Every other cost test in this suite stops one step short of the thing being claimed.
``test_row_cost.py`` and ``test_wall_clock.py`` build a ``RunRecorder`` directly and
assert it writes what it is handed; ``test_tool_call_sites.py`` reads the tools' *source
text* for a ``cost=`` keyword. All three are in-process, and none of them runs a tool.

The distance between "the recorder writes what it is handed" and "the tool hands it the
right thing" is exactly one defect wide, and the defect went through. ``164eb72`` carried
the estimate to the recorder inside each runner's run dict; 1,758 tests passed over it;
the first end-to-end invocation of ``tools/rung0_toy_run.py`` afterwards died with

    TypeError: Object of type CostEstimate is not JSON serializable
    when serializing dict item 'cost' ... when serializing dict item 'runs'

because that dict is also the report the tool prints with ``json.dumps``. ``ea6522a``
fixed it by keeping the run dict JSON-clean and rebuilding the estimate from a factory at
both call sites. Nothing pinned that, so nothing stopped the next author putting an object
back in the dict -- or, in ``tools/rung0_real_run.py``, where the run dict is *not*
serialised today, nothing would have caught it until the first ``json.dumps`` landed in a
sweep that costs GPU hours.

These tests run the runners. Two of them are a CPU-only smoke path deliberately sized in
single-digit seconds; the third prices a run at a real rate, because a cost assertion on a
local device is ``0.0 == 0.0`` and would pass against a runner that recorded nothing at
all.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import rung0_real_run as real_run  # noqa: E402

from qd_train.ledger import Ledger  # noqa: E402
from qd_train.mutate_adapter import parse_example, to_decision  # noqa: E402

#: Rule 4's single-GPU threshold is ``projected_usd >= 20``, and ``projected_usd`` is the
#: rate times ``RUN_CAP_S`` -- one hour. A rate at or above $20/h would make this run need
#: a human yes, which is the rule working, not a reason to move it: the test states a rate
#: below the threshold instead. $18/h is also a rate no local device has, so a row carrying
#: it cannot have come from the zero-priced local branch by accident.
TEST_RATE_USD_PER_HOUR = 18.0

_CLASSES = ("stub", "logic", "cosmetic", "clean")


def _corpus(tmp_path: Path, *, n_files: int = 8, per_file: int = 3) -> tuple[Path, Path]:
    """A qd-mutate corpus small enough to train on in seconds, valid enough to survive.

    Eight files so ``--val-share 0.25`` has something to put on both sides of a split that
    is by file; a body short enough that no span is truncated out of the window, since a
    corpus that refused every example would reach the gates with nothing measured and this
    test would then be asserting about an empty run.

    ``clean`` carries no span, and that is not a detail of this fixture: ``to_decision``
    refuses a clean example that has one -- *"a clean example points at nothing, but
    carries a span"*. The first draft here gave every row a span and quietly lost a quarter
    of the corpus to ``MalformedExample``, which the tool reported and the test did not
    read. A fixture that is 25% invalid still trains; it just is not the corpus its author
    thinks it is.
    """
    body = "".join(f"line {i} of the function body\n" for i in range(12))
    rows = []
    for f in range(n_files):
        for k in range(per_file):
            mutation_class = _CLASSES[(f + k) % len(_CLASSES)]
            rows.append({
                "id": f"f{f}-{k}",
                "function": {
                    "repo": "qwen-decision",
                    "path": f"src/module_{f}.py",
                    "symbol": f"fn_{k}",
                    "arity": 1,
                },
                "language": "python",
                "class": mutation_class,
                "after": body,
                "span": None if mutation_class == "clean" else {"start_line": 2, "end_line": 3},
                "silent": False,
                "hunk_constrained": False,
                "seed": f * 10 + k,
            })
    # Self-checked rather than trusted. The tool counts refusals by kind and trains on what
    # survives, so a fixture that drifts out of the schema degrades silently into a smaller
    # corpus instead of failing -- which is how the clean/span rule went unnoticed above.
    for row in rows:
        to_decision(parse_example(row), max_context_bytes=512)

    examples = tmp_path / "examples.jsonl"
    examples.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"corpus_sha256": "0" * 64, "n_examples": len(rows)}), encoding="utf-8"
    )
    return examples, manifest


def _toy_run(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Ledger]:
    """``tools/rung0_toy_run.py`` at its smallest, against a ledger of its own."""
    ledger_path = tmp_path / "toy-ledger.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "rung0_toy_run.py"),
            "--steps", "2",
            "--rows", "4",
            "--seeds", "0",
            "--ledger", str(ledger_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return proc, Ledger(ledger_path)


def test_the_toy_runner_finishes_and_its_report_is_json(tmp_path: Path) -> None:
    """The defect ``ea6522a`` fixed, pinned by running the thing.

    Against the pre-fix tool this raises ``TypeError: Object of type CostEstimate is not
    JSON serializable`` and the process exits non-zero after having written its rows --
    so the failure is at report time, on a run that already spent whatever it spent.
    """
    proc, _ = _toy_run(tmp_path)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-4000:]}"
    match = re.search(r"^\{$", proc.stdout, re.MULTILINE)
    assert match is not None, f"no JSON report in stdout:\n{proc.stdout[-2000:]}"
    report = json.loads(proc.stdout[match.start() :])
    assert report["runs"], "the report carries no runs"
    for run in report["runs"]:
        assert "cost" not in run, (
            "the run dict is the printed report; an estimate object in it is the defect "
            "this test exists for, and it is only a TypeError because json.dumps is the "
            "next statement"
        )


def test_the_toy_runners_rows_carry_the_runners_own_clock(tmp_path: Path) -> None:
    """``wall_clock_source`` distinguishes a measured run from a timed recorder.

    Before ``a409895`` the recorder timed its own ``__enter__``-to-``__exit__`` lifetime,
    which for every runner in ``tools/`` is the time to write metrics -- microseconds
    against a run of minutes. The field is what makes the two distinguishable in a row
    rather than only in the log, and this is the only test that reads it off a row a tool
    wrote.
    """
    proc, ledger = _toy_run(tmp_path)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-4000:]}"
    rows = ledger.rows()
    assert rows, "the tool wrote no ledger rows"
    for row in rows:
        assert row.wall_clock_source == "caller", (
            f"row {row.row_id} says its duration came from {row.wall_clock_source!r}; the "
            "tool measures the training loop and hands the figure over"
        )
        assert row.wall_clock_s > 0.0, f"row {row.row_id} reports a zero-length run"


def test_a_priced_run_records_the_rate_against_its_own_clock(tmp_path: Path) -> None:
    """The end-to-end claim, on a run whose cost is not zero.

    ``tools/rung0_real_run.py`` builds the estimate twice -- once for the ``RunControl``
    that gates the run, once for the row that records it -- from one factory, so the two
    cannot disagree. This drives ``main()`` on a corpus of eight synthetic files and then
    asserts the row's own two fields against each other: ``cost_usd`` is the stated rate
    applied to ``wall_clock_s``, and neither is zero.

    ``--device cpu`` with an explicit rate is the only way to make that assertion bite
    without a rented GPU: ``CostEstimate.for_device`` prices a local device at zero when
    no rate is given, and ``0.0 == 0.0`` would also pass for a row that recorded nothing.
    """
    examples, manifest = _corpus(tmp_path)
    ledger_path = tmp_path / "real-ledger.jsonl"
    out = tmp_path / "scratch"
    out.mkdir()
    argv = [
        "--out", str(out),
        "--rev", "HEAD",
        "--examples", str(examples),
        "--manifest-in", str(manifest),
        "--device", "cpu",
        "--instance", "rung0-cost-end-to-end",
        "--usd-per-hour", str(TEST_RATE_USD_PER_HOUR),
        "--seeds", "1",
        "--epochs", "1",
        "--width", "16",
        "--layers", "1",
        "--heads", "1",
        "--context-bytes", "512",
        "--batch-size", "2",
        "--ledger", str(ledger_path),
    ]
    # The gates decide whether rung 0 learned anything; on eight synthetic files it plainly
    # did not, and `main` says so with a non-zero exit. That is not what is under test here
    # -- the row is written before the verdict is reached, and "a failed run still cost what
    # it cost" is the property being checked.
    with contextlib.suppress(SystemExit):
        real_run.main(argv)

    rows = Ledger(ledger_path).rows()
    assert rows, "the tool wrote no ledger row"
    row = rows[0]
    assert row.wall_clock_source == "caller"
    assert row.wall_clock_s > 0.0
    assert row.cost_usd > 0.0, (
        "a run priced at $18/h that cost nothing is the zero this test exists to rule out"
    )
    assert row.cost_usd == pytest.approx(
        TEST_RATE_USD_PER_HOUR * row.wall_clock_s / 3600.0, rel=1e-9
    ), "the row's cost is not the stated rate applied to the row's own duration"
