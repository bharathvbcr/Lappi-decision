"""Write the ledger the Rust mirror's schema oracle reads.

The reader is ``crates/qd-train/tests/ledger_schema.rs``.

Six lines, written by ``qd_train.ledger`` itself, in this order:

  1. started(run A)             -- ``RunRecorder.__enter__``
  2. completed(run A)
  3. started(run B)             -- its process is SIGKILLed
  4. killed(run B)              -- ``reconcile_orphan_started_rows``
  5. a row recording both ``val_without_train`` and ``train_manifest_sha256``, in the content
     the Rust writer can produce (``qd_train::ledger::Row``), so the Rust test rebuilds the
     same line and compares bytes
  6. the same row recording neither (a row as written before the fields existed)

``expected.json`` is what ``Ledger.rows()`` (started hidden) and ``Ledger.rows(include_started=
True)`` return, as (row_id, status) pairs, plus the sha256 of every raw line, so the Rust reader
is compared with Python's reading and chain, not with its own.

  PYTHONPATH=python .venv/bin/python tools/gen_ledger_schema_fixture.py \\
      --out crates/qd-train/tests/fixtures/ledger-schema
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from qd_train import ledger as ledger_module
from qd_train.ledger import (
    REQUIRED_CONTROLS,
    REQUIRED_GATES,
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    RunRecorder,
)
from qd_train.tristate import NotRun, Ran

REPO = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO / "python"

#: The recipe row 5 and 6 hash, and the manifest-file digest row 5 records.
RECIPE = {"tool": "tools/gen_ledger_schema_fixture.py", "kind": "throughput"}
TRAIN_MANIFEST_SHA256 = hashlib.sha256(b"ledger-schema fixture: a train manifest file").hexdigest()
VAL_WITHOUT_TRAIN = {"count": 1, "families": {"synth.general/access_route": 3}}

_CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {python!r})
    from qd_train.ledger import Environment, Ledger, Protocol, RunRecorder
    proto = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="bytes-utf8-256",
                     backbone_commit="test-backbone", recipe_hash="r" * 64, seed=1)
    rec = RunRecorder(Ledger(sys.argv[1]), protocol=proto, run_kind="ft", repo={repo!r},
                      env=Environment.detect(), wall_clock_s=None, cost=None,
                      registration_id="reg-fixture", run_cap_s=600.0)
    with rec:
        print("ready", flush=True)
        time.sleep(600)
    """
)


def _rust_shaped_row(row_id: str, **extra: object) -> LedgerRow:
    """A row with the content ``qd_train::ledger::Row::line`` writes, field for field."""
    recipe_hash = hashlib.sha256(
        json.dumps(RECIPE, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    env = Environment(
        torch="n/a: Rust trainer, no torch in the process",
        transformers_sha="n/a: Rust trainer, no transformers in the process",
        device="metal", host="fixture-host",
        fla_present=NotRun(reason="fla dry run not executed"),
        causal_conv1d_present=NotRun(reason="causal-conv1d dry run not executed"),
    )
    return LedgerRow(
        row_id=row_id, written_at="2026-10-18T12:00:00.123456+00:00", prev_row_hash=None,
        protocol=Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64,
                          backbone_commit="b" * 40, recipe_hash=recipe_hash, seed=0),
        run_kind="throughput", status="completed", quick=True,
        quick_reason="ledger-schema fixture", code_commit="c" * 40, env=env,
        metrics={"fixture.n": Ran(passed=True, value=7)},
        noul_rate=NotRun(reason="noul rate not computed by this run"),
        controls={c: NotRun(reason=f"control {c!r} was never evaluated by this run")
                  for c in REQUIRED_CONTROLS},
        gates={g: NotRun(reason=f"gate {g!r} was never evaluated by this run")
               for g in REQUIRED_GATES},
        wall_clock_s=1.5, wall_clock_source="recorder", cost_usd=0.0,
        notes="ledger-schema fixture row", recipe=RECIPE, **extra,  # type: ignore[arg-type]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ledger.jsonl"
    if path.exists():
        path.unlink()
    led = Ledger(path)

    proto = Protocol(data_snapshot_hash="d" * 64, tokenizer_hash="bytes-utf8-256",
                     backbone_commit="test-backbone", recipe_hash="r" * 64, seed=0)
    with RunRecorder(led, protocol=proto, run_kind="ft", repo=REPO, env=Environment.detect(),
                     wall_clock_s=None, cost=None) as rec:
        rec.metric("accuracy.k4", Ran(passed=True, value=0.83, n=300, n_total=300))

    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _CHILD.format(python=str(PYTHON_DIR), repo=str(REPO)),
         str(path)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        if child.stdout is not None:
            child.stdout.close()
    written = ledger_module.reconcile_orphan_started_rows(led, datetime.now(UTC))
    assert len(written) == 1, written

    led.append(_rust_shaped_row(
        "00000000-0000-4000-8000-0000000000c5",
        val_without_train=VAL_WITHOUT_TRAIN, train_manifest_sha256=TRAIN_MANIFEST_SHA256))
    led.append(_rust_shaped_row("00000000-0000-4000-8000-0000000000c6"))
    led.verify_chain()

    visible = [[r.row_id, r.status] for r in led.rows()]
    everything = [[r.row_id, r.status] for r in led.rows(include_started=True)]
    assert [s for _, s in everything] == [
        "started", "completed", "started", "killed", "completed", "completed"], everything
    doc = {
        "rule": "qd_train.ledger.Ledger.rows(include_started=False|True); chain = sha256 of the "
                "previous raw line (started rows included)",
        "command": "PYTHONPATH=python .venv/bin/python tools/gen_ledger_schema_fixture.py "
                   "--out crates/qd-train/tests/fixtures/ledger-schema",
        "rows": visible,
        "rows_include_started": everything,
        "raw_line_sha256": [hashlib.sha256(b).hexdigest() for b in led.raw_lines()],
        "val_without_train": VAL_WITHOUT_TRAIN,
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
    }
    (args.out / "expected.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
