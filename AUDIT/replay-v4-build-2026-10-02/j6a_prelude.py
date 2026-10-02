"""Throwaway verification (never ships): real_ft_run.main's own prelude on J6(a)'s data argv.

The form of /home/ubuntu/scratch/f_prelude_box.py (F's), stopping later: at the first statement
after ``_replay_plan`` instead of at ``_resolve_batch_tokens``. At a502670 that statement is
``Ledger(args.ledger)`` (tools/real_ft_run.py:8512), so ``Ledger`` is patched to stop, and
``_replay_plan`` is wrapped so a stop that comes before it is refused rather than read as OK.
No tower loads, nothing trains, nothing is written.

Usage: python j6a_prelude.py LANE_ROOT ARGS...   (cwd becomes LANE_ROOT)
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import real_ft_run as ft  # noqa: E402


class PreludeDone(Exception):
    pass


seen: dict[str, object] = {}
_plan = ft._replay_plan


def _replay_plan(*args, **kwargs):
    plan = _plan(*args, **kwargs)
    seen["batches"] = len(plan.batches)
    seen["rows"] = sum(int(b.tokens.shape[0]) for b in plan.batches)
    seen["shard_hash"] = plan.shard_hash
    seen["attestation_sha256"] = plan.attestation_sha256
    seen["weight"], seen["every"] = plan.weight, plan.every
    seen["letter_ids"] = list(plan.letter_ids)
    return plan


def _stop(*_a, **_k):
    raise PreludeDone("reached Ledger(...), the first statement after _replay_plan")


ft._replay_plan = _replay_plan
ft.Ledger = _stop

t0 = time.monotonic()
try:
    ft.main(sys.argv[2:])
except PreludeDone as exc:
    if "batches" not in seen:
        raise SystemExit(f"stopped before _replay_plan ran: {exc}") from exc
    print(f"PRELUDE OK in {time.monotonic() - t0:.0f} s: {exc}; replay plan {seen}")
else:
    raise SystemExit("main returned without reaching Ledger(...)")
