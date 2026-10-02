"""J6(a)'s prelude, run by box_q_j6a.sh before J6(a) trains (HANDOFF/replay-v4-plan-2026-10-02.md,
"Prelude, first on the Mac, then on the box").

It runs real_ft_run.main's own prelude from qd-lane8 at a502670 on J6(a)'s training argv and
stops at the first statement after `_replay_plan`, which at a502670 is
`ledger = Ledger(args.ledger)` (tools/real_ft_run.py:8512, after `_replay_plan` at :8498). So
it stops before the ledger is opened and before any tower loads. Up to that point main has
read the train set's header, paired the labels, opened the replay shard set, held the
attestation to it (including the val digest `_replay_plan` recomputes) and built the replay
batches. It has also printed its own line:
`replay: N batches from <dir> (attestation <sha16>), weight <w>, every <k>`.
The waiter requires that line as well as this script's "PRELUDE OK".

This is the form of /home/ubuntu/scratch/f_prelude_box.py (sha256
9f86f797e607b2b9a48714d182a6ef15108bee059a03694b8a8494696037a3dc). That prelude patched
`_resolve_batch_tokens` instead. Every other `Ledger(...)` call in main (:8350, :8369, :8403)
sits in a --score-plan or --score-checkpoint branch that returns before `_replay_plan`, and
J6(a)'s training argv takes neither.

Run with cwd=/home/ubuntu/qd-lane8 and argv = J6(a)'s training argv. It writes nothing:
- no ledger row (it stops at the Ledger constructor);
- no replay cache (`_replay_plan` only records `cache_path`);
- no checkpoint and no verdicts.

Exit codes:
- 0 with "PRELUDE OK" only when main reached Ledger(...);
- non-zero when main refused, raised, or returned without reaching it.
"""

import sys
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/qd-lane8")
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import real_ft_run as ft  # noqa: E402


class PreludeDone(Exception):
    pass


def _stop(path: object, *args: object, **kwargs: object) -> None:
    raise PreludeDone(f"prelude complete; stopped at Ledger({path}) after _replay_plan")


ft.Ledger = _stop  # type: ignore[assignment,misc]

t0 = time.monotonic()
try:
    rc = ft.main(sys.argv[1:])
except PreludeDone as exc:
    print(f"PRELUDE OK in {time.monotonic() - t0:.0f} s: {exc}", flush=True)
else:
    raise SystemExit(f"PRELUDE FAILED: main returned {rc!r} without reaching Ledger(args.ledger)")
