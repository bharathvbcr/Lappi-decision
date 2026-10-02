"""Lane L-prep2's Part C gap record, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. It refuses to append a second copy.
Numbers: ledger/mac-linear-control-refit-2026-10-02.jsonl and
AUDIT/prep2-2026-10-02/refit-vs-record.json (compare_refit.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

GAP = "GAP-LINEAR-CONTROL-RERUN-WOULD-DOUBLE-THE-SELECTED-ROW-2026-10-02"

current = [r for r in read_gaps() if r.get("id") == GAP]
if not current:
    sys.exit(f"{GAP} is not in gaps.jsonl")
if any(r.get("lane") == "L-prep2" for r in current):
    sys.exit(f"{GAP} already has an L-prep2 record; refusing a duplicate")

append_gap(
    {
        "id": GAP,
        "opened": "2026-10-02",
        "resolved": "2026-10-02",
        "lane": "L-prep2",
        "owner": "human",
        "status": "resolved-with-residual",
        "tool": (
            "tools/ft_linear_control.py x3 (one tools/mac_heavy.sh call each, strictly "
            "sequential) with qd-prep sha256 9ad514e8..., --write-ledger "
            "ledger/mac-linear-control-refit-2026-10-02.jsonl; "
            "AUDIT/prep2-2026-10-02/compare_refit.py -> refit-vs-record.json"
        ),
        "question": current[-1]["question"],
        "answer": (
            "Re-run for all three seeds on the Mac CPU, report-only, under Fable's post-F "
            "ruling 3(b) and the human's answer 4 (F's gate stays not_run; no gate reads "
            "these rows). Rows: d877d6bc (seed 0, eval f4feac15), 84510263 (seed 1, aeca8d69), "
            "27a263d5 (seed 2, 8c3a774a), all in ledger/mac-linear-control-refit-2026-10-02.jsonl, "
            "max_iter 8,000, control_engine_sha256 9ad514e8 (so none can hash as a row of "
            "record). Each differs from its row of record (c89b89a1, c0e438e6, 3f72112a) in "
            "exactly one task: intent.domain converges in 6,093 / 6,076 / 6,077 iterations, "
            "grad norm 9.52e-5 / 9.44e-5 / 9.62e-5, where the records stopped unconverged at "
            "6,000 (1.118e-2 / 1.462e-2 / 2.039e-2); every other task's convergence and "
            "margin is identical to its record, as the schedule promises. intent.domain's "
            "paired margin is +0.4727 / +0.4307 / +0.4613 and the aggregate choice margin "
            "+0.4542 [+0.4438, +0.4645] / +0.4521 [+0.4414, +0.4626] / +0.4563 [+0.4458, "
            "+0.4668] over 10,985 letter rows (10,000 bootstrap resamples). The rows' own "
            "gate fields read ran/passed; they are report-only by their notes and by the "
            "human's answer. intent.in_scope's per-family margin is below zero on every seed "
            "(-0.0413 / -0.0733 / -0.0367, unchanged from the records). Wall 1,092 / 1,090 / "
            "1,129 s, max RSS 18.6 / 19.4 / 18.5 GB each."
        ),
        "residual": (
            "The Mac's converged fit is not shown to equal what the box's would be (L-prep "
            "saw the pre-fix cycle amplify platform libm differences); the box's qd-prep-m1 "
            "(9ccaf5aa) was not touched and stays frozen until fsucc decides (Fable 3(b))."
        ),
    }
)
print(f"appended {GAP}")
