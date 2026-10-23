"""Append one gap line per retired set (and one open record for the two held), via append_gap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "python"))
from qd_train.gaps import append_gap  # noqa: E402

verdicts = json.loads((REPO / "AUDIT/x4-v1-headers-2026-10-22/verdicts.json").read_text())
SRC = "AUDIT/x4-v1-headers-2026-10-22/readers.md"
RULING = "Fable 2026-10-22 via lead (owner delegation 'spin agent and fix them')"
parent = ["GAP-W4C-V1-HEADERS-STILL-REFUSED-UNDER-V1-RULE-2026-10-18"]
n = 0
for i, v in enumerate(sorted(verdicts, key=lambda v: v["rel"]), start=1):
    if v["verdict"] == "hold":
        continue
    n += 1
    append_gap(
        {
            "id": f"GAP-X4-RETIRED-V1-SHARD-SET-{n:02d}-2026-10-22",
            "opened": "2026-10-22",
            "lane": "x4-v1-headers",
            "owner": "lead",
            "status": "closed",
            "tool": "AUDIT/x4-v1-headers-2026-10-22/readers.py (text scan, not DevMap)",
            "question": f"retired: no reader. Set ~/qd-campaign/{v['rel']}, shard_hash {v['shard_hash']}, v1 header refused at HEAD. Stays refused, stays on disk.",
            "answer": f"[V] {v['detail']}. Evidence: {SRC}. Ruling: {RULING}.",
        }
    )
append_gap(
    {
        "id": "GAP-X4-V1-V5R-TRAIN-VAL-HELD-FOR-LEAD-2026-10-22",
        "opened": "2026-10-22",
        "lane": "x4-v1-headers",
        "owner": "lead",
        "status": "open",
        "tool": "AUDIT/x4-v1-headers-2026-10-22/readers.py (text scan, not DevMap)",
        "question": "Do ~/qd-campaign/phase4-v5r-2026-10-03/shards/train (shard_hash 108082545a66...) and .../shards/val (a485e7f7056d...) get retired, or does a quick-only reader justify a v5-legacy-rule rebuild?",
        "answer": "[V] Their only non-historical, non-withdrawn readers are two pre-registrations that cannot promote anything (rule 8): campaign/next-train-first-box-preregistered.json (P1 data_fact, NOT LAUNCHED) and campaign/span-refit-mac-preregistered.json (quick, report-only; amendment 2 of 2026-10-22 not yet run). campaign/lora-2b-preregistered.json withdrew both sets (amendment 3). A rebuild moves shard_hash, which both pre-registrations pin: re-pinning is an amendment. Not rebuilt, not retired here. The v5r train set also holds MMLU test/dev and CLINC test items (the lora-2b amendment 3 reason), and the v5-rule rebuild would reproduce them. Both pre-registrations already have a non-flag way to read the sets: the pinned writing-code worktree (next-train line 43).",
    }
)
print(n, "retirements + 1 hold appended")
