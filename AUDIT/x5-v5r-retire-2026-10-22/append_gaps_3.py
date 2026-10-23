"""Third batch of lane X5's gap lines (the lead's pre-merge review).

Usage: python append_gaps_3.py <repo root>.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "python"))
from qd_train.gaps import append_gap  # noqa: E402

COMMON = {"opened": "2026-10-22", "lane": "x5-v5r-retire", "owner": "lead"}
TOOL = "rg and read-only file reads (no DevMap question fits; DevMap was not used)"

RECORDS = [
    {
        "id": "GAP-X5-REBUILD-FROM-A-DIFFERENT-SNAPSHOT-NOT-CAUGHT-2026-10-22",
        **COMMON, "status": "open", "tool": TOOL,
        "question": "Can a set rebuilt from the v5r source pools under the v5 rule be opened?",
        "answer": "[V] Not if its header carries a retired shard_hash or data_snapshot_hash "
                  "(Python and Rust refuse both). [U] A rebuild from a DIFFERENT snapshot of the "
                  "same sources has neither key and is not caught; headers record no token-payload "
                  "digest. The build-time refusal (tools/real_tokenizer_pipeline.py refusing "
                  "--rev 8e6a00952a09e31a09cc3f5270229a935b7235cc under the v5 rule) is out of "
                  "scope for lane X5 and not added.",
    },
    {
        "id": "GAP-X5-PERF-STALE-NUMBERS-CITED-AS-PINNED-CODE-PERFORMANCE-2026-10-22",
        **COMMON, "status": "resolved-with-residual", "tool": TOOL,
        "question": "Which reports cite perf_step numbers measured with --allow-stale-shards as "
                    "the performance of pinned code?",
        "answer": "[V] Five reports end in a dated 'Correction (2026-10-22)', appended, existing "
                  "text untouched: HANDOFF/train-step-perf-2026-10-01.md, "
                  "AUDIT/tierb-nomask-flash-screen-design-2026-10-01.md, "
                  "AUDIT/det-attention-backward-scoping-2026-10-01.md, "
                  "AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md, and "
                  "AUDIT/training-audit-2026-10-06.md:455, whose 60.51 GiB (skip:6) is "
                  "AUDIT/perf-train-step-2026-10-01/session2/bench.jsonl lines 2, 4 and 6 "
                  "(config skip:6, peak_alloc_gib 60.51, allow_stale_shards true), checked "
                  "2026-10-22. The numbers stand as measured, labelled allow_stale_shards, not "
                  "performance of pinned code. 18 rows carry allow_stale_shards: true "
                  "(session1.jsonl 7, bench.jsonl 9, probe-nomask.jsonl 2). Residual [U]: "
                  "session2/parity.jsonl (source of 9,326 pos/s) does not record the flag.",
    },
]

for record in RECORDS:
    append_gap(record, path=REPO / "gaps.jsonl")
print(len(RECORDS), "records appended")
