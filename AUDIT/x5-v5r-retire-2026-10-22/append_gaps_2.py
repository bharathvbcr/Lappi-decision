"""Second batch of lane X5's gap lines (after the lead's adversarial pass), via append_gap.

Same ids as the first batch where a record changes state: the ledger reads the last line per id.
Usage: python append_gaps_2.py <repo root>.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "python"))
from qd_train.gaps import append_gap  # noqa: E402

DIR = "AUDIT/x5-v5r-retire-2026-10-22"
COMMON = {"opened": "2026-10-22", "lane": "x5-v5r-retire", "owner": "lead"}
TOOL = "rg and read-only file reads (no DevMap question fits; DevMap was not used)"

RECORDS = [
    {
        "id": "GAP-X5-V5R-TRAIN-NO-DECODE-NOT-RULED-2026-10-22",
        **COMMON, "status": "closed", "tool": f"{DIR}/retirement.json",
        "question": "Is phase4-v5r-2026-10-03/shards/train-no-decode (shard_hash e8cffbdd9574...) "
                    "retired with its sibling pair?",
        "answer": "[V] Yes, by Fable's ruling (a) of 2026-10-22: same reason. Full hash "
                  "e8cffbdd95747bf5e1dfe4e5505d5aa7b34fb81b229ce65d7b238373d06927f9 and its "
                  "data_snapshot_hash read by parsing its header.json with ShardHeader.from_json "
                  f"only. In the registry {DIR}/retirement.json, Python and Rust.",
    },
    {
        "id": "GAP-X5-RETIREMENT-GUARD-COVERS-HEADER-PARSERS-ONLY-2026-10-22",
        **COMMON, "status": "resolved-with-residual", "tool": TOOL,
        "question": "Which readers can still open the retired v5r sets?",
        "answer": "[V] Both header readers refuse them: Python ShardHeader.from_json and Rust "
                  "ShardHeader::from_json (crates/qd-train/src/shards.rs, shared by "
                  "qd-train-metal), each by shard_hash and by data_snapshot_hash (the header "
                  "records no token-payload digest), each constant tested equal to one registry "
                  f"({DIR}/retirement.json). Residual [U]: a tool that json.loads header.json or "
                  "np.loads tokens.u32 without either parser; a set rebuilt from a different "
                  "snapshot of the same sources (the build-time refusal in "
                  "tools/real_tokenizer_pipeline.py is outside this lane's files). The lead's "
                  "test_break_a_rehashed_copy_of_the_retired_tokens_opens registers only the "
                  "retired shard_hash and expects a sibling with another corpus_rev to be "
                  "refused; no key in that registry can identify it, so it fails as written "
                  "(the data_snapshot_hash variant passes).",
    },
    {
        "id": "GAP-X5-NEXT-TRAIN-QUEUE-STILL-BUILDS-V5-DATA-2026-10-22",
        **COMMON, "status": "resolved-with-residual", "tool": TOOL,
        "question": "Does campaign/next-train-first-box-queue-2026-10-08.json still rebuild the "
                    "v5 data?",
        "answer": "[V] Repointed 2026-10-22 (Fable: repoint now): {v6_data_dir}, the v6 scan's "
                  "exclusion list pinned (37b84644...), the v5 pre-dedupe drop list neither passed "
                  "nor pinned, no v5 shard_hash comparison. [U] Held: the repoint is from the "
                  "text of LoRA-2B entry n = 5, not from a run that reproduced the v6 shard_hash "
                  "from these argv; the placeholder v6_shards_verified is null, which the driver "
                  "refuses, and the file carries a 'hold' object. A decision remains: confirm "
                  "{v6_data_dir} and {decisions_pool} (pool-v6-decisions-v1) on the box.",
    },
    {
        "id": "GAP-X5-PERF-STALE-NUMBERS-CITED-AS-PINNED-CODE-PERFORMANCE-2026-10-22",
        **COMMON, "status": "resolved-with-residual", "tool": TOOL,
        "question": "Which reports cite perf_step numbers measured with --allow-stale-shards as "
                    "the performance of pinned code?",
        "answer": "[V] Five reports, each now ending in a dated 'Correction (2026-10-22)' "
                  "(appended; existing text untouched): HANDOFF/train-step-perf-2026-10-01.md, "
                  "AUDIT/tierb-nomask-flash-screen-design-2026-10-01.md, "
                  "AUDIT/det-attention-backward-scoping-2026-10-01.md, "
                  "AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md, "
                  "AUDIT/training-audit-2026-10-06.md:455 (60.51 GiB, via the first). The numbers "
                  "stand as measured, labelled allow_stale_shards, not performance of pinned code. "
                  "18 rows carry allow_stale_shards: true (session1.jsonl 7, session2/bench.jsonl "
                  "9, session2/probe-nomask.jsonl 2). Residual [U]: session2/parity.jsonl "
                  "(perf_parity arms, source of 9,326 pos/s) does not record the flag.",
    },
]

for record in RECORDS:
    append_gap(record, path=REPO / "gaps.jsonl")
print(len(RECORDS), "records appended")
