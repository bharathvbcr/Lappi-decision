"""Append lane X5's gap lines through qd_train.gaps.append_gap (O_APPEND + fsync, one line each).

Usage: python append_gaps.py <repo root>. Run once; a second run appends the same ids again,
which the ledger's last-line-wins rule reads as the same records.
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
        "id": "GAP-X4-V1-V5R-TRAIN-VAL-HELD-FOR-LEAD-2026-10-22",
        "opened": "2026-10-22", "lane": "x5-v5r-retire", "owner": "lead", "status": "closed",
        "tool": f"{DIR}/retirement.json",
        "question": "Do ~/qd-campaign/phase4-v5r-2026-10-03/shards/train (shard_hash "
                             "108082545a66...) "
                    "and .../shards/val (a485e7f7056d...) get retired, or does a quick-only reader "
                    "justify a v5-legacy-rule rebuild?",
        "answer": "[V] retired: contaminated (LoRA-2B amendment 3). Fable 2026-10-22, relayed "
                           "by the "
                  "lead: no rebuild, no new read, files stay on disk untouched (ledger rows cite "
                  f"the hashes). Record: {DIR}/retirement.json. Guard: "
                  "qd_train.artifacts.RETIRED_SHARD_SETS, checked in ShardHeader.from_json, test "
                  "python/tests/test_retired_shard_sets.py. The two readers are amended to the v6 "
                  "shards (next-train-first-box promoted.amendment_2; span-refit-mac amendment 3).",
    },
    {
        "id": "GAP-X5-SPAN-REFIT-V6-POPULATION-GATE-AND-OVERLAP-PENDING-2026-10-22",
        **COMMON, "status": "open", "tool": TOOL,
        "question": "What does span-refit measure on the v6 val, and against which bars?",
        "answer": "[V] The v6 val carries the pointing rows the scorer needs: 7,243 span "
                           "rows, 5,359 "
                  f"pointing, 1,884 abstain ({DIR}/v6-val-span-rows.json). Amendment 3 of "
                  "campaign/span-refit-mac-preregistered.json moves the inputs and keeps the "
                  "Amendment 2 scorer sha, but the population pin, cell counts, 5,596 +/- 15 gate "
                  "(row 6af73bef scored the v5r val), 6,562 and 5,667 are v5r-val numbers that "
                  "v6 does not carry; the scorer refuses a v6 cache until a further amendment "
                  "fixes "
                  "them and a scorer change (new sha) carries them. [U] Whether v6 val rows were "
                  "in the v5r train set the towers trained on is not measured. Also: the lora-2b "
                  "entry that pins the v6 shards is array entry n = 5, not amendment 4. "
                  "Needs a ruling on the gate and thresholds before any run.",
    },
    {
        "id": "GAP-X5-NEXT-TRAIN-QUEUE-STILL-BUILDS-V5-DATA-2026-10-22",
        **COMMON, "status": "open", "tool": TOOL,
        "question": "Does campaign/next-train-first-box-queue-2026-10-08.json still rebuild "
                             "the v5 data?",
        "answer": "[V] Yes: its _comment says the arms use 'v5's data argv ... verbatim from "
                  "v5_common.sh V5_SPLIT' and compare shard_hash with v5's 108082545a66; that "
                  "rebuilds the retired contaminated set. promoted.amendment_2 of the next-train "
                  "pre-registration moves P1 to the v6 pair and lists repointing this queue (and "
                  "recomputing P1.data_fact on v6) as pending_before_launch. The queue file is "
                  "outside lane X5's files and was not edited.",
    },
    {
        "id": "GAP-X5-V5R-TRAIN-NO-DECODE-NOT-RULED-2026-10-22",
        **COMMON, "status": "open", "tool": f"{DIR}/retirement.json",
        "question": "Is phase4-v5r-2026-10-03/shards/train-no-decode (shard_hash e8cffbdd9574...) "
                    "retired with its sibling pair?",
        "answer": "[I] It comes from the same v5r pipeline run, so it likely carries the same "
                  "benchmark items; the ruling named only train and val, so it is not in "
                  "RETIRED_SHARD_SETS. Needs a ruling; adding it is one entry in artifacts.py and "
                  "one in retirement.json.",
    },
    {
        "id": "GAP-X5-RETIREMENT-GUARD-COVERS-HEADER-PARSERS-ONLY-2026-10-22",
        **COMMON, "status": "open", "tool": TOOL,
        "question": "Which readers can still open the retired v5r pair?",
        "answer": "[V] ShardHeader.from_json refuses both hashes, so ShardReader and every "
                           "tool that "
                  "parses a header through it stop, with or without allow_stale_code. [U] A tool "
                  "that json.loads header.json or np.loads tokens.u32 directly, and the Rust "
                  "qd-train reader (a separate path, refused by its own v1 header rule per "
                  "AUDIT/x4-v1-headers-2026-10-22, not re-run here), are not covered by this "
                  "guard. Neither was changed: outside lane X5's files.",
    },
    {
        "id": "GAP-X5-PERF-STALE-NUMBERS-CITED-AS-PINNED-CODE-PERFORMANCE-2026-10-22",
        **COMMON, "status": "open", "tool": TOOL,
        "question": "Which reports cite perf_step numbers measured with --allow-stale-shards "
                             "as the "
                    "performance of pinned code?",
        "answer": "[V] 18 result rows carry allow_stale_shards: true "
                           "(AUDIT/perf-train-step-2026-10-01/"
                  "session1.jsonl 7, session2/bench.jsonl 9, session2/probe-nomask.jsonl 2) "
                  "and none "
                  "carries stale_code or quick. Cited without that qualifier: "
                  "HANDOFF/train-step-perf-2026-10-01.md:35,44-46; "
                  "AUDIT/tierb-nomask-flash-screen-design-2026-10-01.md:77-82; "
                  "AUDIT/det-attention-backward-scoping-2026-10-01.md:10-15,48; "
                  "AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md:15 (25,703 vs 16,543 "
                  "pos/s, +55%). [U] session2/parity.jsonl (perf_parity arms, source of 9,326 "
                  "pos/s) does not record the flag, and the design note says both P2 shapes read "
                  "v4 with no stale-shard override. History not edited; the lead decides whether "
                  "a note is added.",
    },
]

for record in RECORDS:
    append_gap(record, path=REPO / "gaps.jsonl")
print(len(RECORDS), "records appended")
