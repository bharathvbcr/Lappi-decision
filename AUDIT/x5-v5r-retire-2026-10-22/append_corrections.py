"""Append the dated Correction section to the five reports that cite stale-flag perf numbers.

Appends only; the existing text of each report is not touched. Run once from the repo root.
"""

from __future__ import annotations

from pathlib import Path

GAP = "GAP-X5-PERF-STALE-NUMBERS-CITED-AS-PINNED-CODE-PERFORMANCE-2026-10-22"
ROWS = (
    "AUDIT/perf-train-step-2026-10-01/session1.jsonl (7 rows), session2/bench.jsonl (9) and "
    "session2/probe-nomask.jsonl (2)"
)

CITES = {
    "HANDOFF/train-step-perf-2026-10-01.md": "the benchmark and probe tables (lines 35 and 44-46) "
        "and the peak memory figures",
    "AUDIT/tierb-nomask-flash-screen-design-2026-10-01.md": "the table at lines 77-82 "
        "(15,280 / 25,703 and 9,326 / 23,208 pos/s)",
    "AUDIT/det-attention-backward-scoping-2026-10-01.md": "lines 10-15 and 48 "
        "(9,983, 15,280 and 25,703 pos/s, 3.19 and 1.934 s/step)",
    "AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md": "line 15 "
        "(25,703 vs 16,543 pos/s, +55%)",
    "AUDIT/training-audit-2026-10-06.md": "line 455 (60.51 GiB with skip:6, via "
        "HANDOFF/train-step-perf-2026-10-01.md:36; source rows: "
        "AUDIT/perf-train-step-2026-10-01/session2/bench.jsonl lines 2, 4 and 6, config skip:6, "
        "peak_alloc_gib 60.51, allow_stale_shards true)",
}


def section(cited: str) -> str:
    return (
        "\n\n## Correction (2026-10-22)\n\n"
        f"The numbers this report cites from {cited} stand as measured. They were measured with "
        f"`tools/perf_step.py --allow-stale-shards`, and the result rows say so "
        f"(`allow_stale_shards: true` in {ROWS}). They are labelled allow_stale_shards, "
        "not performance of pinned code: the shard set's `qd_data` code had moved, the rows are "
        "not `quick`-marked, and no row carries `stale_code`. Nothing above this section was "
        f"edited. One gap links all five reports that cite them: {GAP}. Fable's ruling of "
        "2026-10-22 (lane X5) confines the flag: from that date every such row is marked "
        "`stale_code: true` and `quick`.\n"
    )


for path, cited in CITES.items():
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if "## Correction (2026-10-22)" in text:
        raise SystemExit(f"{path} already carries the correction")
    target.write_text(text.rstrip("\n") + section(cited), encoding="utf-8")
print(len(CITES), "reports appended")
