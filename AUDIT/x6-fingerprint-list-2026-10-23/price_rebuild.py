"""Print the recorded cost of the v6 LoRA-2B shard build (ledger row 6bf4abfc): wall clock, peak
footprint metrics, recipe argv. Read-only over the committed ledger."""
from __future__ import annotations

import json
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "ledger" / "mac-v6-lora2b-shards-2026-10-18.jsonl"
for line in LEDGER.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if not str(row.get("row_id", "")).startswith("6bf4abfc"):
        continue
    keep = {k: row.get(k) for k in ("row_id", "status", "run_kind", "wall_clock_s", "cost",
                                    "code_commit", "env")}
    print(json.dumps(keep, indent=1, default=str)[:1500])
    recipe = row.get("recipe") or {}
    print("recipe keys:", sorted(recipe))
    print("argv:", recipe.get("argv") or recipe.get("command"))
    metrics = row.get("metrics") or {}
    print("metric names:", sorted(metrics)[:40])
    for name in sorted(metrics):
        if any(t in name for t in ("footprint", "rss", "wall", "peak", "mem")):
            print(name, json.dumps(metrics[name])[:300])
