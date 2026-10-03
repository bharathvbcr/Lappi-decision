#!/bin/bash
# Throwaway: dryrun_drops.py on the v5 build's exact argv (v5_flags.sh's), stopping at dedupe.
# Run under tools/mac_heavy.sh. Writes only build/v5-build/dryrun-drops.json.
set -uo pipefail
export POOL_EXAMPLES_SHA256=f3942c1475c695099064d792429e348d0172640db8b961c7874a608b7ee8bb79
. /Users/bharath/Code/research/Lappi-decision/build/v5-build/v5_flags.sh
"$PY" -u /Users/bharath/Code/research/Lappi-decision/build/v5-build/dryrun_drops.py \
  /Users/bharath/qd-campaign/v5-predrops-2026-10-03/pre-dedupe-drops.txt \
  --out /Users/bharath/Code/research/Lappi-decision/build/v5-build/dryrun-out "${CORPUS_FLAGS[@]}" \
  --max-seq-len 10240 --vocab full --val-shards --span-collapse-policy refuse-gold --memo-limit 0 \
  --exclude-identity-keys "$SCAN/exclusions.txt"
