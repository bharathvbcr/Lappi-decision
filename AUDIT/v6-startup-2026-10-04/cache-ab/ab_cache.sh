#!/bin/bash
# The split cache's pre-registered A/B (HANDOFF/v6-startup-2026-10-04.md): MISS on an empty
# directory, then OFF, HIT interleaved three times. Run under tools/mac_heavy.sh as one job.
WT=/Users/bharath/Code/research/Lappi-decision/build/v6-startup-wt
OUT=$WT/build/v6-profile
CACHE=$OUT/split-cache-ab
if [ -e "$CACHE" ]; then
  echo "ab_cache: $CACHE exists; the MISS arm needs an empty directory. Not deleting it." >&2
  exit 2
fi
run() { bash "$OUT/run_profile.sh" "$1" ab "${@:2}"; }
run ab-miss --split-cache "$CACHE"
for i in 1 2 3; do
  run "ab-off$i"
  run "ab-hit$i" --split-cache "$CACHE"
done
echo "ab_cache: done"
