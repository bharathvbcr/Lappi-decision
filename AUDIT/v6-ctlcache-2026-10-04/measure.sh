#!/bin/bash
# v6-ctlcache: the control's path to the split on v5's data, Mac. MISS on an empty cache
# directory, then HIT twice, each its own process, under one tools/mac_heavy.sh job. Wall seconds
# and peak RSS come from /usr/bin/time -l around the Python process. See measure_ctl_split.py for
# what runs and where it stops.
#
# The split flags are v5's control flags (V5_CTL_SPLIT: V5_SPLIT without --out and
# --real-backbone), with the Mac's paths, as build/v6-startup-wt/build/v6-profile/run_profile.sh
# gave them to real_ft_run.py (AUDIT/v6-startup-2026-10-04/cache-ab/run_profile.sh). No --max-pairs:
# under --no-repo-history without --commitpackft it bounds nothing and the control refuses it.
#
# Nothing under tools/ or python/ may change between the runs: the key hashes every *.py there.
# Run from the worktree root (build/v6-ctlcache-in_wt.sh), so git reads this worktree.
set -o pipefail
WT=/Users/bharath/Code/research/Lappi-decision/build/v6-ctlcache-wt
SCRATCH=$WT/build/v6-ctlcache-measure
CACHE=$SCRATCH/split-cache
LOGS=$WT/AUDIT/v6-ctlcache-2026-10-04
export QD_PREP_BIN=/Users/bharath/Code/research/Lappi-decision/build/v6-startup-wt/target/release/qd-prep
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
SPLIT=(--no-repo-history --rev 8e6a00952a09e31a09cc3f5270229a935b7235cc
  --defect-class "$WT/data/pool/commitpackft-composed-v2"
  --defect-download /Users/bharath/Code/research/Lappi-decision/data/pool/commitpackft
  --defect-noul "$WT/data/pool/defect-noul-v3c"
  --general-record "$REC" --general-max-rows 200000
  --decisions-pool /Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions-v4
  --drop-before-dedupe /Users/bharath/qd-campaign/v5-predrops-2026-10-03/pre-dedupe-drops.txt
  --exclude-identity-keys /Users/bharath/qd-campaign/v5-containment-predrops-2026-10-03/exclusions.txt)
if [ -e "$CACHE" ]; then
  echo "measure: $CACHE exists; the MISS run needs an empty directory. Not deleting it." >&2
  exit 2
fi
mkdir -p "$SCRATCH"
run() {
  local label=$1
  {
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) start $label HEAD $(git rev-parse HEAD) dirty-tracked $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
    /usr/bin/time -l /Users/bharath/.venvs/ml/bin/python -u "$LOGS/measure_ctl_split.py" \
      --scratch "$SCRATCH" --label "$label" --cache "$CACHE" -- "${SPLIT[@]}" 2>&1
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) end $label rc=$?"
  } | tee "$LOGS/measure-$label.log"
}
run miss
run hit1
run hit2
echo "measure: done"
