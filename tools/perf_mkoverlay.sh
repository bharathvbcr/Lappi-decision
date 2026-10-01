#!/bin/bash
# Build /home/ubuntu/perf/overlay: a clone of the read-only baseline lane (qd-lane4, main
# 7ed66d7) checked out at the train-step branch commit carried in perf/tsp.bundle. The lane
# itself is only read (git clone --local); everything written is under /home/ubuntu/perf.
# Usage: perf_mkoverlay.sh <commit-sha>
set -euo pipefail
SHA=${1:?commit sha}
PERF=/home/ubuntu/perf
rm -rf $PERF/overlay
git clone --quiet --local --no-checkout /home/ubuntu/qd-lane4 $PERF/overlay
cd $PERF/overlay
git fetch --quiet $PERF/tsp.bundle "$SHA"
git checkout --quiet --detach "$SHA"
echo "overlay at $(git rev-parse HEAD) dirty=[$(git status --porcelain | head -3)]"
