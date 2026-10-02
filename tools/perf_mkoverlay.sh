#!/bin/bash
# Build a perf overlay: a clone of a read-only lane checked out at a branch commit carried in a
# git bundle. The lane itself is only read (git clone --local); everything written is under
# /home/ubuntu/perf.
# Usage: perf_mkoverlay.sh <commit-sha> [<lane> <bundle> <dest>]
#   defaults (perf session 2): lane /home/ubuntu/qd-lane4 (main 7ed66d7), bundle perf/tsp.bundle,
#   dest perf/overlay. The no-mask P2 screen passes its own (perf_nomask_p2.sh --build).
set -euo pipefail
SHA=${1:?commit sha}
PERF=/home/ubuntu/perf
LANE=${2:-/home/ubuntu/qd-lane4}
BUNDLE=${3:-$PERF/tsp.bundle}
DEST=${4:-$PERF/overlay}
case "$DEST" in
  $PERF/*) ;;
  *) echo "refusing to write $DEST: overlays live under $PERF"; exit 2 ;;
esac
rm -rf "$DEST"
git clone --quiet --local --no-checkout "$LANE" "$DEST"
cd "$DEST"
git fetch --quiet "$BUNDLE" "$SHA"
git checkout --quiet --detach "$SHA"
echo "overlay $DEST at $(git rev-parse HEAD) dirty=[$(git status --porcelain | head -3)]"
