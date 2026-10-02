#!/bin/bash
# Build the rung (d) torch arms' checkout: an overlay of qd-lane8 at a main commit that contains
# 055a7ec (L-rungd-flags: --max-steps, --train-dtype, --span-head-init-digest), carried in a git
# bundle, in the form of HANDOFF/train-step-perf-2026-10-01.md (perf_mkoverlay.sh: clone the lane
# --local, fetch the commit from the bundle, check it out detached). CPU only; writes only under
# /home/ubuntu/perf. qd-lane8 is only read.
# Then it gives the overlay qd-lane8's git-ignored data inputs (data/pool: F's three pool dirs
# hold symlinks into /home/ubuntu/qwen-decision), file by file as `git ls-files --others
# --ignored` lists them in qd-lane8, so no tracked file is touched, and checks:
#   - the overlay is clean and at the full commit given;
#   - every copied entry is the same symlink target (or the same bytes) as qd-lane8's.
# Usage: box_mk_rungd_overlay.sh <40-char commit>. Refuses an existing destination.
set -euo pipefail
SHA=${1:?40-char commit sha}
PERF=/home/ubuntu/perf
LANE=/home/ubuntu/qd-lane8
BUNDLE=$PERF/rungd.bundle
DEST=$PERF/overlay-rungd
MKOVERLAY=$PERF/perf_mkoverlay.sh
MKOVERLAY_SHA256=0b8860b96b015001eb63fee4744cbce31a99a2809dd8ab1c74a2661f7f8994e8
case "$SHA" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
  *) echo "not a commit sha: $SHA"; exit 2 ;;
esac
if [ "${#SHA}" -ne 40 ]; then echo "give the full 40-character sha, not $SHA"; exit 2; fi
if [ -e "$DEST" ]; then echo "$DEST exists; refusing to rebuild over it (remove it by hand first)"; exit 3; fi
if [ "$(sha256sum "$MKOVERLAY" | cut -c1-64)" != "$MKOVERLAY_SHA256" ]; then echo "$MKOVERLAY is not the pinned copy"; exit 3; fi
git -C "$LANE" bundle verify "$BUNDLE"
bash "$MKOVERLAY" "$SHA" "$LANE" "$BUNDLE" "$DEST"
mapfile -t IGN < <(git -C "$LANE" ls-files --others --ignored --exclude-standard -- data)
if [ "${#IGN[@]}" -eq 0 ]; then echo "$LANE has no ignored data inputs; refusing (F's pool dirs are expected)"; exit 3; fi
for p in "${IGN[@]}"; do
  if [ -e "$DEST/$p" ] || [ -L "$DEST/$p" ]; then echo "$DEST/$p already exists in the checkout; refusing to overwrite it"; exit 3; fi
  mkdir -p "$(dirname "$DEST/$p")"
  cp -a "$LANE/$p" "$DEST/$p"
done
bad=0
for p in "${IGN[@]}"; do
  if [ -L "$LANE/$p" ]; then
    [ "$(readlink "$LANE/$p")" = "$(readlink "$DEST/$p")" ] || { echo "symlink differs: $p"; bad=1; }
  else
    cmp -s "$LANE/$p" "$DEST/$p" || { echo "bytes differ: $p"; bad=1; }
  fi
done
dirty=$(git -C "$DEST" status --porcelain)
head=$(git -C "$DEST" rev-parse HEAD)
echo "overlay $DEST at $head; ${#IGN[@]} ignored data entries copied from $LANE; dirty=[${dirty}]"
if [ "$bad" -ne 0 ] || [ -n "$dirty" ] || [ "$head" != "$SHA" ]; then echo "overlay check FAILED"; exit 4; fi
echo "overlay check OK"
