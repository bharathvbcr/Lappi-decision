#!/bin/bash
# Export the v5-build tree at 92e1c74 (before prompt format 2) into build/char-fmt2/tree-92e1c74,
# with data/ linked to the v5-build worktree's (the commitpackft download is untracked), for the
# decode-and-diff of GAP-L-V5FMT-CHARACTERIZATION-CRITERION-STALE-2026-10-02. Read-only on git.
set -euo pipefail
WT=/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt
DEST=/Users/bharath/Code/research/Lappi-decision/build/char-fmt2/tree-92e1c74
[ ! -e "$DEST" ] || { echo "export_old: $DEST exists; remove it first" >&2; exit 64; }
mkdir -p "$DEST"
cd "$WT"
git archive 92e1c74 | tar -x -C "$DEST"
rm -rf "$DEST/data"
ln -s "$WT/data" "$DEST/data"
echo "exported 92e1c74 to $DEST ($(find "$DEST" -type f | wc -l | tr -d ' ') files)"
