#!/bin/bash
# Block stage times at 150,000 synthetic rows, base then after then base then after.
set -u
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
PY=/Users/bharath/Code/research/Lappi-decision/.venv/bin/python
for t in base-tree2 after-tree2 base-tree2 after-tree2; do
  echo "== $t"
  $PY $W/build/l3/block_dedupe.py $W/build/l3/$t 150000 2>&1 | rg "minhash:|lsh:|enter" | sed 's#/Users/[^ ]*/qd-prep#qd-prep#' | cut -c1-120
done
