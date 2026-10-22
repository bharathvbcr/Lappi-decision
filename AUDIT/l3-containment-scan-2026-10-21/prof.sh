#!/bin/bash
set -u
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
PY=/Users/bharath/Code/research/Lappi-decision/.venv/bin/python
for t in base-tree2 after-tree2; do
  echo "== $t"
  $PY $W/build/l3/enter_profile.py $W/build/l3/$t 60000 2>&1 | rg -v "^$|Ordered by|function calls" | cut -c1-140
done
