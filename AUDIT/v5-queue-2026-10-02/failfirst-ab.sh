#!/bin/bash
# Fable's rulings A and B: the new and changed tests against the queue tree at a094c02 (before
# the change), pytest single-process. Run under tools/mac_heavy.sh. Log into AUDIT.
WT=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-afa17f069a43eb74a
A=$WT/AUDIT/v5-queue-2026-10-02
PYTEST=(uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest python -m pytest -p no:cacheprovider -o addopts= -v --tb=line)
cd "$WT" || exit 3
QD_V5_QUEUE_DIR=$WT/build/l-v5-queue/failfirst-a094c02/campaign/post-f-queue "${PYTEST[@]}" \
  python/tests/test_v5_queue_scripts.py > "$A/failfirst-ab-a094c02.txt" 2>&1
echo "fail-first (a094c02 tree) exit $?"
