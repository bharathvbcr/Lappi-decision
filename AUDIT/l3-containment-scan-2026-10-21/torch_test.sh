#!/bin/bash
# The containment test file in the torch venv, as the Makefile's torch-pytest runs it. Usage:
#   bash build/l3/torch_test.sh <tree> [pytest args...]
set -u
T=$1
shift
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
export QD_PREP_BIN=$W/target/release/qd-prep
export HF_HUB_OFFLINE=1
/Users/bharath/.local/bin/uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis --with datasketch python -m pytest $T/python/tests/test_containment_exclusions.py -o addopts= -rfEs -p no:cacheprovider -q "$@"
echo "torch pytest exit $?"
