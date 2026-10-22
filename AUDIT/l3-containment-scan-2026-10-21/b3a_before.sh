#!/bin/bash
set -u
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
cargo build --release -p qd-prep --manifest-path $W/Cargo.toml || exit 90
export QD_PREP_BIN=$W/target/release/qd-prep
/Users/bharath/Code/research/Lappi-decision/.venv/bin/python -m pytest $W/python/tests/test_containment_exclusions.py $W/python/tests/test_containment_strip.py $W/python/tests/test_qd_prep_containment_parity.py -o addopts= -rfEs -p no:cacheprovider
echo "pytest exit $?"
