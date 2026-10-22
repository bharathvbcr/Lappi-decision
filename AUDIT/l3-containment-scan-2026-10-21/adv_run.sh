#!/bin/bash
# The lane's tests, both adversarial files and the native-minhash parity tests, through the
# Makefile's torch-pytest command (TORCH_PYTEST_RUN narrowed to these files), under mac_heavy.
set -u
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
export QD_PREP_BIN=$W/target/release/qd-prep
export HF_HUB_OFFLINE=1
T=$W/python/tests
FILES="$T/test_footprint_peak.py $T/test_adv_l3_footprint_peak.py $T/test_native_minhash_packed.py $T/test_adv_l3_native_minhash.py $T/test_adv_l3_interrupted_scan.py $T/test_containment_scan_interrupted.py $T/test_containment_exclusions.py $T/test_containment_code_fingerprint.py $T/test_qd_prep_minhash_parity.py $T/test_qd_prep_lsh_parity.py $T/test_native_minhash_composed_files.py $T/test_native_minhash_memory.py $T/test_mac_heavy_lock.py $T/test_gaps_ledger.py"
cd $W && make torch-pytest TORCH_PYTEST_RUN="/Users/bharath/.local/bin/uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis --with datasketch python -m pytest $FILES -o addopts= -rfEs -p no:cacheprovider"
echo "make torch-pytest exit $?"
