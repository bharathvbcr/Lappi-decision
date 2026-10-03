#!/bin/bash
# One serial job (run it through tools/mac_heavy.sh): the 60-pair build from 92e1c74 (format 1)
# and from the v5-build worktree (format 2), then the decode-and-diff between them. The scripts
# are this directory's; the old tree (export_old.sh) and both outputs live in the gitignored
# build/char-fmt2/.
set -uo pipefail
S=$(cd "$(dirname "$0")" && pwd)
D=/Users/bharath/Code/research/Lappi-decision/build/char-fmt2
WT=/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt
export QD_PREP_BIN=$WT/target/release/qd-prep
PY=(uv run --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest --with hypothesis --with datasketch python)
rm -rf "$D/out-old" "$D/out-new"
"${PY[@]}" "$S/build_one.py" "$D/tree-92e1c74" "$D/out-old" || exit 10
"${PY[@]}" "$S/build_one.py" "$WT" "$D/out-new" || exit 11
"${PY[@]}" "$S/decode_diff.py" "$D/out-old" "$D/out-new"
