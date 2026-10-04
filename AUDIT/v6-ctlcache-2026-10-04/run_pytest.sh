#!/bin/bash
# The v6-ctlcache lane's pytest runs, logged under AUDIT/v6-ctlcache-2026-10-04/<label>.log.
#   bash AUDIT/v6-ctlcache-2026-10-04/run_pytest.sh <label> <pytest args...>
# Run from the worktree root (build/v6-ctlcache-in_wt.sh), under tools/mac_heavy.sh. One
# process (no xdist). QD_PREP_BIN is a release qd-prep built from main's unchanged crates/qd-prep,
# so nothing here builds with cargo.
set -o pipefail
LABEL=$1
shift
LOG=AUDIT/v6-ctlcache-2026-10-04/$LABEL.log
export QD_PREP_BIN=/Users/bharath/Code/research/Lappi-decision/build/v6-startup-wt/target/release/qd-prep
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD $(git rev-parse HEAD) dirty-tracked: $(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
  echo "# QD_PREP_BIN=$QD_PREP_BIN sha256 $(shasum -a 256 "$QD_PREP_BIN" | cut -d' ' -f1)"
  echo "# pytest $*"
} | tee "$LOG"
uv run --offline --no-project --python /Users/bharath/.venvs/ml/bin/python --with pytest \
  --with hypothesis --with 'datasketch>=1.6' python -m pytest -p no:cacheprovider -rs "$@" 2>&1 | tee -a "$LOG"
rc=$?
echo "# exit $rc" | tee -a "$LOG"
exit $rc
