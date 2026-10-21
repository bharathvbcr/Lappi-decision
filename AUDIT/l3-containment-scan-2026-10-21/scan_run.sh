#!/bin/bash
# One v6 containment scan, measured. Usage: bash build/l3/scan_run.sh <tag> <tree>
# <tree> is the checkout whose tools/ and python/ run (build/l3/base-tree = the unmodified
# baseline). Runs under tools/mac_heavy.sh (the caller), reads the main checkout's data/ and
# build/ read-only, and writes only to this worktree's build/l3/<tag>.*.
set -u
TAG=$1
T=$2
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
M=/Users/bharath/Code/research/Lappi-decision
export QD_PREP_BIN=$W/target/release/qd-prep
export HF_HUB_OFFLINE=1
PY=/Users/bharath/.venvs/ml/bin/python
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
REV=8e6a00952a09e31a09cc3f5270229a935b7235cc
POOL=$M/build/pools/pool-v6-decisions-v1
OUT=$W/build/l3/$TAG
echo "# $TAG start $(date -u +%FT%TZ) tree $T HEAD $(git -C $T rev-parse HEAD) dirty $(git -C $T status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
got=$($PY -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" $POOL/examples.jsonl)
echo "pool sha256 $got"
[ "$got" = "25a88f6e46af92e82e7edb26c0745552badc9b7ea2f009e10e0f4197948d6e0e" ] || { echo "POOL SHA MISMATCH"; exit 90; }
$PY -I $W/tools/footprint_peak.py --timeline $OUT.timeline.txt -- \
  $PY -u $T/tools/containment_scan.py --out-dir $OUT --rev $REV --no-repo-history \
  --defect-class $T/data/pool/commitpackft-composed-v2 \
  --defect-download $T/data/pool/commitpackft \
  --defect-noul $T/data/pool/defect-noul-v3c \
  --general-record $REC --general-max-rows 200000 \
  --decisions-pool $POOL
rc=$?
echo "# $TAG end $(date -u +%FT%TZ) scan exit $rc"
exit $rc
