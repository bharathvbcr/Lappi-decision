#!/usr/bin/env bash
# Sync a rented GPU box's sources to this checkout, and VERIFY it rather than assume it.
#
# ## Why this is not `git pull`
#
# The box is not a clone that can be fetched into. `git rev-parse HEAD` there reads a commit
# from a previous day with dozens of modified paths, because it is kept up to date by
# COPYING files into a clone pinned at an old commit. `python/qd_train/ledger.py` already
# says so in as many words: *"-dirty is its normal state rather than an exception. The suffix
# is present on every row and distinguishes nothing."* That is
# GAP-CODE-COMMIT-DIRTY-DOES-NOT-PIN-WHAT-RAN, and this script does not fix it.
#
# What replaces it is `what_ran_state`: a sha256 over every module of `qd_train` plus the AST
# import-closure of the invoking tool. That digest is a statement about the bytes that ran and
# it is comparable across machines. So this script does not report "rsync exited 0"; it
# computes the digest on both sides and REFUSES if they differ.
#
# Measured on 2026-09-21: after this ran, the box and this Mac both reported
# 173fa35737ed4b21… over 29 sources; the learning curve's 24 rows all carry that digest; and
# when a later commit changed the tool here, this Mac moved to 9cae19bef139… while the box
# stayed put. The source COUNT was 29 on both — only the digest distinguished them.
#
# ## What is deliberately not synced
#
# `ledger/` and `gaps.jsonl`. Both are written ON the box, and an rsync in either direction
# would overwrite one lane's records with another's. Sources only. Bring ledgers back with
# `scp` after a run finishes, never mid-append.
#
# ## When it refuses
#
# While any training process is live. rsync would replace a tool file under a running
# interpreter, and the already-imported code would not change.
#
# The refusal is right, and the reason above it used to give was not: it said
# `what_ran_state` hashes the file at ROW-WRITE time, so the remaining seeds of a live run
# would record the digest of code they did not run. That WAS true, and stopped being true
# when `RunRecorder.__enter__` began recording the digest before the handlers install -- the
# digest is now captured once at run start, so every row of a run in flight carries the
# code that run actually started with. Verified on 2026-09-22 by doing the forbidden thing:
# a module was scp'd into the package under nine live arms, the box digest moved
# 5bad419d -> 5979f67e for thirteen minutes, and all six rows written in that window carry
# 5bad419d, correctly.
#
# What is still true, and is why this refuses: the NEXT run reads the replaced file. A
# multi-arm driver launches a fresh interpreter per arm, so a sync between arms splits one
# experiment across two code states, and the rows say so individually while the experiment
# as a whole silently stops being one experiment. That is quieter than the failure this
# paragraph used to describe and no less disqualifying.
#
# To check code against the box's data, copy the DATA down or point PYTHONPATH at a scratch
# copy -- never overwrite the package the box trains from.
#
# Usage:  bash tools/sync_box.sh
#         QD_BOX=ubuntu@1.2.3.4 QD_BOX_KEY=~/.ssh/id_ed25519 bash tools/sync_box.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOX="${QD_BOX:-ubuntu@192.222.58.240}"
KEY="${QD_BOX_KEY:-$HOME/.ssh/bharath_m5_macbook_pro.pem}"
REMOTE="${QD_BOX_REPO:-/home/ubuntu/qwen-decision}"
REMOTE_PY="${QD_BOX_PYTHON:-/home/ubuntu/qd-venv/bin/python}"
LOCAL_PY="${QD_PYTHON:-$HOME/.venvs/ml/bin/python}"
TOOL="${QD_SYNC_TOOL:-tools/rung0_real_run.py}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i $KEY"

DIGEST_PY='
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
from qd_train.ledger import what_ran_state
r = what_ran_state(root / "python" / "qd_train", root / sys.argv[2])
print(f"{r.value} {r.n}")
'

echo "=== refusing to sync under a live training process ==="
scp -q -o StrictHostKeyChecking=no -o BatchMode=yes -i "$KEY" \
  "$REPO/tools/check_live_run.sh" "$BOX:/tmp/check_live_run.sh" || exit 1
LIVE=$($SSH "$BOX" 'bash /tmp/check_live_run.sh')
if [ "${LIVE:-1}" != "0" ]; then
  echo "!!! ${LIVE} training process(es) still alive on the box -- NOT syncing."
  $SSH "$BOX" 'ps -eo pid,etime,cmd | grep -E "tools/(rung0_real_run|real_ft_run)\.py" | grep -v grep'
  exit 1
fi
echo "    none"

echo "=== local HEAD ==="
git -C "$REPO" rev-parse HEAD

# This syncs the WORKING TREE, not HEAD, and that difference has teeth in a shared worktree.
# The first run of this script pushed another lane's half-finished edit to the box -- their
# tool file was mid-rewrite and three of their tests were failing on an undefined helper at
# the time. It happened to be harmless, because the break was in a test file the box never
# runs. Nothing about the design made it harmless.
#
# So: name every uncommitted file that falls inside the sync set, and refuse unless the
# caller says they meant it. Printing the list and continuing is what the first version did,
# and a warning nobody has to answer is a warning nobody reads.
DIRTY=$(git -C "$REPO" status --porcelain -- python tools | sed 's/^/    /')
if [ -n "$DIRTY" ]; then
  echo "=== uncommitted files inside the sync set ==="
  echo "$DIRTY"
  if [ "${QD_SYNC_DIRTY:-0}" != "1" ]; then
    echo "!!! These would be pushed to the box and recorded as the code that ran, while not"
    echo "!!! existing in any commit. If they are yours and you meant it, re-run with"
    echo "!!! QD_SYNC_DIRTY=1. If any belong to another lane, ask them first -- ListAgents."
    exit 1
  fi
  echo "    QD_SYNC_DIRTY=1 given; pushing them anyway."
fi

echo "=== rsync python/ and tools/ (ledger/ and gaps.jsonl excluded) ==="
for dir in python tools; do
  rsync -a --itemize-changes \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
    -e "$SSH" "$REPO/$dir/" "$BOX:$REMOTE/$dir/" || exit 1
done

echo "=== what_ran_state for $TOOL, both sides ==="
LOCAL=$("$LOCAL_PY" -c "$DIGEST_PY" "$REPO" "$TOOL")
REMOTE_D=$($SSH "$BOX" "$REMOTE_PY -c '$DIGEST_PY' $REMOTE $TOOL")
echo "    here: $LOCAL"
echo "    box:  $REMOTE_D"

if [ "$LOCAL" != "$REMOTE_D" ]; then
  echo "!!! DIGESTS DIFFER -- the box is not running this checkout's code. Not proceeding."
  exit 1
fi
echo "=== SYNCED AND VERIFIED: box runs the same closure as $(git -C "$REPO" rev-parse --short HEAD) ==="
