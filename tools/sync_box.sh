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
# ## `pull`: the Mac's half of the campaign handoff (tools/campaign_driver.py)
#
# `bash tools/sync_box.sh pull` fetches every campaign snapshot the box has published and the
# Mac has not yet acknowledged. The driver writes `sync/phase-<n>/` (ledgers copied,
# checkpoints hard-linked) and then `sync/phase-<n>.done`, a manifest of sha256s; a snapshot
# is never modified after its `.done` exists. For each pending `<n>` this rsyncs the manifest
# and the directory into `$QD_PULL_DEST`, re-hashes every file against the manifest, and only
# then writes `pulled-<n>.ok` BACK to the box, naming the manifest's own sha256. The driver
# will not terminate the instance until the final snapshot's `pulled-<n>.ok` arrives, so an
# acknowledgement written for data that did not verify would destroy the only copy: a
# mismatch writes nothing and exits 1.
#
# No live-process refusal in this mode, deliberately: the snapshots are immutable, so there is
# no mid-append file to copy. (The source sync above still refuses under a live run.)
#
# Exit status: 0 pulled something (the last line says if it was the FINAL snapshot); 4
# nothing pending; 1 failure. `QD_BOX=local` runs the same steps without ssh, for a driver
# and puller on one host (the Mac smoke test). `tools/campaign_pull_loop.sh` loops it.
#
# Usage:  bash tools/sync_box.sh
#         QD_BOX=ubuntu@1.2.3.4 QD_BOX_KEY=~/.ssh/id_ed25519 bash tools/sync_box.sh
#         QD_PULL_DEST=~/qd-campaign QD_PULL_SYNC_DIR=/home/ubuntu/campaign/state/sync \
#           bash tools/sync_box.sh pull
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

if [ "${1:-}" = "pull" ]; then
  DEST="${QD_PULL_DEST:-}"
  SYNC="${QD_PULL_SYNC_DIR:-}"
  if [ -z "$DEST" ] || [ -z "$SYNC" ]; then
    echo "!!! pull needs QD_PULL_DEST (where snapshots land here) and QD_PULL_SYNC_DIR (the"
    echo "!!! campaign's <state_dir>/sync on the box). There are no defaults: the obvious DEST --"
    echo "!!! this checkout's ledger/ -- is the one place a copy from the box must never write."
    exit 1
  fi
  VERIFY_PY='
import hashlib, json, sys
from pathlib import Path
done, snap = Path(sys.argv[1]), Path(sys.argv[2])
m = json.loads(done.read_text())
bad = []
for rel, want in sorted(m["files"].items()):
    p = snap / rel
    if not p.is_file():
        bad.append(f"missing {rel}")
        continue
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != want:
        bad.append(f"sha256 differs: {rel}")
if bad:
    print("; ".join(bad[:5]))
    sys.exit(1)
print(hashlib.sha256(done.read_bytes()).hexdigest(), len(m["files"]), "final" if m.get("final") else "phase", m.get("phase"))
'
  if [ "$BOX" = "local" ]; then
    LISTING=$(ls "$SYNC" 2>&1) || { echo "=== no sync dir yet at $SYNC; nothing to pull ==="; exit 4; }
    SRC_PREFIX=""
    RSYNC_SSH=()
  else
    LISTING=$($SSH "$BOX" "ls '$SYNC' 2>/dev/null") || { echo "=== no sync dir yet at $BOX:$SYNC; nothing to pull ==="; exit 4; }
    SRC_PREFIX="$BOX:"
    RSYNC_SSH=(-e "$SSH")
  fi
  PENDING=$(printf '%s\n' "$LISTING" | sed -n 's/^phase-\([0-9][0-9]*\)\.done$/\1/p' | sort -n | while read -r n; do
    printf '%s\n' "$LISTING" | grep -qx "pulled-$n.ok" || echo "$n"
  done)
  if [ -z "$PENDING" ]; then
    echo "=== nothing pending: every phase-<n>.done on the box is acknowledged ==="
    exit 4
  fi
  mkdir -p "$DEST" || exit 1
  FINAL=0
  for n in $PENDING; do
    echo "=== pull snapshot $n: ${SRC_PREFIX}${SYNC}/phase-$n/ -> $DEST/phase-$n/ ==="
    rsync -a ${RSYNC_SSH[@]+"${RSYNC_SSH[@]}"} "${SRC_PREFIX}${SYNC}/phase-$n.done" "$DEST/phase-$n.done" || exit 1
    rsync -a ${RSYNC_SSH[@]+"${RSYNC_SSH[@]}"} "${SRC_PREFIX}${SYNC}/phase-$n/" "$DEST/phase-$n/" || exit 1
    RESULT=$("$LOCAL_PY" -c "$VERIFY_PY" "$DEST/phase-$n.done" "$DEST/phase-$n")
    if [ $? -ne 0 ]; then
      echo "!!! snapshot $n does NOT match its manifest: $RESULT"
      echo "!!! no acknowledgement written; the box will not terminate on the strength of it."
      exit 1
    fi
    set -- $RESULT
    DONE_SHA="$1"; NFILES="$2"; KIND="$3"
    OK_TMP="$DEST/pulled-$n.ok"
    printf '{"seq": %s, "done_sha256": "%s", "files": %s, "dest": "%s", "pulled_at": %s}\n' \
      "$n" "$DONE_SHA" "$NFILES" "$(hostname):$DEST/phase-$n" "$(date +%s)" > "$OK_TMP" || exit 1
    # Written to the box under a temp name and renamed, so the driver never reads half an ack.
    if [ "$BOX" = "local" ]; then
      cp "$OK_TMP" "$SYNC/.pulled-$n.ok.tmp" && mv "$SYNC/.pulled-$n.ok.tmp" "$SYNC/pulled-$n.ok" || exit 1
    else
      scp -q -o StrictHostKeyChecking=no -o BatchMode=yes -i "$KEY" "$OK_TMP" "$BOX:$SYNC/.pulled-$n.ok.tmp" || exit 1
      $SSH "$BOX" "mv '$SYNC/.pulled-$n.ok.tmp' '$SYNC/pulled-$n.ok'" || exit 1
    fi
    echo "=== snapshot $n PULLED AND VERIFIED: $NFILES file(s) by sha256; pulled-$n.ok written to the box ==="
    [ "$KIND" = "final" ] && FINAL=1
  done
  if [ "$FINAL" = "1" ]; then
    echo "=== FINAL SNAPSHOT PULLED: the box may now terminate ==="
  fi
  exit 0
fi

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
