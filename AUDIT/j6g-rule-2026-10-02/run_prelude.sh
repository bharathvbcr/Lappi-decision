#!/bin/bash
# J6(g)'s Mac prelude, HANDOFF/j6g-rule-2026-10-02.md section 5, as one serial job.
# The lead runs it through tools/mac_heavy.sh, so it holds the machine's one-heavy-job lock.
# Setup (the worktree at a502670, the links, the prelude copy, its sha256), then qd-prep built
# --locked at -j 4, then the prelude itself. Every step's failure stops the script.
set -euo pipefail

REPO=/Users/bharath/Code/research/Lappi-decision
WT=/Users/bharath/qd-campaign/j6g-prelude-a502670
S=/Users/bharath/qd-campaign/j6g-prelude-2026-10-02
M=$REPO/data/pool
TGT=/Users/bharath/qd-campaign/target-j6g-prelude-a502670
PRELUDE_SHA=e9e0c160b0c69833865c572ec39b4f94546402e793bd6c2529baa7160b45bc8f

free_kb=$(df -k /System/Volumes/Data | awk 'NR == 2 { print $4 }')
[ "$free_kb" -ge $((15 * 1024 * 1024)) ] || { echo "prelude: under 15 GB free; not run" >&2; exit 74; }

if [ ! -d "$WT" ]; then
  git -C "$REPO" worktree add --detach "$WT" a502670
fi
[ "$(git -C "$WT" rev-parse HEAD)" = "$(git -C "$REPO" rev-parse a502670^{commit})" ] || { echo "prelude: $WT is not at a502670" >&2; exit 2; }

git -C "$REPO" show 86914aa:campaign/post-f-queue/j6g_prelude_mac.py > "$S/j6g_prelude_mac.py"
got=$(shasum -a 256 "$S/j6g_prelude_mac.py" | awk '{ print $1 }')
[ "$got" = "$PRELUDE_SHA" ] || { echo "prelude: j6g_prelude_mac.py sha256 $got != $PRELUDE_SHA" >&2; exit 2; }

link() { [ -e "$2" ] || ln -s "$1" "$2"; [ "$(readlink "$2")" = "$1" ] || { echo "prelude: $2 does not link to $1" >&2; exit 2; }; }
link /Users/bharath/qd-campaign/corpora-v4-2026-10-02/commitpackft-composed-v1/examples.jsonl "$WT/data/pool/commitpackft-composed-v1/examples.jsonl"
link /Users/bharath/qd-campaign/noul-v3b-2026-10-01/defect-noul-v3b/examples.jsonl "$WT/data/pool/defect-noul-v3b/examples.jsonl"
link "$M/commitpackft-corpus-v3/examples.jsonl" "$WT/data/pool/commitpackft-corpus-v3/examples.jsonl"
link "$M/commitpackft-pool-v2.jsonl" "$WT/data/pool/commitpackft-pool-v2.jsonl"
for l in go python rust typescript; do link "$M/commitpackft/$l.jsonl" "$WT/data/pool/commitpackft/$l.jsonl"; done

# The handoff's [V] link-target hashes (checked against a502670's manifests by lane L-j6g).
check() { local got; got=$(shasum -a 256 "$1" | awk '{ print $1 }'); case "$got" in "$2"*) ;; *) echo "prelude: $1 sha256 $got does not start $2" >&2; exit 2 ;; esac; }
check "$WT/data/pool/commitpackft-composed-v1/examples.jsonl" a63563e5
check "$WT/data/pool/defect-noul-v3b/examples.jsonl" b194f14c
check "$WT/data/pool/commitpackft-corpus-v3/examples.jsonl" 610bb1f0
check "$WT/data/pool/commitpackft-pool-v2.jsonl" 2cb31231
check "$WT/data/pool/commitpackft/go.jsonl" 05a65e97
check "$WT/data/pool/commitpackft/python.jsonl" d167da37
check "$WT/data/pool/commitpackft/rust.jsonl" 5611f370
check "$WT/data/pool/commitpackft/typescript.jsonl" 43d32596

[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || { echo "prelude: $WT tracked tree is dirty" >&2; git -C "$WT" status --short >&2; exit 2; }

# a502670's Cargo.lock is stale (`--locked` refused at 14:06 UTC: "cannot update the lock file"),
# so per the handoff's fallback qd-prep is built from a second, throwaway worktree at a502670
# without --locked (offline), and the prelude's --root stays the clean $WT.
BWT=/Users/bharath/qd-campaign/j6g-prelude-build-a502670
if [ ! -d "$BWT" ]; then
  git -C "$REPO" worktree add --detach "$BWT" a502670
fi
[ "$(git -C "$BWT" rev-parse HEAD)" = "$(git -C "$REPO" rev-parse a502670^{commit})" ] || { echo "prelude: $BWT is not at a502670" >&2; exit 2; }
cargo build -j 4 --release --offline -p qd-prep --manifest-path "$BWT/Cargo.toml" --target-dir "$TGT"
[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || { echo "prelude: $WT tracked tree is dirty after the build" >&2; exit 2; }

REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
BACKBONE=/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
QD_PREP_BIN="$TGT/release/qd-prep" \
nice -n 10 /usr/bin/time -l /Users/bharath/.venvs/ml/bin/python -u "$S/j6g_prelude_mac.py" \
  --root "$WT" --record "$S/j6g-prelude-record.json" -- \
  --out /Users/bharath/qd-campaign/phase4-v4-2026-10-01 --no-repo-history \
  --rev 881ab304f15ea13529002391dda8520c2ea47af4 \
  --defect-class data/pool/commitpackft-composed-v1 --defect-download data/pool/commitpackft \
  --defect-noul data/pool/defect-noul-v3b --general-record "$REC" --general-max-rows 200000 \
  --real-backbone "$BACKBONE" \
  --optimizer master --lr 1e-5 --epoch --no-memorise --batch-tokens 35403 \
  --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers 6 \
  --option-permutation-seed 20260919 --devices mps --seeds 0 \
  --checkpoint-dir "$S/ckpt" --checkpoint-every 100000 \
  --score-val --needle --ood --ood-general-record "$REC" \
  --verdicts-out "$S/verdicts.jsonl" --suite-verdicts-out "$S/suite-verdicts.jsonl" \
  --wall-clock-cap-s 32400 --ledger "$S/never-opened.jsonl" \
  2>&1 | tee "$S/prelude.log"
