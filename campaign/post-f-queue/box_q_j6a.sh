#!/bin/bash
# Idle queue j6a (Fable's idle-queue ruling, Q1 item 3 and Q2): J6(a), F's recipe + prior_kl
# replay on a v4 rebuild, seed 0, quick by construction (one seed), then --needle-control
# 1024,2048,4096 on its checkpoint, as J6(f)-v4 ran. The LAST item: nothing goes after it.
# Closes GAP-PHASE4-REPLAY-SLICE-OVERLAPS-VAL if it runs on a CLEAN attestation.
# Runs only if BOTH:
#   - fsucc's word ($Q/fsucc.word) is quiet. fires:<arm>, refused, absent or malformed: deferred
#     to the re-plan (logged), exit 0;
#   - the replay set is on the box, pinned, and CLEAN (below). Otherwise: deferred, exit 3.
#
# THE THREE PINS BELOW ARE FAIL-CLOSED, NOT PLACEHOLDERS. The replay set is being built by a
# separate lane (Fable lane task 5); the lead fills these in after that lane lands, from its
# HANDOFF. While any of them is the literal UNSET, this script refuses: it logs "deferred" and
# touches .done, so the chain ends and the GPU is released. It never trains on a set it has not
# pinned.
#   REPLAY_SHARDS      the replay-only shard set (<v4 replay rebuild>/shards/replay); the rebuild
#                      root (--out, two levels up) must hold data/pool/train-replay.json and
#                      shards/train/header.json (the pipeline's --replay-shards layout);
#   REPLAY_SHARDS_SHA256  sha256 of $REPLAY_SHARDS/header.json;
#   REPLAY_ATTESTATION the clean attestation tools/replay_decontam.py wrote for that set;
#   REPLAY_ATTESTATION_SHA256  its sha256;
#   REPLAY_WEIGHT      --replay-weight. tools/real_ft_run.py has NO default ("weight on
#                      prior_kl; no default -- say it", :9191-9192 at main 28000af, :7818-7819
#                      at a502670), and no pre-registration, plan or HANDOFF names a value
#                      (GAP-IDLE-WAITERS-J6A-REPLAY-WEIGHT-HAS-NO-SOURCE-2026-10-02). So it is a
#                      pin the lead must fill from a recorded decision, not a number chosen here.
REPLAY_SHARDS=UNSET
REPLAY_SHARDS_SHA256=UNSET
REPLAY_ATTESTATION=UNSET
REPLAY_ATTESTATION_SHA256=UNSET
REPLAY_WEIGHT=UNSET
#
# Recipe: F's (post_f_common.sh F_RECIPE, F's skip) on the rebuild with --replay-partition (the
# rebuild's replay-only rows left gold training, real_ft_run.py corpus_facts), plus
# --replay-shards / --replay-attestation / --replay-cache (a new .npz, built once) /
# --replay-weight, --replay-every at its default 6. From qd-lane8 at a502670, F's exact path:
# replay.py and replay_decontam.py are byte-identical between a502670 and main.
# Cap 32,400 s ($20.61; the human's 2026-10-01 yes via approved()) + needle control 5,400 s.
# Rows: the v4 ablation ledger ($J6V4_LEDGER). Holds gpu.lock.
# Queue: after j5pp (j5pp.done).
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j6a.done' EXIT
touch $Q/j6a.queued
until [ -f $Q/j5pp.done ]; do sleep 60; done
WORD=$(read_succ_word)
if [ "$WORD" != quiet ]; then
  say "j6a DEFERRED to the re-plan: fsucc's word is '${WORD:-absent or not one of: $SUCC_WORDS}', not quiet (Fable Q2: j6a runs only if fsucc was quiet)"
  exit 0
fi
UNSET_PINS=""
for v in REPLAY_SHARDS REPLAY_SHARDS_SHA256 REPLAY_ATTESTATION REPLAY_ATTESTATION_SHA256 REPLAY_WEIGHT; do
  [ "${!v}" = UNSET ] && UNSET_PINS="$UNSET_PINS $v"
done
if [ -n "$UNSET_PINS" ]; then
  say "j6a DEFERRED to the re-plan: fsucc was quiet, but these pins are UNSET:$UNSET_PINS (fail-closed until the lead fills them from the replay lane's HANDOFF); J6(a) NOT RUN"
  exit 3
fi
REPLAY_OUT=$(dirname "$(dirname "$REPLAY_SHARDS")")
for f in "$REPLAY_SHARDS/header.json" "$REPLAY_OUT/shards/train/header.json" "$REPLAY_OUT/data/pool/train-replay.json" "$REPLAY_ATTESTATION"; do
  if [ ! -f "$f" ]; then say "j6a DEFERRED: $f is absent (the replay rebuild layout is <out>/shards/{train,replay}, <out>/data/pool/train-replay.json); J6(a) NOT RUN"; exit 3; fi
done
pin "$REPLAY_SHARDS/header.json" "$REPLAY_SHARDS_SHA256" || { say "j6a DEFERRED: the replay shard header is not the pinned one"; exit 3; }
pin "$REPLAY_ATTESTATION" "$REPLAY_ATTESTATION_SHA256" || { say "j6a DEFERRED: the attestation is not the pinned one"; exit 3; }
if ! "$PY" -c '
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))
clean = a.get("clean")
hits = a.get("hits") or {}
checked = int(a.get("replay_rows_checked", 0))
ok = clean is True and not any(int(v) for v in hits.values()) and checked >= 1
print(f"attestation clean={clean!r} hits={hits} replay_rows_checked={checked}")
sys.exit(0 if ok else 1)
' "$REPLAY_ATTESTATION"; then
  say "j6a DEFERRED: the attestation is not CLEAN (or unreadable); J6(a) NOT RUN"
  exit 3
fi
exec 9>$Q/gpu.lock
flock 9
touch $Q/j6a.started
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
OUT=/home/ubuntu/j6a-v4
CKPT=/home/ubuntu/ckpt/j6a-v4
if [ -e "$OUT/train.log" ]; then say "j6a: $OUT/train.log exists; refusing to run twice"; exit 3; fi
mkdir -p "$OUT" "$CKPT"
# F's data argv with the rebuild's root and the replay partition.
J6A_SPLIT=(--out "$REPLAY_OUT" --no-repo-history
           --rev 881ab304f15ea13529002391dda8520c2ea47af4
           --defect-class data/pool/commitpackft-composed-v1
           --defect-download data/pool/commitpackft
           --defect-noul data/pool/defect-noul-v3b
           --general-record "$REC" --general-max-rows 200000
           --real-backbone "$BACKBONE" --replay-partition)
say "j6a (J6(a) +replay, seed 0) at $(git rev-parse --short HEAD): rebuild $REPLAY_OUT, replay $REPLAY_SHARDS, weight $REPLAY_WEIGHT"
timeout 34200 "$PY" -u tools/real_ft_run.py "${J6A_SPLIT[@]}" "${F_RECIPE[@]}" --devices cuda --seeds 0 \
  --checkpoint-dir "$CKPT" --checkpoint-every 100000 \
  --replay-shards "$REPLAY_SHARDS" --replay-attestation "$REPLAY_ATTESTATION" \
  --replay-cache "$CKPT/replay-prior-cache.npz" --replay-weight "$REPLAY_WEIGHT" \
  --score-val --needle --ood --ood-general-record "$REC" \
  --verdicts-out "$OUT/verdicts.jsonl" --suite-verdicts-out "$OUT/suite-verdicts.jsonl" \
  --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$J6V4_LEDGER" \
  --approved-by "$(approved "J6(a) +replay seed 0 (idle queue j6a)")" \
  2>&1 | tee "$OUT/train.log"
say "J6(a) train+score done (exit $?)"
FT_ROW=$(log_ft_row "$OUT/train.log")
if [ -z "$FT_ROW" ] || [ ! -f "$CKPT/epoch-seed0-cuda.json" ]; then
  say "J6(a): no ft row or checkpoint from this run; its needle control NOT RUN"; exit 4
fi
timeout 5400 "$PY" -u tools/real_ft_run.py "${J6A_SPLIT[@]}" --devices cuda --seeds 0 \
  --score-val --needle --needle-control 1024,2048,4096 \
  --score-checkpoint "$CKPT/epoch-seed0-cuda.json" --score-dtype fp32 \
  --ft-ledger "$J6V4_LEDGER" --ft-row-id "$FT_ROW" \
  "${COST[@]}" --wall-clock-cap-s 5400 \
  --ledger "$J6V4_LEDGER" --suite-verdicts-out "$OUT/needle-control-1024-2048-4096.jsonl" \
  2>&1 | tee "$OUT/needle-control.log"
say "J6(a) needle control done (exit $?)"
say "j6a all done; the idle queue STOPS here (Fable Q2)"
