#!/bin/bash
# Post-F queue item 10 (campaign/f-j7prime-preregistered.json): J6(d) on v4 -- F's recipe with
# --lr 3e-5 --beta2 0.95 in place of --lr 1e-5 (so --lower-layers-n 8 --lower-layers-lr-scale 0.1
# --checkpoint-skip-layers 6 --batch-tokens 35403 stay), seed 0, quick by construction, then
# --needle-control 1024,2048,4096 on its checkpoint. F's exact path (qd-lane8 at a502670). Cap
# 32,400 s ($20.61; the human's 2026-10-01 yes, as F's). Replaces the dropped v3 J6(d)
# (box_q_j6d2.sh; kill that waiter: its EXIT trap owns j6d.done).
# Queue: last. It waits for every item of this queue that is queued and not done, by name
# (j7pcpu, s34, j7p, fslice, j5p, j6f, tierb2), for the no-mask lane's nomaskp2 / nomask if that
# lane queued them, and logs -- without waiting on -- any other <job>.queued left without a .done.
# Its own checkpoint dir; rows in the v4 ablation ledger. Holds gpu.lock.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j6dv4.done' EXIT
touch $Q/j6dv4.queued
ITEMS="j7pcpu s34 j7p fslice j5p j6f tierb2"
ahead() {
  local n
  for n in $ITEMS; do
    if [ -f "$Q/$n.queued" ] && [ ! -f "$Q/$n.done" ]; then return 0; fi
  done
  nomask_pending
}
until [ -f $Q/avgnp.done ]; do sleep 30; done
sleep 120
# Re-checked under the lock; a job queued meanwhile gets the lock back rather than a deadlock.
while true; do
  while ahead; do sleep 60; done
  exec 9>$Q/gpu.lock
  flock 9
  if ! ahead; then break; fi
  flock -u 9
  exec 9>&-
done
touch $Q/j6dv4.started
for q in $Q/*.queued; do
  n=$(basename "$q" .queued)
  if [ ! -f "$Q/$n.done" ] && [ "$n" != j6dv4 ]; then say "item 10: $n.queued has no .done and is not this queue's; not waited on"; fi
done
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
OUT=/home/ubuntu/j6d-v4
CKPT=/home/ubuntu/ckpt/j6d-v4
mkdir -p "$OUT" "$CKPT"
say "item 10 (J6(d) on v4) at $(git rev-parse --short HEAD)"
timeout 34200 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --optimizer master --lr 3e-5 --beta2 0.95 \
  --devices cuda --seeds 0 --epoch --no-memorise --batch-tokens 35403 \
  --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers "$F_SKIP" \
  --checkpoint-dir "$CKPT" --checkpoint-every 100000 \
  --score-val --needle --ood --ood-general-record "$REC" \
  --verdicts-out "$OUT/verdicts.jsonl" --suite-verdicts-out "$OUT/suite-verdicts.jsonl" \
  --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$J6V4_LEDGER" \
  --approved-by "$(approved "J6(d) on v4 seed 0 (post-F item 10)")" \
  2>&1 | tee "$OUT/train.log"
say "J6(d) v4 train+score done (exit $?)"
FT_ROW=$(grep -oE 'ft row [0-9a-f-]{36}' "$OUT/train.log" | tail -1 | cut -d' ' -f3)
if [ -z "$FT_ROW" ] || [ ! -f "$CKPT/epoch-seed0-cuda.json" ]; then
  say "J6(d) v4: no ft row or checkpoint from this run; its needle control NOT RUN"; exit 4
fi
timeout 5400 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --devices cuda --seeds 0 \
  --score-val --needle --needle-control 1024,2048,4096 \
  --score-checkpoint "$CKPT/epoch-seed0-cuda.json" --score-dtype fp32 \
  --ft-ledger "$J6V4_LEDGER" --ft-row-id "$FT_ROW" \
  "${COST[@]}" --wall-clock-cap-s 5400 \
  --ledger "$J6V4_LEDGER" --suite-verdicts-out "$OUT/needle-control-1024-2048-4096.jsonl" \
  2>&1 | tee "$OUT/needle-control.log"
say "J6(d) v4 needle control done (exit $?)"
say "item 10 all done"
