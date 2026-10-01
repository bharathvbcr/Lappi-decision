#!/bin/bash
# Post-F queue item 6 (campaign/f-j7prime-preregistered.json j6f_position; campaign/
# f-v4-preregistered.json needle_attribution_rule): J6(f) = F's recipe minus --lower-layers-n /
# --lower-layers-lr-scale, seed 0 (F seed 0's data order), on F's exact path (qd-lane8 at
# a502670) with --checkpoint-skip-layers 6, so it differs from F only by the lower-layers flags.
# Quick by construction (one seed). Then --needle-control 1024,2048,4096 on its checkpoint, as
# every run's seed now scores. Cap 32,400 s ($20.61; the human's 2026-10-01 yes, as F's).
# Position, by rule (iii) (qd-post-f-rules j6f, decided once F's rows exist):
#   fires  -> right after J5' (j5p.done): 5 -> 6 -> 9 -> 7 -> 10;
#   quiet  -> after the no-mask outcome run (item 9, nomask.done), before tierb (item 7, filler)
#             and J6(d) (item 10): 5 -> 9 -> 6 -> 7 -> 10;
#   refused -> the late slot, logged as such: a refusal moves only its position, never whether
#             it runs, and the late slot never jumps the queue.
# Item 9 orders itself by the position this writes ($Q/j6f.position), never by a second reading
# of the rule, so the two cannot disagree into a deadlock; item 7 waits for both 6 and 9.
# Rows: a new v4 ablation ledger. Its own checkpoint dir, so F seed 0's epoch-seed0-cuda.json is
# never rewritten. Holds gpu.lock.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j6f.done' EXIT
touch $Q/j6f.queued
until [ -f $Q/avgnp.done ]; do sleep 30; done
pin "$RULES" "$RULES_SHA256" || exit 3
WORD=$(j6f_word item6)
if [ "$WORD" = fires ]; then echo early > $Q/j6f.position; else echo late > $Q/j6f.position; fi
case "$WORD" in
  fires) say "item 6: rule (iii) fired: J6(f) runs right after J5'"
         until [ -f $Q/j5p.done ]; do sleep 30; done ;;
  quiet) say "item 6: rule (iii) quiet: J6(f) runs after the no-mask outcome run (item 9), before tierb and J6(d)"
         until [ -f $Q/j5p.done ]; do sleep 30; done
         wait_queued nomask ;;
  *)     say "item 6: rule (iii) REFUSED (JSON above): J6(f)'s position is undecided; it takes the late slot (after item 9), and still runs"
         until [ -f $Q/j5p.done ]; do sleep 30; done
         wait_queued nomask ;;
esac
exec 9>$Q/gpu.lock
flock 9
touch $Q/j6f.started
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
OUT=/home/ubuntu/j6f-v4
CKPT=/home/ubuntu/ckpt/j6f-v4
mkdir -p "$OUT" "$CKPT"
say "item 6 (J6(f)) at $(git rev-parse --short HEAD)"
timeout 34200 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --optimizer master --lr 1e-5 --devices cuda --seeds 0 \
  --epoch --no-memorise --batch-tokens 35403 --checkpoint-skip-layers "$F_SKIP" \
  --checkpoint-dir "$CKPT" --checkpoint-every 100000 \
  --score-val --needle --ood --ood-general-record "$REC" \
  --verdicts-out "$OUT/verdicts.jsonl" --suite-verdicts-out "$OUT/suite-verdicts.jsonl" \
  --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$J6V4_LEDGER" \
  --approved-by "$(approved "J6(f) seed 0 (post-F item 6)")" \
  2>&1 | tee "$OUT/train.log"
say "J6(f) train+score done (exit $?)"
FT_ROW=$(grep -oE 'ft row [0-9a-f-]{36}' "$OUT/train.log" | tail -1 | cut -d' ' -f3)
if [ -z "$FT_ROW" ] || [ ! -f "$CKPT/epoch-seed0-cuda.json" ]; then
  say "J6(f): no ft row or checkpoint from this run; its needle control NOT RUN"; exit 4
fi
timeout 5400 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --devices cuda --seeds 0 \
  --score-val --needle --needle-control 1024,2048,4096 \
  --score-checkpoint "$CKPT/epoch-seed0-cuda.json" --score-dtype fp32 \
  --ft-ledger "$J6V4_LEDGER" --ft-row-id "$FT_ROW" \
  "${COST[@]}" --wall-clock-cap-s 5400 \
  --ledger "$J6V4_LEDGER" --suite-verdicts-out "$OUT/needle-control-1024-2048-4096.jsonl" \
  2>&1 | tee "$OUT/needle-control.log"
say "J6(f) needle control done (exit $?)"
say "item 6 all done"
