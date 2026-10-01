#!/bin/bash
# Post-F queue item 2S (campaign/f-j7prime-preregistered.json, seeds_3_4): F seeds 3 and 4, iff
# F's three 8K needle worst buckets spread by more than 0.30 (rule (ii), qd-post-f-rules
# seeds34). F's script form, from F's lane (qd-lane8 at a502670), F's argv with --seeds 3 / 4,
# cap 32,400 s each ($20.61; the human's 2026-10-01 yes, as F's), each followed by
# --needle-control 1024,2048,4096 as F's seeds were. Rows go to F's ledger, so J7' reads one
# ledger for five seeds. If it fires, items 0 and 3 use five seeds; if it is quiet this exits at
# once; if the rule refuses, seeds 3-4 are NOT RUN and items 0 and 3 stop on the same refusal.
# Queue: the first GPU job after items 1-2 (avgnp.done); holds gpu.lock for both seeds.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/s34.done' EXIT
touch $Q/s34.queued
until [ -f $Q/avgnp.done ]; do sleep 30; done
pin "$RULES" "$RULES_SHA256" || exit 3
case "$(f_seeds item2S)" in
  "0 1 2") say "item 2S: rule (ii) quiet: seeds 3-4 do not run"; exit 0 ;;
  "0 1 2 3 4") say "item 2S: rule (ii) fired: seeds 3 and 4 run" ;;
  *) say "item 2S: rule (ii) refused: seeds 3-4 NOT RUN (escalate; items 0 and 3 stop on the same refusal)"; exit 3 ;;
esac
exec 9>$Q/gpu.lock
flock 9
touch $Q/s34.started
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
mkdir -p "$F_OUT" "$F_CKPT"
say "item 2S at $(git rev-parse --short HEAD)"
for SEED in 3 4; do
  if [ -e "$F_OUT/train-s$SEED.log" ]; then say "seed $SEED: $F_OUT/train-s$SEED.log exists; refusing to run it twice"; continue; fi
  timeout 34200 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" "${F_RECIPE[@]}" --devices cuda --seeds $SEED \
    --checkpoint-dir "$F_CKPT" --checkpoint-every 100000 \
    --score-val --needle --ood --ood-general-record "$REC" \
    --verdicts-out "$F_OUT/verdicts-s$SEED.jsonl" --suite-verdicts-out "$F_OUT/suite-verdicts-s$SEED.jsonl" \
    --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$F_LEDGER" \
    --approved-by "$(approved "F seed $SEED (post-F item 2S)")" \
    2>&1 | tee "$F_OUT/train-s$SEED.log"
  say "F seed $SEED train+score done (exit $?)"
  FT_ROW=$(f_ft_row $SEED)
  if [ -z "$FT_ROW" ] || [ ! -f "$F_CKPT/epoch-seed$SEED-cuda.json" ]; then
    say "F seed $SEED: no ft row or checkpoint from this run; its needle control NOT RUN"
    continue
  fi
  timeout 5400 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --devices cuda --seeds $SEED \
    --score-val --needle --needle-control 1024,2048,4096 \
    --score-checkpoint "$F_CKPT/epoch-seed$SEED-cuda.json" --score-dtype fp32 \
    --ft-ledger "$F_LEDGER" --ft-row-id "$FT_ROW" \
    "${COST[@]}" --wall-clock-cap-s 5400 \
    --ledger "$F_LEDGER" --suite-verdicts-out "$F_OUT/needle-control-s$SEED.jsonl" \
    2>&1 | tee "$F_OUT/needle-control-s$SEED.log"
  say "F seed $SEED needle control done (exit $?)"
done
say "item 2S all done"
