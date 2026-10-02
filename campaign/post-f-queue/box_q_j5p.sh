#!/bin/bash
# Post-F queue item 5 (campaign/f-j7prime-preregistered.json): J5', the shuffled_label control
# x3 on F's exact path, as J3 (box_j3_shuffled.sh) and J5 (box_j5_shuffled_v3.sh) ran it: one
# full-schedule run per seed, --score-val only, each writing a control row that supplements F's
# epoch-score-val eval row of its seed (recipe.eval_row_id) in F's ledger. From F's lane
# (qd-lane8 at a502670) with F's recipe flags verbatim -- --lower-layers-n 8
# --lower-layers-lr-scale 0.1 --checkpoint-skip-layers 6 --batch-tokens 35403, cap 32,400 s --
# minus what --shuffled-label refuses (--checkpoint-dir/--checkpoint-every, --needle, --ood,
# --verdicts-out, --suite-verdicts-out). real_ft_run refuses, before any tower loads, a run whose
# planned ft recipe differs from the target's ft row in anything but the shuffle and the span
# weight (prepare_shuffled_label), so F's exact path is checked, not assumed. No Tier-B change
# (the ruling's tier_b_changes). Target eval rows: qd-post-f-rules eval-row, from the ft row ids
# in F's logs. Cap 32,400 s per seed ($20.61; the human's 2026-10-01 yes, as F's).
# Queue: after item 4 (fslice.done); holds gpu.lock for all three seeds.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j5p.done' EXIT
touch $Q/j5p.queued
until [ -f $Q/fslice.done ]; do sleep 30; done
pin "$RULES" "$RULES_SHA256" || exit 3
exec 9>$Q/gpu.lock
flock 9
touch $Q/j5p.started
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
OUT=/home/ubuntu/j5p
mkdir -p "$OUT"
say "item 5 (J5') at $(git rev-parse --short HEAD)"
for SEED in 0 1 2; do
  FT=$(f_ft_row $SEED)
  TARGET=$("$RULES" eval-row --ledger "$F_LEDGER" --ft-row "$SEED=$FT") || {
    say "J5' seed $SEED NOT RUN: F seed $SEED has no single completed epoch-score-val eval row scored from ft row '${FT}' (the binary's reason is above)"
    continue
  }
  say "J5' seed $SEED -> eval row $TARGET (ft row $FT)"
  timeout 34200 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" "${F_RECIPE[@]}" --devices cuda --seeds $SEED \
    --score-val --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$F_LEDGER" \
    --approved-by "$(approved "J5' shuffled_label seed $SEED (post-F item 5)")" \
    --shuffled-label "$TARGET" \
    2>&1 | tee "$OUT/train-s$SEED.log"
  say "J5' seed $SEED done (exit $?)"
done
say "item 5 all done"
