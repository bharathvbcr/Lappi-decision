#!/bin/bash
# v5 queue: seeds 3 and 4 (campaign/v5-preregistered.json seeds.seeds_3_4): iff v5's three 8K
# needle worst buckets on the rebuilt suite spread by more than 0.30, read by
# `qd-post-f-rules-v5 seeds34 --preregistration <v5> --f-ledger <v5 ledger>` on v5 seeds 0-2
# (HANDOFF/v5-rules-2026-10-02.md section 7). The word is written once to $Q/v5s34.word.
#   fires:   seeds 3 and 4 in v5's form (box_q_v5.sh's per-seed block: train + score with
#            --batch-order seed, so plan seeds 3 and 4; needle control; the post-seed waiter),
#            rows into v5's ledger; the approved total gains launch.projected_cost_usd.if_seeds_3_4.
#   quiet:   nothing runs.
#   refused, or anything else: nothing runs, logged; never read as quiet.
# Never part of the arm's envelope (seeds.seeds_3_4). Waits on v5.done. Holds gpu.lock per seed.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
# shellcheck source=v5_common.sh
source /home/ubuntu/post-f/v5_common.sh || exit 3
trap 'touch /home/ubuntu/queue/v5s34.done' EXIT
touch $Q/v5s34.queued
UNSET_PINS=$(v5_pins_unset)
if [ -n "$UNSET_PINS" ]; then say "v5s34 deferred: pins UNSET:$UNSET_PINS; seeds 3-4 NOT RUN"; exit 3; fi
PIN_PROBLEMS=$(v5_pins_check)
if [ -n "$PIN_PROBLEMS" ]; then say "v5s34 deferred: pins malformed: $(echo "$PIN_PROBLEMS" | tr '\n' ';'); seeds 3-4 NOT RUN"; exit 3; fi
v5_split_check || exit 3
v5_recipe || exit 3
until [ -f $Q/v5.done ]; do sleep 60; done
if [ -e "$V5S34_WORD" ]; then
  say "v5s34: $V5S34_WORD already exists ('$(head -c 64 "$V5S34_WORD" | tr -d '\n')'): decided once; not deciding again"
  exit 3
fi
v5_verify || { write_atomic "$V5S34_WORD" refused; say "v5s34 refused before the rule; seeds 3-4 NOT RUN"; exit 3; }
v5_ft_args v5 --ft-row 0 1 2 || exit 3
v5_rule seeds34 "fires quiet" seeds34 --preregistration "$V5_PREREG" --f-ledger "$V5_LEDGER" "${V5_FT_ARGS[@]}"
WORD=$V5_WORD
write_atomic "$V5S34_WORD" "$WORD" || { say "v5s34: could not write $V5S34_WORD"; exit 3; }
case "$(v5_s34_action "$WORD")" in
  run) say "v5s34: seeds34 fired: seeds 3 and 4 run (plan seeds 3 and 4)" ;;
  skip) say "v5s34: seeds34 quiet: seeds 3-4 do not run"; exit 0 ;;
  *) say "v5s34: seeds34 '$WORD': seeds 3-4 NOT RUN (escalate; never read as quiet)"; exit 3 ;;
esac
v5_total_line
for SEED in 3 4; do
  v5_lock
  v5_train_seed v5 "$SEED" "v5 seed $SEED (seeds 3-4)"
  FT=$V5_FT
  v5_needle_control v5 "$SEED" "$FT" "v5 seed $SEED"
  v5_post_seed v5 "$SEED" "$FT"
done
v5_total_line
say "v5s34 all done"
