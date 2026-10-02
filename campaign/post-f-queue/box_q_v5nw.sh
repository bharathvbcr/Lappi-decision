#!/bin/bash
# v5 queue: the noul-weight arm x3 (campaign/v5-preregistered.json arm_noul_weight; human item 6;
# after v5 and seeds 3-4, before J5': the human's answer 1 at 0b559bb).
# 1. Launch condition (arm_noul_weight.launch_condition, R7): `qd-post-f-rules-v5 v5-noulw --room`
#    on v5 seeds 0-2's rows (never seeds 3-4), its word written once to $Q/v5nw.room.
#      room:    the arm runs.
#      no_room: the arm runs only if the human wrote $Q/V5NW_HUMAN_YES (non-empty, the human's
#               words); otherwise SKIPPED (R7) and J5' takes the slot.
#      refused, or anything else: the arm does not run, even with V5NW_HUMAN_YES: the arm's reading
#               decides room from the same envelope first, so every arm row would read refused.
# 2. The arm: v5's recipe plus --noul-weight <arm_noul_weight.w.value>, seeds 0, 1, 2, each on v5
#    seed s's batch order (--batch-order seed, plan seed s), in box_q_v5.sh's per-seed form
#    (train + score, needle control, the post-seed waiter v5nwtraj-s<N>), rows into the arm ledger
#    gh200-v5-noulw-<date>.jsonl.
# 3. The arm's reading, `qd-post-f-rules-v5 v5-noulw`: wins / quiet / refused, written once to
#    $Q/v5nw.word. It feeds the next re-plan only; nothing here branches on it.
# Waits on v5.done, then on v5s34 if it is queued.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
# shellcheck source=v5_common.sh
source /home/ubuntu/post-f/v5_common.sh || exit 3
trap 'touch /home/ubuntu/queue/v5nw.done' EXIT
touch $Q/v5nw.queued
UNSET_PINS=$(v5_pins_unset)
if [ -n "$UNSET_PINS" ]; then say "v5nw deferred: pins UNSET:$UNSET_PINS; the arm NOT RUN"; exit 3; fi
PIN_PROBLEMS=$(v5_pins_check)
if [ -n "$PIN_PROBLEMS" ]; then say "v5nw deferred: pins malformed: $(echo "$PIN_PROBLEMS" | tr '\n' ';'); the arm NOT RUN"; exit 3; fi
v5_split_check || exit 3
v5_recipe || exit 3
until [ -f $Q/v5.done ]; do sleep 60; done
if [ -f $Q/v5s34.queued ]; then say "v5nw: v5 is done; v5s34 is queued, so v5nw waits for v5s34.done"; fi
wait_queued v5s34
if [ -e "$V5NW_ROOM" ]; then
  say "v5nw: $V5NW_ROOM already exists ('$(head -c 64 "$V5NW_ROOM" | tr -d '\n')'): room was decided once; not deciding again"
  exit 3
fi
v5_verify || { write_atomic "$V5NW_ROOM" refused; say "v5nw refused before the rule; the arm NOT RUN"; exit 3; }

# --- 1. room ----------------------------------------------------------------------------------
v5_ft_args v5 --ft-row 0 1 2 || exit 3
V5_IDS=("${V5_FT_ARGS[@]}")
v5_rule v5nw-room "room no_room" v5-noulw --room --preregistration "$V5_PREREG" \
  --noul-preregistration "$NOUL_PREREG" --v5-ledger "$V5_LEDGER" "${V5_IDS[@]}"
ROOM=$V5_WORD
write_atomic "$V5NW_ROOM" "$ROOM" || { say "v5nw: could not write $V5NW_ROOM"; exit 3; }
YES=no
if [ -s "$V5NW_HUMAN_YES" ]; then YES=yes; fi
if [ "$(v5_room_action "$ROOM" "$YES")" != run ]; then
  case "$ROOM" in
    no_room) say "v5nw SKIPPED (R7): v5 holds both targets on all three seeds (no_room) and $V5NW_HUMAN_YES is not written; J5' takes the slot"; exit 0 ;;
    *) say "v5nw REFUSED: v5-noulw --room said '$ROOM'; the arm NOT RUN (a refused room makes every arm row read refused; V5NW_HUMAN_YES does not override it); J5' takes the slot"; exit 3 ;;
  esac
fi
NW_LABEL="noul-weight arm"
if [ "$ROOM" = no_room ]; then
  NW_LABEL="noul-weight arm on no_room by $V5NW_HUMAN_YES: '$(head -c 300 "$V5NW_HUMAN_YES" | tr '\n' ' ')'"
  say "v5nw: no_room, and $V5NW_HUMAN_YES is written: the arm runs on the human's yes"
else
  say "v5nw: room: the arm runs"
fi
if [ -e "$V5NW_LEDGER" ]; then say "v5nw: $V5NW_LEDGER exists; the arm's rows go into a new ledger, refusing"; exit 3; fi

# --- 2. the arm -------------------------------------------------------------------------------
v5_total_line
for SEED in 0 1 2; do
  v5_lock
  v5_train_seed v5nw "$SEED" "$NW_LABEL seed $SEED (--noul-weight $V5NW_W)"
  FT=$V5_FT
  v5_needle_control v5nw "$SEED" "$FT" "arm seed $SEED"
  v5_post_seed v5nw "$SEED" "$FT"
done

# --- 3. the arm's reading ---------------------------------------------------------------------
if [ -e "$V5NW_WORD" ]; then say "v5nw: $V5NW_WORD already exists; not reading the arm again"; exit 3; fi
v5_ft_args v5nw --arm-ft-row 0 1 2 || exit 3
v5_rule v5nw "wins quiet" v5-noulw --preregistration "$V5_PREREG" \
  --noul-preregistration "$NOUL_PREREG" --v5-ledger "$V5_LEDGER" "${V5_IDS[@]}" \
  --arm-ledger "$V5NW_LEDGER" "${V5_FT_ARGS[@]}"
write_atomic "$V5NW_WORD" "$V5_WORD" || exit 3
case "$(v5_noulw_action "$V5_WORD")" in
  record) say "v5nw: the arm reads '$V5_WORD' (feeds the next re-plan only; JSON $V5_RULE_JSON)" ;;
  *) say "v5nw: the arm's reading REFUSED ('$V5_WORD'; JSON $V5_RULE_JSON); never read as quiet" ;;
esac
v5_total_line
say "v5nw all done"
