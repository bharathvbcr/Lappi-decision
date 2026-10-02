#!/bin/bash
# Idle queue rungd (Fable's idle-queue ruling, Q1 item 2 and Q2): rung (d)'s two torch reference
# arms on the GH200, T-bf16 then T-fp32, exactly as HANDOFF/rungd-flags-2026-10-02.md writes them
# ("The two arm commands"). Seed 0, the first 200 batches of F's epoch order, quick by
# construction (rule 8: the --max-steps reason is on every row they write).
#   T-bf16: F's flags + --max-steps 200 --train-attention-mask none --deterministic
#           --span-head-init-digest; cap 3,600 s ($2.29).
#   T-fp32: the same + --train-dtype fp32, minus --train-attention-mask none (refused: no fp32
#           kernel on CUDA's flash backend) and minus --checkpoint-skip-layers (full
#           checkpointing); cap 10,800 s ($6.87).
# Both under rule 4's $20 single-GPU line, so no --approved-by. T-fp32 runs whatever T-bf16's
# exit; a capped arm ends wall_clock_cap, is not a parity result, and is rerun (rule 5).
# The flags exist only on main, so the arms run from a NEW overlay of qd-lane8 at main 28000af
# (contains 055a7ec), built from a git bundle by box_mk_rungd_overlay.sh. Before T-bf16 this
# re-checks, read-only: the overlay is clean at the full sha; F's three pool dirs are qd-lane8's
# (same symlink targets); the fetch record F read is the pinned file; the backbone snapshot and
# F's split exist; /home/ubuntu/box_q_f.sh (the argv F_SPLIT copies) is unchanged; f.ckpt-skip is
# still F's 6.
# Queue: after cudadev (cudadev.done). Holds gpu.lock for both arms. New ledger, new dirs.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
OVL=/home/ubuntu/perf/overlay-rungd
OVL_AT=28000af661a062ae54f4a69b17e8638d83b5e7ba
BOX_Q_F=/home/ubuntu/box_q_f.sh
BOX_Q_F_SHA256=a1bab9b033c519c98ac74ee6bba273a5d347031bb1b04657d7c556ec529e308b
F_SPLIT_DIR=/home/ubuntu/phase4-v4-2026-10-01
RLEDGER=/home/ubuntu/ledger/gh200-rungd-torch-2026-10-02.jsonl
ROUT=/home/ubuntu/rungd
TBF16_CAP_S=3600
TFP32_CAP_S=10800
# Outer backstops: the tool's own cap plus 1,800 s, the post-F form (32,400 -> 34,200).
TBF16_OUTER_S=5400
TFP32_OUTER_S=12600
trap 'touch /home/ubuntu/queue/rungd.done' EXIT
touch $Q/rungd.queued
until [ -f $Q/cudadev.done ]; do sleep 60; done
exec 9>$Q/gpu.lock
flock 9
touch $Q/rungd.started

pin "$BOX_Q_F" "$BOX_Q_F_SHA256" || exit 3
pin "$REC" "$REC_SHA256" || exit 3
if [ ! -f "$BACKBONE/config.json" ]; then say "rungd: no $BACKBONE/config.json; NOT RUN"; exit 3; fi
if [ ! -f "$F_SPLIT_DIR/shards/train/header.json" ]; then say "rungd: no $F_SPLIT_DIR/shards/train/header.json; NOT RUN"; exit 3; fi
f_skip_ok || exit 3
for d in commitpackft-composed-v1 commitpackft defect-noul-v3b; do
  for f in "$LANE8/data/pool/$d"/*; do
    rel=data/pool/$d/${f##*/}
    if [ -L "$f" ]; then
      if [ "$(readlink "$f")" != "$(readlink "$OVL/$rel")" ] || [ ! -e "$OVL/$rel" ]; then
        say "rungd: $OVL/$rel is not qd-lane8's $(readlink "$f"); NOT RUN"; exit 3
      fi
    elif ! cmp -s "$f" "$OVL/$rel"; then
      say "rungd: $OVL/$rel differs from qd-lane8's; NOT RUN"; exit 3
    fi
  done
done
lane_full "$OVL" "$OVL_AT" || exit 3
if [ -e "$ROUT/t-bf16.log" ] || [ -e "$ROUT/t-fp32.log" ]; then say "rungd: $ROUT already holds an arm's log; refusing to run twice"; exit 3; fi
mkdir -p "$ROUT" /home/ubuntu/ckpt/rungd-tbf16 /home/ubuntu/ckpt/rungd-tfp32
say "rungd at $(git rev-parse HEAD): T-bf16 (cap ${TBF16_CAP_S}s), then T-fp32 (cap ${TFP32_CAP_S}s) -> $RLEDGER"

timeout "$TBF16_OUTER_S" "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --optimizer master --lr 1e-5 --devices cuda --seeds 0 \
  --epoch --no-memorise --batch-tokens 35403 \
  --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers "$F_SKIP" \
  --max-steps 200 --train-attention-mask none --deterministic --span-head-init-digest \
  --checkpoint-dir /home/ubuntu/ckpt/rungd-tbf16 --checkpoint-every 100000 \
  --wall-clock-cap-s "$TBF16_CAP_S" "${COST[@]}" --ledger "$RLEDGER" \
  2>&1 | tee "$ROUT/t-bf16.log"
say "T-bf16 done (exit $?; 1 with the ft row written is the epoch arm's letter-loss claim, 124 the outer backstop); ft row: $(log_ft_row "$ROUT/t-bf16.log")"

timeout "$TFP32_OUTER_S" "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --optimizer master --lr 1e-5 --devices cuda --seeds 0 \
  --epoch --no-memorise --batch-tokens 35403 \
  --lower-layers-n 8 --lower-layers-lr-scale 0.1 \
  --max-steps 200 --deterministic --train-dtype fp32 --span-head-init-digest \
  --checkpoint-dir /home/ubuntu/ckpt/rungd-tfp32 --checkpoint-every 100000 \
  --wall-clock-cap-s "$TFP32_CAP_S" "${COST[@]}" --ledger "$RLEDGER" \
  2>&1 | tee "$ROUT/t-fp32.log"
say "T-fp32 done (exit $?); ft row: $(log_ft_row "$ROUT/t-fp32.log")"
say "rungd all done"
