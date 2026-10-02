#!/bin/bash
# Idle queue j6ctl (Fable's idle-queue ruling, lane task 4; HANDOFF/succ-rule-2026-10-02.md "The
# contract box_q_j6ctl.sh must meet"): the J6 v4 arms' control rows, CPU only, niced, no
# gpu.lock. box_f_controls.sh's two blocks -- the letter control of record with per-family
# margins, then the report-only option control -- changed only in --ledger and --verdicts:
#   J6(d)-v4: /home/ubuntu/j6d-v4/verdicts.jsonl -> $J6V4_LEDGER (successor reads this letter row:
#             paired_margin_vs_linear.choice.code.defect_class is J6(d)-v4's target);
#   J6(f)-v4: /home/ubuntu/j6f-v4/verdicts.jsonl -> $J6V4_LEDGER (no rule reads it; Fable names
#             both arms).
# From qd-lane10 at 3460afc with qd-prep-m1 (sha256 pinned), as F's controls, so the row's recipe
# equals F seed 0's c89b89a1 in every key but eval_row_id.
# Queue: waits on j6f.done and j6dv4.done (post-F items 6 and 10); runs beside rung0, cudadev and
# rungd. fsucc waits on j6ctl.done.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j6ctl.done' EXIT
touch $Q/j6ctl.queued
until [ -f $Q/j6f.done ] && [ -f $Q/j6dv4.done ]; do sleep 60; done
touch $Q/j6ctl.started
pin "$PREP_M1" "$PREP_M1_SHA256" || exit 3
lane "$LANE10" "$LANE10_AT" || exit 3
say "j6ctl at $(git rev-parse --short HEAD): J6(d)-v4 first (successor reads it), then J6(f)-v4"
controls_block "J6(d)-v4" "$J6V4_LEDGER" "$J6DV4_OUT/verdicts.jsonl" "$Q/j6ctl-j6dv4.letter.started"
controls_block "J6(f)-v4" "$J6V4_LEDGER" "$J6F_OUT/verdicts.jsonl" "$Q/j6ctl-j6f.letter.started"
say "j6ctl all done"
