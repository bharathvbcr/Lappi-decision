#!/bin/bash
# Post-F queue item 4 (campaign/f-j7prime-preregistered.json): F's composed long-context slice,
# report-only (Fable G5(ii)), from qd-lane10 (main 3460afc: the slice decode e242b94 and the
# ens fix 738159b). One --score-plan invocation, separate from item 3, cap 6 h, the argv of
# HANDOFF/ens3-scoring-2026-10-01.md section (f). Kinds, all ["composed"]: seed0, seed1, seed2
# (as the ruling lists them), the J7' average item 0 built, and the logit ensemble of the same
# seeds. If rule (ii) fired those two are over five seeds (no three-seed average exists then),
# making 9 tower decodes instead of 7 under the same pre-registered 6 h cap; the ensemble runs
# last, so the other kinds' rows survive a cap hit. Rows: one quick <tag>-composed-slice row per
# kind in the slice ledger; lines: /home/ubuntu/f-slice/suite-verdicts-<kind>.composed.jsonl.
# Queue: after item 3 (j7p.done) and the no-mask P2 screen in its gap (item 8, nomaskp2, if
# queued); holds gpu.lock.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/fslice.done' EXIT
touch $Q/fslice.queued
until [ -f $Q/j7p.done ]; do sleep 30; done
wait_queued nomaskp2
pin "$RULES" "$RULES_SHA256" || exit 3
OUT=/home/ubuntu/f-slice
mkdir -p "$OUT"

SEEDS=$(f_seeds item4)
if [ -z "$SEEDS" ]; then say "item 4: rule (ii) refused: which average and ensemble are F's candidates is undecided; NOT RUN (escalate)"; exit 3; fi
read -r -a SA <<< "$SEEDS"
N=${#SA[@]}
# shellcheck disable=SC2086 # $SEEDS is a list of seeds
IDS=$(f_ft_ids $SEEDS) || { say "item 4: F's ft rows for seeds $SEEDS are not one configuration (reason above); NOT RUN"; exit 3; }
read -r -a IDA <<< "$IDS"
AVG=$(avg_path "$SEEDS")
if [ ! -f "$AVG" ]; then say "item 4: no average at $AVG; NOT RUN"; exit 3; fi
CKPTS=()
for s in "${SA[@]}"; do CKPTS+=("$F_CKPT/epoch-seed$s-cuda.json"); done

PLAN=$OUT/plan.json
if [ -e "$PLAN" ]; then say "item 4: $PLAN exists; refusing to rescore"; exit 3; fi
KINDS=()
for s in 0 1 2; do
  KINDS+=("{\"name\": \"seed$s\", \"checkpoints\": $(json_strs "${CKPTS[$s]}"), \"ft_row_ids\": $(json_strs "${IDA[$s]}"), \"seeds\": [$s], \"passes\": [\"composed\"]}")
done
KINDS+=("{\"name\": \"avg\", \"checkpoints\": $(json_strs "$AVG"), \"seeds\": $(json_ints "${SA[@]}"), \"passes\": [\"composed\"]}")
KINDS+=("{\"name\": \"ens$N\", \"checkpoints\": $(json_strs "${CKPTS[@]}"), \"ft_row_ids\": $(json_strs "${IDA[@]}"), \"seeds\": $(json_ints "${SA[@]}"), \"passes\": [\"composed\"]}")
{ printf '{"kinds": [\n  %s' "${KINDS[0]}"; printf ',\n  %s' "${KINDS[@]:1}"; printf '\n]}\n'; } > "$PLAN"
say "item 4: plan $PLAN ($(sha256sum "$PLAN" | cut -c1-16)), ${#KINDS[@]} kinds"
cat "$PLAN"

exec 9>$Q/gpu.lock
flock 9
touch $Q/fslice.started
lane "$LANE10" "$LANE10_AT" || exit 3
say "item 4 at $(git rev-parse --short HEAD)"
timeout 6h "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" \
  --score-plan "$PLAN" --ft-ledger "$F_LEDGER" \
  --composed-slice /home/ubuntu/slice-v4-2026-10-01 \
  --score-val --suite-logits \
  --score-dtype fp32 --devices cuda "${COST[@]}" --wall-clock-cap-s 21600 --ledger "$SLICE_LEDGER" \
  --suite-verdicts-out "$OUT/suite-verdicts.jsonl" \
  2>&1 | tee /home/ubuntu/logs/f-composed-slice.log
say "item 4 done (exit $?)"
