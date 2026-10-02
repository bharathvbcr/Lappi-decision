#!/bin/bash
# Post-F queue item 3 (campaign/f-j7prime-preregistered.json): F's J7' decision rows, from
# qd-lane10 (main 3460afc, >= 600b241: the ens combine_readouts host fix 738159b and the
# RECIPE_PIECE_KEYS mirror 0b95adb). fp32, T = 1, into a new J7' ledger.
#   a. one --score-plan invocation (cap 4 h): kinds seed<N> ["ood"] for each of F's J7' seeds,
#      avg ["gates"], ens<N> ["gates"] (the logit ensemble of the same seeds), and avg-np
#      ["gates"] iff rule (i) says J4's avg-np qualifies. The seed set is rule (ii)'s; ft row ids
#      come from F's logs at run time and are checked against F's ledger by qd-post-f-rules.
#   b. separate --needle-control 1024,2048,4096 invocations (cap 1 h each; a plan refuses
#      --needle-control) on avg, on ens<N> (--score-checkpoint of the seeds' checkpoints, the
#      form the Mac argv check accepted) and on avg-np iff it qualified.
#   c. as J7g did (box_q_j7g.sh): the release export's 100-row letter-logit parity on the GPU,
#      then, after the GPU is released, calibration fits on the average's verdicts.
# Queue: after item 2S (s34.done) and item 0's averages (j7pavg.done); holds gpu.lock for a-c.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j7p.done' EXIT
touch $Q/j7p.queued
until [ -f $Q/s34.done ] && [ -f $Q/j7pavg.done ]; do sleep 30; done
pin "$RULES" "$RULES_SHA256" || exit 3
J7P=/home/ubuntu/j7p
mkdir -p "$J7P"

SEEDS=$(f_seeds item3)
if [ -z "$SEEDS" ]; then say "item 3: rule (ii) refused: F's J7' seed set is undecided; NOT RUN (escalate)"; exit 3; fi
read -r -a SA <<< "$SEEDS"
N=${#SA[@]}
# shellcheck disable=SC2086 # $SEEDS is a list of seeds
IDS=$(f_ft_ids $SEEDS) || { say "item 3: F's ft rows for seeds $SEEDS are not one configuration (reason above); NOT RUN"; exit 3; }
read -r -a IDA <<< "$IDS"
AVG=$(avg_path "$SEEDS")
AVGNP=$(avgnp_path "$SEEDS")
if [ ! -f "$AVG" ] || [ ! -f "$AVG.manifest.json" ]; then say "item 3: no average at $AVG (item 0's log says why); NOT RUN"; exit 3; fi
CKPTS=()
for s in "${SA[@]}"; do CKPTS+=("$F_CKPT/epoch-seed$s-cuda.json"); done

AVGNP_IN=0
case "$(rule avgnp item3 --j7-ledger "$J7_LEDGER" --j4-ledger "$J4_LEDGER")" in
  qualifies)
    if [ -f "$AVGNP" ]; then AVGNP_IN=1; say "item 3: rule (i): J4's avg-np qualifies; F's avg-np is scored"
    else say "item 3: rule (i): J4's avg-np qualifies, but F's avg-np was not built ($AVGNP; item 0's log); avg-np NOT RUN"; fi ;;
  fails) say "item 3: rule (i): J4's avg-np fails; avg-np is dropped before F's J7' (the pre-registered if_not); F's J7' scores avg and ens$N only" ;;
  *) say "item 3: rule (i) REFUSED (its JSON above names why: a missing row, a not_run metric, or a clause whose pre-registered decimal and cited row disagree); avg-np NOT RUN; the human decides, and it can be scored later by hand" ;;
esac

PLAN=$J7P/plan-seed$(echo "$SEEDS" | tr -d ' ').json
if [ -e "$PLAN" ]; then say "item 3: $PLAN exists; refusing to rescore"; exit 3; fi
KINDS=()
for i in "${!SA[@]}"; do
  s=${SA[$i]}
  KINDS+=("{\"name\": \"seed$s\", \"checkpoints\": $(json_strs "${CKPTS[$i]}"), \"ft_row_ids\": $(json_strs "${IDA[$i]}"), \"seeds\": [$s], \"passes\": [\"ood\"]}")
done
KINDS+=("{\"name\": \"avg\", \"checkpoints\": $(json_strs "$AVG"), \"seeds\": $(json_ints "${SA[@]}"), \"passes\": [\"gates\"]}")
KINDS+=("{\"name\": \"ens$N\", \"checkpoints\": $(json_strs "${CKPTS[@]}"), \"ft_row_ids\": $(json_strs "${IDA[@]}"), \"seeds\": $(json_ints "${SA[@]}"), \"passes\": [\"gates\"]}")
if [ $AVGNP_IN -eq 1 ]; then
  KINDS+=("{\"name\": \"avg-np\", \"checkpoints\": $(json_strs "$AVGNP"), \"seeds\": $(json_ints "${SA[@]}"), \"passes\": [\"gates\"]}")
fi
{ printf '{"kinds": [\n  %s' "${KINDS[0]}"; printf ',\n  %s' "${KINDS[@]:1}"; printf '\n]}\n'; } > "$PLAN"
say "item 3: plan $PLAN ($(sha256sum "$PLAN" | cut -c1-16)), ${#KINDS[@]} kinds"
cat "$PLAN"

exec 9>$Q/gpu.lock
flock 9
touch $Q/j7p.started
lane "$LANE10" "$LANE10_AT" || exit 3
say "item 3 at $(git rev-parse --short HEAD)"

say "a. J7' plan"
timeout 4h "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" \
  --score-plan "$PLAN" --ft-ledger "$F_LEDGER" \
  --score-val --needle --ood --ood-general-record "$REC" --suite-logits \
  --score-dtype fp32 --devices cuda "${COST[@]}" --wall-clock-cap-s 14400 --ledger "$J7P_LEDGER" \
  --verdicts-out "$J7P/verdicts.jsonl" --suite-verdicts-out "$J7P/suite-verdicts.jsonl" \
  2>&1 | tee "$J7P/plan.log"
say "a. J7' plan done (exit $?)"

control() {  # name, then the --score-checkpoint pairing argv
  local name=$1
  shift
  say "b. needle length control on $name"
  timeout 1h "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" "$@" --ft-ledger "$F_LEDGER" \
    --score-val --needle --needle-control 1024,2048,4096 \
    --score-dtype fp32 --devices cuda "${COST[@]}" --wall-clock-cap-s 3600 --ledger "$J7P_LEDGER" \
    --suite-verdicts-out "$J7P/$name-needle-control-1024-2048-4096.jsonl" \
    2>&1 | tee "$J7P/$name-needle-control.log"
  say "b. needle length control on $name done (exit $?)"
}
control avg --score-checkpoint "$AVG" --seeds "${SA[@]}"
control "ens$N" --score-checkpoint "${CKPTS[@]}" --ft-row-id "${IDA[@]}" --seeds "${SA[@]}"
if [ $AVGNP_IN -eq 1 ]; then
  control avg-np --score-checkpoint "$AVGNP" --seeds "${SA[@]}"
else
  say "b. needle length control on avg-np NOT RUN: avg-np is not in F's J7' (rule (i), above)"
fi

RELEASE=/home/ubuntu/release/p4-v4-avg-masters-v1
PARITY=/home/ubuntu/logs/export-parity-p4-v4-avg-cuda.json
if [ -f "$RELEASE/release_manifest.json" ]; then
  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" -u tools/export_letter_parity.py --source "$AVG" \
    --release "$RELEASE" --base-snapshot "$BACKBONE" --run-out /home/ubuntu/phase4-v4-2026-10-01 \
    --rows 100 --device cuda --out-json "$PARITY"
  PARITY_RC=$?
  say "c. export parity cuda (exit $PARITY_RC)"
else
  say "c. export parity NOT RUN: no $RELEASE/release_manifest.json (item 0's export)"
fi
touch $Q/j7p.done
flock -u 9
say "item 3 GPU work done"

if [ -f "$RELEASE/release_manifest.json" ] && [ "${PARITY_RC:-0}" -ne 0 ] && [ ! -s "$PARITY" ]; then
  "$PY" -u tools/export_letter_parity.py --source "$AVG" --release "$RELEASE" \
    --base-snapshot "$BACKBONE" --run-out /home/ubuntu/phase4-v4-2026-10-01 \
    --rows 100 --device cpu --out-json /home/ubuntu/logs/export-parity-p4-v4-avg-cpu.json
  say "c. export parity cpu (exit $?)"
fi
if [ -f "$J7P/verdicts-avg.jsonl" ]; then
  "$PY" -u tools/calib_fit_row.py --bin /home/ubuntu/bin/qd-calib-fit \
    --verdicts "$J7P/verdicts-avg.jsonl" --population all --name p4-v4-avg-all-v1 \
    --eval-ledger "$J7P_LEDGER" --ledger "$CALIB_LEDGER" --out-dir /home/ubuntu/calib/p4-v4-avg-all
  say "c. calibration avg all (exit $?)"
  "$PY" -u tools/calib_fit_row.py --bin /home/ubuntu/bin/qd-calib-fit \
    --verdicts "$J7P/verdicts-avg.jsonl" --population two-fold --split-key j7-calib \
    --name p4-v4-avg-2fold-v1 --eval-ledger "$J7P_LEDGER" --ledger "$CALIB_LEDGER" \
    --out-dir /home/ubuntu/calib/p4-v4-avg-2fold
  say "c. calibration avg two-fold (exit $?)"
else
  say "c. calibration on the average NOT RUN: no $J7P/verdicts-avg.jsonl"
fi
say "item 3 all done"
