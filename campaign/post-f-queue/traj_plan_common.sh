#!/bin/bash
# v6: a trajectory waiter's GPU block. Every retained snapshot of one seed is scored as v5's
# OOD-only trajectory row (tag trajectory-ood, metrics.checkpoint_step, suite lines gated
# ood_abstain.trajectory) through tools/real_ft_run.py --score-plan 'trajectory' kinds: one
# process startup per plan of up to TRAJ_PLAN_MAX_KINDS snapshots, not one per snapshot.
#
# Why: v5's waiter (box_q_v5traj.sh lines 72-85) runs one --score-checkpoint process per
# snapshot. Each one rebuilds the split (MinHash/LSH), the val set and the OOD suite. On the
# H100 that was ~6.5 min of wall time for ~16 s of GPU per snapshot, so the 3,600 s cap fits ~9 of
# 13 snapshots (HANDOFF/lead-pipeline-2026-10-03.md, ~07:15Z 2026-10-04). Here the prelude is
# paid once per plan, and each further snapshot costs its model load (~24 s) and scoring (~16 s).
#
# What is the same as v5: the snapshots (<ckpt>/epoch-seed<N>-cuda-step<S>.json), their order
# (the final step first, then the rest ascending), the rows (python/tests/
# test_score_plan_trajectory.py holds the plumbing parity on the toy corpus; 2B parity is NOT RUN),
# the per-snapshot suite-verdict file <out>/traj-s<N>-step<S>.jsonl (the plan writes each kind's
# lines to <stem>-<kind>.jsonl, and the kinds are named step<S>), the one cap for all snapshots,
# and "a snapshot whose suite-verdict file exists is not scored twice".
# What differs: one log per plan (<out>/traj-s<N>-<T0>-plan<K>.log) rather than one per snapshot;
# each row's notes carry the plan's sentence and each suite line its kind (score_kind); each
# plan's --wall-clock-cap-s is what was left of the cap when it started. A snapshot that a
# failed plan was scoring is left NOT RUN, the rest of that plan go back on the queue, and a plan
# that scored nothing stops the block: its startup failed, and a retry would pay it again.
#
# Sourced by a waiter that already holds its lane's GPU lock and has sourced post_f_common.sh
# (say). Reads PY (the box's python), REC (--ood-general-record), and the arrays TRAJ_SPLIT (the
# run's data argv, v5's "${V5_SPLIT[@]}") and TRAJ_COST (its cost argv, v5's "${V5_COST[@]}").
#
#   traj_score_plans NAME SEED FT CKPT_DIR OUT_DIR LEDGER CAP_S MIN_LEFT_S T0
#
# Returns 0 when it ran (what it could not score is said NOT RUN), 2 on a usage error and 3 on a
# refusal before anything ran.

#: Kinds one plan may hold: tools/real_ft_run.py's PLAN_MAX_KINDS. python/tests/
#: test_traj_plan_common.py fails if the two differ; raising it is a change to real_ft_run.py.
TRAJ_PLAN_MAX_KINDS=8

traj_score_plans() {
  if [ "$#" -ne 9 ]; then
    say "usage: traj_score_plans NAME SEED FT CKPT_DIR OUT_DIR LEDGER CAP_S MIN_LEFT_S T0; got $# arguments"
    return 2
  fi
  local name=$1 seed=$2 ft=$3 ckpt=$4 out=$5 ledger=$6 cap=$7 min_left=$8 t0=$9
  local f s v steps final order chunk plan log left rc sep scored
  local -a pending=() this=() unscored=() skipped=()
  for v in "$seed" "$cap" "$min_left" "$t0"; do
    case "$v" in ''|*[!0-9]*) say "$name: '$v' (seed, cap, minimum or start time) is not a whole number; NOT RUN"; return 3 ;; esac
  done
  if ! echo "$ft" | grep -Eqx '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'; then
    say "$name: ft row '$ft' is not a row id; NOT RUN"
    return 3
  fi
  # Each path goes into a JSON string as it is, so only characters JSON leaves alone are admitted.
  for v in "$ckpt" "$out" "$ledger"; do
    if ! echo "$v" | grep -Eqx '/[A-Za-z0-9/._-]+'; then
      say "$name: '$v' is not an absolute path of [A-Za-z0-9/._-]; it would be written into the plan unescaped; NOT RUN"
      return 3
    fi
  done
  steps=$(for f in "$ckpt/epoch-seed$seed-cuda-step"*.json; do
            [ -f "$f" ] || continue
            s=${f##*-step}; s=${s%.json}
            echo "$s" | grep -Eqx '[0-9]+' && echo "$s"
          done | sort -n)
  if [ -z "$steps" ]; then
    say "$name: no retained snapshot epoch-seed$seed-cuda-step*.json in $ckpt; trajectory NOT RUN"
    return 0
  fi
  final=$(echo "$steps" | tail -1)
  order="$final $(echo "$steps" | grep -vx "$final" | tr '\n' ' ')"
  say "$name: $(echo "$steps" | wc -l | tr -d ' ') snapshots (steps $(echo "$steps" | tr '\n' ' ')); scoring order $order; cap ${cap}s; one --score-plan per $TRAJ_PLAN_MAX_KINDS snapshots"
  for s in $order; do
    if [ -e "$out/traj-s$seed-step$s.jsonl" ]; then
      say "$name step $s: $out/traj-s$seed-step$s.jsonl exists; refusing to score it twice"
      continue
    fi
    pending+=("$s")
  done
  chunk=0
  while [ "${#pending[@]}" -gt 0 ]; do
    left=$(( cap - ($(date +%s) - t0) ))
    if [ "$left" -lt "$min_left" ]; then break; fi
    this=("${pending[@]:0:TRAJ_PLAN_MAX_KINDS}")
    pending=("${pending[@]:TRAJ_PLAN_MAX_KINDS}")
    chunk=$((chunk + 1))
    plan=$out/traj-s$seed-$t0-plan$chunk.json
    log=$out/traj-s$seed-$t0-plan$chunk.log
    if [ -e "$plan" ] || [ -e "$log" ]; then
      say "$name: $plan or its log exists; refusing to write over another waiter's plan"
      return 3
    fi
    sep=""
    {
      printf '{"kinds": ['
      for s in "${this[@]}"; do
        printf '%s{"name": "step%s", "checkpoints": ["%s"], "ft_row_ids": ["%s"], "seeds": [%s], "passes": ["trajectory"]}' \
          "$sep" "$s" "$ckpt/epoch-seed$seed-cuda-step$s.json" "$ft" "$seed"
        sep=", "
      done
      printf ']}\n'
    } > "$plan"
    say "$name plan $chunk: steps ${this[*]}, ${left}s of the cap left ($plan)"
    timeout "$left" "$PY" -u tools/real_ft_run.py "${TRAJ_SPLIT[@]}" --devices cuda \
      --ood --ood-general-record "$REC" --score-plan "$plan" --ft-ledger "$ledger" \
      "${TRAJ_COST[@]}" --wall-clock-cap-s "$left" \
      --ledger "$ledger" --suite-verdicts-out "$out/traj-s$seed.jsonl" \
      2>&1 | tee "$log"
    # timeout's status (124 when the cap stopped it), never tee's: read here, it does not
    # depend on whether the caller set pipefail.
    rc=${PIPESTATUS[0]}
    say "$name plan $chunk done (exit $rc)"
    scored=0
    unscored=()
    for s in "${this[@]}"; do
      if [ -e "$out/traj-s$seed-step$s.jsonl" ]; then scored=$((scored + 1)); else unscored+=("$s"); fi
    done
    if [ "${#unscored[@]}" -eq 0 ]; then continue; fi
    if [ "$rc" -eq 124 ]; then
      say "$name plan $chunk: the cap stopped it after $scored of ${#this[@]} snapshots"
      break
    fi
    if [ "$rc" -eq 0 ]; then
      say "$name plan $chunk exited 0 and wrote no suite verdicts for steps ${unscored[*]}; a plan that skips a kind is not understood, so the block stops"
      break
    fi
    if [ "$scored" -eq 0 ]; then
      say "$name plan $chunk exited $rc before scoring any snapshot (see $log); not retried: its startup failed, and a retry would pay it again"
      break
    fi
    say "$name plan $chunk exited $rc after $scored of ${#this[@]} snapshots: step ${unscored[0]} is the one it was scoring and is left NOT RUN (see $log); the rest go back on the queue"
    pending=("${unscored[@]:1}" "${pending[@]}")
  done
  for s in $order; do
    [ -e "$out/traj-s$seed-step$s.jsonl" ] || skipped+=("$s")
  done
  if [ "${#skipped[@]}" -gt 0 ]; then
    say "$name: steps ${skipped[*]} NOT RUN (the ${cap}s cap, or a plan that stopped before them; see the plan logs)"
  fi
  return 0
}
