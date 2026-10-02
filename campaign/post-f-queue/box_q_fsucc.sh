#!/bin/bash
# Idle queue fsucc (Fable's idle-queue ruling, Q1 item 1 and Q2; pre-registration
# campaign/f-successor-preregistered.json): decide F's successor once, and on a firing arm train
# F' -- three seeds of that arm's recipe (f_prime_recipe), not quick (rule 8).
#   1. qd-post-f-rules-succ successor (L-room's build, sha256 pinned in idle_common.sh) on F's
#      three seeds and the J6 v4 arms, with the pre-registration's argv. Its stdout word is read
#      against exactly fires:j6f | fires:j6dv4 | quiet (exit 0) and refused (exit 3). Any other
#      word, exit code or pairing is refused, logged as inconsistent: a refusal is never read as
#      quiet. Not post_f_common.sh's rule(), which maps fires:* to refused (waiter_note).
#      The word goes to $Q/fsucc.word (atomic, once; a second decision is refused). The JSON goes
#      to $IDLE_DEC and this log; the reasons are logged from detail.refused_because as a LIST
#      (L-room's shape), or the top-level 'refused' when F's envelope did not resolve (no detail).
#   2. quiet / refused: no F'; the word and the JSON path are logged (Fable Q2: re-plan).
#   3. fires:<arm>: under gpu.lock, from qd-lane8 at a502670 (clean) with F's skip, per seed 0 1 2:
#      training (cap 32,400 s, $20.61, the human's 2026-10-01 yes via approved()), then the
#      --needle-control 1024,2048,4096 block on its checkpoint (cap 5,400 s), as box_q_f.sh.
#      Then the lock is released and $Q/fsuccgpu.done touched, and box_f_controls.sh's two CPU
#      blocks run per seed from qd-lane10 at 3460afc, niced, into the F' ledger.
#      Deviation from the pre-registration's per-seed order, scheduling only (no argv changes):
#      the CPU controls run after the three seeds' GPU work, outside the lock, so j5pp (which
#      waits on fsuccgpu.done and reads only the ft and eval rows) is not held for ~1 h of CPU work.
# Caps: 3 x 32,400 s + 3 x 5,400 s = 113,400 s = $72.14 at $2.29/h (pre-registration caps).
# Queue: after rungd.done, j6ctl.done and '=== F controls all done' in q-fcontrols.log. The last
# is waited on for at most FCTL_WAIT_MAX_S once the first two exist, then refused.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
# Bound on the wait for F's controls line once rungd and j6ctl are done. F's last letter control
# lands minutes after F's last seed (~18 min per seed on the box, q-fcontrols.log), hours before
# rungd ends; if the line is still absent this long after, box_f_controls.sh has died, its rows
# will not come, and waiting on would only idle the GPU. Then: refused, never quiet.
FCTL_WAIT_MAX_S=7200
trap 'touch /home/ubuntu/queue/fsuccgpu.done /home/ubuntu/queue/fsucc.done' EXIT
touch $Q/fsucc.queued
if [ -e "$SUCC_WORD_FILE" ]; then
  say "fsucc: $SUCC_WORD_FILE already exists ('$(head -c 64 "$SUCC_WORD_FILE" | tr -d '\n')'): the successor was decided once; not deciding again"
  exit 3
fi
until [ -f $Q/rungd.done ] && [ -f $Q/j6ctl.done ]; do sleep 60; done
FCTL_T0=$(date +%s)
until grep -qxF "$FCTL_DONE" "$FCTL_LOG" 2>/dev/null; do
  if [ $(( $(date +%s) - FCTL_T0 )) -ge "$FCTL_WAIT_MAX_S" ]; then
    say "fsucc: refused: '$FCTL_DONE' is not in $FCTL_LOG ${FCTL_WAIT_MAX_S}s after rungd and j6ctl finished (F's letter-control rows are rows successor reads); successor NOT RUN"
    write_atomic "$SUCC_WORD_FILE" refused
    exit 3
  fi
  sleep 60
done
touch $Q/fsucc.started
mkdir -p "$IDLE_DEC" || exit 3

# --- 1. the decision ------------------------------------------------------------------------
WORD=refused
OUTJ=$IDLE_DEC/fsucc-successor-$(date -u +%Y%m%dT%H%M%S%N).json
F0=$(f_ft_row 0); F1=$(f_ft_row 1); F2=$(f_ft_row 2)
AF=$(log_ft_row "$J6F_OUT/train.log"); AD=$(log_ft_row "$J6DV4_OUT/train.log")
MISSING=""
[ -n "$F0" ] || MISSING="$MISSING F seed 0 ($F_OUT/train-s0.log)"
[ -n "$F1" ] || MISSING="$MISSING F seed 1 ($F_OUT/train-s1.log)"
[ -n "$F2" ] || MISSING="$MISSING F seed 2 ($F_OUT/train-s2.log)"
[ -n "$AF" ] || MISSING="$MISSING J6(f)-v4 ($J6F_OUT/train.log)"
[ -n "$AD" ] || MISSING="$MISSING J6(d)-v4 ($J6DV4_OUT/train.log)"
if ! pin "$RULES_SUCC" "$RULES_SUCC_SHA256"; then
  say "fsucc: refused: the successor binary is not the pinned one; NOT RUN"
elif [ -n "$MISSING" ]; then
  say "fsucc: refused: no 'ft row' id in the log of:$MISSING; successor NOT RUN"
else
  say "fsucc: successor on F ft rows $F0 $F1 $F2, J6(f)-v4 $AF, J6(d)-v4 $AD"
  RAW=$("$RULES_SUCC" successor --f-ledger "$F_LEDGER" \
    --ft-row "0=$F0" --ft-row "1=$F1" --ft-row "2=$F2" \
    --arm-ledger "$J6V4_LEDGER" --j6f-ft-row "0=$AF" --j6dv4-ft-row "0=$AD" \
    --out "$OUTJ")
  RC=$?
  case "$RC:$RAW" in
    0:fires:j6f|0:fires:j6dv4|0:quiet|3:refused) WORD=$RAW ;;
    *) say "fsucc: refused: inconsistent checker output (exit $RC, stdout '$(printf '%s' "$RAW" | head -c 200)'); never read as quiet" ;;
  esac
  # The checker writes its JSON to --out in every case; a word without it is not one to act on.
  if [ "$WORD" != refused ] && [ ! -s "$OUTJ" ]; then
    say "fsucc: refused: the checker said '$WORD' but wrote no JSON to $OUTJ"
    WORD=refused
  fi
  "$PY" -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as exc:
    print(f"fsucc: decision JSON {sys.argv[1]} unreadable ({exc}); the word above stands")
    sys.exit(0)
det = d.get("detail")
top = d.get("refused")
if not isinstance(det, dict):
    print(f"fsucc: no detail (F envelope did not resolve); refused = {top!r}")
    sys.exit(0)
rb = det.get("refused_because")
if isinstance(rb, list):
    for r in rb:
        print(f"fsucc reason: {r}")
elif rb is not None:
    print(f"fsucc: detail.refused_because is a {type(rb).__name__}, not a list: logged, not parsed: {rb!r}")
cc = det.get("cannot_clear")
print(f"fsucc cannot_clear: {json.dumps(cc)}")
' "$OUTJ"
fi
write_atomic "$SUCC_WORD_FILE" "$WORD" || { say "fsucc: could not write $SUCC_WORD_FILE"; exit 3; }
if [ -f "$OUTJ" ]; then JNOTE="JSON $OUTJ"; else JNOTE="no JSON: successor not run"; fi
say "fsucc: successor word '$WORD' ($JNOTE; word in $SUCC_WORD_FILE)"
case "$WORD" in
  quiet)   say "fsucc: quiet: no F' (Fable Q2: re-plan from J7' avg / ens3 / avg-np)"; exit 0 ;;
  refused) say "fsucc: REFUSED: no F'; the human decides (a refusal is never quiet)"; exit 3 ;;
esac

# --- 3. F' on the firing arm ----------------------------------------------------------------
fp_arm "$WORD" || { say "fsucc: '$WORD' names no arm; no F'"; exit 3; }
exec 9>$Q/gpu.lock
flock 9
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
mkdir -p "$FP_OUT" "$FP_CKPT"
say "fsucc: F' on $FP_ARM at $(git rev-parse --short HEAD): ${FP_RECIPE[*]} -> $FP_LEDGER"
for SEED in 0 1 2; do
  if [ -e "$FP_OUT/train-s$SEED.log" ]; then say "F' seed $SEED: $FP_OUT/train-s$SEED.log exists; refusing to run it twice"; continue; fi
  timeout 34200 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" "${FP_RECIPE[@]}" --devices cuda --seeds $SEED \
    --checkpoint-dir "$FP_CKPT" --checkpoint-every 100000 \
    --score-val --needle --ood --ood-general-record "$REC" \
    --verdicts-out "$FP_OUT/verdicts-s$SEED.jsonl" --suite-verdicts-out "$FP_OUT/suite-verdicts-s$SEED.jsonl" \
    --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$FP_LEDGER" \
    --approved-by "$(approved "F' ($FP_ARM) seed $SEED (idle queue fsucc)")" \
    2>&1 | tee "$FP_OUT/train-s$SEED.log"
  say "F' seed $SEED train+score done (exit $?)"
  FT_ROW=$(log_ft_row "$FP_OUT/train-s$SEED.log")
  if [ -z "$FT_ROW" ] || [ ! -f "$FP_CKPT/epoch-seed$SEED-cuda.json" ]; then
    say "F' seed $SEED: no ft row or checkpoint from this run; its needle control NOT RUN"
    continue
  fi
  timeout 5400 "$PY" -u tools/real_ft_run.py "${F_SPLIT[@]}" --devices cuda --seeds $SEED \
    --score-val --needle --needle-control 1024,2048,4096 \
    --score-checkpoint "$FP_CKPT/epoch-seed$SEED-cuda.json" --score-dtype fp32 \
    --ft-ledger "$FP_LEDGER" --ft-row-id "$FT_ROW" \
    "${COST[@]}" --wall-clock-cap-s 5400 \
    --ledger "$FP_LEDGER" --suite-verdicts-out "$FP_OUT/needle-control-s$SEED.jsonl" \
    2>&1 | tee "$FP_OUT/needle-control-s$SEED.log"
  say "F' seed $SEED needle control done (exit $?)"
done
exec 9>&-
touch $Q/fsuccgpu.done
say "fsucc: F' GPU part done; gpu.lock released; fsuccgpu.done touched (j5pp may start)"

# --- the CPU controls, outside the lock ------------------------------------------------------
pin "$PREP_M1" "$PREP_M1_SHA256" || exit 3
lane "$LANE10" "$LANE10_AT" || exit 3
for SEED in 0 1 2; do
  controls_block "F' ($FP_ARM) seed $SEED" "$FP_LEDGER" "$FP_OUT/verdicts-s$SEED.jsonl" "$Q/fsucc-s$SEED.letter.started"
done
say "fsucc all done"
