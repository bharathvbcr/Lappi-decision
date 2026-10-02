# shellcheck shell=bash
# Sourced, after post_f_common.sh, by the idle-GPU queue's waiters (Fable's idle-queue ruling,
# AUDIT/idle-gpu-queue-2026-10-02/fable-idle-queue.md Q1/Q2; pre-registration
# campaign/f-successor-preregistered.json). Defines no job and starts nothing.
#
# Chain (Fable Q2): rung0.done -> cudadev -> rungd -> fsucc -> j5pp -> j6a -> STOP. Nothing goes
# after j6a. j6ctl (CPU only) waits on j6f.done and j6dv4.done and runs beside them; fsucc also
# waits on j6ctl.done and on F's controls ('=== F controls all done' in q-fcontrols.log).
#
# Markers, locks and pins follow post_f_common.sh: <job>.queued / .started / .done in $Q, an EXIT
# trap that touches .done on every path, GPU work under $Q/gpu.lock, binaries pinned by sha256.
# One deliberate difference: fsucc's word is NOT read through post_f_common.sh's rule(), which
# maps every word but fires|quiet|qualifies|fails to refused and would turn fires:j6f into a
# refusal (campaign/f-successor-preregistered.json outcomes.waiter_note). succ_word() below
# accepts exactly the pre-registration's four words.

# The successor checker: L-room's build (HANDOFF/succ-room-2026-10-02.md, main 28000af), copied
# as is to a NEW path. /home/ubuntu/bin/qd-post-f-rules and RULES_SHA256 (cadbdc74...) are not
# touched; $RULES stays the binary for ft-rows / eval-row.
RULES_SUCC=/home/ubuntu/bin/qd-post-f-rules-succ
RULES_SUCC_SHA256=be02029779eca1510c75f24e22c23cf7d762f1d87bbbca08b238ff9aede5eb62

# Decisions of this queue (new directory; post-F's $DEC is not written).
IDLE_DEC=/home/ubuntu/ledger/idle-decisions-2026-10-02
# fsucc's word, written once, atomically, and read by j5pp and j6a.
SUCC_WORD_FILE=$Q/fsucc.word
# The four words of outcomes.words, verbatim.
SUCC_WORDS="fires:j6f fires:j6dv4 quiet refused"

# F's control engine and lane (box_f_controls.sh, verbatim): qd-lane10 at 3460afc, qd-prep-m1.
PREP_M1=/home/ubuntu/bin/qd-prep-m1
PREP_M1_SHA256=9ccaf5aad5d39a187d547f9b45c1ff37b1d416e30aa0ab7788e6a96c5ac536b9
# box_f_controls.sh's SPLIT, verbatim (no --out, no backbone): the letter-control row's recipe
# must equal c89b89a1's in every key but eval_row_id, or successor refuses on (b).
CTL_SPLIT=(--no-repo-history --rev 881ab304f15ea13529002391dda8520c2ea47af4
           --defect-class data/pool/commitpackft-composed-v1 --defect-download data/pool/commitpackft
           --defect-noul data/pool/defect-noul-v3b --general-record "$REC" --general-max-rows 200000)
# The fetch record F read (box_q_f.sh's REC; it is a real path on the box), pinned.
REC_SHA256=a0841f0d9990bbc0fa8c46ad559e0c33d2f1bedbd2b5e9e30454d4c1d6ba5555
# J6(f)-v4 and J6(d)-v4 (post-F items 6 and 10): their output dirs and training logs.
J6F_OUT=/home/ubuntu/j6f-v4
J6DV4_OUT=/home/ubuntu/j6d-v4
# F's controls waiter's log and its last line.
FCTL_LOG=/home/ubuntu/logs/q-fcontrols.log
FCTL_DONE='=== F controls all done'

# F' (f_prime_recipe): per arm, the recipe flags, checkpoint dir, output dir and ledger, verbatim
# from the pre-registration. Each is F's argv (F_RECIPE) with the arm's one change.
FP_J6F_RECIPE=(--optimizer master --lr 1e-5 --epoch --no-memorise --batch-tokens 35403
               --checkpoint-skip-layers "$F_SKIP")
FP_J6DV4_RECIPE=(--optimizer master --lr 3e-5 --beta2 0.95 --epoch --no-memorise --batch-tokens 35403
                 --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers "$F_SKIP")

# fp_arm WORD: sets FP_ARM, FP_RECIPE, FP_CKPT, FP_OUT, FP_LEDGER for a firing word; returns 3
# for any other word, so nothing defaults to an arm.
fp_arm() {
  case "$1" in
    fires:j6f)
      FP_ARM=j6f; FP_RECIPE=("${FP_J6F_RECIPE[@]}")
      FP_CKPT=/home/ubuntu/ckpt/fsucc-j6f; FP_OUT=/home/ubuntu/fsucc-j6f
      FP_LEDGER=/home/ubuntu/ledger/gh200-fsucc-j6f-2026-10-02.jsonl ;;
    fires:j6dv4)
      FP_ARM=j6dv4; FP_RECIPE=("${FP_J6DV4_RECIPE[@]}")
      FP_CKPT=/home/ubuntu/ckpt/fsucc-j6dv4; FP_OUT=/home/ubuntu/fsucc-j6dv4
      FP_LEDGER=/home/ubuntu/ledger/gh200-fsucc-j6dv4-2026-10-02.jsonl ;;
    *) return 3 ;;
  esac
}

# The 'ft row <uuid>' a training log names (box_q_f.sh's own extraction), or nothing.
log_ft_row() {
  grep -oE 'ft row [0-9a-f-]{36}' "$1" 2>/dev/null | tail -1 | cut -d' ' -f3
}

# cd into a checkout and refuse unless it is clean and at the FULL 40-character commit given
# (post_f_common.sh's lane() checks 7 characters; an overlay is pinned to the whole sha).
lane_full() {
  cd "$1" || return 3
  if [ -n "$(git status --porcelain)" ]; then say "$1 is dirty; refusing"; git status --porcelain | head -20; return 3; fi
  if [ "$(git rev-parse HEAD)" != "$2" ]; then say "$1 is at $(git rev-parse HEAD), not $2; refusing"; return 3; fi
}

# The word fsucc wrote, if it is one of the four, else nothing. Absent or malformed reads as
# nothing, and every caller treats nothing as "do not run".
read_succ_word() {
  local w
  [ -f "$SUCC_WORD_FILE" ] || return 0
  w=$(head -c 64 "$SUCC_WORD_FILE" | tr -d '\n')
  case " $SUCC_WORDS " in *" $w "*) echo "$w" ;; esac
}

# One letter control then the report-only option control, box_f_controls.sh's two blocks, from
# the current directory (qd-lane10 at 3460afc, checked by the caller), niced, no lock.
#   controls_block LABEL LEDGER VERDICTS GUARD
# GUARD is a marker touched before the letter control and refused if it already exists:
# successor refuses two letter-control rows for one eval row, and ft_linear_control.py has no
# duplicate check of its own. The exit codes are logged, never branched on: F seed 0's letter
# control exited 3 (its gate not_run) and still wrote c89b89a1; successor decides whether the
# row is there.
controls_block() {
  local label=$1 ledger=$2 verdicts=$3 guard=$4
  if [ ! -s "$verdicts" ]; then say "$label: no verdicts at $verdicts; controls NOT RUN"; return 0; fi
  if [ -e "$guard" ]; then
    say "$label: $guard exists, so a letter control was already started for these verdicts; a second row would make successor refuse; controls NOT RUN again"
    return 3
  fi
  touch "$guard"
  say "$label letter control at $(git rev-parse --short HEAD) -> $ledger"
  nice -n 10 "$PY" -u tools/ft_linear_control.py --ledger "$ledger" --verdicts "$verdicts" "${CTL_SPLIT[@]}"
  say "$label letter control done (exit $?; 3 with a row written is F seed 0's c89b89a1 case)"
  nice -n 10 "$PY" -u tools/ft_linear_control.py --ledger "$ledger" --verdicts "$verdicts" "${CTL_SPLIT[@]}" --option-control
  say "$label option control done (exit $?)"
}

# Write a small file atomically (tmp + rename in the same directory).
write_atomic() {
  local path=$1 body=$2
  printf '%s\n' "$body" > "$path.tmp.$$" && mv -f "$path.tmp.$$" "$path"
}
