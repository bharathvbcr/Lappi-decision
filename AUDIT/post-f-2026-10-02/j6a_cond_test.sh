#!/bin/bash
# Throwaway: exercise box_q_j6a.sh's fsucc run-condition block (lines 82-106) against fixtures.
set -u
SRC=/Users/bharath/Code/research/Lappi-decision/campaign/post-f-queue/box_q_j6a.sh
T=/private/tmp/claude-501/-Users-bharath-Code-research-Lappi-decision/752954f4-7f08-4414-bdac-8e49ae47f8b4/scratchpad/j6a-cond
rm -rf "$T"; mkdir -p "$T"
BLOCK="$T/block.sh"
if [ "${OLD:-0}" = 1 ]; then
  git -C /Users/bharath/Code/research/Lappi-decision show HEAD:campaign/post-f-queue/box_q_j6a.sh | sed -n '78,83p' > "$BLOCK"
else
  sed -n '82,106p' "$SRC" > "$BLOCK"
fi
head -1 "$BLOCK"; tail -1 "$BLOCK"
pass=0; fail=0
run_case() {  # name expect(run|defer) word pin json
  local name=$1 expect=$2 word=$3 pin=$4 json=$5
  local d="$T/$name"; mkdir -p "$d/q" "$d/dec"
  [ -n "$word" ] && printf '%s' "$word" > "$d/q/fsucc.word"
  [ -n "$pin" ] && printf '%s' "$pin" > "$d/q/j6a-on-room-refusal-yes"
  [ -n "$json" ] && printf '%s' "$json" > "$d/dec/fsucc-successor-20261005T000000000000000.json"
  local out rc
  out=$(Q="$d/q" IDLE_DEC="$d/dec" PY=python3 SUCC_WORDS="fires:j6f fires:j6dv4 quiet refused" SUCC_WORD_FILE="$d/q/fsucc.word" UNSET_PINS="" bash -c '
read_succ_word() { local w; [ -f "$SUCC_WORD_FILE" ] || return 0; w=$(head -c 64 "$SUCC_WORD_FILE" | tr -d "\n"); case " $SUCC_WORDS " in *" $w "*) echo "$w" ;; esac; }
say() { echo "=== $*"; }
source "$1"
echo REACHED_RUN' _ "$BLOCK" 2>&1); rc=$?
  local got=defer; echo "$out" | grep -q REACHED_RUN && got=run
  if [ "$got" = "$expect" ]; then pass=$((pass+1)); echo "PASS $name ($got)"; else fail=$((fail+1)); echo "FAIL $name: expected $expect got $got"; echo "$out" | sed 's/^/    /'; fi
}
C_ONLY='{"decision":"refused","detail":{"arms":{"j6f":{},"j6dv4":{}},"refused_because":["(c) j6f needle_8k_worst_bucket has no room"]}}'
C_AND_B='{"decision":"refused","detail":{"arms":{"j6f":{}},"refused_because":["(c) j6f no room","(b) row x missing"]}}'
C_NO_ARMS='{"decision":"refused","detail":{"refused_because":["(c) j6f no room","(b) arm rows unreadable"]}}'
C_ARMS_NOT_DICT='{"decision":"refused","detail":{"arms":[],"refused_because":["(c) j6f no room"]}}'
A_ONLY='{"decision":"refused","detail":{"arms":{},"refused_because":["(a) both arms win"]}}'
EMPTY_REASONS='{"decision":"refused","detail":{"arms":{},"refused_because":[]}}'
NO_DETAIL='{"decision":"refused","refused":"(b) envelope unreadable"}'
run_case quiet run quiet "" ""
run_case quiet_no_pin_needed run quiet "" "$C_ONLY"
run_case room_only_with_pin run refused "Bharath: yes, 2026-10-02" "$C_ONLY"
run_case room_only_no_pin defer refused "" "$C_ONLY"
run_case room_only_empty_pin defer refused "" "$C_ONLY"
run_case room_and_b defer refused "yes" "$C_AND_B"
run_case room_no_arms defer refused "yes" "$C_NO_ARMS"
run_case arms_not_object defer refused "yes" "$C_ARMS_NOT_DICT"
run_case both_win_a defer refused "yes" "$A_ONLY"
run_case empty_reasons defer refused "yes" "$EMPTY_REASONS"
run_case no_detail defer refused "yes" "$NO_DETAIL"
run_case refused_no_json defer refused "yes" ""
run_case fires_j6f defer fires:j6f "yes" "$C_ONLY"
run_case fires_j6dv4 defer fires:j6dv4 "yes" "$C_ONLY"
run_case absent_word defer "" "yes" "$C_ONLY"
run_case malformed_word defer "maybe" "yes" "$C_ONLY"
echo "pass=$pass fail=$fail"
[ $fail -eq 0 ]
