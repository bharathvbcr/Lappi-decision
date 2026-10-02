# shellcheck shell=bash
# Sourced by every campaign/post-f-queue/box_q_*.sh on the box: the paths, pins and helpers the
# queue after run F shares (Fable's post-F ruling, campaign/f-j7prime-preregistered.json at
# 4fd08cf, with its avg-np reading at 54512e6; its queue is items 0-10). Defines no job and starts
# nothing.
#
# Decisions are never made here or in a script: the three pre-registered rules are
# qd-post-f-rules subcommands (crates/qd-runtime/src/bin/qd_post_f_rules.rs), pinned by sha256.
# Each call writes its JSON to $DEC (beside the ledgers) and to the job's log, and the script
# branches on the one word it prints. A refusal is never a branch taken quietly: every script
# logs it and says what it then does not run.
#
# Box convention (box_q_f.sh, box_q_j4ens3.sh, box_q_avgnp3.sh): markers <job>.queued /
# .started / .done in /home/ubuntu/queue; an EXIT trap touches .done on every path; GPU work
# holds /home/ubuntu/queue/gpu.lock; lanes are pinned clean clones.

Q=/home/ubuntu/queue
PF=/home/ubuntu/post-f
DEC=/home/ubuntu/ledger/post-f-decisions-2026-10-01
RULES=/home/ubuntu/bin/qd-post-f-rules
# Cross-built on the Mac from this branch (HANDOFF/post-f-queue-2026-10-01.md, "The binary").
RULES_SHA256=cadbdc74eabdc26a2395bd31cb20eab35152c958025af46ba50ef30365111339
# AUDIT/j7-avg-ood-diag-2026-10-01/delta_cosine.py at this branch, copied to $PF.
DELTA_COSINE=$PF/delta_cosine.py
DELTA_COSINE_SHA256=f4e373a94c87c54759a4a4e382de1349010e18c3d0b646a4c9f376b8dca34961
PY=/home/ubuntu/qd-venv/bin/python
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
TOKENIZER_SHA256=fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927

# F (box_q_f.sh): its ledger, logs ('ft row <uuid>' in train-s<N>.log), checkpoints and verdicts.
F_LEDGER=/home/ubuntu/ledger/gh200-p4-v4-2026-10-01.jsonl
F_OUT=/home/ubuntu/p4-v4
F_CKPT=/home/ubuntu/ckpt/p4-v4
# F ran with --checkpoint-skip-layers 6 (f.ckpt-skip, read once at seed 0). Every run on F's
# exact path carries the same value, and refuses if the file F read now says otherwise.
F_SKIP=6
# J4's rows that rule (i) reads: J4's ledger (references f3612f73, 2f5fe57a) and J7's (the
# avg-np rows items 1-2 write, references 1d93b3ee, 0b86fae3, 6fcba23d).
J4_LEDGER=/home/ubuntu/ledger/gh200-p4-v3-2026-10-01.jsonl
J7_LEDGER=/home/ubuntu/ledger/gh200-p6-j7-avg-2026-10-01.jsonl
# New ledgers for this queue's rows.
J7P_LEDGER=/home/ubuntu/ledger/gh200-p6-f-j7prime-2026-10-01.jsonl
SLICE_LEDGER=/home/ubuntu/ledger/gh200-f-composed-slice-2026-10-01.jsonl
J6V4_LEDGER=/home/ubuntu/ledger/gh200-j6-v4-ablations-2026-10-01.jsonl
CALIB_LEDGER=/home/ubuntu/ledger/gh200-calib-fit-2026-10-01.jsonl

# Lanes: F's exact path (training) and F's post-run lane (scoring; holds e242b94, 738159b, 0b95adb).
LANE8=/home/ubuntu/qd-lane8
LANE8_AT=a502670
LANE10=/home/ubuntu/qd-lane10
LANE10_AT=3460afc
export HF_HUB_OFFLINE=1 QD_PREP_BIN=/home/ubuntu/bin/qd-prep-m1

# F's data argv, verbatim from box_q_f.sh.
F_SPLIT=(--out /home/ubuntu/phase4-v4-2026-10-01 --no-repo-history
         --rev 881ab304f15ea13529002391dda8520c2ea47af4
         --defect-class data/pool/commitpackft-composed-v1
         --defect-download data/pool/commitpackft
         --defect-noul data/pool/defect-noul-v3b
         --general-record "$REC" --general-max-rows 200000
         --real-backbone "$BACKBONE")
# F's recipe flags, verbatim from box_q_f.sh, with the skip F ran with.
F_RECIPE=(--optimizer master --lr 1e-5 --epoch --no-memorise --batch-tokens 35403
          --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers "$F_SKIP")
COST=(--instance lambda-1xgh200 --usd-per-hour 2.29)
# Rule 4: a 32,400 s cap at $2.29/h is $20.61, over the $20 line. The human's yes (in chat,
# 2026-10-01) lifted that line for single-GPU runs; box_q_f.sh carries the same words.
HUMAN_YES="Bharath (human, in chat 2026-10-01: 'I am okay with run costing at least \$20 ... don't mind the \$20 budget'; lifts rule 4's \$20 bound for single-GPU runs)"
approved() { echo "$HUMAN_YES; $1 capped 32,400 s at \$2.29/h = \$20.61"; }

say() { echo "=== $* ($(date -u +%H:%M:%S))"; }

# The rules binary and the delta_cosine copy are the committed ones, or nothing runs on them.
pin() {
  local path=$1 want=$2 got
  got=$(sha256sum "$path" 2>/dev/null | cut -c1-64)
  if [ "$got" != "$want" ]; then say "$path is sha256 ${got:-absent}, not $want; refusing"; return 3; fi
}

# cd into a lane and refuse unless it is clean and at its pinned commit.
lane() {
  cd "$1" || return 3
  if [ -n "$(git status --porcelain)" ]; then say "$1 is dirty; refusing"; git status --porcelain; return 3; fi
  if [ "$(git rev-parse --short=7 HEAD)" != "$2" ]; then say "$1 is not at $2; refusing"; return 3; fi
}

# F's skip, as F read it. Refuses unless it is still the value F ran with.
f_skip_ok() {
  local n
  n=$(tr -cd '0-9' < $Q/f.ckpt-skip 2>/dev/null)
  if [ "$n" != "$F_SKIP" ]; then say "$Q/f.ckpt-skip says '${n}', not F's $F_SKIP; refusing"; return 3; fi
}

# The ft row id a seed's training log names (F's own extraction, box_q_f.sh), or nothing.
f_ft_row() {
  grep -oE 'ft row [0-9a-f-]{36}' "$F_OUT/train-s$1.log" 2>/dev/null | tail -1 | cut -d' ' -f3
}

# --ft-row SEED=ID for each seed given, ids from the logs. An empty id is a usage error in the
# binary, so a seed with no ft row refuses rather than drops out.
f_ft_args() {
  local s
  for s in "$@"; do printf -- '--ft-row\n%s=%s\n' "$s" "$(f_ft_row "$s")"; done
}

# rule NAME ITEM ARGS... : one pre-registered decision. The JSON goes to $DEC and to this
# job's log (the binary's stderr); the word (fires / quiet / qualifies / fails / refused) to
# stdout. Anything else the binary prints, or a binary that did not run, reads as refused.
rule() {
  local name=$1 item=$2 out word
  shift 2
  mkdir -p "$DEC"
  out=$DEC/$item-$name-$(date -u +%Y%m%dT%H%M%S%N).json
  word=$("$RULES" "$name" "$@" --out "$out")
  case "$word" in fires|quiet|qualifies|fails) ;; *) word=refused ;; esac
  say "rule $name ($item): $word; JSON in $out" >&2
  echo "$word"
}

# Rule (ii) on F's three seeds: the seed set F's J7' uses ("0 1 2" or "0 1 2 3 4"), or nothing
# when the rule refused.
f_seeds() {
  local args word
  mapfile -t args < <(f_ft_args 0 1 2)
  word=$(rule seeds34 "$1" --f-ledger "$F_LEDGER" "${args[@]}")
  case "$word" in fires) echo "0 1 2 3 4" ;; quiet) echo "0 1 2" ;; esac
}

# Rule (iii) on F's three seeds: fires / quiet / refused.
j6f_word() {
  local args
  mapfile -t args < <(f_ft_args 0 1 2)
  rule j6f "$1" --f-ledger "$F_LEDGER" "${args[@]}"
}

# The seeds' ft row ids, in order, checked by the binary as one averageable configuration
# (completed, not quick, tag epoch, the seed claimed, one recipe hash and data snapshot).
# Non-zero, with the reason in the log, when they are not.
f_ft_ids() {
  local args
  mapfile -t args < <(f_ft_args "$@")
  "$RULES" ft-rows --ledger "$F_LEDGER" "${args[@]}"
}

# The averages item 0 builds for a seed set ("0 1 2" -> ...seed012...).
avg_path() { echo "$F_CKPT/avg/epoch-avg-seed$(echo "$1" | tr -d ' ')-masters.safetensors"; }
avgnp_path() { echo "$F_CKPT/avg/epoch-avgnp-seed$(echo "$1" | tr -d ' ')-masters.safetensors"; }

# Wait for each named item that was queued to finish. An item never queued (its waiter not
# launched) is not waited on, so a missing waiter never stalls the chain. Every waiter touches
# its .queued first thing, and the launch loop starts them all together, hours before any of
# them reaches a wait, so "not queued" means "not launched".
wait_queued() {
  local n
  for n in "$@"; do
    while [ -f "$Q/$n.queued" ] && [ ! -f "$Q/$n.done" ]; do sleep 30; done
  done
}

# Items 9 and 7 follow J6(f) only when item 6 took the early slot. They read the position item 6
# wrote ($Q/j6f.position, decided once by rule (iii)), never the rule a second time.
after_early_j6f() {
  while [ -f "$Q/j6f.queued" ] && [ ! -f "$Q/j6f.position" ] && [ ! -f "$Q/j6f.done" ]; do sleep 30; done
  if [ "$(cat "$Q/j6f.position" 2>/dev/null)" = early ]; then
    say "$1: J6(f) took the early slot; waiting for it (j6f.done)"
    wait_queued j6f
  fi
}

# JSON helpers for --score-plan files (paths and ids hold no quote or backslash).
json_strs() { local o="" x; for x in "$@"; do o="$o${o:+, }\"$x\""; done; echo "[$o]"; }
json_ints() { local o="" x; for x in "$@"; do o="$o${o:+, }$x"; done; echo "[$o]"; }
