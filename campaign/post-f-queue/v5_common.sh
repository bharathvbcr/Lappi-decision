# shellcheck shell=bash
# shellcheck disable=SC2034  # the names below are read by the waiters that source this file
# Sourced, after post_f_common.sh and idle_common.sh, by the v5 block's waiters: the two GPU lanes
# (box_q_v5.sh 0 and box_q_v5.sh 1) and the per-seed post-seed waiter box_q_v5traj.sh. Defines
# no job and starts nothing. post_f_common.sh and idle_common.sh are sourced for their helpers
# (say, pin, lane_full, log_ft_row, write_atomic, controls_block); their GH200 values (COST,
# gpu.lock, the post-F chain) are never used here.
#
# Binding text: campaign/v5-preregistered.json, the DRAFT (campaign/v5-preregistered.DRAFT.json,
# with the 2x H100 amendment applied at a29bca1) renamed on main without its top-level `draft`
# key. This file reads only the renamed one, from the pinned v5 lane; the rules binary refuses a
# file that still says draft.
#
# The box: one Lambda 2x H100 80 GB SXM5 instance, two GPU lanes (hardware block). The order and
# the scheduling rule are box_q_v5.sh's header. Every GPU step of a lane holds $Q/gpu<N>.lock
# and runs with CUDA_VISIBLE_DEVICES=<N>; a seed's post-seed waiter takes the same lane's lock.
#
# Decisions are never made here: v5-pause (R9), v5-noulw --room and v5-noulw are qd-post-f-rules
# subcommands (HANDOFF/v5-rules-2026-10-02.md), run from a NEW binary pinned by sha256 and
# verified before EVERY call. seeds34 is not read (seeds 3-4 are unconditional since the
# amendment). post_f_common.sh's rule() is not used: it runs the old cadbdc74 build, which has
# none of these subcommands, and maps continue/pause/room/no_room/wins to refused.
#
# THE UNSET VALUES BELOW ARE FAIL-CLOSED PINS, NOT PLACEHOLDERS. While any is the literal UNSET,
# every v5 waiter logs "deferred: pins UNSET" and exits 3 without touching a lock. The lead fills
# them at deploy time (HANDOFF/v5-2gpu-queue-2026-10-03.md, "Deploy"), stages the filled copy and
# records its sha256. Never edit this file while a waiter that sources it runs.

# The v5 rules build at a new path, built for the box's x86_64.
RULES_V5=/home/ubuntu/bin/qd-post-f-rules-v5
RULES_V5_SHA256=UNSET
# qd-prep at the v5 build commit. post_f_common.sh exports QD_PREP_BIN=qd-prep-m1 (F's); every v5
# split rebuild (real_ft_run.py, ft_linear_control.py) runs this one instead, re-exported only
# after its pin checks (v5_verify).
V5_PREP_BIN=/home/ubuntu/bin/qd-prep-v5
V5_PREP_BIN_SHA256=UNSET
# The v5 lane: a clean clone at the FULL commit on main that carries the v5 build commit and the
# renamed pre-registration. Training, scoring, controls and the rules' text all come from it.
V5_LANE=/home/ubuntu/qd-lane-v5
V5_LANE_AT=UNSET
V5_PREREG_SHA256=UNSET
# The date in the v5 ledgers' names, YYYY-MM-DD.
V5_DATE=UNSET
# The Mac prelude's record (v5_prelude_mac.py --record), copied by the lead, and its sha256 as
# the prelude printed it ("RECORD ... sha256").
V5_PRELUDE_SHA256=UNSET
# The box (the pre-registration's hardware block):
#   V5_BOX                    the ledger names' box, h100x2 (seeds.ledger and hardware.ledger
#                             name it; v5_read_prereg refuses another)
#   V5_INSTANCE               the price list's name for the instance, e.g. lambda-2xH100-80GB-SXM5
#   V5_USD_PER_INSTANCE_HOUR  the whole instance, both GPUs: 8.38
#   V5_USD_PER_GPU_HOUR       the per-GPU column: 4.19. A lane's run sees one GPU
#                             (CUDA_VISIBLE_DEVICES), so this is the rate it is priced at
#                             (v5_lane_set)
#   V5_R9                     keep | waive: R9's seed-0 pause (amendments_pending, the human's call)
#   V5_HUMAN_YES              the human's yes on this plan (11 runs on this box), verbatim. Every
#                             v5 row's --approved-by and the probe's cite it; the d24c865 yes was
#                             on ~$137 / ~60 GPU-h at the GH200's rate and does not cover it.
V5_BOX=UNSET
V5_INSTANCE=UNSET
V5_USD_PER_INSTANCE_HOUR=UNSET
V5_USD_PER_GPU_HOUR=UNSET
V5_R9=UNSET
V5_HUMAN_YES=UNSET
# v5's data argv for tools/real_ft_run.py, verbatim from the v5 build's ledger row
# (ledger/mac-v5-shards-<date>.jsonl), with the box's paths: --out, --rev, the corpora,
# --general-record "$REC", --exclude-identity-keys, --real-backbone "$BACKBONE". Validated by
# v5_split_check. Fill it as one bash array: V5_SPLIT=(--out /home/ubuntu/... --no-repo-history ...).
V5_SPLIT=(UNSET)
# The conditionals as decided, each from its deciding rows (recipe.conditionals; the decisions
# and their row ids go into amendments_pending before the rename):
#   V5_C1     on|off   C1 --option-permutation-seed 20260919 (qd-post-f-rules j6g printed wins)
#   V5_C2A    on|off   C2a --train-attention-mask none
#   V5_C2B    on|off   C2b --fused-adamw
#   V5_LOWER  keep|drop  --lower-layers-n 8 --lower-layers-lr-scale 0.1 (C3 fires:j6f or R4: drop)
#   V5_LRSET  f|j6dv4  --lr 1e-5 (F) or --lr 3e-5 --beta2 0.95 (C3 fires:j6dv4, or the human's
#                      advance answer 3 at 0b559bb on fsucc refused with j6dv4.wins true)
V5_C1=UNSET
V5_C2A=UNSET
V5_C2B=UNSET
V5_LOWER=UNSET
V5_LRSET=UNSET

# --- fixed by the pre-registration ----------------------------------------------------------
V5_PREREG=$V5_LANE/campaign/v5-preregistered.json
NOUL_PREREG=$V5_LANE/campaign/v4-noul-v3b-preregistered.json
V5_PRELUDE=$PF/v5-prelude-record.json
V5_LEDGER=/home/ubuntu/ledger/$V5_BOX-v5-$V5_DATE.jsonl
V5NW_LEDGER=/home/ubuntu/ledger/$V5_BOX-v5-noulw-$V5_DATE.jsonl
# The probe's row goes here, never into v5's ledger: the readings read v5's ledger, and a quick
# probe row there is noise they must not see (hardware.ledger).
V5_PROBE_LEDGER=/home/ubuntu/ledger/$V5_BOX-v5-probe-$V5_DATE.jsonl
DEC_V5=/home/ubuntu/ledger/v5-decisions-$V5_DATE
V5_OUT=/home/ubuntu/v5
V5_CKPT=/home/ubuntu/ckpt/v5
V5NW_OUT=/home/ubuntu/v5-noulw
V5NW_CKPT=/home/ubuntu/ckpt/v5-noulw
V5_LOGS=/home/ubuntu/logs
# The box has two GPUs, one lane each (hardware.lanes).
V5_GPUS=2
# The memory probe (hardware.probe): --probe-shapes at Fable's 12 GiB margin (margin_gib; read
# back from the pre-registration by v5_read_prereg, which refuses another value), cap 1,800 s.
V5_PROBE_MARGIN_GIB=12
V5_PROBE_CAP_S=1800
# Markers. The human's: V5_LAUNCH_YES (the launch), V5_CONTINUE / V5_STOP (R9's hold; V5_STOP
# also ends every lane), V5_PROBE_YES (go on after a probe that failed twice), V5NW_HUMAN_YES (the
# arm on no_room, R7), V5_OVER_BUDGET_YES (a run past the approved total). Each is a file the
# lead writes with the human's words; an empty file is not a yes. The lanes', each written once:
# v5.probe-passed / v5.probe-failed (the probe's outcome), v5r9.word (R9's word, R9 kept),
# v5.paused (R9 held, and why), v5nw.room (the room reading's word), v5nw.launch (run or skip:
# the arm's launch, decided with the room), v5nw.word (the arm's reading). Per job:
# v5job-<job>.claimed (the lane that took it), .done (its result: "ft <row>", "no-ft",
# "ran exit <rc>", "skipped: ...", "refused: ..."), .spend (its seconds, as v5.spend holds them).
V5_LAUNCH_YES=$Q/V5_LAUNCH_YES
V5_CONTINUE=$Q/V5_CONTINUE
V5_STOP=$Q/V5_STOP
V5_PROBE_YES=$Q/V5_PROBE_YES
V5NW_HUMAN_YES=$Q/V5NW_HUMAN_YES
V5_OVER_BUDGET_YES=$Q/V5_OVER_BUDGET_YES
V5_PROBE_PASSED=$Q/v5.probe-passed
V5_PROBE_FAILED=$Q/v5.probe-failed
V5R9_WORD=$Q/v5r9.word
V5_PAUSED=$Q/v5.paused
V5NW_ROOM=$Q/v5nw.room
V5NW_LAUNCH=$Q/v5nw.launch
V5NW_WORD=$Q/v5nw.word
# The lanes' scheduling mutex (fd 8): a pick and the decisions before it are made under it.
V5_SCHED_LOCK=$Q/v5.sched.lock
# One line per GPU step: UTC, label, wall seconds, USD at the per-GPU rate. Appended only.
V5_SPEND=$Q/v5.spend
# The human's order (hardware.lanes; the amended projected_gpu_hours.slot): v5 seeds 0-4, the
# noul-weight arm x3 (seeds 0-2) iff room, J5' x3 (seeds 0-2).
V5_JOBS="v5-s0 v5-s1 v5-s2 v5-s3 v5-s4 v5nw-s0 v5nw-s1 v5nw-s2 j5-s0 j5-s1 j5-s2"
# Caps (recipe.base): training --wall-clock-cap-s 32400 under timeout +1,800 s as F; the needle
# control cap 5,400 s; the trajectory waiter cap 3,600 s per seed (recipe.added[0].scoring).
V5_TRAIN_CAP_S=32400
V5_NEEDLE_CAP_S=5400
V5_TRAJ_CAP_S=3600
# recipe.added[0]: --checkpoint-every 100000 (as F) plus --retain-tower-every 1000.
V5_CKPT_EVERY=100000
V5_RETAIN_EVERY=1000
# C1's seed (recipe.conditionals C1).
V5_C1_SEED=20260919
# Bounds on waits that are not a predecessor's marker: a human marker, a post-seed waiter's start
# once the parent released its lane's lock to it, and a lane with nothing it can take yet. The
# longest legitimate idle is one job on the other lane (32,400 + 1,800 + 5,400 + 3,600 s); twice
# that means a lane waits on one that has stopped, and it refuses rather than wait forever.
V5_HUMAN_WAIT_MAX_S=259200
V5_TRAJ_HANDOFF_MAX_S=900
V5_IDLE_MAX_S=86400
# A trajectory invocation is not started with less than this left of the seed's 3,600 s.
V5_TRAJ_MIN_LEFT_S=120
# The ledger read race (Fable's ruling B, 2026-10-02; v5_read_race): a reading the binary refused
# on a half-written LAST ledger line is made again, at most V5_READ_TRIES attempts in all (the
# first included), V5_READ_RETRY_S apart: at most 4 x 10 s = 40 s of waiting per reading. Then
# it is refused, as now. Never gated on a post-seed waiter's .done instead: that waiter releases
# its lane's lock before its CPU controls, so waiting on it would idle a GPU for up to 5,400 s.
V5_READ_TRIES=5
V5_READ_RETRY_S=10
# The human's words choosing the box, cited by every v5 row's --approved-by beside V5_HUMAN_YES.
V5_BOX_YES="Bharath (human, ~16:26Z 2026-10-03, AUDIT/finalize-2026-10-03/report-to-human-2026-10-03-pipeline.md item 1: 'Go with 2× H100 (Lambda)')"
# The split flags a v5 data argv may carry: data only, never a recipe, run or scoring flag.
V5_SPLIT_FLAGS=" --out --no-repo-history --rev --defect-class --defect-download --defect-max-rows --defect-noul --general-record --general-max-rows --exclude-identity-keys --real-backbone --commitpackft --max-pairs "
# Set by v5_lane_set: this process's lane (its GPU) and its cost argv. Empty until then, and
# v5_lock refuses without a lane.
V5_GPU=""
V5_COST=()
# The job a lane or post-seed waiter is running (v5_run_job, box_q_v5traj.sh): v5_spend_add
# charges its seconds to it as well.
V5_JOB=""
V5_JOB_RESULT=""
V5_IDLE_SINCE=""
V5_IDLE_WHY=""

# v5_approved LABEL [CAP_S]: what a run's --approved-by says. The human's yes on this plan (the
# pin), the human's words choosing the box, post_f_common.sh's 2026-10-01 lifting of rule 4's
# $20 line for single-GPU runs, and the run's cap priced at the per-GPU rate.
v5_approved() {
  local cap=${2:-$V5_TRAIN_CAP_S}
  echo "$V5_HUMAN_YES; $V5_BOX_YES; $HUMAN_YES; $1 capped $cap s at \$${V5_USD_PER_GPU_HOUR}/GPU-h = \$$(awk -v s="$cap" -v r="$V5_USD_PER_GPU_HOUR" 'BEGIN { printf "%.2f", s * r / 3600 }')"
}

# True iff $1 is a decimal number above zero.
v5_positive() {
  echo "$1" | grep -Eqx '[0-9]+(\.[0-9]+)?' && awk -v x="$1" 'BEGIN { exit !(x > 0) }'
}

# The names of the pins still UNSET (space-separated), or nothing.
v5_pins_unset() {
  local v out=""
  for v in RULES_V5_SHA256 V5_PREP_BIN_SHA256 V5_LANE_AT V5_PREREG_SHA256 V5_DATE \
           V5_PRELUDE_SHA256 V5_BOX V5_INSTANCE V5_USD_PER_INSTANCE_HOUR V5_USD_PER_GPU_HOUR \
           V5_R9 V5_HUMAN_YES V5_C1 V5_C2A V5_C2B V5_LOWER V5_LRSET; do
    [ "${!v}" = UNSET ] && out="$out $v"
  done
  if [ "${#V5_SPLIT[@]}" -eq 0 ] || [ "${V5_SPLIT[0]}" = UNSET ]; then out="$out V5_SPLIT"; fi
  echo "$out"
}

# Each filled pin has its form, and each decision one of its words. Prints one line per
# problem, or nothing.
v5_pins_check() {
  local v rates=1
  for v in RULES_V5_SHA256 V5_PREP_BIN_SHA256 V5_PREREG_SHA256 V5_PRELUDE_SHA256; do
    echo "${!v}" | grep -Eqx '[0-9a-f]{64}' || echo "$v is '${!v}', not 64 lower-case hex"
  done
  echo "$V5_LANE_AT" | grep -Eqx '[0-9a-f]{40}' || echo "V5_LANE_AT is '$V5_LANE_AT', not a full 40-hex commit"
  echo "$V5_DATE" | grep -Eqx '20[0-9]{2}-[01][0-9]-[0-3][0-9]' || echo "V5_DATE is '$V5_DATE', not YYYY-MM-DD"
  case "$V5_BOX" in
    gh200*) echo "V5_BOX is '$V5_BOX': v5 runs on the 2x H100 box, never the GH200" ;;
    *) echo "$V5_BOX" | grep -Eqx '[a-z0-9]+(-[a-z0-9]+)*' \
         || echo "V5_BOX is '$V5_BOX', not lower-case letters and digits joined by '-'" ;;
  esac
  echo "$V5_INSTANCE" | grep -Eqx '[A-Za-z0-9][A-Za-z0-9._-]*' \
    || echo "V5_INSTANCE is '$V5_INSTANCE', not a price list's instance name ([A-Za-z0-9][A-Za-z0-9._-]*)"
  for v in V5_USD_PER_INSTANCE_HOUR V5_USD_PER_GPU_HOUR; do
    if ! v5_positive "${!v}"; then echo "$v is '${!v}', not a positive rate"; rates=""; fi
  done
  # the two columns of one price list differ by exactly the GPU count; a swap is DESIGN-4's error
  if [ -n "$rates" ] && ! awk -v i="$V5_USD_PER_INSTANCE_HOUR" -v g="$V5_USD_PER_GPU_HOUR" -v n="$V5_GPUS" \
       'BEGIN { d = i - g * n; exit !(d <= 0.005 && d >= -0.005) }'; then
    echo "V5_USD_PER_INSTANCE_HOUR $V5_USD_PER_INSTANCE_HOUR is not V5_USD_PER_GPU_HOUR $V5_USD_PER_GPU_HOUR x $V5_GPUS GPUs: one came off the wrong column of the price list"
  fi
  case "$V5_R9" in keep|waive) ;; *) echo "V5_R9 is '$V5_R9', not keep or waive" ;; esac
  if [ -z "${V5_HUMAN_YES//[[:space:]]/}" ]; then echo "V5_HUMAN_YES is empty: the human's yes on this plan goes in it verbatim"; fi
  for v in V5_C1 V5_C2A V5_C2B; do
    case "${!v}" in on|off) ;; *) echo "$v is '${!v}', not on or off" ;; esac
  done
  case "$V5_LOWER" in keep|drop) ;; *) echo "V5_LOWER is '$V5_LOWER', not keep or drop" ;; esac
  case "$V5_LRSET" in f|j6dv4) ;; *) echo "V5_LRSET is '$V5_LRSET', not f or j6dv4" ;; esac
}

# v5_lane_set N: this process is lane N (0 or 1) of the box. Every GPU step it starts sees only
# GPU N (CUDA_VISIBLE_DEVICES, exported here once, so the probe, training, needle control and
# trajectory scoring all inherit it), holds $Q/gpu<N>.lock (v5_lock), and is priced as the tool
# prices one visible GPU. tools/run_cost.py counts the visible GPUs (torch.cuda.device_count()),
# and CostEstimate (python/qd_train/run_control.py) then refuses an instance rate beside a
# per-GPU one at one GPU ($8.38 with $4.19), while $8.38 alone would price each lane's run as the
# whole box. So both flags carry the per-GPU rate: that passes at one GPU and is refused at two,
# which makes the tool itself refuse, before the tower loads, a run that sees both GPUs.
v5_lane_set() {
  case "$1" in 0|1) ;; *) say "lane '$1' is not 0 or 1 (the box's two GPUs)"; return 3 ;; esac
  V5_GPU=$1
  export CUDA_VISIBLE_DEVICES=$1
  V5_COST=(--instance "$V5_INSTANCE:gpu$1" --usd-per-hour "$V5_USD_PER_GPU_HOUR" --usd-per-gpu-hour "$V5_USD_PER_GPU_HOUR")
}

# V5_RECIPE: F's recipe flags as ft row 973cd4e3 records them (recipe.base), with C3/R4's lower
# layers and learning rate as pinned, plus recipe.added's --min-lr 0 and --batch-order seed (Fable's
# seed-order ruling: every v5 run, the arm and J5' carry it), plus C1/C2a/C2b as pinned.
# The checkpoint and retention flags are not recipe keys and are added by the training call.
v5_recipe() {
  V5_RECIPE=(--optimizer master)
  case "$V5_LRSET" in
    f) V5_RECIPE+=(--lr 1e-5) ;;
    j6dv4) V5_RECIPE+=(--lr 3e-5 --beta2 0.95) ;;
    *) say "V5_LRSET is '$V5_LRSET'; no recipe"; return 3 ;;
  esac
  V5_RECIPE+=(--epoch --no-memorise --batch-tokens 35403)
  case "$V5_LOWER" in
    keep) V5_RECIPE+=(--lower-layers-n 8 --lower-layers-lr-scale 0.1) ;;
    drop) ;;
    *) say "V5_LOWER is '$V5_LOWER'; no recipe"; return 3 ;;
  esac
  V5_RECIPE+=(--checkpoint-skip-layers "$F_SKIP" --min-lr 0 --batch-order seed)
  case "$V5_C1" in on) V5_RECIPE+=(--option-permutation-seed "$V5_C1_SEED") ;; off) ;; *) return 3 ;; esac
  case "$V5_C2A" in on) V5_RECIPE+=(--train-attention-mask none) ;; off) ;; *) return 3 ;; esac
  case "$V5_C2B" in on) V5_RECIPE+=(--fused-adamw) ;; off) ;; *) return 3 ;; esac
}

# The v5 data argv's shape, before any wait: data flags only, one --out (absolute), one full
# --rev, the box's backbone, and no word naming held-out data or the containment request
# (rule 3: neither is ever read, trained on or rsynced). Sets V5_DATA. Returns 3 with the reasons
# logged.
v5_split_check() {
  local i=0 n=${#V5_SPLIT[@]} a nout=0 nrev=0 nbb=0 bad=""
  V5_DATA=""
  while [ "$i" -lt "$n" ]; do
    a=${V5_SPLIT[$i]}
    case "$a" in
      *heldout*|*held-out*|*held_out*|*containment*) bad="$bad; '$a' names held-out data or the containment request (rule 3)" ;;
    esac
    case "$a" in
      --*)
        case "$V5_SPLIT_FLAGS" in *" $a "*) ;; *) bad="$bad; $a is not a data flag" ;; esac
        case "$a" in
          --out) nout=$((nout + 1)); V5_DATA=${V5_SPLIT[$((i + 1))]} ;;
          --rev) nrev=$((nrev + 1))
                 echo "${V5_SPLIT[$((i + 1))]}" | grep -Eqx '[0-9a-f]{40}' || bad="$bad; --rev '${V5_SPLIT[$((i + 1))]}' is not a full 40-hex commit" ;;
          --real-backbone) nbb=$((nbb + 1))
                 [ "${V5_SPLIT[$((i + 1))]}" = "$BACKBONE" ] || bad="$bad; --real-backbone is not the box's snapshot $BACKBONE" ;;
        esac ;;
    esac
    i=$((i + 1))
  done
  [ "$nout" -eq 1 ] || bad="$bad; $nout --out (one wanted)"
  [ "$nrev" -eq 1 ] || bad="$bad; $nrev --rev (one wanted)"
  [ "$nbb" -eq 1 ] || bad="$bad; $nbb --real-backbone (one wanted)"
  case "$V5_DATA" in /*) ;; *) bad="$bad; --out '$V5_DATA' is not an absolute path" ;; esac
  if [ -n "$bad" ]; then say "V5_SPLIT refused:${bad#;}"; return 3; fi
}

# V5_CTL_SPLIT: V5_SPLIT without --out and --real-backbone, the form ft_linear_control.py takes
# (idle_common.sh CTL_SPLIT is F's, box_f_controls.sh's SPLIT).
v5_ctl_split() {
  local i=0 n=${#V5_SPLIT[@]}
  V5_CTL_SPLIT=()
  while [ "$i" -lt "$n" ]; do
    case "${V5_SPLIT[$i]}" in
      --out|--real-backbone) i=$((i + 2)) ;;
      *) V5_CTL_SPLIT+=("${V5_SPLIT[$i]}"); i=$((i + 1)) ;;
    esac
  done
}

# The renamed pre-registration: no draft key, the 2x H100 amendment's form, and the numbers this
# queue prints, read from it. Sets V5_EST_SEED_USD/H (a v5 or arm seed), V5_EST_J5_USD/H (a J5'
# seed), V5_APPROVED_USD/H (the 11 runs' total) and V5NW_W (arm_noul_weight.w.value). The per-seed
# figures are launch.projected_cost_usd.check's, cross-checked against the four block totals and
# the per-GPU rate V5_USD_PER_GPU_HOUR; the probe's margin and the ledger names' box are checked
# against hardware.probe.margin_gib and seeds.ledger / hardware.ledger. Any disagreement refuses.
v5_read_prereg() {
  local out
  if ! v5_positive "$V5_USD_PER_GPU_HOUR"; then
    say "V5_USD_PER_GPU_HOUR is '$V5_USD_PER_GPU_HOUR', not a positive rate; no rate to price with"; return 3
  fi
  out=$("$PY" - "$V5_PREREG" "$V5_USD_PER_GPU_HOUR" "$V5_BOX" "$V5_PROBE_MARGIN_GIB" <<'PYEOF'
import json, math, re, sys

path, rate, box, margin = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4])


def refuse(msg):
    print(f"REFUSED: {path}: {msg}")
    sys.exit(3)


try:
    d = json.load(open(path, encoding="utf-8"))
except (OSError, ValueError) as exc:
    refuse(f"unreadable ({exc})")
if not isinstance(d, dict) or "draft" in d:
    refuse("it still carries the top-level 'draft' key: it binds only renamed without it")
num = r"([0-9]+(?:\.[0-9]+)?)"
try:
    usd = d["launch"]["projected_cost_usd"]
    hours = d["launch"]["projected_gpu_hours"]
    check = usd["check"]
    v5 = re.search(r"a v5 seed .*?~ " + num + r" h, \$" + num + ";", check)
    j5 = re.search(r"a J5' seed .*?~ " + num + r" h, \$" + num, check)
    seed_h, seed_usd, j5_h, j5_usd = float(v5[1]), float(v5[2]), float(j5[1]), float(j5[2])
    total_usd, total_h = float(usd["total"]), float(hours["total"])
    blocks = (float(usd["v5_seeds_0_2"]), float(usd["v5_seeds_3_4"]),
              float(usd["j5prime_x3"]), float(usd["noul_weight_x3"]))
    w = d["arm_noul_weight"]["w"]["value"]
    probe_margin = d["hardware"]["probe"]["margin_gib"]
    ledgers = d["seeds"]["ledger"] + " " + d["hardware"]["ledger"]
except (KeyError, TypeError, ValueError, IndexError) as exc:
    refuse(f"the cost, weight, probe or ledger text is not the form this queue reads ({exc!r})")
if "if_seeds_3_4" in usd or "if_seeds_3_4" in hours:
    refuse("it prices seeds 3-4 as if_seeds_3_4, the form before the amendment that bound seeds "
           "0-4; this queue runs them unconditionally")
if type(w) not in (int, float) or not math.isfinite(w) or w <= 0:
    refuse(f"arm_noul_weight.w.value is {w!r}")
if type(probe_margin) not in (int, float) or probe_margin != margin:
    refuse(f"hardware.probe.margin_gib is {probe_margin!r}, not V5_PROBE_MARGIN_GIB {margin:g}")
names = [f"{box}-v5-<date>.jsonl", f"{box}-v5-noulw-<date>.jsonl", f"{box}-v5-probe-<date>.jsonl"]
missing = [n for n in names if n not in ledgers]
if missing:
    refuse(f"V5_BOX is {box!r}, but seeds.ledger and hardware.ledger do not name {missing}")
checks = {
    "3 x v5 seed = v5_seeds_0_2": abs(3 * seed_usd - blocks[0]) <= 0.15,
    "2 x v5 seed = v5_seeds_3_4": abs(2 * seed_usd - blocks[1]) <= 0.15,
    "3 x J5' seed = j5prime_x3": abs(3 * j5_usd - blocks[2]) <= 0.15,
    "3 x v5 seed = noul_weight_x3": abs(3 * seed_usd - blocks[3]) <= 0.15,
    "the four blocks = total": abs(sum(blocks) - total_usd) <= 0.15,
    "8 x v5 seed h + 3 x J5' seed h = total h": abs(8 * seed_h + 3 * j5_h - total_h) <= 0.05,
    "v5 seed h x rate = v5 seed $": abs(seed_h * rate - seed_usd) <= 0.1,
    "J5' seed h x rate = J5' seed $": abs(j5_h * rate - j5_usd) <= 0.1,
}
bad = [k for k, ok in checks.items() if not ok]
if bad:
    refuse(f"its cost figures disagree with each other or with ${rate}/GPU-h: {bad}")
print(seed_usd, seed_h, j5_usd, j5_h, total_usd, total_h, w)
PYEOF
)
  case "$out" in
    REFUSED:*|"") say "${out:-the pre-registration read printed nothing}"; return 3 ;;
  esac
  read -r V5_EST_SEED_USD V5_EST_SEED_H V5_EST_J5_USD V5_EST_J5_H V5_APPROVED_USD V5_APPROVED_H \
    V5NW_W <<< "$out"
  say "v5 pre-registration: a v5/arm seed ~ ${V5_EST_SEED_H} h, \$${V5_EST_SEED_USD}; a J5' seed ~ ${V5_EST_J5_H} h, \$${V5_EST_J5_USD}; approved ~ \$${V5_APPROVED_USD} / ${V5_APPROVED_H} GPU-h for the 11 runs at \$${V5_USD_PER_GPU_HOUR}/GPU-h; arm --noul-weight ${V5NW_W}; probe margin ${V5_PROBE_MARGIN_GIB} GiB"
}

# The Mac prelude's record against this waiter: written by v5_prelude_mac.py with exit 0 at the
# lane's commit, --batch-order seed, plan order digests for plan seeds 0-4 (distinct; the shape
# equal across seeds), the box's train header's shard and data hashes, the box's tokenizer, C1's
# permutation seed iff V5_C1 is on, and the recipe flags of its argv equal to V5_RECIPE.
v5_prelude_check() {
  "$PY" - "$V5_PRELUDE" "$V5_DATA/shards/train/header.json" "$V5_LANE_AT" "$TOKENIZER_SHA256" \
    "$V5_C1" "$V5_C1_SEED" -- "${V5_RECIPE[@]}" <<'PYEOF'
import json, re, sys

rec_p, hdr_p, commit, tok, c1, c1_seed = sys.argv[1:7]
recipe = sys.argv[8:]
VALUED = {"--optimizer", "--lr", "--beta2", "--batch-tokens", "--lower-layers-n",
          "--lower-layers-lr-scale", "--checkpoint-skip-layers", "--min-lr", "--batch-order",
          "--option-permutation-seed", "--train-attention-mask", "--noul-weight", "--span-weight",
          "--train-dtype", "--attn-implementation"}
SWITCHES = {"--epoch", "--no-memorise", "--fused-adamw", "--deterministic"}


def recipe_pairs(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in VALUED:
            out.append((argv[i], argv[i + 1] if i + 1 < len(argv) else None))
            i += 2
            continue
        if argv[i] in SWITCHES:
            out.append((argv[i], None))
        i += 1
    return sorted(out, key=repr)


try:
    rec = json.load(open(rec_p, encoding="utf-8"))
    hdr = json.load(open(hdr_p, encoding="utf-8"))
except (OSError, ValueError) as exc:
    print(f"prelude check: unreadable ({exc})")
    sys.exit(1)
wrong = {}
want = {
    "ok": True,
    "code_commit": commit,
    "batch_order": "seed",
    "tokenizer_json_sha256": tok,
    "shape_equal_across_seeds": True,
    "option_permutation_seed": int(c1_seed) if c1 == "on" else None,
}
for k, v in want.items():
    if rec.get(k) != v:
        wrong[k] = {"recorded": rec.get(k), "expected": v}
for k in ("shard_hash", "data_snapshot_hash"):
    if not hdr.get(k) or rec.get(k) != hdr.get(k):
        wrong[k] = {"recorded": rec.get(k), "box_train_header": hdr.get(k)}
dig = rec.get("plan_order_digest")
seeds = ["0", "1", "2", "3", "4"]
if (not isinstance(dig, dict) or sorted(dig) != seeds
        or not all(isinstance(dig[s], str) and re.fullmatch(r"[0-9a-f]{64}", dig[s]) for s in seeds)
        or len(set(dig.values())) != len(seeds)):
    wrong["plan_order_digest"] = dig
argv = rec.get("argv")
if not isinstance(argv, list) or recipe_pairs(argv) != recipe_pairs(recipe):
    wrong["recipe_flags"] = {"recorded": recipe_pairs(argv) if isinstance(argv, list) else argv,
                             "waiter": recipe_pairs(recipe)}
digests = dig if isinstance(dig, dict) else {}
print(f"prelude record: digests {digests}; mismatches: {wrong or None}")
sys.exit(1 if wrong else 0)
PYEOF
}

# Everything checked before a lane's first lock, and again before every job and decision: the
# binaries, the lane, the pre-registration and its numbers, the data (its train header, and no
# held-out split in it), the tokenizer, the prelude record. Leaves the shell in the lane, with
# QD_PREP_BIN re-exported. Returns 3 with the reason logged.
v5_verify() {
  pin "$RULES_V5" "$RULES_V5_SHA256" || return 3
  pin "$V5_PREP_BIN" "$V5_PREP_BIN_SHA256" || return 3
  export QD_PREP_BIN=$V5_PREP_BIN
  lane_full "$V5_LANE" "$V5_LANE_AT" || return 3
  pin "$V5_PREREG" "$V5_PREREG_SHA256" || return 3
  v5_read_prereg || return 3
  v5_data_check || return 3
  pin "$BACKBONE/tokenizer.json" "$TOKENIZER_SHA256" || return 3
  pin "$V5_PRELUDE" "$V5_PRELUDE_SHA256" || return 3
  v5_prelude_check || { say "the prelude record is not this waiter's (above); refusing"; return 3; }
}

# The data the split names: a v5 train header, and no held-out split on the box (rule 3; the
# build's rsync excludes it, build_order step 6).
v5_data_check() {
  if [ ! -f "$V5_DATA/shards/train/header.json" ]; then say "$V5_DATA/shards/train/header.json is absent; refusing"; return 3; fi
  if [ -e "$V5_DATA/data/heldout" ]; then say "$V5_DATA/data/heldout exists on the box (rule 3); refusing"; return 3; fi
}

# A human marker: a non-empty file. Waits at most V5_HUMAN_WAIT_MAX_S; returns 3 on V5_STOP or at
# the bound. Logs the words it was given.
v5_wait_marker() {
  local path=$1 what=$2 t0
  t0=$(date +%s)
  until [ -s "$path" ]; do
    if [ -e "$V5_STOP" ]; then say "$V5_STOP exists: $what not waited for further"; return 3; fi
    if [ $(( $(date +%s) - t0 )) -ge "$V5_HUMAN_WAIT_MAX_S" ]; then
      say "$what: $path not written within ${V5_HUMAN_WAIT_MAX_S}s; refusing"; return 3
    fi
    sleep 60
  done
  say "$what: $path says '$(head -c 300 "$path" | tr '\n' ' ')'"
}

# This lane's GPU lock on fd 9, as every queue waiter holds one: $Q/gpu<N>.lock for lane N. A
# process with no lane (v5_lane_set) takes none.
v5_lock() {
  if [ -z "$V5_GPU" ]; then say "no lane is set (v5_lane_set); refusing to take a GPU lock"; return 3; fi
  exec 9>"$Q/gpu$V5_GPU.lock"
  flock 9
}

# The lanes' scheduling mutex on fd 8. Lock order is always the lane's GPU lock first, then this:
# a post-seed waiter takes only its lane's lock, so no two holders wait on each other.
v5_sched_lock() {
  exec 8>"$V5_SCHED_LOCK"
  flock 8
}

v5_unlock_all() {
  exec 8>&- 9>&-
}

# --- the rules ------------------------------------------------------------------------------
# v5_rules_run MODE NAME ARGS... : the one place the pinned v5 binary runs. MODE is out (a
# reading: each attempt gets its own new --out JSON, and V5_RULE_JSON names the last one) or
# plain (ft-rows, eval-row). The pin is verified before EVERY attempt; when it does not check,
# nothing runs and this returns 3. Otherwise it returns 0 with V5_RAW (the last attempt's stdout)
# and V5_RC (its exit code). Each attempt's stderr is kept beside it in $DEC_V5
# (NAME-<UTC>-try<k>.stderr) and copied to stderr, where this function also logs, so a caller's
# $(...) carries only the binary's answer. An attempt is made again only on the ledger read race
# (v5_read_race), V5_READ_RETRY_S later, up to V5_READ_TRIES attempts in all; every other refusal
# stands at once. Every attempt only reads (the binary refuses an --out that exists, so each
# gets a new one): a retry decides nothing twice.
v5_rules_run() {
  local mode=$1 name=$2 try=1 base errf
  local -a outarg
  shift 2
  V5_RAW=""
  V5_RC=""
  V5_RULE_JSON=""
  if ! mkdir -p "$DEC_V5"; then say "rules $name: cannot create $DEC_V5; NOT RUN" >&2; return 3; fi
  while :; do
    base=$DEC_V5/$name-$(date -u +%Y%m%dT%H%M%S%N)-try$try
    errf=$base.stderr
    outarg=()
    if [ "$mode" = out ]; then V5_RULE_JSON=$base.json; outarg=(--out "$V5_RULE_JSON"); fi
    if ! pin "$RULES_V5" "$RULES_V5_SHA256" >&2; then
      say "rules $name: the v5 rules binary is not the pinned one; NOT RUN" >&2
      V5_RAW=""
      V5_RC=""
      return 3
    fi
    V5_RAW=$("$RULES_V5" "$@" "${outarg[@]}" 2> "$errf")
    V5_RC=$?
    cat "$errf" >&2
    if [ "$V5_RC:$V5_RAW" != 3:refused ] || ! v5_read_race "$errf" "$@"; then return 0; fi
    if [ "$try" -ge "$V5_READ_TRIES" ]; then
      say "rules $name: still refused on a half-written last ledger line after $try attempts; refused" >&2
      return 0
    fi
    say "rules $name: attempt $try of $V5_READ_TRIES refused on a half-written last ledger line ($(tail -n 1 "$errf" | head -c 300)); again in ${V5_READ_RETRY_S}s" >&2
    sleep "$V5_READ_RETRY_S"
    try=$((try + 1))
  done
}

# v5_read_race STDERR_FILE ARGS... : true iff the binary's last stderr line refuses for ONE
# reason, a malformed line that is the LAST line of a ledger this call names (the value after a
# --*ledger flag in ARGS). The binary's words for it (qd_post_f_rules.rs parse_rows and main;
# AUDIT/v5-queue-2026-10-02/read-race-probe.txt): `qd-post-f-rules: refused: <ledger> line <N>:
# malformed ledger line (<error>)`, several reasons joined by "; ". The last line is counted as
# parse_rows counts, on the ledger as it is now: lines split on newline, numbered from 1, the
# empty one after a final newline skipped. The caller has already required exit 3 and stdout
# refused. A ledger that grew past line N between the read and this check is not retried: held.
v5_read_race() {
  local errf=$1 last prev="" a n prefix rest
  shift
  last=$(tail -n 1 "$errf" 2>/dev/null)
  for a in "$@"; do
    case "$prev" in
      --*ledger)
        if [ -f "$a" ]; then
          n=$(wc -l < "$a" | tr -d ' ')
          if [ -n "$(tail -c 1 "$a")" ]; then n=$((n + 1)); fi
          prefix="qd-post-f-rules: refused: $a line $n: malformed ledger line ("
          case "$last" in
            "$prefix"*")")
              rest=${last#"$prefix"}
              case "$rest" in *";"*) ;; *) return 0 ;; esac ;;
          esac
        fi ;;
    esac
    prev=$a
  done
  return 1
}

# v5_rule LABEL "WORDS" SUBCOMMAND ARGS... : one pre-registered reading through the pinned v5
# binary (v5_rules_run). Sets V5_WORD to one of WORDS (exit 0) or refused, and V5_RULE_JSON to the
# last attempt's --out file. Read like fsucc's: exit 0 with a listed word, or exit 3 with
# refused; any other word, exit code or pairing, a binary that is not the pinned one (then it is
# not run), or a word with no JSON written, is refused.
# V5_SAID is what a hold marker carries (Fable's ruling A, 2026-10-02), so the human tells a
# reading from a tool failure without the logs: V5_WORD, except unknown:<word> when the binary
# exited 0 with one word-shaped token (at most 64 of A-Z a-z 0-9 _ . : -) that is none of WORDS
# and not refused. A refusal, an inconsistent exit, anything else printed, a listed word with no
# JSON, and a binary that did not run all say refused. Actions read V5_WORD, never V5_SAID.
v5_rule() {
  local label=$1 words=$2 w
  shift 2
  V5_WORD=refused
  V5_SAID=refused
  if ! v5_rules_run out "$label" "$@" 2>&1; then say "rule $label: NOT RUN; refused"; return 0; fi
  # exact equality with one listed word: "wins quiet" or " room" is not a word
  for w in $words; do
    if [ "$V5_RC" = 0 ] && [ "$V5_RAW" = "$w" ]; then V5_WORD=$V5_RAW; fi
  done
  if [ "$V5_WORD" = refused ] && [ "$V5_RC:$V5_RAW" != 3:refused ]; then
    say "rule $label: inconsistent checker output (exit $V5_RC, stdout '$(printf '%s' "$V5_RAW" | head -c 200)'); refused"
  fi
  if [ "$V5_WORD" != refused ] && [ ! -s "$V5_RULE_JSON" ]; then
    say "rule $label: the checker said '$V5_WORD' but wrote no JSON to $V5_RULE_JSON; refused"
    V5_WORD=refused
  fi
  V5_SAID=$V5_WORD
  if [ "$V5_WORD" = refused ] && [ "$V5_RC" = 0 ] && v5_unlisted_word "$V5_RAW" "$words"; then
    V5_SAID=unknown:$V5_RAW
    say "rule $label: refused: the binary said '$V5_RAW', which is not one of this reading's words ($words); JSON in $V5_RULE_JSON"
  else
    say "rule $label: $V5_WORD; JSON in $V5_RULE_JSON"
  fi
}

# v5_unlisted_word TOKEN "WORDS": true iff TOKEN is one word-shaped token (1-64 of A-Z a-z 0-9
# _ . : -) that is not refused and none of WORDS.
v5_unlisted_word() {
  local t=$1 w
  if [ -z "$t" ] || [ "${#t}" -gt 64 ] || [ "$t" = refused ]; then return 1; fi
  case "$t" in *[!A-Za-z0-9_.:-]*) return 1 ;; esac
  for w in $2; do
    if [ "$t" = "$w" ]; then return 1; fi
  done
  return 0
}

# A plain call into the pinned v5 binary (ft-rows, eval-row) through v5_rules_run: the last
# attempt's stdout and exit code, or 3 without running it when the pin does not check. Its log
# lines and the binary's stderr go to stderr.
v5_rules_call() {
  v5_rules_run plain "$1" "$@" || return 3
  if [ -n "$V5_RAW" ]; then printf '%s\n' "$V5_RAW"; fi
  return "$V5_RC"
}

# What each word does. Anything not named here holds or skips: an unknown or empty word never
# starts a run.
# R9 (readings.R9_pause_after_seed_0): continue -> run seeds 1-4; pause -> hold until V5_CONTINUE;
# refused, unknown:<word> and no-seed-0-row -> the same hold (this queue's fail-closed reading;
# the DRAFT names only pause).
v5_pause_action() { case "$1" in continue) echo run ;; *) echo hold ;; esac; }
# The arm's launch (arm_noul_weight.launch_condition, R7): room -> run; no_room -> run only with
# V5NW_HUMAN_YES written (the second argument "yes"), else skip; refused -> skip even with the
# yes, because the arm's own reading decides room from the same envelope first, so a refused
# room makes every arm row read refused.
v5_room_action() {
  case "$1:$2" in room:*) echo run ;; no_room:yes) echo run ;; *) echo skip ;; esac
}
# The arm's reading (arm_noul_weight.outcomes): every word is recorded; none starts anything.
v5_noulw_action() { case "$1" in wins|quiet) echo record ;; *) echo record-refused ;; esac; }

# --- runs -----------------------------------------------------------------------------------
# v5_run_vars RUN: the output dir, checkpoint dir, ledger and recipe additions of a run kind;
# v5 (v5 seeds 0-4, J5') or v5nw (the arm: --noul-weight, arm_noul_weight.what). 3 otherwise.
v5_run_vars() {
  case "$1" in
    v5) V5R_OUT=$V5_OUT; V5R_CKPT=$V5_CKPT; V5R_LEDGER=$V5_LEDGER; V5R_EXTRA=() ;;
    v5nw)
      if [ -z "$V5NW_W" ]; then say "the arm's weight was not read from the pre-registration"; return 3; fi
      V5R_OUT=$V5NW_OUT; V5R_CKPT=$V5NW_CKPT; V5R_LEDGER=$V5NW_LEDGER; V5R_EXTRA=(--noul-weight "$V5NW_W") ;;
    *) say "unknown run kind '$1'"; return 3 ;;
  esac
}

# V5_FT_ARGS: FLAG SEED=ID for each seed, ids from the run's training logs. An empty id is a
# usage error in the binary, so a seed with no ft row refuses rather than drops out.
v5_ft_args() {
  local run=$1 flag=$2 s
  shift 2
  v5_run_vars "$run" || return 3
  V5_FT_ARGS=()
  for s in "$@"; do V5_FT_ARGS+=("$flag" "$s=$(log_ft_row "$V5R_OUT/train-s$s.log")"); done
}

# --- money ----------------------------------------------------------------------------------
# One line per GPU step into v5.spend, at the per-GPU rate (each step holds one GPU), and its
# seconds to the running job's own v5job-<job>.spend when there is one. Both lanes and their
# post-seed waiters append to these: one short line, one printf, O_APPEND (>>), so lines do not
# interleave.
v5_spend_add() {
  local usd
  usd=$(awk -v s="$2" -v r="$V5_USD_PER_GPU_HOUR" 'BEGIN { printf "%.2f", s * r / 3600 }')
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$usd" >> "$V5_SPEND"
  if [ -n "$V5_JOB" ]; then printf '%s\n' "$2" >> "$Q/v5job-$V5_JOB.spend"; fi
  v5_total_line
}

# "<GPU-h> <USD>" spent so far by the v5 block (V5_SPEND; 0 0 before its first line).
v5_spent() {
  if [ ! -s "$V5_SPEND" ]; then echo "0 0"; return 0; fi
  awk -F'\t' -v r="$V5_USD_PER_GPU_HOUR" '{ s += $3 } END { printf "%.4f %.2f\n", s / 3600, s * r / 3600 }' "$V5_SPEND"
}

v5_total_line() {
  local sh su
  read -r sh su <<< "$(v5_spent)"
  say "v5 running total: ${sh} GPU-h, \$${su} of the approved ~\$${V5_APPROVED_USD} / ${V5_APPROVED_H} GPU-h"
}

# "<USD> <h>": the pre-registration's estimate of a job (a J5' seed, or a v5 or arm seed).
v5_job_est() {
  case "$1" in
    j5-s*) echo "$V5_EST_J5_USD $V5_EST_J5_H" ;;
    *) echo "$V5_EST_SEED_USD $V5_EST_SEED_H" ;;
  esac
}

# The seconds v5.spend already holds for a job (its .spend), 0 when none.
v5_job_spent_s() {
  if [ ! -s "$Q/v5job-$1.spend" ]; then echo 0; return 0; fi
  awk '{ s += $1 } END { print s + 0 }' "$Q/v5job-$1.spend"
}

# A seed job's post-seed waiter's marker name (v5traj-s<N>, v5nwtraj-s<N>); nothing for J5'.
v5_job_traj() {
  case "$1" in
    v5-s[0-4]) echo "v5traj-s${1#v5-s}" ;;
    v5nw-s[0-2]) echo "v5nwtraj-s${1#v5nw-s}" ;;
  esac
}

# True iff a job has ended, its post-seed waiter included (that waiter is the seed's own spend).
v5_job_finished() {
  local traj
  [ -e "$Q/v5job-$1.done" ] || return 1
  traj=$(v5_job_traj "$1")
  [ -z "$traj" ] || [ ! -e "$Q/$traj.queued" ] || [ -e "$Q/$traj.done" ]
}

# "<USD> <h>" the lanes' running jobs may still spend: for each claimed job not finished, its
# estimate less what v5.spend already holds for it, never below zero.
v5_inflight() {
  local j eu eh s tu=0 th=0
  for j in $V5_JOBS; do
    [ -e "$Q/v5job-$j.claimed" ] || continue
    if v5_job_finished "$j"; then continue; fi
    read -r eu eh <<< "$(v5_job_est "$j")"
    s=$(v5_job_spent_s "$j")
    read -r tu th <<< "$(awk -v tu="$tu" -v th="$th" -v eu="$eu" -v eh="$eh" -v s="$s" -v r="$V5_USD_PER_GPU_HOUR" \
      'BEGIN { u = eu - s * r / 3600; h = eh - s / 3600; if (u < 0) u = 0; if (h < 0) h = 0; printf "%.6f %.6f\n", tu + u, th + h }')"
  done
  echo "$tu $th"
}

# v5_budget_ok EST_USD EST_H LABEL: whether a run may start against the approved total
# (launch.projected_cost_usd.total / projected_gpu_hours.total: the 11 runs). 0: it fits beside
# what the running jobs may still spend. 3: it would carry the running total past the approved
# total even alone: refused (the lead's instruction, 2026-10-02), unless V5_OVER_BUDGET_YES is
# written. 4: it fits alone but not beside the running jobs: it waits, and is picked again once
# they end (two lanes must not jointly start what one lane at a time would refuse).
v5_budget_ok() {
  local eu=$1 eh=$2 label=$3 sh su iu ih alone over
  read -r sh su <<< "$(v5_spent)"
  read -r iu ih <<< "$(v5_inflight)"
  alone=$(awk -v su="$su" -v eu="$eu" -v au="$V5_APPROVED_USD" -v sh="$sh" -v eh="$eh" -v ah="$V5_APPROVED_H" \
    'BEGIN { print ((su + eu > au + 0.005 || sh + eh > ah + 0.0005) ? 1 : 0) }')
  over=$(awk -v su="$su" -v iu="$iu" -v eu="$eu" -v au="$V5_APPROVED_USD" -v sh="$sh" -v ih="$ih" -v eh="$eh" -v ah="$V5_APPROVED_H" \
    'BEGIN { print ((su + iu + eu > au + 0.005 || sh + ih + eh > ah + 0.0005) ? 1 : 0) }')
  if [ "$over" = 0 ]; then return 0; fi
  if [ -s "$V5_OVER_BUDGET_YES" ]; then
    say "$label: \$${su} + ~\$${eu} / ${sh} + ~${eh} GPU-h crosses the approved ~\$${V5_APPROVED_USD} / ${V5_APPROVED_H} GPU-h; $V5_OVER_BUDGET_YES says '$(head -c 300 "$V5_OVER_BUDGET_YES" | tr '\n' ' ')', so it runs"
    return 0
  fi
  if [ "$alone" = 1 ]; then
    say "$label REFUSED: \$${su} spent + ~\$${eu} / ${sh} + ~${eh} GPU-h would cross the approved ~\$${V5_APPROVED_USD} / ${V5_APPROVED_H} GPU-h; NOT RUN (the human's words in $V5_OVER_BUDGET_YES let it run)"
    return 3
  fi
  say "$label waits: \$${su} spent + ~\$$(printf '%.2f' "$iu") / ~$(printf '%.2f' "$ih") GPU-h the running jobs may still spend + ~\$${eu} / ~${eh} GPU-h would cross the approved ~\$${V5_APPROVED_USD} / ${V5_APPROVED_H} GPU-h; it is picked again once they end"
  return 4
}

v5_cost_line() {
  say "cost: $1: cap $2 s = \$$(awk -v s="$2" -v r="$V5_USD_PER_GPU_HOUR" 'BEGIN { printf "%.2f", s * r / 3600 }') at \$${V5_USD_PER_GPU_HOUR}/GPU-h; pre-registration estimate $3"
}

# v5_train_seed RUN SEED LABEL: one training run in v5's form (recipe.base + added; the arm adds
# its weight): train + score (--score-val --needle --ood) with the checkpoint and the retained
# towers, under the training cap, on this lane's GPU. The caller holds the lane's lock and has
# checked the budget (v5_job_budget, under the scheduling mutex). Sets V5_FT to the ft row the
# log names (empty when none). Refuses a seed whose log exists.
v5_train_seed() {
  local run=$1 seed=$2 label=$3 log t0 rc
  V5_FT=""
  v5_run_vars "$run" || return 3
  log=$V5R_OUT/train-s$seed.log
  if [ -e "$log" ]; then say "$label: $log exists; refusing to run it twice"; return 3; fi
  mkdir -p "$V5R_OUT" "$V5R_CKPT" || return 3
  v5_cost_line "$label train+score" "$V5_TRAIN_CAP_S" "~ ${V5_EST_SEED_H} h, \$${V5_EST_SEED_USD} for the seed with its needle control and trajectory"
  say "$label on GPU $V5_GPU at $(git rev-parse --short HEAD): ${V5_RECIPE[*]} ${V5R_EXTRA[*]} --seeds $seed -> $V5R_LEDGER"
  t0=$(date +%s)
  timeout $((V5_TRAIN_CAP_S + 1800)) "$PY" -u tools/real_ft_run.py "${V5_SPLIT[@]}" "${V5_RECIPE[@]}" "${V5R_EXTRA[@]}" \
    --devices cuda --seeds "$seed" \
    --checkpoint-dir "$V5R_CKPT" --checkpoint-every "$V5_CKPT_EVERY" --retain-tower-every "$V5_RETAIN_EVERY" \
    --score-val --needle --ood --ood-general-record "$REC" \
    --verdicts-out "$V5R_OUT/verdicts-s$seed.jsonl" --suite-verdicts-out "$V5R_OUT/suite-verdicts-s$seed.jsonl" \
    --wall-clock-cap-s "$V5_TRAIN_CAP_S" "${V5_COST[@]}" --ledger "$V5R_LEDGER" \
    --approved-by "$(v5_approved "$label")" \
    2>&1 | tee "$log"
  rc=$?
  v5_spend_add "$label train+score" $(( $(date +%s) - t0 ))
  say "$label train+score done (exit $rc)"
  V5_FT=$(log_ft_row "$log")
  if [ -z "$V5_FT" ]; then say "$label: no ft row in $log"; return 3; fi
  v5_digest_report "$seed"
}

# Report-only: the ft row's corpus.plan_order_digest beside the one the Mac prelude printed for
# this plan seed (amendments_pending; GAP-V5-RULES-AMENDMENTS-NOT-CROSS-CHECKED-2026-10-02 says no
# checker compares them). Decides nothing.
v5_digest_report() {
  "$PY" - "$V5R_LEDGER" "$V5_FT" "$V5_PRELUDE" "$1" <<'PYEOF'
import json, sys

ledger, row_id, prelude, seed = sys.argv[1:5]
got = None
try:
    with open(ledger, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("row_id") == row_id:
                got = (row.get("metrics") or {}).get("corpus.plan_order_digest", {}).get("value")
    want = json.load(open(prelude, encoding="utf-8"))["plan_order_digest"][seed]
except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
    print(f"plan order digest, plan seed {seed}: not compared ({exc!r}); report-only")
    sys.exit(0)
word = "MATCHES" if got == want else "DIFFERS FROM"
print(f"plan order digest, plan seed {seed}: ft row {got} {word} the prelude's {want}; report-only")
PYEOF
}

# v5_needle_control RUN SEED FT LABEL: F's needle control on the seed's checkpoint (box_q_s34.sh's
# form verbatim: --needle-control 1024,2048,4096 --score-dtype fp32, cap 5,400 s). The caller
# holds the lane's lock.
v5_needle_control() {
  local run=$1 seed=$2 ft=$3 label=$4 log ck t0
  v5_run_vars "$run" || return 3
  ck=$V5R_CKPT/epoch-seed$seed-cuda.json
  log=$V5R_OUT/needle-control-s$seed.log
  if [ -z "$ft" ] || [ ! -f "$ck" ]; then say "$label: no ft row or checkpoint from this run; its needle control NOT RUN"; return 3; fi
  if [ -e "$log" ]; then say "$label: $log exists; refusing to run it twice"; return 3; fi
  v5_cost_line "$label needle control" "$V5_NEEDLE_CAP_S" "(inside the seed's ~\$${V5_EST_SEED_USD})"
  t0=$(date +%s)
  timeout "$V5_NEEDLE_CAP_S" "$PY" -u tools/real_ft_run.py "${V5_SPLIT[@]}" --devices cuda --seeds "$seed" \
    --score-val --needle --needle-control 1024,2048,4096 \
    --score-checkpoint "$ck" --score-dtype fp32 \
    --ft-ledger "$V5R_LEDGER" --ft-row-id "$ft" \
    "${V5_COST[@]}" --wall-clock-cap-s "$V5_NEEDLE_CAP_S" \
    --ledger "$V5R_LEDGER" --suite-verdicts-out "$V5R_OUT/needle-control-s$seed.jsonl" \
    2>&1 | tee "$log"
  say "$label needle control done (exit $?)"
  v5_spend_add "$label needle control" $(( $(date +%s) - t0 ))
}

# v5_post_seed RUN SEED FT: start the seed's own post-seed waiter (box_q_v5traj.sh: trajectory
# rows under this lane's lock, then the CPU controls), release the lane's lock to it, and wait at
# most V5_TRAJ_HANDOFF_MAX_S for it to take the lock (its .started) or end (.done), so this lane's
# next job cannot take the lock first. Its outcome never gates the next job.
v5_post_seed() {
  local run=$1 seed=$2 ft=$3 m t0
  m=$Q/${run}traj-s$seed
  if [ -z "$ft" ]; then
    say "${run}traj-s$seed: no ft row, so no trajectory or controls for this seed"
    exec 9>&-
    return 0
  fi
  if [ -e "$m.queued" ]; then
    say "${run}traj-s$seed is already queued; not starting a second"
  else
    mkdir -p "$V5_LOGS"
    setsid nohup bash "$PF/box_q_v5traj.sh" "$run" "$seed" "$ft" "$V5_GPU" \
      > "$V5_LOGS/q-${run}traj-s$seed.log" 2>&1 < /dev/null 8>&- 9>&- &
    say "${run}traj-s$seed started on lane $V5_GPU (log $V5_LOGS/q-${run}traj-s$seed.log)"
  fi
  exec 9>&-
  t0=$(date +%s)
  until [ -f "$m.started" ] || [ -f "$m.done" ]; do
    if [ $(( $(date +%s) - t0 )) -ge "$V5_TRAJ_HANDOFF_MAX_S" ]; then
      say "${run}traj-s$seed did not take gpu$V5_GPU.lock within ${V5_TRAJ_HANDOFF_MAX_S}s; the lane goes on without waiting for it"
      return 0
    fi
    sleep 5
  done
}

# The CPU controls of box_f_controls.sh through idle_common.sh's controls_block, on v5's split.
# controls_block reads CTL_SPLIT; bash's dynamic scope gives it this function's local copy, so
# the block is reused, not copied.
v5_controls() {
  # shellcheck disable=SC2034  # read by controls_block (dynamic scope)
  local -a CTL_SPLIT
  v5_ctl_split
  # shellcheck disable=SC2034
  CTL_SPLIT=("${V5_CTL_SPLIT[@]}")
  controls_block "$@"
}

# --- the rule -------------------------------------------------------------------------------
# A word file's word: its first line, at most 64 bytes.
v5_word() {
  head -c 64 "$1" 2>/dev/null | head -n 1 | tr -d '\n'
}

# True iff every job named is done (whatever its result).
v5_all_done() {
  local j
  for j in "$@"; do [ -e "$Q/v5job-$j.done" ] || return 1; done
}

# True iff R9 holds the block: kept, its word read and not continue, and no V5_CONTINUE yet.
v5_r9_holding() {
  [ "$V5_R9" = keep ] && [ -e "$V5R9_WORD" ] && [ "$(v5_word "$V5R9_WORD")" != continue ] && [ ! -s "$V5_CONTINUE" ]
}

# v5_word_input FILE WORD: ready when FILE says WORD, possible while it is absent, else impossible.
v5_word_input() {
  if [ ! -e "$1" ]; then echo possible
  elif [ "$(v5_word "$1")" = "$2" ]; then echo ready
  else echo impossible
  fi
}

# v5_job_input JOB: ready, possible (its inputs may still come) or impossible. The inputs
# (hardware.lanes): v5 seed 0 none; v5 seeds 1-4 R9's continue or V5_CONTINUE, only when R9 is
# kept; the arm's seeds v5nw.launch run (decided with v5nw.room, from v5 seeds 0-2's eval rows);
# J5' seed s v5 seed s's ft row (its job's .done says "ft <row>"; the job reads the ft and eval
# rows themselves before it trains).
v5_job_input() {
  case "$1" in
    v5-s0) echo ready ;;
    v5-s[1-4])
      case "$V5_R9" in
        waive) echo ready ;;
        keep)
          if [ ! -e "$V5R9_WORD" ]; then
            if [ -e "$Q/v5job-v5-s0.done" ]; then echo impossible; else echo possible; fi
          elif [ "$(v5_word "$V5R9_WORD")" = continue ] || [ -s "$V5_CONTINUE" ]; then echo ready
          else echo possible
          fi ;;
        *) echo impossible ;;
      esac ;;
    v5nw-s[0-2]) v5_word_input "$V5NW_LAUNCH" run ;;
    j5-s[0-2])
      if [ -e "$Q/v5job-v5-s${1#j5-s}.done" ]; then
        case "$(v5_word "$Q/v5job-v5-s${1#j5-s}.done")" in "ft "*) echo ready ;; *) echo impossible ;; esac
      elif [ -e "$Q/v5job-v5-s${1#j5-s}.claimed" ]; then echo possible
      else
        case "$(v5_job_input "v5-s${1#j5-s}")" in impossible) echo impossible ;; *) echo possible ;; esac
      fi ;;
    *) echo impossible ;;
  esac
}

# v5_next_job: what a lane does next, from the marker files alone (the caller holds the
# scheduling mutex): stop (V5_STOP), hold (R9 held), "decide room|arm" (a decision whose inputs
# are done and whose word is not written: made before any pick), "job <id>" (the earliest job in
# the human's order whose inputs are ready and that no lane has taken), wait (none is ready, but
# one may become so), or none (nothing is left that any lane could take).
v5_next_job() {
  local j waiting=""
  if [ -e "$V5_STOP" ]; then echo stop; return 0; fi
  if v5_r9_holding; then echo hold; return 0; fi
  if v5_all_done v5-s0 v5-s1 v5-s2 && [ ! -e "$V5NW_ROOM" ]; then echo "decide room"; return 0; fi
  if [ "$(v5_word "$V5NW_LAUNCH")" = run ] && v5_all_done v5nw-s0 v5nw-s1 v5nw-s2 && [ ! -e "$V5NW_WORD" ]; then
    echo "decide arm"; return 0
  fi
  for j in $V5_JOBS; do
    if [ -e "$Q/v5job-$j.done" ] || [ -e "$Q/v5job-$j.claimed" ]; then continue; fi
    case "$(v5_job_input "$j")" in
      ready) echo "job $j"; return 0 ;;
      possible) waiting=1 ;;
    esac
  done
  if [ -n "$waiting" ]; then echo wait; else echo none; fi
}

# --- the decisions --------------------------------------------------------------------------
# R9 (readings.R9_pause_after_seed_0), right after v5 seed 0's train+score, inside its job. Kept:
# v5-pause on seed 0's ft row; its word is written once to v5r9.word (and, when it holds, the
# reason to v5.paused first: pause, refused, unknown:<word> or no-seed-0-row, Fable's ruling A).
# Waived (V5_R9=waive, the human's call): nothing is read and nothing written.
v5_r9() {
  local ft=$1
  if [ "$V5_R9" = waive ]; then say "R9 waived (V5_R9=waive): v5-pause NOT RUN; seeds 1-4 need no continue"; return 0; fi
  if [ -n "$ft" ]; then
    v5_rule v5-pause "continue pause" v5-pause --preregistration "$V5_PREREG" --ledger "$V5_LEDGER" --ft-row "0=$ft"
    v5_r9_record "$V5_SAID"
  else
    say "R9: v5 seed 0 wrote no ft row; v5-pause NOT RUN; no-seed-0-row"
    v5_r9_record no-seed-0-row
  fi
}

v5_r9_record() {
  local why=$1
  if [ "$(v5_pause_action "$why")" = run ]; then
    say "R9: continue: seeds 1-4 start"
  else
    write_atomic "$V5_PAUSED" "$why" || return 3
    say "R9: $why: HOLD. $V5_PAUSED says '$why'; no lane starts a v5 job (seeds 1-4, the arm, J5') until the human writes $V5_CONTINUE ($V5_STOP ends v5 here)"
  fi
  write_atomic "$V5R9_WORD" "$why"
}

v5_decide() {
  case "$1" in
    room) v5_decide_room ;;
    arm) v5_decide_arm ;;
    *) say "unknown decision '$1'; refusing"; return 3 ;;
  esac
}

# The arm's launch condition (arm_noul_weight.launch_condition, R7): v5-noulw --room on v5 seeds
# 0-2's rows (never 3-4), once seeds 0-2 are done. Its word is written once to v5nw.room, and the
# launch it gives, with V5NW_HUMAN_YES as it stands then, once to v5nw.launch (run or skip): a
# yes written later does not start an arm that was skipped.
v5_decide_room() {
  local room yes=no launch
  v5_ft_args v5 --ft-row 0 1 2 || return 3
  v5_rule v5nw-room "room no_room" v5-noulw --room --preregistration "$V5_PREREG" \
    --noul-preregistration "$NOUL_PREREG" --v5-ledger "$V5_LEDGER" "${V5_FT_ARGS[@]}"
  room=$V5_WORD
  if [ -s "$V5NW_HUMAN_YES" ]; then yes=yes; fi
  launch=$(v5_room_action "$room" "$yes")
  write_atomic "$V5NW_ROOM" "$room" || { say "v5nw: could not write $V5NW_ROOM"; return 3; }
  write_atomic "$V5NW_LAUNCH" "$launch" || { say "v5nw: could not write $V5NW_LAUNCH"; return 3; }
  case "$launch:$room" in
    run:room) say "v5nw: room: the arm runs (seeds 0-2)" ;;
    run:no_room) say "v5nw: no_room, and $V5NW_HUMAN_YES is written: the arm runs on the human's yes" ;;
    skip:no_room) say "v5nw SKIPPED (R7): v5 holds both targets on all three seeds (no_room) and $V5NW_HUMAN_YES is not written; J5' takes the arm's slots" ;;
    *) say "v5nw REFUSED: v5-noulw --room said '$room'; the arm NOT RUN (a refused room makes every arm row read refused; V5NW_HUMAN_YES does not override it); J5' takes the arm's slots" ;;
  esac
}

# The arm's reading (arm_noul_weight.outcomes), once its three seeds are done: wins / quiet /
# refused, written once to v5nw.word. It feeds the next re-plan only; nothing branches on it.
v5_decide_arm() {
  local -a ids
  v5_ft_args v5 --ft-row 0 1 2 || return 3
  ids=("${V5_FT_ARGS[@]}")
  v5_ft_args v5nw --arm-ft-row 0 1 2 || return 3
  v5_rule v5nw "wins quiet" v5-noulw --preregistration "$V5_PREREG" \
    --noul-preregistration "$NOUL_PREREG" --v5-ledger "$V5_LEDGER" "${ids[@]}" \
    --arm-ledger "$V5NW_LEDGER" "${V5_FT_ARGS[@]}"
  write_atomic "$V5NW_WORD" "$V5_WORD" || { say "v5nw: could not write $V5NW_WORD"; return 3; }
  case "$(v5_noulw_action "$V5_WORD")" in
    record) say "v5nw: the arm reads '$V5_WORD' (feeds the next re-plan only; JSON $V5_RULE_JSON)" ;;
    *) say "v5nw: the arm's reading REFUSED ('$V5_WORD'; JSON $V5_RULE_JSON); never read as quiet" ;;
  esac
}

# --- the jobs -------------------------------------------------------------------------------
v5_job_label() {
  case "$1" in
    v5-s*) echo "v5 seed ${1#v5-s}" ;;
    v5nw-s*) echo "arm seed ${1#v5nw-s}" ;;
    j5-s*) echo "J5' seed ${1#j5-s}" ;;
    *) echo "$1" ;;
  esac
}

# v5_job_budget JOB: v5_budget_ok on the job's estimate (0 run, 3 refused, 4 wait).
v5_job_budget() {
  local eu eh
  read -r eu eh <<< "$(v5_job_est "$1")"
  v5_budget_ok "$eu" "$eh" "$(v5_job_label "$1")"
}

v5_job_done() {
  write_atomic "$Q/v5job-$1.done" "$2"
}

# A job refused before it starts (the budget) is done, NOT RUN. Seed 0 refused with R9 kept holds
# R9 as no-seed-0-row, written before the job's .done so no lane reads seed 0 done without it.
v5_job_refuse() {
  if [ "$1" = v5-s0 ] && [ "$V5_R9" = keep ]; then
    say "R9: v5 seed 0 was not run; v5-pause NOT RUN; no-seed-0-row"
    v5_r9_record no-seed-0-row
  fi
  v5_job_done "$1" "refused: $2"
}

# The arm's label: on no_room it names the human's yes it runs on.
v5_arm_label() {
  if [ "$(v5_word "$V5NW_ROOM")" = no_room ]; then
    echo "noul-weight arm on no_room by $V5NW_HUMAN_YES: '$(head -c 300 "$V5NW_HUMAN_YES" | tr '\n' ' ')'"
  else
    echo "noul-weight arm"
  fi
}

# v5_job_seed RUN SEED: one seed of v5 (RUN v5, seeds 0-4) or of the noul-weight arm (v5nw, seeds
# 0-2: v5's recipe plus --noul-weight, each on v5 seed s's batch order, rows into the arm ledger),
# in v5's per-seed form: train + score; for v5 seed 0, R9; the needle control; then the lane's
# lock passes to the seed's post-seed waiter. Sets V5_JOB_RESULT.
v5_job_seed() {
  local run=$1 seed=$2 label nlabel ft
  if [ "$run" = v5nw ]; then
    label="$(v5_arm_label) seed $seed (--noul-weight $V5NW_W)"
    nlabel="arm seed $seed"
  else
    label="v5 seed $seed"
    nlabel=$label
  fi
  v5_train_seed "$run" "$seed" "$label"
  ft=$V5_FT
  if [ "$run:$seed" = v5:0 ]; then v5_r9 "$ft"; fi
  v5_needle_control "$run" "$seed" "$ft" "$nlabel"
  v5_post_seed "$run" "$seed" "$ft"
  if [ -n "$ft" ]; then V5_JOB_RESULT="ft $ft"; else V5_JOB_RESULT="no-ft"; fi
}

# v5_job_j5 SEED: J5', the shuffled_label control on v5's exact path (campaign/v5-preregistered.json
# j5prime; box_q_j5pp.sh's form). J5' seed s runs once v5 seed s's ft row passes ft-rows on v5's
# ledger (completed, quick false, tag epoch, the seed claimed: hardware.lanes reads J5' seed s's
# inputs per seed) and eval-row names its epoch-score-val row; then v5's recipe flags as v5 ran
# them (--batch-order seed: J5' seed s trains on v5 seed s's batch order; it reads no checkpoint)
# with --score-val --shuffled-label TARGET, cap 32,400 s, rows into v5's ledger. No checkpoint
# retention, no needle or OOD scoring. real_ft_run refuses before any tower loads a run whose
# planned recipe differs from the target's ft row in anything but the shuffle and the span
# weight. Sets V5_JOB_RESULT.
v5_job_j5() {
  local seed=$1 ft log target t0 rc
  ft=$(log_ft_row "$V5_OUT/train-s$seed.log")
  log=$V5_OUT/j5p-s$seed.log
  if [ -e "$log" ]; then say "J5' seed $seed: $log exists; refusing to run it twice"; V5_JOB_RESULT="refused: $log exists"; return 3; fi
  v5_ft_args v5 --ft-row "$seed" || { V5_JOB_RESULT="refused: no run vars"; return 3; }
  if ! v5_rules_call ft-rows --ledger "$V5_LEDGER" "${V5_FT_ARGS[@]}"; then
    say "v5j5 SKIPPED: J5' seed $seed: v5's ft row '${V5_FT_ARGS[*]}' in $V5_LEDGER is not a completed v5 ft row (the binary's reason is above); J5' seed $seed NOT RUN"
    V5_JOB_RESULT="skipped: ft-rows refused ${V5_FT_ARGS[*]}"
    return 0
  fi
  if ! target=$(v5_rules_call eval-row --ledger "$V5_LEDGER" --ft-row "$seed=$ft"); then
    say "J5' seed $seed NOT RUN: v5 seed $seed has no single completed epoch-score-val eval row scored from ft row '${ft}' (the binary's reason is above)"
    V5_JOB_RESULT="skipped: no eval row"
    return 0
  fi
  if ! echo "$target" | grep -Eqx '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'; then
    say "J5' seed $seed NOT RUN: eval-row printed '$(printf '%s' "$target" | head -c 200)', not one row id"
    V5_JOB_RESULT="skipped: eval-row printed no row id"
    return 0
  fi
  say "J5' seed $seed on GPU $V5_GPU at $(git rev-parse --short HEAD) -> eval row $target (ft row $ft): ${V5_RECIPE[*]} -> $V5_LEDGER"
  v5_cost_line "J5' seed $seed" "$V5_TRAIN_CAP_S" "~ ${V5_EST_J5_H} h, \$${V5_EST_J5_USD}"
  t0=$(date +%s)
  timeout $((V5_TRAIN_CAP_S + 1800)) "$PY" -u tools/real_ft_run.py "${V5_SPLIT[@]}" "${V5_RECIPE[@]}" \
    --devices cuda --seeds "$seed" \
    --score-val --wall-clock-cap-s "$V5_TRAIN_CAP_S" "${V5_COST[@]}" --ledger "$V5_LEDGER" \
    --approved-by "$(v5_approved "J5' shuffled_label on v5 seed $seed (v5j5)")" \
    --shuffled-label "$target" \
    2>&1 | tee "$log"
  rc=$?
  v5_spend_add "J5' seed $seed" $(( $(date +%s) - t0 ))
  say "J5' seed $seed done (exit $rc)"
  V5_JOB_RESULT="ran exit $rc"
}

# v5_run_job JOB: run a job this lane has claimed, holding its lock, and write its .done.
v5_run_job() {
  local j=$1
  V5_JOB=$j
  V5_JOB_RESULT=""
  touch "$Q/v5lane$V5_GPU.started"
  say "v5 lane $V5_GPU: $j ($(v5_job_label "$j"))"
  case "$j" in
    v5-s[0-4]) v5_job_seed v5 "${j#v5-s}" ;;
    v5nw-s[0-2]) v5_job_seed v5nw "${j#v5nw-s}" ;;
    j5-s[0-2]) v5_job_j5 "${j#j5-s}" ;;
    *) V5_JOB_RESULT="refused: unknown job" ;;
  esac
  v5_job_done "$j" "${V5_JOB_RESULT:-no-ft}"
  V5_JOB=""
}

# --- the lane -------------------------------------------------------------------------------
# A lane refuses to start over markers an earlier launch left: its own lane markers, and every
# marker the lanes or the human write after the launch. The lead clears them. (V5_LAUNCH_YES is
# the human's and may be written before the launch.)
v5_lane_stale() {
  local f
  for f in "$Q/v5lane$1.queued" "$Q/v5lane$1.started" "$Q/v5lane$1.done" \
           "$V5_PAUSED" "$V5_CONTINUE" "$V5_STOP" "$V5_PROBE_YES" "$V5_PROBE_PASSED" "$V5_PROBE_FAILED" \
           "$V5R9_WORD" "$V5NW_ROOM" "$V5NW_LAUNCH" "$V5NW_WORD" \
           "$Q"/v5job-* "$Q"/v5traj-s* "$Q"/v5nwtraj-s*; do
    if [ -e "$f" ]; then
      say "v5 lane $1: $f exists before this lane started: stale, or decided once by an earlier launch ('$(head -c 64 "$f" | tr -d '\n')'); the lead clears it. v5 NOT RUN"
      return 3
    fi
  done
}

# The last non-blank line of a log, at most 300 bytes.
v5_last_line() {
  grep -v '^[[:space:]]*$' "$1" | tail -n 1 | head -c 300
}

# v5_probe_once ATTEMPT LOG: one probe call (V5_PROBE_ARGV) on this lane's GPU. Attempt 2 is the
# same call with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, set for that call only.
v5_probe_once() {
  local t0 rc
  t0=$(date +%s)
  if [ "$1" = 2 ]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout $((V5_PROBE_CAP_S + 600)) \
      "$PY" -u tools/real_ft_run.py "${V5_PROBE_ARGV[@]}" 2>&1 | tee "$2"
  else
    timeout $((V5_PROBE_CAP_S + 600)) "$PY" -u tools/real_ft_run.py "${V5_PROBE_ARGV[@]}" 2>&1 | tee "$2"
  fi
  rc=$?
  v5_spend_add "v5 memory probe (attempt $1)" $(( $(date +%s) - t0 ))
  return "$rc"
}

# v5_probe: the memory probe (hardware.probe), once, on lane 0's GPU under its lock, before any
# seed: tools/real_ft_run.py --probe-shapes 12 with v5's recipe (--epoch --no-memorise) and data
# argv, no scoring flags, cap 1,800 s, rows into the probe ledger, --approved-by the human's yes.
# A non-zero exit is retried once with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (hardware.probe.fallback; no recipe key moves). Returns 0 when an attempt passed
# ($Q/v5.probe-passed says which), 1 when both failed ($Q/v5.probe-failed says why), 3 when it
# could not run.
v5_probe() {
  local rc why a
  mkdir -p "$V5_OUT" || return 3
  for a in 1 2; do
    if [ -e "$V5_OUT/probe-attempt$a.log" ]; then say "v5 probe: $V5_OUT/probe-attempt$a.log exists; refusing to probe twice"; return 3; fi
  done
  V5_PROBE_ARGV=("${V5_SPLIT[@]}" "${V5_RECIPE[@]}" --devices cuda --seeds 0
                 --probe-shapes "$V5_PROBE_MARGIN_GIB" --wall-clock-cap-s "$V5_PROBE_CAP_S"
                 "${V5_COST[@]}" --ledger "$V5_PROBE_LEDGER"
                 --approved-by "$(v5_approved "v5 memory probe" "$V5_PROBE_CAP_S")")
  v5_cost_line "v5 memory probe" "$V5_PROBE_CAP_S" "hardware.probe.job's \$2.10 at the cap, outside the 11 runs' figures"
  say "v5 probe on GPU $V5_GPU: --probe-shapes $V5_PROBE_MARGIN_GIB -> $V5_PROBE_LEDGER"
  v5_probe_once 1 "$V5_OUT/probe-attempt1.log"
  rc=$?
  if [ "$rc" = 0 ]; then
    write_atomic "$V5_PROBE_PASSED" "attempt 1 passed (exit 0); ft row $(log_ft_row "$V5_OUT/probe-attempt1.log")"
    say "v5 probe passed on attempt 1"
    return 0
  fi
  why="attempt 1 exit $rc: $(v5_last_line "$V5_OUT/probe-attempt1.log")"
  say "v5 probe attempt 1 FAILED ($why); once more with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for that call only (hardware.probe.fallback)"
  v5_probe_once 2 "$V5_OUT/probe-attempt2.log"
  rc=$?
  if [ "$rc" = 0 ]; then
    write_atomic "$V5_PROBE_PASSED" "attempt 2 passed with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (exit 0; attempt 1: $why); ft row $(log_ft_row "$V5_OUT/probe-attempt2.log")"
    say "v5 probe passed on attempt 2 (expandable_segments)"
    return 0
  fi
  why="$why; attempt 2 (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) exit $rc: $(v5_last_line "$V5_OUT/probe-attempt2.log")"
  write_atomic "$V5_PROBE_FAILED" "$why" || return 3
  say "v5 probe FAILED twice ($why). $V5_PROBE_FAILED says so; no seed starts on either lane until the human writes $V5_PROBE_YES"
  return 1
}

# Lane 1's wait for lane 0's probe: until v5.probe-passed or v5.probe-failed. Refuses when lane 0
# is not running (never queued, or ended) without either, on V5_STOP, and at V5_HUMAN_WAIT_MAX_S.
# A failed probe then waits for the human's V5_PROBE_YES, as lane 0 does.
v5_wait_probe() {
  local t0
  t0=$(date +%s)
  until [ -e "$V5_PROBE_PASSED" ] || [ -e "$V5_PROBE_FAILED" ]; do
    if [ ! -e "$Q/v5lane0.queued" ] || [ -e "$Q/v5lane0.done" ]; then
      say "v5 lane 1: lane 0 is not running and wrote no probe result; the probe runs only on lane 0 (GPU 0); refusing"
      return 3
    fi
    if [ -e "$V5_STOP" ]; then say "$V5_STOP exists: the probe's result not waited for further"; return 3; fi
    if [ $(( $(date +%s) - t0 )) -ge "$V5_HUMAN_WAIT_MAX_S" ]; then
      say "v5 lane 1: no probe result within ${V5_HUMAN_WAIT_MAX_S}s; refusing"; return 3
    fi
    sleep 60
  done
  if [ -e "$V5_PROBE_FAILED" ]; then
    say "v5 lane 1: the probe failed ('$(head -c 300 "$V5_PROBE_FAILED" | tr '\n' ' ')'); waiting for $V5_PROBE_YES"
    v5_wait_marker "$V5_PROBE_YES" "the human's yes after the failed probe" || return 3
  fi
  say "v5 lane 1: the probe's result: '$(head -c 300 "${V5_PROBE_PASSED}" 2>/dev/null | tr '\n' ' ')'"
}

# A lane with nothing it can take yet sleeps and picks again, at most V5_IDLE_MAX_S in a row. It
# logs why it waits when the reason changes, not on every pick.
v5_idle() {
  if [ -z "$V5_IDLE_SINCE" ]; then V5_IDLE_SINCE=$(date +%s); fi
  if [ "$1" != "$V5_IDLE_WHY" ]; then
    V5_IDLE_WHY=$1
    say "v5 lane $V5_GPU: $1; waiting"
  fi
  if [ $(( $(date +%s) - V5_IDLE_SINCE )) -ge "$V5_IDLE_MAX_S" ]; then
    say "v5 lane $V5_GPU: idle for ${V5_IDLE_MAX_S}s ($1): it waits on a lane that has stopped; refusing"
    return 3
  fi
  sleep 60
}

# The lane's loop (box_q_v5.sh's rule): take the lane's lock (it waits for this lane's post-seed
# waiter), then under the scheduling mutex make any decision that is due (verified first), or
# check the budget of the earliest ready job, verify and claim it, release the mutex and run the
# job; until nothing is left. Returns 0 when nothing is left, 3 when the lane stops (V5_STOP, a hold not lifted, a check
# that fails, or an idle past its bound).
v5_lane_loop() {
  local next j rc msg
  while :; do
    v5_lock || return 3
    v5_sched_lock
    next=$(v5_next_job)
    case "$next" in
      "decide "*)
        V5_IDLE_SINCE=""
        V5_IDLE_WHY=""
        if ! v5_verify; then v5_unlock_all; say "v5 lane $V5_GPU: verification failed before the $next; this lane stops"; return 3; fi
        if ! v5_decide "${next#decide }"; then v5_unlock_all; say "v5 lane $V5_GPU: the $next failed; this lane stops"; return 3; fi
        v5_unlock_all ;;
      "job "*)
        j=${next#job }
        msg=$(v5_job_budget "$j")
        rc=$?
        # a wait for the budget is logged once per idle reason, not on every pick
        if [ -n "$msg" ] && { [ "$rc" != 4 ] || [ "$V5_IDLE_WHY" != "$j waits for the budget" ]; }; then
          printf '%s\n' "$msg"
        fi
        case "$rc" in
          0)
            V5_IDLE_SINCE=""
            V5_IDLE_WHY=""
            # everything is checked again right before a claim, never a stale check from the start
            if ! v5_verify; then v5_unlock_all; say "v5 lane $V5_GPU: verification failed before $j; this lane stops"; return 3; fi
            write_atomic "$Q/v5job-$j.claimed" "lane $V5_GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" || { v5_unlock_all; return 3; }
            exec 8>&-
            v5_run_job "$j"
            exec 9>&- ;;
          3) v5_job_refuse "$j" budget; v5_unlock_all ;;
          *) v5_unlock_all; v5_idle "$j waits for the budget" || return 3 ;;
        esac ;;
      hold)
        v5_unlock_all
        V5_IDLE_SINCE=""
        V5_IDLE_WHY=""
        v5_wait_marker "$V5_CONTINUE" "R9's hold" || { say "R9: no continue: v5 seeds 1-4 NOT RUN"; return 3; } ;;
      wait) v5_unlock_all; v5_idle "nothing is ready" || return 3 ;;
      stop)
        v5_unlock_all
        if v5_r9_holding; then say "R9: no continue: v5 seeds 1-4 NOT RUN"; fi
        say "v5 lane $V5_GPU: $V5_STOP exists: no new job; this lane stops"
        return 3 ;;
      none) v5_unlock_all; say "v5 lane $V5_GPU: no job is left that this lane could take"; return 0 ;;
      *) v5_unlock_all; say "v5 lane $V5_GPU: the rule printed '$next'; refusing"; return 3 ;;
    esac
  done
}
