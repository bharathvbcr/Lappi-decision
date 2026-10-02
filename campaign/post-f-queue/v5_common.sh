# shellcheck shell=bash
# shellcheck disable=SC2034  # the names below are read by the waiters that source this file
# Sourced, after post_f_common.sh and idle_common.sh, by the v5 block's waiters: box_q_v5.sh,
# box_q_v5s34.sh, box_q_v5nw.sh, box_q_v5j5.sh and the per-seed box_q_v5traj.sh. Defines no job
# and starts nothing.
#
# Binding text: campaign/v5-preregistered.json, the DRAFT (campaign/v5-preregistered.DRAFT.json)
# renamed on main without its top-level `draft` key. This file reads only the renamed one, from
# the pinned v5 lane; the rules binary refuses a file that still says draft. Block order
# (launch.projected_gpu_hours.slot, build_order step 7, the human's answer 1 at 0b559bb):
#   v5 seeds 0-2 (R9 may hold after seed 0) -> seeds 3-4 iff seeds34 fires -> the noul-weight
#   arm x3 iff v5nw.room is room or V5NW_HUMAN_YES is pinned -> J5' x3.
# Each seed of v5, seeds 3-4 and the arm is followed by its own post-seed waiter
# (box_q_v5traj.sh, markers <run>traj-s<N>.*): the trajectory-ood rows under gpu.lock, then the
# CPU letter/option controls outside it. Its .done never gates the next seed.
#
# Decisions are never made here: v5-pause (R9), v5-noulw --room, seeds34 and v5-noulw are
# qd-post-f-rules subcommands (HANDOFF/v5-rules-2026-10-02.md), run from a NEW binary pinned by
# sha256 and verified before EVERY call. post_f_common.sh's rule() is not used: it runs the old
# cadbdc74 build, which has none of these subcommands, and maps continue/pause/room/no_room/wins
# to refused. /home/ubuntu/bin/qd-post-f-rules (cadbdc74) is not replaced.
#
# THE UNSET VALUES BELOW ARE FAIL-CLOSED PINS, NOT PLACEHOLDERS. While any is the literal UNSET,
# every v5 waiter logs "deferred: pins UNSET" and exits 3 without touching gpu.lock. The lead
# fills them at deploy time (HANDOFF/v5-queue-2026-10-02.md, "Deploy"), stages the filled copy
# and records its sha256. Never edit this file while a waiter that sources it runs.

# The v5 rules build (L-v5-rules, cross-built as HANDOFF/post-f-queue-2026-10-01.md "The binary"
# describes) at a new path.
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
# The date in the v5 ledgers' names (seeds.ledger: gh200-v5-<date>.jsonl), YYYY-MM-DD.
V5_DATE=UNSET
# The Mac prelude's record (v5_prelude_mac.py --record), copied by the lead, and its sha256 as
# the prelude printed it ("RECORD ... sha256").
V5_PRELUDE_SHA256=UNSET
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
V5_LEDGER=/home/ubuntu/ledger/gh200-v5-$V5_DATE.jsonl
V5NW_LEDGER=/home/ubuntu/ledger/gh200-v5-noulw-$V5_DATE.jsonl
DEC_V5=/home/ubuntu/ledger/v5-decisions-$V5_DATE
V5_OUT=/home/ubuntu/v5
V5_CKPT=/home/ubuntu/ckpt/v5
V5NW_OUT=/home/ubuntu/v5-noulw
V5NW_CKPT=/home/ubuntu/ckpt/v5-noulw
V5_LOGS=/home/ubuntu/logs
# Markers. The human's: V5_LAUNCH_YES (the launch, human answer 1 at d24c865), V5_CONTINUE /
# V5_STOP (R9's hold), V5NW_HUMAN_YES (the arm on no_room, R7), V5_OVER_BUDGET_YES (a run past
# the approved total). Each is a file the lead writes with the human's words; an empty file is
# not a yes. The waiters': v5.paused (R9 held), and each decision's word, written once.
V5_LAUNCH_YES=$Q/V5_LAUNCH_YES
V5_CONTINUE=$Q/V5_CONTINUE
V5_STOP=$Q/V5_STOP
V5NW_HUMAN_YES=$Q/V5NW_HUMAN_YES
V5_OVER_BUDGET_YES=$Q/V5_OVER_BUDGET_YES
V5_PAUSED=$Q/v5.paused
V5S34_WORD=$Q/v5s34.word
V5NW_ROOM=$Q/v5nw.room
V5NW_WORD=$Q/v5nw.word
# One line per GPU step: UTC, label, wall seconds, USD at COST's rate. Appended only.
V5_SPEND=$Q/v5.spend
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
# Bounds on waits that are not a predecessor's marker: a human marker, and a post-seed waiter's
# start once the parent released gpu.lock to it.
V5_HUMAN_WAIT_MAX_S=259200
V5_TRAJ_HANDOFF_MAX_S=900
# A trajectory invocation is not started with less than this left of the seed's 3,600 s.
V5_TRAJ_MIN_LEFT_S=120
# The ledger read race (Fable's ruling B, 2026-10-02; v5_read_race): a reading the binary refused
# on a half-written LAST ledger line is made again, at most V5_READ_TRIES attempts in all (the
# first included), V5_READ_RETRY_S apart: at most 4 x 10 s = 40 s of waiting per reading. Then
# it is refused, as now. Never gated on a post-seed waiter's .done instead: that waiter releases
# gpu.lock before its CPU controls, so waiting on it would idle the GPU for up to 5,400 s.
V5_READ_TRIES=5
V5_READ_RETRY_S=10
# The human's yes on the v5 block (launch.human_yes); every run also carries post_f_common.sh's
# rule-4 lifting through approved().
V5_HUMAN_YES="Bharath (human, answer 1 at d24c865, AUDIT/v5-plan-2026-10-02/human-answers-v5-review.md: 'Yes, launch when ready (Recommended)' on ~\$137 / ~60 GPU-h, +\$32 / +14 h with seeds 3-4, with R9's seed-0 pause and R7's skip)"
# The split flags a v5 data argv may carry: data only, never a recipe, run or scoring flag.
V5_SPLIT_FLAGS=" --out --no-repo-history --rev --defect-class --defect-download --defect-max-rows --defect-noul --general-record --general-max-rows --exclude-identity-keys --real-backbone --commitpackft --max-pairs "

v5_approved() { echo "$V5_HUMAN_YES; $(approved "$1")"; }

# The names of the pins still UNSET (space-separated), or nothing.
v5_pins_unset() {
  local v out=""
  for v in RULES_V5_SHA256 V5_PREP_BIN_SHA256 V5_LANE_AT V5_PREREG_SHA256 V5_DATE \
           V5_PRELUDE_SHA256 V5_C1 V5_C2A V5_C2B V5_LOWER V5_LRSET; do
    [ "${!v}" = UNSET ] && out="$out $v"
  done
  if [ "${#V5_SPLIT[@]}" -eq 0 ] || [ "${V5_SPLIT[0]}" = UNSET ]; then out="$out V5_SPLIT"; fi
  echo "$out"
}

# Each filled pin has its form, and each decision one of its words. Prints one line per
# problem, or nothing.
v5_pins_check() {
  local v
  for v in RULES_V5_SHA256 V5_PREP_BIN_SHA256 V5_PREREG_SHA256 V5_PRELUDE_SHA256; do
    echo "${!v}" | grep -Eqx '[0-9a-f]{64}' || echo "$v is '${!v}', not 64 lower-case hex"
  done
  echo "$V5_LANE_AT" | grep -Eqx '[0-9a-f]{40}' || echo "V5_LANE_AT is '$V5_LANE_AT', not a full 40-hex commit"
  echo "$V5_DATE" | grep -Eqx '20[0-9]{2}-[01][0-9]-[0-3][0-9]' || echo "V5_DATE is '$V5_DATE', not YYYY-MM-DD"
  for v in V5_C1 V5_C2A V5_C2B; do
    case "${!v}" in on|off) ;; *) echo "$v is '${!v}', not on or off" ;; esac
  done
  case "$V5_LOWER" in keep|drop) ;; *) echo "V5_LOWER is '$V5_LOWER', not keep or drop" ;; esac
  case "$V5_LRSET" in f|j6dv4) ;; *) echo "V5_LRSET is '$V5_LRSET', not f or j6dv4" ;; esac
}

# V5_RECIPE: F's recipe flags as ft row 973cd4e3 records them (recipe.base), with C3/R4's lower
# layers and learning rate as pinned, plus recipe.added's --min-lr 0 and --batch-order seed (Fable's
# seed-order ruling: every v5 run, seeds 3-4, the arm and J5' carry it), plus C1/C2a/C2b as pinned.
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

# The renamed pre-registration: no draft key, and the numbers this queue prints, read from it.
# Sets V5_EST_SEED_USD/H (a v5, seeds 3-4 or arm seed), V5_EST_J5_USD/H (a J5' seed),
# V5_APPROVED_USD/H, V5_S34_USD/H and V5NW_W (arm_noul_weight.w.value). The per-seed figures are
# launch.projected_cost_usd.check's, cross-checked against the block totals and COST's rate;
# any disagreement refuses.
v5_read_prereg() {
  local out
  if [ "${COST[2]}" != --usd-per-hour ]; then say "COST is '${COST[*]}'; no rate to price with"; return 3; fi
  V5_USD_PER_HOUR=${COST[3]}
  out=$("$PY" - "$V5_PREREG" "$V5_USD_PER_HOUR" <<'PYEOF'
import json, math, re, sys

path, rate = sys.argv[1], float(sys.argv[2])


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
    s34_usd, s34_h = float(usd["if_seeds_3_4"]), float(hours["if_seeds_3_4"])
    blocks = float(usd["v5_seeds_0_2"]), float(usd["j5prime_x3"]), float(usd["noul_weight_x3"])
    w = d["arm_noul_weight"]["w"]["value"]
except (KeyError, TypeError, ValueError, IndexError) as exc:
    refuse(f"the cost or weight text is not the form this queue reads ({exc!r})")
if type(w) not in (int, float) or not math.isfinite(w) or w <= 0:
    refuse(f"arm_noul_weight.w.value is {w!r}")
checks = {
    "3 x v5 seed = v5_seeds_0_2": abs(3 * seed_usd - blocks[0]) <= 0.15,
    "3 x J5' seed = j5prime_x3": abs(3 * j5_usd - blocks[1]) <= 0.15,
    "3 x v5 seed = noul_weight_x3": abs(3 * seed_usd - blocks[2]) <= 0.15,
    "the three blocks = total": abs(sum(blocks) - total_usd) <= 0.15,
    "2 x v5 seed = if_seeds_3_4": abs(2 * seed_usd - s34_usd) <= 0.15,
    "6 x v5 seed h + 3 x J5' seed h = total h": abs(6 * seed_h + 3 * j5_h - total_h) <= 0.05,
    "2 x v5 seed h = if_seeds_3_4 h": abs(2 * seed_h - s34_h) <= 0.05,
    "v5 seed h x rate = v5 seed $": abs(seed_h * rate - seed_usd) <= 0.1,
    "J5' seed h x rate = J5' seed $": abs(j5_h * rate - j5_usd) <= 0.1,
}
bad = [k for k, ok in checks.items() if not ok]
if bad:
    refuse(f"its cost figures disagree with each other: {bad}")
print(seed_usd, seed_h, j5_usd, j5_h, total_usd, total_h, s34_usd, s34_h, w)
PYEOF
)
  case "$out" in
    REFUSED:*|"") say "${out:-the pre-registration read printed nothing}"; return 3 ;;
  esac
  read -r V5_EST_SEED_USD V5_EST_SEED_H V5_EST_J5_USD V5_EST_J5_H V5_APPROVED_USD V5_APPROVED_H \
    V5_S34_USD V5_S34_H V5NW_W <<< "$out"
  say "v5 pre-registration: a v5/arm seed ~ ${V5_EST_SEED_H} h, \$${V5_EST_SEED_USD}; a J5' seed ~ ${V5_EST_J5_H} h, \$${V5_EST_J5_USD}; approved ~ \$${V5_APPROVED_USD} / ${V5_APPROVED_H} GPU-h (+\$${V5_S34_USD} / +${V5_S34_H} h iff seeds 3-4 run); arm --noul-weight ${V5NW_W}"
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

# Everything checked once the chain has ended and before gpu.lock: the binaries, the lane, the
# pre-registration and its numbers, the data (its train header, and no held-out split in it),
# the tokenizer, the prelude record. Leaves the shell in the lane, with QD_PREP_BIN re-exported.
# Returns 3 with the reason logged.
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

# gpu.lock on fd 9, as every queue waiter holds it.
v5_lock() {
  exec 9>"$Q/gpu.lock"
  flock 9
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
# R9 (readings.R9_pause_after_seed_0): continue -> run seeds 1-2; pause -> hold until V5_CONTINUE;
# refused, unknown:<word> and no-seed-0-row -> the same hold (this queue's fail-closed reading;
# the DRAFT names only pause).
v5_pause_action() { case "$1" in continue) echo run ;; *) echo hold ;; esac; }
# The arm's launch (arm_noul_weight.launch_condition, R7): room -> run; no_room -> run only with
# V5NW_HUMAN_YES pinned (the second argument "yes"), else skip; refused -> skip even with the
# yes, because the arm's own reading decides room from the same envelope first, so a refused
# room makes every arm row read refused.
v5_room_action() {
  case "$1:$2" in room:*) echo run ;; no_room:yes) echo run ;; *) echo skip ;; esac
}
# Seeds 3-4 (seeds.seeds_3_4): fires -> run; quiet -> skip; anything else -> notrun.
v5_s34_action() { case "$1" in fires) echo run ;; quiet) echo skip ;; *) echo notrun ;; esac; }
# The arm's reading (arm_noul_weight.outcomes): every word is recorded; none starts anything.
v5_noulw_action() { case "$1" in wins|quiet) echo record ;; *) echo record-refused ;; esac; }

# --- runs -----------------------------------------------------------------------------------
# v5_run_vars RUN: the output dir, checkpoint dir, ledger and recipe additions of a run kind;
# v5 (v5, seeds 3-4, J5') or v5nw (the arm: --noul-weight, arm_noul_weight.what). 3 otherwise.
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
v5_spend_add() {
  local usd
  usd=$(awk -v s="$2" -v r="$V5_USD_PER_HOUR" 'BEGIN { printf "%.2f", s * r / 3600 }')
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$usd" >> "$V5_SPEND"
  v5_total_line
}

# "<GPU-h> <USD>" spent so far by the v5 block (V5_SPEND; 0 0 before its first line).
v5_spent() {
  if [ ! -s "$V5_SPEND" ]; then echo "0 0"; return 0; fi
  awk -F'\t' -v r="$V5_USD_PER_HOUR" '{ s += $3 } END { printf "%.4f %.2f\n", s / 3600, s * r / 3600 }' "$V5_SPEND"
}

# The approved total: launch.projected_cost_usd.total / projected_gpu_hours.total, plus
# if_seeds_3_4 once seeds34 has fired.
v5_approved_now() {
  if [ "$(head -c 16 "$V5S34_WORD" 2>/dev/null | tr -d '\n')" = fires ]; then
    awk -v a="$V5_APPROVED_USD" -v b="$V5_S34_USD" -v c="$V5_APPROVED_H" -v d="$V5_S34_H" 'BEGIN { printf "%.2f %.2f\n", a + b, c + d }'
  else
    echo "$V5_APPROVED_USD $V5_APPROVED_H"
  fi
}

v5_total_line() {
  local sh su au ah
  read -r sh su <<< "$(v5_spent)"
  read -r au ah <<< "$(v5_approved_now)"
  say "v5 running total: ${sh} GPU-h, \$${su} of the approved ~\$${au} / ${ah} GPU-h"
}

# v5_budget_ok EST_USD EST_H LABEL: refuse a run whose estimate would carry the running total past
# the approved total, unless V5_OVER_BUDGET_YES is pinned (the lead's instruction, 2026-10-02).
v5_budget_ok() {
  local eu=$1 eh=$2 label=$3 sh su au ah over
  read -r sh su <<< "$(v5_spent)"
  read -r au ah <<< "$(v5_approved_now)"
  over=$(awk -v su="$su" -v eu="$eu" -v au="$au" -v sh="$sh" -v eh="$eh" -v ah="$ah" \
    'BEGIN { print ((su + eu > au + 0.005 || sh + eh > ah + 0.0005) ? 1 : 0) }')
  if [ "$over" = 1 ]; then
    if [ -s "$V5_OVER_BUDGET_YES" ]; then
      say "$label: \$${su} + ~\$${eu} / ${sh} + ~${eh} GPU-h crosses the approved ~\$${au} / ${ah} GPU-h; $V5_OVER_BUDGET_YES says '$(head -c 300 "$V5_OVER_BUDGET_YES" | tr '\n' ' ')', so it runs"
      return 0
    fi
    say "$label REFUSED: \$${su} spent + ~\$${eu} / ${sh} + ~${eh} GPU-h would cross the approved ~\$${au} / ${ah} GPU-h; NOT RUN (the human's words in $V5_OVER_BUDGET_YES let it run)"
    return 3
  fi
}

v5_cost_line() {
  say "cost: $1: cap $2 s = \$$(awk -v s="$2" -v r="$V5_USD_PER_HOUR" 'BEGIN { printf "%.2f", s * r / 3600 }') at \$${V5_USD_PER_HOUR}/h; pre-registration estimate $3"
}

# v5_train_seed RUN SEED LABEL: one training run in v5's form (recipe.base + added; the arm adds
# its weight): train + score (--score-val --needle --ood) with the checkpoint and the retained
# towers, under the training cap. The caller holds gpu.lock. Sets V5_FT to the ft row the log
# names (empty when none). Refuses a seed whose log exists, or whose estimate would cross the
# approved total.
v5_train_seed() {
  local run=$1 seed=$2 label=$3 log t0 rc
  V5_FT=""
  v5_run_vars "$run" || return 3
  log=$V5R_OUT/train-s$seed.log
  if [ -e "$log" ]; then say "$label: $log exists; refusing to run it twice"; return 3; fi
  v5_budget_ok "$V5_EST_SEED_USD" "$V5_EST_SEED_H" "$label" || return 3
  mkdir -p "$V5R_OUT" "$V5R_CKPT" || return 3
  v5_cost_line "$label train+score" "$V5_TRAIN_CAP_S" "~ ${V5_EST_SEED_H} h, \$${V5_EST_SEED_USD} for the seed with its needle control and trajectory"
  say "$label at $(git rev-parse --short HEAD): ${V5_RECIPE[*]} ${V5R_EXTRA[*]} --seeds $seed -> $V5R_LEDGER"
  t0=$(date +%s)
  timeout $((V5_TRAIN_CAP_S + 1800)) "$PY" -u tools/real_ft_run.py "${V5_SPLIT[@]}" "${V5_RECIPE[@]}" "${V5R_EXTRA[@]}" \
    --devices cuda --seeds "$seed" \
    --checkpoint-dir "$V5R_CKPT" --checkpoint-every "$V5_CKPT_EVERY" --retain-tower-every "$V5_RETAIN_EVERY" \
    --score-val --needle --ood --ood-general-record "$REC" \
    --verdicts-out "$V5R_OUT/verdicts-s$seed.jsonl" --suite-verdicts-out "$V5R_OUT/suite-verdicts-s$seed.jsonl" \
    --wall-clock-cap-s "$V5_TRAIN_CAP_S" "${COST[@]}" --ledger "$V5R_LEDGER" \
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
# holds gpu.lock.
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
    "${COST[@]}" --wall-clock-cap-s "$V5_NEEDLE_CAP_S" \
    --ledger "$V5R_LEDGER" --suite-verdicts-out "$V5R_OUT/needle-control-s$seed.jsonl" \
    2>&1 | tee "$log"
  say "$label needle control done (exit $?)"
  v5_spend_add "$label needle control" $(( $(date +%s) - t0 ))
}

# v5_post_seed RUN SEED FT: start the seed's own post-seed waiter (box_q_v5traj.sh: trajectory
# rows under gpu.lock, then the CPU controls), release gpu.lock to it, and wait at most
# V5_TRAJ_HANDOFF_MAX_S for it to take the lock (its .started) or end (.done), so the next seed
# cannot take the lock first. Its outcome never gates the next seed.
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
    setsid nohup bash "$PF/box_q_v5traj.sh" "$run" "$seed" "$ft" \
      > "$V5_LOGS/q-${run}traj-s$seed.log" 2>&1 < /dev/null 9>&- &
    say "${run}traj-s$seed started (log $V5_LOGS/q-${run}traj-s$seed.log)"
  fi
  exec 9>&-
  t0=$(date +%s)
  until [ -f "$m.started" ] || [ -f "$m.done" ]; do
    if [ $(( $(date +%s) - t0 )) -ge "$V5_TRAJ_HANDOFF_MAX_S" ]; then
      say "${run}traj-s$seed did not take gpu.lock within ${V5_TRAJ_HANDOFF_MAX_S}s; the queue goes on without waiting for it"
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
