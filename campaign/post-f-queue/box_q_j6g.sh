#!/bin/bash
# Idle queue j6g. Sources:
#   - Fable's 2026-10-02 optimize ruling, Q1(b), Q3 and "J6(g) pre-registration"
#     (AUDIT/fable-optimize-2026-10-02/fable-optimize-ruling.md);
#   - the reading rule, campaign/j6g-preregistered.json (dec48d8);
#   - the human's yes on the rental hours (AUDIT/fable-optimize-2026-10-02/human-decisions.md
#     item 1, 8a78056).
# J6(g) is F's recipe plus --option-permutation-seed 20260919 (train-time option permutation),
# seed 0, on F's own v4 data. It is quick by construction (one seed). It is the LAST item:
# rung0 -> cudadev -> rungd -> fsucc -> j5pp -> j6a -> j6g -> STOP. Nothing goes after it.
# Its reading is `qd-post-f-rules j6g` (lane L-j6g) and is NOT wired in here. The lead reads it
# by hand after the rows exist. No needle-control or letter-control row is required
# (arm.identity, envelope), so none is run: the human's yes covers the ~5 h of the training run.
#
# Order (Fable Q3): it waits for j5pp.done, then for j6a if j6a is queued (wait_queued j6a),
# then for gpu.lock. It does not wait on j6a.done alone, because j6a.queued is absent while
# j6a's waiter is not launched (its pins are UNSET). Every waiter touches its .queued first
# thing, so LAUNCH ORDER MATTERS: launch this waiter only after box_q_j6a.sh is launched
# (j6a.queued exists), or once the lead has decided j6a will not run. A j6g launched before j6a
# would see no j6a.queued once j5pp.done exists and take the GPU first, against the human's
# "after j6a". The waiter logs which case it is in.
#
# It runs only if every pin below checks (exit 3 if not):
#   - none is the literal UNSET (checked first, before any wait: a waiter launched with an UNSET
#     pin could never run, so it says so at once);
#   - the prelude record exists, its sha256 is its pin, and its fields are the
#     pre-registration's (seed 20260919, no refusal, every choice row permuted, F's 9,683
#     batches at width 7,936, F's shard and data hashes, the box's tokenizer, exactly one
#     recipe key added);
#   - the box's tokenizer.json is TOKENIZER_SHA256 (post_f_common.sh), the file the prelude
#     hashed; F's train header names the record's shard and data hashes;
#   - J6(g)'s ledger and train.log do not exist yet.
# Whatever stops it is logged, and the EXIT trap touches .done.
#
# THE VALUES BELOW ARE FAIL-CLOSED PINS, NOT PLACEHOLDERS.
# While any of them is the literal UNSET, this script logs "j6g deferred: pins UNSET" and
# exits 3, and it touches neither gpu.lock nor the GPU.
#   J6G_PRELUDE          the Mac prelude's record (campaign/post-f-queue/j6g_prelude_mac.py
#                        --record), copied to the box by the lead after it ran;
#   J6G_PRELUDE_SHA256   that record's sha256, as the prelude printed it ("RECORD ... sha256");
#   J6G_HUMAN_HOURS_YES  the human's yes on the rental hours, in their words.
# The lead fills the first two from the prelude's output, stages the filled copy and records its
# sha256. Never edit this file while its waiter runs: bash reads a script as it executes it.
J6G_PRELUDE=UNSET
J6G_PRELUDE_SHA256=UNSET
J6G_HUMAN_HOURS_YES="Bharath (human, in chat 2026-10-02 ~05:15 UTC: 'Yes, after j6a (Recommended)'; AUDIT/fable-optimize-2026-10-02/human-decisions.md item 1)"
#
# Fixed by the pre-registration (campaign/j6g-preregistered.json arm):
#   - option_permutation.seed 20260919; ledger arm.ledger; seed 0;
#   - F's data and recipe (post_f_common.sh F_SPLIT and F_RECIPE, with F's skip 6), so the ft
#     row's recipe is F seed 0's (973cd4e3) plus exactly option_permutation_seed;
#   - F seed 0's train shard hash and data snapshot (973cd4e3), which the record must name.
OPTION_PERMUTATION_SEED=20260919
J6G_LEDGER=/home/ubuntu/ledger/gh200-j6g-v4-2026-10-02.jsonl
F_SHARD_HASH=8bcf56ad89bbf059f9926035b7b798796a0a41dd4960c2939b66671f24d2fb80
F_DATA_SNAPSHOT_HASH=ea3215c4f36d57f74d291fb94c3fa8724fa5a14a303ea7572aa0dafb2a0933a1
# It runs from qd-lane8 at a502670, F's exact path, untouched: the flag exists there
# (tools/real_ft_run.py:7791) and a502670 validates every choice row's option block over the
# whole plan before train_ft (:2220-2231, after the tower loads).
# Cap: 32,400 s ($20.61; the human's 2026-10-01 yes, via approved()). Holds gpu.lock.
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j6g.done' EXIT
touch $Q/j6g.queued
UNSET_PINS=""
for v in J6G_PRELUDE J6G_PRELUDE_SHA256 J6G_HUMAN_HOURS_YES; do
  [ "${!v}" = UNSET ] && UNSET_PINS="$UNSET_PINS $v"
done
if [ -n "$UNSET_PINS" ]; then
  say "j6g deferred: pins UNSET:$UNSET_PINS (fail-closed until the lead runs the Mac prelude and fills them); J6(g) NOT RUN"
  exit 3
fi
until [ -f $Q/j5pp.done ]; do sleep 60; done
if [ -f $Q/j6a.queued ]; then
  say "j6g: j5pp is done; j6a is queued, so j6g waits for j6a.done"
else
  say "j6g: j5pp is done; j6a is NOT queued (its waiter was not launched), so j6g does not wait for it"
fi
wait_queued j6a
for f in "$J6G_PRELUDE" "$BACKBONE/tokenizer.json" "${F_SPLIT[1]}/shards/train/header.json"; do
  if [ ! -f "$f" ]; then say "j6g deferred: $f is absent; J6(g) NOT RUN"; exit 3; fi
done
pin "$J6G_PRELUDE" "$J6G_PRELUDE_SHA256" || { say "j6g deferred: the prelude record is not the pinned one; J6(g) NOT RUN"; exit 3; }
pin "$BACKBONE/tokenizer.json" "$TOKENIZER_SHA256" || { say "j6g deferred: the box's tokenizer.json is not the one the prelude checked; J6(g) NOT RUN"; exit 3; }
if ! "$PY" -c '
import json, sys
rec = json.load(open(sys.argv[1], encoding="utf-8"))
hdr = json.load(open(sys.argv[2], encoding="utf-8"))
seed, shard, data, tok = int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
want = {
    "ok": True,
    "code_commit": "a5026707b6e3e57c253be32003ba32420f2d78e2",
    "option_permutation_seed": seed,
    "permutation_refusals": 0,
    "plan_batches": 9683,
    "plan_max_width": 7936,
    "shard_hash": shard,
    "data_snapshot_hash": data,
    "tokenizer_json_sha256": tok,
    "recipe_piece_delta": {"option_permutation_seed": seed},
    "recipe_piece_removed": [],
}
wrong = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v}
permuted = rec.get("choice_rows_permuted_per_pass")
if not (type(permuted) is int and permuted >= 1 and permuted == rec.get("choice_rows_in_plan")):
    wrong["choice_rows_permuted_per_pass"] = permuted
if hdr.get("shard_hash") != shard or hdr.get("data_snapshot_hash") != data:
    wrong["box_train_header"] = [hdr.get("shard_hash"), hdr.get("data_snapshot_hash")]
sentence = rec.get("sentence")
print(f"prelude record: {sentence!r}; mismatches: {wrong or None}")
sys.exit(1 if wrong else 0)
' "$J6G_PRELUDE" "${F_SPLIT[1]}/shards/train/header.json" "$OPTION_PERMUTATION_SEED" \
    "$F_SHARD_HASH" "$F_DATA_SNAPSHOT_HASH" "$TOKENIZER_SHA256"; then
  say "j6g deferred: the prelude record is not the pre-registration's (seed, refusals, batches, width, hashes, recipe delta) or F's train header disagrees; J6(g) NOT RUN"
  exit 3
fi
if [ -e "$J6G_LEDGER" ]; then
  say "j6g: $J6G_LEDGER exists; J6(g)'s rows go into a new ledger (R2), refusing"; exit 3
fi
exec 9>$Q/gpu.lock
flock 9
touch $Q/j6g.started
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
OUT=/home/ubuntu/j6g-v4
CKPT=/home/ubuntu/ckpt/j6g-v4
if [ -e "$OUT/train.log" ]; then say "j6g: $OUT/train.log exists; refusing to run twice"; exit 3; fi
if [ -e "$J6G_LEDGER" ]; then say "j6g: $J6G_LEDGER appeared while waiting for the lock; refusing"; exit 3; fi
mkdir -p "$OUT" "$CKPT"
# F's argv (box_q_f.sh, through F_SPLIT and F_RECIPE) plus the one flag, seed 0.
J6G_ARGV=("${F_SPLIT[@]}" "${F_RECIPE[@]}" --option-permutation-seed "$OPTION_PERMUTATION_SEED"
          --devices cuda --seeds 0
          --checkpoint-dir "$CKPT" --checkpoint-every 100000
          --score-val --needle --ood --ood-general-record "$REC"
          --verdicts-out "$OUT/verdicts.jsonl" --suite-verdicts-out "$OUT/suite-verdicts.jsonl"
          --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$J6G_LEDGER"
          --approved-by "$(approved "J6(g) option permutation seed 0 (idle queue j6g; rental hours: $J6G_HUMAN_HOURS_YES)")")
say "j6g (J6(g), F + --option-permutation-seed $OPTION_PERMUTATION_SEED, seed 0) at $(git rev-parse --short HEAD): data ${F_SPLIT[1]}, ledger $J6G_LEDGER"
timeout 34200 "$PY" -u tools/real_ft_run.py "${J6G_ARGV[@]}" 2>&1 | tee "$OUT/train.log"
say "J6(g) train+score done (exit $?)"
grep -F 'Option permutation' "$OUT/train.log" || say "J6(g): train.log carries no 'Option permutation' line (a502670 writes the sentence into the ft row's notes, not stdout)"
FT_ROW=$(log_ft_row "$OUT/train.log")
if [ -z "$FT_ROW" ]; then
  say "J6(g): no ft row in $OUT/train.log; nothing to read"; exit 4
fi
say "J6(g) ft row $FT_ROW in $J6G_LEDGER; the lead reads it with qd-post-f-rules j6g"
say "j6g all done; the idle queue STOPS here"
