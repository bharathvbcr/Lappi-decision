#!/bin/bash
# Idle queue j6a. Sources:
#   - Fable's idle-queue ruling, Q1 item 3 and Q2;
#   - Fable's J6(a) +replay ruling, AUDIT/idle-gpu-queue-2026-10-02/fable-j6a-replay.md;
#   - the reading rule, campaign/j6a-preregistered.json.
# J6(a) is F's recipe plus prior-KL replay on build 2's v4 replay set, seed 0. It is quick by
# construction (one seed). After it, --needle-control 1024,2048,4096 runs on its checkpoint, as
# J6(f)-v4's did. This is the LAST item: nothing goes after it.
# Its reading is a future `qd-post-f-rules j6a` (lane L-room) and is NOT wired in here. The lead
# reads it by hand after the row exists.
#
# It runs only if BOTH:
#   - fsucc's word ($Q/fsucc.word) is quiet. fires:<arm>, refused, absent or malformed: deferred
#     to the re-plan (logged), exit 0;
#   - every pin below checks (exit 3 if not). The checks are: none is UNSET; the files exist;
#     the attestation's sha256 equals its pin; it says "clean": true, with zero hits on val and
#     heldout and every replay row checked.
#   Then the prelude must pass (exit 5 if not).
# Whatever stops it is logged, and the EXIT trap touches .done.
#
# THE FOUR VALUES BELOW ARE FAIL-CLOSED PINS, NOT PLACEHOLDERS.
# Lane L-replay builds and rsyncs the set: plan HANDOFF/replay-v4-plan-2026-10-02.md, which
# names /home/ubuntu/phase4-v4-replay-2026-10-02 with shards/replay and replay-attestation.json.
# The lead fills these from that lane's HANDOFF, before launch, and stages the filled copy.
# Never edit this file while its waiter runs: bash reads a script as it executes it.
# While any of them is the literal UNSET, this script logs "deferred: pins UNSET" and touches
# .done, and it trains nothing.
#   J6A_DATA           the train set's path: build 2's rsynced root, J6(a)'s --out. It must
#                      hold shards/train/header.json and data/pool/train-replay.json, it must
#                      have NO data/heldout (rule 3; the plan's rsync excludes it), and it must
#                      not be F's --out (the declared data delta);
#   REPLAY_DIR         --replay-shards: build 2's replay shard set (<J6A_DATA>/shards/replay);
#   ATTESTATION        --replay-attestation: decontam 2's attestation for that set;
#   ATTESTATION_SHA256 its sha256 (campaign/j6a-preregistered.json amendments_pending).
J6A_DATA=UNSET
REPLAY_DIR=UNSET
ATTESTATION=UNSET
ATTESTATION_SHA256=UNSET
#
# Fixed by the pre-registration (campaign/j6a-preregistered.json arm.replay_flags; Fable's J6(a)
# ruling, Q2):
#   - weight 1.0, every 6, direction base_to_model (the code's default direction);
#   - weight 1.0 is the unit choice. a502670's real_ft_run.py has no weight default (:7818-7819).
#     This answers L-replay's GAP-REPLAY-WEIGHT-HAS-NO-PINNED-VALUE-2026-10-02, which that lane owns.
#   - ledger: arm.ledger.
REPLAY_WEIGHT=1.0
REPLAY_EVERY=6
J6A_LEDGER=/home/ubuntu/ledger/gh200-j6a-v4-2026-10-02.jsonl
# The prelude (j6a_prelude_box.py, staged beside this script), pinned. It runs CPU only and
# niced. F's prelude took 249 s; this cap bounds a hung one.
PRELUDE=/home/ubuntu/post-f/j6a_prelude_box.py
PRELUDE_SHA256=415d09eb799b50e9d756ec1527dcf193b01c93c510634b97ff6673848ec96e70
PRELUDE_CAP_S=1800
#
# Recipe: F's (post_f_common.sh F_RECIPE, with F's skip) on build 2's root. It adds:
#   - --replay-partition (corpus_facts requires it when train-replay.json exists);
#   - --replay-shards and --replay-attestation;
#   - --replay-cache, a new .npz built once on the GPU;
#   - --replay-weight 1.0 and --replay-every 6.
# It runs from qd-lane8 at a502670, F's exact path, untouched: replay.py and replay_decontam.py
# are byte-identical between a502670 and main. The set must be built at a502670 (L-replay
# decision point 3: a502670's reader refuses headers written at main).
# Cap: 32,400 s ($20.61; the human's 2026-10-01 yes, via approved()), plus 5,400 s for the
# needle control. Holds gpu.lock.
# Queue: after j5pp (j5pp.done).
set -o pipefail
# shellcheck source=post_f_common.sh
source /home/ubuntu/post-f/post_f_common.sh || exit 3
# shellcheck source=idle_common.sh
source /home/ubuntu/post-f/idle_common.sh || exit 3
trap 'touch /home/ubuntu/queue/j6a.done' EXIT
touch $Q/j6a.queued
until [ -f $Q/j5pp.done ]; do sleep 60; done
UNSET_PINS=""
for v in J6A_DATA REPLAY_DIR ATTESTATION ATTESTATION_SHA256; do
  [ "${!v}" = UNSET ] && UNSET_PINS="$UNSET_PINS $v"
done
WORD=$(read_succ_word)
if [ "$WORD" != quiet ]; then
  say "j6a DEFERRED to the re-plan: fsucc's word is '${WORD:-absent or not one of: $SUCC_WORDS}', not quiet (Fable Q2: j6a runs only if fsucc was quiet)"
  [ -n "$UNSET_PINS" ] && say "j6a: also deferred: pins UNSET:$UNSET_PINS"
  exit 0
fi
if [ -n "$UNSET_PINS" ]; then
  say "j6a deferred: pins UNSET:$UNSET_PINS (fsucc was quiet; fail-closed until the lead fills them from L-replay's HANDOFF); J6(a) NOT RUN"
  exit 3
fi
if [ "$J6A_DATA" = "${F_SPLIT[1]}" ]; then
  say "j6a deferred: J6A_DATA is F's own --out ($J6A_DATA), not build 2's set; J6(a) NOT RUN"; exit 3
fi
if [ -e "$J6A_DATA/data/heldout" ]; then
  say "j6a deferred: $J6A_DATA/data/heldout exists; J6(a)'s data dir is rsynced without held-out data (rule 3); J6(a) NOT RUN"; exit 3
fi
for f in "$J6A_DATA/shards/train/header.json" "$J6A_DATA/data/pool/train-replay.json" "$REPLAY_DIR/header.json" "$ATTESTATION" "$PRELUDE"; do
  if [ ! -f "$f" ]; then say "j6a deferred: $f is absent; J6(a) NOT RUN"; exit 3; fi
done
pin "$ATTESTATION" "$ATTESTATION_SHA256" || { say "j6a deferred: the attestation is not the pinned one; J6(a) NOT RUN"; exit 3; }
pin "$PRELUDE" "$PRELUDE_SHA256" || { say "j6a deferred: the prelude is not the pinned one; J6(a) NOT RUN"; exit 3; }
if ! "$PY" -c '
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))
clean = a.get("clean")
hits = a.get("hits")
total = a.get("replay_rows_total")
checked = a.get("replay_rows_checked")
short = a.get("replay_rows_too_short")
print(f"attestation clean={clean!r} hits={hits!r} replay_rows_checked={checked!r} "
      f"replay_rows_total={total!r} replay_rows_too_short={short!r}")
ok = (
    clean is True
    and isinstance(hits, dict)
    and {"val", "heldout"} <= set(hits)
    and all(type(v) is int and v == 0 for v in hits.values())
    and type(checked) is int
    and type(total) is int
    and checked >= 1
    and checked == total
)
sys.exit(0 if ok else 1)
' "$ATTESTATION"; then
  say "j6a deferred: the attestation is not CLEAN (it needs clean: true, hits 0 on val and heldout, replay_rows_checked == replay_rows_total >= 1) or is unreadable; J6(a) NOT RUN"
  exit 3
fi
exec 9>$Q/gpu.lock
flock 9
touch $Q/j6a.started
lane "$LANE8" "$LANE8_AT" || exit 3
f_skip_ok || exit 3
OUT=/home/ubuntu/j6a-v4
CKPT=/home/ubuntu/ckpt/j6a-v4
if [ -e "$OUT/train.log" ]; then say "j6a: $OUT/train.log exists; refusing to run twice"; exit 3; fi
mkdir -p "$OUT" "$CKPT"
# F's data argv with build 2's root and the replay partition (the needle control's too).
J6A_SPLIT=(--out "$J6A_DATA" --no-repo-history
           --rev 881ab304f15ea13529002391dda8520c2ea47af4
           --defect-class data/pool/commitpackft-composed-v1
           --defect-download data/pool/commitpackft
           --defect-noul data/pool/defect-noul-v3b
           --general-record "$REC" --general-max-rows 200000
           --real-backbone "$BACKBONE" --replay-partition)
J6A_ARGV=("${J6A_SPLIT[@]}" "${F_RECIPE[@]}" --devices cuda --seeds 0
          --checkpoint-dir "$CKPT" --checkpoint-every 100000
          --replay-shards "$REPLAY_DIR" --replay-attestation "$ATTESTATION"
          --replay-cache "$CKPT/replay-prior-cache.npz"
          --replay-weight "$REPLAY_WEIGHT" --replay-every "$REPLAY_EVERY"
          --score-val --needle --ood --ood-general-record "$REC"
          --verdicts-out "$OUT/verdicts.jsonl" --suite-verdicts-out "$OUT/suite-verdicts.jsonl"
          --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$J6A_LEDGER"
          --approved-by "$(approved "J6(a) +replay seed 0 (idle queue j6a)")")
# The prelude, on the training argv itself: a STOP gate (the plan's "Prelude"). It passes only
# if it exits 0, prints PRELUDE OK, and main printed its replay line for exactly this set,
# attestation, weight and cadence.
say "j6a prelude (CPU, nice 10, cap ${PRELUDE_CAP_S}s): real_ft_run.main on J6(a)'s argv, stopping at Ledger(args.ledger) after _replay_plan"
timeout --kill-after=30 "$PRELUDE_CAP_S" nice -n 10 "$PY" -u "$PRELUDE" "${J6A_ARGV[@]}" 2>&1 | tee "$OUT/prelude.log"
PRC=$?
EXPECT="batches from $REPLAY_DIR (attestation ${ATTESTATION_SHA256:0:16}), weight $REPLAY_WEIGHT, every $REPLAY_EVERY"
if [ "$PRC" -ne 0 ] || ! grep -q '^PRELUDE OK' "$OUT/prelude.log" \
   || ! grep -E '^replay: [1-9][0-9]* batches from ' "$OUT/prelude.log" | grep -qF -- "$EXPECT"; then
  say "j6a STOPPED: the prelude failed (exit $PRC; it must exit 0, print PRELUDE OK and main's 'replay: N $EXPECT' line); J6(a) NOT RUN; log $OUT/prelude.log"
  exit 5
fi
say "j6a prelude OK"
say "j6a (J6(a) +replay, seed 0) at $(git rev-parse --short HEAD): data $J6A_DATA, replay $REPLAY_DIR, weight $REPLAY_WEIGHT every $REPLAY_EVERY, ledger $J6A_LEDGER"
timeout 34200 "$PY" -u tools/real_ft_run.py "${J6A_ARGV[@]}" 2>&1 | tee "$OUT/train.log"
say "J6(a) train+score done (exit $?)"
#
# Report only, with no kill criterion (campaign/j6a-preregistered.json
# arm.replay_flags.report_only_requirement): the replay KL beside the concurrent CE. Nothing
# here branches.
# a502670 keeps replay_log (python/qd_train/replay.py:478, :499) in memory only:
#   - its printed report carries replayed, replay_kl_first and replay_kl_last
#     (tools/real_ft_run.py:2457-2459, printed at :8825);
#   - PriorKLReplay.state() persists counts, not the log.
# So the first and last 20 per-step replay KLs cannot be read from an untouched a502670 run.
# This prints what the run does emit:
#   - those keys;
#   - the checkpoint's loss_log, the trainer's per-optimizer-step loss without the replay term,
#     first and last 20;
#   - the replay counters.
say "J6(a) replay report (report-only): NOT MET by this run -- a502670 does not emit per-step replay_log, so the first/last 20 replay KLs are not available (GAP-J6A-REPLAY-LOG-NOT-EMITTED-AT-A502670-2026-10-02). What it emits:"
grep -E '^replay: |"(replayed|replay_kl_first|replay_kl_last|letter_first|letter_last|total_first|total_last)":' "$OUT/train.log" \
  || say "J6(a) replay report: no replay keys in $OUT/train.log"
"$PY" -c '
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
state = body.get("model_state")
replay = state.get("replay") if isinstance(state, dict) else None
print(f"  checkpoint replay state: {json.dumps(replay, sort_keys=True)}")
log = body.get("loss_log")
if not isinstance(log, list) or not log:
    sys.exit(f"  {sys.argv[1]}: no loss_log")
def show(points):
    for x in points:
        step, index, loss = x["step"], x["index"], float.fromhex(x["loss_hex"])
        print(f"  step {step} batch {index} loss {loss:.6f}")
print(f"  loss_log (per optimizer step, no replay term): {len(log)} steps; first 20:")
show(log[:20])
print("  last 20:")
show(log[-20:])
' "$CKPT/epoch-seed0-cuda.json" || say "J6(a) replay report: the checkpoint's loss_log could not be read"
FT_ROW=$(log_ft_row "$OUT/train.log")
if [ -z "$FT_ROW" ] || [ ! -f "$CKPT/epoch-seed0-cuda.json" ]; then
  say "J6(a): no ft row or checkpoint from this run; its needle control NOT RUN"; exit 4
fi
timeout 5400 "$PY" -u tools/real_ft_run.py "${J6A_SPLIT[@]}" --devices cuda --seeds 0 \
  --score-val --needle --needle-control 1024,2048,4096 \
  --score-checkpoint "$CKPT/epoch-seed0-cuda.json" --score-dtype fp32 \
  --ft-ledger "$J6A_LEDGER" --ft-row-id "$FT_ROW" \
  "${COST[@]}" --wall-clock-cap-s 5400 \
  --ledger "$J6A_LEDGER" --suite-verdicts-out "$OUT/needle-control-1024-2048-4096.jsonl" \
  2>&1 | tee "$OUT/needle-control.log"
say "J6(a) needle control done (exit $?)"
say "j6a all done; the idle queue STOPS here (Fable Q2)"
