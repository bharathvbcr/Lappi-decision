#!/bin/bash
# The no-mask Tier-B P2 screen (Fable's ruling, campaign/f-j7prime-preregistered.json "no_mask"):
# training without the padding mask (--train-attention-mask none, SDPA flash only) against the
# masked path, default kernels, on the same batches.
#   T1 = shape A (bt 16,384, width <= 5,383); T2 = shape B (bt 35,403, width 7,001-7,936).
#   Each: 3 masked + 3 no-mask arms, interleaved; then 2 det no-mask arms (the self-repeat the
#   Tier-A golden move needs: "two runs, plus one pair at shape B"). T1 then benchmarks speed at
#   B, T2 measures peak memory at ~35K tokens -- last, so the cap cuts those first.
#   Then, on CPU outside the lock: the P2 statistic and the det pair's bit-identity
#   (perf_parity.py --compare).
#   - The statistic (perf_parity.py --p2) follows Fable's rule as amended 2026-10-01: per
#     channel and per step set (first live step, all live steps), the candidates' mean
#     relative shift from the baseline mean, dev, against tau = 2%, provided the baselines'
#     leave-one-out noise is at most tau/3. Otherwise the result is inconclusive.
#   - A shape is decided from its four applications (letter, span) x (first, all), per Fable's
#     second amendment ("no_mask.p2_amendment_2"):
#     - it fails only when every application is conclusive and at least one fails;
#     - otherwise it is not_run if any application is not_run, else inconclusive if any is,
#       else pass.
#     The verdict row names the failing applications in fail_applications.
#   - The retired D(t) <= S(t)-on->=90% rule could not pass a correct candidate (per step
#     P(D <= S) = 1/5).
# Data: v4 (/home/ubuntu/phase4-v4-2026-10-01, F's set), which current main reads without the
# stale-shard override; v3 would need it (qd_data/defect_class.py changed since its build).
# What a verdict does to the outcome run (item 9):
#   - fail cancels it;
#   - not_run cancels it until a rerun;
#   - inconclusive holds it for the human, with no override (the human reads each
#     application's verdict, dev_i and noise_j, then fail_applications and the step-1 dev);
#   - pass admits nothing on its own.
# Precondition (Fable's p2_timing): before T1, the amended rule with its tests and the committed
# calibration .out are on main, and the overlay is rebuilt at that commit (--build <sha>).
#
# Usage:
#   perf_nomask_p2.sh --build <sha>  CPU: perf/overlay-nomask = qd-lane4 + perf/nomask.bundle at
#                                    <sha>, then a dry-run selection of both shapes (no tower).
#   perf_nomask_p2.sh T1|T2          GPU: one session under the lock, at most 1,200 s, then the
#                                    CPU statistics. Writes only under /home/ubuntu/perf.
set -o pipefail
PERF=/home/ubuntu/perf
PY=/home/ubuntu/qd-venv/bin/python
CODE=$PERF/overlay-nomask
export HF_HUB_OFFLINE=1

if [ "$1" = "--build" ]; then
  SHA=${2:?commit sha}
  bash $PERF/perf_mkoverlay.sh "$SHA" /home/ubuntu/qd-lane4 $PERF/nomask.bundle $CODE || exit 3
  for S in T1 T2; do
    bash $CODE/tools/perf_nomask_p2_body.sh $S --dry-run || exit 4
  done
  exit 0
fi
SESSION=$1
case "$SESSION" in
  T1) PFX=A ;;
  T2) PFX=B ;;
  *) echo "usage: $0 --build <sha> | T1 | T2"; exit 2 ;;
esac
[ -d $CODE/.git ] || { echo "no $CODE; run --build first"; exit 3; }
if [ -n "$(git -C $CODE status --porcelain)" ]; then echo "$CODE is dirty; refusing"; exit 3; fi
RES=$PERF/nomask-p2-$SESSION.jsonl
if [ -e $RES ]; then echo "$RES exists: a rerun takes a fresh file, never a mixed one"; exit 3; fi

echo "=== no-mask P2 $SESSION at $(git -C $CODE rev-parse --short HEAD), $(date -u +%H:%M:%S)"
flock /home/ubuntu/queue/gpu.lock timeout 1200 bash $CODE/tools/perf_nomask_p2_body.sh $SESSION
echo "=== $SESSION GPU part exit $? at $(date -u +%H:%M:%S)"

echo "=== P2 statistic $SESSION"
$PY $CODE/tools/perf_parity.py --p2 $RES --base $PFX-mask-1 $PFX-mask-2 $PFX-mask-3 \
  --cand $PFX-none-1 $PFX-none-2 $PFX-none-3 --p2-out $PERF/nomask-p2-verdicts.jsonl
echo "=== P2 $SESSION exit $? (0 pass, 1 fail, 2 not run or inconclusive)"
echo "=== det no-mask self-repeat $SESSION"
$PY $CODE/tools/perf_parity.py --compare $RES:$PFX-none-det-1 $RES:$PFX-none-det-2
echo "=== self-repeat $SESSION exit $? (0 identical)"
touch $PERF/nomask-p2-$SESSION.done
