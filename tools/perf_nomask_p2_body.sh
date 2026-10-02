#!/bin/bash
# Body of one no-mask P2 session (perf_nomask_p2.sh T1|T2 runs it under the GPU lock). Every
# arm is a fresh process through real_ft_run._train (tools/perf_parity.py) and appends its own
# result line, so the 1,200 s cap loses only what had not finished. Run from the overlay
# checkout: the harness and the code under test are one commit.
#   <session> --dry-run: CPU only, the batch selection and its span-batch count, no tower.
set -o pipefail
SESSION=${1:?T1|T2}
DRY=${2:-}
PERF=/home/ubuntu/perf
PY=/home/ubuntu/qd-venv/bin/python
TOOLS=$(cd "$(dirname "$0")" && pwd)
CODE=$(dirname "$TOOLS")
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
V4=/home/ubuntu/phase4-v4-2026-10-01
LEDGER=$PERF/ledger-nomask-p2.jsonl
RES=$PERF/nomask-p2-$SESSION.jsonl
LOGS=$PERF/logs-nomask-p2
export HF_HUB_OFFLINE=1
case "$SESSION" in
  # Shape A: 100 steps, the span-carrying narrow buckets J4's step ran at.
  T1) PFX=A; SHAPE=(--batch-tokens 16384 --width-max 5383); N=100 ;;
  # Shape B: F's 35,403-token batches at v4's two widest buckets (7,404 and 7,936).
  T2) PFX=B; SHAPE=(--batch-tokens 35403 --width-min 7001 --width-max 7936); N=50 ;;
  *) echo "usage: $0 T1|T2 [--dry-run]"; exit 2 ;;
esac
COMMON=(--code-root "$CODE" --out $V4 --backbone $BACKBONE --min-span-batches 10
        --ledger $LEDGER --result $RES "${SHAPE[@]}")

if [ "$DRY" = "--dry-run" ]; then
  $PY -u "$TOOLS/perf_parity.py" "${COMMON[@]}" --n-batches $N --tag dry --dry-run --device cpu
  exit $?
fi
mkdir -p $LOGS
echo "=== $SESSION lock held at $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
arm() { local tag=$1; shift; echo "=== arm $tag start $(date -u +%H:%M:%S)"
  $PY -u "$TOOLS/perf_parity.py" "${COMMON[@]}" --tag "$tag" "$@" > $LOGS/arm-$tag.log 2>&1
  echo "=== arm $tag exit $?"; grep -v -i "warn" $LOGS/arm-$tag.log | tail -2; }

# Interleaved, so drift over the session lands on both sides alike.
for k in 1 2 3; do
  arm $PFX-mask-$k --n-batches $N
  arm $PFX-none-$k --n-batches $N --train-attention-mask none
done
# The det no-mask self-repeat: Tier-A goldens move onto det no-mask only if these are
# bit-identical (and the outcome run passes) -- Fable's no_mask.tier_a_goldens.
for k in 1 2; do
  arm $PFX-none-det-$k --n-batches 50 --train-attention-mask none --deterministic
done

if [ "$SESSION" = "T1" ]; then
  # Speed at shape B, masked vs not, with and without F's skip 6: interleaved min-of-3.
  echo "=== speed B start $(date -u +%H:%M:%S)"
  $PY -u "$TOOLS/perf_step.py" --code-root "$CODE" --out $V4 --backbone $BACKBONE --shape B \
    --batch-tokens 35403 --width-min 7001 --width-max 7936 --n-batches 5 --warmup 2 --steps 8 \
    --rounds 3 --overlap --budget-s 220 --results $PERF/nomask-bench.jsonl \
    --configs off:0 off:0+nomask skip:6 skip:6+nomask > $LOGS/bench-B.log 2>&1
  echo "=== speed B exit $?"; grep '"config"' $LOGS/bench-B.log
else
  # Peak memory at ~35K tokens of a mid bucket (9 x 3,907 = 35,163), the no-mask path alone
  # and with skip 6.
  echo "=== memory B3907 start $(date -u +%H:%M:%S)"
  $PY -u "$TOOLS/perf_step.py" --code-root "$CODE" --out $V4 --backbone $BACKBONE \
    --shape B3907 --batch-tokens 35403 --width-min 3907 --width-max 3907 --n-batches 4 \
    --warmup 1 --steps 4 --overlap --budget-s 120 --results $PERF/nomask-bench.jsonl \
    --configs off:0+nomask skip:6+nomask > $LOGS/bench-B3907.log 2>&1
  echo "=== memory B3907 exit $?"; grep '"config"' $LOGS/bench-B3907.log
fi
echo "=== $SESSION body done at $(date -u +%H:%M:%S)"
