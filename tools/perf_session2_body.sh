#!/bin/bash
# Body of perf session 2, run under the GPU lock by perf_session2.sh. See that file.
set -o pipefail
PERF=/home/ubuntu/perf
PY=/home/ubuntu/qd-venv/bin/python
BASE=/home/ubuntu/qd-lane4
OVER=$PERF/overlay
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
V3=/home/ubuntu/phase4-v3-2026-10-01
P8K=/home/ubuntu/probe-8k-2026-09-30
LEDGER=$PERF/ledger-parity-2026-10-01.jsonl
RES=$PERF/parity.jsonl
N=6
export HF_HUB_OFFLINE=1
echo "=== lock held at $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

A=(--out $V3 --backbone $BACKBONE --batch-tokens 16384 --width-max 5383 --n-batches 50
   --passes 1 --deterministic --ledger $LEDGER --result $RES)
B=(--out $P8K --backbone $BACKBONE --batch-tokens 35403 --width-min 8001 --width-max 8441
   --n-batches 5 --allow-fewer --passes 10 --allow-stale-shards --deterministic
   --ledger $LEDGER --result $RES)
# Every step's FULL output goes to its own log; the session log gets its tail, so a crash in
# the slot leaves the whole traceback behind rather than three lines of it.
mkdir -p $PERF/logs2
arm() { local tag=$1; shift; echo "=== arm $tag start $(date -u +%H:%M:%S)"
  $PY -u $PERF/perf_parity.py "$@" > $PERF/logs2/arm-$tag.log 2>&1
  echo "=== arm $tag exit $?"; grep -v -i "warn" $PERF/logs2/arm-$tag.log | tail -3; }

arm B-base   --code-root $BASE "${B[@]}" --tag B-base
arm B-skip$N --code-root $OVER "${B[@]}" --tag B-skip$N --checkpoint-skip-layers $N
arm A-base   --code-root $BASE "${A[@]}" --tag A-base
arm A-skip$N --code-root $OVER "${A[@]}" --tag A-skip$N --checkpoint-skip-layers $N
arm A-over0  --code-root $OVER "${A[@]}" --tag A-over0

# F's mode: default kernels. Interleaved A/B min-of-3 at 4 x 8,441, then peak memory at the
# 3,905 bucket (9 x 3,905 = 35,145 tokens, the probe set's nearest to 35,403).
echo "=== speed B default kernels start $(date -u +%H:%M:%S)"
$PY -u $PERF/perf_step.py --code-root $OVER --out $P8K --backbone $BACKBONE --shape B \
  --batch-tokens 35403 --width-min 8001 --width-max 8441 --allow-stale-shards --n-batches 5 \
  --warmup 2 --steps 8 --rounds 3 --budget-s 240 --results $PERF/bench.jsonl \
  --configs off:0 skip:$N > $PERF/logs2/bench-B.log 2>&1
echo "=== speed B exit $?"; grep '"config"' $PERF/logs2/bench-B.log
echo "=== memory B3905 default kernels start $(date -u +%H:%M:%S)"
$PY -u $PERF/perf_step.py --code-root $OVER --out $P8K --backbone $BACKBONE --shape B3905 \
  --batch-tokens 35403 --width-min 3905 --width-max 3905 --allow-stale-shards --n-batches 4 \
  --warmup 1 --steps 4 --budget-s 150 --results $PERF/bench.jsonl \
  --configs skip:$N skip:7 skip:8 > $PERF/logs2/bench-B3905.log 2>&1
echo "=== memory B3905 exit $?"; grep '"config"' $PERF/logs2/bench-B3905.log

arm A-base2  --code-root $BASE "${A[@]}" --tag A-base2
# One comparison per pair, so an arm the cap cut does not void the others.
echo "=== compare $(date -u +%H:%M:%S)"
$PY $PERF/perf_parity.py --compare $RES:B-base $RES:B-skip$N
for other in A-skip$N A-over0 A-base2; do
  $PY $PERF/perf_parity.py --compare $RES:A-base $RES:$other
done
$PY $PERF/perf_step.py --summarize $PERF/bench.jsonl

# LAST, approved as a speed-only probe (no product path): the tower without its padding mask
# (right-padded batches, causal attention), to see whether SDPA then takes flash and how fast
# its deterministic backward is at 8K. CUBLAS_WORKSPACE_CONFIG for the +det config, as session 1.
echo "=== no-mask probe start $(date -u +%H:%M:%S)"
CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY -u $PERF/perf_step.py --code-root $OVER --out $P8K \
  --backbone $BACKBONE --shape B --batch-tokens 35403 --width-min 8001 --width-max 8441 \
  --allow-stale-shards --n-batches 5 --warmup 2 --steps 6 --budget-s 170 \
  --results $PERF/probe-nomask.jsonl --configs off:0+nomask off:0+det+nomask+profile \
  > $PERF/logs2/probe-nomask.log 2>&1
echo "=== no-mask probe exit $?"; grep '"config"' $PERF/logs2/probe-nomask.log
