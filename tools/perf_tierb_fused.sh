#!/bin/bash
# Tier-B outcome run (ii) for --fused-adamw (Fable Round J): one full phase-3 seed-0 run on the
# modified path with default kernels, then the two comparisons Fable named:
#   1. train + --score-val  -> eval row, against the phase-3 envelope (choice 99.7-99.8%,
#      span 99.8-99.9%: 6d170b3c / 60f29b07 / 5c19c0e8);
#   2. ft_linear_control    -> paired_margin_vs_linear CI, against GO eeda5db4
#      (+0.150 [+0.136, +0.165]);
#   3. all-gates re-score of the saved checkpoint, fp32, against 58fd1532.
# Code: /home/ubuntu/perf/p3fused = qd-lane2 (main 884b658, whose qd_data matches the phase-3
# shard set) + the train-step patch (branch commits e728a1d..e19dcf6), committed locally so the
# rows carry a clean code_commit. Built by --build (CPU only); --run takes the GPU lock.
# Writes only under /home/ubuntu/perf.
set -o pipefail
PERF=/home/ubuntu/perf
CODE=$PERF/p3fused
PY=/home/ubuntu/qd-venv/bin/python
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
REV=be3073300cf4efb664f7065a32a44e4b9c12bd37
DEFECT=/home/ubuntu/qwen-decision/data/pool/commitpackft-corpus-v2
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
LEDGER=$PERF/ledger-tierb-fused-2026-10-01.jsonl
OUT=$PERF/tierb-fused
CKPT=$PERF/ckpt-tierb-fused
export HF_HUB_OFFLINE=1 QD_PREP_BIN=/home/ubuntu/bin/qd-prep

if [ "$1" = "--build" ]; then
  set -e
  rm -rf $CODE
  git clone --quiet --local /home/ubuntu/qd-lane2 $CODE
  cd $CODE
  git apply $PERF/trainstep.patch
  git -c user.name="train-step perf lane" -c user.email=noreply@anthropic.com commit --quiet \
    -am "Train-step patch (e728a1d..e19dcf6 on the agent branch) on 884b658, for the Tier-B run"
  echo "p3fused at $(git rev-parse HEAD) on $(git rev-parse HEAD~1) dirty=[$(git status --porcelain)]"
  exit 0
fi
[ "$1" = "--run" ] || { echo "usage: $0 --build | --run"; exit 2; }

cd $CODE || exit 3
if [ -n "$(git status --porcelain)" ]; then echo "p3fused is dirty; refusing"; exit 3; fi
mkdir -p $OUT
SPLIT=(--out /home/ubuntu/phase3 --no-repo-history --defect-class $DEFECT --rev $REV)
echo "=== Tier-B fused at $(git rev-parse --short HEAD), $(date -u +%H:%M:%S)"
flock /home/ubuntu/queue/gpu.lock timeout 2400 $PY -u tools/real_ft_run.py "${SPLIT[@]}" \
  --real-backbone $BACKBONE --optimizer master --fused-adamw --devices cuda --seeds 0 \
  --epoch --no-memorise --score-val --batch-tokens 16384 \
  --checkpoint-dir $CKPT --checkpoint-every 100000 --verdicts-out $OUT/verdicts-s0.jsonl \
  --wall-clock-cap-s 3600 --instance lambda-1xgh200 --usd-per-hour 2.29 --ledger $LEDGER
echo "=== train+score-val exit $? at $(date -u +%H:%M:%S)"

FT_ROW=$($PY -c "
import json
rows = [json.loads(l) for l in open('$LEDGER')]
ft = [r for r in rows if r['run_kind'] == 'ft' and r['status'] == 'completed']
print(ft[-1]['row_id'] if ft else '')")
[ -n "$FT_ROW" ] && [ -f $CKPT/epoch-seed0-cuda.json ] || { echo "no ft row or checkpoint"; exit 4; }

$PY -u tools/ft_linear_control.py --ledger $LEDGER --verdicts $OUT/verdicts-s0.jsonl \
  "${SPLIT[@]:2}"
echo "=== linear control exit $? at $(date -u +%H:%M:%S)"

flock /home/ubuntu/queue/gpu.lock timeout 5400 $PY -u tools/real_ft_run.py "${SPLIT[@]}" \
  --real-backbone $BACKBONE --devices cuda --seeds 0 --score-val --needle --ood \
  --ood-general-record $REC --score-checkpoint $CKPT/epoch-seed0-cuda.json --score-dtype fp32 \
  --ft-ledger $LEDGER --ft-row-id "$FT_ROW" \
  --usd-per-hour 2.29 --instance lambda-1xgh200 --wall-clock-cap-s 5400 --ledger $LEDGER \
  --verdicts-out $OUT/verdicts-s0-allgates.jsonl
echo "=== all-gates re-score exit $? at $(date -u +%H:%M:%S)"
touch $PERF/tierb-fused.done
