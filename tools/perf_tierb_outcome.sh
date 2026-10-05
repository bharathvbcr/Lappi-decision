#!/bin/bash
# Tier-B outcome run for one candidate (Fable H/J; the no-mask ruling in
# campaign/f-j7prime-preregistered.json): one full phase-3 seed-0 run on the candidate path with
# default kernels, then the three comparisons Fable named:
#   1. train + --score-val  -> eval row, against the phase-3 envelope (choice 99.7-99.8%,
#      span 99.8-99.9%: 6d170b3c / 60f29b07 / 5c19c0e8);
#   2. ft_linear_control    -> paired_margin_vs_linear CI, against GO eeda5db4
#      (+0.150 [+0.136, +0.165]);
#   3. all-gates re-score of the saved checkpoint, fp32, against 58fd1532.
# Candidates, each screened alone (Fable: no-mask is not combined with fused):
#   fused  : --fused-adamw                    code perf/p3fused  (trainstep + mirror patches)
#   nomask : --train-attention-mask none      code perf/p3nomask (trainstep + mirror + nomask)
# Code = /home/ubuntu/qd-lane2 (main 884b658, whose qd_data matches the phase-3 shard set) plus
# the candidate's patches, each committed locally so the rows carry a clean code_commit:
#   trainstep.patch  the train-step branch e728a1d..e19dcf6 (already on the box)
#   mirror.patch     git diff 7c09ad3 0b95adb -- tools/real_ft_run.py python/tests/test_real_ft_pieces.py
#   nomask.patch     git diff 4fd08cf 1cf8e6d -- python/qd_train/backbone.py
#                      python/qd_train/trainer.py tools/real_ft_run.py
# (nomask.patch checked on a local 884b658 + trainstep + mirror reconstruction: it applies, and
# the 6 backbone and 17 argv/recipe no-mask tests pass there.) No
# Tier-B result enters a phase-5/6 run of this campaign (Fable); these are for F's successors.
# Usage: perf_tierb_outcome.sh fused|nomask --build | --link | --run | --print
#   --build CPU only; --run takes the GPU lock per stage; --print shows what --run would run.
#   --link gives an existing clone the checkout's git-ignored data links (see link_ignored)
#   without rebuilding it, since a rebuild re-commits the patches and moves the clone's commit.
# Writes only under /home/ubuntu/perf.
set -o pipefail
# TIERB_PERF and TIERB_SRC re-root the perf directory and the phase-3 checkout, for
# python/tests/test_perf_tierb_build.py only.
PERF=${TIERB_PERF:-/home/ubuntu/perf}
SRC=${TIERB_SRC:-/home/ubuntu/qd-lane2}
PY=/home/ubuntu/qd-venv/bin/python
BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
REV=be3073300cf4efb664f7065a32a44e4b9c12bd37
DEFECT=/home/ubuntu/qwen-decision/data/pool/commitpackft-corpus-v2
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
export HF_HUB_OFFLINE=1 QD_PREP_BIN=/home/ubuntu/bin/qd-prep

CAND=$1
ACTION=$2
case "$CAND" in
  fused)
    CODE=$PERF/p3fused
    FLAG=(--fused-adamw)
    PATCHES=(trainstep.patch mirror.patch)
    LEDGER=$PERF/ledger-tierb-fused-2026-10-01.jsonl ;;
  nomask)
    CODE=$PERF/p3nomask
    FLAG=(--train-attention-mask none)
    PATCHES=(trainstep.patch mirror.patch nomask.patch)
    LEDGER=$PERF/ledger-tierb-nomask.jsonl ;;
  *) echo "usage: $0 fused|nomask --build | --link | --run | --print"; exit 2 ;;
esac
OUT=$PERF/tierb-$CAND
CKPT=$PERF/ckpt-tierb-$CAND
SPLIT=(--out /home/ubuntu/phase3 --no-repo-history --defect-class $DEFECT --rev $REV)
TRAIN=(tools/real_ft_run.py "${SPLIT[@]}" --real-backbone $BACKBONE --optimizer master "${FLAG[@]}"
       --devices cuda --seeds 0 --epoch --no-memorise --score-val --batch-tokens 16384
       --checkpoint-dir $CKPT --checkpoint-every 100000 --verdicts-out $OUT/verdicts-s0.jsonl
       --wall-clock-cap-s 3600 --instance lambda-1xgh200 --usd-per-hour 2.29 --ledger $LEDGER)
RESCORE=(tools/real_ft_run.py "${SPLIT[@]}" --real-backbone $BACKBONE --devices cuda --seeds 0
         --score-val --needle --ood --ood-general-record $REC
         --score-checkpoint $CKPT/epoch-seed0-cuda.json --score-dtype fp32 --ft-ledger $LEDGER
         --usd-per-hour 2.29 --instance lambda-1xgh200 --wall-clock-cap-s 5400 --ledger $LEDGER
         --verdicts-out $OUT/verdicts-s0-allgates.jsonl)

# A clone carries no git-ignored file, and the checkout's data is git-ignored symlinks
# (data/pool/commitpackft-pool-v2.jsonl and eight more on the box). Without them, the fused run
# died on its first read of the pool at 2026-10-05 00:41:19Z. This mirrors each ignored symlink
# of $SRC into $CODE, to the same absolute target. Caches (__pycache__, .pytest_cache, target,
# *.pyc) are skipped. Every entry is checked before any link is made, and the whole call refuses
# (3) on any of these:
#   - an ignored regular file or directory (only links are mirrored);
#   - a relative or dangling link;
#   - a path or target under a heldout directory (rule 3);
#   - a destination that exists as anything but the identical link;
#   - a source with no ignored link at all.
link_ignored() {
  local rel s t d n=0 todo="" listing entries
  listing=$(cd "$SRC" && git status --ignored --porcelain=v1) \
    || { echo "refusing: cannot list $SRC's ignored files"; return 3; }
  # grep -v selects nothing when only caches are ignored; the count below refuses that case.
  entries=$(printf '%s\n' "$listing" | sed -n 's/^!! //p' \
    | grep -v -E '(^|/)(__pycache__|\.pytest_cache|target)(/|$)|\.pyc$') || true
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    case "$rel" in */) echo "refusing: $SRC/$rel is an ignored directory; only symlinks are mirrored"; return 3 ;; esac
    s=$SRC/$rel
    d=$CODE/$rel
    [ -L "$s" ] || { echo "refusing: $s is ignored but not a symlink; only symlinks are mirrored"; return 3; }
    t=$(readlink "$s")
    case "$t" in /*) ;; *) echo "refusing: $s -> $t is not an absolute link"; return 3 ;; esac
    case "/$rel/" in */heldout/*) echo "refusing: $rel is under a heldout directory (rule 3)"; return 3 ;; esac
    case "$t/" in */heldout/*) echo "refusing: $s -> $t is under a heldout directory (rule 3)"; return 3 ;; esac
    [ -e "$s" ] || { echo "refusing: $s -> $t dangles"; return 3; }
    if [ -L "$d" ]; then
      [ "$(readlink "$d")" = "$t" ] || { echo "refusing: $d is a link to $(readlink "$d"), not $t"; return 3; }
    elif [ -e "$d" ]; then
      echo "refusing: $d exists and is not a link"; return 3
    else
      [ -d "$(dirname "$d")" ] || { echo "refusing: $(dirname "$d") is not a directory in $CODE"; return 3; }
      todo="$todo$rel"$'\n'
    fi
    n=$((n + 1))
  done <<< "$entries"
  [ "$n" -gt 0 ] || { echo "refusing: $SRC has no ignored data symlinks; the run would find no pool"; return 3; }
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    ln -s "$(readlink "$SRC/$rel")" "$CODE/$rel" || { echo "refusing: ln -s for $rel failed"; return 3; }
  done <<< "$todo"
  echo "$CODE has $SRC's $n ignored symlinks"
}

case "$ACTION" in
  --print)
    echo "candidate $CAND: code $CODE, patches ${PATCHES[*]}, ledger $LEDGER"
    echo "train  : flock gpu.lock timeout 2400 $PY -u ${TRAIN[*]}"
    echo "control: $PY -u tools/ft_linear_control.py --ledger $LEDGER --verdicts $OUT/verdicts-s0.jsonl ${SPLIT[*]:2}"
    echo "rescore: flock gpu.lock timeout 5400 $PY -u ${RESCORE[*]} --ft-row-id <the ft row>"
    exit 0 ;;
  --build)
    set -e
    rm -rf $CODE
    git clone --quiet --local "$SRC" $CODE
    cd $CODE
    for P in "${PATCHES[@]}"; do
      git apply $PERF/$P
      git -c user.name="train-step perf lane" -c user.email=noreply@anthropic.com commit \
        --quiet -am "Tier-B $CAND build: $P on 884b658"
    done
    link_ignored
    echo "$CODE at $(git rev-parse HEAD) on $(git rev-parse HEAD~${#PATCHES[@]}) dirty=[$(git status --porcelain)]"
    exit 0 ;;
  --link)
    top=$(cd "$CODE" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null)
    if [ -z "$top" ] || [ "$top" != "$(cd "$CODE" && pwd -P)" ]; then
      echo "refusing: $CODE is not a git checkout; --build makes it"; exit 3
    fi
    link_ignored || exit 3
    echo "$CODE at $(cd "$CODE" && git rev-parse HEAD) dirty=[$(cd "$CODE" && git status --porcelain)]"
    exit 0 ;;
  --run) ;;
  *) echo "usage: $0 fused|nomask --build | --link | --run | --print"; exit 2 ;;
esac

if [ "$CAND" = "nomask" ]; then
  # Fable: a P2 fail cancels this run. Both shapes must have a P2 verdict, and both 'pass'; a
  # missing shape or a not_run cancels too -- an unexamined screen is not a passed one.
  $PY $PERF/overlay-nomask/tools/perf_parity.py --p2-gate $PERF/nomask-p2-verdicts.jsonl || exit 5
fi
cd $CODE || exit 3
if [ -n "$(git status --porcelain)" ]; then echo "$CODE is dirty; refusing"; exit 3; fi
mkdir -p $OUT
echo "=== Tier-B $CAND at $(git rev-parse --short HEAD), $(date -u +%H:%M:%S)"
flock /home/ubuntu/queue/gpu.lock timeout 2400 $PY -u "${TRAIN[@]}"
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

flock /home/ubuntu/queue/gpu.lock timeout 5400 $PY -u "${RESCORE[@]}" --ft-row-id "$FT_ROW"
echo "=== all-gates re-score exit $? at $(date -u +%H:%M:%S)"
touch $PERF/tierb-$CAND.done
