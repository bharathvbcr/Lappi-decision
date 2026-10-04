#!/bin/bash
# v6-startup item 2: cProfile of the OOD-only trajectory prelude on the v5 data, Mac.
# The checkpoint, ft ledger and ft row are dummies: the run passes every argv check, builds the
# split (MinHash/LSH), the val set and the OOD suite, and stops in _ft_row (no such ledger),
# before any model load. Lines are timestamped (UTC) by perl as they arrive.
set -o pipefail
WT=/Users/bharath/Code/research/Lappi-decision/build/v6-startup-wt
OUT=$WT/build/v6-profile
LABEL=${1:-prof1}
export QD_PREP_BIN=${QD_PREP_BIN_OVERRIDE:-$WT/target/release/qd-prep}
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
PROFILE=()
ENTRY=$WT/tools/real_ft_run.py
if [ "${2:-profile}" = profile ]; then PROFILE=(-m cProfile -o "$OUT/$LABEL.prof"); fi
if [ "${2:-profile}" = timed ]; then ENTRY=$WT/AUDIT/v6-startup-2026-10-04/timed_prelude.py; fi
PRE=()
TIMER=()
# The split cache A/B: plain real_ft_run, wall seconds and peak RSS from /usr/bin/time -l.
if [ "${2:-profile}" = ab ]; then TIMER=(/usr/bin/time -l); fi
if [ "${2:-profile}" = probe ]; then ENTRY=$WT/AUDIT/v6-startup-2026-10-04/split_cache_probe.py; PRE=("$OUT/$LABEL.pkl"); fi
if [ "${2:-profile}" = covered ]; then ENTRY=$WT/AUDIT/v6-startup-2026-10-04/split_inputs_covered.py; PRE=("$OUT/inputs1-inputs.json"); fi
if [ "${2:-profile}" = inputs ]; then ENTRY=$WT/AUDIT/v6-startup-2026-10-04/split_inputs_audit.py; PRE=("$OUT/$LABEL-inputs.json"); fi
# Extra real_ft_run flags (the split cache's A/B): everything after the mode.
EXTRA=("${@:3}")
date -u +"%H:%M:%S start $LABEL" | tee "$OUT/$LABEL.log"
"${TIMER[@]}" /Users/bharath/.venvs/ml/bin/python -u "${PROFILE[@]}" "$ENTRY" "${PRE[@]}" \
  --out /Users/bharath/qd-campaign/phase4-v5r-2026-10-03 --no-repo-history \
  --rev 8e6a00952a09e31a09cc3f5270229a935b7235cc \
  --defect-class "$WT/data/pool/commitpackft-composed-v2" \
  --defect-download /Users/bharath/Code/research/Lappi-decision/data/pool/commitpackft \
  --defect-noul "$WT/data/pool/defect-noul-v3c" \
  --general-record "$REC" --general-max-rows 200000 \
  --decisions-pool /Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions-v4 \
  --drop-before-dedupe /Users/bharath/qd-campaign/v5-predrops-2026-10-03/pre-dedupe-drops.txt \
  --exclude-identity-keys /Users/bharath/qd-campaign/v5-containment-predrops-2026-10-03/exclusions.txt \
  --real-backbone /Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --devices cpu --seeds 2 \
  --ood --ood-general-record "$REC" \
  --score-checkpoint "$OUT/absent/epoch-seed2-cpu-step1000.json" \
  --ft-ledger "$OUT/absent/ft.jsonl" --ft-row-id 00000000-0000-4000-8000-000000000000 \
  --ledger "$OUT/$LABEL-never.jsonl" "${EXTRA[@]}" \
  2>&1 | perl -MPOSIX -ne '$|=1; print strftime("%H:%M:%S ", gmtime), $_' | tee -a "$OUT/$LABEL.log"
rc=$?
date -u +"%H:%M:%S end $LABEL rc=$rc" | tee -a "$OUT/$LABEL.log"
exit 0
