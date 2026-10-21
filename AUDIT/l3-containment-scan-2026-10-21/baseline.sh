#!/bin/bash
# Held under one mac_heavy lock: the v6 containment scan three times on the changed code
# (build/l3/after-tree), then once on the unmodified code (build/l3/base-tree; last, because the old
# code was killed over the footprint cap in 4 of 5 recorded runs and a kill ends this whole job),
# then a byte comparison of each after-run's output with the base run's. Each scan is measured by
# tools/footprint_peak.py (see scan_run.sh).
set -u
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
export QD_PREP_BIN=$W/target/release/qd-prep
for n in after1 after2 after3; do
  bash $W/build/l3/scan_run.sh $n $W/build/l3/after-tree
  echo "$n exit $?"
done
bash $W/build/l3/scan_run.sh base1 $W/build/l3/base-tree
echo "base1 exit $?"
for n in after1 after2 after3; do
  for f in pairs.tsv exclusions.txt attestation.json; do
    if cmp $W/build/l3/base1/$f $W/build/l3/$n/$f; then echo "$n IDENTICAL $f"; else echo "$n DIFFERS $f"; fi
  done
done
shasum -a 256 $W/build/l3/base1/* $W/build/l3/after1/* $W/build/l3/after2/* $W/build/l3/after3/*
