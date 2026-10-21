#!/bin/bash
# ONE mac_heavy window, six v6 containment scans interleaved base/after/base/after/base/after, each
# measured by tools/footprint_peak.py (the fixed tool: kernel lifetime max per process, reaped-child
# RSS, labelled lower bound). base = build/l3/base-tree2 (main d09222d), after = build/l3/after-tree2
# (lane tip eb67c51). qd-prep was built before the window from d09222d's crates, which the tip does
# not change (git diff d09222d eb67c51 -- crates Cargo.toml Cargo.lock is empty).
# After the six: a byte comparison of every output with the first base run and with the digests the
# earlier after-runs wrote.
set -u
W=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286
export QD_PREP_BIN=$W/target/release/qd-prep
for i in 1 2 3; do
  bash $W/build/l3/scan_run.sh six_base$i $W/build/l3/base-tree2
  echo "six_base$i exit $?"
  bash $W/build/l3/scan_run.sh six_after$i $W/build/l3/after-tree2
  echo "six_after$i exit $?"
done
for n in six_base1 six_base2 six_base3 six_after1 six_after2 six_after3; do
  for f in pairs.tsv exclusions.txt attestation.json; do
    if cmp $W/build/l3/six_base1/$f $W/build/l3/$n/$f; then echo "$n IDENTICAL-TO-six_base1 $f"; else echo "$n DIFFERS-FROM-six_base1 $f"; fi
  done
done
shasum -a 256 $W/build/l3/six_*/pairs.tsv $W/build/l3/six_*/exclusions.txt $W/build/l3/six_*/attestation.json
