#!/bin/bash
# Idle queue cudadev (Fable's idle-queue ruling, Q1 item 4 and Q2): the ojas CUDA device checks
# that L-cuda-M0/-gdn/-small/-M1 left NOT RUN, on the GH200, after rung 0.
#   1. runga (L-cuda-M1, HANDOFF/ojas-l-cuda-m1-2026-10-01.md "The box command for runga"):
#      every M0, M1, K2(i) and small check in one process, plus the report-only GDN timing.
#   2. the eleven cross-built device-test binaries the CUDA lanes list -- K0, K1 (M0), K3, K4, K6,
#      K7, K9, K10 (small), K8, K11 (M1), GDN published (gdn) -- each `--ignored
#      --test-threads=1 --nocapture`, each under its own timeout.
# Every binary is pinned by sha256. All twelve came from ONE aarch64 cross-build of the ojas tree
# into /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-idle; that build
# reproduced runga's bb3276e7 and L-cuda-M1's device binaries byte for byte
# (HANDOFF/idle-queue-waiters-2026-10-02.md). A pin mismatch refuses that one binary, loudly.
# A failed device test is a result, not a reason to stop: every exit code is recorded and the
# next binary runs.
# Cap: 1,500 s of wall clock under the lock in total, = $0.95 at $2.29/h. The brief's two bounds
# both bind: <= 1,800 s, and < $1 (1,800 s would be $1.15; $1 is 1,572 s). Each binary's timeout
# is the smaller of its own cap and what is left of the total; with under 30 s left the rest are
# NOT RUN (budget) and recorded as such. GDN goes last: it holds the timing test and needs about
# 15 GiB, and runga already covers its checks and timing once.
# Queue: after rung 0 (rung0.done). Holds gpu.lock. Writes only under /home/ubuntu/ojas-cuda and
# its log; no campaign ledger. Its rows are ojas rows (rule 8: quick), recorded by the lead.
set -o pipefail
Q=/home/ubuntu/queue
B=/home/ubuntu/bin
NV=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia
TOTAL_CAP_S=1500
RUNGA_CAP_S=300
DEVICE_CAP_S=150
GDN_CAP_S=600
MIN_LEFT_S=30
say() { echo "=== $* ($(date -u +%Y-%m-%dT%H:%M:%SZ))"; }
trap 'touch /home/ubuntu/queue/cudadev.done' EXIT
touch $Q/cudadev.queued
until [ -f $Q/rung0.done ]; do sleep 60; done

# name, file under $B, sha256, own cap (s). runga first, GDN last.
NAMES=(runga k0 k1 k8 k11 k3_gates_published k4_conv1d k6_qk_norm_rope k7_rmsnorm k9_embed k10_ce_rows gdn_published)
FILES=(ojas-qwen35-cuda-runga ojas-cuda-device-k0 ojas-cuda-device-k1 ojas-cuda-device-k8 ojas-cuda-device-k11
       ojas-cuda-device-k3-gates-published ojas-cuda-device-k4-conv1d ojas-cuda-device-k6-qk-norm-rope
       ojas-cuda-device-k7-rmsnorm ojas-cuda-device-k9-embed ojas-cuda-device-k10-ce-rows ojas-cuda-device-gdn-published)
SHAS=(bb3276e7a42ae3d0e20171ed78e0df537f2581d8f144d22b8b6c054a0814c805
      b10542091a32f9bf293d1cefd36d754221115f5ed88386c1b2e3f970fa9820a6
      9546b2df0da4334ab735673885b9812a9f02ca4021172b60fb4c1f6b9eaa93bd
      75f05d3a42c7964fc0143dc5149a02d37b2af04aaca3f7ea271cefcdb8074ff1
      bc65b3bca3c48f1814185fc74859e6c623cffceda5bf3f9736d9b8b82c72ad90
      fd4e1e2410b2b8ae11176aa0b3d3f75a469b48fb0c8a4387c7f17fe27819bfbb
      eb12cb197f5d259a4b3bf79396f247f431c222461afbdb8191a763bf70e4b666
      03a641f055494f8c68bb1f8eb320906fe892010e59fe3ac2375037f5498611fe
      c56c71146f7af183c3af1a3252e51818705d1a61e24d0eef7cdfc3440bd095e2
      363a21f68eefd664cf72bb8af6ce78be8a43b4fa99ff0ad4d9b0f4d1053f6d8b
      54c8755464b913adb407c74c52ebfb6d421da0e39e844af34cba61950a5c707e
      97034105198442010a12661228442197a9d20a3eb7685480e239daebdb8f4f28)
CAPS=("$RUNGA_CAP_S" "$DEVICE_CAP_S" "$DEVICE_CAP_S" "$DEVICE_CAP_S" "$DEVICE_CAP_S" "$DEVICE_CAP_S"
      "$DEVICE_CAP_S" "$DEVICE_CAP_S" "$DEVICE_CAP_S" "$DEVICE_CAP_S" "$DEVICE_CAP_S" "$GDN_CAP_S")
if [ "${#NAMES[@]}" -ne 12 ] || [ "${#FILES[@]}" -ne 12 ] || [ "${#SHAS[@]}" -ne 12 ] || [ "${#CAPS[@]}" -ne 12 ]; then
  say "cudadev: the binary table is not 12 x 4; NOT RUN"; exit 3
fi

exec 9>$Q/gpu.lock
flock 9
touch $Q/cudadev.started
OUTD=/home/ubuntu/ojas-cuda/cudadev-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUTD" || exit 3
SUM=$OUTD/summary.tsv
printf 'name\tfile\tsha256_pinned\tstatus\texit\tseconds\ttimeout_s\tlog\n' > "$SUM"
START=$(date +%s)
DEADLINE=$((START + TOTAL_CAP_S))
say "cudadev start, out $OUTD, total cap ${TOTAL_CAP_S}s (deadline $(date -u -d @$DEADLINE +%H:%M:%SZ))"
FAILED=0
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]} bin=$B/${FILES[$i]} want=${SHAS[$i]} cap=${CAPS[$i]}
  log=$OUTD/$name.log
  left=$((DEADLINE - $(date +%s)))
  if [ "$left" -lt "$MIN_LEFT_S" ]; then
    say "$name NOT RUN: ${left}s of the ${TOTAL_CAP_S}s cap left (budget)"
    printf '%s\t%s\t%s\tnot_run_budget\t-\t0\t0\t-\n' "$name" "$bin" "$want" >> "$SUM"
    continue
  fi
  got=$(sha256sum "$bin" 2>/dev/null | cut -c1-64)
  if [ "$got" != "$want" ]; then
    say "$name NOT RUN: $bin is sha256 ${got:-absent}, not the pinned $want"
    printf '%s\t%s\t%s\trefused_pin\t-\t0\t0\t-\n' "$name" "$bin" "$want" >> "$SUM"
    FAILED=1
    continue
  fi
  t=$cap
  if [ $((left - 10)) -lt "$t" ]; then t=$((left - 10)); fi
  t0=$(date +%s)
  if [ "$name" = runga ]; then
    say "runga (timeout ${t}s) -> $OUTD/runga/runga-report.json"
    timeout --kill-after=10 "$t" env LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cuda_nvrtc/lib" NVIDIA_TF32_OVERRIDE=0 \
      "$bin" --out "$OUTD/runga" > "$log" 2>&1
    rc=$?
  else
    say "$name (timeout ${t}s)"
    timeout --kill-after=10 "$t" env LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cuda_nvrtc/lib" NVIDIA_TF32_OVERRIDE=0 \
      "$bin" --ignored --test-threads=1 --nocapture > "$log" 2>&1
    rc=$?
  fi
  dt=$(( $(date +%s) - t0 ))
  case $rc in
    0) st=pass ;;
    124) st=timeout ;;
    137) st=killed ;;
    *) st=fail ;;
  esac
  [ "$rc" -eq 0 ] || FAILED=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$bin" "$want" "$st" "$rc" "$dt" "$t" "$log" >> "$SUM"
  say "$name exit $rc ($st) in ${dt}s; tail of $log:"
  tail -5 "$log"
done
say "cudadev done in $(( $(date +%s) - START ))s; summary $SUM (runga exit: 0 pass, 1 a check failed, 2 refused, 3 binary cap; tests: 0 pass, 101 a test failed; 124 timeout; 137 killed, by timeout's --kill-after or from outside)"
cat "$SUM"
exit $FAILED
