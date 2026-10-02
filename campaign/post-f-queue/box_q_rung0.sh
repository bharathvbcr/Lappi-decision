#!/bin/bash
# ojas CUDA rung 0: the first item after post-F item 10. The user decided on 2026-10-01 ("No
# insert; probe after"): there is no GH200 time for ojas while the post-F queue runs, and a ~5 min
# probe (~$0.08) comes first after item 10. Recorded in HANDOFF/ojas-training-2026-10-01.md.
#
# It runs the cross-built rung0 smoke binary from ojas/ojas-qwen35-cuda, pinned by sha256 (L-cuda-M0,
# HANDOFF/ojas-l-cuda-m0-2026-10-01.md):
# - it probes libcuda, libnvrtc and libcublas;
# - it compiles and checks the K0 kernels against host references;
# - it probes cuBLAS GemmEx bf16 -> f32;
# - it checks K1 GEMM against the f64 host reference.
# It writes one JSON report.
#
# Queue: after item 10 (j6dv4.done, touched by item 10's EXIT trap on every path). Holds gpu.lock.
# Outer timeout 300 s; the binary's own cap is 280 s.
# Writes only under /home/ubuntu/ojas-cuda and its log. It never touches a campaign ledger or
# lane. Its rows are ojas rows (rule 8: quick), recorded by the lead from the report.
set -o pipefail
Q=/home/ubuntu/queue
BIN=/home/ubuntu/bin/ojas-qwen35-cuda-rung0
# sha256 of the file copied from the Mac at
# /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda/aarch64-unknown-linux-gnu/release/rung0
BIN_SHA256=04ab105afcbf315d97f439f18c0baf73c659fd8d880ad34547419aa5ac9224f7
NV=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia
say() { echo "=== $* ($(date -u +%Y-%m-%dT%H:%M:%SZ))"; }
trap 'touch /home/ubuntu/queue/rung0.done' EXIT
touch $Q/rung0.queued
until [ -f $Q/j6dv4.done ]; do sleep 60; done
if ! echo "$BIN_SHA256  $BIN" | sha256sum -c --quiet; then
  say "rung0: $BIN does not match the pinned sha256; NOT RUN"
  exit 3
fi
exec 9>$Q/gpu.lock
flock 9
touch $Q/rung0.started
OUT=/home/ubuntu/ojas-cuda/rung0-$(date -u +%Y%m%dT%H%M%SZ)
say "rung0 start, out $OUT"
LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 timeout 300 "$BIN" --out "$OUT"
rc=$?
say "rung0 done (exit $rc; 0 all passed, 1 a check failed, 2 refused, 3 binary cap, 124 outer timeout)"
