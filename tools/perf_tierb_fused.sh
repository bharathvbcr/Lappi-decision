#!/bin/bash
# The fused-AdamW Tier-B outcome run, kept at this path because the queued
# /home/ubuntu/box_q_tierb.sh calls it. The run itself is perf_tierb_outcome.sh, which serves
# both Tier-B candidates (fused, nomask) so the two runs cannot drift apart.
# Usage: perf_tierb_fused.sh --build | --link | --run | --print
exec bash "$(dirname "$0")/perf_tierb_outcome.sh" fused "$@"
