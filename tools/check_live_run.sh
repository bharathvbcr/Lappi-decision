#!/usr/bin/env bash
# How many training processes are live on this machine. Run it ON the box.
#
# This exists as a FILE rather than as a string passed to ssh, and that is the whole point.
# The first version of the guard in tools/sync_box.sh was
#
#     ssh box 'pgrep -fc "rung0_real_run.py|real_ft_run.py"'
#
# and it counted itself: the remote shell's own command line is
# `bash -c pgrep -fc "rung0_real_run.py|..."`, which contains the pattern, so `pgrep -f`
# matched it. With the capacity sweep running it reported 3 for 2 processes; with nothing
# running it reported 1. It failed closed, which is the right direction and still useless --
# a guard that always says "busy" can never permit the thing it guards.
#
# Keeping the pattern inside a file means the invoking command line is
# `bash tools/check_live_run.sh` and contains none of it. pgrep already excludes its own pid.
#
# Verified in BOTH directions on 2026-09-21, because every refusal observed until then was
# contaminated by the self-match and so was not evidence that it detects anything: started a
# python process at tools/rung0_real_run.py -> 1; killed it -> 0.
#
# Counted with `wc -l` rather than `pgrep -c`, which is a procps flag: BSD pgrep on macOS
# has no `-c`, so the old form printed a usage error and NO number there. Callers read an
# empty answer as "busy" (`${LIVE:-1}`), so it failed closed -- and a guard that can only
# ever say busy on the Mac could never let `sync_box.sh pull` run where the campaign smoke
# test runs (measured 2026-09-29). `pgrep -f` exists on both and still excludes itself.
set -uo pipefail
{ pgrep -f "tools/rung0_real_run\.py|tools/real_ft_run\.py" || true; } | wc -l | tr -d ' '
