#!/usr/bin/env bash
# The data-side stress suite's runner (AUDIT/data-stress-suite-2026-10-06.md, items S1-S12),
# hardened (card ft-b5110a5c0ed). Three defects of the 2026-10-06 runners are closed here:
#
# 1. The suite ran on the shared main checkout with no pinned tree
#    (GAP-STRESS-RUNNER-HEAD-CHECKED-BEFORE-THE-LOCK-2026-10-06: HEAD was checked, then the
#    runner waited 1,980 s for the lock while 4 commits landed, and the suite ran on 25e6f2b).
#    Now: the runner is a launcher plus a job. The job runs INSIDE the heavy-job lock, creates a
#    detached `git worktree add` at the pinned full sha, and checks that worktree's HEAD equals the
#    pin. It never reads the shared checkout's working files. A chain of runners (the span chain,
#    this suite) pins its code tree at its first runner: later runners pass `--tree DIR` of that
#    first worktree, and its HEAD is checked against `--sha` again, inside their own lock.
# 2. A step that ran past its bound recorded the same 101/failed a nondeterministic result
#    does (GAP-STRESS-RUNNER-S1-S8-STEPS-UNBOUNDED-DEBUG-PROFILE-2026-10-06). Now: every step has a
#    wall-clock bound of its own, and a step that passes it, or whose test killed a child at its
#    own bound (stress_v6.rs `wait_bounded`: "ran past ... and was killed"), is `timed_out`. A
#    bound is never a failure, and a failure is never a bound.
# 3. The profile was whatever `cargo test` defaulted to. Now: `--profile debug|release` is a
#    recorded input, written to the run header BEFORE the first step, and a step whose log names
#    another profile is `failed`, not accepted. The criteria and every bound are the
#    pre-registered ones (CLAUDE.md rule 2): this script chooses a profile, never a threshold.
#
# Step states, in <out>/status.tsv as `name<TAB>state<TAB>start<TAB>end`:
#   passed | failed:<rc> | timed_out | not_run:<reason>
# `not_run` and `timed_out` are never `passed`. U4 (determinism, S8) is reported tri-state from the
# three S8 steps: `passed` only if all three passed, `failed` if any failed, else `undetermined`
# (not_run / timed_out) -- an undetermined U4 is never a pass.
#
#   bash tools/stress_runner.sh launch --sha <40-hex> --profile release [--label NAME]
#       [--only STEP[,STEP...]] [--out DIR] [--tree DIR]
#   (the launcher takes the lock by running itself through tools/mac_heavy.sh)
#
# Run from the main checkout only to launch; nothing else here reads it.
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/tools/stress_runner.sh"
PY=${STRESS_PY:-/Users/bharath/.venvs/ml/bin/python}
UV=${STRESS_UV:-/opt/homebrew/bin/uv}

# Per-step wall-clock bounds, seconds. New to this runner: the 2026-10-06 runner had none per
# step. They are generous multiples of the measured release times and are not criteria. The S9
# bounds are the pre-registered item bounds plus the 300 s the old runner added.
BOUND_DEFAULT=3600
BOUND_S8=5700   # three runs at stress_v6.rs CLI_LIMIT (1800 s) each, plus 300 s
BOUND_S9_EMAIL=900
BOUND_S9_CONTAINMENT=1500
BOUND_S9_LSH=900

step_bound() {
  case "$1" in
    s8-email|s8-jarvis|s8-tools) echo "$BOUND_S8" ;;
    s9-email) echo "$BOUND_S9_EMAIL" ;;
    s9-containment) echo "$BOUND_S9_CONTAINMENT" ;;
    s9-lsh) echo "$BOUND_S9_LSH" ;;
    *) echo "$BOUND_DEFAULT" ;;
  esac
}

now() { date -u +%FT%TZ; }

# record <name> <state> <start>
record() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(now)" >> "$OUT/status.tsv"
}

not_run() {
  local name=$1 reason=$2 t
  t=$(now)
  echo "not_run: $reason" > "$OUT/$name.log"
  printf '%s\tnot_run:%s\t%s\t%s\n' "$name" "$reason" "$t" "$t" >> "$OUT/status.tsv"
  echo "== $name not_run: $reason"
}

# Kill a step's whole process group: TERM, then KILL after 10 s.
kill_group() {
  local pgid=$1
  kill -TERM -- "-$pgid" 2> /dev/null
  local i=0
  while [ $i -lt 10 ] && kill -0 -- "-$pgid" 2> /dev/null; do sleep 1; i=$((i + 1)); done
  kill -KILL -- "-$pgid" 2> /dev/null
}

# The step now running, so a signal to the runner can stop it (verify-lane6 D14: a SIGTERM used
# to run the EXIT trap and leave the step alive in its own process group, orphaned, still running
# after the lock was released).
CURRENT_STEP=""
CURRENT_START=""
CURRENT_PGID=""

# on_signal <signal>: stop the running step's process group, record it `interrupted`, exit 143/130.
on_signal() {
  if [ -n "$CURRENT_PGID" ]; then
    kill_group "$CURRENT_PGID"
    record "$CURRENT_STEP" "interrupted:$1" "$CURRENT_START"
    CURRENT_PGID=""
  fi
  [ "$1" = INT ] && exit 130
  exit 143
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

# bounded_step <name> <bound_s> <command...>: run under its own bound, log to $OUT/<name>.log,
# record `passed`, `failed:<rc>` or `timed_out`. A child the test itself killed at its own bound
# ("ran past ... and was killed") is `timed_out` too: it is the same fact one level down.
bounded_step() {
  local name=$1 bound=$2 start pid rc timed_out=0 began
  shift 2
  start=$(now)
  began=$(date +%s)
  echo "== $name start $start (bound ${bound}s)"
  set -m
  "$@" > "$OUT/$name.log" 2>&1 < /dev/null &
  pid=$!
  set +m
  CURRENT_STEP=$name
  CURRENT_START=$start
  CURRENT_PGID=$pid
  while kill -0 "$pid" 2> /dev/null; do
    # Wall clock, not a count of sleeps: a slow `sleep` or a stalled runner cannot stretch it.
    if [ $(($(date +%s) - began)) -ge "$bound" ]; then
      timed_out=1
      kill_group "$pid"
      break
    fi
    sleep 1
  done
  wait "$pid" 2> /dev/null
  rc=$?
  CURRENT_PGID=""
  if [ "$timed_out" -eq 1 ]; then
    echo "TIMED OUT: the step passed its ${bound}s bound and its process group was killed" >> "$OUT/$name.log"
    record "$name" timed_out "$start"
    echo "== $name timed_out"
    return 124
  fi
  if [ "$rc" -eq 0 ]; then
    record "$name" passed "$start"
    echo "== $name passed"
    return 0
  fi
  if grep -Eq 'ran past .* and was killed|the synth run passed [0-9]' "$OUT/$name.log"; then
    record "$name" timed_out "$start"
    echo "== $name timed_out (a child passed its own bound)"
    return 124
  fi
  record "$name" "failed:$rc" "$start"
  echo "== $name failed:$rc"
  return "$rc"
}

# pin_worktree <sha> <dir>: a detached worktree of the main repository at exactly <sha>, then the
# check that its HEAD is <sha>. Run inside the lock. Exit 70 (the lock's "result does not count")
# on any disagreement.
pin_worktree() {
  local sha=$1 dir=$2 got
  if ! printf '%s' "$sha" | grep -Eq '^[0-9a-f]{40}$'; then
    echo "stress_runner: --sha must be a full 40-hex sha, not '$sha'" >&2
    return 70
  fi
  if [ -e "$dir" ]; then
    echo "stress_runner: $dir exists; refusing to reuse it (pass --tree to chain onto a worktree)" >&2
    return 70
  fi
  if ! git -C "$REPO" worktree add --detach "$dir" "$sha" > "$OUT/worktree-add.log" 2>&1; then
    echo "stress_runner: git worktree add $dir $sha failed; see $OUT/worktree-add.log" >&2
    return 70
  fi
  # The cleanup trap is set only after the pin succeeds, so a failed check removes the worktree
  # it just added (verify-lane6 D16).
  check_head "$sha" "$dir" && return 0
  got=$?
  git -C "$REPO" worktree remove --force "$dir" > /dev/null 2>&1
  return "$got"
}

# check_head <sha> <dir>
check_head() {
  local sha=$1 dir=$2 got
  got=$(git -C "$dir" rev-parse HEAD 2> /dev/null)
  if [ "$got" != "$sha" ]; then
    echo "stress_runner: HEAD of $dir is '$got', not the pinned $sha; result does not count" >&2
    return 70
  fi
  if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no 2> /dev/null)" ]; then
    echo "stress_runner: $dir has tracked changes; a pinned tree is clean; result does not count" >&2
    return 70
  fi
  return 0
}

# u4_state: tri-state over the three S8 steps in status.tsv.
u4_state() {
  local passed=0 failed=0 other=0 s
  for s in s8-email s8-jarvis s8-tools; do
    case "$(awk -F'\t' -v n="$s" '$1 == n {st = $2} END {print st}' "$OUT/status.tsv")" in
      passed) passed=$((passed + 1)) ;;
      failed:*) failed=$((failed + 1)) ;;
      *) other=$((other + 1)) ;;
    esac
  done
  if [ "$failed" -gt 0 ]; then
    echo failed
  elif [ "$passed" -eq 3 ]; then
    echo passed
  else
    echo undetermined
  fi
}

if [ "${STRESS_RUNNER_SOURCE_ONLY:-0}" = 1 ]; then
  # Tests source this file for the functions above and run none of the suite.
  return 0 2> /dev/null || exit 0
fi

# ---------------------------------------------------------------------------------------------
# The launcher: parse, then take the lock by running the job through mac_heavy.sh.

usage() {
  echo "usage: bash tools/stress_runner.sh launch --sha <40-hex> --profile debug|release" >&2
  echo "         [--label NAME] [--only STEP,STEP] [--out DIR] [--tree DIR]" >&2
  exit 64
}

MODE=${1:-}
[ -n "$MODE" ] && shift
SHA=""
PROFILE=""
LABEL="run"
ONLY=""
OUT=""
TREE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sha) SHA=${2:-}; shift 2 ;;
    --profile) PROFILE=${2:-}; shift 2 ;;
    --label) LABEL=${2:-}; shift 2 ;;
    --only) ONLY=${2:-}; shift 2 ;;
    --out) OUT=${2:-}; shift 2 ;;
    --tree) TREE=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done
case "$PROFILE" in debug | release) ;; *) usage ;; esac
[ -n "$SHA" ] || usage
if [ -z "$OUT" ]; then
  OUT="$REPO/build/stress-runs/$(printf '%s' "$SHA" | cut -c1-12)-$PROFILE-$LABEL-$(date -u +%Y%m%dT%H%M%SZ)"
fi

case "$MODE" in
  launch)
    # The pin is checked again inside the lock (mode `job`); checking here too only fails fast.
    mkdir -p "$OUT" || exit 70
    MAC_HEAVY_MAX_WAIT_S=${MAC_HEAVY_MAX_WAIT_S:-14400} bash "$REPO/tools/mac_heavy.sh" "stress-$LABEL" \
      bash "$SELF" job --sha "$SHA" --profile "$PROFILE" --label "$LABEL" --out "$OUT" \
      ${ONLY:+--only "$ONLY"} ${TREE:+--tree "$TREE"}
    exit $?
    ;;
  job) ;;
  *) usage ;;
esac

# ---------------------------------------------------------------------------------------------
# The job. Inside the lock or not at all.

if [ -z "${MAC_HEAVY_LABEL:-}" ]; then
  echo "stress_runner: job mode runs only under tools/mac_heavy.sh (MAC_HEAVY_LABEL is unset); use 'launch'" >&2
  exit 64
fi
mkdir -p "$OUT" || exit 70
: > "$OUT/status.tsv"

if [ -n "$TREE" ]; then
  check_head "$SHA" "$TREE" || exit 70
  WT=$TREE
  OWN_TREE=0
else
  WT="$REPO/build/stress-trees/$(printf '%s' "$SHA" | cut -c1-12)-$$"
  mkdir -p "$(dirname "$WT")"
  # The workspace names path dependencies as <root>/../tessl and <root>/../ojas: make them
  # resolve from the worktree's parent, to the same siblings the main checkout uses.
  for sibling in tessl ojas; do
    [ -e "$(dirname "$WT")/$sibling" ] || ln -s "$REPO/../$sibling" "$(dirname "$WT")/$sibling"
  done
  pin_worktree "$SHA" "$WT" || exit 70
  OWN_TREE=1
fi
cleanup() {
  if [ "$OWN_TREE" -eq 1 ]; then
    git -C "$REPO" worktree remove --force "$WT" > /dev/null 2>&1
  fi
}
trap cleanup EXIT

# The run header, written before the first step: the labels, the pin, the profile, the bounds.
{
  echo "run_label: $LABEL"
  echo "sha: $SHA"
  echo "tree: $WT (detached worktree: $([ "$OWN_TREE" -eq 1 ] && echo created || echo chained))"
  echo "profile: $PROFILE"
  echo "started: $(now)"
  echo "lock_label: $MAC_HEAVY_LABEL"
  echo "bounds_s: default=$BOUND_DEFAULT s8=$BOUND_S8 s9-email=$BOUND_S9_EMAIL s9-containment=$BOUND_S9_CONTAINMENT s9-lsh=$BOUND_S9_LSH"
  echo "only: ${ONLY:-all}"
} > "$OUT/header.txt"
cat "$OUT/header.txt"

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$REPO/build/target-stress-$(printf '%s' "$SHA" | cut -c1-12)-$PROFILE}
MANIFEST="$WT/Cargo.toml"
RELEASE_FLAG=""
[ "$PROFILE" = release ] && RELEASE_FLAG="--release"

wants() { [ -z "$ONLY" ] || printf ',%s,' "$ONLY" | grep -q ",$1,"; }

# One stress_v6.rs item, by test-name prefix, N tests at a time.
rust_item() {
  local prefix=$1 threads=$2
  shift 2
  cargo test --manifest-path "$MANIFEST" -p qd-prep $RELEASE_FLAG --test stress_v6 -j 2 -- "${prefix}_" \
    --test-threads "$threads" --nocapture "$@"
}
lib_reuse() {
  cargo test --manifest-path "$MANIFEST" -p qd-prep --lib -j 2 -- --exact "$@"
}
pytest_k() (
  export PYTHONPATH=$WT/python
  "$UV" run --no-project --python "$PY" --with pytest --with hypothesis \
    python -m pytest "$@" -o addopts= -p no:cacheprovider -q --rootdir "$WT"
)
# An S8 step is only accepted from the profile that was written down: its log must say so.
s8_item() {
  local test=$1
  cargo test --manifest-path "$MANIFEST" -p qd-prep $RELEASE_FLAG --test stress_v6 -j 2 -- \
    "$test" --exact --nocapture --test-threads 1 2>&1 | tee "$OUT/.s8-last"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && ! grep -q "S8-PROFILE $PROFILE" "$OUT/.s8-last"; then
    echo "S8-PROFILE $PROFILE was asked for and the test did not report it: not accepted" >&2
    return 71
  fi
  return "$rc"
}
s9_item() {
  local test=$1
  cargo test --manifest-path "$MANIFEST" -p qd-prep --release --test stress_v6 -j 2 -- \
    "$test" --exact --ignored --nocapture --test-threads 1
}
# S6's kill: a run that ends before SIGTERM lands did not exercise the item: not_run. It goes
# through bounded_step like every step, so a timeout there is `timed_out`, never `failed:101`.
s6_kill() {
  bounded_step s6-kill "$(step_bound s6-kill)" \
    cargo test --manifest-path "$MANIFEST" -p qd-prep $RELEASE_FLAG --test stress_v6 -j 2 -- \
    s6_a_kill_during_the_write --exact --nocapture
  if grep -q 'S6-KILL-NOT-EXERCISED' "$OUT/s6-kill.log"; then
    # Replace the row bounded_step wrote: the run finished before SIGTERM reached the write.
    awk -F'\t' -v OFS='\t' '$1 == "s6-kill" {$2 = "not_run:the run finished before SIGTERM reached the write"} {print}' \
      "$OUT/status.tsv" > "$OUT/status.tsv.new" && mv "$OUT/status.tsv.new" "$OUT/status.tsv"
  fi
  return 0
}

step() {
  local name=$1
  shift
  wants "$name" || return 0
  bounded_step "$name" "$(step_bound "$name")" "$@"
}
not_run_step() {
  wants "$1" || return 0
  not_run "$1" "$2"
}

if step stress-build cargo test --manifest-path "$MANIFEST" -p qd-prep $RELEASE_FLAG --test stress_v6 --no-run -j 2; then
  step s1 rust_item s1 2
  step s2 rust_item s2 2
  step s3 rust_item s3 2
  step s4 rust_item s4 2
  step s5 rust_item s5 2
  step s6 rust_item s6 2 --skip s6_a_kill_during_the_write
  wants s6-kill && s6_kill
  step s7 rust_item s7 2
  step s8-email s8_item s8_synth_email_is_byte_identical_at_one_and_eight_threads
  step s8-jarvis s8_item s8_synth_jarvis_is_byte_identical_at_one_and_eight_threads
  step s8-tools s8_item s8_synth_tools_is_byte_identical_at_one_and_eight_threads
  step s11 rust_item s11 2
  step s12 rust_item s12 2
else
  for item in s1 s2 s3 s4 s5 s6 s6-kill s7 s8-email s8-jarvis s8-tools s11 s12; do
    not_run "$item" "stress_v6.rs did not build (stress-build.log); nothing in it ran"
  done
fi

step s2-reuse lib_reuse decisions::tests::structural_refusals_mirror_choice_slot
step s3-reuse lib_reuse pool::tests::nothing_surviving_is_refused \
  convert::tests::a_row_with_an_unassigned_code_point_is_refused_and_counted \
  containment::tests::malformed_requests_are_refused
step s4-reuse lib_reuse pool::tests::a_target_set_not_pinned_or_given_twice_is_refused
step s5-reuse lib_reuse synth_tools::tests::a_one_member_group_is_refused \
  synth_tools::tests::an_underspecified_phrasing_must_name_a_required_parameter \
  synth_tools::tests::a_shared_tool_asks_only_where_its_parameter_is_required
step s8-reuse lib_reuse containment::tests::the_answer_does_not_depend_on_the_thread_count
# S8-decisions needs the pinned upstream sources (the fetch record, decider's teacher_data); it is
# not_run until a fixture that holds them exists, and the runner looks for that fixture by name.
S8_DECISIONS_FIXTURE="$WT/crates/qd-prep/tests/fixtures/decisions-pinned-sources"
if [ -d "$S8_DECISIONS_FIXTURE" ]; then
  not_run_step s8-decisions "a fixture exists at $S8_DECISIONS_FIXTURE but no test reads it; the step is not implemented"
else
  not_run_step s8-decisions "no fixture holds the decisions pool's pinned upstream sources ($S8_DECISIONS_FIXTURE is absent)"
fi
step s12-reuse lib_reuse convert_licence::tests::an_unstated_licence_is_refused_with_its_own_reason \
  convert_licence::tests::aliases_and_whitespace_are_python_s_and_no_more \
  convert_licence::tests::non_commercial_is_disqualifying_by_segment_not_substring \
  convert_licence::tests::the_human_registered_dataset_licences_are_admitted_and_named
step s12-parity cargo test --manifest-path "$MANIFEST" -p qd-prep --test convert_licence_parity -j 2

if [ -f "$WT/crates/qd-train/tests/caller_records_are_held_out.rs" ]; then
  step s11-qdtrain-records cargo test --manifest-path "$MANIFEST" -p qd-train \
    --test caller_records_are_held_out -j 2
else
  not_run_step s11-qdtrain-records "crates/qd-train/tests/caller_records_are_held_out.rs is not in the pinned tree"
fi
step s11-qdtrain-door cargo test --manifest-path "$MANIFEST" -p qd-train --test door -j 2 -- \
  --exact a_held_out_shard_path_is_refused_before_any_byte_is_read \
  segment_markers_are_exact_so_a_lookalike_directory_is_not_a_holdout

step s7-py pytest_k "$WT/python/tests/test_stress_v6.py" -k test_s7_
step s10-py pytest_k "$WT/python/tests/test_stress_v6.py" -k test_s10_
step s10-py-reuse pytest_k "$WT/python/tests/test_decisions_pool.py" -k \
  "test_the_loader_refuses_a_pool_its_manifest_does_not_describe or test_the_loader_refuses_a_family_of_another_source"
step s11-py pytest_k "$WT/python/tests/test_stress_v6.py" -k test_s11_
step s12-py pytest_k "$WT/python/tests/test_stress_v6.py" -k test_s12_
step s12-py-reuse pytest_k "$WT/python/tests/test_decisions_pool.py" -k \
  test_a_non_commercial_row_licence_is_refused

if { [ -z "$ONLY" ] || printf ',%s,' "$ONLY" | grep -Eq ',s9-'; } &&
  step s9-build cargo test --manifest-path "$MANIFEST" -p qd-prep --release --test stress_v6 --no-run -j 2; then
  step s9-email s9_item s9_synth_email_at_100k_rows_within_ten_minutes
  step s9-containment s9_item s9_containment_200k_candidates_by_50k_targets_within_twenty_minutes
  step s9-lsh s9_item s9_lsh_at_one_million_keys_within_ten_minutes
elif [ -z "$ONLY" ] || printf ',%s,' "$ONLY" | grep -Eq ',s9-'; then
  for item in s9-email s9-containment s9-lsh; do
    not_run "$item" "the release build of stress_v6.rs failed (s9-build.log)"
  done
fi

{
  echo "finished: $(now)"
  echo "U4 (S8 determinism, the three synth kinds): $(u4_state)"
} | tee -a "$OUT/header.txt"
echo "== stress suite done $(now); status in $OUT/status.tsv"
