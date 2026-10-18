#!/usr/bin/env bash
# Claude Code PreToolUse hook (Bash): refuse a branch switch or a tree-moving git command while
# the worktree has uncommitted changes to tracked files.
#
# Why: on 2026-10-17 an unattributed `git checkout ... main` in the shared worktree carried
# another session's staged pin bump across branches. The owner's rule is "commit changes before
# switching branches or checkout"; Fable's 2026-10-18 ruling Q6
# (AUDIT/harden-2026-10-17/fable-rulings-2026-10-18.md) is to enforce it with this hook, because
# a memory rule only reaches sessions that read that memory.
#
# Input: the hook payload as JSON on stdin (.tool_input.command, .cwd). Exit 2 blocks the call
# and shows stderr to the model. Exit 0 lets it run. A payload this script cannot read blocks
# too (exit 2): a guard that cannot check must not pass.
#
# Binds Claude Code sessions only. A human shell or another tool is not covered.
set -u

payload=$(cat)
cmd=$(printf '%s' "$payload" | jq -r '.tool_input.command // empty') || {
  echo "guard_branch_switch: could not parse the hook payload; refusing rather than passing" >&2
  exit 2
}
cwd=$(printf '%s' "$payload" | jq -r '.cwd // empty')
[ -n "$cwd" ] || cwd=$PWD

# Commands that move or hide work: switching branches, force-moving a branch, stashing,
# and staging everything (which sweeps up another lane's files in a shared worktree).
# Global options before the subcommand may take a separate value (`git -c k=v`, `git -C dir`).
pattern='(^|[;&|[:space:]])git([[:space:]]+-[^[:space:]]+([[:space:]]+[^-[:space:]][^[:space:]]*)?)*[[:space:]]+(checkout|switch|stash|branch[[:space:]]+(-f|--force)|add[[:space:]]+(-A|--all))([[:space:]]|$)'
printf '%s' "$cmd" | grep -Eq "$pattern" || exit 0

# `git checkout -- <path>` and `git checkout <rev> -- <path>` restore files and do not switch
# branches; `git switch -c`/`checkout -b` from a clean tree is allowed by the dirty test below.
dirty=$(git -C "$cwd" status --porcelain --untracked-files=no 2>&1) || {
  echo "guard_branch_switch: git status failed in $cwd; refusing rather than passing: $dirty" >&2
  exit 2
}
[ -z "$dirty" ] && exit 0

{
  echo "guard_branch_switch: refused. The worktree $cwd has uncommitted changes to tracked files:"
  printf '%s\n' "$dirty" | head -20
  echo "Commit your own changes (explicit paths) before switching branches; never stash or"
  echo "git add -A in a shared worktree. If the changes are another session's, ask it to commit"
  echo "them, or work in a separate worktree. This hook binds Claude Code sessions only, not a"
  echo "human shell (owner rule 2026-10-17; Fable ruling Q6 2026-10-18)."
} >&2
exit 2
