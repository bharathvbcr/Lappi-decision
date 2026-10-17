# Branch consolidation — 2026-10-15 / 2026-10-16

Local only. Nothing was pushed. No remote branch was deleted. No commit was dropped.

## Pass 9 (2026-10-16): merged on Fable's plan after the user said "ask fable for advice and merge"

`main` is `fe42017` → `9bab363`, ahead 208 of `origin/main`. Nothing was pushed. The backups are pass 8's `refs/backup/2026-10-16/*` and `build/lappi-decision-refs-2026-10-16.bundle`.

### Commits on `main`

| Commit | What | Checks run |
| --- | --- | --- |
| `171c879` | Specials work (wave10 lane, orphaned in this checkout), committed directly on `main`: `named_special_ids`, its call site, tests, digest re-pin `82e412c4` → `ff39da60`, and the two `gaps.jsonl` account lines. The lines were staged by blob `c08e767`, so my own appended line stayed out. | `test_remap.py` + `test_containment_exclusions.py`: 72 passed, 3 skipped (repo `.venv`, prebuilt `QD_PREP_BIN`). **The `ff39da60` digest test is Not run**: it skips because `transformers` is not in the repo venv, and the harness refused pointing `uv` at `~/.venvs/ml`. `make lint`: pass. |
| `a1245b2` | The wave10 lane's uncommitted ledger row, as written (failed build row at `5660ad6-dirty`). | `make ledger-verify`: chain verifies, 233 rows. |
| `09238b6` | `git merge --no-ff ojas-pin-88be893` (pin ojas `132b46e`). It merged cleanly with **no conflicts**, since its files (`Cargo.lock`, `upstream.lock.json`, `test_pin_gate.py`) do not overlap `main`'s. | `test_pin_gate.py`: 7 passed. **`make pin`: FAILED**, as it was before the merge: ojas sibling at `040e1ed`, lock `132b46e`. `Cargo.lock` did not drift. |
| `171b9c6` | `ledger/runs.main-gates-2026-10-15.jsonl`: the gate worktree's ledger kept whole as its own file. It forks from `runs.jsonl` after `feae39e0`. Its rows are `d7e6cc75` (status `killed`, SIGTERM) and `7016657c` (`completed`, all three suites `not_run` after the 1800 s cap). Neither appears in `runs.jsonl`: there is no re-chain tool, and rewriting hashes is refused. No code reads `ledger/*.jsonl` by glob. | `qd_train.ledger verify` on the file: chain verifies, 234 rows. |
| `9bab363` | `gaps.jsonl`: `GAP-BRANCH-PASS8-OJAS-ANCESTRY-UNCHECKABLE-2026-10-16`, `GAP-LEDGER-FORK-MAIN-GATES-2026-10-15`. | `make ledger-verify`: 233 rows. |
| `10db431` | `gaps.jsonl`: `GAP-DEVMAP-STALE-AFFECTED-TESTS-UNUSABLE-2026-10-16` (the last item in Fable's plan). | — |

**`make gates` (no `ci:local` exists): Not run.** It takes about 90 min, and its `ledger-record` step appends a row. Proportionate checks were run per commit instead (table above), as Fable advised. DevMap is stale (`is_fresh=false`), so `devmap_affected_tests` could not scope them.

### Deleted

| Branch | Was | Backup | How |
| --- | --- | --- | --- |
| `ojas-pin-88be893` | `21fb7ad` | `refs/backup/2026-10-16/ojas-pin-88be893` (+ `refs/backup/2026-10-15/…`) | `git worktree remove .claude/worktrees/ojas-pin-88be893` (clean, no process cwd), then `git branch -d` |
| `wave10-characterization-specials` | `4b0b3b8` | `refs/backup/2026-10-16/wave10-characterization-specials` | `git branch -d`. Its orphaned work now lives on `main` (`171c879`), so the reason to keep it is gone. |

No `-D` was used.

### Kept

| Branch / worktree | Reason |
| --- | --- |
| `main` | trunk |
| `span-pin-8a531e8-verdicts` | Cherry `-`, already on main. Its worktree is the pin tree hardcoded in `python/tests/test_span_refit_b_launcher.py:26` and `tools/span_refit_b.sh`. |
| `build/main-gates-2026-10-15` (detached) | Still dirty (`Cargo.lock`, ledger, `.venv` symlink). Its ledger rows are now preserved on `main`, so it can be removed later with `git worktree remove --force` once you're happy. |

### Blocked or open

- `make pin` stays red until ojas main is pushed and the pin is re-bumped to a published sha. That commit should change `upstream.lock.json`, `LOCKED` in `test_pin_gate.py`, and `Cargo.lock` together.
- The `ff39da60` characterization digest needs a run in an env with `transformers` + the HF snapshot + the commitpackft download.

### Verify

`git fsck --connectivity-only`: exit 0 (dangling objects only; `build/fsck-2026-10-16.txt`). `gitpulse_insights`: all facets `ok`, 3 worktrees, 0 overlapping files (the `ledger/runs.jsonl` collision is gone), 1 live session (this one). No session lost its branch.

### Restore

```
git branch ojas-pin-88be893 refs/backup/2026-10-16/ojas-pin-88be893
git worktree add .claude/worktrees/ojas-pin-88be893 ojas-pin-88be893
git branch wave10-characterization-specials refs/backup/2026-10-16/wave10-characterization-specials
git branch -f main refs/backup/2026-10-16/main   # undo all of pass 9 on main (run from another branch)
```

## Pass 8 (2026-10-16): backups taken, nothing merged, committed, or deleted

### What changed since pass 7

- This checkout is now on `main` (was `wave10-characterization-specials`). Grok pid 34301 is gone. Its six modified files are still uncommitted in this checkout, now on `main`.
- Ojas packed `refs/heads/main` is `040e1ed6f2d9820465bc0c0b3645ba21469d2632` (was `b47cc33`). `origin/main` is still `bc498c4`.
- No GitPulse task is `in_progress`. The only live agent with cwd in any worktree of this repo is this session (claude pid 33633). Checked with `gitpulse_insights` (all facets `ok`) and `lsof -d cwd`. GitPulse only observes claude and codex, so grok/agy liveness comes from `lsof` alone.

### Inventory

| Branch | Tip | Ahead / behind `main` | Last commit | Merged | `git cherry` | Worktree (dirty?) | Live session | Task |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `main` | `fe42017` | n/a (ahead 202 of `origin/main`) | 2026-10-15 09:06 BHARATH CHANDRA VADDARAM | n/a | n/a | primary checkout (dirty: 6 modified, 9 untracked) | this session | none |
| `wave10-characterization-specials` | `4b0b3b8` | 0 / 1 | 2026-10-14 21:25 same | yes | no lines | none | none | none |
| `span-pin-8a531e8-verdicts` | `1d42f37` | 1 / 243 | 2026-10-14 16:56 same | no | `-` (patch on main) | `/Users/bharath/qd-campaign/worktrees/span-pin-8a531e8` (clean) | none | none |
| `ojas-pin-88be893` | `21fb7ad` | 1 / 2 | 2026-10-15 08:04 same | no | `+` (unique) | `.claude/worktrees/ojas-pin-88be893` (clean) | none | none |
| (detached) | `fe42017` | 0 / 0 | n/a | n/a | n/a | `build/main-gates-2026-10-15` (dirty: `Cargo.lock`, `ledger/runs.jsonl` +2 rows, `.venv` symlink) | none | none |

No remote branches were in scope. `origin` has only `main`.

### Classification (reviewed by Fable)

| Branch | Class | Reason |
| --- | --- | --- |
| `wave10-characterization-specials` | merged, **kept, flagged** | Safe to delete, since its tip is on `main`. It is kept because it is the only lane handle for the orphaned specials work in this checkout (`gaps.jsonl:1675` tags it `"lane": "wave10-characterization-specials"`), and no `HANDOFF/wave10-*.md` exists. Fable recommended keeping it until the owner decides. |
| `span-pin-8a531e8-verdicts` | superseded, **kept (in use by tooling)** | Cherry `-`. Its worktree is the pin tree hardcoded at `python/tests/test_span_refit_b_launcher.py:26` and in `tools/span_refit_b.sh`. Tag `span-pin-8a531e8-verdicts-2026-10-14` also points at `1d42f37`. |
| `ojas-pin-88be893` | unmerged, **kept, flagged** | Pins `132b46e`. That is not on ojas `origin` (`bc498c4`) and is not ojas local main (`040e1ed`), so `make pin` stays red with or without this merge. A pin bump needs your yes on a delta report. The "owner approved 2026-10-15" in the commit message is a claim I could not verify. Ancestry of `132b46e` in `040e1ed` is inferred from the ojas reflog file, not checked with git (`GAP-BRANCH-PASS8-OJAS-ANCESTRY-UNCHECKABLE-2026-10-16`). |

Mergeable this pass: none.

### Backups

| Ref | Sha |
| --- | --- |
| `refs/backup/2026-10-16/main` | `fe4201759566d55584310e2cb46d05eaa2ebc433` |
| `refs/backup/2026-10-16/wave10-characterization-specials` | `4b0b3b859ee9555ce0f5b9f5b08d041511de0eec` |
| `refs/backup/2026-10-16/span-pin-8a531e8-verdicts` | `1d42f37d01aec8c283e8144bd8d701be6a9c99aa` |
| `refs/backup/2026-10-16/ojas-pin-88be893` | `21fb7addefc9df80c948a5dd005c0fe10a3e5b81` |

Bundle: `build/lappi-decision-refs-2026-10-16.bundle` (`git bundle verify` okay, complete history, 8 refs). Sha256 `cc3d3d0d2e66d065728959351f0685b42c2d2e0c3081485f9faa856ac5653fa9`. **Deviation:** the brief asked for a path outside the repo. The MANVI write gate refuses that (`path.outside_root`), so the bundle sits in the gitignored `build/`, as in earlier passes. Copy it out by hand if it must survive losing this directory.

### Pending work (phase 3): not committed

- **Main checkout.** Six modified files hold the specials work: `named_special_ids` in `python/qd_train/remap.py`, its call site in `tools/real_tokenizer_pipeline.py`, tests, the characterization digest re-pinned from `82e412c4…` to `ff39da60…` in `python/tests/test_containment_exclusions.py`, two `gaps.jsonl` lines, and one ledger row. It is another agent's work with no handoff. It also re-pins a characterization digest whose file-by-file account (`gaps.jsonl:1675`) I could not recompute. Needs a human to accept the re-pin. Its ledger row is a failed `build` row at `5660ad6-dirty`.
- **`build/main-gates-2026-10-15`.** Detached, so there is no branch to commit to. It holds two ledger rows that exist only there. Its `Cargo.lock` hunk matches `ojas-pin-88be893`'s.
- **Ledger fork risk.** Both trees appended to `ledger/runs.jsonl` from the same committed tip, and GitPulse reports it as contended. Whichever lands second must be re-chained, not line-merged, and neither row may be dropped.

### Merges, conflicts, deletions

None. No merge was attempted, so there were no conflicts and no merge commits. No branch, worktree, or backup ref was deleted.

### Verify (phase 6)

- `git fsck --connectivity-only`: exit 0. Dangling objects only.
- Branch tips unchanged: `main fe42017`, `ojas-pin-88be893 21fb7ad`, `span-pin-8a531e8-verdicts 1d42f37`, `wave10-characterization-specials 4b0b3b8`.
- `gitpulse_insights` re-run: all facets `ok`, 4 of 4 worktrees scanned, 1 live session (this one). No session lost its branch.
- **`make gates` (this repo has no `ci:local`): Not run this pass.** `main` did not move from `fe42017`, whose gate run on 2026-10-15 was not green (pin FAILED; the suites hit the 1800 s cap and are Not run). See the earlier `make gates` section below and `build/gates-fe42017-r2.log`. A re-run would take about 90 min and would append a third uncommitted row to the contended `ledger/runs.jsonl`.

### Needs you

1. Specials work in this checkout: accept the `ff39da60` re-pin and commit it (on `wave10-characterization-specials` or `main`), or hand it back to a lane.
2. `ojas-pin-88be893`: push ojas main, then re-bump to a published sha with a delta report. Or approve merging `132b46e` as is.
3. The gate worktree's two ledger rows: keep them (re-chain onto `main`) or discard them with the worktree.

### Restore

Nothing was deleted. Any branch can be reset from its backup, e.g. `git branch -f ojas-pin-88be893 refs/backup/2026-10-16/ojas-pin-88be893`, or fetched from the bundle with `git fetch build/lappi-decision-refs-2026-10-16.bundle 'refs/heads/*:refs/restored/*'`.

## Pass 7 (same day): one stale branch deleted, nothing merged

Nothing new belonged on `main`. `ojas-pin-88be893` still pins `132b46e`. Ojas loose `refs/heads/main` is still `b47cc33f0d0d3112b84443fe5e547e070b12d325`. No task is `in_progress`. `main` is still `fe42017`. The full `gitpulse_insights` call was deferred under load (`ok` false on worktrees, changes, and live agents). That is not a clean scan. The delete used `git worktree list`, `git branch --merged`, `git cherry`, `lsof`, and a later `gitpulse_change_context` on the worktree that was removed (`ok` true, `dirty_files` 0, `task_id` empty, `sessions_ok` true).

Deleted:

| Branch | Tip | Backup | Worktree |
| --- | --- | --- | --- |
| `gp/w9-prefill-comment` | `4b0b3b859ee9555ce0f5b9f5b08d041511de0eec` | `refs/backup/2026-10-15/gp/w9-prefill-comment` | `/Users/bharath/Code/research/Lappi-w9-qd` |

It was an ancestor of `main` (ahead 0, behind 1). `git cherry` printed no unique patch. No process had that cwd. `git worktree remove` (no `--force`) then `git branch -d`. Git printed `Deleted branch gp/w9-prefill-comment (was 4b0b3b8).` Grok pid 34301 was still on `wave10-characterization-specials` afterward. Its six modified files were not touched.

Still kept:

| Branch | Reason |
| --- | --- |
| `wave10-characterization-specials` | This checkout. Live grok pid 34301. Dirty. Tip `4b0b3b8` is already on `main`, so the name stays while the session does. |
| `span-pin-8a531e8-verdicts` | Cherry `-`. The worktree is the pin tree named by `tools/span_refit_b.sh` and `python/tests/test_span_refit_b_launcher.py`. |
| `ojas-pin-88be893` | Unique commit. The pin sha is not the ojas checkout. Grok pid 41144 still has cwd `/Users/bharath/Code/research/ojas`. |

```
git branch gp/w9-prefill-comment refs/backup/2026-10-15/gp/w9-prefill-comment
git worktree add /Users/bharath/Code/research/Lappi-w9-qd gp/w9-prefill-comment
```

## Pass 6 (same day, no merge)

Rechecked refs, worktrees, tasks, and the ojas checkout. Five local branches. `main` is still `fe42017`. No task is `in_progress`. GitPulse insights `ok`, live Claude 0, live Codex 0, five worktrees. Nothing was merged or deleted.

`git branch --no-merged main` is `ojas-pin-88be893` (cherry `+`, ahead 1, behind 2) and `span-pin-8a531e8-verdicts` (cherry `-`, ahead 1, behind 243). `gp/w9-prefill-comment` and `wave10-characterization-specials` are ancestors of `main` (ahead 0, behind 1) and have no unique patch. Ojas loose `refs/heads/main` is still `b47cc33f0d0d3112b84443fe5e547e070b12d325`. Packed `refs/remotes/origin/main` is still `bc498c4`. Grok pid 41144 still has cwd `/Users/bharath/Code/research/ojas`. Grok pid 34301 still has cwd in this checkout, with the same six modified tracked files.

## Pass 5 (same day): stale merged branches deleted, nothing merged

No new commit belonged on `main`. `ojas-pin-88be893` still pins `132b46e` while ojas loose `refs/heads/main` is `b47cc33f0d0d3112b84443fe5e547e070b12d325`. No task is `in_progress`. `main` is still `fe42017`. Grok pid 34301 was still on this checkout after the deletes.

Deleted with `git branch -d` after `git worktree remove --force`. Both tips were ancestors of `main`. Neither worktree had a live cwd or a tracked modification. The untracked files were pool manifests, plus an untracked `.venv` on wave4. 200 json files from those two `data/pool` trees were copied to `build/preserved-worktrees-2026-10-15/<branch>/` before the remove. `--force` was required because `git worktree remove` refuses untracked files.

| Branch | Was | Backup |
| --- | --- | --- |
| `wave1-l3-armb-corpora` | `5ba07a7347b2a60ac1ddfe246c0c97948f0ef8ee` | `refs/backup/2026-10-15/wave1-l3-armb-corpora` |
| `wave4-l1-own-swift-clean-rows` | `44840917ef8ef81c2f422d2359b124b7abb0f3bc` | `refs/backup/2026-10-15/wave4-l1-own-swift-clean-rows` |

Worktrees removed: `.claude/worktrees/agent-aee488bfe3833f10e` and `.claude/worktrees/agent-a3ceb68fc1021f5c2`.

Left in place:

| Branch | Reason |
| --- | --- |
| `wave10-characterization-specials` | Primary checkout. Grok pid 34301. Tip `4b0b3b8` is already on `main`. Six tracked files still modified. |
| `gp/w9-prefill-comment` | Created 2026-10-15 13:35:47 -0500 from HEAD. Tip `4b0b3b8` is already on `main`, worktree `/Users/bharath/Code/research/Lappi-w9-qd` is clean, and no process had that cwd. It is a same-day workspace, so the name stays. |
| `span-pin-8a531e8-verdicts` | Cherry `-` (patch already on main). `tools/span_refit_b.sh` and `python/tests/test_span_refit_b_launcher.py` use `/Users/bharath/qd-campaign/worktrees/span-pin-8a531e8` as the pin tree. |
| `ojas-pin-88be893` | Unique commit. Pin sha does not match the ojas checkout. Worktree is clean. |

Restore the two deleted branches:

```
git branch wave1-l3-armb-corpora refs/backup/2026-10-15/wave1-l3-armb-corpora
git branch wave4-l1-own-swift-clean-rows refs/backup/2026-10-15/wave4-l1-own-swift-clean-rows
```

The pool manifests are under `build/preserved-worktrees-2026-10-15/`. The worktree directories themselves were removed.

## Pass 4 (same day, no merge)

Rechecked refs, worktrees, tasks, stash, and the ojas checkout. Same six local branches. `main` is still `fe42017`. `git branch --no-merged main` is still `ojas-pin-88be893` (`21fb7ad`, cherry `+`, ahead 1 behind 2) and `span-pin-8a531e8-verdicts` (`1d42f37`, cherry `-`, ahead 1 behind 243). No task is `in_progress`. Stash is empty. GitPulse insights `ok`, live Claude 0, live Codex 0, six worktrees, primary still dirty with the same six tracked files. Nothing was merged, deleted, or pushed.

Ojas loose `refs/heads/main` is still `b47cc33f0d0d3112b84443fe5e547e070b12d325`. Packed `refs/remotes/origin/main` is still `bc498c4`. Grok pid 41144 still has cwd `/Users/bharath/Code/research/ojas`. Grok pid 34301 still has cwd in this checkout. The pin commit still names `132b46e`.

## Pass 3 (same day, no merge)

Rechecked refs, worktrees, tasks, and the ojas checkout. Same six local branches. `main` is still `fe42017`. `git branch --no-merged main` is still `ojas-pin-88be893` and `span-pin-8a531e8-verdicts`. No task is `in_progress`. Stash is empty. Nothing was merged, deleted, or pushed.

`ojas-pin-88be893` still pins `132b46e`. Ojas loose `refs/heads/main` is still `b47cc33f0d0d3112b84443fe5e547e070b12d325`. Packed `origin/main` is still `bc498c4`. Grok pid 41144 still has cwd `/Users/bharath/Code/research/ojas`. Grok pid 34301 still has cwd in this checkout, whose six tracked modifications are unchanged. Span-pin's cherry line is still `-`.

## Pass 2 (same day, no merge)

Re-inventory after the wave8 delete. `gitpulse_insights` and `gitpulse_collision_risk` both `ok` (`unscanned_worktrees` 0, `sessions_ok` true, `truncated` false). Live Claude 0, live Codex 0. No GitPulse task is `in_progress`. Local branches are unchanged: `main` is still `fe42017`. `git branch --no-merged main` is still only `ojas-pin-88be893` and `span-pin-8a531e8-verdicts`.

Nothing was merged.

| Branch | Why it stayed off main |
| --- | --- |
| `ojas-pin-88be893` (`21fb7ad`, cherry `+`) | The commit pins ojas `132b46e`. The ojas checkout's loose `refs/heads/main` is now `b47cc33f0d0d3112b84443fe5e547e070b12d325`. Packed `refs/remotes/origin/main` is still `bc498c4`. Grok pid 41144 has cwd `/Users/bharath/Code/research/ojas`. Merging this pin would not make `make pin` pass. |
| `span-pin-8a531e8-verdicts` (`1d42f37`, cherry `-`) | The one commit not on main already has its patch-id on main. The worktree at `/Users/bharath/qd-campaign/worktrees/span-pin-8a531e8` is the span-refit pin and is clean. |
| `wave10-characterization-specials` | Tip `4b0b3b8` is already on main. The branch is the primary checkout. Grok pid 34301 still has cwd here (also this session, pid 39434). Six tracked files are still modified and were not committed. |
| `wave1-l3-armb-corpora`, `wave4-l1-own-swift-clean-rows` | Already ancestors of main. Worktrees still dirty (GitPulse 12 and 10). |

No new local branch appeared. No backup ref was deleted. No worktree was removed. The gate worktree `build/main-gates-2026-10-15` is still detached at `fe42017` and still dirty, so `git worktree remove` without `--force` still refuses it.

## Result

| | This pass |
| --- | --- |
| Branches merged | 1 (`wave8-safetensors-coverage` → `fe4201759566d55584310e2cb46d05eaa2ebc433`) |
| Conflicts | 0 |
| Branches deleted | 1, with `git branch -d` (not `-D`) |
| Worktrees removed | 0 |
| Branches kept | 5 local, plus `origin/main` |
| Pushed | no |
| `make gates` on the merge | not green. Pin failed for a pre-existing ojas mismatch. The three recorded suites were cut off at 1800s and are Not run |

`main` moved `5660ad60cbcd7fdd92de4bde641773656f99274b` → `fe4201759566d55584310e2cb46d05eaa2ebc433`. The primary checkout did not move. It is still `wave10-characterization-specials` at `4b0b3b859ee9555ce0f5b9f5b08d041511de0eec`.

## Inventory

Taken before the merge, then recomputed against `fe42017`. `gitpulse_insights` and `gitpulse_collision_risk` both returned `ok` (`unscanned_worktrees` 0, `sessions_ok` true, `truncated` false). Re-run after the delete: same, live Claude 0, live Codex 0, six worktrees, primary branch still `wave10-characterization-specials`.

GitPulse's live-agent scan sees Claude and Codex only. It does not see Grok. That blind spot is recorded here and was not appended to `gaps.jsonl`, because that file has uncommitted lines owned by the other session. Process check at the end of this pass: grok pid 34301 was still running (`Ss+`, elapsed about 13h47m). That pid is the other session on this checkout. This session did not commit its files.

`gitpulse_change_context` on the primary worktree returned `task_id` empty. No GitPulse task is `in_progress` (38 tasks: 30 review, 4 inbox, 3 backlog, 1 done). The task column is inferred from commit subjects, not a board binding.

Ahead / behind is `git rev-list --left-right --count main...<branch>` after the merge: left is behind, right is ahead. Every tip author is BHARATH CHANDRA VADDARAM.

| Branch | Ahead | Behind | Last commit | Merged into `fe42017` | Worktree | Dirty | Live session | Task | Class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `main` | — | — | 2026-10-15 09:06:28 -0500, `fe42017` | — | not checked out. Detached gate worktree at this commit | gate worktree dirty (this pass) | no | none | keep |
| `wave8-safetensors-coverage` | 0 | 1 (the merge commit) | 2026-10-14 21:25:10 -0500, `4b0b3b8` | yes. `git merge-base --is-ancestor` exit 0 | none | — | no | `ft-e0211ba7412` is review, inferred | deleted after backup |
| `wave10-characterization-specials` | 0 | 1 | same `4b0b3b8` | tip is on main | primary checkout | 6 modified tracked files, plus untracked reports and pool manifests (GitPulse dirty_files 12) | grok pid 34301 | none bound | in use. keep |
| `wave1-l3-armb-corpora` | 0 | 76 | 2026-10-13 18:27:24 -0500, `5ba07a7` | yes. `git cherry` empty | `.claude/worktrees/agent-aee488bfe3833f10e` | GitPulse dirty_files 12 | no live cwd. On-disk Claude worktree, not a live process | `ft-b1dfb6da273` review, inferred | merged, keep |
| `wave4-l1-own-swift-clean-rows` | 0 | 53 | 2026-10-13 21:30:53 -0500, `4484091` | yes. `git cherry` empty | `.claude/worktrees/agent-a3ceb68fc1021f5c2` | GitPulse dirty_files 10 | no live cwd | none bound | merged, keep |
| `span-pin-8a531e8-verdicts` | 1 | 243 | 2026-10-14 16:56:14 -0500, `1d42f37` | no. `git cherry` leading `-` (patch-id already on main) | `/Users/bharath/qd-campaign/worktrees/span-pin-8a531e8` | clean | no | none bound | keep. Pin checkout |
| `ojas-pin-88be893` | 1 | 2 | 2026-10-15 08:04:59 -0500, `21fb7ad` | no. `git cherry` leading `+` | `.claude/worktrees/ojas-pin-88be893` | clean | no | `ft-3163ce2315` review, related, not a binding | keep. Pin sha is stale |
| `origin/main` | 0 | 202 | 2026-10-08 15:41:56 -0500, `f7cb935` | ancestor of local main | not checked out | — | — | `ft-be802e0780` inbox asks a human about push | keep. Not touched |

`origin/HEAD` points at `origin/main`. There is no local `origin/outreach-2026-10-06`. The server was not contacted. There is no release branch besides `main`.

Collisions (`overlapping_files` 8, `shared_worktrees` empty): untracked `data/pool/` paths across the primary checkout, wave1, and wave4; `.venv` between wave4 and the gate worktree; `ledger/runs.jsonl` between the primary checkout and the gate worktree. Those pool paths were not committed.

Codeintel facet after the delete: `is_fresh` false, generation 3914, analyzer payload obsolete, `source_freshness` true. This pass did not edit source. The merge was of a commit already on the primary branch.

## Classification review

Fable (`claude -p --model claude-fable-5-1`) reviewed the classification earlier this session. Verdict used here: merge `wave8-safetensors-coverage` only. Keep everything else.

`ojas-pin-88be893` had been the other unique patch. Fable overrode it to keep. Confirmed again at gate time: the commit pins ojas `132b46ea63e87868f468c016e02fe99acf2bb164`, `upstream.lock.json` on `main` pins `bc498c4375c2e4753d784e6e9fc267a8d18b038a`, and the ojas checkout's loose `refs/heads/main` is `a0453c3cfa45544f20447e034a24d388f05d6f61` (it overrides the packed-refs line `ece87d68cd032050de2b18e71934920e9e1b05d7`). Merging `21fb7ad` would not make `make pin` pass. The approval exists in the commit message.

`span-pin-8a531e8-verdicts`: `git cherry -v main` prints `- 1d42f37`. The patch is already on main. The branch stays because `tools/span_refit_b.sh` pins that checkout. Tag `span-pin-8a531e8-verdicts-2026-10-14` peels to `537902565b277fd76a87d9ada5ae958e3d20cbcd`, which is not the branch tip. The 2026-10-14 report said the tag peels to the branch tip. That claim is wrong. The tag was not moved.

Wave1 and wave4 are ancestors with empty `git cherry` output. Their worktrees are dirty, so they stay. Wave10 is the same commit as wave8 and is the live checkout, so the name stays.

No ambiguous branch was deleted.

## Backups

Created before any delete. Each ref matched the branch tip at backup time.

| Ref | Commit |
| --- | --- |
| `refs/backup/2026-10-15/main` | `5660ad60cbcd7fdd92de4bde641773656f99274b` (pre-merge main) |
| `refs/backup/2026-10-15/wave8-safetensors-coverage` | `4b0b3b859ee9555ce0f5b9f5b08d041511de0eec` |
| `refs/backup/2026-10-15/wave10-characterization-specials` | `4b0b3b859ee9555ce0f5b9f5b08d041511de0eec` |
| `refs/backup/2026-10-15/wave1-l3-armb-corpora` | `5ba07a7347b2a60ac1ddfe246c0c97948f0ef8ee` |
| `refs/backup/2026-10-15/wave4-l1-own-swift-clean-rows` | `44840917ef8ef81c2f422d2359b124b7abb0f3bc` |
| `refs/backup/2026-10-15/span-pin-8a531e8-verdicts` | `1d42f37d01aec8c283e8144bd8d701be6a9c99aa` |
| `refs/backup/2026-10-15/ojas-pin-88be893` | `21fb7addefc9df80c948a5dd005c0fe10a3e5b81` |

Older `refs/backup/2026-10-13/*` and `refs/backup/2026-10-14/*` were left in place.

Bundles, both under gitignored `build/`. `git bundle verify` said each is okay and records a complete history. The write gate refused a copy outside the repo.

| File | What it contains | sha256 |
| --- | --- | --- |
| `build/lappi-decision-refs-2026-10-15.bundle` | 81 refs, taken before the merge. `main` in this bundle is `5660ad6`. It does not contain `fe42017` | `48683f4ac2abedb79e9066eac0d1f0f0b62b8c67c220f22189b94cf187bf0ab7` |
| `build/lappi-decision-refs-2026-10-15-postmerge.bundle` | 82 refs, taken after the merge and before `git branch -d`. Contains `fe42017` and `refs/heads/wave8-safetensors-coverage` | `2b58c47e31a3bce7fb8d3f5ca69d9ab08c67dea11ce338e4d9d443b89160efb6` |

Copy them out:

```
cp build/lappi-decision-refs-2026-10-15.bundle "$HOME/qd-campaign/backups/"
cp build/lappi-decision-refs-2026-10-15-postmerge.bundle "$HOME/qd-campaign/backups/"
```

## Pending work

Nothing was committed.

Left unstaged on the primary checkout, owned by grok pid 34301, not by this pass: `gaps.jsonl`, `ledger/runs.jsonl`, `python/qd_train/remap.py`, `python/tests/test_containment_exclusions.py`, `python/tests/test_remap.py`, `tools/real_tokenizer_pipeline.py`. Untracked pool manifests and `BUILD_REPORT.md` were not added. This report is untracked on purpose.

Wave1 and wave4 dirt is untracked pool data and, on wave4, a `.venv`. Not committed.

## Merges

One merge. No rebase. No file in this repo asks for linear history. The primary worktree is the other session's dirty checkout, so this was not a checkout merge. `git merge-tree --write-tree` wrote tree `be7d0c824f2a9343e7987eddfb3d27d54143b3cb`, equal to `4b0b3b8`'s tree. `git commit-tree` then `git update-ref refs/heads/main fe42017 5660ad6` (expected-old check succeeded).

| Merge commit | Parents | Branch | Tree |
| --- | --- | --- | --- |
| `fe4201759566d55584310e2cb46d05eaa2ebc433` | `5660ad6`, `4b0b3b8` | `wave8-safetensors-coverage` | `be7d0c8` |

Message: `Merge branch 'wave8-safetensors-coverage'`.

Files the merge brought in (`git diff-tree --name-only -r 5660ad6 fe42017`):

- `AUDIT/board-sweep-2026-10-14/fable-safetensors-coverage-ruling.md`
- `crates/qd-export/src/safetensors.rs`
- `crates/qd-export/tests/fixtures/safetensors-adversarial.json`
- `crates/qd-export/tests/safetensors_adversarial.rs`
- `gaps.jsonl` (the committed line from `4b0b3b8`, not the other session's unstaged lines)
- `python/qd_train/memory.py`
- `python/tests/test_safetensors_one_reader.py`

Those paths do not include `upstream.lock.json`, `Cargo.lock`, or `python/tests/test_pin_gate.py`.

## Conflicts

None. `git merge-tree` exited 0 and its tree matched the branch tree. There was no ours/theirs resolution. Opus was not asked to review a resolution, because there was no conflict.

## Deleted

| Branch | Backup ref | How |
| --- | --- | --- |
| `wave8-safetensors-coverage` (`4b0b3b8`) | `refs/backup/2026-10-15/wave8-safetensors-coverage` | `git branch -d`. Git printed `Deleted branch wave8-safetensors-coverage (was 4b0b3b8).` Exit 0 |

It was an ancestor of `main`, it was not checked out, and the only tracked mention of the branch name is in untracked `BUILD_REPORT.md`. `wave10-characterization-specials` still points at `4b0b3b8`. No `-D` was used. No worktree was removed.

`git worktree remove build/main-gates-2026-10-15` refused: the gate worktree contains modified or untracked files (the ledger row this pass appended, and a `.venv` symlink). It was left in place. It is detached at `fe42017`. Removing it needs `git worktree remove --force` after you have copied anything you want from `build/main-gates-2026-10-15/ledger/runs.jsonl`. This pass did not pass `--force`.

## Kept

| Kept | Reason |
| --- | --- |
| `main` | The branch this pass updates. 202 commits ahead of `origin/main`. Not pushed. Not checked out in the primary worktree. |
| `wave10-characterization-specials` | Same commit as the deleted branch, checked out here, dirty, and grok pid 34301 is live on it. |
| `wave1-l3-armb-corpora` and its worktree | Commits are on main. Worktree is dirty. |
| `wave4-l1-own-swift-clean-rows` and its worktree | Commits are on main. Worktree is dirty. |
| `span-pin-8a531e8-verdicts` and `/Users/bharath/qd-campaign/worktrees/span-pin-8a531e8` | The one commit not on main has the same patch-id as a commit already on main. The checkout is the span-refit pin. |
| `ojas-pin-88be893` and `.claude/worktrees/ojas-pin-88be893` | Unique patch, and it pins a sha the ojas checkout is no longer on. Clean worktree. Left for a new pin commit. |
| `origin/main` | Remote-tracking ref. A remote delete was not requested, and main has not been pushed. |
| `refs/backup/2026-10-13/*`, `refs/backup/2026-10-14/*`, `refs/backup/2026-10-15/*` | Backup refs. |

## Blocked

No merge was aborted. `ojas-pin-88be893` was not merged. What it needs, before anyone merges a pin: a new commit whose ojas sha is the checkout you intend to build against (today the loose ref is `a0453c3`, origin/main of ojas is still `bc498c4`), and an explicit yes. The existing `21fb7ad` message is not that yes.

## Verify

There is no `ci:local` target. The local gate is `make gates`.

First attempt, `build/gates-fe42017.log`, was aborted. A worktree nested under `build/` makes `../../../tessl` and `../ojas` resolve to `build/tessl` and `build/ojas`, which did not exist. Clippy, fmt, and pin failed while reading that path. Lint had already passed. The run was stopped, including orphaned pytest pid 46564 from that attempt. `build/tessl` and `build/ojas` are now symlinks to `/Users/bharath/Code/research/tessl` and `/Users/bharath/Code/research/ojas`. The side ledger row `d7e6cc75-bcf6-4ee0-b333-d8f0d9283882` is that killed attempt (`status=killed`). It is not a test result.

Second attempt, `build/gates-fe42017-r2.log`, on detached `fe42017`. `make` exit 2. Tool duration 5432 s. Ledger row `7016657c-8d21-4861-83b0-a57213c3b240` in the gate worktree's `ledger/runs.jsonl` only. `status=completed` means the recorder finished the row. It does not mean the suites passed. `code_commit` is `fe4201759566d55584310e2cb46d05eaa2ebc433-dirty` because that worktree had the `.venv` symlink and the new ledger line. The row was not copied onto the primary ledger and was not committed.

| Gate | Result |
| --- | --- |
| lint | PASS |
| clippy | PASS (`cargo clippy --all-targets -- -D warnings`, finished in 27.27s) |
| fmt | PASS (no diff printed) |
| pin | FAILED exit 1. ojas is `a0453c3cfa45544f20447e034a24d388f05d6f61`; the lock pins `bc498c4375c2e4753d784e6e9fc267a8d18b038a`. The merge does not touch the lock. Same class of failure as the 2026-10-14 gate, which failed pin against a different ojas sha. |
| ledger-record | NotRun. Not a pass |
| cargo_test_workspace | Not run. Killed at the recorder's 1800s cap |
| pytest_python_tests | Not run. Killed at the same 1800s cap |
| pytest_torch_python_tests | Not run. Killed at the same 1800s cap |
| ledger-verify | PASS. 234 rows in the gate worktree ledger |

The cap is `DEFAULT_SUITE_TIMEOUT_S = 1800` in `python/qd_train/ledger.py`. It was not raised. On 2026-10-14, cargo_test_workspace was also Not run at that cap, while pytest did finish. This run did not finish pytest inside the cap either. A cut-off suite has no result.

The safetensors tests the merge adds were run afterwards, on the gate worktree, outside the recorder. They are not a ledger row.

| Command | Result |
| --- | --- |
| `cargo test -p qd-export --test safetensors_adversarial` | exit 0. 3 passed |
| `cargo test -p qd-export --lib safetensors` | exit 0. 8 passed, 6 filtered out by the name filter |
| `pytest python/tests/test_safetensors_one_reader.py` | exit 0. 124 passed, 53 skipped |

The merge was kept. Pin's failure is not caused by this diff, and the new tests that did run passed. The full workspace suites were not examined. `make gates` is not green.

`git fsck --full` exit 0. 215 dangling objects. 0 missing or broken lines. Output: `build/fsck-2026-10-15.txt`.

After the delete, `gitpulse_insights` still shows the primary worktree on `wave10-characterization-specials`, grok pid 34301 still alive, and the other worktrees on their same branches. No live session lost a branch. Wave8 had no worktree and no live session.

## Restore

Bring the deleted branch back:

```
git branch wave8-safetensors-coverage refs/backup/2026-10-15/wave8-safetensors-coverage
```

Or, after copying the post-merge bundle out:

```
git fetch /absolute/path/lappi-decision-refs-2026-10-15-postmerge.bundle refs/heads/wave8-safetensors-coverage:refs/heads/wave8-safetensors-coverage
```

Put `main` back to the pre-merge commit, only while `main` is still `fe42017` and no other session has moved it:

```
git update-ref refs/heads/main 5660ad60cbcd7fdd92de4bde641773656f99274b fe4201759566d55584310e2cb46d05eaa2ebc433
```

The other backup tips from this pass, if a name is ever deleted later:

```
git branch wave10-characterization-specials refs/backup/2026-10-15/wave10-characterization-specials
git branch wave1-l3-armb-corpora refs/backup/2026-10-15/wave1-l3-armb-corpora
git branch wave4-l1-own-swift-clean-rows refs/backup/2026-10-15/wave4-l1-own-swift-clean-rows
git branch span-pin-8a531e8-verdicts refs/backup/2026-10-15/span-pin-8a531e8-verdicts
git branch ojas-pin-88be893 refs/backup/2026-10-15/ojas-pin-88be893
git branch main-2026-10-15-pre-wave8 refs/backup/2026-10-15/main
```

## Appendix: restore commands from the 2026-10-14 and 2026-10-13 reports

The previous `BRANCH_REPORT.md` was copied, unchanged, to `build/branch-report-prior-2026-10-14.md` before this file replaced it. Its restore blocks:

```
git branch main-2026-10-14-evening refs/backup/2026-10-14/main
git branch wave1-l3-armb-corpora refs/backup/2026-10-14/wave1-l3-armb-corpora
git branch wave4-l1-own-swift-clean-rows refs/backup/2026-10-14/wave4-l1-own-swift-clean-rows
git branch span-pin-8a531e8-verdicts refs/backup/2026-10-14/span-pin-8a531e8-verdicts
```

```
git fetch /absolute/path/lappi-decision-refs-2026-10-14.bundle 'refs/heads/*:refs/heads/*'
```

```
git branch <name> refs/backup/2026-10-13/<name>
git fetch ~/branch-backup-2026-10-14-final.bundle 'refs/backup/2026-10-13/<name>:refs/heads/<name>'
git worktree add .claude/worktrees/agent-<id> <name>
cp -Rp build/preserved-worktrees-2026-10-14/agent-<id>/. .claude/worktrees/agent-<id>/
git branch pre-consolidation-main refs/backup/2026-10-13/main
```

`refs/backup/2026-10-13/main` is `05b82375ce0a320b7939402ef17353d4e035b0a8`.
