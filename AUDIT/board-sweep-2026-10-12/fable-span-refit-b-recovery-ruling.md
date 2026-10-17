# Fable: span-refit arm B recovery ruling (2026-10-14, main 75f2019)

Asked after main's `build/` was wiped (GAP-MAIN-BUILD-DIR-WIPED-SPAN-CACHES-LOST-2026-10-14,
GAP-SPAN-REFIT-B-LAUNCHER-B-SH-GONE-2026-10-14): is B still worth running, and how.
Labels: **[V]** read or ran this session; **[R]** a record's claim I could not re-run (its file is gone);
**[I]** inferred; **[U]** unverified. I wrote only this file: no gaps append, no branch, no run.

Navigation. `devmap_status`: generation 3890, `is_fresh=false` ("source tree differs from the indexed
generation; stored extraction payload is obsolete"); the CLI `devmap status` says `analyzer_freshness: true`.
The two disagree; the lane records that as a gap. `devmap_explore span_refit` gave the symbol inventory of the
six `tools/span_refit_*` files (`walk_incomplete`: 1 file failed to parse, 36 unbound call sites) — I used
it for the inventory only. `gitpulse_insights`: every facet `ok`; two Claude sessions in this worktree
(pid 12401 busy, 47940 idle), no worktree touches `AUDIT/` or `tools/span_refit_*`. `ListAgents` is not a
tool I have; I mitigated by writing one new file. Everything else below is `rg`, `Read`, `ls` and `git`.

## 1. Decision: RUN (redo), staged so the gate is bought first

**What B feeds** [V]: ft-0f5a10c9's criterion "the served span head is a named single-seed or refit head,
not the 5-seed blend, with its span top-1 cited from a ledger row", and the 2026-10-12 ruling §3: v0.1.x's
head follows the prereg's reading among (b), (c), (d) once B has rows. **Nothing else can produce that on
this Mac** [V]: A2 (the `--verdicts-out` re-score) is refused by the pairing check
(GAP-RESCORE-A2-AVERAGE-INPUT-JSONS-ABSENT-PAIRING-REFUSAL-2026-10-12); the box is deleted; the only
row for a non-blended head is e1bafd6f (seed-1 tower + head, 6,626), which abandons the averaged tower
and the calibration table fitted on it. Arms (a) and (b) come from **one** avg-val pass
(`span_refit_extract.py:487-490` records `live_top_own_head` and `live_top_seed1_head` per row), so the
cheapest stage already answers "does seed 1's head on the averaged tower restore >= 6,562". The
pre-registered reading `b_vs_c` needs (c), so the full chain follows — unless the gate fails, in which
case the prereg's own rule stops everything (`arms.a.role`; scorer exit 3, `span_refit_score.py:290-295`).
Not DEFER: the qd-metal loader (criterion 1) needs a head to load, and that head is this reading. Not
DROP: rule 8 keeps the result report-only, but the human's head choice has no other measurement.

**What is lost** [V dir gone; R contents]: all of `build/needle-span-mac-2026-10-06/` — `cache/avg-val`
(7,238 rows, arm (a) 5,596 and `census.json` b05069fd that amendment 1 cites), `cache/avg-train` (12,000),
`cache/seed1-val` (4,811 of 7,238), `heads/`, `results/`; `build/w3-rescore/`; the worktree and branch
`span-pin-8a531e8-verdicts` (no ref: `git for-each-ref` has no span ref, neither bundle lists one,
`git worktree list` shows main plus two agent worktrees). `~/qd-campaign` holds no `*span*` directory and
`rescore-2026-10-12/` is empty. **Inputs are intact** [V ls]: `phase4-v5r-2026-10-03/shards/{val,train}`,
`preview-2026-10-05/v5-avg/`, `weights-2026-10-05/v5-seed1/` (safetensors + json),
`ledgers-h100-2026-10-05/`, and the backbone snapshot path the extractor defaults to.

## 2. How, in order

**S0 — the pin tree, as a tracked ref (agent, light).** [V] `git cat-file -t a1b1799` → commit,
`8a531e8` → commit, both ancestors of main; a1b1799's `tools/real_ft_run.py` hunk is 41 lines, and
`git merge-tree --write-tree --merge-base=a1b1799~1 8a531e8 a1b1799` merges with no conflict (exit 0).
B itself never calls `_decode`, so the hunk is for A2; one tree serves both, as the 2026-10-12 ruling
set. Build it without a checkout, from the repo root: temporary index ← `git read-tree 8a531e8`;
`git show a1b1799 -- tools/real_ft_run.py | git apply --cached` into that index; `git write-tree`;
`git commit-tree <tree> -p 8a531e8` with the ruling's message; then `git branch span-pin-8a531e8-verdicts
<sha>`, an annotated tag `span-pin-8a531e8-verdicts-2026-10-14` on the same sha (a worktree removal
cannot drop a tag), and `git worktree add /Users/bharath/qd-campaign/worktrees/span-pin-8a531e8
span-pin-8a531e8-verdicts` — not under `build/`. Checks: `git diff --stat 8a531e8 <sha>` names only
`tools/real_ft_run.py`; `git diff a1b1799 <sha> -- tools/real_ft_run.py` is empty. If the harness refuses
the temporary-index form, the human runs `git apply` + `git commit` inside the worktree instead. The sha
goes in the handoff and the row's `code_commit`. [U] whether B needs the 21 pool symlinks the old pin
carried (they provisioned A2's split rebuild); the first extraction log says.

**S1 — avg-val extraction, then the gate (human, MPS ~30-45 min [I: runner 2 ran 18:18-18:47Z before
its OOM]).** `--tower avg --split val --device mps --out $ROOT/cache/avg-val`, then
`span_refit_score.py --gate-only --val-cache $ROOT/cache/avg-val --out $ROOT/results/gate.json` (CPU).
Gate failed → stop, record the gap, nothing else runs. **S2 — avg-train (human, MPS ~1 h+, resumable;
OOM risk at the 48 GiB MPS ceiling, GAP-MAC-HEAVY-RSS-CAP-MAY-NOT-COUNT-METAL-2026-10-06: resume, do not
widen batches).** **S3 — fit c, fit e** (`--cache $ROOT/cache/avg-train --out $ROOT/heads/arm-{c,e}`).
**S4 — seed1-val** (human, MPS ~50 min [I: 23 min to batch 321/652]). **S5 — rows** (agent):
`tools/span_refit_rows.py` with `--val-cache`, `--head-c`, `--head-e`, `--seed1-val-cache` under `$ROOT`,
all five arms, then the handoff with digests. Run S1-S4 when no other lane starts a `mac_heavy` job
(GAP-RESCORE-B1-EXTRACT-WRAPPER-SIGTERMED-TWICE-2026-10-12 is open; a third exit 143 is captured, not retried).

**Where caches live.** `SPAN_REFIT_ROOT=/Users/bharath/qd-campaign/span-refit-2026-10-06/` holding
`cache/`, `heads/`, `results/`, `logs/<date>/`. `build/` is gitignored (`.gitignore:17`) and was wiped
once; `~/qd-campaign` is where every other campaign artefact survived. The four Python scripts keep their
`C.BUILD` defaults untouched (so `span_refit_rows.py`'s script-sha pins and `assert_path_not_held_out`
stay as they are); the launcher passes explicit `--out`/`--cache`/`--val-cache` on every step. The
prereg's `resources.cache` names `build/…`; the file is not edited — the location is a resource, not a
measurement, and the deviation is stated in each row's `notes` and the handoff with this ruling as reason.

**Is the avg-train redo a pure re-run?** Partly. [V] `select()` (`span_refit_extract.py:103-144`) and the
column derivation are deterministic in the shard, the prereg and seed 20261006, and the extractor refuses a
layout that differs from one on disk (`:430-436`): `layout.json`'s `order`, `cand_offsets`, `selection`
and `rows.jsonl`'s gold/status fields will equal the lost ones. [V] The feature bytes are an MPS fp32
forward with no determinism setting (`rg manual_seed|deterministic` over the two scripts: none) and batch
composition depends on where a resume restarts, so `hq.f32`/`cand_*.bf16` are not expected bit-equal [I].
**No digest of the lost caches was ever recorded** [V: `rg` over `gaps.jsonl`, `HANDOFF/`, `AUDIT/` finds
only the row count 12,000 and the byte size 98,304,000]; `heads/*/span_head.manifest.json` would have
carried `inputs_sha256` (`span_refit_fit.py:304-306`) and is gone. So equality to the lost extraction
cannot be checked and must not be claimed. The check the prereg itself provides is the gate: arm (a) live
fp32 in [5,581, 5,611] (the prior run: 5,596 exactly, cache fidelity 7,237/7,238 [R]). This time the
handoff records sha256 of `layout.json`, `rows.jsonl`, `hq.f32`, `cand_start.bf16`, `cand_end.bf16` per cache.

**`tools/span_refit_b.sh` changes (one commit with its tests, red first against 7e224a9):**
1. `SPAN_REFIT_ROOT` (default above) replaces the hard-coded `build/needle-span-mac-2026-10-06` paths and
   `LOG_DIR`; a root that resolves under `$REPO/build` is refused (exit 64) before any step is shown.
2. `--redo`: prepends `extract-avg-val`, `gate` (scorer `--gate-only`; exit 3 → launcher exit 68, no
   further step) and `extract-avg-train`, then the existing verify-then-seed1/fit-c/fit-e. Without
   `--redo` the exit-66 refusal stands. `--print --redo` lists six STEP lines and no SKIP line.
3. In run mode, `SPAN_CODE_PIN` is required beside `SPAN_CODE_REPO`: refuse (exit 67) unless the pin
   tree's HEAD equals it and `git status --porcelain --untracked-files=no` is empty — the remedy
   GAP-SPAN-CHAIN-UNPINNED-TREE-MAIN-CHANGED-UNDER-IT-2026-10-06 names. `--print` still verifies nothing.
4. Tests in `python/tests/test_span_refit_b_launcher.py`: `--redo --print` step order and `--out` paths
   under a tmp root (pre-fix: exit 64 usage); a root under `$REPO/build` refused (pre-fix: accepted); a tmp
   git repo whose HEAD is not `SPAN_CODE_PIN` exits 67 before any step (pre-fix: 66 from the cache check);
   the existing three-step expectation gains the explicit paths (updated, not weakened: the `build/` paths
   it pinned are the defect). The gate exit mapping is verified by S1's log, not by a test.

## 3. Gaps

Close when the evidence exists, by append: GAP-SPAN-REFIT-B-LAUNCHER-B-SH-GONE-2026-10-14 (7e224a9 plus
the `--redo` commit); GAP-SPAN-REFIT-ARMS-B-E-UNSCORED-NO-LEDGER-RECORDER-2026-10-12 (recorder landed in
3fa3785/e723043; closes on the five row ids); GAP-SPAN-CHAIN-UNPINNED-TREE-MAIN-CHANGED-UNDER-IT-2026-10-06
(the pin check, once a run passes it). Stay open: GAP-MAIN-BUILD-DIR-WIPED-SPAN-CACHES-LOST-2026-10-14
(deleter unknown, ft-ad52698d; append the redo rows); GAP-RESCORE-VAL-SHARD-STALE-VS-LOGIT-WRITING-CODE-2026-10-12
(A2's pairing refusal is untouched by this ruling; append the new pin sha);
GAP-RESCORE-B1-EXTRACT-WRAPPER-SIGTERMED-TWICE-2026-10-12 (sender unknown). New, for the lane to append:
the DevMap freshness disagreement above; "no digest was recorded for a cache a decision rested on";
"a local-only branch was lost with its worktree" (the tag + `refs/backup/` remedy). Human's calls, unchanged:
the release table, the v0.1.x head after the reading, and the pin-as-provenance veto from 2026-10-12 §1.
