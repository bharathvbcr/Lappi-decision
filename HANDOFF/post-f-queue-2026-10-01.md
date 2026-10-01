# HANDOFF: post-F queue (items 0, 2S, 3-10), 2026-10-01

Lane: worktree `agent-a0240fba0d690aeda`, branch `worktree-agent-a0240fba0d690aeda`, fast-forwarded
to main `4fd08cf` before any change. The rules are Fable's post-F ruling,
`campaign/f-j7prime-preregistered.json` (`4fd08cf`); its queue is items 0-10. Items 1 (J4 ens3,
`box_q_j4ens3.sh`) and 2 (J4 avg-np, `box_q_avgnp3.sh`) were already deployed by the lead and
are not touched here. Items 8 and 9 run the no-mask lane's staged box scripts (main `669550d`:
`tools/perf_nomask_p2.sh`, `tools/perf_tierb_outcome.sh`), which this branch does not carry or
change; it only names their box paths. Nothing ran on the box or on any GPU; no ssh, no push.

**For the lead to decide before deploying: item 0's start.** The brief says the chain must start
on `avgnp.done`. Every GPU item does. Item 0 (CPU only, holds no lock) starts on `f.done`,
because the ruling runs it "parallel to 1-2" and starting it later leaves the GPU idle after
item 2 while it averages (item 3 needs its `j7pavg.done`). The cost of `f.done`: item 0's
`ckpt_average` passes (niced; avg-np peaks near 44 GiB RAM, inferred) overlap items 1-2's scoring.
To start it on `avgnp.done` instead, change line 22 of `campaign/post-f-queue/box_q_j7p_cpu.sh`,
`until [ -f $Q/f.done ]`, to `avgnp.done` (no other line depends on it).

## What was measured

**On the box or a GPU: nothing.** Every row below is an existing committed row read on the Mac.

`qd-post-f-rules` on J4's committed rows (the brief's expected outcomes, all met):

| Rule | Rows read | Result |
| --- | --- | --- |
| (i) avg-np qualifies | J7g's average scored as if it were avg-np: `1d93b3ee` (gate) + `0b86fae3` (control), retagged in a temp ledger | **fails**: (a) `ood_abstain` 0/180 < 34/180. (b) is *ambiguous* (below), (c) passes |
| (ii) seeds 3-4 | J4 eval rows `2f5fe57a` / `f3612f73` / `02cf5ff4`, pinned to ft rows `80b19a41` / `9b108fe2` / `8b600511` | **fires**: 8K worst 45/60 (seed 0, 60-80%) minus 0/59 (seed 2, 40-60%) = 0.75 > 0.30 |
| (iii) J6(f) early | the same rows | **fires**: 8K worst < 0.95 on 3 of 3 seeds; prose 3 / 9 / 2 (<= 9) and scrambled 3 / 4 / 8 (<= 8) on every seed |

### What is ambiguous in (b), exactly (for Fable)

The pre-registration prints each threshold as a 3-decimal number **and** cites the row it was read
from. The decimals are roundings of those rows' fractions, and they do not sit on them:

| Clause | Printed | Cited row's exact value | A candidate that ties the cited row |
| --- | --- | --- | --- |
| (b) 8K worst | `>= 0.279` | `1d93b3ee` 80-100%: 17/61 = 0.27869 | passes "not below the average", **fails** ">= 0.279" |
| (b) 1K control | "not below the average's 0.895" | `0b86fae3` 0-20%: 51/57 = 0.89474 | same split |
| (b) 2K control | "... 0.767" | `0b86fae3` 80-100%: 46/60 = 0.76667 | same split |
| (b) 4K control | "... 0.426" | `0b86fae3` 80-100%: 26/61 = 0.42623 | both pass (the decimal is below the fraction) |
| (c) choice | `>= 0.765` | `f3612f73`: 8405/10985 = 0.76514 | 8404/10985 = 0.76505 **passes** ">= 0.765", fails "J4 seed minimum" |
| (c) span | `>= 0.884` | `2f5fe57a`: 6399/7238 = 0.88408 | same split, other direction |
| (a) OOD | `>= 34/180` | `6fcba23d` `ood_diagnostic.all`: 34/180 | integer: one reading |

So for the 8K, 1K and 2K parts, a candidate exactly equal to the J7g average passes one reading
and fails the other; for (c), a candidate in the sliver between the decimal and the seed minimum
does the same. **The checker does not pick a reading.** Each comparison is made both ways, exactly
(integer cross-multiplication on `n`/`n_total`, never the float). A clause whose two readings
disagree is `ambiguous`; (i) is the Kleene conjunction of (a), (b), (c):

- any clause **fails under both readings** -> `fails` (the outcome is the same whichever reading
  holds, so no reading was chosen);
- otherwise, any clause ambiguous -> **refused, exit 3** ("which reading holds is the human's call
  (rule 2)"); item 3 then logs avg-np NOT RUN and the human decides;
- otherwise `qualifies`.

On J4's stand-in rows (a) fails 0/180 under its only reading, so the answer is `fails` while (b)
is reported `ambiguous` in the JSON (8K, 1K and 2K parts). If Fable wants a refusal whenever *any*
clause is ambiguous, even when another clause already fails, that is a one-line change in
`rule_avgnp` (`Verdict::Ambiguous` checked before `Fail`); it is not made, because it would refuse
a case the rule decides. Test: `a_tie_with_the_cited_row_where_the_decimals_disagree_refuses`.
The checker also refuses if a cited row ever stops rounding to its printed decimal.

## What changed

| Commit | What |
| --- | --- |
| `0e99282` | `crates/qd-runtime/src/bin/qd_post_f_rules.rs` + its `[[bin]]`; `GAP-POST-F-QUEUE-NAVIGATION-2026-10-01` |
| `9715e8b` | `campaign/post-f-queue/` (8 waiters + `post_f_common.sh`); `AUDIT/j7-avg-ood-diag-2026-10-01/delta_cosine.py` takes argv |
| `0b235ef` | items 8 and 9 wired: `box_q_nomaskp2.sh`, `box_q_nomask.sh`; items 4, 6, 7, 10 re-ordered around them; `wait_queued` replaces the lock re-check loops |
| (this file) | the HANDOFF |

Size: +~2,000 lines (checker 1,040 of which ~560 are tests; scripts ~560; this file), -29
(delta_cosine's hardcoded paths), and -55 +93 in the items 8/9 re-wire (two new waiters; the lock re-check loops removed). `Cargo.lock` unchanged (cargo keeps re-resolving the external
`tessl` path dep; that drift is restored, never committed).

### `qd-post-f-rules` (Rust, `crates/qd-runtime`)

**Placement:** `qd-runtime` already holds the repo's offline ledger-row readers, `qd-gate-report`
and `qd-calib-fit`, and its dependencies (serde_json, clap, sha2) cover JSON, the CLI and the
input hashes. No new dependency; no lockfile change; it does not link the runtime library.

| Subcommand | Prints (stdout) | Refuses (exit 3) on |
| --- | --- | --- |
| `avgnp --j7-ledger --j4-ledger --out` | `qualifies` / `fails` | no or several completed `avg-np-score-val` / `avg-np-needle-length-control` rows naming `1d93b3ee`'s three ft rows; dtype not the average's; a reference row missing or drifted; an ambiguous tie (above) |
| `seeds34 --f-ledger --ft-row 0=ID --ft-row 1=ID --ft-row 2=ID --out` | `fires` / `quiet` | seeds other than 0, 1, 2 in order; an ft row not a completed ft row of that seed; not exactly one completed `epoch-score-val` eval row whose `ft_run_row_id` is that ft row |
| `j6f` (same args) | `fires` / `quiet` | the same, plus an OOD category not out of 60 |
| `ft-rows --ledger --ft-row S=ID...` | the ids, in order | not completed, `quick` not `false`, tag not `epoch`, a `shuffled_label` recipe, wrong seed, or recipe hash / data snapshot differing between rows |
| `eval-row --ledger --ft-row S=ID` | the eval row id | not exactly one |

Common refusals: a malformed ledger line (a half-written last line included), a metric absent or
not `ran` (its reason is quoted), a `value` that is not its own `n/n_total`, a needle gate whose
value is not the minimum of its own five depth buckets, an existing `--out`. JSON goes to `--out`
(create-new + fsync) and to stderr, with every ledger's path, sha256, bytes and row count. clap
usage errors exit 2. Never a default.

**Tests: 23**, in the bin (`cargo test -p qd-runtime --bin qd-post-f-rules`), 4 on J4's real
committed rows, the rest on synthetic boundaries (spread exactly 0.30 quiet / one hit more fires,
with unequal denominators too; 57/60 = 0.95 not below the floor; one seed below vs two; prose 9
vs 10, scrambled 8 vs 9; OOD 34 vs 33; every control length; the decimal/row splits) and
refusals. One test asserts every threshold's text appears verbatim in both pre-registration files;
another that each decimal's number is its text. `cargo clippy -D warnings` clean; full
`cargo test -p qd-runtime` passes.

**Fail-first evidence** (each threshold or comparison set wrong in the source, tests run, source
restored; `git diff` clean afterwards):

| Mutation | Tests that fail |
| --- | --- |
| spread 0.30 -> 0.80 (number only) | `every_decimal_is_the_fraction_its_text_prints`, `a_spread_of_exactly_0_30...`, **`j4s_seeds_spread_by_0_75_so_seeds_3_and_4_fire`** |
| spread 0.30 -> 0.80 (number and text) | `every_constant_is_the_preregistrations_own_text`, `a_spread_of_exactly_0_30...`, **`j4s_seeds_spread...`** |
| spread 0.30 -> 0.25 | the pre-registration test, `a_spread_of_exactly_0_30...` |
| spread `>` -> `>=` | `a_spread_of_exactly_0_30...` |
| needle floor 0.95 -> 0.10 | the decimal test, `j6f_counts_a_seed_below_0_95...`, **`j4_fires_j6f...`** |
| floor needs 2 seeds -> 3 | the pre-registration test, `j6f_counts_a_seed_below_0_95...` |
| prose <= 9 -> <= 8 | the pre-registration test, `j6f_fires_on_prose_at_9...`, **`j4_fires_j6f...`** |
| scrambled `<=` -> `<` | `j6f_fires_on_prose_at_9...`, **`j4_fires_j6f...`** |
| OOD >= 34 -> >= 33 | 7, including **`j4s_average_scored_as_avgnp...`** and `avgnp_needs_34_ood_abstentions_not_33` |
| 8K 0.279 -> 0.250 | 6, including **`j4s_average_scored_as_avgnp...`** |
| choice 0.765 -> 0.766 (number only) | 6 (0.766 no longer rounds from `f3612f73`, so every avg-np case refuses) |

### The binary for the box

The brief's premise that `qd-prep-m1` was built on the box is not what the HANDOFFs record: the
box has no Rust toolchain, and every box binary (`qd-prep`, `qd-prep-m1`, `qd-calib-fit`,
`qd-export`, `qd-margin-probe`) was **cross-built on the Mac** against a sysroot of the box's own
glibc 2.39 (`HANDOFF/needle-ood-followup-2026-09-30.md` "Building Rust for the box";
`j7-calib-fit` and `perf-ft-run-prep` HANDOFFs). This lane ran that build:

    cd /Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-a0240fba0d690aeda
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" \
    cargo build --release -p qd-runtime --bin qd-post-f-rules --target aarch64-unknown-linux-gnu \
      --target-dir /Users/bharath/qd-campaign/target-aarch64-linux-postf

- File: `/Users/bharath/qd-campaign/target-aarch64-linux-postf/aarch64-unknown-linux-gnu/release/qd-post-f-rules`
- sha256 `330e681672897a5369a04d5d1fdafdc8b9071e31e44b8d866927c90daa43c98e`, built from `0e99282`
  (the source is unchanged since). ELF 64-bit aarch64 PIE; highest symbol version GLIBC_2.34
  (`llvm-objdump -T`), under the box's 2.39.
- **Copy this file; do not rebuild it elsewhere.** `post_f_common.sh` pins this sha256 and every
  script refuses another. A build from a different checkout path embeds different paths and will
  not match; if you must rebuild, update `RULES_SHA256` and commit it.
- **Ran on aarch64 Linux:** in a local `python:3.11-trixie` arm64 container (glibc 2.41, no
  network) against J4's committed ledgers: `seeds34` -> fires, `j6f` -> fires, `avgnp` on the
  committed J7 ledger -> refused, exit 3, "missing row: no completed eval row tagged
  avg-np-score-val" (item 2 has not run), an existing `--out` -> refused, `ft-rows` and
  `eval-row` print J4's ids, a short id -> usage error exit 2. **Not run on the box.**

### `delta_cosine.py`

Was hardcoded to J4's paths and three seeds. It now takes `--avg-manifest`, `--base` and N
sidecars, checks each sidecar is its manifest input's (in order), and forms all pairs. On a
synthetic three-seed fixture its JSON is **byte-identical** to the committed version's (old
constants pointed at the same files); it runs on five seeds (10 pairs) and refuses sidecars out of
order. ruff clean. It is the throwaway AUDIT analysis it was; it ships nothing. sha256
`f4e373a9…` is pinned in `post_f_common.sh`.

## The queue

Markers live in `/home/ubuntu/queue`; every script touches `<job>.queued` when launched and
`<job>.done` from an EXIT trap on every path, so the GPU never waits on a human. Decisions are
`qd-post-f-rules` calls; each writes `/home/ubuntu/ledger/post-f-decisions-2026-10-01/<item>-<rule>-<UTC>.json`
and logs it.

| Item | Script (`campaign/post-f-queue/`) | Marker | Waits for | Lane | Lock | What, cap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `box_q_j7p_cpu.sh` | `j7pcpu` (+ `j7pavg`) | `f.done` (CPU, "parallel to 1-2"); `s34.done` too if (ii) fires | lane10 `3460afc` | none | (ii) -> seed set; masters average; avg-np (c from F's seeds); qd-export; delta cosine; per-seed calibration fits. `j7pavg.done` after the export |
| 1 | `box_q_j4ens3.sh` (deployed) | `j4ens3` | `f.done` | lane9 `0062bde` | yes | J4 ens3 gate row, 3 h |
| 2 | `box_q_avgnp3.sh` (deployed) | `avgnp` | `j4ens3.done` | lane9 | yes | J4 avg-np (d) + (e), 2 h + 1 h |
| 2S | `box_q_s34.sh` | `s34` | `avgnp.done` | lane8 `a502670` | yes | iff (ii) fires: F seeds 3, 4 (F's argv), 32,400 s each, each + needle control 1024/2048/4096 (5,400 s) |
| 3 | `box_q_j7p.sh` | `j7p` | `s34.done` and `j7pavg.done` | lane10 | yes | plan: `seed<N>` ood x N, `avg` gates, `ens<N>` gates, `avg-np` gates iff (i) qualifies (4 h); needle controls on avg, ens<N>, avg-np iff (1 h each); export parity; then (lock released) avg calibration |
| 8 | `box_q_nomaskp2.sh` | `nomaskp2` | `j7p.done` (item 3's rows exist) | the overlay its script builds | its run script's (`flock gpu.lock timeout 1200` per session) | `bash /home/ubuntu/perf/perf_nomask_p2.sh T1`, then `T2` whatever T1's exit; each exit logged |
| 4 | `box_q_fslice.sh` | `fslice` | `j7p.done`, then item 8 if queued | lane10 | yes | composed slice: seed0-2, avg, ens<N>; 6 h |
| 5 | `box_q_j5p.sh` | `j5p` | `fslice.done` | lane8 | yes | J5' `--shuffled-label` x3, one invocation per seed, F's recipe flags, 32,400 s each, rows in F's ledger |
| 6 early | `box_q_j6f.sh` | `j6f` | (iii) fires: `j5p.done` | lane8 | yes | J6(f): F minus `--lower-layers-*`, seed 0, skip 6, 32,400 s, + needle control. Writes `j6f.position` |
| 9 | `box_q_nomask.sh` | `nomask` | `j5p.done`; and `j6f.done` if `j6f.position` = early | `perf/p3nomask` | its run script's (per stage) | **iff `/home/ubuntu/queue/nomask-p2-ruled` exists**: `bash /home/ubuntu/perf/perf_tierb_outcome.sh nomask --run` (exit 5 = its P2 gate refused, before the lock). Absent: logs "item 9 not run: P2 rule awaiting Fable" and exits 0. Never blocks |
| 6 late | `box_q_j6f.sh` | `j6f` | (iii) quiet or refused: `j5p.done`, then item 9 | lane8 | yes | as above |
| 7 | `box_q_tierb2.sh` | `tierb2` | `j5p.done`, then items 9 and 6 | p3fused `f6a0928` | its run script's | `perf_tierb_fused.sh --run` (box_q_tierb.sh's call, unchanged) |
| 10 | `box_q_j6dv4.sh` | `j6dv4` | `avgnp.done`, then every item above that is queued, by name | lane8 | yes | J6(d) on v4: F's recipe with `--lr 3e-5 --beta2 0.95`, seed 0, 32,400 s, + needle control |

GPU order, therefore: 1 -> 2 -> 2S (iff (ii)) -> 3 -> 8 -> 4 -> 5 -> [6 if (iii) fires] -> 9 (iff
the marker) -> [6 if (iii) quiet or refused] -> 7 -> 10. Item 0 is CPU only. Every waiter's wait
is a `.done` marker, and every `.done` is touched by an EXIT trap, so a refusal or a failure
anywhere moves the chain on (logged) rather than stalling it.

New ledgers: `gh200-p6-f-j7prime-2026-10-01.jsonl` (item 3), `gh200-f-composed-slice-2026-10-01.jsonl`
(item 4), `gh200-j6-v4-ablations-2026-10-01.jsonl` (items 6, 10). 2S and J5' write F's ledger
`gh200-p4-v4-2026-10-01.jsonl` (J5' must: `--shuffled-label` finds its target there).

Choices made here that the ruling did not spell out:
- **Item 0 starts on `f.done`, not `avgnp.done`**: the decision for the lead at the top (avg-np's
  ~44 GiB peak is inferred from `HANDOFF/ens3-scoring-2026-10-01.md`).
- **Item 8 sits between items 3 and 4** ("a gap after item 3's rows exist"). Each P2 session takes
  the lock itself and releases it between T1 and T2 and for its CPU statistics; item 4 waits on
  `nomaskp2.done`, so nothing else takes the GPU in between. T1's speed benchmark can share the
  CPU with item 3's post-lock tail (calibration on the average) -- not measured.
- **Item 9's marker is read when its turn comes** (after `j5p.done`, and after J6(f) if early),
  not at launch. If the marker is touched after item 9 has already logged NOT RUN, the outcome run
  does not start on its own; run `bash /home/ubuntu/perf/perf_tierb_outcome.sh nomask --run` by
  hand (it takes the lock per stage, so it queues behind whatever holds it).
- **When (ii) fires, item 4's avg and ens are the five-seed ones** (no three-seed average exists
  then; "a three-seed average is not scored as a candidate"), with the per-seed kinds seed0-2 as
  the ruling lists them: 9 tower decodes under the pre-registered 6 h cap instead of 7. The
  ensemble runs last, so the other rows survive a cap hit. The cap is not moved.
- **J6(f) on a refused (iii)** takes the late slot and says so. Item 9 orders itself by the
  position file item 6 wrote (`after_early_j6f`), never by re-reading the rule, so the two cannot
  deadlock; item 7 waits for both.
- **No waiter holds `gpu.lock` while it waits.** Each waits on markers first (`wait_queued`: an
  item that was queued and is not done), then takes the lock (the old `box_q_j6d2.sh` `pending()`
  held it while waiting). An item whose waiter was never launched has no `.queued`, so it is
  skipped, not waited on forever; item 10 sleeps 120 s after `avgnp.done` so that, if it is
  launched late, the others have touched their `.queued` first.
- **`--approved-by`** on 2S, J5', J6(f), J6(d)-v4 (each capped 32,400 s = $20.61) reuses the
  human's 2026-10-01 words exactly as `box_q_f.sh` does, naming the job. Rule 4 refuses these
  runs without it (argv check below). **Confirm that yes covers them before deploying.**
- **Items 7, 8, 9 do not hold the lock**: their run scripts take it per GPU stage. The items that
  could take it between those stages all wait on the running item's `.done` first, so the order
  above holds.

## Mac verification

**Static:** `bash -n` on all eleven files and `shellcheck -S warning -x` on the ten waiters
(following the source into `post_f_common.sh`): clean.

**Container dry runs of the whole chain** (arm64 Debian, no network), re-run after the items 8/9
re-wire: the real scripts, the cross-built binary (sha-pinned), and the real J4/J7 rows as F's and
J4's; `python`, `git`, `sleep`, `qd-export` and the three perf scripts (`perf_tierb_fused.sh`,
`perf_nomask_p2.sh`, `perf_tierb_outcome.sh`, each faked to take `gpu.lock` as the real ones do)
record their argv and fake only the side effects the next script reads. All ten waiters launched
together; order read off the recorded calls:
- *quiet* (F's spread 0.083, (iii) quiet, J4's avg-np edited to qualify, **no** `nomask-p2-ruled`):
  0 (seeds 012) -> 2S exits -> 3 (6 kinds, controls avg/ens3/avg-np, parity) -> 8 (`T1`, `T2`) ->
  4 (5 kinds) -> 5 (x3, targets `2f5fe57a` / `f3612f73` / `02cf5ff4`) -> 9 logs "item 9 not run:
  P2 rule awaiting Fable" -> 6 late -> 7 -> 10.
- *fires* (J4's real rows; avg-np = J7g's average; `nomask-p2-ruled` **present**): 2S trains 3
  and 4 -> 0 averages five -> 3 (7 kinds, avg-np dropped "fails", controls avg/ens5) -> 8 -> 4
  (seed0-2 + avg5 + ens5) -> 5 -> 6 early -> 9 (`perf_tierb_outcome.sh nomask --run`, after
  "J6(f) took the early slot") -> 7 -> 10.

**The wiring, fail-first.** The dry runs above use instant fakes, so a dropped wait could pass
them by timing. So the chain was re-run with slow fakes -- each perf script is two 0.6 s stages
under `gpu.lock` with 0.6 s outside it between and after, the windows a mis-wired waiter would
take the lock in -- and a throwaway checker asserting that every call of each item precedes
every call of the next in the order above. Unchanged scripts: **ORDER OK** in quiet without the
marker, quiet with it, fires with it, and fires with J6(f) polling 1.5 s late. Each wait deleted in
turn: **ORDER BROKEN** every time.

| Wait deleted | Scenario | What the checker saw |
| --- | --- | --- |
| item 4's `wait_queued nomaskp2` | quiet, marker | the slice took the lock between P2's stages (8 -> 4 -> 5 -> 9 -> 8 ...) |
| item 7's `wait_queued nomask j6f` | quiet, marker | tierb interleaved with item 9 and ran before J6(f) |
| late J6(f)'s `wait_queued nomask` | quiet, marker | J6(f) ran before item 9 |
| item 10's `wait_queued ...` | quiet, marker | J6(d)-v4 ran first, before item 3 |
| item 9's `after_early_j6f` | fires, marker | **not caught** with both polling in phase (J6(f) won the race by timing); with J6(f) polling 1.5 s late, item 9 ran before the early J6(f) |

The last row is why the waits are there: on the box every waiter polls every 30 s at its own
phase, so without the wait the order would be a race.

**Argv checks:** every `real_ft_run.py` and `ckpt_average.py` call those two runs made (31) went
through `main()` of a `git archive` of its lane's commit (`a502670` for lane8, `3460afc` for
lane10), with `resolve_rev` (the first disk read after every argv refusal,
`tools/real_ft_run.py:8643` at 3460afc, `:8146` at a502670) and `ckpt_average._refuse_existing`
replaced by a sentinel. **31 / 31 ARGV OK**: 2S training and controls (seeds 3, 4), J5' x3 (both
scenarios), J6(f) and J6(d)-v4 training and controls, the J7' plans (6 and 7 kinds), the slice
plans (5 kinds), the needle controls on avg (3 and 5 seeds), avg-np, **ens3 and ens5**, both
`ckpt_average` forms. `calib_fit_row.py` (with `--bin` pointed at a stand-in file) and
`export_letter_parity.py` parse and stop at the first box path. Negative checks, all refused as
expected:
- the J7' plan + `--needle-control 1024,2048,4096`: "--needle-control ... needs --score-checkpoint
  and --needle" (`real_ft_run.py:8486`; `_check_score_plan_flags` refuses it too, `:6660`);
- J6(f) without `--approved-by`: "rule 4: ... $20.61 at the cap -- NEEDS A HUMAN YES";
- J5' keeping F's `--checkpoint-dir/--checkpoint-every`: "--shuffled-label saves and resumes no
  weights";
- an ens3 control with 2 ft row ids for 3 checkpoints: refused.

Re-run on the re-wired dry runs: the same 31 ARGV OK, the same 4 refusals and the same 18
parse-then-box-path stops (only the recorded order differs). Items 8 and 9 add no
`real_ft_run.py` call of this branch's; their scripts' own argv is the no-mask lane's and is not
checked here.

**The open question:** `--needle-control` is **not** accepted on a `--score-plan` (any kind, ens3
included). It **is** accepted on ens3 as `--score-checkpoint s0.json s1.json s2.json --ft-row-id
a b c --seeds 0 1 2 --needle --needle-control 1024,2048,4096` at `3460afc`, and the code path
supports it (`run_needle_control` -> `_checkpoint_step` builds a `TowerEnsemble`). So item 3 runs
the ens control in that form; it is not dropped. **Its GPU run is not verified** (no ensemble
needle control has ever run).

The dry-run harness is throwaway (session scratchpad: `pfq_sim/` incl. `run2.sh` and
`order_check.py`, `pfq_argv_check.py`,
`pfq_argv_drive.py`, `pfq_mutate.sh`, `pfq_delta_cosine_parity.py`); nothing of it ships.

## Not verified

- Anything on the box: the scripts have not run there; the binary has not run there (only in an
  arm64 container); lane HEADs, `f.ckpt-skip` = 6, p3fused at `f6a0928`, the slice set and corpus
  in lane10, `/home/ubuntu/bin/qd-export` and `qd-calib-fit` are the brief's and HANDOFFs' claims.
- F's real rows: none exist yet. The metric keys the checker reads were confirmed in `a502670`'s
  and `0062bde`'s code (`needle_hunk_recall.depth.*`, `ood_abstain.<category>`, `ft_run_row_id`,
  tag `epoch-score-val`; `avg-np-score-val`, `avg-np-needle-length-control`, `ft_run_row_ids`,
  `score_dtype`) and on J4's rows, not on F's.
- GPU timings: every cap is the pre-registered one; whether the five-seed J7' plan fits 4 h and
  the five-seed slice fits 6 h is inferred, not measured.
- The F logs' `ft row <uuid>` line is `box_q_f.sh`'s own extraction (tail -1); the checker then
  pins every id to F's ledger, so a wrong id refuses rather than scores.
- ens control and ens5 decode cost on the GH200; the qd-export of a five-seed average.
- Items 8/9's box side: that `/home/ubuntu/perf/perf_nomask_p2.sh` and `perf_tierb_outcome.sh`
  are main `669550d`'s, that `perf/overlay-nomask` is built (`perf_nomask_p2.sh --build`) with no
  `nomask-p2-T1/T2.jsonl` left over, that `perf/p3nomask` is built, and that the box's
  `perf_tierb_fused.sh` resolves to `perf_tierb_outcome.sh fused` -- the coordinator's word, not
  checked. A refusal there is logged and the chain moves on.

## What is open (gap ids)

- `GAP-POST-F-QUEUE-NAVIGATION-2026-10-01` (new): GitPulse returned ok=false on every facet
  (REPOSITORY_TRUST_REQUIRED), DevMap has no store in this worktree (answers came from main's
  index, generation 2959), and no ListAgents tool was exposed.
- For Fable: the (b)/(c) decimal-versus-row ambiguity above. Until ruled, a tie refuses (avg-np
  NOT RUN) unless another clause fails outright.
- For the lead: whether the human's $20 yes covers 2S, J5', J6(f), J6(d)-v4.
- For the lead: `touch /home/ubuntu/queue/nomask-p2-ruled` once Fable's amended P2 rule is on the
  box, before item 9's turn (after J5'); until then item 9 is NOT RUN.
- Unchanged: `GAP-QUICK-FLAG-DOES-NOT-ENCODE-THE-SEED-FLOOR` (J6(f) and J6(d)-v4 are one seed but
  their rows will say quick=False, as J6(b)'s did); `GAP-ENSEMBLE-ROW-HAS-NO-PROMOTION-KIND-2026-10-01`.

## Deploy (the lead runs this; nothing here was run against the box)

**0. On the box, before `f.done`: stop the old waiters that would race this chain.** Look first:

    ls /home/ubuntu/queue
    pgrep -af 'box_q_(tierb|j6d|j6d2|v31|avgnp|avgnp2)\.sh'

`box_q_tierb.sh` wakes on `f.done` and its run script flocks `gpu.lock` against item 1;
`box_q_j6d2.sh` (the dropped v3 J6(d)) would wait on these `.queued` markers and then run;
`box_q_v31.sh` is dropped; `box_q_avgnp2.sh` was replaced by `avgnp3`. **Do not touch
`box_q_f.sh`, `box_q_j4ens3.sh` or `box_q_avgnp3.sh`.** If none of the listed ones has a
`.started` marker without `.done` (i.e. none is running GPU work):

    pkill -KILL -f 'box_q_(tierb|j6d|j6d2|v31|avgnp|avgnp2)\.sh'

(`-KILL` so their EXIT traps do not touch `tierb.done` / `j6d.done`; this chain uses other names
either way.) Then check that none of this chain's markers is left from anything earlier -- this
must print nothing (stop and ask if it does):

    ls /home/ubuntu/queue | grep -E '^(j7pcpu|j7pavg|s34|j7p|nomaskp2|fslice|j5p|j6f|nomask|tierb2|j6dv4)\.'

**1. On the Mac: copy the binary, the scripts and delta_cosine.**

    KEY=~/.ssh/bharath_m5_macbook_pro.pem; BOX=ubuntu@192.222.51.246
    WT=/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-a0240fba0d690aeda
    ssh -i $KEY $BOX 'mkdir -p /home/ubuntu/post-f /home/ubuntu/bin'
    scp -i $KEY /Users/bharath/qd-campaign/target-aarch64-linux-postf/aarch64-unknown-linux-gnu/release/qd-post-f-rules $BOX:/home/ubuntu/bin/qd-post-f-rules
    scp -i $KEY $WT/campaign/post-f-queue/*.sh $WT/AUDIT/j7-avg-ood-diag-2026-10-01/delta_cosine.py $BOX:/home/ubuntu/post-f/

**2. On the box: check the pins and smoke the binary on J4's rows.**

    chmod +x /home/ubuntu/bin/qd-post-f-rules
    sha256sum /home/ubuntu/bin/qd-post-f-rules /home/ubuntu/post-f/delta_cosine.py
    #   330e681672897a5369a04d5d1fdafdc8b9071e31e44b8d866927c90daa43c98e  qd-post-f-rules
    #   f4e373a94c87c54759a4a4e382de1349010e18c3d0b646a4c9f376b8dca34961  delta_cosine.py
    cat /home/ubuntu/queue/f.ckpt-skip                                     # 6
    for d in qd-lane8 qd-lane10 perf/p3fused; do (cd /home/ubuntu/$d && echo "$d $(git rev-parse --short=7 HEAD) dirty=[$(git status --porcelain | head -1)]"); done
    #   qd-lane8 a502670, qd-lane10 3460afc, perf/p3fused f6a0928, all dirty=[]
    /home/ubuntu/bin/qd-post-f-rules seeds34 --f-ledger /home/ubuntu/ledger/gh200-p4-v3-2026-10-01.jsonl \
      --ft-row 0=80b19a41-ad0a-4af7-b1d9-2de675d5199b --ft-row 1=9b108fe2-3cf1-41b7-bef0-7dc0dbb7fc45 \
      --ft-row 2=8b600511-887e-4ebc-97d0-7794127c8d0b --out /home/ubuntu/logs/post-f-smoke-j4-seeds34.json 2>/dev/null
    #   fires
    ls /home/ubuntu/perf/perf_nomask_p2.sh /home/ubuntu/perf/perf_tierb_outcome.sh /home/ubuntu/perf/perf_tierb_fused.sh
    bash /home/ubuntu/perf/perf_tierb_outcome.sh nomask --print    # item 9's commands; runs nothing
    ls /home/ubuntu/queue/nomask-p2-ruled                          # absent until Fable rules

**3. On the box: launch the ten waiters together** (before `f.done`; item 10 skips only an item
with no `.queued`):

    for s in j7p_cpu s34 j7p nomaskp2 fslice j5p j6f nomask tierb2 j6dv4; do
      nohup bash /home/ubuntu/post-f/box_q_$s.sh > /home/ubuntu/logs/q-$s.log 2>&1 &
    done
    sleep 5; ls /home/ubuntu/queue | grep -E '^(j7pcpu|s34|j7p|nomaskp2|fslice|j5p|j6f|nomask|tierb2|j6dv4)\.queued$'
    #   ten lines

**4. When Fable's amended P2 rule is on the box** (any time before J5' ends):
`touch /home/ubuntu/queue/nomask-p2-ruled`. Without it item 9 logs NOT RUN and the chain goes on.

**Watch:** `grep -h '^=== ' /home/ubuntu/logs/q-{j7p_cpu,s34,j7p,nomaskp2,fslice,j5p,j6f,nomask,tierb2,j6dv4}.log`
and `ls /home/ubuntu/ledger/post-f-decisions-2026-10-01/`.

## First command for the next lane

After item 2's rows land (`avgnp.done`), read rule (i)'s decision and copy the decisions home:

    scp -r -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246:/home/ubuntu/ledger/post-f-decisions-2026-10-01 /Users/bharath/qd-campaign/
