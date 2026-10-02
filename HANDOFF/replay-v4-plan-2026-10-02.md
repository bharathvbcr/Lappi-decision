# L-replay phase 1: a CLEAN v4 replay slice for J6(a) (2026-10-02)

This is lane L-replay's plan for Fable's idle-queue item 3, `j6a` (lane task 5 in
`AUDIT/idle-gpu-queue-2026-10-02/fable-idle-queue.md:66-73,132`). Its purpose is to close
`GAP-PHASE4-REPLAY-SLICE-OVERLAPS-VAL`.

**Phase 1 only. Nothing was built.** This lane:

- ran no GPU work;
- wrote nothing on the box (every ssh call was a single read-only command);
- did not touch `python/qd_data`, `crates/qd-runtime/src/bin/qd_post_f_rules.rs` (L-room's) or
  `campaign/post-f-queue/box_q_*.sh` (L-waiters').

Every claim below carries one of three labels:

- **verified**: run or read here, with a citation;
- **inferred**: follows from code that was read but not run;
- **unverified**.

Line numbers are at `69a58ae` (main) unless they say `@a502670`.

## Navigation

- **The worktree started at `c65d7da`, not main.** It was fast-forwarded to `69a58ae`
  before anything was read.
- **DevMap:** this worktree has no store, so DevMap answered from the main checkout's store
  (generation 3025, fresh). `devmap_neighbors` came back `walk_incomplete` on every target.
  It also missed `real_ft_run.ft_splits → pipeline.split_off_replay`, which is a
  module-attribute call. Read confirms that call at `tools/real_ft_run.py:3413-3416`.
- **GitPulse:** returned `REPOSITORY_TRUST_REQUIRED` on every facet.
- **ListAgents:** not exposed in this session. `list_sessions` showed no running session in
  this repo.
- All of this is recorded as `GAP-L-REPLAY-NAVIGATION-2026-10-02`. Nothing below is
  graph-confirmed; it comes from Read, rg and git.

## 1. How F's v4 was built and where it lives

- **Build row** (verified): `d96409bd` in `ledger/mac-phase4-v4-shards-2026-10-01.jsonl`.
  - `code_commit` and `--rev` are both `881ab304f15ea13529002391dda8520c2ea47af4`.
  - Built on the Mac in 3,910 s of wall clock. The row is `quick` (one seed).
- **Protocol** (verified): `data_snapshot_hash` `ea3215c4…` (this is the train manifest's own
  hash) and `recipe_hash` `4662ca85…`.
- **Recipe** (verified, from the row):
  - `--no-repo-history`;
  - `--defect-class data/pool/commitpackft-composed-v1`, examples `a63563e5…`;
  - `--defect-download data/pool/commitpackft`;
  - `--defect-noul data/pool/defect-noul-v3b`, examples `b194f14c…`;
  - `--general-record` `a0841f0d…` with `--general-max-rows 200000`;
  - `--max-seq-len 8192`, `--vocab full`, `--val-shards`, `--span-collapse-policy refuse-gold`;
  - **no `--replay-shards`**.
- **No verbatim pipeline argv is recorded anywhere.** The command in §6 is reconstructed from
  the recipe.
  - `--memo-limit 0` is inferred. `memo_limit` is not a recipe key
    (`git show a502670:tools/real_tokenizer_pipeline.py`, around :2741-2830).
  - Build 1's equality gates confirm or refute the reconstruction.
- **On the Mac** (verified): `/Users/bharath/qd-campaign/phase4-v4-2026-10-01/`.

  | Item | Hash | Count |
  | --- | --- | --- |
  | `shards/train` | `8bcf56ad…` | 363,950 sequences |
  | `shards/val` | `ef06ab99eef107dd608424c80117b81b3986ce5ea827d2d4f47f990d960b1bc5` | 18,223 sequences |
  | `data/pool/train.json` | `ea3215c4…` | 289,142 rows |
  | `data/pool/val.json` | `69d45fd04fad01751b3f102f09041c0ba67176e44725dbeb977d550f4b0abbd4` | 16,124 rows |

  - Remap `e7d0890c547b43811f93e50dc0e582ba35eb2e75706ee0ad27186d6a89d502ab`.
  - There is no `train-replay.json` and no `shards/replay`. `data/heldout/` also exists.
- **On the box** (verified, read-only `ls`): `/home/ubuntu/phase4-v4-2026-10-01/` holds the
  same layout and no replay set. F's argv is in `/home/ubuntu/box_q_f.sh` (read).
  - It runs from `/home/ubuntu/qd-lane8` pinned clean at `a502670`.
  - Its `SPLIT` is `--out /home/ubuntu/phase4-v4-2026-10-01 --no-repo-history --rev 881ab304…
    --defect-class data/pool/commitpackft-composed-v1 --defect-download data/pool/commitpackft
    --defect-noul data/pool/defect-noul-v3b --general-record <REC> --general-max-rows 200000`.
  - `post_f_common.sh:43-49` repeats it verbatim.
- **F's rows** (verified, `AUDIT/idle-gpu-queue-2026-10-02/f-ledger-seed0-snapshot-2026-10-02T0113Z.jsonl`):
  - ft `973cd4e3`: `data_snapshot_hash ea3215c4…`, shard `8bcf56ad…`, `code_commit a502670`, no
    `replay_*` recipe key;
  - eval `f4feac15`: `val_shard_hash ef06ab99…`.
- **The v4 corpora on the Mac** (verified by `shasum`):
  - The tracked `data/pool/*/manifest.json` files are in every checkout. The untracked
    examples are not.
  - composed-v1 `a63563e5…` is at
    `/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-a23c0fc7f22b1097c/data/pool/commitpackft-composed-v1/examples.jsonl`.
  - noul-v3b `b194f14c…` is at
    `/Users/bharath/qd-campaign/noul-v3b-2026-10-01/defect-noul-v3b/examples.jsonl`.
  - corpus-v3 `610bb1f0…`, `commitpackft-pool-v2.jsonl` and `commitpackft/{go,python,rust,typescript}.jsonl`
    are in the main checkout's `data/pool/`.
  - Any build lane needs these linked in.

## 2. How `--replay-shards` and `tools/replay_decontam.py` make the slice and attestation

- **The pipeline** (verified, read; identical at `a502670` @a502670 :1457-1480):
  `--replay-shards` calls `split_off_replay` (`tools/real_tokenizer_pipeline.py:1741-1764`,
  run at :2441-2455).
  - It takes the MMLU/CSQA **train** rows and calls
    `qd_data.general.partition_replay(general_train, seed)` with **no** contaminated keys.
  - That function draws a keyed hash of `(seed, identity_key)`: rows with `u < 0.15` become
    replay-only, the rest stay gold (`python/qd_data/general.py:543-607`).
  - The gold train becomes `train − all MMLU/CSQA rows + part.gold_rows`, and those rows are
    appended **at the end** (:1756). So the gold train's row order changes too, not just its
    membership.
  - val and held-out are carried through untouched (:1757-1763).
  - The replay rows are written to `shards/replay` with `write_shards(replay=True)` and the
    manifest `data/pool/train-replay.json` (:2470-2478, :2795-2809).
- **The remap is unchanged under `--vocab full`.** `full_vocab_remap` takes only the vocabulary
  size, tokenizer hash and special ids (:2639-2643). (inferred: read, not run)
- **The decontam tool** (verified, read; unchanged from `881ab30` to `69a58ae`):
  `tools/replay_decontam.py` rebuilds the splits with `real_ft_run.ft_splits`, without the
  partition.
  - It renders every `val` and `heldout` row, keyed `row_id#slot`, and skips unrenderable rows,
    which it counts.
  - It decodes every replay sequence through the remap and tokenizer, then runs
    `qd_train.replay.decontaminate` with 8-gram containment ≥ 0.5.
  - It writes an attestation with:
    - version 1;
    - `replay_shard_hash`;
    - a digest and row count for each target;
    - `hits`;
    - at most 50 `hit_examples` (`replay.py:74,224`);
    - `clean`;
    - the corpus identity (`real_ft_run.replay_corpus_identity`, :8467-8510).
  - Exit status 0 means CLEAN, 1 means CONTAMINATED.
- **What CLEAN requires** (verified, `python/qd_train/replay.py:151-152`, `:236-274`):
  `replay_rows_checked > 0`, and 0 hits on every target.
- **What `real_ft_run` additionally checks** (`_replay_plan`, :8533-8598; identical
  @a502670 :7240):
  - the replay set's sequence-index role is `replay_only`;
  - its `remap_hash` equals the train set's;
  - its `corpus` equals this run's `replay_corpus_identity`;
  - the attestation is for this replay `shard_hash`;
  - it checked both `val` and `heldout`;
  - it has 0 hits.
  - It **recomputes the val digest** itself. It holds held-out only to the corpus identity
    (rule 3).
  - `corpus_facts` requires `--replay-partition` whenever `train-replay.json` exists in
    `--out` (:3590-3601).
  - `pair_labels` refuses rebuilt slots that the writer neither wrote nor excluded (:3770-3777;
    @a502670 :3343).
- **The phase-4 attestation** (verified, read):
  `/Users/bharath/qd-campaign/phase4-fullvocab-2026-09-30/replay-attestation-d43790d.json` is
  CONTAMINATED.
  - Hits: val 185, heldout 0, over 3,568 replay rows.
  - It names 50 hits: 44 against MMLU val rows and 6 against CSQA, with containment 0.50–0.84.
  - **The other 135 hits are not named anywhere** (`GAP-DECONTAM-ATTESTATION-NAMES-50-OF-N-HITS-2026-10-02`).
- **The slice is stable across builds** (verified):
  - Phase-4's and v3's `train-replay.json` both hash to `19a772846da4b759cf6509dca9be642bf8d01ddccb2015e8215f42532946ba84`.
    That is 3,568 rows: knowledge 2,128 and commonsense 1,440.
  - All 3,568 are gold rows in v4's `train.json`, matched by both `row_id` and `identity_key`.
  - v4's 2,682 MMLU/CSQA val rows are identical to phase-4's (`row_id` and `content_hash`).
  - So a v4 `--replay-shards` build should draw the same 3,568 rows and hit the same 185.
    (inferred)

## 3. Options for a CLEAN v4 slice

**No option below changes a val or held-out row.** Every one of them edits train rows only:

- `partition_replay` refuses any row not pinned to train (`general.py:582-585`);
- `split_off_replay` carries val and held-out through untouched.

One alternative raised on 09-29 *would* change val: "a containment check at split time"
(`GAP-DATA-GENERAL-CE-ROWS-OVERLAP-FAMILY-VAL`'s open record). **It moves rows between splits,
it would change val, and it is not proposed here.**

### (a) Exclude the replay-drawn hits only. Recommended.

- **Mechanism:** `partition_replay(…, contaminated_identity_keys = the identity keys of every
  hit replay row)`.
  - Hit rows go to `excluded_rows`.
  - No other row moves, because the keyed draw is per row (`general.py:588-597`).
- **What changes relative to F's v4:**
  - **Gold train:** v4 minus all 3,568 replay-drawn rows. 3,568 − h become replay, and the h
    hits are excluded. h is 185 if phase-4 repeats (inferred).
  - **Gold rows: 285,574** (289,142 − 3,568), with row order changed (§2).
  - **`data_snapshot_hash`:** changes from `ea3215c4…`. The new value is unknown until built.
    The train shard hash changes from `8bcf56ad…`.
  - **Replay rows:** 3,568 − h (3,383 if h = 185). The replay shard hash is new.
  - **Unchanged** (each is a STOP gate): val rows and `69d45fd0…`, val shards `ef06ab99…`,
    held-out manifest hash, and remap `e7d0890c…`.
- **The gold train is identical to a plain `--replay-shards` build's.** `part.gold_rows` does
  not depend on which replay rows are excluded. That is exactly what `real_ft_run`'s own
  `--replay-partition` rebuild produces.
- **So J6(a) runs from `qd-lane8` at `a502670` with no trainer change.** It stays on F's exact
  path, as `box_q_j6f.sh` does. (inferred)
- **Code: two additive changes, neither in `qd_data`.**
  1. A complete hit list. Its canonical owner is `qd_train.replay.decontaminate`, which keeps
     only 50 examples today.
     - Add a field that carries every hit and leaves `to_json` unchanged, so the version-1
       attestation stays byte-compatible with `check_attestation` @a502670.
     - Add `tools/replay_decontam.py --hits-out FILE`. Each hit carries its replay sequence,
       `(row_id, slot)` from the sequence index, `identity_key` from `train-replay.json`, the
       target and the containment.
     - **This touches `python/qd_train`, not `tools/`** (decision point 5).
  2. `tools/real_tokenizer_pipeline.py --replay-exclude FILE`, which requires `--replay-shards`.
     - It threads the keys into `split_off_replay` and from there into `partition_replay`.
     - It refuses any key that is not a replay-drawn row of this build's own partition, so the
       gold train is provably unchanged.
     - It records `replay_exclude_sha256` in the recipe only when given, so default builds hash
       as before.
- **Tests, each failing at the pre-change commit:**
  - decontaminate reports hits beyond 50;
  - the attestation JSON is byte-identical before and after on a fixture;
  - `--hits-out` lists every hit;
  - `--replay-exclude` drops exactly the named keys and leaves the gold train manifest hash
    identical;
  - it refuses a gold-drawn key, and refuses without `--replay-shards`;
  - on a fixture whose plain build is CONTAMINATED, the excluded build attests CLEAN.
- This is Python because it extends the existing Python owners. A Rust port of the pipeline is
  out of scope.

### (b) Draw the slice after decontaminating every MMLU/CSQA training row

This is the lead's 09-29 one rule, applied to both roles.

- **What it does:** every contaminated MMLU/CSQA train row, gold or replay, goes to
  `excluded_rows`.
- **The replay slice is the same as (a)'s.** Exclusion moves no other row, so with the keyed
  draw, "draw after decontamination" yields the same replay rows.
- **The gold train also loses every contaminated gold row.** That count is unmeasured on v4.
  The 983-MMLU figure is from another corpus.
- **`real_ft_run` @a502670 cannot consume it.** Its rebuild would still produce those gold rows,
  and `pair_labels` refuses them as strays (@a502670 :3343).
  - So (b) also needs the exclusion wired into `real_ft_run.ft_splits`, and the exclusion list
    carried in the attestation and recipe so the trainer is held to it.
  - J6(a) then runs from a new lane, off F's path.
- **The comparison is confounded:** J6(a) would differ from F by replay *and* by gold
  decontamination.
- It is the right rule for the *next recipe*, not for this ablation (decision point 6).

### Refused

- **Replay without moving gold rows**, keeping F's train byte-identical.
  - A row that is both gold and replay is the case the partition forbids
    (`general.py:479-541`).
  - `real_ft_run` @a502670 does not cross-check replay row ids against gold. It relies on
    `--replay-partition` (inferred, read of `_replay_plan`).
  - So this would train contaminated by construction. Not proposed.
- **A replay source outside MMLU/CSQA** means new data. That is the human's call, as in
  `GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN`.

## 4. What the replay flags resolve to, and how the attestation reaches the box

### The flags, @a502670 (the commit J6(a) runs at; verified with `git show`)

| Flag | Resolves to |
| --- | --- |
| `--replay-weight` | **No value.** `default=None`, help "weight on prior_kl; no default -- say it" (@a502670 :7818-7819; main :9191-9192). Refused if missing, ≤ 0 or non-finite (main :8257-8266). See `GAP-REPLAY-WEIGHT-HAS-NO-PINNED-VALUE-2026-10-02`. |
| `--replay-every` | 6 (`DEFAULT_REPLAY_EVERY`, @a502670 :308, main :352-354): one replay micro-batch per 6 training ones, the plan's "~15%". |
| direction | Fixed at `base_to_model` (main :432). |
| `--replay-cache` | Required. Built on first use, then key-checked. A resume refuses a missing cache. |
| `--replay-attestation` | Required, must exist (main :8269-8270). |
| `--replay-shards` | Must hold a header. Epoch arm only (main :8260-8264). |

Further constraints:

- **No source pins a weight.**
  - 0.1 appears only as a test fixture (`python/tests/test_real_ft_pieces.py:117,128`).
  - RSI-Jev's `prior_kl` defaults to 0.0, which disables it (`rsijev/fit.py:58` at `8f34a4f`).
    RSI also applies the term per training batch rather than on a separate micro-batch.
- **Replay batch size** (inferred): replay batches use `batch_tokens` equal to the train set's
  widest bucket (7,936 on v4) and are filtered to the run's widest batch (main :8579-8587).
  - The phase-4 replay buckets top out at 1,086 tokens.
  - So the KL term adds about (7,936 / 35,403) / 6 ≈ 3.7% of the training tokens, plus one
    no-grad pass to cache the base's letter logits.
- **What the ledger records:** `replay_shard_hash`, `replay_attestation_sha256`,
  `replay_weight`, `replay_every` and `replay_direction` (main :427-432).
- **An argv refusal on main only:** main refuses `--train-attention-mask` other than `padding`
  together with `--replay-shards` (:8195-8200). a502670 has no such flag.

### Getting the attestation onto the box

- It is a file written on the Mac inside the build's out dir. It is rsynced with the shards to
  `/home/ubuntu/phase4-v4-replay-2026-10-02/replay-attestation.json`.
- L-waiters' `box_q_j6a.sh` should pin its sha256 the way `post_f_common.sh` pins `RULES_SHA256`.
  It should refuse unless the file says `"clean": true`.
  - Otherwise it logs "deferred to the re-plan", as Fable's Q2 requires (`fable-idle-queue.md:92`).
  - `real_ft_run` then re-checks shard hash, val digest and corpus identity at its prelude.

## 5. Cost and where to run

- **Recommendation: Mac CPU for every build and decontam step.**
  - v4 itself was built there.
  - Every corpus is present (§1).
  - It costs F nothing.
- **The box is busy** (verified, 02:33 UTC):
  - F seed 1 is at step 5,895/9,683, at 1.65 s/step, with an ETA of 6,260 s. Seed 2 follows.
  - F's python is pinned at 100% of one core (the feeder). The GPU is at 100% and 72 GB.
  - The machine has 64 cores and a load average of 1.0.
  - A niced build there would very likely not slow F, but nothing needs it.
- **The box gets only a niced CPU prelude**, about 5 min. F's took 249 s
  (`/home/ubuntu/logs/f-prelude.log`).

| Step | Where | Estimate |
| --- | --- | --- |
| build 1, plain `--replay-shards` | Mac, 1 core | ≈ 65 min (v4 took 3,910 s at `881ab30`; a502670 adds Rust LSH, `cd3fd0a`), 8–9 GB+ RSS (09-29 corpus; unmeasured on v4) |
| decontam 1 with `--hits-out` | Mac | 10–30 min (unmeasured; phase-4's wall time was not recorded) |
| code and tests (two additive changes) | Mac | lane time |
| build 2 with `--replay-exclude` | Mac | ≈ 65 min |
| decontam 2, expect CLEAN | Mac | 10–30 min |
| prelude stopping after `_replay_plan` | Mac, then box (niced) | ≈ 5 min each |
| rsync Mac → box | network | ≈ 1.4 GB (train shards 1.2 GB) |

- **Total:** about 3–4 h of Mac CPU, $0, and no GPU.
- **J6(a) itself** (inferred): cap 32,400 s ($20.61).
  - Expected run: F seed 0's 16,024 s, plus ≈ 4% for the KL term, plus about 800 s of scoring
    and needle control. That is ≈ 4.9 h, ≈ $11.
  - Fable estimated ≈ 6 h and $14.

## Decision points for the lead

1. **(a) or (b).**
   - (a) keeps J6(a) on F's exact path and isolates replay.
   - (b) needs trainer changes, a new lane, and confounds replay with gold decontamination.
2. **`--replay-weight`.** No value exists in code, docs or the ported source. Someone must name
   it (`GAP-REPLAY-WEIGHT-HAS-NO-PINNED-VALUE-2026-10-02`). `--replay-every 6` and the direction
   come from the code.
3. **The build commit.**
   - Build on a branch from `a502670` with one additive commit, then merge that commit to main.
   - Main's pipeline writes `span_collapse_policy:refuse-gold` into train and replay headers.
     `a502670`'s reader recomputes `shard_hash` without it and refuses the set
     (`GAP-A502670-READER-REFUSES-HEADERS-WRITTEN-AT-MAIN-2026-10-02`; inferred, not run).
   - So a main-built set cannot be trained from `qd-lane8`.
   - The alternative, running J6(a) from a main lane, moves it off F's path.
4. **J6(a) has no reading rule.**
   - `qd-post-f-rules` has no J6(a) arm, and its `arm_identity` refuses a different
     `data_snapshot_hash` (`qd_post_f_rules.rs:1466-1473`). Every J6(a) build changes that hash.
   - "MMLU's last-option bias" is not a key on F's eval row. It comes from `qd-gate-report`
     over verdict files.
   - The rule needs pre-registering before the row exists: which data delta it accepts,
     `val_shard_hash == ef06ab99…`, and which metric keys it reads
     (`GAP-J6A-HAS-NO-READING-RULE-2026-10-02`).
   - The checker file is L-room's.
   - F's eval row does carry `permutation_consistency.family.{knowledge,commonsense}.multiple_choice`,
     `ood_abstain.in_distribution.family.*` and `ece.family.*` (verified).
5. **Where the full hit list lives.** Its canonical owner is `python/qd_train/replay.py`, not
   `tools/` or Rust as the brief says. The alternative, a second containment implementation in
   `tools/`, would duplicate the owner.
6. **The one rule was never wired in**
   (`GAP-ONE-RULE-EXCLUSION-NOT-WIRED-INTO-ANY-BUILD-2026-10-02`, owner lead).
   - No build applies `partition_replay`'s exclusion.
   - F gold-trains all 23,819 MMLU/CSQA train rows. At least 50 of them, and very likely 185,
     overlap F's own MMLU/CSQA val at ≥ 0.5 containment (verified by composition for the 50).
   - F's general-family val readings are optimistic by an unmeasured amount.
   - J6(a) under (a) gold-trains 185 fewer of those rows than F. That is a small confound in
     the general-family direction J6(a) is read on.
7. **First real replay run** (`GAP-REPLAY-PATH-NEVER-RUN-ON-THE-REAL-TOWER-2026-10-02`).
   - Replay has never trained the 2B through `main`.
   - Either run a capped box-GPU smoke first (`--wall-clock-cap-s` about 1,200, `quick`, under
     $1; a502670 has no `--max-steps`), or accept first-run risk inside J6(a)'s cap.
8. **Do not ship `data/heldout/heldout.json` to the box** (recommended). Nothing J6(a) runs
   reads it (inferred, as for F in `HANDOFF/longctx-compose-2026-10-01.md`).

## 6. Exact commands (phase 2, after the ruling; none was run)

All of these run on the Mac, from the build lane's root, because the data paths are relative.

### Set up the lane

```
git worktree add /Users/bharath/Code/research/Lappi-decision/.claude/worktrees/l-replay-build -b l-replay-build a502670
```

Bring in the untracked corpora (paths in §1):

- **Copy** the composed-v1 `examples.jsonl` to a stable path under `/Users/bharath/qd-campaign/`,
  then check that its sha256 is `a63563e5…`.
- Do not symlink it into another agent's worktree (`agent-a23c0fc7f22b1097c`). A worktree
  cleanup during a 65-minute build would break the link mid-run.
- noul-v3b already lives under `qd-campaign`. The rest are in the main checkout.
- Link the copies into the lane's `data/pool/`.

Then build `qd-prep` at the lane commit:

```
cargo build --release -p qd-prep --target-dir /Users/bharath/qd-campaign/target-replay
```

Set:

```
QD_PREP_BIN=/Users/bharath/qd-campaign/target-replay/release/qd-prep
PY=/Users/bharath/.venvs/ml/bin/python
REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
TOK=/Users/bharath/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/tokenizer.json
V4=(--no-repo-history --rev 881ab304f15ea13529002391dda8520c2ea47af4
    --defect-class data/pool/commitpackft-composed-v1 --defect-download data/pool/commitpackft
    --defect-noul data/pool/defect-noul-v3b --general-record $REC --general-max-rows 200000)
LEDGER=/Users/bharath/qd-campaign/ledger-mac-phase4-v4-replay-2026-10-02.jsonl
```

### Build 1: a plain `--replay-shards` build at `a502670`

Build 1 needs none of the new code, so it can start the moment the lead rules:

```
$PY -u tools/real_tokenizer_pipeline.py --out /Users/bharath/qd-campaign/phase4-v4-replay-probe-2026-10-02 \
  "${V4[@]}" --val-shards --memo-limit 0 --max-seq-len 8192 --vocab full \
  --span-collapse-policy refuse-gold --replay-shards --ledger $LEDGER
```

**STOP gates on build 1.** Any mismatch stops the lane and gets reported. No code is written
past it.

- `shards/val` `shard_hash` = `ef06ab99eef107dd608424c80117b81b3986ce5ea827d2d4f47f990d960b1bc5`.
- `data/pool/val.json` `data_snapshot_hash` = `69d45fd04fad01751b3f102f09041c0ba67176e44725dbeb977d550f4b0abbd4`.
- `data/heldout/heldout.json` `data_snapshot_hash` equals v4's. Read only that field, from both
  files.
- Train header `remap_hash` = `e7d0890c547b43811f93e50dc0e582ba35eb2e75706ee0ad27186d6a89d502ab`.
- `data/pool/train-replay.json` = `19a772846da4b759cf6509dca9be642bf8d01ddccb2015e8215f42532946ba84`,
  with 3,568 rows.
- `train.json` entries = v4's entries minus exactly those 3,568 `row_id`s, which is 285,574
  rows.
- Every recipe key shared with `d96409bd` is equal. The new key is `replay_shards: true`.

### Decontam 1 (needs change 1). Expected CONTAMINATED, exit 1.

```
$PY tools/replay_decontam.py --out /Users/bharath/qd-campaign/phase4-v4-replay-probe-2026-10-02 \
  --replay-shards /Users/bharath/qd-campaign/phase4-v4-replay-probe-2026-10-02/shards/replay \
  --tokenizer-json $TOK "${V4[@]}" \
  --attestation-out /Users/bharath/qd-campaign/phase4-v4-replay-probe-2026-10-02/replay-attestation.json \
  --hits-out /Users/bharath/qd-campaign/phase4-v4-replay-probe-2026-10-02/replay-hits.json
```

The decontam's `--rev` default is `0632f693…`, so `--rev` must come from `V4`. This command
passes it.

### Build 2 (needs change 2)

```
$PY -u tools/real_tokenizer_pipeline.py --out /Users/bharath/qd-campaign/phase4-v4-replay-2026-10-02 \
  "${V4[@]}" --val-shards --memo-limit 0 --max-seq-len 8192 --vocab full \
  --span-collapse-policy refuse-gold --replay-shards \
  --replay-exclude /Users/bharath/qd-campaign/phase4-v4-replay-probe-2026-10-02/replay-hits.json \
  --ledger $LEDGER
```

Gates: everything from build 1, plus:

- the train `shard_hash` equals build 1's;
- the replay row count is 3,568 − h, and the partition metric counts h as excluded.

### Decontam 2. Must print CLEAN and exit 0.

The same command against `phase4-v4-replay-2026-10-02`, with `--attestation-out
…/phase4-v4-replay-2026-10-02/replay-attestation.json` and no `--hits-out`. It must show:

- `hits {val: 0, heldout: 0}`;
- `replay_rows_checked == replay_rows_total`;
- unrenderable targets counted.

### Prelude, first on the Mac, then on the box

- It is `/home/ubuntu/scratch/f_prelude_box.py`'s form, except that it stops at the first
  statement after `_replay_plan`. @a502670 that statement is `Ledger(args.ledger)`, so it
  patches `real_ft_run.Ledger`, not `_resolve_batch_tokens`.
- It runs on the J6(a) data argv below, with `--epoch --no-memorise --score-val --batch-tokens 35403`.
- It must print the `replay: N batches … attestation <sha16>` line. That proves four things:
  1. the train set's header reads at `a502670`;
  2. labels pair;
  3. the attestation is accepted, including the val digest recomputed by `_replay_plan`, which
     raises on an unrenderable val row where the decontam tool skips it;
  4. replay batches build.
- It also runs on the needle-control argv (`--score-checkpoint` form, `--replay-partition`, no
  replay flags). Whether that form needs `--replay-partition` is unverified.

### Ship to the box

This writes on the box, so it is a later, approved step:

```
rsync -a --exclude shards/train-no-decode --exclude data/heldout /Users/bharath/qd-campaign/phase4-v4-replay-2026-10-02/ ubuntu@192.222.51.246:/home/ubuntu/phase4-v4-replay-2026-10-02/ -e "ssh -i ~/.ssh/bharath_m5_macbook_pro.pem"
```

Then compare `shasum -a 256` on the Mac with `sha256sum` on the box, file by file.

### J6(a) argv, for L-waiters' `box_q_j6a.sh`

This is `box_q_j6f.sh`'s form. L-waiters write the file; this lane does not.

```
lane "$LANE8" "$LANE8_AT"; f_skip_ok
D=/home/ubuntu/phase4-v4-replay-2026-10-02; OUT=/home/ubuntu/j6a-v4; CKPT=/home/ubuntu/ckpt/j6a-v4
timeout 34200 "$PY" -u tools/real_ft_run.py --out $D --no-repo-history \
  --rev 881ab304f15ea13529002391dda8520c2ea47af4 --defect-class data/pool/commitpackft-composed-v1 \
  --defect-download data/pool/commitpackft --defect-noul data/pool/defect-noul-v3b \
  --general-record "$REC" --general-max-rows 200000 --real-backbone "$BACKBONE" --replay-partition \
  --optimizer master --lr 1e-5 --devices cuda --seeds 0 --epoch --no-memorise --batch-tokens 35403 \
  --lower-layers-n 8 --lower-layers-lr-scale 0.1 --checkpoint-skip-layers "$F_SKIP" \
  --replay-shards $D/shards/replay --replay-attestation $D/replay-attestation.json \
  --replay-cache "$CKPT/replay-prior-cache.npz" --replay-weight <LEAD/FABLE> --replay-every 6 \
  --checkpoint-dir "$CKPT" --checkpoint-every 100000 \
  --score-val --needle --ood --ood-general-record "$REC" \
  --verdicts-out "$OUT/verdicts.jsonl" --suite-verdicts-out "$OUT/suite-verdicts.jsonl" \
  --wall-clock-cap-s 32400 "${COST[@]}" --ledger "$J6V4_LEDGER" \
  --approved-by "$(approved "J6(a) seed 0 (post-F item j6a)")"
```

- The waiter also needs the attestation sha256 pin and the `"clean": true` check from §4.
- The needle control follows as in `box_q_j6f.sh`, with `--out $D --replay-partition` and no
  replay flags.

## What changed (commits)

One commit on this lane's branch. It contains:

- this file;
- seven `gaps.jsonl` records, appended with `qd_train.gaps.append_gap`.

No code changed.

## Open (gap ids)

**New:**

- `GAP-L-REPLAY-NAVIGATION-2026-10-02`
- `GAP-ONE-RULE-EXCLUSION-NOT-WIRED-INTO-ANY-BUILD-2026-10-02` (lead)
- `GAP-REPLAY-WEIGHT-HAS-NO-PINNED-VALUE-2026-10-02` (lead)
- `GAP-A502670-READER-REFUSES-HEADERS-WRITTEN-AT-MAIN-2026-10-02`
- `GAP-J6A-HAS-NO-READING-RULE-2026-10-02` (lead)
- `GAP-REPLAY-PATH-NEVER-RUN-ON-THE-REAL-TOWER-2026-10-02` (lead)
- `GAP-DECONTAM-ATTESTATION-NAMES-50-OF-N-HITS-2026-10-02`

**Still open, and closed only by decontam 2's CLEAN attestation on the box:**

- `GAP-PHASE4-REPLAY-SLICE-OVERLAPS-VAL`

**Context:**

- `GAP-DATA-GENERAL-CE-ROWS-OVERLAP-FAMILY-VAL` (closed 09-29, never wired)
- `GAP-V4-TRAIN-HEADER-OMITS-SPAN-COLLAPSE-POLICY`
- `GAP-PORTED-HELDOUT-FAMILY-CONTENT-OVERLAPS-TRAIN`

## First command for the next lane

After the lead's ruling on decision points 1–3, build 1 needs no new code:

```
git worktree add /Users/bharath/Code/research/Lappi-decision/.claude/worktrees/l-replay-build -b l-replay-build a502670
```

Then link the corpora and run the build 1 command in §6.
