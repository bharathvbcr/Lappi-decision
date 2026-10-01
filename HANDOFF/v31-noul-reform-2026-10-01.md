# v3.1: the noul rows re-formed (defect-noul-v2) and the v3.1 shard set (2026-10-01)

Lane `v31-noul-reform`, branch `v31-noul-reform` off main `7fc3af7`, on Fable's F2/F3 of
2026-10-01 (`HANDOFF/gh200-phase4-2026-10-01.md` on main, "J4 complete"). Not merged, not
pushed. Every claim is labelled **verified** (run here, output cited), **inferred** or
**unverified**.

## Pre-registered (recorded before any v3.1 run; also in the corpus manifest)

`campaign/v31-noul-preregistered.json`, copied verbatim into
`data/pool/defect-noul-v2/manifest.json` → `preregistered`:

- **Bar:** on v3.1, each OOD category (prose, scrambled, unseen-language) **≥ 30/60 on every
  seed** (0, 1, 2).
- **Fable's data-design bound, not a gate:** `code.defect_class` in-distribution abstention
  **≤ 2% on every seed** (J4 on v1: 33 / 39 / 21 of 2,304 = 1.43% / 1.69% / 0.91%).
- The `ood_abstain` gate itself (Wilson lower ≥ 0.90 OOD, Wilson upper ≤ 0.05 in-distribution)
  is unchanged and read-only (rule 2). The bars are design checks beside it.

**The risk, named:** a noul set designed to span the suite's three forms passes the suite's
forms. If v3.1 clears 30/60, that shows the model abstains on *the suite's* prose, scrambled
and unseen-language shapes — not that it abstains on whatever the product will meet. Whether
those three categories ARE the product's OOD definition is a suite question for the human
(rule 2); this lane changed training data only.

## Free readout first: J4's in-distribution abstentions by gold class (verified)

From `/Users/bharath/qd-campaign/p4-v3-2026-10-01/verdicts-s{0,1,2}.jsonl`, `code.defect_class`
choice rows, rule as `tools/real_ft_run.py::choice_rule_abstentions` (abstain when either pass
tops `noul`, or the permuted pass's winner maps back to another option). Gold class from
`gold_row` over `DEFECT_CLASSES = (stub, logic, cosmetic, clean)`, noul row 4.

| Seed (eval row) | Abstained | stub | logic | cosmetic | clean | Via `noul` top |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (`2f5fe57a`) | 33/2,304 (1.43%) | 7/533 | 8/638 | 10/585 | 8/548 | 0 |
| 1 (`f3612f73`) | 39/2,304 (1.69%) | 5/533 | 13/638 | 12/585 | 9/548 | 0 |
| 2 (`02cf5ff4`) | 21/2,304 (0.91%) | 2/533 | 8/638 | 6/585 | 5/548 | 0 |

- The totals match the phase-4 handoff's 33 / 39 / 21 (verified).
- **`clean` does not dominate:** it is 24% / 23% / 24% of the abstentions against 23.8% of the
  rows — proportional.
- **No in-distribution row abstained through `noul`:** all 93 abstentions are
  permutation disagreements among the four real classes. The model never answers `noul` on a
  real diff, so the clean/noul boundary was not where v1 leaked. Python is 20/1,885, 32/1,885,
  10/1,885; rust 6/116, 4/116, 8/116.
- **What was done about the clean/noul boundary anyway:** the new scrambled rows are real
  corpus diffs with every token kept (the lines form), which sit closer to in-distribution than
  anything in v1. So each must move at least half its body positions, and its word 5-shingle
  Jaccard against its own original must be < 0.5 (dedupe links at 0.8). The scrambled rows
  are drawn from every class of the corpus, not from `clean` alone, so `noul` is not taught as
  "a clean diff, reordered" (inferred design reasoning).

## What changed (commits on `v31-noul-reform`)

| Commit | Change |
| --- | --- |
| `f612ada` | `qd-noul-rows --forms v2` (`corpus.rs`, `questions.rs`, `main.rs`; `prose.rs` / `scramble.rs` expose `path_for` / `wrap` / `shuffle_tokens`), Rust tests, `campaign/v31-noul-preregistered.json`, `data/pool/defect-noul-v2/manifest.json`. **This is the pinned `--rev`**: the data-producing commit |
| the commit holding this file | Python tests, `ledger/mac-phase4-v31-shards-2026-10-01.jsonl`, `GAP-V31-GITPULSE-UNTRUSTED-AND-NO-LISTAGENTS`, this handoff. Nothing the pipeline reads |

Files touched (no `python/qd_data`, no `crates/qd-mutate/src/main.rs`, no pipeline flags, none
of the brief's other editors' files, none of the linear-control lane's files):
- `crates/qd-mutate/src/noul_rows/{main,corpus,questions,prose,scramble}.rs`
- `crates/qd-mutate/tests/noul_rows.rs`
- `python/tests/test_defect_noul.py`
- `campaign/v31-noul-preregistered.json`
- `data/pool/defect-noul-v2/manifest.json`
- `ledger/mac-phase4-v31-shards-2026-10-01.jsonl`
- `gaps.jsonl` (one appended line)
- this file

### The generator: `qd-noul-rows generate --forms v2 --corpus <dir> [--preregistered <json>]`

`--forms v1` is the default and is R2's output byte for byte. v2 keeps 834 rows per source:

- **prose:** 417 v1 paragraphs (the same draw, the first 417 of v1's selection order) and 417
  short-question rows (`crates/qd-mutate/src/noul_rows/questions.rs`).
  - Each question row is 1–3 whole SQuAD v2 **train** questions of one paragraph of an
    allowlisted (train-split, trained-family) title. The count is drawn from {1, 1, 2, 3}, so
    about half are one question.
  - Rows are wrapped at 30–70 characters into a one-hunk diff of at least two lines, with at
    least 8 words. They use v1's path set, never `notes.txt`.
  - A question whose text appeared before is skipped, so no two rows share one.
- **scrambled:** 417 `lines` + 417 `tokens` rows from corpus v3's own diffs
  (`crates/qd-mutate/src/noul_rows/corpus.rs`).
  - Only rows whose `pool_id` the allowlist admits are used (train-split repo, admitted
    licence). That is 45,411 of 49,953; 4,542 are excluded.
  - `lines`: every hunk's body lines are permuted, and a multi-hunk diff's hunk order is
    permuted too. Each header stays on top of its own body with consistent counts. No token
    changes.
  - `tokens`: the same, plus tokens shuffled within each body line (markers and indent kept).
  - One row per repo across both forms; per-language quotas of 104/105.
  - The corpus is refused unless its examples hash to its manifest and its manifest names the
    allowlist's pool.
- **unseen-language:** v1's templates, unchanged.
- The rows carry `noul_form`. The manifest carries `forms`, `totals.by_form`, `corpus` and
  `preregistered`. All of these are absent in v1, which is what keeps v1's bytes.

**Interpretation to check (inferred):** F2 says "half with hunk order shuffled, half with token
order shuffled within lines". This lane read it as: both halves line-shuffled within hunks
(and hunk order shuffled where there are several), with token shuffling added in the second
half. Headers stay at the top of their hunks because the loader refuses a row that does not
start `@@ -` (`qd_data.defect_class._noul_violation`, `diff_is_not_a_hunk`), and because a
headerless row would make "no header → noul" a shape cue. The suite's scrambled cases do
shuffle header lines into the body. That difference is the residual form gap on scrambled.

### Tests (all verified, outputs in this session)

- **Rust (`crates/qd-mutate/tests/noul_rows.rs`):** the fixture world gained real questions and
  a v3-shaped corpus. Five new tests:
  - `v1_output_is_pinned_byte_for_byte_and_is_the_default`: examples sha256 `98853a4e…`, manifest
    `5b257379…`, both taken from the **7fc3af7 binary's** output on the same world;
  - `v2_fills_every_form_exactly_and_is_deterministic`;
  - `v2_scrambled_rows_are_shuffles_of_their_own_corpus_diff`;
  - `v2_question_rows_are_short_questions_of_one_allowlisted_paragraph`;
  - `v2_refuses_a_corpus_it_cannot_trust`.
- **Pre-fix (run against the unmodified 7fc3af7 generator):** all four v2 tests FAILED, and the
  pin test FAILED on `--forms v1` being unknown. Its shas were read from that binary's output.
- **Post-fix:**
  - `cargo test -p qd-mutate --bin qd-noul-rows --test noul_rows`: 23 + 8 passed;
  - `cargo clippy -p qd-mutate --bin qd-noul-rows --tests`: clean;
  - full `cargo test -p qd-mutate`: 214 passed, 0 failed across 10 test binaries.
  - The unit tests cover each form's shape, every skip reason, the Jaccard measure, question
    grouping and dedupe, and the per-title cap.
- **Real v1 reproduced:** the new binary at default flags regenerates `defect-noul-v1` with
  examples sha256 `bb392e41…` and a byte-identical `manifest.json` (`cmp`). This is also a
  test, `test_the_real_v1_regenerates_byte_for_byte_at_the_default_forms`.
- **Python (`python/tests/test_defect_noul.py`, real data):**
  - the two real-corpus tests are parametrized over v1 and v2. For v2: the loader admits all
    2,502 rows in thirds, dedupe keeps all, the split puts all in train, and max 8-gram
    containment is 0.0241 against the suite superset (5 seeds) and 0.0106 against the gate's
    exact suite;
  - new: `test_v2_fills_each_form_exactly_and_records_the_preregistered_bars` (each scrambled
    row is a permutation of a corpus diff of its own pool file);
  - new: `test_v2_question_rows_are_disjoint_from_the_general_val_split`, against `ft_splits`'
    val split at `--general-max-rows 200000`. Zero 8-gram hits both ways and no verbatim
    containment;
  - 7 passed in two runs (4 in 27.8 s, 3 in 67.5 s). Full file: 31 passed, 1 skipped in 211.7 s. The skip is the shard-set relabel test, which needs `commitpackft-corpus-v2` rows that are not linked into this worktree. Ruff clean.

## defect-noul-v2 (verified)

`data/pool/defect-noul-v2/manifest.json` (committed; `examples.jsonl` is git-ignored like
v1's):

- examples sha256 `9bfa93046c9918468da4f8d1029e55839e15647ff652b72a508a70bb04a9877d`, 2,502
  rows; manifest file sha256 `0b33103f…`;
- by source: prose 834, scrambled 834, unseen-language 834;
- by form: paragraph 417, question 417, lines 417, tokens 417, template 834;
- units: prose 295 titles, scrambled 834 repos, unseen-language 35 templates;
- scrambled by language: go 210, python 208, rust 208, typescript 208. 219 of the 834 are
  multi-hunk;
- sources:
  - SQuAD v2 train `6c7bcda5…` (cc-by-sa-4.0);
  - commitpackft pool v2 `2cb31231…`;
  - corpus v3 examples `610bb1f0…`, licences per row through the download join;
  - templates, catalogue `39be8b0c…` (apache-2.0);
- allowlist: v1's, `ee04e0ed…`, unchanged.

## The v3.1 shard set

Ledger row **`fa372ede-e053-4e14-8b7d-68187a07703b`** in
`ledger/mac-phase4-v31-shards-2026-10-01.jsonl` (copied byte for byte from
`/Users/bharath/qd-campaign/ledger-mac-phase4-v31-2026-10-01.jsonl`, sha256 `d2e37aaf…`).
Output: `/Users/bharath/qd-campaign/phase4-v31-2026-10-01`. All verified from the row and the
shard headers.

The command mirrors v3's (`89b619d9`). Its flags are from R1's handoff
(`HANDOFF/r1-multi-hunk-2026-09-30.md`) and agree with v3's recorded recipe. Only `--out`,
`--defect-noul`, `--rev` and `--ledger` changed:

    QD_PREP_BIN=<worktree>/target/release/qd-prep /Users/bharath/.venvs/ml/bin/python -u tools/real_tokenizer_pipeline.py \
      --out /Users/bharath/qd-campaign/phase4-v31-2026-10-01 --rev f612ada68be900d7274d53928c3c080e64e20bfb \
      --val-shards --replay-shards --vocab full \
      --defect-class data/pool/commitpackft-corpus-v3 --defect-download data/pool/commitpackft \
      --defect-noul data/pool/defect-noul-v2 \
      --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
      --general-max-rows 200000 --memo-limit 0 --no-repo-history \
      --ledger /Users/bharath/qd-campaign/ledger-mac-phase4-v31-2026-10-01.jsonl

**Shard hashes** (method checked first: `ShardHeader.from_json(header.json).shard_hash()` on v3
gives `70c9fb3e…` / `1b3387ad…`, J4's recorded train / val hashes):

- train **`a87ed1e726997342f1cd8cdda06dbfa1bf4d6ae7131445bf54880cce5ae43299`**;
- val `83ae8e6757a88c792fbd565796a21cb29e732fc8497e1f8ab71d491937bed90c`;
- replay `8ebd1915a858d0941910e2ae1826d79a8fda7832ed4e2e6fa2a4443e0a854610`;
- `corpus_rev` `f612ada6…` in all three.

**Counts against v3 (`89b619d9`):**

| Metric | v3.1 | v3 |
| --- | --- | --- |
| noul rows | 2,502 (834 / 834 / 834) | 2,502 |
| classes | clean 8,449, cosmetic 7,246, logic 11,293, stub 22,965 | identical |
| mixture rows built | 316,257 of 316,867 | identical |
| train sequences | **300,709** | 300,692 |
| total tokens / positions | 90,448,000 / 93,154,353 | 90,370,528 / 93,068,235 |
| rows refused whole | 139 of 258,072 | identical |
| span slots refused (BPE line-start collapse) | 4,722 | 4,739 |
| span slots refused (UnencodableGold, NFC) | 3 | 3 |
| val | 16,124 rows, 18,223 sequences | identical |
| replay | 3,568 | identical |
| contradictory prompts dropped | 33 groups, 66 rows | identical |
| max length / buckets | 1,105 | 1,105 |
| `padding_waste` | 2.91% | 2.90% |

- **The +17 sequences are all noul span slots (verified from `sequence_index.json`).** Non-noul
  sequences are 295,832 in both sets. The noul span sequences rose from 2,358 to 2,375, and noul
  span-slot exclusions fell from 144 to 127. The re-formed rows' line starts collapse under BPE
  slightly less often.
- **No in-distribution row moved:** not a dedupe drop, not a consistency drop. J4's `width`
  (1,105) still holds.
- **Wall time: 1,659.6 s**, against v3's 749 s. Another lane's `qd-prep linfit` (worktree
  `agent-a55f38d6…`) ran at about 600% CPU and 16 GiB on the same Mac throughout (observed with
  `ps`).
- **`code_commit` is `f612ada…-dirty`** (as v3's was `b381a03…-dirty`). The build itself ran
  from a clean tree at f612ada. This lane then edited `python/tests/test_defect_noul.py`,
  `gaps.jsonl` and this handoff while the build ran.
  - None is a pipeline input. `code_that_ran` (`2fb5c98f…`) hashes the modules that ran and
    lists no test file.
  - `git diff --stat f612ada..HEAD` shows only those files plus this ledger file.
- `quick: true` and `run_kind: smoke`, as v3's build row: one seed, `--no-repo-history`.
  Protocol: `data_snapshot_hash` `fee719c1…`, `recipe_hash` `9f5a5972…`.

## What the box needs

Training reads shards, and `real_ft_run` rebuilds the FT rows to pair labels, which reads the
noul dir. Nothing the box runs changed in Rust: the loader is unchanged Python, and the
generator does not run on the box. **No cross-build is needed** (verified:
`git diff --stat 7fc3af7..HEAD` touches no box-run Rust).

| What | From (Mac) | To (box) | Size |
| --- | --- | --- | --- |
| the v3.1 out dir (shards + `data/`, as v3's was shipped) | `/Users/bharath/qd-campaign/phase4-v31-2026-10-01/` | `/home/ubuntu/phase4-v31-2026-10-01/` | 1.0 G total: `shards/` 837 M (train 404 M, train-no-decode 404 M, val 24 M, replay 5.1 M), `data/` 161 M |
| noul-v2 rows | `/Users/bharath/qd-campaign/phase4-v31-2026-10-01/defect-noul-v2/examples.jsonl` (sha256 `9bfa9304…`) | `/home/ubuntu/qwen-decision/data/pool/defect-noul-v2/examples.jsonl`, symlinked into the lane | 2.2 M |
| noul-v2 manifest | same dir, `manifest.json` (sha256 `0b33103f…`; also committed) | `/home/ubuntu/qwen-decision/data/pool/defect-noul-v2/manifest.json` | 5 K |
| the branch | `v31-noul-reform` (contains `f612ada`) | the training lane's checkout | — |

- `train-no-decode` is the pipeline's stage-8 comparison copy. v3's out dir had it too, so ship
  it or exclude it the way v3's shipment did (unverified which).
- The noul dir is copied out of the worktree into the campaign dir, because the worktree's
  `examples.jsonl` is git-ignored and goes when the worktree does.

## The box training command (J4's invocation with the new `--out` / `--defect-noul` / `--rev`)

J4's literal argv lives in `/home/ubuntu/box_p4_j4.sh` on the box. It is **not** in
`HANDOFF/gh200-phase4-2026-10-01.md`: `git grep` over main's `HANDOFF/` and `docs/` finds only
the script's name, so do not look there again.
Its fixed flags are known from `HANDOFF/j7-avg-scoring-2026-10-01.md`'s check loop and J4's
rows (verified there). Its interpreter, `--real-backbone` and `--tokenizer-json` are box paths
this lane could not read (unverified). So the command is J4's script with exactly three
substitutions plus the ledger and checkpoint names, and each substitution is checked:

    # 0. The lane's checkout must contain f612ada: real_ft_run resolves --rev with git
    #    (tools/repo_git.py::resolve_rev). This lane never pushes, so the parent ships the
    #    branch (push or git bundle) first.
    cd /home/ubuntu/qd-lane3
    git fetch <where the parent shipped v31-noul-reform> && git checkout --detach <its head>
    git merge-base --is-ancestor f612ada68be900d7274d53928c3c080e64e20bfb HEAD || { echo "lane lacks f612ada"; exit 1; }
    cmp data/pool/defect-noul-v2/manifest.json /home/ubuntu/qwen-decision/data/pool/defect-noul-v2/manifest.json || { echo "shipped noul manifest differs from the committed one"; exit 1; }
    test -e data/pool/defect-noul-v2/examples.jsonl || ln -s /home/ubuntu/qwen-decision/data/pool/defect-noul-v2/examples.jsonl data/pool/defect-noul-v2/examples.jsonl
    test "$(sha256sum data/pool/defect-noul-v2/examples.jsonl | cut -d' ' -f1)" = 9bfa93046c9918468da4f8d1029e55839e15647ff652b72a508a70bb04a9877d || { echo "noul-v2 examples sha"; exit 1; }
    export QD_PREP_BIN=/home/ubuntu/bin/qd-prep

    # 1. J4's script, three substitutions plus its ledger / checkpoint / log names.
    J4=/home/ubuntu/box_p4_j4.sh; V31=/home/ubuntu/box_p4_v31.sh
    sed -e 's#/home/ubuntu/phase4-v3-2026-10-01#/home/ubuntu/phase4-v31-2026-10-01#g' \
        -e 's#data/pool/defect-noul-v1#data/pool/defect-noul-v2#g' \
        -e 's#b381a03cb12e816c380915b1983f4e13cd1c843c#f612ada68be900d7274d53928c3c080e64e20bfb#g' \
        -e 's#gh200-p4-v3-2026-10-01#gh200-p4-v31-2026-10-01#g' \
        -e 's#/p4-v3/#/p4-v31/#g' -e 's#p4-v3\.log#p4-v31.log#g' -e 's#p4-v3-control#p4-v31-control#g' \
        "$J4" > "$V31"
    for f in "--out /home/ubuntu/phase4-v31-2026-10-01" "--defect-noul data/pool/defect-noul-v2" \
             "--rev f612ada68be900d7274d53928c3c080e64e20bfb" "--seeds 0 1 2" "--no-repo-history" \
             "--defect-class data/pool/commitpackft-corpus-v3" "--defect-download data/pool/commitpackft" \
             "--general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json" \
             "--general-max-rows 200000" "--replay-partition" "gh200-p4-v31-2026-10-01.jsonl"; do
      grep -q -- "$f" "$V31" || { echo "v31 script lacks: $f"; exit 1; }
    done
    for old in defect-noul-v1 b381a03 phase4-v3-2026-10-01 gh200-p4-v3-2026-10-01; do
      ! grep -q -- "$old" "$V31" || { echo "v31 script still names $old"; exit 1; }
    done
    diff "$J4" "$V31"    # read it: nothing but the substitutions above may differ
    bash "$V31" 2>&1 | tee /home/ubuntu/logs/p4-v31.log

If J4's script spells any of these through shell variables, the flag loop fails. Then a human
reads `box_p4_j4.sh` and writes the literal argv with the same three changes. Do not drop the
check. Schedule the script in the box's marker queue (`/home/ubuntu/queue`, `v31.queued`)
the way the parent queued J4; this lane did not see that queue (unverified).

Expected per seed (inferred from J4): 5,768-ish steps (v3.1's train set is v3's with 2,502
noul rows re-formed), the 10,800 s cap per seed, about 5 h of GPU for three seeds at J4's
rate, at $2.29/h about $11.5 total; single GPU, so no rule-4 human yes.

## What is open

- `GAP-V31-GITPULSE-UNTRUSTED-AND-NO-LISTAGENTS` (new): GitPulse refused every facet
  (repository not trusted) and this session has no `ListAgents`. Mitigated by an isolated
  worktree and disjoint files; not a clean scan.
- The interpretation of F2's scrambled halves (above) is Fable's to confirm.
- Rule 2 / the risk above: whether the suite's categories are the product's OOD definition.
- Not run here: any training or scoring of v3.1 (GPU; the box).

## First command for the next lane

The parent ships the shards and the noul rows, then runs the box command above:

    rsync -a /Users/bharath/qd-campaign/phase4-v31-2026-10-01/ ubuntu@192.222.51.246:/home/ubuntu/phase4-v31-2026-10-01/ -e "ssh -i ~/.ssh/bharath_m5_macbook_pro.pem"
    rsync -a /Users/bharath/qd-campaign/phase4-v31-2026-10-01/defect-noul-v2/ ubuntu@192.222.51.246:/home/ubuntu/qwen-decision/data/pool/defect-noul-v2/ -e "ssh -i ~/.ssh/bharath_m5_macbook_pro.pem"

After v3.1's three seeds land, read the bars against the pre-registration. Use OOD by category
per seed (≥ 30/60 each), and `code.defect_class` in-distribution abstention per seed (≤ 2%,
split by gold class with the rule above). Report a miss as a miss.

**If pairing refuses on the box:** first suspect `QD_PREP_BIN`. The box's `qd-prep`
(`c0de659c…`) is not the one this lane built from `7fc3af7`. 7fc3af7 changed only the
ngrams/linfit input bound, not MinHash signing, so a mismatch is unlikely (inferred).
