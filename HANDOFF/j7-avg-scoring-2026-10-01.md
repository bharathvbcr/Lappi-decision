# J7 avg scoring: average the seeds from their fp32 masters, score the average (2026-10-01)

Phase 6 piece (i). Branch `j7-avg-scoring` off main `9882bb7`; not merged, not pushed.

## What changed (commits)

`c262ca3`: the code and its tests.

- **`tools/ckpt_average.py --from masters`.** The default (`--from tower`) averages exactly as
  before.
  - Under `masters`, each parameter's fp32 master (`model_state['optimizer']['masters']`) is
    averaged; a tensor with no master (a buffer or a frozen parameter) is averaged from the
    tower instead.
  - The sum is float64, divided by N, then cast float64 → float32 → the tower dtype, once.
  - The manifest gains `from`, `accumulator`, `tensor_sources` (per tensor, `master` or
    `tower`) and `master_index`. Each input also gets `resolved_path` beside its
    `payload_digest`. `source` is still the RSI attribution.
- **`real_ft_run.py --score-checkpoint AVG.safetensors`.** It reads `AVG.safetensors.manifest.json`
  and scores through the same `_checkpoint_step` → `_score_checkpoint` → `_record_score`
  path as one seed's checkpoint. Only the weight loading (`_averaged_weights`) differs from
  `_seed_weights`. The needle worker reloads through the same function.
- **`qd_train.run_control`:**
  - `Checkpoint.read_body`: the JSON body with its `payload_digest` verified, and no tensor
    read.
  - Nested `read_weights` paths: `("optimizer", "masters")` reads the masters without the Adam
    moments.
  - `tensor_refs_from_safetensors`: now shared with `Checkpoint.read`.

### How masters map to names

`masters` is a bare list in the order the optimizer was given its parameters, and the
checkpoint body is written with `sort_keys=True`, so that order cannot be read from the file.
What does hold is the invariant `MasterWeightAdamW` keeps:

- `step()` ends with `live.copy_(master)`;
- an fp32 parameter is its own master.

So master *i*, cast to its parameter's dtype, is byte for byte the tensor of that parameter.
The body records every tower and span-head tensor's digest. `master_names` looks each master
up by (dtype, shape, digest). It refuses in four cases:

- no tensor matches;
- more than one tensor matches;
- two masters match one name;
- the names differ between inputs.

Checks:

- **Tiny model (verified):** `test_a_masters_average_takes_each_parameters_own_master_by_name`
  takes its ground truth from the live steps (`_live` paired with `_masters` by identity).
  The test failed under two injected mapping bugs: masters assigned to the names in sorted
  order, and two same-shape masters swapped.
- **Real checkpoints (verified, CPU):** on the three phase-3 checkpoints in
  `/Users/bharath/qd-campaign/ckpt-p3-s{0,1,2}-weights-2026-09-30/`, every one of the 324
  masters (320 tower + 4 span head) maps to exactly one tensor. The mapping is identical across
  seeds and follows registration order (`embed_tokens`, `layers.0.linear_attn.dt_bias`,
  `A_log`, …), not sorted order. Each input took 13.8–15.7 s to read and 5.9–7.5 s to map.
- **Pretrained snapshot (verified):** `b1485b2f` has 320 text tensors, and no two same-shape
  ones are byte-identical.

### Refusals when scoring an average

Every refusal comes before the tower loads. Each line gives the refusal and the test that
covers it.

- **Manifest not one `ckpt_average` wrote, or no `ft_row_ids`:**
  `test_an_average_naming_no_ft_rows_is_refused`.
- **`--seeds` not the average's seeds:** `test_an_average_scored_under_other_seeds_is_refused`.
- **An ft row named in the manifest is absent from `--ft-ledger`:** the `missing_row` case.
- **An ft row is quick, or does not say it is not quick:** the `quick_row` case.
- **Rows differ in `protocol.recipe_hash`, `data_snapshot_hash`, `tokenizer_hash`,
  `backbone_commit`, `recipe` or `train.optimizer_steps`, or the manifest's step differs:** the
  `other_recipe`, `other_snapshot` and `other_steps` cases.
- **Row *i* does not describe input *i* (seed, arm, device, shard set, backbone):** the
  `rows_out_of_order` and `other_shard_set` cases.
- **A source body hashes to a different `payload_digest`:**
  `test_an_average_whose_source_checkpoint_changed_is_refused`.
- **A source body is not on disk:** `test_an_average_whose_source_checkpoint_is_gone_is_refused`.
  This is the fail-closed decision. Only the `.json` body (~0.5 MB) is needed, never the
  24.66 GiB sidecar, so to score elsewhere, copy the three `.json` files to their recorded
  `resolved_path`.
- **Weights not matching `safetensors_sha256`:** `test_altered_averaged_weights_are_refused`.
- **At argv time, `--ft-row-id`, a single `--seeds`, or `--needle-control` with an average:**
  `test_average_argv_is_refused_before_anything_loads`.

The five refusal cases above that have no test of their own are parameters of
`test_an_average_its_ft_rows_do_not_describe_is_refused_before_the_tower`. The positive
control `test_a_well_formed_average_passes_every_pairing_check_and_reaches_the_tower` runs
the same fixture unaltered and gets as far as building the step.

### The averaged eval row

`test_an_average_of_three_tiny_master_checkpoints_is_scored_end_to_end_on_cpu` checks all of
the following:

- **Tags:** run tag `avg`; recipe tag `avg-score-val` (the shared writer's
  `{tag}-score-val`, as `epoch-score-val`).
- **Protocol seed:** `config.seed` = 20260919, which is no single seed's.
- **`recipe.averaged`:** `seeds`, the full `ft_row_ids`, `manifest_sha256`, `source`,
  `n_inputs`, `protocol_seed` and `suite_seed`.
- **`scored_checkpoint`:** `<name>:<safetensors_sha256>`.
- **Metric `ft_run_row_ids`:** not `ft_run_row_id`, which `shuffled_label_target` reads as a
  single id.
- **Note:** says that whether an average can be the promoted artifact is the human's decision.
- **Quick reasons:** the per-seed rule 8 reasons, plus "fewer than 3 seeds". On the CPU
  fixture the only reason is the device one.

**Suite seed (correction to the J7 premise):** the per-seed rows did **not** use their run
seed for the suites. `prepare_needle`, `prepare_ood` and `prepare_second_pass` all take
`config.seed`. The average uses the same seed and now records it.

Main used to stamp `--verdicts-out` and `--suite-verdicts-out` lines with `--seeds[0]`. They
now take the row's own seed.

## What was measured

No ledger row: nothing ran on a GPU. The tests ran on the Mac CPU in the ml env:

- **New and updated tests:** 28 fail on unmodified tools. That is 21 new tests, plus the 6
  existing `--score-checkpoint` tests (they now pass `suite_seed`) and the deliberately updated
  `SCORED_CHECKPOINT_KEYS` pin. All pass on `c262ca3`.
- **Wider affected set (25 files):** 760 passed, 5 skipped, 1 failed. The failure is
  `test_lint_gate::test_ruff_is_installed_not_merely_declared`, because this worktree has no
  `.venv`.
- **Lint:** the whole-repo ruff check with main's `.venv/bin/ruff` passes.

**Peak host RAM for three real checkpoints (inferred, not measured):**

- `average_masters` reads one input at a time: tower 3.51 GiB + masters 7.04 GiB = 10.55 GiB.
- The float64 accumulators take 14.0 GiB (1,881,825,088 params × 8 B).
- The embedding's transient copies add ~5.7 GiB.
- That comes to **about 30 GiB, whatever N is**, against 74 GiB for three `Checkpoint.read`s
  before copies.
- The macOS `ru_maxrss` of the per-input mapping run (5.84 GiB) understates allocation under
  this host's swap pressure. Read the tool's own `peak resident set` line on the box (Linux)
  instead.

## What is open

- `GAP-J7-AVG-SCORING-NAVIGATION-2026-10-01`. DevMap had no store for this worktree, so main's
  generation 2892 answered. GitPulse returned `REPOSITORY_TRUST_REQUIRED` on every facet. No
  ListAgents was available.
- **Not run:**
  - the full `--from masters` average on real checkpoints (Mac RAM);
  - the averaged score on GPU;
  - needle and OOD on the average path. The byte fixture cannot build them; their code is
    shared with the per-seed path.
- **Human-owned:** whether the average can be the promoted artifact. Nothing here decides it.
- **If J4's ft rows are quick, J7 stops (conditional, unverified).** Scoring an average
  refuses a quick ft row, by the binding decision and rule 8. The A2 rows of 2026-09-30 were
  quick for a corpus reason: a data snapshot that was NotRun because of a capped read. J4 ran
  with `--general-max-rows 200000`, and whether its snapshot came back Ran is unknown from
  here. Step 1's one-liner surfaces this before the average runs. If it fires, stop and
  escalate to the human; do not weaken the check (rules 2 and 8).
- **Row shape differs from a per-seed score row.** The average's row carries
  `ft_run_row_ids` rather than `ft_run_row_id`, and recipe tag `avg-score-val`. Readers built
  for `epoch-score-val` rows (`tools/ft_linear_control.py`, the calibration fit) will not
  treat it as one of theirs. J7 writes it to its own ledger file, so nothing that scans J4's
  ledger sees it.

## First command for the next lane: J7 on the box

Preconditions:

- `=== J4 all done` is in `/home/ubuntu/logs/p4-v3.log`.
- `j7-avg-scoring` is merged to main and pushed. That is a human step; this lane does not push.

J7 runs from `qd-lane3`, which J4 has finished with and which carries the data symlinks.

    ssh -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246
    cd /home/ubuntu/qd-lane3
    git fetch origin && git checkout --detach origin/main
    git merge-base --is-ancestor c262ca3 HEAD || { echo "main lacks c262ca3"; exit 1; }
    export QD_PREP_BIN=/home/ubuntu/bin/qd-prep
    J4=/home/ubuntu/box_p4_j4.sh
    PY=$(grep -o '/home/ubuntu/[^ ]*/bin/python[0-9.]*' "$J4" | head -1)
    arg() { grep -o -- "$1 [^ ]*" "$J4" | head -1 | cut -d' ' -f2; }
    BACKBONE=$(arg --real-backbone); OODREC=$(arg --ood-general-record); TOKJSON=$(arg --tokenizer-json)
    test -x "$PY" && test -n "$BACKBONE" && test -n "$OODREC" || { echo "J4's script does not name the interpreter, --real-backbone and --ood-general-record"; exit 1; }
    test -d "$BACKBONE" && test -f "$OODREC" && { test -z "$TOKJSON" || test -f "$TOKJSON"; } || { echo "an extracted path does not exist (J4 may spell it as a variable): BACKBONE=$BACKBONE OODREC=$OODREC TOKJSON=$TOKJSON"; exit 1; }
    for f in "--out /home/ubuntu/phase4-v3-2026-10-01" "--no-repo-history" \
             "--defect-class data/pool/commitpackft-corpus-v3" "--defect-download data/pool/commitpackft" \
             "--defect-noul data/pool/defect-noul-v1" \
             "--general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json" \
             "--general-max-rows 200000" "--replay-partition" "--rev b381a03cb12e816c380915b1983f4e13cd1c843c"; do
      grep -q -- "$f" "$J4" || { echo "J4 did not run with: $f"; exit 1; }
    done

If J4's script builds its argv from shell variables, the extraction or the flag loop above
fails. Then a human reads `box_p4_j4.sh` and fills in the literal values J4 ran with. Do not
drop the check.

**1. The ft row ids** of J4's three epoch arms, in seed order. The command refuses unless the
ledger holds exactly seeds 0, 1 and 2, all three non-quick. Scoring refuses a quick row, so
this check comes before the 30 GiB average rather than after it:

    FT_LEDGER=/home/ubuntu/ledger/gh200-p4-v3-2026-10-01.jsonl
    IDS=$("$PY" -c 'import json,sys; rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; ft=sorted((r for r in rows if r.get("run_kind")=="ft" and r.get("status")=="completed" and r["recipe"].get("tag")=="epoch" and "shuffled_label" not in r["recipe"]), key=lambda r: r["protocol"]["seed"]); seen=[(r["row_id"], r["protocol"]["seed"], r.get("quick")) for r in ft]; assert [s for _, s, _ in seen]==[0,1,2] and all(q is False for _, _, q in seen), seen; print(" ".join(r["row_id"] for r in ft))' "$FT_LEDGER") || exit 1
    echo "$IDS"

**2. The average.** This runs on the CPU from the masters. Expect a few minutes and about 30 GiB
of RAM; it prints its own peak.

    mkdir -p /home/ubuntu/ckpt/p4-v3/avg /home/ubuntu/j7
    "$PY" tools/ckpt_average.py --from masters --ft-row-ids $IDS \
      --out /home/ubuntu/ckpt/p4-v3/avg/epoch-avg-seed012-masters.safetensors \
      /home/ubuntu/ckpt/p4-v3/epoch-seed0-cuda.json \
      /home/ubuntu/ckpt/p4-v3/epoch-seed1-cuda.json \
      /home/ubuntu/ckpt/p4-v3/epoch-seed2-cuda.json \
      2>&1 | tee /home/ubuntu/logs/j7-average.log

**3. Every gate on the average,** fp32 on the GH200, with J4's split flags:

    "$PY" tools/real_ft_run.py \
      --score-checkpoint /home/ubuntu/ckpt/p4-v3/avg/epoch-avg-seed012-masters.safetensors \
      --ft-ledger "$FT_LEDGER" --seeds 0 1 2 \
      --real-backbone "$BACKBONE" ${TOKJSON:+--tokenizer-json "$TOKJSON"} \
      --score-val --needle --ood --ood-general-record "$OODREC" \
      --score-dtype fp32 --devices cuda --instance lambda-1xgh200 --usd-per-hour 2.29 \
      --ledger /home/ubuntu/ledger/gh200-p6-j7-avg-2026-10-01.jsonl \
      --verdicts-out /home/ubuntu/j7/avg-verdicts.jsonl \
      --suite-verdicts-out /home/ubuntu/j7/avg-suite-verdicts.jsonl \
      --out /home/ubuntu/phase4-v3-2026-10-01 --no-repo-history \
      --defect-class data/pool/commitpackft-corpus-v3 --defect-download data/pool/commitpackft \
      --defect-noul data/pool/defect-noul-v1 \
      --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json \
      --general-max-rows 200000 --replay-partition \
      --rev b381a03cb12e816c380915b1983f4e13cd1c843c \
      2>&1 | tee /home/ubuntu/logs/j7-avg-score.log

The cost is priced at the default 30-minute cap: $1.15 projected, under the rule-4 threshold of
$20. The needle worker is bounded at 2 h. Then bring the row home:

    scp -i ~/.ssh/bharath_m5_macbook_pro.pem ubuntu@192.222.51.246:/home/ubuntu/ledger/gh200-p6-j7-avg-2026-10-01.jsonl /Users/bharath/qd-campaign/p4-v3-2026-10-01/

Expect one eval row with tag `avg-score-val`, `recipe.averaged.source == "masters"` and
protocol seed 20260919. On cuda it should not be quick.
