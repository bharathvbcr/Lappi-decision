# HANDOFF: ens3-scoring (Fable I / I-2), 2026-10-01

Lane worktree `agent-a49d73cb7bb3e7dad`, branch `worktree-agent-a49d73cb7bb3e7dad`. It merged main
`b5575cc` cleanly as `b9e4a1c`. **Run the box commands from `eeb1c67`, not the branch tip** (see
"Open").

## What was measured

- **On GPU: nothing.** Every GPU number below is an estimate. Rule 5: nothing here is green
  until the box rows exist.
- **Locally, from rows that already existed:**
  `AUDIT/j7-avg-ood-diag-2026-10-01/noul-rank-j4-j7g.json`. It is `noul_rank.py` run over the OOD
  suite-verdict lines of the J4 seeds and the J7g average:
  - J4 seeds: rows `2f5fe57a`, `f3612f73`, `02cf5ff4`, from the in-training decodes. Their logits
    are bf16-quantized. That is inferred from the values; the rows record no dtype.
  - J7g average: row `1d93b3ee`, fp32 from the masters.
  - The tool's abstention counts match every row's `ood_abstain.<category>` metrics (prose,
    unseen-language, scrambled):

    | Row | Abstentions |
    | --- | --- |
    | seed0 | 3 / 40 / 3 |
    | seed1 | 9 / 20 / 4 |
    | seed2 | 2 / 56 / 8 |
    | avg | 0 / 0 / 0 |

  - **The average's noul row never wins.** On pass 1 it ranks last of 5 on 147 of 180 cases and
    second on 20 unseen-language cases. Its mean signed gap to the best other row is 6.40 nats.
  - **The seeds' noul row wins** 40 (seed0), 20 (seed1) and 53 (seed2) of 60 unseen-language cases
    on pass 1.
  - The box job below rescores the seeds in fp32 from their checkpoints, so seed, avg and ens3
    compare like for like.

## What changed

| Commit | What | Tests (failing before → passing after) |
| --- | --- | --- |
| `fbd20aa` | `--score-checkpoint A.json B.json C.json` scores a **logit ensemble** (`ens<N>`). Every decode reads the mean of the towers' float64 log-softmax over the slot's own rows. Rows name every ft row and each tower's digests, at T = 1, quick like the average. Refuses mismatched recipes or snapshots, N < 2, and an ft row not in `--ft-ledger`. | `test_real_ft_ensemble.py`: 30 new, all fail on `90e7586` |
| `1297003` | `ckpt_average.py --from masters --norm-preserving --base-snapshot DIR` builds **avg-np**: base + λ·mean(θᵢ − base), with λ = √(n/(1+(n−1)c)) and c measured in float64, streamed tensor by tensor. The manifest records c, every pair's cosine, n, λ and the formula. An explicit λ is accepted only behind `--lambda-not-derived-from-c`, which marks the manifest and makes the row quick. Scoring accepts it, with tags `avg-np-score-val` and `avg-np-needle-length-control`. | `test_ckpt_average_norm_preserving.py`: 23, all fail on `fbd20aa` (orthogonal → λ = √3, identical → λ = 1, missing base refused) |
| `08478ba` | `--suite-logits`: needle lines carry `start_logits`/`end_logits` (noul at `noul_row`). OOD lines carried `row_logits_1/2` already. Refused wherever it would be dropped. | `test_real_ft_suite_logits.py`: 10 new, all fail on `1297003` |
| `fe8918d` | `--score-plan PLAN.json`: several kinds in one invocation, with the shard set, val set and suites built once. Pass `gates` is the `--score-checkpoint` row; pass `ood` is a quick `<tag>-ood-diagnostic` row with `ood_diagnostic.*` metrics, no gate, and lines under `gate: ood_abstain.diagnostic`. Each kind is loaded, run, freed, then written to `<stem>-<kind><suffix>`. | `test_real_ft_score_plan.py`: 41, collection fails on `08478ba`. Oracle: each plan kind's row and verdicts equal its single run's; the prelude runs once; the model is unreachable before the next loads. |
| `eeb1c67` | `campaign/fable-i-ood-diag-plan-2026-10-01.json` (the box plan), `AUDIT/.../noul_rank.py` (throwaway analysis) and its J4/J7g output | the committed plan passes every argv check with the box argv |
| `b9e4a1c` | merge of main `b5575cc` | 450 passed, 1 skipped on the merged tree |
| `6c24076` | `qd_train/composed_slice.py`, the report-only slice's scoring core (conditions 6 and 8). `needle.depth_bucket_label` is now the one bucket rule. | `test_composed_slice.py`: 20 (the module is new). Swapping condition 8's `all` for `any` fails 2. |

- **Size:** +4131 / −265 lines over 14 files (`90e7586..eeb1c67`).
- **Ruff:** clean.
- **Single-step byte-identity:** `_decode` was refactored through `BatchReadout`. The seed and
  average verdict JSON is byte-identical before and after.

## Open

- **GAP-MAIN-QD-DATA-DRIFTS-FROM-V3-SHARD-FINGERPRINT-2026-10-01** (human).
  - Main's 351e6b4 changed two comment lines of `python/qd_data/pool_builder.py`.
  - The v3 header fingerprint no longer matches main or `b9e4a1c`. It does match `eeb1c67`,
    qd-lane5 and qd-lane6. This was measured against the v3 train header.
  - `ShardReader` refuses the drift, so the box commands pin `eeb1c67`.
  - What `eeb1c67` lacks from main: main's report-only per-family metrics on eval rows (`45e2d1d`)
    and the train-step perf flags. Neither touches scoring decisions.
- **GAP-SUITE-LOGITS-NEEDLE-WORKER-2026-10-01.**
  - A single `--score-checkpoint` gate row refuses `--suite-logits --needle`, because the needle
    worker (prep-perf's block) decodes without pointer scores.
  - Whichever branch merges second adds `logits=args.suite_logits` to the worker's
    `needle_predictions(...)` call in `main()` and drops the refusal's `--needle` clause.
  - `--needle-control` and `--score-plan` (in-process, cuda) carry the scores today.
- **GAP-ENSEMBLE-ROW-HAS-NO-PROMOTION-KIND-2026-10-01** (human). Neither `promotion_verdict` nor
  `promotion_verdict_avg` judges an `ens3` row. Read from the code.
- **GAP-ENS3-SCORING-NAVIGATION-2026-10-01.** DevMap answered for main's index only, and GitPulse
  asks to trust the repository on every facet.
- **Not run:**
  - every GPU job below;
  - the in-process 8K needle for a 3-tower fp32 ensemble on the GH200 (memory inferred: about
    9 GiB of weights per tower, against about 85 GiB free);
  - avg-np's peak RAM on the real 2B (inferred about 44 GiB: the masters average's about 30 GiB,
    then the float64 means of about 14 GiB plus one tensor per input).
- **Composed long-context slice (task 4): core done, decode blocked.**
  - The torch-free core is committed as `qd_train/composed_slice.py` (`6c24076`, 20 tests on
    three real composed rows). It covers case parsing and checks, hunks, the head-row → token →
    lines map, the gold-alignment refusal, condition 8's hit rule, and condition 6's tables:
    `composed.<set>.<population>.<cut>.<cell>.*` plus deltas, Wilson CIs, and diag halves as sets
    of their own.
  - **Blocked:** the decode that feeds it. Compose's commit adding
    `report_only`/`span_collapse_policy` to `ShardHeader` (and its composed-row loader) is not on
    main. Main's `ShardReader` refuses the slice header (its `shard_hash` covers those fields),
    and the choice labels need that loader. The compose lane lands it after v4's row, the lead
    merges it, and this lane merges main.
  - **Then:** a `composed` plan pass, which:
    - opens `<slice>/shards/val-report-only-composed`, requiring `report_only` true and
      `span_collapse_policy` refuse-gold, the train remap and the tokenizer;
    - pairs the `compose:*` sequences with their corpus rows;
    - decodes one-row batches;
    - feeds `SliceVerdict`s to `slice_metrics` on a quick `<tag>-composed-slice` row.

## Box commands

Every box command runs from a clean clone that holds `data/pool` (as qd-lane5 does), checked out
at `eeb1c67`. No ssh from this lane; the lead deploys.

    cd /home/ubuntu/qd-laneN && git fetch origin && git checkout --detach eeb1c67
    [ -z "$(git status --porcelain)" ] || { echo dirty; exit 3; }
    export HF_HUB_OFFLINE=1 QD_PREP_BIN=/home/ubuntu/bin/qd-prep
    PY=/home/ubuntu/qd-venv/bin/python
    REC=/Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
    BACKBONE=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
    FT_LEDGER=/home/ubuntu/ledger/gh200-p4-v3-2026-10-01.jsonl
    J7_LEDGER=/home/ubuntu/ledger/gh200-p6-j7-avg-2026-10-01.jsonl
    DATA="--out /home/ubuntu/phase4-v3-2026-10-01 --no-repo-history --defect-class data/pool/commitpackft-corpus-v3 --defect-download data/pool/commitpackft --defect-noul data/pool/defect-noul-v1 --general-record $REC --general-max-rows 200000 --replay-partition --rev b381a03cb12e816c380915b1983f4e13cd1c843c"
    mkdir -p /home/ubuntu/fable-i /home/ubuntu/logs

### (a) Diagnostic job: one invocation

The job scores OOD for seed 0, 1, 2 and the average (fp32, quick `*-ood-diagnostic` rows), then
the ensemble's full gate row: val, permutation, needle in-process with pointer scores, and OOD.
Paired with ft rows `80b19a41`, `9b108fe2` and `8b600511`.

    timeout 3h "$PY" -u tools/real_ft_run.py $DATA \
      --score-plan campaign/fable-i-ood-diag-plan-2026-10-01.json --ft-ledger $FT_LEDGER \
      --real-backbone $BACKBONE --score-val --needle --ood --ood-general-record $REC --suite-logits \
      --score-dtype fp32 --devices cuda --instance lambda-1xgh200 --usd-per-hour 2.29 \
      --wall-clock-cap-s 10800 --ledger $J7_LEDGER \
      --verdicts-out /home/ubuntu/fable-i/verdicts.jsonl \
      --suite-verdicts-out /home/ubuntu/fable-i/suite-verdicts.jsonl \
      2>&1 | tee /home/ubuntu/logs/fable-i-diag.log

**Estimate: about 1.5 h.** All parts inferred, not measured:

- prelude: about 5 min;
- four OOD kinds: a load (24–60 s) plus 2 × 180 short cases each, about 8 min together;
- ens3:
  - three tower loads: about 3 min;
  - val, permutation and OOD: 3 × `1d93b3ee`'s 1,089 s decode, about 55 min;
  - 8K needle: 3 × an unmeasured per-tower time; I assumed about 5 min each.
- **Price:** the 3 h cap is $6.87. That is under $20 on a single GPU, so it needs no human yes.

Each kind's files are written when that kind finishes: `suite-verdicts-{seed0,seed1,seed2,avg,ens3}.jsonl`
and `verdicts-ens3.jsonl`. The ensemble runs last, so the OOD kinds survive an ensemble failure.
To leave the needle out, drop `--needle`; the ens3 row then records it as not run.

### (b) Analysis, on the Mac after `scp` of `/home/ubuntu/fable-i/`

    python3 AUDIT/j7-avg-ood-diag-2026-10-01/noul_rank.py \
      fable-i/suite-verdicts-{seed0,seed1,seed2,avg,ens3}.jsonl \
      --out AUDIT/j7-avg-ood-diag-2026-10-01/noul-rank-fable-i-fp32.json

The lines carry `score_kind`, so no `--kind` is needed. Each line's top is checked against its
own logits. Gaps are in nats: a log-softmax only shifts a row, so a seed's raw-logit gap and the
ensemble's mean log-prob gap are both log ratios.

### (c) avg-np build (CPU, Fable I-2)

    IDS=$("$PY" -c 'import json,sys; rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; ft=sorted((r for r in rows if r.get("run_kind")=="ft" and r.get("status")=="completed" and r["recipe"].get("tag")=="epoch" and "shuffled_label" not in r["recipe"]), key=lambda r: r["protocol"]["seed"]); seen=[(r["row_id"], r["protocol"]["seed"], r.get("quick")) for r in ft]; assert [s for _, s, _ in seen]==[0,1,2] and all(q is False for _, _, q in seen), seen; print(" ".join(r["row_id"] for r in ft))' "$FT_LEDGER") || exit 1
    "$PY" -u tools/ckpt_average.py --from masters --norm-preserving --base-snapshot $BACKBONE \
      --ft-row-ids $IDS --out /home/ubuntu/ckpt/p4-v3/avg/epoch-avgnp-seed012-masters.safetensors \
      /home/ubuntu/ckpt/p4-v3/epoch-seed0-cuda.json /home/ubuntu/ckpt/p4-v3/epoch-seed1-cuda.json \
      /home/ubuntu/ckpt/p4-v3/epoch-seed2-cuda.json 2>&1 | tee /home/ubuntu/logs/avgnp-build.log

- It prints c and λ. J4's delta cosine predicts c ≈ 0.20 and λ ≈ 1.46
  (`AUDIT/.../delta-cosine.json`). The tool measures c itself, over every tower tensor with a
  master.
- **Estimate:** the masters average's few minutes, plus one more streamed read of the three
  sidecars and the base, so about 10 min (inferred). Peak RAM about 44 GiB (inferred).

### (d) avg-np gate row, exactly as `1d93b3ee` (`box_q_j7g.sh`'s command)

The command is the same as `1d93b3ee`'s, with the avg-np checkpoint swapped in:

    timeout 2h "$PY" -u tools/real_ft_run.py $DATA \
      --score-checkpoint /home/ubuntu/ckpt/p4-v3/avg/epoch-avgnp-seed012-masters.safetensors \
      --ft-ledger $FT_LEDGER --seeds 0 1 2 \
      --real-backbone $BACKBONE --score-val --needle --ood --ood-general-record $REC \
      --score-dtype fp32 --devices cuda --instance lambda-1xgh200 --usd-per-hour 2.29 \
      --wall-clock-cap-s 5400 --ledger $J7_LEDGER \
      --verdicts-out /home/ubuntu/j7/avgnp-verdicts.jsonl \
      --suite-verdicts-out /home/ubuntu/j7/avgnp-suite-verdicts.jsonl \
      2>&1 | tee /home/ubuntu/logs/avgnp-score.log

- This writes one `avg-np-score-val` row naming the three ft rows. It is quick like the average.
- `--suite-logits` is left off: with `--needle` on a single gate row it is refused
  (GAP-SUITE-LOGITS-NEEDLE-WORKER). The OOD lines carry `row_logits_1/2` regardless.
- To get needle pointer scores as well, use a one-kind plan,
  `{"kinds":[{"name":"avgnp","checkpoints":[".../epoch-avgnp-seed012-masters.safetensors"],"seeds":[0,1,2],"passes":["gates"]}]}`,
  with `--suite-logits`. The recipe is identical (pinned by the plan oracle test), but the needle
  is decoded in-process.
- **Estimate:** about 30 min, like J7g (`1d93b3ee` decode 1,089 s plus the needle worker and the
  prelude). Inferred.

### (e) avg-np needle length control

    timeout 1h "$PY" -u tools/real_ft_run.py $DATA \
      --score-checkpoint /home/ubuntu/ckpt/p4-v3/avg/epoch-avgnp-seed012-masters.safetensors \
      --ft-ledger $FT_LEDGER --seeds 0 1 2 --real-backbone $BACKBONE --score-val \
      --needle --needle-control 1024,2048,4096 --suite-logits \
      --score-dtype fp32 --devices cuda --instance lambda-1xgh200 --usd-per-hour 2.29 \
      --wall-clock-cap-s 5400 --ledger $J7_LEDGER \
      --suite-verdicts-out /home/ubuntu/j7/avgnp-needle-control-1024-2048-4096.jsonl \
      2>&1 | tee /home/ubuntu/logs/avgnp-needle-control.log

- This writes one `avg-np-needle-length-control` row. It is comparable with `0b86fae3` (avg) and
  with J4's `0b6c8d4a`, `21dcda47` and `30539c6a`.
- **Estimate:** about 10 min (`0b86fae3` decoded in 205.8 s), plus the prelude and the load.

## First command for the next lane

From a checkout of this branch (or of main once it is merged):

    /Users/bharath/.local/bin/uv run --no-project --python /Users/bharath/.venvs/ml/bin/python \
      --with pytest --with hypothesis --with datasketch python -m pytest \
      python/tests/test_real_ft_score_plan.py python/tests/test_real_ft_ensemble.py \
      python/tests/test_real_ft_suite_logits.py python/tests/test_ckpt_average_norm_preserving.py \
      -o addopts= -q -k "not mps"
