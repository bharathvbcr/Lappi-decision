# Long-context arm: `qd-mutate compose`, v4, and the report-only slice (2026-10-01)

Lane `longctx-compose`, branch `worktree-agent-a23c0fc7f22b1097c`. It covers Fable G5(ii) and
G11, and rounds I and K and the slice ruling as the lead relayed them. Every claim is labelled
**verified** (run here, output or ledger row cited), **inferred** or **unverified**. Not
merged into main and not pushed: the lead merges.

## What was measured (ledger rows)

| Row | Ledger (in repo) | What | code_commit |
| --- | --- | --- | --- |
| `d96409bd-4890-4d7f-9e7c-ca4cafbdb9a8` | `ledger/mac-phase4-v4-shards-2026-10-01.jsonl` (byte copy of `/Users/bharath/qd-campaign/ledger-mac-phase4-v4-2026-10-01.jsonl`) | **v4 shard build** | `881ab304…` (clean) |
| `dafe86af-ac2a-42ca-a3cd-d2b624740783` | `ledger/mac-slice-v4-2026-10-01.jsonl` (byte copy of `/Users/bharath/qd-campaign/ledger-mac-slice-v4-2026-10-01.jsonl`) | **report-only slice build** | `92a0b528…` (clean) |

- Both rows are `smoke` and `quick`. Every pipeline row is: one seed and `--no-repo-history`
  (rule 8). Both ledger copies pass `verify_chain` (**verified**).
- The pipeline cannot record F, the J-runs or any scoring. None of them ran here (rule 5):
  they need the GH200.

### v4 (`d96409bd`), out dir `/Users/bharath/qd-campaign/phase4-v4-2026-10-01`

- **Recipe** (`recipe_hash` `4662ca85…`, `data_snapshot_hash` `ea3215c4…`):
  - `--defect-class data/pool/commitpackft-composed-v1`: examples `a63563e5…`, which loads its
    base v3 corpus `610bb1f0…` first.
  - `--defect-noul data/pool/defect-noul-v3b` (`b194f14c…`).
  - The general families from record `a0841f0d…` at `--general-max-rows 200000`.
  - `--max-seq-len 8192`, `--vocab full`, `--val-shards`.
  - `--span-collapse-policy refuse-gold`, with scope "train and replay; val refuse-any".
  - No replay. Composed rows only in train (`--val-rows 0`).
- **Split** (**verified**): train 289,142 / val 16,124 / held-out 37,705 rows.
  - `code.defect_class` train is 74,864 rows: 25,000 composed, 44,860 v3 and 5,004 noul.
  - Val holds 2,304 defect rows and **0 composed ids**: I checked all 18,428 entries of
    `shards/val/sequence_index.json` for `:compose:`.
- **Shard sets** (**verified**, read back through `ShardReader`):

| Set | `shard_hash` | Sequences | Tokens | Positions at bucket width | Max width |
| --- | --- | --- | --- | --- | --- |
| `shards/train` | `8bcf56ad89bbf059f9926035b7b798796a0a41dd4960c2939b66671f24d2fb80` | 363,950 | 306,926,895 | 323,073,710 (5.00% padding) | 7,936 |
| `shards/train-no-decode` (stage-8 copy, never trained on) | `fd35cbc5a48d8ea8130e6793aa62963ad17c5fb3459f3a3e9cb724b30091ee3a` | — | — | — | — |
| `shards/val` (refuse-any, a gate population) | `ef06ab99eef107dd608424c80117b81b3986ce5ea827d2d4f47f990d960b1bc5` | 18,223 | 5,031,887 | 5,170,621 (2.68%) | 1,022 |

- 0 rows over 8,192 on train or val.
- Stage-7 read-back: 0 violations of any span invariant over 363,950 sequences.
  `span_mapping_decode_verified` holds over 180,698 span sequences.
- **No composed noul.** Shipping v4 without composed noul was accepted. The noul-v3b rows are
  single-file, and none of the 25,000 composed rows has class noul. **The v4 row's notes do
  not say this**: I did not add it before the build, and the ledger is append-only. This
  handoff is where it is recorded.

### Fable round K's pre-registered readout on `d96409bd` (**verified**, the row's metric detail)

These are composed span slots written, under refuse-any (before) and under refuse-gold
(after). The bar is ≥ 0.90 in every length bin. Every bin passes and the worst is 0.9973, so
there was no STOP.

| Length (tokens) | Rows | refuse-any | refuse-gold |
| --- | --- | --- | --- |
| 0–999 | 91 | 0.4835 | 1.0 |
| 1,000–1,999 | 3,882 | 0.4415 | 0.9992 |
| 2,000–2,999 | 3,819 | 0.3485 | 0.9992 |
| 3,000–3,999 | 3,765 | 0.2643 | 0.9995 |
| 4,000–4,999 | 3,766 | 0.2007 | 0.9973 |
| 5,000–5,999 | 3,799 | 0.1698 | 0.9979 |
| 6,000–6,999 | 3,977 | 0.1489 | 0.9977 |
| 7,000–7,999 | 1,901 | 0.1957 | 0.9984 |

| Files | Rows | refuse-any | refuse-gold |
| --- | --- | --- | --- |
| 0–7 | 4,447 | 0.5788 | 0.9996 |
| 8–15 | 9,503 | 0.3246 | 0.9988 |
| 16–23 | 5,809 | 0.1048 | 0.9985 |
| 24–31 | 3,484 | 0.0456 | 0.998 |
| 32–39 | 1,481 | 0.0128 | 0.9939 |
| 40–47 | 275 | 0.0109 | 1.0 |
| 48 | 1 | 0.0 | 1.0 |

Refused under refuse-gold (all slot-scoped; each row keeps its choice sequence), from
`shards/train/sequence_index.json`:

| Reason | Composed | v3 |
| --- | --- | --- |
| Gold line shares a token | 14 | 15 |
| NFC-unstable context (refused under either policy; not a collision) | 24 | 3 |

So 38 of the 25,000 composed span slots were not written.

### Composed rows by measured length × needle depth (**verified**, v4's train shards)

These are span-sequence lengths from `ShardReader`. Depth is `needle_index / (n_files − 1)`.
`clean` rows have no needle.

| Length | 0–20% | 20–40% | 40–60% | 60–80% | 80–100% | clean | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0–1k | 23 | 15 | 7 | 13 | 18 | 15 | 91 |
| 1–2k | 747 | 577 | 558 | 566 | 833 | 598 | 3,879 |
| 2–3k | 723 | 594 | 509 | 572 | 752 | 666 | 3,816 |
| 3–4k | 670 | 546 | 566 | 611 | 723 | 647 | 3,763 |
| 4–5k | 655 | 608 | 584 | 620 | 749 | 540 | 3,756 |
| 5–6k | 636 | 586 | 616 | 595 | 716 | 642 | 3,791 |
| 6–7k | 703 | 643 | 644 | 633 | 679 | 666 | 3,968 |
| 7–8k | 336 | 319 | 300 | 282 | 338 | 323 | 1,898 |

- **By files**: 0–7: 4,445 · 8–15: 9,492 · 16–23: 5,800 · 24–31: 3,477 · 32–39: 1,472 ·
  40–47: 275 · 48: 1. That is 24,962 written span slots.
- **Corpus facts** (`data/pool/commitpackft-composed-v1/manifest.json`, **verified**):
  - 3–48 files per row. 20,898 needle files, each the needle once (diagnostic 1:
    0 violations).
  - Needle rate: min 0.0333, median 0.0455, max 0.125. 13.9% of files are below the floor of
    10 appearances. Appearances range 8–30.
  - Classes: clean 4,102, cosmetic 4,213, logic 5,710, stub 10,975.

### Total positions and F's projected cost (**inferred** from J2's rate, not measured on v4)

- 323,073,710 positions per epoch at bucket width. At J2's 16.6K pos/s (measured at
  `--batch-tokens 35403`, batches up to 8,441 wide) that is **5.41 h per seed per epoch**.
- At $2.29/h: **$12.38 per seed** and about $37 for three seeds, before scoring and needle
  controls. That fits inside the 32,400 s cap per seed.
- v4's width mix differs from J2's, so the rate is an assumption.
- The cap prices each arm at 9 h × $2.29 = $20.61, which is ≥ `APPROVAL_FREE_USD`.
  `real_ft_run` therefore refuses F without `--approved-by`, and an agent cannot supply it. The
  lead has said the approval is theirs, citing the human's lifting of the $20 single-GPU bound
  on 2026-10-01. **This is a human action, not a value to fill in.**

### The report-only slice (`dafe86af`), out dir `/Users/bharath/qd-campaign/slice-v4-2026-10-01`

- **Corpus**: `data/pool/commitpackft-composed-slice-v1` (manifest committed `92a0b52`;
  examples `f3c8c60c…`, gitignored).
  - Manifest `mode: report_only`, 1,550 val rows:
    - 1,250 composed val rows;
    - 150 `seen_filler` diag rows, whose needle is a train file v4 trained on as a filler
      ≥ 5 times (`diag.train_corpus` = composed-v1 `a63563e5…`);
    - 150 `unseen` diag rows, whose needle is a val file outside the val universe.
  - The corpus manifest's own sha256 is `392d14ae…`.
- **Set**: `shards/val-report-only-composed`, `shard_hash`
  `042d9b5034f9f1b2a4213be1601ed881d16adf9422fc14eefbd812aad28eac9d`, `header.json` sha256
  `8b018743…`.
  - The header carries `report_only: true` and `span_collapse_policy: refuse-gold`, both under
    the hash.
  - `remap_hash` `e7d0890c…` equals v4's train header; the build refuses otherwise.
  - 3,099 sequences, 13,159,610 tokens, max 7,801.
- **Rows**: all 1,550 reached val. 0 were dropped by mixture or dedupe, 0 share an id with v4's
  gate val set (16,124 ids, read through `ShardReader`), and 0 claim the needle suite's
  identity.
- **Exclusions**: one. A slot gold collision on `compose:diag:000206` (`unseen`, 16 files,
  logic).
- The row records:
  - `report_only`;
  - the scope "report-only long-context slice; gate val and needle suite refuse-any";
  - `span_collapse_non_gold`;
  - the manifest and header hashes;
  - `composed_span_survival_by_{length,files}` under both policies, as a readout and not a
    gate.
- Worst length bin 0.9956 under refuse-gold. Under refuse-any, 461 of 1,550 span slots would
  have been written.
- **Content overlap with gate val is by design and is not checked.** The composed val rows'
  needles are v3's val mutated examples, the same diffs gate val scores alone, here embedded
  in longer contexts. Condition 2's check is **by row id**, and it is empty by construction.
- **Its train set is a carrier, not a training set.** It is 271 train-split rows of a 300-row
  sample of v3, which the pipeline needs to measure. Its `padding_waste` gate is recorded and
  decides nothing.
- **Fable condition 9:** fallback 9 is used. The marker is the compose manifest's existing
  `mode: report_only`, enforced in `tools/real_tokenizer_pipeline.py`. `python/qd_data` is
  untouched since v4's build, so its fingerprint equals v4's.

## The raw census closes the secondary source

`target/scratch/census.py` ran over the full bigcode/commitpackft dumps. It is throwaway and
not shipped, and it uses difflib's 3-context diff as an **approximation** of qd-mutate's
renderer:

- 63,058 licence-admitted changed rows: train 57,059, val 3,089, held-out 2,910.
- The longest diff is **1,011 tokens**, one held-out row.
- **No row has 2k–8k tokens**, and no new file exceeds 4,428 characters.

Long raw PRs are not in this corpus. **Composed rows are the campaign's only long data.**

## G11: own-repo inventory (commits **not run**)

`target/scratch/inventory.json` covers 42 repos under `/Users/bharath/Code` (apps 10,
devtools 10, research 11, web 6, other 5).

| Language | Files | Bytes | Files over 8k bytes |
| --- | --- | --- | --- |
| go | 58,425 | 1,316,214,849 | 18,818 |
| python | 54,677 | 862,321,281 | 23,210 |
| typescript | 4,086 | 32,067,265 | 1,211 |
| rust | 2,717 | 55,761,083 | 1,660 |
| swift | 1,633 | 22,459,918 | 822 |

- Licences: none found 18, MIT 14, Apache-2.0 10.
- **Commit counts: not run.** The harness refuses git against any repo but this worktree.
- **Unverified**: how much of the Go and Python bytes is vendored or third-party. 1.3 GB of
  Go across 58k files is far more than these repos' own code would be, so treat the table as
  an upper bound on own-authored code. Using own repos for training and eval was approved on
  2026-09-28 (memory); nothing here was ingested.

## What changed (commits on this branch, after merge-base `a5e5ab8`)

| Commit | What |
| --- | --- |
| `9f8c913` | `qd-mutate compose` (Rust): PR-shaped multi-file defect rows from v3's own examples |
| `ff24d26` | Fable round I: symmetric roles, balanced needle-once, least-used-first fill with floor/soft/hard caps, 3–48 files. Also the composed loader in `qd_data.defect_class` and the pipeline's `--max-seq-len` |
| `5d12a44` | `commitpackft-composed-v1` manifest (v4's 25,000 composed train rows) |
| `881ab30` | Fable round K: `--span-collapse-policy refuse-gold` (train only), the survival readout, and `compose --diag-rows/--diag-train-corpus` |
| `741a019` | v4 ledger copy; GAP-V4-TRAIN-HEADER-OMITS-SPAN-COLLAPSE-POLICY and two more gaps |
| `037c2b8` | Report-only slice: `ShardHeader.{span_collapse_policy, report_only}` under `shard_hash`, `write_shards(report_only=)`, `open_val_set` refusal, `--report-only-slice/--gate-set` |
| `92a0b52` | `commitpackft-composed-slice-v1` manifest |
| (this commit) | slice ledger copy and this handoff |

- Lines from `a5e5ab8` to `92a0b52`: **+6,720 / −23** across 21 files. About 1,500 of the
  added lines are committed JSON manifests.
- Code is `compose.rs` 2,486 + `main.rs` 157 + Rust tests 520, then Python:
  - `real_tokenizer_pipeline.py` +682;
  - `defect_class.py` +222 (frozen since `881ab30`);
  - `shards.py` +125, `artifacts.py` +53;
  - `real_ft_run.py` +1;
  - tests about 890.
- `tools/compose_split_map.py` (102 lines) is the thin bridge that hands the Rust composer
  `qd_data`'s own `assign_repo` split map. It is not a second split.

### Tests (**verified**)

- **Rust**:
  - `cargo test --release -p qd-mutate`: all pass, including 11 compose integration tests;
  - `cargo clippy -p qd-mutate --all-targets -- -D warnings` is clean;
  - `Cargo.lock` was restored after each build.
- **Python**: the full suite on the worktree at `037c2b8`'s content gives 3,025 passed and 3
  failed. None of the 3 is this lane's:
  - `test_gaps_writer`: the worktree directory name;
  - `test_lint_gate`: no ruff in the worktree `.venv`;
  - `test_qd_data_wire_agreement`: two runtime refusal kinds are unmapped, from
    `crates/qd-runtime`, which this lane did not touch. I infer it is pre-existing; I did not
    run it at the base.
- ruff is clean on every changed file.
- Each new test fails on the pre-change code. I verified this by running them against a copy
  of 881ab30's `python/` and `tools/`:
  - 10 in `test_shards.py` (9 new, plus the updated byte-for-byte test) and 1 in
    `test_real_ft_score_val.py` fail;
  - the new `test_pipeline_report_only_slice.py` fails at collection;
  - round K's 4 shard tests were verified the same way when `881ab30` was committed.

### Consumer claims (**verified** against the code)

- Gate scoring of val goes through `tools/real_ft_run.py::open_val_set`, for both
  `--score-val` and `--score-checkpoint`. It now refuses a report-only or non-refuse-any
  header before the remap, labels or tower.
- `python/qd_train/needle.py` builds its synthetic suite (pool id `needle-suite`) and reads no
  header.
- `qd-gate-report` (`crates/qd-runtime/src/bin/qd_gate_report.rs`) reads `--verdicts-out`
  files, not shards.
- `tools/export_letter_parity.py` opens `shards/val` only.
- No train-side consumer infers the policy from a header, as the lead directed.

## What is open (gap ids)

- **GAP-V4-TRAIN-HEADER-OMITS-SPAN-COLLAPSE-POLICY** (new, owner lead).
  - v4's train sets were written refuse-gold, but their headers predate the field:
    `shard_hash` `8bcf56ad…` (train) and `fd35cbc5…` (train-no-decode).
  - For those hashes the recipe is the authority. Header silence is **not** refuse-any.
- **GAP-G9A-RUNTIME-SPAN-CANDIDATES-MUST-SHARE-COLLAPSED-TOKENS** (new, owner lead).
  - A model trained on v4 points over unique line-start tokens.
  - The runtime must build line start → token the same way. Not checked: `crates/qd-runtime`
    was outside this lane.
- **GAP-COMPOSE-NEAR-DUP-FILES-ACROSS-SPLITS-UNCHECKED** (new).
  - Dedupe runs on whole rows, so a composed file whose fork lives in another split is not
    caught.
- **GAP-GITPULSE-UNTRUSTED-AND-NO-LISTAGENTS-LONGCTX-2026-10-01** (open). GitPulse refused
  every facet and `ListAgents` is not exposed here. Mitigated by an isolated worktree and file
  claims agreed by message with the prep-perf and scorer lanes.
- **GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE** (existing). refuse-gold settles it for training
  shards. Gate val is still refuse-any by rule 2, and whether to move it is the human's call.
  The slice's per-cell refuse-any vs refuse-gold delta is the evidence for that call.

## Unverified, named

- **Needle gate vs composed shape.**
  - The needle suite is synthetic: one generated haystack, controlled depth.
  - Composed rows are real multi-file PRs.
  - Neither measures the other. The slice is the first readout on the composed shape.
- **The rate on v4's width mix**: see the F projection above.
- **The token estimate**: compose's estimate (0.902 tokens/piece) is a proxy. Every table
  above is measured from shards, except the corpus manifest's `length_est_histogram`.

## F's data argv (**verified** on the Mac through `real_ft_run.main`'s own prelude)

    SPLIT=(--out /home/ubuntu/phase4-v4-2026-10-01 --no-repo-history
           --rev 881ab304f15ea13529002391dda8520c2ea47af4
           --defect-class data/pool/commitpackft-composed-v1
           --defect-download data/pool/commitpackft
           --defect-noul data/pool/defect-noul-v3b
           --general-record /Users/bharath/.cache/qd-decision/general/fetch-record-2026-09-29.json
           --general-max-rows 200000)

**How it was verified.** `target/scratch/f_prelude.py` (throwaway) calls `main` unchanged
and stops at `_resolve_batch_tokens`, before any tower loads. With `--out` pointed at the Mac
out dir and `--epoch --no-memorise --score-val --batch-tokens 35403` added, it printed
"PRELUDE OK in 222 s":

- every label paired by id to the train sequence index;
- the val set opened with the train set's remap `e7d0890c…`.

**Wrong argv is refused** (both run):

- J4's `--replay-partition` left in: refused by `corpus_facts`, because `train-replay.json` is
  absent.
- J4's `--defect-class data/pool/commitpackft-corpus-v3` left in: refused by `pair_labels`
  ("49962 sequence(s) name a (row_id, slot_name) this rebuild does not have").

**The rest of F's flags.** J4's recipe, plus `--seeds 0 1 2 --batch-tokens 35403
--wall-clock-cap-s 32400`, every seed `--needle-control 1024,2048,4096`, and
`--approved-by` (the lead's). F trains from `shards/train` only. `train-no-decode` is never
read.

### What the box needs beyond the checkout (contains `881ab30`)

| File | Size | sha256 | |
| --- | --- | --- | --- |
| out `shards/train/`, `shards/val/` | 1.2 G, 24 M | `8bcf56ad…`, `ef06ab99…` | |
| out `data/pool/train.json` | 152,784,427 | `7feb6ee6…` | |
| out `data/pool/val.json` | 8,688,279 | `6f81735f…` | |
| `data/pool/commitpackft-composed-v1/examples.jsonl` | 522,745,557 | `a63563e5190aa413c579a911b560e105267c4829da460fb03d02289ab0d075a8` | **new** |
| `data/pool/defect-noul-v3b/examples.jsonl` | 4,760,396 | `b194f14c75e1ac959f3658e0dda6a07ef4c41ff026a416a39dd526a9dfb13a5e` | **new** |
| `data/pool/commitpackft-corpus-v3/examples.jsonl` | 182,816,385 | `610bb1f0…` | J4's; the composed loader reads its base |
| `data/pool/commitpackft-pool-v2.jsonl` | 118,791,181 | `2cb31231…` | J4's |
| `data/pool/commitpackft/{go,python,rust,typescript}.jsonl` | 12.4 M / 135.9 M / 7.4 M / 14.6 M | `05a65e97…` / `d167da37…` / `5611f370…` / `43d32596…` | J4's |
| general record and caches | — | record `a0841f0d…` | J4's |

- Out `data/heldout/heldout.json` is read by nothing F runs. This is **inferred** from the code
  paths, not traced, so it can stay on the Mac.
- The box also needs `QD_PREP_BIN` (qd-prep), as for J4.

## First command for the next lane

The lead merges `741a019`, `037c2b8`, `92a0b52` and this commit into main. Then it ships v4
and starts F as above. The scorer lane (ens3, `python/qd_train/composed_slice.py`) merges main
and decodes the slice, reading it through a reader that requires `report_only` and
refuse-gold:

    rsync -a /Users/bharath/qd-campaign/slice-v4-2026-10-01/shards/val-report-only-composed/ ubuntu@192.222.51.246:/home/ubuntu/slice-v4-2026-10-01/shards/val-report-only-composed/ -e "ssh -i ~/.ssh/bharath_m5_macbook_pro.pem"

The host and key are the ones `HANDOFF/v31-noul-reform-2026-10-01.md` ships to.

The scorer has the slice's complete exclusion list: one gold collision, 0 dropped before
write. It pins its buckets to that list before anything is scored.
