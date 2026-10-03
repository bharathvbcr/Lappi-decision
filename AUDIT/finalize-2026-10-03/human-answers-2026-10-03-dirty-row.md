# The human's answers: the rebuild's -dirty build row, and no-mask (2026-10-03, ~23:26Z)

## 1. The rebuild's ledger row reads `-dirty`

**What happened.** The v5 rebuild passed every pre-registered check:

- stage 1b and stage 2 matched the prediction exactly;
- the rescan was CLEAN;
- A7 at 8e6a009 refused on exactly the 13 predicted keys;
- A7 under reading C passed 45/45;
- code.defect_class's token share was 56.0625%, at or above the 55.8% bound.

But its ledger row records `code_commit` = `8e6a00952a09e31a09cc3f5270229a935b7235cc-dirty`.

- The row is c3bd0374-4b96-48dd-82fc-3cad25fef05a, written at 2026-10-03T23:16:45Z.
- The file is `ledger/mac-v5-shards-2026-10-03.jsonl`, committed on v5-build in 5192bd4.

**Evidence**

- **[V]** `qd_train/ledger.py` `_git_commit` marks a row dirty on any `git status --porcelain` line,
  untracked files included.
- **[V]** The first build's row created that ledger file. It stayed untracked under the rule "no
  v5-build commit between the scan and the build's ledger row".
- **[V]** The build's own tracked-tree check logged `tree: v5-build 8e6a009… clean` at 22:28:36Z
  (`build/v5-build/build-r.log`, `--untracked-files=no`).
- **[V]** At 23:18Z after_rebuild found no tracked change. At 23:19Z the only porcelain line was
  `?? ledger/mac-v5-shards-2026-10-03.jsonl`.
- **[V]** The row's `recipe.rev` is 8e6a009….
- **[I]** Nothing else changed in v5-build-wt between 22:28 and 23:16, and no cargo ran: qd-prep was
  prebuilt and its sha256 583190d9… was checked.
- **[V]** `apply_v5_freeze.py` item 0 writes "not filled" for a `-dirty` build commit. It does not
  refuse.
- The gap is GAP-V5-BUILD-ROW-DIRTY-FROM-UNTRACKED-LEDGER-2026-10-03.

**The question (AskUserQuestion, ~23:25Z)**

> The v5 rebuild passed every pre-registered check, but its ledger row records the build commit
> as '8e6a009…-dirty'. The only 'dirty' thing was the build's own ledger file, still untracked from
> the first build's row. The tracked code was verified clean at the build's start (22:28Z) and
> again after it finished. The freeze won't fill its 'clean build commit' item from a -dirty
> string. How should it be read?

The options were "Accept as a caveat (Recommended)" and "Rebuild clean".

**The answer, verbatim: "Accept as a caveat (Recommended)"**

The freeze fills item 0 only through `--accept-dirty-build-commit <this file>`. The flag is bound to
this row: it refuses unless all of these hold:

- the row's `code_commit` is a 40-hex commit plus `-dirty`;
- that commit equals the row's `recipe.rev`;
- this file names the row's `row_id`.

The amendments record the caveat. No row is edited, and `_git_commit` is unchanged.

## 2. Re-open no-mask for v5?

**New evidence.** `tools/perf_step.py` ran at `off:6+profile`, 35,403 batch tokens, on the box at
23:16-23:24Z (`/home/ubuntu/exp-2026-10-03`). The padded training attention runs SDPA's sm80
memory-efficient kernels (`fmha_cutlassB_bf16_aligned_128x64_k65536_sm80` and its forward), not
flash:

| Shape | Widths | Attention share of CUDA time |
|---|---|---|
| W | 9.6k-wide rows | 51% (41% backward + 10% forward) |
| M | 3.0k-5.4k | 38% |

**The question (AskUserQuestion, ~23:25Z)**

> New GPU evidence: on the H100, v5's padded attention runs PyTorch's older memory-efficient
> kernels (built for A100) instead of FlashAttention. That attention is 51% of step time on the
> 9–10k-token rows and 38% on mid rows. Training without the mask (ruled off for v5; its
> equivalence screen came back inconclusive twice) would likely save more than the earlier 20–40%
> estimate. Re-open it for v5?

The options were "Keep v5 as pre-registered (Recommended)" and "Re-open no-mask now".

**The answer, verbatim: "Keep v5 as pre-registered (Recommended)"**

C2a stays off for v5. No-mask, with a better-powered equivalence screen, is v6's first lever, using
the H100 numbers.
