# W3-b proposals (2026-10-17): written, not applied

Nothing here moves a gate, a threshold or a population (rule 2). Each item needs the human.

## P1. The pinned v5 decision pool fails the v6 length band (M24)

`pool-v5-decisions-v4`, which `data/decisions/v6-allocation.json` passes through "as built", has
P(gold is the longer response) = 0.572 on train (6,851 ordered unequal-length rows; GAP-W3B-V5-
DECISION-POOL-FAILS-THE-V6-LENGTH-BAND-2026-10-17). `DataConfig()` now carries the band
[0.45, 0.55] from the data-clean plan (item 9), and `load_decision_pool(config=...)` refuses it.
Options:

1. Rebuild the pairwise rows with `qd-prep decisions` under a `pairwise_length_balance` config
   (`{"low": 0.45, "high": 0.55}`), then re-pin the pool in `v6-allocation.json`. Recommended: it
   is what the plan says v6 does. It changes the pairwise row count, so the allocation table's
   `pairwise.helpfulness` figure must be re-read.
2. Keep the pool and rule the band off for it. That reopens M24 (the 37-of-55 length prior) and
   is the human's call.

## P2. Ledger rows do not carry `val_without_train` (M25)

Manifests and the pooled gate the producer writes now name a family with val rows and no train
rows. `ledger.py`'s population reading cannot, because a ledger row has no such field.
Proposal: add an optional `val_without_train` mapping to `LedgerRow` (a schema change, so it
belongs to the lane that owns `ledger.py` and its readers). Excluding such a family from the
gate's pool is already what `paired_margin_population` (human-ruled 2026-10-08) does for any
family outside the promotion population; no further exclusion is proposed.
