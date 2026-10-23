# Byte-identity rebuild: priced, not run (X6, 2026-10-23)

Ruling (Fable, via lead): a one-time rebuild under fingerprint rule 3, compared byte for byte
against the pinned train `94bef49a...` and val `58bd753f...`; identical means the two uncovered
modules did not shape the bytes. Queue it only if it fits under mac_heavy's cap (28,672 MiB) and
about 2 h of lock.

## Price (recorded numbers only)

| item | value | source |
| --- | --- | --- |
| pipeline stage (`tools/real_tokenizer_pipeline.py`, v6 default) | 7299.4 s = 2.03 h | ledger row 6bf4abfc `wall_clock_s`, `ledger/mac-v6-lora2b-shards-2026-10-18.jsonl` (run_kind smoke) |
| containment scan stage | about 1.1 ks measured sub-steps (minhash 136 s, lsh 256 s, split 611 s, containment 65 s) plus corpus load | `build/w2d-scratch-logs/logs/scan.log` |
| whole W2-d chain, start to end | 2026-10-19T21:44:11Z to 2026-10-20T00:20:14Z = 2 h 36 min | `build/w2d-scratch-logs/logs/chain.log`, includes cargo wait 326 s and the examples export and qd-train-data stages |
| peak footprint | NOT RECORDED | the row carries no footprint metric; the W2-d chain ran before the 28,672 MiB cap was set from memguard |

A rebuild needs the scan (its `exclusions_sha256` 37b84644... is part of the pinned header and the
scan output directory was in the removed W2-d worktree) and the pipeline: about 2.4 to 2.6 h of
lock, over the "about 2 h" bound, and the footprint cannot be shown to fit under 28,672 MiB because
it was never measured. Inputs that still exist: `build/pools/pool-v6-decisions-v1`,
`~/.cache/qd-decision/general/fetch-record-2026-09-29.json`, `data/pool/*`,
`build/w2d-scratch-logs/targets/`. The chain script is `build/w2d-scratch-logs/chain.sh`.

## Verdict

Not queued. The pinned v2 sets stay report-only (enforced: `promotion_verdict` refuses a row whose
`shard.code_not_covered` metric is non-empty unless `AUDIT/x6-fingerprint-list-2026-10-23/byte_identity.json`
holds an `identical: true` record citing its evidence for that set's `shard_hash`; no such record
exists). To lift it: run the chain into a NEW directory with a measured footprint under a lock hold of
about 2.6 h, compare `tokens.u32`, `offsets.npy`, `supervision.npz`, `sequence_index.json` and the row
identities byte for byte against `build/shards-v6-lora2b-6bf4abfc/{train,val}`, and add the record.
A byte difference means the other branch of the ruling: rebuild and amend every pre-registration that
pins the set.
