# Digest A/B row 3 (sha2 0.11 build): reading declared BEFORE the run (2026-10-03, qdm1b)

Written before row 3 exists. Rows 1 (147da0cc) and 2 (28505f4c) are append-only and untouched.

- Command: identical to rows 1 and 2 (`qd-metal-bench --decision T=131,409,770,2048,8192 k=4
  --arms digest=serial,digest=parallel --snapshot <p4-v4-avg-masters-v1> --ledger
  <worktree>/ledger/mac-qd-metal-2026-10-03.jsonl`), on the sha2 0.11.0 build (lock 44735d5c).
- Statistic: min-of-N (house convention); medians report-only.
- Bound (Fable ruling 2, build order step 7): at task lengths T = 131, 409, 770, the parallel
  arm's min digest_ms in row 3 <= 1/3 of row 2's parallel min digest_ms (row 2: 19.47, 26.11,
  34.01 ms, so <= 6.49, 8.70, 11.34 ms). T = 2048, 8192 report-only.
- Also required: arms bit-identical at every T; t_coverage 5/5; release weight_hash still
  a68f19bcdf155f39137d1321b0e29f5a41dac0f7b0b81d2e06b05373f68f93c2 (the weights are hashed with
  sha2 at load, so a different hash means 0.11 hashes differently).
- If the bound misses at any task-length T, or any identity check fails: report, change nothing.
row 3 start: 04:21:24Z loadavg { 5.18 4.94 5.31 }
