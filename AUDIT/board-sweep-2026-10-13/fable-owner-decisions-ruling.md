# Fable ruling: remaining owner decisions, Lappi-decision (2026-10-13)

The user handed these decisions over with "choose best". Fable ruled on each one, read-only, using direct reads, `git fetch`/`rev-list`/`diff`, `gh repo list` and the GitPulse board. DevMap is stale (gen 3888, `is_fresh=false`), so none of this is graph-confirmed.

Evidence labels: [V] read or run during the ruling, [I] inferred, [U] unverified.

| # | Card | Decision | Trigger / condition | Who runs it |
|---|---|---|---|---|
| 1 | ft-be802e07 push main | DO NOW, after the post-merge `make gates` on 9dddbfa | The delta is safe to publish. Delete `origin/outreach-2026-10-06` only after the HF card's one link is retargeted to `main`. | the session owning main |
| 2 | ft-bd80c52e fund v6 | DO LATER: Arm B only; Arm A declined | (a) the Arm B Mac probe and full mining scan are done ($0); (b) `campaign/v6-preregistered.json` is written; (c) P1's verdict is read; (d) the human says yes to about $175, ceiling $200, on 2×H100 | the human gives the yes; a lane does (a) and (b) |
| 3 | ft-b4f737de caller contract | (1) DECLINE admitting app records; (2) committing the four repos is already done; (3) live check: DO LATER | (3) waits until a v6 export lists a caller family in `trained_families` | a lane writes the decision file |
| 4 | ft-1b5f5f56 public numbers | DO NOW; mostly landed (fdbd81e, `README.md:22-23`, the Hub card) | Add one caveat: overlap between decider-2b's training data and the benchmarks is unverified | a doc lane, then the push |
| 5 | ft-12da8e67 first box session | DO NOW, gated on two human confirmations | The human confirms "≈$9.2, ceiling $12, cap 18,000 s, one GH200" and "download Qwen3.5-4B-Base (~8.4 GB)". The human creates and terminates the instance. | the human launches; a lane fills the queue |
| 6 | ft-3163ce items 4, 9 (qd-train→ojas-train) | DO LATER | After the push (#1) and the reformat (#8). The human reads a delta report before the dependency moves. Lappi depends on ojas by path, with no pinned sha; record that as a GAP. | an ojas lane and a Lappi lane |
| 7 | ft-a37989f8 decode_slots digest pin | DO NOW, as the next solo `mac_heavy` job | Prefill revert 7640c0a is already on main. Run `qd-metal-parity` at tessl 2972027. Re-pin `PINNED_V1` and `PIN_TESSL_HEAD`, citing the ledger row, only if parity passes; otherwise file a tessl kernel regression. | a lane, under `tools/mac_heavy.sh` |
| 8 | ft-f137c48e rustfmt + fmt gate | DO NOW, right after the push | One reformat commit (2,041 hunks over the 9 members; never `--all`), a `fmt` gate in `make gates`, and `.git-blame-ignore-revs`. The clippy half is already done (ca3adae; `-D warnings` is in gates). | the session owning main |
| 9 | ft-afb31d47 linfit UnsafeCell | DECLINE the patch; profile DO LATER | `crates/qd-prep/src/team.rs:23` keeps the crate free of `unsafe`. The gain is under $1 per arm. Revisit only if a profile attributes the residual to the atomics and the control sits on a paid critical path. | nobody now |

Order of work on main: gates → push (#1) → reformat + fmt gate (#8) → push → the #6 and #7 lanes branch from that tip.

## Push delta (#1) [V]

- **Size.** `origin/main` = f7cb935. `rev-list --count origin/main..main` = 164, so the card's "23" is stale. 287 files, +88,835 / −942.
- **Credentials.** The credential-pattern scan found only test fixtures: `crates/qd-prep/tests/caller_families.rs:552-553` and `crates/qd-runtime/tests/caller_records.rs:203-272`. One of them is AWS's documented example key.
- **Emails.** Only `@example.com`.
- **Home paths.** 97 new `/Users/bharath/` lines. 470 files on origin already carry that prefix.
- **Hosts.** The Lambda box IP is already public. `Mac-4557.lan` ×4 in `ledger/arm-b-shards-2026-10-13.jsonl` is new: it is the Mac's hostname, and not a credential.
- **Private repos.** `data/pool/own-repos-2026-10-06.manifest.json` adds the HEAD shas of 4 private repos. Their names are already public on origin. No private file contents are tracked. The manifest is sha-pinned by the pre-registration, so it cannot be redacted without a re-pin.

Ruling: safe to publish. The Mac hostname and the private-repo HEAD shas become public; neither is a secret.
