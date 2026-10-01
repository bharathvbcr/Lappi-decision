# HANDOFF: perf-ft-run-prep (2026-10-01)

**Lane goal.** Cut the CPU prelude that `tools/real_ft_run.py` runs on the GH200 while the GPU sits
idle.

**Branch.** `worktree-agent-a2fc031fbd6c204e8`, based on `73bcf12`. Not merged.
`git merge-tree --write-tree HEAD main` against `main` at `b5575cc` merged cleanly with no
conflicts. No tests were run on the merged tree.

## What was measured

This lane wrote **no ledger rows**: it trains and scores nothing. Every number below comes from
`AUDIT/perf-ft-run-prep-2026-10-01.md`, and the raw rounds are in
`AUDIT/perf-ft-run-prep-2026-10-01.json`.

**Setup.** Mac CPU, phase-4 v3 inputs, an interleaved A/B with the minimum of 3 runs per arm.
"Before" is `73bcf12`. Load averages were 17.6 at the start and 6.2 at the end.

| stage | before min | after min |
| --- | --- | --- |
| `--score-checkpoint --needle --ood` prelude | 152.10 s | 70.54 s |
| `--needle` worker prelude | 132.53 s | 0.77 s |
| J4 training prelude (probes stubbed) | 129.48 s | 95.71 s |

- A `--needle` scoring job runs the score and worker stages back to back. Together they took
  284.63 s before and 71.31 s after.
- In every stage, what the stage hands on hashes the same in both arms.
- The LSH port alone was measured on 50,177 keys: 1.197 s before, 0.123 s after.

## What changed

- **`101dc9b`: the needle worker and the score-mode relabel.**
  - **Needle worker.** It is still a fresh process. The reason is unchanged: MPS keeps a graph
    for each input shape and does not release it. The worker no longer rebuilds the prelude.
    Instead it reads a digest-checked `.npz` handoff of the parent's built suite
    (`write_needle_handoff` / `read_needle_handoff`), and it opens only the train and val
    readers, so the remap pairing still runs.
  - **`--score-checkpoint`.** Score mode now skips the train relabel, the inventory, the
    contradictions and the epoch plan, and each skip prints `NOT RUN -- <reason>`. The skip
    only happens when a tokenizer.json is present and every letter offered by the val and OOD
    non-span rows is a val gold. Otherwise the relabel runs.
  - **`_labels`.** Renders each row once.
  - **`_token_indices_for_chars`.** Replaces the per-position scan; the old scan is kept as the
    test oracle.
  - **Tokenizer.** Loads once per process.
- **`cd3fd0a`: `qd-prep lsh`.**
  - `crates/qd-prep/src/lsh.rs` computes `qd_data.minhash.candidate_pairs`'s pairs in the same
    order. `native_minhash` swaps it into `qd_data.dedupe` and `qd_data.split`.
  - The Python `candidate_pairs` stays in `qd_data` as the oracle. `qd_data` is fingerprinted,
    so it cannot be deleted.
- **The third commit (this file's own).** It adds the benchmark harness, the AUDIT record and
  six `GAP-` records.

`python/qd_data` is untouched: `git status --short python/qd_data` came back empty before each
commit. The code fingerprint and the v3 shard sets on the box are therefore unaffected.

## What is open

| gap | owner | what it says |
| --- | --- | --- |
| `GAP-PERF-PREP-BOX-TIMINGS-AND-CUDA-WORKER-NOT-RUN` | human | Nothing was measured on the GH200. The handoff worker was never run under CUDA. The new binary never ran on the box. |
| `GAP-PERF-PREP-PROBES-AND-LEDGER-ROWS-NOT-MEASURED` | agent | Probes were stubbed, and no score ledger row was written on the Mac. |
| `GAP-PERF-PREP-RENDER-AND-MIXTURE-ARE-IN-FINGERPRINTED-QD-DATA` | human | The next-largest sinks are inside `qd_data`: `render._escape` and `build_mixture`. |
| `GAP-PERF-PREP-TWO-QD-PREP-SUBPROCESS-RUNNERS` | agent | `_run_prep` duplicates `linear_control_native._run`. |
| `GAP-PERF-PREP-GITPULSE-UNTRUSTED-AND-NO-LISTAGENTS` | human | Coordination was done without GitPulse or ListAgents. |
| `GAP-PERF-PREP-DEVMAP-ANSWERED-FROM-THE-MAIN-CHECKOUT` | agent | DevMap answers came from the main checkout's index, not this worktree's. |

**Left on purpose.** The census detail string in `tools/real_tokenizer_pipeline.py`
(`"accepted by design -- _token_index_for_char takes the token CONTAINING ..."`) still names the
old function. That string is the detail of a census metric. I infer, without having checked the
write path, that this detail is recorded with the pipeline's metrics. Recorded output must not
change for the same inputs, so the string stays. The behaviour it describes is unchanged.

## The box needs a new `qd-prep` before any queued job runs this code

`native_minhash` now calls `qd-prep lsh` on every `ft_splits`. The box's current binary has no
`lsh` subcommand, so every job fails closed at its first rebuild.

The lane at `ac16dc1e5177b64f8` (commit `885d5fe`) changed `linwire.rs` and `linfit.rs`. The
binary cross-built from this branch does not include those changes, and theirs does not include
`lsh`. After both branches are on `main`, build once from the merge. That build is the next
lane's **first command**, run from the main checkout at the merge commit:

```
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" cargo build --release -p qd-prep --bin qd-prep --target aarch64-unknown-linux-gnu --target-dir /Users/bharath/qd-campaign/target-aarch64-linux-prepperf
```

Then copy `aarch64-unknown-linux-gnu/release/qd-prep` to the box and point `QD_PREP_BIN` at it.
The binary built from this branch alone has sha256
`da8f52dd312f5e81754e75b212bcff6cbd21d16567ac777c83657762b54eb70d`; it is an ELF aarch64 PIE,
and it has not been run on the box.
