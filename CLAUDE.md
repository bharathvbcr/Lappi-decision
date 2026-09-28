# qwen-decision — binding instructions for every agent in this repo

This repo implements the **qwen-decision Build Plan** (2026-09-19): one Qwen3.5-2B-Base-derived
**typed decision model** that any app calls through a schema, with DevCouncil and DevType as the
first two callers. **Lappi** is the product name (see `docs/lappi.md`). Hosted on GitHub at
[bharathvbcr/Lappi-decision](https://github.com/bharathvbcr/Lappi-decision).

These rules override default behaviour. They are copied into every lane prompt.

## Navigation before reading

1. **DevMap first.** `devmap_search` / `devmap_explore` for where a symbol lives, `devmap_neighbors`
   and `devmap_dependencies` for edges, `devmap_impact` for blast radius, `devmap_trace` for how two
   symbols connect, `devmap_affected_tests` for what to run, `devmap_dead_symbols` before deleting,
   `devmap_clones` before adding a near-copy. Pass the absolute `repo_path` on every call and check
   `repository.root` in the envelope. Read `truncated` and `walk_incomplete` before treating an empty
   list as "does not exist".
2. **GitPulse Insights before changing.** `gitpulse_insights` names the other **worktrees**,
   their uncommitted work and the files contended **between worktrees**. Its facets fail
   independently — check each `ok`, because a facet that could not scan is **not** a facet that
   came back clean.
   Note: `/Users/bharath/Code` is not a git repository; pass a real repo path.
3. **`ListAgents` before changing, too — GitPulse does not cover this half.** The collisions
   facet compares worktrees, so two agents editing one file in the **same** worktree read as
   `overlapping_files=0`: a clean scan that is wrong about the risk. Nine gap records said this
   and it happened on 2026-09-21 — two sessions in this worktree, one of them about to patch
   `tools/real_ft_run.py` while the other was rewriting it. What found the other session was
   `ListAgents`; what prevented the clobber was `SendMessage` to agree who owned which files.
   So: `ListAgents` before touching a file, and if a peer is live here, message it and split the
   tree by file before either of you writes. Treat a peer's claim on a file as binding, and
   never `git stash` or `git add -A` in a shared worktree — both move the other lane's work.
4. When either tool cannot answer, record the gap in `gaps.jsonl` with a `GAP-` id and say so.
   **Never report a grep result as graph-confirmed.** Append with `O_APPEND` + `fsync` and a
   single line — never read-modify-write, which silently drops a concurrent lane's record.

## Environment facts (verified 2026-09-19, re-verify before relying on them)

- `cargo` / `rustc` 1.98.0, `uv` 0.11.28, `python3` 3.14.7, `git` 2.54.0.
- PyTorch env: `/Users/bharath/.venvs/ml` — torch 2.12.1, **MPS available**.
- **Containers are podman, not docker.** Docker is not installed; `podman` 6.1.2 is, with an
  `applehv` machine (`podman-machine-default`, 9 CPUs, **12 GiB RAM**, **160 GiB disk** — both
  raised from the 2 GiB / 100 GiB defaults during S1, forced by an OOM kill and an ENOSPC
  respectively; see `stack/README.md`). Plan item S1
  says "buildx for x86" — the podman equivalent is `podman build --platform linux/amd64` plus
  `podman manifest` for the multi-arch manifest. Two things to establish before S1 is called green:
  cross-arch (aarch64 host -> x86_64 image) emulation inside the machine, and whether 2 GiB of VM
  RAM is enough to build a torch+CUDA image — it very likely is not, and `podman machine set
  --memory` is the fix. Never substitute docker; this host does not have it.
- tree-sitter grammar versions matching `devmap-extract`: `tree-sitter` 0.25,
  `tree-sitter-{rust,go,python,typescript}` 0.23, `tree-sitter-swift` 0.7.
- Harness constraints when shelling out: `cd` is blocked, redirects to `/dev/null` are blocked, and a
  redirect target containing a shell variable is blocked. Use literal absolute paths.

## The ten rules that bind every agent

1. **DevMap and GitPulse first** (above). Record what they could not answer in `gaps.jsonl` with a
   `GAP-` id; never report a grep result as graph-confirmed.
2. **Gates and kill criteria are read-only.** An agent may report that a gate failed; it may not move
   a threshold, drop a seed, shrink a held-out set, or reclassify a run as `quick` to pass it.
3. **Held-out data and the two task-holdout families are never read by a training process.** A path
   check in `qd-train` refuses them; removing that check is a refused change.
4. **Every 8xH100 job carries a wall-clock cap, auto-terminate and a cost estimate, and needs a
   human yes** before launch. Single-GPU jobs under $20 do not.
5. **Nothing is green that was not run.** GPU suites that cannot run in the agent's environment are
   reported as **not run**. A number in a report cites a ledger row or is not in the report.
6. **Nested tessl (`MLSystemsLab/Rust_MLKit/crates/tessl`) is not touched.** All kernel work is on the
   canonical crate at `~/Code/research/tessl`; gemma-metal is repointed by a human later.
7. **Product wiring never emits PASS, `allow`, or discharges a requirement.** Admission only;
   dcverify and the GitPulse hook contract stand.
8. **`quick` runs cannot promote anything.** Fewer than 3 seeds, a truncated schedule or a subsample
   is marked `quick` in the ledger and excluded from decisions.
9. **Parity targets name the rule.** Any GDN fixture or test that does not say `published` is invalid.
   nanolab's *default* is `rule="repo"`, which is **not** the published operator — a fixture generated
   at the default is a silently wrong golden.
10. **Handoff is a file, not a chat.** `HANDOFF/<lane>-<date>.md`: what was measured (ledger row ids),
    what changed (commits), what is open (gap ids), and the exact first command for the next lane.

## Engineering rules

- **No placeholders in delivered code**: no `TODO`, no stub returning a fake value, no commented-out
  alternative. Ship it complete or say what is blocked.
- **A check that could not run must never report the same result as a check that ran and passed.**
  That is how "approved" comes to mean "unexamined". Carry both numbers; never present a capped
  sample as complete coverage.
- **Every fix ships with a test that fails against the pre-fix code.**
- **Fail closed and loud.** Bound every timeout, retry, fan-out, batch and payload.
- **No new dependency without asking.**
- Typed languages: no `any`, no unchecked casts, no wildcard imports, no silently discarded errors.

## Layout

| Path | Holds |
| --- | --- |
| `crates/qd-mutate/` | Rust + tree-sitter mutation engine; labels by construction, with span labels |
| `crates/qd-runtime/` | Rust: graph, schema API, `qd serve` / `qd oneshot` |
| `python/qd_data/` | pool filters, prompt format, splits, dedupe |
| `python/qd_train/` | CPT, FT, eval harness, ledger |
| `ledger/` | JSONL, append-only. A run that did not write its row is rerun, not remembered |
| `HANDOFF/` | one file per lane completion |
| `AUDIT/` | evidence for every external claim the plan rests on |
| `gaps.jsonl` | `GAP-` records: what DevMap/GitPulse could not answer |

Kernel work (K1-K7, `tests/gdn.rs`, fixtures) lands in `~/Code/research/tessl`, not here.
