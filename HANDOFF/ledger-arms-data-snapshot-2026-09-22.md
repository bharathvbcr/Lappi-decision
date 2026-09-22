# Handoff — ledger_arms data-snapshot lane, 2026-09-22

## One paragraph

`tools/ledger_arms.py` keyed an arm on `(recipe_hash, backbone_commit)`, plus the code under
`--split-by-code`. Every row's `protocol` also names `data_snapshot_hash` and `tokenizer_hash`,
so two runs over two data snapshots under one recipe and backbone were one arm. The
repeated-seed refusal from `dd59c68` caught `mac-rung0-prereg` only because both of its runs used
seeds 0 and 1. With disjoint seeds the two corpora would have pooled into one population without
a word. An arm is now keyed on the protocol minus its seed, the set the writer's
`Protocol.hash_without_seed` already defines as a seed family. On every ledger a handoff or audit
reads with this tool, the output is byte-identical before and after the change. The brief expected
that of every ledger except prereg. One more changes: `gh200-2026-09-21`, whose five unmeasured
smoke rows read two snapshots.

## The decision

**Identity.** An arm is `(recipe_hash, backbone_commit, data_snapshot_hash, tokenizer_hash)` plus
the code under `--split-by-code`. That is `qd_train.ledger.Protocol` minus `seed`, which
`hash_without_seed` calls "the thing three rows must share to promote". A population of seeds is
exactly that thing. `ArmKey` is a `NamedTuple` of those fields.
`test_the_arm_key_is_the_writers_protocol_minus_its_seed` holds them equal to the writer's
dataclass fields minus `seed`, so a component added to the protocol fails a test instead of
pooling silently. All 1437 rows in the 29 committed ledgers at `64052f4` carry exactly the five
components, so no row is missing one.

**Label: named only where two arms would otherwise share a name.** "Always named" was rejected.
Every committed sweep except the four-row prereg check reads one snapshot and one tokenizer, so
always naming them would lengthen every label of every report without telling any two arms apart.
It would also change all six documented outputs. Naming them only where needed has one hole: two
arms already told apart by backbone or recipe can read different data, and their comparison would
read as a backbone difference. The committed sweeps read five snapshots:

| snapshot | ledgers (measured rows) |
| --- | --- |
| `22f39f9d10` | rung-0 capacity e10/e30/sw005/sw005-nondet, learning-curve, curve-n24, repro, reprodet |
| `728eee3743` | diff-capacity, fourway, lr-sweep, operator-holdout-model, span-in-diff, tuned-capacity |
| `9689345b34` | commitpackft |
| `aabf37d5b4` | context-source |
| `c2e856e15a`, `6267ab7c85` | mac-rung0-prereg (one run each) |

So an arm-vs-arm line that crosses a snapshot or tokenizer now says so on the line
(`ACROSS DATA SNAPSHOTS a / b: the difference includes whatever the data changed`). Reading the
e30 and span-in-diff ledgers together prints `+41.46pp VISIBLE` for a span-in-diff 256x4 arm minus
an e30 128x4 arm. That line now also says it crosses `728eee3743 / 22f39f9d10`. The tokenizer is
named in full, never cut to a prefix. `bytes-utf8-256` and `bytes-utf8-512` would both print as
`bytes-utf8` at ten characters, the "X minus X" bug the label exists to prevent. The test pins this
exact pair.

## What was measured

Every figure is from the 29 ledgers as committed at `64052f4`, frozen from `git show` so rows
appended by a concurrent lane could not leak into the comparison. That happened once: the first
unfrozen run showed `mac-shards-commitpackft` growing from 4 to 5 rows between the two captures.

**Groups the wider key splits:**

| ledger | group | rows |
| --- | --- | --- |
| `mac-rung0-prereg` | `rung0-scratch:64x2:1layer:ctx2048` recipe `af4704fd40`, code `57affc016a0f10d1` | `9c15fcdd`, `86803442` (seeds 0–1, `c2e856e15a`); `56a0051d`, `694871a2` (seeds 0–1, `6267ab7c85`). All four collapsed at 48.86% against a 48.3% baseline |
| `gh200-2026-09-21` | backbone `b1485b2fa6…` recipe `5c08c0aa4f`, unmeasured smoke rows, seed 20260919 | `ba75bf9e`, `a17c5bdd`, `90a362cc`, `27756134` (`fe79253cd2`); `46e63ff3` (`101168a900`) |

No other `(recipe, backbone[, code])` group in any committed ledger spans two snapshots or two
tokenizers, counting all rows and not only measured ones. The brief's scan counted measured rows
only, which is why it missed the smoke group.

**Before/after, HEAD's tool against this one, on the frozen ledgers.** 87 runs: `ledger_arms` with
and without `--split-by-code`, plus `operator_holdout_report`, over all 29 ledgers. The harness
first ran HEAD against the unmodified tree and got 87/87 identical, so it is deterministic and
loads HEAD's module correctly.

* **79 of 87 runs byte-identical** in exit code, stdout and stderr. That includes both reads of all
  six documented ledgers: capacity-4096-e10, capacity-4096-e30, capacity-sw005-nondet,
  learning-curve, curve-n24 and span-in-diff. It also covers every other measured sweep.
* `operator_holdout_report`: 29 of 29 identical. It imports `SeedClaims`, which this change does not
  touch.
* The 8 that differ:
  * `mac-rung0-prereg`, both modes: exit 2 → 0, two arms named `data 6267ab7c85` and
    `data c2e856e15a`.
  * `gh200-2026-09-21`, both modes: 38 → 39 arms, the smoke group above split and named by
    snapshot.
  * `gh200-rung0-repro` and `-reprodet`, both modes: still refused. The refusal drops "Where the
    snapshots differ they are two arms this key cannot tell apart", which describes a case that
    can no longer happen, and says the rows share recipe, backbone, snapshot and tokenizer.

**Residual scans with the changed tool:**

* Pairwise lines comparing arms with different `code_that_ran` digests, which nothing on the line
  names without `--split-by-code`. In `curve-n24`, 7 of 10. Its three curve points share
  `fdec48ce5b4bd33d`, so the curve itself does not cross code. Recipes `991b57587f` and
  `d6248d9dd9` ran `298481c04273352b` and `20f9156afe0045d0`. In `operator-holdout-model` read by
  this tool, 81 of 153. Recorded, not fixed: flagging these changes curve-n24's documented output.
* Zero-sd samples printing a 0.00pp floor: only the two new prereg arms, in both modes, out of 51
  reads (7 refused). The same display appeared before `dd59c68`, when these rows were pooled.

## What changed

| commit | what |
| --- | --- |
| `c448560` | `tools/ledger_arms.py`: `ArmKey` NamedTuple keyed on the protocol minus seed, labels naming snapshot or tokenizer where needed, `_crossed` notes on pairwise lines, refusal wording, docstrings. `python/tests/test_ledger_arms.py`: 7 new items, 2 wording items and 6 documented-ledger items changed. |
| the commit adding this file | `gaps.jsonl`: 5 records appended with `qd_train.gaps.append_gap`. This handoff. |

Tests against the pre-fix tool, with `tools/ledger_arms.py` confirmed identical to HEAD: 14 of the
46 items fail.

* 6 of the 7 new items fail. The two-snapshot and two-tokenizer cases read as one arm,
  `[[0.5, 0.52, 0.4, 0.42]]`. The cross-snapshot line has no note. Prereg is refused. The key has
  no fields.
* The seventh passes on purpose. It guards against naming a snapshot where it tells nothing apart.
* The 2 wording items fail.
* The 6 documented-ledger items fail only structurally, because the old key has no field names.

After the change:

* `make pytest`: 1771 passed, 78 skipped, exit 0.
* `make torch-pytest`: 2172 passed, 14 skipped, exit 0.
* `make lint`: all checks passed.
* `test_ledger_arms.py` and `test_operator_holdout_report.py`: 72 passed, 0 skipped.
* `test_gaps_ledger.py` and `test_gaps_writer.py`, after the appends: 29 passed.

The three make gates ran on the shared working tree, which also held another lane's uncommitted
pipeline edits.

## What is open

| gap | state |
| --- | --- |
| `GAP-LEDGER-ARMS-KEY-OMITS-THE-DATA-SNAPSHOT` | resolved-with-residual, `c448560` |
| `GAP-OPERATOR-HOLDOUT-CELLS-OMIT-THE-DATA-SNAPSHOT` | **open**. Same omission in `cells_of`, latent because its ledger reads one snapshot. Left alone so its output stays byte-identical for the lane that runs it. |
| `GAP-LEDGER-ARMS-PAIRWISE-SILENT-WHEN-ARMS-RAN-DIFFERENT-CODE` | **open**. curve-n24: 7 of 10 lines cross code unnamed. Extending `_crossed` is small, but it changes a documented output, so it needs a decision. |
| `GAP-LEDGER-ARMS-ZERO-SD-PRINTS-A-ZERO-FLOOR` | **open**, minor. Only prereg's collapsed arms; nothing reads them. |
| `GAP-DEVMAP-NODES-FTS-CORRUPT-WHILE-STATUS-SAYS-ONLY-STALE` | **open, human**. See below. |

Coordination notes for whoever runs next in this worktree:

* **DevMap. The store is healthy; one kind of reader was not.** Revised after a peer reported
  search working again. From about 17:10 −0500, this session's long-lived `devmap mcp` (pid 33291)
  failed every name lookup:
  * `devmap_search` returned "fts5: corruption found reading blob 412316860426 from table
    nodes_fts".
  * `devmap_explore` returned "database disk image is malformed".
  * `devmap_status` reported the index only as stale, at generation 2384.

  `lsof +L1` shows that process holding `devmap.sqlite-wal` and `-shm` files deleted from disk
  while open. A fresh read-only connection sees generation 2433: the FTS query answers and
  `quick_check` is ok. From this same session, the GitPulse plugin's `devmap_*` tools read 2433
  and answer. So the first version of this note ("fails for every session here") was an
  overreach from one session.

  The other long-lived servers on this store, pids 36598 and 83988, hold deleted copies too. Their
  failure is unchecked. If your `devmap_status` shows a generation below the plugin's, restart
  your session's devmap server or use the plugin's tools. What deletes the files under live
  connections is DevMap's to find (DevCouncil `rust/devmap-store`). This lane read
  `python/qd_train/gaps.py` directly while its server was failing, so nothing it says about that
  file is graph-confirmed.
* **GitPulse** reported 0 agent sessions while `ListAgents` showed five live local sessions in this
  worktree, the known same-worktree blind spot (CLAUDE.md rule 3). File claims were agreed by
  `SendMessage`. The repeated-seed lane confirmed it holds none of this lane's files. HEAD moved 5
  times during the lane (another lane's pipeline commits, `5d678b6`…`64052f4`). None touched
  `tools/ledger_arms.py`, its test or the operator report. `e7b4ee8` appended 3 committed lines to
  `gaps.jsonl`, and this lane's 5 lines follow them.
* `HANDOFF/ledger-arms-repeated-seed-2026-09-22.md` gives the prereg refusal as the next lane's
  first command. That command now exits 0 and prints two arms, so this file supersedes that line.

## The exact first command for the next lane

The one documented ledger whose arm-vs-arm lines cross code versions unnamed. Read it before
deciding `GAP-LEDGER-ARMS-PAIRWISE-SILENT-WHEN-ARMS-RAN-DIFFERENT-CODE`:

```bash
/Users/bharath/.venvs/ml/bin/python /Users/bharath/Code/research/qwen-decision/tools/ledger_arms.py /Users/bharath/Code/research/qwen-decision/ledger/gh200-rung0-curve-n24-2026-09-22.jsonl
```
