# commitpackft: the pool the plan names, fetched and measured

**Lane:** coordinator
**Date:** 2026-09-21
**Scope:** data acquisition and measurement. No gate was moved and no threshold was set.

Claims are labelled **[V]** verified (I ran it or read the line), **[I]** inferred,
**[U]** unverified.

---

## 0. Why this file exists

`docs/build-order-2026-09-19.md:117` names `bigcode/commitpackft` as the pool.
`qd_data.sources` has registered it as LOADABLE since S3, `qd_data.loaders.parse_commitpackft`
has existed as long, `docs/plan-corrections.md` BLOCKING-1 promotes it to primary, and
`crates/qd-mutate/src/pool.rs` defines the record it would become.

**Nothing had ever fetched it.** Every corpus behind this project's 988+ ledger rows was
built from qwen-decision's own sources, which is why `rung0_real_run.py` stamps every row
`quick` — correctly, under rule 8. The plan rested on an external claim that had never been
checked.

## 1. Executive summary

| Question | Answer |
|---|---|
| Is the dataset reachable without gating? | Yes — HTTP 200 on `resolve/main/data/<lang>/data.jsonl`, `gated: False` **[V]** |
| Does the repo's own loader parse it? | Yes — 69,893 of 69,893 rows, zero failures **[V]** |
| Are the per-row licences what the card says? | **No.** Dataset is `mit`; 6,774 of 69,893 rows are not permissive **[V]** |
| Are `lang` values inside `LANGUAGE_OPTIONS`? | Yes, for these four, with matching case — 0 refusals **[V]** |
| Does `qd-mutate` consume the converted pool? | Yes — 61,193 records → 50,178 examples **[V]** |
| Are the examples hunk-constrained? | Yes — `examples_without_hunk_constraint: 0` **[V]** |
| Does the constraint actually bite? | Yes — 40,682 sites refused for falling outside a hunk **[V]** |
| Does the corpus survive the byte encoder? | 99.9% at 8192 context bytes (73 of 50,178 refused) **[V]** |
| Does this make a run promotable? | **No.** `quick=True` is unchanged; promotion is a human decision (rule 2) **[V]** |

## 2. What was fetched

Four languages, matching the tree-sitter grammars CLAUDE.md pins. 170 MB against 1,545 MB
for all 282 files. **[V]**

| language | rows | bytes | sha256 (first 16) |
|---|---|---|---|
| python | 56,025 | 135,858,935 | `d167da37e1058371` |
| typescript | 5,868 | 14,620,423 | `43d32596844c62aa` |
| go | 5,004 | 12,417,842 | `05a65e973fcc6894` |
| rust | 2,996 | 7,411,711 | `5611f3701026acf3` |

Written to `/data/pool/`, which `.gitignore` excludes. The three manifests ARE tracked —
which required fixing `.gitignore`, because `/data/pool/` excluded the manifests it claimed
to track and git cannot re-include a file whose parent directory is excluded. **[V]**

## 3. Licences are per row, not per dataset

`qd_data.sources` records the trap in as many words: the card's prose is wrong for three of
its thirteen values. Measured over the four languages: **63,119 ALLOW, 6,774 not**. **[V]**

Dropped by reason, over the pool build:

```
4,829  agpl-3.0     (NEEDS_HUMAN_CALL)
  905  lgpl-2.1     (NEEDS_HUMAN_CALL)
  846  mpl-2.0      (NEEDS_HUMAN_CALL)
   90  artistic-2.0 (NEEDS_HUMAN_CALL)
   60  unknown      (NEEDS_HUMAN_CALL)
   44  epl-1.0      (NEEDS_HUMAN_CALL)
1,926  no changed line span in new_contents
```

Nothing was admitted "because the dataset says mit". **[V]**

## 4. The conversion, and what it adds

`tools/build_commitpackft_pool.py` → 61,193 `PoolRecord`s,
sha256 `25406de13959e27c`, mean hunk length 13.3 lines. **[V]**

The hunks come from diffing `old_contents` against `new_contents`. This is the thing
`qd_data.pool_builder` structurally cannot do: it walks local trees and stamps every example
`hunk_constrained: false`, because a file on disk has no diff attached. `pool.rs` states why
it matters — *"the model learns to read hunks and not file headers."* **[V]**

`qd-mutate generate` over the full pool: **50,178 examples**, sha256 `d52467f21fefe8fe`,
41,728 mutated / 8,450 clean, `examples_without_hunk_constraint: 0`, `outside_hunk: 40,682`,
`mutation_did_not_parse: 1,175`, `span_disagreements: 13`. **[V]**

Independently confirmed: all 41,728 mutated examples have their `source_span` **overlapping**
a hunk, zero exceptions. Note the constraint is overlap, not containment — for `stub.*`
operators `source_span` is the whole function body and the hunk is a line inside it, so a
containment test wrongly reports 7,945 as unconstrained. That mistake was made and corrected
while measuring. **[V]**

## 5. Against the corpus in use

| | repo's own sources | commitpackft |
|---|---|---|
| examples | 1,915 | 50,178 |
| train decisions | 727 | 39,946 |
| val decisions | 288 | 10,159 |
| train files | 117 | 20,787 |
| refused by the encoder | 900 (47%) | 73 (0.1%) |
| hunk-constrained | no | every example |

At `--val-share 0.2`; the arms run at 0.25, which shifts the split but not the survival
rate. **[V]**

## 6. What is NOT established

* **The other ~278 language files were not fetched.** These four are the four most likely to
  match `LANGUAGE_OPTIONS`, so §1's zero-refusal result is the optimistic corner of the
  dataset. A lane widening the pool re-measures. **[U]**
* **No claim that the model does better on this corpus.** At the time of writing no arm has
  trained on it. **[U]**
* **`privileged_hunk` is not made viable by this pool.** Measured: the privileged window is
  the whole file on 48.0% of mutated examples and 76.9% of it at the median, because
  commitpackft rows are small files whose commits often touch everything.
  `GAP-PRIVILEGED-HUNK-IS-NEARLY-VACUOUS-ON-COMMITPACKFT`. **[V]**
* **Promotion is untouched.** `quick=True` remains hardcoded. What changed is that the
  *reason* is now derived from the run's facts instead of asserting a sentence that would be
  false on this corpus. **[V]**

## 7. Reproduction

```
tools/build_commitpackft_pool.py            # rows -> PoolRecord, licences filtered per row
target/release/qd-mutate generate \
  --pool data/pool/commitpackft-pool.jsonl \
  --out  data/pool/commitpackft-corpus/examples.jsonl \
  --manifest data/pool/commitpackft-corpus/manifest.json --seed 0 --limit 200000
```

The fetch itself is four `GET`s to
`https://huggingface.co/datasets/bigcode/commitpackft/resolve/main/data/<lang>/data.jsonl`.
The tracked manifests carry every digest above.
