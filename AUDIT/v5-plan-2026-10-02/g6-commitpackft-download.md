# G6 unseen-language source: eight bigcode/commitpackft files (2026-10-02)

## Approvals

The human gave two answers through AskUserQuestion on 2026-10-02:

1. "Fetch the file list (Recommended)". This covered metadata only (`human-answers-v5-review.md`).
2. "Download all 8 (Recommended)", given after the lead showed the file list. The question was
   recorded verbatim:

   > Download these 8 bigcode/commitpackft files (13.2 MB in total, from huggingface.co) for
   > unseen-language abstention? Sizes: clojure 5.2 MB (2,403 rows), perl 5.1 MB (2,288), erlang
   > 1.2 MB (480), ocaml 0.7 MB (333), julia 0.3 MB (180), tcl 0.3 MB (103), r 0.2 MB (121),
   > fortran 0.1 MB (70). None of these languages appears in training, the OOD suite, the needle
   > test or v3b. Each row keeps its own licence field, and only permissive licences pass the
   > corpus-v3 filter.

   The chosen option said: "13.2 MB into the Mac data cache, each file sha256-pinned, used only by
   the v5 build. 5,978 rows before filtering; the v5 data lane keeps at least 1,000 across 5 or
   more languages."

## The metadata that was read

- `https://huggingface.co/api/datasets/bigcode/commitpackft/tree/main/data` (27,854 bytes, 277
  language directories), then `.../tree/main/data/<lang>` for each of the eight languages.
- The dataset card, `https://huggingface.co/datasets/bigcode/commitpackft/raw/main/README.md`
  (17,288 bytes). It gives `license: mit` for the dataset and per-row `license` fields drawn from
  13 permissive-or-copyleft values. Per-language row counts are at its lines 145-228.
- The `datasets-server` size API answered "busier than usual" twice, so its counts were not used.

## The download

Downloaded on 2026-10-02 at about 12:25 UTC from
`https://huggingface.co/datasets/bigcode/commitpackft/resolve/main/data/<lang>/data.jsonl` to
`/Users/bharath/qd-campaign/commitpackft-g6-2026-10-02/<lang>/data.jsonl`, outside the repo. For
every file, the byte size and the sha256 equal the LFS size and oid that the tree API lists,
which the lead checked:

| Language | Bytes | sha256 | Rows |
| --- | ---: | --- | ---: |
| perl | 5,102,583 | `7be9d71232ee596adc28f31b3c9d165a2ed512adfa1565cfc5a2eb6915259909` | 2,288 |
| clojure | 5,184,210 | `6e0c98f3eca6613f68594636625e683c6384ecda9e9c308d92c6ee2fd87c82c7` | 2,403 |
| erlang | 1,214,008 | `ceaf80113a42769039f297c3041da7f1b42e055f224a93049d53cfa0894cf321` | 480 |
| ocaml | 715,507 | `85dc1a33118dd772ba26ee9c3ab545498b63cc7fbfe4a78934227ab705d47fa7` | 333 |
| julia | 311,477 | `3e6a68ba343c09cf56018b2d5d91a3bddf6e3833ac37600a063216c77f5e76c3` | 180 |
| tcl | 291,471 | `9423b17ba400ba6409779c3570718a38dad52872dc006ee2e5e4e3bc9c5b6fec` | 103 |
| r | 228,938 | `44e911a7983da6ef3fb9ef71302abe55f8fc15430884026568fcbbb5c088b8e4` | 121 |
| fortran | 141,056 | `1937e2c05ec0e8f22fea7a3f3734f22034b7199a0fe0bd466dd06ce887879c47` | 70 |

The rows total 5,978, which equals the dataset card's counts.

Per-row `license` counts, read by the lead:
- **perl:** artistic-2.0 799, mit 515, apache-2.0 366, agpl-3.0 212, bsd-2-clause 116,
  bsd-3-clause 112, isc 93, lgpl-2.1 39, unlicense 14, cc0-1.0 13, mpl-2.0 6, unknown 2,
  epl-1.0 1.
- **clojure:** epl-1.0 1,048, mit 710, apache-2.0 283, unlicense 83, agpl-3.0 79, mpl-2.0 66,
  bsd-2-clause 62, bsd-3-clause 54, isc 8, cc0-1.0 6, lgpl-2.1 3, unknown 1.
- **erlang:** apache-2.0 185, mit 183, bsd-3-clause 58, bsd-2-clause 25, isc 14, epl-1.0 5,
  agpl-3.0 4, unlicense 4, mpl-2.0 2.
- **ocaml:** mit 132, lgpl-2.1 67, isc 50, apache-2.0 34, bsd-3-clause 22, unlicense 13,
  bsd-2-clause 12, agpl-3.0 3.
- **julia:** mit 142, apache-2.0 9, bsd-2-clause 9, lgpl-2.1 7, cc0-1.0 5, agpl-3.0 3,
  bsd-3-clause 3, unlicense 2.
- **r:** mit 67, apache-2.0 42, bsd-3-clause 5, cc0-1.0 3, agpl-3.0 2, mpl-2.0 2.
- **tcl:** bsd-3-clause 45, apache-2.0 25, mit 22, isc 4, bsd-2-clause 3, lgpl-2.1 2,
  agpl-3.0 1, epl-1.0 1.
- **fortran:** bsd-2-clause 26, mit 15, bsd-3-clause 14, apache-2.0 8, lgpl-2.1 7.

## Use

The v5 data lane uses these files only for the `ood_abstain.unseen-language` noul source (G6).
The corpus-v3 per-row licence filter decides which licences pass; the counts above are taken
before it. No training process reads them until the v5 build pins them by these hashes.
