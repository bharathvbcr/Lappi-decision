# The human's answers to Fable's three v5 review questions (2026-10-02)

Asked by the lead through AskUserQuestion at ~07:45 UTC 2026-10-02, on Fable's section 4
(`AUDIT/v5-plan-2026-10-02/fable-v5-plan-review.md`). Recorded verbatim. Each question and the
option the human chose:

1. **v5 launch.** Asked: "Approve the v5 launch? About $137 for about 60 GPU-hours: 3 seeds,
   J5′ on 3 seeds, and the abstention-weight arm on 3 seeds. Add $32 if the long-context rule
   adds seeds 3–4; it drops to $95–116 if no-mask training passes its tests. Earliest start is
   about Oct 4, 21:30 UTC, ending Oct 7–8. It includes the seed-0 pause rule and skips the
   abstention-weight arm if v5 already hits its targets."
   Chosen: **"Yes, launch when ready (Recommended)"**. The option's text: "The v5 queue
   script is appended after j6g and starts when the box frees up. Each run carries its cost cap
   and your approval. If the data build isn't ready at that point, the box idles at $2.29/h."

2. **G6 unseen-language abstention.** Asked whether to keep v4's 834 template rows, or to let the
   lead fetch only the file list (names and sizes, a few KB) of `bigcode/commitpackft`, so the
   lead can bring a specific download to approve.
   Chosen: **"Fetch the file list (Recommended)"**. The option's text: "Metadata only, no data
   download. I come back with names and sizes for languages none of the test suites use (Perl,
   OCaml, Julia, R, Erlang, Clojure, Fortran, Tcl), aiming for 1,000+ rows across 5+ languages,
   licence-filtered per row."
   This approves only the metadata listing. A data download needs its own yes, with sizes.

3. **Own-repo prose.** Asked which of the human's repos may supply prose for the abstention
   examples, with the filters (forks, vendored code, other people's licensed text) applying
   either way.
   Chosen: **"All but Lappi-decision (Recommended)"**. The option's text: "All 29 non-fork
   repos except this one, whose docs describe the test suites. Up to 150 paragraphs per repo."

## Notes on the repo list (the lead's, not part of the answers)

- The inventory (`AUDIT/v5-plan-2026-10-02/own_repo_inventory.json`) classes 31 repos as
  `own`.
- `devtools/DevPrism` has a second remote, `delibae/claude-prism`. Under Fable's admission rule
  (§2.9 (1): a repo with a remote naming another owner is a fork), it is a fork and is excluded.
  Excluding it and Lappi-decision leaves the 29 repos the option named.
- `/Users/bharath/Code/research/Lappi-decision` is excluded by the answer.
- Fable's §2.9 filters (vendored trees, foreign LICENSE/SPDX/copyright lines, model cards,
  byte-identical copies of "other" repos' files) and the 150-paragraph per-repo cap apply to
  every admitted repo.
