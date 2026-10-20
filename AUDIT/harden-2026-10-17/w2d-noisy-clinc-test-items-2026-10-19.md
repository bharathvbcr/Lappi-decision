# Two CLINC test utterances that also sit in CLINC's upstream train split under a different label

Recorded 2026-10-19 by lane W2-d, report only. Source: `w2d-clinc-train-test-duplicates-2026-10-19.txt`
(this directory), read from the sha-checked general-record cache
(`clinc__clinc_oos/155b9c710419136e17307b80d0a13e68cd46b4ec/{train,test}.jsonl`).

The test set is never edited and no item is excluded from scoring (rule 3; Fable Q23 (b); owner
ruling `human-ruling-2026-10-19.md`).

| Utterance | blake2b-8 of the stripped text | upstream test intent | upstream train intent |
|---|---|---|---|
| what's your designation | `9aa7f15e83aa5557` | 11 | 75 |
| where did you grow up | `d0eaefc2c46f9e68` | 77 | 99 |

These are the only two byte-identical train/test duplicates of the 5,500 test utterances
(digest, un-normalised). Each carries a different intent label in the two splits, so the item is
noisy as a test item: a model that learned the train label scores the test item wrong. The
train rows leave train under the exact-short tier (they are among the 92 CLINC rows,
23 utterances times 4 families); nothing else about them changes.
