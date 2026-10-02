"""Evidence script for lane L-v5plan (2026-10-02). Throwaway accounting, never shipped.

What the human-approved CLINC oos re-key (human-decisions.md item 2) does to the CLINC
train / val / held-out populations, computed from the cached fetch
(/Users/bharath/.cache/qd-decision/general/clinc__clinc_oos/155b9c71.../{train,validation,test}.jsonl)
with the repo's own split function (python/qd_data/split.py assign_repo, imported read-only).

v4 keys every CLINC row by intent (mixture.py rewrite_clinc, general.py rewrite_clinc_two_stage:
repo_key = f"clinc-intent:{raw.intent}"), so all oos utterances share repo key
"clinc-intent:oos" and one split. The re-key gives each oos utterance its own repo key.
The exact key spelling is the v5 build lane's choice; this script evaluates the candidate
"clinc-oos:{blake2b-8(utterance.strip())}" (the digest rewrite_clinc already computes for
identity), and also reports how the in-scope intents split, which the re-key does not move.

Not modelled here (stated, not guessed): MinHash dedupe at Jaccard >= 0.8 runs before the split
(split.py: "dedupe across everything, then split") and may drop a few near-duplicate oos
utterances once they no longer share a repo key; the v4 build dropped 8 CLINC rows as
contradictions (train-plan-2026-09-28.md:355). Those move the counts by a handful of rows.
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_data.split import assign_repo

D = Path(
    "/Users/bharath/.cache/qd-decision/general/clinc__clinc_oos/"
    "155b9c710419136e17307b80d0a13e68cd46b4ec"
)
SEED = 20260919  # v4's config_fingerprint.seed
TRAIN_FRACTION = 0.9
VAL_FRACTION = 0.05
OUT = Path(__file__).with_name("clinc_oos_rekey.json")


def main() -> int:
    names = json.loads((D / "intent_names.json").read_text())
    oos_index = names.index("oos")
    upstream = collections.Counter()
    oos_split = collections.Counter()
    oos_by_upstream_and_split = collections.Counter()
    oos_digests: set[str] = set()
    oos_dup = 0
    intents_split = collections.Counter()
    for split_file in ("train", "validation", "test"):
        for line in (D / f"{split_file}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            intent = names[rec["intent"]]
            upstream[(split_file, intent == "oos")] += 1
            if rec["intent"] != oos_index:
                continue
            digest = hashlib.blake2b(rec["text"].strip().encode("utf-8"), digest_size=8).hexdigest()
            if digest in oos_digests:
                oos_dup += 1
            oos_digests.add(digest)
            s = assign_repo(
                f"clinc-oos:{digest}", seed=SEED,
                train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION,
            )
            oos_split[s] += 1
            oos_by_upstream_and_split[(split_file, s)] += 1
    for intent in names:
        intents_split[
            assign_repo(
                f"clinc-intent:{intent}", seed=SEED,
                train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION,
            )
        ] += 1
    v4_oos_split = assign_repo(
        "clinc-intent:oos", seed=SEED, train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION,
    )
    out = {
        "fetch_dir": str(D),
        "seed": SEED,
        "upstream_rows": {
            f"{k[0]}:{'oos' if k[1] else 'in_scope'}": v for k, v in sorted(upstream.items())
        },
        "v4_oos_repo_key_split": v4_oos_split,
        "intent_repo_keys_by_split_including_oos": dict(sorted(intents_split.items())),
        "candidate_key": "clinc-oos:{blake2b-8(utterance.strip())}",
        "oos_utterances": sum(oos_split.values()),
        "oos_exact_duplicate_utterances": oos_dup,
        "oos_rekeyed_split": dict(sorted(oos_split.items())),
        "oos_rekeyed_split_by_upstream_file": {
            f"{k[0]}->{k[1]}": v for k, v in sorted(oos_by_upstream_and_split.items())
        },
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
