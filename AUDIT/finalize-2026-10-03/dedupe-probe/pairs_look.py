"""What are procedural's and decider.commands' high-Jaccard pairs? Throwaway analysis (role 3).

Prints, for the procedural pair the split's re-derivation found crossing a split (J=0.808), and
for decider.commands pairs at J >= 0.8 in a seeded sample, the differing spans of the two
contexts and the two golds.
"""

import difflib
import itertools
import json
import random
import sys

sys.path.insert(0, "/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt/python")
from qd_data.minhash import exact_jaccard, shingle  # noqa: E402

POOL = "/Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions/examples.jsonl"
CROSSING = {
    "procedural:evidence_sufficiency:train:11275:strongest_support_origin",
    "procedural:evidence_sufficiency:train:2982:strongest_support_origin",
}
proc: list[dict] = []
cmds: list[dict] = []
with open(POOL, encoding="utf-8") as fh:
    for line in fh:
        if '"decider.commands"' in line:
            cmds.append(json.loads(line))
        elif any(c in line for c in CROSSING):
            proc.append(json.loads(line))


def text(r: dict) -> str:
    return "\n".join((r["context"], *r["options"]))


def show(a: dict, b: dict) -> None:
    j = exact_jaccard(shingle(text(a), k=5).shingles, shingle(text(b), k=5).shingles)
    ga = None if a["gold_noul"] else a["gold_option"]
    gb = None if b["gold_noul"] else b["gold_option"]
    print(f"--- J={j:.3f} split {a['split']}/{b['split']} gold {ga!r} / {gb!r}")
    print(f"    ids {a['id']} / {b['id']}")
    sm = difflib.SequenceMatcher(None, a["context"], b["context"], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            print(f"    {op}: A[{a['context'][i1:i2][:120]!r}] B[{b['context'][j1:j2][:120]!r}]")
    if a["options"] != b["options"]:
        print(f"    options differ: {a['options'][:4]} / {b['options'][:4]}")


print(f"procedural crossing pair ({len(proc)} rows found):")
if len(proc) == 2:
    print(f"    context length {len(proc[0]['context'])} chars")
    show(proc[0], proc[1])

rng = random.Random(20261003)
sample = rng.sample(cmds, min(400, len(cmds)))
sh = [shingle(text(r), k=5).shingles for r in sample]
shown = 0
for i, j in itertools.combinations(range(len(sample)), 2):
    if exact_jaccard(sh[i], sh[j]) >= 0.8 and shown < 6:
        shown += 1
        show(sample[i], sample[j])
print(f"decider.commands: shown {shown}; median context {sorted(len(r['context']) for r in sample)[200]} chars")
