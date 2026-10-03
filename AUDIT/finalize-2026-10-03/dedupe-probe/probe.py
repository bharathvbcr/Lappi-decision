"""What are the pool's near-duplicate pairs? Throwaway analysis (Python role 3), read-only.

For each family named, a seeded sample of rows from the built pool, every pair's exact Jaccard
over the pipeline's own shingles (qd_data.minhash.shingle, k=5, the dedupe_text the pool row
carries: context + options). For pairs at >= 0.8 it reports how many have DIFFERENT gold options
(distinct decisions the dedupe would merge) and the size of the part of the context that differs.
"""

import difflib
import itertools
import json
import random
import sys

sys.path.insert(0, "/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt/python")
from qd_data.minhash import exact_jaccard, shingle  # noqa: E402

POOL = "/Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions/examples.jsonl"
import re  # noqa: E402

# MODE=spaced: JSON punctuation split into tokens before shingling, so a fact is a token.
MODE = "spaced" if sys.argv[1:2] == ["--spaced"] else "as-built"
ARGS = sys.argv[2:] if MODE == "spaced" else sys.argv[1:]
FAMILIES = ARGS or ["openjev.policy", "openjev.evidence", "openjev.routing", "openjev.rubric", "decider.commands", "procedural.decisions"]
_PUNCT = re.compile(r'([{}\[\],:"])')


def view(text: str) -> str:
    return _PUNCT.sub(r" \1 ", text) if MODE == "spaced" else text
SAMPLE = 400
THRESH = 0.8
SHOW = False

by_family: dict[str, list[dict]] = {f: [] for f in FAMILIES}
with open(POOL, encoding="utf-8") as fh:
    for line in fh:
        for f in FAMILIES:
            if f'"family_id":"{f}"' in line:
                by_family[f].append(json.loads(line))
                break

for fam in FAMILIES:
    rows = by_family[fam]
    rng = random.Random(20261003)
    sample = rng.sample(rows, min(SAMPLE, len(rows)))
    texts = ["\n".join((r["context"], *r["options"])) for r in sample]
    sh = [shingle(view(t), k=5).shingles for t in texts]
    n_pairs = n_hi = n_hi_diff_gold = n_hi_same_group = n_mid = n_hi_same_tmpl = n_hi_same_tmpl_same_gold = 0
    tmpls = {r["group_key"].rsplit("/", 1)[0] for r in rows}
    diffs: list[tuple[float, int, int, int, int]] = []
    shown = 0
    for i, j in itertools.combinations(range(len(sample)), 2):
        n_pairs += 1
        jac = exact_jaccard(sh[i], sh[j])
        n_mid += 0.5 <= jac < THRESH
        if jac < THRESH:
            continue
        n_hi += 1
        a, b = sample[i], sample[j]
        gold_a = None if a["gold_noul"] else a["gold_option"]
        gold_b = None if b["gold_noul"] else b["gold_option"]
        diff_gold = gold_a != gold_b
        n_hi_diff_gold += diff_gold
        n_hi_same_group += a["group_key"] == b["group_key"]
        n_hi_same_tmpl += a["group_key"].rsplit("/", 1)[0] == b["group_key"].rsplit("/", 1)[0]
        n_hi_same_tmpl_same_gold += (not diff_gold) and a["group_key"].rsplit("/", 1)[0] == b["group_key"].rsplit("/", 1)[0]
        sm = difflib.SequenceMatcher(None, a["context"], b["context"], autojunk=False)
        same = sum(m.size for m in sm.get_matching_blocks())
        diffs.append((jac, len(a["context"]), len(b["context"]), same, int(diff_gold)))
        if shown < 2 and diff_gold and SHOW:
            shown += 1
            print(f"--- {fam} pair J={jac:.3f}, golds {gold_a!r} vs {gold_b!r}, groups {a['group_key']} / {b['group_key']}")
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != "equal":
                    print(f"   {op}: A[{a['context'][i1:i2][:160]!r}]  B[{b['context'][j1:j2][:160]!r}]")
    ctx_len = sorted(len(r["context"]) for r in sample)
    est = n_hi / max(n_pairs, 1) * len(rows) * (len(rows) - 1) / 2
    est_mid = n_mid / max(n_pairs, 1) * len(rows) * (len(rows) - 1) / 2
    print(
        f"[{MODE}] {fam}: est. family pairs J>={THRESH} {est:,.0f}, 0.5<=J<{THRESH} {est_mid:,.0f}\n"
        f"{fam}: {len(rows)} rows, sample {len(sample)}, {n_pairs} pairs; J>={THRESH}: {n_hi} "
        f"({n_hi / max(n_pairs, 1):.4%}); of those, different gold {n_hi_diff_gold}, same group "
        f"{n_hi_same_group}, same template {n_hi_same_tmpl} (same template AND same gold "
        f"{n_hi_same_tmpl_same_gold}); {len(tmpls)} templates; median context {ctx_len[len(ctx_len) // 2]} chars"
    )
    if diffs:
        diff_chars = sorted(min(la, lb) - s for _, la, lb, s, _ in diffs)
        print(f"   differing context chars at J>={THRESH}: median {diff_chars[len(diff_chars) // 2]}, "
              f"max {diff_chars[-1]}; min J {min(d[0] for d in diffs):.3f}")
