"""Which pool families drive dedupe's LSH candidate count? Throwaway analysis (Python role 3).

For each family of the built pool: a seeded reservoir sample of up to 400 rows (streamed, so
memory stays small), exact Jaccard for every sampled pair over the pipeline's own shingles, and
the pair's LSH candidate probability under the pipeline's own banding
(``choose_bands(num_perm=128, threshold=0.8)``: P(J) = 1 - (1 - J**r)**b). The expected number of
candidates within the family is the sampled mean of P(J) times N(N-1)/2 [I]. Cross-family pairs
are not estimated here. Also the pairs at J >= 0.8 and how many have different golds [V on the
sample].
"""

import itertools
import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, "/Users/bharath/Code/research/Lappi-decision/build/v5-build-wt/python")
from qd_data.minhash import choose_bands, exact_jaccard, shingle  # noqa: E402

POOL = "/Users/bharath/qd-campaign/v5-decisions-data-2026-10-03/pool-v5-decisions/examples.jsonl"
K = 400
band = choose_bands(num_perm=128, threshold=0.8)


def p_cand(j: float) -> float:
    return 1.0 - (1.0 - j**band.rows) ** band.bands


rng = random.Random(20261003)
seen: dict[str, int] = defaultdict(int)
res: dict[str, list[dict]] = defaultdict(list)
with open(POOL, encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        f = r["family_id"]
        seen[f] += 1
        if len(res[f]) < K:
            res[f].append(r)
        else:
            i = rng.randrange(seen[f])
            if i < K:
                res[f][i] = r

total = 0.0
print(f"banding b={band.bands} r={band.rows}")
for fam in sorted(res, key=lambda f: -seen[f]):
    rows = res[fam]
    n_all = seen[fam]
    sh = [shingle("\n".join((r["context"], *r["options"])), k=5).shingles for r in rows]
    s_p = 0.0
    n_pairs = hi = hi_diff = 0
    for i, j in itertools.combinations(range(len(rows)), 2):
        jac = exact_jaccard(sh[i], sh[j])
        s_p += p_cand(jac)
        n_pairs += 1
        if jac >= 0.8:
            hi += 1
            a, b = rows[i], rows[j]
            hi_diff += (a["gold_noul"], a["gold_option"]) != (b["gold_noul"], b["gold_option"])
    est = s_p / max(n_pairs, 1) * n_all * (n_all - 1) / 2
    total += est
    print(
        f"{fam:24s} rows {n_all:7d}  est. candidates {est:12,.0f}  "
        f"sample J>=0.8 {hi:5d} (different gold {hi_diff:4d}) of {n_pairs} pairs"
    )
print(f"sum of within-family estimates {total:,.0f} [I]")
