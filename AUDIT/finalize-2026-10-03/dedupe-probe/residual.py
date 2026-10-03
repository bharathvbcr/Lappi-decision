"""Fable's condition for a second pool build, measured: does the pipeline's own dedupe (and the
split's near-duplicate search) fit under the 5,000,000 candidate-pair bound once the four structured
Open-Jev families are scoped out of it? Throwaway analysis (Python role 3), read-only.

Same path as the bench's pool_check.py: load_decision_pool -> build_mixture -> dedupe -> split,
with qd_data's reference MinHash (parity-tested against qd-prep's native path, so the candidate
count is the same). Plus the exact-content check Fable asked for, over the WHOLE pool: rows whose
(context, options) digest -- content alone, no group or repo key -- collides, and whether any such
group spans train and val.

usage: PYTHONPATH=<wt>/python:<wt>/tools python residual.py POOL_DIR OUT_JSON
"""

import collections
import hashlib
import json
import sys
import time
from pathlib import Path

from qd_data.decisions import load_decision_pool, pool_data_config
from qd_data.dedupe import dedupe
from qd_data.mixture import build_mixture
from qd_data.split import split
from real_tokenizer_pipeline import PIPELINE_MAX_CONSISTENCY_ROWS

SCOPED_OUT = frozenset({"openjev.policy", "openjev.evidence", "openjev.routing", "openjev.rubric"})

pool = load_decision_pool(Path(sys.argv[1]))
out = Path(sys.argv[2])
config = pool_data_config()

# --- the exact-content check, whole pool --------------------------------------------------------
by_digest: dict[str, list] = collections.defaultdict(list)
for rows in pool.raw.values():
    for r in rows:
        d = hashlib.blake2b("\x1f".join((r.context, *r.options)).encode("utf-8"), digest_size=16).hexdigest()
        by_digest[d].append(r)
collide = {d: rs for d, rs in by_digest.items() if len(rs) > 1}
straddle = {d: rs for d, rs in collide.items() if len({r.split for r in rs}) > 1}
diff_gold = {
    d: rs for d, rs in collide.items()
    if len({(r.gold_noul, None if r.gold_noul else r.gold_option) for r in rs}) > 1
}
content = {
    "pool_rows": sum(len(r) for r in pool.raw.values()),
    "distinct_content": len(by_digest),
    "colliding_groups": len(collide),
    "rows_in_colliding_groups": sum(len(rs) for rs in collide.values()),
    "groups_spanning_train_and_val": len(straddle),
    "groups_with_two_golds": len(diff_gold),
    "colliding_by_family": dict(collections.Counter(rs[0].family_id for rs in collide.values())),
    "spanning_examples": [
        [{"family": r.family_id, "split": r.split, "group_key": r.group_key} for r in rs]
        for rs in list(straddle.values())[:5]
    ],
}
print("content:", json.dumps(content))

# --- the residual candidate count -------------------------------------------------------------
raw = {
    s: [r for r in rows if r.family_id not in SCOPED_OUT] for s, rows in pool.raw.items()
}
raw = {s: rs for s, rs in raw.items() if rs}
n_in = sum(len(r) for r in raw.values())
t0 = time.perf_counter()
mixture = build_mixture(raw, config=config, max_consistency_rows=PIPELINE_MAX_CONSISTENCY_ROWS)
t1 = time.perf_counter()
d = dedupe(list(mixture.rows), config=config)
t2 = time.perf_counter()
s = split(d, config=config)
t3 = time.perf_counter()
dropped = collections.Counter()
kept_ids = {r.row_id for r in d.kept}
for r in mixture.rows:
    if r.row_id not in kept_ids:
        dropped[r.family_id] += 1
report = {
    "pool_examples_sha256": pool.examples_sha256,
    "scoped_out": sorted(SCOPED_OUT),
    "rows_in": n_in,
    "mixture_rows": len(mixture.rows),
    "dedupe_bound": 5_000_000,
    "dedupe_n_candidate_pairs": d.n_candidate_pairs,
    "dedupe_status": d.status.to_json(),
    "dedupe_cross_repo_pairs": d.n_cross_repo_pairs,
    "dedupe_within_repo_pairs": d.n_within_repo_pairs,
    "dedupe_dropped_rows": d.n_dropped_rows,
    "dedupe_dropped_by_family": dict(dropped),
    "split_status": s.status.to_json(),
    "near_duplicate_disjoint": s.near_duplicate_disjoint.to_json(),
    "seconds": {"mixture": round(t1 - t0, 1), "dedupe": round(t2 - t1, 1), "split": round(t3 - t2, 1)},
    "content": content,
}
with out.open("x") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps({k: v for k, v in report.items() if k != "content"}, default=str)[:3000])
