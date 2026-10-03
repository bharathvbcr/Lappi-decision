"""Fable's (A): the true candidate count of the pool's MinHash population, measured, not floored.

The same population as residual.py: the built pool minus the four structured Open-Jev families,
which the ruling dedupes by exact content and keeps out of the search. The pipeline's own path,
as tools/real_tokenizer_pipeline.py runs it: build_mixture, then dedupe and split inside
`native_minhash` (qd-prep signs and bands; QD_PREP_BIN must be set), with a ceiling high enough
not to truncate. It reads n_candidate_pairs for both searches, and the seconds; peak RSS comes
from /usr/bin/time -l around it.

usage: QD_PREP_BIN=... PYTHONPATH=<wt>/python:<wt>/tools python residual_measure.py POOL_DIR OUT_JSON CEILING
"""

import json
import sys
import time
from pathlib import Path

import qd_data.split as split_module
from qd_data.decisions import load_decision_pool, pool_data_config
from qd_data.dedupe import dedupe
from qd_data.mixture import build_mixture
from qd_data.split import split
from real_tokenizer_pipeline import PIPELINE_MAX_CONSISTENCY_ROWS, native_minhash

SCOPED_OUT = frozenset({"openjev.policy", "openjev.evidence", "openjev.routing", "openjev.rubric"})

pool = load_decision_pool(Path(sys.argv[1]))
out = Path(sys.argv[2])
ceiling = int(sys.argv[3])
config = pool_data_config()
raw = {s: [r for r in rows if r.family_id not in SCOPED_OUT] for s, rows in pool.raw.items()}
raw = {s: rs for s, rs in raw.items() if rs}
mixture = build_mixture(raw, config=config, max_consistency_rows=PIPELINE_MAX_CONSISTENCY_ROWS)

# The split's search count is not in its report; count what its candidate_pairs returns.
split_counts: list[tuple[int, bool]] = []
t0 = time.perf_counter()
with native_minhash(mixture.rows, config=config):
    inner = split_module.candidate_pairs

    def counted(signatures, *, config, max_pairs):  # noqa: ANN001, ANN202
        pairs, truncated = inner(signatures, config=config, max_pairs=max_pairs)
        split_counts.append((len(pairs), truncated))
        return pairs, truncated

    t1 = time.perf_counter()
    d = dedupe(list(mixture.rows), config=config, max_candidate_pairs=ceiling)
    t2 = time.perf_counter()
    split_module.candidate_pairs = counted
    try:
        s = split(d, config=config, max_candidate_pairs=ceiling)
    finally:
        split_module.candidate_pairs = inner
    t3 = time.perf_counter()

report = {
    "pool_examples_sha256": pool.examples_sha256,
    "scoped_out": sorted(SCOPED_OUT),
    "mixture_rows": len(mixture.rows),
    "ceiling": ceiling,
    "dedupe_n_candidate_pairs": d.n_candidate_pairs,
    "dedupe_status": d.status.to_json(),
    "dedupe_cross_repo_pairs": d.n_cross_repo_pairs,
    "dedupe_within_repo_pairs": d.n_within_repo_pairs,
    "dedupe_dropped_rows": d.n_dropped_rows,
    "split_candidate_pairs": split_counts,
    "split_status": s.status.to_json(),
    "near_duplicate_disjoint": s.near_duplicate_disjoint.to_json(),
    "seconds": {
        "native_setup": round(t1 - t0, 1), "dedupe": round(t2 - t1, 1), "split": round(t3 - t2, 1),
    },
}
with out.open("x") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, default=str)[:3000], flush=True)
