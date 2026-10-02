"""Evidence script for lane L-v5plan (2026-10-02). Throwaway accounting, never shipped.

Counts rows per (split, family_id, source_id, licence_id) in F's v4 manifests on the Mac
(/Users/bharath/qd-campaign/phase4-v4-2026-10-01/data/...). Reads manifests only: row ids,
families, sources, licences and repo keys. No row content is read; nothing is trained.
Writes counts.json beside this file and prints peak RSS.
"""

import collections
import json
import resource
import sys
from pathlib import Path

ROOT = Path("/Users/bharath/qd-campaign/phase4-v4-2026-10-01/data")
FILES = {
    "train": ROOT / "pool" / "train.json",
    "val": ROOT / "pool" / "val.json",
    "heldout": ROOT / "heldout" / "heldout.json",
}
OUT = Path(__file__).with_name("v4_manifest_counts.json")


def kind_of(row_id: str, family: str) -> str:
    if family != "code.defect_class":
        return "-"
    if ":compose:" in row_id:
        return "composed"
    if "noul" in row_id:
        return "noul"
    return "single"


def main() -> int:
    out: dict = {"source_files": {}, "splits": {}}
    for name, path in FILES.items():
        doc = json.loads(path.read_text())
        rows = doc["entries"]
        out["source_files"][name] = {
            "mixture": doc.get("mixture"),
            "split_report": doc.get("split_report"),
            "path": str(path),
            "data_snapshot_hash": doc.get("data_snapshot_hash"),
            "n_rows": doc.get("n_rows"),
            "rows_listed": len(rows),
        }
        by = collections.Counter()
        clinc_oos = collections.Counter()
        repo_keys = collections.defaultdict(set)
        for r in rows:
            fam = r["family_id"]
            by[(r["split"], fam, r["source_id"], r["licence_id"], kind_of(r["row_id"], fam))] += 1
            repo_keys[fam].add(r["repo_key"])
            if r["source_id"] == "clinc/clinc_oos" and r["repo_key"] == "clinc-intent:oos":
                clinc_oos[fam] += 1
        out["splits"][name] = {
            "by_split_family_source_licence_kind": [
                {
                    "split": k[0],
                    "family": k[1],
                    "source": k[2],
                    "licence": k[3],
                    "kind": k[4],
                    "rows": v,
                }
                for k, v in sorted(by.items())
            ],
            "distinct_repo_keys_per_family": {f: len(s) for f, s in sorted(repo_keys.items())},
            "clinc_oos_repo_key_rows_per_family": dict(sorted(clinc_oos.items())),
        }
        del doc, rows
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["peak_rss_bytes_darwin"] = peak
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
