"""Write the split map ``qd-mutate compose`` reads: every pool repo's split, by the canonical hash.

A thin binding over ``qd_data.split.assign_repo``, the one thing the Rust composer must not
re-implement: a second copy of "which repos are held out" is the copy that drifts into a rule-3
breach (``crates/qd-mutate/src/noul_rows/allowlist.rs`` gives the same reason for the noul
rows). Every repo the pool names is mapped to ``train``, ``val`` or ``heldout`` at
``DataConfig()``'s seed and fractions, and the pool's sha256 and record count are pinned so the
composer can refuse a map computed over other bytes. The composer refuses a repo the map does
not hold rather than defaulting it.

    python tools/compose_split_map.py --pool data/pool/commitpackft-pool-v2.jsonl \
        --out data/pool/commitpackft-pool-v2.split-map.json
    qd-mutate compose --corpus data/pool/commitpackft-corpus-v3 \
        --pool data/pool/commitpackft-pool-v2.jsonl \
        --split-map data/pool/commitpackft-pool-v2.split-map.json \
        --out data/pool/commitpackft-composed-v1 --seed 0

The map is the composer's filter, not the rule-3 guarantee: ``qd_data.defect_class`` re-derives
every composed row's constituent splits at the training run's own config and refuses the corpus
on any row that mixes splits or holds a held-out repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qd_data.config import DataConfig
from qd_data.split import assign_repo
from qd_train.data_access import assert_path_not_held_out
from qd_train.replay import write_text_atomic

REPO = Path(__file__).resolve().parents[1]

#: What ``crates/qd-mutate/src/compose.rs`` (``SPLIT_MAP_SCHEMA``) refuses anything but.
SCHEMA = "qd-compose-split-map/v1"


def split_map(
    pool: Path, *, config: DataConfig, repo_root: Path
) -> tuple[dict[str, object], dict[str, str]]:
    """``{schema, split, pool, repos}`` over every repo ``pool`` names, and the repo map itself."""
    assert_path_not_held_out(Path(pool), config=config, repo_root=Path(repo_root))
    raw = Path(pool).read_bytes()
    repos: dict[str, str] = {}
    records = 0
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > 12 * 1024 * 1024:
            raise ValueError(f"{pool}:{lineno}: over qd-mutate's 12 MiB pool record bound")
        records += 1
        repo = json.loads(line)["repo"]
        if not isinstance(repo, str) or not repo.strip():
            raise ValueError(f"{pool}:{lineno}: a pool record with no repo cannot be split")
        if repo not in repos:
            repos[repo] = assign_repo(
                repo, seed=config.seed, train_fraction=config.train_fraction,
                val_fraction=config.val_fraction,
            )
    if not repos:
        raise ValueError(f"{pool} holds no record; an empty map refuses nothing")
    ordered = dict(sorted(repos.items()))
    return {
        "schema": SCHEMA,
        "split": {
            "seed": config.seed,
            "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
        },
        "pool": {
            "file": Path(pool).name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "records": records,
        },
        "repos": ordered,
    }, ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True, help="the commitpackft pool JSONL")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; refusing to overwrite it")
    out, repos = split_map(args.pool, config=DataConfig(), repo_root=REPO)
    write_text_atomic(args.out, json.dumps(out, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for split in repos.values():
        counts[split] = counts.get(split, 0) + 1
    print(f"split map {args.out}: {len(repos)} repos {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
