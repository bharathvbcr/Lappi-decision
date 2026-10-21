"""Symlink untracked data/pool files into a worktree (read-only use).

    python3 -I build/l3/link_data.py <worktree root>

The defect corpus's parts resolve under REPO/data/pool, and a worktree holds only the tracked
manifests. Sources, first match wins: the main checkout's data/pool, then the v5 data campaign's
(/Users/bharath/qd-campaign/v5-data-2026-10-02), which holds the composed corpora and prose units the
main checkout lacks. Nothing is written under either; existing worktree files are never replaced.
The loaders check every file against the manifest's sha256, so a wrong file is a refusal.
"""

import os
import sys
from pathlib import Path

roots = [
    Path("/Users/bharath/Code/research/Lappi-decision/data/pool"),
    Path("/Users/bharath/qd-campaign/v5-data-2026-10-02/data/pool"),
]
here = Path(sys.argv[1]) / "data" / "pool"
made = skipped = 0
for root in roots:
    for src in sorted(root.rglob("*")):
        rel = src.relative_to(root)
        dst = here / rel
        if src.is_dir() and not src.is_symlink():
            continue
        if dst.exists() or dst.is_symlink():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst)
        made += 1
print(f"linked {made}, left {skipped} existing", file=sys.stderr)
