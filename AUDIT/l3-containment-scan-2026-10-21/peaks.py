"""Per timeline: peak, the sample's processes, and the peak of each stage (qd-prep child or not)."""
import sys
from pathlib import Path

for tag in sys.argv[1:]:
    rows = []
    for line in Path(f"build/l3/{tag}.timeline.txt").read_text().splitlines():
        parts = line.split(None, 2)
        rows.append((int(parts[1]), parts[0], parts[2] if len(parts) > 2 else ""))
    peak = max(rows)
    by_stage: dict[str, int] = {}
    for mib, _t, procs in rows:
        stage = "python only"
        for token in procs.split():
            if ":qd-prep:" in token:
                stage = "qd-prep " + token.split(":")[2]
        by_stage[stage] = max(by_stage.get(stage, 0), mib)
    print(f"{tag}: peak {peak[0]} MiB at {peak[1]} [{peak[2]}]; stage peaks {by_stage}; {len(rows)} samples")
