"""Build the characterization test's 60-pair corpus from a tree and print its build_digest.

    python char_build.py <tree> <out dir>

Runs pipeline.run exactly as test_a_build_without_the_flag_writes_what_it_wrote_before does, from
<tree>'s own tools/ and python/ (v6_data_rules=False where that tree's run() has the parameter).
"""

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

tree, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(tree / "python"), str(tree / "tools")]
os.environ["QD_PREP_BIN"] = "/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286/target/release/qd-prep"
os.environ["HF_HUB_OFFLINE"] = "1"
import real_tokenizer_pipeline as pipeline

extra = {"v6_data_rules": False} if "v6_data_rules" in inspect.signature(pipeline.run).parameters else {}
pipeline.run(
    out=out, max_pairs=60, blank_line_runs=False, rev="dec48d8841db21ad0401effdb9fff724977fd47f",
    commitpackft=tree / "data" / "pool" / "commitpackft", val_shards=True, repo_history=False, **extra,
)


def normalised(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix != ".json":
        return raw
    body = json.loads(raw)
    if path.name == "header.json":
        body.pop("created_at", None)
    if isinstance(body, dict) and isinstance(body.get("provenance"), dict):
        body["provenance"].pop("built_at_utc", None)
    return json.dumps(body, sort_keys=True).encode()


h = hashlib.sha256()
for path in sorted(p for p in out.rglob("*") if p.is_file()):
    h.update(str(path.relative_to(out)).encode() + b"\n")
    h.update(hashlib.sha256(normalised(path)).hexdigest().encode() + b"\n")
print("BUILD_DIGEST", h.hexdigest(), "extra", extra, file=sys.stderr)
