"""Build the characterization test's 60-pair corpus from a tree and print its build_digest.

    X7_COMMITPACKFT=<commitpackft dir> QD_PREP_BIN=<qd-prep> python char_build.py <tree> <out dir>

Adapted from AUDIT/l3-containment-scan-2026-10-21/char_build.py: the qd-prep binary and the
commitpackft download come from the environment rather than a baked-in path. Runs pipeline.run
exactly as test_a_build_without_the_flag_writes_what_it_wrote_before does, from <tree>'s own
tools/ and python/ (v6_data_rules=False where that tree's run() has the parameter).
"""

import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

tree, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(tree / "python"), str(tree / "tools")]
os.environ["HF_HUB_OFFLINE"] = "1"
pipeline = importlib.import_module("real_tokenizer_pipeline")

has_rules = "v6_data_rules" in inspect.signature(pipeline.run).parameters
extra = {"v6_data_rules": False} if has_rules else {}
pipeline.run(
    out=out, max_pairs=60, blank_line_runs=False, rev="dec48d8841db21ad0401effdb9fff724977fd47f",
    commitpackft=Path(os.environ["X7_COMMITPACKFT"]), val_shards=True, repo_history=False, **extra,
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
