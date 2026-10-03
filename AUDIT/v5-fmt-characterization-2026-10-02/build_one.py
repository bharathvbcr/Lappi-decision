"""Throwaway analysis (never shipped): run the 60-pair build of
python/tests/test_containment_exclusions.py::test_a_build_without_the_flag_writes_what_it_wrote_before
from a given source tree into a given output directory, and print its build digest.

Usage: python build_one.py <tree> <out>   (QD_PREP_BIN must name a qd-prep binary)
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(tree / "tools"))
sys.path.insert(0, str(tree / "python"))

import real_tokenizer_pipeline as pipeline  # noqa: E402

import qd_data.render as render  # noqa: E402

for mod in (pipeline, render):
    if not Path(mod.__file__).resolve().is_relative_to(tree):
        raise SystemExit(f"{mod.__name__} came from {mod.__file__}, not {tree}")

# The same constants and guards as the test (python/tests/test_containment_exclusions.py).
PIN_REV = "dec48d8841db21ad0401effdb9fff724977fd47f"
PIN_TOKENIZER_SHA256 = "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"
DOWNLOAD = tree / "data" / "pool" / "commitpackft"

if not os.environ.get("QD_PREP_BIN"):
    raise SystemExit("QD_PREP_BIN is unset")
snapshot = (
    pipeline.MODEL_REF.parent.parent / "snapshots"
    / pipeline.MODEL_REF.read_text(encoding="utf-8").strip() / "tokenizer.json"
)
if hashlib.sha256(snapshot.read_bytes()).hexdigest() != PIN_TOKENIZER_SHA256:
    raise SystemExit(f"{snapshot} is not the pinned tokenizer")
if not all((DOWNLOAD / f"{lang}.jsonl").exists() for lang in ("go", "python")):
    raise SystemExit(f"{DOWNLOAD} lacks the commitpackft download")

out.mkdir(parents=True, exist_ok=False)
pipeline.run(
    out=out, max_pairs=60, blank_line_runs=False, rev=PIN_REV, commitpackft=DOWNLOAD,
    val_shards=True, repo_history=False,
)
print(f"built {out} from {tree} (render.py prompt format: "
      f"{getattr(render, 'PROMPT_FORMAT', 'none (format 1)')})")
