"""dedupe + split inside native_minhash on a synthetic corpus; the block prints its own stage times.

    python block_dedupe.py <tree> <n_rows> [<cprofile out>]
"""
import cProfile
import os
import pstats
import random
import sys
import time
from pathlib import Path

tree, n_rows = Path(sys.argv[1]), int(sys.argv[2])
sys.path[:0] = [str(tree / "python"), str(tree / "tools"), str(tree / "python" / "tests")]
os.environ["QD_PREP_BIN"] = "/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286/target/release/qd-prep"
import real_tokenizer_pipeline as p
from test_composed_file_near_duplicates import CONFIG, _repo_in, _row

from qd_data.dedupe import dedupe
from qd_data.split import split

rnd = random.Random(3)
vocab = [f"w{i}" for i in range(6000)]
repos = [_repo_in(side, f"org/{side[0]}{j}-") for side in ("train", "val") for j in range(2)]
bases = [[rnd.choice(vocab) for _ in range(420)] for _ in range(max(n_rows // 4, 1))]
rows = []
for i in range(n_rows):
    toks = list(bases[i % len(bases)])
    for _ in range(rnd.randrange(0, 40)):
        toks[rnd.randrange(len(toks))] = rnd.choice(vocab)
    rows.append(_row(f"r{i:06d}", repo=repos[rnd.randrange(4)], text=" ".join(toks)))
t = time.perf_counter()
prof = cProfile.Profile() if len(sys.argv) > 3 else None
with p.native_minhash(rows, config=CONFIG):
    entered = time.perf_counter() - t
    if prof:
        prof.enable()
    report = dedupe(list(rows), config=CONFIG)
    split(report, config=CONFIG)
    if prof:
        prof.disable()
print(f"enter {entered:.1f} s, dedupe+split {time.perf_counter() - t - entered:.1f} s", file=sys.stderr)
if prof:
    pstats.Stats(prof).sort_stats("cumulative").print_stats(14)
