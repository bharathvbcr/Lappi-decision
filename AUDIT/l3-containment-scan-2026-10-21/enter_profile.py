"""cProfile of entering native_minhash (table build + signing + checks) on a synthetic corpus.

    python enter_profile.py <tree> <n_rows>
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
prof = cProfile.Profile()
t = time.perf_counter()
try:
    prof.enable()
    with p.native_minhash(rows, config=CONFIG):
        prof.disable()
        print(f"enter {time.perf_counter() - t:.1f} s", file=sys.stderr)
except SystemExit:
    pass  # the block refuses a run that never read its table
pstats.Stats(prof).sort_stats("tottime").print_stats(9)
