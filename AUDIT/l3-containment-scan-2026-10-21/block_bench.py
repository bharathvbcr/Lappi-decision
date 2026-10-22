"""Wall time of entering native_minhash (table build + signing + checks) on a small corpus.

    python block_bench.py <tree> <batch_sets> <n_texts> [repeat]
"""
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path

tree, batch, n_texts = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 3
sys.path[:0] = [str(tree / "python"), str(tree / "tools"), str(tree / "python" / "tests")]
os.environ["QD_PREP_BIN"] = "/Users/bharath/Code/research/Lappi-decision/.claude/worktrees/agent-abad50f046f9b7286/target/release/qd-prep"
import real_tokenizer_pipeline as p
from test_composed_file_near_duplicates import CONFIG, _repo_in, _row

rnd = random.Random(7)
vocab = [f"w{i}" for i in range(4000)]
repo = _repo_in("train", "org/t")
rows = [_row(f"r{i:05d}", repo=repo, text=" ".join(rnd.choice(vocab) for _ in range(420))) for i in range(n_texts)]
kw = {"batch_sets": batch} if "batch_sets" in p.native_minhash.__wrapped__.__code__.co_varnames else {}
times = []
for _ in range(repeat):
    t = time.perf_counter()
    try:
        with p.native_minhash(rows, config=CONFIG, **kw):
            times.append(time.perf_counter() - t)
    except SystemExit:
        pass  # the block refuses a run that never read its table; only the entry is timed
print("enter-block seconds", [round(x, 3) for x in times], "min", round(min(times), 3))
