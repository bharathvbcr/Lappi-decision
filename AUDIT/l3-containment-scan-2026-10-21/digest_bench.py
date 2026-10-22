import hashlib
import random
import sys
import time
import zlib

sys.path[:0] = ["python"]
from qd_data.minhash import shingle

rnd = random.Random(1)
vocab = [f"w{i}" for i in range(4000)]
sets = [shingle(" ".join(rnd.choice(vocab) for _ in range(420)), k=5).shingles for _ in range(300)]


def timed(label, fn):
    t = time.perf_counter()
    for s in sets:
        fn(s)
    print(f"{label}: {(time.perf_counter() - t) / len(sets) * 1e6:.1f} us/set")


timed("sorted+join+blake2b", lambda s: hashlib.blake2b(b"\xff".join(sorted(s)), digest_size=16).digest())
timed("sum(map(crc32))", lambda s: sum(map(zlib.crc32, s)))
timed("join(unsorted)+blake2b (order dependent, not usable)", lambda s: hashlib.blake2b(b"\xff".join(s), digest_size=16).digest())
timed("sorted only", lambda s: sorted(s))
timed("shingle()", lambda s: None)
t = time.perf_counter()
for _ in range(3):
    from qd_data.minhash import MinHasher
    h = MinHasher(num_perm=128, seed=0)
    h.signature(sets[0])
print(f"reference signature of one set: {(time.perf_counter() - t) / 3 * 1e3:.1f} ms")
