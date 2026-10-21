import collections
import json
import sys
from pathlib import Path

pin, head = Path(sys.argv[1]), Path(sys.argv[2])
for rel in sys.argv[3:]:
    a = json.loads((pin / rel).read_text())["entries"]
    b = json.loads((head / rel).read_text())["entries"]
    print("==", rel, "rows", len(a), len(b), "same row ids in order", [e["row_id"] for e in a] == [e["row_id"] for e in b])
    keys = collections.Counter()
    only_pin = collections.Counter()
    only_head = collections.Counter()
    for x, y in zip(a, b, strict=True):
        for k in sorted(set(x) | set(y)):
            if k not in x:
                only_head[k] += 1
            elif k not in y:
                only_pin[k] += 1
            elif x[k] != y[k]:
                keys[k] += 1
    print("   values that differ per key:", dict(keys), "| only in pin:", dict(only_pin), "| only in head:", dict(only_head))
