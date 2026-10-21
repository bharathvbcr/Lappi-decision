import json
import sys
from pathlib import Path


def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                yield f"{path}/{k}: only in head"
            elif k not in b:
                yield f"{path}/{k}: only in pin"
            else:
                yield from walk(a[k], b[k], f"{path}/{k}")
    elif a != b:
        yield f"{path}: {str(a)[:70]!r} -> {str(b)[:70]!r}"


pin, head = Path(sys.argv[1]), Path(sys.argv[2])
for rel in sys.argv[3:]:
    print("==", rel)
    for line in walk(json.loads((pin / rel).read_text()), json.loads((head / rel).read_text())):
        print("  ", line)
