"""Table of the six interleaved runs from build/l3/six.log."""
import ast
import re
import statistics
import sys

text = open("build/l3/six.log").read()
runs = {}
for m in re.finditer(r"# (six_\w+) start.*?\n(.*?)# \1 end", text, re.S):
    tag, body = m.group(1), m.group(2)
    head = re.search(r"footprint_peak: peak (\d+) MiB, wall ([\d.]+) s, exit (-?\d+), (\d+) samples", body)
    low = re.search(
        r"sampled tree sum (\d+) MiB, one process's kernel lifetime max (\d+) MiB, and the largest "
        r"process RSS in the exit report \(reaped children included\) (\d+) MiB; sum of seen "
        r"processes' lifetime maxima (\d+) MiB; per stage, lifetime max MiB: (\{.*\})", body)
    if not (head and low):
        runs[tag] = None
        continue
    stages = ast.literal_eval(low.group(5))
    children = {k: v for k, v in stages.items() if k.startswith("qd-prep")}
    parent = max((v for k, v in stages.items() if not k.startswith("qd-prep")), default=0)
    runs[tag] = dict(
        peak=int(head.group(1)), wall=float(head.group(2)), exit=int(head.group(3)),
        tree=int(low.group(1)), one=int(low.group(2)), reaped=int(low.group(3)),
        seen=int(low.group(4)), parent=parent,
        child=max(children.values(), default=0),
        child_name=max(children, key=children.get) if children else "-",
    )
print("| run | exit | wall s | parent lifetime MiB | largest child lifetime MiB | tree sampled MiB | reaped RSS MiB | headline lower bound MiB |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for tag, r in runs.items():
    if r is None:
        print(f"| {tag} | NOT PARSED | | | | | | |")
        continue
    print(f"| {tag} | {r['exit']} | {r['wall']} | {r['parent']} | {r['child']} ({r['child_name']}) | "
          f"{r['tree']} | {r['reaped']} | {r['peak']} |")
for arm in ("base", "after"):
    rs = [r for t, r in runs.items() if t.startswith(f"six_{arm}") and r]
    if not rs:
        continue
    for key in ("wall", "parent", "child", "tree", "peak"):
        v = [r[key] for r in rs]
        med = statistics.median(v)
        print(f"{arm} {key}: n={len(v)} min {min(v)} median {med} max {max(v)} spread {(max(v) - min(v)) / med * 100:.1f}%")
b = [r["wall"] for t, r in runs.items() if t.startswith("six_base") and r]
a = [r["wall"] for t, r in runs.items() if t.startswith("six_after") and r]
if b and a:
    print(f"wall: after median {statistics.median(a)} vs base median {statistics.median(b)} = "
          f"{(statistics.median(a) / statistics.median(b) - 1) * 100:+.1f}%")
for line in text.splitlines():
    if "IDENTICAL" in line or "DIFFERS" in line:
        print(line)
sys.exit(0)
