import re

text = open("build/l3/six.log").read()
print("| run | minhash block s | lsh block s | split stage s | containment s |")
print("| --- | --- | --- | --- | --- |")
for m in re.finditer(r"# (six_\w+) start.*?\n(.*?)# \1 end", text, re.S):
    tag, body = m.group(1), m.group(2)
    mh = re.search(r"minhash: \d+ distinct sets signed by \S+ in ([\d.]+) s", body)
    ls = re.search(r"lsh: \d+ candidate-pair searches banded by \S+ in ([\d.]+) s", body)
    sp = re.search(r"split \{[^}]*\} in ([\d.]+) s", body)
    ct = re.search(r"qd-prep containment: .* in ([\d.]+) s", body)
    print(f"| {tag} | {mh and mh.group(1)} | {ls and ls.group(1)} | {sp and sp.group(1)} | {ct and ct.group(1)} |")
print()
for line in text.splitlines():
    if re.match(r"^[0-9a-f]{64}  .*(pairs.tsv|exclusions.txt|attestation.json)$", line):
        print(line.split("/build/l3/")[1], line[:12])
