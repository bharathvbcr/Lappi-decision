"""L-lint throwaway: does any record pin the bytes of a file the lint gate flags?

For every file listed in files-with-findings.txt this computes the file's sha256 and its
git blob id at HEAD, then searches every file under AUDIT/, HANDOFF/, ledger/, campaign/
and gaps.jsonl (ignore rules not honoured; this lane's own AUDIT/lint-2026-10-02/ is
skipped) for

  * the first 8 hex characters of the sha256 (records abbreviate as ``677c81c0...``),
  * the first 7 hex characters of the git blob id,
  * a line naming the file's basename together with a word that signals a byte pin
    (sha, byte, identical, unfixed, ruff, verbatim, pinned).

Run from the worktree root with the project venv's python. Prints one block per file.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
OWN = ROOT / "AUDIT" / "lint-2026-10-02"
SEARCH_ROOTS = [ROOT / "AUDIT", ROOT / "HANDOFF", ROOT / "ledger", ROOT / "campaign"]
SEARCH_FILES = [ROOT / "gaps.jsonl"]
PIN_WORDS = re.compile(r"sha|byte|identical|unfix|ruff|verbatim|pinn", re.IGNORECASE)


def corpus() -> list[tuple[Path, list[str]]]:
    out: list[tuple[Path, list[str]]] = []
    paths = list(SEARCH_FILES)
    for base in SEARCH_ROOTS:
        paths.extend(p for p in base.rglob("*") if p.is_file())
    for p in paths:
        if OWN in p.parents:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out.append((p, text.splitlines()))
    return out


def blob_id(rel: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", f"HEAD:{rel}"], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def main() -> int:
    listing = (OWN / "files-with-findings.txt").read_text().split()
    docs = corpus()
    print(f"searched {len(docs)} text files")
    for rel in listing:
        data = (ROOT / rel).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        blob = blob_id(rel)
        base = Path(rel).name
        hits: list[str] = []
        for path, lines in docs:
            if path == ROOT / rel:
                continue
            for n, line in enumerate(lines, 1):
                why = []
                if sha[:8] in line:
                    why.append("sha256[:8]")
                if blob[:7] in line:
                    why.append("blob[:7]")
                if base in line and PIN_WORDS.search(line):
                    why.append("name+pin-word")
                if why:
                    hits.append(f"  {path.relative_to(ROOT)}:{n} [{','.join(why)}] {line[:220]}")
        print(f"=== {rel} sha256={sha} blob={blob} hits={len(hits)}")
        for h in hits:
            print(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
