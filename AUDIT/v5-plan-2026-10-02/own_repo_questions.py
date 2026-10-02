"""Evidence script for lane L-v5plan (2026-10-02). Throwaway accounting, never shipped.

Question-form supply in the human's own repositories' prose, for v5's prose-noul route (i).
Reads own_repo_inventory.json (beside this file) for the repos whose remote names github owner
bharathvbcr, walks their Markdown/reST with the same directory exclusions, and counts distinct
question sentences: a sentence ending in '?', 6-60 words, outside code fences and tables.
Also counts them in this repo's own commit bodies, read from a file produced by
`git log --all --no-merges --format=%B%x1e` in this worktree (the only tree the harness lets
git read). Nothing is ingested; only counts are written.
"""

import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INV = HERE / "own_repo_inventory.json"
COMMITS = HERE / "lappi_commit_bodies.txt"
OUT = HERE / "own_repo_questions.json"
SKIP_DIRS = {
    ".git", "node_modules", "vendor", ".venv", "venv", "env", "target", "build", "dist",
    "external", "third_party", "third-party", "site-packages", "__pycache__", ".next",
    "Pods", "DerivedData", ".build", ".gradle", ".idea", ".cache", "checkpoints", "worktrees",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "coverage", "out", "bin", "obj",
}
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")
SENT = re.compile(r"[^.!?\n]*\?")


def questions(text: str) -> set[str]:
    found: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or s.startswith("|") or s.startswith("    "):
            continue
        for m in SENT.finditer(s):
            q = m.group(0).strip(" -*#>`")
            n = len(WORD.findall(q))
            if 6 <= n <= 60:
                found.add(q)
    return found


def main() -> int:
    inv = json.loads(INV.read_text())
    repos = [Path(r["repo"]) for r in inv["repos"] if r["owner"] == "own"]
    all_repo_paths = {Path(r["repo"]) for r in inv["repos"]}
    per_repo = {}
    total: set[str] = set()
    for repo in repos:
        nested = {r for r in all_repo_paths if r != repo and repo in r.parents}
        qs: set[str] = set()
        stack = [repo]
        while stack:
            d = stack.pop()
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            for e in entries:
                if e.is_symlink():
                    continue
                if e.is_dir():
                    if e.name in SKIP_DIRS or e.name.startswith(".") or e in nested:
                        continue
                    stack.append(e)
                elif e.suffix in (".md", ".mdx", ".rst") and e.stat().st_size <= 2_000_000:
                    qs |= questions(e.read_text(errors="replace"))
        per_repo[str(repo)] = len(qs)
        total |= qs
    commit_qs: set[str] = set()
    commits = 0
    if COMMITS.exists():
        bodies = [b for b in COMMITS.read_text(errors="replace").split("\x1e") if b.strip()]
        commits = len(bodies)
        for b in bodies:
            commit_qs |= questions(b)
    out = {
        "own_repos": len(repos),
        "distinct_doc_questions_6_to_60_words": len(total),
        "per_repo": dict(sorted(per_repo.items(), key=lambda kv: -kv[1])),
        "lappi_commit_bodies_read": commits,
        "lappi_commit_body_distinct_questions": len(commit_qs),
        "doc_question_length_words_histogram": dict(
            sorted(
                collections.Counter(
                    min(60, 10 * (len(WORD.findall(q)) // 10)) for q in total
                ).items()
            )
        ),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
