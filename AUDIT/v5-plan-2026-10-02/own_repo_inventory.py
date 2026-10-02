"""Evidence script for lane L-v5plan (2026-10-02). Throwaway accounting, never shipped.

Inventory of the human's own repositories under /Users/bharath/Code as candidate sources for
v5's prose-noul route (i) (human-decisions.md item 5) and for more own-repo code (item 9).

Reads files only. It runs no git command: the session harness refuses git against any tree
but this lane's worktree (GAP-L-V5PLAN-NAVIGATION-2026-10-02), so commit counts and commit
bodies for other repos are NOT measured here. Ownership is read from each repo's
.git/config remote URLs: a repo is "own" when a remote names github owner bharathvbcr.

Per repo it reports: remotes, owner verdict, LICENSE file presence, README files and their
word counts, other Markdown/reST docs, prose paragraphs (blank-line separated blocks of
>= 25 words that are not code fences, tables or lists), and code files per language
(extension census, vendored/build directories excluded). Nothing is ingested.
"""

import collections
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path("/Users/bharath/Code")
OUT = Path(__file__).with_name("own_repo_inventory.json")
OWNER = "bharathvbcr"
SKIP_DIRS = {
    ".git", "node_modules", "vendor", ".venv", "venv", "env", "target", "build", "dist",
    "external", "third_party", "third-party", "site-packages", "__pycache__", ".next",
    "Pods", "DerivedData", ".build", ".gradle", ".idea", ".cache", "checkpoints", "worktrees",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "coverage", "out", "bin", "obj",
}
LANG_EXT = {
    ".go": "go", ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".rs": "rust",
    ".swift": "swift", ".kt": "kotlin", ".java": "java", ".c": "c", ".h": "c", ".cc": "cpp",
    ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php",
    ".scala": "scala", ".dart": "dart", ".lua": "lua", ".sh": "shell", ".js": "javascript",
    ".jsx": "javascript", ".m": "objc", ".mm": "objc", ".metal": "metal", ".cu": "cuda",
    ".zig": "zig", ".ex": "elixir", ".hs": "haskell", ".jl": "julia", ".r": "r", ".R": "r",
    ".sql": "sql", ".vue": "vue", ".svelte": "svelte",
}
MAX_FILE_BYTES = 2_000_000
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def find_repos() -> list[Path]:
    repos = []
    for depth in range(1, 5):
        for git in CODE_ROOT.glob("/".join(["*"] * depth) + "/.git"):
            p = git.parent
            if any(part in {"node_modules", "worktrees"} for part in p.parts):
                continue
            repos.append(p)
    return sorted(set(repos))


def remotes_of(repo: Path) -> list[str]:
    git = repo / ".git"
    cfg = None
    if git.is_dir():
        cfg = git / "config"
    elif git.is_file():
        line = git.read_text().strip()
        if line.startswith("gitdir:"):
            gd = (repo / line.split(":", 1)[1].strip()).resolve()
            # a linked worktree's config lives in the common dir
            common = gd / "commondir"
            base = (gd / common.read_text().strip()).resolve() if common.exists() else gd
            cfg = base / "config"
    if cfg is None or not cfg.exists():
        return []
    return [
        m.group(1).strip()
        for m in re.finditer(r"^\s*url\s*=\s*(.+)$", cfg.read_text(errors="replace"), re.M)
    ]


def owner_of(urls: list[str]) -> str:
    owners = set()
    for u in urls:
        m = re.search(r"github\.com[:/]([^/]+)/", u)
        if m:
            owners.add(m.group(1))
    if not urls:
        return "no-remote"
    if OWNER in owners:
        return "own"
    return "other:" + ",".join(sorted(owners)) if owners else "other:non-github"


def prose_paragraphs(text: str) -> tuple[int, int]:
    n = 0
    words = 0
    in_fence = False
    block: list[str] = []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            block = []
            continue
        if in_fence:
            continue
        if line.strip() == "":
            if block:
                joined = " ".join(block)
                w = len(WORD.findall(joined))
                is_list_or_table = all(
                    b.lstrip().startswith(("|", "-", "*", "#", ">", "1.", "<")) for b in block
                )
                if w >= 25 and not is_list_or_table:
                    n += 1
                    words += w
            block = []
        else:
            block.append(line)
    return n, words


def walk(repo: Path, nested: set[Path]) -> dict:
    readmes = []
    docs = 0
    doc_words = 0
    paras = 0
    para_words = 0
    langs = collections.Counter()
    lang_bytes = collections.Counter()
    licence_files = []
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
                continue
            name = e.name
            try:
                size = e.stat().st_size
            except OSError:
                continue
            if d == repo and name.upper().startswith(("LICENSE", "LICENCE", "COPYING")):
                licence_files.append(name)
            suf = e.suffix
            if suf in (".md", ".mdx", ".rst") and size <= MAX_FILE_BYTES:
                text = e.read_text(errors="replace")
                w = len(WORD.findall(text))
                p, pw = prose_paragraphs(text)
                if name.upper().startswith("README"):
                    readmes.append({"path": str(e.relative_to(repo)), "words": w, "prose_paragraphs": p})
                else:
                    docs += 1
                    doc_words += w
                paras += p
                para_words += pw
            elif suf in LANG_EXT and size <= MAX_FILE_BYTES:
                langs[LANG_EXT[suf]] += 1
                lang_bytes[LANG_EXT[suf]] += size
    return {
        "licence_files": sorted(licence_files),
        "readmes": sorted(readmes, key=lambda r: r["path"]),
        "readme_count": len(readmes),
        "readme_words": sum(r["words"] for r in readmes),
        "other_doc_files": docs,
        "other_doc_words": doc_words,
        "prose_paragraphs_ge25w": paras,
        "prose_paragraph_words": para_words,
        "code_files_by_language": dict(langs.most_common()),
        "code_bytes_by_language": dict(lang_bytes.most_common()),
    }


def main() -> int:
    repos = find_repos()
    out = {"code_root": str(CODE_ROOT), "owner": OWNER, "repos": []}
    repo_set = set(repos)
    for repo in repos:
        nested = {r for r in repo_set if r != repo and repo in r.parents}
        urls = remotes_of(repo)
        rec = {"repo": str(repo), "remotes": urls, "owner": owner_of(urls)}
        rec.update(walk(repo, nested))
        out["repos"].append(rec)
    tot = collections.Counter()
    for r in out["repos"]:
        key = r["owner"] if r["owner"] in ("own", "no-remote") else "other"
        tot[f"{key}:repos"] += 1
        tot[f"{key}:readmes"] += r["readme_count"]
        tot[f"{key}:readme_words"] += r["readme_words"]
        tot[f"{key}:other_doc_files"] += r["other_doc_files"]
        tot[f"{key}:prose_paragraphs_ge25w"] += r["prose_paragraphs_ge25w"]
        tot[f"{key}:prose_paragraph_words"] += r["prose_paragraph_words"]
        for lang, n in r["code_files_by_language"].items():
            tot[f"{key}:code_files:{lang}"] += n
    out["totals"] = dict(sorted(tot.items()))
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["totals"], indent=1, sort_keys=True))
    for r in out["repos"]:
        print(r["owner"], r["repo"], "readmes", r["readme_count"], "paras", r["prose_paragraphs_ge25w"],
              "licence", r["licence_files"], "code", dict(list(r["code_files_by_language"].items())[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
