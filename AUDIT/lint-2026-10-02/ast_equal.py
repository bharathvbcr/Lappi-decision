"""L-lint throwaway: is every edited Python file the same program as at the base?

    python AUDIT/lint-2026-10-02/ast_equal.py <base-rev>

For every ``.py`` file that differs between ``<base-rev>`` and the working tree (tracked
files only, from ``git diff``), parse both versions and compare ``ast.dump(ast.parse(...))``.
``ast.dump`` omits line and column attributes by default, and the parser folds implicit
string concatenation into one constant, so a formatting-only edit (a split string, a moved
line break, added parentheses, a removed comment or ``noqa``) compares EQUAL.

A file whose AST differs is checked once more. The AST-changing hunks this lane argues for
in its handoff are listed in apply_hand_edits.py (``JUSTIFIED`` and ``RUFF_I001``). Each of
the file's listed hunks is reverted (its new text must occur exactly once, and is put back
to the old text) and the AST compared again. Equal after that means the listed hunks are
the file's only AST change: the verdict is JUSTIFIED. Anything else is DIFFERS, and fails.

Also prints every added, deleted or non-Python path, so nothing edited goes unexamined.
Exits 1 on any DIFFERS, parse error, deletion or a listed hunk that is not found.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return proc.stdout


def listed_hunks() -> dict[str, list[tuple[str, str]]]:
    spec = importlib.util.spec_from_file_location("hand_edits", HERE / "apply_hand_edits.py")
    if spec is None or spec.loader is None:
        sys.exit("cannot load apply_hand_edits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hunks: dict[str, list[tuple[str, str]]] = {}
    for rel, old, new in [*module.JUSTIFIED, *module.RUFF_I001]:
        hunks.setdefault(rel, []).append((old, new))
    return hunks


def same_ast(a: str | bytes, b: str | bytes) -> bool:
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def names(base: str, diff_filter: str) -> list[str]:
    out = git("diff", "--name-only", f"--diff-filter={diff_filter}", base, "--")
    return [p for p in out.splitlines() if p]


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    hunks = listed_hunks()
    changed, added, gone = names(base, "M"), names(base, "A"), names(base, "DRT")
    py = [p for p in changed if p.endswith(".py")]
    failures = len(gone)
    counts = {"EQUAL": 0, "JUSTIFIED": 0, "DIFFERS": 0}
    print(f"base: {base} ({git('rev-parse', base).strip()})")
    print(f"modified tracked files: {len(changed)} ({len(py)} .py); added: {len(added)}")
    for path in gone:
        print(f"{'DELETED-OR-RENAMED':12s} {path}")
    for path in added:
        print(f"{'ADDED':12s} {path}")
    for path in [p for p in changed if not p.endswith(".py")]:
        print(f"{'NOT-PYTHON':12s} {path}")
    for path in py:
        old_src = git("show", f"{base}:{path}")
        new_src = Path(path).read_text(encoding="utf-8")
        shas = (
            f"old_sha256={hashlib.sha256(old_src.encode()).hexdigest()[:16]} "
            f"new_sha256={hashlib.sha256(new_src.encode()).hexdigest()[:16]}"
        )
        try:
            verdict = "EQUAL" if same_ast(old_src, new_src) else "DIFFERS"
            note = ""
            if verdict == "DIFFERS" and path in hunks:
                reverted = new_src
                for old, new in hunks[path]:
                    if reverted.count(new) != 1:
                        raise LookupError(f"listed hunk not found once: {new[:60]!r}")
                    reverted = reverted.replace(new, old)
                if same_ast(old_src, reverted):
                    verdict = "JUSTIFIED"
                    note = f" (equal once its {len(hunks[path])} listed hunk(s) are reverted)"
        except (SyntaxError, LookupError) as exc:
            print(f"{'ERROR':12s} {path}: {exc}")
            failures += 1
            continue
        counts[verdict] += 1
        failures += verdict == "DIFFERS"
        print(f"{verdict:12s} {path} {shas}{note}")
    stray = sorted(set(hunks) - set(py))
    for path in stray:
        print(f"{'NOT-EDITED':12s} {path} has listed hunks but is unchanged from the base")
        failures += 1
    print(
        f"summary: {len(py)} .py files compared: {counts['EQUAL']} EQUAL, "
        f"{counts['JUSTIFIED']} JUSTIFIED, {counts['DIFFERS']} DIFFERS; {failures} failures"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
