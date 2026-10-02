"""L-lint: the hand edits, as exact text replacements, applied under a root directory.

    python AUDIT/lint-2026-10-02/apply_hand_edits.py <root>

Runs after ``ruff check --fix`` (safe fixes only) and split_long_strings.py. Every edit
must match exactly once in its file or the script exits non-zero before writing anything.

FORMAT edits only move line breaks inside brackets or add parentheses around an
expression, so the AST is unchanged; ast_equal.py checks it. JUSTIFIED edits change the
AST without changing what the program does, and each is argued in
HANDOFF/lint-2026-10-02.md:

* B007 (two files): the first of two loops over the same dict binds a value it never
  reads; the variable is renamed with a leading underscore. The second loop rebinds the
  original name before any read, so no later read sees a different value.
* B905: ``zip(..., strict=False)`` spells out zip's default.
* RUF005: ``[*xs, ""]`` builds the same new list as ``xs + [""]`` for the list that
  ``str.splitlines`` returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

V5 = "AUDIT/v5-plan-2026-10-02/"

RETURN_OLD = (
    '    return p.returncode, result[-1] if result else "(no test result: did not compile)", '
    "failed, compiled\n"
)
RETURN_NEW = (
    "    return (\n"
    "        p.returncode,\n"
    '        result[-1] if result else "(no test result: did not compile)",\n'
    "        failed,\n"
    "        compiled,\n"
    "    )\n"
)

FORMAT: list[tuple[str, str, str]] = [
    ("AUDIT/j6a-rule-2026-10-02/mutations.py", RETURN_OLD, RETURN_NEW),
    ("AUDIT/succ-room-2026-10-02/mutations.py", RETURN_OLD, RETURN_NEW),
    (
        V5 + "apply_review_amendments.py",
        '    "exclude the offending own-prose or contrast row and rebuild; never touch a val or '
        'held-out row",\n',
        '    "exclude the offending own-prose or contrast row and rebuild; never touch a val or "\n'
        '    "held-out row",\n',
    ),
    (
        V5 + "clinc_oos_rekey.py",
        "        \"upstream_rows\": {f\"{k[0]}:{'oos' if k[1] else 'in_scope'}\": v "
        "for k, v in sorted(upstream.items())},\n",
        '        "upstream_rows": {\n'
        "            f\"{k[0]}:{'oos' if k[1] else 'in_scope'}\": v "
        "for k, v in sorted(upstream.items())\n"
        "        },\n",
    ),
    (
        V5 + "count_v4_manifests.py",
        '                {"split": k[0], "family": k[1], "source": k[2], "licence": k[3], '
        '"kind": k[4], "rows": v}\n',
        "                {\n"
        '                    "split": k[0],\n'
        '                    "family": k[1],\n'
        '                    "source": k[2],\n'
        '                    "licence": k[3],\n'
        '                    "kind": k[4],\n'
        '                    "rows": v,\n'
        "                }\n",
    ),
    (
        V5 + "own_repo_inventory.py",
        '                    readmes.append({"path": str(e.relative_to(repo)), "words": w, '
        '"prose_paragraphs": p})\n',
        "                    readmes.append(\n"
        '                        {"path": str(e.relative_to(repo)), "words": w, '
        '"prose_paragraphs": p}\n'
        "                    )\n",
    ),
    (
        V5 + "own_repo_inventory.py",
        '        print(r["owner"], r["repo"], "readmes", r["readme_count"], "paras", '
        'r["prose_paragraphs_ge25w"],\n'
        '              "licence", r["licence_files"], "code", '
        'dict(list(r["code_files_by_language"].items())[:4]))\n',
        "        print(\n"
        '            r["owner"], r["repo"], "readmes", r["readme_count"], "paras",\n'
        '            r["prose_paragraphs_ge25w"], "licence", r["licence_files"], "code",\n'
        '            dict(list(r["code_files_by_language"].items())[:4]),\n'
        "        )\n",
    ),
    (
        V5 + "own_repo_questions.py",
        "            sorted(collections.Counter(min(60, 10 * (len(WORD.findall(q)) // 10)) "
        "for q in total).items())\n",
        "            sorted(\n"
        "                collections.Counter(\n"
        "                    min(60, 10 * (len(WORD.findall(q)) // 10)) for q in total\n"
        "                ).items()\n"
        "            )\n",
    ),
    (
        V5 + "v4_token_accounting.py",
        '    noul_cats = [k for k in by if k.startswith("defect.") and k not in '
        '("defect.composed", "defect.qdm")]\n',
        "    noul_cats = [\n"
        '        k for k in by if k.startswith("defect.") and k not in '
        '("defect.composed", "defect.qdm")\n'
        "    ]\n",
    ),
    (
        V5 + "v4_token_accounting.py",
        "    add_noul_tokens = add_noul_rows * (noul_seq / noul_rows) * (noul_tok / noul_seq) "
        "if noul_rows else 0\n",
        "    add_noul_tokens = (\n"
        "        add_noul_rows * (noul_seq / noul_rows) * (noul_tok / noul_seq) "
        "if noul_rows else 0\n"
        "    )\n",
    ),
    (
        V5 + "v4_token_accounting.py",
        "    v5_tokens = non_comp_tokens + v5_comp_tokens + add_noul_tokens + "
        "clinc_delta_tokens + eval_delta_tokens\n",
        "    v5_tokens = (\n"
        "        non_comp_tokens + v5_comp_tokens + add_noul_tokens + clinc_delta_tokens "
        "+ eval_delta_tokens\n"
        "    )\n",
    ),
    (
        V5 + "v5_accounting.py",
        '        if r["family"] == "code.defect_class" and r["kind"] == "noul" and '
        'r["licence"] == "cc-by-sa-4.0"\n',
        '        if r["family"] == "code.defect_class"\n'
        '        and r["kind"] == "noul"\n'
        '        and r["licence"] == "cc-by-sa-4.0"\n',
    ),
]

JUSTIFIED: list[tuple[str, str, str]] = [
    (
        "AUDIT/j6g-rule-2026-10-02/append_gaps_staged.py",
        "for r in read_gaps():\n"
        '    current[r.get("id")] = r\n'
        "for gap, answer in RESOLUTIONS.items():\n",
        "for r in read_gaps():\n"
        '    current[r.get("id")] = r\n'
        "for gap, _answer in RESOLUTIONS.items():\n",
    ),
    (
        "AUDIT/prep2-2026-10-02/append_gaps_clinc_ruling.py",
        '        sys.exit(f"{g[\'id\']} already exists")\n'
        "for gid, u in UPDATES.items():\n"
        "    if gid not in current:\n",
        '        sys.exit(f"{g[\'id\']} already exists")\n'
        "for gid, _u in UPDATES.items():\n"
        "    if gid not in current:\n",
    ),
    (
        V5 + "v4_token_accounting.py",
        "    for s, n in zip(seqs, lengths.tolist()):\n",
        "    for s, n in zip(seqs, lengths.tolist(), strict=False):\n",
    ),
    (
        V5 + "own_repo_inventory.py",
        '    for line in text.splitlines() + [""]:\n',
        '    for line in [*text.splitlines(), ""]:\n',
    ),
]


#: Not applied here: ``ruff check --fix`` makes this one (I001, a safe fix). It is listed so
#: that ast_equal.py can revert it and show it is the file's only AST change. It moves one
#: first-party import above another; neither module's import has a side effect the other
#: depends on.
RUFF_I001: list[tuple[str, str, str]] = [
    (
        "python/tests/test_contrast_rows.py",
        "from qd_train.artifacts import ContrastRows, ShardContractViolation, ShardHeader\n"
        "from qd_train.contrast import (\n"
        "    CONTRAST_PATHS,\n"
        "    ContrastShortfall,\n"
        "    apply_contrast,\n"
        "    contrast_order_key,\n"
        "    contrast_rows_sha256,\n"
        "    derive_contrast_rows,\n"
        ")\n"
        "from qd_train.containment_strip import STRIP_RULE, STRIP_VERSION\n",
        "from qd_train.artifacts import ContrastRows, ShardContractViolation, ShardHeader\n"
        "from qd_train.containment_strip import STRIP_RULE, STRIP_VERSION\n"
        "from qd_train.contrast import (\n"
        "    CONTRAST_PATHS,\n"
        "    ContrastShortfall,\n"
        "    apply_contrast,\n"
        "    contrast_order_key,\n"
        "    contrast_rows_sha256,\n"
        "    derive_contrast_rows,\n"
        ")\n",
    ),
]


def main() -> int:
    root = Path(sys.argv[1])
    staged: dict[Path, str] = {}
    for rel, old, new in FORMAT + JUSTIFIED:
        path = root / rel
        text = staged.get(path, path.read_text(encoding="utf-8"))
        n = text.count(old)
        if n != 1:
            sys.exit(f"{rel}: expected exactly one match, found {n}: {old[:80]!r}")
        staged[path] = text.replace(old, new)
    for path, text in staged.items():
        path.write_text(text, encoding="utf-8")
        print(f"edited {path}")
    print(f"{len(FORMAT)} format edits, {len(JUSTIFIED)} justified edits, {len(staged)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
