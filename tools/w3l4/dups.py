# ruff: noqa: E501
"""Wave-3 L4: close older navigation duplicates with quoted evidence. Dry run unless --apply.

Every duplicate_of record carries the matching text of the duplicate AND of each canonical, both
extracted programmatically and asserted to be verbatim substrings of the latest record of that id.
A class whose pattern does not match the duplicate is dropped from duplicate_of and reported, never
invented. Appends only, through qd_train.gaps.append_gap.
"""

import json
import re
import sys

from qd_train.gaps import append_gap, current_records

APPLY = "--apply" in sys.argv
LANE = "wave3-l4-gaps-remainder (ft-5c92df4cc91)"
CLOSED_ON = "2026-10-13"

C_LIST = "GAP-LISTAGENTS-UNAVAILABLE-TO-SUBAGENT-2026-10-06"
C_TRUST = "GAP-GITPULSE-REPOSITORY-TRUST-REQUIRED-2026-09-30"
C_STORE = "GAP-DEVMAP-NO-STORE-IN-AGENT-WORKTREE"
C_MALF = "GAP-DEVMAP-DATABASE-MALFORMED-2026-10-03"
C_PARSE = "GAP-V5TRAIN-DEVMAP-CANNOT-NAVIGATE-REAL-FT-RUN-2026-10-02"
C_NAV = "GAP-NAV-LANE-NAVIGATION-TOOLS-PARTIAL-IN-AGENT-WORKTREES-2026-10-03"

CANON = {
    "LIST": C_LIST,
    "TRUST": C_TRUST,
    "NOSTORE": C_STORE,
    "MALF": C_MALF,
    "PARSE": C_PARSE,
    "NAV": C_NAV,
}

#: What a duplicate must say for each class, matched case-sensitively on the joined record text.
PATTERNS = {
    "LIST": r"(?:No|no) ListAgents|ListAgents[^.;]{0,100}?(?:not|unavailable)|not [^.;]{0,60}ListAgents",
    "TRUST": r"REPOSITORY_TRUST_REQUIRED|UNTRUSTED|needs this repository trusted",
    "NOSTORE": r"(?:no|No) (?:readable )?(?:DevMap )?store|has no store",
    "MALF": r"malformed",
    "PARSE": r"over the 5 s budget",
    "NAV": (
        r"not graph-confirmed|graph-confirmed|rg/Read|Read/rg|read directly|by rg|rg over"
        r"|rg and read|found with rg|found by rg|rg, not|Nothing in this lane"
    ),
}

#: What the canonical says for each class (verbatim; asserted against the canonical's text).
CANON_QUOTE = {
    "LIST": "ListAgents is not available to a subagent, so CLAUDE.md rule 3 (ListAgents before changing) cannot be followed inside a lane",
    "TRUST": "every facet returned REPOSITORY_TRUST_REQUIRED, which is not a clean scan",
    "NOSTORE": "No store there: devmap_search with repo_path=<worktree> refused",
    "MALF": "returned 'database disk image is malformed'",
    "PARSE": "exceeded the 5s budget for grammar python",
    "NAV": "lane findings cite rg and Read, not the graph",
}

NAV_QUESTION = (
    "In a lane running in an agent worktree, which of DevMap, GitPulse Insights and ListAgents "
    "can answer the navigation questions CLAUDE.md asks, and what do the lane's claims rest on "
    "when they cannot?"
)
NAV_ANSWER = (
    "Composite canonical for the 2026-09-30..2026-10-03 lane navigation records. Three "
    "independent limits recur, each with its own canonical: DevMap has no store in an agent "
    "worktree (" + C_STORE + ") so the main checkout's index answers and indexes main, not the "
    "branch, with walk_incomplete lower bounds; GitPulse Insights refuses the repository "
    "(" + C_TRUST + ") or its collisions facet scans only part of the worktrees; ListAgents is "
    "not exposed to subagents (" + C_LIST + "). Where a record names more than one, it is closed "
    "as a duplicate of this id and of each component canonical it quotes. The common consequence: "
    "lane findings cite rg and Read, not the graph, and none is graph-confirmed. Nothing here "
    "claims any limit is fixed. Owners per ft-5c92df4c: DevMap capability on the DevCouncil board "
    "(ft-620d41f2 subagents without DevMap tools), GitPulse same-worktree blindness on GitPulse "
    "ft-8e09f101."
)

LIST_ONLY = [
    "GAP-OJAS-ADVICE-NO-LISTAGENTS-2026-10-01",
    "GAP-L-CUDA-NO-LISTAGENTS-2026-10-01",
    "GAP-OJAS-QWEN35-NO-LISTAGENTS-2026-10-01",
    "GAP-L-CUDA-ORACLE-LISTAGENTS-UNAVAILABLE-2026-10-01",
    "GAP-L-ORACLE-NO-LISTAGENTS-2026-10-01",
    "GAP-L-CUDA-M0-NO-LISTAGENTS-2026-10-01",
    "GAP-L-CUDA-GDN-NO-LISTAGENTS-2026-10-01",
    "GAP-L-CUDA-SMALL-LISTAGENTS-UNAVAILABLE-2026-10-01",
    "GAP-L-CUDA-M1-NO-LISTAGENTS-2026-10-01",
    "GAP-RUNGD-FLAGS-NO-LISTAGENTS-2026-10-02",
    "GAP-L-FIXTURE-PINS-LISTAGENTS-UNAVAILABLE-2026-10-02",
    "GAP-L-ROOM-LISTAGENTS-NOT-AVAILABLE-2026-10-02",
    "GAP-V5TRAIN-LISTAGENTS-UNAVAILABLE-2026-10-02",
    "GAP-TRAINED-FAMILIES-LANE-LISTAGENTS-UNAVAILABLE-2026-10-03",
    "GAP-V5-2GPU-LISTAGENTS-UNAVAILABLE-2026-10-03",
]
TRUST_ONLY = [
    "GAP-OJAS-ADVICE-GITPULSE-UNTRUSTED-2026-10-01",
    "GAP-L-CUDA-GITPULSE-UNTRUSTED-2026-10-01",
    "GAP-L-CUDA-ORACLE-GITPULSE-TRUST-2026-10-01",
    "GAP-L-ORACLE-GITPULSE-UNTRUSTED-2026-10-01",
    "GAP-L-CUDA-GDN-GITPULSE-LAPPI-UNTRUSTED-2026-10-01",
    "GAP-L-CUDA-SMALL-GITPULSE-TRUST-2026-10-01",
]
ALL3 = [
    "GAP-P2-PER-FAMILY-NAVIGATION-2026-10-01",
    "GAP-J7-EXPORT-NAVIGATION-2026-10-01",
    "GAP-J7-AVG-SCORING-NAVIGATION-2026-10-01",
    "GAP-PERF-LINEAR-CONTROL-NAVIGATION-2026-10-01",
    "GAP-FABLE-G-REPORT-METRICS-NAVIGATION-2026-10-01",
    "GAP-FABLE-G-NAVIGATION-2026-10-01",
    "GAP-FABLE-G-AVG-NEEDLE-CONTROL-NAV-2026-10-01",
    "GAP-ENS3-SCORING-NAVIGATION-2026-10-01",
    "GAP-WIRE-AGREEMENT-ADMISSION-KINDS-NAVIGATION-2026-10-01",
    "GAP-POST-F-QUEUE-NAVIGATION-2026-10-01",
    "GAP-IDLE-WAITERS-NAVIGATION-2026-10-02",
    "GAP-L-ROOM-J6A-NAVIGATION-2026-10-02",
    "GAP-L-J6G-NAVIGATION-2026-10-02",
    "GAP-L-V5PLAN-NAVIGATION-2026-10-02",
    "GAP-LMETAL-NAVIGATION-2026-10-02",
]
SPECIAL: dict[str, list[str]] = {
    "GAP-L-RECENCY-NAVIGATION-2026-10-02": ["NAV", "NOSTORE", "LIST"],
    "GAP-L-V5FMT-NAVIGATION-2026-10-02": ["NAV", "NOSTORE", "LIST"],
    "GAP-L-V5-RULES-NAVIGATION-2026-10-02": ["NAV", "NOSTORE", "LIST"],
    "GAP-L-PREP2-NAVIGATION-2026-10-02": ["NAV", "NOSTORE", "LIST", "PARSE"],
    "GAP-L-PREP3-NAVIGATION-2026-10-02": ["NAV", "NOSTORE", "LIST", "PARSE"],
    "GAP-FABLE-J6A-RULING-NAVIGATION-2026-10-02": ["NAV", "TRUST", "LIST"],
    "GAP-DEVMAP-UNIFIED-CALLERS-UNRESOLVED-R1-2026-09-30": ["NAV", "NOSTORE"],
    "GAP-DEVMAP-NOT-USED-NEEDLE-OOD-LANE-2026-09-30": ["NAV", "TRUST"],
    "GAP-OOD-ABSTAIN-DECISION-NAV-UNGRAPHED-2026-09-30": ["NAV", "TRUST"],
    "GAP-LEAD-A7-INVESTIGATION-NOT-GRAPH-CONFIRMED-2026-10-03": ["NAV", "MALF"],
    "GAP-V6-SPANCHECK-NAVIGATION-DEVMAP-UNAVAILABLE-2026-10-03": [
        "NAV",
        "NOSTORE",
        "MALF",
        "LIST",
    ],
}
#: Same-worktree blindness stated without the word ListAgents: matched by its own text.
OVERRIDE = {
    ("GAP-L-CUDA-M1-GITPULSE-UNTRUSTED-2026-10-01", "LIST"): (
        r"GitPulse cannot see the three CUDA lanes editing one ojas working tree",
        "Same-worktree collisions are invisible to lanes",
    ),
}
SPECIAL["GAP-L-CUDA-M1-GITPULSE-UNTRUSTED-2026-10-01"] = ["TRUST", "LIST"]


def joined(rec: dict) -> str:
    return " ".join(str(v) for k, v in rec.items() if isinstance(v, str))


def window(text: str, pattern: str) -> str | None:
    hit = re.search(pattern, text)
    if hit is None:
        return None
    start = max(0, hit.start() - 90)
    end = min(len(text), hit.end() + 130)
    snippet = text[start:end].strip()
    assert snippet in text
    return snippet


def build_specs() -> dict[str, list[str]]:
    specs: dict[str, list[str]] = {}
    for gid in LIST_ONLY:
        specs[gid] = ["LIST"]
    for gid in TRUST_ONLY:
        specs[gid] = ["TRUST"]
    for gid in ALL3:
        specs[gid] = ["NAV", "NOSTORE", "TRUST", "LIST"]
    specs.update(SPECIAL)
    return specs


def nav_canonical_record() -> dict:
    return {
        "id": C_NAV,
        "opened": "2026-10-13",
        "lane": LANE,
        "owner": "DevCouncil ft-620d41f2 (DevMap tools in subagents) / GitPulse ft-8e09f101 (same-worktree blindness)",
        "status": "open",
        "tool": "DevMap, GitPulse Insights, ListAgents (as recorded by the duplicates closed against this id)",
        "question": NAV_QUESTION,
        "answer": NAV_ANSWER,
        "action_required": (
            "None for this lane. The tool-side fixes live on the boards named in owner; a lane "
            "that cannot navigate by the graph keeps recording its own gap rather than citing this one."
        ),
    }


def main() -> None:
    cur = current_records()
    if C_NAV not in cur:
        nav = nav_canonical_record()
        assert CANON_QUOTE["NAV"] in nav["answer"], "NAV quote must be verbatim in the canonical"
        print("create", C_NAV)
        if APPLY:
            append_gap(nav)
            cur = current_records()
        else:
            cur = {**cur, C_NAV: nav}
    canon_text = {cls: joined(cur[gid]) for cls, gid in CANON.items()}
    for cls, quote in CANON_QUOTE.items():
        assert quote in canon_text[cls], (cls, quote)

    done = 0
    dropped: list[tuple[str, str]] = []
    for gid, classes in build_specs().items():
        rec = cur[gid]
        if rec["status"] in ("closed", "resolved", "resolved-with-residual", "closed-no-defect"):
            print("skip already closed", gid)
            continue
        text = joined(rec)
        matched: dict[str, str] = {}
        canon_used: dict[str, str] = {}
        for cls in classes:
            override = OVERRIDE.get((gid, cls))
            pattern = override[0] if override else PATTERNS[cls]
            snippet = window(text, pattern)
            if snippet is None:
                dropped.append((gid, cls))
                continue
            matched[cls] = snippet
            if override:
                canon_used[CANON[cls]] = override[1]
                assert override[1] in canon_text[cls] or override[1] in json.dumps(cur[CANON[cls]])
            else:
                canon_used[CANON[cls]] = CANON_QUOTE[cls]
        if not matched:
            print("NO MATCH, not closed:", gid)
            continue
        targets = [CANON[c] for c in classes if c in matched]
        unmatched = [c for c in classes if c not in matched]
        question = rec.get("question") or rec.get("finding")
        answer = (
            f"Duplicate of each of {targets}: the text of this record and of each canonical, "
            "quoted below, says the same tooling limit (text match on the quoted sentences, not "
            "a keyword-only match). Nothing here claims any limit is fixed, and this record's "
            "lane-specific mitigation stays readable in its earlier line (the ledger is "
            "append-only). Re-open the canonical ids, not this one."
        )
        if unmatched:
            answer += (
                f" Classes {unmatched} were considered and NOT matched in this record's text, so "
                "no duplicate_of is claimed for them."
            )
        out = {
            "id": gid,
            "question": question,
            "status": "closed",
            "closure_kind": "consolidated-duplicate",
            "duplicate_of": targets,
            "closed": CLOSED_ON,
            "lane": LANE,
            "opened": rec.get("opened", rec.get("date", "")),
            "duplicate_text": matched,
            "canonical_text": canon_used,
            "answer": answer,
        }
        if not out["opened"]:
            del out["opened"]
        done += 1
        if APPLY:
            append_gap(out)
        else:
            print(gid, "->", targets)
    print("closed", done, "dropped classes", dropped)


main()
