"""Lane L-v5plan's gap records (2026-10-02), appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. Running it again would append duplicates;
it refuses if any of its ids is already in gaps.jsonl.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps  # noqa: E402

LANE = "L-v5plan"
OPENED = "2026-10-02"
TOOL = "Read/rg/jq/git in worktree agent-aebccb7350fcc159c; AUDIT/v5-plan-2026-10-02/*.py"

RECORDS = [
    {
        "id": "GAP-L-V5PLAN-NAVIGATION-2026-10-02",
        "question": "Could the v5 plan be navigated by DevMap and checked by GitPulse and ListAgents?",
        "answer": (
            "No. DevMap has no store for this worktree (no .devmap/codeintel/devmap.sqlite); the main "
            "checkout's store is generation 3043, degraded_reason 'stored extraction payload is "
            "obsolete; rebuild with the current analyzer', is_fresh false. gitpulse_insights returned "
            "REPOSITORY_TRUST_REQUIRED on every facet. ListAgents is not exposed in this session. "
            "Nothing in HANDOFF/v5-plan-2026-10-02.md is graph-confirmed: it comes from Read, rg, jq, "
            "git in this worktree and the scripts in AUDIT/v5-plan-2026-10-02/."
        ),
        "owner": "lead",
        "status": "open",
        "action_required": "Trust the repository in GitPulse and rebuild DevMap with the current analyzer.",
    },
    {
        "id": "GAP-L-V5PLAN-OWN-REPO-COMMIT-BODIES-NOT-MEASURED-2026-10-02",
        "question": (
            "How many commit bodies (and how much prose) do the human's own repositories other than "
            "Lappi-decision hold, for v5's prose-noul route (i)?"
        ),
        "answer": (
            "Not measured. The session harness refuses git against any tree but this worktree "
            "(git -C and git --git-dir both blocked), so only Lappi-decision was measured: 630 "
            "non-merge commits, 487 with >= 10 words of body, 1,656 body paragraphs >= 25 words, 7 "
            "question sentences. The other 30 own repos (owner bharathvbcr) were inventoried by file "
            "reads only: 211 READMEs, 22,007 prose paragraphs >= 25 words, 592 question sentences "
            "(AUDIT/v5-plan-2026-10-02/own_repo_inventory.json, own_repo_questions.json). That "
            "supply already exceeds route (i)'s 2,000-row target, so the plan does not depend on it."
        ),
        "owner": "lead",
        "status": "open",
        "action_required": (
            "Optional: a lane started in each repo root (or the human) runs `git log --no-merges "
            "--format=%B` there if commit bodies are wanted in a later noul build."
        ),
    },
    {
        "id": "GAP-V5-CHECKPOINT-RETENTION-AND-OOD-ONLY-SCORING-NEED-CODE-2026-10-02",
        "question": (
            "Can v5 run --checkpoint-every 1000 with per-checkpoint OOD scoring (Fable Q1(b)) using "
            "today's flags?"
        ),
        "answer": (
            "No. --checkpoint-dir writes one file per (tag, seed), rewritten in place "
            "(tools/real_ft_run.py:8700-8714), so only the last checkpoint survives; "
            "--score-checkpoint requires --score-val (help at :8862-8880), which decodes the whole val "
            "set rather than the 180-case OOD suite. A retention flag (step-tagged tower + span-head "
            "copies) and an OOD-only checkpoint scoring mode are needed. Disk: a full checkpoint is "
            "~25 GB (HANDOFF/rungd-flags-2026-10-02.md:261), a tower ~3.51 GiB "
            "(HANDOFF/j7-avg-scoring-2026-10-01.md:129); Fable's '10 x 8 GB' matches neither."
        ),
        "owner": "lead",
        "status": "open",
        "action_required": "Lane L-v5-train (HANDOFF/v5-plan-2026-10-02.md section 6).",
    },
    {
        "id": "GAP-V5-CONTEXT-FIRST-LINE-STARTS-DO-NOT-SEE-THE-QUESTION-2026-10-02",
        "question": (
            "What does the human's context-first order (human-decisions.md item 8) cost the span "
            "pointer?"
        ),
        "answer": (
            "Unmeasured. The pointer head scores line-start hidden states against the answer "
            "position's query; with the question after the context, causal attention means those "
            "line-start states no longer see the question (they still see task and route, which stay "
            "before the context in the plan). qa.answer_span and the needle suite are the span "
            "consumers; val_top1.span is the number to watch. Recorded as a report-only expectation "
            "in campaign/v5-preregistered.DRAFT.json format.expected_cost_report_only."
        ),
        "owner": "lead",
        "status": "open",
        "action_required": "Read val_top1.span and the needle depth buckets on v5's rows; report, no kill criterion.",
    },
    {
        "id": "GAP-V5-G6-UNSEEN-LANGUAGE-HAS-NO-LOCAL-DISTINCT-SUPPLY-2026-10-02",
        "question": (
            "Can v5 add more distinct unseen languages for the noul rows (Fable: G6, 'more distinct "
            "unseen languages, not more repeats') without a download?"
        ),
        "answer": (
            "Not with volume. The OOD suite's unseen-language cases are c, java, ruby, haskell, sql, "
            "lua (python/qd_train/ood.py:66-109); v3b's training templates are csharp, elixir, "
            "kotlin, php, scala, shell; the mutation pool is go/python/rust/typescript; the needle "
            "suite builds swift cases (python/qd_train/needle.py:225), so swift as 'unseen -> Z' "
            "would teach abstention on a needle-gate language. Own repos hold kotlin (1,568 files, "
            "already a template language), javascript/svelte/vue (TypeScript-adjacent), metal "
            "(C-like) and two or three files each of dart, r, scala, php, csharp, objc. More distinct "
            "languages means a download (e.g. bigcode/commitpackft per-language files, dataset "
            "licence mit with the per-row licence filter); sizes are not known locally."
        ),
        "owner": "human",
        "status": "open",
        "action_required": (
            "Human: keep unseen-language at v4's 834 template rows in v5, or allow a metadata-only "
            "Hub listing so the lead can bring filenames and sizes for a download yes."
        ),
    },
    {
        "id": "GAP-V5-FORMAT-CHANGE-REFUSES-V4-EXPORT-ON-MAIN-2026-10-02",
        "question": "When can the context-first format change land on main?",
        "answer": (
            "Only in the v5 build commit, after the last v4 reader on main has run. Once render.rs "
            "and release.rs carry prompt format 2, main's qd-runtime refuses F's v4 export by design, "
            "and render_contract's goldens need both renderers in one commit. Readers to clear first: "
            "L-metal's ledger/mac-qd-metal-2026-10-02.jsonl rows against the v4 export, any J7'/ens3 "
            "export or scoring built from a main checkout, and the excluded re-score for F seeds 1-2 "
            "if rebuilt from main. qd-export must stamp prompt_format from the checkpoint's recipe, "
            "never the current constant."
        ),
        "owner": "lead",
        "status": "open",
        "action_required": "The lead confirms the reader list is empty before cutting the v5 build commit.",
    },
]


def main() -> int:
    existing = {r["id"] for r in read_gaps()}
    clash = sorted(r["id"] for r in RECORDS if r["id"] in existing)
    if clash:
        raise SystemExit(f"already recorded, refusing to append twice: {clash}")
    for rec in RECORDS:
        line = append_gap({**rec, "opened": OPENED, "lane": LANE, "tool": TOOL})
        print(line[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
