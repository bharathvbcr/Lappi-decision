"""Lane L-prep2's Part B gap records, appended once through qd_train.gaps.append_gap.

Kept as the record of exactly what was appended. It refuses to append a second copy of any.
Numbers: AUDIT/prep2-2026-10-02/rates-subsample.json (family_rates.py over the six scans) and
the scans' attestations under /Users/bharath/qd-campaign/prep2-2026-10-02/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qd_train.gaps import append_gap, read_gaps

CLINC = "GAP-CONTAINMENT-CONSTANT-QUESTION-DRIVES-CLINC-HITS-2026-10-02"
PREMISES = "GAP-CONTAINMENT-STRIP-RULE-PREMISES-2026-10-02"
SUBSAMPLE = "GAP-PREP-SUBSAMPLE-HASH-UNRECORDED-2026-10-02"

existing = read_gaps()
current = [r for r in existing if r.get("id") == CLINC]
if not current:
    sys.exit(f"{CLINC} is not in gaps.jsonl")
if any(r.get("lane") == "L-prep2" for r in current):
    sys.exit(f"{CLINC} already has an L-prep2 record; refusing a duplicate")
if any(r.get("id") in (PREMISES, SUBSAMPLE) for r in existing):
    sys.exit("an L-prep2 Part B record already exists; refusing a duplicate")

append_gap(
    {
        "id": CLINC,
        "opened": "2026-10-02",
        "lane": "L-prep2",
        "owner": "human",
        "status": "open",
        "tool": (
            "qd_train.containment_strip.strip_template (369278e) via tools/containment_scan.py "
            "with and without --no-template-strip on re-created 1/2/5% subsamples; "
            "AUDIT/prep2-2026-10-02/family_rates.py -> rates-subsample.json (686cdca)"
        ),
        "question": current[-1]["question"],
        "answer": (
            "The strip Fable ruled (post-F 3(a)) is implemented and measured; as written it "
            "does not fix CLINC, it moves the template to intent.within_domain. Share of "
            "distinct CLINC train utterance keys excluded (union of intent.*), unstripped -> "
            "stripped: 1% 8/246 3.3% -> 155/246 63.0%; 2% 39/464 8.4% -> 293/464 63.1%; 5% "
            "338/1089 31.0% -> 721/1089 66.2%. Every stripped CLINC key is hit by an "
            "intent.within_domain row of its own (5%: 721 of 721; 7,727 enforced pairs). "
            "What the family-wide rule strips: the question line in every family (it is the "
            "family's constant description), intent.domain's 10 domains, yes/no of "
            "intent.in_scope and qa.answerability, code.defect_class's 4 classes. What it "
            "leaves: intent.within_domain asks over ONE domain's intents (10 distinct lists, "
            "largest shared by 136 of 1,222 rows at 5%), and intent.classification over 16 "
            "random sorted intents per row. With the question gone, a within_domain row is a "
            "short utterance plus its domain's ~15 one-word intents, so any two same-domain "
            "rows share the list's 8 internal 8-grams of 13-16, over 0.5: e.g. train 'how do "
            "i set up direct deposit for my fifth third account' hits val 'what are my tax "
            "costs' (8 of 13) and val 'will i pay over $500 in federal taxes' (8 of 16), "
            "sharing only the work domain's list. Elsewhere the strip works: in_scope's "
            "template hits vanish; MMLU 0/128, 25/278 9.0%, 47/635 7.4% -> 0, 0, 1/635 0.2%; "
            "code.defect_class 1.6/1.3/1.5% -> 1.0/1.1/1.1%; CSQA and SQuAD answer_span 0 "
            "throughout. Cost: intent.domain and intent.in_scope rows become the bare "
            "utterance; at 5% 451 of 1,089 train, 21 of 69 val and 34 of 64 held-out rows of "
            "each are under 8 words, so those val/held-out rows can no longer be hit -- "
            "unprotectable, not clean (attestation export.template_strip, by_set). A "
            "per-(family, option-set) rule looks needed; it is NOT implemented (the lead "
            "amends v5's pre-registration first). For the amendment: knowledge.multiple_choice "
            "has 661 distinct option sets over 715 rows at 5%, largest shared by 51 rows, so a "
            "per-option-set rule also strips MMLU's shared option lists unless it is scoped."
        ),
    }
)

append_gap(
    {
        "id": PREMISES,
        "opened": "2026-10-02",
        "lane": "L-prep2",
        "owner": "lead",
        "status": "open",
        "tool": (
            "direct reads: python/qd_data/mixture.py:238-264 (_request: question=family."
            "description), :451-530 (rewrite_clinc), python/qd_data/general.py:181-266 "
            "(MMLU/CSQA context=question), :397-460 (rewrite_clinc_two_stage); rendered rows "
            "in python/tests/test_containment_strip.py"
        ),
        "question": (
            "Does the template-strip text in campaign/v5-preregistered.DRAFT.json "
            "data.decontamination.rule describe the rows it will run on?"
        ),
        "answer": (
            "Four points the amendment should state, each implemented as read here and "
            "recorded in attestation v2's export.template_strip: (1) Every qd_data row's "
            "question line is its family's description (mixture._request), so it is constant "
            "in every family and the rule strips it in MMLU, CSQA, SQuAD and defect_class too; "
            "their per-row question is in the CONTEXT block (general.py: context=question; "
            "SQuAD: question, blank line, passage), which the strip never touches, so it is "
            "kept whole as the DRAFT intends -- the DRAFT's parenthetical ('questions vary per "
            "row') describes the context, not the question line. (2) The per-domain option "
            "lists are intent.within_domain's (rewrite_clinc_two_stage), not "
            "intent.classification's, which samples 15 random distractors plus the gold per "
            "row. (3) 'Byte-identical across every rendered row' is vacuous for a group of one "
            "row; implemented as: a (family, slot) group of fewer than 2 rows strips nothing "
            "(none occurred in the 1/2/5% scans). (4) An option value is constant when it is a "
            "member of every row's option values at seed=None; for the fixed lists measured the "
            "positional reading gives the same set."
        ),
    }
)

append_gap(
    {
        "id": SUBSAMPLE,
        "opened": "2026-10-02",
        "resolved": "2026-10-02",
        "lane": "L-prep2",
        "owner": "lead",
        "status": "resolved-with-residual",
        "tool": "AUDIT/prep2-2026-10-02/make_subsample.py, family_rates.py (686cdca)",
        "question": (
            "Can L-prep's unstripped 1/2/5% subsample rates (CLINC 8.5/9.1/20.1%, MMLU "
            "5.9-6.9%, code.defect_class 0.8-1.1%) be reproduced? Its subsampler and rate "
            "script lived in /private/tmp and were lost."
        ),
        "answer": (
            "Re-created and committed: make_subsample.py keeps a line iff the first 8 bytes "
            "of sha256(line) are below fraction x 2^64, over the eleven general caches, the "
            "composed and the noul rows, with the base through --defect-max-rows "
            "500/999/2,498; family_rates.py takes a family's rate as its distinct train "
            "identity keys listed in exclusions.txt. The rows are not L-prep's (its hash was "
            "not recorded): split rows 3,560/7,008/17,143 against its 3,385/6,945/17,105. "
            "Unstripped rates: CLINC 3.3/8.4/31.0% (L-prep 8.5/9.1/20.1), MMLU 0.0/9.0/7.4% "
            "(5.9-6.9), code.defect_class 1.6/1.3/1.5% (0.8-1.1), CSQA and SQuAD answer_span "
            "0 (0). Not reproduced number for number; the shape is (CLINC grows with the "
            "sample, CSQA/SQuAD 0, defect_class about 1%, pair counts 46/136/875 against "
            "48/135/762)."
        ),
        "residual": (
            "Why the numbers differ is inferred, not shown: different rows, small "
            "denominators at 1% (246 CLINC and 128 MMLU keys), and a CLINC rate that grows "
            "with the number of val targets per source; at 5% 146 of the 338 unstripped CLINC "
            "keys were hit by an intent.within_domain row of their own and 231 by an "
            "intent.in_scope row (some by both), a split L-prep's breakdown did not report."
        ),
    }
)
print(f"appended {CLINC}, {PREMISES}, {SUBSAMPLE}")
