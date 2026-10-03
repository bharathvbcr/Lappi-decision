"""Apply the containment same-family scope (Fable ~20:35Z 2026-10-03, ratified by the human) to the
v5 DRAFT's data.decontamination.rule.

    python3 AUDIT/finalize-2026-10-03/apply_v5_containment_scope_amendment.py <DRAFT path>

The ruling, the measured counts and the human's ratifications are in
AUDIT/finalize-2026-10-03/containment-scope-ruling-2026-10-03.md. The rule text gains a sentence
naming the scope, and amendments_applied gains a record. No gate, threshold or population moves.
Textual edits keep the file's layout. A second run fails, because the rule would already cite the
ruling.
"""

import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit(__doc__)
DRAFT = Path(sys.argv[1]).resolve()
RULING = "AUDIT/finalize-2026-10-03/containment-scope-ruling-2026-10-03.md"
text = DRAFT.read_text(encoding="utf-8")
d = json.loads(text)
rule = d["data"]["decontamination"]["rule"]
if RULING in rule:
    sys.exit("already applied")
if "build-order-ruling-2026-10-03.md" not in d["build_order"][2]:
    sys.exit("the build-order amendment is not applied; apply apply_v5_build_order_amendment.py first")


def q(v):
    return json.dumps(v, ensure_ascii=True)


def replace_once(old, new):
    global text
    if text.count(old) != 1:
        sys.exit(f"anchor not found exactly once ({text.count(old)}): {old[:80]!r}")
    text = text.replace(old, new)


added = (
    " Same-family scope (Fable, ~20:35Z 2026-10-03, ratified by the human, " + RULING + "): for "
    "a build that reads the general-decision pool, a containment pair whose source and target "
    "rows are of the same pool family is listed in pairs.tsv but neither excludes its source "
    "row nor counts as a remaining hit; the pool's families share task templates and rule "
    "prose, so 8-gram containment between two rows of one of them measures the template, and "
    "their train/val hygiene is the pool's own pre-registered checks (group-keyed draw, "
    "near_duplicate_disjoint, exact_content_disjoint). Pairs across families, and every pair of "
    "a non-pool family, are enforced as before. The scope is part of the corpus "
    "(decisions_pool_same_family_not_enforced, from the pool manifest's families), so an "
    "exclusion list made without it is refused; the attestation counts what it left unenforced "
    "(same_family_not_enforced). On the full scan at ca48960 it covered ~97.6k of 104,905 "
    "excluded keys; code.defect_class 6,165, MMLU 1,076, CLINC 3, SQuAD 12 and the 4 "
    "cross-family pool keys stand. CLINC strip version 2's full-scan numbers were ratified with "
    "the two non-zero zero-checks (val 1 exact, held-out 1 subsequence) as a stated caveat on "
    "CLINC's val and held-out numbers."
)
replace_once(q(rule), q(rule + added))
note = (
    " On 2026-10-03 (~20:45Z) the containment same-family scope was applied "
    "(AUDIT/finalize-2026-10-03/apply_v5_containment_scope_amendment.py; " + RULING + "): "
    "data.decontamination.rule gained the scope sentence and the human's two ratifications. No "
    "gate, threshold or population moves."
)
replace_once(q(d["amendments_applied"]), q(d["amendments_applied"] + note))
after = json.loads(text)
assert after["data"]["decontamination"]["rule"] == rule + added
assert after["amendments_applied"].endswith(note)
d["data"]["decontamination"]["rule"] = rule + added
d["amendments_applied"] = d["amendments_applied"] + note
assert after == d
DRAFT.write_text(text, encoding="utf-8")
print(f"applied to {DRAFT}: data.decontamination.rule names the same-family scope")
