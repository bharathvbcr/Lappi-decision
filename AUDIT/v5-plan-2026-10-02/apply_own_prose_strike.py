"""Record the human's own-prose strike and the measured question-form count in the v5 DRAFT.

The lead ran this once on 2026-10-02, before the v5 build. Its changes:
- `data.sources[3]` (own-repo prose) gains `human_strike`, the human's answer recorded in
  AUDIT/v5-plan-2026-10-02/human-answer-own-prose-strike.md;
- `data.sources[3]` gains `form_measured`, lane L-v5-data's measured question count (197, not
  the planning inventory's ~599), which supersedes the `form` clause's estimate;
- `amendments_applied` is appended.
Textual insertion keeps the file's layout. A second run fails, because the key would already
exist.
"""

import json
import sys
from pathlib import Path

DRAFT = Path(__file__).resolve().parents[2] / "campaign" / "v5-preregistered.DRAFT.json"
text = DRAFT.read_text(encoding="utf-8")
d = json.loads(text)
src = d["data"]["sources"][3]
if "human_strike" in src:
    sys.exit("already applied")


def q(v):
    return json.dumps(v, ensure_ascii=True)


anchor = f'"provenance": {q(src["provenance"])}'
if text.count(anchor) != 1:
    sys.exit("provenance anchor not found exactly once")
strike = (
    "the human, 2026-10-02 ~16:45 UTC, 'Strike personal + business (Recommended)' "
    "(AUDIT/v5-plan-2026-10-02/human-answer-own-prose-strike.md; "
    "GAP-L-V5DATA-OWN-PROSE-HUMAN-REVIEW-2026-10-02): struck by walker rule are the repos "
    "web/Lappi-BDay, web/WhimsicalLove and web/bharathvbcr, and the paths "
    "scholarlm/docs/business/** and research/BINN/writing/** (187 of the lane's 2,150 units; "
    "~1,963 remain, above the 1,500 floor); AGENTS.md/CLAUDE.md and CHANGELOG.md are kept"
)
form_measured = (
    "lane L-v5-data (HANDOFF/v5-data-2026-10-02.md section 2): the walker reads a wrapped sentence "
    "whole and refuses markup fragments, non-sentence starts and unbalanced quotes, so the question "
    "form measured 197 units before the strike (1,953 paragraphs), not the planning inventory's "
    "~599; the counts after the strike and after exclusions are filled at build time "
    "(amendments_pending); the 1,500 floor on the route is unchanged"
)
text = text.replace(
    anchor, anchor + f', "human_strike": {q(strike)}, "form_measured": {q(form_measured)}'
)
applied = d["amendments_applied"]
new_applied = applied + (
    " Then on 2026-10-02 (~16:50 UTC), the human's own-prose strike and the measured question-form "
    "count were recorded in data.sources[3] (human_strike, form_measured) by "
    "AUDIT/v5-plan-2026-10-02/apply_own_prose_strike.py."
)
if text.count(q(applied)) != 1:
    sys.exit("amendments_applied not found exactly once")
text = text.replace(q(applied), q(new_applied))
out = json.loads(text)
assert out["data"]["sources"][3]["human_strike"] == strike
assert all(ord(c) < 128 for c in text)
DRAFT.write_text(text, encoding="utf-8")
print("amended", DRAFT)
