# The human's answer on own-prose personal and business content (2026-10-02)

The lead asked through AskUserQuestion at ~16:45 UTC 2026-10-02, on lane L-v5-data's flag in `HANDOFF/v5-data-2026-10-02.md` (GAP-L-V5DATA-OWN-PROSE-HUMAN-REVIEW-2026-10-02). The question and the chosen option, verbatim:

**Asked:** "The v5 abstention examples include prose from your repos. Some of it is personal or business writing: the Lappi-BDay, WhimsicalLove and bharathvbcr web repos (118 paragraphs), ScholarLM's YC applications and pitches (38), and BINN's LinkedIn drafts (31). Training could memorise bits of it. Which should be struck before the build? Striking all of it still leaves 1,963 paragraphs, above the 1,500 minimum."

**Chosen:** "Strike personal + business (Recommended)". The option's text: "Drop all 187 paragraphs: the three personal web repos and the YC/pitch and LinkedIn drafts. Keeps the technical docs (READMEs, AGENTS.md/CLAUDE.md, CHANGELOGs)."

## What it strikes (counts are the lane's, from its 2,150-unit selection)

- `web/Lappi-BDay`: 98 units.
- `web/WhimsicalLove`: 6.
- `web/bharathvbcr`: 14.
- `scholarlm/docs/business/**` (YC applications, pitches): 38.
- `research/BINN/writing/**` (LinkedIn drafts): 31.

That is 187 units. AGENTS.md/CLAUDE.md (158) and CHANGELOG.md (75) stay. The strike is applied by repo exclusion and path rule in the own-prose walker. `own-prose-v1/files.jsonl` is regenerated and its sha256 re-pinned. The DRAFT's own-prose route records the strike.
