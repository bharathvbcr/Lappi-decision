# HANDOFF: lappi-v2-plan, 2026-10-20

Lane: planning only. No code changed, nothing trained, nothing run on a GPU.

## What was measured

Nothing. No ledger row was written or read for a result. Every v2 number in the plan is a
ruling, a citation, or arithmetic over a cited file, labelled as such.

## What changed

Committed with this file:

- `docs/lappi-v2-plan-2026-10-20.md` (new). The v2 plan: owner rulings §2.1-2.15, model lineup,
  Gemma 4 bake-off, task families, data gates, speed and serving operations, phases, open items.
  It was audited three times on 2026-10-20 (citations, Gemma 4 feasibility, rules and
  consistency, then a gap audit) with two Fable advisor reviews.
- `docs/caller-contract.md` §5: the Vani-only exception (teacher readings on real public
  postings, ≤25% rows, loss weight 0.5, two-teacher agreement, content-level held-out door,
  redactor, Mac-only, private adapters). Caller records are unchanged.
- `docs/lappi.md` and `CLAUDE.md`: v2 scope pointers. The v1 Qwen3.5-2B campaign finishes
  unchanged as the baseline.
- `gaps.jsonl`: two records appended, `GAP-V2-PLAN-SURVEY-DEVMAP-STALE-2026-10-20` and
  `GAP-LORA2B-PREREG-L202-GPU-HOURS-UNDERSTATED-2026-10-20`.

Not in this commit: no campaign file was edited. The d1 eval amendment (retire `port_trigger`,
add d1-ultra) and the v2 pre-registration are owed by their cards.

GitPulse boards (label `lappi-v2-train`):

| Board | Cards |
| --- | --- |
| Lappi-decision, new | ft-37919170 owner labels + d1 amendment; ft-951c360d real-row gates; ft-1faada73 first-scope families; ft-094e1acb rung-1 bake-off; ft-c3eaca80 later scope; ft-5dfe48e7 serving ops and schedule; ft-57fc7484 v1 blockers |
| Lappi-decision, relabelled | ft-d066187f, ft-3b8126cd, ft-2ed2fd55, ft-3163ce23 (urgent); ft-0eb828f1, ft-c5ea3e71, ft-8139b5b1 (high, the latter with owner items 13-15 added); ft-63647d3b moved to review (looks done by amendment 5) |
| Vani | ft-7099bd3b caller side |
| ojas | ft-3a9bc23c Gemma training path (gated); 14 LoRA-2B/training cards relabelled |
| tessl | ft-663f5b35 Gemma forward (gated); ft-02b223f0, ft-62af7c2e relabelled |
| Jarvis | ft-0fb664e1 Lappi caller (parked) |

Lowered to normal so the training path leads: Lappi ft-5af5e8ee and ft-0f5a10c9
(code.defect_class serving); ojas ft-dc4bfe22, ft-4f11ce5a, ft-96023b34 and ft-64601da5 (vision).

## What is open

- Owner: ft-8139b5b1 items 1-15, including the ojas pin gate run
  (`GAP-OJAS-PIN-4123DFB-SOURCE-DIFF-AND-PIN-GATE-PENDING-2026-10-19`), the commitpackft held-out
  ratio, the v5 length band, the stale shard headers and the prereg line 202 correction;
  ~230 owner labels (ft-37919170); Gemma download yes and licence read (ft-094e1acb).
- Agent, urgent: ft-57fc7484 (red tests, containment scan footprint, span-null rows, D3 tool,
  `lora_ft_row` producer) and ft-d066187f (qd-train onto ojas-train; pin already at 4123dfb,
  dependency and module deletion not done).
- Gaps: `GAP-V2-PLAN-SURVEY-DEVMAP-STALE-2026-10-20`,
  `GAP-LORA2B-PREREG-L202-GPU-HOURS-UNDERSTATED-2026-10-20`, and the blocking set listed in the
  plan's §10.

## First command for the next lane

Owner, from the repo root: `make pin`

Agent lane on ft-57fc7484, from the repo root: `make torch-pytest`
