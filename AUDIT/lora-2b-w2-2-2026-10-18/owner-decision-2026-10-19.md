# Owner decisions, 2026-10-19 (asked by session lappi-decision-10)

Both were asked through AskUserQuestion in session lappi-decision-10. The owner chose the recommended option each time. The questions and answers are recorded verbatim.

## 1. LoRA-A canonical draw order

**Asked:** "W3-1 makes the LoRA-A init draw order canonical: layer index first, then the suffix table. Today the pre-registration (campaign/lora-2b-preregistered.json:169) pins named_modules() order, so the torch and ojas legs would draw different A matrices on every GDN layer, and every ojas row would be voided. Should I commit the amendment (draft in AUDIT/lora-2b-w2-2-2026-10-18/amendment-draft-lora-a-order.md) and merge W3-1 once its tests pass?"

**Answer:** "Yes, amend and merge (Recommended)".

**Effect:**
- `campaign/lora-2b-preregistered.json` gains `amendments[n=4]`, and `init.lora_a.rule` is replaced by the canonical-order text. The basis is Fable's ruling in `AUDIT/lora-2b-w2-2-2026-10-18/fable-ruling.md`, item 1.
- Before this commit, no LoRA-2B result exists: `RIPGREP_CONFIG_PATH= rg -uu -il 'optimizer_recipe.{0,10}lora|lora-2b' ledger/` returns nothing.
- Branch `w3-1-lora-a-canonical-order-2026-10-19` merges only after its cargo and torch suites have run green.

## 2. ojas pin

**Asked:** "Lappi pins ojas at 04277f8, but ojas main is now 0e95fe7, so every pin check (make pin, gates, preflight, qd-train-metal) is red. Bump the pin?"

**Answer:** "Bump with delta report (Recommended)".

**Effect:**
- `upstream.lock.json` moves from 04277f8 to 0e95fe7.
- A delta report is written to `AUDIT/ojas-pin-0e95fe7-2026-10-19/`.
- The bump is verified with `make pin` and `qd-train-metal` under `tools/mac_heavy.sh` before it merges.
- This clears owner decision 10 in `HANDOFF/harden-2026-10-17.md`.
