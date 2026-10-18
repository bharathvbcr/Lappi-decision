# Fable ruling: W2-2 merge, the LoRA-A draw order, the shard-reader symlink hole (2026-10-18)

Asked by session lappi-decision-10 (GitPulse backlog run, card ft-3b8126cd) after lane W2-2
(branch `w2-2-lora-driver-2026-10-18`, ac88dbf) reported one open adversarial BREAK. Fable answered
read-only. The body below is Fable's answer, copied verbatim.

---

**Fable ruling, W2-2 (read-only; devmap not consulted — every cite below is a file read or command run this session)**

**1. A-digest order: a defect in the rule's wording, and the test is right.**
The pre-registration pins "model.named_modules() order (qd_train.lora.attach_lora)" (`campaign/lora-2b-preregistered.json:169`) and pins no digest value (:170). That order is an implementation artefact, and three orders are now in play [V]:
- Real HF (transformers 5.19.0, `models/qwen3_5/modeling_qwen3_5.py:530-535`): a GDN layer registers `out_proj` before `in_proj_qkv`, `in_proj_z`.
- ojas `lora_targets` (`ojas-model/src/qwen35/params.rs:921-936`, `LORA_KINDS` :717-728): qkv, z, out.
- Lappi's own Rust oracle tree (`tools/qd_train_oracle_lora_plan.py:63-67`, fixture `lora-plan-oracle.json` layer 0): qkv, z, out — the ojas order, not the real model's.

Lappi's verifier takes order from the provider's entries (`crates/qd-train/src/recipe.rs:269-272`, `ledger.rs:1163-1170`), so the ojas leg recomputes a different digest from the torch row on every GDN layer: every ojas row voided, or worse, trained on a different A while d' (:190) stays inside 0.02 nats for early steps because B=0 damps A's effect [I]. The lane's test caught a real break.

Ruling: amend :169 to a canonical order — layer index ascending, then the suffix-table order (`lora.py` `LORA_TARGET_SUFFIXES` = `LORA_KINDS`, ten for ten per my 2026-10-17 ruling) — which ojas and the oracle fixture already produce. `attach_lora`/`lora_a_rule_digest` sort targets into it; `lora_adapter_layout` sorts or refuses. This is an amendment (`amendments_rule` :271), allowed: no LoRA-2B result exists; rule 2 is not engaged (an init rule, not a gate/threshold/seed/held-out set). Owner yes, batched with amendment 1 (section 6 item 2 of my final ruling). Fails-before: the baca7e9 run, already recorded in the gap.

**2. Merge: now, with the test kept and marked `xfail(strict=True, reason=GAP-W2-2-LORA-A-DIGEST-…)`.**
Not (a): a known-red main is where the next red hides. Not (b): the driver is critical-path step 7 and waits on an owner yes. Not (c): holding it off drops the fails-before. Strict xfail is no weakening: the assertion is untouched, XPASS fails the suite, so the amendment lane must flip it. Q15 is satisfied: 13 failed on bd2cbc9 under `w2-2-q15b` [V gaps.jsonl].

Yes, merge main in first: `git merge-tree --write-tree` is clean (4cc1f8a) [V]; main's `real_ft_run.py` diff touches none of the 14 names the driver imports (:69-83) and `RunRecorder`'s new kwargs default to None [V]. Then re-run the two driver files only (same 62-test scope, rest deselected) under the lock; rule 5, as Q16 for W2-a. Not run means not merged.

**3. shards.py hole: lappi-decision-ce fixes it at the owner; not stop-ship.**
`ShardReader.__init__` checks the root (`shards.py:2019-2027`) and opens files by plain path (:2037, :2068); `_resolve` already resolves symlinks (`data_access.py:65-71`), so the fix is one `assert_path_not_held_out` per file opened, plus a fails-before (symlinked tokens file into `data/heldout`), and the driver's copy (`lora_reference_run.py:209-215`) deleted in the same change. The Rust reader's per-file behaviour is [U] (`held_out.rs:9-14`). Not stop-ship: a symlink is no stronger than a copy; the content layer (step-0 containment, :260; write-side manifest matching, `shards.py:65-69`) is what catches held-out content wherever it sits. Lands before the v6 shard set is first read for training. GAP id required.

---

## Lead's notes (lappi-decision-10)

- The amendment to `campaign/lora-2b-preregistered.json:169` (canonical LoRA-A draw order) waits for the
  owner's yes. It is not written here. No LoRA-2B result exists at this commit.
- Item 3 is assigned to session lappi-decision-ce, lane W9-c, which also deletes the driver's per-file
  resolve once W2-2 is on main.
