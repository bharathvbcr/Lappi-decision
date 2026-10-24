# Rule-4 cloud cost estimate for Vani-adapter seeds (DRAFT, 2026-10-21)

**STATUS: DRAFT. NEEDS HUMAN YES. Nothing here is approved, launched or booked.** Card
ft-5dfe48e7, criterion 4. Plan: ruling 8 ("Rule 4 binds every 8xH100 job"), ruling 12, section 11
("A cloud cost estimate under rule 4 before any cloud seed").

## What a cloud run may carry

**Synthetic and public rows only** (ruling 12). Real Vani rows (`source: vani-agreed`) never leave the
Mac. **No real-row cloud run is planned.** A real-row run promotes only as a full-recipe, at least
three-seed Mac run. Before any launch the pool shipped to the box must be checked to hold no
`vani-agreed` row and no owner label. Whether an existing gate enforces that for a cloud upload was
not verified in this lane; the check is owed to the launch checklist below, not assumed.

## Price source

Read only from files already in the repo; nothing recalled.

| Hardware | $/h | Source | Age on 2026-10-21 |
| --- | --- | --- | --- |
| Lambda 1x GH200 (aarch64) | 2.29 | `AUDIT/training-audit-2026-10-06.md` price table, "fetched 2026-10-06" (also `campaign/gh200-phase0-3.json`, `usd_per_hour_source`, read 2026-09-30) | 15 days |
| Lambda H100 SXM, 1x | 4.29 | same table | 15 days |
| Lambda H100 SXM, 2x | 4.19 per GPU, 8.38 per box | same table; `campaign/v5-preregistered.json` (`box`) | 15 days |
| Lambda 8xH100 SXM | 31.92 per box | `docs/plan-corrections.md` DESIGN-4 ("the plan's rates are correct") | older than the table; no fetch date in that file |

Every price is stale for a launch: the launch checklist re-reads the provider's price page and
writes the figure and read date into the run's `usd_per_hour_source`. A price that is not re-read is
**not sourced** for that launch.

## What cannot be estimated yet

The GPU-hours per seed. They need tokens per seed (the v2 pre-registration, written in P1) and a
**measured cloud LoRA rate**, and neither exists:

- No row measures LoRA tokens per second on a GH200 or an H100. The measured LoRA rates are Mac rows
  (`ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L3`, `#L4`), which do not transfer.
- `ojas-cuda`'s Qwen3.5 step refuses compute (plan section 4, `ojas-cuda/src/lib.rs:17-20`), so the
  cloud LoRA path itself is not established.
- The v5 per-seed hour figures in `campaign/v5-preregistered.json` are for a full fine-tune and are
  estimates marked [I] there, not rows for this recipe.

So the **projected hours are not measured.** The run that measures them is the box session of card
ft-12da8e67 (v1 recipe parity on a cloud box), extended by one row: the tape-lora step bench at
T = 2,048 and 9,638, 3 timed steps, on the box's GPU. The estimate below is therefore a **ceiling by
cap**, which is what rule 4 binds anyway: a job's cost is bounded by its wall-clock cap times its
price, whatever its speed.

## Cost at the cap (arithmetic over the sourced prices)

| Box | $/h | Cap 6 h | Cap 12 h | Cap 24 h |
| --- | --- | --- | --- | --- |
| 1x GH200 | 2.29 | 13.74 | 27.48 | 54.96 |
| 1x H100 | 4.29 | 25.74 | 51.48 | 102.96 |
| 2x H100 | 8.38 | 50.28 | 100.56 | 201.12 |
| 8x H100 | 31.92 | 191.52 | 383.04 | 766.08 |

The rule-4 line in this repo's practice is $20 (CLAUDE.md rule 4: single-GPU jobs under $20 do not
need a human yes; every 8xH100 job does). A 1x GH200 job capped at 8 h is $18.32, under the line; the
same GH200 at 12 h is over it. Whether a seed fits 6 or 24 h is the unmeasured part.

## The decision this asks of the human

**Not decided, and not mine to decide.** After P1 writes the v2 pre-registration and the rate row
exists, this document gets one line filled in per planned run:

| Field | Value |
| --- | --- |
| Run, seeds, controls | to be filled (a promotable candidate is at least 3 seeds plus 3 shuffled-label controls, `campaign/lora-2b-preregistered.json` `controls`) |
| Box and price | to be filled, with the re-read date |
| Wall-clock cap | to be filled; computed as the projected hours from the measured rate plus a stated margin, never open-ended |
| Dollar ceiling | cap x price, from the table's arithmetic |
| Auto-terminate | `terminate_command` in the campaign config (below) |
| Human yes | **NEEDS HUMAN YES**: the owner's answer, quoted, with its date, in the config's `approved_by` |

## Auto-terminate and the cap (existing mechanism, not new)

`tools/campaign_driver.py` refuses a config with no `terminate_command` ("rule 4's auto-terminate is
not optional", `campaign_driver.py:17`, `:765-767`) and takes `usd_per_hour`, `usd_per_hour_source`,
`n_gpus`, `campaign_cap_hours` and `approved_by` from the config (`campaign/gh200-phase0-3.json`
shows the shape). Reuse it; do not write a second launcher. The cap counts from boot, as
`campaign/next-train-first-box-queue-2026-10-08.json` does.

## Launch checklist (every item before the first cloud seed)

1. The v2 pre-registration exists and states tokens per seed (P1 exit).
2. A measured cloud LoRA rate row exists and is cited here.
3. The pool shipped is checked to hold no `vani-agreed` row and no owner label.
4. Price re-read from the provider page on the day; figure and date recorded.
5. Cap, ceiling and `terminate_command` written; `approved_by` carries the human's yes.
6. Held-out data and the two task-holdout families stay out of the box's training process (rule 3's
   `qd-train` path check is not bypassed).
