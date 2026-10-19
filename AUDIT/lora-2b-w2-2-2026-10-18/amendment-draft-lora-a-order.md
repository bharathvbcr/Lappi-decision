# DRAFT, NOT IN FORCE: amendment to campaign/lora-2b-preregistered.json, init.lora_a (canonical draw order)

Status: DRAFT written by lane W3-1 (2026-10-19). It is not applied. `campaign/lora-2b-preregistered.json`
is untouched on this branch. Only the owner writes or approves the amendment (CLAUDE.md rule 2;
`amendments_rule`). Branch `w3-1-lora-a-canonical-order-2026-10-19` is HELD until that approval and
must not merge before it.

Basis: Fable ruling item 1, `AUDIT/lora-2b-w2-2-2026-10-18/fable-ruling.md`: the pinned
"model.named_modules() order" is a defect in the rule's wording, an implementation artefact; real HF
registers a GDN layer's `out_proj` before `in_proj_qkv`, `in_proj_z`, ojas lists qkv, z, out. This is
an init rule, not a gate, threshold, seed or held-out set. No LoRA-2B result exists, so
`amendments_rule` allows it (an amendment after a result is read is void).

## Text to replace

File `campaign/lora-2b-preregistered.json`, key `init.lora_a.rule` (currently line 169). Replace the
clause

    visiting the adapted modules in model.named_modules() order (qd_train.lora.attach_lora)

with

    visiting the adapted modules in the canonical order: layer index ascending (read from the
    'layers.<n>.' component directly before the suffix), then the position of the module's suffix in
    LORA_TARGET_SUFFIXES (qd_train.lora.lora_draw_order; ojas LORA_KINDS, ten for ten), independent of
    the order a framework registers or a provider lists the modules; a target with no layer index, or
    two targets at one (layer, suffix), is refused (qd_train.lora.attach_lora)

Key `init.lora_a.pinned_as` (line 170) needs only the pointer "named in qd_train.lora.attach_lora" to
stay true; no digest value is pinned there, so nothing else changes.

## Effect (measured on this branch)

- Fixture `crates/qd-train/tests/fixtures/lora-plan-oracle.json`: its tiny tree already registered in
  the ojas order, so the three seed digests (0, 7, 1234) are UNCHANGED:
  f60be8da0a172a2dbc4f8d26ad833c6f47a155adeaffc660de85296eedd53666,
  72a222dddf456b318a1b406d9e1435edc0fcfa2db190752488c773c9a612e240,
  106cc7f009b2830e9e84f9e3091b6497319f425d8077579a5937dbc7a7bf284b.
- A model registered in HF order changes: on main its seed 0, 7, 1234 digests were
  d9bf82b979d177147526ef722b229a5d3264bfb93aebc4d35dc7e21a20177d96,
  f61338c7885cc69a6786364b47fca21855262cb15493a6ccace17e8cb17decfe,
  45b963ecbaec95d24955d869f90b2dd2704cc26f87c824d4068b00b39095f663; on this branch they equal the
  ojas-order digests above. The fixture now records those under `hf_registration_order`.
- Any real Qwen3.5 run's A digest changes versus main (the real model registers in HF order). No
  LoRA-2B result exists, so no recorded row is invalidated.

## Not in this amendment

`lora_param_groups` and `lora_parameters` keep `named_modules()` order: that is optimizer group
membership, not the A draw, and the pre-registration does not pin it.
