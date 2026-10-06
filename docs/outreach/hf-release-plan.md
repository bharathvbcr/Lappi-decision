# Lappi v0.1 preview on Hugging Face: release plan and licence checklist

Drafted 2026-10-06 by the outreach lane. **Nothing here has been done.** Creating the Hugging Face
repo, uploading, entering a token and choosing the repo's name are the author's actions, each
needing the author's yes. Labels: [V] checked against the cited file or page this lane, [I]
inferred, [U] not verified, **[decide]** the author's call.

The card is `docs/outreach/hf-model-card.md`; it becomes the Hugging Face repo's `README.md`.

## Decided

- **Licence: Apache-2.0** (the author, 2026-10-06). The base, Qwen3.5-2B-Base, is Apache-2.0
  (its `LICENSE`, "Copyright 2026 Alibaba Cloud") [V: the cached base snapshot and its model page].

## Before upload

1. **[decide] The source repository's visibility.** The card points readers at the Rust runtime in
   `github.com/bharathvbcr/Lappi-decision`, which is private. Either make the repository public
   first, or remove those links and say the runtime is not yet public. The card has no other
   supported way to call the model.
2. **[decide] The JevArena results page.** The card links a private claude.ai page. Share it, move
   it to a public location, or delete the link.
3. **[decide] The served-families numbers.** The card gives the release notes' all-families figures
   (0.857; 91.9% at 87.0%) and, beside them, served-families figures this lane recomputed from the
   same score row's verdicts (0.842; 90.7% at 85.6%), labelled [derived]. Rule 5 asks that public
   numbers cite a ledger row: these cite the row's verdict file through
   `docs/outreach/evidence/served_families.py`, which reproduces the published figures exactly before
   splitting them. Keep, or drop the derived lines.
4. **[decide] "21 served families" versus "15".** The brief for this lane said 15. That is two
   different counts: the calibration table's 15 entries (one per option count, 2 to 16), and the 15
   decision-pool families the load test replayed. The runtime admits any trained family
   (`crates/qd-runtime/src/admission.rs:103-125`), so 21 families with choice slots are served; 6 of
   them (CLINC, MMLU and CommonsenseQA-derived) were never sent through the runtime. The card says
   all of this. An alternative is to tell callers to use only the 15 tested families.
5. **Re-run the card's claims against the files** if anything is re-scored or re-exported first.
6. **[decide] The defect-probe, reasoning-pilot and decider-2b results.** The card reports them
   from the campaign bench directory (`defect-probe/report-c1full.txt` and `report.txt`;
   `jev-compare-pilot.txt`) and, for decider-2b, from the harness's run directory, scored by the
   outreach lane with the bench lane's `compare.py`. Their owning lane has not written them up in a
   handoff. Keep them, or wait for that write-up.

## What to upload

From the release directory (read-only; copy, never edit in place):

| File | Upload? | Why |
|---|---|---|
| `model.safetensors` (sha256 `5ade5349…`) | yes | the weights |
| `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | yes | byte-identical to the base snapshot b1485b2f [V: sha256 of each matches] |
| `calibration.json` (sha256 `7e56b34e…`) | yes | serving refuses without it |
| `release_manifest.json` | **[decide] a redacted copy** | It holds five local absolute paths: `base_snapshot.path`, `calibration.source`, `source.manifest`, `source.safetensors`, `train_manifest.path`. The runtime reads only `format`, `decode`, `expected_identity.*`, `files.*.sha256`, `calibration.*` hashes and `trained_families` [I: grep of `crates/qd-runtime/src/release.rs`, not graph-confirmed], so a copy with those five values replaced should still load. **Load the redacted copy with `qd-metal-serve` before upload** (a GPU job: run it under `tools/mac_heavy.sh` once the lock is free) |
| `span_head.safetensors` | **no (recommended)** | The mean of five seeds' near-orthogonal span heads (projection norm 0.447 of one seed's); no serving loader reads it; span is not served. Uploading it invites use of a head the release notes say not to serve |
| `LICENSE` | add | Apache-2.0 text, for the weights |
| `LICENSE-Qwen` | add | a copy of the base's `LICENSE` (Apache-2.0 §4(a)). The cached base snapshot has no `NOTICE` file [I: the snapshot holds the full repo listing, including `README.md` and `.gitattributes`], so there is no NOTICE text to carry |
| `README.md` | add | the card |

## Steps (the author runs them)

1. Decide items 1-4 and 6 above.
2. Make the redacted `release_manifest.json` in a scratch copy of the release directory and
   load-test that copy (one request through `qd oneshot --socket` is enough to show it binds).
3. Create the model repo on Hugging Face under the account you choose. Suggested name:
   `lappi-v0.1-preview`.
4. Upload the files above from the scratch copy, with your own token, from your own shell or the web
   UI. This lane does not handle credentials.
5. Compare the sha256 of each uploaded file with the table above (the Hub shows LFS sha256 per file).
6. Check that the card's metadata block renders (licence, base model, datasets) on the repo page.
   The `base_model_relation: finetune` key was written from memory of the Hub's model-card spec
   [U]; drop it if the Hub flags it.

## Licence and attribution checklist

Training sources are those admitted in the v5 train manifest (16 source ids; 487,407 rows). Row
counts by licence are the manifest's `licence_histogram` [V]. Upstream licences were read from each
dataset's card or repository on 2026-10-06 [V: a read-only pass of the public pages; revisions not
pinned].

| Source | Declared licence | Obligation the card must carry | Status |
|---|---|---|---|
| Qwen/Qwen3.5-2B-Base | Apache-2.0 | ship its licence; say the weights are modified | card says "fine-tuned from"; add `LICENSE-Qwen` [V] |
| bigcode/commitpackft (code files) | per file; dataset card MIT | attribution for MIT/BSD/ISC/Apache files | only mit, apache-2.0, bsd-2/3-clause, isc, cc0-1.0, unlicense rows were admitted [V: train manifest histogram; the card's own list also includes agpl-3.0, lgpl-2.1, mpl-2.0, epl-1.0 and unknown, which `qd_data.licences` refuses]. The card cites OctoPack. Whether per-file attribution reaches weights trained on mutated copies is a legal question [U] |
| rajpurkar/squad_v2 | CC BY-SA 4.0 | attribution; **ShareAlike** | cited. See ShareAlike below |
| allenai/ai2_arc | CC BY-SA 4.0 | attribution; **ShareAlike** | cited |
| google/boolq | CC BY-SA 3.0 | attribution; **ShareAlike** | cited |
| tals/vitaminc | CC BY-SA 3.0 | attribution; **ShareAlike** | cited |
| clinc/clinc_oos | CC BY 3.0 | attribution; **disclose that the test split is training data** | cited. All three splits were read and re-split by intent, so the test utterances of the training intents are training data (`sources.py:486-493`); the card says so [V] |
| nvidia/HelpSteer2 | CC BY 4.0 | attribution | cited. Its prompts are ~95% from ShareGPT and its responses from NVIDIA in-house models [V: card]; no extra terms stated |
| ZefanCai/Open-Jev-v1.1 | card `license: other`, "source-dependent": generated content CC0, WANLI-derived text CC BY 4.0 | attribution per WANLI row | rows admitted only when their own `provenance.license` reads CC-BY-4.0 or CC0-1.0 [V: `crates/qd-prep/src/decisions.rs:559-568`]. The card says customer-control rows keep an unverified provenance; the 10-03 report says those rows were refused per row [V: `AUDIT/finalize-2026-10-03/report-to-human-2026-10-03.md:74`], not re-counted here [U]. WANLI's seed examples were generated by GPT-3 (a provider-output terms flag in `AUDIT/data-licence-survey-2026-10-06.md`, fetched by its research agent; not re-checked here [U]). **[decide]** with the DeepSeek question below |
| tasksource/procedural-typed-decisions | Apache-2.0 | attribution | rule-generated, no LLM [V: card]; cites tasksource |
| LocalLLaMA/typed-decisions | Apache-2.0 | attribution | labels from an unnamed ~4B "teacher" [V: card]: the generator's terms cannot be checked [U] |
| n4ze3m/typed-decisions-synth | MIT | attribution | **generated by DeepSeek V4.1 Flash** [V: card], whose output terms were not read [U]. **[decide]** read them before upload |
| Mapika/decider `teacher_data` | the repository's Apache-2.0 ("Copyright 2026 Mark Marosi") | attribution | `teacher_data/` has no licence file of its own; Apache-2.0 applies by inference [I]. Labelled by a local Qwen3.5-27B teacher [V: README] |
| cais/mmlu | MIT | attribution; **disclose that the test split is training data** | cited; also feeds abstain-contrast rows. Its test and dev splits are trained on (`python/qd_data/sources.py:196-204`, `benchmark_reportable=False`); the card says MMLU scores of this model are not benchmark results [V] |
| tau/commonsense_qa | MIT | attribution | cited; also feeds abstain-contrast rows. Its validation split is a curated internal val, not the published benchmark (`sources.py:219-227`) [V] |
| the author's own repositories | owner-granted | none; do not name private repositories | 1,961 rows; personal and business paths struck by the author on 2026-10-02 [V: v5 pre-registration] |
| bespokelabsai/nimble | none (all rights reserved) | — | **not used**: refused by `qd_data.licences` [V] |
| sources refused for non-commercial or unknown licences (ANLI, SciQ, toxic-chat, OpenBookQA, GLUE, CodeXGLUE defect, …) | — | — | not used [V: the train manifest's `refused_sources`] |

### The one obligation that may conflict with the licence decision

**ShareAlike.** 123,319 training rows (cc-by-sa-4.0 111,045, cc-by-sa-3.0 12,274) come from SQuAD v2,
ARC, BoolQ and VitaminC [V: train manifest]. CC BY-SA requires adaptations of the licensed material
to be shared under the same licence. Whether model weights trained on CC BY-SA text are an
"adaptation" is not settled, and neither this lane nor the cards answer it. Releasing the weights
under Apache-2.0 is common practice for models trained on these datasets [I], but it is the
author's call, ideally with advice. The card attributes every source and states the question
openly. It does not claim the weights are free of ShareAlike.

### Generated-data terms

Two sources were written by other models (DeepSeek V4.1 Flash; Qwen3.5-27B) and one by an unnamed
teacher. Some model providers restrict using outputs to train other models. None of the three
dataset cards states such a restriction [V], but the generator's own terms were not read [U]. Read
DeepSeek's output terms before upload.

Open-Jev's NLI rows derive from WANLI, whose seed examples came from GPT-3
(`AUDIT/data-licence-survey-2026-10-06.md` flags WANLI for provider-output terms) [U: not re-read
by this lane]. WANLI itself is CC BY 4.0. Read OpenAI's terms as they applied to those outputs
alongside DeepSeek's.
