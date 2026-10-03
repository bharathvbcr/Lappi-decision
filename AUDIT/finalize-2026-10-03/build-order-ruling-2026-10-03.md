# Fable's ruling on the v5 build order, 2026-10-03 ~20:05Z

## The question (the lead)

The DRAFT's build_order runs three heavy jobs one after another on the Mac:
1. a standalone v5 build (build_order[2]);
2. the containment scan over that build's rows (build_order[3]);
3. a rebuild with `--exclude-identity-keys`.

v4's build took 3,910 s on the Mac (`ledger/mac-phase4-v4-shards-2026-10-01.jsonl`). v5 adds the
decision pool and composed-v2 at 10,240 tokens, so it is likely longer. The 2× H100 box went up
at ~19:46Z and bills $8.38/h until the queue launches, and the queue can launch only after the
data is attested.

May the scan run on the corpus flags directly, followed by one build with the exclusion list?

## The ruling (Fable, verbatim headline)

> Ruling: granted. Run the scan first, then one build with `--exclude-identity-keys`. It's a
> build_order amendment, not a silent reorder.

## Why it is safe

- **The scan does not need a build.** prep3's subsample scans ran on corpus directories with no
  pipeline build (`HANDOFF/prep3-2026-10-02.md`, the `make_subsample` → `containment_scan`
  block). On v5-build, `tools/containment_scan.py` takes the pipeline's corpus flags, including
  `--decisions-pool` (:339), and renders the rows itself.
- **The standalone build produces nothing the exclusion build does not.** That covers
  `over_max_seq_len_composed_train`, the survival bins, the routes line, and the pool's
  dedupe/split report. Contrast rows only ever appeared on the rebuild.

## The conditions, and how each is met

1. **A7 is the discriminator.** `tools/v5_a7_check.py` reads only the val and held-out manifests
   (`SPLIT_PATHS`, :47). The pipeline refuses an exclusion list unless every key names a train row
   (`real_tokenizer_pipeline.py` `--exclude-identity-keys` help). So exclusions cannot move A7.
   - An A7 missing key on the one build is therefore a dedupe knock-out. The DRAFT's existing
     rule then applies: exclude that row and rebuild.
   - `build/v5-build/v5_build.sh` also prints any A7 missing key that is in `exclusions.txt`;
     there should be none.
   - Path taken: one build, A7 with `--decisions-pool`, reconciled against `exclusions.txt`.
2. **Identical corpus flags, the same `--rev` and a clean tree for the scan and the build.**
   - One flag list (`build/v5-build/v5_flags.sh`) feeds both:
     - `--rev` ca48960. That is v5-build with:
       - both halves of v4 (3a6b9fa);
       - the v5-2gpu merges (ccac87e, db8701e);
       - this amendment (4298020);
       - qd-prep's MinHash request bound raised from 4 GiB to 16 GiB (ca48960).

       None of these changes the Python data or training code. The first scan, at 4298020,
       stopped at `qd-prep minhash`: its request was 4,534,193,280 bytes, over the 4 GiB bound
       (`build/v5-build/scan-attempt2-minhash-bound.log`). Fable ruled to raise the bound, not
       chunk, and named 32 GiB. 16 GiB was used instead, because linwire's test requires its
       32 GiB bound to stay strictly above MinHash's. The class fix is
       GAP-MINHASH-REQUEST-CARRIES-SHINGLE-BYTES-2026-10-03.
     - `--no-repo-history`;
     - `--defect-class data/pool/commitpackft-composed-v2 --defect-download data/pool/commitpackft`;
     - `--defect-noul data/pool/defect-noul-v3c`;
     - `--general-record …/fetch-record-2026-09-29.json --general-max-rows 200000`;
     - `--decisions-pool` set to the bench's v4 pool, checked against its report's examples sha256.
   - No `--max-pairs` and no `--defect-max-rows`.
   - Cargo.lock is restored before each run.
   - v5-2gpu is merged before the scan or after the build's ledger row, never between. It was
     merged before the scan (8ab473b → ccac87e; gaps.jsonl's conflict resolved as a union). Its
     branch changes none of python/qd_data, python/qd_train, tools or crates (`git diff --stat`
     from merge-base 1ab477f), so the merge does not move the build's code fingerprint.
3. **Recorded.** This file, plus a DRAFT amendment to build_order[2] and [3]
   (`apply_v5_build_order_amendment.py`). The DRAFT the lane carries becomes canonical at merge,
   so the amendment goes on top of the lane's.
4. **After the scan, before the build:** `family_rates.py` and `key_ii_blind_zero_checks.py`, as
   prep3 says. Both zero-checks must read 0/0. Otherwise it is a GAP for the human, not a build.
   The scan's exit 0 is the only CLEAN.
