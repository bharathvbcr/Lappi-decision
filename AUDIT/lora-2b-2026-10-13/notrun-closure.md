# ft-2ed2fd55: closing the checks reported NOT RUN (2026-10-13)

Card: ft-2ed2fd55e93dc1019b02deb7e1e2c754 (LoRA-2B lane, Lappi side). Scope: Fable's wave-3 ruling, lane W3-3, in
`AUDIT/board-sweep-2026-10-13/fable-wave3-ruling.md`. This is verification only; no code changed.

The W3-3 lane worktree could not run any of these checks. The MANVI gate refused every path outside the lane's
own root: the Metal lock directory, `uv --python ~/.venvs/ml/...`, and `shasum` on the HF cache. Its closure
file, on branch `wave3-l3-lora-notrun` at 0d83ae1, reports all of them as NOT RUN. That branch is not merged,
and this file supersedes it. The orchestrator session then ran what its own gate allowed, from the main
checkout at `main` 05b8237, the lane's base.

## 1. D17 Metal tests on the tiny fixture: RAN, PASSED

Command:

```
cargo test --manifest-path /Users/bharath/Code/research/Lappi-decision/Cargo.toml --release -p qd-metal --test gpu_d17 -- --ignored --test-threads=1 --skip gpu_snapshot_d17_zero_is_the_tied_embedding_on_the_real_2b
```

Output (verbatim tail):

```
running 2 tests
test gpu_a_release_with_d17_serves_the_letter_rows_through_the_backend ... ok
test gpu_d17_zero_reproduces_todays_scores_bit_for_bit_and_a_real_d17_equals_pythons_rows ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.25s
```

The wave-3 integration branch's gate run repeats this command.

## 2. Torch tests `test_lora_reference.py`, `test_ckpt_average_lora_merged.py`: RAN INSIDE THE SUITE, INDIVIDUAL OUTCOME UNVERIFIED

The gate refused the direct per-file launch, `uv run --no-project --python /Users/bharath/.venvs/ml/bin/python ...`,
because it hands `uv` a path outside the repository. The orchestrator ran the project's own target instead.

Command: `make -C /Users/bharath/Code/research/Lappi-decision torch-pytest`.

Result: `2 failed, 5099 passed, 341 skipped, 4 warnings in 1220.74s`.

Both failures were in `python/tests/test_gaps_ledger.py`, and both were caused by this session's own
uncommitted files in the main checkout:
- a gap record that lacked `question`;
- a ruling file that cited shortened gap-id prefixes.

Both causes were fixed, and `test_gaps_ledger.py` then gave `10 passed`.

The two LoRA files are in `python/tests`, so they were collected. Because the `-q` output does not list
per-file outcomes or skip reasons, **whether they passed or were skipped is not verified** by this run.
Their `importorskip("torch")` and `importorskip("transformers")` are satisfied by the ml venv (torch 2.12.1,
transformers 5.19.0, per Fable's ruling), so a pass is **inferred**, not verified.

## 3. `gpu_snapshot_d17_zero_is_the_tied_embedding_on_the_real_2b`: NOT RUN

This test loads the real 2B on Metal, and the human has not accepted the Mac risk. Its name is confirmed at
`crates/qd-metal/tests/gpu_d17.rs:107-108` (`#[ignore = "GPU + model snapshot"]`).

## 4. Re-hash of the two approved downloads: NOT RUN

The gate refuses `shasum` on paths outside the repository:
`[command.operand.unverifiable/hard] shasum is handed a path outside the repository`.

The recorded values are in `AUDIT/lora-2b-2026-10-09/downloads.md:34-35`:

| File | Bytes | sha256 |
|---|---|---|
| `Qwen3.5-2B-Base/.../model.safetensors-00001-of-00001.safetensors` | 4,548,221,488 | `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d` |
| `Qwen3.5-0.8B-Base/.../model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |

To close this, a human runs:

```
shasum -a 256 ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/model.safetensors-00001-of-00001.safetensors ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/model.safetensors-00001-of-00001.safetensors
```

## Still open on the card

- (b′)/(d′) wait on ojas's Tape StepProvider.
- The real-2B D17 test (item 3).
- The hash re-check (item 4).
- An isolated pass/skip count for the two LoRA files (item 2).
