# The FT linear control on the native engine: parity rows on phase 3 (2026-10-01)

Fable decision E2 (2026-10-01): these rows are **parity evidence for the port**, not rows of
record. They live here, not in `ledger/`.
- **Rows of record:** phase 3's linear-control rows stay the Python ones (`eeda5db4`,
  `2b08a357`, `635c19d9` in `ledger/gh200-seed0-weights-2026-09-30.jsonl`).
- **Why not in `ledger/`:** a second control supplement per eval row would make
  `promotion_verdict`'s `_joined` merge two states for one control.
- **From J4 onward:** the native engine (`qd-prep ngrams` / `qd-prep linfit`, merged at
  `7ed66d7`) is the control of record. J4's Python control subshell is stopped before it can
  write a row (`box_j4_controls_native.sh`).

The engine and its parity definition are in `HANDOFF/perf-linear-control-rust-2026-10-01.md`
and `AUDIT/perf-linear-control-rust-2026-10-01.md`.

## The files (verbatim copies)

| File | sha256 | What |
| --- | --- | --- |
| `linear-control-native-parity-2026-10-01/box-scratch-ledger-lc-parity-s0.jsonl` | `76797d7f…6e060` | The box's scratch ledger. Rows 1–15 are copies of `ledger/gh200-seed0-weights-2026-09-30.jsonl`; row 16, `186b32a7`, is the native control. It was run on the GH200 (aarch64, glibc 2.39) from `qd-lane4` at `7ed66d7` with the cross-built `qd-prep` sha256 `af9d3a76…7887c2` |
| `linear-control-native-parity-2026-10-01/mac-native-s0.jsonl` | `546e61e1…24e57b` | Row `8095435c`, run on the Mac by the port lane |
| `linear-control-native-parity-2026-10-01/mac-native-s1.jsonl` | `94339614…f0bfb8` | Row `6f932163`, Mac |
| `linear-control-native-parity-2026-10-01/mac-native-s2.jsonl` | `f23c2d03…5d7954` | Row `79d27778`, Mac |

## Native vs the Python row of record

All four native rows are non-quick, with `recipe.control_engine = "qd-prep linfit"`. Each was
compared metric by metric against the Python row for the same eval row:

| Native row | Python row | Eval row | Linear iterations | Length iterations | Top-1s, both arms | `paired_margin_vs_linear.choice` | `paired_margin_vs_length_control` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `186b32a7` (box) | `eeda5db4` | `6d170b3c` | 845 = 845 | 1175 = 1175 | equal | +0.1501 [+0.1359, +0.1651], equal | +0.4991 [+0.4786, +0.5193], equal |
| `8095435c` (Mac) | `eeda5db4` | `6d170b3c` | 845 = 845 | 1175 = 1175 | equal | equal | equal |
| `6f932163` (Mac) | `2b08a357` | `60f29b07` | 846 = 846 | 1184 = 1184 | equal | +0.1497 [+0.1355, …], equal | +0.4987, equal |
| `79d27778` (Mac) | `635c19d9` | `5c19c0e8` | 847 = 847 | 1189 = 1189 | equal | +0.1497, equal | +0.4987, equal |

- **The only differing values** are the final gradient norms of the two convergence metrics,
  in the 4th–5th significant digit. For example, `186b32a7` gives 9.94592e-05 against
  `eeda5db4`'s 9.94477e-05. Both are under the 1e-4 tolerance, at the same iteration.
- **Wall clock on the box, phase-3 seed 0:**
  - native: linear fit 68.2 s and length fit 9.8 s on 64 threads (`logs/lc-parity-s0.log`);
  - Python: 700.8 s (`eeda5db4`'s own record).

Verified: by reading the four rows against the committed Python rows, field by field
(values and details), on 2026-10-01.
