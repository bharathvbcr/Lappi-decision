"""Read a cProfile dump of the trajectory prelude: the named phases, then the top cumulative.

Analysis only (stdlib pstats). Usage: python3 read_profile.py PROF [TOP]
"""

from __future__ import annotations

import pstats
import sys

#: (file suffix, function) of the phases the HANDOFF reports, in prelude order.
PHASES = [
    ("real_ft_run.py", "main"),
    ("real_ft_run.py", "resolve_rev"),
    ("real_ft_run.py", "__init__"),  # ShardReader.__init__ lives in qd_train; see below
    ("real_ft_run.py", "check_defect_source"),
    ("real_ft_run.py", "check_exclusion_source"),
    ("real_ft_run.py", "corpus_facts"),
    ("real_ft_run.py", "ft_split_rows"),
    ("real_ft_run.py", "ft_split_report"),
    ("real_tokenizer_pipeline.py", "base_sources"),
    ("defect_class.py", "load_defect_rows"),
    ("real_tokenizer_pipeline.py", "general_rows"),
    ("decisions.py", "load_decision_pool"),
    ("mixture.py", "build_mixture"),
    ("exclusions.py", "drop_before_dedupe"),
    ("real_tokenizer_pipeline.py", "native_minhash"),
    ("real_tokenizer_pipeline.py", "_prep_signature_matrix"),
    ("real_tokenizer_pipeline.py", "_prep_candidate_pairs"),
    ("real_tokenizer_pipeline.py", "_run_prep"),
    ("subprocess.py", "run"),
    ("minhash.py", "shingle"),
    ("minhash.py", "signature"),
    ("minhash.py", "candidate_pairs"),
    ("dedupe.py", "dedupe"),
    ("split.py", "split"),
    ("real_tokenizer_pipeline.py", "exclusions_then_contrast"),
    ("real_ft_run.py", "replay_corpus_identity"),
    ("exclusions.py", "containment_corpus"),
    ("real_ft_run.py", "vocab_letter_ids"),
    ("real_ft_run.py", "open_val_set"),
    ("real_ft_run.py", "prepare_ood"),
    ("real_ft_run.py", "_checkpoint_step"),
]


def main(path: str, top: int) -> None:
    stats = pstats.Stats(path)
    raw = stats.stats  # type: ignore[attr-defined]
    total = max(ct for (_, _, _, ct, _) in raw.values())
    print(f"profile {path}: largest cumulative {total:.1f} s\n")
    print(f"{'cumulative s':>12} {'tottime s':>10} {'calls':>10}  function")
    for suffix, name in PHASES:
        for (file, line, func), (_cc, nc, tt, ct, _) in raw.items():
            if func == name and file.endswith(suffix):
                print(f"{ct:12.1f} {tt:10.1f} {nc:10d}  {func} ({file.rsplit('/', 1)[-1]}:{line})")
    print(f"\ntop {top} by cumulative time:")
    rows = sorted(raw.items(), key=lambda kv: kv[1][3], reverse=True)[:top]
    for (file, line, func), (_cc, nc, tt, ct, _) in rows:
        print(f"{ct:12.1f} {tt:10.1f} {nc:10d}  {func} ({file.rsplit('/', 1)[-1]}:{line})")
    print(f"\ntop {top} by own (tottime) time:")
    rows = sorted(raw.items(), key=lambda kv: kv[1][2], reverse=True)[:top]
    for (file, line, func), (_cc, nc, tt, ct, _) in rows:
        print(f"{ct:12.1f} {tt:10.1f} {nc:10d}  {func} ({file.rsplit('/', 1)[-1]}:{line})")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
