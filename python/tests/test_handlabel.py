"""The hand-labelling harness: resumability, and the intra-rater ceiling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qd_label.session import LabelItem, LabelSession  # noqa: E402
from qd_train.agreement import cohens_kappa  # noqa: E402


def _pool(n: int = 40) -> list[LabelItem]:
    langs = ["rust", "go", "python", "typescript", "swift"]
    return [
        LabelItem(
            item_id=f"d{i:03d}",
            repo=f"org/repo{i % 7}",
            path=f"src/mod{i}.rs",
            language=langs[i % len(langs)],
            diff=f"@@ -1,3 +1,3 @@\n-    let x = {i};\n+    todo!()\n",
        )
        for i in range(n)
    ]


def test_an_empty_pool_is_refused():
    with pytest.raises(ValueError, match="empty pool"):
        LabelSession([], "/tmp/unused.jsonl")


def test_duplicate_item_ids_are_refused(tmp_path: Path):
    dupes = _pool(3) + _pool(1)
    with pytest.raises(ValueError, match="duplicate item_ids"):
        LabelSession(dupes, tmp_path / "s.jsonl")


def test_an_item_with_an_empty_diff_is_refused():
    with pytest.raises(ValueError, match="empty diff"):
        LabelItem(item_id="x", repo="r", path="p", language="rust", diff="   ")


def test_labels_persist_and_the_session_resumes(tmp_path: Path):
    store = tmp_path / "labels.jsonl"
    s1 = LabelSession(_pool(20), store)
    for item, presentation in list(s1.next_items(limit=5)):
        s1.record(item.item_id, "stub", presentation=presentation)

    # A fresh object over the same store picks up exactly where the last one stopped.
    s2 = LabelSession(_pool(20), store)
    assert len(s2.first_pass_done()) == 5
    assert len(s2.pending()) == 15


def test_an_unknown_item_is_refused(tmp_path: Path):
    s = LabelSession(_pool(5), tmp_path / "s.jsonl")
    with pytest.raises(KeyError, match="not in this session"):
        s.record("nope", "stub")


def test_an_invalid_label_is_refused(tmp_path: Path):
    s = LabelSession(_pool(5), tmp_path / "s.jsonl")
    with pytest.raises(ValueError, match="not one of"):
        s.record("d000", "probably-a-stub")


def test_unsure_requires_a_note(tmp_path: Path):
    """An 'unsure' with no reason is the one record that cannot improve the rubric."""
    s = LabelSession(_pool(5), tmp_path / "s.jsonl")
    with pytest.raises(ValueError, match="requires a note"):
        s.record("d000", "unsure")
    s.record("d001", "unsure", note="is a a silent stub if the caller ignores the result?")


# --------------------------------------------------------------------------
# The intra-rater ceiling
# --------------------------------------------------------------------------

def test_repeats_are_offered_and_measure_self_agreement(tmp_path: Path):
    store = tmp_path / "labels.jsonl"
    s = LabelSession(_pool(60), store, repeat_fraction=0.5, seed=1)

    seen_second_pass = 0
    for item, presentation in s.next_items(limit=60):
        # Label consistently except on repeats, where we deliberately flip.
        label = "clean" if presentation == 2 else "stub"
        s.record(item.item_id, label, presentation=presentation)
        seen_second_pass += presentation == 2

    assert seen_second_pass > 0, "no repeats were offered; the ceiling cannot be measured"
    first, second = s.intra_rater_pairs()
    assert len(first) == len(second) == seen_second_pass
    # We flipped every repeat, so self-agreement must be terrible.
    assert cohens_kappa(first, second) <= 0.0


def test_a_consistent_labeller_scores_a_high_ceiling(tmp_path: Path):
    s = LabelSession(_pool(60), tmp_path / "l.jsonl", repeat_fraction=0.5, seed=2)
    for item, presentation in s.next_items(limit=60):
        # Same label for the same item both times, across two classes.
        s.record(item.item_id, "stub" if item.item_id.endswith(("0", "2", "4")) else "clean",
                 presentation=presentation)
    first, second = s.intra_rater_pairs()
    assert len(first) > 0
    # The property under test is element-wise consistency. Kappa is only 1.0 when the
    # repeat sample happens to span both classes; a single-class sample is the
    # degenerate p_e==1 case, which cohens_kappa correctly scores 0, not 1.
    assert first == second
    if len(set(first)) > 1:
        assert cohens_kappa(first, second) == pytest.approx(1.0)


def test_a_repeat_is_never_drawn_from_the_last_ten_items(tmp_path: Path):
    """Re-presenting something just seen measures recall, not consistency."""
    s = LabelSession(_pool(12), tmp_path / "l.jsonl", repeat_fraction=1.0, seed=3)
    offered = list(s.next_items(limit=6))
    # With only <=10 first-pass records on file, nothing is eligible for repeat yet.
    assert all(p == 1 for _, p in offered)


def test_an_item_is_never_repeated_twice(tmp_path: Path):
    s = LabelSession(_pool(80), tmp_path / "l.jsonl", repeat_fraction=0.9, seed=4)
    for item, presentation in s.next_items(limit=80):
        s.record(item.item_id, "clean", presentation=presentation)
    seconds = [r.item_id for r in s.records() if r.presentation == 2]
    assert len(seconds) == len(set(seconds)), "an item was re-presented more than once"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def test_the_manifest_is_marked_held_out(tmp_path: Path):
    s = LabelSession(_pool(10), tmp_path / "l.jsonl")
    for item, presentation in s.next_items(limit=10):
        s.record(item.item_id, "stub", presentation=presentation)
    m = s.export_manifest(tmp_path / "heldout.json")
    assert m["held_out"] is True
    written = json.loads((tmp_path / "heldout.json").read_text())
    assert written["held_out"] is True
    assert written["n_labelled"] == 10


def test_unsure_items_are_kept_but_marked_unusable(tmp_path: Path):
    """Dropping them silently would hide evidence about the rubric."""
    s = LabelSession(_pool(6), tmp_path / "l.jsonl")
    items = [i for i, _ in s.next_items(limit=6)]
    for k, item in enumerate(items):
        if k < 2:
            s.record(item.item_id, "unsure", note="genuinely ambiguous")
        else:
            s.record(item.item_id, "logic")
    m = s.export_manifest(tmp_path / "out.json")
    assert m["n_labelled"] == 6
    assert m["n_usable"] == 4
    assert m["n_unsure"] == 2
    assert sum(1 for i in m["items"] if not i["usable"]) == 2


def test_stats_report_progress_and_distribution(tmp_path: Path):
    s = LabelSession(_pool(20), tmp_path / "l.jsonl")
    for k, (item, presentation) in enumerate(s.next_items(limit=8)):
        s.record(item.item_id, "stub" if k % 2 else "clean", presentation=presentation, seconds=12.0)
    st = s.stats()
    assert st.total_items == 20
    assert st.labelled_first_pass == 8
    assert sum(st.by_label.values()) == 8
    assert sum(st.by_language.values()) == 8
    assert st.median_seconds == 12.0
