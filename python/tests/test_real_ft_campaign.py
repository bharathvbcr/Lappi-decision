"""``tools/real_ft_run.py`` as the campaign's trainer: its cap, its ``quick`` flag, its corpus.

The pre-rental checks of 2026-09-29 found the trainer every campaign phase shells out to was
built as a short probe:

* ``WALL_CLOCK_CAP_S = 1_800.0`` was the only cap, for the control and the cost estimate
  alike, so phase 4's 24-30 h epoch would have stopped at 30 minutes;
* ``_recorder`` wrote ``quick=True`` on every row, so under rule 8 nothing the campaign
  trained could ever promote, however it ran;
* the label rebuild read this repository's git history whether or not the shard set had;
* the memorisation arm always ran first, a Mac smoke convenience the GH200 pays for.

Torch-gated like ``test_real_ft_pieces.py``: the tool raises at import without torch.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch is an optional 'mac' extra, not in .venv")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "python"))

import real_ft_run as rft  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_train.run_control import MAX_CAP_S, hard_exit_on_cap  # noqa: E402

HOURS = 3600.0
GH200 = ["--devices", "cuda", "--instance", "lambda-1xGH200", "--usd-per-hour", "1.49"]


# --- the cap ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cap", ["0", "-5", "nan", "inf", str(MAX_CAP_S + 1)])
def test_a_cap_outside_the_program_cap_is_refused_at_argv_time(tmp_path, cap):
    """The approval is up to 72 h; MAX_CAP_S (40 h) is lower and read-only, so it binds."""
    with pytest.raises(SystemExit, match=r"--wall-clock-cap-s .* is outside"):
        rft.main(["--out", str(tmp_path), "--wall-clock-cap-s", cap])


def test_the_bound_is_the_lower_of_the_approval_and_the_program_cap():
    bound = min(72 * HOURS, MAX_CAP_S)
    assert bound == rft.MAX_WALL_CLOCK_CAP_S


def test_the_control_and_its_cost_carry_the_same_cap():
    ctl = rft._control(100, device="cpu", lr=1e-5, cap_s=30 * HOURS)
    assert ctl.cap.cap_s == 30 * HOURS
    assert ctl.cost.cap.cap_s == 30 * HOURS


def test_the_default_is_the_cap_every_earlier_row_ran_under():
    ctl = rft._control(100, device="cpu", lr=1e-5)
    assert ctl.cap.cap_s == rft.WALL_CLOCK_CAP_S == 1_800.0
    assert ctl.auto_terminate is None


def test_a_campaign_length_cap_on_a_rented_gpu_needs_a_name_at_argv_time(tmp_path):
    """$1.49/h x 30 h = $44.70, over rule 4's $20 line. Refused before the shard set is read
    or a tower loaded, not per arm inside ``_train`` after that time is paid for."""
    with pytest.raises(SystemExit, match=r"rule 4: .*NEEDS A HUMAN YES"):
        rft.main(["--out", str(tmp_path / "absent"), *GH200,
                  "--wall-clock-cap-s", str(30 * HOURS)])


def test_a_name_gets_the_run_past_the_approval_check(tmp_path):
    # Any later refusal will do: the shard directory does not exist.
    with pytest.raises(BaseException) as got:
        rft.main(["--out", str(tmp_path / "absent"), *GH200,
                  "--wall-clock-cap-s", str(30 * HOURS), "--approved-by", "bharath"])
    assert "rule 4" not in str(got.value)


def test_a_short_cap_on_a_rented_gpu_needs_no_name(tmp_path):
    """The default 30 minutes at $1.49/h is $0.75: what every earlier GH200 row ran under."""
    # Any later refusal will do: the shard directory does not exist.
    with pytest.raises(BaseException) as got:
        rft.main(["--out", str(tmp_path / "absent"), *GH200])
    assert "rule 4" not in str(got.value)


def test_a_run_that_needs_a_human_yes_is_armed_with_auto_terminate():
    """RunControl refuses such a run without it; the fixed 30-minute cap never needed one."""
    kw = {"device": "cuda", "lr": 1e-5, "n_gpus": 1, "usd_per_hour": 1.49,
          "instance": "lambda-1xGH200", "approved_by": "bharath"}
    assert rft._control(100, cap_s=30 * HOURS, **kw).auto_terminate is hard_exit_on_cap
    assert rft._control(100, **kw).auto_terminate is None


def test_the_cap_and_no_memorise_reach_the_recipe_only_when_they_differ():
    """So every row written before them hashes as it did."""
    base = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
            "permutation": None, "replay": None}
    assert rft._recipe_pieces(**base) == {}
    assert rft._recipe_pieces(**base, cap_s=rft.WALL_CLOCK_CAP_S, no_memorise=False) == {}
    assert rft._recipe_pieces(**base, cap_s=30 * HOURS, no_memorise=True) == {
        "wall_clock_cap_s": 30 * HOURS, "no_memorise": True,
    }
    assert {"wall_clock_cap_s", "no_memorise"} <= set(rft.RECIPE_PIECE_KEYS)


# --- --batch-tokens --------------------------------------------------------------------------


def test_batch_tokens_reaches_the_recipe_only_when_it_differs():
    """The GH200 throughput row d732111f ran ~1,424 positions per optimizer step: the widest
    bucket, 1,625, was the only batch size the tool could run, and a 2B model at that size
    leaves the GPU waiting on the host. A larger batch is a different recipe, so it hashes
    differently; the default hashes as every row before the flag did."""
    base = {"lower_layers_n": 0, "lower_lr_scale": 1.0, "beta2": rft.DEFAULT_BETA2,
            "permutation": None, "replay": None}
    assert rft._recipe_pieces(**base, batch_tokens=None) == {}
    assert rft._recipe_pieces(**base, batch_tokens=32768) == {"batch_tokens": 32768}
    assert "batch_tokens" in rft.RECIPE_PIECE_KEYS


@pytest.mark.parametrize(("given", "widest", "want"), [
    (None, 1625, (1625, None)),
    (1625, 1625, (1625, None)),
    (32768, 1625, (32768, 32768)),
])
def test_batch_tokens_resolves_against_the_widest_bucket(given, widest, want):
    assert rft._resolve_batch_tokens(given, widest=widest) == want


@pytest.mark.parametrize("given", [1624, 0, -1, (1 << 20) + 1])
def test_a_batch_that_cannot_hold_the_widest_row_or_breaks_the_ceiling_is_refused(given):
    with pytest.raises(SystemExit, match="--batch-tokens"):
        rft._resolve_batch_tokens(given, widest=1625)


# --- --no-memorise ---------------------------------------------------------------------------


def test_no_memorise_needs_the_epoch_arm(tmp_path):
    with pytest.raises(SystemExit, match="without --epoch there is no other arm"):
        rft.main(["--out", str(tmp_path), "--no-memorise"])


def test_a_memorise_checkpoint_has_nowhere_to_go_under_no_memorise(tmp_path):
    ckpt = tmp_path / rft._checkpoint_name("memorise", 0, "cpu")
    ckpt.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="no arm to resume it into"):
        rft.main(["--out", str(tmp_path), "--real-backbone", str(tmp_path), "--devices", "cpu",
                  "--seeds", "0", "--epoch", "--no-memorise", "--resume-from", str(ckpt)])


# --- the corpus revision and the repository history -------------------------------------------


@pytest.mark.parametrize("rev", ["HEAD", "main", "0632f69"])
def test_the_corpus_revision_is_a_full_sha(tmp_path, rev):
    """Every run of this tool writes ledger rows."""
    with pytest.raises(SystemExit, match="not a full 40-character commit sha"):
        rft.main(["--out", str(tmp_path), "--rev", rev])


def test_max_pairs_is_refused_where_the_rebuild_reads_nothing_it_bounds(tmp_path):
    with pytest.raises(SystemExit, match="would determine nothing"):
        rft.main(["--out", str(tmp_path), "--no-repo-history", "--max-pairs", "400"])


def test_ft_splits_rebuilds_through_the_pipelines_own_source_owner(monkeypatch):
    """One owner for "which code and span rows": the pipeline's base_sources."""
    import real_tokenizer_pipeline as pipeline

    seen: list[dict[str, object]] = []

    class Stop(Exception):
        pass

    def spy(**kw):
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(pipeline, "base_sources", spy)
    with pytest.raises(Stop):
        rft.ft_splits(commitpackft=None, max_pairs=3, rev="r", config=rft.DataConfig(),
                      repo_history=False)
    assert seen == [{"repo_history": False, "commitpackft": None, "max_pairs": 3,
                     "blank_line_runs": False, "rev": "r"}]


def test_the_replay_attestation_names_a_history_free_corpus_and_only_that_one():
    """An attestation made over one corpus must not vouch for a run over another; every
    attestation written before the flag keeps matching (the key is absent by default)."""
    kw = {"rev": "r", "max_pairs": 400, "commitpackft": None, "defect_class": None,
          "defect_max_rows": None}
    assert rft.replay_corpus_identity(**kw) == {**kw}
    assert rft.replay_corpus_identity(**kw, repo_history=True) == {**kw}
    assert rft.replay_corpus_identity(**kw, repo_history=False) == {**kw, "repo_history": False}


def test_replay_decontam_rebuilds_the_same_history_free_split(tmp_path, monkeypatch):
    import replay_decontam

    class Stop(Exception):
        pass

    seen: list[dict[str, object]] = []

    def spy(**kw):
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(rft, "check_defect_source", lambda out, *, defect_class: None)
    monkeypatch.setattr(rft, "ft_splits", spy)
    with pytest.raises(Stop):
        replay_decontam.main([
            "--out", str(tmp_path), "--replay-shards", str(tmp_path), "--tokenizer-json",
            str(tmp_path / "t.json"), "--attestation-out", str(tmp_path / "a.json"),
            "--no-repo-history",
        ])
    assert [kw["repo_history"] for kw in seen] == [False]


def _manifest(out: Path, *, n_input: dict[str, int], status: dict[str, object] | None = None,
              snapshot: str = "d" * 64) -> None:
    path = out / rft.TRAIN_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "data_snapshot_hash": snapshot,
        "status": status or {"state": "ran", "passed": True},
        "mixture": {"n_input": n_input},
        "config_fingerprint": DataConfig().fingerprint(),
    }), encoding="utf-8")


def _facts(out: Path, **kw):
    kw.setdefault("repo_history", True)
    kw.setdefault("commitpackft", None)
    kw.setdefault("config", DataConfig())
    return rft.corpus_facts(out, data_snapshot_hash="d" * 64, **kw)


def test_the_manifest_must_be_this_shard_sets(tmp_path):
    with pytest.raises(SystemExit, match="is absent"):
        _facts(tmp_path)
    _manifest(tmp_path, n_input={"bigcode/commitpackft": 5}, snapshot="e" * 64)
    with pytest.raises(SystemExit, match="not the manifest this shard set was written from"):
        _facts(tmp_path)


def test_a_history_free_set_is_not_relabelled_with_history_or_the_reverse(tmp_path):
    defect_only = tmp_path / "defect"
    _manifest(defect_only, n_input={"qd-mutate/commitpackft": 50})
    with pytest.raises(SystemExit, match="built with --no-repo-history"):
        _facts(defect_only)
    assert _facts(defect_only, repo_history=False).history_rows == {}

    history = tmp_path / "history"
    _manifest(history, n_input={"bigcode/commitpackft": 5, "rajpurkar/squad_v2": 4})
    with pytest.raises(SystemExit, match="but --no-repo-history was passed"):
        _facts(history, repo_history=False)
    assert _facts(history).history_rows == {"rajpurkar/squad_v2": 4, "bigcode/commitpackft": 5}
    # With --commitpackft the code rows came from the download, not from history.
    download = _facts(history, commitpackft=tmp_path)
    assert download.history_rows == {"rajpurkar/squad_v2": 4}


def test_a_general_record_set_is_refused_rather_than_mislabelled(tmp_path):
    """Without --general-record the general families (phase 4's full mixture) are not
    rebuilt; test_real_ft_general_record.py covers the set relabelled with its record."""
    _manifest(tmp_path, n_input={"cais/mmlu": 10, "qd-mutate/commitpackft": 50})
    with pytest.raises(SystemExit, match="built with --general-record"):
        _facts(tmp_path, repo_history=False)


def test_a_not_run_snapshot_is_carried_with_its_reason(tmp_path):
    _manifest(tmp_path, n_input={"qd-mutate/commitpackft": 50},
              status={"state": "not_run", "reason": "the read of ['x'] hit its row bound"})
    assert _facts(tmp_path, repo_history=False).snapshot_not_run == (
        "the read of ['x'] hit its row bound"
    )
    ran = tmp_path / "ran"
    _manifest(ran, n_input={"qd-mutate/commitpackft": 50})
    assert _facts(ran, repo_history=False).snapshot_not_run is None


# --- quick, from the run's facts --------------------------------------------------------------

CLEAN = rft.CorpusFacts(snapshot_not_run=None, history_rows={})
CLEAN_RUN = {"tag": "epoch", "device": "cuda", "real_backbone": True, "corpus": CLEAN,
             "termination": "steps_exhausted"}


def test_a_full_epoch_on_the_campaign_device_over_a_clean_corpus_is_not_quick():
    """The all-clear: until 2026-09-29 no row this tool wrote could be anything but quick."""
    assert rft.quick_reasons(**CLEAN_RUN) == []


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"tag": "memorise"}, "memorisation arm is a subsample"),
        ({"termination": "wall_clock_cap"}, "a truncated schedule"),
        ({"termination": "data_exhausted"}, "a truncated schedule"),
        ({"corpus": rft.CorpusFacts(snapshot_not_run="capped", history_rows={})},
         "data snapshot is NotRun"),
        ({"corpus": rft.CorpusFacts(snapshot_not_run=None,
                                    history_rows={"rajpurkar/squad_v2": 4})},
         "this repository's own history"),
        ({"device": "mps"}, "not a campaign device"),
        ({"device": "cpu"}, "not a campaign device"),
        ({"real_backbone": False}, "stand-in"),
    ],
)
def test_each_rule_8_condition_alone_makes_a_row_quick(change, says):
    reasons = rft.quick_reasons(**{**CLEAN_RUN, **change})
    assert len(reasons) == 1, reasons
    assert says in reasons[0]


def test_the_ft_row_is_judged_on_what_is_known_at_launch():
    """termination=None: the ft row's recorder adds its own truncation, from the loop."""
    assert rft.quick_reasons(**{**CLEAN_RUN, "termination": None}) == []


def test_every_reason_that_applies_is_listed_not_just_the_first():
    reasons = rft.quick_reasons(
        tag="memorise", device="mps", real_backbone=False, termination="wall_clock_cap",
        corpus=rft.CorpusFacts(snapshot_not_run="capped", history_rows={"x": 1}),
        memorise_detail="3 of 9 real batches",
    )
    assert len(reasons) == 6
    assert "3 of 9 real batches" in reasons[0]


def test_the_row_is_quick_exactly_when_a_reason_stands(tmp_path):
    """``_recorder`` used to hardcode ``quick=True``."""
    from types import SimpleNamespace

    from qd_train.ledger import Ledger

    header = SimpleNamespace(data_snapshot_hash="d" * 64, tokenizer_hash="t" * 64)
    reader = SimpleNamespace(header=header)
    recipe = {"device": "cuda", "hidden": 128, "heads": 4}
    cost = rft._cost(device="cuda", n_gpus=1, usd_per_hour=1.49, instance="lambda-1xGH200")
    common = {"reader": reader, "seed": 0, "recipe": recipe, "run_kind": "eval", "notes": "t",
              "wall_clock_s": 1.0, "cost": cost}
    ledger = Ledger(tmp_path / "l.jsonl")
    clear = rft._recorder(ledger, quick_reasons=[], **common)
    assert (clear.quick, clear.quick_reason) == (False, None)
    both = rft._recorder(ledger, quick_reasons=["a", "b"], **common)
    assert (both.quick, both.quick_reason) == (True, "a; b")


def test_nothing_here_decides_seed_count():
    """A campaign unit runs one seed, so a row cannot see its family; the family's count is
    enforced by promotion_verdict and the driver's min_seeds. Pinned so a per-row seed
    reason is a deliberate change, not a drift."""
    import inspect

    assert "seed" not in inspect.signature(rft.quick_reasons).parameters
    assert math.isfinite(rft.MAX_WALL_CLOCK_CAP_S)
