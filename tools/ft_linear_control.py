"""Score an FT eval row against its char-n-gram linear control: ``paired_margin_vs_linear``.

## Why this is post hoc

Every FT eval row ``tools/real_ft_run.py --score-val`` has written carries
``paired_margin_vs_linear: not_run`` (``ledger/gh200-ft-commitpackft-2026-09-22.jsonl``): the
run has an accuracy and no verdict. The gate needs two things the GPU run does not need to
own -- a linear control fitted on the FT train split, which is CPU work that saturated 64
cores on the GH200 on 2026-09-22, and the model's **per-row** correctness on the val split.
This tool reads the second from the verdicts file the run writes (``--verdicts-out``), fits
the first here, pairs them by row key, and appends the gate as a new ``eval`` row.

## What it refuses, and why each refusal is not a pass

* **No verdicts file.** The historical FT rows cannot be retrofitted: ``_decode`` computed the
  per-row verdicts and ``_record_score`` kept only the counts. Nothing here reconstructs them.
* **A verdicts file that is not this eval row's.** Every line names its ``eval_row_id`` and
  ``seed``, and the per-kind totals and correct counts must equal the eval row's own
  ``val_top1.*`` metrics. A file paired with the wrong row would produce a margin against a
  model nobody trained.
* **Different populations.** :func:`qd_train.eval_harness.paired_margin_by_key` pairs by
  key and refuses anything but identical key sets.
* **An unconverged control, or one with a single class.** Either is recorded ``not_run``:
  a model cannot beat a control that never trained.

## Its input is the model's input

The control reads ``render(...).prompt_for(slot)`` -- every byte the model conditions on --
through :func:`qd_train.baseline.request_texts`, and is labelled by gold **value**, not by
letter (the letter is a per-example permutation artefact; see ``RequestDoc``). One control
is fitted per task (``family_id/slot_name``), because one task's labels are not candidate
answers to another. Span slots are not scored: a pointer's answer is a line pair a bag of
n-grams cannot produce, and the row says so.

## The rows

``real_ft_run.ft_split_rows`` rebuilds the train/val split exactly as the run did. It is
imported **by name** and a missing function is a refusal: a second copy of the rebuild here
would be free to drift into scoring a different split while still printing a margin.

## The operator-holdout arm

``--hold-out-operator OP --operator-key KEY`` drops every train row whose
``metadata[KEY] == OP`` before fitting, which is the control for an FT arm trained on the
same reduced set (the rung-0 design in ``AUDIT/operator-holdout-model.md``). Validation is
never filtered. The eval row's recipe must name the same holdout, or the control is not that
arm's opponent. Beside the gate it reports the margins the rung-0 report reads: on the
operator's own val rows, on its **siblings** (same task and gold value, other operator --
the rows that separate "lost the generator" from "the prior moved"), and overall.

RUN (on the machine that has torch; it imports the FT runner for the split)
---
    /Users/bharath/.venvs/ml/bin/python tools/ft_linear_control.py \\
      --ledger ledger/<file>.jsonl --verdicts <verdicts.jsonl> \\
      --commitpackft data/pool/commitpackft --max-pairs 2000 --rev <sha>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tools"))

from repo_git import require_full_sha  # noqa: E402

from qd_data.config import DataConfig  # noqa: E402
from qd_data.rows import DataRow  # noqa: E402
from qd_train.baseline import (  # noqa: E402
    CharNGramHasher,
    ContextLengthFeatures,
    Featurizer,
    LinearBaseline,
    RequestDoc,
    fit_budget_refusal,
    request_texts,
)
from qd_train.control_cache import control_key, load_control, store_control  # noqa: E402
from qd_train.eval_harness import paired_margin_by_key  # noqa: E402
from qd_train.ledger import (  # noqa: E402
    Environment,
    Ledger,
    LedgerRow,
    Protocol,
    RunRecorder,
)
from qd_train.tristate import NotRun, Ran, TriState, aggregate  # noqa: E402

#: The gate this tool exists to evaluate, spelled once.
GATE: Final[str] = "paired_margin_vs_linear"

#: The control's iteration budget. The same number as ``rung0_linear_control`` and
#: ``rung0_real_run.LINEAR_CONTROL_MAX_ITER``; restated rather than imported because both of
#: those import torch at module scope, and ``test_ft_linear_control`` pins the equality where
#: torch exists. Raising it is allowed; lowering it to fit a cap weakens the opponent.
DEFAULT_MAX_ITER: Final[int] = 6_000

#: Letter kinds the eval row reports as ``val_top1.<kind>``; the verdicts must agree with them.
LETTER_KIND_NAMES: Final[tuple[str, ...]] = ("choice", "score")

Key = tuple[str, str]


class Refused(SystemExit):
    """A precondition failed. Exits non-zero with the reason; no row is written."""


# --- verdicts --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdicts:
    """One eval row's per-row model correctness, keyed the way the control is keyed."""

    eval_row_id: str
    seed: int
    #: ``(row_id, slot_name)`` when the file carries slot names, else ``(row_id, kind)``.
    correct: dict[Key, bool]
    kind_of: dict[Key, str]
    by_slot_name: bool
    span_rows: int


def load_verdicts(path: Path) -> Verdicts:
    """Parse the ``--verdicts-out`` JSONL, refusing anything that could mispair.

    Keyed by ``(row_id, slot_name)`` when every line has a slot name, else by
    ``(row_id, kind)`` -- and then a row with two slots of one kind cannot be told apart, so
    a duplicate key is a refusal rather than a last-one-wins overwrite.
    """
    if not path.is_file():
        raise Refused(
            f"no verdicts file at {path}. It is written by tools/real_ft_run.py "
            "--score-val --verdicts-out; a run that did not write one has no per-row "
            "correctness to pair, and nothing here reconstructs it"
        )
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise Refused(f"{path} holds no verdicts")
    #: (eval_row_id, seed, row_id, kind, slot_name or None, correct), each field checked.
    records: list[tuple[str, int, str, str, str | None, bool]] = []
    for n, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Refused(f"{path}:{n}: not JSON ({exc})") from exc
        if not isinstance(rec, dict):
            raise Refused(f"{path}:{n}: a verdict must be an object")
        for field, typ in (("eval_row_id", str), ("row_id", str), ("kind", str)):
            if not isinstance(rec.get(field), typ):
                raise Refused(f"{path}:{n}: {field!r} missing or not a {typ.__name__}")
        if not isinstance(rec.get("correct"), bool):
            raise Refused(f"{path}:{n}: 'correct' must be a JSON boolean")
        seed = rec.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise Refused(f"{path}:{n}: 'seed' must be an integer")
        slot = rec.get("slot_name")
        if slot is not None and not isinstance(slot, str):
            raise Refused(f"{path}:{n}: 'slot_name' must be a string when present")
        records.append(
            (rec["eval_row_id"], seed, rec["row_id"], rec["kind"], slot, rec["correct"])
        )

    eval_ids = {r[0] for r in records}
    seeds = {r[1] for r in records}
    if len(eval_ids) != 1 or len(seeds) != 1:
        raise Refused(
            f"{path} mixes eval rows {sorted(eval_ids)} / seeds {sorted(seeds)}; one file is "
            "one eval row's verdicts"
        )
    with_slot = [r[4] is not None for r in records]
    if any(with_slot) and not all(with_slot):
        raise Refused(f"{path}: some verdicts carry slot_name and some do not")
    by_slot_name = all(with_slot)

    correct: dict[Key, bool] = {}
    kind_of: dict[Key, str] = {}
    span_rows = 0
    for _, _, row_id, kind, slot, hit in records:
        if kind == "span":
            span_rows += 1
            continue
        if kind not in LETTER_KIND_NAMES:
            raise Refused(f"{path}: unknown kind {kind!r} on row {row_id}")
        key: Key = (row_id, slot if (by_slot_name and slot is not None) else kind)
        if key in correct:
            raise Refused(
                f"{path}: two verdicts share key {key}. Without slot names a row with two "
                "slots of one kind cannot be paired; the run must write slot_name"
            )
        correct[key] = hit
        kind_of[key] = kind
    return Verdicts(
        eval_row_id=eval_ids.pop(), seed=seeds.pop(), correct=correct, kind_of=kind_of,
        by_slot_name=by_slot_name, span_rows=span_rows,
    )


def check_against_eval_row(verdicts: Verdicts, row: LedgerRow) -> None:
    """The file must be this row's: same id, same seed, same per-kind totals and hits."""
    if verdicts.eval_row_id != row.row_id:
        raise Refused(
            f"the verdicts are eval row {verdicts.eval_row_id}'s and --eval-row is {row.row_id}"
        )
    if verdicts.seed != row.protocol.seed:
        raise Refused(f"verdicts seed {verdicts.seed} != eval row seed {row.protocol.seed}")
    if row.run_kind != "eval":
        raise Refused(f"row {row.row_id} is run_kind {row.run_kind!r}, not 'eval'")
    for kind in LETTER_KIND_NAMES:
        state = row.metrics.get(f"val_top1.{kind}")
        mine = [k for k, v in verdicts.kind_of.items() if v == kind]
        hits = sum(verdicts.correct[k] for k in mine)
        if not isinstance(state, Ran):
            if mine:
                raise Refused(
                    f"the verdicts hold {len(mine)} {kind} row(s) and the eval row scored none"
                )
            continue
        if (state.n_total, state.n) != (len(mine), hits):
            raise Refused(
                f"val_top1.{kind} on the eval row is {state.n}/{state.n_total} and the "
                f"verdicts say {hits}/{len(mine)}: this file is not that run's decode"
            )


def find_row(rows: Sequence[LedgerRow], row_id: str) -> LedgerRow:
    """The one row whose id starts with ``row_id``, or a refusal."""
    hits = [r for r in rows if r.row_id.startswith(row_id)]
    if len(hits) != 1:
        raise Refused(f"--eval-row {row_id!r} matches {len(hits)} row(s); need exactly one")
    return hits[0]


# --- the control -----------------------------------------------------------------------


def doc_key(doc: RequestDoc, *, by_slot_name: bool) -> Key:
    return (doc.row_id, doc.slot_name if by_slot_name else doc.kind)


@dataclass(frozen=True)
class HoldOut:
    """Which training rows the arm and its control never saw."""

    key: str
    operator: str


def apply_holdout(train: list[RequestDoc], hold: HoldOut | None) -> tuple[list[RequestDoc], int]:
    """``(kept, removed)``. Refuses a key no row carries or an operator no row has.

    Both are silent no-ops otherwise: the control would be fitted on the full set, beat the
    arm on data the arm never saw, and the report would call it a holdout.
    """
    if hold is None:
        return train, 0
    if not any(hold.key in d.metadata for d in train):
        raise Refused(
            f"no training row carries metadata {hold.key!r}, so --hold-out-operator "
            f"{hold.operator!r} would remove nothing. The family that records operators "
            "(A3's code.defect_class) must be in this split"
        )
    kept = [d for d in train if d.metadata.get(hold.key) != hold.operator]
    removed = len(train) - len(kept)
    if removed == 0:
        raise Refused(f"no training row has {hold.key}={hold.operator!r}")
    return kept, removed


@dataclass(frozen=True)
class ControlArm:
    """One control: what it is called in the row, how it featurizes, what it reads."""

    name: str
    make_features: Callable[[], Featurizer]
    text_of: Callable[[RequestDoc], str]
    #: Only the n-gram arm is cached: its fit is the expensive one, and the cache key's
    #: ``hasher_params`` names n-gram parameters, which the length arm does not have.
    cacheable: bool


#: The gate's opponent: every byte of the prompt, as hashed char n-grams.
NGRAM_ARM: Final[ControlArm] = ControlArm(
    name="linear_control", make_features=CharNGramHasher, text_of=lambda d: d.text,
    cacheable=True,
)
#: GAP-A3-CLEAN-DIFFS-ARE-LONGER-THAN-MUTATION-DIFFS: the context's size and nothing else.
#: Reported beside the gate, never folded into it -- the gate is the n-gram margin.
LENGTH_ARM: Final[ControlArm] = ControlArm(
    name="length_control", make_features=ContextLengthFeatures, text_of=lambda d: d.context,
    cacheable=False,
)
#: The metric name for the model's paired margin over the length-only control.
LENGTH_MARGIN: Final[str] = "paired_margin_vs_length_control"


@dataclass(frozen=True)
class TaskControl:
    task: str
    correct: dict[Key, bool]
    convergence: TriState
    accuracy: TriState
    fitted_s: float
    cached: bool


def fit_task(
    task: str,
    train: list[RequestDoc],
    val: list[RequestDoc],
    *,
    by_slot_name: bool,
    seed: int,
    max_iter: int,
    dense_budget_bytes: int | None,
    max_fit_minutes: float | None,
    cache_dir: Path | None,
    arm: ControlArm = NGRAM_ARM,
) -> TaskControl:
    """One task's control, scored on that task's val rows. Never raises for a weak fit:
    an unconverged or single-class control comes back ``not_run`` with its reason."""
    val_docs = [arm.text_of(d) for d in val]
    keys = [doc_key(d, by_slot_name=by_slot_name) for d in val]
    classes = sorted({d.value for d in train})
    if len(classes) < 2:
        reason = (
            f"task {task}: the training split holds {len(train)} row(s) with class(es) "
            f"{classes}; a control cannot be fitted on fewer than two classes"
        )
        return TaskControl(task, {}, NotRun(reason=reason), NotRun(reason=reason), 0.0, False)
    train_docs = [arm.text_of(d) for d in train]
    empty = sum(1 for t in (*train_docs, *val_docs) if not t)
    if empty:
        reason = (
            f"task {task}: {empty} doc(s) carry no context for the {arm.name}; scoring them "
            "as zero-length would make a constant, easily beaten control"
        )
        return TaskControl(task, {}, NotRun(reason=reason), NotRun(reason=reason), 0.0, False)
    train_labels = [d.value for d in train]
    features = arm.make_features()
    model = LinearBaseline(
        hasher=features, seed=seed, max_iter=max_iter, dense_budget_bytes=dense_budget_bytes
    )

    if cache_dir is not None and arm.cacheable and isinstance(features, CharNGramHasher):
        cache_key = control_key(
            train_docs=train_docs, train_labels=train_labels, val_docs=val_docs, seed=seed,
            max_iter=max_iter,
            hasher_params=(features.n_min, features.n_max, features.dim),
            l2_grid=model.l2_grid, tol=model.tol, lr=model.lr,
        )
    else:
        cache_key = None
    if cache_dir is not None and cache_key is not None:
        hit = load_control(cache_dir, cache_key, expected_n=len(val_docs))
        if hit is not None:
            correct = dict(zip(keys, (bool(x) for x in hit.correct), strict=True))
            return TaskControl(
                task, correct,
                Ran(passed=True, value=hit.final_grad_norm,
                    detail=f"cached: converged in {hit.iterations} iterations at l2={hit.l2}"),
                _accuracy(task, correct), hit.fitted_s, True,
            )

    refusal = fit_budget_refusal(
        model.projected_fit_seconds(train_docs, n_classes=len(classes)), max_fit_minutes
    )
    if refusal is not None:
        raise Refused(f"task {task}: {refusal}")
    started = time.monotonic()
    model.fit(train_docs, train_labels)
    fitted_s = time.monotonic() - started
    convergence = model.convergence()
    if not (isinstance(convergence, Ran) and convergence.passed):
        reason = convergence.reason if isinstance(convergence, NotRun) else convergence.detail
        return TaskControl(task, {}, convergence, NotRun(reason=reason), fitted_s, False)
    predicted = model.predict(val_docs)
    hits = [p == d.value for p, d in zip(predicted, val, strict=True)]
    fit = model.fit_
    if fit is None:  # pragma: no cover - convergence() refused an unfitted model above
        raise RuntimeError("a converged control has no fit")
    if cache_dir is not None and cache_key is not None:
        store_control(
            cache_dir, cache_key, hits, fitted_s=fitted_s, n_train=len(train_docs),
            l2=fit.l2, iterations=fit.iterations, final_grad_norm=fit.final_grad_norm,
        )
    correct = dict(zip(keys, hits, strict=True))
    return TaskControl(
        task, correct, convergence, _accuracy(task, correct, arm.name), fitted_s, False
    )


def _accuracy(task: str, correct: Mapping[Key, bool], arm: str = "linear_control") -> TriState:
    if not correct:
        return NotRun(reason=f"task {task} has no val rows")
    hits = sum(correct.values())
    return Ran(
        passed=True, value=hits / len(correct), n=hits, n_total=len(correct),
        detail=f"{arm} top-1 on task {task}'s val rows",
    )


@dataclass(frozen=True)
class ControlScore:
    gate: TriState
    metrics: dict[str, TriState]
    train_rows: int
    removed_by_holdout: int
    fit_seconds: float


def score_against_control(
    train: list[RequestDoc],
    val: list[RequestDoc],
    verdicts: Verdicts,
    *,
    seed: int,
    max_iter: int = DEFAULT_MAX_ITER,
    dense_budget_bytes: int | None = None,
    max_fit_minutes: float | None = None,
    cache_dir: Path | None = None,
    hold: HoldOut | None = None,
) -> ControlScore:
    """The gate and its supporting metrics. Pure apart from the optional cache.

    The gate is the paired margin over every letter row pooled; it is ``not_run`` if any
    task's control did not run, because a pooled margin over the tasks that happened to fit
    would be a capped sample reported as complete coverage.
    """
    train, removed = apply_holdout(train, hold)
    by_task_train: dict[str, list[RequestDoc]] = {}
    for d in train:
        by_task_train.setdefault(d.task, []).append(d)
    by_task_val: dict[str, list[RequestDoc]] = {}
    for d in val:
        by_task_val.setdefault(d.task, []).append(d)

    metrics: dict[str, TriState] = {}
    control: dict[Key, bool] = {}
    fit_seconds = 0.0
    task_states: dict[str, TriState] = {}
    for task in sorted(by_task_val):
        tc = fit_task(
            task, by_task_train.get(task, []), by_task_val[task],
            by_slot_name=verdicts.by_slot_name, seed=seed, max_iter=max_iter,
            dense_budget_bytes=dense_budget_bytes, max_fit_minutes=max_fit_minutes,
            cache_dir=cache_dir,
        )
        fit_seconds += tc.fitted_s
        metrics[f"linear_control_convergence.{task}"] = tc.convergence
        metrics[f"linear_control_top1.{task}"] = tc.accuracy
        task_states[task] = tc.convergence
        control.update(tc.correct)

    # The length-only control, per task, beside the n-gram one and never inside the gate.
    length_control: dict[Key, bool] = {}
    length_states: dict[str, TriState] = {}
    for task in sorted(by_task_val):
        lc = fit_task(
            task, by_task_train.get(task, []), by_task_val[task],
            by_slot_name=verdicts.by_slot_name, seed=seed, max_iter=max_iter,
            dense_budget_bytes=dense_budget_bytes, max_fit_minutes=max_fit_minutes,
            cache_dir=None, arm=LENGTH_ARM,
        )
        fit_seconds += lc.fitted_s
        metrics[f"length_control_convergence.{task}"] = lc.convergence
        metrics[f"length_control_top1.{task}"] = lc.accuracy
        length_states[task] = lc.convergence
        length_control.update(lc.correct)
    length_ran = aggregate(length_states, name="length_control_convergence")
    if not (isinstance(length_ran, Ran) and length_ran.passed):
        why = length_ran.reason if isinstance(length_ran, NotRun) else length_ran.detail
        metrics[LENGTH_MARGIN] = NotRun(reason=f"a task's length control did not run: {why}")
    else:
        metrics[LENGTH_MARGIN] = paired_margin_by_key(verdicts.correct, length_control, seed=seed)

    ran = aggregate(task_states, name="linear_control_convergence")
    if not (isinstance(ran, Ran) and ran.passed):
        reason = ran.reason if isinstance(ran, NotRun) else ran.detail
        gate: TriState = NotRun(reason=f"a task's control did not run: {reason}")
    else:
        gate = paired_margin_by_key(verdicts.correct, control, seed=seed)

    for kind in LETTER_KIND_NAMES:
        keys = [k for k, v in verdicts.kind_of.items() if v == kind]
        if keys and isinstance(gate, Ran):
            metrics[f"{GATE}.{kind}"] = _subset_margin(verdicts.correct, control, keys, seed)
    metrics[f"{GATE}.span"] = NotRun(
        reason=(
            f"{verdicts.span_rows} span row(s) not scored: a pointer's answer is a line "
            "pair the char-n-gram control cannot produce"
        )
    )
    if hold is not None:
        metrics.update(_holdout_metrics(val, verdicts, control, hold, seed=seed, gate=gate))
    return ControlScore(gate, metrics, len(train), removed, fit_seconds)


def _subset_margin(
    model: Mapping[Key, bool], control: Mapping[Key, bool], keys: Sequence[Key], seed: int
) -> TriState:
    if not keys:
        return NotRun(reason="the subset holds no rows")
    missing = [k for k in keys if k not in model or k not in control]
    if missing:
        return NotRun(reason=f"{len(missing)} subset row(s) unscored by one arm")
    return paired_margin_by_key(
        {k: model[k] for k in keys}, {k: control[k] for k in keys}, seed=seed
    )


def _holdout_metrics(
    val: list[RequestDoc], verdicts: Verdicts, control: Mapping[Key, bool], hold: HoldOut,
    *, seed: int, gate: TriState,
) -> dict[str, TriState]:
    """Own rows, siblings and their counts: what ``operator_holdout_report`` reads."""
    by = verdicts.by_slot_name
    own = [d for d in val if d.metadata.get(hold.key) == hold.operator]
    classes = {(d.task, d.value) for d in own}
    siblings = [
        d for d in val
        if d.metadata.get(hold.key) not in (None, hold.operator) and (d.task, d.value) in classes
    ]
    out: dict[str, TriState] = {}
    for name, docs in (("own", own), ("siblings", siblings)):
        keys = [doc_key(d, by_slot_name=by) for d in docs]
        if not isinstance(gate, Ran):
            out[f"{GATE}.holdout.{name}"] = NotRun(reason="the pooled gate did not run")
        else:
            out[f"{GATE}.holdout.{name}"] = _subset_margin(verdicts.correct, control, keys, seed)
        for arm, table in (("model", verdicts.correct), ("control", control)):
            scored = [table[k] for k in keys if k in table]
            out[f"holdout.{name}.{arm}_top1"] = (
                Ran(passed=True, value=sum(scored) / len(scored), n=sum(scored),
                    n_total=len(scored), detail=f"{hold.key}={hold.operator!r} {name} rows")
                if scored else NotRun(reason=f"no {name} rows were scored by the {arm}")
            )
    return out


# --- main ------------------------------------------------------------------------------


def split_rows_function() -> Callable[..., tuple[list[DataRow], list[DataRow]]]:
    """``real_ft_run.ft_split_rows``, by name, or a refusal. Never a local copy."""
    try:
        import real_ft_run
    except SystemExit as exc:  # real_ft_run refuses to import without torch
        raise Refused(f"tools/real_ft_run.py could not be imported: {exc}") from exc
    fn = getattr(real_ft_run, "ft_split_rows", None)
    if not callable(fn):
        raise Refused(
            "tools/real_ft_run.py has no ft_split_rows(). This tool rebuilds the split by "
            "calling the run's own function and refuses to carry a copy of it, because a "
            "copy that drifted would score the control on a different split and still print "
            "a margin. It is owned by the lane that owns real_ft_run.py."
        )
    return fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, required=True, help="where the eval row is")
    parser.add_argument(
        "--eval-row", default=None,
        help=(
            "the eval row's id (a prefix is ok). Default: the eval_row_id the verdicts file "
            "names, so a campaign config can name the file without knowing the id in advance"
        ),
    )
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument(
        "--write-ledger", type=Path, default=None,
        help="where to append the gate row; default: --ledger, beside the row it scores",
    )
    parser.add_argument("--commitpackft", type=Path, default=None)
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="exactly as the run was given it. Required, except under --no-repo-history "
             "without --commitpackft, where it bounds nothing and is refused",
    )
    parser.add_argument(
        "--no-repo-history", dest="repo_history", action="store_false",
        help="exactly as the run was given it: the split holds no row from this repository's "
             "git history",
    )
    parser.add_argument("--rev", required=True, help="a full 40-character sha")
    parser.add_argument(
        "--defect-class", type=Path, default=None,
        help="the code.defect_class corpus, exactly as the run was given it: it is part of how "
             "the run built its split, so a control rebuilt without it scores a different split",
    )
    parser.add_argument("--defect-download", type=Path, default=None)
    parser.add_argument("--defect-max-rows", type=int, default=None)
    parser.add_argument("--control-cache", type=Path, default=None)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--dense-budget-gb", type=float, default=24.0)
    parser.add_argument("--max-fit-minutes", type=float, default=None)
    parser.add_argument("--hold-out-operator", default="")
    parser.add_argument("--operator-key", default="operator")
    args = parser.parse_args(argv)
    if args.defect_class is None and (
        args.defect_download is not None or args.defect_max_rows is not None
    ):
        parser.error("--defect-download/--defect-max-rows without --defect-class read nothing")
    max_pairs_bounds_nothing = not args.repo_history and args.commitpackft is None
    if max_pairs_bounds_nothing and args.max_pairs is not None:
        raise Refused(
            "--max-pairs bounds the repository-history rows and the --commitpackft sample; "
            "under --no-repo-history without --commitpackft it bounds nothing"
        )
    if not max_pairs_bounds_nothing and args.max_pairs is None:
        raise Refused("--max-pairs is required: it decides which rows the split holds")
    # This tool writes a ledger row, and the split it rebuilds is named by the revision.
    try:
        rev = require_full_sha(args.rev)
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    # Unread by the rebuild when it bounds nothing; ft_split_rows takes an int.
    max_pairs = 0 if args.max_pairs is None else args.max_pairs

    ledger = Ledger(args.ledger)
    verdicts = load_verdicts(args.verdicts)
    row = find_row(ledger.rows(), args.eval_row or verdicts.eval_row_id)
    check_against_eval_row(verdicts, row)
    recorded_holdout = str((row.recipe or {}).get("hold_out_operator", ""))
    if recorded_holdout != args.hold_out_operator:
        raise Refused(
            f"the eval row's model was trained with hold_out_operator={recorded_holdout!r} "
            f"and this control would hold out {args.hold_out_operator!r}: it would not be "
            "that arm's opponent"
        )
    hold = HoldOut(args.operator_key, args.hold_out_operator) if args.hold_out_operator else None

    config = DataConfig()
    train_rows, val_rows = split_rows_function()(
        commitpackft=args.commitpackft, max_pairs=max_pairs, rev=rev, config=config,
        defect_class=args.defect_class, defect_download=args.defect_download,
        defect_max_rows=args.defect_max_rows, repo_history=args.repo_history,
    )
    # Rule 3 through this door too. A control fitted on a held-out family would not train a
    # model, but it would set the bar the model is measured against with data the model may
    # never see -- the same violation wearing a different hat (fit_linear_control.py).
    held = sorted({r.family_id for r in (*train_rows, *val_rows)
                   if config.is_held_out_family(r.family_id)})
    if held:
        raise Refused(
            f"the split handed to the control holds rows of held-out famil(ies) {held}; "
            "rule 3 refuses them to anything that trains or sets a training bar"
        )
    train_docs, train_excluded = request_texts(train_rows, seed=config.seed)
    val_docs, val_excluded = request_texts(val_rows, seed=config.seed)
    print(f"eval row {row.row_id} seed {row.protocol.seed}: {len(verdicts.correct)} letter "
          f"verdict(s), {verdicts.span_rows} span")
    print(f"control docs: {len(train_docs)} train ({len(train_excluded)} excluded), "
          f"{len(val_docs)} val ({len(val_excluded)} excluded); tasks "
          f"{dict(Counter(d.task for d in val_docs))}")

    recipe = {
        "tool": "tools/ft_linear_control.py", "eval_row_id": row.row_id,
        "max_iter": args.max_iter, "hold_out_operator": args.hold_out_operator,
        "operator_key": args.operator_key if hold else "", "key": "slot_name"
        if verdicts.by_slot_name else "kind", "max_pairs": args.max_pairs, "rev": rev,
        "repo_history": args.repo_history,
        "defect_class": None if args.defect_class is None else args.defect_class.name,
        "defect_max_rows": args.defect_max_rows,
    }
    quick = bool(row.quick) or hold is not None
    quick_reason = (
        f"inherits eval row {row.row_id[:8]}'s quick flag ({row.quick_reason})"
        if row.quick else "an operator-holdout arm is a diagnostic; it promotes nothing"
    ) if quick else None
    # The recorder WRAPS the fit: on a rented box this CPU work is billed with the instance,
    # and a fit killed outside a block would leave no row saying it ran.
    with RunRecorder(
        Ledger(args.write_ledger or args.ledger), entry_point=Path(__file__),
        # The eval row's protocol, so this gate row joins that configuration's seed family.
        protocol=Protocol(**row.protocol.to_json()), run_kind="eval", repo=REPO,
        env=Environment.detect(device="cpu"), wall_clock_s=None, cost=None,
        recipe=recipe, quick=quick, quick_reason=quick_reason,
        notes=(f"tools/ft_linear_control.py: paired_margin_vs_linear for eval row {row.row_id}"
               f" over {len(verdicts.correct)} letter rows"),
    ) as recorder:
        started = time.monotonic()
        result = score_against_control(
            train_docs, val_docs, verdicts, seed=row.protocol.seed, max_iter=args.max_iter,
            dense_budget_bytes=int(args.dense_budget_gb * 1024**3),
            max_fit_minutes=args.max_fit_minutes, cache_dir=args.control_cache, hold=hold,
        )
        recorder.measured(time.monotonic() - started)
        recorder.metric("scored_eval_row_id", Ran(passed=True, value=row.row_id,
                                                  detail="the eval row these verdicts are"))
        recorder.metric("control_train_rows", Ran(
            passed=True, value=result.train_rows, detail=(
                f"letter slots the control was fitted on; the holdout removed "
                f"{result.removed_by_holdout}")))
        for name, state in result.metrics.items():
            recorder.metric(name, state)
        recorder.gate(GATE, result.gate)
    written = recorder.row
    print(f"{GATE}: {result.gate}")
    print(f"wrote row {written.row_id if written else '?'} to {args.write_ledger or args.ledger}")
    return 0 if isinstance(result.gate, Ran) else 3


if __name__ == "__main__":
    raise SystemExit(main())
