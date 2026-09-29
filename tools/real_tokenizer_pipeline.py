"""Drive the whole data -> shards path with the REAL Qwen tokenizer, over real code.

Every test of this path drives it with a byte-level stand-in (``test_shards.py``'s
``byte_tokenize``/``byte_offsets``/``byte_decode``), and that stand-in cannot fail the
checks that exist for BPE: in byte space no token straddles a line boundary, every
``decode([id])`` is exact, and the trailing answer letter is always a token of its own.
``tools/bpe_line_start_collapse.py`` measured *one* piece of the path against the real
tokenizer. This script runs the rest of it: mixture -> dedupe -> split -> manifest ->
``write_shards`` -> ``ShardReader``, with ``tokenize``, ``token_offsets`` **and**
``decode`` all supplied by ``Qwen/Qwen3.5-2B-Base``.

The corpus is real. By default the code rows are this repository's own git history: every
``(commit, path)`` pair whose file was modified by that commit, with the true before and
after contents and the true commit message, rewritten by ``qd_data.mixture`` exactly as a
``bigcode/commitpackft`` row would be. ``--commitpackft`` takes them from the plan's own
``bigcode/commitpackft`` download instead, each file pinned by its manifest's sha256. The
``qa.answer_span`` rows are built over real prose from the tracked Markdown, with real
answer offsets, under either source. Nothing is fetched at run time: the HF
datasets-server was returning HTTP 500 when this was written, and a corpus that needs the
network makes the measurement unrepeatable anyway.

Three things it is careful about:

* **Both numbers, everywhere.** Every stage reports rows in and rows out. A rate alone
  hides whether the denominator was the corpus or what survived the previous stage.
* **Every refusal is classified, counted and exampled.** ``--census`` mirrors the per-row
  work ``write_shards`` does, in the same order, so a refusal can be attributed to the
  function that raised it; the census total is then cross-checked against the
  ``coverage.json`` the real write produces, and a disagreement is reported rather than
  reconciled.
* **The decode check is run.** ``write_shards``'s ``decode=`` argument is the only check in
  the module that does not consult the offsets it is checking, and the repo venv has no
  tokenizer to decode with, so it has never run against a real one.

Run it with the ML venv -- the repo venv deliberately carries no ``transformers``::

    /Users/bharath/.venvs/ml/bin/python tools/real_tokenizer_pipeline.py --out <dir>

``--max-pairs`` bounds the corpus. Any bound that actually binds is reported in the
summary as a cap, and the run is marked a capped sample: a capped read is never presented
as complete coverage.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Inlined rather than assigned to REPO first, matching tools/bpe_line_start_collapse.py:
# ruff's E402 exemption covers a `sys.path` modification before the imports, but an
# ordinary assignment in between is not one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from repo_git import git_bytes, git_text, require_full_sha, resolve_rev, tracked_paths

from qd_data.config import DataConfig
from qd_data.dedupe import dedupe
from qd_data.defect_class import (
    CHOICE_SLOT as DEFECT_CHOICE_SLOT,
)
from qd_data.defect_class import (
    DEFECT_FAMILY_ID,
    DEFECT_SOURCE_ID,
    DefectRow,
    load_defect_rows,
)
from qd_data.errors import QdRefusal
from qd_data.general import (
    CLINC_DOMAINS_COMMIT,
    REPLAY_FAMILIES,
    ClincDomainMap,
    ReplayPartition,
    load_clinc_domains,
    partition_replay,
)
from qd_data.loaders import (
    REFUSED_READS,
    CommitPackFtRow,
    RawRow,
    SquadRow,
    parse_clinc,
    parse_commitpackft,
    parse_csqa,
    parse_mmlu,
    parse_squad,
    read_jsonl,
)
from qd_data.manifest import build_manifests
from qd_data.mixture import build_mixture
from qd_data.rows import DataRow
from qd_data.schema import SpanSlot
from qd_data.split import HELD_OUT, SplitReport, split
from qd_train import shards as shards_module
from qd_train.artifacts import (
    SLOT_SPAN,
    RemapTable,
    ShardContractViolation,
    assign_buckets,
    padding_waste,
)
from qd_train.ledger import (
    DEFAULT_LEDGER_PATH,
    Environment,
    Ledger,
    Protocol,
    RunRecorder,
)
from qd_train.remap import build_remap, count_corpus_tokens
from qd_train.shards import (
    COVERAGE_NAME,
    HEADER_NAME,
    SPAN_CHECK_NAME,
    ShardReader,
    UnencodableGold,
    choose_buckets,
    training_texts,
    write_shards,
)
from qd_train.tristate import NotRun, Ran, TriState

REPO = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.5-2B-Base"

#: Where the HF cache records which revision of ``MODEL`` is on this host. Read rather than
#: written down, so the ledger row names the snapshot that actually produced the numbers.
MODEL_REF = (
    Path.home() / ".cache/huggingface/hub"
    / f"models--{MODEL.replace('/', '--')}" / "refs" / "main"
)

#: Extensions this repository's history offers, mapped onto the closed label set
#: ``qd_data.mixture.LANGUAGE_OPTIONS`` uses. A suffix outside this table is skipped at
#: corpus-build time rather than refused at rewrite time: it would only ever produce the
#: ``lang_not_in_option_set`` refusal, which says nothing about the tokenizer.
LANG_BY_SUFFIX: dict[str, str] = {".py": "Python", ".rs": "Rust", ".md": "Markdown"}

#: The repository's own licence, which is what these rows actually carry.
REPO_LICENCE = "apache-2.0"

#: Bound on the corpus. Stated, not defaulted away: `count_corpus_tokens` demands a
#: sequence bound for the same reason, and an unbounded read here is an unbounded memmap
#: three stages later.
DEFAULT_MAX_PAIRS = 400

#: Bound on the tokenizer memo. A memo without one is a second unbounded allocation
#: hiding behind a speed optimisation.
MEMO_LIMIT = 8192

#: The ``bigcode/commitpackft`` download ``--defect-class`` joins licences from: the qd-mutate
#: corpus and its pool carry none (``qd_data.defect_class``).
DEFAULT_DEFECT_DOWNLOAD = REPO / "data" / "pool" / "commitpackft"


# -- the corpus -------------------------------------------------------------------------


def _git(*args: str) -> str:
    """This repository, bound. The implementation is ``repo_git.git_text``.

    A one-line adapter rather than a second copy: ``tools/rung0_real_run.py`` needs the
    same two calls, and when it had its own pair ``devmap_clones`` reported them as an
    Exact group. Nine call sites below pass only the git arguments, so binding ``REPO``
    here is what keeps them unchanged.
    """
    return git_text(REPO, *args)


def _git_bytes(*args: str) -> bytes:
    """As :func:`_git`, for a blob. See ``repo_git.git_bytes`` on why it stays undecoded."""
    return git_bytes(REPO, *args)


def _split_unit(name: str) -> str:
    """The first two path components of a tracked path, or the file itself at the root."""
    parts = Path(name).parts
    return "/".join(parts[:2]) if len(parts) > 1 else name


def commit_rows(*, max_pairs: int, rev: str) -> tuple[list[CommitPackFtRow], bool]:
    """Real ``(commit, path)`` pairs from this repository, as commitpackft rows.

    Returns the rows and whether the bound actually bound. Only files the commit
    *modified* are used: an added file has no before-contents, and inventing one
    ("" as the old side) would manufacture a diff the repository never contained.

    ``rev`` is explicit because several lanes share this worktree and commit to it while a
    run is in progress: two runs against "HEAD" hours apart read different corpora and
    their numbers are not comparable. This one was measured against a named revision.
    """
    rows: list[CommitPackFtRow] = []
    capped = False
    for commit in _git("rev-list", rev).split():
        # `check=False`: the root commit has no parent and `rev-parse` exits 1 for it.
        # That is the expected answer for one commit in every history, not an error.
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{commit}^"],
            cwd=REPO, capture_output=True, check=False, text=True,
        )
        parents = probe.stdout.split()
        if not parents:
            continue
        parent = parents[0]
        message = _git("log", "-1", "--format=%B", commit).strip()
        subject = _git("log", "-1", "--format=%s", commit).strip()
        names = _git(
            "diff", "--name-only", "--diff-filter=M", parent, commit
        ).splitlines()
        for name in names:
            suffix = Path(name).suffix
            if suffix not in LANG_BY_SUFFIX:
                continue
            if len(rows) >= max_pairs:
                capped = True
                return rows, capped
            try:
                old = _git_bytes("show", f"{parent}:{name}").decode("utf-8")
                new = _git_bytes("show", f"{commit}:{name}").decode("utf-8")
            except (subprocess.CalledProcessError, UnicodeDecodeError):
                continue
            rows.append(
                CommitPackFtRow(
                    commit=commit,
                    # commitpackft's split unit is its `repos` field. This repository is
                    # one repo, so the nearest real structural unit stands in for it: the
                    # first two path components, which is what actually separates
                    # `python/qd_train` from `crates/qd-mutate`. Stated rather than
                    # silently one constant, because with one group the splitter puts
                    # every row on one side and its disjointness checks report not_run.
                    repos=f"qwen-decision/{_split_unit(name)}",
                    old_file=name,
                    new_file=name,
                    old_contents=old,
                    new_contents=new,
                    subject=subject,
                    message=message,
                    lang=LANG_BY_SUFFIX[suffix],
                    licence=REPO_LICENCE,
                )
            )
    return rows, capped


def pool_manifest_shas(root: Path) -> dict[str, str]:
    """``{language: sha256}`` as the local commitpackft download's manifest records them."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_id") != "bigcode/commitpackft":
        raise SystemExit(
            f"{root / 'manifest.json'} names source {manifest.get('source_id')!r}, not "
            "bigcode/commitpackft; refusing to read its rows as commitpackft rows"
        )
    return {lang: str(meta["sha256"]) for lang, meta in sorted(manifest["languages"].items())}


def commitpackft_pool_rows(
    root: Path, *, max_pairs: int
) -> tuple[list[CommitPackFtRow], bool, int]:
    """Rows from the plan's own code source: a local ``bigcode/commitpackft`` download.

    :func:`commit_rows` renders this repository's history AS commitpackft rows, which
    measures the tokenizer path on real code but trains on one repository's commits. The
    plan names commitpackft itself, and it has been on disk under ``data/pool/commitpackft``
    since 2026-09-21: one JSONL per language, each pinned by a sha256 in ``manifest.json``.

    Every file is checked against its recorded sha256 before a row is read, and a mismatch
    refuses the run: a file that changed after it was downloaded holds rows nobody can name.

    A cap is a SAMPLE, not a prefix. The files are read in language order, so the first
    ``max_pairs`` rows would all be Go; instead every row is ordered by a sha256 of its
    ``(commit, old_file)`` and the first ``max_pairs`` are kept, which is reproducible and
    keeps the language mix in expectation. Returns the rows, whether the cap bound, and how
    many rows the download held, so both numbers are reported.
    """
    expected = pool_manifest_shas(root)
    rows: list[CommitPackFtRow] = []
    for lang, sha in expected.items():
        path = root / f"{lang}.jsonl"
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha:
            raise SystemExit(
                f"{path} hashes to {actual} but {root / 'manifest.json'} records {sha}: the "
                "file changed after it was downloaded, so which rows it holds cannot be stated"
            )
        for index, line in enumerate(data.decode("utf-8").splitlines()):
            if line.strip():
                rows.append(parse_commitpackft(json.loads(line), index=index))
    total = len(rows)
    rows.sort(key=lambda r: hashlib.sha256(f"{r.commit}\0{r.old_file}".encode()).hexdigest())
    return rows[:max_pairs], total > max_pairs, total


def code_rows(
    *, commitpackft: Path | None, max_pairs: int, rev: str
) -> tuple[list[CommitPackFtRow], bool, int | None]:
    """The code rows a build with these arguments reads -- for the builder and every reader.

    ``tools/real_ft_run.py`` does not read its labels out of a shard set; it rebuilds them
    by re-running this corpus and refuses unless the result lines up with
    ``supervision.npz`` row for row. When ``--commitpackft`` arrived (7f23355) only this
    tool learned it, so the FT runner kept rebuilding this repository's history over a
    shard set built from the download. Measured on the first such set: "793 labels
    against 3364 sequences" -- a correct refusal, and no way to run at all. One function now
    answers "which rows" for both.

    Returns the rows, whether the cap bound, and how many rows the download held -- ``None``
    for this repository's history, whose population this function does not count.
    """
    if commitpackft is None:
        rows, capped = commit_rows(max_pairs=max_pairs, rev=rev)
        return rows, capped, None
    return commitpackft_pool_rows(commitpackft, max_pairs=max_pairs)


#: The sources whose rows :func:`base_sources` reads out of this repository's git history
#: when nothing else supplies them: the code rows (without ``--commitpackft``) and the span
#: prose that stands in for ``rajpurkar/squad_v2``.
REPO_HISTORY_SOURCES: tuple[str, ...] = ("bigcode/commitpackft", "rajpurkar/squad_v2")


@dataclass(frozen=True)
class BaseSources:
    """The code and span rows every build starts from, before the optional families."""

    raw: dict[str, list[Any]]
    #: Sources whose read hit ``max_pairs``; each makes the mixture ``NotRun``.
    capped: tuple[str, ...]
    n_commits: int
    n_spans: int
    #: Rows the commitpackft download held, or ``None`` when it was not read.
    pool_total: int | None
    #: Which of :data:`REPO_HISTORY_SOURCES` were read from this repository's history.
    from_history: tuple[str, ...]


def base_sources(
    *, repo_history: bool, commitpackft: Path | None, max_pairs: int, blank_line_runs: bool,
    rev: str,
) -> BaseSources:
    """Which code and span rows a build reads -- one owner for the builder and every rebuild.

    ``run`` and ``real_ft_run.ft_splits`` composed this dict separately, and a flag added to
    one of them would have left the other rebuilding a different corpus. ``repo_history``
    False reads nothing from this repository's git history: no span prose, and code rows only
    from the ``--commitpackft`` download when one is given. Absent sources are omitted rather
    than passed empty, so ``build_mixture`` asks nothing of their families.
    """
    raw: dict[str, list[Any]] = {}
    capped: list[str] = []
    from_history: list[str] = []
    commits: list[CommitPackFtRow] = []
    pool_total: int | None = None
    if commitpackft is not None or repo_history:
        commits, commits_capped, pool_total = code_rows(
            commitpackft=commitpackft, max_pairs=max_pairs, rev=rev
        )
        raw["bigcode/commitpackft"] = list(commits)
        if commits_capped:
            capped.append("bigcode/commitpackft")
        if commitpackft is None:
            from_history.append("bigcode/commitpackft")
    spans: list[SquadRow] = []
    if repo_history:
        spans, spans_capped = span_rows(
            max_rows=max_pairs, blank_line_runs=blank_line_runs, rev=rev
        )
        raw["rajpurkar/squad_v2"] = list(spans)
        if spans_capped:
            capped.append("rajpurkar/squad_v2")
        from_history.append("rajpurkar/squad_v2")
    return BaseSources(
        raw=raw, capped=tuple(capped), n_commits=len(commits), n_spans=len(spans),
        pool_total=pool_total, from_history=tuple(from_history),
    )


def quick_reason_for(
    *, commitpackft_rows: tuple[int, int] | None, repo_history: bool = True
) -> str:
    """Why a row this tool writes is ``quick``, stated for the corpus the run actually read.

    ``commitpackft_rows`` is ``(rows read, rows the download held)`` when ``--commitpackft``
    supplied the code rows, and ``None`` when this repository's history did.

    Until 2026-09-22 this was one literal in :func:`main`, written for the only corpus
    the tool then had. ``--commitpackft`` was added without touching it, so the first shard
    set built from the download -- row ``0c3fd775`` -- says its corpus was "drawn from this
    repository alone" beside a recipe pinning four commitpackft sha256s: the defect
    ``tools/rung0_real_run.quick_reason_for`` was written to prevent, repeated one tool
    over. The flag stays ``True`` for both sources: one seed, and the span rows are this
    repository's Markdown standing in for ``rajpurkar/squad_v2`` whichever source the code
    rows came from.

    ``repo_history=False`` (``--no-repo-history``) reads no span prose and no history rows,
    so neither stand-in sentence is true of it; it is still one seed.
    """
    if not repo_history:
        code = (
            "no code rows from bigcode/commitpackft"
            if commitpackft_rows is None
            else (
                f"{commitpackft_rows[0]} of {commitpackft_rows[1]} code rows from the plan's "
                "bigcode/commitpackft download"
            )
        )
        return (
            "one seed; --no-repo-history: no row was read from this repository's git "
            f"history ({code}, no repository-prose span rows). Repo rule 8: fewer than 3 "
            "seeds is marked quick and excluded from decisions."
        )
    if commitpackft_rows is None:
        return (
            "one seed, and a corpus drawn from this repository alone rather than from the "
            "pool the plan names. Repo rule 8: a subsample is marked quick and excluded "
            "from decisions."
        )
    read, held = commitpackft_rows
    code = (
        f"all {held} rows of the plan's bigcode/commitpackft download"
        if read >= held
        else (
            f"a {read}-of-{held} sha256-ordered sample of the plan's bigcode/commitpackft "
            "download (the --max-pairs cap)"
        )
    )
    # Rule 8 is quoted by its conditions rather than as "a subsample is quick": an uncapped
    # read is not a subsample, and the row is quick for its one seed regardless.
    return (
        f"one seed; the code rows are {code}, but the span rows are this repository's own "
        "Markdown standing in for rajpurkar/squad_v2. Repo rule 8: fewer than 3 seeds, or "
        "a subsample, is marked quick and excluded from decisions."
    )


def _paragraphs(text: str, *, min_lines: int) -> list[str]:
    """Blank-line-separated blocks of at least ``min_lines`` lines."""
    out: list[str] = []
    block: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            block.append(line)
        else:
            if len(block) >= min_lines:
                out.append("\n".join(block))
            block = []
    if len(block) >= min_lines:
        out.append("\n".join(block))
    return out


def span_rows(*, max_rows: int, blank_line_runs: bool, rev: str) -> tuple[list[SquadRow], bool]:
    """Span rows over real prose from this repository's tracked Markdown.

    The passage and the answer are real: the answer is a literal substring of the passage,
    at its real offset, so the ``(start_line, end_line)`` gold ``qd_data.mixture._line_span``
    derives is the real line range of real text.

    ``blank_line_runs`` controls the one thing about the passage that is not taken as
    found. With it false, paragraphs are joined by a single blank line, so no run of three
    newlines occurs. That is the documented workaround for
    ``GAP-S4-LINE-STARTS-COLLAPSE-UNDER-BPE``: ``'\\n\\n\\n'`` is one Qwen token and swallows
    two line starts, which the writer refuses, and with it left on almost every span row is
    refused before any of the rest of the span path runs. Both settings are measured.
    """
    rows: list[SquadRow] = []
    capped = False
    joiner = "\n\n\n" if blank_line_runs else "\n\n"
    for name in tracked_paths(REPO, rev=rev, suffixes=frozenset({".md"})):
        try:
            # Read at `rev`, not from the working tree: other lanes edit these files while
            # a run is in progress, and a corpus half from the tree and half from history
            # is not a corpus anyone can reproduce.
            text = _git_bytes("show", f"{rev}:{name}").decode("utf-8")
        except (subprocess.CalledProcessError, UnicodeDecodeError):
            continue
        blocks = _paragraphs(text, min_lines=3)
        if len(blocks) < 2:
            continue
        passage = joiner.join(blocks[:3])
        # A real token of the passage, long enough to be an answer rather than a word
        # fragment, and taken at its real offset -- never searched for by string, which
        # would land on the first of several occurrences.
        lines = [ln for ln in passage.split("\n") if len(ln.strip()) > 24]
        if len(lines) < 2:
            continue
        answer = lines[len(lines) // 2].strip()
        start = passage.index(answer)
        for impossible in (False, True):
            if len(rows) >= max_rows:
                capped = True
                return rows, capped
            # THE TWO QUESTIONS MUST DIFFER. Until 2026-09-20 both rows of this pair asked
            # `f"Which line of {name} states the rule?"` over the same passage, so the
            # prompt was byte-identical and only the gold differed -- one a real line span,
            # the other `noul`. That is a corpus no model can score above chance on, and the
            # REALFT lane measured exactly that: the span loss floor came out at
            # 0.693147 = ln 2, the entropy of a fair coin, and span accuracy was capped at
            # 39 of 78 permanently.
            #
            # Nothing in qd_data refused it, and that is not an oversight this file can fix:
            # `dedupe_text` is `f"{question}\n{passage}"`, identical for both rows, so
            # dedupe correctly makes them ONE content unit for leakage purposes and then
            # keeps both. The contract has no notion of two rows whose prompts agree and
            # whose golds contradict. Recorded separately; what this file owes is a corpus
            # where unanswerable means the passage does not answer THIS question.
            question = (
                f"Which line of {name} gives the release date?"
                if impossible
                else f"Which line of {name} states the rule?"
            )
            assert answer not in question, "the unanswerable question must not contain the gold"
            rows.append(
                SquadRow(
                    qid=f"{name}:{'imp' if impossible else 'ans'}",
                    title=name,
                    context=passage,
                    question=question,
                    answers=() if impossible else (answer,),
                    answer_starts=() if impossible else (start,),
                    is_impossible=impossible,
                )
            )
    return rows, capped


# -- the tokenizer ----------------------------------------------------------------------


@dataclass
class RealTokenizer:
    """``tokenize`` / ``token_offsets`` / ``decode``, all from the same live tokenizer.

    One encode per text, memoized, because ``write_shards`` asks for ids and offsets
    separately and the census asks for both again. The memo is bounded and *raises* when
    it fills rather than evicting silently: an eviction would make the run's cost depend
    on the order rows happen to arrive in.
    """

    tok: Any
    _memo: dict[str, tuple[list[int], list[tuple[int, int]]]]
    #: ``0`` disables the memo: every call encodes, so the cost is still independent of row
    #: order. That is the setting for a corpus far over :data:`MEMO_LIMIT` -- the 50,177-row
    #: ``code.defect_class`` corpus renders two sequences per row, and a memo holding ids
    #: and offsets for all of them is several GB of Python objects.
    memo_limit: int = MEMO_LIMIT

    @classmethod
    def load(cls, *, memo_limit: int = MEMO_LIMIT) -> RealTokenizer:
        if memo_limit < 0:
            raise SystemExit(f"--memo-limit must be >= 0, got {memo_limit}")
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError:
            raise SystemExit(
                "transformers is not importable from this interpreter. The repo venv does "
                "not carry it on purpose; run this with /Users/bharath/.venvs/ml/bin/python."
            ) from None
        return cls(
            tok=AutoTokenizer.from_pretrained(MODEL), _memo={}, memo_limit=memo_limit
        )

    def _encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        value = ([int(i) for i in enc["input_ids"]], [tuple(o) for o in enc["offset_mapping"]])
        if self.memo_limit == 0:
            return value
        if len(self._memo) >= self.memo_limit:
            raise RuntimeError(
                f"the tokenizer memo reached its {self.memo_limit}-entry bound. Raise the "
                "bound deliberately (--memo-limit), or pass --memo-limit 0 to encode every "
                "call, rather than evicting: an eviction policy would make this run's "
                "timing depend on row order."
            )
        self._memo[key] = value
        return value

    def tokenize(self, text: str) -> list[int]:
        return self._encode(text)[0]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self._encode(text)[1]

    def decode(self, ids: Any) -> str:
        return self.tok.decode([int(i) for i in ids], clean_up_tokenization_spaces=False)

    def decode_default_kwargs(self, ids: Any) -> str:
        """``tokenizer.decode`` exactly as a caller would reach for it, with no kwargs.

        Separate from :meth:`decode` because the difference is load-bearing:
        ``_assert_spans_decode_to_their_text`` requires the ids to round-trip to the text
        they were produced from, and HuggingFace's ``clean_up_tokenization_spaces`` rewrites
        punctuation spacing over a *sequence*. If the default rewrites, then passing
        ``tokenizer.decode`` straight into ``write_shards`` -- the obvious call -- refuses
        every span row with a message about the offsets, which is not where the fault is.
        """
        return self.tok.decode([int(i) for i in ids])

    def hash(self) -> str:
        """Identity of this tokenizer's vocabulary, for ``RemapTable.tokenizer_hash``."""
        vocab = json.dumps(self.tok.get_vocab(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(f"{MODEL}\n{vocab}".encode()).hexdigest()

    def byte_token_ids(self) -> frozenset[int]:
        """Ids of the 256 byte-level symbols, the pieces any text can always be spelled in.

        Refuses rather than returning fewer: a byte-fallback cost measured against a
        partial alphabet would under-count every token spelled with a missing byte.
        """
        from tokenizers.pre_tokenizers import ByteLevel

        symbols = sorted(ByteLevel.alphabet())
        ids = self.tok.convert_tokens_to_ids(symbols)
        found = {int(i) for i in ids if i is not None and i != self.tok.unk_token_id}
        if len(symbols) != 256 or len(found) != 256:
            raise SystemExit(
                f"{MODEL}'s vocabulary holds {len(found)} of the {len(symbols)} byte-level "
                "symbols as single tokens; a byte fallback cannot be priced against it"
            )
        return frozenset(found)

    def byte_lengths(self, token_ids: Iterable[int]) -> dict[int, int]:
        """Single-byte tokens needed to spell each id: its byte-level symbol count.

        Refuses an id whose symbols are not all byte-level -- an added or special token --
        because its length in bytes is not the length of its symbol string.
        """
        from tokenizers.pre_tokenizers import ByteLevel

        alphabet = frozenset(ByteLevel.alphabet())
        out: dict[int, int] = {}
        for token_id in token_ids:
            symbols = self.tok.convert_ids_to_tokens(int(token_id))
            if not symbols or not set(symbols) <= alphabet:
                raise SystemExit(
                    f"token id {token_id} is {symbols!r}, which is not spelled in byte-level "
                    "symbols; its byte length cannot be read off its string"
                )
            out[int(token_id)] = len(symbols)
        return out


# -- classifying what refused ------------------------------------------------------------

#: Message fragment -> the refusal class it names. Every fragment is a literal from the
#: refusing function, so a class here is attributable to a line of the module under test.
#: A refusal matching nothing is reported as ``unclassified``, never folded into a
#: neighbour: an unattributed refusal is the one worth reading.
REFUSAL_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("share one candidate", "span:line_starts_collapse_under_bpe"),
    ("lies in no token's offset span", "span:line_start_in_no_token"),
    ("are not line-start candidates", "span:gold_not_a_candidate"),
    ("both fall inside token", "span:multiline_span_in_one_token"),
    ("offsets are not monotonic", "span:offsets_not_monotonic"),
    ("span row whose context has no lines", "span:empty_context"),
    ("needs the tokenizer's character offsets", "span:no_offsets_supplied"),
    ("outside the context's", "span:gold_outside_context"),
    ("decodes to", "decode:token_text_disagrees_with_offsets"),
    ("are recorded as the start of a", "decode:candidate_not_a_line_start"),
    ("do not decode to the text", "decode:ids_do_not_round_trip"),
    ("describe a different string", "offsets:reach_wrong_length"),
    ("describe different tokenizations", "offsets:count_disagrees_with_ids"),
    ("the tokenizer returned", "tokenize:bad_return_value"),
    ("not among the rendered options", "gold:not_a_rendered_option"),
    ("has no gold answer", "gold:missing"),
    ("escaped context has", "render:escape_changed_the_line_count"),
    ("context region was not found", "render:region_not_where_delimiters_say"),
    ("at most", "caps:payload_over_render_cap"),
    ("all whitespace", "caps:empty_context"),
    ("were dropped by the remap", "remap:token_not_in_remap"),
)


def classify_refusal(exc: BaseException) -> str:
    text = str(exc)
    for fragment, name in REFUSAL_SIGNATURES:
        if fragment in text:
            return name
    return f"unclassified:{type(exc).__name__}"


@dataclass
class Census:
    """Per-row outcomes of the work ``write_shards`` does, in the order it does it."""

    rows_in: int = 0
    rows_out: int = 0
    sequences_out: int = 0
    span_rows_out: int = 0
    lengths: list[int] = None  # type: ignore[assignment]
    refused: collections.Counter[str] = None  # type: ignore[assignment]
    examples: dict[str, str] = None  # type: ignore[assignment]
    fatal_classes: set[str] = None  # type: ignore[assignment]
    #: Accepted line-start candidates whose token does NOT begin at that character. The
    #: writer takes "the token *containing* the line start", which it documents, so these
    #: are accepted by design -- but nothing had ever counted them, and a candidate token
    #: that starts mid-previous-line is what the pointer head is asked to select.
    candidates_total: int = 0
    candidates_not_at_token_start: int = 0
    #: Ids of every sequence that TOKENIZED, including those whose row was refused later.
    #: This is the set a remap must cover, and it is deliberately a superset of what gets
    #: written: `write_shards` calls `RemapTable.encode` *before* it decides whether the row
    #: is excludable, so a remap built over the written subset alone is refused by
    #: `TokenNotInRemap` for a token that only ever appears in a row the writer drops.
    #: Measured 2026-09-20; recorded as GAP-PIPELINE-REMAP-DEMANDED-BEFORE-EXCLUSION.
    ids: list[np.ndarray] = None  # type: ignore[assignment]
    #: The row each entry of ``ids`` came from, index for index. A row renders to one
    #: sequence per slot, and "can this row be encoded" is a question about all of them.
    id_rows: list[str] = None  # type: ignore[assignment]
    #: Every row that wrote NO sequence, by id, with its refusal class: refused before any
    #: slot was tokenized, or ``every_slot_refused``. ``refused`` counts the first kind.
    refused_rows: dict[str, str] = None  # type: ignore[assignment]
    #: Slot-scoped refusals: ``(row_id, slot_name)`` -> class, and their counts. Since the
    #: writer refuses only the slot whose span cannot be placed, a two-slot family whose
    #: span collapses under BPE keeps its class sequence; these are counted separately so a
    #: slot refusal is never read as a lost row (GAP-A3-BPE-SPAN-COLLAPSE-DROPS-...).
    refused_slots: dict[tuple[str, str], str] = None  # type: ignore[assignment]
    refused_slot: collections.Counter[str] = None  # type: ignore[assignment]
    slots_in: int = 0

    def __post_init__(self) -> None:
        self.lengths = []
        self.refused = collections.Counter()
        self.examples = {}
        self.fatal_classes = set()
        self.ids = []
        self.id_rows = []
        self.refused_rows = {}
        self.refused_slots = {}
        self.refused_slot = collections.Counter()


def census(rows: list[DataRow], *, tok: RealTokenizer, config: DataConfig) -> Census:
    """Mirror ``write_shards``' per-row work and record what every row did.

    The order is ``training_texts`` -> ``_tokenize_checked`` -> ``_span_token_positions``,
    which is ``write_shards``' order minus ``RemapTable.encode``: the remap is built *from*
    this pass's ids, so it cannot refuse one of them, and building it first would need the
    ids this pass produces. The one thing the mirror adds is that it catches every
    exception rather than only ``UnencodableGold``, which is how a class that kills the
    whole write is told apart from one that is excluded and counted.
    """
    out = Census()
    for row in rows:
        out.rows_in += 1
        out.slots_in += len(row.request.slots)
        staged_ids: list[np.ndarray] = []
        staged_spans = 0
        # The writer's two scopes (qd_train.shards.write_shards): what `training_texts`
        # raises refuses the row; an UnencodableGold from one slot's span projection
        # refuses that slot's sequence and keeps the others.
        try:
            specs = training_texts(row, seed=config.seed)
        except (UnencodableGold, ShardContractViolation, QdRefusal) as exc:
            name = classify_refusal(exc)
            out.refused[name] += 1
            out.refused_rows[row.row_id] = name
            out.examples.setdefault(name, f"{row.row_id}: {exc}")
            if not isinstance(exc, (UnencodableGold, QdRefusal)):
                # write_shards excludes UnencodableGold and QdRefusal at this point and
                # nothing else, so anything else aborts the entire write.
                out.fatal_classes.add(name)
            continue
        slot_refused: dict[str, str] = {}
        for spec in specs:
            where = f"row {row.row_id!r} slot {spec.slot_name!r}"
            try:
                ids = shards_module._tokenize_checked(tok.tokenize, spec.text, where=where)
                # Recorded here, before the span projection can refuse the slot, because
                # this is the point at which `write_shards` itself demands remap coverage.
                out.ids.append(ids)
                out.id_rows.append(row.row_id)
                projected = shards_module._span_token_positions(
                    spec, ids, token_offsets=tok.offsets, decode=tok.decode, where=where
                )
            except (UnencodableGold, ShardContractViolation) as exc:
                name = classify_refusal(exc)
                out.refused_slot[name] += 1
                slot_refused[spec.slot_name] = name
                out.examples.setdefault(name, f"{row.row_id} slot {spec.slot_name}: {exc}")
                if not isinstance(exc, UnencodableGold):
                    # Only UnencodableGold is slot-scoped in the writer; a contract fault
                    # from the tokenizer wiring aborts the whole write.
                    out.fatal_classes.add(name)
                continue
            if projected is not None and spec.line_char_starts:
                offs = tok.offsets(spec.text)
                for position, char in zip(projected[1], spec.line_char_starts, strict=True):
                    out.candidates_total += 1
                    out.candidates_not_at_token_start += 0 if offs[position][0] == char else 1
            staged_ids.append(ids)
            staged_spans += 1 if projected is not None else 0
        for slot_name, name in slot_refused.items():
            out.refused_slots[(row.row_id, slot_name)] = name
        if not staged_ids:
            # Every slot refused on its own account: the writer excludes the row too.
            out.refused_rows[row.row_id] = "every_slot_refused"
            continue
        out.rows_out += 1
        out.sequences_out += len(staged_ids)
        out.span_rows_out += staged_spans
        out.lengths.extend(int(i.size) for i in staged_ids)
    return out


def defect_balance(
    rows: list[DataRow],
    *,
    refused: dict[str, str] | None = None,
    refused_slots: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    """``code.defect_class``'s class balance over the rows whose CHOICE sequence is written.

    With ``refused`` (a census's ``refused_rows``) and ``refused_slots`` (its
    ``refused_slots``) this is the class balance that reaches the shard set: a row counts
    unless its choice slot wrote nothing, whether because the row was refused whole or the
    choice slot was refused on its own. A span-slot refusal leaves the class label in the
    set, so it does not move the balance; it is reported beside it, by class and reason,
    under ``slot_refused_by_class_and_reason``. A refusal that falls unevenly across classes
    moves the majority rate the trained head is judged against.
    """
    refused = refused or {}
    refused_slots = refused_slots or {}
    fam = [r for r in rows if r.family_id == DEFECT_FAMILY_ID]

    def label(r: DataRow) -> str:
        return str(next(g.value for g in r.gold if g.slot_name == DEFECT_CHOICE_SLOT))

    kept = [
        r for r in fam
        if r.row_id not in refused and (r.row_id, DEFECT_CHOICE_SLOT) not in refused_slots
    ]
    classes = collections.Counter(label(r) for r in kept)
    out: dict[str, Any] = {
        "family": DEFECT_FAMILY_ID,
        "rows": len(kept),
        "of": len(fam),
        "repos": len({r.repo_key for r in kept}),
        "classes": dict(sorted(classes.items())),
        "languages": dict(
            sorted(collections.Counter(r.metadata["language"] for r in kept).items())
        ),
    }
    if classes:
        top, k = classes.most_common(1)[0]
        out["majority"] = {"class": top, "n": k, "rate": round(k / len(kept), 4)}
    if refused:
        by = collections.Counter(
            f"{label(r)}:{refused[r.row_id]}" for r in fam if r.row_id in refused
        )
        out["refused_by_class_and_reason"] = dict(sorted(by.items()))
    fam_ids = {r.row_id: r for r in fam}
    slot_by = collections.Counter(
        f"{label(fam_ids[row_id])}:{slot_name}:{reason}"
        for (row_id, slot_name), reason in refused_slots.items()
        if row_id in fam_ids
    )
    if slot_by:
        out["slot_refused_by_class_and_reason"] = dict(sorted(slot_by.items()))
    return out


def shard_sequences_metric(
    *, n_sequences: int, rows: list[DataRow], total_tokens: int
) -> TriState:
    """Sequences written, against the sequences the split's rows render: one per slot.

    Not against rows. Until ``code.defect_class`` every family had one slot, so sequences
    and rows were one number and ``n_total=rows`` held by coincidence; a two-slot family
    writes ~2 sequences per row and the pair became ``78643 of 46167`` -- refused by
    ``Ran`` as n > n_total, after the whole shard set had been written.
    """
    offered = sum(len(r.request.slots) for r in rows)
    return Ran(
        passed=True, value=n_sequences, n=n_sequences, n_total=offered,
        detail=(
            f"total_tokens={total_tokens}; {n_sequences} of {offered} slot sequences "
            f"offered by {len(rows)} rows were written"
        ),
    )


def remap_coverage(
    cen: Census, remap: RemapTable, *, split_name: str, built_from: str
) -> TriState:
    """How many rows of a split the remap can encode at all, when it was not built from them.

    The remap keeps every token the TRAIN rows use and nothing else, and
    :meth:`RemapTable.encode` raises on any other id rather than substituting an ``UNK``
    (``qd_train.remap``'s coverage policy, deliberately). So a row of another split that
    holds one token the train rows never used cannot be written, scored or served under
    that remap -- not scored badly, not scored at all. The plan's S2 gate is a loss on
    held-out sequences under the remap, and row af64957d measured parity only on sequences
    the remap was built from, because nothing had counted how many held-out rows the remap
    can reach.

    Counted over ``cen``, the same per-row work ``write_shards`` does, so a row the writer
    would refuse before tokenizing is not charged to the remap. Both populations are stated:
    the rows that tokenized, and the rows that came in.
    """
    if not cen.ids:
        return NotRun(
            reason=(
                f"no row of the {split_name} split tokenized ({cen.rows_in} in), so there "
                "is nothing to check the remap against"
            )
        )
    tokenized: set[str] = set()
    short: set[str] = set()
    missing: collections.Counter[int] = collections.Counter()
    tokens = 0
    for row_id, ids in zip(cen.id_rows, cen.ids, strict=True):
        tokenized.add(row_id)
        tokens += int(ids.size)
        dropped = ids[remap.old_to_new[ids] < 0]
        if dropped.size:
            short.add(row_id)
            missing.update(int(i) for i in dropped)
    encodable = len(tokenized) - len(short)
    return Ran(
        passed=not short,
        value=encodable,
        n=encodable,
        n_total=len(tokenized),
        detail=(
            f"{encodable} of {len(tokenized)} tokenized {split_name} row(s) use only ids the "
            f"{built_from} remap kept ({cen.rows_in} row(s) in, {cen.rows_in - len(tokenized)} "
            f"never tokenized). {len(short)} hold at least one dropped id: "
            f"{sum(missing.values())} of {tokens} tokens, {len(missing)} distinct ids. "
            "RemapTable.encode raises on each of those rows and has no UNK, so under this "
            "remap they cannot be written, scored or served."
        ),
    )


def byte_fallback_cost(
    cen: Census,
    remap: RemapTable,
    *,
    byte_ids: frozenset[int],
    byte_lengths: Callable[[Iterable[int]], dict[int, int]],
    split_name: str,
) -> TriState:
    """What a lossless byte fallback would add to a split the remap cannot encode.

    GAP-REMAP-CANNOT-ENCODE-THE-ROWS-IT-WAS-NOT-BUILT-FROM leaves the choice of fix to a
    human, and this prices one option on the same rows: keep the 256 byte tokens and spell
    every dropped token in them. The count is an UPPER bound -- a fallback that re-segments
    into the longest kept pieces is never longer than one that spells single bytes -- and
    it says how many of the byte tokens the remap would have to add.
    """
    if not cen.ids:
        return NotRun(
            reason=f"no row of the {split_name} split tokenized ({cen.rows_in} in)"
        )
    tokens = 0
    dropped: collections.Counter[int] = collections.Counter()
    for ids in cen.ids:
        tokens += int(ids.size)
        dropped.update(int(i) for i in ids[remap.old_to_new[ids] < 0])
    lengths = byte_lengths(dropped)
    extra = sum((lengths[i] - 1) * n for i, n in dropped.items())
    unkept = sum(1 for i in byte_ids if remap.old_to_new[i] < 0)
    # `value` is the tokens a fallback adds; the pair is the tokens it re-spells, of all.
    return Ran(
        passed=True,
        value=extra,
        n=sum(dropped.values()),
        n_total=tokens,
        detail=(
            f"spelling the {sum(dropped.values())} dropped token(s) of the {split_name} "
            f"split in single bytes adds at most {extra} token(s) to its {tokens} "
            f"({extra / tokens:.2%}); {unkept} of the 256 byte tokens are not in this remap "
            "and a fallback would add them. An upper bound: re-segmenting into the longest "
            "kept pieces is never longer."
        ),
    )


# -- the run ----------------------------------------------------------------------------


def _default_decode_survives(
    rows: list[DataRow], *, tok: RealTokenizer, config: DataConfig
) -> TriState:
    """Would ``decode=tokenizer.decode`` -- the obvious call -- pass the writer's check?

    Run over the first span row that projects at all, so the answer is about the decode
    argument rather than about a row that was going to be refused anyway. ``NotRun`` when
    no such row exists: a corpus with no usable span row says nothing about decode.
    """
    for row in rows:
        try:
            specs = training_texts(row, seed=config.seed)
        except (UnencodableGold, QdRefusal):
            continue
        for spec in specs:
            if spec.line_char_starts is None:
                continue
            where = f"row {row.row_id!r} slot {spec.slot_name!r}"
            try:
                ids = shards_module._tokenize_checked(tok.tokenize, spec.text, where=where)
                shards_module._span_token_positions(
                    spec, ids, token_offsets=tok.offsets,
                    decode=tok.decode_default_kwargs, where=where,
                )
            except ShardContractViolation as exc:
                return Ran(
                    passed=False, value=row.row_id, n=0, n_total=1,
                    detail=f"refused with: {exc}"[:600],
                )
            except UnencodableGold:
                continue
            return Ran(
                passed=True, value=row.row_id, n=1, n_total=1,
                detail=(
                    "tokenizer.decode with no kwargs satisfies all three statements "
                    "_assert_spans_decode_to_their_text makes, so a caller need not know "
                    "to pass clean_up_tokenization_spaces=False"
                ),
            )
    return NotRun(
        reason=(
            "no span row in this split projected to token positions, so whether the "
            "default decode would have been accepted was never exercised"
        )
    )


def _write_manifests(
    manifests: dict[str, Any], out: Path, *, allow_not_run: bool
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, manifest in manifests.items():
        directory = out / "data" / (HELD_OUT if name == HELD_OUT else "pool")
        path = directory / f"{name}.json"
        manifest.write(path, allow_not_run=allow_not_run)
        paths[name] = path
    return paths


def _read_back(reader: ShardReader, *, tok: RealTokenizer, remap: RemapTable) -> dict[str, Any]:
    """Read every sequence back and re-derive what the shard claims about it.

    The token round-trip goes back through the remap's ``new_to_old`` rather than through
    the ids the writer held, so it checks the bytes on disk rather than a variable.
    """
    findings: dict[str, Any] = {
        "sequences": len(reader),
        "batches": 0,
        "rows_in_batches": 0,
        "span_rows": 0,
        "gold_off_candidate": 0,
        "candidate_in_padding": 0,
        "target_not_before_last": 0,
        "tail_not_one_letter": 0,
        "decode_mismatch": 0,
    }
    for index in range(len(reader)):
        new_ids = reader.sequence(index)
        old_ids = remap.new_to_old[new_ids]
        text = tok.decode(old_ids)
        if not text:
            findings["decode_mismatch"] += 1
        # The convention training_texts documents: target_index is the token before the
        # last, and the last token is the answer letter alone. Checked against the real
        # tokenizer, because in byte space it cannot fail.
        if int(reader._target_index[index]) != int(new_ids.size) - 2:
            findings["target_not_before_last"] += 1
        tail = tok.decode(old_ids[-1:])
        if len(tail) != 1 or not tail.isalpha():
            findings["tail_not_one_letter"] += 1

    for batch in reader.batches(batch_tokens=1 << 16, seed=7, epoch=0):
        findings["batches"] += 1
        findings["rows_in_batches"] += int(batch.tokens.shape[0])
        if batch.span_target is None:
            continue
        for r in range(batch.tokens.shape[0]):
            if int(batch.slot_kind[r]) != SLOT_SPAN:
                continue
            findings["span_rows"] += 1
            mask = batch.line_starts[r]
            real = int(batch.lengths[r])
            if bool(mask[real:].any()):
                findings["candidate_in_padding"] += 1
            start, end = int(batch.span_target[r, 0]), int(batch.span_target[r, 1])
            if start >= 0 and (not mask[start] or not mask[end]):
                findings["gold_off_candidate"] += 1
    return findings


def _artifact_digest(out: Path) -> dict[str, str]:
    """What a later reader could tell about this shard set from its files."""
    header = json.loads((out / HEADER_NAME).read_text(encoding="utf-8"))
    coverage = json.loads((out / COVERAGE_NAME).read_text(encoding="utf-8"))
    header.pop("created_at", None)
    digest = {
        "header_without_created_at": json.dumps(header, sort_keys=True),
        "coverage": json.dumps(coverage, sort_keys=True),
    }
    span_check = out / SPAN_CHECK_NAME
    # Absent on a shard set written before this file existed, which is itself the answer
    # rather than a reason to skip the comparison.
    digest["span_check"] = (
        span_check.read_text(encoding="utf-8") if span_check.exists() else "<absent>"
    )
    return digest


@dataclass
class Measured:
    """What the run measured, in the shape a ledger row carries it.

    Every entry is a ``TriState``, so a stage that could not run is recorded as ``NotRun``
    and never as a zero that reads like a pass.
    """

    metrics: dict[str, TriState]
    gates: dict[str, TriState]
    data_snapshot_hash: str
    tokenizer_hash: str
    notes: str
    #: From :func:`quick_reason_for`, because only the run knows which corpus it read.
    quick_reason: str


# -- the general families, from the approved download ------------------------------------

#: A fetch record is a short JSON list; a file this large is not one.
MAX_FETCH_RECORD_BYTES = 1 << 20
#: Per-line bound for the general caches. SQuAD's longest train context is ~25 KB; a line
#: forty times that is not a row.
GENERAL_MAX_ROW_BYTES = 1 << 20
#: Per-file row bound. SQuAD v2 train, the largest file, is 130,319 rows.
DEFAULT_GENERAL_MAX_ROWS = 200_000
#: The bound this tool passes to build_mixture's consistency pass, stated rather than
#: inherited. qd_data.mixture.DEFAULT_MAX_CONSISTENCY_ROWS (250,000) is a rendering-
#: cost bound for callers that render nothing else; this tool renders every row again
#: in stage 3, so the cost argument does not apply here. At the default, the
#: 2026-09-29 corpus (315,239 rows with the general families) was NotRun, and 42
#: contradictory sequences reached write_shards unremoved. Recorded in the recipe:
#: it decides which rows exist.
PIPELINE_MAX_CONSISTENCY_ROWS = 1_000_000
#: The datasets a fetch record may name, and the source each becomes.
GENERAL_DATASETS = frozenset(
    {"cais/mmlu", "tau/commonsense_qa", "clinc/clinc_oos", "rajpurkar/squad_v2"}
)


@dataclass(frozen=True)
class GeneralLoad:
    """The general-family rows read from a fetch record's caches, every file sha-checked."""

    raw: dict[str, list[RawRow]]
    clinc_domain_map: ClincDomainMap | None
    #: ``{jsonl path: {"dataset", "rows", "sha256"}}`` for the ledger recipe.
    files: dict[str, dict[str, Any]]
    capped: tuple[str, ...]
    record_sha256: str
    #: ``{jsonl path: reason}`` for record files whose split
    #: ``qd_data.loaders.REFUSED_READS`` refuses to read.
    refused_reads: dict[str, str]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def general_rows(record: Path, *, max_rows_per_file: int = DEFAULT_GENERAL_MAX_ROWS) -> GeneralLoad:
    """Read every cache a fetch record names, refusing anything the record does not vouch for.

    Each ``jsonl`` must sit under the record's own directory (the cache root), hash to the
    record's ``jsonl_sha256`` and hold the record's ``rows``; CLINC's ``intent_names.json``
    must hold ``n_intent_names`` names; the CLINC domain map is read through
    ``qd_data.general.load_clinc_domains``, which checks its own pinned sha256. Any mismatch
    stops the run: a corpus the record does not describe is not the approved download.
    """
    raw_bytes = record.read_bytes()
    if len(raw_bytes) > MAX_FETCH_RECORD_BYTES:
        raise SystemExit(f"{record}: {len(raw_bytes)} bytes is not a fetch record")
    entries = json.loads(raw_bytes)
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{record}: a fetch record is a non-empty JSON list")
    root = record.parent.resolve()
    out: dict[str, list[RawRow]] = {}
    files: dict[str, dict[str, Any]] = {}
    capped: list[str] = []
    refused_reads: dict[str, str] = {}
    for entry in entries:
        dataset = str(entry.get("dataset"))
        if dataset not in GENERAL_DATASETS:
            raise SystemExit(
                f"{record}: dataset {dataset!r} is not one of {sorted(GENERAL_DATASETS)}"
            )
        path = Path(str(entry["jsonl"])).resolve()
        if not path.is_relative_to(root):
            raise SystemExit(f"{record}: {path} is outside the cache root {root}")
        found = _sha256_file(path)
        if found != entry["jsonl_sha256"]:
            raise SystemExit(
                f"{path}: sha256 {found} but the fetch record says {entry['jsonl_sha256']}; "
                "the cache is not the approved download"
            )
        rows, hit = read_jsonl(path, limit=max_rows_per_file, max_row_bytes=GENERAL_MAX_ROW_BYTES)
        if hit:
            capped.append(dataset)
        elif len(rows) != int(entry["rows"]):
            raise SystemExit(f"{path}: {len(rows)} rows but the fetch record says {entry['rows']}")
        split_name = path.stem
        why = REFUSED_READS.get((dataset, split_name))
        if why is not None:
            # Fetched, sha-checked and not read: the licence covers the source but not
            # this split (e.g. CommonsenseQA's unlabelled test). Counted, never parsed.
            refused_reads[str(path)] = why
            continue
        parsed: list[RawRow]
        if dataset == "cais/mmlu":
            parsed = [parse_mmlu(r, index=i, split_name=split_name) for i, r in enumerate(rows)]
        elif dataset == "tau/commonsense_qa":
            parsed = [parse_csqa(r, index=i, split_name=split_name) for i, r in enumerate(rows)]
        elif dataset == "clinc/clinc_oos":
            names = json.loads((path.parent / "intent_names.json").read_text(encoding="utf-8"))
            if len(names) != int(entry["n_intent_names"]):
                raise SystemExit(
                    f"{path.parent}/intent_names.json holds {len(names)} names but the fetch "
                    f"record says {entry['n_intent_names']}"
                )
            parsed = [parse_clinc(r, index=i, label_names=names) for i, r in enumerate(rows)]
        else:
            parsed = [parse_squad(r, index=i) for i, r in enumerate(rows)]
        out.setdefault(dataset, []).extend(parsed)
        files[str(path)] = {"dataset": dataset, "rows": len(parsed), "sha256": found}
    domain_map: ClincDomainMap | None = None
    if "clinc/clinc_oos" in out:
        domain_map = load_clinc_domains(
            root / "clinc__oos-eval" / CLINC_DOMAINS_COMMIT / "domains.json"
        )
    return GeneralLoad(
        raw=out, clinc_domain_map=domain_map, files=files,
        capped=tuple(sorted(set(capped))),
        record_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        refused_reads=refused_reads,
    )


def split_off_replay(
    split_report: SplitReport, *, seed: int
) -> tuple[SplitReport, SplitReport, ReplayPartition]:
    """``(gold_report, replay_report, partition)``: the replay slice as its own split report.

    ``qd_data.general.partition_replay`` marks a deterministic share of the replay
    families' TRAIN rows replay-only. Those leave the gold train split -- a row both
    gold-trained and replay-trained is the case the partition exists to prevent -- and
    become a train-only report of their own, so each gets its own manifest and its own
    shard set (``write_shards(replay=True)``). val and held-out are untouched.
    """
    train = list(split_report.rows_by_split.get("train", ()))
    general_train = [r for r in train if r.family_id in REPLAY_FAMILIES]
    part = partition_replay(general_train, seed=seed)
    drawn = {r.row_id for r in general_train}
    gold_train = tuple(r for r in train if r.row_id not in drawn) + part.gold_rows
    gold = dataclasses.replace(
        split_report, rows_by_split={**split_report.rows_by_split, "train": gold_train}
    )
    replay = dataclasses.replace(
        split_report,
        rows_by_split={"train": part.replay_rows, "val": (), HELD_OUT: ()},
    )
    return gold, replay, part


def run(
    *,
    out: Path,
    max_pairs: int,
    blank_line_runs: bool,
    rev: str,
    commitpackft: Path | None = None,
    val_shards: bool = False,
    defect_class: Path | None = None,
    defect_download: Path = DEFAULT_DEFECT_DOWNLOAD,
    defect_max_rows: int | None = None,
    memo_limit: int = MEMO_LIMIT,
    general_record: Path | None = None,
    general_max_rows: int = DEFAULT_GENERAL_MAX_ROWS,
    replay_shards: bool = False,
    repo_history: bool = True,
) -> Measured:
    if replay_shards and general_record is None:
        raise SystemExit(
            "--replay-shards needs --general-record: the replay slice is drawn from the "
            "general families' training rows"
        )
    if not repo_history and commitpackft is None and defect_class is None and (
        general_record is None
    ):
        raise SystemExit(
            "--no-repo-history with no --commitpackft, --defect-class or --general-record "
            "reads no source at all; there would be nothing to build"
        )
    config = DataConfig()
    extra_metrics: dict[str, TriState] = {}
    resolved = resolve_rev(REPO, rev)
    tok = RealTokenizer.load(memo_limit=memo_limit)
    print(
        f"tokenizer {type(tok.tok).__name__} for {MODEL}: vocab_size={tok.tok.vocab_size} "
        f"len={len(tok.tok)} fast={tok.tok.is_fast}"
    )
    print(f"corpus revision: {rev} -> {resolved}")

    base = base_sources(
        repo_history=repo_history, commitpackft=commitpackft, max_pairs=max_pairs,
        blank_line_runs=blank_line_runs, rev=resolved,
    )
    commits_n, spans_n, pool_total = base.n_commits, base.n_spans, base.pool_total
    if not repo_history:
        code_source = (
            "no row from this repository's history (--no-repo-history)"
            + (
                f"; bigcode/commitpackft from {commitpackft} ({commits_n} of {pool_total} "
                "rows, a sha256-ordered sample; files pinned by the download's manifest)"
                if pool_total is not None else ""
            )
        )
        quick_reason = quick_reason_for(
            commitpackft_rows=None if pool_total is None else (commits_n, pool_total),
            repo_history=False,
        )
    elif pool_total is None:
        code_source = f"this repository's own history at {resolved}"
        quick_reason = quick_reason_for(commitpackft_rows=None)
    else:
        code_source = (
            f"bigcode/commitpackft from {commitpackft} ({commits_n} of {pool_total} "
            "rows, a sha256-ordered sample; files pinned by the download's manifest), "
            f"with span prose from this repository at {resolved}"
        )
        quick_reason = quick_reason_for(commitpackft_rows=(commits_n, pool_total))
    raw: dict[str, list[RawRow | DefectRow]] = dict(base.raw)
    capped = list(base.capped)
    if defect_class is not None:
        # The corpus, its pool and the licence download are each checked against the
        # sha256 their manifests record, and any row that does not resolve through the
        # pool to a licence refuses the whole load -- see qd_data.defect_class.
        load = load_defect_rows(
            defect_class, download_root=defect_download, config=config, repo_root=REPO,
            max_rows=defect_max_rows,
        )
        raw[DEFECT_SOURCE_ID] = list(load.rows)
        if load.capped:
            capped.append(DEFECT_SOURCE_ID)
        code_source += (
            f"; {DEFECT_FAMILY_ID} from {defect_class} ({len(load.rows)} of "
            f"{load.n_corpus} qd-mutate examples"
            + (", a sha256-ordered sample" if load.capped else "")
            + f"; classes {load.by_class}; span-rebase refusals {load.span_refusals or 'none'})"
        )
        print(
            f"\n{DEFECT_FAMILY_ID}: {len(load.rows)} of {load.n_corpus} rows, "
            f"capped={load.capped}, by class {load.by_class}, span-rebase refusals "
            f"{load.span_refusals or 'none'}"
        )
    general: GeneralLoad | None = None
    if general_record is not None:
        general = general_rows(general_record, max_rows_per_file=general_max_rows)
        for dataset, rows in general.raw.items():
            # Real SQuAD replaces the repository-prose stand-in under the same source.
            raw[dataset] = list(rows)
        capped = [c for c in capped if c not in general.raw] + list(general.capped)
        code_source += (
            f"; general families from {general_record} (sha256 "
            f"{general.record_sha256[:16]}; "
            + ", ".join(f"{d} {len(r)}" for d, r in sorted(general.raw.items()))
            + "; real SQuAD replaces the prose stand-in)"
        )
        print(
            f"\ngeneral families: {({d: len(r) for d, r in sorted(general.raw.items())})}, "
            f"capped={list(general.capped) or 'none'}, clinc domain map="
            f"{general.clinc_domain_map is not None}"
        )
    print(
        f"\ncorpus: {commits_n} real (commit, path) pairs, {spans_n} real prose "
        f"passages; blank_line_runs={blank_line_runs}; capped={capped or 'none'}"
    )

    mixture = build_mixture(
        raw, config=config, capped_sources=capped,
        clinc_domain_map=general.clinc_domain_map if general is not None else None,
        max_consistency_rows=PIPELINE_MAX_CONSISTENCY_ROWS,
    )
    if isinstance(mixture.prompt_consistency, NotRun):
        # Fail fast: a capped read is accepted below as a deliberate NotRun, but an
        # unchecked corpus reaches write_shards' own contradiction refusal twenty
        # minutes later, having removed nothing.
        raise SystemExit(
            "build_mixture could not run its consistency pass, so contradictory "
            f"prompts were not removed: {mixture.prompt_consistency.reason}"
        )
    extra_metrics["prompt_consistency"] = mixture.prompt_consistency
    n_attempted = sum(
        len(v) for v in raw.values()
    )
    print("\n== stage 1: build_mixture ==")
    print(f"  raw rows in: {n_attempted}   DataRows out: {len(mixture.rows)}")
    print(f"  status: {json.dumps(mixture.status.to_json(), sort_keys=True)[:600]}")
    print(
        "  prompt consistency: "
        f"{json.dumps(mixture.prompt_consistency.to_json(), sort_keys=True)[:800]}"
    )
    for source_id, counts in sorted(mixture.refusals.items()):
        for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    refused {source_id} {code}: {n}")

    report = dedupe(list(mixture.rows), config=config)
    split_report = split(report, config=config)
    print("\n== stage 2: dedupe + split ==")
    print(f"  rows in: {len(mixture.rows)}   kept: {len(report.kept)}")
    print(f"  split counts: {split_report.counts()}")
    if defect_class is not None:
        # The number the rung-3 collapse (46/90, the val majority) is read against: a
        # choice head that learned nothing scores exactly this on each split.
        for split_name, split_rows in sorted(split_report.rows_by_split.items()):
            print(f"  {split_name}: {json.dumps(defect_balance(list(split_rows)))}")
    print(f"  split status: {json.dumps(split_report.status.to_json(), sort_keys=True)[:400]}")

    replay_report: SplitReport | None = None
    if replay_shards:
        split_report, replay_report, partition = split_off_replay(
            split_report, seed=config.seed
        )
        drawn = sum(sum(c.values()) for c in partition.counts().values())
        extra_metrics["replay_partition"] = Ran(
            passed=True, value=len(partition.replay_rows), n=len(partition.replay_rows),
            n_total=drawn,
            detail=(
                f"replay-only rows drawn from the replay families' train rows at "
                f"fraction {partition.fraction}: {partition.counts()}. Not yet "
                "decontaminated: tools/replay_decontam.py attests the replay set."
            ),
        )
        print(f"  replay partition: {partition.counts()}")
    manifests = build_manifests(
        config=config, mixture=mixture, dedupe_report=report, split_report=split_report
    )
    snapshot_status = manifests["train"].status
    not_run_snapshot = snapshot_status.to_json()["state"] == "not_run"
    if not_run_snapshot:
        # Stated, never absorbed. The manifest door and `open_training_data` both refuse a
        # not-run snapshot by default, and this run overrides both -- so the reason it
        # overrode them is printed rather than left in a file nobody reads.
        print(
            "\n  NOT-RUN SNAPSHOT, accepted deliberately for this measurement:\n"
            f"    {snapshot_status.to_json()['reason']}"
        )
    paths = _write_manifests(manifests, out, allow_not_run=not_run_snapshot)
    replay_rows: list[DataRow] = []
    if replay_report is not None:
        replay_rows = list(replay_report.rows_by_split["train"])
        paths["replay"] = out / "data" / "pool" / "train-replay.json"
        build_manifests(
            config=config, mixture=mixture, dedupe_report=report,
            split_report=replay_report,
        )["train"].write(paths["replay"], allow_not_run=not_run_snapshot)
        print(f"  replay manifest: {paths['replay']} ({len(replay_rows)} entries)")
    train_rows = list(split_report.rows_by_split.get("train", ()))
    print(f"  train manifest: {paths['train']} ({len(train_rows)} entries)")

    print("\n== stage 3: census (unmodified write_shards work, per row) ==")
    cen = census(train_rows, tok=tok, config=config)
    print(f"  rows in: {cen.rows_in}   rows that encoded: {cen.rows_out}   "
          f"sequences: {cen.sequences_out}   span sequences: {cen.span_rows_out}")
    for name, n in sorted(cen.refused.items(), key=lambda kv: (-kv[1], kv[0])):
        fatal = " FATAL(aborts the write)" if name in cen.fatal_classes else ""
        print(f"    {name}: {n}{fatal}")
        print(f"      e.g. {cen.examples[name][:300]}")
    if not cen.refused:
        print("    (no row was refused)")
    for name, n in sorted(cen.refused_slot.items(), key=lambda kv: (-kv[1], kv[0])):
        fatal = " FATAL(aborts the write)" if name in cen.fatal_classes else ""
        print(f"    slot-scoped {name}: {n} of {cen.slots_in} slot(s){fatal}")
        print(f"      e.g. {cen.examples[name][:300]}")
    if defect_class is not None:
        print(
            "  train, as written: "
            + json.dumps(defect_balance(
                train_rows, refused=cen.refused_rows, refused_slots=cen.refused_slots
            ))
        )
    print(
        f"  accepted line-start candidates: {cen.candidates_total}, of which "
        f"{cen.candidates_not_at_token_start} sit inside a token that begins earlier "
        "(accepted by design: the writer takes the token containing the line start)"
    )

    if not cen.lengths:
        raise SystemExit(
            f"no row of {cen.rows_in} survived the census, so there is nothing to write. "
            f"Refusals: {dict(cen.refused)}"
        )

    print("\n== stage 3b: does the obvious decode= argument work? ==")
    default_decode = _default_decode_survives(train_rows, tok=tok, config=config)
    print(f"  tokenizer.decode with no kwargs: {json.dumps(default_decode.to_json())[:400]}")

    print("\n== stage 4: buckets and padding, under the real tokenizer ==")
    buckets = choose_buckets(cen.lengths)
    waste = padding_waste(cen.lengths, buckets)
    per_bucket = collections.Counter(assign_buckets(cen.lengths, buckets))
    print(f"  lengths: n={len(cen.lengths)} min={min(cen.lengths)} "
          f"max={max(cen.lengths)} mean={sum(cen.lengths) / len(cen.lengths):.1f}")
    ordered = sorted(cen.lengths)
    print("  quantiles: " + " ".join(
        f"p{q}={ordered[min(len(ordered) - 1, (q * len(ordered)) // 100)]}"
        for q in (10, 50, 90, 99)
    ))
    print(f"  choose_buckets -> {list(buckets)}")
    print(f"  occupancy: {dict(sorted(per_bucket.items()))}")
    print(f"  padding_waste: {waste.to_json()}")

    val_rows = list(split_report.rows_by_split.get("val", ()))
    val_census = census(val_rows, tok=tok, config=config)
    if defect_class is not None:
        print(
            "  val, as written: "
            + json.dumps(defect_balance(
                val_rows, refused=val_census.refused_rows,
                refused_slots=val_census.refused_slots,
            ))
        )
    # With --val-shards the val rows are part of the written corpus, and the remap policy is
    # "keep every token the written corpus uses" -- so it is counted over them too. val is
    # a TRAINING_SPLITS member; the held-out rows never enter the count (rule 3).
    remap_ids = cen.ids + val_census.ids if val_shards else cen.ids
    replay_census = census(replay_rows, tok=tok, config=config) if replay_rows else None
    if replay_census is not None:
        # The replay set is written under the same remap, so its tokens are kept too.
        remap_ids = remap_ids + replay_census.ids
    counts = count_corpus_tokens(
        remap_ids, source_vocab_size=len(tok.tok), max_sequences=len(remap_ids) + 1
    )
    remap = build_remap(
        counts=counts,
        source_vocab_size=len(tok.tok),
        tokenizer_hash=tok.hash(),
        special_ids=tuple(sorted({int(i) for i in tok.tok.all_special_ids})),
        target_vocab_size=None,
    )
    print("\n== stage 5: remap over the real vocabulary ==")
    print(f"  source vocab {remap.source_vocab_size} -> kept {remap.vocab_size} "
          f"({counts.n_distinct} distinct ids used by {counts.n_tokens} tokens)")
    print(f"  counted over {len(remap_ids)} tokenized sequence(s) -- "
          f"{'train and val' if val_shards else 'train only'} -- of which "
          f"{len(cen.lengths)} train sequences reach the shard set; the difference is rows "
          "the writer excludes but still demands remap coverage for")

    # The held-out rows are tokenized here and counted, never written: nothing below stage
    # 5b sees them, so no training artifact depends on held-out text (rule 3).
    print("\n== stage 5b: the rows the remap was not built from ==")
    unseen: dict[str, TriState] = {}
    fallback: dict[str, TriState] = {}
    byte_ids = tok.byte_token_ids()
    for split_name in ("val", HELD_OUT):
        if split_name == "val" and val_shards:
            by_construction = NotRun(
                reason=(
                    "--val-shards built the remap over the val rows, so every one of them "
                    "encodes by construction and a count here would measure nothing"
                )
            )
            unseen[split_name] = fallback[split_name] = by_construction
            print(f"  val: {by_construction.reason}")
            continue
        split_census = (
            val_census
            if split_name == "val"
            else census(
                list(split_report.rows_by_split.get(split_name, ())), tok=tok, config=config
            )
        )
        unseen[split_name] = remap_coverage(
            split_census, remap, split_name=split_name,
            built_from="train and val" if val_shards else "train",
        )
        fallback[split_name] = byte_fallback_cost(
            split_census, remap, byte_ids=byte_ids, byte_lengths=tok.byte_lengths,
            split_name=split_name,
        )
        print(f"  {split_name}: {json.dumps(unseen[split_name].to_json())[:600]}")
        print(f"  {split_name} byte fallback: {json.dumps(fallback[split_name].to_json())[:600]}")

    print("\n== stage 6: write_shards with tokenize + token_offsets + decode ==")
    shard_dir = out / "shards" / "train"
    header = write_shards(
        paths["train"],
        train_rows,
        out_dir=shard_dir,
        remap=remap,
        tokenize=tok.tokenize,
        token_offsets=tok.offsets,
        decode=tok.decode,
        config=config,
        repo_root=out,
        allow_unencodable=True,
        allow_not_run_snapshot=not_run_snapshot,
        # The revision the corpus above was read at. Nothing else in the header covers it:
        # data_snapshot_hash hashes the rows that came out and code_fingerprint hashes the
        # code that made them, so a set built from the wrong rev is self-consistent in both.
        # `resolved`, never `rev`: --rev defaults to "HEAD", and "HEAD" in a header compares
        # equal to "HEAD" tomorrow, so it would read as verified while naming no commit.
        corpus_rev=resolved,
    )
    print(f"  header: n_sequences={header.n_sequences} total_tokens={header.total_tokens} "
          f"max_seq_len={header.max_seq_len} vocab_size={header.vocab_size}")
    print(f"  buckets: {list(header.buckets)}")
    coverage = json.loads((shard_dir / COVERAGE_NAME).read_text(encoding="utf-8"))
    print(f"  coverage.json: {json.dumps(coverage, sort_keys=True)[:400]}")
    if header.n_sequences != cen.sequences_out:
        print(
            f"  MISMATCH: the census counted {cen.sequences_out} sequences and the writer "
            f"wrote {header.n_sequences}. One of them is wrong."
        )

    val_coverage: TriState = NotRun(
        reason="--val-shards was not passed, so no val shard set was written"
    )
    val_slot_coverage: TriState = NotRun(
        reason="--val-shards was not passed, so no val shard set was written"
    )
    if val_shards:
        # Through the same door and the same writer as train: val is a TRAINING_SPLITS
        # member, so open_training_data admits its manifest, and scoring a model on it
        # reads it without a gradient. Its own buckets, from its own lengths.
        print("\n== stage 6b: the val split, under the same remap ==")
        val_dir = out / "shards" / "val"
        val_header = write_shards(
            paths["val"],
            val_rows,
            out_dir=val_dir,
            remap=remap,
            tokenize=tok.tokenize,
            token_offsets=tok.offsets,
            decode=tok.decode,
            config=config,
            repo_root=out,
            allow_unencodable=True,
            allow_not_run_snapshot=not_run_snapshot,
            corpus_rev=resolved,
        )
        val_reader = ShardReader(val_dir, config=config, repo_root=out)
        val_coverage = val_reader.coverage
        val_slot_coverage = val_reader.slot_coverage
        print(f"  header: n_sequences={val_header.n_sequences} "
              f"total_tokens={val_header.total_tokens} vocab_size={val_header.vocab_size}")
        print(f"  coverage: {json.dumps(val_coverage.to_json())[:400]}")

    if replay_rows:
        print("\n== stage 6c: the replay slice, as its own shard set ==")
        replay_dir = out / "shards" / "replay"
        write_shards(
            paths["replay"], replay_rows, out_dir=replay_dir, remap=remap,
            tokenize=tok.tokenize, token_offsets=tok.offsets, decode=tok.decode,
            config=config, repo_root=out, allow_unencodable=True,
            allow_not_run_snapshot=not_run_snapshot, corpus_rev=resolved, replay=True,
        )
        replay_reader = ShardReader(replay_dir, config=config, repo_root=out)
        extra_metrics["replay_shard_slots_written"] = replay_reader.slot_coverage
        extra_metrics["replay_shard_padding_waste"] = replay_reader.padding_waste()
        print(f"  replay: n_sequences={len(replay_reader)} "
              f"{json.dumps(replay_reader.slot_coverage.to_json())[:300]}")

    print("\n== stage 7: read back through ShardReader/Batch ==")
    reader = ShardReader(shard_dir, config=config, repo_root=out)
    print(f"  checks: {sorted(reader.checks)}")
    print(f"  coverage: {reader.coverage.to_json()}")
    print(f"  padding_waste: {reader.padding_waste().to_json()}")
    findings = _read_back(reader, tok=tok, remap=remap)
    for k, v in findings.items():
        print(f"    {k}: {v}")

    print("\n== stage 8: the same corpus written WITHOUT decode= ==")
    undecoded_dir = out / "shards" / "train-no-decode"
    write_shards(
        paths["train"],
        train_rows,
        out_dir=undecoded_dir,
        remap=remap,
        tokenize=tok.tokenize,
        token_offsets=tok.offsets,
        config=config,
        repo_root=out,
        allow_unencodable=True,
        allow_not_run_snapshot=not_run_snapshot,
        # The revision the corpus above was read at. Nothing else in the header covers it:
        # data_snapshot_hash hashes the rows that came out and code_fingerprint hashes the
        # code that made them, so a set built from the wrong rev is self-consistent in both.
        # `resolved`, never `rev`: --rev defaults to "HEAD", and "HEAD" in a header compares
        # equal to "HEAD" tomorrow, so it would read as verified while naming no commit.
        corpus_rev=resolved,
    )
    checked = _artifact_digest(shard_dir)
    unchecked = _artifact_digest(undecoded_dir)
    same = sorted(k for k in checked if checked[k] == unchecked[k])
    differ = sorted(k for k in checked if checked[k] != unchecked[k])
    print(f"  artifacts identical between decode-checked and unchecked writes: {same}")
    print(f"  artifacts that differ: {differ}")
    r2 = ShardReader(undecoded_dir, config=config, repo_root=out)
    to_json_identical = _strip_created(reader.to_json()) == _strip_created(r2.to_json())
    print(f"  ShardReader.to_json() identical: {to_json_identical}")
    print(f"  span_check, decode supplied : {json.dumps(reader.span_check.to_json())[:220]}")
    print(f"  span_check, decode omitted  : {json.dumps(r2.span_check.to_json())[:220]}")

    spans_offered = sum(
        1 for r in train_rows for slot in r.request.slots if isinstance(slot, SpanSlot)
    )
    clean = [k for k, v in findings.items() if k.endswith(
        ("off_candidate", "in_padding", "before_last", "one_letter", "mismatch")
    )]
    metrics: dict[str, TriState] = {
        "mixture_rows_built": Ran(
            passed=True, value=len(mixture.rows), n=len(mixture.rows),
            n_total=len(mixture.rows) + sum(sum(c.values()) for c in mixture.refusals.values()),
            detail="DataRows built against rows attempted, over every family",
        ),
        "shard_rows_written": reader.coverage,
        "shard_slots_written": reader.slot_coverage,
        "val_shard_slots_written": val_slot_coverage,
        "shard_sequences": shard_sequences_metric(
            n_sequences=header.n_sequences, rows=train_rows,
            total_tokens=header.total_tokens,
        ),
        "shard_max_seq_len": Ran(passed=True, value=header.max_seq_len,
                                 detail=f"buckets={list(header.buckets)}"),
        "span_sequences_written": Ran(
            passed=cen.span_rows_out == spans_offered,
            value=cen.span_rows_out, n=cen.span_rows_out, n_total=spans_offered,
            detail="span rows that reached the shard set against span rows in the split",
        ),
        "span_mapping_decode_verified": reader.span_check,
        "default_decode_kwargs_accepted": default_decode,
        "candidates_starting_mid_token": Ran(
            passed=True,
            value=cen.candidates_not_at_token_start,
            n=cen.candidates_not_at_token_start,
            n_total=cen.candidates_total,
            detail=(
                "accepted by design -- _token_index_for_char takes the token CONTAINING "
                "the line start -- and counted here because nothing had counted it"
            ),
        ),
        "read_back_span_invariants": Ran(
            passed=all(findings[k] == 0 for k in clean),
            value=sum(findings[k] for k in clean),
            n=findings["sequences"], n_total=findings["sequences"],
            detail=f"violations over the whole set: {({k: findings[k] for k in clean})}",
        ),
        "remap_vocabulary": Ran(
            passed=True, value=remap.vocab_size, n=remap.vocab_size,
            n_total=remap.source_vocab_size,
            detail=(
                f"counted over {len(remap_ids)} tokenized sequence(s), "
                f"{'train and val' if val_shards else 'train only'}"
            ),
        ),
        "remap_covers_val_rows": unseen["val"],
        "remap_covers_heldout_rows": unseen[HELD_OUT],
        "remap_byte_fallback_val_tokens": fallback["val"],
        "remap_byte_fallback_heldout_tokens": fallback[HELD_OUT],
        "val_shard_coverage": val_coverage,
        "decode_check_leaves_a_trace": Ran(
            passed=not to_json_identical or cen.span_rows_out == 0,
            value=str(not to_json_identical),
            detail=(
                "ShardReader.to_json() for a set written with decode= against one written "
                f"without; identical={to_json_identical}, span sequences={cen.span_rows_out}"
            ),
        ),
    }
    for name, n in sorted(cen.refused.items()):
        metrics[f"refused:{name}"] = Ran(
            passed=False, value=n, n=n, n_total=cen.rows_in,
            detail=cen.examples[name][:400],
        )
    metrics.update(extra_metrics)
    for name, n in sorted(cen.refused_slot.items()):
        # Slot-scoped: this slot's sequence was refused and the row's other slots written.
        metrics[f"refused_slot:{name}"] = Ran(
            passed=False, value=n, n=n, n_total=cen.slots_in,
            detail=cen.examples[name][:400],
        )
    return Measured(
        metrics=metrics,
        # The S4 gate, reported and never touched: MAX_PADDING_WASTE is read-only.
        gates={"padding_waste": reader.padding_waste()},
        data_snapshot_hash=header.data_snapshot_hash,
        tokenizer_hash=remap.tokenizer_hash,
        notes=(
            f"real-tokenizer end-to-end over {code_source}; "
            f"blank_line_runs={blank_line_runs}; {commits_n} commit pairs, "
            f"{spans_n} prose passages; snapshot status="
            f"{snapshot_status.to_json()['state']}"
        ),
        quick_reason=quick_reason,
    )


def _strip_created(blob: dict[str, Any]) -> str:
    body = json.loads(json.dumps(blob))
    body["header"].pop("created_at", None)
    body["root"] = ""
    return json.dumps(body, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="directory for artifacts")
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help=(
            f"default {DEFAULT_MAX_PAIRS}. The bound on the repository-history code and span "
            "rows, or the sample size of --commitpackft. Refused under --no-repo-history "
            "without --commitpackft, where it bounds nothing"
        ),
    )
    parser.add_argument(
        "--no-repo-history",
        dest="repo_history",
        action="store_false",
        help=(
            "read no row from this repository's git history: no span prose standing in for "
            "rajpurkar/squad_v2, and code rows only from --commitpackft when it is given. "
            "For a set built from --defect-class (and/or --commitpackft, --general-record) "
            "alone -- the campaign's phase 3 is code.defect_class only"
        ),
    )
    parser.add_argument(
        "--rev",
        default="HEAD",
        help=(
            "the revision the corpus is read at. Several lanes commit to this worktree "
            "while a run is in progress, so two runs against HEAD hours apart read "
            "different corpora; name a commit to reproduce a recorded row. With --ledger "
            "it must be a full 40-character sha: a row recorded against a name cannot be "
            "rebuilt"
        ),
    )
    parser.add_argument(
        "--blank-line-runs",
        action="store_true",
        help=(
            "leave runs of two or more blank lines in the span passages. With it, almost "
            "every span row is refused by the recorded line-start collapse; without it, "
            "the rest of the span path runs."
        ),
    )
    parser.add_argument(
        "--ledger",
        nargs="?",
        const=str(DEFAULT_LEDGER_PATH),
        default=None,
        help=(
            "append one `smoke` row carrying every number this run measured. Without it "
            "the run prints its numbers and records nothing, which under repo rule 5 means "
            "they may not appear in a report."
        ),
    )
    parser.add_argument(
        "--commitpackft",
        type=Path,
        default=None,
        help=(
            "a local bigcode/commitpackft download (one <lang>.jsonl per language plus the "
            "manifest.json pinning their sha256s, e.g. data/pool/commitpackft). The code "
            "rows then come from the plan's own source instead of this repository's "
            "history; --max-pairs becomes a sha256-ordered sample of it."
        ),
    )
    parser.add_argument(
        "--val-shards",
        action="store_true",
        help=(
            "also write the val split as a shard set (shards/val), for scoring a trained "
            "model on rows it did not train on. The remap is then built over the train AND "
            "val rows -- the written corpus -- because RemapTable.encode refuses any token "
            "it did not keep; the held-out rows still never enter it."
        ),
    )
    parser.add_argument(
        "--defect-class",
        type=Path,
        default=None,
        help=(
            "a qd-mutate corpus directory (examples.jsonl + manifest.json, e.g. "
            "data/pool/commitpackft-corpus-v2) to add as the code.defect_class family. "
            "Its pool and licence download are joined and sha256-checked; a row that does "
            "not resolve refuses the whole load."
        ),
    )
    parser.add_argument(
        "--defect-download",
        type=Path,
        default=DEFAULT_DEFECT_DOWNLOAD,
        help="the bigcode/commitpackft download the defect rows take their licence from",
    )
    parser.add_argument(
        "--defect-max-rows",
        type=int,
        default=None,
        help="cap the defect rows at a sha256-ordered sample; a cap that binds is reported",
    )
    parser.add_argument(
        "--memo-limit",
        type=int,
        default=MEMO_LIMIT,
        help=(
            "entries the tokenizer memo may hold before the run refuses; 0 disables the "
            "memo. The full defect corpus needs 0 -- it is ~100k sequences."
        ),
    )
    parser.add_argument(
        "--general-record",
        type=Path,
        default=None,
        help=(
            "a fetch record (e.g. ~/.cache/qd-decision/general/fetch-record-2026-09-29.json) "
            "naming the approved MMLU / CommonsenseQA / CLINC / SQuAD v2 JSONL caches. Each "
            "file must sit under the record's directory and match its recorded sha256 and "
            "row count; real SQuAD then replaces the repository-prose stand-in."
        ),
    )
    parser.add_argument(
        "--general-max-rows",
        type=int,
        default=DEFAULT_GENERAL_MAX_ROWS,
        help="per-file row bound for the general caches; a bound that binds is reported",
    )
    parser.add_argument(
        "--replay-shards",
        action="store_true",
        help=(
            "split qd_data.general.partition_replay's replay-only slice off the gold train "
            "split and write it as its own shard set (shards/replay, write_shards(replay=True)) "
            "under the same remap. Needs --general-record."
        ),
    )
    args = parser.parse_args(argv)
    # `--max-pairs` bounds what `base_sources` reads from history or samples from the
    # download. With neither in play it would determine nothing and still land in the
    # recipe, which feeds recipe_hash.
    max_pairs_bounds_nothing = not args.repo_history and args.commitpackft is None
    if max_pairs_bounds_nothing and args.max_pairs is not None:
        raise SystemExit(
            "--max-pairs bounds the repository-history rows and the --commitpackft sample; "
            "under --no-repo-history without --commitpackft it bounds nothing, and a value "
            "that determined nothing would still move recipe_hash. Drop it."
        )
    if not args.repo_history and args.blank_line_runs:
        raise SystemExit(
            "--blank-line-runs shapes the repository-prose span rows, and --no-repo-history "
            "reads none; it would determine nothing"
        )
    if args.max_pairs is None:
        args.max_pairs = DEFAULT_MAX_PAIRS
    if args.ledger is not None:
        # Before any work: a pipeline row is what every FT run downstream trains on, and a
        # recipe that says "HEAD" names a corpus that no longer exists after the next commit.
        try:
            require_full_sha(args.rev)
        except ValueError as exc:
            raise SystemExit(f"--ledger: {exc}") from exc
    args.out.mkdir(parents=True, exist_ok=True)
    run_kwargs: dict[str, Any] = {
        "out": args.out, "max_pairs": args.max_pairs,
        "blank_line_runs": args.blank_line_runs, "rev": args.rev,
        "commitpackft": args.commitpackft, "val_shards": args.val_shards,
        "defect_class": args.defect_class, "defect_download": args.defect_download,
        "defect_max_rows": args.defect_max_rows, "memo_limit": args.memo_limit,
        "general_record": args.general_record, "general_max_rows": args.general_max_rows,
        "replay_shards": args.replay_shards, "repo_history": args.repo_history,
    }
    if args.ledger is None:
        run(**run_kwargs)
        return 0

    if not MODEL_REF.exists():
        raise SystemExit(
            f"{MODEL_REF} is absent, so which revision of {MODEL} produced these numbers "
            "cannot be stated. A protocol naming no backbone identifies nothing; refusing "
            "to write a row rather than inventing one."
        )
    # Named so the row can carry it as well as be identified by it.
    recipe: dict[str, Any] = {
        "tool": "tools/real_tokenizer_pipeline.py",
        "model": MODEL,
        "max_pairs": args.max_pairs,
        "blank_line_runs": bool(args.blank_line_runs),
        "rev": args.rev,
    }
    if not args.repo_history:
        # Only when used, like every key below: a build with history hashes as before.
        recipe["repo_history"] = False
        # Neither bounds anything without history rows to read or span prose to join.
        del recipe["blank_line_runs"]
        if max_pairs_bounds_nothing:
            del recipe["max_pairs"]
    if args.commitpackft is not None:
        # Only when used: every row written before the flag existed hashed the five keys
        # above, and adding a sixth to all of them would rename that protocol family.
        # The sha256s, not the path, identify the corpus; run() refuses a file that no
        # longer matches them.
        recipe["commitpackft_sha256"] = pool_manifest_shas(args.commitpackft)
    if args.val_shards:
        # Only when used, for the same reason: it changes the remap, so it is the recipe.
        recipe["val_shards"] = True
    if args.defect_class is not None:
        # Only when used. The corpus sha256, not its path, names it; load_defect_rows
        # refuses a file that no longer matches. The memo limit is not recipe: it
        # changes how often the tokenizer is called, never what it returns.
        defect_manifest = json.loads(
            (args.defect_class / "manifest.json").read_text(encoding="utf-8")
        )
        recipe["defect_class_examples_sha256"] = str(defect_manifest["examples_sha256"])
        recipe["defect_download_sha256"] = pool_manifest_shas(args.defect_download)
        if args.defect_max_rows is not None:
            recipe["defect_max_rows"] = args.defect_max_rows
    if args.general_record is not None:
        # Only when used. The record's sha256 names the corpus; general_rows() refuses a
        # cache file that no longer matches the record.
        recipe["general_record_sha256"] = hashlib.sha256(
            args.general_record.read_bytes()
        ).hexdigest()
        recipe["general_max_rows"] = args.general_max_rows
    if args.replay_shards:
        recipe["replay_shards"] = True
    # It decides which rows exist (the contradictory-prompt drop runs only under it).
    recipe["max_consistency_rows"] = PIPELINE_MAX_CONSISTENCY_ROWS
    protocol = Protocol(
        # Filled after the run, which is why this is a placeholder only until then: a
        # Protocol is frozen, so the real one is built from what the run measured.
        data_snapshot_hash="pending",
        tokenizer_hash="pending",
        backbone_commit=MODEL_REF.read_text(encoding="utf-8").strip(),
        recipe_hash=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        seed=DataConfig().seed,
    )
    # `run()` is the work this row describes, and it finishes before the recorder is built
    # below, so the recorder would otherwise time its own metric writes.
    work_t0 = time.monotonic()
    measured = run(**run_kwargs)
    work_s = time.monotonic() - work_t0
    protocol = Protocol(
        data_snapshot_hash=measured.data_snapshot_hash,
        tokenizer_hash=measured.tokenizer_hash,
        backbone_commit=protocol.backbone_commit,
        recipe_hash=protocol.recipe_hash,
        seed=protocol.seed,
    )
    recorder = RunRecorder(
        Ledger(Path(args.ledger)),
        entry_point=Path(__file__),
        protocol=protocol,
        run_kind="smoke",
        repo=REPO,
        env=Environment.detect(transformers_sha=_transformers_version()),
        wall_clock_s=work_s,
        # This pipeline tokenises on whatever machine it is run on, and on a Mac that is
        # already bought nothing is billed by the hour. `None` states that. It is not a
        # blanket exemption: `Environment.detect` reports the real device, and on anything
        # outside CostEstimate.LOCAL_DEVICES the recorder refuses this rather than writing
        # an unstated zero that reads like a measured one.
        cost=None,
        # The same object the `recipe_hash` above was taken of.
        recipe=recipe,
        quick=True,
        quick_reason=measured.quick_reason,
        notes=measured.notes,
    )
    with recorder:
        # The shard sets this writes are what every FT run downstream trains on, so "which
        # code produced this corpus" is a question asked of its rows more than of any
        # other. `code_commit` cannot answer it: one `-dirty` bit stands for any
        # uncommitted change at all, and is permanent on a box synced by copying files.
        # `RunRecorder.__enter__` has already recorded `code_that_ran` from `entry_point`.
        for name, value in measured.metrics.items():
            recorder.metric(name, value)
        for name, value in measured.gates.items():
            recorder.gate(name, value)
        recorder.noul_rate = NotRun(
            reason="this run tokenizes a corpus; it runs no model, so no noul rate exists"
        )
    row = recorder.row
    if row is None:
        raise SystemExit("the recorder exited without a row; nothing was recorded")
    print(f"\nledger row: {row.row_id}\n  {Path(args.ledger)}")
    return 0


def _transformers_version() -> str:
    try:
        import transformers
    except ModuleNotFoundError:
        return "not-installed"
    return f"transformers=={transformers.__version__}"


if __name__ == "__main__":
    raise SystemExit(main())
