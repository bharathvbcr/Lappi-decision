"""MinHash over a hash family we own, because the dedupe decision is load-bearing.

``docs/hardening.md`` section 2:

    **Repo-level is not sufficient by itself.** Vendored trees, forks and copied
    files put the same code in two repos. MinHash dedupe at 0.8 Jaccard runs
    **across pool and held-out before splitting**.

Two design decisions here, both for the same reason -- the dedupe outcome feeds
``data_snapshot_hash``, and a hash that is not reproducible makes the whole ledger
protocol a fiction:

1. **The permutation family is derived from blake2b, not from a library's RNG.**
   ``datasketch`` is a declared dependency and its estimator is the reference this
   module is cross-checked against (``tests/test_minhash.py``), but its hash values
   come from ``numpy.random.RandomState`` seeded per ``MinHash`` object. That stream
   is a property of a library version, so a ``datasketch`` upgrade would silently
   change which rows are near-duplicates and therefore change the snapshot hash of
   an unchanged corpus. Here the signature of a document is a pure function of
   (document, ``num_perm``, ``seed``) and of nothing else -- no process state, no
   library version, no platform.
2. **LSH is a candidate generator only, and its recall is a number, not an
   assumption.** Every candidate pair is confirmed with an *exact* Jaccard over the
   shingle sets before it is called a duplicate, so the banding choice can cost
   recall but can never produce a false duplicate -- the error is one-sided, and
   dedupe can under-remove but never over-remove.

   What it costs is on the record rather than implied away.
   :attr:`BandConfig.recall_at_threshold` states the per-pair probability that a pair
   sitting exactly at the threshold is ever proposed, and ``qd_data.dedupe`` carries
   it into the report and the manifest. At the configured ``num_perm=128,
   threshold=0.8`` that is **0.947**: roughly one pair in nineteen at the threshold
   itself is never proposed and therefore never confirmed. For the population this
   targets -- vendored trees, forks, copied files, typically J > 0.95 -- the same
   curve gives 1.0 to six decimal places, which is why the configuration is usable;
   but "usable" and "exhaustive" are different claims and only one of them is true.
   ``tests/test_minhash.py`` derives that number from the banding and then measures
   it against a brute-force pairwise baseline at exactly J = 0.80, so the bound in
   the report is checked rather than quoted.

   v6's agreement prefilter (``BandConfig.min_agreement_permille``) trades a little more
   of that recall for a candidate set that is not swamped by one-band collisions far below
   the threshold; what it can cost at the threshold is subtracted from the stated recall
   (:meth:`BandConfig.prefilter_miss_at`), and the error stays one-sided.

The estimator itself is the standard one: ``P[min_h(A) == min_h(B)] = J(A, B)``, so
the fraction of agreeing signature positions is an unbiased estimate of Jaccard with
standard error ``sqrt(J(1-J)/num_perm)`` -- about 4% at ``num_perm=128``, J=0.8.
That error is why the confirmation pass is exact rather than estimated.
"""

from __future__ import annotations

import functools
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEFAULT_FALSE_NEGATIVE_WEIGHT",
    "DEFAULT_FALSE_POSITIVE_WEIGHT",
    "MERSENNE_PRIME",
    "BandConfig",
    "MinHasher",
    "ShingleResult",
    "candidate_pairs",
    "choose_bands",
    "estimate_jaccard",
    "exact_jaccard",
    "shingle",
]

#: 2**61 - 1. The permutation family is ``(a*h + b) mod p`` over this prime, which is
#: the standard construction and is what ``datasketch`` uses too, so the two
#: estimators are comparable in the cross-check test.
MERSENNE_PRIME: Final[int] = (1 << 61) - 1

#: Documents longer than this are shingled over their first ``N`` bytes only. The
#: count of truncated documents is carried to the manifest: a bound that silently
#: changes what "duplicate" means is the same defect class as a truncated context.
DEFAULT_MAX_DOC_BYTES: Final[int] = 262_144

_WS: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ShingleResult:
    """A document's shingle set, plus whether the document was bounded to produce it."""

    shingles: frozenset[bytes]
    truncated: bool
    n_tokens: int

    def __bool__(self) -> bool:
        return bool(self.shingles)


def shingle(
    text: str,
    *,
    k: int,
    max_doc_bytes: int = DEFAULT_MAX_DOC_BYTES,
) -> ShingleResult:
    """Whitespace-normalised token ``k``-shingles.

    Normalisation collapses every run of whitespace to a single space and nothing
    else. Case is **kept**: the duplicates this is hunting are vendored trees, forks
    and copied files, which are byte-identical or nearly so, and case-folding code
    would merge ``Foo`` with ``foo`` for no gain against that population.

    A document with fewer than ``k`` tokens yields exactly one shingle -- the whole
    token tuple. Returning an empty set instead would make every short row a
    duplicate of every other short row under Jaccard's 0/0 convention, which is the
    quiet hole this avoids.
    """
    if k < 1:
        raise ValueError(f"shingle size must be >= 1, got {k}")
    if max_doc_bytes < 1:
        raise ValueError(f"max_doc_bytes must be >= 1, got {max_doc_bytes}")

    raw = text.encode("utf-8")
    truncated = len(raw) > max_doc_bytes
    if truncated:
        # Cut on a codepoint boundary; a partial sequence would decode-error.
        raw = raw[:max_doc_bytes]
        text = raw.decode("utf-8", errors="ignore")

    tokens = _WS.sub(" ", text).strip().split(" ")
    if tokens == [""]:
        tokens = []

    if len(tokens) < k:
        grams = [" ".join(tokens)] if tokens else []
    else:
        grams = [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]

    return ShingleResult(
        shingles=frozenset(g.encode("utf-8") for g in grams),
        truncated=truncated,
        n_tokens=len(tokens),
    )


class MinHasher:
    """A reproducible MinHash signature generator.

    The signature of a document depends on the document, ``num_perm`` and ``seed``,
    and on nothing else. Two processes, two Python versions and two machines produce
    the same integers.
    """

    __slots__ = ("_a", "_b", "_key", "num_perm", "seed")

    def __init__(self, *, num_perm: int, seed: int) -> None:
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, got {num_perm}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        self.num_perm = num_perm
        self.seed = seed
        material = f"qd_data.minhash.v1|{seed}".encode()
        self._key = hashlib.blake2b(material, digest_size=32).digest()
        a: list[int] = []
        b: list[int] = []
        for i in range(num_perm):
            blob = hashlib.blake2b(i.to_bytes(8, "big"), key=self._key, digest_size=32).digest()
            # a must be non-zero modulo p or the permutation collapses to a constant.
            ai = int.from_bytes(blob[:16], "big") % (MERSENNE_PRIME - 1) + 1
            bi = int.from_bytes(blob[16:], "big") % MERSENNE_PRIME
            a.append(ai)
            b.append(bi)
        self._a = tuple(a)
        self._b = tuple(b)

    def base_hash(self, item: bytes) -> int:
        """64-bit hash of one shingle, folded into the prime field."""
        return int.from_bytes(hashlib.blake2b(item, key=self._key, digest_size=8).digest(), "big")

    def signature(self, shingles: frozenset[bytes] | set[bytes]) -> tuple[int, ...]:
        """The MinHash signature.

        An empty shingle set has no minimum, and inventing one (say, ``max_int``)
        would make every empty document identical to every other. It is refused
        instead; :func:`shingle` never returns an empty set for a non-empty document,
        so reaching here means a genuinely empty row, which the mixture refuses
        upstream.
        """
        if not shingles:
            raise ValueError(
                "cannot sign an empty shingle set: every empty document would hash "
                "identically and be deduped against every other empty document"
            )
        bases = [self.base_hash(s) for s in shingles]
        p = MERSENNE_PRIME
        return tuple(
            min((self._a[i] * h + self._b[i]) % p for h in bases) for i in range(self.num_perm)
        )


def exact_jaccard(a: frozenset[bytes] | set[bytes], b: frozenset[bytes] | set[bytes]) -> float:
    """|A and B| / |A or B|. Two empty sets are **not** similar: 0.0, not 1.0.

    The 0/0 convention matters. Calling two empty documents identical would collapse
    every degenerate row into one cluster and drop real data.
    """
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def estimate_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Fraction of agreeing signature positions."""
    if len(sig_a) != len(sig_b):
        raise ValueError(
            f"signatures must share a permutation count: {len(sig_a)} vs {len(sig_b)}"
        )
    if not sig_a:
        raise ValueError("empty signature")
    return sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x == y) / len(sig_a)


@dataclass(frozen=True, slots=True)
class BandConfig:
    """The banding used, reported rather than hidden, because it sets LSH recall."""

    bands: int
    rows: int
    threshold: float
    #: The agreement prefilter (``DataConfig.lsh_min_agreement_permille``): a pair that shares
    #: a band is a candidate only if its first ``bands * rows`` signature values agree on at
    #: least this many positions per mille. 0, v5's, is no prefilter.
    #: GAP-DEDUPE-LSH-BAND-CANDIDATES-NOT-DUPLICATES-2026-10-03: at b=16, r=8 one shared band
    #: proposes a J=0.6 pair with probability 0.24, and v5's templated decision rows proposed
    #: millions of pairs at J 0.5-0.8 that the exact confirmation then rejected.
    min_agreement_permille: int = 0

    def __post_init__(self) -> None:
        p = self.min_agreement_permille
        if not isinstance(p, int) or isinstance(p, bool) or not 0 <= p <= 1000:
            raise ValueError(f"min_agreement_permille must be an int in [0, 1000], got {p!r}")

    @property
    def num_perm_used(self) -> int:
        return self.bands * self.rows

    def probability_at(self, jaccard: float) -> float:
        """S-curve: ``1 - (1 - J**r)**b``. The chance a pair at this Jaccard is a
        candidate."""
        return 1.0 - (1.0 - jaccard**self.rows) ** self.bands

    def prefilter_miss_at(self, jaccard: float) -> float:
        """The chance a pair at this Jaccard fails the agreement prefilter: each of the
        ``bands * rows`` positions agrees independently with probability ``J``, so this is
        ``P(Binomial(width, J) * 1000 < permille * width)``. 0.0 with no prefilter."""
        width = self.bands * self.rows
        if not self.min_agreement_permille or width == 0:
            return 0.0
        need = -(-self.min_agreement_permille * width // 1000)  # the least passing count
        return math.fsum(
            math.comb(width, k) * jaccard**k * (1.0 - jaccard) ** (width - k)
            for k in range(need)
        )

    @property
    def recall_at_threshold(self) -> float:
        """Per-pair recall at exactly ``threshold`` -- the bound, stated.

        With the agreement prefilter it is the union-bound floor ``P(band) - P(prefilter
        fails)`` (:meth:`prefilter_miss_at`), never the unfiltered S-curve: the two events are
        correlated, so their product would be a guess, and the difference is a floor.

        ``GAP-DATA-LSH-RECALL-BOUND``. Banded LSH is a probabilistic filter and this
        is the number that says so. It is a **floor** for the population above the
        threshold, because :meth:`probability_at` is non-decreasing in ``J``: at
        ``num_perm=128, threshold=0.8`` the search picks ``b=16, r=8``, giving 0.9470
        at the threshold, 0.9938 at 0.85 and 0.9999 at 0.90.

        It is exposed as a property rather than left implicit in the banding because
        a report that prints ``bands`` and ``rows`` and stops has told the reader the
        cause without telling them the effect, and every reader then has to
        re-derive the S-curve or assume the filter was exhaustive. It is not.
        """
        banded = self.probability_at(self.threshold)
        if not self.min_agreement_permille:
            return banded
        return max(0.0, banded - self.prefilter_miss_at(self.threshold))


#: The banding objective is **deliberately not balanced**. A false positive costs one
#: exact-Jaccard comparison and nothing else, because every candidate is confirmed
#: exactly before it is called a duplicate -- so precision is exact whatever the
#: banding does. A false negative is an undetected near-duplicate, which for a
#: *leakage* control means contamination that invalidates every downstream number.
#: The two are not symmetric costs and must not be given symmetric weights.
#:
#: Measured at ``num_perm=128``, ``threshold=0.8``: balanced weights pick ``b=9,
#: r=13``, whose probability of even proposing a genuine 0.85-Jaccard pair is
#: **0.686** -- it would miss three near-duplicate pairs in ten. These weights pick
#: ``b=16, r=8``: 0.947 at the threshold itself, 0.994 at 0.85, 0.9999 at 0.9, and a
#: 0.061 candidate rate at 0.5, which is the price paid in confirmations.
DEFAULT_FALSE_POSITIVE_WEIGHT: Final[float] = 0.05
DEFAULT_FALSE_NEGATIVE_WEIGHT: Final[float] = 0.95


@functools.lru_cache(maxsize=64)
def choose_bands(
    *,
    num_perm: int,
    threshold: float,
    false_positive_weight: float = DEFAULT_FALSE_POSITIVE_WEIGHT,
    false_negative_weight: float = DEFAULT_FALSE_NEGATIVE_WEIGHT,
) -> BandConfig:
    """Pick ``(b, r)`` minimising the weighted false-negative/false-positive area.

    Deterministic and exhaustive over the grid ``b*r <= num_perm``, so the choice is
    a pure function of the arguments and lands in the manifest. Cached because the
    search is quadratic in ``num_perm`` and the answer never changes.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if num_perm < 1:
        raise ValueError(f"num_perm must be >= 1, got {num_perm}")
    if false_positive_weight < 0 or false_negative_weight <= 0:
        raise ValueError(
            "weights must be non-negative and the false-negative weight positive; got "
            f"fp={false_positive_weight}, fn={false_negative_weight}. A zero "
            "false-negative weight would choose a banding that finds nothing."
        )

    best: tuple[float, int, int] | None = None
    steps = 200
    for b in range(1, num_perm + 1):
        max_r = num_perm // b
        for r in range(1, max_r + 1):
            # Riemann sum of the miss probability above the threshold plus the
            # spurious-candidate probability below it.
            fn = 0.0
            fp = 0.0
            for i in range(steps + 1):
                j = i / steps
                prob = 1.0 - (1.0 - j**r) ** b
                if j >= threshold:
                    fn += (1.0 - prob) / (steps + 1)
                else:
                    fp += prob / (steps + 1)
            cost = false_positive_weight * fp + false_negative_weight * fn
            if best is None or cost < best[0] - 1e-12:
                best = (cost, b, r)
    assert best is not None  # num_perm >= 1 guarantees at least (1, 1)
    return BandConfig(bands=best[1], rows=best[2], threshold=threshold)


def candidate_pairs(
    signatures: dict[str, tuple[int, ...]],
    *,
    config: BandConfig,
    max_pairs: int,
) -> tuple[frozenset[tuple[str, str]], bool]:
    """Banded LSH. Returns ``(pairs, truncated)``; pairs are ordered ``(lo, hi)``.

    ``max_pairs`` is a hard bound. When it is hit the function stops and says so in
    the second element -- it never returns a short list that looks complete. The
    caller (``qd_data.dedupe``) turns a truncated candidate set into a ``NotRun``
    rather than into a clean dedupe report.

    With ``config.min_agreement_permille`` set, a pair two keys' band proposes is added only if
    their first ``bands * rows`` values agree on ``agree * 1000 >= permille * width`` positions
    -- integers, so ``qd-prep lsh`` decides every boundary pair the same way -- and a pair it
    rejects never counts toward ``max_pairs``. Every signature must then hold the whole banded
    prefix before any band is read, since the prefilter reads all of it.
    """
    if max_pairs < 0:
        raise ValueError(f"max_pairs must be >= 0, got {max_pairs}")
    band_rows = config.rows
    pairs: set[tuple[str, str]] = set()
    truncated = False
    permille = config.min_agreement_permille
    width = config.bands * band_rows
    rejected: set[tuple[str, str]] = set()
    if permille:
        for key, sig in signatures.items():
            if len(sig) < width:
                raise ValueError(
                    f"signature for {key!r} has {len(sig)} permutations, but the band "
                    f"configuration needs {width}"
                )

    for band in range(config.bands):
        buckets: dict[bytes, list[str]] = {}
        lo = band * band_rows
        hi = lo + band_rows
        for key, sig in signatures.items():
            if len(sig) < hi:
                raise ValueError(
                    f"signature for {key!r} has {len(sig)} permutations, but the band "
                    f"configuration needs {hi}"
                )
            chunk = b"|".join(v.to_bytes(8, "big") for v in sig[lo:hi])
            digest = hashlib.blake2b(chunk, digest_size=16).digest()
            buckets.setdefault(digest, []).append(key)
        for members in buckets.values():
            if len(members) < 2:
                continue
            members.sort()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair = (members[i], members[j])
                    if permille and pair not in pairs:
                        if pair in rejected:
                            continue
                        agree = sum(
                            1
                            for x, y in zip(
                                signatures[pair[0]][:width], signatures[pair[1]][:width],
                                strict=True,
                            )
                            if x == y
                        )
                        if agree * 1000 < permille * width:
                            rejected.add(pair)
                            continue
                    pairs.add(pair)
                    if len(pairs) > max_pairs:
                        return frozenset(pairs), True
        if truncated:
            break

    return frozenset(pairs), truncated
