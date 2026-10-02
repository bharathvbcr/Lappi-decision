"""Read-only oracle over L-prep2's stripped subsample requests (Fable ruling, 2026-10-02).

For each QDPCTIN1 request: take every intent.domain row's stripped text (= the bare
utterance under the as-written strip: question and all ten domain options removed), key it
by identity key per set, and compute option (c) -- utterance-only containment at n=8,
threshold 0.5 -- train -> val and train -> heldout, per identity key. Also: too-short keys
per set, and whether any too-short val/heldout utterance equals (normalized) or is a
contiguous word-subsequence of a train utterance (what an exact key would catch).
"""
from __future__ import annotations

import hashlib
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

MAGIC = b"QDPCTIN1"
N = 8
T = 0.5
WORD = re.compile(r"\w+")


def _s(fh):
    (n,) = struct.unpack("<I", fh.read(4))
    return fh.read(n).decode("utf-8")


def rows(request: Path):
    with request.open("rb") as fh:
        assert fh.read(8) == MAGIC
        fh.read(12)
        _s(fh); _s(fh)
        (nc,) = struct.unpack("<I", fh.read(4))
        for _ in range(nc):
            _s(fh); fh.read(1); _s(fh)
        (ns,) = struct.unpack("<I", fh.read(4))
        names = [_s(fh) for _ in range(ns)]
        (nsc,) = struct.unpack("<I", fh.read(4))
        fh.read(9 * nsc)
        (nr,) = struct.unpack("<Q", fh.read(8))
        for _ in range(nr):
            (i,) = struct.unpack("<I", fh.read(4))
            yield names[i], _s(fh), _s(fh), _s(fh), _s(fh)


def words(text):
    return WORD.findall(text.lower())


def grams(ws):
    return frozenset(
        hashlib.blake2b(" ".join(ws[i:i + N]).encode(), digest_size=8).digest()
        for i in range(len(ws) - N + 1)
    )


def main(paths):
    for p in paths:
        req = Path(p)
        utt = defaultdict(dict)          # set -> identity -> utterance
        fam_texts = defaultdict(dict)    # set -> (identity, family) -> text (as-written strip)
        for sname, key, ident, fam, text in rows(req):
            if fam == "intent.domain":
                utt[sname][ident] = text
            if fam.startswith("intent."):
                fam_texts[sname][(ident, fam)] = text
        train = utt["train"]
        tg = {k: grams(words(v)) for k, v in train.items()}
        out = {"request": str(req), "train_keys": len(train)}
        excluded = set()
        for target in ("val", "heldout"):
            tv = utt.get(target, {})
            short = {k for k, v in tv.items() if len(words(v)) < N}
            hits = set(); pairs = 0
            for k, v in tv.items():
                g = grams(words(v))
                if not g:
                    continue
                for tk, tgr in tg.items():
                    c = len(g & tgr)
                    if c / len(g) >= T:
                        hits.add(tk); pairs += 1
            excluded |= hits
            # what an exact / subsequence key would catch for the too-short targets
            train_norm = defaultdict(list)
            for tk, v in train.items():
                train_norm[" ".join(words(v))].append(tk)
            exact = {k for k in short if " ".join(words(tv[k])) in train_norm}
            exact_any = {k for k in tv if " ".join(words(tv[k])) in train_norm}
            sub = set()
            for k in short:
                s = " " + " ".join(words(tv[k])) + " "
                if any(s in (" " + tn + " ") for tn in train_norm):
                    sub.add(k)
            out[target] = {
                "keys": len(tv), "too_short_keys": len(short),
                "train_keys_hit_utterance_only": len(hits), "pairs": pairs,
                "too_short_with_exact_train_match": len(exact),
                "any_key_with_exact_train_match": len(exact_any),
                "too_short_contained_as_word_subsequence_of_a_train_utterance": len(sub),
                "examples_subsequence": [tv[k] for k in sorted(sub)][:8],
            }
        out["clinc_keys_excluded_utterance_only"] = len(excluded)
        out["rate"] = round(len(excluded) / len(train), 4)
        # sanity: the as-written stripped intent.domain / in_scope texts are the same string
        same = all(
            fam_texts[s][(k, "intent.domain")] == fam_texts[s][(k, "intent.in_scope")]
            for s in fam_texts for (k, f) in fam_texts[s] if f == "intent.domain"
        )
        out["domain_text_equals_in_scope_text"] = same
        # how many train utterances are < 8 words (cannot hit anything either)
        out["train_too_short_keys"] = sum(1 for v in train.values() if len(words(v)) < N)
        # examples of utterance-only hits
        ex = []
        for target in ("val", "heldout"):
            for k, v in utt.get(target, {}).items():
                g = grams(words(v))
                if not g:
                    continue
                for tk, tgr in tg.items():
                    if len(g & tgr) / len(g) >= T:
                        ex.append((train[tk], v, len(g & tgr), len(g)))
        out["examples_hits"] = ex[:12]
        import json
        print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
