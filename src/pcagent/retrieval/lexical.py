"""BM25 over the taxonomy titles.

Written out rather than pulled in. The corpus here is a couple of thousand very
short documents, so the whole index is a few dictionaries and the scoring is
twenty lines. Having it in the repo also means the knobs below are visible and
arguable instead of buried in someone else's defaults.

Lexical retrieval earns its place in this problem because procurement text is
full of exact tokens that embeddings blur: part numbers, "8140", "MRO", "NSN",
unit sizes like "30MM". A dense model will happily decide those are roughly the
same as their neighbours. BM25 will not.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

# Deliberately small. Aggressive stopword lists hurt here, because words like
# "other" and "not" carry real meaning in taxonomy titles ("Other Crop Farming",
# "not elsewhere classified").
STOPWORDS = frozenset(
    """a an and are as at be by for from in into is it of on or that the to with""".split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+")

K1 = 1.2
"""Term frequency saturation. Low, because these documents are titles. A term
appearing twice in a six word title is not twice as relevant."""

B = 0.75
"""Length normalisation, at the usual default. Titles do vary enough in length
to be worth correcting for."""


def normalise(token: str) -> str:
    """Fold a token to its index form.

    Crude plural folding only. No stemmer. "Services" and "service" must match
    or half the service codes become unreachable, but anything more aggressive
    starts colliding codes that the taxonomy keeps deliberately distinct.

    The trailing "e" is only dropped after a sibilant, where it belongs to the
    plural ("boxes" to "box", "churches" to "church"). Dropping it everywhere
    turns "services" into "servic" while leaving "service" alone, which breaks
    the exact pair this function exists to join up.
    """
    if len(token) <= 3 or not token.endswith("s") or token.endswith("ss"):
        return token
    stem = token[:-1]
    if len(stem) > 3 and stem.endswith("e") and stem[:-1].endswith(("s", "x", "z", "ch", "sh")):
        return stem[:-1]
    return stem


def tokenize(text: str) -> list[str]:
    return [
        normalise(token)
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS
    ]


class BM25Index:
    """A static BM25 index. Built once, queried many times, never updated."""

    def __init__(self, documents: list[str]) -> None:
        self.doc_count = len(documents)
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        term_counts: list[dict[str, int]] = []
        for text in documents:
            tokens = tokenize(text)
            self._lengths.append(len(tokens))
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            term_counts.append(counts)

        for doc_index, counts in enumerate(term_counts):
            for term, count in counts.items():
                self._postings[term].append((doc_index, count))

        total_length = sum(self._lengths)
        self.average_length = total_length / self.doc_count if self.doc_count else 0.0

        # Precomputed because it only depends on the corpus, not the query.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (self.doc_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return ``(document index, score)`` for the best ``limit`` matches.

        Terms absent from the corpus are skipped rather than penalised. A query
        with no known terms returns nothing, which the caller is expected to
        handle, and the hybrid layer does.
        """
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_index, term_frequency in postings:
                length_norm = 1.0 - B + B * (self._lengths[doc_index] / self.average_length)
                scores[doc_index] += idf * (
                    term_frequency * (K1 + 1.0) / (term_frequency + K1 * length_norm)
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit]
