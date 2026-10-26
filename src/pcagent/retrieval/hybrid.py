"""Hybrid retrieval: BM25 and dense, fused, then reranked.

The shape of the pipeline
-------------------------
    query
      |-- BM25 over the whole corpus          -> ranked list A
      |-- cosine over the whole corpus        -> ranked list B
      |
      +-- reciprocal rank fusion of A and B   -> candidate list
             |
             +-- cross encoder over the top N -> final ordering

Why fuse by rank rather than by score
-------------------------------------
BM25 scores are unbounded sums of IDF terms. Cosine similarities live in a
narrow band near the top of the range, where 0.67 is a strong match and 0.61 is
a poor one. There is no honest way to put those two numbers on one scale, and
the usual fix of normalising each to 0..1 per query just invents a scale that
changes with every query.

Reciprocal rank fusion sidesteps the problem by throwing the scores away and
keeping only the ordering, which is the part both rankers agree is meaningful.
It has one parameter, it needs no tuning per corpus, and it degrades gracefully
when one ranker returns nothing.

Degrading
---------
Both the dense index and the reranker are optional at runtime. If the ONNX
models cannot be loaded, retrieval falls back to BM25 alone and says so on the
result rather than failing. A reviewer with no network still gets a working
system, and the eval table reports what each stage was actually worth.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..domain import Candidate, Taxonomy
from ..taxonomy.loader import TaxonomyCatalog
from .dense import DenseIndex, DenseUnavailable
from .lexical import BM25Index
from .rerank import CrossEncoderReranker, RerankUnavailable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalSettings:
    lexical_k: int = 50
    """How deep to go in BM25. Generous, because this stage is nearly free and
    its job is recall, not precision."""

    dense_k: int = 50

    rrf_k: int = 60
    """The usual constant from the reciprocal rank fusion paper. It controls how
    sharply rank 1 outweighs rank 10. Left at the published value because tuning
    it on this corpus would be fitting noise."""

    rerank_depth: int = 30
    """How many fused candidates reach the cross encoder. This is the one real
    cost and latency knob in the pipeline. Measured in evals/report.md."""

    final_k: int = 8
    """How many candidates the LLM is shown. Small on purpose. A longer list
    does not help the model and gives it more ways to pick something plausible
    but wrong."""

    rerank_enabled: bool = True
    """Whether the cross encoder runs at all, decided per corpus from measured
    results rather than assumed. On the procurement corpora it is worth 28%
    relative recall@5. On arXiv it is actively harmful, dropping MRR from 0.398
    to 0.251, because a cross encoder trained on web passages is being asked to
    match a paper abstract against a subject definition, which is not the kind
    of pair it was trained on. "Retrieve then rerank" is a default worth
    questioning, not a law."""

    rerank_query_chars: int = 400
    """Queries are truncated before reranking. A cross encoder reads the query
    and the document in one window, so a 1400 character abstract pushes the
    document out of it and the model ends up scoring almost nothing. Measured:
    truncating to 300 characters lifted MRR from 0.185 to 0.251 on arXiv."""


@dataclass
class RetrievalOutcome:
    candidates: list[Candidate]
    used_dense: bool
    used_rerank: bool
    notes: list[str] = field(default_factory=list)


class HybridRetriever:
    """Owns the indexes for one taxonomy and answers queries against them."""

    def __init__(
        self,
        catalog: TaxonomyCatalog,
        *,
        dense: DenseIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.catalog = catalog
        self.settings = settings or RetrievalSettings()
        self._documents = [code.index_text for code in catalog.codes]
        self._lexical = BM25Index(self._documents)
        self._dense = dense if dense is not None else DenseIndex(self._documents)
        self._reranker = reranker if reranker is not None else CrossEncoderReranker()
        self._dense_broken = False
        self._rerank_broken = False

    @property
    def taxonomy(self) -> Taxonomy:
        return self.catalog.taxonomy

    async def warm(self) -> None:
        """Build the indexes before serving traffic.

        Both model loads are blocking, so they go to a thread even here. The
        rule in this repo is that nothing CPU bound runs on the loop, including
        during startup, because startup is exactly when a health check is
        waiting on the loop to answer.
        """
        try:
            await asyncio.to_thread(self._dense.warm)
        except DenseUnavailable as exc:
            self._dense_broken = True
            logger.warning("dense retrieval unavailable, falling back to lexical: %s", exc)
        if not self.settings.rerank_enabled:
            # Loading it would download 150MB of weights for a stage this
            # corpus measured as harmful. The eval harness constructs its own
            # reranker, so the ablation still runs.
            logger.info("reranking disabled for %s, not loading the model", self.taxonomy)
            return

        try:
            await asyncio.to_thread(self._reranker.warm)
        except RerankUnavailable as exc:
            self._rerank_broken = True
            logger.warning("reranking unavailable, using fused order: %s", exc)

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        rerank: bool | None = None,
    ) -> RetrievalOutcome:
        """Retrieve candidates for ``query``.

        ``rerank`` defaults to whatever this corpus measured best at. Passing
        it explicitly overrides that, which is what the eval harness does when
        it builds the ablation table.
        """
        query = query.strip()
        if not query:
            return RetrievalOutcome(candidates=[], used_dense=False, used_rerank=False)

        settings = self.settings
        limit = limit or settings.final_k
        rerank = settings.rerank_enabled if rerank is None else rerank
        notes: list[str] = []

        # BM25 over a few thousand short documents is tens of microseconds. A
        # thread hop would cost more than the work, so it runs inline. The rule
        # is "do not block the loop", not "never compute on the loop".
        lexical_hits = self._lexical.search(query, settings.lexical_k)

        dense_hits: list[tuple[int, float]] = []
        used_dense = False
        if not self._dense_broken:
            try:
                dense_hits = await asyncio.to_thread(
                    self._dense.search, query, settings.dense_k
                )
                used_dense = True
            except DenseUnavailable as exc:
                self._dense_broken = True
                notes.append("dense retrieval unavailable: " + str(exc))
                logger.warning("dense retrieval unavailable: %s", exc)

        fused = reciprocal_rank_fusion(
            [hits for hits in (lexical_hits, dense_hits) if hits],
            rrf_k=settings.rrf_k,
        )
        if not fused:
            return RetrievalOutcome(
                candidates=[], used_dense=used_dense, used_rerank=False, notes=notes
            )

        used_rerank = False
        ordering = fused
        if rerank and not self._rerank_broken:
            head = fused[: settings.rerank_depth]
            try:
                scores = await asyncio.to_thread(
                    self._reranker.score,
                    query[: settings.rerank_query_chars],
                    [self._documents[index] for index, _ in head],
                )
            except RerankUnavailable as exc:
                self._rerank_broken = True
                notes.append("reranking unavailable: " + str(exc))
                logger.warning("reranking unavailable: %s", exc)
            else:
                reranked = sorted(
                    zip((index for index, _ in head), scores, strict=True),
                    key=lambda pair: -pair[1],
                )
                # Anything past rerank_depth keeps its fused order and stays
                # behind everything the cross encoder actually looked at.
                ordering = reranked + fused[settings.rerank_depth :]
                used_rerank = True

        fused_rank_by_index = {index: rank for rank, (index, _) in enumerate(fused, start=1)}
        candidates = [
            Candidate(
                code=self.catalog.codes[index].code,
                title=self.catalog.codes[index].title,
                taxonomy=self.taxonomy,
                score=float(score),
                retrieved_rank=fused_rank_by_index[index],
                reranked_rank=position if used_rerank else None,
            )
            for position, (index, score) in enumerate(ordering[:limit], start=1)
        ]

        return RetrievalOutcome(
            candidates=candidates,
            used_dense=used_dense,
            used_rerank=used_rerank,
            notes=notes,
        )


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    *,
    rrf_k: int,
) -> list[tuple[int, float]]:
    """Fuse ranked lists into one, scoring each document by 1 / (k + rank).

    Public because the eval harness fuses stages itself to measure each one in
    isolation. It has to use the same function the service uses, or the ablation
    table would be measuring a reimplementation.

    Documents found by both rankers rise, which is the entire point. A document
    only one ranker found can still place well if it placed very highly there.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (document_index, _score) in enumerate(ranking, start=1):
            scores[document_index] = scores.get(document_index, 0.0) + 1.0 / (rrf_k + rank)
    # Ties break on document index so the output is stable run to run, which
    # matters when an eval diff is supposed to mean something changed.
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
