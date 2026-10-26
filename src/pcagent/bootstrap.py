"""Building the service graph once, at startup.

Indexes take a few seconds to warm and model clients hold connection pools, so
all of it is constructed once and shared. Both transports and the evals call
``build_services``, which is the only place that knows how the pieces fit
together.
"""

from __future__ import annotations

import asyncio
import logging

from .config import Settings
from .llm.base import BaselineDecider, Decider
from .retrieval.hybrid import HybridRetriever, RetrievalSettings
from .store.db import Store
from .taxonomy.loader import load_catalog
from .tools.base import Services

logger = logging.getLogger(__name__)

# Chosen on the dev split, reported on test. See evals/report-*.md.
#
# Reranking is off everywhere, which was not the plan. Measured on dev, the
# cross encoder lowers MRR on all three corpora:
#
#     psc     hybrid 0.123 -> hybrid+rerank 0.068
#     naics   hybrid 0.171 -> hybrid+rerank 0.157
#     arxiv   hybrid 0.398 -> hybrid+rerank 0.251
#
# It did help earlier, when the dense index was bge-small: reranking was worth
# 28% relative recall@5 then. Upgrading the embedder to bge-base is what killed
# it. A cross encoder adds value by fixing a bi-encoder's mistakes, and once the
# bi-encoder stopped making those mistakes the reranker was left reordering
# candidates it understood worse than the embedder did, for 170ms a query.
#
# The stage is kept in the codebase and in the ablation rather than deleted,
# because this is a property of these corpora and these models. Swap either and
# the answer can flip back, and the ablation is what would say so.
RETRIEVAL_SETTINGS: dict[str, RetrievalSettings] = {
    "psc": RetrievalSettings(rerank_enabled=False),
    "naics": RetrievalSettings(rerank_enabled=False),
    "arxiv": RetrievalSettings(rerank_enabled=False),
}


def build_decider(settings: Settings) -> Decider:
    if settings.decider == "claude":
        from .llm.claude import LLMDecider

        return LLMDecider(
            model=settings.model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return BaselineDecider()


async def build_services(settings: Settings, *, warm: bool = True) -> Services:
    retrievers: dict[str, HybridRetriever] = {}
    for taxonomy in settings.taxonomies:
        catalog = load_catalog(taxonomy, settings.data_dir)  # type: ignore[arg-type]
        retrievers[taxonomy] = HybridRetriever(
            catalog, settings=RETRIEVAL_SETTINGS.get(taxonomy, RetrievalSettings())
        )
        logger.info("loaded %s catalog with %d codes", taxonomy, len(catalog))

    if warm:
        # Warming both taxonomies at once is worth it because each one spends
        # most of its time inside ONNX, which releases the GIL.
        await asyncio.gather(*(retriever.warm() for retriever in retrievers.values()))

    store = Store(settings.database_path)
    await store.connect()

    return Services(
        retrievers=retrievers,
        decider=build_decider(settings),
        store=store,
    )
