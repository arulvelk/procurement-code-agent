"""Shared fixtures.

The suite runs in a couple of seconds and downloads nothing. The ONNX models
are replaced by stubs, which is not a shortcut: the point of these tests is the
tool contract, the authorization rules and the idempotency protocol, and none of
those get more true with a real embedding model behind them. The models are
measured in ``evals/``, which is where model quality belongs.

The stubs are also how the degradation paths get tested at all. ``BrokenDense``
and ``BrokenReranker`` make "what happens with no models available" an ordinary
test rather than something you find out about in production.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio

from pcagent.auth import READ_ONLY, REVIEWER, Principal, Scope
from pcagent.domain import TaxonomyCode
from pcagent.llm.base import BaselineDecider
from pcagent.retrieval.dense import DenseUnavailable
from pcagent.retrieval.hybrid import HybridRetriever
from pcagent.retrieval.rerank import RerankUnavailable
from pcagent.store.db import Store
from pcagent.taxonomy.loader import TaxonomyCatalog
from pcagent.tools.base import Services, ToolContext

TINY_CODES = [
    ("7021", "COMPUTERS, DIGITAL", "70", "IT EQUIPMENT"),
    ("7025", "INPUT OUTPUT AND STORAGE DEVICES", "70", "IT EQUIPMENT"),
    ("7110", "OFFICE FURNITURE", "71", "FURNITURE"),
    ("7125", "CABINETS AND SHELVING", "71", "FURNITURE"),
    ("S201", "HOUSEKEEPING CUSTODIAL JANITORIAL", "S2", "HOUSEKEEPING SERVICES"),
    ("S205", "TRASH AND GARBAGE COLLECTION", "S2", "HOUSEKEEPING SERVICES"),
    ("V121", "TRANSPORTATION AIR CHARTER", "V1", "TRANSPORTATION OF THINGS"),
    ("1005", "GUNS THROUGH 30MM", "10", "WEAPONS"),
]


class StubDense:
    """Scores documents by how many query words they contain.

    Crude, deterministic, and enough to produce a ranking that differs from
    BM25, which is what the fusion tests need.
    """

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.warmed = False

    def warm(self) -> None:
        self.warmed = True

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        words = {w.lower() for w in query.split() if len(w) > 2}
        scored = [
            (index, float(sum(1 for w in words if w in text.lower())))
            for index, text in enumerate(self.documents)
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]


class BrokenDense:
    def warm(self) -> None:
        raise DenseUnavailable("no model in tests")

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        raise DenseUnavailable("no model in tests")


class StubReranker:
    """Reverses the candidate order so a test can prove reranking was applied."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def warm(self) -> None:
        return None

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        return [float(index) for index in range(len(documents))]


class BrokenReranker:
    def warm(self) -> None:
        raise RerankUnavailable("no model in tests")

    def score(self, query: str, documents: list[str]) -> list[float]:
        raise RerankUnavailable("no model in tests")


@pytest.fixture
def catalog() -> TaxonomyCatalog:
    codes = [
        TaxonomyCode(
            taxonomy="psc",
            code=code,
            title=title,
            parent_code=parent,
            parent_title=parent_title,
            category="Product" if code[0].isdigit() else "Service",
        )
        for code, title, parent, parent_title in TINY_CODES
    ]
    return TaxonomyCatalog("psc", codes)


@pytest.fixture
def retriever(catalog: TaxonomyCatalog) -> HybridRetriever:
    documents = [code.index_text for code in catalog.codes]
    return HybridRetriever(
        catalog, dense=StubDense(documents), reranker=StubReranker()
    )


@pytest.fixture
def lexical_only_retriever(catalog: TaxonomyCatalog) -> HybridRetriever:
    """Everything optional is unavailable, so only BM25 is left."""
    return HybridRetriever(catalog, dense=BrokenDense(), reranker=BrokenReranker())


@pytest_asyncio.fixture
async def store(tmp_path) -> AsyncIterator[Store]:
    store = Store(tmp_path / "test.db")
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def services(retriever: HybridRetriever, store: Store) -> Services:
    return Services(
        retrievers={"psc": retriever}, decider=BaselineDecider(), store=store
    )


@pytest.fixture
def reader(services: Services) -> ToolContext:
    return ToolContext(Principal("svc-reader", READ_ONLY), services)


@pytest.fixture
def reviewer(services: Services) -> ToolContext:
    return ToolContext(Principal("ana.reviewer", REVIEWER), services)


@pytest.fixture
def nobody(services: Services) -> ToolContext:
    return ToolContext(Principal("drive-by", frozenset()), services)


@pytest.fixture
def all_scopes() -> frozenset[Scope]:
    return frozenset(Scope)


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    import logging

    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)
