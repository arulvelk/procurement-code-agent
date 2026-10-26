"""Proof that the CPU bound work really does leave the event loop alone.

Claiming "no blocking calls in the loop" is cheap. This file makes it a test.

The trick is a stub index whose ``search`` calls ``time.sleep``, which blocks a
thread for real. A heartbeat coroutine ticks alongside it. If the retriever
called ``search`` inline the heartbeat would stall for the whole sleep, and if
it goes through ``asyncio.to_thread`` the heartbeat keeps ticking. The
assertion is on the heartbeat, not on the implementation, so the test still
means something after a refactor.
"""

from __future__ import annotations

import asyncio
import time

from pcagent.retrieval.hybrid import HybridRetriever

BLOCKING_SECONDS = 0.30
HEARTBEAT_INTERVAL = 0.01


class BlockingDense:
    """Synchronous, slow, and honest about it, like the real ONNX runtime."""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents

    def warm(self) -> None:
        time.sleep(BLOCKING_SECONDS)

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        time.sleep(BLOCKING_SECONDS)
        return [(0, 1.0)]


class BlockingReranker:
    def warm(self) -> None:
        time.sleep(BLOCKING_SECONDS)

    def score(self, query: str, documents: list[str]) -> list[float]:
        time.sleep(BLOCKING_SECONDS)
        return [float(len(documents) - i) for i in range(len(documents))]


async def _count_heartbeats(stop: asyncio.Event) -> int:
    ticks = 0
    while not stop.is_set():
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        ticks += 1
    return ticks


async def _ticks_during(coro) -> tuple[int, float]:
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_count_heartbeats(stop))
    started = time.perf_counter()
    await coro
    elapsed = time.perf_counter() - started
    stop.set()
    return await heartbeat, elapsed


async def test_retrieval_does_not_block_the_event_loop(catalog) -> None:
    documents = [code.index_text for code in catalog.codes]
    retriever = HybridRetriever(
        catalog, dense=BlockingDense(documents), reranker=BlockingReranker()
    )
    await retriever.warm()

    ticks, elapsed = await _ticks_during(retriever.retrieve("office furniture"))

    # Two blocking stages at 0.3s each, so the call itself takes about 0.6s.
    assert elapsed > BLOCKING_SECONDS
    # A loop that stayed responsive gets most of its ticks. A blocked one gets
    # almost none. The bound is loose because CI machines are not quiet.
    expected = elapsed / HEARTBEAT_INTERVAL
    assert ticks > expected * 0.5, (
        f"event loop stalled: {ticks} heartbeats in {elapsed:.2f}s, "
        f"expected roughly {expected:.0f}"
    )


async def test_warm_does_not_block_the_event_loop(catalog) -> None:
    """Startup counts too, because that is when health checks are waiting."""
    documents = [code.index_text for code in catalog.codes]
    retriever = HybridRetriever(
        catalog, dense=BlockingDense(documents), reranker=BlockingReranker()
    )
    ticks, elapsed = await _ticks_during(retriever.warm())
    assert ticks > (elapsed / HEARTBEAT_INTERVAL) * 0.5


async def test_two_retrievals_overlap_rather_than_queue(catalog) -> None:
    """Threads mean concurrent requests actually run concurrently.

    Two requests that each block for 0.6s should finish in well under 1.2s. If
    they serialised, the thread offload would be pointless.
    """
    documents = [code.index_text for code in catalog.codes]
    retriever = HybridRetriever(
        catalog, dense=BlockingDense(documents), reranker=BlockingReranker()
    )
    await retriever.warm()

    started = time.perf_counter()
    await asyncio.gather(
        retriever.retrieve("office furniture"), retriever.retrieve("janitorial")
    )
    elapsed = time.perf_counter() - started

    serial = BLOCKING_SECONDS * 4
    assert elapsed < serial * 0.8, f"took {elapsed:.2f}s, serial would be {serial:.2f}s"
