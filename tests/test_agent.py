"""The batch agent: concurrency limits, timeouts, and failure isolation.

These are the three things a batch agent exists to get right, so they are the
three things tested. Throughput is not tested, because throughput is a property
of the machine and asserting on it produces a flaky suite.
"""

from __future__ import annotations

import asyncio

from pcagent.agent.batch import BatchTriageAgent, TriageSettings
from pcagent.tools import ListReviewsParams, classify_purchase, list_reviews

DESCRIPTIONS = [
    "office furniture for the regional building",
    "janitorial and custodial services",
    "desktop computers and monitors",
    "trash and garbage collection",
    "cabinets and shelving units",
    "guns through 30mm",
]


async def test_results_come_back_in_input_order(reader) -> None:
    report = await BatchTriageAgent().run(reader, DESCRIPTIONS)
    assert [r.description for r in report.results] == DESCRIPTIONS
    assert report.classified == len(DESCRIPTIONS)
    assert report.failed == 0


async def test_concurrency_never_exceeds_the_limit(reader, monkeypatch) -> None:
    """The semaphore is the whole point of the class.

    Without it, a ten thousand row batch opens ten thousand concurrent calls,
    and what falls over is usually the service being called, not this one.
    """
    limit = 2
    in_flight = 0
    high_water = 0
    original = classify_purchase

    async def counting_classify(ctx, params):
        nonlocal in_flight, high_water
        in_flight += 1
        high_water = max(high_water, in_flight)
        try:
            await asyncio.sleep(0.02)
            return await original(ctx, params)
        finally:
            in_flight -= 1

    monkeypatch.setattr("pcagent.agent.batch.classify_purchase", counting_classify)

    agent = BatchTriageAgent(TriageSettings(concurrency=limit))
    await agent.run(reader, DESCRIPTIONS)
    assert high_water <= limit
    assert high_water > 1, "nothing ran concurrently, the test proved nothing"


async def test_one_slow_item_does_not_hold_the_batch_open(reader, monkeypatch) -> None:
    original = classify_purchase

    async def sometimes_hangs(ctx, params):
        if "janitorial" in params.description:
            await asyncio.sleep(10)
        return await original(ctx, params)

    monkeypatch.setattr("pcagent.agent.batch.classify_purchase", sometimes_hangs)

    agent = BatchTriageAgent(
        TriageSettings(concurrency=4, item_timeout_seconds=0.05)
    )
    report = await agent.run(reader, DESCRIPTIONS)

    timed_out = [r for r in report.results if r.error_code == "timeout"]
    assert len(timed_out) == 1
    assert "janitorial" in timed_out[0].description
    assert report.classified == len(DESCRIPTIONS) - 1
    assert report.elapsed_seconds < 5


async def test_one_failing_item_does_not_cancel_the_others(reader, monkeypatch) -> None:
    original = classify_purchase

    async def sometimes_explodes(ctx, params):
        if "guns" in params.description:
            raise RuntimeError("upstream fell over")
        return await original(ctx, params)

    monkeypatch.setattr("pcagent.agent.batch.classify_purchase", sometimes_explodes)

    report = await BatchTriageAgent().run(reader, DESCRIPTIONS)
    assert report.failed == 1
    assert report.classified == len(DESCRIPTIONS) - 1
    failure = next(r for r in report.results if r.error)
    assert failure.error_code == "unexpected_error"


async def test_low_confidence_results_are_routed_to_review(reader) -> None:
    agent = BatchTriageAgent(TriageSettings(review_threshold=1.1))
    report = await agent.run(reader, DESCRIPTIONS)
    assert report.needs_review == len(DESCRIPTIONS)


async def test_auto_accept_requires_the_write_scope(reader) -> None:
    """The agent does not get to decide it is allowed to auto accept.

    The tool layer refuses, and the refusal is a ToolError the batch records
    per item rather than a crash.
    """
    agent = BatchTriageAgent(
        TriageSettings(auto_accept_above=0.0, concurrency=2)
    )
    report = await agent.run(reader, DESCRIPTIONS[:2])
    assert report.auto_accepted == 0
    assert all(r.error_code == "not_authorized" for r in report.results)


async def test_rerunning_a_batch_does_not_double_record(reviewer) -> None:
    """The reason the idempotency key is derived from the classification.

    Re-running a half finished batch is the normal way to recover from a
    crash, so it has to be safe.
    """
    agent = BatchTriageAgent(TriageSettings(auto_accept_above=0.0, concurrency=3))

    first = await agent.run(reviewer, DESCRIPTIONS)
    assert first.auto_accepted == len(DESCRIPTIONS)
    after_first = len((await list_reviews(reviewer, ListReviewsParams(limit=100))).reviews)

    second = await agent.run(reviewer, DESCRIPTIONS)
    after_second = len((await list_reviews(reviewer, ListReviewsParams(limit=100))).reviews)

    assert second.failed == 0
    assert after_second == after_first == len(DESCRIPTIONS)


async def test_summary_reads_sensibly(reader) -> None:
    report = await BatchTriageAgent().run(reader, DESCRIPTIONS[:2])
    assert "2 items" in report.summary()


async def test_empty_batch_is_not_an_error(reader) -> None:
    report = await BatchTriageAgent().run(reader, [])
    assert report.results == []
    assert report.classified == 0
