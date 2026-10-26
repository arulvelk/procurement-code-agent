"""The idempotency protocol, including the cases that only show up under load.

Every test here corresponds to something that actually goes wrong in a system
that retries: the duplicate write, the changed body, the key shared between two
callers, the process that died holding a reservation, and two requests racing
with the same key.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from pcagent.domain import Classification, Review, fingerprint
from pcagent.errors import IdempotencyConflict, RequestInFlight
from pcagent.store.db import Proceed, Replay, Store

OPERATION = "submit_review"


def a_classification(classification_id: str = "cls_1") -> Classification:
    return Classification(
        classification_id=classification_id,
        taxonomy="psc",
        description="office chairs",
        code="7110",
        title="OFFICE FURNITURE",
        confidence=0.8,
        rationale="because",
    )


def a_review(review_id: str = "rev_1", classification_id: str = "cls_1") -> Review:
    return Review(
        review_id=review_id,
        classification_id=classification_id,
        decision="accept",
        reviewer="ana",
    )


async def reserve(store: Store, key: str, body: object, subject: str = "ana"):
    return await store.begin_idempotent(
        subject=subject,
        operation=OPERATION,
        key=key,
        request_fingerprint=fingerprint(body),
    )


async def test_first_call_proceeds_and_second_replays(store: Store) -> None:
    await store.save_classification(a_classification())
    body = {"decision": "accept"}

    assert isinstance(await reserve(store, "k1", body), Proceed)

    review = a_review()
    await store.complete_idempotent(
        subject="ana",
        operation=OPERATION,
        key="k1",
        response=review.model_dump(mode="json"),
        write=store.insert_review_statement(review),
    )

    replayed = await reserve(store, "k1", body)
    assert isinstance(replayed, Replay)
    assert replayed.response["review_id"] == "rev_1"
    # The important assertion: the replay did not write a second row.
    assert len(await store.list_reviews()) == 1


async def test_same_key_with_a_different_body_is_a_conflict(store: Store) -> None:
    await reserve(store, "k1", {"decision": "accept"})
    await store.complete_idempotent(
        subject="ana", operation=OPERATION, key="k1", response={"ok": True}
    )
    with pytest.raises(IdempotencyConflict):
        await reserve(store, "k1", {"decision": "reject"})


async def test_field_order_does_not_change_the_fingerprint(store: Store) -> None:
    await reserve(store, "k1", {"a": 1, "b": 2})
    await store.complete_idempotent(
        subject="ana", operation=OPERATION, key="k1", response={"ok": True}
    )
    replayed = await reserve(store, "k1", {"b": 2, "a": 1})
    assert isinstance(replayed, Replay)


async def test_keys_are_scoped_per_caller(store: Store) -> None:
    """Two callers both using "retry-1" are making two different requests."""
    await reserve(store, "retry-1", {"decision": "accept"}, subject="ana")
    assert isinstance(
        await reserve(store, "retry-1", {"decision": "accept"}, subject="bob"), Proceed
    )


async def test_a_second_attempt_while_the_first_is_running_is_told_to_wait(
    store: Store,
) -> None:
    await reserve(store, "k1", {"decision": "accept"})
    with pytest.raises(RequestInFlight) as caught:
        await reserve(store, "k1", {"decision": "accept"})
    assert caught.value.retryable is True


async def test_release_lets_an_honest_retry_through(store: Store) -> None:
    """A failed attempt must not wedge the key forever."""
    await reserve(store, "k1", {"decision": "accept"})
    await store.release_idempotent(subject="ana", operation=OPERATION, key="k1")
    assert await store.reservation_state(
        subject="ana", operation=OPERATION, key="k1"
    ) is None
    assert isinstance(await reserve(store, "k1", {"decision": "accept"}), Proceed)


async def test_an_abandoned_reservation_is_taken_over(tmp_path) -> None:
    """A process that dies mid write leaves a reservation nobody will finish.

    Holding the key forever would be worse than the small double write window
    that taking it over opens, so reservations expire.
    """
    store = Store(tmp_path / "stale.db", stale_reservation_seconds=0)
    await store.connect()
    try:
        await reserve(store, "k1", {"decision": "accept"})
        assert isinstance(await reserve(store, "k1", {"decision": "accept"}), Proceed)
    finally:
        await store.close()


async def test_only_one_of_many_racing_callers_proceeds(store: Store) -> None:
    """The reason BEGIN IMMEDIATE is there.

    Without the write lock taken up front, several callers can all read "no
    such key" before any of them inserts, and all of them proceed.
    """
    body = {"decision": "accept"}
    results = await asyncio.gather(
        *(reserve(store, "race", body) for _ in range(8)),
        return_exceptions=True,
    )
    proceeded = [r for r in results if isinstance(r, Proceed)]
    in_flight = [r for r in results if isinstance(r, RequestInFlight)]
    assert len(proceeded) == 1
    assert len(in_flight) == len(results) - 1


async def test_the_write_and_its_replayable_response_commit_together(
    store: Store,
) -> None:
    """A crash must not be able to split the two.

    If the insert fails, the reservation must not be marked completed, or a
    retry would replay a success for a review that was never recorded.
    """
    await store.save_classification(a_classification())
    await reserve(store, "k1", {"decision": "accept"})

    review = a_review()
    bad_write = ("INSERT INTO reviews (review_id) VALUES (?)", (review.review_id,))
    with pytest.raises(sqlite3.IntegrityError):
        await store.complete_idempotent(
            subject="ana",
            operation=OPERATION,
            key="k1",
            response=review.model_dump(mode="json"),
            write=bad_write,
        )

    assert await store.reservation_state(
        subject="ana", operation=OPERATION, key="k1"
    ) == "in_flight"
    assert await store.list_reviews() == []
