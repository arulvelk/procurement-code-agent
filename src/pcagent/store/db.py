"""SQLite persistence, and the idempotency protocol that sits on top of it.

Why idempotency lives here rather than in the tool
--------------------------------------------------
``submit_review`` is the only tool in this repo that changes anything, and it is
exactly the kind of call an agent retries. A timeout on the caller's side says
nothing about whether the write landed, so a naive retry double records a human
decision. Pushing the guarantee down into the same transaction as the write is
the only place it can actually be enforced. A check in the tool body would race.

The protocol has three steps and is the one payment APIs settled on:

1. ``begin_idempotent`` reserves the key inside a write transaction. The first
   caller wins and gets ``Proceed``. A later caller with the same key and the
   same body gets ``Replay`` carrying the original response. A later caller with
   the same key and a *different* body gets a conflict, because one of the two
   is a bug and guessing which would be worse than failing.
2. ``complete_idempotent`` stores the response in the same transaction as the
   write itself, so the record and its replayable answer commit together or not
   at all.
3. ``release_idempotent`` clears the reservation if the work failed, so an
   honest retry is not locked out by a reservation nothing will ever complete.

A process that dies between 1 and 2 leaves a reservation behind. Those are taken
over once they are older than ``stale_reservation_seconds`` rather than held
forever, which trades a small double write window for not wedging a key.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast

import aiosqlite

from ..domain import Classification, Review, utcnow
from ..errors import IdempotencyConflict, NotFound, RequestInFlight

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

DEFAULT_STALE_RESERVATION_SECONDS = 120


@dataclass(frozen=True)
class Proceed:
    """The caller holds the reservation and should do the work."""


@dataclass(frozen=True)
class Replay:
    """This exact request already completed. Return the stored response."""

    response: dict[str, Any]


Reservation = Proceed | Replay


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


class Store:
    """One SQLite connection, opened once and shared.

    SQLite takes a single writer, so a pool would buy nothing here beyond a way
    to produce "database is locked". Reads run concurrently under WAL, and the
    busy timeout covers the brief overlap while a write commits.

    One consequence of sharing a connection has to be handled explicitly. A
    transaction is a property of the connection, not of the coroutine, so two
    coroutines that both issue BEGIN IMMEDIATE on it get "cannot start a
    transaction within a transaction" rather than the serialisation they were
    expecting. ``_write_lock`` makes each transactional method an atomic unit
    in the event loop as well as in the database.

    This is not belt and braces on top of BEGIN IMMEDIATE, the two solve
    different problems. The lock serialises coroutines sharing this connection.
    BEGIN IMMEDIATE serialises this process against any other process on the
    same file. Both are needed, and a test for each is in test_idempotency.py.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stale_reservation_seconds: int = DEFAULT_STALE_RESERVATION_SECONDS,
    ) -> None:
        self.path = str(path)
        self.stale_reservation_seconds = stale_reservation_seconds
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._connection is not None:
            return
        # isolation_level=None turns off the driver's implicit transaction
        # handling. The idempotency protocol needs to say exactly when a
        # transaction starts and ends, so it issues BEGIN IMMEDIATE itself.
        connection = await aiosqlite.connect(self.path, isolation_level=None)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout = 5000")
        await connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._connection = connection

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Store.connect() has not been awaited")
        return self._connection

    async def __aenter__(self) -> Store:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ----------------------------------------------------------- records

    async def save_classification(self, classification: Classification) -> None:
        await self.db.execute(
            """
            INSERT OR REPLACE INTO classifications (
                classification_id, taxonomy, description, code, title, confidence,
                rationale, evidence_json, alternatives_json, decided_by,
                latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                classification.classification_id,
                classification.taxonomy,
                classification.description,
                classification.code,
                classification.title,
                classification.confidence,
                classification.rationale,
                json.dumps(list(classification.evidence)),
                json.dumps([c.model_dump() for c in classification.alternatives]),
                classification.decided_by,
                classification.latency_ms,
                _iso(classification.created_at),
            ),
        )

    async def get_classification(self, classification_id: str) -> Classification:
        cursor = await self.db.execute(
            "SELECT * FROM classifications WHERE classification_id = ?",
            (classification_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise NotFound(
                "no classification " + repr(classification_id),
                classification_id=classification_id,
            )
        return Classification(
            classification_id=row["classification_id"],
            taxonomy=row["taxonomy"],
            description=row["description"],
            code=row["code"],
            title=row["title"],
            confidence=row["confidence"],
            rationale=row["rationale"],
            evidence=tuple(json.loads(row["evidence_json"])),
            alternatives=tuple(json.loads(row["alternatives_json"])),
            decided_by=row["decided_by"],
            latency_ms=row["latency_ms"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def insert_review_statement(self, review: Review) -> tuple[str, tuple[Any, ...]]:
        """The review insert, as a statement the idempotency commit can carry.

        Returned rather than executed so that the write and the stored response
        land in one transaction. See ``complete_idempotent``.
        """
        return (
            """
            INSERT INTO reviews (
                review_id, classification_id, decision, reviewer,
                assigned_code, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review.review_id,
                review.classification_id,
                review.decision,
                review.reviewer,
                review.assigned_code,
                review.note,
                _iso(review.created_at),
            ),
        )

    async def list_reviews(
        self, *, classification_id: str | None = None, limit: int = 50
    ) -> list[Review]:
        if classification_id:
            cursor = await self.db.execute(
                "SELECT * FROM reviews WHERE classification_id = ?"
                " ORDER BY created_at DESC, review_id DESC LIMIT ?",
                (classification_id, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM reviews ORDER BY created_at DESC, review_id DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [
            Review(
                review_id=row["review_id"],
                classification_id=row["classification_id"],
                decision=row["decision"],
                reviewer=row["reviewer"],
                assigned_code=row["assigned_code"],
                note=row["note"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------- idempotency

    async def begin_idempotent(
        self,
        *,
        subject: str,
        operation: str,
        key: str,
        request_fingerprint: str,
    ) -> Reservation:
        """Reserve ``key``, or tell the caller what happened to it last time."""
        now = utcnow()
        stale_before = now - timedelta(seconds=self.stale_reservation_seconds)

        # BEGIN IMMEDIATE takes the database write lock up front. Without it,
        # two callers could both read "no such key" before either one inserted.
        # The asyncio lock is the same guarantee one level up, between
        # coroutines sharing this connection.
        async with self._write_lock:
            return await self._begin_locked(
                subject=subject,
                operation=operation,
                key=key,
                request_fingerprint=request_fingerprint,
                now=now,
                stale_before=stale_before,
            )

    async def _begin_locked(
        self,
        *,
        subject: str,
        operation: str,
        key: str,
        request_fingerprint: str,
        now: datetime,
        stale_before: datetime,
    ) -> Reservation:
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.db.execute(
                "SELECT request_fingerprint, state, response_json, created_at"
                " FROM idempotency_keys WHERE subject = ? AND operation = ? AND key = ?",
                (subject, operation, key),
            )
            row = await cursor.fetchone()

            if row is not None:
                if row["request_fingerprint"] != request_fingerprint:
                    raise IdempotencyConflict(
                        "idempotency key "
                        + repr(key)
                        + " was already used with a different request body",
                        key=key,
                        operation=operation,
                    )
                if row["state"] == "completed":
                    await self.db.execute("COMMIT")
                    return Replay(json.loads(row["response_json"]))

                reserved_at = datetime.fromisoformat(row["created_at"])
                if reserved_at > stale_before:
                    raise RequestInFlight(
                        "idempotency key " + repr(key) + " is already being processed",
                        key=key,
                        operation=operation,
                    )
                # Abandoned by a process that died. Take it over.
                await self.db.execute(
                    "UPDATE idempotency_keys SET created_at = ?"
                    " WHERE subject = ? AND operation = ? AND key = ?",
                    (_iso(now), subject, operation, key),
                )
                await self.db.execute("COMMIT")
                return Proceed()

            await self.db.execute(
                "INSERT INTO idempotency_keys"
                " (subject, operation, key, request_fingerprint, state,"
                "  response_json, created_at)"
                " VALUES (?, ?, ?, ?, 'in_flight', NULL, ?)",
                (subject, operation, key, request_fingerprint, _iso(now)),
            )
            await self.db.execute("COMMIT")
            return Proceed()
        except BaseException:
            await self.db.execute("ROLLBACK")
            raise

    async def complete_idempotent(
        self,
        *,
        subject: str,
        operation: str,
        key: str,
        response: dict[str, Any],
        write: tuple[str, tuple[Any, ...]] | None = None,
    ) -> None:
        """Record the response, together with the write it describes.

        Passing ``write`` puts the business row and the replayable response in
        one transaction. That is the point of the whole exercise. A crash cannot
        leave a review recorded with no way to replay it, nor a replayable
        response for a review that was never recorded.
        """
        async with self._write_lock:
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                if write is not None:
                    statement, parameters = write
                    await self.db.execute(statement, parameters)
                await self.db.execute(
                    "UPDATE idempotency_keys SET state = 'completed',"
                    " response_json = ?"
                    " WHERE subject = ? AND operation = ? AND key = ?",
                    (json.dumps(response), subject, operation, key),
                )
                await self.db.execute("COMMIT")
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise

    async def release_idempotent(self, *, subject: str, operation: str, key: str) -> None:
        """Drop a reservation whose work failed, so a retry is not locked out."""
        await self.db.execute(
            "DELETE FROM idempotency_keys"
            " WHERE subject = ? AND operation = ? AND key = ? AND state = 'in_flight'",
            (subject, operation, key),
        )

    async def reservation_state(
        self, *, subject: str, operation: str, key: str
    ) -> Literal["in_flight", "completed"] | None:
        cursor = await self.db.execute(
            "SELECT state FROM idempotency_keys"
            " WHERE subject = ? AND operation = ? AND key = ?",
            (subject, operation, key),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return cast("Literal['in_flight', 'completed']", row["state"])
