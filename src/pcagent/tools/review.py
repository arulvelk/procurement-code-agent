"""The human review queue. The only tools here that change anything.

``submit_review`` is where most of this repo's opinions show up at once:

* The reviewer is not a parameter. It is ``ctx.principal.subject``. A model
  cannot record a decision in someone else's name because there is nowhere to
  put the name.
* The scope check is the first line of the body, not a decorator and not a
  sentence in a prompt.
* Cross field rules are enforced in the model, so "reassign without a code"
  cannot reach the database.
* The write is idempotent on a caller supplied key, and the key covers the
  request body, so a retry is free and a changed body is an error.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..auth import Scope
from ..domain import Review, ReviewDecision, fingerprint, utcnow
from ..errors import ValidationFailed
from ..store.db import Replay
from .base import ToolContext, tool

OPERATION = "submit_review"


class SubmitReviewParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification_id: str = Field(min_length=4, max_length=64)
    decision: ReviewDecision
    assigned_code: str | None = Field(
        default=None,
        description="Required when decision is reassign, forbidden otherwise.",
    )
    note: str = Field(default="", max_length=2000)
    idempotency_key: str = Field(
        min_length=8,
        max_length=128,
        description=(
            "Caller supplied. Retrying with the same key and the same body "
            "returns the original result instead of recording a second review."
        ),
    )

    @model_validator(mode="after")
    def _check_assigned_code(self) -> SubmitReviewParams:
        # Enforced here rather than in the tool body so that every transport
        # gets the rule for free, including MCP, whose schema the model reads.
        if self.decision == "reassign" and not self.assigned_code:
            raise ValueError("assigned_code is required when decision is reassign")
        if self.decision != "reassign" and self.assigned_code:
            raise ValueError("assigned_code is only valid when decision is reassign")
        return self


def _review_id(subject: str, key: str) -> str:
    """Derive the id from the caller and their key.

    Deterministic so that a replayed request returns the identifier the first
    attempt returned. A random id would mean the retry of a request that
    actually succeeded hands back something different from what was stored.
    """
    digest = hashlib.sha256(f"{subject}:{OPERATION}:{key}".encode()).hexdigest()
    return f"rev_{digest[:16]}"


@tool(
    name="submit_review",
    scope=Scope.REVIEWS_WRITE,
    summary="Record a human decision about a classification. Idempotent on a caller supplied key.",
)
async def submit_review(ctx: ToolContext, params: SubmitReviewParams) -> Review:
    ctx.require(Scope.REVIEWS_WRITE)

    store = ctx.services.store
    subject = ctx.principal.subject

    # Validate before reserving the key. A request that was never going to
    # succeed should not burn an idempotency key, or the caller's retry with a
    # corrected body would come back as a conflict.
    classification = await store.get_classification(params.classification_id)

    if params.assigned_code:
        catalog = ctx.services.retriever(classification.taxonomy).catalog
        if not catalog.contains(params.assigned_code):
            raise ValidationFailed(
                f"{params.assigned_code!r} is not a {classification.taxonomy} code",
                assigned_code=params.assigned_code,
                taxonomy=classification.taxonomy,
            )

    # The fingerprint covers the whole request but not the key itself. Reusing
    # a key with a different body is the conflict we want to catch.
    request_fingerprint = fingerprint(
        {
            "classification_id": params.classification_id,
            "decision": params.decision,
            "assigned_code": params.assigned_code,
            "note": params.note,
        }
    )

    reservation = await store.begin_idempotent(
        subject=subject,
        operation=OPERATION,
        key=params.idempotency_key,
        request_fingerprint=request_fingerprint,
    )
    if isinstance(reservation, Replay):
        return Review.model_validate(reservation.response)

    review = Review(
        review_id=_review_id(subject, params.idempotency_key),
        classification_id=params.classification_id,
        decision=params.decision,
        reviewer=subject,
        assigned_code=params.assigned_code,
        note=params.note,
        created_at=utcnow(),
    )

    try:
        await store.complete_idempotent(
            subject=subject,
            operation=OPERATION,
            key=params.idempotency_key,
            response=review.model_dump(mode="json"),
            write=store.insert_review_statement(review),
        )
    except BaseException:
        # Hand the key back so an honest retry is not locked out by a
        # reservation that will never be completed.
        await store.release_idempotent(
            subject=subject, operation=OPERATION, key=params.idempotency_key
        )
        raise

    return review


class ListReviewsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification_id: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class ListReviewsResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    reviews: tuple[Review, ...]


@tool(
    name="list_reviews",
    scope=Scope.REVIEWS_READ,
    summary="List recorded review decisions, newest first.",
)
async def list_reviews(ctx: ToolContext, params: ListReviewsParams) -> ListReviewsResult:
    ctx.require(Scope.REVIEWS_READ)

    reviews = await ctx.services.store.list_reviews(
        classification_id=params.classification_id, limit=params.limit
    )
    return ListReviewsResult(reviews=tuple(reviews))
