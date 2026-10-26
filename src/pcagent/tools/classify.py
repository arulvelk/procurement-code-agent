"""Classify a purchase description into one procurement code.

Pipeline: retrieve, rerank, decide, persist.

On the identifier
-----------------
``classification_id`` is derived from the taxonomy and the normalised
description rather than generated at random. A classification is a function of
its input, so the same text asked twice is the same classification, and the
record is updated rather than duplicated. That also means a review recorded
against an id stays attached to the thing a human actually looked at: the
purchase description.

The cost of the choice is that re-running with a better model overwrites the
earlier verdict. That is the right default for a queue whose job is to hold the
current best answer, and the wrong one for an audit log of every run. Which it
should be is a product decision, and this repo makes the one it can defend in a
README rather than pretending the question does not exist.
"""

from __future__ import annotations

import hashlib
import time

from pydantic import BaseModel, ConfigDict, Field

from ..auth import Scope
from ..domain import Classification, Taxonomy
from ..errors import ValidationFailed
from ..llm.base import DecisionRequest
from .base import ToolContext, tool


class ClassifyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=3, max_length=4000)
    taxonomy: Taxonomy = "psc"
    persist: bool = Field(
        default=True,
        description="Store the result so it can be reviewed later.",
    )


def classification_id_for(taxonomy: Taxonomy, description: str) -> str:
    normalised = " ".join(description.split()).casefold()
    digest = hashlib.sha256(f"{taxonomy}:{normalised}".encode()).hexdigest()
    return f"cls_{digest[:16]}"


@tool(
    name="classify_purchase",
    scope=Scope.CLASSIFY_RUN,
    summary="Assign a procurement code to a free text purchase description.",
)
async def classify_purchase(ctx: ToolContext, params: ClassifyParams) -> Classification:
    ctx.require(Scope.CLASSIFY_RUN)

    description = " ".join(params.description.split())
    if not description:
        raise ValidationFailed("description is blank once whitespace is stripped")

    started = time.perf_counter()
    retriever = ctx.services.retriever(params.taxonomy)
    outcome = await retriever.retrieve(description)

    if not outcome.candidates:
        raise ValidationFailed(
            "retrieval found no candidate codes for this description",
            taxonomy=params.taxonomy,
        )

    decision = await ctx.services.decider.decide(
        DecisionRequest(
            description=description,
            taxonomy=params.taxonomy,
            candidates=tuple(outcome.candidates),
        )
    )

    chosen = next(
        (c for c in outcome.candidates if c.code.upper() == decision.code.upper()),
        None,
    )
    if chosen is None:
        # The decider is contractually required to pick from the list it was
        # given. Reaching here means a bug in a decider, not a bad input, so it
        # fails loudly rather than inventing a title for an unknown code.
        raise ValidationFailed(
            "decider returned a code that was not among the candidates",
            returned=decision.code,
        )

    classification = Classification(
        classification_id=classification_id_for(params.taxonomy, description),
        taxonomy=params.taxonomy,
        description=description,
        code=chosen.code,
        title=chosen.title,
        confidence=decision.confidence,
        rationale=decision.rationale,
        evidence=decision.evidence,
        # The runners up are kept on the record. A reviewer rejecting the top
        # answer almost always wants the second one, and making them search
        # again for something the system already computed is wasteful.
        alternatives=tuple(c for c in outcome.candidates if c.code != chosen.code)[:4],
        decided_by=decision.decided_by,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )

    if params.persist:
        await ctx.services.store.save_classification(classification)

    return classification
