"""A batch triage agent, written out rather than imported.

What it does: classify a list of purchase descriptions, let the confident ones
through, and route the rest to a human.

Why the loop is hand written
----------------------------
The interesting decisions in a batch agent are all about restraint, and a
framework that hides them is hiding the part worth reviewing:

* **Bounded concurrency.** A semaphore caps in flight work. Without one,
  ``gather`` over ten thousand rows opens ten thousand concurrent calls, and the
  thing that falls over is usually not this process but whatever it is calling.
* **A deadline on every item.** ``asyncio.wait_for`` means one pathological row
  cannot hold the batch open forever. A hung item is a failed item.
* **Isolated failures.** One row raising must not cancel the other nine
  thousand. Each item is caught individually and reported.
* **Ordered output.** Results come back in input order regardless of completion
  order, because a caller matching rows by position should not have to care how
  the scheduler felt that day.

The optional auto accept path is where idempotency earns its keep. The key is
derived from the classification id, so running the same batch twice records each
decision exactly once. Re-running a half finished batch is the normal way to
recover from a crash, and it has to be safe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from ..domain import Classification, Taxonomy
from ..errors import ToolError
from ..tools.base import ToolContext
from ..tools.classify import ClassifyParams, classify_purchase
from ..tools.review import SubmitReviewParams, submit_review

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriageSettings:
    concurrency: int = 4
    review_threshold: float = 0.65
    """At or below this confidence, a human looks at it."""

    item_timeout_seconds: float = 45.0

    auto_accept_above: float | None = None
    """When set, accept anything above this confidence without a human. Off by
    default. Turning it on requires the caller to hold reviews:write, and the
    tool layer enforces that rather than this agent."""


@dataclass(frozen=True)
class TriageResult:
    index: int
    description: str
    classification: Classification | None = None
    error: str | None = None
    error_code: str | None = None
    needs_review: bool = False
    auto_accepted: bool = False


@dataclass
class TriageReport:
    results: list[TriageResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def classified(self) -> int:
        return sum(1 for r in self.results if r.classification is not None)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def needs_review(self) -> int:
        return sum(1 for r in self.results if r.needs_review)

    @property
    def auto_accepted(self) -> int:
        return sum(1 for r in self.results if r.auto_accepted)

    def summary(self) -> str:
        return (
            f"{len(self.results)} items in {self.elapsed_seconds:.1f}s: "
            f"{self.classified} classified, {self.failed} failed, "
            f"{self.needs_review} routed to review, {self.auto_accepted} auto accepted"
        )


class BatchTriageAgent:
    def __init__(self, settings: TriageSettings | None = None) -> None:
        self.settings = settings or TriageSettings()

    async def run(
        self,
        ctx: ToolContext,
        descriptions: list[str],
        *,
        taxonomy: Taxonomy = "psc",
    ) -> TriageReport:
        settings = self.settings
        limiter = asyncio.Semaphore(settings.concurrency)
        started = time.perf_counter()

        async def handle(index: int, description: str) -> TriageResult:
            # Acquire before starting the clock on the item. Time spent queued
            # behind the semaphore is not the item's fault, and counting it
            # would make the timeout depend on how busy the batch is.
            async with limiter:
                try:
                    return await asyncio.wait_for(
                        self._process(ctx, index, description, taxonomy),
                        timeout=settings.item_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning("item %d timed out", index)
                    return TriageResult(
                        index=index,
                        description=description,
                        error=f"timed out after {settings.item_timeout_seconds}s",
                        error_code="timeout",
                    )
                except ToolError as exc:
                    return TriageResult(
                        index=index,
                        description=description,
                        error=exc.message,
                        error_code=exc.code,
                    )
                except Exception as exc:
                    logger.exception("item %d failed", index)
                    return TriageResult(
                        index=index,
                        description=description,
                        error=str(exc),
                        error_code="unexpected_error",
                    )

        results = await asyncio.gather(
            *(handle(index, text) for index, text in enumerate(descriptions))
        )

        return TriageReport(
            # gather already preserves input order, but sorting makes that a
            # guarantee of this function rather than a detail of gather.
            results=sorted(results, key=lambda r: r.index),
            elapsed_seconds=time.perf_counter() - started,
        )

    async def _process(
        self,
        ctx: ToolContext,
        index: int,
        description: str,
        taxonomy: Taxonomy,
    ) -> TriageResult:
        settings = self.settings
        classification = await classify_purchase(
            ctx, ClassifyParams(description=description, taxonomy=taxonomy)
        )

        threshold = settings.auto_accept_above
        if threshold is not None and classification.confidence > threshold:
            await submit_review(
                ctx,
                SubmitReviewParams(
                    classification_id=classification.classification_id,
                    decision="accept",
                    note=(
                        f"Auto accepted at confidence {classification.confidence:.2f}, "
                        f"above the {threshold:.2f} threshold."
                    ),
                    # Derived from the classification, so re-running the batch
                    # replays instead of recording a second acceptance.
                    idempotency_key=f"auto-accept-{classification.classification_id}",
                ),
            )
            return TriageResult(
                index=index,
                description=description,
                classification=classification,
                auto_accepted=True,
            )

        return TriageResult(
            index=index,
            description=description,
            classification=classification,
            needs_review=classification.confidence <= settings.review_threshold,
        )
