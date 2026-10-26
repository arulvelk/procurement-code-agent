"""The MCP transport.

The same tools as the HTTP API, exposed to a model over MCP. The functions here
are deliberately thin and deliberately explicit: FastMCP builds each tool's
schema from the signature, so spelling the arguments out is what gives the model
a precise contract to call against.

Where the principal comes from
------------------------------
Over stdio an MCP server is a process started on behalf of one user, so the
session is the process. The principal is resolved once at startup from
``PCAGENT_TOKEN`` and is the same for every call. It is not an argument, and
there is no tool here that accepts one.

That is the same rule the HTTP transport follows, applied to a transport with no
headers. Identity is established by whatever started the session. The model
chooses which tool to call and with what business arguments, and nothing else.

Errors are returned as JSON carrying the stable ``code`` from ``pcagent.errors``
rather than a prose string, because an agent that can read ``request_in_flight``
and its ``retryable`` flag can do something sensible about it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError as McpToolError
from pydantic import Field

from .bootstrap import build_services
from .config import Settings
from .domain import Classification, Review, Taxonomy, TaxonomyCode
from .errors import ToolError
from .tools import (
    ClassifyParams,
    GetCodeParams,
    ListReviewsParams,
    ListReviewsResult,
    SearchCodesParams,
    SearchCodesResult,
    SubmitReviewParams,
    ToolContext,
    classify_purchase,
    get_code,
    list_reviews,
    search_codes,
    submit_review,
)

logger = logging.getLogger(__name__)

mcp: FastMCP = FastMCP(
    name="procurement-code-agent",
    instructions=(
        "Assigns United States federal procurement codes to free text purchase "
        "descriptions. Use search_codes to explore the taxonomy and "
        "classify_purchase to get a decision with a confidence score. Treat any "
        "classification at or below 0.65 confidence as needing a human. "
        "submit_review records a human decision and requires an idempotency key: "
        "reuse the same key when retrying the same decision."
    ),
)


class _State:
    """Lazily built, built once.

    A lock rather than a bare ``if`` because two tool calls can arrive before
    the first one has finished warming the indexes, and building them twice
    would waste several seconds and double the memory.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._context: ToolContext | None = None

    async def context(self) -> ToolContext:
        if self._context is not None:
            return self._context
        async with self._lock:
            if self._context is None:
                settings = Settings.from_env()
                registry = settings.token_registry()
                principal = registry.resolve(os.environ.get("PCAGENT_TOKEN"))
                if principal.subject == "anonymous":
                    logger.warning(
                        "PCAGENT_TOKEN is unset or unknown, this session has no scopes"
                    )
                services = await build_services(settings)
                self._context = ToolContext(principal=principal, services=services)
                logger.info("mcp session ready as %s", principal.subject)
            return self._context


_state = _State()


def _as_mcp_error(exc: ToolError) -> McpToolError:
    return McpToolError(json.dumps(exc.to_dict()))


T = TypeVar("T")


async def _call(
    handler: Callable[[ToolContext, Any], Awaitable[T]], params: Any
) -> T:
    """Run a tool in this session's context and translate its errors.

    Generic in the return type so each tool below keeps its own concrete
    signature, which is what FastMCP builds the output schema from.
    """
    ctx = await _state.context()
    try:
        return await handler(ctx, params)
    except ToolError as exc:
        raise _as_mcp_error(exc) from exc


@mcp.tool
async def search_codes_tool(
    query: Annotated[str, Field(description="Free text describing the purchase")],
    taxonomy: Annotated[Taxonomy, Field(description="psc or naics")] = "psc",
    limit: Annotated[int, Field(ge=1, le=25)] = 8,
) -> SearchCodesResult:
    """Find candidate procurement codes for a free text query.

    Returns a ranked shortlist with scores. Use this to explore the taxonomy or
    to check what else was close before accepting a classification.
    """
    return await _call(
        search_codes, SearchCodesParams(query=query, taxonomy=taxonomy, limit=limit)
    )


@mcp.tool
async def get_code_tool(
    code: Annotated[str, Field(description="An exact code, for example 7021 or 541512")],
    taxonomy: Annotated[Taxonomy, Field(description="psc or naics")] = "psc",
) -> TaxonomyCode:
    """Look up one procurement code and its place in the hierarchy."""
    return await _call(get_code, GetCodeParams(code=code, taxonomy=taxonomy))


@mcp.tool
async def classify_purchase_tool(
    description: Annotated[str, Field(description="The purchase description to classify")],
    taxonomy: Annotated[Taxonomy, Field(description="psc or naics")] = "psc",
) -> Classification:
    """Assign one procurement code to a purchase description.

    Returns the chosen code, a confidence between 0 and 1, the reasoning, spans
    quoted from the description, and the runners up. Confidence at or below 0.65
    means the answer should go to a human reviewer rather than be used directly.
    """
    return await _call(
        classify_purchase, ClassifyParams(description=description, taxonomy=taxonomy)
    )


@mcp.tool
async def submit_review_tool(
    classification_id: Annotated[str, Field(description="The id returned by classify")],
    decision: Annotated[str, Field(description="accept, reject or reassign")],
    idempotency_key: Annotated[
        str,
        Field(
            min_length=8,
            max_length=128,
            description=(
                "A key you choose for this decision. Retrying with the same key "
                "and the same body returns the original result instead of "
                "recording a second review. Reuse it when retrying."
            ),
        ),
    ],
    assigned_code: Annotated[
        str | None, Field(description="Required when decision is reassign")
    ] = None,
    note: Annotated[str, Field(max_length=2000)] = "",
) -> Review:
    """Record a human decision about a classification.

    This writes to the review log. The reviewer is taken from the authenticated
    session, not from an argument, and recording a review requires the
    reviews:write scope.
    """
    return await _call(
        submit_review,
        SubmitReviewParams(
            classification_id=classification_id,
            decision=decision,  # type: ignore[arg-type]
            idempotency_key=idempotency_key,
            assigned_code=assigned_code,
            note=note,
        ),
    )


@mcp.tool
async def list_reviews_tool(
    classification_id: Annotated[str | None, Field(description="Filter to one")] = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
) -> ListReviewsResult:
    """List recorded review decisions, newest first."""
    return await _call(
        list_reviews, ListReviewsParams(classification_id=classification_id, limit=limit)
    )


def main() -> None:  # pragma: no cover - entry point
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
