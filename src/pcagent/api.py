"""The HTTP transport.

A thin shell over the tools in ``pcagent.tools``. Its entire job is to turn an
HTTP request into a ``ToolContext`` plus a params model, and to turn a
``ToolError`` back into a status code. No business rule lives here, which is why
the MCP server next door can offer the same capabilities without either of them
drifting from the other.

One difference between the transports is deliberate. Over HTTP the idempotency
key arrives in the ``Idempotency-Key`` header, which is where an HTTP client
expects to put it. Over MCP it is a tool argument, because MCP has no headers.
Same guarantee, each transport spelling it the way its callers already speak.

A note on the dependencies below: they are module level, not nested inside
``create_app``. With ``from __future__ import annotations`` every annotation is
a string that FastAPI resolves against module globals, so a type alias defined
inside the factory resolves to nothing and the parameter silently degrades into
a query string field.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from .auth import Principal, TokenRegistry
from .bootstrap import build_services
from .config import Settings
from .domain import Classification, Review, ReviewDecision, Taxonomy, TaxonomyCode
from .errors import ToolError
from .tools import (
    ClassifyParams,
    GetCodeParams,
    ListReviewsParams,
    ListReviewsResult,
    SearchCodesParams,
    SearchCodesResult,
    Services,
    SubmitReviewParams,
    ToolContext,
    classify_purchase,
    get_code,
    list_reviews,
    search_codes,
    submit_review,
)

logger = logging.getLogger(__name__)


class ReviewBody(BaseModel):
    """The review request body, minus the key, which travels in the header.

    Kept separate from ``SubmitReviewParams`` rather than subclassing it,
    because the two differ on exactly the field that cannot be defaulted. The
    params model is still what validates: this body is turned into one, and its
    cross field rules run there.
    """

    model_config = ConfigDict(extra="forbid")

    classification_id: str
    decision: ReviewDecision
    assigned_code: str | None = None
    note: str = ""


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def get_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the bearer token into a principal.

    Returns anonymous for a missing or unknown token rather than raising. The
    tool decides what anonymous may do, so an endpoint never has to remember to
    check, and a new endpoint cannot forget to.
    """
    registry: TokenRegistry = request.app.state.tokens
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    return registry.resolve(token)


def get_context(
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[Services, Depends(get_services)],
) -> ToolContext:
    return ToolContext(principal=principal, services=services)


Ctx = Annotated[ToolContext, Depends(get_context)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = getattr(app.state, "settings", None) or Settings.from_env()
    app.state.settings = settings
    app.state.tokens = settings.token_registry()
    # Warming here rather than on first request means the readiness probe does
    # not pass until the service can actually answer.
    app.state.services = await build_services(settings)
    logger.info("ready, decider=%s", settings.decider)
    try:
        yield
    finally:
        await app.state.services.store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Procurement Code Agent",
        version="0.1.0",
        summary="Classify purchase descriptions into US federal procurement codes.",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings

    @app.exception_handler(ToolError)
    async def tool_error_handler(request: Request, exc: ToolError) -> JSONResponse:
        # One mapping, in one place. Tools raise domain errors and never know
        # that HTTP exists.
        if exc.http_status >= 500:
            logger.error("%s on %s: %s", exc.code, request.url.path, exc.message)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Params models built inside a handler raise pydantic, not ToolError.

        Without this the cross field rule on reviews would surface as a 500,
        which would be a lie about whose fault it is.
        """
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_failed",
                "message": "request did not satisfy the tool contract",
                "retryable": False,
                # include_context drops the original exception object, which
                # pydantic keeps for custom validators and which json cannot
                # serialise. include_input drops request values, which keeps
                # user supplied content out of the error body.
                "details": {
                    "errors": exc.errors(
                        include_url=False, include_context=False, include_input=False
                    )
                },
            },
        )

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/search", response_model=SearchCodesResult, tags=["codes"])
    async def http_search(ctx: Ctx, params: SearchCodesParams) -> SearchCodesResult:
        return await search_codes(ctx, params)

    @app.get("/v1/codes/{taxonomy}/{code}", response_model=TaxonomyCode, tags=["codes"])
    async def http_get_code(ctx: Ctx, taxonomy: Taxonomy, code: str) -> TaxonomyCode:
        return await get_code(ctx, GetCodeParams(taxonomy=taxonomy, code=code))

    @app.post("/v1/classify", response_model=Classification, tags=["classify"])
    async def http_classify(ctx: Ctx, params: ClassifyParams) -> Classification:
        return await classify_purchase(ctx, params)

    @app.post("/v1/reviews", response_model=Review, status_code=201, tags=["reviews"])
    async def http_submit_review(
        ctx: Ctx,
        body: ReviewBody,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    ) -> Review:
        return await submit_review(
            ctx,
            SubmitReviewParams(
                classification_id=body.classification_id,
                decision=body.decision,
                assigned_code=body.assigned_code,
                note=body.note,
                idempotency_key=idempotency_key,
            ),
        )

    @app.get("/v1/reviews", response_model=ListReviewsResult, tags=["reviews"])
    async def http_list_reviews(
        ctx: Ctx,
        classification_id: str | None = None,
        limit: int = 20,
    ) -> ListReviewsResult:
        return await list_reviews(
            ctx, ListReviewsParams(classification_id=classification_id, limit=limit)
        )

    return app


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
