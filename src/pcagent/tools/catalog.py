"""Read only tools over the taxonomy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..auth import Scope
from ..domain import Candidate, Taxonomy, TaxonomyCode
from .base import ToolContext, tool


class SearchCodesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=2000)
    taxonomy: Taxonomy = "psc"
    limit: int = Field(default=8, ge=1, le=25)
    rerank: bool = Field(
        default=True,
        description="Run the cross encoder. Turning it off is faster and worse.",
    )


class SearchCodesResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    taxonomy: Taxonomy
    query: str
    candidates: tuple[Candidate, ...]
    used_dense: bool
    used_rerank: bool
    notes: tuple[str, ...] = ()


@tool(
    name="search_codes",
    scope=Scope.CODES_READ,
    summary="Find candidate procurement codes for a free text query.",
)
async def search_codes(ctx: ToolContext, params: SearchCodesParams) -> SearchCodesResult:
    ctx.require(Scope.CODES_READ)

    retriever = ctx.services.retriever(params.taxonomy)
    outcome = await retriever.retrieve(
        params.query, limit=params.limit, rerank=params.rerank
    )
    return SearchCodesResult(
        taxonomy=params.taxonomy,
        query=params.query,
        candidates=tuple(outcome.candidates),
        used_dense=outcome.used_dense,
        used_rerank=outcome.used_rerank,
        notes=tuple(outcome.notes),
    )


class GetCodeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=2, max_length=10)
    taxonomy: Taxonomy = "psc"


@tool(
    name="get_code",
    scope=Scope.CODES_READ,
    summary="Look up one procurement code and its place in the hierarchy.",
)
async def get_code(ctx: ToolContext, params: GetCodeParams) -> TaxonomyCode:
    ctx.require(Scope.CODES_READ)

    # Raises NotFound, which each transport turns into its own 404 shaped
    # answer. An agent that asked for a code that does not exist needs to be
    # told that plainly rather than handed an empty result it might read as
    # "no information available".
    return ctx.services.retriever(params.taxonomy).catalog.get(params.code)
