"""The tool contract: context, registry, and the rule about who the caller is.

Every tool in this package has the same shape::

    async def tool_name(ctx: ToolContext, params: SomeParams) -> SomeResult

``params`` is what the caller, model included, is allowed to specify. ``ctx`` is
what only the transport can supply. The split is the whole security model:

* ``params`` carries business arguments. A model can put anything it likes here
  and the worst case is a badly chosen, validated, authorised call.
* ``ctx.principal`` carries identity and scopes. It comes from an authenticated
  session. There is no argument a model can set to change it, which is why
  ``submit_review`` has no ``reviewer`` field. The reviewer is whoever the token
  says it is.

Tools are registered so the suite can assert properties across all of them at
once, for example that every one declares a scope. The registry is metadata, not
dispatch. Both transports call the functions directly, because a thin explicit
wrapper reads better than a dynamic dispatcher and gives MCP a real signature to
build its schema from.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from ..auth import Principal, Scope
from ..domain import Taxonomy
from ..llm.base import Decider
from ..retrieval.hybrid import HybridRetriever
from ..store.db import Store


@dataclass
class Services:
    """Everything the tools need in order to do their work.

    Built once at startup and shared. The indexes take seconds to warm and the
    model clients hold connection pools, so constructing these per request would
    be the single easiest way to make this service slow.
    """

    retrievers: dict[str, HybridRetriever]
    decider: Decider
    store: Store

    def retriever(self, taxonomy: Taxonomy) -> HybridRetriever:
        return self.retrievers[taxonomy]


@dataclass(frozen=True)
class ToolContext:
    """Who is calling, and what they can call into.

    Constructed by the transport after authentication. Never constructed from
    anything a model produced.
    """

    principal: Principal
    services: Services

    def require(self, scope: Scope) -> None:
        self.principal.require(scope)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    scope: Scope
    summary: str
    handler: Callable[..., Awaitable[Any]]


REGISTRY: dict[str, ToolSpec] = {}

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def tool(*, name: str, scope: Scope, summary: str) -> Callable[[F], F]:
    """Register a tool and record the scope it enforces.

    The decorator does not wrap the function or check the scope itself. That is
    deliberate. An authorisation check performed by a decorator is invisible at
    the call site and silently disappears the day someone calls the underlying
    function directly. Each tool body calls ``ctx.require(...)`` on its own first
    line, where it can be read, and the suite checks that the scope recorded here
    matches the one the body actually enforces.
    """

    def register(handler: F) -> F:
        if name in REGISTRY:
            raise RuntimeError(f"tool {name!r} is registered twice")
        REGISTRY[name] = ToolSpec(name=name, scope=scope, summary=summary, handler=handler)
        return handler

    return register
