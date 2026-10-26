"""Authorization, and the property that matters most: it cannot be bypassed.

The individual scope checks are easy to test. The interesting test is the last
one in this file, which walks the whole tool registry and proves that every
tool actually enforces the scope it advertises. A new tool added without a
check fails that test on the day it is written, which is the only time anyone
will remember why it mattered.
"""

from __future__ import annotations

import pytest

from pcagent.auth import ANONYMOUS, READ_ONLY, REVIEWER, Principal, Scope, TokenRegistry
from pcagent.errors import NotAuthorized
from pcagent.tools import REGISTRY, ToolContext
from pcagent.tools.catalog import GetCodeParams, SearchCodesParams
from pcagent.tools.classify import ClassifyParams
from pcagent.tools.review import ListReviewsParams, SubmitReviewParams


def test_principal_requires_exact_scope() -> None:
    principal = Principal("svc", frozenset({Scope.CODES_READ}))
    principal.require(Scope.CODES_READ)
    with pytest.raises(NotAuthorized) as caught:
        principal.require(Scope.REVIEWS_WRITE)
    assert caught.value.details["required_scope"] == "reviews:write"
    assert caught.value.http_status == 403


def test_reader_cannot_write_reviews() -> None:
    assert Scope.REVIEWS_WRITE not in READ_ONLY
    assert Scope.REVIEWS_WRITE in REVIEWER


def test_unknown_token_is_anonymous_not_an_error() -> None:
    """Authentication and authorization stay separate.

    An unknown token produces a principal with no scopes, and the tool decides
    what that means. Raising here instead would push the decision into every
    transport, where one of them would eventually forget.
    """
    registry = TokenRegistry({"good": Principal("svc", READ_ONLY)})
    assert registry.resolve("good").subject == "svc"
    assert registry.resolve("bad") is ANONYMOUS
    assert registry.resolve(None) is ANONYMOUS
    assert registry.resolve("") is ANONYMOUS


def test_registry_holds_no_plaintext_tokens() -> None:
    registry = TokenRegistry({"sekrit-token": Principal("svc", READ_ONLY)})
    stored = repr(registry.__dict__)
    assert "sekrit-token" not in stored


def test_anonymous_is_rejected_by_the_cheapest_tool() -> None:
    with pytest.raises(NotAuthorized):
        ANONYMOUS.require(Scope.CODES_READ)


# One minimally valid params object per registered tool. Adding a tool without
# adding it here fails test_every_tool_enforces_its_declared_scope, which is
# the intended nudge.
PARAMS_FOR_TOOL = {
    "search_codes": lambda: SearchCodesParams(query="office chairs"),
    "get_code": lambda: GetCodeParams(code="7021"),
    "classify_purchase": lambda: ClassifyParams(description="office chairs"),
    "submit_review": lambda: SubmitReviewParams(
        classification_id="cls_whatever",
        decision="accept",
        idempotency_key="key-00000001",
    ),
    "list_reviews": lambda: ListReviewsParams(),
}


def test_every_tool_declares_a_scope() -> None:
    assert REGISTRY, "no tools registered, the package did not import"
    for name, spec in REGISTRY.items():
        assert isinstance(spec.scope, Scope), f"{name} has no scope"
        assert spec.summary, f"{name} has no summary"


def test_registry_and_params_table_agree() -> None:
    assert set(REGISTRY) == set(PARAMS_FOR_TOOL)


@pytest.mark.parametrize("tool_name", sorted(PARAMS_FOR_TOOL))
async def test_every_tool_enforces_its_declared_scope(
    tool_name: str, services, all_scopes
) -> None:
    """Hold every scope except the declared one, and the call must still fail.

    This is what stops the declaration in the decorator from drifting away from
    the check in the body. A tool that forgets ``ctx.require`` passes every
    other test in this suite and fails this one.
    """
    spec = REGISTRY[tool_name]
    missing_one = frozenset(all_scopes) - {spec.scope}
    ctx = ToolContext(Principal("almost-everything", missing_one), services)

    with pytest.raises(NotAuthorized) as caught:
        await spec.handler(ctx, PARAMS_FOR_TOOL[tool_name]())

    assert caught.value.details["required_scope"] == spec.scope.value


@pytest.mark.parametrize("tool_name", sorted(PARAMS_FOR_TOOL))
async def test_no_tool_accepts_an_identity_argument(tool_name: str) -> None:
    """The caller must not be able to name itself.

    If a tool ever grows a ``reviewer`` or ``subject`` or ``scopes`` field, a
    model can set it, and the authorization model is over. The reviewer is read
    from the authenticated principal, so the field must not exist.
    """
    params = PARAMS_FOR_TOOL[tool_name]()
    forbidden = {"reviewer", "subject", "principal", "scopes", "user", "actor"}
    assert not (set(type(params).model_fields) & forbidden)
