"""The two adapters: LangGraph and MCP.

Both are thin by design, so both get thin tests. What is worth asserting is
that the tool core really is reusable without modification, and that neither
adapter has quietly grown a way around the authorization rules.
"""

from __future__ import annotations

import pytest

# The LangGraph adapter is an optional extra, so its tests skip rather than
# error when someone installed only ".[dev]". The MCP tests below still run.
pytest.importorskip("langgraph", reason="install with .[graph] to test the adapter")

from pcagent.agent.graph import build_triage_graph
from pcagent.tools import ListReviewsParams, list_reviews


async def test_high_confidence_is_auto_accepted(reviewer) -> None:
    graph = build_triage_graph(reviewer, review_threshold=0.1, auto_accept_above=0.2)
    final = await graph.ainvoke(
        {"description": "office furniture cabinets", "taxonomy": "psc"}
    )
    assert final["auto_accepted"] is True
    assert final["needs_review"] is False
    assert len((await list_reviews(reviewer, ListReviewsParams())).reviews) == 1


async def test_low_confidence_is_flagged_and_writes_nothing(reviewer) -> None:
    graph = build_triage_graph(reviewer, review_threshold=1.1, auto_accept_above=1.2)
    final = await graph.ainvoke({"description": "office furniture", "taxonomy": "psc"})
    assert final["needs_review"] is True
    assert final.get("auto_accepted") is not True
    assert (await list_reviews(reviewer, ListReviewsParams())).reviews == ()


async def test_middle_band_is_kept_without_a_review(reviewer) -> None:
    """Neither rubber stamped nor dumped on a human."""
    graph = build_triage_graph(reviewer, review_threshold=0.0, auto_accept_above=1.5)
    final = await graph.ainvoke({"description": "office furniture", "taxonomy": "psc"})
    assert final["needs_review"] is False
    assert final["auto_accepted"] is False
    assert (await list_reviews(reviewer, ListReviewsParams())).reviews == ()


async def test_the_graph_cannot_escape_the_scope_check(reader) -> None:
    """A reader driving the graph gets review routing, not a write.

    The graph node asks, the tool refuses, and the graph degrades to flagging
    for a human instead of failing.
    """
    graph = build_triage_graph(reader, review_threshold=0.1, auto_accept_above=0.2)
    final = await graph.ainvoke({"description": "office furniture", "taxonomy": "psc"})
    assert final["needs_review"] is True
    assert final["error_code"] == "not_authorized"


async def test_graph_reports_a_classification_failure_rather_than_raising(
    reviewer, monkeypatch
) -> None:
    async def explode(ctx, params):
        from pcagent.errors import UpstreamUnavailable

        raise UpstreamUnavailable("model is down")

    monkeypatch.setattr("pcagent.agent.graph.classify_purchase", explode)
    graph = build_triage_graph(reviewer)
    final = await graph.ainvoke({"description": "office furniture", "taxonomy": "psc"})
    assert final["needs_review"] is True
    assert final["error_code"] == "upstream_unavailable"


def test_mcp_server_exposes_the_expected_tools() -> None:
    """Importing the module must not build services or touch the network."""
    from pcagent import mcp_server

    assert mcp_server.mcp.name == "procurement-code-agent"


@pytest.mark.parametrize(
    "tool_name",
    [
        "search_codes_tool",
        "get_code_tool",
        "classify_purchase_tool",
        "submit_review_tool",
        "list_reviews_tool",
    ],
)
def test_mcp_tool_functions_exist(tool_name: str) -> None:
    from pcagent import mcp_server

    assert hasattr(mcp_server, tool_name)


def test_no_mcp_tool_accepts_an_identity_argument() -> None:
    """Over stdio the principal comes from the environment at startup.

    If a tool here grew a reviewer argument, a model could set it, and the
    whole authorization story would be decoration.
    """
    import inspect

    from pcagent import mcp_server

    forbidden = {"reviewer", "subject", "principal", "scopes", "user", "actor", "ctx"}
    for name in dir(mcp_server):
        if not name.endswith("_tool"):
            continue
        target = getattr(mcp_server, name)
        function = getattr(target, "fn", target)
        if not callable(function):
            continue
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & forbidden), f"{name} accepts an identity argument"
