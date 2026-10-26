"""The same triage flow expressed as a LangGraph graph.

This exists to make a point about the architecture rather than to replace
``batch.py``. The tools in ``pcagent.tools`` know nothing about LangGraph, and
nothing about MCP or FastAPI either. Swapping the orchestrator is a new file,
not a refactor, because the authorization, validation and idempotency live
underneath all of them.

Which one to reach for:

``batch.py``
    High volume, one shape of work, and the things you need to control are
    concurrency, timeouts and failure isolation. A plain loop expresses those
    directly.

``graph.py``
    Branching work where the state machine is the interesting part and you want
    it declared rather than implied by control flow. Checkpointing, resuming a
    half finished run and streaming intermediate state come free, which a hand
    written loop has to grow for itself.

The graph here is deliberately small: classify, then branch on confidence into
auto accept, keep as is, or route to a human.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ..domain import Classification, Taxonomy
from ..errors import ToolError
from ..tools.base import ToolContext
from ..tools.classify import ClassifyParams, classify_purchase
from ..tools.review import SubmitReviewParams, submit_review


class TriageState(TypedDict, total=False):
    description: str
    taxonomy: Taxonomy
    classification: Classification | None
    needs_review: bool
    auto_accepted: bool
    error: str | None
    error_code: str | None


def build_triage_graph(
    ctx: ToolContext,
    *,
    review_threshold: float = 0.65,
    auto_accept_above: float = 0.9,
) -> Any:
    """Compile a graph that classifies one description and routes the result.

    ``ctx`` is captured here rather than carried in the state. State is data
    that gets checkpointed and streamed, and an authenticated principal is
    neither. Keeping it out of the state also means no graph node can rewrite
    who the caller is.
    """

    async def classify_node(state: TriageState) -> TriageState:
        try:
            classification = await classify_purchase(
                ctx,
                ClassifyParams(
                    description=state["description"],
                    taxonomy=state.get("taxonomy", "psc"),
                ),
            )
        except ToolError as exc:
            return {"classification": None, "error": exc.message, "error_code": exc.code}
        return {"classification": classification}

    def route(state: TriageState) -> Literal["accept", "keep", "flag", "failed"]:
        """Three bands, not two.

        Above auto_accept_above the system records an acceptance on its own.
        At or below review_threshold a human is asked. The band in between is
        the interesting one: the answer is good enough to return, but not good
        enough to sign someone's name to, so it is kept without a review being
        recorded. Collapsing that band into either neighbour is how a triage
        system ends up either rubber stamping or drowning its reviewers.
        """
        classification = state.get("classification")
        if classification is None:
            return "failed"
        if classification.confidence > auto_accept_above:
            return "accept"
        if classification.confidence <= review_threshold:
            return "flag"
        return "keep"

    async def accept_node(state: TriageState) -> TriageState:
        classification = state["classification"]
        assert classification is not None
        try:
            await submit_review(
                ctx,
                SubmitReviewParams(
                    classification_id=classification.classification_id,
                    decision="accept",
                    note=f"Auto accepted at confidence {classification.confidence:.2f}.",
                    idempotency_key=f"graph-accept-{classification.classification_id}",
                ),
            )
        except ToolError as exc:
            # A missing scope here is not a crash, it is the system telling the
            # graph it is not allowed to auto accept. Fall back to review.
            return {"needs_review": True, "error": exc.message, "error_code": exc.code}
        return {"auto_accepted": True, "needs_review": False}

    async def keep_node(state: TriageState) -> TriageState:
        return {"needs_review": False, "auto_accepted": False}

    async def flag_node(state: TriageState) -> TriageState:
        return {"needs_review": True, "auto_accepted": False}

    async def failed_node(state: TriageState) -> TriageState:
        return {"needs_review": True, "auto_accepted": False}

    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_node)
    graph.add_node("accept", accept_node)
    graph.add_node("keep", keep_node)
    graph.add_node("flag", flag_node)
    graph.add_node("failed", failed_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {"accept": "accept", "keep": "keep", "flag": "flag", "failed": "failed"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("keep", END)
    graph.add_edge("flag", END)
    graph.add_edge("failed", END)

    return graph.compile()
