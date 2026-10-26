"""The Claude backed decider.

Everything outbound in this file is bounded. There is a per call timeout, a
single retry budget, and a fallback that returns a usable answer instead of an
exception when the provider is unreachable. An agent that can hang is an agent
that takes a queue down with it.

The model's answer is not trusted on arrival. Three checks run before it becomes
a ``Decision``:

* the chosen code must be one of the candidates we offered,
* the confidence must be a number in range,
* quoted evidence must actually appear in the input text.

A structured output schema makes the response well formed. It does not make it
correct, and the difference matters.
"""

from __future__ import annotations

import logging
import os
from typing import cast

from pydantic import BaseModel, Field

from ..errors import UpstreamUnavailable
from .base import BaselineDecider, Decision, DecisionRequest, ground_evidence
from .prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_TOKENS = 1024


class Verdict(BaseModel):
    """The shape the model must answer in."""

    code: str = Field(description="One code, copied exactly from the candidate list")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One or two sentences")
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim spans copied from the purchase description",
    )


class LLMDecider:
    """Picks from the shortlist with Claude, and degrades to the baseline.

    ``fallback_to_baseline`` is on by default. A classification service whose
    model provider is briefly unreachable is better returning the retrieval
    answer labelled as such than returning a 503. The label travels on the
    decision as ``decided_by``, so a degraded answer is never mistaken for a
    model answer downstream or in the evals.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = "low",
        client: object | None = None,
        fallback_to_baseline: bool = True,
    ) -> None:
        self.model = model or os.environ.get("PCAGENT_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        # Classification is a judgement call over eight short strings, not a
        # reasoning marathon. Low effort keeps latency and cost where they
        # belong for a step that runs on every row of a batch.
        self.effort = effort
        self.name = f"claude:{self.model}"
        self._client = client
        self._baseline = BaselineDecider()
        self.fallback_to_baseline = fallback_to_baseline

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - import guard
                raise UpstreamUnavailable(
                    "the anthropic package is not installed, "
                    "install it with pip install 'procurement-code-agent[llm]'"
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise UpstreamUnavailable("ANTHROPIC_API_KEY is not set")
            self._client = AsyncAnthropic()
        return self._client

    async def decide(self, request: DecisionRequest) -> Decision:
        if not request.candidates:
            return await self._baseline.decide(request)

        try:
            verdict = await self._ask(request)
        except UpstreamUnavailable:
            if not self.fallback_to_baseline:
                raise
            logger.warning("model unavailable, falling back to retrieval top 1")
            fallback = await self._baseline.decide(request)
            return fallback.model_copy(
                update={"decided_by": f"{self._baseline.name} (fallback from {self.name})"}
            )

        return self._to_decision(verdict, request)

    async def _ask(self, request: DecisionRequest) -> Verdict:
        client = self._get_client()
        allowed = {candidate.code.upper() for candidate in request.candidates}

        system = build_system_prompt(request.taxonomy)
        user = build_user_prompt(request.description, request.candidates)
        messages: list[dict[str, str]] = [{"role": "user", "content": user}]

        # One retry, and only for an answer that is off the candidate list.
        # Network level retries belong to the SDK, which already backs off, so
        # layering another loop here would multiply the worst case latency.
        for attempt in (1, 2):
            verdict = await self._call(client, system, messages)
            if verdict.code.strip().upper() in allowed:
                return verdict

            logger.info(
                "model returned off list code %r on attempt %d", verdict.code, attempt
            )
            if attempt == 2:
                raise UpstreamUnavailable(
                    "model did not choose from the candidate list",
                    returned=verdict.code,
                )
            messages = [
                {"role": "user", "content": user},
                {
                    "role": "user",
                    "content": (
                        f"{verdict.code} is not on the candidate list. Choose one of "
                        "the numbered codes above, copying it exactly."
                    ),
                },
            ]
        raise AssertionError("unreachable")

    async def _call(
        self, client: object, system: str, messages: list[dict[str, str]]
    ) -> Verdict:
        import anthropic

        try:
            response = await client.messages.parse(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                output_format=Verdict,
                output_config={"effort": self.effort},
                timeout=self.timeout_seconds,
            )
        except anthropic.APITimeoutError as exc:
            raise UpstreamUnavailable(
                f"model call exceeded {self.timeout_seconds}s", model=self.model
            ) from exc
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            raise UpstreamUnavailable(f"model call failed: {exc}", model=self.model) from exc

        parsed = response.parsed_output
        if parsed is None:
            raise UpstreamUnavailable("model returned no parseable output", model=self.model)
        # The SDK returns the parsed object loosely typed. It was produced from
        # the Verdict schema, and pydantic already validated it on the way out.
        return cast(Verdict, parsed)

    def _to_decision(self, verdict: Verdict, request: DecisionRequest) -> Decision:
        # Return the code as the catalogue spells it, not as the model typed it.
        canonical = next(
            candidate.code
            for candidate in request.candidates
            if candidate.code.upper() == verdict.code.strip().upper()
        )
        return Decision(
            code=canonical,
            confidence=min(max(verdict.confidence, 0.0), 1.0),
            rationale=verdict.rationale.strip(),
            evidence=ground_evidence(request.description, verdict.evidence),
            decided_by=self.name,
        )
