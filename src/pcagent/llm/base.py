"""Choosing one code from the shortlist.

Retrieval narrows a few thousand codes to about eight. Something then has to pick
one and say how sure it is. That step is behind a protocol with two
implementations, which is what makes the ablation in evals/report.md possible:

``BaselineDecider``
    Take the top reranked candidate. No model, no API key, no cost. This is the
    number every LLM result has to beat to justify its existence.

``LLMDecider``
    Ask Claude to pick from the shortlist and ground the choice in spans of the
    caller's own text.

Reporting the baseline next to the model is not modesty, it is the only way to
know whether the model is adding judgement or just adding latency.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..domain import Candidate, Taxonomy


class DecisionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str
    taxonomy: Taxonomy
    candidates: tuple[Candidate, ...]


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: tuple[str, ...] = ()
    decided_by: str


@runtime_checkable
class Decider(Protocol):
    name: str

    async def decide(self, request: DecisionRequest) -> Decision: ...


class BaselineDecider:
    """Takes the top candidate. The honest floor for every other approach.

    Confidence is the gap between the best candidate and the runner up, divided
    by the spread across the whole shortlist. A clear winner is a confident
    answer, a near tie is not.

    The division is the part that matters, and it is there because the first
    version of this class got it wrong. Squashing the raw margin through a
    sigmoid works when the scores are cross encoder logits, where a margin of 1
    means something. Fed reciprocal rank fusion scores, where the whole
    shortlist spans about 0.0003, every margin is effectively zero and the
    sigmoid returns 0.5 for everything. The eval's calibration table showed a
    single bucket holding all 257 predictions at exactly 0.500, which is what a
    broken confidence signal looks like from the outside.

    Dividing by the spread makes the number scale free, so it means the same
    thing whichever ranker produced the scores.

    It is still a heuristic and not a probability. That is precisely why the
    eval measures calibration rather than taking it on trust, and why a
    retrieval only baseline is a weak thing to route humans with.
    """

    name = "baseline-top1"

    def __init__(self, *, floor: float = 0.05, ceiling: float = 0.95) -> None:
        # Never claims certainty and never claims nothing. A ranker that put
        # one candidate far ahead of the rest is still only a ranker.
        self.floor = floor
        self.ceiling = ceiling

    async def decide(self, request: DecisionRequest) -> Decision:
        if not request.candidates:
            return Decision(
                code="",
                confidence=0.0,
                rationale="retrieval returned no candidates",
                decided_by=self.name,
            )

        candidates = request.candidates
        best = candidates[0]
        if len(candidates) == 1:
            confidence = self.ceiling
        else:
            margin = best.score - candidates[1].score
            spread = best.score - candidates[-1].score
            share = (margin / spread) if spread > 1e-12 else 0.0
            confidence = self.floor + (self.ceiling - self.floor) * min(
                max(share, 0.0), 1.0
            )

        return Decision(
            code=best.code,
            confidence=round(confidence, 4),
            rationale=(
                "Top ranked candidate after hybrid retrieval. No model was "
                "consulted, and the confidence is a margin heuristic rather "
                "than a calibrated probability."
            ),
            decided_by=self.name,
        )


def ground_evidence(description: str, spans: list[str]) -> tuple[str, ...]:
    """Keep only the spans that really appear in the description.

    A model asked for verbatim quotes will sometimes paraphrase, and a
    paraphrase presented as a quote is worse than no quote at all, because a
    reviewer skims it and assumes the text was checked. Matching is done on a
    casefolded, whitespace collapsed copy so that trivial differences do not
    throw away a genuine quote.
    """
    haystack = " ".join(description.split()).casefold()
    kept: list[str] = []
    for span in spans:
        needle = " ".join(span.split())
        if needle and needle.casefold() in haystack and needle not in kept:
            kept.append(needle)
    return tuple(kept)
