"""The data contracts.

Everything that crosses a boundary in this project is one of these models. The
MCP tools, the HTTP API, the store and the evals all speak the same types, which
is the whole reason a tool can be exposed over two transports without either one
growing its own notion of what a classification is.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Taxonomy = Literal["psc", "naics", "arxiv"]
"""Which federal code set a request is about.

psc is the Product Service Code, four characters, describing *what was bought*.
naics is the six digit industry code, describing *what the seller does*. They
answer different questions about the same purchase, so the tools keep them
separate rather than pretending one implies the other.

arxiv is the arXiv subject taxonomy. It is here as a control. The procurement
corpora turned out to have a low ceiling from text alone, and without a second
corpus where the text really does determine the label there would be no way to
tell a weak pipeline apart from a hard dataset. See the README.
"""

ReviewDecision = Literal["accept", "reject", "reassign"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaxonomyCode(Frozen):
    """One leaf code in a taxonomy, with the parent titles that give it context."""

    taxonomy: Taxonomy
    code: str
    title: str
    parent_code: str = ""
    parent_title: str = ""
    category: str = ""
    description: str = ""
    """An official prose description, where the taxonomy publishes one. arXiv
    does, the procurement code sets do not."""

    @property
    def index_text(self) -> str:
        """The text the retrieval layer actually indexes.

        A leaf title alone is often close to meaningless out of context. "GUNS,
        OVER 300MM" only retrieves well when "WEAPONS" travels with it, and
        "Soybean Farming" is easier to reach through "Oilseed and Grain Farming".
        So the parent and category titles are folded into the indexed text.
        """
        parts = [self.title, self.parent_title, self.category, self.description]
        seen: list[str] = []
        for part in parts:
            cleaned = part.strip()
            if cleaned and cleaned.lower() not in {s.lower() for s in seen}:
                seen.append(cleaned)
        return " | ".join(seen)


class Candidate(Frozen):
    """A code the retrieval layer put forward, with how it got there."""

    code: str
    title: str
    taxonomy: Taxonomy
    score: float
    # Rank in the fused candidate list, 1 based. Kept so the evals can measure
    # what reranking actually moved rather than just reporting a final number.
    retrieved_rank: int
    reranked_rank: int | None = None


class Classification(Frozen):
    """The answer to "what code is this purchase"."""

    classification_id: str
    taxonomy: Taxonomy
    description: str
    code: str
    title: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    rationale: str
    # Verbatim spans lifted from the input description. Grounding the answer in
    # the caller's own words makes a wrong answer obvious to a human reviewer in
    # a way that a free text rationale does not.
    evidence: tuple[str, ...] = ()
    alternatives: tuple[Candidate, ...] = ()
    decided_by: str = "unknown"
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("code")
    @classmethod
    def _code_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("code must not be blank")
        return value


class Review(Frozen):
    """A human decision recorded against a classification."""

    review_id: str
    classification_id: str
    decision: ReviewDecision
    reviewer: str
    assigned_code: str | None = None
    note: str = ""
    created_at: datetime = Field(default_factory=utcnow)


def fingerprint(payload: object) -> str:
    """A stable hash of a request body, used to police idempotency key reuse.

    Sorting keys matters. Two callers sending the same fields in a different
    order are making the same request and must not be told otherwise.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
