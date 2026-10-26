"""Settings, read from the environment once at startup.

Defaults are chosen so that ``pip install -e . && pytest`` works with no
configuration and no API key. The deliberate consequence is that the shipped
default decider is the retrieval baseline, not a model. Turning the model on is
one environment variable, and the README says what it costs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .auth import Principal, Scope, TokenRegistry

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DeciderName = Literal["baseline", "claude"]

# Stand in for an identity provider. Real deployments set PCAGENT_TOKENS, or
# replace TokenRegistry with a call to whatever issues their tokens. These exist
# so that the quickstart in the README is three commands rather than thirteen.
DEV_TOKENS: dict[str, tuple[str, frozenset[Scope]]] = {
    "dev-reader-token": (
        "svc-reader",
        frozenset({Scope.CODES_READ, Scope.CLASSIFY_RUN, Scope.REVIEWS_READ}),
    ),
    "dev-reviewer-token": (
        "ana.reviewer",
        frozenset(
            {Scope.CODES_READ, Scope.CLASSIFY_RUN, Scope.REVIEWS_READ, Scope.REVIEWS_WRITE}
        ),
    ),
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    database_path: Path = ROOT / "pcagent.db"
    taxonomies: tuple[str, ...] = ("psc", "naics", "arxiv")

    decider: DeciderName = "baseline"
    model: str = "claude-opus-5-5"
    llm_timeout_seconds: float = 20.0

    review_threshold: float = 0.65
    """Classifications at or below this confidence are routed to a human. Set
    from the calibration table in evals/report.md rather than by taste."""

    max_concurrency: int = 4
    """Ceiling on in flight classifications in the batch agent. Bounded because
    unbounded fan out is how a batch job turns into an outage somewhere else."""

    tokens: dict[str, tuple[str, frozenset[Scope]]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        decider = os.environ.get("PCAGENT_DECIDER", "baseline").strip().lower()
        if decider not in ("baseline", "claude"):
            raise ValueError(f"PCAGENT_DECIDER must be baseline or claude, got {decider!r}")

        return cls(
            data_dir=Path(os.environ.get("PCAGENT_DATA_DIR", ROOT / "data")),
            database_path=Path(os.environ.get("PCAGENT_DB", ROOT / "pcagent.db")),
            decider=decider,  # type: ignore[arg-type]
            model=os.environ.get("PCAGENT_MODEL", "claude-opus-5-5"),
            llm_timeout_seconds=_env_float("PCAGENT_LLM_TIMEOUT", 20.0),
            review_threshold=_env_float("PCAGENT_REVIEW_THRESHOLD", 0.65),
            max_concurrency=_env_int("PCAGENT_MAX_CONCURRENCY", 4),
            tokens=_load_tokens(),
        )

    def token_registry(self) -> TokenRegistry:
        registry = TokenRegistry()
        for token, (subject, scopes) in (self.tokens or DEV_TOKENS).items():
            registry.register(token, Principal(subject=subject, scopes=scopes))
        return registry


def _load_tokens() -> dict[str, tuple[str, frozenset[Scope]]]:
    """Read PCAGENT_TOKENS, a JSON object of token to subject and scopes."""
    raw = os.environ.get("PCAGENT_TOKENS")
    if not raw:
        logger.warning(
            "PCAGENT_TOKENS is not set, using built in development tokens. "
            "Do not run this configuration anywhere real."
        )
        return {}

    parsed = json.loads(raw)
    tokens: dict[str, tuple[str, frozenset[Scope]]] = {}
    for token, spec in parsed.items():
        tokens[token] = (
            spec["subject"],
            frozenset(Scope(scope) for scope in spec["scopes"]),
        )
    return tokens
