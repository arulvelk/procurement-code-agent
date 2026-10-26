"""Who is calling, and what they are allowed to do.

The single most important rule in this repo lives here: a principal is supplied
by the transport that authenticated the caller, never by the model. Tool
arguments carry business parameters only. There is no ``reviewer`` or ``scopes``
field a model can fill in, because a model that can name its own permissions has
none.

The practical consequence is that prompt injection against this agent can at
worst cause a *badly chosen* tool call. It cannot cause an unauthorised one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from .errors import NotAuthorized


class Scope(str, Enum):
    """Permissions, split by blast radius rather than by tool."""

    CODES_READ = "codes:read"
    """Read the taxonomy. Cheap, no side effects."""

    CLASSIFY_RUN = "classify:run"
    """Run a classification. No side effects, but it spends money and time, so
    it is separable from plain taxonomy reads."""

    REVIEWS_READ = "reviews:read"
    """Read the human review queue."""

    REVIEWS_WRITE = "reviews:write"
    """Record a review decision. The only scope in this list that mutates
    anything."""


@dataclass(frozen=True)
class Principal:
    subject: str
    scopes: frozenset[Scope]

    def has(self, scope: Scope) -> bool:
        return scope in self.scopes

    def require(self, scope: Scope) -> None:
        """Raise unless this principal holds ``scope``.

        Called at the top of every tool body. Not in a wrapper that a future
        refactor could forget to apply, and not in the prompt.
        """
        if scope not in self.scopes:
            raise NotAuthorized(
                f"{self.subject} is not permitted to {scope.value}",
                subject=self.subject,
                required_scope=scope.value,
                held_scopes=sorted(s.value for s in self.scopes),
            )


ANONYMOUS = Principal(subject="anonymous", scopes=frozenset())

READ_ONLY = frozenset({Scope.CODES_READ, Scope.CLASSIFY_RUN, Scope.REVIEWS_READ})
REVIEWER = READ_ONLY | {Scope.REVIEWS_WRITE}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenRegistry:
    """Resolves a bearer token to a principal.

    Tokens are held as SHA-256 digests and looked up by digest, so the registry
    never keeps a secret in memory in a comparable form and the lookup does not
    compare secrets byte by byte. A real deployment would put an identity
    provider here. The shape of the call site would not change, which is the
    point of keeping this behind one method.
    """

    def __init__(self, tokens: dict[str, Principal] | None = None) -> None:
        self._by_digest: dict[str, Principal] = {
            _digest(token): principal for token, principal in (tokens or {}).items()
        }

    def register(self, token: str, principal: Principal) -> None:
        self._by_digest[_digest(token)] = principal

    def resolve(self, token: str | None) -> Principal:
        """Return the principal for ``token``, or ANONYMOUS if it is unknown.

        Returning anonymous rather than raising keeps authentication and
        authorisation separate. An unknown token is simply a caller with no
        scopes, and the tool it tried to reach is what decides the error.
        """
        if not token:
            return ANONYMOUS
        return self._by_digest.get(_digest(token), ANONYMOUS)
