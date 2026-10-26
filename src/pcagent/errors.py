"""The error vocabulary every tool speaks.

Each transport maps these onto its own conventions. FastAPI turns them into HTTP
status codes, MCP turns them into tool errors the model can read and act on. The
tool layer itself never knows which transport called it.

The ``code`` strings are part of the public contract. A caller is expected to
branch on ``code``, never on the message text.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for every failure a tool raises deliberately."""

    code: str = "tool_error"
    http_status: int = 500
    # Whether a caller that sees this error should try the same request again.
    retryable: bool = False

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            **({"details": self.details} if self.details else {}),
        }


class ValidationFailed(ToolError):
    code = "validation_failed"
    http_status = 422


class NotAuthorized(ToolError):
    """The principal is known but lacks the scope the tool requires.

    Deliberately not called AuthenticationError. Everything reaching the tool
    layer has already been authenticated by the transport.
    """

    code = "not_authorized"
    http_status = 403


class NotFound(ToolError):
    code = "not_found"
    http_status = 404


class IdempotencyConflict(ToolError):
    """An idempotency key was reused with a different request body.

    Not retryable. Replaying the original key with the original body returns the
    original result, but a changed body means the caller has a bug and silently
    picking one of the two bodies would be worse than failing.
    """

    code = "idempotency_conflict"
    http_status = 409


class RequestInFlight(ToolError):
    """The same idempotency key is being processed right now by someone else."""

    code = "request_in_flight"
    http_status = 409
    retryable = True


class UpstreamUnavailable(ToolError):
    """A model provider or other outbound dependency timed out or failed."""

    code = "upstream_unavailable"
    http_status = 503
    retryable = True
