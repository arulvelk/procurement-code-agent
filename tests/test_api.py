"""The HTTP transport.

These tests exist to check the translation layer, not the business rules. The
rules already have tests against the tools directly, and repeating them here
would just mean two places to update. What is tested here is the part only HTTP
has: bearer token parsing, the error to status code mapping, and the
Idempotency-Key header.

The app is built without running its lifespan, and the services are injected,
so the suite never loads an ONNX model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pcagent.api import create_app, get_services
from pcagent.auth import READ_ONLY, REVIEWER, Principal, TokenRegistry

READER = {"Authorization": "Bearer reader-token"}
REVIEWER_HEADERS = {"Authorization": "Bearer reviewer-token"}


@pytest.fixture
def client(services) -> TestClient:
    app = create_app()
    # No `with` block, so the lifespan never runs and nothing is downloaded.
    app.dependency_overrides[get_services] = lambda: services
    app.state.tokens = TokenRegistry(
        {
            "reader-token": Principal("svc-reader", READ_ONLY),
            "reviewer-token": Principal("ana.reviewer", REVIEWER),
        }
    )
    return TestClient(app)


def a_classification(client: TestClient, description: str = "office furniture") -> str:
    response = client.post("/v1/classify", json={"description": description}, headers=READER)
    assert response.status_code == 200, response.text
    return response.json()["classification_id"]


def test_health_needs_no_token(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_missing_token_is_forbidden_not_unauthorised(client: TestClient) -> None:
    """403 rather than 401 is deliberate.

    The transport authenticates and the tool authorises. A request with no
    token is a known caller with no scopes, so the honest answer is "you may
    not", not "who are you".
    """
    assert client.post("/v1/search", json={"query": "desks"}).status_code == 403


@pytest.mark.parametrize(
    "header", ["Bearer wrong-token", "Basic reader-token", "reader-token", ""]
)
def test_malformed_or_unknown_credentials_are_refused(
    client: TestClient, header: str
) -> None:
    response = client.post(
        "/v1/search", json={"query": "desks"}, headers={"Authorization": header}
    )
    assert response.status_code == 403


def test_bearer_prefix_is_case_insensitive(client: TestClient) -> None:
    response = client.post(
        "/v1/search", json={"query": "desks"}, headers={"Authorization": "bearer reader-token"}
    )
    assert response.status_code == 200


def test_search_returns_candidates(client: TestClient) -> None:
    body = client.post(
        "/v1/search", json={"query": "office furniture", "limit": 3}, headers=READER
    ).json()
    assert body["candidates"]
    assert len(body["candidates"]) <= 3


def test_unknown_code_maps_to_404(client: TestClient) -> None:
    response = client.get("/v1/codes/psc/ZZZZ", headers=READER)
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_reader_cannot_write_a_review(client: TestClient) -> None:
    classification_id = a_classification(client)
    response = client.post(
        "/v1/reviews",
        json={"classification_id": classification_id, "decision": "accept"},
        headers={**READER, "Idempotency-Key": "http-key-1"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "not_authorized"


def test_idempotency_key_header_is_required(client: TestClient) -> None:
    classification_id = a_classification(client)
    response = client.post(
        "/v1/reviews",
        json={"classification_id": classification_id, "decision": "accept"},
        headers=REVIEWER_HEADERS,
    )
    assert response.status_code == 422


def test_replay_returns_the_original_review(client: TestClient) -> None:
    classification_id = a_classification(client)
    payload = {"classification_id": classification_id, "decision": "accept"}
    headers = {**REVIEWER_HEADERS, "Idempotency-Key": "http-key-2"}

    first = client.post("/v1/reviews", json=payload, headers=headers)
    second = client.post("/v1/reviews", json=payload, headers=headers)

    assert first.status_code == 201
    assert first.json()["review_id"] == second.json()["review_id"]
    assert len(client.get("/v1/reviews", headers=REVIEWER_HEADERS).json()["reviews"]) == 1


def test_same_key_different_body_is_409(client: TestClient) -> None:
    classification_id = a_classification(client)
    headers = {**REVIEWER_HEADERS, "Idempotency-Key": "http-key-3"}
    client.post(
        "/v1/reviews",
        json={"classification_id": classification_id, "decision": "accept"},
        headers=headers,
    )
    conflict = client.post(
        "/v1/reviews",
        json={"classification_id": classification_id, "decision": "reject"},
        headers=headers,
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"
    assert conflict.json()["retryable"] is False


def test_cross_field_rule_is_422_not_500(client: TestClient) -> None:
    """The params model raises pydantic, not ToolError. Without a handler for
    that, a caller error would be reported as a server error."""
    classification_id = a_classification(client)
    response = client.post(
        "/v1/reviews",
        json={"classification_id": classification_id, "decision": "reassign"},
        headers={**REVIEWER_HEADERS, "Idempotency-Key": "http-key-4"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_failed"


def test_unknown_body_fields_are_rejected(client: TestClient) -> None:
    """extra="forbid" everywhere. A caller setting `reviewer` should be told
    the field does not exist rather than have it silently ignored."""
    classification_id = a_classification(client)
    response = client.post(
        "/v1/reviews",
        json={
            "classification_id": classification_id,
            "decision": "accept",
            "reviewer": "someone.else",
        },
        headers={**REVIEWER_HEADERS, "Idempotency-Key": "http-key-5"},
    )
    assert response.status_code == 422


def test_recorded_reviewer_comes_from_the_token(client: TestClient) -> None:
    classification_id = a_classification(client)
    body = client.post(
        "/v1/reviews",
        json={"classification_id": classification_id, "decision": "accept"},
        headers={**REVIEWER_HEADERS, "Idempotency-Key": "http-key-6"},
    ).json()
    assert body["reviewer"] == "ana.reviewer"


def test_openapi_schema_builds(client: TestClient) -> None:
    """Cheap, and it catches response models that cannot be serialised."""
    schema = client.get("/openapi.json").json()
    assert "/v1/classify" in schema["paths"]
    assert "/v1/reviews" in schema["paths"]
