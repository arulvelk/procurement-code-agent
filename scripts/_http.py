"""Tiny stdlib HTTP helper shared by the one-off data scripts.

These scripts run once to produce the committed files under data/. They use only
the standard library on purpose, so that regenerating the corpus never depends on
whatever happens to be installed in the project environment.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "procurement-code-agent/0.1 (public dataset builder)"
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "http"
)
MAX_ATTEMPTS = 6
THROTTLE_SECONDS = 0.75  # this is a free public API, so one request every 750ms


def _sleep_backoff(attempt: int) -> None:
    # Full jitter. Spreads retries out so a transient upstream blip does not turn
    # into a synchronised stampede from the worker pool.
    time.sleep(random.uniform(0.0, min(8.0, 0.5 * 2**attempt)))


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """GET or POST a JSON endpoint, retrying on transient failures."""
    body = json.dumps(payload, sort_keys=True).encode() if payload is not None else None

    # Responses are cached on disk so that a rerun never asks for anything it has
    # already been given. That makes the walk resumable after an interruption and
    # keeps repeated runs off the upstream API entirely.
    cache_path = _cache_path(url, body)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        time.sleep(THROTTLE_SECONDS)
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode())
            _write_cache(cache_path, parsed)
            return parsed
        except urllib.error.HTTPError as exc:
            # 4xx other than rate limiting will not get better by trying again.
            if exc.code < 500 and exc.code != 429:
                raise
            last_error = exc
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            # OSError covers URLError, connection resets and timeouts. The public
            # API drops connections under load, so these are expected and retried.
            last_error = exc
        _sleep_backoff(attempt)

    raise RuntimeError(f"giving up on {url} after {MAX_ATTEMPTS} attempts") from last_error


def _cache_path(url: str, body: bytes | None) -> str:
    # url and body are hashed separately so that no byte sequence in one can
    # be confused with the start of the other.
    digest = hashlib.sha256(
        hashlib.sha256(url.encode()).digest() + hashlib.sha256(body or b"").digest()
    ).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{digest}.json")


def _read_cache(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_cache(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(temporary, path)


def quote(segment: str) -> str:
    return urllib.parse.quote(segment, safe="")


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
