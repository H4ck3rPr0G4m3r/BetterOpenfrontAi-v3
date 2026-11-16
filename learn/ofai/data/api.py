"""Polite client for OpenFront's documented public API (spec §4.1, §8.1; roadmap §2 Crawling).

Uses only the `/public/*` endpoints from the engine's docs/API.md, which are meant
for third parties. The in-game endpoints (`/game/{id}`, `/leaderboard/ranked`) sit
behind a Cloudflare challenge on purpose and are never used here.

Rules, all enforced here so no caller can break them:
- at most one request per `min_interval` seconds (default 1 s), retries included;
- exponential backoff on 429 / 5xx / network errors: 2 s, 4 s, ... capped at 120 s,
  honouring `Retry-After`;
- a generic User-Agent with no personal data;
- 403 stops immediately with `ApiForbidden` (never retried, never worked around:
  a Cloudflare challenge is a human check, and no code here solves or hides from one).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

USER_AGENT = "ofai-crawler/0.1"
DEFAULT_BASE = "https://api.openfront.io"
MAX_ATTEMPTS = 6
BACKOFF_START_S = 2.0
BACKOFF_CAP_S = 120.0


class ApiError(Exception):
    """The API failed for good (after retries) or answered something unusable."""


class ApiForbidden(ApiError):
    """403: stop crawling and surface it; never retried or worked around."""


class ApiNotFound(ApiError):
    """404 for one resource (a deleted game, an unknown player)."""


def backoff_delay(attempt: int) -> float:
    """Delay before retry number `attempt + 1` (attempt 0 is the first failure)."""
    return min(BACKOFF_START_S * 2**attempt, BACKOFF_CAP_S)


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


class OpenFrontApi:
    def __init__(
        self,
        base: str = DEFAULT_BASE,
        min_interval: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 60.0,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self.requests = 0
        self._client = httpx.Client(
            base_url=base,
            transport=transport,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenFrontApi:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- endpoints -----------------------------------------------------------

    def leaderboard(self, page: int) -> dict:
        """Ranked ladders (`1v1`, `2v2`, `ffa`, ...), 50 per page; pages 1 and 2 exist."""
        return self._get("/public/leaderboard/ranked", {"page": page})

    def list_games(
        self,
        start: str,
        end: str,
        *,
        type: str | None = "Public",
        mode: str | None = "Free For All",
        ranked_type: str | None = None,
        limit: int = 1000,
        after: str | None = None,
    ) -> list[dict]:
        """Game metadata for games started in [start, end] (at most 2 days), sorted by start then id."""
        params: dict = {"start": start, "end": end, "limit": limit}
        if type:
            params["type"] = type
        if mode:
            params["mode"] = mode
        if ranked_type:
            params["rankedType"] = ranked_type
        if after:
            params["after"] = after
        body = self._get("/public/games", params)
        if not isinstance(body, list):
            raise ApiError("game list is not a list")
        return body

    def player_games(self, public_id: str, cursor: str | None = None) -> tuple[list[dict], str | None]:
        params = {"cursor": cursor} if cursor else None
        body = self._get(f"/public/player/{public_id}/games", params)
        if not isinstance(body, dict):
            raise ApiError(f"player history for {public_id} is not an object")
        results = body.get("results")
        if not isinstance(results, list):
            raise ApiError(f"player history for {public_id} has no results list")
        return results, body.get("nextCursor") or None

    def game(self, game_id: str) -> dict:
        body = self._get(f"/public/game/{game_id}")
        if not isinstance(body, dict):
            raise ApiError(f"game {game_id} is not an object")
        return body

    # -- transport -------------------------------------------------------------

    def _space(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _get(self, path: str, params: dict | None = None) -> Any:
        last_problem = ""
        for attempt in range(MAX_ATTEMPTS):
            self._space()
            self.requests += 1
            try:
                resp = self._client.get(path, params=params)
            except httpx.TransportError as e:
                last_problem = f"network error: {e!r}"
                delay: float | None = backoff_delay(attempt)
            else:
                status = resp.status_code
                if status == 403:
                    why = "Cloudflare challenge" if resp.headers.get("cf-mitigated") == "challenge" else "forbidden"
                    raise ApiForbidden(
                        f"403 from {path} ({why}). The crawler stops here: a human check is never bypassed. "
                        "Ask the OpenFront devs to allow this client."
                    )
                if status == 404:
                    raise ApiNotFound(f"404 from {path}")
                if status == 429 or status >= 500:
                    last_problem = f"HTTP {status}"
                    delay = _retry_after(resp)
                    if delay is None:
                        delay = backoff_delay(attempt)
                elif 200 <= status < 300:
                    try:
                        return resp.json()
                    except ValueError as e:
                        raise ApiError(f"{path} answered non-JSON ({resp.headers.get('content-type')})") from e
                else:
                    raise ApiError(f"HTTP {status} from {path}")
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(delay)
                # The backoff already spaced us; don't stack the interval on top.
                self._last_request = self._clock() - self.min_interval
        raise ApiError(f"{path} failed after {MAX_ATTEMPTS} attempts ({last_problem})")
