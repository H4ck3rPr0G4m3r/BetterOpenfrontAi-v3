import httpx
import pytest

from ofai.data.api import USER_AGENT, ApiError, ApiForbidden, OpenFrontApi


class FakeTime:
    """A clock that only moves when someone sleeps; records every sleep and
    the clock reading at every request."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.request_times: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def make_api(handler, t: FakeTime | None = None) -> tuple[OpenFrontApi, FakeTime]:
    t = t or FakeTime()

    def wrapped(request: httpx.Request) -> httpx.Response:
        t.request_times.append(t.now)
        return handler(request)

    api = OpenFrontApi(transport=httpx.MockTransport(wrapped), clock=t.clock, sleep=t.sleep)
    return api, t


def test_requests_are_spaced_one_second():
    api, t = make_api(lambda r: httpx.Response(200, json={"ok": 1}))
    for _ in range(3):
        api.game("abc")
    gaps = [b - a for a, b in zip(t.request_times, t.request_times[1:])]
    assert sum(t.sleeps) >= 2.0
    assert all(g >= 1.0 for g in gaps)


def test_user_agent_is_generic():
    seen = {}

    def handler(r: httpx.Request) -> httpx.Response:
        seen["ua"] = r.headers["user-agent"]
        return httpx.Response(200, json={})

    api, _ = make_api(handler)
    api.game("abc")
    assert seen["ua"] == "ofai-crawler/0.1" == USER_AGENT


def test_429_backs_off_and_honours_retry_after():
    calls = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={"x": 1})])
    api, t = make_api(lambda r: next(calls))
    assert api.game("abc") == {"x": 1}
    assert 7 in t.sleeps


def test_5xx_backoff_doubles_from_2s_capped_120():
    calls = iter([httpx.Response(503)] * 5 + [httpx.Response(200, json={})])
    api, t = make_api(lambda r: next(calls))
    api.game("abc")
    backoffs = [s for s in t.sleeps if s >= 2]
    assert backoffs == [2, 4, 8, 16, 32]


def test_backoff_is_capped_at_120():
    from ofai.data.api import backoff_delay

    assert backoff_delay(10) == 120


def test_gives_up_after_6_attempts():
    count = {"n": 0}

    def handler(r):
        count["n"] += 1
        return httpx.Response(503)

    api, _ = make_api(handler)
    with pytest.raises(ApiError):
        api.game("abc")
    assert count["n"] == 6


def test_403_raises_forbidden_immediately():
    count = {"n": 0}

    def handler(r):
        count["n"] += 1
        return httpx.Response(403, headers={"cf-mitigated": "challenge"}, text="<html>Just a moment...</html>")

    api, _ = make_api(handler)
    with pytest.raises(ApiForbidden) as e:
        api.leaderboard(1)
    assert count["n"] == 1
    assert "challenge" in str(e.value)


def test_player_games_returns_results_and_cursor():
    seen = {}

    def handler(r: httpx.Request) -> httpx.Response:
        seen["url"] = str(r.url)
        return httpx.Response(200, json={"results": [{"gameId": "a"}], "nextCursor": "abc"})

    api, _ = make_api(handler)
    results, cursor = api.player_games("P1", cursor="c0")
    assert results == [{"gameId": "a"}]
    assert cursor == "abc"
    assert seen["url"].endswith("/public/player/P1/games?cursor=c0")


def test_player_games_without_cursor_is_none():
    api, _ = make_api(lambda r: httpx.Response(200, json={"results": []}))
    assert api.player_games("P1") == ([], None)


def test_uses_documented_public_endpoints():
    urls = []

    def handler(r):
        urls.append(r.url.path)
        return httpx.Response(200, json=[] if r.url.path == "/public/games" else {"results": []})

    api, _ = make_api(handler)
    api.game("g1")
    api.leaderboard(1)
    api.list_games("2026-11-14T00:00:00Z", "2026-11-15T00:00:00Z")
    api.player_games("p1")
    assert urls == ["/public/game/g1", "/public/leaderboard/ranked", "/public/games", "/public/player/p1/games"]
