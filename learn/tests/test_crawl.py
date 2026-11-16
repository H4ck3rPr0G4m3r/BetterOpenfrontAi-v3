import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ofai.data.api import ApiError, ApiForbidden
from ofai.data.crawl import CrawlConfig, Crawler
from ofai.data.index import Index

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def entry(gid, days_ago=1, **kw):
    return {"gameId": gid, "start": iso(days_ago), "type": "Public", "mode": "Free For All",
            "durationSeconds": 900, "ffaRating": {"before": 2000, "after": 2010}, **kw}


def record(gid, **cfg):
    return {"gitCommit": "c" * 40, "info": {"gameID": gid, "config": {"gameMap": "World", "gameMapSize": "Normal", **cfg},
            "players": [{"clientID": "AAAAAAAA", "username": "t1", "publicID": "T1"}], "start": 0, "duration": 900}}


class FakeApi:
    def __init__(self, boards=None, histories=None, games=None, forbid=False):
        # boards: {"ffa": [[page1 entries], [page2 entries]], "ffaSeason": [...]}
        self.boards = boards or {"ffa": [[{"public_id": "T1", "elo": 2500}, {"public_id": "T2", "elo": 2400}]],
                                 "ffaSeason": [[{"public_id": "T2", "elo": 2450}, {"public_id": "T3", "elo": 1000}]]}
        self.histories = histories or {}  # pid -> list of pages: [(entries, next_cursor)]
        self.games = games or {}
        self.forbid = forbid
        self.requests = 0
        self.game_calls: list[str] = []
        self.history_calls: list[tuple[str, str | None]] = []
        self.on_game = None

    def leaderboard(self, page):
        self.requests += 1
        if self.forbid:
            raise ApiForbidden("403 from /leaderboard/ranked (Cloudflare challenge)")
        out = {}
        for k, pages in self.boards.items():
            out[k] = pages[page - 1] if page - 1 < len(pages) else []
        if all(not v for v in out.values()):
            raise ApiError("HTTP 400")
        return out

    def player_games(self, pid, cursor=None):
        self.requests += 1
        self.history_calls.append((pid, cursor))
        pages = self.histories.get(pid, [([], None)])
        i = 0 if cursor is None else int(cursor.split(":")[1])
        return pages[i]

    def game(self, gid):
        self.requests += 1
        self.game_calls.append(gid)
        if self.on_game:
            self.on_game(gid)
        return self.games[gid]


@pytest.fixture
def env(tmp_path: Path):
    return {"index": Index(tmp_path / "i.sqlite"), "progress_file": tmp_path / "progress.json", "records_root": tmp_path / "rec"}


def crawler(api, env, **cfg):
    stop = cfg.pop("stop", None)
    return Crawler(api, env["index"], CrawlConfig(**cfg), now=lambda: NOW, stop=stop,
                   progress_file=env["progress_file"], records_root=env["records_root"])


def test_teachers_from_both_leaderboards_dedup_top_n(env):
    c = crawler(FakeApi(), env, teachers=2)
    c.run()
    assert c.teacher_ids == ["T1", "T2"]
    assert env["index"].player_elo("T2") == 2450  # best rating across boards


def test_history_stops_at_window_start(env):
    api = FakeApi(histories={"T1": [([entry("g1", 1), entry("g2", 5)], "T1:1"), ([entry("g3", 25), entry("g4", 26)], "T1:2")]},
                  games={"g1": record("g1"), "g2": record("g2")})
    c = crawler(api, env, teachers=1)
    out = c.run()
    assert out["phase"] == "done" and out["fetched"] == 2
    assert sorted(api.game_calls) == ["g1", "g2"]
    assert ("T1", "T1:2") not in api.history_calls  # never paged past the window


def test_modifier_games_are_skipped_and_counted(env):
    api = FakeApi(histories={"T1": [([entry("g1"), entry("g2")], None)]},
                  games={"g1": record("g1"), "g2": record("g2", randomSpawn=True)})
    out = crawler(api, env, teachers=1).run()
    assert out["skipped"]["modifiers"] == 1 and out["fetched"] == 1
    assert env["index"].has_game("g1") and not env["index"].has_game("g2")


def test_filtered_entries_are_never_fetched(env):
    api = FakeApi(histories={"T1": [([entry("g1", type="Private"), entry("g2", durationSeconds=100), {"x": 1}], None)]})
    out = crawler(api, env, teachers=1).run()
    assert api.game_calls == [] and out["skipped"]["filter"] == 2 and out["skipped"]["malformed"] == 1


def test_resume_skips_stored_and_continues_cursor(env):
    gids = [f"g{i}" for i in range(6)]
    api = FakeApi(histories={"T1": [([entry(g) for g in gids], None)]}, games={g: record(g) for g in gids})
    stop = threading.Event()
    api.on_game = lambda gid: stop.set() if len(api.game_calls) == 3 else None
    first = crawler(api, env, teachers=1, stop=stop).run()
    assert first["phase"] == "stopped" and first["fetched"] == 3
    api2 = FakeApi(histories=api.histories, games=api.games)
    second = crawler(api2, env, teachers=1).run()
    assert second["phase"] == "done"
    assert sorted(api2.game_calls) == sorted(set(gids) - set(api.game_calls))  # no refetch
    assert env["index"].stats()["games"] == 6


def test_stop_event_ends_with_phase_stopped(env):
    stop = threading.Event()
    stop.set()
    assert crawler(FakeApi(), env, stop=stop).run()["phase"] == "stopped"


def test_forbidden_ends_with_phase_error(env):
    out = crawler(FakeApi(forbid=True), env).run()
    assert out["phase"] == "error" and "403" in out["error"]
    assert env["progress_file"].exists()


def test_seed_players_skip_the_leaderboard(env):
    api = FakeApi(forbid=True, histories={"S1": [([entry("g1")], None)]}, games={"g1": record("g1")})
    out = crawler(api, env, seed_players=["S1"]).run()
    assert out["phase"] == "done" and out["fetched"] == 1
    assert env["index"].player_elo("S1") == 2010  # rating learnt from the history


def test_target_games_caps_fetching(env):
    gids = [f"g{i}" for i in range(5)]
    api = FakeApi(histories={"T1": [([entry(g) for g in gids], None)]}, games={g: record(g) for g in gids})
    out = crawler(api, env, teachers=1, target_games=2).run()
    assert out["fetched"] == 2 and len(api.game_calls) == 2
