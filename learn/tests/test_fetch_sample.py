from pathlib import Path

import pytest

from ofai import cli
from ofai.data import records
from ofai.data.api import ApiForbidden


def rec(gid: str, commit: str = "c1", **cfg) -> dict:
    return {"gitCommit": commit, "info": {"gameID": gid, "config": {"gameMapSize": "Normal", **cfg}}, "turns": []}


class FakeApi:
    requests = 0

    def __init__(self, board=None, history=None, games=None, forbid=False):
        self.board = board or {"ffa": [{"public_id": "P1"}, {"public_id": "P2"}]}
        self.history = history or {}
        self.games = games or {}
        self.forbid = forbid

    def leaderboard(self, page):
        if self.forbid:
            raise ApiForbidden("403 challenge")
        return self.board

    def player_games(self, pid, cursor=None):
        return self.history.get(pid, []), None

    def game(self, gid):
        return self.games[gid]


def entry(gid, **kw):
    return {"gameId": gid, "type": "Public", "mode": "Free For All", "durationSeconds": 900, **kw}


@pytest.fixture(autouse=True)
def records_root(tmp_path, monkeypatch):
    monkeypatch.setattr(records.paths, "RECORDS", tmp_path / "records")
    return tmp_path / "records"


def test_stores_clean_records_and_counts(records_root: Path):
    api = FakeApi(
        history={
            "P1": [entry("g1"), entry("g2"), {"broken": True}, entry("g3", type="Private")],
            "P2": [entry("g1"), entry("g4")],
        },
        games={"g1": rec("g1"), "g2": rec("g2", randomSpawn=True), "g4": rec("g4", "c2")},
    )
    s = cli.fetch_sample(api, players=2, games=10)
    assert s["error"] is None
    assert s["fetched"] == 2
    assert s["skipped"] == {"malformed": 1, "filter": 1, "modifiers": 1}
    assert s["modifiers"] == {"randomSpawn": 1}
    assert s["commits"] == {"c1": 1, "c2": 1}
    assert (records_root / "c1" / "g1.json.gz").exists()
    assert records.read_record(records_root / "c2" / "g4.json.gz")["info"]["gameID"] == "g4"


def test_already_stored_is_not_refetched(records_root: Path):
    records.write_record(rec("g1"))
    api = FakeApi(history={"P1": [entry("g1")]}, games={})
    s = cli.fetch_sample(api, players=1, games=10)
    assert s["skipped"] == {"stored": 1}


def test_forbidden_is_reported_not_raised():
    s = cli.fetch_sample(FakeApi(forbid=True), players=1, games=1)
    assert "403" in s["error"]
    assert s["fetched"] == 0


def test_seed_players_skip_the_leaderboard():
    api = FakeApi(forbid=True, history={"X": [entry("g9")]}, games={"g9": rec("g9")})
    s = cli.fetch_sample(api, players=5, games=5, seed_players=["X"])
    assert s["error"] is None and s["fetched"] == 1
