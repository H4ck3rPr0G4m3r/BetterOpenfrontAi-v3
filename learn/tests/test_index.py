from datetime import datetime, timedelta, timezone
from pathlib import Path

from ofai.data.index import Index

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def rec(gid: str, *, commit="c1", map="World", start=None, players=None, winner=None) -> dict:
    players = players or [
        {"clientID": "AAAAAAAA", "username": "alice", "publicID": "pubA"},
        {"clientID": "BBBBBBBB", "username": "bob", "publicID": "pubB"},
        {"clientID": "CCCCCCCC", "username": "anon"},
    ]
    return {
        "gitCommit": commit,
        "info": {"gameID": gid, "config": {"gameMap": map}, "players": players, "start": start or ms(NOW),
                 "duration": 900, "winner": winner or ["player", "AAAAAAAA"]},
    }


def test_add_record_and_stats(tmp_path: Path):
    ix = Index(tmp_path / "i.sqlite")
    ix.add_record(rec("g1", map="World"), tmp_path / "g1", teachers={"pubA"})
    ix.add_record(rec("g2", map="Pangaea"), tmp_path / "g2", teachers={"pubA", "pubB"})
    s = ix.stats()
    assert s["games"] == 2 and s["perMap"] == {"World": 1, "Pangaea": 1}
    assert s["teachers"] == 2 and s["uniqueTeacherGames"] == 2
    g = ix.game("g1")
    assert g["winner_client"] == "AAAAAAAA" and g["teachers"] == ["alice"] and g["players"] == 3
    assert ix.game("g2")["teacher_seats"] == 2


def test_add_is_idempotent(tmp_path: Path):
    ix = Index(tmp_path / "i.sqlite")
    for _ in range(2):
        ix.add_record(rec("g1"), tmp_path / "g1", teachers={"pubA"})
    assert ix.stats()["games"] == 1
    assert len(ix.teacher_seats("g1")) == 1


def test_window_commits_last_21_days_cap_10(tmp_path: Path):
    ix = Index(tmp_path / "i.sqlite")
    for d in range(12):  # commit k first seen d days ago (k=0 newest)
        ix.add_record(rec(f"g{d}", commit=f"k{d}", start=ms(NOW - timedelta(days=d, hours=1))), tmp_path, teachers=set())
    ix.add_record(rec("old", commit="ancient", start=ms(NOW - timedelta(days=30))), tmp_path, teachers=set())
    w = ix.window_commits(NOW)
    assert w == [f"k{d}" for d in range(10)]
    assert "ancient" not in w
    ix.refresh_window(NOW)
    assert ix.stats()["inWindow"] == 10
    assert ix.game("old")["in_window"] is False


def test_cursor_roundtrip(tmp_path: Path):
    ix = Index(tmp_path / "i.sqlite")
    assert ix.get_cursor("p") is None
    ix.set_cursor("p", "abc", "2026-11-01")
    assert ix.get_cursor("p") == "abc"
    assert ix.cursor_state("p") == ("abc", "2026-11-01")
    ix.set_cursor("p", None, "2026-11-02")
    assert ix.get_cursor("p") is None


def test_teachers_ordered_by_elo(tmp_path: Path):
    ix = Index(tmp_path / "i.sqlite")
    ix.upsert_player("low", 1500, "ffa")
    ix.upsert_player("high", 2400, "ffa")
    ix.upsert_player("mid", 2000, "ffaSeason")
    ix.upsert_player("none", None, "history")
    assert ix.teachers(2) == ["high", "mid"]
    ix.upsert_player("low", 2600, "history")  # a better rating seen later wins
    assert ix.teachers(1) == ["low"]
    ix.upsert_player("low", 1000, "history")  # a worse one never lowers it
    assert ix.player_elo("low") == 2600


def test_mark_replay_and_filters(tmp_path: Path):
    ix = Index(tmp_path / "i.sqlite")
    ix.add_record(rec("g1", map="World", commit="a"), tmp_path, teachers=set())
    ix.add_record(rec("g2", map="Sol", commit="b"), tmp_path, teachers=set())
    ix.mark_replay("g1", True)
    ix.mark_replay("g2", False)
    assert [g["id"] for g in ix.games(map="Sol")] == ["g2"]
    assert [g["id"] for g in ix.games(commit="a")] == ["g1"]
    assert [g["id"] for g in ix.games(replay_ok=True)] == ["g1"]
    assert ix.stats()["replay"] == {"ok": 1, "rejected": 1, "unchecked": 0}


def test_placements_and_ranked_count(tmp_path: Path):
    ix = Index(tmp_path / "i.sqlite")
    ix.add_record(rec("g1"), tmp_path / "g1", teachers={"pubA", "pubB"})
    ix.set_placements("g1", {"AAAAAAAA": 1, "BBBBBBBB": 7}, ranked=40)
    seats = {s["client_id"]: s for s in ix.teacher_seats("g1")}
    assert seats["AAAAAAAA"]["placement"] == 1 and seats["BBBBBBBB"]["placement"] == 7
    assert seats["AAAAAAAA"]["ranked"] == 40


def test_old_index_gains_ranked_column(tmp_path: Path):
    import sqlite3

    path = tmp_path / "i.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE games (id TEXT PRIMARY KEY, commit_sha TEXT NOT NULL, map TEXT NOT NULL, start INTEGER,"
               " duration_s REAL, players INTEGER, winner_client TEXT, in_window INTEGER NOT NULL DEFAULT 0,"
               " path TEXT NOT NULL, replay_ok INTEGER)")
    db.commit()
    db.close()
    ix = Index(path)
    ix.add_record(rec("g1"), tmp_path / "g1", teachers={"pubA"})
    ix.set_placements("g1", {"AAAAAAAA": 2}, ranked=5)
    assert ix.teacher_seats("g1")[0]["ranked"] == 5
