"""SQLite index of crawled games (spec §8.1 step 4): `data/index.sqlite`.

One connection per Index, guarded by a lock so the service's threads can share it.
WAL mode so readers (the app) never block the crawler.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ofai import paths

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY,
  commit_sha TEXT NOT NULL,
  map TEXT NOT NULL,
  start INTEGER,               -- ms since epoch
  duration_s REAL,
  players INTEGER,
  winner_client TEXT,
  in_window INTEGER NOT NULL DEFAULT 0,
  path TEXT NOT NULL,
  replay_ok INTEGER,           -- NULL unchecked, 1 ok, 0 rejected
  ranked INTEGER               -- players the seats' placements are out of
);
CREATE INDEX IF NOT EXISTS games_commit ON games(commit_sha);
CREATE INDEX IF NOT EXISTS games_map ON games(map);
CREATE INDEX IF NOT EXISTS games_start ON games(start);
CREATE TABLE IF NOT EXISTS seats (
  game TEXT NOT NULL,
  client_id TEXT NOT NULL,
  public_id TEXT,
  username TEXT,
  is_teacher INTEGER NOT NULL DEFAULT 0,
  placement INTEGER,
  PRIMARY KEY (game, client_id)
);
CREATE INDEX IF NOT EXISTS seats_public ON seats(public_id);
CREATE TABLE IF NOT EXISTS players (
  public_id TEXT PRIMARY KEY,
  ffa_elo REAL,
  source TEXT,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS queue (
  game_id TEXT PRIMARY KEY,
  source TEXT,
  queued_at TEXT
);
CREATE TABLE IF NOT EXISTS cursors (
  public_id TEXT PRIMARY KEY,
  cursor TEXT,
  done_before TEXT,
  updated_at TEXT
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _winner_client(winner: object) -> str | None:
    if isinstance(winner, list) and len(winner) >= 2 and winner[0] == "player" and isinstance(winner[1], str):
        return winner[1]
    return None


class Index:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (paths.DATA / "index.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(games)")}
            if "ranked" not in cols:  # indexes made before placements existed
                self._db.execute("ALTER TABLE games ADD COLUMN ranked INTEGER")
            self._db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema', ?)", (str(SCHEMA_VERSION),))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # -- games ---------------------------------------------------------------

    def add_record(self, rec: dict, path: Path, *, teachers: set[str]) -> None:
        info = rec["info"]
        cfg = info.get("config", {})
        players = info.get("players", [])
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._db.execute(
                    """INSERT INTO games(id, commit_sha, map, start, duration_s, players, winner_client, path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET commit_sha=excluded.commit_sha, map=excluded.map,
                         start=excluded.start, duration_s=excluded.duration_s, players=excluded.players,
                         winner_client=excluded.winner_client, path=excluded.path""",
                    (
                        info["gameID"],
                        rec.get("gitCommit") or "unknown",
                        str(cfg.get("gameMap", "unknown")),
                        info.get("start"),
                        info.get("duration"),
                        len(players),
                        _winner_client(info.get("winner")),
                        str(path),
                    ),
                )
                for p in players:
                    pid = p.get("publicID") or p.get("publicId")
                    self._db.execute(
                        """INSERT INTO seats(game, client_id, public_id, username, is_teacher)
                           VALUES (?, ?, ?, ?, ?)
                           ON CONFLICT(game, client_id) DO UPDATE SET public_id=excluded.public_id,
                             username=excluded.username, is_teacher=excluded.is_teacher""",
                        (info["gameID"], p.get("clientID"), pid, p.get("username"), int(pid in teachers if pid else 0)),
                    )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def count_games(self) -> int:
        return self._q("SELECT count(*) FROM games")[0][0]

    def has_game(self, game_id: str) -> bool:
        return bool(self._q("SELECT 1 FROM games WHERE id = ?", (game_id,)))

    def mark_replay(self, game_id: str, ok: bool) -> None:
        self._q("UPDATE games SET replay_ok = ? WHERE id = ?", (int(ok), game_id))

    def set_placements(self, game_id: str, placements: dict[str, int], ranked: int | None = None) -> None:
        with self._lock:
            if ranked is not None:
                self._db.execute("UPDATE games SET ranked = ? WHERE id = ?", (ranked, game_id))
            for client_id, place in placements.items():
                self._db.execute("UPDATE seats SET placement = ? WHERE game = ? AND client_id = ?", (place, game_id, client_id))

    def game(self, game_id: str) -> dict | None:
        rows = self.games(game_id=game_id, limit=1)
        return rows[0] if rows else None

    def games(
        self,
        *,
        map: str | None = None,
        commit: str | None = None,
        game_id: str | None = None,
        only_in_window: bool = False,
        replay_ok: bool | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        where, args = [], []
        if map:
            where.append("g.map = ?")
            args.append(map)
        if commit:
            where.append("g.commit_sha = ?")
            args.append(commit)
        if game_id:
            where.append("g.id = ?")
            args.append(game_id)
        if only_in_window:
            where.append("g.in_window = 1")
        if replay_ok is not None:
            where.append("g.replay_ok = ?")
            args.append(int(replay_ok))
        sql = f"""SELECT g.*, (SELECT group_concat(s.username, '|') FROM seats s WHERE s.game = g.id AND s.is_teacher = 1)
                    AS teacher_names,
                    (SELECT count(*) FROM seats s WHERE s.game = g.id AND s.is_teacher = 1) AS teacher_seats
                  FROM games g {"WHERE " + " AND ".join(where) if where else ""}
                  ORDER BY g.start DESC, g.id LIMIT ? OFFSET ?"""
        out = []
        for r in self._q(sql, (*args, limit, offset)):
            d = dict(r)
            d["commit"] = d.pop("commit_sha")
            d["teachers"] = [n for n in (d.pop("teacher_names") or "").split("|") if n]
            d["replay_ok"] = None if d["replay_ok"] is None else bool(d["replay_ok"])
            d["in_window"] = bool(d["in_window"])
            out.append(d)
        return out

    def teacher_seats(self, game_id: str) -> list[dict]:
        rows = self._q(
            """SELECT s.client_id, s.public_id, s.username, s.placement, p.ffa_elo, g.ranked
               FROM seats s LEFT JOIN players p ON p.public_id = s.public_id JOIN games g ON g.id = s.game
               WHERE s.game = ? AND s.is_teacher = 1 ORDER BY p.ffa_elo DESC""",
            (game_id,),
        )
        return [dict(r) for r in rows]

    # -- players and cursors -------------------------------------------------

    def upsert_player(self, public_id: str, ffa_elo: float | None, source: str) -> None:
        self._q(
            """INSERT INTO players(public_id, ffa_elo, source, updated_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(public_id) DO UPDATE SET
                 ffa_elo = COALESCE(MAX(excluded.ffa_elo, players.ffa_elo), excluded.ffa_elo, players.ffa_elo),
                 source = excluded.source, updated_at = excluded.updated_at""",
            (public_id, ffa_elo, source, _now_iso()),
        )

    def teachers(self, limit: int) -> list[str]:
        return [r["public_id"] for r in self._q(
            "SELECT public_id FROM players WHERE ffa_elo IS NOT NULL ORDER BY ffa_elo DESC, public_id LIMIT ?", (limit,)
        )]

    def player_elo(self, public_id: str) -> float | None:
        rows = self._q("SELECT ffa_elo FROM players WHERE public_id = ?", (public_id,))
        return rows[0]["ffa_elo"] if rows else None

    def get_cursor(self, public_id: str) -> str | None:
        rows = self._q("SELECT cursor FROM cursors WHERE public_id = ?", (public_id,))
        return rows[0]["cursor"] if rows else None

    def cursor_state(self, public_id: str) -> tuple[str | None, str | None] | None:
        rows = self._q("SELECT cursor, done_before FROM cursors WHERE public_id = ?", (public_id,))
        return (rows[0]["cursor"], rows[0]["done_before"]) if rows else None

    def set_cursor(self, public_id: str, cursor: str | None, done_before: str | None) -> None:
        self._q(
            """INSERT INTO cursors(public_id, cursor, done_before, updated_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(public_id) DO UPDATE SET cursor=excluded.cursor, done_before=excluded.done_before,
                 updated_at=excluded.updated_at""",
            (public_id, cursor, done_before, _now_iso()),
        )

    # -- fetch queue (survives a killed crawl) ---------------------------------

    def queue_add(self, game_id: str, source: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO queue(game_id, source, queued_at) VALUES (?, ?, ?)", (game_id, source, _now_iso())
            )
            return cur.rowcount > 0

    def queue_list(self, limit: int = 1_000_000) -> list[str]:
        return [r[0] for r in self._q("SELECT game_id FROM queue ORDER BY queued_at, game_id LIMIT ?", (limit,))]

    def queue_remove(self, game_id: str) -> None:
        self._q("DELETE FROM queue WHERE game_id = ?", (game_id,))

    def queue_size(self) -> int:
        return self._q("SELECT count(*) FROM queue")[0][0]

    # -- version window ------------------------------------------------------

    def window_commits(self, now: datetime, days: int = 21, max_commits: int = 10) -> list[str]:
        """Commits seen in games started within `days`, newest first by first-seen game start, at most `max_commits`."""
        cutoff_ms = int((now - timedelta(days=days)).timestamp() * 1000)
        rows = self._q(
            """SELECT commit_sha, MIN(start) AS first_seen FROM games
               WHERE commit_sha IN (SELECT DISTINCT commit_sha FROM games WHERE start >= ?)
               GROUP BY commit_sha ORDER BY first_seen DESC LIMIT ?""",
            (cutoff_ms, max_commits),
        )
        return [r["commit_sha"] for r in rows]

    def refresh_window(self, now: datetime, days: int = 21, max_commits: int = 10) -> list[str]:
        commits = self.window_commits(now, days, max_commits)
        cutoff_ms = int((now - timedelta(days=days)).timestamp() * 1000)
        with self._lock:
            self._db.execute("UPDATE games SET in_window = 0")
            if commits:
                marks = ",".join("?" * len(commits))
                self._db.execute(
                    f"UPDATE games SET in_window = 1 WHERE start >= ? AND commit_sha IN ({marks})", (cutoff_ms, *commits)
                )
        return commits

    # -- stats ---------------------------------------------------------------

    def stats(self) -> dict:
        one = lambda sql: self._q(sql)[0][0]  # noqa: E731
        return {
            "games": one("SELECT count(*) FROM games"),
            "inWindow": one("SELECT count(*) FROM games WHERE in_window = 1"),
            "teachers": one("SELECT count(DISTINCT public_id) FROM seats WHERE is_teacher = 1"),
            "knownPlayers": one("SELECT count(*) FROM players"),
            "uniqueTeacherGames": one("SELECT count(DISTINCT game) FROM seats WHERE is_teacher = 1"),
            "replay": {
                "ok": one("SELECT count(*) FROM games WHERE replay_ok = 1"),
                "rejected": one("SELECT count(*) FROM games WHERE replay_ok = 0"),
                "unchecked": one("SELECT count(*) FROM games WHERE replay_ok IS NULL"),
            },
            "perMap": {r[0]: r[1] for r in self._q("SELECT map, count(*) FROM games GROUP BY map ORDER BY 2 DESC")},
            "perCommit": {
                r[0]: {"games": r[1], "firstSeen": r[2], "inWindow": r[3], "replayOk": r[4], "replayRejected": r[5]}
                for r in self._q(
                    """SELECT commit_sha, count(*), MIN(start), SUM(in_window), SUM(replay_ok = 1), SUM(replay_ok = 0)
                       FROM games GROUP BY commit_sha ORDER BY MIN(start) DESC"""
                )
            },
        }
