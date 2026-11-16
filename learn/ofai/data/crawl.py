"""Teacher crawler (spec §8.1): leaderboards -> teacher histories -> records.

Resumable at every step: teachers and ratings live in the index, history
walks keep a cursor per teacher, and queued game IDs are persisted, so a
killed crawl restarts where it stopped and never refetches a stored record
(Review Focus 1). All requests go through one `OpenFrontApi`, which holds the
1 request/s limit and stops on 403.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ofai import paths
from ofai.data.api import ApiError, ApiForbidden, ApiNotFound
from ofai.data.filters import is_wellformed_summary, modifiers, wanted_summary
from ofai.data.index import Index
from ofai.data.records import find_record, write_record


@dataclass
class CrawlConfig:
    teachers: int = 200
    days: int = 21
    target_games: int = 2000
    max_history_pages: int = 50
    max_commits: int = 10
    # Public IDs to treat as teachers when the leaderboard can't be read (e.g. it is challenged).
    seed_players: list[str] = field(default_factory=list)


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _rating_after(entry: dict) -> float | None:
    r = entry.get("ffaRating")
    if isinstance(r, dict):
        for k in ("after", "before"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    if isinstance(r, (int, float)):
        return float(r)
    return None


class Crawler:
    def __init__(
        self,
        api,
        index: Index,
        cfg: CrawlConfig | None = None,
        progress: Callable[[dict], None] = lambda p: None,
        stop: threading.Event | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        progress_file: Path | None = None,
        records_root: Path | None = None,
    ) -> None:
        self.api = api
        self.index = index
        self.cfg = cfg or CrawlConfig()
        self._progress_cb = progress
        self.stop = stop or threading.Event()
        self.now = now
        self.progress_file = progress_file or (paths.DATA / "crawl_progress.json")
        self.records_root = records_root
        self.state: dict = {
            "phase": "teachers", "teachers": 0, "historiesDone": 0, "queued": 0, "fetched": 0,
            "skipped": Counter(), "targetGames": self.cfg.target_games, "error": None, "updatedAt": None,
        }
        self._requests_at_save = 0

    # -- progress ------------------------------------------------------------

    def _emit(self, force: bool = False) -> None:
        self.state["updatedAt"] = self.now().isoformat(timespec="seconds")
        self.state["queued"] = self.index.queue_size()
        snap = {**self.state, "skipped": dict(self.state["skipped"])}
        requests = getattr(self.api, "requests", 0)
        if force or requests - self._requests_at_save >= 10:
            self._requests_at_save = requests
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.progress_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
            os.replace(tmp, self.progress_file)
        self._progress_cb(snap)

    def _stopped(self) -> bool:
        if self.stop.is_set():
            self.state["phase"] = "stopped"
            return True
        return False

    # -- phases --------------------------------------------------------------

    def run(self) -> dict:
        try:
            for phase in (self._teachers, self._histories, self._records):
                if self._stopped():
                    break
                phase()
                if self._stopped():
                    break
            else:
                self.state["phase"] = "done"
        except ApiForbidden as e:
            self.state["phase"] = "error"
            self.state["error"] = str(e)
        except ApiError as e:
            self.state["phase"] = "error"
            self.state["error"] = f"API failed: {e}"
        self.index.refresh_window(self.now(), self.cfg.days, self.cfg.max_commits)
        self._emit(force=True)
        return {**self.state, "skipped": dict(self.state["skipped"])}

    def _teachers(self) -> None:
        self.state["phase"] = "teachers"
        self._emit()
        if self.cfg.seed_players:
            for pid in self.cfg.seed_players:
                self.index.upsert_player(pid, self.index.player_elo(pid) or 0.0, "seed")
        else:
            for board in ("ffa", "ffaSeason"):
                page = 1
                while not self._stopped():
                    try:
                        body = self.api.leaderboard(page)
                    except ApiForbidden:
                        raise
                    except ApiError:
                        break  # the API answers a page past the end with a 400
                    entries = body.get(board) if isinstance(body, dict) else None
                    if not entries:
                        break
                    for e in entries:
                        if isinstance(e, dict) and isinstance(e.get("public_id"), str):
                            elo = e.get("elo")
                            self.index.upsert_player(e["public_id"], float(elo) if isinstance(elo, (int, float)) else None, board)
                    page += 1
                    self._emit()
        self.teacher_ids = self.index.teachers(self.cfg.teachers)
        if self.cfg.seed_players:
            self.teacher_ids = list(dict.fromkeys([*self.cfg.seed_players, *self.teacher_ids]))[: max(self.cfg.teachers, len(self.cfg.seed_players))]
        self.state["teachers"] = len(self.teacher_ids)
        self._emit()

    def _histories(self) -> None:
        self.state["phase"] = "histories"
        window_start = self.now() - timedelta(days=self.cfg.days)
        walk_started = self.now().isoformat(timespec="seconds")
        for pid in self.teacher_ids:
            if self._stopped():
                return
            state = self.index.cursor_state(pid)
            cursor = state[0] if state else None
            stop_at = window_start
            if state and state[0] is None and state[1]:
                # Finished before: only entries newer than the last finished walk are new.
                done = _parse_iso(state[1])
                if done and done > stop_at:
                    stop_at = done - timedelta(hours=1)
            finished = False
            for _ in range(self.cfg.max_history_pages):
                if self._stopped():
                    return
                try:
                    entries, next_cursor = self.api.player_games(pid, cursor)
                except ApiNotFound:
                    self.state["skipped"]["unknown_player"] += 1
                    finished = True
                    break
                reached_old = False
                for entry in entries:
                    if not is_wellformed_summary(entry):
                        self.state["skipped"]["malformed"] += 1
                        continue
                    started = _parse_iso(entry.get("start"))
                    if started is not None and started < stop_at:
                        reached_old = True
                        break
                    rating = _rating_after(entry)
                    if rating is not None:
                        self.index.upsert_player(pid, rating, "history")
                    if not wanted_summary(entry):
                        self.state["skipped"]["filter"] += 1
                        continue
                    gid = entry["gameId"]
                    if self.index.has_game(gid):
                        continue
                    self.index.queue_add(gid, pid)
                cursor = next_cursor
                if reached_old or cursor is None:
                    finished = True
                    break
                self.index.set_cursor(pid, cursor, state[1] if state else None)
                self._emit()
            if finished:
                self.index.set_cursor(pid, None, walk_started)
            self.state["historiesDone"] += 1
            self._emit()

    def _records(self) -> None:
        self.state["phase"] = "records"
        teachers = set(self.teacher_ids)
        for gid in self.index.queue_list():
            if self._stopped():
                return
            if self.index.count_games() >= self.cfg.target_games:
                break
            existing = find_record(gid, self.records_root)
            if existing is not None or self.index.has_game(gid):
                self.state["skipped"]["stored"] += 1
                self.index.queue_remove(gid)
                continue
            try:
                record = self.api.game(gid)
            except ApiNotFound:
                self.state["skipped"]["not_found"] += 1
                self.index.queue_remove(gid)
                continue
            found = modifiers(record.get("info", {}).get("config", {}))
            if found:
                self.state["skipped"]["modifiers"] += 1
                self.index.queue_remove(gid)
                continue
            path = write_record(record, self.records_root)
            self.index.add_record(record, path, teachers=teachers)
            self.index.queue_remove(gid)
            self.state["fetched"] += 1
            self._emit()
