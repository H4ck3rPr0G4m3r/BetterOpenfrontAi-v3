"""Teacher sequences for BC (M3 Task 4).

One SimPool (replay role) per engine commit. Every pool row is a lane; the
stream keeps each pool's record queue filled with that commit's games of the
split part (shuffled per epoch, teachers weighted by rating and placement)
and yields `SeqBatch`es of `seq_len` consecutive decision steps for every
lane: numpy arrays [T, lanes, ...] plus `fresh` and `valid` [T, lanes].
Lane order is fixed (pools in commit order, rows in pool order), so a lane's
LSTM state carries from one SeqBatch to the next.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np

from ofai.bc.split import load_split
from ofai.bc.weights import placement_value, teacher_weight
from ofai.bridge import protocol as P
from ofai.bridge.pool import SimWorkerDied
from ofai.data.index import Index

SeqBatch = dict[str, Any]


class Pool(Protocol):
    rows: int

    def load(self, items: list[dict]) -> None: ...
    def next(self) -> dict: ...


def elo_percentiles(index: Index) -> tuple[float, float]:
    elos = sorted(r["ffa_elo"] for r in index._q("SELECT ffa_elo FROM players WHERE ffa_elo IS NOT NULL"))
    if not elos:
        return 0.0, 1.0
    q = lambda f: elos[min(len(elos) - 1, int(f * (len(elos) - 1)))]  # noqa: E731
    return q(0.5), q(0.9)


class TeacherStream:
    def __init__(self, index: Index, split: str | dict, part: str, pools: dict[str, Pool], lanes: int | None = None,
                 seq_len: int = 64, seed: int = 0, refill_below: int | None = None) -> None:
        if part not in ("train", "heldout"):
            raise ValueError(part)
        self.index = index
        self.pools = dict(sorted(pools.items()))
        self.lanes = sum(p.rows for p in self.pools.values())
        if lanes is not None and lanes != self.lanes:
            raise ValueError(f"pools have {self.lanes} lanes, asked for {lanes}")
        self.seq_len = seq_len
        self.seed = seed
        self.epoch = 0
        sp = load_split(split) if isinstance(split, str) else split
        self.games: dict[str, list[str]] = {c: [] for c in self.pools}
        for gid in sp[part]:
            g = index.game(gid)
            if g is not None and g["commit"] in self.games:
                self.games[g["commit"]].append(gid)
        self.p50, self.p90 = elo_percentiles(index)
        self.order: dict[str, list[str]] = {}
        self.cursor: dict[str, int] = {c: 0 for c in self.pools}
        self.queued: dict[str, int] = {c: 0 for c in self.pools}
        self.refill_below = refill_below
        self.loaded_total = 0
        self._shuffle()

    # -- resume state ---------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "cursor": dict(self.cursor)}

    def load_state_dict(self, d: dict) -> None:
        self.epoch = int(d.get("epoch", 0))
        self._shuffle()
        self.cursor.update({c: int(v) for c, v in d.get("cursor", {}).items() if c in self.cursor})

    # -- queue ------------------------------------------------------------------------

    def _shuffle(self) -> None:
        for c, games in self.games.items():
            order = list(games)
            random.Random(f"{self.seed}:{self.epoch}:{c}").shuffle(order)
            self.order[c] = order

    def item(self, game_id: str) -> dict | None:
        g = self.index.game(game_id)
        if g is None:
            return None
        teachers = []
        for s in self.index.teacher_seats(game_id):
            value = placement_value(s.get("placement"), s.get("ranked"))
            teachers.append({"clientID": s["client_id"], "value": value,
                             "weight": teacher_weight(s.get("ffa_elo"), value, self.p50, self.p90)})
        return {"path": g["path"], "teachers": teachers} if teachers else None

    def _refill(self, commit: str, pool: Pool) -> None:
        want = self.refill_below if self.refill_below is not None else max(4, pool.rows)
        if self.queued[commit] >= want or not self.order[commit]:
            return
        items = []
        while self.queued[commit] + len(items) < 2 * want:
            if self.cursor[commit] >= len(self.order[commit]):
                if all(self.cursor[c] >= len(o) for c, o in self.order.items() if o):
                    self.epoch += 1
                    self._shuffle()
                    self.cursor = {c: 0 for c in self.cursor}
                else:
                    break  # this commit's part is done for the epoch; others still running
            gid = self.order[commit][self.cursor[commit]]
            self.cursor[commit] += 1
            it = self.item(gid)
            if it is not None:
                items.append(it)
        if items:
            pool.load(items)
            self.queued[commit] += len(items)
            self.loaded_total += len(items)

    def _note_queue(self, commit: str, batch: dict) -> None:
        infos = batch.get("info")
        if isinstance(infos, list) and infos and all(isinstance(i, dict) and "queued" in i for i in infos):
            self.queued[commit] = sum(int(i["queued"]) for i in infos)

    def _guarded(self, commit: str, fn):
        try:
            return fn()
        except SimWorkerDied as e:
            e.commit = commit  # type: ignore[attr-defined]
            # Drain the other pools' in-flight replies so their workers stay usable.
            for c, p in self.pools.items():
                if c != commit and getattr(p, "_sent", None) is not None:
                    try:
                        p.recv_next()
                    except SimWorkerDied:
                        pass
                    p._sent = None
            raise

    def lane_range(self, commit: str, worker: int | None = None) -> tuple[int, int]:
        """Lanes [start, end) of one pool, or of one worker of it."""
        start = 0
        for c, p in self.pools.items():
            if c == commit:
                if worker is None:
                    return start, start + p.rows
                per = p.rows // max(1, len(getattr(p, "workers", [None])))
                return start + worker * per, start + (worker + 1) * per
            start += p.rows
        raise KeyError(commit)

    def restart(self, err: SimWorkerDied) -> tuple[int, int]:
        """Restarts the dead worker; returns the lanes whose LSTM state must restart.

        The partial sequence is dropped, so every pool's lanes jump ahead by up to
        seq_len steps: the trainer treats all lanes as fresh after a death."""
        commit = getattr(err, "commit")
        self.pools[commit].restart(err.index)
        self.queued[commit] = 0  # the new worker's queue is empty
        return self.lane_range(commit, err.index)

    def close(self) -> None:
        for p in self.pools.values():
            close = getattr(p, "close", None)
            if close:
                close()

    # -- batches ----------------------------------------------------------------------

    def __iter__(self) -> Iterator[SeqBatch]:
        while True:
            yield self.next_seq()

    def next_seq(self) -> SeqBatch:
        steps: list[dict[str, np.ndarray]] = []
        for _ in range(self.seq_len):
            # Start every pool's step before reading any reply: pools of different
            # commits then simulate in parallel instead of one after another.
            split = all(hasattr(p, "send_next") for p in self.pools.values())
            for c, pool in self.pools.items():
                self._refill(c, pool)
                if split:
                    self._guarded(c, pool.send_next)
            parts = []
            for c, pool in self.pools.items():
                b = self._guarded(c, pool.recv_next if split else pool.next)
                self._note_queue(c, b)
                parts.append(b)
            steps.append({k: np.concatenate([p[k] for p in parts], 0) for k in parts[0] if k != "info"})
        out: SeqBatch = {k: np.stack([s[k] for s in steps], 0) for k in steps[0]}
        flags = out["flags"]
        out["fresh"] = (flags & P.FLAG_FRESH) != 0
        out["valid"] = (flags & P.FLAG_VALID) != 0
        return out
