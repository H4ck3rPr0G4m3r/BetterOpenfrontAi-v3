from datetime import datetime, timezone

import numpy as np
import pytest

from ofai.bc import split as split_mod
from ofai.bc.split import heldout_fraction, make_split
from ofai.bc.stream import TeacherStream
from ofai.bridge import protocol as P
from ofai.data.index import Index

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def _index(tmp_path, n=40, commits=("c1",)) -> Index:
    ix = Index(tmp_path / "i.sqlite")
    for i in range(n):
        gid = f"game{i:04d}"
        rec = {"gitCommit": commits[i % len(commits)],
               "info": {"gameID": gid, "config": {"gameMap": "World"}, "start": int(NOW.timestamp() * 1000) - 3600_000,
                        "duration": 900, "players": [{"clientID": "AAAAAAAA", "username": "a", "publicID": "pA"},
                                                     {"clientID": "BBBBBBBB", "username": "b", "publicID": "pB"}]}}
        ix.add_record(rec, tmp_path / f"{gid}.json.gz", teachers={"pA", "pB"})
        ix.mark_replay(gid, True)
        ix.set_placements(gid, {"AAAAAAAA": 1, "BBBBBBBB": 3}, ranked=5)
    ix.upsert_player("pA", 2200, "test")
    ix.upsert_player("pB", 1500, "test")
    ix.refresh_window(NOW)
    return ix


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(split_mod.paths, "DATA", tmp_path / "data")


def test_split_is_stable_and_disjoint(tmp_path):
    ix = _index(tmp_path, 200)
    p = make_split(ix, "s1", heldout=0.1)
    d = split_mod.load_split("s1")
    assert set(d["train"]).isdisjoint(d["heldout"])
    assert len(d["train"]) + len(d["heldout"]) == 200
    assert 5 <= len(d["heldout"]) <= 40
    assert all(heldout_fraction(g) < 0.1 for g in d["heldout"])
    p.unlink()
    make_split(ix, "s1", heldout=0.1)
    assert split_mod.load_split("s1")["heldout"] == d["heldout"]


def test_split_refuses_overwrite(tmp_path):
    ix = _index(tmp_path, 5)
    make_split(ix, "s2")
    with pytest.raises(FileExistsError):
        make_split(ix, "s2")


class FakePool:
    """Two rows; each NEXT stamps every row with (lane id, step counter); FRESH every 5 steps on row 1."""

    def __init__(self, lane_base: int, rows: int = 2) -> None:
        self.rows = rows
        self.lane_base = lane_base
        self.step = 0
        self.queue: list[dict] = []
        self.loads = 0

    def load(self, items):
        self.loads += 1
        self.queue.extend(items)

    def next(self):
        if self.step % 3 == 0 and self.queue:
            self.queue.pop(0)  # a game starts
        flags = np.full(self.rows, P.FLAG_VALID, dtype=np.uint8)
        if self.step % 5 == 0:
            flags[1] |= P.FLAG_FRESH
        stamp = np.stack([np.arange(self.rows) + self.lane_base, np.full(self.rows, self.step)], 1).astype(np.int32)
        self.step += 1
        return {"stamp": stamp, "flags": flags, "weight": np.ones(self.rows, np.float32),
                "info": [{"queued": len(self.queue)}]}


def _stream(tmp_path, pools, seq_len=4, refill_below=3):
    ix = _index(tmp_path, 40, commits=tuple(pools))
    make_split(ix, "s", heldout=0.0)
    return TeacherStream(ix, "s", "train", pools, seq_len=seq_len, refill_below=refill_below)


def test_sequences_keep_lane_order_across_chunks(tmp_path):
    s = _stream(tmp_path, {"c2": FakePool(10), "c1": FakePool(0)})
    a, b = s.next_seq(), s.next_seq()
    assert a["stamp"].shape == (4, 4, 2)
    for x in (a, b):
        assert (x["stamp"][:, :, 0] == np.array([0, 1, 10, 11])).all()  # commit order c1, c2; rows in order
    assert (b["stamp"][0, :, 1] == a["stamp"][-1, :, 1] + 1).all()  # time continues across chunks


def test_fresh_marks_reset_points(tmp_path):
    s = _stream(tmp_path, {"c1": FakePool(0)}, seq_len=10)
    x = s.next_seq()
    assert x["fresh"].shape == (10, 2)
    assert list(np.nonzero(x["fresh"][:, 1])[0]) == [0, 5]
    assert not x["fresh"][:, 0].any()
    assert x["valid"].all()


def test_queue_is_refilled_before_it_drains(tmp_path):
    pool = FakePool(0)
    s = _stream(tmp_path, {"c1": pool}, seq_len=8, refill_below=3)
    for _ in range(6):
        s.next_seq()
        assert len(pool.queue) > 0
    assert pool.loads >= 2
    item = pool.queue[0]
    assert {t["clientID"] for t in item["teachers"]} == {"AAAAAAAA", "BBBBBBBB"}
    a = next(t for t in item["teachers"] if t["clientID"] == "AAAAAAAA")
    assert a["value"] == 1.0 and a["weight"] == 1.0
