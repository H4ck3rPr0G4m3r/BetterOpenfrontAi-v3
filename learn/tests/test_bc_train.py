import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from conftest import layout, tiny_config
from ofai.bc.train import Trainer, lr_at
from ofai.bridge import protocol as P
from ofai.bridge.pool import SimWorkerDied
from ofai.runlog import RunLog


class FakeStream:
    def __init__(self, seq, deaths=0):
        self.seq = seq
        self.lanes = seq["flags"].shape[1]
        self.deaths_left = deaths
        self.restarts = 0
        self.state = {"n": 0}

    def next_seq(self):
        if self.deaths_left:
            self.deaths_left -= 1
            e = SimWorkerDied(0, "worker stderr: boom")
            e.commit = "c1"
            raise e
        self.state["n"] += 1
        return self.seq

    def restart(self, e):
        self.restarts += 1
        return 0, self.lanes

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, d):
        self.state = dict(d)

    def close(self):
        pass


@pytest.fixture(scope="module")
def seq(synth_dump):
    obs, _, _, rows = synth_dump
    lay = layout()
    T, lanes = 3, rows // 3
    out = {k: v[: T * lanes].reshape(T, lanes, *v.shape[1:]).copy() for k, v in obs.items()}
    labels = np.full((T, lanes, lay.K, lay.A), -1, np.int32)
    labels[..., 0, lay.af("type")] = lay.at("ATTACK")
    labels[..., 0, lay.af("player")] = 33
    labels[..., 0, lay.af("troops")] = 3
    labels[..., 1, lay.af("type")] = lay.at("STOP")
    flags = np.full((T, lanes), P.FLAG_VALID, np.uint8)
    flags[0] |= P.FLAG_FRESH
    out.update(labels=labels, weight=np.ones((T, lanes), np.float32), value=np.full((T, lanes), 0.7, np.float32),
               flags=flags, fresh=(flags & P.FLAG_FRESH) != 0, valid=np.ones((T, lanes), bool))
    return out


def _cfg(**kw):
    base = {"model": tiny_config().to_dict(), "lr": 3e-3, "warmup_steps": 1, "log_every": 1, "micro_lanes": 2,
            "total_samples": 10_000_000, "eval_every": 10**12, "checkpoint_minutes": 999}
    base.update(kw)
    return base


def _trainer(tmp_path, seq, deaths=0, **kw):
    stream = FakeStream(seq, deaths)
    return Trainer(_cfg(**kw), tmp_path, layout(), lambda part: stream if part == "train" else None), stream


def _metrics(tmp_path):
    return [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]


def test_loss_decreases_on_a_fixed_batch(tmp_path, seq):
    tr, _ = _trainer(tmp_path, seq)
    out = tr.run(max_steps=25)
    assert out["reason"] == "max_steps"
    losses = [m["loss"] for m in _metrics(tmp_path) if "loss" in m]
    assert losses[-1] < 0.7 * losses[0], losses


def test_resume_continues_step_count_and_lr(tmp_path, seq):
    tr, _ = _trainer(tmp_path, seq)
    tr.run(max_steps=5)
    tr2, stream2 = _trainer(tmp_path, seq)
    out = tr2.run(max_steps=8)
    assert out["step"] == 8
    ev = [e for e in RunLog(tmp_path).events() if e["kind"] == "resume"]
    assert ev and ev[0]["step"] == 5
    # Restored, then advanced (prefetch may have read one or two sequences ahead before each stop).
    assert 5 + 3 <= stream2.state["n"] <= 5 + 3 + 4
    steps = [m["step"] for m in _metrics(tmp_path) if "loss" in m]
    assert steps == list(range(1, 9))
    m6 = [m for m in _metrics(tmp_path) if m.get("step") == 6][0]
    assert m6["lr"] == pytest.approx(lr_at(tr2.cfg, 5, 5 * 12))


def test_stop_file_stops_and_checkpoints(tmp_path, seq):
    (tmp_path / "STOP").write_text("")
    tr, _ = _trainer(tmp_path, seq)
    out = tr.run(max_steps=100)
    assert out["reason"] == "stop_file"
    assert list(tmp_path.glob("step-*.pt")) and list(tmp_path.glob("step-*.pt.sha256"))


def test_until_stops_at_time(tmp_path, seq):
    tr, _ = _trainer(tmp_path, seq)
    out = tr.run(until=datetime.now() - timedelta(seconds=1), max_steps=100)
    assert out["reason"] == "until" and out["step"] == 0


def test_worker_death_restarts_lanes(tmp_path, seq):
    tr, stream = _trainer(tmp_path, seq, deaths=1)
    out = tr.run(max_steps=3)
    assert out["reason"] == "max_steps" and stream.restarts == 1
    kinds = [e["kind"] for e in RunLog(tmp_path).events()]
    assert "worker_died" in kinds and "error" not in kinds


def test_three_deaths_in_ten_minutes_stop_the_run(tmp_path, seq):
    tr, stream = _trainer(tmp_path, seq, deaths=3)
    out = tr.run(max_steps=3)
    assert out["reason"] == "worker_deaths"
    err = [e for e in RunLog(tmp_path).events() if e["kind"] == "error"]
    assert err and "boom" in err[0]["stderr"]
