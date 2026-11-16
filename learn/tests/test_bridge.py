import numpy as np
import pytest

from conftest import SMALL_MAP, pool_kwargs
from ofai.bridge import SimPool, SimWorkerDied
from ofai.bridge import protocol as P

TEACHERS = [{"clientID": "SEAT0000", "weight": 0.8, "value": 1.0}, {"clientID": "SEAT0001", "weight": 0.5, "value": 0.0}]


@pytest.fixture
def replay_pool(synth_records):
    with SimPool("SYNTHETIC", "replay", 2, games=1, seats=2, **pool_kwargs()) as pool:
        pool.load([{"path": str(f), "teachers": TEACHERS} for f in synth_records])
        yield pool


def test_hello_carries_layout(replay_pool):
    lay = replay_pool.layout
    assert lay.version >= 1
    assert lay.K == 4 and lay.A == 10
    assert lay.field("mm_owner").shape == (128, 128)
    assert replay_pool.rows == 4


def test_next_returns_arrays_shaped_by_layout(replay_pool):
    b = replay_pool.next()
    for f in replay_pool.layout.fields:
        assert b[f.name].shape == (4, *f.shape), f.name
        assert b[f.name].dtype == f.dtype
    assert (b["flags"] & P.FLAG_VALID).all()
    assert (b["flags"] & P.FLAG_FRESH).all()
    # Weights follow the teachers (highest weight first in each game's rows).
    assert np.allclose(sorted(b["weight"]), [0.5, 0.5, 0.8, 0.8])
    assert len(b["info"]) == 2


def test_labels_shape_rows_x_k_x_a(replay_pool):
    b = replay_pool.next()
    assert b["labels"].shape == (4, 4, 10)
    assert b["labels"].dtype == np.int32
    # Synthetic seats ask to spawn on turns 2 and 3: the first step's labels.
    assert (b["labels"][:, 0, 0] == replay_pool.layout.at("SPAWN")).all()
    assert (b["labels"][:, 1, 0] == replay_pool.layout.at("STOP")).all()


def test_worker_death_raises_sim_worker_died(replay_pool):
    replay_pool.next()
    replay_pool.workers[1].proc.kill()
    replay_pool.workers[1].proc.wait()
    with pytest.raises(SimWorkerDied) as e:
        replay_pool.next()
    assert e.value.index == 1
    # The healthy worker was drained: after a restart the pool works again.
    replay_pool.restart(1)
    b = replay_pool.next()
    assert b["flags"].shape == (4,)


def test_env_role_reset_and_step():
    spec = {"map": SMALL_MAP, "seed": "py-env", "seats": 2, "bots": 0, "nations": "disabled", "difficulty": "Medium",
            "tickCap": 50, "latency": [1, 3], "record": False}
    with SimPool("SYNTHETIC", "env", 1, games=2, seats=2, **pool_kwargs()) as pool:
        b = pool.reset([spec, None])
        assert list(b["flags"]) == [P.FLAG_NEXT | P.FLAG_FRESH] * 2 + [0, 0]
        stop = np.full((4, 4, 10), -1, dtype=np.int32)
        stop[:, 0, 0] = 0
        b = pool.step(stop)
        assert b["stats"].shape == (4, 5)
        assert list(b["flags"][:2]) == [P.FLAG_VALID | P.FLAG_NEXT] * 2
        assert b["info"][0]["over"] == [False, True]
