import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ofai.engines import BundleMeta
from ofai.service.app import HOST, Services, create_app, uvicorn_config
from ofai.service.jobs import JobManager


class FakeIndex:
    def stats(self):
        return {"games": 0}

    def window_commits(self, now):
        return []

    def queue_size(self):
        return 0

    def games(self, **kw):
        return []

    def game(self, gid):
        return None


class SlowCrawler:
    started = threading.Event()

    def __init__(self, api, index, cfg, progress, stop):
        self.progress, self.stop, self.cfg = progress, stop, cfg

    def run(self):
        SlowCrawler.started.set()
        n = 0
        while not self.stop.is_set() and n < 200:
            n += 1
            self.progress({"phase": "records", "fetched": n, "targetGames": self.cfg.target_games})
            time.sleep(0.01)
        return {"phase": "stopped" if self.stop.is_set() else "done", "fetched": n}


@pytest.fixture
def client(tmp_path: Path):
    s = Services(index=FakeIndex(), jobs=JobManager(tmp_path / "jobs.json"), api_factory=lambda: object(),
                 crawler_cls=SlowCrawler, ensure=lambda c: BundleMeta(commit=c, status="ok"),
                 list_bundles=lambda: [BundleMeta(commit="a" * 40, status="failed", stage="fetch")])
    with TestClient(create_app(s)) as c:
        c.services = s
        yield c


def test_status_ok(client):
    r = client.get("/status").json()
    assert r["ok"] is True and "version" in r and r["jobs"] == []


def test_crawl_job_singleton(client):
    a = client.post("/jobs/crawl", json={"target_games": 50}).json()
    b = client.post("/jobs/crawl", json={}).json()
    assert a["id"] == b["id"] and a["state"] == "running"
    client.post(f"/jobs/{a['id']}/stop")
    client.services.jobs.wait(a["id"], 5)


def test_unknown_crawl_setting_is_rejected(client):
    assert client.post("/jobs/crawl", json={"bogus": 1}).status_code == 422


def test_stop_job(client):
    job = client.post("/jobs/crawl", json={}).json()
    SlowCrawler.started.wait(5)
    client.post(f"/jobs/{job['id']}/stop")
    done = client.services.jobs.wait(job["id"], 5)
    assert done["state"] == "stopped"
    # Persisted for the next start of the app.
    saved = json.loads(Path(client.services.jobs.state_file).read_text())
    assert saved["jobs"][0]["state"] == "stopped"


def test_events_stream_emits_progress(client):
    q = client.services.jobs.subscribe()
    job = client.post("/jobs/crawl", json={}).json()
    seen = []
    deadline = time.time() + 5
    while time.time() < deadline and not any(e["job"]["progress"].get("fetched") for e in seen):
        seen.append(q.get(timeout=5))
    assert any(e["type"] == "job" and e["job"]["progress"].get("fetched") for e in seen)
    client.post(f"/jobs/{job['id']}/stop")
    client.services.jobs.wait(job["id"], 5)


def test_engine_job_and_list(client):
    job = client.post("/jobs/engine/" + "b" * 40).json()
    done = client.services.jobs.wait(job["id"], 5)
    assert done["state"] == "done" and done["result"]["status"] == "ok"
    assert client.post("/jobs/engine/nothex").status_code == 422
    assert client.get("/engines").json()[0]["stage"] == "fetch"


def test_interrupted_job_is_marked_failed_on_restart(tmp_path):
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps({"jobs": [{"id": "x", "kind": "crawl", "state": "running", "startedAt": "2026-01-01"}]}))
    assert JobManager(f).get("x")["state"] == "failed"


def test_binds_localhost_only():
    assert HOST == "127.0.0.1"
    assert uvicorn_config().host == "127.0.0.1"
