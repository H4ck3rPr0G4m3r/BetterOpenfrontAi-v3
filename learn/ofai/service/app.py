"""The app's local service: `python -m ofai serve` on 127.0.0.1:47800 (roadmap §3.7).

Binds localhost only. Exposes the data index, records, engine bundles and the
background jobs (crawl, replay-check, engine builds), plus an SSE stream of job
changes. Later milestones add training sessions (M4) and inference (M5).
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ofai import __version__, paths
from ofai.service.jobs import JobManager

HOST = "127.0.0.1"
PORT = 47800


class Services:
    """What the endpoints need; tests replace pieces with fakes."""

    def __init__(self, *, index=None, jobs: JobManager | None = None, api_factory=None, ensure=None,
                 check_index=None, list_bundles=None, crawler_cls=None, engine_update=None) -> None:
        self._index = index
        self.jobs = jobs or JobManager(paths.DATA / "app" / "jobs.json")
        self.api_factory = api_factory
        self.ensure = ensure
        self.check_index = check_index
        self.list_bundles = list_bundles
        self.crawler_cls = crawler_cls
        self.engine_update = engine_update

    @property
    def index(self):
        if self._index is None:
            from ofai.data.index import Index

            self._index = Index()
        return self._index


def create_app(services: Services | None = None) -> FastAPI:
    s = services or Services()
    app = FastAPI(title="ofai service", version=__version__)
    app.state.services = s

    @app.get("/status")
    def status() -> dict:
        return {"ok": True, "version": __version__, "jobs": s.jobs.list()[:20]}

    # -- data ----------------------------------------------------------------

    @app.get("/data/stats")
    def data_stats() -> dict:
        stats = s.index.stats()
        stats["window"] = s.index.window_commits(datetime.now(timezone.utc))
        stats["queued"] = s.index.queue_size() if hasattr(s.index, "queue_size") else 0
        progress_file = paths.DATA / "crawl_progress.json"
        stats["crawl"] = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else None
        return stats

    @app.get("/data/games")
    def data_games(map: str | None = None, commit: str | None = None, offset: int = 0, limit: int = 100) -> list[dict]:
        return s.index.games(map=map, commit=commit, offset=offset, limit=min(limit, 1000))

    @app.get("/data/games/{game_id}/record")
    def data_record(game_id: str) -> JSONResponse:
        from ofai.data.records import read_record

        g = s.index.game(game_id)
        if g is None or not Path(g["path"]).exists():
            raise HTTPException(404, f"no stored record for {game_id}")
        return JSONResponse(read_record(Path(g["path"])))

    # -- engines -------------------------------------------------------------

    @app.get("/engines")
    def engines() -> list[dict]:
        lister = s.list_bundles
        if lister is None:
            from ofai.engines import list_bundles as lister
        return [m.to_json() for m in lister()]

    # -- jobs ------------------------------------------------------------------

    @app.post("/jobs/crawl")
    def job_crawl(cfg: dict[str, Any] = Body(default_factory=dict)) -> dict:
        from ofai.data.crawl import CrawlConfig

        known = {f.name for f in fields(CrawlConfig)}
        bad = sorted(set(cfg) - known)
        if bad:
            raise HTTPException(422, f"unknown crawl settings: {', '.join(bad)}")
        crawl_cfg = CrawlConfig(**cfg)

        def run(progress, stop):
            crawler_cls = s.crawler_cls
            if crawler_cls is None:
                from ofai.data.crawl import Crawler as crawler_cls
            api_factory = s.api_factory
            if api_factory is None:
                from ofai.data.api import OpenFrontApi as api_factory
            api = api_factory()
            try:
                return crawler_cls(api, s.index, crawl_cfg, progress=progress, stop=stop).run()
            finally:
                close = getattr(api, "close", None)
                if close:
                    close()

        return s.jobs.start("crawl", run, cfg)

    @app.post("/jobs/replay-check")
    def job_replay_check() -> dict:
        def run(progress, stop):
            check = s.check_index
            if check is None:
                from ofai.data.replay_check import check_index as check
            return check(s.index, progress=progress, stop=stop.is_set)

        return s.jobs.start("replay-check", run)

    @app.post("/jobs/engine/{commit}")
    def job_engine(commit: str) -> dict:
        if not (len(commit) == 40 and all(c in "0123456789abcdefABCDEF" for c in commit)):
            raise HTTPException(422, "commit must be a 40-hex sha")

        def run(progress, stop):
            ensure = s.ensure
            if ensure is None:
                from ofai.engines import ensure_bundle as ensure
            progress({"stage": "building", "commit": commit})
            return ensure(commit).to_json()

        return s.jobs.start(f"engine:{commit}", run, {"commit": commit})

    @app.post("/jobs/engine-update")
    def job_engine_update() -> dict:
        def run(progress, stop):
            update = s.engine_update
            if update is None:
                from ofai.engine_update import update_engine as update
            return update(s.index, progress=progress)

        return s.jobs.start("engine-update", run)

    @app.post("/jobs/{job_id}/stop")
    def job_stop(job_id: str) -> dict:
        job = s.jobs.stop(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job

    @app.get("/jobs")
    def jobs() -> list[dict]:
        return s.jobs.list()

    @app.get("/events")
    async def events() -> StreamingResponse:
        q = s.jobs.subscribe()

        async def stream():
            try:
                yield "retry: 2000\n\n"
                for job in s.jobs.running():
                    yield f"data: {json.dumps({'type': 'job', 'job': job})}\n\n"
                while True:
                    try:
                        event = await asyncio.to_thread(q.get, True, 15)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                s.jobs.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def uvicorn_config(port: int = PORT):
    import uvicorn

    return uvicorn.Config(create_app(), host=HOST, port=port, log_level="warning")


def serve(port: int = PORT) -> None:
    import uvicorn

    server = uvicorn.Server(uvicorn_config(port))
    threading.current_thread().name = "ofai-service"
    server.run()
