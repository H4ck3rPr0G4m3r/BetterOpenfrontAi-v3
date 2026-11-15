"""Background jobs for the app (crawl, replay-check, engine builds).

One job per kind at a time: starting a kind that is already running returns
the running job. Jobs run in threads and report progress through a callback;
every change is pushed to subscribers (the SSE stream) and persisted to
`data/jobs.json`, so the app can show the last result after a restart.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

JobFn = Callable[[Callable[[dict], None], threading.Event], dict]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._stops: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._subscribers: list[queue.Queue] = []
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for job in data.get("jobs", []):
            if job.get("state") == "running":
                # The service died while it ran; say so instead of pretending it still runs.
                job["state"] = "failed"
                job["error"] = job.get("error") or "interrupted (the service stopped while this job ran)"
            self._jobs[job["id"]] = job

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        recent = sorted(self._jobs.values(), key=lambda j: j["startedAt"])[-50:]
        tmp.write_text(json.dumps({"jobs": recent}, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_file)

    # -- events --------------------------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, job: dict) -> None:
        event = {"type": "job", "job": dict(job)}
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    # -- jobs ------------------------------------------------------------------

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in sorted(self._jobs.values(), key=lambda j: j["startedAt"], reverse=True)]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None

    def latest(self, kind: str) -> dict | None:
        jobs = [j for j in self.list() if j["kind"] == kind]
        return jobs[0] if jobs else None

    def start(self, kind: str, fn: JobFn, params: dict | None = None) -> dict:
        with self._lock:
            for job in self._jobs.values():
                if job["kind"] == kind and job["state"] == "running":
                    return dict(job)
            job = {"id": uuid.uuid4().hex[:12], "kind": kind, "state": "running", "progress": {}, "error": None,
                   "result": None, "params": params or {}, "startedAt": _now(), "endedAt": None}
            self._jobs[job["id"]] = job
            stop = threading.Event()
            self._stops[job["id"]] = stop
            self._save()
        self._publish(job)

        def progress(p: dict) -> None:
            with self._lock:
                job["progress"] = p
            self._publish(job)

        def body() -> None:
            try:
                result = fn(progress, stop)
                with self._lock:
                    job["result"] = result
                    job["state"] = "stopped" if stop.is_set() else "done"
                    if isinstance(result, dict) and result.get("phase") == "error":
                        job["state"], job["error"] = "failed", result.get("error")
            except Exception as e:  # noqa: BLE001 - a job failure is shown, never crashes the service
                with self._lock:
                    job["state"] = "failed"
                    job["error"] = f"{type(e).__name__}: {e}"
                    job["trace"] = traceback.format_exc(limit=8)
            with self._lock:
                job["endedAt"] = _now()
                self._save()
            self._publish(job)

        t = threading.Thread(target=body, name=f"job-{kind}", daemon=True)
        self._threads[job["id"]] = t
        t.start()
        return dict(job)

    def stop(self, job_id: str) -> dict | None:
        with self._lock:
            stop = self._stops.get(job_id)
            job = self._jobs.get(job_id)
        if stop is not None:
            stop.set()
        return dict(job) if job else None

    def wait(self, job_id: str, timeout: float | None = None) -> dict | None:
        t = self._threads.get(job_id)
        if t is not None:
            t.join(timeout)
        return self.get(job_id)

    def running(self) -> list[dict]:
        return [j for j in self.list() if j["state"] == "running"]
