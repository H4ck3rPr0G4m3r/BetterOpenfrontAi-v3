"""A pool of sim worker processes for one engine commit (roadmap §3.4).

Each worker is `sim.mjs serve` with its own BridgeClient. Requests go to all
workers at once (send everything, then read every reply) so the processes
work in parallel; batches are concatenated in worker order. A worker that
dies raises SimWorkerDied carrying its stderr tail; `restart(i)` replaces it
(its rows then start FRESH on new games).
"""

from __future__ import annotations

import collections
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ofai import paths
from ofai.bridge.client import Batch, BridgeClient, BridgeError


class SimWorkerDied(RuntimeError):
    def __init__(self, index: int, stderr_tail: str, cause: BaseException | None = None) -> None:
        super().__init__(f"sim worker {index} died: {cause!r}\n--- stderr tail ---\n{stderr_tail}")
        self.index = index
        self.stderr_tail = stderr_tail


class _Worker:
    def __init__(self, cmd: list[str], cwd: Path | None, startup_timeout: float) -> None:
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.tail: collections.deque[str] = collections.deque(maxlen=200)
        threading.Thread(target=self._drain, daemon=True).start()
        line = self._first_line(startup_timeout)
        try:
            ready = json.loads(line)
        except ValueError:
            ready = {}
        if "listening" not in ready:
            self.kill()
            raise RuntimeError(f"sim worker did not start: {line!r}\n{self.stderr()}")
        self.client = BridgeClient("127.0.0.1", int(ready["listening"]))

    def _first_line(self, timeout: float) -> str:
        box: list[str] = []
        t = threading.Thread(target=lambda: box.append(self.proc.stdout.readline().decode("utf-8", "replace")), daemon=True)
        t.start()
        t.join(timeout)
        if not box:
            self.kill()
            raise RuntimeError(f"sim worker gave no ready line in {timeout}s\n{self.stderr()}")
        return box[0].strip()

    def _drain(self) -> None:
        for raw in iter(self.proc.stderr.readline, b""):
            self.tail.append(raw.decode("utf-8", "replace").rstrip())

    def stderr(self) -> str:
        return "\n".join(list(self.tail)[-40:])

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


class SimPool:
    def __init__(
        self,
        commit: str,
        role: str,
        workers: int,
        *,
        games: int = 1,
        seats: int = 1,
        command: list[str] | None = None,
        cwd: Path | None = None,
        engine_root: Path | None = None,
        startup_timeout: float = 120.0,
    ) -> None:
        if role not in ("replay", "env"):
            raise ValueError(f"role must be replay or env, not {role}")
        self.commit, self.role, self.games, self.seats = commit, role, games, seats
        bundle = paths.ENGINES / commit
        self._command = command or ["node", str(bundle / "sim.mjs")]
        self._cwd = cwd
        self._engine_root = engine_root or bundle / "src"
        self._startup_timeout = startup_timeout
        self.workers: list[_Worker] = []
        self.restarts = 0
        self._next_load = 0
        self._sent: tuple | None = None
        try:
            for _ in range(workers):
                self.workers.append(self._spawn())
        except BaseException:
            self.close()
            raise
        first = self.workers[0].client
        self.layout = first.layout
        self.rows_per_worker = first.rows
        self.rows = first.rows * len(self.workers)

    def _spawn(self) -> _Worker:
        args = ["serve", "--role", self.role, "--games", str(self.games), "--seats", str(self.seats),
                "--port", "0", "--engine-root", str(self._engine_root), "--commit", self.commit]
        return _Worker(self._command + args, self._cwd, self._startup_timeout)

    # -- control ------------------------------------------------------------------

    def restart(self, i: int) -> None:
        self.workers[i].kill()
        self.workers[i] = self._spawn()
        self.restarts += 1

    def close(self) -> None:
        for w in self.workers:
            try:
                w.client.close() if hasattr(w, "client") else None
            finally:
                w.kill()
        self.workers = []

    def __enter__(self) -> SimPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- replay role ----------------------------------------------------------------

    def load(self, items: list[dict], worker: int | None = None) -> None:
        """Queues records on one worker, or spreads them round-robin."""
        if worker is not None:
            self._guard(worker, lambda c: c.load(items))
            return
        per: list[list[dict]] = [[] for _ in self.workers]
        for item in items:
            per[self._next_load % len(self.workers)].append(item)
            self._next_load += 1
        for i, chunk in enumerate(per):
            if chunk:
                self._guard(i, lambda c, chunk=chunk: c.load(chunk))

    def next(self) -> Batch:
        return self._all(lambda c: c.send_next(), None)

    # -- env role -----------------------------------------------------------------

    def reset(self, games: list[dict | None]) -> Batch:
        if len(games) != self.games * len(self.workers):
            raise ValueError(f"reset needs {self.games * len(self.workers)} entries, got {len(games)}")
        return self._all(lambda c, i: c.send_reset(games[i * self.games : (i + 1) * self.games]), None, with_index=True)

    def step(self, actions: np.ndarray) -> Batch:
        n = self.rows_per_worker
        return self._all(lambda c, i: c.send_step(actions[i * n : (i + 1) * n]), None, with_index=True)

    # -- plumbing -------------------------------------------------------------------

    def _guard(self, i: int, fn: Any) -> Any:
        w = self.workers[i]
        try:
            return fn(w.client)
        except (OSError, ConnectionError, BridgeError) as e:
            if isinstance(e, BridgeError) and w.alive():
                raise
            w.proc.poll()
            raise SimWorkerDied(i, w.stderr(), e) from e

    def send_next(self) -> None:
        """First half of next(): every worker starts its step (lets several pools run at once)."""
        self._sent = self._send_all(lambda c: c.send_next(), False)

    def recv_next(self) -> Batch:
        sent, self._sent = self._sent, None
        return self._recv_all(sent)

    def _all(self, send: Any, _unused: Any, with_index: bool = False) -> Batch:
        return self._recv_all(self._send_all(send, with_index))

    def _send_all(self, send: Any, with_index: bool) -> tuple[list[bool], SimWorkerDied | None]:
        sent: list[bool] = []
        died: SimWorkerDied | None = None
        for i in range(len(self.workers)):
            try:
                self._guard(i, (lambda c, i=i: send(c, i)) if with_index else send)
                sent.append(True)
            except SimWorkerDied as e:
                died = died or e
                sent.append(False)
        return sent, died

    def _recv_all(self, state: tuple[list[bool], SimWorkerDied | None]) -> Batch:
        # Every live worker gets its request and has its reply read, even when
        # another one dies, so no request stays in flight on a healthy worker.
        sent, died = state
        batches = []
        for i in range(len(self.workers)):
            if not sent[i]:
                continue
            try:
                batches.append(self._guard(i, lambda c: c.recv()))
            except SimWorkerDied as e:
                died = died or e
        if died is not None:
            raise died
        return concat_batches(batches)


def concat_batches(batches: list[Batch]) -> Batch:
    out: Batch = {}
    for key in batches[0]:
        if key == "info":
            out["info"] = [b["info"] for b in batches]
        else:
            out[key] = np.concatenate([b[key] for b in batches], axis=0)
    return out
