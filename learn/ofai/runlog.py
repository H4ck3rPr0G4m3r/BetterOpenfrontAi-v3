"""Run directory logs (roadmap §3.8): metrics.jsonl and events.jsonl."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _clean(v: Any) -> Any:
    if isinstance(v, float) and v != v:  # NaN is not JSON
        return None
    if hasattr(v, "item"):
        return _clean(v.item())
    return v


class RunLog:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, obj: dict) -> None:
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: _clean(v) for k, v in obj.items()}) + "\n")

    def metric(self, **kv: Any) -> None:
        self._append("metrics.jsonl", {"t": round(time.time(), 3), **kv})

    def event(self, kind: str, **kv: Any) -> None:
        self._append("events.jsonl", {"t": round(time.time(), 3), "kind": kind, **kv})

    def events(self) -> list[dict]:
        p = self.dir / "events.jsonl"
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
