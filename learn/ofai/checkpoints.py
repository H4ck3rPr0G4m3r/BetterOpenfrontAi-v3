"""Atomic checkpoints (M3 Task 5, roadmap §3.8).

save: write `<path>.tmp`, fsync, rename over `<path>`, then write `<path>.sha256`
the same way. A checkpoint counts only when its sha256 file exists and
matches, so a power cut mid-write never yields a truncated "latest".
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ofai import paths


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def save_checkpoint(path: Path, payload: dict) -> None:
    buf = io.BytesIO()
    torch.save(payload, buf)
    data = buf.getvalue()
    # The sha file is removed first and written last: in between, the checkpoint doesn't count.
    _sha_path(path).unlink(missing_ok=True)
    _atomic_write(path, data)
    _atomic_write(_sha_path(path), hashlib.sha256(data).hexdigest().encode())


def verify(path: Path) -> bool:
    sha = _sha_path(path)
    if not path.exists() or not sha.exists():
        return False
    try:
        want = sha.read_text().strip()
    except OSError:
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == want


def load_checkpoint(path: Path, map_location: Any = "cpu") -> dict:
    if not verify(path):
        raise ValueError(f"checkpoint {path} is missing its sha256 or does not match it")
    return torch.load(path, map_location=map_location, weights_only=False)


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Newest `*.pt` in run_dir whose sha256 verifies."""
    cands = sorted(Path(run_dir).glob("*.pt"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    for p in cands:
        if verify(p):
            return p
    return None


def register(name: str, path: Path, phase: str, step: int, extra: dict | None = None,
             index_path: Path | None = None) -> None:
    """Adds or replaces `name` in checkpoints/index.json (written atomically)."""
    index_path = index_path or paths.CHECKPOINTS / "index.json"
    try:
        entries = json.loads(index_path.read_text()) if index_path.exists() else []
    except (OSError, ValueError):
        entries = []
    entries = [e for e in entries if e.get("name") != name]
    entry = {"name": name, "path": str(path), "phase": phase, "step": step,
             "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}
    entries.append(entry)
    _atomic_write(index_path, json.dumps(entries, indent=1).encode())
