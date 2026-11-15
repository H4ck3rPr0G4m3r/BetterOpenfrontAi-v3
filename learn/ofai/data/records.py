"""Game records on disk: `data/records/<gitCommit>/<gameId>.json.gz` (roadmap §3.1)."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from ofai import paths


def record_path(commit: str, game_id: str, root: Path | None = None) -> Path:
    return (root or paths.RECORDS) / commit / f"{game_id}.json.gz"


def find_record(game_id: str, root: Path | None = None) -> Path | None:
    base = root or paths.RECORDS
    if not base.exists():
        return None
    for commit_dir in base.iterdir():
        p = commit_dir / f"{game_id}.json.gz"
        if p.exists():
            return p
    return None


def write_record(record: dict, root: Path | None = None) -> Path:
    """Atomic write (temp file + rename), so a crash never leaves half a record."""
    commit = record.get("gitCommit") or "unknown"
    game_id = record["info"]["gameID"]
    path = record_path(commit, game_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(record, f, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def read_record(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def all_records(root: Path | None = None) -> list[Path]:
    base = root or paths.RECORDS
    if not base.exists():
        return []
    return sorted(base.glob("*/*.json.gz"))
