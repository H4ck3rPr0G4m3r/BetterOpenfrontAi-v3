"""Frozen train / held-out split by game (M3 Task 4). Never by step."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ofai import paths
from ofai.data.index import Index


def heldout_fraction(game_id: str) -> float:
    """Stable position of a game in [0, 1) from its id."""
    return int(hashlib.sha1(game_id.encode()).hexdigest()[:12], 16) / float(1 << 48)


def split_path(name: str) -> Path:
    return paths.DATA / "splits" / f"{name}.json"


def make_split(index: Index, name: str = "bc-v1", heldout: float = 0.05) -> Path:
    """Writes data/splits/<name>.json from replay-ok, in-window games. Refuses to overwrite."""
    out = split_path(name)
    if out.exists():
        raise FileExistsError(f"split {name} already exists at {out}; splits are frozen")
    games = sorted(g["id"] for g in index.games(only_in_window=True, replay_ok=True, limit=10_000_000))
    held = [g for g in games if heldout_fraction(g) < heldout]
    train = [g for g in games if heldout_fraction(g) >= heldout]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "heldout_fraction": heldout, "train": train, "heldout": held}, indent=1))
    tmp.replace(out)
    return out


def load_split(name: str) -> dict:
    return json.loads(split_path(name).read_text())
