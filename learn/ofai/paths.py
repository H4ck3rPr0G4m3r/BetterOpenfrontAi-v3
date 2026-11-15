"""Repo-relative locations every module agrees on (roadmap §3.1, §3.7)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "engine"
ENGINES = REPO / "engines"
SIM = REPO / "sim"
DATA = REPO / "data"
RECORDS = DATA / "records"
RUNS = REPO / "runs"
CHECKPOINTS = REPO / "checkpoints"
REPLAYS = REPO / "replays"
