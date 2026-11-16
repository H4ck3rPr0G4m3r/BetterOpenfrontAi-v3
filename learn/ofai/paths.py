"""Repo-relative locations every module agrees on (roadmap §3.1, §3.7)."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "engine"
ENGINES = REPO / "engines"
SIM = REPO / "sim"
# OFAI_DATA moves crawled data elsewhere (tests, a bigger disk, cloud boxes).
DATA = Path(os.environ["OFAI_DATA"]).resolve() if os.environ.get("OFAI_DATA") else REPO / "data"
RECORDS = DATA / "records"
RUNS = REPO / "runs"
CHECKPOINTS = REPO / "checkpoints"
REPLAYS = REPO / "replays"
