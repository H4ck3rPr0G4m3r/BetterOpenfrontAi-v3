"""Shared fixtures: the sim run from source (no bundle needed) and synthetic records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ofai import paths

# The sim straight from TypeScript: `node --import tsx src/cli.ts ...` in sim/.
SIM_CMD = ["node", "--import", "tsx", "src/cli.ts"]
SMALL_MAP = "Onion"


def run_sim_src(*args: str, timeout: float = 600) -> list[dict]:
    out = subprocess.run(SIM_CMD + list(args), cwd=paths.SIM, capture_output=True, timeout=timeout, check=True)
    return [json.loads(line) for line in out.stdout.decode().splitlines() if line.strip().startswith("{")]


def pool_kwargs() -> dict:
    return {"command": SIM_CMD, "cwd": paths.SIM, "engine_root": paths.ENGINE}


@pytest.fixture(scope="session")
def synth_records(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Two small synthetic games (2 seats, 400 turns) on the smallest map."""
    d = tmp_path_factory.mktemp("records")
    files = []
    for seed in ("py-a", "py-b"):
        f = d / f"{seed}.json.gz"
        run_sim_src("synth", str(f), "--map", SMALL_MAP, "--seats", "2", "--turns", "400", "--seed", seed,
                    "--engine-root", str(paths.ENGINE))
        files.append(f)
    return files
