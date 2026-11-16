"""Shared fixtures: the sim run from source (no bundle needed) and synthetic records."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from ofai import paths

SMALL_MAP = "Onion"
_BUNDLE: list[Path] = []


def sim_cmd() -> list[str]:
    """The sim from the current sources, bundled with esbuild against the pinned engine.

    Not tsx: tsx ignores `useDefineForClassFields: false`, which the engine
    relies on (field initializers that read constructor parameters)."""
    if not _BUNDLE:
        out = Path(tempfile.mkdtemp(prefix="ofai-sim-")) / "sim.mjs"
        subprocess.run(["node", str(paths.SIM / "scripts" / "bundle.mjs"), "--engine-root", str(paths.ENGINE),
                        "--out", str(out)], cwd=paths.SIM, check=True, capture_output=True, timeout=300)
        _BUNDLE.append(out)
    return ["node", str(_BUNDLE[0])]


def run_sim_src(*args: str, timeout: float = 600) -> list[dict]:
    out = subprocess.run(sim_cmd() + list(args), capture_output=True, timeout=timeout, check=True)
    return [json.loads(line) for line in out.stdout.decode().splitlines() if line.strip().startswith("{")]


def pool_kwargs() -> dict:
    return {"command": sim_cmd(), "engine_root": paths.ENGINE}


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
