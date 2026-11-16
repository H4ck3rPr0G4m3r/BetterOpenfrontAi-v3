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


_LAYOUT: list = []


def layout():
    """The layout the current sim sources export."""
    from ofai.layout import Layout

    if not _LAYOUT:
        _LAYOUT.append(Layout.from_json(run_sim_src("layout")[0]))
    return _LAYOUT[0]


def dump_obs(record: Path, out: Path, steps: int, every: int):
    """`sim dump-obs`: (obs arrays, minimap reference planes, window reference planes, rows)."""
    import numpy as np

    run_sim_src("dump-obs", str(record), "--out", str(out), "--steps", str(steps), "--every", str(every),
                "--engine-root", str(paths.ENGINE))
    rows = json.loads((out / "header.json").read_text())["rows"]
    lay = layout()
    raw = (out / "obs.bin").read_bytes()
    obs, end = lay.parse_obs(raw, 0, rows)
    assert end == len(raw)
    P = len(lay.planes)
    MM, WIN, NW = lay.sizes["MM"], lay.sizes["WIN"], lay.sizes["NUM_WINDOWS"]
    mm = np.frombuffer((out / "mm_ref.bin").read_bytes(), dtype="<f4").reshape(rows, P, MM, MM)
    win = np.frombuffer((out / "win_ref.bin").read_bytes(), dtype="<f4").reshape(rows, NW, P, WIN, WIN)
    return obs, mm, win, rows


@pytest.fixture(scope="session")
def synth_dump(tmp_path_factory):
    """Observation rows from a 4-seat game with nations and bots, 3 points in time."""
    d = tmp_path_factory.mktemp("dump")
    rec = d / "rich.json.gz"
    run_sim_src("synth", str(rec), "--map", SMALL_MAP, "--seats", "4", "--turns", "1500", "--seed", "planes",
                "--nations", "default", "--bots", "30", "--engine-root", str(paths.ENGINE))
    return dump_obs(rec, d / "out", steps=3, every=90)


def tiny_config(**kw):
    """A small model for fast tests (same structure as the real one)."""
    from ofai.model.config import ModelConfig

    base = dict(width=64, lstm=64, fusion_layers=1, player_layers=1, heads=4, cnn_channels=(16, 16, 16),
                win_channels=(16, 16), ff_mult=2, ptr_dim=16, fine_dim=4, blocks_per_stage=1, amp=False)
    base.update(kw)
    return ModelConfig(**base)
