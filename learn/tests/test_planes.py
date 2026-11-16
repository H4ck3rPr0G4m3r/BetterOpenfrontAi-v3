import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SMALL_MAP, run_sim_src
from ofai import paths
from ofai.layout import Layout
from ofai.model.planes import expand_minimap, expand_windows, to_tensors


def _layout() -> Layout:
    return Layout.from_json(run_sim_src("layout")[0])


def _dump(record: Path, out: Path, steps: int, every: int) -> tuple[dict, np.ndarray, np.ndarray, int]:
    run_sim_src("dump-obs", str(record), "--out", str(out), "--steps", str(steps), "--every", str(every),
                "--engine-root", str(paths.ENGINE))
    header = json.loads((out / "header.json").read_text())
    rows = header["rows"]
    lay = _layout()
    raw = (out / "obs.bin").read_bytes()
    obs, end = lay.parse_obs(raw, 0, rows)
    assert end == len(raw)
    P = len(lay.planes)
    MM, WIN, NW = lay.sizes["MM"], lay.sizes["WIN"], lay.sizes["NUM_WINDOWS"]
    mm = np.frombuffer((out / "mm_ref.bin").read_bytes(), dtype="<f4").reshape(rows, P, MM, MM)
    win = np.frombuffer((out / "win_ref.bin").read_bytes(), dtype="<f4").reshape(rows, NW, P, WIN, WIN)
    return obs, mm, win, rows


def _check(obs: dict, mm: np.ndarray, win: np.ndarray, device: str) -> None:
    lay = _layout()
    t = to_tensors(obs, device)
    got_mm = expand_minimap(t, lay).cpu().numpy()
    got_win = expand_windows(t, lay).cpu().numpy()
    for p, plane in enumerate(lay.planes):
        assert np.array_equal(got_mm[:, p], mm[:, p]), f"minimap plane {plane['name']}"
        assert np.array_equal(got_win[:, :, p], win[:, :, p]), f"window plane {plane['name']}"


@pytest.fixture(scope="session")
def synth_dump(tmp_path_factory):
    d = tmp_path_factory.mktemp("dump")
    rec = d / "rich.json.gz"
    run_sim_src("synth", str(rec), "--map", SMALL_MAP, "--seats", "4", "--turns", "1500", "--seed", "planes",
                "--nations", "default", "--bots", "30", "--engine-root", str(paths.ENGINE))
    return _dump(rec, d / "out", steps=3, every=90)


def test_matches_ts_reference(synth_dump):
    obs, mm, win, rows = synth_dump
    assert rows >= 6
    assert mm[:, [i for i, p in enumerate(_layout().planes) if p["name"] == "own"][0]].sum() > 0
    _check(obs, mm, win, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_matches_ts_reference_on_cuda(synth_dump):
    obs, mm, win, _ = synth_dump
    _check(obs, mm, win, "cuda")


def _pin_records() -> list[Path]:
    pin = subprocess.run(["git", "-C", str(paths.ENGINE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return sorted((paths.RECORDS / pin).glob("*.json.gz")) if pin else []


@pytest.mark.skipif(not _pin_records(), reason="no real records for the pinned engine in data/records")
def test_matches_ts_reference_on_a_real_game(tmp_path):
    # A real lobby has units of every kind (SAMs, nukes, warships) by mid-game.
    obs, mm, win, rows = _dump(_pin_records()[0], tmp_path, steps=3, every=400)
    assert rows > 0
    _check(obs, mm, win, "cuda" if torch.cuda.is_available() else "cpu")
