"""bench-v1 definition and freezing (M3 Task 8). Never edited after freezing;
a change means bench-v2.

Maps: World, the smallest (Onion), the largest (Sol), a portrait map (Las
Vegas Strip), and the 8 maps seen most often in in-window public FFA games
of the index (the live rotation weights come from the game's API, not the
engine source, so the observed rotation stands in for them). 5 fixed seeds
per map; bots and nations per map are the medians of those games' configs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ofai import paths
from ofai.data.index import Index

BENCH_NAME = "bench-v1"
FIXED_MAPS = ["World", "Onion", "Sol", "Las Vegas Strip"]
FALLBACK_MAPS = ["Europe", "Africa", "Mare Nostrum", "Balkans", "Scandinavia", "Russia", "Great Lakes", "Horn Of Africa",
                 "North America", "Asia", "South America", "Britannia"]
SEEDS_PER_MAP = 5
SUITES = {
    # metric: survived 20 minutes
    "solo_medium": {"seats": 1, "proxy_seats": 0, "difficulty": "Medium", "minutes": 30},
    # metrics: placement, top-3, survival at 20 / 30 min, land-share curve
    "solo_hard": {"seats": 1, "proxy_seats": 0, "difficulty": "Hard", "minutes": 30},
    # 1 AI seat + 20 seats of the frozen proxy policy at temperature 1.0; metric: placement
    "human_proxy": {"seats": 21, "proxy_seats": 20, "difficulty": "Medium", "minutes": 30},
}
LATENCY = [1, 3]


def bench_path() -> Path:
    return paths.DATA / "bench" / f"{BENCH_NAME}.json"


def proxy_path() -> Path:
    return paths.CHECKPOINTS / f"{BENCH_NAME}-proxy.pt"


def pinned_commit() -> str:
    return subprocess.run(["git", "-C", str(paths.ENGINE), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _configs_by_map(index: Index) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for g in index.games(only_in_window=True, limit=10_000_000):
        try:
            with gzip.open(g["path"], "rt", encoding="utf-8") as f:
                cfg = json.load(f)["info"]["config"]
        except (OSError, ValueError, KeyError):
            continue
        out.setdefault(g["map"], []).append(cfg)
    return out


def choose_maps(index: Index) -> list[str]:
    counts = Counter(g["map"] for g in index.games(only_in_window=True, limit=10_000_000))
    extra = [m for m, _ in counts.most_common() if m not in FIXED_MAPS][:8]
    for m in FALLBACK_MAPS:
        if len(extra) >= 8:
            break
        if m not in extra and m not in FIXED_MAPS:
            extra.append(m)
    return FIXED_MAPS + extra


def bench_definition(index: Index, commit: str) -> dict:
    configs = _configs_by_map(index)
    maps = []
    for m in choose_maps(index):
        cfgs = configs.get(m, [])
        bots = [c["bots"] for c in cfgs if isinstance(c.get("bots"), (int, float))]
        nations = Counter(str(c.get("nations", "default")) for c in cfgs)
        nat = nations.most_common(1)[0][0] if nations else "default"
        maps.append({
            "map": m,
            "seeds": [f"{BENCH_NAME}:{m}:{i}" for i in range(SEEDS_PER_MAP)],
            "bots": int(statistics.median(bots)) if bots else 400,
            "nations": int(nat) if nat.isdigit() else nat,
            "games": len(cfgs),
        })
    return {"bench": BENCH_NAME, "commit": commit, "maps": maps, "suites": SUITES, "latency": LATENCY}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_bench_v1(index: Index, proxy_checkpoint: Path, *, commit: str | None = None) -> Path:
    out = bench_path()
    if out.exists():
        raise FileExistsError(f"{out} exists: {BENCH_NAME} is frozen (a change means bench-v2)")
    from ofai.checkpoints import verify

    if not verify(Path(proxy_checkpoint)):
        raise ValueError(f"proxy checkpoint {proxy_checkpoint} does not verify")
    d = bench_definition(index, commit or pinned_commit())
    proxy = proxy_path()
    proxy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(proxy_checkpoint, proxy)
    (proxy.with_name(proxy.name + ".sha256")).write_text(_sha(proxy))
    d["proxy"] = {"path": str(proxy), "sha256": _sha(proxy), "source": str(proxy_checkpoint)}
    d["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(out)
    return out


def load_bench() -> dict:
    return json.loads(bench_path().read_text())


def bench_file_sha() -> str:
    return _sha(bench_path())
