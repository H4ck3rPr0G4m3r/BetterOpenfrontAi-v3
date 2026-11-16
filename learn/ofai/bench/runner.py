"""Runs bench suites with a checkpoint (M3 Task 8).

Each suite runs its (map, seed) games on an env-role SimPool. Seat 0 of every
game is the checkpoint under test; human_proxy fills seats 1..20 with the
frozen proxy at temperature 1.0. Every AI name carries the `[AI] ` tag (the
env turns it into clan tag AI). A game that hits a sim error is recorded with
its seed and left out of the rates; the suite still completes.
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ofai import checkpoints as ck
from ofai import paths
from ofai.bench.agent import PolicyAgent
from ofai.bench.mind import MindWriter
from ofai.bench.suites import load_bench
from ofai.bridge import protocol as P
from ofai.bridge.pool import SimPool

TICKS_PER_MIN = 600


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()


def game_spec(entry: dict, seed: str, suite: dict, record: str | bool, minutes: float | None = None) -> dict:
    seats = suite["seats"]
    names = ["[AI] ofai"] + [f"[AI] proxy{i}" for i in range(1, seats)]
    return {"map": entry["map"], "seed": seed, "seats": seats, "names": names, "bots": entry["bots"],
            "nations": entry["nations"], "difficulty": suite["difficulty"],
            "tickCap": int((minutes or suite["minutes"]) * TICKS_PER_MIN), "latency": [1, 3], "record": record}


class _Game:
    def __init__(self, spec: dict, suite: str, mind: MindWriter | None) -> None:
        self.spec = spec
        self.suite = suite
        self.mind = mind
        self.curve: list[float] = []  # AI land share at each whole minute
        self.result: dict | None = None
        self.fresh = True


def run_suite(name: str, suite: dict, games: list[tuple[dict, str]], agent: PolicyAgent, proxy: PolicyAgent | None,
              pool: SimPool, out_dir: Path, replays: bool, minutes: float | None = None) -> list[dict]:
    lay = agent.layout
    seats, per_worker = pool.seats, pool.games
    n_slots = per_worker * len(pool.workers)
    queue = list(games)
    slots: list[_Game | None] = [None] * n_slots
    results: list[dict] = []

    def start(i: int) -> dict | None:
        if not queue:
            slots[i] = None
            return None
        entry, seed = queue.pop(0)
        stem = f"{name}-{_slug(entry['map'])}-{_slug(seed.split(':')[-1])}"
        rec = str(out_dir / "replays" / f"{stem}.json.gz") if replays else False
        mind = MindWriter(out_dir / "replays" / f"{stem}.mind.jsonl.gz") if replays else None
        spec = game_spec(entry, seed, suite, rec, minutes)
        slots[i] = _Game(spec, name, mind)
        return spec

    batch = pool.reset([start(i) for i in range(n_slots)])
    reset_errors = {}
    for w, info in enumerate(batch["info"]):
        for g, err in (info.get("resetErrors") or {}).items():
            reset_errors[w * per_worker + int(g)] = err
    for gi, err in reset_errors.items():
        _finish(slots[gi], results, terminal="sim_error", error=err)
        slots[gi] = None

    while any(s is not None for s in slots):
        flags = batch["flags"]
        acts = np.full((pool.rows, lay.K, lay.A), -1, dtype=np.int32)
        acts[:, 0, lay.af("type")] = lay.at("STOP")
        ai_rows = np.array([gi * seats for gi, s in enumerate(slots) if s is not None and flags[gi * seats] & P.FLAG_NEXT], dtype=np.int64)
        if ai_rows.size:
            fresh = np.array([slots[r // seats].fresh for r in ai_rows])
            a, mind = agent.act(batch, ai_rows, fresh)
            acts[ai_rows] = a
            for j, r in enumerate(ai_rows):
                g = slots[r // seats]
                g.fresh = False
                if g.mind is not None:
                    g.mind.write_step(lay, int(batch["game_meta"][r][list(lay.raw["gameMetaFields"]).index("ticks")]),
                                      "SEAT0000", float(mind["value"][j]), a[j], mind["top"][j], batch["game_meta"][r],
                                      batch["win_meta"][r])
        if proxy is not None and seats > 1:
            prow = np.array([r for r in range(pool.rows) if r % seats and flags[r] & P.FLAG_NEXT and slots[r // seats] is not None],
                            dtype=np.int64)
            if prow.size:
                pa, _ = proxy.act(batch, prow, (flags[prow] & P.FLAG_FRESH) != 0)
                acts[prow] = pa
        batch = pool.step(acts)
        land = [x for info in batch["info"] for x in info.get("land", [])]
        over = [x for info in batch["info"] for x in info.get("over", [])]
        errors = {w * per_worker + e["game"]: e["error"] for w, info in enumerate(batch["info"]) for e in info.get("simErrors", [])}
        to_reset: list[dict | None] = [None] * n_slots
        need_reset = False
        for gi, g in enumerate(slots):
            if g is None:
                continue
            r = gi * seats
            st = batch["stats"][r]
            ticks = float(st[P.STATS.index("ticks")])
            if land and land[gi] > 0:
                while len(g.curve) < int(ticks // TICKS_PER_MIN):
                    g.curve.append(float(st[P.STATS.index("tiles")]) / land[gi])
            if batch["flags"][r] & P.FLAG_DONE and g.result is None:
                g.result = {"terminal": P.TERMINALS[int(batch["terminal"][r])], "ticks": ticks,
                            "placement": int(st[P.STATS.index("placement")]), "kills": int(st[P.STATS.index("kills")])}
            if over and over[gi]:
                _finish(g, results, error=errors.get(gi))
                spec = start(gi)
                to_reset[gi] = spec
                need_reset = need_reset or spec is not None
        if need_reset:
            fresh_batch = pool.reset(to_reset)
            for gi, spec in enumerate(to_reset):
                if spec is None:
                    continue
                rows = slice(gi * seats, (gi + 1) * seats)
                for k, v in fresh_batch.items():
                    if k != "info":
                        batch[k][rows] = v[rows]
                err = next((info.get("resetErrors", {}).get(str(gi % per_worker)) for w, info in enumerate(fresh_batch["info"])
                            if w == gi // per_worker), None)
                if err:
                    _finish(slots[gi], results, terminal="sim_error", error=err)
                    slots[gi] = None
    return results


def _finish(g: _Game | None, results: list[dict], terminal: str | None = None, error: str | None = None) -> None:
    if g is None:
        return
    if g.mind is not None:
        if g.result is None and (terminal == "sim_error" or error):
            g.mind.discard()
        else:
            g.mind.close()
    res = dict(g.result or {"terminal": terminal or "none", "ticks": 0, "placement": 0, "kills": 0})
    if terminal:
        res["terminal"] = terminal
    if error:
        res["terminal"] = "sim_error"
        res["error"] = error[:2000]
    res.update(map=g.spec["map"], seed=g.spec["seed"], suite=g.suite, landCurve=g.curve,
               record=g.spec["record"] or None)
    results.append(res)


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r["terminal"] != "sim_error"]
    out: dict[str, Any] = {"games": len(results), "simErrors": [{"map": r["map"], "seed": r["seed"], "error": r.get("error")}
                                                                for r in results if r["terminal"] == "sim_error"]}
    if not ok:
        return out
    minutes = lambda r: r["ticks"] / TICKS_PER_MIN  # noqa: E731
    alive_at = lambda r, m: r["terminal"] != "death" or minutes(r) >= m  # noqa: E731
    out["survived20"] = sum(alive_at(r, 20) for r in ok) / len(ok)
    out["survived30"] = sum(alive_at(r, 30) for r in ok) / len(ok)
    out["wins"] = sum(r["terminal"] == "win" for r in ok) / len(ok)
    placed = [r["placement"] for r in ok if r["placement"] > 0]
    if placed:
        out["placementMedian"] = statistics.median(placed)
        out["top3"] = sum(p <= 3 for p in placed) / len(ok)
    curves = [r["landCurve"] for r in ok if r["landCurve"]]
    if curves:
        n = max(len(c) for c in curves)
        out["landCurve"] = [statistics.median([c[i] for c in curves if len(c) > i]) for i in range(n)]
    return out


def land_curve_gap(curve: list[float], teacher: list[float], minutes: int = 20) -> float | None:
    pairs = [(a, b) for a, b in zip(curve[:minutes], teacher[:minutes])]
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None


def run_bench(checkpoint: Path, suites: list[str] | None = None, *, workers: int = 4, games_per_worker: int = 2,
              replays: bool = True, bench: dict | None = None, out_dir: Path | None = None, minutes: float | None = None,
              teacher_curves: dict[str, list[float]] | None = None, pool_kwargs: dict | None = None,
              device: str | None = None) -> dict:
    bench = bench or load_bench()
    suites = suites or ["solo_medium", "solo_hard", "human_proxy"]
    out_dir = out_dir or paths.RUNS / f"{datetime.now():%Y%m%d-%H%M}-bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = PolicyAgent(checkpoint, temperature=1.0, device=device, seed=1)
    proxy_ckpt = (bench.get("proxy") or {}).get("path")
    results: dict[str, Any] = {"bench": bench["bench"], "checkpoint": str(checkpoint), "commit": bench["commit"], "suites": {}}
    all_games: dict[str, list[dict]] = {}
    for name in suites:
        suite = bench["suites"][name]
        proxy = None
        if suite["proxy_seats"]:
            proxy = PolicyAgent(Path(proxy_ckpt) if proxy_ckpt else checkpoint, temperature=1.0, device=device, seed=2)
        games = [(m, s) for m in bench["maps"] for s in m["seeds"]]
        pool = SimPool(bench["commit"], "env", min(workers, max(1, len(games))), games=games_per_worker,
                       seats=suite["seats"], **(pool_kwargs or {}))
        try:
            rs = run_suite(name, suite, games, agent, proxy, pool, out_dir, replays, minutes)
        finally:
            pool.close()
        agg = aggregate(rs)
        if teacher_curves and name == "solo_hard":
            gaps = [land_curve_gap(r["landCurve"], teacher_curves.get(r["map"], [])) for r in rs if r["terminal"] != "sim_error"]
            gaps = [g for g in gaps if g is not None]
            if gaps:
                agg["landCurveGap"] = sum(gaps) / len(gaps)
        results["suites"][name] = agg
        all_games[name] = rs
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    (out_dir / "games.json").write_text(json.dumps(all_games, indent=1))
    try:
        ck.register(Path(checkpoint).stem, Path(checkpoint), "bc", 0, {"bench": {bench["bench"]: results["suites"]}})
    except OSError:
        pass
    return results


def teacher_curves(index: Any, maps: list[str], per_map: int = 10) -> dict[str, list[float]]:
    """Per-map median teacher land share per minute, from in-window replay-ok records (sim land-curves)."""
    from ofai.engines import run_sim

    out: dict[str, list[float]] = {}
    for m in maps:
        curves: list[list[float]] = []
        for g in index.games(map=m, only_in_window=True, replay_ok=True, limit=per_map):
            clients = [s["client_id"] for s in index.teacher_seats(g["id"])]
            if not clients:
                continue
            for line in run_sim(g["commit"], "land-curves", g["path"], "--clients", ",".join(clients)):
                if line.get("error") is None and line.get("mismatchAt") is None:
                    curves.extend(c for c in line["curves"].values() if c)
        if curves:
            n = max(len(c) for c in curves)
            out[m] = [statistics.median([c[i] for c in curves if len(c) > i]) for i in range(n)]
    return out
